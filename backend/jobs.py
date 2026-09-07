"""Single-worker job queue.

The GPU can only usefully run one generation at a time, so every request is
funnelled through one worker thread. Progress hops back to the event loop via
`call_soon_threadsafe` and leaves as SSE.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import storage
from .backends import Cancelled, GenerationBackend
from .config import settings
from .grid import Cell, run_id as _run_id
from .schemas import GenerateRequest
from .seeds import random_seed, wrap_seed

log = logging.getLogger(__name__)

_TERMINAL = {"done", "error", "cancelled"}


def short_error(exc: BaseException, limit: int = 320) -> str:
    """A one-paragraph, UI-presentable form of an exception.

    HuggingFace errors in particular carry a request id and a stack of URLs that
    are noise in the browser; keep the actionable sentences and drop the rest.
    """
    lines = [
        line.strip()
        for line in str(exc).splitlines()
        if line.strip() and "Request ID" not in line
    ]
    message = " ".join(lines) or exc.__class__.__name__
    if len(message) > limit:
        message = message[: limit - 1].rstrip() + "…"
    return f"{type(exc).__name__}: {message}"


@dataclass
class Job:
    id: str
    # The first cell's settings. What the job actually runs is `cells`: a plain
    # request expands to `runs` of them, a grid to its cartesian product, and
    # nothing below this line needs to know which it was handed.
    request: GenerateRequest
    cells: list[Cell] = field(default_factory=list)
    # The base seed of the first cell - the one the client has to be given back,
    # because every image it will see carries a derived one - and the run the
    # cells belong to, which a resume shares with the job it continues.
    seed: int | None = None
    run_id: str | None = None
    status: str = "queued"  # queued | running | done | error | cancelled
    step: int = 0
    total_steps: int = 0
    run: int = 0
    total_runs: int = 1
    # What the current cell varies, for a grid; None for an ordinary sweep.
    label: str | None = None
    images: list[dict] = field(default_factory=list)
    error: str | None = None
    # One queue per attached stream. A single shared queue is destructive: a
    # reconnecting client that overlaps the stream it replaces splits the events
    # between the two readers, and only one of them ever sees `done`.
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    # Set from the event loop, read from the worker thread between steps.
    cancel_requested: bool = False
    done_sent: bool = False

    def publish(self, event: dict) -> None:
        """Hand an event to every attached stream. Event-loop thread only."""
        for queue in self.subscribers:
            queue.put_nowait(event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "run_id": self.run_id,
            "seed": self.seed,
            "status": self.status,
            "step": self.step,
            "total_steps": self.total_steps,
            "run": self.run,
            "total_runs": self.total_runs,
            "label": self.label,
            "images": self.images,
            "error": self.error,
        }


class JobManager:
    def __init__(
        self,
        backend: GenerationBackend,
        ensure_loaded: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.backend = backend
        self._ensure_loaded = ensure_loaded
        self._jobs: dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
        self._gate = asyncio.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        """Jobs that have not reached a terminal state, newest last."""
        return [job for job in self._jobs.values() if job.status not in _TERMINAL]

    def busy(self) -> bool:
        """True while anything is queued or running.

        Wider than `self._gate.locked()` on purpose: a job still waiting its turn
        starts on its own, so acting in front of it is the same mistake one step
        later.
        """
        return bool(self.active())

    async def shutdown(self) -> None:
        """Stop every live job and wait for the worker thread to drain.

        uvicorn does not wait for a generation thread, so without this a Ctrl-C
        mid-sweep either hangs on an open SSE stream or has the weights pulled
        out from under a running denoise. The cancel is the same cooperative one
        the button uses, so it lands within a step and keeps what it finished.
        """
        for job in self.active():
            job.cancel_requested = True
        await asyncio.to_thread(self._pool.shutdown, True)

    async def submit(self, request: GenerateRequest) -> Job:
        """An ordinary request: `runs` passes of one setting, seed stepped.

        Each pass steps the seed past the whole of the previous batch, so every
        image in a sweep keeps a seed that reproduces it on its own. Wrapped
        rather than allowed to overflow the signed 64-bit seed column.
        """
        base = request.seed if request.seed is not None else random_seed()
        return await self.submit_cells(
            [
                Cell(request=request, seed=wrap_seed(base + index * request.num_images))
                for index in range(request.runs)
            ]
        )

    async def submit_cells(self, cells: list[Cell], run_id: str | None = None) -> Job:
        """Queue an already-expanded list of pipeline calls - a grid, usually.

        `run_id` names the run the cells belong to. A resumed grid hands in the
        id of the *whole* grid, computed before the finished cells were
        subtracted, so its images join the night they continue; left out, the
        cells themselves are the definition.
        """
        job = Job(
            id=uuid.uuid4().hex[:12],
            request=cells[0].request,
            cells=cells,
            seed=cells[0].seed,
            run_id=run_id or _run_id(cells, settings.model_id, settings.quantization),
            total_steps=cells[0].request.steps,
            total_runs=len(cells),
        )
        self._jobs[job.id] = job
        # Keep the dict from growing without bound over a long session.
        if len(self._jobs) > 200:
            for stale in list(self._jobs)[:-100]:
                if self._jobs[stale].status in _TERMINAL:
                    self._jobs.pop(stale, None)
        asyncio.create_task(self._run(job))
        return job

    def cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop. Returns None only for an unknown id.

        A running job unwinds at its next step boundary and keeps the images it
        has already written. A job still waiting its turn is finished off right
        here, so cancelling it does not first require the job ahead of it to end.
        """
        job = self._jobs.get(job_id)
        if job is None or job.status in _TERMINAL:
            return job
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            self._finish(asyncio.get_running_loop(), job)
        return job

    def _emit(self, loop: asyncio.AbstractEventLoop, job: Job, event: dict) -> None:
        loop.call_soon_threadsafe(job.publish, event)

    def _finish(self, loop: asyncio.AbstractEventLoop, job: Job) -> None:
        """Emit the terminal event exactly once.

        Cancellation gives two paths to the end - the endpoint for a queued job,
        `_run` for everything else - and the SSE stream stops at the first done.
        """
        if job.done_sent:
            return
        job.done_sent = True
        self._emit(loop, job, {"type": "done", **job.snapshot()})

    async def _run(self, job: Job) -> None:
        loop = asyncio.get_running_loop()
        # Wait for the shared load rather than letting the worker thread start a
        # second one. Failing here keeps the error to a single report.
        if self._ensure_loaded is not None:
            try:
                await self._ensure_loaded()
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                job.status = "error"
                job.error = short_error(exc)
                self._finish(loop, job)
                return
        # One generation at a time; queued jobs simply wait their turn here. A
        # cancel that landed during the load or the wait ends the job untouched.
        async with self._gate:
            if job.cancel_requested:
                job.status = "cancelled"
                self._finish(loop, job)
                return
            job.status = "running"
            self._emit(loop, job, {"type": "status", **job.snapshot()})
            try:
                await loop.run_in_executor(self._pool, self._generate, job, loop)
                job.status = "cancelled" if job.cancel_requested else "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                log.error("job %s failed: %s", job.id, short_error(exc))
                log.debug("job traceback", exc_info=True)
                job.status = "error"
                job.error = short_error(exc)
            self._finish(loop, job)

    def _generate(self, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        """Run the job's cells back to back, publishing images as they land.

        The loop cannot tell a sweep from a grid, which is the point: the
        cancellation checks, the progress events and the per-image seed maths
        are written once and both paths get them.
        """

        def should_cancel() -> bool:
            return job.cancel_requested

        def on_progress(step: int, total: int) -> None:
            job.step, job.total_steps = step, total
            self._emit(
                loop,
                job,
                {
                    "type": "progress",
                    "step": step,
                    "total_steps": total,
                    "run": job.run,
                    "total_runs": job.total_runs,
                    "label": job.label,
                },
            )

        for index, cell in enumerate(job.cells):
            if job.cancel_requested:
                break
            req, seed = cell.request, cell.seed
            # A grid varies the step count between cells, so the totals the
            # progress events carry are the cell's, not the job's.
            job.run, job.step, job.total_steps, job.label = index + 1, 0, req.steps, cell.label
            self._emit(loop, job, {"type": "status", **job.snapshot()})

            started = time.perf_counter()
            try:
                images = self.backend.generate(
                    req, seed=seed, on_progress=on_progress, should_cancel=should_cancel
                )
            except Cancelled:
                log.info(
                    "job %s cancelled during run %d/%d", job.id, index + 1, len(job.cells)
                )
                break
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            # Record what ran, not what was asked: a negative prompt the
            # pipeline dropped below its threshold, or a mu it had no way to
            # take, must not reach the row - resume compares against it.
            ran = self.backend.applied(req)

            for offset, image in enumerate(images):
                # Each image in a batch gets its own reproducible seed. Wrapped
                # for the same reason the run seed is - and with the same helper
                # the generator uses, so the two cannot disagree.
                record = storage.save_image(
                    image,
                    ran,
                    seed=wrap_seed(seed + offset),
                    duration_ms=elapsed_ms // max(len(images), 1),
                    model_id=settings.model_id,
                    job_id=job.id,
                    run_id=job.run_id,
                    quantization=settings.quantization,
                )
                job.images.append(record.model_dump())
                # Publish it now: a long sweep should fill the page as it goes
                # rather than all at once at the end.
                self._emit(
                    loop,
                    job,
                    {
                        "type": "image",
                        "image": record.model_dump(),
                        "run": job.run,
                        "total_runs": job.total_runs,
                    },
                )

    async def events(self, job: Job):
        """Yield SSE payloads until the job finishes.

        Subscribing before the snapshot rather than after it: an event emitted
        in between would otherwise fall down the gap. A duplicate is harmless -
        the snapshot carries the accumulated images and the client keys them by
        id - but a missed one is not.
        """
        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)
        try:
            # A stream opened on a job that has already ended - a reload, or one
            # the cancel endpoint finished off while it was still queued - gets
            # the terminal event rather than a status snapshot it would then
            # wait on for good.
            done = job.status in _TERMINAL
            yield {"type": "done" if done else "status", **job.snapshot()}
            if done:
                return
            while True:
                event = await queue.get()
                yield event
                if event.get("type") == "done":
                    return
        finally:
            job.subscribers.remove(queue)
