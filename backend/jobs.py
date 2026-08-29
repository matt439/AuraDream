"""Single-worker job queue.

The GPU can only usefully run one generation at a time, so every request is
funnelled through one worker thread. Progress hops back to the event loop via
`call_soon_threadsafe` and leaves as SSE.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import storage
from .backends import Cancelled, GenerationBackend
from .config import settings
from .schemas import GenerateRequest

log = logging.getLogger(__name__)

_MAX_SEED = 2**63 - 1

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
    request: GenerateRequest
    status: str = "queued"  # queued | running | done | error | cancelled
    step: int = 0
    total_steps: int = 0
    run: int = 0
    total_runs: int = 1
    images: list[dict] = field(default_factory=list)
    error: str | None = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Set from the event loop, read from the worker thread between steps.
    cancel_requested: bool = False
    done_sent: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "step": self.step,
            "total_steps": self.total_steps,
            "run": self.run,
            "total_runs": self.total_runs,
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

    async def submit(self, request: GenerateRequest) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            request=request,
            total_steps=request.steps,
            total_runs=request.runs,
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
        loop.call_soon_threadsafe(job.queue.put_nowait, event)

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
        """Run `request.runs` passes back to back, publishing images as they land."""
        req = job.request
        base = req.seed if req.seed is not None else random.randint(0, _MAX_SEED)

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
                },
            )

        for index in range(req.runs):
            if job.cancel_requested:
                break
            # Each pass steps the seed past the whole of the previous batch, so
            # every image in a sweep keeps a seed that reproduces it on its own.
            # Wrap rather than overflow the signed 64-bit seed column.
            seed = (base + index * req.num_images) % (_MAX_SEED + 1)
            job.run, job.step = index + 1, 0
            self._emit(loop, job, {"type": "status", **job.snapshot()})

            started = time.perf_counter()
            try:
                images = self.backend.generate(
                    req, seed=seed, on_progress=on_progress, should_cancel=should_cancel
                )
            except Cancelled:
                log.info("job %s cancelled during run %d/%d", job.id, index + 1, req.runs)
                break
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            for offset, image in enumerate(images):
                # Each image in a batch gets its own reproducible seed.
                record = storage.save_image(
                    image,
                    req,
                    seed=seed + offset,
                    duration_ms=elapsed_ms // max(len(images), 1),
                    model_id=settings.model_id,
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
        """Yield SSE payloads until the job finishes."""
        # A stream opened on a job that has already ended - a reload, or one the
        # cancel endpoint finished off while it was still queued - gets the
        # terminal event rather than a status snapshot it would then wait on.
        done = job.status in _TERMINAL
        yield {"type": "done" if done else "status", **job.snapshot()}
        if done:
            return
        while True:
            event = await job.queue.get()
            yield event
            if event.get("type") == "done":
                return
