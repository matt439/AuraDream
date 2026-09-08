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
from .latent_preview import data_url
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
        # Cut from the middle: PyTorch's OOM message puts the diagnosis first
        # and the remedy (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
        # last, and a cut from the end landed inside that variable's name.
        head = limit * 2 // 3
        tail = limit - head - 3
        message = message[:head].rstrip() + " … " + message[-tail:].lstrip()
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
    # What a queued job is waiting on: "load" while the weights are on their
    # way, "queue" behind another job. Not a status of its own - cancel()
    # keys its immediate-finish path on `queued`, and a fourth state would have
    # turned Cancel-during-load from instant into a multi-minute no-op - but
    # the browser used to read "waiting for the job ahead…" for the whole of a
    # 32 GB download, under a top bar saying "loading weights…".
    waiting_on: str | None = None
    step: int = 0
    total_steps: int = 0
    run: int = 0
    total_runs: int = 1
    # What the current cell varies, for a grid; None for an ordinary sweep.
    label: str | None = None
    images: list[dict] = field(default_factory=list)
    error: str | None = None
    # Cells that raised and were skipped. A grid survives one bad cell - a
    # transient OOM, a locked database - and reports it, rather than dropping
    # every remaining cell of the night on the first.
    failures: int = 0
    # One queue per attached stream. A single shared queue is destructive: a
    # reconnecting client that overlaps the stream it replaces splits the events
    # between the two readers, and only one of them ever sees `done`.
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    # Set from the event loop, read from the worker thread between steps.
    cancel_requested: bool = False
    # Pause is cooperative like cancel: the endpoint asks, the worker answers
    # at the next cell boundary by setting `paused` - a worker-written
    # acknowledgement, so the UI cannot believe the card is free while a
    # denoise is still running on it.
    pause_requested: bool = False
    paused: bool = False
    done_sent: bool = False
    # Unix seconds as floats - JSON-native, since snapshot() goes straight
    # through json.dumps on the SSE path. `last_progress_at` is what tells a
    # sweep that is moving from one wedged behind a hung load.
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    last_progress_at: float | None = None
    # Start no cell after this moment (a grid's time budget), and what ended
    # the job when it was not the user: "budget" so far. Ending at the deadline
    # reuses the cancel path - a fourth terminal state would have left
    # `active()` and `busy()` holding on and unload answering 409 for the
    # session - and the field tells the UI which kind of stop it was.
    deadline: float | None = None
    stopped_by: str | None = None

    def publish(self, event: dict) -> None:
        """Hand an event to every attached stream. Event-loop thread only."""
        for queue in self.subscribers:
            queue.put_nowait(event)

    def snapshot(self, images: bool = True) -> dict[str, Any]:
        """The job as one JSON frame.

        `images=False` leaves the accumulated records out. The per-cell status
        event used to carry every record produced so far - cell N re-sending
        N-1 records already delivered as their own `image` events, 258 MiB over
        a 1000-cell grid - where only the attach snapshot and `done` need them.
        A copy, not the live list: what json.dumps serialises later on the loop
        must be the list as of now.
        """
        frame = {
            "job_id": self.id,
            "run_id": self.run_id,
            "seed": self.seed,
            "status": self.status,
            "waiting_on": self.waiting_on,
            "pause_requested": self.pause_requested,
            "paused": self.paused,
            "step": self.step,
            "total_steps": self.total_steps,
            "run": self.run,
            "total_runs": self.total_runs,
            "label": self.label,
            "error": self.error,
            "failures": self.failures,
            # The count travels even when the records do not: a listing of
            # finished jobs has to say how much a night produced without
            # carrying the night.
            "image_count": len(self.images),
            # How many different pictures those are. A grid whose axis the
            # pipeline ignored - a negative-prompt variable on klein - renders
            # the same image N times, and this is the number that says so.
            "distinct_count": len({image.get("content_hash") for image in self.images} - {None}),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_progress_at": self.last_progress_at,
            "deadline": self.deadline,
            "stopped_by": self.stopped_by,
        }
        if images:
            frame["images"] = list(self.images)
        return frame


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

    def finished(self) -> list[Job]:
        """Jobs that have ended, the most recently finished first.

        In memory only, so it answers "what happened to last night's run" for
        as long as this server process is up - the gallery's rows answer "how
        many and when" for good, but never "why".
        """
        done = [job for job in self._jobs.values() if job.status in _TERMINAL]
        return sorted(done, key=lambda job: job.finished_at or 0.0, reverse=True)

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

    async def submit_cells(
        self, cells: list[Cell], run_id: str | None = None, budget_minutes: float | None = None
    ) -> Job:
        """Queue an already-expanded list of pipeline calls - a grid, usually.

        `run_id` names the run the cells belong to. A resumed grid hands in the
        id of the *whole* grid, computed before the finished cells were
        subtracted, so its images join the night they continue; left out, the
        cells themselves are the definition. `budget_minutes` is plumbed here
        rather than read off `job.request`: GridRequest-only fields are gone
        by the time the cells are built.
        """
        job = Job(
            id=uuid.uuid4().hex[:12],
            request=cells[0].request,
            cells=cells,
            seed=cells[0].seed,
            run_id=run_id or _run_id(cells, settings.model_id, settings.quantization),
            total_steps=cells[0].request.steps,
            total_runs=len(cells),
            deadline=time.time() + budget_minutes * 60 if budget_minutes else None,
        )
        self._jobs[job.id] = job
        # A line at the start, one per cell and one at the end: an overnight
        # grid started from the browser used to leave nothing in the log unless
        # it broke, where the CLI writes one line per cell when redirected.
        log.info(
            "job %s queued: %d cell(s), run %s, seed %s, %dx%d",
            job.id, len(cells), job.run_id, job.seed,
            cells[0].request.width, cells[0].request.height,
        )
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
            self._finish(asyncio.get_running_loop(), job, "cancelled")
        return job

    def pause(self, job_id: str, paused: bool) -> Job | None:
        """Ask a job to hold at its next cell boundary, or to carry on.

        The hold lands between cells, where `backend.generate()` has returned
        and the card holds little but allocator reserve, which the worker then
        hands back. A running job is the only kind this means anything for; a
        queued one holds its place either way, and a finished one is left alone.
        """
        job = self._jobs.get(job_id)
        if job is None or job.status in _TERMINAL:
            return job
        job.pause_requested = paused
        return job

    def _emit(self, loop: asyncio.AbstractEventLoop, job: Job, event: dict) -> None:
        loop.call_soon_threadsafe(job.publish, event)

    def _finish(
        self,
        loop: asyncio.AbstractEventLoop,
        job: Job,
        status: str,
        error: str | None = None,
    ) -> None:
        """Move the job to its terminal state and emit the event, exactly once.

        Cancellation gives two paths to the end - the endpoint for a queued job,
        `_run` for everything else - and the SSE stream stops at the first done.
        The *state* is written here too, behind the same guard: a job the
        endpoint had already cancelled used to be rewritten to `error` when the
        load it was still parked on then failed, with the event suppressed and
        the REST snapshot contradicting the stream.
        """
        if job.done_sent:
            return
        job.status = status
        if error is not None:
            job.error = error
        job.done_sent = True
        job.finished_at = time.time()
        since = job.started_at or job.created_at
        log.info(
            "job %s %s: %d image(s) in %.0fs%s",
            job.id, status, len(job.images), job.finished_at - since,
            f", {job.failures} cell(s) failed" if job.failures else "",
        )
        self._emit(loop, job, {"type": "done", **job.snapshot()})

    async def _run(self, job: Job) -> None:
        loop = asyncio.get_running_loop()
        # Wait for the shared load rather than letting the worker thread start a
        # second one. Failing here keeps the error to a single report.
        if self._ensure_loaded is not None:
            # Only a load that is actually needed gets the label: a job that
            # is genuinely queued behind another must not read "loading".
            job.waiting_on = "load" if not self.backend.is_ready() else "queue"
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
            try:
                await self._ensure_loaded()
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                self._finish(loop, job, "error", short_error(exc))
                return
        # One generation at a time; queued jobs simply wait their turn here. A
        # cancel that landed during the load or the wait ends the job untouched.
        job.waiting_on = "queue" if self._gate.locked() else None
        async with self._gate:
            if job.cancel_requested:
                self._finish(loop, job, "cancelled")
                return
            job.status = "running"
            job.waiting_on = None
            job.started_at = time.time()
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
            try:
                stopped = await loop.run_in_executor(self._pool, self._generate, job, loop)
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                # The whole message, here, where it can be read: the banner's
                # copy is cut, and an OOM's remedy is its last sentence.
                log.error("job %s failed: %s", job.id, exc)
                log.debug("job traceback", exc_info=True)
                self._finish(loop, job, "error", short_error(exc))
                return
            # Cancelled only if the loop actually stopped early. A cancel that
            # lands after the last step - during the decode or the encode -
            # changes nothing, and used to relabel a complete run anyway.
            if job.failures and not stopped:
                job.error = f"{job.failures} cell(s) failed and were skipped; last: {job.error}"
            self._finish(loop, job, "cancelled" if stopped else "done")

    # A cell that raises is skipped and the next one runs - up to this many in a
    # row, or this share of the grid in total, after which the failure is taken
    # as the night's and the job ends. One bad cell should cost one image; a
    # marginal failure hitting every third cell should not limp until morning.
    MAX_CONSECUTIVE_FAILURES = 3

    @staticmethod
    def _too_many_failures(consecutive: int, total: int, cells: int) -> bool:
        return consecutive >= JobManager.MAX_CONSECUTIVE_FAILURES or total > max(3, cells // 3)

    def _generate(self, job: Job, loop: asyncio.AbstractEventLoop) -> bool:
        """Run the job's cells back to back, publishing images as they land.

        Returns True if it left the loop early - a cancel - and False if every
        cell was reached, so the caller can label the job honestly.

        The loop cannot tell a sweep from a grid, which is the point: the
        cancellation checks, the progress events and the per-image seed maths
        are written once and both paths get them.
        """

        def should_cancel() -> bool:
            return job.cancel_requested

        def on_progress(step: int, total: int) -> None:
            job.step, job.total_steps = step, total
            job.last_progress_at = time.time()
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
                    # Seconds since the job started running, on the server's
                    # clock, so the browser's elapsed time and its ETA do not
                    # depend on the two clocks agreeing. The ETA is the job's
                    # own finished-cell timing, not the gallery's per-step
                    # median: it tracks the slowdown it is meant to reveal.
                    "elapsed": time.time() - (job.started_at or job.created_at),
                },
            )

        def on_preview(jpeg: bytes, step: int, total: int) -> None:
            # A few kilobytes of JPEG as a data URL, straight into the stream:
            # the transport to the browser is already built, and an <img> takes
            # the string as it is. Only ever to subscribers - nothing is kept -
            # and checked *here*, per frame: the browser opens its stream after
            # the POST that queued the job returns, by which time an idle
            # server is already inside the first cell, so deciding once at the
            # cell's start would have skipped every first cell.
            if not job.subscribers:
                return
            self._emit(
                loop,
                job,
                {
                    "type": "preview",
                    "step": step,
                    "total_steps": total,
                    "run": job.run,
                    "total_runs": job.total_runs,
                    "image": data_url(jpeg),
                },
            )

        consecutive = 0
        for index, cell in enumerate(job.cells):
            if job.cancel_requested:
                return True
            if job.pause_requested:
                self._hold(job, loop, index)
                if job.cancel_requested:
                    return True
            # The budget means "start no cell after": a cell in flight runs to
            # its end, and what was not reached is a resume, not a loss.
            if job.deadline is not None and time.time() > job.deadline:
                job.stopped_by = "budget"
                log.info(
                    "job %s: time budget reached before cell %d/%d", job.id, index + 1, len(job.cells)
                )
                return True
            req, seed = cell.request, cell.seed
            # A grid varies the step count between cells, so the totals the
            # progress events carry are the cell's, not the job's. Without the
            # images: they were each delivered as their own event already.
            job.run, job.step, job.total_steps, job.label = index + 1, 0, req.steps, cell.label
            log.info(
                "job %s: cell %d/%d seed %d%s",
                job.id, index + 1, len(job.cells), seed, f"  {cell.label}" if cell.label else "",
            )
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})

            started = time.perf_counter()
            try:
                images = self.backend.generate(
                    req,
                    seed=seed,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                    on_preview=on_preview,
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                self._save(job, loop, cell, images, elapsed_ms)
            except Cancelled:
                log.info(
                    "job %s cancelled during run %d/%d", job.id, index + 1, len(job.cells)
                )
                return True
            except Exception as exc:  # noqa: BLE001 - a cell's failure, not the night's
                consecutive += 1
                job.failures += 1
                job.error = short_error(exc)
                log.error(
                    "job %s: cell %d/%d failed: %s", job.id, index + 1, len(job.cells), exc
                )
                log.debug("cell traceback", exc_info=True)
                if self._too_many_failures(consecutive, job.failures, len(job.cells)):
                    raise
                continue
            consecutive = 0
        return False

    def _hold(self, job: Job, loop: asyncio.AbstractEventLoop, index: int) -> None:
        """Wait at a cell boundary until resumed or cancelled, the card handed back.

        Polled, not `Event.wait()`: a bare wait would deadlock `shutdown()`,
        which cancels and then joins this thread, and would make `cancel()` a
        no-op on a paused job. A quarter second is well inside what a resume
        click can feel.
        """
        job.paused = True
        try:
            self.backend.free_cache()
        except Exception:  # noqa: BLE001 - a pause must not fail the night
            log.debug("free_cache failed", exc_info=True)
        log.info("job %s paused before cell %d/%d", job.id, index + 1, len(job.cells))
        self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
        while job.pause_requested and not job.cancel_requested:
            time.sleep(0.25)
        job.paused = False
        if not job.cancel_requested:
            log.info("job %s resumed at cell %d/%d", job.id, index + 1, len(job.cells))
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})

    def _save(
        self,
        job: Job,
        loop: asyncio.AbstractEventLoop,
        cell: Cell,
        images: list,
        elapsed_ms: int,
    ) -> None:
        """Write a cell's images and publish each one as it lands."""
        # Record what ran, not what was asked: a negative prompt the pipeline
        # dropped below its threshold, or a mu it had no way to take, must not
        # reach the row - resume compares against it.
        ran = self.backend.applied(cell.request)
        seed = cell.seed
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
                variables=cell.variables,
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
