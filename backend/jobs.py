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
from .grid import Cell, progress_steps, run_id as _run_id
from .latent_preview import data_url
from .schemas import GenerateRequest
from .seeds import random_seed, wrap_seed

log = logging.getLogger(__name__)

_TERMINAL = {"done", "error", "cancelled"}


class BackendBusy(RuntimeError):
    """An interactive backend operation would cut ahead of an image job."""


class LoadFailed(RuntimeError):
    """The image model's load failed. `detail` is the sentence the status
    gives for it (`main.ensure_loaded`), what to do included, so a job that
    ended at the load says what the topbar and the viewer say - not the
    library's six lines - and the cause is the exception it was raised from."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


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
    # Behind the gate it names what holds it (`JobManager._waiting_label`):
    # "paused" for a job ahead that is paused - which keeps the gate, so a job
    # queued from another tab waited all night under "the job ahead" - and
    # "critic" or "expand" for an editor action on the card, where no job is
    # ahead at all.
    waiting_on: str | None = None
    step: int = 0
    total_steps: int = 0
    run: int = 0
    total_runs: int = 1
    # What the current cell varies, for a grid; None for an ordinary sweep.
    label: str | None = None
    images: list[dict] = field(default_factory=list)
    # The ids of those images deleted since (`JobManager.forget_images`).
    # `images` keeps them - `image_count` is how much the job produced, which
    # a delete does not undo - and the replay leaves them out.
    deleted: set[int] = field(default_factory=set)
    error: str | None = None
    # Cells that raised and were skipped. A grid survives one bad cell - a
    # transient OOM, a locked database - and reports it, rather than dropping
    # every remaining cell of the night on the first.
    failures: int = 0
    # Which cells those were - each one's label, seed and error - so the page
    # can name the hole in the sheet rather than count it.
    failed_cells: list[dict] = field(default_factory=list)
    # Submitted as cells (a grid, a sweep, a re-render) rather than as one
    # request's repeats: what the page calls each one, a cell or an image.
    grid: bool = False
    # A grid's prompt as submitted, its {variables} unfilled, which names
    # the job: its cells' prompts differ, and the first one's named the whole
    # night long after that cell had finished. None for a plain request, whose
    # cells share theirs, and for a re-render, whose cells are its records'.
    prompt_template: str | None = None
    # A prompt comparison's grid (`prompt_variants`): its versions are not in
    # the panel, so the panel's grid can never resume one, and a page offering
    # Resume for a stopped job has to know - the snapshot is otherwise a grid
    # with a prompt like any other.
    comparison: bool = False
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
    # the job when it was not the user: "budget" (through the cancel path),
    # "shutdown" (the server stopping, through the cancel path too - a page
    # must not report that as a Cancel pressed somewhere) or "failures" (the
    # failure rule, through `error`). Ending at the deadline
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
        must be the list as of now. The records replayed are the live ones: a
        page reattaching after a delete got the deleted image back as a broken
        tile, and deleting it again answered "unknown image". The skipped
        cells travel with the records, for the same reason: at most a third
        of a grid, but every per-cell event would have carried them.
        """
        frame = {
            "job_id": self.id,
            "run_id": self.run_id,
            "seed": self.seed,
            # What the job is, for a page that did not submit it - a reload,
            # the queue note naming each job waiting, the line over the bars:
            # the first cell's prompt and a grid's own, a pass's source, the
            # size, and how many images the whole job makes.
            "prompt": self.request.prompt[:120],
            "refine": self.request.refine,
            "width": self.request.width,
            "height": self.request.height,
            "planned_images": sum(cell.request.num_images for cell in self.cells),
            "grid": self.grid,
            "prompt_template": None if self.prompt_template is None else self.prompt_template[:120],
            "comparison": self.comparison,
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
            # How many of them have been deleted since, so a banner can say
            # what is left of a night without lowering what it produced.
            "deleted_count": len(self.deleted),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_progress_at": self.last_progress_at,
            "deadline": self.deadline,
            "stopped_by": self.stopped_by,
        }
        if images:
            frame["images"] = [image for image in self.images if image["id"] not in self.deleted]
            frame["failed_cells"] = list(self.failed_cells)
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
        # Who holds the gate: a job's id, or the editor action `run_when_idle`
        # was given ("critic", "expand"), None while it is free. What a job
        # queued behind it waits on is read off this (`_waiting_label`).
        self._holder: str | None = None

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

    def forget_images(self, ids) -> None:
        """Note gallery images deleted, on every job that made one of them.

        Called by the delete routes once storage has committed. Live and
        finished jobs alike: a reload after the run has ended reads the
        finished job's snapshot as much as a reattach reads a running one's.
        Event-loop thread; the worker only ever appends to `images`, which a
        list survives being read across.
        """
        gone = set(ids)
        for job in self._jobs.values():
            job.deleted |= gone & {image["id"] for image in job.images}

    def busy(self) -> bool:
        """True while anything is queued or running.

        Wider than `self._gate.locked()` on purpose: a job still waiting its turn
        starts on its own, so acting in front of it is the same mistake one step
        later.
        """
        return bool(self.active())

    async def run_when_idle(self, fn: Callable[[], Any], holder: str = "editor") -> Any:
        """Run a synchronous backend operation without racing the job queue.

        Prompt expansion and the prompt critic use the same accelerator as
        image generation - ERNIE's rewriter through the pipeline, the critic
        through a llama-server holding ~5 GB - but are interactive rather than
        jobs. Refuse when work is already promised; once admitted, hold the
        generation gate so a newly submitted job waits behind it instead of
        running both models together. The check is made again inside the
        gate: a job queued while this waited for it came first. `holder`
        names the operation ("critic", "expand") to a job queued behind it,
        whose `waiting_on` says so: "waiting for the job ahead…" named a job
        that did not exist.
        """
        if self.busy():
            raise BackendBusy("an image job is queued or running")
        async with self._gate:
            if self.busy():
                raise BackendBusy("an image job was queued first")
            self._holder = holder
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(self._pool, fn)
            finally:
                self._holder = None

    def _waiting_label(self) -> str:
        """What a job waiting for the gate is behind: the editor action that
        holds it, a job ahead that is paused - it keeps the gate until it is
        resumed or cancelled, where it was started - or a job ahead."""
        holder = self._holder
        if holder is None:
            return "queue"
        ahead = self._jobs.get(holder)
        if ahead is None:
            return holder
        return "paused" if ahead.paused else "queue"

    def _relabel_queued(self, loop: asyncio.AbstractEventLoop) -> None:
        """Say again what each job waiting for the gate is behind, to its
        streams: after the job holding it pauses or resumes. A job still
        waiting on the load is left to say so."""
        label = self._waiting_label()
        for job in self.active():
            if job.status == "queued" and job.waiting_on not in (None, "load") and job.waiting_on != label:
                job.waiting_on = label
                self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})

    def stop_all(self, reason: str = "shutdown") -> None:
        """Ask every live job to stop, saying why (`stopped_by`): the server
        is going, which a page must not report as a Cancel pressed in some
        other tab. The cancel itself is the button's, cooperative."""
        for job in self.active():
            job.stopped_by = job.stopped_by or reason
            job.cancel_requested = True

    async def wait_for_gate(self) -> None:
        """Return once nothing holds the generation gate, without keeping it.

        For the image model's load, which moves the weights onto the card but
        runs outside the gate: a queued job awaits the load before it takes its
        turn. Without this a job submitted during a critique started loading
        13 GB beside the critic's server. Nothing that holds the gate ever
        awaits the load lock - `_run` loads before it enters, and
        `run_when_idle`'s operations are synchronous and never await
        `ensure_loaded` (Expand's first use does load ERNIE's rewriter inside
        the gate, but through the backend, not the lock) - so this cannot
        deadlock.
        """
        async with self._gate:
            pass

    async def shutdown(self) -> None:
        """Stop every live job and wait for the worker thread to drain.

        uvicorn does not wait for a generation thread, so without this a Ctrl-C
        mid-sweep either hangs on an open SSE stream or has the weights pulled
        out from under a running denoise. The cancel is the same cooperative one
        the button uses, so it lands within a step and keeps what it finished.
        """
        self.stop_all("shutdown")
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
            ],
            grid=False,
        )

    async def submit_cells(
        self,
        cells: list[Cell],
        run_id: str | None = None,
        budget_minutes: float | None = None,
        grid: bool = True,
        prompt_template: str | None = None,
        comparison: bool = False,
    ) -> Job:
        """Queue an already-expanded list of pipeline calls - a grid, usually.

        `run_id` names the run the cells belong to. A resumed grid hands in the
        id of the *whole* grid, computed before the finished cells were
        subtracted, so its images join the night they continue; left out, the
        cells themselves are the definition. `budget_minutes` is plumbed here
        rather than read off `job.request`: GridRequest-only fields are gone
        by the time the cells are built - as is the grid's own prompt, which
        `prompt_template` carries for the snapshot to name the job by, and
        whether its prompts were a comparison's versions (`comparison`).
        """
        job = Job(
            id=uuid.uuid4().hex[:12],
            request=cells[0].request,
            cells=cells,
            seed=cells[0].seed,
            run_id=run_id or _run_id(cells, settings.model_id, settings.quantization),
            total_steps=progress_steps(cells[0].request),
            total_runs=len(cells),
            deadline=time.time() + budget_minutes * 60 if budget_minutes else None,
            grid=grid,
            prompt_template=prompt_template,
            comparison=comparison,
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
            # is genuinely queued behind another must not read "loading" -
            # nor one with nothing ahead of it "the job ahead".
            job.waiting_on = (
                "load" if not self.backend.is_ready()
                else self._waiting_label() if self._gate.locked() else None
            )
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
            try:
                await self._ensure_loaded()
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                detail = exc.detail if isinstance(exc, LoadFailed) else short_error(exc)
                self._finish(loop, job, "error", detail)
                return
        # One generation at a time; queued jobs simply wait their turn here. A
        # cancel that landed during the load or the wait ends the job untouched.
        # Waiting, it says what it waits behind - a label the load's frame
        # could not know, or one the gate's holder has changed since - and
        # the page hears it rather than "loading weights" for the wait.
        waiting = self._waiting_label() if self._gate.locked() else None
        if waiting is not None and waiting != job.waiting_on:
            job.waiting_on = waiting
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
        job.waiting_on = waiting
        async with self._gate:
            self._holder = job.id
            try:
                await self._run_holding(job, loop)
            finally:
                self._holder = None

    async def _run_holding(self, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        """The job's turn, the gate held."""
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
            error = short_error(exc)
            if job.stopped_by == "failures":
                # The failure rule ended it, which is three in a row or a
                # third of the grid - not one bad cell, which the last
                # exception alone made it read as. Where, how many and
                # what was kept come first; Resume runs the rest.
                kept = len(job.images)
                error = (
                    f"Stopped at cell {job.run} of {job.total_runs} after {job.failures} "
                    f"cells failed; {kept} image{'' if kept == 1 else 's'} kept. Last: {error}"
                )
            self._finish(loop, job, "error", error)
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

        # The live class may disagree with the profile used at submit. Check
        # the entire job before it saves anything, rather than fail each cell
        # or quietly normalize its source away.
        for cell in job.cells:
            self.backend.check_refine_supported(cell.request)

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
            # progress events carry are the cell's, not the job's - and on a
            # refine cell the part of the schedule its strength leaves, until
            # the first callback says what the pipeline made of that. Without
            # the images: they were each delivered as their own event already.
            job.run, job.step, job.label = index + 1, 0, cell.label
            job.total_steps = progress_steps(req)
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
                job.failed_cells.append({"label": cell.label, "seed": seed, "error": job.error})
                log.error(
                    "job %s: cell %d/%d failed: %s", job.id, index + 1, len(job.cells), exc
                )
                log.debug("cell traceback", exc_info=True)
                if self._too_many_failures(consecutive, job.failures, len(job.cells)):
                    job.stopped_by = "failures"
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

        A paused job keeps the gate, so every job queued behind it is told -
        "paused", then "queue" again - on the event loop, which owns the job
        table this thread must not walk while a submission adds to it.
        """
        job.paused = True
        try:
            self.backend.free_cache()
        except Exception:  # noqa: BLE001 - a pause must not fail the night
            log.debug("free_cache failed", exc_info=True)
        log.info("job %s paused before cell %d/%d", job.id, index + 1, len(job.cells))
        self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
        loop.call_soon_threadsafe(self._relabel_queued, loop)
        while job.pause_requested and not job.cancel_requested:
            time.sleep(0.25)
        job.paused = False
        if not job.cancel_requested:
            log.info("job %s resumed at cell %d/%d", job.id, index + 1, len(job.cells))
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
        loop.call_soon_threadsafe(self._relabel_queued, loop)

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
