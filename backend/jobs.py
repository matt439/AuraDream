"""Single-worker job queue.

The GPU can only usefully run one generation at a time, so every request is
funnelled through one worker thread. Progress hops back to the event loop via
`call_soon_threadsafe` and leaves as SSE.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from typing import Any

from . import gallery_lock, gallery_reports, storage
from .backends import Cancelled, GenerationBackend
from .model_selection import current_settings
from .config import settings
from .error_text import short_error
from .execution import require_configured_loader_available, require_default_execution, resolve_execution
from .grid import Cell, CellKey, cell_keys, progress_steps, run_id as _run_id
from .job_state import _TERMINAL, Job
from .latent_preview import data_url
from .more_seeds_intents import IntentLedger
from .schemas import GenerateRequest, ThenRefine
from .seeds import random_seed, wrap_seed
from .then_refine import execution_units, pass_run_id, plan_passes, stand_in

log = logging.getLogger(__name__)


class BackendBusy(RuntimeError):
    """An interactive backend operation would cut ahead of an image job."""


class MaskGone(ValueError):
    """An inpaint's mask went between the submit's check and its hold: the
    job is not queued, and the route answers 422 with these words."""


class OutpaintStampFailed(RuntimeError):
    """An outpaint's mask could not be stored as its job was submitted -
    the disk, or the gallery reserved by another operation: the job is not
    queued, and the route answers 503 with these words."""


class RunPersistenceFailed(RuntimeError):
    """Resolved cells could not be recorded, so the job was not queued."""


class LoadFailed(RuntimeError):
    """The image model's load failed. `detail` is the sentence the status
    gives for it (`main.ensure_loaded`), what to do included, so a job that
    ended at the load says what the topbar and the viewer say - not the
    library's six lines - and the cause is the exception it was raised from."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _cell_masks(cells: list[Cell]) -> set[str]:
    """The inpaint masks `cells` will read."""
    return {
        cell.request.mask
        for cell in cells
        if cell.request.refine is not None and cell.request.mask is not None
    }


def _release_masks(job_id: str) -> None:
    try:
        storage.release_masks(job_id)
    except Exception:  # noqa: BLE001 - a stale hold only spares a mask until the next start
        log.warning("could not release the masks of job %s", job_id, exc_info=True)


def _retain_outcome(job_id: str, outcome: dict[str, Any], database) -> None:
    try:
        storage.finish_run_job(job_id, outcome, database=database)
    except Exception:  # noqa: BLE001 - NULL falls back to held-versus-missing cells
        log.warning("could not record the outcome of job %s", job_id, exc_info=True)


class JobManager:
    def __init__(
        self,
        backend: GenerationBackend,
        ensure_loaded: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.backend = backend
        self._ensure_loaded = ensure_loaded
        self._jobs: dict[str, Job] = {}
        # Admission is ours, rather than asyncio.Lock's FIFO waiters. A moved
        # head is checked again inside the gate before it may generate.
        self._pending: list[str] = []
        self._order_changed = asyncio.Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
        self._gate = asyncio.Lock()
        # Opt-in admission for reruns and continuations that subtract live cells: hold from
        # the queue snapshot through gallery reads and every submit_cells
        # await (outpaint stamping and mask holds included). Ordinary
        # Generate and deliberate repeats do not take it.
        self.rerun_admission = asyncio.Lock()
        self.more_seeds_intents = IntentLedger()
        # Who holds the gate: a job's id, or the editor action `run_when_idle`
        # was given ("critic", "expand"), None while it is free. What a job
        # queued behind it waits on is read off this (`_waiting_label`).
        self._holder: str | None = None
        # The ticket the editor action holding the gate was sent with, if any
        # (`run_when_idle`): a page whose Expand is waiting tells its own from
        # another tab's, which holds the card while this one waits for it.
        self._ticket: str | None = None
        # Jobs past their load and waiting for the gate: the ones whose
        # `waiting_on` follows the holder. A job still awaiting the load keeps
        # saying so, whatever holds the gate.
        self._at_gate: set[str] = set()
        # The loop the jobs run on, taken at the first submission: `stop_all`
        # is called on a signal's thread and finishes queued jobs on it.
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        """The running job first, followed by the manager's admission order."""
        running = [job for job in self._jobs.values() if job.status == "running"]
        queued = [self._jobs[job_id] for job_id in self._pending
                  if self._jobs[job_id].status == "queued"]
        return running + queued

    def _wake_order(self) -> None:
        changed, self._order_changed = self._order_changed, asyncio.Event()
        changed.set()

    def move_front(self, job_id: str) -> Job | None:
        """Put one still-queued job next; never interrupt the current holder."""
        job = self.get(job_id)
        if job is None:
            return None
        if job.status != "queued" or job_id not in self._pending:
            raise ValueError("Only a queued job can move to the front")
        self._pending.remove(job_id)
        self._pending.insert(0, job_id)
        self._wake_order()
        return job

    def skip(self, job_id: str, call: int) -> Job | None:
        """Request Skip for an exact active cell, atomically with its token."""
        job = self.get(job_id)
        if job is None:
            return None
        with job._call_lock:
            if job.status != "running" or job.paused or job.stage == "pass" or job.cancel_requested:
                # Do not call skip_state while holding its non-reentrant lock.
                refused = True
            else:
                refused = False
                if job._current_call != call:
                    raise ValueError("That cell has already finished; no other cell was skipped")
                job._skip_call = call
        if refused:
            raise ValueError(job.skip_state()[2])
        self._emit(asyncio.get_running_loop(), job, {"type": "status", **job.snapshot(images=False)})
        return job

    def finished(self) -> list[Job]:
        """Jobs that have ended, the most recently finished first.

        In memory only, so it answers "what happened to last night's run" for
        as long as this server process is up - the gallery's rows answer "how
        many and when" for good, but never "why".
        """
        done = [job for job in self._jobs.values() if job.status in _TERMINAL]
        return sorted(done, key=lambda job: job.finished_at or 0.0, reverse=True)

    def forget_images(self, ids, binned: bool = False) -> None:
        """Note gallery images gone - deleted, or moved to the recycle bin
        (`binned`) - on every job that made one of them.

        Called by the delete and bin routes once storage has committed. Live
        and finished jobs alike: a reload after the run has ended reads the
        finished job's snapshot as much as a reattach reads a running one's.
        Event-loop thread; the worker only ever appends to `images`, which a
        list survives being read across.
        """
        gone = set(ids)
        for job in self._jobs.values():
            job.deleted |= gone & {image["id"] for image in job.images}
            # And a pass made from one of them cannot be made again: its
            # record says so in a replay, as the gallery's does - until a
            # Restore, where the source was only binned.
            for image in job.images:
                if (image.get("edit") or {}).get("source") in gone:
                    image["edit_missing"] = True
                    image["edit_binned"] = binned
                if image.get("refine") in gone:
                    image["refine_missing"] = True
                    image["refine_binned"] = binned

    def restore_images(self, ids) -> None:
        """Undo `forget_images(binned=True)` for images restored from the
        recycle bin: replayed again, counted again, and a pass made from one
        of them usable again. Event-loop thread, like `forget_images`."""
        back = set(ids)
        for job in self._jobs.values():
            job.deleted -= back
            for image in job.images:
                if (image.get("edit") or {}).get("source") in back:
                    # Availability requires fresh pixel proof; restoration alone
                    # cannot make an archived source link trustworthy again.
                    record = storage.get_image(image["id"])
                    image["edit_missing"] = record is None or record.edit_missing
                    image["edit_binned"] = record is not None and record.edit_binned
                if image.get("refine") in back:
                    image["refine_missing"] = False
                    image["refine_binned"] = False

    def planned_keys(self) -> dict[tuple[int, int], set[CellKey]]:
        """What the live jobs will write, by size: every image's resume key
        as `applied()` runs its cell. More seeds counts these as made, so a
        second press while the first is still queued continues past it
        rather than rendering the same seeds again. Event-loop thread; a
        job's cells are fixed when it is made."""
        planned: dict[tuple[int, int], set[CellKey]] = {}
        editable = self.backend.instruction_edit_refusal(current_settings(settings)) is None
        for job in self.active():
            for cell in [*job.cells, *list(job.pending_passes), *list(job.source_passes)]:
                if cell.request.edit is not None and not editable:
                    # Its backend has closed the gate since, or the selection
                    # left the measured envelope: the cell fails as it begins
                    # (`_recheck_edit`) and writes nothing.
                    continue
                ran = Cell(request=self.backend.applied(cell.request), seed=cell.seed)
                planned.setdefault((cell.request.width, cell.request.height), set()).update(cell_keys(ran))
        return planned

    def masks_in_flight(self) -> set[str]:
        """The inpaint masks the live jobs' cells will read: what a mask sweep
        (`gallery_delete.sweep_masks`) spares beside the rows, since a queued
        cell's mask is in no row yet. Called from the sweep's thread, so the
        jobs and cells are copied before they are walked; a job's cells are
        fixed when it is made."""
        masks: set[str] = set()
        for job in list(self._jobs.values()):
            if job.status in _TERMINAL:
                continue
            masks.update(_cell_masks(list(job.cells)))
        return masks

    def busy(self) -> bool:
        """True while work is queued, running or settling an owned admission.

        Wider than `self._gate.locked()` on purpose: a job still waiting its turn
        starts on its own, so acting in front of it is the same mistake one step
        later.
        """
        return bool(self.active()) or self.more_seeds_intents.pending

    async def run_when_idle(
        self, fn: Callable[[], Any], holder: str = "editor", ticket: str | None = None
    ) -> Any:
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
        that did not exist. `ticket` is the caller's own name for this call,
        answered by `card()` while it holds the gate.
        """
        if self.busy():
            raise BackendBusy("an image job is queued or running")
        async with self._gate:
            if self.busy():
                raise BackendBusy("an image job was queued first")
            self._holder = holder
            self._ticket = ticket
            try:
                loop = asyncio.get_running_loop()
                worker = loop.run_in_executor(self._pool, fn)
                try:
                    return await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # A disconnected editor request cannot stop synchronous
                    # GPU work. Keep its gate and ownership until the worker
                    # has actually unwound, even after repeated cancellation.
                    while not worker.done():
                        try:
                            await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if worker.done() and not worker.cancelled():
                        worker.exception()  # observe a failure after disconnect
                    raise
            finally:
                self._holder = None
                self._ticket = None

    def card(self) -> tuple[str | None, str | None]:
        """What holds the generation gate, for the status: "job", or the
        editor action's name ("critic", "expand"), with the ticket it was
        sent with; (None, None) while the gate is free. A page whose Expand
        is waiting says it waits for the card, and for whose: Expand's
        running line said "Expanding…" for the whole of another tab's
        critique, which held the card."""
        holder = self._holder
        if holder is None:
            return None, None
        if holder in self._jobs:
            return "job", None
        return holder, self._ticket

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

    def _wait_on(self, job: Job, label: str | None) -> None:
        """Set what `job` waits behind, and since when the job ahead has
        been paused where that is what it waits on."""
        job.waiting_on = label
        ahead = self._jobs.get(self._holder) if label == "paused" and self._holder else None
        job.ahead_paused_at = None if ahead is None else ahead.paused_at

    def _relabel_queued(self, loop: asyncio.AbstractEventLoop) -> None:
        """Say again what each job waiting for the gate is behind, to its
        streams: when the gate changes hands, and after the job holding it
        pauses or resumes. Behind the critic, "critic" outlived the critic's
        answer: the second of two jobs queued during a critique said so for
        the whole of the first's run. A job still waiting on the load is
        left to say so."""
        label = self._waiting_label()
        for job in self.active():
            if job.id in self._at_gate and job.status == "queued" and job.waiting_on != label:
                self._wait_on(job, label)
                self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})

    def stop_all(self, reason: str = "shutdown") -> None:
        """Ask every live job to stop, saying why (`stopped_by`): the server
        is going, which a page must not report as a Cancel pressed in some
        other tab. The cancel itself is the button's, cooperative.

        A job still queued is finished here, as `cancel()` finishes one: one
        waiting on the load otherwise held its stream open until the load
        ended, and uvicorn drains streams before the lifespan runs - a
        Ctrl-C during a first load waited out the 30 s graceful timeout, and
        the stream was cut with no `done`. Called on the signal's thread, so
        the finish goes to the loop the jobs run on; `_run` then returns
        quietly behind `_finish`'s guard, wherever the load leaves it.
        """
        loop = self._loop
        for job in self.active():
            job.stopped_by = job.stopped_by or reason
            job.cancel_requested = True
            if job.status == "queued" and loop is not None:
                try:
                    loop.call_soon_threadsafe(self._finish_queued, loop, job)
                except RuntimeError:  # the loop has closed: nothing listens
                    pass

    def _finish_queued(self, loop: asyncio.AbstractEventLoop, job: Job) -> None:
        """`stop_all`'s finish, on the loop: still queued by the time it runs
        - a job that took its turn meanwhile stops at its next step."""
        if job.status == "queued":
            self._finish(loop, job, "cancelled")

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
        # Admission owns its task across a disconnected HTTP caller. Settle it
        # before stopping jobs, so no task can enqueue into a closed executor.
        await self.more_seeds_intents.close()
        self.stop_all("shutdown")
        await asyncio.to_thread(self._pool.shutdown, True)

    @staticmethod
    async def _price(
        job_id: str, cells: list[Cell], then_refine: ThenRefine | None, model_id: str,
        source_passes: list[Cell] | None = None,
    ) -> float | None:
        """The job's time from matching history, off the loop and without a
        PNG stat: Finish run's missing passes, then the cells as the worker
        runs them, each followed by Then refine's stand-in passes, as the
        grid preview prices them. Never a reason to refuse the job - a busy
        gallery or any error is None."""
        try:
            return await asyncio.to_thread(
                gallery_reports.job_seconds, [*(source_passes or []), *execution_units(cells, then_refine)],
                model_id,
            )
        except Exception as exc:
            log.warning("job %s: no time estimate (%s)", job_id, exc)
            return None

    async def submit(self, request: GenerateRequest, then_refine: ThenRefine | None = None) -> Job:
        """An ordinary request: `runs` passes of one setting, seed stepped.

        Each pass steps the seed past the whole of the previous batch, so every
        image in a sweep keeps a seed that reproduces it on its own. Wrapped
        rather than allowed to overflow the signed 64-bit seed column.
        """
        require_default_execution(request.execution)
        selected = current_settings(settings)
        require_configured_loader_available(selected)
        request = request.with_execution(selected.loader_execution)
        require_default_execution(request.execution)
        base = request.seed if request.seed is not None else random_seed()
        return await self.submit_cells(
            [
                Cell(request=request, seed=wrap_seed(base + index * request.num_images))
                for index in range(request.runs)
            ],
            grid=False,
            then_refine=then_refine,
        )

    async def submit_cells(
        self,
        cells: list[Cell],
        run_id: str | None = None,
        budget_minutes: float | None = None,
        grid: bool = True,
        prompt_template: str | None = None,
        comparison: bool = False,
        swap: str | None = None,
        refine_batch: float | None = None,
        more_seeds: int | None = None,
        then_refine: ThenRefine | None = None,
        pass_sources: list[int] | None = None,
    ) -> Job:
        """Queue an already-expanded list of pipeline calls - a grid, usually.

        `run_id` names the run the cells belong to. A resumed grid hands in the
        id of the *whole* grid, computed before the finished cells were
        subtracted, so its images join the night they continue; left out, the
        cells themselves are the definition. `budget_minutes` is plumbed here
        rather than read off `job.request`: GridRequest-only fields are gone
        by the time the cells are built - as is the grid's own prompt, which
        `prompt_template` carries for the snapshot to name the job by, and
        whether its prompts were a comparison's versions (`comparison`), and
        Swap tags' rows (`swap`), Mass refine's scale (`refine_batch`),
        More seeds' count (`more_seeds`) and Then refine's pass of every image
        (`then_refine`, its strength resolved by the route).

        `pass_sources` are Finish run's missing passes: the gallery ids of
        bases of this run saved without their Then refine pass, each run as
        a unit of its own before the first cell. They need `then_refine` and
        `run_id`, and `cells` may then be empty - a run missing only passes -
        where the job's request is its first pass's.
        """
        sources = list(dict.fromkeys(pass_sources or []))
        if sources and (then_refine is None or run_id is None):
            raise ValueError("missing passes need their run's Then refine and run id")
        source_passes = (
            await asyncio.to_thread(self._plan_source_passes, sources, then_refine)
            if sources and then_refine is not None else []
        )
        if not cells and not source_passes:
            raise ValueError("nothing to queue: every base whose pass was missing has gone since the check")
        # Internal model_copy cells bypass request validation. Refuse the
        # whole batch before stamping masks, retaining work or starting a load,
        # including callers that supply their own run id. Stamped outpaints
        # deliberately cannot go through whole-request validation again.
        for cell in cells:
            if cell.request.edit is not None:
                from .instruction_edit import bound_geometry_refusal, require_execution_available

                # Only a bound edit (proved by `bind_request`) queues, and
                # only behind the backend instance's own gate, inside the
                # measured envelope: its loader and its source's aspect.
                bound = cell.request.validate_edit(require_bound=True)
                require_execution_available(bound, qualified=self.backend.instruction_edit_qualified())
                reason = self.backend.instruction_edit_refusal(current_settings(settings)) or (
                    bound_geometry_refusal(bound)
                )
                if reason is not None:
                    raise ValueError(reason)
            require_default_execution(cell.request.execution)
            if "step_cache" in cell.request.model_fields_set:
                raise ValueError("Resolve composer step cache before submitting expanded cells.")
        selected = current_settings(settings)
        require_configured_loader_available(selected)
        # Expanded/stored cells already describe complete requests. Missing
        # legacy execution is Eager replay here, never fresh composer intent.
        cells = [replace(cell, request=cell.request.model_copy(update={
            "execution": resolve_execution(selected.loader_execution, cell.request.execution),
        })) for cell in cells]
        self._loop = asyncio.get_running_loop()
        raw_cells = list(cells)
        # An outpaint's mask, built and stored before anything reads the
        # cells - the job's own list, its request, the holds below - so the
        # worker runs the stamped cell (`storage.stamp_outpaints`). The run
        # id keys an outpaint by its spec, so it is the same either side.
        if any(cell.request.outpaint is not None for cell in cells):
            try:
                cells = await asyncio.to_thread(storage.stamp_outpaints, cells)
            except gallery_lock.GalleryBusy as exc:
                raise OutpaintStampFailed(
                    "the gallery is busy with another operation; submit the outpaint again"
                ) from exc
            except OSError as exc:
                raise OutpaintStampFailed(
                    f"the outpaint's mask could not be written: {exc.strerror or exc}"
                ) from exc
        resolved_run_id = run_id or _run_id(cells, selected.model_id, selected.quantization)
        # A job of missing passes alone reads its first pass where a job
        # reads its first cell: its request, seed, size and batch.
        first = (cells or source_passes)[0]
        job = Job(
            id=uuid.uuid4().hex[:12],
            request=first.request,
            cells=cells,
            seed=first.seed,
            run_id=resolved_run_id,
            total_steps=progress_steps(first.request),
            total_runs=len(sources) + len(cells),
            pass_sources=sources,
            source_passes=source_passes,
            deadline=time.time() + budget_minutes * 60 if budget_minutes else None,
            grid=grid,
            prompt_template=prompt_template,
            comparison=comparison,
            swap=swap,
            refine_batch=refine_batch,
            more_seeds=more_seeds,
            then_refine=then_refine,
        )
        if then_refine is not None:
            job.pass_run_id = pass_run_id(resolved_run_id, then_refine)
        job.estimate_seconds = await self._price(job.id, cells, then_refine, selected.model_id, source_passes)
        # Held where every process's mask sweep looks, before the job is
        # queued: a CLI `clear --bin` an hour on must not take the mask of a
        # cell still waiting its turn (`storage.hold_masks`).
        masks = _cell_masks(cells)
        if masks:
            gone = await asyncio.to_thread(storage.hold_masks, job.id, masks)
            if gone:
                await asyncio.to_thread(storage.release_masks, job.id)
                # An outpaint's user never painted: its mask is built again
                # by submitting it again.
                built = any(
                    cell.request.mask == gone[0] and cell.request.outpaint is not None for cell in cells
                )
                raise MaskGone(
                    f"mask {gone[0][:16]} has just gone from this gallery; "
                    + ("submit it again" if built else "paint it again")
                )
            job.holds_masks = True
        origin = (
            "more_seeds" if more_seeds is not None else
            "refine_batch" if refine_batch is not None else
            "swap" if swap is not None else
            "comparison" if comparison else
            "grid" if prompt_template is not None else
            "generate" if not grid else "cells"
        )
        definition = {
            "job_id": job.id, "run_id": job.run_id, "created_at": job.created_at,
            "origin": origin, "model_id": selected.model_id, "quantization": selected.quantization,
            "width": job.request.width, "height": job.request.height,
            "batch": job.request.num_images, "comparison": comparison,
            "metadata": {
                "grid": grid, "prompt_template": prompt_template, "swap": swap,
                "refine_batch": refine_batch, "more_seeds": more_seeds,
                "budget_minutes": budget_minutes,
                "then_refine": None if then_refine is None else {
                    **then_refine.model_dump(), "pass_run_id": job.pass_run_id,
                },
                **({"pass_sources": sources} if sources else {}),
            },
        }
        try:
            while True:
                try:
                    await asyncio.to_thread(storage.retain_run_job, definition, raw_cells)
                    break
                except gallery_lock.GalleryBusy:
                    log.warning("job %s: gallery busy; still waiting to record its cells", job.id)
        except BaseException as exc:
            if job.holds_masks:
                await asyncio.to_thread(_release_masks, job.id)
            if isinstance(exc, (OSError, sqlite3.Error)):
                detail = getattr(exc, "strerror", None) or str(exc)
                raise RunPersistenceFailed(
                    f"the job's cells could not be recorded; nothing was queued: {detail}"
                ) from exc
            raise
        self._jobs[job.id] = job
        self._pending.append(job.id)
        self._wake_order()
        # A line at the start, one per cell and one at the end: an overnight
        # grid started from the browser used to leave nothing in the log unless
        # it broke, where the CLI writes one line per cell when redirected.
        log.info(
            "job %s queued: %d cell(s)%s, run %s, seed %s, %dx%d",
            job.id, len(cells), f" and {len(sources)} missing pass(es)" if sources else "",
            job.run_id, job.seed, job.request.width, job.request.height,
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
        hands back. A queued job keeps the request and holds before its first
        cell once its turn comes - keeping the gate, as any pause does - which
        is the boundary a job not yet started has; a finished one is left
        alone.
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
        job.freeze_execution_diagnostics()
        if job.id in self._pending:
            self._pending.remove(job.id)
        self._wake_order()
        if job.holds_masks:
            # Off the loop and never awaited: a hold left by a release that
            # fails is dropped at the next start, and only spares a mask.
            threading.Thread(
                target=_release_masks, args=(job.id,), daemon=True, name="mask-release"
            ).start()
        job.status = status
        if error is not None:
            job.error = error
        job.done_sent = True
        job.finished_at = time.time()
        # This transition is synchronous on the loop. The daemon's
        # best-effort UPDATE cannot delay SSE or interpreter exit; a failed
        # or interrupted write deliberately leaves outcome NULL.
        outcome = {
            "status": status, "reached": job.run, "failed_indices": list(job.failed_indices),
            "completed_indices": list(job.completed_indices),
            "skipped_indices": list(job.skipped_indices),
            **({"completed_passes": list(job.completed_passes)} if job.completed_passes else {}),
            "stopped_by": job.stopped_by, "finished_at": job.finished_at,
        }
        threading.Thread(
            target=_retain_outcome, args=(job.id, outcome, storage.settings.db_path),
            daemon=True, name="run-outcome",
        ).start()
        since = job.started_at or job.created_at
        log.info(
            "job %s %s: %d image(s) in %.0fs%s",
            job.id, status, len(job.images), job.finished_at - since,
            f", {job.failures} cell(s) failed" if job.failures else "",
        )
        self._emit(loop, job, {"type": "done", **job.snapshot()})

    async def _run(self, job: Job) -> None:
        loop = asyncio.get_running_loop()
        # Retained/internal jobs cannot bypass the public unavailable boundary:
        # an edit job runs only while the backend still answers qualified and
        # the selection is still inside the measured envelope.
        if any(cell.request.edit is not None for cell in job.cells):
            reason = self.backend.instruction_edit_refusal(current_settings(settings))
            if reason is not None:
                self._finish(loop, job, "error", reason)
                return
        # Wait for the shared load rather than letting the worker thread start a
        # second one. Failing here keeps the error to a single report.
        if self._ensure_loaded is not None:
            # Only a load that is actually needed gets the label: a job that
            # is genuinely queued behind another must not read "loading" -
            # nor one with nothing ahead of it "the job ahead".
            self._wait_on(job, (
                "load" if not self.backend.is_ready()
                else self._waiting_label() if self._gate.locked() else None
            ))
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
            self._wait_on(job, waiting)
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
        self._wait_on(job, waiting)
        self._at_gate.add(job.id)
        try:
            while job.status not in _TERMINAL:
                changed = self._order_changed
                if not self._pending or self._pending[0] != job.id:
                    await changed.wait()
                    continue
                async with self._gate:
                    # A Move to front may land while this coroutine waits on
                    # the gate. Its waiter position must not overrule the move.
                    if job.status in _TERMINAL:
                        return
                    if not self._pending or self._pending[0] != job.id:
                        continue
                    self._pending.pop(0)
                    self._at_gate.discard(job.id)
                    self._holder = job.id
                    self._relabel_queued(loop)
                    try:
                        await self._run_holding(job, loop)
                    finally:
                        self._holder = None
                        self._wake_order()
                    return
        finally:
            self._at_gate.discard(job.id)

    async def _run_holding(self, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        """The job's turn, the gate held."""
        if job.cancel_requested:
            self._finish(loop, job, "cancelled")
            return
        job.status = "running"
        self._wait_on(job, None)
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
                    f"Stopped at {job.unit} {job.run} of {job.total_runs} after {job.units(job.failures)} "
                    f"failed; {kept} image{'' if kept == 1 else 's'} kept. Last: {error}"
                )
            self._finish(loop, job, "error", error)
            return
        # Cancelled only if the loop actually stopped early. A cancel that
        # lands after the last step - during the decode or the encode -
        # changes nothing, and used to relabel a complete run anyway.
        if job.failures and not stopped:
            were = "was" if job.failures == 1 else "were"
            job.error = f"{job.units(job.failures)} failed and {were} skipped; last: {job.error}"
        self._finish(loop, job, "cancelled" if stopped else "done")

    # A cell that raises is skipped and the next one runs - up to this many in a
    # row, or this share of the grid in total, after which the failure is taken
    # as the night's and the job ends. One bad cell should cost one image; a
    # marginal failure hitting every third cell should not limp until morning.
    MAX_CONSECUTIVE_FAILURES = 3

    @staticmethod
    def _too_many_failures(consecutive: int, total: int, cells: int) -> bool:
        return consecutive >= JobManager.MAX_CONSECUTIVE_FAILURES or total > max(3, cells // 3)

    def _recheck_edit(self, request: GenerateRequest) -> None:
        """An edit cell's checks as it begins, before the backend is called.

        The gate and the measured envelope are the backend's answer and the
        selection now, not at submit; the source is proved again against the
        identity the cell carries, and its aspect must be a measured one.
        Raises a ValueError in words that fails this cell alone.
        """
        from . import instruction_edit_admission
        from .instruction_edit import StoredInstructionEdit, geometry_refusal, require_execution_available

        spec = request.validate_edit(require_bound=True)
        if not isinstance(spec, StoredInstructionEdit):
            raise ValueError("Bind and verify the instruction-edit source before running it.")
        require_execution_available(spec, qualified=self.backend.instruction_edit_qualified())
        reason = self.backend.instruction_edit_refusal(current_settings(settings))
        if reason is not None:
            raise ValueError(reason)
        try:
            _, resolved = instruction_edit_admission.bind_source(spec)
        except ValueError as exc:
            raise ValueError(
                f"the edit's source #{spec.source} is missing, in the recycle bin or "
                "changed since it was queued; submit the edit again"
            ) from exc
        reason = geometry_refusal(resolved)
        if reason is not None:
            raise ValueError(reason)

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
        for cell in [*job.cells, *job.source_passes]:
            self.backend.check_refine_supported(cell.request)
        # And a stand-in of the passes Then refine will run: a profile's guess
        # a load contradicts fails here, before the first base image. A job
        # of missing passes alone has checked its passes themselves above.
        if job.then_refine is not None and job.cells:
            self.backend.check_refine_supported(stand_in(job.cells[0].request, job.then_refine))

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
                    **({"execution_diagnostics": asdict(value)}
                       if (value := job.execution_diagnostics()) is not None else {}),
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
        # Then refine's passes in a row that failed, apart from the cells':
        # a base made between them does not say the pass size will work.
        pass_streak = 0
        # Finish run's missing passes first, each a unit of its own: pause,
        # the budget and cancel act at its boundary as at a cell's, and its
        # failures count under the pass streak.
        for position, source in enumerate(job.pass_sources):
            if self._stops_before(job, loop, position):
                return True
            job.run, job.step, job.label = position + 1, 0, f"pass of #{source}"
            try:
                cancelled, pass_streak = self._source_pass(
                    job, loop, source, pass_streak, on_progress, should_cancel, on_preview,
                )
            finally:
                job.source_passes = [c for c in job.source_passes if c.request.refine != source]
            if cancelled:
                return True
        offset = len(job.pass_sources)
        for index, cell in enumerate(job.cells):
            if self._stops_before(job, loop, offset + index):
                return True
            req, seed = cell.request, cell.seed
            # A grid varies the step count between cells, so the totals the
            # progress events carry are the cell's, not the job's - and on a
            # refine cell the part of the schedule its strength leaves, until
            # the first callback says what the pipeline made of that. Without
            # the images: they were each delivered as their own event already.
            job.run, job.step, job.label = offset + index + 1, 0, cell.label
            job.total_steps = progress_steps(req)
            call = job.begin_call()
            log.info(
                "job %s: cell %d/%d seed %d%s",
                job.id, offset + index + 1, job.total_runs, seed, f"  {cell.label}" if cell.label else "",
            )
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})

            started = time.perf_counter()
            try:
                # An inpaint's mask is a file beside the rows, not one: one
                # lost since the submit fails this cell in words as it
                # begins, before any backend reads it.
                if req.refine is not None and req.mask is not None:
                    try:
                        storage.mask_path(req.mask)
                    except ValueError as exc:
                        if req.outpaint is None:
                            raise
                        raise ValueError(
                            "the outpaint's mask has gone from this gallery; submit it again"
                        ) from exc
                # An instruction edit asks again, as it begins, whether the
                # backend is still qualified and its source still the pixels
                # it was queued with: a source binned, deleted or rewritten
                # since the submit fails this cell in words.
                if req.edit is not None:
                    self._recheck_edit(req)
                try:
                    try:
                        reader = self.backend.execution_diagnostics_reader()
                    except Exception:  # noqa: BLE001 - optional cold observation
                        reader = None
                    if job.bind_execution_diagnostics(reader):
                        self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
                    images = self.backend.generate(
                        req,
                        seed=seed,
                        on_progress=on_progress,
                        should_cancel=lambda call=call: should_cancel() or job.skips_call(call),
                        on_preview=on_preview,
                    )
                finally:
                    job.freeze_execution_diagnostics()
                    # Saving is not an interruptible cell call. A late Skip
                    # refuses here instead of poisoning the following cell.
                    job.end_call(call)
                    self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                records = self._save(job, loop, cell, images, elapsed_ms, self.backend.last_peak())
                if len(images) == req.num_images:
                    job.completed_indices.append(index)
            except Cancelled:
                if not job.cancel_requested and job.skips_call(call):
                    job.skipped_indices.append(index)
                    job.skipped_cells.append({"label": cell.label, "seed": seed, "error": "Skipped by user"})
                    consecutive = 0
                    log.info("job %s skipped cell %d/%d", job.id, index + 1, len(job.cells))
                    self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
                    continue
                log.info(
                    "job %s cancelled during run %d/%d", job.id, index + 1, len(job.cells)
                )
                return True
            except Exception as exc:  # noqa: BLE001 - a cell's failure, not the night's
                consecutive += 1
                job.failures += 1
                job.failed_indices.append(index)
                job.error = short_error(exc)
                job.failed_cells.append({"label": cell.label, "seed": seed, "error": job.error})
                log.error(
                    "job %s: cell %d/%d failed: %s", job.id, index + 1, len(job.cells), exc
                )
                log.debug("cell traceback", exc_info=True)
                if self._too_many_failures(consecutive, job.failures, self._units(job)):
                    job.stopped_by = "failures"
                    raise
                continue
            finally:
                job.end_call(call, clear_skip=True)
            consecutive = 0
            if job.then_refine is not None:
                cancelled, pass_streak = self._passes(
                    job, loop, records, pass_streak, on_progress, should_cancel, on_preview,
                )
                if cancelled:
                    return True
        return False

    @staticmethod
    def _units(job: Job) -> int:
        """What the failure rule's share is of: the cells, under Then refine
        every image's pass beside them, and Finish run's missing passes."""
        if job.then_refine is None:
            return len(job.cells)
        return len(job.cells) + sum(cell.request.num_images for cell in job.cells) + len(job.pass_sources)

    def _stops_before(self, job: Job, loop: asyncio.AbstractEventLoop, position: int) -> bool:
        """A unit's boundary, `position` its zero-based place in the job:
        True where the job ends here - a cancel, or the budget - after any
        pause has held."""
        if job.cancel_requested:
            return True
        if job.pause_requested:
            self._hold(job, loop, position)
            if job.cancel_requested:
                return True
        # The budget means "start no cell after": a cell in flight runs to
        # its end, and what was not reached is a resume, not a loss.
        if job.deadline is not None and time.time() > job.deadline:
            job.stopped_by = "budget"
            log.info("job %s: time budget reached before unit %d/%d", job.id, position + 1, job.total_runs)
            return True
        return False

    def _pass_failed(self, job: Job, label: str, seed: int | None, exc: Exception, streak: int) -> int:
        """A Then refine pass's failure: reported as `label`, counted in
        `failures` and the pass streak, never in `failed_indices` (they index
        the cells). Raises `exc` where the failure rule ends the job; else
        the streak."""
        streak += 1
        job.failures += 1
        job.error = short_error(exc)
        job.failed_cells.append({"label": label, "seed": seed, "error": job.error})
        log.error("job %s: %s failed: %s", job.id, label, exc)
        log.debug("pass traceback", exc_info=exc)
        if self._too_many_failures(streak, job.failures, self._units(job)):
            job.stopped_by = "failures"
            raise exc
        return streak

    def _source_pass(self, job, loop, source, streak, on_progress, should_cancel, on_preview):
        """One of Finish run's missing passes: its base re-read by id at its
        turn and passed through the stage every base's passes run through.
        A base binned or deleted since the check is a failed pass in those
        words, never a crash. (cancelled, streak)."""
        label = f"pass of #{source}"
        record = storage.get_image(source, include_binned=True)
        if record is None or record.binned_at is not None:
            where = "deleted" if record is None else "moved to the recycle bin"
            planned = next((c for c in job.source_passes if c.request.refine == source), None)
            streak = self._pass_failed(
                job, label, planned.seed if planned is not None else None,
                ValueError(f"#{source} was {where} after Finish run was checked; its pass was not made"), streak,
            )
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
            return False, streak
        return self._passes(job, loop, [record], streak, on_progress, should_cancel, on_preview)

    @staticmethod
    def _plan_source_passes(sources: list[int], spec: ThenRefine) -> list[Cell]:
        """Finish run's missing passes as cells, read from the gallery at
        submit: what `planned_keys` counts until each unit runs. A source
        gone since the check plans nothing here; its unit reports it."""
        from .refine_batch import QUEUED

        records = storage.get_images(sources)
        return [p.cell for p in plan_passes(records, spec) if p.status == QUEUED]

    def _passes(self, job, loop, records, streak, on_progress, should_cancel, on_preview):
        """Then refine's stage: a pass of each record a cell just saved, run
        before the next cell, so a base and its passes are one unit - pause
        and the budget wait for both; a cancel stops inside a pass, the base
        kept. Each pass is Mass refine's of that record (`refine_batch.plan`)
        and saved under the pass run. A failed pass keeps its base and counts
        under the cells' failure rule - three passes in a row (`streak`,
        which a base made between them does not reset) or the share - but is
        no failed cell: the base cell did complete, and Finish reads
        `failed_indices` as cells. (cancelled, streak)."""
        from .refine_batch import QUEUED

        planned = plan_passes(records, job.then_refine)
        job.pending_passes = [p.cell for p in planned if p.status == QUEUED]
        try:
            for outcome in planned:
                label = f"pass of #{outcome.record.id}"
                if job.cancel_requested:
                    return True, streak
                try:
                    if outcome.status != QUEUED:
                        raise ValueError(outcome.detail or outcome.status)
                    cell = outcome.cell
                    job.stage, job.step, job.label = "pass", 0, label
                    job.total_steps = progress_steps(cell.request)
                    self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
                    started = time.perf_counter()
                    images = self.backend.generate(
                        cell.request, seed=cell.seed, on_progress=on_progress,
                        should_cancel=should_cancel, on_preview=on_preview,
                    )
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    self._save(
                        job, loop, cell, images, elapsed_ms, self.backend.last_peak(), run_id=job.pass_run_id,
                    )
                    job.completed_passes.append(outcome.record.id)
                except Cancelled:
                    log.info("job %s cancelled during %s", job.id, label)
                    return True, streak
                except Exception as exc:  # noqa: BLE001 - a pass's failure, not the night's
                    streak = self._pass_failed(job, label, outcome.record.seed, exc, streak)
                    continue
                finally:
                    if outcome.cell in job.pending_passes:
                        job.pending_passes.remove(outcome.cell)
                streak = 0
            return False, streak
        finally:
            job.pending_passes = []
            job.stage = None

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
        job.paused_at = time.time()
        job.paused = True
        try:
            self.backend.free_cache()
        except Exception:  # noqa: BLE001 - a pause must not fail the night
            log.debug("free_cache failed", exc_info=True)
        log.info("job %s paused before unit %d/%d", job.id, index + 1, job.total_runs)
        self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
        loop.call_soon_threadsafe(self._relabel_queued, loop)
        held_from = time.monotonic()
        while job.pause_requested and not job.cancel_requested:
            time.sleep(0.25)
        # A time budget is working time: the pause moves the deadline on by
        # as long as it held. Left where it was, a grid paused past its
        # budget ended at the Resume with no cell run, "Time budget reached".
        # Before the resume's frame, which carries the new deadline to the
        # page's "left (budget)".
        if job.deadline is not None:
            job.deadline += time.monotonic() - held_from
        job.paused = False
        job.paused_at = None
        if not job.cancel_requested:
            log.info("job %s resumed at unit %d/%d", job.id, index + 1, job.total_runs)
            self._emit(loop, job, {"type": "status", **job.snapshot(images=False)})
        loop.call_soon_threadsafe(self._relabel_queued, loop)

    def _save(
        self,
        job: Job,
        loop: asyncio.AbstractEventLoop,
        cell: Cell,
        images: list,
        elapsed_ms: int,
        peak: tuple[int, int] | None = None,
        run_id: str | None = None,
    ) -> list:
        """Write a cell's images and publish each one as it lands; the
        records written. `run_id` is the run they join, the job's by default
        (Then refine's passes join theirs).

        `peak` is the call's, so every image of the batch gets it whole: a
        maximum does not divide as the duration does."""
        # Record what ran, not what was asked: a negative prompt the pipeline
        # dropped below its threshold, or a mu it had no way to take, must not
        # reach the row - resume compares against it.
        ran = self.backend.applied(cell.request)
        selected = current_settings(settings)
        seed = cell.seed
        written = []
        for offset, image in enumerate(images):
            # Each image in a batch gets its own reproducible seed. Wrapped
            # for the same reason the run seed is - and with the same helper
            # the generator uses, so the two cannot disagree.
            while True:
                try:
                    record = storage.save_image(
                        image,
                        ran,
                        seed=wrap_seed(seed + offset),
                        duration_ms=elapsed_ms // max(len(images), 1),
                        model_id=selected.model_id,
                        job_id=job.id,
                        run_id=run_id or job.run_id,
                        quantization=selected.quantization,
                        variables=cell.variables,
                        batch=cell.request.num_images,
                        peak=peak,
                    )
                    break
                except gallery_lock.GalleryBusy:
                    # Another process has held the gallery past one
                    # SESSION_WAIT (a CLI reindex, another server's export).
                    # The render is in memory: wait again rather than lose
                    # it, until a cancel says otherwise.
                    if job.cancel_requested:
                        raise Cancelled() from None
                    log.warning("job %s: gallery busy in another process; still waiting to save", job.id)
            job.images.append(record.model_dump())
            written.append(record)
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
        return written

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
