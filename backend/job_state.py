"""A queued job's state: the `Job` record and its status snapshot.

Kept apart from `jobs.JobManager`, which runs it: nothing here reads the
gallery, the settings or the backend, and `jobs` re-exports both names.
"""
from __future__ import annotations

import asyncio
import functools
import threading
import time
from dataclasses import asdict, dataclass, field
from types import MethodType
from typing import Any

from .backends.base import ExecutionDiagnosticsReader
from .backends.regional_compile import RegionalDiagnosticOwner, RegionalDiagnostics
from .grid import Cell
from .schemas import GenerateRequest, ThenRefine

_TERMINAL = {"done", "error", "cancelled"}


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
    # Behind a paused job, when it paused (Unix seconds; None otherwise): the
    # waiting tab says how long the job ahead has held the gate.
    ahead_paused_at: float | None = None
    step: int = 0
    total_steps: int = 0
    run: int = 0
    total_runs: int = 1
    # What the current cell varies, for a grid; None for an ordinary sweep.
    label: str | None = None
    images: list[dict] = field(default_factory=list)
    # The ids of those images gone from the gallery since - deleted, or in
    # the recycle bin until a restore (`JobManager.forget_images`,
    # `restore_images`). `images` keeps them - `image_count` is how much the job produced, which
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
    # Zero-based positions, independent of a label or seed that can repeat.
    # Retained outcomes use these to identify failed calls after a restart.
    failed_indices: list[int] = field(default_factory=list)
    completed_indices: list[int] = field(default_factory=list)
    # Intentional skips are missing cells, not backend failures. Keep their
    # indices in the terminal outcome without invoking the failure stop rule.
    skipped_cells: list[dict] = field(default_factory=list)
    skipped_indices: list[int] = field(default_factory=list)
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
    # Swap tags' rows ("red hair → silver hair"), which name its job as a
    # grid's prompt names a grid's: its cells' prompts are its records'.
    swap: str | None = None
    # Mass refine's scale: its cells are passes of as many sources, so the
    # job is named by the scale and the strength they share, not by the
    # first cell's source as a single pass is.
    refine_batch: float | None = None
    # More seeds' count: its cells continue batches the gallery already
    # holds, so the job is named as more of its prompt, not as a new one.
    more_seeds: int | None = None
    # Then refine: a pass of each image right after its cell, the strength
    # resolved, saved under `pass_run_id`. `stage` is "pass" while one runs,
    # and `pending_passes` the cell's passes still to run - planned, so More
    # seeds and Mass refine count them as made.
    then_refine: ThenRefine | None = None
    pass_run_id: str | None = None
    stage: str | None = None
    pending_passes: list[Cell] = field(default_factory=list)
    # Finish run's missing passes: the gallery ids of bases saved without
    # their pass (a cancel or a failure between the stages). Each is a unit
    # of its own, run before the first base cell; `source_passes` are their
    # pass cells as planned at submit, so `planned_keys` counts them from the
    # start, and each leaves the list once its unit has run.
    pass_sources: list[int] = field(default_factory=list)
    source_passes: list[Cell] = field(default_factory=list)
    # The bases whose Then refine pass this job saved, inline or as one of
    # `pass_sources`: in the retained outcome, so a pass permanently
    # deleted later is one Finish run says it will redraw.
    completed_passes: list[int] = field(default_factory=list)
    # One queue per attached stream. A single shared queue is destructive: a
    # reconnecting client that overlaps the stream it replaces splits the events
    # between the two readers, and only one of them ever sees `done`.
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    # Set from the event loop, read from the worker thread between steps.
    cancel_requested: bool = False
    # An exact call token prevents a slow Skip POST from stopping the next
    # cell. The loop requests it; the worker starts/ends calls under this lock.
    _call_lock: Any = field(default_factory=threading.Lock, repr=False)
    _call_serial: int = 0
    _current_call: int | None = None
    _skip_call: int | None = None
    # Pause is cooperative like cancel: the endpoint asks, the worker answers
    # at the next cell boundary by setting `paused` - a worker-written
    # acknowledgement, so the UI cannot believe the card is free while a
    # denoise is still running on it.
    pause_requested: bool = False
    paused: bool = False
    # When the worker took the pause (Unix seconds), None while it is not
    # paused: a pause keeps the gate for as long as it lasts, and "⏸ Paused"
    # said nothing of whether that was a minute or the night.
    paused_at: float | None = None
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
    # Seconds the job's cells would take by the gallery's matching history,
    # priced once at submission (`JobManager._price`), or None where any cell
    # has none. Presentation only: never repriced, never a reproduction field
    # - not in `CellKey`, `run_id`, `applied()`, a row, a PNG or a bundle -
    # and not retained, so a restarted server has none.
    estimate_seconds: float | None = None
    # Whether `storage.hold_masks` holds masks for this job, which `_finish`
    # then releases.
    holds_masks: bool = False
    # Runtime observation only: never a request/definition/reproduction field.
    # The active reader is bound to pure state, not a backend or registration.
    _execution_reader: ExecutionDiagnosticsReader | None = field(default=None, repr=False)
    _execution_diagnostics: RegionalDiagnostics | None = field(default=None, repr=False)

    def bind_execution_diagnostics(self, reader: ExecutionDiagnosticsReader | None) -> bool:
        if (not isinstance(reader, MethodType) or type(reader.__self__) is not RegionalDiagnosticOwner
                or reader.__func__ is not RegionalDiagnosticOwner.snapshot):
            reader = None
        with self._call_lock:
            self._execution_reader, self._execution_diagnostics = reader, None
        return reader is not None

    @staticmethod
    def _read_execution_diagnostics(reader: ExecutionDiagnosticsReader) -> RegionalDiagnostics | None:
        try:
            value = reader()
        except Exception:  # noqa: BLE001 - an optional observation never fails an image
            return None
        return value if type(value) is RegionalDiagnostics else None

    def execution_diagnostics(self) -> RegionalDiagnostics | None:
        with self._call_lock:
            reader, frozen = self._execution_reader, self._execution_diagnostics
        if reader is None:
            return frozen
        value = self._read_execution_diagnostics(reader)
        with self._call_lock:
            # A late read cannot overwrite/follow another cell's reader.
            return value if reader is self._execution_reader else self._execution_diagnostics

    def freeze_execution_diagnostics(self) -> None:
        with self._call_lock:
            reader = self._execution_reader
        if reader is None:
            return
        value = self._read_execution_diagnostics(reader)
        with self._call_lock:
            if reader is self._execution_reader:
                self._execution_diagnostics, self._execution_reader = value, None

    def begin_call(self) -> int:
        with self._call_lock:
            self._call_serial += 1
            self._current_call = self._call_serial
            self._skip_call = None
            return self._current_call

    def end_call(self, call: int, *, clear_skip: bool = False) -> None:
        with self._call_lock:
            if self._current_call == call:
                self._current_call = None
            if clear_skip and self._skip_call == call:
                self._skip_call = None

    def skips_call(self, call: int) -> bool:
        with self._call_lock:
            return self._skip_call == call

    def skip_state(self) -> tuple[int | None, bool, str]:
        with self._call_lock:
            call, skipping = self._current_call, self._skip_call == self._current_call
            skipping = skipping and call is not None
        reason = (
            "This job has finished" if self.status in _TERMINAL
            else "This job is stopping" if self.cancel_requested
            else "No cell is running while this job is queued" if self.status == "queued"
            else "Resume this job before skipping a cell" if self.paused
            else "Then refine passes cannot be skipped: Finish run adds a pass a stop left missing"
            if self.stage == "pass"
            else "No cell is running between calls" if call is None
            else "Skipping this cell after the current step" if skipping
            else ""
        )
        return call, skipping, reason

    @functools.cached_property
    def one_source(self) -> tuple[int | None, str | None, Any]:
        """The refine source, mask and outpaint every cell shares, else
        (None, None, None): what names the job "refine of #N", "inpaint of
        #N" or "outpaint of #N". A job's cells are fixed when it is made; a
        batch refine is named by its scale, and a job of Finish run's
        missing passes alone by its Then refine."""
        if self.refine_batch is not None or not self.cells:
            return None, None, None
        shared = {(cell.request.refine, cell.request.mask, cell.request.outpaint) for cell in self.cells} or {
            (self.request.refine, self.request.mask, self.request.outpaint)
        }
        return shared.pop() if len(shared) == 1 else (None, None, None)

    @property
    def unit(self) -> str:
        """What the job's units are called, as the page's `unitWord` says:
        a grid's cells; a plain job's images, or batches where each repeat
        makes several. "cell" is a grid's word - a plain job of three
        failing images said "after 3 cells failed"."""
        if self.grid:
            return "cell"
        if sum(cell.request.num_images for cell in self.cells) > len(self.cells):
            return "batch"
        return "image"

    @functools.cached_property
    def canvas_sizes(self) -> list[tuple[int, int]]:
        """Concrete canvases, fixed with the retained cells rather than their
        header; a job of missing passes alone, its passes'."""
        return sorted({(cell.request.width, cell.request.height) for cell in self.cells or self.source_passes})

    def units(self, count: int) -> str:
        """`count` of the job's units: "1 image", "3 batches"."""
        if count == 1:
            return f"1 {self.unit}"
        return f"{count} {'batches' if self.unit == 'batch' else self.unit + 's'}"

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
        current_call, skip_requested, skip_reason = self.skip_state()
        diagnostics = self.execution_diagnostics()
        frame = {
            "job_id": self.id,
            "run_id": self.run_id,
            "seed": self.seed,
            **({"execution_diagnostics": asdict(diagnostics)} if diagnostics is not None else {}),
            # What the job is, for a page that did not submit it - a reload,
            # the queue note naming each job waiting, the line over the bars:
            # the first cell's prompt and a grid's own, a pass's source, the
            # size, and how many images the whole job makes.
            "prompt": self.request.prompt[:120],
            "edit": None if self.request.edit is None else self.request.edit.model_dump(mode="json"),
            # Only where every cell shares them: a swap or re-render over many
            # sources is no "inpaint of #N" because its oldest record is one.
            "refine": self.one_source[0],
            # Beside it, an inpaint's mask: the page names the job "inpaint of #N".
            "mask": self.one_source[1],
            # And an outpaint's spec: "outpaint of #N +400/+0/+400/+0".
            "outpaint": None if self.one_source[2] is None else self.one_source[2].model_dump(),
            "refine_batch": self.refine_batch,
            "strength": self.request.strength if self.refine_batch is not None else None,
            "width": self.request.width,
            "height": self.request.height,
            "sizes": self.canvas_sizes,
            "size_count": len(self.canvas_sizes),
            # With Then refine every image has its pass after it, and Finish
            # run's missing passes come first, one image each.
            "planned_images": sum(cell.request.num_images for cell in self.cells)
            * (2 if self.then_refine is not None else 1) + len(self.pass_sources),
            "pass_sources": list(self.pass_sources),
            "then_refine": None if self.then_refine is None else self.then_refine.model_dump(),
            "pass_run_id": self.pass_run_id,
            "stage": self.stage,
            "grid": self.grid,
            "prompt_template": None if self.prompt_template is None else self.prompt_template[:120],
            "comparison": self.comparison,
            "swap": None if self.swap is None else self.swap[:120],
            "more_seeds": self.more_seeds,
            "status": self.status,
            "waiting_on": self.waiting_on,
            "ahead_paused_at": self.ahead_paused_at,
            "pause_requested": self.pause_requested,
            "paused": self.paused,
            "paused_at": self.paused_at,
            "step": self.step,
            "total_steps": self.total_steps,
            "run": self.run,
            "total_runs": self.total_runs,
            "label": self.label,
            "error": self.error,
            "failures": self.failures,
            "skipped": len(self.skipped_indices),
            "current_call": current_call,
            "skip_requested": skip_requested,
            "skip_reason": skip_reason,
            # The count travels even when the records do not: a listing of
            # finished jobs has to say how much a night produced without
            # carrying the night.
            "image_count": len(self.images),
            # How many different pictures those are. A grid whose axis the
            # pipeline ignored - a negative-prompt variable on klein - renders
            # the same image N times, and this is the number that says so.
            "distinct_count": len({image.get("content_hash") for image in self.images} - {None}),
            # How many of them have gone since - deleted, or in the recycle
            # bin - so a banner can say what is left of a night without
            # lowering what it produced.
            "deleted_count": len(self.deleted),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_progress_at": self.last_progress_at,
            "deadline": self.deadline,
            "stopped_by": self.stopped_by,
            # The stored price, read and never computed here: the snapshot
            # runs on the loop for every listing and status frame.
            "estimate_seconds": self.estimate_seconds,
        }
        if images:
            frame["images"] = [image for image in self.images if image["id"] not in self.deleted]
            frame["failed_cells"] = list(self.failed_cells)
            frame["skipped_cells"] = list(self.skipped_cells)
        return frame
