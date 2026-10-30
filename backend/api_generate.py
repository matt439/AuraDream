"""Single and grid submission: Generate, Run grid, its preview, and Re-render."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field, replace

from fastapi import APIRouter, HTTPException

from . import gallery_reports, grid, storage, submission, then_refine
from .api_more_seeds import more_seeds_plan
from .app_context import AppServices
from .execution import require_configured_loader_available, resolve_execution
from .model_selection import current_settings
from .sizes import nearest_preset
from .schemas import (
    GenerateRequest,
    GenerateSubmission,
    ImageRecord,
    GridPreview,
    GridRequest,
    JobAccepted,
    RerenderAccepted,
    RerenderGone,
    RerenderPreview,
    RerenderRequest,
    RerenderSize,
    RerenderUnmatched,
    MoreSeedsRequest,
    ThenRefine,
    error_detail,
)
from .schemas_rerun import (
    FinishRunAccepted, FinishRunPreview, FinishRunRequest, RunAction, RunDetails,
)


async def expand_grid(
    services: AppServices, request: GridRequest
) -> tuple[list[grid.Cell], dict[str, list], int, int, str]:
    """The grid as cells, its resolved axes, the numbers resume dropped as
    held by the gallery and as queued in a live job, its run id.

    The cells come back as spelled, duplicates included: `preview_grid`
    counts the cells that would render an identical image and says so, and
    the run renders what was asked. Dropping them here would leave a
    `{avoid}` axis on klein with one cell of three and a sheet that cannot
    say why - the warning is the honest half, the fix is the user's grid.

    The axes come back from here rather than being resolved again by the
    preview, so a prompt whose placeholders do not line up with its variables
    fails once, as a 422, instead of raising out of the response model. The run
    id is computed over the *whole* grid, before resume subtracts anything, so
    the job that continues a night is stamped with the same id as the job that
    started it. Prompt variants, when given, are the prompt axis; the preview
    then shows their text under `prompt` and their labels under `variant`.

    Resume drops the cells a live job already plans too, so the grid tick,
    the banner's Resume and a finish line's, pressed while the same run is
    still queued, render nothing twice; counted apart, since the preview's
    "already done" is the gallery's.
    """
    selected = current_settings(services.settings())
    try:
        require_configured_loader_available(selected)
        base = request.base(selected.loader_execution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
    if request.refine is not None:
        unavailable = (
            await submission.inpaint_unavailable(services, outpainting=request.outpaint is not None)
            if request.inpainting else await submission.refine_unavailable(services)
        )
        if unavailable is not None:
            raise HTTPException(status_code=422, detail=unavailable)
    try:
        # Listed seeds are an axis the preview shows like the others; counted
        # seeds are derived from the base and are not.
        seed_list = grid.parse_seeds(request.seed_axis)
        cells = grid.expand(
            base,
            prompt=request.prompt_axis,
            steps=request.steps_axis,
            cfg=request.cfg_axis,
            mu=request.mu_axis,
            lora_scale=request.lora_scale_axis,
            strength=request.strength_axis,
            seeds=request.seeds,
            seed_list=seed_list or None,
            variants=request.prompt_variants,
            prompt_mode=request.prompt_mode,
            size=request.size_axis,
        )
        # Random choices resolve once. A second expansion with an unset seed
        # would list another sample beside the cells whose counts we report.
        values = grid.axis_values(
            request, request.steps_axis, request.cfg_axis, request.mu_axis,
            request.prompt_axis, request.lora_scale_axis,
            variants=request.prompt_variants, strength=request.strength_axis,
            prompt_mode=request.prompt_mode, cells=cells,
            size=request.size_axis,
        )
        values["seed"] = list(seed_list)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
    # Sources and masks are held to each actual canvas; the base size may be
    # replaced completely by the paired size axis.
    for group in submission.cells_by_size(cells).values():
        await submission.refuse_unusable_inputs(services, group[0].request)
    run_id = grid.run_id(cells, services.model_id(), services.quantization())
    if not request.resume:
        return cells, values, 0, 0, run_id
    # The queue first, then the gallery: see `not_yet_planned`.
    planned = services.jobs().planned_keys()
    unheld = await submission.not_yet_rendered(services, cells, request.width, request.height)
    kept = submission.not_yet_planned(services, unheld, planned)
    return kept, values, len(cells) - len(unheld), len(unheld) - len(kept), run_id


def foreign_refusal(
    services: AppServices, count: int, foreign: list[str], picked: list[int] | None = None
) -> str:
    """Re-render's refusal of a selection that reaches other checkpoints, with the way out.

    Unticking All models is the exact way: the gallery's scope is the running
    id itself (`AppServices.model_scope`). The search box's `model:` is a substring, so
    it is offered only where the running id, whole, is in none of the other
    ids selected - and only then does adding it leave them out. The id's tail
    was offered, and `model:Anima` is Animagine too; nor is the whole id safe
    everywhere (`Qwen/Qwen-Image` is in `Qwen/Qwen-Image-Edit`). Picked cards
    are no filter's - neither way out reaches them - so for those (`picked`,
    the foreign ids) the way out is unpicking them, named as the page names
    images, the first six.
    """
    running = services.model_id()
    lead = (
        f"{count} of the selected images {'was' if count == 1 else 'were'} made on "
        f"{', '.join(foreign)}; the server is running {running}, whose steps, CFG and mu "
        f"differ."
    )
    if picked is not None:
        names = [f"#{image_id}" for image_id in picked[:6]]
        if len(picked) > 6:
            names.append(f"{len(picked) - 6} more")
        listed = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]
        return f"{lead} Unpick {listed}."
    narrows = not any(ch.isspace() for ch in running) and not any(
        running.lower() in other.lower() for other in foreign
    )
    search = f", or add model:{running} to the search" if narrows else ""
    return f"{lead} Untick All models{search}."


@dataclass
class FinishPlan:
    preview: FinishRunPreview
    definition: dict | None
    cells: list[grid.Cell]
    records: list
    sizes: list[tuple[int, int]]
    refusal_code: int = 409
    # Then refine's missing passes, one per base saved without its own,
    # and the one spec the run's retained jobs agree on.
    passes: list[grid.Cell] = field(default_factory=list)
    spec: ThenRefine | None = None


@dataclass
class _RerenderGroup:
    """One size of a re-render: its cells, those it would queue now, and
    the run id of the job it makes."""

    width: int
    height: int
    cells: list[grid.Cell]
    kept: list[grid.Cell]
    run_id: str


@dataclass
class _RerenderPlan:
    ids: list[int]
    deleted: list[RerenderGone]
    groups: list[_RerenderGroup]
    unmatched: list[RerenderUnmatched]
    pruned: int
    planned: int


def _finish_key(services: AppServices, cell: grid.Cell) -> tuple:
    ran = grid.Cell(request=services.backend().applied(cell.request), seed=cell.seed)
    return cell.request.width, cell.request.height, tuple(grid.cell_keys(ran))


async def then_refine_spec(
    services: AppServices, spec: ThenRefine | None, requests: list[GenerateRequest],
) -> ThenRefine | None:
    """Then refine as the job runs it, its strength resolved, or a 422 in
    words where these cells cannot each be followed by a pass."""
    if spec is None or not requests:
        return None
    spec = then_refine.resolved(spec)
    reason = then_refine.refusal(requests, spec, await submission.refine_unavailable(services))
    if reason is not None:
        raise HTTPException(status_code=422, detail=reason)
    return spec


def _retained_then_refine(metadata: dict) -> ThenRefine | None:
    """A retained job's Then refine, without the pass run id beside it."""
    saved = metadata.get("then_refine")
    if not isinstance(saved, dict):
        return None
    return ThenRefine.model_validate({k: v for k, v in saved.items() if k != "pass_run_id"})


async def finish_plan(services: AppServices, run_id: str) -> FinishPlan:
    """Resolve retained cells without the composer, preserving stored order.

    Snapshot the queue before any gallery read. Submit holds rerun_admission
    across this complete plan and submit_cells, including its disk awaits.
    """
    planned = services.jobs().planned_keys()
    try:
        definitions, records = await asyncio.gather(
            asyncio.to_thread(storage.retained_run_jobs, run_id),
            asyncio.to_thread(storage.lineage_images, [run_id], []),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not definitions and not records:
        raise HTTPException(status_code=404, detail="unknown run")

    definition = definitions[0] if definitions else None
    if any(cell.request.edit is not None for saved in definitions for cell in saved["cells"]) or any(
        record.edit is not None for record in records
    ):
        # Imported/retained edits remain inspectable, but no applied() call
        # may erase their pass kind or turn an unavailable editor into a 500.
        # Finish stays held for edits: More seeds is their replay, with
        # each new seed's source proved again.
        from .instruction_edit import INSTRUCTION_EDIT_REPLAY_HELD, INSTRUCTION_EDIT_UNAVAILABLE, replay_held_reason

        # The same split as `submission.unusable_input`: a closed gate says
        # the editor is unavailable, an unmeasured loader says so, else where
        # edits are queued from.
        reason = replay_held_reason(
            services.backend().instruction_edit_refusal(current_settings(services.settings()))
        )
        pinned = {INSTRUCTION_EDIT_UNAVAILABLE: "unavailable", INSTRUCTION_EDIT_REPLAY_HELD: "replay-held"}
        saved_cells = [cell for saved in definitions for cell in saved["cells"]]
        sizes = sorted({(cell.request.width, cell.request.height) for cell in saved_cells}) or sorted(
            {(record.width, record.height) for record in records}
        )
        preview = FinishRunPreview(
            run_id=run_id, available=False, reason=reason,
            total=len(saved_cells), cells=0, images=0, rendered=0, queued=0, redrawn=0,
            budget_minutes=definition["metadata"].get("budget_minutes") if definition else None,
            pin=hashlib.sha256(
                f"{run_id}:instruction-edit-{pinned.get(reason, 'loader-unqualified')}".encode()
            ).hexdigest()[:16],
        )
        return FinishPlan(preview, definition, [], records, sizes, 422)
    # Retained cells are complete replay, including old payloads written
    # before execution existed. Keep the portable decoder selection-free.
    selected = current_settings(services.settings())
    try:
        require_configured_loader_available(selected)
        for saved in definitions:
            saved["cells"] = [replace(cell, request=cell.request.model_copy(update={
                "execution": resolve_execution(selected.loader_execution, cell.request.execution),
            })) for cell in saved["cells"]]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
    saved_job_ids = {saved["job_id"] for saved in definitions}
    cells: list[grid.Cell] = []
    keys: set[tuple] = set()
    succeeded: set[tuple] = set()
    for saved in definitions:
        for cell in saved["cells"]:
            key = _finish_key(services, cell)
            if key not in keys:
                keys.add(key)
                cells.append(cell)
        for index in (saved.get("outcome") or {}).get("completed_indices", []):
            if isinstance(index, int) and 0 <= index < len(saved["cells"]):
                succeeded.add(_finish_key(services, saved["cells"][index]))

    budget = definition["metadata"].get("budget_minutes") if definition else None
    reason = ""
    if definition is None:
        reason = "made before runs kept their cells: use the panel's Resume tick"
        # A Then refine pass run keeps no cells of its own: its records' job
        # is the base job, whose definition names it.
        base = await asyncio.to_thread(
            storage.retained_run_metadata, sorted({r.job_id for r in records if r.job_id}),
        )
        for base_run, metadata in base.values():
            if (metadata.get("then_refine") or {}).get("pass_run_id") == run_id:
                reason = f"the refine passes of run {base_run}: finish that run, and its passes follow"
                break
    elif any(saved["model_id"] != services.model_id() for saved in definitions):
        reason = f"made on {definition['model_id']}; the server is running {services.model_id()}"
    elif any(saved["quantization"] != services.quantization() for saved in definitions):
        reason = f"made under {definition['quantization']}; every cell would render again under {services.quantization()}"
    elif not cells:
        reason = "this run has no retained cells"

    # Then refine's spec, one per run: a Resume tick with other numbers made
    # another pass run, which no single Finish can continue both of.
    specs: list[ThenRefine] = []
    spec_jobs: set[str] = set()
    for saved in definitions:
        retained = _retained_then_refine(saved["metadata"])
        if retained is not None:
            spec_jobs.add(saved["job_id"])
            if retained not in specs:
                specs.append(retained)
    spec = specs[0] if specs else None
    if not reason and len(specs) > 1:
        made = [then_refine.words(each) for each in specs]
        listed = f"{', '.join(made[:-1])} and at {made[-1]}"
        reason = (
            f"this run's passes were made at {listed}; Mass refine over run:{run_id} "
            f"recovers {'either' if len(specs) == 2 else 'any of them'}"
        )

    missing: list[grid.Cell] = []
    kept: list[grid.Cell] = []
    kept_passes: list[grid.Cell] = []
    rendered = queued = passes_queued = 0
    if not reason:
        by_size: dict[tuple[int, int], list[grid.Cell]] = {}
        for cell in cells:
            by_size.setdefault((cell.request.width, cell.request.height), []).append(cell)
        missing_ids: set[int] = set()
        matched_any = False
        for (width, height), group in by_size.items():
            done = await submission.rendered_keys(services, group, width, height)
            for cell in group:
                image_keys = _finish_key(services, cell)[2]
                matched_any |= any(key in done for key in image_keys)
                if not all(key in done for key in image_keys):
                    missing_ids.add(id(cell))
        missing = [cell for cell in cells if id(cell) in missing_ids]
        kept = submission.not_yet_planned(services, missing, planned)
        rendered, queued = len(cells) - len(missing), len(missing) - len(kept)
        if spec is not None:
            # A base a Then refine job saved, live, whose pass is not in the
            # gallery (Mass refine's own check, so a pass it made counts) and
            # not queued: a cancel or a failure between the stages. The pass
            # is the one the worker makes of that base. A binned base is a
            # discard; a plain job's bases never had a pass to miss.
            bases = [
                record for record in records
                if record.job_id in spec_jobs and record.refine is None and record.edit is None
                and record.binned_at is None
            ]
            pass_cells = [p.cell for p in then_refine.plan_passes(bases, spec) if p.cell is not None]
            unheld = await submission.not_yet_rendered(services, pass_cells)
            kept_passes = submission.not_yet_planned(services, unheld, planned)
            passes_queued = len(unheld) - len(kept_passes)
        live = next((job for job in services.jobs().active() if job.run_id == run_id), None)
        if live is not None:
            reason = f"the rest is queued in job {live.id}"
        elif any(record.job_id in saved_job_ids for record in records) and not matched_any:
            # A first retained job may itself be Resume: older completed
            # prefix rows have no retained cells. They alone are not proof
            # that this job's still-missing suffix drifted across builds.
            reason = "this build no longer reproduces this run's cells"
        elif not kept and not kept_passes:
            reason = (
                "the missing cells and passes are already queued" if queued and passes_queued
                else "the missing cells are already queued" if queued
                else "the missing passes are already queued" if passes_queued
                else "nothing in this run is missing"
            )

    refusal_code = 409
    if not reason and spec is not None:
        unavailable = await submission.refine_unavailable(services)
        if unavailable is not None:
            reason, refusal_code = unavailable, 422
    if not reason:
        checked: dict[tuple, tuple[str, str] | None] = {}
        for cell in [*kept, *kept_passes]:
            request = cell.request
            try:
                request.check_budget()
                request.check_refine_steps()
            except ValueError as exc:
                reason, refusal_code = error_detail(exc), 422
                break
            inputs = (request.refine, request.mask, request.outpaint, tuple(request.reference),
                      request.lora, request.width, request.height, request.upscaler,
                      grid.edit_key(request.edit))
            if inputs not in checked:
                checked[inputs] = await submission.unusable_input(services, request)
            unusable = checked[inputs]
            if unusable is not None:
                reason, refusal_code = unusable[1], 422
                break

    pin = hashlib.sha256(json.dumps(
        [run_id, [_finish_key(services, cell) for cell in kept],
         [_finish_key(services, cell) for cell in kept_passes], budget],
        separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()[:16]
    preview = FinishRunPreview(
        run_id=run_id, available=not reason, reason=reason,
        total=len(cells), cells=len(kept), images=sum(cell.request.num_images for cell in kept),
        rendered=rendered, queued=queued,
        redrawn=sum(_finish_key(services, cell) in succeeded for cell in kept),
        passes=len(kept_passes), budget_minutes=budget, pin=pin,
    )
    sizes = sorted({(cell.request.width, cell.request.height) for cell in cells}) if definitions else sorted(
        {(record.width, record.height) for record in records}
    )
    return FinishPlan(preview, definition, kept, records, sizes, refusal_code, kept_passes, spec)


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.get("/api/runs/{run_id}", response_model=RunDetails)
    async def run_details(run_id: str) -> RunDetails:
        plan = await finish_plan(services, run_id)
        definition, records = plan.definition, plan.records
        first = records[0] if records else None
        live = [record for record in records if record.binned_at is None]
        if definition:
            made_on = definition["model_id"]
            quantization, width, height = definition["quantization"], definition["width"], definition["height"]
            batch, comparison = definition["batch"], definition["comparison"]
        else:
            assert first is not None, "a known run has a retained definition or gallery records"
            made_on = first.model_id
            quantization, width, height = first.quantization, first.width, first.height
            batch, comparison = first.batch or 1, False
        seed_reason = (
            f"made on {made_on}; the server is running {services.model_id()}"
            if made_on != services.model_id() else
            ("this run has no live images; restore binned images to add more seeds" if records
             else "this run has no live images to extend") if not live else ""
        )
        seed_count = 0
        if not seed_reason:
            try:
                _, results, batches = await more_seeds_plan(services, MoreSeedsRequest(run=run_id, count=1))
                seed_count = sum(batch.status == "queued" for batch in batches)
                if not seed_count:
                    seed_reason = next((result.detail for result in results if result.detail), "no batch can add more seeds")
            except HTTPException as exc:
                seed_reason = str(exc.detail)
        return RunDetails(
            run_id=run_id, model_id=made_on,
            quantization=quantization,
            width=width,
            height=height,
            sizes=plan.sizes, size_count=len(plan.sizes),
            batch=batch,
            comparison=comparison,
            images=len(live), binned=len(records) - len(live), cells=plan.preview.total,
            actions={
                "finish": RunAction(available=plan.preview.available, reason=plan.preview.reason, count=plan.preview.cells),
                "seeds": RunAction(available=not seed_reason, reason=seed_reason, count=seed_count),
            },
        )

    @router.post("/api/runs/{run_id}/finish/preview", response_model=FinishRunPreview)
    async def preview_finish(run_id: str) -> FinishRunPreview:
        return (await finish_plan(services, run_id)).preview

    @router.post("/api/runs/{run_id}/finish", response_model=FinishRunAccepted)
    async def finish_run(run_id: str, request: FinishRunRequest | None = None) -> FinishRunAccepted:
        async with services.jobs().rerun_admission:
            plan = await finish_plan(services, run_id)
            preview = plan.preview
            if request is not None and request.expected is not None and request.expected != preview.pin:
                raise HTTPException(status_code=409, detail="the run changed since the preview; check Finish run again")
            if not preview.available:
                raise HTTPException(status_code=plan.refusal_code, detail=preview.reason)
            definition = plan.definition
            assert definition is not None, "an available Finish run has its retained definition"
            metadata = definition["metadata"]
            try:
                job = await services.jobs().submit_cells(
                    plan.cells, run_id=run_id, budget_minutes=preview.budget_minutes,
                    grid=metadata.get("grid", True), prompt_template=metadata.get("prompt_template"),
                    comparison=definition["comparison"], swap=metadata.get("swap"),
                    refine_batch=metadata.get("refine_batch"), more_seeds=metadata.get("more_seeds"),
                    then_refine=plan.spec,
                    pass_sources=[cell.request.refine for cell in plan.passes if cell.request.refine is not None],
                )
            except ValueError as exc:
                if plan.cells or not plan.passes:
                    raise
                # Every base whose pass was missing went between the check
                # and the submit: nothing is queued.
                raise HTTPException(status_code=409, detail=f"{exc}; check Finish run again") from exc
            return FinishRunAccepted.model_validate({
                "job_id": job.id, "seed": job.seed, "run_id": run_id, "estimate_seconds": job.estimate_seconds,
                "cells": preview.cells, "images": preview.images,
                "pruned": preview.rendered, "planned": preview.queued, "redrawn": preview.redrawn,
                "passes": preview.passes, "budget_minutes": preview.budget_minutes,
            })

    @router.post("/api/generate", response_model=JobAccepted)
    async def generate(submitted: GenerateSubmission) -> JobAccepted:
        try:
            selected = current_settings(services.settings())
            require_configured_loader_available(selected)
            request = submitted.base(selected.loader_execution)
            request.check_budget()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await submission.refuse_unusable_inputs(services, request)
        try:
            request.check_refine_steps()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # The job waits on ensure_loaded() itself, so a previously failed load is
        # retried here rather than short-circuiting - the user may have just been
        # granted repo access.
        spec = await then_refine_spec(services, submitted.then_refine, [request])
        job = await services.jobs().submit(request, then_refine=spec)
        return JobAccepted.model_validate({
            "job_id": job.id, "seed": job.seed, "run_id": job.run_id, "estimate_seconds": job.estimate_seconds,
        })


    @router.post("/api/grid", response_model=JobAccepted)
    async def run_grid(request: GridRequest) -> JobAccepted:
        if request.resume:
            async with services.jobs().rerun_admission:
                return await submit_grid(request)
        return await submit_grid(request)


    async def submit_grid(request: GridRequest) -> JobAccepted:
        """Queue a sweep over the sampling axes as one job.

        One job rather than one per cell: a grid is a night's work, and it has to be
        cancellable, reattachable and ahead of nothing in the queue as a unit.
        """
        cells, _, _, queued, run_id = await expand_grid(services, request)
        if not cells:
            # Every cell is already in the gallery, or queued. Queueing an empty
            # job would report "done" having rendered nothing, which reads like
            # a failure.
            raise HTTPException(status_code=409, detail=(
                "every cell is already rendered or queued" if queued else "every cell is already rendered"
            ))
        spec = await then_refine_spec(services, request.then_refine, [cell.request for cell in cells])
        job = await services.jobs().submit_cells(
            cells,
            run_id=run_id,
            budget_minutes=request.budget_minutes,
            prompt_template=request.prompt,
            comparison=bool(request.prompt_variants),
            then_refine=spec,
        )
        return JobAccepted.model_validate({
            "job_id": job.id, "seed": job.seed, "run_id": job.run_id, "estimate_seconds": job.estimate_seconds,
        })


    @router.post("/api/rerender/preview", response_model=RerenderPreview)
    async def preview_rerender(request: RerenderRequest) -> RerenderPreview:
        """What Re-render would queue, size by size, without queueing it.

        No side effects, so it is not a model-dependent request in the
        middleware's sense, as Swap tags' preview is not: the queueing
        request is, and carries the revision. A selection Re-render would
        refuse is refused here in the same words.
        """
        plan = await rerender_plan(request)
        return RerenderPreview(
            ids=plan.ids,
            sizes=[
                RerenderSize(width=group.width, height=group.height, images=len(group.cells), queued=len(group.kept))
                for group in plan.groups
            ],
            unmatched=plan.unmatched, deleted=plan.deleted,
            queued=sum(len(group.kept) for group in plan.groups),
            jobs=sum(1 for group in plan.groups if group.kept),
            pruned=plan.pruned, planned=plan.planned,
        )


    @router.post("/api/rerender", response_model=RerenderAccepted)
    async def rerender(request: RerenderRequest) -> RerenderAccepted:
        """Re-render the selection, leaving out gallery and live-job cells."""
        async with services.jobs().rerun_admission:
            return await rerender_selection(request)


    async def rerender_plan(request: RerenderRequest) -> _RerenderPlan:
        """The selection as cells, one group per size, each with what it would
        queue now - or the refusal, in the words the queueing request says it.

        At one size (`width`/`height`) every record is one group at it; at a
        `tier`, each record goes to its own aspect's entry at that tier
        (`sizes.nearest_preset`), and a record whose size is no preset's
        aspect is answered in `unmatched`, never guessed or queued. Every
        refusal below holds per size: an inpaint is refused at any size but
        its own, a budget at the size that passes it.
        """
        if request.tier is None:
            assert request.width is not None and request.height is not None, "the request model holds a size"
            probe = GenerateRequest(prompt="x", width=request.width, height=request.height)
            try:
                probe.check_budget()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        records, gone, binned = await submission.resolve_selection(
            services, ids=request.ids, search=request.search, favorites=request.favorites,
            all_models=request.all_models,
        )
        deleted = [RerenderGone(id=image_id, detail=submission.gone_detail(image_id, binned)) for image_id in gone]
        if not records:
            said = "; ".join(f"#{item.id} is {item.detail}" for item in deleted[:6])
            more = f"; and {len(deleted) - 6} more" if len(deleted) > 6 else ""
            raise HTTPException(status_code=404, detail=f"none of the selected images is left: {said}{more}")
        # Steps, CFG and mu are checkpoint-specific - Turbo's 8 / 0 / 1.15 on FLUX
        # is a broken image recorded under FLUX - so a record from another
        # checkpoint is refused rather than rendered wrong. From a filter, only
        # reachable with `all_models`: the scope keeps them out of it otherwise;
        # picked cards are another checkpoint's wherever they were picked.
        foreign = sorted({r.model_id for r in records if r.model_id != services.model_id()})
        if foreign:
            others = [r.id for r in records if r.model_id != services.model_id()]
            raise HTTPException(status_code=422, detail=foreign_refusal(
                services, len(others), foreign, others if request.ids is not None else None,
            ))
        # An instruction edit's size is its source's frozen geometry: never
        # re-rendered at another, and at its own already there. Named by id
        # before any cell is built (an unresolved import would refuse in
        # the request model's words); More seeds is its way to new seeds.
        edits = [record for record in records if record.edit is not None]
        if edits:
            raise HTTPException(status_code=422, detail=submission.edit_rerender_detail(
                edits, picked=request.ids is not None, others=len(edits) < len(records),
            ))
        # The records by the size each renders at, in the resolver's order.
        sized: dict[tuple[int, int], list[ImageRecord]] = {}
        unmatched: list[RerenderUnmatched] = []
        if request.tier is None:
            assert request.width is not None and request.height is not None
            sized[(request.width, request.height)] = records
        else:
            for record in records:
                size = nearest_preset(record.width, record.height, request.tier)
                if size is None:
                    unmatched.append(RerenderUnmatched(id=record.id, width=record.width, height=record.height))
                else:
                    sized.setdefault(size, []).append(record)
            if not sized:
                said = "; ".join(f"#{item.id} is {item.width}x{item.height}" for item in unmatched[:6])
                more = f"; and {len(unmatched) - 6} more" if len(unmatched) > 6 else ""
                raise HTTPException(status_code=422, detail=(
                    f"none of the picked images is a preset's aspect, so none has a {request.tier:g} MP size: "
                    f"{said}{more}"
                ))
            # Each size its own budget: the one that passes it is refused
            # by name, and nothing is queued.
            for width, height in sized:
                try:
                    GenerateRequest(prompt="x", width=width, height=height).check_budget()
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
        matched = [record for group in sized.values() for record in group]
        # An inpaint whose mask this gallery lacks is named by its id, which
        # the per-combination check below could only say by the mask's hash.
        lost = await asyncio.to_thread(submission.missing_masks, matched)
        if lost:
            raise HTTPException(status_code=422, detail=submission.missing_mask_detail(lost))
        # And an outpaint whose spec does not build its own row's mask: run
        # again it would paint under another mask under the same key.
        mismatched = await asyncio.to_thread(submission.mismatched_outpaints, matched)
        if mismatched:
            raise HTTPException(status_code=422, detail=submission.mismatched_outpaint_detail(mismatched))
        # So is an inpaint at another size than its own - its source's, which
        # its mask covers - where the per-combination check would name the
        # source the user never picked.
        for (width, height), group in sized.items():
            resized = submission.resized_inpaints(group, width, height)
            if resized:
                raise HTTPException(status_code=422, detail=submission.resized_inpaint_detail(
                    resized, width, height, picked=request.ids is not None,
                    others=len(resized) < len(records),
                ))
        model, quantization = services.model_id(), services.quantization()
        groups: list[_RerenderGroup] = []
        for (width, height), group in sized.items():
            # Oldest first (the resolver's order), so the re-render lands in the
            # order the originals did; every record's reproduction set but its
            # size. A row the request models refuse (a spec another build or an
            # import let through) is a worded refusal, never a 500.
            try:
                cells = grid.cells_from_records(group, width=width, height=height)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
            # One check per distinct combination rather than per cell: a selection of
            # 400 images carries at most a handful of adapters and sources between them.
            # Every cell is at its group's one size, so an inpaint's mask is held
            # to it here: re-rendered elsewhere, its mask no longer covers it. An
            # outpaint (frozen, so it keys) is checked at that size per padding,
            # never sharing a plain refine's verdict over the same source.
            for spec in {
                (c.request.lora, tuple(c.request.reference), c.request.refine, c.request.mask, c.request.outpaint,
                 c.request.upscaler, c.request.edit)
                for c in cells
            }:
                if spec != (None, (), None, None, None, None, None):
                    await submission.refuse_unusable_inputs(services, cells[0].request.model_copy(
                        update={"lora": spec[0], "reference": list(spec[1]), "refine": spec[2], "mask": spec[3],
                                "outpaint": spec[4], "upscaler": spec[5], "edit": spec[6]}
                    ))
            try:
                for cell in cells:
                    cell.request.check_refine_steps()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            # One run id per size, each the shape every other run's has; at
            # one size, the whole selection's, as it always was.
            groups.append(_RerenderGroup(width, height, cells, cells, grid.run_id(cells, model, quantization)))
        # The queue first, then the gallery: a live cell can finish while
        # the scan runs. Admission stays held until submit_cells has added
        # these jobs, including their awaited mask writes.
        planned = services.jobs().planned_keys()
        pruned = already_planned = 0
        for group in groups:
            unheld = await submission.not_yet_rendered(services, group.cells, group.width, group.height)
            group.kept = submission.not_yet_planned(services, unheld, planned)
            pruned += len(group.cells) - len(unheld)
            already_planned += len(unheld) - len(group.kept)
        return _RerenderPlan(
            ids=[record.id for record in records], deleted=deleted, groups=groups, unmatched=unmatched,
            pruned=pruned, planned=already_planned,
        )


    async def rerender_selection(request: RerenderRequest) -> RerenderAccepted:
        """Queue the gallery's current selection again at another size: one
        job at one size, or one per size at a tier.

        The documented workflow ends at "star the winners"; getting them at full
        size then meant Reuse, retype the size, Generate, twenty times over - and
        the clicking is where a seed gets mistyped. Here every selected record is
        a cell with its own settings and seed, deduplicated against what the
        gallery already holds at the target size, so a re-render can itself be
        resumed. A sibling of `run_grid`, and it shares the run-id and resume
        machinery with it. The selection is the picked cards where `ids` names
        them, else the filter (`submission.resolve_selection`); a picked image
        gone since is left out and named in `deleted`, and with none left the
        answer is 404 naming them all. `expected` is the ids the preview
        resolved: a selection that changed since is refused (409). At a tier,
        one job per size, the first at the top level and the rest in `more`,
        as Swap tags answers.
        """
        plan = await rerender_plan(request)
        if request.expected is not None and sorted(set(request.expected)) != sorted(plan.ids):
            raise HTTPException(
                status_code=409, detail="the selection changed since the check; check Re-render again",
            )
        groups = [group for group in plan.groups if group.kept]
        if not groups:
            pruned, already_planned = plan.pruned, plan.planned
            state = "rendered or queued" if already_planned and pruned else "queued" if already_planned else "rendered"
            where = (
                f"at {request.width}x{request.height}" if request.tier is None
                else f"at its {request.tier:g} MP size"
            )
            raise HTTPException(status_code=409, detail=f"every selected image is already {state} {where}")
        accepted = []
        for group in groups:
            job = await services.jobs().submit_cells(group.kept, run_id=group.run_id)
            accepted.append(JobAccepted.model_validate({
                "job_id": job.id, "seed": job.seed, "run_id": job.run_id, "estimate_seconds": job.estimate_seconds,
            }))
        first = accepted[0]
        return RerenderAccepted.model_validate({
            **first.model_dump(),
            "deleted": plan.deleted,
            "pruned": plan.pruned, "planned": plan.planned,
            "more": accepted[1:], "unmatched": plan.unmatched,
        })


    @router.post("/api/grid/preview", response_model=GridPreview)
    async def preview_grid(request: GridRequest) -> GridPreview:
        """What that grid would do, without queueing it.

        Same expansion as the real thing, so the count and the error messages the
        panel shows are the ones the run would have produced.

        Without Resume, the cells the gallery already holds are counted too: Run
        grid again after a night with a failed cell renders the whole night again,
        and the count is what says Resume would not. Only where the seeds are
        fixed - an explicit base or a list - since a random base matches nothing.
        """
        cells, values, skipped, queued, _ = await expand_grid(services, request)
        in_gallery = 0
        if not request.resume and (request.seed is not None or request.seed_axis.strip()):
            in_gallery = len(cells) - len(
                await submission.not_yet_rendered(services, cells, request.width, request.height)
            )
        images = sum(cell.request.num_images for cell in cells)
        listed = len(values.get("seed", []))
        spec = await then_refine_spec(services, request.then_refine, [cell.request for cell in cells])
        priced = then_refine.execution_units(cells, spec)
        pass_sizes = []
        if spec is not None and cells:
            pass_sizes = sorted({
                (probe.width, probe.height) for probe in (then_refine.stand_in(cell.request, spec) for cell in cells)
            })
        prices, disk_bytes = await asyncio.to_thread(
            gallery_reports.execution_estimates, priced, services.model_id(),
        )
        # Under Resume the seeds left are no longer alike, and the page says
        # nothing of the budget there.
        budget_seeds = (
            grid.seeds_within_budget(
                priced, listed or request.seeds, None, request.budget_minutes * 60, by_size=prices,
            )
            if request.budget_minutes and not request.resume
            else None
        )
        # Where a refine pass starts on the schedule, for the Refine dialog: only
        # where every cell starts at one point - the same steps, mu and strength
        # as the pipeline will run them - which a sweep's do not. Schedulers
        # only, no weights, and None before a load.
        start_sigma = None
        if request.refine is not None and cells:
            applied = [services.backend().applied(cell.request) for cell in cells]
            if len({(r.width, r.height, r.steps, r.mu, r.strength) for r in applied}) == 1 and applied[0].refine is not None:
                start_sigma = services.backend().start_sigma(applied[0])
        elif spec is not None and cells and len({(cell.request.width, cell.request.height, cell.request.steps) for cell in cells}) == 1:
            # Then refine's pass, where every cell's starts at one point.
            start_sigma = services.backend().start_sigma(
                services.backend().applied(then_refine.stand_in(cells[0].request, spec))
            )
        return GridPreview(
            values=values,
            prompt_sample=(
                request.prompt_mode == "random" and request.seed is None and not request.seed_axis.strip()
            ),
            cells=len(cells),
            images=images,
            skipped=skipped,
            queued=queued,
            in_gallery=in_gallery,
            # As the pipeline would run them: the capability flags the status
            # endpoint publishes, applied cell by cell, say which cells collapse
            # into one picture before the night is spent finding out.
            duplicates=grid.duplicate_count(cells, services.backend().applied),
            summary=grid.describe(
                values, listed or request.seeds, listed=bool(listed),
                prompt_mode=request.prompt_mode, batch=request.num_images,
            ),
            seconds=grid.estimate_seconds(
                priced, None, by_size=prices
            ),
            budget_seeds=budget_seeds,
            bytes=disk_bytes,
            start_sigma=start_sigma,
            passes=images if spec is not None else 0,
            pass_width=pass_sizes[0][0] if len(pass_sizes) == 1 else None,
            pass_height=pass_sizes[0][1] if len(pass_sizes) == 1 else None,
            pass_sizes=pass_sizes,
        )

    return router
