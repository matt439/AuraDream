"""Run details and Finish run (`/api/runs/{id}`, `finish_plan`)."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field, replace

from fastapi import APIRouter, HTTPException

from . import grid, storage, submission, then_refine
from .api_more_seeds import more_seeds_plan
from .app_context import AppServices
from .execution import require_configured_loader_available, resolve_execution
from .model_selection import current_settings
from .schemas import MoreSeedsRequest, ThenRefine, error_detail
from .schemas_rerun import (
    FinishRunAccepted, FinishRunPreview, FinishRunRequest, RunAction, RunDetails,
)


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


def _finish_key(services: AppServices, cell: grid.Cell) -> tuple:
    ran = grid.Cell(request=services.backend().applied(cell.request), seed=cell.seed)
    return cell.request.width, cell.request.height, tuple(grid.cell_keys(ran))


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
    # Bases whose pass a retained job saved: one missing now was
    # permanently deleted, and Finish says it redraws it, as for a cell.
    passes_made: set[int] = set()
    for saved in definitions:
        retained = then_refine.retained(saved["metadata"])
        if retained is not None:
            spec_jobs.add(saved["job_id"])
            if retained not in specs:
                specs.append(retained)
            made = (saved.get("outcome") or {}).get("completed_passes", [])
            if isinstance(made, list):
                passes_made.update(base for base in made if type(base) is int)
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
        passes=len(kept_passes),
        redrawn_passes=sum(cell.request.refine in passes_made for cell in kept_passes),
        budget_minutes=budget, pin=pin,
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
                # Bases only: a batch whose pass is refused still makes
                # bases with the dialog's tick off, so it can have more.
                _, results, batches = await more_seeds_plan(
                    services, MoreSeedsRequest(run=run_id, count=1, then_refine=False),
                )
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
                "passes": preview.passes, "redrawn_passes": preview.redrawn_passes, "budget_minutes": preview.budget_minutes,
            })

    return router
