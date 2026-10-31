"""Mass refine: the per-image plan, its preview (time and start-sigma
range) and its submission."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import gallery_reports, grid, refine_batch, storage, submission
from .app_context import AppServices
from .model_defaults import defaults_for
from .schemas import (
    GenerateRequest,
    JobAccepted,
    RefineBatchAccepted,
    RefineBatchPreview,
    RefineBatchRequest,
    RefineBatchResult,
    default_strength,
)


# How many images the Mass refine preview lists; its counts cover all of them.
REFINE_BATCH_PREVIEW_ROWS = 200
# How many distinct passes (size, steps, mu, strength as they run) the
# preview reads a start sigma for: each builds a scheduler copy, and a
# selection of many shapes is priced without one rather than slowly.
REFINE_BATCH_SIGMA_KEYS = 8


async def refine_batch_plan(
    services: AppServices, request: RefineBatchRequest,
) -> tuple[list[int], list[RefineBatchResult], list[grid.Cell], list[refine_batch.Planned]]:
    """What Mass refine does to every selected image, and the cells it queues.

    Per image, never for the whole selection (`refine_batch.plan` for what
    the record alone decides): one that is a pass itself, that another
    checkpoint made and was not asked for, whose pass is past the size
    guard, that the strength leaves no step, whose file has gone, whose
    LoRA is not here, or whose pass the gallery already holds is reported as
    such and left out, and the rest are queued, oldest first. Only a server
    with no refine pass at all refuses the whole of it (422), as a single
    pass is refused, and so do more passes than `max_grid_images` (the
    images that would queue × `versions`), by count, as More seeds refuses
    a press. With versions each is a cell the gallery answers for itself:
    an image is `rendered` only where every version is, and otherwise
    queues the versions it lacks. Returns the ids the selection resolved (what `expected`
    pins), every result in id order, the cells to queue in the same order,
    and every record's plan.
    """
    reason = await submission.refine_unavailable(services)
    if reason is not None:
        raise HTTPException(status_code=422, detail=reason)
    records, gone, binned = await submission.resolve_selection(
        services, ids=request.ids, search=request.search, favorites=request.favorites,
        all_models=request.all_models,
    )
    running = services.model_id()
    profile = defaults_for(running)
    strength = request.strength if request.strength is not None else default_strength()
    plans = [
        refine_batch.plan(
            record, running=running, profile=profile, scale=request.scale, strength=strength,
            steps=request.steps, include_foreign=request.include_foreign,
            skip_passes=request.skip_passes, versions=request.versions,
        )
        for record in records
    ]
    passes = sum(len(p.cells) for p in plans)
    limit = services.settings().max_grid_images
    if passes > limit:
        images = sum(p.cell is not None for p in plans)
        raise HTTPException(
            status_code=422,
            detail=(
                f"{images} image{'' if images == 1 else 's'} × {request.versions} versions is {passes} passes; "
                f"a press may queue {limit}. Ask for fewer versions, or pick fewer images."
            ),
        )
    # The record's own file, which the pass starts from: one look per image,
    # off the loop.
    missing = await asyncio.to_thread(
        storage.missing_files, [p.record for p in plans if p.cell is not None]
    )
    checked: dict[str, tuple[str, str] | None] = {}
    for p in plans:
        if p.cell is None:
            continue
        if p.record.id in missing:
            p.drop("file_missing", f"the file for #{p.record.id} is missing from the gallery directory")
            continue
        # Another checkpoint's adapter was dropped by the plan; this one's
        # has to be here, and fit, or the pass would run without it.
        lora = p.cell.request.lora
        if lora:
            if lora not in checked:
                checked[lora] = await submission.unusable_input(services, GenerateRequest(prompt="x", lora=lora))
            unusable = checked[lora]
            if unusable is not None:
                p.drop("lora_missing", unusable[1])
    unheld, kept = await submission.prune_held(services, [cell for p in plans for cell in p.cells])
    cells: list[grid.Cell] = []
    for p in plans:
        if p.cell is None:
            continue
        mine = p.cells
        queued = [cell for cell in mine if id(cell) in kept]
        if queued:
            cells.extend(queued)
            if len(queued) < len(mine):
                # The versions the gallery or a live job holds are left out;
                # the rest queue.
                p.notes.append(
                    f"{len(mine) - len(queued)} of {len(mine)} versions already in the gallery or queued"
                )
        elif any(id(cell) in unheld for cell in mine):
            p.drop("planned", "this pass is already queued or running")
        else:
            p.drop("rendered", "already has this pass in the gallery")
    results = [
        RefineBatchResult(id=image_id, status="deleted", detail=submission.gone_detail(image_id, binned))
        for image_id in gone
    ] + [
        RefineBatchResult.model_validate({
            "id": p.record.id, "thumb_url": p.record.thumb_url, "width": p.record.width,
            "height": p.record.height, "target_width": p.width, "target_height": p.height,
            "seed": p.record.seed, "status": p.status, "detail": p.detail, "notes": p.notes,
        })
        for p in plans
    ]
    results.sort(key=lambda result: result.id)
    return [record.id for record in records], results, cells, plans


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.post("/api/refine/batch/preview", response_model=RefineBatchPreview)
    async def preview_refine_batch(request: RefineBatchRequest) -> RefineBatchPreview:
        """What Mass refine would queue, image by image, without queueing it,
        with its time and where the passes start on the schedule.

        No side effects, so not a model-dependent request in the middleware's
        sense, as Swap tags' preview is not: the queueing request is.
        """
        ids, results, cells, plans = await refine_batch_plan(services, request)
        model = services.model_id()
        sizes = submission.cells_by_size(cells)
        # Time, summed per size from that size's refine history; a size with
        # none leaves the whole estimate out, and how many have none is said.
        seconds = 0.0
        unpriced = 0
        for (width, height), group in sizes.items():
            costs = await asyncio.to_thread(gallery_reports.refine_cost_ms, width, height, model)
            priced = grid.estimate_seconds(group, None, refine_costs=costs)
            if priced is None:
                unpriced += 1
            else:
                seconds += priced
        # Where the passes start: one strength is more noise on a larger pass of
        # a shifted flow schedule, so a mixed selection is a range. Advisory,
        # never a refusal, and read only under a cap of distinct passes.
        low = high = None
        applied: dict[tuple, GenerateRequest] = {}
        for cell in cells:
            ran = services.backend().applied(cell.request)
            applied.setdefault((ran.width, ran.height, ran.steps, ran.mu, ran.strength), ran)
        if applied and len(applied) <= REFINE_BATCH_SIGMA_KEYS:
            sigmas = [services.backend().start_sigma(ran) for ran in applied.values()]
            known_sigmas = [sigma for sigma in sigmas if isinstance(sigma, (int, float))]
            if len(known_sigmas) == len(sigmas):
                low, high = min(known_sigmas), max(known_sigmas)
        # The scale applies to every image not left out before its size is
        # looked at: the largest of those caps the dialog's opening scale.
        scaled = [p.record for p in plans if p.status not in ("pass", "foreign")]
        largest = max(scaled, key=lambda r: (r.width * r.height, r.width), default=None)
        foreign = [p for p in plans if p.foreign]
        return RefineBatchPreview(
            ids=ids,
            counts=submission.status_counts(results),
            queued=len(cells),
            jobs=len(sizes),
            images=results[:REFINE_BATCH_PREVIEW_ROWS],
            total=len(results),
            queued_images=sum(p.cell is not None for p in plans),
            versions=request.versions,
            strength=request.strength if request.strength is not None else default_strength(),
            foreign=len(foreign),
            models=sorted({p.record.model_id for p in foreign}),
            passes=sum(p.record.refine is not None for p in plans),
            largest=None if largest is None else [largest.width, largest.height],
            seconds=None if unpriced or not cells else round(seconds, 1),
            unpriced_sizes=unpriced,
            start_sigma_low=low,
            start_sigma_high=high,
        )


    @router.post("/api/refine/batch", response_model=RefineBatchAccepted)
    async def refine_batch_submit(request: RefineBatchRequest) -> RefineBatchAccepted:
        """Queue passes, leaving out gallery and live-job cells."""
        async with services.jobs().rerun_admission:
            return await submit_refine_batch(request)


    async def submit_refine_batch(request: RefineBatchRequest) -> RefineBatchAccepted:
        """Queue a refine pass of every selected image at one strength and scale.

        The fifth producer of cells: each image a cell of its own per version
        (`refine_batch.plan` through `grid.cell_from_record`) with its own
        prompt pair, seed, CFG, mu and LoRA, a pass of itself. One job per
        distinct size, as Swap tags queues, so each job keeps one size and the
        run id every run's has; `expected` pins the ids the preview resolved,
        and a pass the gallery holds at queue time is left out and counted in
        `pruned`. With nothing left: 409 where every image already has the
        pass, else 422 with every count - the already rendered among them - so
        the images the user could still fix (too big, a LoRA missing) are said.
        """
        ids, results, cells, _ = await refine_batch_plan(services, request)
        if request.expected is not None and sorted(set(request.expected)) != ids:
            raise HTTPException(
                status_code=409,
                detail="the selection changed since the check; review the refine again",
            )
        counts = submission.status_counts(results)
        if not cells:
            # As the dialog's hold words it (`refineBatchHold`): "already has the
            # pass" only where that is every image's status.
            if set(counts) == {"rendered"}:
                raise HTTPException(
                    status_code=409,
                    detail="every image that could be refined already has this pass in the gallery",
                )
            if set(counts) <= {"rendered", "planned"}:
                detail = (
                    "every image that could be refined already has this pass in the gallery or queued"
                    if counts.get("rendered") else
                    "every image that could be refined already has this pass queued or running"
                )
                raise HTTPException(status_code=409, detail=detail)
            said = ", ".join(
                f"{n} {'already queued' if status == 'planned' else status.replace('_', ' ')}"
                for status, n in counts.items()
            )
            raise HTTPException(status_code=422, detail=f"no selected image can be refined ({said})")
        model, quantization = services.model_id(), services.quantization()
        accepted = []
        for group in submission.cells_by_size(cells).values():
            job = await services.jobs().submit_cells(
                group, run_id=grid.run_id(group, model, quantization), refine_batch=request.scale,
            )
            accepted.append(JobAccepted.model_validate({
                "job_id": job.id, "seed": job.seed, "run_id": job.run_id, "estimate_seconds": job.estimate_seconds,
            }))
        first = accepted[0]
        return RefineBatchAccepted(
            job_id=first.job_id, seed=first.seed, run_id=first.run_id, estimate_seconds=first.estimate_seconds,
            more=accepted[1:],
            queued=len(cells), pruned=counts.get("rendered", 0), planned=counts.get("planned", 0),
        )

    return router
