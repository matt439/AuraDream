"""More seeds: the per-batch plan, its preview and its submission."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import gallery_reports, grid, more_seeds, storage, submission
from .app_context import AppServices
from .schemas import (
    ImageRecord,
    JobAccepted,
    MoreSeedsAccepted,
    MoreSeedsPreview,
    MoreSeedsRequest,
    MoreSeedsResult,
)
from .seeds import MAX_SEED


# How many batches the More seeds preview lists; its counts cover all of them.
MORE_SEEDS_PREVIEW_ROWS = 200


async def more_seeds_plan(
    services: AppServices, request: MoreSeedsRequest,
) -> tuple[list[int], list[MoreSeedsResult], list[more_seeds.Batch]]:
    """What More seeds makes of every selected batch, and the batches.

    Per batch, never for the whole selection (`more_seeds.plan` for the
    seeds, the Batch and the cells): one another checkpoint made, whose
    record makes no valid request, or whose refine source or adapter has
    gone is reported as such and left out, and the rest are queued. Only
    more than `max_grid_images` images in all refuses the whole of it (422),
    by count, as a filter past it is refused. Each batch's seeds continue
    past every image of it the gallery holds and step over what the gallery
    and the live jobs hold at its settings, so nothing is ever `rendered`.
    Returns the ids the selection resolved (what `expected` pins), every
    result in id order, and the batches, queued or not.
    """
    records, gone, binned = await submission.resolve_selection(
        services, ids=request.ids, search=request.search, favorites=request.favorites,
        all_models=request.all_models,
    )
    running = services.model_id()
    batches = more_seeds.group(records)
    mine: list[more_seeds.Batch] = []
    for batch in batches:
        made_on = batch.template.model_id
        if made_on != running:
            # Steps, CFG and mu are the checkpoint's: refused as Re-render
            # refuses them, but per batch.
            batch.status = "foreign"
            batch.detail = f"made on {made_on}; the server is running {running}"
        else:
            mine.append(batch)
    images = len(mine) * request.count
    if images > services.settings().max_grid_images:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{len(mine)} batch{'' if len(mine) == 1 else 'es'} × {request.count} is {images} images; "
                f"a press may queue {services.settings().max_grid_images}. Ask for fewer, or pick fewer batches."
            ),
        )
    # Every image of each batch's run or job, picked or not: where its
    # seeds end, and the Batch it was made at.
    lineages = [batch.key[0] for batch in mine]
    gallery = await asyncio.to_thread(
        storage.lineage_images,
        [name for kind, name in lineages if kind == "run"],
        [name for kind, name in lineages if kind == "job"],
    )
    by_key: dict[tuple, list[ImageRecord]] = {}
    for record in gallery:
        by_key.setdefault(more_seeds.batch_key(record), []).append(record)
    for batch in mine:
        batch.mates = by_key.get(batch.key, [])
    # What the gallery and the live jobs hold at each size, as `applied()`
    # runs it: `completed_cells` is scoped to one size.
    quantization = services.quantization()
    planned = services.jobs().planned_keys()
    held: dict[tuple[int, int], set[grid.CellKey]] = {}
    for (width, height), group in _batches_by_size(mine).items():
        done = await asyncio.to_thread(
            storage.completed_cells, [batch.template.prompt for batch in group],
            width, height, running, quantization,
        )
        held[(width, height)] = done | planned.get((width, height), set())
    checked: dict[tuple, tuple[str, str] | None] = {}
    for batch in mine:
        template = batch.template
        more_seeds.plan(
            batch, count=request.count, fallback=request.batch, applied=services.backend().applied,
            held=held[(template.width, template.height)],
        )
        if batch.status != more_seeds.QUEUED:
            continue
        # New seeds are new pictures under any quantizer, but the batch's
        # images were all made under one; said, not refused.
        if template.quantization and quantization and template.quantization != quantization:
            batch.notes.append(f"made under {template.quantization}; these run under {quantization}")
        first = batch.cells[0].request
        spec = (first.lora, tuple(first.reference), first.refine, first.mask, first.outpaint)
        if spec != (None, (), None, None, None):
            if spec not in checked:
                checked[spec] = await submission.unusable_input(services, first)
            if checked[spec] is not None:
                kind, why = checked[spec]
                batch.status = submission.UNUSABLE_STATUS.get(kind, "invalid")
                batch.detail = submission.unusable_detail(kind, why, batch.template.id)
    results = [
        MoreSeedsResult(id=image_id, status="deleted", detail=submission.gone_detail(image_id, binned))
        for image_id in gone
    ] + [_more_seeds_result(batch) for batch in batches]
    results.sort(key=lambda result: result.id)
    return [record.id for record in records], results, batches


def _more_seeds_result(batch: more_seeds.Batch) -> MoreSeedsResult:
    template = batch.template
    held = [record.seed for record in batch.held]
    queued = batch.status == more_seeds.QUEUED
    seeds = batch.seeds if queued else []
    return MoreSeedsResult(
        id=template.id, ids=[record.id for record in batch.picked], thumb_url=template.thumb_url,
        width=template.width, height=template.height, prompt=template.prompt[:300],
        images=len(held), seed_low=min(held), seed_high=max(held),
        first=seeds[0] if seeds else None, last=seeds[-1] if seeds else None, queued=len(seeds),
        # The seeds from the batch's highest to the last new one, less the
        # new ones: what was stepped over. Through the wrap, as they are.
        skipped=(seeds[-1] - max(held)) % (MAX_SEED + 1) - len(seeds) if seeds else 0,
        batch=batch.size if queued else None, recorded=batch.recorded,
        status=batch.status, detail=batch.detail, notes=batch.notes if queued else [],
    )


def _batches_by_size(batches: list[more_seeds.Batch]) -> dict[tuple[int, int], list[more_seeds.Batch]]:
    sizes: dict[tuple[int, int], list[more_seeds.Batch]] = {}
    for batch in batches:
        sizes.setdefault((batch.template.width, batch.template.height), []).append(batch)
    return sizes


def _more_seeds_jobs(batches: list[more_seeds.Batch]) -> dict[tuple, list[more_seeds.Batch]]:
    """The queued batches by the job they go in: one per run (else job, else
    image) and size, so each job's images join the one run they extend and
    the job keeps one size, as every job does."""
    jobs_of: dict[tuple, list[more_seeds.Batch]] = {}
    for batch in batches:
        template = batch.template
        jobs_of.setdefault((batch.key[0], template.width, template.height), []).append(batch)
    return jobs_of


def _more_seeds_counts(results: list[MoreSeedsResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.post("/api/more-seeds/preview", response_model=MoreSeedsPreview)
    async def preview_more_seeds(request: MoreSeedsRequest) -> MoreSeedsPreview:
        """What More seeds would queue, batch by batch, without queueing it,
        and how long it would take.

        No side effects, so not a model-dependent request in the middleware's
        sense, as Swap tags' preview is not: the queueing request is.
        """
        ids, results, batches = await more_seeds_plan(services, request)
        queued = [batch for batch in batches if batch.status == more_seeds.QUEUED]
        model = services.model_id()
        # Time, summed per size from that size's history; a size with none
        # leaves the whole estimate out, and how many have none is said.
        seconds = 0.0
        unpriced = 0
        for (width, height), group in _batches_by_size(queued).items():
            cells = [cell for batch in group for cell in batch.cells]
            per_step_ms = await asyncio.to_thread(gallery_reports.step_cost_ms, width, height, model)
            refine_costs = await asyncio.to_thread(gallery_reports.refine_cost_ms, width, height, model)
            inpaint_costs = (
                await asyncio.to_thread(gallery_reports.refine_cost_ms, width, height, model, inpaint=True)
                if any(cell.request.inpainting for cell in cells) else None
            )
            priced = grid.estimate_seconds(
                cells, per_step_ms, refine_costs=refine_costs, inpaint_costs=inpaint_costs
            )
            if priced is None:
                unpriced += 1
            else:
                seconds += priced
        return MoreSeedsPreview(
            ids=ids,
            counts=_more_seeds_counts(results),
            queued=sum(len(batch.seeds) for batch in queued),
            batches=len(queued),
            jobs=len(_more_seeds_jobs(queued)),
            results=results[:MORE_SEEDS_PREVIEW_ROWS],
            total=len(results),
            unrecorded=sum(not batch.recorded for batch in queued),
            seconds=None if unpriced or not queued else round(seconds, 1),
            unpriced_sizes=unpriced,
        )


    @router.post("/api/more-seeds", response_model=MoreSeedsAccepted)
    async def more_seeds_submit(request: MoreSeedsRequest) -> MoreSeedsAccepted:
        """Queue `count` more images of every selected batch.

        The sixth producer of cells: each batch's next seeds, consecutive ones
        in a call of its Batch size, every other setting the batch's own
        (`more_seeds.plan` through `grid.cell_from_record`). One job per run the
        batches extend, stamped with that run's id, so the new images join it -
        the batch hover, the Runs picker and `run:` show one set - and each job
        keeps one size. `expected` pins the ids the preview resolved. With
        nothing queueable, 422 naming every count.
        """
        ids, results, batches = await more_seeds_plan(services, request)
        if request.expected is not None and sorted(set(request.expected)) != ids:
            raise HTTPException(
                status_code=409,
                detail="the selection changed since the check; review the batches again",
            )
        queued = [batch for batch in batches if batch.status == more_seeds.QUEUED]
        if not queued:
            said = ", ".join(
                f"{n} {status.replace('_', ' ')}" for status, n in _more_seeds_counts(results).items()
            )
            raise HTTPException(status_code=422, detail=f"no selected batch can have more seeds ({said})")
        accepted = []
        for group in _more_seeds_jobs(queued).values():
            job = await services.jobs().submit_cells(
                [cell for batch in group for cell in batch.cells],
                run_id=group[0].run_id, more_seeds=request.count,
            )
            accepted.append(JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id))
        first = accepted[0]
        return MoreSeedsAccepted(
            job_id=first.job_id, seed=first.seed, run_id=first.run_id, more=accepted[1:],
            queued=sum(len(batch.seeds) for batch in queued), batches=len(queued),
        )

    return router
