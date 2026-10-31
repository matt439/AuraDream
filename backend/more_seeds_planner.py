"""More seeds' plan: what it makes of every selected batch.

The planner both its routes (`api_more_seeds.py`) and run details
(`api_runs.py`) call, so no router imports another; the batches, seeds and
cells themselves are `more_seeds.py`'s, pure.
"""
from __future__ import annotations

import asyncio

from fastapi import HTTPException

from . import grid, more_seeds, storage, submission, then_refine
from .app_context import AppServices
from .schemas import (
    ImageRecord,
    MoreSeedsRequest,
    MoreSeedsResult,
    ThenRefine,
)
from .seeds import MAX_SEED


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
    With `run`, the selection is every live image of the run, Starred or
    not (`submission.resolve_selection`); its binned images are no batches
    but still hold their seeds, as `lineage_images` reads every row. No cap
    on its images: the batches × `count` refusal bounds a press.
    Returns the ids the selection resolved (what `expected` pins), every
    result in id order, and the batches, queued or not.
    """
    records, gone, binned = await submission.resolve_selection(
        services, ids=request.ids, search=request.search, favorites=request.favorites,
        all_models=request.all_models, run=request.run,
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
                f"a press may queue {services.settings().max_grid_images}. "
                + ("Ask for fewer." if request.run is not None else "Ask for fewer, or pick fewer batches.")
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
    # A batch whose outpaint spec does not build its row's mask would paint
    # its new seeds under another mask, held as the same set.
    mismatched = set(await asyncio.to_thread(
        submission.mismatched_outpaints, [batch.template for batch in mine]
    ))
    for batch in mine:
        template = batch.template
        more_seeds.plan(
            batch, count=request.count, fallback=request.batch, applied=services.backend().applied,
            held=held[(template.width, template.height)],
        )
        if batch.status != more_seeds.QUEUED:
            continue
        if template.id in mismatched:
            batch.status = "outpaint"
            batch.detail = submission.mismatched_outpaint_detail([template.id])
            continue
        # New seeds are new pictures under any quantizer, but the batch's
        # images were all made under one; said, not refused.
        if template.quantization and quantization and template.quantization != quantization:
            batch.notes.append(f"made under {template.quantization}; these run under {quantization}")
        first = batch.cells[0].request
        spec = submission.input_spec(first)
        if spec != submission.NO_INPUTS:
            if spec not in checked:
                # An edit's batch queues here (the producer), its source
                # proved against its recorded pixels and held to the envelope.
                checked[spec] = await submission.unusable_input(services, first, producer="more_seeds")
            unusable = checked[spec]
            if unusable is not None:
                kind, why = unusable
                batch.status = submission.UNUSABLE_STATUS.get(kind, "invalid")
                batch.detail = submission.unusable_detail(kind, why, batch.template.id)
    await _carry_then_refine(services, [batch for batch in mine if batch.status == more_seeds.QUEUED],
                             carry=request.then_refine)
    results = [
        MoreSeedsResult(id=image_id, status="deleted", detail=submission.gone_detail(image_id, binned))
        for image_id in gone
    ] + [_more_seeds_result(batch) for batch in batches]
    results.sort(key=lambda result: result.id)
    return [record.id for record in records], results, batches


async def _carry_then_refine(services: AppServices, queued: list[more_seeds.Batch], *, carry: bool) -> None:
    """Each queued base batch's Then refine: the spec the jobs of its images
    retained (`then_refine.retained`), where they agree, so its new bases
    are each followed by the pass the run's own bases had. Every mate's job
    is read, not the template's alone: the template can be a base an
    earlier More seeds made before it carried a pass. A batch none of whose
    jobs carried one is unchanged; a pass batch, an inpaint or an edit
    never carries one (a pass of a pass is no Then refine). With no
    refine pass on this checkpoint the batches make bases only, each noted
    so whatever `carry` says; with `carry` off they make bases only; else mates of two specs, or a pass `then_refine.refusal` refuses at
    this batch's size and steps, is a `then_refine` status in words, and
    the rest still queue."""
    bases = [batch for batch in queued if batch.template.refine is None and batch.template.edit is None]
    job_ids = sorted({record.job_id for batch in bases for record in batch.held if record.job_id})
    retained = await asyncio.to_thread(storage.retained_run_metadata, job_ids) if job_ids else {}
    unavailable: str | None = None
    asked = False
    for batch in bases:
        specs: list[ThenRefine] = []
        for job_id in dict.fromkeys(record.job_id for record in batch.held if record.job_id):
            if job_id not in retained:
                continue
            spec = then_refine.retained(retained[job_id][1])
            if spec is not None and spec not in specs:
                specs.append(spec)
        if not specs:
            continue
        batch.run_then_refine = True
        # Where this checkpoint has no pass the note says so whatever the
        # tick: the page unticks its held tick and asks again with `carry`
        # off, and the row still says in words why its run's pass is left.
        if not asked:
            unavailable, asked = await submission.refine_unavailable(services), True
        if unavailable is not None:
            batch.notes.append("bases only: this checkpoint has no refine pass")
            continue
        if not carry:
            continue
        if len(specs) > 1:
            made = [then_refine.words(each) for each in specs]
            batch.status = "then_refine"
            batch.detail = (
                f"its images were refined at {', '.join(made[:-1])} and at {made[-1]}; untick "
                "Refine each new image for bases only, and Mass refine them at the numbers wanted"
            )
            continue
        spec = then_refine.resolved(specs[0])
        reason = then_refine.refusal([cell.request for cell in batch.cells], spec, None)
        if reason is not None:
            batch.status, batch.detail = "then_refine", f"its run's pass: {reason}"
            continue
        batch.then_refine = spec
        batch.notes.append(f"each then refined at {then_refine.words(spec)}")


def _more_seeds_result(batch: more_seeds.Batch) -> MoreSeedsResult:
    template = batch.template
    held = [record.seed for record in batch.held]
    queued = batch.status == more_seeds.QUEUED
    seeds = batch.seeds if queued else []
    return MoreSeedsResult.model_validate({
        "id": template.id, "ids": [record.id for record in batch.picked], "thumb_url": template.thumb_url,
        "width": template.width, "height": template.height, "prompt": template.prompt[:300],
        "images": len(held), "seed_low": min(held), "seed_high": max(held),
        "first": seeds[0] if seeds else None, "last": seeds[-1] if seeds else None, "queued": len(seeds),
        # The seeds from the batch's highest to the last new one, less the
        # new ones: what was stepped over. Through the wrap, as they are.
        "skipped": (seeds[-1] - max(held)) % (MAX_SEED + 1) - len(seeds) if seeds else 0,
        "batch": batch.size if queued else None, "recorded": batch.recorded,
        "status": batch.status, "detail": batch.detail, "notes": batch.notes if queued else [],
    })


def _batches_by_size(batches: list[more_seeds.Batch]) -> dict[tuple[int, int], list[more_seeds.Batch]]:
    sizes: dict[tuple[int, int], list[more_seeds.Batch]] = {}
    for batch in batches:
        sizes.setdefault((batch.template.width, batch.template.height), []).append(batch)
    return sizes
