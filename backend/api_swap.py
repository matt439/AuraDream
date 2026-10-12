"""Swap tags: the per-image plan, its preview and its submission."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import grid, submission, tag_swap
from .app_context import AppServices
from .schemas import (
    JobAccepted,
    SwapResult,
    SwapTagsAccepted,
    SwapTagsPreview,
    SwapTagsRequest,
    error_detail,
)


# How many images the Swap tags preview lists; its counts cover all of them.
# A thousand records with their prompts, every keystroke, is megabytes.
SWAP_PREVIEW_ROWS = 200

_SWAP_SAYS = {
    "structured": "an XML or JSON prompt is not rewritten",
    "unchanged": "the swaps change nothing here",
    "rendered": "already in the gallery",
}


async def swap_plan(
    services: AppServices, request: SwapTagsRequest
) -> tuple[list[int], list[SwapResult], list[grid.Cell]]:
    """What Swap tags does to every selected image, and the cells it queues.

    Per image, never for the whole selection: one record that is missing a
    tag, that the swap leaves empty or too long, whose refine source or
    adapter has gone, or that another checkpoint made is reported as such and
    left out, and the rest are queued. Two records that come out as the same
    cell at the same size are queued once ("same as #N"), and a cell the
    gallery already holds is `rendered`, as resume would skip it. Returns the
    ids the selection resolved to (what `expected` pins), every result in id
    order, and the cells to queue in the same order.
    """
    if request.field == "negative_prompt":
        info = await asyncio.to_thread(services.backend().describe)
        if not services.merged_defaults(info)["negative_prompt_supported"]:
            raise HTTPException(
                status_code=422,
                detail=f"{info['model_id']} takes no negative prompt; swap tags in the prompt",
            )
    rows = tag_swap.parse_rows([(row.find, row.replace) for row in request.swaps])
    records, gone, binned = await submission.resolve_selection(
        services, ids=request.ids, search=request.search, favorites=request.favorites,
        all_models=request.all_models,
    )
    running = services.model_id()
    results: list[SwapResult] = [
        SwapResult(id=image_id, status="deleted", detail=submission.gone_detail(image_id, binned))
        for image_id in gone
    ]
    queued: list[tuple[SwapResult, grid.Cell]] = []
    checked: dict[tuple, tuple[str, str] | None] = {}
    seen: dict[tuple, int] = {}
    for record in records:
        result = SwapResult(
            id=record.id, thumb_url=record.thumb_url, width=record.width, height=record.height,
            seed=record.seed, status="unchanged",
            text=(record.prompt if request.field == "prompt" else record.negative_prompt) or "",
        )
        results.append(result)
        if record.model_id != running:
            # Steps, CFG and mu are the checkpoint's: refused as Re-render
            # refuses them, but per image.
            result.status = "foreign"
            result.detail = f"made on {record.model_id}; the server is running {running}"
            continue
        swapped = tag_swap.swap_record(
            record.prompt, record.negative_prompt, record.variables, rows, request.field
        )
        outcome = swapped.outcome
        result.notes = outcome.notes
        if outcome.status == "missing":
            result.status, result.detail = "missing", f"missing: {', '.join(outcome.missing)}"
            continue
        if outcome.status == "by_hand":
            result.status = "by_hand"
            result.detail = f"needs editing by hand - a phrase in prose: {'; '.join(outcome.by_hand)}"
            continue
        if outcome.status != "changed":
            result.status, result.detail = outcome.status, _SWAP_SAYS[outcome.status]
            continue
        result.text, result.spans, result.removed = outcome.text, outcome.spans, outcome.removed
        try:
            cell = grid.cell_from_record(
                record, prompt=swapped.prompt, negative_prompt=swapped.negative_prompt,
                variables=swapped.variables,
            )
            cell.request.check_budget()
            cell.request.check_refine_steps()
        except ValueError as exc:
            result.status, result.detail = "invalid", error_detail(exc)
            continue
        spec = (cell.request.lora, tuple(cell.request.reference), cell.request.refine)
        if spec != (None, (), None):
            if spec not in checked:
                checked[spec] = await submission.unusable_input(services, cell.request)
            if checked[spec] is not None:
                kind, why = checked[spec]
                result.status, result.detail = submission.UNUSABLE_STATUS.get(kind, "invalid"), why
                continue
        ran = grid.Cell(request=services.backend().applied(cell.request), seed=cell.seed)
        key = (tuple(grid.cell_keys(ran)), cell.request.width, cell.request.height)
        if key in seen:
            result.status, result.detail = "duplicate", f"same as #{seen[key]}"
            continue
        seen[key] = record.id
        result.status = "queued"
        queued.append((result, cell))
    # The gallery's own answer, per size: `completed_cells` is scoped to one.
    sizes: dict[tuple[int, int], list[grid.Cell]] = {}
    for _, cell in queued:
        sizes.setdefault((cell.request.width, cell.request.height), []).append(cell)
    kept: set[int] = set()
    for (width, height), group in sizes.items():
        kept.update(id(cell) for cell in await submission.not_yet_rendered(services, group, width, height))
    cells = []
    for result, cell in queued:
        if id(cell) in kept:
            cells.append(cell)
        else:
            result.status, result.detail = "rendered", _SWAP_SAYS["rendered"]
    results.sort(key=lambda result: result.id)
    return [record.id for record in records], results, cells


def _swap_counts(results: list[SwapResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.post("/api/swap-tags/preview", response_model=SwapTagsPreview)
    async def preview_swap_tags(request: SwapTagsRequest) -> SwapTagsPreview:
        """What Swap tags would queue, image by image, without queueing it.

        No side effects, so it is not a model-dependent request in the
        middleware's sense: the queueing request is, and carries the revision.
        """
        ids, results, cells = await swap_plan(services, request)
        return SwapTagsPreview(
            ids=ids,
            counts=_swap_counts(results),
            queued=len(cells),
            jobs=len({(cell.request.width, cell.request.height) for cell in cells}),
            images=results[:SWAP_PREVIEW_ROWS],
            total=len(results),
        )


    @router.post("/api/swap-tags", response_model=SwapTagsAccepted)
    async def swap_tags(request: SwapTagsRequest) -> SwapTagsAccepted:
        """Queue the selected images again with some of their tags swapped.

        The fourth producer of cells, beside Generate, Run grid and Re-render:
        each image is a cell with its own seed, size and settings
        (`grid.cell_from_record`), only the prompt or the negative prompt - and
        the variables that name parts of it - rewritten, server-side
        (`tag_swap`). One job per distinct size, so every job keeps one size
        and its run id the shape every other run's has. `expected` is the ids
        the preview resolved: a selection that changed since is refused. An
        image whose cell the gallery holds at queue time is left out and
        counted in `pruned` - the preview's "already rendered" ones and any
        rendered since; with nothing left, 409 where it was all already
        rendered, else 422.
        """
        ids, results, cells = await swap_plan(services, request)
        if request.expected is not None and sorted(set(request.expected)) != ids:
            raise HTTPException(
                status_code=409,
                detail="the selection changed since the check; review the swaps again",
            )
        counts = _swap_counts(results)
        if not cells:
            if counts.get("rendered"):
                raise HTTPException(
                    status_code=409, detail="every image the swap would change is already in the gallery"
                )
            said = ", ".join(f"{n} {status.replace('_', ' ')}" for status, n in counts.items())
            raise HTTPException(status_code=422, detail=f"no selected image would change ({said})")
        rows = tag_swap.parse_rows([(row.find, row.replace) for row in request.swaps])
        summary = "; ".join(row.describe() for row in rows)
        if request.field == "negative_prompt":
            summary = f"negative: {summary}"
        sizes: dict[tuple[int, int], list[grid.Cell]] = {}
        for cell in cells:
            sizes.setdefault((cell.request.width, cell.request.height), []).append(cell)
        model, quantization = services.model_id(), services.quantization()
        accepted = []
        for group in sizes.values():
            job = await services.jobs().submit_cells(group, run_id=grid.run_id(group, model, quantization), swap=summary)
            accepted.append(JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id))
        first = accepted[0]
        return SwapTagsAccepted(
            job_id=first.job_id, seed=first.seed, run_id=first.run_id, more=accepted[1:],
            queued=len(cells), pruned=counts.get("rendered", 0),
        )

    return router
