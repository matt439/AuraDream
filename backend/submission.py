"""What every cell producer checks before it queues, and the gallery
selection Re-render, Swap tags, Mass refine and More seeds act on."""
from __future__ import annotations

import asyncio

from fastapi import HTTPException

from . import grid, loras, storage
from .app_context import AppServices
from .schemas import GenerateRequest, ImageRecord


# The per-image status a reason from `unusable_input` becomes.
UNUSABLE_STATUS = {"source": "source_deleted", "lora": "lora_missing"}


async def refuse_unusable_inputs(services: AppServices, request: GenerateRequest) -> None:
    """Refuse a reference, an adapter or a refine source this server cannot
    run - at submit.

    All three are all-or-nothing inputs, unlike a negative prompt: a night
    queued with references against a pipeline that takes none, with an adapter
    no file backs, or as a refine pass on a checkpoint that has none, would
    render every cell as a plain text-to-image and look like a success.
    `applied()` keeps the *record* honest about that; this is what stops it
    happening at cell 1 of 64.
    """
    unusable = await unusable_input(services, request)
    if unusable is not None:
        raise HTTPException(status_code=422, detail=unusable[1])


async def unusable_input(services: AppServices, request: GenerateRequest) -> tuple[str, str] | None:
    """`refuse_unusable_inputs`' finding without the refusal: (kind, why) or
    None. The kind - `refine`, `source`, `reference`, `references`, `lora` -
    is what Swap tags reports per image (a deleted source, a missing adapter)
    where a whole submission is refused for one."""
    if request.refine is not None:
        reason = await refine_unavailable(services)
        if reason is not None:
            return "refine", reason
        try:
            await asyncio.to_thread(storage.reference_paths, [request.refine], "source image")
        except ValueError as exc:
            return "source", str(exc)
    if request.reference:
        info = await asyncio.to_thread(services.backend().describe)
        if not services.merged_defaults(info)["reference_supported"]:
            return "reference", (
                f"{info['model_id']} takes no reference images - its pipeline has no "
                "`image` parameter. FLUX.2 [klein] is the checkpoint here that conditions "
                "on them."
            )
        try:
            await asyncio.to_thread(storage.reference_paths, request.reference)
        except ValueError as exc:
            return "references", str(exc)
    if request.lora:
        reason = services.backend().lora_unavailable_reason()
        if reason is not None:
            return "lora", reason
        try:
            loras.path_for(request.lora)
        except ValueError as exc:
            return "lora", str(exc)
        # And one that is there but for another checkpoint: a FLUX.1 adapter
        # on Z-Image fails at the first cell, deep in PEFT, naming neither.
        reason = await asyncio.to_thread(services.backend().lora_mismatch, request.lora)
        if reason is not None:
            return "lora", reason
    return None


async def refine_unavailable(services: AppServices) -> str | None:
    """Why this server has no refine pass, or None where it has one: what a
    pass's submission is refused with, Mass refine's included."""
    info = await asyncio.to_thread(services.backend().describe)
    if services.merged_defaults(info)["refine_supported"]:
        return None
    return (
        f"{info['model_id']} has no refine pass in this app - its integration "
        "has no compatible image-to-image counterpart. The FLUX.1, SDXL, Z-Image and "
        "Qwen-Image checkpoints have one."
    )


async def not_yet_rendered(
    services: AppServices, cells: list[grid.Cell], width: int, height: int
) -> list[grid.Cell]:
    """The cells the gallery does not hold yet at this size: what Resume runs.

    Compare what the pipeline would *run*, not what the panel says: the rows
    were written from `backend.applied()`, so a negative prompt a Turbo cell
    at CFG 0 never saw is None on both sides rather than on one. Off the
    loop: up to three chunked scans, fired by the preview on every keystroke.
    """
    done = await asyncio.to_thread(
        storage.completed_cells,
        grid.prompts_of(cells),
        width,
        height,
        services.model_id(),
        services.quantization(),
    )
    return [
        cell
        for cell in cells
        if grid.prune_done([grid.Cell(request=services.backend().applied(cell.request), seed=cell.seed)], done)
    ]


async def resolve_selection(
    services: AppServices, *, ids: list[int] | None, search: str | None, favorites: bool, all_models: bool,
) -> tuple[list[ImageRecord], list[int], set[int]]:
    """The images an action on several of them acts on, oldest first, the
    picked ids the gallery no longer holds, and those of them that are in
    the recycle bin.

    With `ids` - the gallery's picked cards (`gallery-select.js`) - exactly
    those, whatever the filter; an id deleted or binned since it was picked
    is returned in the second list for the action to report per image, not
    refused, and the third says which were binned (`gone_detail`). Without
    them, the gallery's filter as Re-render has always taken it: the search,
    Starred and the checkpoint scope (`AppServices.model_scope`), 404 where nothing
    matches and 422 past `max_grid_images` - by count, never truncated to
    the newest N. The recycle bin is never part of a filter's selection. Re-render, Swap tags and Mass refine share it;
    what to do with another checkpoint's records is each caller's.
    """
    if ids is not None:
        wanted = list(dict.fromkeys(ids))
        records = await asyncio.to_thread(storage.get_images, wanted)
        held = {record.id for record in records}
        gone = sorted(image_id for image_id in wanted if image_id not in held)
        binned = (
            {record.id for record in await asyncio.to_thread(storage.get_images, gone, True)}
            if gone else set()
        )
        return records, gone, binned
    # One past the ceiling, so an oversize selection is refused by count rather
    # than silently truncated to the newest N.
    records, total, _ = await asyncio.to_thread(
        storage.list_images,
        limit=services.settings().max_grid_images + 1,
        search=search,
        favorites_only=favorites,
        model_id=services.model_scope(all_models),
    )
    if not records:
        raise HTTPException(status_code=404, detail="nothing in the gallery matches that selection")
    if total > services.settings().max_grid_images:
        raise HTTPException(
            status_code=422,
            detail=f"{total} images match; a job may queue {services.settings().max_grid_images}. Narrow the search.",
        )
    return list(reversed(records)), [], set()


def gone_detail(image_id: int, binned: set[int]) -> str:
    """Why a picked id is `deleted`: binned since it was picked, or gone."""
    return "in the recycle bin" if image_id in binned else "no longer in the gallery"


def cells_by_size(cells: list[grid.Cell]) -> dict[tuple[int, int], list[grid.Cell]]:
    sizes: dict[tuple[int, int], list[grid.Cell]] = {}
    for cell in cells:
        sizes.setdefault((cell.request.width, cell.request.height), []).append(cell)
    return sizes
