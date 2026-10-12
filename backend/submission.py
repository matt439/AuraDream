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
    """Refuse a reference, an adapter, a refine source or an inpaint's mask
    this server cannot run - at submit.

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
    None. The kind - `refine`, `inpaint`, `source`, `mask`, `reference`,
    `references`, `lora` - is what Swap tags reports per image (a deleted
    source, a missing adapter) where a whole submission is refused for one.

    Whether a pass can run from a source splits on the mask: with one it is
    the inpaint class's question, without one the image-to-image class's, as
    `applied()` asks it."""
    if request.refine is not None:
        inpainting = request.mask is not None
        reason = await (inpaint_unavailable(services) if inpainting else refine_unavailable(services))
        if reason is not None:
            return ("inpaint" if inpainting else "refine"), reason
        try:
            await asyncio.to_thread(storage.reference_paths, [request.refine], "source image")
        except ValueError as exc:
            return "source", str(exc)
        if inpainting:
            reason = await asyncio.to_thread(mask_unusable, request)
            if reason is not None:
                return "mask", reason
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


async def inpaint_unavailable(services: AppServices) -> str | None:
    """Why this server has no inpaint, or None where it has one: what an
    inpaint's submission is refused with. The profile's answer before a load,
    the loaded class's after, as `refine_unavailable` asks."""
    info = await asyncio.to_thread(services.backend().describe)
    if services.merged_defaults(info)["inpaint_supported"]:
        return None
    return (
        f"{info['model_id']} has no inpaint in this app - its pipeline has no inpaint class "
        "in the installed diffusers, or its family's is not enabled yet. The SDXL and "
        "Z-Image checkpoints have one."
    )


def missing_masks(records) -> list[int]:
    """The ids of the inpaints among `records` whose mask this gallery cannot
    find - imported without it (a bundle's `masks_missing`), or its file lost.
    The row keeps the hash, which says what ran; running it again needs the
    picture. Blocking: reads the database."""
    lost = []
    for record in records:
        if record.refine is None or record.mask is None:
            continue
        try:
            storage.mask_path(record.mask)
        except ValueError:
            lost.append(record.id)
    return lost


def missing_mask_detail(ids: list[int]) -> str:
    """Why a selection with those inpaints is refused, naming each one: the
    run would otherwise be a refine of the whole picture, or fail per cell."""
    said = "; ".join(f"the mask of #{image_id} is missing" for image_id in ids[:6])
    more = f"; and {len(ids) - 6} more" if len(ids) > 6 else ""
    return (
        f"{said}{more} - its inpaint cannot run again until the mask is back: import the "
        "bundle that carries it, or paint it again on the source"
    )


def mask_unusable(request: GenerateRequest) -> str | None:
    """Why an inpaint's mask cannot run over its source, or None: the mask is
    held and its file there (`storage.mask_path`), and the mask, the source
    and the request are one size - an inpaint runs at its source's size,
    which the mask was painted at. The source has been checked live first.
    Blocking: reads the database."""
    try:
        storage.mask_path(request.mask)
    except ValueError as exc:
        return str(exc)
    size = storage.mask_size(request.mask)
    source = storage.get_image(request.refine)
    if size is None or source is None:
        # Gone between the two reads; the cell's own check says which.
        return f"mask {request.mask[:16]} or source image {request.refine} has just gone"
    if size != (source.width, source.height):
        return (
            f"mask {request.mask[:16]} is {size[0]}x{size[1]}, but source image {request.refine} "
            f"is {source.width}x{source.height}; paint the mask on that image again"
        )
    if (request.width, request.height) != size:
        return (
            f"an inpaint runs at its source's size: source image {request.refine} is "
            f"{source.width}x{source.height}, not {request.width}x{request.height}"
        )
    return None


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
