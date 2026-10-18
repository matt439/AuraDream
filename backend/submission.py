"""What every cell producer checks before it queues, and the gallery
selection Re-render, Swap tags, Mass refine and More seeds act on."""
from __future__ import annotations

import asyncio

from fastapi import HTTPException

from . import grid, loras, outpaint, storage
from .app_context import AppServices
from .schemas import GenerateRequest, ImageRecord


# The per-image status a reason from `unusable_input` becomes.
UNUSABLE_STATUS = {
    "source": "source_deleted", "lora": "lora_missing", "mask_missing": "mask_missing", "outpaint": "outpaint",
}


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
    None. The kind - `refine`, `inpaint`, `source`, `mask_missing`, `mask`,
    `reference`, `references`, `lora` - is what Swap tags reports per image (a
    deleted source, a missing adapter or mask) where a whole submission is
    refused for one; a `mask_missing` is said per image with
    `missing_mask_detail`, which names it.

    Whether a pass can run from a source splits on `request.inpainting`: with
    a mask or an outpaint it is the inpaint class's question, else the
    image-to-image class's, as `applied()` asks it. The mask file's checks
    run where there is a mask - an outpaint has none until its stamp - and
    an outpaint's own (`outpaint_unusable`, kind `outpaint`) where there is
    one."""
    if request.refine is not None:
        inpainting = request.inpainting
        if inpainting:
            reason = await inpaint_unavailable(services, outpainting=request.outpaint is not None)
        else:
            reason = await refine_unavailable(services)
        if reason is not None:
            return ("inpaint" if inpainting else "refine"), reason
        try:
            await asyncio.to_thread(storage.reference_paths, [request.refine], "source image")
        except ValueError as exc:
            return "source", str(exc)
        if request.outpaint is not None:
            reason = await asyncio.to_thread(outpaint_unusable, request)
            if reason is not None:
                return "outpaint", reason
        if request.mask is not None:
            # A mask this gallery cannot find is its own kind: the gallery
            # actions name it by the image (`missing_mask_detail`), never by
            # its hash, and the way back is the bundle as often as the brush.
            try:
                await asyncio.to_thread(storage.mask_path, request.mask)
            except ValueError as exc:
                return "mask_missing", str(exc)
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


async def inpaint_unavailable(services: AppServices, outpainting: bool = False) -> str | None:
    """Why this server has no inpaint, or None where it has one: what an
    inpaint's submission is refused with - and an outpaint's, which runs
    through the same class. The profile's answer before a load, the loaded
    class's after, as `refine_unavailable` asks."""
    info = await asyncio.to_thread(services.backend().describe)
    if services.merged_defaults(info)["inpaint_supported"]:
        return None
    through = ", which an outpaint runs through" if outpainting else ""
    return (
        f"{info['model_id']} has no inpaint in this app - its pipeline has no inpaint class "
        f"in the installed diffusers{through}, or its family's is not enabled yet. The SDXL and "
        "Z-Image checkpoints have one."
    )


def missing_masks(records) -> list[int]:
    """The ids of the inpaints among `records` whose mask this gallery cannot
    find - imported without it (a bundle's `masks_missing`), or its file lost.
    The row keeps the hash, which says what ran; running it again needs the
    picture. Blocking: reads the database."""
    lost = []
    for record in records:
        # An outpaint's mask is a function of its request, built again at
        # the submit - which heals a lost file, so it is never missing.
        if record.refine is None or record.mask is None or record.outpaint is not None:
            continue
        try:
            storage.mask_path(record.mask)
        except ValueError:
            lost.append(record.id)
    return lost


def mismatched_outpaints(records) -> list[int]:
    """The ids of the outpaints among `records` whose spec does not build the
    mask the row was made under - a bundle's `masks_missing` row, a reindexed
    chunk, another build's mask code. Run again, such a row would paint
    under another mask while keeping its key, so resume and More seeds would
    take the new image for the same reproduction set; `--from-image` refuses
    it the same way. One build per distinct canvas and spec. Blocking: builds
    masks."""
    built: dict[tuple[int, int, str], str | None] = {}
    mismatched = []
    for record in records:
        if record.refine is None or record.outpaint is None:
            continue
        key = (record.width, record.height, record.outpaint.key())
        if key not in built:
            built[key] = storage.outpaint_mask_hash(record.width, record.height, record.outpaint)
        if record.mask is None or built[key] != record.mask:
            mismatched.append(record.id)
    return mismatched


def mismatched_outpaint_detail(ids: list[int]) -> str:
    """Why a selection with those outpaints is refused, naming each one."""
    said = "; ".join(
        f"the outpaint of #{image_id} was made under a mask this build does not build from its spec"
        for image_id in ids[:6]
    )
    more = f"; and {len(ids) - 6} more" if len(ids) > 6 else ""
    return f"{said}{more} - run again, it would paint under another mask; outpaint its source again instead"


def unusable_detail(kind: str, why: str, image_id: int) -> str:
    """What a gallery action says of image `image_id` for an `unusable_input`
    finding: its own words, but a missing mask by the image it belongs to."""
    return missing_mask_detail([image_id]) if kind == "mask_missing" else why


def resized_inpaints(records, width: int, height: int) -> list[ImageRecord]:
    """The inpaints among `records` that are not `width`x`height`: an inpaint
    runs at its source's size, so none of them can be re-rendered at it."""
    return [
        record for record in records
        if record.refine is not None and record.mask is not None
        and (record.width, record.height) != (width, height)
    ]


def resized_inpaint_detail(
    records: list[ImageRecord], width: int, height: int, picked: bool, others: bool = True
) -> str:
    """Re-render's refusal of those inpaints, each named with its source, and
    the way out where the selection holds `others`: unpicking them, or
    leaving inpaints out of the filter."""
    said = "; ".join(
        f"#{record.id} is an {'outpaint' if record.outpaint is not None else 'inpaint'} of "
        f"#{record.refine} at {record.width}x{record.height}"
        for record in records[:6]
    )
    outpaints = any(record.outpaint is not None for record in records)
    rule = (
        "an outpaint runs at its source's size plus its padding"
        if outpaints and all(record.outpaint is not None for record in records)
        else "an inpaint runs at its source's size" + (
            ", and an outpaint at its source's size plus its padding" if outpaints else ""
        )
    )
    more = f"; and {len(records) - 6} more" if len(records) > 6 else ""
    names = ", ".join(f"#{record.id}" for record in records[:6]) + (" and the rest" if len(records) > 6 else "")
    way = f"Unpick {names}" if picked else "Add inpaint:no to the search"
    tail = f" {way} to re-render the others." if others else ""
    return (
        f"{said}{more} - {rule}, so it cannot be re-rendered at "
        f"{width}x{height}.{tail}"
    )


def missing_mask_detail(ids: list[int]) -> str:
    """Why a selection with those inpaints is refused, naming each one: the
    run would otherwise be a refine of the whole picture, or fail per cell."""
    said = "; ".join(f"the mask of #{image_id} is missing" for image_id in ids[:6])
    more = f"; and {len(ids) - 6} more" if len(ids) > 6 else ""
    return (
        f"{said}{more} - its inpaint cannot run again until the mask is back: import the "
        "bundle that carries it, or paint it again on the source"
    )


def outpaint_unusable(request: GenerateRequest) -> str | None:
    """Why an outpaint cannot run over its source, or None. In order: the
    source is on the 16-pixel grid, the fill is one this build has, the
    request's size is the source's plus the padding, and the band leaves a
    kept rectangle of at least 16 px each way. The source has been checked
    live first. Blocking: reads the database."""
    spec = request.outpaint
    source = storage.get_image(request.refine)
    if source is None:
        return f"source image {request.refine} has just gone"
    name = f"#{source.id}"
    off_grid = outpaint.source_off_grid(source.width, source.height, name)
    if off_grid is not None:
        return off_grid
    if spec.fill not in outpaint.FILLS:
        return outpaint.unknown_fill(spec.fill)
    width, height = source.width + spec.left + spec.right, source.height + spec.top + spec.bottom
    if (request.width, request.height) != (width, height):
        return (
            f"an outpaint of {name} ({source.width}x{source.height}) by {spec.sides()} is "
            f"{width}x{height}, not the {request.width}x{request.height} asked for"
        )
    return outpaint.band_too_wide(source.width, source.height, spec, name)


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


async def rendered_keys(
    services: AppServices, cells: list[grid.Cell], width: int, height: int
) -> set[grid.CellKey]:
    """Gallery keys at one size, including binned rows, off the event loop."""
    return await asyncio.to_thread(
        storage.completed_cells, grid.prompts_of(cells), width, height,
        services.model_id(), services.quantization(),
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
    done = await rendered_keys(services, cells, width, height)
    return [
        cell
        for cell in cells
        if grid.prune_done([grid.Cell(request=services.backend().applied(cell.request), seed=cell.seed)], done)
    ]


def not_yet_planned(
    services: AppServices,
    cells: list[grid.Cell],
    planned: dict[tuple[int, int], set[grid.CellKey]],
) -> list[grid.Cell]:
    """The cells no live job will write: what Resume and gallery reruns
    run after `not_yet_rendered`, so another press while the same cells
    are still queued plans none of them twice.

    Opt-in, never inside `not_yet_rendered`: its result and the callers'
    rendered counts are claims about the gallery, with queued cells counted
    separately. Keyed
    as `planned_keys` keys them, through `applied()` and by each cell's own
    size; a live job runs under the server's checkpoint and quantizer, the
    scope `completed_cells` reads.

    `planned` is a `planned_keys` snapshot taken on the event loop *before*
    the gallery read awaits: a cell's keys only move from the queue to the
    gallery, so a snapshot taken first covers every cell the scan misses -
    one taken after it would miss a cell whose job finished mid-scan.
    """
    if not planned:
        return cells
    return [
        cell
        for cell in cells
        if grid.prune_done(
            [grid.Cell(request=services.backend().applied(cell.request), seed=cell.seed)],
            planned.get((cell.request.width, cell.request.height), set()),
        )
    ]


async def resolve_selection(
    services: AppServices, *, ids: list[int] | None, search: str | None, favorites: bool, all_models: bool,
    run: str | None = None,
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

    With `run` (More seeds of a run), every live image of that run, every
    checkpoint's, whatever the filter and Starred: 404 where the gallery
    holds none of it, or holds it only in the recycle bin. Not capped: what
    a press may queue is the caller's to bound.
    """
    if run is not None:
        rows = await asyncio.to_thread(storage.lineage_images, [run], [])
        records = [record for record in rows if record.binned_at is None]
        if not records:
            raise HTTPException(status_code=404, detail=(
                f"every image of run {run} is in the recycle bin" if rows
                else f"run {run} is not in the gallery"
            ))
        return records, [], set()
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
