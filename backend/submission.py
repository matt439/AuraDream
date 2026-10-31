"""What every cell producer checks before it queues, and the gallery
selection Re-render, Swap tags, Mass refine and More seeds act on."""
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import HTTPException

from . import grid, instruction_edit, instruction_edit_admission, loras, refine, storage
from .app_context import AppServices
from .execution import configured_loader_unavailable_reason, unavailable_reason
from .model_selection import current_settings
from .schemas import GenerateRequest, ImageRecord

# The checks and wording that need no services, kept where the CLI can
# import them without the HTTP layer; re-exported, so `submission.X`
# still names each.
from .submission_rules import (
    cells_by_size as cells_by_size,
    edit_rerender_detail as edit_rerender_detail,
    gone_detail as gone_detail,
    mask_unusable as mask_unusable,
    mismatched_outpaint_detail as mismatched_outpaint_detail,
    mismatched_outpaints as mismatched_outpaints,
    missing_mask_detail as missing_mask_detail,
    missing_masks as missing_masks,
    outpaint_unusable as outpaint_unusable,
    resized_inpaint_detail as resized_inpaint_detail,
    resized_inpaints as resized_inpaints,
    unusable_detail as unusable_detail,
)


# The per-image status a reason from `unusable_input` becomes.
UNUSABLE_STATUS: dict[str, Literal["source_deleted", "lora_missing", "mask_missing", "outpaint"]] = {
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


async def unusable_input(
    services: AppServices, request: GenerateRequest, producer: Literal["generate", "more_seeds"] = "generate",
) -> tuple[str, str] | None:
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
    one.

    An edit (kind `edit`) is said by its `producer`. More seeds runs one:
    the recorded proof against the source's pixels now (`bind_request`),
    then the gate, the exact model and the loader envelope, then the
    source's aspect - None only where all pass. Every other producer
    (Generate, Swap tags, Finish) never queues one, whatever the gate: a
    closed one says the editor is unavailable, an unmeasured loader says
    so, else where edits are queued from (`replay_held_reason`)."""
    if request.edit is not None:
        try:
            bound = await asyncio.to_thread(instruction_edit_admission.bind_request, request, services.model_id())
        except ValueError as exc:
            return "edit", str(exc)
        refusal = services.backend().instruction_edit_refusal(current_settings(services.settings()))
        if producer != "more_seeds":
            return "edit", instruction_edit.replay_held_reason(refusal)
        reason = refusal or instruction_edit.bound_geometry_refusal(bound.edit)
        return None if reason is None else ("edit", reason)
    reason = unavailable_reason(request.execution)
    if reason is not None:
        return "execution", reason
    selected = current_settings(services.settings())
    reason = configured_loader_unavailable_reason(selected)
    if reason is not None:
        return "execution", reason
    try:
        request = request.with_execution(selected.loader_execution)
    except ValueError as exc:
        return "execution", str(exc)
    reason = unavailable_reason(request.execution)
    if reason is not None:
        return "execution", reason
    from .model_defaults import defaults_for

    if not defaults_for(selected.model_id)["text_to_image_supported"]:
        backend = services.backend()
        return "edit", instruction_edit.text_to_image_refusal(
            selected.model_id, qualified=backend.instruction_edit_qualified() is True,
            loader=backend.instruction_edit_loader(selected),
        )
    reason = refine.upscaler_reason(request, services.model_id())
    if reason is not None:
        return "upscaler", reason
    if request.upscaler is not None:
        assert request.refine is not None, "a validated upscaler request names its source"
        try:
            await asyncio.to_thread(refine.check_upscaler_source, request.refine)
        except ValueError as exc:
            return "upscaler", str(exc)
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
        f"in the installed diffusers{through}, or its family's is not enabled yet. The FLUX.1, SDXL "
        "and Z-Image checkpoints have one."
    )


async def rendered_keys(
    services: AppServices, cells: list[grid.Cell], width: int, height: int
) -> set[grid.CellKey]:
    """Gallery keys at one size, including binned rows, off the event loop."""
    return await asyncio.to_thread(
        storage.completed_cells, grid.prompts_of(cells), width, height,
        services.model_id(), services.quantization(),
    )


async def not_yet_rendered(
    services: AppServices, cells: list[grid.Cell], width: int | None = None, height: int | None = None
) -> list[grid.Cell]:
    """The cells the gallery does not hold at each actual size: what Resume runs.

    Compare what the pipeline would *run*, not what the panel says: the rows
    were written from `backend.applied()`, so a negative prompt a Turbo cell
    at CFG 0 never saw is None on both sides rather than on one. Off the
    loop: up to three chunked scans, fired by the preview on every keystroke.
    """
    done: set[grid.CellKey] = set()
    for (actual_width, actual_height), group in cells_by_size(cells).items():
        done.update(await rendered_keys(services, group, actual_width, actual_height))
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



async def prune_held(services: AppServices, cells: list[grid.Cell]) -> tuple[set[int], set[int]]:
    """The ids (`id(cell)`) of `cells` the gallery does not hold, and of
    those the ids no live job plans either: what a gallery rerun queues.

    Per size, since `completed_cells` is scoped to one. The queue snapshot
    is taken first, as `not_yet_planned` requires; the caller holds rerun
    admission until every job it makes of them is added.
    """
    # The gallery's own answer, per size: `completed_cells` is scoped to one.
    sizes = cells_by_size(cells)
    # Snapshot before any size's awaited scan: a live cell can finish
    # during it. Submission holds rerun admission until every job is added.
    planned = services.jobs().planned_keys()
    unheld: set[int] = set()
    kept: set[int] = set()
    for (width, height), group in sizes.items():
        missing = await not_yet_rendered(services, group, width, height)
        unheld.update(id(cell) for cell in missing)
        kept.update(id(cell) for cell in not_yet_planned(services, missing, planned))
    return unheld, kept


# What `input_spec` answers for a request that names no input to check.
NO_INPUTS: tuple = (None, (), None, None, None, None, None)


def input_spec(request: GenerateRequest) -> tuple:
    """The inputs `unusable_input` checks, as one hashable key: adapter,
    references, refine source, mask, outpaint, upscaler and edit. Producers
    check each distinct key once, and skip `NO_INPUTS`; one tuple, so a new
    input cannot be left out of one producer's key and shared there by two
    requests that differ in it."""
    return (
        request.lora, tuple(request.reference), request.refine, request.mask, request.outpaint,
        request.upscaler, request.edit,
    )


def status_counts(results) -> dict[str, int]:
    """How many of a rerun preview's `results` carry each status, in the
    order each status first appears."""
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


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
