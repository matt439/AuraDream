"""The submit checks and refusal wording that need no services: what a
cell's inpaint mask or outpaint must hold to run, the selections a rerun
refuses and how each is said, and cells grouped by their size.

`submission` re-exports every name, which the routes call it by; the CLI
(`generate.py`) imports this module instead, so it reaches the server's
own checks without importing FastAPI or `AppServices`.
"""
from __future__ import annotations

from . import grid, outpaint, storage
from .schemas import GenerateRequest, ImageRecord


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


def edit_rerender_detail(records: list[ImageRecord], picked: bool, others: bool = True) -> str:
    """Re-render's refusal of those instruction edits, each named with its
    size, and the way out: an edit's size is frozen by its source's
    geometry, so at its own size it is already rendered and at any other it
    is not that edit. A selection with others is refused whole, as
    `resized_inpaint_detail`'s, with the way to leave them out."""
    sizes = "; ".join(f"{record.width}x{record.height} for #{record.id}" for record in records[:6])
    more = f"; and {len(records) - 6} more" if len(records) > 6 else ""
    rule = (
        f"An instruction edit renders at its source's frozen size ({sizes}{more})" if len(records) == 1
        else f"Instruction edits render at their sources' frozen sizes ({sizes}{more})"
    )
    names = ", ".join(f"#{record.id}" for record in records[:6]) + (" and the rest" if len(records) > 6 else "")
    way = f"Unpick {names}" if picked else "Add edit:no to the search"
    tail = f" {way} to re-render the others." if others else ""
    return f"{rule}; use More seeds for new seeds, or Reuse to edit again.{tail}"


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
    assert spec is not None and request.refine is not None, "an outpaint names its spec and source"
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
    assert request.mask is not None and request.refine is not None, "an inpaint names its mask and source"
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


def gone_detail(image_id: int, binned: set[int]) -> str:
    """Why a picked id is `deleted`: binned since it was picked, or gone."""
    return "in the recycle bin" if image_id in binned else "no longer in the gallery"


def cells_by_size(cells: list[grid.Cell]) -> dict[tuple[int, int], list[grid.Cell]]:
    sizes: dict[tuple[int, int], list[grid.Cell]] = {}
    for cell in cells:
        sizes.setdefault((cell.request.width, cell.request.height), []).append(cell)
    return sizes
