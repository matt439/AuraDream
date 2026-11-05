"""What a bundle import gives back to its new rows: an outpaint's mask built
again from its spec, a mask recovered from the copy its staged PNG carries,
an outpaint spec or masked area an older build's merge stripped, recovered
from the staged PNG's A1111 line, and a masked area's region proved against
its mask. `bundles` imports them back by name for its merge."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from . import storage


def _rebuilt_member(work: Path, values: dict) -> Path | None:
    """A new outpaint row's mask, built again from its spec and the row's
    size and written beside the unpacked members as one, where it hashes to
    the row's mask; None where it does not (the row then counts in
    `masks_missing`, as any inpaint's lost mask does)."""
    digest = values["mask"]
    try:
        spec = storage.OutpaintSpec.parse_key(values["outpaint"])
    except ValueError:
        return None
    mask = storage.rebuilt_outpaint_mask(values["width"], values["height"], spec, digest)
    if mask is None:
        return None
    target = work / storage.MASKS_DIR / f"{digest}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    mask.save(target, format="PNG", compress_level=6)
    return target


def _recovered_outpaint(work: Path, values: dict, source: dict | None):
    """An outpaint spec an older build's merge stripped from a new row,
    from its staged PNG's A1111 line (`storage.recovered_outpaint`), or None."""
    if source is None:
        return None
    staged = work / "images" / values["filename"]
    try:
        with Image.open(staged) as image:
            parameters = image.info.get("parameters")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    return storage.recovered_outpaint(
        parameters, values["width"], values["height"], values["mask"], (source["width"], source["height"])
    )


def _staged_parameters(work: Path, values: dict) -> Any:
    """A new row's staged PNG's A1111 line, or None."""
    staged = work / "images" / values["filename"]
    try:
        with Image.open(staged) as image:
            return image.info.get("parameters")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _mask_picture(work: Path, digest: str, embedded: dict[str, Path], outputs: Path) -> Image.Image | None:
    """A new row's mask as a picture, from its unpacked member, the copy
    recovered from its PNG, or this gallery's own file - each verified to be
    the mask its name says - or None. Read only to prove (or draw again) a
    masked area's region."""
    for path in (work / storage.MASKS_DIR / f"{digest}.png", embedded.get(digest),
                 outputs / storage.MASKS_DIR / f"{digest}.png"):
        if path is not None and path.is_file():
            mask = storage.open_mask(path, digest)
            if mask is not None:
                return mask
    return None


def _mask_crop(work: Path, values: dict, embedded: dict[str, Path], outputs: Path) -> tuple[bool, bool]:
    """A new row's masked area, settled in `values`: never without its source
    and mask; its region proved against the mask under the frozen rule where
    the mask is here (drawn again where none was recorded); recovered from
    the A1111 line an older build's import left, its region drawn from the
    mask, where the column was stripped. Answers (recovered, mismatched): a
    recorded region that is not its mask's is the caller's refusal."""
    if values.get("mask") is None or values.get("refine") is None:
        values["mask_crop"] = values["mask_crop_box"] = None
        return False, False
    key = storage.stored_mask_crop(values.get("mask_crop"))
    box = storage.stored_mask_crop_box(values.get("mask_crop_box")) if key is not None else None
    values["mask_crop"], values["mask_crop_box"] = key, box
    parameters = _staged_parameters(work, values) if key is None else None
    if key is None and storage.parameters_mask_crop(parameters) is None:
        return False, False
    painted = _mask_picture(work, values["mask"], embedded, outputs)
    if key is None:
        recovered = storage.recovered_mask_crop(parameters, painted)
        if recovered is None:
            return False, False
        values["mask_crop"], values["mask_crop_box"] = recovered[0].key(), storage.mask_crop_box(recovered[1])
        return True, False
    if painted is None:
        return False, False
    recorded = storage.parse_mask_crop_box(box) if box is not None else None
    proven = storage.proven_mask_crop(storage.MaskCropSpec.parse_key(key), painted, recorded)
    if proven is None:
        return False, True
    values["mask_crop_box"] = storage.mask_crop_box(proven)
    return False, False


def _embedded_member(work: Path, values: dict) -> Path | None:
    """A new row's mask, from the copy its own staged PNG carries, written
    beside the unpacked members as one; None where the PNG holds no copy of
    the mask its row names, at the row's size. A whole-gallery or selection
    bundle carries only the masks whose files were there, so a mask lost in
    the gallery it came from travels only this way - as `reindex` finds it."""
    digest = values["mask"]
    target = work / storage.MASKS_DIR / f"{digest}.png"
    if target.is_file():
        return target
    staged = work / "images" / values["filename"]
    try:
        with Image.open(staged) as image:
            mask = storage.embedded_mask_picture(image, digest)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if mask is None or mask.size != (values["width"], values["height"]):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    mask.save(target, format="PNG", compress_level=6)
    return target
