"""The pure field codecs gallery rows and PNG chunks share.

A PNG chunk's mask, upscaler, batch and peak sanitisers; a mask's hash and its
`mask_source` provenance; an outpaint's chunk, its A1111 line, and recovering
or rebuilding one from either. None reads settings, a session or a lock.
`storage` imports every name back, so `storage.<name>` still answers for the
reindex, the bundle layer and the CLI; the mask writers, holds and
`mask_path` stay in storage, where tests patch them.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from PIL import Image

from . import refine
from .schemas import OutpaintSpec


_MASK_HASH = re.compile(r"[0-9a-f]{64}")


def _mask(value: Any) -> str | None:
    """A PNG chunk's `mask`, where it is a mask hash: the same shape the
    request's field takes, so a hand-written block cannot put a path, or
    anything else `mask_path` was never meant to look up, on a row."""
    return value if isinstance(value, str) and _MASK_HASH.fullmatch(value) else None


def _upscaler(
    value: Any, *, source: int | None, model_id: str | None,
    mask: str | None = None, outpaint: Any = None,
) -> str | None:
    """A known GAN only beside an ordinary pass of a documented checkpoint.

    PNG recovery accepts older records and handwritten metadata: an unknown
    choice or one without its source cannot become an executable setting.
    """
    return (
        refine.ANIME6B
        if value == refine.ANIME6B and type(source) is int and source > 0 and mask is None and outpaint is None
        and isinstance(model_id, str) and refine.upscaler_supported(model_id)
        else None
    )


def _batch(value: Any) -> int | None:
    """A PNG chunk's `batch`, where it is one: a block written by hand or by
    another tool says what it likes, and a row is not refused for the one
    field that is no part of the reproduction set."""
    return value if type(value) is int and value >= 1 else None


def _peak(value: Any) -> int | None:
    """A peak's bytes, where it is a count of them; anything else - a block
    written by hand, a string - is None, never a refused row, as `_batch`."""
    return value if type(value) is int and value >= 0 else None


def peak_chunk(row: Mapping[str, Any]) -> dict:
    """The peak keys as `save_image` writes them: absent where unmeasured."""
    return {
        key: _peak(row.get(key))
        for key in ("vram_peak_allocated", "vram_peak_reserved")
        if _peak(row.get(key)) is not None
    }


def mask_hash(mask: Image.Image) -> str:
    """The name a mask is stored and asked for by: all 64 hex digits of the
    SHA-256 of its size and its decoded `L` levels.

    Of the pixels, not the file, as `content_hash` is - the encoder's settings
    and the chunks would otherwise split one painting into two names - and the
    whole digest, since a mask is an input looked up by this name, which the
    16-digit `content_hash` is too short to be. The size is hashed with the
    levels because the levels alone do not say it: an all-white 832x1216 and
    an all-white 1216x832 are the same bytes and different masks.
    """
    if mask.mode != "L":
        raise ValueError(f"a mask is one luminance channel, not {mask.mode}")
    digest = hashlib.sha256(f"L {mask.width}x{mask.height}\n".encode("ascii"))
    digest.update(mask.tobytes())
    return digest.hexdigest()


def mask_provenance(mask: str | None, width: int, height: int) -> dict[str, Any] | None:
    """A result's `mask_source` in its PNG chunk: the mask's file, its hash
    and its size - which is the result's own, since an inpaint runs at its
    source's size - or None where the row is no inpaint."""
    if mask is None:
        return None
    return {"filename": f"{mask}.png", "hash": mask, "width": width, "height": height}


def outpaint_chunk(spec: OutpaintSpec | None) -> dict[str, int] | None:
    """An outpaint's `outpaint` key in a result's PNG chunk: its padding and
    band (the fill rides beside it as `outpaint_fill`), or None."""
    if spec is None:
        return None
    return {"left": spec.left, "top": spec.top, "right": spec.right, "bottom": spec.bottom, "band": spec.band}


def outpaint_parameters(spec: OutpaintSpec) -> str:
    """The A1111 line's outpaint, slashes where A1111 would split on a comma:
    the one copy of the spec an older build's import leaves in place."""
    return (
        f", Outpaint: {spec.left}/{spec.top}/{spec.right}/{spec.bottom}, "
        f"Outpaint band: {spec.band}, Outpaint fill: {spec.fill}"
    )


def _canonical_outpaint(text: str | None) -> str | None:
    """A column's outpaint key in `OutpaintSpec.key()`'s one spelling - a
    row an older import stored as given (`0400`) keys as its re-render
    does - or the text itself where it is no key."""
    if not text:
        return text
    try:
        return OutpaintSpec.parse_key(text).key()
    except ValueError:
        return text


def chunk_outpaint(meta: dict[str, Any]) -> OutpaintSpec | None:
    """A PNG chunk's `outpaint` and `outpaint_fill` as the stored spec, or
    None where it carries none or a malformed one. Any well-formed fill is
    kept, known to this build or not: the submit refuses an unknown one by
    name, so the row still shows as what it is."""
    value, fill = meta.get("outpaint"), meta.get("outpaint_fill")
    if not isinstance(value, dict) or not isinstance(fill, str):
        return None
    sides: list[int] = []
    for name in ("left", "top", "right", "bottom", "band"):
        side = value.get(name)
        if type(side) is not int:
            return None
        sides.append(side)
    try:
        spec = OutpaintSpec(fill=fill, left=sides[0], top=sides[1], right=sides[2], bottom=sides[3], band=sides[4])
    except ValueError:
        return None
    return spec if spec.wellformed() else None


def malformed_outpaint(meta: dict[str, Any]) -> bool:
    """Whether a chunk names an outpaint that `chunk_outpaint` cannot read."""
    return meta.get("outpaint") is not None and chunk_outpaint(meta) is None


_PARAMETERS_OUTPAINT = re.compile(
    r"Outpaint: (\d{1,6})/(\d{1,6})/(\d{1,6})/(\d{1,6}), Outpaint band: (\d{1,6}), "
    r"Outpaint fill: ([a-z0-9][a-z0-9-]{0,31})(?:,|$|\s)"
)


def outpaint_mask_hash(width: int, height: int, spec: OutpaintSpec) -> str | None:
    """The hash of the mask an outpaint of this canvas and spec builds, or
    None where the canvas cannot hold it."""
    from . import outpaint

    try:
        return mask_hash(outpaint.mask(width, height, spec))
    except ValueError:
        return None


def recovered_outpaint(
    parameters: Any, width: int, height: int, mask: str | None, source_size: tuple[int, int] | None,
) -> OutpaintSpec | None:
    """An outpaint an older build's import stripped, from the A1111 line it
    left in place - exact or nothing: only where the row has a mask, the
    canvas is the source's size plus the parsed padding (and so not the
    source's own), and the mask this build builds for it hashes to the
    row's."""
    if not isinstance(parameters, str) or mask is None or source_size is None:
        return None
    if (width, height) == tuple(source_size):
        return None
    match = _PARAMETERS_OUTPAINT.search(parameters)
    if match is None:
        return None
    left, top, right, bottom, band = (int(value) for value in match.groups()[:5])
    try:
        spec = OutpaintSpec(fill=match.group(6), left=left, top=top, right=right, bottom=bottom, band=band)
    except ValueError:
        return None
    if not (spec.left or spec.top or spec.right or spec.bottom):
        return None
    if (width - left - right, height - top - bottom) != tuple(source_size):
        return None
    return spec if outpaint_mask_hash(width, height, spec) == mask else None


def rebuilt_outpaint_mask(width: int, height: int, spec: OutpaintSpec, digest: str) -> Image.Image | None:
    """An outpaint's mask built again from its spec and the row's size,
    where it hashes to the row's `digest`; None where it does not."""
    from . import outpaint

    try:
        built = outpaint.mask(width, height, spec)
    except ValueError:
        return None
    return built if mask_hash(built) == digest else None
