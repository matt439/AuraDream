"""An outpaint's pure parts: the padded source, the mask, the presets.

An outpaint extends a gallery image's canvas and paints the new area. In
this app it is an inpaint over a padded copy of the source
(`docs/plan-outpainting.md`): the picture handed to the inpaint class is
the source placed on a larger canvas whose new area holds a fill derived
from the source's own edge (`padded`), and the mask is built here from the
canvas size, the padding and the band (`mask`) rather than painted.

Both builders are pure functions of their arguments with integer
arithmetic only, so the same source and spec give the same bytes on every
machine; `tests/test_outpaint.py` pins the SHA-256 of each. **Each fill in
`FILLS` and `mask` are frozen**: a change to a fill is a new `FILL` name
(the old one may stay in `FILLS`, keeping its outpaints reproducible), and
a change to the mask is a change to every outpaint.

A leaf: numpy, Pillow and `sizes` only. `presets` takes the refine guard's
limits as arguments, which its callers read from the settings; the stamp
that stores a mask lives in `storage` (`stamp_outpaints`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .sizes import SIZES

if TYPE_CHECKING:
    from PIL.Image import Image

# Every side of an outpaint is on the request's 16-pixel grid, which keeps
# the source on whole latent cells in both enabled families.
GRID = 16

# The band is on the 8-pixel latent grid, so its inner edge is a cell edge.
BAND_GRID = 8

# The band's starting width: this app's value, not a sourced or measured one.
BAND = 32

# The widest band a request may ask for: a constant, never a setting.
MAX_BAND = 256

# The fewest pixels each way the band must leave unrepainted on the source.
MIN_KEPT = 16

# The one fill this build has: the source's edge stretched outward, then
# three passes of a box blur of radius 16 along each axis.
FILL = "edge-box3-r16"

_RADIUS = 16
_WINDOW = 2 * _RADIUS + 1
_PASSES = 3


class Spec(Protocol):
    fill: str
    left: int
    top: int
    right: int
    bottom: int
    band: int


def _box_pass(array: np.ndarray, axis: int) -> np.ndarray:
    """One integer box blur of radius 16 along `axis`, rounding half up.

    Edge-padded by the radius, summed through a cumulative sum in int64
    with a leading zero, `(sum + 16) // 33`: the same bytes on every numpy.
    """
    pad = [(0, 0)] * array.ndim
    pad[axis] = (_RADIUS, _RADIUS)
    widened = np.pad(array, pad, mode="edge").astype(np.int64)
    sums = np.cumsum(widened, axis=axis, dtype=np.int64)
    shape = list(sums.shape)
    shape[axis] = 1
    sums = np.concatenate([np.zeros(shape, dtype=np.int64), sums], axis=axis)
    length = array.shape[axis]
    upper = np.take(sums, np.arange(_WINDOW, _WINDOW + length), axis=axis)
    lower = np.take(sums, np.arange(0, length), axis=axis)
    return ((upper - lower + _RADIUS) // _WINDOW).astype(np.uint8)


def padded_edge_box3_r16(src: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """`edge-box3-r16`: the source's edge pixels stretched outward over the
    padding, three box passes of radius 16 along the width and then three
    along the height over the whole canvas, and the source put back exactly.
    `src` is `uint8` RGB of shape (height, width, 3)."""
    stretched = np.pad(src, ((top, bottom), (left, right), (0, 0)), mode="edge")
    channels = []
    for channel in range(stretched.shape[2]):
        plane = stretched[:, :, channel]
        for _ in range(_PASSES):
            plane = _box_pass(plane, 1)
        for _ in range(_PASSES):
            plane = _box_pass(plane, 0)
        channels.append(plane)
    canvas = np.stack(channels, axis=2)
    height, width = src.shape[:2]
    canvas[top : top + height, left : left + width] = src
    return canvas


FILLS = {FILL: padded_edge_box3_r16}


def padded(source: "Image", spec: Spec) -> "Image":
    """The source on its outpainted canvas, the new area holding the fill
    `spec.fill` names. Raises ValueError on a fill this build lacks - the
    backstop behind the submit check, which names it first."""
    from PIL import Image as PILImage

    fill = FILLS.get(spec.fill)
    if fill is None:
        raise ValueError(unknown_fill(spec.fill))
    src = np.asarray(source.convert("RGB"), dtype=np.uint8)
    return PILImage.fromarray(fill(src, spec.left, spec.top, spec.right, spec.bottom), "RGB")


def unknown_fill(name: str) -> str:
    """The refusal of a fill this build does not have, by name."""
    return f"this build has no outpaint fill '{name}'; it fills with {FILL}"


def source_size(width: int, height: int, spec: Spec) -> tuple[int, int]:
    """The source's size inside a canvas of `width` x `height`."""
    return width - spec.left - spec.right, height - spec.top - spec.bottom


def kept_size(source_width: int, source_height: int, spec: Spec) -> tuple[int, int]:
    """The rectangle of the source the band leaves unrepainted: the source
    inset by the band on each of its padded sides."""
    band = spec.band
    return (
        source_width - band * (spec.left > 0) - band * (spec.right > 0),
        source_height - band * (spec.top > 0) - band * (spec.bottom > 0),
    )


def band_too_wide(source_width: int, source_height: int, spec: Spec, name: str = "the source") -> str | None:
    """Why the band leaves too little of the source unrepainted, or None:
    the short axis, the padded sides the band is taken off and what it
    leaves, against `MIN_KEPT`."""
    kept_width, kept_height = kept_size(source_width, source_height, spec)
    if kept_width >= MIN_KEPT and kept_height >= MIN_KEPT:
        return None
    if kept_width < MIN_KEPT:
        axis, kept, named = "width", kept_width, (("left", spec.left), ("right", spec.right))
    else:
        axis, kept, named = "height", kept_height, (("top", spec.top), ("bottom", spec.bottom))
    padded_sides = [side for side, pad in named if pad > 0]
    plural = "s" if len(padded_sides) > 1 else ""
    return (
        f"a band of {spec.band} px on the {' and '.join(padded_sides)} side{plural} leaves "
        f"{max(kept, 0)} px of {name}'s {axis} unrepainted; at least {MIN_KEPT} must be kept, "
        "so narrow the band"
    )


def mask(width: int, height: int, spec: Spec) -> "Image":
    """The outpaint's mask, mode `L` at the canvas's size, white repainted.

    255 outside the source's rectangle; inside it, for each padded side, a
    linear ramp `(255 * (band - d) + band // 2) // band` over the band, `d`
    the pixel's distance from that edge of the source (0 at its first row or
    column), and the most of them where two bands meet; 0 past every band.
    A side with no padding has no band, and `band = 0` is a hard edge."""
    from PIL import Image as PILImage

    sw, sh = source_size(width, height, spec)
    if sw <= 0 or sh <= 0:
        raise ValueError(f"a {width}x{height} canvas holds no source inside +{_sides(spec)}")
    levels = np.full((height, width), 255, dtype=np.uint8)
    inside = np.zeros((sh, sw), dtype=np.int64)
    band = spec.band
    if band > 0:
        columns = np.arange(sw, dtype=np.int64)
        rows = np.arange(sh, dtype=np.int64)
        for padded_side, distance, axis in (
            (spec.left, columns, 1),
            (spec.right, sw - 1 - columns, 1),
            (spec.top, rows, 0),
            (spec.bottom, sh - 1 - rows, 0),
        ):
            if padded_side <= 0:
                continue
            ramp = np.where(distance < band, (255 * (band - distance) + band // 2) // band, 0)
            inside = np.maximum(inside, ramp[None, :] if axis == 1 else ramp[:, None])
    levels[spec.top : spec.top + sh, spec.left : spec.left + sw] = inside.astype(np.uint8)
    return PILImage.fromarray(levels, "L")


def _sides(spec: Spec) -> str:
    return f"{spec.left}/+{spec.top}/+{spec.right}/+{spec.bottom}"


def sides(spec: Spec) -> str:
    """The padding as the page and the CLI say it: `+400/+0/+400/+0`."""
    return f"+{_sides(spec)}"


def source_off_grid(width: int, height: int, name: str = "the source") -> str | None:
    """Why no outpaint of a `width` x `height` source can run, or None: its
    canvas would be off the 16-pixel grid whatever the padding."""
    if width % GRID or height % GRID:
        return f"{name} is {width} x {height}, off the 16-px grid, so no outpaint of it is either"
    return None


def _split(pad: int) -> tuple[int, int]:
    """A padding on one axis, split evenly: the first side takes
    `pad // 32 * 16`, the second the rest (the odd 16 to the right or bottom)."""
    first = pad // (2 * GRID) * GRID
    return first, pad - first


def _ratio(aspect: str) -> tuple[int, int]:
    a, _, b = aspect.partition(":")
    return int(a), int(b)


def _preset_tier(aspect: str, width: int, height: int) -> float | None:
    for tier, size in SIZES[aspect].items():
        if size == (width, height):
            return tier
    return None


def presets(
    width: int, height: int, *, max_side: int, max_pixels: int, name: str = "the source"
) -> list[dict[str, Any]]:
    """Each preset's padding over a `width` x `height` source, in integer
    arithmetic: "+25%" first, then a "to a:b" for every aspect of
    `sizes.SIZES` in the table's order. Each answers its four sides, the
    canvas, and `held` - the words that hold it, or None.

    "to a:b" extends only the axis short of the ratio and never crops; it is
    held where the source is that aspect's preset at some tier, or where its
    whole padding would be under 32 px. "+25%" grows each axis by
    `max(32, (side + 32) // 64 * 16)`, split evenly. Any preset past the
    refine guard's side or pixels is held with the size it would be; over an
    off-grid source every one is held with `source_off_grid`'s words.
    """
    off_grid = source_off_grid(width, height, name)
    answers = []

    def add(preset: str, label: str, pads: tuple[int, int, int, int], held: str | None) -> None:
        left, top, right, bottom = pads
        canvas_w, canvas_h = width + left + right, height + top + bottom
        if off_grid is not None:
            held = off_grid
        elif held is None and (max(canvas_w, canvas_h) > max_side or canvas_w * canvas_h > max_pixels):
            held = (
                f"{label} is {canvas_w} x {canvas_h}, past the {max_side}-pixel side or "
                f"{max_pixels:,} pixels a refine pass takes"
            )
        answers.append({
            "name": preset, "label": label, "left": left, "top": top, "right": right, "bottom": bottom,
            "width": canvas_w, "height": canvas_h, "held": held,
        })

    across = _split(max(2 * GRID, (width + 2 * GRID) // 64 * GRID))
    down = _split(max(2 * GRID, (height + 2 * GRID) // 64 * GRID))
    add("+25%", "+25% all round", (across[0], down[0], across[1], down[1]), None)
    for aspect in SIZES:
        a, b = _ratio(aspect)
        if width * b < height * a:
            # Narrower than a:b: widen to the smallest multiple of 16 at or
            # above height * a / b.
            target = -(-height * a // (b * GRID)) * GRID
            first, second = _split(target - width)
            pads = (first, 0, second, 0)
        else:
            target = -(-width * b // (a * GRID)) * GRID
            first, second = _split(target - height)
            pads = (0, first, 0, second)
        total = sum(pads)
        tier = _preset_tier(aspect, width, height)
        held = None
        if tier is not None:
            held = f"{name} is already {aspect} - the {aspect} preset at {tier:g} MP"
        elif total < 2 * GRID:
            held = (
                f"{name} is already {aspect}" if total == 0 else f"{name} is within {total} px of {aspect}"
            )
        add(aspect, f"to {aspect}", pads, held)
    return answers


def preset(width: int, height: int, which: str, *, max_side: int, max_pixels: int,
           name: str = "the source") -> dict[str, Any]:
    """One preset of `presets` by its name ("+25%" or an aspect), raising
    ValueError naming the presets there are where it is none of them."""
    for answer in presets(width, height, max_side=max_side, max_pixels=max_pixels, name=name):
        if answer["name"] == which:
            return answer
    raise ValueError(f"no outpaint preset {which!r}; one of +25%, {', '.join(SIZES)}")
