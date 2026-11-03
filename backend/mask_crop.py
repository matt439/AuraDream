"""Only the masked area: the region an inpaint or outpaint denoises, and the
way its result goes back (TODO.md INP#9).

An ordinary inpaint runs the whole source at its own size. Cropped, the app
cuts a box around the mask - its extent plus a padding - resizes that box to
about one megapixel (`region`), runs the inpaint class on it alone
(`crop_inputs`), and pastes the result back over the source (`paste_back`),
before the unchanged `refine.composite` puts every unmasked pixel back
exactly. A small region gets the checkpoint's full detail; a large picture
stays within the card. The app crops, never the class: diffusers'
`padding_mask_crop` is not passed, so the region is this module's and runs
the same on every inpaint family.

**The rule `bbox16` is frozen.** `region` is a pure function of the painted
mask (content-addressed by its hash) and the spec, with integer arithmetic
only, so the same mask and spec give the same box and working size on every
machine; `tests/test_mask_crop.py` pins the SHA-256 of a sweep of them. A
changed rule is a new rule name, as a changed outpaint fill is a new fill.

The box and working size are results, recorded for display and pricing and
never part of a key: the spec's `key()` is. A leaf: Pillow, numpy, `sizes`
and `refine` (itself a leaf), none of the package's settings; only
`request_region` reads a stored mask, through `refine.mask_image`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import numpy as np

from . import refine
from .sizes import largest_tier_pixels

if TYPE_CHECKING:
    from PIL.Image import Image

# The one rule this build has: the mask's bounding box plus the padding,
# snapped outward to the 16-pixel grid, its aspect held within 12:5, at the
# tier's pixels in the box's own aspect.
RULE = "bbox16"
RULES = (RULE,)

# The padding's default, step and bound, in source (or canvas) pixels. No
# padding has been measured on the card: 32 is the default's guess.
PAD = 32
PAD_GRID = 8
MAX_PAD = 256

# The one working tier: about one megapixel (`sizes.largest_tier_pixels(1)`).
TIER = 1
TIERS = (TIER,)

# The grid the box and the working size are on: the request's size grid,
# which keeps both on whole latent cells.
GRID = 16

# The widest a box may be before its short side grows: 12:5 either way.
ASPECT_LONG, ASPECT_SHORT = 12, 5

# The least working side.
MIN_SIDE = 256

# Pillow's Lanczos reaches 3 pixels of the smaller side's spacing.
_LANCZOS_SUPPORT = 3

_POSITIVE = [0] + [255] * 255


class Spec(Protocol):
    # Read-only, so a frozen spec's narrower literals satisfy it.
    @property
    def rule(self) -> str: ...
    @property
    def pad(self) -> int: ...
    @property
    def tier(self) -> int: ...


class Crop(NamedTuple):
    """What `region` decides: the box on the source, `(x1, y1, x2, y2)` with
    the right and bottom edges exclusive, and the working `(width, height)`
    the class runs at."""

    box: tuple[int, int, int, int]
    working: tuple[int, int]

    @property
    def box_size(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return x2 - x1, y2 - y1

    def scale(self) -> float:
        """How much larger the working size is than the box, along its
        longer stretch: above 1 is more detail, below 1 is less memory."""
        rw, rh = self.box_size
        return max(self.working[0] / rw, self.working[1] / rh)


class CropInputs(NamedTuple):
    """What the inpaint class is handed for a cropped run: the source's box
    at the working size, the painted mask there thresholded and widened to
    cover the paste back's reach (`mask`), and that as the class's mask
    (`refine.pipeline_mask`)."""

    image: "Image"
    mask: "Image"
    pipeline_mask: "Image"


def _ceil_to(value: int, grid: int) -> int:
    return -(-value // grid) * grid


def _nearest_16ths(numerator: int, denominator: int) -> int:
    """The integer nearest `sqrt(numerator / denominator) / 16`, halves up,
    in integers only: the largest `n` with `(2n - 1)^2 * 64 * d <= num`."""
    root = math.isqrt(numerator // (64 * denominator))
    return (root + 1) // 2


def _grow(low: int, high: int, length: int, limit: int) -> tuple[int, int]:
    """`[low, high)` grown to `length` about its middle on the 16 grid,
    shifted back inside `[0, limit)`, and clamped only where the image runs
    out."""
    if high - low >= length:
        return low, high
    if length >= limit:
        return 0, limit
    extra = length - (high - low)
    low -= extra // (2 * GRID) * GRID
    high = low + length
    if low < 0:
        low, high = 0, length
    if high > limit:
        low, high = limit - length, limit
    return low, high


def working_size(box_width: int, box_height: int, tier: int = TIER) -> tuple[int, int]:
    """The working size for a box: the tier's pixels in the box's own aspect,
    each side on the 16 grid and at least `MIN_SIDE`. Never a preset: a
    preset's aspect would stretch the box."""
    if tier not in TIERS:
        raise ValueError(f"{tier} is not a working tier; this build has {', '.join(map(str, TIERS))}")
    pixels = largest_tier_pixels(tier)
    width = _nearest_16ths(pixels * box_width, box_height) * GRID
    height = _nearest_16ths(pixels * box_height, box_width) * GRID
    return max(width, MIN_SIDE), max(height, MIN_SIDE)


def region(painted: "Image", spec: Spec) -> Crop:
    """The box `spec` crops around `painted`'s painted (> 0) pixels and the
    working size it runs at, under rule `bbox16`:

    1. the bounding box of every pixel above 0;
    2. plus `pad` on each side, clamped to the image;
    3. snapped outward to the 16 grid, clamped to the image;
    4. the short side grown to hold the aspect within 12:5, shifted back
       inside the image, clamped only where the image runs out;
    5. the tier's pixels in the box's own aspect (`working_size`).

    Raises ValueError for a rule or tier this build lacks, a picture off the
    16 grid, or a mask that paints nothing.
    """
    if spec.rule not in RULES:
        raise ValueError(f"this build has no masked-area rule {spec.rule!r}; it crops with {RULE}")
    if not (0 <= spec.pad <= MAX_PAD) or spec.pad % PAD_GRID:
        raise ValueError(f"a masked-area padding is 0 to {MAX_PAD} in steps of {PAD_GRID}, not {spec.pad}")
    width, height = painted.size
    if width % GRID or height % GRID:
        raise ValueError(f"a {width}x{height} picture is off the {GRID}-pixel grid a masked area is cut on")
    levels = painted if painted.mode == "L" else painted.convert("L")
    bbox = levels.point(_POSITIVE).getbbox()
    if bbox is None:
        raise ValueError("the mask paints nothing")
    x1, y1, x2, y2 = bbox
    pad = spec.pad
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
    x1, y1 = x1 // GRID * GRID, y1 // GRID * GRID
    x2, y2 = min(width, _ceil_to(x2, GRID)), min(height, _ceil_to(y2, GRID))
    rw, rh = x2 - x1, y2 - y1
    if rw * ASPECT_SHORT > rh * ASPECT_LONG:
        y1, y2 = _grow(y1, y2, _ceil_to(_ceil_to(rw * ASPECT_SHORT, ASPECT_LONG) // ASPECT_LONG, GRID), height)
    elif rh * ASPECT_SHORT > rw * ASPECT_LONG:
        x1, x2 = _grow(x1, x2, _ceil_to(_ceil_to(rh * ASPECT_SHORT, ASPECT_LONG) // ASPECT_LONG, GRID), width)
    return Crop((x1, y1, x2, y2), working_size(x2 - x1, y2 - y1, spec.tier))


def request_region(request: Any) -> Crop | None:
    """The region a request's crop decides, or None without one: its stored
    mask read at the request's size (`refine.mask_image`, the mask file
    only), or an outpaint's before its stamp built from its spec
    (`outpaint.mask`, the same picture the stamp stores). The backend's call
    and its start sigma both read the working size from here, through
    `GenerateRequest.denoise_size`."""
    spec = getattr(request, "mask_crop", None)
    if spec is None:
        return None
    if request.mask is not None:
        painted = refine.mask_image(request.mask, request.width, request.height)
    elif request.outpaint is not None:
        from . import outpaint

        painted = outpaint.mask(request.width, request.height, request.outpaint)
    else:
        from .schemas_mask_crop import MASK_CROP_NEEDS_A_MASK

        raise ValueError(MASK_CROP_NEEDS_A_MASK)
    return region(painted, spec)


def no_gain(crop: Crop, width: int, height: int) -> bool:
    """Whether cropping gains nothing: the box is the whole picture, which is
    no larger than the working size - the class would run on the same
    picture, only resized up."""
    return crop.box == (0, 0, width, height) and width * height <= crop.working[0] * crop.working[1]


def radius(crop: Crop, block: int = refine.LATENT_BLOCK) -> int:
    """How far, in working pixels, the thresholded mask is widened so the
    class repaints every working pixel the paste back's Lanczos reads for a
    painted pixel: 3 of the box's pixels where the working size is larger
    (the resize back shrinks), 3 working pixels where it is smaller, and at
    least one latent cell. `tests/test_mask_crop.py` proves the coverage."""
    rw, rh = crop.box_size
    width, height = crop.working
    reach = max(1.0, width / rw, height / rh)
    return max(int(block), math.ceil(_LANCZOS_SUPPORT * reach))


def _window_max(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """The largest value within `radius` along `axis`, the edge padded with
    zeros: a window of `2 * radius + 1` built by doubling."""
    if radius <= 0:
        return array
    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad)
    length = padded.shape[axis]
    # `out[i]` is the max of `padded[i : i + span]`.
    out, span = padded, 1
    window = 2 * radius + 1
    while span < window:
        step = min(span, window - span)
        shifted = np.take(out, np.arange(step, length), axis=axis)
        head = np.take(out, np.arange(0, length - step), axis=axis)
        out = np.maximum(head, shifted)
        length -= step
        span += step
    return out


def widened(mask: "Image", radius: int) -> "Image":
    """`mask` (0 or 255) dilated by a square of `radius` pixels each way."""
    from PIL import Image as PILImage

    levels = np.asarray(mask.convert("L"))
    levels = _window_max(_window_max(levels, radius, 0), radius, 1)
    return PILImage.fromarray(np.ascontiguousarray(levels), "L")


def crop_inputs(source: "Image", painted: "Image", crop: Crop, block: int = refine.LATENT_BLOCK) -> CropInputs:
    """The class's inputs for `crop`: the source's box resized to the working
    size by Lanczos; the painted mask thresholded (> 0 is 255) before its
    box resize, thresholded again after it, and widened by `radius`; and
    that widened mask as `refine.pipeline_mask` hands it to the class."""
    from PIL import Image as PILImage

    if source.size != painted.size:
        raise ValueError(
            f"cannot crop a {source.width}x{source.height} source with a {painted.width}x{painted.height} mask"
        )
    image = source.convert("RGB").crop(crop.box).resize(crop.working, PILImage.Resampling.LANCZOS)
    levels = painted if painted.mode == "L" else painted.convert("L")
    thresholded = (
        levels.point(_POSITIVE).crop(crop.box).resize(crop.working, PILImage.Resampling.BOX).point(_POSITIVE)
    )
    mask = widened(thresholded, radius(crop, block))
    return CropInputs(image, mask, refine.pipeline_mask(mask, block))


def paste_back(source: "Image", out: "Image", box: tuple[int, int, int, int]) -> "Image":
    """`out`, the class's result at the working size, resized to `box` by
    Lanczos and pasted into a copy of `source`: the picture
    `refine.composite` then blends over the source with the painted mask."""
    from PIL import Image as PILImage

    x1, y1, x2, y2 = box
    pasted = source.convert("RGB").copy()
    pasted.paste(out.convert("RGB").resize((x2 - x1, y2 - y1), PILImage.Resampling.LANCZOS), (x1, y1))
    return pasted

