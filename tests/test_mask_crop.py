"""Only the masked area's pure parts (`mask_crop.py`, INP#9): the spec and
its key, rule `bbox16`'s golden boxes and pinned bytes, the box's
invariants, the widened mask's coverage of the paste back's reach, and the
composite keeping every unmasked pixel exactly."""
from __future__ import annotations

import hashlib
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter
from pydantic import ValidationError

from backend import mask_crop, refine
from backend.schemas_mask_crop import MaskCropSpec


def _painted(width: int, height: int, *boxes: tuple[int, int, int, int], level: int = 255) -> Image.Image:
    """A painted mask: each `(x1, y1, x2, y2)` (right and bottom exclusive)
    filled at `level`, the rest black."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=level)
    return mask


def _source(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")


# ------------------------------------------------------------------ the spec


def test_the_spec_defaults_and_its_key_round_trip():
    spec = MaskCropSpec()
    assert (spec.rule, spec.pad, spec.tier) == ("bbox16", 32, 1)
    assert spec.key() == "bbox16:32@1"
    for pad in (0, 8, 32, 256):
        crop = MaskCropSpec(pad=pad)
        assert MaskCropSpec.parse_key(crop.key()) == crop
    assert MaskCropSpec.parse_key("bbox16:64@1") == MaskCropSpec(pad=64)


def test_the_spec_refuses_what_no_request_can_ask():
    with pytest.raises(ValidationError, match="12 is not a multiple of 8: 8 or 16"):
        MaskCropSpec(pad=12)
    with pytest.raises(ValidationError):
        MaskCropSpec(pad=264)
    with pytest.raises(ValidationError):
        MaskCropSpec(pad=-8)
    with pytest.raises(ValidationError):
        MaskCropSpec(rule="bbox8")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError):
        MaskCropSpec(tier=2)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError):
        MaskCropSpec(box="0,0,16,16")  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValidationError):
        MaskCropSpec().pad = 64  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    "text", ["", "bbox16", "bbox16:32", "bbox16:032@1", "bbox16:32@01", "bbox8:32@1", "bbox16:12@1",
             "bbox16:264@1", "bbox16:32@2", "BBOX16:32@1", "bbox16:32@1 ", None]
)
def test_parse_key_refuses_what_key_never_spells(text):
    with pytest.raises(ValueError):
        MaskCropSpec.parse_key(text)  # pyright: ignore[reportArgumentType]


# ------------------------------------------------------------------ golden boxes


SPEC = MaskCropSpec()

GOLDEN = [
    # 1024 square, a 40x30 mask: a 112 square box at 1040 square (9.3x up).
    ((1024, 1024), (500, 500, 540, 530), 32, (464, 464, 576, 576), (1040, 1040)),
    # 1344x768, a wide mask: the short side grown to 12:5 about its middle.
    ((1344, 768), (100, 300, 1200, 400), 32, (64, 96, 1232, 592), (1584, 672)),
    # 832x1216, an 80x40 mask.
    ((832, 1216), (400, 600, 480, 640), 32, (368, 560, 512, 672), (1168, 912)),
    # The whole picture.
    ((1024, 1024), (0, 0, 1024, 1024), 32, (0, 0, 1024, 1024), (1040, 1040)),
    # Touching the top-left corner: clamped at 0, snapped to 48.
    ((1024, 1024), (0, 0, 10, 10), 32, (0, 0, 48, 48), (1040, 1040)),
    # Touching the bottom-right corner.
    ((1024, 1024), (1014, 1014, 1024, 1024), 32, (976, 976, 1024, 1024), (1040, 1040)),
    # No padding: the bounding box on the grid.
    ((1024, 1024), (500, 500, 540, 530), 0, (496, 496, 544, 544), (1040, 1040)),
    # Too wide: grown down and up to 12:5.
    ((2048, 1024), (0, 500, 2048, 520), 32, (0, 80, 2048, 944), (1600, 672)),
    # Too tall: grown left and right to 5:12.
    ((1024, 2048), (500, 0, 520, 2048), 32, (80, 0, 944, 2048), (672, 1600)),
    # Too wide near the top: grown, then shifted back inside.
    ((2048, 1024), (0, 0, 2048, 8), 0, (0, 0, 2048, 864), (1600, 672)),
    # Too wide on a picture too short to hold 12:5: the whole height.
    ((2048, 256), (0, 100, 2048, 110), 0, (0, 0, 2048, 256), (2928, 368)),
    # A large picture: the box larger than the working size (0.35x).
    ((4096, 3072), (64, 64, 3008, 3008), 0, (64, 64, 3008, 3008), (1040, 1040)),
]


@pytest.mark.parametrize("size, painted_box, pad, box, working", GOLDEN)
def test_the_golden_boxes(size, painted_box, pad, box, working):
    crop = mask_crop.region(_painted(*size, painted_box), MaskCropSpec(pad=pad))
    assert crop == mask_crop.Crop(box, working)


def test_a_faint_feather_counts_as_painted():
    assert mask_crop.region(_painted(1024, 1024, (500, 500, 540, 530), level=1), SPEC).box == (464, 464, 576, 576)


def test_the_working_size_follows_the_box_never_a_preset():
    # 1344x768's preset at 1 MP is itself; the box's own aspect gives this.
    assert mask_crop.working_size(1344, 768) == (1376, 784)
    assert mask_crop.working_size(16, 16) == (1040, 1040)
    # 1168x496 misses every preset aspect by more than its tolerance.
    assert mask_crop.working_size(1168, 496) == (1584, 672)
    with pytest.raises(ValueError, match="2 is not a working tier"):
        mask_crop.working_size(64, 64, tier=2)


def test_region_refuses_by_name():
    with pytest.raises(ValueError, match="the mask paints nothing"):
        mask_crop.region(Image.new("L", (64, 64), 0), SPEC)
    with pytest.raises(ValueError, match="72x64 picture is off the 16-pixel grid"):
        mask_crop.region(_painted(72, 64, (0, 0, 8, 8)), SPEC)
    with pytest.raises(ValueError, match="no masked-area rule 'bbox8'; it crops with bbox16"):
        mask_crop.region(_painted(64, 64, (0, 0, 8, 8)), SimpleNamespace(rule="bbox8", pad=32, tier=1))
    with pytest.raises(ValueError, match="0 to 256 in steps of 8, not 12"):
        mask_crop.region(_painted(64, 64, (0, 0, 8, 8)), SimpleNamespace(rule="bbox16", pad=12, tier=1))


def test_no_gain_is_the_whole_picture_no_larger_than_the_working_size():
    whole = mask_crop.region(_painted(1024, 1024, (0, 0, 1024, 1024)), SPEC)
    assert mask_crop.no_gain(whole, 1024, 1024)
    small = mask_crop.region(_painted(1024, 1024, (500, 500, 540, 530)), SPEC)
    assert not mask_crop.no_gain(small, 1024, 1024)
    # The whole of a large picture still saves memory.
    large = mask_crop.region(_painted(2048, 2048, (0, 0, 2048, 2048)), SPEC)
    assert large.box == (0, 0, 2048, 2048) and not mask_crop.no_gain(large, 2048, 2048)


# ------------------------------------------------------------------ the frozen rule


def _sweep():
    """A fixed grid of pictures, masks and paddings."""
    sizes = [(512, 512), (1024, 1024), (1344, 768), (832, 1216), (2048, 256), (256, 2048), (2400, 1600)]
    for (width, height), pad in itertools.product(sizes, (0, 8, 32, 64, 256)):
        for fx1, fy1, fx2, fy2 in ((0.1, 0.2, 0.15, 0.22), (0.0, 0.0, 0.05, 0.9), (0.4, 0.4, 1.0, 1.0),
                                   (0.0, 0.5, 1.0, 0.52), (0.3, 0.1, 0.31, 0.9), (0.0, 0.0, 1.0, 1.0)):
            box = (int(fx1 * width), int(fy1 * height), max(int(fx2 * width), int(fx1 * width) + 1),
                   max(int(fy2 * height), int(fy1 * height) + 1))
            yield (width, height), box, pad


# A change to the rule fails here rather than silently changing which box an
# old row's crop names. A changed rule is a new rule name.
PINNED_RULE = "66923b79fd8129ccd263820189a2d32931cba2c890ee4451b12118db2ddced34"


def test_the_rule_is_pinned():
    lines = []
    for size, box, pad in _sweep():
        crop = mask_crop.region(_painted(*size, box), MaskCropSpec(pad=pad))
        lines.append(f"{size} {box} {pad} -> {crop.box} {crop.working}")
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == PINNED_RULE


def test_every_box_lies_inside_and_holds_the_mask_and_its_padding():
    for (width, height), (x1, y1, x2, y2), pad in _sweep():
        crop = mask_crop.region(_painted(width, height, (x1, y1, x2, y2)), MaskCropSpec(pad=pad))
        bx1, by1, bx2, by2 = crop.box
        assert 0 <= bx1 < bx2 <= width and 0 <= by1 < by2 <= height
        assert bx1 <= max(0, x1 - pad) and by1 <= max(0, y1 - pad)
        assert bx2 >= min(width, x2 + pad) and by2 >= min(height, y2 + pad)
        assert not any(edge % 16 for edge in crop.box)
        rw, rh = crop.box_size
        # Within 12:5 either way unless the picture ran out.
        assert rw * 5 <= rh * 12 or rh == height
        assert rh * 5 <= rw * 12 or rw == width
        ww, wh = crop.working
        assert ww % 16 == 0 and wh % 16 == 0 and ww >= 256 and wh >= 256
        # The working aspect is the box's, within one 16 step.
        if ww > 256 and wh > 256:
            assert abs(ww / wh - rw / rh) <= (rw / rh) * (16 / min(ww, wh))


# ------------------------------------------------------------------ the widened mask


def test_the_widened_mask_is_a_square_max_filter():
    rng = np.random.default_rng(3)
    mask = Image.fromarray(np.where(rng.random((61, 47)) > 0.97, 255, 0).astype(np.uint8), "L")
    for radius in (0, 1, 2, 5, 8):
        expected = mask.filter(ImageFilter.MaxFilter(2 * radius + 1)) if radius else mask
        assert mask_crop.widened(mask, radius).tobytes() == expected.tobytes()


def test_the_radius_is_the_lanczos_reach_and_at_least_a_cell():
    up = mask_crop.Crop((464, 464, 576, 576), (1040, 1040))
    assert mask_crop.radius(up) == 28  # ceil(3 * 1040 / 112)
    down = mask_crop.Crop((64, 64, 3008, 3008), (1040, 1040))
    assert mask_crop.radius(down) == 8  # 3 working pixels, raised to the cell
    assert mask_crop.radius(down, block=16) == 16


def _coverage_cases():
    """Thin strokes, faint feathers and corners, from 0.35x to over 10x."""
    yield (1024, 1024), [(500, 500, 501, 530)], 255, 0  # a 1 px stroke, ~22x up
    yield (1024, 1024), [(500, 500, 540, 530)], 1, 32  # a level-1 feather
    yield (1024, 1024), [(0, 0, 3, 3)], 255, 0  # a corner, 65x up
    yield (1024, 1024), [(1020, 1000, 1024, 1024)], 1, 8  # the other corner, faint
    yield (512, 512), [(100, 100, 101, 101), (180, 140, 200, 141)], 255, 0  # a dot and a line
    yield (1344, 768), [(100, 300, 1200, 301)], 1, 32  # a long faint hairline, ~1.4x
    yield (2048, 2048), [(0, 0, 2048, 2048)], 255, 0  # the whole of 4 MP, 0.5x
    yield (3072, 3072), [(0, 0, 1, 1), (3071, 3071, 3072, 3072), (1500, 0, 1501, 3072)], 1, 0  # 0.34x
    yield (2400, 1600), [(1200, 800, 1201, 1300), (300, 200, 304, 202)], 255, 64


def test_the_widened_mask_covers_every_pixel_the_paste_back_reads_for_a_painted_one():
    """For every painted pixel, every working pixel the Lanczos resize back
    to the box gives any weight lies inside the widened mask: a probe that
    is 0 inside the mask and large outside reads back exactly 0 there."""
    scales = []
    rng = np.random.default_rng(7)
    for (width, height), boxes, level, pad in _coverage_cases():
        painted = _painted(width, height, *boxes, level=level)
        crop = mask_crop.region(painted, MaskCropSpec(pad=pad))
        inputs = mask_crop.crop_inputs(_source(width, height), painted, crop)
        assert inputs.image.size == inputs.mask.size == inputs.pipeline_mask.size == crop.working
        covered = np.asarray(inputs.mask) > 0
        # The class repaints at least the widened mask.
        assert (np.asarray(inputs.pipeline_mask)[covered] == 255).all()
        probe = np.where(covered, 0.0, 1e6 * (1.0 + rng.random(covered.shape))).astype(np.float32)
        back = np.asarray(Image.fromarray(probe, "F").resize(crop.box_size, Image.Resampling.LANCZOS))
        inside = np.asarray(painted.crop(crop.box)) > 0
        assert inside.any()
        assert (back[inside] == 0.0).all(), f"{(width, height)} {boxes}: a painted pixel reads an unrepainted one"
        scales.append(crop.scale())
    assert min(scales) < 0.36 and max(scales) > 10


# ------------------------------------------------------------------ paste back


def test_paste_back_then_composite_keeps_every_unmasked_pixel_exactly():
    source = _source(512, 512, seed=1)
    painted = _painted(512, 512, (200, 220, 260, 300))
    # A feather around it, levels 1 to 254.
    feather = painted.filter(ImageFilter.GaussianBlur(6))
    painted = Image.fromarray(np.maximum(np.asarray(painted), np.asarray(feather)), "L")
    crop = mask_crop.region(painted, SPEC)
    inputs = mask_crop.crop_inputs(source, painted, crop)
    out = _source(*crop.working, seed=2)
    pasted = mask_crop.paste_back(source, out, crop.box)
    assert pasted.size == source.size and pasted.mode == "RGB"
    levels = np.asarray(painted)
    # Outside the box the paste back is the source.
    outside_box = np.ones(levels.shape, dtype=bool)
    x1, y1, x2, y2 = crop.box
    outside_box[y1:y2, x1:x2] = False
    assert (np.asarray(pasted)[outside_box] == np.asarray(source)[outside_box]).all()
    result = refine.composite(painted, source, pasted)
    unmasked = levels == 0
    assert np.abs(np.asarray(result, dtype=np.int16) - np.asarray(source, dtype=np.int16))[unmasked].max() == 0
    full = levels == 255
    assert (np.asarray(result)[full] == np.asarray(pasted)[full]).all()
    assert inputs.image.size == crop.working


def test_crop_inputs_refuses_a_mask_of_another_size():
    crop = mask_crop.Crop((0, 0, 64, 64), (1040, 1040))
    with pytest.raises(ValueError, match="cannot crop a 64x64 source with a 128x64 mask"):
        mask_crop.crop_inputs(_source(64, 64), Image.new("L", (128, 64), 255), crop)
