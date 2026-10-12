"""An outpaint's pure parts: both types' validation, the key, the presets'
integer arithmetic, the mask's levels, the fill, and their pinned bytes."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from backend import outpaint, refine
from backend.schemas import GenerateRequest, Outpaint, OutpaintSpec
from backend.sizes import SIZES

LIMITS = {"max_side": 4448, "max_pixels": 4 * 1776 * 1184}


def _fixture(width: int = 64, height: int = 48) -> Image.Image:
    """A source whose every pixel differs from its neighbours'."""
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([(x * 4) % 256, (y * 5) % 256, ((x + y) * 3) % 256], axis=2).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


# ------------------------------------------------------------------ the types


def test_a_side_off_the_grid_is_refused_with_its_neighbours():
    with pytest.raises(ValidationError, match="50 is not a multiple of 16: 48 or 64"):
        Outpaint(left=50)
    with pytest.raises(ValidationError, match="50 is not a multiple of 16: 48 or 64"):
        OutpaintSpec(fill="edge-box3-r16", left=50, top=0, right=0, bottom=0, band=32)


def test_the_band_is_on_the_latent_grid_and_bounded_on_the_request():
    with pytest.raises(ValidationError, match="30 is not a multiple of 8: 24 or 32"):
        Outpaint(left=16, band=30)
    with pytest.raises(ValidationError, match="less than or equal to 256"):
        Outpaint(left=16, band=264)
    # The stored type has no bound a request's constant could move.
    assert OutpaintSpec(fill="edge-box3-r16", left=16, top=0, right=0, bottom=0, band=264).band == 264
    assert Outpaint(left=16, band=0).band == 0


def test_the_request_extends_a_side_and_the_stored_type_parses_any():
    with pytest.raises(ValidationError, match="an outpaint extends at least one side"):
        Outpaint()
    assert OutpaintSpec(fill="edge-box3-r16", left=0, top=0, right=0, bottom=0, band=0).left == 0


def test_a_fill_name_is_one_word_of_lowercase_and_digits():
    for bad in ("Edge", "edge box", "", "-edge", "a" * 33, "edge:3"):
        with pytest.raises(ValidationError):
            Outpaint(left=16, fill=bad)
    # Any well-formed name is a fill; whether this build has it is the submit's.
    assert Outpaint(left=16, fill="edge-box5-r8").fill == "edge-box5-r8"
    assert Outpaint(left=16).fill == outpaint.FILL == "edge-box3-r16"
    assert Outpaint(left=16).band == outpaint.BAND == 32


def test_the_stored_type_parses_sides_past_any_setting():
    # A row written under a larger refine guard still reads.
    spec = OutpaintSpec.parse_key("edge-box3-r16:8000,0,8000,0@32")
    assert (spec.left, spec.right) == (8000, 8000)


def test_the_key_round_trips_with_its_fill():
    spec = Outpaint(left=400, right=400)
    assert spec.key() == "edge-box3-r16:400,0,400,0@32"
    assert OutpaintSpec.parse_key(spec.key()) == OutpaintSpec(**spec.model_dump())
    other = Outpaint(top=16, bottom=32, band=0, fill="edge-box5-r8")
    assert OutpaintSpec.parse_key(other.key()).key() == "edge-box5-r8:0,16,0,32@0"
    for junk in ("", "edge-box3-r16:1,2,3@4", "edge-box3-r16:16,0,0,0", "x:16,0,0,0@8@8", "../x:16,0,0,0@8"):
        with pytest.raises(ValueError):
            OutpaintSpec.parse_key(junk)
    with pytest.raises(ValueError):
        OutpaintSpec.parse_key("edge-box3-r16:17,0,0,0@32")
    assert hash(spec) == hash(Outpaint(left=400, right=400))


def test_an_outpaint_needs_its_source_and_no_mask():
    with pytest.raises(ValidationError, match="an outpaint needs the image it extends"):
        GenerateRequest(prompt="x", outpaint={"left": 16})
    with pytest.raises(ValidationError, match="an outpaint builds its own mask; send none with it"):
        GenerateRequest(prompt="x", refine=1, mask="a" * 64, outpaint={"left": 16})
    request = GenerateRequest(prompt="x", refine=1, width=272, height=256, outpaint={"left": 16})
    assert request.inpainting and request.mask is None


def test_applied_drops_an_outpaint_with_its_source():
    request = GenerateRequest(prompt="x", refine=1, width=272, height=256, outpaint={"left": 16}, strength=0.5)
    kept = request.applied(
        negative_prompt_supported=True, cfg_threshold=1.0, refine_supported=False, inpaint_supported=True,
    )
    assert kept.outpaint == request.outpaint and kept.refine == 1
    # The inpaint class's question, not the image-to-image class's.
    dropped = request.applied(
        negative_prompt_supported=True, cfg_threshold=1.0, refine_supported=True, inpaint_supported=False,
    )
    assert (dropped.refine, dropped.outpaint, dropped.mask) == (None, None, None)


# ---------------------------------------------------------------- the presets


def _by_name(width, height, **limits):
    return {p["name"]: p for p in outpaint.presets(width, height, **{**LIMITS, **limits}, name="#12")}


def _pads(preset):
    return preset["left"], preset["top"], preset["right"], preset["bottom"]


def test_presets_over_a_square_source_by_hand():
    presets = _by_name(1024, 1024)
    assert _pads(presets["16:9"]) == (400, 0, 400, 0)
    assert (presets["16:9"]["width"], presets["16:9"]["height"]) == (1824, 1024)
    assert _pads(presets["2:3"]) == (0, 256, 0, 256)
    assert (presets["2:3"]["width"], presets["2:3"]["height"]) == (1024, 1536)
    assert _pads(presets["+25%"]) == (128, 128, 128, 128)
    assert (presets["+25%"]["width"], presets["+25%"]["height"]) == (1280, 1280)
    assert all(presets[name]["held"] is None for name in ("16:9", "2:3", "+25%"))
    # The square is already 1:1: the 1:1 preset at 1 MP.
    assert presets["1:1"]["held"] == "#12 is already 1:1 - the 1:1 preset at 1 MP"
    # "+25%" first, then every aspect in the table's order.
    assert list(presets) == ["+25%", *SIZES]


def test_an_odd_padding_puts_the_extra_sixteen_right_or_bottom():
    # 1024 x 1024 to 7:4: 1792 wide, 768 to pad.
    assert _pads(_by_name(1024, 1024)["7:4"]) == (384, 0, 384, 0)
    # 1056 -> +25% is (1056 + 32) // 64 * 16 = 272: 128 and 144.
    assert _pads(_by_name(1056, 1056)["+25%"]) == (128, 128, 144, 144)
    # A small source still grows by at least 16 on each end.
    assert _pads(_by_name(256, 256)["+25%"]) == (32, 32, 32, 32)
    assert _pads(_by_name(64, 64)["+25%"]) == (16, 16, 16, 16)


@pytest.mark.parametrize("aspect", list(SIZES))
def test_each_preset_size_is_held_to_its_own_aspect_at_every_tier(aspect):
    for tier, (width, height) in SIZES[aspect].items():
        held = _by_name(width, height)[aspect]["held"]
        assert held == f"#12 is already {aspect} - the {aspect} preset at {tier:g} MP", (width, height)


def test_a_source_within_sixteen_pixels_of_the_ratio_is_held():
    presets = _by_name(1376, 768)
    assert _pads(presets["16:9"]) == (0, 0, 0, 16)
    assert presets["16:9"]["held"] == "#12 is within 16 px of 16:9"
    # 1360 x 768 is the 16:9 preset itself, which the arithmetic alone pads.
    assert _by_name(1360, 768)["16:9"]["held"] == "#12 is already 16:9 - the 16:9 preset at 1 MP"


def test_a_preset_past_the_refine_guard_is_held_with_its_size():
    presets = _by_name(1024, 1024, max_side=1600)
    assert presets["16:9"]["held"] == (
        f"to 16:9 is 1824 x 1024, past the 1600-pixel side or {LIMITS['max_pixels']:,} pixels a refine pass takes"
    )
    assert presets["+25%"]["held"] is None
    assert _by_name(1024, 1024, max_pixels=1_000_000)["+25%"]["held"].startswith("+25% all round is 1280 x 1280")


def test_every_preset_over_an_off_grid_source_is_held():
    words = "#12 is 1016 x 1016, off the 16-px grid, so no outpaint of it is either"
    assert outpaint.source_off_grid(1016, 1016, "#12") == words
    assert outpaint.source_off_grid(1024, 1024, "#12") is None
    assert {p["held"] for p in _by_name(1016, 1016).values()} == {words}


def test_one_preset_by_name():
    assert _pads(outpaint.preset(1024, 1024, "16:9", **LIMITS)) == (400, 0, 400, 0)
    with pytest.raises(ValueError, match="no outpaint preset '5:4'"):
        outpaint.preset(1024, 1024, "5:4", **LIMITS)


# ------------------------------------------------------------------- the mask


def _levels(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.int64)


def test_the_mask_is_white_outside_and_ramps_over_the_band():
    spec = Outpaint(left=32, right=16, band=32)
    built = outpaint.mask(96 + 48, 48, spec)
    assert built.mode == "L" and built.size == (144, 48)
    levels = _levels(built)
    assert (levels[:, :32] == 255).all() and (levels[:, 128:] == 255).all()
    # From 255 at the original's first column to 0 at the band's inner edge.
    for d, expected in ((0, 255), (1, (255 * 31 + 16) // 32), (31, (255 + 16) // 32), (32, 0)):
        assert (levels[:, 32 + d] == expected).all(), d
        assert (levels[:, 127 - d] == expected).all(), d
    # The unpadded top and bottom have no band.
    assert (levels[0, 32 + 32 : 128 - 32] == 0).all()


def test_two_bands_meeting_in_a_corner_take_the_most():
    spec = Outpaint(left=16, top=16, band=8)
    levels = _levels(outpaint.mask(64, 64, spec))
    # (d_left, d_top) = (2, 5): the left ramp is higher.
    assert levels[16 + 5, 16 + 2] == max((255 * 6 + 4) // 8, (255 * 3 + 4) // 8)
    assert levels[16 + 8, 16 + 8] == 0 and (levels[:16] == 255).all() and (levels[:, :16] == 255).all()


def test_a_band_of_zero_is_a_hard_edge():
    levels = _levels(outpaint.mask(80, 48, Outpaint(left=16, band=0)))
    assert (levels[:, :16] == 255).all() and (levels[:, 16:] == 0).all()


def test_the_pipeline_mask_of_an_outpaint_mask_is_its_own_threshold():
    for spec, size in (
        (Outpaint(left=32, right=16, band=32), (112, 48)),
        (Outpaint(top=16, bottom=48, band=8), (64, 112)),
        (Outpaint(left=16, top=16, right=16, bottom=16, band=0), (96, 80)),
    ):
        built = outpaint.mask(*size, spec)
        threshold = built.point(lambda level: 255 if level > 0 else 0)
        assert refine.pipeline_mask(built, 8).tobytes() == threshold.tobytes()


def test_the_kept_rectangle():
    assert outpaint.kept_size(64, 48, Outpaint(left=16, band=32)) == (32, 48)
    assert outpaint.kept_size(64, 48, Outpaint(left=16, right=16, band=32)) == (0, 48)


def test_a_band_too_wide_is_said_by_its_axis_and_padded_sides():
    # One padded side: the band comes off that side alone.
    assert outpaint.band_too_wide(256, 256, Outpaint(left=16, band=248), "#4") == (
        "a band of 248 px on the left side leaves 8 px of #4's width unrepainted; "
        "at least 16 must be kept, so narrow the band"
    )
    assert outpaint.band_too_wide(256, 64, Outpaint(top=16, bottom=16, band=32), "#4") == (
        "a band of 32 px on the top and bottom sides leaves 0 px of #4's height unrepainted; "
        "at least 16 must be kept, so narrow the band"
    )
    # Exactly MIN_KEPT kept is enough.
    assert outpaint.band_too_wide(64, 48, Outpaint(left=16, right=16, band=24)) is None


# ------------------------------------------------------------------- the fill


def test_the_padded_source_keeps_every_source_pixel():
    source = _fixture()
    spec = Outpaint(left=32, top=16, right=16, bottom=48)
    canvas = outpaint.padded(source, spec)
    assert canvas.size == (64 + 48, 48 + 64) and canvas.mode == "RGB"
    assert canvas.crop((32, 16, 96, 64)).tobytes() == source.tobytes()


def test_the_fill_is_the_edge_stretched_outward():
    # Red on the left half, blue on the right: far enough from the seam,
    # the stretched and blurred new area is the edge's own colour.
    source = Image.new("RGB", (128, 64), (200, 0, 0))
    source.paste((0, 0, 200), (64, 0, 128, 64))
    canvas = outpaint.padded(source, Outpaint(left=64, right=64))
    assert canvas.getpixel((0, 10)) == (200, 0, 0) and canvas.getpixel((10, 63)) == (200, 0, 0)
    assert canvas.getpixel((255, 10)) == (0, 0, 200)
    # A flat source fills flat.
    flat = outpaint.padded(Image.new("RGB", (32, 32), (7, 80, 160)), Outpaint(left=32, top=48, right=16))
    assert (np.asarray(flat) == np.array([7, 80, 160], dtype=np.uint8)).all()


def test_an_unknown_fill_raises_by_name():
    with pytest.raises(ValueError, match="this build has no outpaint fill 'edge-box5-r8'; it fills with edge-box3-r16"):
        outpaint.padded(_fixture(), Outpaint(left=16, fill="edge-box5-r8"))


# The frozen bytes: a change to the fill or the mask fails here rather than
# silently making old outpaints unreproducible. A new fill is a new name.
PINNED_FILL = "ddaa9949c75a9c1dfb1a8a79a9a919f53801797c85a2ab35e4d62470947389e3"
PINNED_MASK = "ffee7784726776e550e867dbd67781e26dbd9f85f96a13493541d4d601a7b3e5"


def test_the_fill_and_the_mask_are_pinned():
    spec = Outpaint(left=32, top=16, right=16, bottom=48, band=24)
    canvas = outpaint.padded(_fixture(), spec)
    built = outpaint.mask(canvas.width, canvas.height, spec)
    assert hashlib.sha256(canvas.tobytes()).hexdigest() == PINNED_FILL
    assert hashlib.sha256(built.tobytes()).hexdigest() == PINNED_MASK


def test_the_composite_over_the_padded_canvas_keeps_the_kept_rectangle():
    source = _fixture()
    spec = Outpaint(left=32, right=16, band=16)
    canvas = outpaint.padded(source, spec)
    built = outpaint.mask(canvas.width, canvas.height, spec)
    result = refine.composite(built, canvas, Image.new("RGB", canvas.size, (1, 2, 3)))
    # Inside the band's inner edges the source exactly; the border the result's.
    assert result.crop((32 + 16, 0, 32 + 64 - 16, 48)).tobytes() == source.crop((16, 0, 48, 48)).tobytes()
    assert result.getpixel((0, 0)) == (1, 2, 3) and result.getpixel((canvas.width - 1, 47)) == (1, 2, 3)
