"""record_fields' exact-or-nothing readers: each malformed or mismatched
input reads as None (or as itself, for a column key), never as a guess.

The well-formed round trips are held by the storage, bundle and reindex
tests that use these codecs; this file holds the refusals they pass through.
"""
from __future__ import annotations

import pytest
from PIL import Image

from backend import outpaint, record_fields
from backend.schemas_outpaint import OutpaintSpec

SPEC = OutpaintSpec(fill=outpaint.FILL, left=64, top=0, right=32, bottom=16, band=16)
SOURCE = (256, 256)
CANVAS = (SOURCE[0] + 64 + 32, SOURCE[1] + 0 + 16)


def _mask_digest(spec: OutpaintSpec = SPEC, size: tuple[int, int] = CANVAS) -> str:
    return record_fields.mask_hash(outpaint.mask(*size, spec))


def test_mask_hash_refuses_anything_but_one_luminance_channel():
    with pytest.raises(ValueError, match="one luminance channel, not RGB"):
        record_fields.mask_hash(Image.new("RGB", (8, 8)))


def test_mask_hash_tells_apart_the_same_bytes_at_transposed_sizes():
    wide, tall = Image.new("L", (16, 8), 255), Image.new("L", (8, 16), 255)
    assert wide.tobytes() == tall.tobytes()
    assert record_fields.mask_hash(wide) != record_fields.mask_hash(tall)


@pytest.mark.parametrize("text", ["", None, "not-a-key", "outpaint:"])
def test_a_column_that_is_no_outpaint_key_reads_as_itself(text):
    assert record_fields._canonical_outpaint(text) == text


def test_an_outpaint_key_reads_in_its_one_spelling():
    assert record_fields._canonical_outpaint(SPEC.key()) == SPEC.key()


@pytest.mark.parametrize(("text", "expected"), [("", None), (None, None), ("no-such-key", "no-such-key")])
def test_a_column_that_is_no_mask_crop_key_reads_as_itself_or_none(text, expected):
    assert record_fields._canonical_mask_crop(text) == expected


@pytest.mark.parametrize(("box", "working"), [
    ("0,0,8,8", [8, 8]),            # not a sequence
    ([0, 0, 8, 8], "8x8"),
    ([0, 0, 8], [8, 8]),            # wrong lengths
    ([0, 0, 8, 8], [8]),
    ([0, 0, 8.0, 8], [8, 8]),       # not ints
    ([True, 0, 8, 8], [8, 8]),      # a bool is no int here
    ([-1, 0, 8, 8], [8, 8]),        # negative
    ([8, 0, 8, 8], [8, 8]),         # empty width
    ([0, 8, 8, 8], [8, 8]),         # empty height
    ([0, 0, 8, 8], [0, 8]),         # empty working size
])
def test_a_malformed_region_reads_as_none(box, working):
    assert record_fields._region(box, working) is None


@pytest.mark.parametrize("meta", [
    {"outpaint": None, "outpaint_fill": outpaint.FILL},
    {"outpaint": {"left": 64, "top": 0, "right": 32, "bottom": 16, "band": 16}},
    {"outpaint": [64, 0, 32, 16, 16], "outpaint_fill": outpaint.FILL},
    {"outpaint": {"left": 64, "top": 0, "right": 32, "bottom": 16}, "outpaint_fill": outpaint.FILL},
    {"outpaint": {"left": "64", "top": 0, "right": 32, "bottom": 16, "band": 16}, "outpaint_fill": outpaint.FILL},
    {"outpaint": {"left": 63, "top": 0, "right": 32, "bottom": 16, "band": 16}, "outpaint_fill": outpaint.FILL},
    {"outpaint": {"left": 64, "top": 0, "right": 32, "bottom": 16, "band": 16}, "outpaint_fill": "Not A Name"},
])
def test_a_malformed_outpaint_chunk_reads_as_none(meta):
    assert record_fields.chunk_outpaint(meta) is None
    assert record_fields.malformed_outpaint(meta) is (meta.get("outpaint") is not None)


def test_an_unknown_but_well_formed_fill_is_kept():
    meta = {"outpaint": record_fields.outpaint_chunk(SPEC), "outpaint_fill": "some-later-fill"}
    assert record_fields.chunk_outpaint(meta) == SPEC.model_copy(update={"fill": "some-later-fill"})


def test_an_outpaint_mask_that_the_canvas_cannot_hold_has_no_hash():
    assert record_fields.outpaint_mask_hash(64, 16, SPEC) is None
    assert record_fields.outpaint_mask_hash(*CANVAS, SPEC) == _mask_digest()


def test_a_stripped_outpaint_is_recovered_exactly():
    line = "Steps: 28" + record_fields.outpaint_parameters(SPEC)
    assert record_fields.recovered_outpaint(line, *CANVAS, _mask_digest(), SOURCE) == SPEC


@pytest.mark.parametrize("case", [
    "not text", "no mask", "no source", "same size as the source", "no outpaint line",
    "off the grid", "no padding", "padding that does not add up", "another mask",
])
def test_a_stripped_outpaint_is_recovered_or_nothing(case):
    line = "Steps: 28" + record_fields.outpaint_parameters(SPEC)
    args = {"parameters": line, "width": CANVAS[0], "height": CANVAS[1], "mask": _mask_digest(), "source_size": SOURCE}
    if case == "not text":
        args["parameters"] = {"Outpaint": line}
    elif case == "no mask":
        args["mask"] = None
    elif case == "no source":
        args["source_size"] = None
    elif case == "same size as the source":
        args["source_size"] = CANVAS
    elif case == "no outpaint line":
        args["parameters"] = "Steps: 28, Seed: 1"
    elif case == "off the grid":
        args["parameters"] = line.replace("Outpaint: 64/", "Outpaint: 63/")
    elif case == "no padding":
        args["parameters"] = line.replace("64/0/32/16", "0/0/0/0")
        args["width"], args["height"] = 512, 512
    elif case == "padding that does not add up":
        args["width"] += 16
    elif case == "another mask":
        args["mask"] = "0" * 64
    assert record_fields.recovered_outpaint(**args) is None


def test_a_rebuilt_outpaint_mask_must_hash_to_the_rows():
    assert record_fields.rebuilt_outpaint_mask(*CANVAS, SPEC, _mask_digest()) is not None
    assert record_fields.rebuilt_outpaint_mask(*CANVAS, SPEC, "0" * 64) is None
    assert record_fields.rebuilt_outpaint_mask(64, 16, SPEC, _mask_digest()) is None
