"""The card sessions' image metrics (`tools/image_compare.py`) give known answers.

docs/plan-card-harness.md, "Image metrics": small synthetic PNGs made here,
each with an answer worked out by hand - the difference, the largest change
outside a mask, the seam step and the soft gradient - plus the refusals and
the hash held equal to the gallery's own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import storage
from tools import image_compare


def _png(tmp_path: Path, name: str, pixels: np.ndarray) -> str:
    path = tmp_path / name
    Image.fromarray(pixels.astype(np.uint8)).save(path)
    return str(path)


def _flat(value: int, w: int = 4, h: int = 4) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, list[str], str]:
    code = image_compare.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def test_identical_images_have_no_psnr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _png(tmp_path, "a.png", _flat(100))
    b = _png(tmp_path, "b.png", _flat(100))
    code, out, _ = _cli(capsys, a, b, "--row")
    assert code == 0
    result = json.loads(out[0])
    assert result["identical"] is True
    assert result["psnr"] is None
    assert result["mean"] == 0 and result["max"] == 0 and result["differing_pixels"] == 0
    assert out[1] == "0.0 / identical / 0.0"
    assert "Infinity" not in out[0]


def test_one_pixel_off_by_17_on_one_channel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pixels = _flat(100)
    other = pixels.copy()
    other[1, 2, 0] = 117
    code, out, _ = _cli(capsys, _png(tmp_path, "a.png", pixels), _png(tmp_path, "b.png", other), "--row")
    assert code == 0
    result = json.loads(out[0])
    # 17 over 16 pixels x 3 channels; MSE 289/48, PSNR 10 log10(65025 x 48 / 289).
    assert result["mean"] == 0.354
    assert result["max"] == 17
    assert result["psnr"] == 40.33
    assert result["pct_over_threshold"] == 6.25
    assert result["differing_pixels"] == 1 and result["pixels"] == 16
    assert result["identical"] is False
    assert out[1] == "0.4 / 40.3 / 6.2"


def test_uniform_offset_of_8_against_two_thresholds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _png(tmp_path, "a.png", _flat(100))
    b = _png(tmp_path, "b.png", _flat(108))
    _, out, _ = _cli(capsys, a, b)
    result = json.loads(out[0])
    assert (result["mean"], result["max"], result["threshold"]) == (8, 8, 16)
    assert result["pct_over_threshold"] == 0
    assert result["psnr"] == 30.07  # 10 log10(65025 / 64)
    _, out, _ = _cli(capsys, a, b, "--threshold", "7")
    assert json.loads(out[0])["pct_over_threshold"] == 100
    assert len(out) == 1  # no --row, no row


def test_largest_difference_outside_the_mask(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mask_pixels = np.zeros((4, 4), dtype=np.uint8)
    mask_pixels[1:3, 1:3] = 255
    mask = _png(tmp_path, "mask.png", mask_pixels)
    source = _flat(50)
    kept = source.copy()
    kept[1:3, 1:3] = 200  # changed only inside
    a = _png(tmp_path, "source.png", source)
    _, out, _ = _cli(capsys, a, _png(tmp_path, "kept.png", kept), "--mask", mask)
    result = json.loads(out[0])
    assert result["outside_mask_max"] == 0
    assert result["max"] == 150
    leaked = kept.copy()
    leaked[0, 3, 2] = 61
    _, out, _ = _cli(capsys, a, _png(tmp_path, "leaked.png", leaked), "--mask", mask)
    assert json.loads(out[0])["outside_mask_max"] == 11


def _vertical_edge(left: int, right: int) -> np.ndarray:
    pixels = np.zeros((3, 4, 3), dtype=np.uint8)
    pixels[:, :2] = left
    pixels[:, 2:] = right
    return pixels


def test_hard_vertical_edge_step(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mask_pixels = np.zeros((3, 4), dtype=np.uint8)
    mask_pixels[:, 2:] = 255
    mask = _png(tmp_path, "mask.png", mask_pixels)
    source = _png(tmp_path, "source.png", _vertical_edge(10, 40))
    round_trip = _png(tmp_path, "rt.png", _vertical_edge(10, 22))
    result = _png(tmp_path, "result.png", _vertical_edge(100, 95))
    code, out, _ = _cli(capsys, "--seam", "--mask", mask, source, round_trip, result, "--row")
    assert code == 0
    measured = json.loads(out[0])
    assert measured["seam"] == {
        "source": {"step": 30, "soft_gradient": None},
        "round_trip": {"step": 12, "soft_gradient": None},
        "result": {"step": 5, "soft_gradient": None},
    }
    assert measured["soft_pixels"] == 0
    assert out[1] == "30.0 / 12.0 / 5.0 | -"
    _, out, _ = _cli(capsys, "--seam", "--mask", mask, source, result, "--row")
    assert set(json.loads(out[0])["seam"]) == {"source", "result"}
    assert out[1] == "30.0 / - / 5.0 | -"


def test_two_level_soft_band_gradient(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mask_pixels = np.tile(np.array([0, 85, 170, 255, 255], dtype=np.uint8), (2, 1))
    mask = _png(tmp_path, "mask.png", mask_pixels)
    ramp = np.zeros((2, 5, 3), dtype=np.uint8)
    ramp[:] = (np.arange(5) * 20)[None, :, None]
    flat = np.full((2, 5, 3), 60, dtype=np.uint8)
    source = _png(tmp_path, "source.png", ramp)
    result = _png(tmp_path, "result.png", flat)
    _, out, _ = _cli(capsys, "--seam", "--mask", mask, source, result, "--row")
    measured = json.loads(out[0])
    # The edge pair is column 0 against 1 (20 levels); columns 1 and 2 are
    # soft, each 20 to the right and 0 below.
    assert measured["seam"]["source"] == {"step": 20, "soft_gradient": 20}
    assert measured["seam"]["result"] == {"step": 0, "soft_gradient": 0}
    assert measured["soft_pixels"] == 4
    assert out[1] == "20.0 / - / 0.0 | 20.0 / - / 0.0"


def test_sizes_that_differ_are_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _png(tmp_path, "a.png", _flat(0, 4, 4))
    b = _png(tmp_path, "b.png", _flat(0, 6, 3))
    code, out, err = _cli(capsys, a, b)
    assert code == 2 and out == []
    assert "4x4" in err and "6x3" in err


def test_other_inputs_are_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _png(tmp_path, "a.png", _flat(0))
    b = _png(tmp_path, "b.png", _flat(1))
    rgb_mask = _png(tmp_path, "m.png", _flat(255))
    cases = [
        ([a, b, "--mask", rgb_mask], "not L"),
        (["--seam", a, b], "needs --mask"),
        ([a, b, a], "need --seam"),
        ([a, "--row"], "--row needs"),
        ([a, str(tmp_path / "missing.png")], "cannot read"),
        ([a, b, "--threshold", "-1"], "--threshold"),
    ]
    for argv, words in cases:
        code, out, err = _cli(capsys, *argv)
        assert code == 2 and out == [], argv
        assert words in err, argv
    flat_mask = _png(tmp_path, "flat.png", np.zeros((4, 4), dtype=np.uint8))
    code, _, err = _cli(capsys, "--seam", "--mask", flat_mask, a, b)
    assert code == 2 and "no edge" in err


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_hash_is_the_gallerys(tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str) -> None:
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)).convert(mode)
    path = tmp_path / "a.png"
    image.save(path)
    code, out, _ = _cli(capsys, str(path))
    assert code == 0
    described = json.loads(out[0])["images"][0]
    with Image.open(path) as opened:
        assert described["hash"] == storage.content_hash(opened)
    assert described["size"] == [7, 5]
