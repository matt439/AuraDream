"""Blind review publication preserves pixels, conceals labels and refuses stale packs."""
import hashlib
from pathlib import Path

from PIL import Image, PngImagePlugin
import pytest

from tools import blind_review as review
from tools.blind_review_receipts import Pair, Side


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    cases = []
    for seed in (5, 8):
        sides = []
        for color, method in enumerate(("Baseline", "Candidate"), 1):
            path = root / f"{method}-{seed}.png"
            info = PngImagePlugin.PngInfo()
            info.add_text("source_method", method)
            with Image.new("RGB", (4, 3), (color, seed, 17)) as image:
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                image.save(path, pnginfo=info)
            sides.append(Side(method=method, seconds=2.3 * color, path=path, rgb_sha256=digest))
        cases.append(Pair(family="Synthetic", model="Fixture", seed=seed, prompt="Two trees",
                          width=4, height=3, left=sides[0], right=sides[1]))
    monkeypatch.setattr(review, "load_pairs", lambda _root: list(cases))
    return root, tmp_path / "pack", cases


def test_pack_is_blind_balanced_and_relocatable(inputs, tmp_path):
    root, pack, cases = inputs
    assert review.build(root, pack) == 2
    data = review._read_script(pack / "data.js", "REVIEW")
    key = review._read_script(pack / "key.js", "REVIEW_KEY")
    assert len(data["id"]) == 64 and key["id"] == data["id"]
    public_text = (pack / "data.js").read_text(encoding="utf-8")
    assert all(word not in public_text for word in ("Synthetic", "Baseline", "Candidate", "seconds", "source_method"))
    assert sorted(pair["a"]["method"] for pair in key["pairs"].values()) == ["Baseline", "Candidate"]
    for row in data["pairs"]:
        assert set(row) == {"id", "a", "b", "width", "height", "prompt"}
        for which in ("a", "b"):
            with Image.open(pack / row[which]) as image:
                assert image.mode == "RGB" and image.size == (4, 3)
                assert not image.info
                assert hashlib.sha256(image.tobytes()).hexdigest() == key["pairs"][row["id"]][which]["rgb_sha256"]
    moved = pack.rename(tmp_path / "moved")
    asset = moved / "review.js"
    asset.write_text(asset.read_text(encoding="utf-8"), encoding="utf-8", newline="\r\n")
    assert review.verify(root, moved) == len(cases)


def test_build_never_overwrites_or_partially_publishes(inputs):
    root, pack, cases = inputs
    original = cases[-1].left.path
    with Image.new("RGB", (4, 3), "white") as changed:
        changed.save(original)
    with pytest.raises(ValueError, match="decoded RGB"):
        review.build(root, pack)
    assert not pack.exists()
    assert list(pack.parent.glob(".blind-review-*")) == []
    pack.mkdir()
    backup = pack / "votes.json"
    backup.write_text("existing")
    with pytest.raises(ValueError, match="new pack directory"):
        review.build(root, pack)
    assert backup.read_text() == "existing"
    with pytest.raises(ValueError, match="outside"):
        review.build(root, root / "output")


@pytest.mark.parametrize("corruption", ["votes identity", "changed prompt", "side pixels", "metadata", "trailing"])
def test_verification_detects_substitution(inputs, corruption):
    root, pack, _cases = inputs
    review.build(root, pack)
    data = review._read_script(pack / "data.js", "REVIEW")
    row = data["pairs"][0]
    if corruption == "votes identity":
        data["id"] = "a" * 64
        review._write_script(pack, "data.js", "REVIEW", data)
    elif corruption == "changed prompt":
        row["prompt"] = "Three cars"
        review._write_script(pack, "data.js", "REVIEW", data)
    elif corruption == "side pixels":
        (pack / row["a"]).write_bytes((pack / row["b"]).read_bytes())
    elif corruption == "trailing":
        with (pack / row["a"]).open("ab") as png:
            png.write(b"Candidate")
    else:
        path: Path = pack / row["a"]
        with Image.open(path) as image:
            pixels = image.copy()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("method", "Candidate")
        pixels.save(path, pnginfo=metadata)
    with pytest.raises(ValueError):
        review.verify(root, pack)
