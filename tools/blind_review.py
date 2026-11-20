"""Build or verify the offline review of the retained November 18 performance pairs.

Run from the checkout with python -m tools.blind_review {build,verify} RECEIPTS PACK.
The source PNGs, receipts and any existing pack are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from tools.blind_review_receipts import Pair, Side, load_pairs

ASSETS = Path(__file__).with_name("blind_review_assets")
STATIC_FILES = ("index.html", "review.css", "inspection.js", "results.js", "review.js")


def _json(data: object) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _write_script(pack: Path, name: str, variable: str, data: object) -> None:
    (pack / name).write_text(f"window.{variable} = {_json(data)};\n", encoding="utf-8")


def _read_script(path: Path, variable: str) -> dict[str, Any]:
    script = path.read_text(encoding="utf-8").strip()
    prefix = f"window.{variable} = "
    if not script.startswith(prefix) or not script.endswith(";"):
        raise ValueError(f"invalid {path.name}")
    data = json.loads(script[len(prefix):-1])
    if not isinstance(data, dict):
        raise ValueError(f"invalid {path.name}")
    return data


def _pixels(path: Path, pair: Pair, side: Side, *, clean: bool = False) -> bytes:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (pair.width, pair.height):
            raise ValueError(f"wrong format, mode or dimensions: {path}")
        pixels = image.tobytes()
    if hashlib.sha256(pixels).hexdigest() != side.rgb_sha256:
        raise ValueError(f"decoded RGB differs from the retained receipt: {path}")
    if clean:
        # Pillow exposes known metadata in .info. Check all chunk types here,
        # including unknown or trailing chunks which might disclose an arm.
        with path.open("rb") as stream:
            stream.seek(8)
            while header := stream.read(8):
                size, kind = int.from_bytes(header[:4], "big"), header[4:]
                if kind == b"IEND" and size == 0 and len(stream.read(5)) == 4:
                    break
                if kind not in (b"IHDR", b"IDAT"):
                    raise ValueError(f"review PNG contains metadata or trailing bytes: {path}")
                stream.seek(size + 4, 1)
            else:
                raise ValueError(f"review PNG is incomplete: {path}")
    return pixels


def _side_info(root: Path, side: Side) -> dict[str, Any]:
    return {"method": side.method, "seconds": side.seconds, "rgb_sha256": side.rgb_sha256,
            "source": side.path.resolve().relative_to(root).as_posix()}


def _records(root: Path, number: int, pair: Pair, flipped: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    ident = f"pair-{number:02}"
    a, b = (pair.right, pair.left) if flipped else (pair.left, pair.right)
    public = {"id": ident, "prompt": pair.prompt, "width": pair.width, "height": pair.height,
              "a": f"images/{ident}-a.png", "b": f"images/{ident}-b.png"}
    key = {"family": pair.family, "model": pair.model, "seed": pair.seed,
           "a": _side_info(root, a), "b": _side_info(root, b)}
    return public, key


def _identity(rows: object, key: object) -> str:
    # Backups bind to this exact pair order, side assignment, pixels and method key.
    return hashlib.sha256(_json([rows, key]).encode()).hexdigest()


def build(root: Path, pack: Path) -> int:
    root, pack = root.resolve(), pack.resolve()
    if pack.exists() or pack.is_relative_to(root):
        raise ValueError("choose a new pack directory outside the retained evidence")
    pairs = load_pairs(root)
    rng = secrets.SystemRandom()
    rng.shuffle(pairs)
    flips = {}
    for family, count in Counter(pair.family for pair in pairs).items():
        assignments = [i % 2 == 1 for i in range(count)]
        rng.shuffle(assignments)
        flips[family] = assignments

    pack.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete directory. A concurrent successful builder owns the
    # destination; directory rename cannot replace its nonempty pack.
    with TemporaryDirectory(dir=pack.parent, prefix=".blind-review-") as temporary:
        stage = Path(temporary) / "pack"
        (stage / "images").mkdir(parents=True)
        rows = []
        key = {}
        for number, pair in enumerate(pairs, 1):
            flipped = flips[pair.family].pop()
            public, private = _records(root, number, pair, flipped)
            for label, side in zip(("a", "b"), (pair.right, pair.left) if flipped else (pair.left, pair.right)):
                pixels = _pixels(side.path, pair, side)
                target = stage / public[label]
                with Image.frombytes("RGB", (pair.width, pair.height), pixels) as clean:
                    clean.save(target)
                _pixels(target, pair, side, clean=True)
            rows.append(public)
            key[public["id"]] = private
        ident = _identity(rows, key)
        _write_script(stage, "data.js", "REVIEW", {"id": ident, "pairs": rows})
        _write_script(stage, "key.js", "REVIEW_KEY", {"id": ident, "pairs": key})
        for name in STATIC_FILES:
            shutil.copyfile(ASSETS / name, stage / name)
        stage.rename(pack)
    return len(pairs)


def verify(root: Path, pack: Path) -> int:
    root = root.resolve()
    pairs = load_pairs(root)
    unmatched = {(p.family, p.prompt, p.width, p.height, p.seed): p for p in pairs}
    manifest = _read_script(pack / "data.js", "REVIEW")
    secret = _read_script(pack / "key.js", "REVIEW_KEY")
    rows, key = manifest.get("pairs"), secret.get("pairs")
    if (not isinstance(rows, list) or not isinstance(key, dict) or
            len(rows) != len(pairs) or len(key) != len(pairs)):
        raise ValueError("review pair count differs from the retained matrix")
    ident = _identity(rows, key)
    if manifest != {"id": ident, "pairs": rows} or secret != {"id": ident, "pairs": key}:
        raise ValueError("pack id differs from the reviewed assignment")
    for number, public in enumerate(rows, 1):
        private = key[f"pair-{number:02}"]
        pair = unmatched.pop((private["family"], public["prompt"], public["width"], public["height"], private["seed"]))
        flipped = private["a"]["method"] == pair.right.method
        if (public, private) != _records(root, number, pair, flipped):
            raise ValueError(f"pair {number} differs from the retained receipt")
        for label, side in zip(("a", "b"), (pair.right, pair.left) if flipped else (pair.left, pair.right)):
            _pixels(side.path, pair, side)
            _pixels(pack / public[label], pair, side, clean=True)
    for name in STATIC_FILES:
        # Git may give the same source CRLF or LF on a different checkout.
        if (pack / name).read_text(encoding="utf-8") != (ASSETS / name).read_text(encoding="utf-8"):
            raise ValueError(f"{name} differs from the viewer in this checkout")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("receipts", type=Path, help="the retained gpu-20261118 directory")
    parser.add_argument("pack", type=Path, help="review directory (must be new for build)")
    args = parser.parse_args()
    try:
        count = (build if args.action == "build" else verify)(args.receipts, args.pack)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Review pack refused: {exc}", file=sys.stderr)
        return 2
    print(f"Verified {count} pairs ({2 * count} RGB images). Open {args.pack.resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
