"""Image metrics for card sessions (`DEV#22`): pixels already made, by number.

    uv run python tools/image_compare.py a.png                 # the gallery hash
    uv run python tools/image_compare.py a.png b.png [--threshold 16] [--mask m.png] [--row]
    uv run python tools/image_compare.py --seam --mask m.png source.png [roundtrip.png] result.png [--row]

The definitions are the ones verified.md's tables were made with
(docs/plan-card-harness.md, "Image metrics"): the hash is the gallery's own
(`storage.content_hash`), the difference is the quantization tables' (mean
absolute difference in 8-bit levels / PSNR in dB / % of pixels off by more
than the threshold), the largest difference where the mask is 0 is the
Inpaint tables' check that the source was kept, and the seam step is "The
seam by number". One JSON object goes to stdout; `--row` adds the line in
verified.md's column order.

CPU only: Pillow and numpy, nothing from `backend`, no model and no VAE. A
round trip picture is an input the card session made with its own pipeline.
Exit 2 refuses an input (unreadable, sizes that differ, a mask not in `L`);
the images are never resized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from typing import Any

import numpy as np
from PIL import Image

THRESHOLD = 16  # verified.md's quantization tables; the Nunchaku rows used 8


class Refused(Exception):
    """An input this tool will not measure; exit 2 with the reason."""


def content_hash(image: Image.Image) -> str:
    """`storage.content_hash`'s rule: the decoded pixels, in the mode the PNG opens in."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def _open(path: str) -> Image.Image:
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read {path}: {exc}") from exc


def _rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.int16)


def _same_size(named: list[tuple[str, Image.Image]]) -> None:
    sizes = {im.size for _, im in named}
    if len(sizes) > 1:
        listed = ", ".join(f"{name} {im.width}x{im.height}" for name, im in named)
        raise Refused(f"sizes differ: {listed}")


def _mask(path: str) -> Image.Image:
    mask = _open(path)
    if mask.mode != "L":
        raise Refused(f"the mask {path} is mode {mask.mode}, not L")
    return mask


def describe(path: str, image: Image.Image) -> dict[str, Any]:
    return {"path": path, "hash": content_hash(image), "size": [image.width, image.height]}


def difference(a: Image.Image, b: Image.Image, threshold: int = THRESHOLD,
               mask: Image.Image | None = None) -> dict[str, Any]:
    """A against B, both as RGB in int16; rounded as card4-quant/compare.py did."""
    _same_size([("a", a), ("b", b)] + ([("mask", mask)] if mask is not None else []))
    d = np.abs(_rgb(a) - _rgb(b))
    mse = float((d.astype(np.float64) ** 2).mean())
    per_pixel = d.max(axis=2)
    out: dict[str, Any] = {
        "identical": mse == 0,
        "mean": round(float(d.mean()), 3),
        "max": int(d.max()),
        # Strict JSON has no Infinity: identical images say so instead.
        "psnr": None if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 2),
        "threshold": threshold,
        "pct_over_threshold": round(100 * float((per_pixel > threshold).mean()), 2),
        "differing_pixels": int((per_pixel > 0).sum()),
        "pixels": int(per_pixel.size),
    }
    if mask is not None:
        outside = np.asarray(mask) == 0
        out["outside_mask_max"] = int(per_pixel[outside].max()) if outside.any() else None
    return out


def seam(images: list[Image.Image], mask: Image.Image) -> list[dict[str, Any]]:
    """Each picture's step across the mask's 0 / above-0 edge, and its soft-region gradient.

    seam_insert.py's measure: the step is the mean absolute difference over
    every RGB channel of the horizontal and vertical neighbour pairs the
    edge separates; the gradient is the mean of the summed forward
    horizontal and vertical differences (0 past the last column or row)
    over the pixels where 0 < mask < 255, None without any.
    """
    _same_size([(f"image {i + 1}", im) for i, im in enumerate(images)] + [("mask", mask)])
    level = np.asarray(mask)
    inside = level > 0
    across_h = inside[:, :-1] != inside[:, 1:]
    across_v = inside[:-1] != inside[1:]
    soft = (level > 0) & (level < 255)
    if not (across_h.any() or across_v.any()):
        raise Refused("the mask has no edge between 0 and above 0")
    rows = []
    for image in images:
        x = _rgb(image).astype(np.float64)
        steps = np.concatenate([np.abs(x[:, :-1] - x[:, 1:])[across_h].ravel(),
                                np.abs(x[:-1] - x[1:])[across_v].ravel()])
        gradient = None
        if soft.any():
            gx = np.zeros_like(x)
            gx[:, :-1] = np.abs(x[:, 1:] - x[:, :-1])
            gy = np.zeros_like(x)
            gy[:-1] = np.abs(x[1:] - x[:-1])
            gradient = round(float((gx + gy)[soft].mean()), 2)
        rows.append({"step": round(float(steps.mean()), 2), "soft_gradient": gradient})
    return rows


def _one(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def difference_row(result: dict[str, Any]) -> str:
    """verified.md's "mean / PSNR / % off by more than 16", one place each."""
    psnr = "identical" if result["psnr"] is None else _one(result["psnr"])
    return f"{_one(result['mean'])} / {psnr} / {_one(result['pct_over_threshold'])}"


def seam_row(pictures: list[dict[str, Any]]) -> str:
    """verified.md's "Step: source / round trip / result | Soft region gradient"."""
    if len(pictures) == 2:  # no round trip supplied
        pictures = [pictures[0], {"step": None, "soft_gradient": None}, pictures[1]]
    step = " / ".join(_one(p["step"]) for p in pictures)
    if pictures[0]["soft_gradient"] is None:
        return f"{step} | -"
    return f"{step} | " + " / ".join(_one(p["soft_gradient"]) for p in pictures)


def run(argv: list[str] | None = None) -> tuple[dict[str, Any], str | None]:
    """The JSON object and, with `--row`, the row; Refused for an input it will not take."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("images", nargs="+", help="one (hash), two (A B) or with --seam source [round trip] result")
    ap.add_argument("--threshold", type=int, default=THRESHOLD, help="the %% off counts pixels over this")
    ap.add_argument("--mask", help="an L PNG: the largest difference where it is 0; required by --seam")
    ap.add_argument("--seam", action="store_true", help="the step across the mask's edge in each picture")
    ap.add_argument("--row", action="store_true", help="also print the line in verified.md's column order")
    args = ap.parse_args(argv)
    if args.threshold < 0:
        raise Refused("--threshold must be 0 or more")
    named = [(path, _open(path)) for path in args.images]
    mask = _mask(args.mask) if args.mask else None
    out: dict[str, Any] = {"images": [describe(path, im) for path, im in named]}
    row = None
    if args.seam:
        if mask is None:
            raise Refused("--seam needs --mask")
        if len(named) not in (2, 3):
            raise Refused("--seam takes source [round trip] result: two or three images")
        labels = ["source", "result"] if len(named) == 2 else ["source", "round_trip", "result"]
        rows = seam([im for _, im in named], mask)
        out["seam"] = dict(zip(labels, rows))
        out["soft_pixels"] = int(((np.asarray(mask) > 0) & (np.asarray(mask) < 255)).sum())
        row = seam_row(rows)
    elif len(named) == 2:
        out.update(difference(named[0][1], named[1][1], args.threshold, mask))
        row = difference_row(out)
    elif len(named) == 1 and mask is None:
        if args.row:
            raise Refused("--row needs two images or --seam")
    else:
        raise Refused("one image (its hash) or two (A against B); three need --seam")
    return out, row if args.row else None


def main(argv: list[str] | None = None) -> int:
    try:
        out, row = run(argv)
    except Refused as exc:
        print(f"image_compare: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out))
    if row is not None:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
