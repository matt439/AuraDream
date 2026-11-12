"""Fit a live-preview map (`backend/latent_preview.py`) for one VAE, on the CPU.

    uv run python tools/fit_latent_preview.py --vae <repo id> [--subfolder vae] IMAGE_DIR [...]
    uv run python tools/fit_latent_preview.py --vae <repo id> --check AutoencoderKL/4 IMAGE_DIR [...]

The recipe is the one `latent_preview`'s docstring records for the first two
maps: each image is encoded by the real VAE (the posterior's mean, float32 on
the CPU), normalised the way the *pipeline* normalises latents
(`AutoencoderKL`: `(z - shift_factor) * scaling_factor`, a missing shift being
0 - SDXL has none), and paired pixel for pixel with the same image
area-averaged to latent resolution in [-1, 1]. One least-squares solve gives
a C x 3 matrix and a bias. Every fifth image is held out, and the R-squared
printed is over the held-out pixels (pooled over R, G and B, against each
channel's own mean). `--check KEY` fits nothing: it scores the map already in
`latent_preview.FACTORS` under KEY on all the images, which is how the
recipe was checked against a map fitted before this script existed.

Each image gives two square crops at `--size` (its left and right ends, one
crop when it is square), so a wallpaper's whole width is used. The VAE is
read from the Hugging Face cache only (`local_files_only`): this tool never
downloads, never touches CUDA and imports nothing from `backend` but the
existing maps for `--check`. One JSON object goes to stdout, the matrix in
the module's row format beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _images(dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for directory in dirs:
        found.extend(p for p in sorted(directory.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES)
    return found


def _crops(path: Path, size: int) -> list[Any]:
    """The image as one or two (3, size, size) tensors in [-1, 1]."""
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    scale = size / min(image.size)
    image = image.resize((max(size, round(image.width * scale)), max(size, round(image.height * scale))),
                         Image.Resampling.LANCZOS)
    boxes = [(0, 0, size, size)]
    if image.width > size:
        boxes.append((image.width - size, 0, image.width, size))
    elif image.height > size:
        boxes.append((0, image.height - size, size, image.height))
    out = []
    for box in boxes:
        array = np.asarray(image.crop(box), dtype=np.float32) / 127.5 - 1.0
        out.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return out


def _pairs(vae: Any, crop: Any) -> tuple[Any, Any]:
    """(pixels x C) normalised latents and (pixels x 3) targets for one crop."""
    import torch

    config = vae.config
    with torch.no_grad():
        z = vae.encode(crop[None]).latent_dist.mean[0]
    z = (z - (getattr(config, "shift_factor", None) or 0.0)) * config.scaling_factor
    factor = crop.shape[-1] // z.shape[-1]
    target = torch.nn.functional.avg_pool2d(crop[None], factor)[0]
    return z.reshape(z.shape[0], -1).T, target.reshape(3, -1).T


def _r_squared(predicted: Any, target: Any) -> float:
    residual = ((target - predicted) ** 2).sum()
    total = ((target - target.mean(dim=0)) ** 2).sum()
    return float(1.0 - residual / total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit a live-preview map for one cached VAE, on the CPU.")
    parser.add_argument("dirs", nargs="+", type=Path, help="folders of images to fit on")
    parser.add_argument("--vae", required=True, help="Hugging Face repo id holding the VAE (cached)")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--size", type=int, default=512, help="crop side in image pixels")
    parser.add_argument("--check", metavar="KEY", help="score latent_preview.FACTORS[KEY] instead of fitting")
    args = parser.parse_args(argv)

    import torch
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    torch.set_grad_enabled(False)
    # Loaded on the CPU (from_pretrained's default) and left there.
    vae: Any = AutoencoderKL.from_pretrained(args.vae, subfolder=args.subfolder, local_files_only=True,
                                             torch_dtype=torch.float32)
    vae.eval()
    key = f"{type(vae).__name__}/{vae.config.latent_channels}"
    paths = _images(args.dirs)
    if len(paths) < 5:
        print("need at least five images", file=sys.stderr)
        return 2
    fit_x, fit_y, held_x, held_y = [], [], [], []
    for index, path in enumerate(paths):
        for crop in _crops(path, args.size):
            x, y = _pairs(vae, crop)
            if index % 5 == 4:
                held_x.append(x)
                held_y.append(y)
            else:
                fit_x.append(x)
                fit_y.append(y)
    x_fit, y_fit = torch.cat(fit_x), torch.cat(fit_y)
    x_held, y_held = torch.cat(held_x), torch.cat(held_y)
    report: dict[str, Any] = {
        "vae": args.vae, "key": key, "scaling_factor": vae.config.scaling_factor,
        "shift_factor": getattr(vae.config, "shift_factor", None),
        "images": len(paths), "crop": args.size,
        "pixels_fit": int(x_fit.shape[0]), "pixels_held_out": int(x_held.shape[0]),
    }
    if args.check:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backend.latent_preview import FACTORS

        factors = FACTORS[args.check]
        matrix = torch.tensor(factors.matrix, dtype=torch.float32)
        bias = torch.tensor(factors.bias, dtype=torch.float32)
        x_all, y_all = torch.cat([x_fit, x_held]), torch.cat([y_fit, y_held])
        report["check"] = args.check
        report["r_squared_all"] = round(_r_squared(x_all @ matrix + bias, y_all), 4)
        print(json.dumps(report, indent=2))
        return 0
    design = torch.cat([x_fit, torch.ones(x_fit.shape[0], 1)], dim=1).double()
    solution = torch.linalg.lstsq(design, y_fit.double()).solution.float()
    matrix, bias = solution[:-1], solution[-1]
    report["r_squared_fit"] = round(_r_squared(x_fit @ matrix + bias, y_fit), 4)
    report["r_squared_held_out"] = round(_r_squared(x_held @ matrix + bias, y_held), 4)
    report["matrix"] = [[round(float(v), 4) for v in row] for row in matrix]
    report["bias"] = [round(float(v), 4) for v in bias]
    print(json.dumps(report, indent=2))
    for row in report["matrix"]:
        print("        (" + ", ".join(f"{v:+.4f}" for v in row) + "),")
    print("        bias=(" + ", ".join(f"{v:+.4f}" for v in report["bias"]) + "),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
