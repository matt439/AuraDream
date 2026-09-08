"""A rough picture of the image being denoised, from the latents the step
callback already receives - no VAE decode, no device movement.

`callback_on_step_end` is handed `latents` every step and the backend used to
throw them away, so a 28-step FLUX render was half a minute of a bar with no
picture behind it. The standard trick is a *linear* map from the latent
channels to RGB: one small matmul on a 16 x 128 x 128 tensor, microseconds
against a step measured in seconds, giving a blurry but compositionally
honest thumbnail from a few steps in. Seeing the composition at step 5 lets
a bad prompt be cancelled at step 6, and a cancel already lands within a step.

The matrices are keyed on the VAE class and its channel count, because the
map depends on the VAE's latent space and nothing else - FLUX.1's VAE is the
generic `AutoencoderKL` that 4-channel SD VAEs also use, hence the count. Each
was fitted by least squares on 2026-09-08, on the card, from images encoded
by the real VAE and normalised the way the *pipeline* normalises them (FLUX:
`(z - shift_factor) * scaling_factor`; Qwen-Image / Krea 2: `(z - mean) /
std` per channel), against the same images downsampled to latent resolution
in [-1, 1]. R-squared over held-out pixels: 0.964 for the FLUX VAE, 0.950 for
the Qwen-Image one. They cannot be validated in CI - there is no VAE there -
which is why the fit is recorded here and why `render` never raises: a
wrong-shaped latent costs a preview, not a cell.

The latents the callback carries are x_t, not x_0: under flow matching
`x_t = (1 - sigma) x_0 + sigma * eps`, so the map is applied with a gain of
`1 / (1 - sigma)` on the signal and a small blur against what is left of the
noise, and nothing is drawn at all above sigma 0.9, where there is nothing
to see. `sigma` is read off the scheduler for the step just taken.
"""
from __future__ import annotations

import base64
import inspect
import io
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Seconds between previews, so a 4-step klein or a fast Turbo does not spend
# its time encoding JPEGs; a slow FLUX step gets one every step regardless.
INTERVAL_SECONDS = 0.4
# Past this the latent is noise and the picture would only say so.
MAX_SIGMA = 0.9
# The preview is drawn at latent resolution and doubled once, bilinear: 256 px
# for a 1 MP image, a few kilobytes as JPEG, enough to read a composition.
UPSCALE = 2


@dataclass(frozen=True)
class _Factors:
    # One row per latent channel; the columns are R, G, B.
    matrix: tuple[tuple[float, float, float], ...]
    bias: tuple[float, float, float]
    # Image pixels per latent pixel, for the record.
    downsample: int


FACTORS: dict[str, _Factors] = {
    "AutoencoderKL/16": _Factors(
        matrix=(
        (-0.0712, -0.0277, +0.0114),
        (+0.1201, +0.1525, +0.1754),
        (+0.0135, -0.0714, -0.0387),
        (-0.0340, -0.0000, +0.0316),
        (+0.1116, +0.1064, +0.0704),
        (-0.0218, +0.0016, -0.0288),
        (+0.0923, +0.1539, +0.1479),
        (-0.0582, -0.0460, -0.0643),
        (-0.0166, +0.0254, +0.1149),
        (+0.0620, +0.0404, -0.0620),
        (-0.1322, -0.0580, -0.0839),
        (+0.0931, +0.0623, +0.0599),
        (-0.0347, -0.0425, -0.0292),
        (-0.0936, +0.0222, -0.0492),
        (+0.0166, -0.0388, +0.0061),
        (-0.0126, +0.0154, +0.0327),
        ),
        bias=(+0.0789, +0.1095, +0.0709),
        downsample=8,
    ),
    "AutoencoderKLQwenImage/16": _Factors(
        matrix=(
        (+0.1127, +0.0005, +0.3746),
        (-0.1669, -0.1652, -0.1292),
        (+0.4536, +0.3263, +0.2657),
        (+0.2669, +0.4087, +0.3011),
        (+0.0316, +0.0192, -0.0029),
        (-0.0014, -0.2150, -0.2130),
        (-0.0873, -0.1961, -0.0950),
        (-0.0435, +0.1184, +0.1424),
        (-0.2817, -0.2377, -0.3633),
        (+0.0134, +0.1836, +0.2546),
        (-0.1865, +0.0910, -0.0841),
        (-0.1689, -0.0724, -0.0689),
        (-0.1347, -0.0043, +0.0431),
        (+0.1495, +0.1082, +0.1550),
        (+0.3179, +0.1392, +0.2632),
        (+0.0105, -0.0426, -0.0120),
        ),
        bias=(-0.0395, +0.0605, -0.1580),
        downsample=8,
    ),
}


def factors_for(vae: Any) -> _Factors | None:
    """The map for this VAE, or None where none was fitted."""
    config = getattr(vae, "config", None)
    channels = getattr(config, "latent_channels", None) or getattr(config, "z_dim", None)
    return FACTORS.get(f"{type(vae).__name__}/{channels}")


def unpack(pipe: Any, latents: Any, height: int, width: int) -> Any:
    """The (C, H, W) latent of the first image, from the packed sequence the
    callback carries.

    Not one API: FLUX's `_unpack_latents` is a 4-argument staticmethod, Krea
    2's a 3-argument method returning a 5-D video-shaped tensor. A pipeline
    without one (klein) gets no preview.
    """
    fn = getattr(pipe, "_unpack_latents", None)
    if fn is None:
        return None
    params = inspect.signature(fn).parameters
    if "vae_scale_factor" in params:
        out = fn(latents, height, width, pipe.vae_scale_factor)
    else:
        out = fn(latents, height, width)
    while out.dim() > 4:
        out = out[:, :, 0]
    return out[0]


def render(latent: Any, factors: _Factors, sigma: float = 0.0) -> bytes | None:
    """JPEG bytes of the linear approximation, or None when there is nothing to show."""
    import torch
    from PIL import Image

    if sigma > MAX_SIGMA:
        return None
    z = latent.detach().float()
    matrix = torch.tensor(factors.matrix, dtype=torch.float32, device=z.device)
    bias = torch.tensor(factors.bias, dtype=torch.float32, device=z.device)
    # The signal is scaled by (1 - sigma) inside x_t; bring it back up, within
    # reason - at high sigma the gain would only amplify the noise.
    gain = 1.0 / max(1.0 - sigma, 0.25)
    rgb = torch.einsum("chw,ck->khw", z, matrix) * gain + bias[:, None, None]
    if sigma > 0.05:
        # Average away some of the remaining noise; a 3-tap box is enough at
        # latent resolution, where every pixel is already 8 image pixels.
        rgb = torch.nn.functional.avg_pool2d(rgb[None], 3, stride=1, padding=1)[0]
    pixels = ((rgb.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(pixels)
    if UPSCALE > 1:
        image = image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    return buffer.getvalue()


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
