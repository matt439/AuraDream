"""The refine pass: a gallery image scaled up and re-denoised part of the way.

The size guides say it in the authors' own words - FLUX.1's "render at 1-2 MP
and upscale", WAI's 1024x1344 followed by a 1.5x hires pass at denoising
strength 0.35-0.5 - and until this module the app could only render directly
at the requested size, where the same seed is a new sample and a checkpoint
past its documented range doubles subjects or drops them. A refine pass keeps
the picture: the source is resized to the requested size, encoded by the VAE,
noised to the point on the schedule its `strength` names and denoised from
there under the request's own prompt and settings.

The default Lanczos pass loads no new weights. diffusers ships an image-to-image class beside
most text-to-image ones (`FluxImg2ImgPipeline` beside `FluxPipeline`), taking
the same components, so the backend builds that sibling over the pipeline it
already holds (`build`) and the two share every module, hook and adapter.
Which checkpoints have one is therefore the installed diffusers' answer
(`sibling`), not a table here: FLUX.1, SDXL, Z-Image and Qwen-Image do; Krea
2, FLUX.2 [klein], ERNIE-Image and Lumina 2 do not, and neither do the local
Anima and NewBie pipelines. WAI and RouWei can explicitly opt into the pinned
Anime6B GAN before the VAE encode; its temporary net is released before the
diffusion pass. The default path and its existing pixels stay unchanged.

An inpaint is a refine pass with a mask: the same source and strength, at the
source's own size, through the inpaint class diffusers ships beside the
pipeline (`sibling(..., "Inpaint")`, the same naming and exact-export rule),
built the same way (`build`). The classes re-impose the kept region on the
latents every step, but decode the whole picture, so what they keep is the
VAE's round trip of the source; `composite` pastes the result back over the
source's own pixels with the mask as alpha. Which checkpoints get one is the
profile's (`inpaint_supported`) as well as the library's: see
`model_defaults.INPAINT_DEFERRED`.

The source is a gallery *id*, for the reason a reference image is: a path
from the request joined onto a directory is an arbitrary-file read.

A leaf module like `seeds`: `schemas_generate`, `grid` and `storage` all import it, so
it imports none of them at the top.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from PIL.Image import Image

# The strength a pass runs at where the checkpoint's profile names none. Where
# a strength starts is the profile's (`refine_strength` in `model_defaults`,
# through `schemas_generate.default_strength`), because a strength is an index into the
# checkpoint's own schedule: 0.4 is inside the 0.35-0.5 WAI's author documents
# for SDXL's hires pass, and on FLUX.1 at 1.5x it repaints the face. This is
# what is left for a checkpoint the table does not know.
DEFAULT_STRENGTH = 0.4

# What the Refine button multiplies the source's sides by, and the CLI's
# `--scale` default: WAI's documented hires factor.
DEFAULT_SCALE = 1.5

# The sole optional GAN preprocess. The WAI and RouWei guides cite this
# kind of hires pass; other checkpoints and masked passes stay Lanczos-only.
Upscaler = Literal["realesrgan-x4plus-anime6b"]
ANIME6B: Upscaler = "realesrgan-x4plus-anime6b"
ANIME6B_LABEL = "R-ESRGAN 4x+ Anime6B"
ANIME6B_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
ANIME6B_BYTES = 17_938_799
# Computed from the bounded official release asset 43714889 on 2026-10-18.
ANIME6B_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"
# Bound the stitched 4x RGB canvas to 64 Mi pixels (about 192 MB of RGB
# bytes). A tiny requested target must not bypass this preprocessing bound.
ANIME6B_MAX_SOURCE_PIXELS = 4_194_304
ANIME6B_MAX_SOURCE_SIDE = 4096


def upscaler_supported(model_id: str) -> bool:
    return model_id.lower() in {"lyliaengine/waiillustrioussdxl_v170", "minthy/rouwei-0.8"}


def upscaler_reason(request: Any, model_id: str) -> str | None:
    """Refuse an explicit choice before any applied() normalisation."""
    choice = getattr(request, "upscaler", None)
    if choice is None:
        return None
    if choice != ANIME6B:
        return "the requested Refine upscaler is not supported by this build"
    if request.refine is None or request.mask is not None or request.outpaint is not None:
        return f"{ANIME6B_LABEL} needs an ordinary Refine pass; it cannot be paired with a mask or outpaint"
    if not upscaler_supported(model_id):
        return f"{ANIME6B_LABEL} is offered only for WAI v17 and RouWei ordinary Refine passes"
    return None


def check_upscaler_size(width: int, height: int) -> None:
    if min(width, height) < 1 or max(width, height) > ANIME6B_MAX_SOURCE_SIDE or width * height > ANIME6B_MAX_SOURCE_PIXELS:
        raise ValueError(f"{ANIME6B_LABEL} supports source images up to 4.19 MP and 4096 pixels per side; use a smaller source")


def check_upscaler_source(image_id: int) -> None:
    from PIL import Image as PILImage

    from . import storage

    (path,) = storage.reference_paths([image_id], noun="source image")
    try:
        with PILImage.open(path) as handle:
            check_upscaler_size(*handle.size)
    except (OSError, PILImage.DecompressionBombError) as exc:
        raise ValueError(f"source image {image_id} could not be read for {ANIME6B_LABEL}: {exc}") from exc


def _upscale_cancel(should_cancel) -> None:
    if should_cancel is not None and should_cancel():
        from .backends.base import Cancelled

        raise Cancelled


def _anime6b_path():
    from pathlib import Path

    from huggingface_hub.constants import HF_HOME

    return Path(HF_HOME) / "local-diffusion" / "upscalers" / ANIME6B_SHA256 / "RealESRGAN_x4plus_anime_6B.pth"


def _verify_anime6b(path) -> None:
    import hashlib

    if path.stat().st_size != ANIME6B_BYTES:
        raise ValueError(f"{ANIME6B_LABEL} has the wrong file size; the cached weights need repair")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(262_144):
            digest.update(chunk)
    if digest.hexdigest() != ANIME6B_SHA256:
        raise ValueError(f"{ANIME6B_LABEL} failed its SHA256 check; the cached weights need repair")


def anime6b_weights(should_cancel=None):
    """Fetch the one pinned author asset only after an explicit job opt-in.

    Existing wrong-sized/hash files refuse rather than being overwritten.
    A new download is capped, verified and atomically published; cleanup only
    touches the unique partial file this call created.
    """
    import os
    import shutil
    import tempfile
    import urllib.request
    from pathlib import Path

    _upscale_cancel(should_cancel)
    target = _anime6b_path()
    if target.exists():
        _verify_anime6b(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < ANIME6B_BYTES * 2:
        raise OSError(f"not enough free disk space to download {ANIME6B_LABEL}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix="anime6b-", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(ANIME6B_URL, headers={"User-Agent": "Local-Diffusion"})
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != ANIME6B_BYTES:
                    raise ValueError(f"{ANIME6B_LABEL} download has the wrong file size")
                size = 0
                while chunk := response.read(min(262_144, ANIME6B_BYTES - size + 1)):
                    _upscale_cancel(should_cancel)
                    size += len(chunk)
                    if size > ANIME6B_BYTES:
                        raise ValueError(f"{ANIME6B_LABEL} download exceeded its expected file size")
                    output.write(chunk)
        _upscale_cancel(should_cancel)
        _verify_anime6b(temporary)
        if target.exists():  # Another process may have completed this same pin.
            _verify_anime6b(target)
        else:
            os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _anime6b_net():
    """The pinned BasicSR RRDBNet's 4x, six-block configuration.

    Adapted from BasicSR 651835a1b9d38dbbdaf45750f56906be2364f01a,
    basicsr/archs/rrdbnet_arch.py (Apache-2.0). Parameter names and forward
    arithmetic are unchanged. Registry, unused scale-1/2 branches and custom
    initialization are omitted: every tensor is strictly restored below.
    Full licences and attribution: LICENSES/RealESRGAN-Anime6B.txt.
    Imports stay local so status, schemas and previews remain cold.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(96, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 32, 3, 1, 1)
            self.conv4 = nn.Conv2d(160, 32, 3, 1, 1)
            self.conv5 = nn.Conv2d(192, 64, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = ResidualDenseBlock()
            self.rdb2 = ResidualDenseBlock()
            self.rdb3 = ResidualDenseBlock()

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
            self.body = nn.Sequential(*(RRDB() for _ in range(6)))
            self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    # Constructing Conv2d runs random initialization even though strict state
    # loading replaces it. Do not consume the diffusion process's RNG state.
    with torch.random.fork_rng(devices=[]):
        return RRDBNet()


def anime6b_upscale(image: "Image", should_cancel=None, *, device: str = "cpu") -> "Image":
    """The frozen GAN preprocess: 4x, FP32, 256px tiles, 10px context.

    Follows Real-ESRGAN's RGB conversion, right/bottom reflection pre-pad,
    tile context/cropping and round-to-8-bit output. Tiles and the net leave
    the device before the VAE runs. These pixel-deciding choices belong to
    ANIME6B: changing them needs a new reproduction identifier.
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage
    from torch.nn import functional as F

    check_upscaler_size(*image.size)
    path = anime6b_weights(should_cancel)
    _upscale_cancel(should_cancel)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("params_ema"), dict):
        raise ValueError(f"{ANIME6B_LABEL} weights have no params_ema state")
    model = _anime6b_net()
    model.load_state_dict(state["params_ema"], strict=True)
    del state
    _upscale_cancel(should_cancel)
    model.eval().to(device=device, dtype=torch.float32)
    source = image.convert("RGB")
    width, height = source.size
    # App gallery images are >=256px. Still handle a tiny test/input safely.
    pad = min(10, width - 1, height - 1)
    inputs = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    if pad:
        inputs = F.pad(inputs, (0, pad, 0, pad), "reflect")
    result = PILImage.new("RGB", (width * 4, height * 4))
    try:
        with torch.no_grad():
            for top in range(0, height, 256):
                for left in range(0, width, 256):
                    _upscale_cancel(should_cancel)
                    right, bottom = min(width, left + 256), min(height, top + 256)
                    x0, y0 = max(0, left - 10), max(0, top - 10)
                    x1, y1 = min(width + pad, right + 10), min(height + pad, bottom + 10)
                    tile = inputs[:, :, y0:y1, x0:x1].to(device)
                    output = model(tile)
                    _upscale_cancel(should_cancel)
                    pixels = output[0, :, (top - y0) * 4:(bottom - y0) * 4,
                                    (left - x0) * 4:(right - x0) * 4]
                    array = pixels.clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8).permute(1, 2, 0).numpy()
                    result.paste(PILImage.fromarray(array), (left * 4, top * 4))
                    del tile, output, pixels, array
    finally:
        del model
    return result

# What the sibling's `__call__` has to take to be a refine pass at all: the
# source and how far to re-denoise it, and the two things `load()` insists on
# of any pipeline - the per-image seed and the cancellation point.
REQUIRED = frozenset({"image", "strength", "generator", "callback_on_step_end"})

# And an inpaint's: the same, and the mask. Every inpaint class diffusers 0.40
# exports for the catalogue's pipelines takes all of it.
INPAINT_REQUIRED = REQUIRED | {"mask_image"}

# The sibling kinds, by the infix diffusers names each class with.
KINDS = ("Img2Img", "Inpaint")

# The arguments `DiffusersBackend.generate` hands a pipeline besides the size,
# which the source image carries (SDXL's image-to-image call has no width or
# height at all). A sibling must take every one of these its text-to-image
# class takes: one that dropped `negative_prompt` or `sigmas` would render a
# refine the row then describes wrongly, and the drop would be silent.
CARRIED = frozenset(
    {
        "prompt",
        "num_inference_steps",
        "guidance_scale",
        "num_images_per_prompt",
        "generator",
        "callback_on_step_end",
        "negative_prompt",
        "true_cfg_scale",
        "sigmas",
        "use_pe",
        "system_prompt",
    }
)


def sibling(pipeline: type, kind: str = "Img2Img") -> type | None:
    """The image-to-image (or inpaint) class diffusers ships beside this one.

    Found by the library's own naming - `StableDiffusionXLPipeline`'s is
    `StableDiffusionXLImg2ImgPipeline`, its inpaint class
    `StableDiffusionXLInpaintPipeline` - and only for the class diffusers
    itself exports under that name: a local pipeline that merely shares one
    (the tests' stand-ins, a custom class) has components nobody checked
    against the sibling's. None where there is no such class, or where its
    call could not run what the text-to-image one runs (`REQUIRED`, or
    `INPAINT_REQUIRED` for an inpaint, and `CARRIED`).
    """
    if kind not in KINDS:
        raise ValueError(f"no sibling kind {kind!r}; one of {', '.join(KINDS)}")
    name = getattr(pipeline, "__name__", "")
    if not name.endswith("Pipeline") or any(infix in name for infix in KINDS):
        return None
    import diffusers

    if getattr(diffusers, name, None) is not pipeline:
        return None
    found = getattr(diffusers, f"{name[: -len('Pipeline')]}{kind}Pipeline", None)
    if found is None:
        return None
    try:
        own = set(inspect.signature(pipeline.__call__).parameters)
        params = set(inspect.signature(found.__call__).parameters)
    except (TypeError, ValueError):
        return None
    required = INPAINT_REQUIRED if kind == "Inpaint" else REQUIRED
    if not required <= params or (CARRIED & own) - params:
        return None
    return found


def library_strength(cls: type) -> float | None:
    """The `strength` default of a sibling's own call signature, or None.

    Where an inpaint's Strength box starts until a profile has a measured
    value: no catalogue author documents inpainting with base weights, so the
    only number there is is the library's, and it is said as the library's,
    never the checkpoint's. Read rather than copied, so an upgrade of
    diffusers cannot leave it stale.
    """
    try:
        param = inspect.signature(cls.__call__).parameters.get("strength")
    except (TypeError, ValueError):
        return None
    default = None if param is None else param.default
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return None
    return float(default)


def build(pipe: Any, cls: type) -> Any:
    """`cls` over the loaded pipeline's own components.

    Constructed directly rather than through `cls.from_pipe`, which ends in
    `new_pipeline.to(dtype=...)` - float32 unless told otherwise - over
    modules this pipeline is still using, torchao-quantized ones among them.
    A constructor only registers what it is given: no weight is read, cast or
    moved, and the offload hooks, which live on the modules, come along.
    Config values the sibling's constructor takes (SDXL's
    `force_zeros_for_empty_prompt`) are the loaded pipeline's.
    """
    components = pipe.components
    config = pipe.config
    kwargs: dict[str, Any] = {}
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in components:
            kwargs[name] = components[name]
        elif name in config:
            kwargs[name] = config[name]
        elif param.default is param.empty:
            raise RuntimeError(
                f"{cls.__name__} needs a {name}, which {type(pipe).__name__} has not got"
            )
    return cls(**kwargs)


def start_index(steps: int, strength: float, model_id: str, inpaint: bool = False) -> int | None:
    """The supported checkpoint's integer start in its denoising schedule.

    SDXL floors the steps that run; the flow siblings floor those skipped.
    The inpaint classes round as their image-to-image ones do (each one's
    `get_timesteps` is the same arithmetic), so `inpaint` only changes which
    capability the profile has to have. Keep the operations in diffusers'
    order: rearranging the multiplication changes boundary cases through
    floating-point rounding. Unknown profiles return None rather than
    promising two strengths produce the same image.
    """
    # Imported here to keep this a leaf during model_defaults' initialization.
    from .model_defaults import defaults_for

    profile = defaults_for(model_id)
    if not profile["inpaint_supported" if inpaint else "refine_supported"]:
        return None
    if profile["mu_supported"]:
        return int(max(steps - min(steps * strength, steps), 0))
    return max(steps - min(int(steps * strength), steps), 0)


def steps_run(
    steps: int, strength: float, model_id: str | None = None, inpaint: bool = False
) -> int:
    """The profiled sibling's count, or the floor until its family is known.

    The progress callback still reads the actual pipeline's num_timesteps;
    this count is for the bar before that first callback and timing history.
    """
    start = start_index(steps, strength, model_id, inpaint) if model_id is not None else None
    return max(1, int(steps * strength) if start is None else steps - start)


def leaves_a_step(steps: int, strength: float) -> bool:
    """Whether every pipeline here runs at least one step at this strength.

    SDXL's rule, the stricter of the two: `int(steps * strength)` steps, and
    the call raises below one. Checked after the capability refusal, on each
    resolved grid cell, so a 4-step checkpoint at strength 0.2 is refused at
    submit rather than at generation.
    """
    return int(steps * strength) >= 1


def scaled(width: int, height: int, factor: float) -> tuple[int, int]:
    """A source's size times `factor`, each side on the VAE's 16-pixel grid.

    Half rounds up, written out: `round()` rounds half to even and the page's
    `Math.round` does not, and the Refine button and `--scale` must land on
    the same size for the same image.
    """
    return (
        max(16, int(width * factor / 16 + 0.5) * 16),
        max(16, int(height * factor / 16 + 0.5) * 16),
    )


CROP_ANCHORS = (
    "center", "top", "bottom", "left", "right",
    "top-left", "top-right", "bottom-left", "bottom-right",
)


def fitted(image: "Image", width: int, height: int, crop_anchor: str = "center") -> "Image":
    """`image` at exactly `width` x `height`: scaled to cover, then cropped.

    Never stretched. A size that is the source's own times a factor loses a
    few pixels at most to the 16-pixel grid; a size of another shape is a
    reframing, and what falls outside it is cropped rather than squeezed in.
    Lanczos: the refine pass redraws the detail, so the resize only has to
    hand it clean edges to start from.
    """
    from PIL import Image as PILImage

    if crop_anchor not in CROP_ANCHORS:
        raise ValueError(f"unknown crop anchor {crop_anchor!r}")
    if image.size == (width, height):
        return image
    cover = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(width, round(image.width * cover)), max(height, round(image.height * cover))),
        PILImage.LANCZOS,
    )
    horizontal, vertical = resized.width - width, resized.height - height
    left = 0 if "left" in crop_anchor else horizontal if "right" in crop_anchor else horizontal // 2
    top = 0 if "top" in crop_anchor else vertical if "bottom" in crop_anchor else vertical // 2
    return resized.crop((left, top, left + width, top + height))


def source_image(image_id: int, width: int, height: int, *, upscaler: str | None = None,
                 should_cancel=None, device: str = "cpu", crop_anchor: str = "center") -> "Image":
    """The gallery image a refine pass starts from, at the requested size.

    Decoded here and the file handle closed, as the reference images are: the
    pipeline holds it across the whole denoise. Raises ValueError naming an
    id the gallery has not got, or a row whose file is gone.
    """
    from PIL import Image as PILImage

    from . import storage

    (path,) = storage.reference_paths([image_id], noun="source image")
    with PILImage.open(path) as handle:
        if upscaler is not None:
            check_upscaler_size(*handle.size)
        source = handle.convert("RGB")
    if upscaler is not None:
        if upscaler != ANIME6B:
            raise ValueError("the requested Refine upscaler is not supported by this build")
        source = anime6b_upscale(source, should_cancel, device=device)
    return fitted(source, width, height, crop_anchor)


def inpaint_source(image_id: int, width: int, height: int) -> "Image":
    """The gallery image an inpaint starts from, at its own size.

    Never resized: the mask was painted on these pixels, and the composite
    puts them back exactly. A request at another size is a mistake the
    submit check names first; here it raises rather than misaligning the mask.
    """
    from PIL import Image as PILImage

    from . import storage

    (path,) = storage.reference_paths([image_id], noun="source image")
    with PILImage.open(path) as handle:
        image = handle.convert("RGB")
    if image.size != (width, height):
        raise ValueError(
            f"source image {image_id} is {image.width}x{image.height}; "
            f"an inpaint runs at its source's size, not {width}x{height}"
        )
    return image


def outpaint_source(image_id: int, spec: Any, width: int, height: int) -> "Image":
    """The picture an outpaint hands the inpaint class: the gallery image on
    its padded canvas, the new area holding the fill `spec` names
    (`outpaint.padded`), at exactly `width` x `height`.

    The source is never resized, as an inpaint's is not; a request whose
    size is not the source's plus the padding is a mistake the submit check
    names first, and here it raises rather than misplacing the source.
    """
    from PIL import Image as PILImage

    from . import outpaint, storage

    (path,) = storage.reference_paths([image_id], noun="source image")
    with PILImage.open(path) as handle:
        image = handle.convert("RGB")
    canvas = (image.width + spec.left + spec.right, image.height + spec.top + spec.bottom)
    if canvas != (width, height):
        raise ValueError(
            f"source image {image_id} is {image.width}x{image.height}; an outpaint of it by "
            f"{outpaint.sides(spec)} runs at {canvas[0]}x{canvas[1]}, not {width}x{height}"
        )
    return outpaint.padded(image, spec)


def canonical_mask(image: "Image") -> "Image":
    """Any picture given as a mask, as the one `L` channel a mask is kept in.

    Pillow's own `convert("L")` is right for `L`, `1`, `RGB` and an opaque
    palette, and wrong for two kinds an editor commonly exports: it clips a
    16-bit level instead of scaling it (32768 of 65535 becomes 255, and a
    near-black "keep" of 100 a grey the composite blends by), and it drops
    alpha, so white strokes on a transparent white layer read as all white.
    16-bit levels are scaled to 8 bits; a picture with transparency is laid
    on opaque black first, so what is transparent is kept. Any other mode
    (`F`, `CMYK`, ...) is refused by name, as a `ValueError`.
    """
    from PIL import Image as PILImage

    mode = image.mode
    if mode == "L":
        return image.copy()
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        import numpy as np

        levels = np.asarray(image, dtype=np.int64)
        return PILImage.fromarray((np.clip(levels, 0, 65535) >> 8).astype(np.uint8), "L")
    transparent = "A" in image.getbands() or (mode == "P" and "transparency" in image.info)
    if transparent:
        rgba = image.convert("RGBA")
        return PILImage.alpha_composite(PILImage.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba).convert("L")
    if mode in ("1", "RGB", "P"):
        return image.convert("L")
    raise ValueError(f"a mask is a greyscale, RGB or palette picture, not one in mode {mode}")


def mask_image(mask: str, width: int, height: int) -> "Image":
    """The stored mask a request names by hash, as one luminance channel.

    Resolved by `storage.mask_path`, the one place a hash becomes a file -
    never a path from the request. White is repainted and black kept, as
    every inpaint class reads it; grey levels are the feather, which the
    classes binarise and `composite` blends with.
    """
    from PIL import Image as PILImage

    from . import storage

    path = storage.mask_path(mask)
    with PILImage.open(path) as handle:
        image = handle.convert("L")
    if image.size != (width, height):
        raise ValueError(
            f"mask {mask[:16]} is {image.width}x{image.height}, not the {width}x{height} it covers"
        )
    return image


# The side of the square of pixels one latent cell covers in both enabled
# inpaint families (SDXL's VAE and Z-Image's both downsample by 8), where the
# runner names none of its own (`vae_scale_factor`).
LATENT_BLOCK = 8


def pipeline_mask(mask: "Image", block: int = LATENT_BLOCK) -> "Image":
    """The mask the inpaint class is handed for a painted `mask`: every
    latent cell that holds any painted pixel, all white; the rest black.

    Both enabled classes binarise their mask at 0.5 and then shrink it to the
    latent grid by nearest sampling - the top-left pixel of each `block`
    square - so the painted mask as it is repaints only the cells whose
    corner pixel is past 128. A stroke narrower than a cell could repaint
    nothing, the top and left edge of a mask were never repainted, and a
    feather was repainted only on its brighter half. `composite` blends with
    the painted grey levels, so what the class repaints has to cover every
    pixel the composite gives any weight to: thresholded at > 0 and widened
    to whole cells (a max-pool onto the latent grid and back), it does, and
    each cell's corner pixel then says what the whole cell is.
    """
    from PIL import Image as PILImage

    block = max(1, int(block))
    width, height = mask.size
    painted = mask.convert("L").point(lambda level: 255 if level > 0 else 0)
    if block == 1:
        return painted
    # Padded out to whole cells, the pad unpainted.
    cells_w, cells_h = -(-width // block), -(-height // block)
    padded = PILImage.new("L", (cells_w * block, cells_h * block), 0)
    padded.paste(painted, (0, 0))
    # `reduce` averages each cell: any painted pixel leaves it above zero.
    cells = padded.reduce(block).point(lambda level: 255 if level > 0 else 0)
    return cells.resize(padded.size, PILImage.NEAREST).crop((0, 0, width, height))


def composite(mask: "Image", source: "Image", image: "Image", processor: Any = None) -> "Image":
    """`image` over `source` where `mask` is white, `source` where it is black.

    The library's own `VaeImageProcessor.apply_overlay` - what the inpaint
    classes call themselves with `padding_mask_crop` - with the mask's grey
    levels as the alpha, so a feathered edge blends and every unmasked pixel
    is the source's exactly rather than its VAE round trip. Always, not a
    setting: it is what an inpaint is in this app, and the PNG records it as
    `"mask_blend": "overlay"`. `processor` is the runner's own where there is
    one; any `VaeImageProcessor` does the same without crop coordinates.
    """
    if not (mask.size == source.size == image.size):
        raise ValueError(
            f"cannot composite a {image.width}x{image.height} result over a "
            f"{source.width}x{source.height} source with a {mask.width}x{mask.height} mask"
        )
    if not callable(getattr(processor, "apply_overlay", None)):
        from diffusers.image_processor import VaeImageProcessor

        processor = VaeImageProcessor()
    return processor.apply_overlay(mask.convert("L"), source.convert("RGB"), image.convert("RGB"))
