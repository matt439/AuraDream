"""The refine pass: a gallery image scaled up and re-denoised part of the way.

The size guides say it in the authors' own words - FLUX.1's "render at 1-2 MP
and upscale", WAI's 1024x1344 followed by a 1.5x hires pass at denoising
strength 0.35-0.5 - and until this module the app could only render directly
at the requested size, where the same seed is a new sample and a checkpoint
past its documented range doubles subjects or drops them. A refine pass keeps
the picture: the source is resized to the requested size, encoded by the VAE,
noised to the point on the schedule its `strength` names and denoised from
there under the request's own prompt and settings.

Nothing new is loaded for it. diffusers ships an image-to-image class beside
most text-to-image ones (`FluxImg2ImgPipeline` beside `FluxPipeline`), taking
the same components, so the backend builds that sibling over the pipeline it
already holds (`build`) and the two share every module, hook and adapter.
Which checkpoints have one is therefore the installed diffusers' answer
(`sibling`), not a table here: FLUX.1, SDXL, Z-Image and Qwen-Image do; Krea
2, FLUX.2 [klein], ERNIE-Image and Lumina 2 do not, and neither do the local
Anima and NewBie pipelines.

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
from typing import TYPE_CHECKING, Any

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


def fitted(image: "Image", width: int, height: int) -> "Image":
    """`image` at exactly `width` x `height`: scaled to cover, centre-cropped.

    Never stretched. A size that is the source's own times a factor loses a
    few pixels at most to the 16-pixel grid; a size of another shape is a
    reframing, and what falls outside it is cropped rather than squeezed in.
    Lanczos: the refine pass redraws the detail, so the resize only has to
    hand it clean edges to start from.
    """
    from PIL import Image as PILImage

    if image.size == (width, height):
        return image
    cover = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(width, round(image.width * cover)), max(height, round(image.height * cover))),
        PILImage.LANCZOS,
    )
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def source_image(image_id: int, width: int, height: int) -> "Image":
    """The gallery image a refine pass starts from, at the requested size.

    Decoded here and the file handle closed, as the reference images are: the
    pipeline holds it across the whole denoise. Raises ValueError naming an
    id the gallery has not got, or a row whose file is gone.
    """
    from PIL import Image as PILImage

    from . import storage

    (path,) = storage.reference_paths([image_id], noun="source image")
    with PILImage.open(path) as handle:
        return fitted(handle.convert("RGB"), width, height)


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
