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

The source is a gallery *id*, for the reason a reference image is: a path
from the request joined onto a directory is an arbitrary-file read.

A leaf module like `seeds`: `schemas`, `grid` and `storage` all import it, so
it imports none of them at the top.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

# The strength a pass runs at where the checkpoint's profile names none. Where
# a strength starts is the profile's (`refine_strength` in `model_defaults`,
# through `schemas.default_strength`), because a strength is an index into the
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


def sibling(pipeline: type) -> type | None:
    """The image-to-image class diffusers ships beside this pipeline class.

    Found by the library's own naming - `StableDiffusionXLPipeline`'s is
    `StableDiffusionXLImg2ImgPipeline` - and only for the class diffusers
    itself exports under that name: a local pipeline that merely shares one
    (the tests' stand-ins, a custom class) has components nobody checked
    against the sibling's. None where there is no such class, or where its
    call could not run what the text-to-image one runs (`REQUIRED`, `CARRIED`).
    """
    name = getattr(pipeline, "__name__", "")
    if not name.endswith("Pipeline") or "Img2Img" in name:
        return None
    import diffusers

    if getattr(diffusers, name, None) is not pipeline:
        return None
    found = getattr(diffusers, f"{name[: -len('Pipeline')]}Img2ImgPipeline", None)
    if found is None:
        return None
    try:
        own = set(inspect.signature(pipeline.__call__).parameters)
        params = set(inspect.signature(found.__call__).parameters)
    except (TypeError, ValueError):
        return None
    if not REQUIRED <= params or (CARRIED & own) - params:
        return None
    return found


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


def steps_run(steps: int, strength: float) -> int:
    """About how many of `steps` a refine pass at `strength` denoises.

    The pipeline starts part of the way down the schedule and runs the rest,
    so a 28-step refine at 0.4 is 11 or 12 steps of work, not 28. Which of the
    two is the pipeline's rounding - SDXL floors `steps * strength`, the flow
    pipelines round the skipped part down instead - so this is the floor, at
    least one, for the estimate and the progress bar's first frame; the step
    callback reports the count the call really made.
    """
    return max(1, int(steps * strength))


def leaves_a_step(steps: int, strength: float) -> bool:
    """Whether every pipeline here runs at least one step at this strength.

    SDXL's rule, the stricter of the two: `int(steps * strength)` steps, and
    the call raises below one. Checked when the request is built, so a
    4-step checkpoint at strength 0.2 is refused at submit rather than at the
    first cell.
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
