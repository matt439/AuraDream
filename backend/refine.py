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


def start_index(steps: int, strength: float, model_id: str) -> int | None:
    """The supported checkpoint's integer start in its denoising schedule.

    SDXL floors the steps that run; the flow siblings floor those skipped.
    Keep the operations in diffusers' order: rearranging the multiplication
    changes boundary cases through floating-point rounding. Unknown profiles
    return None rather than promising two strengths produce the same image.
    """
    # Imported here to keep this a leaf during model_defaults' initialization.
    from .model_defaults import defaults_for

    profile = defaults_for(model_id)
    if not profile["refine_supported"]:
        return None
    if profile["mu_supported"]:
        return int(max(steps - min(steps * strength, steps), 0))
    return max(steps - min(int(steps * strength), steps), 0)


def steps_run(steps: int, strength: float, model_id: str | None = None) -> int:
    """The profiled sibling's count, or the floor until its family is known.

    The progress callback still reads the actual pipeline's num_timesteps;
    this count is for the bar before that first callback and timing history.
    """
    start = start_index(steps, strength, model_id) if model_id is not None else None
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
