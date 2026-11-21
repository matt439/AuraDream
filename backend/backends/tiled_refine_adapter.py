"""Serial Tiled Refine over the loaded SDXL inpaint sibling."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from typing import TYPE_CHECKING, Any

from .. import tiled_refine, tiled_refine_admission
from ..seeds import MAX_BATCH, MAX_SEED, wrap_seed
from ..tiled_refine_contract import StoredTiledRefine
from .base import CancelFn, Cancelled, ProgressFn

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas import GenerateRequest
    from .diffusers_backend import DiffusersBackend


_INPAINT_PARAMS = frozenset({
    "prompt", "negative_prompt", "image", "mask_image", "width", "height", "strength",
    "num_inference_steps", "guidance_scale", "num_images_per_prompt", "generator", "callback_on_step_end",
})
_SHARED_COMPONENTS = ("unet", "vae", "scheduler", "text_encoder", "text_encoder_2")
ResultFn = Callable[[tiled_refine.Plan, tuple[tiled_refine.TileResult, ...]], None]


def generate_candidate(
    backend: DiffusersBackend, request: GenerateRequest, seed: int,
    on_progress: ProgressFn | None, should_cancel: CancelFn | None, on_result: ResultFn | None,
) -> list[Image]:
    """Run one complete composite at a time; release owned images on failure."""
    spec = request.validate_tiled_refine(require_bound=True)
    if not isinstance(spec, StoredTiledRefine):
        raise ValueError("Bind the original Tiled Refine source before rendering.")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("Tiled Refine requires an exact nonnegative storable seed.")
    count = request.num_images
    if type(count) is not int or not 1 <= count <= MAX_BATCH:
        raise ValueError("Tiled Refine needs a valid image count.")
    request = backend.applied(backend._require_execution(request))
    pipe = backend._require_tiled_refine_owner()
    if should_cancel is not None and should_cancel():
        raise Cancelled()
    with closing(tiled_refine_admission.load_bound_source(spec)) as canvas, backend._measure_generation_peak():
        runner = backend._inpainter(pipe)
        if (
            type(runner).__name__ != "StableDiffusionXLInpaintPipeline"
            or getattr(runner, "vae_scale_factor", None) != tiled_refine.RECIPE.latent_block
            or any(
                getattr(pipe, name, None) is None or getattr(runner, name, None) is not getattr(pipe, name)
                for name in _SHARED_COMPONENTS
            )
        ):
            raise ValueError("Tiled Refine needs the loaded SDXL owner's exact inpaint sibling.")
        missing = sorted(_INPAINT_PARAMS - backend._inpaint_params)
        if missing:
            raise ValueError(f"The Tiled Refine inpaint sibling is missing {', '.join(missing)}.")

        def prove_tile() -> None:
            if backend._require_tiled_refine_owner() is not pipe:
                raise ValueError("The Tiled Refine pipeline changed while rendering.")
            tiled_refine_admission.prove_root(spec)

        def render_tile(call: tiled_refine.TileCall) -> Image:
            return _invoke_tile(backend, pipe, runner, request, call)

        def progress(ordinal: int, completed: int, steps: int) -> None:
            if on_progress is not None:
                done = (image_index * tiled_refine.TILE_COUNT + ordinal) * steps + completed
                on_progress(done, count * tiled_refine.TILE_COUNT * steps)

        images: list[Image] = []
        try:
            for image_index in range(count):
                result = tiled_refine.run(
                    canvas, wrap_seed(seed + image_index), render_tile, before_tile=prove_tile,
                    should_cancel=should_cancel, on_progress=progress,
                )
                images.append(result.image)
                prove_tile()
                if on_result is not None:
                    on_result(result.plan, result.tiles)
            if should_cancel is not None and should_cancel():
                raise Cancelled()
        except BaseException:
            for image in images:
                image.close()
            raise
        return images


def _invoke_tile(
    backend: DiffusersBackend, pipe: Any, runner: Any, request: GenerateRequest, call: tiled_refine.TileCall,
) -> Any:
    import torch

    def step_end(active, step, _timestep, values):
        call.on_step(step + 1, active.num_timesteps)
        return values

    width, height = call.plan.working
    kwargs = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "image": call.image,
        "mask_image": call.pipeline_mask,
        "width": width,
        "height": height,
        "strength": request.strength,
        "num_inference_steps": request.steps,
        "guidance_scale": request.guidance_scale,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device="cpu").manual_seed(call.seed),
        "callback_on_step_end": step_end,
    }
    try:
        with torch.no_grad():
            images = backend._invoke_pipeline(runner, kwargs).images
    except BaseException as error:
        backend._cleanup_failed_text_image(pipe, error, pass_name="Tiled Refine")
        raise
    try:
        # The sibling borrows the weights. The owner has the offload hooks.
        pipe.maybe_free_model_hooks()
    except BaseException as error:
        from PIL import Image as PILImage

        candidates = images if isinstance(images, (list, tuple)) else (images,)
        for image in candidates:
            if isinstance(image, PILImage.Image) and all(
                image is not borrowed for borrowed in (call.image, call.mask, call.pipeline_mask)
            ):
                image.close()
        backend._cleanup_failed_text_image(pipe, error, pass_name="Tiled Refine")
        raise
    return images[0] if isinstance(images, (list, tuple)) and len(images) == 1 else images
