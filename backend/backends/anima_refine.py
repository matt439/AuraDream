"""Private image-to-image pass over Diffusers' Anima blocks (GEN#48 stage 1).

Not registered anywhere: `refine.sibling` still answers None for every Anima
class, the Anima profiles keep `refine_supported` False, and nothing in the
backend imports this module. It exists so the pass can be qualified on CPU
now and measured on the card later (docs/maintainer/plan-anima-refine.md)
without opening an unqualified public route.

The upstream implementation is reused unchanged: `AnimaImg2ImgVaeEncoderStep`
(preprocess, posterior-mode encode, latents mean/std normalisation),
`AnimaImg2ImgCoreDenoiseStep` (strength-sliced timesteps on the supplied
scheduler, `scale_noise` at the first sliced timestep, the denoising loop)
and `AnimaDecodeStep`. The only extension is the text pipeline's own
completed-step callback, plus `num_timesteps` set to the sliced count before
the first callback, so progress, previews and cancellation see the tail
that actually runs.

The classes take the loaded text pipeline's components through
`refine.build` - no `from_pipe`, cast, download or placement - and so share
every module and hook with it.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

# Set remote-kernel policy before importing Diffusers, including direct imports.
from .. import config as _config  # noqa: F401

import torch
from diffusers.guiders.classifier_free_guidance import ClassifierFreeGuidance
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from diffusers.modular_pipelines.anima.before_denoise import AnimaImg2ImgSetTimestepsStep
from diffusers.modular_pipelines.anima.encoders import AnimaImg2ImgVaeEncoderStep
from diffusers.modular_pipelines.anima.modular_blocks_anima import AnimaImg2ImgCoreDenoiseStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from PIL.Image import Image
from transformers import Qwen2Tokenizer, Qwen3Model
from transformers.models.t5.tokenization_t5 import T5Tokenizer as T5TokenizerFast

from .anima_pipeline import AnimaAestheticPipeline, AnimaPreviewPipeline, AnimaTurboPipeline
from .schedule import static_shift


def start_index(steps: int, strength: float) -> int:
    """Where the Anima pass starts: the skipped steps, floored.

    `get_timesteps` in the installed Anima `before_denoise` module, in its
    order of operations (flow rounding: eight steps at 0.4 run four). Not
    `refine.start_index`, which picks flow or SDXL rounding by `mu_supported`
    - false for Anima's fixed shift - and would answer SDXL's five.
    """
    init_timestep = min(steps * strength, steps)
    return int(max(steps - init_timestep, 0))


def steps_run(steps: int, strength: float) -> int:
    """How many denoising steps the pass runs (zero refuses the call)."""
    return steps - start_index(steps, strength)


def refuse_empty(steps: int, strength: float) -> None:
    """Raise ValueError for a strength the pass cannot run.

    Outside (0, 1], or a slice the flooring leaves empty (strength 0 would
    leave only the schedule's trailing 0.0). The call and `start_sigma`
    share it, so a preview never names a sigma for a refused pass.
    """
    if not 0 < strength <= 1:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    if steps_run(steps, strength) < 1:
        raise ValueError(f"strength {strength} leaves no denoising step of {steps}")


class AnimaTurboImg2ImgPipeline(AnimaTurboPipeline):
    """Anima's image-to-image blocks with the app's callback contract.

    The text pipeline's components and call conventions: CFG guides only
    above 1 (a fresh guider per call), the callback is cleared and the
    hooks freed in `finally`, and every block runs directly so a cooperative
    cancel unwinds without the sequential runner's error log.
    """

    # Spelled out rather than *args: `refine.build` reads this signature to
    # hand over the loaded pipeline's components by name.
    def __init__(
        self,
        tokenizer: Qwen2Tokenizer,
        t5_tokenizer: T5TokenizerFast,
        text_encoder: Qwen3Model,
        text_conditioner: AnimaTextConditioner,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLQwenImage,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer, t5_tokenizer=t5_tokenizer,
            text_encoder=text_encoder, text_conditioner=text_conditioner,
            transformer=transformer, vae=vae, scheduler=scheduler,
        )
        self._vae_encode = AnimaImg2ImgVaeEncoderStep()
        self._denoise = AnimaImg2ImgCoreDenoiseStep()
        self._denoise.sub_blocks["denoise"] = self._step
        self._num_timesteps: int | None = None

    @property
    def num_timesteps(self) -> int | None:
        """The sliced count of the call under way, set before its first step."""
        return self._num_timesteps

    def start_sigma(self, num_inference_steps: int, strength: float) -> float:
        """The sigma this pass would noise the source to, read on a copy.

        The installed timestep block is run against a scheduler built from
        the loaded one's config and given its static shift, so the answer is
        the block's own schedule and slice; the live scheduler, which a job
        may be using, is never touched. A strength the call would refuse
        raises the call's ValueError rather than reading a sigma.
        """
        refuse_empty(num_inference_steps, strength)
        live = self.scheduler
        copy = type(live).from_config(live.config)
        if callable(getattr(copy, "set_shift", None)):
            copy.set_shift(static_shift(self))
        components = SimpleNamespace(scheduler=copy, _execution_device=torch.device("cpu"))
        state = PipelineState(values={"num_inference_steps": num_inference_steps, "strength": strength})
        # The block reads only `scheduler` and `_execution_device`.
        AnimaImg2ImgSetTimestepsStep()(components, state)  # pyright: ignore[reportArgumentType]
        begin = copy.begin_index
        assert begin is not None, "get_timesteps sets the begin index"
        return float(copy.sigmas[begin])

    @torch.no_grad()
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str | list[str],
        image: Image | list[Image],
        strength: float,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        callback_on_step_end: Callable | None = None,
        output_type: str = "pil",
    ) -> ImagePipelineOutput:
        # Refused before any component runs: strength 0 slices the schedule
        # empty and the upstream noise step would fail on it.
        refuse_empty(num_inference_steps, strength)
        self.guider = ClassifierFreeGuidance(
            guidance_scale=guidance_scale, enabled=guidance_scale > 1.0,
        )
        state = PipelineState(values={
            "prompt": prompt, "negative_prompt": negative_prompt, "image": image,
            "strength": strength, "height": height, "width": width,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator, "output_type": output_type,
            "max_sequence_length": 512,
        })
        self._num_timesteps = None
        self._step.callback = callback_on_step_end
        try:
            # The blocks' annotations name AnimaModularPipeline; they read only
            # the standard components registered by the text pipeline.
            _, state = self._encode(self, state)  # pyright: ignore[reportArgumentType]
            _, state = self._vae_encode(self, state)  # pyright: ignore[reportArgumentType]
            for name, block in self._denoise.sub_blocks.items():
                _, state = block(self, state)
                if name == "set_timesteps":
                    self._num_timesteps = len(state.get("timesteps"))
            _, state = self._decode(self, state)
            return ImagePipelineOutput(images=state.values["images"])
        finally:
            self._step.callback = None
            self.maybe_free_model_hooks()


class AnimaAestheticImg2ImgPipeline(AnimaTurboImg2ImgPipeline):
    """Aesthetic v1.1's non-distilled call defaults over the same pass."""

    @torch.no_grad()
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str | list[str],
        image: Image | list[Image],
        strength: float,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        callback_on_step_end: Callable | None = None,
        output_type: str = "pil",
    ) -> ImagePipelineOutput:
        return super().__call__(
            prompt=prompt, image=image, strength=strength, height=height, width=width,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            negative_prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt,
            generator=generator, callback_on_step_end=callback_on_step_end,
            output_type=output_type,
        )


class AnimaPreviewImg2ImgPipeline(AnimaAestheticImg2ImgPipeline):
    """The 40-layer preview's defaults equal Aesthetic's."""


# Each local text class's private counterpart. Deliberately not consulted by
# `refine.sibling`: release adds an exact-class extension after the card.
IMG2IMG: dict[type, type] = {
    AnimaTurboPipeline: AnimaTurboImg2ImgPipeline,
    AnimaAestheticPipeline: AnimaAestheticImg2ImgPipeline,
    AnimaPreviewPipeline: AnimaPreviewImg2ImgPipeline,
}
