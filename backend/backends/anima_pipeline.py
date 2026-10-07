"""Standard pipeline interface around Diffusers' Anima text-to-image blocks.

Keep the upstream encoding, conditioning, schedule and decode implementations.
The small loop extension supplies the app's mandatory per-step callback; the
ordinary DiffusionPipeline owns component loading, quantization and offload.
"""
from __future__ import annotations

from typing import Callable

# Set remote-kernel policy before importing Diffusers, including direct imports.
from .. import config as _config  # noqa: F401

import torch
from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.anima.denoise import AnimaDenoiseStep
from diffusers.modular_pipelines.anima.encoders import AnimaTextEncoderStep
from diffusers.modular_pipelines.anima.modular_blocks_anima import AnimaCoreDenoiseStep, AnimaDecodeStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast


class _CallbackDenoiseStep(AnimaDenoiseStep):
    """Report only completed scheduler steps, allowing Cancelled to unwind."""

    callback: Callable | None = None

    def loop_step(self, components, state, *, i, t):
        components, state = super().loop_step(components, state, i=i, t=t)
        if self.callback is not None:
            updates = self.callback(components, i, t, {"latents": state.latents})
            if updates is not None and "latents" in updates:
                state.latents = updates["latents"]
        return components, state


class AnimaTurboPipeline(DiffusionPipeline):
    """Pinned Turbo weights with the upstream Anima inference implementation.

    A dedicated interface intentionally exposes only supported app controls.
    User LoRAs and custom Mu are not enabled by this initial integration.
    """

    model_cpu_offload_seq = "text_encoder->text_conditioner->transformer->vae"
    _optional_components: list[str] = []

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
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer, t5_tokenizer=t5_tokenizer,
            text_encoder=text_encoder, text_conditioner=text_conditioner,
            transformer=transformer, vae=vae, scheduler=scheduler,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._encode = AnimaTextEncoderStep()
        self._denoise = AnimaCoreDenoiseStep()
        self._step = _CallbackDenoiseStep()
        self._denoise.sub_blocks["denoise"] = self._step
        self._decode = AnimaDecodeStep()

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** len(self.vae.temperal_downsample)

    @property
    def num_channels_latents(self) -> int:
        return self.transformer.config.in_channels

    @property
    def default_height(self) -> int:
        return 128 * self.vae_scale_factor

    @property
    def default_width(self) -> int:
        return self.default_height

    @staticmethod
    def _unpack_latents(latents, height, width):
        # Anima already stores Qwen-VAE latents as B,C,1,H,W. The shared
        # preview helper removes the singleton frame dimension.
        return latents

    def set_progress_bar_config(self, **kwargs) -> None:
        super().set_progress_bar_config(**kwargs)
        self._step.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
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
        # The upstream guider guides at every value except exactly 1. Align
        # the actual pass with this app's >1 threshold, including CFG 0.
        # A fresh guider also clears the previous call's last-step state.
        self.guider = ClassifierFreeGuidance(
            guidance_scale=guidance_scale, enabled=guidance_scale > 1.0,
        )
        state = PipelineState(values={
            "prompt": prompt, "negative_prompt": negative_prompt,
            "height": height, "width": width,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator, "output_type": output_type,
            "max_sequence_length": 512,
        })
        self._step.callback = callback_on_step_end
        try:
            _, state = self._encode(self, state)
            # SequentialPipelineBlocks logs every escaping exception as an
            # error. Call its blocks directly so cooperative cancellation is
            # handled by the job layer without a misleading error traceback.
            for block in self._denoise.sub_blocks.values():
                _, state = block(self, state)
            _, state = self._decode(self, state)
            return ImagePipelineOutput(images=state.get("images"))
        finally:
            self._step.callback = None
            # In particular, cancellation must offload the resident denoiser.
            self.maybe_free_model_hooks()


class AnimaAestheticPipeline(AnimaTurboPipeline):
    """Aesthetic v1.1 shares Anima's blocks with non-distilled call defaults."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
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
            prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            negative_prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt,
            generator=generator, callback_on_step_end=callback_on_step_end,
            output_type=output_type,
        )
