"""Standard pipeline interface around Diffusers' FLUX 3 [Klein] modular blocks.

diffusers 0.42.1 runs FLUX 3 Image Klein only as a `ModularPipeline`, whose
call has no `callback_on_step_end`, and the repositories carry no
`model_index.json`. This keeps the upstream encoding, latent preparation,
schedule, flow-map denoising and decode implementations, and adds what the app
needs around them: the mandatory per-step callback (progress, previews and the
cooperative cancel), a per-call guider on the undistilled Base, and component
loading from the pinned repository's own subfolders (`flux3_loading`). The
prompt upsampler block is never run inside the image call: the prompt the row
records is the prompt the model saw, as with ERNIE's `use_pe=False`. Expand
registers the upsampler as an optional component (`flux3_upsampler`), first in
the offload sequence, and runs it on its own when asked.
"""
from __future__ import annotations

import os
from typing import Any, Callable

# Set remote-kernel policy before importing Diffusers, including direct imports.
from .. import config as _config  # noqa: F401

import torch
from diffusers.guiders.classifier_free_guidance import ClassifierFreeGuidance
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep
from diffusers.modular_pipelines.flux3.denoise import Flux3DenoiseStep, Flux3DistilledDenoiseStep
from diffusers.modular_pipelines.flux3.encoders import Flux3DistilledTextEncoderStep, Flux3TextEncoderStep
from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
from diffusers.modular_pipelines.flux3.modular_blocks_flux3 import (
    Flux3CoreDenoiseStep, Flux3ImageConditionedCoreDenoiseStep, Flux3VaeEncoderSequentialStep,
)
from diffusers.modular_pipelines.flux3.modular_blocks_flux3_distilled import (
    Flux3DistilledCoreDenoiseStep, Flux3DistilledImageConditionedCoreDenoiseStep,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState, SequentialPipelineBlocks
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3_5ForConditionalGeneration, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

# The text window the encoder step may fill (its own default, passed
# explicitly so `tests/test_prompt_encoding.py` can hold `encoder_table` to it).
MAX_SEQUENCE_LENGTH = 8192


def _with_callback(base: type) -> type:
    """`base` (one of the FLUX 3 denoise loops) reporting each finished step."""

    class CallbackDenoiseStep(base):
        """Report only completed scheduler steps, allowing Cancelled to unwind."""

        callback: Callable | None = None

        def loop_step(self, components, state, **kwargs):
            components, state = super().loop_step(components, state, **kwargs)
            if self.callback is not None:
                # The loop passes `i` and `t` as keywords (Flux3DenoiseLoopWrapper).
                updates = self.callback(components, kwargs["i"], kwargs["t"], {"latents": state.latents})
                if updates is not None and "latents" in updates:
                    # loop_step annotates PipelineState but returns the loop's
                    # dynamic BlockState, which has the attribute.
                    state.latents = updates["latents"]  # pyright: ignore[reportAttributeAccessIssue]
            return components, state

    CallbackDenoiseStep.__name__ = CallbackDenoiseStep.__qualname__ = f"Callback{base.__name__}"
    return CallbackDenoiseStep


_CallbackDistilledDenoiseStep = _with_callback(Flux3DistilledDenoiseStep)
_CallbackDenoiseStep = _with_callback(Flux3DenoiseStep)


class Flux3KleinPipeline(DiffusionPipeline):
    """FLUX 3 Image Klein (distilled, flow-map, 16 steps, no guidance).

    A dedicated interface exposes only the app's controls. User LoRAs and
    layout boxes are not part of this integration; the prompt upsampler is
    never part of the image call, only of Expand (`upsample_prompt`).
    """

    # The upsampler first, as ERNIE's `pe`: under model offload it is on the
    # card only while Expand runs it, never beside the transformer.
    model_cpu_offload_seq = "prompt_upsampler->text_encoder->transformer->vae"
    # Absent until a first Expand registers them (`flux3_upsampler.load`).
    _optional_components: list[str] = ["prompt_upsampler", "prompt_upsampler_processor"]
    # Which upstream blocks this variant runs; the Base subclass swaps them.
    _distilled = True

    def __init__(
        self,
        tokenizer: Qwen3VLProcessor,
        text_encoder: Qwen3VLForConditionalGeneration,
        transformer: Flux3Transformer2DModel,
        vae: AutoencoderKLFlux2,
        scheduler: FlowMatchEulerDiscreteScheduler,
        prompt_upsampler: Qwen3_5ForConditionalGeneration | None = None,
        prompt_upsampler_processor: Any = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder,
            transformer=transformer, vae=vae, scheduler=scheduler,
            prompt_upsampler=prompt_upsampler, prompt_upsampler_processor=prompt_upsampler_processor,
        )
        # The upstream decode and reference steps read this processor (its
        # 16-pixel token grid and its reference sizing), not a VaeImageProcessor.
        self.image_processor = Flux3ImageProcessor()
        self.guider: ClassifierFreeGuidance | None = None
        self._references = Flux3VaeEncoderSequentialStep()
        # Annotated as returning the state alone, though it returns (components, state).
        self._decode: Any = Flux3DecodeStep()
        if self._distilled:
            self._encode: Any = Flux3DistilledTextEncoderStep()
            self._text: SequentialPipelineBlocks = Flux3DistilledCoreDenoiseStep()
            self._conditioned: SequentialPipelineBlocks = Flux3DistilledImageConditionedCoreDenoiseStep()
            step_class = _CallbackDistilledDenoiseStep
        else:
            self._encode = Flux3TextEncoderStep()
            self._text = Flux3CoreDenoiseStep()
            self._conditioned = Flux3ImageConditionedCoreDenoiseStep()
            step_class = _CallbackDenoiseStep
        # One callback loop, shared by both cores (text-to-image and with
        # references): they differ only before it.
        self._step: Any = step_class()
        self._text.sub_blocks["denoise"] = self._step
        self._conditioned.sub_blocks["denoise"] = self._step

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any) -> Any:
        """Build from the pinned snapshot's own subfolders (`flux3_loading`).

        The repositories publish no `model_index.json`, so DiffusionPipeline's
        resolver cannot; each component loads from its subfolder instead, with
        the same per-component quantization mapping a standard load takes.
        """
        from . import flux3_loading

        return flux3_loading.load_components(cls, str(pretrained_model_name_or_path), **kwargs)

    @property
    def vae_scale_factor(self) -> int:
        # As Flux3ModularPipeline's: the VAE's own 8x; the 2x2 packing makes 16.
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    @property
    def num_channels_latents(self) -> int:
        return self.transformer.config.in_channels

    @property
    def requires_unconditional_embeds(self) -> bool:
        guider = self.guider
        return guider is not None and guider._enabled and guider.num_conditions > 1

    def upsample_prompt(self, prompt: str, should_cancel: Callable[[], bool] | None = None) -> str:
        """Expand: the registered upsampler's caption for `prompt` (t2i).

        Never part of `__call__`. The caller frees the offload hooks after,
        as it does after ERNIE's rewriter.
        """
        from . import flux3_upsampler

        model = getattr(self, "prompt_upsampler", None)
        processor = getattr(self, "prompt_upsampler_processor", None)
        if model is None or processor is None:
            raise RuntimeError("FLUX 3's prompt upsampler is not loaded")
        return flux3_upsampler.upsample(model, processor, prompt, self._execution_device, should_cancel)

    def set_progress_bar_config(self, **kwargs) -> None:
        super().set_progress_bar_config(**kwargs)
        self._step.set_progress_bar_config(**kwargs)

    def _run(
        self, state: PipelineState, image: Any, callback_on_step_end: Callable | None,
    ) -> ImagePipelineOutput:
        self._step.callback = callback_on_step_end
        try:
            # Each block returns (components, state); their annotations name
            # the modular pipeline, but they read only the standard components
            # and properties registered above.
            _, state = self._encode(self, state)
            blocks: list[Any] = []
            if image:
                # Reference images, as the upstream auto blocks would take them:
                # preprocessed and VAE-encoded, then the conditioned core.
                blocks.extend(self._references.sub_blocks.values())
                blocks.extend(self._conditioned.sub_blocks.values())
            else:
                blocks.extend(self._text.sub_blocks.values())
            # SequentialPipelineBlocks logs every escaping exception as an
            # error. Call its blocks directly so cooperative cancellation is
            # handled by the job layer without a misleading error traceback.
            for block in blocks:
                _, state = block(self, state)
            _, state = self._decode(self, state)
            return ImagePipelineOutput(images=state.values["images"])
        finally:
            self._step.callback = None
            # In particular, cancellation must offload the resident denoiser.
            self.maybe_free_model_hooks()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: list[Any] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 16,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        callback_on_step_end: Callable | None = None,
        sigmas: list[float] | None = None,
        output_type: str = "pil",
    ) -> ImagePipelineOutput:
        # The distilled checkpoint samples without guidance: `guidance_scale`
        # is accepted for the app's uniform call and never reaches the model.
        del guidance_scale
        self.guider = None
        state = PipelineState(values={
            "prompt": prompt, "image": image or None,
            "height": height, "width": width,
            "num_inference_steps": num_inference_steps, "sigmas": sigmas,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator, "output_type": output_type,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        })
        return self._run(state, image, callback_on_step_end)


class Flux3KleinBasePipeline(Flux3KleinPipeline):
    """FLUX 3 Image Klein Base: undistilled, 50 steps, classifier-free guidance."""

    _distilled = False

    @torch.no_grad()
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str,
        image: list[Any] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        negative_prompt: str | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        callback_on_step_end: Callable | None = None,
        sigmas: list[float] | None = None,
        output_type: str = "pil",
    ) -> ImagePipelineOutput:
        # The upstream guider guides at every value except exactly 1. Align
        # the actual pass with this app's >1 threshold, including CFG 0.
        # A fresh guider also clears the previous call's last-step state.
        self.guider = ClassifierFreeGuidance(
            guidance_scale=guidance_scale, enabled=guidance_scale > 1.0,
        )
        values: dict[str, Any] = {
            "prompt": prompt, "image": image or None,
            "height": height, "width": width,
            "num_inference_steps": num_inference_steps, "sigmas": sigmas,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator, "output_type": output_type,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        }
        # Only guided: the encoder step warns about a negative prompt it was
        # handed with the guider off, and encodes "" when guided without one.
        if guidance_scale > 1.0 and negative_prompt is not None:
            values["negative_prompt"] = negative_prompt
        return self._run(PipelineState(values=values), image, callback_on_step_end)
