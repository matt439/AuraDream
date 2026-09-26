"""NewBie's dual text conditioning behind the app's standard pipeline API.

The release's NextDiT is represented by a small Lumina 2 extension. Encoding
uses its Gemma 3 penultimate hidden state and Jina CLIP masked mean, without a
chat template or an invented system instruction. The flow loop follows the
upstream Diffusers Newbie proposal; source pins and differences are recorded
in docs/design-notes.md.
"""
from __future__ import annotations

import logging
from typing import Callable

from .. import config as _config  # noqa: F401 - before Diffusers

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import Gemma3TextModel, PreTrainedTokenizerBase

from .newbie_text_encoder import NewbieJinaTextEncoder
from .newbie_transformer import NewbieTransformer2DModel

log = logging.getLogger(__name__)


class NewbiePipeline(DiffusionPipeline):
    """Pinned Exp0.1 inference with seeded batches and cooperative cancellation."""

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components: list[str] = []

    def __init__(
        self,
        transformer: NewbieTransformer2DModel,
        text_encoder: Gemma3TextModel,
        text_encoder_2: NewbieJinaTextEncoder,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_2: PreTrainedTokenizerBase,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer, text_encoder=text_encoder,
            text_encoder_2=text_encoder_2, tokenizer=tokenizer,
            tokenizer_2=tokenizer_2, vae=vae, scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer.padding_side = "right"

    def _encode(self, prompts: list[str], device, num_images: int):
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        mask = inputs.attention_mask.to(device)
        encoded = self.text_encoder(
            input_ids=inputs.input_ids.to(device), attention_mask=mask,
            output_hidden_states=True, use_cache=False,
        ).hidden_states[-2]
        inputs_2 = self.tokenizer_2(
            prompts, padding=True, truncation=True, max_length=2048, return_tensors="pt",
        )
        for name, tokens, limit in (("Gemma", inputs, 512), ("Jina", inputs_2, 2048)):
            if tokens.input_ids.shape[-1] >= limit:
                log.warning("NewBie %s prompt reached its %s-token limit; text may be truncated", name, limit)
        # Calling forward (rather than get_text_features on the original remote
        # Jina model) lets Accelerate move/offload both encoders normally.
        pooled = self.text_encoder_2(
            input_ids=inputs_2.input_ids.to(device),
            attention_mask=inputs_2.attention_mask.to(device),
        ).pooler_output
        dtype = self.transformer.dtype
        return (
            encoded.to(device=device, dtype=dtype).repeat_interleave(num_images, dim=0),
            mask.repeat_interleave(num_images, dim=0),
            pooled.to(device=device, dtype=dtype).repeat_interleave(num_images, dim=0),
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        sigmas: list[float] | None = None,
        callback_on_step_end: Callable | None = None,
        output_type: str = "pil",
    ) -> ImagePipelineOutput:
        if height % 16 or width % 16 or min(height, width) < 16:
            raise ValueError("NewBie dimensions must be positive multiples of 16")
        if num_inference_steps < 1 or num_images_per_prompt < 1:
            raise ValueError("NewBie requires at least one step and one image")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            raise ValueError("NewBie requires a prompt")
        guided = guidance_scale > 1.0
        batch = len(prompts) * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch:
            raise ValueError(f"Expected {batch} generators, got {len(generator)}")
        negative = "" if negative_prompt is None else negative_prompt
        negatives = [negative] * len(prompts) if isinstance(negative, str) else list(negative)
        if guided and len(negatives) != len(prompts):
            raise ValueError("negative_prompt must have the same batch size as prompt")
        device = self._execution_device
        try:
            # Encode together so each encoder is brought onto the card once.
            embeddings, mask, pooled = self._encode(
                prompts + negatives if guided else prompts, device, num_images_per_prompt,
            )
            latents = randn_tensor(
                (batch, self.transformer.config.in_channels,
                 height // self.vae_scale_factor, width // self.vae_scale_factor),
                generator=generator, device=device, dtype=self.transformer.dtype,
            )
            if sigmas is None:
                sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            timesteps = self.scheduler.timesteps
            with self.progress_bar(total=len(timesteps)) as progress:
                for index, timestep in enumerate(timesteps):
                    # NextDiT uses 0 for noise and 1 for image; FlowMatch Euler
                    # uses the reverse and expects the negative model velocity.
                    model_t = (1 - timestep / self.scheduler.config.num_train_timesteps).expand(batch)
                    conditional = self.transformer(
                        hidden_states=latents, timestep=model_t,
                        encoder_hidden_states=embeddings[:batch],
                        encoder_attention_mask=mask[:batch], pooled_projections=pooled[:batch],
                        return_dict=False,
                    )[0]
                    prediction = conditional
                    if guided:
                        unconditional = self.transformer(
                            hidden_states=latents, timestep=model_t,
                            encoder_hidden_states=embeddings[batch:],
                            encoder_attention_mask=mask[batch:], pooled_projections=pooled[batch:],
                            return_dict=False,
                        )[0]
                        prediction = unconditional + guidance_scale * (conditional - unconditional)
                        # The author's renorm_cfg=1 caps the whole sample's
                        # norm. It never boosts a smaller guided prediction.
                        axes = tuple(range(1, prediction.ndim))
                        norm = torch.linalg.vector_norm(prediction, dim=axes, keepdim=True).clamp_min(1e-12)
                        cond_norm = torch.linalg.vector_norm(conditional, dim=axes, keepdim=True)
                        scale = (cond_norm / norm).clamp(max=1)
                        prediction = prediction * scale
                    latents = self.scheduler.step(
                        -prediction, timestep, latents, return_dict=False,
                    )[0]
                    if callback_on_step_end is not None:
                        updates = callback_on_step_end(self, index, timestep, {"latents": latents})
                        if updates is not None:
                            latents = updates.get("latents", latents)
                    progress.update()
            if output_type == "latent":
                images = latents
            else:
                decoded = self.vae.decode(
                    latents / self.vae.config.scaling_factor + self.vae.config.shift_factor,
                    return_dict=False,
                )[0]
                images = self.image_processor.postprocess(decoded, output_type=output_type)
            return ImagePipelineOutput(images=images)
        finally:
            # Cancellation must release the resident denoiser too.
            self.maybe_free_model_hooks()
