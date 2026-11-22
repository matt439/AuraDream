"""Blend SDXL cross-attention by the user-drawn region for each spatial query."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..spatial_conditioning import PromptRegion, RegionalPrompts

if TYPE_CHECKING:
    from torch import Tensor

    from .base import CancelFn


def _coverage(regions: tuple[PromptRegion, ...], side: int, device: Any) -> tuple[Tensor, ...]:
    import torch

    # Test each query's center against percentages without rounding region
    # edges to another latent grid at each UNet resolution.
    centers = (torch.arange(side, device=device) * 2 + 1) * 50
    return tuple(
        (
            ((centers >= region.top * side) & (centers < region.bottom * side))[:, None]
            & ((centers >= region.left * side) & (centers < region.right * side))[None, :]
        ).reshape(1, side * side, 1)
        for region in regions
    )


class _Mix:
    def __init__(self, regions: tuple[PromptRegion, ...], embeddings: tuple[Tensor, ...]):
        self._regions = regions
        self._embeddings = embeddings
        self._grids: dict[tuple[int, Any], tuple[Tensor, ...]] = {}

    def coverage(self, side: int, device: Any) -> tuple[Tensor, ...]:
        key = (side, device)
        if key not in self._grids:
            self._grids[key] = _coverage(self._regions, side, device)
        return self._grids[key]

    def __call__(
        self,
        processor: Any,
        attn: Any,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None,
        temb: Tensor | None,
    ) -> Tensor:
        import torch

        side = math.isqrt(hidden_states.shape[1])
        if (
            hidden_states.ndim != 3 or side * side != hidden_states.shape[1]
            or len(hidden_states) != 2 or len(encoder_hidden_states) != 2
            or attention_mask is not None or temb is not None
        ):
            raise ValueError("Regional prompts need square SDXL queries and the standard CFG attention layout.")
        global_tokens = encoder_hidden_states[1:]
        base = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)
        positive = base[1:]
        mixed = torch.zeros_like(positive, dtype=torch.float32)
        weights = torch.zeros_like(positive[:, :, :1], dtype=torch.float32)
        for embedding, coverage in zip(self._embeddings, self.coverage(side, positive.device), strict=True):
            # Every region can attend to the full global text as well as its
            # local text. The pooled SDXL text condition stays global.
            context = torch.cat((global_tokens, embedding), dim=1)
            conditioned = processor(attn, hidden_states[1:], encoder_hidden_states=context)
            mixed.add_(conditioned.float() * coverage)
            weights.add_(coverage)
        result = torch.where(weights > 0, (mixed / weights.clamp_min(1)).to(positive.dtype), positive)
        return torch.cat((base[:1], result), dim=0)


class _RegionalProcessor:
    def __init__(self, processor: Any, mix: _Mix):
        self._processor = processor
        self._mix = mix

    def __call__(
        self,
        attn: Any,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        temb: Tensor | None = None,
    ) -> Tensor:
        if encoder_hidden_states is None:
            raise ValueError("A regional cross-attention layer received no text tokens.")
        return self._mix(self._processor, attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def _embeddings(pipe: Any, spec: RegionalPrompts, should_cancel: CancelFn | None) -> tuple[Tensor, ...]:
    from .base import Cancelled

    embeddings = []
    for region in spec.regions:
        if should_cancel is not None and should_cancel():
            raise Cancelled
        for tokenizer in (pipe.tokenizer, pipe.tokenizer_2):
            if len(tokenizer(region.prompt, truncation=False).input_ids) > tokenizer.model_max_length:
                raise ValueError("A regional prompt exceeds an SDXL tokenizer's limit. Shorten that region's text.")
        local, _, _, _ = pipe.encode_prompt(
            prompt=region.prompt, device=pipe._execution_device, do_classifier_free_guidance=False,
        )
        embeddings.append(local)
    return tuple(embeddings)


@contextmanager
def prepared(pipe: Any, spec: RegionalPrompts, should_cancel: CancelFn | None) -> Iterator[None]:
    """Restore all processors on success, cancel, or a failed forward."""
    from diffusers.models.attention_processor import AttnProcessor2_0

    prior = pipe.unet.attn_processors
    cross = {key for key in prior if key.endswith(".attn2.processor")}
    if not cross or any(type(prior[key]) is not AttnProcessor2_0 for key in cross):
        raise ValueError("Regional prompts require native SDXL cross-attention processors.")
    mix = _Mix(spec.regions, _embeddings(pipe, spec, should_cancel))
    replacement = {
        key: _RegionalProcessor(processor, mix) if key in cross else processor
        for key, processor in prior.items()
    }
    try:
        pipe.unet.set_attn_processor(replacement)
        yield
    finally:
        pipe.unet.set_attn_processor(prior)
