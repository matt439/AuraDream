# Copyright 2025 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NewBie's pooled-CLIP conditioning on the installed Lumina 2 blocks.

Adapted from the Apache-2.0 diffusers implementation and its NewBie PR #12803,
revision e061f982c6c1e162309f9df919662e6fa61e60ea. Only the extra conditioning
and the forward call which passes it differ from the installed Lumina model.
The original checkpoint mapping is strict and runs on CPU before quantization.
"""
from __future__ import annotations

from typing import Any

from .. import config as _config  # noqa: F401 - before any diffusers imports

import torch
from torch import nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import ModelMixin  # noqa: F401 - custom pipeline component loader
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_lumina2 import (
    Lumina2CombinedTimestepCaptionEmbedding,
    Lumina2Transformer2DModel,
)
from diffusers.utils.peft_utils import apply_lora_scale


class NewbieTimestepCaptionEmbedding(Lumina2CombinedTimestepCaptionEmbedding):
    def __init__(self, hidden_size: int, cap_feat_dim: int, pooled_projection_dim: int, norm_eps: float):
        super().__init__(hidden_size=hidden_size, cap_feat_dim=cap_feat_dim, norm_eps=norm_eps)
        self.clip_text_pooled_proj = nn.Sequential(
            RMSNorm(pooled_projection_dim, eps=norm_eps),
            nn.Linear(pooled_projection_dim, pooled_projection_dim),
        )
        time_dim = min(hidden_size, 1024)
        self.time_text_embed = nn.Sequential(nn.SiLU(), nn.Linear(time_dim + pooled_projection_dim, time_dim))

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections):
        time_embed, caption_embed = super().forward(hidden_states, timestep, encoder_hidden_states)
        pooled_embed = self.clip_text_pooled_proj(pooled_projections)
        return self.time_text_embed(torch.cat([time_embed, pooled_embed], dim=-1)), caption_embed


class NewbieTransformer2DModel(Lumina2Transformer2DModel):
    """Serializable Lumina transformer with NewBie's mandatory pooled text path."""

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 2304,
        num_layers: int = 36,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
        scaling_factor: float = 1.0,
        axes_dim_rope: tuple[int, int, int] = (32, 32, 32),
        axes_lens: tuple[int, int, int] = (1024, 512, 512),
        cap_feat_dim: int = 2560,
        pooled_projection_dim: int = 1024,
    ) -> None:
        super().__init__(
            sample_size=sample_size, patch_size=patch_size, in_channels=in_channels,
            out_channels=out_channels, hidden_size=hidden_size, num_layers=num_layers,
            num_refiner_layers=num_refiner_layers, num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps, scaling_factor=scaling_factor, axes_dim_rope=axes_dim_rope,
            axes_lens=axes_lens, cap_feat_dim=cap_feat_dim,
        )
        self.time_caption_embed = NewbieTimestepCaptionEmbedding(
            hidden_size, cap_feat_dim, pooled_projection_dim, norm_eps,
        )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        # Keep conditioning explicit: a mutable hook would leak one prompt's
        # pooled vector into another call and interfere with offload hooks.
        batch_size, _, height, width = hidden_states.shape
        temb, encoder_hidden_states = self.time_caption_embed(
            hidden_states, timestep, encoder_hidden_states, pooled_projections,
        )
        (
            hidden_states, context_rotary_emb, noise_rotary_emb, rotary_emb,
            encoder_seq_lengths, seq_lengths,
        ) = self.rope_embedder(hidden_states, encoder_attention_mask)
        hidden_states = self.x_embedder(hidden_states)

        for layer in self.context_refiner:
            encoder_hidden_states = layer(encoder_hidden_states, encoder_attention_mask, context_rotary_emb)
        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, None, noise_rotary_emb, temb)

        max_seq_len = max(seq_lengths)
        use_mask = len(set(seq_lengths)) > 1
        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config["hidden_size"])
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = encoder_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = hidden_states[i]
        hidden_states = joint_hidden_states

        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                checkpoint = self._gradient_checkpointing_func
                assert checkpoint is not None, "enabled gradient checkpointing installs its callback"
                hidden_states = checkpoint(
                    layer, hidden_states, attention_mask if use_mask else None, rotary_emb, temb,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask if use_mask else None, rotary_emb, temb)
        hidden_states = self.norm_out(hidden_states, temb)

        patch = self.config["patch_size"]
        output = []
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            output.append(
                hidden_states[i][encoder_seq_len:seq_len]
                .view(height // patch, width // patch, patch, patch, self.out_channels)
                .permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)
            )
        output = torch.stack(output, dim=0)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)


def load_original_transformer(checkpoint: dict[str, torch.Tensor]) -> NewbieTransformer2DModel:
    """Strictly map the pinned official 535-tensor NextDiT checkpoint.

    Lumina's converter removes only its known unused ``norm_final.weight``.
    The five extra pooled-text tensors are mapped without discarding anything;
    strict loading catches missing, extra and incorrectly shaped parameters.
    The published config omits Gemma's caption dimension, verified as 2560 in
    the pinned safetensors header. Defaults above describe that exact release.
    """
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_utils import convert_lumina2_to_diffusers

    state = convert_lumina2_to_diffusers(dict(checkpoint))
    for key in list(state):
        if key.startswith(("clip_text_pooled_proj.", "time_text_embed.")):
            state[f"time_caption_embed.{key}"] = state.pop(key)
    expected_count = (
        len(checkpoint) - ("norm_final.weight" in checkpoint)
        + 2 * sum("qkv" in key for key in checkpoint)
    )
    if len(state) != expected_count:
        raise RuntimeError("Unexpected overlapping keys in the original NewBie checkpoint")
    with init_empty_weights():
        model = NewbieTransformer2DModel()
    model.load_state_dict(state, strict=True, assign=True)
    return model
