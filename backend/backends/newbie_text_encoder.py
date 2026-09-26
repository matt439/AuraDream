"""NewBie's Jina CLIP v2 text tower, using native Transformers modules.

The author's text-only ``clip_model/jina-clip-v2.safetensors`` at revision
4f7825a1a7e1ed5c1882a52e3c51d1119a43e298 already contains merged weights.
It is distinct from ``clip_model/model.safetensors``, which also contains a
vision tower and parametrized adapters and must not be passed to this mapper.

Architecture and pooling references:
* jinaai/jina-embeddings-v3 config at 343dbf534c76fe845f304fa5c2d1fd87e1e78918
* the author's pinned clip_model/hf_model.py (masked mean, no normalization)
* ComfyUI jina_clip_2.py at 79be670e2d9be63e238785af307369d2b9039ed1

No remote Python, FlashAttention, or vision encoder is needed. Calling the
normal forward method also lets Accelerate's offload hooks move the tower.
"""
from __future__ import annotations

import re
from typing import Any

import torch
from transformers import JinaEmbeddingsV3Config, JinaEmbeddingsV3Model, PreTrainedModel

# Diffusers discovers the inherited load/save methods through this exported
# base class when the pipeline records this local component module.
__all__ = ["NewbieJinaTextEncoder", "PreTrainedModel"]


def jina_config() -> JinaEmbeddingsV3Config:
    """The released text tower's exact architecture, without a CLS pooler."""
    return JinaEmbeddingsV3Config(
        vocab_size=250002,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=8194,
        type_vocab_size=1,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 20000.0},
        use_cache=False,
    )


class NewbieJinaTextEncoder(JinaEmbeddingsV3Model):
    """Native Jina layers with the checkpoint's unnormalized masked mean."""

    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__(config, add_pooling_layer=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Author HFTextEncoder derives the mask from pad ids, including when
        # the caller only passes input_ids to get_text_features().
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("NewBie Jina requires input_ids or an explicit attention_mask")
            attention_mask = input_ids.ne(self.config.pad_token_id)
        return_dict = kwargs.pop("return_dict", True)
        output = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs,
        )
        mask = attention_mask.to(dtype=output.last_hidden_state.dtype).unsqueeze(-1)
        # Tokenizers always supply BOS/EOS, but reject an all-padding input
        # rather than creating non-finite conditioning.
        count = mask.sum(dim=1)
        if torch.any(count == 0):
            raise ValueError("NewBie Jina cannot pool an all-padding prompt")
        output["pooler_output"] = (output.last_hidden_state * mask).sum(dim=1) / count
        return output if return_dict else output.to_tuple()


def convert_jina_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Map the author's merged text-only weights; refuse every unknown key.

    Shape and missing-key validation are additionally enforced by the strict
    load below. The sole non-weight entry, spiece_model, is an embedded tokenizer
    used by ComfyUI; this app loads the matching author's tokenizer files.
    """
    mapped = {}
    fixed = {
        "model.emb_ln": "embeddings.LayerNorm",
        "model.embeddings.word_embeddings": "embeddings.word_embeddings",
        "model.embeddings.token_type_embeddings": "embeddings.token_type_embeddings",
    }
    layer_names = {
        "mixer.out_proj": "self_attn.o_proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
        "norm1": "post_attention_layernorm",
        "norm2": "post_mlp_layernorm",
    }
    for key, value in checkpoint.items():
        if key == "spiece_model":
            if value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("NewBie Jina spiece_model must be a byte vector")
            continue
        prefix, _, suffix = key.rpartition(".")
        if prefix in fixed and suffix in {"weight", "bias"}:
            mapped[f"{fixed[prefix]}.{suffix}"] = value
            continue
        match = re.fullmatch(r"model\.encoder\.layers\.(\d+)\.(.+)\.(weight|bias)", key)
        if match:
            layer, name, suffix = match.groups()
            if name == "mixer.Wqkv":
                if value.ndim != (2 if suffix == "weight" else 1) or value.shape[0] % 3:
                    raise ValueError(f"Malformed NewBie Jina QKV tensor: {key}")
                for projection, tensor in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                    mapped[f"layers.{layer}.self_attn.{projection}.{suffix}"] = tensor.contiguous()
                continue
            if name in layer_names:
                mapped[f"layers.{layer}.{layer_names[name]}.{suffix}"] = value
                continue
        raise ValueError(f"Unexpected NewBie Jina text tensor: {key}")
    return mapped


def load_jina_text_encoder(
    checkpoint: dict[str, Any], config: JinaEmbeddingsV3Config | None = None,
) -> NewbieJinaTextEncoder:
    """Materialize the strictly converted weights on CPU before quantization."""
    from accelerate import init_empty_weights

    state = convert_jina_state_dict(checkpoint)
    with init_empty_weights():
        encoder = NewbieJinaTextEncoder(config or jina_config())
    encoder.load_state_dict(state, strict=True, assign=True)
    return encoder.eval()
