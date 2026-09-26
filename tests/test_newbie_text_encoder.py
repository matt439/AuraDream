"""CPU checks of the real native Jina layer mapping and conditioning math."""
from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F
from transformers import JinaEmbeddingsV3Config

from backend.backends.newbie_text_encoder import (
    NewbieJinaTextEncoder,
    convert_jina_state_dict,
    jina_config,
    load_jina_text_encoder,
)


def _config():
    return JinaEmbeddingsV3Config(
        vocab_size=19, hidden_size=8, num_attention_heads=2,
        num_hidden_layers=2, intermediate_size=12, max_position_embeddings=32,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        rope_parameters={"rope_type": "default", "rope_theta": 20000.0},
        _attn_implementation="eager",
    )


def _checkpoint(config):
    """Tiny original-layout weights, created independently of the mapper."""
    generator = torch.Generator().manual_seed(601)
    d, f = config.hidden_size, config.intermediate_size
    state = {
        "model.embeddings.word_embeddings.weight": torch.randn(config.vocab_size, d, generator=generator),
        "model.embeddings.token_type_embeddings.weight": torch.randn(1, d, generator=generator),
        "model.emb_ln.weight": torch.randn(d, generator=generator),
        "model.emb_ln.bias": torch.randn(d, generator=generator),
        "spiece_model": torch.tensor([1, 2, 3], dtype=torch.uint8),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.encoder.layers.{layer}."
        for name, rows, columns in [
            ("mixer.Wqkv", 3 * d, d), ("mixer.out_proj", d, d),
            ("mlp.fc1", f, d), ("mlp.fc2", d, f),
        ]:
            state[prefix + name + ".weight"] = torch.randn(rows, columns, generator=generator) / 4
            state[prefix + name + ".bias"] = torch.randn(rows, generator=generator) / 4
        for name in ("norm1", "norm2"):
            state[prefix + name + ".weight"] = torch.randn(d, generator=generator)
            state[prefix + name + ".bias"] = torch.randn(d, generator=generator)
    return state


def _reference(state, ids, config):
    """Original combined-QKV, post-norm Jina equations, without HF modules.

Cross-checks the layout against the source architecture instead of merely
round-tripping the implementation's renamed tensors.
"""
    d, heads = config.hidden_size, config.num_attention_heads
    head_dim = d // heads
    mask = ids.ne(config.pad_token_id)
    x = F.embedding(ids, state["model.embeddings.word_embeddings.weight"])
    x = x + state["model.embeddings.token_type_embeddings.weight"][0]
    x = F.layer_norm(x, (d,), state["model.emb_ln.weight"], state["model.emb_ln.bias"], config.layer_norm_eps)
    theta = config.rope_parameters["rope_theta"]
    freqs = torch.outer(torch.arange(ids.shape[1]), 1 / theta ** (torch.arange(0, head_dim, 2) / head_dim))
    phase = torch.cat((freqs, freqs), dim=-1)[None, :, None, :]

    def rotate(value):
        a, b = value.chunk(2, dim=-1)
        return value * phase.cos() + torch.cat((-b, a), dim=-1) * phase.sin()

    for layer in range(config.num_hidden_layers):
        prefix = f"model.encoder.layers.{layer}."

        def linear(value, name):
            return F.linear(value, state[prefix + name + ".weight"], state[prefix + name + ".bias"])

        def norm(value, name):
            return F.layer_norm(value, (d,), state[prefix + name + ".weight"], state[prefix + name + ".bias"], config.layer_norm_eps)

        q, k, v = linear(x, "mixer.Wqkv").reshape(*ids.shape, 3, heads, head_dim).unbind(2)
        q, k, v = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2), v.transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(*ids.shape, d)
        x = norm(x + linear(attended, "mixer.out_proj"), "norm1")
        x = norm(x + linear(F.gelu(linear(x, "mlp.fc1")), "mlp.fc2"), "norm2")
    pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    return x, pooled


def test_jina_converted_forward_matches_original_layout_and_padding():
    config = _config()
    state = _checkpoint(config)
    encoder = load_jina_text_encoder(state, config)
    ids = torch.tensor([[0, 7, 3, 2, 1], [0, 9, 2, 1, 1]])
    expected, pooled = _reference(state, ids, config)
    actual = encoder(ids)
    torch.testing.assert_close(actual.last_hidden_state, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.pooler_output, pooled, atol=2e-6, rtol=2e-6)
    # Padding must not change the actual pooled prompt or normalize its length.
    torch.testing.assert_close(encoder(ids[:1, :4]).pooler_output, actual.pooler_output[:1])
    assert not torch.allclose(actual.pooler_output.norm(dim=-1), torch.ones(2))


def test_jina_release_config_and_strict_rejection():
    released = jina_config()
    assert (released.hidden_size, released.num_hidden_layers, released.num_attention_heads) == (1024, 24, 16)
    assert released.rope_parameters["rope_theta"] == 20000
    state = _checkpoint(_config())
    with pytest.raises(ValueError, match="Unexpected"):
        convert_jina_state_dict({**state, "vision_model.weight": torch.ones(1)})
    missing = dict(state)
    missing.pop("model.emb_ln.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_jina_text_encoder(missing, _config())
    malformed = dict(state)
    malformed["model.encoder.layers.0.mixer.Wqkv.weight"] = torch.ones(23, 8)
    with pytest.raises(ValueError, match="Malformed"):
        convert_jina_state_dict(malformed)


def test_jina_normal_forward_hooks_and_local_roundtrip(tmp_path):
    encoder = load_jina_text_encoder(_checkpoint(_config()), _config())
    ids = torch.tensor([[0, 7, 2]])
    calls = []
    hook = encoder.register_forward_pre_hook(lambda *args: calls.append(True))
    expected = encoder(ids).pooler_output
    hook.remove()
    assert calls == [True]
    torch.testing.assert_close(encoder(ids, return_dict=False)[1], expected)
    encoder.save_pretrained(tmp_path, safe_serialization=True)
    restored = NewbieJinaTextEncoder.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(restored(ids).pooler_output, expected)
    assert restored.pooler is None
    with pytest.raises(ValueError, match="all-padding"):
        restored(torch.ones((1, 3), dtype=torch.long))
