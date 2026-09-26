"""CPU checks for NewBie's pooled conditioning and strict original mapping."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from backend.backends.newbie_transformer import NewbieTransformer2DModel, load_original_transformer


@pytest.fixture
def tiny_model():
    torch.manual_seed(2026)
    return NewbieTransformer2DModel(
        sample_size=4, in_channels=4, hidden_size=24, num_layers=1,
        num_refiner_layers=1, num_attention_heads=4, num_kv_heads=2,
        multiple_of=8, axes_dim_rope=(2, 2, 2), axes_lens=(8, 8, 8),
        cap_feat_dim=8, pooled_projection_dim=6,
    ).eval()


def _inputs():
    return dict(
        hidden_states=torch.randn(2, 4, 4, 4), timestep=torch.tensor([0.7, 0.2]),
        encoder_hidden_states=torch.randn(2, 4, 8),
        encoder_attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        pooled_projections=torch.randn(2, 6),
    )


def test_pooled_conditioning_matches_upstream_formula(tiny_model):
    embed = tiny_model.time_caption_embed
    inputs = _inputs()
    pooled = inputs["pooled_projections"]
    states = inputs["encoder_hidden_states"]
    # Explicit formula from the pinned upstream PR, independent of the local
    # wrapper's forward: RMS -> projection -> concat(time, pooled) -> SiLU -> linear.
    pooled_norm = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-5)
    pooled_norm *= embed.clip_text_pooled_proj[0].weight
    pooled_linear = embed.clip_text_pooled_proj[1]
    pooled_embed = F.linear(pooled_norm, pooled_linear.weight, pooled_linear.bias)
    time_embed = embed.timestep_embedder(embed.time_proj(inputs["timestep"]))
    time_linear = embed.time_text_embed[1]
    expected_time = F.linear(F.silu(torch.cat([time_embed, pooled_embed], -1)),
                             time_linear.weight, time_linear.bias)
    state_norm = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + 1e-5)
    state_norm *= embed.caption_embedder[0].weight
    caption_linear = embed.caption_embedder[1]
    expected_caption = F.linear(state_norm, caption_linear.weight, caption_linear.bias)
    actual_time, actual_caption = embed(
        inputs["hidden_states"], inputs["timestep"], states, pooled,
    )
    torch.testing.assert_close(actual_time, expected_time)
    torch.testing.assert_close(actual_caption, expected_caption)


def test_full_forward_uses_each_pooled_vector_and_survives_serialization(tiny_model, tmp_path):
    inputs = _inputs()
    with torch.no_grad():
        expected = tiny_model(**inputs).sample
        repeated = tiny_model(**inputs, return_dict=False)[0]
        changed = tiny_model(**{**inputs, "pooled_projections": inputs["pooled_projections"] + 2}).sample
    assert expected.shape == (2, 4, 4, 4)
    torch.testing.assert_close(expected, repeated)
    assert not torch.allclose(expected, changed)

    tiny_model.save_pretrained(tmp_path, safe_serialization=True)
    restored = NewbieTransformer2DModel.from_pretrained(tmp_path, local_files_only=True)
    assert restored.config.pooled_projection_dim == 6
    assert restored.config.cap_feat_dim == 8
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs).sample, expected)


@pytest.fixture
def official_checkpoint():
    # Shapes verified against the pinned official safetensors header by HTTP
    # range, without downloading its 6.97 GB tensor payload. Expanded scalars
    # give full parameter shapes while allocating only a few KB of CPU memory.
    shapes = {
        "cap_embedder.0.weight": (2560,),
        "cap_embedder.1.bias": (2304,),
        "cap_embedder.1.weight": (2304, 2560),
        "clip_text_pooled_proj.0.weight": (1024,),
        "clip_text_pooled_proj.1.bias": (1024,),
        "clip_text_pooled_proj.1.weight": (1024, 1024),
        "final_layer.adaLN_modulation.1.bias": (2304,),
        "final_layer.adaLN_modulation.1.weight": (2304, 1024),
        "final_layer.linear.bias": (64,),
        "final_layer.linear.weight": (64, 2304),
        "norm_final.weight": (2304,),
        "t_embedder.mlp.0.bias": (1024,),
        "t_embedder.mlp.0.weight": (1024, 256),
        "t_embedder.mlp.2.bias": (1024,),
        "t_embedder.mlp.2.weight": (1024, 1024),
        "time_text_embed.1.bias": (1024,),
        "time_text_embed.1.weight": (1024, 2048),
        "x_embedder.bias": (2304,),
        "x_embedder.weight": (2304, 64),
    }
    block = {
        "attention.k_norm.weight": (96,),
        "attention.out.weight": (2304, 2304),
        "attention.q_norm.weight": (96,),
        "attention.qkv.weight": (3840, 2304),
        "attention_norm1.weight": (2304,),
        "attention_norm2.weight": (2304,),
        "feed_forward.w1.weight": (9216, 2304),
        "feed_forward.w2.weight": (2304, 9216),
        "feed_forward.w3.weight": (9216, 2304),
        "ffn_norm1.weight": (2304,),
        "ffn_norm2.weight": (2304,),
    }
    for group, count in (("layers", 36), ("context_refiner", 2), ("noise_refiner", 2)):
        for index in range(count):
            prefix = f"{group}.{index}."
            shapes.update({prefix + key: shape for key, shape in block.items()})
            if group != "context_refiner":
                shapes[prefix + "adaLN_modulation.1.bias"] = (9216,)
                shapes[prefix + "adaLN_modulation.1.weight"] = (9216, 1024)
    state = {key: torch.tensor(7.0).expand(shape) for key, shape in shapes.items()}
    state["layers.0.attention.qkv.weight"] = torch.arange(3840.0)[:, None].expand(3840, 2304)
    return state


def test_original_checkpoint_maps_all_tensors_and_qkv_in_order(official_checkpoint):
    model = load_original_transformer(official_checkpoint)
    assert len(official_checkpoint) == 535  # the caller's state is not consumed
    assert len(model.state_dict()) == 614  # one unused norm removed, 40 QKV split
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    attention = model.layers[0].attn
    assert torch.equal(attention.to_q.weight[:, 0], torch.arange(2304.0))
    assert torch.equal(attention.to_k.weight[:, 0], torch.arange(2304.0, 3072.0))
    assert torch.equal(attention.to_v.weight[:, 0], torch.arange(3072.0, 3840.0))
    assert model.time_caption_embed.clip_text_pooled_proj[1].bias[0].item() == 7.0
    assert model.time_caption_embed.time_text_embed[1].weight[0, -1].item() == 7.0


@pytest.mark.parametrize("damage, error", [
    ("missing", "Missing key"), ("shape", "size mismatch"), ("extra", "Unexpected key"),
    ("overlap", "Unexpected overlapping keys"),
])
def test_bad_original_checkpoint_fails_strictly(official_checkpoint, damage, error):
    if damage == "missing":
        del official_checkpoint["clip_text_pooled_proj.0.weight"]
    elif damage == "shape":
        official_checkpoint["time_text_embed.1.weight"] = torch.zeros(1)
    elif damage == "overlap":
        official_checkpoint["time_caption_embed.clip_text_pooled_proj.0.weight"] = torch.zeros(1024)
    else:
        official_checkpoint["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match=error):
        load_original_transformer(official_checkpoint)
