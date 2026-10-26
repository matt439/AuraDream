"""A quantizing load lets go of the bf16 files it read (`GEN#22`).

safetensors hands each tensor out as a view of its shard's memory map, and
the loaders keep every tensor they leave unquantized as that view, so before
`loading.checkpoint_files_released` a quantized pipeline held its whole bf16
checkpoint mapped beside its quantized copy - about 46 GiB of a FLUX.1 load
on the card's machine. Here a tiny two-component pipeline, saved in several
shards, loads through the real diffusers and transformers paths with torchao
int8 on the CPU; on Linux `/proc/self/maps` says which files stay mapped.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch

from backend.backends import loading

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/maps")


def mapped(root: Path) -> set[str]:
    gc.collect()
    found = set()
    with open("/proc/self/maps", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].strip().startswith(str(root)):
                found.add(Path(parts[5].strip()).name)
    return found


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A pipeline of a FLUX transformer (diffusers) and a T5 encoder
    (transformers), each in several shards, as the Hub lays one out."""
    from diffusers import DiffusionPipeline, FluxTransformer2DModel
    from transformers import T5Config, T5EncoderModel

    class TwoPartPipeline(DiffusionPipeline):
        def __init__(self, text_encoder, transformer):
            super().__init__()
            self.register_modules(text_encoder=text_encoder, transformer=transformer)

    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("two-part")
    transformer = FluxTransformer2DModel(
        patch_size=1, in_channels=16, num_layers=2, num_single_layers=4, attention_head_dim=64,
        num_attention_heads=8, joint_attention_dim=512, pooled_projection_dim=256,
        axes_dims_rope=(16, 24, 24),
    ).to(torch.bfloat16)
    encoder = T5EncoderModel(
        T5Config(vocab_size=2000, d_model=512, d_kv=64, d_ff=2048, num_layers=4, num_heads=8)
    ).to(torch.bfloat16)
    transformer.save_pretrained(root / "transformer", max_shard_size="20MB")
    encoder.save_pretrained(root / "text_encoder", max_shard_size="10MB")
    (root / "model_index.json").write_text(
        '{"_class_name": "TwoPartPipeline", "_diffusers_version": "0.40.0",'
        ' "text_encoder": ["transformers", "T5EncoderModel"],'
        ' "transformer": ["diffusers", "FluxTransformer2DModel"]}',
        encoding="utf-8",
    )
    shards = {p.name for p in root.rglob("*.safetensors")}
    assert len(shards) >= 5, shards
    return TwoPartPipeline, root, shards


def int8_config():
    from diffusers import PipelineQuantizationConfig
    from diffusers import TorchAoConfig as DiffusersTorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    return PipelineQuantizationConfig(quant_mapping={
        "transformer": DiffusersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
        "text_encoder": TransformersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
    })


def load(saved, released):
    cls, root, _ = saved
    with loading.checkpoint_files_released(released):
        return cls.from_pretrained(root, dtype=torch.bfloat16, quantization_config=int8_config())


@linux_only
def test_without_the_release_every_shard_stays_mapped(saved):
    # The fault itself, kept so a diffusers or transformers that stops
    # holding the views is noticed and the copying can go.
    pipe = load(saved, released=False)
    assert mapped(saved[1]) == saved[2]
    del pipe


@linux_only
def test_a_released_load_keeps_no_shard_mapped(saved):
    pipe = load(saved, released=True)
    assert mapped(saved[1]) == set()
    del pipe


@linux_only
def test_a_diffusers_model_holds_one_shard_at_a_time(saved, monkeypatch):
    # The peak, not only the end: each shard goes as the next is read. Seen
    # just after a shard's copies, while its state dict is still alive.
    seen = []
    real = loading.own_tensors

    def watched(module, names=None):
        count = real(module, names)
        seen.append({n for n in mapped(saved[1]) if n.startswith("diffusion_pytorch_model")})
        return count

    monkeypatch.setattr(loading, "own_tensors", watched)
    pipe = load(saved, released=True)
    shards = {n for n in saved[2] if n.startswith("diffusion_pytorch_model")}
    assert len(shards) >= 3
    assert {n for s in seen for n in s} == shards
    assert all(len(s) == 1 for s in seen if s), seen
    del pipe


def test_the_released_load_is_the_same_model(saved):
    torch.manual_seed(1)
    plain, released = load(saved, released=False), load(saved, released=True)
    for name in ("transformer", "text_encoder"):
        a, b = getattr(plain, name).state_dict(), getattr(released, name).state_dict()
        assert a.keys() == b.keys()
        for key in a:
            left, right = a[key], b[key]
            if type(left) is torch.Tensor:
                assert torch.equal(left, right), key
            else:
                # Quantized: still the torchao subclass, never a plain copy.
                assert type(right) is type(left), key
                assert torch.equal(left.dequantize(), right.dequantize()), key
    # T5 ties its encoder's embedding to `shared`; the copy keeps the tie.
    encoder = released.text_encoder
    assert encoder.encoder.embed_tokens.weight is encoder.shared.weight


def test_the_patches_come_off_and_an_inactive_release_patches_nothing():
    import importlib

    def current():
        return [getattr(importlib.import_module(m), n)
                for m, n in (loading._SHARD_LOADER, loading._COMPONENT_LOADER)]

    before = current()
    with loading.checkpoint_files_released(False):
        assert current() == before
    with pytest.raises(RuntimeError), loading.checkpoint_files_released(True):
        assert all(a is not b for a, b in zip(current(), before))
        raise RuntimeError("the load failed")
    assert current() == before


def test_own_tensors_copies_plain_cpu_tensors_by_name_and_keeps_ties():
    shared = torch.nn.Linear(4, 4)
    model = torch.nn.Sequential(shared, torch.nn.LayerNorm(4), shared)
    model.register_buffer("steps", torch.arange(3))
    weight, norm, steps = shared.weight, model[1].weight, model.steps

    assert loading.own_tensors(model, {"2.weight", "steps"}) == 2
    assert model[0].weight is model[2].weight is not weight
    assert torch.equal(model[0].weight, weight) and model[0].weight.requires_grad
    assert model[1].weight is norm  # not named, not copied
    assert model.steps is not steps and torch.equal(model.steps, steps)
    assert loading.own_tensors(model) == 5  # weight, bias, norm's two, steps
