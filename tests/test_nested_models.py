"""Nested Hub trees must share one source across preflight and weight loading."""
from __future__ import annotations

import dataclasses
import fnmatch
import json
import sys
from pathlib import Path

import diffusers
import huggingface_hub
import pytest
import torch
import transformers

from backend.backends import diffusers_backend as module
from backend.backends import loading
from backend.model_defaults import defaults_for, guide_for
from backend.schemas import GenerateRequest

MODEL = "SeeSee21/Z-Anime"


@pytest.fixture
def nested(monkeypatch, tmp_path):
    files = {
        "diffusers/model_index.json": {
            "_class_name": "ZImagePipeline",
            "transformer": ["diffusers", "ZImageTransformer2DModel"],
            "text_encoder": ["transformers", "Qwen3ForCausalLM"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        },
        "diffusers/transformer/config.json": {},
        "diffusers/text_encoder/config.json": {},
        "diffusers/vae/config.json": {},
        "diffusers/scheduler/scheduler_config.json": {
            "shift": 6.0, "use_dynamic_shifting": False,
        },
        "diffusers/transformer/diffusion_pytorch_model.safetensors": "weights",
        "diffusers/text_encoder/model.safetensors": "incorrectly packaged scaled FP8",
        "diffusers/tokenizer/chat_template.jinja": "template",
        "aio/z-anime-base-aio-bf16.safetensors": "unwanted",
        "diffusion_models/z-anime-distill-4step-bf16.safetensors": "unwanted",
    }
    calls = []

    def snapshot(model_id, *, revision, allow_patterns, ignore_patterns=()):
        assert model_id == MODEL
        assert revision == loading.NESTED_MODELS[MODEL][0]
        selected = [name for name in files if
                    any(fnmatch.fnmatchcase(name, p) for p in allow_patterns)
                    and not any(fnmatch.fnmatchcase(name, p) for p in ignore_patterns)]
        assert not any(name.startswith(("aio/", "diffusion_models/")) for name in selected)
        calls.append(selected)
        for name in selected:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            value = files[name]
            path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    # The autouse Hub refusal stays in force; this case reads only local files.
    def local_file(model_id, filename):
        assert Path(model_id) == tmp_path / "diffusers"
        return str(Path(model_id) / filename)

    monkeypatch.setattr(loading, "hub_file", local_file)
    return tmp_path / "diffusers", calls, files


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_nested_load_preflights_then_downloads_only_its_tree(monkeypatch, nested, quant):
    tree, calls, _ = nested
    monkeypatch.setattr(module, "settings", dataclasses.replace(
        module.settings, model_id=MODEL, quantization=quant, device="cpu", offload="none",
    ))
    original_quantization = loading.quantization_config
    owned_configs = []
    encoder = object()

    def quantization(settings):
        assert Path(settings.model_id) == tree
        assert len(calls) == 1
        assert not list(tree.rglob("*.safetensors"))
        config = original_quantization(settings)
        if quant != "none":
            assert set(config.quant_mapping) == {"transformer", "text_encoder"}
            owned_configs.append(config.quant_mapping["text_encoder"])
        return config

    def load_encoder(**kwargs):
        assert kwargs["pretrained_model_name_or_path"] == "Tongyi-MAI/Z-Image"
        assert kwargs["revision"] == loading.Z_ANIME_TEXT_ENCODER["revision"]
        assert kwargs["subfolder"] == "text_encoder"
        assert kwargs["dtype"] == torch.bfloat16
        assert kwargs.get("quantization_config") is (owned_configs[0] if owned_configs else None)
        assert not (tree / "text_encoder/model.safetensors").exists()
        return encoder

    class ZImagePipeline:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=6.0)

        def to(self, device):
            return self

        def __call__(self, generator=None, callback_on_step_end=None, negative_prompt="", sigmas=None):
            pass

    def load(model_id, **kwargs):
        assert Path(model_id) == tree
        assert len(calls) == 2
        assert (tree / "tokenizer/chat_template.jinja").is_file()
        assert list(tree.rglob("*.safetensors"))
        assert "subfolder" not in kwargs
        assert ("quantization_config" in kwargs) == (quant != "none")
        assert kwargs["text_encoder"] is encoder
        return ZImagePipeline()

    monkeypatch.setattr(loading, "quantization_config", quantization)
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", load)
    monkeypatch.setattr(transformers.Qwen3ForCausalLM, "from_pretrained", load_encoder)
    backend = module.DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert backend.describe()["model_id"] == MODEL
    request = backend.applied(GenerateRequest(prompt="a landscape", negative_prompt="blur", guidance_scale=4))
    assert request.negative_prompt == "blur" and request.mu is None
    assert backend.cfg_threshold() == 0.0
    assert backend.describe()["mu_supported"]


@pytest.mark.parametrize("quant", ["fp8", "nunchaku"])
def test_bad_quantizer_pairings_fail_before_nested_weights(monkeypatch, nested, quant):
    tree, calls, files = nested
    monkeypatch.setattr(module, "settings", dataclasses.replace(
        module.settings, model_id=MODEL, quantization=quant, device="cpu", offload="none",
    ))
    if quant == "fp8":
        files["diffusers/transformer/config.json"] = {"quantization_config": {"quant_method": "other"}}
        message = "already quantized"
    else:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(loading, "importable", lambda name: True)
        message = "not a Nunchaku Lite checkpoint"
    with pytest.raises(RuntimeError, match=message):
        module.DiffusersBackend().load()
    assert len(calls) == 1
    assert not list(tree.rglob("*.safetensors"))


def test_ordinary_hub_and_local_sources_do_not_snapshot(tmp_path):
    for source in ("Tongyi-MAI/Z-Image-Turbo", str(tmp_path)):
        assert loading.resolve_model(source) == source
        assert loading.resolve_model(source, weights=True) == source
        assert loading.component_overrides(source, None) == {}


def test_z_anime_uses_base_defaults_and_natural_language_guide():
    profile = defaults_for(MODEL)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (40, 4.0, None)
    guide = guide_for(MODEL)
    assert guide["tag_helper"] is None
    assert guide["pixels_word"] == "documented"
    assert guide["pixels"][1] >= 2048**2 / 1e6
