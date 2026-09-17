"""Pinned conversion, atomic cache publication and the normal quantized load."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import diffusers
import huggingface_hub
import pytest
import torch

from backend.backends import diffusers_backend as module
from backend.backends import loading, single_file
from backend.schemas import GenerateRequest


@pytest.fixture
def conversion(monkeypatch, tmp_path):
    monkeypatch.setattr(single_file, "cache_root", lambda: tmp_path)
    calls = []

    def convert(path):
        calls.append(path)
        path.mkdir()
        (path / "model_index.json").write_text("{}")

    monkeypatch.setattr(single_file, "_convert", convert)
    return calls


def test_conversion_is_published_once_and_reused_across_quantizers(conversion):
    first = single_file.resolve_model(single_file.ILLUSTRIOUS_ID, "fp8")
    assert Path(first, "model_index.json").is_file()
    assert json.loads(Path(first, "local-diffusion-conversion.json").read_text()) == single_file.RECIPE
    for quant in ("int8", "none", "fp8"):
        assert single_file.resolve_model(single_file.ILLUSTRIOUS_ID, quant) == first
    assert len(conversion) == 1


def test_failed_conversion_is_not_reused(monkeypatch, conversion, tmp_path):
    convert = single_file._convert

    def fail(path):
        convert(path)
        raise RuntimeError("disk full")

    monkeypatch.setattr(single_file, "_convert", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        single_file.resolve_model(single_file.ILLUSTRIOUS_ID, "fp8")
    assert not list(tmp_path.glob("*/model_index.json"))
    monkeypatch.setattr(single_file, "_convert", convert)
    result = single_file.resolve_model(single_file.ILLUSTRIOUS_ID, "fp8")
    assert Path(result, "model_index.json").is_file()
    assert len(conversion) == 2


def test_ordinary_models_pass_through_and_nunchaku_refuses_before_conversion(conversion):
    assert single_file.resolve_model("cagliostrolab/animagine-xl-4.0", "fp8") == "cagliostrolab/animagine-xl-4.0"
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        single_file.resolve_model(single_file.ILLUSTRIOUS_ID, "nunchaku")
    assert not conversion


def test_conversion_uses_only_official_weights_and_fixed_epsilon_euler_a(monkeypatch, tmp_path):
    downloads = []
    kwargs_seen = {}

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    pipe = SimpleNamespace(scheduler=diffusers.EulerDiscreteScheduler())

    def save(path, **kwargs):
        assert isinstance(pipe.scheduler, diffusers.EulerAncestralDiscreteScheduler)
        assert pipe.scheduler.config.prediction_type == "epsilon"
        assert kwargs == {"safe_serialization": True}

    pipe.save_pretrained = save

    def load(path, **kwargs):
        kwargs_seen.update(kwargs)
        return pipe

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_single_file", load)
    single_file._convert(tmp_path)
    config, weights = downloads
    assert weights == single_file.ILLUSTRIOUS_SOURCE
    assert config["revision"] == single_file.SDXL_CONFIG["revision"]
    assert config["allow_patterns"] == ["*.json", "tokenizer/*.txt", "tokenizer_2/*.txt"]
    assert kwargs_seen == {
        "config": str(tmp_path), "local_files_only": True,
        "dtype": torch.bfloat16, "add_watermarker": False,
    }


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_backend_quantizes_the_converted_tree_but_keeps_the_official_id(monkeypatch, tmp_path, quant):
    model_id = single_file.ILLUSTRIOUS_ID
    tree = str(tmp_path / "converted")
    configured = dataclasses.replace(
        module.settings, model_id=model_id, quantization=quant, device="cpu", offload="none",
    )
    monkeypatch.setattr(module, "settings", configured)
    monkeypatch.setattr(single_file, "resolve_model", lambda model, mode: tree)
    quant_config = object() if quant != "none" else None

    def quantize(settings):
        assert settings.model_id == tree and settings.quantization == quant
        return quant_config

    def unused(model):
        assert model == tree
        return "StableDiffusionXLPipeline", {}

    class Pipe:
        scheduler = diffusers.EulerAncestralDiscreteScheduler()

        def to(self, device):
            return self

        def __call__(self, generator=None, callback_on_step_end=None, negative_prompt=""):
            pass

    def load(model, **kwargs):
        assert model == tree
        assert kwargs.get("quantization_config") is quant_config
        return Pipe()

    monkeypatch.setattr(loading, "quantization_config", quantize)
    monkeypatch.setattr(loading, "unused_components", unused)
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", load)
    backend = module.DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert module.settings.model_id == model_id
    assert backend.describe()["model_id"] == model_id
    assert backend.applied(GenerateRequest(
        prompt="a landscape", mu=1.15,
    )).mu is None
