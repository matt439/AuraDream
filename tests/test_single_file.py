"""Pinned conversion, atomic cache publication and the normal quantized load."""
from __future__ import annotations

import dataclasses
import hashlib
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


@pytest.fixture(params=[single_file.ILLUSTRIOUS_ID, single_file.WAI_ID])
def model_id(request):
    return request.param


@pytest.fixture
def conversion(monkeypatch, tmp_path):
    monkeypatch.setattr(single_file, "cache_root", lambda: tmp_path)
    calls = []

    def convert(path, model_id=single_file.ILLUSTRIOUS_ID):
        calls.append((path, model_id))
        path.mkdir()
        (path / "model_index.json").write_text("{}")

    monkeypatch.setattr(single_file, "_convert", convert)
    return calls


def test_conversion_is_published_once_and_reused_across_quantizers(conversion, model_id):
    first = single_file.resolve_model(model_id, "fp8")
    assert Path(first, "model_index.json").is_file()
    expected = single_file.RECIPE if model_id == single_file.ILLUSTRIOUS_ID else single_file.WAI_RECIPE
    assert json.loads(Path(first, "local-diffusion-conversion.json").read_text()) == expected
    for quant in ("int8", "none", "fp8"):
        assert single_file.resolve_model(model_id, quant) == first
    assert single_file.converted_tree(model_id) == first
    assert len(conversion) == 1


def test_failed_conversion_is_not_reused(monkeypatch, conversion, tmp_path, model_id):
    convert = single_file._convert

    def fail(path, model_id):
        convert(path, model_id)
        raise RuntimeError("disk full")

    monkeypatch.setattr(single_file, "_convert", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        single_file.resolve_model(model_id, "fp8")
    assert not list(tmp_path.glob("*/model_index.json"))
    monkeypatch.setattr(single_file, "_convert", convert)
    result = single_file.resolve_model(model_id, "fp8")
    assert Path(result, "model_index.json").is_file()
    assert len(conversion) == 2


def test_ordinary_models_pass_through_and_nunchaku_refuses_before_conversion(conversion, model_id):
    assert single_file.resolve_model("cagliostrolab/animagine-xl-4.0", "fp8") == "cagliostrolab/animagine-xl-4.0"
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        single_file.resolve_model(model_id, "nunchaku")
    assert not conversion


def test_single_file_caches_cannot_be_confused(conversion):
    assert single_file.converted_tree(single_file.WAI_ID) is None
    illustrious = single_file.resolve_model(single_file.ILLUSTRIOUS_ID, "fp8")
    assert single_file.converted_tree(single_file.WAI_ID) is None
    wai = single_file.resolve_model(single_file.WAI_ID, "fp8")
    assert illustrious != wai
    assert single_file.converted_tree(single_file.WAI_ID.upper()) == wai
    marker = Path(wai, "local-diffusion-conversion.json")
    marker.write_text(Path(illustrious, marker.name).read_text(), encoding="utf-8")
    assert single_file.converted_tree(single_file.WAI_ID) is None


def test_conversion_uses_only_pinned_weights_and_fixed_epsilon_euler_a(monkeypatch, tmp_path, model_id):
    downloads = []
    kwargs_seen = {}

    source = tmp_path / "checkpoint.safetensors"
    source.write_bytes(b"checkpoint fixture")
    monkeypatch.setitem(single_file.WAI_RECIPE, "source_sha256", hashlib.sha256(source.read_bytes()).hexdigest())

    def download(**kwargs):
        downloads.append(kwargs)
        return str(source) if "filename" in kwargs else str(tmp_path)

    pipe = SimpleNamespace(scheduler=diffusers.EulerDiscreteScheduler())

    def save(path, **kwargs):
        assert isinstance(pipe.scheduler, diffusers.EulerAncestralDiscreteScheduler)
        assert pipe.scheduler.config.prediction_type == "epsilon"
        assert kwargs == {"safe_serialization": True}

    pipe.save_pretrained = save

    def load(path, **kwargs):
        assert path == str(source)
        kwargs_seen.update(kwargs)
        return pipe

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_single_file", load)
    single_file._convert(tmp_path, model_id)
    config, weights = downloads
    assert weights == (single_file.ILLUSTRIOUS_SOURCE if model_id == single_file.ILLUSTRIOUS_ID else single_file.WAI_SOURCE)
    assert config["revision"] == single_file.SDXL_CONFIG["revision"]
    assert config["allow_patterns"] == ["*.json", "tokenizer/*.txt", "tokenizer_2/*.txt"]
    assert kwargs_seen == {
        "config": str(tmp_path), "local_files_only": True,
        "dtype": torch.bfloat16, "add_watermarker": False,
    }


def test_wai_hash_mismatch_refuses_before_loading_or_publishing(monkeypatch, tmp_path):
    source = tmp_path / "bad.safetensors"
    source.write_bytes(b"wrong checkpoint")
    monkeypatch.setattr(single_file, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(source))

    def load(*args, **kwargs):
        raise AssertionError("must verify checkpoint before the loader runs")

    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_single_file", load)
    with pytest.raises(RuntimeError, match="SHA-256"):
        single_file.resolve_model(single_file.WAI_ID, "fp8")
    assert single_file.converted_tree(single_file.WAI_ID) is None
    assert not list((tmp_path / "cache").glob("*/model_index.json"))


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_backend_quantizes_the_converted_tree_but_keeps_the_public_id(monkeypatch, tmp_path, quant, model_id):
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
