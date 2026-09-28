"""Official Turbo weights, strict conversion and atomic reusable CPU cache."""
from __future__ import annotations

import fnmatch
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import diffusers
import huggingface_hub
import pytest
import safetensors.torch
import torch
import transformers

from backend.backends import anima_loading as module


@pytest.fixture
def conversion(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "cache_root", lambda: tmp_path)
    calls = []

    def convert(path):
        calls.append(path)
        path.mkdir()
        (path / "model_index.json").write_text("{}")

    monkeypatch.setattr(module, "_convert", convert)
    return calls


def test_conversion_reused_across_quantizers_and_read_only_lookup(conversion):
    assert module.converted_tree(module.MODEL_ID) is None
    assert not conversion
    first = module.resolve_model(module.MODEL_ID, "fp8")
    assert module.converted_tree(module.MODEL_ID) == first
    assert json.loads(Path(first, "local-diffusion-conversion.json").read_text()) == module.RECIPE
    for quant in ("int8", "none", "fp8"):
        assert module.resolve_model(module.MODEL_ID, quant) == first
    assert len(conversion) == 1


def test_other_models_pass_through_and_nunchaku_refuses_before_download(conversion):
    assert module.resolve_model("SeeSee21/Z-Anime", "fp8") == "SeeSee21/Z-Anime"
    assert module.converted_tree("SeeSee21/Z-Anime") is None
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        module.resolve_model(module.MODEL_ID, "nunchaku")
    assert not conversion


def test_failed_conversion_cannot_publish_partial_tree(monkeypatch, conversion, tmp_path):
    convert = module._convert

    def fail(path):
        convert(path)
        raise RuntimeError("disk full")

    monkeypatch.setattr(module, "_convert", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        module.resolve_model(module.MODEL_ID, "none")
    assert module.converted_tree(module.MODEL_ID) is None
    assert not list(tmp_path.glob("*/model_index.json"))
    monkeypatch.setattr(module, "_convert", convert)
    assert Path(module.resolve_model(module.MODEL_ID, "none"), "model_index.json").is_file()
    assert len(conversion) == 2


def test_a_failed_conversion_reports_its_own_error_past_an_open_file(monkeypatch, conversion, tmp_path):
    # On Windows a file still open stops the work directory's cleanup: the
    # caller still hears why the conversion failed, and the next load sweeps.
    convert = module._convert
    held = []

    def fail(path):
        convert(path)
        held.append((path / "model_index.json").open("rb"))
        raise RuntimeError("disk full")

    monkeypatch.setattr(module, "_convert", fail)
    try:
        with pytest.raises(RuntimeError, match="disk full"):
            module.resolve_model(module.MODEL_ID, "none")
    finally:
        for handle in held:
            handle.close()
    monkeypatch.setattr(module, "_convert", convert)
    result = module.resolve_model(module.MODEL_ID, "none")
    assert [path.name for path in tmp_path.iterdir() if path.is_dir()] == [Path(result).name]
    assert len(conversion) == 2


def test_an_interrupted_conversions_work_directory_is_swept(conversion, tmp_path):
    target, _, _ = module._target()
    stale = tmp_path / f"{target.name}-k3j9_0q1"
    (stale / "tree" / "transformer").mkdir(parents=True)
    # Named like one but not one: a longer or shorter suffix and a copy of a
    # published tree.
    longer = tmp_path / f"{target.name}-k3j9_0q1x"
    (longer / "tree").mkdir(parents=True)
    shorter = tmp_path / f"{target.name}-old"
    (shorter / "tree").mkdir(parents=True)
    copy = tmp_path / f"{target.name}-backup01"
    copy.mkdir()
    (copy / "model_index.json").write_text("{}")
    first = module.resolve_model(module.MODEL_ID, "fp8")
    assert not stale.exists()
    assert (longer / "tree").is_dir() and (shorter / "tree").is_dir() and (copy / "model_index.json").is_file()
    # Once converted, the next load still sweeps and converts nothing.
    (stale / "tree").mkdir(parents=True)
    assert module.resolve_model(module.MODEL_ID, "fp8") == first
    assert not stale.exists()
    assert len(conversion) == 1


def test_the_sweep_waits_for_a_conversion_holding_the_lock(conversion, tmp_path):
    # Another process's conversion, mid-save: it holds the lock, and its work
    # directory is live until it lets go.
    import threading

    from huggingface_hub.utils import WeakFileLock

    target, _, _ = module._target()
    live = tmp_path / f"{target.name}-live0001"
    answer: list[str] = []
    with WeakFileLock(tmp_path / f"{target.name}.lock"):
        (live / "tree").mkdir(parents=True)
        worker = threading.Thread(target=lambda: answer.append(module.resolve_model(module.MODEL_ID, "fp8")))
        worker.start()
        worker.join(0.5)
        waited, kept = worker.is_alive(), live.is_dir()
    # Joined before any assert, so a failure never leaves it running into the next test.
    worker.join(10)
    assert waited and kept
    assert answer == [str(target)]
    assert not live.exists()


def test_existing_incomplete_cache_is_not_overwritten(conversion):
    target, marker, recipe = module._target()
    target.mkdir()
    (target / "model_index.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Incomplete converted model cache"):
        module.resolve_model(module.MODEL_ID, "fp8")
    assert not marker.exists()
    assert not conversion


def test_official_v11_prefix_split_and_cosmos_rename_preserve_tensors():
    denoiser = torch.arange(6).reshape(2, 3)
    conditioner = torch.arange(8).reshape(2, 4)
    transformer, adapter = module._split_checkpoint({
        "model.diffusion_model.blocks.0.self_attn.q_proj.weight": denoiser,
        "model.diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.weight": conditioner,
    })
    assert list(transformer) == ["transformer_blocks.0.attn1.to_q.weight"]
    assert transformer["transformer_blocks.0.attn1.to_q.weight"] is denoiser
    assert list(adapter) == ["blocks.0.self_attn.q_proj.weight"]
    assert adapter["blocks.0.self_attn.q_proj.weight"] is conditioner


@pytest.mark.parametrize("checkpoint", [
    {"model.diffusion_model.blocks.0.weight": torch.zeros(1)},
    {"model.diffusion_model.llm_adapter.embed.weight": torch.zeros(1)},
    {"unrecognized.weights": torch.zeros(1)},
])
def test_missing_components_and_unexpected_checkpoint_format_fail(checkpoint):
    with pytest.raises(ValueError, match="Anima Turbo checkpoint"):
        module._split_checkpoint(checkpoint)


def test_converted_conditioner_is_strict_and_has_no_meta_parameters(tmp_path):
    original = diffusers.AnimaTextConditioner(
        source_dim=8, target_dim=8, model_dim=8, num_layers=1,
        num_attention_heads=2, target_vocab_size=16, min_sequence_length=4,
    )
    original.save_config(tmp_path)
    weights = original.state_dict()
    result = module._load_component(diffusers.AnimaTextConditioner, tmp_path, weights)
    assert all(parameter.device.type == "cpu" for parameter in result.parameters())
    for key, value in result.state_dict().items():
        assert torch.equal(value, weights[key])
    missing = dict(weights)
    del missing["embed.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        module._load_component(diffusers.AnimaTextConditioner, tmp_path, missing)
    malformed = {**weights, "embed.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError, match="size mismatch"):
        module._load_component(diffusers.AnimaTextConditioner, tmp_path, malformed)


def test_conversion_downloads_only_pinned_turbo_and_shared_components(monkeypatch, tmp_path):
    from backend.backends import anima_pipeline

    downloads, loads, components = [], [], {}

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path, device: {
        "model.diffusion_model.blocks.0.self_attn.q_proj.weight": torch.ones(1),
        "model.diffusion_model.llm_adapter.embed.weight": torch.full((1,), 2.0),
    })

    class Component:
        def to(self, **kwargs):
            assert kwargs == {"dtype": torch.bfloat16}
            return self

    def converted(cls, config, state):
        components[Path(config).name] = (cls, state)
        return Component()

    monkeypatch.setattr(module, "_load_component", converted)

    def load(path, **kwargs):
        loads.append((Path(path).name, kwargs))
        return Component()

    for cls in (transformers.AutoTokenizer, transformers.T5TokenizerFast,
                transformers.Qwen3Model, diffusers.AutoencoderKLQwenImage):
        monkeypatch.setattr(cls, "from_pretrained", load)

    def pipeline(**kwargs):
        assert kwargs["text_conditioner"] is not None
        scheduler = kwargs["scheduler"]
        assert isinstance(scheduler, diffusers.FlowMatchEulerDiscreteScheduler)
        assert scheduler.config.shift == 3.0
        assert not scheduler.config.use_dynamic_shifting

        def save(path, **save_kwargs):
            assert path == tmp_path
            assert save_kwargs == {"safe_serialization": True}

        return SimpleNamespace(save_pretrained=save)

    monkeypatch.setattr(anima_pipeline, "AnimaTurboPipeline", pipeline)
    module._convert(tmp_path)
    shared, source = downloads
    assert source == module.SOURCE
    assert shared == {**module.SHARED, "allow_patterns": module.SHARED_PATTERNS}
    for filename in ("transformer/diffusion_pytorch_model.safetensors",
                     "text_conditioner/diffusion_pytorch_model.safetensors"):
        assert not any(fnmatch.fnmatch(filename, pattern) for pattern in shared["allow_patterns"])
    assert set(components) == {"transformer", "text_conditioner"}
    assert torch.equal(components["text_conditioner"][1]["embed.weight"], torch.full((1,), 2.0))
    assert len(loads) == 4
    assert all(kwargs["local_files_only"] for _, kwargs in loads)
    assert {name for name, _ in loads} == {"tokenizer", "t5_tokenizer", "text_encoder", "vae"}


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_backend_quantizes_converted_tree_with_concrete_pipeline_and_keeps_public_id(monkeypatch, tmp_path, quant):
    from backend.backends import anima_pipeline, diffusers_backend, loading
    from backend.schemas import GenerateRequest

    tree = str(tmp_path / "converted")
    configured = replace(
        diffusers_backend.settings, model_id=module.MODEL_ID,
        quantization=quant, device="cpu", offload="none",
    )
    monkeypatch.setattr(diffusers_backend, "settings", configured)
    monkeypatch.setattr(module, "resolve_model", lambda model_id, quantization: tree)
    quant_config = object() if quant != "none" else None

    def quantize(settings):
        assert settings.model_id == tree and settings.quantization == quant
        return quant_config

    def unused(model_id):
        assert model_id == tree
        return "AnimaTurboPipeline", {}

    class Pipe:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)

        def to(self, device):
            assert device == torch.device("cpu")
            return self

        def __call__(self, generator=None, callback_on_step_end=None, negative_prompt=""):
            pass

    concrete_loads = []

    def load(model_id, **kwargs):
        assert model_id == tree
        assert kwargs.get("quantization_config") is quant_config
        concrete_loads.append(model_id)
        return Pipe()

    def wrong_loader(*args, **kwargs):
        pytest.fail("Anima must load through its concrete app-owned pipeline")

    monkeypatch.setattr(loading, "quantization_config", quantize)
    monkeypatch.setattr(loading, "unused_components", unused)
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", wrong_loader)
    monkeypatch.setattr(anima_pipeline.AnimaTurboPipeline, "from_pretrained", load)
    backend = diffusers_backend.DiffusersBackend()
    request = GenerateRequest(prompt="a landscape", mu=1.15)
    assert not backend.lora_supported()
    assert backend.applied(request).mu is None
    backend.load()
    assert backend.is_ready()
    assert concrete_loads == [tree]
    assert diffusers_backend.settings.model_id == module.MODEL_ID
    assert backend.describe()["model_id"] == module.MODEL_ID
    assert not backend.lora_supported()
    assert backend.describe()["mu_supported"] is False
    assert backend.applied(request).mu is None
