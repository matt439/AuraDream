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

    def convert(path, model_id=module.MODEL_ID):
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
    with pytest.raises(ValueError, match="Anima checkpoint"):
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


@pytest.mark.parametrize("model_id,source,pipeline_name", [
    (module.MODEL_ID, module.SOURCE, "AnimaTurboPipeline"),
    (module.AESTHETIC_MODEL_ID, module.AESTHETIC_SOURCE, "AnimaAestheticPipeline"),
    (module.PREVIEW_MODEL_ID, module.PREVIEW_SOURCE, "AnimaPreviewPipeline"),
])
def test_conversion_downloads_only_pinned_variant_and_shared_components(
    monkeypatch, tmp_path, model_id, source, pipeline_name,
):
    from backend.backends import anima_pipeline

    downloads, loads, components = [], [], {}

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    prefix = "net." if model_id == module.PREVIEW_MODEL_ID else "model.diffusion_model."
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path, device: {
        prefix + "blocks.0.self_attn.q_proj.weight": torch.ones(1),
        prefix + "llm_adapter.embed.weight": torch.full((1,), 2.0),
    })

    class Component:
        def to(self, **kwargs):
            assert kwargs == {"dtype": torch.bfloat16}
            return self

    def converted(cls, config, state, **overrides):
        expected = {"num_layers": 40} if model_id == module.PREVIEW_MODEL_ID and cls is diffusers.CosmosTransformer3DModel else {}
        assert overrides == expected
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

    monkeypatch.setattr(anima_pipeline, pipeline_name, pipeline)
    module._convert(tmp_path, model_id)
    shared, downloaded_source = downloads
    assert downloaded_source == source
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
@pytest.mark.parametrize("model_id,pipeline_name", [
    (module.MODEL_ID, "AnimaTurboPipeline"),
    (module.AESTHETIC_MODEL_ID, "AnimaAestheticPipeline"),
    (module.PREVIEW_MODEL_ID, "AnimaPreviewPipeline"),
])
def test_backend_quantizes_converted_tree_with_concrete_pipeline_and_keeps_public_id(
    monkeypatch, tmp_path, quant, model_id, pipeline_name,
):
    from backend.backends import anima_pipeline, diffusers_backend, loading
    from backend.schemas import GenerateRequest

    tree = str(tmp_path / "converted")
    configured = replace(
        diffusers_backend.settings, model_id=model_id,
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
        return pipeline_name, {}

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
    monkeypatch.setattr(getattr(anima_pipeline, pipeline_name), "from_pretrained", load)
    backend = diffusers_backend.DiffusersBackend()
    request = GenerateRequest(prompt="a landscape", mu=1.15)
    assert not backend.lora_supported()
    assert backend.applied(request).mu is None
    backend.load()
    assert backend.is_ready()
    assert concrete_loads == [tree]
    assert diffusers_backend.settings.model_id == model_id
    assert backend.describe()["model_id"] == model_id
    assert not backend.lora_supported()
    assert backend.describe()["mu_supported"] is False
    assert backend.applied(request).mu is None


def test_variants_have_separate_pinned_caches_without_changing_turbo(conversion):
    turbo = module.resolve_model(module.MODEL_ID, "fp8")
    assert module.converted_tree(module.AESTHETIC_MODEL_ID) is None
    aesthetic = module.resolve_model(module.AESTHETIC_MODEL_ID.upper(), "fp8")
    assert turbo != aesthetic
    assert Path(turbo).name.startswith("anima-turbo-v1.1-")
    assert Path(aesthetic).name.startswith("anima-aesthetic-v1.1-")
    assert json.loads(Path(turbo, "local-diffusion-conversion.json").read_text()) == module.RECIPE
    assert json.loads(Path(aesthetic, "local-diffusion-conversion.json").read_text()) == module.AESTHETIC_RECIPE
    assert module.AESTHETIC_SOURCE["repo_id"] == module.MODEL_ID
    assert module.AESTHETIC_SOURCE["revision"] == "594c27fea35648b87c86a9b4d5436a6024c820b5"
    for quant in ("fp8", "int8", "none"):
        assert module.resolve_model(module.AESTHETIC_MODEL_ID, quant) == aesthetic
    assert len(conversion) == 2


def test_aesthetic_nunchaku_refuses_and_unknown_variants_pass_through(conversion):
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        module.resolve_model(module.AESTHETIC_MODEL_ID, "nunchaku")
    unknown = module.MODEL_ID + "@aesthetic-v2"
    assert module.resolve_model(unknown, "fp8") == unknown
    assert module.converted_tree(unknown) is None
    assert not conversion


def test_preview_prefix_preserves_all_expanded_layers_and_its_conditioner():
    weights = {f"net.blocks.{layer}.self_attn.q_proj.weight": torch.full((1,), layer)
               for layer in range(40)}
    conditioner = torch.ones(1)
    weights["net.llm_adapter.embed.weight"] = conditioner
    transformer, adapter = module._split_checkpoint(weights, prefix="net.")
    assert len(transformer) == 40
    for layer in range(40):
        assert transformer[f"transformer_blocks.{layer}.attn1.to_q.weight"].item() == layer
    assert adapter == {"embed.weight": conditioner}
    # A different release's format is not guessed from its keys.
    with pytest.raises(ValueError, match="Unexpected Anima checkpoint key"):
        module._split_checkpoint(weights)


def test_preview_depth_override_strictly_loads_the_full_expanded_transformer(tmp_path):
    original = diffusers.CosmosTransformer3DModel(
        num_attention_heads=2, attention_head_dim=16, in_channels=2, out_channels=2,
        num_layers=3, text_embed_dim=8, adaln_lora_dim=4,
    )
    original.save_config(tmp_path)
    expanded = diffusers.CosmosTransformer3DModel.from_config(original.config, num_layers=40)
    weights = expanded.state_dict()
    loaded = module._load_component(
        diffusers.CosmosTransformer3DModel, tmp_path, weights, num_layers=40,
    )
    assert loaded.config.num_layers == 40
    assert len(loaded.transformer_blocks) == 40
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, weights[key])
    # Removing a newly inserted layer fails; it cannot silently load Base.
    missing = {key: value for key, value in weights.items()
               if not key.startswith("transformer_blocks.39.")}
    with pytest.raises(RuntimeError, match="Missing key"):
        module._load_component(
            diffusers.CosmosTransformer3DModel, tmp_path, missing, num_layers=40,
        )


def test_preview_cache_is_distinct_and_reused_without_changing_existing_recipes(conversion):
    trees = {model_id: module.resolve_model(model_id, "fp8")
             for model_id in (module.MODEL_ID, module.AESTHETIC_MODEL_ID, module.PREVIEW_MODEL_ID)}
    assert len(set(trees.values())) == 3
    preview = trees[module.PREVIEW_MODEL_ID]
    assert Path(preview).name.startswith("Anima-2.9B-preview-v1-")
    assert json.loads(Path(preview, "local-diffusion-conversion.json").read_text()) == module.PREVIEW_RECIPE
    assert module.PREVIEW_SOURCE == {
        "repo_id": "Gazingstars123/Anima-2.9B",
        "filename": "Anima-2.9B-preview-v1.safetensors",
        "revision": "9f9cb502dbae7a616c3cc5a530633427fe735665",
    }
    for quant in ("fp8", "int8", "none"):
        assert module.resolve_model(module.PREVIEW_MODEL_ID.upper(), quant) == preview
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        module.resolve_model(module.PREVIEW_MODEL_ID, "nunchaku")
    assert len(conversion) == 3
