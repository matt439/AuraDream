"""Pinned NewBie conversion, quantizer ownership and backend integration on CPU."""
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

from backend.backends import diffusers_backend, loading, newbie_loading as module
from backend.model_defaults import defaults_for
from backend.schemas import GenerateRequest


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


def test_atomic_cache_is_reused_across_quantizers_and_lookup_never_converts(conversion):
    assert module.converted_tree(module.MODEL_ID) is None
    assert not conversion
    first = module.resolve_model(module.MODEL_ID, "fp8")
    assert module.converted_tree(module.MODEL_ID.upper()) == first
    assert json.loads(Path(first, "local-diffusion-conversion.json").read_text()) == module.RECIPE
    for quant in ("fp8", "int8", "none"):
        assert module.resolve_model(module.MODEL_ID, quant) == first
    assert len(conversion) == 1


@pytest.fixture
def tiny_gemma():
    config = transformers.Gemma3TextConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, max_position_embeddings=32, sliding_window=16,
        query_pre_attn_scalar=4, layer_types=["sliding_attention", "full_attention"],
    )
    original = transformers.Gemma3TextModel(config).eval()
    state = {f"model.{key}": tensor for key, tensor in original.state_dict().items()}
    state["spiece_model"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    return original, state, config.to_dict()


def test_gemma_conversion_preserves_all_text_layers_and_forward(tiny_gemma):
    original, state, config = tiny_gemma
    converted = module._load_gemma(state, config)
    assert "spiece_model" in state  # conversion leaves the caller's mapping intact
    assert not converted.training
    assert all(parameter.device.type == "cpu" for parameter in converted.parameters())
    assert set(converted.state_dict()) == set(original.state_dict())
    for key, tensor in converted.state_dict().items():
        torch.testing.assert_close(tensor, original.state_dict()[key])
    ids = torch.tensor([[2, 4, 6, 8]])
    with torch.no_grad():
        torch.testing.assert_close(
            converted(ids, output_hidden_states=True).hidden_states[-2],
            original(ids, output_hidden_states=True).hidden_states[-2],
        )


@pytest.mark.parametrize("damage, exception, message", [
    ("missing", RuntimeError, "Missing key"),
    ("shape", RuntimeError, "size mismatch"),
    ("vision", ValueError, "Unexpected NewBie Gemma tensor"),
    ("extra_text", RuntimeError, "Unexpected key"),
])
def test_gemma_conversion_rejects_incomplete_or_extra_weights(tiny_gemma, damage, exception, message):
    _, state, config = tiny_gemma
    if damage == "missing":
        del state["model.embed_tokens.weight"]
    elif damage == "shape":
        state["model.embed_tokens.weight"] = torch.zeros(1)
    elif damage == "vision":
        state["vision_tower.weight"] = torch.zeros(1)
    else:
        state["model.unexpected.weight"] = torch.zeros(1)
    with pytest.raises(exception, match=message):
        module._load_gemma(state, config)


def test_conversion_uses_only_pinned_components_and_local_loaders(monkeypatch, tmp_path):
    from backend.backends import newbie_pipeline, newbie_text_encoder, newbie_transformer

    downloads, reads, loads, conversions = [], [], [], []
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder/config.json").write_text(json.dumps({"text_config": {"hidden_size": 8}}))

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    def read(path, device):
        assert device == "cpu"
        reads.append(Path(path).relative_to(tmp_path).as_posix())
        return {"source": Path(path).parent.name}

    class Component:
        def to(self, **kwargs):
            assert kwargs == {"dtype": torch.bfloat16}
            return self

    def convert(state, config=None):
        conversions.append((state, config))
        return Component()

    def local_load(path, **kwargs):
        loads.append((Path(path).name, kwargs))
        assert kwargs.get("local_files_only") is True
        assert "trust_remote_code" not in kwargs
        return Component()

    def pipeline(**components):
        assert set(components) == {
            "transformer", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae", "scheduler",
        }
        scheduler = components["scheduler"]
        assert isinstance(scheduler, diffusers.FlowMatchEulerDiscreteScheduler)
        assert scheduler.config.shift == 6.0 and not scheduler.config.use_dynamic_shifting

        def save(path, **kwargs):
            assert path == tmp_path
            assert kwargs == {"safe_serialization": True}

        return SimpleNamespace(save_pretrained=save)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(safetensors.torch, "load_file", read)
    monkeypatch.setattr(newbie_transformer, "load_original_transformer", convert)
    monkeypatch.setattr(newbie_text_encoder, "load_jina_text_encoder", convert)
    monkeypatch.setattr(module, "_load_gemma", convert)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", local_load)
    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", local_load)
    monkeypatch.setattr(newbie_pipeline, "NewbiePipeline", pipeline)
    module._convert(tmp_path)

    assert downloads == [{**module.SOURCE, "allow_patterns": module.PATTERNS}]
    assert module.SOURCE["revision"] == "4f7825a1a7e1ed5c1882a52e3c51d1119a43e298"
    assert reads == [
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/gemma3-4b-it.safetensors", "clip_model/jina-clip-v2.safetensors",
    ]
    assert conversions == [
        ({"source": "transformer"}, None),
        ({"source": "text_encoder"}, {"hidden_size": 8}),
        ({"source": "clip_model"}, None),
    ]
    assert {name for name, _ in loads} == {"text_encoder", "clip_model", "vae"}
    for name in reads + ["vae/diffusion_pytorch_model.safetensors", "text_encoder/tokenizer.model",
                         "text_encoder/tokenizer.json", "clip_model/tokenizer.json"]:
        assert any(fnmatch.fnmatchcase(name, pattern) for pattern in module.PATTERNS)
    for name in ("clip_model/model.safetensors", "clip_model/hf_model.py", "clip_model/modeling.py",
                 "text_encoder/model-00001-of-00002.safetensors", "text_encoder/vision_tower/model.safetensors",
                 "transformer/consolidated.00-of-01.pth", "model.safetensors"):
        assert not any(fnmatch.fnmatchcase(name, pattern) for pattern in module.PATTERNS)


@pytest.fixture
def converted_manifest(monkeypatch, tmp_path):
    index = {
        "_class_name": "NewbiePipeline",
        "transformer": ["backend.backends.newbie_transformer", "NewbieTransformer2DModel"],
        "text_encoder": ["transformers", "Gemma3TextModel"],
        "text_encoder_2": ["backend.backends.newbie_text_encoder", "NewbieJinaTextEncoder"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    for component in ("transformer", "text_encoder", "text_encoder_2", "vae"):
        (tmp_path / component).mkdir()
        (tmp_path / component / "config.json").write_text("{}")

    def local_file(model_id, filename):
        assert Path(model_id) == tmp_path
        return str(tmp_path / filename)

    monkeypatch.setattr(loading, "hub_file", local_file)
    monkeypatch.setattr(module, "resolve_model", lambda model_id, quantization: str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_backend_uses_custom_pipeline_and_owned_quantizers_with_public_metadata(monkeypatch, converted_manifest, quant):
    from backend.backends.newbie_pipeline import NewbiePipeline
    from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig

    tree = converted_manifest
    monkeypatch.setattr(diffusers_backend, "settings", replace(
        diffusers_backend.settings, model_id=module.MODEL_ID,
        quantization=quant, device="cpu", offload="none",
    ))
    loaded = []

    class Pipe:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=6.0)

        def to(self, device):
            assert device == torch.device("cpu")
            return self

        def __call__(self, generator=None, callback_on_step_end=None, negative_prompt="", sigmas=None):
            pass

    def load(model_id, **kwargs):
        assert Path(model_id) == tree
        assert kwargs["dtype"] == torch.bfloat16
        if quant == "none":
            assert "quantization_config" not in kwargs
        else:
            mapping = kwargs["quantization_config"].quant_mapping
            assert set(mapping) == {"transformer", "text_encoder", "text_encoder_2"}
            assert isinstance(mapping["transformer"], diffusers.TorchAoConfig)
            assert isinstance(mapping["text_encoder"], transformers.TorchAoConfig)
            assert isinstance(mapping["text_encoder_2"], transformers.TorchAoConfig)
            expected_type = Float8WeightOnlyConfig if quant == "fp8" else Int8WeightOnlyConfig
            assert all(isinstance(config.quant_type, expected_type) for config in mapping.values())
        loaded.append(model_id)
        return Pipe()

    def wrong_loader(*args, **kwargs):
        pytest.fail("NewBie must use its concrete local pipeline")

    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", wrong_loader)
    monkeypatch.setattr(NewbiePipeline, "from_pretrained", load)
    backend = diffusers_backend.DiffusersBackend()
    profile = defaults_for(module.MODEL_ID)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (28, 4.0, None)
    request = GenerateRequest(prompt="a landscape", negative_prompt="blur", guidance_scale=4.0,
                              mu=1.15, true_cfg=6.0, reference=[1], lora="example.safetensors", lora_scale=0.5)
    before = backend.applied(request)
    assert before.mu == 1.15 and before.negative_prompt == "blur" and before.true_cfg == 4.0
    assert before.reference == [] and before.lora is None and before.lora_scale == 1.0
    backend.load()
    backend.load()  # idempotent once ready
    assert loaded == [str(tree)]
    assert backend.is_ready()
    assert backend.applied(request) == before
    assert not backend.lora_supported()
    status = backend.describe()
    assert status["model_id"] == module.MODEL_ID and status["quantization"] == quant
    assert status["mu_supported"] and status["negative_prompt_supported"]
    assert not status["reference_supported"] and not status["true_cfg_supported"]
    assert diffusers_backend.settings.model_id == module.MODEL_ID


@pytest.mark.parametrize("missing", ["generator", "callback_on_step_end"])
def test_backend_refuses_pipeline_without_seed_or_cancel_contract(monkeypatch, converted_manifest, missing):
    from backend.backends.newbie_pipeline import NewbiePipeline

    monkeypatch.setattr(diffusers_backend, "settings", replace(
        diffusers_backend.settings, model_id=module.MODEL_ID,
        quantization="none", device="cpu", offload="none",
    ))

    class Pipe:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=6.0)

        def to(self, device):
            return self

    if missing == "generator":
        Pipe.__call__ = lambda self, callback_on_step_end=None: None
    else:
        Pipe.__call__ = lambda self, generator=None: None
    monkeypatch.setattr(NewbiePipeline, "from_pretrained", lambda *args, **kwargs: Pipe())
    backend = diffusers_backend.DiffusersBackend()
    with pytest.raises(RuntimeError, match=f"takes no {missing}"):
        backend.load()
    assert not backend.is_ready()
