"""Pinned NetaYume conversion and real Lumina scheduling without model weights."""
from __future__ import annotations

import fnmatch
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import diffusers
import huggingface_hub
import pytest
import safetensors.torch
import torch
import transformers

from backend.backends import diffusers_backend, loading, netayume_loading as module, schedule
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


def test_atomic_conversion_is_reused_by_every_supported_quantizer(conversion):
    assert module.converted_tree(module.MODEL_ID) is None
    assert not conversion
    first = module.resolve_model(module.MODEL_ID, "fp8")
    assert module.converted_tree(module.MODEL_ID.upper()) == first
    assert json.loads(Path(first, "local-diffusion-conversion.json").read_text()) == module.RECIPE
    for quant in ("none", "int8", "fp8"):
        assert module.resolve_model(module.MODEL_ID, quant) == first
    assert len(conversion) == 1


def test_other_models_pass_through_and_nunchaku_refuses_before_conversion(conversion):
    other = "Alpha-VLLM/Lumina-Image-2.0"
    assert module.resolve_model(other, "fp8") == other
    assert module.converted_tree(other) is None
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        module.resolve_model(module.MODEL_ID, "nunchaku")
    assert not conversion


def test_failed_conversion_leaves_no_partial_tree_and_can_retry(monkeypatch, conversion, tmp_path):
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


@pytest.mark.parametrize("marker_text", [None, "stale recipe"])
def test_existing_incomplete_or_stale_cache_is_never_overwritten(conversion, marker_text):
    target, marker, _ = module._target()
    target.mkdir()
    (target / "model_index.json").write_text("{}")
    if marker_text is not None:
        marker.write_text(marker_text)
    assert module.converted_tree(module.MODEL_ID) is None
    with pytest.raises(RuntimeError, match="Incomplete converted model cache"):
        module.resolve_model(module.MODEL_ID, "fp8")
    assert not conversion


@pytest.fixture
def original_checkpoint(tmp_path):
    # Shapes come from the pinned v4 safetensors header, read with HTTP ranges
    # during integration. Expanded scalars have full shapes but allocate no
    # model-sized storage; strict loading still checks all 400 source tensors.
    shapes = {
        "cap_embedder.0.weight": (2304,),
        "cap_embedder.1.bias": (2304,),
        "cap_embedder.1.weight": (2304, 2304),
        "final_layer.adaLN_modulation.1.bias": (2304,),
        "final_layer.adaLN_modulation.1.weight": (2304, 1024),
        "final_layer.linear.bias": (64,),
        "final_layer.linear.weight": (64, 2304),
        "norm_final.weight": (2304,),
        "t_embedder.mlp.0.bias": (1024,),
        "t_embedder.mlp.0.weight": (1024, 256),
        "t_embedder.mlp.2.bias": (1024,),
        "t_embedder.mlp.2.weight": (1024, 1024),
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
    for group, count in (("layers", 26), ("context_refiner", 2), ("noise_refiner", 2)):
        for index in range(count):
            prefix = f"{group}.{index}."
            shapes.update({prefix + key: shape for key, shape in block.items()})
            if group != "context_refiner":
                shapes[prefix + "adaLN_modulation.1.bias"] = (9216,)
                shapes[prefix + "adaLN_modulation.1.weight"] = (9216, 1024)
    state = {key: torch.tensor(7.0).expand(shape) for key, shape in shapes.items()}
    state["layers.0.attention.qkv.weight"] = torch.arange(3840.0)[:, None].expand(3840, 2304)
    with torch.device("meta"):
        model = diffusers.Lumina2Transformer2DModel(cap_feat_dim=2304)
    model.save_config(tmp_path)
    return tmp_path, state


def test_original_mapping_preserves_qkv_and_materializes_every_parameter(original_checkpoint):
    config, weights = original_checkpoint
    model = module._load_transformer(config, weights)
    assert len(weights) == 400  # conversion must not consume the caller's mapping
    assert len(model.state_dict()) == 459
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    attention = model.layers[0].attn
    assert torch.equal(attention.to_q.weight[:, 0], torch.arange(2304.0))
    assert torch.equal(attention.to_k.weight[:, 0], torch.arange(2304.0, 3072.0))
    assert torch.equal(attention.to_v.weight[:, 0], torch.arange(3072.0, 3840.0))
    assert model.x_embedder.bias[0].item() == 7.0


@pytest.mark.parametrize("damage, error", [
    ("missing", "Missing key"), ("shape", "size mismatch"), ("extra", "Unexpected key"),
])
def test_incomplete_or_incompatible_checkpoint_is_refused(original_checkpoint, damage, error):
    config, weights = original_checkpoint
    if damage == "missing":
        del weights["x_embedder.bias"]
    elif damage == "shape":
        weights["x_embedder.bias"] = torch.zeros(1)
    else:
        weights["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match=error):
        module._load_transformer(config, weights)


def test_conversion_downloads_only_pinned_denoiser_and_shared_components(monkeypatch, tmp_path):
    downloads, loads, components = [], [], {}

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    checkpoint = {"sentinel": torch.ones(1)}

    def read(path, device):
        assert path == str(tmp_path) and device == "cpu"
        return checkpoint

    monkeypatch.setattr(safetensors.torch, "load_file", read)

    class Component:
        def to(self, **kwargs):
            assert kwargs == {"dtype": torch.bfloat16}
            return self

    def convert(config, state):
        assert config == tmp_path / "transformer"
        assert state is checkpoint
        return Component()

    monkeypatch.setattr(module, "_load_transformer", convert)

    def load(path, **kwargs):
        loads.append((Path(path).name, kwargs))
        return Component()

    for cls in (transformers.Gemma2Model, transformers.GemmaTokenizerFast, diffusers.AutoencoderKL):
        monkeypatch.setattr(cls, "from_pretrained", load)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=6.0, use_dynamic_shifting=False)
    scheduler.save_pretrained(tmp_path / "scheduler")

    def pipeline(**kwargs):
        components.update(kwargs)

        def save(path, **save_kwargs):
            assert path == tmp_path
            assert save_kwargs == {"safe_serialization": True}

        return SimpleNamespace(save_pretrained=save)

    monkeypatch.setattr(diffusers, "Lumina2Pipeline", pipeline)
    module._convert(tmp_path)
    shared, source = downloads
    assert source == module.SOURCE
    assert source["filename"] == "Unet/v4/NetaYumev4_unet.safetensors"
    assert source["revision"] == "69c29c441119d43405d8373c654b16827d3cf46e"
    assert shared == {**module.SHARED, "allow_patterns": module.SHARED_PATTERNS}
    assert shared["revision"] == "53504abd8178b30685b6c4c7a4cd181ff78b73e9"
    for filename in ("consolidated.00-of-01.pth", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"):
        assert not any(fnmatch.fnmatch(filename, pattern) for pattern in shared["allow_patterns"])
    assert {name for name, _ in loads} == {"text_encoder", "vae", "tokenizer"}
    assert all(kwargs["local_files_only"] for _, kwargs in loads)
    assert all(kwargs.get("dtype") == torch.bfloat16 for name, kwargs in loads if name != "tokenizer")
    assert isinstance(components["scheduler"], diffusers.FlowMatchEulerDiscreteScheduler)
    assert components["scheduler"].config.shift == 6.0
    assert not components["scheduler"].config.use_dynamic_shifting


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_normal_loader_quantizes_converted_tree_and_keeps_public_model_id(monkeypatch, tmp_path, quant):
    tree = str(tmp_path / "converted")
    configured = replace(diffusers_backend.settings, model_id=module.MODEL_ID,
                         quantization=quant, device="cpu", offload="none")
    monkeypatch.setattr(diffusers_backend, "settings", configured)
    monkeypatch.setattr(module, "resolve_model", lambda model_id, quantization: tree)
    quant_config = object() if quant != "none" else None

    def quantize(settings):
        assert settings.model_id == tree and settings.quantization == quant
        return quant_config

    class Pipe:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=6.0)

        def to(self, device):
            assert device == torch.device("cpu")
            return self

        def __call__(self, generator=None, callback_on_step_end=None, negative_prompt="", sigmas=None):
            pass

    def load(model_id, **kwargs):
        assert model_id == tree
        assert kwargs.get("quantization_config") is quant_config
        return Pipe()

    monkeypatch.setattr(loading, "quantization_config", quantize)
    monkeypatch.setattr(loading, "unused_components", lambda model_id: ("Lumina2Pipeline", {}))
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", load)
    backend = diffusers_backend.DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert diffusers_backend.settings.model_id == module.MODEL_ID
    assert backend.describe()["model_id"] == module.MODEL_ID
    assert backend.describe()["mu_supported"] is True


class _CPUModel(diffusers.ModelMixin):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, device="cpu"))


class _Tokenizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        ids = torch.tensor([[5 if prompt == "rain" else 2] for prompt in prompts])
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


class _TextEncoder(_CPUModel):
    def forward(self, input_ids, attention_mask, output_hidden_states):
        embedding = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(hidden_states=(embedding, embedding, embedding))


class _Transformer(_CPUModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=3, sample_size=32)
        self.calls = []

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask, return_dict, attention_kwargs):
        self.calls.append(encoder_hidden_states.mean().item())
        condition = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        return (hidden_states * 0.1 + condition * 0.01,)


class _VAE(_CPUModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

    def decode(self, latents, return_dict):
        return (latents.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1),)


@pytest.fixture
def live_backend(monkeypatch):
    monkeypatch.setattr(diffusers_backend, "settings", replace(
        diffusers_backend.settings, model_id=module.MODEL_ID, device="cpu", offload="none",
    ))
    pipe = diffusers.Lumina2Pipeline(
        transformer=_Transformer(), text_encoder=_TextEncoder(), tokenizer=_Tokenizer(),
        vae=_VAE(), scheduler=diffusers.FlowMatchEulerDiscreteScheduler(shift=6.0),
    )
    pipe.set_progress_bar_config(disable=True)
    backend = diffusers_backend.DiffusersBackend()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend, pipe


@pytest.mark.parametrize("guidance", [0.0, 1.0, 5.0])
def test_author_system_prompt_and_negative_cfg_reach_real_lumina_pipeline(live_backend, guidance):
    backend, pipe = live_backend
    request = GenerateRequest(prompt="sunny garden", negative_prompt="rain", width=256,
                              height=256, steps=2, guidance_scale=guidance)
    pipe.system_prompt = "a changed default must not change this checkpoint recipe"
    images = backend.generate(request, seed=17)
    assert len(images) == 1 and images[0].size == (256, 256)
    encoded = set(pipe.tokenizer.prompts)
    expected = {module.SYSTEM_PROMPT + " <Prompt Start> sunny garden"}
    if guidance > 1:
        expected.add("rain")
    assert encoded == expected
    assert pipe.transformer.calls == ([2.0, 5.0] * 2 if guidance > 1 else [2.0] * 2)
    applied = backend.applied(request)
    assert applied.prompt == "sunny garden"
    assert applied.negative_prompt == ("rain" if guidance > 1 else None)
    assert applied.true_cfg == (guidance if guidance > 1 else None)


@pytest.mark.parametrize("mu", [None, 0.0, 1.15])
def test_mu_is_the_total_shift_on_real_lumina_scheduler(live_backend, mu):
    backend, pipe = live_backend
    request = GenerateRequest(prompt="sunny garden", width=256, height=256, steps=4,
                              guidance_scale=1.0, mu=mu)
    backend.generate(request, seed=17)
    expected = schedule.shifted_sigmas(4, math.log(6.0) if mu is None else mu)
    assert pipe.scheduler.sigmas[:-1].tolist() == pytest.approx(expected, abs=1e-6)
    assert backend.applied(request).mu == mu
