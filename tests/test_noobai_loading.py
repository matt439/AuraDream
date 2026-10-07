"""NoobAI's official component tree needs its documented v-pred schedule."""
from __future__ import annotations

import dataclasses
import fnmatch
import json
from pathlib import Path
from types import SimpleNamespace

import diffusers
import huggingface_hub
import pytest
import torch

from backend import refine
from backend.backends import diffusers_backend as module
from backend.backends import loading
from backend.schemas import GenerateRequest

MODEL = loading.NOOB_MODEL_ID
# The official scheduler at 66aa55e3469c27c29a89813cd35dd95fb7485fa1.
# Its epsilon and zero-SNR fields contradict the same revision's model card.
PACKAGED_SCHEDULER = {
    "_class_name": "EulerDiscreteScheduler", "num_train_timesteps": 1000,
    "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon", "rescale_betas_zero_snr": False,
    "timestep_spacing": "leading", "steps_offset": 1,
    "interpolation_type": "linear", "final_sigmas_type": "zero",
    "use_karras_sigmas": False, "use_exponential_sigmas": False, "use_beta_sigmas": False,
}


@pytest.fixture
def official_tree(monkeypatch, tmp_path):
    files = {
        "model_index.json": {
            "_class_name": "StableDiffusionXLPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        },
        "unet/config.json": {}, "text_encoder/config.json": {},
        "text_encoder_2/config.json": {}, "vae/config.json": {},
        "scheduler/scheduler_config.json": PACKAGED_SCHEDULER,
        "unet/diffusion_pytorch_model.safetensors": "UNet",
        "text_encoder/model.safetensors": "CLIP-L",
        "text_encoder_2/model.safetensors": "CLIP-G",
        "vae/diffusion_pytorch_model.safetensors": "VAE",
        "tokenizer/merges.txt": "merges", "tokenizer_2/vocab.json": {},
        "NoobAI-XL-Vpred-v1.0.safetensors": "unwanted duplicate single file",
        "vae/a6d5fd6a7c4c7f1e0af299bc3a0d4fab.png": "unwanted illustration",
    }
    calls = []

    def snapshot(model_id, *, revision, allow_patterns):
        assert model_id == MODEL
        assert revision == "66aa55e3469c27c29a89813cd35dd95fb7485fa1"
        selected = [name for name in files if any(fnmatch.fnmatchcase(name, p) for p in allow_patterns)]
        assert "NoobAI-XL-Vpred-v1.0.safetensors" not in selected
        assert not any(name.endswith(".png") for name in selected)
        calls.append(selected)
        for name in selected:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            value = files[name]
            path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")
        return str(tmp_path)

    def local_file(model_id, filename):
        assert Path(model_id) == tmp_path
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(loading, "hub_file", local_file)
    return tmp_path, calls, files


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_noobai_load_keeps_both_encoders_and_fixes_only_scheduler(monkeypatch, official_tree, quant):
    tree, calls, _ = official_tree
    monkeypatch.setattr(module, "settings", dataclasses.replace(
        module.settings, model_id=MODEL, quantization=quant, device="cpu", offload="none",
    ))
    ordinary_quantization = loading.quantization_config

    def quantization(settings):
        assert Path(settings.model_id) == tree
        assert len(calls) == 1
        assert not list(tree.rglob("*.safetensors"))
        config = ordinary_quantization(settings)
        if quant != "none":
            assert set(config.quant_mapping) == {"unet", "text_encoder", "text_encoder_2"}
            assert type(config.quant_mapping["unet"]).__module__.startswith("diffusers.")
            assert type(config.quant_mapping["text_encoder"]).__module__.startswith("transformers.")
            assert type(config.quant_mapping["text_encoder_2"]).__module__.startswith("transformers.")
        return config

    class StableDiffusionXLPipeline:
        def __init__(self, scheduler):
            self.scheduler = scheduler

        def to(self, device):
            return self

        def __call__(self, generator=None, callback_on_step_end=None, negative_prompt="", sigmas=None):
            pass

    def load(model_id, **kwargs):
        assert Path(model_id) == tree and len(calls) == 2
        assert len(list(tree.rglob("*.safetensors"))) == 4
        assert (tree / "tokenizer/merges.txt").is_file()
        assert (tree / "tokenizer_2/vocab.json").is_file()
        assert not {"vae", "text_encoder", "text_encoder_2", "unet"} & kwargs.keys()
        assert ("quantization_config" in kwargs) == (quant != "none")
        scheduler = kwargs["scheduler"]
        assert type(scheduler) is diffusers.EulerDiscreteScheduler
        assert scheduler.config.prediction_type == "v_prediction"
        assert scheduler.config.rescale_betas_zero_snr
        assert scheduler.config.timestep_spacing == "trailing"
        return StableDiffusionXLPipeline(scheduler)

    monkeypatch.setattr(loading, "quantization_config", quantization)
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", load)
    backend = module.DiffusersBackend()
    backend.load()
    assert backend.is_ready() and backend.describe()["model_id"] == MODEL
    assert not backend.describe()["mu_supported"]
    request = backend.applied(GenerateRequest(prompt="a landscape", negative_prompt="blur", guidance_scale=5, mu=1))
    assert request.negative_prompt == "blur" and request.mu is None
    assert backend.cfg_threshold() == 1.0
    # Correcting the scheduler must never alter the pinned Hub files.
    assert json.loads((tree / "scheduler/scheduler_config.json").read_text()) == PACKAGED_SCHEDULER


@pytest.mark.parametrize("quant", ["fp8", "nunchaku"])
def test_bad_quantizer_refuses_before_any_noobai_weights(monkeypatch, official_tree, quant):
    tree, calls, files = official_tree
    monkeypatch.setattr(module, "settings", dataclasses.replace(
        module.settings, model_id=MODEL, quantization=quant, device="cpu", offload="none",
    ))
    files["unet/config.json"] = {"quantization_config": {"quant_method": "other"}}
    expected = "already quantized" if quant == "fp8" else "not a Nunchaku Lite checkpoint"
    with pytest.raises(RuntimeError, match=expected):
        module.DiffusersBackend().load()
    assert len(calls) == (1 if quant == "fp8" else 0)
    assert not list(tree.rglob("*.safetensors"))


def test_noobai_lora_fit_reads_pinned_configs_without_weights(monkeypatch, official_tree):
    tree, calls, _ = official_tree
    monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, model_id=MODEL.upper()))
    sentinel = object()

    def denoiser(source):
        assert Path(source) == tree
        assert len(calls) == 1 and (tree / "unet/config.json").is_file()
        assert not list(tree.rglob("*.safetensors"))
        return sentinel

    monkeypatch.setattr(loading, "denoiser", denoiser)
    assert module.DiffusersBackend._configured_denoiser() is sentinel


@pytest.mark.parametrize("steps", [1, 28, 35])
def test_noobai_real_euler_starts_at_zero_snr_and_resets_after_refine(official_tree, steps):
    tree, _, _ = official_tree
    loading.resolve_model(MODEL)
    scheduler = loading.component_overrides(MODEL, None, resolved_model_id=str(tree))["scheduler"]
    scheduler.set_timesteps(steps, device="cpu")
    initial_sigmas = scheduler.sigmas.clone()
    assert scheduler.timesteps[0].item() == 999
    assert scheduler.sigmas[0].item() == pytest.approx(4096, rel=1e-6)
    assert scheduler.sigmas[-1].item() == 0
    assert torch.isfinite(scheduler.sigmas).all()
    assert torch.all(scheduler.sigmas[:-1] > scheduler.sigmas[1:])
    assert scheduler.alphas_cumprod[-1].item() == pytest.approx(2**-24)

    # Exercise the installed SDXL Refine slicing, including a later direct
    # call on the same scheduler, as the backend's two pipelines share it.
    strength = 1.0 if steps == 1 else 0.4
    pipe = SimpleNamespace(scheduler=scheduler)
    timesteps, count = diffusers.StableDiffusionXLImg2ImgPipeline.get_timesteps(pipe, steps, strength, "cpu")
    begin = steps - int(steps * strength)
    assert scheduler.begin_index == begin
    assert count == len(timesteps) == steps - begin
    assert scheduler.sigmas[begin] == initial_sigmas[begin]
    assert refine.sibling(diffusers.StableDiffusionXLPipeline) is diffusers.StableDiffusionXLImg2ImgPipeline
    sample, prediction = torch.full((1, 4, 2, 2), 0.25), torch.full((1, 4, 2, 2), 0.1)
    scheduler.scale_model_input(sample, timesteps[0])
    sigma = initial_sigmas[begin]
    output = scheduler.step(prediction, timesteps[0], sample)
    # This is velocity prediction's x0, distinct from epsilon's x - sigma*e.
    expected = prediction * (-sigma / (sigma**2 + 1)**0.5) + sample / (sigma**2 + 1)
    assert torch.allclose(output.pred_original_sample, expected)
    assert torch.isfinite(output.prev_sample).all()
    scheduler.set_timesteps(steps, device="cpu")
    assert scheduler.begin_index is None and scheduler.step_index is None
    assert torch.equal(scheduler.sigmas, initial_sigmas)
