"""FLUX 3 [Klein] and Klein Base: the app's pipeline over Diffusers' blocks.

Run on CPU with tiny, deterministic stand-ins for the learned models and the
processor only: chat rendering, the 80-token window, latent noise, timesteps,
the flow-map targets, guidance, scheduler steps, unpacking and image
postprocessing are all Diffusers' own. Nothing downloads; the loader's Hub
calls are replaced, and its configs are written to a temporary tree.
"""
from __future__ import annotations

import inspect
import json
import logging
import math
from types import SimpleNamespace

import pytest
import torch

from backend import model_defaults
from backend.backends import Cancelled, flux3_loading, loading
from backend.backends import schedule
from backend.backends.diffusers_backend import _REQUIRED_PARAMS
from backend.backends.flux3_pipeline import MAX_SEQUENCE_LENGTH, Flux3KleinBasePipeline, Flux3KleinPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin

CHANNELS = 8      # packed latent channels (the real transformer's 128)
HIDDEN = 2        # per stacked layer (Qwen3-VL-4B's 2560)


class _Processor:
    """`AutoProcessor` as the encoder step uses it: a chat template and its tokenizer."""

    def __init__(self):
        self.prompts: list[str] = []
        self.windows: list[int] = []
        self.tokenizer = self._tokenize

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        (message,) = messages
        assert message["role"] == "user" and not tokenize and add_generation_prompt
        return f"<user>{message['content']}</user><assistant>"

    def _tokenize(self, text, return_tensors, truncation, max_length, padding=None, padding_side=None):
        content = text.removeprefix("<user>").split("</user>")[0]
        words = content.split() or [""]
        ids = [3 if word == "sunny" else 7 if word == "rain" else 5 for word in words][:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            self.prompts.append(content)
            self.windows.append(max_length)
            mask += [0] * (max_length - len(ids))
            ids += [0] * (max_length - len(ids))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


class _CPUModel(ModelMixin):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, device="cpu"))


class _TextEncoder(_CPUModel):
    def forward(self, input_ids, attention_mask, position_ids, output_hidden_states, use_cache):
        assert output_hidden_states and not use_cache
        # Padding positions sit at 1, as the training environment had them.
        assert torch.equal(position_ids[attention_mask == 0], torch.ones_like(position_ids[attention_mask == 0]))
        base = (input_ids.float() * attention_mask).unsqueeze(-1).repeat(1, 1, HIDDEN)
        return SimpleNamespace(hidden_states=tuple(base * (layer + 1) / 100 for layer in range(37)))


class _Transformer(_CPUModel):
    def __init__(self, flow_map: bool):
        super().__init__()
        self.config = SimpleNamespace(in_channels=CHANNELS)
        self.flow_map = flow_map
        self.calls: list[dict] = []

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, return_dict,
                target_timestep=None, image_cond_hidden_states=None, image_cond_ids=None):
        # The real transformer refuses a target on a non-flow-map model and
        # requires one on a flow-map model; so does this one.
        assert (target_timestep is not None) is self.flow_map
        assert encoder_hidden_states.shape[-1] == 8 * HIDDEN
        assert txt_ids.shape[0] == encoder_hidden_states.shape[1]
        condition = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1)
        reference = 0 if image_cond_hidden_states is None else image_cond_hidden_states.mean()
        prediction = hidden_states * 0.1 + condition + reference
        self.calls.append({
            "latents": hidden_states.clone(), "prediction": prediction.clone(),
            "timestep": timestep.clone(),
            "target": None if target_timestep is None else target_timestep.clone(),
            "references": image_cond_hidden_states is not None,
        })
        return (prediction,)


class _VAE(_CPUModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_out_channels=[1, 1, 1, 1], batch_norm_eps=1e-4)
        self.bn = SimpleNamespace(running_mean=torch.zeros(CHANNELS), running_var=torch.ones(CHANNELS))
        self.decode_calls = 0
        self.encode_calls = 0

    def decode(self, latents, return_dict):
        self.decode_calls += 1
        assert latents.shape[1] == CHANNELS // 4
        image = latents[:, :1].repeat(1, 3, 1, 1).repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
        return (torch.tanh(image),)

    def encode(self, image):
        self.encode_calls += 1
        pooled = torch.nn.functional.avg_pool2d(image[:, :1], 8).repeat(1, CHANNELS // 4, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pooled))


def _pipe(cls):
    pipeline = cls(
        tokenizer=_Processor(), text_encoder=_TextEncoder(),
        transformer=_Transformer(flow_map=cls is Flux3KleinPipeline), vae=_VAE(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=flux3_loading.SHIFT),
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@pytest.fixture(params=[Flux3KleinPipeline, Flux3KleinBasePipeline])
def pipe(request):
    return _pipe(request.param)


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _render(pipe, **changes):
    args = dict(prompt="sunny garden", width=32, height=32, num_inference_steps=3, generator=_generator(17))
    args.update(changes)
    return pipe(**args).images


# --- the call ------------------------------------------------------------------


def test_the_call_takes_what_load_requires_and_the_profiles_say():
    for cls, model_id in ((Flux3KleinPipeline, flux3_loading.MODEL_ID),
                          (Flux3KleinBasePipeline, flux3_loading.BASE_MODEL_ID)):
        params = inspect.signature(cls.__call__).parameters
        profile = model_defaults.defaults_for(model_id)
        assert _REQUIRED_PARAMS <= set(params)
        assert ("negative_prompt" in params) is profile["negative_prompt_supported"]
        assert ("image" in params) is profile["reference_supported"]
        assert "sigmas" in params and profile["mu_supported"]
        assert "true_cfg_scale" not in params and not profile["true_cfg_supported"]
        assert params["num_inference_steps"].default == profile["steps"]
        assert params["guidance_scale"].default == profile["guidance_scale"]
        assert profile["mu_static"] == math.log(flux3_loading.SHIFT)
    # The cards' recipes: 16 distilled steps without guidance, Base 50 at 3.5.
    assert model_defaults.defaults_for(flux3_loading.MODEL_ID)["steps"] == 16
    base = model_defaults.defaults_for(flux3_loading.BASE_MODEL_ID)
    assert (base["steps"], base["guidance_scale"]) == (50, 3.5)


def test_the_profiles_are_their_own_never_flux2_kleins_or_flux1s():
    for model_id in (flux3_loading.MODEL_ID, flux3_loading.BASE_MODEL_ID):
        profile = model_defaults._profile_for(model_id)
        assert profile is model_defaults._profile_for(model_id.upper())
        assert profile["guide"] is model_defaults.guides.FLUX3_KLEIN_GUIDE
        assert profile is not model_defaults._profile_for("black-forest-labs/FLUX.2-klein-base-4B")
        assert model_defaults.defaults_for(model_id)["refine_settled"]
        assert not model_defaults.defaults_for(model_id)["inpaint_deferred"]
    # FLUX.2's ids still reach FLUX.2's profiles.
    assert model_defaults.defaults_for("black-forest-labs/FLUX.2-klein-4B")["steps"] == 4


@pytest.mark.parametrize("cls", [Flux3KleinPipeline, Flux3KleinBasePipeline])
def test_batch_images_reproduce_individually_with_cpu_generators(cls):
    pipe = _pipe(cls)
    seeds = [17, 18, 19]
    images = _render(pipe, generator=[_generator(seed) for seed in seeds], num_images_per_prompt=len(seeds))
    assert len(images) == len(seeds)
    assert len({image.tobytes() for image in images}) == len(seeds)
    for seed, image in zip(seeds, images):
        again = _render(pipe, generator=_generator(seed))[0]
        assert image.mode == "RGB" and image.size == (32, 32)
        assert image.tobytes() == again.tobytes()


def test_the_window_is_the_prompt_padded_to_eighty_with_the_stacked_layers(pipe):
    _render(pipe, guidance_scale=1.0)
    # One prompt, rounded up to the pipeline's 80-token multiple, under 8192.
    assert pipe.tokenizer.prompts == ["sunny garden"]
    assert pipe.tokenizer.windows == [80]
    assert MAX_SEQUENCE_LENGTH == 8192
    embeds = pipe.transformer.calls[0]
    assert embeds["latents"].shape == (1, 4, CHANNELS)   # (32/16)^2 tokens


def test_the_distilled_loop_jumps_to_each_next_timestep():
    pipe = _pipe(Flux3KleinPipeline)
    _render(pipe, num_inference_steps=4, guidance_scale=7.0)
    calls = pipe.transformer.calls
    assert len(calls) == 4   # guidance never adds a pass on the distilled model
    timesteps = pipe.scheduler.timesteps / 1000
    for i, call in enumerate(calls):
        torch.testing.assert_close(call["timestep"], timesteps[i].expand(1))
        expected = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0)
        torch.testing.assert_close(call["target"], expected.expand(1))
    assert pipe.guider is None


@pytest.mark.parametrize("guidance", [0.0, 1.0, 3.5])
def test_base_guides_with_its_negative_prompt_only_above_one(guidance):
    pipe = _pipe(Flux3KleinBasePipeline)
    finished = []

    def record(pipeline, step, timestep, values):
        finished.append(values["latents"].clone())
        return values

    _render(pipe, num_inference_steps=1, guidance_scale=guidance, negative_prompt="rain",
            callback_on_step_end=record)
    expected = ["sunny garden", "rain"] if guidance > 1 else ["sunny garden"]
    assert pipe.tokenizer.prompts == expected
    calls = pipe.transformer.calls
    assert len(calls) == len(expected)
    positive = calls[0]["prediction"]
    prediction = positive
    if guidance > 1:
        negative = calls[1]["prediction"]
        assert not torch.equal(positive, negative)
        prediction = negative + guidance * (positive - negative)
    # One real FlowMatch step from the first sigma to zero.
    sigma = pipe.scheduler.sigmas[0]
    torch.testing.assert_close(finished[0], calls[0]["latents"] - sigma * prediction)


def test_base_guided_without_a_negative_encodes_an_empty_one_and_a_fresh_guider_each_call():
    pipe = _pipe(Flux3KleinBasePipeline)
    _render(pipe, guidance_scale=0.0)
    unguided = pipe.guider
    first = _render(pipe, guidance_scale=3.5)
    guided = pipe.guider
    second = _render(pipe, guidance_scale=3.5)
    assert pipe.guider is not guided and guided is not unguided
    assert first[0].tobytes() == second[0].tobytes()
    assert pipe.tokenizer.prompts == ["sunny garden", "sunny garden", "", "sunny garden", ""]


def test_requested_sigmas_land_as_the_whole_shift(pipe):
    # The scheduler applies its fixed 6.93 to whatever it is handed, so the
    # backend sends the schedule less that (`schedule.requested_shift`) and
    # the total is e^mu. Asserted on the scheduler, never on the list.
    mu = 1.2
    assert schedule.mu_supported(pipe, set(inspect.signature(pipe.__call__).parameters))
    assert schedule.static_shift(pipe) == pytest.approx(flux3_loading.SHIFT)
    sigmas, forced = schedule.requested_shift(pipe, 4, mu)
    assert forced is None and sigmas is not None
    _render(pipe, num_inference_steps=4, sigmas=sigmas)
    torch.testing.assert_close(
        pipe.scheduler.sigmas[:-1], torch.tensor(schedule.shifted_sigmas(4, mu)), atol=1e-5, rtol=0,
    )
    # A blank Mu is the blocks' own linear ramp under the same fixed shift.
    _render(pipe, num_inference_steps=4)
    torch.testing.assert_close(
        pipe.scheduler.sigmas[:-1], torch.tensor(schedule.shifted_sigmas(4, math.log(flux3_loading.SHIFT))),
        atol=1e-5, rtol=0,
    )


def test_references_run_the_conditioned_core_at_the_requested_size(pipe):
    from PIL import Image

    reference = Image.new("RGB", (48, 48), (200, 40, 40))
    images = _render(pipe, image=[reference])
    assert pipe.vae.encode_calls == 1
    assert all(call["references"] for call in pipe.transformer.calls)
    # The requested size wins over the reference's own (no crop-back).
    assert images[0].size == (32, 32)
    plain = _pipe(type(pipe))
    _render(plain)
    assert plain.vae.encode_calls == 0
    assert not any(call["references"] for call in plain.transformer.calls)


def test_callback_observes_completed_steps_and_cancel_skips_decode(pipe, monkeypatch, caplog):
    # Diffusers disables propagation by default; opt in so caplog would catch
    # a misleading "Error in block" traceback for a cooperative cancel.
    monkeypatch.setattr(logging.getLogger("diffusers"), "propagate", True)
    caplog.set_level(logging.ERROR, logger="diffusers")
    pipe.enable_model_cpu_offload(device="cpu")
    previous_hooks = list(pipe._all_hooks)
    freed = []
    original_cleanup = pipe.maybe_free_model_hooks

    def cleanup():
        freed.append(True)
        original_cleanup()

    monkeypatch.setattr(pipe, "maybe_free_model_hooks", cleanup)
    seen = []

    def cancel(pipeline, step, timestep, values):
        assert pipeline is pipe
        assert pipeline.scheduler.step_index == step + 1
        assert values["latents"].shape == (1, 4, CHANNELS)
        assert timestep == pipeline.scheduler.timesteps[step]
        seen.append(step)
        if step == 1:
            raise Cancelled
        return values

    with pytest.raises(Cancelled):
        _render(pipe, callback_on_step_end=cancel)
    assert not any(
        record.levelno >= logging.ERROR and "Error in block" in record.getMessage()
        for record in caplog.records
    )
    assert seen == [0, 1]
    assert pipe.vae.decode_calls == 0
    assert pipe._step.callback is None
    assert freed == [True]
    assert len(pipe._all_hooks) == len(previous_hooks) == 3
    # The next call inherits neither the callback nor a half-run scheduler.
    images = _render(pipe)
    assert len(images) == 1 and seen == [0, 1]
    assert pipe.vae.decode_calls == 1
    assert freed == [True, True]


def test_a_callback_may_replace_the_latents(pipe):
    def zero(pipeline, step, timestep, values):
        return {"latents": torch.zeros_like(values["latents"])}

    _render(pipe, num_inference_steps=2, guidance_scale=1.0, callback_on_step_end=zero)
    assert torch.count_nonzero(pipe.transformer.calls[1]["latents"]) == 0


def test_generator_count_must_match_batch(pipe):
    with pytest.raises(ValueError, match="list of generators of length 1"):
        _render(pipe, generator=[_generator(17)], num_images_per_prompt=2)
    assert pipe.transformer.calls == []
    assert pipe.vae.decode_calls == 0
    assert pipe._step.callback is None


def test_no_preview_map_for_its_vae():
    # Previews switch themselves off (no fitted AutoencoderKLFlux2 map yet).
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    from backend import latent_preview

    assert not any(key.startswith(AutoencoderKLFlux2.__name__) for key in latent_preview.FACTORS)


# --- loading -------------------------------------------------------------------


def _tree(tmp_path, model_id, *, index_class=None, flow_map=None, transformer_class="Flux3Transformer2DModel"):
    pipeline, distilled = flux3_loading._VARIANTS[model_id.lower()]
    index = {"_class_name": index_class or pipeline, "_diffusers_version": "0.41.0.dev0"}
    for name, (library, cls) in flux3_loading.EXPECTED.items():
        index[name] = [library, cls, {"pretrained_model_name_or_path": model_id, "subfolder": name}]
    index["tokenizer"] = ["transformers", "Qwen3VLProcessor", {"subfolder": "text_encoder"}]
    index["prompt_upsampler"] = ["transformers", "Qwen3_5ForConditionalGeneration", {"subfolder": ""}]
    (tmp_path / flux3_loading.INDEX).write_text(json.dumps(index), encoding="utf-8")
    for name in flux3_loading.COMPONENTS:
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / "config.json").write_text("{}", encoding="utf-8")
    config = {"_class_name": transformer_class, "flow_map": distilled if flow_map is None else flow_map}
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("model_id", [flux3_loading.MODEL_ID, flux3_loading.BASE_MODEL_ID])
def test_the_snapshot_is_pinned_and_never_the_root_files_or_variants(tmp_path, monkeypatch, model_id):
    import huggingface_hub

    calls = []

    def snapshot(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, list(allow_patterns)))
        return _tree(tmp_path, model_id)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    configs = loading.resolve_model(model_id, quantization="fp8")
    weights = loading.resolve_model(model_id, weights=True)
    assert configs == weights == str(tmp_path)
    pin = flux3_loading.source(model_id)
    assert [(repo, revision) for repo, revision, _ in calls] == [(pin["repo_id"], pin["revision"])] * 2
    assert len(pin["revision"]) == 40
    assert calls[0][2] == flux3_loading.CONFIG_PATTERNS
    assert calls[1][2] == flux3_loading.WEIGHT_PATTERNS
    for _, _, patterns in calls:
        assert not any(p.startswith(("variants", "model.", "ae.", "*")) for p in patterns)
        assert all(p == flux3_loading.INDEX or p.split("/")[0] in flux3_loading.COMPONENTS for p in patterns)


@pytest.mark.parametrize("change, words", [
    ({"index_class": "Flux3ModularPipeline"}, "does not name Flux3DistilledModularPipeline"),
    ({"flow_map": False}, "not the step-distilled (flow-map) one"),
    ({"transformer_class": "Flux2Transformer2DModel"}, "is not a Flux3Transformer2DModel"),
])
def test_the_preflight_refuses_another_variant_before_weights(tmp_path, change, words):
    tree = _tree(tmp_path, flux3_loading.MODEL_ID, **change)
    with pytest.raises(RuntimeError, match=words.replace("(", r"\(").replace(")", r"\)")):
        flux3_loading.preflight(flux3_loading.MODEL_ID, tree)


def test_the_base_preflight_wants_the_undistilled_transformer(tmp_path):
    tree = _tree(tmp_path, flux3_loading.BASE_MODEL_ID, flow_map=True)
    with pytest.raises(RuntimeError, match="not the undistilled one"):
        flux3_loading.preflight(flux3_loading.BASE_MODEL_ID, tree)
    flux3_loading.preflight(flux3_loading.BASE_MODEL_ID, _tree(tmp_path, flux3_loading.BASE_MODEL_ID))


def test_an_index_naming_another_encoder_is_refused(tmp_path):
    tree = _tree(tmp_path, flux3_loading.MODEL_ID)
    index = json.loads((tmp_path / flux3_loading.INDEX).read_text(encoding="utf-8"))
    index["text_encoder"][1] = "Qwen3ForCausalLM"
    (tmp_path / flux3_loading.INDEX).write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Qwen3VLForConditionalGeneration as text_encoder"):
        flux3_loading.preflight(flux3_loading.MODEL_ID, tree)


def test_nunchaku_is_refused_before_any_download(monkeypatch):
    import huggingface_hub

    def refuse(*args, **kwargs):
        raise AssertionError("nothing downloads")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    with pytest.raises(RuntimeError, match="no Nunchaku Lite build"):
        loading.resolve_model(flux3_loading.MODEL_ID, quantization="nunchaku")


def test_quantization_is_per_component_from_the_modular_index(tmp_path, monkeypatch):
    import os

    from diffusers.quantizers.quantization_config import TorchAoConfig as DiffusersTorchAoConfig
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    tree = _tree(tmp_path, flux3_loading.MODEL_ID)
    # The local snapshot's own files (the configs pass put them there).
    monkeypatch.setattr(loading, "hub_file", lambda model_id, filename: os.path.join(model_id, filename))
    assert loading.component_libraries(tree) == {"transformer": "diffusers", "text_encoder": "transformers"}
    settings = SimpleNamespace(model_id=tree, quantization="fp8")
    config = loading.quantization_config(settings)
    assert set(config.quant_mapping) == {"transformer", "text_encoder"}
    assert isinstance(config.quant_mapping["transformer"], DiffusersTorchAoConfig)
    assert isinstance(config.quant_mapping["text_encoder"], TransformersTorchAoConfig)
    # A directory with an ordinary index keeps the ordinary path.
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    assert flux3_loading.component_libraries(tree) is None


@pytest.mark.parametrize("quantized", [False, True])
def test_components_load_from_their_folders_and_release_the_encoders_files(tmp_path, monkeypatch, quantized):
    import diffusers.models.autoencoders.autoencoder_kl_flux2 as vae_module
    import diffusers.models.transformers.transformer_flux3 as transformer_module
    import transformers

    tree = _tree(tmp_path, flux3_loading.MODEL_ID)
    loads = {}

    def loader(name, built):
        def from_pretrained(path, **kwargs):
            loads[name] = (str(path), kwargs)
            return built
        return from_pretrained

    built = {
        "transformer": _Transformer(flow_map=True), "text_encoder": _TextEncoder(),
        "vae": _VAE(), "tokenizer": _Processor(),
    }
    monkeypatch.setattr(transformer_module.Flux3Transformer2DModel, "from_pretrained",
                        loader("transformer", built["transformer"]))
    monkeypatch.setattr(transformers.Qwen3VLForConditionalGeneration, "from_pretrained",
                        loader("text_encoder", built["text_encoder"]))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", loader("tokenizer", built["tokenizer"]))
    monkeypatch.setattr(vae_module.AutoencoderKLFlux2, "from_pretrained", loader("vae", built["vae"]))
    owned = []
    monkeypatch.setattr(loading, "own_tensors", lambda module, names=None: owned.append(module) or 0)
    mapping = {"transformer": object(), "text_encoder": object()} if quantized else {}
    quant = SimpleNamespace(quant_mapping=mapping) if quantized else None

    pipe = Flux3KleinPipeline.from_pretrained(tree, dtype=torch.bfloat16, quantization_config=quant)
    assert isinstance(pipe, Flux3KleinPipeline)
    assert pipe.transformer is built["transformer"] and pipe.vae is built["vae"]
    assert loads["transformer"] == (tree, {
        "subfolder": "transformer", "dtype": torch.bfloat16,
        "quantization_config": mapping.get("transformer"), "local_files_only": True,
    })
    assert loads["text_encoder"][0].endswith("text_encoder")
    assert loads["text_encoder"][1]["quantization_config"] is mapping.get("text_encoder")
    assert loads["tokenizer"][0] == loads["text_encoder"][0]
    # The README's dtypes: the VAE stays float32, never quantized.
    assert loads["vae"][1]["dtype"] is torch.float32 and "quantization_config" not in loads["vae"][1]
    assert isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler)
    assert pipe.scheduler.shift == pytest.approx(flux3_loading.SHIFT)
    assert not pipe.scheduler.config.use_dynamic_shifting
    assert owned == ([built["text_encoder"]] if quantized else [])


def test_the_loader_refuses_what_it_cannot_honour(tmp_path):
    tree = _tree(tmp_path, flux3_loading.MODEL_ID)
    with pytest.raises(TypeError, match="takes no pe"):
        Flux3KleinPipeline.from_pretrained(tree, pe=None)
    with pytest.raises(ValueError, match="no quantizable vae"):
        Flux3KleinPipeline.from_pretrained(tree, quantization_config=SimpleNamespace(quant_mapping={"vae": 1}))


def test_the_loader_and_lora_refusal_are_keyed_by_the_exact_ids():
    assert flux3_loading.pipeline_class(flux3_loading.MODEL_ID) is Flux3KleinPipeline
    assert flux3_loading.pipeline_class(flux3_loading.BASE_MODEL_ID.upper()) is Flux3KleinBasePipeline
    assert not flux3_loading.matches("black-forest-labs/FLUX.2-klein-4B")
    with pytest.raises(ValueError):
        flux3_loading.pipeline_class("black-forest-labs/flux-3-image-klein-pup")
    for model_id in (flux3_loading.MODEL_ID, flux3_loading.BASE_MODEL_ID):
        assert "not supported by this FLUX 3 [Klein]" in loading.lora_refusal(model_id)
    assert loading.lora_refusal("black-forest-labs/FLUX.2-klein-4B") is None


# --- the backend's load and a cell ---------------------------------------------


@pytest.mark.parametrize("quant", ["fp8", "none"])
@pytest.mark.parametrize("cls", [Flux3KleinPipeline, Flux3KleinBasePipeline])
def test_the_backend_loads_the_app_pipeline_from_the_pinned_snapshot_and_runs_a_cell(
    tmp_path, monkeypatch, quant, cls,
):
    import os
    from dataclasses import replace

    import diffusers
    import huggingface_hub

    from backend.backends import diffusers_backend
    from backend.schemas import GenerateRequest

    model_id = flux3_loading.MODEL_ID if cls is Flux3KleinPipeline else flux3_loading.BASE_MODEL_ID
    tree = _tree(tmp_path, model_id)
    configured = replace(diffusers_backend.settings, model_id=model_id, quantization=quant,
                         device="cpu", offload="none")
    monkeypatch.setattr(diffusers_backend, "settings", configured)
    snapshots = []

    def snapshot(repo_id, revision, allow_patterns):
        snapshots.append(list(allow_patterns))
        return tree

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(loading, "hub_file", lambda model_id, filename: os.path.join(model_id, filename))

    def wrong_loader(*args, **kwargs):
        pytest.fail("FLUX 3 must load through its app-owned pipeline")

    built = _pipe(cls)
    loads = []

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return built

    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", wrong_loader)
    monkeypatch.setattr(cls, "from_pretrained", load)
    backend = diffusers_backend.DiffusersBackend()
    assert "not supported by this FLUX 3" in backend.lora_unavailable_reason()
    backend.load()
    assert backend.is_ready()
    assert snapshots == [flux3_loading.CONFIG_PATTERNS, flux3_loading.WEIGHT_PATTERNS]
    ((path, kwargs),) = loads
    assert path == tree and kwargs["dtype"] is torch.bfloat16
    quantized = kwargs.get("quantization_config")
    assert (quantized is not None) is (quant != "none")
    if quantized is not None:
        assert set(quantized.quant_mapping) == {"transformer", "text_encoder"}
    described = backend.describe()
    assert described["model_id"] == model_id
    assert described["mu_supported"] is True and described["cfg_threshold"] == 1.0
    assert described["negative_prompt_supported"] is (cls is Flux3KleinBasePipeline)
    assert backend._reference_supported()

    progress = []
    request = GenerateRequest(prompt="sunny garden", negative_prompt="rain", width=256, height=256,
                              steps=2, guidance_scale=1.0, mu=1.2)
    images = backend.generate(request, seed=5, on_progress=lambda done, total: progress.append((done, total)))
    assert len(images) == 1 and images[0].size == (256, 256)
    assert progress == [(1, 2), (2, 2)]
    # The recorded Mu is the whole shift on the scheduler.
    torch.testing.assert_close(built.scheduler.sigmas[:-1], torch.tensor(schedule.shifted_sigmas(2, 1.2)),
                               atol=1e-5, rtol=0)
    applied = backend.applied(request)
    assert applied.mu == 1.2
    # No negative prompt on the distilled model; below Base's threshold none runs either.
    assert applied.negative_prompt is None
    assert built.tokenizer.prompts == ["sunny garden"]


def test_the_catalogue_lists_both_before_the_editor_with_their_cold_facts():
    from backend import hub_access, launch, model_info

    repos = [entry.repo for entry in launch.CHECKPOINTS]
    assert repos[-3:-1] == [flux3_loading.MODEL_ID, flux3_loading.BASE_MODEL_ID]
    for model_id in (flux3_loading.MODEL_ID, flux3_loading.BASE_MODEL_ID):
        entry = next(entry for entry in launch.CHECKPOINTS if entry.repo == model_id)
        assert entry.gate == "click-through gate" and entry.lite is None
        detail = model_info.details_for(model_id, quantization="fp8")
        assert detail["access"] == "Accept licence" and detail["refine"] == "No"
        # The shard maps were read on 2026-11-12 (tools/model_downloads.py):
        # the loader's two snapshot passes, about 20.3 GB.
        assert detail["download"] == "~20.3 GB"
        assert "Only 1024×1024 has been measured on this card" in detail["megapixels_note"]
        # Measured 2026-11-10: 6.08 GiB allocated at fp8 on a plain image, under
        # the 6.8 GiB below which a card is not sized, as FLUX.2 [klein] 4B.
        assert model_id not in model_info._CARD_NEEDS
        assert model_info.details_for(model_id, quantization="fp8", card_memory=4 * 1024 ** 3)["fit"] == ""
    # Klein's BF16 peak (10.95 GiB) is over an 8 GB card; Base's is unmeasured.
    bf16 = model_info.details_for(flux3_loading.MODEL_ID, quantization="none", card_memory=8 * 1024 ** 3)
    assert bf16["fit"] == "Over 8 GB" and bf16["fit_note"].startswith("BF16 with model offload peaked at 10.95 GiB")
    assert model_info.details_for(flux3_loading.MODEL_ID, quantization="none", card_memory=12 * 1024 ** 3)["fit"] == ""
    assert flux3_loading.BASE_MODEL_ID not in model_info._BF16_NEEDS
    for model_id in (flux3_loading.MODEL_ID, flux3_loading.BASE_MODEL_ID):
        # A switch asks the Hub about the pinned repository itself, with a HEAD
        # on its modular index now that the selection is known.
        pin = flux3_loading.source(model_id)
        assert hub_access.sources(model_id) == [
            hub_access.Source(pin["repo_id"], pin["revision"], flux3_loading.INDEX)]
