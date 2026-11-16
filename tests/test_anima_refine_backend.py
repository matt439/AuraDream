"""The public backend's Anima Refine orchestration with real CPU blocks.

Only the learned components/tokenizers are tiny stand-ins. These checks use
the installed encoder, schedule, denoising and decode blocks plus the real
gallery-source reader, backend dispatcher and CPU offload hooks. They make
no claim about a checkpoint's GPU pixels, quality, fit or speed.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
import torch

from backend import card_facts, config, model_selection, refine, refine_admission, schemas_generate
from backend.backends import Cancelled, base, diffusers_backend
from backend.backends.base import _refine_native_observed
from backend.backends.anima_pipeline import AnimaAestheticPipeline, AnimaPreviewPipeline, AnimaTurboPipeline
from backend.backends.anima_refine import (
    AnimaAestheticImg2ImgPipeline, AnimaPreviewImg2ImgPipeline, AnimaTurboImg2ImgPipeline,
)
from backend.schemas import GenerateRequest
from tests.backend_helpers import loaded_backend
from tests.test_anima_refine import _text_pipe as text_pipe


STRENGTHS = (.125, .25, .4)
HOST = ("win32", "0.42.1", "NVIDIA GeForce RTX 5080", True)
SEED = 4471


@dataclasses.dataclass(frozen=True)
class Recipe:
    name: str
    model_id: str
    text_class: type
    refine_class: type
    steps: int
    cfg: float
    opening: float
    counts: tuple[int, int, int]
    sigmas: tuple[float, float, float]
    guidance_passes: int

    def count(self, strength=None):
        return self.counts[STRENGTHS.index(self.opening if strength is None else strength)]

    def sigma(self, strength):
        return self.sigmas[STRENGTHS.index(strength)]

    def forwards(self, strength=None):
        return self.count(strength) * self.guidance_passes


# Fixed expectations from the three independently measured card recipes, not
# read back from the production profile/envelope that these tests exercise.
RECIPES = (
    Recipe("turbo", "circlestone-labs/Anima", AnimaTurboPipeline, AnimaTurboImg2ImgPipeline,
           8, 1., .25, (1, 2, 4), (.3, .5, .75), 1),
    Recipe("aesthetic", "circlestone-labs/Anima@aesthetic-v1.1", AnimaAestheticPipeline,
           AnimaAestheticImg2ImgPipeline, 30, 4., .125, (4, 8, 12),
           (.315789521, .521739185, .666666687), 2),
    Recipe("preview", "Gazingstars123/Anima-2.9B", AnimaPreviewPipeline,
           AnimaPreviewImg2ImgPipeline, 30, 4., .125, (4, 8, 12),
           (.315789521, .521739185, .666666687), 2),
)


class NegativeAwareTokenizer:
    """Keep empty and explicit negatives observably distinct on the CPU."""
    def __init__(self):
        self.prompts = []

    def __call__(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        values = [[2 if prompt == "sunny garden" else 8 if prompt == "rain" else 5]
                  for prompt in prompts]
        ids = torch.tensor(values, dtype=torch.long, device="cpu")
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


@pytest.fixture(params=RECIPES, ids=lambda recipe: recipe.name)
def anima(request, store, monkeypatch):
    recipe = request.param
    selected = dataclasses.replace(
        store.settings, model_id=recipe.model_id, quantization="fp8", offload="model",
        device="cuda", loader_execution=None, loader_execution_error=None,
    )
    for module in (config, diffusers_backend, schemas_generate):
        monkeypatch.setattr(module, "settings", selected)
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    # The CPU execution device and supplied qualification facts are distinct:
    # no test asks the workstation's driver/card whether it may run.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pipeline = text_pipe(recipe.text_class)
    pipeline.tokenizer = NegativeAwareTokenizer()
    pipeline.t5_tokenizer = NegativeAwareTokenizer()
    pipeline.enable_model_cpu_offload(device="cpu")
    backend = loaded_backend(pipeline)
    monkeypatch.setattr(backend, "_peak_device", lambda: None)
    monkeypatch.setattr(backend, "refine_host", lambda: HOST)
    backend.set_console_progress(False)
    source = Image.new("RGB", (1024, 1024), (190, 40, 80))
    source.paste((10, 210, 40), (0, 0, 256, 512))
    record = store.save_image(
        source, GenerateRequest(prompt="sunny garden", width=1024, height=1024,
                                steps=recipe.steps, guidance_scale=recipe.cfg),
        seed=SEED, duration_ms=1, model_id=recipe.model_id, quantization="fp8", run_id="cpu-source",
    )
    cleanups = []
    free = pipeline.maybe_free_model_hooks

    def release_owner():
        cleanups.append((pipeline.vae.decode_calls, len(pipeline.transformer.calls)))
        free()

    monkeypatch.setattr(pipeline, "maybe_free_model_hooks", release_owner)
    yield SimpleNamespace(backend=backend, pipeline=pipeline, source=source, record=record,
                          cleanups=cleanups, store=store, recipe=recipe)
    pipeline.remove_all_hooks()


def request(anima, **changes):
    return GenerateRequest(**{
        "prompt": "sunny garden", "negative_prompt": "rain" if anima.recipe.cfg == 1 else "",
        "width": 1024, "height": 1024,
        "steps": anima.recipe.steps, "guidance_scale": anima.recipe.cfg, "refine": anima.record.id,
        "strength": anima.recipe.opening, "num_images": 1, **changes,
    })


def private(anima, *, strength=None, seed=SEED, negative_prompt=""):
    strength = anima.recipe.opening if strength is None else strength
    runner = refine.build(anima.pipeline, anima.recipe.refine_class)
    runner.set_progress_bar_config(disable=True)
    try:
        return runner(
            prompt="sunny garden", image=anima.source.copy(), strength=strength,
            width=1024, height=1024, num_inference_steps=anima.recipe.steps,
            guidance_scale=anima.recipe.cfg, negative_prompt=negative_prompt,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
    finally:
        anima.pipeline.maybe_free_model_hooks()


@pytest.mark.parametrize("strength", STRENGTHS)
def test_public_backend_matches_private_pass_with_real_source_and_actual_progress(anima, strength):
    count = anima.recipe.count(strength)
    expected = private(anima, strength=strength)
    anima.cleanups.clear()
    before_hooks = list(anima.pipeline._all_hooks)
    progress = []
    raw = request(anima, strength=strength)
    normalized = anima.backend.applied(raw)
    assert normalized.refine == raw.refine and normalized.strength == strength
    assert normalized.negative_prompt in (None, "") and normalized.mu is None
    anima.pipeline.tokenizer.prompts.clear()
    anima.pipeline.t5_tokenizer.prompts.clear()
    image, = anima.backend.generate(raw, SEED, lambda done, total: progress.append((done, total)))
    assert image.mode == "RGB" and image.size == (1024, 1024)
    # Reading pixels catches premature closure of the same-size source image.
    assert image.tobytes() == expected.tobytes()
    assert progress == [(step, count) for step in range(1, count + 1)]
    prompts = ["sunny garden"] if anima.recipe.cfg == 1 else ["sunny garden", ""]
    assert anima.pipeline.tokenizer.prompts == prompts
    assert anima.pipeline.t5_tokenizer.prompts == prompts
    assert len(anima.cleanups) == 1
    assert anima.pipeline._all_hooks and list(anima.pipeline._all_hooks) != before_hooks
    assert all(parameter.device.type == "cpu" for component in anima.pipeline.components.values()
               if isinstance(component, torch.nn.Module) for parameter in component.parameters())


def test_two_image_backend_batch_has_one_owner_cleanup_per_image_and_reproduces_solo(anima):
    progress = []
    images = anima.backend.generate(request(anima, strength=.4, num_images=2), SEED,
                                    lambda done, total: progress.append((done, total)))
    assert len(images) == 2 and len(anima.cleanups) == 2
    total = 2 * anima.recipe.count(.4)
    assert progress == [(step, total) for step in range(1, total + 1)]
    assert images[0].tobytes() != images[1].tobytes()
    for offset, image in enumerate(images):
        solo, = anima.backend.generate(request(anima, strength=.4), SEED + offset)
        assert image.tobytes() == solo.tobytes()
        assert image.tobytes() == private(anima, strength=.4, seed=SEED + offset).tobytes()


def test_real_backend_source_pixels_affect_the_result(anima):
    red, = anima.backend.generate(request(anima), SEED)
    blue_source = Image.new("RGB", (1024, 1024), (20, 40, 220))
    blue = anima.store.save_image(
        blue_source, GenerateRequest(prompt="sunny garden", width=1024, height=1024,
                                     steps=anima.recipe.steps, guidance_scale=anima.recipe.cfg),
        seed=SEED, duration_ms=1, model_id=anima.recipe.model_id, quantization="fp8",
    )
    changed, = anima.backend.generate(request(anima, refine=blue.id), SEED)
    assert red.tobytes() != changed.tobytes()
    assert (anima.store.settings.outputs_dir / anima.record.filename).is_file()


def test_cancel_frees_loaded_owner_without_decode_then_recovery_reproduces(anima):
    expected = private(anima, strength=.4)
    anima.cleanups.clear()
    before_decode = anima.pipeline.vae.decode_calls
    before_hooks = list(anima.pipeline._all_hooks)
    progress = []
    with pytest.raises(Cancelled):
        anima.backend.generate(
            request(anima, strength=.4), SEED,
            lambda done, total: progress.append((done, total)), lambda: bool(progress),
        )
    assert progress == [(1, anima.recipe.count(.4))]
    assert anima.pipeline.vae.decode_calls == before_decode
    assert len(anima.cleanups) == 1
    assert anima.pipeline._all_hooks and list(anima.pipeline._all_hooks) != before_hooks
    assert anima.backend._refine_pipe._step.callback is None
    recovered, = anima.backend.generate(request(anima, strength=.4), SEED)
    assert recovered.tobytes() == expected.tobytes()
    assert len(anima.cleanups) == 2


def test_forward_error_frees_loaded_owner_and_does_not_poison_next_pass(anima, monkeypatch):
    expected = private(anima)
    anima.cleanups.clear()
    before_decode = anima.pipeline.vae.decode_calls
    original = anima.pipeline.transformer.forward

    def fail(*args, **kwargs):
        raise RuntimeError("injected denoiser failure")

    monkeypatch.setattr(anima.pipeline.transformer, "forward", fail)
    with pytest.raises(RuntimeError, match="injected denoiser failure"):
        anima.backend.generate(request(anima), SEED)
    assert len(anima.cleanups) == 1 and anima.pipeline.vae.decode_calls == before_decode
    assert anima.backend._refine_pipe._step.callback is None
    monkeypatch.setattr(anima.pipeline.transformer, "forward", original)
    recovered, = anima.backend.generate(request(anima), SEED)
    assert recovered.tobytes() == expected.tobytes() and len(anima.cleanups) == 2


def test_direct_generation_keeps_pixels_after_success_and_cancelled_refine(anima):
    direct = GenerateRequest(prompt="sunny garden", width=1024, height=1024,
                             steps=anima.recipe.steps, guidance_scale=anima.recipe.cfg)
    before, = anima.backend.generate(direct, SEED)
    anima.backend.generate(request(anima), SEED)
    progress = []
    with pytest.raises(Cancelled):
        anima.backend.generate(request(anima, strength=.4), SEED,
                               lambda done, total: progress.append(done), lambda: bool(progress))
    after, = anima.backend.generate(direct, SEED)
    assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize("strength", STRENGTHS)
def test_backend_start_sigma_reads_scheduler_copy_without_mutating_loaded_state(anima, strength):
    scheduler = anima.pipeline.scheduler
    scheduler.set_timesteps(5)
    before = scheduler.timesteps.clone(), scheduler.sigmas.clone(), scheduler.begin_index, scheduler.shift
    assert anima.backend.start_sigma(anima.backend.applied(request(anima, strength=strength))) == pytest.approx(
        anima.recipe.sigma(strength), abs=1e-7,
    )
    assert torch.equal(scheduler.timesteps, before[0]) and torch.equal(scheduler.sigmas, before[1])
    assert (scheduler.begin_index, scheduler.shift) == before[2:]


@pytest.mark.parametrize("host, words", [
    (("win32", "0.42.1", None, True), "GPU has not been verified"),
    (("win32", "0.42.1", "NVIDIA GeForce RTX 3070", True), "only the RTX 5080"),
    (("win32", "0.42.1", "NVIDIA GeForce RTX 5080", False), "external attention override"),
    (("linux", "0.42.1", "NVIDIA GeForce RTX 5080", True), "only Windows"),
    (("win32", "0.42.2", "NVIDIA GeForce RTX 5080", True), "only 0.42.1"),
])
def test_unqualified_host_refuses_before_source_decode_or_counterpart_construction(anima, monkeypatch, host, words):
    monkeypatch.setattr(anima.backend, "refine_host", lambda: host)

    def no_source(*args, **kwargs):
        pytest.fail("a held request must not decode source pixels")

    monkeypatch.setattr(refine_admission, "load_source", no_source)
    with pytest.raises(ValueError, match=words):
        anima.backend.generate(request(anima), SEED)
    assert anima.backend._refine_pipe is None
    assert anima.pipeline.vae.encoded == [] and anima.pipeline.vae.decode_calls == 0
    assert anima.pipeline.transformer.calls == [] and anima.cleanups == []


@pytest.mark.parametrize("changes, words", [
    ({"width": 1280}, "only at 1024"), ({"steps": 12}, "only at {steps} steps"),
    ({"guidance_scale": 2}, "only at CFG {cfg:g}"), ({"strength": .3}, "only at strengths"),
    ({"mu": 1.0}, "without LoRAs, references or a Mu override"),
    ({"crop_anchor": "top"}, "centre crop anchor"),
])
def test_copied_unmeasured_request_is_refused_before_source_or_model_work(anima, monkeypatch, changes, words):
    raw = request(anima).model_copy(update=changes)

    def no_source(*args, **kwargs):
        pytest.fail("an unmeasured copied request must not decode its source")

    monkeypatch.setattr(refine_admission, "load_source", no_source)
    with pytest.raises(ValueError, match=words.format(steps=anima.recipe.steps, cfg=anima.recipe.cfg)):
        anima.backend.generate(raw, SEED)
    assert anima.backend._refine_pipe is None
    assert anima.pipeline.vae.encoded == [] and anima.pipeline.transformer.calls == []


@pytest.mark.parametrize("changes, words", [
    ({"quantization": "int8"}, "only fp8"),
    ({"offload": "sequential"}, "only model offload"),
    ({"device": "cpu"}, "only CUDA device 0"),
])
def test_unmeasured_loader_refuses_before_decoding_or_model_work(anima, monkeypatch, changes, words):
    # The base admission and concrete loader read one selected settings
    # snapshot in the app; keep their test bindings consistent too.
    changed = dataclasses.replace(config.settings, **changes)
    for module in (config, diffusers_backend, schemas_generate):
        monkeypatch.setattr(module, "settings", changed)
    with pytest.raises(ValueError, match=words):
        anima.backend.generate(request(anima), SEED)
    assert anima.backend._refine_pipe is None
    assert anima.pipeline.vae.encoded == [] and anima.pipeline.transformer.calls == []


def test_changed_source_pixels_are_refused_before_vae_encoding(anima):
    Image.new("RGB", (1024, 1024), (1, 2, 3)).save(anima.store.settings.outputs_dir / anima.record.filename)
    with pytest.raises(ValueError, match="pixels no longer match"):
        anima.backend.generate(request(anima), SEED)
    assert anima.pipeline.vae.encoded == [] and anima.pipeline.vae.decode_calls == 0
    assert anima.pipeline.transformer.calls == []


def observe_native_policy(anima, monkeypatch):
    """Supply measured platform/GPU facts but use the real Native observer."""
    monkeypatch.setattr(config, "ATTENTION_ENVIRONMENT_WARNING", None)
    monkeypatch.delenv("DIFFUSERS_ATTN_BACKEND", raising=False)
    native, other = object(), object()

    def native_function(*args, **kwargs):
        raise AssertionError("the tiny CPU denoiser never executes an attention kernel")

    state = SimpleNamespace(active=(native, native_function))
    constants = ModuleType("diffusers.utils.constants")
    constants.DIFFUSERS_ATTN_BACKEND = "native"
    dispatch = ModuleType("diffusers.models.attention_dispatch")
    dispatch.DIFFUSERS_ATTN_BACKEND = "native"
    dispatch.AttentionBackendName = SimpleNamespace(NATIVE=native)
    dispatch._native_attention = native_function
    dispatch._AttentionBackendRegistry = SimpleNamespace(get_active_backend=lambda: state.active)
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setitem(sys.modules, dispatch.__name__, dispatch)
    monkeypatch.setattr(anima.backend, "refine_host", lambda: (*HOST[:3], _refine_native_observed()))
    assert _refine_native_observed() is True
    return state, native, other, native_function


def test_policy_changed_by_forward_discards_all_batch_output_and_frees_owner(anima, monkeypatch):
    state, _, other, function = observe_native_policy(anima, monkeypatch)
    original = anima.pipeline.transformer.forward

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        state.active = (other, function)
        return result

    monkeypatch.setattr(anima.pipeline.transformer, "forward", drift)
    returned = []
    with pytest.raises(ValueError, match="external attention override"):
        returned.extend(anima.backend.generate(request(anima, num_images=2), SEED))
    assert returned == []
    assert len(anima.pipeline.transformer.calls) == anima.recipe.forwards()  # first image only; no retry
    assert anima.pipeline.vae.decode_calls == 1
    assert len(anima.cleanups) == 1 and anima.pipeline._all_hooks
    assert anima.backend._refine_pipe._step.callback is None


def test_policy_changed_between_batch_calls_is_refused_before_the_next_forward(anima, monkeypatch):
    state, _, other, function = observe_native_policy(anima, monkeypatch)
    release = anima.pipeline.maybe_free_model_hooks

    def drift_after_first_cleanup():
        release()
        if len(anima.cleanups) == 1:
            state.active = (other, function)

    monkeypatch.setattr(anima.pipeline, "maybe_free_model_hooks", drift_after_first_cleanup)
    returned = []
    with pytest.raises(ValueError, match="external attention override"):
        returned.extend(anima.backend.generate(request(anima, num_images=2), SEED))
    assert returned == [] and anima.pipeline.vae.decode_calls == 1
    assert len(anima.pipeline.transformer.calls) == anima.recipe.forwards()
    # Both the completed call and the next refused call release the owner.
    assert len(anima.cleanups) == 2 and anima.pipeline._all_hooks


@pytest.mark.parametrize("cancel", [False, True])
def test_native_drift_keeps_primary_forward_error_or_cancellation(anima, monkeypatch, cancel):
    state, native, other, function = observe_native_policy(anima, monkeypatch)
    expected = private(anima)
    anima.cleanups.clear()
    before_decode = anima.pipeline.vae.decode_calls
    before_calls = len(anima.pipeline.transformer.calls)
    original = anima.pipeline.transformer.forward

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        state.active = (other, function)
        if not cancel:
            raise RuntimeError("primary denoiser failure")
        return result

    monkeypatch.setattr(anima.pipeline.transformer, "forward", drift)
    progress = []
    expected_error = Cancelled if cancel else RuntimeError
    with pytest.raises(expected_error, match=None if cancel else "primary denoiser failure"):
        anima.backend.generate(request(anima), SEED,
                               lambda done, total: progress.append(done), lambda: bool(progress))
    assert anima.pipeline.vae.decode_calls == before_decode
    assert len(anima.pipeline.transformer.calls) == before_calls + (anima.recipe.guidance_passes if cancel else 1)
    assert len(anima.cleanups) == 1 and anima.pipeline._all_hooks
    assert anima.backend._refine_pipe._step.callback is None
    # Only the test restores the observed foreign state; the backend must not
    # overwrite someone else's environment or registry to manufacture a pass.
    assert state.active == (other, function)
    state.active = (native, function)
    monkeypatch.setattr(anima.pipeline.transformer, "forward", original)
    recovered, = anima.backend.generate(request(anima), SEED)
    assert recovered.tobytes() == expected.tobytes() and len(anima.cleanups) == 2


def test_successful_load_refreshes_unknown_cached_hardware_after_preparation(anima, monkeypatch):
    observe_native_policy(anima, monkeypatch)
    monkeypatch.setattr(base, "sys", SimpleNamespace(platform="win32", modules=sys.modules))
    monkeypatch.setattr(base, "installed_diffusers_version", lambda: "0.42.1")
    monkeypatch.setattr(card_facts, "_OBSERVED", {"cuda": None})
    cached_reads = []

    @functools.cache
    def cached_card(device="cuda"):
        cached_reads.append(device)
        return None

    monkeypatch.setattr(card_facts, "card", cached_card)
    assert card_facts.card("cuda") is None
    backend = diffusers_backend.DiffusersBackend()
    assert backend.refine_host() == ("win32", "0.42.1", None, True)
    assert "GPU has not been verified" in backend.refine_refusal(config.settings)
    events = []
    measured = card_facts.Card(16 * 1024**3, (12, 0), "NVIDIA GeForce RTX 5080")

    def prepare(execution):
        assert execution is None
        events.append("prepare")
        backend._pipe = anima.pipeline  # supplied CPU components, never a load

    def fresh_probe(device, torch=None):
        assert device == "cuda" and torch is None
        assert events == ["prepare"] and backend.is_ready()
        events.append("probe")
        card_facts._OBSERVED[device] = measured
        return measured

    monkeypatch.setattr(backend, "_load_prepared", prepare)
    monkeypatch.setattr(card_facts, "probe", fresh_probe)
    backend.load()
    assert events == ["prepare", "probe"]
    # The old cache can remain unknown; qualification reads actual observed
    # evidence refreshed by the load-owned uncached probe instead.
    assert card_facts.card("cuda") is None and cached_reads == ["cuda"]
    assert card_facts.observed("cuda") is measured
    assert backend.refine_host() == HOST
    assert backend.refine_refusal(config.settings) is None


def test_failed_load_never_attempts_hardware_probe_or_publishes_facts(anima, monkeypatch):
    observe_native_policy(anima, monkeypatch)
    monkeypatch.setattr(base, "sys", SimpleNamespace(platform="win32", modules=sys.modules))
    monkeypatch.setattr(base, "installed_diffusers_version", lambda: "0.42.1")
    monkeypatch.setattr(card_facts, "_OBSERVED", {"cuda": None})
    backend = diffusers_backend.DiffusersBackend()
    events = []

    def fail_prepare(execution):
        assert execution is None
        events.append("prepare")
        raise RuntimeError("preparation refused before publication")

    def forbidden_probe(*args, **kwargs):
        pytest.fail("failed preparation must not probe or claim a known GPU")

    monkeypatch.setattr(backend, "_load_prepared", fail_prepare)
    monkeypatch.setattr(card_facts, "probe", forbidden_probe)
    with pytest.raises(RuntimeError, match="preparation refused before publication"):
        backend.load()
    assert events == ["prepare"] and not backend.is_ready()
    assert card_facts.observed("cuda") is None
    assert backend.refine_host() == ("win32", "0.42.1", None, True)


def test_explicit_negative_is_conditioned_reproduced_and_retained_at_cfg_four(anima):
    empty, = anima.backend.generate(request(anima, negative_prompt=""), SEED)
    raw = request(anima, negative_prompt="rain")
    normalized = anima.backend.applied(raw)
    expected_negative = "rain" if anima.recipe.cfg == 4 else None
    assert normalized.negative_prompt == expected_negative
    anima.pipeline.tokenizer.prompts.clear()
    anima.pipeline.t5_tokenizer.prompts.clear()
    explicit, = anima.backend.generate(raw, SEED)
    encoded = ["sunny garden", "rain"] if anima.recipe.cfg == 4 else ["sunny garden"]
    assert anima.pipeline.tokenizer.prompts == encoded
    assert anima.pipeline.t5_tokenizer.prompts == encoded
    assert (empty.tobytes() != explicit.tobytes()) is (anima.recipe.cfg == 4)
    again, = anima.backend.generate(raw, SEED)
    assert explicit.tobytes() == again.tobytes()
    assert explicit.tobytes() == private(anima, negative_prompt="rain").tobytes()
    saved = anima.store.save_image(
        explicit, normalized, seed=SEED, duration_ms=1,
        model_id=anima.recipe.model_id, quantization="fp8",
    )
    assert saved.negative_prompt == expected_negative
    with Image.open(anima.store.settings.outputs_dir / saved.filename) as handle:
        metadata = json.loads(handle.info["auradream"])
    assert metadata["negative_prompt"] == expected_negative
    assert metadata["refine"] == anima.record.id
