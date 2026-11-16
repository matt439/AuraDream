"""The public backend's Anima Refine orchestration with real CPU blocks.

Only the learned components/tokenizers are tiny stand-ins. These checks use
the installed encoder, schedule, denoising and decode blocks plus the real
gallery-source reader, backend dispatcher and CPU offload hooks. They make
no claim about a checkpoint's GPU pixels, quality, fit or speed.
"""
from __future__ import annotations

import dataclasses
import functools
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
import torch

from backend import card_facts, config, model_selection, refine, refine_admission, schemas_generate
from backend.backends import Cancelled, base, diffusers_backend
from backend.backends.base import _refine_native_observed
from backend.backends.anima_pipeline import AnimaTurboPipeline
from backend.backends.anima_refine import AnimaTurboImg2ImgPipeline
from backend.schemas import GenerateRequest
from tests.backend_helpers import loaded_backend
from tests.test_anima_refine import _text_pipe as text_pipe


TURBO = "circlestone-labs/Anima"
HOST = ("win32", "0.42.1", "NVIDIA GeForce RTX 5080", True)
SEED = 4471


@pytest.fixture
def turbo(store, monkeypatch):
    selected = dataclasses.replace(
        store.settings, model_id=TURBO, quantization="fp8", offload="model",
        device="cuda", loader_execution=None, loader_execution_error=None,
    )
    for module in (config, diffusers_backend, schemas_generate):
        monkeypatch.setattr(module, "settings", selected)
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    # The CPU execution device and supplied qualification facts are distinct:
    # no test asks the workstation's driver/card whether it may run.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pipeline = text_pipe(AnimaTurboPipeline)
    pipeline.enable_model_cpu_offload(device="cpu")
    backend = loaded_backend(pipeline)
    monkeypatch.setattr(backend, "_peak_device", lambda: None)
    monkeypatch.setattr(backend, "refine_host", lambda: HOST)
    backend.set_console_progress(False)
    source = Image.new("RGB", (1024, 1024), (190, 40, 80))
    source.paste((10, 210, 40), (0, 0, 256, 512))
    record = store.save_image(
        source, GenerateRequest(prompt="sunny garden", width=1024, height=1024,
                                steps=8, guidance_scale=1),
        seed=SEED, duration_ms=1, model_id=TURBO, quantization="fp8", run_id="cpu-source",
    )
    cleanups = []
    free = pipeline.maybe_free_model_hooks

    def release_owner():
        cleanups.append((pipeline.vae.decode_calls, len(pipeline.transformer.calls)))
        free()

    monkeypatch.setattr(pipeline, "maybe_free_model_hooks", release_owner)
    yield SimpleNamespace(backend=backend, pipeline=pipeline, source=source, record=record,
                          cleanups=cleanups, store=store)
    pipeline.remove_all_hooks()


def request(turbo, **changes):
    return GenerateRequest(**{
        "prompt": "sunny garden", "negative_prompt": "rain", "width": 1024, "height": 1024,
        "steps": 8, "guidance_scale": 1.0, "refine": turbo.record.id,
        "strength": .25, "num_images": 1, **changes,
    })


def private(turbo, *, strength=.25, seed=SEED):
    runner = refine.build(turbo.pipeline, AnimaTurboImg2ImgPipeline)
    runner.set_progress_bar_config(disable=True)
    try:
        return runner(
            prompt="sunny garden", image=turbo.source.copy(), strength=strength,
            width=1024, height=1024, num_inference_steps=8, guidance_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
    finally:
        turbo.pipeline.maybe_free_model_hooks()


@pytest.mark.parametrize("strength, count", [(.125, 1), (.25, 2), (.4, 4)])
def test_public_backend_matches_private_pass_with_real_source_and_actual_progress(turbo, strength, count):
    expected = private(turbo, strength=strength)
    turbo.cleanups.clear()
    before_hooks = list(turbo.pipeline._all_hooks)
    progress = []
    raw = request(turbo, strength=strength)
    normalized = turbo.backend.applied(raw)
    assert normalized.refine == raw.refine and normalized.strength == strength
    assert normalized.negative_prompt is None and normalized.mu is None
    image, = turbo.backend.generate(raw, SEED, lambda done, total: progress.append((done, total)))
    assert image.mode == "RGB" and image.size == (1024, 1024)
    # Reading pixels catches premature closure of the same-size source image.
    assert image.tobytes() == expected.tobytes()
    assert progress == [(step, count) for step in range(1, count + 1)]
    assert len(turbo.cleanups) == 1
    assert turbo.pipeline._all_hooks and list(turbo.pipeline._all_hooks) != before_hooks
    assert all(parameter.device.type == "cpu" for component in turbo.pipeline.components.values()
               if isinstance(component, torch.nn.Module) for parameter in component.parameters())


def test_two_image_backend_batch_has_one_owner_cleanup_per_image_and_reproduces_solo(turbo):
    progress = []
    images = turbo.backend.generate(request(turbo, strength=.4, num_images=2), SEED,
                                    lambda done, total: progress.append((done, total)))
    assert len(images) == 2 and len(turbo.cleanups) == 2
    assert progress == [(step, 8) for step in range(1, 9)]
    assert images[0].tobytes() != images[1].tobytes()
    for offset, image in enumerate(images):
        solo, = turbo.backend.generate(request(turbo, strength=.4), SEED + offset)
        assert image.tobytes() == solo.tobytes()
        assert image.tobytes() == private(turbo, strength=.4, seed=SEED + offset).tobytes()


def test_real_backend_source_pixels_affect_the_result(turbo):
    red, = turbo.backend.generate(request(turbo), SEED)
    blue_source = Image.new("RGB", (1024, 1024), (20, 40, 220))
    blue = turbo.store.save_image(
        blue_source, GenerateRequest(prompt="sunny garden", width=1024, height=1024,
                                     steps=8, guidance_scale=1),
        seed=SEED, duration_ms=1, model_id=TURBO, quantization="fp8",
    )
    changed, = turbo.backend.generate(request(turbo, refine=blue.id), SEED)
    assert red.tobytes() != changed.tobytes()
    assert (turbo.store.settings.outputs_dir / turbo.record.filename).is_file()


def test_cancel_frees_loaded_owner_without_decode_then_recovery_reproduces(turbo):
    expected = private(turbo, strength=.4)
    turbo.cleanups.clear()
    before_decode = turbo.pipeline.vae.decode_calls
    before_hooks = list(turbo.pipeline._all_hooks)
    progress = []
    with pytest.raises(Cancelled):
        turbo.backend.generate(
            request(turbo, strength=.4), SEED,
            lambda done, total: progress.append((done, total)), lambda: bool(progress),
        )
    assert progress == [(1, 4)]
    assert turbo.pipeline.vae.decode_calls == before_decode
    assert len(turbo.cleanups) == 1
    assert turbo.pipeline._all_hooks and list(turbo.pipeline._all_hooks) != before_hooks
    assert turbo.backend._refine_pipe._step.callback is None
    recovered, = turbo.backend.generate(request(turbo, strength=.4), SEED)
    assert recovered.tobytes() == expected.tobytes()
    assert len(turbo.cleanups) == 2


def test_forward_error_frees_loaded_owner_and_does_not_poison_next_pass(turbo, monkeypatch):
    expected = private(turbo)
    turbo.cleanups.clear()
    before_decode = turbo.pipeline.vae.decode_calls
    original = turbo.pipeline.transformer.forward

    def fail(*args, **kwargs):
        raise RuntimeError("injected denoiser failure")

    monkeypatch.setattr(turbo.pipeline.transformer, "forward", fail)
    with pytest.raises(RuntimeError, match="injected denoiser failure"):
        turbo.backend.generate(request(turbo), SEED)
    assert len(turbo.cleanups) == 1 and turbo.pipeline.vae.decode_calls == before_decode
    assert turbo.backend._refine_pipe._step.callback is None
    monkeypatch.setattr(turbo.pipeline.transformer, "forward", original)
    recovered, = turbo.backend.generate(request(turbo), SEED)
    assert recovered.tobytes() == expected.tobytes() and len(turbo.cleanups) == 2


def test_direct_generation_keeps_pixels_after_success_and_cancelled_refine(turbo):
    direct = GenerateRequest(prompt="sunny garden", width=1024, height=1024, steps=8, guidance_scale=1)
    before, = turbo.backend.generate(direct, SEED)
    turbo.backend.generate(request(turbo), SEED)
    progress = []
    with pytest.raises(Cancelled):
        turbo.backend.generate(request(turbo, strength=.4), SEED,
                               lambda done, total: progress.append(done), lambda: bool(progress))
    after, = turbo.backend.generate(direct, SEED)
    assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize("strength, sigma", [(.125, .3), (.25, .5), (.4, .75)])
def test_backend_start_sigma_reads_scheduler_copy_without_mutating_loaded_state(turbo, strength, sigma):
    scheduler = turbo.pipeline.scheduler
    scheduler.set_timesteps(5)
    before = scheduler.timesteps.clone(), scheduler.sigmas.clone(), scheduler.begin_index, scheduler.shift
    assert turbo.backend.start_sigma(turbo.backend.applied(request(turbo, strength=strength))) == pytest.approx(sigma)
    assert torch.equal(scheduler.timesteps, before[0]) and torch.equal(scheduler.sigmas, before[1])
    assert (scheduler.begin_index, scheduler.shift) == before[2:]


@pytest.mark.parametrize("host, words", [
    (("win32", "0.42.1", None, True), "GPU has not been verified"),
    (("win32", "0.42.1", "NVIDIA GeForce RTX 3070", True), "only the RTX 5080"),
    (("win32", "0.42.1", "NVIDIA GeForce RTX 5080", False), "external attention override"),
    (("linux", "0.42.1", "NVIDIA GeForce RTX 5080", True), "only Windows"),
    (("win32", "0.42.2", "NVIDIA GeForce RTX 5080", True), "only 0.42.1"),
])
def test_unqualified_host_refuses_before_source_decode_or_counterpart_construction(turbo, monkeypatch, host, words):
    monkeypatch.setattr(turbo.backend, "refine_host", lambda: host)

    def no_source(*args, **kwargs):
        pytest.fail("a held request must not decode source pixels")

    monkeypatch.setattr(refine_admission, "load_source", no_source)
    with pytest.raises(ValueError, match=words):
        turbo.backend.generate(request(turbo), SEED)
    assert turbo.backend._refine_pipe is None
    assert turbo.pipeline.vae.encoded == [] and turbo.pipeline.vae.decode_calls == 0
    assert turbo.pipeline.transformer.calls == [] and turbo.cleanups == []


@pytest.mark.parametrize("changes, words", [
    ({"width": 1280}, "only at 1024"), ({"steps": 12}, "only at 8 steps"),
    ({"guidance_scale": 2}, "only at CFG 1"), ({"strength": .3}, "only at strengths"),
    ({"mu": 1.0}, "without LoRAs, references or a Mu override"),
    ({"crop_anchor": "top"}, "centre crop anchor"),
])
def test_copied_unmeasured_request_is_refused_before_source_or_model_work(turbo, monkeypatch, changes, words):
    raw = request(turbo).model_copy(update=changes)

    def no_source(*args, **kwargs):
        pytest.fail("an unmeasured copied request must not decode its source")

    monkeypatch.setattr(refine_admission, "load_source", no_source)
    with pytest.raises(ValueError, match=words):
        turbo.backend.generate(raw, SEED)
    assert turbo.backend._refine_pipe is None
    assert turbo.pipeline.vae.encoded == [] and turbo.pipeline.transformer.calls == []


@pytest.mark.parametrize("changes, words", [
    ({"quantization": "int8"}, "only fp8"),
    ({"offload": "sequential"}, "only model offload"),
    ({"device": "cpu"}, "only CUDA device 0"),
])
def test_unmeasured_loader_refuses_before_decoding_or_model_work(turbo, monkeypatch, changes, words):
    # The base admission and concrete loader read one selected settings
    # snapshot in the app; keep their test bindings consistent too.
    changed = dataclasses.replace(config.settings, **changes)
    for module in (config, diffusers_backend, schemas_generate):
        monkeypatch.setattr(module, "settings", changed)
    with pytest.raises(ValueError, match=words):
        turbo.backend.generate(request(turbo), SEED)
    assert turbo.backend._refine_pipe is None
    assert turbo.pipeline.vae.encoded == [] and turbo.pipeline.transformer.calls == []


def test_changed_source_pixels_are_refused_before_vae_encoding(turbo):
    Image.new("RGB", (1024, 1024), (1, 2, 3)).save(turbo.store.settings.outputs_dir / turbo.record.filename)
    with pytest.raises(ValueError, match="pixels no longer match"):
        turbo.backend.generate(request(turbo), SEED)
    assert turbo.pipeline.vae.encoded == [] and turbo.pipeline.vae.decode_calls == 0
    assert turbo.pipeline.transformer.calls == []


def observe_native_policy(turbo, monkeypatch):
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
    monkeypatch.setattr(turbo.backend, "refine_host", lambda: (*HOST[:3], _refine_native_observed()))
    assert _refine_native_observed() is True
    return state, native, other, native_function


def test_policy_changed_by_forward_discards_all_batch_output_and_frees_owner(turbo, monkeypatch):
    state, _, other, function = observe_native_policy(turbo, monkeypatch)
    original = turbo.pipeline.transformer.forward

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        state.active = (other, function)
        return result

    monkeypatch.setattr(turbo.pipeline.transformer, "forward", drift)
    returned = []
    with pytest.raises(ValueError, match="external attention override"):
        returned.extend(turbo.backend.generate(request(turbo, num_images=2), SEED))
    assert returned == []
    assert len(turbo.pipeline.transformer.calls) == 2  # first image only; no retry
    assert turbo.pipeline.vae.decode_calls == 1
    assert len(turbo.cleanups) == 1 and turbo.pipeline._all_hooks
    assert turbo.backend._refine_pipe._step.callback is None


def test_policy_changed_between_batch_calls_is_refused_before_the_next_forward(turbo, monkeypatch):
    state, _, other, function = observe_native_policy(turbo, monkeypatch)
    release = turbo.pipeline.maybe_free_model_hooks

    def drift_after_first_cleanup():
        release()
        if len(turbo.cleanups) == 1:
            state.active = (other, function)

    monkeypatch.setattr(turbo.pipeline, "maybe_free_model_hooks", drift_after_first_cleanup)
    returned = []
    with pytest.raises(ValueError, match="external attention override"):
        returned.extend(turbo.backend.generate(request(turbo, num_images=2), SEED))
    assert returned == [] and turbo.pipeline.vae.decode_calls == 1
    assert len(turbo.pipeline.transformer.calls) == 2
    # Both the completed call and the next refused call release the owner.
    assert len(turbo.cleanups) == 2 and turbo.pipeline._all_hooks


@pytest.mark.parametrize("cancel", [False, True])
def test_native_drift_keeps_primary_forward_error_or_cancellation(turbo, monkeypatch, cancel):
    state, native, other, function = observe_native_policy(turbo, monkeypatch)
    expected = private(turbo)
    turbo.cleanups.clear()
    before_decode = turbo.pipeline.vae.decode_calls
    before_calls = len(turbo.pipeline.transformer.calls)
    original = turbo.pipeline.transformer.forward

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        state.active = (other, function)
        if not cancel:
            raise RuntimeError("primary denoiser failure")
        return result

    monkeypatch.setattr(turbo.pipeline.transformer, "forward", drift)
    progress = []
    expected_error = Cancelled if cancel else RuntimeError
    with pytest.raises(expected_error, match=None if cancel else "primary denoiser failure"):
        turbo.backend.generate(request(turbo), SEED,
                               lambda done, total: progress.append(done), lambda: bool(progress))
    assert turbo.pipeline.vae.decode_calls == before_decode
    assert len(turbo.pipeline.transformer.calls) == before_calls + 1
    assert len(turbo.cleanups) == 1 and turbo.pipeline._all_hooks
    assert turbo.backend._refine_pipe._step.callback is None
    # Only the test restores the observed foreign state; the backend must not
    # overwrite someone else's environment or registry to manufacture a pass.
    assert state.active == (other, function)
    state.active = (native, function)
    monkeypatch.setattr(turbo.pipeline.transformer, "forward", original)
    recovered, = turbo.backend.generate(request(turbo), SEED)
    assert recovered.tobytes() == expected.tobytes() and len(turbo.cleanups) == 2


def test_successful_load_refreshes_unknown_cached_hardware_after_preparation(turbo, monkeypatch):
    observe_native_policy(turbo, monkeypatch)
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
        backend._pipe = turbo.pipeline  # supplied CPU components, never a load

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


def test_failed_load_never_attempts_hardware_probe_or_publishes_facts(turbo, monkeypatch):
    observe_native_policy(turbo, monkeypatch)
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
