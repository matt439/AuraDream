"""Supplied-fact Anima Refine admission, without frameworks or card work."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend import card_facts
from backend import refine_qualification as qualification


MODEL = "circlestone-labs/Anima"
AESTHETIC = "circlestone-labs/Anima@aesthetic-v1.1"
PREVIEW = "Gazingstars123/Anima-2.9B"
VARIANT_IDS = (MODEL, AESTHETIC, PREVIEW)
OTHER_MODELS = (
    "circlestone-labs/Anima@aesthetic-v1.2", "Gazingstars123/Anima-2.9B-preview-v2",
    "circlestone-labs/Anima-Base-v1.0-Diffusers", "cagliostrolab/animagine-xl-4.0",
    "black-forest-labs/FLUX.1-Krea-dev", "nobody/another-model",
)


def loader(**changes):
    values = dict(quantization="fp8", offload="model", device="cuda", platform="win32",
                  diffusers_version="0.42.1", gpu_name="NVIDIA GeForce RTX 5080",
                  native_attention=True, loader_execution=None)
    return qualification.LoaderFacts(**(values | changes))


def request(**changes):
    values = dict(refine=17, inpainting=False, edit=None, upscaler=None,
                  width=1024, height=1024, steps=8, guidance_scale=1., true_cfg=None,
                  strength=.25, lora=None, reference=[], mu=None, crop_anchor="center",
                  execution=None, negative_prompt="")
    return SimpleNamespace(**(values | changes))


def variant_request(model, **changes):
    envelope = qualification.envelope_for(model)
    return request(**(dict(steps=envelope.steps, guidance_scale=envelope.guidance_scale,
                          strength=envelope.default_strength) | changes))


def test_measured_envelope_is_explicit_and_frozen():
    envelope = qualification.ANIMA_TURBO
    assert dataclasses.is_dataclass(envelope)
    assert (envelope.model_id, envelope.gpu_name, envelope.platform) == (
        MODEL, "NVIDIA GeForce RTX 5080", "win32",
    )
    assert (envelope.diffusers_version, envelope.quantization, envelope.offload) == ("0.42.1", "fp8", "model")
    assert envelope.devices == ("cuda", "cuda:0")
    assert (envelope.size, envelope.steps, envelope.guidance_scale) == ((1024, 1024), 8, 1.)
    assert envelope.strengths == (.125, .25, .4) and envelope.default_strength == .25
    with pytest.raises(dataclasses.FrozenInstanceError):
        envelope.steps = 30


@pytest.mark.parametrize("model, attribute, steps, cfg, opening", [
    (MODEL, "ANIMA_TURBO", 8, 1., .25),
    (AESTHETIC, "ANIMA_AESTHETIC", 30, 4., .125),
    (PREVIEW, "ANIMA_PREVIEW", 30, 4., .125),
])
def test_each_variant_has_its_own_frozen_exact_measured_recipe(model, attribute, steps, cfg, opening):
    envelope = qualification.envelope_for(model)
    assert envelope is getattr(qualification, attribute)
    assert envelope.model_id == model and envelope.steps == steps
    assert envelope.guidance_scale == cfg and envelope.default_strength == opening
    assert envelope.size == (1024, 1024) and envelope.strengths == (.125, .25, .4)
    assert envelope.platform == "win32" and envelope.gpu_name == "NVIDIA GeForce RTX 5080"
    assert (envelope.quantization, envelope.offload, envelope.diffusers_version) == ("fp8", "model", "0.42.1")
    assert envelope.devices == ("cuda", "cuda:0")
    assert isinstance(envelope.empty_negative, bool)
    assert qualification.envelope_for(model.swapcase()) is envelope
    with pytest.raises(dataclasses.FrozenInstanceError):
        envelope.default_strength = .5


def test_the_variant_table_cannot_be_mutated_or_extended_by_a_caller():
    assert set(qualification.ENVELOPES) == {model.casefold() for model in VARIANT_IDS}
    with pytest.raises(TypeError):
        qualification.ENVELOPES["nobody/unmeasured"] = qualification.ANIMA_TURBO


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_only_the_measured_supplied_loader_is_admitted(model, device):
    assert qualification.loader_refusal(model, loader(device=device)) is None


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("execution", [None, {}, {"compilation": None, "attention_backend": None}])
def test_empty_loader_execution_keeps_eager_identity(model, execution):
    assert qualification.loader_refusal(model, loader(loader_execution=execution)) is None


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("changes, detail", [
    ({"quantization": "int8"}, "quantization"), ({"quantization": "none"}, "quantization"),
    ({"quantization": "nunchaku"}, "quantization"), ({"quantization": "FP8"}, "quantization"),
    ({"offload": "sequential"}, "offload"), ({"offload": "none"}, "offload"),
    ({"device": "cpu"}, "device"), ({"device": "mps"}, "device"),
    ({"device": "cuda:1"}, "device"), ({"device": "cuda:00"}, "device"),
    ({"platform": "linux"}, "Windows"), ({"platform": "darwin"}, "Windows"),
    ({"platform": "Windows"}, "Windows"),
    ({"diffusers_version": None}, "diffusers"), ({"diffusers_version": "0.40.0"}, "diffusers"),
    ({"diffusers_version": "0.42.2"}, "diffusers"), ({"diffusers_version": "0.42.1+local"}, "diffusers"),
    ({"gpu_name": None}, "not been verified"),
    ({"gpu_name": "NVIDIA GeForce RTX 3070"}, "GPU"),
    ({"gpu_name": "NVIDIA GeForce RTX 5090"}, "GPU"), ({"gpu_name": "RTX 5080"}, "GPU"),
    ({"native_attention": False}, "attention override"),
])
def test_each_unmeasured_supplied_loading_fact_holds(model, changes, detail):
    facts = loader(**changes)
    before = dataclasses.asdict(facts)
    reason = qualification.loader_refusal(model, facts)
    assert reason is not None and qualification.envelope_for(model).name in reason and detail in reason
    assert dataclasses.asdict(facts) == before


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("execution", [
    {"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"},
    {"step_cache": "fbc-005-v1"}, {"compilation": "future"}, {"extra": "value"},
    "regional-v1", [], 17,
])
def test_nonempty_or_unreadable_loader_execution_never_grants_admission(model, execution):
    reason = qualification.loader_refusal(model, loader(loader_execution=execution))
    assert reason is not None and "configuration" in reason


@pytest.mark.parametrize("canonical, model", [
    (canonical, alias) for canonical in VARIANT_IDS
    for alias in (canonical.lower(), canonical.upper(), canonical.swapcase())
])
def test_case_variants_cannot_escape_the_exact_id_requirement(canonical, model):
    assert qualification.applies(model)
    assert qualification.loader_refusal(model, loader()) == (
        f"{qualification.envelope_for(canonical).name} is held: select {canonical} exactly."
    )
    assert qualification.source_refusal(model, (1024, 768)) is not None
    assert qualification.request_refusal(model, variant_request(canonical, steps=qualification.envelope_for(canonical).steps + 1))


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("gpu_name", [None, "NVIDIA GeForce RTX 3070", "unverified"])
def test_potential_loader_check_skips_only_gpu_name(model, gpu_name):
    facts = loader(gpu_name=gpu_name)
    assert qualification.loader_refusal(model, facts) is not None
    assert qualification.loader_refusal(model, facts, check_hardware=False) is None
    for changes in ({"platform": "linux"}, {"device": "cpu"}, {"native_attention": False},
                    {"quantization": "none"}, {"offload": "sequential"},
                    {"diffusers_version": None}, {"loader_execution": {"attention_backend": "cudnn-v1"}}):
        assert qualification.loader_refusal(model, dataclasses.replace(facts, **changes), check_hardware=False)


def test_loader_facts_of_snapshots_selected_scalars_and_preserves_explicit_execution():
    selected = SimpleNamespace(quantization="fp8", offload="model", device="cuda:0",
                               loader_execution={"attention_backend": "cudnn-v1"})
    facts = qualification.LoaderFacts.of(selected, platform="win32", diffusers_version="0.42.1",
                                         gpu_name="NVIDIA GeForce RTX 5080", native_attention=True)
    selected.quantization = "none"
    assert facts.quantization == "fp8" and facts.device == "cuda:0"
    assert qualification.loader_refusal(MODEL, facts) is not None
    plain = SimpleNamespace(quantization="fp8", offload="model", device="cuda")
    assert qualification.LoaderFacts.of(plain, platform="win32", diffusers_version="0.42.1",
        gpu_name="NVIDIA GeForce RTX 5080", native_attention=True) == loader()


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("strength", [.125, .25, .4])
def test_each_measured_actual_pass_is_admitted_without_normalization(model, strength):
    actual = variant_request(model, strength=strength)
    before = vars(actual).copy()
    assert qualification.request_refusal(model, actual) is None
    assert vars(actual) == before
    assert qualification.source_refusal(model, (1024, 1024)) is None


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("execution", [None, {}, {"compilation": None, "attention_backend": None}])
def test_empty_per_render_execution_preserves_the_measured_eager_pass(model, execution):
    assert qualification.request_refusal(model, variant_request(model, execution=execution)) is None
    legacy = variant_request(model)
    del legacy.execution
    assert qualification.request_refusal(model, legacy) is None


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("execution", [
    {"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"},
    {"step_cache": "fbc-005-v1"}, {"step_cache": "future"}, {"unknown": "value"},
    "cudnn-v1", [], 17,
])
def test_nonempty_or_unreadable_per_render_execution_is_held_without_dropping_it(model, execution):
    actual = variant_request(model, execution=execution)
    reason = qualification.request_refusal(model, actual)
    assert reason is not None and qualification.envelope_for(model).name in reason
    assert actual.execution is execution


@pytest.mark.parametrize("changes, detail", [
    ({"inpainting": True}, "ordinary"), ({"edit": object()}, "ordinary"),
    ({"upscaler": "realesrgan-x4plus-anime6b"}, "ordinary"),
    ({"width": 512, "height": 512}, "output sizes"), ({"width": 1024, "height": 768}, "output sizes"),
    ({"width": 1280, "height": 1280}, "output sizes"), ({"width": 1536, "height": 1536}, "output sizes"),
    ({"steps": 0}, "8 steps"), ({"steps": 7}, "8 steps"), ({"steps": 9}, "8 steps"),
    ({"steps": 30}, "8 steps"), ({"steps": 8.0}, "8 steps"), ({"steps": True}, "8 steps"),
    ({"steps": None}, "8 steps"),
    ({"guidance_scale": 0.}, "CFG 1"), ({"guidance_scale": 4.}, "CFG 1"),
    ({"guidance_scale": float("nan")}, "CFG 1"), ({"guidance_scale": float("inf")}, "CFG 1"),
    ({"true_cfg": 0.}, "CFG 1"), ({"true_cfg": 1.}, "CFG 1"),
    ({"strength": 0.}, "strengths"), ({"strength": .15}, "strengths"), ({"strength": .3}, "strengths"),
    ({"strength": .5}, "strengths"), ({"strength": 1.}, "strengths"), ({"strength": None}, "strengths"),
    ({"strength": float("nan")}, "strengths"),
    ({"lora": "adapter"}, "LoRAs"), ({"lora": ""}, "LoRAs"),
    ({"reference": [3]}, "references"), ({"mu": 0.}, "Mu"), ({"mu": 1.}, "Mu"),
    ({"crop_anchor": "top"}, "crop anchor"), ({"crop_anchor": "bottom-right"}, "crop anchor"),
])
def test_each_unmeasured_shape_or_sampling_intent_holds_without_dropping_it(changes, detail):
    actual = request(**changes)
    before = vars(actual).copy()
    reason = qualification.request_refusal(MODEL, actual)
    assert reason is not None and detail in reason
    assert vars(actual) == before


@pytest.mark.parametrize("model", [AESTHETIC, PREVIEW])
@pytest.mark.parametrize("changes", [
    {"inpainting": True}, {"edit": object()}, {"upscaler": "realesrgan-x4plus-anime6b"},
    {"width": 512, "height": 512}, {"width": 1024, "height": 768},
    {"width": 1280, "height": 1280},
    {"steps": 8}, {"steps": 29}, {"steps": 31}, {"steps": 30.0}, {"steps": True}, {"steps": None},
    {"guidance_scale": 1.}, {"guidance_scale": 3.}, {"guidance_scale": 5.},
    {"guidance_scale": float("nan")}, {"guidance_scale": float("inf")},
    {"true_cfg": 0.}, {"true_cfg": 1.},
    {"strength": 0.}, {"strength": .15}, {"strength": .3}, {"strength": 1.},
    {"strength": None}, {"strength": float("nan")},
    {"lora": "adapter"}, {"lora": ""}, {"reference": [3]}, {"mu": 0.}, {"mu": 1.},
    {"crop_anchor": "top"}, {"crop_anchor": "bottom-right"},
])
def test_cfg_four_variants_keep_their_own_bounds_without_inheriting_turbo_sampling(model, changes):
    actual = variant_request(model, **changes)
    before = vars(actual).copy()
    reason = qualification.request_refusal(model, actual)
    assert reason is not None and qualification.envelope_for(model).name in reason
    assert vars(actual) == before


@pytest.mark.parametrize("model", [AESTHETIC, PREVIEW])
@pytest.mark.parametrize("negative", ["rain", " ", "no people"])
def test_cfg_four_negative_prompt_follows_its_measured_data_flag_and_is_never_dropped(model, negative):
    envelope = qualification.envelope_for(model)
    actual = variant_request(model, negative_prompt=negative)
    reason = qualification.request_refusal(model, actual)
    if envelope.empty_negative:
        assert reason is not None and "negative prompt" in reason
    else:
        assert reason is None
    assert actual.negative_prompt == negative


@pytest.mark.parametrize("model", [AESTHETIC, PREVIEW])
@pytest.mark.parametrize("empty_negative", [False, True])
def test_negative_prompt_guard_is_per_recipe_data_not_a_permanent_cfg_four_capability_loss(
    monkeypatch, model, empty_negative,
):
    envelope = dataclasses.replace(qualification.envelope_for(model), empty_negative=empty_negative)
    monkeypatch.setattr(qualification, "envelope_for", lambda _: envelope)
    actual = variant_request(model, negative_prompt="preserve these exclusion words")
    reason = qualification.request_refusal(model, actual)
    assert (reason is not None) is empty_negative
    assert actual.negative_prompt == "preserve these exclusion words"
    for negative in (None, ""):
        assert qualification.request_refusal(model, variant_request(model, negative_prompt=negative)) is None
    assert qualification.request_refusal(model, SimpleNamespace(refine=None, negative_prompt="rain")) is None


def test_turbo_ignored_negative_text_does_not_invent_an_active_negative_branch():
    actual = request(negative_prompt="rain")
    assert qualification.request_refusal(MODEL, actual) is None
    assert actual.negative_prompt == "rain"


@pytest.mark.parametrize("model", VARIANT_IDS)
@pytest.mark.parametrize("size", [(512, 512), (1280, 1280), (768, 1024), (1024, 768), (0, 1024)])
def test_the_original_source_has_its_own_exact_size_requirement(model, size):
    assert qualification.source_refusal(model, size) is not None


class UnreadableFacts:
    def __getattr__(self, name):
        raise AssertionError(f"unrelated request must not read {name}")


@pytest.mark.parametrize("model", OTHER_MODELS)
def test_other_checkpoint_families_keep_their_existing_admission(model):
    assert not qualification.applies(model)
    assert qualification.loader_refusal(model, UnreadableFacts()) is None
    assert qualification.request_refusal(model, UnreadableFacts()) is None
    assert qualification.source_refusal(model, (512, 768)) is None
    assert qualification.suggestions(model) is None


@pytest.mark.parametrize("model", VARIANT_IDS)
def test_direct_generation_does_not_read_or_normalize_refine_fields(model):
    assert qualification.request_refusal(model, SimpleNamespace(refine=None)) is None


@pytest.mark.parametrize("model", VARIANT_IDS)
def test_measured_strength_suggestions_are_fresh_copies_without_a_profile_guess(model):
    first = qualification.suggestions(model)
    second = qualification.suggestions(model)
    assert first == second
    assert first["steps"] == qualification.envelope_for(model).steps and first["strengths"] == [.125, .25, .4]
    first["strengths"].clear()
    assert second["strengths"] == [.125, .25, .4]
    assert qualification.suggestions(model)["strengths"] == [.125, .25, .4]


def fake_torch(*, name="NVIDIA GeForce RTX 5080", available=True):
    reads = []
    properties = SimpleNamespace(total_memory=16 * 1024**3, name=name)

    def device(value):
        reads.append(("device", value))
        kind, _, index = value.partition(":")
        return SimpleNamespace(type=kind, index=int(index) if index else None)

    def get_properties(index):
        reads.append(("properties", index))
        return properties

    cuda = SimpleNamespace(is_available=lambda: available, get_device_properties=get_properties,
                           get_device_capability=lambda index: (12, 0))
    return SimpleNamespace(device=device, version=SimpleNamespace(cuda="13.0"), cuda=cuda), reads


def test_observed_card_reads_only_previously_supplied_facts_and_keeps_device_scope(monkeypatch):
    monkeypatch.setattr(card_facts, "_OBSERVED", {})
    assert card_facts.observed("cuda") is None
    fake, reads = fake_torch()
    found = card_facts.probe("cuda", fake)
    assert found == card_facts.Card(16 * 1024**3, (12, 0), "NVIDIA GeForce RTX 5080")
    before = list(reads)

    def forbidden(*args, **kwargs):
        raise AssertionError("observed() must not probe or import a framework")

    monkeypatch.setattr(card_facts, "probe", forbidden)
    assert card_facts.observed("cuda") is found
    assert card_facts.observed("cuda:0") is None
    assert card_facts.observed("cuda:1") is None
    assert reads == before


def test_failed_reprobe_clears_stale_qualification_evidence(monkeypatch):
    monkeypatch.setattr(card_facts, "_OBSERVED", {})
    fake, _ = fake_torch()
    assert card_facts.probe("cuda", fake).name == "NVIDIA GeForce RTX 5080"
    fake.cuda.is_available = lambda: False
    assert card_facts.probe("cuda", fake) is None
    assert card_facts.observed("cuda") is None
    assert qualification.loader_refusal(MODEL, loader(gpu_name=None)) is not None


@pytest.mark.parametrize("name", [None, 5080])
def test_unnamed_or_legacy_card_facts_never_invent_a_measured_gpu_name(monkeypatch, name):
    monkeypatch.setattr(card_facts, "_OBSERVED", {})
    assert card_facts.Card(16 * 1024**3, (12, 0)).name is None
    fake, _ = fake_torch(name=name)
    found = card_facts.probe("cuda", fake)
    assert found.name is None and card_facts.observed("cuda") is found
    assert qualification.loader_refusal(MODEL, loader(gpu_name=found.name)) is not None


def test_module_import_and_supplied_fact_calls_stay_cold_in_an_isolated_process():
    script = r'''
import builtins, json, os, socket
from types import SimpleNamespace
attempts = {"framework": 0, "network": 0}
before = dict(os.environ)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {"torch", "diffusers", "transformers"}:
        attempts["framework"] += 1
        raise AssertionError("cold qualification imported " + name)
    return original(name, *args, **kwargs)
def network(*args, **kwargs):
    attempts["network"] += 1
    raise AssertionError("qualification attempted network work")
builtins.__import__ = guarded
socket.socket.connect = network
socket.socket.connect_ex = network
socket.create_connection = network
from backend import card_facts, refine_qualification as q
facts = q.LoaderFacts("fp8", "model", "cuda", "win32", "0.42.1", q.ANIMA_TURBO.gpu_name, True)
assert q.loader_refusal(q.ANIMA_TURBO.model_id, facts) is None
assert q.loader_refusal(q.ANIMA_TURBO.model_id, q.LoaderFacts("fp8", "model", "cuda", "win32", "0.42.1", None, True))
actual = SimpleNamespace(refine=1, inpainting=False, edit=None, upscaler=None, width=1024,
    height=1024, steps=8, guidance_scale=1., true_cfg=None, strength=.25, lora=None,
    reference=[], mu=None, crop_anchor="center", execution=None)
assert q.request_refusal(q.ANIMA_TURBO.model_id, actual) is None
assert q.source_refusal(q.ANIMA_TURBO.model_id, (1024, 1024)) is None
assert q.suggestions(q.ANIMA_TURBO.model_id)["strengths"] == [.125, .25, .4]
for envelope in q.ENVELOPES.values():
    actual.steps = envelope.steps
    actual.guidance_scale = envelope.guidance_scale
    actual.strength = envelope.default_strength
    actual.negative_prompt = ""
    assert q.envelope_for(envelope.model_id) is envelope
    assert q.loader_refusal(envelope.model_id, facts) is None
    assert q.request_refusal(envelope.model_id, actual) is None
    assert q.source_refusal(envelope.model_id, (1024, 1024)) is None
    assert q.suggestions(envelope.model_id)["steps"] == envelope.steps
card_facts.probe = network
assert card_facts.observed() is None
assert os.environ == before and not any(attempts.values())
print(json.dumps(attempts))
'''
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout) == {"framework": 0, "network": 0}
