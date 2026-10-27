"""Reserved execution recipes are explicit contracts, never silent eager runs."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from backend import execution, grid, jobs, submission
from backend.backends.base import GenerationBackend
from backend.backends.diffusers_backend import DiffusersBackend
from backend.metadata import PNG_METADATA_KEY
from backend.schemas import GenerateRequest, Outpaint
from tests.api_helpers import run_one


RECIPES = [
    {"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"},
    {"step_cache": "fbc-005-v1"},
    {"compilation": "regional-v1", "attention_backend": "cudnn-v1", "step_cache": "fbc-005-v1"},
]


@pytest.mark.parametrize("value", [None, {}, {"compilation": None}, execution.ExecutionSpec()])
def test_empty_execution_has_one_canonical_key(value):
    assert execution.canonical_execution(value) is None
    assert execution.execution_key(value) is None
    assert execution.unavailable_reason(value) is None


def test_execution_uses_named_sorted_fields_and_is_frozen():
    first = execution.ExecutionSpec(**RECIPES[-1])
    reverse = dict(reversed(list(RECIPES[-1].items())))
    assert execution.execution_key(first) == execution.execution_key(reverse) == (
        '{"attention_backend":"cudnn-v1","compilation":"regional-v1","step_cache":"fbc-005-v1"}'
    )
    with pytest.raises(ValidationError, match="frozen"):
        first.compilation = None


@pytest.mark.parametrize("value", [
    {"compilation": "future"}, {"attention_backend": "flash"}, {"step_cache": "different"},
    {"unknown": None}, {"compilation": False}, "regional-v1", [],
])
def test_unknown_execution_data_is_refused_instead_of_discarded(value):
    with pytest.raises(ValidationError):
        execution.canonical_execution(value)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="a fox", execution=value)


@pytest.mark.parametrize("changes", [{"compilation": "future"}, {"unknown": "regional-v1"}])
def test_internal_spec_copies_cannot_hide_unknown_recipe_data(changes):
    copied = execution.ExecutionSpec().model_copy(update=changes)
    with pytest.raises(ValidationError):
        execution.canonical_execution(copied)


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_nonempty_recipe_is_unavailable_even_when_named(recipe):
    spec = execution.ExecutionSpec(**recipe)
    with pytest.raises(ValidationError, match="not available"):
        GenerateRequest(prompt="a fox", execution=spec)
    with pytest.raises(ValueError, match="not available"):
        execution.require_default_execution(spec)


def _internal_request(recipe):
    # Retained/internal model_copy updates deliberately bypass pydantic. The
    # shared admission and direct backend boundaries must still refuse them.
    return GenerateRequest(prompt="a fox").model_copy(update={"execution": execution.ExecutionSpec(**recipe)})


@pytest.mark.parametrize("recipe", RECIPES)
def test_internal_requests_cannot_bypass_applied_or_submission(recipe, fake):
    request = _internal_request(recipe)
    with pytest.raises(ValueError, match="not available"):
        request.applied(negative_prompt_supported=True, cfg_threshold=1.0)
    with pytest.raises(ValueError, match="not available"):
        GenerationBackend.applied(fake, request)
    with pytest.raises(ValueError, match="not available"):
        fake.applied(request)
    # No services object is needed: refusal precedes all source/backend reads.
    assert asyncio.run(submission.unusable_input(SimpleNamespace(), request)) == ("execution", execution.UNAVAILABLE)


@pytest.mark.parametrize("recipe", RECIPES)
def test_direct_backend_refuses_before_peak_measurement_or_pipeline_work(recipe, monkeypatch):
    backend = DiffusersBackend()
    backend._pipe = object()
    backend._peak = (1, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("execution refusal reached device or pipeline work")

    monkeypatch.setattr(backend, "_peak_device", forbidden)
    monkeypatch.setattr(backend, "_generate_images", forbidden)
    monkeypatch.setattr(backend, "cfg_threshold", forbidden)
    request = _internal_request(recipe)
    with pytest.raises(ValueError, match="not available"):
        backend.applied(request)
    with pytest.raises(ValueError, match="not available"):
        backend.generate(request, 42)
    assert backend.last_peak() is None


@pytest.mark.parametrize("recipe", RECIPES)
def test_direct_fake_cannot_draw_or_record_an_unavailable_recipe(recipe, fake):
    with pytest.raises(ValueError, match="not available"):
        fake.generate(_internal_request(recipe), 42)
    assert fake.requests == [] and fake.seeds == []


@pytest.mark.parametrize("spec", [
    execution.ExecutionSpec(compilation="regional-v1"),
    execution.ExecutionSpec().model_copy(update={"compilation": "future"}),
    execution.ExecutionSpec().model_copy(update={"unknown": "regional-v1"}),
])
@pytest.mark.parametrize("kind", ["ordinary", "outpaint", "stamped-outpaint"])
@pytest.mark.parametrize("invalid_last", [False, True])
def test_queue_refuses_every_copied_recipe_before_any_admission_side_effect(
    spec, kind, invalid_last, fake, monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("an unavailable execution recipe reached queue admission work")

    async def forbidden_load():
        forbidden()

    # Selection, stamping, holds and retained work must all remain untouched,
    # even with a valid first cell and a caller-supplied id bypassing run_id().
    monkeypatch.setattr(jobs, "current_settings", forbidden)
    for name in ("stamp_outpaints", "hold_masks", "retain_run_job"):
        monkeypatch.setattr(jobs.storage, name, forbidden)
    request = GenerateRequest(prompt="a fox")
    if kind != "ordinary":
        request = request.model_copy(update={"refine": 1, "outpaint": Outpaint(left=16)})
    if kind == "stamped-outpaint":
        request = request.model_copy(update={"mask": "a" * 64})
    invalid = grid.Cell(request=request.model_copy(update={"execution": spec}), seed=42)
    valid = grid.Cell(request=GenerateRequest(prompt="a bear"), seed=43)
    cells = [valid, invalid] if invalid_last else [invalid, valid]

    async def scenario():
        manager = jobs.JobManager(fake, forbidden_load)
        try:
            with pytest.raises(ValueError):
                await manager.submit_cells(cells, run_id="explicit-existing-run")
            assert manager._loop is None
            assert manager._jobs == {} and manager._pending == []
            assert fake.requests == [] and fake.seeds == []
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [None, execution.ExecutionSpec()])
def test_queue_keeps_eager_admission_and_explicit_run_id(value, fake, store):
    async def scenario():
        loads = 0

        async def load():
            nonlocal loads
            loads += 1
            fake.load()

        manager = jobs.JobManager(fake, load)
        request = GenerateRequest(prompt="a fox", execution=value, width=256, height=256, steps=1)
        raw_execution = request.execution
        try:
            job = await manager.submit_cells([grid.Cell(request=request, seed=42)], run_id="explicit-eager-run")
            events = [event async for event in manager.events(job)]
            assert events[-1]["status"] == "done"
            assert job.run_id == "explicit-eager-run" and len(job.images) == 1
            assert loads == 1 and fake.seeds == [42]
            # Expanded cells resolve complete Eager replay before admission
            # identities; the caller's raw empty spelling stays untouched.
            assert fake.requests == [request.model_copy(update={"execution": None})]
            assert request.execution is raw_execution
            assert execution.execution_key(fake.applied(request).execution) is None
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["/api/generate", "/api/grid", "/api/grid/preview"])
@pytest.mark.parametrize("recipe", RECIPES)
def test_http_recipes_refuse_before_load_or_generation(route, recipe, client, fake, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an unavailable execution recipe reached the model")

    monkeypatch.setattr(fake, "load", forbidden)
    monkeypatch.setattr(fake, "generate", forbidden)
    response = client.post(route, json={"prompt": "a fox", "execution": recipe})
    assert response.status_code == 422, response.text
    assert "not available" in response.text
    assert fake.requests == []


@pytest.mark.parametrize("value", [
    {}, {"compilation": None}, {"attention_backend": None}, {"step_cache": None},
    {"compilation": None, "attention_backend": None, "step_cache": None}, execution.ExecutionSpec(),
])
def test_applied_alone_normalizes_empty_execution_and_preserves_stamped_outpaint(value):
    request = GenerateRequest(prompt="a fox", execution=value)
    assert isinstance(request.execution, execution.ExecutionSpec)
    # A stamped outpaint must not go back through model validation, which
    # rightly refuses a client sending both its spec and its derived mask.
    stamped = request.model_copy(update={"refine": 1, "outpaint": Outpaint(left=16), "mask": "a" * 64})
    applied = stamped.applied(negative_prompt_supported=True, cfg_threshold=1.0, inpaint_supported=True)
    assert applied.execution is None
    assert applied.mask == stamped.mask and applied.outpaint is stamped.outpaint


def test_empty_execution_keeps_existing_cell_and_run_identity():
    base = GenerateRequest(
        prompt="a {size} fox", negative_prompt="blurry", width=256, height=256, steps=8,
        guidance_scale=2.0, mu=1.15, seed=42,
    )
    plain = grid.expand(base, prompt="size = big | small", steps="6,8", seeds=2)
    empty = grid.expand(base.model_copy(update={"execution": execution.ExecutionSpec()}),
                        prompt="size = big | small", steps="6,8", seeds=2)
    assert [grid.cell_key(cell) for cell in plain] == [grid.cell_key(cell) for cell in empty]
    assert grid.run_id(empty, "krea/Krea-2-Turbo", "fp8") == "0ebe5101c492"


def test_an_empty_http_recipe_runs_eager_and_adds_no_image_metadata(client, fake, store):
    fake.cfg_threshold = 1.0
    _, events = run_one(client, execution={})
    assert events[-1]["status"] == "done"
    # Jobs hand the fake the validated request and save backend.applied().
    # Only that applied boundary normalizes its empty execution object.
    assert execution.execution_key(fake.requests[-1].execution) is None
    assert fake.applied(fake.requests[-1]).execution is None
    record = events[-1]["images"][0]
    with Image.open(store.settings.outputs_dir / record["filename"]) as image:
        assert "execution" not in json.loads(image.info[PNG_METADATA_KEY])


FACTS = execution.CompilerMetadata(
    platform="win32", python_version=(3, 13), torch="2.13.0+cu130", diffusers="0.40.0",
    torchao="0.18.0", triton=None, triton_windows="3.7.1.post27",
)
TARGET = dict(model_id="black-forest-labs/FLUX.1-Krea-dev", quantization="fp8", offload="model", device="cuda")


def test_a_matching_cold_candidate_is_still_unavailable():
    spec = execution.ExecutionSpec(compilation="regional-v1")
    assert execution.regional_candidate_reason(spec, FACTS, **TARGET) is None
    linux = dataclasses.replace(FACTS, platform="linux", triton="3.7.1", triton_windows=None)
    assert execution.regional_candidate_reason(spec, linux, **{**TARGET, "device": "cuda:0"}) is None
    with pytest.raises(ValueError, match="not available"):
        execution.require_default_execution(spec)


@pytest.mark.parametrize("changes", [
    {"model_id": "another/model"}, {"quantization": "int8"}, {"offload": "sequential"},
    {"offload": "none"}, {"device": "cpu"}, {"device": "cuda:"}, {"num_images": 2},
    {"lora": "weight-zero.safetensors"}, {"refine": 1}, {"mask": "a" * 64}, {"outpaint": True},
])
def test_cold_candidate_refuses_unsupported_targets(changes):
    assert execution.regional_candidate_reason(
        execution.ExecutionSpec(compilation="regional-v1"), FACTS, **{**TARGET, **changes}
    ) is not None


@pytest.mark.parametrize("changes", [
    {"platform": "darwin"}, {"python_version": (3, 14)}, {"torch": None},
    {"torch": "2.13.0+cpu"}, {"diffusers": "0.39.0"}, {"torchao": None},
    {"triton_windows": None}, {"triton_windows": "3.8.0.post29"},
    {"platform": "linux", "triton": None},
])
def test_cold_candidate_requires_the_recorded_dependency_environment(changes):
    assert execution.regional_candidate_reason(
        execution.ExecutionSpec(compilation="regional-v1"), dataclasses.replace(FACTS, **changes), **TARGET
    ) is not None


def test_cold_candidate_refuses_stacked_recipes():
    assert execution.regional_candidate_reason(execution.ExecutionSpec(**RECIPES[-1]), FACTS, **TARGET) is not None


def test_compiler_facts_read_only_distribution_metadata():
    seen = []

    def lookup(name):
        seen.append(name)
        if name == "triton-windows":
            raise PackageNotFoundError(name)
        return "recorded-" + name

    facts = execution.compiler_metadata(lookup=lookup, platform="win32", python_version=(3, 13))
    assert seen == ["torch", "diffusers", "torchao", "triton", "triton-windows"]
    assert facts.torch == "recorded-torch" and facts.triton_windows is None
    done = subprocess.run(
        [sys.executable, "-c", (
            "import sys; from backend.execution import compiler_metadata; compiler_metadata(); "
            "assert not {'torch', 'diffusers', 'huggingface_hub'} & sys.modules.keys()"
        )], capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stdout + done.stderr
