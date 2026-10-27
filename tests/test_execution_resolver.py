"""Composer identity and explicit replay agree without granting execution."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from backend import api_generate, api_png, execution, generate, grid, jobs, main, model_selection, submission
from backend.backends.base import GenerationBackend
from backend.backends.diffusers_backend import DiffusersBackend
from backend.schemas import GenerateRequest, GenerateSubmission, GridRequest, Outpaint

LOADERS = [None, {"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"}]
EMPTIES = [None, {}, {"compilation": None, "attention_backend": None, "step_cache": None}]


@pytest.fixture(autouse=True)
def fresh_selection(monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", (None, 0))


@pytest.mark.parametrize("loader", LOADERS)
def test_new_composer_stamps_one_canonical_selected_loader_without_availability(loader):
    expected = execution.canonical_execution(loader)
    resolved = execution.resolve_execution(loader)
    assert resolved == expected
    assert execution.resolve_execution(loader, resolved) == expected
    if resolved is not None:
        with pytest.raises(ValueError, match="not available"):
            execution.require_default_execution(resolved)


@pytest.mark.parametrize("empty", EMPTIES)
@pytest.mark.parametrize("loader", LOADERS)
def test_explicit_eager_replay_never_adopts_selected_loader(loader, empty):
    if loader is None:
        assert execution.resolve_execution(loader, empty) is None
    else:
        with pytest.raises(ValueError, match="records different"):
            execution.resolve_execution(loader, empty)


@pytest.mark.parametrize("recorded", LOADERS[1:])
def test_nonempty_replay_must_match_loading_components(recorded):
    assert execution.resolve_execution(recorded, recorded) == execution.canonical_execution(recorded)
    for loader in LOADERS:
        if loader != recorded:
            with pytest.raises(ValueError, match="records different"):
                execution.resolve_execution(loader, recorded)


@pytest.mark.parametrize("loader", LOADERS)
def test_composer_cache_does_not_replace_loader_or_open_combinations(loader):
    if loader is None:
        expected = execution.ExecutionSpec(step_cache="fbc-005-v1")
        assert execution.resolve_execution(loader, step_cache="fbc-005-v1") == expected
        assert execution.resolve_execution(loader, expected) == expected
    else:
        with pytest.raises(ValueError, match="combinations"):
            execution.resolve_execution(loader, step_cache="fbc-005-v1")
        with pytest.raises(ValueError, match="records different"):
            execution.resolve_execution(loader, {"step_cache": "fbc-005-v1"})


@pytest.mark.parametrize("recorded", [None, {}, {"step_cache": "fbc-005-v1"}])
@pytest.mark.parametrize("cache", [None, "fbc-005-v1"])
def test_both_channels_are_refused_even_when_explicit_cache_is_off(recorded, cache):
    with pytest.raises(ValueError, match="not both"):
        execution.resolve_execution(None, recorded, step_cache=cache)


@pytest.mark.parametrize("value", [False, [], {"unknown": None}, {"compilation": "unknown"},
                                  execution.ExecutionSpec().model_copy(update={"step_cache": "unknown"})])
def test_invalid_or_copied_intent_is_not_normalized_into_eager(value):
    with pytest.raises(ValueError):
        execution.resolve_execution(None, value)


@pytest.mark.parametrize("submission", [GenerateSubmission, GridRequest])
def test_submission_only_cache_is_consumed_and_never_copied_as_eager(submission):
    request = submission(prompt="fox", step_cache="fbc-005-v1")
    resolved = request.with_execution(None)
    assert type(resolved) is GenerateRequest and resolved.execution.step_cache == "fbc-005-v1"
    assert "step_cache" not in type(resolved).model_fields and "execution" in resolved.model_fields_set
    with pytest.raises(ValueError, match="not available"):
        request.base(None)
    with pytest.raises(ValueError, match="not both"):
        submission(prompt="fox", execution=None, step_cache=None)


def test_narrow_runtime_copy_preserves_a_stamped_mask_without_revalidation():
    request = GenerateSubmission(prompt="fox", refine=1).model_copy(update={
        "outpaint": Outpaint(left=16), "mask": "a" * 64,
    })
    resolved = request.with_execution(None)
    assert resolved.mask == request.mask and resolved.outpaint is request.outpaint
    assert resolved.execution is None and "execution" in resolved.model_fields_set


def test_direct_admission_refuses_resolved_cache_before_any_later_service_access():
    services = SimpleNamespace(settings=lambda: main.settings)
    request = GenerateSubmission(prompt="fox", step_cache="fbc-005-v1")
    assert asyncio.run(submission.unusable_input(services, request)) == ("execution", execution.UNAVAILABLE)


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("operation", ["pure_apply", "fake_apply", "fake_generate", "base_apply", "backend_apply", "backend_generate"])
def test_submission_subtype_cannot_drop_cache_at_direct_runtime_boundary(fake, monkeypatch, bypass, operation):
    request = GenerateSubmission(prompt="fox", step_cache="fbc-005-v1")
    if bypass:
        request = request.model_copy(update={"execution": None})
    backend = DiffusersBackend()
    backend._pipe = object()
    monkeypatch.setattr(backend, "_peak_device", lambda: pytest.fail("cache refusal reached device work"))
    calls = {
        "pure_apply": lambda: request.applied(negative_prompt_supported=True, cfg_threshold=1),
        "fake_apply": lambda: fake.applied(request),
        "fake_generate": lambda: fake.generate(request, 1),
        "base_apply": lambda: GenerationBackend.applied(fake, request),
        "backend_apply": lambda: backend.applied(request),
        "backend_generate": lambda: backend.generate(request, 1),
    }
    expected = "Resolve composer" if operation == "pure_apply" else "not both" if bypass else "not available"
    with pytest.raises(ValueError, match=expected):
        calls[operation]()
    assert fake.requests == [] and fake.loads == 0


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_queue_refuses_unconsumed_submission_cache_before_identity_or_retention(fake, monkeypatch, expanded, bypass):
    effects = []

    def forbidden(*_args, **_kwargs):
        effects.append("effect")
        pytest.fail("unconsumed cache reached queue side effects")

    monkeypatch.setattr(jobs, "_run_id", forbidden)
    monkeypatch.setattr(jobs.storage, "retain_run_job", forbidden)
    request = GenerateSubmission(prompt="fox", step_cache="fbc-005-v1")
    if bypass:
        request = request.model_copy(update={"execution": None})

    async def scenario():
        manager = jobs.JobManager(fake)
        try:
            with pytest.raises(ValueError):
                if expanded:
                    await manager.submit_cells([grid.Cell(request=request, seed=4)], run_id="given")
                else:
                    await manager.submit(request)
            assert manager._jobs == {} and manager._loop is None
        finally:
            await manager.shutdown()

    asyncio.run(scenario())
    assert effects == [] and fake.requests == []


def test_grid_resolves_before_run_identity_and_plain_generate_keeps_explicit_eager(client, fake, monkeypatch):
    original = api_generate.grid.run_id
    observed = []

    def identified(cells, *args, **kwargs):
        assert all("execution" in cell.request.model_fields_set for cell in cells)
        assert all(cell.request.execution is None for cell in cells)
        observed.append(True)
        return original(cells, *args, **kwargs)

    monkeypatch.setattr(api_generate.grid, "run_id", identified)
    payload = {"prompt": "fox", "width": 256, "height": 256, "steps": 1, "seed": 4}
    response = client.post("/api/grid/preview", json=payload)
    assert response.status_code == 200, response.text
    assert observed
    for extra in ({"step_cache": "fbc-005-v1"}, {"execution": None, "step_cache": None}):
        rejected = client.post("/api/generate", json={**payload, **extra})
        assert rejected.status_code == 422, rejected.text
    assert fake.requests == [] and fake.loads == 0


def test_legacy_retained_finish_stamps_explicit_eager_before_first_identity(client, fake, store, monkeypatch):
    request = GenerateRequest(prompt="retained", width=256, height=256, steps=1)
    store.retain_run_job({
        "job_id": "resolver-legacy", "run_id": "resolver-legacy-run", "created_at": time.time(),
        "origin": "grid", "model_id": main.settings.model_id, "quantization": main.settings.quantization,
        "width": 256, "height": 256, "batch": 1, "comparison": False, "metadata": {"grid": True},
    }, [grid.Cell(request=request, seed=7)])
    decoded = store.retained_run_jobs("resolver-legacy-run")
    legacy = decoded[0]["cells"][0].request
    legacy.model_fields_set.discard("execution")  # Equivalent to a pre-column payload.
    monkeypatch.setattr(store, "retained_run_jobs", lambda *_: decoded)
    applied = fake.applied
    observations = []

    def checked(request):
        assert "execution" in request.model_fields_set and request.execution is None
        observations.append(True)
        return applied(request)

    monkeypatch.setattr(fake, "applied", checked)
    response = client.post("/api/runs/resolver-legacy-run/finish/preview", json={})
    assert response.status_code == 200, response.text
    assert observations and "execution" not in legacy.model_fields_set
    assert fake.requests == []


def test_png_origin_is_explicit_replay_for_local_record_and_composer_for_a1111(monkeypatch):
    monkeypatch.setattr(api_png, "_verified_references", lambda *_: ([], 0))
    fields, *_ = api_png._from_record({"prompt": "fox", "size": "256x256"}, "legacy.png")
    assert "execution" in fields and fields["execution"] is None
    fields, *_ = api_png._from_parameters("fox\nSteps: 20, Sampler: Euler, Size: 256x256", "external.png", None)
    assert "execution" not in fields


@pytest.mark.parametrize("loader", LOADERS)
def test_cli_uses_same_composer_and_strict_replay_resolver(loader):
    args = SimpleNamespace(step_cache=None, execution_explicit=False)
    assert generate._resolved_cli_execution(args, loader, None) == execution.canonical_execution(loader)
    if loader is None:
        assert generate._resolved_cli_execution(args, loader, {}) is None
    else:
        with pytest.raises(ValueError, match="records different"):
            generate._resolved_cli_execution(args, loader, {})
    args.step_cache = "off"
    with pytest.raises(ValueError, match="not both"):
        generate._resolved_cli_execution(args, loader, {"execution": loader})


def test_cli_record_cannot_hide_conflicting_explicit_eager_flags():
    selected = {"attention_backend": "cudnn-v1"}
    args = SimpleNamespace(step_cache=None, execution_explicit=True, execution_recipe=None)
    with pytest.raises(ValueError, match="records different"):
        generate._resolved_cli_execution(args, selected, {"execution": selected})
