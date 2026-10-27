"""Saved performance intent never grants an Eager fallback or runtime permission."""
from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend import execution, grid, jobs, main, schemas_generate, submission
from backend.backends import diffusers_backend
from backend.schemas import GenerateRequest, GenerateSubmission, Outpaint
from tests import fakes


MODEL = "black-forest-labs/FLUX.1-Krea-dev"
PLACEMENT = dict(model_id=MODEL, quantization="fp8", offload="model", device="cuda")
LOADERS = [execution.ExecutionSpec(compilation="regional-v1"),
           execution.ExecutionSpec(attention_backend="cudnn-v1")]


def configure(monkeypatch, spec=None, error=None):
    snapshot = dataclasses.replace(main.settings, loader_execution=spec, loader_execution_error=error)
    for module in (main, jobs, diffusers_backend, fakes):
        monkeypatch.setattr(module, "settings", snapshot)
    return snapshot


@pytest.mark.parametrize("spec", LOADERS)
def test_configuration_is_independent_of_qualification_and_never_grants_availability(spec):
    assert execution.loader_configuration_reason(spec, **PLACEMENT) is None
    assert execution.loader_unavailable_reason(spec) == execution.UNAVAILABLE
    with pytest.raises(ValueError, match="not available"):
        execution.require_loader_available(spec)


@pytest.mark.parametrize("changes", [
    {"model_id": "another/model"}, {"quantization": "none"}, {"offload": "none"},
    {"offload": "sequential"}, {"device": "cpu"}, {"device": "cuda:"}, {"device": "cuda:bad"},
])
def test_nonempty_configuration_refuses_unsupported_placement_without_changing_eager(changes):
    context = {**PLACEMENT, **changes}
    assert execution.loader_configuration_reason(LOADERS[0], **context)
    assert execution.loader_configuration_reason(None, **context) is None


def test_conflicting_loader_and_render_components_are_explicit():
    both = execution.ExecutionSpec(compilation="regional-v1", attention_backend="cudnn-v1")
    assert "combined" in execution.loader_configuration_reason(both, **PLACEMENT)
    cached = execution.ExecutionSpec(step_cache="fbc-005-v1")
    with pytest.raises(ValueError, match="per-render"):
        execution.canonical_loader_execution(cached)
    assert execution.step_cache_configuration_reason(cached, **PLACEMENT) is None
    assert "combined" in execution.step_cache_configuration_reason(cached, loader_execution=LOADERS[0], **PLACEMENT)
    assert "cannot replace" in execution.step_cache_configuration_reason(LOADERS[0], **PLACEMENT)


@pytest.mark.parametrize("value", [None, "", "sensitive fixture value", False])
def test_bad_startup_diagnostic_is_fixed_and_not_an_empty_eager_escape(value):
    context = SimpleNamespace(loader_execution=None, loader_execution_error=value)
    expected = None if value is None else execution.INVALID_LOADER_CONFIGURATION
    assert execution.configured_loader_unavailable_reason(context) == expected


def test_malformed_internal_settings_do_not_become_eager():
    context = SimpleNamespace(loader_execution={"compilation": "unknown"})
    with pytest.raises(ValueError, match="configuration is invalid"):
        execution.require_configured_loader_available(context)


@pytest.mark.parametrize("spec", LOADERS)
@pytest.mark.parametrize("payload", [{}, {"execution": None}, {"execution": {}}])
def test_selected_recipe_refuses_omitted_and_explicit_eager_before_sources_or_backend(
    spec, payload, fake, monkeypatch,
):
    snapshot = configure(monkeypatch, spec)
    request = GenerateRequest(prompt="a fox", **payload)
    # Portable normalization and codecs stay independent of selected runtime.
    assert request.applied(negative_prompt_supported=True, cfg_threshold=1).execution is None
    assert execution.execution_from_json(execution.execution_key(spec)) == spec
    services = SimpleNamespace(settings=lambda: snapshot)
    assert asyncio.run(submission.unusable_input(services, request)) == ("execution", execution.UNAVAILABLE)
    with pytest.raises(ValueError, match="not available"):
        fake.applied(request)
    with pytest.raises(ValueError, match="not available"):
        fake.generate(request, 1)
    assert fake.requests == [] and fake.loads == 0


@pytest.mark.parametrize("error", [None, "fixed invalid profile"])
def test_queue_refuses_selected_intent_before_stamps_holds_retention_or_run_identity(fake, monkeypatch, error):
    configure(monkeypatch, None if error else LOADERS[0], error)
    effects = []
    def forbidden(*_args, **_kwargs):
        effects.append("side effect")
        raise AssertionError("unavailable selection reached queue work")
    for name in ("stamp_outpaints", "hold_masks", "retain_run_job"):
        monkeypatch.setattr(jobs.storage, name, forbidden)
    async def load():
        forbidden()
    async def scenario():
        manager = jobs.JobManager(fake, load)
        try:
            request = GenerateRequest(prompt="a fox").model_copy(update={"refine": 1, "outpaint": Outpaint(left=16)})
            with pytest.raises(ValueError):
                await manager.submit_cells([grid.Cell(request=request, seed=1)], run_id="explicit-id")
            assert manager._loop is None and manager._jobs == {} and manager._pending == []
        finally:
            await manager.shutdown()
    asyncio.run(scenario())
    assert effects == [] and fake.requests == []


@pytest.mark.parametrize("spec,error", [(LOADERS[0], None), (LOADERS[1], None), (None, "invalid profile")])
def test_direct_public_backend_blocks_load_normalization_and_device_work(spec, error, fake, monkeypatch):
    configure(monkeypatch, spec, error)
    backend = diffusers_backend.DiffusersBackend()
    backend._pipe = object()  # An already-ready Eager pipe is not permission.
    effects = []
    def forbidden(*_args, **_kwargs):
        effects.append("device or load")
        raise AssertionError("unavailable selection reached runtime work")
    monkeypatch.setattr(backend, "_load_prepared", forbidden)
    monkeypatch.setattr(backend, "_peak_device", forbidden)
    request = GenerateRequest(prompt="a fox")
    for operation in (backend.load, lambda: backend.applied(request), lambda: backend.generate(request, 1)):
        with pytest.raises(ValueError):
            operation()
    assert effects == []


@pytest.mark.parametrize("route", ["/api/generate", "/api/grid", "/api/grid/preview"])
@pytest.mark.parametrize("payload", [{}, {"execution": None}, {"execution": {}}])
def test_http_selected_recipe_cannot_render_an_omitted_or_explicit_eager_request(route, payload, client, fake, monkeypatch):
    configure(monkeypatch, LOADERS[0])
    response = client.post(route, json={"prompt": "a fox", **payload})
    assert response.status_code == 422, response.text
    assert execution.UNAVAILABLE in response.text
    assert fake.loads == 0 and fake.requests == [] and main.jobs.active() == []


@pytest.mark.parametrize("payload,explicit", [({}, False), ({"execution": None}, True), ({"execution": {}}, True)])
def test_composer_resolution_preserves_original_omission_then_stamps_base_and_grid(payload, explicit):
    request = GenerateSubmission(prompt="a fox", seed=4, **payload)
    assert ("execution" in request.model_fields_set) is explicit
    base = request.base(None)
    assert "execution" in base.model_fields_set
    cells = grid.expand(base, seeds=2)
    assert all("execution" in cell.request.model_fields_set for cell in cells)
    assert all(execution.execution_key(cell.request.execution) is None for cell in cells)


@pytest.mark.parametrize("spec", LOADERS)
def test_pure_normalization_retains_a_recipe_if_availability_was_already_checked(spec, monkeypatch):
    # Simulate a caller's future capability decision in the pure normalizer
    # only. Every actual API/queue/backend availability boundary stays closed.
    request = GenerateRequest(prompt="a fox").model_copy(update={"execution": spec})
    monkeypatch.setattr(schemas_generate, "require_default_execution", lambda _value: None)
    normalized = request.applied(negative_prompt_supported=True, cfg_threshold=1)
    assert normalized.execution == spec
    assert execution.execution_key(normalized.execution) == execution.execution_key(spec)
    with pytest.raises(ValueError, match="not available"):
        execution.require_default_execution(normalized.execution)


def test_configuration_helpers_are_import_cold_and_do_not_read_metadata_or_change_environment():
    source = r'''
import importlib.abc, importlib.metadata, os, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'diffusers', 'huggingface_hub'} or name == 'backend.config':
            raise AssertionError('unexpected runtime import: ' + name)
def forbidden(*args, **kwargs): raise AssertionError('unexpected metadata query')
sys.meta_path.insert(0, Block())
importlib.metadata.version = forbidden
before = dict(os.environ)
from backend import execution as e
spec = e.loader_execution_from_choices('regional-v1', 'native')
assert e.loader_configuration_reason(spec, model_id='black-forest-labs/FLUX.1-Krea-dev', quantization='fp8', offload='model', device='cuda') is None
assert e.loader_unavailable_reason(spec) == e.UNAVAILABLE
assert os.environ == before
print('cold configuration, unavailable execution')
'''
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cold configuration, unavailable execution"
