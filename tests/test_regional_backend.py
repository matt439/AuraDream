"""Private regional wiring over synthetic pipelines, blocks and policy only."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest
from pydantic import ValidationError

from backend.backends import Cancelled
from backend.backends import diffusers_backend as module
from backend.backends import regional_compile as regional
from backend.execution import CompilerMetadata, ExecutionSpec
from tests.test_fbc_backend import native_world
from tests.test_pipeline_ownership import synthetic_loader
from tests.test_regional_compile import (
    ARGS, FluxSingleTransformerBlock, FluxTransformer2DModel, FluxTransformerBlock,
)


SPEC = ExecutionSpec(compilation="regional-v1")
FACTS = CompilerMetadata("win32", (3, 13), "2.13.0+cu130", "0.40.0", "0.18.0", None, "3.7.1.post27")
NATIVE_GUARD = module.attention_selection.require_native_attention
CACHE_BINDING = module.compiler_cache.require_binding


@pytest.fixture
def prepared_loader(synthetic_loader, monkeypatch):
    backend, control, events = synthetic_loader
    selected = replace(module.settings, model_id="black-forest-labs/FLUX.1-Krea-dev",
                       quantization="fp8", offload="model", device="cuda")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    monkeypatch.setattr(module, "compiler_metadata", lambda: FACTS)
    monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", None)
    control.binding_checks = []

    def bound(db, outputs, bundles, facts, *, web):
        control.binding_checks.append((db, outputs, bundles, facts, tuple(events)))

    monkeypatch.setattr(module.compiler_cache, "require_binding", bound)
    # This is the fixture's ModuleType runtime, never the installed torch.
    sys.modules["torch"].cuda.is_available = lambda: True
    monkeypatch.setattr(regional, "_flux_types", lambda: (
        FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock,
    ))
    control.root = FluxTransformer2DModel()
    control.policy_active = False
    control.native = True
    control.native_checks = []

    def native(root):
        control.native_checks.append((root is control.root, control.native, list(control.root.calls)))
        if root is not control.root or not control.native:
            raise ValueError("synthetic nonnative regional attention")

    monkeypatch.setattr(module.attention_selection, "require_native_attention", native)

    @contextmanager
    def policy():
        assert not control.policy_active
        control.policy_active = True
        events.append("strict-enter")
        try:
            yield
        finally:
            control.policy_active = False
            events.append("strict-exit")

    monkeypatch.setattr(regional, "_dynamo_patch", policy)
    place = module.loading.place_pipeline

    def placement(pipe, settings):
        place(pipe, settings)
        pipe.transformer = control.root

    monkeypatch.setattr(module.loading, "place_pipeline", placement)
    return backend, control, events


def test_public_eager_load_uses_actual_private_path_without_metadata_registration_or_policy(prepared_loader, monkeypatch):
    backend, control, events = prepared_loader

    def forbidden(*args, **kwargs):
        pytest.fail("eager load reached experimental preparation")

    monkeypatch.setattr(module, "compiler_metadata", forbidden)
    monkeypatch.setattr(regional, "register_regions", forbidden)
    monkeypatch.setattr(regional, "_dynamo_patch", forbidden)
    monkeypatch.setattr(module.attention_selection, "require_native_attention", forbidden)
    monkeypatch.setattr(module.compiler_cache, "require_binding", forbidden)
    backend.load()
    assert backend.is_ready() and backend._loaded_execution is None
    assert control.root.calls == [] and "strict-enter" not in events
    backend.unload()
    assert control.releases == 1


def test_private_regional_load_registers_after_placement_and_checks_before_atomic_publication(prepared_loader):
    backend, control, events = prepared_loader
    backend._load_prepared(SPEC)
    registration = backend._regional_registration
    assert type(registration) is regional.RegionalRegistration
    assert registration.require(control.root).total_blocks == 3
    assert backend._loaded_execution == SPEC and control.root.calls == [ARGS]
    assert control.native_checks == [(True, True, []), (True, True, [ARGS])]
    assert len(control.binding_checks) == 2 and control.binding_checks[0][-2] == FACTS
    assert control.binding_checks[0][-1] == ()
    assert "weights-exit" in control.binding_checks[1][-1]
    assert all(block.calls == [ARGS] for block in control.root.blocks)
    assert events.index("place") < events.index("console") < events.index("post") < events.index("strict-enter")
    assert events.index("strict-exit") < events.index("weights-exit") < events.index("ready-log")
    assert not control.policy_active and control.releases == 0
    backend._load_prepared(SPEC)
    assert control.root.calls == [ARGS]  # no warm-up or second registration
    assert len(control.binding_checks) == 3
    request = SimpleNamespace(execution=None)
    backend._peak_device = lambda: pytest.fail("public admission reached a device query")
    for call in (lambda: backend.applied(request), lambda: backend.generate(request, 1),
                 lambda: backend._generate_images(request, 1)):
        with pytest.raises(ValueError, match="not available"):
            call()
    backend.unload()
    assert registration.closed and control.releases == 1


def test_missing_startup_cache_binding_refuses_before_weights(prepared_loader, monkeypatch):
    backend, control, events = prepared_loader
    monkeypatch.setattr(module.compiler_cache, "require_binding", CACHE_BINDING)
    monkeypatch.setattr(module.compiler_cache, "_attempt", None)
    with pytest.raises(RuntimeError, match="binding from fresh startup"):
        backend._load_prepared(SPEC)
    assert events == [] and not backend.is_ready() and control.releases == 0
    assert control.root.calls == [] and control.native_checks == []


def test_repeated_private_load_rechecks_binding_without_discarding_working_pipeline(prepared_loader, monkeypatch):
    backend, control, events = prepared_loader
    backend._load_prepared(SPEC)
    pipe = backend._pipe
    def changed(*args, **kwargs):
        raise RuntimeError("synthetic changed compiler cache")
    monkeypatch.setattr(module.compiler_cache, "require_binding", changed)
    before = list(events)
    with pytest.raises(RuntimeError, match="changed compiler cache"):
        backend._load_prepared(SPEC)
    assert backend._pipe is pipe and events == before and control.releases == 0
    backend.unload()


def test_binding_drift_at_weights_exit_refuses_publication_and_releases_outside_locks(prepared_loader, monkeypatch):
    backend, control, events = prepared_loader
    bound = module.compiler_cache.require_binding
    def changed(*args, **kwargs):
        if "weights-exit" in events:
            raise RuntimeError("synthetic late cache drift")
        return bound(*args, **kwargs)
    monkeypatch.setattr(module.compiler_cache, "require_binding", changed)
    with pytest.raises(RuntimeError, match="late cache drift"):
        backend._load_prepared(SPEC)
    assert not backend.is_ready() and control.releases == 1 and control.pipe_ref() is None


def test_backend_construction_seals_unused_cache_startup_window(monkeypatch):
    monkeypatch.setattr(module.compiler_cache, "_sealed", False)
    module.DiffusersBackend()
    assert module.compiler_cache._sealed


@pytest.mark.parametrize("spec", [
    ExecutionSpec(compilation="regional-v1", attention_backend="cudnn-v1"), ExecutionSpec(compilation="regional-v1", step_cache="fbc-005-v1"),
    SPEC.model_copy(update={"unknown": None}), SPEC.model_copy(update={"compilation": "future"}),
])
def test_invalid_or_combined_private_recipe_refuses_before_any_loading(prepared_loader, spec):
    backend, control, events = prepared_loader
    with pytest.raises((ValueError, ValidationError)):
        backend._load_prepared(spec)
    assert events == [] and not backend.is_ready() and control.releases == 0


def test_missing_compiler_refuses_before_weights_or_pipeline_imports(prepared_loader, monkeypatch):
    backend, control, events = prepared_loader
    monkeypatch.setattr(module, "compiler_metadata", lambda: replace(FACTS, triton_windows=None))
    with pytest.raises(ValueError, match="triton-windows 3.7.1.post27"):
        backend._load_prepared(SPEC)
    assert events == [] and not backend.is_ready() and control.releases == 0


def test_fixed_external_override_observation_refuses_before_even_metadata(prepared_loader, monkeypatch):
    backend, control, events = prepared_loader
    monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", "fixed external warning")
    monkeypatch.setattr(module, "compiler_metadata", lambda: pytest.fail("override refusal read metadata"))
    with pytest.raises(ValueError, match="without a nonnative external override"):
        backend._load_prepared(SPEC)
    assert events == [] and control.releases == 0


@pytest.mark.parametrize("failure", [RuntimeError("partial registration"), Cancelled("setup cancelled")])
def test_partial_registration_never_publishes_or_falls_back_and_releases_after_locks(prepared_loader, failure):
    backend, control, events = prepared_loader
    control.root.blocks[1].failure = failure
    with pytest.raises(type(failure)) as raised:
        backend._load_prepared(SPEC)
    assert raised.value is failure and not getattr(failure, "__notes__", [])
    assert not backend.is_ready() and backend._loaded_execution is None
    assert backend._regional_registration is None and control.pipe_ref() is None
    assert control.root.blocks[0]._compiled_call_impl is not None  # never relabel the partial model Eager
    assert control.releases == 1 and not control.policy_active and events[-1] == "release"


def test_invalidation_blocks_publication_even_if_a_registrar_returns_after_its_failure_callback(prepared_loader, monkeypatch):
    backend, control, events = prepared_loader
    returned = []

    def invalid_registration(root, *, discard_on_failure):
        discard_on_failure()
        handle = regional.RegionalRegistration(root, regional.RegionFacts("regional-v1", (
            ("FluxTransformerBlock", 2), ("FluxSingleTransformerBlock", 1),
        )))
        returned.append(handle)
        return handle

    monkeypatch.setattr(regional, "register_regions", invalid_registration)
    with pytest.raises(RuntimeError, match="invalidated") as raised:
        backend._load_prepared(SPEC)
    assert not getattr(raised.value, "__notes__", [])
    assert not backend.is_ready() and returned[0].closed and control.releases == 1


def test_private_preparation_cannot_replace_an_already_loaded_eager_pipeline(prepared_loader):
    backend, control, events = prepared_loader
    backend.load()
    original = backend._pipe
    before = list(events)
    with pytest.raises(RuntimeError, match="unload the current"):
        backend._load_prepared(SPEC)
    assert backend._pipe is original and backend._loaded_execution is None
    assert events == before and control.releases == 0


@pytest.fixture
def private_call(monkeypatch):
    backend = module.DiffusersBackend()
    events = []
    state = SimpleNamespace(active=False, failure=None, native=True, change_native=False, native_checks=[],
                            cache_valid=True, change_cache=False, cache_checks=[])
    monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", None)

    class Pipeline:
        transformer = object()

        def __call__(self, **kwargs):
            events.append(("call", dict(kwargs), state.active))
            if state.change_native:
                state.native = False
            if state.change_cache:
                state.cache_valid = False
            if state.failure is not None:
                raise state.failure
            return SimpleNamespace(images=["synthetic image"])

    @contextmanager
    def policy():
        assert not state.active
        state.active = True
        events.append("enter")
        try:
            yield
        finally:
            state.active = False
            events.append("exit")

    monkeypatch.setattr(regional, "_dynamo_patch", policy)
    pipe = backend._pipe = Pipeline()
    backend._loaded_execution = SPEC
    backend._regional_registration = regional.RegionalRegistration(pipe.transformer, regional.RegionFacts(
        "regional-v1", (("FluxTransformerBlock", 2), ("FluxSingleTransformerBlock", 1)),
    ))
    backend._regional_registration.diagnostics.publish(1)

    def native(root):
        state.native_checks.append((root is pipe.transformer, state.native))
        if not state.native:
            raise ValueError("synthetic nonnative regional attention")

    monkeypatch.setattr(module.attention_selection, "require_native_attention", native)
    def cache_binding(*args):
        state.cache_checks.append((state.cache_valid, state.active))
        if not state.cache_valid:
            raise RuntimeError("synthetic compiler cache drift")
    monkeypatch.setattr(backend, "_require_compiler_cache", cache_binding)
    return backend, pipe, state, events


@pytest.mark.parametrize("failure", [None, RuntimeError("lazy compiler failed"), Cancelled("call cancelled")])
def test_private_call_encloses_exactly_one_forward_and_restores_policy_without_retry(private_call, failure):
    backend, pipe, state, events = private_call
    kwargs = {"num_images_per_prompt": 1, "generator": object(), "callback_on_step_end": object()}
    before = dict(kwargs)
    state.failure = failure
    if failure is None:
        assert backend._invoke_pipeline(pipe, kwargs).images == ["synthetic image"]
    else:
        with pytest.raises(type(failure)) as raised:
            backend._invoke_pipeline(pipe, kwargs)
        assert raised.value is failure
    assert events == ["enter", ("call", before, True), "exit"]
    assert kwargs == before and not state.active
    assert state.native_checks == [(True, True), (True, True)]


def test_diagnostics_publish_only_after_success_and_retire_on_unload_and_reload(prepared_loader):
    backend, control, _events = prepared_loader
    assert backend.execution_diagnostics_reader() is None and backend.execution_diagnostics() is None
    backend._load_prepared(SPEC)
    registration = backend._regional_registration
    reader = backend.execution_diagnostics_reader()
    assert reader.__self__ is registration.diagnostics
    first = regional.RegionalDiagnostics(1)
    assert reader() == backend.execution_diagnostics() == first
    backend.unload()
    assert reader() is None and backend.execution_diagnostics_reader() is None
    assert backend.execution_diagnostics() is None and first == regional.RegionalDiagnostics(1)
    control.root = FluxTransformer2DModel()
    backend._load_prepared(SPEC)
    assert backend.execution_diagnostics() == regional.RegionalDiagnostics(2)
    assert reader() is None
    backend.unload()


def test_failed_preparation_never_publishes_diagnostics(prepared_loader):
    backend, control, _events = prepared_loader
    control.stage = "post"
    with pytest.raises(RuntimeError, match="synthetic preparation failed"):
        backend._load_prepared(SPEC)
    assert backend.execution_diagnostics_reader() is None and backend.execution_diagnostics() is None


def test_try_unload_refusal_keeps_observation_then_success_retires_it(prepared_loader):
    backend, _control, _events = prepared_loader
    backend._load_prepared(SPEC)
    reader = backend.execution_diagnostics_reader()
    with backend._lock:
        assert not backend.try_unload()
        assert reader() == regional.RegionalDiagnostics(1)
    assert backend.try_unload() and reader() is None
    assert backend.execution_diagnostics() is None


@pytest.mark.parametrize("fault", ["batch", "native", "cache", "policy-entry"])
def test_first_forward_preflight_or_policy_entry_refusal_leaves_pending(private_call, monkeypatch, fault):
    backend, pipe, state, events = private_call
    kwargs = {"num_images_per_prompt": 1}
    failure = RuntimeError("synthetic policy entry failure")
    if fault == "batch":
        kwargs["num_images_per_prompt"] = 2
    elif fault == "native":
        state.native = False
    elif fault == "cache":
        state.cache_valid = False
    else:
        class BrokenEntry:
            def __enter__(self):
                raise failure
            def __exit__(self, *args):
                pytest.fail("an unentered policy was exited")
        monkeypatch.setattr(regional, "_dynamo_patch", BrokenEntry)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        backend._invoke_pipeline(pipe, kwargs)
    if fault == "policy-entry":
        assert caught.value is failure
    assert events == [] and not state.active
    assert backend.execution_diagnostics() == regional.RegionalDiagnostics(1)


@pytest.mark.parametrize("failure_kind", ["success", "error", "cancel"])
@pytest.mark.parametrize("exit_kind", ["normal", "failure", "suppress"])
def test_policy_exit_preserves_first_outcome_primary_cause_and_never_claims_return_early(private_call, monkeypatch, failure_kind, exit_kind):
    backend, pipe, state, events = private_call
    primary = None if failure_kind == "success" else (
        Cancelled("synthetic first cancellation") if failure_kind == "cancel" else RuntimeError("synthetic first failure")
    )
    cause = ValueError("original pipeline cause")
    if primary is not None:
        primary.__cause__ = cause
    state.failure = primary
    cleanup = OSError("synthetic policy exit failure")
    during_exit = []
    class Policy:
        def __enter__(self):
            state.active = True
            events.append("enter")
        def __exit__(self, kind, error, trace):
            during_exit.append(backend.execution_diagnostics())
            state.active = False
            events.append("exit")
            if exit_kind == "failure":
                raise cleanup
            return exit_kind == "suppress"
    monkeypatch.setattr(regional, "_dynamo_patch", Policy)
    expected_error = primary if primary is not None else cleanup if exit_kind == "failure" else None
    if expected_error is None:
        assert backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1}).images == ["synthetic image"]
        expected = regional.RegionalDiagnostics(1, "returned")
    else:
        with pytest.raises(type(expected_error)) as caught:
            backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
        assert caught.value is expected_error
        if primary is not None:
            assert primary.__cause__ is cause
            if exit_kind == "failure":
                assert "Regional compiler policy cleanup also failed" in " ".join(primary.__notes__)
            else:
                assert not getattr(primary, "__notes__", [])
        expected = regional.RegionalDiagnostics(1, "cancelled" if failure_kind == "cancel" else "failed",
                                                 "forward" if primary is not None else "policy_exit")
    assert during_exit == [regional.RegionalDiagnostics(1, "running")]
    assert backend.execution_diagnostics() == expected
    assert events == ["enter", ("call", {"num_images_per_prompt": 1}, True), "exit"]
    assert not state.active


def test_first_post_verification_failure_is_terminal_and_later_success_does_not_relabel_it(private_call):
    backend, pipe, state, _events = private_call
    state.change_native = True
    with pytest.raises(ValueError, match="nonnative regional attention"):
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    expected = regional.RegionalDiagnostics(1, "failed", "verification")
    assert backend.execution_diagnostics() == expected
    state.change_native, state.native = False, True
    assert backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1}).images == ["synthetic image"]
    assert backend.execution_diagnostics() == expected


def test_current_diagnostic_read_drops_a_snapshot_if_registration_changes_mid_read(private_call, monkeypatch):
    backend, _pipe, _state, _events = private_call
    owner = backend._regional_registration.diagnostics
    old_reader = owner.snapshot
    def changed():
        backend._regional_registration = None
        return old_reader()
    monkeypatch.setattr(backend, "execution_diagnostics_reader", lambda: changed)
    assert backend.execution_diagnostics() is None


def test_retired_first_call_cannot_update_replacement_diagnostics(private_call, monkeypatch):
    backend, pipe, state, _events = private_call
    old = backend._regional_registration
    reader = backend.execution_diagnostics_reader()
    replacement_root = object()
    replacement = regional.RegionalRegistration(replacement_root, regional.RegionFacts(
        "regional-v1", (("FluxTransformerBlock", 1), ("FluxSingleTransformerBlock", 1)),
    ))
    replacement.diagnostics.publish(2)
    def forward(self, **kwargs):
        assert reader() == regional.RegionalDiagnostics(1, "running")
        old.close()
        backend._regional_registration = replacement
        backend._pipe = SimpleNamespace(transformer=replacement_root)
        return SimpleNamespace(images=["stale synthetic image"])
    monkeypatch.setattr(type(pipe), "__call__", forward)
    with pytest.raises(RuntimeError, match="does not own"):
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert reader() is None and not state.active
    assert backend.execution_diagnostics() == regional.RegionalDiagnostics(2)


def test_diagnostics_refuse_orphaned_root_without_runtime_work(private_call):
    backend, pipe, _state, events = private_call
    original = pipe.transformer
    pipe.transformer = object()
    assert backend.execution_diagnostics_reader() is None and backend.execution_diagnostics() is None
    assert events == []
    pipe.transformer = original
    assert backend.execution_diagnostics() == regional.RegionalDiagnostics(1)


def test_private_regional_call_refuses_cache_drift_before_policy_or_forward(private_call):
    backend, pipe, state, events = private_call
    state.cache_valid = False
    with pytest.raises(RuntimeError, match="compiler cache drift"):
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert events == [] and not state.active and state.cache_checks == [(False, False)]


@pytest.mark.parametrize("failure", [None, RuntimeError("forward failed while cache changed"), Cancelled("cancelled while cache changed")])
def test_cache_drift_after_one_forward_refuses_result_or_preserves_exact_primary(private_call, failure):
    backend, pipe, state, events = private_call
    state.change_cache = True
    state.failure = failure
    with pytest.raises(RuntimeError if failure is None else type(failure)) as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    if failure is None:
        assert "compiler cache drift" in str(caught.value)
    else:
        assert caught.value is failure
        assert "Regional compiler cache verification also failed" in " ".join(failure.__notes__)
    assert state.cache_checks == [(True, False), (False, True)]
    assert events == ["enter", ("call", {"num_images_per_prompt": 1}, True), "exit"]
    assert not state.active


def test_nonnative_regional_preparation_refuses_before_registration(prepared_loader):
    backend, control, events = prepared_loader
    control.native = False
    with pytest.raises(ValueError, match="nonnative regional attention"):
        backend._load_prepared(SPEC)
    assert control.root.calls == [] and "strict-enter" not in events
    assert not backend.is_ready() and control.releases == 1


@pytest.mark.parametrize("fault", ["global", "processor"])
def test_shared_effective_native_guard_refuses_private_regional_setup(prepared_loader, native_world, monkeypatch, fault):
    backend, control, events = prepared_loader
    root, native = native_world
    control.root = root
    if fault == "global":
        native.active = "_native_cudnn"
    else:
        root.transformer_blocks[0].attn.processor._attention_backend = "_native_cudnn"
    monkeypatch.setattr(module.attention_selection, "require_native_attention", NATIVE_GUARD)
    with pytest.raises(ValueError, match="requires.*Native"):
        backend._load_prepared(SPEC)
    assert root.calls == [] and "strict-enter" not in events
    # Global baseline refusal now precedes the weights hold; a processor-layout
    # refusal still occurs after the synthetic pipeline has been placed.
    assert not backend.is_ready() and control.releases == (0 if fault == "global" else 1)


def test_native_state_is_rechecked_after_weights_exit_before_publication(prepared_loader, monkeypatch):
    backend, control, _events = prepared_loader
    weights = module.loading.weights_loading

    @contextmanager
    def changed_at_exit():
        with weights():
            yield
        control.native = False

    monkeypatch.setattr(module.loading, "weights_loading", changed_at_exit)
    with pytest.raises(ValueError, match="nonnative regional attention"):
        backend._load_prepared(SPEC)
    assert control.root.calls == [ARGS] and not backend.is_ready() and control.releases == 1
    assert backend._regional_registration is None


def test_nonnative_regional_call_refuses_before_policy_and_forward(private_call):
    backend, pipe, state, events = private_call
    state.native = False
    with pytest.raises(ValueError, match="nonnative regional attention"):
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert events == [] and not state.active


@pytest.mark.parametrize("fault", ["global", "processor"])
def test_shared_effective_native_guard_refuses_private_regional_call(private_call, native_world, monkeypatch, fault):
    backend, pipe, state, events = private_call
    root, native = native_world
    pipe.transformer = root
    backend._regional_registration = regional.RegionalRegistration(root, regional.RegionFacts(
        "regional-v1", (("FluxTransformerBlock", 2), ("FluxSingleTransformerBlock", 1)),
    ))
    if fault == "global":
        native.active = "_native_cudnn"
    else:
        root.single_transformer_blocks[0].attn.processor._attention_backend = "_native_cudnn"
    monkeypatch.setattr(module.attention_selection, "require_native_attention", NATIVE_GUARD)
    with pytest.raises(ValueError, match="requires.*Native"):
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert events == [] and not state.active and root.calls == []


@pytest.mark.parametrize("failure", [None, RuntimeError("regional forward failed"), Cancelled("regional forward cancelled")])
def test_native_change_during_regional_forward_preserves_primary_and_restores_policy(private_call, failure):
    backend, pipe, state, events = private_call
    state.failure, state.change_native = failure, True
    with pytest.raises(ValueError if failure is None else type(failure)) as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    if failure is not None:
        assert caught.value is failure
        assert "Regional Native attention verification also failed" in " ".join(failure.__notes__)
    assert events == ["enter", ("call", {"num_images_per_prompt": 1}, True), "exit"]
    assert not state.active and state.native_checks == [(True, True), (True, False)]


@pytest.mark.parametrize("change", ["closed", "foreign-root", "wrong-handle", "wrong-recipe", "bad-counts",
                                  "combination", "missing", "lora-zero", "batch", "sibling", "external"])
def test_private_call_refuses_incoherent_ownership_before_policy_or_forward(private_call, monkeypatch, change):
    backend, pipe, state, events = private_call
    runner, kwargs = pipe, {"num_images_per_prompt": 1}
    if change == "closed":
        backend._regional_registration.close()
    elif change == "foreign-root":
        pipe.transformer = object()
    elif change == "wrong-handle":
        backend._regional_registration = SimpleNamespace(require=lambda *args: pytest.fail("foreign handle called"))
    elif change == "wrong-recipe":
        backend._regional_registration._facts = regional.RegionFacts("future", (("FluxTransformerBlock", 2), ("FluxSingleTransformerBlock", 1)))
    elif change == "bad-counts":
        backend._regional_registration._facts = regional.RegionFacts("regional-v1", (("FluxTransformerBlock", True), ("FluxSingleTransformerBlock", 1)))
    elif change == "combination":
        backend._loaded_execution = ExecutionSpec(compilation="regional-v1", step_cache="fbc-005-v1")
    elif change == "missing":
        backend._regional_registration = None
    elif change == "lora-zero":
        backend._lora = ("adapter.safetensors", 0.0)
    elif change == "batch":
        kwargs["num_images_per_prompt"] = 2
    elif change == "sibling":
        runner = SimpleNamespace(transformer=pipe.transformer)
    elif change == "external":
        monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", "fixed warning")
    with pytest.raises((ValueError, RuntimeError)):
        backend._invoke_pipeline(runner, kwargs)
    assert events == [] and not state.active


def test_actual_eager_refine_loop_uses_call_boundary_preserving_generators_and_finally_cleanup(private_call):
    backend, pipe, state, events = private_call
    backend._loaded_execution = backend._regional_registration = None
    freed = []
    pipe.maybe_free_model_hooks = lambda: freed.append(True)
    generators = [object(), object()]
    batch = {"index": 0}
    kwargs = {"num_images_per_prompt": 2, "prompt": "a synthetic scene"}
    assert backend._refine(pipe, pipe, kwargs, generators, batch, None) == ["synthetic image"] * 2
    assert [event[1]["generator"] for event in events] == generators
    assert all(event[0] == "call" and event[1]["num_images_per_prompt"] == 1 and not event[2] for event in events)
    assert kwargs["num_images_per_prompt"] == 2 and batch["index"] == 1 and freed == [True]
    events.clear()
    state.failure = Cancelled("refine cancelled")
    with pytest.raises(Cancelled):
        backend._refine(pipe, pipe, kwargs, generators, batch, None)
    assert len(events) == 1 and freed == [True, True] and not state.active
    assert state.native_checks == []


def test_actual_eager_direct_path_keeps_no_grad_mu_callbacks_and_single_call(synthetic_loader, monkeypatch):
    from backend.schemas import GenerateRequest

    backend, control, events = synthetic_loader
    backend.load()
    pipe = backend._pipe
    state = {"grad": False, "mu": False, "calls": 0}

    class Generator:
        def __init__(self, **kwargs):
            pass

        def manual_seed(self, seed):
            self.seed = seed
            return self

    @contextmanager
    def context(name):
        state[name] = True
        try:
            yield
        finally:
            state[name] = False

    sys.modules["torch"].Generator = Generator
    sys.modules["torch"].no_grad = lambda: context("grad")
    monkeypatch.setattr(module.schedule, "requested_shift", lambda *args: (None, None))
    monkeypatch.setattr(module.schedule, "forced_mu", lambda *args: context("mu"))
    monkeypatch.setattr(regional, "_dynamo_patch", lambda: pytest.fail("Eager entered compiler policy"))
    monkeypatch.setattr(backend, "applied", lambda request: request)
    monkeypatch.setattr(backend, "_apply_lora", lambda *args: None)

    def forward(self, generator=None, callback_on_step_end=None):
        assert state["grad"] and state["mu"]
        assert generator[0].seed == 42
        assert callback_on_step_end(self, 0, None, {}) == {}
        state["calls"] += 1
        return SimpleNamespace(images=["synthetic direct image"])

    monkeypatch.setattr(type(pipe), "__call__", forward)
    assert backend._generate_images(GenerateRequest(prompt="a scene", steps=1, width=256, height=256), 42) == ["synthetic direct image"]
    assert state == {"grad": False, "mu": False, "calls": 1}


def test_private_preflight_is_import_cold_in_a_fresh_process(tmp_path):
    script = r'''
import importlib.abc
import json
import os
import sys
from types import SimpleNamespace
blocked = {"torch", "torchao", "diffusers", "huggingface_hub", "triton"}
class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError("cold refusal imported runtime: " + fullname)
        return None
sys.meta_path.insert(0, ColdOnly())
from backend.backends import diffusers_backend as module
from backend.execution import CompilerMetadata, ExecutionSpec
before = dict(os.environ)
module.current_settings = lambda _: SimpleNamespace(model_id="black-forest-labs/FLUX.1-Krea-dev",
    quantization="fp8", offload="model", device="cuda")
module.ATTENTION_ENVIRONMENT_WARNING = None
module.compiler_metadata = lambda: CompilerMetadata("win32", (3, 13), "2.13.0+cu130", "0.40.0", "0.18.0", None, None)
def forbidden(*args, **kwargs):
    raise AssertionError("cold refusal entered the weights hold")
module.loading.weights_loading = forbidden
backend = module.DiffusersBackend()
for spec in (ExecutionSpec(compilation="regional-v1"), ExecutionSpec(compilation="regional-v1", attention_backend="cudnn-v1"),
             ExecutionSpec().model_copy(update={"unknown": None})):
    try:
        backend._load_prepared(spec)
    except ValueError:
        pass
    else:
        raise AssertionError("private unsupported candidate was accepted")
module.ATTENTION_ENVIRONMENT_WARNING = "fixed warning"
module.compiler_metadata = forbidden
try:
    backend._load_prepared(ExecutionSpec(compilation="regional-v1"))
except ValueError as error:
    assert "external override" in str(error)
else:
    raise AssertionError("private preparation ignored external override")
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
assert before == dict(os.environ)
assert not backend.is_ready()
print(json.dumps({"cold": True, "refused": 4, "unchanged_environment": True}))
'''
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    done = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)
    (tmp_path / "preflight.stdout").write_text(done.stdout, encoding="utf-8")
    (tmp_path / "preflight.stderr").write_text(done.stderr, encoding="utf-8")
    assert done.returncode == 0, done.stdout + "\n" + done.stderr
    assert json.loads(done.stdout) == {"cold": True, "refused": 4, "unchanged_environment": True}
