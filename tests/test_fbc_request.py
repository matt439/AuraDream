"""Private request-owned cache lifetimes over one synthetic ordinary Eager load.

These tests exercise an internal call-intent seam, never an admission bypass or
real diffusers hooks, framework arithmetic, compiler, model, device or browser.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.backends import Cancelled
from backend.backends import diffusers_backend as module
from backend.backends import fbc_cache as fbc
from backend.execution import ExecutionSpec, canonical_execution, resolve_execution
from backend.schemas import GenerateRequest
from tests.test_fbc_backend import cache_loader as cache_loader
from tests.test_fbc_cache import (
    HEAD, FluxTransformer2DModel, Hook, Registry, world as world,
)
from tests.test_pipeline_ownership import synthetic_loader as synthetic_loader


ON = ExecutionSpec(step_cache="fbc-005-v1")


def _clean_root(root):
    assert root._cache_config is None
    for _, child in root.named_modules():
        assert not hasattr(child, "_diffusers_hook")
        assert "forward" not in vars(child)


@pytest.fixture
def request_runtime(cache_loader, monkeypatch):
    backend, control, events, state = cache_loader
    backend.load()  # Ordinary Eager load, not the private load-owned cache path.
    runtime = SimpleNamespace(backend=backend, control=control, events=events, state=state,
                              pipe=backend._pipe, root=control.root, owners=[], calls=[],
                              failure=None, fail_reset=False, during_forward=None)
    initialize = fbc.FirstBlockCacheOwner.__init__

    def owner_init(owner, *args, **kwargs):
        initialize(owner, *args, **kwargs)
        runtime.owners.append(owner)

    def forward(pipe, generator=None, callback_on_step_end=None, **kwargs):
        # Keep the loader's required callable contract visible on recovery.
        kwargs.update(generator=generator, callback_on_step_end=callback_on_step_end)
        assert pipe is runtime.pipe and pipe.transformer is runtime.root
        assert backend._loaded_execution is None
        cached = runtime.root._cache_config is not None
        owner = backend._cache_owner
        if cached:
            assert owner is runtime.owners[-1] and owner.active
            assert runtime.root._cache_config.threshold == 0.05
            manager = state.managers[-1]
            assert manager._state_cache == {} and manager._current_context is None
            with runtime.root.cache_context("cond"):
                manager.get_state()["residual"] = object()
            with runtime.root.cache_context("uncond"):
                assert manager.get_state() == {}
                manager.get_state()["residual"] = object()
            manager.fail_reset = runtime.fail_reset
        else:
            assert owner is None and backend._cache_disposal is None
        runtime.calls.append((cached, dict(kwargs)))
        if runtime.during_forward is not None:
            runtime.during_forward(kwargs)
        if runtime.failure is not None:
            raise runtime.failure
        return SimpleNamespace(images=["synthetic image"])

    monkeypatch.setattr(fbc.FirstBlockCacheOwner, "__init__", owner_init)
    monkeypatch.setattr(type(runtime.pipe), "__call__", forward)
    yield runtime
    if backend.is_ready():
        backend.unload()


def _call(runtime, execution, **kwargs):
    arguments = {"num_images_per_prompt": 1, "generator": [object()],
                 "callback_on_step_end": object()}
    arguments.update(kwargs)
    with module._CallDisposal(runtime.backend, request_execution=execution) as call:
        result = runtime.backend._invoke_pipeline(runtime.pipe, arguments, disposal=call)
    return result, call


def _resolved_request(execution, **overrides):
    # The public schema refuses nonempty recipes. Only this synthetic internal
    # fixture represents intent already resolved by the private call boundary.
    return GenerateRequest(prompt="synthetic", steps=1, width=256, height=256).model_copy(
        update={"execution": execution, **overrides},
    )


def test_on_off_on_has_fresh_call_owners_and_state_without_reloading(request_runtime):
    runtime = request_runtime
    backend, state = runtime.backend, runtime.state
    generators, callbacks, scopes = [], [], []
    for index, choice in enumerate(("fbc-005-v1", None, "fbc-005-v1")):
        generator, callback = [object()], object()
        generators.append(generator)
        callbacks.append(callback)
        effective = resolve_execution(None, step_cache=choice)
        result, scope = _call(runtime, effective, generator=generator, callback_on_step_end=callback,
                              prompt=f"cell {index}", width=256 + index * 64)
        scopes.append(scope)
        assert result.images == ["synthetic image"]
        assert backend._pipe is runtime.pipe and backend.is_ready()
        assert backend._loaded_execution is None
        assert backend._cache_owner is None and backend._cache_disposal is None
        assert backend._pending_cache_disposal is None and runtime.control.releases == 0
        _clean_root(runtime.root)
        assert all(manager._state_cache == {} and manager._current_context is None for manager in state.managers)
    assert [cached for cached, _ in runtime.calls] == [True, False, True]
    assert all(kwargs["generator"] is generators[index]
               and kwargs["callback_on_step_end"] is callbacks[index]
               and kwargs["prompt"] == f"cell {index}"
               and kwargs["width"] == 256 + index * 64
               for index, (_, kwargs) in enumerate(runtime.calls))
    assert len(runtime.owners) == 2 and runtime.owners[0] is not runtime.owners[1]
    assert all(owner.closed and not owner.active and not owner.invalidated for owner in runtime.owners)
    assert len(state.configs) == len(state.managers) == 2
    assert state.configs[0] is not state.configs[1] and state.managers[0] is not state.managers[1]
    assert scopes[0].disposal is not scopes[2].disposal
    assert runtime.events.count("pretrained") == 1


@pytest.mark.parametrize("off", [None, {}, {"compilation": None, "attention_backend": None, "step_cache": None}])
def test_explicit_off_never_enters_cache_runtime_or_native_checks(request_runtime, monkeypatch, off):
    runtime = request_runtime

    def forbidden(*args, **kwargs):
        pytest.fail("an explicit Off call entered cache or native-attention preparation")

    monkeypatch.setattr(fbc, "_api", forbidden)
    monkeypatch.setattr(fbc, "FirstBlockCacheOwner", forbidden)
    monkeypatch.setattr(fbc, "require_native_attention", forbidden)
    result, _ = _call(runtime, off)
    assert result.images == ["synthetic image"] and runtime.calls[0][0] is False
    assert runtime.owners == runtime.state.configs == []
    assert runtime.events.count("pretrained") == 1


def test_call_intent_is_an_internal_snapshot_not_a_mutable_borrow(request_runtime):
    runtime = request_runtime
    supplied = {"step_cache": "fbc-005-v1"}
    with module._CallDisposal(runtime.backend, request_execution=supplied) as call:
        supplied["step_cache"] = None
        runtime.backend._invoke_pipeline(runtime.pipe, {"num_images_per_prompt": 1}, disposal=call)
    assert runtime.calls[0][0] is True and supplied == {"step_cache": None}
    assert runtime.owners[0].closed and runtime.backend._loaded_execution is None


def test_request_on_scope_allows_only_one_actual_forward(request_runtime):
    runtime = request_runtime
    with module._CallDisposal(runtime.backend, request_execution=ON) as call:
        runtime.backend._invoke_pipeline(runtime.pipe, {"num_images_per_prompt": 1}, disposal=call)
        with pytest.raises(RuntimeError, match="one pipeline call"):
            runtime.backend._invoke_pipeline(runtime.pipe, {"num_images_per_prompt": 1}, disposal=call)
        assert len(runtime.calls) == len(runtime.state.configs) == 1
    assert runtime.owners[0].closed and runtime.backend.is_ready()


@pytest.mark.parametrize("failure", [RuntimeError("synthetic forward failure"), Cancelled("synthetic cancellation")])
def test_clean_failed_or_cancelled_cache_call_allows_off_without_reload(request_runtime, failure):
    runtime = request_runtime
    runtime.failure = failure
    with pytest.raises(type(failure)) as caught:
        _call(runtime, ON)
    assert caught.value is failure and len(runtime.calls) == 1
    owner = runtime.owners[0]
    assert owner.closed and not owner.invalidated
    assert runtime.backend.is_ready() and runtime.control.releases == 0
    assert runtime.backend._cache_owner is None and runtime.backend._cache_disposal is None
    _clean_root(runtime.root)
    runtime.failure = None
    _call(runtime, None)
    _call(runtime, ON)
    assert [cached for cached, _ in runtime.calls] == [True, False, True]
    assert runtime.owners[1] is not owner and runtime.owners[1].closed
    assert runtime.events.count("pretrained") == 1


@pytest.mark.parametrize("primary", [None, RuntimeError("synthetic primary before native drift")])
def test_native_drift_after_request_cache_forward_is_not_returned_or_retried(request_runtime, primary):
    runtime = request_runtime
    runtime.failure = primary
    runtime.during_forward = lambda _: setattr(runtime.control, "native", False)
    with pytest.raises(ValueError if primary is None else RuntimeError) as caught:
        _call(runtime, ON)
    if primary is not None:
        assert caught.value is primary and "Native attention verification" in " ".join(primary.__notes__)
    assert len(runtime.calls) == 1 and runtime.owners[0].closed
    assert runtime.backend.is_ready() and runtime.control.releases == 0
    assert runtime.backend._cache_owner is None and runtime.backend._cache_disposal is None
    _clean_root(runtime.root)


@pytest.mark.parametrize("partial", ["before", "after"])
def test_partial_setup_is_cleaned_before_later_off_and_new_on(request_runtime, partial):
    runtime = request_runtime
    setattr(runtime.state, f"fail_{partial}", 1)
    with pytest.raises(RuntimeError) as caught:
        _call(runtime, ON)
    assert caught.value is runtime.state.setup_failure and runtime.calls == []
    assert runtime.owners[0].closed and not runtime.owners[0].invalidated
    assert runtime.backend.is_ready() and runtime.control.releases == 0
    _clean_root(runtime.root)
    setattr(runtime.state, f"fail_{partial}", None)
    _call(runtime, None)
    _call(runtime, ON)
    assert [cached for cached, _ in runtime.calls] == [False, True]
    assert runtime.events.count("pretrained") == 1


@pytest.mark.parametrize("conflict", ["cache", "named-hook", "stateful-hook"])
def test_foreign_state_is_refused_before_installation_and_left_untouched(request_runtime, conflict):
    runtime = request_runtime
    foreign = Hook()
    registry = None
    if conflict == "cache":
        runtime.root._cache_config = foreign
    else:
        foreign._is_stateful = conflict == "stateful-hook"
        registry = Registry.check_if_exists_or_initialize(runtime.root)
        registry.register_hook(foreign, HEAD if conflict == "named-hook" else "foreign")
        before = (dict(registry.hooks), tuple(registry._fn_refs), runtime.root.forward)
    with pytest.raises(ValueError):
        _call(runtime, ON)
    assert runtime.calls == runtime.state.configs == runtime.state.removed == []
    assert runtime.backend._pipe is runtime.pipe and runtime.control.releases == 0
    if registry is None:
        assert runtime.root._cache_config is foreign
    else:
        assert (registry.hooks, tuple(registry._fn_refs), runtime.root.forward) == before


def test_unrelated_stateless_hook_survives_on_off_on(request_runtime):
    runtime = request_runtime
    child = runtime.root.transformer_blocks[0]
    registry = Registry.check_if_exists_or_initialize(child)
    foreign = Hook()
    registry.register_hook(foreign, "unrelated")
    before = (child.forward, tuple(registry._fn_refs))
    for execution in (ON, None, ON):
        _call(runtime, execution)
        assert registry.hooks == {"unrelated": foreign}
        assert child.forward is before[0] and tuple(registry._fn_refs) == before[1]
        assert foreign not in runtime.state.removed
    assert len(runtime.owners) == 2 and all(owner.closed for owner in runtime.owners)


@pytest.mark.parametrize("execution", [
    ON.model_copy(update={"step_cache": "future"}),
    ON.model_copy(update={"unknown": None}),
    ExecutionSpec(compilation="regional-v1", step_cache="fbc-005-v1"),
    ExecutionSpec(attention_backend="cudnn-v1", step_cache="fbc-005-v1"),
    ExecutionSpec(compilation="regional-v1"),
    ExecutionSpec(attention_backend="cudnn-v1"),
])
def test_invalid_or_different_loader_intent_refuses_before_hooks_and_forward(request_runtime, execution):
    runtime = request_runtime
    with pytest.raises((ValueError, ValidationError)):
        _call(runtime, execution)
    assert runtime.calls == runtime.state.configs == [] and runtime.control.releases == 0
    assert runtime.backend._loaded_execution is None and runtime.backend._pipe is runtime.pipe


def test_recorded_off_cannot_borrow_a_different_loaded_loader_recipe(request_runtime):
    runtime = request_runtime
    runtime.backend._loaded_execution = ExecutionSpec(compilation="regional-v1")
    try:
        with pytest.raises(ValueError):
            _call(runtime, None)
        assert runtime.calls == runtime.state.configs == []
    finally:
        runtime.backend._loaded_execution = None  # Undo only synthetic fault injection.


@pytest.mark.parametrize("fault", ["batch", "adapter-zero", "true-cfg", "sibling", "wrong-root", "external", "native"])
def test_request_cache_boundary_refuses_unsupported_call_before_installation(request_runtime, monkeypatch, fault):
    runtime = request_runtime
    backend = runtime.backend
    runner = runtime.pipe
    kwargs = {"num_images_per_prompt": 1}
    if fault == "batch":
        kwargs["num_images_per_prompt"] = 2
    elif fault == "adapter-zero":
        backend._lora = ("synthetic adapter", 0.0)
    elif fault == "true-cfg":
        kwargs["true_cfg_scale"] = 2.0
    elif fault == "sibling":
        runner = object()
    elif fault == "wrong-root":
        runtime.pipe.transformer = FluxTransformer2DModel(runtime.state)
    elif fault == "external":
        monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", "fixed warning")
    else:
        runtime.control.native = False
    with pytest.raises((ValueError, RuntimeError)):
        with module._CallDisposal(backend, request_execution=ON) as call:
            backend._invoke_pipeline(runner, kwargs, disposal=call)
    assert runtime.calls == runtime.state.configs == [] and runtime.control.releases == 0
    assert backend.is_ready() and backend._loaded_execution is None


def test_explicit_off_refuses_legacy_load_owned_cache_while_omitted_intent_keeps_private_compatibility(cache_loader, monkeypatch):
    backend, _control, _events, state = cache_loader
    backend._load_prepared(ON)
    owner = backend._cache_owner
    calls = []

    def forward(pipe, **kwargs):
        assert owner.active
        calls.append(kwargs)
        return SimpleNamespace(images=["legacy synthetic image"])

    monkeypatch.setattr(type(backend._pipe), "__call__", forward)
    with pytest.raises((ValueError, RuntimeError)):
        with module._CallDisposal(backend, request_execution=None) as call:
            backend._invoke_pipeline(backend._pipe, {"num_images_per_prompt": 1}, disposal=call)
    assert calls == state.configs == [] and backend._cache_owner is owner
    with module._CallDisposal(backend) as call:
        backend._invoke_pipeline(backend._pipe, {"num_images_per_prompt": 1}, disposal=call)
    assert len(calls) == 1 and not owner.closed and not owner.active
    backend.unload()
    assert owner.closed


@pytest.mark.parametrize("off", [None, {}])
def test_outer_explicit_off_cannot_become_omitted_legacy_cache_intent(cache_loader, monkeypatch, off):
    backend, _control, _events, state = cache_loader
    backend._load_prepared(ON)
    owner = backend._cache_owner
    request = GenerateRequest(prompt="synthetic", execution=off)
    assert "execution" in request.model_fields_set and canonical_execution(request.execution) is None
    monkeypatch.setattr(backend, "_require_execution", lambda request: request)

    def forbidden(*args, **kwargs):
        pytest.fail("explicit outer Off entered the legacy cached generation body")

    monkeypatch.setattr(backend, "_generate_images_prepared", forbidden)
    with pytest.raises(ValueError, match="load-owned cache"):
        backend._generate_images(request, 1)
    assert backend._cache_owner is owner and not owner.closed and state.configs == []
    backend.unload()
    assert owner.closed


@pytest.mark.parametrize("cleanup", ["remove", "disable", "reset"])
def test_unsafe_cleanup_detaches_and_holds_disposal_until_outer_scope_exits(request_runtime, cleanup):
    runtime = request_runtime
    backend = runtime.backend
    backend._refine_pipe, backend._inpaint_pipe = object(), object()
    if cleanup == "reset":
        runtime.fail_reset = True
    else:
        setattr(runtime.state, f"{cleanup}_failure", RuntimeError(f"synthetic {cleanup} failure"))
    with module._CallDisposal(backend, request_execution=ON) as call:
        with pytest.raises(fbc.CacheCleanupFailed):
            backend._invoke_pipeline(runtime.pipe, {"num_images_per_prompt": 1}, disposal=call)
        owner = runtime.owners[0]
        assert owner.invalidated and not owner.active and not owner.closed
        assert call.pending and backend._pending_cache_disposal is call.disposal
        assert not backend.is_ready() and backend._pipe is None and backend._cache_owner is None
        assert backend._refine_pipe is None and backend._inpaint_pipe is None
        assert runtime.control.releases == 0 and len(runtime.calls) == 1
        with pytest.raises(RuntimeError, match="pending cache disposal"):
            backend.load()
        with pytest.raises(RuntimeError, match="pending cache disposal"):
            _call(runtime, None)
    assert owner.closed and call.disposal.done and backend._pending_cache_disposal is None
    assert runtime.control.releases == 1 and len(runtime.calls) == 1
    call.disposal.drain()
    assert runtime.control.releases == 1
    with pytest.raises((ValueError, RuntimeError)):
        _call(runtime, None)
    assert len(runtime.calls) == 1  # No implicit forward or generation-time reload.
    with pytest.raises(fbc.CacheCleanupFailed, match="discarded"):
        fbc.FirstBlockCacheOwner(runtime.root, discard_shared=lambda: None)
    runtime.control.root = FluxTransformer2DModel(runtime.state)
    backend.load()  # Stand-in for the next ordinary ensure_loaded recovery.
    runtime.pipe, runtime.root = backend._pipe, runtime.control.root
    _call(runtime, None)
    assert runtime.calls[-1][0] is False and runtime.events.count("pretrained") == 2


@pytest.mark.parametrize("failure", [RuntimeError("synthetic primary"), Cancelled("synthetic primary cancellation")])
def test_primary_error_and_cleanup_release_notes_survive_without_retry(request_runtime, failure):
    runtime = request_runtime
    runtime.failure, runtime.fail_reset = failure, True
    runtime.control.release_error = OSError("synthetic allocator release failure")
    with pytest.raises(type(failure)) as caught:
        _call(runtime, ON)
    assert caught.value is failure and len(runtime.calls) == 1
    assert runtime.owners[0].closed and runtime.owners[0].invalidated
    assert "reset failed" in " ".join(failure.__notes__)
    assert "allocator release failure" in " ".join(failure.__notes__)
    assert runtime.control.releases == 1 and not runtime.backend.is_ready()
    assert runtime.backend._pending_cache_disposal is None


@pytest.mark.parametrize("primary", [None, RuntimeError("synthetic outer body failure")])
def test_owner_close_failure_still_detaches_expected_pipeline_and_preserves_foreign_state(request_runtime, primary):
    runtime = request_runtime
    foreign = object()
    with pytest.raises(ValueError if primary is None else RuntimeError) as caught:
        with module._CallDisposal(runtime.backend, request_execution=ON) as call:
            runtime.backend._invoke_pipeline(runtime.pipe, {"num_images_per_prompt": 1}, disposal=call)
            # Inject a foreign replacement only after cached_call's own
            # successful cleanup. close/assert_off must still own detachment.
            runtime.root._cache_config = foreign
            if primary is not None:
                raise primary
    if primary is not None:
        assert caught.value is primary and "Closing the request cache owner" in " ".join(primary.__notes__)
    assert runtime.root._cache_config is foreign
    assert runtime.owners[0].invalidated and runtime.owners[0].closed
    assert not runtime.backend.is_ready() and runtime.backend._pending_cache_disposal is None
    assert runtime.control.releases == len(runtime.calls) == 1
    assert call.disposal.done


def test_close_and_discard_failure_cannot_skip_expected_owner_detachment(request_runtime, monkeypatch):
    runtime = request_runtime
    primary = RuntimeError("synthetic outer primary")
    close = fbc.FirstBlockCacheOwner.close
    attempts, drain_observations = [], []
    drain = module._CacheDisposal.drain

    def fail_initial_closes(owner):
        attempts.append(owner)
        if len(attempts) <= 2:
            raise RuntimeError("synthetic owner close failure")
        return close(owner)

    def observe_drain(disposal):
        drain_observations.append((not runtime.backend.is_ready(),
                                   runtime.backend._pending_cache_disposal is disposal,
                                   disposal.pending is not None, runtime.control.releases))
        return drain(disposal)

    monkeypatch.setattr(fbc.FirstBlockCacheOwner, "close", fail_initial_closes)
    monkeypatch.setattr(module._CacheDisposal, "drain", observe_drain)
    with pytest.raises(RuntimeError) as caught:
        with module._CallDisposal(runtime.backend, request_execution=ON) as call:
            runtime.backend._invoke_pipeline(runtime.pipe, {"num_images_per_prompt": 1}, disposal=call)
            raise primary
    assert caught.value is primary
    notes = " ".join(primary.__notes__)
    assert "Closing the request cache owner" in notes and "Discarding the request cache owner" in notes
    assert drain_observations == [(True, True, True, 0)]
    assert len(attempts) == 3 and runtime.owners[0].closed and runtime.owners[0].invalidated
    assert not runtime.backend.is_ready() and runtime.backend._pending_cache_disposal is None
    assert runtime.control.releases == len(runtime.calls) == 1


def test_previous_request_callback_cannot_detach_a_later_owner_or_replacement(request_runtime):
    runtime = request_runtime
    _, first = _call(runtime, ON)
    _, second = _call(runtime, ON)
    assert first.disposal is not second.disposal
    first.disposal.detach()
    assert runtime.backend._pipe is runtime.pipe and runtime.backend.is_ready()
    runtime.backend.unload()
    runtime.control.root = FluxTransformer2DModel(runtime.state)
    runtime.backend.load()
    runtime.pipe, runtime.root = runtime.backend._pipe, runtime.control.root
    first.disposal.detach()
    second.disposal.detach()
    assert runtime.backend._pipe is runtime.pipe and runtime.backend._pending_cache_disposal is None
    _call(runtime, None)


def _prepare_generation(runtime, monkeypatch):
    backend, control = runtime.backend, runtime.control
    generated = []

    class Generator:
        def __init__(self, *, device):
            assert device == "cpu"
            self.seed = None
            generated.append(self)

        def manual_seed(self, seed):
            self.seed = seed
            return self

    @contextmanager
    def context(name):
        setattr(control, name, True)
        try:
            yield
        finally:
            setattr(control, name, False)

    sys.modules["torch"].Generator = Generator
    sys.modules["torch"].no_grad = lambda: context("grad")
    monkeypatch.setattr(module.schedule, "requested_shift", lambda *_: (None, None))
    monkeypatch.setattr(module.schedule, "forced_mu", lambda *_: context("sigma"))
    # This isolated synthetic test supplies already-resolved intent directly;
    # the real public admission guards are tested closed below.
    monkeypatch.setattr(backend, "_require_execution", lambda request: request)
    monkeypatch.setattr(backend, "applied", lambda request: request)
    monkeypatch.setattr(backend, "_apply_lora", lambda *_: None)
    backend._call_params = {"prompt", "width", "height", "num_inference_steps", "guidance_scale",
                            "num_images_per_prompt", "generator", "callback_on_step_end"}
    return generated


def test_actual_generation_keeps_seed_callback_and_disposes_after_no_grad_and_sigma(request_runtime, monkeypatch):
    runtime = request_runtime
    backend, control = runtime.backend, runtime.control
    generated = _prepare_generation(runtime, monkeypatch)

    def check_callback(kwargs):
        assert control.grad and control.sigma
        assert kwargs["generator"] == [generated[-1]]
        sentinel = {"unchanged": object()}
        assert kwargs["callback_on_step_end"](runtime.pipe, 0, None, sentinel) is sentinel

    runtime.during_forward = check_callback
    for seed, execution in ((41, ON), (42, None), (43, ON)):
        result = backend._generate_images(_resolved_request(execution), seed)
        assert result == ["synthetic image"] and generated[-1].seed == seed
        assert backend._loaded_execution is None and backend._pipe is runtime.pipe
    assert [cached for cached, _ in runtime.calls] == [True, False, True]
    assert len({id(generator) for generator in generated}) == 3
    assert runtime.events.count("pretrained") == 1 and control.releases == 0
    runtime.state.remove_failure = RuntimeError("synthetic outer cleanup failure")
    with pytest.raises(fbc.CacheCleanupFailed):
        backend._generate_images(_resolved_request(ON), 44)
    assert control.release_observations == [(False, False, False, None, None, None)]
    assert runtime.owners[-1].closed and not backend.is_ready()
    assert backend._pending_cache_disposal is None and len(runtime.calls) == 4


def test_retained_primary_and_late_close_failure_drop_application_aliases_before_drain(request_runtime, monkeypatch):
    runtime = request_runtime
    _prepare_generation(runtime, monkeypatch)
    primary = runtime.failure = RuntimeError("synthetic retained forward primary")
    close, drain = fbc.FirstBlockCacheOwner.close, module._CacheDisposal.drain
    close_attempts, observations = [], []

    def fail_first_close(owner):
        close_attempts.append(owner)
        if len(close_attempts) == 1:
            raise ValueError("synthetic late close verification failure")
        return close(owner)

    def observe_drain(disposal):
        frames = {}
        traceback = primary.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            name = frame.f_code.co_name
            if name in {"_invoke_pipeline", "_invoke_pipeline_prepared", "_generate_images_prepared"}:
                frames[name] = (frame.f_locals.get("runner") is None,
                                frame.f_locals.get("pipe") is None)
            traceback = traceback.tb_next
        observations.append((frames, runtime.control.grad, runtime.control.sigma,
                             runtime.backend._lock.locked(), runtime.control.releases,
                             not runtime.backend.is_ready(), disposal.pending is not None))
        return drain(disposal)

    monkeypatch.setattr(fbc.FirstBlockCacheOwner, "close", fail_first_close)
    monkeypatch.setattr(module._CacheDisposal, "drain", observe_drain)
    with pytest.raises(RuntimeError) as caught:
        runtime.backend._generate_images(_resolved_request(ON), 41)
    assert caught.value is primary
    assert "late close verification failure" in " ".join(primary.__notes__)
    assert observations == [({"_invoke_pipeline": (True, True), "_invoke_pipeline_prepared": (True, True),
                             "_generate_images_prepared": (True, True)}, False, False, False, 0, True, True)]
    assert len(runtime.calls) == runtime.control.releases == 1
    assert runtime.owners[0].closed and not runtime.backend.is_ready()


@pytest.mark.parametrize("overrides", [
    {"num_images": 2},
    {"refine": 1},
    {"refine": 1, "mask": "a" * 64},
    {"reference": [1]},
    {"lora": "synthetic adapter", "lora_scale": 0.0},
    {"true_cfg": 2.0, "negative_prompt": "synthetic unconditioned prompt"},
])
def test_outer_request_refuses_batch_and_unsupported_pass_before_any_preparation(request_runtime, monkeypatch, overrides):
    runtime = request_runtime
    backend = runtime.backend

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported cache request reached adapter, sibling, input, hooks or forward preparation")

    # Supply private resolved intent without opening any real schema/admission
    # gate. In particular, validate the original requested batch before a later
    # pass could split it into individually cached one-image pipeline calls.
    monkeypatch.setattr(backend, "_require_execution", lambda request: request)
    monkeypatch.setattr(backend, "applied", lambda request: request)
    for name in ("_apply_lora", "_refiner", "_inpainter", "_reference_images", "_invoke_pipeline"):
        monkeypatch.setattr(backend, name, forbidden)
    monkeypatch.setattr(fbc, "_api", forbidden)
    monkeypatch.setattr(module.schedule, "requested_shift", forbidden)
    monkeypatch.setattr(module.schedule, "forced_mu", forbidden)
    sys.modules["torch"].Generator = forbidden
    sys.modules["torch"].no_grad = forbidden
    with pytest.raises(ValueError):
        backend._generate_images(_resolved_request(ON, **overrides), 41)
    assert runtime.calls == runtime.owners == runtime.state.configs == []
    assert backend._loaded_execution is None and backend._pipe is runtime.pipe
    assert backend.is_ready() and runtime.events.count("pretrained") == 1
    assert backend._cache_owner is None and backend._cache_disposal is None
    assert runtime.control.releases == 0


def test_request_context_does_not_open_any_public_nonempty_admission(request_runtime, monkeypatch):
    runtime = request_runtime

    def forbidden(*args, **kwargs):
        pytest.fail("a public nonempty request reached peak/device or pipeline work")

    monkeypatch.setattr(runtime.backend, "_peak_device", forbidden)
    request = _resolved_request(ON)
    for invoke in (lambda: runtime.backend.applied(request),
                   lambda: runtime.backend.generate(request, 1),
                   lambda: runtime.backend._generate_images(request, 1)):
        with pytest.raises(ValueError, match="not available"):
            invoke()
    assert runtime.calls == runtime.owners == runtime.state.configs == []
    assert runtime.backend.is_ready() and runtime.events.count("pretrained") == 1
