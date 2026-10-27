"""Synthetic Native-environment policy contract; no real dispatcher imports.

Private runtime contract. Status/candidate validation and
Eager stay cold. Startup/current/frozen import observations must all be Native;
the guard never repairs user environment, registry state or processor fields.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import ModuleType, SimpleNamespace

import pytest

from backend import config
from backend.backends import attention_selection as selection
from backend.backends.base import Cancelled
from backend.backends import diffusers_backend as backend_module
from backend.backends import fbc_cache as fbc
from backend.execution import CompilerMetadata, ExecutionSpec, attention_candidate_reason, cache_candidate_reason
from tests.test_pipeline_ownership import synthetic_loader
from tests.test_fbc_backend import cache_loader as cache_loader
from tests.test_fbc_cache import world as world
from tests.test_fbc_request import _call, request_runtime as request_runtime


REASON = "Private execution requires a Native attention startup, environment and imported dispatcher baseline."
VARIABLE = "DIFFUSERS_ATTN_BACKEND"
FACTS = CompilerMetadata("win32", (3, 13), "2.13.0+cu130", "0.40.0", "0.18.0", None, None)
ENVIRONMENT_GUARD = selection.require_native_environment
DISPATCH_GUARD = selection.require_native_dispatch
PRIVATE_RECIPES = [ExecutionSpec(compilation="regional-v1"), ExecutionSpec(attention_backend="cudnn-v1"),
                   ExecutionSpec(step_cache="fbc-005-v1")]


@pytest.fixture
def native_environment(monkeypatch):
    # Other ownership fixtures supply no dispatcher. Restore the actual policy
    # entry points here, then satisfy their imports only with the stubs below.
    monkeypatch.setattr(selection, "require_native_environment", ENVIRONMENT_GUARD)
    monkeypatch.setattr(selection, "require_native_dispatch", DISPATCH_GUARD)
    blocked = {"torch", "torchao", "diffusers", "triton", "huggingface_hub"}
    class SyntheticOnly(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".", 1)[0] in blocked:
                raise AssertionError("attention policy attempted a real runtime import")
            return None
    # Existing ordinary conftest availability imports must not make a new
    # accidental body-side runtime import succeed from sys.modules either.
    for name in tuple(sys.modules):
        if name.split(".", 1)[0] in blocked:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [SyntheticOnly(), *sys.meta_path])
    state = SimpleNamespace(reads=[], writes=0, getter_calls=0, getter_error=None,
                            contexts=0, registry_writes=0, processor_writes=0)
    class Environment(dict):
        def get(self, key, default=None):
            state.reads.append(key)
            return super().get(key, default)
        def __setitem__(self, key, value):
            state.writes += 1
            raise AssertionError("the attention guard must not write the environment")
        def __delitem__(self, key):
            state.writes += 1
            raise AssertionError("the attention guard must not clear the environment")
    environment = Environment()

    class Backend(str, Enum):
        NATIVE = "native"
        _NATIVE_CUDNN = "_native_cudnn"

    def native_function(*args, **kwargs):
        raise AssertionError("the policy guard must not execute attention")

    def foreign_function(*args, **kwargs):
        raise AssertionError("the policy guard must not execute foreign attention")

    class Registry:
        _active_backend = Backend.NATIVE
        _backends = {Backend.NATIVE: native_function, Backend._NATIVE_CUDNN: foreign_function}
        @classmethod
        def get_active_backend(cls):
            state.getter_calls += 1
            if state.getter_error is not None:
                raise state.getter_error
            return cls._active_backend, cls._backends[cls._active_backend]
        @classmethod
        def set_active_backend(cls, value):
            raise AssertionError("the policy guard must not set the global dispatcher")

    @contextmanager
    def owned_context(value):
        state.contexts += 1
        prior = Registry._active_backend
        Registry._active_backend = Backend(value)
        state.registry_writes += 1
        try:
            yield
        finally:
            Registry._active_backend = prior
            state.registry_writes += 1

    class Processor:
        _attention_backend = None
        _parallel_config = None
        def __setattr__(self, key, value):
            if key == "_attention_backend":
                state.processor_writes += 1
            object.__setattr__(self, key, value)
        def __delattr__(self, key):
            if key == "_attention_backend":
                state.processor_writes += 1
            object.__delattr__(self, key)

    class Attention:
        def __init__(self):
            self.processor = Processor()

    class Joint:
        def __init__(self):
            self.attn = Attention()

    class Single:
        def __init__(self):
            self.attn = Attention()

    class Transformer:
        def __init__(self):
            self.config = {"num_layers": 1, "num_single_layers": 1}
            self.transformer_blocks, self.single_transformer_blocks = [Joint()], [Single()]
            self.calls = []
            self.after_set = None
        @property
        def attn_processors(self):
            return {f"{name}.{index}.attn.processor": block.attn.processor
                    for name in ("transformer_blocks", "single_transformer_blocks")
                    for index, block in enumerate(getattr(self, name))}
        def set_attention_backend(self, backend):
            self.calls.append(backend)
            for processor in self.attn_processors.values():
                processor._attention_backend = Backend(backend)
            # The owned setter's incidental change is restored by its existing
            # public context. The new baseline guard itself never calls it.
            Registry._active_backend = Backend(backend)
            state.registry_writes += 1
            if self.after_set is not None:
                self.after_set()

    packages = {}
    for name in ("diffusers", "diffusers.models", "diffusers.utils"):
        packages[name] = ModuleType(name)
        packages[name].__path__ = []
        monkeypatch.setitem(sys.modules, name, packages[name])
    constants = ModuleType("diffusers.utils.constants")
    constants.DIFFUSERS_ATTN_BACKEND = "native"
    dispatch = ModuleType("diffusers.models.attention_dispatch")
    dispatch.DIFFUSERS_ATTN_BACKEND = "native"
    dispatch.AttentionBackendName = Backend
    dispatch._AttentionBackendRegistry = Registry
    dispatch._native_attention = native_function
    dispatch.attention_backend = owned_context
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setitem(sys.modules, dispatch.__name__, dispatch)
    monkeypatch.setattr(config, "ATTENTION_ENVIRONMENT_WARNING", None)
    monkeypatch.setattr(selection, "os", SimpleNamespace(
        environ=environment, getenv=lambda key, default=None: environment.get(key, default),
    ), raising=False)
    bindings = selection._Bindings(Transformer, Joint, Single, Attention, Processor, owned_context, Backend._NATIVE_CUDNN)
    monkeypatch.setattr(selection, "_load_bindings", lambda: bindings)
    root = Transformer()
    return SimpleNamespace(state=state, environment=environment, constants=constants, dispatch=dispatch,
                           registry=Registry, backend=Backend, native=native_function, foreign=foreign_function,
                           root=root, bindings=bindings)


def refuse(action):
    with pytest.raises(ValueError) as caught:
        action()
    assert str(caught.value) == REASON
    assert "PRIVATE_ATTENTION_PAYLOAD" not in "".join(traceback.format_exception(caught.value))
    return caught.value


@pytest.mark.parametrize("current", [None, "native"])
def test_absent_and_exact_native_are_equivalent_read_only_observations(native_environment, current):
    world = native_environment
    if current is not None:
        dict.__setitem__(world.environment, VARIABLE, current)
    before = (dict(world.environment), world.constants.DIFFUSERS_ATTN_BACKEND,
              world.dispatch.DIFFUSERS_ATTN_BACKEND, world.registry._active_backend)
    selection.require_native_environment()
    selection.require_native_dispatch()
    selection.require_native_attention(world.root)
    assert before == (dict(world.environment), world.constants.DIFFUSERS_ATTN_BACKEND,
                      world.dispatch.DIFFUSERS_ATTN_BACKEND, world.registry._active_backend)
    assert world.state.writes == world.state.contexts == world.state.registry_writes == world.state.processor_writes == 0
    assert world.root.calls == []


@pytest.mark.parametrize("raw", ["", "Native", "NATIVE", " native", "native ", "_native_cudnn", "PRIVATE_ATTENTION_PAYLOAD" * 100])
def test_current_override_is_rejected_without_echo_mutation_or_lazy_binding(native_environment, monkeypatch, raw):
    world = native_environment
    dict.__setitem__(world.environment, VARIABLE, raw)
    monkeypatch.setattr(selection, "_load_bindings", lambda: pytest.fail("nonnative environment reached model binding"))
    before = dict(world.environment)
    refuse(selection.require_native_environment)
    refuse(selection.require_native_dispatch)
    assert dict(world.environment) == before and world.state.writes == 0
    assert world.state.getter_calls == world.state.contexts == world.state.registry_writes == 0


def test_startup_warning_cannot_be_cleared_by_a_later_native_environment(native_environment, monkeypatch):
    world = native_environment
    monkeypatch.setattr(config, "ATTENTION_ENVIRONMENT_WARNING", "fixed startup override observation")
    dict.__setitem__(world.environment, VARIABLE, "native")
    monkeypatch.setattr(selection, "_load_bindings", lambda: pytest.fail("startup refusal reached model binding"))
    refuse(selection.require_native_environment)
    refuse(selection.require_native_dispatch)
    assert world.state.writes == world.state.getter_calls == 0


@pytest.mark.parametrize("source", ["constants", "dispatch"])
@pytest.mark.parametrize("raw", ["", "NATIVE", "native ", "_native_cudnn", "PRIVATE_ATTENTION_PAYLOAD", None])
def test_frozen_import_aliases_refuse_even_when_current_env_and_global_dispatch_are_native(native_environment, source, raw):
    world = native_environment
    getattr(world, source).DIFFUSERS_ATTN_BACKEND = raw
    before = (dict(world.environment), world.registry._active_backend, world.state.registry_writes)
    refuse(selection.require_native_dispatch)
    assert (dict(world.environment), world.registry._active_backend, world.state.registry_writes) == before
    assert world.state.contexts == world.state.processor_writes == 0


@pytest.mark.parametrize("source", ["constants", "dispatch"])
def test_missing_frozen_import_observation_is_uncertain_not_implicitly_native(native_environment, source):
    delattr(getattr(native_environment, source), "DIFFUSERS_ATTN_BACKEND")
    refuse(selection.require_native_dispatch)


@pytest.mark.parametrize("fault", ["global", "missing-function", "foreign-callable", "missing-native-export", "getter-error"])
def test_live_global_dispatch_must_be_the_verified_native_enum_and_function(native_environment, fault):
    world = native_environment
    if fault == "global":
        world.registry._active_backend = world.backend._NATIVE_CUDNN
    elif fault == "missing-function":
        world.registry._backends[world.backend.NATIVE] = None
    elif fault == "foreign-callable":
        world.registry._backends[world.backend.NATIVE] = world.foreign
    elif fault == "missing-native-export":
        del world.dispatch._native_attention
    else:
        world.state.getter_error = RuntimeError("PRIVATE_ATTENTION_PAYLOAD")
    before = (world.registry._active_backend, dict(world.registry._backends))
    refuse(selection.require_native_dispatch)
    assert before == (world.registry._active_backend, world.registry._backends)
    assert world.state.writes == world.state.contexts == world.state.registry_writes == 0


def test_absence_native_drift_is_permitted_but_non_native_drift_is_not_repaired(native_environment):
    world = native_environment
    selection.require_native_dispatch()
    dict.__setitem__(world.environment, VARIABLE, "native")
    selection.require_native_dispatch()
    dict.__delitem__(world.environment, VARIABLE)
    selection.require_native_dispatch()
    dict.__setitem__(world.environment, VARIABLE, "PRIVATE_ATTENTION_PAYLOAD")
    refuse(selection.require_native_dispatch)
    assert world.environment[VARIABLE] == "PRIVATE_ATTENTION_PAYLOAD" and world.state.writes == 0


@pytest.mark.parametrize("drift", ["environment", "global", "constants", "dispatch"])
def test_owned_cudnn_cleanup_is_not_gated_by_the_new_activation_policy(native_environment, drift):
    world = native_environment
    owner = selection.configure_attention(world.root, "cudnn-v1")
    assert owner is not None
    try:
        owner.require(world.root)
        assert all(p._attention_backend == world.backend._NATIVE_CUDNN for p in world.root.attn_processors.values())
        if drift == "environment":
            dict.__setitem__(world.environment, VARIABLE, "PRIVATE_ATTENTION_PAYLOAD")
        elif drift == "global":
            world.registry._active_backend = world.backend._NATIVE_CUDNN
        else:
            getattr(world, drift).DIFFUSERS_ATTN_BACKEND = "PRIVATE_ATTENTION_PAYLOAD"
        refuse(lambda: owner.require(world.root))
        before = (dict(world.environment), world.registry._active_backend, world.state.contexts,
                  world.state.registry_writes, len(world.state.reads), world.state.getter_calls)
        owner.restore()
        assert owner.closed
        assert all(p._attention_backend is None for p in world.root.attn_processors.values())
        assert before == (dict(world.environment), world.registry._active_backend, world.state.contexts,
                          world.state.registry_writes, len(world.state.reads), world.state.getter_calls)
    finally:
        if not owner.closed:
            owner.discard()


def test_policy_drift_during_owned_setup_refuses_return_and_rolls_back_without_repairing_environment(native_environment):
    world = native_environment
    world.root.after_set = lambda: dict.__setitem__(world.environment, VARIABLE, "PRIVATE_ATTENTION_PAYLOAD")
    refuse(lambda: selection.configure_attention(world.root, "cudnn-v1"))
    assert all(p._attention_backend is None for p in world.root.attn_processors.values())
    assert world.registry._active_backend == world.backend.NATIVE
    assert world.environment[VARIABLE] == "PRIVATE_ATTENTION_PAYLOAD" and world.state.writes == 0


@pytest.mark.parametrize("primary", [RuntimeError("synthetic forward failure"), Cancelled("synthetic cancellation")])
def test_post_forward_environment_failure_preserves_primary_and_allows_owned_cleanup(native_environment, monkeypatch, primary):
    # This is the existing private attention call over a synthetic pipeline;
    # no tensors, imported dispatcher implementation or model forward exists.
    world = native_environment
    cause = ValueError("original synthetic forward cause")
    primary.__cause__ = cause
    owner = selection.configure_attention(world.root, "cudnn-v1")
    backend = backend_module.DiffusersBackend()
    calls, releases = [], []
    class Pipeline:
        transformer = world.root
        def __call__(self, **kwargs):
            calls.append(dict(kwargs))
            dict.__setitem__(world.environment, VARIABLE, "PRIVATE_ATTENTION_PAYLOAD")
            raise primary
    pipe = Pipeline()
    backend._pipe, backend._loaded_execution, backend._attention_selection = pipe, ExecutionSpec(attention_backend="cudnn-v1"), owner
    monkeypatch.setattr(backend_module, "ATTENTION_ENVIRONMENT_WARNING", None)
    monkeypatch.setattr(backend, "_release", lambda: releases.append(backend._lock.locked()))
    with pytest.raises(type(primary)) as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert caught.value is primary
    assert primary.__cause__ is cause
    assert REASON in " ".join(primary.__notes__)
    assert "PRIVATE_ATTENTION_PAYLOAD" not in " ".join(primary.__notes__)
    assert calls == [{"num_images_per_prompt": 1}]
    backend.unload()
    assert releases == [False] and owner.closed and backend._pipe is None
    assert all(p._attention_backend is None for p in world.root.attn_processors.values())
    assert world.environment[VARIABLE] == "PRIVATE_ATTENTION_PAYLOAD" and world.state.writes == 0


@pytest.mark.parametrize("recipe", PRIVATE_RECIPES, ids=["regional", "attention", "cache"])
@pytest.mark.parametrize("startup", [False, True], ids=["current", "startup"])
def test_private_preparation_refuses_override_before_lazy_binding_or_weights(
    synthetic_loader, native_environment, monkeypatch, recipe, startup,
):
    backend, control, events = synthetic_loader
    world = native_environment
    if startup:
        monkeypatch.setattr(config, "ATTENTION_ENVIRONMENT_WARNING", "fixed startup observation")
    else:
        dict.__setitem__(world.environment, VARIABLE, "PRIVATE_ATTENTION_PAYLOAD")
    monkeypatch.setattr(selection, "_load_bindings", lambda: pytest.fail("override reached a runtime binding"))
    refuse(lambda: backend._load_prepared(recipe))
    assert not backend.is_ready() and events == [] and control.releases == 0
    assert world.state.getter_calls == world.state.writes == 0


@pytest.mark.parametrize("recipe", PRIVATE_RECIPES, ids=["regional", "attention", "cache"])
@pytest.mark.parametrize("fault", ["constants", "dispatch", "global"])
def test_private_preparation_refuses_imported_baseline_before_weights(
    synthetic_loader, native_environment, monkeypatch, recipe, fault,
):
    backend, _control, events = synthetic_loader
    world = native_environment
    selected = replace(backend_module.settings, model_id="black-forest-labs/FLUX.1-Krea-dev",
                       quantization="fp8", offload="model", device="cuda")
    monkeypatch.setattr(backend_module, "current_settings", lambda _: selected)
    monkeypatch.setattr(backend_module, "compiler_metadata", lambda: replace(FACTS, triton_windows="3.7.1.post27"))
    monkeypatch.setattr(backend, "_require_compiler_cache", lambda *args: None)
    monkeypatch.setattr(backend_module, "ATTENTION_ENVIRONMENT_WARNING", None)
    if fault == "global":
        world.registry._active_backend = world.backend._NATIVE_CUDNN
    else:
        getattr(world, fault).DIFFUSERS_ATTN_BACKEND = "PRIVATE_ATTENTION_PAYLOAD"
    refuse(lambda: backend._load_prepared(recipe))
    assert not backend.is_ready() and "weights-enter" not in events and "pretrained" not in events
    assert world.state.contexts == world.state.writes == world.state.processor_writes == 0


@pytest.mark.parametrize("recipe", PRIVATE_RECIPES, ids=["regional", "attention", "cache"])
def test_repeated_private_load_does_not_accept_a_later_imported_baseline_change(
    native_environment, monkeypatch, recipe,
):
    world = native_environment
    backend = backend_module.DiffusersBackend()
    pipe = object()
    backend._pipe, backend._loaded_execution = pipe, recipe
    monkeypatch.setattr(backend, "_require_compiler_cache", lambda *args: None)
    world.dispatch.DIFFUSERS_ATTN_BACKEND = "PRIVATE_ATTENTION_PAYLOAD"
    refuse(lambda: backend._load_prepared(recipe))
    assert backend._pipe is pipe and backend._loaded_execution is recipe
    assert world.state.writes == world.state.contexts == 0


@pytest.mark.parametrize("outcome", ["returned", "failed", "cancelled"])
def test_request_cache_on_environment_drift_preserves_cleanup_and_exact_primary(
    request_runtime, native_environment, monkeypatch, outcome,
):
    runtime = request_runtime
    world = native_environment
    # The existing cache fixture owns its exact synthetic hook layout; supply
    # the new real baseline guard without claiming that its root is this fake
    # processor fixture's different exact transformer class.
    monkeypatch.setattr(fbc, "require_native_attention", lambda _: selection.require_native_dispatch())
    runtime.during_forward = lambda _: dict.__setitem__(world.environment, VARIABLE, "PRIVATE_ATTENTION_PAYLOAD")
    primary = None if outcome == "returned" else (
        RuntimeError("synthetic cache failure") if outcome == "failed" else Cancelled("synthetic cache cancellation")
    )
    runtime.failure = primary
    with pytest.raises(ValueError if primary is None else type(primary)) as caught:
        _call(runtime, ExecutionSpec(step_cache="fbc-005-v1"))
    if primary is None:
        assert str(caught.value) == REASON
    else:
        assert caught.value is primary and REASON in " ".join(primary.__notes__)
    assert len(runtime.calls) == len(runtime.owners) == len(runtime.state.configs) == 1
    assert runtime.owners[0].closed and not runtime.owners[0].active
    assert runtime.backend._cache_owner is None and runtime.backend._cache_disposal is None
    assert runtime.backend._pending_cache_disposal is None and runtime.backend.is_ready()
    assert runtime.root._cache_config is None and runtime.state.disable_calls == 1
    assert world.environment[VARIABLE] == "PRIVATE_ATTENTION_PAYLOAD" and world.state.writes == 0


def test_none_and_eager_never_enter_runtime_environment_policy(synthetic_loader, monkeypatch):
    backend, control, _events = synthetic_loader
    def forbidden(*args, **kwargs):
        pytest.fail("Eager/None entered experimental attention policy")
    monkeypatch.setattr(selection, "require_native_environment", forbidden)
    monkeypatch.setattr(selection, "require_native_dispatch", forbidden)
    assert selection.configure_attention(object(), None) is None
    backend.load()
    assert backend.is_ready() and backend.execution_diagnostics() is None
    backend.unload()
    assert control.releases == 1


def test_cold_candidate_checks_do_not_read_runtime_environment_or_imported_dispatch(native_environment, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("cold validation entered runtime attention policy")
    monkeypatch.setattr(selection, "require_native_environment", forbidden)
    monkeypatch.setattr(selection, "require_native_dispatch", forbidden)
    values = dict(model_id="black-forest-labs/FLUX.1-Krea-dev", quantization="fp8", offload="model", device="cuda")
    assert attention_candidate_reason(ExecutionSpec(attention_backend="cudnn-v1"), FACTS, **values) is None
    assert cache_candidate_reason(ExecutionSpec(step_cache="fbc-005-v1"), FACTS, **values) is None
    assert native_environment.state.reads == [] and native_environment.state.getter_calls == 0


def test_cold_status_and_none_configuration_stay_runtime_import_and_environment_read_free(tmp_path):
    script = r'''
import importlib.abc
import os
import sys
blocked = {"torch", "torchao", "diffusers", "triton", "huggingface_hub"}
class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError("cold status imported a runtime")
        return None
sys.meta_path.insert(0, ColdOnly())
from backend.backends import attention_selection
assert attention_selection.configure_attention(object(), None) is None
assert "backend.config" not in sys.modules
from backend import config, model_selection
from backend.execution import ExecutionSpec
def forbidden(*args, **kwargs):
    raise AssertionError("cold status reread the environment or entered runtime policy")
attention_selection.require_native_environment = forbidden
attention_selection.require_native_dispatch = forbidden
saved_getenv = os.getenv
saved_environ = os.environ
class NoEnvironmentReads(dict):
    get = forbidden
    __getitem__ = forbidden
    __contains__ = forbidden
os.getenv = forbidden
os.environ = NoEnvironmentReads()
try:
    selected = model_selection.execution_status(config.settings)
    assert selected["execution"] is None and selected["execution_available"]
    model_selection.execution_options(config.settings, execution=ExecutionSpec(attention_backend="cudnn-v1"))
finally:
    os.getenv = saved_getenv
    os.environ = saved_environ
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
'''
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    environment.pop("DIFFUSERS_ATTN_BACKEND", None)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
