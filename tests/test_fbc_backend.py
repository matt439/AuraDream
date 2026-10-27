"""Private cache lifecycle with synthetic pipelines/hooks and no real runtime."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.backends import Cancelled
from backend.backends import attention_selection as attention
from backend.backends import diffusers_backend as module
from backend.backends import fbc_cache as fbc
from backend.execution import CompilerMetadata, ExecutionSpec
from tests.test_attention_selection import (
    Attention, Joint, Processor, Single, Transformer, backend_context,
)
from tests.test_fbc_cache import (
    HEAD, FluxTransformer2DModel, Hook, Registry, assert_clean, world,
)
from tests.test_pipeline_ownership import synthetic_loader


SPEC = ExecutionSpec(step_cache="fbc-005-v1")
FACTS = CompilerMetadata("win32", (3, 13), "2.13.0+cu130", "0.40.0", "0.18.0", None, None)
NATIVE_DISPATCH = attention.require_native_dispatch


@pytest.fixture
def cache_loader(synthetic_loader, world, monkeypatch):
    backend, control, events = synthetic_loader
    selected = replace(module.settings, model_id="black-forest-labs/FLUX.1-Krea-dev",
                       quantization="fp8", offload="model", device="cuda")
    monkeypatch.setattr(module, "current_settings", lambda _: selected)
    monkeypatch.setattr(module, "compiler_metadata", lambda: FACTS)
    monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", None)
    sys.modules["torch"].cuda.is_available = lambda: True  # Fixture ModuleType only.
    control.root = FluxTransformer2DModel(world)
    control.native = True
    control.grad = control.sigma = False
    control.release_observations = []
    placement = module.loading.place_pipeline

    def place(pipe, settings):
        placement(pipe, settings)
        pipe.transformer = control.root

    def native(root):
        events.append("native-check")
        if root is not control.root or not control.native:
            raise ValueError("synthetic nonnative attention")

    release = backend._release

    def released():
        control.release_observations.append((backend._lock.locked(), control.grad, control.sigma,
                                             backend._cache_owner, backend._refine_pipe, backend._inpaint_pipe))
        release()

    monkeypatch.setattr(module.loading, "place_pipeline", place)
    monkeypatch.setattr(fbc, "require_native_attention", native)
    monkeypatch.setattr(backend, "_release", released)
    return backend, control, events, world


def test_eager_loading_never_enters_cache_metadata_runtime_or_hooks(cache_loader, monkeypatch):
    backend, control, _events, state = cache_loader

    def forbidden(*args, **kwargs):
        pytest.fail("Eager entered private cache preparation")

    monkeypatch.setattr(module, "compiler_metadata", forbidden)
    monkeypatch.setattr(fbc, "FirstBlockCacheOwner", forbidden)
    monkeypatch.setattr(fbc, "require_native_attention", forbidden)
    backend.load()
    assert backend._cache_owner is None and backend._cache_disposal is None and state.configs == []
    backend.unload()
    assert control.releases == 1


def test_private_load_owns_hook_free_cache_after_placement_and_keeps_public_calls_closed(cache_loader):
    backend, control, events, state = cache_loader
    backend._load_prepared(SPEC)
    owner, disposal = backend._cache_owner, backend._cache_disposal
    assert type(owner) is fbc.FirstBlockCacheOwner and backend._loaded_execution == SPEC
    assert owner.require(control.root).threshold == 0.05
    disposal.require(backend, owner)
    assert state.configs == state.managers == [] and not owner.active
    assert events.index("post") < events.index("native-check") < events.index("weights-exit")
    backend._load_prepared(SPEC)
    assert backend._cache_owner is owner and state.configs == []
    backend._peak_device = lambda: pytest.fail("public cached state reached a device query")
    request = SimpleNamespace(execution=None)
    for call in (lambda: backend.applied(request), lambda: backend.generate(request, 1),
                 lambda: backend._generate_images(request, 1)):
        with pytest.raises(ValueError, match="not available"):
            call()
    backend.unload()
    assert owner.closed and control.releases == 1 and state.configs == []


@pytest.mark.parametrize("spec", [
    ExecutionSpec(step_cache="fbc-005-v1", compilation="regional-v1"),
    ExecutionSpec(step_cache="fbc-005-v1", attention_backend="cudnn-v1"),
    SPEC.model_copy(update={"unknown": None}), SPEC.model_copy(update={"step_cache": "future"}),
])
def test_invalid_private_cache_recipe_refuses_before_loading(cache_loader, spec):
    backend, control, events, state = cache_loader
    with pytest.raises((ValueError, ValidationError)):
        backend._load_prepared(spec)
    assert events == [] and control.releases == 0 and state.configs == []


@pytest.mark.parametrize("fault", ["model", "metadata", "external"])
def test_cache_cold_preflight_refuses_before_weights(cache_loader, monkeypatch, fault):
    backend, control, events, state = cache_loader
    if fault == "model":
        selected = module.current_settings(module.settings)
        monkeypatch.setattr(module, "current_settings", lambda _: replace(selected, model_id="unsupported/model"))
    elif fault == "metadata":
        monkeypatch.setattr(module, "compiler_metadata", lambda: replace(FACTS, diffusers="0.39.0"))
    else:
        monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", "fixed warning")
        monkeypatch.setattr(module, "compiler_metadata", lambda: pytest.fail("external refusal read metadata"))
    with pytest.raises(ValueError):
        backend._load_prepared(SPEC)
    assert events == [] and state.configs == [] and control.releases == 0


@pytest.mark.parametrize("stage", ["weights-exit", "ready-log"])
def test_late_preparation_failure_closes_hook_free_owner_outside_locks(cache_loader, stage):
    backend, control, _events, state = cache_loader
    control.stage = stage
    with pytest.raises(RuntimeError) as caught:
        backend._load_prepared(SPEC)
    assert caught.value is control.error and not getattr(control.error, "__notes__", [])
    assert control.releases == 1 and backend._cache_owner is None and backend._cache_disposal is None
    assert control.release_observations == [(False, False, False, None, None, None)]
    assert state.configs == []


def forward(backend, control, state, monkeypatch, *, failure=None, reset_failure=False, native_change=False):
    calls = []
    pipe = backend._pipe

    def called(self, generator=None, callback_on_step_end=None, **kwargs):
        recorded = dict(kwargs)
        if generator is not None:
            recorded["generator"] = generator
        if callback_on_step_end is not None:
            recorded["callback_on_step_end"] = callback_on_step_end
        calls.append(recorded)
        owner = backend._cache_owner
        assert owner.active and self.transformer._cache_config.threshold == 0.05
        manager = state.managers[-1]
        with self.transformer.cache_context("cond"):
            manager.get_state()["synthetic residual"] = object()
            if reset_failure:
                manager.fail_reset = True
            if native_change:
                control.native = False
            if failure is not None:
                raise failure
        return SimpleNamespace(images=["synthetic cached image"])

    monkeypatch.setattr(type(pipe), "__call__", called)
    return pipe, calls


@pytest.mark.parametrize("failure", [None, RuntimeError("synthetic forward failure"), Cancelled("synthetic cancellation")])
def test_one_forward_owns_fresh_cache_context_and_preserves_original_failure(cache_loader, monkeypatch, failure):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch, failure=failure)
    kwargs = {"num_images_per_prompt": 1, "generator": object()}
    if failure is None:
        assert backend._invoke_pipeline(pipe, kwargs).images == ["synthetic cached image"]
    else:
        with pytest.raises(type(failure)) as caught:
            backend._invoke_pipeline(pipe, kwargs)
        assert caught.value is failure and not getattr(failure, "__notes__", [])
    assert calls == [kwargs] and len(state.configs) == 1 and not owner.active
    assert state.managers[-1]._state_cache == {} and state.managers[-1]._current_context is None
    assert_clean(control.root, owner)
    backend.unload()


@pytest.mark.parametrize("failure", [None, RuntimeError("synthetic native-change failure"), Cancelled("synthetic native-change cancellation")])
def test_native_change_during_forward_refuses_result_or_preserves_primary_with_cleanup(cache_loader, monkeypatch, failure):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch, failure=failure, native_change=True)
    with pytest.raises(ValueError if failure is None else type(failure)) as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    if failure is not None:
        assert caught.value is failure
        assert "Native attention verification also failed" in " ".join(failure.__notes__)
    assert len(calls) == 1 and len(state.configs) == 1 and state.disable_calls == 1
    assert state.managers[-1]._state_cache == {} and state.managers[-1]._current_context is None
    assert_clean(control.root, owner)
    assert backend._pending_cache_disposal is None
    backend.unload()


def test_partial_installation_with_successful_cleanup_keeps_owned_pipeline_usable(cache_loader, monkeypatch):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch)
    state.fail_after = 1
    with pytest.raises(RuntimeError) as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert caught.value is state.setup_failure and not getattr(caught.value, "__notes__", [])
    assert calls == [] and backend.is_ready() and control.releases == 0
    assert_clean(control.root, owner)
    state.fail_after = None
    backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert len(calls) == 1 and not owner.invalidated
    backend.unload()


def test_uncertain_partial_installation_detaches_pipeline_before_any_forward(cache_loader, monkeypatch):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch)
    state.break_bookkeeping = True
    with pytest.raises(RuntimeError, match="forward rewrite") as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert any("forward" in note for note in caught.value.__notes__)
    assert calls == [] and owner.invalidated and owner.closed
    assert not backend.is_ready() and backend._pending_cache_disposal is None and control.releases == 1


def test_consecutive_cached_calls_and_off_reload_never_reuse_call_state(cache_loader, monkeypatch):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch)
    for seed in (11, 12):
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1, "generator": seed, "prompt": str(seed)})
        assert_clean(control.root, owner)
    assert len(calls) == 2 and state.configs[0] is not state.configs[1]
    assert state.managers[0] is not state.managers[1]
    backend.unload()
    control.root = FluxTransformer2DModel(state)
    backend.load()
    assert backend._cache_owner is None and control.root._cache_config is None
    backend.unload()
    control.root = FluxTransformer2DModel(state)
    backend._load_prepared(SPEC)
    assert backend._cache_owner is not owner and len(state.configs) == 2
    backend.unload()


@pytest.mark.parametrize("fault", ["batch", "lora-zero", "true-cfg", "sibling", "root", "owner", "closed",
                                  "stacked", "attention-owner", "regional-owner", "disposal", "external", "native"])
def test_incoherent_cached_call_refuses_before_hooks_or_forward(cache_loader, monkeypatch, fault):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch)
    kwargs, runner = {"num_images_per_prompt": 1}, pipe
    if fault == "batch":
        kwargs["num_images_per_prompt"] = 2
    elif fault == "lora-zero":
        backend._lora = ("adapter.safetensors", 0.0)
    elif fault == "true-cfg":
        kwargs["true_cfg_scale"] = 2.0
    elif fault == "sibling":
        runner = SimpleNamespace(transformer=pipe.transformer)
    elif fault == "root":
        pipe.transformer = FluxTransformer2DModel(state)
    elif fault == "owner":
        backend._cache_owner = object()
    elif fault == "closed":
        owner.close()
    elif fault == "stacked":
        backend._loaded_execution = ExecutionSpec(step_cache="fbc-005-v1", attention_backend="cudnn-v1")
    elif fault == "attention-owner":
        backend._attention_selection = object()
    elif fault == "regional-owner":
        backend._regional_registration = object()
    elif fault == "disposal":
        backend._cache_disposal.owner = lambda: None
    elif fault == "external":
        monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", "fixed warning")
    else:
        control.native = False
    with pytest.raises((ValueError, RuntimeError)):
        backend._invoke_pipeline(runner, kwargs)
    assert calls == [] and state.configs == []
    # Restore only test-injected private fields, not failed real operations.
    backend._regional_registration = backend._attention_selection = None
    backend._cache_owner = owner
    backend.unload()


def test_cleanup_failure_detaches_synchronously_but_disposes_once_after_explicit_call_scope(cache_loader, monkeypatch):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner, retirement = backend._cache_owner, backend._cache_disposal
    backend._refine_pipe, backend._inpaint_pipe = object(), object()
    pipe, calls = forward(backend, control, state, monkeypatch)
    state.remove_failure = RuntimeError("synthetic removal failure")
    with module._CallDisposal(backend) as scope:
        with pytest.raises(fbc.CacheCleanupFailed, match="synthetic removal failure"):
            backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1}, disposal=scope)
        assert owner.invalidated and not owner.active and not owner.closed
        assert not backend.is_ready() and backend._pipe is None and backend._cache_owner is None
        assert backend._lora is None and backend._refine_pipe is None and backend._inpaint_pipe is None
        assert backend._pending_cache_disposal is retirement and retirement.pending.cache is owner
        assert control.releases == 0
        with pytest.raises(RuntimeError, match="pending cache disposal"):
            backend.load()
    assert owner.closed and retirement.done and backend._pending_cache_disposal is None
    assert control.releases == 1 and len(calls) == 1
    retirement.drain()
    assert control.releases == 1
    with pytest.raises(fbc.CacheCleanupFailed, match="discarded"):
        fbc.FirstBlockCacheOwner(control.root, discard_shared=lambda: None)


@pytest.mark.parametrize("failure", [RuntimeError("synthetic primary"), Cancelled("synthetic primary cancellation")])
def test_cleanup_and_release_failures_preserve_primary_without_retry(cache_loader, monkeypatch, failure):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch, failure=failure, reset_failure=True)
    control.release_error = OSError("synthetic allocator release failure")
    with pytest.raises(type(failure)) as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert caught.value is failure and owner.invalidated and owner.closed
    assert "reset failed" in " ".join(failure.__notes__)
    assert "allocator release failure" in " ".join(failure.__notes__)
    assert not backend.is_ready() and backend._pending_cache_disposal is None
    assert control.releases == 1 and len(calls) == 1


def test_actual_outer_generate_disposes_after_sigma_and_no_grad_exit(cache_loader, monkeypatch):
    from backend.schemas import GenerateRequest

    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    pipe, calls = forward(backend, control, state, monkeypatch)
    state.remove_failure = RuntimeError("synthetic outer cleanup failure")

    class Generator:
        def __init__(self, **kwargs):
            pass

        def manual_seed(self, seed):
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
    # Exercise the private flow under an explicitly stubbed public gate. The
    # real gate remains tested closed above; production has no bypass flag.
    monkeypatch.setattr(backend, "_require_execution", lambda request: request)
    monkeypatch.setattr(backend, "applied", lambda request: request)
    monkeypatch.setattr(backend, "_apply_lora", lambda *_: None)
    backend._call_params = {"prompt", "width", "height", "num_inference_steps", "guidance_scale",
                            "num_images_per_prompt", "generator", "callback_on_step_end"}
    with pytest.raises(fbc.CacheCleanupFailed):
        backend._generate_images(GenerateRequest(prompt="synthetic", steps=1, width=256, height=256), 1)
    assert owner.closed and len(calls) == 1
    assert control.release_observations == [(False, False, False, None, None, None)]
    assert not backend.is_ready() and backend._pending_cache_disposal is None


def test_old_discard_callback_cannot_detach_a_new_load(cache_loader):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    old = backend._cache_disposal
    owner = backend._cache_owner
    backend.unload()
    control.root = FluxTransformer2DModel(state)
    backend._load_prepared(SPEC)
    current = backend._cache_owner
    old.detach()
    assert owner.closed and backend.is_ready() and backend._cache_owner is current
    assert backend._pending_cache_disposal is None and old.pending is None
    backend.unload()


def test_terminal_unload_discard_leaves_foreign_cache_untouched_and_poisoned(cache_loader):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    owner = backend._cache_owner
    foreign = control.root._cache_config = object()
    with pytest.raises(ValueError, match="foreign cache"):
        backend.unload()
    assert owner.closed and owner.invalidated and control.root._cache_config is foreign
    assert state.removed == [] and state.disable_calls == 0 and control.releases == 1
    assert not backend.is_ready() and backend._cache_owner is None
    with pytest.raises(fbc.CacheCleanupFailed, match="discarded"):
        fbc.FirstBlockCacheOwner(control.root, discard_shared=lambda: None)


def test_foreign_hook_is_preserved_across_an_owned_cached_call(cache_loader, monkeypatch):
    backend, control, _events, state = cache_loader
    backend._load_prepared(SPEC)
    block = control.root.transformer_blocks[0]
    registry = Registry.check_if_exists_or_initialize(block)
    foreign = Hook()
    registry.register_hook(foreign, "foreign")
    before = (block.forward, tuple(registry._fn_refs))
    pipe, _calls = forward(backend, control, state, monkeypatch)
    backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1, "true_cfg_scale": 1.0})
    assert registry.hooks == {"foreign": foreign} and foreign not in state.removed
    assert (block.forward, tuple(registry._fn_refs)) == before
    backend.unload()


@pytest.fixture
def native_world(monkeypatch):
    # This fixture isolates the live registry/processor contract; environment
    # and frozen-import observations have separate synthetic policy coverage.
    monkeypatch.setattr(attention, "require_native_environment", lambda: None)
    monkeypatch.setattr(attention, "require_native_dispatch", NATIVE_DISPATCH)
    state = SimpleNamespace(active="native", function=lambda: None)
    bindings = attention._Bindings(Transformer, Joint, Single, Attention, Processor, backend_context, "_native_cudnn")
    api = attention._NativeAttentionApi("native", lambda: (state.active, state.function), bindings, state.function)
    monkeypatch.setattr(attention, "_native_attention_api", lambda: api)
    return Transformer(), state


@pytest.mark.parametrize("override", [None, "native"])
def test_effective_native_attention_is_read_only(native_world, override):
    root, state = native_world
    root.transformer_blocks[0].attn.processor._attention_backend = override
    before = dict(vars(root.transformer_blocks[0].attn.processor))
    fbc.require_native_attention(root)
    assert state.active == "native" and root.calls == []
    assert vars(root.transformer_blocks[0].attn.processor) == before


@pytest.mark.parametrize("fault", ["global", "processor", "missing-function", "parallel", "layout"])
def test_nonnative_or_uncertain_attention_refuses_without_resetting(native_world, fault):
    root, state = native_world
    processor = root.transformer_blocks[0].attn.processor
    if fault == "global":
        state.active = "_native_cudnn"
    elif fault == "processor":
        processor._attention_backend = "_native_cudnn"
    elif fault == "missing-function":
        state.function = None
    elif fault == "parallel":
        processor._parallel_config = object()
    else:
        root.config["num_layers"] = 999
    before = (state.active, dict(vars(processor)))
    with pytest.raises(ValueError):
        fbc.require_native_attention(root)
    assert (state.active, vars(processor)) == before and root.calls == []


def test_installed_active_dispatch_getter_is_only_a_read_only_lookup():
    path = distribution("diffusers").locate_file("diffusers/models/attention_dispatch.py")
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    registry = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_AttentionBackendRegistry")
    getter = next(node for node in registry.body if isinstance(node, ast.FunctionDef) and node.name == "get_active_backend")
    assert len(getter.body) == 1
    assert ast.unparse(getter.body[0]) == "return (cls._active_backend, cls._backends[cls._active_backend])"


def test_private_cache_refusals_are_cold_and_leave_environment_unchanged(tmp_path):
    script = r'''
import importlib.abc,json,os,sys
from types import SimpleNamespace
blocked={"torch","torchao","diffusers","huggingface_hub","triton"}
class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".",1)[0] in blocked:raise AssertionError("runtime import: "+fullname)
sys.meta_path.insert(0,ColdOnly())
from backend.backends import attention_selection
assert "backend.config" not in sys.modules
try:attention_selection._native_attention_api()
except AssertionError as error:
    assert "runtime import: diffusers" in str(error) and "backend.config" in sys.modules
else:raise AssertionError("native adapter imported a real runtime")
from backend.backends import diffusers_backend as module
from backend.execution import CompilerMetadata,ExecutionSpec
before=dict(os.environ)
module.current_settings=lambda _:SimpleNamespace(model_id="unsupported/model",quantization="fp8",offload="model",device="cuda")
module.ATTENTION_ENVIRONMENT_WARNING=None
module.compiler_metadata=lambda:CompilerMetadata("win32",(3,13),"2.13.0+cu130","0.40.0","0.18.0",None,None)
module.loading.weights_loading=lambda:(_ for _ in ()).throw(AssertionError("weights entered"))
backend=module.DiffusersBackend()
try:backend._load_prepared(ExecutionSpec(step_cache="fbc-005-v1"))
except ValueError:pass
else:raise AssertionError("unsupported cache loaded")
backend._pipe=object();backend._cache_owner=object()
for call in (lambda:backend.applied(SimpleNamespace(execution=None)),lambda:backend.generate(SimpleNamespace(execution=None),1),lambda:backend._generate_images(SimpleNamespace(execution=None),1)):
    try:call()
    except ValueError:pass
    else:raise AssertionError("orphan cache executed")
assert not any(name.split(".",1)[0] in blocked for name in sys.modules)
assert before==dict(os.environ)
print(json.dumps({"cold":True,"refused":4,"unchanged_environment":True}))
'''
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    done = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)
    (tmp_path / "cache.stdout").write_text(done.stdout, encoding="utf-8")
    (tmp_path / "cache.stderr").write_text(done.stderr, encoding="utf-8")
    assert done.returncode == 0, done.stdout + "\n" + done.stderr
    assert json.loads(done.stdout) == {"cold": True, "refused": 4, "unchanged_environment": True}
