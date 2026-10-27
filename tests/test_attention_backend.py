"""Private attention lifecycle over synthetic runtime/processor modules only."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
from pydantic import ValidationError

from backend.backends import Cancelled
from backend.backends import attention_selection as selection
from backend.backends import diffusers_backend as module
from backend.execution import CompilerMetadata, ExecutionSpec
from tests.test_attention_selection import (
    Attention, Joint, Processor, Registry, Single, Transformer, _before, _processors,
    _unchanged, backend_context, bindings,
)
from tests.test_pipeline_ownership import synthetic_loader


SPEC = ExecutionSpec(attention_backend="cudnn-v1")
FACTS = CompilerMetadata("win32", (3, 13), "2.13.0+cu130", "0.40.0", "0.18.0", None, None)


@pytest.fixture
def attention_loader(synthetic_loader, monkeypatch):
    backend, control, events = synthetic_loader
    selected = replace(module.settings, model_id="black-forest-labs/FLUX.1-Krea-dev",
                       quantization="fp8", offload="model", device="cuda")
    monkeypatch.setattr(module, "current_settings", lambda _: selected)
    monkeypatch.setattr(module, "compiler_metadata", lambda: FACTS)
    monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", None)
    # ModuleType installed by synthetic_loader, never the installed framework.
    sys.modules["torch"].cuda.is_available = lambda: True
    monkeypatch.setattr(Registry, "active", "unrelated-default")
    control.root = Transformer()
    control.before = _before(control.root)

    @contextmanager
    def context(name):
        events.append("attention-enter")
        with backend_context(name):
            yield
        events.append("attention-exit")

    api = selection._Bindings(Transformer, Joint, Single, Attention, Processor, context, "_native_cudnn")
    monkeypatch.setattr(selection, "_load_bindings", lambda: api)
    placement = module.loading.place_pipeline

    def place(pipe, settings):
        placement(pipe, settings)
        pipe.transformer = control.root

    monkeypatch.setattr(module.loading, "place_pipeline", place)
    release = backend._release

    def released():
        assert backend._attention_selection is None
        release()

    monkeypatch.setattr(backend, "_release", released)
    return backend, control, events


def test_eager_load_never_enters_attention_bindings_or_metadata(attention_loader, monkeypatch):
    backend, control, events = attention_loader

    def forbidden(*args, **kwargs):
        pytest.fail("Eager entered private attention preparation")

    monkeypatch.setattr(module, "compiler_metadata", forbidden)
    monkeypatch.setattr(selection, "configure_attention", forbidden)
    backend.load()
    assert backend._attention_selection is None and backend._loaded_execution is None
    assert control.root.calls == [] and "attention-enter" not in events
    backend.unload()
    assert control.releases == 1


def test_attention_load_prepares_after_checks_publishes_owned_layout_and_keeps_public_calls_closed(attention_loader):
    backend, control, events = attention_loader
    backend._load_prepared(SPEC)
    owner = backend._attention_selection
    assert type(owner) is selection.AttentionSelection and backend._loaded_execution == SPEC
    owner.require(control.root)
    assert control.root.calls == ["_native_cudnn"] and Registry.active == "unrelated-default"
    assert events.index("post") < events.index("attention-enter") < events.index("attention-exit")
    assert events.index("attention-exit") < events.index("weights-exit") < events.index("ready-log")
    backend._load_prepared(SPEC)
    assert control.root.calls == ["_native_cudnn"]
    backend._peak_device = lambda: pytest.fail("public attention state reached a device query")
    request = SimpleNamespace(execution=None)
    for call in (lambda: backend.applied(request), lambda: backend.generate(request, 1),
                 lambda: backend._generate_images(request, 1)):
        with pytest.raises(ValueError, match="not available"):
            call()
    backend.unload()
    assert owner.closed and control.releases == 1 and Registry.active == "unrelated-default"
    _unchanged(control.before)


@pytest.mark.parametrize("spec", [
    ExecutionSpec(attention_backend="cudnn-v1", compilation="regional-v1"),
    ExecutionSpec(attention_backend="cudnn-v1", step_cache="fbc-005-v1"),
    SPEC.model_copy(update={"unknown": None}), SPEC.model_copy(update={"attention_backend": "future"}),
])
def test_malformed_combined_private_attention_refuses_before_weights(attention_loader, spec):
    backend, control, events = attention_loader
    with pytest.raises((ValueError, ValidationError)):
        backend._load_prepared(spec)
    assert events == [] and not backend.is_ready() and control.releases == 0


@pytest.mark.parametrize("fault", ["model", "metadata", "external"])
def test_private_attention_cold_matrix_refuses_before_weights(attention_loader, monkeypatch, fault):
    backend, control, events = attention_loader
    if fault == "model":
        original = module.current_settings(module.settings)
        monkeypatch.setattr(module, "current_settings", lambda _: replace(original, model_id="other/model"))
    elif fault == "metadata":
        monkeypatch.setattr(module, "compiler_metadata", lambda: replace(FACTS, diffusers="0.39.0"))
    else:
        monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", "fixed warning")
        monkeypatch.setattr(module, "compiler_metadata", lambda: pytest.fail("external refusal read metadata"))
    with pytest.raises(ValueError):
        backend._load_prepared(SPEC)
    assert events == [] and control.releases == 0


@pytest.mark.parametrize("stage", ["weights-exit", "ready-log"])
def test_late_preparation_failure_restores_attention_outside_backend_lock(attention_loader, monkeypatch, stage):
    backend, control, events = attention_loader
    control.stage = stage
    original_restore = selection.AttentionSelection.restore
    attempts = []

    def restored(owner):
        assert not backend._lock.locked()
        attempts.append(True)
        original_restore(owner)

    monkeypatch.setattr(selection.AttentionSelection, "restore", restored)
    with pytest.raises(RuntimeError) as caught:
        backend._load_prepared(SPEC)
    assert caught.value is control.error and not getattr(caught.value, "__notes__", [])
    assert attempts == [True] and backend._pipe is None and backend._attention_selection is None
    assert control.pipe_ref() is None and control.releases == 1
    _unchanged(control.before)


def test_setter_failure_restores_global_state_never_publishes_or_retries(attention_loader):
    backend, control, events = attention_loader
    original = Cancelled("synthetic setup cancellation")

    def fail():
        raise original

    control.root.after_set = fail
    with pytest.raises(Cancelled) as caught:
        backend._load_prepared(SPEC)
    assert caught.value is original and not getattr(original, "__notes__", [])
    assert control.root.calls == ["_native_cudnn"] and Registry.active == "unrelated-default"
    assert backend._pipe is None and backend._attention_selection is None and control.releases == 1
    _unchanged(control.before)


@pytest.mark.parametrize("method", ["unload", "try_unload"])
def test_both_unload_paths_drop_siblings_restore_original_processors_and_release_outside_locks(attention_loader, monkeypatch, method):
    backend, control, _events = attention_loader
    backend._load_prepared(SPEC)
    owner = backend._attention_selection
    backend._refine_pipe, backend._inpaint_pipe = object(), object()
    restore = owner.restore
    observations = []

    def restored():
        observations.append((backend._lock.locked(), backend._refine_pipe, backend._inpaint_pipe,
                             backend._attention_selection))
        restore()

    monkeypatch.setattr(owner, "restore", restored)
    result = getattr(backend, method)()
    assert result is (True if method == "try_unload" else None)
    assert observations == [(False, None, None, None)] and owner.closed and control.releases == 1
    _unchanged(control.before)
    backend.unload()
    assert len(observations) == 1 and control.releases == 1


def test_contended_try_unload_preserves_attention_ownership(attention_loader):
    backend, control, _events = attention_loader
    backend._load_prepared(SPEC)
    owner = backend._attention_selection
    with backend._lock:
        assert backend.try_unload() is False
    owner.require(control.root)
    assert backend._attention_selection is owner and control.releases == 0
    backend.unload()


@pytest.fixture
def attention_call(bindings, monkeypatch):
    backend = module.DiffusersBackend()
    root = Transformer()
    owner = selection.configure_attention(root, "cudnn-v1")
    state = SimpleNamespace(failure=None, change=False)
    calls = []

    class Pipeline:
        transformer = root

        def __call__(self, **kwargs):
            calls.append(dict(kwargs))
            assert Registry.active == "unrelated-default"
            if state.change:
                self.transformer.transformer_blocks[0].attn.processor._attention_backend = "changed"
            if state.failure is not None:
                raise state.failure
            return SimpleNamespace(images=["synthetic attention image"])

    pipe = Pipeline()
    backend._pipe, backend._loaded_execution, backend._attention_selection = pipe, SPEC, owner
    monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", None)
    monkeypatch.setattr(module.regional_compile, "_dynamo_patch", lambda: pytest.fail("attention entered compiler policy"))
    yield backend, pipe, owner, state, calls
    if not owner.closed:
        try:
            owner.restore()
        except selection.AttentionSelectionError:
            owner.discard()


@pytest.mark.parametrize("failure", [None, RuntimeError("synthetic attention failure"), Cancelled("synthetic cell cancellation")])
def test_attention_invocation_is_one_forward_without_global_policy_retry_or_argument_changes(attention_call, failure):
    backend, pipe, owner, state, calls = attention_call
    state.failure = failure
    kwargs = {"num_images_per_prompt": 1, "generator": object()}
    before = dict(kwargs)
    if failure is None:
        assert backend._invoke_pipeline(pipe, kwargs).images == ["synthetic attention image"]
    else:
        with pytest.raises(type(failure)) as caught:
            backend._invoke_pipeline(pipe, kwargs)
        assert caught.value is failure
    assert calls == [before] and kwargs == before and Registry.active == "unrelated-default"
    owner.require(pipe.transformer)


@pytest.mark.parametrize("change", ["closed", "foreign-root", "wrong-handle", "missing", "backend", "layout",
                                  "combination", "regional-owner", "lora-zero", "batch", "sibling", "external"])
def test_incoherent_attention_refuses_before_forward(attention_call, monkeypatch, change):
    backend, pipe, owner, _state, calls = attention_call
    runner, kwargs = pipe, {"num_images_per_prompt": 1}
    if change == "closed":
        owner.restore()
    elif change == "foreign-root":
        pipe.transformer = Transformer()
    elif change == "wrong-handle":
        backend._attention_selection = SimpleNamespace(require=lambda *_: pytest.fail("foreign handle called"))
    elif change == "missing":
        backend._attention_selection = None
    elif change == "backend":
        pipe.transformer.transformer_blocks[0].attn.processor._attention_backend = "changed"
    elif change == "layout":
        pipe.transformer.transformer_blocks[0].attn.processor = Processor()
    elif change == "combination":
        backend._loaded_execution = ExecutionSpec(attention_backend="cudnn-v1", step_cache="fbc-005-v1")
    elif change == "regional-owner":
        backend._regional_registration = object()
    elif change == "lora-zero":
        backend._lora = ("adapter.safetensors", 0.0)
    elif change == "batch":
        kwargs["num_images_per_prompt"] = 2
    elif change == "sibling":
        runner = SimpleNamespace(transformer=pipe.transformer)
    else:
        monkeypatch.setattr(module, "ATTENTION_ENVIRONMENT_WARNING", "fixed warning")
    with pytest.raises((ValueError, RuntimeError)):
        backend._invoke_pipeline(runner, kwargs)
    assert calls == [] and Registry.active == "unrelated-default"


def test_changed_attention_during_forward_refuses_result_before_save(attention_call):
    backend, pipe, _owner, state, calls = attention_call
    state.change = True
    with pytest.raises(selection.AttentionSelectionError, match="backend changed"):
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [RuntimeError("synthetic forward failure"), Cancelled("synthetic forward cancellation")])
def test_changed_attention_during_failed_forward_preserves_primary_and_diagnostic(attention_call, failure):
    backend, pipe, _owner, state, calls = attention_call
    state.change, state.failure = True, failure
    with pytest.raises(type(failure)) as caught:
        backend._invoke_pipeline(pipe, {"num_images_per_prompt": 1})
    assert caught.value is failure and len(calls) == 1
    assert "Attention ownership verification also failed" in " ".join(failure.__notes__)
    assert "backend changed" in " ".join(failure.__notes__)


def test_orphan_attention_refuses_eager_paths_before_device_work(attention_call):
    backend, pipe, _owner, _state, calls = attention_call
    backend._loaded_execution = None
    backend._peak_device = lambda: pytest.fail("orphan attention reached device work")
    request = SimpleNamespace(execution=None)
    for call in (lambda: backend.applied(request), lambda: backend.generate(request, 1),
                 lambda: backend._generate_images(request, 1)):
        with pytest.raises(ValueError, match="not available"):
            call()
    with pytest.raises(RuntimeError, match="unrecorded attention"):
        backend._invoke_pipeline(pipe, {})
    assert calls == []


def test_terminal_failed_restore_discards_weak_identities_without_retaining_models(bindings):
    root = Transformer()
    processors = _processors(root)
    owner = selection.configure_attention(root, "cudnn-v1")
    processors[0].fail_restore = True
    with pytest.raises(selection.AttentionSelectionError, match="restoration failed"):
        owner.restore()
    owner.discard()
    assert owner.closed and owner._transformer is None and owner._snapshots == () and owner._identity_refs == ()
    processors[0].fail_restore = False
    with pytest.raises(selection.AttentionSelectionError, match="discarded"):
        selection.configure_attention(root, "cudnn-v1")
    other = Transformer()
    other.transformer_blocks[0].attn.processor = processors[0]
    with pytest.raises(selection.AttentionSelectionError, match="discarded"):
        selection.configure_attention(other, "cudnn-v1")
    fresh = Transformer()
    next_owner = selection.configure_attention(fresh, "cudnn-v1")
    next_owner.restore()
    references = [weakref.ref(root), *(weakref.ref(processor) for processor in processors)]
    identities = [id(root), *(id(processor) for processor in processors)]
    del owner, root, processors, other
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not any(identity in selection._discarded for identity in identities)


def test_failed_setup_rollback_poisons_originals_but_not_unrelated_replacements(bindings):
    root = Transformer()
    original = _processors(root)
    replacement = Processor()
    original[0].fail_restore = True

    def changed_then_failed():
        root.transformer_blocks[1].attn.processor = replacement
        raise RuntimeError("synthetic setter and rollback failure")

    root.after_set = changed_then_failed
    with pytest.raises(RuntimeError, match="setter and rollback") as caught:
        selection.configure_attention(root, "cudnn-v1")
    assert "rollback failed" in " ".join(caught.value.__notes__)
    assert selection._discarded[id(root)]() is root
    assert all(selection._discarded[id(processor)]() is processor for processor in original)
    assert id(replacement) not in selection._discarded and replacement._attention_backend is None
    assert Registry.active == "unrelated-default"


def test_failed_setup_layout_replacement_cannot_be_adopted_after_successful_rollback(bindings):
    root = Transformer()
    original = _processors(root)
    replacement = Processor()
    root.after_set = lambda: setattr(root.transformer_blocks[0].attn, "processor", replacement)
    with pytest.raises(selection.AttentionSelectionError, match="coverage changed"):
        selection.configure_attention(root, "cudnn-v1")
    assert all(processor._attention_backend is None for processor in original)
    assert id(replacement) not in selection._discarded
    with pytest.raises(selection.AttentionSelectionError, match="discarded"):
        selection.configure_attention(root, "cudnn-v1")
    other = Transformer()
    other.transformer_blocks[0].attn.processor = original[0]
    with pytest.raises(selection.AttentionSelectionError, match="discarded"):
        selection.configure_attention(other, "cudnn-v1")


def test_weak_quarantine_callback_cannot_forget_a_reused_identity(monkeypatch):
    key = -12345  # Synthetic identity seam, never an actual object address.
    monkeypatch.setattr(selection, "id", lambda _: key, raising=False)
    old, current = Processor(), Processor()
    old_reference = selection._identity_reference(old)
    current_reference = selection._identity_reference(current)
    selection._discarded[key] = current_reference
    try:
        del old
        gc.collect()
        assert old_reference() is None and selection._discarded[key] is current_reference
        with pytest.raises(selection.AttentionSelectionError, match="discarded"):
            selection._require_fresh(current)
        del current
        gc.collect()
        assert current_reference() is None and key not in selection._discarded
    finally:
        selection._discarded.pop(key, None)


def test_unsupported_weak_root_refuses_before_setter_or_global_mutation(bindings, monkeypatch):
    class NonWeakTransformer:
        __slots__ = ("config", "transformer_blocks", "single_transformer_blocks", "extra", "missing",
                     "reported", "calls", "after_set", "skip")
        __init__ = Transformer.__init__
        attn_processors = Transformer.attn_processors
        modules = Transformer.modules
        set_attention_backend = Transformer.set_attention_backend

    monkeypatch.setattr(selection, "_load_bindings", lambda: replace(bindings, transformer_type=NonWeakTransformer))
    root = NonWeakTransformer()
    with pytest.raises(TypeError, match="weak reference"):
        selection.configure_attention(root, "cudnn-v1")
    assert root.calls == [] and Registry.active == "unrelated-default"
    assert all(processor._attention_backend is None for processor in _processors(root))


def test_terminal_backend_cleanup_preserves_primary_and_releases_after_failed_restore(attention_loader):
    backend, control, _events = attention_loader
    control.stage = "weights-exit"
    control.root.after_set = lambda: setattr(control.root.transformer_blocks[0].attn.processor, "fail_restore", True)
    with pytest.raises(RuntimeError) as caught:
        backend._load_prepared(SPEC)
    assert caught.value is control.error and "restoration failed" in " ".join(caught.value.__notes__)
    assert control.releases == 1 and backend._pipe is None and backend._attention_selection is None
    assert selection._discarded[id(control.root)]() is control.root


def test_private_attention_preflight_and_orphan_refusal_are_import_cold(tmp_path):
    script = r'''
import importlib.abc,json,os,sys
from types import SimpleNamespace
blocked={"torch","torchao","diffusers","huggingface_hub","triton"}
class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".",1)[0] in blocked: raise AssertionError("runtime import: "+fullname)
sys.meta_path.insert(0,ColdOnly())
from backend.backends import diffusers_backend as module
from backend.execution import CompilerMetadata,ExecutionSpec
before=dict(os.environ)
module.current_settings=lambda _:SimpleNamespace(model_id="unsupported/model",quantization="fp8",offload="model",device="cuda")
module.ATTENTION_ENVIRONMENT_WARNING=None
module.compiler_metadata=lambda:CompilerMetadata("win32",(3,13),"2.13.0+cu130","0.40.0","0.18.0",None,None)
module.loading.weights_loading=lambda:(_ for _ in ()).throw(AssertionError("weights entered"))
backend=module.DiffusersBackend()
try: backend._load_prepared(ExecutionSpec(attention_backend="cudnn-v1"))
except ValueError: pass
else: raise AssertionError("unsupported target prepared")
backend._pipe=object();backend._attention_selection=object()
for call in (lambda:backend.applied(SimpleNamespace(execution=None)),lambda:backend.generate(SimpleNamespace(execution=None),1),lambda:backend._generate_images(SimpleNamespace(execution=None),1)):
    try:call()
    except ValueError:pass
    else:raise AssertionError("orphan reached work")
assert not any(name.split(".",1)[0] in blocked for name in sys.modules)
assert before==dict(os.environ)
print(json.dumps({"cold":True,"refused":4,"unchanged_environment":True}))
'''
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    done = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)
    (tmp_path / "attention.stdout").write_text(done.stdout, encoding="utf-8")
    (tmp_path / "attention.stderr").write_text(done.stderr, encoding="utf-8")
    assert done.returncode == 0, done.stdout + "\n" + done.stderr
    assert json.loads(done.stdout) == {"cold": True, "refused": 4, "unchanged_environment": True}
