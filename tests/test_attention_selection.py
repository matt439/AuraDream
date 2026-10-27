"""Inactive attention ownership with stubs; no tensors or device probes.

Installed public context/setter bodies are also executed against stubs.
Importing their defining modules is deliberately avoided: torch_utils
initialization calls get_device() in the installed Diffusers release.
"""
from __future__ import annotations

import ast
import contextlib
from dataclasses import replace
from enum import Enum
from importlib.metadata import distribution
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend.backends import attention_selection as selection


class Processor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        self.fail_target = False
        self.fail_restore = False
        self.backend_writes = 0

    def __setattr__(self, name, value):
        if name == "_attention_backend":
            if self.fail_target and value == "_native_cudnn":
                raise RuntimeError("injected target failure")
            if self.fail_restore and value != "_native_cudnn":
                raise RuntimeError("injected restore failure")
            self.backend_writes += 1
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name == "_attention_backend":
            if self.fail_restore:
                raise RuntimeError("injected restore failure")
            self.backend_writes += 1
        object.__delattr__(self, name)


class Attention:
    def __init__(self, processor=None):
        self.processor = Processor() if processor is None else processor


class Joint:
    def __init__(self):
        self.attn = Attention()


class Single:
    def __init__(self):
        self.attn = Attention()


class Transformer:
    def __init__(self):
        self.config = {"num_layers": 2, "num_single_layers": 2}
        self.transformer_blocks = [Joint(), Joint()]
        self.single_transformer_blocks = [Single(), Single()]
        self.extra = {}
        self.missing = set()
        self.reported = {}
        self.calls = []
        self.after_set = None
        self.skip = None

    @property
    def attn_processors(self):
        result = {
            f"{name}.{index}.attn.processor": block.attn.processor
            for name, blocks in (("transformer_blocks", self.transformer_blocks),
                                 ("single_transformer_blocks", self.single_transformer_blocks))
            for index, block in enumerate(blocks)
        }
        result.update(self.extra)
        result.update(self.reported)
        return {name: processor for name, processor in result.items() if name not in self.missing}

    def modules(self):
        return [block.attn for block in [*self.transformer_blocks, *self.single_transformer_blocks]]

    def set_attention_backend(self, backend):
        self.calls.append(backend)
        for index, attention in enumerate(self.modules()):
            if index != self.skip:
                attention.processor._attention_backend = backend
        Registry.active = backend
        if self.after_set is not None:
            self.after_set()


class Registry:
    active = "unrelated-default"

    @classmethod
    def set_active_backend(cls, backend):
        cls.active = backend


@contextlib.contextmanager
def backend_context(backend):
    before = Registry.active
    Registry.active = backend
    try:
        yield
    finally:
        Registry.active = before


@pytest.fixture
def bindings(monkeypatch):
    # These layout/context tests supply no imported dispatcher implementation.
    # The separate policy suite exercises the real guard over synthetic modules.
    monkeypatch.setattr(selection, "require_native_dispatch", lambda: None)
    monkeypatch.setattr(Registry, "active", "unrelated-default")
    value = selection._Bindings(Transformer, Joint, Single, Attention, Processor, backend_context, "_native_cudnn")
    monkeypatch.setattr(selection, "_load_bindings", lambda: value)
    return value


def _processors(model):
    return [attention.processor for attention in model.modules()]


def _before(model):
    return [(processor, processor._attention_backend, "_attention_backend" in vars(processor))
            for processor in _processors(model)]


def _unchanged(before):
    assert all(processor._attention_backend == value
               and ("_attention_backend" in vars(processor)) == owned for processor, value, owned in before)


def test_import_and_native_default_are_cold():
    code = """
import sys
from backend.backends.attention_selection import configure_attention
assert configure_attention(object(), None) is None
assert not any(name == 'torch' or name.startswith(('torch.', 'diffusers')) for name in sys.modules)
assert 'backend.config' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("recipe", ["native", "_native_cudnn", "future", False, {}])
def test_unknown_recipe_refuses_before_loading_apis(recipe, monkeypatch):
    def forbidden():
        raise AssertionError("unsupported recipe imported Diffusers")

    monkeypatch.setattr(selection, "_load_bindings", forbidden)
    with pytest.raises(selection.AttentionSelectionError, match="Unsupported attention recipe"):
        selection.configure_attention(object(), recipe)


def test_complete_joint_and_single_coverage_preserves_prior_values(bindings):
    model = Transformer()
    processors = _processors(model)
    processors[0]._attention_backend = "native"
    processors[1]._attention_backend = "_native_math"
    before = _before(model)
    unrelated = Processor()
    handle = selection.configure_attention(model, "cudnn-v1")
    assert handle is not None
    assert model.calls == ["_native_cudnn"]
    assert all(processor._attention_backend == "_native_cudnn" for processor in processors)
    assert unrelated._attention_backend is None and "_attention_backend" not in vars(unrelated)
    assert Registry.active == "unrelated-default"
    handle.verify()
    handle.restore()
    handle.restore()
    _unchanged(before)
    assert handle._transformer is None and handle._coverage == {} and handle._snapshots == ()
    with pytest.raises(selection.AttentionSelectionError, match="restored"):
        handle.verify()


def test_siblings_share_one_owner_and_unrelated_transformers_remain_native(bindings):
    base = SimpleNamespace(transformer=Transformer())
    sibling = SimpleNamespace(transformer=base.transformer)
    unrelated = Transformer()
    handle = selection.configure_attention(base.transformer, "cudnn-v1")
    assert handle is not None
    assert all(processor._attention_backend == "_native_cudnn" for processor in _processors(sibling.transformer))
    with pytest.raises(selection.AttentionSelectionError, match="already has an attention owner"):
        selection.configure_attention(sibling.transformer, "cudnn-v1")
    assert all(processor._attention_backend is None for processor in _processors(unrelated))
    handle.restore()
    assert all(processor._attention_backend is None for processor in _processors(sibling.transformer))
    again = selection.configure_attention(base.transformer, "cudnn-v1")
    assert again is not None
    again.restore()


def test_shared_processor_identity_is_restored_only_once(bindings):
    model = Transformer()
    shared = model.transformer_blocks[0].attn.processor
    model.single_transformer_blocks[0].attn.processor = shared
    handle = selection.configure_attention(model, "cudnn-v1")
    assert handle is not None and len(handle._snapshots) == 3
    written = shared.backend_writes
    handle.restore()
    assert shared.backend_writes == written + 1
    assert shared._attention_backend is None and "_attention_backend" not in vars(shared)


@pytest.mark.parametrize("all_blocks", [False, True])
def test_other_transformer_cannot_take_shared_processor_ownership(bindings, all_blocks):
    model = Transformer()
    other = Transformer()
    if all_blocks:
        other.transformer_blocks = model.transformer_blocks
        other.single_transformer_blocks = model.single_transformer_blocks
    else:
        other.single_transformer_blocks[0].attn.processor = model.transformer_blocks[0].attn.processor
    handle = selection.configure_attention(model, "cudnn-v1")
    assert handle is not None
    with pytest.raises(selection.AttentionSelectionError, match="processors already have an owner"):
        selection.configure_attention(other, "cudnn-v1")
    assert other.calls == []
    handle.verify()
    handle.restore()
    again = selection.configure_attention(other, "cudnn-v1")
    assert again is not None
    again.restore()


@pytest.mark.parametrize("change", [
    lambda model: model.config.update(num_layers=3),
    lambda model: model.config.update(num_single_layers=True),
    lambda model: model.single_transformer_blocks.clear(),
    lambda model: model.missing.add("single_transformer_blocks.1.attn.processor"),
    lambda model: model.extra.update({"outside.processor": Processor()}),
    lambda model: model.reported.update({"transformer_blocks.0.attn.processor": Processor()}),
    lambda model: setattr(model.transformer_blocks[0], "attn", object()),
    lambda model: setattr(model.transformer_blocks[0].attn, "processor", object()),
    lambda model: setattr(model.transformer_blocks[0].attn.processor, "_parallel_config", object()),
    lambda model: setattr(model, "set_attention_backend", None),
])
def test_coverage_gaps_refuse_before_any_mutation(bindings, change):
    model = Transformer()
    change(model)
    original = [(processor, dict(vars(processor))) for processor in model.attn_processors.values()
                if type(processor) is Processor] if type(model.transformer_blocks[0].attn) is Attention else []
    with pytest.raises(selection.AttentionSelectionError):
        selection.configure_attention(model, "cudnn-v1")
    assert model.calls == [] and Registry.active == "unrelated-default"
    assert all(vars(processor) == values for processor, values in original)


@pytest.mark.parametrize("which", ["transformer", "joint", "single", "attention", "processor"])
def test_subclasses_and_ipadapter_style_processors_are_not_accepted(bindings, which):
    model = Transformer()
    if which == "transformer":
        model = type("ChangedTransformer", (Transformer,), {})()
    elif which == "joint":
        model.transformer_blocks[0] = type("ChangedJoint", (Joint,), {})()
    elif which == "single":
        model.single_transformer_blocks[0] = type("ChangedSingle", (Single,), {})()
    elif which == "attention":
        model.transformer_blocks[0].attn = type("ChangedAttention", (Attention,), {})()
    else:
        model.transformer_blocks[0].attn.processor = type("IPAdapterProcessor", (Processor,), {})()
    with pytest.raises(selection.AttentionSelectionError, match="exact|Unsupported"):
        selection.configure_attention(model, "cudnn-v1")
    assert model.calls == [] and Registry.active == "unrelated-default"


def test_partial_setter_failure_restores_all_processors_and_original_error(bindings):
    model = Transformer()
    model.transformer_blocks[1].attn.processor.fail_target = True
    before = _before(model)
    with pytest.raises(RuntimeError, match="injected target failure"):
        selection.configure_attention(model, "cudnn-v1")
    _unchanged(before)
    assert Registry.active == "unrelated-default"


def test_failure_after_global_mutation_restores_registry_and_processor_values(bindings):
    model = Transformer()
    original = RuntimeError("setter changed the registry then failed")

    def fail():
        raise original

    model.after_set = fail
    before = _before(model)
    with pytest.raises(RuntimeError) as caught:
        selection.configure_attention(model, "cudnn-v1")
    assert caught.value is original
    _unchanged(before)
    assert Registry.active == "unrelated-default"


@pytest.mark.parametrize("before_yield", [False, True])
def test_context_entry_or_exit_failure_keeps_original_error_and_restores_fields(bindings, monkeypatch, before_yield):
    original = RuntimeError("injected context failure")

    @contextlib.contextmanager
    def broken_context(backend):
        if before_yield:
            raise original
        with backend_context(backend):
            yield
        raise original

    monkeypatch.setattr(selection, "_load_bindings", lambda: replace(bindings, backend_context=broken_context))
    model = Transformer()
    before = _before(model)
    with pytest.raises(RuntimeError) as caught:
        selection.configure_attention(model, "cudnn-v1")
    assert caught.value is original
    assert model.calls == ([] if before_yield else ["_native_cudnn"])
    _unchanged(before)
    assert Registry.active == "unrelated-default"


def test_interruption_after_mutation_also_rolls_back(bindings):
    model = Transformer()
    original = KeyboardInterrupt("injected interruption")

    def stop():
        raise original

    model.after_set = stop
    before = _before(model)
    with pytest.raises(KeyboardInterrupt) as caught:
        selection.configure_attention(model, "cudnn-v1")
    assert caught.value is original
    _unchanged(before)
    assert Registry.active == "unrelated-default"


def test_silently_skipped_processor_is_a_setup_failure_with_rollback(bindings):
    model = Transformer()
    model.skip = 2
    before = _before(model)
    with pytest.raises(selection.AttentionSelectionError, match="backend changed"):
        selection.configure_attention(model, "cudnn-v1")
    _unchanged(before)
    assert Registry.active == "unrelated-default"


def test_replacement_during_setup_is_refused_and_original_processors_restore(bindings):
    model = Transformer()
    before = _before(model)
    replacement = Processor()
    model.after_set = lambda: setattr(model.transformer_blocks[0].attn, "processor", replacement)
    with pytest.raises(selection.AttentionSelectionError, match="coverage changed"):
        selection.configure_attention(model, "cudnn-v1")
    _unchanged(before)
    assert replacement._attention_backend is None
    assert Registry.active == "unrelated-default"


def test_rollback_failure_keeps_original_cause_and_still_restores_other_processors(bindings):
    model = Transformer()
    first, second, *others = _processors(model)
    first.fail_restore = True
    second.fail_target = True
    with pytest.raises(RuntimeError, match="injected target failure") as caught:
        selection.configure_attention(model, "cudnn-v1")
    assert "Attention rollback failed; discard this transformer" in " ".join(caught.value.__notes__)
    assert first._attention_backend == "_native_cudnn"
    assert all(processor._attention_backend is None for processor in [second, *others])
    assert Registry.active == "unrelated-default"


@pytest.mark.parametrize("change", [
    lambda model: setattr(model.transformer_blocks[0].attn.processor, "_attention_backend", "native"),
    lambda model: setattr(model.single_transformer_blocks[0].attn.processor, "_parallel_config", object()),
    lambda model: model.extra.update({"new.processor": Processor()}),
    lambda model: setattr(model.transformer_blocks[0].attn, "processor", Processor()),
])
def test_handle_detects_later_changes_and_restores_only_its_original_processors(bindings, change):
    model = Transformer()
    before = _before(model)
    handle = selection.configure_attention(model, "cudnn-v1")
    assert handle is not None
    change(model)
    with pytest.raises(selection.AttentionSelectionError):
        handle.verify()
    with pytest.raises(selection.AttentionSelectionError, match="discard this transformer"):
        handle.restore()
    _unchanged(before)
    assert handle._changed is None and handle._transformer is None
    handle.restore()


@pytest.mark.parametrize("replace_block", [False, True])
def test_handle_detects_exact_module_replacement_even_with_the_original_processor(bindings, replace_block):
    model = Transformer()
    before = _before(model)
    handle = selection.configure_attention(model, "cudnn-v1")
    assert handle is not None
    original = model.transformer_blocks[0].attn.processor
    if replace_block:
        model.transformer_blocks[0] = Joint()
    else:
        model.transformer_blocks[0].attn = Attention()
    model.transformer_blocks[0].attn.processor = original
    with pytest.raises(selection.AttentionSelectionError, match="coverage changed"):
        handle.verify()
    with pytest.raises(selection.AttentionSelectionError, match="discard this transformer"):
        handle.restore()
    _unchanged(before)


def test_failed_restore_attempts_all_snapshots_and_can_finish_idempotently(bindings):
    model = Transformer()
    before = _before(model)
    first, *others = _processors(model)
    handle = selection.configure_attention(model, "cudnn-v1")
    assert handle is not None
    first.fail_restore = True
    with pytest.raises(selection.AttentionSelectionError, match="restoration failed") as caught:
        handle.restore()
    assert isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert all(processor._attention_backend is None for processor in others)
    with pytest.raises(selection.AttentionSelectionError, match="already has an attention owner"):
        selection.configure_attention(model, "cudnn-v1")
    first.fail_restore = False
    handle.restore()
    handle.restore()
    _unchanged(before)


def _installed(relative):
    # Distribution metadata finds source without importing a module whose
    # initialization could ask for a device.
    return Path(distribution("diffusers").locate_file("diffusers")) / relative


def _function(path, name, namespace, *, owner=None, remove_imports=False):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if owner is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == owner)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    if remove_imports:
        imports = [node for node in function.body if isinstance(node, ast.ImportFrom)]
        assert [(node.module, node.level) for node in imports] == [
            ("attention", 1), ("attention_dispatch", 1), ("attention_processor", 1),
        ]
        function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("partial_failure", [False, True])
def test_installed_public_context_and_setter_restore_global_state_against_stubs(bindings, monkeypatch, partial_failure):
    # Execute the installed bodies, not their imports or an attention forward.
    # The only stripped statements are the three asserted relative imports;
    # their defining types/APIs are provided as tensor-free stand-ins.
    class Backend(str, Enum):
        NATIVE = "native"
        _NATIVE_CUDNN = "_native_cudnn"

    namespace = {
        "contextlib": contextlib, "AttentionBackendName": Backend,
        "_AttentionBackendRegistry": Registry,
        "_check_attention_backend_requirements": lambda backend: None,
        "_maybe_download_kernel_for_backend": lambda backend: None,
        "AttentionModuleMixin": Attention, "Attention": Attention, "MochiAttention": type("Mochi", (), {}),
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    monkeypatch.setattr(Registry, "_backends", {Backend.NATIVE: object(), Backend._NATIVE_CUDNN: object()}, raising=False)
    # The actual context reads this field; the stub registry's public setter
    # writes it as well as active, without touching the real Diffusers registry.
    monkeypatch.setattr(Registry, "_active_backend", "unrelated-default", raising=False)

    def set_active_backend(cls, backend):
        cls.active = cls._active_backend = backend

    monkeypatch.setattr(Registry, "set_active_backend", classmethod(set_active_backend))
    context = _function(_installed("models/attention_dispatch.py"), "attention_backend", namespace)
    setter = _function(_installed("models/modeling_utils.py"), "set_attention_backend", namespace,
                       owner="ModelMixin", remove_imports=True)
    monkeypatch.setattr(Transformer, "set_attention_backend", setter)
    monkeypatch.setattr(selection, "_load_bindings", lambda: selection._Bindings(
        Transformer, Joint, Single, Attention, Processor, context, Backend._NATIVE_CUDNN,
    ))
    model = Transformer()
    before = _before(model)
    if partial_failure:
        model.transformer_blocks[1].attn.processor.fail_target = True
        with pytest.raises(RuntimeError, match="injected target failure"):
            selection.configure_attention(model, "cudnn-v1")
    else:
        handle = selection.configure_attention(model, "cudnn-v1")
        assert handle is not None
        assert all(processor._attention_backend is Backend._NATIVE_CUDNN for processor in _processors(model))
        handle.restore()
    _unchanged(before)
    assert Registry.active == Registry._active_backend == "unrelated-default"
