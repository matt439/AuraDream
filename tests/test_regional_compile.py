"""Inactive registration contracts: stubs and installed config contexts only."""
from __future__ import annotations

import dataclasses
import gc
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import ModuleType

import pytest

from backend.backends import Cancelled
from backend.backends import regional_compile as regional


ARGS = {"backend": "inductor", "fullgraph": True, "dynamic": False, "mode": "default"}


@pytest.fixture(autouse=True)
def no_compiler_or_device_calls(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("these helper tests never compile, allocate tensors or query a device")

    monkeypatch.setattr(torch, "compile", forbidden)
    monkeypatch.setattr(torch.nn.Module, "compile", forbidden)
    for name in ("tensor", "empty", "zeros", "ones", "randn"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("is_available", "device_count", "get_device_capability", "current_device", "synchronize", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


class FluxTransformerBlock:
    def __init__(self, failure=None):
        self._compiled_call_impl = None
        self.calls = []
        self.failure = failure

    def compile(self, **kwargs):
        self.calls.append(kwargs)
        self._compiled_call_impl = object()
        if self.failure is not None:
            raise self.failure


class FluxSingleTransformerBlock(FluxTransformerBlock):
    pass


class FluxTransformer2DModel:
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    def __init__(self, blocks=None):
        self.blocks = blocks if blocks is not None else [
            FluxTransformerBlock(), FluxTransformerBlock(), FluxSingleTransformerBlock(),
        ]
        self.calls = []

    def modules(self):
        yield self
        yield object()  # Surrounding Python/other module kinds stay eager.
        yield from self.blocks

    def compile_repeated_blocks(self, **kwargs):
        self.calls.append(kwargs)
        # Match the installed library's sequential, in-place registration.
        for block in self.modules():
            if type(block).__name__ in self._repeated_blocks:
                block.compile(**kwargs)


@pytest.fixture
def stub_types(monkeypatch):
    monkeypatch.setattr(regional, "_flux_types", lambda: (
        FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock,
    ))
    events = []

    @contextmanager
    def patch():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(regional, "_dynamo_patch", patch)
    return events


def test_module_import_is_cold_and_does_not_import_optional_runtime_libraries():
    done = subprocess.run([sys.executable, "-c", (
        "import sys; import backend.backends.regional_compile; "
        "assert not {'torch', 'diffusers', 'torch._dynamo', 'triton'} & sys.modules.keys()"
    )], capture_output=True, text=True, timeout=15)
    assert done.returncode == 0, done.stdout + done.stderr


def test_discovery_is_read_only_and_facts_are_frozen(stub_types):
    model = FluxTransformer2DModel()
    facts = regional.inspect_regions(model)
    assert facts.recipe == "regional-v1" and facts.total_blocks == 3
    assert facts.block_counts == (("FluxTransformerBlock", 2), ("FluxSingleTransformerBlock", 1))
    assert model.calls == [] and all(block.calls == [] for block in model.blocks)
    assert stub_types == []
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.recipe = "other"


def test_registration_forwards_exact_recipe_only_after_discovery(stub_types):
    model = FluxTransformer2DModel()
    discarded = []
    registered = regional.register_regions(model, discard_on_failure=lambda: discarded.append(True))
    assert model.calls == [ARGS]
    assert all(block.calls == [ARGS] for block in model.blocks)
    assert registered.facts.total_blocks == 3 and not registered.closed
    assert discarded == [] and stub_types == ["enter", "exit"]
    markers = [block._compiled_call_impl for block in model.blocks]
    registered.close()
    registered.close()
    assert registered.closed and registered.facts is None
    assert [block._compiled_call_impl for block in model.blocks] == markers
    assert discarded == []  # Host unload owns the actual pipeline separately.


@pytest.mark.parametrize("declared", [[], None, "FluxTransformerBlock", ["FluxTransformerBlock"],
                                    ["FluxTransformerBlock", "unknown"],
                                    ["FluxTransformerBlock", "FluxTransformerBlock"], [False, 1]])
def test_invalid_declarations_discard_before_any_registration(stub_types, declared):
    model = FluxTransformer2DModel()
    model._repeated_blocks = declared
    discarded = []
    with pytest.raises(ValueError, match="declarations"):
        regional.register_regions(model, discard_on_failure=lambda: discarded.append(True))
    assert model.calls == [] and all(block.calls == [] for block in model.blocks)
    assert discarded == [True] and stub_types == []


@pytest.mark.parametrize("blocks", [[], [FluxTransformerBlock()], [FluxSingleTransformerBlock()]])
def test_both_families_must_exist_before_registration(stub_types, blocks):
    model = FluxTransformer2DModel(blocks)
    with pytest.raises(ValueError, match="each declared"):
        regional.inspect_regions(model)
    assert model.calls == [] and stub_types == []


def test_names_and_subclasses_cannot_impersonate_the_defining_model_class(stub_types):
    class Derived(FluxTransformer2DModel):
        pass

    for model in (object(), type("FluxTransformer2DModel", (), {})(), Derived()):
        with pytest.raises(ValueError, match="installed FluxTransformer2DModel"):
            regional.inspect_regions(model)
    assert stub_types == []


def test_a_compiled_root_is_refused_before_any_regional_registration(stub_types):
    model = FluxTransformer2DModel()
    marker = object()
    model._compiled_call_impl = marker
    root_calls, discarded = [], []
    model.compile = lambda **kwargs: root_calls.append(kwargs)
    with pytest.raises(ValueError, match="transformer is already compiled"):
        regional.register_regions(model, discard_on_failure=lambda: discarded.append(True))
    assert root_calls == [] and model.calls == []
    assert all(block.calls == [] and block._compiled_call_impl is None for block in model.blocks)
    assert model._compiled_call_impl is marker
    assert discarded == [True] and stub_types == []


@pytest.mark.parametrize("problem", ["spoof-family", "missing-compile", "already-compiled", "missing-model-api"])
def test_every_region_is_checked_before_the_first_mutation(stub_types, problem):
    first, last = FluxTransformerBlock(), FluxSingleTransformerBlock()
    model = FluxTransformer2DModel([first, last])
    if problem == "spoof-family":
        model.blocks[-1] = type("FluxSingleTransformerBlock", (), {})()
    elif problem == "missing-compile":
        last.compile = None
    elif problem == "already-compiled":
        last._compiled_call_impl = object()
    else:
        model.compile_repeated_blocks = None
    discarded = []
    with pytest.raises(ValueError):
        regional.register_regions(model, discard_on_failure=lambda: discarded.append(True))
    assert first.calls == [] and model.calls == []
    assert discarded == [True] and stub_types == []


@pytest.mark.parametrize("failure", [RuntimeError("late setup failure"), Cancelled(), KeyboardInterrupt()])
def test_partial_registration_discards_unpublished_candidate_and_preserves_error(stub_types, failure):
    first, last = FluxTransformerBlock(), FluxSingleTransformerBlock(failure)
    model = FluxTransformer2DModel([first, last])
    discarded, published = [], []
    with pytest.raises(type(failure)) as raised:
        registered = regional.register_regions(model, discard_on_failure=lambda: discarded.append(True))
        published.append(registered)
    assert raised.value is failure
    assert discarded == [True] and published == []
    assert first.calls == last.calls == [ARGS]
    assert first._compiled_call_impl is not None  # Discard, do not unwrap/reuse.
    assert stub_types == ["enter", "exit"]


def test_discard_failure_does_not_replace_the_original_exception_or_cause(stub_types):
    original = RuntimeError("registration failed")
    cause = ValueError("source cause")
    original.__cause__ = cause
    model = FluxTransformer2DModel([FluxTransformerBlock(), FluxSingleTransformerBlock(original)])

    def discard():
        raise OSError("discard failed")

    with pytest.raises(RuntimeError) as raised:
        regional.register_regions(model, discard_on_failure=discard)
    assert raised.value is original and original.__cause__ is cause
    assert any("OSError: discard failed" in note for note in original.__notes__)


def test_policy_exit_failure_also_discards_the_candidate(stub_types, monkeypatch):
    failure = RuntimeError("policy exit failed")

    @contextmanager
    def patch():
        yield
        raise failure

    monkeypatch.setattr(regional, "_dynamo_patch", patch)
    discarded = []
    with pytest.raises(RuntimeError) as raised:
        regional.register_regions(FluxTransformer2DModel(), discard_on_failure=lambda: discarded.append(True))
    assert raised.value is failure and discarded == [True]


def test_close_releases_owned_model_and_diagnostics_references(stub_types):
    model = FluxTransformer2DModel()
    reference = weakref.ref(model)
    registered = regional.register_regions(model, discard_on_failure=lambda: None)
    del model
    gc.collect()
    assert reference() is not None
    registered.close()
    gc.collect()
    assert registered.closed and registered.facts is None and reference() is None


@pytest.fixture
def installed_config(monkeypatch):
    # Exercise the installed ConfigModule.patch implementation on an in-memory
    # flag container. Importing real Dynamo/Flux can probe optional devices, so
    # their import graphs are deliberately not executed by these tests.
    from torch.utils._config_module import install_config_module

    config = ModuleType("test_regional_policy_config")
    config.__file__ = __file__  # install_config_module reads source annotations.
    config.suppress_errors = True
    config.fail_on_recompile_limit_hit = False
    install_config_module(config)
    original = regional.import_module

    def imported(name, package=None):
        if name == "torch._dynamo.config":
            return config
        return original(name, package)

    monkeypatch.setattr(regional, "import_module", imported)
    return config


def flags(config):
    return config.suppress_errors, config.fail_on_recompile_limit_hit


@pytest.mark.parametrize("failure", [None, RuntimeError("call failed"), Cancelled(), KeyboardInterrupt()])
def test_installed_config_policy_restores_flags_on_every_exit(installed_config, failure):
    config = installed_config
    assert flags(config) == (True, False)
    try:
        with regional.strict_compiler_policy():
            assert flags(config) == (False, True)
            if failure is not None:
                raise failure
    except BaseException as caught:
        assert caught is failure
    assert flags(config) == (True, False)


def test_installed_config_nested_policies_restore_their_own_prior_values(installed_config):
    config = installed_config
    with regional.strict_compiler_policy():
        assert flags(config) == (False, True)
        with config.patch({"suppress_errors": True, "fail_on_recompile_limit_hit": False}):
            with regional.strict_compiler_policy():
                assert flags(config) == (False, True)
            assert flags(config) == (True, False)
        assert flags(config) == (False, True)
    assert flags(config) == (True, False)


def test_installed_config_policy_is_context_local_across_threads(installed_config):
    config = installed_config

    def worker():
        before = flags(config)
        with regional.strict_compiler_policy():
            inside = flags(config)
        return before, inside, flags(config)

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(flags, config).result(timeout=5) == (True, False)
        with regional.strict_compiler_policy():
            assert pool.submit(worker).result(timeout=5) == ((True, False), (False, True), (True, False))
            assert flags(config) == (False, True)
        assert pool.submit(flags, config).result(timeout=5) == (True, False)
    assert flags(config) == (True, False)


def test_missing_policy_flag_fails_before_changing_any_flag(installed_config):
    config = installed_config
    del config._config["fail_on_recompile_limit_hit"]
    with pytest.raises(AttributeError):
        with regional.strict_compiler_policy():
            pytest.fail("an incomplete strict policy was entered")
    assert config.suppress_errors is True
