"""Import-cold observations of the measured Refine Native attention policy."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from backend import config, refine_qualification
from backend.backends.base import _refine_native_observed


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(config, "ATTENTION_ENVIRONMENT_WARNING", None)
    monkeypatch.delenv("DIFFUSERS_ATTN_BACKEND", raising=False)
    native_enum, other_enum = object(), object()

    def function(*args, **kwargs):
        raise AssertionError("policy observation must not execute attention")

    state = SimpleNamespace(active=(native_enum, function), reads=0)

    def read():
        state.reads += 1
        return state.active

    constants = ModuleType("diffusers.utils.constants")
    constants.DIFFUSERS_ATTN_BACKEND = "native"
    dispatch = ModuleType("diffusers.models.attention_dispatch")
    dispatch.DIFFUSERS_ATTN_BACKEND = "native"
    dispatch.AttentionBackendName = SimpleNamespace(NATIVE=native_enum)
    dispatch._native_attention = function
    dispatch._AttentionBackendRegistry = SimpleNamespace(get_active_backend=read)
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setitem(sys.modules, dispatch.__name__, dispatch)
    return SimpleNamespace(state=state, constants=constants, dispatch=dispatch,
                           enum=native_enum, other=other_enum, function=function)


def refusal(native_attention):
    facts = refine_qualification.LoaderFacts(
        "fp8", "model", "cuda", "win32", "0.42.1", "NVIDIA GeForce RTX 5080", native_attention,
    )
    return refine_qualification.loader_refusal("circlestone-labs/Anima", facts)


@pytest.mark.parametrize("value", ["flash", "", " native", "NATIVE", "unrecorded-private-value"])
def test_late_environment_change_is_reobserved_and_held_with_fixed_words(native, monkeypatch, value):
    assert _refine_native_observed() is True
    monkeypatch.setenv("DIFFUSERS_ATTN_BACKEND", value)
    assert _refine_native_observed() is False
    reason = refusal(_refine_native_observed())
    assert reason == (
        "Anima Turbo Refine is held: the external attention override; only Native attention was measured."
    )
    if value and value not in {"native", "NATIVE"}:
        assert value not in reason
    monkeypatch.setenv("DIFFUSERS_ATTN_BACKEND", "native")
    assert _refine_native_observed() is True and refusal(True) is None
    monkeypatch.delenv("DIFFUSERS_ATTN_BACKEND")
    assert _refine_native_observed() is True


def test_startup_warning_cannot_be_hidden_by_a_later_native_environment(native, monkeypatch):
    monkeypatch.setenv("DIFFUSERS_ATTN_BACKEND", "native")
    monkeypatch.setattr(config, "ATTENTION_ENVIRONMENT_WARNING", "fixed startup observation")
    assert _refine_native_observed() is False
    assert "external attention override" in refusal(_refine_native_observed())
    monkeypatch.setattr(config, "ATTENTION_ENVIRONMENT_WARNING", None)
    assert _refine_native_observed() is True


@pytest.mark.parametrize("module_name", ["constants", "dispatch"])
@pytest.mark.parametrize("value", ["flash", "", None])
def test_imported_frozen_default_must_still_be_exact_native(native, monkeypatch, module_name, value):
    assert _refine_native_observed() is True
    module = getattr(native, module_name)
    monkeypatch.setattr(module, "DIFFUSERS_ATTN_BACKEND", value)
    assert _refine_native_observed() is False


@pytest.mark.parametrize("module_name", ["diffusers.utils.constants", "diffusers.models.attention_dispatch"])
def test_explicitly_unavailable_imported_module_is_held(native, monkeypatch, module_name):
    monkeypatch.setitem(sys.modules, module_name, None)
    assert _refine_native_observed() is False


def test_active_backend_and_function_need_identity_not_native_looking_names(native):
    assert _refine_native_observed() is True
    native.state.active = (native.other, native.function)
    assert _refine_native_observed() is False
    native.state.active = ("native", native.function)
    assert _refine_native_observed() is False

    def replacement(*args, **kwargs):
        raise AssertionError("replacement must not execute")

    replacement.__name__ = "_native_attention"
    native.state.active = (native.enum, replacement)
    assert _refine_native_observed() is False
    native.state.active = (native.enum, None)
    assert _refine_native_observed() is False
    native.state.active = (native.enum, native.function)
    assert _refine_native_observed() is True


@pytest.mark.parametrize("outcome", ["raise", "malformed"])
def test_unreadable_registry_fails_closed_without_policy_mutation(native, monkeypatch, outcome):
    before = native.state.active

    def unreadable():
        if outcome == "raise":
            raise RuntimeError("unreadable registry")
        return (native.enum,)

    monkeypatch.setattr(native.dispatch._AttentionBackendRegistry, "get_active_backend", unreadable)
    assert _refine_native_observed() is False
    assert native.state.active == before


def test_cold_absence_observation_imports_no_framework_or_device_probe_in_a_fresh_process():
    script = r'''
import builtins, json, os, socket, sys
attempts = {"framework": 0, "probe": 0, "network": 0}
os.environ.pop("DIFFUSERS_ATTN_BACKEND", None)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {"torch", "diffusers", "transformers"}:
        attempts["framework"] += 1
        raise AssertionError("Native observation imported " + name)
    return original(name, *args, **kwargs)
def no_probe(*args, **kwargs):
    attempts["probe"] += 1
    raise AssertionError("Native observation probed a device")
def no_network(*args, **kwargs):
    attempts["network"] += 1
    raise AssertionError("Native observation reached the network")
builtins.__import__ = guarded
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
from backend import config, card_facts
from backend.backends.base import _refine_native_observed
card_facts.card = card_facts.probe = no_probe
assert config.ATTENTION_ENVIRONMENT_WARNING is None
assert "diffusers.utils.constants" not in sys.modules
assert "diffusers.models.attention_dispatch" not in sys.modules
before = dict(os.environ)
assert _refine_native_observed() is True
assert _refine_native_observed() is True
assert os.environ == before and not any(attempts.values())
print(json.dumps(attempts))
'''
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout) == {"framework": 0, "probe": 0, "network": 0}
