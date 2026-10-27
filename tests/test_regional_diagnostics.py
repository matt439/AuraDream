"""Pure regional observations/tokens, never framework/compiler/device work."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

import pytest

from backend.backends.regional_compile import (
    RegionFacts, RegionalDiagnosticOwner, RegionalDiagnostics, RegionalRegistration,
)


def test_unpublished_is_cold_and_cannot_begin_an_observed_forward():
    owner = RegionalDiagnosticOwner()
    assert owner.snapshot() is None
    with pytest.raises(RuntimeError, match="published live"):
        owner.begin_first_call()
    assert owner.snapshot() is None


@pytest.mark.parametrize("identity", [0, -1, True, 1.0, None, "private-payload"])
def test_invalid_identity_refuses_without_publication_or_echo(identity):
    owner = RegionalDiagnosticOwner()
    with pytest.raises(ValueError, match="positive per-load identity") as failure:
        owner.publish(identity)
    assert "private-payload" not in str(failure.value)
    assert owner.snapshot() is None


def test_published_value_is_frozen_bounded_and_independent_of_mutable_copies():
    owner = RegionalDiagnosticOwner()
    owner.publish(1)
    value = owner.snapshot()
    assert value == RegionalDiagnostics(1)
    with pytest.raises(FrozenInstanceError):
        value.first_call = "returned"
    copied = asdict(value)
    assert copied == {
        "load_id": 1, "recipe": "regional-v1", "registration": "registered",
        "first_call": "pending", "failure_stage": None,
    }
    copied["first_call"] = "returned"
    assert owner.snapshot() == value
    assert json.loads(json.dumps(asdict(value))) == asdict(value)


@pytest.mark.parametrize("changes", [
    {"load_id": 0}, {"load_id": True}, {"recipe": "other"}, {"registration": "warm"},
    {"first_call": "warm"}, {"first_call": None},
    {"failure_stage": "forward"}, {"first_call": "running", "failure_stage": "forward"},
    {"first_call": "returned", "failure_stage": "policy_exit"},
    {"first_call": "failed"}, {"first_call": "cancelled", "failure_stage": "preflight"},
])
def test_value_refuses_unbounded_or_misleading_combinations(changes):
    with pytest.raises(ValueError):
        RegionalDiagnostics(**{"load_id": 1, **changes})


@pytest.mark.parametrize("outcome,stage", [
    ("returned", None), ("failed", "forward"), ("failed", "verification"),
    ("failed", "policy_exit"), ("cancelled", "forward"), ("cancelled", "verification"),
    ("cancelled", "policy_exit"),
])
def test_first_attempt_freezes_one_terminal_outcome_without_claiming_warmth(outcome, stage):
    owner = RegionalDiagnosticOwner()
    owner.publish(12)
    token = owner.begin_first_call()
    running = owner.snapshot()
    assert running == RegionalDiagnostics(12, "running")
    assert owner.finish_first_call(token, outcome, failure_stage=stage)
    terminal = RegionalDiagnostics(12, outcome, stage)
    assert owner.snapshot() == terminal
    assert running.first_call == "running"  # A handed-off snapshot never changes.
    assert owner.begin_first_call() is None
    assert not owner.finish_first_call(None, "returned")
    assert not owner.finish_first_call(token, "returned")
    assert owner.snapshot() == terminal


def test_same_publication_is_idempotent_without_replacing_running_or_terminal_state():
    owner = RegionalDiagnosticOwner()
    owner.publish(3)
    token = owner.begin_first_call()
    owner.publish(3)
    assert owner.snapshot() == RegionalDiagnostics(3, "running")
    owner.finish_first_call(token, "failed", failure_stage="forward")
    owner.publish(3)
    assert owner.snapshot() == RegionalDiagnostics(3, "failed", "forward")
    with pytest.raises(RuntimeError, match="another load"):
        owner.publish(4)
    assert owner.snapshot() == RegionalDiagnostics(3, "failed", "forward")


def test_foreign_token_uses_identity_and_does_not_consume_the_real_attempt():
    class Foreign:
        def __eq__(self, _other):
            raise AssertionError("diagnostic tokens must never compare equality")
    owner = RegionalDiagnosticOwner()
    owner.publish(4)
    token = owner.begin_first_call()
    assert not owner.finish_first_call(Foreign(), "returned")
    assert owner.snapshot() == RegionalDiagnostics(4, "running")
    assert owner.finish_first_call(token, "returned")


@pytest.mark.parametrize("outcome,stage", [
    ("pending", None), ("arbitrary-payload", None), ("failed", None),
    ("cancelled", "preflight"), ("returned", "forward"),
])
def test_malformed_active_update_keeps_the_running_token_available(outcome, stage):
    owner = RegionalDiagnosticOwner()
    owner.publish(5)
    token = owner.begin_first_call()
    with pytest.raises(ValueError) as failure:
        owner.finish_first_call(token, outcome, failure_stage=stage)
    assert "arbitrary-payload" not in str(failure.value)
    assert owner.snapshot() == RegionalDiagnostics(5, "running")
    assert owner.finish_first_call(token, "returned")


@pytest.mark.parametrize("phase", ["unpublished", "pending", "running", "returned", "failed", "cancelled"])
def test_retirement_is_idempotent_and_never_republishes_or_accepts_late_updates(phase):
    owner = RegionalDiagnosticOwner()
    token = None
    if phase != "unpublished":
        owner.publish(6)
    if phase in ("running", "returned", "failed", "cancelled"):
        token = owner.begin_first_call()
    if phase in ("returned", "failed", "cancelled"):
        owner.finish_first_call(token, phase, failure_stage=None if phase == "returned" else "forward")
    owner.retire()
    owner.retire()
    assert owner.snapshot() is None
    assert not owner.finish_first_call(token, "returned")
    with pytest.raises(RuntimeError, match="retired"):
        owner.publish(6)
    with pytest.raises(RuntimeError, match="published live"):
        owner.begin_first_call()


def test_old_owner_token_cannot_modify_a_replacement_load():
    old, new = RegionalDiagnosticOwner(), RegionalDiagnosticOwner()
    old.publish(7)
    token = old.begin_first_call()
    old.retire()
    new.publish(8)
    current = new.begin_first_call()
    assert not old.finish_first_call(token, "failed", failure_stage="forward")
    assert not new.finish_first_call(token, "failed", failure_stage="forward")
    assert new.snapshot() == RegionalDiagnostics(8, "running")
    assert new.finish_first_call(current, "returned")


def test_duplicate_first_begin_is_refused_without_consuming_the_live_token():
    owner = RegionalDiagnosticOwner()
    owner.publish(9)
    token = owner.begin_first_call()
    with pytest.raises(RuntimeError, match="already running"):
        owner.begin_first_call()
    assert owner.snapshot() == RegionalDiagnostics(9, "running")
    assert owner.finish_first_call(token, "returned")


def test_competing_first_begins_have_one_winner_and_no_implicit_second_attempt():
    owner = RegionalDiagnosticOwner()
    owner.publish(10)
    barrier = threading.Barrier(2)
    def begin():
        barrier.wait(timeout=5)
        try:
            return owner.begin_first_call()
        except RuntimeError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: begin(), range(2)))
    tokens = [token for token in results if token is not None]
    assert len(tokens) == 1
    assert owner.finish_first_call(tokens[0], "returned")


def test_reader_keeps_no_registration_or_model_reference_and_close_retires_it():
    class Model:
        pass
    root = Model()
    reference = weakref.ref(root)
    facts = RegionFacts("regional-v1", (("FluxTransformerBlock", 2), ("FluxSingleTransformerBlock", 1)))
    registration = RegionalRegistration(root, facts)
    reader = registration.diagnostics.snapshot
    registration.diagnostics.publish(11)
    snapshot = reader()
    registration.close()
    registration.close()
    del registration, root
    gc.collect()
    assert reference() is None and reader() is None
    assert snapshot == RegionalDiagnostics(11)


def test_diagnostic_reader_without_explicit_close_still_retains_no_weights():
    class Model:
        pass
    root = Model()
    reference = weakref.ref(root)
    registration = RegionalRegistration(root, RegionFacts("regional-v1", (("FluxTransformerBlock", 1), ("FluxSingleTransformerBlock", 1))))
    owner = registration.diagnostics
    owner.publish(13)
    reader = owner.snapshot
    del registration, root
    gc.collect()
    assert reference() is None
    assert reader() == RegionalDiagnostics(13)
    owner.retire()
    assert reader() is None


def test_helper_import_is_cold_in_a_fresh_guarded_process():
    source = r'''
import importlib.abc
import sys
blocked = {"torch", "torchao", "diffusers", "triton", "huggingface_hub"}
class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError("regional diagnostics imported a runtime library")
        return None
sys.meta_path.insert(0, ColdOnly())
from backend.backends.regional_compile import RegionalDiagnosticOwner
owner = RegionalDiagnosticOwner()
owner.publish(1)
token = owner.begin_first_call()
assert owner.snapshot().first_call == "running"
owner.finish_first_call(token, "returned")
assert owner.snapshot().first_call == "returned"
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
