"""Cold HTTP observations describe a published load, never qualification."""
from __future__ import annotations

import pytest

from backend import main, model_selection
from backend.backends.regional_compile import RegionalDiagnosticOwner


def _published(load_id=1, outcome="pending"):
    owner = RegionalDiagnosticOwner()
    owner.publish(load_id)
    if outcome != "pending":
        token = owner.begin_first_call()
        if outcome != "running":
            owner.finish_first_call(token, outcome, failure_stage="forward" if outcome in {"failed", "cancelled"} else None)
    return owner


def _catalogue(monkeypatch):
    # This surface needs neither a model-cache inventory nor external services.
    monkeypatch.setattr(main, "details_for", lambda *_args, **_kwargs: {})


@pytest.mark.parametrize("outcome", ["pending", "running", "returned", "failed", "cancelled"])
def test_status_serves_only_bounded_per_load_observation(client, fake, monkeypatch, outcome):
    owner = _published(7, outcome)
    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: owner.snapshot)
    body = client.get("/api/status").json()
    assert body["execution_diagnostics"] == {
        "load_id": 7, "recipe": "regional-v1", "registration": "registered",
        "first_call": outcome, "failure_stage": "forward" if outcome in {"failed", "cancelled"} else None,
    }
    assert fake.requests == [] and fake.loads == 0


def test_default_unpublished_and_retired_diagnostics_stay_null(client, fake, monkeypatch):
    assert client.get("/api/status").json()["execution_diagnostics"] is None
    owner = RegionalDiagnosticOwner()
    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: owner.snapshot)
    assert client.get("/api/status").json()["execution_diagnostics"] is None
    owner.publish(4)
    owner.retire()
    assert client.get("/api/status").json()["execution_diagnostics"] is None
    assert fake.requests == [] and fake.loads == 0


def test_status_reads_current_load_after_awaited_description(client, fake, monkeypatch):
    old, replacement = _published(1, "returned"), _published(2)
    current = [old]
    describe = fake.describe

    def stale_description():
        result = dict(describe(), execution_diagnostics=old.snapshot())
        old.retire()
        current[0] = replacement
        return result

    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: current[0].snapshot)
    monkeypatch.setattr(fake, "describe", stale_description)
    body = client.get("/api/status").json()
    assert body["execution_diagnostics"]["load_id"] == 2
    assert body["execution_diagnostics"]["first_call"] == "pending"
    assert fake.requests == []


def test_models_reports_actual_load_separately_from_draft_options(client, fake, monkeypatch):
    _catalogue(monkeypatch)
    owner = _published(9, "returned")
    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: owner.snapshot)
    body = client.get("/api/models?compilation=regional-v1").json()
    assert body["execution"] is None
    assert body["execution_diagnostics"]["load_id"] == 9
    assert body["execution_diagnostics"]["first_call"] == "returned"
    assert all("execution_diagnostics" not in detail for detail in body["details"].values())
    assert fake.requests == [] and fake.loads == 0


def test_models_omits_live_observation_when_catalogue_selection_changed(client, fake, monkeypatch):
    _catalogue(monkeypatch)
    owner = _published(3)
    reads = []

    def reader():
        reads.append(True)
        return owner.snapshot

    async def switched(operation, **_kwargs):
        result = operation()
        monkeypatch.setattr(model_selection, "_selection", (None, model_selection.revision() + 1))
        return result

    monkeypatch.setattr(fake, "execution_diagnostics_reader", reader)
    monkeypatch.setattr(main, "_detached", switched)
    body = client.get("/api/models").json()
    assert body["execution_diagnostics"] is None and reads == []


def test_models_reads_same_selection_reload_after_metadata_await(client, fake, monkeypatch):
    _catalogue(monkeypatch)
    old, replacement = _published(1, "returned"), _published(2)
    current = [old]

    async def reloaded(operation, **_kwargs):
        result = operation()
        old.retire()
        current[0] = replacement
        return result

    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: current[0].snapshot)
    monkeypatch.setattr(main, "_detached", reloaded)
    body = client.get("/api/models").json()
    assert body["execution_diagnostics"]["load_id"] == 2
    assert body["execution_diagnostics"]["first_call"] == "pending"
    assert fake.requests == [] and fake.loads == 0
