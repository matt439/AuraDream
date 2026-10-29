"""Saved-setup capture holds the checkpoint without entering gallery/GPU work."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from backend import api_presets, main, model_selection, presets
from backend.execution import ExecutionSpec
from backend.schemas_presets import PresetUpdate

OTHER = "cagliostrolab/animagine-xl-4.0"


def capture(**changes):
    return {
        "name": "夜の setup", "model_id": main.services.model_id(),
        "model_revision": model_selection.revision(),
        "payload": {"prompt": "  a cat\n猫  ", "negative_prompt": "  blur  "},
        **changes,
    }


def create(client, **changes):
    response = client.post("/api/presets", json=capture(**changes))
    assert response.status_code == 201, response.text
    return response.json()["items"][-1]


def check(client, item, **changes):
    return client.post(f"/api/presets/{item['id']}/validate", json={
        "revision": item["revision"], "model_revision": model_selection.revision(), **changes,
    })


def test_crud_and_revisioned_validation_keep_the_raw_draft(client, store):
    assert client.get("/api/presets").json()["items"] == []
    assert not presets.path_for(store.settings.db_path).exists()
    item = create(client)
    assert client.get(f"/api/presets/{item['id']}").json() == item
    checked = check(client, item)
    assert checked.status_code == 200, checked.text
    result = checked.json()
    assert result["item_id"] == item["id"] and result["revision"] == 1
    assert result["saved"] == item["payload"]
    assert result["effective"]["seed"] is None and result["saved"]["steps"] is None
    assert item["payload"]["prompt"] == "  a cat\n猫  "
    body = capture(name="Renamed", revision=1)
    response = client.put(f"/api/presets/{item['id']}", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["revision"] == 2
    assert client.put(f"/api/presets/{item['id']}", json=body).status_code == 409
    assert check(client, item).status_code == 409
    assert client.delete(f"/api/presets/{item['id']}?revision=1").status_code == 409
    assert client.delete(f"/api/presets/{item['id']}").status_code == 422
    assert client.delete(f"/api/presets/{item['id']}?revision=2").json()["items"] == []
    assert client.get(f"/api/presets/{item['id']}").status_code == 404


@pytest.mark.parametrize("changes", [
    {"reference": [1]}, {"lora": "a.safetensors"}, {"refine": 1},
    {"mask": "a" * 64}, {"then_refine": {"scale": 1.5}}, {"edit": {"source": 1}},
    {"execution": None}, {"step_cache": None}, {"resume": True},
    {"grid": {"resume": True}}, {"seed": True}, {"width": "1024"},
    {"prompt": "\ud800"}, {"steps": 0}, {"guidance_scale": float("nan")},
])
def test_payload_refuses_unknown_or_invalid_data_without_writes(client, store, changes):
    body = capture(payload={"prompt": "a cat", **changes})
    response = client.post("/api/presets", content=json.dumps(body), headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert not presets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("changes", [
    {"model_id": OTHER}, {"model_revision": -1}, {"model_revision": 123},
    {"step_cache": "fbc-005-v1"}, {"step_cache": "invalid"}, {"execution": {"compilation": "regional-v1"}},
])
def test_capture_refuses_wrong_checkpoint_revision_and_performance(client, store, changes):
    response = client.post("/api/presets", json=capture(**changes))
    assert response.status_code in (409, 422), response.text
    assert not presets.path_for(store.settings.db_path).exists()


def test_stale_header_refuses_before_schema_validation(client, store):
    response = client.post("/api/presets", json={}, headers={"X-AuraDream-Model-Revision": "999"})
    assert response.status_code == 409, response.text
    assert not presets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("selection", [
    (None, 1, None, None, ExecutionSpec(compilation="regional-v1"), None),
    (None, 1, None, None, ExecutionSpec(attention_backend="cudnn-v1"), None),
    (None, 1, None, None, None, "invalid persisted performance"),
])
def test_held_loader_does_not_change_saved_setups(client, monkeypatch, store, selection):
    item = create(client)
    path = presets.path_for(store.settings.db_path)
    original = path.read_bytes()
    monkeypatch.setattr(model_selection, "_selection", selection)
    assert client.post("/api/presets", json=capture()).status_code == 422
    assert check(client, item).status_code == 422
    assert client.put(f"/api/presets/{item['id']}", json=capture(revision=1)).status_code == 422
    assert path.read_bytes() == original
    assert model_selection._selection == selection


def test_foreign_setup_is_readable_but_cannot_be_relabelled_or_validated(client, monkeypatch):
    item = create(client)
    monkeypatch.setattr(model_selection, "_selection", (OTHER, 1))
    assert client.get(f"/api/presets/{item['id']}").json() == item
    response = check(client, item)
    assert response.status_code == 409 and item["model_id"] in response.json()["detail"]
    response = client.put(f"/api/presets/{item['id']}", json=capture(revision=1))
    assert response.status_code == 409 and "keeps its checkpoint" in response.json()["detail"]


def test_broken_file_and_cross_site_writes_preserve_file(client, store):
    path = presets.path_for(store.settings.db_path)
    path.write_text("broken", encoding="utf-8")
    assert client.get("/api/presets").status_code == 503
    assert client.post("/api/presets", json=capture()).status_code == 503
    assert client.post("/api/presets", json=capture(), headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert path.read_text(encoding="utf-8") == "broken"


def test_cold_routes_bypass_gallery_gate_and_never_describe_load_or_submit(client, fake, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Saved setups must not perform gallery, model, tokenizer or job work")

    monkeypatch.setitem(main.bundle_manager.state, "state", "running")
    for obj, names in (
        (fake, ("describe", "load", "generate")),
        (main.jobs, ("submit", "run_when_idle", "wait_for_gate")),
        (main, ("ensure_loaded",)),
        (main.prompt_encoding, ("measure",)),
        (main.hub_access, ("check",)),
    ):
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    original_read = presets.read

    def independent_read(database):
        assert main.bundle_manager.readers == 0 and main.jobs._gate.locked() is False
        return original_read(database)

    monkeypatch.setattr(presets, "read", independent_read)
    item = create(client)
    assert check(client, item).status_code == 200
    assert client.get("/api/presets").status_code == 200
    assert client.get("/api/presets-other").status_code == 409
    assert client.get(f"/api/presets/{item['id']}/unrelated").status_code == 409
    assert not fake.requests


def test_switch_holds_capture_but_inspection_and_delete_remain_available(client, monkeypatch):
    item = create(client)
    monkeypatch.setattr(main, "_switching", True)
    assert client.post("/api/presets", json=capture()).status_code == 409
    assert check(client, item).status_code == 409
    assert client.put(f"/api/presets/{item['id']}", json=capture(revision=1)).status_code == 409
    headers = {"X-AuraDream-Model-Revision": "malformed"}
    assert client.get("/api/presets", headers=headers).status_code == 200
    assert client.get(f"/api/presets/{item['id']}", headers=headers).status_code == 200
    assert client.delete(f"/api/presets/{item['id']}?revision=1", headers=headers).status_code == 200


def test_item_change_during_validation_is_refused(client, store, monkeypatch):
    item = create(client)
    original = api_presets.validate_payload

    def edit_during_validation(*args, **kwargs):
        answer = original(*args, **kwargs)
        presets.change(store.settings.db_path, PresetUpdate(
            name="new", model_id=item["model_id"], payload=item["payload"], revision=1,
        ), item_id=item["id"])
        return answer

    monkeypatch.setattr(api_presets, "validate_payload", edit_during_validation)
    response = check(client, item)
    assert response.status_code == 409 and "during validation" in response.json()["detail"]


def test_model_switch_waits_for_capture_to_finish(client, monkeypatch):
    entered, release = Event(), Event()
    original = api_presets.validate_payload

    def waiting_validation(*args, **kwargs):
        assert main._model_requests == 1
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(api_presets, "validate_payload", waiting_validation)
    # A zero drain budget makes the refusal deterministic while validation
    # remains explicitly held, without racing an artificial short sleep.
    monkeypatch.setattr(main, "SWITCH_DRAIN_SECONDS", 0)
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(client.post, "/api/presets", json=capture())
            assert entered.wait(5)
            response = client.post("/api/model/switch", json={"model_id": OTHER, "load": False})
            assert response.status_code == 409, response.text
            assert "/api/presets" in response.json()["detail"]
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 201
    assert main._model_requests == 0
    assert client.get("/api/presets").json()["items"][0]["model_id"] != OTHER
