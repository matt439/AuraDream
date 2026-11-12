"""Model favourites' routes: model-free, revisioned, outside gallery maintenance."""
from __future__ import annotations

import json

import pytest

from backend import main, model_favourites, passcode

ROOT = "/api/models/favourites"
KREA = "black-forest-labs/FLUX.1-Krea-dev"
KREA_LITE = "lite-infer/flux.1-krea-dev-nunchaku-lite-int4_r32-bnb4-text-encoder"


def test_star_unstar_and_import_answer_the_whole_revisioned_set(client, store):
    path = model_favourites.path_for(store.settings.db_path)
    assert client.get(ROOT).json() == {"items": [], "revision": 0, "max_items": 200, "imported": None}
    assert not path.exists()
    starred = client.post(ROOT, json={"model_id": KREA_LITE, "favourite": True})
    assert starred.status_code == 200, starred.text
    assert starred.json()["items"] == [KREA] and starred.json()["revision"] == 1
    imported = client.post(ROOT + "/import", json={"model_ids": [KREA, "someone/custom"]}).json()
    assert imported == {"items": [KREA, "someone/custom"], "revision": 2, "max_items": 200, "imported": 1}
    unstarred = client.post(ROOT, json={"model_id": KREA, "favourite": False}).json()
    assert unstarred["items"] == ["someone/custom"] and unstarred["revision"] == 3
    assert client.get(ROOT).json()["revision"] == 3


@pytest.mark.parametrize("body", [
    {"model_id": "", "favourite": True}, {"model_id": "private\nsecret", "favourite": True},
    {"model_id": "a", "favourite": "yes"}, {"model_id": "a"}, {"private-field": "private-value"},
])
def test_invalid_fields_have_a_fixed_redacted_answer(client, store, body):
    response = client.post(ROOT, content=json.dumps(body), headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "Model favourite fields are invalid. Check the model ids and the star."
    assert "private" not in response.text
    assert client.post(ROOT + "/import", json={"model_ids": ["ok", "private\nsecret"]}).status_code == 422
    assert not model_favourites.path_for(store.settings.db_path).exists()


def test_a_corrupt_file_is_kept_and_a_foreign_origin_refused(client, store):
    path = model_favourites.path_for(store.settings.db_path)
    path.write_bytes(b"private corrupted favourites")
    assert client.get(ROOT).status_code == 503
    result = client.post(ROOT, json={"model_id": KREA, "favourite": True})
    assert result.status_code == 503 and "private corrupted" not in result.text
    path.unlink()
    assert client.post(ROOT, json={"model_id": KREA, "favourite": True},
                       headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert not path.exists()


@pytest.mark.parametrize("maintenance", [False, True])
def test_routes_bypass_gallery_maintenance_model_load_and_gpu_gate(client, monkeypatch, maintenance):
    manager = main.bundle_manager
    if maintenance:
        monkeypatch.setitem(manager.state, "state", "running")
    monkeypatch.setattr(main, "_switching", True)

    async def forbidden(*args, **kwargs):
        pytest.fail("model favourites touched the model load or GPU gate")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    read = model_favourites.read

    def independent(database):
        assert manager.readers == 0
        return read(database)

    monkeypatch.setattr(model_favourites, "read", independent)
    assert client.get(ROOT).json()["items"] == []
    assert client.post(ROOT, json={"model_id": KREA, "favourite": True}).status_code == 200
    assert client.post(ROOT + "/import", json={"model_ids": ["a/b"]}).status_code == 200
    if maintenance:
        assert client.get("/api/gallery").status_code == 409
        assert client.get(ROOT + "-other").status_code == 409


def test_behind_the_passcode_the_routes_need_a_session(client, monkeypatch):
    gate = passcode.Gate("favourites-pass-1", port=main.settings.port)
    monkeypatch.setattr(main, "_passcode_gate", gate)
    refused = client.get(ROOT)
    assert refused.status_code == 401 and refused.json()["code"] == "passcode_required"
    assert client.post(ROOT, json={"model_id": KREA, "favourite": True}).status_code == 401
    signed = client.get(ROOT, headers={"Authorization": "Bearer favourites-pass-1"})
    assert signed.status_code == 200, signed.text
