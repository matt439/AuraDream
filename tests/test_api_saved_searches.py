"""Model-free saved-search routes retain strict definitions and stale-write guards."""
from __future__ import annotations

import json

import pytest

from backend import main, saved_searches

ROOT = "/api/gallery/saved-searches"
BODY = {"name": "Keepers", "query": " watercolor date:today ", "favorites": True, "all_models": False}


def test_api_crud_requires_revisions_and_keeps_query_exact(client, store):
    path = saved_searches.path_for(store.settings.db_path)
    empty = client.get(ROOT).json()
    assert empty == {"items": [], "max_items": 50, "max_name": 80, "max_query": 4000}
    assert not path.exists()
    created = client.post(ROOT, json=BODY)
    assert created.status_code == 201, created.text
    item = created.json()["items"][0]
    assert item["query"] == BODY["query"]
    route = f"{ROOT}/{item['id']}"
    updated = client.put(route, json={**BODY, "query": "", "revision": 1})
    assert updated.status_code == 200, updated.text
    assert updated.json()["items"][0]["revision"] == 2
    assert client.put(route, json={**BODY, "revision": 1}).status_code == 409
    assert client.delete(route + "?revision=1").status_code == 409
    assert client.delete(route).status_code == 422
    assert client.get(ROOT).json()["items"][0]["query"] == ""
    assert client.delete(route + "?revision=2").json()["items"] == []
    assert client.delete(route + "?revision=2").status_code == 404


def test_duplicate_response_identifies_existing_item_without_echoing_query(client):
    created = client.post(ROOT, json={**BODY, "name": "Straße"}).json()["items"][0]
    response = client.post(ROOT, json={**BODY, "name": "STRASSE", "query": "private query"})
    assert response.status_code == 409
    assert response.json()["conflict_id"] == created["id"]
    assert "private query" not in response.text and "STRASSE" not in response.text
    # Explicitly updating the named entry uses its existing optimistic revision.
    assert client.put(f"{ROOT}/{created['id']}", json={**BODY, "name": "STRASSE", "revision": 1}).status_code == 200


@pytest.mark.parametrize("updates", [
    {"name": ""}, {"name": "x" * 81}, {"query": "x" * 4001}, {"query": "steps:private-secret"},
    {"name": "a\x7f"}, {"query": "\ud800"}, {"favorites": "true"}, {"all_models": 1},
    {"private-field": "private-value"}, {"revision": 1},
])
def test_invalid_api_fields_have_fixed_redacted_response(client, store, updates):
    response = client.post(ROOT, content=json.dumps({**BODY, **updates}), headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "Saved search fields are invalid. Check the name, query, ticks and revision."
    assert not saved_searches.path_for(store.settings.db_path).exists()


def test_required_ticks_and_bad_json_or_revision_are_refused(client):
    assert client.post(ROOT, json={"name": "a", "query": ""}).status_code == 422
    assert client.post(ROOT, content=b"{broken", headers={"Content-Type": "application/json"}).status_code == 422
    item = client.post(ROOT, json=BODY).json()["items"][0]
    route = f"{ROOT}/{item['id']}"
    assert client.put(route, json={**BODY, "revision": True}).status_code == 422
    assert client.delete(route + "?revision=0").status_code == 422


def test_corrupt_file_is_preserved_and_foreign_origin_is_refused(client, store):
    path = saved_searches.path_for(store.settings.db_path)
    path.write_bytes(b"private corrupted query")
    assert client.get(ROOT).status_code == 503
    result = client.post(ROOT, json=BODY)
    assert result.status_code == 503 and "private corrupted query" not in result.text
    assert client.post(ROOT, json=BODY, headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert path.read_bytes() == b"private corrupted query"


@pytest.mark.parametrize("maintenance", [False, True])
def test_routes_bypass_gallery_maintenance_model_load_and_gpu_gate(client, monkeypatch, maintenance):
    manager = main.bundle_manager
    if maintenance:
        monkeypatch.setitem(manager.state, "state", "running")
    monkeypatch.setattr(main, "_switching", True)

    async def forbidden(*args, **kwargs):
        pytest.fail("saved search touched the model load or GPU gate")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    read = saved_searches.read

    def independent(database):
        assert manager.readers == 0
        return read(database)

    monkeypatch.setattr(saved_searches, "read", independent)
    assert client.get(ROOT).json()["items"] == []
    item = client.post(ROOT, json=BODY).json()["items"][0]
    route = f"{ROOT}/{item['id']}"
    assert client.put(route, json={**BODY, "revision": 1}).status_code == 200
    assert client.delete(route + "?revision=2").status_code == 200
    if maintenance:
        assert client.get("/api/gallery").status_code == 409
        assert client.get(ROOT + "-other").status_code == 409
        assert client.get(route + "/unrelated").status_code == 409
