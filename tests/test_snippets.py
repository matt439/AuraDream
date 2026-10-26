"""Personal snippets are bounded, atomic and independent of generation."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from backend import main, snippets
from backend.schemas_snippets import MAX_SNIPPETS, SnippetFields, SnippetUpdate


def test_missing_collection_is_read_without_creating_files(tmp_path):
    database = tmp_path / "not-created" / "gallery.db"
    assert snippets.read(database) == []
    assert not database.parent.exists()


def test_saved_text_and_destination_survive_another_read(tmp_path):
    database = tmp_path / "gallery.db"
    fields = SnippetFields(name="  My negative  ", text="  blurry, lowres\n色ずれ  ", target="negative")
    first = snippets.change(database, fields)[0]
    assert (first.name, first.text, first.target, first.revision) == (
        "My negative", fields.text, "negative", 1,
    )
    assert snippets.read(database) == [first]
    assert not database.exists()
    assert not (tmp_path / "local-diffusion.settings.json").exists()
    updated = snippets.change(database, SnippetUpdate(
        name="New name", text="a prefix", target="prompt", revision=1,
    ), item_id=first.id)[0]
    assert updated.id == first.id and updated.revision == 2
    with pytest.raises(snippets.SnippetsError, match="changed in another tab") as error:
        snippets.change(database, fields.model_copy(), item_id=first.id, revision=1)
    assert error.value.status == 409
    with pytest.raises(snippets.SnippetsError, match="changed in another tab"):
        snippets.change(database, None, item_id=first.id, revision=1)
    assert snippets.read(database) == [updated]
    assert snippets.change(database, None, item_id=first.id, revision=2) == []


def test_atomic_replace_failure_keeps_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    snippets.change(database, SnippetFields(name="kept", text="original"))
    path = snippets.path_for(database)
    before = path.read_bytes()

    def fail_replace(*args):
        raise PermissionError("locked file")

    monkeypatch.setattr(snippets.os, "replace", fail_replace)
    with pytest.raises(snippets.SnippetsError, match="previous saved list"):
        snippets.change(database, SnippetFields(name="not saved", text="new"))
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".local-diffusion.snippets.*.tmp")) == []


@pytest.mark.parametrize("data", [
    b"not JSON", b'{"version":2,"items":[]}', b'{"version":true,"items":[]}',
    b'{"version":1,"items":[],"unknown":true}',
    b'{"version":1,"items":[{"id":"bad","name":"a","text":"b","target":"prompt","revision":1}]}',
    b'{"version":1,"items":[{"id":"00000000000000000000000000000000","name":"a","text":"b","revision":1}]}',
    b"[" * 2000 + b"]" * 2000,
])
def test_invalid_saved_file_is_never_overwritten(tmp_path, data):
    database = tmp_path / "gallery.db"
    path = snippets.path_for(database)
    path.write_bytes(data)
    with pytest.raises(snippets.SnippetsError, match="left unchanged"):
        snippets.read(database)
    with pytest.raises(snippets.SnippetsError, match="left unchanged"):
        snippets.change(database, SnippetFields(name="a", text="b"))
    assert path.read_bytes() == data


def test_duplicate_ids_and_oversized_file_are_refused(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    item = snippets.change(database, SnippetFields(name="a", text="b"))[0]
    path = snippets.path_for(database)
    path.write_text(json.dumps({"version": 1, "items": [item.model_dump(), item.model_dump()]}), encoding="utf-8")
    with pytest.raises(snippets.SnippetsError):
        snippets.read(database)
    monkeypatch.setattr(snippets, "MAX_FILE_BYTES", 1)
    with pytest.raises(snippets.SnippetsError):
        snippets.read(database)


def test_collection_limit_and_concurrent_creates(tmp_path):
    database = tmp_path / "gallery.db"

    def add(number):
        return snippets.change(database, SnippetFields(name=str(number), text=f"text {number}"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(add, range(4)))
    assert {item.name for item in snippets.read(database)} == {"0", "1", "2", "3"}
    original = snippets.read(database)[0]
    full = [original.model_copy(update={"id": f"{number:032x}"}) for number in range(MAX_SNIPPETS)]
    snippets._write(snippets.path_for(database), full)
    with pytest.raises(snippets.SnippetsError, match="at most") as error:
        add(100)
    assert error.value.status == 422


def test_concurrent_edits_only_accept_one_writer(tmp_path):
    database = tmp_path / "gallery.db"
    original = snippets.change(database, SnippetFields(name="a", text="original"))[0]
    start = Barrier(2)

    def update(text):
        start.wait(timeout=5)
        try:
            return snippets.change(database, SnippetUpdate(
                name="a", text=text, revision=original.revision,
            ), item_id=original.id)
        except snippets.SnippetsError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(update, ["first writer", "second writer"]))
    saved = [result for result in results if isinstance(result, list)]
    assert len(saved) == 1 and results.count(409) == 1
    assert snippets.read(database) == saved[0]
    assert saved[0][0].revision == original.revision + 1


def test_api_crud_is_shared_and_rejects_stale_requests(client, fake, store):
    path = snippets.path_for(store.settings.db_path)
    assert client.get("/api/snippets").json()["items"] == []
    assert not path.exists()
    body = {"name": "My style", "text": "watercolor\nsoft light", "target": "prompt"}
    created = client.post("/api/snippets", json=body)
    assert created.status_code == 201, created.text
    item = created.json()["items"][0]
    response = client.put(f"/api/snippets/{item['id']}", json={**body, "text": "new style", "revision": 1})
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["revision"] == 2
    assert client.put(f"/api/snippets/{item['id']}", json={**body, "revision": 1}).status_code == 409
    assert client.delete(f"/api/snippets/{item['id']}?revision=1").status_code == 409
    assert client.delete(f"/api/snippets/{item['id']}").status_code == 422
    assert client.get("/api/snippets").json()["items"][0]["text"] == "new style"
    assert client.delete(f"/api/snippets/{item['id']}?revision=2").json()["items"] == []
    assert client.delete(f"/api/snippets/{item['id']}?revision=2").status_code == 404


@pytest.mark.parametrize("body", [
    {"name": "", "text": "x"}, {"name": "a", "text": " "},
    {"name": "a" * 81, "text": "x"}, {"name": "a", "text": "x" * 16001},
    {"name": "a", "text": "x", "target": "path"}, {"name": "a", "text": "x", "id": "../x"},
    {"name": "a\nsecret", "text": "x"}, {"name": "a", "text": "\ud800"},
])
def test_api_refuses_invalid_fields_without_a_write(client, fake, store, body):
    response = client.post("/api/snippets", content=json.dumps(body), headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert not snippets.path_for(store.settings.db_path).exists()


def test_api_reports_broken_file_and_refuses_foreign_origin(client, fake, store):
    path = snippets.path_for(store.settings.db_path)
    path.write_text("broken", encoding="utf-8")
    response = client.get("/api/snippets")
    assert response.status_code == 503
    assert "left unchanged" in response.json()["detail"]
    body = {"name": "a", "text": "b"}
    assert client.post("/api/snippets", json=body).status_code == 503
    assert client.post("/api/snippets", json=body, headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert path.read_text(encoding="utf-8") == "broken"


@pytest.mark.parametrize("maintenance", [False, True])
def test_snippets_bypass_gallery_admission_and_model_switch(client, monkeypatch, maintenance):
    manager = main.bundle_manager
    if maintenance:
        monkeypatch.setitem(manager.state, "state", "running")
    monkeypatch.setattr(main, "_switching", True)
    original_read = snippets.read

    def outside_gallery_admission(database):
        assert manager.readers == 0
        return original_read(database)

    monkeypatch.setattr(snippets, "read", outside_gallery_admission)
    assert client.get("/api/snippets").json()["items"] == []
    body = {"name": "a", "text": "original"}
    created = client.post("/api/snippets", json=body)
    assert created.status_code == 201, created.text
    item = created.json()["items"][0]
    updated = client.put(f"/api/snippets/{item['id']}", json={**body, "text": "new", "revision": 1})
    assert updated.status_code == 200, updated.text
    deleted = client.delete(f"/api/snippets/{item['id']}?revision=2")
    assert deleted.status_code == 200 and deleted.json()["items"] == []
    if maintenance:
        assert client.get("/api/gallery").status_code == 409
        assert client.get("/api/snippets-other").status_code == 409
        assert client.get(f"/api/snippets/{item['id']}/unrelated").status_code == 409
