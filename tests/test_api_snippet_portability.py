"""Snippet portability previews are cold; only a pinned atomic commit saves."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from threading import Barrier

import pytest

from backend import gallery_lock, main, snippet_portability as portable, snippets
from backend.schemas_snippet_portability import SnippetImportCommit
from backend.schemas_snippets import MAX_SNIPPETS, SnippetFields, SnippetUpdate


FORMAT = "auradream.prompt-snippets"
MAX_REQUEST_BYTES = 8 * 1024 * 1024 + 64 * 1024


def fields(name="Style", text="soft light", target="prompt"):
    return {"name": name, "text": text, "target": target}


def payload(*items):
    return {"format": FORMAT, "version": 1, "items": list(items)}


def public(item):
    return {key: item[key] for key in ("name", "text", "target")}


def preview(client, document, decisions=None):
    body = {"payload": document}
    if decisions is not None:
        body["decisions"] = decisions
    response = client.post("/api/snippets/import/preview", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def chosen_rows(shown):
    result = []
    for row in shown["rows"]:
        decision = {"index": row["index"], "action": row["action"]}
        if row["action"] == "copy":
            decision["name"] = row["snippet"]["name"]
        if row["action"] == "replace":
            decision.update(item_id=row["item_id"], revision=row["revision"])
        result.append(decision)
    return result


def commit_body(document, shown, decisions=None):
    return {
        "payload": document,
        "payload_digest": shown["payload_digest"],
        "collection_revision": shown["collection_revision"],
        "decisions": chosen_rows(shown) if decisions is None else decisions,
    }


def storage_body(database, document, decisions):
    parsed = portable.validate_payload(document)
    return SnippetImportCommit.model_validate({
        "payload": document, "decisions": decisions,
        "payload_digest": portable.payload_digest(parsed),
        "collection_revision": portable.collection_revision(snippets.read(database)),
    })


def test_export_is_one_saved_snapshot_without_local_identity_or_a_write(client, store, monkeypatch):
    database = store.settings.db_path
    saved = snippets.change(database, SnippetFields(**fields(text="  exact\n日本語 🎨  ")))[0]
    snippets.change(database, SnippetFields(**fields("Negative", "blurry\n lowres", "negative")))
    before = snippets.path_for(database).read_bytes()

    def no_write(*args, **kwargs):
        pytest.fail("export wrote the snippet file")

    monkeypatch.setattr(snippets, "_write", no_write)
    exported = client.get("/api/snippets/export")
    assert exported.status_code == 200, exported.text
    assert "auradream-snippets.json" in exported.headers["content-disposition"]
    assert "attachment" in exported.headers["content-disposition"]
    document = json.loads(exported.content.decode("utf-8"))
    assert set(document) == {"format", "version", "items"}
    assert document["format"] == FORMAT and document["version"] == 1
    assert document["items"] == [
        fields(text=saved.text), fields("Negative", "blurry\n lowres", "negative"),
    ]
    assert snippets.path_for(database).read_bytes() == before


def test_export_keeps_its_complete_snapshot_when_crud_commits_during_the_read(client, store, monkeypatch):
    database = store.settings.db_path
    original = snippets.change(database, SnippetFields(**fields()))[0]
    read = snippets.read
    calls = []

    def capture_then_edit(selected):
        result = read(selected)
        calls.append(selected)
        snippets.change(selected, SnippetUpdate(**fields(text="concurrent update"), revision=1), item_id=original.id)
        return result

    monkeypatch.setattr(snippets, "read", capture_then_edit)
    response = client.get("/api/snippets/export")
    assert response.status_code == 200, response.text
    assert calls == [database]
    assert response.json()["items"] == [fields()]
    assert read(database)[0].text == "concurrent update"


def test_export_and_preview_of_missing_collection_create_no_sidecar(client, store):
    path = snippets.path_for(store.settings.db_path)
    assert not path.exists()
    assert client.get("/api/snippets/export").json() == payload()
    shown = preview(client, payload(fields()))
    assert shown["can_import"] and shown["changed"] and shown["add_count"] == 1
    assert not path.exists()


def test_preview_shows_duplicate_candidates_exact_ids_and_file_index_collisions(client, store):
    database = store.settings.db_path
    first = snippets.change(database, SnippetFields(**fields()))[0]
    second = snippets.change(database, SnippetFields(**fields(text="local variant")))[1]
    path = snippets.path_for(database)
    before = path.read_bytes()
    shown = preview(client, payload(fields(), fields(text="incoming"), fields("New"), fields("NEW", "copy")))
    assert [row["action"] for row in shown["rows"]] == ["skip", "skip", "add", "add"]
    assert shown["rows"][0]["exact_ids"] == [first.id]
    assert [row["id"] for row in shown["rows"][1]["matches"]] == [first.id, second.id]
    assert shown["rows"][2]["incoming_collisions"] == [3]
    assert shown["rows"][3]["incoming_collisions"] == [2]
    assert (shown["add_count"], shown["replace_count"], shown["skip_count"]) == (2, 0, 2)
    assert path.read_bytes() == before


def test_commit_replaces_only_the_explicit_duplicate_and_appends_copies_in_file_order(client, store):
    database = store.settings.db_path
    first = snippets.change(database, SnippetFields(**fields()))[0]
    second = snippets.change(database, SnippetFields(**fields(text="local variant")))[1]
    document = payload(fields(text="new variant"), fields(), fields("Other", "negative text", "negative"))
    decisions = [
        {"index": 1, "action": "copy", "name": "Copy"},
        {"index": 0, "action": "replace", "item_id": second.id, "revision": second.revision},
        {"index": 2, "action": "add"},
    ]
    shown = preview(client, document, decisions)
    committed = client.post("/api/snippets/import", json=commit_body(document, shown, decisions))
    assert committed.status_code == 200, committed.text
    saved = committed.json()["items"]
    assert [item["name"] for item in saved] == ["Style", "Style", "Copy", "Other"]
    assert saved[0] == first.model_dump()
    assert saved[1] == {**second.model_dump(), "text": "new variant", "revision": 2}
    assert [item["revision"] for item in saved[2:]] == [1, 1]
    assert len({item["id"] for item in saved}) == 4
    assert {item["id"] for item in saved[2:]}.isdisjoint({first.id, second.id})
    assert [item.model_dump() for item in snippets.read(database)] == saved


def test_commit_uses_one_existing_atomic_writer_and_does_not_partially_publish(client, store, monkeypatch):
    document = payload(fields("First"), fields("Second"))
    shown = preview(client, document)
    calls = []
    write = snippets._write

    def counted_write(path, items):
        calls.append([item.name for item in items])
        assert not path.exists()
        return write(path, items)

    monkeypatch.setattr(snippets, "_write", counted_write)
    result = client.post("/api/snippets/import", json=commit_body(document, shown))
    assert result.status_code == 200, result.text
    assert calls == [["First", "Second"]]


def test_unrelated_crud_change_invalidates_whole_preview_without_saving_any_incoming_row(client, store):
    database = store.settings.db_path
    snippets.change(database, SnippetFields(**fields("Existing")))
    document = payload(fields("Incoming"))
    shown = preview(client, document)
    snippets.change(database, SnippetFields(**fields("Other tab")))
    before = snippets.path_for(database).read_bytes()
    result = client.post("/api/snippets/import", json=commit_body(document, shown))
    assert result.status_code == 409, result.text
    assert snippets.path_for(database).read_bytes() == before


def test_replaying_a_successful_commit_is_stale_and_never_adds_copies_twice(client, store):
    database = store.settings.db_path
    snippets.change(database, SnippetFields(**fields()))
    document = payload(fields())
    shown = preview(client, document, [{"index": 0, "action": "copy"}])
    body = commit_body(document, shown)
    assert client.post("/api/snippets/import", json=body).status_code == 200
    before = snippets.path_for(database).read_bytes()
    assert client.post("/api/snippets/import", json=body).status_code == 409
    assert snippets.path_for(database).read_bytes() == before
    assert len(snippets.read(database)) == 2


@pytest.mark.parametrize("change", ["digest", "payload", "missing_choice", "duplicate_choice", "extra_choice"])
def test_tampered_or_incomplete_commit_is_refused_without_a_sidecar(client, store, change):
    document = payload(fields())
    shown = preview(client, document)
    body = commit_body(document, shown)
    if change == "digest":
        body["payload_digest"] = "0" * 64
    elif change == "payload":
        body["payload"] = payload(fields(text="different"))
    elif change == "missing_choice":
        body["decisions"] = []
    elif change == "duplicate_choice":
        body["decisions"] *= 2
    else:
        body["decisions"].append({"index": 1, "action": "skip"})
    result = client.post("/api/snippets/import", json=body)
    assert result.status_code == 422, result.text
    assert not snippets.path_for(store.settings.db_path).exists()


def test_stale_target_revision_is_refused_even_with_a_fresh_collection_pin(client, store):
    database = store.settings.db_path
    original = snippets.change(database, SnippetFields(**fields()))[0]
    snippets.change(database, SnippetUpdate(**fields(text="updated"), revision=1), item_id=original.id)
    document = payload(fields(text="incoming"))
    fresh = preview(client, document)
    stale = [{"index": 0, "action": "replace", "item_id": original.id, "revision": 1}]
    before = snippets.path_for(database).read_bytes()
    result = client.post("/api/snippets/import", json=commit_body(document, fresh, stale))
    assert result.status_code == 409, result.text
    assert snippets.path_for(database).read_bytes() == before


def test_conflicting_replacement_choices_are_recomputed_before_any_write(client, store):
    database = store.settings.db_path
    local = snippets.change(database, SnippetFields(**fields()))[0]
    document = payload(fields(text="first incoming"), fields(text="second incoming"))
    shown = preview(client, document)
    choices = [
        {"index": index, "action": "replace", "item_id": local.id, "revision": local.revision}
        for index in range(2)
    ]
    before = snippets.path_for(database).read_bytes()
    response = client.post("/api/snippets/import", json=commit_body(document, shown, choices))
    assert response.status_code == 422, response.text
    assert snippets.path_for(database).read_bytes() == before


@pytest.mark.parametrize("document", [payload(), payload(fields())])
def test_empty_and_exact_match_only_previews_are_held_and_commit_does_not_write(client, store, document, monkeypatch):
    database = store.settings.db_path
    snippets.change(database, SnippetFields(**fields()))
    before = snippets.path_for(database).read_bytes()
    shown = preview(client, document)
    assert not shown["changed"] and not shown["can_import"] and shown["reason"]

    def forbidden(*args, **kwargs):
        pytest.fail("a no-op import attempted a write")

    monkeypatch.setattr(snippets, "_write", forbidden)
    result = client.post("/api/snippets/import", json=commit_body(document, shown))
    assert result.status_code == 422, result.text
    assert snippets.path_for(database).read_bytes() == before


@pytest.mark.parametrize("raw", [
    b'{"payload":{"format":"auradream.prompt-snippets","version":1,"items":[]},"payload":{}}',
    b'{"payload":{"format":"auradream.prompt-snippets","version":1,"items":[{"name":"a","name":"b","text":"x","target":"prompt"}]}}',
    b'{"payload":{"format":"auradream.prompt-snippets","version":1,"items":[],"private_personal_text":true}}',
    b'{"payload":{"format":"auradream.prompt-snippets","version":1,"items":[{"name":"a","text":"\\ud800","target":"prompt"}]}}',
    b'{"payload":{"format":"auradream.prompt-snippets","version":true,"items":[]}}',
    b'{"payload":{"format":"auradream.prompt-snippets","version":NaN,"items":[]}}',
    b"[" * 17 + b"0" + b"]" * 17,
    b'\xff',
    json.dumps({"payload": payload()}).encode("utf-16"),
])
@pytest.mark.parametrize("route", ["/api/snippets/import/preview", "/api/snippets/import"])
def test_raw_parser_refusals_are_fixed_and_never_echo_personal_input(client, store, raw, route):
    response = client.post(route, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert "private_personal_text" not in response.text
    assert not snippets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("route", ["/api/snippets/import/preview", "/api/snippets/import"])
def test_request_byte_limit_is_applied_before_json_parsing(client, store, route):
    raw = b" " * (MAX_REQUEST_BYTES + 1)
    response = client.post(route, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.text
    assert response.json()["detail"] == "The snippet import request is too large."
    assert len(response.content) < 1024
    assert not snippets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("raw", [b"broken", b'{"version":2,"items":[]}', b'{"version":true,"items":[]}'])
def test_unreadable_destination_blocks_export_preview_and_commit_without_replacing_file(client, store, raw):
    database = store.settings.db_path
    document = payload(fields())
    body = storage_body(database, document, [{"index": 0, "action": "add"}]).model_dump(mode="json")
    path = snippets.path_for(database)
    path.write_bytes(raw)
    assert client.get("/api/snippets/export").status_code == 503
    assert client.post("/api/snippets/import/preview", json={"payload": document}).status_code == 503
    assert client.post("/api/snippets/import", json=body).status_code == 503
    assert path.read_bytes() == raw


def test_round_trip_between_scratch_databases_assigns_fresh_ids_and_repeats_as_exact_skips(tmp_path):
    source = tmp_path / "source" / "gallery.db"
    destination = tmp_path / "destination" / "gallery.db"
    source.parent.mkdir()
    destination.parent.mkdir()
    original = snippets.change(source, SnippetFields(**fields(text="  exact\n🎨  ")))
    original = snippets.change(source, SnippetFields(**fields("Negative", "blurry", "negative")))
    raw = portable.export_portable(original)
    document = json.loads(raw)
    body = storage_body(destination, document, [
        {"index": 0, "action": "add"}, {"index": 1, "action": "add"},
    ])
    saved = snippets.import_collection(destination, body)
    assert [public(item.model_dump()) for item in saved] == document["items"]
    assert {item.id for item in saved}.isdisjoint({item.id for item in original})
    assert [item.revision for item in saved] == [1, 1]
    assert snippets.read(source) == original and snippets.read(destination) == saved
    repeated = portable.plan_merge(saved, portable.parse_portable(raw))
    assert not repeated.changed and repeated.skip_count == 2
    assert not source.exists() and not destination.exists()


def test_two_simultaneous_imports_from_one_snapshot_allow_only_one_atomic_commit(tmp_path):
    database = tmp_path / "gallery.db"
    body = storage_body(database, payload(fields("Incoming")), [{"index": 0, "action": "add"}])
    barrier = Barrier(2)

    def import_once():
        barrier.wait(timeout=5)
        try:
            return snippets.import_collection(database, body)
        except (snippets.SnippetsError, portable.PortabilityError) as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: import_once(), range(2)))
    successes = [result for result in results if isinstance(result, list)]
    assert len(successes) == 1 and results.count(409) == 1
    assert snippets.read(database) == successes[0] and len(successes[0]) == 1


@pytest.mark.parametrize("operation", ["write", "fsync", "replace"])
def test_import_write_failures_preserve_every_old_row_and_remove_temporary_files(tmp_path, monkeypatch, operation):
    database = tmp_path / "gallery.db"
    snippets.change(database, SnippetFields(**fields("Kept")))
    body = storage_body(database, payload(fields("Incoming 1"), fields("Incoming 2")), [
        {"index": 0, "action": "add"}, {"index": 1, "action": "add"},
    ])
    path = snippets.path_for(database)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise PermissionError("synthetic storage failure")

    if operation == "write":
        temporary_file = snippets.tempfile.NamedTemporaryFile

        class BrokenWriter:
            def __init__(self, *args, **kwargs):
                self.file = temporary_file(*args, **kwargs)

            def __enter__(self):
                self.file.__enter__()
                return self

            def __exit__(self, *args):
                return self.file.__exit__(*args)

            @property
            def name(self):
                return self.file.name

            write = staticmethod(fail)

        monkeypatch.setattr(snippets.tempfile, "NamedTemporaryFile", BrokenWriter)
    else:
        monkeypatch.setattr(snippets.os, operation, fail)
    with pytest.raises((snippets.SnippetsError, portable.PortabilityError)) as error:
        snippets.import_collection(database, body)
    assert error.value.status == 503
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".auradream.snippets.*.tmp")) == []


def test_snippet_lock_contention_refuses_without_changing_the_destination(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    snippets.change(database, SnippetFields(**fields("Kept")))
    body = storage_body(database, payload(fields("Incoming")), [{"index": 0, "action": "add"}])
    path = snippets.path_for(database)
    before = path.read_bytes()

    @contextmanager
    def busy(selected, **kwargs):
        assert selected == path
        raise gallery_lock.GalleryBusy("synthetic external holder")
        yield

    monkeypatch.setattr(gallery_lock, "hold", busy)
    with pytest.raises((snippets.SnippetsError, portable.PortabilityError)) as error:
        snippets.import_collection(database, body)
    assert error.value.status == 409
    assert path.read_bytes() == before


def test_commit_rechecks_final_count_under_lock_without_partial_additions(tmp_path):
    database = tmp_path / "gallery.db"
    first = snippets.change(database, SnippetFields(**fields("Local 0")))[0]
    full = [first.model_copy(update={"id": f"{index:032x}", "name": f"Local {index}"}) for index in range(MAX_SNIPPETS)]
    snippets._write(snippets.path_for(database), full)
    body = storage_body(database, payload(fields("Too many")), [{"index": 0, "action": "add"}])
    before = snippets.path_for(database).read_bytes()
    with pytest.raises((snippets.SnippetsError, portable.PortabilityError)) as error:
        snippets.import_collection(database, body)
    assert error.value.status == 422
    assert snippets.path_for(database).read_bytes() == before


def test_commit_rechecks_combined_encoded_bytes_without_partial_additions(tmp_path):
    database = tmp_path / "gallery.db"
    escaped = "\x00" * 15999 + "x"
    first = snippets.change(database, SnippetFields(**fields("Local 0", escaped)))[0]
    local = [first.model_copy(update={"id": f"{index:032x}", "name": f"Local {index}"}) for index in range(50)]
    snippets._write(snippets.path_for(database), local)
    incoming = payload(*(fields(f"Incoming {index}", escaped) for index in range(50)))
    body = storage_body(database, incoming, [{"index": index, "action": "add"} for index in range(50)])
    before = snippets.path_for(database).read_bytes()
    with pytest.raises((snippets.SnippetsError, portable.PortabilityError)) as error:
        snippets.import_collection(database, body)
    assert error.value.status == 422
    assert snippets.path_for(database).read_bytes() == before


@pytest.mark.parametrize("maintenance", [False, True])
def test_portability_routes_bypass_gallery_model_and_vault_work(client, store, fake, monkeypatch, maintenance):
    manager = main.bundle_manager
    if maintenance:
        monkeypatch.setitem(manager.state, "state", "running")
    monkeypatch.setattr(main, "_switching", True)

    def forbidden(*args, **kwargs):
        pytest.fail("snippet portability reached model/gallery work")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(fake, "load", forbidden)
    monkeypatch.setattr(main.jobs, "run_when_idle", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    monkeypatch.setattr(main.jobs, "submit_cells", forbidden)
    read = snippets.read
    hold = gallery_lock.hold
    snippet_path = snippets.path_for(store.settings.db_path)

    def outside_gallery(selected):
        assert manager.readers == 0
        return read(selected)

    @contextmanager
    def only_snippet_lock(selected, **kwargs):
        assert selected == snippet_path
        assert manager.readers == 0
        with hold(selected, **kwargs):
            yield

    monkeypatch.setattr(snippets, "read", outside_gallery)
    monkeypatch.setattr(gallery_lock, "hold", only_snippet_lock)
    document = payload(fields("Portable"))
    assert client.get("/api/snippets/export").status_code == 200
    shown = preview(client, document)
    result = client.post("/api/snippets/import", json=commit_body(document, shown))
    assert result.status_code == 200, result.text
    if maintenance:
        assert client.get("/api/snippets/import/unrelated").status_code == 409
        assert client.get("/api/snippets/export/unrelated").status_code == 409


@pytest.mark.parametrize("route", ["/api/snippets/import/preview", "/api/snippets/import"])
def test_import_routes_refuse_foreign_origin_before_writing(client, store, route):
    result = client.post(route, json={"payload": payload(fields())}, headers={"Origin": "https://elsewhere.example"})
    assert result.status_code == 403
    assert not snippets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("declared_length", [None, "1"])
def test_stream_limit_ignores_missing_or_incorrect_content_length(client, store, monkeypatch, declared_length):
    from backend import api_snippets

    monkeypatch.setattr(api_snippets, "MAX_IMPORT_REQUEST_BYTES", 128)
    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    response = client.post(
        "/api/snippets/import/preview", content=iter([b" " * 80, b" " * 80]), headers=headers,
    )
    assert response.status_code == 413
    assert not snippets.path_for(store.settings.db_path).exists()


def test_payload_has_its_own_byte_bound_inside_a_bounded_request(client, store, monkeypatch):
    monkeypatch.setattr(portable, "MAX_PORTABLE_BYTES", 256)
    response = client.post("/api/snippets/import/preview", json={
        "payload": payload(fields(text="x" * 500)),
    })
    assert response.status_code == 422
    assert not snippets.path_for(store.settings.db_path).exists()


def test_duplicate_keys_in_local_sidecar_refuse_export_and_import_without_rewriting(client, store):
    database = store.settings.db_path
    document = payload(fields())
    body = storage_body(database, document, [{"index": 0, "action": "add"}]).model_dump(mode="json")
    path = snippets.path_for(database)
    data = b'{"version":1,"items":[],"items":[]}'
    path.write_bytes(data)
    assert client.get("/api/snippets/export").status_code == 503
    assert client.post("/api/snippets/import/preview", json={"payload": document}).status_code == 503
    assert client.post("/api/snippets/import", json=body).status_code == 503
    assert path.read_bytes() == data
