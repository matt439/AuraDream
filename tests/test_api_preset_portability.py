"""Setup portability previews are cold; only a pinned atomic commit saves raw drafts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from threading import Barrier

import pytest

from backend import api_presets, gallery_lock, main, model_selection, preset_portability as portable, presets
from backend.schemas_preset_portability import PresetImportCommit
from backend.schemas_presets import MAX_PRESETS, Preset, PresetFields, PresetUpdate


FORMAT = "auradream.generation-presets"
FOREIGN = "example/Exact-Checkpoint"
OTHER = "cagliostrolab/animagine-xl-4.0"
MAX_REQUEST_BYTES = 8 * 1024 * 1024 + 64 * 1024


def fields(name="Study", model_id=FOREIGN, **changes):
    raw = {
        "mode": "generate", "prompt": "  first line\n日本語 🎨  ",
        "negative_prompt": "  blurry\r\n lowres  ",
        "steps": None, "guidance_scale": 0, "mu": None, "true_cfg": None,
        "seed": None, "num_images": 2, "runs": 3, **changes,
    }
    return {"name": name, "model_id": model_id, "payload": raw}


def grid_fields(name="Grid", model_id=FOREIGN, **changes):
    return fields(name, model_id, mode="grid", grid={
        "prompt_axis": "  light = night | day\ncolour = 青 | red  ",
        "prompt_mode": "random", "steps_axis": "28, 12", "cfg_axis": "3,0",
        "mu_axis": "", "size_axis": "1024x768,768x1024",
        "seeds": 5, "seed_axis": "44,11,22", "budget_minutes": 45,
    }, **changes)


def complete(item):
    return PresetFields.model_validate(item).model_dump()


def payload(*items):
    return {"format": FORMAT, "version": 1, "items": list(items)}


def save(database, item):
    return presets.change(database, PresetFields.model_validate(item))


def preview(client, document, decisions=None):
    body = {"payload": document}
    if decisions is not None:
        body["decisions"] = decisions
    response = client.post("/api/presets/import/preview", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def chosen_rows(shown):
    result = []
    for row in shown["rows"]:
        decision = {"index": row["index"], "action": row["action"]}
        if row["action"] == "copy":
            decision["name"] = row["preset"]["name"]
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
    return PresetImportCommit.model_validate({
        "payload": parsed, "decisions": decisions,
        "payload_digest": portable.payload_digest(parsed),
        "collection_revision": portable.collection_revision(presets.read(database)),
    })


def test_export_is_one_complete_snapshot_of_every_checkpoint_without_identity_or_write(client, store, monkeypatch):
    database = store.settings.db_path
    save(database, fields())
    save(database, grid_fields(model_id=OTHER))
    save(database, fields("Selected", main.services.model_id()))
    before = presets.path_for(database).read_bytes()

    def no_write(*args, **kwargs):
        pytest.fail("export wrote the setup file")

    monkeypatch.setattr(presets, "_write", no_write)
    exported = client.get("/api/presets/export")
    assert exported.status_code == 200, exported.text
    disposition = exported.headers["content-disposition"]
    assert "attachment" in disposition and "auradream-generation-setups.json" in disposition
    assert exported.headers["cache-control"] == "no-store"
    assert exported.headers["content-type"].startswith("application/json")
    document = json.loads(exported.content.decode("utf-8"))
    assert set(document) == {"format", "version", "items"}
    assert document["format"] == FORMAT and document["version"] == 1
    assert document["items"] == [
        complete(fields()), complete(grid_fields(model_id=OTHER)),
        complete(fields("Selected", main.services.model_id())),
    ]
    assert document["items"][0]["payload"]["seed"] is None
    assert document["items"][0]["payload"]["guidance_scale"] == 0
    assert "id" not in document["items"][0] and "revision" not in document["items"][0]
    assert presets.path_for(database).read_bytes() == before


def test_export_keeps_its_snapshot_when_crud_commits_during_the_read(client, store, monkeypatch):
    database = store.settings.db_path
    original = save(database, fields())[0]
    read = presets.read
    calls = []

    def capture_then_edit(selected):
        result = read(selected)
        calls.append(selected)
        presets.change(selected, PresetUpdate.model_validate({**fields("Edited"), "revision": 1}), item_id=original.id)
        return result

    monkeypatch.setattr(presets, "read", capture_then_edit)
    response = client.get("/api/presets/export")
    assert response.status_code == 200, response.text
    assert calls == [database]
    assert response.json()["items"] == [complete(fields())]
    assert read(database)[0].name == "Edited"


def test_export_route_is_registered_before_the_item_route(client, store):
    router = api_presets.create_router(main.services)
    paths = [getattr(route, "path", None) for route in router.routes]
    assert paths.index("/api/presets/export") < paths.index("/api/presets/{item_id}")
    response = client.get("/api/presets/export")
    assert response.status_code == 200 and response.json() == payload()


def test_export_and_preview_of_missing_collection_create_no_sidecar(client, store):
    path = presets.path_for(store.settings.db_path)
    assert client.get("/api/presets/export").json() == payload()
    shown = preview(client, payload(fields()))
    assert shown["can_import"] and shown["changed"] and shown["add_count"] == 1
    assert not path.exists() and not list(path.parent.glob("*presets*"))


def test_preview_shows_every_candidate_exact_ids_and_file_index_collisions(client, store):
    database = store.settings.db_path
    first = save(database, fields())[0]
    second = save(database, fields(prompt="local variant"))[1]
    save(database, fields(model_id=OTHER))
    path = presets.path_for(database)
    before = path.read_bytes()
    shown = preview(client, payload(
        fields(), fields(prompt="incoming"), fields("New"), grid_fields("NEW"),
    ))
    assert [row["action"] for row in shown["rows"]] == ["skip", "skip", "add", "add"]
    assert shown["rows"][0]["exact_ids"] == [first.id]
    assert [row["id"] for row in shown["rows"][1]["matches"]] == [first.id, second.id]
    assert shown["rows"][2]["incoming_collisions"] == [3]
    assert shown["rows"][3]["incoming_collisions"] == [2]
    assert (shown["add_count"], shown["replace_count"], shown["skip_count"]) == (2, 0, 2)
    assert path.read_bytes() == before


def test_commit_replaces_only_the_explicit_target_and_appends_in_file_order(client, store):
    database = store.settings.db_path
    first = save(database, fields())[0]
    second = save(database, fields(prompt="local variant"))[1]
    document = payload(fields(prompt="new variant"), fields(), grid_fields("Other", OTHER))
    decisions = [
        {"index": 1, "action": "copy", "name": "Copy"},
        {"index": 0, "action": "replace", "item_id": second.id, "revision": second.revision},
        {"index": 2, "action": "add"},
    ]
    shown = preview(client, document, decisions)
    committed = client.post("/api/presets/import", json=commit_body(document, shown, decisions))
    assert committed.status_code == 200, committed.text
    saved = committed.json()["items"]
    assert [item["name"] for item in saved] == ["Study", "Study", "Copy", "Other"]
    assert saved[0] == first.model_dump()
    assert saved[1] == {**second.model_dump(), "payload": complete(fields(prompt="new variant"))["payload"], "revision": 2}
    assert saved[3]["model_id"] == OTHER and saved[3]["payload"]["mode"] == "grid"
    assert [item["revision"] for item in saved[2:]] == [1, 1]
    assert len({item["id"] for item in saved}) == 4
    assert [item.model_dump() for item in presets.read(database)] == saved


def test_commit_uses_one_existing_atomic_writer_without_capture_validation(client, store, monkeypatch):
    document = payload(fields("First"), grid_fields("Second"))
    shown = preview(client, document)
    calls = []
    write = presets._write

    def counted_write(path, items):
        calls.append([item.name for item in items])
        assert not path.exists()
        return write(path, items)

    def forbidden(*args, **kwargs):
        pytest.fail("import reached capture validation or the model selection")

    monkeypatch.setattr(presets, "_write", counted_write)
    monkeypatch.setattr(api_presets, "validate_payload", forbidden)
    monkeypatch.setattr(model_selection, "current_settings", forbidden)
    result = client.post("/api/presets/import", json=commit_body(document, shown))
    assert result.status_code == 200, result.text
    assert calls == [["First", "Second"]]


def test_unrelated_crud_change_invalidates_whole_preview_without_saving_any_row(client, store):
    database = store.settings.db_path
    save(database, fields("Existing"))
    document = payload(fields("Incoming"))
    shown = preview(client, document)
    save(database, fields("Other tab"))
    before = presets.path_for(database).read_bytes()
    result = client.post("/api/presets/import", json=commit_body(document, shown))
    assert result.status_code == 409, result.text
    assert "changed" in result.json()["detail"]
    assert presets.path_for(database).read_bytes() == before


def test_replaying_a_successful_commit_is_stale_and_never_adds_copies_twice(client, store):
    database = store.settings.db_path
    save(database, fields())
    document = payload(fields())
    shown = preview(client, document, [{"index": 0, "action": "copy"}])
    body = commit_body(document, shown)
    assert client.post("/api/presets/import", json=body).status_code == 200
    before = presets.path_for(database).read_bytes()
    assert client.post("/api/presets/import", json=body).status_code == 409
    assert presets.path_for(database).read_bytes() == before
    assert len(presets.read(database)) == 2


@pytest.mark.parametrize("change", [
    "digest", "payload", "missing_choice", "partial_choice", "duplicate_choice", "extra_choice", "all_skip",
])
def test_stale_file_digest_or_incomplete_choices_are_refused_without_a_sidecar(client, store, change):
    document = payload(fields(), fields("Second"))
    shown = preview(client, document)
    body = commit_body(document, shown)
    if change == "digest":
        body["payload_digest"] = "0" * 64
    elif change == "payload":
        body["payload"] = payload(fields(prompt="different"), fields("Second"))
    elif change == "missing_choice":
        body["decisions"] = []
    elif change == "partial_choice":
        body["decisions"] = body["decisions"][:1]
    elif change == "duplicate_choice":
        body["decisions"] = [body["decisions"][0]] * 2
    elif change == "extra_choice":
        body["decisions"].append({"index": 2, "action": "skip"})
    else:
        body["decisions"] = [{"index": 0, "action": "skip"}, {"index": 1, "action": "skip"}]
    result = client.post("/api/presets/import", json=body)
    assert result.status_code == 422, result.text
    assert "first line" not in result.text
    assert not presets.path_for(store.settings.db_path).exists()


def test_stale_target_revision_is_refused_even_with_a_fresh_collection_pin(client, store):
    database = store.settings.db_path
    original = save(database, fields())[0]
    presets.change(database, PresetUpdate.model_validate({**fields(prompt="updated"), "revision": 1}), item_id=original.id)
    document = payload(fields(prompt="incoming"))
    fresh = preview(client, document)
    stale = [{"index": 0, "action": "replace", "item_id": original.id, "revision": 1}]
    before = presets.path_for(database).read_bytes()
    result = client.post("/api/presets/import", json=commit_body(document, fresh, stale))
    assert result.status_code == 409, result.text
    assert presets.path_for(database).read_bytes() == before


def test_replacement_never_crosses_checkpoints_or_targets_one_row_twice(client, store):
    database = store.settings.db_path
    local = save(database, fields())[0]
    other = save(database, fields(model_id=OTHER))[1]
    before = presets.path_for(database).read_bytes()
    document = payload(fields(prompt="first incoming"), fields(prompt="second incoming"))
    shown = preview(client, document)
    twice = [
        {"index": index, "action": "replace", "item_id": local.id, "revision": local.revision}
        for index in range(2)
    ]
    crossed = [
        {"index": 0, "action": "replace", "item_id": other.id, "revision": other.revision},
        {"index": 1, "action": "skip"},
    ]
    for choices in (twice, crossed):
        response = client.post("/api/presets/import", json=commit_body(document, shown, choices))
        assert response.status_code == 422, response.text
    assert presets.path_for(database).read_bytes() == before


@pytest.mark.parametrize("document", [payload(), payload(fields())])
def test_empty_and_exact_match_only_previews_are_held_and_commit_does_not_write(client, store, document, monkeypatch):
    database = store.settings.db_path
    save(database, fields())
    before = presets.path_for(database).read_bytes()
    shown = preview(client, document)
    assert not shown["changed"] and not shown["can_import"] and shown["reason"]

    def forbidden(*args, **kwargs):
        pytest.fail("a no-op import attempted a write")

    monkeypatch.setattr(presets, "_write", forbidden)
    result = client.post("/api/presets/import", json=commit_body(document, shown))
    assert result.status_code == 422, result.text
    assert presets.path_for(database).read_bytes() == before


_ITEM = '{"name":"a","model_id":"m/x","payload":{"prompt":"private_personal_text"}}'


@pytest.mark.parametrize("raw", [
    b'{"payload":{"format":"auradream.generation-presets","version":1,"items":[]},"payload":{}}',
    b'{"payload":{"format":"auradream.generation-presets","version":1,"items":[{"name":"a","name":"b",'
    b'"model_id":"m/x","payload":{"prompt":"private_personal_text"}}]}}',
    b'{"payload":{"format":"auradream.generation-presets","version":1,"items":[],"private_personal_text":true}}',
    ('{"payload":{"format":"auradream.generation-presets","version":1,"items":['
     + _ITEM.replace('"m/x"', '"m/x","id":"' + "0" * 32 + '"') + "]}}").encode(),
    ('{"payload":{"format":"auradream.generation-presets","version":1,"items":['
     + _ITEM.replace('"private_personal_text"', '"private_personal_text","step_cache":"fbc"') + "]}}").encode(),
    b'{"payload":{"format":"auradream.generation-presets","version":1,"items":[{"name":"a","model_id":"m/x",'
    b'"payload":{"prompt":"\\ud800"}}]}}',
    b'{"payload":{"format":"auradream.generation-presets","version":true,"items":[]}}',
    b'{"payload":{"format":"auradream.generation-presets","version":2,"items":[]}}',
    b'{"payload":{"format":"auradream.generation-presets","version":NaN,"items":[]}}',
    b'{"payload":{"format":"auradream.prompt-snippets","version":1,"items":[]}}',
    b"[" * 17 + b"0" + b"]" * 17,
    b"\xff",
    json.dumps({"payload": payload()}).encode("utf-16"),
])
@pytest.mark.parametrize("route", ["/api/presets/import/preview", "/api/presets/import"])
def test_raw_parser_refusals_are_fixed_and_never_echo_personal_input(client, store, raw, route):
    response = client.post(route, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert "private_personal_text" not in response.text
    assert not presets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("route", ["/api/presets/import/preview", "/api/presets/import"])
def test_request_byte_limit_is_applied_before_json_parsing(client, store, route):
    raw = b" " * (MAX_REQUEST_BYTES + 1)
    response = client.post(route, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.text
    assert response.json()["detail"] == "The setup import request is too large."
    assert not presets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("declared_length", [None, "1"])
def test_stream_limit_ignores_missing_or_incorrect_content_length(client, store, monkeypatch, declared_length):
    monkeypatch.setattr(api_presets, "MAX_IMPORT_REQUEST_BYTES", 128)
    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    response = client.post(
        "/api/presets/import/preview", content=iter([b" " * 80, b" " * 80]), headers=headers,
    )
    assert response.status_code == 413
    assert not presets.path_for(store.settings.db_path).exists()


def test_payload_has_its_own_byte_bound_inside_a_bounded_request(client, store, monkeypatch):
    monkeypatch.setattr(portable, "MAX_PORTABLE_BYTES", 256)
    response = client.post("/api/presets/import/preview", json={"payload": payload(fields(prompt="x" * 500))})
    assert response.status_code == 422
    assert response.json()["detail"] == "The portable setup file would exceed 8 MiB."
    assert not presets.path_for(store.settings.db_path).exists()


@pytest.mark.parametrize("raw", [
    b"broken", b'{"version":2,"items":[]}', b'{"version":true,"items":[]}', b'{"version":1,"items":[],"items":[]}',
])
def test_unreadable_destination_blocks_export_preview_and_commit_without_replacing_file(client, store, raw):
    database = store.settings.db_path
    document = payload(fields())
    body = storage_body(database, document, [{"index": 0, "action": "add"}]).model_dump(mode="json")
    path = presets.path_for(database)
    path.write_bytes(raw)
    assert client.get("/api/presets/export").status_code == 503
    assert client.post("/api/presets/import/preview", json={"payload": document}).status_code == 503
    assert client.post("/api/presets/import", json=body).status_code == 503
    assert path.read_bytes() == raw


def test_import_shares_the_existing_sidecar_with_crud(client, store):
    database = store.settings.db_path
    document = payload(fields())
    shown = preview(client, document)
    result = client.post("/api/presets/import", json=commit_body(document, shown))
    assert result.status_code == 200, result.text
    assert presets.path_for(database).name == "auradream.presets.json"
    assert client.get("/api/presets").json()["items"] == result.json()["items"]
    names = {path.name.lower() for path in database.parent.glob("*presets*")} - {"auradream.presets.json.gallery.lock"}
    assert names == {"auradream.presets.json"}


def test_round_trip_between_scratch_databases_assigns_fresh_ids_and_repeats_as_exact_skips(tmp_path):
    source = tmp_path / "source" / "gallery.db"
    destination = tmp_path / "destination" / "gallery.db"
    save(source, fields())
    original = save(source, grid_fields(model_id=OTHER))
    raw = portable.export_portable(original)
    document = json.loads(raw)
    body = storage_body(destination, document, [{"index": 0, "action": "add"}, {"index": 1, "action": "add"}])
    saved = presets.import_collection(destination, body)
    assert [item.model_dump(include={"name", "model_id", "payload"}) for item in saved] == document["items"]
    assert {item.id for item in saved}.isdisjoint({item.id for item in original})
    assert [item.revision for item in saved] == [1, 1]
    assert presets.read(source) == original and presets.read(destination) == saved
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
            return presets.import_collection(database, body)
        except (presets.PresetsError, portable.PortabilityError) as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: import_once(), range(2)))
    successes = [result for result in results if isinstance(result, list)]
    assert len(successes) == 1 and results.count(409) == 1
    assert presets.read(database) == successes[0] and len(successes[0]) == 1


def test_concurrent_crud_and_import_from_one_snapshot_never_merge_partially(tmp_path):
    database = tmp_path / "gallery.db"
    local = save(database, fields("Local"))[0]
    body = storage_body(database, payload(fields("Incoming")), [{"index": 0, "action": "add"}])
    barrier = Barrier(2)

    def crud():
        barrier.wait(timeout=5)
        try:
            return presets.change(
                database, PresetUpdate.model_validate({**fields("Renamed"), "revision": 1}), item_id=local.id,
            )
        except presets.PresetsError as exc:
            return exc.status

    def import_once():
        barrier.wait(timeout=5)
        try:
            return presets.import_collection(database, body)
        except (presets.PresetsError, portable.PortabilityError) as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        edited, imported = pool.submit(crud), pool.submit(import_once)
        edited, imported = edited.result(), imported.result()
    assert isinstance(edited, list)
    final = presets.read(database)
    if isinstance(imported, list):
        # The import won the reservation; the CRUD edit then applied on top.
        assert [item.name for item in final] == ["Renamed", "Incoming"]
    else:
        assert imported == 409
        assert final == edited and [item.name for item in final] == ["Renamed"]


@pytest.mark.parametrize("operation", ["write", "fsync", "replace"])
def test_import_write_failures_preserve_every_old_row_and_remove_temporary_files(tmp_path, monkeypatch, operation):
    database = tmp_path / "gallery.db"
    save(database, fields("Kept"))
    body = storage_body(database, payload(fields("Incoming 1"), fields("Incoming 2")), [
        {"index": 0, "action": "add"}, {"index": 1, "action": "add"},
    ])
    path = presets.path_for(database)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise PermissionError("synthetic storage failure")

    if operation == "write":
        temporary_file = presets.tempfile.NamedTemporaryFile

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

        monkeypatch.setattr(presets.tempfile, "NamedTemporaryFile", BrokenWriter)
    else:
        monkeypatch.setattr(presets.os, operation, fail)
    with pytest.raises((presets.PresetsError, portable.PortabilityError)) as error:
        presets.import_collection(database, body)
    assert error.value.status == 503
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".auradream.presets.*.tmp")) == []


def test_preset_lock_contention_refuses_without_changing_the_destination(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    save(database, fields("Kept"))
    body = storage_body(database, payload(fields("Incoming")), [{"index": 0, "action": "add"}])
    path = presets.path_for(database)
    before = path.read_bytes()

    @contextmanager
    def busy(selected, **kwargs):
        assert selected == path
        raise gallery_lock.GalleryBusy("synthetic external holder")
        yield

    monkeypatch.setattr(gallery_lock, "hold", busy)
    with pytest.raises((presets.PresetsError, portable.PortabilityError)) as error:
        presets.import_collection(database, body)
    assert error.value.status == 409
    assert path.read_bytes() == before


def test_commit_rechecks_final_count_under_lock_without_partial_additions(tmp_path):
    database = tmp_path / "gallery.db"
    first = save(database, fields("Local 0"))[0]
    full = [first.model_copy(update={"id": f"{index:032x}", "name": f"Local {index}"}) for index in range(MAX_PRESETS)]
    presets._write(presets.path_for(database), full)
    body = storage_body(database, payload(fields("Too many")), [{"index": 0, "action": "add"}])
    before = presets.path_for(database).read_bytes()
    with pytest.raises((presets.PresetsError, portable.PortabilityError)) as error:
        presets.import_collection(database, body)
    assert error.value.status == 422
    assert presets.path_for(database).read_bytes() == before


def test_commit_rechecks_combined_encoded_bytes_without_partial_additions(tmp_path):
    database = tmp_path / "gallery.db"
    # Each escaped NUL is six encoded bytes; one full Grid draft is about 89 KB,
    # so 50 local plus 50 incoming exceed the 8-MiB sidecar only together.
    def nul(length):
        return chr(0) * (length - 1) + "x"

    heavy = grid_fields(prompt=nul(4000), negative_prompt=nul(4000))
    heavy["payload"]["grid"].update(
        prompt_axis=nul(2000), size_axis=nul(2000), seed_axis=nul(2000),
        steps_axis=nul(200), cfg_axis=nul(200), mu_axis=nul(200),
    )
    first = save(database, {**heavy, "name": "Local 0"})[0]
    local = [first.model_copy(update={"id": f"{index:032x}", "name": f"Local {index}"}) for index in range(50)]
    presets._write(presets.path_for(database), local)
    incoming = payload(*({**heavy, "name": f"Incoming {index}"} for index in range(50)))
    body = storage_body(database, incoming, [{"index": index, "action": "add"} for index in range(50)])
    before = presets.path_for(database).read_bytes()
    with pytest.raises((presets.PresetsError, portable.PortabilityError)) as error:
        presets.import_collection(database, body)
    assert error.value.status == 422
    assert "8 MiB" in str(error.value)
    assert presets.path_for(database).read_bytes() == before


def test_foreign_and_unavailable_checkpoints_are_stored_raw_without_switching(client, store):
    model_before = model_selection._selection
    document = payload(fields(model_id="someone/Not-In-The-Catalogue"), grid_fields(model_id=OTHER))
    shown = preview(client, document)
    result = client.post("/api/presets/import", json=commit_body(document, shown))
    assert result.status_code == 200, result.text
    saved = result.json()["items"]
    assert [item["model_id"] for item in saved] == ["someone/Not-In-The-Catalogue", OTHER]
    assert [item["payload"] for item in saved] == [complete(item)["payload"] for item in document["items"]]
    assert model_selection._selection == model_before


@pytest.mark.parametrize("maintenance", [False, True])
def test_portability_routes_bypass_gallery_model_switch_hub_and_queue_work(client, store, fake, monkeypatch, maintenance):
    manager = main.bundle_manager
    if maintenance:
        monkeypatch.setitem(manager.state, "state", "running")
    # A switch in progress refuses model-dependent requests; transfer is not one.
    monkeypatch.setattr(main, "_switching", True)

    def forbidden(*args, **kwargs):
        pytest.fail("setup portability reached model/gallery/queue/Hub work")

    for obj, names in (
        (fake, ("describe", "load", "generate")),
        (main.jobs, ("submit", "submit_cells", "run_when_idle", "wait_for_gate")),
        (main, ("ensure_loaded",)),
        (main.hub_access, ("check",)),
        (api_presets, ("validate_payload",)),
        (model_selection, ("current_settings",)),
    ):
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    read = presets.read
    hold = gallery_lock.hold
    preset_path = presets.path_for(store.settings.db_path)

    def outside_gallery(selected):
        assert manager.readers == 0 and main._model_requests == 0
        return read(selected)

    @contextmanager
    def only_preset_lock(selected, **kwargs):
        assert selected == preset_path
        assert manager.readers == 0 and main._model_requests == 0
        with hold(selected, **kwargs):
            yield

    monkeypatch.setattr(presets, "read", outside_gallery)
    monkeypatch.setattr(gallery_lock, "hold", only_preset_lock)
    headers = {"X-AuraDream-Model-Revision": "malformed"}
    document = payload(fields("Portable"))
    assert client.get("/api/presets/export", headers=headers).status_code == 200
    shown = preview(client, document)
    result = client.post("/api/presets/import", json=commit_body(document, shown), headers=headers)
    assert result.status_code == 200, result.text
    assert not fake.requests
    if maintenance:
        assert client.get("/api/presets/import/unrelated").status_code == 409
        assert client.get("/api/presets/export/unrelated").status_code == 409


def test_transfer_routes_are_not_model_dependent():
    for method, path in (
        ("GET", "/api/presets/export"), ("POST", "/api/presets/import/preview"), ("POST", "/api/presets/import"),
    ):
        assert not api_presets.depends_on_model(method, path)
    assert api_presets.depends_on_model("POST", "/api/presets")


@pytest.mark.parametrize("route", ["/api/presets/import/preview", "/api/presets/import"])
def test_import_routes_refuse_foreign_origin_before_writing(client, store, route):
    result = client.post(route, json={"payload": payload(fields())}, headers={"Origin": "https://elsewhere.example"})
    assert result.status_code == 403
    assert not presets.path_for(store.settings.db_path).exists()


def test_imported_setups_keep_preset_model_identity(client, store):
    document = payload(fields())
    shown = preview(client, document)
    saved = client.post("/api/presets/import", json=commit_body(document, shown)).json()["items"]
    assert Preset.model_validate(saved[0]).model_id == FOREIGN
