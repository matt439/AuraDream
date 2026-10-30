"""Saved filters have their own bounded, database-scoped atomic definitions."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from pydantic import ValidationError

from backend import gallery_lock, saved_searches
from backend.schemas_saved_searches import MAX_SAVED_SEARCHES, SavedSearchFields, SavedSearchUpdate
from tests.bundle_helpers import _holding_child


def fields(**updates):
    return SavedSearchFields(**{"name": "Keepers", "query": "", "favorites": True, "all_models": False, **updates})


def test_first_read_creates_nothing(tmp_path):
    database = tmp_path / "not-created" / "gallery.db"
    assert saved_searches.read(database) == []
    assert not database.parent.exists()


def test_crud_and_two_databases_in_one_directory_are_independent(tmp_path):
    one, two = tmp_path / "one.db", tmp_path / "two.db"
    text = "  watercolor\nmodel:=black-forest-labs/FLUX.1-Krea-dev date:today  "
    original = saved_searches.change(one, fields(name="  Keepers  ", query=text))[0]
    assert (original.name, original.query, original.favorites, original.all_models, original.revision) == (
        "Keepers", text, True, False, 1,
    )
    assert saved_searches.path_for(one).name == "auradream.saved-searches-one.db.json"
    assert saved_searches.read(one) == [original]
    assert saved_searches.read(two) == []
    assert not one.exists() and not two.exists()
    saved_searches.change(two, fields())
    changed = saved_searches.change(one, SavedSearchUpdate(
        name="Changed", query="size:512x512", favorites=False, all_models=True, revision=1,
    ), item_id=original.id)[0]
    assert changed.id == original.id and changed.revision == 2
    with pytest.raises(saved_searches.SavedSearchesError) as error:
        saved_searches.change(one, fields(), item_id=original.id, revision=1)
    assert error.value.status == 409
    with pytest.raises(saved_searches.SavedSearchesError) as error:
        saved_searches.change(one, item_id=original.id, revision=1)
    assert error.value.status == 409
    assert saved_searches.read(one) == [changed]
    assert saved_searches.change(one, item_id=original.id, revision=2) == []
    assert len(saved_searches.read(two)) == 1


@pytest.mark.parametrize("updates", [
    {"name": ""}, {"name": " "}, {"name": "x" * 81}, {"name": "x\x7f"}, {"name": "x\x85y"},
    {"name": "x\ny"}, {"name": "\ud800"}, {"query": "\ud800"}, {"query": "x" * 4001},
    {"query": "seed:99999999999999999999"}, {"query": "size:invalid"}, {"query": "date:invalid"},
    {"query": "steps:secret"}, {"query": "model:="}, {"favorites": 1}, {"all_models": "true"},
    {"name": 3}, {"query": None}, {"extra": True},
])
def test_strict_fields_and_grammar_refuse_invalid_values(updates):
    with pytest.raises(ValidationError):
        fields(**updates)


def test_trimmed_name_length_and_query_boundary_are_accepted():
    item = fields(name=" " * 100 + "x" * 80 + " " * 100, query="x" * 4000)
    assert item.name == "x" * 80 and len(item.query) == 4000
    assert fields(query="").query == ""
    assert fields(query="unknown:value odd ' quotes").query == "unknown:value odd ' quotes"


@pytest.mark.parametrize("first, duplicate", [("Caf\u00e9", "CAFE\u0301"), ("Straße", "STRASSE"), ("ΟΣ", "ος")])
def test_unique_names_use_nfc_and_casefold(tmp_path, first, duplicate):
    database = tmp_path / "gallery.db"
    original = saved_searches.change(database, fields(name=first))[0]
    with pytest.raises(saved_searches.SavedSearchesError) as error:
        saved_searches.change(database, fields(name=duplicate))
    assert error.value.status == 409 and error.value.conflict_id == original.id
    assert saved_searches.read(database) == [original]
    other = saved_searches.change(database, fields(name="other"))[-1]
    with pytest.raises(saved_searches.SavedSearchesError) as error:
        saved_searches.change(database, SavedSearchUpdate(
            **fields(name=duplicate).model_dump(), revision=1,
        ), item_id=other.id)
    assert error.value.conflict_id == original.id
    # A spelling-only update to the same entry is allowed.
    updated = saved_searches.change(database, SavedSearchUpdate(
        **fields(name=duplicate).model_dump(), revision=1,
    ), item_id=original.id)[0]
    assert updated.name == duplicate and updated.revision == 2


@pytest.mark.parametrize("data", [
    b"not JSON", b'{"version":2,"items":[]}', b'{"version":true,"items":[]}',
    b'{"version":1,"items":[],"extra":true}', b'{"version":1,"items":{},"extra":true}',
    b'{"version":1,"version":1,"items":[]}',
    b'{"version":1,"items":[{"id":"bad","name":"x","query":"","favorites":true,"all_models":false,"revision":1}]}',
    b'{"version":1,"items":[{"id":"00000000000000000000000000000000","name":"x","query":"","revision":1}]}',
    b"[" * 2000 + b"]" * 2000,
    '{"version":1,"items":[]}'.encode("utf-16"),
])
def test_invalid_or_newer_files_are_never_overwritten(tmp_path, data):
    database = tmp_path / "gallery.db"
    path = saved_searches.path_for(database)
    path.write_bytes(data)
    with pytest.raises(saved_searches.SavedSearchesError, match="left unchanged"):
        saved_searches.read(database)
    with pytest.raises(saved_searches.SavedSearchesError, match="left unchanged"):
        saved_searches.change(database, fields())
    assert path.read_bytes() == data


@pytest.mark.parametrize("duplicate", ["id", "name"])
def test_persisted_duplicate_ids_or_names_are_refused(tmp_path, duplicate):
    database = tmp_path / "gallery.db"
    item = saved_searches.change(database, fields())[0].model_dump()
    second = {**item, "id": "f" * 32, "name": "other"}
    second[duplicate] = item[duplicate]
    path = saved_searches.path_for(database)
    data = json.dumps({"version": 1, "items": [item, second]}).encode("utf-8")
    path.write_bytes(data)
    with pytest.raises(saved_searches.SavedSearchesError):
        saved_searches.change(database, fields(name="new"))
    assert path.read_bytes() == data


def test_file_and_collection_limits_are_enforced_before_publication(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    original = saved_searches.change(database, fields())[0]
    full = [original.model_copy(update={"id": f"{n:032x}", "name": f"name {n}"}) for n in range(MAX_SAVED_SEARCHES)]
    path = saved_searches.path_for(database)
    saved_searches._write(path, full)
    before = path.read_bytes()
    with pytest.raises(saved_searches.SavedSearchesError, match="at most"):
        saved_searches.change(database, fields(name="one too many"))
    assert path.read_bytes() == before
    with pytest.raises(saved_searches.SavedSearchesError) as error:
        saved_searches.change(database, fields(name="NAME 0"))
    assert error.value.status == 409 and error.value.conflict_id == full[0].id
    assert path.read_bytes() == before
    monkeypatch.setattr(saved_searches, "MAX_FILE_BYTES", len(before) - 1)
    with pytest.raises(saved_searches.SavedSearchesError, match="left unchanged"):
        saved_searches.read(database)
    assert path.read_bytes() == before
    path.unlink()
    monkeypatch.setattr(saved_searches, "MAX_FILE_BYTES", 1)
    with pytest.raises(saved_searches.SavedSearchesError, match="1 MiB"):
        saved_searches.change(database, fields())
    assert not path.exists()


@pytest.mark.parametrize("operation", ["replace", "fsync"])
def test_write_failures_keep_previous_file_and_only_clean_owned_temp(tmp_path, monkeypatch, operation):
    database = tmp_path / "gallery.db"
    saved_searches.change(database, fields())
    path = saved_searches.path_for(database)
    before = path.read_bytes()
    foreign = tmp_path / ".auradream.saved-searches.foreign.tmp"
    foreign.write_bytes(b"someone else's temporary")

    def fail(*args):
        raise PermissionError("secret query must never be logged")

    monkeypatch.setattr(saved_searches.os, operation, fail)
    with pytest.raises(saved_searches.SavedSearchesError, match="previous saved list") as error:
        saved_searches.change(database, fields(name="new"))
    assert error.value.__cause__ is None
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".auradream.saved-searches.*.tmp")) == [foreign]


def test_unreadable_file_is_preserved(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    path = saved_searches.path_for(database)
    original_open = type(path).open

    def refuse(target, *args, **kwargs):
        if target == path:
            raise PermissionError("unreadable")
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", refuse)
    with pytest.raises(saved_searches.SavedSearchesError, match="left unchanged"):
        saved_searches.change(database, fields())


def test_parallel_creates_and_update_delete_are_serialized(tmp_path):
    database = tmp_path / "gallery.db"
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda n: saved_searches.change(database, fields(name=f"name {n}")), range(8)))
    assert {item.name for item in saved_searches.read(database)} == {f"name {n}" for n in range(8)}
    original = saved_searches.read(database)[0]
    start = Barrier(2)

    def modify(delete):
        start.wait(timeout=5)
        try:
            return saved_searches.change(
                database, None if delete else SavedSearchUpdate(**fields(name=original.name).model_dump(), revision=1),
                item_id=original.id, revision=1,
            )
        except saved_searches.SavedSearchesError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(modify, [False, True]))
    assert sum(isinstance(result, list) for result in results) == 1
    assert any(result in (404, 409) for result in results if isinstance(result, int))


def test_sidecar_lock_is_independent_of_gallery_reservation(tmp_path):
    database = tmp_path / "gallery.db"
    with gallery_lock.hold(database):
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(saved_searches.change, database, fields()).result(timeout=5)


def test_lock_refusal_is_reported_with_fixed_message(tmp_path, monkeypatch):
    def busy(*args, **kwargs):
        raise gallery_lock.GalleryBusy("private detail")

    monkeypatch.setattr(gallery_lock, "hold", busy)
    with pytest.raises(saved_searches.SavedSearchesError, match="Another AuraDream process") as error:
        saved_searches.change(tmp_path / "gallery.db", fields())
    assert error.value.status == 409 and error.value.__cause__ is None


def test_cross_process_reservation_serializes_create_update_delete_and_stays_independent(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    # A held sidecar lock refuses only after the writer's wait, read per call.
    monkeypatch.setattr(gallery_lock, "SESSION_WAIT", 0.3)
    original = saved_searches.change(database, fields())[0]
    path = saved_searches.path_for(database)
    before = path.read_bytes()
    child = _holding_child(path)
    try:
        changes = [
            lambda: saved_searches.change(database, fields(name="other")),
            lambda: saved_searches.change(database, SavedSearchUpdate(
                **fields(query="new query").model_dump(), revision=1,
            ), item_id=original.id),
            lambda: saved_searches.change(database, item_id=original.id, revision=1),
        ]
        for change in changes:
            with pytest.raises(saved_searches.SavedSearchesError) as error:
                change()
            assert error.value.status == 409
            assert path.read_bytes() == before
        # Readers need neither reservation, including while another process
        # is preparing its replacement under the sidecar's own reservation.
        assert saved_searches.read(database) == [original]
    finally:
        child.stdin.write("\n")
        child.stdin.flush()
        child.stdin.close()
        child.wait(timeout=15)

    code = """
import sys
from pathlib import Path
from backend import saved_searches
from backend.schemas_saved_searches import SavedSearchFields, SavedSearchUpdate
db = Path(sys.argv[1])
original = saved_searches.read(db)[0]
fields = SavedSearchFields(name='child', query='date:yesterday', favorites=False, all_models=True)
saved_searches.change(db, fields)
saved_searches.change(db, SavedSearchUpdate(**{**fields.model_dump(), 'name': 'updated'}, revision=1), item_id=original.id)
saved_searches.change(db, item_id=original.id, revision=2)
"""
    # Holding the actual gallery cannot block a different process's saved
    # search CRUD; no SQL, model loading or gallery reservation is needed.
    with gallery_lock.hold(database):
        result = subprocess.run(
            [sys.executable, "-c", code, str(database)], capture_output=True, text=True, timeout=15,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert [(item.name, item.query) for item in saved_searches.read(database)] == [("child", "date:yesterday")]
