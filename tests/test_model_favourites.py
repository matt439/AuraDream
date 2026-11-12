"""Model favourites: one bounded, revisioned, atomic set per gallery database."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend import model_favourites
from backend.schemas_model_favourites import (
    MAX_MODEL_FAVOURITES, ModelFavouriteChange, ModelFavouriteImport,
)

KREA = "black-forest-labs/FLUX.1-Krea-dev"
KREA_LITE = "lite-infer/flux.1-krea-dev-nunchaku-lite-int4_r32-bnb4-text-encoder"


def test_first_read_creates_nothing(tmp_path):
    database = tmp_path / "not-created" / "gallery.db"
    assert model_favourites.read(database) == ([], 0)
    assert not database.parent.exists()


def test_set_changes_are_revisioned_and_databases_are_independent(tmp_path):
    one, two = tmp_path / "one.db", tmp_path / "two.db"
    assert model_favourites.path_for(one).name == "auradream.model-favourites-one.db.json"
    assert model_favourites.change(one, KREA, True) == ([KREA], 1)
    # A repeated star writes nothing and keeps the revision.
    before = model_favourites.path_for(one).read_bytes()
    assert model_favourites.change(one, KREA, True) == ([KREA], 1)
    assert model_favourites.path_for(one).read_bytes() == before
    assert model_favourites.change(one, "someone/custom", True) == ([KREA, "someone/custom"], 2)
    assert model_favourites.change(one, KREA, False) == (["someone/custom"], 3)
    assert model_favourites.change(one, KREA, False) == (["someone/custom"], 3)
    assert model_favourites.read(two) == ([], 0)
    assert not one.exists() and not two.exists()


def test_builds_of_one_entry_share_its_canonical_id(tmp_path):
    database = tmp_path / "gallery.db"
    assert model_favourites.canonical(KREA_LITE) == KREA
    assert model_favourites.canonical("someone/custom") == "someone/custom"
    assert model_favourites.change(database, KREA_LITE, True) == ([KREA], 1)
    assert model_favourites.change(database, KREA_LITE, False) == ([], 2)


def test_merge_adds_only_new_ids_and_never_removes(tmp_path):
    database = tmp_path / "gallery.db"
    model_favourites.change(database, "a/one", True)
    items, revision, added = model_favourites.merge(database, ["a/one", KREA_LITE, KREA, "b/two"])
    assert (items, revision, added) == (["a/one", KREA, "b/two"], 2, 2)
    assert model_favourites.merge(database, ["a/one"]) == (["a/one", KREA, "b/two"], 2, 0)


def test_the_set_is_bounded_before_publication(tmp_path):
    database = tmp_path / "gallery.db"
    path = model_favourites.path_for(database)
    full = [f"a/{n}" for n in range(MAX_MODEL_FAVOURITES)]
    model_favourites._write(path, full, 7)
    before = path.read_bytes()
    with pytest.raises(model_favourites.ModelFavouritesError, match="at most") as error:
        model_favourites.change(database, "a/new", True)
    assert error.value.status == 422
    with pytest.raises(model_favourites.ModelFavouritesError, match="at most"):
        model_favourites.merge(database, ["a/new"])
    assert path.read_bytes() == before
    # Unstarring a full set still works.
    assert model_favourites.change(database, "a/0", False)[1] == 8


@pytest.mark.parametrize("data", [
    b"not JSON", b'{"version":2,"revision":0,"items":[]}', b'{"version":1,"items":[]}',
    b'{"version":1,"revision":-1,"items":[]}', b'{"version":1,"revision":true,"items":[]}',
    b'{"version":1,"revision":0,"items":["a","a"]}', b'{"version":1,"revision":0,"items":[" a"]}',
    b'{"version":1,"revision":0,"items":[""]}', b'{"version":1,"revision":0,"items":[3]}',
    b'{"version":1,"revision":0,"items":["a\\u0007"]}', b'{"version":1,"revision":0,"revision":0,"items":[]}',
    b'{"version":1,"revision":0,"items":[],"extra":1}', b"[" * 2000 + b"]" * 2000,
    '{"version":1,"revision":0,"items":[]}'.encode("utf-16"),
])
def test_invalid_or_newer_files_are_never_overwritten(tmp_path, data):
    database = tmp_path / "gallery.db"
    path = model_favourites.path_for(database)
    path.write_bytes(data)
    with pytest.raises(model_favourites.ModelFavouritesError, match="left unchanged"):
        model_favourites.read(database)
    with pytest.raises(model_favourites.ModelFavouritesError, match="left unchanged"):
        model_favourites.change(database, KREA, True)
    assert path.read_bytes() == data


def test_a_too_large_file_is_refused(tmp_path):
    database = tmp_path / "gallery.db"
    path = model_favourites.path_for(database)
    path.write_bytes(b" " * (model_favourites.MAX_FILE_BYTES + 1))
    with pytest.raises(model_favourites.ModelFavouritesError):
        model_favourites.read(database)


def test_a_failed_replace_keeps_the_previous_set(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    model_favourites.change(database, KREA, True)
    before = model_favourites.path_for(database).read_bytes()

    def refuse(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(model_favourites.os, "replace", refuse)
    with pytest.raises(model_favourites.ModelFavouritesError, match="previous set"):
        model_favourites.change(database, "a/new", True)
    assert model_favourites.path_for(database).read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


@pytest.mark.parametrize("body", [
    {"model_id": "", "favourite": True}, {"model_id": "  ", "favourite": True},
    {"model_id": "x" * 513, "favourite": True}, {"model_id": "a\nb", "favourite": True},
    {"model_id": "\ud800", "favourite": True}, {"model_id": 3, "favourite": True},
    {"model_id": "a", "favourite": "true"}, {"model_id": "a", "favourite": True, "extra": 1},
])
def test_change_fields_are_strict(body):
    with pytest.raises(ValidationError):
        ModelFavouriteChange(**body)


def test_import_fields_are_strict_and_trimmed():
    assert ModelFavouriteImport(model_ids=["  a/b  "]).model_ids == ["a/b"]
    for body in ({"model_ids": "a"}, {"model_ids": [""]}, {"model_ids": ["a"] * (MAX_MODEL_FAVOURITES + 1)}):
        with pytest.raises(ValidationError):
            ModelFavouriteImport(**body)


def test_the_file_is_plain_versioned_json(tmp_path):
    database = tmp_path / "gallery.db"
    model_favourites.change(database, "Ünïcode/model", True)
    payload = json.loads(model_favourites.path_for(database).read_text(encoding="utf-8"))
    assert payload == {"version": 1, "revision": 1, "items": ["Ünïcode/model"]}
