"""Instruction-edit recovery keeps intent while proving every source remap."""
from __future__ import annotations

import dataclasses
import json

import pytest
from PIL import Image

from backend import reindex, storage
from backend.instruction_edit import edit_key
from tests.bundle_helpers import _metadata, _save, _snapshot
from tests.instruction_edit_helpers import save_edit, stored_edit


def _rewrite(path, meta):
    path.write_bytes(reindex._with_auradream_metadata(path.read_bytes(), meta))


def _destination(store, monkeypatch, tmp_path):
    destination = dataclasses.replace(store.settings, db_path=tmp_path / "new.db", outputs_dir=tmp_path / "new")
    monkeypatch.setattr(storage, "settings", destination)
    monkeypatch.setattr(reindex, "settings", destination)
    storage.init_db()
    return destination


@pytest.mark.parametrize("database", [False, True])
def test_recovery_remaps_full_proof_and_keeps_source_gallery_unchanged(store, monkeypatch, tmp_path, database):
    source = _save(seed=9)
    child = save_edit(source, seed=1)
    original_settings = store.settings
    original_files = {path.name: path.read_bytes() for path in original_settings.outputs_dir.glob("*.png")}
    destination = _destination(store, monkeypatch, tmp_path)
    unrelated = _save("unrelated", seed=70)
    counts = reindex.reindex(original_settings.outputs_dir, database=original_settings.db_path if database else None)
    records = storage.list_images()[0]
    imported_source = next(record for record in records if record.filename == source.filename)
    imported_child = next(record for record in records if record.filename == child.filename)
    assert imported_source.id != source.id and unrelated.id == source.id
    assert counts["indexed"] == 2 and counts["unresolved_edits"] == 0
    assert imported_child.edit.source == imported_source.id
    assert imported_child.edit.edit_source == child.edit.edit_source
    assert not imported_child.edit_missing
    assert _metadata(destination.outputs_dir / child.filename)["edit"] == imported_child.edit.model_dump(mode="json")
    assert {path.name: path.read_bytes() for path in original_settings.outputs_dir.glob("*.png")} == original_files
    repeated = reindex.reindex(original_settings.outputs_dir, database=original_settings.db_path if database else None)
    assert repeated["present"] == 2 and repeated["indexed"] == 0


@pytest.mark.parametrize("database", [False, True])
@pytest.mark.parametrize("broken", ["missing", "hash", "pixels", "unresolved"])
def test_unproven_source_retains_null_intent_and_descriptor(store, monkeypatch, tmp_path, database, broken):
    source = _save(seed=9)
    child = save_edit(source, seed=1)
    old = store.settings
    path = old.outputs_dir / child.filename
    meta = _metadata(path)
    if broken == "missing":
        (old.outputs_dir / source.filename).unlink()
    elif broken == "pixels":
        Image.new("RGB", (256, 256), "red").save(old.outputs_dir / source.filename)
    elif broken == "hash":
        meta["edit"]["edit_source"]["hash"] = "0" * 64
    else:
        meta["edit"]["source"] = None
    _rewrite(path, meta)
    with storage._session() as conn:
        conn.execute("UPDATE images SET edit = ? WHERE id = ?", (json.dumps(meta["edit"]), child.id))
    destination = _destination(store, monkeypatch, tmp_path)
    _save("numeric collision", seed=40)
    counts = reindex.reindex(old.outputs_dir, database=old.db_path if database else None)
    imported = next(record for record in storage.list_images()[0] if record.filename == child.filename)
    assert counts["unresolved_edits"] == 1
    assert imported.edit.source is None and imported.edit_missing
    assert imported.edit.edit_source.model_dump(mode="json") == meta["edit"]["edit_source"]
    assert imported.prompt == child.prompt and imported.edit.recipe == "kontext-v1"
    assert _metadata(destination.outputs_dir / child.filename)["edit"] == imported.edit.model_dump(mode="json")
    retained = edit_key(imported.edit)
    reindex.reindex(old.outputs_dir, database=old.db_path if database else None)
    assert edit_key(storage.get_image(imported.id).edit) == retained


def test_database_edit_source_can_arrive_after_child_in_id_order(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    old = store.settings
    meta = _metadata(old.outputs_dir / child.filename)
    meta["edit"]["source"] = 99
    _rewrite(old.outputs_dir / child.filename, meta)
    with storage._session() as conn:
        conn.execute("UPDATE images SET id = 99 WHERE id = ?", (source.id,))
        conn.execute("UPDATE images SET edit = ? WHERE id = ?", (json.dumps(meta["edit"]), child.id))
    _destination(store, monkeypatch, tmp_path)
    counts = reindex.reindex(old.outputs_dir, database=old.db_path)
    imported = {record.filename: record for record in storage.list_images()[0]}
    assert counts["unresolved_edits"] == 0
    assert imported[child.filename].edit.source == imported[source.filename].id


@pytest.mark.parametrize("database", [False, True])
def test_dry_run_changes_no_files_or_database_and_counts_proven_links(store, monkeypatch, tmp_path, database):
    source = _save(seed=9)
    save_edit(source, seed=1)
    old = store.settings
    destination = _destination(store, monkeypatch, tmp_path)
    destination.db_path.unlink()
    before = {path.name: path.read_bytes() for path in old.outputs_dir.glob("*.png")}
    counts = reindex.reindex(old.outputs_dir, database=old.db_path if database else None, dry_run=True)
    assert counts["indexed"] == 2 and counts["unresolved_edits"] == 0
    assert not destination.db_path.exists() and not list(destination.outputs_dir.iterdir())
    assert {path.name: path.read_bytes() for path in old.outputs_dir.glob("*.png")} == before


def test_in_place_rebuild_rewrites_edit_source_id_and_preserves_proof(store, monkeypatch):
    removed = _save(seed=4)
    source = _save(seed=9)
    child = save_edit(source, seed=1)
    monkeypatch.setattr(reindex, "settings", store.settings)
    (store.settings.outputs_dir / removed.filename).unlink()
    store.settings.db_path.unlink()
    counts = reindex.reindex(store.settings.outputs_dir)
    rebuilt = {record.filename: record for record in storage.list_images()[0]}
    assert counts["unresolved_edits"] == 0
    assert rebuilt[source.filename].id != source.id
    assert rebuilt[child.filename].edit.source == rebuilt[source.filename].id
    assert rebuilt[child.filename].edit.edit_source == child.edit.edit_source
    assert _metadata(store.settings.outputs_dir / child.filename)["edit"] == rebuilt[child.filename].edit.model_dump(mode="json")


@pytest.mark.parametrize("disagreement", ["png", "sql", "descriptor"])
def test_database_representation_disagreement_refuses_before_any_destination_mutation(store, monkeypatch, tmp_path, disagreement):
    source = _save()
    child = save_edit(source)
    old = store.settings
    path = old.outputs_dir / child.filename
    meta = _metadata(path)
    if disagreement == "png":
        meta.pop("edit")
    elif disagreement == "descriptor":
        meta["edit"]["edit_source"]["hash"] = "1" * 64
    else:
        with storage._session() as conn:
            conn.execute("UPDATE images SET edit = NULL WHERE id = ?", (child.id,))
    _rewrite(path, meta)
    _destination(store, monkeypatch, tmp_path)
    _save("retained", seed=8)
    before = _snapshot(storage)
    with pytest.raises(ValueError, match="disagree"):
        reindex.reindex(old.outputs_dir, database=old.db_path)
    assert _snapshot(storage) == before


@pytest.mark.parametrize("database", [False, True])
def test_existing_file_fast_path_checks_its_own_sql_png_edit_agreement(store, monkeypatch, database):
    source = _save()
    child = save_edit(source)
    monkeypatch.setattr(reindex, "settings", store.settings)
    path = store.settings.outputs_dir / child.filename
    meta = _metadata(path)
    meta.pop("edit")
    _rewrite(path, meta)
    before = _snapshot(store)
    with pytest.raises(ValueError, match="disagree"):
        reindex.reindex(store.settings.outputs_dir, database=store.settings.db_path if database else None)
    assert _snapshot(store) == before


@pytest.mark.parametrize("database", [False, True])
def test_cyclic_edit_sources_remain_unresolved_after_recovery(store, monkeypatch, tmp_path, database):
    source = _save()
    first = save_edit(source, seed=2)
    second = save_edit(first, seed=3)
    old = store.settings
    cyclic = stored_edit(second)
    path = old.outputs_dir / first.filename
    meta = _metadata(path)
    meta["edit"] = cyclic.model_dump(mode="json")
    _rewrite(path, meta)
    with storage._session() as conn:
        conn.execute("UPDATE images SET edit = ? WHERE id = ?", (edit_key(cyclic), first.id))
    _destination(store, monkeypatch, tmp_path)
    counts = reindex.reindex(old.outputs_dir, database=old.db_path if database else None)
    edits = [record for record in storage.list_images()[0] if record.edit is not None]
    assert counts["unresolved_edits"] == 2 and len(edits) == 2
    assert all(record.edit.source is None and record.edit_missing for record in edits)


def test_png_conflicting_workflow_refuses_before_copies_or_rows(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    old = store.settings
    path = old.outputs_dir / child.filename
    meta = _metadata(path)
    meta["refine"] = source.id
    _rewrite(path, meta)
    _destination(store, monkeypatch, tmp_path)
    before = _snapshot(storage)
    with pytest.raises(ValueError, match="another generation workflow"):
        reindex.reindex(old.outputs_dir)
    assert _snapshot(storage) == before


def test_database_binned_source_retains_proven_link_and_unavailable_state(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    storage.bin_images([source.id])
    old = store.settings
    _destination(store, monkeypatch, tmp_path)
    counts = reindex.reindex(old.outputs_dir, database=old.db_path)
    imported = next(record for record in storage.list_images()[0] if record.filename == child.filename)
    assert counts["unresolved_edits"] == 0
    assert imported.edit.source is not None and imported.edit_missing and imported.edit_binned
