"""PNG recovery must never interpret another database's conditioning ids."""
from __future__ import annotations

import dataclasses
import io
import json
import sqlite3
from contextlib import closing
from shutil import copyfile

import pytest
from PIL import Image

from backend import reindex, storage
from backend.schemas import GenerateRequest


def _save(prompt, seed, **kwargs):
    return storage.save_image(
        Image.new("RGB", (32, 24), "red" if prompt == "source" else "blue"),
        GenerateRequest(prompt=prompt, steps=28, **kwargs),
        seed=seed, duration_ms=10, model_id="test/model",
    )


def _metadata(path):
    with Image.open(path) as image:
        return json.loads(image.info["auradream"])


def _rewrite_metadata(path, meta):
    path.write_bytes(reindex._with_auradream_metadata(path.read_bytes(), meta))


@pytest.mark.parametrize("legacy", [False, True])
def test_png_import_remaps_proven_sources_and_drops_legacy_ids(store, monkeypatch, tmp_path, legacy):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    unrelated = _save("unrelated", 9)
    other = dataclasses.replace(mine, db_path=tmp_path / "other.db", outputs_dir=tmp_path / "other")
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    # The dependent name sorts first within the same timestamp. Recovery must
    # visit its named source before assigning the dependent's gallery id.
    source = _save("source", 9)
    conditioned = _save("reference", 1, reference=[source.id])
    refined = _save("refined", 2, refine=source.id, strength=0.3)
    assert source.id == unrelated.id
    if legacy:
        for record in (conditioned, refined):
            path = other.outputs_dir / record.filename
            meta = _metadata(path)
            meta.pop("reference_sources")
            meta.pop("refine_source")
            _rewrite_metadata(path, meta)
    before = {p.name: p.read_bytes() for p in other.outputs_dir.glob("*.png")}
    monkeypatch.setattr(storage, "settings", mine)

    counts = reindex.reindex(other.outputs_dir)

    by_prompt = {r.prompt: r for r in store.list_images()[0]}
    assert counts["indexed"] == 3
    assert counts["unresolved_references"] == (2 if legacy else 0)
    imported = by_prompt["source"]
    assert imported.id != source.id
    assert by_prompt["reference"].reference == (None if legacy else [imported.id])
    assert by_prompt["refined"].refine == (None if legacy else imported.id)
    assert by_prompt["refined"].strength == (None if legacy else 0.3)
    for record in (conditioned, refined):
        meta = _metadata(mine.outputs_dir / record.filename)
        rebuilt = by_prompt[record.prompt]
        assert (meta["reference"], meta["refine"], meta["strength"]) == (
            rebuilt.reference, rebuilt.refine, rebuilt.strength,
        )
    assert {p.name: p.read_bytes() for p in other.outputs_dir.glob("*.png")} == before
    assert reindex.reindex(other.outputs_dir)["present"] == 3


def test_rebuild_after_deleted_ids_remaps_png_metadata_too(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    removed = _save("removed", 1)
    source = _save("source", 9)
    refined = _save("refined", 2, refine=source.id, strength=0.3)
    conditioned = _save("reference", 3, reference=[source.id])
    (store.settings.outputs_dir / removed.filename).unlink()
    store.settings.db_path.unlink()
    before = {p.name: p.read_bytes() for p in store.settings.outputs_dir.glob("*.png")}

    dry = reindex.reindex(store.settings.outputs_dir, dry_run=True)
    assert dry["indexed"] == 3 and dry["unresolved_references"] == 0
    assert not store.settings.db_path.exists()
    assert {p.name: p.read_bytes() for p in store.settings.outputs_dir.glob("*.png")} == before
    counts = reindex.reindex(store.settings.outputs_dir)

    rebuilt = {r.filename: r for r in store.list_images()[0]}
    new_source = rebuilt[source.filename]
    assert new_source.id != source.id
    assert counts["unresolved_references"] == 0
    assert rebuilt[refined.filename].refine == new_source.id
    assert rebuilt[conditioned.filename].reference == [new_source.id]
    assert _metadata(store.settings.outputs_dir / refined.filename)["refine"] == new_source.id
    assert _metadata(store.settings.outputs_dir / conditioned.filename)["reference"] == [new_source.id]
    assert (store.settings.outputs_dir / source.filename).read_bytes() == before[source.filename]


@pytest.mark.parametrize("broken", ["missing", "hash", "size", "filename", "pixels"])
def test_unproven_source_is_dropped_instead_of_binding_a_valid_id(store, monkeypatch, broken):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = _save("source", 9)
    refined = _save("refined", 1, refine=source.id, strength=0.3)
    path = store.settings.outputs_dir / refined.filename
    meta = _metadata(path)
    if broken == "missing":
        (store.settings.outputs_dir / source.filename).unlink()
    elif broken == "pixels":
        # The row still claims the original hash, but this is what a pass
        # would actually load through the otherwise-valid gallery id.
        Image.new("RGB", (32, 24), "green").save(store.settings.outputs_dir / source.filename)
    elif broken == "hash":
        meta["refine_source"]["hash"] = "0" * 16
    elif broken == "size":
        meta["refine_source"]["width"] += 1
    else:
        meta["refine_source"]["filename"] = "../outside.png"
    _rewrite_metadata(path, meta)
    with store._session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (refined.id,))

    counts = reindex.reindex(store.settings.outputs_dir)

    rebuilt = next(r for r in store.list_images()[0] if r.filename == refined.filename)
    assert counts["unresolved_references"] == 1
    assert rebuilt.refine is None and rebuilt.strength is None
    assert _metadata(path)["refine"] is None


def test_partial_same_gallery_reindex_preserves_verified_links(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = _save("source", 9)
    refined = _save("refined", 1, refine=source.id, strength=0.3)
    with store._session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (refined.id,))

    counts = reindex.reindex(store.settings.outputs_dir)

    rebuilt = next(r for r in store.list_images()[0] if r.filename == refined.filename)
    assert counts["unresolved_references"] == 0
    assert (rebuilt.refine, rebuilt.strength) == (source.id, 0.3)


def test_database_merge_dry_run_with_sources_does_not_create_destination_db(store, monkeypatch, tmp_path):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    other = dataclasses.replace(mine, db_path=tmp_path / "other.db", outputs_dir=tmp_path / "other")
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    source = _save("source", 9)
    _save("reference", 1, reference=[source.id])
    _save("refined", 2, refine=source.id, strength=0.3)
    monkeypatch.setattr(storage, "settings", mine)
    mine.db_path.unlink()
    before = set(mine.outputs_dir.iterdir())

    counts = reindex.reindex(other.outputs_dir, database=other.db_path, dry_run=True)

    assert counts["indexed"] == 3 and counts["unresolved_references"] == 0
    assert not mine.db_path.exists()
    assert set(mine.outputs_dir.iterdir()) == before


@pytest.mark.parametrize("database", [False, True])
def test_legacy_destination_preview_keeps_closed_database_and_png_bytes(store, monkeypatch, tmp_path, database):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    other = dataclasses.replace(mine, db_path=tmp_path / "other.db", outputs_dir=tmp_path / "other")
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    source = _save("source", 9)
    monkeypatch.setattr(storage, "settings", mine)
    merge_db = other.db_path if database else None
    assert reindex.reindex(other.outputs_dir, database=merge_db)["indexed"] == 1
    retained, = store.list_images()[0]
    monkeypatch.setattr(storage, "settings", other)
    child = _save("reference", 1, reference=[source.id])
    monkeypatch.setattr(storage, "settings", mine)

    with closing(sqlite3.connect(mine.db_path)) as conn:
        with conn:
            conn.execute("ALTER TABLE images DROP COLUMN detailer")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        before_schema = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
        before_rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    preview = reindex.reindex(other.outputs_dir, database=merge_db, dry_run=True)

    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    with closing(sqlite3.connect(mine.db_path)) as conn:
        assert conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == before_schema
        assert conn.execute("SELECT * FROM images ORDER BY id").fetchall() == before_rows
    assert preview == reindex.reindex(other.outputs_dir, database=merge_db)
    assert (preview["indexed"], preview["present"], preview["unresolved_references"]) == (1, 1, 0)
    records = {record.filename: record for record in store.list_images()[0]}
    assert records[child.filename].reference == [retained.id]


@pytest.mark.parametrize("database", [False, True])
def test_preview_reads_wal_from_a_database_copy_without_creating_a_sidecar(store, monkeypatch, tmp_path, database):
    mine = store.settings
    snapshot = tmp_path / "snapshot.db"
    original_wal = mine.db_path.with_name(mine.db_path.name + "-wal")
    snapshot_wal = snapshot.with_name(snapshot.name + "-wal")
    with closing(sqlite3.connect(mine.db_path)) as live:
        live.execute("PRAGMA wal_autocheckpoint=0")
        assert live.execute("SELECT COUNT(*) FROM images").fetchone() == (0,)
        _save("source", 9)
        copyfile(mine.db_path, snapshot)
        copyfile(original_wal, snapshot_wal)
    monkeypatch.setattr(reindex, "settings", dataclasses.replace(mine, db_path=snapshot))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    counts = reindex.reindex(mine.outputs_dir, database=snapshot if database else None, dry_run=True)

    assert (counts["indexed"], counts["present"]) == (0, 1)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("database", [False, True])
@pytest.mark.parametrize("damage", ["missing", "unreadable", "size", "pixels", "intact"])
def test_same_name_source_requires_decoded_identity(store, monkeypatch, tmp_path, database, damage):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    # Force foreign IDs to change so a valid repeat compares files with
    # different metadata bytes. This also distinguishes local and foreign IDs.
    _save("unrelated", 33)
    other = dataclasses.replace(mine, db_path=tmp_path / "other.db", outputs_dir=tmp_path / "other")
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    source = _save("source", 9)
    monkeypatch.setattr(storage, "settings", mine)
    merge_db = other.db_path if database else None
    assert reindex.reindex(other.outputs_dir, database=merge_db)["indexed"] == 1
    retained = next(row for row in store.list_images()[0] if row.filename == source.filename)
    target = mine.outputs_dir / source.filename
    if damage == "missing":
        target.unlink()
    elif damage == "unreadable":
        target.write_bytes(b"broken PNG")
    elif damage == "size":
        Image.new("RGB", (24, 32), "red").save(target)
    elif damage == "pixels":
        Image.new("RGB", (32, 24), "green").save(target)
    before = target.read_bytes() if target.exists() else None
    monkeypatch.setattr(storage, "settings", other)
    referenced = _save("reference", 1, reference=[source.id])
    refined = _save("refined", 2, refine=source.id, strength=0.3)
    monkeypatch.setattr(storage, "settings", mine)

    preview_out, import_out = io.StringIO(), io.StringIO()
    preview = reindex.reindex(other.outputs_dir, database=merge_db, dry_run=True, out=preview_out)
    assert not (mine.outputs_dir / referenced.filename).exists()
    assert preview == reindex.reindex(other.outputs_dir, database=merge_db, out=import_out)
    healthy = damage == "intact"
    assert (preview["indexed"], preview["present"], preview["conflicts"], preview["unresolved_references"]) == (
        2, int(healthy), int(not healthy), 0 if healthy else 2,
    )
    if not healthy:
        for message in (preview_out, import_out):
            assert f"skip  {source.filename}: cannot use the retained image" in message.getvalue()
    records = {row.filename: row for row in store.list_images()[0]}
    assert records[source.filename].id == retained.id != source.id
    assert records[referenced.filename].reference == ([retained.id] if healthy else None)
    assert records[refined.filename].refine == (retained.id if healthy else None)
    assert records[refined.filename].strength == (0.3 if healthy else None)
    assert (target.read_bytes() if target.exists() else None) == before
    if healthy:
        for child in (referenced, refined):
            assert (mine.outputs_dir / child.filename).read_bytes() != (other.outputs_dir / child.filename).read_bytes()
        repeat = reindex.reindex(other.outputs_dir, database=merge_db)
        assert (repeat["indexed"], repeat["present"], repeat["conflicts"]) == (0, 3, 0)
        assert {row.filename: row.id for row in store.list_images()[0]} == {
            filename: row.id for filename, row in records.items()
        }
