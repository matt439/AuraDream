"""Upgrade and recover real historical writer output under the current application."""
from __future__ import annotations

import dataclasses
import sqlite3
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest

from backend import legacy_names, reindex, storage
from tests.bundle_helpers import bundle_snapshot, save_source
from tests.historical_gallery_helpers import ARCHIVES, assert_gallery, filename


def _unpack(version, directory, *, database=True):
    """Restore the writer's original layout from these small checked-in backups."""
    images = directory / "outputs"
    images.mkdir(parents=True)
    name = "local-diffusion.db" if version == "legacy-v1" else "auradream.db"
    db = directory / name
    with ZipFile(ARCHIVES / f"{version}.zip") as archive:
        if database:
            db.write_bytes(archive.read("database.sqlite3"))
        for member in archive.namelist():
            path = Path(member)
            if path.parts[0] == "images":
                target = images.joinpath(*path.parts[1:])
                target.parent.mkdir(exist_ok=True, parents=True)
                target.write_bytes(archive.read(member))
    return images, db


def _files(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("version", ["legacy-v1", "edit-v2"])
def test_open_historical_gallery_in_place(tmp_path, monkeypatch, version):
    images, db = _unpack(version, tmp_path)
    # Mirrors configuration on an existing installation, including the old database name.
    selected = legacy_names.default_db_path(tmp_path)
    assert selected == db
    monkeypatch.setattr(storage, "settings", dataclasses.replace(storage.settings, db_path=selected, outputs_dir=images))
    # An ordinary SQLite read must not migrate the evidence before init_db sees it.
    with closing(sqlite3.connect(db.as_uri() + "?immutable=1", uri=True)) as conn:
        old_columns = [row[1] for row in conn.execute("PRAGMA table_info(images)")]
        original = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    assert "tiled_refine" not in old_columns
    assert ("edit" in old_columns) == (version == "edit-v2")
    before = {path.name: path.read_bytes() for path in images.glob("*.png")}

    storage.init_db()
    chunk = "local-diffusion" if version == "legacy-v1" else "auradream"
    records = assert_gallery(version, 1, expected_chunk=chunk)
    assert [record.id for record in records.values()] == list(range(1, len(original) + 1))
    with closing(sqlite3.connect(db)) as conn:
        migrated = [row[1] for row in conn.execute("PRAGMA table_info(images)")]
        assert {"tiled_refine", "detailer", "crop_anchor"} <= set(migrated)
        projection = ", ".join(f'"{column}"' for column in old_columns)
        assert conn.execute(f"SELECT {projection} FROM images ORDER BY id").fetchall() == original
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert {path.name: path.read_bytes() for path in images.glob("*.png")} == before
    after = bundle_snapshot(storage)
    storage.init_db()
    assert bundle_snapshot(storage) == after
    assert sorted(tmp_path.glob("*.db")) == [db]


@pytest.mark.parametrize("version,amount", [("legacy-v1", 3), ("edit-v2", 2)])
@pytest.mark.parametrize("database", [True, False], ids=["with-database", "pngs-only"])
def test_reindex_historical_gallery_into_occupied_ids(store, tmp_path, monkeypatch, capsys, version, amount, database):
    original = tmp_path / "historical"
    images, db = _unpack(version, original, database=database)
    before = _files(original)
    monkeypatch.setattr(reindex, "settings", store.settings)
    collision = save_source("occupied source id", seed=91)
    assert collision.id == 1
    collision_bytes = (store.settings.outputs_dir / collision.filename).read_bytes()
    args = ["--from", str(images)]
    if database:
        args.extend(["--database", str(db)])
    verb = "Imported" if database else "Indexed"

    assert reindex.main(args) == 0
    assert f"{verb} {amount} file(s); 0 already in the index." in capsys.readouterr().out
    assert_gallery(version, 2, png_only=not database)
    assert storage.get_image(1) == collision
    assert (store.settings.outputs_dir / collision.filename).read_bytes() == collision_bytes
    after = bundle_snapshot(store)
    assert len(after[0]) == amount + 1

    assert reindex.main(args) == 0
    assert f"{verb} 0 file(s); {amount} already in the index." in capsys.readouterr().out
    assert bundle_snapshot(store) == after
    assert _files(original) == before


@pytest.mark.parametrize("database", [True, False], ids=["with-database", "pngs-only"])
def test_reindex_historical_edit_missing_its_source(store, tmp_path, monkeypatch, capsys, database):
    images, db = _unpack("edit-v2", tmp_path / "historical", database=database)
    (images / filename(201)).unlink()
    monkeypatch.setattr(reindex, "settings", store.settings)
    collision = save_source("occupied source id", seed=99)
    assert collision.id == 1
    args = ["--from", str(images)]
    if database:
        args.extend(["--database", str(db)])

    assert reindex.main(args) == 0
    assert "1 unresolved instruction-edit source(s)" in capsys.readouterr().out
    records = assert_gallery("edit-v2", None, orphan=True, png_only=not database)
    assert records[202].id == 2
    assert storage.get_image(1) == collision
    after = bundle_snapshot(store)
    assert reindex.main(args) == 0
    assert bundle_snapshot(store) == after
