"""Bundles that keep the gallery: a verified backup, and a bundle of chosen images."""
from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundle_archive, bundles, gallery_delete
from backend.bundle_common import BundleError
from tests.bundle_helpers import _destination, _save, _snapshot


def _members(archive) -> set[str]:
    with ZipFile(archive["path"]) as bundle:
        return set(bundle.namelist())


def test_a_backup_keeps_every_file_and_row_and_imports_like_any_bundle(store, monkeypatch, tmp_path):
    source = _save("source", 11)
    refined = _save("refined", 12, refine=source.id, strength=0.35)
    store.set_favorite(refined.id, True)
    orphan = store.settings.outputs_dir / "unindexed.png"
    Image.new("RGB", (8, 8), "red").save(orphan)
    before = _snapshot(store)

    # No typed count: the count gate is the clear's, and a backup clears nothing.
    archive = bundles.export_bundle(keep_gallery=True)
    assert archive["kept"] is True and archive["cleared"] is False
    assert archive["deleted_ids"] == [] and archive["retained"] == [] and "cleanup_error" not in archive
    assert (archive["records"], archive["images"], archive["links_outside"]) == (2, 3, 0)
    assert _snapshot(store) == before
    assert {"images/unindexed.png", "images/" + source.filename, "images/" + refined.filename} <= _members(archive)

    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["unresolved_references"]) == (2, 0)
    rows = {row.filename: row for row in store.list_images()[0]}
    assert rows[refined.filename].favorite and rows[refined.filename].refine == rows[source.filename].id


def test_a_backup_ignores_a_stale_count_but_keeps_the_verified_read_back(store, monkeypatch):
    _save()
    assert bundles.export_bundle(expected=99, keep_gallery=True)["kept"] is True
    before = _snapshot(store)
    published = {entry["name"] for entry in bundles.list_bundles()}

    def damaged(*args, **kwargs):
        raise BundleError("the bundle checksum failed for images/x.png")

    monkeypatch.setattr(bundles, "_verify_archive", damaged)
    with pytest.raises(BundleError, match="checksum failed"):
        bundles.export_bundle(keep_gallery=True)
    assert {entry["name"] for entry in bundles.list_bundles()} == published
    assert not list(bundles._directory().glob("*.part"))
    assert _snapshot(store) == before


def test_a_bundle_of_chosen_images_holds_only_them_and_says_which_sources_it_lacks(store, monkeypatch, tmp_path):
    source = _save("a secret lighthouse", 11)
    refined = _save("refined harbour", 12, refine=source.id, strength=0.35)
    conditioned = _save("conditioned harbour", 13, reference=[source.id, refined.id])
    other = _save("unrelated meadow", 14)
    gallery_delete.bin_images([other.id])
    (store.settings.outputs_dir / "unindexed.png").write_bytes(b"not carried")
    before = _snapshot(store)

    archive = bundles.export_bundle(keep_gallery=True, ids=[refined.id, conditioned.id, 404])
    assert archive["kept"] is True and archive["deleted_ids"] == []
    assert (archive["records"], archive["images"], archive["binned"]) == (2, 2, 0)
    assert archive["left_out"] == [404]
    # refined's source and one of conditioned's two references are outside.
    assert archive["links_outside"] == 2
    assert _snapshot(store) == before
    stems = [Path(record.filename).stem for record in (refined, conditioned)]
    assert _members(archive) == {
        "database.sqlite3", "manifest.json",
        *(f"images/{stem}.png" for stem in stems), *(f"images/thumbnails/{stem}.webp" for stem in stems),
    }
    with ZipFile(archive["path"]) as bundle:
        database = bundle.read("database.sqlite3")
    # Neither the rows left out nor their search index travel, not even in
    # free pages: a bundle given to someone carries no other prompt.
    assert b"lighthouse" not in database and b"meadow" not in database

    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["unresolved_references"]) == (2, 2)
    rows = {row.filename: row for row in store.list_images()[0]}
    assert rows[refined.filename].refine is None
    assert rows[conditioned.filename].reference == [rows[refined.filename].id]
    assert store.list_images(search="harbour")[1] == 2


def test_a_chosen_bundle_carries_a_binned_row_and_is_never_a_clear(store):
    kept = _save("kept", 1)
    gallery_delete.bin_images([kept.id])
    with pytest.raises(BundleError, match="only a bundle of the whole gallery"):
        bundles.export_bundle(expected=1, ids=[kept.id])
    archive = bundles.export_bundle(keep_gallery=True, ids=[kept.id])
    assert (archive["records"], archive["binned"]) == (1, 1)
    with pytest.raises(BundleError, match="none of the chosen images"):
        bundles.export_bundle(keep_gallery=True, ids=[999])
    assert store.list_images(binned=True)[1] == 1


def test_a_missing_file_stops_a_chosen_bundle_only_when_it_is_chosen(store):
    chosen = _save("chosen", 1)
    lost = _save("lost", 2)
    (store.settings.outputs_dir / lost.filename).unlink()
    with pytest.raises(bundles.MissingImage):
        bundles.preview()
    assert bundles.export_bundle(keep_gallery=True, ids=[chosen.id])["records"] == 1
    with pytest.raises(bundles.MissingImage, match=f"image #{lost.id} "):
        bundles.export_bundle(keep_gallery=True, ids=[chosen.id, lost.id])


def test_a_chosen_bundle_needs_room_for_the_log_and_the_vacuumed_copy(store, monkeypatch):
    # One image, so the whole gallery and the selection archive the same
    # files: only the selection's second copy of the database tells them apart.
    saved = _save()
    db = store.settings.db_path
    wal = db.with_name(db.name + "-wal")
    database = db.stat().st_size + (wal.stat().st_size if wal.exists() else 0)
    files = sum(p.stat().st_size for _, p in bundles._inventory())
    room = shutil.disk_usage(store.settings.db_path.parent)._replace(free=files + 2 * database + 1)
    monkeypatch.setattr(bundle_archive.shutil, "disk_usage", lambda where: room)
    with pytest.raises(BundleError, match="not enough free space to write the bundle"):
        bundles.export_bundle(keep_gallery=True, ids=[saved.id])
    assert bundles.export_bundle(keep_gallery=True)["records"] == 1


@pytest.mark.parametrize("keep", [True, False])
def test_an_export_without_room_for_its_files_is_refused_before_it_writes(store, monkeypatch, keep):
    _save()
    before = _snapshot(store)
    full = shutil.disk_usage(store.settings.db_path.parent)._replace(free=1024)
    monkeypatch.setattr(bundle_archive.shutil, "disk_usage", lambda where: full)
    with pytest.raises(BundleError, match="not enough free space to write the bundle"):
        bundles.export_bundle(expected=1, keep_gallery=keep)
    assert _snapshot(store) == before
    assert bundles.list_bundles() == [] and not list(bundles._directory().iterdir())
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
