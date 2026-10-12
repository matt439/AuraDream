"""Verified gallery archives: export, verification and safe clearing."""
from __future__ import annotations

import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundles
from tests.bundle_helpers import _save, _export, _destination, _snapshot, _metadata, _locked_database


def test_export_round_trip_merges_fields_stars_timings_and_conditioning_ids(store, monkeypatch, tmp_path):
    source = _save("source", 11)
    reference = _save(
        "conditioned", 12, reference=[source.id], negative_prompt="mist",
        guidance_scale=3.5, true_cfg=4.5, mu=0.9, lora="portrait", lora_scale=0.7,
        variables={"season": "winter"},
    )
    refined = _save("refined", 13, refine=source.id, strength=0.35)
    store.set_favorite(reference.id, True)
    before = {row.filename: row for row in store.list_images()[0]}
    archive = _export()

    assert archive["images"] == archive["records"] == 3
    assert archive["cleared"] is True and archive["retained"] == []
    assert set(archive["deleted_ids"]) == {source.id, reference.id, refined.id}
    assert store.list_images()[1] == 0
    assert not list(store.settings.outputs_dir.rglob("*.png"))
    assert not list(store.settings.outputs_dir.rglob("*.webp"))
    bundle = Path(archive["path"])
    assert bundle.is_file() and bundle.stat().st_size == archive["bytes"]
    assert bundles.bundle_path(archive["name"]) == bundle
    assert archive["name"] in {entry["name"] for entry in bundles.list_bundles()}

    dest = _destination(store, monkeypatch, tmp_path)
    existing = [_save("existing one", 100), _save("existing two", 101)]
    existing_files = {row.filename: (dest.outputs_dir / row.filename).read_bytes() for row in existing}
    imported = bundles.import_bundle(bundle)
    assert imported["imported"] == 3 and imported["present"] == 0
    rows = {row.filename: row for row in store.list_images()[0]}
    assert len(rows) == 5
    mapped_source = rows[source.filename].id
    assert mapped_source != source.id
    assert rows[reference.filename].reference == [mapped_source]
    assert rows[refined.filename].refine == mapped_source
    for filename, original in before.items():
        actual = rows[filename]
        ignored = {"id", "url", "thumb_url", "reference", "refine"}
        assert actual.model_dump(exclude=ignored) == original.model_dump(exclude=ignored)
        assert actual.thumb_url and (dest.outputs_dir / "thumbnails" / Path(filename).with_suffix(".webp")).is_file()
    assert _metadata(dest.outputs_dir / reference.filename)["reference"] == [mapped_source]
    assert _metadata(dest.outputs_dir / refined.filename)["refine"] == mapped_source
    assert store.list_images(search="conditioned")[1] == 1  # FTS comes across with the rows.
    assert {name: (dest.outputs_dir / name).read_bytes() for name in existing_files} == existing_files

    # A repeated import must also preserve choices made since the first import.
    store.set_favorite(rows[reference.filename].id, False)
    snapshot = _snapshot(store)
    again = bundles.import_bundle(bundle)
    assert again["imported"] == 0 and again["present"] == 3
    assert _snapshot(store) == snapshot


def test_a_database_replacement_refused_after_files_went_still_reports_what_was_deleted(store, monkeypatch):
    first = _save("first", 1)
    second = _save("second", 2)
    monkeypatch.setattr(bundles, "_replace_database", _locked_database)
    archive = _export()
    assert archive["cleared"] is False and "locked" in archive["cleanup_error"]
    assert Path(archive["path"]).is_file()
    assert str(store.settings.db_path.resolve()) in archive["retained"]
    # The files and rows really went; the browser must still drop their ids.
    assert sorted(archive["deleted_ids"]) == sorted([first.id, second.id])
    assert store.get_image(first.id) is None and store.get_image(second.id) is None
    assert not list(store.settings.outputs_dir.glob("*.png"))


def test_export_preserves_unindexed_images_and_thumbnails_byte_for_byte(store, monkeypatch, tmp_path):
    _save()
    orphan = store.settings.outputs_dir / "unindexed.png"
    Image.new("RGB", (8, 8), "red").save(orphan)
    orphan_thumb = store.settings.outputs_dir / "thumbnails" / "unindexed.webp"
    orphan_thumb.write_bytes(b"an orphan thumbnail must survive unchanged")
    expected = {"unindexed.png": orphan.read_bytes(), "thumbnails/unindexed.webp": orphan_thumb.read_bytes()}
    preview = bundles.preview()
    assert (preview["records"], preview["images"], preview["thumbnails"]) == (1, 2, 2)
    assert preview["bytes"] >= sum(map(len, expected.values()))
    archive = _export()
    assert not orphan.exists() and not orphan_thumb.exists()
    _destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(Path(archive["path"]))
    for name, data in expected.items():
        assert (store.settings.outputs_dir / name).read_bytes() == data
    assert store.list_images()[1] == 1  # An orphan must not invent a gallery row.


def test_export_refuses_a_stale_preview_without_touching_gallery(store):
    _save(seed=1)
    expected = bundles.preview()["records"]
    _save(seed=2)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleChanged):
        bundles.export_bundle(expected=expected)
    assert _snapshot(store) == snapshot
    assert bundles.list_bundles() == []


@pytest.mark.parametrize("problem", ["missing", "partial"])
def test_export_refuses_missing_tracked_images_or_unfinished_writes(store, problem):
    row = _save()
    if problem == "missing":
        (store.settings.outputs_dir / row.filename).unlink()
    else:
        (store.settings.outputs_dir / "unfinished.png.part").write_bytes(b"not committed")
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError):
        _export()
    assert _snapshot(store) == snapshot
    assert bundles.list_bundles() == []


@pytest.mark.parametrize("problem", ["corrupt", "dimensions"])
def test_unrestorable_tracked_image_prevents_clearing_any_other_images_or_rows(store, problem):
    valid = _save("valid image", 1)
    damaged = _save("damaged image", 2)
    path = store.settings.outputs_dir / damaged.filename
    if problem == "corrupt":
        path.write_bytes(b"this tracked file is no longer a PNG")
    else:
        Image.new("RGB", (32, 24), "red").save(path)
    snapshot = _snapshot(store)

    try:
        archive = _export()
    except bundles.BundleError:
        pass
    else:
        assert archive["cleared"] is False
        assert archive["deleted_ids"] == []
    assert _snapshot(store) == snapshot
    assert store.get_image(valid.id) is not None
    assert store.get_image(damaged.id) is not None


def test_failed_archive_write_keeps_every_source_file_and_row(store, monkeypatch):
    _save()
    snapshot = _snapshot(store)

    real_open = ZipFile.open

    def fail(self, name, mode="r", *args, **kwargs):
        if mode == "w":
            raise OSError("archive disk full")
        return real_open(self, name, mode, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", fail)
    with pytest.raises((OSError, bundles.BundleError), match="disk full"):
        _export()
    assert _snapshot(store) == snapshot
    assert bundles.list_bundles() == []


def test_failed_archive_verification_keeps_every_source_file_and_row(store, monkeypatch):
    _save()
    snapshot = _snapshot(store)

    def fail(*args, **kwargs):
        raise bundles.BundleError("verification failed")

    monkeypatch.setattr(bundles, "_verify_archive", fail)
    with pytest.raises(bundles.BundleError, match="verification failed"):
        _export()
    assert _snapshot(store) == snapshot
    assert bundles.list_bundles() == []


def test_clear_compacts_old_data_and_preserves_highwater_after_highest_row_was_deleted(store):
    marker = "archived-gallery-prompt "
    first = _save(prompt=marker * 80, seed=1)
    for seed in range(2, 13):
        highest = _save(prompt=marker * 80, seed=seed)
    store.delete_image(highest.id)
    before_bytes = store.settings.db_path.stat().st_size
    _export()
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0
        assert conn.execute("SELECT seq FROM sqlite_sequence WHERE name='images'").fetchone()[0] == highest.id
    assert store.settings.db_path.stat().st_size < before_bytes
    assert marker.encode() not in store.settings.db_path.read_bytes()
    next_row = _save(seed=3)
    assert next_row.id > highest.id > first.id


@pytest.mark.parametrize("change", ["changed", "same-size", "added"])
def test_a_file_changed_or_added_after_verification_keeps_the_durable_bundle_without_clearing(store, monkeypatch, change):
    first = _save("first", 1)
    _save("second", 2)
    path = store.settings.outputs_dir / first.filename
    original = path.read_bytes()
    real_verify = bundles._verify_archive
    snapshots = []

    def verify_then_change(*args, **kwargs):
        verified = real_verify(*args, **kwargs)
        if change == "changed":
            path.write_bytes(original + b"external edit after archive verification")
        elif change == "same-size":
            # Same size and mtime, one byte different: only the hash can tell.
            st = path.stat()
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        else:
            Image.new("RGB", (8, 8), "green").save(store.settings.outputs_dir / "new-external.png")
        snapshots.append(_snapshot(store))
        return verified

    monkeypatch.setattr(bundles, "_verify_archive", verify_then_change)
    archive = _export()
    assert archive["cleared"] is False
    assert archive["deleted_ids"] == []
    assert archive["retained"] and archive["cleanup_error"]
    assert _snapshot(store) == snapshots[-1]
    assert Path(archive["path"]).is_file()
    with ZipFile(archive["path"]) as bundle:
        assert bundle.read("images/" + first.filename) == original


def test_bundle_database_contains_committed_wal_rows(store, monkeypatch, tmp_path):
    # Keep the connection alive so closing storage's connection does not
    # checkpoint away the WAL that a raw copy of the .db would miss.
    with closing(sqlite3.connect(store.settings.db_path)) as keeper:
        keeper.execute("PRAGMA journal_mode=WAL")
        keeper.execute("PRAGMA wal_autocheckpoint=0")
        row = _save()
        keeper.execute("UPDATE images SET favorite=1, duration_ms=8765 WHERE id=?", (row.id,))
        keeper.commit()
        archive = _export()
    _destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(Path(archive["path"]))
    restored = store.list_images()[0][0]
    assert restored.favorite and restored.duration_ms == 8765


def test_locked_image_is_reported_and_keeps_its_row_after_a_verified_export(store, monkeypatch):
    locked = _save("locked", 1)
    removed = _save("removed", 2)
    path = store.settings.outputs_dir / locked.filename
    original = path.read_bytes()
    real_unlink = Path.unlink

    def refuse_one(self, *args, **kwargs):
        if self == path:
            raise PermissionError("file is open in another application")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_one)
    archive = _export()
    assert archive["cleared"] is False
    assert str(path) in archive["retained"]
    assert archive["deleted_ids"] == [removed.id]
    assert path.read_bytes() == original
    assert store.get_image(locked.id) is not None
    assert store.get_image(removed.id) is None
    with ZipFile(archive["path"]) as bundle:
        assert bundle.read("images/" + locked.filename) == original
        assert "images/" + removed.filename in bundle.namelist()


def test_export_reservation_holds_until_clear_so_a_new_image_cannot_be_lost(store, monkeypatch):
    # The save encodes its `.png.part` before it waits for the reservation,
    # so the export's post-verification inventory finds an incomplete write
    # and clears nothing; the save then lands once the export lets go.
    original = _save()
    verifying = threading.Event()
    release = threading.Event()
    save_started = threading.Event()
    saved = threading.Event()
    real_verify = bundles._verify_archive

    def held_verification(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        verifying.set()
        assert release.wait(10), "test did not release export"
        return result

    def save_after_export_started():
        save_started.set()
        try:
            return _save("new image", 2)
        finally:
            saved.set()

    monkeypatch.setattr(bundles, "_verify_archive", held_verification)
    with ThreadPoolExecutor(max_workers=2) as pool:
        export = pool.submit(bundles.export_bundle, expected=1)
        try:
            assert verifying.wait(10), "export did not reach verification"
            save = pool.submit(save_after_export_started)
            assert save_started.wait(10)
            assert not saved.wait(0.1), "save entered while export owned the gallery"
        finally:
            release.set()
        result = export.result(timeout=10)
        new = save.result(timeout=10)
    assert result["deleted_ids"] == [] and result["cleared"] is False
    assert "image write still in progress" in result["cleanup_error"]
    assert Path(result["path"]).is_file()
    assert new.id > original.id
    assert store.list_images()[1] == 2
    assert (store.settings.outputs_dir / original.filename).is_file()
    assert (store.settings.outputs_dir / new.filename).is_file()


def test_bundle_names_carry_the_local_wall_clock_time(store, monkeypatch):
    # A clock whose local time and UTC differ by hours: a name stamped from
    # UTC reads 1506, the local one 0506.
    import datetime as real

    class Clock(real.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return real.datetime(2026, 3, 4, 5, 6, 7)
            return real.datetime(2026, 3, 4, 15, 6, 7, tzinfo=real.timezone.utc).astimezone(tz)

    monkeypatch.setattr(bundles, "datetime", Clock)
    _save("named", 1)
    name = bundles.export_bundle(expected=1)["name"]
    assert name.startswith("local-diffusion-20260304-050607-") and name.endswith(".zip")
    assert len(name) == len("local-diffusion-20260304-050607-") + 8 + len(".zip")


@pytest.mark.parametrize("name",["../outside.zip", "..\\outside.zip", "/outside.zip", "C:/outside.zip"])
def test_download_names_cannot_escape_the_bundle_directory(store, name):
    with pytest.raises(bundles.BundleError):
        bundles.bundle_path(name)


def test_a_locked_database_stops_the_clear_before_any_file_is_removed(store, monkeypatch):
    first = _save("first", 1)
    second = _save("second", 2)
    snapshot = _snapshot(store)
    real_connect = sqlite3.connect
    # Another program's open write transaction; storage's 15 s wait shortened.
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: real_connect(*a, **{**k, "timeout": 0.2}))
    with closing(real_connect(store.settings.db_path, isolation_level=None)) as other:
        other.execute("BEGIN IMMEDIATE")
        archive = _export()
        other.execute("ROLLBACK")
    assert "locked" in archive["cleanup_error"]
    assert archive["deleted_ids"] == [] and archive["cleared"] is False
    assert Path(archive["path"]).is_file()
    assert _snapshot(store) == snapshot
    assert store.get_image(first.id) and store.get_image(second.id)
