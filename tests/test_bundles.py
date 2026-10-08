"""Verified gallery archives: preservation, safe clearing and atomic merges."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundles, gallery_lock, storage
from backend.schemas import GenerateRequest


def _save(prompt="source", seed=1, **kwargs):
    variables = kwargs.pop("variables", None)
    return storage.save_image(
        Image.new("RGB", (256, 256), (seed % 256, 24, 80)),
        GenerateRequest(prompt=prompt, width=256, height=256, steps=28, **kwargs),
        seed=seed, duration_ms=1234 + seed, model_id="test/model",
        job_id="job-one", run_id="run-one", quantization="fp8", variables=variables,
    )


def _export():
    return bundles.export_bundle(expected=bundles.preview()["records"])


def _destination(store, monkeypatch, tmp_path):
    settings = dataclasses.replace(
        store.settings, db_path=tmp_path / "destination.db", outputs_dir=tmp_path / "destination",
    )
    monkeypatch.setattr(storage, "settings", settings)
    storage.init_db()
    return settings


def _snapshot(store):
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    files = {
        path.relative_to(store.settings.outputs_dir).as_posix(): path.read_bytes()
        for path in store.settings.outputs_dir.rglob("*") if path.is_file()
    }
    return rows, files


def _metadata(path):
    with Image.open(path) as image:
        return json.loads(image.info["local-diffusion"])


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


def test_import_publish_failure_rolls_back_rows_and_files_already_added(store, monkeypatch, tmp_path):
    _save("first", 1)
    _save("second", 2)
    archive = _export()
    dest = _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    real_open = Path.open
    published = 0

    def fail_second_image(self, mode="r", *args, **kwargs):
        nonlocal published
        if mode == "xb" and self.parent == dest.outputs_dir and self.suffix == ".png":
            published += 1
            if published == 2:
                raise OSError("destination disk full")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_second_image)
    with pytest.raises((OSError, bundles.BundleError), match="disk full"):
        bundles.import_bundle(Path(archive["path"]))
    assert published == 2
    assert _snapshot(store) == snapshot


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


@pytest.mark.parametrize("change", ["changed", "added"])
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


def test_invalid_zip_does_not_mutate_the_destination(store, tmp_path):
    _save("existing")
    snapshot = _snapshot(store)
    bundle = tmp_path / "broken.zip"
    bundle.write_bytes(b"not a ZIP archive")
    with pytest.raises(bundles.BundleError):
        bundles.import_bundle(bundle)
    assert _snapshot(store) == snapshot


def test_filename_conflict_refuses_the_whole_merge_before_adding_any_rows(store, monkeypatch, tmp_path):
    _save("would import first", 1)
    conflicting = _save("would conflict last", 2)
    archive = _export()
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 9)
    conflict_path = store.settings.outputs_dir / conflicting.filename
    conflict_path.write_bytes(b"unrelated local file")
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError):
        bundles.import_bundle(Path(archive["path"]))
    assert _snapshot(store) == snapshot


@pytest.mark.parametrize("damage", ["checksum", "database", "traversal", "backslash", "duplicate", "case"])
def test_corrupt_or_unsafe_bundle_is_rejected_before_destination_changes(store, monkeypatch, tmp_path, damage):
    row = _save()
    archive = _export()
    with ZipFile(archive["path"]) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(entries["manifest.json"])
    image_name = "images/" + row.filename
    duplicate = None
    if damage == "checksum":
        entries[image_name] += b"corrupt"
    elif damage == "database":
        entries["database.sqlite3"] = b"not a database"
        manifest["files"]["database.sqlite3"] = {
            "size": len(entries["database.sqlite3"]),
            "sha256": hashlib.sha256(entries["database.sqlite3"]).hexdigest(),
        }
    elif damage == "duplicate":
        duplicate = image_name
    else:
        unsafe = {
            "traversal": "images/../../outside.png",
            "backslash": "images\\outside.png",
            "case": image_name.upper(),
        }[damage]
        entries[unsafe] = entries[image_name]
        manifest["files"][unsafe] = dict(manifest["files"][image_name])
    entries["manifest.json"] = json.dumps(manifest).encode()
    damaged = tmp_path / "damaged.zip"
    with ZipFile(damaged, "w") as target:
        for name, data in entries.items():
            target.writestr(name, data)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                target.writestr(duplicate, entries[duplicate])
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError):
        bundles.import_bundle(damaged)
    assert _snapshot(store) == snapshot
    assert not (tmp_path / "outside.png").exists()


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
        assert export.result(timeout=10)["deleted_ids"] == [original.id]
        new = save.result(timeout=10)
    assert new.id > original.id
    assert store.list_images()[1] == 1
    assert (store.settings.outputs_dir / new.filename).is_file()


def test_gallery_reservation_is_also_visible_to_a_separate_local_process(store):
    code = """
from pathlib import Path
import sys
from backend.gallery_lock import GalleryBusy, hold
try:
    with hold(Path(sys.argv[1])):
        raise SystemExit('the child acquired an already reserved gallery')
except GalleryBusy:
    pass
"""
    with gallery_lock.hold(store.settings.db_path):
        child = subprocess.run(
            [sys.executable, "-c", code, str(store.settings.db_path)],
            capture_output=True, text=True, timeout=15,
        )
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize("name", ["../outside.zip", "..\\outside.zip", "/outside.zip", "C:/outside.zip"])
def test_download_names_cannot_escape_the_bundle_directory(store, name):
    with pytest.raises(bundles.BundleError):
        bundles.bundle_path(name)
