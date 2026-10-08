"""Verified gallery archives: preservation, safe clearing and atomic merges."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundle_crypto, bundles, gallery_lock, storage
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


@pytest.mark.parametrize("route", ["move", "copy"])
def test_import_publish_failure_rolls_back_rows_and_files_already_added(store, monkeypatch, tmp_path, route):
    _save("first", 1)
    _save("second", 2)
    archive = _export()
    dest = _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    published = 0

    if route == "move":
        real_install = bundles._install

        def fail_second_image(source, target, installed):
            nonlocal published
            if target.suffix == ".png":
                published += 1
                if published == 2:
                    raise OSError("destination disk full")
            return real_install(source, target, installed)

        monkeypatch.setattr(bundles, "_install", fail_second_image)
    else:
        # Another volume: the move is refused and installation copies instead.
        def cross_device(*args):
            raise OSError(18, "cross-device link")

        monkeypatch.setattr(bundles.os, "rename", cross_device)
        monkeypatch.setattr(bundles.os, "link", cross_device, raising=False)
        real_open = Path.open

        def fail_second_copy(self, mode="r", *args, **kwargs):
            nonlocal published
            # The copy is written as a `.part` beside its target.
            if mode == "xb" and self.parent == dest.outputs_dir and self.name.endswith(".png.part"):
                published += 1
                if published == 2:
                    raise OSError("destination disk full")
            return real_open(self, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_second_copy)
    with pytest.raises((OSError, bundles.BundleError), match="disk full"):
        bundles.import_bundle(Path(archive["path"]))
    assert published == 2
    assert _snapshot(store) == snapshot


def _locked_database(staged):
    raise sqlite3.OperationalError("database is locked")


def test_a_database_replacement_refused_after_every_file_is_installed_rolls_them_back(store, monkeypatch, tmp_path):
    _save("first", 1)
    _save("second", 2)
    archive = _export()
    dest = _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    installed = []
    real_install = bundles._install

    def counted(source, target, done):
        real_install(source, target, done)
        installed.append(target)

    monkeypatch.setattr(bundles, "_install", counted)
    monkeypatch.setattr(bundles, "_replace_database", _locked_database)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        bundles.import_bundle(Path(archive["path"]))
    assert any(target.suffix == ".png" for target in installed)
    assert not any(target.exists() for target in installed)
    assert _snapshot(store) == snapshot
    assert sorted(p.name for p in dest.outputs_dir.glob("*.png")) == sorted(
        name for name in snapshot[1] if name.endswith(".png") and "/" not in name)


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


@pytest.mark.parametrize("links", [True, False])
def test_a_copy_across_volumes_takes_its_final_name_only_once_complete(store, monkeypatch, tmp_path, links):
    row = _save("first", 1)
    archive = _export()
    dest = _destination(store, monkeypatch, tmp_path)
    real_rename, real_link = bundles.os.rename, getattr(bundles.os, "link", None)

    def cross_device(source, target):
        # The staged file is on another volume; within the images directory
        # an exclusive rename (or, with `links`, a hard link) still works.
        if Path(source).parent != Path(target).parent or not links:
            raise OSError(18, "cross-device link")
        return (real_rename if bundles.os.name == "nt" else real_link)(source, target)

    monkeypatch.setattr(bundles.os, "rename", cross_device)
    monkeypatch.setattr(bundles.os, "link", cross_device, raising=False)
    seen = []
    real_copy = bundles.shutil.copyfileobj

    def copy(src, dst, *args):
        target = dest.outputs_dir / row.filename
        if Path(dst.name).parent == dest.outputs_dir and Path(dst.name).name == row.filename + ".part":
            # A crash here leaves a partial the sweeps know, not the image.
            seen.append(target.exists())
        return real_copy(src, dst, *args)

    monkeypatch.setattr(bundles.shutil, "copyfileobj", copy)
    assert bundles.import_bundle(Path(archive["path"]))["imported"] == 1
    assert seen == [False]
    assert [r.filename for r in store.list_images()[0]] == [row.filename]
    assert (dest.outputs_dir / row.filename).is_file()
    assert not list(dest.outputs_dir.rglob("*.part"))


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
    assert "incomplete image write" in result["cleanup_error"]
    assert Path(result["path"]).is_file()
    assert new.id > original.id
    assert store.list_images()[1] == 2
    assert (store.settings.outputs_dir / original.filename).is_file()
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


# The child reserves the gallery, says so, and keeps it until its stdin has a
# line (or closes) - so the test, not a sleep, decides when it lets go.
_HOLDING_CHILD = """
from pathlib import Path
import sys
from backend.gallery_lock import hold
with hold(Path(sys.argv[1])):
    print('held', flush=True)
    sys.stdin.readline()
"""


def _holding_child(db_path):
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDING_CHILD, str(db_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert child.stdout.readline().strip() == "held"
    return child


def test_a_waiting_session_gets_the_gallery_once_another_process_lets_go(store):
    child = _holding_child(store.settings.db_path)
    try:
        # Maintenance fails fast, as documented.
        with pytest.raises(gallery_lock.GalleryBusy):
            with gallery_lock.hold(store.settings.db_path):
                pass
        threading.Timer(0.3, lambda: (child.stdin.write("\n"), child.stdin.flush())).start()
        started = time.monotonic()
        with gallery_lock.hold(store.settings.db_path, wait=15):
            waited = time.monotonic() - started
        assert 0.2 < waited < 15
    finally:
        child.stdin.close()
        child.wait(timeout=15)


def test_a_waiting_session_refuses_once_its_deadline_passes(store, monkeypatch):
    child = _holding_child(store.settings.db_path)
    try:
        started = time.monotonic()
        with pytest.raises(gallery_lock.GalleryBusy):
            with gallery_lock.hold(store.settings.db_path, wait=0.4):
                pass
        assert time.monotonic() - started >= 0.35
        # The ordinary storage calls wait SESSION_WAIT, read per call.
        monkeypatch.setattr(gallery_lock, "SESSION_WAIT", 0.2)
        with pytest.raises(gallery_lock.GalleryBusy):
            storage.get_image(1)
    finally:
        child.stdin.close()
        child.wait(timeout=15)


def test_threads_waiting_on_another_process_share_one_deadline(store):
    # Each thread's wait covers the in-process lock too: queued behind a
    # thread polling for the child, four waiters once refused at 1, 2, 3, 4 x.
    child = _holding_child(store.settings.db_path)
    waited = []

    def wait():
        started = time.monotonic()
        try:
            with gallery_lock.hold(store.settings.db_path, wait=0.5):
                pass
        except gallery_lock.GalleryBusy:
            waited.append(time.monotonic() - started)

    try:
        threads = [threading.Thread(target=wait) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        assert len(waited) == 4 and max(waited) < 0.9, waited
    finally:
        child.stdin.close()
        child.wait(timeout=15)


@pytest.mark.parametrize("name",["../outside.zip", "..\\outside.zip", "/outside.zip", "C:/outside.zip"])
def test_download_names_cannot_escape_the_bundle_directory(store, name):
    with pytest.raises(bundles.BundleError):
        bundles.bundle_path(name)


def test_password_export_is_encrypted_and_imports_only_with_its_password(store, monkeypatch, tmp_path):
    first = _save("first", 1)
    _save("second", 2, reference=[first.id])
    archive = bundles.export_bundle(expected=2, password="tea kettle")
    bundle = Path(archive["path"])
    assert archive["cleared"] and bundle.suffix == bundle_crypto.SUFFIX
    blob = bundle.read_bytes()
    assert blob.startswith(bundle_crypto.MAGIC) and b"manifest.json" not in blob and first.filename.encode() not in blob
    assert [b["encrypted"] for b in bundles.list_bundles() if b["name"] == archive["name"]] == [True]

    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    for password, message in [(None, "password-protected"), ("wrong", "incorrect")]:
        with pytest.raises(bundles.BundleError, match=message):
            bundles.import_bundle(bundle, password)
        assert _snapshot(store) == snapshot
    imported = bundles.import_bundle(bundle, "tea kettle")
    assert imported["imported"] == 2 and imported["unresolved_references"] == 0
    assert bundles.import_bundle(bundle, "tea kettle")["present"] == 2


def test_a_password_bundle_that_fails_its_read_back_clears_nothing(store, monkeypatch):
    _save("kept", 1)
    snapshot = _snapshot(store)
    # Without its final chunk the ciphertext cannot authenticate.
    monkeypatch.setattr(bundle_crypto.EncryptingWriter, "finish", lambda self: None)
    with pytest.raises(bundles.BundleError):
        bundles.export_bundle(expected=1, password="pw")
    assert _snapshot(store) == snapshot
    assert not list(bundles._directory().iterdir())


def test_altered_password_bundle_is_rejected_before_destination_changes(store, monkeypatch, tmp_path):
    _save("source", 1)
    bundle = Path(bundles.export_bundle(expected=1, password="pw")["path"])
    blob = bytearray(bundle.read_bytes())
    blob[len(blob) // 2] ^= 1
    bundle.write_bytes(bytes(blob))
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match="damaged or was changed"):
        bundles.import_bundle(bundle, "pw")
    assert _snapshot(store) == snapshot


def test_a_password_bundle_altered_past_its_first_chunk_is_refused_and_staging_discarded(store, monkeypatch, tmp_path):
    # Only the header and the final chunk are checked on open; any other
    # chunk is authenticated as it is read, after earlier members are staged.
    monkeypatch.setitem(bundle_crypto.EncryptingWriter.__init__.__kwdefaults__, "chunk", 4096)
    _save("first", 1)
    _save("second", 2)
    bundle = Path(bundles.export_bundle(expected=2, password="pw")["path"])
    blob = bytearray(bundle.read_bytes())
    assert len(blob) > bundle_crypto.HEADER_SIZE + 3 * (4096 + 16)
    blob[bundle_crypto.HEADER_SIZE + 4096 + 16 + 10] ^= 1  # inside the second chunk
    bundle.write_bytes(bytes(blob))
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match="damaged or was changed"):
        bundles.import_bundle(bundle, "pw")
    assert _snapshot(store) == snapshot
    staging = store.bundle_staging_dir()
    assert not staging.exists() or not list(staging.iterdir())
    assert not list(store.settings.db_path.parent.glob(".bundle-*"))


def test_repeat_import_recognises_a_re_encoded_existing_image_by_its_pixels(store, monkeypatch, tmp_path):
    row = _save("source", 1)
    bundle = Path(_export()["path"])
    dest = _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    path = dest.outputs_dir / row.filename
    with Image.open(path) as image:
        image.load()
        image.save(path, compress_level=0)  # different IDAT bytes, same pixels
    assert bundles.import_bundle(bundle)["present"] == 1
    Image.new("RGB", (256, 256), "white").save(path)
    with pytest.raises(bundles.BundleError, match="already uses the filename"):
        bundles.import_bundle(bundle)


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


def test_re_import_restores_an_image_whose_row_outlived_its_file(store, monkeypatch, tmp_path):
    source = _save("source", 1)
    conditioned = _save("conditioned", 2, reference=[source.id])
    bundle = Path(_export()["path"])
    dest = _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    assert bundles.import_bundle(bundle)["imported"] == 2
    rows = {r.filename: r for r in store.list_images()[0]}
    path = dest.outputs_dir / conditioned.filename
    original = path.read_bytes()
    # What a clear interrupted between removing files and rows leaves.
    path.unlink()
    snapshot_rows = _snapshot(store)[0]
    result = bundles.import_bundle(bundle)
    assert (result["imported"], result["present"], result["restored"]) == (0, 1, 1)
    assert _snapshot(store)[0] == snapshot_rows  # the row is unchanged
    assert path.read_bytes() == original
    assert _metadata(path)["reference"] == [rows[source.filename].id]

    # Only the very picture the row recorded, with the same identity.
    path.unlink()
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET content_hash = '0123456789abcdef' WHERE filename = ?", (conditioned.filename,))
    with pytest.raises(bundles.BundleError, match="lost its image"):
        bundles.import_bundle(bundle)
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET content_hash = ?, seed = 7 WHERE filename = ?",
                     (rows[conditioned.filename].content_hash, conditioned.filename))
    with pytest.raises(bundles.BundleError, match="different image metadata"):
        bundles.import_bundle(bundle)
    assert not path.exists()

    # The short hash is of the raw bytes alone: the size must match the row.
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET seed = ?, width = 2 * width WHERE filename = ?",
                     (conditioned.seed, conditioned.filename))
    with pytest.raises(bundles.BundleError, match="lost its image"):
        bundles.import_bundle(bundle)
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET width = ?, height = ? WHERE filename = ?",
                     (conditioned.width, conditioned.height, conditioned.filename))
    assert bundles.import_bundle(bundle)["restored"] == 1


def test_re_import_restores_only_a_plain_rgb_picture(store, monkeypatch, tmp_path):
    # A palette copy's raw bytes are indices, so its short hash says nothing
    # of the colours; a render is always RGB, so nothing else is restored.
    row = _save("source", 1)
    image = Image.new("P", (row.width, row.height), 0)
    image.putpalette([255, 0, 0, *[0] * 765])
    image.save(store.settings.outputs_dir / row.filename)
    bundle = Path(_export()["path"])
    dest = _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    (dest.outputs_dir / row.filename).unlink()
    with pytest.raises(bundles.BundleError, match="lost its image"):
        bundles.import_bundle(bundle)
    assert not (dest.outputs_dir / row.filename).exists()


@pytest.mark.parametrize("existing", ["palette", "gray"])
def test_repeat_import_tells_apart_pictures_whose_raw_bytes_agree(store, monkeypatch, tmp_path, existing):
    row = _save("source", 1)

    def indexed(colour):
        image = Image.new("P", (256, 256), 0)
        image.putpalette([*colour, *[0] * 765])
        return image

    # A palette image's bytes are its indices: these three share them all.
    indexed((255, 0, 0)).save(store.settings.outputs_dir / row.filename)
    bundle = Path(_export()["path"])
    dest = _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    path = dest.outputs_dir / row.filename
    other = indexed((0, 0, 255)) if existing == "palette" else Image.new("L", (256, 256), 0)
    other.save(path)
    with pytest.raises(bundles.BundleError, match="already uses the filename"):
        bundles.import_bundle(bundle)


def test_repeat_import_keeps_a_known_image_s_own_thumbnail(store, monkeypatch, tmp_path):
    kept = _save("kept", 1)
    removed = _save("removed", 2)
    bundle = Path(_export()["path"])
    dest = _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 2
    thumbs = dest.outputs_dir / "thumbnails"
    # Another Pillow build, or a regenerated cache: different bytes.
    regenerated = thumbs / Path(kept.filename).with_suffix(".webp").name
    Image.new("RGB", (32, 32), "white").save(regenerated, format="WEBP")
    ours = regenerated.read_bytes()
    assert bundles.import_bundle(bundle)["present"] == 2
    assert regenerated.read_bytes() == ours

    # A new image's thumbnail is still held to its bundle bytes.
    store.delete_image(next(r.id for r in store.list_images()[0] if r.filename == removed.filename))
    (thumbs / Path(removed.filename).with_suffix(".webp").name).write_bytes(b"an unrelated file")
    with pytest.raises(bundles.BundleError, match="another file already uses the filename thumbnails/"):
        bundles.import_bundle(bundle)


def test_an_oversized_tracked_image_is_refused_before_it_is_read(store, monkeypatch, tmp_path):
    _save("source", 1)
    bundle = Path(_export()["path"])
    _destination(store, monkeypatch, tmp_path)
    snapshot = _snapshot(store)
    monkeypatch.setattr(bundles, "_MAX_IMAGE_BYTES", 1024)
    with pytest.raises(bundles.BundleError, match="too large to be a gallery image"):
        bundles.import_bundle(bundle)
    assert _snapshot(store) == snapshot


def test_a_member_padded_past_its_deflate_stream_is_refused_before_it_is_read(store, monkeypatch, tmp_path):
    # A small file_size passes every size cap, but read() would fetch the
    # whole compress_size at once; the CRC and SHA-256 still match.
    row = _save()
    archive = _export()
    with ZipFile(archive["path"]) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    image_name = "images/" + row.filename
    real = zipfile._get_compressor

    class Padded:
        def __init__(self, inner):
            self.inner = inner

        def compress(self, data):
            return self.inner.compress(data)

        def flush(self):
            return self.inner.flush() + b"\0" * (256 * 1024)

    padded = tmp_path / "padded.zip"
    with ZipFile(padded, "w") as target:
        for name, data in entries.items():
            if name == image_name:
                monkeypatch.setattr(zipfile, "_get_compressor", lambda *a, **k: Padded(real(*a, **k)))
                target.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
                monkeypatch.setattr(zipfile, "_get_compressor", real)
            else:
                target.writestr(name, data)
    with ZipFile(padded) as check:
        info = check.getinfo(image_name)
        assert info.compress_size > 256 * 1024 > info.file_size
        assert check.read(image_name) == entries[image_name]  # zipfile alone accepts it
    _destination(store, monkeypatch, tmp_path)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match="stored larger than its contents"):
        bundles.import_bundle(padded)
    assert _snapshot(store) == snapshot


def _leftovers(store):
    beside = store.settings.db_path.parent
    staging = store.bundle_staging_dir()
    (beside / "bundles").mkdir(exist_ok=True)
    made = {
        "archive": beside / "bundles" / "local-diffusion-x.zip.part",
        "export": beside / ".bundle-ab12cd_3",
        "merge": beside / ".bundle-merge-ab12cd34",
        "old upload": beside / ".bundle-upload-ab12cd34.part",
        "import": staging / ".bundle-import-ab12cd34",
        "upload": staging / ".bundle-upload-ab12cd34.part",
    }
    for path in made.values():
        if path.suffix == ".part":
            path.write_bytes(b"left by a killed process")
        else:
            (path / "images").mkdir(parents=True)
            (path / "images" / "unpacked.png").write_bytes(b"a gallery-sized copy")
    old = time.time() - storage.PARTIAL_SETTLE_SECONDS - 60
    for path in made.values():
        os.utime(path, (old, old))
    fresh = staging / ".bundle-upload-fresh123.part"
    fresh.write_bytes(b"an upload still arriving")
    unrelated = beside / ".bundle-notes.txt"
    unrelated.write_bytes(b"not ours")
    os.utime(unrelated, (old, old))
    return made, [fresh, unrelated]


@pytest.mark.parametrize("sweep", ["startup", "export", "import", "clear"])
def test_what_a_killed_bundle_operation_left_is_swept_once_settled(store, sweep):
    _save("source", 1)
    made, kept = _leftovers(store)
    assert store.summarise()["partial_files"] == len(made)
    if sweep == "startup":
        assert store.discard_bundle_leftovers() == len(made)
    elif sweep == "export":
        bundle = Path(_export()["path"])
    elif sweep == "import":
        bad = store.settings.db_path.parent / "broken.zip"
        bad.write_bytes(b"not a ZIP archive")
        with pytest.raises(bundles.BundleError):
            bundles.import_bundle(bad)
    else:
        assert store.clear_history()["partials"] == len(made)
    assert not any(path.exists() for path in made.values())
    assert all(path.exists() for path in kept)
    if sweep == "export":
        assert bundle.is_file()


def test_import_unpacks_on_the_images_volume_and_merges_beside_the_database(store, monkeypatch, tmp_path):
    _save("source", 1)
    bundle = Path(_export()["path"])
    dest = _destination(store, monkeypatch, tmp_path / "elsewhere")
    seen = {}
    real_verify = bundles._verify_archive

    def capture(path, destination=None, **kwargs):
        seen.update(destination=destination, merge=kwargs["merge"][0])
        return real_verify(path, destination, **kwargs)

    monkeypatch.setattr(bundles, "_verify_archive", capture)
    assert bundles.import_bundle(bundle)["imported"] == 1
    assert seen["destination"].parent == store.bundle_staging_dir()
    assert store.bundle_staging_dir().parent == dest.outputs_dir.resolve().parent
    assert seen["merge"].parent == dest.db_path.parent
    assert not list(store.bundle_staging_dir().iterdir())


def test_staging_moves_inside_an_images_directory_that_is_its_own_volume(store, monkeypatch):
    _save("source", 1)
    outputs = store.settings.outputs_dir.resolve()
    real_stat = os.stat

    def mounted(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if Path(path) == outputs.parent:
            return os.stat_result((result.st_mode, result.st_ino, result.st_dev + 1, *result[3:10]))
        return result

    monkeypatch.setattr(os, "stat", mounted)
    staging = store.bundle_staging_dir()
    monkeypatch.setattr(os, "stat", real_stat)
    assert staging.parent == outputs
    # The inventory skips the directory, and an upload inside it.
    (staging / ".bundle-upload-ab12cd34.part").write_bytes(b"arriving")
    assert bundles.preview()["images"] == 1
    assert store.discard_bundle_leftovers() == 0


def test_free_space_is_checked_on_each_volume_written(monkeypatch, tmp_path):
    images, database = tmp_path / "images", tmp_path / "database"
    images.mkdir()
    database.mkdir()
    devices = {images: 1, database: 2}
    free = {images: 100, database: 100}
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        return os.stat_result((result.st_mode, result.st_ino, devices[Path(path)], *result[3:10]))

    class Usage:
        def __init__(self, path):
            self.free = free[Path(path)]

    monkeypatch.setattr(bundles.os, "stat", stat)
    monkeypatch.setattr(bundles.shutil, "disk_usage", Usage)
    bundles._check_space([(images, 90, "unpack this bundle"), (database, 90, "merge")])
    with pytest.raises(bundles.BundleError, match="unpack this bundle"):
        bundles._check_space([(images, 101, "unpack this bundle"), (database, 90, "merge")])
    with pytest.raises(bundles.BundleError, match="merge"):
        bundles._check_space([(images, 90, "unpack this bundle"), (database, 101, "merge")])
    devices[database] = 1  # one volume: both land on it
    with pytest.raises(bundles.BundleError, match="unpack this bundle"):
        bundles._check_space([(images, 60, "unpack this bundle"), (database, 60, "merge")])


def test_a_password_export_derives_its_key_once(store, monkeypatch, tmp_path):
    _save("sealed", 1)
    calls = []
    real = bundle_crypto._derive
    monkeypatch.setattr(bundle_crypto, "_derive", lambda *a: calls.append(a) or real(*a))
    bundle = Path(bundles.export_bundle(expected=1, password="pw")["path"])
    assert len(calls) == 1  # the read-back reuses the writer's key for its own header
    _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle, "pw")["imported"] == 1
    assert len(calls) == 2  # import derives from the password it was given


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "1", "I;16", "LA"])
def test_raw_digest_is_tobytes(mode):
    image = Image.effect_noise((37, 11), 64).convert(mode)
    assert bundles._raw_digest(image) == hashlib.sha256(image.tobytes()).hexdigest()
    assert bundles._raw_digest(Image.new(mode, (0, 5))) == hashlib.sha256(b"").hexdigest()


def test_repeat_import_neither_decodes_nor_unpacks_what_is_already_there(store, monkeypatch, tmp_path):
    for seed in (1, 2, 3):
        _save(f"image {seed}", seed)
    bundle = Path(_export()["path"])
    _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 3
    snapshot = _snapshot(store)
    decoded, staged = [], []
    real_pixels, real_stage = bundles._pixels, bundles._write_staged
    monkeypatch.setattr(bundles, "_pixels", lambda *a, **k: decoded.append(a) or real_pixels(*a, **k))
    monkeypatch.setattr(bundles, "_write_staged", lambda target, *a: staged.append(target.name) or real_stage(target, *a))
    assert bundles.import_bundle(bundle)["present"] == 3
    assert decoded == [] and staged == []
    assert _snapshot(store) == snapshot


def test_a_present_image_s_bundle_row_is_still_validated_without_a_decode(store, monkeypatch, tmp_path):
    row = _save("source", 1)
    bundle = Path(_export()["path"])
    _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    with ZipFile(bundle) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    database = tmp_path / "bundle.sqlite3"
    database.write_bytes(members["database.sqlite3"])
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("UPDATE images SET width = 999 WHERE filename = ?", (row.filename,))
        conn.commit()
    members["database.sqlite3"] = database.read_bytes()
    manifest = json.loads(members["manifest.json"])
    manifest["files"]["database.sqlite3"] = {"size": len(members["database.sqlite3"]),
                                             "sha256": hashlib.sha256(members["database.sqlite3"]).hexdigest()}
    members["manifest.json"] = json.dumps(manifest).encode()
    altered = tmp_path / "altered.zip"
    with ZipFile(altered, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match="dimensions disagree"):
        bundles.import_bundle(altered)
    assert _snapshot(store) == snapshot


def test_a_declined_member_whose_destination_vanished_refuses_the_import(store, monkeypatch, tmp_path):
    row = _save("source", 1)
    thumb = Path(row.thumb_url.removeprefix("/outputs/"))
    bundle = Path(_export()["path"])
    with ZipFile(bundle) as archive:
        thumb_bytes = archive.read("images/" + thumb.as_posix())
    dest = _destination(store, monkeypatch, tmp_path)
    # A leftover copy of the new image's thumbnail: identical, so kept in place.
    left = dest.outputs_dir.joinpath(*thumb.parts)
    left.parent.mkdir(parents=True, exist_ok=True)
    left.write_bytes(thumb_bytes)
    real_verify = bundles._verify_archive

    def vanish(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        left.unlink()
        return result
    monkeypatch.setattr(bundles, "_verify_archive", vanish)
    with pytest.raises(bundles.BundleError, match="gallery changed during import"):
        bundles.import_bundle(bundle)
    assert _snapshot(store) == ([], {})  # nothing installed, no row added
    monkeypatch.setattr(bundles, "_verify_archive", real_verify)
    left.write_bytes(thumb_bytes)
    assert bundles.import_bundle(bundle)["imported"] == 1
    assert left.read_bytes() == thumb_bytes


def test_a_failed_metadata_rewrite_refuses_the_whole_merge(store, monkeypatch, tmp_path):
    for seed in range(1, 7):
        _save(f"image {seed}", seed)
    bundle = Path(_export()["path"])
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    real = bundles._rewrite_metadata
    victim = sorted(r for r in ZipFile(bundle).namelist() if r.count("/") == 1 and r.endswith(".png"))[3]

    def rewrite(path, metadata):
        if path.name == victim.removeprefix("images/"):
            raise bundles.BundleError(f"not a PNG image: {path.name}")
        real(path, metadata)
    monkeypatch.setattr(bundles, "_rewrite_metadata", rewrite)
    with pytest.raises(bundles.BundleError, match="invalid metadata for"):
        bundles.import_bundle(bundle)
    assert _snapshot(store) == snapshot
