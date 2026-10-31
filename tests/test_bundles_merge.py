"""Bundle imports: atomic merges and archives refused before the destination changes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import warnings
import zipfile
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest

from backend import bundle_archive, bundle_images, bundles
from tests.bundle_helpers import save_source, export_all, use_destination, bundle_snapshot, locked_database


@pytest.mark.parametrize("route", ["move", "copy"])
def test_import_publish_failure_rolls_back_rows_and_files_already_added(store, monkeypatch, tmp_path, route):
    save_source("first", 1)
    save_source("second", 2)
    archive = export_all()
    dest = use_destination(store, monkeypatch, tmp_path)
    save_source("existing", 99)
    snapshot = bundle_snapshot(store)
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
    assert bundle_snapshot(store) == snapshot


def test_a_database_replacement_refused_after_every_file_is_installed_rolls_them_back(store, monkeypatch, tmp_path):
    save_source("first", 1)
    save_source("second", 2)
    archive = export_all()
    dest = use_destination(store, monkeypatch, tmp_path)
    save_source("existing", 99)
    snapshot = bundle_snapshot(store)
    installed = []
    real_install = bundles._install

    def counted(source, target, done):
        real_install(source, target, done)
        installed.append(target)

    monkeypatch.setattr(bundles, "_install", counted)
    monkeypatch.setattr(bundles, "_replace_database", locked_database)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        bundles.import_bundle(Path(archive["path"]))
    assert any(target.suffix == ".png" for target in installed)
    assert not any(target.exists() for target in installed)
    assert bundle_snapshot(store) == snapshot
    assert sorted(p.name for p in dest.outputs_dir.glob("*.png")) == sorted(
        name for name in snapshot[1] if name.endswith(".png") and "/" not in name)


@pytest.mark.parametrize("links", [True, False])
def test_a_copy_across_volumes_takes_its_final_name_only_once_complete(store, monkeypatch, tmp_path, links):
    row = save_source("first", 1)
    archive = export_all()
    dest = use_destination(store, monkeypatch, tmp_path)
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


def test_invalid_zip_does_not_mutate_the_destination(store, tmp_path):
    save_source("existing")
    snapshot = bundle_snapshot(store)
    bundle = tmp_path / "broken.zip"
    bundle.write_bytes(b"not a ZIP archive")
    with pytest.raises(bundles.BundleError):
        bundles.import_bundle(bundle)
    assert bundle_snapshot(store) == snapshot


def test_filename_conflict_refuses_the_whole_merge_before_adding_any_rows(store, monkeypatch, tmp_path):
    save_source("would import first", 1)
    conflicting = save_source("would conflict last", 2)
    archive = export_all()
    use_destination(store, monkeypatch, tmp_path)
    save_source("existing", 9)
    conflict_path = store.settings.outputs_dir / conflicting.filename
    conflict_path.write_bytes(b"unrelated local file")
    snapshot = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError):
        bundles.import_bundle(Path(archive["path"]))
    assert bundle_snapshot(store) == snapshot


@pytest.mark.parametrize("damage", ["checksum", "database", "traversal", "backslash", "duplicate", "case"])
def test_corrupt_or_unsafe_bundle_is_rejected_before_destination_changes(store, monkeypatch, tmp_path, damage):
    row = save_source()
    archive = export_all()
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
    use_destination(store, monkeypatch, tmp_path)
    save_source("existing", 99)
    snapshot = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError):
        bundles.import_bundle(damaged)
    assert bundle_snapshot(store) == snapshot
    assert not (tmp_path / "outside.png").exists()


def test_an_oversized_tracked_image_is_refused_before_it_is_read(store, monkeypatch, tmp_path):
    save_source("source", 1)
    bundle = Path(export_all()["path"])
    use_destination(store, monkeypatch, tmp_path)
    snapshot = bundle_snapshot(store)
    monkeypatch.setattr(bundle_archive, "_MAX_IMAGE_BYTES", 1024)
    with pytest.raises(bundles.BundleError, match="too large to be a gallery image"):
        bundles.import_bundle(bundle)
    assert bundle_snapshot(store) == snapshot


def test_a_member_padded_past_its_deflate_stream_is_refused_before_it_is_read(store, monkeypatch, tmp_path):
    # A small file_size passes every size cap, but read() would fetch the
    # whole compress_size at once; the CRC and SHA-256 still match.
    row = save_source()
    archive = export_all()
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
    use_destination(store, monkeypatch, tmp_path)
    snapshot = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError, match="stored larger than its contents"):
        bundles.import_bundle(padded)
    assert bundle_snapshot(store) == snapshot


def test_a_declined_member_whose_destination_vanished_refuses_the_import(store, monkeypatch, tmp_path):
    row = save_source("source", 1)
    thumb = Path(row.thumb_url.removeprefix("/outputs/"))
    bundle = Path(export_all()["path"])
    with ZipFile(bundle) as archive:
        thumb_bytes = archive.read("images/" + thumb.as_posix())
    dest = use_destination(store, monkeypatch, tmp_path)
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
    assert bundle_snapshot(store) == ([], {})  # nothing installed, no row added
    monkeypatch.setattr(bundles, "_verify_archive", real_verify)
    left.write_bytes(thumb_bytes)
    assert bundles.import_bundle(bundle)["imported"] == 1
    assert left.read_bytes() == thumb_bytes


def test_a_failed_metadata_rewrite_refuses_the_whole_merge(store, monkeypatch, tmp_path):
    for seed in range(1, 7):
        save_source(f"image {seed}", seed)
    bundle = Path(export_all()["path"])
    use_destination(store, monkeypatch, tmp_path)
    save_source("existing", 99)
    snapshot = bundle_snapshot(store)
    real = bundle_images._rewrite_metadata
    victim = sorted(r for r in ZipFile(bundle).namelist() if r.count("/") == 1 and r.endswith(".png"))[3]

    def rewrite(path, metadata):
        if path.name == victim.removeprefix("images/"):
            raise bundles.BundleError(f"not a PNG image: {path.name}")
        real(path, metadata)
    monkeypatch.setattr(bundle_images, "_rewrite_metadata", rewrite)
    with pytest.raises(bundles.BundleError, match="invalid metadata for"):
        bundles.import_bundle(bundle)
    assert bundle_snapshot(store) == snapshot


def test_a_bundle_from_before_the_bin_imports_its_images_live(store, monkeypatch, tmp_path):
    row = save_source("source", 1)
    bundle = Path(export_all()["path"])
    with ZipFile(bundle) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    database = tmp_path / "bundle.sqlite3"
    database.write_bytes(members["database.sqlite3"])
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_images_binned")
        conn.execute("ALTER TABLE images DROP COLUMN binned_at")
        conn.commit()
    members["database.sqlite3"] = database.read_bytes()
    manifest = json.loads(members["manifest.json"])
    manifest["files"]["database.sqlite3"] = {"size": len(members["database.sqlite3"]),
                                             "sha256": hashlib.sha256(members["database.sqlite3"]).hexdigest()}
    members["manifest.json"] = json.dumps(manifest).encode()
    older = tmp_path / "older.zip"
    with ZipFile(older, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    use_destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(older)
    assert (imported["imported"], imported["binned"]) == (1, 0)
    (record,) = store.list_images()[0]
    assert record.filename == row.filename and record.binned_at is None


@pytest.mark.parametrize("name", ["local-diffusion-gallery", "auradream-gallery"])
def test_a_bundle_with_the_old_or_new_format_id_imports(store, monkeypatch, tmp_path, name):
    row = save_source("source", 1)
    bundle = Path(export_all()["path"])
    with ZipFile(bundle) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(members["manifest.json"])
    assert manifest["format"] == "auradream-gallery"
    manifest["format"] = name
    members["manifest.json"] = json.dumps(manifest).encode()
    renamed = tmp_path / "renamed.zip"
    with ZipFile(renamed, "w") as archive:
        for member, data in members.items():
            archive.writestr(member, data)
    use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(renamed)["imported"] == 1
    assert [record.filename for record in store.list_images()[0]] == [row.filename]


def test_a_bundle_with_another_format_id_is_refused(store, monkeypatch, tmp_path):
    save_source("source", 1)
    bundle = Path(export_all()["path"])
    with ZipFile(bundle) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(members["manifest.json"])
    manifest["format"] = "someone-elses-gallery"
    members["manifest.json"] = json.dumps(manifest).encode()
    other = tmp_path / "other.zip"
    with ZipFile(other, "w") as archive:
        for member, data in members.items():
            archive.writestr(member, data)
    use_destination(store, monkeypatch, tmp_path)
    snapshot = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError, match="not a supported"):
        bundles.import_bundle(other)
    assert bundle_snapshot(store) == snapshot
