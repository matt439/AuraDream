"""Where bundle work happens on disk: staging per volume, free space and the leftover sweep."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from backend import bundle_archive, bundles, storage
from tests.bundle_helpers import _save, _export, _destination


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

    monkeypatch.setattr(bundle_archive.os, "stat", stat)
    monkeypatch.setattr(bundle_archive.shutil, "disk_usage", Usage)
    bundle_archive._check_space([(images, 90, "unpack this bundle"), (database, 90, "merge")])
    with pytest.raises(bundles.BundleError, match="unpack this bundle"):
        bundle_archive._check_space([(images, 101, "unpack this bundle"), (database, 90, "merge")])
    with pytest.raises(bundles.BundleError, match="merge"):
        bundle_archive._check_space([(images, 90, "unpack this bundle"), (database, 101, "merge")])
    devices[database] = 1  # one volume: both land on it
    with pytest.raises(bundles.BundleError, match="unpack this bundle"):
        bundle_archive._check_space([(images, 60, "unpack this bundle"), (database, 60, "merge")])
