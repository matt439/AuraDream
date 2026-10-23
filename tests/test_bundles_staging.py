"""Where bundle work happens on disk: staging per volume, free space and the leftover sweep."""
from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from backend import bundle_archive, bundles, storage
from tests.bundle_helpers import _save, _export, _destination


def _leftovers(store):
    beside = store.settings.db_path.parent
    staging = store.bundle_staging_dir()
    directory = store.bundle_directory()
    directory.mkdir(parents=True, exist_ok=True)
    made = {
        "archive": directory / "local-diffusion-x.zip.part",
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
@pytest.mark.parametrize("configured", [False, True])
def test_what_a_killed_bundle_operation_left_is_swept_once_settled(store, sweep, configured, monkeypatch, tmp_path):
    if configured:
        monkeypatch.setattr(store, "settings", replace(store.settings, bundles_dir=tmp_path / "archive-drive" / "saved"))
    _save("source", 1)
    made, kept = _leftovers(store)
    assert store.summarise()["partial_files"] == len(made)
    if sweep == "startup":
        assert store.settle_partials() == {"recovered": 0, "discarded": len(made)}
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
    assert store.settle_partials() == {"recovered": 0, "discarded": 0}


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


def _age(*paths):
    old = time.time() - storage.PARTIAL_SETTLE_SECONDS - 60
    for path in paths:
        os.utime(path, (old, old))


@pytest.mark.parametrize("sweep", ["preview", "startup", "export", "import"])
def test_a_stale_image_partial_no_longer_blocks_every_bundle_operation(store, sweep):
    # A power cut mid-encode leaves a *.png.part with no row, and its
    # thumbnail's *.webp.part. Only `clear` swept them, so the bundle counts,
    # export and import were refused for good, restarts included.
    _save("source", 1)
    outputs = store.settings.outputs_dir
    debris = [outputs / "20261001-000000-5-abc.png.part",
              outputs / "thumbnails" / "20261001-000000-5-abc.webp.part"]
    for path in debris:
        path.write_bytes(b"half an encode")
    _age(*debris)
    if sweep == "preview":
        assert bundles.preview()["records"] == 1
    elif sweep == "startup":
        assert store.settle_partials() == {"recovered": 0, "discarded": 2}
    elif sweep == "export":
        assert Path(_export()["path"]).is_file()
    else:
        # Refused for what the file is, past the inventory that used to stop it.
        bad = store.settings.db_path.parent / "broken.zip"
        bad.write_bytes(b"not a ZIP archive")
        with pytest.raises(bundles.BundleError) as refused:
            bundles.import_bundle(bad)
        assert "image write" not in str(refused.value)
    assert not any(path.exists() for path in debris)


def test_a_committed_images_interrupted_save_is_finished_not_discarded(store):
    # `save_image` commits the row, then renames: a crash between the two
    # leaves the only copy of a gallery image as .part. The sweep puts it in
    # place. A thumbnail's partial may be cut short (`thumbnail_for` encodes
    # for tracked rows) and a placed one is never encoded again: it is
    # discarded, and the card falls back to the PNG.
    record = _save("source", 1)
    png = store.settings.outputs_dir / record.filename
    thumb = store._thumb_path(png)
    pixels, small = png.read_bytes(), thumb.read_bytes()
    partials = [png.with_name(png.name + ".part"), thumb.with_name(thumb.name + ".part")]
    png.replace(partials[0])
    thumb.unlink()
    partials[1].write_bytes(small[: len(small) // 3])
    _age(*partials)
    preview = bundles.preview()
    assert (preview["records"], preview["images"], preview["thumbnails"]) == (1, 1, 0)
    assert png.read_bytes() == pixels and not thumb.exists()
    assert not any(path.exists() for path in partials)
    assert store.get_image(record.id).thumb_url is None
    # One whose file is already in place is debris like any other.
    partials[0].write_bytes(b"a second copy")
    _age(partials[0])
    assert store.settle_partials() == {"recovered": 0, "discarded": 1}
    assert png.read_bytes() == pixels


def test_a_fresh_partial_names_its_folder_and_what_to_do(store):
    _save("source", 1)
    fresh = store.settings.outputs_dir / "20261012-000000-7-abc.png.part"
    fresh.write_bytes(b"encoding right now")
    with pytest.raises(bundles.BundleError) as refused:
        bundles.preview()
    message = str(refused.value)
    assert f"20261012-000000-7-abc.png.part in {store.settings.outputs_dir}" in message
    assert "still in progress; try again when the render finishes" in message
    assert fresh.exists()


def test_a_missing_image_is_named_by_its_id_and_the_way_out(store):
    # Its thumbnail outlives the PNG, so the card looks whole, and a filename
    # is no search key: the refusal names the id the gallery's `id:` finds.
    live, binned = _save("source", 1), _save("source", 2)
    storage.bin_images([binned.id])
    (store.settings.outputs_dir / binned.filename).unlink()
    with pytest.raises(bundles.MissingImage) as refused:
        bundles.preview()
    assert (refused.value.image_id, refused.value.binned) == (binned.id, True)
    assert str(refused.value).startswith(f"image #{binned.id} ({binned.filename}) has no file in ")
    assert "delete it permanently from the recycle bin, or put the file back" in str(refused.value)
    storage.delete_image(binned.id)
    (store.settings.outputs_dir / live.filename).unlink()
    with pytest.raises(bundles.MissingImage, match="move it to the recycle bin and delete it permanently there"):
        bundles.preview()
