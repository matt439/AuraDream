"""GAL#29: a permanently deleted image's file that was held open is retried,
remembered and removed once released, so no bundle carries it."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from zipfile import ZipFile

import pytest

from backend import bundles, deleted_files, gallery_delete, gallery_lock, sidecar_io
from tests.bundle_helpers import save_source


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    # conftest empties the remembered files and holds back the exit hook.
    monkeypatch.setattr(sidecar_io, "WINDOWS", True)
    monkeypatch.setattr(sidecar_io.time, "sleep", lambda _s: None)


def sharing_violation() -> PermissionError:
    error = PermissionError(13, "The process cannot access the file")
    error.winerror = 32  # type: ignore[misc]
    return error


@pytest.fixture
def hold(monkeypatch):
    """Paths whose unlink fails as a held file's does, until released."""
    held: set[Path] = set()
    original = Path.unlink

    def unlink(self, *args, **kwargs):
        if self in held:
            raise sharing_violation()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    return held


def files_of(store, filename: str) -> list[Path]:
    png = store.settings.outputs_dir / filename
    return [png, store._thumb_path(png)]


def test_a_held_png_is_removed_once_released_and_no_bundle_carries_it(store, hold):
    image = save_source(seed=1)
    png, thumb = files_of(store, image.filename)
    assert png.exists() and thumb.exists()
    hold.add(png)
    assert gallery_delete.delete_image(image.id)
    assert png.exists() and not thumb.exists()
    assert deleted_files.pending() == [png]
    hold.clear()  # the stream ends
    names = {name for name, _ in bundles._inventory()}
    assert "images/" + image.filename not in names
    assert not png.exists() and deleted_files.pending() == []


def test_the_next_permanent_delete_sweeps_a_released_file(store, hold):
    first, second = save_source(seed=1), save_source(seed=2)
    png = files_of(store, first.filename)[0]
    hold.add(png)
    store.bin_images([first.id, second.id])
    result = gallery_delete.purge_images([first.id], [first.id])
    assert result["files"] == 0 and png.exists()
    hold.clear()
    gallery_delete.purge_images([second.id], [second.id])
    assert not png.exists() and deleted_files.pending() == []


def test_a_brief_hold_is_ridden_out_without_remembering(store, monkeypatch):
    image = save_source(seed=1)
    png = files_of(store, image.filename)[0]
    original = Path.unlink
    failures = [sharing_violation(), sharing_violation()]

    def flaky(self, *args, **kwargs):
        if self == png and failures:
            raise failures.pop()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky)
    store.bin_images([image.id])
    assert gallery_delete.empty_bin()["files"] == 1
    assert not png.exists() and deleted_files.pending() == []


def test_a_file_a_row_names_again_is_never_swept(store, hold):
    image = save_source(seed=1)
    png = files_of(store, image.filename)[0]
    hold.add(png)
    gallery_delete.delete_image(image.id)
    hold.clear()
    # A bundle import (or a reindex) gives the image a row again.
    deleted_files._pending[png] = "kept.png"
    with store._session() as conn:
        conn.execute("UPDATE images SET filename = 'kept.png' WHERE id = ?", (save_source(seed=2).id,))
    assert deleted_files.sweep() == 0
    assert png.exists() and deleted_files.pending() == []


def test_a_file_still_held_stays_remembered(store, hold):
    image = save_source(seed=1)
    png = files_of(store, image.filename)[0]
    hold.add(png)
    gallery_delete.delete_image(image.id)
    assert deleted_files.sweep() == 0
    assert png.exists() and deleted_files.pending() == [png]


@pytest.mark.parametrize("released", ["never", "between the inventories"])
def test_bundle_and_clear_leaves_out_a_held_file_and_still_clears(store, hold, monkeypatch, released):
    """The review's case: released while the archive was written, the second
    inventory's sweep removed a file the first had listed, and the clear was
    refused as an outside change with the deleted image in the archive."""
    gone, kept = save_source(seed=1), save_source(seed=2)
    png, thumb = files_of(store, gone.filename)
    hold.update({png, thumb})
    gallery_delete.delete_image(gone.id)
    assert png.exists() and thumb.exists()
    expected = bundles.preview()["records"]  # before the wrapper: the export's own calls count
    if released != "never":
        original, calls = bundles._inventory, []

        def inventory(*args, **kwargs):
            # The export's preview, its archive's inventory, then the check
            # after writing: the holder lets go before that last one.
            if len(calls) == 2:
                hold.clear()
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(bundles, "_inventory", inventory)
    archive = bundles.export_bundle(expected=expected)
    assert archive["cleared"] is True and not archive.get("cleanup_error")
    with ZipFile(archive["path"]) as bundle:
        names = bundle.namelist()
    assert "images/" + kept.filename in names
    assert not any(gone.filename.removesuffix(".png") in name for name in names)
    assert png.exists() == (released == "never")


def test_a_failed_sweep_never_fails_the_delete_it_follows(store, hold, monkeypatch, caplog):
    first, second = save_source(seed=1), save_source(seed=2)
    hold.add(files_of(store, first.filename)[0])
    gallery_delete.delete_image(first.id)
    original = store._session
    calls = []

    def failing_second(*args, **kwargs):
        # The delete's own session works, the sweep's after it does not,
        # and the mask sweep's after that works again.
        calls.append(1)
        if len(calls) == 2:
            raise gallery_lock.GalleryBusy("busy")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_session", failing_second)
    assert gallery_delete.delete_image(second.id) is True
    assert "not swept" in caplog.text
    assert len(deleted_files.pending()) == 1


def test_the_exit_sweep_never_waits_behind_a_running_bundle(store, hold):
    image = save_source(seed=1)
    png = files_of(store, image.filename)[0]
    hold.add(png)
    gallery_delete.delete_image(image.id)
    hold.clear()
    done = threading.Event()
    release = threading.Event()

    def bundle_running():
        with gallery_lock.hold(store.settings.db_path, wait=30):
            done.set()
            release.wait(30)

    worker = threading.Thread(target=bundle_running)
    worker.start()
    assert done.wait(30)
    started = time.monotonic()
    deleted_files._sweep_at_exit()  # no traceback, no 30 s wait
    assert time.monotonic() - started < 5
    assert png.exists() and deleted_files.pending() == [png]
    release.set()
    worker.join(30)
    deleted_files._sweep_at_exit()
    assert not png.exists() and deleted_files.pending() == []


@pytest.mark.skipif(os.name != "nt", reason="a held file blocks its unlink on Windows only")
def test_a_real_open_handle_on_windows(store, monkeypatch):
    monkeypatch.setattr(sidecar_io, "WINDOWS", True)
    image = save_source(seed=1)
    png = files_of(store, image.filename)[0]
    with png.open("rb"):
        assert gallery_delete.delete_image(image.id)
        assert png.exists() and deleted_files.pending() == [png]
    assert deleted_files.sweep() == 1
    assert not png.exists()
