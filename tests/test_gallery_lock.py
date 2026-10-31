"""The gallery reservation across processes and threads."""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from backend import gallery_lock, storage
from tests.bundle_helpers import holding_child


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


def test_a_waiting_session_gets_the_gallery_once_another_process_lets_go(store):
    child = holding_child(store.settings.db_path)
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


def test_maintenance_rides_out_another_processs_brief_read(store):
    # A reader holds the file lock for a millisecond or two - every open
    # tab's status poll reads the newest id - and a maintenance call whose
    # one attempt landed inside it was refused with the server idle.
    child = holding_child(store.settings.db_path)
    try:
        threading.Timer(0.05, lambda: (child.stdin.write("\n"), child.stdin.flush())).start()
        started = time.monotonic()
        with gallery_lock.hold(store.settings.db_path):
            waited = time.monotonic() - started
        assert waited < gallery_lock.MAINTENANCE_GRACE + 0.2
    finally:
        child.stdin.close()
        child.wait(timeout=15)


def test_a_waiting_session_refuses_once_its_deadline_passes(store, monkeypatch):
    child = holding_child(store.settings.db_path)
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
    child = holding_child(store.settings.db_path)
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


def _sidecar_writers():
    from backend import presets, saved_searches, snippets
    from backend.schemas_presets import PresetFields, PresetPayload
    from backend.schemas_saved_searches import SavedSearchFields
    from backend.schemas_snippets import SnippetFields
    return [
        (presets, lambda db: presets.change(db, PresetFields(
            name="setup", model_id="model", payload=PresetPayload(prompt="a cat"),
        ))),
        (snippets, lambda db: snippets.change(db, SnippetFields(name="snippet", text="a cat"))),
        (saved_searches, lambda db: saved_searches.change(db, SavedSearchFields(
            name="search", query="", favorites=True, all_models=False,
        ))),
    ]


@pytest.mark.parametrize("index", range(3), ids=["presets", "snippets", "saved_searches"])
def test_a_sidecar_write_waits_out_another_processs_write(tmp_path, index):
    # Two app processes saving the same sidecar at once: the second used
    # maintenance's MAINTENANCE_GRACE (0.25 s) and was refused as busy when
    # the first's read-validate-fsync-replace ran longer under load. A
    # single-record write waits SESSION_WAIT, as the gallery's do.
    module, write = _sidecar_writers()[index]
    database = tmp_path / "gallery.db"
    child = holding_child(module.path_for(database))
    outcome = []
    writer = threading.Thread(target=lambda: outcome.append(write(database)))
    try:
        writer.start()
        # Far past the maintenance grace, and the holder has not let go.
        writer.join(timeout=4 * gallery_lock.MAINTENANCE_GRACE + 1.0)
        assert writer.is_alive() and not outcome
    finally:
        child.stdin.write("\n")
        child.stdin.flush()
        child.stdin.close()
        child.wait(timeout=15)
    writer.join(timeout=15)
    assert not writer.is_alive() and len(outcome) == 1 and len(outcome[0]) == 1
