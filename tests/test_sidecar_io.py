"""GAL#28: the sidecars' reads and atomic replaces ride out a Windows
sharing violation instead of refusing the read or the save."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import pytest

from backend import gallery_lock, model_favourites, sidecar_io

ROOT = Path(__file__).resolve().parents[1]
SIDECARS = (
    "model_favourites", "snippets", "presets", "saved_searches",
    "startup_preferences", "backup_settings", "backup_receipts",
)


def sharing_violation(winerror: int = 32) -> PermissionError:
    error = PermissionError(13, "The process cannot access the file")
    error.winerror = winerror  # type: ignore[misc]
    return error


@pytest.fixture(autouse=True)
def windows(monkeypatch):
    """The retry is Windows behaviour; these tests claim it everywhere."""
    monkeypatch.setattr(sidecar_io, "WINDOWS", True)


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(sidecar_io.time, "sleep", slept.append)
    return slept


def open_refusal() -> PermissionError:
    """What `open` raises on Windows during a replace: EACCES, no winerror."""
    return PermissionError(13, "Permission denied")


@pytest.mark.parametrize("refusal", [open_refusal, sharing_violation])
def test_a_read_retries_a_sharing_violation(tmp_path, monkeypatch, no_sleep, refusal):
    path = tmp_path / "side.json"
    path.write_bytes(b"{}")
    original = Path.open
    failures = [refusal(), refusal()]

    def flaky(self, *args, **kwargs):
        if self == path and failures:
            raise failures.pop()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky)
    with sidecar_io.open_read(path) as stream:
        assert stream.read() == b"{}"
    assert len(no_sleep) == 2


def test_a_replace_retries_a_sharing_violation(tmp_path, monkeypatch, no_sleep):
    source, target = tmp_path / "new.tmp", tmp_path / "side.json"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    original = os.replace
    failures = [sharing_violation()]

    def flaky(src, dst):
        if failures:
            raise failures.pop()
        original(src, dst)

    monkeypatch.setattr(sidecar_io.os, "replace", flaky)
    sidecar_io.replace(source, target)
    assert target.read_bytes() == b"new" and not source.exists()
    assert len(no_sleep) == 1


def test_a_held_file_gives_up_after_a_bounded_wait(tmp_path, monkeypatch, no_sleep):
    def held(*_args):
        raise sharing_violation()

    monkeypatch.setattr(sidecar_io.os, "replace", held)
    with pytest.raises(PermissionError):
        sidecar_io.replace(tmp_path / "a", tmp_path / "b")
    assert no_sleep == list(sidecar_io.RETRY_DELAYS)
    assert sum(sidecar_io.RETRY_DELAYS) < 1


@pytest.mark.parametrize("error", [
    PermissionError("read-only"),  # no errno: a refusal a wait does not clear
    sharing_violation(winerror=13),
    OSError("disk"),
])
def test_other_errors_are_raised_at_once(tmp_path, monkeypatch, no_sleep, error):
    def refuse(*_args):
        raise error

    monkeypatch.setattr(sidecar_io.os, "replace", refuse)
    with pytest.raises(type(error)):
        sidecar_io.replace(tmp_path / "a", tmp_path / "b")
    assert no_sleep == []


def test_nothing_is_retried_off_windows(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setattr(sidecar_io, "WINDOWS", False)

    def refuse(*_args):
        raise open_refusal()

    monkeypatch.setattr(sidecar_io.os, "replace", refuse)
    with pytest.raises(PermissionError):
        sidecar_io.replace(tmp_path / "a", tmp_path / "b")
    assert no_sleep == []


@pytest.mark.parametrize("module", SIDECARS)
def test_every_sidecar_reads_and_publishes_through_the_helper(module):
    source = (ROOT / "backend" / f"{module}.py").read_text(encoding="utf-8")
    assert "sidecar_io.open_read(" in source and "sidecar_io.replace(" in source
    assert not re.search(r"\bos\.replace\(", source), module
    assert not re.search(r"\.open\(\s*\"rb\"", source), module


def test_a_read_waits_for_this_process_save_and_creates_no_file(tmp_path):
    database = tmp_path / "gallery.db"
    path = model_favourites.path_for(database)
    saving, done = threading.Event(), threading.Event()
    answers: list[object] = []

    def read():
        saving.wait(30)
        answers.append(model_favourites.read(database))
        done.set()

    reader = threading.Thread(target=read)
    reader.start()
    with gallery_lock.hold(path, create=False, wait=30):
        saving.set()
        assert not done.wait(0.3)  # held behind the save's reservation
    assert done.wait(30)
    reader.join(30)
    assert answers == [([], 0)]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="the sharing violation is Windows behaviour")
def test_a_star_and_a_read_overlapping_on_windows_both_succeed(tmp_path):
    """The probe that found it (78 of 20,000 reads refused, and saves too)."""
    database = tmp_path / "gallery.db"
    model_favourites.change(database, "a/one", True)
    errors: list[BaseException] = []
    stop = threading.Event()

    def toggle():
        favourite = False
        while not stop.is_set():
            try:
                model_favourites.change(database, "a/two", favourite)
            except BaseException as exc:  # noqa: BLE001 - the test reports them
                errors.append(exc)
            favourite = not favourite

    writer = threading.Thread(target=toggle, daemon=True)
    writer.start()
    deadline = time.monotonic() + 3
    reads = 0
    try:
        while time.monotonic() < deadline:
            try:
                model_favourites.read(database)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            reads += 1
    finally:
        stop.set()
        writer.join(30)
    assert reads > 10  # reads wait behind saves now; a loaded runner reads fewer
    assert errors == []
