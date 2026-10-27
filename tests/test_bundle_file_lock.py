"""Windows scratch lock acceptance and injected path/native failure contracts.

No archive/companion is removed, no permission is changed and no inventory
outside these temporary paths is inspected. Subprocesses are owned Popen
handles with bounded receipt/pipe handshakes, never discovered processes.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import queue
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from backend import bundle_file_lock as locks
from backend.bundle_leases import LeaseBusy, LeaseRegistry


WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Windows scratch acceptance; Linux branch is unqualified")


@pytest.mark.parametrize("name", ["CON.zip", "copy.zip:stream", "bad.zip.", "bad.zip ", "bad\n.zip"])
def test_unsafe_basename_refuses_before_open(tmp_path, monkeypatch, name):
    monkeypatch.setattr(locks, "_open", lambda path: pytest.fail("unsafe lease reached open"))
    with pytest.raises(locks.LockSafetyError, match="basename"):
        locks.acquire(tmp_path / name, exclusive=False)


@pytest.mark.parametrize("changes", [
    {"st_mode": stat.S_IFDIR}, {"st_mode": stat.S_IFLNK}, {"st_nlink": 2},
    {"st_ino": 0}, {"st_file_attributes": stat.FILE_ATTRIBUTE_REPARSE_POINT},
])
def test_special_linked_or_uncertain_companion_identity_is_refused(tmp_path, changes):
    metadata = dict(st_mode=stat.S_IFREG, st_nlink=1, st_dev=1, st_ino=2, st_file_attributes=0)
    metadata.update(changes)
    with pytest.raises(locks.LockSafetyError):
        locks._regular(SimpleNamespace(**metadata), tmp_path / "copy.zip.lease.lock")


def test_directory_companion_is_not_opened_or_modified(tmp_path, monkeypatch):
    archive = tmp_path / "copy.zip"
    archive.write_bytes(b"archive")
    companion = locks.companion_path(archive)
    companion.mkdir()
    monkeypatch.setattr(locks, "_open", lambda path: pytest.fail("directory companion reached open"))
    with pytest.raises(locks.LockSafetyError):
        locks.acquire(archive, exclusive=True)
    assert archive.read_bytes() == b"archive" and companion.is_dir()


def test_archive_and_parent_links_refuse_before_resolution_or_open(tmp_path, monkeypatch):
    archive = tmp_path / "copy.zip"
    original = Path.lstat
    monkeypatch.setattr(Path, "lstat", lambda path: SimpleNamespace(st_mode=stat.S_IFLNK)
                        if path == archive.parent else original(path))
    monkeypatch.setattr(locks, "_open", lambda path: pytest.fail("linked parent reached open"))
    with pytest.raises(locks.LockSafetyError, match="directory"):
        locks.acquire(archive, exclusive=False)


def test_open_denial_preserves_the_exact_original_error_without_retry(tmp_path, monkeypatch):
    original = PermissionError(13, "injected companion open denial", str(tmp_path / "copy.zip.lease.lock"))
    calls = []

    def denied(path):
        calls.append(path)
        raise original

    monkeypatch.setattr(locks, "_open", denied)
    with pytest.raises(PermissionError) as caught:
        locks.acquire(tmp_path / "copy.zip", exclusive=False)
    assert caught.value is original and calls == [tmp_path / "copy.zip.lease.lock"]


def test_changed_opened_identity_closes_only_its_fd_before_locking(tmp_path, monkeypatch):
    archive = tmp_path / "copy.zip"
    companion = locks.companion_path(archive)
    companion.write_bytes(b"unchanged")
    fd = os.open(companion, os.O_RDWR)
    real_fstat = os.fstat
    real_close = os.close
    closed = []
    monkeypatch.setattr(locks, "_open", lambda path: fd)

    def changed(handle):
        value = real_fstat(handle)
        return SimpleNamespace(st_mode=value.st_mode, st_dev=value.st_dev, st_ino=value.st_ino + 1,
                               st_nlink=1, st_file_attributes=0)

    monkeypatch.setattr(os, "fstat", changed)
    monkeypatch.setattr(os, "close", lambda handle: (closed.append(handle), real_close(handle))[-1])
    monkeypatch.setattr(locks, "_lock", lambda *args: pytest.fail("changed identity reached native lock"))
    with pytest.raises(locks.LockSafetyError, match="changed"):
        locks.acquire(archive, exclusive=False)
    assert closed == [fd] and companion.read_bytes() == b"unchanged"


@pytest.mark.parametrize("unlock_fails, close_fails", [(True, False), (False, True), (True, True)])
def test_release_attempts_close_after_unlock_error_and_keeps_uncertain_ownership(
    tmp_path, monkeypatch, unlock_fails, close_fails,
):
    archive = tmp_path / "copy.zip"
    companion = locks.companion_path(archive)
    companion.write_bytes(b"")
    fd = os.open(companion, os.O_RDWR)
    owned = locks.FileLock(archive, companion, fd, locks._identity(tmp_path.stat()))
    owned._locked = True
    events = []
    real_close = os.close

    def unlock(*args):
        events.append("unlock")
        if unlock_fails:
            raise OSError("injected unlock failure")

    def close(handle):
        events.append("close")
        if close_fails:
            raise PermissionError("injected close failure")
        real_close(handle)

    monkeypatch.setattr(locks, "_unlock", unlock)
    monkeypatch.setattr(os, "close", close)
    try:
        with pytest.raises(OSError) as caught:
            owned.release()
        assert events == ["unlock", "close"]
        assert owned.closed is not close_fails
        if unlock_fails and close_fails:
            assert "also failed" in " ".join(caught.value.__notes__)
        if owned.closed:
            owned.release()
            assert events == ["unlock", "close"]
    finally:
        if not owned.closed:
            # Only an injected close failure left this test-owned fd open.
            real_close(fd)


@WINDOWS
def test_windows_flags_and_fixed_range_are_shared_or_exclusive_without_waiting(tmp_path, monkeypatch):
    import msvcrt

    events = []

    def call(*args):
        overlapped = ctypes.cast(args[-1], ctypes.POINTER(locks._Overlapped)).contents
        assert (overlapped.Internal, overlapped.InternalHigh, overlapped.Offset,
                overlapped.OffsetHigh, overlapped.hEvent) == (0, 0, 0, 0, None)
        events.append(args[:-1])
        return True

    monkeypatch.setattr(locks, "_windows_api", lambda: SimpleNamespace(LockFileEx=call, UnlockFileEx=call))
    monkeypatch.setattr(msvcrt, "get_osfhandle", lambda fd: 123)
    locks._lock(7, tmp_path / "copy.zip.lease.lock", False)
    locks._lock(7, tmp_path / "copy.zip.lease.lock", True)
    locks._unlock(7, tmp_path / "copy.zip.lease.lock")
    assert events == [(123, 1, 0, 1, 0), (123, 3, 0, 1, 0), (123, 0, 1, 0)]


@WINDOWS
@pytest.mark.parametrize("code, expected", [(33, locks.LockContended), (5, PermissionError), (32, OSError)])
def test_windows_only_lock_violation_is_contention(tmp_path, monkeypatch, code, expected):
    import msvcrt

    monkeypatch.setattr(locks, "_windows_api", lambda: SimpleNamespace(LockFileEx=lambda *args: False))
    monkeypatch.setattr(msvcrt, "get_osfhandle", lambda fd: 123)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: code)
    with pytest.raises(expected) as caught:
        locks._lock(7, tmp_path / "copy.zip.lease.lock", False)
    if code != 33:
        assert not isinstance(caught.value, locks.LockContended)
        assert caught.value.winerror == code
        assert "LockFileEx lease companion" in " ".join(caught.value.__notes__)


@WINDOWS
def test_real_windows_companion_is_empty_persistent_noninheritable_and_preserves_archive(tmp_path):
    archive = tmp_path / "copy.zip"
    archive.write_bytes(b"archive stays unchanged")
    first = locks.acquire(archive, exclusive=False)
    try:
        companion = first.path
        identity = locks._identity(companion.stat())
        assert companion.read_bytes() == b"" and not os.get_inheritable(first._fd)
        first.validate()
    finally:
        first.release()
    first.release()
    second = locks.acquire(archive, exclusive=True)
    try:
        assert locks._identity(companion.stat()) == identity
    finally:
        second.release()
    assert companion.exists() and companion.read_bytes() == b""
    assert archive.read_bytes() == b"archive stays unchanged"


_CHILD = """
import json, os, sys
from pathlib import Path
from backend.bundle_leases import LeaseBusy, LeaseRegistry
registry = LeaseRegistry(cross_process=True)
try:
    lease = (registry.exclusive if sys.argv[2] == 'exclusive' else registry.reader)(Path(sys.argv[1]))
except LeaseBusy:
    print('busy', flush=True)
else:
    print('held', flush=True)
    if sys.stdin.readline().strip() == 'crash':
        os._exit(0)
    lease.release()
    print('released', flush=True)
"""


@contextmanager
def _child(path, mode, receipt):
    process = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(path), mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    receipt.write_text(json.dumps({"state": "spawned", "pid": process.pid, "mode": mode,
                                   "archive": str(path)}), encoding="utf-8")
    messages = queue.Queue()
    threading.Thread(target=lambda: messages.put(process.stdout.readline()), daemon=True).start()
    try:
        line = messages.get(timeout=15).strip()
        if line not in {"held", "busy"}:
            output, errors = process.communicate(timeout=15)
            pytest.fail(f"Owned child failed: {line!r} {output} {errors}")
        receipt.write_text(json.dumps({"state": line, "pid": process.pid, "mode": mode}), encoding="utf-8")
        yield process, line
    finally:
        if process.poll() is None:
            try:
                output, errors = process.communicate("release\n", timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()  # Only this owned scratch helper.
                output, errors = process.communicate(timeout=15)
                pytest.fail(f"Owned child handshake timed out: {output} {errors}")
        else:
            output, errors = process.communicate(timeout=15)
        receipt.write_text(json.dumps({"state": "finished", "pid": process.pid, "returncode": process.returncode,
                                       "stdout": output, "stderr": errors}), encoding="utf-8")
        assert process.returncode == 0, output + errors


@WINDOWS
def test_windows_process_readers_share_and_exclusive_waits_for_the_last_reader(tmp_path):
    archive = tmp_path / "copy.zip"
    parent = LeaseRegistry(cross_process=True)
    one = parent.reader(archive)
    try:
        with _child(archive, "reader", tmp_path / "reader-child.json") as (_process, outcome):
            assert outcome == "held"
            with _child(archive, "exclusive", tmp_path / "claim-child.json") as (_probe, claim):
                assert claim == "busy"
            one.release()
            with pytest.raises(LeaseBusy):
                parent.exclusive(archive)
    finally:
        one.release()
    with parent.exclusive(archive):
        pass


@WINDOWS
def test_windows_mixed_mode_keeps_shared_handle_until_default_reader_releases(tmp_path):
    archive = tmp_path / "copy.zip"
    default = LeaseRegistry().reader(archive)
    shared = LeaseRegistry(cross_process=True).reader(archive)
    try:
        shared.release()
        with _child(archive, "exclusive", tmp_path / "mixed-mode-child.json") as (_process, outcome):
            assert outcome == "busy"
    finally:
        shared.release()
        default.release()
    with _child(archive, "exclusive", tmp_path / "mixed-mode-released-child.json") as (_process, outcome):
        assert outcome == "held"


@WINDOWS
@pytest.mark.parametrize("mode", ["reader", "exclusive"])
def test_windows_exclusive_claim_blocks_other_processes(tmp_path, mode):
    archive = tmp_path / "copy.zip"
    with LeaseRegistry(cross_process=True).exclusive(archive):
        with _child(archive, mode, tmp_path / "claim-probe.json") as (_process, outcome):
            assert outcome == "busy"


@WINDOWS
def test_windows_process_exit_releases_lock_without_deleting_companion(tmp_path):
    archive = tmp_path / "copy.zip"
    with _child(archive, "reader", tmp_path / "crash-child.json") as (process, outcome):
        assert outcome == "held"
        companion = locks.companion_path(archive)
        identity = locks._identity(companion.stat())
        with pytest.raises(LeaseBusy):
            LeaseRegistry(cross_process=True).exclusive(archive)
        process.stdin.write("crash\n")
        process.stdin.flush()
        process.wait(timeout=15)
    with LeaseRegistry(cross_process=True).exclusive(archive):
        assert locks._identity(companion.stat()) == identity
    assert companion.exists() and companion.read_bytes() == b""
