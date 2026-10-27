"""Shared-mode cutover over owned temporary paths/processes, never a live gallery."""
from __future__ import annotations

import asyncio
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
from uuid import uuid4

import pytest

from backend import bundle_file_lock, bundle_leases, bundle_mode as mode
from backend.bundle_leases import LeaseBusy, LeaseRegistry, LeasedFileResponse, ModeAwareRegistry

WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Windows fixed NTFS scratch qualification only")


@pytest.fixture(autouse=True)
def isolated_observations(monkeypatch):
    monkeypatch.setattr(mode, "_observed", {})
    monkeypatch.setattr(mode, "_named_observed", {})
    monkeypatch.setattr(mode, "_failed_directory_close", {})


def wire(directory):
    value = directory.stat()
    return {"format": mode._FORMAT, "version": 1, "installation_id": str(uuid4()),
            "directory": str(directory.resolve()), "device": value.st_dev, "inode": value.st_ino,
            "other_servers_stopped": "operator_confirmed", "machine_verified": False,
            "platform": "windows-fixed-ntfs"}


def marker(destination, **changes):
    value = wire(destination)
    value.update(changes)
    path = destination / mode.MARKER_NAME
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def scope():
    return {"type": "http", "method": "GET", "path": "/copy.zip", "headers": [],
            "http_version": "1.1", "scheme": "http", "query_string": b"",
            "extensions": {"http.response.pathsend": {}}}


async def receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def test_operator_confirmation_is_required_before_any_destination_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr(mode, "_qualified", lambda path: pytest.fail("unconfirmed cutover inspected destination"))
    for value in (False, None, 1, "yes"):
        with pytest.raises(mode.ModeError, match="operator-supplied"):
            mode.install(tmp_path / "absent", other_servers_stopped=value)
    assert list(tmp_path.iterdir()) == []


def test_absent_marker_downloads_are_portable_and_never_need_writing(tmp_path, monkeypatch):
    path = tmp_path / "copy.zip"
    path.write_bytes(b"synthetic archive bytes")

    def forbidden(*_args, **_kwargs):
        pytest.fail("default recovery download tried native qualification or a write")

    monkeypatch.setattr(mode, "_qualified", forbidden)
    monkeypatch.setattr(mode.tempfile, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(mode.os, "rename", forbidden)
    monkeypatch.setattr(mode.os, "fsync", forbidden)
    monkeypatch.setattr(bundle_file_lock, "acquire", forbidden)
    assert mode.discover(tmp_path) is None

    async def run():
        response = LeasedFileResponse(path)
        messages = []

        async def send(message):
            messages.append(message)

        await response(scope(), receive, send)
        assert messages[0]["status"] == 200
        assert b"".join(item.get("body", b"") for item in messages) == path.read_bytes()

    asyncio.run(run())
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", [
    {"format": "another application"}, {"version": 2}, {"version": True},
    {"directory": "C:\\foreign-destination"}, {"device": 0}, {"inode": 0},
    {"installation_id": "not-a-uuid"}, {"other_servers_stopped": True}, {"machine_verified": True},
    {"platform": "linux"}, {"unknown": "field"},
])
def test_bad_or_foreign_markers_refuse_without_overwrite_or_unlocked_fallback(tmp_path, monkeypatch, change):
    archive = tmp_path / "copy.zip"
    archive.write_bytes(b"unchanged archive")
    path = marker(tmp_path, **change)
    before = path.read_bytes()
    monkeypatch.setattr(mode, "_qualified", lambda path: None)
    monkeypatch.setattr(bundle_file_lock, "acquire", lambda *_args, **_kwargs: pytest.fail("invalid mode reached native acquisition"))
    with pytest.raises(mode.ModeError):
        ModeAwareRegistry().reader(archive)
    assert path.read_bytes() == before and archive.read_bytes() == b"unchanged archive"


@pytest.mark.parametrize("data", [b"{", b'{"version":1,"version":1}', b"x" * (65536 + 1)],
                         ids=["invalid_json", "duplicate_fields", "oversized"])
def test_marker_reads_are_strict_and_bounded(tmp_path, monkeypatch, data):
    path = tmp_path / mode.MARKER_NAME
    path.write_bytes(data)
    monkeypatch.setattr(mode, "_qualified", lambda path: None)
    with pytest.raises(mode.ModeError):
        mode.discover(tmp_path)
    assert path.read_bytes() == data


@pytest.mark.parametrize("changes", [
    {"st_mode": stat.S_IFLNK}, {"st_mode": stat.S_IFDIR}, {"st_nlink": 2},
    {"st_ino": 0}, {"st_file_attributes": stat.FILE_ATTRIBUTE_REPARSE_POINT},
])
def test_special_marker_identity_is_rejected(changes):
    value = dict(st_mode=stat.S_IFREG, st_nlink=1, st_dev=1, st_ino=2, st_size=1,
                 st_mtime_ns=1, st_ctime_ns=1, st_file_attributes=0)
    value.update(changes)
    with pytest.raises(mode.ModeError):
        mode._file_identity(SimpleNamespace(**value))


def test_observed_marker_cannot_disappear_or_be_replaced_into_default_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(mode, "_qualified", lambda path: None)
    path = marker(tmp_path)
    known = mode.discover(tmp_path)
    assert known and mode.discover(tmp_path) == known
    path.unlink()  # Owned synthetic marker, never an archive or user file.
    with pytest.raises(mode.ModeError, match="missing"):
        mode.discover(tmp_path)
    marker(tmp_path)
    with pytest.raises(mode.ModeError, match="changed"):
        mode.discover(tmp_path)


def test_known_named_destination_retarget_cannot_become_an_uninstalled_alias(tmp_path, monkeypatch):
    named, target = tmp_path / "known", tmp_path / "uninstalled"
    named.mkdir()
    target.mkdir()
    marker(named)
    monkeypatch.setattr(mode, "_qualified", lambda path: None)
    assert mode.discover(named)
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path: target if path == named else original(path))
    with pytest.raises(mode.ModeError, match="changed path"):
        ModeAwareRegistry().reader(named / "copy.zip")


def test_marker_access_error_is_preserved_without_retry_or_fallback(tmp_path, monkeypatch):
    path = marker(tmp_path)
    archive = tmp_path / "copy.zip"
    archive.write_bytes(b"synthetic")
    monkeypatch.setattr(mode, "_qualified", lambda path: None)
    original_open = Path.open
    calls = []
    denied = PermissionError(13, "injected marker read refusal", str(path))

    def fail(selected, *args, **kwargs):
        if selected == path:
            calls.append(selected)
            raise denied
        return original_open(selected, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(bundle_file_lock, "acquire", lambda *_args, **_kwargs: pytest.fail("unlocked/native fallback after marker refusal"))
    with pytest.raises(PermissionError) as caught:
        ModeAwareRegistry().reader(archive)
    assert caught.value is denied and calls == [path]


def test_cutover_refuses_local_readers_and_claims_from_every_registry(tmp_path):
    archive = tmp_path / "copy.zip"
    for method in ("reader", "exclusive"):
        with getattr(LeaseRegistry(), method)(archive):
            with pytest.raises(LeaseBusy, match="active"):
                with bundle_leases.installation_guard(tmp_path):
                    pytest.fail("cutover admitted a live local response/claim")


def test_cutover_blocks_new_local_streams_and_claims_until_its_finally(tmp_path):
    archive = tmp_path / "copy.zip"
    with bundle_leases.installation_guard(tmp_path):
        for registry in (LeaseRegistry(), ModeAwareRegistry()):
            for method in (registry.reader, registry.exclusive):
                with pytest.raises(LeaseBusy, match="being installed"):
                    method(archive)
    with LeaseRegistry().reader(archive):
        pass


@WINDOWS
def test_actual_windows_ntfs_installation_is_exclusive_idempotent_and_binds_destination(tmp_path):
    result = mode.install(tmp_path, other_servers_stopped=True)
    assert result.created and not result.machine_verified and not result.retention_enabled
    assert result.operator_confirmed_other_servers_stopped
    path = tmp_path / mode.MARKER_NAME
    before = path.read_bytes()
    again = mode.install(tmp_path, other_servers_stopped=True)
    assert not again.created and again.mode == result.mode and path.read_bytes() == before
    assert list(tmp_path.glob(".auradream.bundle-leases.*.part")) == []


@WINDOWS
def test_actual_install_refuses_local_active_response_before_creating_marker(tmp_path):
    archive = tmp_path / "copy.zip"
    archive.write_bytes(b"synthetic")
    with LeaseRegistry().reader(archive):
        with pytest.raises(LeaseBusy, match="active"):
            mode.install(tmp_path, other_servers_stopped=True)
    assert not (tmp_path / mode.MARKER_NAME).exists()
    assert list(tmp_path.iterdir()) == [archive]


@WINDOWS
def test_existing_bad_marker_is_never_overwritten_or_adopted(tmp_path, monkeypatch):
    path = marker(tmp_path, format="another application")
    original = path.read_bytes()
    monkeypatch.setattr(mode.tempfile, "NamedTemporaryFile", lambda **_kwargs: pytest.fail("bad marker reached publication"))
    with pytest.raises(mode.ModeError):
        mode.install(tmp_path, other_servers_stopped=True)
    assert path.read_bytes() == original and mode._observed == {}


@WINDOWS
def test_publication_cannot_claim_changed_contents_and_does_not_adopt_them(tmp_path, monkeypatch):
    real_rename = mode.os.rename

    def changed(source, target):
        # Same owned inode, different valid installation payload before rename.
        source.write_text(json.dumps(wire(tmp_path)), encoding="utf-8")
        return real_rename(source, target)

    monkeypatch.setattr(mode.os, "rename", changed)
    with pytest.raises(mode.ModeError, match="owned file and contents"):
        mode.install(tmp_path, other_servers_stopped=True)
    assert (tmp_path / mode.MARKER_NAME).is_file()
    assert mode._observed == {} and mode._named_observed == {}


@WINDOWS
def test_successful_publication_never_cleans_recreated_former_temporary_name(tmp_path, monkeypatch):
    real_rename = mode.os.rename
    recreated = []

    def claim(source, target):
        real_rename(source, target)
        source.write_bytes(b"another writer's new file")
        recreated.append(source)

    monkeypatch.setattr(mode.os, "rename", claim)
    assert mode.install(tmp_path, other_servers_stopped=True).created
    assert len(recreated) == 1 and recreated[0].read_bytes() == b"another writer's new file"


@WINDOWS
def test_existing_final_name_wins_publication_race_without_overwrite(tmp_path, monkeypatch):
    real_rename = mode.os.rename
    foreign = b"another application's existing marker"

    def collision(source, target):
        target.write_bytes(foreign)
        return real_rename(source, target)

    monkeypatch.setattr(mode.os, "rename", collision)
    with pytest.raises(FileExistsError):
        mode.install(tmp_path, other_servers_stopped=True)
    assert (tmp_path / mode.MARKER_NAME).read_bytes() == foreign
    assert list(tmp_path.glob(".auradream.bundle-leases.*.part")) == []


@WINDOWS
def test_cleanup_failure_does_not_mask_original_publication_failure(tmp_path, monkeypatch):
    original = OSError("injected marker publication failure")

    def fail(*_args):
        raise original

    def cleanup(*_args):
        raise PermissionError("injected installer cleanup refusal")

    monkeypatch.setattr(mode.os, "rename", fail)
    monkeypatch.setattr(mode, "_remove_owned", cleanup)
    with pytest.raises(OSError) as caught:
        mode.install(tmp_path, other_servers_stopped=True)
    assert caught.value is original and "cleanup refusal" in " ".join(original.__notes__)
    assert not (tmp_path / mode.MARKER_NAME).exists()


@WINDOWS
def test_directory_close_failure_retains_tombstone_without_masking_body_error(tmp_path, monkeypatch):
    original = OSError("injected installation body failure")
    real_close = mode.os.close
    held = []

    def close(fd):
        held.append(fd)
        raise PermissionError("injected directory close refusal")

    monkeypatch.setattr(mode.os, "close", close)
    try:
        with pytest.raises(OSError) as caught:
            with mode._pin_directory(tmp_path, mode._directory_identity(tmp_path)):
                raise original
        assert caught.value is original and "close refusal" in " ".join(original.__notes__)
        assert held and mode._failed_directory_close
        with pytest.raises(mode.ModeError, match="earlier"):
            with mode._pin_directory(tmp_path, mode._directory_identity(tmp_path)):
                pytest.fail("uncertain handle was implicitly retried")
    finally:
        for fd in held:
            real_close(fd)  # The injected close did not execute; exact owned fd.


@WINDOWS
def test_failed_raw_handle_transfer_and_cleanup_poison_future_install_attempts(tmp_path, monkeypatch):
    import msvcrt

    api = bundle_file_lock._windows_api()
    real_close = api.CloseHandle
    original = ValueError("injected handle transfer failure")
    handles = []

    def transfer(handle, _flags):
        handles.append(handle)
        raise original

    def close(_handle):
        ctypes.set_last_error(5)
        return False  # Injected; the native close was never called.

    monkeypatch.setattr(msvcrt, "open_osfhandle", transfer)
    monkeypatch.setattr(api, "CloseHandle", close)
    try:
        with pytest.raises(ValueError) as caught:
            with mode._pin_directory(tmp_path, mode._directory_identity(tmp_path)):
                pytest.fail("failed transfer reached installer body")
        assert caught.value is original and mode._failed_directory_close
        assert "CloseHandle" in " ".join(original.__notes__)
        with pytest.raises(mode.ModeError, match="earlier"):
            with mode._pin_directory(tmp_path, mode._directory_identity(tmp_path)):
                pytest.fail("uncertain raw handle was implicitly retried")
    finally:
        for handle in handles:
            assert real_close(handle)  # First real close, using the exact owned handle.


@WINDOWS
def test_fsync_failure_leaves_manual_archives_and_no_published_marker(tmp_path, monkeypatch):
    manual = tmp_path / "manual.zip"
    manual.write_bytes(b"unchanged manual archive")

    def fail(_fd):
        raise OSError("injected marker fsync failure")

    monkeypatch.setattr(mode.os, "fsync", fail)
    with pytest.raises(OSError, match="injected marker fsync"):
        mode.install(tmp_path, other_servers_stopped=True)
    assert not (tmp_path / mode.MARKER_NAME).exists()
    assert manual.read_bytes() == b"unchanged manual archive"
    assert list(tmp_path.glob(".auradream.bundle-leases.*.part")) == []


@WINDOWS
def test_installer_gate_spans_publication_and_response_constructed_before_cutover(tmp_path, monkeypatch):
    archive = tmp_path / "copy.zip"
    archive.write_bytes(b"synthetic")
    response = LeasedFileResponse(archive)  # Construction must not fix the old mode.
    real_rename = mode.os.rename
    entered, release = threading.Event(), threading.Event()
    outcomes = []

    def held(source, target):
        entered.set()
        assert release.wait(10)
        return real_rename(source, target)

    monkeypatch.setattr(mode.os, "rename", held)

    def install():
        try:
            outcomes.append(mode.install(tmp_path, other_servers_stopped=True))
        except BaseException as error:
            outcomes.append(error)

    owner = threading.Thread(target=install)
    owner.start()
    try:
        assert entered.wait(10)
        with pytest.raises(LeaseBusy, match="being installed"):
            bundle_leases.downloads.reader(archive)
    finally:
        release.set()
        owner.join(10)
    assert not owner.is_alive() and len(outcomes) == 1 and isinstance(outcomes[0], mode.Installation)

    async def run():
        async def send(message):
            with pytest.raises(LeaseBusy):
                LeaseRegistry(cross_process=True).exclusive(archive)
        await response(scope(), receive, send)

    asyncio.run(run())
    assert bundle_file_lock.companion_path(archive).is_file()


@WINDOWS
@pytest.mark.parametrize("drive,filesystem", [(4, "NTFS"), (2, "NTFS"), (3, "ReFS")])
def test_mapped_network_removable_and_unqualified_filesystems_refuse(tmp_path, monkeypatch, drive, filesystem):
    def root(_directory, output, _length):
        output.value = "Z:\\"
        return True

    def information(_root, _name, _n, _serial, _maximum, _flags, output, _length):
        output.value = filesystem
        return True

    monkeypatch.setattr(mode, "_volume_api", lambda: SimpleNamespace(
        GetVolumePathNameW=root, GetDriveTypeW=lambda _root: drive, GetVolumeInformationW=information,
    ))
    with pytest.raises(mode.UnsupportedMode):
        mode._qualified(tmp_path)


_CHILD = r'''
import asyncio, json, sys
from pathlib import Path
from backend.bundle_leases import LeasedFileResponse
archive=Path(sys.argv[1])
scope={"type":"http","method":"GET","path":"/copy.zip","headers":[],"http_version":"1.1","extensions":{"http.response.pathsend":{}}}
async def receive(): return {"type":"http.request","body":b"","more_body":False}
async def send(message):
    if message["type"]=="http.response.start":
        print(json.dumps({"phase":"response_started","status":message["status"]}),flush=True)
        await asyncio.to_thread(sys.stdin.readline)
asyncio.run(LeasedFileResponse(archive)(scope,receive,send))
print(json.dumps({"phase":"response_finished"}),flush=True)
'''


def line(process, seconds=15):
    results = queue.Queue()
    threading.Thread(target=lambda: results.put(process.stdout.readline()), daemon=True).start()
    try:
        value = results.get(timeout=seconds)
    except queue.Empty:
        raise AssertionError("owned response child did not deliver its handshake") from None
    assert value, "owned response child exited before its handshake"
    return json.loads(value)


@WINDOWS
def test_fresh_owned_process_discovers_marker_and_holds_native_lease_for_asgi_lifetime(tmp_path):
    archive = tmp_path / "copy.zip"
    archive.write_bytes(b"owned cohort response")
    installed = mode.install(tmp_path, other_servers_stopped=True)
    # Only this test-owned cohort is attested; no host process was inventoried.
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(archive)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", cwd=Path(__file__).resolve().parent.parent)
    receipt_path = tmp_path / "cohort-receipt.json"
    receipt = {"installation_id": installed.mode.installation_id, "owned_child_pid": child.pid,
               "archive": str(archive), "state": "spawned", "host_quiescence_verified": False,
               "retention_enabled": False}
    events = []
    receipt_started = False
    passed = False
    try:
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_started = True
        events.append(line(child))
        assert events[-1] == {"phase": "response_started", "status": 200}
        with pytest.raises(LeaseBusy):
            LeaseRegistry(cross_process=True).exclusive(archive)
        child.stdin.write("finish\n")
        child.stdin.flush()
        events.append(line(child))
        assert events[-1]["phase"] == "response_finished"
        assert child.wait(timeout=15) == 0
        stderr = child.stderr.read()
        assert stderr == ""
        with LeaseRegistry(cross_process=True).exclusive(archive):
            pass
        passed = True
    finally:
        if child.poll() is None:
            child.stdin.close()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream and not stream.closed:
                stream.close()
        if receipt_started:
            receipt_path.write_text(json.dumps({**receipt, "events": events, "state": "finished",
                                               "passed": passed, "owned_child_exit": child.returncode}), encoding="utf-8")


def test_marker_name_and_format_are_frozen_from_the_old_name():
    # Installed destinations and their receipts predate the rename to Auradream.
    assert mode.MARKER_NAME == "local-diffusion.bundle-leases.json"
    assert mode._FORMAT == "local-diffusion.shared-bundle-leases"
