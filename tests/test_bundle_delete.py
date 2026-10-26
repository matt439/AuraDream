"""Unconnected deletion mechanism, using only owned Windows scratch archives."""
from __future__ import annotations

import ctypes
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend import backup_receipts as receipts, bundle_delete as deletion, bundle_leases, bundle_mode

WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Windows fixed-local-NTFS scratch mechanism only")


@pytest.fixture(autouse=True)
def observations(monkeypatch):
    monkeypatch.setattr(bundle_mode, "_observed", {})
    monkeypatch.setattr(bundle_mode, "_named_observed", {})
    monkeypatch.setattr(deletion, "_quarantined", [])
    monkeypatch.setattr(bundle_leases, "_retained_exclusives", [])


@pytest.fixture
def candidate(tmp_path):
    directory = tmp_path / "owned-deletion-destination"
    directory.mkdir()
    archive = directory / "owned-copy.zip"
    archive.write_bytes(b"only this scratch archive may be deleted\n" * 4096)
    identity = receipts.ManagedIdentity(*(str(uuid4()) for _ in range(3)))
    receipt = receipts.Receipt(identity, str(directory.resolve()), archive.name,
                               hashlib.sha256(archive.read_bytes()).hexdigest(),
                               receipts.FileStamp.of(archive.lstat()), datetime.now(timezone.utc),
                               "unencrypted", 1, 1)
    installed = bundle_mode.install(directory, other_servers_stopped=True).mode
    return archive, receipt, installed


def proxy_api(monkeypatch, **changes):
    original = deletion._windows_api()
    names = ("CreateFileW", "CloseHandle", "GetFileInformationByHandleEx", "SetFileInformationByHandle")
    proxy = SimpleNamespace(**{name: changes.get(name, getattr(original, name)) for name in names})
    monkeypatch.setattr(deletion, "_windows_api", lambda: proxy)
    return original


def recover_injected_fault(bound, real_close):
    """Only for our injected no-op close/mark faults, never a real OS denial."""
    for fd in tuple(bound._fds.values()):
        real_close(fd)
    bound._fds.clear()
    bound._claim._retained = None
    bound._claim.__exit__(None, None, None)
    bound._closed = True


def test_disposition_abi_is_one_boolean_byte():
    value = deletion._DispositionInfo(1)
    assert ctypes.sizeof(value) == 1 and bytes(value) == b"\x01"


def test_expected_mode_cannot_fall_back_to_an_uninstalled_destination(tmp_path, monkeypatch):
    path = tmp_path / "copy.zip"
    path.write_bytes(b"unmanaged")
    expected = bundle_mode.SharedMode(str(uuid4()), tmp_path.resolve(), (1, 2), (1, 2, 3, 4, 5), "0" * 64)
    monkeypatch.setattr(bundle_leases.bundle_file_lock, "acquire", lambda *_a, **_k: pytest.fail("missing mode acquired a native claim"))
    with pytest.raises(bundle_mode.ModeError, match="expected installed"):
        with bundle_leases.exclusive_for_mode(path, expected):
            pytest.fail("missing mode was accepted")
    assert list(tmp_path.iterdir()) == [path]


@WINDOWS
@pytest.mark.parametrize("field", ["installation_id", "directory_identity", "marker_identity", "digest"])
def test_full_expected_mode_must_match_before_archive_open(candidate, monkeypatch, field):
    path, receipt, installed = candidate
    altered = {"installation_id": str(uuid4()), "directory_identity": (0, 1),
               "marker_identity": (0, 1, 2, 3, 4), "digest": "0" * 64}[field]
    monkeypatch.setattr(deletion.BoundArchive, "_open", lambda *_a: pytest.fail("foreign mode opened a mutation input"))
    with pytest.raises(bundle_mode.ModeError):
        with deletion.bind_managed_archive(receipt, replace(installed, **{field: altered}), receipt.identity):
            pytest.fail("foreign installed mode accepted")
    assert path.exists() and not path.with_name(path.name + ".lease.lock").exists()


@WINDOWS
@pytest.mark.parametrize("field", ["gallery_id", "destination_id", "operation_id"])
def test_entire_expected_managed_identity_is_required_before_native_claim(candidate, monkeypatch, field):
    path, receipt, installed = candidate
    monkeypatch.setattr(bundle_leases, "exclusive_for_mode", lambda *_a: pytest.fail("wrong identity acquired a claim"))
    with pytest.raises(deletion.DeleteSafetyError, match="intended identity"):
        deletion.bind_managed_archive(receipt, installed, replace(receipt.identity, **{field: str(uuid4())}))
    assert path.exists()


@WINDOWS
def test_bind_without_mark_is_nonmutating_and_blocks_local_readers(candidate):
    path, receipt, installed = candidate
    before = path.read_bytes()
    with deletion.bind_managed_archive(receipt, installed, receipt.identity) as bound:
        with pytest.raises(bundle_leases.LeaseBusy):
            bundle_leases.ModeAwareRegistry().reader(path)
        assert bound.outcome == deletion.DeletionOutcome("not_requested", False)
    assert bound.outcome == deletion.DeletionOutcome("not_requested", True)
    assert path.read_bytes() == before
    bound.close()


@WINDOWS
def test_exclusive_capability_requires_its_original_acquired_path(candidate):
    path, _receipt, installed = candidate
    other = path.with_name("other.zip")
    other.write_bytes(b"unrelated")
    with bundle_leases.exclusive_for_mode(path, installed) as claim:
        claim.path = other
        with pytest.raises(bundle_mode.ModeError, match="acquired path"):
            claim.require()
    assert path.exists() and other.exists()


@WINDOWS
def test_mark_uses_verified_same_handle_exact_flags_and_never_path_unlink(candidate, monkeypatch):
    import msvcrt

    path, receipt, installed = candidate
    calls, marks = [], []
    native = deletion._windows_api()

    def create(name, access, share, security, disposition, flags, template):
        calls.append((name, access, share, disposition, flags))
        return native.CreateFileW(name, access, share, security, disposition, flags, template)

    bound = deletion.bind_managed_archive(receipt, installed, receipt.identity)

    def mark(handle, information, pointer, size):
        assert handle == msvcrt.get_osfhandle(bound._fds["archive"])
        assert information == 4 and size == 1 and ctypes.string_at(pointer, size) == b"\x01"
        assert bound._opened.same_file(receipt.stamp)
        marks.append(handle)
        return native.SetFileInformationByHandle(handle, information, pointer, size)

    proxy_api(monkeypatch, CreateFileW=create, SetFileInformationByHandle=mark)
    monkeypatch.setattr(Path, "unlink", lambda *_a, **_k: pytest.fail("mutation used path unlink"))
    with bound:
        assert bound.mark_for_deletion() == deletion.DeletionOutcome("accepted", False)
        with pytest.raises(RuntimeError, match="another deletion"):
            bound.mark_for_deletion()
    assert bound.outcome == deletion.DeletionOutcome("accepted", True)
    assert not path.exists() and len(marks) == 1
    archive_calls = [call for call in calls if call[0] == str(path)]
    assert archive_calls == [(str(path), 0x80010000, 0, 3, 0x00200000)]
    assert [call[2] for call in calls if call[0].endswith(bundle_mode.MARKER_NAME)] == [1]
    assert path.with_name(path.name + ".lease.lock").exists()  # persistent, never removed


@WINDOWS
def test_single_full_raw_hash_is_deferred_to_the_mark_boundary(candidate, monkeypatch):
    path, receipt, installed = candidate
    real_read, reads = os.read, []
    bound = deletion.bind_managed_archive(receipt, installed, receipt.identity)

    def track(fd, count):
        chunk = real_read(fd, count)
        if fd == bound._fds.get("archive"):
            reads.append(chunk)
        return chunk

    monkeypatch.setattr(deletion.os, "read", track)
    with bound:
        assert reads == []
        bound.mark_for_deletion()
        assert len(b"".join(reads)) == receipt.stamp.size
        assert hashlib.sha256(b"".join(reads)).hexdigest() == receipt.sha256
    assert not path.exists()


@WINDOWS
def test_close_waits_for_owned_mark_operation_without_recycling_its_fd(candidate, monkeypatch):
    _path, receipt, installed = candidate
    native = deletion._windows_api()
    reached_mark, allow_mark, close_started, close_finished = (threading.Event() for _ in range(4))
    failures = []
    bound = deletion.bind_managed_archive(receipt, installed, receipt.identity).__enter__()

    def mark(*args):
        reached_mark.set()
        assert allow_mark.wait(10)
        return native.SetFileInformationByHandle(*args)

    def perform_mark():
        try:
            bound.mark_for_deletion()
        except BaseException as error:
            failures.append(error)

    def perform_close():
        close_started.set()
        try:
            bound.close()
        except BaseException as error:
            failures.append(error)
        finally:
            close_finished.set()

    proxy_api(monkeypatch, SetFileInformationByHandle=mark)
    writer, closer = threading.Thread(target=perform_mark), threading.Thread(target=perform_close)
    writer.start()
    try:
        assert reached_mark.wait(10)
        closer.start()
        assert close_started.wait(10)
        assert not close_finished.is_set() and "archive" in bound._fds
    finally:
        allow_mark.set()
        writer.join(10)
        if closer.ident is not None:
            closer.join(10)
        bound.close()
    assert not writer.is_alive() and not closer.is_alive() and failures == []
    assert bound.outcome == deletion.DeletionOutcome("accepted", True)


@WINDOWS
@pytest.mark.parametrize("cleanup_kind", ["false_return", "raised_exception"])
def test_failed_crt_transfer_and_uncertain_raw_cleanup_keep_primary_and_native_claim(candidate, monkeypatch, cleanup_kind):
    import msvcrt

    _path, receipt, installed = candidate
    native, real_transfer, real_close = deletion._windows_api(), msvcrt.open_osfhandle, os.close
    handles, cleanups = [], []
    transfer_error = ValueError("injected CRT transfer failure")
    bound = deletion.bind_managed_archive(receipt, installed, receipt.identity)

    def create(name, *args):
        handle = native.CreateFileW(name, *args)
        if name == str(bound.path):
            handles.append(handle)
        return handle

    def transfer(handle, flags):
        if handle in handles:
            raise transfer_error
        return real_transfer(handle, flags)

    def close_raw(handle):
        if handle in handles:
            cleanups.append(handle)
            if cleanup_kind == "raised_exception":
                raise PermissionError("injected no-op raw handle cleanup failure")
            ctypes.set_last_error(5)  # Deliberately injected FALSE, not an OS denial.
            return 0
        return native.CloseHandle(handle)

    proxy_api(monkeypatch, CreateFileW=create, CloseHandle=close_raw)
    monkeypatch.setattr(msvcrt, "open_osfhandle", transfer)
    with pytest.raises(ValueError) as caught:
        with bound:
            pytest.fail("failed raw transfer reached the caller")
    assert caught.value is transfer_error and bound._raw_handles == handles
    assert len(cleanups) == 1 and transfer_error.__notes__
    assert bound._claim._retained is not None
    with pytest.raises(RuntimeError, match="no implicit retry"):
        bound.close()
    assert len(cleanups) == 1
    monkeypatch.undo()
    # Both injected CloseHandle paths were no-ops, so these exact owned raw
    # handles are known live; do not use this recovery for an actual denial.
    for handle in handles:
        assert native.CloseHandle(handle)
    recover_injected_fault(bound, real_close)


@WINDOWS
def test_raw_hash_mismatch_retains_archive_and_releases_native_claim(candidate):
    path, receipt, installed = candidate
    with pytest.raises(deletion.DeleteSafetyError, match="raw digest"):
        with deletion.bind_managed_archive(replace(receipt, sha256="0" * 64), installed, receipt.identity) as bound:
            bound.mark_for_deletion()
    assert path.exists()
    with bundle_leases.downloads.exclusive(path):
        pass


@WINDOWS
def test_replaced_same_name_archive_is_preserved(candidate):
    path, receipt, installed = candidate
    original = path.with_name("kept-original.zip")
    path.rename(original)
    path.write_bytes(b"same name, different owner")
    with pytest.raises(deletion.DeleteSafetyError, match="receipt identity"):
        with deletion.bind_managed_archive(receipt, installed, receipt.identity):
            pytest.fail("replacement accepted")
    assert path.read_bytes() == b"same name, different owner" and original.exists()


@WINDOWS
def test_hardlinked_archive_is_preserved(candidate):
    path, receipt, installed = candidate
    alias = path.with_name("alias.zip")
    os.link(path, alias)
    with pytest.raises(deletion.DeleteSafetyError, match="single-link"):
        with deletion.bind_managed_archive(receipt, installed, receipt.identity):
            pytest.fail("multiply linked archive accepted")
    assert path.exists() and alias.exists()


@WINDOWS
@pytest.mark.parametrize("fault", ["delete_pending", "directory", "native_links", "reparse"])
def test_native_archive_safety_facts_are_required(candidate, monkeypatch, fault):
    path, receipt, installed = candidate
    native = deletion._windows_api()

    def facts(handle, information, pointer, size):
        result = native.GetFileInformationByHandleEx(handle, information, pointer, size)
        if information == 1:
            value = ctypes.cast(pointer, ctypes.POINTER(deletion._StandardInfo)).contents
            if fault == "delete_pending":
                value.DeletePending = 1
            elif fault == "directory":
                value.Directory = 1
            elif fault == "native_links":
                value.NumberOfLinks = 2
        elif information == 9 and fault == "reparse":
            ctypes.cast(pointer, ctypes.POINTER(deletion._AttributeInfo)).contents.FileAttributes |= 0x400
        return result

    proxy_api(monkeypatch, GetFileInformationByHandleEx=facts)
    with pytest.raises(deletion.DeleteSafetyError, match="delete-pending"):
        with deletion.bind_managed_archive(receipt, installed, receipt.identity):
            pytest.fail("unsafe native facts accepted")
    assert path.exists()


@WINDOWS
def test_marker_is_pinned_against_writes_and_replacement_until_close(candidate):
    path, receipt, installed = candidate
    marker = installed.directory / bundle_mode.MARKER_NAME
    with deletion.bind_managed_archive(receipt, installed, receipt.identity):
        # Deliberate sharing-conflict probes on this test's marker only.
        with pytest.raises(PermissionError):
            marker.open("r+b")
        with pytest.raises(PermissionError):
            marker.rename(marker.with_name("replaced-marker"))
    assert path.exists() and bundle_mode.discover(installed.directory) == installed


@WINDOWS
def test_mode_is_rechecked_immediately_before_mark(candidate, monkeypatch):
    path, receipt, installed = candidate
    with deletion.bind_managed_archive(receipt, installed, receipt.identity) as bound:
        monkeypatch.setattr(bundle_mode, "discover", lambda *_a, **_k: replace(installed, digest="0" * 64))
        proxy_api(monkeypatch, SetFileInformationByHandle=lambda *_a: pytest.fail("changed mode marked deletion"))
        with pytest.raises(bundle_mode.ModeError, match="changed"):
            bound.mark_for_deletion()
    assert path.exists()


@WINDOWS
def test_native_disposition_false_is_reported_as_refused_and_never_retried(candidate, monkeypatch):
    path, receipt, installed = candidate
    calls = []

    def refuse(*args):
        calls.append(args)
        ctypes.set_last_error(5)  # Injected refusal; the native mark is not called.
        return 0

    proxy_api(monkeypatch, SetFileInformationByHandle=refuse)
    with deletion.bind_managed_archive(receipt, installed, receipt.identity) as bound:
        with pytest.raises(PermissionError):
            bound.mark_for_deletion()
        assert bound.outcome == deletion.DeletionOutcome("refused", False)
        with pytest.raises(RuntimeError, match="another deletion"):
            bound.mark_for_deletion()
    assert len(calls) == 1 and bound.outcome == deletion.DeletionOutcome("refused", True)
    assert path.exists()


@WINDOWS
@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("failed_kind", ["archive", "marker", "directory"])
def test_unknown_close_retains_original_claim_and_outcome_without_retry(candidate, monkeypatch, accepted, failed_kind):
    path, receipt, installed = candidate
    real_close = os.close
    bound = deletion.bind_managed_archive(receipt, installed, receipt.identity).__enter__()
    if accepted:
        bound.mark_for_deletion()
    failed_fd, calls = bound._fds[failed_kind], []
    # close() explicitly closes archive first, then marker, then directory.
    expected_closes = [bound._fds[kind] for kind in ("archive", "marker", "directory")]
    expected_closes = expected_closes[:expected_closes.index(failed_fd) + 1]

    def refuse(fd):
        calls.append(fd)
        if fd == failed_fd:
            # Injected no-op; the OS close has definitely NOT executed.
            raise PermissionError(f"injected {failed_kind} close refusal")
        return real_close(fd)

    monkeypatch.setattr(deletion.os, "close", refuse)
    with pytest.raises(PermissionError, match="injected") as caught:
        bound.close()
    assert bound._claim._retained is not caught.value
    assert bound._claim._retained.__traceback__ is None
    assert bound.outcome == deletion.DeletionOutcome("accepted" if accepted else "not_requested", failed_kind != "archive")
    if accepted:
        assert any("Do not assume" in note for note in caught.value.__notes__)
    # Resolving a retargeted name is forbidden in quarantine/retry handling.
    monkeypatch.setattr(bound.path.__class__, "resolve", lambda *_a, **_k: pytest.fail("quarantine resolved a name"))
    with pytest.raises(RuntimeError, match="no implicit retry"):
        bound.close()
    assert calls == expected_closes
    monkeypatch.undo()
    recover_injected_fault(bound, real_close)


@WINDOWS
def test_body_error_remains_primary_when_close_is_uncertain(candidate, monkeypatch):
    _path, receipt, installed = candidate
    real_close = os.close
    bound = deletion.bind_managed_archive(receipt, installed, receipt.identity)
    body = ValueError("original caller failure")
    with pytest.raises(ValueError) as caught:
        with bound:
            archive_fd = bound._fds["archive"]

            def refuse(fd):
                if fd == archive_fd:
                    raise PermissionError("injected no-op close refusal")
                return real_close(fd)

            monkeypatch.setattr(deletion.os, "close", refuse)
            raise body
    assert caught.value is body and any("injected no-op" in note for note in body.__notes__)
    monkeypatch.undo()
    recover_injected_fault(bound, real_close)


@WINDOWS
@pytest.mark.parametrize("actually_accepted", [False, True])
def test_unknown_mark_result_is_quarantined_and_never_claims_preserved(candidate, monkeypatch, actually_accepted):
    _path, receipt, installed = candidate
    native, real_close = deletion._windows_api(), os.close
    bound = deletion.bind_managed_archive(receipt, installed, receipt.identity).__enter__()

    def interrupted(*args):
        if actually_accepted:
            assert native.SetFileInformationByHandle(*args)
        raise RuntimeError("injected loss of native mark result")

    proxy_api(monkeypatch, SetFileInformationByHandle=interrupted)
    with pytest.raises(RuntimeError, match="injected loss") as caught:
        bound.mark_for_deletion()
    assert bound.outcome == deletion.DeletionOutcome("unknown", False)
    assert any("Do not assume" in note for note in caught.value.__notes__)
    with pytest.raises(RuntimeError, match="another deletion"):
        bound.mark_for_deletion()
    with pytest.raises(bundle_leases.LeaseBusy):
        bundle_leases.downloads.reader(bound.path)
    monkeypatch.undo()
    recover_injected_fault(bound, real_close)


_PROBE = """
import json, sys
from pathlib import Path
from backend.bundle_leases import ModeAwareRegistry, LeaseBusy
try:
    with ModeAwareRegistry().reader(Path(sys.argv[1])):
        result = 'claimed'
except LeaseBusy:
    result = 'busy'
assert 'torch' not in sys.modules
print(json.dumps({'result': result}))
"""


def owned_probe(path, receipt_path):
    child = subprocess.Popen([sys.executable, "-c", _PROBE, str(path)], cwd=Path(__file__).resolve().parents[1],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    receipt = {"pid": child.pid, "archive": str(path), "host_quiescence_verified": False,
               "retention_enabled": False, "state": "running"}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    try:
        stdout, stderr = child.communicate(timeout=15)
    except BaseException:
        child.kill()
        child.wait(timeout=10)
        raise
    finally:
        receipt.update(state="closed", exit_code=child.returncode)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert child.returncode == 0, stderr
    return json.loads(stdout)["result"]


@WINDOWS
def test_owned_other_process_reader_is_excluded_until_bound_handle_closes(candidate, tmp_path):
    path, receipt, installed = candidate
    with deletion.bind_managed_archive(receipt, installed, receipt.identity):
        assert owned_probe(path, tmp_path / "reader-busy.json") == "busy"
    assert owned_probe(path, tmp_path / "reader-released.json") == "claimed"
