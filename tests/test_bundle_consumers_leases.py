"""Mode-aware archive consumers over scratch files, galleries and owned probes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from backend import backup_receipts, bundle_crypto, bundle_leases, bundle_mode, bundles, gallery_lock
from backend.bundle_leases import LeaseBusy, LeaseRegistry
from tests.bundle_helpers import use_destination, save_source, bundle_snapshot

WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Installed mode is qualified only on Windows fixed NTFS")


@pytest.fixture(autouse=True)
def isolate_injected_close_failures(monkeypatch):
    monkeypatch.setattr(bundle_crypto, "_failed_readers", [])
    monkeypatch.setattr(bundle_leases, "_retained_reads", [])
    monkeypatch.setattr(bundle_leases, "_failed_inputs", [])


def watch(monkeypatch):
    original = bundle_leases.reading
    entries, releases = [], []

    @contextmanager
    def observed(path):
        entries.append(Path(path))
        with original(path) as owner:
            try:
                yield owner
            finally:
                releases.append(Path(path))

    monkeypatch.setattr(bundle_leases, "reading", observed)
    return entries, releases


def sealed(directory, password="synthetic password"):
    path = directory / "synthetic.ldbundle"
    with path.open("wb") as stream:
        writer = bundle_crypto.EncryptingWriter(stream, password, chunk=4096)
        writer.write(b"synthetic plaintext")
        writer.finish()
    return path


def fact(directory):
    directory.mkdir(exist_ok=True)
    path = directory / "managed.zip"
    path.write_bytes(b"verified synthetic fixture")
    identity = backup_receipts.ManagedIdentity(*(str(uuid4()) for _ in range(3)))
    return backup_receipts.Receipt(
        identity, str(directory.resolve()), path.name, hashlib.sha256(path.read_bytes()).hexdigest(),
        backup_receipts.FileStamp.of(path.stat()), datetime.now(timezone.utc), "unencrypted", 1, 1,
    )


def test_public_header_and_decryptor_own_exactly_one_lifetime_claim(tmp_path, monkeypatch):
    archive = sealed(tmp_path)
    entries, releases = watch(monkeypatch)
    assert bundle_crypto.is_encrypted(archive)
    assert entries == releases == [archive]
    entries.clear()
    releases.clear()
    reader = bundle_crypto.DecryptingReader(archive, "synthetic password")
    assert entries == [archive] and releases == []
    assert bundle_leases.downloads.leased(archive)
    with pytest.raises(LeaseBusy):
        LeaseRegistry().exclusive(archive)
    assert reader.read() == b"synthetic plaintext"
    reader.close()
    reader.close()
    assert releases == [archive] and not bundle_leases.downloads.leased(archive)
    with LeaseRegistry().exclusive(archive):
        pass


@pytest.mark.parametrize("damage", ["wrong_password", "broken_header", "missing_file"])
def test_decryptor_initialization_failures_release_the_claim(tmp_path, monkeypatch, damage):
    archive = sealed(tmp_path)
    if damage == "broken_header":
        archive.write_bytes(b"invalid header")
    elif damage == "missing_file":
        archive.unlink()  # Only this test's synthetic input.
    entries, releases = watch(monkeypatch)
    with pytest.raises((bundle_crypto.CryptoError, OSError)):
        bundle_crypto.DecryptingReader(archive, "wrong" if damage == "wrong_password" else "synthetic password")
    assert entries == releases == [archive]
    assert not bundle_leases.downloads.leased(archive)


class CloseRefusal:
    def __init__(self, source):
        self.source = source
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.source, name)

    def close(self):
        self.calls += 1
        raise PermissionError("injected encrypted-file close refusal")


@pytest.mark.parametrize("during_init", [False, True])
def test_uncertain_encrypted_file_close_retains_claim_and_never_implicitly_retries(tmp_path, monkeypatch, during_init):
    archive = sealed(tmp_path)
    monkeypatch.setattr(bundle_crypto, "_failed_readers", [])
    reader = None
    original_open = Path.open
    wrapped = []

    def opened(path, *args, **kwargs):
        source = original_open(path, *args, **kwargs)
        if path == archive and args and args[0] == "rb":
            wrapper = CloseRefusal(source)
            wrapped.append(wrapper)
            return wrapper
        return source

    monkeypatch.setattr(Path, "open", opened)
    try:
        if during_init:
            with pytest.raises(bundle_crypto.WrongPassword) as caught:
                bundle_crypto.DecryptingReader(archive, "wrong")
            assert "injected" in " ".join(caught.value.__notes__)
            reader = bundle_crypto._failed_readers[0]
        else:
            reader = bundle_crypto.DecryptingReader(archive, "synthetic password")
            with pytest.raises(PermissionError, match="injected"):
                reader.close()
        assert wrapped[0].calls == 1 and bundle_leases.downloads.leased(archive)
        with pytest.raises(RuntimeError, match="earlier"):
            reader.close()
        assert wrapped[0].calls == 1
        with pytest.raises(LeaseBusy):
            LeaseRegistry().exclusive(archive)
    finally:
        if reader is not None:
            # The refusal was injected and executed no OS close. Restore the
            # exact owned stream, then perform its first real close/release.
            reader._file = wrapped[0].source
            reader._close_failure = None
            reader._reader_owner._retained = None
            reader.close()
        assert not bundle_leases.downloads.leased(archive)


def test_context_body_error_survives_encrypted_close_refusal(tmp_path):
    archive = sealed(tmp_path)
    reader = bundle_crypto.DecryptingReader(archive, "synthetic password")
    source = reader._file
    refused = CloseRefusal(source)
    reader._file = refused
    original = ValueError("synthetic consumer body failure")
    try:
        with pytest.raises(ValueError) as caught:
            with reader:
                raise original
        assert caught.value is original and "close refusal" in " ".join(original.__notes__)
        assert refused.calls == 1 and bundle_leases.downloads.leased(archive)
    finally:
        reader._file = source
        reader._close_failure = None
        reader._reader_owner._retained = None
        reader.close()


@pytest.mark.parametrize("encrypted", [False, True])
def test_borrowed_import_file_close_retains_outer_claim_even_if_path_retargets(store, monkeypatch, tmp_path, encrypted):
    save_source("synthetic retained source")
    exported = bundles.export_bundle(keep_gallery=True, password="synthetic" if encrypted else None)
    archive = Path(exported["path"])
    if os.name == "nt":
        bundle_mode.install(archive.parent, other_servers_stopped=True)
    use_destination(store, monkeypatch, tmp_path)
    before = bundle_snapshot(store)
    original_open, original_resolve = Path.open, Path.resolve
    opened = 0
    refused = []

    class RetargetedClose(CloseRefusal):
        def close(self):
            monkeypatch.setattr(Path, "resolve", lambda path: tmp_path / "retargeted.zip" if path == archive else original_resolve(path))
            super().close()

    def input_file(path, *args, **kwargs):
        nonlocal opened
        source = original_open(path, *args, **kwargs)
        if path == archive and args and args[0] == "rb":
            opened += 1
            if opened == 2:  # Sniff first; fail the actual ZIP/decryption handle close.
                wrapped = RetargetedClose(source)
                refused.append(wrapped)
                return wrapped
        return source

    monkeypatch.setattr(Path, "open", input_file)
    owner = None
    try:
        with pytest.raises(PermissionError, match="injected"):
            bundles.import_bundle(archive, "synthetic" if encrypted else None)
        monkeypatch.setattr(Path, "resolve", original_resolve)
        assert refused[0].calls == 1 and bundle_snapshot(store) == before
        owner = bundle_leases._retained_reads[0]
        assert bundle_leases.downloads.leased(archive)
        with pytest.raises(LeaseBusy):
            LeaseRegistry().exclusive(archive)
        if os.name == "nt":
            assert probe(archive, tmp_path / "retained-import.json") == "busy"
    finally:
        monkeypatch.setattr(Path, "resolve", original_resolve)
        if refused:
            refused[0].source.close()  # First real close; the refusal was injected.
        if owner is not None:
            owner._retained = None
            owner.__exit__(None, None, None)
        for reader in bundle_crypto._failed_readers:
            reader._file = refused[0].source
            reader._close_failure = None
            reader.close()
    assert not bundle_leases.downloads.leased(archive)


def test_listing_holds_one_claim_over_header_and_reported_metadata(store, monkeypatch):
    save_source()
    exported = bundles.export_bundle(keep_gallery=True)
    archive = Path(exported["path"])
    entries, releases = watch(monkeypatch)
    original = bundle_crypto._is_encrypted

    def header(path, owner=None):
        assert bundle_leases.downloads.leased(path)
        with pytest.raises(LeaseBusy):
            LeaseRegistry().exclusive(path)
        return original(path, owner)

    monkeypatch.setattr(bundle_crypto, "_is_encrypted", header)
    items = bundles.list_bundles()
    assert len(items) == 1 and items[0]["name"] == archive.name
    assert entries == releases == [archive]


@pytest.mark.parametrize("consumer", ["listing", "receipt", "header", "decryptor", "import"])
@pytest.mark.parametrize("error_kind", ["invalid_mode", "native_open"])
def test_acquisition_errors_escape_availability_catches_before_archive_bytes(store, monkeypatch, tmp_path, consumer, error_kind):
    save_source()
    exported = bundles.export_bundle(keep_gallery=True)
    archive = Path(exported["path"])
    stamp = backup_receipts.FileStamp.of(archive.stat())
    receipt = backup_receipts.Receipt(
        backup_receipts.ManagedIdentity(*(str(uuid4()) for _ in range(3))), str(archive.parent), archive.name,
        hashlib.sha256(archive.read_bytes()).hexdigest(), stamp, datetime.now(timezone.utc), "unencrypted", 1, 1,
    )
    before = bundle_snapshot(store)
    original_error = bundle_mode.ModeError("injected invalid installed mode") if error_kind == "invalid_mode" else PermissionError(13, "injected native reader refusal", str(archive))

    def refuse(_path):
        raise original_error

    original_open = Path.open

    def no_bytes(path, *args, **kwargs):
        if path == archive:
            pytest.fail("archive bytes were opened before its reader was admitted")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(bundle_leases.downloads, "reader", refuse)
    monkeypatch.setattr(Path, "open", no_bytes)
    call = {"listing": bundles.list_bundles,
            "receipt": lambda: backup_receipts.matches_current(receipt, archive.parent),
            "header": lambda: bundle_crypto.is_encrypted(archive),
            "decryptor": lambda: bundle_crypto.DecryptingReader(archive, "not used"),
            "import": lambda: bundles.import_bundle(archive)}[consumer]
    with pytest.raises(type(original_error)) as caught:
        call()
    assert caught.value is original_error and bundle_snapshot(store) == before


@pytest.mark.parametrize("consumer", ["listing", "receipt", "header"])
def test_reader_release_errors_are_not_swallowed_as_missing_files(store, monkeypatch, tmp_path, consumer):
    save_source()
    exported = bundles.export_bundle(keep_gallery=True)
    archive = Path(exported["path"])
    receipt = backup_receipts.Receipt(
        backup_receipts.ManagedIdentity(*(str(uuid4()) for _ in range(3))), str(archive.parent), archive.name,
        hashlib.sha256(archive.read_bytes()).hexdigest(), backup_receipts.FileStamp.of(archive.stat()),
        datetime.now(timezone.utc), "unencrypted", 1, 1,
    )
    real = bundle_leases.reading
    original_error = OSError("injected reader release failure")

    @contextmanager
    def released(path):
        with real(path) as owner:
            yield owner
        raise original_error

    monkeypatch.setattr(bundle_leases, "reading", released)
    call = {"listing": bundles.list_bundles,
            "receipt": lambda: backup_receipts.matches_current(receipt, archive.parent),
            "header": lambda: bundle_crypto.is_encrypted(archive)}[consumer]
    with pytest.raises(OSError) as caught:
        call()
    assert caught.value is original_error


def test_receipt_hash_has_one_outer_claim_even_for_false_result(tmp_path, monkeypatch):
    directory = tmp_path / "archives"
    receipt = fact(directory)
    archive = directory / receipt.name
    entries, releases = watch(monkeypatch)
    original = backup_receipts._matches_current

    def checked(item, destination, owner=None):
        assert bundle_leases.downloads.leased(archive)
        with pytest.raises(LeaseBusy):
            LeaseRegistry().exclusive(archive)
        return original(item, destination, owner)

    monkeypatch.setattr(backup_receipts, "_matches_current", checked)
    assert backup_receipts.matches_current(receipt, directory)
    assert not backup_receipts.matches_current(replace(receipt, sha256="0" * 64), directory)
    assert entries == releases == [archive, archive]


@WINDOWS
def test_wrong_receipt_scope_cannot_create_companions_in_an_unrelated_destination(tmp_path):
    receipt = fact(tmp_path / "source")
    other = tmp_path / "another-destination"
    other.mkdir()
    bundle_mode.install(other, other_servers_stopped=True)
    assert not backup_receipts.matches_current(receipt, other)
    assert list(other.glob("*.lease.lock")) == []


@WINDOWS
@pytest.mark.parametrize("encrypted", [False, True])
def test_owned_export_readback_never_creates_partial_companions_or_nested_claims(store, monkeypatch, encrypted):
    save_source()
    directory = store.bundle_directory()
    directory.mkdir()
    bundle_mode.install(directory, other_servers_stopped=True)
    entries, releases = watch(monkeypatch)
    original = bundles._verify_archive
    verifies = []

    def verify(path, *args, **kwargs):
        verifies.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(bundles, "_verify_archive", verify)
    exported = bundles.export_bundle(keep_gallery=True, password="synthetic" if encrypted else None)
    assert exported["kept"] and len(verifies) == 1 and verifies[0].suffix == ".part"
    assert entries == releases == []
    assert list(directory.glob("*.lease.lock")) == []


_PROBE = r'''
import json,sys
from pathlib import Path
from backend.bundle_leases import LeaseBusy,ModeAwareRegistry
try:
    with ModeAwareRegistry().exclusive(Path(sys.argv[1])):
        answer="claimed"
except LeaseBusy:
    answer="busy"
assert "torch" not in sys.modules
print(json.dumps({"state":answer,"retention_enabled":False}),flush=True)
'''


def probe(archive, receipt_path):
    child = subprocess.Popen([sys.executable, "-c", _PROBE, str(archive)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, encoding="utf-8",
                             cwd=Path(__file__).resolve().parent.parent)
    receipt = {"state": "spawned", "owned_child_pid": child.pid, "archive": str(archive),
               "host_quiescence_verified": False, "retention_enabled": False}
    recorded = False
    try:
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        recorded = True
        output, error = child.communicate(timeout=15)
        assert child.returncode == 0 and error == "", error
        return json.loads(output)["state"]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if recorded:
            receipt_path.write_text(json.dumps({**receipt, "state": "finished", "owned_child_exit": child.returncode}), encoding="utf-8")


@WINDOWS
@pytest.mark.parametrize("encrypted", [False, True])
def test_cross_database_import_keeps_native_source_claim_until_merge_finishes(store, monkeypatch, tmp_path, encrypted):
    save_source("synthetic source gallery")
    exported = bundles.export_bundle(keep_gallery=True, password="synthetic" if encrypted else None)
    archive = Path(exported["path"])
    bundle_mode.install(archive.parent, other_servers_stopped=True)
    destination = use_destination(store, monkeypatch, tmp_path)
    original = bundles._replace_database
    phases = []
    entries, releases = watch(monkeypatch)

    def committed(staged):
        assert os.path.normcase(str(destination.db_path.resolve())) in gallery_lock._local.held
        assert bundle_leases.downloads.leased(archive)
        assert probe(archive, tmp_path / "during-import.json") == "busy"
        original(staged)
        phases.append("committed_under_source_claim")

    monkeypatch.setattr(bundles, "_replace_database", committed)
    assert bundles.import_bundle(archive, "synthetic" if encrypted else None)["imported"] == 1
    assert phases == ["committed_under_source_claim"] and entries == releases == [archive]
    assert probe(archive, tmp_path / "after-import.json") == "claimed"


@WINDOWS
def test_owned_upload_in_installed_staging_has_no_partial_companion(client, store, monkeypatch):
    save_source("synthetic upload source")
    exported = bundles.export_bundle(keep_gallery=True)
    payload = Path(exported["path"]).read_bytes()
    staging = store.bundle_staging_dir()
    bundle_mode.install(staging, other_servers_stopped=True)
    entries, _releases = watch(monkeypatch)
    response = client.post("/api/gallery/bundles/upload", content=payload,
                           headers={"content-type": "application/zip"})
    assert response.status_code == 202
    until = time.monotonic() + 15
    while time.monotonic() < until:
        status = client.get("/api/gallery/bundles/status").json()
        if status["state"] != "running":
            break
        time.sleep(0.01)
    assert status["state"] == "done" and status["result"]["present"] == 1
    assert not any(path.name.startswith(".bundle-upload-") for path in entries)
    assert list(staging.glob("*.lease.lock")) == []
