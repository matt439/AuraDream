"""Credential source wiring uses only fake secrets/providers and scratch galleries."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
import os
from uuid import uuid4

import pytest

from backend import api_bundles, backup_credentials as credentials, backup_receipts, backup_settings as settings
from backend import bundle_crypto, bundle_leases, bundles, gallery_lock, main
from backend.backup_coordinator import BackupCoordinator
from backend.backup_service import BackupService
from backend.bundle_leases import LeaseRegistry
from backend.failure_state import retained_failure
from tests.bundle_helpers import _destination, _save, _snapshot


def profile(db, directory):
    reference = credentials.new_reference(str(uuid4()), str(uuid4()))
    pending = settings.save_encrypted_profile(db, revision=settings.load(db).revision, directory=directory,
                                               reference=reference, recovery_confirmed=True)
    ready = settings.complete_encrypted_profile(db, revision=pending.revision, reference=reference)
    return reference, ready


def enable(db, directory, **changes):
    values = dict(revision=settings.load(db).revision, enabled=True, directory=directory,
                  minimum_age_hours=1, keep_last=2, protection="encrypted", allow_encrypted=True, confirm_recovery=True)
    values.update(changes)
    return settings.configure(db, **values)


def test_inactive_pending_ready_profiles_are_secret_free_and_native_cold(tmp_path):
    db, directory = tmp_path / "gallery.db", tmp_path / "archives"
    reference = credentials.new_reference(str(uuid4()), str(uuid4()))
    pending = settings.save_encrypted_profile(db, revision="missing", directory=directory,
                                               reference=reference, recovery_confirmed=True)
    assert pending.version == 4 and not pending.settings.enabled and pending.settings.credential_state == "pending"
    for changes in ({"protection": "unencrypted", "allow_unencrypted": True}, {}):
        with pytest.raises(settings.SettingsError, match="pending"):
            enable(db, directory, **changes)
    disabled = settings.configure(db, revision=pending.revision, enabled=False, directory=directory)
    assert disabled.settings.credential_reference == reference and disabled.settings.credential_state == "pending"
    with pytest.raises(settings.SettingsError, match="pending"):
        settings.save_encrypted_profile(db, revision=disabled.revision, directory=directory,
                                         reference=credentials.new_reference(reference.gallery_id, str(uuid4())), recovery_confirmed=True)
    ready = settings.complete_encrypted_profile(db, revision=disabled.revision, reference=reference)
    assert not ready.settings.enabled and ready.settings.credential_state == "ready"
    assert settings.load(db) == ready and not directory.exists()
    assert set(json.loads(settings.backup_preferences.path_for(db).read_text())) == settings._FIELDS_V4
    for changes in ({"allow_encrypted": False}, {"confirm_recovery": False}, {"allow_unencrypted": True, "allow_encrypted": False}):
        with pytest.raises(settings.SettingsError):
            enable(db, directory, **changes)
    active = enable(db, directory)
    assert active.settings.policy().protection == "encrypted" and not active.settings.allow_unencrypted


def test_switch_to_plaintext_starts_new_set_and_does_not_revive_old_reference(tmp_path):
    db, directory = tmp_path / "gallery.db", tmp_path / "archives"
    reference, _ready = profile(db, directory)
    active = enable(db, directory)
    plain = settings.configure(db, revision=active.revision, enabled=True, directory=directory,
                               minimum_age_hours=1, keep_last=2, protection="unencrypted", allow_unencrypted=True)
    assert plain.settings.destination_id != reference.destination_id
    assert plain.settings.credential_reference is None and plain.settings.retention is None
    with pytest.raises(settings.SettingsError, match="dedicated"):
        enable(db, directory)


@pytest.mark.parametrize("invalid", [None, "", "\ud800", "x" * 1025, 123])
def test_fake_provider_invalid_values_never_reach_plaintext_export(store, monkeypatch, invalid):
    async def scenario():
        _save()
        _reference, _ready = profile(store.settings.db_path, store.bundle_directory())
        enable(store.settings.db_path, store.bundle_directory())

        @contextmanager
        def fake(_reference):
            yield invalid

        monkeypatch.setattr(credentials, "read_secret", fake)
        monkeypatch.setattr(bundles, "export_bundle", lambda **_k: pytest.fail("invalid credential reached password truthiness"))
        service = BackupService(api_bundles.BundleManager(lambda: main.jobs, lambda: False), store.settings.db_path)
        service.prepare()
        service.start_once()
        await service.finish()
        assert service.status().state == "failed"
        assert not store.bundle_directory().exists()

    asyncio.run(scenario())


def test_due_worker_fake_secret_exports_once_and_independent_password_restores(store, monkeypatch, tmp_path):
    async def scenario():
        _save(prompt="owned encrypted fixture", seed=62)
        before = _snapshot(store)
        reference, _ready = profile(store.settings.db_path, store.bundle_directory())
        chosen = enable(store.settings.db_path, store.bundle_directory())
        events, verifications = [], []
        secret = "synthetic independent recovery password"
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
        real_verify = bundles._verify_archive

        @contextmanager
        def fake(requested):
            assert requested == reference and manager.running
            assert os.path.normcase(str(store.settings.db_path.resolve())) in gallery_lock._local.held
            events.append("read")
            try:
                yield secret
            finally:
                events.append("released")

        def verify(*args, **kwargs):
            verifications.append(True)
            return real_verify(*args, **kwargs)

        monkeypatch.setattr(credentials, "read_secret", fake)
        monkeypatch.setattr(bundles, "_verify_archive", verify)
        service = BackupService(manager, store.settings.db_path)
        service.prepare()
        assert events == []
        service.status()
        assert events == []
        service.start_once()
        await service.finish()
        status = service.status()
        assert status.state == "succeeded" and status.protection == "encrypted"
        assert events == ["read", "released"] and verifications == [True]
        assert secret not in repr(manager.status()) and secret not in repr(status)
        assert _snapshot(store) == before
        archive = store.bundle_directory() / status.archive_name
        assert archive.suffix == ".ldbundle" and bundle_crypto.is_encrypted(archive)
        # Recovery uses only an independently supplied user password, not the provider.
        _destination(store, monkeypatch, tmp_path)
        result = bundles.import_bundle(archive, password=secret)
        assert result["imported"] == 1
        assert events == ["read", "released"]
        assert chosen.settings.credential_reference == reference

    asyncio.run(scenario())


@pytest.mark.parametrize("young", [False, True])
def test_not_due_never_resolves_even_a_missing_named_credential(store, monkeypatch, young):
    async def scenario():
        reference, _ready = profile(store.settings.db_path, store.bundle_directory())
        chosen = enable(store.settings.db_path, store.bundle_directory())
        if young:
            _save()
            bundles.export_bundle(keep_gallery=True, password="synthetic old password", managed=backup_receipts.ManagedIdentity(
                reference.gallery_id, reference.destination_id, str(uuid4())))
        monkeypatch.setattr(credentials, "read_secret", lambda *_a: pytest.fail("not-due worker looked up a vault reference"))
        worker = BackupCoordinator(api_bundles.BundleManager(lambda: main.jobs, lambda: False),
                                   db_path=store.settings.db_path, directory=store.bundle_directory(), leases=LeaseRegistry())
        worker.start_once(chosen.settings.policy(), backup_receipts.ManagedIdentity(reference.gallery_id, reference.destination_id, str(uuid4())),
                          clock=lambda: datetime.now(timezone.utc), password=None, ownership_trusted=True, credential=reference)
        outcome = await worker.finish()
        assert outcome.state == "skipped" and outcome.reason == ("young" if young else "empty")

    asyncio.run(scenario())


def test_process_lifetime_failure_snapshots_and_terminal_crypto_drop_secret_roots():
    def fail_with_secret():
        password = "synthetic traceback secret"
        try:
            raise PermissionError(password)
        except PermissionError as original:
            snapshot = retained_failure()
            assert password not in str(snapshot) and snapshot.__traceback__ is None
            assert snapshot.__context__ is snapshot.__cause__ is None
            owner = bundle_leases.ArchiveRead.__new__(bundle_leases.ArchiveRead)
            owner._lease = type("Claim", (), {"_released": False})()
            owner._retained = None
            owner.retain(original)
            assert owner._retained is not original and owner._retained.__traceback__ is None
            bundle_leases._retained_reads.remove(owner)

    fail_with_secret()
    writer = bundle_crypto.EncryptingWriter(io.BytesIO(), "synthetic writer password")
    writer.write(b"plaintext held briefly")
    writer.close()
    assert writer._aead is None and writer._key == b"" and not writer._buffer


def test_malformed_pending_profile_preserves_its_bytes_for_explicit_repair(tmp_path):
    db, directory = tmp_path / "gallery.db", tmp_path / "archives"
    reference = credentials.new_reference(str(uuid4()), str(uuid4()))
    pending = settings.save_encrypted_profile(db, revision="missing", directory=directory,
                                               reference=reference, recovery_confirmed=True)
    path = settings.backup_preferences.path_for(db)
    payload = json.loads(path.read_text())
    payload["credential_reference"]["target"] = "damaged pending target"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    loaded = settings.load(db)
    assert loaded.revision is None and loaded.requires_confirmation and not loaded.settings.enabled
    with pytest.raises(settings.SettingsError):
        settings.save_encrypted_profile(db, revision=pending.revision, directory=directory,
                                         reference=credentials.new_reference(reference.gallery_id, str(uuid4())), recovery_confirmed=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("reason", ["credential_missing", "credential_session_unavailable", "credential_access_denied"])
def test_provider_repair_failures_are_fixed_outcomes_without_manager_logging(store, monkeypatch, caplog, reason):
    async def scenario():
        _save()
        _reference, _ready = profile(store.settings.db_path, store.bundle_directory())
        enable(store.settings.db_path, store.bundle_directory())
        marker = "synthetic provider secret must not escape"

        @contextmanager
        def fake(_reference):
            error = credentials.CredentialError(marker)
            error.reason = reason
            raise error
            yield  # pragma: no cover

        monkeypatch.setattr(credentials, "read_secret", fake)
        monkeypatch.setattr(bundles, "export_bundle", lambda **_k: pytest.fail("unavailable provider exported"))
        service = BackupService(api_bundles.BundleManager(lambda: main.jobs, lambda: False), store.settings.db_path)
        service.prepare()
        service.start_once()
        await service.finish()
        assert service.status().state == "failed" and service.status().reason == reason
        assert service.manager.state["state"] == "done"
        assert marker not in repr(service.status()) + caplog.text + repr(service.manager.state)

    asyncio.run(scenario())


def test_known_closed_reader_error_and_constructor_failure_drop_terminal_crypto(tmp_path):
    path = tmp_path / "owned.ldbundle"
    with path.open("xb") as stream:
        writer = bundle_crypto.EncryptingWriter(stream, "synthetic reader password")
        writer.write(b"owned plaintext")
        writer.finish()
        writer.close()
    reader = bundle_crypto.DecryptingReader(path, "synthetic reader password")
    original = reader._file

    class ClosedThenRaises:
        @property
        def closed(self):
            return original.closed

        def close(self):
            original.close()
            raise OSError("injected after known close")

    reader._file = ClosedThenRaises()
    with pytest.raises(OSError, match="known close"):
        reader.close()
    assert original.closed and reader._aead is None and reader._cache is None
    assert reader._close_failure is None
    with bundle_leases.downloads.exclusive(path):
        pass
    reader._file = original
    reader.close()  # Already-confirmed closed file, no uncertain fd retry.
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    failed = bundle_crypto._DecryptingReader.__new__(bundle_crypto._DecryptingReader)
    with pytest.raises(bundle_crypto.CryptoError):
        failed.__init__(path, "synthetic reader password")
    assert failed._file.closed and failed._aead is None and failed._cache is None
