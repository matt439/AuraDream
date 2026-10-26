"""Explicit consent and post-success cleanup, synthetic/owned scratch only."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from uuid import uuid4
from types import SimpleNamespace

import pytest

from backend import api_bundles, backup_receipts as receipts, backup_settings as settings, bundles, bundle_mode, main
from backend import backup_retention as retention
from backend.backup_coordinator import BackupCoordinator, BackupOutcome
from backend.backup_policy import BackupPolicy, CopyFacts
from backend.backup_service import BackupService
from backend.bundle_leases import LeaseRegistry, downloads
from backend.schemas_backup import BackupSettingsRequest
from tests.bundle_helpers import _save, _snapshot

WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Owned Windows fixed-local-NTFS scratch qualification")


@pytest.fixture(autouse=True)
def observations(monkeypatch):
    monkeypatch.setattr(bundle_mode, "_observed", {})
    monkeypatch.setattr(bundle_mode, "_named_observed", {})


def synthetic_mode(directory):
    return bundle_mode.SharedMode(str(uuid4()), directory.resolve(), (1, 2), (1, 3, 100, 4, 5), "a" * 64)


def enable(db, directory, mode=None, **changes):
    values = dict(revision=settings.load(db).revision, enabled=True, directory=directory,
                  minimum_age_hours=1, keep_last=2, allow_unencrypted=True)
    if mode is not None:
        values.update(retention_enabled=True, allow_managed_deletion=True,
                      expected_mode_revision=settings.mode_revision(mode))
    values.update(changes)
    return settings.configure(db, **values)


def test_v2_count_is_inert_and_new_v3_needs_separate_fresh_ack_mode_and_values(tmp_path, monkeypatch):
    db, directory = tmp_path / "gallery.db", tmp_path / "archives"
    initial = enable(db, directory)
    assert initial.version == 2 and initial.settings.retention is None
    installed = synthetic_mode(directory)
    monkeypatch.setattr(bundle_mode, "discover", lambda path: installed)
    for changes in ({"allow_managed_deletion": False}, {"expected_mode_revision": None},
                    {"expected_mode_revision": "stale"}, {"keep_last": None}):
        with pytest.raises(settings.SettingsError):
            enable(db, directory, installed, **changes)
        assert settings.load(db) == initial
    chosen = enable(db, directory, installed)
    assert chosen.version == 3 and settings.load(db) == chosen
    assert chosen.settings.retention.mode == installed
    assert chosen.settings.retention.keep_last == chosen.settings.keep_last == 2
    payload = json.loads(settings.backup_preferences.path_for(db).read_text())
    assert payload["version"] == 3 and payload["retention"]["allow_managed_deletion"] is True
    # Export-only save does not revive yesterday's cleanup permission.
    assert enable(db, directory).settings.retention is None


def test_disable_does_not_inspect_destination_or_native_mode(tmp_path, monkeypatch):
    db, directory = tmp_path / "gallery.db", tmp_path / "archives"
    installed = synthetic_mode(directory)
    monkeypatch.setattr(bundle_mode, "discover", lambda path: installed)
    chosen = enable(db, directory, installed)
    monkeypatch.setattr(bundle_mode, "discover", lambda *_a: pytest.fail("disable inspected native destination"))
    disabled = settings.configure(db, revision=chosen.revision, enabled=False, directory=tmp_path / "unavailable")
    assert not disabled.settings.enabled and disabled.settings.retention is None
    assert not directory.exists()


@pytest.mark.parametrize("pending,uncertain", [(True, False), (False, True)])
def test_cached_status_effective_cleanup_turns_off_for_live_pending_or_unknown(tmp_path, monkeypatch, pending, uncertain):
    installed = synthetic_mode(tmp_path)
    gallery, destination = str(uuid4()), str(uuid4())
    permission = settings.RetentionPermission(gallery, destination, str(tmp_path), 2, "unencrypted", installed)
    saved = settings.BackupSettings(True, 1, 2, "unencrypted", True, gallery, destination, str(tmp_path), permission)
    service = BackupService(api_bundles.BundleManager(lambda: main.jobs, lambda: False), tmp_path / "gallery.db")
    service.loaded, service.directory, service.mode = settings.LoadedSettings(saved, "synthetic", version=3), tmp_path, installed
    outcome = BackupOutcome("succeeded", "verified_publication", retention=retention.RetentionOutcome(
        "warning", "cleanup_incomplete", pending=pending, uncertain=uncertain))
    service.coordinator = SimpleNamespace(status=lambda: outcome)
    monkeypatch.setattr(bundle_mode, "discover", lambda *_a: pytest.fail("cached status read native mode"))
    monkeypatch.setattr(receipts, "load_state", lambda *_a: pytest.fail("cached status read journal"))
    status = service.status()
    assert status.state == "succeeded" and not status.retention_enabled
    assert status.retention_pending == pending and status.retention_uncertain == uncertain


@pytest.fixture
def cohort(tmp_path):
    directory = tmp_path / "owned-retention"
    directory.mkdir()
    db = tmp_path / "gallery.db"
    gallery, destination = str(uuid4()), str(uuid4())
    now = datetime.now(timezone.utc)
    copies = []
    for index in range(5):
        identity = receipts.ManagedIdentity(gallery, destination, str(uuid4()))
        path = directory / f"owned-{index}.zip"
        path.write_bytes(f"synthetic owned raw archive {index}".encode())
        item = receipts.Receipt(identity, str(directory.resolve()), path.name, hashlib.sha256(path.read_bytes()).hexdigest(),
                                receipts.FileStamp.of(path.lstat()), now - timedelta(days=4 - index), "unencrypted", 1, 1)
        receipts.append(db, item)
        copies.append(item)
    installed = bundle_mode.install(directory, other_servers_stopped=True).mode
    permission = settings.RetentionPermission(gallery, destination, str(directory.resolve()), 2, "unencrypted", installed)
    policy = BackupPolicy(True, timedelta(hours=1), 2, "unencrypted")
    return db, directory, copies, permission, policy, now


def run(cohort, **changes):
    db, _directory, copies, permission, policy, now = cohort
    facts = [CopyFacts(item.identity.operation_id, item.identity.gallery_id, item.identity.destination_id,
                       item.verified_at, item.protection, "managed", False) for item in copies[:-1]]
    values = dict(now=now, previous_facts=facts)
    values.update(changes)
    return retention.prune_after_success(db, permission, policy, copies[-1].identity, copies[-1], **values)


@WINDOWS
def test_actual_owned_native_cleanup_protects_newest_n_and_does_not_reread_new_success(cohort, monkeypatch):
    db, directory, copies, _permission, _policy, _now = cohort
    real_matches, checked = receipts._matches_current, []

    def matches(item, *args):
        checked.append(item.name)
        assert item != copies[-1], "new archive was redundantly read back"
        return real_matches(item, *args)

    monkeypatch.setattr(receipts, "_matches_current", matches)
    outcome = run(cohort)
    state = receipts.load_state(db, copies[-1].identity.gallery_id)
    assert outcome.state == "succeeded" and outcome.marked == 3 and not outcome.pending
    assert [item.name for item in state.receipts] == [item.name for item in copies[-2:]]
    assert len(state.retired) == 3 and state.pending is None
    assert all((directory / item.name).exists() for item in copies[-2:])
    assert all(not (directory / item.name).exists() for item in copies[:-2])
    assert checked == [copies[-2].name]


@WINDOWS
def test_leased_older_excess_is_kept_and_counted_without_becoming_candidate(cohort):
    _db, directory, copies, _permission, _policy, _now = cohort
    with downloads.reader(directory / copies[0].name):
        facts = [CopyFacts(item.identity.operation_id, item.identity.gallery_id, item.identity.destination_id,
                           item.verified_at, item.protection, "managed", index == 0)
                 for index, item in enumerate(copies[:-1])]
        outcome = run(cohort, previous_facts=facts)
    assert outcome.reason == "busy_copies" and outcome.retained == 1 and outcome.marked == 2
    assert (directory / copies[0].name).exists()


@WINDOWS
def test_replaced_protected_copy_cannot_count_toward_n(cohort):
    _db, directory, copies, _permission, _policy, _now = cohort
    path = directory / copies[-2].name
    path.write_bytes(b"not the verified copy")
    outcome = run(cohort)
    assert outcome.marked == 2
    assert (directory / copies[-3].name).exists()  # next CURRENT good copy fills N
    assert path.read_bytes() == b"not the verified copy"


@WINDOWS
def test_manual_and_foreign_destination_copies_never_become_cleanup_candidates(cohort):
    db, directory, copies, _permission, _policy, _now = cohort
    manual = directory / "manual.zip"
    manual.write_bytes(b"manual copy without an ownership receipt")
    foreign = directory / "foreign.zip"
    foreign.write_bytes(b"another destination namespace")
    item = receipts.Receipt(replace(copies[0].identity, destination_id=str(uuid4()), operation_id=str(uuid4())),
                            str(directory), foreign.name, hashlib.sha256(foreign.read_bytes()).hexdigest(),
                            receipts.FileStamp.of(foreign.lstat()), copies[0].verified_at, "unencrypted", 1, 1)
    receipts.append(db, item)
    assert run(cohort).marked == 3
    assert manual.read_bytes() == b"manual copy without an ownership receipt"
    assert foreign.read_bytes() == b"another destination namespace"
    assert item in receipts.load(db, copies[-1].identity.gallery_id)


@WINDOWS
@pytest.mark.parametrize("failure", ["mode_changed", "journal_unusable", "begin_write"])
def test_pre_intent_failures_do_not_invent_durable_pending(cohort, monkeypatch, failure):
    if failure == "mode_changed":
        monkeypatch.setattr(bundle_mode, "discover", lambda *_a: None)
    elif failure == "journal_unusable":
        monkeypatch.setattr(receipts, "load_state", lambda *_a: (_ for _ in ()).throw(receipts.ReceiptError("synthetic damaged journal")))
    else:
        monkeypatch.setattr(receipts, "begin_deletion", lambda *_a: (_ for _ in ()).throw(OSError("synthetic before intent write")))
    outcome = run(cohort)
    assert not outcome.pending and outcome.marked == 0
    assert outcome.uncertain == (failure == "journal_unusable")


@WINDOWS
@pytest.mark.parametrize("failure", ["before_mark", "accepted_cleanup", "complete_write"])
def test_pending_transitions_do_not_retry_uncertain_removal(cohort, monkeypatch, failure):
    db, directory, copies, _permission, _policy, _now = cohort
    real_bind = retention.bundle_delete.bind_managed_archive
    real_complete = receipts.complete_deletion

    def bind(*args):
        bound = real_bind(*args)
        if failure == "before_mark":
            monkeypatch.setattr(bound, "mark_for_deletion", lambda: (_ for _ in ()).throw(ValueError("injected before mark")))
        elif failure == "accepted_cleanup":
            # Inject a cleanup error AFTER real owned handles have all closed;
            # the entire context still failed and pending must not be retired.
            real_exit = bound.__class__.__exit__

            def exit_with_failure(self, *exc):
                real_exit(self, *exc)
                raise RuntimeError("injected later cleanup failure")

            monkeypatch.setattr(bound.__class__, "__exit__", exit_with_failure)
        return bound

    monkeypatch.setattr(retention.bundle_delete, "bind_managed_archive", bind)
    if failure == "complete_write":
        monkeypatch.setattr(receipts, "complete_deletion", lambda *_a: (_ for _ in ()).throw(OSError("injected final journal write")))
    outcome = run(cohort)
    state = receipts.load_state(db, copies[-1].identity.gallery_id)
    assert outcome.state == "warning"
    if failure == "before_mark":
        assert state.pending is None and (directory / copies[0].name).exists()
    else:
        assert state.pending is not None and outcome.pending
        assert outcome.marked == 1
        monkeypatch.setattr(receipts, "complete_deletion", real_complete)
        before = receipts.path_for(db).read_bytes()
        assert run(cohort).reason == "pending_intent"
        assert receipts.path_for(db).read_bytes() == before


@WINDOWS
def test_disabling_preserves_pending_visibility_without_destination_access(cohort, monkeypatch):
    db, directory, copies, permission, _policy, _now = cohort
    monkeypatch.setattr(bundle_mode, "discover", lambda *_a: permission.mode)
    previous = settings.BackupSettings(True, 1, 2, "unencrypted", True, permission.gallery_id,
                                       permission.destination_id, str(directory))
    settings.backup_preferences.path_for(db).write_text(json.dumps(
        {"version": 2, **{key: value for key, value in previous.__dict__.items() if key in settings._FIELDS - {"version"}}}))
    chosen = enable(db, directory, permission.mode)
    intent = receipts.PendingDeletion(copies[0], str(uuid4()), copies[-1].identity.operation_id, permission.mode)
    receipts.begin_deletion(db, intent)
    service = BackupService(api_bundles.BundleManager(lambda: main.jobs, lambda: False), db)
    service.loaded = settings.load(db)
    monkeypatch.setattr(bundle_mode, "discover", lambda *_a: pytest.fail("disable inspected native mode"))
    monkeypatch.setattr(retention, "prune_after_success", lambda *_a: pytest.fail("save pruned"))
    service.configure(BackupSettingsRequest(revision=service.loaded.revision, enabled=False))
    status = service.status()
    assert not status.enabled and status.retention_pending and status.retention_reason == "pending_intent"
    assert service.loaded.settings.retention is None and chosen.settings.enabled


@WINDOWS
def test_real_coordinator_backup_success_survives_retention_warning(store, monkeypatch):
    async def scenario():
        _save()
        directory = store.bundle_directory()
        directory.mkdir()
        installed = bundle_mode.install(directory, other_servers_stopped=True).mode
        chosen = enable(store.settings.db_path, directory, installed)
        before = _snapshot(store)
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
        worker = BackupCoordinator(manager, db_path=store.settings.db_path, directory=directory, leases=LeaseRegistry())
        identity = receipts.ManagedIdentity(chosen.settings.gallery_id, chosen.settings.destination_id, str(uuid4()))
        monkeypatch.setattr("backend.backup_coordinator.prune_after_success",
                            lambda *_a, **_k: retention.RetentionOutcome("warning", "cleanup_incomplete", warning="fixed scratch warning"))
        worker.start_once(chosen.settings.policy(), identity, clock=lambda: datetime.now(timezone.utc),
                          password=None, ownership_trusted=True, retention=chosen.settings.retention)
        outcome = await worker.finish()
        assert outcome.state == "succeeded" and outcome.retention.warning == "fixed scratch warning"
        assert outcome.verified_at
        assert any(item.identity == identity for item in receipts.load(store.settings.db_path, identity.gallery_id))
        service = BackupService(manager, store.settings.db_path)
        monkeypatch.setattr(receipts, "load", lambda *_a: pytest.fail("known success time reread a damaged journal"))
        assert service._verification_time(worker, identity.gallery_id) == outcome.verified_at
        assert _snapshot(store) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["failed_export", "unmanaged", "empty", "young"])
def test_no_cleanup_after_failed_unmanaged_empty_or_young_attempt(store, monkeypatch, failure):
    async def scenario():
        if failure != "empty":
            _save()
        identity = receipts.ManagedIdentity(*(str(uuid4()) for _ in range(3)))
        directory = store.bundle_directory()
        policy = BackupPolicy(True, timedelta(hours=1), 1, "unencrypted")
        permission = settings.RetentionPermission(identity.gallery_id, identity.destination_id, str(directory), 1,
                                                   "unencrypted", synthetic_mode(directory))
        if failure == "young":
            bundles.export_bundle(keep_gallery=True, managed=replace(identity, operation_id=str(uuid4())))
        elif failure == "failed_export":
            monkeypatch.setattr(bundles, "export_bundle", lambda **_k: (_ for _ in ()).throw(OSError("synthetic export failure")))
        elif failure == "unmanaged":
            monkeypatch.setattr(bundles, "export_bundle", lambda **_k: {"name": "unmanaged.zip", "managed": False})
        monkeypatch.setattr("backend.backup_coordinator.prune_after_success", lambda *_a, **_k: pytest.fail("unsuccessful attempt pruned"))
        worker = BackupCoordinator(api_bundles.BundleManager(lambda: main.jobs, lambda: False),
                                   db_path=store.settings.db_path, directory=directory, leases=LeaseRegistry())
        worker.start_once(policy, identity, clock=lambda: datetime.now(timezone.utc), password=None,
                          ownership_trusted=True, retention=permission)
        outcome = await worker.finish()
        assert outcome.state != "succeeded" and outcome.retention.state == "disabled"

    asyncio.run(scenario())


@WINDOWS
def test_actual_startup_export_then_cleanup_and_retired_retry_keeps_gallery(store, monkeypatch):
    async def scenario():
        _save(prompt="scratch lifecycle", seed=18)
        directory = store.bundle_directory()
        directory.mkdir(exist_ok=True)
        installed = bundle_mode.install(directory, other_servers_stopped=True).mode
        chosen = enable(store.settings.db_path, directory, installed, keep_last=1)
        old_identity = receipts.ManagedIdentity(chosen.settings.gallery_id, chosen.settings.destination_id, str(uuid4()))
        old = bundles.export_bundle(keep_gallery=True, managed=old_identity)
        journal_path = receipts.path_for(store.settings.db_path)
        payload = json.loads(journal_path.read_text())
        payload["receipts"][0]["verified_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        journal_path.write_text(json.dumps(payload))  # Owned synthetic age fixture only.
        before = _snapshot(store)
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
        service = BackupService(manager, store.settings.db_path)
        service.prepare()
        service.start_once()
        await service.finish()
        status = service.status()
        assert status.state == "succeeded" and status.verified_at
        assert status.retention_marked == 1 and not status.retention_pending
        state = receipts.load_state(store.settings.db_path, old_identity.gallery_id)
        assert len(state.receipts) == 1 and len(state.retired) == 1 and state.pending is None
        assert not (directory / old["name"]).exists()
        assert _snapshot(store) == before
        # A later coordinator cannot re-export a retired operation ID.
        monkeypatch.setattr(bundles, "export_bundle", lambda **_k: pytest.fail("retired operation was exported again"))
        worker = BackupCoordinator(manager, db_path=store.settings.db_path, directory=directory, leases=LeaseRegistry())
        worker.start_once(chosen.settings.policy(), old_identity, clock=lambda: datetime.now(timezone.utc) + timedelta(days=3),
                          password=None, ownership_trusted=True, retention=chosen.settings.retention)
        outcome = await worker.finish()
        assert outcome.state == "skipped" and outcome.reason == "operation_already_recorded"

    asyncio.run(scenario())
