"""Inactive coordinator sequencing over scratch galleries and owned event barriers."""
from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from backend import api_bundles, backup_receipts, bundles, gallery_lock, main
from backend.backup_coordinator import BackupCoordinator
from backend.backup_policy import BackupPolicy
from backend.bundle_leases import LeaseRegistry, LeasedFileResponse
from tests.bundle_helpers import _save, _snapshot


@pytest.fixture
def identity():
    return backup_receipts.ManagedIdentity(str(uuid4()), str(uuid4()), str(uuid4()))


@pytest.fixture
def policy():
    return BackupPolicy(True, timedelta(hours=1), 2, "unencrypted")


def coordinator(store, *, manager=None, directory=None, database=None):
    manager = manager or api_bundles.BundleManager(lambda: main.jobs, lambda: False)
    return BackupCoordinator(manager, db_path=database or store.settings.db_path,
                             directory=directory or store.bundle_directory(), leases=LeaseRegistry())


def launch(worker, policy, identity, *, clock=None, password=None, ownership_trusted=True):
    return worker.start_once(policy, identity, clock=clock or (lambda: datetime.now(timezone.utc)),
                             password=password, ownership_trusted=ownership_trusted)


def test_disabled_is_no_io_no_clock_and_no_admission(store, identity, monkeypatch, tmp_path):
    worker = coordinator(store, database=tmp_path / "absent" / "never.db")

    def unexpected(*_args, **_kwargs):
        pytest.fail("disabled policy must do no work")

    monkeypatch.setattr(worker.manager, "reserve", unexpected)
    monkeypatch.setattr(backup_receipts, "load", unexpected)
    outcome = worker.start_once(BackupPolicy(), identity, clock=unexpected, password=None, ownership_trusted=False)
    assert outcome.state == "disabled" and worker.status() == outcome
    assert not (tmp_path / "absent").exists()
    assert worker.manager.task is None


@pytest.mark.parametrize("protection,password", [
    ("encrypted", None), ("encrypted", ""), ("encrypted", "bad\ud800"), ("encrypted", "x" * 1025),
    ("unencrypted", "secret-value"), ("unencrypted", ""),
])
def test_exact_protection_mismatch_refuses_before_admission_or_export(store, policy, identity, protection, password):
    worker = coordinator(store)
    outcome = launch(worker, replace(policy, protection=protection), identity, password=password)
    assert outcome.state == "failed" and outcome.reason == "invalid_identity_or_protection"
    assert "secret-value" not in repr(outcome)
    assert worker.manager.task is None and not worker.manager.running
    assert not store.bundle_directory().exists()
    assert not backup_receipts.path_for(store.settings.db_path).exists()


def test_untrusted_inventory_is_skipped_without_admission(store, policy, identity):
    worker = coordinator(store)
    outcome = launch(worker, policy, identity, ownership_trusted=False)
    assert outcome.state == "skipped" and outcome.reason == "untrusted_inventory"
    assert worker.manager.task is None


@pytest.mark.parametrize("encrypted", [False, True])
def test_explicit_attempt_succeeds_without_rehashing_new_archive(store, policy, identity, monkeypatch, encrypted):
    async def scenario():
        _save()
        before = _snapshot(store)
        worker = coordinator(store)
        checked = []
        verified = []
        real_matches = backup_receipts.matches_current
        real_verify = bundles._verify_archive
        real_load = backup_receipts.load

        def matches(receipt, directory):
            checked.append(receipt.identity.operation_id)
            return real_matches(receipt, directory)

        def verify(*args, **kwargs):
            verified.append(True)
            return real_verify(*args, **kwargs)

        def load(db, gallery):
            assert os.path.normcase(str(db.resolve())) in gallery_lock._local.held
            return real_load(db, gallery)

        monkeypatch.setattr(backup_receipts, "matches_current", matches)
        monkeypatch.setattr(backup_receipts, "load", load)
        monkeypatch.setattr(bundles, "_verify_archive", verify)
        chosen = replace(policy, protection="encrypted") if encrypted else policy
        assert launch(worker, chosen, identity, password="test secret" if encrypted else None).state == "running"
        outcome = await worker.finish()
        assert outcome.state == "succeeded" and outcome.archive_name
        assert checked == [] and verified == [True]
        assert _snapshot(store) == before
        assert worker.manager.state["state"] == "done"
        assert worker.manager.state["result"]["state"] == "succeeded"
        assert "test secret" not in repr(worker.manager.status())
        receipt, = real_load(store.settings.db_path, identity.gallery_id)
        assert receipt.protection == chosen.protection

    asyncio.run(scenario())


def test_due_rechecks_current_receipt_and_exact_clock_boundaries(store, policy, identity):
    async def scenario():
        _save()
        worker = coordinator(store)
        launch(worker, policy, identity)
        assert (await worker.finish()).state == "succeeded"
        receipt, = backup_receipts.load(store.settings.db_path, identity.gallery_id)
        journal = backup_receipts.path_for(store.settings.db_path)
        before = journal.read_bytes()
        next_identity = replace(identity, operation_id=str(uuid4()))
        launch(worker, policy, next_identity, clock=lambda: receipt.verified_at + policy.minimum_age - timedelta(microseconds=1))
        skipped = await worker.finish()
        assert skipped.state == "skipped" and skipped.reason == "young"
        assert journal.read_bytes() == before
        launch(worker, policy, next_identity, clock=lambda: receipt.verified_at - timedelta(microseconds=1))
        assert (await worker.finish()).clock_warning
        assert journal.read_bytes() == before
        launch(worker, policy, next_identity, clock=lambda: receipt.verified_at + policy.minimum_age)
        assert (await worker.finish()).state == "succeeded"
        assert len(backup_receipts.load(store.settings.db_path, identity.gallery_id)) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("fresh_coordinator", [False, True])
def test_receipted_operation_is_not_exported_again_even_after_minimum_age(store, policy, identity, monkeypatch, fresh_coordinator):
    async def scenario():
        _save()
        worker = coordinator(store)
        launch(worker, policy, identity)
        assert (await worker.finish()).state == "succeeded"
        receipt, = backup_receipts.load(store.settings.db_path, identity.gallery_id)
        journal = backup_receipts.path_for(store.settings.db_path)
        before = journal.read_bytes()
        archives = bundles.list_bundles()
        if fresh_coordinator:
            worker = coordinator(store)

        def unexpected(**_kwargs):
            pytest.fail("a receipted operation cannot create another copy")

        monkeypatch.setattr(bundles, "export_bundle", unexpected)
        launch(worker, policy, identity, clock=lambda: receipt.verified_at + policy.minimum_age)
        outcome = await worker.finish()
        assert outcome.state == "skipped" and outcome.reason == "operation_already_recorded"
        assert outcome.archive_name == receipt.name
        assert journal.read_bytes() == before and bundles.list_bundles() == archives

    asyncio.run(scenario())


def test_refused_start_cannot_replace_terminal_outcome_before_admission_ends(store, policy, identity, monkeypatch):
    async def scenario():
        _save()
        worker = coordinator(store)
        computed, release = threading.Event(), threading.Event()
        real_complete = worker._complete

        def held(outcome):
            result = real_complete(outcome)
            computed.set()
            assert release.wait(10)
            return result

        monkeypatch.setattr(worker, "_complete", held)
        launch(worker, policy, identity)
        try:
            assert await asyncio.to_thread(computed.wait, 10)
            admitted = worker.status()
            assert admitted.state == "succeeded" and worker.manager.running
            refused = launch(worker, policy, replace(identity, operation_id=str(uuid4())))
            assert refused.state == "deferred" and worker.status() == admitted
        finally:
            release.set()
        assert await worker.finish() == admitted
        # Subsequent refusals still leave the admitted result available, while
        # a genuinely admitted attempt may install its own running outcome.
        assert launch(worker, BackupPolicy(), identity).state == "disabled"
        assert worker.status() == admitted
        monkeypatch.setattr(worker, "_complete", real_complete)
        next_identity = replace(identity, operation_id=str(uuid4()))
        assert launch(worker, policy, next_identity).state == "running"
        assert worker.status().operation_id == next_identity.operation_id
        assert (await worker.finish()).state == "skipped"

    asyncio.run(scenario())


def test_empty_gallery_is_skipped_without_advancing_any_receipt(store, policy, identity):
    async def scenario():
        worker = coordinator(store)
        launch(worker, policy, identity)
        outcome = await worker.finish()
        assert outcome.state == "skipped" and outcome.reason == "empty"
        assert not backup_receipts.path_for(store.settings.db_path).exists()
        assert bundles.list_bundles() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("busy", ["readers", "jobs", "switch", "manager"])
def test_admission_contention_is_deferred_without_a_worker(store, policy, identity, monkeypatch, busy):
    async def scenario():
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: busy == "switch")
        if busy == "readers":
            manager.readers = 1
        elif busy == "jobs":
            monkeypatch.setattr(main.jobs, "busy", lambda: True)
        elif busy == "manager":
            manager.reserve("import")
        worker = coordinator(store, manager=manager)
        assert launch(worker, policy, identity).state == "deferred"
        assert manager.task is None and not store.bundle_directory().exists()

    asyncio.run(scenario())


def test_gallery_lock_contention_is_bounded_deferred_and_does_not_retry(store, policy, identity):
    entered, release = threading.Event(), threading.Event()

    def hold():
        with gallery_lock.hold(store.settings.db_path):
            entered.set()
            assert release.wait(10)

    async def scenario():
        worker = coordinator(store)
        launch(worker, policy, identity)
        outcome = await worker.finish()
        assert outcome.state == "deferred" and outcome.reason == "gallery_busy"
        assert not backup_receipts.path_for(store.settings.db_path).exists()

    with ThreadPoolExecutor(1) as pool:
        held = pool.submit(hold)
        assert entered.wait(10)
        try:
            asyncio.run(scenario())
        finally:
            release.set()
        held.result(10)


@pytest.mark.parametrize("cancel", ["supervisor", "finish_waiter"])
def test_duplicate_start_and_cancellation_keep_admission_until_worker_finishes(store, policy, identity, monkeypatch, cancel):
    async def scenario():
        _save()
        entered, release = threading.Event(), threading.Event()
        real = bundles.export_bundle
        calls = []

        def held(**kwargs):
            calls.append(True)
            entered.set()
            assert release.wait(10)
            return real(**kwargs)

        monkeypatch.setattr(bundles, "export_bundle", held)
        worker = coordinator(store)
        launch(worker, policy, identity)
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            duplicate = launch(worker, policy, replace(identity, operation_id=str(uuid4())))
            assert duplicate.state == "deferred" and worker.status().state == "running"
            if cancel == "supervisor":
                worker.manager.task.cancel()
                await asyncio.sleep(0)
            else:
                waiting = asyncio.create_task(worker.finish())
                await asyncio.sleep(0)
                waiting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiting
            assert worker.manager.running
            with pytest.raises(HTTPException):
                worker.manager.reserve("import")
        finally:
            release.set()
        assert (await worker.finish()).state == "succeeded" and calls == [True]
        assert not worker.manager.running

    asyncio.run(scenario())


def test_immediate_supervisor_cancellation_fails_without_starting_a_writer(store, policy, identity, monkeypatch):
    async def scenario():
        _save()
        before = _snapshot(store)
        worker = coordinator(store)
        real = bundles.export_bundle
        calls = []

        def export(**kwargs):
            calls.append(True)
            return real(**kwargs)

        monkeypatch.setattr(bundles, "export_bundle", export)
        assert launch(worker, policy, identity).state == "running"
        worker.manager.task.cancel()  # No event-loop turn or worker entry yet.
        outcome = await worker.finish()
        assert outcome.state == "failed" and outcome.reason == "worker_cancelled_before_start"
        assert worker.manager.state["state"] == "failed" and not worker.manager.running
        assert calls == [] and _snapshot(store) == before
        assert not backup_receipts.path_for(store.settings.db_path).exists()
        assert launch(worker, policy, identity).state == "running"
        assert (await worker.finish()).state == "succeeded" and calls == [True]

    asyncio.run(scenario())


def test_manual_manager_cancelled_before_first_turn_cleans_owned_upload_once(store, tmp_path, monkeypatch):
    async def scenario():
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
        upload = tmp_path / "owned-upload.part"
        upload.write_bytes(b"owned upload")
        called = []
        unlinked = []
        real_unlink = type(upload).unlink

        def unlink(path, *args, **kwargs):
            if path == upload:
                unlinked.append(True)
            return real_unlink(path, *args, **kwargs)

        def writer():
            called.append(True)
            return {"imported": 1}

        monkeypatch.setattr(type(upload), "unlink", unlink)
        manager.reserve("import")
        manager.start(writer, cleanup=upload)
        manager.task.cancel()
        await manager.finish()
        await asyncio.sleep(0)  # Any queued done callback must remain idempotent.
        assert not manager.running and manager.state["state"] == "failed"
        assert called == [] and unlinked == [True] and not upload.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("queued", [False, True])
def test_executor_submission_failure_cannot_leave_an_unowned_writer_or_running_status(
    store, policy, identity, monkeypatch, caplog, queued,
):
    with ThreadPoolExecutor(1) as executor:
        async def scenario():
            _save()
            before = _snapshot(store)
            worker = coordinator(store)
            calls = []
            submitted = []

            def export(**_kwargs):
                calls.append(True)
                pytest.fail("failed submission cannot run an exporter")

            def fail_submission(_executor, fn, *args):
                if queued:
                    submitted.append(executor.submit(fn, *args))
                raise RuntimeError("secret executor details must not be echoed")

            monkeypatch.setattr(bundles, "export_bundle", export)
            loop = asyncio.get_running_loop()
            monkeypatch.setattr(loop, "run_in_executor", fail_submission)
            launch(worker, policy, identity)
            outcome = await worker.finish()
            for pending in submitted:
                assert pending.result(10) == {}
            assert outcome.state == "failed" and outcome.reason == "worker_start_failed"
            assert worker.manager.state["state"] == "failed" and not worker.manager.running
            assert worker.manager.state["error"] == "The gallery bundle worker could not start."
            assert calls == [] and _snapshot(store) == before
            assert not backup_receipts.path_for(store.settings.db_path).exists()
            assert bundles.list_bundles() == []
            assert "secret executor details" not in repr(worker.manager.status()) + caplog.text

        asyncio.run(scenario())


def test_manual_manager_submission_failure_cleans_owned_upload(store, tmp_path, monkeypatch):
    async def scenario():
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
        upload = tmp_path / "owned-upload.part"
        upload.write_bytes(b"owned upload")
        calls = []

        def fail_submission(*_args):
            raise RuntimeError("executor is shut down")

        monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", fail_submission)
        manager.reserve("import")
        manager.start(lambda: calls.append(True), cleanup=upload)
        await manager.finish()
        assert manager.state["state"] == "failed" and not manager.running
        assert not upload.exists() and calls == []

    asyncio.run(scenario())


def test_failed_receipt_keeps_old_success_and_new_publication_is_unmanaged(store, policy, identity, monkeypatch):
    async def scenario():
        _save()
        worker = coordinator(store)
        launch(worker, policy, identity)
        await worker.finish()
        receipt, = backup_receipts.load(store.settings.db_path, identity.gallery_id)
        journal = backup_receipts.path_for(store.settings.db_path)
        before = journal.read_bytes()

        def fail(*_args):
            raise OSError("secret must not be echoed")

        monkeypatch.setattr(backup_receipts, "append", fail)
        launch(worker, policy, replace(identity, operation_id=str(uuid4())),
               clock=lambda: receipt.verified_at + policy.minimum_age)
        outcome = await worker.finish()
        assert outcome.state == "unmanaged" and outcome.archive_name
        assert journal.read_bytes() == before and len(bundles.list_bundles()) == 2
        assert "secret must not be echoed" not in repr(worker.manager.status())

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["journal", "export", "clock", "source", "destination"])
def test_failures_are_reported_without_secret_details_or_new_receipts(store, policy, identity, monkeypatch, tmp_path, failure):
    async def scenario():
        _save()
        before = _snapshot(store)
        worker = coordinator(store, database=tmp_path / "wrong" / "gallery.db" if failure == "source" else None,
                             directory=tmp_path / "wrong" if failure == "destination" else None)
        journal = backup_receipts.path_for(store.settings.db_path)
        if failure == "journal":
            journal.write_bytes(b"malformed secret bytes")
        elif failure == "export":
            def fail(**_kwargs):
                raise OSError("secret must not be echoed")
            monkeypatch.setattr(bundles, "export_bundle", fail)
        launch(worker, policy, identity, clock=(lambda: datetime(2026, 10, 26)) if failure == "clock" else None)
        outcome = await worker.finish()
        assert outcome.state == "failed"
        assert "secret" not in repr(worker.manager.status())
        assert _snapshot(store) == before and bundles.list_bundles() == []
        assert not (tmp_path / "wrong").exists()
        if failure == "journal":
            assert journal.read_bytes() == b"malformed secret bytes"
        else:
            assert not journal.exists()

    asyncio.run(scenario())


def test_replaced_old_archive_never_covers_age_or_gets_removed(store, policy, identity):
    async def scenario():
        _save()
        worker = coordinator(store)
        launch(worker, policy, identity)
        await worker.finish()
        receipt, = backup_receipts.load(store.settings.db_path, identity.gallery_id)
        old = store.bundle_directory() / receipt.name
        old.write_bytes(b"an unowned replacement")
        launch(worker, policy, replace(identity, operation_id=str(uuid4())))
        assert (await worker.finish()).state == "succeeded"
        assert old.read_bytes() == b"an unowned replacement" and len(bundles.list_bundles()) == 2

    asyncio.run(scenario())


def test_download_is_leased_and_available_during_ordinary_maintenance(client, store, monkeypatch):
    _save()
    archive = bundles.export_bundle(keep_gallery=True)
    registry = LeaseRegistry()
    monkeypatch.setattr(LeasedFileResponse, "registry", registry)
    real = LeasedFileResponse._handle_simple
    seen = []

    async def handle(response, *args, **kwargs):
        assert registry.leased(response.path) and main.bundle_manager.running
        seen.append(True)
        return await real(response, *args, **kwargs)

    monkeypatch.setattr(LeasedFileResponse, "_handle_simple", handle)
    main.bundle_manager.reserve("backup")
    try:
        response = client.get(f"/api/gallery/bundles/files/{archive['name']}")
        assert response.status_code == 200 and seen == [True]
        assert registry.snapshot() == frozenset()
    finally:
        main.bundle_manager.fail("test completed")
