"""One admitted startup check/export with separately consented post-success cleanup.

The caller supplies policy, stable identities, clock, transient protection and
an ownership-trust fact. No settings, credentials or defaults are selected here.
Only currently matching, nonempty managed copies cover age. Missing ownership
must not be reconstructed from archive names; a new namespace's trust must be
supplied explicitly. There is no retry loop or independent age-state sidecar.
Cleanup additionally requires the v3 permission and installed native protocol.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import HTTPException

from . import backup_receipts, bundles, gallery_lock, storage
from .api_bundles import BundleManager
from .backup_policy import BackupPolicy, CopyFacts, backup_due
from .backup_retention import RetentionOutcome, prune_after_success
from .backup_settings import RetentionPermission
from .bundle_leases import LeaseRegistry

State = Literal["idle", "disabled", "running", "skipped", "deferred", "failed", "unmanaged", "succeeded"]


@dataclass(frozen=True)
class BackupOutcome:
    state: State
    reason: str
    operation_id: str | None = None
    archive_name: str | None = None
    retention: RetentionOutcome = field(default_factory=RetentionOutcome)
    verified_at: str | None = None

    @property
    def clock_warning(self) -> bool:
        return self.reason == "clock_backwards"


class BackupCoordinator:
    def __init__(self, manager: BundleManager, *, db_path: Path, directory: Path, leases: LeaseRegistry):
        self.manager = manager
        self.db_path = Path(db_path)
        self.directory = Path(directory)
        self.leases = leases
        self._outcome = BackupOutcome("idle", "not_started")
        self._has_admitted_attempt = False

    def status(self) -> BackupOutcome:
        return self._outcome

    def _immediate(self, outcome: BackupOutcome) -> BackupOutcome:
        # A worker can compute its terminal outcome before BundleManager ends
        # admission. Keep the latest admitted outcome through that interval and
        # later refusals; only a newly admitted attempt replaces it.
        if not self._has_admitted_attempt:
            self._outcome = outcome
        return outcome

    def start_once(
        self, policy: BackupPolicy, identity: backup_receipts.ManagedIdentity, *,
        clock: Callable[[], datetime], password: str | None, ownership_trusted: bool,
        retention: RetentionPermission | None = None,
    ) -> BackupOutcome:
        """Reserve synchronously, then check/export in one owned worker.

        Disabled is a no-op. Enabled checks briefly reserve admission even
        when the worker finds no export due; the lifespan owns this worker.
        Protection errors never change mode or fall back to plaintext.
        """
        try:
            policy = BackupPolicy(policy.enabled, policy.minimum_age, policy.keep_last, policy.protection)
        except (AttributeError, ValueError, TypeError):
            return self._immediate(BackupOutcome("failed", "invalid_policy"))
        if not policy.enabled:
            return self._immediate(BackupOutcome("disabled", "disabled"))
        try:
            identity = backup_receipts.read_marker(identity.marker())
            if type(ownership_trusted) is not bool:
                raise ValueError("ownership must be an explicit boolean")
            if policy.protection == "encrypted":
                if not isinstance(password, str) or not password or len(password) > 1024:
                    raise ValueError("protection mismatch")
                password.encode("utf-8")
            elif password is not None:
                raise ValueError("protection mismatch")
        except (AttributeError, ValueError, TypeError):
            return self._immediate(BackupOutcome("failed", "invalid_identity_or_protection"))
        if not ownership_trusted:
            return self._immediate(BackupOutcome("skipped", "untrusted_inventory", identity.operation_id))
        # Catch invocation outside an event loop before reserving anything.
        asyncio.get_running_loop()
        try:
            self.manager.reserve("automatic_backup")
        except HTTPException:
            return self._immediate(BackupOutcome("deferred", "admission_busy", identity.operation_id))
        self._has_admitted_attempt = True
        self._outcome = BackupOutcome("running", "checking", identity.operation_id)

        def not_started(reason: str) -> None:
            self._outcome = BackupOutcome("failed", reason, identity.operation_id)

        try:
            self.manager.start(lambda: self._work(policy, identity, clock, password, retention), on_not_started=not_started)
        except Exception:
            self.manager.fail("The automatic backup worker could not start.")
            self._outcome = BackupOutcome("failed", "worker_start_failed", identity.operation_id)
        return self._outcome

    def _complete(self, outcome: BackupOutcome) -> dict:
        self._outcome = outcome
        return asdict(outcome)

    def _work(
        self, policy: BackupPolicy, identity: backup_receipts.ManagedIdentity,
        clock: Callable[[], datetime], password: str | None,
        retention: RetentionPermission | None = None,
    ) -> dict:
        try:
            # Bind the explicit source before creating a reservation anywhere.
            if self.db_path.resolve() != storage.settings.db_path.resolve():
                return self._complete(BackupOutcome("failed", "source_mismatch", identity.operation_id))
            with gallery_lock.hold(self.db_path, wait=gallery_lock.MAINTENANCE_GRACE):
                directory = storage.bundle_directory()
                if self.directory.resolve() != directory:
                    return self._complete(BackupOutcome("failed", "destination_mismatch", identity.operation_id))
                journal_state = backup_receipts.load_state(self.db_path, identity.gallery_id)
                journal = journal_state.receipts
                previous = next((receipt for receipt in journal_state.all_receipts
                                 if receipt.identity.operation_id == identity.operation_id), None)
                if previous is not None:
                    return self._complete(BackupOutcome("skipped", "operation_already_recorded",
                                                        identity.operation_id, previous.name))
                copies: list[CopyFacts] = []
                for receipt in journal:
                    if receipt.identity.destination_id != identity.destination_id:
                        continue
                    if Path(receipt.directory) != directory:
                        return self._complete(BackupOutcome("failed", "destination_identity_mismatch", identity.operation_id))
                    if receipt.protection != policy.protection or not (receipt.records or receipt.images):
                        continue
                    current = backup_receipts.matches_current(receipt, directory)
                    copies.append(CopyFacts(
                        receipt.identity.operation_id, identity.gallery_id, identity.destination_id,
                        receipt.verified_at, receipt.protection, "managed" if current else "unknown",
                        self.leases.leased(directory / receipt.name),
                    ))
                before = bundles.preview()
                due = backup_due(
                    policy, copies, now=clock(), gallery_id=identity.gallery_id,
                    destination_id=identity.destination_id,
                    gallery_nonempty=bool(before["records"] or before["images"]), ownership_trusted=True,
                )
                if not due.due:
                    return self._complete(BackupOutcome("skipped", due.reason, identity.operation_id))
                result = bundles.export_bundle(keep_gallery=True, managed=identity, password=password,
                                               progress=self.manager.report)
                name = result.get("name")
                if not isinstance(name, str):
                    raise ValueError("invalid export result")
                if result.get("managed") is not True:
                    return self._complete(BackupOutcome("unmanaged", "receipt_not_saved", identity.operation_id, name))
                # The exporter verified and journaled this copy. Do not read it
                # again or synthesize an independent successful-backup clock.
                cleanup = RetentionOutcome()
                verified_at = None
                if retention is not None:
                    try:
                        current = backup_receipts.load_state(self.db_path, identity.gallery_id)
                        fresh = next((item for item in current.receipts if item.identity == identity and item.name == name), None)
                        if fresh is None:
                            raise ValueError("new receipt missing")
                        verified_at = fresh.verified_at.isoformat()
                        cleanup = prune_after_success(self.db_path, retention, policy, identity, fresh,
                                                      now=clock(), previous_facts=copies)
                    except Exception:
                        cleanup = RetentionOutcome("warning", "journal_unusable", uncertain=True,
                                                   warning="The backup succeeded, but cleanup could not confirm its journal. No further cleanup was attempted.")
                return self._complete(BackupOutcome("succeeded", "verified_publication", identity.operation_id, name, cleanup, verified_at))
        except gallery_lock.GalleryBusy:
            return self._complete(BackupOutcome("deferred", "gallery_busy", identity.operation_id))
        except backup_receipts.ReceiptError:
            return self._complete(BackupOutcome("failed", "journal_unusable", identity.operation_id))
        except Exception:
            # Provider, filesystem and injected errors may contain a secret;
            # status stores a fixed reason, never their message or traceback.
            return self._complete(BackupOutcome("failed", "backup_failed", identity.operation_id))

    async def finish(self) -> BackupOutcome:
        await self.manager.finish()
        return self.status()
