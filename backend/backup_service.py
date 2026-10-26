"""App startup/configuration boundary for explicitly enabled plaintext backups.

No pruning or credential provider is installed. Status is an in-memory snapshot:
it remains readable during maintenance without taking the gallery reservation.
"""
from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import backup_receipts, backup_settings, gallery_lock, storage
from .api_bundles import BundleManager
from .backup_coordinator import BackupCoordinator
from .bundle_leases import LeaseRegistry
from .schemas_backup import BackupSettingsRequest, BackupStatus


class BackupService:
    def __init__(self, manager: BundleManager, db_path: Path):
        self.manager = manager
        self.db_path = db_path
        self.loaded = backup_settings.LoadedSettings(backup_settings.BackupSettings(), "missing")
        self.directory: Path | None = None
        self.directory_warning: str | None = None
        self.coordinator: BackupCoordinator | None = None
        self.verified_at: str | None = None
        self.reading = False
        self._attempt_gallery_id: str | None = None
        self._attempt_destination_id: str | None = None
        self._state_lock = threading.Lock()

    def prepare(self) -> None:
        """Read settings and destination without creating directories or locks."""
        loaded = backup_settings.load(self.db_path)
        try:
            directory = storage.bundle_directory()
            warning = None
        except Exception:
            directory = None
            warning = "The backup destination is unavailable. Automatic backups remain inactive; repair the configured bundles directory."
        # Publish only immutable snapshots under this short lock; no file read,
        # gallery reservation or hashing holds up the memory-only status route.
        with self._state_lock:
            self.loaded, self.directory, self.directory_warning = loaded, directory, warning
            if ((loaded.settings.gallery_id, loaded.settings.destination_id) != (self._attempt_gallery_id, self._attempt_destination_id)
                    or not self._destination_matches(loaded.settings, directory)):
                self.verified_at = None

    @staticmethod
    def _destination_matches(saved: backup_settings.BackupSettings, directory: Path | None) -> bool:
        return saved.directory is not None and directory is not None and os.path.normcase(saved.directory) == os.path.normcase(str(directory))

    def status(self) -> BackupStatus:
        with self._state_lock:
            loaded, directory, directory_warning = self.loaded, self.directory, self.directory_warning
            coordinator, verified_at = self.coordinator, self.verified_at
        saved = loaded.settings
        destination_matches = self._destination_matches(saved, directory)
        enabled = saved.enabled and directory is not None and destination_matches
        warning = loaded.warning or directory_warning
        state, reason = ("scheduled", "next_start") if enabled else ("disabled", "disabled")
        confirmation = loaded.requires_confirmation
        if saved.enabled and directory is not None and not destination_matches:
            warning = "The backup destination changed. Confirm the new destination to enable a new managed set; old archives are kept."
            state, reason, confirmation = "needs_confirmation", "destination_changed", True
        elif loaded.requires_confirmation:
            state, reason = "needs_confirmation", "legacy_settings" if loaded.legacy else "settings_unreadable"
        elif directory_warning:
            state, reason = "unavailable", "destination_unavailable"
        archive_name = None
        if coordinator is not None:
            outcome = coordinator.status()
            state, reason, archive_name = outcome.state, outcome.reason, outcome.archive_name
        return BackupStatus(
            revision=loaded.revision, enabled=enabled,
            minimum_age_hours=saved.minimum_age_hours, keep_last=saved.keep_last,
            protection=saved.protection,
            destination=str(directory) if directory is not None else None,
            warning=warning, state=state, reason=reason, archive_name=archive_name,
            verified_at=verified_at, requires_confirmation=confirmation,
        )

    def start_once(self) -> None:
        """Called synchronously before lifespan yields; no automatic retries."""
        if not self.status().enabled:
            return
        with self._state_lock:
            saved, directory = self.loaded.settings, self.directory
        assert saved.gallery_id is not None and saved.destination_id is not None and directory is not None
        coordinator = BackupCoordinator(self.manager, db_path=self.db_path, directory=directory, leases=LeaseRegistry())
        with self._state_lock:
            self.coordinator = coordinator
            self._attempt_gallery_id, self._attempt_destination_id = saved.gallery_id, saved.destination_id
        coordinator.start_once(
            saved.policy(), backup_receipts.ManagedIdentity(saved.gallery_id, saved.destination_id, str(uuid4())),
            clock=lambda: datetime.now(timezone.utc), password=None, ownership_trusted=True,
        )

    async def finish(self) -> None:
        with self._state_lock:
            coordinator, gallery_id, destination_id = self.coordinator, self._attempt_gallery_id, self._attempt_destination_id
        if coordinator is not None:
            await coordinator.finish()
            stamp = await asyncio.to_thread(self._verification_time, coordinator, gallery_id)
            with self._state_lock:
                # A configuration worker may have cleared/replaced this attempt
                # while the receipt was read. Never label the next policy with
                # a completed prior attempt's timestamp.
                if (self.coordinator is coordinator
                        and self.loaded.settings.gallery_id == gallery_id
                        and self.loaded.settings.destination_id == destination_id
                        and self.directory == coordinator.directory
                        and self._destination_matches(self.loaded.settings, self.directory)):
                    self.verified_at = stamp

    def _verification_time(self, coordinator: BackupCoordinator, gallery_id: str | None) -> str | None:
        # Read the new durable receipt once, never hash/reverify its archive.
        outcome = coordinator.status()
        if outcome.state != "succeeded":
            return None
        assert gallery_id is not None
        try:
            found = next((receipt for receipt in backup_receipts.load(self.db_path, gallery_id)
                          if receipt.identity.operation_id == outcome.operation_id), None)
            if found is not None:
                return found.verified_at.isoformat()
        except backup_receipts.ReceiptError:
            pass  # The successful worker outcome survives a later damaged journal.
        return None

    def configure(self, request: BackupSettingsRequest) -> dict:
        """Owned manager worker: reserve before reading/writing the policy."""
        with gallery_lock.hold(self.db_path, wait=gallery_lock.MAINTENANCE_GRACE):
            # Turning protection off must remain possible even after an invalid
            # destination was configured. The disabled writer ignores this path.
            with self._state_lock:
                previous_directory = self.directory
            directory = storage.bundle_directory() if request.enabled else previous_directory or self.db_path.parent / "bundles"
            if request.enabled and request.destination != str(directory):
                raise backup_settings.SettingsConflict("The backup destination changed. Reload the settings and confirm the displayed destination.")
            try:
                loaded = backup_settings.configure(
                    self.db_path, revision=request.revision, enabled=request.enabled, directory=directory,
                    minimum_age_hours=request.minimum_age_hours, keep_last=request.keep_last,
                    allow_unencrypted=request.allow_unencrypted,
                )
            except Exception:
                # Reconcile an uncertain publication before another request may
                # retry. GET remains a cached snapshot, including after failure.
                reconciled = backup_settings.load(self.db_path)
                with self._state_lock:
                    self.loaded = reconciled
                    if (reconciled.settings.gallery_id, reconciled.settings.destination_id) != (self._attempt_gallery_id, self._attempt_destination_id):
                        self.verified_at = None
                raise
            else:
                with self._state_lock:
                    self.loaded = loaded
                    if request.enabled:
                        self.directory, self.directory_warning = directory, None
                    self.coordinator, self.verified_at = None, None
                    self._attempt_gallery_id, self._attempt_destination_id = None, None
        return {"configured": True}
