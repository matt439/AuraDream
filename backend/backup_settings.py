"""Explicit startup-backup activation; legacy policy files never activate.

Version 2 records plaintext acknowledgement/scope, v3 explicit retention and v4
inactive/ready opaque Windows credential profiles with separate activation.
Reading missing/disabled files creates nothing. This is an accidental-activation
guard, not authentication against someone editing this server's local files.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from . import backup_credentials, backup_preferences, backup_receipts, bundle_mode
from .backup_policy import MAX_KEEP_LAST, BackupPolicy

MAX_AGE_HOURS = 876_000
_MAX_BYTES = 64 * 1024
_FIELDS = {"version", "enabled", "minimum_age_hours", "keep_last", "protection",
           "allow_unencrypted", "gallery_id", "destination_id", "directory"}
_FIELDS_V3 = _FIELDS | {"retention"}
_FIELDS_V4 = _FIELDS_V3 | {"credential_reference", "credential_state", "allow_encrypted", "recovery_confirmed"}
_UNREADABLE = "Automatic backup settings could not be read. Backups remain disabled; repair this database's backup-policy file."
_LEGACY = "Earlier backup settings are inactive. Choose the settings and explicitly confirm unencrypted startup backups to enable them."


class SettingsError(ValueError):
    """Fixed public errors; never include file contents or exception details."""


class SettingsConflict(SettingsError):
    pass


@dataclass(frozen=True)
class RetentionPermission:
    gallery_id: str
    destination_id: str
    directory: str
    keep_last: int
    protection: str
    mode: bundle_mode.SharedMode


def mode_revision(mode: bundle_mode.SharedMode) -> str:
    data = json.dumps(backup_receipts.shared_mode_wire(mode), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _retention_wire(permission: RetentionPermission) -> dict:
    return {"allow_managed_deletion": True, "gallery_id": permission.gallery_id,
            "destination_id": permission.destination_id, "directory": permission.directory,
            "keep_last": permission.keep_last, "protection": permission.protection,
            "mode": backup_receipts.shared_mode_wire(permission.mode)}


@dataclass(frozen=True)
class BackupSettings:
    enabled: bool = False
    minimum_age_hours: int | None = None
    keep_last: int | None = None
    protection: str | None = None
    allow_unencrypted: bool = False
    gallery_id: str | None = None
    destination_id: str | None = None
    directory: str | None = None
    retention: RetentionPermission | None = None
    credential_reference: backup_credentials.CredentialReference | None = None
    credential_state: str = "not_configured"
    allow_encrypted: bool = False
    recovery_confirmed: bool = False

    def policy(self) -> BackupPolicy:
        return BackupPolicy(
            self.enabled,
            timedelta(hours=self.minimum_age_hours) if self.minimum_age_hours is not None else None,
            self.keep_last, "encrypted" if self.protection == "encrypted" else "unencrypted" if self.protection == "unencrypted" else None,
        )


@dataclass(frozen=True)
class LoadedSettings:
    settings: BackupSettings
    revision: str | None
    warning: str | None = None
    requires_confirmation: bool = False
    legacy: bool = False
    version: int = 2


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise ValueError("nonfinite field")


def _decode(value: object) -> BackupSettings:
    if (not isinstance(value, dict) or type(value.get("version")) is not int
            or value.get("version") not in (2, 3, 4)
            or value.keys() != (_FIELDS if value["version"] == 2 else _FIELDS_V3 if value["version"] == 3 else _FIELDS_V4)):
        raise ValueError("unknown settings")
    if type(value["enabled"]) is not bool or type(value["allow_unencrypted"]) is not bool:
        raise ValueError("invalid flag")
    age, count = value["minimum_age_hours"], value["keep_last"]
    if age is not None and (type(age) is not int or not 1 <= age <= MAX_AGE_HOURS):
        raise ValueError("invalid age")
    if count is not None and (type(count) is not int or not 1 <= count <= MAX_KEEP_LAST):
        raise ValueError("invalid count")
    if value["protection"] not in ((None, "unencrypted", "encrypted") if value["version"] == 4 else (None, "unencrypted")):
        raise ValueError("unsupported protection")
    gallery, destination, directory = value["gallery_id"], value["destination_id"], value["directory"]
    if any(item is not None for item in (gallery, destination, directory)):
        backup_receipts.ManagedIdentity(gallery, destination, "00000000-0000-0000-0000-000000000001")
        if not isinstance(directory, str) or not directory or len(directory) > 32768 or not Path(directory).is_absolute():
            raise ValueError("invalid directory")
    result = BackupSettings(value["enabled"], age, count, value["protection"], value["allow_unencrypted"], gallery, destination, directory)
    if value["version"] == 4:
        if (type(value["allow_encrypted"]) is not bool or type(value["recovery_confirmed"]) is not bool
                or value["credential_state"] not in {"not_configured", "pending", "ready"}):
            raise ValueError("invalid credential profile")
        reference = backup_credentials.read_reference(value["credential_reference"]) if value["credential_reference"] is not None else None
        state = value["credential_state"]
        if ((reference is None) != (state == "not_configured")
                or (reference is not None and (reference.gallery_id != gallery or reference.destination_id != destination))):
            raise ValueError("credential scope differs")
        result = replace(result, credential_reference=reference, credential_state=state,
                         allow_encrypted=value["allow_encrypted"], recovery_confirmed=value["recovery_confirmed"])
    result.policy()
    encrypted = result.protection == "encrypted"
    if result.enabled and (result.gallery_id is None or (encrypted and (
            not result.allow_encrypted or not result.recovery_confirmed or result.credential_state != "ready"
            or result.credential_reference is None or result.allow_unencrypted))
            or (not encrypted and (not result.allow_unencrypted or result.protection != "unencrypted"))):
        raise ValueError("activation is incomplete")
    if result.credential_state == "pending" and result.enabled:
        raise ValueError("pending setup cannot activate")
    retention = value.get("retention")
    if retention is not None:
        fields = {"allow_managed_deletion", "gallery_id", "destination_id", "directory", "keep_last", "protection", "mode"}
        if (not isinstance(retention, dict) or retention.keys() != fields or retention["allow_managed_deletion"] is not True
                or not result.enabled or retention["gallery_id"] != gallery or retention["destination_id"] != destination
                or retention["directory"] != directory or type(retention["keep_last"]) is not int
                or retention["keep_last"] != count or retention["protection"] != result.protection):
            raise ValueError("retention permission is incomplete")
        mode = backup_receipts.read_shared_mode(retention["mode"])
        if os.path.normcase(str(mode.directory)) != os.path.normcase(str(directory)):
            raise ValueError("retention destination differs")
        assert isinstance(gallery, str) and isinstance(destination, str) and isinstance(directory, str) and isinstance(count, int)
        result = replace(result, retention=RetentionPermission(gallery, destination, directory, count, str(result.protection), mode))
    return result


def load(db_path: Path) -> LoadedSettings:
    """Bounded, strict, read-only load; no implicit legacy activation/migration."""
    path = backup_preferences.path_for(db_path)
    revision: str | None = None
    try:
        if path.is_symlink() or path.is_junction():
            raise ValueError("linked settings")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_BYTES:
            raise ValueError("invalid settings file")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            data = stream.read(_MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
        ) or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) > _MAX_BYTES:
            raise ValueError("settings changed while reading")
        revision = hashlib.sha256(data).hexdigest()
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        if isinstance(payload, dict) and type(payload.get("version")) is int and payload["version"] == 1:
            # Version 1 was deliberately inactive, even when hand-edited enabled.
            return LoadedSettings(BackupSettings(), revision, _LEGACY, True, True)
        if isinstance(payload, dict) and payload.get("version") not in (2, 3, 4):
            revision = None  # Never replace a newer/unknown format through this API.
        try:
            settings = _decode(payload)
        except (ValueError, TypeError):
            if isinstance(payload, dict) and payload.get("version") == 4:
                revision = None  # Preserve a damaged provisioning target for explicit repair.
            raise
        assert isinstance(payload, dict)
        return LoadedSettings(settings, revision, version=payload["version"])
    except FileNotFoundError:
        return LoadedSettings(BackupSettings(), "missing")
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return LoadedSettings(BackupSettings(), revision, _UNREADABLE, True)


def configure(
    db_path: Path, *, revision: str, enabled: bool, directory: Path,
    minimum_age_hours: int | None = None, keep_last: int | None = None,
    allow_unencrypted: bool = False,
    retention_enabled: bool = False, allow_managed_deletion: bool = False,
    expected_mode_revision: str | None = None,
    protection: str | None = None, allow_encrypted: bool = False, confirm_recovery: bool = False,
) -> LoadedSettings:
    """Save under the caller's gallery reservation, never export or delete.

    Enabling always needs fresh consent and complete values. Existing valid
    scopes survive disabling; a changed destination gets a new managed set.
    """
    current = load(db_path)
    if current.revision is None:
        raise SettingsError("Repair the unreadable or unsupported backup settings file before changing automatic backups.")
    if revision != current.revision:
        raise SettingsConflict("Backup settings changed in another tab or process. Reload them before saving.")
    if any(type(flag) is not bool for flag in (enabled, allow_unencrypted, retention_enabled, allow_managed_deletion, allow_encrypted, confirm_recovery)):
        raise SettingsError("Choose whether automatic backups are enabled.")
    previous = current.settings
    if not enabled:
        chosen = replace(previous, enabled=False, retention=None)
    else:
        if previous.credential_state == "pending":
            raise SettingsError("A credential setup has an unresolved pending target. Inspect its outcome before enabling or changing protection; disabling remains available.")
        selected = protection or "unencrypted"
        if selected not in {"unencrypted", "encrypted"}:
            raise SettingsError("Choose a supported automatic backup protection mode.")
        encrypted = selected == "encrypted"
        if not encrypted and not allow_unencrypted:
            raise SettingsError("Explicitly allow unencrypted archives, including prompts, before enabling startup backups.")
        if encrypted and (not allow_encrypted or not confirm_recovery):
            raise SettingsError("Explicitly allow unattended encrypted backups and confirm independent password recovery before enabling them.")
        if type(minimum_age_hours) is not int or not 1 <= minimum_age_hours <= MAX_AGE_HOURS:
            raise SettingsError(f"Choose a whole number of hours from 1 to {MAX_AGE_HOURS}.")
        if type(keep_last) is not int or not 1 <= keep_last <= MAX_KEEP_LAST:
            raise SettingsError(f"Choose a future retention count from 1 to {MAX_KEEP_LAST}.")
        canonical = str(directory.resolve())
        same_destination = (previous.directory is not None and os.path.normcase(previous.directory) == os.path.normcase(canonical)
                            and previous.protection == selected)
        if encrypted and (not same_destination or previous.credential_state != "ready" or previous.credential_reference is None):
            raise SettingsError("Complete the dedicated Windows credential setup for this exact destination before enabling encrypted backups.")
        gallery_id = previous.gallery_id or str(uuid4())
        destination_id = previous.destination_id if same_destination else str(uuid4())
        # Existing journals are never adopted into a newly invented gallery ID.
        try:
            backup_receipts.load(db_path, gallery_id)
        except backup_receipts.ReceiptError:
            raise SettingsError("The managed backup journal needs repair before automatic backups can be enabled. Existing archives were kept.") from None
        chosen = BackupSettings(True, minimum_age_hours, keep_last, selected, not encrypted,
                                gallery_id, destination_id, canonical)
        if encrypted:
            chosen = replace(chosen, credential_reference=previous.credential_reference, credential_state="ready",
                             allow_encrypted=True, recovery_confirmed=True)
        if retention_enabled:
            if not allow_managed_deletion:
                raise SettingsError("Explicitly acknowledge irreversible deletion of older managed copies before enabling cleanup.")
            try:
                installed = bundle_mode.discover(directory)
                if installed is None:
                    raise ValueError("mode missing")
                actual_revision = mode_revision(installed)
            except Exception:
                raise SettingsError("Cleanup requires the explicit Windows fixed-local-NTFS shared mode. Read the saved settings after installing it separately.") from None
            if not expected_mode_revision or expected_mode_revision != actual_revision:
                raise SettingsConflict("The installed shared mode changed. Reload its identity and confirm deletion again before saving.")
            if backup_receipts.load_state(db_path, gallery_id).pending is not None:
                raise SettingsError("An earlier cleanup has an unresolved pending intent. Repair that journal before enabling cleanup; existing files were kept or their removal is uncertain.")
            chosen = replace(chosen, retention=RetentionPermission(gallery_id, str(destination_id), canonical,
                                                                  keep_last, selected, installed))
    version = 4 if current.version == 4 else 3 if chosen.retention is not None or current.version == 3 else 2
    return _persist(db_path, chosen, version)


def _persist(db_path: Path, chosen: BackupSettings, version: int) -> LoadedSettings:
    payload = {"version": version, **{key: getattr(chosen, key) for key in _FIELDS - {"version"}}}
    if version >= 3:
        payload["retention"] = _retention_wire(chosen.retention) if chosen.retention is not None else None
    if version == 4:
        payload.update(credential_reference=backup_credentials.reference_wire(chosen.credential_reference) if chosen.credential_reference else None,
                       credential_state=chosen.credential_state, allow_encrypted=chosen.allow_encrypted,
                       recovery_confirmed=chosen.recovery_confirmed)
    _decode(payload)
    data = (json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > _MAX_BYTES:
        raise SettingsError("The backup settings exceed the supported file size; choose a shorter destination path.")
    path = backup_preferences.path_for(db_path)
    temporary: Path | None = None
    owned: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".auradream.backup-policy.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            value = os.fstat(stream.fileno())
            owned = value.st_dev, value.st_ino
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                value = temporary.lstat()
                if (value.st_dev, value.st_ino) == owned:
                    temporary.unlink()
            except FileNotFoundError:
                pass
    return LoadedSettings(chosen, hashlib.sha256(data).hexdigest(), version=version)


def save_encrypted_profile(db_path: Path, *, revision: str, directory: Path,
                           reference: backup_credentials.CredentialReference, recovery_confirmed: bool) -> LoadedSettings:
    """Persist secret-free pending intent BEFORE any credential write; never activate."""
    current = load(db_path)
    if (current.revision is None or current.revision != revision
            or (current.warning is not None and not current.legacy)):
        raise SettingsConflict("Backup settings changed or are unusable. Inspect them before another credential setup.")
    previous = current.settings
    if previous.credential_state == "pending":
        raise SettingsError("An earlier credential setup is pending. Inspect its exact reference; do not retry automatically.")
    reference = backup_credentials.read_reference(backup_credentials.reference_wire(reference))
    if recovery_confirmed is not True or (previous.gallery_id is not None and reference.gallery_id != previous.gallery_id):
        raise SettingsError("Confirm independent recovery and the existing gallery identity before credential setup.")
    if reference.destination_id == previous.destination_id:
        raise SettingsError("Credential setup requires a fresh managed-set identity; old entries are never overwritten.")
    backup_receipts.load_state(db_path, reference.gallery_id)
    chosen = BackupSettings(False, previous.minimum_age_hours, previous.keep_last, "encrypted", False,
                            reference.gallery_id, reference.destination_id, str(directory.resolve()), None,
                            reference, "pending", False, True)
    return _persist(db_path, chosen, 4)


def complete_encrypted_profile(db_path: Path, *, revision: str,
                               reference: backup_credentials.CredentialReference) -> LoadedSettings:
    """A successful explicit vault writer may mark its exact inactive intent ready."""
    current = load(db_path)
    if (current.revision != revision or current.settings.credential_state != "pending"
            or current.settings.credential_reference != reference):
        raise SettingsConflict("The credential setup profile changed. Inspect its outcome before another action.")
    return _persist(db_path, replace(current.settings, credential_state="ready"), 4)
