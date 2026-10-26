"""Explicit startup-backup activation; legacy policy files never activate.

Version 2 records a new, explicit plaintext acknowledgement and durable scope.
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

from . import backup_preferences, backup_receipts
from .backup_policy import MAX_KEEP_LAST, BackupPolicy

MAX_AGE_HOURS = 876_000
_MAX_BYTES = 64 * 1024
_FIELDS = {"version", "enabled", "minimum_age_hours", "keep_last", "protection",
           "allow_unencrypted", "gallery_id", "destination_id", "directory"}
_UNREADABLE = "Automatic backup settings could not be read. Backups remain disabled; repair this database's backup-policy file."
_LEGACY = "Earlier backup settings are inactive. Choose the settings and explicitly confirm unencrypted startup backups to enable them."


class SettingsError(ValueError):
    """Fixed public errors; never include file contents or exception details."""


class SettingsConflict(SettingsError):
    pass


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

    def policy(self) -> BackupPolicy:
        return BackupPolicy(
            self.enabled,
            timedelta(hours=self.minimum_age_hours) if self.minimum_age_hours is not None else None,
            self.keep_last, "unencrypted" if self.protection == "unencrypted" else None,
        )


@dataclass(frozen=True)
class LoadedSettings:
    settings: BackupSettings
    revision: str | None
    warning: str | None = None
    requires_confirmation: bool = False
    legacy: bool = False


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
    if not isinstance(value, dict) or value.keys() != _FIELDS or type(value["version"]) is not int or value["version"] != 2:
        raise ValueError("unknown settings")
    if type(value["enabled"]) is not bool or type(value["allow_unencrypted"]) is not bool:
        raise ValueError("invalid flag")
    age, count = value["minimum_age_hours"], value["keep_last"]
    if age is not None and (type(age) is not int or not 1 <= age <= MAX_AGE_HOURS):
        raise ValueError("invalid age")
    if count is not None and (type(count) is not int or not 1 <= count <= MAX_KEEP_LAST):
        raise ValueError("invalid count")
    if value["protection"] not in (None, "unencrypted"):
        raise ValueError("unsupported protection")
    gallery, destination, directory = value["gallery_id"], value["destination_id"], value["directory"]
    if any(item is not None for item in (gallery, destination, directory)):
        backup_receipts.ManagedIdentity(gallery, destination, "00000000-0000-0000-0000-000000000001")
        if not isinstance(directory, str) or not directory or len(directory) > 32768 or not Path(directory).is_absolute():
            raise ValueError("invalid directory")
    result = BackupSettings(value["enabled"], age, count, value["protection"], value["allow_unencrypted"], gallery, destination, directory)
    result.policy()
    if result.enabled and (not result.allow_unencrypted or result.protection != "unencrypted" or result.gallery_id is None):
        raise ValueError("activation is incomplete")
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
        if isinstance(payload, dict) and payload.get("version") != 2:
            revision = None  # Never replace a newer/unknown format through this API.
        return LoadedSettings(_decode(payload), revision)
    except FileNotFoundError:
        return LoadedSettings(BackupSettings(), "missing")
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return LoadedSettings(BackupSettings(), revision, _UNREADABLE, True)


def configure(
    db_path: Path, *, revision: str, enabled: bool, directory: Path,
    minimum_age_hours: int | None = None, keep_last: int | None = None,
    allow_unencrypted: bool = False,
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
    if type(enabled) is not bool or type(allow_unencrypted) is not bool:
        raise SettingsError("Choose whether automatic backups are enabled.")
    previous = current.settings
    if not enabled:
        chosen = replace(previous, enabled=False)
    else:
        if not allow_unencrypted:
            raise SettingsError("Explicitly allow unencrypted archives, including prompts, before enabling startup backups.")
        if type(minimum_age_hours) is not int or not 1 <= minimum_age_hours <= MAX_AGE_HOURS:
            raise SettingsError(f"Choose a whole number of hours from 1 to {MAX_AGE_HOURS}.")
        if type(keep_last) is not int or not 1 <= keep_last <= MAX_KEEP_LAST:
            raise SettingsError(f"Choose a future retention count from 1 to {MAX_KEEP_LAST}.")
        canonical = str(directory.resolve())
        same_destination = previous.directory is not None and os.path.normcase(previous.directory) == os.path.normcase(canonical)
        gallery_id = previous.gallery_id or str(uuid4())
        destination_id = previous.destination_id if same_destination else str(uuid4())
        # Existing journals are never adopted into a newly invented gallery ID.
        try:
            backup_receipts.load(db_path, gallery_id)
        except backup_receipts.ReceiptError:
            raise SettingsError("The managed backup journal needs repair before automatic backups can be enabled. Existing archives were kept.") from None
        chosen = BackupSettings(True, minimum_age_hours, keep_last, "unencrypted", True,
                                gallery_id, destination_id, canonical)
    payload = {"version": 2, **chosen.__dict__}
    _decode(payload)
    data = (json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > _MAX_BYTES:
        raise SettingsError("The backup settings exceed the supported file size; choose a shorter destination path.")
    path = backup_preferences.path_for(db_path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".local-diffusion.backup-policy.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return LoadedSettings(chosen, hashlib.sha256(data).hexdigest())
