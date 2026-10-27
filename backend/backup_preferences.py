"""Default-off, database-specific backup policy persistence, without activation.

Version 1 stores the complete policy with nullable ``minimum_age_us``,
``keep_last`` and ``protection`` fields. Durations are integer microseconds,
computed from timedelta's integer parts, so even timedelta.max round-trips
without floating-point loss. Reads and writes share BackupPolicy's retention
ceiling. There are no age, retention or protection defaults and no credential
fields. These helpers neither export nor delete anything.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Never

from . import legacy_names
from .backup_policy import BackupPolicy, Protection

_VERSION = 1
_MAX_BYTES = 64 * 1024
_FIELDS = {"version", "enabled", "minimum_age_us", "keep_last", "protection"}
_WARNING = (
    "Automatic backup settings could not be read. Automatic backups remain "
    "disabled; repair or replace this database's backup-policy file."
)


@dataclass(frozen=True)
class LoadedPolicy:
    policy: BackupPolicy
    warning: str | None = None


def path_for(db_path: Path) -> Path:
    """Keep distinct database names in one directory independent of each other."""
    database = Path(db_path).expanduser()
    if not database.name:
        raise ValueError("backup preferences need a database filename")
    return legacy_names.sidecar(database, f"backup-policy-{database.name}.json")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate settings field")
        result[key] = value
    return result


def _refuse_constant(_value: str) -> Never:
    raise ValueError("non-finite settings number")


def _decode(payload: object) -> BackupPolicy:
    if not isinstance(payload, dict) or payload.keys() != _FIELDS:
        raise ValueError("backup settings need the complete known schema")
    if type(payload["version"]) is not int or payload["version"] != _VERSION:
        raise ValueError("unsupported backup settings version")
    enabled = payload["enabled"]
    if type(enabled) is not bool:
        raise ValueError("backup enabled must be a boolean")
    age_us = payload["minimum_age_us"]
    if age_us is not None and type(age_us) is not int:
        raise ValueError("backup age must be integer microseconds")
    age = None if age_us is None else timedelta(microseconds=age_us)
    keep_last = payload["keep_last"]
    if keep_last is not None and type(keep_last) is not int:
        raise ValueError("backup retention must be an integer")
    raw_protection = payload["protection"]
    protection: Protection | None
    if raw_protection is None:
        protection = None
    elif raw_protection == "unencrypted":
        protection = "unencrypted"
    elif raw_protection == "encrypted":
        protection = "encrypted"
    else:
        raise ValueError("unknown backup protection")
    return BackupPolicy(enabled, age, keep_last, protection)


def load(db_path: Path) -> LoadedPolicy:
    """Read at most 64 KiB; missing settings are disabled, bad settings warn.

    Loading never creates, repairs or overwrites a file. Warning text is fixed:
    it includes no payload, exception details, paths or credential material.
    """
    try:
        with path_for(db_path).open("rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise ValueError("backup settings exceed the read limit")
        payload: object = json.loads(
            data.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_refuse_constant,
        )
        return LoadedPolicy(_decode(payload))
    except FileNotFoundError:
        return LoadedPolicy(BackupPolicy())
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return LoadedPolicy(BackupPolicy(), _WARNING)


def save(db_path: Path, policy: BackupPolicy) -> None:
    """Atomically replace this database's sidecar, propagating write failures.

    Only this call's same-directory temporary is cleaned up. Until replacement
    succeeds, a failed write, fsync or replace leaves previous settings intact.
    Concurrent saves are complete replacements; coordination belongs to callers.
    """
    if not isinstance(policy, BackupPolicy):
        raise TypeError("backup preferences need a BackupPolicy")
    # Revalidate unusual construction without serializing arbitrary attributes.
    checked = BackupPolicy(
        policy.enabled, policy.minimum_age, policy.keep_last, policy.protection,
    )
    age = checked.minimum_age
    age_us = None if age is None else (
        (age.days * 86400 + age.seconds) * 1_000_000 + age.microseconds
    )
    data = (json.dumps({
        "version": _VERSION,
        "enabled": checked.enabled,
        "minimum_age_us": age_us,
        "keep_last": checked.keep_last,
        "protection": checked.protection,
    }, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > _MAX_BYTES:
        raise ValueError("backup settings exceed the read limit")
    path = path_for(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".auradream.backup-policy.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
