"""Plant a legacy (version 1) backup policy sidecar for the tests that check
it stays inactive. The app no longer writes this format; this is the writer
`backend/backup_preferences.py` used to have, minus its atomic replace."""
from __future__ import annotations

import json
from pathlib import Path

from backend import backup_preferences
from backend.backup_policy import BackupPolicy


def save_v1_policy(db_path: Path, policy: BackupPolicy) -> Path:
    """Write `policy` in the version-1 wire format; return the sidecar's path."""
    age = policy.minimum_age
    age_us = None if age is None else (
        (age.days * 86400 + age.seconds) * 1_000_000 + age.microseconds
    )
    path = backup_preferences.path_for(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps({
        "version": 1,
        "enabled": policy.enabled,
        "minimum_age_us": age_us,
        "keep_last": policy.keep_last,
        "protection": policy.protection,
    }, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))
    return path
