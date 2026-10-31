"""Where a database's backup settings sidecar lives.

`backup_settings` reads and writes the file; this module only names it, so
every caller and test patch finds the same path. The file's first format
(version 1, a bare policy) is legacy: `backup_settings.load` reads it as
inactive and never activates it. Its writer survives only as a test fixture
(`tests/backup_helpers.py`) that plants such a file.
"""
from __future__ import annotations

from pathlib import Path

from . import legacy_names


def path_for(db_path: Path) -> Path:
    """Keep distinct database names in one directory independent of each other."""
    database = Path(db_path).expanduser()
    if not database.name:
        raise ValueError("backup preferences need a database filename")
    return legacy_names.sidecar(database, f"backup-policy-{database.name}.json")
