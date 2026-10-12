"""What every bundle layer shares: errors, progress, and reading a gallery database.

The error types, the progress callback type, the read chunk and worker-count
constants, safe basenames, and reading any gallery database (live or
unpacked from a bundle) read-only with its integrity and safety checks.
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

from . import reindex


class BundleError(ValueError):
    pass


class BundleChanged(BundleError):
    pass


def sentence(text: str) -> str:
    # `BundleError` and `CryptoError` say a lowercase clause ("the bundle
    # password is incorrect"); the page shows a sentence, on every route.
    text = text.strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:] + ("" if text.endswith((".", "!", "?")) else ".")


_CHUNK = 1024 * 1024
# Decoding every tracked PNG dominates both directions; it runs on a small
# pool while the archive is read. Admission is bounded by the decoded size
# the PNG header announces, so a few huge images cannot multiply memory.
_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

Progress = Callable[[str, int, int], None]


def _quiet(phase: str, done: int, total: int) -> None:
    pass


def _basename(name: str) -> bool:
    return (
        isinstance(name, str) and bool(name) and name not in {".", ".."}
        and not any(c in name for c in '/\\:\x00<>"|?*')
        and not any(ord(c) < 32 for c in name) and not name.endswith((".", " "))
        and name.split(".")[0].upper() not in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
    )


def _connect_read(path: Path):
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA trusted_schema=OFF")
    return conn


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    conn = _connect_read(path)
    try:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise BundleError("the gallery database failed its integrity check")
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(images)")}
        if reindex._DB_REQUIRED - columns:
            raise BundleError("the bundle has no usable Local Diffusion images table")
        rows = [dict(r) for r in conn.execute("SELECT * FROM images ORDER BY id")]
        for row in rows:
            if (not _basename(row["filename"]) or Path(row["filename"]).suffix.lower() != ".png"
                    or not isinstance(row["id"], int) or row["id"] <= 0):
                raise BundleError("the database contains an unsafe image filename or id")
        return rows
    except sqlite3.Error as exc:
        raise BundleError(f"cannot read the gallery database: {exc}") from exc
    finally:
        conn.close()
