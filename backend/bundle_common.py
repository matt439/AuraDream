"""What every bundle layer shares: errors, progress, and reading a gallery database.

The error types, the progress callback type, the read chunk and worker-count
constants, safe basenames, and reading any gallery database (live or
unpacked from a bundle) read-only with its integrity and safety checks.
"""
from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from .bundle_file_lock import safe_basename


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
    return safe_basename(name)


# The original schema's required columns. Newer reproduction columns are read
# when present and become NULL when importing an older gallery.
_DB_REQUIRED = {
    "id", "filename", "prompt", "negative_prompt", "width", "height",
    "steps", "guidance_scale", "mu", "seed", "model_id", "duration_ms",
    "favorite", "created_at",
}


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
        if _DB_REQUIRED - columns:
            raise BundleError("the bundle has no usable AuraDream images table")
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


# A mask's name in a bundle and in `masks/`: its whole hash (`storage.mask_hash`).
MASK_HASH = re.compile(r"[0-9a-f]{64}")


def _mask_rows(path: Path) -> dict[str, dict]:
    """A gallery database's `masks` table by hash, read-only: empty for a
    database from before inpainting, which has none. A row that is not one
    this app writes - a hash of another shape, a size that is no size -
    refuses the whole database, as an unsafe filename does."""
    if not path.exists():
        return {}
    conn = _connect_read(path)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'masks'").fetchone() is None:
            return {}
        found = {}
        for row in conn.execute("SELECT hash, width, height, created_at FROM masks"):
            if (not isinstance(row["hash"], str) or not MASK_HASH.fullmatch(row["hash"])
                    or any(type(row[key]) is not int or row[key] <= 0 for key in ("width", "height"))):
                raise BundleError("the database contains an invalid inpaint mask record")
            found[row["hash"]] = dict(row)
        return found
    except sqlite3.Error as exc:
        raise BundleError(f"cannot read the gallery database's masks: {exc}") from exc
    finally:
        conn.close()
