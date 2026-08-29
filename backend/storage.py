"""Gallery persistence: SQLite index plus PNGs carrying their own metadata."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from PIL import Image, PngImagePlugin

from .config import settings
from .schemas import ImageRecord

if TYPE_CHECKING:
    from .schemas import GenerateRequest

_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL UNIQUE,
    prompt          TEXT    NOT NULL,
    negative_prompt TEXT,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    steps           INTEGER NOT NULL,
    guidance_scale  REAL    NOT NULL,
    mu              REAL,
    seed            INTEGER NOT NULL,
    model_id        TEXT    NOT NULL,
    duration_ms     INTEGER NOT NULL,
    favorite        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_created ON images (created_at DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts
    USING fts5(prompt, content='images', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, prompt) VALUES (new.id, new.prompt);
END;
CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, prompt) VALUES('delete', old.id, old.prompt);
END;
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # init_db() only runs at startup, but the file can disappear under a running
    # server - deleted by hand, moved, restored from a backup. Without this the
    # gallery 500s on every request until restart. The lookup is a sqlite_master
    # hit, cheap enough to pay per connection.
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='images'"
    ).fetchone() is None:
        conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _session():
    """A connection that commits on success and is always closed.

    `with sqlite3.connect(...) as conn` commits but does *not* close, which leaks
    a handle per call and keeps the file locked on Windows - deleting or moving
    krea.db then fails while the server runs.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _session() as conn:
        conn.executescript(_SCHEMA)


def _row_to_record(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        id=row["id"],
        filename=row["filename"],
        url=f"/outputs/{row['filename']}",
        prompt=row["prompt"],
        negative_prompt=row["negative_prompt"],
        width=row["width"],
        height=row["height"],
        steps=row["steps"],
        guidance_scale=row["guidance_scale"],
        seed=row["seed"],
        model_id=row["model_id"],
        duration_ms=row["duration_ms"],
        created_at=row["created_at"],
        favorite=bool(row["favorite"]),
    )


def save_image(
    image: Image.Image,
    request: "GenerateRequest",
    seed: int,
    duration_ms: int,
    model_id: str,
) -> ImageRecord:
    created = datetime.now(timezone.utc)
    filename = f"{created:%Y%m%d-%H%M%S}-{seed}-{uuid.uuid4().hex[:6]}.png"
    path = settings.outputs_dir / filename

    meta: dict[str, Any] = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "steps": request.steps,
        "guidance_scale": request.guidance_scale,
        "mu": request.mu,
        "seed": seed,
        "model": model_id,
        "size": f"{image.width}x{image.height}",
    }
    info = PngImagePlugin.PngInfo()
    # A1111-style single-line block, so other tools can read the params back.
    info.add_text(
        "parameters",
        f"{request.prompt}\n"
        f"Negative prompt: {request.negative_prompt or ''}\n"
        f"Steps: {request.steps}, CFG scale: {request.guidance_scale}, "
        f"Seed: {seed}, Size: {image.width}x{image.height}, Model: {model_id}",
    )
    info.add_text("krea", json.dumps(meta))
    image.save(path, format="PNG", pnginfo=info, compress_level=4)

    with _write_lock, _session() as conn:
        cur = conn.execute(
            """INSERT INTO images (filename, prompt, negative_prompt, width, height,
                                   steps, guidance_scale, mu, seed, model_id,
                                   duration_ms, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                filename,
                request.prompt,
                request.negative_prompt,
                image.width,
                image.height,
                request.steps,
                request.guidance_scale,
                request.mu,
                seed,
                model_id,
                duration_ms,
                created.isoformat(),
            ),
        )
        row = conn.execute("SELECT * FROM images WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _row_to_record(row)


def list_images(
    limit: int = 50, offset: int = 0, search: str | None = None, favorites_only: bool = False
) -> tuple[list[ImageRecord], int]:
    where, params = [], []
    if favorites_only:
        where.append("favorite = 1")
    if search:
        # Match the FTS index, falling back to a LIKE for punctuation-heavy input.
        where.append("id IN (SELECT rowid FROM images_fts WHERE images_fts MATCH ?)")
        params.append(" ".join(f'"{t}"' for t in search.split()))
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _session() as conn:
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM images {clause}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM images {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        except sqlite3.OperationalError:
            if not search:
                raise
            like = f"%{search}%"
            total = conn.execute(
                "SELECT COUNT(*) FROM images WHERE prompt LIKE ?", (like,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM images WHERE prompt LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (like, limit, offset),
            ).fetchall()
    return [_row_to_record(r) for r in rows], total


def get_image(image_id: int) -> ImageRecord | None:
    with _session() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    return _row_to_record(row) if row else None


def set_favorite(image_id: int, favorite: bool) -> ImageRecord | None:
    with _write_lock, _session() as conn:
        conn.execute("UPDATE images SET favorite = ? WHERE id = ?", (int(favorite), image_id))
    return get_image(image_id)


def delete_image(image_id: int) -> bool:
    record = get_image(image_id)
    if record is None:
        return False
    with _write_lock, _session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    (settings.outputs_dir / record.filename).unlink(missing_ok=True)
    return True


def summarise() -> dict[str, int]:
    """Counts for the clear-history preview."""
    with _session() as conn:
        total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        favorites = conn.execute("SELECT COUNT(*) FROM images WHERE favorite = 1").fetchone()[0]
    tracked = {r["filename"] for r in _tracked_files()}
    on_disk = {p.name for p in settings.outputs_dir.glob("*.png")}
    return {
        "records": total,
        "favorites": favorites,
        "files_on_disk": len(on_disk),
        "orphan_files": len(on_disk - tracked),
        "bytes_on_disk": sum(p.stat().st_size for p in settings.outputs_dir.glob("*.png")),
    }


def _tracked_files():
    with _session() as conn:
        return conn.execute("SELECT filename FROM images").fetchall()


def clear_history(keep_favorites: bool = False) -> dict[str, int]:
    """Delete image records and the PNGs they point at.

    Rows are deleted rather than the database file, so this is safe while the
    server holds the connection open - dropping the file mid-run would leave a
    locked handle on Windows and lose the schema.
    """
    where = " WHERE favorite = 0" if keep_favorites else ""
    with _write_lock, _session() as conn:
        doomed = [r["filename"] for r in conn.execute(f"SELECT filename FROM images{where}")]
        conn.execute(f"DELETE FROM images{where}")

    removed = 0
    for filename in doomed:
        path = settings.outputs_dir / filename
        if path.exists():
            path.unlink()
            removed += 1

    # Anything left with no row behind it: earlier deletions, interrupted runs.
    keep = {r["filename"] for r in _tracked_files()}
    orphans = 0
    for path in settings.outputs_dir.glob("*.png"):
        if path.name not in keep:
            path.unlink()
            orphans += 1

    return {"records": len(doomed), "files": removed, "orphans": orphans}


def vacuum() -> None:
    """Shrink the database file after a bulk delete."""
    conn = _connect()
    try:
        conn.isolation_level = None  # VACUUM cannot run inside a transaction
        conn.execute("VACUUM")
    finally:
        conn.close()
