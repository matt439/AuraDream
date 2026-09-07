"""Gallery persistence: SQLite index plus PNGs carrying their own metadata."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from PIL import Image, PngImagePlugin

from .config import settings
from .schemas import ImageRecord

if TYPE_CHECKING:
    from .grid import CellKey
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
    created_at      TEXT    NOT NULL,
    job_id          TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_created ON images (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_images_job ON images (job_id);
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts
    USING fts5(prompt, content='images', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, prompt) VALUES (new.id, new.prompt);
END;
CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, prompt) VALUES('delete', old.id, old.prompt);
END;
"""


# Set once the running process has checked; the ALTER is a schema change, not
# something to attempt on every connection.
_migrated = False


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that a database written by an older build has not got.

    `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it was, so
    a gallery from before this column simply never gains it - and every INSERT
    then fails. Cheap enough to check once per process; the ALTER itself runs
    at most once, and races with another process end in "duplicate column",
    which is the outcome we wanted anyway.
    """
    global _migrated
    if _migrated:
        return
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
    if "job_id" not in columns:
        try:
            conn.execute("ALTER TABLE images ADD COLUMN job_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_job ON images (job_id)")
        except sqlite3.OperationalError:  # another process got there first
            pass
    _migrated = True


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
    _migrate(conn)
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
        # Databases written before mu was indexed have no such column.
        mu=row["mu"] if "mu" in row.keys() else None,
        seed=row["seed"],
        model_id=row["model_id"],
        # Rows written before runs were grouped belong to no job.
        job_id=row["job_id"] if "job_id" in row.keys() else None,
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
    job_id: str | None = None,
) -> ImageRecord:
    created = datetime.now(timezone.utc)
    filename = f"{created:%Y%m%d-%H%M%S}-{seed}-{uuid.uuid4().hex[:6]}.png"
    path = settings.outputs_dir / filename
    # Written under a temporary name and renamed only once the row is committed.
    # A 2048x2048 PNG takes ~240ms to encode and `_write_lock` means nothing to
    # `clear.py`, which is a separate process: for that quarter second the file
    # would otherwise be an untracked *.png for its orphan sweep to delete -
    # taking a good image with it, or failing on the open handle with WinError
    # 32 after the row DELETE had already committed. A *.png.part matches
    # neither that glob nor the gallery's.
    partial = path.with_name(path.name + ".part")

    meta: dict[str, Any] = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "steps": request.steps,
        "guidance_scale": request.guidance_scale,
        "mu": request.mu,
        "seed": seed,
        "model": model_id,
        "job": job_id,
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
    image.save(partial, format="PNG", pnginfo=info, compress_level=4)

    try:
        with _write_lock, _session() as conn:
            cur = conn.execute(
                """INSERT INTO images (filename, prompt, negative_prompt, width, height,
                                       steps, guidance_scale, mu, seed, model_id,
                                       duration_ms, created_at, job_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    job_id,
                ),
            )
            row = conn.execute("SELECT * FROM images WHERE id = ?", (cur.lastrowid,)).fetchone()
        partial.replace(path)
    except BaseException:
        # A failed INSERT must not leave a *.png.part behind for nobody - and
        # tidying up must not replace the error that got us here.
        _unlink(partial)
        raise
    return _row_to_record(row)


def step_cost_ms(width: int, height: int, sample: int = 20) -> float | None:
    """Median milliseconds per denoising step at this size, from recent images.

    Per *step*, not per image: a grid varies the step count across its cells, so
    an average image duration would over- or under-state most of them. Matched
    on exact dimensions - cost scales with pixels and offload behaviour, neither
    of which extrapolates honestly from a different size. None when the gallery
    has nothing to measure, which is the caller's cue to promise nothing.
    """
    with _session() as conn:
        rows = conn.execute(
            """SELECT duration_ms, steps FROM images
               WHERE width = ? AND height = ? AND steps > 0 AND duration_ms > 0
               ORDER BY id DESC LIMIT ?""",
            (width, height, sample),
        ).fetchall()
    if not rows:
        return None
    costs = sorted(row["duration_ms"] / row["steps"] for row in rows)
    return costs[len(costs) // 2]


def _page(conn: sqlite3.Connection, clause: str, params: list, limit: int, offset: int):
    """Count and fetch one page over the *same* predicate.

    Both branches below go through here so the count badge, the grid and the
    paging cannot end up reading different result sets - which is how the LIKE
    fallback used to drop the favorites filter and then report a total for rows
    it never showed.
    """
    total = conn.execute(f"SELECT COUNT(*) FROM images {clause}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM images {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    ).fetchall()
    return total, rows


# `key:value` in the search box. The column, and what a bare value means.
_FILTERS: dict[str, tuple[str, Any]] = {
    "seed": ("seed", int),
    "steps": ("steps", int),
    "cfg": ("guidance_scale", float),
    "mu": ("mu", float),
    "model": ("model_id", str),
    "job": ("job_id", str),
}
_COMPARISONS = ("<=", ">=", "<", ">")
_NULL_TOKENS = {"none", "null", "off"}


def _filter_clause(key: str, raw: str) -> tuple[str, list]:
    """One `key:value` term as SQL. Raises ValueError on a value that is not one.

    Exact (`cfg:0.5`), comparison (`steps:>=8`), range (`cfg:0-1`) or `none`,
    which is the only way to ask for the images that ran without a mu at all -
    `mu:0` is a different image and SQL will not match NULL with either.
    """
    column, cast = _FILTERS[key]
    if cast is str:
        return rf"{column} LIKE ? ESCAPE '\'", [f"%{_like_escape(raw)}%"]
    if raw.lower() in _NULL_TOKENS:
        return f"{column} IS NULL", []
    for operator in _COMPARISONS:
        if raw.startswith(operator):
            return f"{column} {operator} ?", [_cast(cast, key, raw[len(operator):])]
    # A range borrows the grid's own spelling, so `cfg:0-1` reads the same in
    # the search box as it does in the axis that produced those images.
    low, dash, high = raw.partition("-")
    if dash and low and high:
        return f"{column} BETWEEN ? AND ?", [_cast(cast, key, low), _cast(cast, key, high)]
    return f"{column} = ?", [_cast(cast, key, raw)]


def _cast(cast, key: str, raw: str):
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{key}: {raw!r} is not a number") from None


def _split_search(search: str) -> tuple[list[str], list, str]:
    """Split the box into SQL predicates and the words left over for FTS.

    Anything that is not a known `key:value` stays free text, so a prompt that
    happens to contain a colon still searches for itself.
    """
    clauses: list[str] = []
    params: list = []
    words: list[str] = []
    for token in search.split():
        key, sep, raw = token.partition(":")
        if sep and raw and key.lower() in _FILTERS:
            clause, values = _filter_clause(key.lower(), raw)
            clauses.append(clause)
            params.extend(values)
        else:
            words.append(token)
    return clauses, params, " ".join(words)


def _like_escape(term: str) -> str:
    r"""Neutralise LIKE wildcards, for use with ESCAPE '\'."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_images(
    limit: int = 50, offset: int = 0, search: str | None = None, favorites_only: bool = False
) -> tuple[list[ImageRecord], int]:
    where: list[str] = []
    params: list = []
    if favorites_only:
        where.append("favorite = 1")
    text = ""
    if search:
        # `seed:42 cfg:<=0.5 fox` - the terms that name a column become real
        # predicates, and only what is left goes to the full-text index.
        filters, filter_params, text = _split_search(search)
        where.extend(filters)
        params.extend(filter_params)
    if text:
        # Match the FTS index, falling back to LIKE for punctuation-heavy input.
        # The search predicate is appended last; the fallback relies on that.
        where.append("id IN (SELECT rowid FROM images_fts WHERE images_fts MATCH ?)")
        params.append(" ".join(f'"{t}"' for t in text.split()))
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _session() as conn:
        try:
            total, rows = _page(conn, clause, params, limit, offset)
        except sqlite3.OperationalError:
            if not text:
                raise
            # A token carrying an odd number of double quotes - 26", a 6" figure
            # - quotes into a malformed MATCH. Swap *only* the search predicate
            # so favorites and anything added later still applies, and AND one
            # LIKE per token: FTS matches tokens in any order, where a single
            # LIKE over the raw string would demand them contiguous.
            terms = text.split()
            where[-1:] = [r"prompt LIKE ? ESCAPE '\'"] * len(terms)
            params[-1:] = [f"%{_like_escape(t)}%" for t in terms]
            clause = f"WHERE {' AND '.join(where)}" if where else ""
            total, rows = _page(conn, clause, params, limit, offset)
    return [_row_to_record(r) for r in rows], total


def completed_cells(
    prompts: Sequence[str], width: int, height: int, model_id: str
) -> set[CellKey]:
    """The (prompt, negative, steps, CFG, mu, seed) of everything already rendered.

    What a resumed grid subtracts from its cells. Keyed on the settings that
    make the image rather than on the job that made it, so a night that died
    can be resumed by a *new* job - and a cell rendered by hand earlier counts
    just the same. Rounded to the same six decimals the axis parser produces,
    so a float that survived a round trip through SQLite still matches.

    `prompts` is a list because a grid may vary the prompt: the caller passes
    the distinct prompts its cells resolved to. Chunked, since SQLite's default
    ceiling on host parameters is 999 and a grid may queue more cells than that.
    """
    wanted = list(dict.fromkeys(prompts))
    keys: set[CellKey] = set()
    with _session() as conn:
        for start in range(0, len(wanted), 400):
            chunk = wanted[start : start + 400]
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""SELECT prompt, negative_prompt, steps, guidance_scale, mu, seed FROM images
                    WHERE width = ? AND height = ? AND model_id = ?
                      AND prompt IN ({marks})""",
                (width, height, model_id, *chunk),
            ).fetchall()
            keys.update(
                (
                    row["prompt"],
                    # '' and NULL both mean "no negative prompt", and which one
                    # a row holds depends on which client wrote it.
                    row["negative_prompt"] or None,
                    row["steps"],
                    round(row["guidance_scale"], 6),
                    None if row["mu"] is None else round(row["mu"], 6),
                    row["seed"],
                )
                for row in rows
            )
    return keys


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
    # The row is already gone, so a file someone else has open must not turn a
    # delete that did happen into a 500. clear_history's orphan sweep collects
    # whatever gets left behind.
    _unlink(settings.outputs_dir / record.filename)
    return True


def _unlink(path) -> bool:
    """Delete a file, tolerating both "already gone" and "someone has it open"."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # WinError 32, most often: another process is streaming it
        return False
    return True


def summarise() -> dict[str, int]:
    """Counts for the clear-history preview.

    An absent database reports as empty rather than being created on the spot:
    `clear.py --dry-run` promises it changed nothing, and _connect() writes the
    whole schema into any file it opens.
    """
    total = favorites = 0
    tracked: set[str] = set()
    if settings.db_path.exists():
        with _session() as conn:
            total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            favorites = conn.execute(
                "SELECT COUNT(*) FROM images WHERE favorite = 1"
            ).fetchone()[0]
        tracked = {r["filename"] for r in _tracked_files()}
    files = list(settings.outputs_dir.glob("*.png"))

    total_bytes = 0
    for path in files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            # Deleted between the glob and the stat, or locked. A summary is not
            # worth raising over.
            continue

    on_disk = {p.name for p in files}
    return {
        "records": total,
        "favorites": favorites,
        "files_on_disk": len(on_disk),
        "orphan_files": len(on_disk - tracked),
        "bytes_on_disk": total_bytes,
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
        if path.exists() and _unlink(path):
            removed += 1

    # Anything left with no row behind it: earlier deletions, interrupted runs.
    # An image still being written is a *.png.part and matches neither glob, so
    # a clear during a sweep can no longer delete one out from under it.
    keep = {r["filename"] for r in _tracked_files()}
    orphans = 0
    for path in settings.outputs_dir.glob("*.png"):
        if path.name not in keep and _unlink(path):
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
