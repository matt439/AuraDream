"""Gallery persistence: SQLite index plus PNGs carrying their own metadata."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from PIL import Image, PngImagePlugin

from .config import settings
# The three key elements that are built from columns here and from a request
# in `grid`: one spelling of each, or a resume compares strings that only look
# alike. `grid` imports neither this module nor anything that does. VARIANT is
# the variable a comparison's versions are recorded under, which `keep_rates`
# groups by.
from .grid import VARIANT, lora_key, reference_key, refine_key
from .metadata import PNG_METADATA_KEY
from .refine import start_index
from .search import TRUE_CFG_SQL, like_escape, split_search, strict_scope
from .schemas import ImageRecord, RunSummary, effective_true_cfg

if TYPE_CHECKING:
    from .grid import CellKey
    from .schemas import GenerateRequest

log = logging.getLogger(__name__)

_write_lock = threading.Lock()

# The long side of the WEBP written beside every PNG. The gallery's cards are
# 160 px squares under 1800 px and 200-260 CSS px from there (--thumb-min's
# clamp: 245 at 2560, 217 at 2048, 271 device px at 125%), each showing the
# whole frame on its long side, so 384 covers a card at any shape. A sheet
# cell larger than this takes the PNG instead (sheet.js's srcset). A
# 2048x2048 PNG is 5 MB on disk and 17 MB decoded, and a page of 48 of them
# was 400 MB of bitmap to paint 48 squares.
THUMB_SIDE = 384

# A *.png.part older than this is not being written. Nothing streams a partial
# and an encode takes well under a second, so an hour is four orders of
# magnitude of slack; a younger one may be another process mid-encode, which is
# the case the `.part` naming exists to protect.
PARTIAL_SETTLE_SECONDS = 3600


def _thumb_path(path):
    return path.with_suffix(".webp")


def _write_thumb(image: Image.Image, path) -> None:
    """A small WEBP beside the PNG, for the cards and the sheet.

    Written *after* the PNG's rename, so a failed INSERT leaves no thumbnail
    for a row that never existed, and through the same `.part`-then-rename
    step so the orphan sweep never sees a half-written one. A thumbnail that
    fails costs bandwidth, not the image: the record falls back to the PNG.
    """
    partial = path.with_name(path.name + ".part")
    try:
        small = image.copy()
        small.thumbnail((THUMB_SIDE, THUMB_SIDE))
        small.save(partial, format="WEBP", quality=82, method=4)
        partial.replace(path)
    except Exception:  # noqa: BLE001 - the PNG and its row are already safe
        log.warning("thumbnail for %s not written", path.name, exc_info=True)
        _unlink(partial)

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
    job_id          TEXT,
    run_id          TEXT,
    quantization    TEXT,
    variables       TEXT,
    true_cfg        REAL,
    content_hash    TEXT,
    reference       TEXT,
    lora            TEXT,
    lora_scale      REAL,
    refine          INTEGER,
    strength        REAL
);
CREATE INDEX IF NOT EXISTS idx_images_created ON images (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_images_job ON images (job_id);
CREATE INDEX IF NOT EXISTS idx_images_run ON images (run_id);
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts
    USING fts5(prompt, content='images', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, prompt) VALUES (new.id, new.prompt);
END;
CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, prompt) VALUES('delete', old.id, old.prompt);
END;
"""


# Which database file the running process has checked; the ALTER is a schema
# change, not something to attempt on every connection. Keyed on the path, not
# a bare flag: `_connect` re-creates the schema per connection precisely because
# the file can be replaced under a running server - restored from a backup, say
# - and a restored older file needs its columns again.
_migrated: str | None = None


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that a database written by an older build has not got.

    `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it was, so
    a gallery from before this column simply never gains it - and every INSERT
    then fails. Cheap enough to check once per process; the ALTER itself runs
    at most once, and races with another process end in "duplicate column",
    which is the outcome we wanted anyway.
    """
    global _migrated
    if _migrated == str(settings.db_path):
        return
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
    added = (
        ("job_id", "TEXT", "idx_images_job"),
        ("run_id", "TEXT", "idx_images_run"),
        ("quantization", "TEXT", None),
        # The prompt-variable assignment, as JSON.
        ("variables", "TEXT", None),
        # The scale the negative prompt ran at; NULL on older rows means
        # "guidance_scale, if there was a negative prompt" - see
        # `schemas.effective_true_cfg`.
        ("true_cfg", "REAL", None),
        # sha256 of the pixels, 16 hex digits.
        ("content_hash", "TEXT", None),
        # The gallery ids this image was conditioned on, as JSON.
        ("reference", "TEXT", None),
        # The adapter that was applied and the weight it ran at.
        ("lora", "TEXT", None),
        ("lora_scale", "REAL", None),
        # The gallery id a refine pass started from and its denoising
        # strength; NULL together on a text-to-image row.
        ("refine", "INTEGER", None),
        ("strength", "REAL", None),
    )
    for column, kind, index in added:
        if column in columns:
            continue
        try:
            conn.execute(f"ALTER TABLE images ADD COLUMN {column} {kind}")
            if index:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {index} ON images ({column})")
        except sqlite3.OperationalError:  # another process got there first
            pass
    _migrated = str(settings.db_path)


def ensure_outputs_dir() -> None:
    """Create the images directory. Called before anything writes into it.

    Here rather than at import of `config`: `clear.py --dry-run` promises to
    change nothing, and an import-time mkdir created the directory anyway - and
    on an unmounted `LOCAL_DIFFUSION_OUTPUTS_DIR` created the mount point as a plain
    directory, the state that later hides the real data behind an empty mount.
    """
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    # A restored or replaced file has a fresh identity; forget the old check.
    global _migrated
    if _migrated is not None and _migrated != str(settings.db_path):
        _migrated = None
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
    local-diffusion.db then fails while the server runs.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    ensure_outputs_dir()
    with _session() as conn:
        conn.executescript(_SCHEMA)


def _row_to_record(row: sqlite3.Row, refine_missing: bool = False) -> ImageRecord:
    """A gallery row as the record the API serves.

    Every column is read unguarded. Rows only ever come out of a connection
    `_connect` opened, and `_connect` runs `_migrate` first, so a database from
    an older build has every column by the time a row is fetched - NULL where
    the build that wrote it knew nothing of the field. A new column therefore
    needs an entry in `_migrate`, not a read guard here: a guard would let the
    SELECT succeed and every INSERT after it fail on the missing column.
    `refine_missing` is no column: the reader that has looked the source up
    (`_missing_sources`) says so.
    """
    # Rows from before thumbnails were written have none; the client falls back
    # to the PNG. One stat per row, cheap next to the query.
    thumb = _thumb_path(settings.outputs_dir / row["filename"])
    return ImageRecord(
        id=row["id"],
        filename=row["filename"],
        url=f"/outputs/{row['filename']}",
        thumb_url=f"/outputs/{thumb.name}" if thumb.exists() else None,
        prompt=row["prompt"],
        negative_prompt=row["negative_prompt"],
        variables=json.loads(row["variables"]) if row["variables"] else None,
        width=row["width"],
        height=row["height"],
        steps=row["steps"],
        guidance_scale=row["guidance_scale"],
        true_cfg=effective_true_cfg(row["true_cfg"], row["negative_prompt"], row["guidance_scale"]),
        mu=row["mu"],
        seed=row["seed"],
        model_id=row["model_id"],
        quantization=row["quantization"],
        reference=json.loads(row["reference"]) if row["reference"] else None,
        lora=row["lora"],
        lora_scale=row["lora_scale"],
        refine=row["refine"],
        refine_missing=refine_missing,
        strength=row["strength"],
        job_id=row["job_id"],
        run_id=row["run_id"],
        duration_ms=row["duration_ms"],
        created_at=row["created_at"],
        favorite=bool(row["favorite"]),
        content_hash=row["content_hash"],
    )


# Every column a writer supplies, in one place: `save_image` and the reindex
# both insert through `_insert_row`, so the list cannot drift between them.
_COLUMNS = (
    "filename", "prompt", "negative_prompt", "width", "height", "steps", "guidance_scale",
    "mu", "seed", "model_id", "duration_ms", "favorite", "created_at", "job_id", "run_id",
    "quantization", "variables", "true_cfg", "content_hash", "reference", "lora", "lora_scale",
    "refine", "strength",
)


def content_hash(image: Image.Image) -> str:
    """The first 16 hex digits of the sha256 of the decoded pixels.

    Of the pixels, not the file: PNG bytes vary with the encoder's settings
    and the metadata chunks, and two identical renders would then hash apart.
    A few milliseconds against a generation measured in seconds. What it is
    for is exactness: "the mu axis changed nothing across 96 cells" becomes a
    count rather than a squint, and a `hash:` search finds every image that
    is the same picture. It rests on the pipeline being deterministic through
    the quantized kernels - verified once on the card (see the review
    checklist) rather than assumed.
    """
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def _insert_row(conn: sqlite3.Connection, values: dict[str, Any]) -> sqlite3.Row:
    """One row in, the FTS trigger firing with it, the row read back."""
    columns = ", ".join(_COLUMNS)
    marks = ", ".join("?" * len(_COLUMNS))
    cur = conn.execute(
        f"INSERT INTO images ({columns}) VALUES ({marks})", tuple(values[c] for c in _COLUMNS)
    )
    return conn.execute("SELECT * FROM images WHERE id = ?", (cur.lastrowid,)).fetchone()


def human_bytes(n: int | float) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def image_provenance(image_id: int | None) -> dict[str, Any] | None:
    """Portable identity for a gallery-local conditioning id in a PNG.

    Dimensions accompany the decoded-pixel hash because the hash itself does
    not encode the shape. A filename locates the source during PNG recovery;
    the hash and size verify it before any destination id can be assigned.
    """
    record = get_image(image_id) if image_id is not None else None
    if record is None or not record.content_hash:
        return None
    return {
        "filename": record.filename,
        "hash": record.content_hash,
        "width": record.width,
        "height": record.height,
    }


def save_image(
    image: Image.Image,
    request: "GenerateRequest",
    seed: int,
    duration_ms: int,
    model_id: str,
    job_id: str | None = None,
    run_id: str | None = None,
    quantization: str | None = None,
    variables: dict[str, str] | None = None,
) -> ImageRecord:
    """Write the PNG and its row. `request` must be what actually ran -
    `backend.applied()` - not what was asked for: a negative prompt the
    pipeline never saw, recorded here, defeats resume and misleads Reuse.
    `variables` is the prompt-variable assignment behind a grid cell, kept so
    the sheet can show what varied rather than reconstruct it from prose.
    """
    created = datetime.now(timezone.utc)
    variables_json = json.dumps(variables) if variables else None
    digest = content_hash(image)
    filename = f"{created:%Y%m%d-%H%M%S}-{seed}-{uuid.uuid4().hex[:6]}.png"
    # The directory can disappear under a running server - renamed, deleted by
    # hand - and the full denoise then died here. One mkdir per image is cheap.
    ensure_outputs_dir()
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
        "true_cfg": request.true_cfg,
        "mu": request.mu,
        "seed": seed,
        "model": model_id,
        "quantization": quantization,
        "job": job_id,
        "run": run_id,
        "size": f"{image.width}x{image.height}",
        "variables": variables or None,
        "hash": digest,
        # None rather than [] and null rather than 1.0: a text-to-image row and
        # a row with no adapter say so by absence, which is what `applied()`
        # has already normalised them to.
        "reference": list(request.reference) or None,
        "reference_sources": [image_provenance(i) for i in request.reference] or None,
        "lora": request.lora,
        "lora_scale": request.lora_scale if request.lora else None,
        # Likewise null together: the image this one refines and how far it
        # was re-denoised, or a text-to-image render.
        "refine": request.refine,
        "refine_source": image_provenance(request.refine),
        "strength": request.strength if request.refine is not None else None,
    }
    info = PngImagePlugin.PngInfo()
    # A1111-style single-line block, so other tools can read the params back.
    # Mu is not an A1111 key, but it is part of what reproduces the image, which
    # is the reason the block exists. Nor is True CFG: on FLUX.1 the CFG
    # scale is the distilled guidance value, and the real one is this.
    # `Denoising strength` is A1111's own key for an image-to-image pass;
    # the source it started from is ours.
    info.add_text(
        "parameters",
        f"{request.prompt}\n"
        f"Negative prompt: {request.negative_prompt or ''}\n"
        f"Steps: {request.steps}, CFG scale: {request.guidance_scale}, "
        f"Seed: {seed}, Size: {image.width}x{image.height}, Model: {model_id}"
        + (f", True CFG: {request.true_cfg}" if request.true_cfg is not None else "")
        + (f", Mu: {request.mu}" if request.mu is not None else "")
        + (f", Quantization: {quantization}" if quantization else "")
        + (
            f", LoRA: {request.lora}:{request.lora_scale:g}"
            if request.lora
            else ""
        )
        + (
            f", Reference: {','.join(str(i) for i in request.reference)}"
            if request.reference
            else ""
        )
        + (
            f", Denoising strength: {request.strength:g}, Refine: {request.refine}"
            if request.refine is not None
            else ""
        ),
    )
    info.add_text(PNG_METADATA_KEY, json.dumps(meta))

    committed = False
    try:
        # Inside the guard: a KeyboardInterrupt mid-encode - the CLI's second
        # Ctrl-C - is a BaseException Pillow's own cleanup does not catch, and
        # it left a *.png.part that nothing swept.
        image.save(partial, format="PNG", pnginfo=info, compress_level=4)
        with _write_lock, _session() as conn:
            row = _insert_row(
                conn,
                {
                    "filename": filename,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": image.width,
                    "height": image.height,
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                    "mu": request.mu,
                    "seed": seed,
                    "model_id": model_id,
                    "duration_ms": duration_ms,
                    "favorite": 0,
                    "created_at": created.isoformat(),
                    "job_id": job_id,
                    "run_id": run_id,
                    "quantization": quantization,
                    "variables": variables_json,
                    "true_cfg": request.true_cfg,
                    "content_hash": digest,
                    "reference": json.dumps(list(request.reference)) if request.reference else None,
                    "lora": request.lora,
                    "lora_scale": request.lora_scale if request.lora else None,
                    "refine": request.refine,
                    "strength": request.strength if request.refine is not None else None,
                },
            )
            # A pass whose source was deleted while it ran - read as the
            # cell began, gone by its end - says so as list_images would,
            # under the lock a delete takes: returned as present, the page
            # showed it as one that could be made again until a reload.
            missing = _missing_sources(conn, [row])
        committed = True
        _rename_with_retry(partial, path)
    except BaseException:
        # A failed encode or INSERT must not leave a *.png.part behind for
        # nobody - and tidying up must not replace the error that got us here.
        # Only while the row did *not* commit, though: past that point the
        # partial is the only copy of an image the database already names, and
        # deleting it turned a transient rename failure into a phantom row.
        if not committed:
            _unlink(partial)
        raise
    _write_thumb(image, _thumb_path(path))
    return _row_to_record(row, row["refine"] in missing)


def _rename_with_retry(src, dst, attempts: int = 5) -> None:
    """`src.replace(dst)`, retried over a few hundred milliseconds.

    On Windows a just-closed multi-megabyte file is exactly what an antivirus
    scanner, a search indexer or a sync client opens without FILE_SHARE_DELETE,
    and the rename fails with a sharing violation (WinError 32) for the
    milliseconds the handle is held.
    """
    for attempt in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _stale_partials() -> list:
    """Partials nothing can still be writing - see PARTIAL_SETTLE_SECONDS."""
    cutoff = time.time() - PARTIAL_SETTLE_SECONDS
    stale = []
    for path in settings.outputs_dir.glob("*.part"):
        try:
            if path.stat().st_mtime < cutoff:
                stale.append(path)
        except OSError:
            continue
    return stale


def step_cost_ms(
    width: int, height: int, model_id: str | None = None, sample: int = 20
) -> float | None:
    """Median direct-render milliseconds per step at this size and checkpoint.

    Per *step*, not per image: a grid varies the step count across its cells, so
    an average image duration would over- or under-state most of them. Matched
    on exact dimensions - cost scales with pixels and offload behaviour, neither
    of which extrapolates honestly from a different size - and on the checkpoint,
    since klein's step is about half of Turbo's at the same size. None when the
    gallery has nothing to measure, which is the caller's cue to promise nothing.

    Refine rows are kept out: their fixed encode/decode costs dominate at low
    strengths and cannot price a direct render. See `refine_cost_ms`.
    """
    scope = "AND model_id = ?" if model_id else ""
    scope_params = (model_id,) if model_id else ()
    with _session() as conn:
        rows = conn.execute(
            f"""SELECT duration_ms, steps FROM images
               WHERE width = ? AND height = ? AND steps > 0 AND duration_ms > 0
                 AND refine IS NULL {scope}
               ORDER BY id DESC LIMIT ?""",
            (width, height, *scope_params, sample),
        ).fetchall()
    if not rows:
        return None
    costs = sorted(row["duration_ms"] / row["steps"] for row in rows)
    return costs[len(costs) // 2]


def refine_cost_ms(
    width: int, height: int, model_id: str, sample: int = 100,
) -> dict[int, float]:
    """Median full refine duration per image, grouped by denoised step count.

    Same size and checkpoint only, like the direct estimate. Matching the
    count keeps the fixed resize, text and VAE work in every image's price;
    one strength's timing is never extrapolated as a per-step rate. Unknown
    families provide no estimate until their schedule rounding is known.
    """
    with _session() as conn:
        rows = conn.execute(
            """SELECT duration_ms, steps, strength FROM images
               WHERE width = ? AND height = ? AND model_id = ? AND steps > 0
                 AND duration_ms > 0 AND refine IS NOT NULL AND strength IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (width, height, model_id, sample),
        ).fetchall()
    durations: dict[int, list[float]] = {}
    for row in rows:
        start = start_index(row["steps"], row["strength"], model_id)
        if start is not None:
            durations.setdefault(row["steps"] - start, []).append(row["duration_ms"])
    return {steps: sorted(values)[len(values) // 2] for steps, values in durations.items()}


def _page(
    conn: sqlite3.Connection,
    clause: str,
    params: list,
    limit: int,
    offset: int,
    before: int | None = None,
):
    """Count and fetch one page over the *same* predicate.

    Both branches below go through here so the count badge, the grid and the
    paging cannot end up reading different result sets - which is how the LIKE
    fallback used to drop the favorites filter and then report a total for rows
    it never showed.

    `before` is keyset paging: rows with an id below it, newest first. OFFSET
    over a table being inserted into is wrong in both directions - a row
    landing at the top shifts every fetched row down one, so the tail of the
    last page comes back as the head of the next and as many older rows are
    skipped for good. An id is stable under inserts and deletes alike. The
    count is over the whole predicate either way: it is the badge's number,
    not the page's.
    """
    total = conn.execute(f"SELECT COUNT(*) FROM images {clause}", params).fetchone()[0]
    page_clause, page_params = clause, list(params)
    if before is not None:
        page_clause = f"{clause} AND id < ?" if clause else "WHERE id < ?"
        page_params.append(before)
    rows = conn.execute(
        f"SELECT * FROM images {page_clause} ORDER BY id DESC LIMIT ? OFFSET ?",
        (*page_params, limit, offset),
    ).fetchall()
    return total, rows


def list_images(
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
    favorites_only: bool = False,
    before: int | None = None,
    model_id: str | None = None,
) -> tuple[list[ImageRecord], int, int]:
    """One page of the gallery, the total the filters match, and what a
    checkpoint scope left out.

    `before` pages by id rather than by offset - see `_page`. The two compose,
    but a client wants one or the other: the browser carries the last id it
    was shown, and `offset` stays for anyone scripting against the API.

    `model_id` narrows the page to one checkpoint's images, exactly - the
    search box's `model:` is a substring, and `model:flux` is both FLUX.1
    checkpoints - and the third number is how many rows the *other* filters
    match on other checkpoints: the badge for the switch that shows them,
    and the difference between "this run made nothing" and "this run was
    made on the other model". 0 when unscoped.
    """
    # The scope stays out of `where`: the fallback below swaps the *last*
    # predicate, and the count of what the scope hides needs the rest alone.
    scope = "model_id = ?" if model_id else ""
    where: list[str] = []
    params: list = []
    if favorites_only:
        where.append("favorite = 1")
    text = ""
    if search:
        # `seed:42 cfg:<=0.5 fox` - the terms that name a column become real
        # predicates, and only what is left goes to the full-text index.
        filters, filter_params, text = split_search(search)
        where.extend(filters)
        params.extend(filter_params)
    if text:
        # Match the FTS index, falling back to LIKE for punctuation-heavy input.
        # The search predicate is appended last; the fallback relies on that.
        where.append("id IN (SELECT rowid FROM images_fts WHERE images_fts MATCH ?)")
        params.append(" ".join(f'"{t}"' for t in text.split()))

    def clause_for(head: str) -> tuple[str, list]:
        parts = [head, *where] if head else where
        return (f"WHERE {' AND '.join(parts)}" if parts else ""), (
            [model_id, *params] if head else list(params)
        )

    with _session() as conn:
        try:
            total, rows = _page(conn, *clause_for(scope), limit, offset, before)
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
            params[-1:] = [f"%{like_escape(t)}%" for t in terms]
            total, rows = _page(conn, *clause_for(scope), limit, offset, before)
        others = 0
        if model_id:
            other_clause, other_params = clause_for("model_id <> ?")
            others = conn.execute(
                f"SELECT COUNT(*) FROM images {other_clause}", other_params
            ).fetchone()[0]
        missing = _missing_sources(conn, rows)
    return [_row_to_record(r, r["refine"] in missing) for r in rows], total, others


def _missing_sources(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> set[int]:
    """The refine sources these rows name that are no longer in the gallery.

    A pass keeps its source's id after the source is deleted - the id is the
    reproduction set's, and a reindex can remap it - so the record says the
    source has gone (`ImageRecord.refine_missing`) rather than every page
    naming an image it cannot show. Chunked like `completed_cells`: a sheet's
    page is a whole run.
    """
    wanted = list({row["refine"] for row in rows if row["refine"] is not None})
    present: set[int] = set()
    for start in range(0, len(wanted), 400):
        chunk = wanted[start : start + 400]
        marks = ",".join("?" * len(chunk))
        present.update(
            row[0] for row in conn.execute(f"SELECT id FROM images WHERE id IN ({marks})", chunk)
        )
    return set(wanted) - present


def completed_cells(
    prompts: Sequence[str],
    width: int,
    height: int,
    model_id: str,
    quantization: str | None = None,
) -> set[CellKey]:
    """The key of everything already rendered: the prompt pair, the settings,
    the references, adapter and refine source, the seed.

    What a resumed grid subtracts from its cells. Keyed on the settings that
    make the image rather than on the job that made it, so a night that died
    can be resumed by a *new* job - and a cell rendered by hand earlier counts
    just the same. Rounded to the same six decimals the axis parser produces,
    so a float that survived a round trip through SQLite still matches. The
    true-CFG element is read through `effective_true_cfg`, so a row from
    before the column comes back under the key it would be written with now.

    Scoped to the size, the checkpoint and - when given - the quantizer: the
    same seed through fp8 and through int8 is two different pictures, and a
    resume under the other one must not count the first as done. Rows that
    predate the column match any quantizer, so an older gallery keeps resuming.

    `prompts` is a list because a grid may vary the prompt: the caller passes
    the distinct prompts its cells resolved to. Chunked, since SQLite's default
    ceiling on host parameters is 999 and a grid may queue more cells than that.
    """
    wanted = list(dict.fromkeys(prompts))
    keys: set[CellKey] = set()
    scope = "AND (quantization IS NULL OR quantization = ?)" if quantization else ""
    scope_params = (quantization,) if quantization else ()
    with _session() as conn:
        for start in range(0, len(wanted), 400):
            chunk = wanted[start : start + 400]
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""SELECT prompt, negative_prompt, steps, guidance_scale, mu, true_cfg,
                           reference, lora, lora_scale, refine, strength, seed
                    FROM images
                    WHERE width = ? AND height = ? AND model_id = ? {scope}
                      AND prompt IN ({marks})""",
                (width, height, model_id, *scope_params, *chunk),
            ).fetchall()
            for row in rows:
                true_cfg = effective_true_cfg(
                    row["true_cfg"], row["negative_prompt"], row["guidance_scale"]
                )
                keys.add(
                    (
                        row["prompt"],
                        # '' and NULL both mean "no negative prompt", and which
                        # one a row holds depends on which client wrote it.
                        row["negative_prompt"] or None,
                        row["steps"],
                        round(row["guidance_scale"], 6),
                        None if row["mu"] is None else round(row["mu"], 6),
                        None if true_cfg is None else round(true_cfg, 6),
                        reference_key(json.loads(row["reference"]) if row["reference"] else None),
                        lora_key(row["lora"], row["lora_scale"]),
                        refine_key(row["refine"], row["strength"]),
                        row["seed"],
                    )
                )
    return keys


# The images a starred one was made from: a refine pass's source and a klein
# render's references, by id. They are part of the starred image's
# reproduction set, and a sweep that took them left a kept pass that could
# never be made again - with nothing before the click saying it depended on
# them. One level: the source of a kept source is not the starred image's to
# need. Neither arm can yield a NULL, which would turn every `NOT id IN` over
# the set into NULL and keep - or, counted, miscount - every row.
_STARRED_SOURCE = (
    "id IN (SELECT refine FROM images WHERE favorite = 1 AND refine IS NOT NULL"
    " UNION SELECT ref.value FROM images AS starred, json_each(starred.reference) AS ref"
    " WHERE starred.favorite = 1 AND starred.reference IS NOT NULL AND ref.type = 'integer')"
)
# What `keep_favorites` leaves out of a delete: the stars and their sources.
# The one predicate `delete_matching`, `clear_history` and the previews' counts
# share, so the number a dialog asks to have typed back is the number the
# delete then finds under its lock.
_UNKEPT = f"favorite = 0 AND NOT ({_STARRED_SOURCE})"


def matching_preview(
    search: str | None, model_id: str | None = None, newer_than: int | None = None
) -> dict[str, Any]:
    """What a strict scope selects, named: counts, prompts, and the jobs and
    checkpoints a substring term (`job:`, `model:`) actually matched.

    An empty scope is the whole gallery - the browser's Delete unstarred asks
    with the search box blank - so the "a filter is required" refusal is the
    CLI's, at its own prompt. `model_id` is the gallery's checkpoint scope,
    exact, and the preview has to take it because the delete does: a count
    over images the page is not showing is the wrong number to confirm.

    `sources_kept` is what `keep_favorites` keeps besides the stars - the
    unstarred images in scope that a starred image anywhere was made from -
    and `sources` the first few of their ids, so `records - favorites -
    sources_kept` is the count such a delete takes. `runs`, `first` and
    `last` describe that delete too: how many runs it reaches into and the
    span of their creation times, which three sample prompts never say.

    `newer` is how many of the images that delete takes have an id above
    `newer_than` - the newest the asking page held - or None where it was
    not asked: the page's own count of them could not leave out a kept
    source past the few `sources` names.
    """
    filters, params = strict_scope(search)
    if model_id:
        filters.append("model_id = ?")
        params.append(model_id)

    def where(*extra: str) -> str:
        parts = [*filters, *extra]
        return f"WHERE {' AND '.join(parts)}" if parts else ""

    clause = where()
    with _session() as conn:
        records, favorites, sources_kept = conn.execute(
            f"""SELECT COUNT(*), COALESCE(SUM(favorite), 0),
                       COALESCE(SUM(favorite = 0 AND {_STARRED_SOURCE}), 0)
                FROM images {clause}""",
            params,
        ).fetchone()
        sources = [
            row[0]
            for row in conn.execute(
                f"SELECT id FROM images {where('favorite = 0', _STARRED_SOURCE)} ORDER BY id LIMIT 6",
                params,
            )
        ]
        runs, first, last = conn.execute(
            f"SELECT COUNT(DISTINCT run_id), MIN(created_at), MAX(created_at) FROM images {where(_UNKEPT)}",
            params,
        ).fetchone()
        newer = None
        if newer_than is not None:
            newer = conn.execute(
                f"SELECT COUNT(*) FROM images {where(_UNKEPT, 'id > ?')}", (*params, newer_than)
            ).fetchone()[0]
        distinct = lambda column, limit: [  # noqa: E731
            row[0]
            for row in conn.execute(
                f"SELECT DISTINCT {column} FROM images {clause} ORDER BY {column} LIMIT ?",
                (*params, limit),
            )
        ]
        return {
            "records": records,
            "favorites": favorites,
            "sources_kept": sources_kept,
            "sources": sources,
            "runs": runs,
            "first": first,
            "last": last,
            "newer": newer,
            "prompts": distinct("prompt", 4),
            "jobs": [j for j in distinct("job_id", 6) if j],
            "models": distinct("model_id", 6),
        }


class SelectionChanged(Exception):
    """A scoped delete was confirmed for one count and finds another."""

    def __init__(self, found: int, expected: int):
        super().__init__(f"{found} images match now, not the {expected} confirmed")
        self.found = found
        self.expected = expected


def delete_matching(
    search: str | None,
    keep_favorites: bool = False,
    expected: int | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Delete the rows a strict scope selects, and their files - nothing else.

    `DELETE ... RETURNING` in one statement, so a row committed while the
    user was reading the confirmation prompt is not deleted from disk on the
    strength of an id list resolved a moment earlier. `expected` is the count
    that prompt showed: compared under the write lock, so a sweep that landed
    more unstarred images while the dialog was open raises `SelectionChanged`
    and deletes nothing rather than taking them on a number confirmed for
    fewer. `keep_favorites` is a scope of its own - an otherwise empty one
    means every unstarred image - where a delete with no scope at all is
    refused; it keeps the images a starred one was made from as well as the
    stars (`_UNKEPT`). No orphan sweep: a filtered clear that also removed every
    untracked PNG would be a new all-or-nothing hazard inside the feature
    meant to remove one. `model_id` is the gallery's checkpoint scope, the
    same exact predicate `matching_preview` counted under.
    """
    filters, params = strict_scope(search)
    if keep_favorites:
        filters.append(_UNKEPT)
    if model_id:
        filters.append("model_id = ?")
        params.append(model_id)
    if not filters:
        raise ValueError("a filter is required; the whole gallery is `clear` without --where")
    where = " AND ".join(filters)
    with _write_lock, _session() as conn:
        if expected is not None:
            (found,) = conn.execute(f"SELECT COUNT(*) FROM images WHERE {where}", params).fetchone()
            if found != expected:
                raise SelectionChanged(found, expected)
        doomed = conn.execute(
            f"DELETE FROM images WHERE {where} RETURNING id, filename", params
        ).fetchall()
    removed = 0
    for row in doomed:
        path = settings.outputs_dir / row["filename"]
        if path.exists() and _unlink(path):
            removed += 1
        _unlink(_thumb_path(path))
    return {"records": len(doomed), "files": removed, "ids": [row["id"] for row in doomed]}


def image_bytes(width: int, height: int, model_id: str | None = None, sample: int = 20) -> int | None:
    """Median PNG size on disk at this size on this checkpoint, or None.

    The third of the preview's three estimates, and the weakest: PNG size
    varies with content far more than duration does. Same rule as the
    others - no files, no estimate - and the same scoping, since a klein
    render and a Turbo render of one size are not the same picture.
    """
    scope = "AND model_id = ?" if model_id else ""
    scope_params = (model_id,) if model_id else ()
    with _session() as conn:
        names = [
            row["filename"]
            for row in conn.execute(
                f"""SELECT filename FROM images WHERE width = ? AND height = ? {scope}
                    ORDER BY id DESC LIMIT ?""",
                (width, height, *scope_params, sample),
            )
        ]
    sizes = []
    for name in names:
        try:
            sizes.append((settings.outputs_dir / name).stat().st_size)
        except OSError:
            continue
    if not sizes:
        return None
    sizes.sort()
    return sizes[len(sizes) // 2]


def existing_filenames() -> set[str]:
    """Every filename the index knows - what a reindex skips."""
    with _session() as conn:
        return {row["filename"] for row in conn.execute("SELECT filename FROM images")}


def existing_file_ids() -> dict[str, int]:
    """Every indexed filename and its gallery id, for an id-remapping import."""
    with _session() as conn:
        return {row["filename"]: row["id"] for row in conn.execute("SELECT id, filename FROM images")}


def import_file(values: dict[str, Any]) -> ImageRecord:
    """Insert one row copied from another gallery database.

    The caller has already copied the named PNG into ``outputs_dir`` and
    remapped any gallery-local reference ids.  Unlike ``index_file``, this
    path preserves the fields that are not embedded in the PNG: favorite and
    generation duration.
    """
    with _write_lock, _session() as conn:
        row = _insert_row(conn, {column: values.get(column) for column in _COLUMNS})
    return _row_to_record(row)


def index_file(
    filename: str,
    meta: dict[str, Any],
    width: int,
    height: int,
    created_at: datetime,
    digest: str | None = None,
) -> ImageRecord:
    """A row for a PNG already in `outputs_dir`, from its own metadata block.

    The reindex path: everything an `ImageRecord` needs is in the file except
    `favorite`, which lives in SQLite alone and is lost, and `duration_ms`,
    which is recorded as 0 so `step_cost_ms` (which wants `> 0`) does not price
    a grid off an unknown. Through the same INSERT as `save_image`, so the FTS
    trigger indexes the prompt. `digest` is the hash of the pixels as the
    caller decoded them, which is the truth; the block's own `hash` stands in
    for it when the caller did not decode.
    """
    with _write_lock, _session() as conn:
        row = _insert_row(
            conn,
            {
                "filename": filename,
                "prompt": meta["prompt"],
                "negative_prompt": meta.get("negative_prompt"),
                "width": width,
                "height": height,
                "steps": int(meta["steps"]),
                "guidance_scale": float(meta["guidance_scale"]),
                "mu": meta.get("mu"),
                "seed": int(meta["seed"]),
                "model_id": meta["model"],
                "duration_ms": 0,
                "favorite": 0,
                "created_at": created_at.isoformat(),
                "job_id": meta.get("job"),
                "run_id": meta.get("run"),
                "quantization": meta.get("quantization"),
                "variables": json.dumps(meta["variables"]) if meta.get("variables") else None,
                "true_cfg": meta.get("true_cfg"),
                "content_hash": digest or meta.get("hash"),
                "reference": json.dumps(meta["reference"]) if meta.get("reference") else None,
                "lora": meta.get("lora"),
                "lora_scale": meta.get("lora_scale") if meta.get("lora") else None,
                "refine": meta.get("refine"),
                "strength": meta.get("strength") if meta.get("refine") is not None else None,
            },
        )
    return _row_to_record(row)


def thumbnail_for(image: Image.Image, filename: str) -> None:
    """Write the WEBP beside an existing PNG, if it has none."""
    path = _thumb_path(settings.outputs_dir / filename)
    if not path.exists():
        _write_thumb(image, path)


# The columns `keep_rates` can group by, as SQL. CFG and mu are rounded to the
# six decimals the axis parser produces, so a value that went through REAL and
# back does not split one setting into two buckets.
_GROUPS: dict[str, str] = {
    "steps": "steps",
    "cfg": "round(guidance_scale, 6)",
    "tcfg": f"round({TRUE_CFG_SQL}, 6)",
    "mu": "round(mu, 6)",
    "prompt": "prompt",
    "negative": "negative_prompt",
    "seed": "seed",
    "size": "width || 'x' || height",
    "model": "model_id",
    "quant": "quantization",
    # The adapter and the weight it ran at: "which settings produced my
    # favourites" is exactly the question a LoRA weight is chosen by.
    "lora": "lora",
    "lorascale": "round(lora_scale, 6)",
    # A refine pass's strength, and the image it started from: a strength
    # sweep is starred by how much of the source it kept.
    "strength": "round(strength, 6)",
    "refine": "refine",
    # A comparison's version by its label (`prompt_variants`): the starred
    # rate per version. `prompt` cannot say it - reorders share their opening
    # words, which is all a table cell has room for.
    VARIANT: f"json_extract(variables, '$.{VARIANT}')",
}


def keep_rates(
    search: str | None = None, by: Sequence[str] = ("steps", "cfg", "mu"), min_made: int = 1
) -> list[dict[str, Any]]:
    """Which settings the starred images cluster under: made, starred, and the rate.

    The naive count rewards whichever setting was used most; the rate is what
    is wanted, and every input is in the table. `search` is the gallery's own
    `key:value` grammar and scopes the rows - a run, where `expand()` crossed
    the axes against the same prompts and seeds, is the honest scope; all-time
    is confounded by whatever settings were in use while the prompts happened
    to be good. Groups with fewer than `min_made` images are dropped so a
    lucky 1-of-1 does not take first place. The column is *starred*, not
    kept: nothing records a rejection, so an untouched night reads 0% rather
    than unknown. Size is always in the grouping - a 4 MP re-render is not
    the same setting as its 1 MP source. Rows rank by rate, then by count;
    a tie keeps the order the settings were first made in, so an untouched
    comparison reads its versions as the grid wrote them.
    """
    unknown = [name for name in by if name not in _GROUPS]
    if unknown:
        raise ValueError(f"cannot group by {', '.join(unknown)}; one of {', '.join(_GROUPS)}")
    columns = list(dict.fromkeys([*by, "size"]))
    where, params = strict_scope(search)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    select = ", ".join(f"{_GROUPS[name]} AS {name}" for name in columns)
    group = ", ".join(_GROUPS[name] for name in columns)
    with _session() as conn:
        rows = conn.execute(
            f"""SELECT {select}, COUNT(*) AS made, SUM(favorite) AS starred
                FROM images {clause}
                GROUP BY {group}
                HAVING made >= ?
                ORDER BY CAST(starred AS REAL) / made DESC, made DESC, MIN(id)""",
            (*params, min_made),
        ).fetchall()
        if VARIANT in columns and "prompt" not in columns and rows:
            # A run can join comparisons with swapped original/yours labels.
            # Check complete assignments within each emitted stats group:
            # variant x colour is an ordinary grid and may aggregate colours,
            # and grouping by steps/seed can already separate the versions.
            emitted = {tuple(row[name] for name in columns) for row in rows}
            prompts = {}
            for row in conn.execute(
                f"SELECT {select}, variables, prompt FROM images {clause} "
                f"GROUP BY {group}, variables, prompt", params
            ):
                grouped = tuple(row[name] for name in columns)
                if grouped not in emitted or row[VARIANT] is None:
                    continue
                assignment = json.dumps(json.loads(row["variables"]), sort_keys=True)
                key = (grouped, assignment)
                if key in prompts and prompts[key] != row["prompt"]:
                    raise ValueError(
                        "a variant label refers to more than one prompt for the same "
                        "variable assignment in this scope; use --by variant,prompt "
                        "or narrow the scope with --job"
                    )
                prompts[key] = row["prompt"]
    return [
        {**{name: row[name] for name in columns}, "made": row["made"], "starred": row["starred"],
         "rate": row["starred"] / row["made"]}
        for row in rows
    ]


# The settings a run can vary, as the Runs picker names them: each one's
# count of distinct values over the run (a NULL counted as a value of its own
# where the column takes one), read back as RunSummary.axes.
_RUN_AXES = {
    "steps": "steps",
    "cfg": "guidance_scale",
    "mu": "COALESCE(mu, 'none')",
    "true_cfg": "COALESCE(true_cfg, 'none')",
    "lora_scale": "COALESCE(lora || '@' || lora_scale, 'none')",
    "strength": "COALESCE(strength, 'none')",
    "reference": "COALESCE(reference, 'none')",
    "seed": "seed",
    "variant": "CASE WHEN json_valid(variables) THEN json_extract(variables, '$.variant') END",
}
_AXIS_COUNTS = ", ".join(f"COUNT(DISTINCT {expr}) AS axis_{name}" for name, expr in _RUN_AXES.items())


def runs(limit: int = 20, model_id: str | None = None) -> list[RunSummary]:
    """The gallery grouped by run, the most recently written first.

    What `GET /api/runs` serves and the Runs picker lists: the night's output
    as one line per grid rather than 480 cards to scroll past. Rows written
    before runs had an id are not a run and are left out. Size, checkpoint and
    quantizer are constant within a run - `run_id` hashes them - so they are
    read off the oldest image along with its prompt and its seed, which is the
    base seed the grid was started from. `model_id` keeps the list to one
    checkpoint's runs, as the gallery it is a table of contents for is kept.
    """
    scope = "AND model_id = ?" if model_id else ""
    scope_params = (model_id,) if model_id else ()
    with _session() as conn:
        groups = conn.execute(
            f"""SELECT run_id, COUNT(*) AS images, SUM(favorite) AS favorites,
                      COUNT(DISTINCT prompt) AS prompts, COUNT(DISTINCT job_id) AS jobs,
                      COUNT(content_hash) AS hashed,
                      COUNT(DISTINCT content_hash) AS distinct_hashes,
                      MIN(id) AS first_id, MAX(id) AS last_id,
                      MIN(created_at) AS started_at, MAX(created_at) AS finished_at,
                      {_AXIS_COUNTS},
                      COUNT(DISTINCT COALESCE(refine, -1)) AS n_sources, MIN(refine) AS refine,
                      MIN(strength) AS strength_min, MAX(strength) AS strength_max
               FROM images WHERE run_id IS NOT NULL {scope}
               GROUP BY run_id ORDER BY last_id DESC LIMIT ?""",
            (*scope_params, limit),
        ).fetchall()
        firsts: dict[int, sqlite3.Row] = {}
        if groups:
            marks = ",".join("?" * len(groups))
            for row in conn.execute(
                f"SELECT * FROM images WHERE id IN ({marks})", [g["first_id"] for g in groups]
            ):
                firsts[row["id"]] = row
    summaries = []
    for group in groups:
        first = firsts[group["first_id"]]
        summaries.append(
            RunSummary(
                run_id=group["run_id"],
                images=group["images"],
                # Only when every image of the run is hashed: a count over
                # half a run is a number that looks like an answer.
                distinct=(
                    group["distinct_hashes"] if group["hashed"] == group["images"] else None
                ),
                favorites=group["favorites"],
                prompts=group["prompts"],
                jobs=group["jobs"],
                prompt=first["prompt"],
                seed=first["seed"],
                model_id=first["model_id"],
                quantization=first["quantization"],
                width=first["width"],
                height=first["height"],
                started_at=group["started_at"],
                finished_at=group["finished_at"],
                axes={name: group[f"axis_{name}"] for name in _RUN_AXES if group[f"axis_{name}"] > 1},
                refine=group["refine"] if group["n_sources"] == 1 else None,
                strengths=(
                    [group["strength_min"], group["strength_max"]]
                    if group["n_sources"] == 1 and group["refine"] is not None else None
                ),
            )
        )
    return summaries


def reference_paths(ids: Sequence[int], noun: str = "reference image") -> list:
    """The PNGs behind a request's reference ids, in the order they were given.

    The whole reason references are ids: a path taken from the request and
    joined onto `outputs_dir` is an arbitrary-file read reachable from
    /api/generate, while a row is something this app wrote. Missing ids and
    missing files are the same kind of mistake - the reference cannot be
    encoded - and both are named, so a night refused at submit says which
    image it could not find rather than dying at cell 1. A refine pass's
    source is resolved here too, under its own `noun`.
    """
    if not ids:
        return []
    wanted = list(dict.fromkeys(ids))
    marks = ",".join("?" * len(wanted))
    with _session() as conn:
        rows = conn.execute(
            f"SELECT id, filename FROM images WHERE id IN ({marks})", tuple(wanted)
        ).fetchall()
    by_id = {row["id"]: settings.outputs_dir / row["filename"] for row in rows}
    unknown = [image_id for image_id in wanted if image_id not in by_id]
    if unknown:
        raise ValueError(
            f"no gallery image with id {', '.join(str(i) for i in unknown)}"
        )
    gone = [image_id for image_id in wanted if not by_id[image_id].exists()]
    if gone:
        raise ValueError(
            f"the file for {noun} "
            f"{', '.join(str(i) for i in gone)} is missing from the gallery directory"
        )
    return [by_id[image_id] for image_id in ids]


def get_image(image_id: int) -> ImageRecord | None:
    with _session() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        missing = _missing_sources(conn, [row]) if row else set()
    return _row_to_record(row, row["refine"] in missing) if row else None


def set_favorite(image_id: int, favorite: bool) -> ImageRecord | None:
    with _write_lock, _session() as conn:
        conn.execute("UPDATE images SET favorite = ? WHERE id = ?", (int(favorite), image_id))
    return get_image(image_id)


def delete_image(image_id: int) -> bool:
    # Looked up and deleted under the one lock: two deletes of the same image
    # (a double-click, a second tab) both found the row before either removed
    # it, and both answered as though they had deleted it.
    with _write_lock, _session() as conn:
        row = conn.execute("SELECT filename FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    # The row is already gone, so a file someone else has open must not turn a
    # delete that did happen into a 500. clear_history's orphan sweep collects
    # whatever gets left behind.
    _unlink(settings.outputs_dir / row["filename"])
    _unlink(_thumb_path(settings.outputs_dir / row["filename"]))
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
    total = favorites = sources_kept = 0
    tracked: set[str] = set()
    # The directory first, the rows second: `save_image` commits before it
    # renames, so a PNG the scan can see already has a row by the time the rows
    # are read. The other order judged an image that landed in between as an
    # orphan.
    files = list(settings.outputs_dir.glob("*.png"))
    thumbs = list(settings.outputs_dir.glob("*.webp"))
    if settings.db_path.exists():
        with _session() as conn:
            total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            favorites = conn.execute(
                "SELECT COUNT(*) FROM images WHERE favorite = 1"
            ).fetchone()[0]
            # What `--keep-favorites` keeps besides them, as `_UNKEPT` reads it.
            sources_kept = conn.execute(
                f"SELECT COUNT(*) FROM images WHERE favorite = 0 AND {_STARRED_SOURCE}"
            ).fetchone()[0]
        tracked = {r["filename"] for r in _tracked_files()}

    total_bytes = 0
    for path in files + thumbs:
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
        "sources_kept": sources_kept,
        "files_on_disk": len(on_disk),
        "orphan_files": len(on_disk - tracked),
        "partial_files": len(_stale_partials()),
        "bytes_on_disk": total_bytes,
    }


def _tracked_files():
    with _session() as conn:
        return conn.execute("SELECT filename FROM images").fetchall()


def clear_history(keep_favorites: bool = False) -> dict[str, int]:
    """Delete image records and the PNGs they point at.

    Rows are deleted rather than the database file, so this is safe while the
    server holds the connection open - dropping the file mid-run would leave a
    locked handle on Windows and lose the schema. `keep_favorites` keeps what
    it keeps on the `--where` path too: the stars and their sources.
    """
    where = f" WHERE {_UNKEPT}" if keep_favorites else ""
    with _write_lock, _session() as conn:
        doomed = [r["filename"] for r in conn.execute(f"SELECT filename FROM images{where}")]
        conn.execute(f"DELETE FROM images{where}")

    removed = 0
    for filename in doomed:
        path = settings.outputs_dir / filename
        if path.exists() and _unlink(path):
            removed += 1
        _unlink(_thumb_path(path))

    # Anything left with no row behind it: earlier deletions, interrupted runs.
    # An image still being written is a *.png.part and matches neither glob, so
    # a clear during a sweep can no longer delete one out from under it. The
    # scan is materialised *before* the rows are read: a save commits its row
    # and then renames, so any PNG the scan saw has a row by the time `keep`
    # is built - read the other way round, an image landing in between was
    # deleted with its row left pointing at nothing.
    files = list(settings.outputs_dir.glob("*.png"))
    keep = {r["filename"] for r in _tracked_files()}
    orphans = 0
    for path in files:
        if path.name not in keep and _unlink(path):
            orphans += 1
    # A thumbnail whose PNG is gone - deleted above, or by hand.
    for path in settings.outputs_dir.glob("*.webp"):
        if not path.with_suffix(".png").exists():
            _unlink(path)
    # Partials from a power cut or a killed process, by age - never the fresh
    # one another process may be encoding right now.
    partials = sum(1 for path in _stale_partials() if _unlink(path))

    return {"records": len(doomed), "files": removed, "orphans": orphans, "partials": partials}


def vacuum() -> None:
    """Shrink the database file after a bulk delete."""
    conn = _connect()
    try:
        conn.isolation_level = None  # VACUUM cannot run inside a transaction
        conn.execute("VACUUM")
    finally:
        conn.close()
