"""Gallery persistence: SQLite index plus PNGs carrying their own metadata.

The read-only aggregate queries (estimates, keep rates, runs) are
`gallery_reports`; taking images out (scoped delete, recycle bin, clear) is
`gallery_delete`. Both read this module's settings and helpers at call time,
and their names still resolve here through `__getattr__` at the end.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib
import json
import logging
import re
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, PngImagePlugin

from .config import settings
from .execution import canonical_execution, execution_chunk, execution_from_json, execution_key
from . import gallery_lock, refine
# The key elements that are built from columns here and from a request
# in `grid`: one spelling of each, or a resume compares strings that only look
# alike. `grid` imports neither this module nor anything that does.
from .grid import lora_key, mask_key, reference_key, refine_key, shape_key
from .instruction_edit import InstructionEditSource, descriptor_from_image, edit_key
from .metadata import PNG_MASK_KEY, PNG_METADATA_KEY, canonical_edit, edit_chunk, edit_from_json, validate_edit_record
from .png_settings import crop_anchor as _crop_anchor
from .search import like_escape, split_search
from .schemas import ImageRecord, OutpaintSpec, effective_true_cfg

if TYPE_CHECKING:
    from .grid import Cell, CellKey
    from .schemas import GenerateRequest

log = logging.getLogger(__name__)

_write_lock = threading.Lock()
# `store_mask`'s, held across its encode, which `_write_lock` is not.
_mask_lock = threading.Lock()

# The long side of the WEBP written in outputs/thumbnails. The gallery's cards are
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
    return path.parent / "thumbnails" / path.with_suffix(".webp").name


def _thumb_files() -> list:
    """New thumbnails and any old siblings a migration could not move."""
    return [
        *settings.outputs_dir.glob("*.webp"),
        *(settings.outputs_dir / "thumbnails").glob("*.webp"),
    ]


def _delete_thumbnails(path) -> None:
    _unlink(_thumb_path(path))
    _unlink(path.with_suffix(".webp"))


def _move_thumbnail(png) -> None:
    """Move a known PNG's legacy thumbnail, without replacing a destination."""
    old = png.with_suffix(".webp")
    target = _thumb_path(png)
    try:
        if target.exists() or not png.is_file() or not old.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        old.rename(target)
    except OSError:
        # A locked file can wait for another startup; record reads still find
        # it in the old location. A cache failure must never cost the PNG.
        log.warning("thumbnail %s could not be moved to thumbnails/", old.name, exc_info=True)


def _migrate_thumbnails() -> None:
    """Relocate indexed thumbnails at startup, leaving unrelated WebPs alone."""
    tracked = {row["filename"] for row in _tracked_files()}
    for old in settings.outputs_dir.glob("*.webp"):
        png = old.with_suffix(".png")
        if png.name in tracked:
            _move_thumbnail(png)


def _encode_thumb(image: Image.Image, path):
    """A small WEBP for the cards and the sheet, encoded to `<path>.part`.

    Returns the partial for `_place_thumb`, or None when encoding failed. It
    is encoded outside the gallery reservation, so readers on other threads
    do not wait on it, and renamed into place *after* the PNG's rename, so a
    failed INSERT leaves no thumbnail for a row that never existed. The
    `.part` name keeps the orphan sweep off a half-written one. A thumbnail
    that fails costs bandwidth, not the image: the record falls back to the PNG.
    """
    partial = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        small = image.copy()
        small.thumbnail((THUMB_SIDE, THUMB_SIDE))
        small.save(partial, format="WEBP", quality=82, method=4)
    except Exception:  # noqa: BLE001 - the PNG and its row are already safe
        log.warning("thumbnail for %s not written", path.name, exc_info=True)
        _unlink(partial)
        return None
    except BaseException:
        _unlink(partial)
        raise
    return partial


def _place_thumb(partial, path) -> None:
    """Rename an encoded thumbnail into place; called under the reservation."""
    try:
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
    strength        REAL,
    crop_anchor     TEXT,
    batch           INTEGER,
    binned_at       TEXT,
    mask            TEXT,
    outpaint        TEXT,
    upscaler        TEXT,
    execution       TEXT,
    edit            TEXT,
    vram_peak_allocated INTEGER,
    vram_peak_reserved  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_images_created ON images (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_images_job ON images (job_id);
CREATE INDEX IF NOT EXISTS idx_images_run ON images (run_id);
CREATE INDEX IF NOT EXISTS idx_images_binned ON images (binned_at);
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts
    USING fts5(prompt, content='images', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, prompt) VALUES (new.id, new.prompt);
END;
CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, prompt) VALUES('delete', old.id, old.prompt);
END;
CREATE INDEX IF NOT EXISTS idx_images_mask ON images (mask);
CREATE INDEX IF NOT EXISTS idx_images_outpaint ON images (outpaint);
"""

# The inpaint masks the server holds, by the hash `mask_hash` names each one
# by. Not gallery images: no row in `images`, no card, no bin state, never a
# reference or a source. Created on every connection, as `_migrate` runs on
# every one: a database file replaced under a running server - restored
# from a backup - has no such table until it is given one.
_MASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS masks (
    hash        TEXT    PRIMARY KEY,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    created_at  TEXT    NOT NULL
)
"""


# The masks a server's queued and running jobs will read, by job: the hold a
# mask sweep in any process honours (`gallery_delete.sweep_masks`), since a
# queued cell's mask is in no row until it saves and the CLI cannot ask the
# server's queue. Written before a job is queued (`JobManager.submit_cells`),
# removed when it ends, and all dropped when a server starts - no job
# survives a restart. Created on every connection, like `masks`.
_MASK_HOLDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS mask_holds (
    hash    TEXT NOT NULL,
    job_id  TEXT NOT NULL,
    PRIMARY KEY (hash, job_id)
)
"""


# Resolved browser jobs, including cells the worker never reached. Unlike
# mask_holds these survive a restart; bundles deliberately leave them out.
_RUN_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_jobs (
    job_id       TEXT    PRIMARY KEY,
    run_id       TEXT    NOT NULL,
    created_at   REAL    NOT NULL,
    origin       TEXT    NOT NULL,
    model_id     TEXT    NOT NULL,
    quantization TEXT    NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    batch        INTEGER NOT NULL,
    comparison   INTEGER NOT NULL,
    metadata     TEXT    NOT NULL,
    cells        BLOB    NOT NULL,
    outcome      TEXT
)
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that a database written by an older build has not got.

    `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it was, so
    a gallery from before this column simply never gains it - and every INSERT
    then fails. Checked on every connection, as the table's existence is: the
    file can be replaced under a running server - restored from a backup at
    the same path - and a once-per-process flag, even one keyed on the path,
    left that older file unmigrated until a restart, every gallery read
    failing on `binned_at`. `PRAGMA table_info` costs what the sqlite_master
    lookup beside it does; the ALTER itself runs at most once per file, and
    races with another process end in "duplicate column", which is the
    outcome we wanted anyway.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
    added = (
        ("job_id", "TEXT", "idx_images_job"),
        ("run_id", "TEXT", "idx_images_run"),
        ("quantization", "TEXT", None),
        # The prompt-variable assignment, as JSON.
        ("variables", "TEXT", None),
        # The scale the negative prompt ran at; NULL on older rows means
        # "guidance_scale, if there was a negative prompt" - see
        # `schemas_generate.effective_true_cfg`.
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
        # NULL preserves the legacy centered crop; a choice belongs only to
        # an ordinary Refine source.
        ("crop_anchor", "TEXT", None),
        # How many images the pipeline call that made this one held - the
        # panel's Batch - which More seeds runs a batch's extra images at.
        # NULL on rows from before it was recorded.
        ("batch", "INTEGER", None),
        # When the image was moved to the recycle bin, ISO UTC like
        # `created_at`; NULL while it is live. TEXT rather than a flag so a
        # bundle merged into an older table gains it through the same
        # fallback as any other unlisted column, and never NOT NULL: a row
        # from an older database or bundle reads None and is live.
        ("binned_at", "TEXT", "idx_images_binned"),
        # An inpaint's mask, the hash of a `masks` row; NULL on every row
        # that is not one, and only ever set beside `refine`.
        ("mask", "TEXT", "idx_images_mask"),
        # An outpaint's spec, `OutpaintSpec.key()`: NULL on every row that
        # is not one, and only ever set beside `refine` and `mask`.
        ("outpaint", "TEXT", "idx_images_outpaint"),
        # The optional GAN preparing an ordinary Refine source.
        ("upscaler", "TEXT", None),
        # Named arithmetic recipes as canonical JSON; NULL is legacy Eager.
        ("execution", "TEXT", None),
        # Complete instruction-edit intent, including unresolved source proof.
        ("edit", "TEXT", None),
        # The peak GPU memory PyTorch's allocator held during the call that
        # made this image, in bytes, allocated and reserved: a result, as
        # `duration_ms` is, never part of the reproduction set. NULL where
        # nothing was measured (a CPU, the fake, rows from before it).
        ("vram_peak_allocated", "INTEGER", None),
        ("vram_peak_reserved", "INTEGER", None),
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


def ensure_outputs_dir() -> None:
    """Create the images directory. Called before anything writes into it.

    Here rather than at import of `config`: `clear.py --dry-run` promises to
    change nothing, and an import-time mkdir created the directory anyway - and
    on an unmounted `AURADREAM_OUTPUTS_DIR` created the mount point as a plain
    directory, the state that later hides the real data behind an empty mount.
    """
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)


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
    conn.execute(_MASKS_SCHEMA)
    conn.execute(_MASK_HOLDS_SCHEMA)
    conn.execute(_RUN_JOBS_SCHEMA)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_jobs_run ON run_jobs (run_id)")
    return conn


@contextmanager
def _session():
    """A connection that commits on success and is always closed.

    `with sqlite3.connect(...) as conn` commits but does *not* close, which leaks
    a handle per call and keeps the file locked on Windows - deleting or moving
    auradream.db then fails while the server runs.
    """
    with gallery_lock.hold(settings.db_path, wait=gallery_lock.SESSION_WAIT):
        conn = _connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def init_db() -> None:
    ensure_outputs_dir()
    with _session() as conn:
        conn.executescript(_SCHEMA)
    _migrate_thumbnails()


def _row_to_record(
    row: sqlite3.Row | Mapping[str, Any], refine_missing: bool = False, refine_binned: bool = False,
    edit_missing: bool = False, edit_binned: bool = False,
) -> ImageRecord:
    """A gallery row as the record the API serves.

    Every column is read unguarded. Database rows come out of a connection
    `_connect` opened, and `_connect` runs `_migrate` first, so a database from
    an older build has every column by the time a row is fetched - NULL where
    the build that wrote it knew nothing of the field. A new column therefore
    needs an entry in `_migrate`, not a read guard here: a guard would let the
    SELECT succeed and every INSERT after it fail on the missing column.
    `refine_missing` and `refine_binned` are no columns: the reader that has
    looked the source up (`_missing_sources`) says so.
    Bundle validation supplies the same complete columns as a mapping.
    """
    # Rows from before thumbnails were written have none; the client falls back
    # to the PNG. A locked legacy thumbnail may still be beside its PNG.
    thumb = _thumb_path(settings.outputs_dir / row["filename"])
    if not thumb.is_file():
        thumb = (settings.outputs_dir / row["filename"]).with_suffix(".webp")
    edit = edit_from_json(row["edit"])
    validate_edit_record(edit, dict(row))
    return ImageRecord(
        id=row["id"],
        filename=row["filename"],
        url=f"/outputs/{row['filename']}",
        thumb_url=f"/outputs/{thumb.relative_to(settings.outputs_dir).as_posix()}" if thumb.is_file() else None,
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
        refine_binned=refine_binned,
        strength=row["strength"],
        crop_anchor=_crop_anchor(
            row["crop_anchor"], source=row["refine"], mask=row["mask"], outpaint=row["outpaint"],
        ),
        upscaler=row["upscaler"],
        execution=execution_from_json(row["execution"]),
        edit=edit,
        edit_missing=edit_missing or (edit is not None and edit.source is None),
        edit_binned=edit_binned,
        batch=row["batch"],
        job_id=row["job_id"],
        run_id=row["run_id"],
        duration_ms=row["duration_ms"],
        created_at=row["created_at"],
        favorite=bool(row["favorite"]),
        content_hash=row["content_hash"],
        binned_at=row["binned_at"],
        mask=row["mask"],
        # Unguarded, as every column: the stored type has no bound a setting
        # could move, and only validated keys are ever written.
        outpaint=OutpaintSpec.parse_key(row["outpaint"]) if row["outpaint"] else None,
        vram_peak_allocated=row["vram_peak_allocated"],
        vram_peak_reserved=row["vram_peak_reserved"],
    )


# Every column a writer supplies, in one place: `save_image` and the reindex
# both insert through `_insert_row`, so the list cannot drift between them.
_COLUMNS = (
    "filename", "prompt", "negative_prompt", "width", "height", "steps", "guidance_scale",
    "mu", "seed", "model_id", "duration_ms", "favorite", "created_at", "job_id", "run_id",
    "quantization", "variables", "true_cfg", "content_hash", "reference", "lora", "lora_scale",
    "refine", "strength", "crop_anchor", "batch", "binned_at", "mask", "outpaint", "upscaler",
    "execution", "edit", "vram_peak_allocated", "vram_peak_reserved",
)

# What every live-facing read adds: an image in the recycle bin is gone from
# listings, selections, references, stars, runs and stats, and present for
# every read about what exists on disk or has already been made - the orphan
# sweep, reindex, bundles, resume keys, lineage, the cost samples.
_LIVE = "binned_at IS NULL"


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
    validate_edit_record(edit_from_json(values.get("edit")), values)
    columns = ", ".join(_COLUMNS)
    marks = ", ".join("?" * len(_COLUMNS))
    cur = conn.execute(
        f"INSERT INTO images ({columns}) VALUES ({marks})", tuple(values[c] for c in _COLUMNS)
    )
    return conn.execute("SELECT * FROM images WHERE id = ?", (cur.lastrowid,)).fetchone()


def image_provenance(image_id: int | None) -> dict[str, Any] | None:
    """Portable identity for a gallery-local conditioning id in a PNG.

    Dimensions accompany the decoded-pixel hash because the hash itself does
    not encode the shape. A filename locates the source during PNG recovery;
    the hash and size verify it before any destination id can be assigned.
    """
    # Every row: provenance describes what exists, and a source binned while
    # its pass ran is still the picture the pass was made from.
    record = get_image(image_id, include_binned=True) if image_id is not None else None
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
    batch: int | None = None,
    peak: tuple[int, int] | None = None,
) -> ImageRecord:
    """Write the PNG and its row. `request` must be what actually ran -
    `backend.applied()` - not what was asked for: a negative prompt the
    pipeline never saw, recorded here, defeats resume and misleads Reuse.
    `variables` is the prompt-variable assignment behind a grid cell, kept so
    the sheet can show what varied rather than reconstruct it from prose.
    `batch` is how many images the call that made this one held: not part
    of the reproduction set (one generator per image), but what More seeds
    runs the batch's next images at. `peak` is the (allocated, reserved)
    bytes the call held at most - its batch's whole, a result like the
    duration - or None where nothing was measured.
    """
    created = datetime.now(timezone.utc)
    execution = canonical_execution(request.execution)
    edit = edit_from_json(edit_key(request.edit))
    allocated, reserved = (_peak(peak[0]), _peak(peak[1])) if peak is not None else (None, None)
    variables_json = json.dumps(variables) if variables else None
    mask = request.mask if request.refine is not None else None
    # An outpaint never without its source and its stamped mask.
    outpaint = request.outpaint if mask is not None else None
    upscaler = _upscaler(
        request.upscaler, source=request.refine, model_id=model_id, mask=mask, outpaint=request.outpaint,
    )
    crop_anchor = _crop_anchor(
        request.crop_anchor, source=request.refine, mask=request.mask, outpaint=request.outpaint,
    )
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
        # An inpaint's mask and what it was, null together on every other
        # row as the source is: `applied()` never leaves a mask without its
        # source. The composite over the source is what an inpaint is here,
        # not a setting, and is named so that a later rule can be told apart.
        "mask": mask,
        "mask_source": mask_provenance(mask, image.width, image.height),
        "mask_blend": "overlay" if mask is not None else None,
        # An outpaint's padding and band, and the fill that made its new
        # area, null together on every other row.
        "outpaint": outpaint_chunk(outpaint),
        "outpaint_fill": outpaint.fill if outpaint is not None else None,
        "batch": batch,
        # Absent for the default so legacy PNG metadata stays byte-for-byte
        # unchanged when no crop position was requested.
        **({"crop_anchor": crop_anchor} if crop_anchor is not None else {}),
        **({"upscaler": upscaler} if upscaler is not None else {}),
        **execution_chunk(execution),
        **edit_chunk(edit),
        # Results, not settings, so absent rather than null when unmeasured:
        # a chunk from a CPU or before them reads the same.
        **({"vram_peak_allocated": allocated} if allocated is not None else {}),
        **({"vram_peak_reserved": reserved} if reserved is not None else {}),
    }
    info = PngImagePlugin.PngInfo()
    validate_edit_record(edit, meta)
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
        )
        + (f", Inpaint mask: {mask[:16]}" if mask is not None else "")
        + (f", Hires upscaler: {refine.ANIME6B_LABEL}" if upscaler is not None else "")
        + (f", Crop anchor: {crop_anchor}" if crop_anchor is not None else "")
        + (f", Instruction edit: {edit.recipe}" if edit is not None else "")
        + (outpaint_parameters(outpaint) if outpaint is not None else ""),
    )
    info.add_text(PNG_METADATA_KEY, json.dumps(meta))
    # The mask itself, so this PNG alone reproduces the image (`PNG_MASK_KEY`).
    embedded = mask_png(mask) if mask is not None else None
    if embedded is not None:
        info.add_text(PNG_MASK_KEY, base64.b64encode(embedded).decode("ascii"))

    committed = False
    thumb_path = _thumb_path(path)
    thumb = None
    try:
        # Inside the guard: a KeyboardInterrupt mid-encode - the CLI's second
        # Ctrl-C - is a BaseException Pillow's own cleanup does not catch, and
        # it left a *.png.part that nothing swept. Both encodes run before the
        # gallery reservation, which other threads' reads also need: held
        # through a 2048x2048 encode, a 3 ms `get_image` took 330 ms. Only the
        # row and the renames are reserved. Sweeps do glob `*.part`: only
        # PARTIAL_SETTLE_SECONDS keeps them off an encode still running here,
        # and a bundle inventory refuses while one exists.
        image.save(partial, format="PNG", pnginfo=info, compress_level=4)
        thumb = _encode_thumb(image, thumb_path)
        with gallery_lock.hold(settings.db_path, wait=gallery_lock.SESSION_WAIT):
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
                        "crop_anchor": crop_anchor,
                        "batch": batch,
                        "binned_at": None,
                        "mask": mask,
                        "outpaint": outpaint.key() if outpaint is not None else None,
                        "upscaler": upscaler,
                        "execution": execution_key(execution),
                        "edit": edit_key(edit),
                        "vram_peak_allocated": allocated,
                        "vram_peak_reserved": reserved,
                    },
                )
                # A pass whose source was deleted while it ran - read as the
                # cell began, gone by its end - says so as list_images would,
                # under the lock a delete takes: returned as present, the page
                # showed it as one that could be made again until a reload.
                missing, binned = _missing_sources(conn, [row])
                edit_states = _edit_states(conn, [row])
            committed = True
            _rename_with_retry(partial, path)
            if thumb is not None:
                _place_thumb(thumb, thumb_path)
                thumb = None
    except BaseException:
        # A failed encode or INSERT must not leave a *.png.part behind for
        # nobody - and tidying up must not replace the error that got us here.
        # Only while the row did *not* commit, though: past that point the
        # partial is the only copy of an image the database already names, and
        # deleting it turned a transient rename failure into a phantom row.
        if not committed:
            _unlink(partial)
        raise
    finally:
        if thumb is not None:
            _unlink(thumb)
    return _row_to_record(row, row["refine"] in missing, row["refine"] in binned, *edit_states[row["id"]])


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


# Bundle imports unpack, and uploads land, in this directory on the images'
# volume, so installing a staged image is a rename rather than a second copy.
# Its name matches none of the sweeps' globs or `_BUNDLE_WORK`.
BUNDLE_STAGING = ".auradream-staging"
# The same directory as Local Diffusion named it. Nothing stages there any
# more; only what a killed operation of that build left is swept from it.
LEGACY_BUNDLE_STAGING = (".local-diffusion-staging",)
# What bundles.py and api_bundles.py create beside the database: export's
# snapshot directory and import's merge directory (tempfile's eight random
# characters), and uploads from before they moved to BUNDLE_STAGING.
_BUNDLE_WORK = re.compile(r"\.bundle-(?:import-|merge-|upload-)?[a-z0-9_]{8}(?:\.part)?")


def bundle_directory() -> Path:
    """The saved archives' directory, shared by reads, writes and cleanup."""
    # bundle_common imports reindex, which imports storage: defer the error
    # import until this module is ready rather than introducing a cycle.
    from .bundle_common import BundleError

    directory = settings.bundles_dir
    if directory is None:
        directory = settings.db_path.parent.resolve() / "bundles"
    # Check the configured name before resolve follows a direct junction or
    # symlink and hides it. Parent links still resolve for the containment test.
    if directory.is_symlink() or directory.is_junction():
        raise BundleError("the bundles directory must not be a symbolic link")
    directory = directory.resolve()
    outputs = settings.outputs_dir.resolve()
    if directory == outputs or directory.is_relative_to(outputs):
        raise BundleError("the bundles directory must be outside the images directory")
    return directory


def bundle_staging_dir() -> Path:
    """The staging directory on the images' volume, created on demand.

    Beside the images directory, out of what `/outputs` serves, unless that
    is another volume (the images directory is a mount point); then inside
    it, where the inventory skips directories.
    """
    ensure_outputs_dir()
    outputs = settings.outputs_dir.resolve()
    root = outputs.parent
    try:
        if root.stat().st_dev != outputs.stat().st_dev:
            root = outputs
    except OSError:
        root = outputs
    staging = root / BUNDLE_STAGING
    staging.mkdir(exist_ok=True)
    return staging


def _bundle_leftovers() -> list:
    """Everything a bundle operation writes outside the gallery proper.

    A finished or failed operation removes its own; a killed one (power cut,
    a second Ctrl-C) leaves a gallery-sized copy that nothing else reaches.
    """
    beside = settings.db_path.parent
    outputs = settings.outputs_dir
    return [
        *bundle_directory().glob("*.part"),
        *(p for p in beside.glob(".bundle-*") if _BUNDLE_WORK.fullmatch(p.name)),
        *(
            path
            for name in (BUNDLE_STAGING, *LEGACY_BUNDLE_STAGING)
            for root in (outputs.parent, outputs)
            for path in (root / name).glob("*")
        ),
    ]


def _stale_partials() -> list:
    """Partials nothing can still be writing - see PARTIAL_SETTLE_SECONDS.

    Bundle leftovers too: an hour is as safe for them. Every bundle operation
    holds the gallery reservation throughout, as every sweep does, and an
    upload still arriving keeps writing its file.
    """
    cutoff = time.time() - PARTIAL_SETTLE_SECONDS
    stale = []
    paths = [
        *settings.outputs_dir.glob("*.part"),
        *(settings.outputs_dir / "thumbnails").glob("*.part"),
        *masks_dir().glob("*.part"),
        *_bundle_leftovers(),
    ]
    for path in paths:
        try:
            if path.stat().st_mtime < cutoff:
                stale.append(path)
        except OSError:
            continue
    return stale


def _discard(path) -> bool:
    """Remove a stale partial: a file, or a bundle's work directory."""
    if path.is_dir() and not path.is_symlink() and not path.is_junction():
        shutil.rmtree(path, ignore_errors=True)
        return not path.exists()
    return _unlink(path)


def _settle_plan(stale: Sequence, tracked: set[str] | None = None) -> list:
    """Each stale partial with the name it should be renamed to, or None.

    `save_image` commits the row and then renames, so a `<name>.png.part`
    whose row exists and whose `<name>.png` does not is the only copy of a
    committed image - the rename failed, or the process died between the two.
    That one is finished, never discarded; every other stale partial is
    debris. A thumbnail's `.webp.part` is debris too, even beside a committed
    row: `thumbnail_for` encodes for rows already tracked, so one may be cut
    short, and a placed thumbnail is never encoded again. The card falls back
    to the PNG until a reindex writes a whole one. A mask's `.png.part` whose
    `masks` row exists is finished likewise: `store_mask` commits the row
    before the rename too - but only once it decodes to the mask its name
    says (`_whole_mask`). A row can already be there while a lost file is
    encoded again, so a partial cut short by a crash is debris, never
    installed as the mask.
    """
    if tracked is None:
        tracked = {row["filename"] for row in _tracked_files()}
    outputs = settings.outputs_dir
    masks = masks_dir()
    held = _tracked_masks() if any(path.parent == masks for path in stale) else set()
    plan = []
    for path in stale:
        name = path.name[: -len(".part")] if path.name.endswith(".part") else None
        target = None
        if name and path.parent == outputs and name.endswith(".png") and name in tracked:
            target = path.with_name(name)
        elif (
            name and path.parent == masks and name.endswith(".png") and name[: -len(".png")] in held
            and _whole_mask(path, name[: -len(".png")])
        ):
            target = path.with_name(name)
        plan.append((path, None if target is None or target.exists() else target))
    return plan


def _whole_mask(path: Path, digest: str) -> bool:
    """Whether the file at `path` is a whole PNG of the mask `digest` names."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.format == "PNG" and image.mode == "L" and mask_hash(image) == digest
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return False


def _settle(stale: Sequence) -> tuple[int, int]:
    """Finish or discard stale partials, under the reservation the caller holds.

    Returns (recovered, discarded). A recovery that fails leaves the partial
    where it is: it is the image, and the next sweep tries again.
    """
    recovered = discarded = 0
    for path, target in _settle_plan(stale):
        if target is None:
            discarded += _discard(path)
            continue
        try:
            _rename_with_retry(path, target)
            recovered += 1
            log.info("recovered %s from an interrupted save", target.name)
        except OSError:
            log.warning("could not finish the interrupted save of %s", target.name, exc_info=True)
    return recovered, discarded


@gallery_lock.protected(lambda: settings.db_path)
def settle_partials() -> dict[str, int]:
    """Finish or sweep what killed saves and bundle operations left.

    At startup, at every bundle preview and before each bundle operation. Only
    partials older than PARTIAL_SETTLE_SECONDS: a fresh one may be an encode
    in flight, which a bundle inventory refuses instead.
    """
    recovered, discarded = _settle(_stale_partials())
    return {"recovered": recovered, "discarded": discarded}


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
    binned: bool = False,
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

    The recycle bin is left out, or with `binned` is all there is: the bin's
    own view, under the same search and paging.
    """
    # The scope stays out of `where`: the fallback below swaps the *last*
    # predicate, and the count of what the scope hides needs the rest alone.
    scope = "model_id = ?" if model_id else ""
    where: list[str] = []
    params: list = []
    if favorites_only:
        where.append("favorite = 1")
    # Before the search predicate, which the LIKE fallback swaps as the last.
    where.append("binned_at IS NOT NULL" if binned else _LIVE)
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
        missing, in_bin = _missing_sources(conn, rows)
        edit_states = _edit_states(conn, rows)
    return _records(rows, missing, in_bin, edit_states), total, others


def _records(
    rows: Sequence[sqlite3.Row], missing: set[int], binned: set[int],
    edit_states: Mapping[int, tuple[bool, bool]],
) -> list[ImageRecord]:
    return [_row_to_record(r, r["refine"] in missing, r["refine"] in binned, *edit_states[r["id"]]) for r in rows]


def _edit_states(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> dict[int, tuple[bool, bool]]:
    """Derived availability, proved from current pixels rather than short DB hashes."""
    states = {}
    proven: dict[int, tuple[InstructionEditSource | None, bool]] = {}
    for row in rows:
        edit = edit_from_json(row["edit"])
        missing, binned = False, False
        if edit is not None:
            if edit.source is None:
                missing = True
            else:
                if edit.source not in proven:
                    source = conn.execute("SELECT filename, width, height, binned_at FROM images WHERE id = ?", (edit.source,)).fetchone()
                    descriptor = None
                    if source is not None:
                        try:
                            filename = source["filename"]
                            # Validate the basename before joining a database-supplied filename.
                            InstructionEditSource._bare_png(filename)
                            path = settings.outputs_dir / filename
                            if path.is_symlink():
                                raise ValueError("instruction-edit source cannot be a linked file")
                            with Image.open(path) as image:
                                if image.size == (source["width"], source["height"]):
                                    descriptor = descriptor_from_image(filename, image)
                        except (OSError, ValueError, Image.DecompressionBombError):
                            pass
                    proven[edit.source] = descriptor, source is not None and source["binned_at"] is not None
                descriptor, binned = proven[edit.source]
                missing = binned or descriptor != edit.edit_source
        states[row["id"]] = missing, binned
    return states


def _missing_sources(
    conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]
) -> tuple[set[int], set[int]]:
    """The refine sources these rows name that are no longer in the gallery,
    and those of them that are in the recycle bin.

    A pass keeps its source's id after the source is deleted - the id is the
    reproduction set's, and a reindex can remap it - so the record says the
    source has gone (`ImageRecord.refine_missing`) rather than every page
    naming an image it cannot show. A binned source is missing too, since it
    cannot be used until it is restored, and the second set says which, for
    `refine_binned`. Chunked like `completed_cells`: a sheet's page is a
    whole run.
    """
    wanted = list({row["refine"] for row in rows if row["refine"] is not None})
    present: set[int] = set()
    binned: set[int] = set()
    for start in range(0, len(wanted), 400):
        chunk = wanted[start : start + 400]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(f"SELECT id, binned_at FROM images WHERE id IN ({marks})", chunk):
            (binned if row["binned_at"] is not None else present).add(row["id"])
    return set(wanted) - present, binned


def completed_cells(
    prompts: Sequence[str],
    width: int,
    height: int,
    model_id: str,
    quantization: str | None = None,
) -> set[CellKey]:
    """The key of everything already rendered: the prompt pair, the settings,
    the references, adapter, refine source, mask, upscaler and canvas/crop,
    the seed.

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

    The recycle bin counts as made: a binned cell is a rendered one rejected,
    so Resume, Re-render's "not yet rendered", Mass refine's "rendered" and
    More seeds' held seeds do not make it again. Emptying the bin makes the
    cell eligible again, as a delete always has.
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
                           reference, lora, lora_scale, refine, strength, mask, outpaint, upscaler, edit, execution,
                           crop_anchor, seed
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
                        mask_key(row["mask"], _canonical_outpaint(row["outpaint"])),
                        row["upscaler"],
                        edit_key(edit_from_json(row["edit"])),
                        execution_key(execution_from_json(row["execution"])),
                        shape_key(width, height, row["crop_anchor"]),
                        row["seed"],
                    )
                )
    return keys


# These two and `_tracked_files` read every row, the recycle bin's included:
# filtered to live rows, the orphan sweep would delete binned PNGs as untracked
# and a reindex would add them back as new rows.
def existing_filenames() -> set[str]:
    """Every filename the index knows - what a reindex skips."""
    with _session() as conn:
        return {row["filename"] for row in conn.execute("SELECT filename FROM images")}


def existing_file_ids() -> dict[str, int]:
    """Every indexed filename and its gallery id, for an id-remapping import."""
    with _session() as conn:
        return {row["filename"]: row["id"] for row in conn.execute("SELECT id, filename FROM images")}


# How long the status poll's newest-id read waits for the gallery reservation
# before answering "unknown": a save holds it for a moment, a bundle operation
# or a clear for as long as it runs, and the poll must not stall behind either.
NEWEST_ID_WAIT = 0.2


def newest_image_id() -> int | None:
    """The gallery's highest id over every row, the recycle bin's included, or
    0 for an empty gallery; None where the gallery is reserved or unreadable.

    For the status poll: a page sees images it did not watch being made - a
    job of another tab's that ran between two polls, the CLI's - by this
    number moving up. Every row, because only an insert raises it: ids are
    AUTOINCREMENT and stay monotonic across a clear and a replaced database,
    so a new image always lands above it, while a bin or a restore leaves it
    where it was (a live-only maximum would fall at a bin and rise again at
    its restore, an image the page already knew). A permanent delete of the
    newest lowers it, and the next insert lands above the old value anyway.
    MAX(id) on the rowid is one b-tree descent. Short wait, no queueing: the
    poll answers "unknown" while maintenance holds the gallery."""
    try:
        with gallery_lock.hold(settings.db_path, wait=NEWEST_ID_WAIT):
            conn = _connect()
            try:
                return _newest_id(conn)
            finally:
                conn.close()
    except (gallery_lock.GalleryBusy, sqlite3.Error, OSError):
        return None


def _newest_id(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT MAX(id) FROM images").fetchone()[0] or 0)


def list_images_at_head(**kwargs: Any) -> tuple[list[ImageRecord], int, int, int]:
    """`list_images`, and `newest_image_id` read under the same reservation.

    The page's starting point for the status's newest id: taken from the
    first status, a separate read, an image saved between the gallery's
    SELECT and the status's was counted as seen and never shown. Every insert
    takes the reservation, so no row lands between these two reads, and the
    number says exactly what this page could have held."""
    with gallery_lock.hold(settings.db_path, wait=gallery_lock.SESSION_WAIT):
        items, total, others = list_images(**kwargs)
        with _session() as conn:
            newest = _newest_id(conn)
    return items, total, others, newest


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def import_file(values: dict[str, Any]) -> ImageRecord:
    """Insert one row copied from another gallery database.

    The caller has already copied the named PNG into ``outputs_dir`` and
    remapped any gallery-local reference ids.  Unlike ``index_file``, this
    path preserves the fields that are not embedded in the PNG: favorite and
    generation duration.
    """
    values = {
        **values,
        "execution": execution_key(execution_from_json(values.get("execution"))),
        "edit": edit_key(edit_from_json(values.get("edit"))),
        "crop_anchor": _crop_anchor(
            values.get("crop_anchor"), source=values.get("refine"),
            mask=values.get("mask"), outpaint=values.get("outpaint"),
        ),
    }
    with _write_lock, _session() as conn:
        row = _insert_row(conn, {column: values.get(column) for column in _COLUMNS})
        states = _edit_states(conn, [row])
    return _row_to_record(row, edit_missing=states[row["id"]][0], edit_binned=states[row["id"]][1])


@gallery_lock.protected(lambda: settings.db_path, waits=True)
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
    a grid off an unknown. Nor is the recycle bin in the file: an image that
    was binned comes back live, like an unstarred favourite. Through the same INSERT as `save_image`, so the FTS
    trigger indexes the prompt. `digest` is the hash of the pixels as the
    caller decoded them, which is the truth; the block's own `hash` stands in
    for it when the caller did not decode.
    """
    mask = _mask(meta.get("mask")) if meta.get("refine") is not None else None
    # Never without its source and its mask; a malformed one is left off.
    outpaint = chunk_outpaint(meta) if mask is not None else None
    crop_anchor = _crop_anchor(
        meta.get("crop_anchor"), source=meta.get("refine"),
        mask=meta.get("mask"), outpaint=meta.get("outpaint"),
    )
    execution = execution_key(meta.get("execution"))
    edit = edit_key(meta.get("edit"))
    validate_edit_record(canonical_edit(meta.get("edit")), {**meta, "width": width, "height": height})
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
                "crop_anchor": crop_anchor,
                "batch": _batch(meta.get("batch")),
                # Bin state lives in SQLite alone, as `favorite` does: a PNG
                # recovered from its own chunk comes back live.
                "binned_at": None,
                "mask": mask,
                "outpaint": outpaint.key() if outpaint is not None else None,
                "upscaler": _upscaler(
                    meta.get("upscaler"), source=meta.get("refine"), model_id=meta.get("model"),
                    mask=mask, outpaint=meta.get("outpaint"),
                ),
                "execution": execution,
                "edit": edit,
                "vram_peak_allocated": _peak(meta.get("vram_peak_allocated")),
                "vram_peak_reserved": _peak(meta.get("vram_peak_reserved")),
            },
        )
        states = _edit_states(conn, [row])
    return _row_to_record(row, edit_missing=states[row["id"]][0], edit_binned=states[row["id"]][1])


_MASK_HASH = re.compile(r"[0-9a-f]{64}")


def _mask(value: Any) -> str | None:
    """A PNG chunk's `mask`, where it is a mask hash: the same shape the
    request's field takes, so a hand-written block cannot put a path, or
    anything else `mask_path` was never meant to look up, on a row."""
    return value if isinstance(value, str) and _MASK_HASH.fullmatch(value) else None


def _upscaler(
    value: Any, *, source: int | None, model_id: str | None,
    mask: str | None = None, outpaint: Any = None,
) -> str | None:
    """A known GAN only beside an ordinary pass of a documented checkpoint.

    PNG recovery accepts older records and handwritten metadata: an unknown
    choice or one without its source cannot become an executable setting.
    """
    return (
        refine.ANIME6B
        if value == refine.ANIME6B and type(source) is int and source > 0 and mask is None and outpaint is None
        and isinstance(model_id, str) and refine.upscaler_supported(model_id)
        else None
    )


def _batch(value: Any) -> int | None:
    """A PNG chunk's `batch`, where it is one: a block written by hand or by
    another tool says what it likes, and a row is not refused for the one
    field that is no part of the reproduction set."""
    return value if type(value) is int and value >= 1 else None


def _peak(value: Any) -> int | None:
    """A peak's bytes, where it is a count of them; anything else - a block
    written by hand, a string - is None, never a refused row, as `_batch`."""
    return value if type(value) is int and value >= 0 else None


def thumbnail_for(image: Image.Image, filename: str) -> None:
    """Write the WEBP in thumbnails/ for an existing PNG, if it has none."""
    png = settings.outputs_dir / filename
    path = _thumb_path(png)
    # Encoded before the reservation, as in save_image; only the move and the
    # rename are reserved, and both checks are repeated under it.
    partial = None
    if not path.exists() and not png.with_suffix(".webp").is_file():
        partial = _encode_thumb(image, path)
    try:
        with gallery_lock.hold(settings.db_path, wait=gallery_lock.SESSION_WAIT):
            # A lost index has no rows for the startup migration; reindex
            # reaches this after recovering each PNG. Move its old cache
            # instead of duplicating it.
            _move_thumbnail(png)
            if partial is not None and not path.exists() and not png.with_suffix(".webp").is_file():
                _place_thumb(partial, path)
                partial = None
    finally:
        if partial is not None:
            _unlink(partial)


def reference_paths(ids: Sequence[int], noun: str = "reference image") -> list:
    """The PNGs behind a request's reference ids, in the order they were given.

    The whole reason references are ids: a path taken from the request and
    joined onto `outputs_dir` is an arbitrary-file read reachable from
    /api/generate, while a row is something this app wrote. Missing ids and
    missing files are the same kind of mistake - the reference cannot be
    encoded - and both are named, so a night refused at submit says which
    image it could not find rather than dying at cell 1. A refine pass's
    source is resolved here too, under its own `noun`. An image in the
    recycle bin is refused until it is restored, before its file is looked
    for: what reaches here is a submission, a queued cell and the CLI alike.
    """
    if not ids:
        return []
    wanted = list(dict.fromkeys(ids))
    marks = ",".join("?" * len(wanted))
    with _session() as conn:
        rows = conn.execute(
            f"SELECT id, filename, binned_at FROM images WHERE id IN ({marks})", tuple(wanted)
        ).fetchall()
    by_id = {row["id"]: settings.outputs_dir / row["filename"] for row in rows}
    unknown = [image_id for image_id in wanted if image_id not in by_id]
    if unknown:
        raise ValueError(
            f"no gallery image with id {', '.join(str(i) for i in unknown)}"
        )
    binned = sorted((row["id"] for row in rows if row["binned_at"] is not None), key=wanted.index)
    if binned:
        it = "it" if len(binned) == 1 else "them"
        raise ValueError(
            f"{noun} {', '.join(str(i) for i in binned)} "
            f"{'is' if len(binned) == 1 else 'are'} in the recycle bin; restore {it} to use {it}"
        )
    gone = [image_id for image_id in wanted if not by_id[image_id].exists()]
    if gone:
        raise ValueError(
            f"the file for {noun} "
            f"{', '.join(str(i) for i in gone)} is missing from the gallery directory"
        )
    return [by_id[image_id] for image_id in ids]


# Where the masks live: a directory of their own inside the images', which
# no gallery glob (`*.png` in the images directory) reaches.
MASKS_DIR = "masks"


def masks_dir() -> Path:
    return settings.outputs_dir / MASKS_DIR


def mask_hash(mask: Image.Image) -> str:
    """The name a mask is stored and asked for by: all 64 hex digits of the
    SHA-256 of its size and its decoded `L` levels.

    Of the pixels, not the file, as `content_hash` is - the encoder's settings
    and the chunks would otherwise split one painting into two names - and the
    whole digest, since a mask is an input looked up by this name, which the
    16-digit `content_hash` is too short to be. The size is hashed with the
    levels because the levels alone do not say it: an all-white 832x1216 and
    an all-white 1216x832 are the same bytes and different masks.
    """
    if mask.mode != "L":
        raise ValueError(f"a mask is one luminance channel, not {mask.mode}")
    digest = hashlib.sha256(f"L {mask.width}x{mask.height}\n".encode("ascii"))
    digest.update(mask.tobytes())
    return digest.hexdigest()


def mask_provenance(mask: str | None, width: int, height: int) -> dict[str, Any] | None:
    """A result's `mask_source` in its PNG chunk: the mask's file, its hash
    and its size - which is the result's own, since an inpaint runs at its
    source's size - or None where the row is no inpaint."""
    if mask is None:
        return None
    return {"filename": f"{mask}.png", "hash": mask, "width": width, "height": height}


def _mask_row(conn: sqlite3.Connection, mask: str) -> sqlite3.Row | None:
    return conn.execute("SELECT hash, width, height FROM masks WHERE hash = ?", (mask,)).fetchone()


def mask_size(mask: str) -> tuple[int, int] | None:
    """The size a stored mask was painted at, or None for a hash this server
    holds no mask under. The table's answer; `mask_path` checks the file."""
    with _session() as conn:
        row = _mask_row(conn, mask)
    return None if row is None else (row["width"], row["height"])


def mask_path(mask: str) -> Path:
    """The stored PNG behind a request's mask hash: the counterpart of
    `reference_paths`, and the one place a hash becomes a file.

    Through the `masks` table, never by joining request text onto a
    directory: the hash is the table's key, and only a row this app wrote
    names a file. An unknown hash and a missing file are both refused in
    words, so a submission says which mask it could not find, and a queued
    cell whose mask went fails in a sentence rather than in Pillow.
    """
    with _session() as conn:
        row = _mask_row(conn, mask)
    if row is None:
        raise ValueError(f"no stored mask {mask[:16]}; paint it again")
    path = masks_dir() / f"{row['hash']}.png"
    if not path.is_file():
        raise ValueError(f"the file for mask {mask[:16]} is missing from the masks directory")
    return path


def store_mask(mask: Image.Image) -> tuple[str, bool]:
    """Keep a mask in canonical form - `L` levels, 0 to 255, white repainted -
    and answer its hash, and whether this call wrote it.

    Content-addressed: the same picture is the same hash and one file, so the
    Inpaint dialog's upload after every stroke costs a lookup once the mask
    is held. Written as `save_image` writes an image: encoded to `.png.part`
    outside the gallery reservation, then the row, then the rename after the
    row commits, so a sweep never sees a file whose row is not there yet. A
    row whose file was lost is given its file again; a file whose row was lost
    (a database replaced under the server) its row.
    """
    mask = mask if mask.mode == "L" else refine.canonical_mask(mask)
    digest = mask_hash(mask)
    ensure_outputs_dir()
    directory = masks_dir()
    directory.mkdir(exist_ok=True)
    path = directory / f"{digest}.png"
    partial = path.with_name(path.name + ".part")
    # One upload at a time in this process: two of the same picture would
    # otherwise encode into the one partial name together.
    with _mask_lock:
        return _store_mask(mask, digest, path, partial)


def _store_mask(mask: Image.Image, digest: str, path: Path, partial: Path) -> tuple[str, bool]:
    encoded = False
    committed = False
    try:
        if not path.is_file():
            # Again here, as before the encode below: a whole `clear` removes
            # the directory under the reservation this has not taken yet.
            path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(partial, format="PNG", compress_level=6)
            encoded = True
        with gallery_lock.hold(settings.db_path, wait=gallery_lock.SESSION_WAIT):
            if not encoded and not path.is_file():
                # Swept while this waited for the reservation: the file the
                # look above found is gone, and a row alone would name none.
                path.parent.mkdir(parents=True, exist_ok=True)
                mask.save(partial, format="PNG", compress_level=6)
                encoded = True
            with _write_lock, _session() as conn:
                now = datetime.now(timezone.utc).isoformat()
                created = conn.execute(
                    "INSERT OR IGNORE INTO masks (hash, width, height, created_at) VALUES (?, ?, ?, ?)",
                    (digest, mask.width, mask.height, now),
                ).rowcount == 1
                if not created:
                    # The upload is the mask being used again - a dialog
                    # reopened on an old one - so the sweep's settle window
                    # (`gallery_delete.sweep_masks`) starts over from now.
                    conn.execute("UPDATE masks SET created_at = ? WHERE hash = ?", (now, digest))
            committed = True
            if encoded and not path.is_file():
                _rename_with_retry(partial, path)
                created = True
    except BaseException:
        if not committed:
            _unlink(partial)
        raise
    finally:
        # Another upload of the same picture placed the file first; a partial
        # whose rename failed is the only copy, which the sweep finishes.
        if encoded and committed and partial.exists() and path.is_file():
            _unlink(partial)
    return digest, created


def outpaint_chunk(spec: OutpaintSpec | None) -> dict[str, int] | None:
    """An outpaint's `outpaint` key in a result's PNG chunk: its padding and
    band (the fill rides beside it as `outpaint_fill`), or None."""
    if spec is None:
        return None
    return {"left": spec.left, "top": spec.top, "right": spec.right, "bottom": spec.bottom, "band": spec.band}


def outpaint_parameters(spec: OutpaintSpec) -> str:
    """The A1111 line's outpaint, slashes where A1111 would split on a comma:
    the one copy of the spec an older build's import leaves in place."""
    return (
        f", Outpaint: {spec.left}/{spec.top}/{spec.right}/{spec.bottom}, "
        f"Outpaint band: {spec.band}, Outpaint fill: {spec.fill}"
    )


def _canonical_outpaint(text: str | None) -> str | None:
    """A column's outpaint key in `OutpaintSpec.key()`'s one spelling - a
    row an older import stored as given (`0400`) keys as its re-render
    does - or the text itself where it is no key."""
    if not text:
        return text
    try:
        return OutpaintSpec.parse_key(text).key()
    except ValueError:
        return text


def chunk_outpaint(meta: dict[str, Any]) -> OutpaintSpec | None:
    """A PNG chunk's `outpaint` and `outpaint_fill` as the stored spec, or
    None where it carries none or a malformed one. Any well-formed fill is
    kept, known to this build or not: the submit refuses an unknown one by
    name, so the row still shows as what it is."""
    value, fill = meta.get("outpaint"), meta.get("outpaint_fill")
    if not isinstance(value, dict) or not isinstance(fill, str):
        return None
    sides: list[int] = []
    for name in ("left", "top", "right", "bottom", "band"):
        side = value.get(name)
        if type(side) is not int:
            return None
        sides.append(side)
    try:
        spec = OutpaintSpec(fill=fill, left=sides[0], top=sides[1], right=sides[2], bottom=sides[3], band=sides[4])
    except ValueError:
        return None
    return spec if spec.wellformed() else None


def malformed_outpaint(meta: dict[str, Any]) -> bool:
    """Whether a chunk names an outpaint that `chunk_outpaint` cannot read."""
    return meta.get("outpaint") is not None and chunk_outpaint(meta) is None


_PARAMETERS_OUTPAINT = re.compile(
    r"Outpaint: (\d{1,6})/(\d{1,6})/(\d{1,6})/(\d{1,6}), Outpaint band: (\d{1,6}), "
    r"Outpaint fill: ([a-z0-9][a-z0-9-]{0,31})(?:,|$|\s)"
)


def outpaint_mask_hash(width: int, height: int, spec: OutpaintSpec) -> str | None:
    """The hash of the mask an outpaint of this canvas and spec builds, or
    None where the canvas cannot hold it."""
    from . import outpaint

    try:
        return mask_hash(outpaint.mask(width, height, spec))
    except ValueError:
        return None


def recovered_outpaint(
    parameters: Any, width: int, height: int, mask: str | None, source_size: tuple[int, int] | None,
) -> OutpaintSpec | None:
    """An outpaint an older build's import stripped, from the A1111 line it
    left in place - exact or nothing: only where the row has a mask, the
    canvas is the source's size plus the parsed padding (and so not the
    source's own), and the mask this build builds for it hashes to the
    row's."""
    if not isinstance(parameters, str) or mask is None or source_size is None:
        return None
    if (width, height) == tuple(source_size):
        return None
    match = _PARAMETERS_OUTPAINT.search(parameters)
    if match is None:
        return None
    left, top, right, bottom, band = (int(value) for value in match.groups()[:5])
    try:
        spec = OutpaintSpec(fill=match.group(6), left=left, top=top, right=right, bottom=bottom, band=band)
    except ValueError:
        return None
    if not (spec.left or spec.top or spec.right or spec.bottom):
        return None
    if (width - left - right, height - top - bottom) != tuple(source_size):
        return None
    return spec if outpaint_mask_hash(width, height, spec) == mask else None


def rebuilt_outpaint_mask(width: int, height: int, spec: OutpaintSpec, digest: str) -> Image.Image | None:
    """An outpaint's mask built again from its spec and the row's size,
    where it hashes to the row's `digest`; None where it does not."""
    from . import outpaint

    try:
        built = outpaint.mask(width, height, spec)
    except ValueError:
        return None
    return built if mask_hash(built) == digest else None


def stamp_outpaints(cells: Sequence[Any]) -> list[Any]:
    """`cells` with each outpaint's mask built and stored: one build per
    distinct canvas and spec (normally one per job), kept by `store_mask`
    (idempotent, content addressed; it takes the gallery reservation and
    restarts the settle window), and each outpaint cell answered with
    `mask` set to its hash. The first step of `JobManager.submit_cells`, and
    the CLI's before its load, so every later step - the job's cells, the
    holds, the cell's check as it begins, the row - sees the stamped cell.
    `model_copy`, which does not validate: a stamped request holds a mask
    and an outpaint together, which no client may send. Blocking: encodes
    and writes. Raises what `store_mask` raises (`OSError`, `GalleryBusy`)."""
    import dataclasses

    from . import outpaint

    hashes: dict[tuple[int, int, str], str] = {}
    stamped = []
    for cell in cells:
        request = cell.request
        if request.outpaint is None or request.refine is None:
            stamped.append(cell)
            continue
        key = (request.width, request.height, request.outpaint.key())
        if key not in hashes:
            hashes[key], _ = store_mask(outpaint.mask(request.width, request.height, request.outpaint))
        stamped.append(dataclasses.replace(cell, request=request.model_copy(update={"mask": hashes[key]})))
    return stamped


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def touch_mask(mask: str) -> Path:
    """Restart a stored mask's settle window, as a re-upload of it does, and
    answer its file: an Inpaint dialog that reused it or sat open on it
    keeps it from a sweep (`gallery_delete.sweep_masks`) without sending
    the picture again. `ValueError` in `mask_path`'s words where it went."""
    path = mask_path(mask)
    with _write_lock, _session() as conn:
        now = datetime.now(timezone.utc).isoformat()
        if conn.execute("UPDATE masks SET created_at = ? WHERE hash = ?", (now, mask)).rowcount != 1:
            raise ValueError(f"no stored mask {mask[:16]}; paint it again")
    return path


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def hold_masks(job_id: str, masks: Iterable[str]) -> list[str]:
    """Hold `masks` for job `job_id` until `release_masks`: no sweep in any
    process removes a held mask. Answers the ones this server no longer has,
    which are not held - gone between the submit's check and here.

    Under `BEGIN IMMEDIATE`, as the sweep's own look is: either the hold is
    in before a sweep looks, or the sweep's delete is in before this one
    looks, and then the mask is named as gone rather than queued unheld.
    """
    wanted = sorted(set(masks))
    if not wanted:
        return []
    with _write_lock, _session() as conn:
        conn.execute("BEGIN IMMEDIATE")
        gone = [digest for digest in wanted if _mask_row(conn, digest) is None]
        conn.executemany(
            "INSERT OR IGNORE INTO mask_holds (hash, job_id) VALUES (?, ?)",
            [(digest, job_id) for digest in wanted if digest not in gone],
        )
    return gone


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def release_masks(job_id: str | None = None) -> None:
    """End job `job_id`'s holds (`hold_masks`), or every hold where it is
    None: at a server's start, since no job survives a restart."""
    with _write_lock, _session() as conn:
        if job_id is None:
            conn.execute("DELETE FROM mask_holds")
        else:
            conn.execute("DELETE FROM mask_holds WHERE job_id = ?", (job_id,))


def mask_png(mask: str) -> bytes | None:
    """The stored mask's own PNG bytes, for the result's embedded copy, or
    None where it cannot be read - the copy is the backstop, never a reason
    to lose the image it would have gone into."""
    try:
        return mask_path(mask).read_bytes()
    except (ValueError, OSError, sqlite3.Error):
        log.warning("mask %s could not be embedded in its result", mask[:16], exc_info=True)
        return None


def missing_files(records: Sequence[ImageRecord]) -> set[int]:
    """The ids of those `records` whose PNG is not in the gallery directory:
    a row whose file went - deleted by hand, a disk restored without it -
    which Mass refine reports per image rather than at its cell."""
    return {record.id for record in records if not (settings.outputs_dir / record.filename).exists()}


def get_image(image_id: int, include_binned: bool = False) -> ImageRecord | None:
    """One live image's record, or None; with `include_binned`, any row's."""
    live = "" if include_binned else f" AND {_LIVE}"
    with _session() as conn:
        row = conn.execute(f"SELECT * FROM images WHERE id = ?{live}", (image_id,)).fetchone()
        missing, binned = _missing_sources(conn, [row]) if row else (set(), set())
        states = _edit_states(conn, [row]) if row else {}
    return _row_to_record(row, row["refine"] in missing, row["refine"] in binned, *states[row["id"]]) if row else None


def get_images(ids: Sequence[int], include_binned: bool = False) -> list[ImageRecord]:
    """The records of those of `ids` the gallery holds, oldest first; an id
    it does not hold is simply absent, and so is one in the recycle bin
    unless `include_binned`. In chunks, under SQLite's variable limit, for a
    selection of up to `max_grid_images` cards."""
    wanted = sorted(set(ids))
    live = "" if include_binned else f" AND {_LIVE}"
    rows: list[sqlite3.Row] = []
    with _session() as conn:
        for at in range(0, len(wanted), 500):
            chunk = wanted[at:at + 500]
            rows.extend(conn.execute(
                f"SELECT * FROM images WHERE id IN ({','.join('?' * len(chunk))}){live} ORDER BY id",
                chunk,
            ).fetchall())
        missing, binned = _missing_sources(conn, rows) if rows else (set(), set())
        states = _edit_states(conn, rows)
    return _records(rows, missing, binned, states)


def lineage_images(run_ids: Sequence[str], job_ids: Sequence[str]) -> list[ImageRecord]:
    """Every image of these runs, and of these jobs where the image has no
    run id, oldest first: what More seeds reads a batch's highest seed and
    its Batch from, picked or not. The recycle bin's included: a binned seed
    was made, so the next ones follow on past it and restoring it never
    duplicates one. In chunks, under SQLite's variable limit."""
    rows: list[sqlite3.Row] = []
    with _session() as conn:
        for column, names, extra in (("run_id", run_ids, ""), ("job_id", job_ids, " AND run_id IS NULL")):
            wanted = sorted(set(names))
            for at in range(0, len(wanted), 500):
                chunk = wanted[at:at + 500]
                rows.extend(conn.execute(
                    f"SELECT * FROM images WHERE {column} IN ({','.join('?' * len(chunk))}){extra}",
                    chunk,
                ).fetchall())
        missing, binned = _missing_sources(conn, rows)
        states = _edit_states(conn, rows)
    return _records(sorted(rows, key=lambda row: row["id"]), missing, binned, states)


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def retain_run_job(definition: dict[str, Any], cells: Sequence[Cell]) -> None:
    """Keep a job's resolved, pre-outpaint-stamp cells before admission.

    The raw requests are intentional: applied settings are compared when a
    run is finished, and an outpaint's mask must be built again at submit.
    Compression happens before the write lock; every input is JSON-native.
    A failed write raises to the submitter, which releases any mask holds
    and never admits the job. This table is local recovery state, not PNG
    metadata or part of an exported bundle.
    """
    encoded = gzip.compress(json.dumps([
        {"request": cell.request.model_dump(mode="json"), "seed": cell.seed,
         "label": cell.label, "variables": cell.variables}
        for cell in cells
    ], ensure_ascii=True, separators=(",", ":")).encode("utf-8"), mtime=0)
    with _write_lock, _session() as conn:
        conn.execute(
            """INSERT INTO run_jobs
               (job_id, run_id, created_at, origin, model_id, quantization,
                width, height, batch, comparison, metadata, cells)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (definition["job_id"], definition["run_id"], definition["created_at"],
             definition["origin"], definition["model_id"], definition["quantization"],
             definition["width"], definition["height"], definition["batch"],
             int(definition["comparison"]), json.dumps(definition["metadata"], ensure_ascii=True), encoded),
        )


def retained_run_metadata(job_ids: Sequence[str]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each retained job's (run id, metadata), by job id, without decoding
    its cells: what names a Then refine pass run's base (its records' job is
    the base job, whose metadata carries the pass run id). A job not
    retained, or whose metadata is not an object, is left out."""
    ids = list(dict.fromkeys(job_ids))
    if not ids:
        return {}
    with _session() as conn:
        rows = conn.execute(
            f"SELECT job_id, run_id, metadata FROM run_jobs WHERE job_id IN ({','.join('?' * len(ids))})", ids,
        ).fetchall()
    found = {}
    for row in rows:
        try:
            metadata = json.loads(row["metadata"])
        except (TypeError, ValueError):
            continue
        if isinstance(metadata, dict):
            found[row["job_id"]] = (row["run_id"], metadata)
    return found


def retained_run_jobs(run_id: str | None = None) -> list[dict[str, Any]]:
    """Retained jobs in creation order, with raw cells and metadata decoded.

    A NULL outcome means the process ended before its terminal write, or
    that write failed. It never means the job is still running: the live
    JobManager alone answers that. A malformed/obsolete request is a worded
    ValueError for the recovery API rather than an unvalidated pipeline call.
    """
    from .grid import Cell
    from .schemas import GenerateRequest
    from .seeds import MAX_SEED

    where = " WHERE run_id = ?" if run_id is not None else ""
    with _session() as conn:
        rows = conn.execute(
            f"SELECT * FROM run_jobs{where} ORDER BY created_at, job_id",
            (run_id,) if run_id is not None else (),
        ).fetchall()
    jobs = []
    for row in rows:
        item = dict(row)
        try:
            raw = json.loads(gzip.decompress(item["cells"]))
            if not isinstance(raw, list):
                raise ValueError("cells must be a list")
            for cell in raw:
                if (not isinstance(cell, dict) or type(cell.get("seed")) is not int
                        or not 0 <= cell["seed"] <= MAX_SEED
                        or cell.get("label") is not None and not isinstance(cell["label"], str)
                        or cell.get("variables") is not None and (
                            not isinstance(cell["variables"], dict)
                            or not all(isinstance(key, str) and isinstance(value, str)
                                       for key, value in cell["variables"].items()))):
                    raise ValueError("invalid retained cell")
            item["cells"] = [Cell(
                request=GenerateRequest.model_validate(cell["request"]), seed=cell["seed"],
                label=cell["label"], variables=cell["variables"],
            ) for cell in raw]
            item["metadata"] = json.loads(item["metadata"])
            item["outcome"] = json.loads(item["outcome"]) if item["outcome"] is not None else None
            metadata, outcome = item["metadata"], item["outcome"]
            if not isinstance(metadata, dict) or outcome is not None and not isinstance(outcome, dict):
                raise ValueError("invalid retained metadata or outcome")
            budget = metadata.get("budget_minutes")
            if budget is not None and (type(budget) not in (int, float) or not 0 < budget <= 7 * 24 * 60):
                raise ValueError("invalid retained budget")
            if metadata.get("prompt_template") is not None and not isinstance(metadata["prompt_template"], str):
                raise ValueError("invalid retained prompt template")
            if outcome is not None:
                indices = outcome.get("completed_indices", [])
                if not isinstance(indices, list) or any(type(index) is not int for index in indices):
                    raise ValueError("invalid retained completion indices")
                if outcome.get("finished_at") is not None:
                    if type(outcome["finished_at"]) not in (int, float):
                        raise ValueError("invalid retained finish time")
                    datetime.fromtimestamp(outcome["finished_at"], timezone.utc)
        except (ValueError, TypeError, KeyError, OSError, EOFError, OverflowError) as exc:
            raise ValueError(f"the saved cells of job {row['job_id']} cannot be read by this build") from exc
        item["comparison"] = bool(item["comparison"])
        jobs.append(item)
    return jobs


def finish_run_job(job_id: str, outcome: dict[str, Any], *, database: Path | None = None) -> None:
    """Record one terminal outcome, without recreating a cleared job.

    Called on a daemon thread after `_finish` has made its terminal
    transition. Interruption or a disk error may leave outcome NULL; Finish
    still compares the cells with what the gallery holds.
    """
    # The caller captures the path before starting its daemon. A late thread
    # must not follow changed settings to another gallery (or recreate a
    # database that has gone). This writer needs no image schema migration.
    path = (database or settings.db_path).resolve()
    with gallery_lock.hold(path, create=False, wait=gallery_lock.SESSION_WAIT), _write_lock:
        conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=15)
        try:
            with conn:
                if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_jobs'"
                ).fetchone():
                    conn.execute(
                        "UPDATE run_jobs SET outcome = ? WHERE job_id = ? AND outcome IS NULL",
                        (json.dumps(outcome, ensure_ascii=True), job_id),
                    )
        finally:
            conn.close()


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def set_favorite(image_id: int, favorite: bool) -> ImageRecord | None:
    """Star or unstar a live image; None for one deleted or in the recycle bin,
    which keeps the flag it was binned with until it is restored."""
    with _write_lock, _session() as conn:
        conn.execute(
            f"UPDATE images SET favorite = ? WHERE id = ? AND {_LIVE}", (int(favorite), image_id)
        )
    return get_image(image_id)


@gallery_lock.protected(lambda: settings.db_path, waits=True)
def set_favorites(image_ids: Sequence[int], favorite: bool) -> dict[str, list[int]]:
    """Star or unstar several live images at once (the gallery's picked
    cards): the ids changed, those already so, those in the recycle bin -
    left with the flag they were binned with, as `set_favorite` leaves one -
    and those the gallery does not hold. The lookup runs under the write's
    lock, so a second press answers already, never changed twice."""
    wanted = sorted(set(image_ids))
    changed: list[int] = []
    already: list[int] = []
    binned: list[int] = []
    with _write_lock, _session() as conn:
        for at in range(0, len(wanted), 500):
            chunk = wanted[at:at + 500]
            marks = ",".join("?" * len(chunk))
            changed.extend(row["id"] for row in conn.execute(
                f"UPDATE images SET favorite = ? WHERE id IN ({marks}) AND {_LIVE}"
                " AND favorite != ? RETURNING id",
                (int(favorite), *chunk, int(favorite)),
            ).fetchall())
            moved = set(changed)
            for row in conn.execute(f"SELECT id, binned_at FROM images WHERE id IN ({marks})", chunk):
                if row["id"] in moved:
                    continue
                (already if row["binned_at"] is None else binned).append(row["id"])
    held = set(changed) | set(already) | set(binned)
    return {
        "changed": sorted(changed), "already": sorted(already), "binned": sorted(binned),
        "missing": [image_id for image_id in wanted if image_id not in held],
    }


def _unlink(path) -> bool:
    """Delete a file, tolerating both "already gone" and "someone has it open"."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # WinError 32, most often: another process is streaming it
        return False
    return True


def _tracked_files():
    with _session() as conn:
        return conn.execute("SELECT filename FROM images").fetchall()


def _tracked_masks() -> set[str]:
    """Every mask hash the `masks` table holds."""
    with _session() as conn:
        return {row["hash"] for row in conn.execute("SELECT hash FROM masks")}


# The names `gallery_reports` and `gallery_delete` took out of this module,
# still reachable as `storage.<name>` for the callers and tests that use them.
# PEP 562 rather than an import at the bottom: both modules import this one at
# their top, so a caller that imported one of them first would find it half
# initialised. Each name resolves to the one object its module defines - an
# `except storage.SelectionChanged` catches what `gallery_delete` raises.
_MOVED = {
    **dict.fromkeys(
        ("human_bytes", "step_cost_ms", "refine_cost_ms", "image_bytes",
         "_GROUPS", "keep_rates", "_RUN_AXES", "runs"),
        "gallery_reports",
    ),
    **dict.fromkeys(
        ("_UNKEPT", "matching_preview", "SelectionChanged", "delete_matching",
         "delete_image", "bin_images", "restore_images", "bin_summary", "empty_bin",
         "summarise", "clear_history", "vacuum"),
        "gallery_delete",
    ),
}


def __getattr__(name: str):
    if name in _MOVED:
        return getattr(importlib.import_module(f"{__package__}.{_MOVED[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
