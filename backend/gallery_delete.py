"""Taking images out of the gallery: the keep-favourites predicates, the
scoped delete and its preview, the per-image permanent delete, the recycle bin
(bin, restore, summary, empty), and the whole-gallery clear with its summary
and VACUUM.

Like `bundles`, everything storage owns - `settings`, the one `_write_lock`,
`_session`, the file helpers - is read through `storage` at call time, so the
tests' replacement of `storage.settings` reaches these writes too, and a bin
and a save still serialise on the same lock.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from . import gallery_lock, storage
from .search import strict_scope


# The images a starred one was made from: a refine pass's source and a klein
# render's references, by id. They are part of the starred image's
# reproduction set, and a sweep that took them left a kept pass that could
# never be made again - with nothing before the click saying it depended on
# them. One level: the source of a kept source is not the starred image's to
# need. Neither arm can yield a NULL, which would turn every `NOT id IN` over
# the set into NULL and keep - or, counted, miscount - every row.
# Only a live star protects: a starred image in the recycle bin keeps its
# flag, so a Restore brings the star back, but is itself unkept until then.
_STARRED_SOURCE = (
    "id IN (SELECT refine FROM images WHERE favorite = 1 AND binned_at IS NULL"
    " AND refine IS NOT NULL"
    " UNION SELECT ref.value FROM images AS starred, json_each(starred.reference) AS ref"
    " WHERE starred.favorite = 1 AND starred.binned_at IS NULL"
    " AND starred.reference IS NOT NULL AND ref.type = 'integer')"
)
# What `keep_favorites` leaves out of a delete: the live stars and their
# sources. The one predicate `delete_matching`, `clear_history` and the
# previews' counts share, so the number a dialog asks to have typed back is
# the number the delete then finds under its lock.
# The stars `keep_favorites` keeps, and the rest of what it might delete.
_LIVE_STAR = "favorite = 1 AND binned_at IS NULL"
_NOT_LIVE_STAR = "(favorite = 0 OR binned_at IS NOT NULL)"
_UNKEPT = f"{_NOT_LIVE_STAR} AND NOT ({_STARRED_SOURCE})"

# The masks the server's queued and running cells will read, asked as a sweep
# runs (`JobManager.masks_in_flight`): a queued cell's mask is in no row yet.
# The CLI has no jobs of its own and passes none.
InFlight = Callable[[], set[str]]


def matching_preview(
    search: str | None,
    model_id: str | None = None,
    newer_than: int | None = None,
    live_only: bool = False,
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

    `live_only` leaves the recycle bin out, as the browser's Delete unstarred
    does, which moves what it takes there; the CLI's `--where` reaches the
    bin too, and `binned` is how many of `records` are in it. `favorites` is
    the live stars: a binned star keeps nothing.
    """
    filters, params = strict_scope(search)
    if model_id:
        filters.append("model_id = ?")
        params.append(model_id)
    if live_only:
        filters.append(storage._LIVE)

    def where(*extra: str) -> str:
        parts = [*filters, *extra]
        return f"WHERE {' AND '.join(parts)}" if parts else ""

    clause = where()
    with storage._session() as conn:
        records, favorites, sources_kept, binned = conn.execute(
            f"""SELECT COUNT(*), COALESCE(SUM({_LIVE_STAR}), 0),
                       COALESCE(SUM({_NOT_LIVE_STAR} AND {_STARRED_SOURCE}), 0),
                       COALESCE(SUM(binned_at IS NOT NULL), 0)
                FROM images {clause}""",
            params,
        ).fetchone()
        sources = [
            row[0]
            for row in conn.execute(
                f"SELECT id FROM images {where(_NOT_LIVE_STAR, _STARRED_SOURCE)} ORDER BY id LIMIT 6",
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
            "binned": binned,
            "runs": runs,
            "first": first,
            "last": last,
            "newer": newer,
            "prompts": distinct("prompt", 4),
            "jobs": [j for j in distinct("job_id", 6) if j],
            "models": distinct("model_id", 6),
        }


class SelectionChanged(Exception):
    """A scoped delete was confirmed for one count and finds another - or,
    for Empty recycle bin's pin, the same count of other images."""

    def __init__(self, found: int, expected: int):
        super().__init__(
            f"{found} images match now, not the {expected} confirmed" if found != expected
            else f"other images match now than the {expected} confirmed"
        )
        self.found = found
        self.expected = expected


@gallery_lock.protected(lambda: storage.settings.db_path)
def delete_matching(
    search: str | None,
    keep_favorites: bool = False,
    expected: int | None = None,
    model_id: str | None = None,
    to_bin: bool = False,
    in_flight: InFlight | None = None,
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

    `to_bin` moves the live rows in scope to the recycle bin instead - the
    browser's Delete unstarred, counted by `matching_preview(live_only=True)`
    - and touches no file, so `files` is 0. Without it the delete is
    permanent and reaches binned rows too, as `clear --where` does, and then
    sweeps the inpaint masks nothing names any more (`sweep_masks`, with
    `in_flight`), which the bin never does.
    """
    filters, params = strict_scope(search)
    if keep_favorites:
        filters.append(_UNKEPT)
    if model_id:
        filters.append("model_id = ?")
        params.append(model_id)
    if not filters:
        raise ValueError("a filter is required; the whole gallery is `clear` without --where")
    if to_bin:
        filters.append(storage._LIVE)
    where = " AND ".join(filters)
    with storage._write_lock, storage._session() as conn:
        if expected is not None:
            (found,) = conn.execute(f"SELECT COUNT(*) FROM images WHERE {where}", params).fetchone()
            if found != expected:
                raise SelectionChanged(found, expected)
        if to_bin:
            moved = conn.execute(
                f"UPDATE images SET binned_at = ? WHERE {where} RETURNING id", (_now(), *params)
            ).fetchall()
            return {"records": len(moved), "files": 0, "ids": sorted(row["id"] for row in moved)}
        doomed = conn.execute(
            f"DELETE FROM images WHERE {where} RETURNING id, filename", params
        ).fetchall()
    removed = 0
    for row in doomed:
        path = storage.settings.outputs_dir / row["filename"]
        if path.exists() and storage._unlink(path):
            removed += 1
        storage._delete_thumbnails(path)
    masks = sweep_masks(in_flight)
    return {"records": len(doomed), "files": removed, "ids": [row["id"] for row in doomed], "masks": masks}


@gallery_lock.protected(lambda: storage.settings.db_path, waits=True)
def delete_image(image_id: int, in_flight: InFlight | None = None) -> bool:
    """Delete one image and its files for good, live or in the recycle bin."""
    # Looked up and deleted under the one lock: two deletes of the same image
    # (a double-click, a second tab) both found the row before either removed
    # it, and both answered as though they had deleted it.
    with storage._write_lock, storage._session() as conn:
        row = conn.execute("SELECT filename FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    # The row is already gone, so a file someone else has open must not turn a
    # delete that did happen into a 500. clear_history's orphan sweep collects
    # whatever gets left behind.
    storage._unlink(storage.settings.outputs_dir / row["filename"])
    storage._delete_thumbnails(storage.settings.outputs_dir / row["filename"])
    sweep_masks(in_flight)
    return True


def _now() -> str:
    """The time a row is binned at, written as `created_at` is."""
    return datetime.now(timezone.utc).isoformat()


def _settled(created_at: Any, cutoff: float) -> bool:
    """Whether a `masks` row was written (or last uploaded) before `cutoff`,
    Unix seconds. A stamp that does not parse is an old one: the row is
    still spared while anything names it."""
    try:
        return datetime.fromisoformat(created_at).timestamp() < cutoff
    except (TypeError, ValueError):
        return True


def sweep_masks(in_flight: InFlight | None = None, *, settle: bool = True, orphans: bool = False) -> int:
    """Remove the inpaint masks nothing needs any more: their `masks` rows
    and their files. Answers how many went.

    A mask is needed while any row names it, live or binned - a binned
    inpaint is still what was made, and a Restore must find its mask - and
    while a queued or running cell will read it (`in_flight`, asked here,
    under the gallery reservation every caller already holds). It is spared
    too while younger than `PARTIAL_SETTLE_SECONDS`: the Inpaint dialog
    uploads a mask before anything names it, and its re-upload restarts the
    clock (`storage.store_mask`). `settle=False` is the whole `clear`, which
    takes every mask no kept row names, whatever its age.

    Rows first, under `BEGIN IMMEDIATE` so no save names a mask between the
    look and the delete; files after the commit, as images are removed. A
    file that will not go is left for the next sweep, its row already gone -
    `orphans` collects such files, and any in `masks/` with no row, once
    they too are older than the window (the whole `clear`'s orphan sweep).
    """
    cutoff = time.time() - storage.PARTIAL_SETTLE_SECONDS
    held = set(in_flight()) if in_flight is not None else set()
    with storage._write_lock, storage._session() as conn:
        conn.execute("BEGIN IMMEDIATE")
        held |= {row[0] for row in conn.execute("SELECT DISTINCT mask FROM images WHERE mask IS NOT NULL")}
        doomed = [
            row["hash"]
            for row in conn.execute("SELECT hash, created_at FROM masks")
            if row["hash"] not in held and (not settle or _settled(row["created_at"], cutoff))
        ]
        conn.executemany("DELETE FROM masks WHERE hash = ?", [(digest,) for digest in doomed])
        kept = {row[0] for row in conn.execute("SELECT hash FROM masks")}
    directory = storage.masks_dir()
    for digest in doomed:
        storage._unlink(directory / f"{digest}.png")
    if orphans and directory.is_dir():
        for path in directory.glob("*.png"):
            try:
                old = path.stat().st_mtime < cutoff
            except OSError:
                continue
            if path.stem not in kept and path.stem not in held and old:
                storage._unlink(path)
    return len(doomed)


def _toggle_bin(ids: Sequence[int], *, to_bin: bool) -> tuple[list[int], list[int], list[int]]:
    """Move live rows into the recycle bin or binned rows out of it: the ids
    changed, those already on the asked side, and those the gallery does not
    hold.

    The write and the lookup of what was already there run under the one
    lock, as `delete_image` does: a double press answers changed once and
    already once, never changed twice. Files and thumbnails stay where they
    are - a bundle's inventory takes only images/ and images/thumbnails/, and
    a restore is then one UPDATE.
    """
    wanted = sorted(set(ids))
    changed: list[int] = []
    already: list[int] = []
    if to_bin:
        update, stays = "binned_at = ?", "binned_at IS NULL"
        there = "binned_at IS NOT NULL"
    else:
        update, stays = "binned_at = NULL", "binned_at IS NOT NULL"
        there = storage._LIVE
    stamp = (_now(),) if to_bin else ()
    with storage._write_lock, storage._session() as conn:
        for at in range(0, len(wanted), 500):
            chunk = wanted[at:at + 500]
            marks = ",".join("?" * len(chunk))
            changed.extend(row["id"] for row in conn.execute(
                f"UPDATE images SET {update} WHERE id IN ({marks}) AND {stays} RETURNING id",
                (*stamp, *chunk),
            ).fetchall())
            # After the UPDATE, so the rows it just changed are not counted.
            moved = set(changed)
            already.extend(
                row["id"]
                for row in conn.execute(
                    f"SELECT id FROM images WHERE id IN ({marks}) AND {there}", chunk
                )
                if row["id"] not in moved
            )
    held = set(changed) | set(already)
    return sorted(changed), sorted(already), [i for i in wanted if i not in held]


@gallery_lock.protected(lambda: storage.settings.db_path, waits=True)
def bin_images(ids: Sequence[int]) -> dict[str, list[int]]:
    """Move these images to the recycle bin: the page's delete, undone by
    `restore_images`. A starred one keeps its flag, so a Restore brings the
    star back; while binned it is unkept and protects nothing."""
    binned, already, missing = _toggle_bin(ids, to_bin=True)
    return {"binned": binned, "already": already, "missing": missing}


@gallery_lock.protected(lambda: storage.settings.db_path, waits=True)
def restore_images(ids: Sequence[int]) -> dict[str, list[int]]:
    """Bring these images back out of the recycle bin, usable at once."""
    restored, not_binned, missing = _toggle_bin(ids, to_bin=False)
    return {"restored": restored, "not_binned": not_binned, "missing": missing}


def bin_summary() -> dict[str, Any]:
    """How much is in the recycle bin, and which of it a live image needs.

    `bytes` is the binned PNGs and their thumbnails as they are on disk now,
    a file gone or locked counted as nothing. `sources_in_use` is the binned
    ids a live image names as its refine source or a reference - the shape
    of `_STARRED_SOURCE`'s arms - which Empty recycle bin's confirmation
    names: those images can never be made again once it is emptied.
    """
    with storage._session() as conn:
        rows = conn.execute("SELECT id, filename FROM images WHERE binned_at IS NOT NULL").fetchall()
        names = [row["filename"] for row in rows]
        in_use = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM images WHERE binned_at IS NOT NULL AND id IN ("
                " SELECT refine FROM images WHERE binned_at IS NULL AND refine IS NOT NULL"
                " UNION SELECT ref.value FROM images AS live, json_each(live.reference) AS ref"
                " WHERE live.binned_at IS NULL AND live.reference IS NOT NULL"
                " AND ref.type = 'integer') ORDER BY id"
            )
        ]
    size = 0
    for name in names:
        png = storage.settings.outputs_dir / name
        for path in (png, storage._thumb_path(png), png.with_suffix(".webp")):
            try:
                size += path.stat().st_size
            except OSError:
                continue
    return {"records": len(names), "bytes": size, "sources_in_use": in_use,
            "pin": bin_pin(row["id"] for row in rows)}


def bin_pin(ids) -> str:
    """Which images are in the bin, in 16 hex digits: the set, not its size.

    A count alone let a Restore in another tab and a bin of a different image
    keep it equal, and Empty recycle bin then deleted for good an image its
    confirmation never covered.
    """
    joined = ",".join(str(i) for i in sorted(ids))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()[:16]


@gallery_lock.protected(lambda: storage.settings.db_path)
def empty_bin(
    expected: int | None = None, pin: str | None = None, in_flight: InFlight | None = None
) -> dict[str, Any]:
    """Delete every image in the recycle bin, rows and files, for good.

    Fails fast on a busy gallery like `delete_matching`, and like it pins
    what the confirmation showed: `expected`, its count, and `pin`, its ids
    (`bin_summary`'s `bin_pin`). A bin another tab changed while the dialog
    was open raises `SelectionChanged` and deletes nothing - with the count
    unchanged too, as a Restore and a bin of another image leave it. The
    rows go in one `DELETE ... RETURNING`, the files after the commit,
    counted as `delete_matching` counts them, and then the masks no row
    names any more (`sweep_masks`). No orphan sweep and no vacuum.
    """
    with storage._write_lock, storage._session() as conn:
        if expected is not None or pin is not None:
            ids = [row[0] for row in conn.execute("SELECT id FROM images WHERE binned_at IS NOT NULL")]
            if expected is not None and len(ids) != expected:
                raise SelectionChanged(len(ids), expected)
            if pin is not None and bin_pin(ids) != pin:
                raise SelectionChanged(len(ids), len(ids) if expected is None else expected)
        doomed = conn.execute(
            "DELETE FROM images WHERE binned_at IS NOT NULL RETURNING id, filename"
        ).fetchall()
    removed = 0
    for row in doomed:
        path = storage.settings.outputs_dir / row["filename"]
        if path.exists() and storage._unlink(path):
            removed += 1
        storage._delete_thumbnails(path)
    masks = sweep_masks(in_flight)
    return {"records": len(doomed), "files": removed, "ids": sorted(row["id"] for row in doomed),
            "masks": masks}


def summarise() -> dict[str, int]:
    """Counts for the clear-history preview.

    An absent database reports as empty rather than being created on the spot:
    `clear.py --dry-run` promises it changed nothing, and _connect() writes the
    whole schema into any file it opens.
    """
    total = favorites = sources_kept = binned = 0
    tracked: set[str] = set()
    # The directory first, the rows second: `save_image` commits before it
    # renames, so a PNG the scan can see already has a row by the time the rows
    # are read. The other order judged an image that landed in between as an
    # orphan.
    files = list(storage.settings.outputs_dir.glob("*.png"))
    thumbs = storage._thumb_files()
    # Inpaint masks, which a whole clear takes with the images they cover.
    masks = list(storage.masks_dir().glob("*.png"))
    if storage.settings.db_path.exists():
        with storage._session() as conn:
            total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            # The live stars: a binned one is unkept, as `_UNKEPT` reads it.
            favorites = conn.execute(
                f"SELECT COUNT(*) FROM images WHERE {_LIVE_STAR}"
            ).fetchone()[0]
            # What `--keep-favorites` keeps besides them, as `_UNKEPT` reads it.
            sources_kept = conn.execute(
                f"SELECT COUNT(*) FROM images WHERE {_NOT_LIVE_STAR} AND {_STARRED_SOURCE}"
            ).fetchone()[0]
            binned = conn.execute(
                "SELECT COUNT(*) FROM images WHERE binned_at IS NOT NULL"
            ).fetchone()[0]
        tracked = {r["filename"] for r in storage._tracked_files()}
    # What a sweep would finish (a committed image's interrupted save) and
    # what it would discard; with no database, nothing is committed.
    plan = storage._settle_plan(storage._stale_partials(), tracked)
    recoverable = sum(target is not None for _, target in plan)
    partials = len(plan) - recoverable

    total_bytes = 0
    for path in files + thumbs + masks:
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
        "binned": binned,
        "files_on_disk": len(on_disk),
        "orphan_files": len(on_disk - tracked),
        "partial_files": partials,
        "recoverable_partials": recoverable,
        "bytes_on_disk": total_bytes,
        "masks_on_disk": len(masks),
    }


@gallery_lock.protected(lambda: storage.settings.db_path)
def clear_history(keep_favorites: bool = False, in_flight: InFlight | None = None) -> dict[str, int]:
    """Delete image records and the PNGs they point at.

    Rows are deleted rather than the database file, so this is safe while the
    server holds the connection open - dropping the file mid-run would leave a
    locked handle on Windows and lose the schema. `keep_favorites` keeps what
    it keeps on the `--where` path too: the stars and their sources.
    """
    where = f" WHERE {_UNKEPT}" if keep_favorites else ""
    with storage._write_lock, storage._session() as conn:
        doomed = [r["filename"] for r in conn.execute(f"SELECT filename FROM images{where}")]
        conn.execute(f"DELETE FROM images{where}")

    removed = 0
    for filename in doomed:
        path = storage.settings.outputs_dir / filename
        if path.exists() and storage._unlink(path):
            removed += 1
        storage._delete_thumbnails(path)

    # Partials from a power cut or a killed process, by age - never the fresh
    # one another process may be encoding right now. A kept row's committed
    # image is finished rather than swept (`storage._settle_plan`), and
    # before the thumbnail sweep below, which would take its thumbnail for
    # one whose PNG is gone.
    recovered, partials = storage._settle(storage._stale_partials())

    # Anything left with no row behind it: earlier deletions, interrupted runs.
    # An image still being written is a *.png.part and matches neither glob, so
    # a clear during a sweep can no longer delete one out from under it. The
    # scan is materialised *before* the rows are read: a save commits its row
    # and then renames, so any PNG the scan saw has a row by the time `keep`
    # is built - read the other way round, an image landing in between was
    # deleted with its row left pointing at nothing.
    files = list(storage.settings.outputs_dir.glob("*.png"))
    keep = {r["filename"] for r in storage._tracked_files()}
    orphans = 0
    for path in files:
        if path.name not in keep and storage._unlink(path):
            orphans += 1
    # A thumbnail whose PNG is gone - deleted above, or by hand.
    for path in storage._thumb_files():
        if not (storage.settings.outputs_dir / path.with_suffix(".png").name).exists():
            storage._unlink(path)
    # The inpaint masks go with the rows that named them: the whole clear
    # takes every one no kept row names, whatever its age - a mask uploaded a
    # minute ago covers an image that is gone - and `--keep-favorites` sweeps
    # as a delete does. Then the files with no row, after the settle window.
    masks = sweep_masks(in_flight, settle=keep_favorites, orphans=True)
    if not keep_favorites:
        try:
            storage.masks_dir().rmdir()  # only where nothing is left in it
        except OSError:
            pass

    return {"records": len(doomed), "files": removed, "orphans": orphans, "partials": partials,
            "recovered": recovered, "masks": masks}


@gallery_lock.protected(lambda: storage.settings.db_path)
def vacuum() -> None:
    """Shrink the database file after a bulk delete."""
    conn = storage._connect()
    try:
        conn.isolation_level = None  # VACUUM cannot run inside a transaction
        conn.execute("VACUUM")
    finally:
        conn.close()
