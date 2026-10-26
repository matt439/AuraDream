"""Verified, portable gallery archives and staged, non-destructive imports."""
from __future__ import annotations

import collections
import hashlib
import io
import json
import logging
import os
import shutil
import sqlite3
import struct
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

from . import backup_receipts, bundle_capacity, bundle_crypto, bundle_images, bundle_leases, gallery_lock, storage
from .bundle_archive import _DATABASE, _FORMAT, _MANIFEST, _READ_AHEAD_MAX, _check_space, _verify_archive
from .bundle_common import (
    _CHUNK,
    _WORKERS,
    MASK_HASH,
    BundleChanged,
    BundleError,
    Progress,
    _basename,
    _connect_read,
    _mask_rows,
    _quiet,
    _rows,
)
from .bundle_images import _critical_digest, _inspect_source, _metadata, _references, _rewrite_all, _validate_row
from .metadata import require_header_metadata

_SUFFIXES = (".zip", bundle_crypto.SUFFIX)


def _settings():
    return storage.settings


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _directory() -> Path:
    return storage.bundle_directory()


def bundle_path(name: str) -> Path:
    if not _basename(name) or Path(name).suffix.lower() not in _SUFFIXES:
        raise BundleError(f"choose a bundle filename ending in .zip or {bundle_crypto.SUFFIX}")
    path = _directory() / name
    if _is_link(path):
        raise BundleError("a bundle must be a regular file")
    return path


def list_bundles() -> list[dict]:
    directory = _directory()
    if not directory.exists():
        return []
    found = []
    for p in directory.iterdir():
        if p.suffix.lower() not in _SUFFIXES or not p.is_file() or _is_link(p):
            continue
        # Acquisition and release remain outside the availability catch: an
        # installed-mode/native failure must not silently hide an archive.
        with bundle_leases.reading(p) as read_owner:
            try:
                encrypted = bundle_crypto._is_encrypted(p, read_owner)
                info = p.stat()
            except OSError:
                continue
            found.append({"name": p.name, "bytes": info.st_size, "created_at": info.st_mtime,
                          "encrypted": encrypted})
    return sorted(found, key=lambda b: b["name"], reverse=True)


class MissingImage(BundleError):
    """A gallery row whose PNG is not on disk, which no bundle can carry.

    Named by its id, which the gallery's `id:` search finds (a filename is
    no search key), with the two ways out: the file back, or the row gone -
    and only Delete permanently removes a row, from the recycle bin. Its
    thumbnail can outlive the PNG, so the card looks whole. `image_id` and
    `binned` travel in the 422 beside `detail` (main's handler), for the
    dialog's Show it.
    """

    def __init__(self, row: dict, outputs: Path):
        self.image_id = row["id"]
        self.binned = bool(row.get("binned_at"))
        way = ("delete it permanently from the recycle bin" if self.binned
               else "move it to the recycle bin and delete it permanently there")
        super().__init__(
            f"image #{self.image_id} ({row['filename']}) has no file in {outputs}; {way}, "
            "or put the file back; nothing was archived"
        )


def _partial_refusal(path: Path) -> str:
    """Why a `.part` beside the images stops a bundle, and what to do.

    `storage.settle_partials` has run first, so one still here is an encode
    in flight or one it could not move: a stale one is finished (a committed
    image's) or discarded an hour after it was written, and only a file
    another program holds open outlives that.
    """
    try:
        settled = path.stat().st_mtime < time.time() - storage.PARTIAL_SETTLE_SECONDS
    except OSError:
        settled = False
    if settled:
        return (f"The interrupted image write {path.name} in {path.parent} could not be "
                "finished or removed; close any program that has it open and try again")
    return (f"The image write {path.name} in {path.parent} is still in progress; try again when "
            "the render finishes (one a crash left behind is settled an hour after it began)")


log = logging.getLogger(__name__)


def _carried_mask(path: Path) -> bool:
    """Whether a file in the masks directory is a mask a bundle can carry:
    named by all 64 lowercase digits of its hash, `.png`, and a whole `L`
    PNG whose levels hash to that name - what the verifying read-back
    (`bundle_archive._verify_archive`) accepts. Anything else there - a
    user's own `scarf-mask.png`, a sync tool's `<hash> (1).png`, a file cut
    short or edited - no row can be using as the mask it names, and is left
    where it is, as `reindex` leaves it, rather than failing the export
    after the whole archive is written."""
    if path.suffix != ".png" or MASK_HASH.fullmatch(path.stem) is None:
        return False
    return storage._whole_mask(path, path.stem)


def _inventory(skipped: list[str] | None = None) -> list[tuple[str, Path]]:
    """Every file a whole-gallery bundle carries, as (member, path). A file
    in the masks directory that is no carried mask (`_carried_mask`) is left
    out and its name added to `skipped`, where a list is given."""
    outputs = _settings().outputs_dir
    found = []
    for directory, prefix, suffixes in (
        (outputs, "images/", {".png", ".webp"}),
        (outputs / "thumbnails", "images/thumbnails/", {".webp"}),
        # The inpaint masks, under their own prefix: not gallery images, and
        # named by their hash, which a merge never renumbers.
        (outputs / storage.MASKS_DIR, "masks/", {".png"}),
    ):
        if not directory.exists():
            continue
        if _is_link(directory):
            raise BundleError("image directories must not be symbolic links during a bundle operation")
        for path in sorted(directory.iterdir()):
            if path.name.lower().endswith(".part"):
                raise BundleError(_partial_refusal(path))
            if path.suffix.lower() not in suffixes:
                continue
            if _is_link(path) or not path.is_file() or not _basename(path.name):
                raise BundleError(f"unsafe image file: {path.name}")
            if prefix == "masks/" and not _carried_mask(path):
                log.warning("bundle: leaving out %s, which is no stored mask", path)
                if skipped is not None:
                    skipped.append(path.name)
                continue
            found.append((prefix + path.name, path))
    names = [name.casefold() for name, _ in found]
    if len(set(names)) != len(names):
        raise BundleError("image filenames differ only by letter case")
    return found


def preview() -> dict:
    settings = _settings()
    with gallery_lock.hold(settings.db_path, create=False):
        # A stale partial refused every count until something swept it, and
        # nothing in the page could: a killed save's committed image is
        # finished here, debris discarded, and only a write in flight remains.
        if settings.db_path.exists():
            storage.settle_partials()
        rows = _rows(settings.db_path)
        skipped: list[str] = []
        files = _inventory(skipped)
        names = {name for name, _ in files}
        for row in rows:
            if "images/" + row["filename"] not in names:
                raise MissingImage(row, settings.outputs_dir)
        return {
            # Every row, the recycle bin's included, which the export carries
            # and clears with the rest; `binned` says how many of them are.
            "records": len(rows),
            "binned": sum(bool(r.get("binned_at")) for r in rows),
            # The live starred images, every checkpoint's: what a bundle of
            # the starred takes (`submission.resolve_selection`'s filter).
            "starred": sum(bool(r["favorite"]) and not r.get("binned_at") for r in rows),
            "images": sum(name.startswith("images/") and p.suffix.lower() == ".png" for name, p in files),
            "thumbnails": sum(p.suffix.lower() == ".webp" for _, p in files),
            "masks": sum(name.startswith("masks/") for name, _ in files),
            # Files in the masks directory that are no stored mask, which the
            # bundle leaves where they are (`_carried_mask`).
            "masks_left_out": sorted(skipped),
            "bytes": sum(p.stat().st_size for _, p in files)
                + (settings.db_path.stat().st_size if settings.db_path.exists() else 0),
        }


def _backup(source: Path, target: Path) -> None:
    dest = sqlite3.connect(target)
    try:
        if source.exists():
            origin = _connect_read(source)
            try:
                origin.backup(dest)
            finally:
                origin.close()
        else:
            dest.executescript(storage._SCHEMA)
        dest.execute("PRAGMA journal_mode=DELETE")
        dest.commit()
    finally:
        dest.close()
    _rows(target)


def _merge_masks(conn: sqlite3.Connection, bundle: dict[str, dict], members: dict[str, tuple[int, int]]) -> set[str]:
    """Merge a bundle's inpaint masks into the staged database, by hash, and
    answer every mask hash this gallery holds once the import lands.

    `bundle` is its `masks` table, `members` the size each `masks/` member
    decoded at (`_inspect_mask` held it to its name and its row). A mask is
    content-addressed, so a row this gallery has already is the same mask -
    unless it records another size, which a hash that covers the size never
    can, and is refused. Only a mask whose file the bundle carries gains a
    row; a bundle row without one would name a file nobody has. A member
    with no row of its own is given one from its pixels, as
    `storage.store_mask` gives a lost row back.
    """
    conn.execute(storage._MASKS_SCHEMA)
    here = {row["hash"]: (row["width"], row["height"])
            for row in conn.execute("SELECT hash, width, height FROM masks")}
    for digest, row in bundle.items():
        if digest in here and here[digest] != (row["width"], row["height"]):
            raise BundleError(f"inpaint mask {digest[:16]} has another size in this gallery")
    now = datetime.now(timezone.utc).isoformat()
    for digest, size in members.items():
        if digest in here:
            if here[digest] != tuple(size):
                raise BundleError(f"inpaint mask {digest[:16]} has another size in this gallery")
            continue
        created = bundle.get(digest, {}).get("created_at")
        conn.execute("INSERT INTO masks (hash, width, height, created_at) VALUES (?, ?, ?, ?)",
                     (digest, size[0], size[1], created if isinstance(created, str) else now))
    directory = _settings().outputs_dir / storage.MASKS_DIR
    return {digest for digest in here if (directory / f"{digest}.png").is_file()} | set(members)


def _rebuilt_member(work: Path, values: dict) -> Path | None:
    """A new outpaint row's mask, built again from its spec and the row's
    size and written beside the unpacked members as one, where it hashes to
    the row's mask; None where it does not (the row then counts in
    `masks_missing`, as any inpaint's lost mask does)."""
    digest = values["mask"]
    try:
        spec = storage.OutpaintSpec.parse_key(values["outpaint"])
    except ValueError:
        return None
    mask = storage.rebuilt_outpaint_mask(values["width"], values["height"], spec, digest)
    if mask is None:
        return None
    target = work / storage.MASKS_DIR / f"{digest}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    mask.save(target, format="PNG", compress_level=6)
    return target


def _recovered_outpaint(work: Path, values: dict, source: dict | None):
    """An outpaint spec an older build's merge stripped from a new row,
    from its staged PNG's A1111 line (`storage.recovered_outpaint`), or None."""
    if source is None:
        return None
    staged = work / "images" / values["filename"]
    try:
        with Image.open(staged) as image:
            parameters = image.info.get("parameters")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    return storage.recovered_outpaint(
        parameters, values["width"], values["height"], values["mask"], (source["width"], source["height"])
    )


def _embedded_member(work: Path, values: dict) -> Path | None:
    """A new row's mask, from the copy its own staged PNG carries, written
    beside the unpacked members as one; None where the PNG holds no copy of
    the mask its row names, at the row's size. A whole-gallery or selection
    bundle carries only the masks whose files were there, so a mask lost in
    the gallery it came from travels only this way - as `reindex` finds it."""
    from .reindex import _embedded_mask

    digest = values["mask"]
    target = work / storage.MASKS_DIR / f"{digest}.png"
    if target.is_file():
        return target
    staged = work / "images" / values["filename"]
    try:
        with Image.open(staged) as image:
            mask = _embedded_mask(image, digest)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if mask is None or mask.size != (values["width"], values["height"]):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    mask.save(target, format="PNG", compress_level=6)
    return target


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint() -> None:
    path = _settings().db_path
    if not path.exists():
        return
    conn = sqlite3.connect(path)
    try:
        result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result and result[0]:
            raise BundleError("another program has the gallery database open; close it and try again")
    finally:
        conn.close()


def _replace_database(staged: Path) -> None:
    path = _settings().db_path
    _checkpoint()
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    staged.replace(path)


def _remove_file(path: Path) -> None:
    path.unlink()



def _claim(source: Path, target: Path) -> None:
    if os.name == "nt":
        os.rename(source, target)  # refuses an existing target on Windows
    else:
        os.link(source, target)  # rename would replace one on POSIX


def _remove_owned_partial(path: Path, owned: tuple[int, int] | None) -> None:
    """A claimed name may already have been reused by an unrelated writer."""
    if owned is None:
        return
    try:
        current = path.lstat()
        if (current.st_dev, current.st_ino) == owned:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def _install(source: Path, target: Path, installed: list[Path]) -> None:
    """Move a synced staged file into place, never replacing an existing file."""
    try:
        _claim(source, target)
        installed.append(target)
        return
    except FileExistsError:
        raise
    except OSError:
        pass  # another volume, or no hard links: copy instead
    # The copy is written as a `.part` beside its target and only then takes
    # the final name, so a crash mid-copy leaves a partial the sweeps know,
    # never a truncated image that every later import calls a conflict.
    partial = target.with_name(target.name + ".part")
    try:
        with partial.open("xb") as dest:
            with source.open("rb") as src:
                shutil.copyfileobj(src, dest, _CHUNK)
            dest.flush()
            os.fsync(dest.fileno())
        try:
            _claim(partial, target)
        except FileExistsError:
            raise
        except OSError:
            # No hard links either: claim the name exclusively, then swap
            # the complete copy over that empty claim of our own.
            os.close(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL))
            installed.append(target)
            os.replace(partial, target)
            return
        installed.append(target)
    finally:
        partial.unlink(missing_ok=True)


def _read_whole(path: Path) -> tuple[bytes, str] | None:
    with path.open("rb") as source:
        data = source.read(_READ_AHEAD_MAX + 1)
    if len(data) > _READ_AHEAD_MAX:
        return None
    return data, hashlib.sha256(data).hexdigest()


class _ReadAhead:
    """Reads export's source files, in order, a few ahead of the archive writer.

    Opening each file is most of the cost on Windows, where it waits for
    on-access scanning; the pool overlaps that and the SHA-256 with the
    writing. Each result is the file's bytes and digest, or None for one over
    `_READ_AHEAD_MAX`, which the writer streams itself.
    """

    def __init__(self, paths: list[Path]):
        self._paths = iter(paths)
        self._queue = collections.deque()
        self._executor = ThreadPoolExecutor(_WORKERS, thread_name_prefix="bundle-read")

    def next(self) -> tuple[bytes, str] | None:
        while len(self._queue) < 2 * _WORKERS and (path := next(self._paths, None)) is not None:
            self._queue.append(self._executor.submit(_read_whole, path))
        return self._queue.popleft().result()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._queue.clear()


def _unchanged(path: Path, saved: dict) -> bool:
    return path.stat().st_size == saved["size"] and _hash(path) == saved["sha256"]


def _chosen(ids: list[int]) -> tuple[list[dict], list[tuple[str, Path]]]:
    """The rows of `ids` a bundle of a selection carries, and their files.

    Each row's PNG and its thumbnail where it has one (in thumbnails/, or an
    old one beside the PNG), and an inpaint's mask where its file is here;
    never an unindexed image or another row's. A chosen row whose PNG is
    missing refuses as the whole gallery's preview does; a row the selection
    leaves out does not stop it. A mask whose file is gone is not carried,
    and the import counts the row (`masks_missing`).
    """
    settings = _settings()
    if settings.db_path.exists():
        storage.settle_partials()
    wanted = set(ids)
    rows = [row for row in _rows(settings.db_path) if row["id"] in wanted]
    found = dict(_inventory())
    files = []
    for row in rows:
        member = "images/" + row["filename"]
        if member not in found:
            raise MissingImage(row, settings.outputs_dir)
        files.append((member, found[member]))
        stem = Path(row["filename"]).stem
        files += [(name, found[name]) for name in (f"images/thumbnails/{stem}.webp", f"images/{stem}.webp")
                  if name in found]
    masks = sorted({f"masks/{row['mask']}.png" for row in rows if row.get("mask")} & set(found))
    files += [(name, found[name]) for name in masks]
    return rows, files


def _remove_run_recovery(snapshot: Path, *, compact: bool) -> None:
    """Remove local job definitions from an export's private database only.

    `_backup` is also used by import staging and replacement: stripping it
    there would discard this gallery's recovery state during an import.
    VACUUM removes freed cells, including prompts no exported image has.
    A chosen export already vacuums its restricted copy below.
    """
    conn = sqlite3.connect(snapshot)
    try:
        held = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_jobs'"
        ).fetchone()
        had_cells = held and conn.execute("SELECT 1 FROM run_jobs LIMIT 1").fetchone()
        conn.execute("DROP TABLE IF EXISTS run_jobs")
        conn.commit()
        if compact and had_cells:
            conn.execute("VACUUM")
    finally:
        conn.close()


def _restrict(snapshot: Path, keep: set[int]) -> None:
    """Leave only the chosen rows in export's private copy of the database.

    FTS5 keeps a deleted row's tokens in its segments until they merge and
    SQLite keeps freed pages, so the index is rebuilt and the copy vacuumed:
    a bundle of ten images given to someone carries no other prompt.
    """
    conn = sqlite3.connect(snapshot)
    try:
        conn.execute("CREATE TEMP TABLE chosen (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO chosen VALUES (?)", [(i,) for i in keep])
        conn.execute("DELETE FROM images WHERE id NOT IN (SELECT id FROM chosen)")
        # And the masks no chosen row names: a mask is a picture painted over
        # an image the bundle need not hold.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'masks'").fetchone():
            conn.execute("DELETE FROM masks WHERE hash NOT IN (SELECT mask FROM images WHERE mask IS NOT NULL)")
        conn.execute("INSERT INTO images_fts(images_fts) VALUES('rebuild')")
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


def _links_outside(rows: list[dict]) -> int:
    """References and refine sources naming an image the bundle does not hold.

    An import clears and counts each one (`unresolved_references`); export
    says so first, since a bundle of picked images leaves their sources out
    unless they were picked too.
    """
    held = {row["id"] for row in rows}
    return sum(
        sum(i not in held for i in _references(row)) + (row.get("refine") is not None and row["refine"] not in held)
        for row in rows
    )


def export_bundle(expected: int | None = None, password: str | None = None, progress: Progress = _quiet, *,
                  keep_gallery: bool = False, ids: list[int] | None = None,
                  managed: backup_receipts.ManagedIdentity | None = None) -> dict:
    """Write and verify a bundle; then, unless `keep_gallery`, clear the gallery.

    `expected` is the record count typed into the dialog, held to the
    gallery only when it is to be cleared. A backup (`keep_gallery`) keeps
    the one verifying read-back and skips only the second inventory and the
    cleanup. `ids` bundles just those rows, with their PNGs and thumbnails,
    and is always a backup: a part of the gallery is never cleared.
    """
    if ids is not None and not keep_gallery:
        raise BundleError("only a bundle of the whole gallery can clear it; a selection is backed up and kept")
    if managed is not None:
        if not keep_gallery or ids is not None:
            raise BundleError("managed exports must keep the whole gallery")
        if not isinstance(managed, backup_receipts.ManagedIdentity):
            raise BundleError("invalid managed export identity")
        managed = backup_receipts.read_marker(managed.marker())
    settings = _settings()
    with gallery_lock.hold(settings.db_path):
        if ids is None:
            before = preview()  # which settles stale partials first
            if not keep_gallery and before["records"] != expected:
                raise BundleChanged("the gallery changed; review its new count before creating a bundle")
            files = _inventory()
        else:
            chosen, files = _chosen(ids)
            if not chosen:
                raise BundleError("none of the chosen images is in the gallery any more; nothing was archived")
            before = {"images": len(chosen), "binned": sum(bool(r.get("binned_at")) for r in chosen)}
        directory = _directory()
        directory.mkdir(parents=True, exist_ok=True)
        # The archive holds every file whole, and the snapshot is a full copy
        # of the database - its write-ahead log folded in. Removing retained
        # cells or an unchosen selection needs VACUUM's second copy: refuse
        # here rather than fill the disk partway through.
        wal = settings.db_path.with_name(settings.db_path.name + "-wal")
        database = sum(p.stat().st_size for p in (settings.db_path, wal) if p.exists())
        with storage._session() as conn:
            retained_cells = conn.execute("SELECT 1 FROM run_jobs LIMIT 1").fetchone() is not None
        if managed is None:
            _check_space([(directory, database + sum(p.stat().st_size for _, p in files), "write the bundle"),
                          (settings.db_path.parent, database * (2 if ids is not None or retained_cells else 1),
                           "copy the gallery database for the bundle")])
        else:
            # Early DB+WAL heuristic only. A VACUUM can need two additional
            # copies while the snapshot already exists; reserve three here.
            # Final archive sizing below uses the actual compacted snapshot.
            _check_space([(directory, database + sum(p.stat().st_size for _, p in files), "write the bundle"),
                          (settings.db_path.parent, database * (3 if retained_cells else 1),
                           "copy the gallery database for the managed bundle"),
                          (settings.db_path.parent, bundle_capacity.JOURNAL_TEMP_BYTES,
                           "record the managed bundle receipt")])
        # The name is for the person saving it, on their own machine: local
        # wall-clock time. Nothing reads a time back out of the name.
        name = f"local-diffusion-{datetime.now().astimezone():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        name += bundle_crypto.SUFFIX if password else ".zip"
        final = bundle_path(name)
        partial = final.with_name(final.name + ".part")
        with tempfile.TemporaryDirectory(prefix=".bundle-", dir=settings.db_path.parent) as work:
            work = Path(work)
            snapshot = work / _DATABASE
            _backup(settings.db_path, snapshot)
            _remove_run_recovery(snapshot, compact=ids is None)
            if ids is not None:
                _restrict(snapshot, {row["id"] for row in chosen})
            rows = _rows(snapshot)
            if ids is not None and rows != chosen:
                raise BundleChanged("the gallery changed while its bundle was prepared; try again")
            manifest = {"format": _FORMAT, "version": 1, "records": len(rows), "files": {}}
            if managed is not None:
                manifest[backup_receipts.MARKER] = managed.marker()
            members = [(_DATABASE, snapshot), *files]
            if managed is not None:
                try:
                    capacity = bundle_capacity.managed_capacity(
                        [(member, path.stat().st_size) for member, path in members], manifest,
                        manifest_name=_MANIFEST, encrypted=bool(password),
                    )
                except (ValueError, TypeError, OverflowError) as exc:
                    raise BundleError("the managed bundle capacity could not be estimated") from exc
                # Snapshot/VACUUM space is already consumed. Check remaining
                # archive + full atomic journal temporary per actual volume;
                # encryption is streamed, never another plaintext copy.
                _check_space([(directory, capacity.archive_bytes, "write the managed bundle"),
                              (settings.db_path.parent, capacity.receipt_bytes,
                               "record the managed bundle receipt")])
            proof: backup_receipts.DigestWriter | None = None
            written: backup_receipts.FileStamp | None = None
            owned: tuple[int, int] | None = None
            try:
                with partial.open("xb") as stream:
                    if managed is not None:
                        opened = os.fstat(stream.fileno())
                        owned = (opened.st_dev, opened.st_ino)
                    # A password streams the ZIP through the cipher; no
                    # plaintext copy of the archive is ever written.
                    proof = backup_receipts.DigestWriter(stream) if managed is not None else None
                    outer = proof if proof is not None else stream
                    cipher = bundle_crypto.EncryptingWriter(outer, password) if password else None
                    sink = cipher if cipher is not None else outer
                    with (zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive,
                          _ReadAhead([path for _, path in members]) as ahead):
                        for done, (member, path) in enumerate(members, 1):
                            read = ahead.next()
                            with archive.open(member, "w", force_zip64=True) as dest:
                                if read is not None:
                                    data, sha256 = read
                                    dest.write(data)
                                    entry = {"size": len(data), "sha256": sha256}
                                else:
                                    digest = hashlib.sha256()
                                    size = 0
                                    with path.open("rb") as source:
                                        while chunk := source.read(_CHUNK):
                                            digest.update(chunk)
                                            size += len(chunk)
                                            dest.write(chunk)
                                    entry = {"size": size, "sha256": digest.hexdigest()}
                            # The manifest describes the very bytes written.
                            manifest["files"][member] = entry
                            read = data = None
                            progress("writing", done, len(members))
                        archive.writestr(_MANIFEST, json.dumps(manifest, ensure_ascii=True))
                    if cipher is not None:
                        cipher.finish()
                    stream.flush()
                    os.fsync(stream.fileno())
                if managed is not None:
                    # Windows finalizes write timestamps when its last writable
                    # handle closes; snapshot after close, before read-back.
                    written = backup_receipts.FileStamp.of(partial.lstat())
                    if (written.device, written.inode) != owned:
                        raise BundleError("the managed archive changed after writing")
                # The one read-back checks every member (and, with a password,
                # every chunk's tag) and decodes the archived originals from
                # the bytes it read, so a damaged tracked PNG can never be
                # cleared into an unusable import.
                _verify_archive(partial, password=password, rows=rows, progress=progress,
                                derived=cipher.derived if cipher is not None else None, masks=_mask_rows(snapshot),
                                managed=managed)
                if managed is None:
                    partial.rename(final)
                else:
                    if backup_receipts.FileStamp.of(partial.lstat()) != written:
                        raise BundleError("the managed archive changed during verification")
                    # Unlike replace/rename on POSIX, claim never overwrites an
                    # existing archive. Both paths remain in the same directory.
                    _claim(partial, final)
                    _remove_owned_partial(partial, owned)
            except BaseException:
                if managed is None:
                    partial.unlink(missing_ok=True)
                else:
                    _remove_owned_partial(partial, owned)
                raise

            result = {
                "name": name, "path": str(final), "images": before["images"],
                "records": len(rows), "binned": before["binned"], "bytes": final.stat().st_size,
                "cleared": False, "retained": [], "deleted_ids": [], "kept": keep_gallery,
                # Chosen ids that had gone by the time the gallery was reserved.
                "left_out": sorted(set(ids) - {row["id"] for row in rows}) if ids is not None else [],
                "links_outside": _links_outside(rows),
                "masks": sum(member.startswith("masks/") for member, _ in files),
            }
            if keep_gallery:
                if managed is not None:
                    result["managed"] = False
                    try:
                        assert proof is not None
                        stamp = backup_receipts.FileStamp.of(final.lstat())
                        if (written is None or stamp.size != proof.count
                                or (stamp.device, stamp.inode, stamp.modified_ns) !=
                                (written.device, written.inode, written.modified_ns)):
                            raise backup_receipts.ReceiptError("published archive size changed")
                        receipt = backup_receipts.Receipt(
                            managed, str(directory.resolve()), name, proof.sha256, stamp,
                            datetime.now(timezone.utc), "encrypted" if password else "unencrypted",
                            len(rows), before["images"],
                        )
                        backup_receipts.append(settings.db_path, receipt)
                        result["managed"] = True
                    except (OSError, ValueError, TypeError):
                        # Publication already succeeded. Never remove the copy
                        # or advance ownership on an uncertain journal update.
                        result["managed_warning"] = backup_receipts.JOURNAL_WARNING
                # A backup ends at the verified read-back: nothing is cleared,
                # so there is nothing for the second inventory to protect.
                return result
            try:
                current_files = _inventory()
                if ({name for name, _ in current_files} != {name for name, _ in files}
                        or _rows(settings.db_path) != rows):
                    raise BundleError("the gallery changed outside the app while its bundle was written; nothing was cleared")
                progress("checking", 0, len(current_files))
                with ThreadPoolExecutor(_WORKERS, thread_name_prefix="bundle-hash") as pool:
                    same = list(pool.map(lambda item: _unchanged(item[1], manifest["files"][item[0]]), current_files))
                for (_, source), unchanged in zip(current_files, same):
                    if not unchanged:
                        raise BundleError(f"{source.name} changed while its bundle was written; nothing was cleared")
                progress("clearing", 0, len(files))
                with storage._session() as conn:
                    # The write lock first: a database another program holds
                    # refuses here, before any file is removed. Rows are
                    # deleted only for files that really went, and reported
                    # only once that commits; a crash in between leaves rows
                    # whose files a re-import of this bundle restores.
                    conn.execute("BEGIN IMMEDIATE")
                    for _, source in files:
                        try:
                            _remove_file(source)
                        except OSError:
                            result["retained"].append(str(source.resolve()))
                    remaining = {p.name for member, p in files
                                 if member.startswith("images/") and p.suffix.lower() == ".png" and p.exists()}
                    deleted = [r["id"] for r in rows if r["filename"] not in remaining]
                    conn.executemany("DELETE FROM images WHERE id = ?", [(i,) for i in deleted])
                    # A mask's row goes with its file, as an image's does.
                    conn.executemany("DELETE FROM masks WHERE hash = ?", [
                        (PurePosixPath(member).stem,) for member, p in files
                        if member.startswith("masks/") and not p.exists()
                    ])
                    if not result["retained"]:
                        # A successful bundle-and-clear ends every retained
                        # run, including jobs that never wrote an image.
                        conn.execute("DELETE FROM run_jobs")
                # The surviving rows remain accurate even if Windows refuses
                # the subsequent replacement of the original database file.
                result["deleted_ids"] = deleted
                replacement = work / "empty.sqlite3"
                _backup(settings.db_path, replacement)
                conn = sqlite3.connect(replacement)
                try:
                    conn.execute("VACUUM")
                finally:
                    conn.close()
                _replace_database(replacement)
                result["cleared"] = not result["retained"]
            except Exception as exc:
                result["cleanup_error"] = str(exc)
                # The archive is already durable: always return its location,
                # even when external edits or a filesystem error prevent cleanup.
                try:
                    survivors = _inventory()
                except Exception:
                    survivors = files
                result["retained"] = list(dict.fromkeys([
                    *result["retained"], *(str(p.resolve()) for _, p in survivors if p.exists()),
                    str(settings.db_path.resolve()),
                ]))
            return result


def import_bundle(path: Path, password: str | None = None, progress: Progress = _quiet) -> dict:
    """Consume a published archive under one lease through staging and merge."""
    return _import_bundle(path, password, progress, lambda: bundle_leases.reading(path))


def _import_uploaded_bundle(path: Path, password: str | None = None, progress: Progress = _quiet) -> dict:
    """Only the API worker's owned unpublished upload; never infer by suffix."""
    return _import_bundle(path, password, progress, nullcontext)


def _import_bundle(path: Path, password: str | None, progress: Progress,
                   read_scope: Callable[[], AbstractContextManager[bundle_leases.ArchiveRead | None]]) -> dict:
    settings = _settings()
    # Every path reserves the destination first. Only published inputs acquire
    # a source reader; private uploads have no shared archive to protect.
    with gallery_lock.hold(settings.db_path), read_scope() as read_owner:
        storage.settle_partials()
        _inventory()  # A partial write or linked destination is not an import target.
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Members unpack on the images' volume, so installing one is a rename
        # there; the merged database is built beside the one it replaces.
        with (tempfile.TemporaryDirectory(prefix=".bundle-import-", dir=storage.bundle_staging_dir()) as work,
              tempfile.TemporaryDirectory(prefix=".bundle-merge-", dir=settings.db_path.parent) as merging):
            work = Path(work)
            current = _rows(settings.db_path)
            existing = {r["filename"].casefold(): r for r in current}

            def present_file(relative: str) -> Path | None:
                known = existing.get(relative.casefold())
                target = settings.outputs_dir / known["filename"] if known else None
                return target if known and known["filename"] == relative and target is not None and target.is_file() else None

            def mask_file(name: str) -> Path:
                return settings.outputs_dir / storage.MASKS_DIR / name

            def inspect(row: dict, data: bytes):
                target = present_file(row["filename"])
                if target is None:
                    pixels = _inspect_source(row, data)
                    return pixels, False, pixels.execution
                # Keeping the local file also keeps its metadata. Validate
                # that header against its own row before either identity
                # path can call it present; pixel equality says nothing
                # about which arithmetic recipe the PNG records.
                try:
                    with target.open("rb") as stream:
                        require_header_metadata(stream)
                        with Image.open(stream) as image:
                            if image.format != "PNG":
                                raise BundleError(f"not a PNG image: {target.name}")
                            bundle_images._execution(existing[row["filename"].casefold()], image.info)
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    raise BundleError(f"cannot read existing execution metadata for {target.name}: {exc}") from exc
                # A present image is kept and its bundle copy never unpacked.
                # Identical pixel-deciding chunks mean identical pixels, so the
                # copy is not decoded: its header's size and its row are still
                # validated. Anything else decodes both and compares them,
                # mode and palette included, never by the short hash.
                with target.open("rb") as stream:
                    known = _critical_digest(stream)
                if (known is not None and data[12:16] == b"IHDR"
                        and known == _critical_digest(io.BytesIO(data))):
                    width, height = struct.unpack(">II", data[16:24])
                    stream = io.BytesIO(data)
                    try:
                        require_header_metadata(stream)
                    except ValueError as exc:
                        raise BundleError(f"invalid PNG metadata for {row['filename']}: {exc}") from exc
                    with Image.open(stream) as image:
                        execution = bundle_images._execution(row, image.info)
                    _validate_row(row, width, height, existing[row["filename"].casefold()]["content_hash"])
                    return None, True, execution
                pixels = _inspect_source(row, data)
                return pixels, bundle_images._pixels(target).identity == pixels.identity, pixels.execution

            def extract(member: str) -> bool:
                # An existing image wins on repeated import, so its bundle copy
                # is verified from memory but never unpacked. Any other member
                # whose destination exists is compared in place with its
                # verified manifest entry, so it is not unpacked either; only
                # PNGs, which new rows rewrite, are compared as staged.
                bits = member.split("/")
                if bits[0] == "masks":
                    # Content-addressed: a mask already here under its name
                    # is this one, so its bundle copy is verified but kept.
                    return not mask_file(bits[1]).is_file()
                if bits[-1].lower().endswith(".png"):
                    return not (len(bits) == 2 and present_file(bits[1]))
                return not settings.outputs_dir.joinpath(*bits[1:]).is_file()

            live = settings.db_path.stat().st_size if settings.db_path.exists() else 0
            manifest, source_rows, inspected = _verify_archive(
                Path(path), work, password=password, inspect=inspect, extract=extract,
                merge=(Path(merging), live), progress=progress, read_owner=read_owner)
            assert source_rows is not None  # A destination makes the verifier read the staged database.
            stage_db = Path(merging) / "merged.sqlite3"
            _backup(settings.db_path, stage_db)
            bundle_masks = _mask_rows(work / _DATABASE)
            conn = sqlite3.connect(stage_db)
            conn.row_factory = sqlite3.Row
            installed: list[Path] = []
            try:
                carried = _merge_masks(conn, bundle_masks, {
                    PurePosixPath(member).stem: size for member, size in inspected.items()
                    if member.startswith("masks/")
                })
                # Build all new rows against a private database; no live change
                # precedes complete PNG, reference, conflict and metadata checks.
                columns = {r["name"] for r in conn.execute("PRAGMA table_info(images)")}
                for column in storage._COLUMNS:
                    if column not in columns:
                        kind = "REAL" if column in {"true_cfg", "lora_scale", "strength"} else "INTEGER" if column in {"refine", "batch", "vram_peak_allocated", "vram_peak_reserved"} else "TEXT"
                        conn.execute(f"ALTER TABLE images ADD COLUMN {column} {kind}")
                seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='images'").fetchone()
                next_id = max([r["id"] for r in current] + [seq[0] if seq else 0])
                mapping = {}
                pending = []
                present = 0
                restoring = []
                progress("merging", 0, len(source_rows))
                for row in source_rows:
                    name = row["filename"]
                    pixels, same, execution = inspected["images/" + name]
                    digest = pixels.hash if pixels else None  # None: present and unchanged
                    known = existing.get(name.casefold())
                    if known:
                        image = settings.outputs_dir / name
                        lost = known["filename"] == name and not (image.exists() or image.is_symlink())
                        if lost:
                            # The row outlived its file (a clear interrupted
                            # between the two, or a file deleted by hand). The
                            # bundle's copy restores it only if it is the very
                            # picture the row recorded; the row is unchanged.
                            # The short hash covers only the raw bytes, so the
                            # size and an RGB mode are held to the row as well.
                            if (pixels is None or not pixels.rgb or digest != known["content_hash"]
                                    or (pixels.width, pixels.height) != (known["width"], known["height"])):
                                raise BundleError(f"the gallery's record for {name} has lost its image, "
                                                  "and the bundle's copy is a different picture")
                        elif not same:
                            raise BundleError(f"another image already uses the filename {name}")
                        for key in ("prompt", "negative_prompt", "steps", "guidance_scale", "mu", "seed", "model_id"):
                            if known[key] != row[key]:
                                raise BundleError(f"different image metadata already uses the filename {name}")
                        if bundle_images._execution(known) != execution:
                            raise BundleError(f"different execution already uses the filename {name}")
                        incoming_anchor = row.get("crop_anchor")
                        if known.get("refine") is None:
                            # A prior import of this source-less selection
                            # already dropped both its source and its crop.
                            # Known rows win even if this archive now includes
                            # the source: importing it cannot relink that row.
                            incoming_anchor = None
                        if (known.get("crop_anchor") or "center") != (incoming_anchor or "center"):
                            raise BundleError(f"different crop anchor already uses the filename {name}")
                        mapping[row["id"]] = known["id"]
                        if lost:
                            restoring.append(known)
                        else:
                            present += 1
                    else:
                        next_id += 1
                        mapping[row["id"]] = next_id
                        values = {key: row.get(key) for key in storage._COLUMNS}
                        values.update(id=next_id, content_hash=digest, execution=execution)
                        # A result, so a bad one is dropped, never the bundle.
                        for key in ("vram_peak_allocated", "vram_peak_reserved"):
                            values[key] = storage._peak(values[key])
                        pending.append((row, values))
                unresolved = 0
                masks_missing = 0
                outpaints_recovered = 0
                outpaint_masks_rebuilt = 0
                # Masks recovered from their results' PNGs, staged as members.
                embedded: dict[str, Path] = {}
                by_id = {r["id"]: r for r in current}
                for original, values in pending:
                    refs = _references(original)
                    unresolved += sum(i not in mapping for i in refs)
                    refs = [mapping[i] for i in refs if i in mapping]
                    values["reference"] = json.dumps(refs) if refs else None
                    source_refine = original.get("refine")
                    if source_refine is not None and (type(source_refine) is not int or source_refine <= 0):
                        raise BundleError(f"invalid refine source on image {original['id']}")
                    values["refine"] = mapping.get(source_refine)
                    if source_refine is not None and values["refine"] is None:
                        unresolved += 1
                        values["strength"] = None
                        values["upscaler"] = None
                        values["crop_anchor"] = None
                        # Its mask covered that source's pixels, and a mask is
                        # never kept without the source it covers.
                        values["mask"] = None
                    values["crop_anchor"] = storage._crop_anchor(
                        values.get("crop_anchor"), source=values.get("refine"),
                        mask=values.get("mask"), outpaint=values.get("outpaint"),
                    )
                    if values.get("mask") is None or values.get("refine") is None:
                        # An outpaint never without its source and its mask.
                        values["outpaint"] = None
                    elif values.get("outpaint") is not None:
                        # Stored in `key()`'s one spelling (`0400` is `400`),
                        # so the row keys as its own re-render does.
                        values["outpaint"] = storage.OutpaintSpec.parse_key(values["outpaint"]).key()
                    else:
                        # An older build's import drops the column and the
                        # chunk keys; the A1111 line it left gives the spec
                        # back where the rebuilt mask is the row's.
                        spec = _recovered_outpaint(work, values, by_id.get(values["refine"]))
                        if spec is not None:
                            values["outpaint"] = spec.key()
                            outpaints_recovered += 1
                    if values.get("mask") is not None and values["mask"] not in carried:
                        # Lost in the gallery it came from, so no member - but
                        # the result's own PNG carries it (`PNG_MASK_KEY`),
                        # which is the picture where it hashes to the name;
                        # an outpaint's is built again from its spec.
                        recovered = _embedded_member(work, values)
                        if recovered is None and values.get("outpaint") is not None:
                            recovered = _rebuilt_member(work, values)
                            outpaint_masks_rebuilt += recovered is not None
                        if recovered is not None:
                            conn.execute(
                                "INSERT OR IGNORE INTO masks (hash, width, height, created_at) VALUES (?, ?, ?, ?)",
                                (values["mask"], values["width"], values["height"],
                                 datetime.now(timezone.utc).isoformat()),
                            )
                            embedded[values["mask"]] = recovered
                            carried.add(values["mask"])
                        else:
                            # The hash stays - it still says what ran - but this
                            # gallery cannot run it again until the mask is back.
                            masks_missing += 1
                    by_id[values["id"]] = values
                rewrites = []
                for _original, values in pending:
                    try:
                        inserted = storage._insert_row(conn, values)
                        if inserted["id"] != values["id"]:
                            raise BundleError("the staged image id sequence changed")
                        storage._row_to_record(inserted)
                        rewrites.append((values["filename"], _metadata(values, by_id)))
                    except (TypeError, ValueError, sqlite3.Error) as exc:
                        raise BundleError(f"invalid metadata for {values['filename']}: {exc}") from exc
                for known in restoring:
                    # Its PNG chunk names this gallery's ids, as the row does.
                    try:
                        rewrites.append((known["filename"], _metadata(known, by_id)))
                    except (TypeError, ValueError) as exc:
                        raise BundleError(f"invalid metadata for {known['filename']}: {exc}") from exc
                _rewrite_all(work / "images", rewrites)
                conn.commit()
                if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise BundleError("the merged database failed its integrity check")
                conn.close()
                conn = None
                targets = []
                known_names = set(existing)
                disk_names = {p.relative_to(settings.outputs_dir).as_posix().casefold(): p for _, p in _inventory()}
                for member in manifest["files"]:
                    if member.startswith("masks/"):
                        source = work.joinpath(*PurePosixPath(member).parts)
                        target = mask_file(PurePosixPath(member).name)
                        if target.is_file():
                            continue  # the same mask, by its name; verified above
                        if not source.exists():
                            raise BundleError(f"the gallery changed during import; try again ({member})")
                        targets.append((source, target))
                        continue
                    if not member.startswith("images/"):
                        continue
                    relative = member.removeprefix("images/")
                    source = work.joinpath(*PurePosixPath(member).parts)
                    target = settings.outputs_dir.joinpath(*PurePosixPath(relative).parts)
                    actual = disk_names.get(relative.casefold())
                    if actual:
                        if actual != target:
                            raise BundleError(f"a filename differing only by letter case already exists: {relative}")
                        if relative.casefold() in known_names and target.suffix.lower() == ".png":
                            continue
                        # A thumbnail caches a known image's pixels, and the
                        # gallery's own wins as the image does, whichever
                        # encoder wrote it. One of a new image is still held
                        # to its bundle bytes below.
                        if (target.suffix.lower() == ".webp"
                                and f"{PurePosixPath(relative).stem}.png".casefold() in known_names):
                            continue
                        # Rewritten PNGs are compared as staged; other members
                        # (not unpacked) are held to their verified manifest entry.
                        if source.suffix.lower() == ".png":
                            size, digest = source.stat().st_size, None
                        else:
                            saved = manifest["files"][member]
                            size, digest = saved["size"], saved["sha256"]
                        if size != actual.stat().st_size or _hash(actual) != (digest or _hash(source)):
                            raise BundleError(f"another file already uses the filename {relative}")
                        continue
                    if not source.exists():
                        # Not unpacked because its destination existed, which
                        # has gone since.
                        raise BundleError(f"the gallery changed during import; try again ({relative})")
                    targets.append((source, target))
                for digest, source in embedded.items():
                    target = mask_file(f"{digest}.png")
                    if not target.is_file():
                        targets.append((source, target))
                _checkpoint()
                try:
                    for done, (source, target) in enumerate(targets, 1):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        # Exclusive creation also protects an external file that
                        # appeared after validation; rollback removes only ours.
                        _install(source, target, installed)
                        progress("installing", done, len(targets))
                    _replace_database(stage_db)
                except BaseException:
                    for target in reversed(installed):
                        target.unlink(missing_ok=True)
                    raise
                # A new row keeps the bin state it was exported with; a
                # known one keeps this gallery's, as it keeps its star.
                return {"imported": len(pending), "present": present, "restored": len(restoring),
                        # Only newly inserted rows: known rows and repaired
                        # files must never widen Show newly added's range.
                        "first_id": pending[0][1]["id"] if pending else None,
                        "last_id": pending[-1][1]["id"] if pending else None,
                        "binned": sum(bool(values.get("binned_at")) for _, values in pending),
                        "images": sum(name.startswith("images/") and name.lower().endswith(".png") for name in manifest["files"]),
                        "unresolved_references": unresolved,
                        "masks": sum(target.parent.name == storage.MASKS_DIR for _, target in targets),
                        "masks_missing": masks_missing,
                        "outpaint_masks_rebuilt": outpaint_masks_rebuilt,
                        "outpaints_recovered": outpaints_recovered}
            finally:
                if conn is not None:
                    conn.close()
