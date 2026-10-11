"""Verified, portable gallery archives and staged, non-destructive imports."""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import shutil
import sqlite3
import struct
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath

from . import bundle_crypto, bundle_images, gallery_lock, storage
from .bundle_archive import _DATABASE, _FORMAT, _MANIFEST, _READ_AHEAD_MAX, _verify_archive
from .bundle_common import (
    _CHUNK,
    _WORKERS,
    BundleChanged,
    BundleError,
    Progress,
    _basename,
    _connect_read,
    _quiet,
    _rows,
)
from .bundle_images import _critical_digest, _inspect_source, _metadata, _references, _rewrite_all, _validate_row

_SUFFIXES = (".zip", bundle_crypto.SUFFIX)


def _settings():
    return storage.settings


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _directory() -> Path:
    directory = _settings().db_path.parent.resolve() / "bundles"
    outputs = _settings().outputs_dir.resolve()
    if directory == outputs or directory.is_relative_to(outputs):
        raise BundleError("the bundles directory must be outside the images directory")
    if _is_link(directory):
        raise BundleError("the bundles directory must not be a symbolic link")
    return directory


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
        try:
            encrypted = bundle_crypto.is_encrypted(p)
        except OSError:
            continue
        found.append({"name": p.name, "bytes": p.stat().st_size, "created_at": p.stat().st_mtime,
                      "encrypted": encrypted})
    return sorted(found, key=lambda b: b["name"], reverse=True)


def _inventory() -> list[tuple[str, Path]]:
    outputs = _settings().outputs_dir
    found = []
    for directory, prefix, suffixes in (
        (outputs, "images/", {".png", ".webp"}),
        (outputs / "thumbnails", "images/thumbnails/", {".webp"}),
    ):
        if not directory.exists():
            continue
        if _is_link(directory):
            raise BundleError("image directories must not be symbolic links during a bundle operation")
        for path in sorted(directory.iterdir()):
            if path.name.lower().endswith(".part"):
                raise BundleError(f"finish or recover the incomplete image write first: {path.name}")
            if path.suffix.lower() not in suffixes:
                continue
            if _is_link(path) or not path.is_file() or not _basename(path.name):
                raise BundleError(f"unsafe image file: {path.name}")
            found.append((prefix + path.name, path))
    names = [name.casefold() for name, _ in found]
    if len(set(names)) != len(names):
        raise BundleError("image filenames differ only by letter case")
    return found


def preview() -> dict:
    settings = _settings()
    with gallery_lock.hold(settings.db_path, create=False):
        rows = _rows(settings.db_path)
        files = _inventory()
        names = {name for name, _ in files}
        for row in rows:
            if "images/" + row["filename"] not in names:
                raise BundleError(f"the original image is missing: {row['filename']}; nothing was archived")
        return {
            # Every row, the recycle bin's included, which the export carries
            # and clears with the rest; `binned` says how many of them are.
            "records": len(rows),
            "binned": sum(bool(r.get("binned_at")) for r in rows),
            "images": sum(p.suffix.lower() == ".png" for _, p in files),
            "thumbnails": sum(p.suffix.lower() == ".webp" for _, p in files),
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
    storage._migrated = None


def _remove_file(path: Path) -> None:
    path.unlink()



def _claim(source: Path, target: Path) -> None:
    if os.name == "nt":
        os.rename(source, target)  # refuses an existing target on Windows
    else:
        os.link(source, target)  # rename would replace one on POSIX


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


def export_bundle(expected: int, password: str | None = None, progress: Progress = _quiet) -> dict:
    settings = _settings()
    with gallery_lock.hold(settings.db_path):
        storage.discard_bundle_leftovers()
        before = preview()
        if before["records"] != expected:
            raise BundleChanged("the gallery changed; review its new count before creating a bundle")
        files = _inventory()
        directory = _directory()
        directory.mkdir(parents=True, exist_ok=True)
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
            rows = _rows(snapshot)
            manifest = {"format": _FORMAT, "version": 1, "records": len(rows), "files": {}}
            members = [(_DATABASE, snapshot), *files]
            try:
                with partial.open("xb") as stream:
                    # A password streams the ZIP through the cipher; no
                    # plaintext copy of the archive is ever written.
                    sink = bundle_crypto.EncryptingWriter(stream, password) if password else stream
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
                    if password:
                        sink.finish()
                    stream.flush()
                    os.fsync(stream.fileno())
                # The one read-back checks every member (and, with a password,
                # every chunk's tag) and decodes the archived originals from
                # the bytes it read, so a damaged tracked PNG can never be
                # cleared into an unusable import.
                _verify_archive(partial, password=password, rows=rows, progress=progress,
                                derived=sink.derived if password else None)
                partial.rename(final)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise

            result = {
                "name": name, "path": str(final), "images": before["images"],
                "records": len(rows), "binned": before["binned"], "bytes": final.stat().st_size,
                "cleared": False, "retained": [], "deleted_ids": [],
            }
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
                    remaining = {p.name for _, p in files if p.suffix.lower() == ".png" and p.exists()}
                    deleted = [r["id"] for r in rows if r["filename"] not in remaining]
                    conn.executemany("DELETE FROM images WHERE id = ?", [(i,) for i in deleted])
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
    settings = _settings()
    with gallery_lock.hold(settings.db_path):
        storage.discard_bundle_leftovers()
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
                return target if known and known["filename"] == relative and target.is_file() else None

            def inspect(row: dict, data: bytes):
                target = present_file(row["filename"])
                if target is None:
                    return _inspect_source(row, data), False
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
                    _validate_row(row, width, height, existing[row["filename"].casefold()]["content_hash"])
                    return None, True
                pixels = _inspect_source(row, data)
                return pixels, bundle_images._pixels(target).identity == pixels.identity

            def extract(member: str) -> bool:
                # An existing image wins on repeated import, so its bundle copy
                # is verified from memory but never unpacked. Any other member
                # whose destination exists is compared in place with its
                # verified manifest entry, so it is not unpacked either; only
                # PNGs, which new rows rewrite, are compared as staged.
                bits = member.split("/")
                if bits[-1].lower().endswith(".png"):
                    return not (len(bits) == 2 and present_file(bits[1]))
                return not settings.outputs_dir.joinpath(*bits[1:]).is_file()

            live = settings.db_path.stat().st_size if settings.db_path.exists() else 0
            manifest, source_rows, inspected = _verify_archive(
                Path(path), work, password=password, inspect=inspect, extract=extract,
                merge=(Path(merging), live), progress=progress)
            stage_db = Path(merging) / "merged.sqlite3"
            _backup(settings.db_path, stage_db)
            conn = sqlite3.connect(stage_db)
            conn.row_factory = sqlite3.Row
            installed: list[Path] = []
            try:
                # Build all new rows against a private database; no live change
                # precedes complete PNG, reference, conflict and metadata checks.
                columns = {r["name"] for r in conn.execute("PRAGMA table_info(images)")}
                for column in storage._COLUMNS:
                    if column not in columns:
                        kind = "REAL" if column in {"true_cfg", "lora_scale", "strength"} else "INTEGER" if column in {"refine", "batch"} else "TEXT"
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
                    pixels, same = inspected["images/" + name]
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
                        mapping[row["id"]] = known["id"]
                        if lost:
                            restoring.append(known)
                        else:
                            present += 1
                    else:
                        next_id += 1
                        mapping[row["id"]] = next_id
                        values = {key: row.get(key) for key in storage._COLUMNS}
                        values.update(id=next_id, content_hash=digest)
                        pending.append((row, values))
                unresolved = 0
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
                    by_id[values["id"]] = values
                rewrites = []
                for original, values in pending:
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
                        "binned": sum(bool(values.get("binned_at")) for _, values in pending),
                        "images": sum(name.startswith("images/") and name.lower().endswith(".png") for name in manifest["files"]),
                        "unresolved_references": unresolved}
            finally:
                if conn is not None:
                    conn.close()
