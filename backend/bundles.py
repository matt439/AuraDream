"""Verified, portable gallery archives and staged, non-destructive imports."""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import shutil
import sqlite3
import stat
import struct
import tempfile
import threading
import uuid
import zipfile
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from PIL import Image, ImageFile

from . import bundle_crypto, gallery_lock, reindex, storage


class BundleError(ValueError):
    pass


class BundleChanged(BundleError):
    pass


_FORMAT = "local-diffusion-gallery"
_DATABASE = "database.sqlite3"
_MANIFEST = "manifest.json"
_CHUNK = 1024 * 1024
_SUFFIXES = (".zip", bundle_crypto.SUFFIX)
# Decoding every tracked PNG dominates both directions; it runs on a small
# pool while the archive is read. Admission is bounded by the decoded size
# the PNG header announces, so a few huge images cannot multiply memory.
_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
_DECODE_BUDGET = 1024 * 1024 * 1024
# A tracked PNG is held in memory to be decoded, and a hostile bundle can
# declare (and deflate to) any size; this is far above any generated image.
_MAX_IMAGE_BYTES = 256 * 1024 * 1024
# Export reads files up to this size whole, ahead of the writer; at most
# 2 * _WORKERS of them are held at once. Larger ones are streamed.
_READ_AHEAD_MAX = 16 * 1024 * 1024
# Import writes unpacked members held whole (tracked PNGs, and others up to
# _READ_AHEAD_MAX) on a pool: creating each file, and its fsync, is
# per-file latency that would otherwise stall the reader.
_STAGE_THREADS = 16
_STAGE_BUDGET = 256 * 1024 * 1024

Progress = Callable[[str, int, int], None]


def _quiet(phase: str, done: int, total: int) -> None:
    pass


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
            "records": len(rows),
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


def _valid_member(name: str) -> bool:
    if name == _DATABASE:
        return True
    bits = name.split("/")
    if not all(_basename(bit) for bit in bits):
        return False
    return (
        len(bits) == 2 and bits[0] == "images" and Path(bits[1]).suffix.lower() in {".png", ".webp"}
    ) or (
        len(bits) == 3 and bits[:2] == ["images", "thumbnails"] and Path(bits[2]).suffix.lower() == ".webp"
    )


def _png_weight(data: bytes) -> int:
    """Bytes held while decoding: the file plus the decoded pixels its header announces."""
    if data[:8] == reindex._PNG_SIGNATURE and data[12:16] == b"IHDR" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return len(data) + 8 * width * height
    return len(data)


class _Pool:
    """Inspects tracked PNGs in parallel while the archive is still being read.

    Results are collected in row order, so the first failure reported is the
    one a serial pass would have reported.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(_WORKERS, thread_name_prefix="bundle-decode")
        self._room = threading.Condition()
        self._used = 0
        self._pending = 0
        self.futures = {}

    def submit(self, key, weight: int, fn, *args) -> None:
        weight = min(weight, _DECODE_BUDGET)
        with self._room:
            self._room.wait_for(lambda: self._pending < 2 * _WORKERS and self._used + weight <= _DECODE_BUDGET)
            self._used += weight
            self._pending += 1

        def run():
            try:
                return fn(*args)
            finally:
                with self._room:
                    self._used -= weight
                    self._pending -= 1
                    self._room.notify_all()
        self.futures[key] = self._executor.submit(run)

    def results(self, keys) -> dict:
        return {key: self.futures[key].result() for key in keys}

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        for future in self.futures.values():
            future.cancel()
        self._executor.shutdown(wait=True)


@contextmanager
def _open_archive(path: Path, password: str | None, derived: tuple[bytes, bytes] | None = None):
    try:
        encrypted = bundle_crypto.is_encrypted(path)
    except OSError as exc:
        raise BundleError(f"cannot read the bundle: {exc}") from exc
    if not encrypted:
        with zipfile.ZipFile(path) as archive:
            yield archive
        return
    if not password:
        raise BundleError("this bundle is password-protected; enter its password")
    # Plaintext exists only in memory: every chunk is authenticated as read.
    with (bundle_crypto.DecryptingReader(path, password, derived=derived) as reader,
          zipfile.ZipFile(reader) as archive):
        yield archive


def _inspect_source(row: dict, data: bytes):
    return _validate_source_row(row, io.BytesIO(data), name=row["filename"])


def _verify_archive(path: Path, destination: Path | None = None, *, password: str | None = None,
                    rows: list[dict] | None = None, inspect=_inspect_source,
                    extract: Callable[[str], bool] | None = None,
                    merge: tuple[Path, int] | None = None,
                    derived: tuple[bytes, bytes] | None = None,
                    progress: Progress = _quiet) -> tuple[dict, list[dict] | None, dict]:
    """Read every member once, verifying ZIP CRC, the manifest SHA-256 and, for
    a password-protected bundle, every chunk's tag.

    The database member is read first. Each tracked PNG (from `rows`, or from
    the database unpacked into `destination`) is passed to `inspect` from the
    bytes just read, on the decode pool. `extract` may decline to write a
    member that will never be installed. `merge` names the directory where
    the live database's copy (of the given size) will take the bundle's rows,
    for the free-space check. `derived` is export's own (header, key), which
    spares a second scrypt run. Returns the manifest, the rows and each tracked
    member's inspection result.
    """
    try:
        with _open_archive(path, password, derived) as archive, _Pool() as pool, _Stager() as stager:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(n.casefold() for n in names)):
                raise BundleError("the bundle contains duplicate filenames")
            if _MANIFEST not in names or archive.getinfo(_MANIFEST).file_size > 32 * 1024 * 1024:
                raise BundleError("the bundle manifest is missing or too large")
            for info in infos:
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                    raise BundleError("the bundle contains a directory or special file")
                if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise BundleError("the bundle uses an unsupported encryption or compression")
                if info.filename != _MANIFEST and not _valid_member(info.filename):
                    raise BundleError(f"unsafe bundle member: {info.filename}")
                # A whole-member read() fetches all of compress_size in one
                # call, so padding after an early-ending deflate stream would
                # bypass every file_size cap. Deflate never grows data by more.
                if info.compress_size > info.file_size + info.file_size // 1000 + 1024:
                    raise BundleError(f"{info.filename} is stored larger than its contents")
            manifest = json.loads(archive.read(_MANIFEST))
            if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT or manifest.get("version") != 1:
                raise BundleError("this is not a supported Local Diffusion bundle")
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != set(names) - {_MANIFEST} or _DATABASE not in files:
                raise BundleError("the bundle manifest does not match its files")
            if destination is not None:
                needs = [(destination, sum(info.file_size for info in infos), "unpack this bundle")]
                if merge is not None:
                    needs.append((merge[0], merge[1] + archive.getinfo(_DATABASE).file_size,
                                  "merge its records beside the gallery database"))
                _check_space(needs)

            def track(found: list[dict]) -> dict[str, dict]:
                if len(found) != manifest.get("records"):
                    raise BundleError("the manifest's record count does not match its database")
                for row in found:
                    if "images/" + row["filename"] not in files:
                        raise BundleError(f"the bundle is missing {row['filename']}")
                return {"images/" + row["filename"]: row for row in found}

            tracked = track(rows) if rows is not None else {}
            order = sorted(files, key=lambda n: n != _DATABASE)
            for done, name in enumerate(order, 1):
                expected = files[name]
                info = archive.getinfo(name)
                if (not isinstance(expected, dict) or type(expected.get("size")) is not int
                        or expected["size"] != info.file_size or not isinstance(expected.get("sha256"), str)):
                    raise BundleError(f"invalid size or checksum for {name}")
                digest = hashlib.sha256()
                write = destination is not None and (name == _DATABASE or extract is None or extract(name))
                target = destination.joinpath(*PurePosixPath(name).parts) if write else None
                if target:
                    stager.directory(target.parent)
                row = tracked.get(name)
                if row and info.file_size > _MAX_IMAGE_BYTES:
                    raise BundleError(f"{name} is too large to be a gallery image")
                # Tracked PNGs are rewritten (and synced) before installation;
                # everything else may be installed as is, so it is synced.
                # Both are written off the reading thread when held whole.
                whole = row or (name != _DATABASE and info.file_size <= _READ_AHEAD_MAX)
                data = None
                output = target.open("xb") if target and not whole else None
                try:
                    with archive.open(info) as source:
                        if whole:
                            # A tracked PNG's size is bounded above; it is
                            # hashed on the pool, before its inspection.
                            data = source.read()
                            if not row:
                                digest.update(data)
                            if target:
                                stager.write(target, data, sync=not row)
                        while not whole and (chunk := source.read(_CHUNK)):
                            digest.update(chunk)
                            if output:
                                output.write(chunk)
                    if output and name != _DATABASE:
                        output.flush()
                        stager.close(output)
                        output = None
                finally:
                    if output:
                        output.close()
                if row:
                    pool.submit(name, _png_weight(data), _checked, name, expected["sha256"], inspect, row, data)
                    data = None  # the bytes go with their inspection, not with the next member
                elif digest.hexdigest() != expected["sha256"]:
                    raise BundleError(f"the bundle checksum failed for {name}")
                if name == _DATABASE and rows is None and destination is not None:
                    rows = _rows(destination / _DATABASE)
                    tracked = track(rows)
                progress("verifying", done, len(order))
            results = pool.results(["images/" + row["filename"] for row in rows or []])
            stager.wait()
            return manifest, rows, results
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeError, EOFError) as exc:
        raise BundleError(f"the bundle is damaged: {exc}") from exc
    except bundle_crypto.CryptoError as exc:
        raise BundleError(str(exc)) from exc


def _checked(name: str, sha256: str, inspect, row: dict, data: bytes):
    if hashlib.sha256(data).hexdigest() != sha256:
        raise BundleError(f"the bundle checksum failed for {name}")
    return inspect(row, data)


class _Stager:
    """Writes, syncs and closes unpacked members off the archive-reading thread.

    Admission is bounded by count and by the bytes waiting to be written (a
    tracked PNG may be up to `_MAX_IMAGE_BYTES`). Every queued task runs to
    completion, on any path, so no handle is left open when the staging
    directory is removed.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(_STAGE_THREADS, thread_name_prefix="bundle-stage")
        self._room = threading.Condition()
        self._used = 0
        self._pending = 0
        self._futures = []
        self._made: set[Path] = set()

    def directory(self, path: Path) -> None:
        if path not in self._made:
            path.mkdir(parents=True, exist_ok=True)
            self._made.add(path)

    def _submit(self, weight: int, fn, *args) -> None:
        weight = min(weight, _STAGE_BUDGET)
        with self._room:
            self._room.wait_for(lambda: self._pending < 2 * _STAGE_THREADS
                                and self._used + weight <= _STAGE_BUDGET)
            self._used += weight
            self._pending += 1

        def run():
            try:
                fn(*args)
            finally:
                with self._room:
                    self._used -= weight
                    self._pending -= 1
                    self._room.notify_all()
        self._futures.append(self._executor.submit(run))

    def write(self, target: Path, data: bytes, *, sync: bool) -> None:
        self._submit(len(data), _write_staged, target, data, sync)

    def close(self, output) -> None:
        """Sync and close a file the reading thread wrote."""
        self._submit(0, _sync_close, output)

    def wait(self) -> None:
        for future in self._futures:
            future.result()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._executor.shutdown(wait=True)


def _write_staged(target: Path, data: bytes, sync: bool) -> None:
    with target.open("xb") as output:
        output.write(data)
        if sync:
            output.flush()
            os.fsync(output.fileno())


def _sync_close(output) -> None:
    try:
        os.fsync(output.fileno())
    finally:
        output.close()


def _check_space(needs: list[tuple[Path, int, str]]) -> None:
    """Refuse unless each volume has room for everything written to it."""
    volumes: dict[int, list] = {}
    for where, size, purpose in needs:
        volume = volumes.setdefault(os.stat(where).st_dev, [where, 0, purpose])
        volume[1] += size
    for where, size, purpose in volumes.values():
        if size > shutil.disk_usage(where).free:
            raise BundleError(f"not enough free space to {purpose}")


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


def _rewrite_metadata(path: Path, metadata: dict) -> None:
    """Replace one PNG text chunk while streaming the untouched image data."""
    payload = reindex.PNG_METADATA_KEY.encode("ascii") + b"\0" + json.dumps(metadata).encode("ascii")
    chunk = len(payload).to_bytes(4, "big") + b"tEXt" + payload
    chunk += zlib.crc32(b"tEXt" + payload).to_bytes(4, "big")
    temp = path.with_suffix(".metadata.part")
    inserted = False
    with path.open("rb") as source, temp.open("xb") as target:
        signature = source.read(8)
        if signature != reindex._PNG_SIGNATURE:
            raise BundleError(f"not a PNG image: {path.name}")
        target.write(signature)
        while header := source.read(8):
            if len(header) != 8:
                raise BundleError(f"truncated PNG: {path.name}")
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            start = source.tell()
            keyword = reindex._text_keyword(kind, source.read(min(length, 80)))
            source.seek(start)
            if keyword in {key.encode("ascii") for key in reindex.PNG_METADATA_KEYS}:
                source.seek(length + 4, 1)
                continue
            if kind in {b"IDAT", b"IEND"} and not inserted:
                target.write(chunk)
                inserted = True
            target.write(header)
            remaining = length + 4
            while remaining:
                block = source.read(min(_CHUNK, remaining))
                if not block:
                    raise BundleError(f"truncated PNG: {path.name}")
                target.write(block)
                remaining -= len(block)
            if kind == b"IEND":
                break
        if not inserted:
            raise BundleError(f"PNG has no image data: {path.name}")
        # Installation moves this file into place rather than copying it.
        target.flush()
        os.fsync(target.fileno())
    temp.replace(path)


def _rewrite_all(directory: Path, rewrites: list[tuple[str, dict]]) -> None:
    """Rewrite staged PNGs' metadata on a pool: each is file I/O and an fsync.

    Failures are raised in order, and every worker has stopped before the
    staging directory is cleaned up.
    """
    with ThreadPoolExecutor(_WORKERS, thread_name_prefix="bundle-rewrite") as pool:
        futures = [(name, pool.submit(_rewrite_metadata, directory / name, metadata))
                   for name, metadata in rewrites]
        for name, future in futures:
            try:
                future.result()
            except (TypeError, ValueError) as exc:
                pool.shutdown(cancel_futures=True)
                raise BundleError(f"invalid metadata for {name}: {exc}") from exc
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise


def _critical_digest(source) -> str | None:
    """SHA-256 of the chunks that decide a PNG's pixels (None if malformed).

    Equal digests mean equal decoded pixels; metadata chunks are ignored.
    """
    try:
        if source.read(8) != reindex._PNG_SIGNATURE:
            return None
        digest = hashlib.sha256()
        while header := source.read(8):
            if len(header) != 8:
                return None
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            if kind in {b"IHDR", b"PLTE", b"tRNS", b"IDAT"}:
                digest.update(header)
                remaining = length
                while remaining:
                    block = source.read(min(_CHUNK, remaining))
                    if not block:
                        return None
                    digest.update(block)
                    remaining -= len(block)
                source.seek(4, 1)
            else:
                source.seek(length + 4, 1)
            if kind == b"IEND":
                return digest.hexdigest()
        return None
    except (OSError, ValueError):
        return None


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
                "records": len(rows), "bytes": final.stat().st_size,
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


class _Pixels(NamedTuple):
    hash: str  # storage.content_hash: the gallery's own, of the raw bytes only
    width: int
    height: int
    # What decides the picture: the mode, size, palette and transparency as
    # well as the full digest of the raw bytes. A palette image's bytes are
    # indices, so two different pictures can share `hash`.
    identity: str
    # Plain RGB, no transparency: what every render is saved as, and so the
    # only form in which `hash` alone (with the size) decides the picture.
    rgb: bool


def _raw_digest(image: Image.Image) -> str:
    """SHA-256 of `image.tobytes()`, hashed block by block as Pillow encodes it.

    The same bytes in the same order (`test_raw_digest_is_tobytes`), without
    the whole-image copy and join, which serialised the decode pool.
    """
    digest = hashlib.sha256()
    if image.width and image.height:
        encoder = Image._getencoder(image.mode, "raw", image.mode)
        encoder.setimage(image.im, (0, 0) + image.size)
        bufsize = max(ImageFile.MAXBLOCK, image.width * 4)  # as tobytes
        while True:
            _, status, data = encoder.encode(bufsize)
            digest.update(data)
            if status:
                break
        if status < 0:
            raise ValueError(f"encoder error {status} reading the pixels")
    return digest.hexdigest()


def _pixels(path, *, name: str | None = None) -> _Pixels:
    name = name or path.name
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise BundleError(f"not a PNG image: {name}")
            image.load()
            raw = _raw_digest(image)
            identity = hashlib.sha256(repr((
                image.mode, image.size, (image.palette.mode, image.palette.tobytes()) if image.palette else None,
                image.info.get("transparency"), raw,
            )).encode()).hexdigest()
            # The same digest storage.content_hash takes, without a second pass.
            rgb = image.mode == "RGB" and "transparency" not in image.info
            return _Pixels(raw[:16], image.width, image.height, identity, rgb)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BundleError(f"cannot read image {name}: {exc}") from exc


def _references(row: dict) -> list[int]:
    value = row.get("reference")
    if value is None:
        return []
    try:
        refs = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"invalid references on image {row['id']}") from exc
    if not isinstance(refs, list) or any(type(i) is not int or i <= 0 for i in refs):
        raise BundleError(f"invalid references on image {row['id']}")
    return refs


def _validate_source_row(row: dict, source, *, name: str | None = None) -> _Pixels:
    pixels = _pixels(source, name=name)
    _validate_row(row, pixels.width, pixels.height, pixels.hash)
    return pixels


def _validate_row(row: dict, width: int, height: int, content_hash: str) -> None:
    if (width, height) != (row["width"], row["height"]):
        raise BundleError(f"image dimensions disagree with the database: {row['filename']}")
    _references(row)
    refine = row.get("refine")
    if refine is not None and (type(refine) is not int or refine <= 0):
        raise BundleError(f"invalid refine source on image {row['id']}")
    values = {key: row.get(key) for key in storage._COLUMNS}
    values.update(id=row["id"], content_hash=content_hash)
    try:
        storage._row_to_record(values)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"invalid metadata for {row['filename']}: {exc}") from exc


def _metadata(row: dict, by_id: dict[int, dict]) -> dict:
    def provenance(image_id):
        source = by_id.get(image_id)
        if not source:
            return None
        return {"filename": source["filename"], "hash": source["content_hash"],
                "width": source["width"], "height": source["height"]}
    refs = _references(row)
    return {
        "prompt": row["prompt"], "negative_prompt": row["negative_prompt"],
        "steps": row["steps"], "guidance_scale": row["guidance_scale"], "true_cfg": row["true_cfg"],
        "mu": row["mu"], "seed": row["seed"], "model": row["model_id"],
        "quantization": row["quantization"], "job": row["job_id"], "run": row["run_id"],
        "size": f"{row['width']}x{row['height']}",
        "variables": json.loads(row["variables"]) if row["variables"] else None,
        "hash": row["content_hash"], "reference": refs or None,
        "reference_sources": [provenance(i) for i in refs] or None,
        "refine": row["refine"], "refine_source": provenance(row["refine"]),
        "strength": row["strength"], "lora": row["lora"], "lora_scale": row["lora_scale"],
    }


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
                return pixels, _pixels(target).identity == pixels.identity

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
                        kind = "REAL" if column in {"true_cfg", "lora_scale", "strength"} else "INTEGER" if column == "refine" else "TEXT"
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
                return {"imported": len(pending), "present": present, "restored": len(restoring),
                        "images": sum(name.startswith("images/") and name.lower().endswith(".png") for name in manifest["files"]),
                        "unresolved_references": unresolved}
            finally:
                if conn is not None:
                    conn.close()
