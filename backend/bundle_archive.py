"""The gallery bundle's archive format and its single verifying read.

Member validation, opening a plain or password-protected archive, the one
pass that checks every member, the bounded decode pool and staging writer,
and the per-volume free-space check.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import struct
import threading
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from . import backup_receipts, bundle_crypto, bundle_leases, reindex
from .bundle_common import _CHUNK, _WORKERS, MASK_HASH, BundleError, Progress, _basename, _mask_rows, _quiet, _rows
from .bundle_images import _inspect_mask, _inspect_source
from .metadata import edit_from_json


_FORMAT = "local-diffusion-gallery"
_DATABASE = "database.sqlite3"
_MANIFEST = "manifest.json"
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
    ) or _mask_member(name)


def _mask_member(name: str) -> bool:
    """`masks/<hash>.png`: an inpaint mask, by the whole hash it is named by.
    Its own top-level prefix, which a build from before masks skips (it
    installs only `images/`), so the manifest stays version 1."""
    bits = name.split("/")
    return (len(bits) == 2 and bits[0] == "masks" and bits[1].endswith(".png")
            and MASK_HASH.fullmatch(bits[1][: -len(".png")]) is not None)


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
def _open_archive(path: Path, password: str | None, derived: tuple[bytes, bytes] | None = None,
                  read_owner: bundle_leases.ArchiveRead | None = None):
    # Private callers own either a published-source reader (import) or this
    # unpublished partial (export). Do not add nested/.part companion claims.
    try:
        encrypted = bundle_crypto._is_encrypted(path, read_owner)
    except OSError as exc:
        raise BundleError(f"cannot read the bundle: {exc}") from exc
    if not encrypted:
        with bundle_leases._open_input(path, read_owner) as source, zipfile.ZipFile(source) as archive:
            yield archive
        return
    if not password:
        raise BundleError("this bundle is password-protected; enter its password")
    # Plaintext exists only in memory: every chunk is authenticated as read.
    with (bundle_crypto._DecryptingReader(path, password, derived=derived, read_owner=read_owner) as reader,
          zipfile.ZipFile(reader) as archive):
        yield archive


def _verify_archive(path: Path, destination: Path | None = None, *, password: str | None = None,
                    rows: list[dict] | None = None, inspect=_inspect_source,
                    extract: Callable[[str], bool] | None = None,
                    merge: tuple[Path, int] | None = None,
                    derived: tuple[bytes, bytes] | None = None,
                    masks: dict[str, dict] | None = None,
                    managed: backup_receipts.ManagedIdentity | None = None,
                    read_owner: bundle_leases.ArchiveRead | None = None,
                    progress: Progress = _quiet) -> tuple[dict, list[dict] | None, dict]:
    """Read every member once, verifying ZIP CRC, the manifest SHA-256 and, for
    a password-protected bundle, every chunk's tag.

    The database member is read first. Each tracked PNG (from `rows`, or from
    the database unpacked into `destination`) is passed to `inspect` from the
    bytes just read, on the decode pool. `extract` may decline to write a
    member that will never be installed. `merge` names the directory where
    the live database's copy (of the given size) will take the bundle's rows,
    for the free-space check. `derived` is export's own (header, key), which
    spares a second scrypt run. Each `masks/` member is decoded on the same
    pool and held to its name and to its row in `masks` (that table, by
    hash; read from the unpacked database where it is not given), its
    result its size. Returns the manifest, the rows and each tracked
    member's inspection result, the masks' included.
    """
    try:
        with _open_archive(path, password, derived, read_owner) as archive, _Pool() as pool, _Stager() as stager:
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
            if (not isinstance(manifest, dict) or manifest.get("format") != _FORMAT
                    or type(manifest.get("version")) is not int or manifest["version"] not in (1, 2)):
                raise BundleError("this is not a supported Local Diffusion bundle")
            if backup_receipts.MARKER in manifest or managed is not None:
                try:
                    marker = backup_receipts.read_marker(manifest.get(backup_receipts.MARKER))
                except backup_receipts.ReceiptError:
                    raise BundleError("the bundle's managed identity is invalid") from None
                if managed is not None and marker != managed:
                    raise BundleError("the bundle's managed identity does not match this export")
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
                needed_sources = set()
                for row in found:
                    try:
                        edit = edit_from_json(row.get("edit"))
                    except (TypeError, ValueError) as exc:
                        raise BundleError(f"invalid instruction-edit metadata for {row['filename']}: {exc}") from exc
                    if edit is not None:
                        if manifest["version"] == 1:
                            raise BundleError("version 1 bundles cannot carry instruction edits")
                        if edit.source is not None:
                            needed_sources.add(edit.source)
                    if "images/" + row["filename"] not in files:
                        raise BundleError(f"the bundle is missing {row['filename']}")
                # Private copies leave export's pre-cleanup snapshot unchanged.
                return {"images/" + row["filename"]: {
                    **row, "_bundle_version": manifest["version"],
                    "_edit_source_needed": row["id"] in needed_sources,
                } for row in found}

            tracked = track(rows) if rows is not None else {}
            masks = {} if masks is None and rows is not None else masks
            inspected_masks = []
            order = sorted(files, key=lambda n: n != _DATABASE)
            for done, name in enumerate(order, 1):
                expected = files[name]
                info = archive.getinfo(name)
                if (not isinstance(expected, dict) or type(expected.get("size")) is not int
                        or expected["size"] != info.file_size or not isinstance(expected.get("sha256"), str)):
                    raise BundleError(f"invalid size or checksum for {name}")
                digest = hashlib.sha256()
                write = destination is not None and (name == _DATABASE or extract is None or extract(name))
                target = destination.joinpath(*PurePosixPath(name).parts) if destination is not None and write else None
                if target:
                    stager.directory(target.parent)
                row = tracked.get(name)
                mask = _mask_member(name)
                if (row or mask) and info.file_size > _MAX_IMAGE_BYTES:
                    raise BundleError(f"{name} is too large to be a gallery image")
                # Tracked PNGs are rewritten (and synced) before installation;
                # everything else may be installed as is, so it is synced.
                # Both are written off the reading thread when held whole.
                # A mask is decoded as a tracked PNG is, so it is held whole too.
                whole = row or mask or (name != _DATABASE and info.file_size <= _READ_AHEAD_MAX)
                data = None
                output = target.open("xb") if target and not whole else None
                try:
                    with archive.open(info) as source:
                        if whole:
                            # A tracked PNG's size is bounded above; it is
                            # hashed on the pool, before its inspection.
                            data = source.read()
                            if not row and not mask:
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
                    assert data is not None  # Tracked members take the bounded whole-read branch above.
                    pool.submit(name, _png_weight(data), _checked, name, expected["sha256"], inspect, row, data)
                    data = None  # the bytes go with their inspection, not with the next member
                elif mask:
                    assert data is not None
                    known = (masks or {}).get(PurePosixPath(name).stem)
                    pool.submit(name, _png_weight(data), _checked, name, expected["sha256"], _mask_check,
                                (name, known), data)
                    inspected_masks.append(name)
                    data = None
                elif digest.hexdigest() != expected["sha256"]:
                    raise BundleError(f"the bundle checksum failed for {name}")
                if name == _DATABASE and rows is None and destination is not None:
                    rows = _rows(destination / _DATABASE)
                    tracked = track(rows)
                    if masks is None:
                        masks = _mask_rows(destination / _DATABASE)
                progress("verifying", done, len(order))
            results = pool.results(["images/" + row["filename"] for row in rows or []] + inspected_masks)
            stager.wait()
            return manifest, rows, results
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeError, EOFError) as exc:
        raise BundleError(f"the bundle is damaged: {exc}") from exc
    except bundle_crypto.CryptoError as exc:
        raise BundleError(str(exc)) from exc


def _mask_check(member: tuple[str, dict | None], data: bytes) -> tuple[int, int]:
    return _inspect_mask(member[0], data, member[1])


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
