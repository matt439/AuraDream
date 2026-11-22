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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import backup_receipts, bundle_crypto, bundle_leases, tiled_refine_contract
from .bundle_common import _CHUNK, _WORKERS, MASK_HASH, BrowseUnavailable, BundleError, Progress, _basename, _mask_rows, _quiet, _rows
from .bundle_images import _inspect_mask, _inspect_source
from .metadata import PNG_SIGNATURE, detailer_from_json, edit_from_json, tiled_refine_from_json
from .controlnet_contract import ControlNetSpec
from .spatial_conditioning import conditioning_from_json


_FORMAT = "auradream-gallery"
# Bundles written before the rename to AuraDream carry the old id; they
# import unchanged (version 1 and 2 alike).
_LEGACY_FORMATS = frozenset({"local-diffusion-gallery"})
_DATABASE = "database.sqlite3"
_MANIFEST = "manifest.json"
# The largest manifest the verified read accepts; the managed export's
# capacity preflight budgets the same bound.
_MANIFEST_MAX = 32 * 1024 * 1024
_BROWSE_DATABASE_MAX = 64 * 1024 * 1024
_DECODE_BUDGET = 1024 * 1024 * 1024
# Preparation may hold a Lanczos intermediate beside the 2048-square RGB core.
_TILED_PREPARE_BYTES = 8 * tiled_refine_contract.CANVAS_SIZE[0] * tiled_refine_contract.CANVAS_SIZE[1]
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


@dataclass(frozen=True)
class ArchiveSelection:
    """Members chosen only from a successfully verified, pinned browse view."""

    image_id: int
    image_ids: frozenset[int]
    members: frozenset[str]
    verified_manifest: dict


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
    if data[:8] == PNG_SIGNATURE and data[12:16] == b"IHDR" and len(data) >= 24:
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
        del password
        with bundle_leases._open_input(path, read_owner) as source, zipfile.ZipFile(source) as archive:
            yield archive
        return
    if not password:
        raise BrowseUnavailable("this bundle is password-protected; enter its password")
    # Plaintext exists only in memory: every chunk is authenticated as read.
    with (bundle_crypto._DecryptingReader(path, password, derived=derived, read_owner=read_owner) as reader,
          zipfile.ZipFile(reader) as archive):
        del password
        yield archive


def _verify_archive(path: Path, destination: Path | None = None, *, password: str | None = None,
                    rows: list[dict] | None = None, inspect=_inspect_source,
                    extract: Callable[[str], bool] | None = None,
                    merge: tuple[Path, int] | None = None,
                    derived: tuple[bytes, bytes] | None = None,
                    masks: dict[str, dict] | None = None,
                    managed: backup_receipts.ManagedIdentity | None = None,
                    read_owner: bundle_leases.ArchiveRead | None = None,
                    opened: zipfile.ZipFile | None = None,
                    catalogue: Callable[[bytes], tuple[list[dict], dict[str, dict]]] | None = None,
                    selection: ArchiveSelection | None = None,
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
    member's inspection result, the masks' included. A read-only browser
    borrows its pinned `opened` ZIP and consumes the verified database
    through `catalogue` in memory, with neither extraction nor a merge.
    A restore borrows that same ZIP. It checks the already verified manifest
    again and rechecks the database and selected members as it stages them.
    """
    if selection is not None and (opened is None or destination is None):
        raise ValueError("an archive selection requires its verified view and a staging directory")
    source_scope = nullcontext(opened) if opened is not None else _open_archive(path, password, derived, read_owner)
    try:
        with source_scope as archive, _Pool() as pool, _Stager() as stager:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(n.casefold() for n in names)):
                raise BundleError("the bundle contains duplicate filenames")
            if _MANIFEST not in names or archive.getinfo(_MANIFEST).file_size > _MANIFEST_MAX:
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
            if (not isinstance(manifest, dict) or manifest.get("format") not in {_FORMAT, *_LEGACY_FORMATS}
                    or type(manifest.get("version")) is not int or manifest["version"] not in (1, 2, 3, 4)):
                raise BundleError("this is not a supported AuraDream bundle")
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
            active = set(files)
            if selection is not None:
                if manifest != selection.verified_manifest:
                    raise BrowseUnavailable("the saved bundle changed since it was opened")
                active = {_DATABASE, *selection.members}
            if destination is not None:
                unpacked = sum(info.file_size for info in infos
                               if selection is None or info.filename in active)
                needs = [(destination, unpacked, "unpack this bundle")]
                if merge is not None:
                    needs.append((merge[0], merge[1] + archive.getinfo(_DATABASE).file_size,
                                  "merge its records beside the gallery database"))
                _check_space(needs)

            def track(found: list[dict]) -> dict[str, dict]:
                if len(found) != manifest.get("records"):
                    raise BundleError("the manifest's record count does not match its database")
                needed_sources = set()
                tiled_roots = set()
                has_tiled_results = False
                has_spatial_results = False
                for row in found:
                    try:
                        edit = edit_from_json(row.get("edit"))
                        detailer = detailer_from_json(row.get("detailer"))
                    except (TypeError, ValueError) as exc:
                        raise BundleError(f"invalid instruction-edit metadata for {row['filename']}: {exc}") from exc
                    if edit is not None:
                        if manifest["version"] == 1:
                            raise BundleError("version 1 bundles cannot carry instruction edits")
                        if edit.source is not None:
                            needed_sources.add(edit.source)
                    if detailer is not None and detailer.plan.source_id is not None:
                        needed_sources.add(detailer.plan.source_id)
                    try:
                        tiled_refine = tiled_refine_from_json(row.get("tiled_refine"))
                    except (TypeError, ValueError) as exc:
                        raise BundleError(f"invalid Tiled Refine metadata for {row['filename']}: {exc}") from exc
                    if tiled_refine is not None:
                        if manifest["version"] < 3:
                            raise BundleError("Tiled Refine requires bundle version 3")
                        has_tiled_results = True
                        if tiled_refine.source is not None:
                            tiled_roots.add(tiled_refine.source)
                    try:
                        conditioning = conditioning_from_json(row.get("spatial_conditioning"))
                    except (TypeError, ValueError) as exc:
                        raise BundleError(f"invalid spatial conditioning metadata for {row['filename']}: {exc}") from exc
                    if conditioning is not None:
                        if manifest["version"] < 4:
                            raise BundleError("Spatial conditioning requires bundle version 4")
                        has_spatial_results = True
                    if isinstance(conditioning, ControlNetSpec) and conditioning.source is not None:
                        needed_sources.add(conditioning.source)
                    if "images/" + row["filename"] not in files:
                        raise BundleError(f"the bundle is missing {row['filename']}")
                if manifest["version"] == 3 and not has_tiled_results:
                    raise BundleError("bundle version 3 is only supported for archives with Tiled Refine results")
                if manifest["version"] == 4 and not has_spatial_results:
                    raise BundleError("bundle version 4 is only supported for archives with spatial conditioning results")
                # Private copies leave export's pre-cleanup snapshot unchanged.
                return {"images/" + row["filename"]: {
                    **row, "_bundle_version": manifest["version"],
                    "_full_source_proof_needed": row["id"] in needed_sources or row["id"] in tiled_roots,
                    "_tiled_root_needed": row["id"] in tiled_roots,
                } for row in found}

            tracked = track(rows) if rows is not None else {}
            masks = {} if masks is None and rows is not None else masks
            inspected_masks = []
            order = sorted((name for name in files if name in active), key=lambda n: n != _DATABASE)
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
                memory_catalogue = name == _DATABASE and catalogue is not None
                if memory_catalogue and info.file_size > _BROWSE_DATABASE_MAX:
                    raise BrowseUnavailable("this catalogue exceeds the 64 MiB limit for browsing in memory")
                whole = row or mask or memory_catalogue or (name != _DATABASE and info.file_size <= _READ_AHEAD_MAX)
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
                    weight = _png_weight(data)
                    if row.get("_tiled_root_needed"):
                        weight += _TILED_PREPARE_BYTES
                    pool.submit(name, weight, _checked, name, expected["sha256"], inspect, row, data)
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
                if name == _DATABASE and catalogue is not None:
                    assert data is not None
                    rows, masks = catalogue(data)
                    tracked = track(rows)
                    del data
                elif name == _DATABASE and rows is None and destination is not None:
                    rows = _rows(destination / _DATABASE)
                    tracked = track(rows)
                    if selection is not None:
                        rows = [row for row in rows if row["id"] in selection.image_ids]
                    if masks is None:
                        masks = _mask_rows(destination / _DATABASE)
                progress("verifying", done, len(order))
            results = pool.results(["images/" + row["filename"] for row in rows or []] + inspected_masks)
            stager.wait()
            if selection is not None:
                manifest = {**manifest, "files": {name: files[name] for name in order}}
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
