"""Verified, portable gallery archives and staged, non-destructive imports."""
from __future__ import annotations

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
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

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
def _open_archive(path: Path, password: str | None):
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
    with bundle_crypto.DecryptingReader(path, password) as reader, zipfile.ZipFile(reader) as archive:
        yield archive


def _inspect_source(row: dict, data: bytes):
    return _validate_source_row(row, io.BytesIO(data), name=row["filename"])


def _verify_archive(path: Path, destination: Path | None = None, *, password: str | None = None,
                    rows: list[dict] | None = None, inspect=_inspect_source,
                    extract: Callable[[str], bool] | None = None,
                    progress: Progress = _quiet) -> tuple[dict, list[dict] | None, dict]:
    """Read every member once, verifying ZIP CRC, the manifest SHA-256 and, for
    a password-protected bundle, every chunk's tag.

    The database member is read first. Each tracked PNG (from `rows`, or from
    the database unpacked into `destination`) is passed to `inspect` from the
    bytes just read, on the decode pool. `extract` may decline to write a
    member that will never be installed. Returns the manifest, the rows and
    each tracked member's inspection result.
    """
    try:
        with _open_archive(path, password) as archive, _Pool() as pool:
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
            manifest = json.loads(archive.read(_MANIFEST))
            if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT or manifest.get("version") != 1:
                raise BundleError("this is not a supported Local Diffusion bundle")
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != set(names) - {_MANIFEST} or _DATABASE not in files:
                raise BundleError("the bundle manifest does not match its files")
            if destination is not None:
                required = sum(info.file_size for info in infos)
                if required > shutil.disk_usage(destination).free:
                    raise BundleError("not enough free space to unpack this bundle")

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
                    target.parent.mkdir(parents=True, exist_ok=True)
                row = tracked.get(name)
                held = [] if row else None
                output = target.open("xb") if target else None
                try:
                    with archive.open(info) as source:
                        while chunk := source.read(_CHUNK):
                            digest.update(chunk)
                            if output:
                                output.write(chunk)
                            if held is not None:
                                held.append(chunk)
                    # Tracked PNGs are rewritten (and synced) before
                    # installation; everything else may be installed as is.
                    if output and name != _DATABASE and not row:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    if output:
                        output.close()
                if digest.hexdigest() != expected["sha256"]:
                    raise BundleError(f"the bundle checksum failed for {name}")
                if row:
                    data = b"".join(held)
                    pool.submit(name, _png_weight(data), inspect, row, data)
                if name == _DATABASE and rows is None and destination is not None:
                    rows = _rows(destination / _DATABASE)
                    tracked = track(rows)
                progress("verifying", done, len(order))
            results = pool.results(["images/" + row["filename"] for row in rows or []])
            return manifest, rows, results
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeError, EOFError) as exc:
        raise BundleError(f"the bundle is damaged: {exc}") from exc
    except bundle_crypto.CryptoError as exc:
        raise BundleError(str(exc)) from exc


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


def _install(source: Path, target: Path, installed: list[Path]) -> None:
    """Move a synced staged file into place, never replacing an existing file."""
    try:
        if os.name == "nt":
            os.rename(source, target)  # refuses an existing target on Windows
        else:
            os.link(source, target)  # rename would replace one on POSIX
        installed.append(target)
        return
    except FileExistsError:
        raise
    except OSError:
        pass  # another volume, or no hard links: copy instead
    with target.open("xb") as dest:
        installed.append(target)
        with source.open("rb") as src:
            shutil.copyfileobj(src, dest, _CHUNK)
        dest.flush()
        os.fsync(dest.fileno())


def _unchanged(path: Path, saved: dict) -> bool:
    return path.stat().st_size == saved["size"] and _hash(path) == saved["sha256"]


def export_bundle(expected: int, password: str | None = None, progress: Progress = _quiet) -> dict:
    settings = _settings()
    with gallery_lock.hold(settings.db_path):
        before = preview()
        if before["records"] != expected:
            raise BundleChanged("the gallery changed; review its new count before creating a bundle")
        files = _inventory()
        directory = _directory()
        directory.mkdir(parents=True, exist_ok=True)
        name = f"local-diffusion-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
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
                    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                        for done, (member, path) in enumerate(members, 1):
                            digest = hashlib.sha256()
                            size = 0
                            with path.open("rb") as source, archive.open(member, "w", force_zip64=True) as dest:
                                while chunk := source.read(_CHUNK):
                                    digest.update(chunk)
                                    size += len(chunk)
                                    dest.write(chunk)
                            manifest["files"][member] = {"size": size, "sha256": digest.hexdigest()}
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
                _verify_archive(partial, password=password, rows=rows, progress=progress)
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
                for _, source in files:
                    try:
                        _remove_file(source)
                    except OSError:
                        result["retained"].append(str(source.resolve()))
                remaining = {p.name for _, p in files if p.suffix.lower() == ".png" and p.exists()}
                result["deleted_ids"] = [r["id"] for r in rows if r["filename"] not in remaining]
                # The surviving rows remain accurate even if Windows refuses
                # the subsequent replacement of the original database file.
                with storage._session() as conn:
                    conn.executemany("DELETE FROM images WHERE id = ?", [(i,) for i in result["deleted_ids"]])
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


def _pixels(path, *, name: str | None = None) -> tuple[str, int, int]:
    name = name or path.name
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise BundleError(f"not a PNG image: {name}")
            image.load()
            return storage.content_hash(image), image.width, image.height
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


def _validate_source_row(row: dict, source, *, name: str | None = None) -> tuple[str, int, int]:
    pixels = _pixels(source, name=name)
    if pixels[1:] != (row["width"], row["height"]):
        raise BundleError(f"image dimensions disagree with the database: {row['filename']}")
    _references(row)
    refine = row.get("refine")
    if refine is not None and (type(refine) is not int or refine <= 0):
        raise BundleError(f"invalid refine source on image {row['id']}")
    values = {key: row.get(key) for key in storage._COLUMNS}
    values.update(id=row["id"], content_hash=pixels[0])
    try:
        storage._row_to_record(values)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"invalid metadata for {row['filename']}: {exc}") from exc
    return pixels


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
        _inventory()  # A partial write or linked destination is not an import target.
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".bundle-import-", dir=settings.db_path.parent) as work:
            work = Path(work)
            current = _rows(settings.db_path)
            existing = {r["filename"].casefold(): r for r in current}

            def present_file(relative: str) -> Path | None:
                known = existing.get(relative.casefold())
                target = settings.outputs_dir / known["filename"] if known else None
                return target if known and known["filename"] == relative and target.is_file() else None

            def inspect(row: dict, data: bytes):
                pixels = _inspect_source(row, data)
                target = present_file(row["filename"])
                same = False
                if target is not None:
                    # Identical pixel-deciding chunks skip a second decode;
                    # anything else is compared by decoding the existing image.
                    with target.open("rb") as stream:
                        known = _critical_digest(stream)
                    same = (known is not None and known == _critical_digest(io.BytesIO(data))) or _pixels(target) == pixels
                return pixels, same

            def extract(member: str) -> bool:
                # An existing image wins on repeated import, so its bundle copy
                # is verified and decoded from memory but never unpacked.
                bits = member.split("/")
                return not (len(bits) == 2 and bits[1].lower().endswith(".png") and present_file(bits[1]))

            manifest, source_rows, inspected = _verify_archive(
                Path(path), work, password=password, inspect=inspect, extract=extract, progress=progress)
            stage_db = work / "merged.sqlite3"
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
                progress("merging", 0, len(source_rows))
                for row in source_rows:
                    name = row["filename"]
                    (digest, _, _), same = inspected["images/" + name]
                    known = existing.get(name.casefold())
                    if known:
                        if not same:
                            raise BundleError(f"another image already uses the filename {name}")
                        for key in ("prompt", "negative_prompt", "steps", "guidance_scale", "mu", "seed", "model_id"):
                            if known[key] != row[key]:
                                raise BundleError(f"different image metadata already uses the filename {name}")
                        mapping[row["id"]] = known["id"]
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
                for original, values in pending:
                    try:
                        inserted = storage._insert_row(conn, values)
                        if inserted["id"] != values["id"]:
                            raise BundleError("the staged image id sequence changed")
                        storage._row_to_record(inserted)
                        image_path = work / "images" / values["filename"]
                        _rewrite_metadata(image_path, _metadata(values, by_id))
                    except (TypeError, ValueError, sqlite3.Error) as exc:
                        raise BundleError(f"invalid metadata for {values['filename']}: {exc}") from exc
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
                        # Rewritten PNGs are compared as staged; other members
                        # are byte-identical to their verified manifest entry.
                        saved = manifest["files"][member]
                        if source.stat().st_size != actual.stat().st_size or (
                                _hash(actual) != (_hash(source) if source.suffix.lower() == ".png" else saved["sha256"])):
                            raise BundleError(f"another file already uses the filename {relative}")
                        continue
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
                return {"imported": len(pending), "present": present,
                        "images": sum(name.startswith("images/") and name.lower().endswith(".png") for name in manifest["files"]),
                        "unresolved_references": unresolved}
            finally:
                if conn is not None:
                    conn.close()
