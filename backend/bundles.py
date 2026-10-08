"""Verified, portable gallery archives and staged, non-destructive imports."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

from . import gallery_lock, reindex, storage


class BundleError(ValueError):
    pass


class BundleChanged(BundleError):
    pass


_FORMAT = "local-diffusion-gallery"
_DATABASE = "database.sqlite3"
_MANIFEST = "manifest.json"
_CHUNK = 1024 * 1024


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
    if not _basename(name) or Path(name).suffix.lower() != ".zip":
        raise BundleError("choose a bundle filename ending in .zip")
    path = _directory() / name
    if _is_link(path):
        raise BundleError("a bundle must be a regular file")
    return path


def list_bundles() -> list[dict]:
    directory = _directory()
    if not directory.exists():
        return []
    return [
        {"name": p.name, "bytes": p.stat().st_size, "created_at": p.stat().st_mtime}
        for p in sorted(directory.glob("*.zip"), reverse=True)
        if p.is_file() and not _is_link(p)
    ]


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


def _verify_archive(path: Path, destination: Path | None = None) -> dict:
    """Read every member, verifying both ZIP CRC and the manifest SHA-256."""
    try:
        with zipfile.ZipFile(path) as archive:
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
            for name, expected in files.items():
                info = archive.getinfo(name)
                if (not isinstance(expected, dict) or type(expected.get("size")) is not int
                        or expected["size"] != info.file_size or not isinstance(expected.get("sha256"), str)):
                    raise BundleError(f"invalid size or checksum for {name}")
                digest = hashlib.sha256()
                target = destination.joinpath(*PurePosixPath(name).parts) if destination else None
                if target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open("xb") if target else None
                try:
                    with archive.open(info) as source:
                        while chunk := source.read(_CHUNK):
                            digest.update(chunk)
                            if output:
                                output.write(chunk)
                finally:
                    if output:
                        output.close()
                if digest.hexdigest() != expected["sha256"]:
                    raise BundleError(f"the bundle checksum failed for {name}")
            if destination:
                rows = _rows(destination / _DATABASE)
                if len(rows) != manifest.get("records"):
                    raise BundleError("the manifest's record count does not match its database")
                for row in rows:
                    if "images/" + row["filename"] not in files:
                        raise BundleError(f"the bundle is missing {row['filename']}")
            return manifest
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeError, EOFError) as exc:
        raise BundleError(f"the bundle is damaged: {exc}") from exc


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
    temp.replace(path)


def export_bundle(expected: int) -> dict:
    settings = _settings()
    with gallery_lock.hold(settings.db_path):
        before = preview()
        if before["records"] != expected:
            raise BundleChanged("the gallery changed; review its new count before creating a bundle")
        files = _inventory()
        directory = _directory()
        directory.mkdir(parents=True, exist_ok=True)
        name = f"local-diffusion-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.zip"
        final = bundle_path(name)
        partial = final.with_suffix(".zip.part")
        with tempfile.TemporaryDirectory(prefix=".bundle-", dir=settings.db_path.parent) as work:
            work = Path(work)
            snapshot = work / _DATABASE
            _backup(settings.db_path, snapshot)
            rows = _rows(snapshot)
            manifest = {"format": _FORMAT, "version": 1, "records": len(rows), "files": {}}
            try:
                with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                    for member, path in [(_DATABASE, snapshot), *files]:
                        digest = hashlib.sha256()
                        size = 0
                        with path.open("rb") as source, archive.open(member, "w", force_zip64=True) as dest:
                            while chunk := source.read(_CHUNK):
                                digest.update(chunk)
                                size += len(chunk)
                                dest.write(chunk)
                        manifest["files"][member] = {"size": size, "sha256": digest.hexdigest()}
                    archive.writestr(_MANIFEST, json.dumps(manifest, ensure_ascii=True))
                with partial.open("r+b") as stream:
                    os.fsync(stream.fileno())
                _verify_archive(partial)
                # Validate the archived originals themselves, so a damaged
                # tracked PNG can never be cleared into an unusable import.
                # ZipExtFile streams avoid a second extracted gallery on disk.
                with zipfile.ZipFile(partial) as archive:
                    for row in rows:
                        with archive.open("images/" + row["filename"]) as source:
                            _validate_source_row(row, source, name=row["filename"])
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
                for member, source in current_files:
                    saved = manifest["files"][member]
                    if source.stat().st_size != saved["size"] or _hash(source) != saved["sha256"]:
                        raise BundleError(f"{source.name} changed while its bundle was written; nothing was cleared")
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


def import_bundle(path: Path) -> dict:
    settings = _settings()
    with gallery_lock.hold(settings.db_path):
        _inventory()  # A partial write or linked destination is not an import target.
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".bundle-import-", dir=settings.db_path.parent) as work:
            work = Path(work)
            manifest = _verify_archive(Path(path), work)
            source_rows = _rows(work / _DATABASE)
            current = _rows(settings.db_path)
            existing = {r["filename"].casefold(): r for r in current}
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
                for row in source_rows:
                    name = row["filename"]
                    digest, width, height = _validate_source_row(row, work / "images" / name)
                    known = existing.get(name.casefold())
                    if known:
                        target = settings.outputs_dir / known["filename"]
                        if known["filename"] != name or not target.is_file() or _pixels(target) != (digest, width, height):
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
                        if _hash(actual) != _hash(source):
                            raise BundleError(f"another file already uses the filename {relative}")
                        continue
                    targets.append((source, target))
                _checkpoint()
                try:
                    for source, target in targets:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        # Exclusive creation also protects an external file that
                        # appeared after validation; rollback removes only ours.
                        with target.open("xb") as dest:
                            installed.append(target)
                            with source.open("rb") as src:
                                shutil.copyfileobj(src, dest, _CHUNK)
                            dest.flush()
                            os.fsync(dest.fileno())
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
