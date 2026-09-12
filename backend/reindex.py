"""Rebuild gallery rows from the PNGs' own metadata.

    uv run python -m backend.reindex                  # outputs/ in place
    uv run python -m backend.reindex --from /mnt/d/wsl-outputs   # copy in and index
    uv run python -m backend.reindex --from D:\\backup\\outputs --database D:\\backup\\krea.db
    uv run python -m backend.reindex --dry-run

Every image carries its whole reproduction set in a `krea` text chunk, and
nothing could read it back: krea.db was a single point of failure for an
otherwise self-describing gallery, the WSL and Windows galleries could never be
merged, and a row whose database went missing could never be repaired. This
walks a directory of PNGs, skips the names the index already has (filename is
UNIQUE, which makes the whole thing idempotent), copies files from elsewhere
in with the same `.part`-then-rename discipline `save_image` uses, and inserts
through `storage` so the search index is built with the rows.

With ``--database``, rows come from that gallery's database, so favorites and
durations survive too. Gallery ids are local to a database; reference-image
ids are remapped as the source rows are merged in ascending id order. Without
a database, the favorite cannot come back and duration is recorded as 0 so the
time estimate ignores it. The timestamp is parsed from the filename prefix as
UTC, which is how it was written; failing that, the file's own mtime.

Not `import.py`: `from backend import import` is a SyntaxError, so no test
could follow the house style. Running the file directly works too - see the
sys.path bootstrap below.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image

    from backend import storage
    from backend.config import settings
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        f"Running under: {sys.executable}\n"
        "That interpreter is not the project environment. Use:\n\n"
        "    uv run python -m backend.reindex\n"
    )

# `20260907-235118-42-6a1b2c.png`: the UTC timestamp save_image writes first.
_NAME = re.compile(r"^(\d{8})-(\d{6})-")

# The keys a `krea` block must carry to make a row at all.
_REQUIRED = ("prompt", "steps", "guidance_scale", "seed", "model")

# The original schema's required columns. Newer reproduction columns are read
# when present and become NULL when importing an older gallery.
_DB_REQUIRED = {
    "id", "filename", "prompt", "negative_prompt", "width", "height",
    "steps", "guidance_scale", "mu", "seed", "model_id", "duration_ms",
    "favorite", "created_at",
}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _created_at(path: Path) -> datetime:
    match = _NAME.match(path.name)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _read(path: Path) -> tuple[dict | None, tuple[int, int], Image.Image | None]:
    """The krea block, the size and a copy of the image for its thumbnail.

    `.info`, not `.text`: the latter is a property that decodes every pixel to
    answer a question the header settles.
    """
    with Image.open(path) as image:
        raw = image.info.get("krea")
        size = image.size
        if not raw:
            return None, size, image.copy()
        try:
            meta = json.loads(raw)
        except ValueError:
            return None, size, image.copy()
        if not isinstance(meta, dict) or any(meta.get(key) is None for key in _REQUIRED):
            return None, size, image.copy()
        return meta, size, image.copy()


def _text_keyword(kind: bytes, data: bytes) -> bytes | None:
    """The keyword from a PNG text chunk, without decompressing its value."""
    if kind not in {b"tEXt", b"zTXt", b"iTXt"} or b"\0" not in data:
        return None
    return data.split(b"\0", 1)[0]


def _with_krea_metadata(data: bytes, meta: dict) -> bytes:
    """Replace the krea text chunk without recompressing or changing pixels."""
    if not data.startswith(_PNG_SIGNATURE):
        raise OSError("not a PNG")
    payload = b"krea\0" + json.dumps(meta).encode("ascii")
    replacement = (
        len(payload).to_bytes(4, "big")
        + b"tEXt"
        + payload
        + zlib.crc32(b"tEXt" + payload).to_bytes(4, "big")
    )
    out = bytearray(_PNG_SIGNATURE)
    offset = len(_PNG_SIGNATURE)
    inserted = False
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 12 + length
        if end > len(data):
            raise OSError("truncated PNG")
        kind = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        if _text_keyword(kind, chunk_data) == b"krea":
            offset = end
            continue
        # Pillow reads metadata preceding the image data during Image.open().
        # A text chunk immediately before IEND is legal but stays invisible
        # until load(), which would make the gallery's normal header-only read
        # miss it.
        if kind in {b"IDAT", b"IEND"} and not inserted:
            out.extend(replacement)
            inserted = True
        out.extend(data[offset:end])
        offset = end
        if kind == b"IEND":
            break
    if not inserted:
        raise OSError("PNG has no IEND chunk")
    return bytes(out)


def _copy_in(source: Path, target: Path, meta: dict | None = None) -> None:
    """Copy into outputs under a temporary name, optionally replacing krea metadata."""
    partial = target.with_name(target.name + ".part")
    data = source.read_bytes()
    partial.write_bytes(_with_krea_metadata(data, meta) if meta is not None else data)
    partial.replace(target)


def _source_rows(database: Path) -> list[dict]:
    """Read another app database without migrating or otherwise changing it."""
    uri = database.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=15)
    except sqlite3.Error as exc:
        raise ValueError(f"cannot open source database: {exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
        missing = sorted(_DB_REQUIRED - columns)
        if missing:
            raise ValueError("source database has no usable images table; missing " + ", ".join(missing))
        return [dict(row) for row in conn.execute("SELECT * FROM images ORDER BY id")]
    except sqlite3.Error as exc:
        raise ValueError(f"cannot read source database: {exc}") from exc
    finally:
        conn.close()


def _json_list(value) -> list:
    if value is None:
        return []
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def _database_meta(row: dict, width: int, height: int, digest: str, references: list[int]) -> dict:
    variables = _json_list(row.get("variables"))
    if not variables and row.get("variables"):
        try:
            candidate = json.loads(row["variables"])
            variables = candidate if isinstance(candidate, dict) else None
        except (TypeError, ValueError):
            variables = None
    return {
        "prompt": row["prompt"],
        "negative_prompt": row.get("negative_prompt"),
        "steps": row["steps"],
        "guidance_scale": row["guidance_scale"],
        "true_cfg": row.get("true_cfg"),
        "mu": row.get("mu"),
        "seed": row["seed"],
        "model": row["model_id"],
        "quantization": row.get("quantization"),
        "job": row.get("job_id"),
        "run": row.get("run_id"),
        "size": f"{width}x{height}",
        "variables": variables,
        "hash": digest,
        "reference": references or None,
        "lora": row.get("lora"),
        "lora_scale": row.get("lora_scale") if row.get("lora") else None,
    }


def merge_database(
    source: Path, database: Path, dry_run: bool = False, out=sys.stdout
) -> dict[str, int]:
    """Merge the files and rows of another gallery into the configured one."""
    counts = {
        "indexed": 0,
        "present": 0,
        "missing": 0,
        "invalid": 0,
        "conflicts": 0,
        "unresolved_references": 0,
    }
    rows = _source_rows(database)
    outputs = settings.outputs_dir.resolve()
    source = source.resolve()
    existing = storage.existing_file_ids() if settings.db_path.exists() else {}
    id_map: dict[int, int] = {}
    if not dry_run:
        storage.init_db()

    for row in rows:
        filename = row["filename"]
        source_id = int(row["id"])
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".png"
        ):
            counts["invalid"] += 1
            print(f"  skip  source row {source_id}: unsafe PNG filename {filename!r}", file=out)
            continue
        if filename in existing:
            counts["present"] += 1
            id_map[source_id] = existing[filename]
            continue
        path = source / filename
        if not path.is_file():
            counts["missing"] += 1
            print(f"  skip  {filename}: source image is missing", file=out)
            continue
        target = outputs / filename
        if source != outputs and target.exists():
            counts["conflicts"] += 1
            print(f"  skip  {filename}: a file of that name is already in outputs", file=out)
            continue
        try:
            _, (width, height), image = _read(path)
        except OSError:
            counts["invalid"] += 1
            print(f"  skip  {filename}: not a readable PNG", file=out)
            continue
        if image is None:  # Defensive: database imports do not require embedded metadata.
            counts["invalid"] += 1
            print(f"  skip  {filename}: not a readable PNG", file=out)
            continue

        source_references = [
            value for value in _json_list(row.get("reference")) if isinstance(value, int)
        ]
        references = []
        for reference in source_references:
            mapped = id_map.get(reference)
            if mapped is None:
                counts["unresolved_references"] += 1
            else:
                references.append(mapped)
        digest = storage.content_hash(image)
        meta = _database_meta(row, width, height, digest, references)
        counts["indexed"] += 1
        if dry_run:
            print(f"  would import  {filename}", file=out)
            # Source references always point backwards in a gallery. Keep a
            # placeholder mapping so the dry run reports only genuinely
            # missing/skipped references, not every row it merely did not add.
            id_map[source_id] = source_id
            continue

        values = {
            "filename": filename,
            "prompt": row["prompt"],
            "negative_prompt": row.get("negative_prompt"),
            "width": width,
            "height": height,
            "steps": int(row["steps"]),
            "guidance_scale": float(row["guidance_scale"]),
            "mu": row.get("mu"),
            "seed": int(row["seed"]),
            "model_id": row["model_id"],
            "duration_ms": int(row.get("duration_ms") or 0),
            "favorite": int(bool(row.get("favorite"))),
            "created_at": row.get("created_at") or _created_at(path).isoformat(),
            "job_id": row.get("job_id"),
            "run_id": row.get("run_id"),
            "quantization": row.get("quantization"),
            "variables": row.get("variables"),
            "true_cfg": row.get("true_cfg"),
            "content_hash": digest,
            "reference": json.dumps(references) if references else None,
            "lora": row.get("lora"),
            "lora_scale": row.get("lora_scale") if row.get("lora") else None,
        }
        copied = source != outputs
        try:
            if copied:
                _copy_in(path, target, meta)
            record = storage.import_file(values)
        except BaseException:
            if copied:
                target.unlink(missing_ok=True)
            raise
        id_map[source_id] = record.id
        existing[filename] = record.id
        storage.thumbnail_for(image, filename)
    return counts


def reindex(
    source: Path, dry_run: bool = False, out=sys.stdout, database: Path | None = None
) -> dict[str, int]:
    if database is not None:
        return merge_database(source, database, dry_run=dry_run, out=out)
    counts = {"indexed": 0, "present": 0, "no_metadata": 0, "conflicts": 0}
    source = source.resolve()
    outputs = settings.outputs_dir.resolve()
    in_place = source == outputs
    present = storage.existing_filenames() if settings.db_path.exists() else set()
    if not dry_run:
        storage.init_db()
    for path in sorted(source.glob("*.png")):
        if path.name in present:
            counts["present"] += 1
            continue
        try:
            meta, (width, height), image = _read(path)
        except OSError:  # not a PNG at all, or truncated - UnidentifiedImageError is one
            meta = None
        if meta is None:
            counts["no_metadata"] += 1
            print(f"  skip  {path.name}: no krea metadata", file=out)
            continue
        target = outputs / path.name
        if not in_place and target.exists():
            # A different file under a name the gallery already uses on disk
            # but not in the index: not ours to overwrite.
            counts["conflicts"] += 1
            print(f"  skip  {path.name}: a different file of that name is already in outputs", file=out)
            continue
        counts["indexed"] += 1
        if dry_run:
            print(f"  would index  {path.name}", file=out)
            continue
        if not in_place:
            _copy_in(path, target)
        # The hash of the pixels as decoded here, not the block's copy of it:
        # the file is the truth, and a PNG from before the hash has no copy.
        storage.index_file(
            path.name,
            meta,
            width,
            height,
            _created_at(path),
            digest=storage.content_hash(image) if image is not None else None,
        )
        if image is not None:
            storage.thumbnail_for(image, path.name)
        present.add(path.name)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.reindex",
        description="Rebuild gallery rows from the PNGs' own metadata; idempotent.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        metavar="DIR",
        help="a directory of this app's PNGs to copy into the gallery and index "
        "(default: the gallery's own outputs directory, in place)",
    )
    parser.add_argument(
        "--database",
        "--db",
        metavar="FILE",
        help="the krea.db belonging to --from; preserves favorites and durations and remaps references",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="report what would be indexed, change nothing"
    )
    args = parser.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else settings.outputs_dir
    if not source.is_dir():
        print(f"{source} is not a directory", file=sys.stderr)
        return 2
    database = Path(args.database).expanduser() if args.database else None
    if database is not None and not args.source:
        print("--database requires --from so its matching outputs folder is explicit", file=sys.stderr)
        return 2
    if database is not None and not database.is_file():
        print(f"{database} is not a file", file=sys.stderr)
        return 2

    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print(f"source   : {source}{' (in place)' if source.resolve() == settings.outputs_dir.resolve() else ''}")
    if database is not None:
        print(f"source db: {database}")
    print()
    started = time.perf_counter()
    try:
        counts = reindex(source, dry_run=args.dry_run, database=database)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if database:
        verb = "Would import" if args.dry_run else "Imported"
    else:
        verb = "Would index" if args.dry_run else "Indexed"
    summary = f"\n{verb} {counts['indexed']} file(s); {counts['present']} already in the index"
    if database:
        summary += (
            (f", {counts['missing']} missing image(s)" if counts["missing"] else "")
            + (f", {counts['invalid']} invalid row/file(s)" if counts["invalid"] else "")
            + (
                f", {counts['unresolved_references']} unresolved reference(s)"
                if counts["unresolved_references"]
                else ""
            )
        )
    else:
        summary += f", {counts['no_metadata']} without metadata" if counts["no_metadata"] else ""
    summary += f", {counts['conflicts']} name conflict(s)" if counts["conflicts"] else ""
    print(summary + f". {time.perf_counter() - started:.1f}s.")
    if args.dry_run:
        print("Dry run - nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
