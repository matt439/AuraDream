"""Rebuild gallery rows from the PNGs' own metadata.

    uv run python -m backend.reindex                  # outputs/ in place
    uv run python -m backend.reindex --from /mnt/d/wsl-outputs   # copy in and index
    uv run python -m backend.reindex --from D:\\backup\\outputs --database D:\\backup\\auradream.db
    uv run python -m backend.reindex --dry-run

Every image carries its whole reproduction set in an `auradream` text chunk, and
nothing could read it back: auradream.db was a single point of failure for an
otherwise self-describing gallery, the WSL and Windows galleries could never be
merged, and a row whose database went missing could never be repaired. This
walks a directory of PNGs, skips the names the index already has (filename is
UNIQUE, which makes the whole thing idempotent), copies files from elsewhere
in with the same `.part`-then-rename discipline `save_image` uses, and inserts
through `storage` so the search index is built with the rows.

With ``--database``, rows come from that gallery's database, so favorites and
durations survive too. Gallery ids are local to a database; reference-image
and refine-source ids are remapped as the source rows are merged in ascending
id order. PNG-only imports use each source's recorded filename, pixel hash and
size to recover its new id; legacy links without this provenance are dropped
and counted, never treated as destination ids. Without a database, the
favorite cannot come back and duration is recorded as 0 so the
time estimate ignores it; nor can the recycle bin, so an image that was in
it comes back live. The timestamp is parsed from the filename prefix as
UTC, which is how it was written; failing that, the file's own mtime.

An inpaint's mask is a picture of its own, named by its hash and so never
remapped. Masks come back from `masks/*.png` beside the images (a file
whose pixels are not the hash it is named by is skipped and counted), from
the source database's `masks` table with them, and - where the file is gone -
from the copy every result carries in its `auradream-mask` chunk. A
result whose mask is found nowhere keeps its hash, which still says what
ran, and is counted.

Not `import.py`: `from backend import import` is a SyntaxError, so no test
could follow the house style. Running the file directly works too - see the
sys.path bootstrap below.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
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

    from backend import gallery_lock, storage
    from backend.config import settings
    from backend.execution import ExecutionSpec, canonical_execution, execution_chunk, execution_from_json, execution_key
    from backend.instruction_edit import StoredInstructionEdit, descriptor_from_image, edit_key
    from backend.metadata import (
        PNG_METADATA_KEY, PNG_METADATA_KEYS, canonical_edit, edit_chunk, edit_from_info,
        edit_from_json, edit_from_sources, embedded_mask, embedded_metadata, require_header_metadata,
        validate_edit_record,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        f"Running under: {sys.executable}\n"
        "That interpreter is not the project environment. Use:\n\n"
        "    uv run python -m backend.reindex\n"
    )

# `20260907-235118-42-6a1b2c.png`: the UTC timestamp save_image writes first.
_NAME = re.compile(r"^(\d{8})-(\d{6})-")

# The keys an embedded metadata block must carry to make a row at all.
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


def _read(path: Path, *, require_metadata: bool = True) -> tuple[dict | None, tuple[int, int], Image.Image | None]:
    """The embedded record, the size and a copy of the image for its thumbnail.

    `.info`, not `.text`: the latter is a property that decodes every pixel to
    answer a question the header settles.
    """
    with path.open("rb") as stream:
        require_header_metadata(stream)
        with Image.open(stream) as image:
            raw = embedded_metadata({key: image.info.get(key) for key in PNG_METADATA_KEYS})
            size = image.size
            if not raw:
                return None, size, image.copy()
            try:
                meta = json.loads(raw)
            except ValueError:
                return None, size, image.copy()
            if not isinstance(meta, dict) or (require_metadata and any(meta.get(key) is None for key in _REQUIRED)):
                return None, size, image.copy()
            return meta, size, image.copy()


def _execution_from_sources(row: dict, meta: dict | None) -> ExecutionSpec | None:
    """Read stored recipes without granting generation availability.

    An older SQL schema or PNG may omit execution entirely: recover the
    other source's recipe then. Explicit NULL/empty is an Eager statement,
    so two present fields must agree, including Eager versus a recipe.
    Validate both values even when one is absent or empty.
    """
    database = execution_from_json(row.get("execution"))
    embedded = canonical_execution(meta.get("execution") if meta is not None else None)
    if "execution" in row and meta is not None and "execution" in meta and database != embedded:
        raise ValueError("execution recipes disagree between the database and PNG")
    return database if "execution" in row else embedded


def _execution_from_info(row: dict, info: dict | None) -> ExecutionSpec | None:
    """Strict recipe reconciliation from PNG header metadata, without decoding pixels."""
    raw = embedded_metadata(info) if info is not None else None
    metadata = json.loads(raw) if raw else None
    if raw is not None and not isinstance(metadata, dict):
        raise ValueError("PNG metadata must be a JSON object")
    return _execution_from_sources(row, metadata)


def _edit_metadata(path: Path) -> dict | None:
    """Read the app header without decoding pixels, before recovery mutation."""
    try:
        with path.open("rb") as stream:
            require_header_metadata(stream)
            with Image.open(stream) as image:
                raw = embedded_metadata({key: image.info.get(key) for key in PNG_METADATA_KEYS})
                value = json.loads(raw) if raw else None
                return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        # Ordinary unreadable images retain the existing skip/report behavior.
        # A SQL edit with no readable PNG representation is refused below.
        return None


def _preflight_edits(source: Path, rows: list[dict] | None = None) -> None:
    """Refuse malformed/contradictory intent before rows, copies or masks change."""
    if rows is None:
        for path in source.glob("*.png"):
            meta = _edit_metadata(path)
            if meta is not None:
                validate_edit_record(canonical_edit(meta.get("edit")), meta)
        return
    for row in rows:
        stored = edit_from_json(row.get("edit"))
        filename = row.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            continue  # The normal row validator reports unsafe ordinary names.
        path = source / filename
        if path.is_file():
            meta = _edit_metadata(path)
            if meta is not None or stored is not None:
                edit_from_sources(row, meta or {})


def _preflight_retained_edits(filenames, existing: dict[str, int], outputs: Path) -> None:
    for filename in filenames:
        if filename not in existing:
            continue
        known = storage.get_image(existing[filename], include_binned=True)
        if known is None:
            continue
        path = outputs / filename
        # A missing ordinary destination retains its normal conflict behavior;
        # an edit cannot claim its durable PNG representation is still present.
        if path.is_file() or known.edit is not None:
            edit_from_sources({
                "edit": edit_key(known.edit), "model_id": known.model_id,
                "width": known.width, "height": known.height, "prompt": known.prompt,
            }, _edit_metadata(path) or {})


def _proves_edit(path: Path, stored: StoredInstructionEdit) -> bool:
    """A source's full decoded RGB identity, never an old row's short digest."""
    try:
        if path.is_symlink() or path.is_junction():
            return False
        with Image.open(path) as image:
            return descriptor_from_image(path.name, image) == stored.edit_source
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def _remapped_edit(stored: StoredInstructionEdit | None, source: int | None, counts: dict[str, int]) -> StoredInstructionEdit | None:
    if stored is None:
        return None
    if source is None:
        counts["unresolved_edits"] += 1
    return StoredInstructionEdit(source=source, recipe=stored.recipe, edit_source=stored.edit_source)


def _same_edit(left: StoredInstructionEdit | None, right: StoredInstructionEdit | None) -> bool:
    """Local ids may differ across galleries; intent and complete proof cannot."""
    if left is None or right is None:
        return left is right
    return (left.recipe, left.edit_source) == (right.recipe, right.edit_source)


def _dependency_order(rows: list[dict], cycles: set[int]) -> list[dict]:
    """Make a complete id map for forward edit sources as well as old links.

    Cycles cannot prove a link: the missing map entry stays unresolved. Visit
    ordinary reference/refine dependencies too, so an edit-source reorder
    cannot break an otherwise earlier ordinary dependency.
    """
    by_id = {row["id"]: row for row in rows}
    ordered: list[dict] = []
    visited: set[int] = set()
    visiting: set[int] = set()

    def visit(row: dict) -> None:
        identifier = row["id"]
        if identifier in visiting:
            cycles.update(visiting)
            return
        if identifier in visited:
            return
        visiting.add(identifier)
        stored = edit_from_json(row.get("edit"))
        links = [*_json_list(row.get("reference")), row.get("refine"), stored.source if stored else None]
        for source_id in links:
            if type(source_id) is int and source_id in by_id:
                visit(by_id[source_id])
        visiting.remove(identifier)
        visited.add(identifier)
        ordered.append(row)

    for row in rows:
        visit(row)
    return ordered


def _text_keyword(kind: bytes, data: bytes) -> bytes | None:
    """The keyword from a PNG text chunk, without decompressing its value."""
    if kind not in {b"tEXt", b"zTXt", b"iTXt"} or b"\0" not in data:
        return None
    return data.split(b"\0", 1)[0]


def _with_auradream_metadata(data: bytes, meta: dict) -> bytes:
    """Replace app metadata without recompressing or changing pixels."""
    if not data.startswith(_PNG_SIGNATURE):
        raise OSError("not a PNG")
    payload = PNG_METADATA_KEY.encode("ascii") + b"\0" + json.dumps(meta).encode("ascii")
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
        if _text_keyword(kind, chunk_data) in {
            key.encode("ascii") for key in PNG_METADATA_KEYS
        }:
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
    """Copy into outputs under a temporary name, optionally replacing app metadata."""
    partial = target.with_name(target.name + ".part")
    data = source.read_bytes()
    partial.write_bytes(
        _with_auradream_metadata(data, meta) if meta is not None else data
    )
    partial.replace(target)


def _open_mask(source, digest: str) -> Image.Image | None:
    """The mask `source` (a path, or a PNG's bytes) holds, where it is the
    mask `digest` names: one `L` channel whose `storage.mask_hash` - its size
    and its levels - is that hash. None for anything else, a file that is
    not a PNG included."""
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            if image.format != "PNG" or image.mode != "L":
                return None
            image.load()
            return image.copy() if storage.mask_hash(image) == digest else None
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _embedded_mask(image: Image.Image | None, digest: str) -> Image.Image | None:
    """The mask a result carries in its own `auradream-mask` chunk (or
    Local Diffusion's legacy one), where it is the one its record names."""
    raw = embedded_mask(image.info) if image is not None else None
    if raw is None:
        return None
    try:
        data = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None
    return _open_mask(data, digest)


class _Masks:
    """The masks a reindex can find, and the ones it put back.

    `held` answers whether this gallery has a mask (its row and its file),
    from what this run restored and, against a gallery that exists, from
    storage - never creating one, which a dry run promises not to do.
    `restore` keeps a verified mask through `storage.store_mask`, the one
    writer of the `masks` table and directory.
    """

    def __init__(self, counts: dict[str, int], dry_run: bool, out):
        self.counts = counts
        self.dry_run = dry_run
        self.out = out
        self.restored: set[str] = set()

    def held(self, digest: str) -> bool:
        if digest in self.restored:
            return True
        if not settings.db_path.exists():
            return False
        try:
            storage.mask_path(digest)
        except ValueError:
            return False
        return True

    def restore(self, digest: str, mask: Image.Image) -> None:
        if not self.dry_run:
            storage.store_mask(mask)
        self.restored.add(digest)
        self.counts["masks"] += 1

    def from_directory(self, directory: Path) -> None:
        """Every `masks/*.png` named by the hash of its own pixels."""
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.png")):
            digest = path.stem
            if storage._MASK_HASH.fullmatch(digest) is None or (mask := _open_mask(path, digest)) is None:
                self.counts["bad_masks"] += 1
                print(f"  skip  masks/{path.name}: not the mask its name says", file=self.out)
                continue
            if not self.held(digest):
                self.restore(digest, mask)

    def resolve(
        self, digest: str | None, directory: Path, image: Image.Image | None,
        outpaint=None, size: tuple[int, int] | None = None,
    ) -> None:
        """Make sure a result's mask is held: from its file in `directory`,
        else from the result's own embedded copy, else - an outpaint's - built
        again from its spec at the result's `size` where it hashes to the
        name; counted where none of them."""
        if digest is None or self.held(digest):
            return
        path = directory / f"{digest}.png"
        mask = _open_mask(path, digest) if path.is_file() else None
        if mask is None:
            mask = _embedded_mask(image, digest)
        if mask is None and outpaint is not None and size is not None:
            mask = storage.rebuilt_outpaint_mask(size[0], size[1], outpaint, digest)
            if mask is not None:
                self.counts["outpaint_masks_rebuilt"] += 1
        if mask is None:
            self.counts["unresolved_masks"] += 1
            print(f"  note  mask {digest[:16]} was found neither in masks/ nor in its result", file=self.out)
            return
        self.restore(digest, mask)


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


def _outpaint(mask: str | None, spec, image: Image.Image | None, width: int, height: int,
              source_size: tuple[int, int] | None, counts: dict[str, int]):
    """A result's outpaint as this reindex records it: its spec only beside
    a mask (and so a source); where none came with it, one an older build's
    import stripped, from the A1111 line, where the rebuilt mask is the
    row's (`storage.recovered_outpaint`), counted."""
    if mask is None:
        return None
    if spec is not None:
        return spec
    parameters = image.info.get("parameters") if image is not None else None
    recovered = storage.recovered_outpaint(parameters, width, height, mask, source_size)
    if recovered is not None:
        counts["outpaints_recovered"] += 1
    return recovered


def _provenance_size(provenance) -> tuple[int, int] | None:
    if not isinstance(provenance, dict):
        return None
    width, height = provenance.get("width"), provenance.get("height")
    return (width, height) if type(width) is int and type(height) is int else None


def _database_meta(
    row: dict, width: int, height: int, digest: str, references: list[int], refine: int | None,
    mask: str | None = None, outpaint=None,
) -> dict:
    variables = _json_list(row.get("variables"))
    if not variables and row.get("variables"):
        try:
            candidate = json.loads(row["variables"])
            variables = candidate if isinstance(candidate, dict) else None
        except (TypeError, ValueError):
            variables = None
    upscaler = storage._upscaler(
        row.get("upscaler"), source=refine, model_id=row.get("model_id"), mask=mask, outpaint=outpaint,
    )
    crop_anchor = storage._crop_anchor(
        row.get("crop_anchor") if refine is not None else None,
        source=refine, mask=mask, outpaint=outpaint,
    )
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
        "reference_sources": [storage.image_provenance(i) for i in references] or None,
        "lora": row.get("lora"),
        "lora_scale": row.get("lora_scale") if row.get("lora") else None,
        "refine": refine,
        "refine_source": storage.image_provenance(refine),
        "strength": row.get("strength") if refine is not None else None,
        "mask": mask,
        "mask_source": storage.mask_provenance(mask, width, height),
        "mask_blend": "overlay" if mask is not None else None,
        "outpaint": storage.outpaint_chunk(outpaint),
        "outpaint_fill": outpaint.fill if outpaint is not None else None,
        "batch": storage._batch(row.get("batch")),
        **({"crop_anchor": crop_anchor} if crop_anchor is not None else {}),
        **({"upscaler": upscaler} if upscaler is not None else {}),
        **execution_chunk(execution_from_json(row.get("execution"))),
        **edit_chunk(edit_from_json(row.get("edit"))),
        **{
            key: storage._peak(row.get(key))
            for key in ("vram_peak_allocated", "vram_peak_reserved")
            if storage._peak(row.get(key)) is not None
        },
    }


@gallery_lock.protected(lambda: settings.db_path)
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
        "unresolved_edits": 0,
        "masks": 0,
        "bad_masks": 0,
        "unresolved_masks": 0,
        "outpaint_malformed": 0,
        "outpaints_recovered": 0,
        "outpaint_masks_rebuilt": 0,
    }
    rows = _source_rows(database)
    sizes = {row["id"]: (row["width"], row["height"]) for row in rows}
    outputs = settings.outputs_dir.resolve()
    source = source.resolve()
    _preflight_edits(source, rows)
    edit_cycles: set[int] = set()
    rows = _dependency_order(rows, edit_cycles)
    source_rows = {row["id"]: row for row in rows}
    existing = storage.existing_file_ids() if settings.db_path.exists() else {}
    _preflight_retained_edits((row["filename"] for row in rows if isinstance(row.get("filename"), str)), existing, outputs)
    id_map: dict[int, int] = {}
    if not dry_run:
        storage.init_db()
    # The masks first, every one the other gallery's directory holds, as
    # its `masks` table would list them: a mask is named by its pixels, so
    # its file is the record, and a result below then finds it held.
    masks = _Masks(counts, dry_run, out)
    masks.from_directory(source / storage.MASKS_DIR)

    for row in rows:
        filename = row["filename"]
        source_id = int(row["id"])
        try:
            execution_from_json(row.get("execution"))
            edit_from_json(row.get("edit"))
            storage._crop_anchor(
                row.get("crop_anchor"), source=row.get("refine"),
                mask=row.get("mask"), outpaint=row.get("outpaint"),
            )
        except ValueError as exc:
            counts["invalid"] += 1
            print(f"  skip  source row {source_id}: {exc}", file=out)
            continue
        if row.get("upscaler") is not None and storage._upscaler(
            row["upscaler"], source=row.get("refine"), model_id=row.get("model_id"),
            mask=row.get("mask"), outpaint=row.get("outpaint"),
        ) is None:
            counts["invalid"] += 1
            print(f"  skip  source row {source_id}: invalid upscaler for its Refine pass", file=out)
            continue
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".png"
        ):
            counts["invalid"] += 1
            print(f"  skip  source row {source_id}: unsafe PNG filename {filename!r}", file=out)
            continue
        path = source / filename
        if not path.is_file():
            counts["missing"] += 1
            print(f"  skip  {filename}: source image is missing", file=out)
            continue
        target = outputs / filename
        if filename not in existing and source != outputs and target.exists():
            counts["conflicts"] += 1
            print(f"  skip  {filename}: a file of that name is already in outputs", file=out)
            continue
        try:
            png_meta, (width, height), image = _read(path, require_metadata=False)
        except (OSError, ValueError) as exc:
            counts["invalid"] += 1
            print(f"  skip  {filename}: not a readable PNG: {exc}", file=out)
            continue
        if image is None:  # Defensive: database imports do not require embedded metadata.
            counts["invalid"] += 1
            print(f"  skip  {filename}: not a readable PNG", file=out)
            continue
        try:
            execution = _execution_from_sources(row, png_meta)
            stored_edit = edit_from_sources(row, png_meta or {})
        except ValueError as exc:
            counts["invalid"] += 1
            print(f"  skip  {filename}: invalid execution metadata: {exc}", file=out)
            continue
        if filename in existing:
            try:
                known = storage.get_image(existing[filename], include_binned=True)
                if known is None:
                    raise ValueError("the retained row disappeared")
                with target.open("rb") as stream:
                    require_header_metadata(stream)
                    with Image.open(stream) as kept:
                        if kept.format != "PNG":
                            raise ValueError("the retained image is not a PNG")
                        retained = _execution_from_info({"execution": execution_key(known.execution)}, kept.info)
                        retained_edit = edit_from_info({
                            "edit": edit_key(known.edit), "model_id": known.model_id,
                            "width": known.width, "height": known.height, "prompt": known.prompt,
                        }, {key: kept.info.get(key) for key in PNG_METADATA_KEYS})
                if retained != execution:
                    raise ValueError("different execution already uses that filename")
                if not _same_edit(retained_edit, stored_edit) or (
                    stored_edit is not None and known.prompt != row["prompt"]
                ):
                    raise ValueError("different instruction edit already uses that filename")
            except (OSError, ValueError) as exc:
                counts["conflicts"] += 1
                print(f"  skip  {filename}: cannot keep its existing execution metadata: {exc}", file=out)
                continue
            counts["present"] += 1
            id_map[source_id] = existing[filename]
            continue
        # A recovered PNG-only recipe becomes the row's value too. The
        # source database is read-only and is never migrated or repaired.
        edit_source = None
        if stored_edit is not None and stored_edit.source is not None and source_id not in edit_cycles:
            original_source = source_rows.get(stored_edit.source)
            mapped_source = id_map.get(stored_edit.source)
            if (
                original_source is not None and original_source["filename"] == stored_edit.edit_source.filename
                and mapped_source is not None and _proves_edit(source / original_source["filename"], stored_edit)
                and (dry_run or _proves_edit(outputs / original_source["filename"], stored_edit))
            ):
                edit_source = mapped_source
        remapped_edit = _remapped_edit(stored_edit, edit_source, counts)
        row = {**row, "execution": execution_key(execution), "edit": edit_key(remapped_edit)}

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
        # A refine source is a gallery id too, and points backwards the same
        # way: remapped with the references, and dropped - with its strength,
        # and counted - where the image it names did not come across.
        refine = row.get("refine")
        if isinstance(refine, int):
            refine = id_map.get(refine)
            if refine is None:
                counts["unresolved_references"] += 1
        else:
            refine = None
        # A mask only beside the source it covers, as a hash and never
        # anything else; one this gallery cannot find keeps its hash.
        mask = storage._mask(row.get("mask")) if refine is not None else None
        # And an outpaint only beside them both, as its key.
        spec = None
        if mask is not None and row.get("outpaint") is not None:
            try:
                spec = storage.OutpaintSpec.parse_key(row["outpaint"])
            except ValueError:
                spec = None
            # One rule with the chunk's and the bundle's: a spec no request
            # could make (no side extended, a band past the bound) is left off.
            if spec is None or not spec.wellformed():
                spec = None
                counts["outpaint_malformed"] += 1
        spec = _outpaint(mask, spec, image, width, height, sizes.get(row.get("refine")), counts)
        masks.resolve(mask, source / storage.MASKS_DIR, image, spec, (width, height))
        digest = storage.content_hash(image)
        counts["indexed"] += 1
        if dry_run:
            print(f"  would import  {filename}", file=out)
            # Source references always point backwards in a gallery. Keep a
            # placeholder mapping so the dry run reports only genuinely
            # missing/skipped references, not every row it merely did not add.
            id_map[source_id] = source_id
            continue

        meta = _database_meta(row, width, height, digest, references, refine, mask, spec)
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
            "refine": refine,
            "strength": row.get("strength") if refine is not None else None,
            "crop_anchor": meta.get("crop_anchor"),
            "upscaler": meta.get("upscaler"),
            "execution": execution_key(execution),
            "edit": edit_key(remapped_edit),
            "batch": storage._batch(row.get("batch")),
            # The recycle bin is SQLite's alone, as the star is: carried with
            # the database, and None (live) from one that predates it.
            "binned_at": row.get("binned_at"),
            "mask": mask,
            "outpaint": spec.key() if spec is not None else None,
            "vram_peak_allocated": storage._peak(row.get("vram_peak_allocated")),
            "vram_peak_reserved": storage._peak(row.get("vram_peak_reserved")),
        }
        copied = source != outputs
        try:
            png_execution = canonical_execution(png_meta.get("execution") if png_meta is not None else None)
            if (copied or row.get("refine") != refine or row.get("crop_anchor") != meta.get("crop_anchor")
                    or execution != png_execution or remapped_edit != stored_edit):
                # A database rebuilt beside its original PNGs can renumber or
                # lose a source too. Keep --from-image's crop and source in
                # agreement with the new row, without rewriting an unchanged
                # legacy centered PNG in place.
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


@gallery_lock.protected(lambda: settings.db_path)
def reindex(
    source: Path, dry_run: bool = False, out=sys.stdout, database: Path | None = None
) -> dict[str, int]:
    if database is not None:
        return merge_database(source, database, dry_run=dry_run, out=out)
    counts = {
        "indexed": 0, "present": 0, "no_metadata": 0, "conflicts": 0,
        "unresolved_references": 0, "masks": 0, "bad_masks": 0, "unresolved_masks": 0,
        "unresolved_edits": 0,
        "outpaint_malformed": 0, "outpaints_recovered": 0, "outpaint_masks_rebuilt": 0,
        "invalid_upscalers": 0,
        "invalid_crop_anchors": 0,
        "invalid_execution": 0,
    }
    source = source.resolve()
    outputs = settings.outputs_dir.resolve()
    in_place = source == outputs
    existing = storage.existing_file_ids() if settings.db_path.exists() else {}
    paths = {path.name: path for path in sorted(source.glob("*.png"))}
    _preflight_edits(source)
    _preflight_retained_edits(paths, existing, outputs)
    visited: set[str] = set()
    visiting: set[str] = set()
    identities: dict[str, dict] = {}
    edit_cycles: set[str] = set()
    edit_identities: dict[str, object] = {}
    if not dry_run:
        storage.init_db()
    # The `masks` rows a lost database took with it, from the files first;
    # each result's own embedded copy then covers a file that went too.
    masks = _Masks(counts, dry_run, out)
    masks.from_directory(source / storage.MASKS_DIR)

    def resolve_edit(stored: StoredInstructionEdit) -> int | None:
        # Already unresolved metadata stays unresolved; explicit repair is a
        # separate operation, never a side effect of repeating recovery.
        if stored.source is None:
            return None
        filename = stored.edit_source.filename
        if filename in visiting:
            edit_cycles.update(visiting)
            return None
        if filename in paths:
            index(paths[filename])
        if filename not in existing or filename in edit_cycles:
            return None
        actual = edit_identities.get(filename)
        if actual is None:
            path = outputs / filename
            if not _proves_edit(path, stored):
                return None
        elif actual != stored.edit_source:
            return None
        return existing[filename]

    def resolve_source(provenance) -> int | None:
        if not isinstance(provenance, dict):
            return None
        filename = provenance.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            return None
        # Visit dependencies first: names within one timestamp can sort in
        # either order, and an in-place rebuild assigns entirely new ids.
        if filename in paths and filename not in visiting:
            index(paths[filename])
        if filename not in existing or filename in visiting:
            return None
        actual = identities.get(filename)
        if actual is None and (outputs / filename).is_file():
            # An existing row's hash can be stale after a file was replaced
            # outside the app. Verify the pixels the destination id loads.
            try:
                with Image.open(outputs / filename) as image:
                    actual = {
                        "filename": filename, "hash": storage.content_hash(image),
                        "width": image.width, "height": image.height,
                    }
            except OSError:
                return None
            identities[filename] = actual
        if actual is None or any(
            actual.get(key) != provenance.get(key) for key in ("filename", "hash", "width", "height")
        ):
            return None
        return existing[filename]

    def index(path: Path) -> None:
        if path.name in visited:
            return
        visited.add(path.name)
        if path.name in existing:
            try:
                incoming = canonical_edit((_edit_metadata(path) or {}).get("edit"))
                known = storage.get_image(existing[path.name], include_binned=True)
                if known is None:
                    raise ValueError("the retained row disappeared")
                target_meta = _edit_metadata(outputs / path.name)
                retained = edit_from_sources({
                    "edit": edit_key(known.edit), "model_id": known.model_id,
                    "width": known.width, "height": known.height, "prompt": known.prompt,
                }, target_meta or {})
                if not _same_edit(incoming, retained) or (
                    incoming is not None and known.prompt != (_edit_metadata(path) or {}).get("prompt")
                ):
                    raise ValueError("different instruction edit already uses that filename")
            except (OSError, ValueError) as exc:
                counts["conflicts"] += 1
                print(f"  skip  {path.name}: cannot keep existing edit metadata: {exc}", file=out)
                return
            counts["present"] += 1
            return
        try:
            meta, (width, height), image = _read(path)
        except ValueError as exc:
            counts["invalid_execution"] += 1
            print(f"  skip  {path.name}: invalid PNG metadata: {exc}", file=out)
            return
        except OSError:  # not a PNG at all, or truncated - UnidentifiedImageError is one
            meta = None
        if meta is None:
            counts["no_metadata"] += 1
            print(f"  skip  {path.name}: no Auradream metadata", file=out)
            return
        try:
            execution = canonical_execution(meta.get("execution"))
            stored_edit = canonical_edit(meta.get("edit"))
        except ValueError as exc:
            counts["invalid_execution"] += 1
            print(f"  skip  {path.name}: invalid execution metadata: {exc}", file=out)
            return
        try:
            original_crop_anchor = storage._crop_anchor(
                meta.get("crop_anchor"), source=meta.get("refine"),
                mask=meta.get("mask"), outpaint=meta.get("outpaint"),
            )
        except ValueError as exc:
            counts["invalid_crop_anchors"] += 1
            print(f"  skip  {path.name}: {exc}", file=out)
            return
        if meta.get("upscaler") is not None and storage._upscaler(
            meta["upscaler"], source=meta.get("refine"), model_id=meta.get("model"),
            mask=meta.get("mask"), outpaint=meta.get("outpaint"),
        ) is None:
            counts["invalid_upscalers"] += 1
            print(f"  skip  {path.name}: invalid upscaler for its Refine pass", file=out)
            return
        target = outputs / path.name
        if not in_place and target.exists():
            # A different file under a name the gallery already uses on disk
            # but not in the index: not ours to overwrite.
            counts["conflicts"] += 1
            print(f"  skip  {path.name}: a different file of that name is already in outputs", file=out)
            return
        visiting.add(path.name)
        edit_source = resolve_edit(stored_edit) if stored_edit is not None else None
        if path.name in edit_cycles:
            edit_source = None
        # Dry-run ids are negative stand-ins for legacy provenance. Stored edit
        # contracts require positive ids: retain a positive planning placeholder
        # only in memory; no dry-run metadata or file is ever written.
        planned_source = abs(edit_source) if dry_run and edit_source is not None else edit_source
        remapped_edit = _remapped_edit(stored_edit, planned_source, counts)
        references = []
        reference_sources = []
        sources = _json_list(meta.get("reference_sources"))
        for i, _reference in enumerate(_json_list(meta.get("reference"))):
            provenance = sources[i] if i < len(sources) else None
            mapped = resolve_source(provenance)
            if mapped is None:
                counts["unresolved_references"] += 1
            else:
                references.append(mapped)
                reference_sources.append(provenance)
        refine = None
        if meta.get("refine") is not None:
            refine = resolve_source(meta.get("refine_source"))
            if refine is None:
                counts["unresolved_references"] += 1
        visiting.remove(path.name)
        # A mask goes where its source went: it covers that image's pixels.
        mask = storage._mask(meta.get("mask")) if refine is not None else None
        # An outpaint goes where its mask went; a malformed one is left off.
        spec = storage.chunk_outpaint(meta) if mask is not None else None
        if mask is not None and storage.malformed_outpaint(meta):
            counts["outpaint_malformed"] += 1
        spec = _outpaint(mask, spec, image, width, height, _provenance_size(meta.get("refine_source")), counts)
        masks.resolve(mask, source / storage.MASKS_DIR, image, spec, (width, height))
        outpaint_meta = storage.outpaint_chunk(spec)
        outpaint_fill = spec.fill if spec is not None else None
        upscaler = storage._upscaler(
            meta.get("upscaler"), source=refine, model_id=meta.get("model"), mask=mask, outpaint=spec,
        )
        # Original metadata was validated before resolving links. Dry-run
        # source ids are negative stand-ins, so do not validate those as real
        # gallery ids; only the loss of a source changes its crop setting.
        crop_anchor = original_crop_anchor if refine is not None else None
        links_changed = (
            meta.get("reference") != (references or None)
            or meta.get("refine") != refine
            or (refine is None and meta.get("strength") is not None)
            or meta.get("mask") != mask
            or meta.get("outpaint") != outpaint_meta
            or meta.get("outpaint_fill") != outpaint_fill
            or meta.get("upscaler") != upscaler
            or meta.get("crop_anchor") != crop_anchor
            or stored_edit != remapped_edit
        )
        meta = {
            **meta,
            "reference": references or None,
            "reference_sources": reference_sources or None,
            "refine": refine,
            "refine_source": meta.get("refine_source") if refine is not None else None,
            "strength": meta.get("strength") if refine is not None else None,
            "mask": mask,
            "mask_source": storage.mask_provenance(mask, width, height),
            "mask_blend": "overlay" if mask is not None else None,
            "outpaint": outpaint_meta,
            "outpaint_fill": outpaint_fill,
        }
        if upscaler is not None:
            meta["upscaler"] = upscaler
        else:
            meta.pop("upscaler", None)
        if crop_anchor is not None:
            meta["crop_anchor"] = crop_anchor
        else:
            meta.pop("crop_anchor", None)
        meta.pop("execution", None)
        meta.update(execution_chunk(execution))
        meta.pop("edit", None)
        meta.update(edit_chunk(remapped_edit))
        digest = storage.content_hash(image) if image is not None else None
        identities[path.name] = {
            "filename": path.name, "hash": digest, "width": width, "height": height,
        }
        if image is not None:
            edit_identities[path.name] = descriptor_from_image(path.name, image)
        counts["indexed"] += 1
        if dry_run:
            # Synthetic ids exist only in this plan, so dependent files can
            # resolve their provenance without creating a database or a copy.
            existing[path.name] = -counts["indexed"]
            print(f"  would index  {path.name}", file=out)
            return
        if not in_place or links_changed:
            # Keep --from-image safe after an in-place id remap as well. Only
            # metadata changes; _copy_in preserves the compressed pixels.
            _copy_in(path, target, meta)
        # The hash of the pixels as decoded here, not the block's copy of it:
        # the file is the truth, and a PNG from before the hash has no copy.
        record = storage.index_file(
            path.name,
            meta,
            width,
            height,
            _created_at(path),
            digest=digest,
        )
        if image is not None:
            storage.thumbnail_for(image, path.name)
        existing[path.name] = record.id

    for path in paths.values():
        index(path)
    return counts


def main(argv: list[str] | None = None) -> int:
    # Redirected or piped, stdout on Windows encodes with the locale's code
    # page, where a CJK file or folder name in --from was a UnicodeEncodeError
    # partway through the report. So does `> NUL`, which answers isatty(); the
    # Windows console is UTF-8 already. Elsewhere a terminal keeps its own
    # encoding. A StringIO has none to change, and under pythonw there is no
    # stdout.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure) and (sys.platform == "win32" or not sys.stdout.isatty()):
        reconfigure(encoding="utf-8", errors="backslashreplace")
    from backend.launch import warn_legacy_env

    warn_legacy_env()  # a pre-rename variable left in a shell: named, not read
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
        help="the Auradream database belonging to --from; preserves favorites, "
        "durations and remaps references",
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
        )
    else:
        summary += f", {counts['no_metadata']} without metadata" if counts["no_metadata"] else ""
        if counts["invalid_upscalers"]:
            summary += f", {counts['invalid_upscalers']} with an invalid upscaler"
        if counts["invalid_crop_anchors"]:
            summary += f", {counts['invalid_crop_anchors']} with an invalid crop anchor"
        if counts["invalid_execution"]:
            summary += f", {counts['invalid_execution']} with invalid execution metadata"
    if counts["unresolved_references"]:
        summary += f", {counts['unresolved_references']} unresolved reference/refine source(s)"
    if counts["unresolved_edits"]:
        summary += f", {counts['unresolved_edits']} unresolved instruction-edit source(s)"
    if counts["masks"]:
        summary += f", {counts['masks']} inpaint mask(s) {'to restore' if args.dry_run else 'restored'}"
    if counts["bad_masks"]:
        summary += f", {counts['bad_masks']} mask file(s) not named by their pixels"
    if counts["unresolved_masks"]:
        summary += f", {counts['unresolved_masks']} inpaint mask(s) found nowhere"
    if counts["outpaint_masks_rebuilt"]:
        summary += f", {counts['outpaint_masks_rebuilt']} outpaint mask(s) built again from their spec"
    if counts["outpaints_recovered"]:
        summary += f", {counts['outpaints_recovered']} outpaint(s) recovered from the A1111 line"
    if counts["outpaint_malformed"]:
        summary += f", {counts['outpaint_malformed']} malformed outpaint(s) left off"
    summary += f", {counts['conflicts']} name conflict(s)" if counts["conflicts"] else ""
    print(summary + f". {time.perf_counter() - started:.1f}s.")
    if args.dry_run:
        print("Dry run - nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
