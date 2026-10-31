"""One tracked PNG and its row in a gallery bundle.

Pixel identity, the pixel-deciding chunk digest, checking a row against its
image, the embedded reproduction metadata, and streaming rewrites of the
PNG's auradream tEXt chunk.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import IO, NamedTuple

from PIL import Image, ImageFile

from . import storage
from .bundle_common import _CHUNK, _WORKERS, MASK_HASH, BundleError
from .execution import execution_chunk, execution_from_json, execution_key
from .instruction_edit import edit_key
from .metadata import (
    PNG_METADATA_KEYS, PNG_SIGNATURE, edit_chunk, edit_from_info, edit_from_json, execution_from_info,
    metadata_chunk, require_header_metadata, text_keyword, validate_edit_record,
)


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
    # The strict SQL/PNG recipe carried by a tracked image, not its pixels.
    execution: str | None = None
    edit: str | None = None
    # Full decoded RGB digest for portable instruction-source proof. Ordinary
    # RGB images reuse the raw digest; other modes are converted after decoding.
    rgb_hash: str | None = None


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


def _pixels(path: Path | str | IO[bytes], *, name: str | None = None, row: dict | None = None) -> _Pixels:
    name = name or (Path(path).name if isinstance(path, (str, Path)) else str(getattr(path, "name", "PNG")))
    try:
        with (open(path, "rb") if isinstance(path, (str, Path)) else nullcontext(path)) as stream:
            require_header_metadata(stream)
            with Image.open(stream) as image:
                if image.format != "PNG":
                    raise BundleError(f"not a PNG image: {name}")
                execution = _execution(row, image.info) if row is not None else None
                edit = _edit(row, image.info) if row is not None else None
                image.load()
                raw = _raw_digest(image)
                identity = hashlib.sha256(repr((
                    image.mode, image.size, (image.palette.mode, image.palette.tobytes()) if image.palette else None,
                    image.info.get("transparency"), raw,
                )).encode()).hexdigest()
                # The same digest storage.content_hash takes, without a second pass.
                rgb = image.mode == "RGB" and "transparency" not in image.info
                if image.mode == "RGB":
                    rgb_hash = raw
                else:
                    with image.convert("RGB") as converted:
                        rgb_hash = _raw_digest(converted)
                return _Pixels(raw[:16], image.width, image.height, identity, rgb, execution, edit, rgb_hash)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BundleError(f"cannot read image {name}: {exc}") from exc


def _critical_digest(source) -> str | None:
    """SHA-256 of the chunks that decide a PNG's pixels (None if malformed).

    Equal digests mean equal decoded pixels; metadata chunks are ignored.
    """
    try:
        if source.read(8) != PNG_SIGNATURE:
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
    pixels = _pixels(source, name=name, row=row)
    _validate_row(row, pixels.width, pixels.height, pixels.hash)
    return pixels


def _validate_row(row: dict, width: int, height: int, content_hash: str) -> None:
    if (width, height) != (row["width"], row["height"]):
        raise BundleError(f"image dimensions disagree with the database: {row['filename']}")
    _references(row)
    _execution(row)
    _edit(row)
    refine = row.get("refine")
    if refine is not None and (type(refine) is not int or refine <= 0):
        raise BundleError(f"invalid refine source on image {row['id']}")
    # A mask hash, never a path, and only beside the source it covers - the
    # shape `applied()` and `storage.index_file` leave on every row.
    mask = row.get("mask")
    if mask is not None and (not isinstance(mask, str) or not MASK_HASH.fullmatch(mask) or refine is None):
        raise BundleError(f"invalid inpaint mask on image {row['id']}")
    # An outpaint's spec, only beside its source and its mask, well formed
    # (`OutpaintSpec.wellformed`: a side extended, the band within bound).
    outpaint = row.get("outpaint")
    spec = None
    if outpaint is not None:
        try:
            spec = storage.OutpaintSpec.parse_key(outpaint)
        except (TypeError, ValueError):
            spec = None
        if spec is None or refine is None or mask is None or not spec.wellformed():
            raise BundleError(f"invalid outpaint on image {row['id']}")
    upscaler = row.get("upscaler")
    if upscaler is not None and storage._upscaler(
        upscaler, source=refine, model_id=row.get("model_id"), mask=mask, outpaint=outpaint,
    ) is None:
        raise BundleError(f"invalid upscaler on image {row['id']}")
    try:
        storage._crop_anchor(row.get("crop_anchor"), source=refine, mask=mask, outpaint=outpaint)
    except ValueError as exc:
        raise BundleError(f"invalid crop anchor on image {row['id']}: {exc}") from exc
    values = {key: row.get(key) for key in storage._COLUMNS}
    values.update(id=row["id"], content_hash=content_hash)
    # A bad peak is dropped by the merge, as a result; it refuses nothing.
    for key in ("vram_peak_allocated", "vram_peak_reserved"):
        values[key] = storage._peak(values[key])
    try:
        storage._row_to_record(values)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"invalid metadata for {row['filename']}: {exc}") from exc


def _execution(row: dict, info: dict | None = None) -> str | None:
    """Validate both stored sources, retaining a recipe an older schema lacks.

    Called during the existing in-memory image inspection, including the
    repeat-import fast path; it neither grants runtime availability nor
    changes the export's source snapshot before its cleanup comparison.
    """
    try:
        return execution_key(execution_from_info(row, info))
    except (TypeError, ValueError) as exc:
        raise BundleError(f"invalid execution metadata for {row['filename']}: {exc}") from exc


def _edit(row: dict, info: dict | None = None) -> str | None:
    """Require matching durable SQL/PNG intent, even in pixel-equality paths."""
    try:
        edit = edit_from_json(row.get("edit")) if info is None else edit_from_info(row, info)
        validate_edit_record(edit, row)
        if row.get("_bundle_version") == 1 and edit is not None:
            raise ValueError("version 1 bundles cannot carry instruction edits")
        return edit_key(edit)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"invalid instruction-edit metadata for {row['filename']}: {exc}") from exc


def _portable_edit(value: str | None) -> tuple | None:
    edit = edit_from_json(value)
    return None if edit is None else (edit.recipe, edit.edit_source)


def _inspect_source(row: dict, data: bytes):
    return _validate_source_row(row, io.BytesIO(data), name=row["filename"])


def _inspect_mask(member: str, data: bytes, known: dict | None) -> tuple[int, int]:
    """Verify a bundle's `masks/<hash>.png` as the mask its name says: a PNG
    of one `L` channel whose `storage.mask_hash` - its size and decoded
    levels - is the name, at the size its `masks` row records (`known`, None
    where the bundle's table has no row for it). Answers the size.

    The pixels decide, never the file's bytes, as an image's identity does:
    a mask re-encoded by another program is the same mask; one painted
    differently, or renamed, is refused before anything is merged.
    """
    digest = PurePosixPath(member).stem
    try:
        with Image.open(io.BytesIO(data)) as image:
            fmt, mode = image.format, image.mode
            if fmt == "PNG" and mode == "L":
                image.load()
                actual, size = storage.mask_hash(image), image.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BundleError(f"cannot read inpaint mask {member}: {exc}") from exc
    if fmt != "PNG":
        raise BundleError(f"not a PNG image: {member}")
    if mode != "L":
        raise BundleError(f"inpaint mask {digest[:16]} is {mode}, not one luminance channel")
    if actual != digest:
        raise BundleError(f"inpaint mask {digest[:16]} holds other pixels than its name says")
    if known is not None and (known["width"], known["height"]) != size:
        raise BundleError(
            f"inpaint mask {digest[:16]} is {size[0]}x{size[1]}, but the bundle's record of it says "
            f"{known['width']}x{known['height']}"
        )
    return size


def _metadata(row: dict, by_id: dict[int, dict]) -> dict:
    def provenance(image_id):
        source = by_id.get(image_id)
        if not source:
            return None
        return {"filename": source["filename"], "hash": source["content_hash"],
                "width": source["width"], "height": source["height"]}
    refs = _references(row)
    outpaint = (
        storage.OutpaintSpec.parse_key(row["outpaint"]) if row.get("outpaint") and row.get("mask") else None
    )
    crop_anchor = storage._crop_anchor(
        row.get("crop_anchor"), source=row.get("refine"), mask=row.get("mask"), outpaint=row.get("outpaint"),
    )
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
        # A mask is content-addressed, so a merge never renumbers it; a row
        # from a build before masks has none.
        "mask": row.get("mask"),
        "mask_source": storage.mask_provenance(row.get("mask"), row["width"], row["height"]),
        "mask_blend": "overlay" if row.get("mask") else None,
        # And an outpaint's spec, null together on every other row: an
        # imported outpaint's PNG keeps what it is, for a later reindex.
        "outpaint": storage.outpaint_chunk(outpaint),
        "outpaint_fill": outpaint.fill if outpaint is not None else None,
        "batch": row.get("batch"),
        **({"crop_anchor": crop_anchor} if crop_anchor is not None else {}),
        **({"upscaler": row["upscaler"]} if row.get("upscaler") is not None else {}),
        **execution_chunk(execution_from_json(row.get("execution"))),
        **edit_chunk(edit_from_json(row.get("edit"))),
        **_peak_chunk(row),
    }


def _peak_chunk(row: dict) -> dict:
    """The peak keys as `save_image` writes them: absent where unmeasured."""
    return {
        key: storage._peak(row.get(key))
        for key in ("vram_peak_allocated", "vram_peak_reserved")
        if storage._peak(row.get(key)) is not None
    }


def _rewrite_metadata(path: Path, metadata: dict) -> None:
    """Replace one PNG text chunk while streaming the untouched image data."""
    chunk = metadata_chunk(metadata)
    temp = path.with_suffix(".metadata.part")
    inserted = False
    with path.open("rb") as source, temp.open("xb") as target:
        signature = source.read(8)
        if signature != PNG_SIGNATURE:
            raise BundleError(f"not a PNG image: {path.name}")
        target.write(signature)
        while header := source.read(8):
            if len(header) != 8:
                raise BundleError(f"truncated PNG: {path.name}")
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            start = source.tell()
            keyword = text_keyword(kind, source.read(min(length, 80)))
            source.seek(start)
            if keyword in {key.encode("ascii") for key in PNG_METADATA_KEYS}:
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
