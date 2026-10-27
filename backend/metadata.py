"""Names and compatibility reads for metadata embedded in gallery PNGs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, IO

from .instruction_edit import StoredInstructionEdit, geometry, recipe_for, stored_from_json

PNG_METADATA_KEY = "auradream"

# Images written before each project rename carry an earlier app name: Local
# Diffusion's "local-diffusion" and, before it, "krea". Keep reading those keys
# so old galleries and backups remain reproducible; every new write, copy and
# database-backed import uses ``PNG_METADATA_KEY``, replacing a legacy one.
LEGACY_PNG_METADATA_KEYS = ("local-diffusion", "krea")
PNG_METADATA_KEYS = (PNG_METADATA_KEY, *LEGACY_PNG_METADATA_KEYS)

# An inpaint's mask, its own PNG's bytes in base64, in a text chunk beside the
# record: the result's PNG alone then reproduces it, and a PNG-only reindex
# can put back a mask file that was lost. Text, so no pixel-deciding chunk and
# no decoded identity changes with it.
PNG_MASK_KEY = "auradream-mask"
# The mask chunk as Local Diffusion wrote it. Metadata rewrites leave a mask
# chunk as it is, so an old result keeps this key and is read by it.
LEGACY_PNG_MASK_KEYS = ("local-diffusion-mask",)
PNG_MASK_KEYS = (PNG_MASK_KEY, *LEGACY_PNG_MASK_KEYS)


def require_header_metadata(stream: IO[bytes]) -> None:
    """Refuse app text after IDAT, which header-only readers cannot see.

    Inspect chunk headers and at most 80 bytes of each text keyword; skip
    image data and text values without reading or decompressing them. The
    caller's position is restored on both success and refusal. This checks
    metadata placement, not PNG CRCs or decoded image validity.
    """
    position = stream.tell()
    owned = {key.encode("ascii") for key in (*PNG_METADATA_KEYS, *PNG_MASK_KEYS)}
    image_data = False
    try:
        stream.seek(0)
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG image")
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError("truncated PNG chunk header")
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            size = min(length, 80) if kind in {b"tEXt", b"zTXt", b"iTXt"} else 0
            keyword = stream.read(size).split(b"\0", 1)[0] if size else None
            if image_data and keyword in owned:
                raise ValueError("Auradream metadata must precede PNG image data; app text after IDAT is unsupported")
            stream.seek(length - size + 4, 1)  # Unread payload and CRC.
            if kind == b"IDAT":
                image_data = True
            elif kind == b"IEND":
                return
        raise ValueError("PNG has no IEND chunk")
    finally:
        stream.seek(position)


def embedded_metadata(info: Mapping[str, Any]) -> str | None:
    """Return this app's JSON text from either the current or a legacy key."""
    for key in PNG_METADATA_KEYS:
        raw = info.get(key)
        if raw:
            return raw
    return None


def embedded_mask(info: Mapping[Any, Any]) -> str | None:
    """Return the base64 mask text from either the current or a legacy key."""
    for key in PNG_MASK_KEYS:
        raw = info.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return None


def canonical_edit(value: object) -> StoredInstructionEdit | None:
    """Strict stored intent; absence alone means an ordinary image."""
    return None if value is None else StoredInstructionEdit.model_validate(value)


def edit_from_json(value: str | None) -> StoredInstructionEdit | None:
    return None if value is None else stored_from_json(value)


def edit_chunk(value: object) -> dict[str, Any]:
    edit = canonical_edit(value)
    return {} if edit is None else {"edit": edit.model_dump(mode="json")}


def edit_from_sources(row: Mapping[str, Any], meta: Mapping[str, Any] | None) -> StoredInstructionEdit | None:
    """SQL and PNG must both retain the same edit, including unresolved proof.

    Unlike older optional reproduction data, an absent representation cannot
    recover an edit from the other: accepting that would permit an older
    reader to erase the pass kind. SQL-only validation passes ``meta=None``.
    """
    database = edit_from_json(row.get("edit"))
    validate_edit_record(database, row)
    if meta is None:
        return database
    embedded = canonical_edit(meta.get("edit"))
    validate_edit_record(embedded, meta)
    if database != embedded or (database is not None and row.get("prompt") != meta.get("prompt")):
        raise ValueError("instruction edits disagree between the database and PNG")
    return database


def validate_edit_record(edit: StoredInstructionEdit | None, record: Mapping[str, Any]) -> None:
    """A stored edit is the frozen pass, never a disguised direct/refine result."""
    if edit is None:
        return
    recipe = recipe_for(edit.recipe)
    expected = geometry(edit.edit_source.width, edit.edit_source.height, recipe=edit.recipe).output_size
    size = (record.get("width"), record.get("height"))
    if "size" in record and "width" not in record and "height" not in record:
        size = tuple(str(record["size"]).split("x"))
        if size == tuple(str(value) for value in expected):
            size = expected
    if record.get("model_id", record.get("model")) != recipe.model_id or size != expected:
        raise ValueError("instruction-edit result must use its frozen editor checkpoint and output geometry")
    if record.get("negative_prompt") not in (None, "") or record.get("reference") not in (None, [], "[]"):
        raise ValueError("instruction edits cannot carry negative prompts or references")
    if any(record.get(key) is not None for key in (
        "refine", "strength", "mask", "outpaint", "upscaler", "crop_anchor", "lora", "lora_scale", "true_cfg", "mu",
    )) or record.get("execution") not in (None, {}, "{}", "null"):
        raise ValueError("an instruction edit cannot also carry another generation workflow")
    batch = record.get("batch")
    if batch is not None and (type(batch) is not int or batch != 1):
        raise ValueError("instruction edits require one image per pipeline call")


def edit_from_info(row: Mapping[str, Any], info: Mapping[str, Any] | None) -> StoredInstructionEdit | None:
    import json

    raw = embedded_metadata(info) if info is not None else None
    meta = json.loads(raw) if raw is not None else {}
    if not isinstance(meta, dict):
        raise ValueError("PNG metadata must be a JSON object")
    return edit_from_sources(row, meta)
