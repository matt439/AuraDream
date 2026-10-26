"""Names and compatibility reads for metadata embedded in gallery PNGs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, IO

PNG_METADATA_KEY = "local-diffusion"

# Images written before the project rename used the original app name. Keep
# reading that key so old galleries and backups remain reproducible; every new
# write, copy and database-backed import uses ``PNG_METADATA_KEY``.
LEGACY_PNG_METADATA_KEYS = ("krea",)
PNG_METADATA_KEYS = (PNG_METADATA_KEY, *LEGACY_PNG_METADATA_KEYS)

# An inpaint's mask, its own PNG's bytes in base64, in a text chunk beside the
# record: the result's PNG alone then reproduces it, and a PNG-only reindex
# can put back a mask file that was lost. Text, so no pixel-deciding chunk and
# no decoded identity changes with it.
PNG_MASK_KEY = "local-diffusion-mask"


def require_header_metadata(stream: IO[bytes]) -> None:
    """Refuse app text after IDAT, which header-only readers cannot see.

    Inspect chunk headers and at most 80 bytes of each text keyword; skip
    image data and text values without reading or decompressing them. The
    caller's position is restored on both success and refusal. This checks
    metadata placement, not PNG CRCs or decoded image validity.
    """
    position = stream.tell()
    owned = {key.encode("ascii") for key in (*PNG_METADATA_KEYS, PNG_MASK_KEY)}
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
                raise ValueError("Local Diffusion metadata must precede PNG image data; app text after IDAT is unsupported")
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
