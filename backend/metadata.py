"""Names and compatibility reads for metadata embedded in gallery PNGs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def embedded_metadata(info: Mapping[str, Any]) -> str | None:
    """Return this app's JSON text from either the current or a legacy key."""
    for key in PNG_METADATA_KEYS:
        raw = info.get(key)
        if raw:
            return raw
    return None
