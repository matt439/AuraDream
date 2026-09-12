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


def embedded_metadata(info: Mapping[str, Any]) -> str | None:
    """Return this app's JSON text from either the current or a legacy key."""
    for key in PNG_METADATA_KEYS:
        raw = info.get(key)
        if raw:
            return raw
    return None
