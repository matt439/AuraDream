"""Settings read back from a PNG's text chunks, shared by `--from-image` and
the page's Load settings from PNG (`api_png.py`).

Two blocks are read: this app's own JSON record (`metadata.embedded_metadata`,
the whole reproduction set) and the A1111 'parameters' text that A1111,
Forge, Civitai and this app's own PNGs carry. Pure, like `metadata`: nothing
here imports config, so the CLI can call it before `apply_env`, and Pillow
only where a file is opened.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from backend.metadata import embedded_metadata, require_header_metadata
from backend.execution import canonical_execution
from backend.metadata import canonical_edit
from backend.refine import CROP_ANCHORS

if TYPE_CHECKING:
    from .schemas_generate import CropAnchor

# A1111's own keyword for the block (`modules/images.py`), which Forge and
# Civitai keep and `storage.save_image` writes beside ours.
PARAMETERS_KEY = "parameters"

# The A1111 infotext format: one `Key: value` per comma, a value that holds a
# comma quoted as JSON. The settings line is the last line, and only when it
# holds three or more.
_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
_NEGATIVE = "Negative prompt:"


class PngSettingsError(ValueError):
    """A PNG whose settings cannot be read, in words for the CLI or the page."""


def crop_anchor(value: Any, *, source: Any, mask: Any = None, outpaint: Any = None) -> CropAnchor | None:
    """A stored crop anchor, with the legacy centered choice represented by NULL.

    Noncenter crops describe an ordinary Refine source. Validate before
    provenance remapping; a caller dropping that source must drop its anchor
    too rather than turn a source-dependent setting into a direct render.
    Pure so the CLI can read a PNG before selecting its configuration.
    """
    if value is None or value == "center":
        return None
    if not isinstance(value, str) or value not in CROP_ANCHORS:
        raise ValueError("invalid crop anchor")
    if type(source) is not int or source <= 0 or mask is not None or outpaint is not None:
        raise ValueError("a noncenter crop anchor needs an ordinary Refine source")
    return value


def record(chunks: Mapping[str, Any], name: str) -> dict | None:
    """This app's record in `chunks`, or None when it carries none.

    A record that is there but is not a JSON object is refused rather than
    passed over: another reader of the same file would take the 'parameters'
    line and fill a panel that is not what made the image.
    """
    raw = embedded_metadata(chunks)
    if not raw:
        return None
    try:
        meta = json.loads(raw)
    except (ValueError, RecursionError):
        raise PngSettingsError(f"the AuraDream metadata in {name} is not JSON") from None
    if not isinstance(meta, dict):
        raise PngSettingsError(f"the AuraDream metadata in {name} is not a settings record")
    try:
        if canonical_edit(meta.get("edit")) is not None:
            raise ValueError("instruction-edit settings must be reopened from the gallery image")
        canonical_execution(meta.get("execution"))
        crop_anchor(
            meta.get("crop_anchor"), source=meta.get("refine"),
            mask=meta.get("mask"), outpaint=meta.get("outpaint"),
        )
    except ValueError as exc:
        raise PngSettingsError(f"the AuraDream metadata in {name} has {exc}") from None
    return meta


def image_settings(path: str) -> dict:
    """The AuraDream record a gallery PNG on disk carries, as a dict.

    `Image.open(p).info` - not `.text`, which is a property that calls `load()`
    and decodes every pixel to answer a question the header settles. A JPEG
    has no such block and neither has a PNG from another tool; both are
    refusals in words, not tracebacks.
    """
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.format == "PNG":
                if image.fp is None:
                    raise ValueError("the PNG metadata stream is unavailable")
                require_header_metadata(image.fp)
            info = {key: value for key, value in image.info.items() if isinstance(key, str)}
    except (OSError, ValueError) as exc:
        raise PngSettingsError(f"cannot open {path}: {exc}") from None
    meta = record(info, path)
    if meta is None:
        raise PngSettingsError(f"{path} carries no AuraDream metadata; only this app's PNGs do")
    return meta


def _unquote(value: str) -> str:
    if len(value) > 1 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def parse_parameters(text: str) -> tuple[str, str, dict[str, str]]:
    """An A1111 'parameters' block as (prompt, negative prompt, settings).

    A1111's own reading of it: the prompt is every line before the one that
    opens with "Negative prompt:", the negative that line and the rest, and
    the last line is the settings - `Steps: 20, Sampler: Euler a, ...` - when
    it holds at least three `Key: value` pairs; otherwise it is prompt text
    too. Each line is stripped, as A1111 strips it. The settings keep A1111's
    keys and their order, unread here: what they mean is the caller's to say.
    """
    lines = text.strip().split("\n")
    last = lines.pop() if lines else ""
    if len(_PARAM.findall(last)) < 3:
        lines.append(last)
        last = ""
    prompt: list[str] = []
    negative: list[str] = []
    in_negative = False
    for line in lines:
        line = line.strip()
        if line.startswith(_NEGATIVE):
            in_negative = True
            line = line[len(_NEGATIVE):].strip()
        (negative if in_negative else prompt).append(line)
    params: dict[str, str] = {}
    for key, value in _PARAM.findall(last):
        params.setdefault(key.strip(), _unquote(value.strip()))
    if any(key.casefold() == "instruction edit" for key in params):
        raise PngSettingsError("instruction-edit settings must be reopened from the gallery image")
    return "\n".join(prompt).strip(), "\n".join(negative).strip(), params
