"""The PNG chunk keys: the current names written, every earlier one read."""
from __future__ import annotations

import io
import zlib

import pytest
from PIL import Image, PngImagePlugin

from backend import metadata


def test_the_current_keys_come_first_and_every_earlier_name_is_read():
    assert metadata.PNG_METADATA_KEYS == ("auradream", "local-diffusion", "krea")
    assert metadata.PNG_MASK_KEYS == ("auradream-mask", "local-diffusion-mask")
    assert metadata.embedded_metadata({"local-diffusion": "{}"}) == "{}"
    assert metadata.embedded_metadata({"krea": "old", "auradream": "new"}) == "new"
    assert metadata.embedded_metadata({"parameters": "a fox"}) is None
    assert metadata.embedded_mask({"local-diffusion-mask": "b64"}) == "b64"
    assert metadata.embedded_mask({"local-diffusion-mask": "old", "auradream-mask": "new"}) == "new"
    assert metadata.embedded_mask({"auradream-mask": ""}) is None


def _late(key: str) -> io.BytesIO:
    """A PNG whose `key` text chunk follows its image data."""
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, format="PNG")
    raw = stream.getvalue()
    end = raw.rindex(b"IEND") - 4
    payload = key.encode("ascii") + b"\0value"

    chunk = len(payload).to_bytes(4, "big") + b"tEXt" + payload
    chunk += zlib.crc32(b"tEXt" + payload).to_bytes(4, "big")
    return io.BytesIO(raw[:end] + chunk + raw[end:])


@pytest.mark.parametrize("key", [*metadata.PNG_METADATA_KEYS, *metadata.PNG_MASK_KEYS])
def test_an_owned_key_after_the_image_data_is_refused_under_any_name(key):
    with pytest.raises(ValueError, match="metadata must precede PNG image data"):
        metadata.require_header_metadata(_late(key))


def test_another_tool_s_late_text_is_left_alone():
    metadata.require_header_metadata(_late("Comment"))


def test_a_header_record_under_an_old_name_passes():
    info = PngImagePlugin.PngInfo()
    info.add_text("local-diffusion", "{}")
    info.add_text("local-diffusion-mask", "b64")
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, format="PNG", pnginfo=info)
    metadata.require_header_metadata(stream)
