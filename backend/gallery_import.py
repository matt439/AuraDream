"""Decode a user-selected picture for the gallery without importing its metadata."""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from . import storage
from .schemas_gallery import ImageRecord

MAX_UPLOAD_BYTES = 32 << 20
MAX_PIXELS = 32_000_000
FORMATS = ("PNG", "JPEG", "WEBP")


def _pixels(data: bytes | bytearray) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data), formats=list(FORMATS)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(f"The image exceeds {MAX_PIXELS // 1_000_000} megapixels.")
            if getattr(image, "is_animated", False):
                raise ValueError("Choose a still image. Animated images cannot be imported.")
            oriented = ImageOps.exif_transpose(image)
            # A sketch's transparent background is white in the stored RGB source.
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                pixels = Image.new("RGB", rgba.size, "white")
                pixels.paste(rgba, mask=rgba.getchannel("A"))
            else:
                pixels = oriented.convert("RGB")
            pixels.info.clear()
            return pixels
    except Image.DecompressionBombError:
        raise ValueError(f"The image exceeds {MAX_PIXELS // 1_000_000} megapixels.") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("Choose a readable PNG, JPEG or WebP image.") from None


def store(data: bytes | bytearray, name: str) -> ImageRecord:
    with _pixels(data) as pixels:
        return storage.save_imported_source(pixels, name)
