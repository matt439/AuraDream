"""Bounded gallery-only RGB source proof for detailer previews and admission."""
from __future__ import annotations

from PIL import Image

from . import storage
from .instruction_edit import InstructionEditSource, descriptor_from_image
from .refusal_codes import DETAILER_STALE, IMAGE_BINNED, IMAGE_UNKNOWN, CodedError
from .schemas_gallery import ImageRecord

# CPU detection is bounded separately from the diffusion model's size guards.
MAX_SOURCE_PIXELS = 4_194_304


def load_source(
    source_id: int, recorded: InstructionEditSource | None = None,
) -> tuple[ImageRecord, InstructionEditSource, Image.Image]:
    """Return an owned RGB copy and proof from the same open file handle.

    Never interpret a descriptor's filename as an input path. A binned source
    is refused even when its PNG remains on disk.
    """
    record = storage.get_image(source_id)
    if record is None:
        binned = storage.get_image(source_id, True) is not None
        raise CodedError(
            f"Detailer source #{source_id} is missing or in the recycle bin.",
            IMAGE_BINNED if binned else IMAGE_UNKNOWN, [source_id],
        )
    InstructionEditSource._bare_png(record.filename)
    path = storage.reference_paths([source_id], "detailer source")[0]
    if path.is_symlink() or path.is_junction():
        raise ValueError("A detailer source cannot be a linked file.")
    rgb: Image.Image | None = None
    try:
        with Image.open(path) as handle:
            if handle.width * handle.height > min(MAX_SOURCE_PIXELS, storage.settings.refine_max_pixels):
                raise ValueError(f"The detailer accepts sources up to {MAX_SOURCE_PIXELS:,} pixels.")
            if handle.size != (record.width, record.height):
                raise CodedError("The detailer source no longer matches its gallery dimensions; detect regions again.", DETAILER_STALE)
            rgb = handle.convert("RGB")
            proof = descriptor_from_image(record.filename, rgb)
        if recorded is not None and proof != recorded:
            raise CodedError("The detailer source changed; detect its regions again.", DETAILER_STALE)
        return record, proof, rgb
    except BaseException:
        if rgb is not None:
            rgb.close()
        raise
