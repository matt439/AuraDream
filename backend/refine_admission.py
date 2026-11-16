"""Gallery source checks for the deliberately bounded Anima Refine release."""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import refine_qualification

if TYPE_CHECKING:
    from PIL.Image import Image
    from .schemas_generate import GenerateRequest


def _source(request: GenerateRequest, model_id: str) -> Image:
    """Prove the gallery's RGB provenance and decode through one open handle."""
    from PIL import Image as PILImage
    from . import storage

    assert request.refine is not None
    (path,) = storage.reference_paths([request.refine], noun="source image")
    record = storage.get_image(request.refine)
    if record is None:
        raise ValueError("the Refine source is no longer in the live gallery")
    try:
        with PILImage.open(path) as handle:
            reason = refine_qualification.source_refusal(model_id, handle.size)
            if reason is not None:
                raise ValueError(reason)
            if handle.size != (record.width, record.height):
                raise ValueError("the Refine source dimensions no longer match its gallery record")
            source = handle.convert("RGB")
    except OSError as exc:
        raise ValueError("the Refine source image could not be read; restore its PNG before refining it") from exc
    if not record.content_hash or storage.content_hash(source) != record.content_hash:
        source.close()
        raise ValueError("the Refine source pixels no longer match its gallery provenance; restore the original PNG")
    return source


def check_source(request: GenerateRequest, model_id: str) -> None:
    """Check real gallery pixels, never a caller path or a Then-refine id."""
    if refine_qualification.applies(model_id) and request.refine is not None:
        _source(request, model_id).close()


def load_source(request: GenerateRequest, model_id: str) -> Image:
    """Validate and decode the original through one handle before fitting it."""
    from . import refine

    source = _source(request, model_id)
    try:
        prepared = refine.fitted(source, request.width, request.height, request.crop_anchor)
    except BaseException:
        source.close()
        raise
    if prepared is not source:
        source.close()
    return prepared
