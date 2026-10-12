"""An inpaint's mask: stored from the Inpaint dialog's upload and served back
(`POST /api/masks`, `GET /api/masks/{hash}`).

A mask is a picture the user painted over a gallery image, kept by the hash
of its pixels (`storage.mask_hash`) and asked for by that hash alone - never
by a path. It is no gallery image: no row in `images`, no card, never a
reference or a source, and what is stored can only ever be used as a mask
over the image it was sized to, so the upload is not a way to import a
picture. Model-free: nothing here reads the checkpoint, so the switch never
waits on it; `api_bundles.GalleryAdmission` counts it as a gallery reader.

An outpaint's mask is never uploaded - the server builds it at submit
(`storage.stamp_outpaints`) - but its presets are answered here
(`GET /api/images/{id}/outpaint-presets`): the padding each one gives over a
source, decided by the server (`outpaint.presets`), model-free like the rest.
"""
from __future__ import annotations

import asyncio
import io

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse

from . import outpaint, refine, storage
from .app_context import AppServices
from .schemas import MaskStored, OutpaintPreset, OutpaintPresets

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_HASH = r"^[0-9a-f]{64}$"


def _body_cap(services: AppServices) -> int:
    """The most bytes one upload may be: an RGBA PNG of the largest picture a
    refine pass takes, stored without compression, and a margin for its
    chunks. The page sends one `L` channel, a small fraction of this."""
    return services.settings().refine_max_pixels * 4 + (1 << 20)


async def _capped_body(request: Request, cap: int) -> bytes:
    """The body, refused past `cap` before or while it arrives."""
    refusal = HTTPException(
        status_code=413,
        detail=f"The mask is over {cap // (1 << 20)} MiB, more than any picture a refine pass takes; "
        "it was not stored.",
    )
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > cap:
        raise refusal
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > cap:
            raise refusal
    return bytes(body)


def _store(services: AppServices, data: bytes, source: int) -> MaskStored:
    """Decode, check and keep one upload; `ValueError` is a refusal in words,
    `LookupError` an unknown source. Blocking: decodes and writes."""
    from PIL import Image, UnidentifiedImageError

    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("a mask is sent as a PNG, and this is not one")
    # The source first: live, its file there - the words a submission would
    # use for it, so a dialog over a binned image says the same thing.
    storage.reference_paths([source], "source image")
    record = storage.get_image(source)
    if record is None:
        raise LookupError(f"no gallery image with id {source}")
    cap = services.settings().refine_max_pixels
    try:
        with Image.open(io.BytesIO(data)) as handle:
            if handle.format != "PNG":
                raise ValueError("a mask is sent as a PNG, and this is not one")
            # The header's size, before a pixel is decoded: a picture past
            # the guard is refused without being held in memory.
            if handle.width * handle.height > cap:
                raise ValueError(
                    f"a {handle.width}x{handle.height} mask is past the {cap:,} pixels a refine pass takes"
                )
            if handle.size != (record.width, record.height):
                raise ValueError(
                    f"the mask is {handle.width}x{handle.height}, but image {source} is "
                    f"{record.width}x{record.height}; a mask covers its image exactly"
                )
            # Scaled from 16 bits and laid on black where it has alpha, as
            # the CLI's --mask is: what was painted, not what Pillow's
            # `convert` makes of it.
            mask = refine.canonical_mask(handle)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError(f"the mask PNG cannot be read ({exc})") from None
    # An all-white mask is allowed - it repaints the whole picture, a refine
    # at that strength - and an all-black one paints nothing to run.
    if mask.getextrema()[1] == 0:
        raise ValueError("nothing is masked: paint the part of the image to repaint")
    digest, created = storage.store_mask(mask)
    return MaskStored(hash=digest, width=mask.width, height=mask.height, created=created)


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter()

    @router.post("/api/masks", response_model=MaskStored)
    async def store_mask(
        request: Request,
        source: int = Query(ge=1, description="the gallery id of the image the mask covers"),
    ) -> MaskStored:
        """Keep the PNG in the body as a mask over gallery image `source`.

        Converted to one luminance channel, white repainted; refused when it
        is no PNG, past the refine guard's pixels, another size than its
        image, or paints nothing. The same picture answers the same hash, so
        uploading after every stroke stores each painting once.
        """
        data = await _capped_body(request, _body_cap(services))
        try:
            return await asyncio.to_thread(_store, services, data, source)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            message = str(exc)
            # "no gallery image with id N" from the source check is a 404.
            status = 404 if message.startswith("no gallery image") else 422
            raise HTTPException(status_code=status, detail=message[:1].upper() + message[1:] + ".") from exc
        except OSError as exc:
            # The disk, or the gallery cleared while this waited on it.
            raise HTTPException(
                status_code=503,
                detail=f"The mask could not be written ({exc.strerror or exc}); paint again to send it once more.",
            ) from exc

    @router.post("/api/masks/{mask}/keep", response_model=MaskStored)
    async def keep_mask(mask: str = Path(pattern=_HASH)) -> MaskStored:
        """Keep a stored mask from the sweep for another settle window, as
        uploading it again would: what the Inpaint dialog sends for a reused
        mask, and while it stays open on one. 404 where it has gone, which
        the dialog answers by uploading the picture it still holds."""
        try:
            await asyncio.to_thread(storage.touch_mask, mask)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        size = await asyncio.to_thread(storage.mask_size, mask)
        if size is None:
            raise HTTPException(status_code=404, detail=f"no stored mask {mask[:16]}; paint it again")
        return MaskStored(hash=mask, width=size[0], height=size[1], created=False)

    @router.get("/api/masks/{mask}")
    async def read_mask(mask: str = Path(pattern=_HASH)):
        """A stored mask's PNG, by its hash: for Reuse and the lightbox's
        outline. Through the table, never a path joined from the request."""
        try:
            path = await asyncio.to_thread(storage.mask_path, mask)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # Content-addressed: the bytes behind a hash never change.
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=31536000, immutable"})

    @router.get("/api/images/{image_id}/outpaint-presets", response_model=OutpaintPresets)
    async def outpaint_presets(image_id: int = Path(ge=1)) -> OutpaintPresets:
        """Every outpaint preset over gallery image `image_id`: "+25%" and a
        "to a:b" per aspect of the size presets, each with its four sides,
        its canvas and the words that hold it - already at the ratio, past
        the refine guard, or a source off the 16-pixel grid (`off_grid`,
        which holds every one). 404 for an image that is gone or binned, a
        binned one saying so - the page offers a Restore rather than calling
        it deleted."""
        record = await asyncio.to_thread(storage.get_image, image_id, True)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no gallery image with id {image_id}")
        if record.binned_at is not None:
            raise HTTPException(
                status_code=404,
                detail=f"no gallery image with id {image_id} (#{image_id} is in the recycle bin)",
            )
        limits = services.settings()
        name = f"#{record.id}"
        answers = outpaint.presets(
            record.width, record.height, max_side=limits.refine_max_side,
            max_pixels=limits.refine_max_pixels, name=name,
        )
        return OutpaintPresets(
            id=record.id, width=record.width, height=record.height,
            off_grid=outpaint.source_off_grid(record.width, record.height, name),
            max_side=limits.refine_max_side, max_pixels=limits.refine_max_pixels,
            fill=outpaint.FILL, band=outpaint.BAND, max_band=outpaint.MAX_BAND, min_kept=outpaint.MIN_KEPT,
            presets=[OutpaintPreset(**answer) for answer in answers],
        )

    return router
