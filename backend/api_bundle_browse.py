"""Browse one explicitly selected saved bundle, independent of gallery admission."""
from __future__ import annotations

import asyncio
import re
import secrets

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from . import bundle_leases, refusal_codes
from .api_bundles import _password
from .bundle_common import BrowseUnavailable, BundleError, sentence
from .bundle_manager import BROWSE_PREFIX
from .bundle_views import BundleViews, ViewClosed, ViewsFull
from .refusals import Refused
from .schemas_bundles import ImportRequest

VIEW_HEADER = "X-Bundle-View"


class BrowseFilter(BaseModel):
    prompt: str = Field(default="", max_length=8192)
    run: str = Field(default="", max_length=512)
    starred: bool = False
    offset: int = Field(default=0, ge=0, le=2**31 - 1)


class _BrowseRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # The default validation response includes rejected input.
                raise HTTPException(422, "The bundle browsing request is invalid.") from None
            except ViewClosed:
                raise Refused(410, "This bundle view has closed. Browse the saved bundle again.",
                              refusal_codes.BUNDLE_VIEW_CLOSED) from None
            except ViewsFull as exc:
                raise HTTPException(409, str(exc)) from None
            except bundle_leases.LeaseBusy:
                raise Refused(409, "This saved bundle is in use. Try again when it is available.",
                              refusal_codes.BUNDLE_IN_USE) from None
            except FileNotFoundError:
                raise HTTPException(404, "Bundle file not found.") from None
            except BrowseUnavailable as exc:
                raise HTTPException(422, sentence(str(exc))) from None
            except (OSError, ValueError):
                # A malformed member's name or value in a verifier error
                # may contain private text. Only fixed refusals are public.
                raise HTTPException(422, "This saved bundle could not be opened and verified.") from None

        return private


router = APIRouter(route_class=_BrowseRoute)


@router.post(BROWSE_PREFIX)
async def open_bundle(request: Request, selection: ImportRequest):
    password = _password(selection.password)
    views: BundleViews = request.app.state.bundle_views
    # The page draws 32 random bytes so it can close even if it loses the
    # open receipt. API clients may instead accept a freshly generated token.
    token = request.headers.get(VIEW_HEADER)
    if token is None:
        token = secrets.token_hex(32)
    elif not re.fullmatch("[0-9a-f]{64}", token):
        raise HTTPException(422, "The bundle view token is invalid.")
    cancelled = views.reserve(token)
    try:
        worker = asyncio.get_running_loop().run_in_executor(None, views.open, token, selection.name, password, cancelled)
    except BaseException:
        views.discard_opening(token, cancelled)
        raise

    async def wait_for_disconnect():
        # FastAPI already consumed the body. A blocking receive propagates
        # through the app's middleware; a nonblocking probe can miss the event.
        while (await request.receive())["type"] != "http.disconnect":
            pass

    disconnect = asyncio.create_task(wait_for_disconnect())
    try:
        await asyncio.wait({worker, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if disconnect.done():
            disconnect.result()
            raise HTTPException(400, "The bundle view was closed.")
        worker.result()
        return {"name": selection.name, "token": token, "idle_seconds": views.idle_seconds}
    except BaseException:
        cancelled.set()
        await asyncio.to_thread(views.close, token, cancelled)
        # A verification already in a decoder still owns its resources.
        # Its cancellation flag discards the result; retrieve any exception.
        worker.add_done_callback(lambda future: None if future.cancelled() else future.exception())
        raise
    finally:
        disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)


@router.post(BROWSE_PREFIX + "/images")
def list_images(request: Request, filters: BrowseFilter):
    views: BundleViews = request.app.state.bundle_views
    with views.read(request.headers.get(VIEW_HEADER, "")) as archive:
        return JSONResponse(archive.images(filters.prompt, filters.run, filters.starred, filters.offset))


@router.get(BROWSE_PREFIX + "/thumb/{image_id}")
def thumbnail(request: Request, image_id: int):
    token = request.headers.get(VIEW_HEADER, "")
    views: BundleViews = request.app.state.bundle_views
    try:
        with views.read(token) as archive:
            return Response(archive.thumbnail(image_id), media_type="image/webp")
    except KeyError:
        raise HTTPException(404, "This image is not in the saved bundle.") from None
    except (BundleError, OSError):
        # A changed or unreadable input can no longer support its open view.
        views.close(token)
        raise ViewClosed from None


@router.delete(BROWSE_PREFIX, status_code=204)
def close_bundle(request: Request):
    views: BundleViews = request.app.state.bundle_views
    views.close(request.headers.get(VIEW_HEADER, ""))
    return Response(status_code=204)
