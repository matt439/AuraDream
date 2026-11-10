"""The two headers every response carries, stdlib and Starlette only.

`Content-Security-Policy: frame-ancestors 'self'` stops another site framing
the page. A framed page is the app's own origin inside the frame, so a click
the framing site steers lands as a same-origin write: it passes the Host
check and `main.block_cross_site_writes`, which only ever see the app's own
Origin. `'self'` and not `'none'`: the page frames itself for a second tab
in the smokes (`smoke-jobs`, `smoke-queue-controls`, `smoke-analysis`).

`X-Content-Type-Options: nosniff` makes the browser take each file as the
type it was served with, so a stored file is never run as a script or a
stylesheet because of what its bytes look like.

This is not a script policy: no `script-src`, nothing about what the page
itself may load.
"""
from __future__ import annotations

import mimetypes
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.responses import PlainTextResponse

FRAME_POLICY = "frame-ancestors 'self'"
HEADERS = {
    "Content-Security-Policy": FRAME_POLICY,
    "X-Content-Type-Options": "nosniff",
}

# Under nosniff a browser refuses a script not served as JavaScript and a
# stylesheet not served as text/css. StaticFiles asks `mimetypes`, which on
# Windows reads the registry over its own table, and some programs register
# .js as text/plain there: without these two lines such a machine would get
# a page with no scripts.
PAGE_TYPES = {".js": "text/javascript", ".css": "text/css"}


def pin_page_types() -> None:
    """Fix the types the page's scripts and stylesheets are served with."""
    for suffix, media_type in PAGE_TYPES.items():
        mimetypes.add_type(media_type, suffix)


pin_page_types()


class ResponseHeadersMiddleware:
    """ASGI: add `HEADERS` to the start of every HTTP response.

    A plain ASGI layer, not `@app.middleware("http")`, which would wrap every
    response, the job streams included, in another task and stream (what
    `keep_awake.ActivityMiddleware` records). Added last in `main`, so it is
    the outermost of the app's layers and the Host check's 400 and the
    cross-site 403 pass through it too, with the routes and both mounts.

    The policy is appended, never replaced: a response that sets a policy of
    its own keeps it, and a browser enforces every policy it is given.

    An unhandled error is answered here, in the words Starlette's own error
    layer uses, and raised again. That layer sits outside every layer an app
    can add, so its 500 would pass through none of them; it finds a response
    already begun, sends nothing more and raises the error on for the server
    to log, as it always did.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def send_with_headers(message: Any) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
                headers = MutableHeaders(scope=message)
                if FRAME_POLICY not in headers.getlist("content-security-policy"):
                    headers.append("Content-Security-Policy", FRAME_POLICY)
                headers["X-Content-Type-Options"] = HEADERS["X-Content-Type-Options"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            if not started:
                failed = PlainTextResponse("Internal Server Error", status_code=500)
                await failed(scope, receive, send_with_headers)
            raise
