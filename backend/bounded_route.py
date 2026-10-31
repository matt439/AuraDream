"""The route class a router with a bounded JSON import uses.

Saved setups (`api_presets.py`) and personal snippets (`api_snippets.py`)
each read their import bodies as bounded raw bytes and parse them with
their own strict parser before FastAPI's does, so a duplicate object key is
refused rather than silently dropped. `bounded_json_route` builds that
route class once; each router keeps its own sizes, words, parser and error
mapping (`translate`).
"""
from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from fastapi import HTTPException
from fastapi.routing import APIRoute


def bounded_json_route(
    *,
    import_paths: Collection[str],
    max_bytes: Callable[[], int],
    too_large: str,
    parse: Callable[..., Any],
    translate: Callable[[Any, Exception], None],
) -> type[APIRoute]:
    """An `APIRoute` class whose POSTs to `import_paths` read at most
    `max_bytes()` bytes (413 `too_large` past them) and are parsed by
    `parse(body, max_bytes=...)` before the handler runs. Any exception the
    read or the handler raises goes to `translate(request, exc)`, which
    raises the router's own answer for it or returns to let it propagate.
    The bound and the parser are read per request, through the router's
    module, so a test patching `MAX_IMPORT_REQUEST_BYTES` there still
    reaches them."""

    class BoundedJsonRoute(APIRoute):
        def get_route_handler(self):
            handler = super().get_route_handler()

            async def validated(request):
                try:
                    if request.method == "POST" and request.url.path in import_paths:
                        limit = max_bytes()
                        # Consume bounded raw bytes before FastAPI's JSON parser can
                        # discard duplicate object keys. Never trust Content-Length.
                        raw = bytearray()
                        async for chunk in request.stream():
                            if len(raw) + len(chunk) > limit:
                                raise HTTPException(status_code=413, detail=too_large)
                            raw.extend(chunk)
                        request._body = bytes(raw)
                        request._json = parse(request._body, max_bytes=limit)
                    return await handler(request)
                except Exception as exc:
                    translate(request, exc)
                    raise

            return validated

    return BoundedJsonRoute
