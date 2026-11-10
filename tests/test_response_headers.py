"""Every response says who may frame it and that its type is not to be guessed.

`backend/response_headers.py`: one ASGI layer, outermost of the app's own, so
the routes, both mounts and the refusals of the layers under it all carry
`frame-ancestors 'self'` and `nosniff`.
"""
from __future__ import annotations

import asyncio
import mimetypes

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from backend import main, response_headers
from backend.response_headers import FRAME_POLICY, ResponseHeadersMiddleware


def carries_both(response) -> bool:
    return (response.headers.get_list("content-security-policy") == [FRAME_POLICY]
            and response.headers.get_list("x-content-type-options") == ["nosniff"])


def test_the_policy_lets_the_page_frame_itself_and_nobody_else():
    # 'self', not 'none': the smokes open a second tab as an iframe of "/".
    assert FRAME_POLICY == "frame-ancestors 'self'"


@pytest.mark.parametrize("path, status", [
    ("/", 200),
    ("/sw.js", 200),
    ("/api/status", 200),
    ("/api/gallery", 200),
    ("/api/jobs", 200),
    ("/openapi.json", 200),
    ("/docs", 200),
    ("/static/app.js", 200),
    ("/static/style.css", 200),
    ("/static/offline.html", 200),
    ("/static/no-such-file.js", 404),
    ("/outputs/no-such-image.png", 404),
    ("/no-such-route", 404),
    ("/api/images/not-a-number", 422),
])
def test_routes_mounts_and_their_errors_carry_both_headers(client, path, status):
    response = client.get(path)
    assert response.status_code == status
    assert carries_both(response), dict(response.headers)


def test_a_mounts_stored_file_carries_both_headers(tmp_path):
    # As main mounts /outputs: the layer is added before the mount here too,
    # and the file's own response carries the headers, not only a 404.
    (tmp_path / "probe.png").write_bytes(b"not really a picture")
    app = FastAPI()
    app.add_middleware(ResponseHeadersMiddleware)
    app.mount("/outputs", StaticFiles(directory=tmp_path), name="outputs")
    with TestClient(app) as client:
        response = client.get("/outputs/probe.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert carries_both(response)


def test_the_refusals_of_the_layers_under_it_carry_both_headers(client):
    # The Host check and the cross-site write block answer without reaching a
    # route; the layer is outside both.
    rebound = client.get("/api/status", headers={"Host": "evil.example"})
    assert rebound.status_code == 400 and carries_both(rebound)
    cross_site = client.post("/api/model/unload", headers={"Origin": "http://evil.example"})
    assert cross_site.status_code == 403 and carries_both(cross_site)
    wrong_method = client.post("/api/status")
    assert wrong_method.status_code == 405 and carries_both(wrong_method)


def test_it_is_the_outermost_of_the_apps_own_layers():
    # Starlette runs the last one added first. Anything added after it would
    # answer without the headers.
    assert main.app.user_middleware[0].cls is ResponseHeadersMiddleware


def test_the_pages_scripts_and_stylesheets_keep_their_types(client):
    # nosniff refuses a script or a stylesheet served as anything else.
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/sw.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/style.css").headers["content-type"].startswith("text/css")


def test_a_registry_naming_scripts_as_text_is_overruled():
    # What Windows' registry does on some machines: .js as text/plain.
    mimetypes.add_type("text/plain", ".js")
    mimetypes.add_type("text/plain", ".css")
    try:
        assert mimetypes.guess_type("app.js")[0] == "text/plain"
    finally:
        response_headers.pin_page_types()
    assert mimetypes.guess_type("app.js")[0] == "text/javascript"
    assert mimetypes.guess_type("style.css")[0] == "text/css"


def small_app() -> FastAPI:
    app = FastAPI()

    @app.get("/own-policy")
    async def own_policy() -> PlainTextResponse:
        return PlainTextResponse("x", headers={
            "Content-Security-Policy": "default-src 'none'",
            "X-Content-Type-Options": "sniff-away",
        })

    @app.get("/same-policy")
    async def same_policy() -> PlainTextResponse:
        return PlainTextResponse("x", headers={"Content-Security-Policy": FRAME_POLICY})

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    @app.middleware("http")
    async def a_layer_under_it(request, call_next):
        return await call_next(request)

    app.add_middleware(ResponseHeadersMiddleware)
    return app


def test_a_response_with_a_policy_of_its_own_keeps_it_beside_this_one():
    with TestClient(small_app()) as client:
        own = client.get("/own-policy")
        same = client.get("/same-policy")
    # A browser enforces every policy it is given, so both hold.
    assert own.headers.get_list("content-security-policy") == ["default-src 'none'", FRAME_POLICY]
    assert own.headers.get_list("x-content-type-options") == ["nosniff"]
    assert carries_both(same)


def test_an_unhandled_error_carries_both_headers_and_is_still_raised():
    # Starlette's own error layer sits outside every layer an app can add,
    # so the layer answers the error itself, in Starlette's words, and
    # raises it on for the server to log.
    with TestClient(small_app(), raise_server_exceptions=False) as client:
        response = client.get("/boom")
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers["content-type"].startswith("text/plain")
    assert carries_both(response)
    with TestClient(small_app()) as client, pytest.raises(RuntimeError, match="boom"):
        client.get("/boom")


def test_an_error_after_the_response_began_sends_no_second_start():
    sent = []

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("boom")

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(ResponseHeadersMiddleware(inner)({"type": "http"}, receive, send))
    assert [m["type"] for m in sent] == ["http.response.start"]
    assert sent[0]["status"] == 200


def test_a_scope_that_is_not_http_passes_straight_through():
    seen = []

    async def inner(scope, receive, send):
        seen.append(send)
        await send({"type": "websocket.accept", "headers": []})

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {}

    asyncio.run(ResponseHeadersMiddleware(inner)({"type": "websocket"}, receive, send))
    assert seen == [send]
    assert sent == [{"type": "websocket.accept", "headers": []}]
