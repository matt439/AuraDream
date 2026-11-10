"""Every surface the server has, walked from the app's own route table.

The table is `main.app.routes`: every API route and method, the page, the
service worker, both mounts and FastAPI's four documentation routes. With
no passcode asked each answers a caller that sends no credentials, as it
always has. With one asked (`backend/passcode.py`) each is a 401 except the
short open list, so a route added later is covered here or this fails.
"""
from __future__ import annotations

import re

import pytest
from fastapi.routing import APIRoute
from starlette.routing import Mount, Route

from backend import main, passcode

PASSCODE = "walk-the-table-9"
DOCUMENTATION = {"/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.json"}
# One file under each mount: a page file, the offline page, a stored image.
MOUNTED = ["/static/app.js", "/static/offline.html", "/static/no-such-file.js", "/outputs/no-such-image.png"]


def _concrete(path: str) -> str:
    """A route's path with every parameter filled in."""
    return re.sub(r"\{[^}]+\}", "1", path)


def _walk(routes, prefix: str, found: set[tuple[str, str]]) -> None:
    for route in routes:
        included = getattr(route, "original_router", None)
        if isinstance(route, Mount):
            found.update(("GET", path) for path in MOUNTED if path.startswith(prefix + route.path + "/"))
        elif isinstance(route, Route | APIRoute):
            for method in route.methods or ():
                found.add((method, _concrete(prefix + route.path)))
        elif included is not None:
            # An `include_router`, which this FastAPI keeps as one entry.
            _walk(included.routes, prefix + route.include_context.prefix, found)
        else:
            pytest.fail(f"a kind of route this walk does not know: {type(route).__name__}")


def surfaces() -> list[tuple[str, str]]:
    """Every (method, path) the app answers, from its route table."""
    found: set[tuple[str, str]] = set()
    _walk(main.app.routes, "", found)
    return sorted(found)


def test_the_walk_misses_nothing_the_schema_lists():
    # The schema is FastAPI's own reading of the same table.
    found = set(surfaces())
    listed = {
        (method.upper(), _concrete(path))
        for path, operations in main.app.openapi()["paths"].items()
        for method in operations
    }
    assert listed and listed <= found


def test_the_walk_sees_the_whole_app():
    found = surfaces()
    paths = {path for _, path in found}
    assert DOCUMENTATION <= paths
    assert {"/", "/sw.js", "/api/status", "/api/generate", "/api/jobs/1/stream"} <= paths
    assert set(MOUNTED) <= paths
    assert sorted(route.path for route in main.app.routes if isinstance(route, Mount)) == ["/outputs", "/static"]
    # 88 API operations when this was written; more is fine, far fewer is a walk gone wrong.
    assert len([1 for method, path in found if path.startswith("/api/") and method != "HEAD"]) >= 88
    # The layer's own two routes are not the app's: nothing here walks them.
    assert not paths & {passcode.SIGN_IN_PATH, passcode.SIGN_OUT_PATH}


def test_with_no_passcode_asked_every_surface_answers_without_credentials(client):
    assert main._passcode_gate is None
    asked = []
    for method, path in surfaces():
        response = client.request(method, path)
        if response.status_code in (401, 403, 429) or "set-cookie" in response.headers:
            asked.append((method, path, response.status_code))
    assert asked == []


@pytest.fixture
def gate(monkeypatch):
    installed = passcode.Gate(PASSCODE, port=main.settings.port)
    monkeypatch.setattr(main, "_passcode_gate", installed)
    return installed


def test_with_the_passcode_asked_every_surface_is_closed_but_the_open_list(client, gate):
    open_answers = {}
    wrong = []
    for method, path in surfaces() + [("GET", path) for path in sorted(passcode.OPEN_PATHS)]:
        response = client.request(method, path)
        if (method == "GET" and path == "/") or (method == "GET" and path in passcode.OPEN_PATHS):
            open_answers[(method, path)] = response.status_code
            continue
        closed = (
            response.status_code == 401
            and response.headers.get("www-authenticate") == "Bearer"
            and (method == "HEAD" or response.json() == {
                "detail": "This server asks for a passcode. Reload the page and sign in.",
                "code": "passcode_required",
            })
        )
        if not closed:
            wrong.append((method, path, response.status_code))
    assert wrong == []
    # The page's address is the sign-in page; the manifest and its icons pass.
    assert open_answers == {
        ("GET", "/"): 200,
        **{("GET", path): 200 for path in passcode.OPEN_PATHS},
    }
    # Walking the table was nobody's guess at the passcode.
    assert gate.limiter.wrong_in_a_row == 0


def test_signed_in_the_same_walk_reaches_the_app_again(client, gate):
    headers = {"Authorization": f"Bearer {PASSCODE}"}
    for method, path in [("GET", "/api/status"), ("GET", "/static/app.js"), ("GET", "/sw.js"),
                         ("GET", "/openapi.json"), ("GET", "/docs"), ("GET", "/api/gallery")]:
        assert client.request(method, path, headers=headers).status_code == 200, (method, path)
    assert client.get("/outputs/no-such-image.png", headers=headers).status_code == 404
