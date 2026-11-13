"""The passcode over HTTP: the layer in its place in `main`, its two routes,
the sign-in page, and `main.run`'s refusal and log lines.

The parts alone are tests/test_passcode.py; every route, walked from the
app's table, is tests/test_api_passcode_routes.py. No test here puts the
passcode in a URL or a process argument.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from backend import main, passcode
from backend.response_headers import FRAME_POLICY
from tests.api_helpers import ORIGIN, run_one, stream_job

PASSCODE = "seven-gulls-over-the-pier"
WRONG = "six-gulls-under-the-pier"
BEARER = {"Authorization": f"Bearer {PASSCODE}"}
WEB = Path(__file__).resolve().parent.parent / "web"
REQUIRED = {
    "detail": "This server asks for a passcode. Reload the page and sign in.",
    "code": "passcode_required",
}


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


@pytest.fixture
def clocks():
    """(the limiter's monotonic clock, the token's wall clock)."""
    return Clock(500.0), Clock(1_800_000_000.0)


@pytest.fixture
def gate(monkeypatch, clocks):
    installed = passcode.Gate(PASSCODE, port=main.settings.port, clock=clocks[0], now=clocks[1])
    monkeypatch.setattr(main, "_passcode_gate", installed)
    return installed


def sign_in(client, value: str = PASSCODE, **kwargs):
    return client.post("/api/passcode", json={"passcode": value}, **kwargs)


def session_cookie(response) -> str:
    cookies = response.headers.get_list("set-cookie")
    assert len(cookies) == 1
    return cookies[0]


# -------------------------------------------------------------- the gate off


def test_with_no_passcode_asked_nothing_changes_and_no_cookie_is_set(client):
    assert main._passcode_gate is None
    page = client.get("/")
    assert page.status_code == 200 and page.content == (WEB / "index.html").read_bytes()
    assert "set-cookie" not in page.headers and "cache-control" not in page.headers
    status = client.get("/api/status")
    assert status.status_code == 200 and status.json()["passcode"] is False
    assert "set-cookie" not in status.headers and "www-authenticate" not in status.headers
    # The layer's own routes are nobody's while it is off: the app's 404.
    assert sign_in(client).status_code == 404
    assert client.post("/api/passcode/sign-out").status_code == 404
    # A header or a cookie it would read is not looked at.
    assert client.get("/api/status", headers={"Authorization": f"Bearer {WRONG}"}).status_code == 200
    client.cookies.set(f"auradream_session_{main.settings.port}", "v1.1.aa")
    assert client.get("/api/gallery").status_code == 200
    assert client.head("/").status_code == 405


def test_the_layers_stand_in_the_designs_order():
    # Outermost first: the response headers, the cross-site write block, the
    # Host check, the passcode, the model hold, activity, gallery admission.
    layers = main.app.user_middleware
    names = [getattr(layer.kwargs.get("dispatch"), "__name__", layer.cls.__name__) for layer in layers]
    assert names == [
        "ResponseHeadersMiddleware", "block_cross_site_writes", "TrustedHostMiddleware",
        "PasscodeMiddleware", "hold_model_for_request", "ActivityMiddleware", "GalleryAdmission",
    ]


# ---------------------------------------------------------- the sign-in page


def test_signed_out_the_pages_address_is_the_sign_in_page(client, gate):
    page = client.get("/")
    assert page.status_code == 200 and not page.history
    assert page.headers["content-type"] == "text/html; charset=utf-8"
    assert page.headers["cache-control"] == "no-store"
    assert page.content == (WEB / "passcode.html").read_bytes()
    assert "set-cookie" not in page.headers
    # What the title probes read (`server_probe.auradream_at`, the scripts).
    assert b"<title>AuraDream</title>" in page.content[:65536]
    head = client.head("/")
    assert head.status_code == 200 and head.headers["cache-control"] == "no-store"
    # Only that exact address: nothing else is the page.
    assert client.get("/index.html").status_code == 401
    assert client.post("/").status_code == 401


def test_the_sign_in_page_is_self_contained():
    text = (WEB / "passcode.html").read_text(encoding="utf-8")
    assert "<title>AuraDream</title>" in text
    # Nothing under /static/ answers before sign-in: no file of the page's.
    assert "/static/" not in re.sub(r"<!--.*?-->", "", text, flags=re.S)
    assert not re.search(r"<(script|link|img)\b[^>]*\b(src|href)=", text)
    assert "serviceWorker" not in text
    assert 'type="password"' in text and 'autocomplete="current-password"' in text
    assert re.search(r'<label for="passcode">', text) and 'aria-live="polite"' in text
    assert "min-height: 44px" in text
    assert "'/api/passcode'" in text and "location.replace('/')" in text
    # The passcode travels in the body, never the address.
    assert "?passcode" not in text and "location.search" not in text


def test_the_open_list_is_what_the_manifest_and_the_pages_head_name(client, gate):
    manifest = json.loads((WEB / "manifest.webmanifest").read_text(encoding="utf-8"))
    head = (WEB / "index.html").read_text(encoding="utf-8").split("</head>", 1)[0]
    named = {icon["src"] for icon in manifest["icons"]}
    named |= set(re.findall(r'<link rel="(?:icon|apple-touch-icon|manifest)" href="([^"]+)"', head))
    assert named == passcode.OPEN_PATHS
    for path in sorted(passcode.OPEN_PATHS):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "set-cookie" not in response.headers
        assert client.post(path).status_code == 401
    # The page's other head files stay shut.
    assert client.get("/static/style.css").status_code == 401
    assert client.get("/static/icons/").status_code == 401


@pytest.mark.parametrize("path", [
    # The open list and the page's address are matched exactly: no doubled
    # or trailing slash, no other case, no way round through an open folder.
    "//", "/%2e", "/%2f", "/index.html", "/static", "/static/",
    "/static/manifest.webmanifest/", "/static//manifest.webmanifest", "//static/manifest.webmanifest",
    "/STATIC/manifest.webmanifest", "/static/Manifest.webmanifest", "/static/manifest.webmanifest.bak",
    "/static/icons", "/static/icons/", "/static/icons/icon.svg/", "/static/icons/ICON.SVG",
    "/static/icons/%2e%2e/app.js", "/static/icons/..%2fapp.js", "/static/icons/icon.svg%2f..%2f..%2fapp.js",
    "/static/icons/icon.svg%00", "/static/manifest.webmanifest;x=1", "/static%2fapp.js",
    "/outputs/", "/api", "/api/", "/api/status/", "/API/status", "/api//status", "/sw.js/", "/SW.JS",
    "/api/passcode/", "/api/Passcode", "/api/passcode/sign-out/",
])
def test_no_spelling_of_a_path_gets_round_the_gate(client, gate, path):
    for method in ("GET", "HEAD", "POST", "OPTIONS", "PUT", "DELETE", "PATCH", "TRACE", "PROPFIND"):
        # The whole address, so the client sends a doubled slash as it is.
        response = client.request(method, ORIGIN + path, follow_redirects=False)
        assert response.status_code == 401, (method, path)
        assert response.headers["www-authenticate"] == "Bearer"
        assert "set-cookie" not in response.headers
    assert gate.limiter.wrong_in_a_row == 0


def test_a_query_or_an_odd_method_opens_nothing_more(client, gate):
    # The page's address with a query is still only the sign-in page.
    page = client.get("/?next=/api/status")
    assert page.status_code == 200 and page.content == (WEB / "passcode.html").read_bytes()
    for method in ("POST", "OPTIONS", "PUT", "DELETE", "PATCH", "TRACE", "PROPFIND"):
        assert client.request(method, "/").status_code == 401, method
        for path in sorted(passcode.OPEN_PATHS):
            assert client.request(method, path).status_code == 401, (method, path)
    # An encoded spelling of an open file is that same file and nothing else.
    encoded = client.get("/static/icons/icon%2Esvg")
    assert encoded.status_code == 200 and encoded.content == (WEB / "icons" / "icon.svg").read_bytes()
    # A preflight carries no cookie: it is refused like any other request.
    preflight = client.options("/api/status", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert preflight.status_code == 401 and "access-control-allow-origin" not in preflight.headers


# -------------------------------------------------- signing in and the cookie


def test_the_right_passcode_is_a_204_and_the_cookie(client, gate, clocks):
    response = sign_in(client)
    assert response.status_code == 204 and response.content == b""
    cookie = session_cookie(response)
    name = f"auradream_session_{main.settings.port}"
    value, *attributes = cookie.split("; ")
    assert value.startswith(name + "=v1.")
    assert sorted(attributes) == ["HttpOnly", "Max-Age=2592000", "Path=/", "SameSite=Lax"]
    assert gate.valid(value.split("=", 1)[1])
    assert value.split(".")[1] == str(int(clocks[1].at) + 30 * 24 * 3600)
    # The browser now carries it: the app answers.
    status = client.get("/api/status")
    assert status.status_code == 200 and status.json()["passcode"] is True
    page = client.get("/")
    assert page.content == (WEB / "index.html").read_bytes()
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/sw.js").status_code == 200
    assert client.get("/static/offline.html").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_a_passcode_typed_with_a_space_round_it_signs_in(client, gate):
    assert sign_in(client, f"  {PASSCODE} ").status_code == 204


def test_the_cookie_is_secure_only_over_https(fake, gate):
    with TestClient(main.app, base_url=ORIGIN.replace("http://", "https://")) as https:
        assert session_cookie(sign_in(https)).endswith("; SameSite=Lax; Secure")
        assert session_cookie(https.post("/api/passcode/sign-out")).endswith("; Secure")
    with TestClient(main.app, base_url=ORIGIN) as http:
        assert "Secure" not in session_cookie(sign_in(http))
        assert "Secure" not in session_cookie(http.post("/api/passcode/sign-out"))


def test_a_loopback_proxys_forwarded_scheme_makes_the_cookie_secure(fake, gate, monkeypatch):
    # Behind a TLS proxy on this machine the scheme is the proxy's word,
    # which uvicorn takes from 127.0.0.1 only (its default, and `main.run`
    # leaves it so): the layer it puts round the app stands here.
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    served = []

    class Kept(Quiet):
        def __init__(self, config) -> None:
            super().__init__(config)
            served.append(config)

    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, port=0))
    monkeypatch.setattr(main, "Server", Kept)
    main.run()
    config = served[0]
    assert config.proxy_headers is True and config.forwarded_allow_ips == "127.0.0.1"

    forwarded = {"X-Forwarded-Proto": "https", "X-Forwarded-For": "192.168.1.50"}
    behind = ProxyHeadersMiddleware(main.app, trusted_hosts=config.forwarded_allow_ips)
    with TestClient(behind, base_url=ORIGIN, client=("127.0.0.1", 50000)) as proxied:
        assert session_cookie(sign_in(proxied, headers=forwarded)).endswith("; Secure")
        assert "Secure" not in session_cookie(sign_in(proxied))
    # The same header from another machine is not believed.
    with TestClient(behind, base_url=ORIGIN, client=("192.168.1.66", 50000)) as direct:
        assert "Secure" not in session_cookie(sign_in(direct, headers=forwarded))


def test_a_wrong_passcode_is_a_401_in_its_own_words(client, gate, caplog):
    with caplog.at_level("DEBUG"):
        response = sign_in(client, WRONG)
    assert response.status_code == 401
    assert response.json() == {"detail": "That passcode is not right."}
    assert "set-cookie" not in response.headers
    assert gate.limiter.wrong_in_a_row == 1
    assert client.get("/api/status").status_code == 401
    warnings = [record for record in caplog.records if record.name == "auradream.passcode"]
    assert [record.levelname for record in warnings] == ["WARNING"]
    # The peer's address and never the value.
    assert "testclient" in warnings[0].getMessage()
    assert WRONG not in caplog.text and PASSCODE not in caplog.text


def test_the_passcode_is_read_from_a_json_body_only(client, gate):
    assert client.post("/api/passcode", content=b"not json").status_code == 422
    assert client.post("/api/passcode", json=[PASSCODE]).status_code == 422
    assert client.post("/api/passcode", json={"passcode": 12345678}).status_code == 422
    assert client.post("/api/passcode", json={"password": PASSCODE}).status_code == 422
    assert client.post("/api/passcode").status_code == 422
    assert client.post("/api/passcode", data={"passcode": PASSCODE}).status_code == 422
    too_long = client.post("/api/passcode", json={"passcode": PASSCODE, "padding": "x" * 5000})
    assert too_long.status_code == 413
    # None of those was a guess.
    assert gate.limiter.wrong_in_a_row == 0
    assert all("set-cookie" not in r.headers for r in [too_long])
    # A value past 256 characters is refused unread, as a wrong one.
    over = sign_in(client, PASSCODE + " " * 10 + "x" * 300)
    assert over.status_code == 401 and gate.limiter.wrong_in_a_row == 1
    # Other methods on the layer's routes are nobody's.
    assert client.get("/api/passcode").status_code == 401
    assert client.get("/api/passcode/sign-out").status_code == 401


def test_a_body_sent_in_pieces_is_cut_off_at_four_kib_too(client, gate):
    # No Content-Length to refuse it by: the limit is on what is read.
    def pieces(total: int):
        yield b'{"passcode": "'
        for _ in range(total // 512):
            yield b"x" * 512
        yield b'"}'

    long = client.post("/api/passcode", content=pieces(64 * 1024))
    assert "content-length" not in long.request.headers
    assert long.status_code == 413 and "set-cookie" not in long.headers
    assert gate.limiter.wrong_in_a_row == 0
    # Under the limit the same way of sending is read as any other.
    whole = json.dumps({"passcode": PASSCODE}).encode()
    short = client.post("/api/passcode", content=iter([whole[:5], whole[5:]]))
    assert "content-length" not in short.request.headers and short.status_code == 204


def test_sign_out_clears_the_cookie_on_this_device_and_needs_no_session(client, gate):
    signed_out = client.post("/api/passcode/sign-out")
    assert signed_out.status_code == 204
    name = f"auradream_session_{main.settings.port}"
    assert session_cookie(signed_out) == f"{name}=; HttpOnly; Max-Age=0; Path=/; SameSite=Lax"
    assert sign_in(client).status_code == 204
    assert client.get("/api/status").status_code == 200
    assert client.post("/api/passcode/sign-out").status_code == 204
    assert client.get("/api/status").status_code == 401
    assert client.get("/").content == (WEB / "passcode.html").read_bytes()


def test_a_cookie_that_is_not_ours_is_signed_out_and_no_attempt(client, gate, clocks):
    name = gate.cookie_name
    token = gate.mint()
    version, expires, mac = token.split(".")
    for bad in ["", "v1.1.aa", f"{version}.{int(expires) + 1}.{mac}", f"v2.{expires}.{mac}", token + "0"]:
        client.cookies.clear()
        client.cookies.set(name, bad)
        assert client.get("/api/status").status_code == 401, bad
    client.cookies.clear()
    # The same token under another server's port name is not this one's.
    client.cookies.set(f"auradream_session_{main.settings.port + 1}", token)
    assert client.get("/api/status").status_code == 401
    client.cookies.clear()
    client.cookies.set(name, token)
    assert client.get("/api/status").status_code == 200
    # Thirty days on it has expired.
    clocks[1].at += 30 * 24 * 3600
    assert client.get("/api/status").status_code == 401
    assert gate.limiter.wrong_in_a_row == 0


def test_the_page_renews_a_cookie_a_day_old_and_is_never_served_cached(client, gate, clocks):
    assert sign_in(client).status_code == 204
    first = client.cookies.get(gate.cookie_name)
    page = client.get("/")
    assert page.headers["cache-control"] == "no-cache" and "set-cookie" not in page.headers
    clocks[1].at += 24 * 3600 + 60
    # A poll or a file does not renew: an idle open tab is not someone using it.
    assert "set-cookie" not in client.get("/api/status").headers
    assert "set-cookie" not in client.get("/static/app.js").headers
    assert "cache-control" not in client.get("/api/status").headers
    page = client.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-cache"
    renewed = session_cookie(page)
    fresh = renewed.split("; ")[0].split("=", 1)[1]
    assert fresh != first and gate.valid(fresh) and not gate.renewable(fresh)
    assert sorted(renewed.split("; ")[1:]) == ["HttpOnly", "Max-Age=2592000", "Path=/", "SameSite=Lax"]
    # So a device used within the month stays signed in past the first's end.
    clocks[1].at += 29 * 24 * 3600 + 3600
    assert not gate.valid(first)
    assert client.get("/api/status").status_code == 200


def test_a_counted_api_request_renews_a_cookie_a_day_old_for_a_tab_left_open(client, gate, clocks):
    """DEV#56: a tab or PWA left open and used daily never loads `/` again,
    so a request someone makes (`keep_awake.counts_as_activity`) renews too;
    the page's timed polls, job streams and files still never do."""
    assert sign_in(client).status_code == 204
    first = client.cookies.get(gate.cookie_name)
    # Under a day old: nothing is renewed, counted or not.
    assert "set-cookie" not in client.get("/api/gallery").headers
    clocks[1].at += 24 * 3600 + 60
    for path in ("/api/status", "/api/jobs", "/api/gallery/bundles/status", "/static/app.js"):
        assert "set-cookie" not in client.get(path).headers, path
    assert "set-cookie" not in client.head("/api/gallery").headers
    answer = client.get("/api/gallery")
    assert answer.status_code == 200
    renewed = session_cookie(answer)
    fresh = renewed.split("; ")[0].split("=", 1)[1]
    assert fresh != first and gate.valid(fresh) and not gate.renewable(fresh)
    assert sorted(renewed.split("; ")[1:]) == ["HttpOnly", "Max-Age=2592000", "Path=/", "SameSite=Lax"]
    assert client.cookies.get(gate.cookie_name) == fresh
    # Renewed once: the next counted request carries the fresh cookie.
    assert "set-cookie" not in client.get("/api/gallery").headers
    # A write renews as well, whatever it answers.
    clocks[1].at += 24 * 3600 + 60
    assert session_cookie(client.post("/api/model/unload"))
    # A month after the first sign-in the tab is still signed in.
    clocks[1].at += 28 * 24 * 3600
    assert not gate.valid(first)
    assert client.get("/api/status").status_code == 200


def test_a_request_the_page_marks_as_a_poll_never_renews(client, gate, clocks):
    """DEV#64: the page's timers that reuse a counted route send
    `keep_awake.POLL_HEADER` (the catalogue and model-shots retries, the
    maintenance re-reads, the grid preview's re-reads). They keep firing in
    a tab nobody uses, so they never renew, whatever they answer; the same
    requests unmarked still do."""
    from backend.keep_awake import POLL_HEADER

    marked = {POLL_HEADER: "1"}
    assert sign_in(client).status_code == 204
    clocks[1].at += 24 * 3600 + 60
    assert "set-cookie" not in client.get("/api/models", headers=marked).headers
    preview = client.post("/api/grid/preview", json={}, headers=marked)
    assert "set-cookie" not in preview.headers, preview.status_code
    # A marked write that is not a polled route still counts.
    assert session_cookie(client.post("/api/model/unload", headers=marked))
    clocks[1].at += 24 * 3600 + 60
    assert session_cookie(client.get("/api/models"))
    clocks[1].at += 24 * 3600 + 60
    assert session_cookie(client.post("/api/grid/preview", json={}))


def test_a_bearer_request_is_never_given_a_renewed_cookie(client, gate, clocks):
    clocks[1].at += 24 * 3600 + 60
    answer = client.get("/api/gallery", headers=BEARER)
    assert answer.status_code == 200 and "set-cookie" not in answer.headers


# ------------------------------------------------------------------- Bearer


def test_the_passcode_as_a_bearer_header_passes_each_request(client, gate):
    assert client.get("/api/status").status_code == 401
    status = client.get("/api/status", headers=BEARER)
    assert status.status_code == 200 and status.json()["passcode"] is True
    # No session comes of it: the header is sent each time.
    assert "set-cookie" not in status.headers
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"Authorization": f"bearer   {PASSCODE}  "}).status_code == 200
    assert client.post("/api/model/unload", headers=BEARER).status_code == 200
    page = client.get("/", headers=BEARER)
    assert page.content == (WEB / "index.html").read_bytes()
    assert gate.limiter.wrong_in_a_row == 0


def test_a_wrong_bearer_counts_as_a_wrong_passcode(client, gate, caplog):
    with caplog.at_level("DEBUG"):
        response = client.get("/api/status", headers={"Authorization": f"Bearer {WRONG}"})
    assert response.status_code == 401 and response.json() == REQUIRED
    assert response.headers["www-authenticate"] == "Bearer"
    assert gate.limiter.wrong_in_a_row == 1
    assert WRONG not in caplog.text and "wrong passcode from testclient" in caplog.text
    # Another scheme, or an empty one, is no attempt at all.
    for value in ["Basic dXNlcjpwYXNz", "Bearer", "Bearer    ", PASSCODE, f"Token {PASSCODE}"]:
        assert client.get("/api/status", headers={"Authorization": value}).status_code == 401, value
    assert gate.limiter.wrong_in_a_row == 1
    # A signed-in browser's stray header is not one either.
    assert sign_in(client).status_code == 204
    assert client.get("/api/status", headers={"Authorization": f"Bearer {WRONG}"}).status_code == 200
    assert gate.limiter.wrong_in_a_row == 0


def test_a_passcode_outside_ascii_passes_as_a_header_and_in_the_form(client, monkeypatch):
    spoken = "pässwörd ☂ mit Leerzeichen"
    monkeypatch.setattr(main, "_passcode_gate", passcode.Gate(spoken, port=main.settings.port))
    utf8 = {"Authorization": b"Bearer " + spoken.encode("utf-8")}
    assert client.get("/api/status", headers=utf8).status_code == 200
    assert sign_in(client, spoken).status_code == 204


# ------------------------------------------------------------------ the limiter


def test_after_five_wrong_the_next_waits_and_the_right_one_waits_too(client, gate, clocks):
    for _ in range(5):
        assert sign_in(client, WRONG).status_code == 401
    locked = sign_in(client, WRONG)
    assert locked.status_code == 429 and locked.headers["retry-after"] == "1"
    assert locked.json() == {"detail": "Too many wrong passcodes. Wait 1 second and try again."}
    # Neither compared nor counted; the right one is refused the same way,
    # in the form and as a header.
    right = sign_in(client)
    assert right.status_code == 429 and "set-cookie" not in right.headers
    by_header = client.get("/api/status", headers=BEARER)
    assert by_header.status_code == 429 and by_header.headers["retry-after"] == "1"
    assert gate.limiter.wrong_in_a_row == 5
    # A malformed sign-in during a lock is a 429 before its body is read.
    assert client.post("/api/passcode", content=b"x" * 9000).status_code == 429
    # Without credentials it is still only a 401, and the page still shows.
    assert client.get("/api/status").status_code == 401
    assert client.get("/").status_code == 200
    clocks[0].at += 1
    assert sign_in(client, WRONG).status_code == 401
    waiting = sign_in(client)
    assert waiting.status_code == 429 and waiting.headers["retry-after"] == "2"
    assert waiting.json()["detail"] == "Too many wrong passcodes. Wait 2 seconds and try again."
    clocks[0].at += 2
    assert sign_in(client).status_code == 204
    assert gate.limiter.wrong_in_a_row == 0


def test_a_signed_in_device_is_untouched_by_a_lock(client, gate):
    assert sign_in(client).status_code == 204
    # Another device, sharing the app (and so the one limiter), keeps guessing.
    other = TestClient(main.app, base_url=ORIGIN)
    for _ in range(3):
        assert sign_in(other, WRONG).status_code == 401
    for _ in range(2):
        assert other.get("/api/gallery", headers={"Authorization": f"Bearer {WRONG}"}).status_code == 401
    assert sign_in(other).status_code == 429
    assert other.get("/api/status", headers=BEARER).status_code == 429
    # A cookie is not an attempt.
    assert client.get("/api/status").status_code == 200
    assert client.get("/").headers["cache-control"] == "no-cache"


# ------------------------------------------------ its place among the layers


def test_a_cross_site_sign_in_is_refused_without_counting(client, gate):
    # A hostile page in a signed-out browser cannot spend the attempts.
    for _ in range(8):
        response = sign_in(client, WRONG, headers={"Origin": "http://evil.example"})
        assert response.status_code == 403
    assert sign_in(client, headers={"Origin": "http://evil.example"}).status_code == 403
    assert gate.limiter.wrong_in_a_row == 0
    assert client.post("/api/passcode/sign-out", headers={"Origin": "http://evil.example"}).status_code == 403
    # The page's own origin signs in.
    assert sign_in(client, headers={"Origin": ORIGIN}).status_code == 204


def test_a_wrong_host_is_still_a_400_before_the_passcode(client, gate):
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/api/status", headers={"Host": "evil.example", **BEARER}).status_code == 400
    wrong = client.get("/api/status", headers={"Host": "evil.example", "Authorization": f"Bearer {WRONG}"})
    assert wrong.status_code == 400 and gate.limiter.wrong_in_a_row == 0


def test_a_refusal_and_the_sign_in_page_carry_the_response_headers(client, gate):
    refusal = client.get("/sw.js")
    assert refusal.status_code == 401 and refusal.headers["cache-control"] == "no-store"
    for response in [refusal, client.get("/api/status"), client.get("/"), sign_in(client, WRONG),
                     client.get("/static/app.js"), client.get("/outputs/x.png"), sign_in(client)]:
        assert response.headers.get_list("content-security-policy") == [FRAME_POLICY], response.url
        assert response.headers["x-content-type-options"] == "nosniff"


def test_a_signed_out_request_is_never_activity(client, gate, monkeypatch):
    noted = []
    monkeypatch.setattr(main._keep_awake, "note_activity", lambda: noted.append(1))
    assert client.get("/").status_code == 200                # the sign-in page
    assert client.get("/api/gallery").status_code == 401
    assert client.post("/api/generate", json={"prompt": "a lighthouse"}).status_code == 401
    assert sign_in(client, WRONG).status_code == 401
    assert noted == []
    assert sign_in(client).status_code == 204
    assert noted == []                                       # the layer's own answer
    assert client.get("/").status_code == 200
    assert client.get("/api/gallery").status_code == 200
    assert len(noted) == 2


def test_a_signed_out_request_is_never_a_model_reader(client, gate, monkeypatch):
    # During a switch a model-dependent request is a 409; one that is not
    # signed in never gets that far, nor is it counted while the model idles.
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256}
    monkeypatch.setattr(main, "_switching", True)
    assert client.post("/api/generate", json=body).status_code == 401
    assert client.get("/api/loras").status_code == 401
    assert client.post("/api/model/switch", json={}).status_code == 401
    held = client.post("/api/generate", json=body, headers=BEARER)
    assert held.status_code == 409 and "switching" in held.json()["detail"]
    monkeypatch.setattr(main, "_switching", False)
    stale = client.post("/api/generate", json=body, headers={"x-auradream-model-revision": "999"})
    assert stale.status_code == 401
    assert main._model_requests == 0 and main._model_request_paths == {}


def test_the_host_check_and_the_write_block_now_come_before_the_model_hold(client, monkeypatch):
    # The reorder, with no passcode asked: during a switch a rebound Host is
    # the Host check's 400 and a cross-site write the block's 403, where the
    # model hold's 409 used to answer first.
    monkeypatch.setattr(main, "_switching", True)
    assert client.get("/api/loras", headers={"Host": "evil.example"}).status_code == 400
    assert client.post("/api/model/unload", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/loras").status_code == 409


def test_a_websocket_is_refused_while_the_passcode_is_asked():
    # The app has none today, so a small one stands under the layer.
    async def echo(websocket):
        await websocket.accept()
        await websocket.send_text("open")
        await websocket.close()

    installed: list[passcode.Gate | None] = [None]
    small = Starlette(routes=[WebSocketRoute("/ws", echo)])
    small.add_middleware(passcode.PasscodeMiddleware, get_gate=lambda: installed[0], page=WEB / "passcode.html")
    with TestClient(small) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_text() == "open"
        installed[0] = passcode.Gate(PASSCODE, port=7860)
        # Closed before it was accepted, even with the passcode sent.
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers=BEARER):
                pass


# ---------------------------------------------------------------- the stream


def test_a_job_stream_opens_with_the_cookie_and_is_refused_without(client, gate):
    assert sign_in(client).status_code == 204
    job_id, events = run_one(client)
    assert events[-1]["type"] == "done"
    # The page's images ride the same cookie (the mount is the real
    # outputs directory, where this test's image is not: a 404 is past the gate).
    assert client.get("/outputs/no-such-image.png").status_code == 404
    client.cookies.clear()
    refused = client.get(f"/api/jobs/{job_id}/stream")
    assert refused.status_code == 401 and refused.json() == REQUIRED
    assert client.get("/outputs/no-such-image.png").status_code == 401
    # And a client that sends the passcode each time follows it too.
    client.headers.update(BEARER)
    assert stream_job(client, job_id)[-1]["type"] == "done"


# -------------------------------------------------- nothing says the passcode


def test_the_passcode_is_in_no_answer_and_no_log_line(client, gate, caplog):
    said = []
    with caplog.at_level("DEBUG"):
        said.append(client.get("/"))
        said.append(sign_in(client, WRONG))
        said.append(client.get("/api/status", headers={"Authorization": f"Bearer {WRONG}"}))
        said.append(sign_in(client))
        said.append(client.get("/api/status"))
        said.append(client.get("/api/models"))
        said.append(client.get("/openapi.json"))
        said.append(client.post("/api/passcode/sign-out"))
    for response in said:
        text = response.text + repr(dict(response.headers))
        assert PASSCODE not in text and WRONG not in text, response.url
    assert PASSCODE not in caplog.text and WRONG not in caplog.text
    assert "passcode" not in json.dumps(main.app.openapi()["paths"])


# ------------------------------------------------- the gate `main` is built with


@pytest.mark.parametrize(("environment", "expected"), [
    ({}, "None"),
    # Set, on a server this machine alone reaches: not asked.
    ({"AURADREAM_PASSCODE": PASSCODE}, "None"),
    ({"AURADREAM_PASSCODE": "   "}, "None"),
    ({"AURADREAM_HOST": "0.0.0.0"}, "None"),
    ({"AURADREAM_PASSCODE": PASSCODE, "AURADREAM_HOST": "0.0.0.0", "AURADREAM_PORT": "7999"}, "auradream_session_7999"),
    # A proxy in front of a loopback bind, by the name it is reached under.
    ({"AURADREAM_PASSCODE": PASSCODE, "AURADREAM_ALLOWED_HOSTS": "127.0.0.1,localhost,my-box"}, "auradream_session_7860"),
])
def test_main_builds_its_gate_from_the_bind_the_allowed_hosts_and_the_port(tmp_path, environment, expected):
    # A fresh interpreter: `main` builds its gate once, at import, from the
    # environment's settings. The passcode goes in by the environment only.
    kept = {name: value for name, value in os.environ.items() if not name.startswith("AURADREAM_")}
    env = {
        **kept, **environment,
        "AURADREAM_DB_PATH": str(tmp_path / "auradream.db"), "AURADREAM_OUTPUTS_DIR": str(tmp_path / "outputs"),
        "AURADREAM_BUNDLES_DIR": str(tmp_path / "bundles"), "AURADREAM_KEEP_AWAKE": "false",
    }
    code = ("from backend import main; gate = main._passcode_gate; "
            "print('gate:', None if gate is None else gate.cookie_name)")
    done = subprocess.run([sys.executable, "-c", code], cwd=WEB.parent, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines()[-1] == f"gate: {expected}"
    assert PASSCODE not in done.stdout and PASSCODE not in done.stderr


# ------------------------------------------------------------------ main.run


class Quiet:
    """In uvicorn's Server's place: starts nothing, binds nothing."""

    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> None:
        return None


def _settings(monkeypatch, **changes) -> None:
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, **{"port": 0, **changes}))


def _messages(caplog) -> list[tuple[str, str]]:
    return [(record.levelname, record.getMessage()) for record in caplog.records if record.name == "auradream"]


@pytest.mark.parametrize("value", ["seven77", "x", "abc def"])
def test_a_passcode_under_eight_characters_stops_the_start_before_it_binds(fake, monkeypatch, capsys, value):
    def never(*args, **kwargs):
        pytest.fail("the server went on with a passcode too short to be one")

    _settings(monkeypatch, passcode=value, port=7860)
    monkeypatch.setattr(main, "_port_busy", never)
    monkeypatch.setattr(main, "Server", never)
    with pytest.raises(SystemExit) as stopped:
        main.run()
    assert stopped.value.code == main.SETTINGS_REFUSED_STATUS == 78
    said = capsys.readouterr()
    assert "AURADREAM_PASSCODE is shorter than 8 characters" in said.err
    assert value not in said.err and value not in said.out


def test_a_passcode_of_eight_characters_starts(fake, monkeypatch, capsys):
    _settings(monkeypatch, passcode="8 chars!")
    monkeypatch.setattr(main, "Server", Quiet)
    main.run()
    assert capsys.readouterr().err == ""


def test_the_start_up_log_says_the_passcode_is_asked_and_what_plain_http_means(fake, gate, monkeypatch, caplog):
    _settings(monkeypatch, passcode=PASSCODE, host="0.0.0.0",
              allowed_hosts=("127.0.0.1", "localhost", "dream.home.arpa"))
    monkeypatch.setattr(main, "Server", Quiet)
    with caplog.at_level("INFO", logger="auradream"):
        main.run()
    said = _messages(caplog)
    asked = [message for level, message in said if level == "INFO" and "Asking for the passcode" in message]
    assert len(asked) == 1 and "AURADREAM_PASSCODE" in asked[0]
    assert "unencrypted" in asked[0] and "capture the network's traffic" in asked[0]
    assert not [message for level, message in said if level == "WARNING"]
    assert PASSCODE not in caplog.text


def test_the_start_up_log_says_a_set_passcode_is_not_asked_on_this_machine_only(fake, monkeypatch, caplog):
    assert main._passcode_gate is None
    _settings(monkeypatch, passcode=PASSCODE)
    monkeypatch.setattr(main, "Server", Quiet)
    with caplog.at_level("INFO", logger="auradream"):
        main.run()
    said = _messages(caplog)
    assert ("INFO", "AURADREAM_PASSCODE is set but not asked: the server listens on this machine only.") in said
    assert not [message for level, message in said if "Asking" in message or level == "WARNING"]
    assert PASSCODE not in caplog.text


def test_without_a_passcode_a_bind_beyond_loopback_is_warned_about_by_its_name(fake, monkeypatch, caplog):
    _settings(monkeypatch, host="0.0.0.0")
    monkeypatch.setattr(main, "Server", Quiet)
    with caplog.at_level("INFO", logger="auradream"):
        main.run()
    warned = [message for level, message in _messages(caplog) if level == "WARNING"]
    assert len(warned) == 2
    assert warned[0].startswith("Serving on 0.0.0.0 with no passcode")
    assert "Set AURADREAM_PASSCODE" in warned[0]
    assert warned[1].startswith("Only this machine's names are allowed.")
    assert "AURADREAM_ALLOWED_HOSTS" in warned[1] and "restart" in warned[1]


def test_on_loopback_with_no_passcode_the_start_says_nothing_of_it(fake, monkeypatch, caplog):
    _settings(monkeypatch)
    monkeypatch.setattr(main, "Server", Quiet)
    with caplog.at_level("INFO", logger="auradream"):
        main.run()
    assert not [message for _, message in _messages(caplog) if "passcode" in message.lower()]


# ------------------------------------------------------------------ the page


def test_the_page_reads_the_gates_code_and_draws_nothing_until_it_is_asked():
    """The app's side of it (stage 3): `api()` acts on the 401's code, the
    gate's own, never its sentence; Sign in and Sign out are in the page
    hidden, so a server that asks for nothing draws neither; and the fake's
    knob builds the same gate with its own knobs left open."""
    state = (WEB / "state.js").read_text(encoding="utf-8")
    assert f"res.status === 401 && code === '{passcode.REQUIRED_CODE}'" in state
    assert passcode.REQUIRED_DETAIL not in state
    status = (WEB / "status.js").read_text(encoding="utf-8")
    assert f"api('{passcode.SIGN_OUT_PATH}', {{ method: 'POST' }})" in status
    assert "els.signOutRow.hidden = s.passcode !== true;" in status
    page = (WEB / "index.html").read_text(encoding="utf-8")
    sign_in = re.search(r'<button[^>]*id="btn-sign-in"[^>]*>', page)
    sign_out = re.search(r'<p[^>]*id="model-sign-out"[^>]*>', page)
    assert sign_in and sign_out
    assert " hidden>" in sign_in.group(0) and " hidden>" in sign_out.group(0)
    # Sign out's row is in the Models panel's footer, after its note.
    footer = page.split('class="model-picker-footer"', 1)[1].split('id="models-foot"', 1)[0]
    assert footer.index('id="model-picker-note"') < footer.index('id="model-sign-out"')
    fake = (WEB.parent / "tools" / "fake_server.py").read_text(encoding="utf-8")
    assert 'open_prefixes=("/api/fake/",)' in fake and "app_main._passcode_gate = None" in fake
