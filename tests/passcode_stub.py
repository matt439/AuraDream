"""A loopback stand-in for a server that asks for the passcode, for the
callers that are not a browser (the probe, the doctor, the CLI).

It answers as `backend.passcode.PasscodeMiddleware` does, over real HTTP:
signed out, `/` is the sign-in page and everything else the coded 401; a
request carrying the passcode as `Authorization: Bearer` (its UTF-8 bytes)
gets the route's JSON; while `locked`, a request that carries any passcode
is a 429 with `Retry-After`. The layer itself is held by
tests/test_api_passcode.py; `test_server_probe.py` holds the header these
callers send to the real gate's own reading of it.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SIGN_IN_PAGE = (Path(__file__).resolve().parents[1] / "web" / "passcode.html").read_bytes()
REQUIRED = {"detail": "This server asks for a passcode. Reload the page and sign in.", "code": "passcode_required"}
RETRY_AFTER = 7


class AskingServer:
    """`passcode` None is a server that asks for nothing. `routes` maps a
    path to its JSON (or to a callable given the server, for an answer that
    changes it). `requests` records (method, path, carried a passcode)."""

    def __init__(self, passcode: str | None, routes: dict[str, Any]) -> None:
        self.passcode = passcode
        self.routes = routes
        self.locked = False
        self.required = dict(REQUIRED)        # the 401's body
        self.requests: list[tuple[str, str, bool]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, code: int, body: bytes, kind: str = "application/json", **headers: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                for name, value in headers.items():
                    self.send_header(name.replace("_", "-"), value)
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                sent = self.headers.get("Authorization")
                owner.requests.append((method, self.path, sent is not None))
                if owner.passcode is not None:
                    # http.server reads a header as Latin-1: back to its bytes.
                    right = sent is not None and sent.encode("latin-1") == b"Bearer " + owner.passcode.encode("utf-8")
                    if sent is not None and owner.locked:
                        detail = {"detail": f"Too many wrong passcodes. Wait {RETRY_AFTER} seconds and try again."}
                        self._answer(429, json.dumps(detail).encode(), Retry_After=str(RETRY_AFTER))
                        return
                    if not right:
                        if self.path == "/":
                            self._answer(200, SIGN_IN_PAGE, "text/html; charset=utf-8")
                        else:
                            self._answer(401, json.dumps(owner.required).encode(), WWW_Authenticate="Bearer")
                        return
                if self.path == "/":
                    self._answer(200, b"<!doctype html><title>AuraDream</title>", "text/html")
                    return
                route = owner.routes.get(self.path.partition("?")[0])
                if route is None:
                    self._answer(404, json.dumps({"detail": "Not Found"}).encode())
                    return
                if isinstance(route, tuple):            # (status, headers): a redirect, say
                    self._answer(route[0], b"", **route[1])
                    return
                answer = route(owner) if callable(route) else route
                self._answer(200, json.dumps(answer).encode())

            def do_GET(self) -> None:  # noqa: N802 - http.server's name
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802 - http.server's name
                self._handle("POST")

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
