"""The opt-in passcode for a server reachable beyond this machine.

`AURADREAM_PASSCODE` (`config.settings.passcode`, the environment only) is
asked when it is set and the server can be reached by any name but this
machine's own (`asked`): a bind that is not loopback, or an allowed host
that is not. While it is asked a request from 127.0.0.1 is asked like any
other: behind a proxy on the same machine every request arrives from there.

All of it is here: the gate (`Gate`: the comparison, the derived session
key, the token), the limiter on wrong passcodes (`Limiter`) and the ASGI
layer (`PasscodeMiddleware`) with its two routes, `POST /api/passcode` and
`POST /api/passcode/sign-out`. The design and its reasons are
docs/design-notes.md, "A passcode beyond loopback".

The passcode itself is never kept: the gate holds its SHA-256 digest and a
scrypt-derived key, and nothing here logs, returns or raises a submitted
value. Over plain http the passcode and its cookie still cross the network
unencrypted; it keeps out people who can merely reach the address.

The limiter's state is read and written on the event loop only.

Standard library and Starlette only (with `keep_awake`'s and
`server_probe`'s, which are stdlib); no `config` import.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.requests import cookie_parser
from starlette.responses import JSONResponse, Response

from .keep_awake import scope_counts_as_activity
from .server_probe import is_loopback

log = logging.getLogger("auradream.passcode")

VARIABLE = "AURADREAM_PASSCODE"
# At the limiter's cap a guesser gets about 1,400 tries a day, which is why
# eight characters is the floor.
MIN_LENGTH = 8
# A submitted value longer than this is refused unread, so a setting longer
# than this could never be entered.
MAX_LENGTH = 256
MAX_BODY_BYTES = 4096

SIGN_IN_PATH = "/api/passcode"
SIGN_OUT_PATH = "/api/passcode/sign-out"

# What answers without a session while the gate is on, by exact path: a
# browser fetches a manifest without cookies, so a closed one would cost a
# phone its home-screen name and icon. Held equal to what the manifest and
# index.html's head name by tests/test_api_passcode.py.
OPEN_PATHS = frozenset({
    "/static/manifest.webmanifest",
    "/static/icons/icon.svg",
    "/static/icons/icon-192.png",
    "/static/icons/icon-512.png",
    "/static/icons/favicon-32.png",
    "/static/icons/apple-touch-icon.png",
})

REQUIRED_CODE = "passcode_required"
REQUIRED_DETAIL = "This server asks for a passcode. Reload the page and sign in."
WRONG_DETAIL = "That passcode is not right."

TOKEN_VERSION = "v1"
SESSION_SECONDS = 30 * 24 * 60 * 60
# A valid cookie older than this is replaced on a GET of the page or on any
# request someone using it makes (`keep_awake.scope_counts_as_activity`: never
# the page's timed polls, the job streams or its files), so a device used
# within the month stays signed in, a tab or PWA left open included, while
# an idle open tab does not keep its session alive by polling.
RENEW_AFTER_SECONDS = 24 * 60 * 60
# A fixed label, not a random salt: there is one secret, not a table of them.
SESSION_SALT = b"auradream.passcode.session.v1"
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def normalise(raw: str | None) -> str | None:
    """The setting as it is used: stripped, and empty is unset."""
    value = (raw or "").strip()
    return value or None


def setting_problem(passcode: str | None) -> str | None:
    """Why this setting cannot be served with, in plain words that never
    hold the value; None when it is unset or usable."""
    if passcode is None:
        return None
    if len(passcode) < MIN_LENGTH:
        return (f"{VARIABLE} is shorter than {MIN_LENGTH} characters. "
                f"Set a longer passcode, or unset it, and start again.")
    if len(passcode) > MAX_LENGTH:
        return (f"{VARIABLE} is longer than {MAX_LENGTH} characters, which the sign-in refuses. "
                f"Set a shorter passcode, or unset it, and start again.")
    return None


def asked(host: str, allowed_hosts: Iterable[str]) -> bool:
    """Whether a set passcode is asked: the server can be reached by a name
    that is not this machine's own. The bind, or (a proxy in front of a
    loopback bind) a name the Host check lets through."""
    return not is_loopback(host) or any(not is_loopback(name) for name in allowed_hosts)


# ------------------------------------------------------------------ the token


def _mac(key: bytes, expires: int) -> str:
    return hmac.new(key, f"{TOKEN_VERSION}.{expires}".encode("ascii"), hashlib.sha256).hexdigest()


def mint_token(key: bytes, expires: int) -> str:
    """`v1.<expires>.<mac>`: whole Unix seconds, and the hex HMAC-SHA256 of
    `v1.<expires>` under the session key."""
    return f"{TOKEN_VERSION}.{expires}.{_mac(key, expires)}"


def token_expiry(key: bytes, token: str | None, now: float) -> int | None:
    """When a valid token expires; None for anything else (not three parts,
    another version, a MAC that does not match, or expired)."""
    if not token or len(token) > 128:
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != TOKEN_VERSION:
        return None
    digits = parts[1]
    if not (digits.isascii() and digits.isdigit()) or len(digits) > 12 or str(int(digits)) != digits:
        return None
    expires = int(digits)
    if not hmac.compare_digest(parts[2].encode("utf-8"), _mac(key, expires).encode("ascii")):
        return None
    return expires if expires > now else None


# ---------------------------------------------------------------- the limiter


class Limiter:
    """Wrong passcodes, counted for the whole server: on a LAN an address is
    free to change. After `FREE` wrong in a row each further attempt first
    waits out a lock of 1, 2, 4 ... seconds, capped at `CAP`; a right one
    resets the count. The caller neither compares nor counts while locked."""

    FREE = 5
    CAP = 60.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.wrong_in_a_row = 0
        self._locked_until: float | None = None

    def locked_for(self) -> float:
        """Seconds of lock left; 0 when an attempt may be made."""
        if self._locked_until is None:
            return 0.0
        return max(0.0, self._locked_until - self._clock())

    def wrong(self) -> None:
        self.wrong_in_a_row += 1
        over = self.wrong_in_a_row - self.FREE
        if over >= 0:
            # 2**6 is already past the cap; the bound keeps the power finite.
            self._locked_until = self._clock() + min(self.CAP, 2.0 ** min(over, 6))

    def right(self) -> None:
        self.wrong_in_a_row = 0
        self._locked_until = None

    reset = right


# ------------------------------------------------------------------- the gate


@dataclass(frozen=True)
class Attempt:
    """One try at the passcode: right, wrong (counted), or not looked at
    because the limiter is locked for `retry_after` more seconds."""

    ok: bool
    retry_after: int | None = None


class Gate:
    """A passcode that is asked. Built once from the setting; holds its
    digest and the session key derived from it, never the passcode.

    `clock` (monotonic, the limiter's) and `now` (Unix time, the token's)
    are injectable for tests. `open_prefixes` are paths left open besides
    `OPEN_PATHS`: the fake server's own knobs, nothing on the real one.
    """

    def __init__(self, passcode: str, *, port: int,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], float] = time.time,
                 open_prefixes: tuple[str, ...] = ()) -> None:
        secret = passcode.strip().encode("utf-8")
        if not secret:
            raise ValueError("an empty passcode is no passcode")
        self._digest = hashlib.sha256(secret).digest()
        # scrypt, not a plain hash: a cookie read off the network costs one
        # scrypt per guess to turn back into the passcode.
        self._key = hashlib.scrypt(secret, salt=SESSION_SALT, **_SCRYPT)
        self.cookie_name = f"auradream_session_{port}"
        self.limiter = Limiter(clock)
        self.open_prefixes = open_prefixes
        self._now = now
        # Sign-outs from a signed-in device so far. The gate stores no
        # sessions, so a renewal minted after a sign-out would sign that
        # device back in: a request renews only if none came while it ran.
        # Read and written on the event loop only.
        self.sign_outs = 0

    def matches(self, submitted: str) -> bool:
        """The comparison alone, never counted. Digests, so that neither
        length nor a non-ASCII character reaches `compare_digest` raw."""
        submitted = submitted.strip()
        if len(submitted) > MAX_LENGTH:
            return False
        return hmac.compare_digest(hashlib.sha256(submitted.encode("utf-8")).digest(), self._digest)

    def attempt(self, submitted: str) -> Attempt:
        """Try a passcode through the limiter. During a lock a right one is
        refused as well: letting it through would tell a guesser which
        guess was right."""
        left = self.limiter.locked_for()
        if left > 0:
            return Attempt(False, max(1, math.ceil(left)))
        if self.matches(submitted):
            self.limiter.right()
            return Attempt(True)
        self.limiter.wrong()
        return Attempt(False)

    def mint(self) -> str:
        return mint_token(self._key, int(self._now()) + SESSION_SECONDS)

    def valid(self, token: str | None) -> bool:
        return token_expiry(self._key, token, self._now()) is not None

    def renewable(self, token: str | None) -> bool:
        """A valid token minted more than `RENEW_AFTER_SECONDS` ago."""
        expires = token_expiry(self._key, token, self._now())
        return expires is not None and expires - self._now() < SESSION_SECONDS - RENEW_AFTER_SECONDS

    def cookie(self, token: str, *, secure: bool) -> str:
        """The Set-Cookie value. Lax, not Strict, so a bookmark opens the
        page signed in; the port is in the name because cookies ignore
        ports. An empty token clears the cookie on that device."""
        age = SESSION_SECONDS if token else 0
        value = f"{self.cookie_name}={token}; HttpOnly; Max-Age={age}; Path=/; SameSite=Lax"
        return value + "; Secure" if secure else value

    def is_open(self, method: str, path: str) -> bool:
        if path.startswith(self.open_prefixes):     # an empty tuple matches nothing
            return True
        return method in ("GET", "HEAD") and path in OPEN_PATHS


def gate_for(passcode: str | None, host: str, allowed_hosts: Iterable[str], port: int) -> Gate | None:
    """The gate a server with these settings runs behind: None while the
    passcode is unset, or set and not asked (`asked`)."""
    if passcode is None or not asked(host, allowed_hosts):
        return None
    return Gate(passcode, port=port)


# ------------------------------------------------------------------ the layer


def _header(scope: Any, name: bytes) -> bytes | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value
    return None


def _text(raw: bytes) -> str:
    # A client that sends the passcode's UTF-8 bytes, or (urllib's own
    # encoding of a header) Latin-1.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _bearer(scope: Any) -> str | None:
    """The value of `Authorization: Bearer <value>`; None for no header or
    another scheme, which is not an attempt."""
    raw = _header(scope, b"authorization")
    if raw is None:
        return None
    scheme, _, value = raw.partition(b" ")
    if scheme.lower() != b"bearer" or not value.strip():
        return None
    return _text(value)


def _peer(scope: Any) -> str:
    client = scope.get("client")
    return str(client[0]) if client else "an unknown address"


async def _read_body(receive: Any, limit: int) -> bytes | None:
    """The request's body, or None once it is longer than `limit` or the
    client has gone."""
    body = b""
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            return None
        body += message.get("body", b"")
        if len(body) > limit:
            return None
        if not message.get("more_body"):
            return body


class PasscodeMiddleware:
    """ASGI: ask for the passcode while a gate is installed.

    A plain ASGI layer, not `@app.middleware("http")`, for the reason
    `keep_awake.ActivityMiddleware` gives: the job streams. Always
    installed; with no gate (`get_gate` answers None) everything passes
    through untouched, the two routes below included, so the fake and the
    tests turn it on without rebuilding the app.

    Its place in `main`: inside the response headers, the cross-site write
    block and the Host check (a hostile page cannot spend the sign-in
    attempts, and a 401 carries the headers), outside the model hold,
    keep-awake's activity and gallery admission (a request that is not
    signed in is none of those).

    A request passes with a valid session cookie or the passcode as
    `Authorization: Bearer`. A cookie a day old is replaced on the page and
    on a request that `scope_counts_as_activity`, never on a poll (the
    page's own or one it marks) or a file
    (`RENEW_AFTER_SECONDS`), nor on an answer a sign-out overtook. Otherwise GET or HEAD of exactly `/` is the
    sign-in page (`page`, a 200: a service worker rejects a redirect for a
    navigation, and the title probes raise on a 401), `OPEN_PATHS` pass,
    and everything else is a 401 whose body is never read.
    """

    def __init__(self, app: Any, get_gate: Callable[[], Gate | None], page: Path) -> None:
        self.app = app
        self.get_gate = get_gate
        self.page = page

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        kind = scope.get("type")
        gate = self.get_gate() if kind in ("http", "websocket") else None
        if gate is None:
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            # There are none today; one added later is not open by default.
            await send({"type": "websocket.close", "code": 1008})
            return

        method = str(scope.get("method", "")).upper()
        path = scope.get("path", "/")
        if gate.is_open(method, path):
            await self.app(scope, receive, send)
            return
        if method == "POST" and path == SIGN_IN_PATH:
            await self._sign_in(gate, scope, receive, send)
            return
        if method == "POST" and path == SIGN_OUT_PATH:
            await self._sign_out(gate, scope, receive, send)
            return

        cookies = cookie_parser(_text(_header(scope, b"cookie") or b""))
        token = cookies.get(gate.cookie_name)
        signed_in = gate.valid(token)
        renew = signed_in and gate.renewable(token)
        if not signed_in:
            # A cookie is not an attempt; only a passcode sent as a header is.
            submitted = _bearer(scope)
            if submitted is not None:
                attempt = gate.attempt(submitted)
                if attempt.retry_after is not None:
                    await self._locked(attempt.retry_after)(scope, receive, send)
                    return
                if not attempt.ok:
                    self._log_wrong(scope, "header")
                signed_in = attempt.ok
        is_page = method in ("GET", "HEAD") and path == "/"
        sign_outs = gate.sign_outs
        if not signed_in:
            response = self._page() if is_page else self._required()
            await response(scope, receive, send)
            return
        secure = scope.get("scheme") == "https"

        def renewal_stands() -> bool:
            # A sign-out answered while this request ran cleared a cookie
            # that this answer must not set again.
            return renew and gate.sign_outs == sign_outs

        if not is_page:
            # The poll mark is read too: a marked GET or grid preview is a
            # timer, not someone using the tab.
            if not (renew and scope_counts_as_activity(scope)):
                await self.app(scope, receive, send)
                return

            # Someone using an open tab: a fresh cookie once the one sent is
            # a day old, on whatever the answer is.
            async def send_renewed(message: Any) -> None:
                if message.get("type") == "http.response.start" and renewal_stands():
                    MutableHeaders(scope=message).append("Set-Cookie", gate.cookie(gate.mint(), secure=secure))
                await send(message)

            await self.app(scope, receive, send_renewed)
            return

        # The page itself, signed in: never the cached copy on a signed-out
        # reload, and a fresh cookie once the one sent is a day old.

        async def send_page(message: Any) -> None:
            if message.get("type") == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "no-cache"
                if renewal_stands():
                    headers.append("Set-Cookie", gate.cookie(gate.mint(), secure=secure))
            await send(message)

        await self.app(scope, receive, send_page)

    def _page(self) -> Response:
        return Response(
            self.page.read_bytes(), media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @staticmethod
    def _required() -> Response:
        # Bearer, never Basic, which opens the browser's own dialog.
        return JSONResponse(
            {"detail": REQUIRED_DETAIL, "code": REQUIRED_CODE}, status_code=401,
            headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
        )

    @staticmethod
    def _locked(retry_after: int) -> Response:
        unit = "second" if retry_after == 1 else "seconds"
        return JSONResponse(
            {"detail": f"Too many wrong passcodes. Wait {retry_after} {unit} and try again."},
            status_code=429, headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
        )

    @staticmethod
    def _log_wrong(scope: Any, where: str) -> None:
        # The peer's address, never the value.
        log.warning("wrong passcode from %s (%s)", _peer(scope), where)

    async def _sign_in(self, gate: Gate, scope: Any, receive: Any, send: Any) -> None:
        """`POST /api/passcode` with `{"passcode": "..."}`: 204 and the
        cookie, or 401. Never a query parameter: addresses are logged."""
        left = gate.limiter.locked_for()
        if left > 0:
            # Before the body is read: neither compared nor counted.
            await self._locked(max(1, math.ceil(left)))(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        too_long = JSONResponse({"detail": "That request is too long for a passcode."}, status_code=413)
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            await too_long(scope, receive, send)
            return
        body = await _read_body(receive, MAX_BODY_BYTES)
        if body is None:
            await too_long(scope, receive, send)
            return
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            parsed = None
        submitted = parsed.get("passcode") if isinstance(parsed, dict) else None
        if not isinstance(submitted, str):
            malformed = JSONResponse(
                {"detail": 'Send the passcode as JSON: {"passcode": "..."}.'}, status_code=422,
            )
            await malformed(scope, receive, send)
            return
        attempt = gate.attempt(submitted)
        if attempt.retry_after is not None:
            await self._locked(attempt.retry_after)(scope, receive, send)
            return
        if not attempt.ok:
            self._log_wrong(scope, "sign-in")
            wrong = JSONResponse({"detail": WRONG_DETAIL}, status_code=401, headers={"Cache-Control": "no-store"})
            await wrong(scope, receive, send)
            return
        signed = Response(status_code=204, headers={
            "Set-Cookie": gate.cookie(gate.mint(), secure=scope.get("scheme") == "https"),
            "Cache-Control": "no-store",
        })
        await signed(scope, receive, send)

    @staticmethod
    async def _sign_out(gate: Gate, scope: Any, receive: Any, send: Any) -> None:
        """`POST /api/passcode/sign-out`: clear the cookie on this device.
        Needs no session; no session is stored, so no other device is
        signed out. One sent with a valid cookie stops the renewal of every
        request still running (`Gate.sign_outs`), which would otherwise set
        the cleared cookie again; a signed-out caller cannot do that."""
        cookies = cookie_parser(_text(_header(scope, b"cookie") or b""))
        if gate.valid(cookies.get(gate.cookie_name)):
            gate.sign_outs += 1
        cleared = Response(status_code=204, headers={
            "Set-Cookie": gate.cookie("", secure=scope.get("scheme") == "https"),
            "Cache-Control": "no-store",
        })
        await cleared(scope, receive, send)
