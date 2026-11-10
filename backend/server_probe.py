"""Whether the server's port is free, and whether AuraDream is the program on it.

`main.run` asks before uvicorn binds (a second launch finds the first);
`doctor` asks the same without starting anything. Standard library only, and
no `config` import, so both can use it before anything heavy is imported.

It also holds what every caller that is not a browser shares about the
passcode (`passcode.py`; docs/design-notes.md, "A passcode beyond
loopback"): the header (`passcode_headers`), the gate's refusal read from an
answer (`refusal`, `PasscodeRefused`) and an opener that follows no redirect
(`no_redirect_opener`). The title probe needs none of it: a signed-out `/`
is the sign-in page, under the same title. Nothing here prints, logs or
raises the passcode itself.
"""
from __future__ import annotations

import errno
import ipaddress
import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

# How long the busy-port check waits for the listener's page. A loopback
# answer is immediate; a program that accepts and never answers is not ours.
PROBE_SECONDS = 2.0

# A bind refused because the port is taken or reserved: POSIX errno values,
# and Winsock's WSAEADDRINUSE / WSAEACCES (Windows sockets report those).
_BUSY_ERRNOS = frozenset({errno.EADDRINUSE, errno.EACCES, 10048, 10013})
_BUSY_WINERRORS = frozenset({10048, 10013})

# The server's setting, which a caller reads from its own environment and
# hands a child through the environment, never as an argument.
PASSCODE_VARIABLE = "AURADREAM_PASSCODE"
# `passcode.REQUIRED_CODE`, the 401's own mark. This module imports nothing
# of the project's; tests/test_server_probe.py holds the two equal.
PASSCODE_REQUIRED_CODE = "passcode_required"


def is_loopback(host: str) -> bool:
    """Whether binding `host` keeps the server to this machine."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def page_url(host: str, port: int) -> str:
    """The page's address for a browser on this machine: a wildcard bind is
    reached on loopback, which the default allowed hosts accept."""
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}/"


def port_busy(host: str, port: int) -> bool:
    """Whether uvicorn's bind of `host:port` would fail: a trial bind made as
    asyncio makes uvicorn's (address reuse on POSIX only, so a live listener
    refuses it and a closed connection's TIME_WAIT does not). A port Windows
    reserves refuses it too. Released at once; a program that takes the port
    in between still meets uvicorn's own error. Only "in use" and "refused"
    count: any other failure (a host that does not resolve, an address not on
    this machine) is left to uvicorn, which names it."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as trial:
        if os.name == "posix":
            trial.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            trial.bind((host, port))
        except socket.gaierror:
            return False
        except OSError as exc:
            return exc.errno in _BUSY_ERRNOS or getattr(exc, "winerror", None) in _BUSY_WINERRORS
    return False


def auradream_at(url: str) -> bool:
    """Whether the page at `url` is AuraDream's: the index's own title, read
    with a short timeout. Anything else, or no answer, is another program.
    Never through a proxy: an environment or Windows system proxy would
    answer for loopback in AuraDream's place."""
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct.open(url, timeout=PROBE_SECONDS) as response:
            return b"<title>AuraDream</title>" in response.read(65536)
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------- the passcode


def environment_passcode(environ: Mapping[str, str] | None = None) -> str | None:
    """`AURADREAM_PASSCODE` as the server reads it: stripped, empty is unset."""
    value = (os.environ if environ is None else environ).get(PASSCODE_VARIABLE, "").strip()
    return value or None


def passcode_headers(passcode: str | None) -> dict[str, str]:
    """The header a caller that is not a browser sends: `Authorization:
    Bearer <passcode>`, and nothing without one. The value is the passcode's
    UTF-8 bytes spelled as Latin-1, which `urllib` puts on the wire
    unchanged: the server reads a header as UTF-8 first. A control character
    cannot travel in a header and is refused here, in words that do not hold
    the value (`http.client`'s own error would quote it)."""
    if not passcode:
        return {}
    if any((ord(ch) < 32 and ch != "\t") or ord(ch) == 127 for ch in passcode):
        raise ValueError(f"{PASSCODE_VARIABLE} holds a control character, which a request header cannot carry")
    return {"Authorization": "Bearer " + passcode.encode("utf-8").decode("latin-1")}


class PasscodeRefused(Exception):
    """The server's passcode gate refused a request: a 401 (`sent` says
    whether a passcode went with it) or, during the lock after too many
    wrong ones, a 429. Its text never holds a passcode."""

    def __init__(self, status: int, sent: bool, retry_after: int | None = None) -> None:
        self.status = status
        self.sent = sent
        self.retry_after = retry_after
        super().__init__(f"the server {self.reason}")

    @property
    def locked(self) -> bool:
        return self.status == 429

    @property
    def reason(self) -> str:
        """What the server did, to follow its name: "the server <reason>"."""
        if self.locked:
            wait = f"{self.retry_after} s" if self.retry_after else "a minute"
            return f"is refusing every passcode for now, after too many wrong ones (wait {wait})"
        if self.sent:
            return f"did not accept the passcode in {PASSCODE_VARIABLE}"
        return f"asks for a passcode, and {PASSCODE_VARIABLE} is not set here"

    @property
    def fix(self) -> str:
        """What to do about it. Never "run this again": a caller that has
        already queued work must not be told to submit it twice."""
        if self.locked:
            return ("Each further wrong passcode makes that wait longer, for every device that is "
                    "not signed in.")
        return f"Set {PASSCODE_VARIABLE} in this shell to the server's passcode."


def refusal(error: urllib.error.HTTPError, *, sent: bool) -> PasscodeRefused | None:
    """The passcode gate's refusal in an HTTP error, or None for any other
    error: a 401 carrying the gate's own code, or a 429 (only the gate
    answers one). Reads the error's body."""
    if error.code == 429:
        after = str(error.headers.get("Retry-After") or "") if error.headers is not None else ""
        return PasscodeRefused(429, sent, int(after) if after.isascii() and after.isdigit() else None)
    if error.code != 401:
        return None
    try:
        answer = json.loads(error.read(65536))
    except (OSError, ValueError):
        return None
    if isinstance(answer, dict) and answer.get("code") == PASSCODE_REQUIRED_CODE:
        return PasscodeRefused(401, sent)
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any) -> None:
        return None


def no_redirect_opener(*, direct: bool = False) -> urllib.request.OpenerDirector:
    """An opener for a request that carries the passcode: `urllib` would
    send the header on to wherever a redirect points, so none is followed
    (the redirect arrives as the HTTP error it is). `direct` skips proxies."""
    handlers: list[urllib.request.BaseHandler] = [_NoRedirect()]
    if direct:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def status_json(url: str, passcode: str | None = None, timeout: float = PROBE_SECONDS) -> Any:
    """The JSON at `url` on this machine's server, asked directly (never
    through a proxy). Asked first without the passcode: only a server that
    answers with the gate's own 401 is then sent it, so a server that asks
    for none never sees it, and another program holding the port is not
    handed it. Raises `PasscodeRefused` for the gate's no, else `OSError`
    or `ValueError` as `urllib` and `json` do."""
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct.open(url, timeout=timeout) as response:
            return json.loads(response.read(1 << 20))
    except urllib.error.HTTPError as exc:
        refused = refusal(exc, sent=False)
        if refused is None:
            raise
        if refused.locked or not passcode:
            raise refused from None
    request = urllib.request.Request(url, headers=passcode_headers(passcode))
    try:
        with no_redirect_opener(direct=True).open(request, timeout=timeout) as response:
            return json.loads(response.read(1 << 20))
    except urllib.error.HTTPError as exc:
        refused = refusal(exc, sent=True)
        if refused is None:
            raise
        raise refused from None


def running_commit(url: str, passcode: str | None = None) -> str | None:
    """The short commit the AuraDream at `url` (its page's address) started
    from: `commit` in its `/api/status`. None when nothing answers, the
    answer is not a status, or that server could not tell (or predates the
    field). As direct as `auradream_at`, with its timeout. `passcode` goes
    only to a server that asks for one (`status_json`); a gate that refuses
    is `PasscodeRefused`, never None: that server is running."""
    try:
        status = status_json(url.rstrip("/") + "/api/status", passcode)
    except (OSError, ValueError):
        return None
    commit = status.get("commit") if isinstance(status, dict) else None
    return commit if isinstance(commit, str) and commit else None
