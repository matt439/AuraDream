"""Whether the server's port is free, and whether AuraDream is the program on it.

`main.run` asks before uvicorn binds (a second launch finds the first);
`doctor` asks the same without starting anything. Standard library only, and
no `config` import, so both can use it before anything heavy is imported.
"""
from __future__ import annotations

import errno
import ipaddress
import json
import os
import socket
import urllib.request

# How long the busy-port check waits for the listener's page. A loopback
# answer is immediate; a program that accepts and never answers is not ours.
PROBE_SECONDS = 2.0

# A bind refused because the port is taken or reserved: POSIX errno values,
# and Winsock's WSAEADDRINUSE / WSAEACCES (Windows sockets report those).
_BUSY_ERRNOS = frozenset({errno.EADDRINUSE, errno.EACCES, 10048, 10013})
_BUSY_WINERRORS = frozenset({10048, 10013})


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


def running_commit(url: str) -> str | None:
    """The short commit the AuraDream at `url` (its page's address) started
    from: `commit` in its `/api/status`. None when nothing answers, the
    answer is not a status, or that server could not tell (or predates the
    field). As direct as `auradream_at`, with its timeout."""
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct.open(url.rstrip("/") + "/api/status", timeout=PROBE_SECONDS) as response:
            status = json.loads(response.read(1 << 20))
    except (OSError, ValueError):
        return None
    commit = status.get("commit") if isinstance(status, dict) else None
    return commit if isinstance(commit, str) and commit else None
