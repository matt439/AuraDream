"""Strict browser origins and optional direct TLS; stdlib only, import cold."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit


def origin_key(origin: str) -> tuple[str, str, int] | None:
    """An origin has no credentials, path, query, fragment or wildcard."""
    if any(char.isspace() for char in origin) or "\\" in origin:
        return None
    try:
        parts = urlsplit(origin)
        port = parts.port
        host = parts.hostname
    except ValueError:
        return None
    if (parts.scheme not in {"http", "https"} or not host or "*" in host
            or parts.username is not None or parts.password is not None
            or parts.path or parts.query or parts.fragment
            or "?" in origin or "#" in origin or parts.netloc.endswith(":")):
        return None
    port = port if port is not None else (443 if parts.scheme == "https" else 80)
    if not 1 <= port <= 65535:
        return None
    return parts.scheme, host, port


def origin_allowed(origin: str, *, hosts: Iterable[str], port: int,
                   extra_origins: Iterable[str] = ()) -> bool:
    key = origin_key(origin)
    if key is None or key[1] not in hosts:
        return False
    return key[2] == port or key in {origin_key(value) for value in extra_origins}


def setting_problem(certfile: Path | None, keyfile: Path | None) -> str | None:
    if (certfile is None) != (keyfile is None):
        return "Set both AURADREAM_SSL_CERTFILE and AURADREAM_SSL_KEYFILE, or unset both, and start again."
    for name, path in (("AURADREAM_SSL_CERTFILE", certfile), ("AURADREAM_SSL_KEYFILE", keyfile)):
        if path is not None and not path.is_file():
            return f"{name} must name an existing PEM file. Check it and start again."
    return None
