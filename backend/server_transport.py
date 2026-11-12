"""Strict browser origins and optional direct TLS; stdlib only, import cold."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import ssl
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


class _KeyAsksForPassword(Exception):
    """Raised by the password callback: the server is never given one."""


def _no_password() -> bytes:
    raise _KeyAsksForPassword


def setting_problem(certfile: Path | None, keyfile: Path | None) -> str | None:
    """Why the TLS pair cannot serve, in words, or None. The pair is loaded
    as uvicorn will load it, so a file that is not PEM, a key that does not
    match the certificate or a key protected by a password is refused here,
    before the log line that says the server is running, never as an
    `[SSL] PEM lib` traceback from inside uvicorn."""
    if (certfile is None) != (keyfile is None):
        return "Set both AURADREAM_SSL_CERTFILE and AURADREAM_SSL_KEYFILE, or unset both, and start again."
    for name, path in (("AURADREAM_SSL_CERTFILE", certfile), ("AURADREAM_SSL_KEYFILE", keyfile)):
        if path is not None and not path.is_file():
            return f"{name} must name an existing PEM file. Check it and start again."
    if certfile is None or keyfile is None:
        return None
    try:
        # The certificate alone first, so a refusal can name the file at fault.
        ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cafile=certfile)
    except OSError as error:
        return _unreadable("AURADREAM_SSL_CERTFILE", error) or (
            "AURADREAM_SSL_CERTFILE holds no PEM certificate (a block starting "
            "-----BEGIN CERTIFICATE-----). Point it at the certificate itself, "
            "converted to PEM if it is DER, and start again.")
    try:
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(certfile, keyfile, password=_no_password)
    except _KeyAsksForPassword:
        return ("AURADREAM_SSL_KEYFILE is protected by a password, which AuraDream never asks for. "
                "Write an unencrypted copy readable only by you (openssl pkey -in <key> -out <new key>) "
                "and start again.")
    except OSError as error:
        problem = _unreadable("AURADREAM_SSL_KEYFILE", error)
        if problem is not None:
            return problem
        if isinstance(error, ssl.SSLError) and error.reason == "KEY_VALUES_MISMATCH":
            return ("AURADREAM_SSL_KEYFILE is not the private key of the certificate in "
                    "AURADREAM_SSL_CERTFILE. Set the pair that were made together and start again.")
        return ("AURADREAM_SSL_KEYFILE holds no PEM private key (a block starting "
                "-----BEGIN PRIVATE KEY----- or similar). Point it at the certificate's key, "
                "converted to PEM if it is DER, and start again.")
    return None


def _unreadable(name: str, error: OSError) -> str | None:
    """A file the server may not read (an SSLError is not this: the bytes were read)."""
    if isinstance(error, ssl.SSLError):
        return None
    return f"{name} could not be read ({error.strerror or type(error).__name__}). Check it and start again."
