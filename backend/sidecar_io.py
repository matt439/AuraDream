"""Reading and publishing the small JSON sidecars beside the gallery
database on Windows (GAL#28).

Each sidecar reads with a plain open and no lock and publishes with
`os.replace`. On Windows a handle opened without FILE_SHARE_DELETE (every
Python `open`) makes a replace onto that file fail, and an open that lands
while the replace runs fails too. The replace raises `PermissionError` with
a sharing violation (WinError 32) or access denied (WinError 5, the file
pending deletion); `open` goes through the C runtime, which reports the
same thing as EACCES with no `winerror`. The overlap lasts milliseconds, so
both sides retry for a bounded time before the caller's own refusal. Any
other error, and every error off Windows, is raised at once.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from typing import BinaryIO

WINDOWS = os.name == "nt"
SHARING_WINERRORS = frozenset({5, 32})
# About half a second in all; a writer waits under its sidecar's reservation.
RETRY_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.15, 0.2)


def transient(exc: BaseException) -> bool:
    """True for a Windows sharing violation that a moment's wait clears."""
    if not WINDOWS or not isinstance(exc, PermissionError):
        return False
    winerror = getattr(exc, "winerror", None)
    if winerror is None:
        return exc.errno == errno.EACCES
    return winerror in SHARING_WINERRORS


def open_read(path: Path) -> BinaryIO:
    """`path.open("rb")`, retried while another handle's replace holds it."""
    for delay in (*RETRY_DELAYS, None):
        try:
            return path.open("rb")
        except PermissionError as exc:
            if delay is None or not transient(exc):
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")


def replace(source: Path, target: Path) -> None:
    """`os.replace(source, target)`, retried while a reader holds the target."""
    for delay in (*RETRY_DELAYS, None):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if delay is None or not transient(exc):
                raise
            time.sleep(delay)
