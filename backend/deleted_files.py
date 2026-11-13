"""Removing a permanently deleted image's files (GAL#29).

A permanent delete commits the row first, then removes the PNG and its
thumbnails. On Windows a file another handle holds without
FILE_SHARE_DELETE - the `/outputs` mount still streaming the full PNG to a
lightbox or phone, a scanner on a just-saved file - cannot be unlinked, and
a single attempt left it on disk with no row. A whole-gallery bundle then
carried it (`bundles._inventory` takes every PNG under outputs) and an
import installed it again.

So each file is retried for about half a second (`sidecar_io`'s delays, on
a Windows sharing violation only), and one still held is remembered with the
image filename it belonged to. `sweep` removes the remembered files once
their holder has let go: after every permanent delete, before every bundle
inventory and when the process exits; one still held is left out of every
bundle inventory (`bundles._inventory`), so the two inventories of a
Bundle & clear agree and no archive carries it. A sweep never raises: the
delete it follows has already happened. A remembered file is never removed if
a row names its image again (a bundle import restored it); nothing else
under outputs is ever touched, since an untracked file there may be the
user's own. The memory is the process's: a file still held when the process
is killed stays, as before.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from pathlib import Path

from . import gallery_lock, sidecar_io, storage

log = logging.getLogger(__name__)

_pending: dict[Path, str] = {}  # a held file -> the image filename it belongs to
_lock = threading.Lock()
_at_exit = False
# At exit, how long the sweep waits for the gallery before leaving the files.
EXIT_WAIT = 0.5


def _unlink(path: Path, *, retry: bool) -> bool:
    """True when the file is gone (or never was), False while it is held."""
    for delay in (*sidecar_io.RETRY_DELAYS, None) if retry else (None,):
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            if delay is None or not sidecar_io.transient(exc):
                return False
            time.sleep(delay)
    return False


def _remember(path: Path, filename: str) -> None:
    global _at_exit
    with _lock:
        _pending[path] = filename
        if not _at_exit:
            atexit.register(_sweep_at_exit)
            _at_exit = True
    log.warning("delete: %s could not be removed (held open?); retried once released", path.name)


def remove(filename: str) -> bool:
    """Remove the files of an image whose row is already deleted: its PNG,
    its thumbnail and any legacy sibling. True when the PNG was there and
    is gone; a file still held is remembered for `sweep`."""
    png = storage.settings.outputs_dir / filename
    existed = png.exists()
    removed = True
    for path in (png, storage._thumb_path(png), png.with_suffix(".webp")):
        if not _unlink(path, retry=True):
            _remember(path, filename)
            removed = removed and path != png
    return existed and removed


def pending() -> list[Path]:
    with _lock:
        return sorted(_pending)


def sweep() -> int:
    """Remove the remembered files whose holder has let go; how many went.
    Never raises: it follows a delete that has already happened."""
    try:
        return _sweep()
    except Exception as exc:  # noqa: BLE001 - logged; the files stay remembered
        log.error("delete: held files not swept: %s", exc)
        log.debug("delete: sweep failed", exc_info=True)
        return 0


def _sweep_at_exit() -> None:
    """The exit sweep waits briefly, never behind a bundle still running."""
    try:
        with gallery_lock.hold(storage.settings.db_path, create=False, wait=EXIT_WAIT):
            sweep()
    except Exception:  # noqa: BLE001 - never a traceback at exit
        log.warning("delete: gallery busy at exit; %d held file(s) left", len(pending()))


def _sweep() -> int:
    with _lock:
        held = dict(_pending)
    if not held or not storage.settings.db_path.exists():
        return 0
    owners = sorted(set(held.values()))
    named: set[str] = set()
    with storage._session() as conn:
        for at in range(0, len(owners), 500):
            chunk = owners[at:at + 500]
            marks = ",".join("?" * len(chunk))
            named.update(row[0] for row in conn.execute(
                f"SELECT filename FROM images WHERE filename IN ({marks})", chunk,
            ))
    gone = 0
    for path, owner in held.items():
        if owner in named:
            # A row names this image again: the file is that row's now.
            done = True
        else:
            done = _unlink(path, retry=False)
            gone += done
        if done:
            with _lock:
                if _pending.get(path) == owner:
                    del _pending[path]
    return gone
