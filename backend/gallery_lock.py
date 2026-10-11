"""Reentrant gallery reservation shared by server threads and local CLIs."""
from __future__ import annotations

import functools
import inspect
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class GalleryBusy(RuntimeError):
    """Another process is currently using this gallery."""


# How long an ordinary SQLite session or a single-record write waits for
# another process's reservation before refusing. A CLI query or one save holds
# it briefly; SQLite's own busy timeout absorbed that overlap before the
# reservation existed. Whole maintenance operations (bundles, clear, reindex)
# pass no wait: a long operation is refused (after MAINTENANCE_GRACE) rather
# than queued.
SESSION_WAIT = 30.0
# And how long one of those maintenance operations still polls for the file
# lock: a reader holds it for a millisecond or two (a gallery page, the status
# poll's newest id, every 3 s per open tab), and a single attempt landing in
# that window refused a CLI `clear` or `reindex` with the server idle. Long
# enough to ride out a read, short enough that a real holder - another
# maintenance operation, an export - is still refused at once to a person.
MAINTENANCE_GRACE = 0.25
_POLL = 0.05
_BUSY = "another Local Diffusion process is using the gallery; try again when it finishes"

_mutex = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_local = threading.local()


@contextmanager
def hold(database: Path, *, create: bool = True, wait: float = 0.0):
    key = os.path.normcase(str(database.resolve()))
    with _mutex:
        lock = _locks.setdefault(key, threading.RLock())
    # One deadline for the in-process lock and the file lock together. A
    # thread polling for another process holds `lock` meanwhile; acquired
    # without a limit, N threads behind it waited N x `wait` in turn. A
    # maintenance call (wait=0) still queues behind this process's own
    # holders, which admission keeps short.
    deadline = time.monotonic() + wait
    if not lock.acquire(timeout=wait if wait > 0 else -1):
        raise GalleryBusy(_BUSY)
    try:
        held = getattr(_local, "held", None)
        if held is None:
            held = _local.held = set()
        if key in held:
            yield
            return
        path = Path(key + ".gallery.lock")
        if not create and not path.exists():
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Maintenance (no wait) queued behind this process's own holders
        # above; another process's brief read gets MAINTENANCE_GRACE.
        if wait <= 0:
            deadline = time.monotonic() + MAINTENANCE_GRACE
        stream = path.open("a+b")
        try:
            if stream.seek(0, 2) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            # Polled rather than blocking: Windows' LK_LOCK gives up after
            # about ten seconds of its own, and the deadline is ours to set.
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GalleryBusy(_BUSY) from exc
                    time.sleep(min(_POLL, remaining))
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
    finally:
        lock.release()


def protected(database, *, waits: bool = False):
    """Reserve a whole operation, including its file work outside SQLite.

    `waits` gives a per-record writer `SESSION_WAIT` (read per call) to get
    the reservation; maintenance operations leave it False and fail fast.
    """
    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            dry = signature.bind_partial(*args, **kwargs).arguments.get("dry_run", False)
            with hold(database(), create=not dry, wait=SESSION_WAIT if waits else 0.0):
                return fn(*args, **kwargs)
        return wrapped
    return decorate
