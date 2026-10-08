"""Reentrant gallery reservation shared by server threads and local CLIs."""
from __future__ import annotations

import functools
import inspect
import os
import threading
from contextlib import contextmanager
from pathlib import Path


class GalleryBusy(RuntimeError):
    """Another process is currently using this gallery."""


_mutex = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_local = threading.local()


@contextmanager
def hold(database: Path, *, create: bool = True):
    key = os.path.normcase(str(database.resolve()))
    with _mutex:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
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
        stream = path.open("a+b")
        try:
            if stream.seek(0, 2) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise GalleryBusy("another Local Diffusion process is using the gallery; try again when it finishes") from exc
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


def protected(database):
    """Reserve a whole operation, including its file work outside SQLite."""
    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            dry = signature.bind_partial(*args, **kwargs).arguments.get("dry_run", False)
            with hold(database(), create=not dry):
                return fn(*args, **kwargs)
        return wrapped
    return decorate
