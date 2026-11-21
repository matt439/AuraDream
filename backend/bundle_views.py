"""Short-lived capabilities for reading a verified saved bundle.

Threads using the same archive share a single seekable reader. Closing a
view revokes its capability before waiting for that reader to finish.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

from . import bundle_browse_archive

IDLE_SECONDS = 600
MAX_VIEWS = 2


class ViewClosed(Exception):
    pass


class ViewsFull(Exception):
    pass


@dataclass
class _View:
    archive: bundle_browse_archive.BrowseArchive
    resources: ExitStack
    owner: threading.Event
    deadline: float
    timer: threading.Timer | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    pins: int = 0
    retired: bool = False


class BundleViews:
    def __init__(self, idle_seconds: float = IDLE_SECONDS):
        self.idle_seconds = idle_seconds
        self._condition = threading.Condition()
        self._views: dict[str, _View] = {}
        self._opening: dict[str, threading.Event] = {}
        self._slots = 0  # Includes opening and closing readers.
        self._stopping = threading.Event()

    def _release_slot(self) -> None:
        with self._condition:
            self._slots -= 1
            self._condition.notify_all()

    def reserve(self, token: str) -> threading.Event:
        """Register the pending capability before verification can be queued."""
        with self._condition:
            if self._stopping.is_set():
                raise ViewClosed
            if token in self._opening or token in self._views:
                raise ViewsFull("This bundle view is already open. Choose Browse again.")
            if self._slots >= MAX_VIEWS:
                raise ViewsFull("Close an open bundle view or wait for it to expire, then try again.")
            cancelled = threading.Event()
            self._opening[token] = cancelled
            self._slots += 1
            return cancelled

    def discard_opening(self, token: str, cancelled: threading.Event) -> None:
        """Retire this specific reservation after failure or cancellation."""
        with self._condition:
            cancelled.set()
            if self._opening.get(token) is cancelled:
                del self._opening[token]
                self._release_slot()

    def open(self, token: str, name: str, password: str | None, cancelled: threading.Event) -> None:
        resources = ExitStack()
        try:
            def stopped() -> bool:
                return self._stopping.is_set() or cancelled.is_set()

            archive = resources.enter_context(bundle_browse_archive.open_view(name, password, cancelled=stopped))
            with self._condition:
                if stopped():
                    raise ViewClosed
                view = _View(archive, resources, cancelled, time.monotonic() + self.idle_seconds)
                self._schedule(token, view, self.idle_seconds)
                self._views[token] = view
                del self._opening[token]
        except BaseException:
            try:
                resources.close()
            finally:
                self.discard_opening(token, cancelled)
            raise

    def _schedule(self, token: str, view: _View, delay: float) -> None:
        # Recheck the moving deadline when the timer wakes. A page of thumbnails
        # only updates that deadline; it doesn't spawn a timer per thumbnail.
        view.timer = threading.Timer(delay, self._expire, args=(token, view.owner))
        view.timer.daemon = True
        view.timer.start()

    def _expire(self, token: str, owner: threading.Event) -> None:
        with self._condition:
            view = self._views.get(token)
            if view is None or view.owner is not owner:
                return
            remaining = view.deadline - time.monotonic()
            if remaining > 0:
                self._schedule(token, view, remaining)
                return
            del self._views[token]
            view.retired = True
            if view.pins:
                return
        self._dispose(view)

    def pin(self, token: str) -> _View:
        """Give an admitted restore a read that survives closing its browse view."""
        with self._condition:
            view = self._views.get(token)
            if view is None or view.deadline <= time.monotonic():
                raise ViewClosed
            view.pins += 1
            return view

    def release_pin(self, view: _View) -> None:
        with self._condition:
            view.pins -= 1
            dispose = view.retired and not view.pins
        if dispose:
            self._dispose(view)

    @contextmanager
    def read_pinned(self, view: _View) -> Iterator[bundle_browse_archive.BrowseArchive]:
        with view.lock:
            yield view.archive

    @contextmanager
    def read(self, token: str) -> Iterator[bundle_browse_archive.BrowseArchive]:
        with self._condition:
            view = self._views.get(token)
        if view is None:
            raise ViewClosed
        with view.lock:
            with self._condition:
                now = time.monotonic()
                if self._views.get(token) is not view or view.deadline <= now:
                    raise ViewClosed
                view.deadline = now + self.idle_seconds
            yield view.archive

    def _dispose(self, view: _View) -> None:
        try:
            with view.lock:
                view.resources.close()
        finally:
            self._release_slot()

    def close(self, token: str, owner: threading.Event | None = None) -> None:
        with self._condition:
            pending = self._opening.get(token)
            if pending is not None and (owner is None or pending is owner):
                pending.set()
            view = self._views.get(token)
            if view is not None:
                if owner is not None and view.owner is not owner:
                    return
                del self._views[token]
                view.retired = True
                dispose = not view.pins
                if view.timer is not None:
                    view.timer.cancel()
            else:
                dispose = False
        if view is not None and dispose:
            self._dispose(view)

    def shutdown(self) -> None:
        with self._condition:
            self._stopping.set()
            tokens = list(self._views)
        for token in tokens:
            self.close(token)
        with self._condition:
            self._condition.wait_for(lambda: not self._slots)
