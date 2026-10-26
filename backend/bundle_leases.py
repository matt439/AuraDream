"""Per-process archive reader leases and exclusive claims, without deletion.

These locks cover cooperating threads in this process only. Shared leases are
still required before enabling retention across Local Diffusion processes.
A snapshot is informational; an exclusive token must cover any later mutation.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from starlette.responses import FileResponse, JSONResponse
from starlette.types import Receive, Scope, Send


class LeaseBusy(RuntimeError):
    pass


@dataclass
class Lease:
    _registry: LeaseRegistry
    _key: str
    _exclusive: bool
    _released: bool = field(default=False, init=False)
    _entered: bool = field(default=False, init=False)

    def release(self) -> None:
        self._registry._release(self)

    def __enter__(self) -> Lease:
        with self._registry._lock:
            if self._released or self._entered:
                raise RuntimeError("This lease token has already been entered or released.")
            self._entered = True
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class LeaseRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readers: dict[str, int] = {}
        self._claims: set[str] = set()

    @staticmethod
    def _key(path: Path) -> str:
        return os.path.normcase(str(Path(path).resolve()))

    def reader(self, path: Path) -> Lease:
        key = self._key(path)
        with self._lock:
            if key in self._claims:
                raise LeaseBusy("This archive is temporarily reserved; try the download again.")
            self._readers[key] = self._readers.get(key, 0) + 1
        return Lease(self, key, False)

    def exclusive(self, path: Path) -> Lease:
        key = self._key(path)
        with self._lock:
            if key in self._claims or self._readers.get(key, 0):
                raise LeaseBusy("This archive is in use.")
            self._claims.add(key)
        return Lease(self, key, True)

    def _release(self, lease: Lease) -> None:
        with self._lock:
            if lease._released:
                return
            if lease._exclusive:
                self._claims.remove(lease._key)
            else:
                remaining = self._readers[lease._key] - 1
                if remaining:
                    self._readers[lease._key] = remaining
                else:
                    del self._readers[lease._key]
            lease._released = True

    def leased(self, path: Path) -> bool:
        key = self._key(path)
        with self._lock:
            return bool(self._readers.get(key, 0))

    def snapshot(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._readers)


downloads = LeaseRegistry()


class LeasedFileResponse(FileResponse):
    """Preserve FileResponse semantics, holding a lease through its ASGI call.

    Acquire before stat/open or any await. Finally releases on normal bodies,
    ranges, errors, cancellation and background completion. Path handoff is
    disabled: its send completion does not establish that transfer has ended.
    """

    registry = downloads

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            lease = self.registry.reader(Path(self.path))
        except LeaseBusy as exc:
            await JSONResponse({"detail": str(exc)}, status_code=409)(scope, receive, send)
            return
        with lease:
            extensions = {key: value for key, value in scope.get("extensions", {}).items()
                          if key != "http.response.pathsend"}
            streamed: Scope = dict(scope)
            streamed["extensions"] = extensions
            await super().__call__(streamed, receive, send)
