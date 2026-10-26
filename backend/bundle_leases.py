"""Archive reader leases and exclusive claims, without deletion.

Default downloads retain per-process behavior without sidecar writes. An
explicit cross_process registry also locks persistent archive companions.
Install that mode only while quiescent, before admitting any streams; mixed
old/default/noncooperating processes cannot establish safe retention.
A snapshot is informational; an exclusive token must cover any later mutation.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from starlette.responses import FileResponse, JSONResponse
from starlette.types import Receive, Scope, Send

from . import bundle_file_lock


class LeaseBusy(RuntimeError):
    pass


@dataclass
class _State:
    readers: int = 0
    exclusive: bool = False
    native: bundle_file_lock.FileLock | None = None
    failure: BaseException | None = None


_process_lock = threading.Lock()
_states: dict[str, _State] = {}


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

    def __exit__(self, _kind: object, error: BaseException | None, _traceback: object) -> None:
        try:
            self.release()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note(f"Archive lease release failed: {self._key}: {cleanup} "
                           + " ".join(getattr(cleanup, "__notes__", ())))


class LeaseRegistry:
    def __init__(self, *, cross_process: bool = False) -> None:
        self._lock = _process_lock
        self._cross_process = cross_process
        self._readers: dict[str, int] = {}
        self._claims: set[str] = set()

    def _key(self, path: Path) -> str:
        canonical = bundle_file_lock.archive_path(path) if self._cross_process else Path(path).resolve()
        return os.path.normcase(str(canonical))

    def _native(self, key: str, state: _State, *, exclusive: bool) -> None:
        if state.failure is not None:
            raise RuntimeError(f"An archive lease release failed; restart before reusing {key}") from state.failure
        if not self._cross_process:
            return
        try:
            if state.native is None:
                state.native = bundle_file_lock.acquire(Path(key), exclusive=exclusive)
            else:
                state.native.validate()
        except bundle_file_lock.LockContended as exc:
            raise LeaseBusy("This archive is in use by another process.") from exc

    def reader(self, path: Path) -> Lease:
        key = self._key(path)
        with self._lock:
            state = _states.get(key, _State())
            if state.exclusive:
                raise LeaseBusy("This archive is temporarily reserved; try the download again.")
            self._native(key, state, exclusive=False)
            state.readers += 1
            _states[key] = state
            self._readers[key] = self._readers.get(key, 0) + 1
        return Lease(self, key, False)

    def exclusive(self, path: Path) -> Lease:
        key = self._key(path)
        with self._lock:
            if key in _states:
                raise LeaseBusy("This archive is in use.")
            state = _State(exclusive=True)
            self._native(key, state, exclusive=True)
            _states[key] = state
            self._claims.add(key)
        return Lease(self, key, True)

    def _release(self, lease: Lease) -> None:
        with self._lock:
            if lease._released:
                return
            state = _states[lease._key]
            if state.failure is not None:
                raise RuntimeError("The earlier archive lease release failed; no implicit retry") from state.failure
            failure: BaseException | None = None
            if state.exclusive or state.readers == 1:
                if state.native is not None:
                    try:
                        state.native.release()
                    except BaseException as exc:
                        if not state.native.closed:
                            state.failure = exc
                            raise
                        failure = exc
                del _states[lease._key]
            else:
                state.readers -= 1
            if lease._exclusive:
                self._claims.remove(lease._key)
            else:
                remaining = self._readers[lease._key] - 1
                if remaining:
                    self._readers[lease._key] = remaining
                else:
                    del self._readers[lease._key]
            lease._released = True
            if failure is not None:
                raise failure

    def leased(self, path: Path) -> bool:
        """Informational readers across all registries in this process."""
        key = self._key(path)
        with self._lock:
            state = _states.get(key)
            return state is not None and bool(state.readers)

    def snapshot(self) -> frozenset[str]:
        """A process-wide reader snapshot, never deletion authorization."""
        with self._lock:
            return frozenset(key for key, state in _states.items() if state.readers)


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
