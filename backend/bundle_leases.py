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
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from starlette.responses import FileResponse, JSONResponse
from starlette.types import Receive, Scope, Send

from . import bundle_file_lock, bundle_mode
from .failure_state import retained_failure


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
_installing: set[str] = set()


def _directory_key(directory: Path) -> str:
    return os.path.normcase(str(Path(directory).resolve()))


def _check_installation(key: str) -> None:
    if _directory_key(Path(key).parent) in _installing:
        raise LeaseBusy("Shared archive mode is being installed; try the download again after cutover.")


@contextmanager
def installation_guard(directory: Path):
    """Local quiescence only; keep all registries out through marker publish."""
    key = _directory_key(directory)
    with _process_lock:
        if key in _installing or any(_directory_key(Path(path).parent) == key for path in _states):
            raise LeaseBusy("An archive response or claim is still active in this process; wait for it to finish before installing shared mode.")
        _installing.add(key)
    try:
        yield
    finally:
        with _process_lock:
            _installing.remove(key)


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

    def _native(self, key: str, state: _State, *, exclusive: bool, shared: bool | None = None) -> None:
        if state.failure is not None:
            raise RuntimeError(f"An archive lease release failed; restart before reusing {key}") from state.failure
        if not (self._cross_process if shared is None else shared):
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
            _check_installation(key)
            return self._reader_locked(key)

    def _reader_locked(self, key: str, *, shared: bool | None = None) -> Lease:
        state = _states.get(key, _State())
        if state.exclusive:
            raise LeaseBusy("This archive is temporarily reserved; try the download again.")
        self._native(key, state, exclusive=False, shared=shared)
        state.readers += 1
        _states[key] = state
        self._readers[key] = self._readers.get(key, 0) + 1
        return Lease(self, key, False)

    def exclusive(self, path: Path) -> Lease:
        key = self._key(path)
        with self._lock:
            _check_installation(key)
            return self._exclusive_locked(key)

    def _exclusive_locked(self, key: str, *, shared: bool | None = None) -> Lease:
        if key in _states:
            raise LeaseBusy("This archive is in use.")
        state = _State(exclusive=True)
        self._native(key, state, exclusive=True, shared=shared)
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
                            state.failure = retained_failure()
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


class ModeAwareRegistry(LeaseRegistry):
    """Discover the destination's immutable opt-in at each response entry."""

    def _acquire(self, path: Path, *, exclusive: bool) -> Lease:
        with self._lock:
            # The mode read and local reader registration share the cutover
            # mutex, including responses constructed before installation.
            _check_installation(str(path))
            mode = bundle_mode.discover(Path(path).parent)
            key = self._key(path)
            _check_installation(key)
            if mode is None and _directory_key(Path(key).parent) != _directory_key(Path(path).parent):
                mode = bundle_mode.discover(Path(key).parent)
            if mode is not None:
                key = os.path.normcase(str(bundle_file_lock.archive_path(path)))
            claim = self._exclusive_locked if exclusive else self._reader_locked
            return claim(key, shared=mode is not None or self._cross_process)

    def reader(self, path: Path) -> Lease:
        return self._acquire(path, exclusive=False)

    def exclusive(self, path: Path) -> Lease:
        return self._acquire(path, exclusive=True)


downloads = ModeAwareRegistry()


class ModeExclusive(AbstractContextManager["ModeExclusive"]):
    """Mandatory installed-mode native claim, never a process-only fallback.

    This capability alone does not authorize deletion. Mutation must also pin
    the destination/marker and verify one owned archive handle.
    """

    def __init__(self, path: Path, expected: bundle_mode.SharedMode):
        self.path, self.expected = Path(path), expected
        self._lease: Lease | None = None
        self._retained: BaseException | None = None

    def __enter__(self) -> ModeExclusive:
        if self._lease is not None:
            raise RuntimeError("This exclusive mode capability has already been entered.")
        with downloads._lock:
            mode = bundle_mode.discover(self.path.parent)
            if mode is None or not isinstance(self.expected, bundle_mode.SharedMode) or mode != self.expected:
                raise bundle_mode.ModeError("The expected installed shared mode is required for archive mutation.")
            canonical = bundle_file_lock.archive_path(self.path)
            if _directory_key(canonical.parent) != _directory_key(mode.directory):
                raise bundle_mode.ModeError("The archive is outside the expected shared-mode destination.")
            key = os.path.normcase(str(canonical))
            _check_installation(key)
            self._lease = downloads._exclusive_locked(key, shared=True)
        self._lease.__enter__()
        return self

    def require(self) -> None:
        lease = self._lease
        with downloads._lock:
            state = _states.get(lease._key) if lease is not None else None
            if (lease is None or lease._released or not lease._entered or state is None
                    or not state.exclusive or state.native is None or state.failure is not None):
                raise RuntimeError("The installed-mode native exclusive capability is no longer active.")
            if os.path.normcase(str(bundle_file_lock.archive_path(self.path))) != lease._key:
                raise bundle_mode.ModeError("The exclusive archive capability no longer names its acquired path.")
            state.native.validate()
            if bundle_mode.discover(self.path.parent) != self.expected:
                raise bundle_mode.ModeError("The installed shared mode changed during archive mutation.")

    def retain(self, error: BaseException) -> None:
        # Retain the original acquired key, never resolve a failed-close name.
        if self._lease is None or self._lease._released:
            raise RuntimeError("The exclusive mode capability is no longer active.")
        if self._retained is None:
            self._retained = retained_failure()
            _retained_exclusives.append(self)

    def __exit__(self, kind: object, error: BaseException | None, traceback: object) -> None:
        if self._retained is not None:
            if error is None:
                raise RuntimeError("Archive mutation ownership is quarantined until restart.") from self._retained
            error.add_note("The original native exclusive claim remains held until restart.")
            return
        assert self._lease is not None
        self._lease.__exit__(kind, error, traceback)


_retained_exclusives: list[ModeExclusive] = []


def exclusive_for_mode(path: Path, expected: bundle_mode.SharedMode) -> ModeExclusive:
    return ModeExclusive(path, expected)


class ArchiveRead(AbstractContextManager["ArchiveRead"]):
    """One explicit read owner, including quarantined borrowed-file failures."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lease: Lease | None = None
        self._retained: BaseException | None = None

    def __enter__(self) -> ArchiveRead:
        if self._lease is not None:
            raise RuntimeError("This archive read owner has already been entered.")
        self._lease = downloads.reader(self.path)
        self._lease.__enter__()
        return self

    def require(self, path: Path) -> None:
        lease = self._lease
        if (lease is None or lease._released
                or os.path.normcase(str(Path(path).resolve())) != lease._key):
            raise RuntimeError("The borrowed archive reader does not own this input.")

    def retain(self, error: BaseException) -> None:
        # Quarantine the already acquired key. A failed close is precisely
        # when pathname resolution may have changed or become unavailable.
        if self._lease is None or self._lease._released:
            raise RuntimeError("The archive read owner is no longer active.")
        if self._retained is None:
            self._retained = retained_failure()
            _retained_reads.append(self)

    def __exit__(self, kind: object, error: BaseException | None, traceback: object) -> None:
        if self._retained is not None:
            if error is None:
                raise RuntimeError("An archive file close failed; its reader claim is retained until restart.") from self._retained
            error.add_note("The archive reader claim remains held after an uncertain file close; restart before reusing this input.")
            return
        assert self._lease is not None
        self._lease.__exit__(kind, error, traceback)


_retained_reads: list[ArchiveRead] = []
_failed_inputs: list[object] = []


@contextmanager
def reading(path: Path):
    """Defer acquisition until entry, preserving gallery-before-archive order."""
    with ArchiveRead(path) as owner:
        yield owner


@contextmanager
def _open_input(path: Path, read_owner: ArchiveRead | None = None):
    """Private file primitive: borrowed reader or an explicitly owned input."""
    if read_owner is not None:
        read_owner.require(path)
    source = Path(path).open("rb")
    body_error: BaseException | None = None
    try:
        yield source
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            source.close()
        except BaseException as cleanup:
            if not source.closed:
                _failed_inputs.append(source)
                if read_owner is not None:
                    read_owner.retain(cleanup)
            if body_error is None:
                raise
            body_error.add_note(f"Closing archive input also failed: {path}: {cleanup} "
                                + " ".join(getattr(cleanup, "__notes__", ())))


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
