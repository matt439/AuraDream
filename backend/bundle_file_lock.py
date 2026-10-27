"""Native nonblocking archive-companion locks, without archive mutations.

Companions are persistent: never unlink, replace, truncate or initialize
their bytes. Locks cover cooperating processes only. Windows handles deny
delete sharing while held; other filesystem/platform behavior needs its
own acceptance. This module does not choose a backup/retention policy.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import errno
from functools import lru_cache
import os
from pathlib import Path
import stat


class LockContended(OSError):
    """The native shared/exclusive byte-range request conflicts."""


class LockSafetyError(ValueError):
    """A path or opened companion cannot establish a stable lock identity."""


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE),
    ]


def _linked(value: os.stat_result) -> bool:
    return (stat.S_ISLNK(value.st_mode)
            or bool(getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT))


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _regular(value: os.stat_result, path: Path) -> tuple[int, int]:
    if (_linked(value) or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1
            or not value.st_ino):
        raise LockSafetyError(f"The lease companion must be an unlinked regular file: {path}")
    return _identity(value)


def archive_path(path: Path) -> Path:
    """Canonical parent plus safe leaf; never resolve through an archive link."""
    path = Path(path)
    name = path.name
    if (not name or name in {".", ".."} or name.endswith((".", " "))
            or any(character in name for character in '/\\:\x00<>"|?*')
            or any(ord(character) < 32 for character in name)
            or name.split(".", 1)[0].upper() in {
                "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }):
        raise LockSafetyError(f"Unsafe archive lease basename: {path}")
    # A configured bundle directory is checked by storage before its callers
    # arrive here. Check the direct names too, before resolve hides a link.
    try:
        named_parent = path.parent.lstat()
    except FileNotFoundError:
        pass  # Normalize lexical aliases such as an unused/../ component.
    else:
        if _linked(named_parent):
            raise LockSafetyError(f"The archive lease directory must not be linked: {path.parent}")
    parent = path.parent.resolve()
    # Explicit stat preserves permission/I/O errors which Path.is_dir() may
    # otherwise turn into False. Only a missing archive itself is allowed.
    if not stat.S_ISDIR(parent.stat().st_mode):
        raise LockSafetyError(f"The archive lease directory is not a directory: {parent}")
    result = parent / name
    try:
        value = result.lstat()
    except FileNotFoundError:
        pass  # FileResponse retains its existing missing-archive behavior.
    else:
        if _linked(value) or not stat.S_ISREG(value.st_mode):
            raise LockSafetyError(f"The archive lease target must be a regular file: {result}")
    return result


def companion_path(archive: Path) -> Path:
    path = archive_path(archive)
    return path.with_name(path.name + ".lease.lock")


@lru_cache(maxsize=1)
def _windows_api():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                               wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    api.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                              wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    api.LockFileEx.restype = wintypes.BOOL
    api.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                ctypes.POINTER(_Overlapped)]
    api.UnlockFileEx.restype = wintypes.BOOL
    return api


def _windows_error(code: int, operation: str, path: Path) -> OSError:
    error = ctypes.WinError(code)
    error.filename = str(path)
    error.add_note(f"{operation}: {path}")
    return error


def _open(path: Path) -> int:
    if os.name != "nt":
        return os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    import msvcrt

    api = _windows_api()
    # READ|WRITE sharing, deliberately no DELETE sharing; no inherited or
    # asynchronous handle. OPEN_REPARSE_POINT refuses leaf traversal once
    # the opened handle's metadata is checked. OPEN_ALWAYS never truncates.
    handle = api.CreateFileW(str(path), 0xC0000000, 3, None, 4, 0x00200080, None)
    if handle == ctypes.c_void_p(-1).value:
        raise _windows_error(ctypes.get_last_error(), "CreateFileW lease companion", path)
    try:
        # Success transfers sole handle ownership to this fd. Only os.close
        # may then close it; never also call CloseHandle on the raw handle.
        return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
    except BaseException as exc:
        if not api.CloseHandle(handle):
            error = _windows_error(ctypes.get_last_error(), "CloseHandle untransferred lease companion", path)
            exc.add_note(str(error) + " " + " ".join(error.__notes__))
        raise


def _lock(fd: int, path: Path, exclusive: bool) -> None:
    if os.name == "nt":
        import msvcrt

        # _locking's LK_NBRLCK is exclusive too. LockFileEx is required for
        # genuine shared readers; one byte beyond an empty file is valid.
        flags = 1 | (2 if exclusive else 0)  # FAIL_IMMEDIATELY | EXCLUSIVE_LOCK
        if not _windows_api().LockFileEx(msvcrt.get_osfhandle(fd), flags, 0, 1, 0,
                                         ctypes.byref(_Overlapped())):
            code = ctypes.get_last_error()
            if code == 33:  # ERROR_LOCK_VIOLATION, never access/open failures.
                raise LockContended(errno.EWOULDBLOCK, "Archive lease is in use", str(path))
            raise _windows_error(code, "LockFileEx lease companion", path)
    else:
        import fcntl

        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise LockContended(errno.EWOULDBLOCK, "Archive lease is in use", str(path)) from exc
            raise


def _unlock(fd: int, path: Path) -> None:
    if os.name == "nt":
        import msvcrt

        if not _windows_api().UnlockFileEx(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(_Overlapped())):
            raise _windows_error(ctypes.get_last_error(), "UnlockFileEx lease companion", path)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """One owned fd/range; closing never removes its persistent companion."""

    def __init__(self, archive: Path, path: Path, fd: int, parent: tuple[int, int]) -> None:
        self.archive = archive
        self.path = path
        self._fd = fd
        self._parent = parent
        self._identity = _regular(os.fstat(fd), path)
        self._locked = False
        self.closed = False

    def validate(self) -> None:
        if self.closed:
            raise LockSafetyError(f"The archive lease handle is closed: {self.path}")
        parent = self.path.parent.lstat()
        if (_linked(parent) or not stat.S_ISDIR(parent.st_mode)
                or _identity(parent) != self._parent
                or _regular(os.fstat(self._fd), self.path) != self._identity
                or _regular(self.path.lstat(), self.path) != self._identity):
            raise LockSafetyError(f"The archive lease companion or directory changed: {self.path}")
        if archive_path(self.archive) != self.archive:
            raise LockSafetyError(f"The archive lease path changed: {self.archive}")

    def release(self) -> None:
        if self.closed:
            return
        failure: BaseException | None = None
        if self._locked:
            try:
                _unlock(self._fd, self.path)
                self._locked = False
            except BaseException as exc:
                failure = exc
        try:
            os.close(self._fd)
            self.closed = True
            self._locked = False  # Successful close releases even a failed unlock.
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"Closing lease companion also failed: {self.path}: {exc}")
        if failure is not None:
            raise failure


# If acquisition cleanup itself fails, retain its handle and refuse future
# acquisition on that path. Nothing retries a real native denial implicitly.
_failed_cleanup: dict[str, FileLock | int] = {}


def acquire(archive: Path, *, exclusive: bool) -> FileLock:
    archive = archive_path(archive)
    path = companion_path(archive)
    key = os.path.normcase(str(path))
    if key in _failed_cleanup:
        raise LockSafetyError(f"An earlier archive lease cleanup failed; restart before reusing: {path}")
    parent = _identity(path.parent.stat())
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    else:
        _regular(existing, path)
    fd = _open(path)
    owned: FileLock | None = None
    try:
        owned = FileLock(archive, path, fd, parent)
        owned.validate()
        _lock(fd, path, exclusive)
        owned._locked = True
        owned.validate()
        return owned
    except BaseException as exc:
        try:
            if owned is None:
                os.close(fd)
            else:
                owned.release()
        except BaseException as cleanup:
            _failed_cleanup[key] = owned if owned is not None else fd
            exc.add_note(f"Archive lease cleanup failed: {path}: {cleanup}")
        raise
