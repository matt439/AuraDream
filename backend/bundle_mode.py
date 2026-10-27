"""Explicit Windows/NTFS shared-download mode, without retention activation.

The marker is destination-wide and immutable. Other-server quiescence is an
operator attestation, never a claim inferred from this process's lease table.
Absent markers permit ordinary non-writing downloads on every platform.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from uuid import UUID, uuid4

from . import bundle_file_lock

# The marker name and format are frozen from the old name (Local Diffusion).
# The marker is destination-wide and immutable, and receipts bind its identity
# and digest: a second, renamed marker would be another installation that an
# older server, or an existing receipt, would not recognise.
MARKER_NAME = "local-diffusion.bundle-leases.json"
_FORMAT = "local-diffusion.shared-bundle-leases"
_LIMIT = 64 * 1024
_FIELDS = {"format", "version", "installation_id", "directory", "device", "inode",
           "other_servers_stopped", "machine_verified", "platform"}


class ModeError(ValueError):
    """An installed mode cannot be safely identified; never use a fallback."""


class UnsupportedMode(ModeError):
    pass


@dataclass(frozen=True)
class SharedMode:
    installation_id: str
    directory: Path
    directory_identity: tuple[int, int]
    marker_identity: tuple[int, int, int, int, int]
    digest: str


@dataclass(frozen=True)
class Installation:
    mode: SharedMode
    created: bool
    operator_confirmed_other_servers_stopped: bool = True
    machine_verified: bool = False
    retention_enabled: bool = False


_observed: dict[str, SharedMode] = {}
_named_observed: dict[str, SharedMode] = {}
_observed_lock = threading.Lock()
_failed_directory_close: dict[str, int] = {}


def _linked(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _directory_identity(directory: Path) -> tuple[int, int]:
    value = directory.lstat()
    if _linked(value) or not stat.S_ISDIR(value.st_mode) or not value.st_ino:
        raise ModeError("The shared-mode destination must be an unlinked directory with stable identity.")
    return value.st_dev, value.st_ino


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    if _linked(value) or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or not value.st_ino:
        raise ModeError("The shared-mode marker must be an unlinked regular file with stable identity.")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@lru_cache(maxsize=1)
def _volume_api():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    api.GetVolumePathNameW.restype = wintypes.BOOL
    api.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    api.GetDriveTypeW.restype = wintypes.UINT
    api.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                         ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
    api.GetVolumeInformationW.restype = wintypes.BOOL
    return api


def _qualified(directory: Path) -> None:
    if os.name != "nt":
        raise UnsupportedMode("Shared bundle mode is qualified only on Windows fixed NTFS volumes.")
    if str(directory).startswith(("\\\\", "//")):
        raise UnsupportedMode("Shared bundle mode is not qualified on network destinations.")
    api = _volume_api()
    root = ctypes.create_unicode_buffer(32768)
    if not api.GetVolumePathNameW(str(directory), root, len(root)):
        raise bundle_file_lock._windows_error(ctypes.get_last_error(), "GetVolumePathNameW shared-mode destination", directory)
    if api.GetDriveTypeW(root.value) != 3:  # DRIVE_FIXED; mapped network drives are DRIVE_REMOTE.
        raise UnsupportedMode("Shared bundle mode requires a fixed local NTFS volume; network/removable volumes are unqualified.")
    filesystem = ctypes.create_unicode_buffer(256)
    if not api.GetVolumeInformationW(root.value, None, 0, None, None, None, filesystem, len(filesystem)):
        raise bundle_file_lock._windows_error(ctypes.get_last_error(), "GetVolumeInformationW shared-mode destination", directory)
    if filesystem.value.upper() != "NTFS":
        raise UnsupportedMode("Shared bundle mode is qualified only on fixed local NTFS volumes.")


@contextmanager
def _pin_directory(directory: Path, identity: tuple[int, int]):
    """Deny directory rename/delete during the install, owning one native fd."""
    import msvcrt

    key = os.path.normcase(str(directory))
    if key in _failed_directory_close:
        raise ModeError("An earlier directory handle close failed; restart before another shared-mode operation.")
    api = bundle_file_lock._windows_api()
    # READ_ATTRIBUTES, READ|WRITE sharing (no DELETE), OPEN_EXISTING,
    # BACKUP_SEMANTICS|OPEN_REPARSE_POINT. Never create a destination here.
    handle = api.CreateFileW(str(directory), 0x80, 3, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise bundle_file_lock._windows_error(ctypes.get_last_error(), "CreateFileW shared-mode directory", directory)
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
    except BaseException as error:
        if not api.CloseHandle(handle):
            _failed_directory_close[key] = handle
            cleanup = bundle_file_lock._windows_error(ctypes.get_last_error(), "CloseHandle shared-mode directory", directory)
            error.add_note(str(cleanup) + " " + " ".join(cleanup.__notes__))
        raise
    body_error: BaseException | None = None
    try:
        opened = os.fstat(fd)
        if (_linked(opened) or not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != identity or _directory_identity(directory) != identity):
            raise ModeError("The shared-mode destination changed while it was opened.")
        yield
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            os.close(fd)
        except OSError as cleanup:
            _failed_directory_close[key] = fd
            if body_error is None:
                raise
            body_error.add_note(f"Closing the shared-mode directory also failed: {directory}: {cleanup}")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ModeError("The shared-mode marker has duplicate fields.")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise ModeError("The shared-mode marker contains invalid numbers.")


def _decode(data: bytes, directory: Path, directory_identity: tuple[int, int], marker_identity: tuple[int, int, int, int, int]) -> SharedMode:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        if (not isinstance(value, dict) or value.keys() != _FIELDS
                or value["format"] != _FORMAT or type(value["version"]) is not int or value["version"] != 1
                or value["platform"] != "windows-fixed-ntfs"
                or value["other_servers_stopped"] != "operator_confirmed" or value["machine_verified"] is not False):
            raise ValueError
        identifier = value["installation_id"]
        if not isinstance(identifier, str) or len(identifier) != 36 or str(UUID(identifier)) != identifier or not UUID(identifier).int:
            raise ValueError
        if (not isinstance(value["directory"], str)
                or os.path.normcase(value["directory"]) != os.path.normcase(str(directory))
                or type(value["device"]) is not int or type(value["inode"]) is not int
                or (value["device"], value["inode"]) != directory_identity):
            raise ValueError
        return SharedMode(identifier, directory, directory_identity, marker_identity, hashlib.sha256(data).hexdigest())
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ModeError("The shared-mode marker is malformed, unsupported or belongs to another destination; it was not changed.") from None


def discover(directory: Path, *, _expected: tuple[str, tuple[int, int], str] | None = None) -> SharedMode | None:
    """Read-only lookup; disappearance/replacement of an observed mode refuses."""
    named = Path(directory)
    named_absolute = Path(os.path.abspath(named))
    named_key = os.path.normcase(str(named_absolute))
    with _observed_lock:
        prior_name = _named_observed.get(named_key)
    if prior_name is not None:
        try:
            named_identity = _directory_identity(named_absolute)
        except FileNotFoundError:
            raise ModeError("The previously observed shared-mode destination is missing; no unlocked fallback is permitted.") from None
        if named_identity != prior_name.directory_identity:
            raise ModeError("The previously observed shared-mode destination changed identity; no unlocked fallback is permitted.")
    canonical = named.resolve()
    if prior_name is not None and canonical != prior_name.directory:
        raise ModeError("The previously observed shared-mode destination changed path; no unlocked fallback is permitted.")
    key = os.path.normcase(str(canonical))
    with _observed_lock:
        previous = _observed.get(key)
    path = canonical / MARKER_NAME
    try:
        before = path.lstat()
    except FileNotFoundError:
        if previous is not None:
            raise ModeError("The previously observed shared-mode marker is missing; no unlocked fallback is permitted.") from None
        return None
    # Marker absence above remains portable and never asks for native access.
    _qualified(canonical)
    _directory_identity(named)
    parent = _directory_identity(canonical)
    identity = _file_identity(before)
    if before.st_size > _LIMIT:
        raise ModeError("The shared-mode marker exceeds its bounded format.")
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            data = stream.read(_LIMIT + 1)
            after = os.fstat(stream.fileno())
        # Windows path/handle ctime can differ; compare within one source.
        if (_file_identity(opened)[:4] != identity[:4] or _file_identity(after) != _file_identity(opened)
                or _file_identity(path.lstat()) != identity or _directory_identity(canonical) != parent or len(data) > _LIMIT):
            raise ModeError("The shared-mode marker or destination changed during its read.")
    except FileNotFoundError:
        raise ModeError("The shared-mode marker disappeared during its read; no unlocked fallback is permitted.") from None
    mode = _decode(data, canonical, parent, identity)
    if _expected is not None and (mode.installation_id, mode.marker_identity[:2], mode.digest) != _expected:
        raise ModeError("The published shared-mode marker does not match the installer's owned file and contents; check the outcome before retrying.")
    with _observed_lock:
        known = _observed.get(key)
        named_known = _named_observed.get(named_key)
        if (known is not None and known != mode) or (named_known is not None and named_known != mode):
            raise ModeError("The previously observed shared-mode marker changed; no unlocked fallback is permitted.")
        _observed[key] = mode
        _named_observed[named_key] = mode
    return mode


def _remove_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return
    if _linked(value) or not stat.S_ISREG(value.st_mode) or (value.st_dev, value.st_ino) != identity:
        raise ModeError("The installer temporary changed ownership and was retained.")
    path.unlink()


def install(directory: Path, *, other_servers_stopped: bool) -> Installation:
    """Explicit cutover; never discover host processes or activate retention.

    The caller/operator must stop EVERY other server using this destination.
    Only this process's quiescence is mechanically checked by the lease guard.
    """
    if type(other_servers_stopped) is not bool or not other_servers_stopped:
        raise ModeError("Stop every other server using this destination and explicitly confirm it; this fact is operator-supplied, not machine-verified.")
    named = Path(directory)
    canonical = named.resolve()
    _qualified(canonical)
    _directory_identity(named)
    identity = _directory_identity(canonical)
    # Lazy import avoids a cycle: the download registry calls discover above.
    from .bundle_leases import installation_guard

    with installation_guard(canonical), _pin_directory(canonical, identity):
        existing = discover(canonical)
        if existing is not None:
            return Installation(existing, False)
        installation_id = str(uuid4())
        payload = {"format": _FORMAT, "version": 1, "installation_id": installation_id,
                   "directory": str(canonical), "device": identity[0], "inode": identity[1],
                   "other_servers_stopped": "operator_confirmed", "machine_verified": False,
                   "platform": "windows-fixed-ntfs"}
        data = (json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
        if len(data) > _LIMIT:
            raise ModeError("The shared-mode destination is too long for its bounded marker format.")
        temporary: Path | None = None
        owned: tuple[int, int] | None = None
        body_error: BaseException | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=canonical, prefix=".auradream.bundle-leases.", suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                stamp = os.fstat(stream.fileno())
                owned = stamp.st_dev, stamp.st_ino
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            current = temporary.lstat()
            if ((current.st_dev, current.st_ino) != owned or _linked(current)
                    or _directory_identity(canonical) != identity):
                raise ModeError("The shared-mode installer temporary or destination changed before publication.")
            # Windows rename refuses an existing final name. Never use replace;
            # installation is Windows-only and the pinned directory cannot move.
            os.rename(temporary, canonical / MARKER_NAME)
            # Publication transfers the name. This operation no longer owns
            # its former temporary path, even if another writer recreates it.
            temporary = None
            assert owned is not None
            mode = discover(canonical, _expected=(installation_id, owned, hashlib.sha256(data).hexdigest()))
            if mode is None:
                raise ModeError("The installed shared-mode marker disappeared; check the outcome before retrying.")
            return Installation(mode, True)
        except BaseException as error:
            body_error = error
            raise
        finally:
            if temporary is not None and owned is not None:
                try:
                    _remove_owned(temporary, owned)
                except BaseException as cleanup:
                    if body_error is None:
                        raise
                    body_error.add_note(f"Cleaning the shared-mode installer temporary also failed: {temporary}: {cleanup}")
