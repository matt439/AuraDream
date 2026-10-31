"""Explicit Windows Credential Manager adapter; imports never open the vault.

Generic credentials are bound to one gallery and destination generation. There
is no enumeration, deletion, prompting, fallback provider or automatic setup.
CredWrite is not an atomic create-only API: callers reserve their gallery and
use a new random destination generation; a noncooperating writer can still race
the preflight. Uncertain/orphaned entries are never deleted or retried here.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import os
import unicodedata
from uuid import UUID

WINDOWS_PROVIDER = "windows-credential-manager-v1"
# The target prefix, blob magic and username are frozen from the old name
# (Local Diffusion): they name and bind secrets already in the vault, and
# renaming them would orphan every stored backup credential.
_PREFIX = "LocalDiffusion/AutomaticBackup/v1/"
_MAGIC = b"LocalDiffusion.AutomaticBackup.Credential.v1\x00"
_MAX_BLOB_BYTES = 2560
_MAX_SECRET_CHARS = 1024
_USERNAME = "LocalDiffusion"
_ERROR = "The Windows backup credential could not be safely used. Review the inactive profile and credential setup before retrying."
_MISSING = "The named Windows backup credential is missing. Set up a new generation explicitly; no plaintext fallback is used."
_EXISTS = "That Windows backup credential target already exists. It was not overwritten; inspect the inactive profile before another setup."
_SESSION = "The Windows credential session is unavailable. Unlock the intended Windows user session before another explicit attempt."
_ACCESS = "Windows refused access to the named backup credential. Review credential access without a plaintext fallback."
_SECRET_ERROR = "Choose a nonempty NFC password of at most 1024 characters that fits the Windows credential byte limit."
_REFERENCE_ERROR = "The backup credential reference is invalid or belongs to another scope."


class CredentialError(ValueError):
    """Fixed public errors, without provider messages or secret-bearing chains."""
    reason = "credential_unusable"


class CredentialMissing(CredentialError):
    reason = "credential_missing"


class CredentialExists(CredentialError):
    reason = "credential_exists"


class CredentialSessionUnavailable(CredentialError):
    reason = "credential_session_unavailable"


class CredentialAccessDenied(CredentialError):
    reason = "credential_access_denied"


def _uuid(value: object) -> str:
    if not isinstance(value, str) or len(value) != 36:
        raise CredentialError(_REFERENCE_ERROR)
    try:
        parsed = UUID(value)
    except ValueError:
        parsed = None
    if parsed is None or not parsed.int or str(parsed) != value:
        raise CredentialError(_REFERENCE_ERROR)
    return value


@dataclass(frozen=True)
class CredentialReference:
    gallery_id: str
    destination_id: str
    target: str

    def __post_init__(self) -> None:
        gallery, destination = _uuid(self.gallery_id), _uuid(self.destination_id)
        if self.target != _PREFIX + gallery + "/" + destination:
            raise CredentialError(_REFERENCE_ERROR)

    @property
    def provider(self) -> str:
        return WINDOWS_PROVIDER


def new_reference(gallery_id: str, destination_id: str) -> CredentialReference:
    gallery, destination = _uuid(gallery_id), _uuid(destination_id)
    return CredentialReference(gallery, destination, _PREFIX + gallery + "/" + destination)


def reference_wire(reference: CredentialReference) -> dict[str, str]:
    if not isinstance(reference, CredentialReference):
        raise CredentialError(_REFERENCE_ERROR)
    checked = new_reference(reference.gallery_id, reference.destination_id)
    if checked != reference:
        raise CredentialError(_REFERENCE_ERROR)
    return {"provider": WINDOWS_PROVIDER, "gallery_id": checked.gallery_id,
            "destination_id": checked.destination_id, "target": checked.target}


def read_reference(value: object) -> CredentialReference:
    if (not isinstance(value, dict) or value.keys() != {"provider", "gallery_id", "destination_id", "target"}
            or value["provider"] != WINDOWS_PROVIDER or not isinstance(value["target"], str)):
        raise CredentialError(_REFERENCE_ERROR)
    return CredentialReference(_uuid(value["gallery_id"]), _uuid(value["destination_id"]), value["target"])


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


_WidePointer = ctypes.POINTER(ctypes.c_wchar)
_BytePointer = ctypes.POINTER(ctypes.c_ubyte)


class _CredentialAttribute(ctypes.Structure):
    _fields_ = [("Keyword", _WidePointer), ("Flags", ctypes.c_uint32),
                ("ValueSize", ctypes.c_uint32), ("Value", _BytePointer)]


class _Credential(ctypes.Structure):
    _fields_ = [("Flags", ctypes.c_uint32), ("Type", ctypes.c_uint32),
                ("TargetName", _WidePointer), ("Comment", _WidePointer),
                ("LastWritten", _FileTime), ("CredentialBlobSize", ctypes.c_uint32),
                ("CredentialBlob", _BytePointer), ("Persist", ctypes.c_uint32),
                ("AttributeCount", ctypes.c_uint32), ("Attributes", ctypes.POINTER(_CredentialAttribute)),
                ("TargetAlias", _WidePointer), ("UserName", _WidePointer)]


@lru_cache(maxsize=1)
def _credential_api():
    if not supported_platform():
        raise CredentialError("Windows Credential Manager is available only on Windows; no alternate provider is used.")
    api = ctypes.WinDLL("Advapi32", use_last_error=True)
    api.CredReadW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.POINTER(ctypes.POINTER(_Credential))]
    api.CredReadW.restype = ctypes.c_int
    api.CredWriteW.argtypes = [ctypes.POINTER(_Credential), ctypes.c_uint32]
    api.CredWriteW.restype = ctypes.c_int
    api.CredFree.argtypes = [ctypes.c_void_p]
    api.CredFree.restype = None
    return api


def supported_platform() -> bool:
    """Cold platform fact only; never load a DLL or inspect any vault entry."""
    return os.name == "nt"


def _last_error() -> int:
    return ctypes.get_last_error()


def _text(pointer, maximum: int) -> str:
    if not pointer:
        raise CredentialError(_ERROR)
    characters: list[str] = []
    for index in range(maximum + 1):
        character = pointer[index]
        if character == "\x00":
            return "".join(characters)
        characters.append(character)
    raise CredentialError(_ERROR)


def checked_secret(password: object) -> str:
    """Normalize like bundle encryption and bound even an injected provider."""
    valid = isinstance(password, str) and bool(password) and len(password) <= _MAX_SECRET_CHARS
    if valid:
        assert isinstance(password, str)
        try:
            password = unicodedata.normalize("NFC", password)
            encoded = password.encode("utf-8")
            valid = (bool(encoded) and len(password) <= _MAX_SECRET_CHARS
                     and len(encoded) + len(_MAGIC) + 74 <= _MAX_BLOB_BYTES)
        except UnicodeError:
            valid = False
    if not valid:
        raise CredentialError(_SECRET_ERROR)
    assert isinstance(password, str)
    return password


def _scope(reference: CredentialReference) -> bytes:
    return _MAGIC + reference.gallery_id.encode("ascii") + b"\x00" + reference.destination_id.encode("ascii") + b"\x00"


def _blob(reference: CredentialReference, password: str) -> bytearray:
    password = checked_secret(password)
    data = bytearray(_scope(reference))
    data.extend(password.encode("utf-8"))
    if len(data) > _MAX_BLOB_BYTES:
        _wipe(data)
        raise CredentialError(_SECRET_ERROR)
    return data


def _wipe(data: bytearray) -> None:
    # Best effort for buffers owned here. Python strings/codec temporaries
    # cannot be promised deterministic zeroing.
    try:
        if data:
            ctypes.memset((ctypes.c_ubyte * len(data)).from_buffer(data), 0, len(data))
    except Exception:
        pass


_failed_targets: set[str] = set()
_uncertain_buffers: list[object] = []


def _failure_message(code: int) -> str:
    return _MISSING if code == 1168 else _SESSION if code == 1312 else _ACCESS if code == 5 else _ERROR


def _raise_failure(message: str) -> None:
    kind = (CredentialMissing if message == _MISSING else CredentialSessionUnavailable if message == _SESSION
            else CredentialAccessDenied if message == _ACCESS else CredentialError)
    raise kind(message)


def _read(reference: CredentialReference, *, preflight: bool) -> str | None:
    reference = read_reference(reference_wire(reference))
    if reference.target in _failed_targets:
        raise CredentialError(_ERROR)
    pointer = ctypes.POINTER(_Credential)()
    api = None
    data = bytearray()
    secret: str | None = None
    failure: str | None = None
    existing = False
    try:
        api = _credential_api()
        found = api.CredReadW(reference.target, 1, 0, ctypes.byref(pointer))
        if not found:
            if preflight and _last_error() == 1168 and not pointer:  # ERROR_NOT_FOUND
                pass
            else:
                failure = _failure_message(_last_error())
        elif preflight:
            existing = True  # Refuse anything, without adopting/parsing it.
        elif not pointer:
            failure = _ERROR
        else:
            item = pointer.contents
            if (item.Flags != 0 or item.Type != 1 or item.Persist != 2
                    or item.AttributeCount != 0 or bool(item.Attributes) or bool(item.TargetAlias) or bool(item.Comment)
                    or _text(item.TargetName, 256) != reference.target or _text(item.UserName, 32) != _USERNAME
                    or not item.CredentialBlob or not 1 <= item.CredentialBlobSize <= _MAX_BLOB_BYTES):
                raise CredentialError(_ERROR)
            data = bytearray(item.CredentialBlobSize)
            ctypes.memmove((ctypes.c_ubyte * len(data)).from_buffer(data), item.CredentialBlob, len(data))
            prefix = _scope(reference)
            if data[:len(prefix)] != prefix:
                raise CredentialError(_ERROR)
            secret = checked_secret(data[len(prefix):].decode("utf-8"))
    except BaseException:
        failure = _ERROR
    finally:
        _wipe(data)
        if pointer and api is not None:
            try:
                api.CredFree(pointer)  # Exactly one call, even failed/malformed reads.
            except BaseException:
                _failed_targets.add(reference.target)
                _uncertain_buffers.append((api, pointer))
                failure = _ERROR
    # Raise only after provider exception handling has ended: no raw error
    # context/cause, arbitrary messages, filenames or secret text escapes.
    if failure is not None:
        secret = None  # A fixed raised error must not retain a decoded secret in its own frame.
        _raise_failure(failure)
    if existing:
        raise CredentialExists(_EXISTS)
    return secret


@contextmanager
def read_secret(reference: CredentialReference):
    secret = _read(reference, preflight=False)
    if secret is None:
        raise CredentialError(_ERROR)
    try:
        yield secret
    finally:
        secret = None


def create_credential(reference: CredentialReference, password: str) -> None:
    """Explicit setup (`tools/backup_credential.py`) under caller reservation;
    no atomic-create claim.

    An inactive durable profile intent must exist BEFORE calling. No previous
    entry is overwritten intentionally, deleted, enumerated or repaired here.
    """
    reference = read_reference(reference_wire(reference))
    data = _blob(reference, password)
    failure: str | None = None
    try:
        _read(reference, preflight=True)
        target = ctypes.create_unicode_buffer(reference.target)
        username = ctypes.create_unicode_buffer(_USERNAME)
        buffer = (ctypes.c_ubyte * len(data)).from_buffer(data)
        item = _Credential(Flags=0, Type=1, TargetName=ctypes.cast(target, _WidePointer),
                           CredentialBlobSize=len(data), CredentialBlob=ctypes.cast(buffer, _BytePointer),
                           Persist=2, UserName=ctypes.cast(username, _WidePointer))
        try:
            if not _credential_api().CredWriteW(ctypes.byref(item), 0):
                failure = _failure_message(_last_error())
        except BaseException:
            failure = _ERROR
    finally:
        _wipe(data)
    if failure is not None:
        _failed_targets.add(reference.target)
        _raise_failure(failure)
