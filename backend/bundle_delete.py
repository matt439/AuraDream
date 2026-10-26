"""Unconnected Windows handle-bound managed archive deletion mechanism.

The caller must supply a receipt validated against its journal, exact intended
scope and a policy-approved candidate under the gallery reservation BEFORE
binding. This module chooses no retention count, writes no journal and has no
startup/API caller. A receipt or installed lease mode alone is not consent.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import stat
import threading
from typing import Literal

from . import backup_receipts as receipts, bundle_file_lock, bundle_leases, bundle_mode

_CHUNK = 1024 * 1024


class DeleteSafetyError(ValueError):
    """The bound archive cannot establish the requested deletion identity."""


class _StandardInfo(ctypes.Structure):
    _fields_ = [("AllocationSize", ctypes.c_int64), ("EndOfFile", ctypes.c_int64),
                ("NumberOfLinks", wintypes.DWORD), ("DeletePending", ctypes.c_ubyte),
                ("Directory", ctypes.c_ubyte)]


class _AttributeInfo(ctypes.Structure):
    _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]


class _DispositionInfo(ctypes.Structure):
    # Win32 BOOLEAN is one BYTE, not the four-byte BOOL.
    _fields_ = [("DeleteFile", ctypes.c_ubyte)]


@dataclass(frozen=True)
class DeletionOutcome:
    mark: Literal["not_requested", "refused", "accepted", "unknown"]
    owned_archive_close_confirmed: bool
    physical_removal_confirmed: bool = False


@lru_cache(maxsize=1)
def _windows_api():
    api = bundle_file_lock._windows_api()
    api.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    api.GetFileInformationByHandleEx.restype = wintypes.BOOL
    api.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    api.SetFileInformationByHandle.restype = wintypes.BOOL
    return api


def _error(operation: str, path: Path) -> OSError:
    return bundle_file_lock._windows_error(ctypes.get_last_error(), operation, path)


def _regular(value: os.stat_result) -> receipts.FileStamp:
    if (bundle_mode._linked(value) or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1):
        raise DeleteSafetyError("The managed archive must be a single-link regular non-reparse file.")
    return receipts.FileStamp.of(value)


_quarantined: list[BoundArchive] = []


class BoundArchive(AbstractContextManager["BoundArchive"]):
    """Single-use mechanism; all pins and native exclusion survive uncertainty."""

    def __init__(self, receipt: receipts.Receipt, installed: bundle_mode.SharedMode,
                 expected_identity: receipts.ManagedIdentity):
        # Reuse the strict receipt codec; these facts do not prove journal membership.
        self.receipt = receipts._decode(receipts._wire(receipt))
        if not isinstance(expected_identity, receipts.ManagedIdentity) or self.receipt.identity != expected_identity:
            raise DeleteSafetyError("The managed archive does not match the caller's intended identity.")
        if not isinstance(installed, bundle_mode.SharedMode):
            raise bundle_mode.ModeError("A positive expected installed shared mode is required.")
        self.installed = installed
        self.path = Path(self.receipt.directory) / self.receipt.name
        if os.path.normcase(self.receipt.directory) != os.path.normcase(str(installed.directory)):
            raise DeleteSafetyError("The receipt is outside the expected shared-mode destination.")
        self._claim = bundle_leases.exclusive_for_mode(self.path, installed)
        self._entered = False
        self._closed = False
        self._failed: BaseException | None = None
        self._fds: dict[str, int] = {}
        self._raw_handles: list[int] = []
        self._opened: receipts.FileStamp | None = None
        self._mark: Literal["not_requested", "refused", "accepted", "unknown"] = "not_requested"
        self._mark_attempted = False
        self._archive_close_confirmed = False
        self._operation_lock = threading.RLock()

    @property
    def outcome(self) -> DeletionOutcome:
        # Even a successful owned close cannot account for every foreign
        # attribute-only handle, nor assert power-loss durability.
        with self._operation_lock:
            return DeletionOutcome(self._mark, self._archive_close_confirmed)

    def _quarantine(self, error: BaseException) -> None:
        if self._failed is None:
            self._failed = error
            _quarantined.append(self)
            self._claim.retain(error)
        if self._mark in {"accepted", "unknown"}:
            error.add_note("Deletion was accepted or its result is uncertain; physical removal is unknown. Do not assume this archive was preserved.")
        error.add_note("Native mutation handles and the original exclusive claim are quarantined until restart; no implicit retry.")

    def _open(self, kind: str, path: Path, access: int, sharing: int, flags: int) -> None:
        import msvcrt

        api = _windows_api()
        handle = api.CreateFileW(str(path), access, sharing, None, 3, flags, None)  # OPEN_EXISTING
        if handle == ctypes.c_void_p(-1).value:
            raise _error(f"CreateFileW managed deletion {kind}", path)
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
        except BaseException as error:
            cleanup: BaseException | None = None
            try:
                if not api.CloseHandle(handle):
                    cleanup = _error(f"CloseHandle untransferred managed deletion {kind}", path)
            except BaseException as failure:
                cleanup = failure
            if cleanup is not None:
                self._raw_handles.append(handle)
                self._quarantine(cleanup)
                error.add_note(str(cleanup) + " " + " ".join(getattr(cleanup, "__notes__", ())))
            raise
        # CRT now owns the handle. Only os.close(fd) may close it.
        self._fds[kind] = fd

    def _validate_scope(self) -> None:
        self._claim.require()
        directory = self.installed.directory
        opened = os.fstat(self._fds["directory"])
        if (bundle_mode._linked(opened) or not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != self.installed.directory_identity
                or bundle_mode._directory_identity(directory) != self.installed.directory_identity):
            raise DeleteSafetyError("The pinned managed archive destination changed identity.")
        marker = directory / bundle_mode.MARKER_NAME
        fd = self._fds["marker"]
        before = bundle_mode._file_identity(os.fstat(fd))
        if before[:4] != self.installed.marker_identity[:4]:
            raise DeleteSafetyError("The pinned shared-mode marker changed identity.")
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, bundle_mode._LIMIT + 1)
        if (len(data) > bundle_mode._LIMIT or hashlib.sha256(data).hexdigest() != self.installed.digest
                or bundle_mode._file_identity(os.fstat(fd)) != before
                or bundle_mode._file_identity(marker.lstat()) != self.installed.marker_identity):
            raise DeleteSafetyError("The pinned shared-mode marker changed contents or identity.")

    def _validate_archive(self) -> receipts.FileStamp:
        import msvcrt

        fd = self._fds["archive"]
        opened = _regular(os.fstat(fd))
        named = _regular(self.path.lstat())
        if named != self.receipt.stamp or not opened.same_file(self.receipt.stamp):
            raise DeleteSafetyError("The managed archive no longer matches its receipt identity.")
        api, handle = _windows_api(), msvcrt.get_osfhandle(fd)  # Borrowed, never CloseHandle.
        standard, attributes = _StandardInfo(), _AttributeInfo()
        for information, value in ((1, standard), (9, attributes)):  # FileStandardInfo, FileAttributeTagInfo
            if not api.GetFileInformationByHandleEx(handle, information, ctypes.byref(value), ctypes.sizeof(value)):
                raise _error("GetFileInformationByHandleEx managed archive", self.path)
        if (standard.NumberOfLinks != 1 or standard.DeletePending or standard.Directory
                or standard.EndOfFile != opened.size or attributes.FileAttributes & (0x400 | 0x10)):
            raise DeleteSafetyError("The opened managed archive is linked, reparse, a directory or already delete-pending.")
        if self._opened is not None and opened != self._opened:
            raise DeleteSafetyError("The bound managed archive changed during verification.")
        return opened

    def __enter__(self) -> BoundArchive:
        with self._operation_lock:
            return self._enter()

    def _enter(self) -> BoundArchive:
        if self._entered:
            raise RuntimeError("This bound archive has already been entered.")
        if os.name != "nt":
            raise bundle_mode.UnsupportedMode("Handle-bound managed deletion is qualified only on Windows fixed local NTFS.")
        self._claim.__enter__()
        self._entered = True
        try:
            directory = self.installed.directory
            self._open("directory", directory, 0x80, 3, 0x02200000)  # attributes, no DELETE sharing, directory+reparse
            self._open("marker", directory / bundle_mode.MARKER_NAME, 0x80000000, 1, 0x00200000)  # READ only sharing
            self._validate_scope()
            # Exactly one archive handle, READ|DELETE, share0, OPEN_REPARSE_POINT.
            # Never DELETE_ON_CLOSE: nothing marks deletion before validation.
            self._open("archive", self.path, 0x80010000, 0, 0x00200000)
            self._opened = self._validate_archive()
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def mark_for_deletion(self) -> DeletionOutcome:
        # Never let another thread close/recycle this fd during the raw hash
        # or native mark. A second mark remains refused after waiting.
        with self._operation_lock:
            return self._mark_once()

    def _mark_once(self) -> DeletionOutcome:
        if not self._entered or self._closed or self._failed is not None or self._mark_attempted:
            raise RuntimeError("This bound archive cannot issue another deletion mark.")
        self._mark_attempted = True
        self._validate_scope()
        self._validate_archive()
        # One full raw-byte pass at the mutation boundary, rather than a proof
        # taken before an arbitrarily long caller context body.
        fd = self._fds["archive"]
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(fd, _CHUNK):
            digest.update(chunk)
        if digest.hexdigest() != self.receipt.sha256:
            raise DeleteSafetyError("The managed archive raw digest no longer matches its receipt.")
        self._validate_scope()
        self._validate_archive()  # Same fd and named identity immediately before mark.
        import msvcrt

        disposition = _DispositionInfo(1)
        try:
            accepted = _windows_api().SetFileInformationByHandle(
                msvcrt.get_osfhandle(self._fds["archive"]), 4, ctypes.byref(disposition), ctypes.sizeof(disposition),
            )
        except BaseException as error:
            self._mark = "unknown"
            self._quarantine(error)
            raise
        if not accepted:
            self._mark = "refused"
            raise _error("SetFileInformationByHandle managed archive deletion", self.path)
        self._mark = "accepted"
        return self.outcome

    def close(self) -> None:
        with self._operation_lock:
            self._close()

    def _close(self) -> None:
        if self._closed:
            return
        if self._failed is not None:
            raise RuntimeError("An earlier native mutation result/close is uncertain; no implicit retry.") from self._failed
        if not self._entered:
            return
        # Keep every remaining pin and the native exclusive claim if any close
        # is uncertain. Never resolve paths or retry a possibly reused fd.
        for kind in ("archive", "marker", "directory"):
            fd = self._fds.get(kind)
            if fd is None:
                continue
            try:
                os.close(fd)
            except BaseException as error:
                self._quarantine(error)
                raise
            del self._fds[kind]
            if kind == "archive":
                self._archive_close_confirmed = True
        try:
            self._claim.__exit__(None, None, None)
        except BaseException as error:
            lease = self._claim._lease
            if lease is not None and not lease._released:
                self._quarantine(error)
            else:
                self._closed = True
            raise
        self._closed = True

    def __exit__(self, _kind: object, error: BaseException | None, _traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note(f"Closing the bound managed archive also failed: {cleanup} "
                           + " ".join(getattr(cleanup, "__notes__", ())))


def bind_managed_archive(receipt: receipts.Receipt, installed: bundle_mode.SharedMode,
                         expected_identity: receipts.ManagedIdentity) -> BoundArchive:
    """Bind one caller-authorized candidate; never select policy or activate pruning."""
    return BoundArchive(receipt, installed, expected_identity)
