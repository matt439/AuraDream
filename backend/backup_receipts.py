"""Internal managed-export evidence; no activation, age decisions or deletion.

UUIDs are caller-supplied canonical, nonzero identifiers, not inferred from
filenames. The v1 journal binds a verified publication to its outer bytes and
file identity. Missing journals establish no ownership; malformed journals are
never repaired or reconstructed automatically. Bounds are engineering limits.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
from collections.abc import Buffer
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Never
from uuid import UUID

from . import gallery_lock
from .backup_policy import Protection

MARKER = "managed_backup"
MAX_RECEIPTS = 20_000
_MAX_BYTES = 16 * 1024 * 1024
_CHUNK = 1024 * 1024
_ERROR = "The managed backup journal is unusable; repair it before recording managed copies."
JOURNAL_WARNING = "The archive was published, but its receipt could not be saved; it remains unmanaged."


class ReceiptError(ValueError):
    pass


def _uuid(value: object) -> str:
    if not isinstance(value, str) or len(value) != 36:
        raise ReceiptError("managed identities must be canonical nonzero UUIDs")
    try:
        parsed = UUID(value)
    except ValueError:
        raise ReceiptError("managed identities must be canonical nonzero UUIDs") from None
    if str(parsed) != value or not parsed.int:
        raise ReceiptError("managed identities must be canonical nonzero UUIDs")
    return value


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReceiptError("invalid receipt integer")
    return value


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ReceiptError("receipt fields do not match the known schema")
    return {key: value[key] for key in keys}


@dataclass(frozen=True)
class ManagedIdentity:
    gallery_id: str
    destination_id: str
    operation_id: str

    def __post_init__(self) -> None:
        for value in (self.gallery_id, self.destination_id, self.operation_id):
            _uuid(value)

    def marker(self) -> dict[str, object]:
        return {"version": 1, **asdict(self)}


def read_marker(value: object) -> ManagedIdentity:
    fields = _object(value, {"version", "gallery_id", "destination_id", "operation_id"})
    if type(fields["version"]) is not int or fields["version"] != 1:
        raise ReceiptError("unsupported managed marker version")
    return ManagedIdentity(*(_uuid(fields[key]) for key in ("gallery_id", "destination_id", "operation_id")))


class DigestWriter(io.RawIOBase):
    """Forward-only outer sink: hash exactly the ZIP bytes or ciphertext written.

    ZIP uses data descriptors because seek is unsupported. This wrapper does
    not own or close its stream; the exporter flushes and fsyncs that stream.
    """

    def __init__(self, stream: BinaryIO):
        super().__init__()
        self._stream = stream
        self._digest = hashlib.sha256()
        self.count = 0

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.count

    def write(self, b: Buffer) -> int:
        data = memoryview(b).cast("B")
        written = self._stream.write(data)
        if written != len(data):
            raise OSError("short managed archive write")
        self._digest.update(data)
        self.count += written
        return written

    def flush(self) -> None:
        pass

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


@dataclass(frozen=True)
class FileStamp:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            _integer(value)
        if self.inode == 0:
            raise ReceiptError("the archive has no stable file identity")

    @classmethod
    def of(cls, value: os.stat_result) -> FileStamp:
        if not stat.S_ISREG(value.st_mode):
            raise ReceiptError("a managed archive must be a regular file")
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def same_file(self, other: FileStamp) -> bool:
        # Windows path stat and handle fstat can expose different ctime values.
        # Compare change times only within one source, never across those APIs.
        return (self.device, self.inode, self.size, self.modified_ns) == (
            other.device, other.inode, other.size, other.modified_ns,
        )


@dataclass(frozen=True)
class Receipt:
    identity: ManagedIdentity
    directory: str
    name: str
    sha256: str
    stamp: FileStamp
    verified_at: datetime
    protection: Protection
    records: int
    images: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ManagedIdentity) or not isinstance(self.stamp, FileStamp):
            raise ReceiptError("invalid receipt identity")
        # Absolute, already resolved destination supplied by bundle_directory().
        if (not isinstance(self.directory, str) or len(self.directory) > 4096
                or any(ord(char) < 32 for char in self.directory)
                or not Path(self.directory).is_absolute() or ".." in Path(self.directory).parts
                or str(Path(self.directory)) != self.directory):
            raise ReceiptError("a receipt needs an absolute destination")
        if not isinstance(self.name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,240}\.(zip|ldbundle)", self.name) is None:
            raise ReceiptError("invalid managed archive basename")
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", self.name.split(".", 1)[0], re.IGNORECASE):
            raise ReceiptError("invalid managed archive basename")
        if not isinstance(self.sha256, str) or re.fullmatch("[0-9a-f]{64}", self.sha256) is None:
            raise ReceiptError("invalid managed archive digest")
        if self.protection not in {"unencrypted", "encrypted"}:
            raise ReceiptError("invalid managed archive protection")
        if self.name.endswith(".ldbundle") != (self.protection == "encrypted") or self.stamp.size < 1:
            raise ReceiptError("managed archive protection or size mismatch")
        _integer(self.records)
        _integer(self.images)
        if not isinstance(self.verified_at, datetime) or self.verified_at.utcoffset() is None:
            raise ReceiptError("receipt verification time must be timezone-aware")
        object.__setattr__(self, "verified_at", self.verified_at.astimezone(timezone.utc))


def path_for(db_path: Path) -> Path:
    database = Path(db_path).expanduser()
    if not database.name:
        raise ReceiptError("a journal needs a database filename")
    return database.parent / f"local-diffusion.backup-receipts-{database.name}.json"


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(_ERROR)
        result[key] = value
    return result


def _constant(_value: str) -> Never:
    raise ReceiptError(_ERROR)


def _wire(receipt: Receipt) -> dict[str, object]:
    return {**asdict(receipt), "identity": receipt.identity.marker(), "verified_at": receipt.verified_at.isoformat()}


def _decode(value: object) -> Receipt:
    fields = _object(value, {"identity", "directory", "name", "sha256", "stamp", "verified_at", "protection", "records", "images"})
    stamp = _object(fields["stamp"], {"device", "inode", "size", "modified_ns", "changed_ns"})
    for key in ("directory", "name", "sha256", "verified_at"):
        if not isinstance(fields[key], str):
            raise ReceiptError(_ERROR)
    directory, name, digest, timestamp = (str(fields[key]) for key in ("directory", "name", "sha256", "verified_at"))
    if len(timestamp) > 40:
        raise ReceiptError(_ERROR)
    protection = fields["protection"]
    if protection not in ("unencrypted", "encrypted"):
        raise ReceiptError(_ERROR)
    return Receipt(
        read_marker(fields["identity"]), directory, name, digest,
        FileStamp(*(_integer(stamp[key]) for key in ("device", "inode", "size", "modified_ns", "changed_ns"))),
        datetime.fromisoformat(timestamp), "encrypted" if protection == "encrypted" else "unencrypted",
        _integer(fields["records"]), _integer(fields["images"]),
    )


def _validate(receipts: tuple[Receipt, ...], gallery_id: str) -> None:
    if len(receipts) > MAX_RECEIPTS:
        raise ReceiptError(_ERROR)
    operations: set[str] = set()
    archives: set[tuple[str, str]] = set()
    destinations: dict[str, str] = {}
    for receipt in receipts:
        if receipt.identity.gallery_id != gallery_id:
            raise ReceiptError(_ERROR)
        archive = (os.path.normcase(receipt.directory), receipt.name.casefold())
        operation = receipt.identity.operation_id
        if operation in operations or archive in archives:
            raise ReceiptError(_ERROR)
        destination = destinations.setdefault(receipt.identity.destination_id, receipt.directory)
        if destination != receipt.directory:
            raise ReceiptError(_ERROR)
        operations.add(operation)
        archives.add(archive)


def load(db_path: Path, gallery_id: str) -> tuple[Receipt, ...]:
    """Read a strict bounded journal; missing means no managed copies, no writes."""
    _uuid(gallery_id)
    path = path_for(db_path)
    try:
        if path.is_symlink() or path.is_junction():
            raise ReceiptError(_ERROR)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ReceiptError(_ERROR)
        with path.open("rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise ReceiptError(_ERROR)
        payload: object = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        fields = _object(payload, {"version", "gallery_id", "receipts"})
        if type(fields["version"]) is not int or fields["version"] != 1 or fields["gallery_id"] != gallery_id:
            raise ReceiptError(_ERROR)
        entries = fields["receipts"]
        if not isinstance(entries, list) or len(entries) > MAX_RECEIPTS:
            raise ReceiptError(_ERROR)
        receipts = tuple(_decode(entry) for entry in entries)
        _validate(receipts, gallery_id)
        return receipts
    except FileNotFoundError:
        return ()
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        raise ReceiptError(_ERROR) from None


def append(db_path: Path, receipt: Receipt) -> None:
    """Record an already verified publication under the existing reservation.

    Failure leaves the previous journal intact. This does not adopt orphaned
    archives or confer age/retention authority, including on an empty export.
    """
    with gallery_lock.hold(db_path):
        # Round-trip validation also catches unusually mutated dataclasses.
        receipt = _decode(_wire(receipt))
        receipts = (*load(db_path, receipt.identity.gallery_id), receipt)
        _validate(receipts, receipt.identity.gallery_id)
        data = (json.dumps({"version": 1, "gallery_id": receipt.identity.gallery_id,
                            "receipts": [_wire(item) for item in receipts]}, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
        if len(data) > _MAX_BYTES:
            raise ReceiptError(_ERROR)
        path = path_for(db_path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                             prefix=".local-diffusion.backup-receipts.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def matches_current(receipt: Receipt, directory: Path) -> bool:
    """Check current raw identity without a password; never delete or adopt.

    A later coordinator must still hold admission/reservation and account for
    download leases. A true result is a snapshot, not deletion authorization.
    """
    # Keep lease errors outside the False-return catch. Otherwise a broken
    # installed mode could masquerade as a missing copy and trigger an export.
    from .bundle_leases import reading

    # Reject the wrong scope before creating any companion in an unrelated
    # installed destination. The leased check below repeats these cheap facts.
    try:
        if directory.is_symlink() or directory.is_junction() or not directory.is_dir() or directory.resolve() != Path(receipt.directory):
            return False
    except (OSError, ValueError, TypeError):
        return False
    with reading(directory / receipt.name) as read_owner:
        return _matches_current(receipt, directory, read_owner)


def _matches_current(receipt: Receipt, directory: Path, read_owner=None) -> bool:
    """Unleased fact check, not an exclusive/handle-bound mutation authority."""
    try:
        if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
            return False
        if directory.resolve() != Path(receipt.directory):
            return False
        path = directory / receipt.name
        if FileStamp.of(path.lstat()) != receipt.stamp:
            return False
        digest = hashlib.sha256()
        from .bundle_leases import _open_input

        with _open_input(path, read_owner) as stream:
            opened = FileStamp.of(os.fstat(stream.fileno()))
            if not opened.same_file(receipt.stamp):
                return False
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
            if FileStamp.of(os.fstat(stream.fileno())) != opened:
                return False
        return FileStamp.of(path.lstat()) == receipt.stamp and digest.hexdigest() == receipt.sha256
    except (OSError, ValueError, TypeError):
        return False
