"""Conservative logical-byte budgets for the current managed export format.

These are preflights, not allocation guarantees: filesystem metadata,
allocation units, SQLite growth/temp placement and other writers can race
the checks. No file content, password, cipher or archive is read or written.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .backup_receipts import _MAX_BYTES as JOURNAL_TEMP_BYTES
from .bundle_crypto import HEADER_SIZE, _CHUNK as _CIPHER_CHUNK, _TAG as _CIPHER_TAG


# Current ZIP_STORED writer: no member/archive comments or user extra fields.
# Local header30 + forced-ZIP64 extra20 + descriptor24 + central header46
# + maximal central ZIP64 extra28. Names occur in both headers.
_ENTRY_OVERHEAD = 30 + 20 + 24 + 46 + 28
_END_OVERHEAD = 22 + 56 + 20  # EOCD plus maximal ZIP64 EOCD and locator.
_MANIFEST_LIMIT = 32 * 1024 * 1024  # Existing verified archive reader's limit.
_UINT64_MAX = (1 << 64) - 1


def _size(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("capacity sizes must be nonnegative integers")
    return value


def encrypted_size(plaintext_bytes: int) -> int:
    """Current framing: final full chunks are held, not followed by an empty one."""
    count = max(1, (_size(plaintext_bytes) + _CIPHER_CHUNK - 1) // _CIPHER_CHUNK)
    if count > 1 << 32:
        raise ValueError("the managed archive exceeds the encrypted format's chunk limit")
    return plaintext_bytes + HEADER_SIZE + _CIPHER_TAG * count


@dataclass(frozen=True)
class ManagedCapacity:
    zip_bytes: int
    archive_bytes: int
    manifest_bytes: int
    receipt_bytes: int


def managed_capacity(
    members: Sequence[tuple[str, int]], manifest_base: Mapping[str, object], *,
    manifest_name: str, encrypted: bool,
) -> ManagedCapacity:
    """Bound forward-only managed output using stats and exact JSON-name escaping.

    Real SHA-256 strings have exactly the placeholder's encoded length. UTF-8
    name lengths also bound zipfile's shorter CP437 path. Count maximal ZIP64
    extras/descriptors for every member, including the manifest, regardless
    of which size/count/offset thresholds this particular export crosses.
    """
    if type(encrypted) is not bool:
        raise ValueError("the capacity protection flag must be a boolean")
    files: dict[str, dict[str, int | str]] = {}
    names = [manifest_name]
    payload = 0
    for name, size in members:
        if name in files or name == manifest_name:
            raise ValueError("managed capacity members must have unique names")
        payload += _size(size)
        files[name] = {"size": size, "sha256": "0" * 64}
        names.append(name)
    if any(not isinstance(name, str) or not name or "\0" in name for name in names):
        raise ValueError("managed capacity members need nonempty text names without NUL")
    manifest_bytes = len(json.dumps({**manifest_base, "files": files}, ensure_ascii=True).encode("utf-8"))
    if manifest_bytes > _MANIFEST_LIMIT:
        raise ValueError("the managed archive manifest exceeds the verified format limit")
    name_bytes = sum(len(name.encode("utf-8")) for name in names)
    zip_bytes = payload + manifest_bytes + len(names) * _ENTRY_OVERHEAD + 2 * name_bytes + _END_OVERHEAD
    if zip_bytes > _UINT64_MAX:
        raise ValueError("the managed archive exceeds the ZIP64 format limit")
    return ManagedCapacity(zip_bytes, encrypted_size(zip_bytes) if encrypted else zip_bytes,
                           manifest_bytes, JOURNAL_TEMP_BYTES)
