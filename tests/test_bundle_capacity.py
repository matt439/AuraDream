"""Logical-byte format bounds over in-memory/synthetic members only."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile

import pytest

from backend import backup_receipts, bundle_capacity as capacity, bundle_crypto


BASE = {"format": "auradream-gallery", "version": 1, "records": 1,
        "managed_backup": {"version": 1, "gallery_id": "g", "destination_id": "d", "operation_id": "o"}}


def written_zip(members):
    output = io.BytesIO()
    manifest = {**BASE, "files": {}}
    sink = backup_receipts.DigestWriter(output)
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, data in members:
            with archive.open(name, "w", force_zip64=True) as member:
                member.write(data)
            manifest["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True))
    return output.getvalue()


@pytest.mark.parametrize("name", ["images/fox.png", "images/café.png", "images/犬-U0001f415.png"])
@pytest.mark.parametrize("force_large", [False, True])
def test_forward_only_zip_bytes_fit_bound_including_real_json_names_and_zip64(name, force_large, monkeypatch):
    # Small bodies force every relevant stdlib branch without a multi-GB
    # file or 65,536-member fixture. The bound counts maxima regardless.
    if force_large:
        monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 64)
        monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 2)
    members = [("database.sqlite3", b"d" * 128), (name, b"image" * 30)]
    bound = capacity.managed_capacity([(key, len(data)) for key, data in members], BASE,
                                     manifest_name="manifest.json", encrypted=False)
    raw = written_zip(members)
    assert len(raw) <= bound.zip_bytes == bound.archive_bytes
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert all(info.flag_bits & 8 for info in archive.infolist())
        assert len(archive.read("manifest.json")) == bound.manifest_bytes
    assert bound.receipt_bytes == backup_receipts._MAX_BYTES


@pytest.mark.parametrize("size", [0, 1, bundle_crypto._CHUNK - 1, bundle_crypto._CHUNK,
                                  bundle_crypto._CHUNK + 1, 2 * bundle_crypto._CHUNK, 2 * bundle_crypto._CHUNK + 1])
def test_encrypted_bound_is_exact_for_actual_chunk_framing_including_split_writes(size):
    output = io.BytesIO()
    writer = bundle_crypto.EncryptingWriter(output, "scratch-format-only password")
    data = b"x" * size
    cut = min(size, bundle_crypto._CHUNK - 1)
    writer.write(data[:cut])
    writer.write(data[cut:])
    writer.finish()
    assert len(output.getvalue()) == capacity.encrypted_size(size)


def test_virtual_large_sizes_keep_integer_precision_without_allocating_payload():
    sizes = [("database.sqlite3", (1 << 31) + 1), ("images/large.png", (1 << 40) + 3)]
    plain = capacity.managed_capacity(sizes, BASE, manifest_name="manifest.json", encrypted=False)
    encrypted = capacity.managed_capacity(sizes, BASE, manifest_name="manifest.json", encrypted=True)
    assert plain.archive_bytes > sum(size for _, size in sizes)
    assert encrypted.archive_bytes == capacity.encrypted_size(plain.zip_bytes)
    assert encrypted.manifest_bytes == plain.manifest_bytes


def test_manifest_base_is_not_mutated_and_budget_facts_are_frozen():
    base = {**BASE, "files": {}}
    bound = capacity.managed_capacity([("database.sqlite3", 0)], base, manifest_name="manifest.json", encrypted=False)
    assert base["files"] == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        bound.archive_bytes = 0


@pytest.mark.parametrize("size", [-1, True, 1.5, "4", None])
def test_capacity_rejects_invalid_sizes(size):
    with pytest.raises(ValueError):
        capacity.encrypted_size(size)
    with pytest.raises(ValueError):
        capacity.managed_capacity([("database.sqlite3", size)], BASE, manifest_name="manifest.json", encrypted=False)


@pytest.mark.parametrize("members", [[("same", 0), ("same", 1)], [("manifest.json", 0)],
                                   [("", 0)], [("null\0name", 0)], [("\ud800", 0)]])
def test_duplicate_or_unencodable_names_cannot_understate_a_budget(members):
    with pytest.raises(ValueError):
        capacity.managed_capacity(members, BASE, manifest_name="manifest.json", encrypted=False)


def test_current_format_limits_refuse_before_any_write(monkeypatch):
    maximum_plaintext = bundle_crypto._CHUNK * (1 << 32)
    assert capacity.encrypted_size(maximum_plaintext) > maximum_plaintext
    with pytest.raises(ValueError, match="chunk limit"):
        capacity.encrypted_size(maximum_plaintext + 1)
    with pytest.raises(ValueError, match="ZIP64"):
        capacity.managed_capacity([("database.sqlite3", (1 << 64) - 1)], BASE,
                                  manifest_name="manifest.json", encrypted=False)
    monkeypatch.setattr(capacity, "_MANIFEST_LIMIT", 1)
    with pytest.raises(ValueError, match="manifest"):
        capacity.managed_capacity([], BASE, manifest_name="manifest.json", encrypted=False)


def test_protection_is_explicit_boolean():
    with pytest.raises(ValueError, match="boolean"):
        capacity.managed_capacity([], BASE, manifest_name="manifest.json", encrypted="encrypted")
