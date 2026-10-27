"""Bounded journal and raw identity checks over explicitly supplied scratch facts."""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend import backup_receipts as receipts

GALLERY = "11111111-1111-4111-8111-111111111111"
DESTINATION = "22222222-2222-4222-8222-222222222222"
OPERATION = "33333333-3333-4333-8333-333333333333"
OTHER = "44444444-4444-4444-8444-444444444444"
IDENTITY = receipts.ManagedIdentity(GALLERY, DESTINATION, OPERATION)


def fact(directory: Path, *, identity=IDENTITY, name="copy.zip", data=b"verified fixture") -> receipts.Receipt:
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return receipts.Receipt(identity, str(directory.resolve()), name, hashlib.sha256(data).hexdigest(),
                            receipts.FileStamp.of(path.stat()), datetime(2026, 10, 26, tzinfo=timezone.utc),
                            "unencrypted", 1, 1)


@pytest.mark.parametrize("value", [None, True, 1, "", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", OPERATION.replace("-", ""), "0" * 36,
                                   "00000000-0000-0000-0000-000000000000", "not-a-uuid"])
def test_identity_requires_canonical_nonzero_uuid(value):
    with pytest.raises(receipts.ReceiptError, match="UUID"):
        receipts.ManagedIdentity(GALLERY, DESTINATION, value)


def test_marker_is_strict_and_round_trips():
    assert receipts.read_marker(IDENTITY.marker()) == IDENTITY
    for value in (None, {}, {**IDENTITY.marker(), "password": "hidden"},
                  {**IDENTITY.marker(), "version": True}, {**IDENTITY.marker(), "version": 2}):
        with pytest.raises(receipts.ReceiptError):
            receipts.read_marker(value)


@pytest.mark.parametrize("name", ["CON.zip", "nul.zip", "COM1.zip", "lpt9.ldbundle", "CON.extra.zip",
                                   "prn.data.zip", "aUx.zip"])
def test_receipt_rejects_portably_reserved_device_basenames(tmp_path, name):
    one = fact(tmp_path / "archives")
    with pytest.raises(receipts.ReceiptError, match="basename"):
        replace(one, name=name)


@pytest.mark.parametrize("name", ["console.zip", "com0.zip", "COM10.zip", "lpt0.zip", "copy.CON.zip"])
def test_portable_nondevice_names_are_allowed(tmp_path, name):
    one = fact(tmp_path / "archives")
    assert replace(one, name=name).name == name


def test_forward_writer_hashes_outer_bytes_and_refuses_seeking():
    stream = io.BytesIO()
    writer = receipts.DigestWriter(stream)
    assert writer.write(b"first") == 5
    assert writer.write(memoryview(b"second")) == 6
    assert writer.count == writer.tell() == 11
    assert writer.sha256 == hashlib.sha256(stream.getvalue()).hexdigest()
    with pytest.raises(io.UnsupportedOperation):
        writer.seek(0)
    writer.close()
    assert not stream.closed


def test_short_write_cannot_produce_a_complete_proof():
    class Short(io.BytesIO):
        def write(self, data):
            return super().write(data[:1])

    writer = receipts.DigestWriter(Short())
    with pytest.raises(OSError, match="short"):
        writer.write(b"incomplete")
    assert writer.count == 0


def test_missing_journal_does_not_create_it_and_databases_are_isolated(tmp_path):
    db = tmp_path / "absent" / "one.db"
    assert receipts.load(db, GALLERY) == ()
    assert not db.parent.exists()
    assert receipts.path_for(db) != receipts.path_for(db.with_name("two.db"))


def test_append_round_trips_and_preserves_earlier_receipts(tmp_path):
    db = tmp_path / "gallery.db"
    one = fact(tmp_path / "archives")
    two = fact(tmp_path / "archives", identity=replace(IDENTITY, operation_id=OTHER), name="two.zip")
    receipts.append(db, one)
    receipts.append(db, two)
    assert receipts.load(db, GALLERY) == (one, two)
    assert receipts.load(db.with_name("another.db"), GALLERY) == ()
    assert receipts.matches_current(one, Path(one.directory))


@pytest.mark.parametrize("kind", ["operation", "archive", "gallery", "destination"])
def test_conflicting_or_duplicate_receipts_never_overwrite_journal(tmp_path, kind):
    db = tmp_path / "gallery.db"
    one = fact(tmp_path / "archives")
    receipts.append(db, one)
    before = receipts.path_for(db).read_bytes()
    two = fact(tmp_path / "elsewhere", identity=replace(IDENTITY, operation_id=OTHER), name="two.zip")
    if kind == "operation":
        two = replace(two, identity=IDENTITY)
    elif kind == "archive":
        two = replace(one, identity=replace(IDENTITY, operation_id=OTHER))
    elif kind == "gallery":
        two = replace(two, identity=replace(two.identity, gallery_id=OTHER))
    # destination kind keeps one destination UUID but gives it another path.
    with pytest.raises(receipts.ReceiptError):
        receipts.append(db, two)
    assert receipts.path_for(db).read_bytes() == before


@pytest.mark.parametrize("raw", [b"", b"null", b"[]", b"{}", b"\xff", b'{"version":NaN}',
    b'{"version":1,"version":1}', b'{"version":2,"gallery_id":"secret","receipts":[]}',
    b"[" * 2000 + b"]" * 2000,
], ids=lambda raw: f"{len(raw)}-bytes")
def test_malformed_journal_is_fixed_error_and_never_rewritten(tmp_path, raw):
    db = tmp_path / "gallery.db"
    path = receipts.path_for(db)
    path.write_bytes(raw)
    with pytest.raises(receipts.ReceiptError) as error:
        receipts.load(db, GALLERY)
    assert str(error.value) == receipts._ERROR
    assert "secret" not in str(error.value)
    assert path.read_bytes() == raw


def test_unknown_or_duplicate_entry_fields_and_wrong_gallery_refuse(tmp_path):
    db = tmp_path / "gallery.db"
    one = fact(tmp_path / "archives")
    receipts.append(db, one)
    path = receipts.path_for(db)
    original = path.read_bytes()
    payload = json.loads(original)
    for change in (lambda p: p["receipts"].append(p["receipts"][0]),
                   lambda p: p["receipts"][0].update(password="secret"),
                   lambda p: p.update(gallery_id=OTHER),
                   lambda p: p["receipts"][0].update(records=True)):
        payload = json.loads(original)
        change(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(receipts.ReceiptError):
            receipts.load(db, GALLERY)


@pytest.mark.parametrize("boundary", ["bytes", "entries"])
def test_journal_limits_refuse_without_destroying_previous_bytes(tmp_path, monkeypatch, boundary):
    db = tmp_path / "gallery.db"
    one = fact(tmp_path / "archives")
    receipts.append(db, one)
    path = receipts.path_for(db)
    before = path.read_bytes()
    if boundary == "bytes":
        monkeypatch.setattr(receipts, "_MAX_BYTES", len(before) - 1)
        with pytest.raises(receipts.ReceiptError):
            receipts.load(db, GALLERY)
    else:
        monkeypatch.setattr(receipts, "MAX_RECEIPTS", 1)
        two = fact(tmp_path / "archives", identity=replace(IDENTITY, operation_id=OTHER), name="two.zip")
        with pytest.raises(receipts.ReceiptError):
            receipts.append(db, two)
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_journal_write_failure_preserves_previous_bytes_and_owned_temp_cleanup(tmp_path, monkeypatch, failure):
    db = tmp_path / "gallery.db"
    one = fact(tmp_path / "archives")
    receipts.append(db, one)
    path = receipts.path_for(db)
    before = path.read_bytes()
    unrelated = tmp_path / ".auradream.backup-receipts.unrelated.tmp"
    unrelated.write_bytes(b"keep")
    two = fact(tmp_path / "archives", identity=replace(IDENTITY, operation_id=OTHER), name="two.zip")

    def fail(*_args):
        raise OSError("injected journal failure")

    monkeypatch.setattr(receipts.os, failure, fail)
    with pytest.raises(OSError):
        receipts.append(db, two)
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".auradream.backup-receipts.*.tmp")) == [unrelated]


def test_missing_replaced_changed_and_wrong_destination_files_match_no_receipt(tmp_path):
    one = fact(tmp_path / "archives")
    directory = Path(one.directory)
    path = directory / one.name
    assert not receipts.matches_current(one, tmp_path)
    # Same length and original manifest-like prefix still cannot establish identity.
    path.write_bytes(b"differentfixture")
    os.utime(path, ns=(path.stat().st_atime_ns, one.stamp.modified_ns))
    altered_stamp = receipts.FileStamp.of(path.stat())
    assert not receipts.matches_current(replace(one, stamp=altered_stamp), directory)
    path.unlink()
    assert not receipts.matches_current(one, directory)
    path.write_bytes(b"verified fixture")
    assert not receipts.matches_current(one, directory)  # Same bytes, new file identity.


def test_archive_changed_while_hashing_is_refused(tmp_path, monkeypatch):
    one = fact(tmp_path / "archives")
    real = receipts.os.fstat
    calls = 0

    def changed(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            (Path(one.directory) / one.name).write_bytes(b"changed mid-check")
        return real(fd)

    monkeypatch.setattr(receipts.os, "fstat", changed)
    assert not receipts.matches_current(one, Path(one.directory))


def test_a_journal_named_before_the_rename_is_used_in_place(tmp_path):
    db = tmp_path / "gallery.db"
    one = fact(tmp_path / "archives")
    receipts.append(db, one)
    legacy = tmp_path / "local-diffusion.backup-receipts-gallery.db.json"
    receipts.path_for(db).rename(legacy)
    two = fact(tmp_path / "archives", identity=replace(IDENTITY, operation_id=OTHER), name="two.zip")
    receipts.append(db, two)
    assert receipts.load(db, GALLERY) == (one, two)
    assert receipts.path_for(db) == legacy
    assert not (tmp_path / "auradream.backup-receipts-gallery.db.json").exists()
