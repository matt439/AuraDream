"""Pure journal transitions over owned synthetic facts; no native deletion."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from backend import backup_receipts as receipts, bundle_mode

GALLERY = "11111111-1111-4111-8111-111111111111"
DESTINATION = "22222222-2222-4222-8222-222222222222"


def fact(directory, *, name="old.zip", operation=None, destination=DESTINATION, records=1):
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_bytes(b"synthetic receipt proof only: " + name.encode())
    return receipts.Receipt(receipts.ManagedIdentity(GALLERY, destination, operation or str(uuid4())),
                            str(directory.resolve()), name, hashlib.sha256(path.read_bytes()).hexdigest(),
                            receipts.FileStamp.of(path.stat()), datetime(2026, 10, 26, tzinfo=timezone.utc),
                            "unencrypted", records, 0)


def mode(directory):
    value = directory.stat()
    return bundle_mode.SharedMode(str(uuid4()), directory.resolve(), (value.st_dev, value.st_ino),
                                  (value.st_dev, 12345, 100, 20, 30), "a" * 64)


@pytest.fixture
def journal(tmp_path):
    database, directory = tmp_path / "gallery.db", tmp_path / "archives"
    old = fact(directory)
    new = fact(directory, name="new.zip")
    receipts.append(database, old)
    receipts.append(database, new)
    intent = receipts.PendingDeletion(old, str(uuid4()), new.identity.operation_id, mode(directory))
    return database, old, new, intent


def test_missing_state_is_read_only_and_v1_append_remains_exactly_v1(tmp_path):
    absent = tmp_path / "absent" / "gallery.db"
    assert receipts.load_state(absent, GALLERY) == receipts.Journal()
    assert receipts.load(absent, GALLERY) == () and not absent.parent.exists()
    database = tmp_path / "gallery.db"
    one = fact(tmp_path / "archives")
    receipts.append(database, one)
    assert json.loads(receipts.path_for(database).read_bytes()) == {
        "version": 1, "gallery_id": GALLERY, "receipts": [receipts._wire(one)],
    }


def test_full_mode_codec_is_strict_and_does_no_native_or_path_resolution(journal, monkeypatch):
    _database, _old, _new, intent = journal
    encoded = receipts.shared_mode_wire(intent.mode)
    monkeypatch.setattr(bundle_mode, "discover", lambda *_a, **_k: pytest.fail("codec did native discovery"))
    monkeypatch.setattr(Path, "resolve", lambda *_a, **_k: pytest.fail("codec resolved a path"))
    assert receipts.read_shared_mode(encoded) == intent.mode
    for change in ({"extra": "secret"}, {"installation_id": "not-a-uuid"}, {"directory": "relative"},
                   {"directory_identity": [0, 0]}, {"directory_identity": [True, 1]},
                   {"marker_identity": [1, 0, 100, 2, 3]}, {"marker_identity": [1, 2, 0, 3, 4]},
                   {"marker_identity": [1, 2, 3, -1, 4]}, {"digest": "f" * 63}):
        with pytest.raises(receipts.ReceiptError):
            receipts.read_shared_mode({**encoded, **change})


def test_begin_moves_exact_receipt_to_pending_and_survives_read_only_restart(journal, monkeypatch):
    database, old, new, intent = journal
    original_open = Path.open

    def journal_only(path, *args, **kwargs):
        if path.name in {old.name, new.name}:
            pytest.fail("journal transition opened archive bytes")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", journal_only)
    receipts.begin_deletion(database, intent)
    before = receipts.path_for(database).read_bytes()
    state = receipts.load_state(database, GALLERY)
    assert state == receipts.Journal((new,), intent, None, 2)
    assert state.all_receipts == (new, old) and receipts.load(database, GALLERY) == (new,)
    assert receipts.load_state(database, GALLERY) == state
    assert receipts.path_for(database).read_bytes() == before


def test_append_preserves_pending_without_reactivation(journal):
    database, old, new, intent = journal
    receipts.begin_deletion(database, intent)
    later = fact(Path(old.directory), name="later.zip")
    receipts.append(database, later)
    state = receipts.load_state(database, GALLERY)
    assert state.version == 2 and state.receipts == (new, later) and state.pending == intent
    assert old not in receipts.load(database, GALLERY)


@pytest.mark.parametrize("conflict", ["operation", "archive", "destination"])
def test_pending_participates_in_all_append_uniqueness_checks(journal, tmp_path, conflict):
    database, old, _new, intent = journal
    receipts.begin_deletion(database, intent)
    before = receipts.path_for(database).read_bytes()
    if conflict == "operation":
        extra = fact(Path(old.directory), name="another.zip", operation=old.identity.operation_id)
    elif conflict == "archive":
        extra = replace(old, identity=replace(old.identity, operation_id=str(uuid4())))
    else:
        extra = fact(tmp_path / "elsewhere", name="another.zip")
    with pytest.raises(receipts.ReceiptError):
        receipts.append(database, extra)
    assert receipts.path_for(database).read_bytes() == before


@pytest.mark.parametrize("invalid", ["missing_candidate", "missing_success", "empty_success", "foreign_destination", "foreign_directory", "protection"])
def test_begin_requires_matching_nonempty_success_and_exact_candidate(journal, tmp_path, invalid):
    database, old, new, intent = journal
    if invalid == "missing_candidate":
        intent = replace(intent, receipt=replace(old, sha256="0" * 64))
    elif invalid == "missing_success":
        intent = replace(intent, new_success=str(uuid4()))
    else:
        updated = new
        if invalid == "empty_success":
            updated = replace(new, records=0, images=0)
        elif invalid == "foreign_destination":
            updated = replace(new, identity=replace(new.identity, destination_id=str(uuid4())))
        elif invalid == "foreign_directory":
            updated = fact(tmp_path / "foreign", name="new.zip", operation=new.identity.operation_id,
                           destination=str(uuid4()))
        elif invalid == "protection":
            updated = replace(new, name="new.ldbundle", protection="encrypted")
        receipts._store(database, GALLERY, receipts.Journal((old, updated)))
    before = receipts.path_for(database).read_bytes()
    with pytest.raises(receipts.ReceiptError):
        receipts.begin_deletion(database, intent)
    assert receipts.path_for(database).read_bytes() == before


def test_missing_journal_or_already_pending_never_invents_authority(journal, tmp_path):
    database, _old, _new, intent = journal
    absent = tmp_path / "missing.db"
    with pytest.raises(receipts.ReceiptError):
        receipts.begin_deletion(absent, intent)
    assert not receipts.path_for(absent).exists()
    receipts.begin_deletion(database, intent)
    before = receipts.path_for(database).read_bytes()
    with pytest.raises(receipts.ReceiptError):
        receipts.begin_deletion(database, intent)
    assert receipts.path_for(database).read_bytes() == before
    with pytest.raises(receipts.ReceiptError):
        replace(intent, new_success=intent.receipt.identity.operation_id)


@pytest.mark.parametrize("mark", ["not_requested", "refused"])
def test_known_no_mark_can_restore_only_the_exact_pending_receipt(journal, mark):
    database, old, new, intent = journal
    receipts.begin_deletion(database, intent)
    receipts.restore_deletion(database, intent, mark=mark)
    state = receipts.load_state(database, GALLERY)
    assert state.version == 2 and state.receipts == (new, old) and state.pending is None
    assert state.last_retirement == receipts.RetirementSummary(intent, mark)


def test_accepted_normal_close_retires_authority_without_claiming_physical_removal(journal):
    database, old, new, intent = journal
    receipts.begin_deletion(database, intent)
    # This journal-only test simulates its caller's accepted+normal-exit fact.
    # The archive is deliberately NOT deleted, proving no byte/path mutation.
    receipts.complete_deletion(database, intent)
    state = receipts.load_state(database, GALLERY)
    assert state == receipts.Journal((new,), None, receipts.RetirementSummary(intent, "accepted"), 2, (old,))
    assert state.last_retirement.owned_archive_close_confirmed
    assert not state.last_retirement.physical_removal_confirmed
    assert (Path(old.directory) / old.name).exists()
    later = fact(Path(old.directory), name="later.zip")
    receipts.append(database, later)
    assert receipts.load_state(database, GALLERY).last_retirement == state.last_retirement
    assert receipts.load_state(database, GALLERY).retired == (old,)


@pytest.mark.parametrize("conflict", ["operation", "archive"])
def test_retired_receipt_refuses_republication_or_same_name_adoption(journal, conflict):
    database, old, _new, intent = journal
    receipts.begin_deletion(database, intent)
    receipts.complete_deletion(database, intent)
    before = receipts.path_for(database).read_bytes()
    if conflict == "operation":
        extra = replace(old, name="different-name.zip")
    else:
        extra = replace(old, identity=replace(old.identity, operation_id=str(uuid4())))
    with pytest.raises(receipts.ReceiptError):
        receipts.append(database, extra)
    assert receipts.path_for(database).read_bytes() == before
    state = receipts.load_state(database, GALLERY)
    assert old in state.all_receipts and old not in state.receipts


def test_older_retired_operations_survive_later_retirements_and_summary_replacement(journal):
    database, old, new, intent = journal
    receipts.begin_deletion(database, intent)
    receipts.complete_deletion(database, intent)
    latest = fact(Path(old.directory), name="latest.zip")
    receipts.append(database, latest)
    next_intent = receipts.PendingDeletion(new, str(uuid4()), latest.identity.operation_id, intent.mode)
    receipts.begin_deletion(database, next_intent)
    receipts.complete_deletion(database, next_intent)
    state = receipts.load_state(database, GALLERY)
    assert state.retired == (old, new) and state.receipts == (latest,)
    assert state.all_receipts == (latest, old, new)
    assert state.last_retirement.intent == next_intent
    before = receipts.path_for(database).read_bytes()
    with pytest.raises(receipts.ReceiptError):
        receipts.append(database, replace(old, name="retry-old.zip"))
    assert receipts.path_for(database).read_bytes() == before


def test_lifetime_record_cap_includes_retired_proofs_without_automatic_eviction(journal, monkeypatch):
    database, old, new, intent = journal
    monkeypatch.setattr(receipts, "MAX_RECEIPTS", 2)
    receipts.begin_deletion(database, intent)
    receipts.complete_deletion(database, intent)
    before = receipts.path_for(database).read_bytes()
    extra = fact(Path(old.directory), name="exceeds-cap.zip")
    with pytest.raises(receipts.ReceiptError):
        receipts.append(database, extra)
    assert receipts.path_for(database).read_bytes() == before
    assert receipts.load_state(database, GALLERY).all_receipts == (new, old)


def test_byte_cap_includes_pending_and_retired_full_evidence(journal, monkeypatch):
    database, old, _new, intent = journal
    receipts.begin_deletion(database, intent)
    receipts.complete_deletion(database, intent)
    before = receipts.path_for(database).read_bytes()
    monkeypatch.setattr(receipts, "_MAX_BYTES", len(before))
    extra = fact(Path(old.directory), name="extra-proof.zip")
    with pytest.raises(receipts.ReceiptError):
        receipts.append(database, extra)
    assert receipts.path_for(database).read_bytes() == before
    assert receipts.load_state(database, GALLERY).retired == (old,)


@pytest.mark.parametrize("operation", ["restore", "complete"])
@pytest.mark.parametrize("different", ["attempt", "receipt", "mode", "new_success"])
def test_resolution_uses_exact_pending_cas_and_never_replays(journal, operation, different):
    database, old, _new, intent = journal
    receipts.begin_deletion(database, intent)
    changed = {
        "attempt": {"attempt_id": str(uuid4())}, "receipt": {"receipt": replace(old, sha256="0" * 64)},
        "mode": {"mode": replace(intent.mode, digest="0" * 64)}, "new_success": {"new_success": str(uuid4())},
    }[different]
    before = receipts.path_for(database).read_bytes()
    action = receipts.restore_deletion if operation == "restore" else receipts.complete_deletion
    with pytest.raises(receipts.ReceiptError):
        action(database, replace(intent, **changed))
    assert receipts.path_for(database).read_bytes() == before
    action(database, intent)
    after = receipts.path_for(database).read_bytes()
    with pytest.raises(receipts.ReceiptError):
        action(database, intent)
    assert receipts.path_for(database).read_bytes() == after


@pytest.mark.parametrize("phase", ["begin", "restore", "complete"])
@pytest.mark.parametrize("fault", ["fsync", "replace"])
def test_atomic_transition_failures_preserve_previous_bytes_and_pending(journal, monkeypatch, phase, fault):
    database, _old, _new, intent = journal
    if phase != "begin":
        receipts.begin_deletion(database, intent)
    before = receipts.path_for(database).read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("injected journal publication failure")

    monkeypatch.setattr(receipts.os, fault, fail)
    action = {"begin": receipts.begin_deletion, "restore": receipts.restore_deletion, "complete": receipts.complete_deletion}[phase]
    with pytest.raises(OSError, match="injected"):
        action(database, intent)
    assert receipts.path_for(database).read_bytes() == before
    assert receipts.load_state(database, GALLERY).pending == (None if phase == "begin" else intent)
    assert not list(database.parent.glob(".auradream.backup-receipts.*.tmp"))


@pytest.mark.parametrize("damage", ["unknown", "duplicate_active", "missing_success", "summary_claims_gone", "mode_bad_inode"])
def test_malformed_v2_state_has_fixed_error_and_is_never_repaired(journal, damage):
    database, old, _new, intent = journal
    receipts.begin_deletion(database, intent)
    path = receipts.path_for(database)
    payload = json.loads(path.read_bytes())
    if damage == "unknown":
        payload["password"] = "secret never echoed"
    elif damage == "duplicate_active":
        payload["receipts"].append(receipts._wire(old))
    elif damage == "missing_success":
        payload["receipts"] = []
    elif damage == "summary_claims_gone":
        payload["last_retirement"] = receipts._summary_wire(receipts.RetirementSummary(intent, "accepted"))
        payload["last_retirement"]["physical_removal_confirmed"] = True
    elif damage == "mode_bad_inode":
        payload["pending"]["mode"]["directory_identity"][1] = 0
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    with pytest.raises(receipts.ReceiptError) as caught:
        receipts.load_state(database, GALLERY)
    assert str(caught.value) == receipts._ERROR and "secret" not in str(caught.value)
    assert path.read_bytes() == raw
