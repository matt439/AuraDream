"""Strict new activation envelopes; every path here is task-owned scratch."""
from __future__ import annotations

import json
from datetime import timedelta
from uuid import uuid4

import pytest

from backend import backup_preferences, backup_receipts, backup_settings
from backend.backup_policy import BackupPolicy


def enable(database, directory, **changes):
    values = dict(revision=backup_settings.load(database).revision, enabled=True,
                  directory=directory, minimum_age_hours=24, keep_last=3, allow_unencrypted=True)
    values.update(changes)
    return backup_settings.configure(database, **values)


def test_missing_and_disabled_loads_create_nothing_and_generate_no_identifiers(tmp_path, monkeypatch):
    db = tmp_path / "absent" / "gallery.db"
    monkeypatch.setattr(backup_settings, "uuid4", lambda: pytest.fail("reading must not invent an identity"))
    loaded = backup_settings.load(db)
    assert loaded.revision == "missing" and not loaded.settings.enabled
    assert not db.parent.exists()
    db = tmp_path / "gallery.db"
    disabled = backup_settings.configure(db, revision="missing", enabled=False, directory=tmp_path / "archives")
    assert disabled.settings == backup_settings.BackupSettings()
    assert backup_settings.load(db) == disabled
    assert not (tmp_path / "archives").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_every_legacy_file_is_preserved_and_inactive_until_new_explicit_consent(tmp_path, enabled):
    db = tmp_path / "gallery.db"
    backup_preferences.save(db, BackupPolicy(enabled, timedelta(hours=1), 2, "unencrypted"))
    path = backup_preferences.path_for(db)
    before = path.read_bytes()
    loaded = backup_settings.load(db)
    assert not loaded.settings.enabled and loaded.requires_confirmation
    assert loaded.revision and path.read_bytes() == before
    with pytest.raises(backup_settings.SettingsError, match="Explicitly allow"):
        enable(db, tmp_path / "archives", allow_unencrypted=False)
    assert path.read_bytes() == before
    chosen = enable(db, tmp_path / "archives")
    assert chosen.settings.enabled and not chosen.requires_confirmation
    assert json.loads(path.read_text())["version"] == 2


def test_explicit_values_stable_scope_and_destination_changes(tmp_path):
    db, directory = tmp_path / "gallery.db", tmp_path / "archives"
    first = enable(db, directory)
    second = enable(db, directory, minimum_age_hours=48, keep_last=4)
    assert second.settings.gallery_id == first.settings.gallery_id
    assert second.settings.destination_id == first.settings.destination_id
    disabled = backup_settings.configure(db, revision=second.revision, enabled=False, directory=directory)
    assert not disabled.settings.enabled and disabled.settings.gallery_id == first.settings.gallery_id
    changed = enable(db, tmp_path / "another-destination")
    assert changed.settings.gallery_id == first.settings.gallery_id
    assert changed.settings.destination_id != first.settings.destination_id
    assert not directory.exists() and not (tmp_path / "another-destination").exists()


@pytest.mark.parametrize("changes", [
    {"minimum_age_hours": None}, {"minimum_age_hours": True}, {"minimum_age_hours": 0},
    {"minimum_age_hours": 1.2}, {"minimum_age_hours": backup_settings.MAX_AGE_HOURS + 1},
    {"keep_last": None}, {"keep_last": True}, {"keep_last": 0}, {"keep_last": 10001},
    {"allow_unencrypted": False},
])
def test_enable_never_invents_missing_or_invalid_policy_choices(tmp_path, changes):
    db = tmp_path / "gallery.db"
    with pytest.raises(backup_settings.SettingsError):
        enable(db, tmp_path / "archives", **changes)
    assert not backup_preferences.path_for(db).exists()


@pytest.mark.parametrize("change", [
    {"allow_unencrypted": False}, {"protection": "encrypted"}, {"gallery_id": None},
    {"destination_id": "not-a-uuid"}, {"minimum_age_hours": True}, {"keep_last": 0},
    {"extra": "unknown"},
])
def test_invalid_v2_never_activates_and_is_not_rewritten(tmp_path, change):
    db = tmp_path / "gallery.db"
    enable(db, tmp_path / "archives")
    path = backup_preferences.path_for(db)
    wire = json.loads(path.read_text())
    wire.update(change)
    before = json.dumps(wire).encode()
    path.write_bytes(before)
    loaded = backup_settings.load(db)
    assert not loaded.settings.enabled and loaded.warning and loaded.requires_confirmation
    assert path.read_bytes() == before


def test_future_oversized_and_duplicate_formats_refuse_implicit_repair(tmp_path):
    db = tmp_path / "gallery.db"
    path = backup_preferences.path_for(db)
    for data in (b'{"version":3}', b"x" * (65536 + 1)):
        path.write_bytes(data)
        loaded = backup_settings.load(db)
        assert loaded.revision is None and not loaded.settings.enabled
        with pytest.raises(backup_settings.SettingsError, match="Repair"):
            enable(db, tmp_path / "archives")
        assert path.read_bytes() == data
    path.write_bytes(b'{"version":2,"version":2}')
    assert backup_settings.load(db).warning


def test_stale_revision_and_failed_publication_preserve_saved_bytes(tmp_path, monkeypatch):
    db = tmp_path / "gallery.db"
    first = enable(db, tmp_path / "archives")
    path = backup_preferences.path_for(db)
    before = path.read_bytes()
    with pytest.raises(backup_settings.SettingsConflict):
        enable(db, tmp_path / "archives", revision="missing")
    assert path.read_bytes() == before

    def fail(*_args):
        raise OSError("injected atomic publication failure")

    monkeypatch.setattr(backup_settings.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        backup_settings.configure(db, revision=first.revision, enabled=False, directory=tmp_path / "archives")
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".local-diffusion.backup-policy.*.tmp")) == []


def test_existing_journal_is_not_adopted_when_scope_settings_are_lost(tmp_path):
    db = tmp_path / "gallery.db"
    journal = backup_receipts.path_for(db)
    data = json.dumps({"version": 1, "gallery_id": str(uuid4()), "receipts": []}).encode()
    journal.write_bytes(data)
    with pytest.raises(backup_settings.SettingsError, match="journal needs repair"):
        enable(db, tmp_path / "archives")
    assert journal.read_bytes() == data
    assert not backup_preferences.path_for(db).exists()


def test_serialized_settings_must_fit_the_same_limit_as_the_reader(tmp_path, monkeypatch):
    db = tmp_path / "gallery.db"
    directory = tmp_path / ("é" * 12000)
    # No directory is created or probed: exercise JSON expansion, not OS limits.
    monkeypatch.setattr(type(directory), "resolve", lambda path: path)
    with pytest.raises(backup_settings.SettingsError, match="file size"):
        enable(db, directory)
    assert not backup_preferences.path_for(db).exists()
