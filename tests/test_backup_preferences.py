"""Pure sidecar tests: scratch files only, no app fixtures or backup execution."""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest

from backend import backup_preferences as preferences
from backend.backup_policy import MAX_KEEP_LAST, BackupPolicy, Protection


def _wire(**changes: object) -> bytes:
    values: dict[str, object] = {
        "version": 1, "enabled": True, "minimum_age_us": 1,
        "keep_last": 2, "protection": "unencrypted",
    }
    values.update(changes)
    return json.dumps(values).encode("utf-8")


def test_missing_load_is_disabled_and_creates_nothing(tmp_path: Path) -> None:
    database = tmp_path / "not-created" / "gallery.db"
    loaded = preferences.load(database)
    assert loaded.policy == BackupPolicy()
    assert loaded.warning is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("age", [
    timedelta(microseconds=1), timedelta(days=987654, seconds=123, microseconds=456789),
    timedelta.max,
])
@pytest.mark.parametrize("protection", ["unencrypted", "encrypted"])
def test_explicit_policy_round_trip_preserves_exact_duration(
    tmp_path: Path, age: timedelta, protection: Protection,
) -> None:
    database = tmp_path / "gallery.db"
    policy = BackupPolicy(True, age, 7, protection)
    preferences.save(database, policy)
    assert preferences.load(database) == preferences.LoadedPolicy(policy)
    wire = json.loads(preferences.path_for(database).read_bytes())
    assert wire == {
        "version": 1, "enabled": True,
        "minimum_age_us": (age.days * 86400 + age.seconds) * 1_000_000 + age.microseconds,
        "keep_last": 7, "protection": protection,
    }
    assert type(wire["minimum_age_us"]) is int


@pytest.mark.parametrize("policy", [
    BackupPolicy(), BackupPolicy(False, timedelta(microseconds=11), 3, "encrypted"),
])
def test_disabled_policy_round_trip_chooses_no_defaults(
    tmp_path: Path, policy: BackupPolicy,
) -> None:
    database = tmp_path / "nested" / "gallery.db"
    preferences.save(database, policy)
    assert preferences.load(database) == preferences.LoadedPolicy(policy)
    if policy == BackupPolicy():
        assert json.loads(preferences.path_for(database).read_bytes()) == {
            "version": 1, "enabled": False, "minimum_age_us": None,
            "keep_last": None, "protection": None,
        }


def test_databases_in_one_directory_have_independent_sidecars(tmp_path: Path) -> None:
    first, second, third = (tmp_path / name for name in ("gallery.db", "gallery.sqlite", "other.db"))
    assert len({preferences.path_for(path) for path in (first, second, third)}) == 3
    policy = BackupPolicy(True, timedelta(microseconds=3), 1, "encrypted")
    preferences.save(first, policy)
    preferences.save(second, BackupPolicy())
    assert preferences.load(first).policy == policy
    assert preferences.load(second).policy == BackupPolicy()
    assert preferences.load(third).policy == BackupPolicy()
    assert not first.exists() and not second.exists() and not third.exists()


@pytest.mark.parametrize("keep_last", [1, MAX_KEEP_LAST - 1, MAX_KEEP_LAST])
@pytest.mark.parametrize("enabled", [False, True])
def test_retention_bounds_round_trip_without_choosing_a_default(
    tmp_path: Path, keep_last: int, enabled: bool,
) -> None:
    database = tmp_path / "gallery.db"
    policy = BackupPolicy(enabled, timedelta(microseconds=1), keep_last, "unencrypted")
    preferences.save(database, policy)
    assert preferences.load(database) == preferences.LoadedPolicy(policy)


@pytest.mark.parametrize("raw", [
    b"", b"{", b"null", b"[]", b"true", b"12", b'"settings"', b"{}", b"\xff",
    _wire(version=0), _wire(version=2), _wire(version=True), _wire(version=1.0),
    _wire(version="1"), _wire(enabled=1), _wire(enabled="false"), _wire(enabled=None),
    _wire(minimum_age_us=True), _wire(minimum_age_us=1.0), _wire(minimum_age_us="1"),
    _wire(minimum_age_us=0), _wire(minimum_age_us=-1), _wire(minimum_age_us=10**30),
    _wire(minimum_age_us=None), _wire(keep_last=True), _wire(keep_last=2.0),
    _wire(keep_last="2"), _wire(keep_last=0), _wire(keep_last=-1), _wire(keep_last=None),
    _wire(keep_last=MAX_KEEP_LAST + 1), _wire(keep_last=10**100),
    _wire(enabled=False, keep_last=MAX_KEEP_LAST + 1), _wire(enabled=False, keep_last=10**100),
    _wire(protection=None), _wire(protection="guess"), _wire(protection=[]),
    _wire(password="a-secret-that-must-not-appear"), _wire(minimum_age_us=float("nan")),
    _wire(minimum_age_us=float("inf")), _wire(minimum_age_us=float("-inf")),
    _wire().replace(b'"enabled": true', b'"enabled": false, "enabled": true'),
    _wire(extra={"duplicate": 1}).replace(b'"duplicate": 1', b'"duplicate": 1, "duplicate": 2'),
    _wire().replace(b'"keep_last": 2, ', b""),
    b"[" * 2000 + b"]" * 2000,
    b" " * (preferences._MAX_BYTES + 1),
], ids=lambda raw: f"{len(raw)}-bytes")
def test_unusable_settings_are_fixed_warning_disabled_and_untouched(
    tmp_path: Path, raw: bytes,
) -> None:
    database = tmp_path / "gallery.db"
    path = preferences.path_for(database)
    path.write_bytes(raw)
    loaded = preferences.load(database)
    assert loaded == preferences.LoadedPolicy(BackupPolicy(), preferences._WARNING)
    assert "a-secret-that-must-not-appear" not in repr(loaded)
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


def test_read_limit_is_applied_before_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[int] = []

    class Reader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            assert size == preferences._MAX_BYTES + 1
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Reader(b" " * (preferences._MAX_BYTES + 2)))
    assert preferences.load(tmp_path / "gallery.db").warning == preferences._WARNING
    assert requested == [preferences._MAX_BYTES + 1]


def test_exact_read_limit_accepts_complete_settings(tmp_path: Path) -> None:
    database = tmp_path / "gallery.db"
    raw = _wire()
    preferences.path_for(database).write_bytes(raw + b" " * (preferences._MAX_BYTES - len(raw)))
    assert preferences.load(database) == preferences.LoadedPolicy(
        BackupPolicy(True, timedelta(microseconds=1), 2, "unencrypted"),
    )


def test_unreadable_settings_use_the_same_nonsecret_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("secret-path-and-credential")

    monkeypatch.setattr(Path, "open", denied)
    loaded = preferences.load(tmp_path / "gallery.db")
    assert loaded == preferences.LoadedPolicy(BackupPolicy(), preferences._WARNING)
    assert "secret-path-and-credential" not in repr(loaded)


def test_success_replaces_only_its_target_and_leaves_no_temporary(tmp_path: Path) -> None:
    database = tmp_path / "gallery.db"
    path = preferences.path_for(database)
    path.write_bytes(b"old settings")
    unrelated = tmp_path / ".local-diffusion.backup-policy.someone-else.tmp"
    unrelated.write_bytes(b"unrelated")
    preferences.save(database, BackupPolicy())
    assert preferences.load(database) == preferences.LoadedPolicy(BackupPolicy())
    assert unrelated.read_bytes() == b"unrelated"
    assert set(tmp_path.iterdir()) == {path, unrelated}


@pytest.mark.parametrize("failure", ["open", "write", "fsync", "replace"])
def test_failed_save_preserves_previous_bytes_and_cleans_only_owned_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    database = tmp_path / "gallery.db"
    old_policy = BackupPolicy(True, timedelta(microseconds=1), 2, "unencrypted")
    preferences.save(database, old_policy)
    path = preferences.path_for(database)
    before = path.read_bytes()
    unrelated = tmp_path / ".local-diffusion.backup-policy.someone-else.tmp"
    unrelated.write_bytes(b"unrelated")

    def fail(*_args: object, **_kwargs: object) -> None:
        assert path.read_bytes() == before
        raise OSError("injected write failure")

    if failure == "open":
        monkeypatch.setattr(preferences.tempfile, "NamedTemporaryFile", fail)
    elif failure == "write":
        original = preferences.tempfile.NamedTemporaryFile

        @contextmanager
        def failed_writer(*args: object, **kwargs: object):
            with original(*args, **kwargs) as stream:
                monkeypatch.setattr(stream, "write", fail)
                yield stream

        monkeypatch.setattr(preferences.tempfile, "NamedTemporaryFile", failed_writer)
    else:
        monkeypatch.setattr(preferences.os, failure, fail)
    with pytest.raises(OSError, match="injected write failure"):
        preferences.save(database, BackupPolicy())
    assert path.read_bytes() == before
    assert preferences.load(database).policy == old_policy
    assert unrelated.read_bytes() == b"unrelated"
    assert set(tmp_path.iterdir()) == {path, unrelated}


def test_save_revalidates_before_any_filesystem_change(tmp_path: Path) -> None:
    database = tmp_path / "not-created" / "gallery.db"
    invalid = BackupPolicy()
    object.__setattr__(invalid, "enabled", True)
    with pytest.raises(ValueError, match="explicit age"):
        preferences.save(database, invalid)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("keep_last", [MAX_KEEP_LAST + 1, 10**100])
@pytest.mark.parametrize("enabled", [False, True])
def test_save_revalidates_shared_retention_ceiling_without_replacing_previous_bytes(
    tmp_path: Path, keep_last: int, enabled: bool,
) -> None:
    database = tmp_path / "gallery.db"
    good = BackupPolicy(enabled, timedelta(microseconds=1), MAX_KEEP_LAST, "unencrypted")
    preferences.save(database, good)
    path = preferences.path_for(database)
    before = path.read_bytes()
    invalid = BackupPolicy(enabled, timedelta(microseconds=1), 2, "unencrypted")
    object.__setattr__(invalid, "keep_last", keep_last)
    with pytest.raises(ValueError, match=f"from 1 to {MAX_KEEP_LAST}"):
        preferences.save(database, invalid)
    assert path.read_bytes() == before
    assert preferences.load(database) == preferences.LoadedPolicy(good)
    assert list(tmp_path.iterdir()) == [path]
