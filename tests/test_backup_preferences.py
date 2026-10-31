"""Pure sidecar-path tests: scratch files only, no app fixtures or backup execution."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend import backup_preferences as preferences


def test_databases_in_one_directory_have_independent_sidecars(tmp_path: Path) -> None:
    first, second, third = (tmp_path / name for name in ("gallery.db", "gallery.sqlite", "other.db"))
    assert len({preferences.path_for(path) for path in (first, second, third)}) == 3
    assert all(path.parent == tmp_path for path in map(preferences.path_for, (first, second, third)))
    assert list(tmp_path.iterdir()) == [], "naming a sidecar creates nothing"


def test_a_database_needs_a_filename() -> None:
    with pytest.raises(ValueError, match="database filename"):
        preferences.path_for(Path(""))


def test_a_sidecar_named_before_the_rename_is_used_in_place(tmp_path: Path) -> None:
    database = tmp_path / "gallery.db"
    legacy = tmp_path / "local-diffusion.backup-policy-gallery.db.json"
    assert preferences.path_for(database) == tmp_path / "auradream.backup-policy-gallery.db.json"
    legacy.write_bytes(b"{}")
    assert preferences.path_for(database) == legacy


def test_a_current_sidecar_wins_over_a_legacy_one(tmp_path: Path) -> None:
    database = tmp_path / "gallery.db"
    current = tmp_path / "auradream.backup-policy-gallery.db.json"
    legacy = tmp_path / "local-diffusion.backup-policy-gallery.db.json"
    current.write_bytes(b"{}")
    legacy.write_bytes(b"stale")
    assert preferences.path_for(database) == current
    assert legacy.read_bytes() == b"stale"
