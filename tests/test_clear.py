"""The destructive path: what it says it will do, and what it leaves alone."""
from __future__ import annotations

import dataclasses

import pytest

from backend import clear, storage


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A gallery that has never been written to - no database file at all."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    settings = dataclasses.replace(
        storage.settings, db_path=tmp_path / "krea.db", outputs_dir=outputs
    )
    monkeypatch.setattr(storage, "settings", settings)
    monkeypatch.setattr(clear, "settings", settings)
    return settings


def test_a_dry_run_creates_nothing(fresh, capsys):
    (fresh.outputs_dir / "left-over.png").write_bytes(b"png")

    assert clear.main(["--dry-run"]) == 0
    # summarise() used to connect, and connecting writes the whole schema.
    assert not fresh.db_path.exists()
    assert (fresh.outputs_dir / "left-over.png").exists()


def test_the_preview_counts_the_orphans_it_will_delete(fresh, capsys):
    for name in ("one.png", "two.png"):
        (fresh.outputs_dir / name).write_bytes(b"png")

    clear.main(["--dry-run"])
    out = capsys.readouterr().out
    # "Would delete 0 record(s)" on its own read as "nothing will happen".
    assert "2 untracked PNG(s)" in out


def test_a_real_run_removes_them(fresh, capsys):
    (fresh.outputs_dir / "one.png").write_bytes(b"png")

    assert clear.main(["--yes"]) == 0
    assert list(fresh.outputs_dir.glob("*.png")) == []
    assert "1 orphan(s)" in capsys.readouterr().out
