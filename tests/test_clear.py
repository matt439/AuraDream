"""The destructive path: what it says it will do, and what it leaves alone."""
from __future__ import annotations

import dataclasses
import sqlite3

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


def test_the_report_is_printed_even_when_vacuum_fails(fresh, capsys, monkeypatch):
    # VACUUM is an optimisation that raises "database is locked" against a
    # concurrent writer; the deletion had already happened, and the report of
    # it used to be lost behind the traceback with the abort code, 1.
    (fresh.outputs_dir / "one.png").write_bytes(b"png")

    def locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(storage, "vacuum", locked)
    assert clear.main(["--yes"]) == 0
    captured = capsys.readouterr()
    assert "Deleted 0 record(s), 0 file(s), 1 orphan(s)." in captured.out
    assert "could not compact" in captured.err


def test_a_failure_after_the_deletion_has_its_own_exit_code(fresh, monkeypatch):
    (fresh.outputs_dir / "one.png").write_bytes(b"png")

    def boom(keep_favorites=False):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "clear_history", boom)
    # 1 is "aborted, nothing deleted"; this is the opposite case.
    assert clear.main(["--yes"]) == 3


def test_a_stale_partial_is_named_in_the_preview(fresh, capsys):
    import os
    import time

    stale = fresh.outputs_dir / "night.png.part"
    stale.write_bytes(b"power cut")
    old = time.time() - 2 * storage.PARTIAL_SETTLE_SECONDS
    os.utime(stale, (old, old))
    # It is the only thing here, and it must not read as "Nothing to clear".
    assert clear.main(["--dry-run"]) == 0
    assert "1 stale .part file(s)" in capsys.readouterr().out
    assert stale.exists()
