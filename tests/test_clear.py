"""The destructive path: what it says it will do, and what it leaves alone."""
from __future__ import annotations

import dataclasses
import io
import sqlite3
import sys

import pytest

from backend import clear, storage


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A gallery that has never been written to - no database file at all."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    settings = dataclasses.replace(
        storage.settings, db_path=tmp_path / "local-diffusion.db", outputs_dir=outputs
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


def _seed(fresh, **rows):
    from PIL import Image

    from backend.schemas import GenerateRequest

    storage.init_db()
    made = {}
    for name, (seed, job) in rows.items():
        made[name] = storage.save_image(
            Image.new("RGB", (16, 16)),
            GenerateRequest(prompt=f"{name} prompt", width=256, height=256),
            seed=seed, duration_ms=1, model_id="m", job_id=job,
        )
    return made


def test_where_deletes_only_what_the_terms_select(fresh, capsys):
    # A 200-cell grid with a typo could only be removed cell by cell or with
    # the whole gallery; the filtered path takes the search grammar.
    made = _seed(fresh, a=(1, "night01"), b=(2, "night01"), c=(3, "night02"))
    (fresh.outputs_dir / "orphan.png").write_bytes(b"png")  # not swept on this path

    assert clear.main(["--where", "job:night01", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 records match" in out and "jobs: night01" in out and "a prompt" in out
    assert storage.list_images()[1] == 3

    assert clear.main(["--where", "job:night01", "--yes"]) == 0
    assert "Deleted 2 record(s), 2 file(s)." in capsys.readouterr().out
    remaining = storage.list_images()[0]
    assert [r.id for r in remaining] == [made["c"].id]
    assert (fresh.outputs_dir / made["c"].filename).exists()
    assert not (fresh.outputs_dir / made["a"].filename).exists()
    assert (fresh.outputs_dir / "orphan.png").exists()


def test_where_keeps_favorites_and_refuses_free_text(fresh, capsys):
    made = _seed(fresh, a=(1, "n"), b=(2, "n"))
    storage.set_favorite(made["a"].id, True)
    assert clear.main(["--where", "job:n", "--keep-favorites", "--yes"]) == 0
    assert [r.id for r in storage.list_images()[0]] == [made["a"].id]
    # Free text would go through the LIKE fallback, whose meaning changes on an
    # odd quote; a destructive filter is not confirmed blind.
    assert clear.main(["--where", "a prompt", "--yes"]) == 2
    assert "free text" in capsys.readouterr().err
    assert clear.main(["--where", "seed:999", "--yes"]) == 0
    assert "Nothing matches" in capsys.readouterr().out
    # A blank scope is the whole gallery, which is `clear` without --where;
    # storage answers the question for the browser, the CLI refuses to ask it.
    assert clear.main(["--where", " ", "--keep-favorites", "--yes"]) == 2
    assert "filter is required" in capsys.readouterr().err
    assert [r.id for r in storage.list_images()[0]] == [made["a"].id]


def test_keep_favorites_keeps_what_a_favorite_was_made_from(fresh, capsys):
    # The browser's Delete unstarred and the CLI share `delete_matching`, so
    # a starred refine pass keeps its source on both paths - and the CLI's
    # count says so rather than promising one more deletion than happens.
    from PIL import Image

    from backend.schemas import GenerateRequest

    made = _seed(fresh, source=(1, "n"), loose=(2, "n"))
    refined = storage.save_image(
        Image.new("RGB", (16, 16)),
        GenerateRequest(prompt="pass", width=256, height=256, refine=made["source"].id, strength=0.5),
        seed=3, duration_ms=1, model_id="m", job_id="later",
    )
    storage.set_favorite(refined.id, True)

    assert clear.main(["--where", "job:n", "--keep-favorites", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Would delete 1 record(s) and their files, keeping 0 starred and 1 image(s) a starred one was made from." in out
    assert clear.main(["--where", "job:n", "--keep-favorites", "--yes"]) == 0
    assert "Deleted 1 record(s)" in capsys.readouterr().out
    assert sorted(r.id for r in storage.list_images()[0]) == [made["source"].id, refined.id]

    # The whole-gallery path keeps it as well.
    storage.save_image(
        Image.new("RGB", (16, 16)), GenerateRequest(prompt="more", width=256, height=256),
        seed=4, duration_ms=1, model_id="m", job_id="n",
    )
    assert clear.main(["--keep-favorites", "--yes"]) == 0
    assert "keeping 1 starred and 1 image(s) a starred one was made from." in capsys.readouterr().out
    assert sorted(r.id for r in storage.list_images()[0]) == [made["source"].id, refined.id]
    assert (fresh.outputs_dir / made["source"].filename).exists()


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


def test_the_preview_writes_utf8_where_its_output_is_redirected(fresh, monkeypatch):
    # `> clear.txt` on Windows: stdout is the locale's code page, strict, and a
    # CJK or emoji prompt in the preview was a UnicodeEncodeError - after the
    # count, before the confirmation, with nothing deleted.
    from PIL import Image

    from backend.schemas import GenerateRequest

    storage.init_db()
    storage.save_image(
        Image.new("RGB", (16, 16)), GenerateRequest(prompt="一只狐狸 🦊", width=256, height=256),
        seed=1, duration_ms=1, model_id="m", job_id="night01",
    )
    redirected = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(redirected, encoding="cp1252"))
    assert clear.main(["--where", "job:night01", "--dry-run"]) == 0
    sys.stdout.flush()
    assert "- 一只狐狸 🦊" in redirected.getvalue().decode("utf-8")
