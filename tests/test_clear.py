"""The destructive path: what it says it will do, and what it leaves alone."""
from __future__ import annotations

import dataclasses
import io
import sqlite3
import sys

import pytest

from backend import clear, gallery_delete, storage


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A gallery that has never been written to - no database file at all."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    settings = dataclasses.replace(
        storage.settings, db_path=tmp_path / "auradream.db", outputs_dir=outputs
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

    monkeypatch.setattr(gallery_delete, "vacuum", locked)
    assert clear.main(["--yes"]) == 0
    captured = capsys.readouterr()
    assert "Deleted 0 record(s), 0 file(s), 1 orphan(s)." in captured.out
    assert "could not compact" in captured.err


def test_a_failure_after_the_deletion_has_its_own_exit_code(fresh, monkeypatch):
    (fresh.outputs_dir / "one.png").write_bytes(b"png")

    def boom(keep_favorites=False):
        raise OSError("disk full")

    monkeypatch.setattr(gallery_delete, "clear_history", boom)
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


def test_bin_empties_the_recycle_bin_alone(fresh, capsys):
    made = _seed(fresh, a=(1, "n"), b=(2, "n"), c=(3, "n"))
    storage.bin_images([made["a"].id, made["b"].id])
    (fresh.outputs_dir / "orphan.png").write_bytes(b"png")  # not swept on this path

    assert clear.main(["--bin", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 images in the recycle bin" in out and "Would delete 2 record(s)" in out
    assert storage.bin_summary()["records"] == 2

    assert clear.main(["--bin", "--yes"]) == 0
    assert "Deleted 2 record(s), 2 file(s)." in capsys.readouterr().out
    assert [r.id for r in storage.list_images()[0]] == [made["c"].id]
    assert not (fresh.outputs_dir / made["a"].filename).exists()
    assert (fresh.outputs_dir / "orphan.png").exists()
    assert clear.main(["--bin", "--yes"]) == 0
    assert "The recycle bin is empty." in capsys.readouterr().out
    # The bin alone: a scope or a keep is another path's.
    with pytest.raises(SystemExit):
        clear.main(["--bin", "--where", "job:n"])
    with pytest.raises(SystemExit):
        clear.main(["--bin", "--keep-favorites"])


def test_bin_refuses_a_bin_that_changed_after_its_preview(fresh, capsys, monkeypatch):
    made = _seed(fresh, a=(1, "n"), b=(2, "n"))
    storage.bin_images([made["a"].id])
    summary = storage.bin_summary

    def then_another(*args, **kwargs):
        found = summary()
        storage.bin_images([made["b"].id])  # binned in the browser meanwhile
        return found

    monkeypatch.setattr(gallery_delete, "bin_summary", then_another)
    assert clear.main(["--bin", "--yes"]) == 1
    assert "Nothing was deleted" in capsys.readouterr().err
    assert all((fresh.outputs_dir / r.filename).exists() for r in made.values())


def test_bin_refuses_a_bin_with_the_same_count_of_other_images(fresh, capsys, monkeypatch):
    # A Restore and a bin of a different image in the browser keep the count:
    # the newly binned image was never in the preview, and must not go.
    made = _seed(fresh, a=(1, "n"), b=(2, "n"))
    storage.bin_images([made["a"].id])
    summary = storage.bin_summary

    def then_swapped(*args, **kwargs):
        found = summary()
        storage.restore_images([made["a"].id])
        storage.bin_images([made["b"].id])
        return found

    monkeypatch.setattr(gallery_delete, "bin_summary", then_swapped)
    assert clear.main(["--bin", "--yes"]) == 1
    assert "other images are in it now than the 1 shown" in capsys.readouterr().err
    assert all((fresh.outputs_dir / r.filename).exists() for r in made.values())


def test_where_refuses_a_row_that_arrived_while_the_prompt_waited(fresh, capsys, monkeypatch):
    # The server or a CLI grid still writing the run being cleared: a third
    # image of the job lands between "Would delete 2" and the 'clear' typed.
    from PIL import Image

    from backend.schemas import GenerateRequest

    made = _seed(fresh, a=(1, "abcdef12"), b=(2, "abcdef12"))

    def then_another(_yes):
        made["c"] = storage.save_image(
            Image.new("RGB", (16, 16)), GenerateRequest(prompt="c prompt", width=256, height=256),
            seed=3, duration_ms=1, model_id="m", job_id="abcdef12",
        )
        return True

    monkeypatch.setattr(clear, "_confirm", then_another)
    assert clear.main(["--where", "job:abcdef12"]) == 1
    captured = capsys.readouterr()
    assert "Would delete 2 record(s)" in captured.out
    assert "3 record(s) match now, not the 2 shown. Nothing was deleted" in captured.err
    assert storage.list_images()[1] == 3
    assert all((fresh.outputs_dir / r.filename).exists() for r in made.values())
    # Asked again, the preview is the new count and the delete goes ahead.
    monkeypatch.setattr(clear, "_confirm", lambda _yes: True)
    assert clear.main(["--where", "job:abcdef12", "--yes"]) == 0
    assert "Deleted 3 record(s), 3 file(s)." in capsys.readouterr().out


def test_the_whole_clear_finishes_a_kept_images_interrupted_save(fresh, capsys):
    # The row committed and the rename did not: the .part is the only copy of
    # a starred image the clear keeps, and is put in place, not swept.
    import os
    import time

    made = _seed(fresh, a=(1, "n"), b=(2, "n"))
    storage.set_favorite(made["a"].id, True)
    png = fresh.outputs_dir / made["a"].filename
    partial = png.with_name(png.name + ".part")
    png.replace(partial)
    # Its thumbnail's partial too: debris, and no second image to finish.
    thumb = storage._thumb_path(png)
    thumb_partial = thumb.with_name(thumb.name + ".part")
    thumb.replace(thumb_partial)
    old = time.time() - 2 * storage.PARTIAL_SETTLE_SECONDS
    os.utime(partial, (old, old))
    os.utime(thumb_partial, (old, old))
    assert storage.summarise()["recoverable_partials"] == 1
    assert storage.summarise()["partial_files"] == 1
    thumb_partial.replace(thumb)

    assert clear.main(["--keep-favorites", "--yes"]) == 0
    assert "Finished 1 interrupted save(s)." in capsys.readouterr().out
    assert png.is_file() and not partial.exists()
    # Before the thumbnail sweep, which takes a thumbnail whose PNG is gone.
    assert thumb.is_file()
    assert [r.id for r in storage.list_images()[0]] == [made["a"].id]


def test_the_whole_clear_and_where_reach_the_recycle_bin(fresh, capsys):
    made = _seed(fresh, a=(1, "n"), b=(2, "n"), c=(3, "m"))
    storage.set_favorite(made["a"].id, True)
    storage.bin_images([made["a"].id, made["c"].id])

    assert clear.main(["--where", "job:n", "--dry-run"]) == 0
    assert "(1 in the recycle bin)" in capsys.readouterr().out
    assert clear.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    # A binned star is no favourite: --keep-favorites would not keep it.
    assert "3 records (0 favorite)" in out and "2 of those are in the recycle bin" in out
    assert clear.main(["--where", "job:n", "--keep-favorites", "--yes"]) == 0
    assert "Deleted 2 record(s), 2 file(s)." in capsys.readouterr().out
    assert clear.main(["--yes"]) == 0
    assert "Deleted 1 record(s), 1 file(s)" in capsys.readouterr().out
    assert storage.summarise()["records"] == 0
