"""`python -m backend.stats`: its scopes and refusals, on a throwaway gallery.

The keep-rate arithmetic itself is gallery_reports' and tested there; these
hold the command's own paths: which scope it reads, and each exit code.
"""
from __future__ import annotations

import dataclasses

import pytest

from backend import stats, storage


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A gallery that has never been written to - no database file at all."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    settings = dataclasses.replace(
        storage.settings, db_path=tmp_path / "auradream.db", outputs_dir=outputs
    )
    monkeypatch.setattr(storage, "settings", settings)
    monkeypatch.setattr(stats, "settings", settings)
    return settings


def _seed(count: int, *, job: str = "job-1", run: str = "run-1", steps: int = 28, starred: int = 0):
    from PIL import Image

    from backend.schemas import GenerateRequest

    storage.init_db()
    made = []
    for i in range(count):
        made.append(storage.save_image(
            Image.new("RGB", (16, 16)),
            GenerateRequest(prompt="a fox", width=256, height=256, steps=steps),
            seed=i, duration_ms=1, model_id="m", job_id=job, run_id=run,
        ))
    for record in made[:starred]:
        storage.set_favorite(record.id, True)
    return made


def test_a_missing_gallery_is_refused_without_creating_one(fresh, capsys):
    assert stats.main([]) == 1
    assert "No gallery yet." in capsys.readouterr().err
    assert not fresh.db_path.exists()


def test_a_gallery_without_runs_names_all(fresh, capsys):
    storage.init_db()

    assert stats.main([]) == 1
    assert "--all" in capsys.readouterr().err


def test_the_default_scope_is_the_newest_run(fresh, capsys):
    _seed(5, run="older", job="j1", steps=20)
    _seed(5, run="newer", job="j2", steps=30, starred=2)

    assert stats.main([]) == 0
    out = capsys.readouterr().out
    assert "the newest run, newer (5 images, 2 starred)" in out
    assert "40%" in out


@pytest.mark.parametrize(("argv", "label"), [
    (["--run", "run-1"], "scope    : run run-1"),
    (["--job", "job-1"], "scope    : job job-1"),
    (["--where", "model:m"], "scope    : model:m"),
    (["--all"], "scope    : the whole gallery"),
])
def test_each_scope_is_named_in_the_report(fresh, capsys, argv, label):
    _seed(5, starred=1)

    assert stats.main([*argv, "--min", "1"]) == 0
    out = capsys.readouterr().out
    assert label in out
    assert "20%" in out


def test_an_unknown_grouping_is_refused(fresh, capsys):
    _seed(1)

    assert stats.main(["--all", "--by", "colour"]) == 2
    assert "cannot group by colour" in capsys.readouterr().err


def test_too_few_images_per_setting_says_so(fresh, capsys):
    _seed(2)

    assert stats.main(["--all", "--min", "5"]) == 0
    out = capsys.readouterr().out
    assert "at least 5 image(s) each" in out
    assert "Nothing to compare" in out


def test_min_below_one_is_read_as_one(fresh, capsys):
    _seed(1)

    assert stats.main(["--all", "--min", "0"]) == 0
    assert "at least 1 image(s) each" in capsys.readouterr().out
