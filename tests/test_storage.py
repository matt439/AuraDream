"""The gallery index: migrations, the search filters, and what resume reads."""
from __future__ import annotations

import dataclasses
import sqlite3

import pytest
from PIL import Image

from backend import storage
from backend.schemas import GenerateRequest

# The table as it was written before runs were grouped. Deliberately a literal
# rather than _SCHEMA with a line removed: it has to keep meaning "an old
# database" however the current schema moves on.
OLD_SCHEMA = """
CREATE TABLE images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL UNIQUE,
    prompt          TEXT    NOT NULL,
    negative_prompt TEXT,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    steps           INTEGER NOT NULL,
    guidance_scale  REAL    NOT NULL,
    mu              REAL,
    seed            INTEGER NOT NULL,
    model_id        TEXT    NOT NULL,
    duration_ms     INTEGER NOT NULL,
    favorite        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL
);
CREATE VIRTUAL TABLE images_fts USING fts5(prompt, content='images', content_rowid='id');
CREATE TRIGGER images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, prompt) VALUES (new.id, new.prompt);
END;
"""


def keep(store, prompt="a fox in the snow", **overrides):
    """Write one image, returning its record."""
    fields = {"prompt": prompt, "width": 256, "height": 256, "steps": 8, "guidance_scale": 0.0}
    job_id = overrides.pop("job_id", None)
    run_id = overrides.pop("run_id", None)
    seed = overrides.pop("seed", 1)
    fields.update(overrides)
    # The row records the *image's* size, not the request's, so the two have to
    # agree for anything that matches on dimensions - resume included.
    return store.save_image(
        Image.new("RGB", (fields["width"], fields["height"])),
        GenerateRequest(**fields),
        seed=seed,
        duration_ms=1000,
        model_id=fields.pop("model_id", "krea/Krea-2-Turbo"),
        job_id=job_id,
        run_id=run_id,
    )


# ------------------------------------------------------------------ migration


def test_a_database_from_before_job_ids_gains_the_column(tmp_path, monkeypatch):
    # `CREATE TABLE IF NOT EXISTS` leaves an existing table alone, so without a
    # migration every insert into an older gallery fails on the missing column.
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(OLD_SCHEMA)
    conn.close()

    monkeypatch.setattr(
        storage,
        "settings",
        dataclasses.replace(storage.settings, db_path=db, outputs_dir=outputs),
    )
    monkeypatch.setattr(storage, "_migrated", False)  # another test may have run first

    record = keep(storage, job_id="abc123", run_id="run001")
    assert (record.job_id, record.run_id) == ("abc123", "run001")
    items, total = storage.list_images()
    assert total == 1 and (items[0].job_id, items[0].run_id) == ("abc123", "run001")


# -------------------------------------------------------------------- filters


def test_a_filter_term_is_a_predicate_not_a_word(store):
    keep(store, seed=42, steps=8)
    keep(store, seed=43, steps=12)
    assert [r.seed for r in store.list_images(search="seed:42")[0]] == [42]
    assert store.list_images(search="steps:12")[1] == 1


def test_filters_compare_and_span(store):
    for scale in (0.0, 0.5, 1.0, 2.0):
        keep(store, guidance_scale=scale, seed=int(scale * 10))
    assert store.list_images(search="cfg:<=0.5")[1] == 2
    assert store.list_images(search="cfg:0.5-1")[1] == 2
    assert store.list_images(search="cfg:>1")[1] == 1


def test_mu_none_is_not_mu_zero(store):
    keep(store, mu=None, seed=1)
    keep(store, mu=0.0, seed=2)
    assert [r.seed for r in store.list_images(search="mu:none")[0]] == [1]
    assert [r.seed for r in store.list_images(search="mu:0")[0]] == [2]


def test_filters_and_free_text_combine(store):
    keep(store, prompt="a fox in the snow", seed=1)
    keep(store, prompt="a lighthouse in fog", seed=2)
    assert store.list_images(search="fox seed:1")[1] == 1
    assert store.list_images(search="fox seed:2")[1] == 0


def test_a_colon_in_a_prompt_is_still_a_prompt(store):
    # Only the keys we know become filters; anything else stays free text.
    keep(store, prompt="portrait: side lit")
    assert store.list_images(search="portrait:")[1] == 1


def test_a_filter_that_is_not_a_number_says_so(store):
    with pytest.raises(ValueError, match="not a number"):
        store.list_images(search="seed:tuesday")


def test_the_job_filter_selects_one_run(store):
    keep(store, seed=1, job_id="night01")
    keep(store, seed=2, job_id="night01")
    keep(store, seed=3, job_id="night02")
    assert store.list_images(search="job:night01")[1] == 2


def test_the_run_filter_joins_the_jobs_that_continued_it(store):
    # A resumed grid lands under a second job id; the run id is the grid's own.
    keep(store, seed=1, job_id="night01", run_id="grid-a")
    keep(store, seed=2, job_id="night02", run_id="grid-a")
    keep(store, seed=3, job_id="night03", run_id="grid-b")
    assert store.list_images(search="run:grid-a")[1] == 2
    assert store.list_images(search="job:night01")[1] == 1


# --------------------------------------------------------------------- resume


def test_completed_cells_keys_on_the_settings_not_the_run(store):
    keep(store, seed=7, steps=8, guidance_scale=0.5, mu=1.15, job_id="first")
    done = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo")
    assert ("a fox in the snow", None, 8, 0.5, 1.15, 7) in done
    # A different prompt, size or checkpoint is a different image entirely.
    assert store.completed_cells(["something else"], 256, 256, "krea/Krea-2-Turbo") == set()
    assert store.completed_cells(["a fox in the snow"], 512, 512, "krea/Krea-2-Turbo") == set()
    assert store.completed_cells(["a fox in the snow"], 256, 256, "other/model") == set()


def test_completed_cells_answers_for_every_prompt_of_a_grid(store):
    # A prompt grid resumes on several prompts at once, and each row has to come
    # back under the prompt that made it or the wrong cells get skipped.
    keep(store, prompt="a big fox", seed=1)
    keep(store, prompt="a small fox", seed=2)
    done = store.completed_cells(["a big fox", "a small fox"], 256, 256, "krea/Krea-2-Turbo")
    assert {key[0] for key in done} == {"a big fox", "a small fox"}
    assert ("a big fox", None, 8, 0.0, None, 2) not in done


def test_a_mu_free_cell_comes_back_as_none(store):
    keep(store, seed=3, mu=None)
    done = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo")
    assert ("a fox in the snow", None, 8, 0.0, None, 3) in done
