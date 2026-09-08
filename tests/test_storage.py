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
    quantization = overrides.pop("quantization", None)
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
        quantization=quantization,
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
    # No reset of the once-per-process flag: another test has migrated *its*
    # database already, and a file swapped under a running server - restored
    # from a backup, say - has to be checked again regardless.

    record = keep(storage, job_id="abc123", run_id="run001")
    assert (record.job_id, record.run_id) == ("abc123", "run001")
    items, total = storage.list_images()
    assert total == 1 and (items[0].job_id, items[0].run_id) == ("abc123", "run001")


# ---------------------------------------------------------------- the rename


def test_a_transient_rename_failure_is_retried(store, monkeypatch):
    from pathlib import Path

    real = Path.replace
    failures = {"left": 2}

    def flaky(self, target):
        if self.suffix == ".part" and failures["left"]:
            failures["left"] -= 1
            raise PermissionError(32, "sharing violation")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(storage.time, "sleep", lambda s: None)
    record = keep(store)
    assert (store.settings.outputs_dir / record.filename).exists()
    assert list(store.settings.outputs_dir.glob("*.part")) == []


def test_a_committed_image_survives_a_rename_that_keeps_failing(store, monkeypatch):
    # The row is durable once the INSERT commits; deleting the partial then
    # threw away the only copy of an image the database names, leaving a card
    # that 404s forever. The partial stays; the error still propagates.
    from pathlib import Path

    def always(self, target):
        if self.suffix == ".part":
            raise PermissionError(32, "sharing violation")
        return Path.rename(self, target)

    monkeypatch.setattr(Path, "replace", always)
    monkeypatch.setattr(storage.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        keep(store)
    assert store.list_images()[1] == 1
    assert len(list(store.settings.outputs_dir.glob("*.png.part"))) == 1


def test_stale_partials_are_counted_and_swept_but_fresh_ones_are_not(store):
    import os
    import time

    fresh = store.settings.outputs_dir / "fresh.png.part"
    stale = store.settings.outputs_dir / "stale.png.part"
    fresh.write_bytes(b"encoding right now")
    stale.write_bytes(b"from a power cut")
    old = time.time() - 2 * store.PARTIAL_SETTLE_SECONDS
    os.utime(stale, (old, old))
    assert store.summarise()["partial_files"] == 1
    assert store.clear_history()["partials"] == 1
    assert fresh.exists() and not stale.exists()


def test_step_cost_is_measured_on_the_same_checkpoint(store):
    # klein at 1000 ms/step and Turbo at 1875 for the same size: a user who
    # restarted on Turbo was quoted klein's number.
    keep(store, seed=1, steps=10, model_id="klein")  # 100 ms/step
    keep(store, seed=2, steps=5, model_id="turbo")  # 200 ms/step
    assert store.step_cost_ms(256, 256, "turbo") == 200.0
    assert store.step_cost_ms(256, 256, "klein") == 100.0
    assert store.step_cost_ms(256, 256, "other") is None


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


def test_the_date_filter_is_a_local_day(store, monkeypatch):
    # Rows are stamped in UTC and an overnight run crosses midnight; `date:`
    # means the day on the user's clock, as a half-open range rather than a
    # LIKE on the UTC prefix.
    from datetime import datetime, timezone

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 7, 23, 30, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(store, "datetime", Clock)
    late = keep(store, seed=1)  # 23:30 UTC on the 7th
    Clock.now = classmethod(lambda cls, tz=None: datetime(2026, 9, 8, 1, 30, tzinfo=timezone.utc).astimezone(tz))
    early = keep(store, seed=2)  # 01:30 UTC on the 8th
    # Which local day each fell on depends on the zone the test runs in, so
    # ask the same question the code does.
    local_day = lambda rec: datetime.fromisoformat(rec.created_at).astimezone().date()  # noqa: E731
    for rec in (late, early):
        found = store.list_images(search=f"date:{local_day(rec).isoformat()}")[0]
        assert rec.id in {r.id for r in found}
    if local_day(late) != local_day(early):
        assert store.list_images(search=f"date:{local_day(late).isoformat()}")[1] == 1
    assert store.list_images(search="date:2020-01-01")[1] == 0
    assert store.list_images(search="date:today")[1] in (0, 1, 2)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        store.list_images(search="date:tuesday")


# ---------------------------------------------------------------------- stats


def test_keep_rates_rank_settings_by_starred_rate_not_count(store):
    # Six-step images were made most; eight-step ones were kept most. The
    # naive count would put 6 first. A lucky 1-of-1 at ten steps is dropped
    # by the floor rather than leading the table.
    ids = {}
    for steps, seeds in ((6, (1, 2, 3)), (8, (4, 5, 6)), (10, (7,))):
        for seed in seeds:
            ids[(steps, seed)] = keep(store, steps=steps, seed=seed, run_id="grid-a").id
    for key in ((6, 1), (8, 4), (8, 5), (8, 6), (10, 7)):
        store.set_favorite(ids[key], True)
    keep(store, steps=8, seed=99, run_id="grid-b")  # another run, not starred

    rows = store.keep_rates("run:grid-a", by=["steps"], min_made=2)
    assert [(r["steps"], r["made"], r["starred"]) for r in rows] == [(8, 3, 3), (6, 3, 1)]
    assert rows[0]["rate"] == 1.0 and rows[0]["size"] == "256x256"
    # Scope matters: all-time folds the other run's unstarred eight in.
    everything = store.keep_rates(None, by=["steps"], min_made=1)
    eight = next(r for r in everything if r["steps"] == 8)
    assert (eight["made"], eight["starred"]) == (4, 3)
    with pytest.raises(ValueError, match="free text"):
        store.keep_rates("fox", by=["steps"])
    with pytest.raises(ValueError, match="cannot group by"):
        store.keep_rates(None, by=["colour"])


def test_the_stats_tool_defaults_to_the_newest_run(store, monkeypatch, capsys):
    from backend import stats

    monkeypatch.setattr(stats, "settings", store.settings)
    for seed in (1, 2, 3):
        rec = keep(store, steps=8, seed=seed, run_id="night")
        if seed != 3:
            store.set_favorite(rec.id, True)
    assert stats.main(["--min", "2"]) == 0
    out = capsys.readouterr().out
    assert "the newest run, night (3 images, 2 starred)" in out
    assert "67%" in out and "steps" in out
    assert stats.main(["--all", "--by", "steps,cfg", "--min", "1"]) == 0
    assert "indicative" in capsys.readouterr().out
    assert stats.main(["--by", "colour"]) == 2


# ----------------------------------------------------------------------- runs


def test_runs_group_the_gallery_by_run_newest_first(store):
    # A resumed grid is two jobs under one run; a row from before runs had an
    # id belongs to none. The oldest image carries the base seed and the prompt.
    keep(store, prompt="a big fox", seed=100, job_id="night01", run_id="grid-a")
    keep(store, prompt="a small fox", seed=100, job_id="night01", run_id="grid-a")
    keep(store, prompt="a big fox", seed=101, job_id="night02", run_id="grid-a")
    keep(store, seed=7, job_id="solo", run_id="grid-b")
    keep(store, seed=8)  # no run
    store.set_favorite(store.list_images(search="seed:101")[0][0].id, True)

    runs = store.runs()
    assert [r.run_id for r in runs] == ["grid-b", "grid-a"]
    grid_a = runs[1]
    assert (grid_a.images, grid_a.favorites, grid_a.prompts, grid_a.jobs) == (3, 1, 2, 2)
    assert (grid_a.prompt, grid_a.seed) == ("a big fox", 100)
    assert (grid_a.width, grid_a.height, grid_a.model_id) == (256, 256, "krea/Krea-2-Turbo")
    assert grid_a.started_at <= grid_a.finished_at
    assert store.runs(limit=1)[0].run_id == "grid-b"


# --------------------------------------------------------------------- resume


def test_completed_cells_keys_on_the_settings_not_the_run(store):
    keep(store, seed=7, steps=8, guidance_scale=0.5, mu=1.15, job_id="first")
    done = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo")
    assert ("a fox in the snow", None, 8, 0.5, 1.15, None, 7) in done
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
    assert ("a big fox", None, 8, 0.0, None, None, 2) not in done


def test_completed_cells_are_scoped_to_the_quantizer(store):
    # The same seed through fp8 and int8 is two pictures; a resume under int8
    # must not count the fp8 night as done. A row from before the column was
    # recorded matches either, so an older gallery keeps resuming.
    keep(store, seed=1, quantization="fp8")
    keep(store, seed=2, quantization=None)
    turbo = ("a fox in the snow", None, 8, 0.0, None, None)
    fp8 = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo", "fp8")
    int8 = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo", "int8")
    assert {key[-1] for key in fp8} == {1, 2}
    assert {key[-1] for key in int8} == {2}
    assert (*turbo, 1) in fp8
    # And the quantizer is searchable, and lands in the PNG.
    assert [r.seed for r in store.list_images(search="quant:fp8")[0]] == [1]
    record = store.list_images(search="quant:fp8")[0][0]
    from PIL import Image as PILImage

    with PILImage.open(store.settings.outputs_dir / record.filename) as png:
        assert '"quantization": "fp8"' in png.info["krea"]
        assert "Quantization: fp8" in png.info["parameters"]


def test_a_mu_free_cell_comes_back_as_none(store):
    keep(store, seed=3, mu=None)
    done = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo")
    assert ("a fox in the snow", None, 8, 0.0, None, None, 3) in done


def test_a_row_from_before_true_cfg_reads_back_the_scale_it_ran_at(store):
    # Before the column, a negative prompt ran at `guidance_scale` on every
    # pipeline. Such a row - NULL in the column - has to come back under the
    # key a request would be written with today, or the night before the
    # upgrade can never be resumed; and the record, the filter and the
    # grouping have to read it the same way.
    old = keep(store, seed=1, negative_prompt="blurry", guidance_scale=2.0, run_id="r")
    plain = keep(store, seed=2, guidance_scale=2.0, run_id="r")
    with store._session() as conn:
        conn.execute("UPDATE images SET true_cfg = NULL")
    by_seed = {r.seed: r for r in store.list_images()[0]}
    assert (by_seed[1].true_cfg, by_seed[2].true_cfg) == (2.0, None)
    done = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo")
    assert ("a fox in the snow", "blurry", 8, 2.0, None, 2.0, 1) in done
    assert ("a fox in the snow", None, 8, 2.0, None, None, 2) in done
    # The SQL reading and the Python one agree, row for row.
    assert [r.seed for r in store.list_images(search="tcfg:2")[0]] == [1]
    assert [r.seed for r in store.list_images(search="tcfg:none")[0]] == [2]
    rates = store.keep_rates("run:r", by=("tcfg",))
    assert sorted(row["tcfg"] for row in rates if row["tcfg"] is not None) == [2.0]
    assert old.content_hash == plain.content_hash  # both blank 256x256 canvases
    # A row written now carries the value outright and reads back the same.
    fresh = keep(store, seed=3, negative_prompt="blurry", guidance_scale=2.0)
    assert fresh.true_cfg == 2.0 and store.get_image(fresh.id).true_cfg == 2.0
