"""The gallery index: migrations, the search filters, and what resume reads."""
from __future__ import annotations

import dataclasses
import io
import sqlite3
import sys
from pathlib import Path

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
    db = tmp_path / "auradream.db"
    monkeypatch.setattr(
        storage,
        "settings",
        dataclasses.replace(storage.settings, db_path=db, outputs_dir=outputs),
    )
    # This process has already checked the file at this path: the server ran
    # on it. Then last week's backup is restored over it, at the same path,
    # under the running server. A once-per-process check, keyed on the path,
    # never looked again, and every gallery read failed on `binned_at`.
    storage.init_db()
    keep(storage, job_id="new")
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(OLD_SCHEMA)
    conn.close()

    record = keep(storage, job_id="abc123", run_id="run001")
    assert (record.job_id, record.run_id) == ("abc123", "run001")
    items, total, _ = storage.list_images()
    assert total == 1 and (items[0].job_id, items[0].run_id) == ("abc123", "run001")
    # The recycle bin's column too, NULL - live - on every older row.
    with storage._session() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
    assert "binned_at" in columns and items[0].binned_at is None


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


def test_a_read_does_not_wait_for_another_threads_png_encode(store, monkeypatch):
    # The encode runs before the gallery reservation; held through it, every
    # gallery read on another thread waited out each save.
    import threading

    first = keep(store)
    encoding, release = threading.Event(), threading.Event()
    real = Image.Image.save

    def slow(self, fp, *args, **kwargs):
        if str(fp).endswith(".png.part"):
            encoding.set()
            assert release.wait(10)
        return real(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", slow)
    saver = threading.Thread(target=keep, args=(store,))
    saver.start()
    try:
        assert encoding.wait(10)
        reader = threading.Thread(target=store.get_image, args=(first.id,))
        reader.start()
        reader.join(5)
        assert not reader.is_alive(), "a read waited on another thread's encode"
    finally:
        release.set()
        saver.join(10)
    assert store.list_images()[1] == 2


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


def test_a_killed_bundle_s_leftovers_under_the_pre_rename_staging_name_are_swept(store):
    import os
    import time

    store.ensure_outputs_dir()
    legacy = store.settings.outputs_dir.resolve().parent / ".local-diffusion-staging"
    work = legacy / ".bundle-import-ab12cd34"
    (work / "images").mkdir(parents=True)
    (work / "images" / "unpacked.png").write_bytes(b"a gallery-sized copy")
    fresh = legacy / ".bundle-upload-fresh123.part"
    fresh.write_bytes(b"settling")
    old = time.time() - 2 * store.PARTIAL_SETTLE_SECONDS
    os.utime(work, (old, old))
    assert store.settle_partials() == {"recovered": 0, "discarded": 1}
    assert not work.exists() and fresh.exists()
    # Nothing new stages under the old name.
    assert store.bundle_staging_dir().name == ".auradream-staging"


def test_step_cost_is_measured_on_the_same_checkpoint(store):
    # klein at 1000 ms/step and Turbo at 1875 for the same size: a user who
    # restarted on Turbo was quoted klein's number.
    keep(store, seed=1, steps=10, model_id="klein")  # 100 ms/step
    keep(store, seed=2, steps=5, model_id="turbo")  # 200 ms/step
    assert store.step_cost_ms(256, 256, "turbo") == 200.0
    assert store.step_cost_ms(256, 256, "klein") == 100.0
    assert store.step_cost_ms(256, 256, "other") is None


def test_sizes_are_binary_and_say_so():
    # The arithmetic was always 1024's and the labels decimal's: the critic's
    # 4,856,283,677 bytes read "4.5 GB" where the Hub says 4.86 GB. The CLI's
    # disk line and the page (state.js humanBytes) label what they compute.
    assert storage.human_bytes(4_856_283_677) == "4.5 GiB"
    assert storage.human_bytes(312_087_009) == "297.6 MiB"
    assert storage.human_bytes(40 * 1024) == "40.0 KiB"
    assert storage.human_bytes(512) == "512.0 B"


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


def test_the_size_filter_is_width_and_height(store):
    # As the lightbox's facts print it (×) and as it is typed (x).
    keep(store, seed=1, width=1024, height=1024)
    keep(store, seed=2, width=1024, height=768)
    keep(store, seed=3, width=768, height=1024)
    assert [r.seed for r in store.list_images(search="size:1024x768")[0]] == [2]
    assert [r.seed for r in store.list_images(search="size:768×1024")[0]] == [3]
    assert store.list_images(search="size:1024X1024 seed:1")[1] == 1
    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        store.list_images(search="size:large")


def test_the_page_knows_every_search_key(store):
    # The lightbox's search buttons tell a prompt word that would read as a
    # filter from one that stays text by this list (gallery-search.js).
    import re
    from pathlib import Path

    from backend.search import FILTERS

    source = (Path(__file__).resolve().parents[1] / "web" / "gallery-search.js").read_text(encoding="utf-8")
    listed = re.search(r"SEARCH_KEYS = new Set\(\[([^\]]*)\]\)", source).group(1)
    known = set(re.findall(r"'([a-z]+)'", listed))
    assert known == set(FILTERS)


def test_a_filter_that_is_not_a_number_says_so(store):
    with pytest.raises(ValueError, match="not a number"):
        store.list_images(search="seed:tuesday")


@pytest.mark.parametrize("search", [
    "seed:99999999999999999999", "id:>=99999999999999999999", "steps:1-99999999999999999999",
    "seed:-9223372036854775809", "size:99999999999999999999x1024",
])
def test_a_number_past_sqlite_integers_is_refused_not_overflowed(store, search):
    # sqlite3 raises OverflowError, not ValueError, on an int past 64 bits:
    # the routes' 422 path never saw it and the gallery answered 500.
    keep(store, seed=1)
    with pytest.raises(ValueError, match="out of range"):
        store.list_images(search=search)
    assert store.list_images(search="seed:9223372036854775807")[1] == 0


def test_lora_none_is_the_images_made_without_an_adapter(store):
    keep(store, seed=1)
    keep(store, seed=2, lora="none.safetensors")
    keep(store, seed=3, lora="ink.safetensors")
    assert [r.seed for r in store.list_images(search="lora:none")[0]] == [1]
    assert [r.seed for r in store.list_images(search="lora:OFF")[0]] == [1]
    assert [r.seed for r in store.list_images(search="lora:=none.safetensors")[0]] == [2]
    assert [r.seed for r in store.list_images(search="lora:ink")[0]] == [3]
    # `none` is a stored quantization mode (an unquantised load), not NULL.
    keep(store, seed=4, quantization="none")
    assert [r.seed for r in store.list_images(search="quant:none")[0]] == [4]


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


def test_the_stats_tool_writes_utf8_where_its_output_is_redirected(store, monkeypatch):
    # `> out.txt` on Windows: stdout is the locale's code page, strict, and a
    # CJK or emoji prompt was a UnicodeEncodeError partway through the table.
    from backend import stats

    monkeypatch.setattr(stats, "settings", store.settings)
    redirected = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(redirected, encoding="cp1252"))
    keep(store, prompt="一只狐狸 🦊", run_id="night")
    assert stats.main(["--by", "prompt", "--min", "1"]) == 0
    sys.stdout.flush()
    assert "一只狐狸 🦊" in redirected.getvalue().decode("utf-8")
    # A StringIO holds text, with no encoding to change, and still works.
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert stats.main(["--by", "prompt", "--min", "1"]) == 0
    assert "一只狐狸 🦊" in sys.stdout.getvalue()


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
    # (prompt, negative, steps, CFG, mu, true CFG, references, LoRA, refine, mask, upscaler, edit, execution, shape, seed).
    assert ("a fox in the snow", None, 8, 0.5, 1.15, None, None, None, None, None, None, None, None, "256x256@center", 7) in done
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
    assert ("a big fox", None, 8, 0.0, None, None, None, None, None, None, None, None, None, "256x256@center", 2) not in done


def test_completed_cells_are_scoped_to_the_quantizer(store):
    # The same seed through fp8 and int8 is two pictures; a resume under int8
    # must not count the fp8 night as done. A row from before the column was
    # recorded matches either, so an older gallery keeps resuming.
    keep(store, seed=1, quantization="fp8")
    keep(store, seed=2, quantization=None)
    turbo = ("a fox in the snow", None, 8, 0.0, None, None, None, None, None, None, None, None, None, "256x256@center")
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
        assert '"quantization": "fp8"' in png.info["auradream"]
        assert "Quantization: fp8" in png.info["parameters"]


def test_a_mu_free_cell_comes_back_as_none(store):
    keep(store, seed=3, mu=None)
    done = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo")
    assert ("a fox in the snow", None, 8, 0.0, None, None, None, None, None, None, None, None, None, "256x256@center", 3) in done


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
    assert ("a fox in the snow", "blurry", 8, 2.0, None, 2.0, None, None, None, None, None, None, None, "256x256@center", 1) in done
    assert ("a fox in the snow", None, 8, 2.0, None, None, None, None, None, None, None, None, None, "256x256@center", 2) in done
    # The SQL reading and the Python one agree, row for row.
    assert [r.seed for r in store.list_images(search="tcfg:2")[0]] == [1]
    assert [r.seed for r in store.list_images(search="tcfg:none")[0]] == [2]
    rates = store.keep_rates("run:r", by=("tcfg",))
    assert sorted(row["tcfg"] for row in rates if row["tcfg"] is not None) == [2.0]
    assert old.content_hash == plain.content_hash  # both blank 256x256 canvases
    # A row written now carries the value outright and reads back the same.
    fresh = keep(store, seed=3, negative_prompt="blurry", guidance_scale=2.0)
    assert fresh.true_cfg == 2.0 and store.get_image(fresh.id).true_cfg == 2.0


# ---------------------------------------------------------------- recycle bin


def test_the_gallery_leaves_the_recycle_bin_out_and_the_bin_lists_only_it(store):
    # Total, page and the other-checkpoint count all read one predicate, the
    # LIKE fallback's included: a `26"` search swaps only the search term.
    live = keep(store, prompt='a 26" monitor on a desk', seed=1)
    gone = keep(store, prompt='a 26" monitor in a shop', seed=2)
    other = keep(store, prompt='a 26" monitor elsewhere', seed=3, model_id="other/model")
    binned_other = keep(store, prompt='a 26" monitor binned', seed=4, model_id="other/model")
    store.bin_images([gone.id, binned_other.id])

    items, total, others = store.list_images(search='26"', model_id="krea/Krea-2-Turbo")
    assert ([r.id for r in items], total, others) == ([live.id], 1, 1)
    items, total, _ = store.list_images(search='26"', binned=True)
    assert ([r.id for r in items], total) == ([binned_other.id, gone.id], 2)
    assert all(r.binned_at for r in items)
    items, total, others = store.list_images(binned=True, model_id="krea/Krea-2-Turbo")
    assert ([r.id for r in items], total, others) == ([gone.id], 1, 1)
    assert store.get_image(gone.id) is None
    assert store.get_image(gone.id, include_binned=True).binned_at is not None
    assert [r.id for r in store.get_images([live.id, gone.id, other.id])] == [live.id, other.id]
    assert len(store.get_images([gone.id], include_binned=True)) == 1


def test_runs_and_keep_rates_leave_the_bin_out_but_resume_and_lineage_do_not(store):
    kept = keep(store, seed=1, run_id="night", job_id="j")
    binned = keep(store, seed=2, run_id="night", job_id="j")
    alone = keep(store, seed=3, run_id="gone", job_id="k")
    store.set_favorite(kept.id, True)
    store.bin_images([binned.id, alone.id])

    runs = store.runs()
    assert [(r.run_id, r.images) for r in runs] == [("night", 1)]
    rows = store.keep_rates(None, by=["steps"])
    assert [(r["made"], r["starred"]) for r in rows] == [(1, 1)]
    # A binned cell was made: a resume does not make it again.
    done = store.completed_cells(["a fox in the snow"], 256, 256, "krea/Krea-2-Turbo")
    assert {key[-1] for key in done} == {1, 2, 3}
    assert [r.id for r in store.lineage_images(["night"], [])] == [kept.id, binned.id]


def test_a_binned_star_is_unkept_and_protects_no_source(store):
    source = keep(store, seed=1)
    star = keep(store, seed=2, refine=source.id, strength=0.5)
    store.set_favorite(star.id, True)
    preview = store.matching_preview(None)
    assert (preview["favorites"], preview["sources_kept"], preview["binned"]) == (1, 1, 0)

    store.bin_images([star.id])
    # The flag stays, so a Restore brings the star back; until then nothing
    # is kept for it, and it cannot be starred again.
    assert store.get_image(star.id, include_binned=True).favorite is True
    assert store.set_favorite(star.id, True) is None
    preview = store.matching_preview(None)
    assert (preview["favorites"], preview["sources_kept"], preview["binned"]) == (0, 0, 1)
    assert store.matching_preview(None, live_only=True)["records"] == 1
    summary = store.summarise()
    assert (summary["favorites"], summary["sources_kept"], summary["binned"]) == (0, 0, 1)
    result = store.delete_matching(None, keep_favorites=True, expected=2)
    assert sorted(result["ids"]) == [source.id, star.id]


def test_delete_matching_into_the_bin_counts_and_moves_only_live_rows(store):
    first = keep(store, seed=1)
    second = keep(store, seed=2)
    already = keep(store, seed=3)
    store.bin_images([already.id])
    with pytest.raises(store.SelectionChanged):
        store.delete_matching(None, keep_favorites=True, expected=3, to_bin=True)
    result = store.delete_matching(None, keep_favorites=True, expected=2, to_bin=True)
    assert (result["ids"], result["files"]) == ([first.id, second.id], 0)
    assert all((store.settings.outputs_dir / r.filename).exists() for r in (first, second))
    assert store.list_images(binned=True)[1] == 3


def test_a_binned_image_is_refused_as_a_reference_or_source(store):
    live = keep(store, seed=1)
    gone = keep(store, seed=2)
    store.bin_images([gone.id])
    with pytest.raises(ValueError, match=f"source image {gone.id} is in the recycle bin; restore it"):
        store.reference_paths([gone.id], "source image")
    with pytest.raises(ValueError, match="recycle bin"):
        store.reference_paths([live.id, gone.id])
    store.restore_images([gone.id])
    assert len(store.reference_paths([live.id, gone.id])) == 2


def test_a_pass_says_its_source_is_in_the_bin(store):
    source = keep(store, seed=1)
    made = keep(store, seed=2, refine=source.id, strength=0.5)
    assert (made.refine_missing, made.refine_binned) == (False, False)
    store.bin_images([source.id])
    record = store.get_image(made.id)
    assert (record.refine_missing, record.refine_binned) == (True, True)
    assert store.list_images()[0][0].refine_binned is True
    store.delete_image(source.id)
    record = store.get_image(made.id)
    assert (record.refine_missing, record.refine_binned) == (True, False)


# ---------------------------------------------------------------- newest id


def test_the_newest_id_counts_every_row_and_only_an_insert_raises_it(store):
    # The status poll's: a page catches up when it moves past what it saw.
    assert store.newest_image_id() == 0
    first, second = keep(store), keep(store, seed=2)
    assert store.newest_image_id() == second.id > first.id
    # A bin is not a new image, nor is its restore: the binned row counts.
    store.bin_images([second.id])
    assert store.newest_image_id() == second.id
    store.restore_images([second.id])
    assert store.newest_image_id() == second.id
    # A permanent delete of the newest lowers it; the next insert lands past
    # the old value anyway (AUTOINCREMENT).
    store.delete_image(second.id)
    assert store.newest_image_id() == first.id
    assert keep(store, seed=3).id > second.id
    # A clear empties the table, and ids stay monotonic past it.
    before = store.newest_image_id()
    store.clear_history()
    assert store.newest_image_id() == 0
    assert keep(store, seed=4).id > before == store.newest_image_id() - 1


def test_the_newest_id_is_unknown_rather_than_waited_for_while_reserved(store):
    from tests.bundle_helpers import holding_child
    import time

    keep(store)
    child = holding_child(store.settings.db_path)
    try:
        started = time.monotonic()
        assert store.newest_image_id() is None
        assert time.monotonic() - started < store.NEWEST_ID_WAIT + 2
    finally:
        child.stdin.close()
        child.wait(timeout=15)
    assert store.newest_image_id() == 1
