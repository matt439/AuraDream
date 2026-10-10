"""More seeds' planning without a server: batches, seeds, calls, the Batch."""
from __future__ import annotations

import dataclasses

from backend import more_seeds
from backend.config import settings
from backend.schemas import ImageRecord
from backend.seeds import MAX_SEED


def record(id_, seed, *, run_id="r", job_id="j", prompt="a fox", batch=None, **fields):
    fields = {"width": 256, "height": 256, "steps": 8, "guidance_scale": 1.5, **fields}
    return ImageRecord(
        id=id_, filename=f"{id_}.png", url=f"/outputs/{id_}.png", prompt=prompt, negative_prompt=None,
        seed=seed, model_id="m", run_id=run_id, job_id=job_id, batch=batch, duration_ms=1,
        created_at="2026-10-10T00:00:00", **fields,
    )


def test_a_batch_is_one_run_that_agrees_on_everything_but_the_seed():
    a, b = record(1, 10), record(2, 11)
    other_cfg = record(3, 10, guidance_scale=2.5)
    other_run = record(4, 12, run_id="s")
    # No run id: the job; neither: the image alone.
    by_job = [record(5, 1, run_id=None), record(6, 2, run_id=None)]
    alone = [record(7, 1, run_id=None, job_id=None), record(8, 2, run_id=None, job_id=None)]
    batches = more_seeds.group([a, b, other_cfg, other_run, *by_job, *alone])
    assert [[r.id for r in batch.picked] for batch in batches] == [[1, 2], [3], [4], [5, 6], [7], [8]]
    assert batches[0].run_id == "r" and batches[3].run_id is None
    assert batches[0].template.id == 2


def test_the_batch_is_the_largest_recorded_one_else_the_pages():
    # The 2 is a call that ended a batch of 4s, not a Batch of its own.
    assert more_seeds.batch_size([record(1, 1, batch=4), record(2, 2, batch=2), record(3, 3)], 6) == (4, True)
    assert more_seeds.batch_size([record(1, 1), record(2, 2)], 6) == (6, False)


def test_the_batch_is_held_to_the_pixel_budget_but_a_pass_is_not(monkeypatch):
    monkeypatch.setattr(more_seeds, "settings", dataclasses.replace(settings, max_batch_pixels=3 * 1024 * 1024))
    assert more_seeds.fitting(8, 1024, 1024, None) == 3
    assert more_seeds.fitting(2, 1024, 1024, None) == 2
    assert more_seeds.fitting(8, 2048, 2048, None) == 1
    assert more_seeds.fitting(8, 2048, 2048, 12) == 8


def test_next_seeds_step_over_the_taken_ones_and_wrap():
    taken = {12, 13, 15}
    assert more_seeds.next_seeds(11, 3, taken.__contains__) == [14, 16, 17]
    assert more_seeds.next_seeds(MAX_SEED - 1, 3, lambda seed: False) == [MAX_SEED, 0, 1]


def test_calls_are_consecutive_seeds_at_most_the_batch_each():
    assert more_seeds.calls([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert more_seeds.calls([1, 2, 4, 5, 6], 4) == [[1, 2], [4, 5, 6]]
    assert more_seeds.calls([MAX_SEED, 0, 1], 3) == [[MAX_SEED, 0, 1]]
