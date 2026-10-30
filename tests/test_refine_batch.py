"""Mass refine's per-record plan (`refine_batch.plan`), without the routes:
the order its reasons are decided in, the cell it builds through
`grid.cell_from_record`, and that helper's named changes."""
from __future__ import annotations

import dataclasses

import pytest

from backend import grid, refine_batch
from backend.config import settings
from backend.model_defaults import defaults_for
from backend.schemas import ImageRecord

RUNNING = settings.model_id


def record(**fields) -> ImageRecord:
    base = dict(
        id=5, filename="x.png", url="/images/x.png", prompt="a fox", negative_prompt="blurry",
        width=512, height=384, steps=20, guidance_scale=3.5, true_cfg=2.0, mu=0.8, seed=42,
        model_id=RUNNING, lora="mine", lora_scale=0.7, reference=[3], variables={"who": "fox"},
        duration_ms=1, created_at="2026-10-10T00:00:00",
    )
    return ImageRecord(**{**base, **fields})


def plan(rec, **kw):
    args = dict(running=RUNNING, profile=defaults_for(RUNNING), scale=1.5, strength=0.3,
                steps=None, include_foreign=False, skip_passes=True)
    return refine_batch.plan(rec, **{**args, **kw})


def test_a_record_becomes_a_pass_of_itself_keeping_its_settings():
    out = plan(record())
    assert out.status == "queued" and (out.width, out.height) == (768, 576)
    request = out.cell.request
    assert (request.refine, request.strength, request.width, request.height) == (5, 0.3, 768, 576)
    assert (request.prompt, request.negative_prompt, request.seed, request.steps) == ("a fox", "blurry", 42, 20)
    assert (request.guidance_scale, request.true_cfg, request.mu) == (3.5, 2.0, 0.8)
    assert (request.lora, request.lora_scale, request.reference) == ("mine", 0.7, [])
    assert (out.cell.seed, out.cell.label, out.cell.variables) == (42, "#5", {"who": "fox"})
    assert plan(record(), steps=12).cell.request.steps == 12


def test_the_reasons_are_decided_in_order():
    passed = record(refine=2, strength=0.5)
    assert plan(passed).status == "pass"
    # Not skipped, a pass is refined itself: its source is the pass.
    assert plan(passed, skip_passes=False).cell.request.refine == 5
    foreign = record(model_id="somebody/else", refine=2, strength=0.5)
    assert plan(foreign).status == "pass"
    assert plan(foreign, skip_passes=False).status == "foreign"
    big = record(width=settings.refine_max_side // 2 + 16, height=256)
    assert plan(big, scale=2).status == "too_big"
    assert plan(record(steps=2), strength=0.3).status == "no_step"


def test_another_checkpoints_record_takes_the_running_profile():
    profile = defaults_for(RUNNING)
    out = plan(record(model_id="somebody/else"), include_foreign=True)
    request = out.cell.request
    assert (request.steps, request.guidance_scale, request.mu) == (
        profile["steps"], profile["guidance_scale"], profile["mu"],
    )
    assert (request.true_cfg, request.lora, request.prompt, request.seed) == (None, None, "a fox", 42)
    assert out.foreign and len(out.notes) == 2
    assert plan(record(model_id="somebody/else"), include_foreign=True, steps=9).cell.request.steps == 9


def test_cell_from_record_takes_only_request_fields():
    with pytest.raises(TypeError, match="no GenerateRequest field colour"):
        grid.cell_from_record(record(), colour="red")
    # And what it is given is validated as any request is.
    with pytest.raises(ValueError):
        grid.cell_from_record(record(), strength=2.0, refine=5)
    assert dataclasses.asdict(grid.cell_from_record(record()))["label"] is None


def test_versions_count_seeds_up_from_the_records_one_cell_each():
    one = plan(record())
    out = plan(record(), versions=2)
    assert [(c.seed, c.request.seed, c.request.num_images) for c in out.cells] == [(42, 42, 1), (43, 43, 1)]
    assert [c.label for c in out.cells] == ["#5", "#5 v2"]
    # The first version is the cell a press without versions makes.
    assert grid.cell_key(out.cell) == grid.cell_key(one.cell) and not one.more
    assert [c.seed for c in plan(record(), versions=4).cells] == [42, 43, 44, 45]
    # Only the seed differs.
    first, second = (c.request.model_dump(exclude={"seed"}) for c in out.cells)
    assert first == second
    # A record not queued makes no version, and dropping one drops them all.
    assert plan(record(steps=2), versions=4).cells == []
    out.drop("file_missing", "gone")
    assert (out.cells, out.more, out.status) == ([], [], "file_missing")


def test_versions_wrap_at_the_seed_ceiling():
    top = 2**63 - 1
    assert [c.seed for c in plan(record(seed=top - 1), versions=4).cells] == [top - 1, top, 0, 1]
