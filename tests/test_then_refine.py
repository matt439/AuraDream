"""Then refine: a pass of every image a Generate or a grid makes, run right
after its cell, as an ordinary refine row under a run of its own.

The base rows are what a plain Generate of the same settings writes; each
pass is Mass refine's of its base (`refine_batch.plan`), so no key, run id
or Resume count of the base changes, and Mass refine reports the passes
already made.
"""
from __future__ import annotations

import dataclasses

from backend import grid, refine_batch, storage, submission, then_refine
from backend.config import settings
from backend.model_defaults import defaults_for
from backend.model_selection import current_model_id
from backend.schemas import GenerateRequest, ThenRefine, default_strength
from tests.api_helpers import run_grid, run_one, stream_job, wait_for

SPEC = {"scale": 1.5, "strength": 0.5}


def _rows(store, **search):
    return sorted(store.list_images(limit=200, **search)[0], key=lambda r: r.id)


def test_each_image_is_followed_by_its_pass(client, fake, store):
    job_id, events = run_one(client, num_images=2, seed=40, steps=8, then_refine=SPEC)
    assert events[-1]["status"] == "done"
    rows = _rows(store)
    bases = [r for r in rows if r.refine is None]
    passes = [r for r in rows if r.refine is not None]
    assert len(bases) == 2 and len(passes) == 2
    assert {(r.width, r.height) for r in bases} == {(256, 256)}
    assert {(r.width, r.height) for r in passes} == {(384, 384)}
    by_source = {p.refine: p for p in passes}
    assert set(by_source) == {b.id for b in bases}
    for base in bases:
        made = by_source[base.id]
        # The base's own seed, prompt and steps; the checkpoint's strength.
        assert (made.seed, made.prompt, made.steps) == (base.seed, base.prompt, base.steps)
        assert made.strength == 0.5 and made.batch == 1
        assert made.job_id == base.job_id == job_id
    spec = then_refine.resolved(ThenRefine(**SPEC))
    assert {p.run_id for p in passes} == {then_refine.pass_run_id(bases[0].run_id, spec)}
    assert bases[0].run_id != passes[0].run_id
    # Both bases of the batch, then their passes: the cell is one unit.
    assert [r.refine is not None for r in fake.requests] == [False, True, True]
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot["planned_images"] == 4 and snapshot["then_refine"]["strength"] == spec.strength
    assert snapshot["pass_run_id"] == passes[0].run_id and snapshot["stage"] is None


def test_runs_alternate_base_and_pass(client, fake, store):
    run_one(client, runs=3, seed=50, steps=8, then_refine=SPEC)
    assert [r.refine is not None for r in fake.requests] == [False, True] * 3


def test_a_pass_is_the_one_mass_refine_makes_of_its_base(client, fake, store):
    run_one(client, seed=60, steps=8, then_refine={"scale": 1.25, "strength": 0.5, "steps": 12})
    base, made = _rows(store)
    running = current_model_id(settings)
    planned = refine_batch.plan(
        base, running=running, profile=defaults_for(running), scale=1.25, strength=0.5, steps=12,
        include_foreign=False, skip_passes=False,
    )
    assert grid.cell_key(planned.cell) == grid.cell_key(grid.cell_from_record(made))
    assert (made.width, made.height, made.steps, made.strength) == (320, 320, 12, 0.5)
    # And Mass refine counts it made.
    answer = client.post("/api/refine/batch/preview", json={
        "ids": [base.id], "scale": 1.25, "strength": 0.5, "steps": 12,
    }).json()
    assert answer["images"][0]["status"] == "rendered"


def test_the_base_rows_are_a_plain_generates(client, fake, store):
    run_one(client, seed=70, steps=8, then_refine=SPEC)
    measured = _rows(store)[0]
    _, events = run_one(client, seed=70, steps=8)
    plain = _rows(store)[-1]
    assert plain.refine is None and plain.run_id == measured.run_id
    assert grid.cell_key(grid.cell_from_record(plain)) == grid.cell_key(grid.cell_from_record(measured))
    assert "then_refine" not in GenerateRequest.model_fields


def test_the_submit_refuses_in_words(client, fake, store, monkeypatch):
    def refused(**body):
        response = client.post("/api/generate", json={
            "prompt": "a fox", "width": 256, "height": 256, "then_refine": SPEC, **body,
        })
        assert response.status_code == 422, response.text
        return response.json()["detail"]

    source = _rows(store) or None
    run_one(client, seed=1, steps=8)
    source = _rows(store)[0]
    assert refused(refine=source.id, strength=0.5).startswith(
        f"Then refine follows a direct render; this is already a refine pass of #{source.id}"
    )
    assert "leaves no step" in refused(steps=4, then_refine={"scale": 1.5, "strength": 0.1})
    monkeypatch.setattr(refine_batch, "settings", dataclasses.replace(settings, refine_max_pixels=300 * 300))
    assert refused().startswith("256×256 at 1.5× is 384×384, past the")
    monkeypatch.undo()

    async def none(services):
        return "nothing here refines"

    monkeypatch.setattr(submission, "refine_unavailable", none)
    assert refused() == "nothing here refines"
    assert len(fake.requests) == 1


def test_a_failed_pass_keeps_its_base_and_is_no_failed_cell(client, fake, store):
    fake.fail_on = {2}
    job_id, events = run_one(client, runs=2, seed=80, steps=8, then_refine=SPEC)
    done = events[-1]
    assert done["status"] == "done" and done["failures"] == 1
    assert [c["label"] for c in done["failed_cells"]] == [f"pass of #{_rows(store)[0].id}"]
    rows = _rows(store)
    assert [r.refine is not None for r in rows] == [False, False, True]
    from backend import main

    job = main.jobs._jobs[job_id]
    assert job.failed_indices == [] and job.completed_indices == [0, 1]


def test_three_failed_passes_end_the_night(client, fake, store):
    fake.fail_on = {2, 4, 6}
    _, events = run_one(client, runs=5, seed=90, steps=8, then_refine=SPEC)
    assert events[-1]["status"] == "error" and events[-1]["stopped_by"] == "failures"
    assert len([r for r in _rows(store) if r.refine is None]) == 3


def test_a_cancel_during_a_pass_keeps_the_base(client, fake, store):
    fake.step_seconds = 0.05
    response = client.post("/api/generate", json={
        "prompt": "a fox", "width": 256, "height": 256, "steps": 20, "seed": 100, "then_refine": SPEC,
    })
    job_id = response.json()["job_id"]
    wait_for(client, job_id, lambda s: s["stage"] == "pass", timeout=20)
    client.post(f"/api/jobs/{job_id}/cancel")
    wait_for(client, job_id, lambda s: s["status"] == "cancelled", timeout=20)
    rows = _rows(store)
    assert len(rows) == 1 and rows[0].refine is None


def test_a_grid_refines_every_cell_and_keeps_the_spec_for_finish(client, fake, store):
    job_id, events = run_grid(client, seed=110, steps_axis="6,8", then_refine=SPEC)
    assert events[-1]["status"] == "done"
    rows = _rows(store)
    assert sorted(r.steps for r in rows if r.refine is not None) == [6, 8]
    (definition,) = storage.retained_run_jobs(events[-1]["run_id"])
    saved = definition["metadata"]["then_refine"]
    assert saved["scale"] == 1.5 and saved["pass_run_id"] == events[-1]["pass_run_id"]
    # The pass run has no cells of its own, and says whose passes it holds.
    reason = client.post(f"/api/runs/{saved['pass_run_id']}/finish/preview", json={}).json()["reason"]
    assert reason.startswith(f"the refine passes of run {events[-1]['run_id']}")


def test_the_preview_counts_the_passes(client, fake, store):
    answer = client.post("/api/grid/preview", json={
        "prompt": "a fox", "width": 256, "height": 256, "seed": 1, "steps_axis": "6,8",
        "num_images": 2, "then_refine": SPEC,
    }).json()
    assert answer["images"] == 4 and answer["passes"] == 4
    assert (answer["pass_width"], answer["pass_height"]) == (384, 384)
    plain = client.post("/api/grid/preview", json={
        "prompt": "a fox", "width": 256, "height": 256, "seed": 1,
    }).json()
    assert plain["passes"] == 0 and plain["pass_width"] is None


def _missing_passes(client, fake, runs):
    """A run of `runs` bases whose every pass failed: Finish run's missing
    passes, `runs` of them. The fake counts calls across the test."""
    first = len(fake.requests)
    fake.fail_on = {first + 2 * n for n in range(1, runs + 1)}
    _, events = run_one(client, runs=runs, seed=130, steps=8, then_refine=SPEC)
    fake.fail_on = set()
    run_id = events[-1]["run_id"]
    preview = client.post(f"/api/runs/{run_id}/finish/preview", json={}).json()
    assert preview["available"] and (preview["cells"], preview["passes"]) == (0, runs), preview
    return run_id, preview


def _finish(client, run_id, preview):
    response = client.post(f"/api/runs/{run_id}/finish", json={"expected": preview["pin"]})
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def test_missing_passes_run_as_units_that_pause_between_them(client, fake, store):
    from backend import main

    run_id, preview = _missing_passes(client, fake, 2)
    bases = [r for r in _rows(store) if r.refine is None]
    spec = then_refine.resolved(ThenRefine(**SPEC))
    second = then_refine.plan_passes([bases[1]], spec)[0].cell

    def pause_after_the_first():
        fake.after_generate = None
        for job in main.jobs.active():
            main.jobs.pause(job.id, True)

    fake.after_generate = pause_after_the_first
    job_id = _finish(client, run_id, preview)
    paused = wait_for(client, job_id, lambda s: s["paused"], timeout=20)
    assert paused["image_count"] == 1 and paused["run"] == 1 and paused["total_runs"] == 2
    assert paused["planned_images"] == 2 and paused["pass_sources"] == [b.id for b in bases]
    # The pass still to run is planned, so a Mass refine or a Finish meanwhile
    # does not queue it again; the one made has left the plan.
    planned = main.jobs.planned_keys()
    ran = grid.Cell(request=fake.applied(second.request), seed=second.seed)
    assert set(grid.cell_keys(ran)) <= planned[(384, 384)]
    again = client.post(f"/api/runs/{run_id}/finish/preview", json={}).json()
    assert not again["available"] and "queued" in again["reason"]
    client.post(f"/api/jobs/{job_id}/pause", params={"value": "false"})
    done = stream_job(client, job_id)[-1]
    assert done["status"] == "done" and len(done["images"]) == 2
    assert {r.refine for r in _rows(store) if r.refine is not None} == {b.id for b in bases}
    assert main.jobs.planned_keys() == {}


def test_a_budget_stops_missing_passes_between_units(client, fake, store, monkeypatch):
    from backend import main

    run_id, preview = _missing_passes(client, fake, 2)
    # The budget is the retained job's: 3 s, and each pass takes 4 s.
    real = storage.retained_run_jobs

    def budgeted(run=None):
        found = real(run)
        for saved in found:
            saved["metadata"]["budget_minutes"] = 0.05
        return found

    monkeypatch.setattr(storage, "retained_run_jobs", budgeted)
    preview = client.post(f"/api/runs/{run_id}/finish/preview", json={}).json()
    fake.step_seconds = 1.0
    done = stream_job(client, _finish(client, run_id, preview))[-1]
    assert done["stopped_by"] == "budget" and len(done["images"]) == 1
    assert main.jobs.get(done["job_id"]).failures == 0


def test_a_cancel_stops_inside_a_missing_pass(client, fake, store):
    run_id, preview = _missing_passes(client, fake, 2)
    fake.step_seconds = 0.05
    job_id = _finish(client, run_id, preview)
    wait_for(client, job_id, lambda s: s["stage"] == "pass", timeout=20)
    client.post(f"/api/jobs/{job_id}/cancel")
    done = stream_job(client, job_id)[-1]
    assert done["status"] == "cancelled" and done["images"] == []
    assert not [r for r in _rows(store) if r.refine is not None]


def test_missing_passes_failing_three_in_a_row_end_the_job(client, fake, store):
    from backend import main

    run_id, preview = _missing_passes(client, fake, 3)
    bases = [r for r in _rows(store) if r.refine is None]
    fake.fail_on = {len(fake.requests) + n for n in (1, 2, 3)}
    done = stream_job(client, _finish(client, run_id, preview))[-1]
    assert done["status"] == "error" and done["stopped_by"] == "failures"
    assert [c["label"] for c in done["failed_cells"]] == [f"pass of #{b.id}" for b in bases]
    job = main.jobs.get(done["job_id"])
    assert job.failed_indices == [] and job.cells == [] and job.failures == 3


def test_finish_run_carries_then_refine(client, fake, store):
    fake.fail_on = {1}
    _, events = run_grid(client, seed=120, steps_axis="6,8", then_refine=SPEC)
    run_id = events[-1]["run_id"]
    preview = client.post(f"/api/runs/{run_id}/finish/preview", json={}).json()
    assert preview["available"] and preview["cells"] == 1, preview
    accepted = client.post(f"/api/runs/{run_id}/finish", json={"pin": preview["pin"]})
    assert accepted.status_code == 200, accepted.text
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert done["then_refine"]["scale"] == 1.5 and done["pass_run_id"] == events[-1]["pass_run_id"]
    assert len([r for r in _rows(store) if r.refine is not None]) == 2
