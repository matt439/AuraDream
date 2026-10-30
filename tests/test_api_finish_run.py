"""Finish retained runs independently of a composer or an earlier process."""
from __future__ import annotations

import dataclasses
import gzip
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from backend import grid, main, storage
from backend.config import settings
from backend.jobs import JobManager
from backend.schemas import GenerateRequest, Outpaint
from tests.api_helpers import stream_job, wait_for


def cell(seed, *, prompt="a retained fox", batch=1, **fields):
    return grid.Cell(
        request=GenerateRequest(**{"prompt": prompt, "width": 256, "height": 256, "steps": 1,
                                   "num_images": batch, **fields}),
        seed=seed, label=f"seed {seed}", variables={"take": str(seed)},
    )


def retain(store, cells, run="retained-night", *, model=None, quant=None, metadata=None, completed=()):
    job_id = uuid.uuid4().hex[:12]
    first = cells[0]
    store.retain_run_job({
        "job_id": job_id, "run_id": run, "created_at": time.time(), "origin": "grid",
        "model_id": model or settings.model_id, "quantization": quant or settings.quantization,
        "width": first.request.width, "height": first.request.height,
        "batch": first.request.num_images, "comparison": bool((metadata or {}).get("comparison")),
        "metadata": {"grid": True, **(metadata or {})},
    }, cells)
    if completed:
        store.finish_run_job(job_id, {"status": "done", "reached": len(cells),
                                     "completed_indices": list(completed), "failed_indices": [],
                                     "stopped_by": None, "finished_at": time.time()})
    return run


def plant(store, item, run, *, offset=0):
    saved = store.retained_run_jobs(run)
    return store.save_image(
        Image.new("RGB", (item.request.width, item.request.height)),
        main.backend.applied(item.request), seed=item.seed + offset, duration_ms=1,
        model_id=settings.model_id, quantization=settings.quantization,
        run_id=run, job_id=saved[0]["job_id"] if saved else "prior-process",
        batch=item.request.num_images, variables=item.variables,
    )


def preview(client, run):
    response = client.post(f"/api/runs/{run}/finish/preview")
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, run, answer=None):
    body = {"expected": answer["pin"]} if answer is not None else {}
    response = client.post(f"/api/runs/{run}/finish", json=body)
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["status"] == "done", done
    return response.json(), done


def test_finish_retains_order_labels_variables_and_original_id_without_grid_expansion(client, store, fake):
    cells = [cell(10), cell(50, prompt="a different fox")]
    run = retain(store, cells)
    answer = preview(client, run)
    assert (answer["available"], answer["total"], answer["cells"], answer["images"]) == (True, 2, 2, 2)
    accepted, done = finish(client, run, answer)
    assert accepted["run_id"] == done["run_id"] == run
    assert fake.seeds == [10, 50]
    assert [row["variables"] for row in done["images"]] == [c.variables for c in cells]
    assert main.jobs.get(accepted["job_id"]).cells == cells
    again = preview(client, run)
    assert (again["total"], again["rendered"], again["cells"]) == (2, 2, 0)
    assert not again["available"] and "nothing" in again["reason"]


def test_finish_after_a_budget_stop_uses_same_budget_and_only_remaining_seeds(client, store, fake):
    cells = [cell(seed) for seed in range(100, 108)]
    # 3 s and 1 s cells: the first pass stops partway through the eight, and
    # the finish, with instant cells, fits the rest in the same budget even on
    # CI's Windows runner (0.6 s did not).
    run = retain(store, cells, metadata={"budget_minutes": 0.05})
    fake.step_seconds = 1.0
    response = client.post(f"/api/runs/{run}/finish", json={})
    assert response.status_code == 200, response.text
    stopped = stream_job(client, response.json()["job_id"])[-1]
    assert stopped["stopped_by"] == "budget" and 0 < len(stopped["images"]) < 8
    before = len(stopped["images"])
    fake.step_seconds = 0
    answer = preview(client, run)
    assert answer["cells"] == 8 - before and answer["budget_minutes"] == 0.05
    accepted, done = finish(client, run, answer)
    assert accepted["budget_minutes"] == 0.05 and len(done["images"]) == 8 - before
    assert fake.seeds == list(range(100, 108))


def test_stopped_plain_repeat_finishes_after_manager_restart(client, fake, monkeypatch):
    fake.step_seconds = 0.03
    response = client.post("/api/generate", json={"prompt": "repeat", "width": 256, "height": 256,
                                                   "seed": 80, "runs": 4, "steps": 5})
    assert response.status_code == 200, response.text
    job, run = response.json()["job_id"], response.json()["run_id"]
    client.post(f"/api/jobs/{job}/pause")
    state = wait_for(client, job, lambda value: value["paused"])
    client.post(f"/api/jobs/{job}/cancel")
    stream_job(client, job)
    made = state["image_count"]
    assert 0 < made < 4
    client.portal.call(main.jobs.shutdown)
    monkeypatch.setattr(main, "jobs", JobManager(fake, main.ensure_loaded))
    fake.step_seconds = 0
    answer = preview(client, run)
    assert answer["total"] == 4 and answer["cells"] == 4 - made
    accepted, done = finish(client, run, answer)
    assert done["grid"] is False and accepted["run_id"] == run
    assert fake.seeds == [80, 81, 82, 83]


@pytest.mark.parametrize("all_deleted", [False, True])
def test_deleted_base_seed_and_all_deleted_runs_redraw_exact_cells(client, store, fake, all_deleted):
    cells = [cell(10), cell(11), cell(12)]
    run = retain(store, cells, completed=range(3))
    records = [plant(store, item, run) for item in cells]
    for record in records if all_deleted else records[:1]:
        store.delete_image(record.id)
    answer = preview(client, run)
    assert answer["available"] and answer["redrawn"] == (3 if all_deleted else 1)
    finish(client, run, answer)
    assert fake.seeds == ([10, 11, 12] if all_deleted else [10])


def test_binned_cells_count_as_rendered_and_empty_bin_makes_them_eligible(client, store, fake):
    cells = [cell(1), cell(2)]
    run = retain(store, cells, completed=range(2))
    records = [plant(store, item, run) for item in cells]
    store.bin_images([row.id for row in records])
    answer = preview(client, run)
    assert not answer["available"] and (answer["rendered"], answer["redrawn"]) == (2, 0)
    details = client.get(f"/api/runs/{run}").json()
    assert details["binned"] == 2 and not details["actions"]["seeds"]["available"]
    summary = store.bin_summary()
    store.empty_bin(expected=2, pin=summary["pin"])
    answer = preview(client, run)
    assert answer["available"] and answer["redrawn"] == 2


def test_partly_saved_batch_runs_again_in_full_instead_of_triggering_drift_hold(client, store, fake):
    item = cell(20, batch=2)
    run = retain(store, [item])
    plant(store, item, run)
    answer = preview(client, run)
    assert answer["available"] and (answer["cells"], answer["images"]) == (1, 2)
    finish(client, run, answer)
    assert fake.seeds == [20] and store.list_images()[1] == 3


@pytest.mark.parametrize("kind, words", [("model", "made on"), ("quant", "made under")])
def test_foreign_checkpoint_and_quantizer_hold_before_any_admission(client, store, fake, kind, words):
    run = retain(store, [cell(1)], **({"model": "other/checkpoint"} if kind == "model" else {"quant": "int8"}))
    answer = preview(client, run)
    assert not answer["available"] and words in answer["reason"]
    assert client.post(f"/api/runs/{run}/finish", json={}).status_code == 409
    assert not fake.seeds


def test_nonempty_run_with_no_matching_keys_is_held_as_drift(client, store, fake):
    run = retain(store, [cell(1)])
    plant(store, cell(99, prompt="different old settings"), run)
    answer = preview(client, run)
    assert not answer["available"] and "no longer reproduces" in answer["reason"]
    assert not fake.seeds


def test_legacy_run_has_seeds_but_no_finish_and_unknown_run_is_404(client, store):
    plant(store, cell(1), "legacy")
    details = client.get("/api/runs/legacy").json()
    assert not details["actions"]["finish"]["available"]
    assert "before runs kept" in details["actions"]["finish"]["reason"]
    assert details["actions"]["seeds"]["available"]
    for path in ("/api/runs/gone", "/api/runs/gone/finish/preview", "/api/runs/gone/finish"):
        response = client.get(path) if path == "/api/runs/gone" else client.post(path, json={})
        assert response.status_code == 404


def test_preview_pin_refuses_a_changed_missing_set_then_new_preview_queues_the_rest(client, store, fake):
    cells = [cell(1), cell(2)]
    run = retain(store, cells)
    answer = preview(client, run)
    plant(store, cells[0], run)
    refused = client.post(f"/api/runs/{run}/finish", json={"expected": answer["pin"]})
    assert refused.status_code == 409 and "changed" in refused.json()["detail"]
    fresh = preview(client, run)
    finish(client, run, fresh)
    assert fake.seeds == [2]


def test_live_run_is_held_with_job_id_and_stale_revision_is_refused(client, store, fake):
    run = retain(store, [cell(1), cell(2)])
    fake.step_seconds = 0.1
    response = client.post(f"/api/runs/{run}/finish", json={})
    assert response.status_code == 200, response.text
    job = response.json()["job_id"]
    details = client.get(f"/api/runs/{run}").json()
    assert not details["actions"]["finish"]["available"]
    assert job in details["actions"]["finish"]["reason"]
    revision = main.model_selection.revision()
    for method, path in (("GET", f"/api/runs/{run}"), ("POST", f"/api/runs/{run}/finish/preview"),
                         ("POST", f"/api/runs/{run}/finish")):
        stale = client.request(method, path, json={} if method == "POST" else None,
                               headers={"X-AuraDream-Model-Revision": str(revision - 1)})
        assert stale.status_code == 409 and "checkpoint changed" in stale.json()["detail"]
    stream_job(client, job)


def test_concurrent_finish_submissions_admit_each_missing_cell_once(client, store, fake, monkeypatch):
    cells = [cell(1), cell(2)]
    run = retain(store, cells)
    entered, release = threading.Event(), threading.Event()
    original = storage.completed_cells
    calls = 0

    def blocked(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "completed_cells", blocked)
    fake.step_seconds = 0.05
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, f"/api/runs/{run}/finish", json={})
        assert entered.wait(5)
        second = pool.submit(client.post, f"/api/runs/{run}/finish", json={})
        release.set()
        answers = [first.result(10), second.result(10)]
    assert sorted(answer.status_code for answer in answers) == [200, 409]
    accepted = next(answer.json() for answer in answers if answer.status_code == 200)
    stream_job(client, accepted["job_id"])
    assert fake.seeds == [1, 2]


def test_comparison_cells_and_extensions_keep_original_variants(client, store, fake):
    cells = [cell(1, prompt="version one"), cell(1, prompt="version two")]
    cells = [dataclasses.replace(item, variables={"variant": version})
             for item, version in zip(cells, ("one", "two"))]
    run = retain(store, cells, metadata={"comparison": True, "prompt_template": "comparison"})
    retain(store, [cell(2, prompt="version one")], run)
    plant(store, cells[0], run)
    answer = preview(client, run)
    accepted, done = finish(client, run, answer)
    assert done["comparison"] and done["prompt_template"] == "comparison"
    assert [request.prompt for request in fake.requests] == ["version two", "version one"]
    assert accepted["run_id"] == run and done["images"][0]["variables"] == {"variant": "two"}


def test_first_retained_job_can_be_a_resume_with_only_a_legacy_prefix_in_gallery(client, store, fake):
    run = "resumed-before-first-new-image"
    plant(store, cell(1), run)  # Its earlier job predates retained cells.
    retain(store, [cell(2), cell(3)], run)
    answer = preview(client, run)
    assert answer["available"] and answer["cells"] == 2
    finish(client, run, answer)
    assert fake.seeds == [2, 3]


@pytest.mark.parametrize("deleted", [False, True])
def test_zero_image_runs_remain_in_picker_after_restart(client, store, fake, monkeypatch, deleted):
    run = retain(store, [cell(1), cell(2)])
    if deleted:
        record = plant(store, cell(1), run)
        store.delete_image(record.id)
    client.portal.call(main.jobs.shutdown)
    monkeypatch.setattr(main, "jobs", JobManager(fake, main.ensure_loaded))
    runs = client.get("/api/runs").json()
    recovered = next(item for item in runs if item["run_id"] == run)
    assert recovered["images"] == 0 and recovered["seed"] == 1
    assert preview(client, run)["available"]


def test_finish_has_no_whole_grid_cap_on_already_retained_cells(client, store, fake, monkeypatch):
    run = retain(store, [cell(1), cell(2), cell(3)])
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, max_grid_images=1))
    accepted, done = finish(client, run, preview(client, run))
    assert accepted["cells"] == len(done["images"]) == 3


def test_one_unreadable_retained_run_cannot_break_other_runs_or_admit_work(client, store, caplog):
    retain(store, [cell(1)], "corrupt-run")
    retain(store, [cell(2)], "healthy-empty-run")
    plant(store, cell(3), "healthy-live-run")
    with store._write_lock, store._session() as conn:
        conn.execute("UPDATE run_jobs SET cells = ? WHERE run_id = ?", (b"not-gzip", "corrupt-run"))
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert {run["run_id"] for run in response.json()} == {"healthy-empty-run", "healthy-live-run"}
    assert "omitting unreadable retained run corrupt-run" in caplog.text
    for suffix in ("", "/finish/preview", "/finish"):
        response = client.get("/api/runs/corrupt-run") if not suffix else client.post("/api/runs/corrupt-run" + suffix, json={})
        assert response.status_code == 422 and "cannot be read" in response.json()["detail"]
    assert not main.jobs.active()


@pytest.mark.parametrize("broken", ["metadata-list", "outcome-list", "indices-string", "budget-string",
                                   "seed-string", "seed-negative", "seed-bool", "seed-overflow",
                                   "variables-list", "finish-time-string"])
def test_invalid_retained_shapes_are_isolated_and_refused(client, store, broken):
    retain(store, [cell(1)], "broken")
    retain(store, [cell(2)], "healthy")
    with store._session() as conn:
        saved = dict(conn.execute("SELECT * FROM run_jobs WHERE run_id = 'broken'").fetchone())
    raw = json.loads(gzip.decompress(saved["cells"]))
    metadata, outcome = {"grid": True}, None
    if broken == "metadata-list":
        metadata = []
    elif broken == "outcome-list":
        outcome = [1]
    elif broken == "indices-string":
        outcome = {"completed_indices": "0"}
    elif broken == "budget-string":
        metadata["budget_minutes"] = "ten"
    elif broken == "variables-list":
        raw[0]["variables"] = []
    elif broken == "finish-time-string":
        outcome = {"finished_at": "yesterday"}
    else:
        raw[0]["seed"] = {"seed-string": "1", "seed-negative": -1,
                          "seed-bool": True, "seed-overflow": 2**63}[broken]
    from backend import gallery_lock
    with gallery_lock.hold(settings.db_path, wait=gallery_lock.SESSION_WAIT), store._write_lock, store._session() as conn:
        conn.execute("UPDATE run_jobs SET metadata = ?, outcome = ?, cells = ? WHERE run_id = 'broken'",
                     (json.dumps(metadata), json.dumps(outcome) if outcome is not None else None,
                      gzip.compress(json.dumps(raw).encode())))
    assert {item["run_id"] for item in client.get("/api/runs").json()} == {"healthy"}
    for suffix in ("", "/finish/preview", "/finish"):
        path = "/api/runs/broken" + suffix
        response = client.get(path) if not suffix else client.post(path, json={})
        assert response.status_code == 422 and "cannot be read" in response.json()["detail"]
    assert not main.jobs.active()


@pytest.mark.parametrize("first_route", ["resume", "more-seeds"])
def test_finish_and_other_continuations_share_admission_before_disk_await(client, store, fake, monkeypatch, first_route):
    base = GenerateRequest(prompt="continuation", width=256, height=256, steps=1, seed=1)
    cells = grid.expand(base, seeds=2, steps="1")
    run = grid.run_id(cells, settings.model_id, settings.quantization)
    retain(store, cells, run)
    plant(store, cells[0], run)
    entered, release, worker = threading.Event(), threading.Event(), threading.Event()
    original = storage.retain_run_job
    calls = 0

    def blocked(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "retain_run_job", blocked)
    fake.after_generate = lambda: worker.wait(5)
    route, body = ("/api/grid", {**base.model_dump(), "steps_axis": "1", "seeds": 2, "resume": True}) if first_route == "resume" else (
        "/api/more-seeds", {"run": run, "count": 1})
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(client.post, route, json=body)
            assert entered.wait(5) and not main.jobs.active()
            second = pool.submit(client.post, f"/api/runs/{run}/finish", json={})
            deadline = time.monotonic() + 5
            while main._model_requests < 2:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            release.set()
            answers = [first.result(10), second.result(10)]
            assert [answer.status_code for answer in answers] == [200, 409], [answer.text for answer in answers]
        finally:
            release.set()
            worker.set()
    stream_job(client, answers[0].json()["job_id"])
    assert fake.seeds == [2]


@pytest.mark.parametrize("missing", ["source", "reference", "lora", "mask"])
def test_missing_inputs_refuse_finish_before_admission(client, store, fake, missing):
    if missing == "source":
        item = cell(1, steps=20, refine=999, strength=0.5)
    elif missing == "reference":
        item = cell(1, reference=[999])
    elif missing == "lora":
        item = cell(1, lora="missing-adapter")
    else:
        source = plant(store, cell(9), "source")
        item = cell(1, steps=20, refine=source.id, strength=0.5, mask="a" * 64)
    run = retain(store, [item])
    answer = preview(client, run)
    assert not answer["available"] and answer["reason"]
    response = client.post(f"/api/runs/{run}/finish", json={})
    assert response.status_code == 422, response.text
    assert not main.jobs.active() and not fake.requests


@pytest.mark.parametrize("awaited", ["hold_masks", "stamp_outpaints"])
def test_finish_admission_covers_awaited_mask_work(client, store, fake, monkeypatch, awaited):
    source = plant(store, cell(9), "source")
    fields = {"steps": 20, "refine": source.id, "strength": 0.5}
    if awaited == "hold_masks":
        digest, _ = store.store_mask(Image.new("L", (256, 256), 255))
        item = cell(1, mask=digest, **fields)
    else:
        item = cell(1, width=304, outpaint=Outpaint(left=32, right=16, band=16), **fields)
    run = retain(store, [item])
    entered, release, worker = threading.Event(), threading.Event(), threading.Event()
    original = getattr(storage, awaited)
    calls = 0

    def blocked(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, awaited, blocked)
    fake.after_generate = lambda: worker.wait(5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(client.post, f"/api/runs/{run}/finish", json={})
            assert entered.wait(5) and not main.jobs.active()
            second = pool.submit(client.post, f"/api/runs/{run}/finish", json={})
            release.set()
            answers = [first.result(10), second.result(10)]
            assert sorted(answer.status_code for answer in answers) == [200, 409]
        finally:
            release.set()
            worker.set()
    accepted = next(answer.json() for answer in answers if answer.status_code == 200)
    stream_job(client, accepted["job_id"])
    assert len(fake.requests) == 1


def test_queue_snapshot_covers_completion_during_gallery_scan(client, store, fake, monkeypatch):
    run = retain(store, [cell(1), cell(2)])
    release_job, started = threading.Event(), threading.Event()

    def hold():
        started.set()
        assert release_job.wait(5)

    fake.after_generate = hold
    ordinary = client.post("/api/generate", json=cell(1).request.model_copy(update={"seed": 1}).model_dump()).json()
    assert started.wait(5)
    original = storage.completed_cells
    scanned = False

    def scan(*args, **kwargs):
        nonlocal scanned
        result = original(*args, **kwargs)
        if not scanned:
            scanned = True
            release_job.set()
            deadline = time.monotonic() + 5
            while main.jobs.get(ordinary["job_id"]).status not in {"done", "error", "cancelled"}:
                assert time.monotonic() < deadline
                time.sleep(0.005)
        return result

    monkeypatch.setattr(storage, "completed_cells", scan)
    try:
        response = client.post(f"/api/runs/{run}/finish", json={})
        assert response.status_code == 200, response.text
        assert response.json()["planned"] == 1 and response.json()["pruned"] == 0
        stream_job(client, response.json()["job_id"])
    finally:
        release_job.set()
    assert fake.seeds == [1, 2]


# ------------------------------------------------- Then refine's missing passes

SPEC = {"scale": 1.5, "strength": 0.5}


def then_refine_run(client, fake, *, runs=1, failing=(), **body):
    """A Then refine Generate whose `failing` calls (1-based, counted from
    this run's first) raise: a failed pass leaves its base without one."""
    fake.fail_on = {len(fake.requests) + call for call in failing}
    response = client.post("/api/generate", json={
        "prompt": "a fox at dusk", "width": 256, "height": 256, "steps": 8, "seed": 500,
        "runs": runs, "then_refine": SPEC, **body,
    })
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    fake.fail_on = set()
    return done


def rows(store):
    return sorted(store.list_images(limit=200)[0], key=lambda row: row.id)


def mass_refine_key(record):
    from backend import refine_batch
    from backend.model_defaults import defaults_for
    from backend.model_selection import current_model_id

    running = current_model_id(settings)
    planned = refine_batch.plan(
        record, running=running, profile=defaults_for(running), scale=1.5, strength=0.5, steps=None,
        include_foreign=False, skip_passes=False,
    )
    return grid.cell_key(planned.cell)


def holding(fake):
    """Hold the fake's next call at its end until released: a job kept live."""
    release, started = threading.Event(), threading.Event()

    def hold():
        started.set()
        assert release.wait(5)

    fake.after_generate = hold
    return release, started


def test_a_pass_a_cancel_left_missing_is_finished_under_the_pass_run(client, store, fake):
    fake.step_seconds = 0.05
    response = client.post("/api/generate", json={
        "prompt": "a fox at dusk", "width": 256, "height": 256, "steps": 20, "seed": 700, "then_refine": SPEC,
    })
    job = response.json()["job_id"]
    wait_for(client, job, lambda state: state["stage"] == "pass", timeout=20)
    client.post(f"/api/jobs/{job}/cancel")
    stopped = wait_for(client, job, lambda state: state["status"] == "cancelled", timeout=20)
    fake.step_seconds = 0
    run, pass_run = stopped["run_id"], stopped["pass_run_id"]
    (base,) = rows(store)
    assert base.refine is None
    answer = preview(client, run)
    assert answer["available"] and (answer["cells"], answer["passes"], answer["rendered"]) == (0, 1, 1), answer
    assert client.get(f"/api/runs/{run}").json()["actions"]["finish"]["available"]

    # Held inside its pass, so the second preview meets the live job.
    release, started = holding(fake)
    try:
        accepted = client.post(f"/api/runs/{run}/finish", json={"expected": answer["pin"]})
        assert accepted.status_code == 200, accepted.text
        assert (accepted.json()["cells"], accepted.json()["passes"]) == (0, 1)
        assert started.wait(5)
        again = preview(client, run)
        assert not again["available"] and "queued" in again["reason"]
    finally:
        release.set()
    done = stream_job(client, accepted.json()["job_id"])[-1]
    fake.after_generate = None
    assert done["status"] == "done" and done["pass_run_id"] == pass_run and done["pass_sources"] == [base.id]
    made = [row for row in rows(store) if row.refine == base.id]
    assert len(made) == 1 and made[0].run_id == pass_run and made[0].seed == base.seed
    assert grid.cell_key(grid.cell_from_record(made[0])) == mass_refine_key(base)
    # Nothing left: the run says so, and the pass run still names its base.
    final = preview(client, run)
    assert not final["available"] and final["reason"] == "nothing in this run is missing"
    assert preview(client, pass_run)["reason"].startswith(f"the refine passes of run {run}")


def test_a_binned_base_gets_no_pass(client, store, fake):
    done = then_refine_run(client, fake, runs=2, failing=(2, 4))
    assert done["failures"] == 2
    first, second = rows(store)
    store.bin_images([first.id])
    answer = preview(client, done["run_id"])
    assert answer["available"] and (answer["cells"], answer["passes"]) == (0, 1)
    finish(client, done["run_id"], answer)
    assert [row.refine for row in rows(store) if row.refine is not None] == [second.id]


def test_a_pass_mass_refine_made_counts_as_made_and_a_queued_one_as_queued(client, store, fake):
    done = then_refine_run(client, fake, runs=2, failing=(2, 4))
    first, second = rows(store)
    made = client.post("/api/refine/batch", json={"ids": [first.id], "scale": 1.5, "strength": 0.5})
    assert made.status_code == 200, made.text
    stream_job(client, made.json()["job_id"])
    answer = preview(client, done["run_id"])
    assert answer["available"] and answer["passes"] == 1
    # A Mass refine of the other, still running: its pass is planned, not missing.
    release, started = holding(fake)
    try:
        queued = client.post("/api/refine/batch", json={"ids": [second.id], "scale": 1.5, "strength": 0.5})
        assert queued.status_code == 200, queued.text
        assert started.wait(5)
        held = preview(client, done["run_id"])
        assert not held["available"] and held["reason"] == "the missing passes are already queued"
    finally:
        release.set()
    stream_job(client, queued.json()["job_id"])
    fake.after_generate = None
    assert preview(client, done["run_id"])["reason"] == "nothing in this run is missing"


def test_two_specs_in_one_run_are_refused_in_words(client, store, fake):
    retain(store, [cell(1)], metadata={"then_refine": {"scale": 1.5, "strength": 0.4, "steps": None,
                                                       "pass_run_id": "a"}})
    run = retain(store, [cell(2)], metadata={"then_refine": {"scale": 2.0, "strength": 0.35, "steps": None,
                                                             "pass_run_id": "b"}})
    answer = preview(client, run)
    assert not answer["available"]
    assert answer["reason"] == (
        f"this run's passes were made at 1.5× at 0.4 and at 2× at 0.35; Mass refine over run:{run} recovers either"
    )
    refused = client.post(f"/api/runs/{run}/finish", json={})
    assert refused.status_code == 409 and refused.json()["detail"] == answer["reason"]
    assert not fake.seeds


def test_the_pin_changes_when_a_missing_pass_lands(client, store, fake):
    done = then_refine_run(client, fake, runs=2, failing=(2, 4))
    first, _ = rows(store)
    answer = preview(client, done["run_id"])
    assert answer["passes"] == 2
    made = client.post("/api/refine/batch", json={"ids": [first.id], "scale": 1.5, "strength": 0.5})
    stream_job(client, made.json()["job_id"])
    refused = client.post(f"/api/runs/{done['run_id']}/finish", json={"expected": answer["pin"]})
    assert refused.status_code == 409 and "changed" in refused.json()["detail"]
    fresh = preview(client, done["run_id"])
    assert fresh["passes"] == 1 and fresh["pin"] != answer["pin"]
    finish(client, done["run_id"], fresh)


def test_a_source_deleted_before_its_turn_is_a_failed_pass(client, store, fake):
    done = then_refine_run(client, fake, runs=2, failing=(2, 4))
    first, second = rows(store)
    answer = preview(client, done["run_id"])
    assert answer["passes"] == 2

    def delete_second():
        fake.after_generate = None
        store.delete_image(second.id)

    fake.after_generate = delete_second
    accepted, finished = finish(client, done["run_id"], answer)
    assert finished["failures"] == 1
    (failed,) = finished["failed_cells"]
    assert failed["label"] == f"pass of #{second.id}" and "deleted" in failed["error"]
    assert [row.refine for row in rows(store) if row.refine is not None] == [first.id]
    job = main.jobs.get(accepted["job_id"])
    assert job.failed_indices == [] and job.cells == []


def retained_outcome(store, run, job_id):
    """The retained job's outcome, written by a daemon after `done`."""
    deadline = time.time() + 5
    while time.time() < deadline:
        saved = next(saved for saved in store.retained_run_jobs(run) if saved["job_id"] == job_id)
        if saved["outcome"] is not None:
            return saved["outcome"]
        time.sleep(0.02)
    raise AssertionError("the job's outcome was never retained")


def test_a_permanently_deleted_pass_is_redrawn_and_says_so(client, store, fake):
    done = then_refine_run(client, fake)
    assert done["status"] == "done" and done["failures"] == 0
    base, made = rows(store)
    assert made.refine == base.id
    assert retained_outcome(store, done["run_id"], done["job_id"])["completed_passes"] == [base.id]
    assert preview(client, done["run_id"])["reason"] == "nothing in this run is missing"
    # Binned, the pass still counts as made; permanently deleted, Finish
    # draws it again and counts it apart from a pass a stop left missing.
    store.bin_images([made.id])
    assert preview(client, done["run_id"])["reason"] == "nothing in this run is missing"
    store.delete_image(made.id)
    answer = preview(client, done["run_id"])
    assert answer["available"] and (answer["passes"], answer["redrawn_passes"], answer["redrawn"]) == (1, 1, 0)
    accepted, _ = finish(client, done["run_id"], answer)
    assert (accepted["passes"], accepted["redrawn_passes"]) == (1, 1)
    again = [row for row in rows(store) if row.refine == base.id]
    assert len(again) == 1 and again[0].run_id == done["pass_run_id"]


def test_a_pass_a_stop_left_missing_is_not_counted_as_redrawn(client, store, fake):
    done = then_refine_run(client, fake, runs=2, failing=(2,))
    assert done["failures"] == 1
    first, second, made = rows(store)
    assert made.refine == second.id
    assert retained_outcome(store, done["run_id"], done["job_id"])["completed_passes"] == [second.id]
    answer = preview(client, done["run_id"])
    assert (answer["passes"], answer["redrawn_passes"]) == (1, 0)


def test_a_plain_first_job_does_not_hide_a_later_jobs_spec(client, store, fake):
    from backend import then_refine
    from backend.schemas import ThenRefine

    run = "mixed-night"
    spec = ThenRefine(scale=1.5, strength=0.5)
    pass_run = then_refine.pass_run_id(run, spec)
    retain(store, [cell(1, steps=8)], run=run, completed=(0,))
    retain(store, [cell(2, steps=8)], run=run, completed=(0,),
           metadata={"then_refine": {**spec.model_dump(), "pass_run_id": pass_run}})
    plain, refined = store.retained_run_jobs(run)
    assert "then_refine" not in plain["metadata"]
    for item, saved in ((cell(1, steps=8), plain), (cell(2, steps=8), refined)):
        store.save_image(
            Image.new("RGB", (256, 256)), main.backend.applied(item.request), seed=item.seed, duration_ms=1,
            model_id=settings.model_id, quantization=settings.quantization, run_id=run,
            job_id=saved["job_id"], batch=1, variables=item.variables,
        )
    answer = preview(client, run)
    # Only the Then refine job's base has a pass to miss.
    assert answer["available"] and (answer["cells"], answer["passes"]) == (0, 1), answer
    accepted, done = finish(client, run, answer)
    assert accepted["passes"] == 1 and done["pass_run_id"] == pass_run
    (made,) = [row for row in rows(store) if row.refine is not None]
    assert made.run_id == pass_run and made.seed == 2
