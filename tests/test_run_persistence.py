"""Resolved job recovery survives the worker, and stays local to its gallery."""
from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundles, gallery_lock, grid, main, storage
from backend.jobs import JobManager
from backend.schemas import GenerateRequest, Outpaint
from tests.api_helpers import stream_job
from tests.bundle_helpers import _destination, _save
from tests.fakes import FakeBackend


def definition(cells, *, job_id="saved-job", run_id="saved-run", created_at=1.0):
    request = cells[0].request
    return {
        "job_id": job_id, "run_id": run_id, "created_at": created_at,
        "origin": "comparison", "model_id": "test/model", "quantization": "fp8",
        "width": request.width, "height": request.height, "batch": request.num_images,
        "comparison": True,
        "metadata": {"grid": True, "prompt_template": "a {subject}", "budget_minutes": 17,
                     "swap": None, "refine_batch": None, "more_seeds": None},
    }


def cells(prompt="a lighthouse", *, seed=10):
    return [grid.Cell(GenerateRequest(prompt=prompt, width=256, height=256), seed)]


def retained_outcome(job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = next((row for row in storage.retained_run_jobs() if row["job_id"] == job_id), None)
        if row is not None and row["outcome"] is not None:
            return row["outcome"]
        time.sleep(0.005)
    pytest.fail(f"no retained outcome for {job_id}")


def test_an_older_database_gains_the_job_table_and_index_on_its_next_connection(store):
    with store._session() as conn:
        conn.execute("DROP TABLE run_jobs")
    with store._session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_jobs").fetchone()[0] == 0
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(run_jobs)")}
    assert "idx_run_jobs_run" in indexes


def test_resolved_cells_round_trip_before_the_outpaint_stamp_with_their_job_metadata(store):
    source = _save()
    request = GenerateRequest(
        prompt="a fox 🦊", width=304, height=256, refine=source.id,
        strength=0.5, outpaint=Outpaint(left=32, right=16, band=16),
    )
    raw = [grid.Cell(request, 987, label="variant A", variables={"variant": "A"})]
    meta = definition(raw)
    store.retain_run_job(meta, raw)
    with store._session() as conn:
        encoded = conn.execute("SELECT cells FROM run_jobs").fetchone()[0]
    assert isinstance(encoded, bytes) and encoded.startswith(b"\x1f\x8b")
    assert json.loads(gzip.decompress(encoded))[0]["request"]["mask"] is None
    [restored] = store.retained_run_jobs("saved-run")
    assert restored["cells"] == raw
    assert restored["metadata"] == meta["metadata"]
    assert restored["comparison"] is True and restored["origin"] == "comparison"
    assert restored["outcome"] is None
    [stamped] = store.stamp_outpaints(restored["cells"])
    assert stamped.request.mask is not None and request.mask is None
    assert store.retained_run_jobs("another-run") == []


def test_submitting_an_outpaint_grid_retains_the_unstamped_cells_and_original_budget(client, fake, store):
    source = _save()
    response = client.post("/api/grid", json={
        "prompt": "a {subject}", "prompt_axis": "subject = fox | wolf",
        "width": 304, "height": 256, "refine": source.id, "strength": 0.5,
        "outpaint": {"left": 32, "right": 16, "band": 16},
        "seed": 123, "seeds": 2, "budget_minutes": 17,
    })
    assert response.status_code == 200, response.text
    answer = response.json()
    [retained] = store.retained_run_jobs(answer["run_id"])
    job = main.jobs.get(answer["job_id"])
    assert retained["job_id"] == job.id and retained["origin"] == "grid"
    assert retained["metadata"]["budget_minutes"] == 17
    assert retained["metadata"]["prompt_template"] == "a {subject}"
    assert [cell.seed for cell in retained["cells"]] == [123, 123, 124, 124]
    assert [cell.variables for cell in retained["cells"]] == [{"subject": value} for value in ("fox", "wolf", "fox", "wolf")]
    assert all(cell.request.mask is None and cell.request.outpaint is not None for cell in retained["cells"])
    assert all(cell.request.mask is not None for cell in job.cells)
    assert stream_job(client, job.id)[-1]["status"] == "done"
    assert retained_outcome(job.id)["completed_indices"] == [0, 1, 2, 3]


@pytest.mark.parametrize("keep_favorites", [False, True])
def test_clear_ends_retained_jobs_even_when_it_keeps_favorite_images(store, keep_favorites):
    source = _save()
    store.set_favorite(source.id, True)
    raw = cells()
    store.retain_run_job(definition(raw), raw)
    store.clear_history(keep_favorites=keep_favorites)
    assert store.retained_run_jobs() == []
    assert (store.get_image(source.id) is not None) is keep_favorites


def test_outcome_updates_only_once_and_never_recreates_a_cleared_definition(store):
    raw = cells()
    store.retain_run_job(definition(raw), raw)
    store.finish_run_job("saved-job", {"status": "cancelled", "reached": 0})
    store.finish_run_job("saved-job", {"status": "error", "reached": 1})
    assert store.retained_run_jobs()[0]["outcome"] == {"status": "cancelled", "reached": 0}
    store.clear_history()
    store.finish_run_job("saved-job", {"status": "done", "reached": 1})
    assert store.retained_run_jobs() == []


def test_a_late_outcome_uses_its_original_database_without_recreating_a_missing_one(store, monkeypatch, tmp_path):
    raw = cells()
    store.retain_run_job(definition(raw), raw)
    original = store.settings.db_path
    replacement = dataclasses.replace(store.settings, db_path=tmp_path / "other.db")
    monkeypatch.setattr(storage, "settings", replacement)
    store.init_db()
    store.finish_run_job("saved-job", {"status": "done"}, database=original)
    assert store.retained_run_jobs() == []
    with closing(sqlite3.connect(original)) as conn:
        assert json.loads(conn.execute("SELECT outcome FROM run_jobs").fetchone()[0]) == {"status": "done"}
    missing = tmp_path / "gone.db"
    with pytest.raises(sqlite3.OperationalError):
        store.finish_run_job("saved-job", {"status": "done"}, database=missing)
    assert not missing.exists()


def test_a_job_interrupted_before_finish_keeps_a_null_outcome_for_a_new_manager(store):
    async def interrupted():
        async def loading():
            await asyncio.Event().wait()

        manager = JobManager(FakeBackend(), loading)
        job = await manager.submit(GenerateRequest(prompt="a night", width=256, height=256, runs=3, seed=123))
        assert store.retained_run_jobs(job.run_id)[0]["outcome"] is None
        return job.run_id

    # Closing this event loop cancels its parked load without a terminal
    # transition, as an abrupt process loss cannot call _finish.
    run_id = asyncio.run(interrupted())
    restarted = JobManager(FakeBackend())
    assert restarted.active() == []
    [saved] = store.retained_run_jobs(run_id)
    assert saved["outcome"] is None
    assert [cell.seed for cell in saved["cells"]] == [123, 124, 125]


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"), sqlite3.OperationalError("disk I/O error")])
def test_a_cell_recording_failure_refuses_before_admission_and_releases_mask_holds(
    client, fake, store, monkeypatch, error,
):
    source = _save()
    digest, _ = store.store_mask(Image.new("L", (256, 256), 255))

    def fail(*args):
        with store._session() as conn:
            assert conn.execute("SELECT COUNT(*) FROM mask_holds").fetchone()[0] == 1
        raise error

    monkeypatch.setattr(storage, "retain_run_job", fail)
    response = client.post("/api/generate", json={
        "prompt": "a night", "width": 256, "height": 256,
        "refine": source.id, "strength": 0.5, "mask": digest,
    })
    assert response.status_code == 503, response.text
    assert "cells could not be recorded; nothing was queued" in response.json()["detail"]
    assert not main.jobs.active() and not fake.requests and store.retained_run_jobs() == []
    with store._session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM mask_holds").fetchone()[0] == 0


def test_job_recording_waits_again_when_the_gallery_remains_reserved(client, fake, store, monkeypatch):
    retain = store.retain_run_job
    calls = []

    def temporarily_busy(*args):
        calls.append(1)
        if len(calls) == 1:
            raise gallery_lock.GalleryBusy("another process still holds the gallery")
        return retain(*args)

    monkeypatch.setattr(storage, "retain_run_job", temporarily_busy)
    response = client.post("/api/generate", json={"prompt": "a night", "width": 256, "height": 256})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    assert stream_job(client, job_id)[-1]["status"] == "done"
    assert len(calls) == 2 and retained_outcome(job_id)["status"] == "done"


@pytest.mark.parametrize("ending,status,reached,failed,completed,stopped", [
    ("done", "done", 3, [], [0, 1, 2], None),
    ("skipped", "done", 3, [1], [0, 2], None),
    ("failures", "error", 3, [0, 1, 2], [], "failures"),
    ("cancelled", "cancelled", 1, [], [0], None),
    ("budget", "cancelled", 0, [], [], "budget"),
    ("load", "error", 0, [], [], None),
    ("shutdown", "cancelled", 0, [], [], "shutdown"),
])
def test_every_terminal_path_records_its_honest_cell_outcome(
    store, ending, status, reached, failed, completed, stopped,
):
    async def scenario():
        backend = FakeBackend()
        if ending in {"skipped", "failures"}:
            backend.fail_on = {2} if ending == "skipped" else {1, 2, 3}

        async def load():
            if ending == "load":
                raise RuntimeError("model access refused")
            if ending == "shutdown":
                await asyncio.Event().wait()

        manager = JobManager(backend, load if ending in {"load", "shutdown"} else None)
        raw = [grid.Cell(GenerateRequest(prompt="a night", width=256, height=256), 100 + i) for i in range(3)]
        job = await manager.submit_cells(raw, budget_minutes=0.000001 if ending == "budget" else None)
        if ending == "cancelled":
            backend.after_generate = lambda: setattr(job, "cancel_requested", True)
        if ending == "shutdown":
            manager.stop_all("shutdown")
        async for event in manager.events(job):
            if event["type"] == "done":
                break
        await manager.shutdown()
        return job.id

    outcome = retained_outcome(asyncio.run(scenario()))
    assert outcome["status"] == status
    assert outcome["reached"] == reached
    assert outcome["failed_indices"] == failed
    assert outcome["completed_indices"] == completed
    assert outcome["stopped_by"] == stopped
    assert outcome["finished_at"] > 0


def test_a_finished_job_does_not_wait_for_its_outcome_write(client, fake, store, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    finish = store.finish_run_job

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return finish(*args, **kwargs)

    monkeypatch.setattr(storage, "finish_run_job", slow)
    try:
        response = client.post("/api/generate", json={"prompt": "a night", "width": 256, "height": 256})
        job_id = response.json()["job_id"]
        assert stream_job(client, job_id)[-1]["status"] == "done"
        assert entered.wait(5)
        assert store.retained_run_jobs()[0]["outcome"] is None
    finally:
        release.set()
    assert retained_outcome(job_id)["status"] == "done"


def test_an_outcome_write_failure_is_logged_and_leaves_null_without_failing_the_job(
    client, fake, store, monkeypatch, caplog,
):
    attempted = threading.Event()

    def fail(*args, **kwargs):
        attempted.set()
        raise OSError("disk stopped answering")

    monkeypatch.setattr(storage, "finish_run_job", fail)
    response = client.post("/api/generate", json={"prompt": "a night", "width": 256, "height": 256})
    assert stream_job(client, response.json()["job_id"])[-1]["status"] == "done"
    assert attempted.wait(5)
    assert store.retained_run_jobs()[0]["outcome"] is None
    deadline = time.monotonic() + 5
    while "could not record the outcome" not in caplog.text and time.monotonic() < deadline:
        time.sleep(0.005)
    assert "could not record the outcome" in caplog.text


@pytest.mark.parametrize("picked", [False, True])
def test_bundles_exclude_retained_definitions_and_missing_cell_prompts_but_backup_keeps_local_state(
    store, tmp_path, picked,
):
    source = _save()
    raw = cells("private missing-cell prompt")
    store.retain_run_job(definition(raw), raw)
    result = bundles.export_bundle(keep_gallery=True, ids=[source.id] if picked else None)
    assert len(store.retained_run_jobs()) == 1
    with ZipFile(result["path"]) as archive:
        data = archive.read("database.sqlite3")
    snapshot = tmp_path / "archive.sqlite3"
    snapshot.write_bytes(data)
    with closing(sqlite3.connect(snapshot)) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'run_jobs'").fetchone() is None
    # The compressed requests must not linger on SQLite's freed pages.
    with store._session() as conn:
        encoded = conn.execute("SELECT cells FROM run_jobs").fetchone()[0]
    assert encoded not in data


def test_import_preserves_local_definitions_but_successful_bundle_clear_ends_them(store, monkeypatch, tmp_path):
    _save()
    raw = cells()
    store.retain_run_job(definition(raw), raw)
    bundle = Path(bundles.export_bundle(expected=1)["path"])
    assert store.retained_run_jobs() == []
    _destination(store, monkeypatch, tmp_path)
    local = cells("a different local run")
    store.retain_run_job(definition(local, job_id="local-job", run_id="local-run"), local)
    assert bundles.import_bundle(bundle)["imported"] == 1
    [retained] = store.retained_run_jobs()
    assert retained["run_id"] == "local-run" and retained["cells"] == local


def test_failed_bundle_cleanup_keeps_definitions_for_its_retained_images(store, monkeypatch):
    _save()
    raw = cells()
    store.retain_run_job(definition(raw), raw)

    def locked(_path):
        raise PermissionError("the image is locked")

    monkeypatch.setattr(bundles, "_remove_file", locked)
    result = bundles.export_bundle(expected=1)
    assert result["retained"] and not result["cleared"]
    assert len(store.retained_run_jobs()) == 1
