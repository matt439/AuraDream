"""Bounded Runs reads, retained sizes and isolation of unreadable jobs."""
from __future__ import annotations

import gzip
import zlib
from datetime import datetime, timezone

import pytest
from PIL import Image

from backend import gallery_reports, grid, main, storage
from backend.schemas import GenerateRequest
from tests.test_run_persistence import definition

RUNS = 30


def plant(run_id: str, *, saved=(256, 256), retained=((256, 256), (512, 256)), seed=1):
    request = GenerateRequest(prompt=f"a fox in {run_id}", width=saved[0], height=saved[1])
    record = storage.save_image(Image.new("RGB", saved), request, seed=seed, duration_ms=1,
                                model_id="test/model", job_id=f"job-{run_id}", run_id=run_id)
    cells = [grid.Cell(GenerateRequest(prompt="a fox", width=w, height=h), seed + i)
             for i, (w, h) in enumerate(retained)]
    storage.retain_run_job(definition(cells, job_id=f"job-{run_id}", run_id=run_id, created_at=float(seed)), cells)
    return record


def retain_empty(run_id, created_at, *, job_id=None, prompt="a fox", size=(256, 256), model_id="test/model"):
    request = GenerateRequest(prompt=prompt, width=size[0], height=size[1])
    cells = [grid.Cell(request, int(created_at))]
    storage.retain_run_job({
        **definition(cells, job_id=job_id or f"job-{run_id}", run_id=run_id, created_at=created_at),
        "model_id": model_id,
    }, cells)


def counted_sessions(monkeypatch) -> list[int]:
    opened: list[int] = []
    real = storage._session

    def counting(*args, **kwargs):
        opened.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(storage, "_session", counting)
    return opened


def test_the_list_reads_every_runs_sizes_in_one_session_without_validating(store, monkeypatch):
    for index in range(RUNS):
        plant(f"run-{index}", seed=index + 1)
    opened = counted_sessions(monkeypatch)
    validated: list[object] = []
    real_validate = GenerateRequest.model_validate
    monkeypatch.setattr(GenerateRequest, "model_validate",
                        classmethod(lambda cls, *a, **k: validated.append(1) or real_validate(*a, **k)))
    listed = gallery_reports.runs(200)
    assert len(listed) == RUNS
    # The listed runs' session and the recovery candidates', however many runs.
    assert len(opened) == 2
    assert validated == []
    # The retained cells name the unfinished 512x256 too.
    assert all(summary.sizes == [(256, 256), (512, 256)] and summary.size_count == 2 for summary in listed)


def test_an_unreadable_retained_row_takes_its_images_sizes_and_spares_the_others(store):
    plant("broken", saved=(320, 256), seed=1)
    plant("intact", seed=2)
    with store._session() as conn:
        conn.execute("UPDATE run_jobs SET cells = ? WHERE run_id = 'broken'", (gzip.compress(b'[{"seed": 1}]'),))
    by_run = {summary.run_id: summary.sizes for summary in gallery_reports.runs(200)}
    assert by_run == {"broken": [(320, 256)], "intact": [(256, 256), (512, 256)]}


def test_a_run_without_a_retained_row_takes_its_images_sizes(store):
    plant("kept", seed=1)
    with store._session() as conn:
        conn.execute("DELETE FROM run_jobs")
    [summary] = gallery_reports.runs(200)
    assert summary.sizes == [(256, 256)] and summary.size_count == 1


def test_both_history_limits_batch_zero_output_jobs(store, monkeypatch):
    for index in range(201):
        retain_empty(f"empty-{index}", index + 1)
    opened = counted_sessions(monkeypatch)
    for limit in (20, 200):
        opened.clear()
        summaries = gallery_reports.runs(limit, model_id="test/model")
        assert len(opened) == 2
        assert [run.run_id for run in summaries] == [f"empty-{i}" for i in range(200, 200 - limit, -1)]
        assert [run.seed for run in summaries] == list(range(201, 201 - limit, -1))
        assert all(run.images == 0 and run.jobs == 1 for run in summaries)


def test_zero_output_summary_values_scope_and_bin_match_live_history(store):
    retain_empty("several-jobs", 1)
    retain_empty("several-jobs", 3, job_id="job-several-jobs-2", size=(512, 256), prompt="a grey fox")
    storage.finish_run_job("job-several-jobs-2", {"finished_at": 5})
    retain_empty("one-job", 2)
    retain_empty("foreign", 4, model_id="foreign/model")
    storage.bin_images([plant("binned").id])
    plant("live")

    assert [run.run_id for run in gallery_reports.runs(20)] == ["live", "foreign", "one-job", "several-jobs"]
    runs = gallery_reports.runs(20, model_id="test/model")
    assert [run.run_id for run in runs] == ["live", "one-job", "several-jobs"]
    summary = runs[-1]
    assert (summary.images, summary.distinct, summary.favorites, summary.prompts, summary.jobs) == (0, 0, 0, 2, 2)
    assert (summary.prompt, summary.seed, summary.model_id, summary.quantization) == ("a {subject}", 1, "test/model", "fp8")
    assert (summary.width, summary.height, summary.sizes, summary.size_count) == (256, 256, [(256, 256), (512, 256)], 2)
    assert summary.started_at == datetime.fromtimestamp(1, timezone.utc).isoformat()
    assert summary.finished_at == datetime.fromtimestamp(5, timezone.utc).isoformat()


def test_invalid_deflate_spares_the_runs_list_and_refuses_direct_recovery(client, store, fake):
    plant("damaged-live", saved=(320, 256))
    plant("healthy-live")
    retain_empty("damaged-empty", 1)
    retain_empty("damaged-empty", 2, job_id="healthy-but-same-run")
    retain_empty("healthy-empty", 3)

    damaged = bytearray(gzip.compress(b"[]", mtime=0))
    damaged[10] |= 0b110  # Reserved deflate block type, after the valid gzip header.
    with pytest.raises(zlib.error):
        gzip.decompress(damaged)
    with store._session() as conn:
        conn.execute("UPDATE run_jobs SET cells = ? WHERE job_id IN (?, ?)", (
            bytes(damaged), "job-damaged-live", "job-damaged-empty",
        ))

    response = client.get("/api/runs?all_models=true")
    assert response.status_code == 200, response.text
    by_run = {run["run_id"]: run["sizes"] for run in response.json()}
    assert by_run == {
        "healthy-live": [[256, 256], [512, 256]],
        "damaged-live": [[320, 256]],
        "healthy-empty": [[256, 256]],
    }
    for run_id in ("damaged-live", "damaged-empty"):
        path = f"/api/runs/{run_id}"
        for refused in (client.get(path), client.post(path + "/finish/preview"), client.post(path + "/finish")):
            assert refused.status_code == 422, refused.text
            assert refused.json()["detail"] == f"the saved cells of job job-{run_id} cannot be read by this build"
    assert not fake.requests and not main.jobs.active()
