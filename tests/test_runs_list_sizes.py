"""GAL#30: the Runs list reads every listed run's sizes in its own session,
from the retained requests' raw JSON, never a session and a validation per
run; a run whose retained row cannot be read takes its images' sizes."""
from __future__ import annotations

import gzip

from PIL import Image

from backend import gallery_reports, grid, storage
from backend.schemas import GenerateRequest
from tests.test_run_persistence import definition

RUNS = 30


def plant(run_id: str, *, saved=(256, 256), retained=((256, 256), (512, 256)), seed=1):
    request = GenerateRequest(prompt=f"a fox in {run_id}", width=saved[0], height=saved[1])
    storage.save_image(Image.new("RGB", saved), request, seed=seed, duration_ms=1,
                       model_id="test/model", job_id=f"job-{run_id}", run_id=run_id)
    cells = [grid.Cell(GenerateRequest(prompt="a fox", width=w, height=h), seed + i)
             for i, (w, h) in enumerate(retained)]
    storage.retain_run_job(definition(cells, job_id=f"job-{run_id}", run_id=run_id, created_at=float(seed)), cells)


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
