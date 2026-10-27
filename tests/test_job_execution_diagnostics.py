"""Job observations retain scalar first-call facts, never a model owner."""
from __future__ import annotations

import gc
import threading
import weakref

import pytest

from backend import main
from backend.backends.base import Cancelled
from backend.backends.regional_compile import RegionalDiagnosticOwner
from backend.jobs import Job
from backend.schemas import GenerateRequest
from tests.api_helpers import stream_job


def _owner(load_id=1):
    owner = RegionalDiagnosticOwner()
    owner.publish(load_id)
    return owner


def _job():
    return Job("observed", GenerateRequest(prompt="fox", width=256, height=256, seed=1))


def test_pure_job_freezes_first_call_then_ignores_retirement_and_replacement():
    job, owner = _job(), _owner(4)
    assert job.bind_execution_diagnostics(owner.snapshot)
    assert job.snapshot()["execution_diagnostics"]["first_call"] == "pending"
    token = owner.begin_first_call()
    assert job.snapshot()["execution_diagnostics"]["first_call"] == "running"
    owner.finish_first_call(token, "returned")
    job.freeze_execution_diagnostics()
    frozen = job.snapshot()["execution_diagnostics"]
    owner.retire()
    assert not owner.finish_first_call(token, "failed", failure_stage="forward")
    assert job.snapshot()["execution_diagnostics"] == frozen
    assert job._execution_reader is None
    replacement = _owner(5)
    assert job.bind_execution_diagnostics(replacement.snapshot)
    assert job.snapshot()["execution_diagnostics"]["load_id"] == 5
    assert job.snapshot()["execution_diagnostics"]["first_call"] == "pending"


def test_job_rejects_readers_that_could_keep_a_backend_or_model_alive():
    job, owner = _job(), _owner()

    class ModelObserver:
        def snapshot(self):
            return owner.snapshot()

    observer = ModelObserver()
    reference = weakref.ref(observer)
    assert not job.bind_execution_diagnostics(observer.snapshot)
    assert "execution_diagnostics" not in job.snapshot()
    assert not job.bind_execution_diagnostics(lambda: owner.snapshot())
    del observer
    gc.collect()
    assert reference() is None
    assert job._execution_reader is None


def test_job_handles_a_retired_active_reader_without_inventing_an_outcome():
    job, owner = _job(), _owner()
    job.bind_execution_diagnostics(owner.snapshot)
    owner.begin_first_call()
    owner.retire()
    assert "execution_diagnostics" not in job.snapshot()
    job.freeze_execution_diagnostics()
    assert job._execution_reader is None
    assert "execution_diagnostics" not in job.snapshot()


@pytest.mark.parametrize("outcome,status", [("returned", "done"), ("failed", "done"), ("cancelled", "cancelled")])
def test_active_job_observes_running_before_any_step_then_freezes_terminal(
    client, fake, store, monkeypatch, outcome, status,
):
    owner, entered, release = _owner(8), threading.Event(), threading.Event()
    original = fake.generate

    def held(*args, **kwargs):
        token = owner.begin_first_call()
        entered.set()
        try:
            assert release.wait(20), "test did not release its synthetic forward"
            if outcome == "failed":
                raise RuntimeError("synthetic forward failed")
            images = original(*args, **kwargs)
        except BaseException as error:
            owner.finish_first_call(token, "cancelled" if isinstance(error, Cancelled) else "failed", failure_stage="forward")
            raise
        owner.finish_first_call(token, "returned")
        return images

    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: owner.snapshot)
    monkeypatch.setattr(fake, "generate", held)
    response = client.post("/api/generate", json={"prompt": "fox", "width": 256, "height": 256, "steps": 2})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    try:
        assert entered.wait(10), "synthetic job never entered its forward"
        active = client.get(f"/api/jobs/{job_id}").json()
        assert active["step"] == 0
        assert active["execution_diagnostics"]["first_call"] == "running"
        if outcome == "cancelled":
            assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    finally:
        release.set()
    done = stream_job(client, job_id)[-1]
    assert done["status"] == status
    assert done["failures"] == (1 if outcome == "failed" else 0)
    assert done["execution_diagnostics"]["first_call"] == outcome
    frozen = done["execution_diagnostics"]
    owner.retire()
    replacement = _owner(9)
    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: replacement.snapshot)
    assert client.get(f"/api/jobs/{job_id}").json()["execution_diagnostics"] == frozen
    actual = main.jobs.get(job_id)
    assert actual is not None and actual._execution_reader is None
    retained = store.retained_run_jobs(done["run_id"])[0]
    assert "execution_diagnostics" not in retained["metadata"]
    assert all("execution_diagnostics" not in cell.request.model_dump() for cell in retained["cells"])


def test_late_job_cancel_preserves_the_first_returned_forward(client, fake, monkeypatch):
    owner, returned, release = _owner(11), threading.Event(), threading.Event()
    original = fake.generate

    def held_return(*args, **kwargs):
        token = owner.begin_first_call()
        images = original(*args, **kwargs)
        owner.finish_first_call(token, "returned")
        returned.set()
        assert release.wait(20), "test did not release its returned synthetic call"
        return images

    monkeypatch.setattr(fake, "execution_diagnostics_reader", lambda: owner.snapshot)
    monkeypatch.setattr(fake, "generate", held_return)
    response = client.post("/api/generate", json={
        "prompt": "fox", "width": 256, "height": 256, "steps": 2, "runs": 2,
    })
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    try:
        assert returned.wait(10), "synthetic first forward never returned"
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    finally:
        release.set()
    done = stream_job(client, job_id)[-1]
    assert done["status"] == "cancelled" and len(fake.requests) == 1
    assert done["execution_diagnostics"]["first_call"] == "returned"


def test_eager_job_adds_no_diagnostic_frame_fields(client, fake):
    response = client.post("/api/generate", json={"prompt": "fox", "width": 256, "height": 256, "steps": 2})
    assert response.status_code == 200
    frames = stream_job(client, response.json()["job_id"])
    assert all("execution_diagnostics" not in frame for frame in frames)
    assert len(fake.requests) == 1
