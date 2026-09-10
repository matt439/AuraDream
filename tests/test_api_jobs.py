"""The job stream and cancellation over HTTP: every image reported, every
stream terminated, cancel and pause landing where they should."""
from __future__ import annotations

import asyncio
import time

from backend.jobs import JobManager
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid, run_one, stream_job, wait_for
from tests.fakes import FakeBackend


# ---------------------------------------------------------------- the stream


def test_a_sweep_reports_every_image_and_finishes(client, fake):
    _, events = run_one(client, runs=2, num_images=2, seed=7)
    images = [e for e in events if e["type"] == "image"]
    assert len(images) == 4
    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    # seed + run * batch, then +1 per image in the batch.
    assert [i["image"]["seed"] for i in images] == [7, 8, 9, 10]
    assert fake.seeds == [7, 9]


def test_a_stream_opened_after_the_end_still_terminates(client):
    job_id, _ = run_one(client)
    events = stream_job(client, job_id)  # attach again, job long finished
    assert len(events) == 1 and events[0]["type"] == "done"
    assert events[0]["images"]


def test_active_jobs_are_listed_without_their_images(client, fake):
    fake.step_seconds = 0.02
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 20}
    ).json()["job_id"]
    live = client.get("/api/jobs").json()
    assert [job["job_id"] for job in live] == [job_id]
    assert "images" not in live[0]
    stream_job(client, job_id)
    assert client.get("/api/jobs").json() == []


def test_every_stream_sees_the_terminal_event(store):
    """One shared queue split the events between two readers."""

    async def scenario():
        manager = JobManager(FakeBackend(step_seconds=0.01))
        job = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=4, num_images=2)
        )

        async def collect(stream):
            seen = []
            async for event in stream:
                seen.append(event)
                if event["type"] == "done":
                    return seen
            return seen

        return await asyncio.gather(
            collect(manager.events(job)), collect(manager.events(job))
        )

    first, second = asyncio.run(scenario())
    for seen in (first, second):
        assert seen[-1]["type"] == "done"
        assert len(seen[-1]["images"]) == 2


# --------------------------------------------------------------- cancellation


def test_cancel_mid_sweep_keeps_what_finished(client, fake):
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 5, "runs": 8}
    ).json()["job_id"]
    # Poll for the first image rather than sleeping a guessed interval.
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    events = stream_job(client, job_id)
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1 and done[0]["status"] == "cancelled"
    # Landed within a step: fewer pipeline calls than runs, and everything that
    # finished before it is kept and reported.
    assert 1 <= len(done[0]["images"]) < 8
    assert len(fake.requests) < 8
    assert client.get("/api/gallery").json()["total"] == len(done[0]["images"])


def test_a_job_pauses_between_cells_and_hands_the_card_back(client, fake):
    # One person, one GPU: the card eight hours into a grid is the card they
    # want for twenty minutes. The hold lands at the cell boundary, the
    # worker acknowledges it, the allocator reserve is returned, and a resume
    # carries on from the next cell.
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/grid", json={"prompt": "x", "width": 256, "height": 256, "steps": 4, "steps_axis": "4,5,6,7,8,9", "seed": 1},
    ).json()["job_id"]
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    assert client.post(f"/api/jobs/{job_id}/pause").json()["pause_requested"] is True
    held = wait_for(client, job_id, lambda s: s["paused"])
    started = len(fake.requests)
    time.sleep(0.15)
    assert len(fake.requests) == started  # nothing new starts while held
    assert held["status"] == "running" and fake.cache_freed == 1
    assert client.get("/api/jobs").json()[0]["paused"] is True
    # Unload is still refused: the weights are needed for the resume.
    assert client.post("/api/model/unload").status_code == 409

    assert client.post(f"/api/jobs/{job_id}/pause?value=false").json()["pause_requested"] is False
    events = stream_job(client, job_id)
    assert events[-1]["status"] == "done" and len(events[-1]["images"]) == 6
    assert events[-1]["paused"] is False


def test_cancel_lands_on_a_paused_job(client, fake):
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/grid", json={"prompt": "x", "width": 256, "height": 256, "steps": 4, "steps_axis": "4,5,6,7", "seed": 1},
    ).json()["job_id"]
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    client.post(f"/api/jobs/{job_id}/pause")
    wait_for(client, job_id, lambda s: s["paused"])
    started = time.perf_counter()
    client.post(f"/api/jobs/{job_id}/cancel")
    done = stream_job(client, job_id)[-1]
    # A polled hold, not a bare Event.wait(): the cancel lands within the poll.
    assert done["status"] == "cancelled" and time.perf_counter() - started < 2.0
    assert 1 <= len(done["images"]) < 4


def test_cancelling_a_queued_job_ends_it_at_once(client, fake):
    fake.step_seconds = 0.02
    ahead = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 40}
    ).json()["job_id"]
    queued = client.post(
        "/api/generate", json={"prompt": "y", "width": 256, "height": 256}
    ).json()["job_id"]
    started = time.perf_counter()
    assert client.post(f"/api/jobs/{queued}/cancel").json()["status"] == "cancelled"
    events = stream_job(client, queued)
    # One terminal event, delivered without waiting for the job ahead.
    assert [e["type"] for e in events] == ["done"]
    assert time.perf_counter() - started < 0.5
    stream_job(client, ahead)
    assert [r.prompt for r in fake.requests] == ["x"]


def test_a_cancel_after_the_last_step_does_not_relabel_a_complete_run(client, fake):
    # The flag is polled between steps; a cancel landing during the decode or
    # the encode of the final image changes nothing and used to still flip the
    # status to "cancelled", so a script treating that as "resume it" re-queued
    # a finished grid.
    job_id = None
    fake.after_generate = lambda: client.post(f"/api/jobs/{job_id}/cancel")
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 2}
    ).json()["job_id"]
    events = stream_job(client, job_id)
    assert events[-1]["status"] == "done"
    assert len(events[-1]["images"]) == 1


def test_one_failed_cell_does_not_end_the_grid(client, fake):
    fake.fail_on = {2}
    _, events = run_grid(client, steps_axis="6,8,10", seed=1)
    done = events[-1]
    assert done["status"] == "done"
    assert done["failures"] == 1
    assert "1 cell(s) failed" in done["error"] and "blew up" in done["error"]
    # The two other cells rendered; the failed one is what resume will find missing.
    assert [r.steps for r in fake.requests] == [6, 8, 10]
    assert sorted(r["steps"] for r in done["images"]) == [6, 10]


def test_three_failures_in_a_row_end_the_job(client, fake):
    fake.fail_on = {1, 2, 3}
    _, events = run_grid(client, steps_axis="6,8,10,12", seed=1)
    assert events[-1]["status"] == "error"
    assert len(fake.requests) == 3
