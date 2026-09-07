"""The job manager on its own: the terminal-state guard and the status payload."""
from __future__ import annotations

import asyncio

from backend.jobs import JobManager, short_error
from backend.schemas import GenerateRequest

from tests.test_api import FakeBackend


def test_a_cancelled_queued_job_is_not_rewritten_when_its_load_fails(store):
    # cancel() finishes a still-queued job on the spot, but its _run is parked
    # in ensure_loaded(); when that then raises, the except branch used to
    # rewrite the state to "error" with the event suppressed - the stream said
    # cancelled, the REST snapshot said error.
    async def scenario():
        async def failing_load():
            await asyncio.sleep(0.05)
            raise RuntimeError("403 gated repo")

        manager = JobManager(FakeBackend(), failing_load)
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256))
        await asyncio.sleep(0.01)
        manager.cancel(job.id)
        assert job.status == "cancelled"
        await asyncio.sleep(0.1)  # the load fails in here
        return job

    job = asyncio.run(scenario())
    assert job.status == "cancelled"
    assert job.error is None


def test_per_cell_status_events_leave_the_images_out(store):
    async def scenario():
        manager = JobManager(FakeBackend())
        job = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=2, runs=3)
        )
        seen = []
        async for event in manager.events(job):
            seen.append(event)
            if event["type"] == "done":
                break
        return seen

    events = asyncio.run(scenario())
    statuses = [e for e in events if e["type"] == "status"]
    # The opening snapshot carries the list (a reattached client needs it);
    # the per-cell events do not - they used to re-send every record so far,
    # quadratic over a night.
    assert "images" in statuses[0]
    assert all("images" not in e for e in statuses[1:])
    assert len(statuses) > 1
    assert len(events[-1]["images"]) == 3


def test_short_error_keeps_the_remedy_at_the_end():
    remedy = "try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation."
    message = "CUDA out of memory. " + "Tried to allocate lots. " * 30 + remedy
    short = short_error(RuntimeError(message))
    assert len(short) < 400
    assert short.endswith(remedy[-40:])
    assert short.startswith("RuntimeError: CUDA out of memory")
