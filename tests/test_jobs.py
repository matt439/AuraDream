"""The job manager on its own: the terminal-state guard and the status payload."""
from __future__ import annotations

import asyncio
import threading

from backend.jobs import JobManager, short_error
from backend.schemas import GenerateRequest

from tests.fakes import FakeBackend


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


def test_a_queued_job_says_what_it_is_waiting_on(store):
    # "waiting for the job ahead…" for the whole of a 32 GB download, under a
    # top bar saying "loading weights…": the first thing a new user saw was a
    # queue behind work that did not exist. The label is not a status - cancel
    # keys its instant path on `queued` - but a field beside it.
    async def scenario():
        gate = asyncio.Event()
        fake = FakeBackend(loaded=False)

        async def slow_load():
            await gate.wait()
            fake.load()

        manager = JobManager(fake, slow_load)
        # Slow enough that the first job is still running when the second is
        # queued behind it.
        fake.step_seconds = 0.01
        first = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=40))
        await asyncio.sleep(0.02)
        loading = first.snapshot()["waiting_on"]
        gate.set()
        await asyncio.sleep(0.05)
        # Behind a running job, not behind a load.
        second = await manager.submit(GenerateRequest(prompt="y", width=256, height=256, steps=2))
        await asyncio.sleep(0.02)
        queued = second.snapshot()["waiting_on"]
        for job in (first, second):
            async for event in manager.events(job):
                if event["type"] == "done":
                    break
        return loading, queued, first.snapshot()["waiting_on"]

    assert asyncio.run(scenario()) == ("load", "queue", None)


def test_progress_events_carry_the_elapsed_time(store):
    async def scenario():
        manager = JobManager(FakeBackend(step_seconds=0.005))
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=3))
        seen = []
        async for event in manager.events(job):
            if event["type"] == "progress":
                seen.append(event["elapsed"])
            if event["type"] == "done":
                break
        return seen

    elapsed = asyncio.run(scenario())
    assert len(elapsed) == 3 and elapsed == sorted(elapsed) and elapsed[0] >= 0


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


async def _statuses_until(manager, job, predicate, timeout=5.0):
    """The status frames a job's stream carries until one matches."""
    seen = []

    async def follow():
        async for event in manager.events(job):
            if event["type"] == "status":
                seen.append(event["waiting_on"])
                if predicate(event):
                    return
            if event["type"] == "done":
                return

    await asyncio.wait_for(follow(), timeout)
    return seen


def test_a_job_behind_a_paused_one_says_so_and_hears_the_resume(store):
    # A paused job keeps the gate: a job queued behind it from another tab
    # read "waiting for the job ahead…" all night, with nothing saying the
    # job ahead was paused - and when it was resumed, nothing said that
    # either. Its stream is told both, as they happen.
    async def scenario():
        manager = JobManager(FakeBackend(step_seconds=0.005))
        ahead = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=4, runs=4)
        )
        manager.pause(ahead.id, True)
        while not ahead.paused:
            await asyncio.sleep(0.01)
        behind = await manager.submit(GenerateRequest(prompt="y", width=256, height=256, steps=2))
        await asyncio.sleep(0.05)
        paused = behind.snapshot()["waiting_on"]
        following = asyncio.create_task(
            _statuses_until(manager, behind, lambda e: e["waiting_on"] == "queue")
        )
        await asyncio.sleep(0.05)
        manager.pause(ahead.id, False)
        heard = await following
        for job in (ahead, behind):
            async for event in manager.events(job):
                if event["type"] == "done":
                    break
        return paused, heard, behind.snapshot()

    paused, heard, end = asyncio.run(scenario())
    assert paused == "paused"
    # The opening snapshot says "paused"; the resume's frame says "queue".
    assert heard[0] == "paused" and heard[-1] == "queue"
    assert end["status"] == "done" and end["waiting_on"] is None


def test_a_job_behind_an_editor_action_names_it(store):
    # Behind the prompt critic (or Expand) there is no job ahead at all.
    async def scenario():
        manager = JobManager(FakeBackend())
        release = threading.Event()
        critic = asyncio.create_task(manager.run_when_idle(release.wait, holder="critic"))
        while not manager._gate.locked():
            await asyncio.sleep(0)
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=2))
        await asyncio.sleep(0.05)
        waiting = job.snapshot()["waiting_on"]
        release.set()
        await critic
        async for event in manager.events(job):
            if event["type"] == "done":
                break
        held = manager._holder
        await manager.shutdown()
        return waiting, job.status, held

    assert asyncio.run(scenario()) == ("critic", "done", None)


def test_a_shutdown_says_it_stopped_the_job(store):
    # The server stopping ends a job through the cancel path, and a page
    # read that as a Cancel pressed in some other tab. `stopped_by` says it
    # was the shutdown; a user's own cancel leaves it unset.
    async def scenario():
        manager = JobManager(FakeBackend(step_seconds=0.01))
        cancelled = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=50))
        await asyncio.sleep(0.05)
        manager.cancel(cancelled.id)
        async for event in manager.events(cancelled):
            if event["type"] == "done":
                break
        stopped = await manager.submit(GenerateRequest(prompt="y", width=256, height=256, steps=50))
        await asyncio.sleep(0.05)
        await manager.shutdown()
        return cancelled.snapshot(), stopped.snapshot()

    by_user, by_shutdown = asyncio.run(scenario())
    assert by_user["status"] == "cancelled" and by_user["stopped_by"] is None
    assert by_shutdown["status"] == "cancelled" and by_shutdown["stopped_by"] == "shutdown"


def test_a_job_with_nothing_ahead_does_not_wait_for_the_job_ahead(store):
    # Its first frame, sent before it takes the gate, said "queue" on an idle
    # server - "waiting for the job ahead…" with no job ahead.
    async def scenario():
        manager = JobManager(FakeBackend(), _ready)
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=2))
        # Subscribed before the job's task has run at all: every frame it sends.
        frames: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(frames)
        first = await asyncio.wait_for(frames.get(), 5)
        while (await asyncio.wait_for(frames.get(), 5))["type"] != "done":
            pass
        return first

    first = asyncio.run(scenario())
    assert first["type"] == "status" and first["status"] == "queued"
    assert first["waiting_on"] is None


async def _ready():
    return None
