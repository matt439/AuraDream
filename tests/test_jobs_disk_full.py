"""A full disk under a running job (GEN#52).

Before the pause, `jobs._save` retried only on `GalleryBusy`: `ENOSPC` or
SQLite's "database or disk is full" from `storage.save_image` was a failed
cell, and the job ended after three in a row with three renders thrown away.
Now the render is held in memory and the job pauses with the reason; every
other save error is still the cell's failure, under the same rule.
"""
from __future__ import annotations

import asyncio
import errno
import sqlite3
import time

import pytest
from PIL import Image

from backend import jobs
from backend.jobs import JobManager
from backend.schemas import GenerateRequest
from tests.fakes import FakeBackend


def _enospc() -> OSError:
    return OSError(errno.ENOSPC, "No space left on device")


def _sqlite_full() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("database or disk is full")


FULL = pytest.mark.parametrize("full", [_enospc, _sqlite_full], ids=["enospc", "sqlite-full"])


async def _until(predicate):
    # Seconds of margin for a loaded CI runner (tests/test_jobs.py).
    deadline = time.monotonic() + 15
    while not predicate():
        assert time.monotonic() < deadline, "the job did not reach the expected state"
        await asyncio.sleep(0.005)


async def _done(manager, job):
    async for event in manager.events(job):
        if event["type"] == "done":
            return


def _parts(store):
    return [p.name for p in store.settings.outputs_dir.rglob("*.part")]


def test_what_counts_as_a_full_disk():
    assert jobs.disk_full(_enospc())
    assert jobs.disk_full(_sqlite_full())
    windows = OSError(errno.EIO, "There is not enough space on the disk")
    windows.winerror = 112
    assert jobs.disk_full(windows)
    assert not jobs.disk_full(PermissionError(errno.EACCES, "Permission denied"))
    assert not jobs.disk_full(sqlite3.OperationalError("database is locked"))
    assert not jobs.disk_full(ValueError("No space left on device"))


@FULL
def test_a_save_refused_at_the_row_leaves_no_partial_file(store, monkeypatch, full):
    # The PNG and thumbnail are encoded first: the row's refusal must take
    # both back, or each retry after a full disk would leave a `.png.part`.
    def refuse(*_args, **_kwargs):
        raise full()

    monkeypatch.setattr(store, "_insert_row", refuse)
    with pytest.raises((OSError, sqlite3.OperationalError)):
        store.save_image(
            Image.new("RGB", (64, 64)), GenerateRequest(prompt="x", width=256, height=256, steps=1),
            seed=1, duration_ms=1, model_id="m",
        )
    assert _parts(store) == []
    assert [p for p in store.settings.outputs_dir.rglob("*") if p.is_file()] == []
    assert store.list_images()[0] == []


def test_a_save_refused_at_the_encode_leaves_no_partial_file(store, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise _enospc()

    monkeypatch.setattr(store, "_encode_thumb", refuse)
    with pytest.raises(OSError):
        store.save_image(
            Image.new("RGB", (64, 64)), GenerateRequest(prompt="x", width=256, height=256, steps=1),
            seed=1, duration_ms=1, model_id="m",
        )
    assert _parts(store) == []
    assert store.list_images()[0] == []


@FULL
def test_a_full_disk_pauses_the_job_holding_the_render(store, monkeypatch, full):
    real = store.save_image
    disk = {"full": True, "refused": 0}

    def save(*args, **kwargs):
        if disk["full"]:
            disk["refused"] += 1
            raise full()
        return real(*args, **kwargs)

    monkeypatch.setattr(store, "save_image", save)

    async def scenario():
        fake = FakeBackend()
        manager = JobManager(fake)
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=2, runs=4, seed=5))
        await _until(lambda: job.paused)
        held = job.snapshot()
        renders = len(fake.requests)
        # Resume with the disk still full: the same render is tried once
        # more and the job pauses again, nothing rendered a second time.
        refused = disk["refused"]
        manager.pause(job.id, False)
        await _until(lambda: disk["refused"] > refused and job.paused)
        assert len(fake.requests) == renders == 1
        disk["full"] = False
        manager.pause(job.id, False)
        await _done(manager, job)
        return job, held, fake

    job, held, fake = asyncio.run(scenario())
    assert held["status"] == "running" and held["paused"] and held["pause_requested"]
    assert held["pause_reason"] == jobs.DISK_FULL_PAUSE
    assert held["failures"] == 0 and held["run"] == 1
    assert job.status == "done" and job.failures == 0 and job.stopped_by is None
    assert job.pause_reason is None and not job.paused
    # Four cells, four renders, four images: the held render was the one saved.
    assert len(fake.requests) == 4 and [image["seed"] for image in job.images] == [5, 6, 7, 8]
    assert _parts(store) == []


@FULL
def test_a_cancel_ends_a_job_paused_on_a_full_disk(store, monkeypatch, full):
    def refuse(*_args, **_kwargs):
        raise full()

    monkeypatch.setattr(store, "save_image", refuse)

    async def scenario():
        manager = JobManager(FakeBackend())
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=2, runs=3))
        await _until(lambda: job.paused)
        manager.cancel(job.id)
        await _done(manager, job)
        return job

    job = asyncio.run(scenario())
    assert job.status == "cancelled" and job.failures == 0 and job.failed_cells == []
    assert job.pause_reason is None and not job.paused


def test_a_users_pause_carries_no_reason(store):
    async def scenario():
        manager = JobManager(FakeBackend(step_seconds=0.01))
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=2, runs=3))
        manager.pause(job.id, True)
        await _until(lambda: job.paused)
        held = job.snapshot()
        manager.pause(job.id, False)
        await _done(manager, job)
        return held, job

    held, job = asyncio.run(scenario())
    assert held["paused"] and held["pause_reason"] is None
    assert job.status == "done"


def test_any_other_save_error_is_still_the_cells_failure(store, monkeypatch):
    # Today's rule, kept: three failed cells in a row end the job, the
    # error the last one's.
    calls = []

    def refuse(*_args, **_kwargs):
        calls.append(1)
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(store, "save_image", refuse)

    async def scenario():
        manager = JobManager(FakeBackend())
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=2, runs=5))
        await _done(manager, job)
        return job

    job = asyncio.run(scenario())
    assert job.status == "error" and job.stopped_by == "failures"
    assert job.failures == 3 and len(calls) == 3 and job.failed_indices == [0, 1, 2]
    assert not job.paused and job.pause_reason is None


def _full_from(store, monkeypatch, nth):
    """`save_image` refusing as a full disk from its `nth` call (1-based)
    until `disk["full"]` is cleared."""
    real = store.save_image
    disk = {"full": True, "calls": 0, "refused": 0}

    def save(*args, **kwargs):
        disk["calls"] += 1
        if disk["full"] and disk["calls"] >= nth:
            disk["refused"] += 1
            raise _enospc()
        return real(*args, **kwargs)

    monkeypatch.setattr(store, "save_image", save)
    return disk


def test_a_batch_keeps_what_it_saved_and_holds_the_rest(store, monkeypatch):
    # The first image of a batch of two is on the disk when the second is
    # refused: Resume saves the second under its own seed, the first once.
    disk = _full_from(store, monkeypatch, 2)

    async def scenario():
        fake = FakeBackend()
        manager = JobManager(fake)
        job = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=2, num_images=2, seed=40)
        )
        await _until(lambda: job.paused)
        held = job.snapshot()
        disk["full"] = False
        manager.pause(job.id, False)
        await _done(manager, job)
        return job, held, fake

    job, held, fake = asyncio.run(scenario())
    assert held["pause_reason"] == jobs.DISK_FULL_PAUSE and held["image_count"] == 1
    assert job.status == "done" and job.failures == 0 and len(fake.requests) == 1
    assert [image["seed"] for image in job.images] == [40, 41]
    assert [row.seed for row in sorted(store.list_images()[0], key=lambda row: row.id)] == [40, 41]
    assert job.completed_indices == [0]


def test_a_cancel_on_a_full_disk_keeps_the_batchs_saved_image(store, monkeypatch):
    _full_from(store, monkeypatch, 2)

    async def scenario():
        manager = JobManager(FakeBackend())
        job = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=2, num_images=2, seed=40)
        )
        await _until(lambda: job.paused)
        manager.cancel(job.id)
        await _done(manager, job)
        return job

    job = asyncio.run(scenario())
    assert job.status == "cancelled" and job.failures == 0
    assert [image["seed"] for image in job.images] == [40]
    assert [row.seed for row in store.list_images()[0]] == [40]


def test_a_then_refine_pass_is_held_on_a_full_disk(store, monkeypatch):
    # The base is saved; its pass's save is the one refused. The pass is
    # held, not failed, and saved under the pass run at Resume.
    from backend.schemas import ThenRefine

    disk = _full_from(store, monkeypatch, 2)

    async def scenario():
        fake = FakeBackend()
        manager = JobManager(fake)
        job = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=4, runs=2, seed=60),
            then_refine=ThenRefine(scale=1.5, strength=0.5),
        )
        await _until(lambda: job.paused)
        held = job.snapshot()
        renders = len(fake.requests)
        disk["full"] = False
        manager.pause(job.id, False)
        await _done(manager, job)
        return job, held, renders, fake

    job, held, renders, fake = asyncio.run(scenario())
    assert held["pause_reason"] == jobs.DISK_FULL_PAUSE and held["stage"] == "pass"
    assert held["failures"] == 0 and held["image_count"] == 1 and renders == 2
    assert job.status == "done" and job.failures == 0 and job.failed_cells == []
    # Two bases and a pass of each: four renders, none made twice.
    assert len(fake.requests) == 4 and len(job.images) == 4
    assert len(job.completed_passes) == 2
    rows = sorted(store.list_images()[0], key=lambda row: row.id)
    assert [row.refine is not None for row in rows] == [False, True, False, True]
    assert {row.run_id for row in rows if row.refine is not None} == {job.pass_run_id}


def test_sqlites_own_full_error_is_read_and_leaves_the_gallery_usable(store, monkeypatch):
    # The real error, not a stand-in: a database capped at its present
    # pages refuses the row as a full disk does.
    request = GenerateRequest(prompt="x" * 4000, negative_prompt="y" * 2000, width=256, height=256, steps=1)
    real = store._connect
    capped = {"on": True}

    def connect():
        conn = real()
        if capped["on"]:
            pages = conn.execute("PRAGMA page_count").fetchone()[0]
            conn.execute(f"PRAGMA max_page_count = {pages}")
        return conn

    monkeypatch.setattr(store, "_connect", connect)
    with pytest.raises(sqlite3.Error) as refused:
        store.save_image(Image.new("RGB", (64, 64)), request, seed=1, duration_ms=1, model_id="m")
    assert jobs.disk_full(refused.value), refused.value
    assert _parts(store) == []
    assert [p for p in store.settings.outputs_dir.rglob("*") if p.is_file()] == []
    capped["on"] = False
    record = store.save_image(Image.new("RGB", (64, 64)), request, seed=1, duration_ms=1, model_id="m")
    assert [row.id for row in store.list_images()[0]] == [record.id]
