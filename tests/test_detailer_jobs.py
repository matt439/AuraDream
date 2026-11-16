"""Dependent detailer passes use the ordinary worker, saves and mask holds."""
from __future__ import annotations

import asyncio
import errno
import threading
import time
from dataclasses import replace

import pytest
from PIL import Image

from backend.detailer import DetailerPlan, DetectorFacts, Region, StoredDetailer, region_mask
from backend import gallery_delete, grid
from backend.grid import Cell
from backend.instruction_edit import descriptor_from_image
from backend.jobs import JobManager
from backend.schemas import GenerateRequest
from backend.seeds import wrap_seed
from tests.fakes import FakeBackend


async def _until(predicate):
    deadline = time.monotonic() + 20
    while not predicate():
        assert time.monotonic() < deadline, "detailer did not reach the expected state"
        await asyncio.sleep(0.005)


async def _done(manager, job):
    async for event in manager.events(job):
        if event["type"] == "done":
            return


def _chain(store):
    with Image.new("RGB", (256, 256), (30, 60, 90)) as image:
        source = store.save_image(
            image, GenerateRequest(prompt="source", width=256, height=256), seed=7,
            duration_ms=1, model_id="black-forest-labs/FLUX.1-Krea-dev",
        )
        proof = descriptor_from_image(source.filename, image)
    regions = [
        Region(id="face-1", kind="face", box=(16, 16, 64, 64), confidence=0.9, detector="fixture"),
        Region(id="face-2", kind="face", box=(128, 16, 176, 64), confidence=0.8, detector="fixture"),
        Region(id="hand-1", kind="hand", box=(32, 128, 80, 176), confidence=0.9, detector="fixture"),
    ]
    masks = []
    for region in regions:
        with region_mask(256, 256, region, 0, 0) as image:
            digest, _ = store.store_mask(image)
            masks.append(digest)
    plan = DetailerPlan(
        source_id=source.id, source=proof,
        detectors=[DetectorFacts(
            id="fixture", runtime="fixture", version="1", asset_revision="1",
            asset_sha256="0" * 64, preprocessing="RGB pixels",
        )],
        candidates=regions, accepted=[region.id for region in regions], masks=masks,
        padding=0, feather=0, threshold=0.5, nms_iou=0.5, base_seed=100,
        recipe={"prompt": "detail", "steps": 2, "strength": 0.75},
    )
    cells = [Cell(
        request=GenerateRequest(
            prompt="detail", width=256, height=256, steps=2, refine=source.id,
            mask=mask, strength=0.75, seed=wrap_seed(plan.base_seed + index),
            detailer=StoredDetailer(plan=plan, pass_index=index),
        ), seed=wrap_seed(plan.base_seed + index), label=f"region {index + 1}/{len(masks)}",
    ) for index, mask in enumerate(masks)]
    return source, plan, cells


def test_regions_use_previous_committed_result_and_keep_every_pass(store):
    source, plan, cells = _chain(store)

    async def scenario():
        backend = FakeBackend()
        manager = JobManager(backend)
        try:
            job = await manager.submit_detailer(cells, detailer=plan)
            await _done(manager, job)
            assert job.status == "done" and job.completed_indices == [0, 1, 2]
            assert [request.refine for request in backend.requests] == [source.id, *[i["id"] for i in job.images[:2]]]
            assert backend.seeds == [100, 101, 102]
            assert job.run_id == plan.group_id
            assert all(image["run_id"] == plan.group_id for image in job.images)
            assert [image["detailer"]["pass_index"] for image in job.images] == [0, 1, 2]
            assert job.snapshot()["detailer"] == {
                "source_id": source.id, "group_id": plan.group_id, "regions": 3,
                "completed": 3, "last_image_id": job.images[-1]["id"], "partial": False,
            }
            # Each subsequent saved result retains its predecessor outside
            # the next mask; the original is still an ordinary gallery row.
            first, last = [store.reference_paths([image["id"]])[0] for image in (job.images[0], job.images[-1])]
            with Image.open(first) as a, Image.open(last) as b:
                assert a.getpixel((24, 24)) == b.getpixel((24, 24))
                assert b.getpixel((240, 240)) == (30, 60, 90)
            assert store.get_image(source.id) is not None
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_first_failed_region_stops_chain_and_reports_partial_result(store):
    _, plan, cells = _chain(store)

    async def scenario():
        backend = FakeBackend()
        backend.fail_on = {2}
        manager = JobManager(backend)
        try:
            job = await manager.submit_detailer(cells, detailer=plan)
            await _done(manager, job)
            assert job.status == "error" and job.stopped_by == "detailer_failure"
            assert len(backend.requests) == 2 and len(job.images) == 1
            assert job.completed_indices == [0] and job.failed_indices == [1]
            assert job.snapshot()["detailer"]["partial"] is True
            assert "1 completed pass kept" in job.error
            assert "new detailer review" in job.error
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_cancel_during_region_keeps_prior_pass_and_skip_is_held(store):
    _, plan, cells = _chain(store)

    async def scenario():
        backend = FakeBackend()
        backend.hold = threading.Event()
        backend.hold_when = lambda request: request.detailer.pass_index == 1
        manager = JobManager(backend)
        try:
            job = await manager.submit_detailer(cells, detailer=plan)
            await _until(backend.holding.is_set)
            with pytest.raises(ValueError, match="cannot be skipped"):
                manager.skip(job.id, job.snapshot()["current_call"])
            manager.cancel(job.id)
            await _done(manager, job)
            assert job.status == "cancelled"
            assert len(job.images) == 1 and len(backend.requests) == 2
            assert not job.failures and not job.skipped_indices
            assert job.snapshot()["detailer"]["partial"] is True
        finally:
            backend.hold.set()
            await manager.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("rewrite_intermediate", [False, True])
def test_pause_lands_between_regions_and_rewritten_intermediate_refuses(store, rewrite_intermediate):
    _, plan, cells = _chain(store)

    async def scenario():
        backend = FakeBackend()
        backend.hold = threading.Event()
        backend.hold_when = lambda request: request.detailer.pass_index == 0
        manager = JobManager(backend)
        try:
            job = await manager.submit_detailer(cells, detailer=plan)
            await _until(backend.holding.is_set)
            manager.pause(job.id, True)
            backend.hold.set()
            await _until(lambda: job.paused)
            assert len(backend.requests) == len(job.images) == 1
            assert job.completed_indices == [0]
            if rewrite_intermediate:
                (path,) = store.reference_paths([job.images[0]["id"]])
                with Image.new("RGB", (256, 256), "red") as changed:
                    changed.save(path)
            manager.pause(job.id, False)
            await _done(manager, job)
            assert job.status == ("error" if rewrite_intermediate else "done")
            assert len(backend.requests) == (1 if rewrite_intermediate else 3)
            if rewrite_intermediate:
                assert "changed" in job.error
        finally:
            backend.hold.set()
            await manager.shutdown()

    asyncio.run(scenario())


def test_queued_chain_holds_all_masks_and_rechecks_original_before_first_call(store):
    source, plan, cells = _chain(store)

    async def scenario():
        backend = FakeBackend()
        manager = JobManager(backend)
        await manager._gate.acquire()
        try:
            job = await manager.submit_detailer(cells, detailer=plan)
            assert manager.masks_in_flight() == set(plan.masks)
            with store._session() as conn:
                assert {row[0] for row in conn.execute("SELECT hash FROM mask_holds WHERE job_id = ?", (job.id,))} == set(plan.masks)
            (path,) = store.reference_paths([source.id])
            with Image.new("RGB", (256, 256), "red") as changed:
                changed.save(path)
        finally:
            manager._gate.release()
        try:
            await _done(manager, job)
            assert job.status == "error" and not backend.requests
            assert "changed" in job.error

            def released():
                with store._session() as conn:
                    return conn.execute("SELECT count(*) FROM mask_holds WHERE job_id = ?", (job.id,)).fetchone()[0] == 0

            await _until(released)
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_disk_full_holds_region_without_rendering_it_again(store, monkeypatch):
    _, plan, cells = _chain(store)
    save = store.save_image
    failed = False

    def no_room_once(image, request, *args, **kwargs):
        nonlocal failed
        if request.detailer.pass_index == 1 and not failed:
            failed = True
            raise OSError(errno.ENOSPC, "No space left on device")
        return save(image, request, *args, **kwargs)

    monkeypatch.setattr(store, "save_image", no_room_once)

    async def scenario():
        backend = FakeBackend()
        manager = JobManager(backend)
        try:
            job = await manager.submit_detailer(cells, detailer=plan)
            await _until(lambda: job.paused and job.pause_reason is not None)
            assert len(job.images) == 1 and len(backend.requests) == 2
            assert not job.failures
            manager.pause(job.id, False)
            await _done(manager, job)
            assert job.status == "done" and len(job.images) == 3
            assert backend.seeds == [100, 101, 102]
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_inconsistent_chain_is_refused_before_retaining_or_admission(store):
    _, plan, cells = _chain(store)

    async def scenario():
        manager = JobManager(FakeBackend())
        try:
            cells[1] = replace(cells[1], seed=999)
            with pytest.raises(ValueError, match="frozen"):
                await manager.submit_detailer(cells, detailer=plan)
            assert not manager.active()
            with store._session() as conn:
                assert conn.execute("SELECT count(*) FROM run_jobs").fetchone()[0] == 0
                assert conn.execute("SELECT count(*) FROM mask_holds").fetchone()[0] == 0
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_corrupt_future_mask_stops_chain_before_that_region_renders(store):
    _, plan, cells = _chain(store)
    with Image.new("L", (256, 256), 255) as changed:
        changed.save(store.mask_path(plan.masks[1]))

    async def scenario():
        backend = FakeBackend()
        manager = JobManager(backend)
        try:
            job = await manager.submit_detailer(cells, detailer=plan)
            await _done(manager, job)
            assert job.status == "error" and len(backend.requests) == 1
            assert len(job.images) == 1 and "mask changed" in job.error
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_individual_replay_holds_full_manifest_after_original_row_is_deleted(store):
    _, plan, cells = _chain(store)
    with Image.new("RGB", (256, 256), (60, 90, 120)) as image:
        saved = store.save_image(
            image, cells[0].request, seed=100, duration_ms=1,
            model_id="black-forest-labs/FLUX.1-Krea-dev", run_id=plan.group_id,
        )
    with store._session() as conn:
        conn.execute("UPDATE masks SET created_at = '2000-01-01T00:00:00+00:00'")

    async def scenario():
        manager = JobManager(FakeBackend())
        await manager._gate.acquire()
        try:
            job = await manager.submit_cells([grid.cell_from_record(saved, seed=200)], grid=False)
            assert job.detailer is None  # an individual ordinary inpaint
            assert manager.masks_in_flight() == set(plan.masks)
            with store._session() as conn:
                held = {row[0] for row in conn.execute("SELECT hash FROM mask_holds WHERE job_id = ?", (job.id,))}
            assert held == set(plan.masks)
            assert gallery_delete.delete_image(saved.id)
            assert gallery_delete.sweep_masks() == 0
            assert all(store.mask_path(digest).is_file() for digest in plan.masks)
        finally:
            manager._gate.release()
        try:
            await _done(manager, job)
            assert job.status == "done" and len(job.images) == 1
            assert job.images[0]["detailer"]["plan"]["masks"] == plan.masks
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_individual_replay_refuses_changed_mask_pixels_before_backend_call(store):
    _, _, cells = _chain(store)
    with Image.new("L", (256, 256), 255) as changed:
        changed.save(store.mask_path(cells[0].request.mask))

    async def scenario():
        backend = FakeBackend()
        manager = JobManager(backend)
        try:
            job = await manager.submit_cells([cells[0]], grid=False)
            await _done(manager, job)
            assert job.status == "done" and job.failures == 1
            assert not backend.requests and not job.images
            assert "mask changed" in job.error
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_daemon_idle_work_keeps_gate_through_cancel_until_worker_unwinds(store):
    async def scenario():
        manager = JobManager(FakeBackend())
        entered, release = threading.Event(), threading.Event()
        worker_threads = []

        def detection():
            worker_threads.append(threading.current_thread())
            entered.set()
            assert release.wait(30)
            return "done"

        pending = asyncio.create_task(manager.run_when_idle(detection, holder="detailer", daemon=True))
        try:
            await _until(entered.is_set)
            assert worker_threads[0].daemon and worker_threads[0].name == "detailer"
            pending.cancel()
            await asyncio.sleep(0)
            assert manager.card() == ("detailer", None) and not pending.done()
            job = await manager.submit(GenerateRequest(prompt="queued", width=256, height=256, steps=1))
            await asyncio.sleep(0)
            assert not manager.backend.requests
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await _done(manager, job)
        assert job.status == "done"
        await manager.shutdown()

    asyncio.run(scenario())
