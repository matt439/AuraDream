"""Frozen region receipts, cold CPU admission, and the real serial queue over HTTP."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from backend import api_detailer, detailer, grid, main, model_selection
from backend.detailer_detectors import DetailerDetectorError
from tests.api_helpers import run_one, stream_job


class Detectors:
    def __init__(self):
        self.calls = []
        self.empty = False
        self.error = False

    def status(self):
        return [{"id": "test", "label": "Test regions", "classes": ["face", "hand"], "available": True}]

    def prepare(self, name, cancel):
        self.calls.append(("prepare", threading.current_thread().daemon))

    def detect(self, name, image, cancel):
        self.calls.append(("detect", threading.current_thread().daemon))
        if self.error:
            raise DetailerDetectorError("Detector bytes are corrupt.")
        facts = [detailer.DetectorFacts(
            id="test", runtime="fake", version="1", asset_revision="1", asset_sha256="a" * 64,
            preprocessing="test RGB CPU",
        )]
        regions = [] if self.empty else [
            detailer.Region(id="hand", kind="hand", box=(96, 64, 176, 144), confidence=0.8, detector="test"),
            detailer.Region(id="face", kind="face", box=(64, 48, 128, 112), confidence=0.9, detector="test"),
        ]
        return regions, facts


@pytest.fixture
def detectors(fake, monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", ("cagliostrolab/animagine-xl-4.0", 1))
    adapter = Detectors()
    monkeypatch.setattr(api_detailer, "detectors_for", lambda services: adapter)
    return adapter


def source(client):
    run_one(client, steps=4)
    return client.get("/api/gallery").json()["items"][0]


def preview(client, record):
    answer = client.post("/api/detailer/preview", json={"source": record["id"], "detector": "test"})
    assert answer.status_code == 200, answer.text
    return answer.json()


def queue(client, reviewed, **extra):
    return client.post("/api/detailer", json={
        "receipt": reviewed["receipt"], "accepted": [r["id"] for r in reviewed["regions"]],
        "recipe": {"prompt": "restored detail", "steps": 4, "strength": 0.6, "seed": 17}, **extra,
    })


def edit(client, reviewed, **change):
    return client.post("/api/detailer/review", json={"receipt": reviewed["receipt"], **change})


def test_status_is_cold_and_detection_uses_daemon_without_loading(client, detectors, fake):
    fake.unload()
    assert client.get("/api/detailer/status").json()["supported"]
    assert detectors.calls == [] and not fake.is_ready()
    # Store the source, then unload to distinguish CPU preview from model work.
    record = source(client)
    assert client.post("/api/model/unload").status_code == 200
    reviewed = preview(client, record)
    assert [r["id"] for r in reviewed["regions"]] == ["face", "hand"]
    assert reviewed["regions"][0]["overlaps"] == ["hand"]
    assert reviewed["regions"][0]["mask_box"] == [48, 32, 144, 128]
    assert detectors.calls == [("detect", True)] and not fake.is_ready()


def test_receipt_queues_one_serial_chain_and_replay_uses_no_detector(client, detectors, fake, store):
    record = source(client)
    reviewed = preview(client, record)
    answer = queue(client, reviewed)
    assert answer.status_code == 200, answer.text
    assert queue(client, reviewed).json()["code"] == "detailer_receipt"
    done = stream_job(client, answer.json()["job_id"])[-1]
    assert done["status"] == "done" and done["detailer"]["completed"] == 2
    rows = sorted((store.get_image(r["id"]) for r in client.get("/api/gallery").json()["items"]
                   if r["detailer"] is not None), key=lambda r: r.id)
    assert len(rows) == 2
    assert rows[0].refine == record["id"] and rows[1].refine == rows[0].id
    assert [row.seed for row in rows] == [17, 18]
    assert rows[0].detailer.plan.group_id == rows[1].detailer.plan.group_id == rows[0].run_id
    # A saved region's ordinary replay never redetects or executes the chain.
    before = len(fake.requests)
    replay = grid.cell_from_record(rows[0])
    response = client.post("/api/generate", json=replay.request.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    assert len(fake.requests) == before + 1 and len(detectors.calls) == 1


def test_empty_detections_and_empty_selection_queue_nothing(client, detectors, fake):
    record = source(client)
    detectors.empty = True
    reviewed = preview(client, record)
    assert reviewed["regions"] == []
    count = len(fake.requests)
    assert queue(client, reviewed).status_code == 422
    assert len(fake.requests) == count and not main.jobs.active()


@pytest.mark.parametrize("detect_first", [False, True])
def test_drawn_region_queues_with_or_without_empty_detection(client, detectors, store, detect_first):
    record = source(client)
    detectors.empty = True
    reviewed = (preview(client, record) if detect_first else
                client.post("/api/detailer/preview", json={"source": record["id"]}).json())
    assert not reviewed["regions"]
    box = [30, 20, 72, 93]
    drawn = edit(client, reviewed, kind="hand", box=box)
    assert drawn.status_code == 200, drawn.text
    answer = queue(client, drawn.json())
    assert answer.status_code == 200, answer.text
    done = stream_job(client, answer.json()["job_id"])[-1]
    assert done["detailer"]["completed"] == 1
    row = store.get_image(done["images"][0]["id"])
    assert row.refine == record["id"]
    plan = row.detailer.plan
    assert plan.version == "detailer-v2"
    assert bool(plan.detectors) is detect_first
    region = plan.accepted_regions[0]
    assert isinstance(region, detailer.DrawnRegion) and list(region.box) == box
    assert row.mask == plan.masks[0]
    assert detectors.calls == ([("detect", True)] if detect_first else [])


def test_corrected_and_drawn_boxes_follow_new_server_order_and_replace_receipt(client, detectors, store):
    record = source(client)
    reviewed = preview(client, record)
    assert edit(client, reviewed, kind="face", box=[0, 3, record["width"] + 1, 40]).status_code == 422
    assert edit(client, reviewed, kind="face", box=["0", 3, 30, 40]).status_code == 422
    corrected = edit(client, reviewed, region_id="face", box=[40, 125, 81, 175]).json()
    assert queue(client, reviewed).json()["code"] == "detailer_receipt"
    drawn = edit(client, corrected, kind="face", box=[8, 10, 38, 43]).json()
    assert [region.get("origin") for region in drawn["regions"]] == ["drawn", "corrected", None]
    assert [region["box"] for region in drawn["regions"]] == [
        [8, 10, 38, 43], [40, 125, 81, 175], [96, 64, 176, 144],
    ]
    answer = queue(client, drawn, accepted=[region["id"] for region in reversed(drawn["regions"])])
    assert answer.status_code == 200, answer.text
    done = stream_job(client, answer.json()["job_id"])[-1]
    rows = [store.get_image(item["id"]) for item in done["images"]]
    plan = rows[0].detailer.plan
    assert plan.accepted == [region["id"] for region in drawn["regions"]]
    assert isinstance(plan.accepted_regions[1], detailer.CorrectedRegion)
    assert plan.accepted_regions[1].detected_box == (64, 48, 128, 112)
    assert [row.mask for row in rows] == plan.masks
    assert [row.refine for row in rows] == [record["id"], rows[0].id, rows[1].id]
    assert detectors.calls == [("detect", True)]


def test_reset_detected_box_and_remove_drawn_box_restore_original_geometry(client, detectors, store):
    reviewed = preview(client, source(client))
    edited = edit(client, reviewed, region_id="face", box=[5, 22, 31, 44]).json()
    drawn = edit(client, edited, kind="hand", box=[20, 20, 30, 44]).json()
    added = next(region["id"] for region in drawn["regions"] if region.get("origin") == "drawn")
    removed = edit(client, drawn, region_id=added).json()
    reset = edit(client, removed, region_id="face").json()
    assert reset["regions"] == reviewed["regions"]
    queued = queue(client, reset)
    assert queued.status_code == 200, queued.text
    done = stream_job(client, queued.json()["job_id"])[-1]
    stored = store.get_image(done["images"][0]["id"]).detailer.plan.model_dump(mode="json")
    assert stored["version"] == "detailer-v1"
    assert all("origin" not in region for region in stored["candidates"])


def test_engine_failure_is_not_an_empty_success(client, detectors):
    record = source(client)
    detectors.error = True
    answer = client.post("/api/detailer/preview", json={"source": record["id"], "detector": "test"})
    assert answer.status_code == 422 and "corrupt" in answer.text
    assert "receipt" not in answer.json()


@pytest.mark.parametrize("change", ["pixels", "bin", "model", "expire"])
def test_stale_review_never_queues(client, detectors, fake, store, monkeypatch, change):
    record = source(client)
    reviewed = preview(client, record)
    if change == "pixels":
        Image.new("RGB", (256, 256), "magenta").save(store.settings.outputs_dir / record["filename"])
    elif change == "bin":
        assert client.post("/api/images/bin", json={"ids": [record["id"]]}).status_code == 200
    elif change == "model":
        model_selection.select("cagliostrolab/animagine-xl-4.0")
    else:
        monkeypatch.setattr(api_detailer, "monotonic", lambda: 1e20)
    made = len(fake.requests)
    answer = queue(client, reviewed)
    assert answer.status_code in (409, 422), answer.text
    assert answer.json()["code"] in ("detailer_stale", "image_binned", "detailer_receipt")
    assert len(fake.requests) == made and not main.jobs.active()


def test_forged_regions_and_unsupported_model_are_refused(client, detectors):
    record = source(client)
    reviewed = preview(client, record)
    assert queue(client, reviewed, accepted=["forged"]).json()["code"] == "detailer_receipt"
    model_selection.select("black-forest-labs/FLUX.1-Krea-dev")
    state = client.get("/api/detailer/status").json()
    assert not state["supported"] and "SDXL" in state["reason"]
    assert client.post("/api/detailer/preview", json={"source": record["id"], "detector": "test"}).status_code == 422


def test_revision_middleware_refuses_before_cpu_work(client, detectors):
    answer = client.post("/api/detailer/preview", json={"source": 1, "detector": "test"},
                         headers={"x-auradream-model-revision": "0"})
    assert answer.status_code == 409 and detectors.calls == []


def test_recipe_correction_keeps_geometry_review_without_redetection(client, detectors):
    reviewed = preview(client, source(client))
    invalid = queue(client, reviewed, recipe={"prompt": "detail", "steps": 4, "strength": 0})
    assert invalid.status_code == 422
    corrected = queue(client, reviewed)
    assert corrected.status_code == 200, corrected.text
    stream_job(client, corrected.json()["job_id"])
    assert len(detectors.calls) == 1


def test_simultaneous_queue_requests_admit_receipt_once(client, detectors, monkeypatch):
    reviewed = preview(client, source(client))
    entered, release = threading.Event(), threading.Event()
    original = api_detailer._mask_hashes

    def held(*args):
        entered.set()
        assert release.wait(30), "test did not release mask staging"
        return original(*args)

    monkeypatch.setattr(api_detailer, "_mask_hashes", held)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(queue, client, reviewed)
        try:
            assert entered.wait(30), "mask staging never started"
            duplicate = queue(client, reviewed)
            assert duplicate.status_code == 409 and duplicate.json()["code"] == "detailer_receipt"
        finally:
            release.set()
        response = first.result(30)
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["detailer"]["completed"] == 2


def test_prepare_filesystem_failure_is_a_worded_refusal(client, detectors, monkeypatch):
    def refused(*args):
        raise PermissionError("cache cannot be written")
    monkeypatch.setattr(detectors, "prepare", refused)
    response = client.post("/api/detailer/prepare", json={"detector": "test"})
    assert response.status_code == 422 and "cache cannot be written" in response.text
