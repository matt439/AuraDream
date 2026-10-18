"""Reruns subtract live work separately from the gallery and admit overlaps
once, including awaited mask/outpaint writes before a job is registered."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from backend import grid, main, storage
from backend.config import settings
from backend.schemas import GenerateRequest, Outpaint
from tests.api_helpers import stream_job, wait_for


ROUTES = ("/api/rerender", "/api/swap-tags", "/api/refine/batch")


def plant(store, seed, *, prompt="1girl, red hair", width=256, height=256, quantization=None, **fields):
    request = GenerateRequest(
        prompt=prompt, width=width, height=height, steps=20, guidance_scale=1.5, **fields,
    )
    return store.save_image(
        Image.new("RGB", (width, height)), request, seed=seed, duration_ms=1,
        model_id=settings.model_id, quantization=quantization,
    )


def payload(route, ids):
    if route == "/api/rerender":
        return {"ids": ids, "width": 384, "height": 256}
    if route == "/api/swap-tags":
        return {"ids": ids, "swaps": [{"find": "red hair", "replace": "silver hair"}]}
    return {"ids": ids, "scale": 1.5, "strength": 0.5}


def accepted(client, route, body):
    response = client.post(route, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def preview(client, route, body):
    response = client.post(f"{route}/preview", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, answer):
    for job in [answer, *answer.get("more", [])]:
        assert stream_job(client, job["job_id"])[-1]["status"] == "done"


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate(), "timed out waiting for concurrent request"


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("waiting", [False, True], ids=["running", "queued"])
def test_repeated_reruns_skip_live_cells_and_partial_overlaps_only_queue_the_rest(
    client, fake, store, route, waiting,
):
    a, b = plant(store, 1), plant(store, 2)
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    blocker = None
    try:
        if waiting:
            blocker = accepted(client, "/api/generate", {
                "prompt": "a different job", "width": 256, "height": 256,
            })
            wait_for(client, blocker["job_id"], lambda job: job["status"] == "running")
        body = payload(route, [a.id])
        first = accepted(client, route, body)
        wait_for(client, first["job_id"], lambda job: job["status"] == ("queued" if waiting else "running"))
        if route != "/api/rerender":
            answer = preview(client, route, body)
            assert answer["counts"] == {"planned": 1}
            assert (answer["queued"], answer["jobs"]) == (0, 0)
            assert answer["images"][0]["status"] == "planned"
            assert "queued or running" in answer["images"][0]["detail"]
            assert "gallery" not in answer["images"][0]["detail"]
        refused = client.post(route, json=body)
        assert refused.status_code == 409 and "queued" in refused.json()["detail"]
        assert "gallery" not in refused.json()["detail"]
        wider = payload(route, [a.id, b.id])
        if route != "/api/rerender":
            answer = preview(client, route, wider)
            assert answer["counts"] == {"planned": 1, "queued": 1}
            wider["expected"] = answer["ids"]
        second = accepted(client, route, wider)
        assert (second["pruned"], second["planned"]) == (0, 1)
        if route != "/api/rerender":
            assert second["queued"] == 1
    finally:
        release.set()
    if blocker:
        finish(client, blocker)
    finish(client, first)
    finish(client, second)
    requests = [request for request in fake.requests if request.prompt != "a different job"]
    assert len(requests) == 2
    if route != "/api/rerender":
        assert preview(client, route, wider)["counts"] == {"rendered": 2}


@pytest.mark.parametrize("route", ROUTES)
def test_gallery_and_live_cells_have_separate_counts_and_all_skipped_words(client, fake, store, route):
    a, b, c = (plant(store, seed) for seed in (1, 2, 3))
    finish(client, accepted(client, route, payload(route, [a.id])))
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        live = accepted(client, route, payload(route, [b.id]))
        body = payload(route, [a.id, b.id])
        if route != "/api/rerender":
            answer = preview(client, route, body)
            assert answer["counts"] == {"rendered": 1, "planned": 1}
        refused = client.post(route, json=body)
        assert refused.status_code == 409
        assert "queued" in refused.json()["detail"]
        assert ("gallery" in refused.json()["detail"] or "rendered" in refused.json()["detail"])
        rest = accepted(client, route, payload(route, [a.id, b.id, c.id]))
        assert (rest["pruned"], rest["planned"]) == (1, 1)
    finally:
        release.set()
    finish(client, live)
    finish(client, rest)
    assert len(fake.requests) == 3


@pytest.mark.parametrize("route,check", [
    ("/api/rerender", False), ("/api/swap-tags", False), ("/api/swap-tags", True),
    ("/api/refine/batch", False), ("/api/refine/batch", True),
], ids=["rerender-submit", "swap-submit", "swap-preview", "refine-submit", "refine-preview"])
def test_completion_during_the_gallery_scan_keeps_the_earlier_queue_snapshot(
    client, fake, store, monkeypatch, route, check,
):
    a = plant(store, 1)
    body = payload(route, [a.id])
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    first = accepted(client, route, body)
    wait_for(client, first["job_id"], lambda job: job["status"] == "running")
    scan = storage.completed_cells

    def racing(*args, **kwargs):
        done = scan(*args, **kwargs)
        release.set()
        wait_until(lambda: main.jobs.get(first["job_id"]).status == "done")
        return done

    monkeypatch.setattr(storage, "completed_cells", racing)
    try:
        if check:
            answer = preview(client, route, body)
            assert answer["counts"] == {"planned": 1} and answer["queued"] == 0
        else:
            refused = client.post(route, json=body)
            assert refused.status_code == 409 and "queued" in refused.json()["detail"]
            assert "gallery" not in refused.json()["detail"]
    finally:
        release.set()
    assert len(fake.requests) == 1


@pytest.mark.parametrize("route", ROUTES)
def test_overlapping_submissions_are_coordinated_across_the_gallery_await(
    client, fake, store, monkeypatch, route,
):
    a, b = plant(store, 1), plant(store, 2)
    scanned, release_scan, release_job = (threading.Event() for _ in range(3))
    fake.after_generate = lambda: release_job.wait(5)
    scan = storage.completed_cells
    calls = []

    def held_scan(*args, **kwargs):
        done = scan(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            scanned.set()
            assert release_scan.wait(5)
        return done

    monkeypatch.setattr(storage, "completed_cells", held_scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(client.post, route, json=payload(route, [a.id]))
            assert scanned.wait(5)
            second = pool.submit(client.post, route, json=payload(route, [a.id, b.id]))
            wait_until(lambda: main._model_requests >= 2)
            # Ordinary Generate can still enter while rerun admission waits.
            ordinary = accepted(client, "/api/generate", {
                "prompt": "a deliberate repeat", "width": 256, "height": 256,
            })
            release_scan.set()
            one, two = first.result(timeout=5), second.result(timeout=5)
            assert one.status_code == two.status_code == 200
            assert two.json()["planned"] == 1 and two.json()["pruned"] == 0
        finally:
            release_scan.set()
            release_job.set()
    finish(client, ordinary)
    finish(client, one.json())
    finish(client, two.json())
    assert len([request for request in fake.requests if request.prompt != "a deliberate repeat"]) == 2


@pytest.mark.parametrize("route,awaited", [
    ("/api/rerender", "hold_masks"), ("/api/swap-tags", "hold_masks"),
    ("/api/rerender", "stamp_outpaints"), ("/api/swap-tags", "stamp_outpaints"),
])
def test_overlapping_masked_reruns_are_coordinated_before_job_admission(
    client, fake, store, monkeypatch, route, awaited,
):
    source = plant(store, 1)
    if awaited == "hold_masks":
        digest, _ = store.store_mask(Image.new("L", (256, 256), 255))
        image = plant(store, 2, refine=source.id, strength=0.5, mask=digest, quantization="none")
    else:
        spec = Outpaint(left=32, right=16, band=16)
        request = GenerateRequest(
            prompt="1girl, red hair", width=304, height=256, steps=20,
            guidance_scale=1.5, refine=source.id, strength=0.5, outpaint=spec,
        )
        (cell,) = store.stamp_outpaints([grid.Cell(request=request, seed=2)])
        image = store.save_image(
            Image.new("RGB", (304, 256)), cell.request, seed=2, duration_ms=1,
            model_id=settings.model_id, quantization="none",
        )
    body = payload(route, [image.id])
    if route == "/api/rerender":
        body.update(width=image.width, height=image.height)
    entered, release_admission, release_job = (threading.Event() for _ in range(3))
    original = getattr(storage, awaited)
    fake.after_generate = lambda: release_job.wait(5)
    calls = []

    def held(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release_admission.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, awaited, held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(client.post, route, json=body)
            assert entered.wait(5)
            assert not main.jobs.active()  # The awaited mask work is before admission.
            second = pool.submit(client.post, route, json=body)
            wait_until(lambda: main._model_requests >= 2)
            # A cold read gets a turn while both HTTP requests are in flight.
            assert client.get("/api/status").status_code == 200
            release_admission.set()
            one, two = first.result(timeout=5), second.result(timeout=5)
            assert one.status_code == 200, one.text
            assert two.status_code == 409 and "queued" in two.json()["detail"]
        finally:
            release_admission.set()
            release_job.set()
    finish(client, one.json())
    assert len(fake.requests) == 1


@pytest.mark.parametrize("route", ROUTES[1:])
def test_mixed_planned_and_invalid_results_name_both_reasons(client, fake, store, route):
    a = plant(store, 1)
    bad = plant(store, 2, lora="missing-adapter")
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        first = accepted(client, route, payload(route, [a.id]))
        body = payload(route, [a.id, bad.id])
        answer = preview(client, route, body)
        assert answer["counts"] == {"planned": 1, "lora_missing": 1}
        body["expected"] = answer["ids"]
        refused = client.post(route, json=body)
        assert refused.status_code == 422
        assert "1 already queued" in refused.json()["detail"]
        assert "1 lora missing" in refused.json()["detail"]
        assert "gallery" not in refused.json()["detail"]
        # `expected` still pins the entire selection, including held cells.
        changed = client.post(route, json={**body, "expected": [a.id]})
        assert changed.status_code == 409 and "selection changed" in changed.json()["detail"]
    finally:
        release.set()
    finish(client, first)


@pytest.mark.parametrize("route", ROUTES)
def test_matching_uses_applied_settings_and_unrelated_reruns_remain_eligible(client, fake, store, route):
    a = plant(store, 1, negative_prompt="ignored from the row")
    fake.cfg_threshold = 5.0
    # An ordinary job can promise the same image as any rerun. Both its
    # negative and the record's different negative disappear in applied().
    body = {
        "prompt": "1girl, red hair", "negative_prompt": "ignored from the job",
        "width": 384, "height": 256, "steps": 20, "guidance_scale": 1.5, "seed": 1,
    }
    rerun = payload(route, [a.id])
    if route == "/api/swap-tags":
        body.update(prompt="1girl, silver hair", width=256)
        unrelated = {**rerun, "swaps": [{"find": "red hair", "replace": "gold hair"}]}
    elif route == "/api/refine/batch":
        body.update(height=384, refine=a.id, strength=0.5)
        unrelated = {**rerun, "strength": 0.6}
    else:
        unrelated = {**rerun, "width": 512}
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        first = accepted(client, "/api/generate", body)
        if route != "/api/rerender":
            assert preview(client, route, rerun)["counts"] == {"planned": 1}
        refused = client.post(route, json=rerun)
        assert refused.status_code == 409 and "queued" in refused.json()["detail"]
        other = accepted(client, route, unrelated)
        assert (other["planned"], other["pruned"]) == (0, 0)
    finally:
        release.set()
    finish(client, first)
    finish(client, other)
    assert len(fake.requests) == 2
    # The fake captures raw requests at its call boundary; saved records
    # use the backend's applied settings, like the matching above.
    assert all(
        image["negative_prompt"] is None
        for answer in (first, other) for image in main.jobs.get(answer["job_id"]).images
    )


@pytest.mark.parametrize("route", ROUTES[:2])
@pytest.mark.parametrize("conditioning", ["mask", "outpaint"])
def test_a_different_mask_or_outpaint_spec_is_a_different_live_cell(
    client, fake, store, route, conditioning,
):
    source = plant(store, 1)
    images = []
    for level in (128, 255):
        if conditioning == "mask":
            digest, _ = store.store_mask(Image.new("L", (256, 256), level))
            image = plant(store, 2, refine=source.id, strength=0.5, mask=digest, quantization="none")
        else:
            spec = Outpaint(left=32, right=16, band=16 if level == 128 else 24)
            request = GenerateRequest(
                prompt="1girl, red hair", width=304, height=256, steps=20,
                guidance_scale=1.5, refine=source.id, strength=0.5, outpaint=spec,
            )
            (cell,) = store.stamp_outpaints([grid.Cell(request=request, seed=2)])
            image = store.save_image(
                Image.new("RGB", (304, 256)), cell.request, seed=2, duration_ms=1,
                model_id=settings.model_id, quantization="none",
            )
        images.append(image)
    first_body, second_body = (payload(route, [image.id]) for image in images)
    if route == "/api/rerender":
        first_body.update(width=images[0].width, height=images[0].height)
        second_body.update(width=images[1].width, height=images[1].height)
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        first = accepted(client, route, first_body)
        if route != "/api/rerender":
            assert preview(client, route, second_body)["counts"] == {"queued": 1}
        second = accepted(client, route, second_body)
        assert (second["planned"], second["pruned"]) == (0, 0)
    finally:
        release.set()
    finish(client, first)
    finish(client, second)
    assert len(fake.requests) == 2
    assert fake.requests[0].mask != fake.requests[1].mask


def test_different_rerun_routes_share_admission_for_the_same_cell(client, fake, store, monkeypatch):
    a = plant(store, 1, prompt="1girl, silver hair")
    b = plant(store, 1, width=384)
    scanned, release_scan, release_job = (threading.Event() for _ in range(3))
    fake.after_generate = lambda: release_job.wait(5)
    scan = storage.completed_cells
    calls = []

    def held_scan(*args, **kwargs):
        done = scan(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            scanned.set()
            assert release_scan.wait(5)
        return done

    monkeypatch.setattr(storage, "completed_cells", held_scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(client.post, "/api/rerender", json=payload("/api/rerender", [a.id]))
            assert scanned.wait(5)
            second = pool.submit(client.post, "/api/swap-tags", json=payload("/api/swap-tags", [b.id]))
            wait_until(lambda: main._model_requests >= 2)
            assert client.get("/api/status").status_code == 200
            release_scan.set()
            one, two = first.result(timeout=5), second.result(timeout=5)
            assert one.status_code == 200, one.text
            assert two.status_code == 409 and "queued" in two.json()["detail"]
        finally:
            release_scan.set()
            release_job.set()
    finish(client, one.json())
    assert len(fake.requests) == 1
