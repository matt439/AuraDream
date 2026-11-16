"""Measured-only Turbo Refine across HTTP admission, planning and recovery.

The source PNGs and backend are CPU stand-ins. These checks establish no card
fit or output quality; real integrated acceptance remains a separate session.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
import uuid

from PIL import Image
import pytest

from backend import config, grid, jobs, main, refine_batch, schemas, schemas_generate, then_refine
from backend.backends import loading
from backend.execution import ExecutionSpec
from backend.model_defaults import defaults_for
from backend.schemas import GenerateRequest
from tests import fakes
from tests.api_helpers import stream_job


TURBO = "circlestone-labs/Anima"
MEASURED_HOST = ("win32", "0.42.1", "NVIDIA GeForce RTX 5080", True)
SETTINGS_READERS = (config, main, jobs, schemas, schemas_generate, fakes, then_refine, refine_batch)


def select(monkeypatch, **changes):
    for module in SETTINGS_READERS:
        monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, **changes))


@pytest.fixture
def turbo(fake, monkeypatch):
    select(monkeypatch, model_id=TURBO, quantization="fp8", offload="model", device="cuda")
    profile = defaults_for(TURBO)
    fake._profile_model = TURBO
    for key in ("cfg_threshold", "true_cfg_supported", "mu_supported", "negative_prompt_supported",
                "reference_supported", "refine_supported", "inpaint_supported"):
        setattr(fake, key, profile[key])
    fake.lora_reason = loading.lora_refusal(TURBO)
    fake.refine_host_facts = MEASURED_HOST
    return fake


def plant(store, *, seed=101, width=1024, height=1024, **fields):
    request = GenerateRequest(**{
        "prompt": "a keeper", "width": width, "height": height, "steps": 8,
        "guidance_scale": 1.0, **fields,
    })
    return store.save_image(
        Image.new("RGB", (width, height), (40, 60, 80)), request,
        seed=seed, duration_ms=1, model_id=TURBO, quantization="fp8",
        run_id=f"fixture-{seed}", batch=1,
    )


def pass_body(source, **changes):
    return {"prompt": "a keeper", "width": 1024, "height": 1024, "steps": 8,
            "guidance_scale": 1.0, "refine": source, "seed": 7001, **changes}


def assert_no_work(fake):
    assert fake.requests == []
    assert fake.loads == 0
    assert not main.jobs.active()


def test_cold_and_loaded_status_share_the_measured_recipe(client, turbo):
    for loaded in (False, True):
        turbo.loaded = loaded
        status = client.get("/api/status").json()
        assert status["defaults"]["refine_supported"] is True
        assert status["defaults"]["refine_settled"] is True
        assert status["defaults"]["refine_strength"] == .25
        assert status["defaults"]["refine_reason"] is None
        assert status["defaults"]["mu_supported"] is False
        assert status["defaults"]["refine_tight_pixels"] == 1024 ** 2
        suggestions = status["refine_strength_suggestions"]
        assert suggestions["steps"] == 8
        assert suggestions["strengths"] == [.125, .25, .4]
        assert "8" in suggestions["steps_reason"]
    assert_no_work(turbo)


@pytest.mark.parametrize("host, words", [
    (("win32", "0.42.1", None, True), "GPU has not been verified"),
    (("win32", "0.42.1", "NVIDIA GeForce RTX 3070", True), "only the RTX 5080"),
    (("win32", "0.42.1", "NVIDIA GeForce RTX 5080", False), "Native attention"),
    (("linux", "0.42.1", "NVIDIA GeForce RTX 5080", True), "only Windows"),
    (("win32", "0.42.2", "NVIDIA GeForce RTX 5080", True), "only 0.42.1"),
])
def test_unqualified_host_is_held_before_load_or_queue(client, turbo, store, host, words):
    source = plant(store)
    turbo.loaded = False
    turbo.refine_host_facts = host
    status = client.get("/api/status").json()
    assert status["defaults"]["refine_supported"] is False
    assert words in status["defaults"]["refine_reason"]
    for route in ("/api/generate", "/api/grid", "/api/grid/preview"):
        response = client.post(route, json=pass_body(source.id))
        assert response.status_code == 422, response.text
        assert words in response.json()["detail"]
    assert_no_work(turbo)


@pytest.mark.parametrize("choice, words", [
    ({"quantization": "int8"}, "only fp8"),
    ({"offload": "sequential"}, "only model offload"),
    ({"device": "cpu"}, "only CUDA device 0"),
])
def test_unmeasured_loader_does_not_turn_a_refine_into_generation(client, turbo, store, monkeypatch, choice, words):
    source = plant(store)
    select(monkeypatch, **choice)
    turbo.loaded = False
    response = client.post("/api/generate", json=pass_body(source.id))
    assert response.status_code == 422, response.text
    assert words in response.json()["detail"]
    assert_no_work(turbo)


def test_refine_qualification_does_not_restrict_ordinary_turbo_generation(client, turbo, monkeypatch):
    select(monkeypatch, quantization="int8", offload="sequential", device="cpu")
    turbo.refine_host_facts = ("linux", "0.42.2", None, False)
    response = client.post("/api/generate", json={"prompt": "ordinary generation", "width": 256, "height": 256})
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 1
    assert done["images"][0]["refine"] is None


@pytest.mark.parametrize("changes, words", [
    ({"width": 1280, "height": 1280}, "1024"),
    ({"width": 1024, "height": 768}, "1024"),
    ({"steps": 9}, "8 steps"),
    ({"guidance_scale": 2}, "CFG 1"),
    ({"strength": .3}, "strengths 0.125, 0.25 and 0.4"),
    ({"crop_anchor": "left"}, "centre crop anchor"),
])
def test_unmeasured_pass_is_refused_without_loading(client, turbo, store, changes, words):
    source = plant(store)
    turbo.loaded = False
    response = client.post("/api/generate", json=pass_body(source.id, **changes))
    assert response.status_code == 422, response.text
    assert words in response.json()["detail"]
    assert_no_work(turbo)


@pytest.mark.parametrize("axis, values, words", [
    ("steps_axis", "8,9", "8 steps"),
    ("cfg_axis", "1,2", "CFG 1"),
    ("strength_axis", "0.25,0.3", "strengths 0.125, 0.25 and 0.4"),
    ("size_axis", "1024x1024,1280x1280", "1024"),
])
def test_every_expanded_grid_cell_is_checked_not_only_the_valid_first(client, turbo, store, axis, values, words):
    source = plant(store)
    turbo.loaded = False
    for route in ("/api/grid/preview", "/api/grid"):
        response = client.post(route, json=pass_body(source.id, **{axis: values}))
        assert response.status_code == 422, response.text
        assert words in response.json()["detail"]
    assert_no_work(turbo)


@pytest.mark.parametrize("strength, count", [(.125, 1), (.25, 2), (.4, 4)])
def test_preview_and_saved_row_use_the_profiled_flow_count(client, turbo, store, strength, count):
    source = plant(store)
    preview = client.post("/api/grid/preview", json=pass_body(source.id, strength=strength, num_images=2))
    assert preview.status_code == 200, preview.text
    assert preview.json()["denoised_steps"] == count
    response = client.post("/api/generate", json=pass_body(source.id, strength=strength, num_images=2))
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["status"] == "done", done
    assert (done["step"], done["total_steps"]) == (count * 2, count * 2)
    assert len(done["images"]) == 2
    for row in done["images"]:
        assert (row["refine"], row["strength"], row["steps"]) == (source.id, strength, 8)
        with Image.open(store.settings.outputs_dir / row["filename"]) as image:
            chunk = json.loads(image.info["auradream"])
        assert chunk["refine"] == source.id and chunk["strength"] == strength


def test_default_strength_and_mixed_preview_do_not_invent_a_count(client, turbo, store):
    source = plant(store)
    default = client.post("/api/grid/preview", json=pass_body(source.id))
    assert default.status_code == 200, default.text
    assert default.json()["values"]["strength"] == [.25]
    assert default.json()["denoised_steps"] == 2
    mixed = client.post("/api/grid/preview", json=pass_body(source.id, strength_axis="0.125,0.25,0.4"))
    assert mixed.status_code == 200, mixed.text
    assert mixed.json()["denoised_steps"] is None
    direct = client.post("/api/grid/preview", json={"prompt": "direct", "width": 1024, "height": 1024})
    assert direct.status_code == 200, direct.text
    assert direct.json()["denoised_steps"] is None
    assert_no_work(turbo)


def test_source_header_must_match_even_when_the_database_still_says_square_1024(client, turbo, store):
    source = plant(store)
    Image.new("RGB", (512, 512), "red").save(store.settings.outputs_dir / source.filename)
    assert store.get_image(source.id).width == 1024
    response = client.post("/api/generate", json=pass_body(source.id))
    assert response.status_code == 422, response.text
    assert "source" in response.json()["detail"] and "512" in response.json()["detail"]
    assert_no_work(turbo)


@pytest.mark.parametrize("queued", [False, True], ids=["admission", "queued-recheck"])
def test_same_size_source_pixel_rewrite_cannot_reuse_the_gallery_proof(client, turbo, store, queued):
    source = plant(store)
    original_hash = source.content_hash
    assert original_hash
    release = threading.Event()
    first = None
    if queued:
        turbo.hold, turbo.hold_when = release, lambda request: request.refine is None
        first = client.post("/api/generate", json={"prompt": "holder", "width": 256, "height": 256, "steps": 8})
        assert first.status_code == 200, first.text
        assert turbo.holding.wait(30)
    try:
        if queued:
            response = client.post("/api/generate", json=pass_body(source.id))
            assert response.status_code == 200, response.text
        # The actual dimensions still match both the row and the measured
        # envelope. Only the decoded RGB pixels invalidate the prior proof.
        with Image.new("RGB", (1024, 1024), (180, 20, 10)) as replacement:
            assert store.content_hash(replacement) != original_hash
            replacement.save(store.settings.outputs_dir / source.filename)
        unchanged = store.get_image(source.id)
        assert (unchanged.width, unchanged.height, unchanged.content_hash) == (1024, 1024, original_hash)
        if not queued:
            response = client.post("/api/generate", json=pass_body(source.id))
            assert response.status_code == 422, response.text
            assert "pixels" in response.json()["detail"] and "provenance" in response.json()["detail"]
            assert_no_work(turbo)
    finally:
        release.set()
    if queued:
        assert first is not None
        stream_job(client, first.json()["job_id"])
        done = stream_job(client, response.json()["job_id"])[-1]
        assert done["status"] == "done" and done["failures"] == 1 and done["images"] == [], done
        assert "pixels" in done["error"] and "provenance" in done["error"]
        assert len(turbo.requests) == 1 and turbo.requests[0].refine is None


def test_mass_refine_keeps_valid_sources_and_reports_unqualified_records(client, turbo, store):
    good = plant(store, seed=201)
    wrong_size = plant(store, seed=202, width=768, height=768)
    wrong_sampling = plant(store, seed=203, guidance_scale=2)
    ids = [good.id, wrong_size.id, wrong_sampling.id]
    body = {"ids": ids, "favorites": False, "scale": 1, "strength": .25}
    response = client.post("/api/refine/batch/preview", json=body)
    assert response.status_code == 200, response.text
    preview = response.json()
    records = {item["id"]: item for item in preview["images"]}
    assert records[good.id]["status"] == "queued"
    assert records[wrong_size.id]["status"] == "invalid" and "1024" in records[wrong_size.id]["detail"]
    assert records[wrong_sampling.id]["status"] == "invalid" and "CFG 1" in records[wrong_sampling.id]["detail"]
    accepted = client.post("/api/refine/batch", json={**body, "expected": preview["ids"]})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["queued"] == 1
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert done["status"] == "done" and [row["refine"] for row in done["images"]] == [good.id]


def test_then_refine_uses_future_base_geometry_not_real_gallery_id_one(client, turbo, store):
    # #1 is deliberately outside the source envelope. The stand-in's id=1
    # must never grant it authority to open this unrelated source.
    unrelated = plant(store, width=512, height=512)
    assert unrelated.id == 1
    response = client.post("/api/generate", json={
        "prompt": "new square source", "width": 1024, "height": 1024,
        "steps": 8, "guidance_scale": 1, "seed": 400,
        "then_refine": {"scale": 1, "strength": .25},
    })
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["status"] == "done", done
    assert len(done["images"]) == 2
    base, passed = done["images"]
    assert base["refine"] is None and base["id"] != unrelated.id
    assert passed["refine"] == base["id"] and passed["strength"] == .25


def test_then_refine_does_not_qualify_a_small_source_just_because_target_is_1024(client, turbo):
    response = client.post("/api/generate", json={
        "prompt": "small source", "width": 512, "height": 512,
        "steps": 8, "guidance_scale": 1, "then_refine": {"scale": 2, "strength": .25},
    })
    assert response.status_code == 422, response.text
    assert "source" in response.json()["detail"] and "512" in response.json()["detail"]
    assert_no_work(turbo)


@pytest.mark.parametrize("side, cfg, scale, words", [
    (512, 1, 2, "source"),
    (1024, 2, 1, "CFG 1"),
], ids=["unmeasured-future-source", "unmeasured-pass-cfg"])
def test_finish_checks_retained_then_refine_before_any_base_exists(client, turbo, store, side, cfg, scale, words):
    request = GenerateRequest(prompt="retained future base", width=side, height=side,
                              steps=8, guidance_scale=cfg, seed=1201)
    cell = grid.Cell(request=request, seed=1201, label="base not rendered yet")
    spec = schemas.ThenRefine(scale=scale, strength=.25, steps=8)
    run_id = "anima-retained-then-" + uuid.uuid4().hex[:8]
    store.retain_run_job({
        "job_id": uuid.uuid4().hex[:12], "run_id": run_id, "created_at": time.time(),
        "origin": "grid", "model_id": TURBO, "quantization": "fp8",
        "width": side, "height": side, "batch": 1, "comparison": False,
        "metadata": {"grid": True, "then_refine": {
            **spec.model_dump(), "pass_run_id": then_refine.pass_run_id(run_id, spec),
        }},
    }, [cell])
    assert store.list_images(limit=10)[0] == []
    turbo.loaded = False
    response = client.post(f"/api/runs/{run_id}/finish/preview")
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["available"] is False
    assert words in preview["reason"]
    if side == 512:
        assert "512" in preview["reason"]
    finished = client.post(f"/api/runs/{run_id}/finish", json={"expected": preview["pin"]})
    assert finished.status_code == 422, finished.text
    assert words in finished.json()["detail"]
    assert store.list_images(limit=10)[0] == []
    assert_no_work(turbo)


def test_rerender_checks_sampling_of_every_record_sharing_the_same_source(client, turbo, store):
    source = plant(store)
    good = plant(store, seed=301, refine=source.id, strength=.25)
    old_unqualified = plant(store, seed=302, refine=source.id, strength=.3)
    response = client.post("/api/rerender/preview", json={
        "ids": [good.id, old_unqualified.id], "width": 1024, "height": 1024, "favorites": False,
    })
    assert response.status_code == 422, response.text
    assert "strengths 0.125, 0.25 and 0.4" in response.json()["detail"]
    assert_no_work(turbo)


def test_more_seeds_does_not_share_a_valid_verdict_with_an_unqualified_batch(client, turbo, store):
    source = plant(store)
    good = plant(store, seed=501, refine=source.id, strength=.25)
    old_unqualified = plant(store, seed=502, refine=source.id, strength=.3)
    response = client.post("/api/more-seeds/preview", json={
        "ids": [good.id, old_unqualified.id], "favorites": False, "count": 1,
    })
    assert response.status_code == 200, response.text
    records = {item["id"]: item for item in response.json()["results"]}
    assert records[good.id]["status"] == "queued"
    assert records[old_unqualified.id]["status"] != "queued"
    assert "strengths 0.125, 0.25 and 0.4" in records[old_unqualified.id]["detail"]
    assert_no_work(turbo)


@pytest.mark.parametrize("strength, available", [(.25, True), (.3, False)])
def test_finish_rechecks_retained_refine_intent_before_recovery(client, turbo, store, strength, available):
    source = plant(store)
    request = GenerateRequest(**pass_body(source.id, strength=strength))
    cell = grid.Cell(request=request, seed=901, label="retained pass")
    run_id = "anima-retained-" + uuid.uuid4().hex[:8]
    store.retain_run_job({
        "job_id": uuid.uuid4().hex[:12], "run_id": run_id, "created_at": time.time(),
        "origin": "grid", "model_id": TURBO, "quantization": "fp8",
        "width": 1024, "height": 1024, "batch": 1, "comparison": False, "metadata": {"grid": True},
    }, [cell])
    response = client.post(f"/api/runs/{run_id}/finish/preview")
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["available"] is available
    finished = client.post(f"/api/runs/{run_id}/finish", json={"expected": preview["pin"]})
    if not available:
        assert "strengths 0.125, 0.25 and 0.4" in preview["reason"]
        assert finished.status_code == 422, finished.text
        assert_no_work(turbo)
    else:
        assert finished.status_code == 200, finished.text
        done = stream_job(client, finished.json()["job_id"])[-1]
        assert done["status"] == "done", done
        assert done["run_id"] == run_id
        assert [(row["refine"], row["strength"], row["seed"]) for row in done["images"]] == [(source.id, .25, 901)]


def test_queued_source_geometry_is_rechecked_before_the_backend_is_called(client, turbo, store):
    source = plant(store)
    release = threading.Event()
    turbo.hold, turbo.hold_when = release, lambda request: request.refine is None
    try:
        first = client.post("/api/generate", json={"prompt": "holder", "width": 256, "height": 256, "steps": 8})
        assert first.status_code == 200, first.text
        assert turbo.holding.wait(30)
        queued = client.post("/api/generate", json=pass_body(source.id))
        assert queued.status_code == 200, queued.text
        Image.new("RGB", (512, 512), "red").save(store.settings.outputs_dir / source.filename)
    finally:
        release.set()
    stream_job(client, first.json()["job_id"])
    done = stream_job(client, queued.json()["job_id"])[-1]
    # An ordinary failed cell completes the job with its retained failure;
    # it is not a model-load failure or a reason to call the backend anyway.
    assert done["status"] == "done", done
    assert done["images"] == [] and done["failures"] == 1
    assert "512" in done["error"]
    assert len(turbo.requests) == 1 and turbo.requests[0].refine is None


def test_queued_pass_rechecks_host_qualification_after_its_admission(client, turbo, store):
    source = plant(store)
    release = threading.Event()
    turbo.hold, turbo.hold_when = release, lambda request: request.refine is None
    try:
        first = client.post("/api/generate", json={"prompt": "holder", "width": 256, "height": 256, "steps": 8})
        assert first.status_code == 200, first.text
        assert turbo.holding.wait(30)
        queued = client.post("/api/generate", json=pass_body(source.id))
        assert queued.status_code == 200, queued.text
        turbo.refine_host_facts = ("win32", "0.42.1", "NVIDIA GeForce RTX 3070", True)
    finally:
        release.set()
    stream_job(client, first.json()["job_id"])
    done = stream_job(client, queued.json()["job_id"])[-1]
    assert done["status"] == "error" and done["images"] == [], done
    assert "only the RTX 5080" in done["error"]
    assert len(turbo.requests) == 1 and turbo.requests[0].refine is None


@pytest.mark.parametrize("where", ["copied-later-inpaint", "configured-loader"])
def test_nonempty_execution_rejects_the_entire_turbo_batch_before_source_io(turbo, store, monkeypatch, where):
    from backend import refine_admission

    source = plant(store)
    ordinary = GenerateRequest(**pass_body(source.id, strength=.25))
    nonempty = ExecutionSpec(attention_backend="cudnn-v1")
    later = ordinary.model_copy(update={
        "mask": "a" * 64,
        "execution": nonempty if where == "copied-later-inpaint" else None,
    })
    assert later.inpainting
    if where == "configured-loader":
        select(monkeypatch, loader_execution=nonempty)
    cells = [grid.Cell(request=ordinary, seed=801), grid.Cell(request=later, seed=802)]
    touched = []

    def poison(name):
        def forbidden(*args, **kwargs):
            touched.append(name)
            raise AssertionError("nonempty execution reached " + name)
        return forbidden

    monkeypatch.setattr(refine_admission, "check_source", poison("source proof"))
    for name in ("reference_paths", "stamp_outpaints", "hold_masks", "retain_run_job"):
        monkeypatch.setattr(store, name, poison(name))

    async def forbidden_load():
        raise AssertionError("nonempty execution reached a model load")

    async def scenario():
        manager = jobs.JobManager(turbo, forbidden_load)
        try:
            with pytest.raises(ValueError, match="not available"):
                await manager.submit_cells(cells, run_id="explicit-turbo-admission-order")
            assert manager._loop is None
            assert manager._jobs == {} and manager._pending == []
        finally:
            await manager.shutdown()

    asyncio.run(scenario())
    assert touched == []
    assert_no_work(turbo)


@pytest.mark.parametrize("kind", ["mask", "outpaint"])
def test_copied_eager_inpainting_refuses_before_any_turbo_source_or_mask_work(turbo, store, monkeypatch, kind):
    from backend import refine_admission

    source = plant(store)
    ordinary = GenerateRequest(**pass_body(source.id, strength=.25))
    extra = {"mask": "b" * 64} if kind == "mask" else {"outpaint": schemas.Outpaint(left=16)}
    copied = ordinary.model_copy(update={**extra, "execution": None})
    assert copied.inpainting and copied.execution is None
    cells = [grid.Cell(request=ordinary, seed=811), grid.Cell(request=copied, seed=812)]
    touched = []

    def poison(name):
        def forbidden(*args, **kwargs):
            touched.append(name)
            raise AssertionError("unqualified copied inpainting reached " + name)
        return forbidden

    monkeypatch.setattr(refine_admission, "check_source", poison("source proof"))
    for name in ("reference_paths", "stamp_outpaints", "hold_masks", "retain_run_job"):
        monkeypatch.setattr(store, name, poison(name))

    async def forbidden_load():
        raise AssertionError("unqualified copied inpainting reached a model load")

    async def scenario():
        manager = jobs.JobManager(turbo, forbidden_load)
        try:
            with pytest.raises(ValueError, match="ordinary whole-image Refine pass"):
                await manager.submit_cells(cells, run_id="explicit-turbo-inpaint-refusal")
            assert manager._loop is None
            assert manager._jobs == {} and manager._pending == []
        finally:
            await manager.shutdown()

    asyncio.run(scenario())
    assert touched == []
    assert_no_work(turbo)


@pytest.mark.parametrize("model_id", ["circlestone-labs/Anima@aesthetic-v1.1", "Gazingstars123/Anima-2.9B"])
def test_other_anima_variants_keep_their_refine_hold(client, turbo, store, monkeypatch, model_id):
    source = plant(store)
    select(monkeypatch, model_id=model_id)
    status = client.get("/api/status").json()
    assert status["defaults"]["refine_supported"] is False
    assert status["refine_strength_suggestions"] is None
    response = client.post("/api/generate", json=pass_body(source.id))
    assert response.status_code == 422 and "no refine pass" in response.json()["detail"]
    assert_no_work(turbo)
