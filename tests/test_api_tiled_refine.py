"""Serve and queue the exact Tiled Refine recipe without a model or GPU."""
from __future__ import annotations

from PIL import Image

from backend import gallery_reports, model_selection, tiled_refine_contract as tiled
from backend.schemas import GenerateRequest
from tests.api_helpers import stream_job


def _source(store, size=tiled.ROOT_SIZE):
    with Image.new("RGB", size, (50, 80, 120)) as image:
        return store.save_image(
            image, GenerateRequest(prompt="an old cabin", width=size[0], height=size[1]),
            4, 5, tiled.MODEL_ID, quantization=tiled.QUANTIZATION,
        )


def _select(fake):
    model_selection.select(tiled.MODEL_ID, quantization=tiled.QUANTIZATION, offload=tiled.OFFLOAD)
    fake.tiled_refine_qualified = True


def _request(source):
    return {
        "prompt": "an old cabin", "negative_prompt": "haze", "seed": 7,
        "width": 2048, "height": 2048, "steps": 28, "guidance_scale": 5, "strength": .35,
        "tiled_refine": {"source": source, "recipe": tiled.RECIPE_NAME},
    }


def test_preview_uses_status_recipe_and_reports_stale_and_invalid_sources(client, fake, store):
    _select(fake)
    square = _source(store)
    wrong_size = _source(store, (768, 1024))
    status = client.get("/api/status").json()
    assert status["defaults"]["tiled_refine_reason"] is None
    plan = status["defaults"]["tiled_refine"]
    assert (plan["tiles"], plan["denoised_steps"], plan["width"], plan["height"]) == (4, 36, 2048, 2048)
    path = "/api/tiled-refine/preview"
    assert client.post(path, json={"source": square.id}).json() == {**plan, "reason": None}
    assert "1024x1024 RGB" in client.post(path, json={"source": wrong_size.id}).json()["reason"]
    assert client.post(path, json={"source": square.id}, headers={
        "X-AuraDream-Model-Revision": str(status["model_revision"] + 1),
    }).status_code == 409
    fake.tiled_refine_qualified = False
    assert client.get("/api/status").json()["defaults"]["tiled_refine_reason"] == tiled.HELD
    assert client.post(path, json={"source": square.id}).json()["reason"] == tiled.HELD
    assert not fake.requests


def test_generate_splits_batch_and_repeat_and_keeps_the_bound_source(client, fake, store):
    _select(fake)
    root = _source(store)
    request = _request(root.id)
    response = client.post("/api/generate", json={**request, "num_images": 2, "runs": 2})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    done = stream_job(client, job_id)[-1]
    assert (done["status"], done["step"], done["total_steps"], done["total_runs"]) == ("done", 36, 36, 4)
    assert (done["tiled_refine"], done["refine"]) == (request["tiled_refine"], None)
    assert fake.seeds == [7, 8, 9, 10]
    assert all(req.num_images == 1 and req.runs == 1 for req in fake.requests)
    results = done["images"]
    assert [image["seed"] for image in results] == fake.seeds
    spec = results[0]["tiled_refine"]
    assert spec["root_source"]["filename"] == root.filename
    assert all(row["tiled_refine"] == spec and row["true_cfg"] == 5 for row in results)
    assert gallery_reports.step_cost_ms(2048, 2048, tiled.MODEL_ID) is None
    timing = gallery_reports.gallery_timing(tiled.QUANTIZATION)[tiled.MODEL_ID]
    assert (timing["width"], timing["height"], timing["images"]) == (1024, 1024, 1)

    assert client.post("/api/images/bin", json={"ids": [root.id]}).status_code == 200
    replay = client.post("/api/tiled-refine/preview", json={"source": root.id, "tiled_refine": spec}).json()
    assert replay["code"] == "image_binned" and replay["ids"] == [root.id]
    refused = client.post("/api/generate", json={**request, "tiled_refine": spec})
    assert refused.status_code == 422 and refused.json()["code"] == "image_binned"
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert all(row["tiled_refine_binned"] for row in snapshot["images"])
    assert client.post("/api/images/restore", json={"ids": [root.id]}).status_code == 200
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert not any(row["tiled_refine_missing"] for row in snapshot["images"])
