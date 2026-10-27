"""Preview selection and reproduction stay distinct from the other Anima releases."""
from __future__ import annotations

import dataclasses
import json

import pytest
from PIL import Image

from backend import jobs, main, schemas, schemas_generate
from backend.backends import loading
from backend.backends.anima_loading import PREVIEW_MODEL_ID
from backend.model_defaults import defaults_for
from tests import fakes
from tests.api_helpers import run_one, stream_job


@pytest.fixture
def preview(fake, monkeypatch):
    for module in (main, jobs, schemas, schemas_generate, fakes):
        monkeypatch.setattr(
            module, "settings", dataclasses.replace(module.settings, model_id=PREVIEW_MODEL_ID)
        )
    profile = defaults_for(PREVIEW_MODEL_ID)
    for key in ("cfg_threshold", "true_cfg_supported", "negative_prompt_supported",
                "mu_supported", "reference_supported", "refine_supported", "inpaint_supported"):
        setattr(fake, key, profile[key])
    fake.lora_reason = loading.lora_refusal(PREVIEW_MODEL_ID)
    return fake


def test_preview_applied_settings_and_public_identity_survive_png_and_resume(client, preview, store):
    # A Turbo row with identical generation settings must not count as this
    # preview's completed cell. Profiles alone cannot establish this boundary.
    request = schemas.GenerateRequest(prompt="a fox", width=256, height=256, seed=7911)
    turbo = store.save_image(
        Image.new("RGB", (256, 256)), request, seed=7911, duration_ms=10,
        model_id="circlestone-labs/Anima", quantization=main.settings.quantization,
    )
    body = {**request.model_dump(), "resume": True}
    plan = client.post("/api/grid/preview", json=body).json()
    assert (plan["cells"], plan["skipped"], plan["in_gallery"]) == (1, 0, 0)
    response = client.post("/api/grid", json=body)
    assert response.status_code == 200
    stream_job(client, response.json()["job_id"])
    gallery = client.get("/api/gallery").json()
    assert gallery["other_models"] == 1
    [row] = gallery["items"]
    assert row["id"] != turbo.id
    assert row["model_id"] == PREVIEW_MODEL_ID
    assert client.post("/api/grid", json=body).status_code == 409

    job_id, _ = run_one(client, negative_prompt="blurry", seed=7912, mu=0.8, true_cfg=8)
    [row] = client.get(f"/api/jobs/{job_id}").json()["images"]
    assert (row["model_id"], row["steps"], row["guidance_scale"]) == (PREVIEW_MODEL_ID, 30, 4)
    assert (row["negative_prompt"], row["true_cfg"], row["mu"]) == ("blurry", 4, None)
    with Image.open(store.settings.outputs_dir / row["filename"]) as image:
        metadata = json.loads(image.info["auradream"])
    assert metadata["model"] == PREVIEW_MODEL_ID
    for key in ("steps", "guidance_scale", "negative_prompt", "true_cfg", "mu", "seed"):
        assert metadata[key] == row[key], key


@pytest.mark.parametrize(
    "extra, reason",
    [
        ({"reference": [1]}, "takes no reference images"),
        ({"lora": "demo-style.safetensors"}, "not supported by this Anima 2.9B Preview integration"),
        ({"refine": 1, "strength": 0.4}, "has no refine pass"),
    ],
)
def test_preview_refuses_unsupported_inputs_before_queuing(client, preview, extra, reason):
    for route in ("/api/generate", "/api/grid"):
        response = client.post(
            route, json={"prompt": "a fox", "width": 256, "height": 256, "seed": 1, **extra}
        )
        assert response.status_code == 422
        assert reason in response.json()["detail"]
    assert preview.requests == []
