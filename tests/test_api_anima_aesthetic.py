"""The Aesthetic release's identity and supported inputs through HTTP."""
from __future__ import annotations

import dataclasses
import json

import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from PIL import Image

from backend import jobs, main, schemas, schemas_generate
from backend.backends import loading
from backend.model_defaults import defaults_for
from tests import fakes
from tests.api_helpers import run_one, stream_job


AESTHETIC = "circlestone-labs/Anima@aesthetic-v1.1"
TURBO = "circlestone-labs/Anima"


@pytest.fixture
def aesthetic(fake, monkeypatch):
    # As a server started with this selection: schema defaults, the gallery's
    # exact model scope and saved identities all read the same settings.
    for module in (main, jobs, schemas, schemas_generate, fakes):
        monkeypatch.setattr(
            module, "settings", dataclasses.replace(module.settings, model_id=AESTHETIC)
        )
    profile = defaults_for(AESTHETIC)
    fake.cfg_threshold = profile["cfg_threshold"]
    fake.true_cfg_supported = profile["true_cfg_supported"]
    fake.mu_supported = profile["mu_supported"]
    fake.reference_supported = profile["reference_supported"]
    fake.refine_supported = profile["refine_supported"]
    fake.lora_reason = loading.lora_refusal(AESTHETIC)
    return fake


def test_aesthetic_defaults_and_applied_settings_reach_the_row_and_png(client, aesthetic, store):
    job_id, _ = run_one(client, negative_prompt="blurry", seed=7895, mu=0.8, true_cfg=8)
    row = client.get(f"/api/jobs/{job_id}").json()["images"][0]
    assert (row["model_id"], row["steps"], row["guidance_scale"]) == (AESTHETIC, 30, 4)
    # True CFG is not another knob here: the row names the negative prompt's
    # actual CFG, and the unsupported Mu cannot survive into reproduction.
    assert (row["negative_prompt"], row["true_cfg"], row["mu"]) == ("blurry", 4, None)
    with Image.open(store.settings.outputs_dir / row["filename"]) as image:
        metadata = json.loads(image.info["auradream"])
    assert metadata["model"] == row["model_id"]
    for key in ("steps", "guidance_scale", "negative_prompt", "true_cfg", "mu", "seed"):
        assert metadata[key] == row[key], key


@pytest.mark.parametrize(
    "error_class, kind", [(RepositoryNotFoundError, "not_found"), (GatedRepoError, "gated")]
)
def test_aesthetic_load_failure_links_to_the_hub_repo_not_the_selector(aesthetic, error_class, kind):
    error = error_class.__new__(error_class)
    error.args = ("Cannot fetch the selected checkpoint",)
    detail, actual_kind, url = main._load_failure(error)
    assert actual_kind == kind
    assert url == "https://huggingface.co/circlestone-labs/Anima"
    assert "@aesthetic-v1.1" not in detail
    if kind == "gated":
        assert detail.startswith("Anima Aesthetic v1.1 is gated on Hugging Face:")
    else:
        assert detail.startswith("circlestone-labs/Anima is not on Hugging Face,")


@pytest.mark.parametrize("cfg, active", [(0, False), (1, False), (1.5, True)])
def test_aesthetic_negative_prompt_switches_on_only_above_one(client, aesthetic, cfg, active):
    job_id, _ = run_one(client, guidance_scale=cfg, negative_prompt="blurry", true_cfg=9)
    row = client.get(f"/api/jobs/{job_id}").json()["images"][0]
    assert row["negative_prompt"] == ("blurry" if active else None)
    assert row["true_cfg"] == (cfg if active else None)


@pytest.mark.parametrize(
    "extra, reason",
    [
        ({"reference": [1]}, "takes no reference images"),
        ({"lora": "demo-style.safetensors"}, "not supported by this Anima Aesthetic integration"),
        ({"refine": 1, "strength": 0.4}, "has no refine pass"),
    ],
)
def test_aesthetic_refuses_unsupported_inputs_before_queuing(client, aesthetic, extra, reason):
    for route in ("/api/generate", "/api/grid"):
        response = client.post(
            route, json={"prompt": "a fox", "width": 256, "height": 256, "seed": 1, **extra}
        )
        assert response.status_code == 422
        assert reason in response.json()["detail"]
    assert aesthetic.requests == []


def test_aesthetic_resume_does_not_consume_turbos_identical_settings(client, aesthetic, store):
    # Give Turbo exactly Aesthetic's settings, so only the recorded model
    # identity separates them. A different profile alone cannot prove this.
    request = schemas.GenerateRequest(prompt="a fox", width=256, height=256, seed=7895)
    turbo = store.save_image(
        Image.new("RGB", (256, 256)), request, seed=7895, duration_ms=10,
        model_id=TURBO, quantization=main.settings.quantization,
    )
    body = request.model_dump()
    preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    assert (preview["cells"], preview["skipped"], preview["in_gallery"]) == (1, 0, 0)
    response = client.post("/api/grid", json={**body, "resume": True})
    assert response.status_code == 200
    stream_job(client, response.json()["job_id"])
    gallery = client.get("/api/gallery").json()
    assert gallery["other_models"] == 1
    assert len(gallery["items"]) == 1
    assert gallery["items"][0]["model_id"] == AESTHETIC
    assert gallery["items"][0]["id"] != turbo.id
    assert client.post("/api/grid", json={**body, "resume": True}).status_code == 409
