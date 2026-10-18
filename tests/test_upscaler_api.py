"""Raw GAN choices are refused before admission; fake jobs retain the choice."""
from __future__ import annotations

from PIL import Image
import pytest

from backend import main, model_selection, refine
from backend.schemas import GenerateRequest
from tests.api_helpers import stream_job


WAI = "LyliaEngine/waiIllustriousSDXL_v170"


def source(store):
    return store.save_image(Image.new("RGB", (256, 256), "red"),
                            GenerateRequest(prompt="source", width=256, height=256),
                            seed=1, duration_ms=1, model_id=WAI)


@pytest.mark.parametrize("model", ["cagliostrolab/animagine-xl-4.0", "black-forest-labs/FLUX.1-Krea-dev"])
def test_unsupported_gan_is_refused_without_a_load_or_job(client, fake, store, model):
    model_selection.select(model)
    record = source(store)
    before = len(store.retained_run_jobs())
    response = client.post("/api/generate", json={"prompt": "x", "refine": record.id,
                                                 "upscaler": refine.ANIME6B})
    assert response.status_code == 422 and "only for WAI" in response.json()["detail"]
    assert not fake.requests and fake.loads == 0 and not main.jobs.active()
    assert len(store.retained_run_jobs()) == before


def test_source_cap_is_an_admission_refusal_without_decoding_or_queueing(client, fake, store, monkeypatch):
    model_selection.select(WAI)
    record = source(store)
    monkeypatch.setattr(refine, "check_upscaler_source", lambda image_id: refine.check_upscaler_size(4097, 256))
    response = client.post("/api/generate", json={"prompt": "x", "refine": record.id,
                                                 "upscaler": refine.ANIME6B})
    assert response.status_code == 422 and "4.19 MP" in response.json()["detail"]
    assert not fake.requests and not main.jobs.active()


def test_mixed_gan_and_inpaint_rerender_checks_each_own_complete_input_spec(client, fake, store):
    model_selection.select(WAI)
    original = source(store)
    mask, _ = store.store_mask(Image.new("L", (256, 256), 255))
    records = []
    for seed, fields in ((10, {"upscaler": refine.ANIME6B}), (20, {"mask": mask})):
        request = GenerateRequest(prompt=f"take {seed}", width=256, height=256, steps=2,
                                  refine=original.id, strength=0.5, **fields)
        records.append(store.save_image(Image.new("RGB", (256, 256), "blue"), request,
                                       seed=seed, duration_ms=1, model_id=WAI, quantization="none"))
    response = client.post("/api/rerender", json={"ids": [record.id for record in records],
                                                 "width": 256, "height": 256})
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 2
    assert [request.upscaler for request in fake.requests] == [refine.ANIME6B, None]
    assert [request.mask for request in fake.requests] == [None, mask]
