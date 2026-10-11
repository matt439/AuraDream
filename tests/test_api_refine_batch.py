"""Mass refine over HTTP: the preview's per-image statuses, the jobs it
queues - one cell per image, a pass of itself, one job per size - the
selection it pins, and what it prices and reads off the schedule."""
from __future__ import annotations

import dataclasses
import json

from PIL import Image

from backend import api_refine_batch, refine as refine_module, refine_batch, submission
from backend.config import settings
from backend.model_defaults import defaults_for
from backend.model_selection import current_model_id
from backend.schemas import GenerateRequest, default_strength
from tests.api_helpers import stream_job


def plant(store, prompt, *, seed, width=256, height=256, steps=20, model_id=None, variables=None, **settings_):
    return store.save_image(
        Image.new("RGB", (width, height)),
        GenerateRequest(prompt=prompt, width=width, height=height, steps=steps, guidance_scale=1.5, **settings_),
        seed=seed, duration_ms=1, model_id=model_id or settings.model_id,
        variables=variables,
    )


def preview(client, **body):
    response = client.post("/api/refine/batch/preview", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def by_id(answer):
    return {image["id"]: image for image in answer["images"]}


def test_the_preview_says_what_happens_to_each_image(client, fake, store, monkeypatch):
    a = plant(store, "a fox", seed=1)
    passed = plant(store, "a pass", seed=2, refine=a.id, strength=0.5)
    other = plant(store, "elsewhere", seed=3, model_id="somebody/else")
    adapter = plant(store, "with an adapter", seed=4, lora="nowhere")
    large = plant(store, "large", seed=5, width=512, height=512)
    lost = plant(store, "file gone", seed=6)
    (store.settings.outputs_dir / lost.filename).unlink()
    # A smaller guard than any preset's, so a small image goes past it.
    monkeypatch.setattr(refine_batch, "settings", dataclasses.replace(settings, refine_max_pixels=600 * 600))
    answer = preview(client, ids=[large.id, a.id, passed.id, other.id, adapter.id, lost.id, 999], scale=1.5)
    images = by_id(answer)
    assert answer["ids"] == sorted([a.id, passed.id, other.id, adapter.id, large.id, lost.id])
    assert [image["id"] for image in answer["images"]] == sorted([*answer["ids"], 999])
    assert images[a.id]["status"] == "queued"
    assert (images[a.id]["target_width"], images[a.id]["target_height"]) == (384, 384)
    assert images[passed.id]["status"] == "pass" and f"#{a.id}" in images[passed.id]["detail"]
    assert images[other.id]["status"] == "foreign" and "somebody/else" in images[other.id]["detail"]
    assert images[adapter.id]["status"] == "lora_missing"
    assert images[large.id]["status"] == "too_big"
    assert images[large.id]["detail"].startswith("512×512 at 1.5× is 768×768, past the")
    assert images[lost.id]["status"] == "file_missing"
    assert images[999]["status"] == "deleted"
    assert answer["counts"] == {
        "queued": 1, "pass": 1, "foreign": 1, "lora_missing": 1, "too_big": 1, "file_missing": 1, "deleted": 1,
    }
    assert (answer["queued"], answer["jobs"], answer["total"]) == (1, 1, 7)
    assert (answer["foreign"], answer["models"], answer["passes"]) == (1, ["somebody/else"], 1)
    # The scale's cap is the largest image it applies to, too big or not.
    assert answer["largest"] == [512, 512]
    assert answer["strength"] == default_strength()
    # Nothing queued by a preview.
    assert not fake.requests


def test_a_strength_that_leaves_no_step_is_per_image_and_steps_overrides(client, fake, store):
    few = plant(store, "few steps", seed=1, steps=6)
    many = plant(store, "many steps", seed=2, steps=40)
    images = by_id(preview(client, ids=[few.id, many.id], strength=0.1))
    assert images[few.id]["status"] == "no_step"
    assert images[few.id]["detail"] == "strength 0.1 of 6 steps leaves no step; set Steps or raise the strength"
    assert images[many.id]["status"] == "queued"
    # Steps set for every image frees the one made at few.
    assert preview(client, ids=[few.id, many.id], strength=0.1, steps=20)["queued"] == 2


def test_queueing_refines_each_image_from_itself_with_its_own_settings(client, fake, store):
    a = plant(store, "a fox", seed=11, negative_prompt="blurry", mu=0.8, variables={"who": "fox"})
    b = plant(store, "a hare", seed=12, width=384, height=256)
    c = plant(store, "an owl", seed=13, reference=[a.id])
    ids = [a.id, b.id, c.id]
    expected = preview(client, ids=ids, strength=0.5, scale=1.5)["ids"]
    response = client.post("/api/refine/batch", json={"ids": ids, "strength": 0.5, "scale": 1.5, "expected": expected})
    assert response.status_code == 200, response.text
    accepted = response.json()
    # One job per size: 384x384 (a then c, oldest first), then 576x384.
    assert (accepted["queued"], accepted["pruned"], len(accepted["more"])) == (3, 0, 1)
    first = stream_job(client, accepted["job_id"])[-1]
    second = stream_job(client, accepted["more"][0]["job_id"])[-1]
    assert first["status"] == second["status"] == "done"
    assert (first["refine"], first["refine_batch"], first["strength"], first["grid"]) == (None, 1.5, 0.5, True)
    assert (first["width"], first["height"], first["planned_images"]) == (384, 384, 2)
    asked = [
        (r.refine, r.prompt, r.negative_prompt, r.seed, r.width, r.height, r.steps, r.guidance_scale, r.mu,
         r.strength, r.reference)
        for r in fake.requests
    ]
    assert asked == [
        (a.id, "a fox", "blurry", 11, 384, 384, 20, 1.5, 0.8, 0.5, []),
        (c.id, "an owl", None, 13, 384, 384, 20, 1.5, c.mu, 0.5, []),
        (b.id, "a hare", None, 12, 576, 384, 20, 1.5, b.mu, 0.5, []),
    ]
    made = first["images"][0]
    assert made["variables"] == {"who": "fox"} and made["refine"] == a.id
    with Image.open(store.settings.outputs_dir / made["filename"]) as png:
        chunk = json.loads(png.info["local-diffusion"])
    assert (chunk["refine"], chunk["seed"], chunk["strength"]) == (a.id, 11, 0.5)

    # Asked again, every pass is in the gallery already.
    again = preview(client, ids=ids, strength=0.5, scale=1.5)
    assert again["counts"] == {"rendered": 3} and again["queued"] == 0
    refused = client.post("/api/refine/batch", json={"ids": ids, "strength": 0.5, "scale": 1.5})
    assert refused.status_code == 409
    assert refused.json()["detail"] == "every image that could be refined already has this pass in the gallery"
    # The passes themselves are skipped by default, and refined when asked.
    passes = [image["id"] for image in first["images"]]
    assert preview(client, ids=passes)["counts"] == {"pass": 2}
    assert preview(client, ids=passes, skip_passes=False)["queued"] == 2


def test_a_partial_overlap_queues_the_rest_and_counts_what_was_pruned(client, fake, store):
    a = plant(store, "one", seed=1)
    b = plant(store, "two", seed=2)
    body = {"ids": [a.id], "strength": 0.5, "scale": 1.25}
    stream_job(client, client.post("/api/refine/batch", json=body).json()["job_id"])
    fake.requests.clear()
    answer = preview(client, ids=[a.id, b.id], strength=0.5, scale=1.25)
    assert answer["counts"] == {"rendered": 1, "queued": 1}
    response = client.post("/api/refine/batch", json={
        "ids": [a.id, b.id], "strength": 0.5, "scale": 1.25, "expected": answer["ids"],
    })
    assert (response.json()["queued"], response.json()["pruned"]) == (1, 1)
    stream_job(client, response.json()["job_id"])
    assert [r.refine for r in fake.requests] == [b.id]


def test_another_checkpoints_images_are_refined_only_when_asked(client, fake, store):
    other = plant(store, "elsewhere", seed=7, model_id="somebody/else", lora="theirs", mu=3.0)
    answer = preview(client, ids=[other.id])
    assert answer["counts"] == {"foreign": 1} and answer["foreign"] == 1
    answer = preview(client, ids=[other.id], include_foreign=True)
    image = by_id(answer)[other.id]
    assert image["status"] == "queued"
    running = current_model_id(settings)
    assert image["notes"] == [
        f"made on somebody/else: steps, CFG and mu are {running}'s",
        "LoRA theirs not applied: it was for another checkpoint",
    ]
    response = client.post("/api/refine/batch", json={"ids": [other.id], "include_foreign": True})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    profile = defaults_for(running)
    assert (ran.prompt, ran.seed, ran.refine) == ("elsewhere", 7, other.id)
    assert (ran.steps, ran.guidance_scale, ran.mu, ran.lora) == (
        profile["steps"], profile["guidance_scale"], profile["mu"], None,
    )


def test_a_checkpoint_with_no_refine_pass_refuses_the_whole_of_it(client, fake, store):
    a = plant(store, "a fox", seed=1)
    fake.refine_supported = False
    for route in ("/api/refine/batch/preview", "/api/refine/batch"):
        response = client.post(route, json={"ids": [a.id]})
        assert response.status_code == 422
        assert "has no refine pass in this app" in response.json()["detail"]
    assert not fake.requests


def test_a_live_capability_drop_fails_the_job_before_any_image(client, fake, store, monkeypatch):
    a = plant(store, "a fox", seed=1)
    b = plant(store, "a hare", seed=2)
    # Accepted as though the pass were there; the worker's own check is what
    # stops it, before cell 1 saves anything.
    async def available(services):
        return None

    monkeypatch.setattr(submission, "refine_unavailable", available)
    fake.refine_supported = False
    response = client.post("/api/refine/batch", json={"ids": [a.id, b.id]})
    assert response.status_code == 200, response.text
    final = stream_job(client, response.json()["job_id"])[-1]
    assert final["status"] == "error" and "has no refine pass" in final["error"]
    assert final["image_count"] == 0 and not fake.requests


def test_a_source_deleted_mid_run_fails_only_its_own_cell(client, fake, store, monkeypatch):
    a = plant(store, "one", seed=1)
    b = plant(store, "two", seed=2)
    c = plant(store, "three", seed=3)
    real = fake.generate

    # The real backend opens the source as the cell starts (refine.source_image).
    def generate(request, *args, **kwargs):
        refine_module.source_image(request.refine, request.width, request.height)
        return real(request, *args, **kwargs)

    monkeypatch.setattr(fake, "generate", generate)
    fake.after_generate = lambda: (
        client.delete(f"/api/images/{b.id}") if len(fake.requests) == 1 else None
    )
    response = client.post("/api/refine/batch", json={"ids": [a.id, b.id, c.id], "favorites": False})
    assert response.status_code == 200, response.text
    final = stream_job(client, response.json()["job_id"])[-1]
    # Cells run oldest first; the deleted one is named by its own label.
    assert final["status"] == "done" and final["failures"] == 1
    assert [image["refine"] for image in final["images"]] == [a.id, c.id]
    assert final["failed_cells"][0]["label"] == f"#{b.id}"


def test_nothing_to_queue_names_every_reason_unless_all_are_rendered(client, fake, store):
    a = plant(store, "a fox", seed=1)
    first = client.post("/api/refine/batch", json={"ids": [a.id], "strength": 0.5})
    assert first.status_code == 200, first.text
    assert stream_job(client, first.json()["job_id"])[-1]["status"] == "done"
    adapter = plant(store, "with an adapter", seed=2, lora="nowhere")
    # The rendered one and one whose LoRA is gone: not "every image already
    # has this pass", which would hide the one the user could fix.
    refused = client.post("/api/refine/batch", json={"ids": [a.id, adapter.id], "strength": 0.5})
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert detail.startswith("no selected image can be refined (")
    assert "1 rendered" in detail and "1 lora missing" in detail
    # Only the rendered one: the 409 that says so.
    alone = client.post("/api/refine/batch", json={"ids": [a.id], "strength": 0.5})
    assert alone.status_code == 409
    assert alone.json()["detail"] == "every image that could be refined already has this pass in the gallery"


def test_the_queue_is_pinned_to_the_ids_the_preview_answered(client, fake, store):
    a = plant(store, "one", seed=1)
    b = plant(store, "two", seed=2)
    expected = preview(client, ids=[a.id, b.id])["ids"]
    assert client.delete(f"/api/images/{b.id}").status_code == 200
    response = client.post("/api/refine/batch", json={"ids": [a.id, b.id], "expected": expected})
    assert response.status_code == 409
    assert response.json()["detail"] == "the selection changed since the check; review the refine again"
    assert not fake.requests


def test_without_ids_the_selection_is_the_gallerys_filter(client, fake, store):
    a = plant(store, "a fox", seed=1)
    plant(store, "a hare", seed=2)
    client.post(f"/api/images/{a.id}/favorite?value=true")
    assert preview(client)["ids"] == [a.id]
    assert preview(client, favorites=False, search="hare")["queued"] == 1
    response = client.post("/api/refine/batch/preview", json={"favorites": False, "search": "seed:999"})
    assert response.status_code == 404
    too_many = list(range(1, settings.max_grid_images + 2))
    assert client.post("/api/refine/batch/preview", json={"ids": too_many}).status_code == 422
    assert client.post("/api/refine/batch/preview", json={"ids": [0]}).status_code == 422
    refused = client.post("/api/refine/batch/preview", json={"ids": [a.id], "strength": 1.5})
    assert refused.status_code == 422 and "not a fraction" in refused.text
    assert client.post("/api/refine/batch/preview", json={"ids": [a.id], "scale": 3}).status_code == 422


def test_nothing_refinable_is_a_422_that_counts_why(client, fake, store):
    other = plant(store, "elsewhere", seed=1, model_id="somebody/else")
    few = plant(store, "few", seed=2, steps=6)
    response = client.post("/api/refine/batch", json={"ids": [other.id, few.id], "strength": 0.1})
    assert response.status_code == 422
    assert response.json()["detail"] == "no selected image can be refined (1 foreign, 1 no step)"


def test_the_start_sigma_is_a_range_over_the_passes_sizes(client, fake, store, monkeypatch):
    small = plant(store, "small", seed=1)
    wide = plant(store, "wide", seed=2, width=512, height=256)
    asked = []
    monkeypatch.setattr(fake, "start_sigma", lambda request: asked.append(request) or request.width / 1000)
    answer = preview(client, ids=[small.id, wide.id], scale=1.5)
    assert (answer["start_sigma_low"], answer["start_sigma_high"]) == (0.384, 0.768)
    # Read with the pass as it runs.
    assert {r.refine for r in asked} == {small.id, wide.id}
    monkeypatch.setattr(fake, "start_sigma", lambda request: None if request.width > 500 else 0.4)
    answer = preview(client, ids=[small.id, wide.id], scale=1.5)
    assert answer["start_sigma_low"] is None and answer["start_sigma_high"] is None
    # Past the cap of distinct passes, none is read.
    monkeypatch.setattr(api_refine_batch, "REFINE_BATCH_SIGMA_KEYS", 1)
    monkeypatch.setattr(fake, "start_sigma", lambda request: 0.5)
    assert preview(client, ids=[small.id, wide.id])["start_sigma_low"] is None
    assert preview(client, ids=[small.id])["start_sigma_low"] == 0.5


def test_the_time_is_priced_per_size_from_refine_history(client, fake, store):
    # A duration of a millisecond or more, which the history counts.
    fake.step_seconds = 0.002
    a = plant(store, "one", seed=1)
    b = plant(store, "two", seed=2)
    c = plant(store, "three", seed=3, width=384, height=256)
    answer = preview(client, ids=[a.id, b.id, c.id], strength=0.5)
    assert (answer["seconds"], answer["unpriced_sizes"]) == (None, 2)
    stream_job(client, client.post("/api/refine/batch", json={"ids": [a.id], "strength": 0.5}).json()["job_id"])
    answer = preview(client, ids=[b.id, c.id], strength=0.5)
    assert (answer["seconds"], answer["unpriced_sizes"]) == (None, 1)
    answer = preview(client, ids=[b.id], strength=0.5)
    assert answer["unpriced_sizes"] == 0 and isinstance(answer["seconds"], float)


def test_the_queue_is_a_model_dependent_request_and_the_preview_is_not(client, fake, store):
    a = plant(store, "a fox", seed=1)
    stale = {"X-Local-Diffusion-Model-Revision": "999"}
    body = {"ids": [a.id]}
    refused = client.post("/api/refine/batch", json=body, headers=stale)
    assert refused.status_code == 409 and "checkpoint changed" in refused.json()["detail"]
    assert client.post("/api/refine/batch/preview", json=body, headers=stale).status_code == 200


def test_a_picked_image_binned_since_says_it_is_in_the_bin(client, fake, store):
    a = plant(store, "a fox", seed=1)
    b = plant(store, "a hare", seed=2)
    store.bin_images([b.id])
    images = by_id(preview(client, ids=[a.id, b.id]))
    assert (images[b.id]["status"], images[b.id]["detail"]) == ("deleted", "in the recycle bin")
    assert images[a.id]["status"] == "queued"
