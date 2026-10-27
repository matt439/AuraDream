"""Swap tags over HTTP: the preview's per-image statuses, the jobs it queues,
the selection it pins, and the helpers it shares with Re-render."""
from __future__ import annotations

import json

from PIL import Image

from backend import main, submission
from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import run_one, stream_job

SWAP = [{"find": "red hair, long hair", "replace": "silver hair"}]


def plant(store, prompt, *, seed, width=256, height=256, model_id=None, variables=None, **settings_):
    return store.save_image(
        Image.new("RGB", (width, height)),
        GenerateRequest(prompt=prompt, width=width, height=height, steps=6, guidance_scale=1.5, **settings_),
        seed=seed, duration_ms=1, model_id=model_id or settings.model_id,
        variables=variables,
    )


def preview(client, **body):
    response = client.post("/api/swap-tags/preview", json={"swaps": SWAP, **body})
    assert response.status_code == 200, response.text
    return response.json()


def by_id(answer):
    return {image["id"]: image for image in answer["images"]}


def test_the_preview_says_what_happens_to_each_image(client, fake, store):
    a = plant(store, "1girl, red hair, long hair, smile", seed=1)
    b = plant(store, "1girl, (red_hair:1.2), long_hair", seed=2)
    c = plant(store, "1girl, red hair, smile", seed=3)
    d = plant(store, "1girl, blue eyes", seed=4)
    answer = preview(client, ids=[d.id, a.id, b.id, c.id, 999])
    images = by_id(answer)
    assert [image["id"] for image in answer["images"]] == sorted([a.id, b.id, c.id, d.id, 999])
    assert answer["ids"] == [a.id, b.id, c.id, d.id]
    assert images[a.id]["status"] == "queued"
    assert images[a.id]["text"] == "1girl, silver hair, smile"
    start, end = images[a.id]["spans"][0]
    assert images[a.id]["text"][start:end] == "silver hair"
    assert images[a.id]["removed"] == ["red hair", "long hair"]
    assert images[b.id]["text"] == "1girl, (silver_hair:1.2)"
    assert images[b.id]["removed"] == ["(red_hair:1.2)", "long_hair"]
    assert images[c.id]["removed"] == []
    assert (images[c.id]["status"], images[c.id]["detail"]) == ("missing", "missing: long hair")
    assert images[c.id]["text"] == "1girl, red hair, smile"
    assert images[d.id]["status"] == "missing"
    assert images[999]["status"] == "deleted"
    assert answer["counts"] == {"queued": 2, "missing": 2, "deleted": 1}
    assert (answer["queued"], answer["jobs"], answer["total"]) == (2, 1, 5)
    # Nothing queued by a preview.
    assert not fake.requests


def test_queueing_keeps_each_images_seed_size_and_settings(client, fake, store):
    a = plant(store, "1girl, red hair, long hair", seed=11, width=256, height=256)
    b = plant(store, "2girls, red hair, long hair", seed=12, width=384, height=256,
              variables={"who": "2girls"})
    c = plant(store, "3girls, red hair, long hair", seed=13, width=256, height=256)
    ids = [a.id, b.id, c.id]
    expected = preview(client, ids=ids)["ids"]
    response = client.post("/api/swap-tags", json={"swaps": SWAP, "ids": ids, "expected": expected})
    assert response.status_code == 200, response.text
    accepted = response.json()
    # One job per size: 256x256 (two images, oldest first), then 384x256.
    assert accepted["queued"] == 3 and len(accepted["more"]) == 1
    first = stream_job(client, accepted["job_id"])[-1]
    second = stream_job(client, accepted["more"][0]["job_id"])[-1]
    assert first["status"] == second["status"] == "done"
    assert first["swap"] == "red hair, long hair → silver hair"
    asked = [(r.prompt, r.seed, r.width, r.height, r.steps, r.guidance_scale) for r in fake.requests]
    assert asked == [
        ("1girl, silver hair", 11, 256, 256, 6, 1.5),
        ("3girls, silver hair", 13, 256, 256, 6, 1.5),
        ("2girls, silver hair", 12, 384, 256, 6, 1.5),
    ]
    made = second["images"][0]
    assert made["variables"] == {"who": "2girls"}
    with Image.open(store.settings.outputs_dir / made["filename"]) as png:
        chunk = json.loads(png.info["auradream"])
    assert (chunk["prompt"], chunk["seed"]) == ("2girls, silver hair", 12)

    # Asked again, everything is in the gallery already.
    again = preview(client, ids=ids)
    assert again["counts"] == {"rendered": 3} and again["queued"] == 0
    refused = client.post("/api/swap-tags", json={"swaps": SWAP, "ids": ids})
    assert refused.status_code == 409 and "already in the gallery" in refused.json()["detail"]


def test_the_queue_is_pinned_to_the_ids_the_preview_answered(client, fake, store):
    a = plant(store, "red hair, long hair", seed=1)
    b = plant(store, "red hair, long hair, smile", seed=2)
    expected = preview(client, ids=[a.id, b.id])["ids"]
    assert client.delete(f"/api/images/{b.id}").status_code == 200
    response = client.post("/api/swap-tags", json={"swaps": SWAP, "ids": [a.id, b.id], "expected": expected})
    assert response.status_code == 409
    assert response.json()["detail"] == "the selection changed since the check; review the swaps again"
    assert not fake.requests


def test_a_cell_rendered_since_the_preview_is_left_out_and_counted(client, fake, store):
    a = plant(store, "red hair, long hair", seed=1)
    b = plant(store, "red hair, long hair, smile", seed=2)
    expected = preview(client, ids=[a.id, b.id])["ids"]
    # Rendered elsewhere meanwhile: the very cell b would make.
    plant(store, "silver hair, smile", seed=2)
    response = client.post("/api/swap-tags", json={"swaps": SWAP, "ids": [a.id, b.id], "expected": expected})
    assert response.status_code == 200, response.text
    assert (response.json()["queued"], response.json()["pruned"]) == (1, 1)
    stream_job(client, response.json()["job_id"])
    assert [r.prompt for r in fake.requests] == ["silver hair"]


def test_pruned_counts_every_cell_held_at_queue_time(client, fake, store):
    # What `pruned` means: the preview's "already rendered" ones too, not
    # only those rendered since it.
    a = plant(store, "red hair, long hair", seed=1)
    b = plant(store, "red hair, long hair, smile", seed=2)
    plant(store, "silver hair, smile", seed=2)
    answer = preview(client, ids=[a.id, b.id])
    assert (answer["queued"], answer["counts"].get("rendered")) == (1, 1)
    response = client.post("/api/swap-tags", json={"swaps": SWAP, "ids": [a.id, b.id], "expected": answer["ids"]})
    assert response.status_code == 200, response.text
    assert (response.json()["queued"], response.json()["pruned"]) == (1, 1)
    stream_job(client, response.json()["job_id"])


def test_one_image_never_refuses_the_rest(client, fake, store):
    good = plant(store, "red hair, long hair, smile", seed=1)
    emptied = plant(store, "red hair, long hair", seed=2)
    twin = plant(store, "long hair, red hair, smile", seed=1)
    gone_source = plant(store, "red hair, long hair, tree", seed=3, refine=good.id, strength=0.5)
    adapter = plant(store, "red hair, long hair, sky", seed=4, lora="nowhere")
    other = plant(store, "red hair, long hair", seed=5, model_id="somebody/else")
    swaps = [{"find": "red hair, long hair", "replace": ""}]
    answer = client.post("/api/swap-tags/preview", json={
        "swaps": swaps, "ids": [good.id, emptied.id, twin.id, gone_source.id, adapter.id, other.id],
    }).json()
    images = by_id(answer)
    assert images[good.id]["status"] == "queued"
    assert images[emptied.id]["status"] == "invalid" and "prompt" in images[emptied.id]["detail"]
    assert (images[twin.id]["status"], images[twin.id]["detail"]) == ("duplicate", f"same as #{good.id}")
    assert images[adapter.id]["status"] == "lora_missing"
    assert images[other.id]["status"] == "foreign"
    assert "somebody/else" in images[other.id]["detail"]
    # The source exists here; deleted, the pass that refines it cannot run.
    assert images[gone_source.id]["status"] == "queued"
    assert client.delete(f"/api/images/{good.id}").status_code == 200
    images = by_id(client.post("/api/swap-tags/preview", json={
        "swaps": swaps, "ids": [twin.id, gone_source.id],
    }).json())
    assert images[gone_source.id]["status"] == "source_deleted"
    assert images[twin.id]["status"] == "queued"
    # Nothing at all queueable is a 422 that counts why.
    response = client.post("/api/swap-tags", json={"swaps": swaps, "ids": [emptied.id, other.id]})
    assert response.status_code == 422
    assert response.json()["detail"] == "no selected image would change (1 invalid, 1 foreign)"


def test_without_ids_the_selection_is_the_gallerys_filter(client, fake, store):
    a = plant(store, "red hair, long hair, smile", seed=1)
    plant(store, "red hair, long hair, frown", seed=2)
    client.post(f"/api/images/{a.id}/favorite?value=true")
    assert preview(client)["ids"] == [a.id]
    assert preview(client, favorites=False, search="frown")["queued"] == 1
    response = client.post("/api/swap-tags/preview", json={"swaps": SWAP, "favorites": False, "search": "seed:999"})
    assert response.status_code == 404


def test_the_negative_prompt_is_swapped_only_when_asked(client, fake, store):
    a = plant(store, "red hair, long hair", seed=1, negative_prompt="red hair, long hair, blurry")
    images = by_id(preview(client, ids=[a.id], field="negative_prompt"))
    assert images[a.id]["text"] == "silver hair, blurry"
    response = client.post("/api/swap-tags", json={"swaps": SWAP, "ids": [a.id], "field": "negative_prompt"})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    assert (ran.prompt, ran.negative_prompt) == ("red hair, long hair", "silver hair, blurry")


def test_a_checkpoint_with_no_negative_prompt_refuses_that_field(client, fake, store, monkeypatch):
    monkeypatch.setattr(main, "_merged_defaults", lambda info: {"negative_prompt_supported": False})
    response = client.post("/api/swap-tags/preview", json={
        "swaps": SWAP, "ids": [1], "field": "negative_prompt",
    })
    assert response.status_code == 422 and "takes no negative prompt" in response.json()["detail"]


def test_rows_are_checked_before_anything_is_read(client):
    response = client.post("/api/swap-tags/preview", json={
        "swaps": [{"find": "red hair", "replace": "x"}, {"find": "Red_Hair, smile", "replace": "y"}],
    })
    assert response.status_code == 422 and "red hair is in two rows' Find" in response.text
    assert client.post("/api/swap-tags/preview", json={"swaps": SWAP, "ids": [0]}).status_code == 422
    too_many = list(range(1, settings.max_grid_images + 2))
    assert client.post("/api/swap-tags/preview", json={"swaps": SWAP, "ids": too_many}).status_code == 422


def test_the_queue_is_a_model_dependent_request_and_the_preview_is_not(client, fake, store):
    a = plant(store, "red hair, long hair", seed=1)
    stale = {"X-AuraDream-Model-Revision": "999"}
    body = {"swaps": SWAP, "ids": [a.id]}
    refused = client.post("/api/swap-tags", json=body, headers=stale)
    assert refused.status_code == 409 and "checkpoint changed" in refused.json()["detail"]
    assert client.post("/api/swap-tags/preview", json=body, headers=stale).status_code == 200


def test_a_rerender_of_mixed_sizes_is_still_one_job_at_one_size(client, fake, store):
    # Re-render builds its cells through the helper Swap tags shares
    # (grid.cells_from_records); the size it asks for still wins.
    run_one(client, prompt="first", width=256, height=256, seed=1)
    run_one(client, prompt="second", width=384, height=256, seed=2)
    fake.requests.clear()
    response = client.post("/api/rerender", json={"width": 512, "height": 512, "favorites": False})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    assert [(r.prompt, r.seed, r.width, r.height) for r in fake.requests] == [
        ("first", 1, 512, 512), ("second", 2, 512, 512),
    ]


def test_the_selection_resolver_reports_ids_gone_and_sorts_oldest_first(client, fake, store):
    import asyncio

    a = plant(store, "one", seed=1)
    b = plant(store, "two", seed=2)
    records, gone, binned = asyncio.run(submission.resolve_selection(
        main.services,
        ids=[b.id, 77, a.id, b.id], search=None, favorites=False, all_models=False,
    ))
    assert ([r.id for r in records], gone, binned) == ([a.id, b.id], [77], set())
    # A binned one is gone from the selection, and the third list says why.
    store.bin_images([a.id])
    records, gone, binned = asyncio.run(submission.resolve_selection(
        main.services,
        ids=[b.id, 77, a.id], search=None, favorites=False, all_models=False,
    ))
    assert ([r.id for r in records], gone, binned) == ([b.id], [a.id, 77], {a.id})


def test_a_foreign_record_through_all_models_is_reported_not_refused(client, fake, store):
    # Re-render refuses the whole selection; here the one image is left out.
    plant(store, "red hair, long hair", seed=1, model_id="somebody/else")
    mine = plant(store, "red hair, long hair, smile", seed=2)
    answer = preview(client, favorites=False, all_models=True)
    images = by_id(answer)
    assert images[mine.id]["status"] == "queued" and answer["counts"]["foreign"] == 1


def test_a_picked_image_binned_since_says_it_is_in_the_bin(client, fake, store):
    a = plant(store, "1girl, red hair, long hair", seed=1)
    b = plant(store, "1girl, red hair, smile", seed=2)
    source = plant(store, "1girl, red hair", seed=3)
    passed = plant(store, "1girl, red hair, long hair", seed=4, refine=source.id, strength=0.5)
    store.bin_images([b.id, source.id])
    images = by_id(preview(client, ids=[a.id, b.id, passed.id, 999]))
    assert (images[b.id]["status"], images[b.id]["detail"]) == ("deleted", "in the recycle bin")
    assert (images[999]["status"], images[999]["detail"]) == ("deleted", "no longer in the gallery")
    # A pass whose source is binned cannot be made again until a Restore.
    assert images[passed.id]["status"] == "source_deleted"
    assert "recycle bin" in images[passed.id]["detail"]
    assert images[a.id]["status"] == "queued"
