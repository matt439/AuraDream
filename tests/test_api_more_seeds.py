"""More seeds over HTTP: the selection grouped into batches, each batch's
next seeds and Batch, the jobs that join the runs they extend, and the
per-batch statuses."""
from __future__ import annotations

import dataclasses
import json
import threading

from PIL import Image

from backend import more_seeds
from backend.config import settings
from backend.model_selection import current_settings
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid, run_one, stream_job


def preview(client, **body):
    response = client.post("/api/more-seeds/preview", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def queue(client, **body):
    response = client.post("/api/more-seeds", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def batch_of(client, *, seed, num_images=2, runs=2, **payload):
    """A Generate of `num_images` x `runs` at `seed`: its images, oldest first."""
    _, events = run_one(client, seed=seed, num_images=num_images, runs=runs, steps=2, **payload)
    final = events[-1]
    assert final["status"] == "done", final
    return final["images"]


def plant(store, prompt, *, seed, width=256, height=256, model_id=None, run_id=None, batch=None,
          quantization=None, **settings_):
    fields = {"steps": 2, "guidance_scale": 1.5, **settings_}
    return store.save_image(
        Image.new("RGB", (width, height)),
        GenerateRequest(prompt=prompt, width=width, height=height, **fields),
        seed=seed, duration_ms=1, model_id=model_id or settings.model_id, run_id=run_id, batch=batch,
        quantization=quantization,
    )


def test_one_picked_image_extends_its_whole_batch_at_its_own_batch(client, fake):
    images = batch_of(client, seed=100)
    assert [image["seed"] for image in images] == [100, 101, 102, 103]
    assert {image["batch"] for image in images} == {2}
    run = images[0]["run_id"]
    answer = preview(client, ids=[images[1]["id"]], count=4)
    assert answer["ids"] == [images[1]["id"]]
    assert (answer["queued"], answer["batches"], answer["jobs"], answer["total"]) == (4, 1, 1, 1)
    (row,) = answer["results"]
    assert row["status"] == "queued" and row["ids"] == [images[1]["id"]]
    # The whole batch, not just the picked image: its four images and seeds.
    assert (row["images"], row["seed_low"], row["seed_high"]) == (4, 100, 103)
    assert (row["first"], row["last"], row["queued"], row["skipped"]) == (104, 107, 4, 0)
    assert (row["batch"], row["recorded"], row["notes"]) == (2, True, [])
    fake.requests.clear()
    fake.seeds.clear()
    accepted = queue(client, ids=[images[1]["id"]], count=4, expected=answer["ids"])
    assert (accepted["queued"], accepted["batches"], accepted["more"]) == (4, 1, [])
    # It joins the run it extends.
    assert accepted["run_id"] == run
    final = stream_job(client, accepted["job_id"])[-1]
    assert final["status"] == "done" and final["more_seeds"] == 4 and final["grid"]
    assert final["planned_images"] == 4 and final["run_id"] == run
    # Two calls of the batch's Batch, consecutive seeds in each.
    assert fake.seeds == [104, 106]
    assert [r.num_images for r in fake.requests] == [2, 2]
    made = final["images"]
    assert [image["seed"] for image in made] == [104, 105, 106, 107]
    assert {image["run_id"] for image in made} == {run}
    assert {image["batch"] for image in made} == {2}
    assert {(image["prompt"], image["steps"], image["width"]) for image in made} == {
        (images[0]["prompt"], 2, 256),
    }
    # Asked again, it continues past the extension, which is in its run now.
    again = preview(client, ids=[images[0]["id"]], count=2)["results"][0]
    assert (again["images"], again["seed_high"], again["first"], again["last"]) == (8, 107, 108, 109)


def test_two_picked_images_of_one_batch_are_one_batch(client, fake):
    images = batch_of(client, seed=10)
    answer = preview(client, ids=[images[0]["id"], images[3]["id"]], count=3)
    assert (answer["queued"], answer["batches"], answer["total"]) == (3, 1, 1)
    row = answer["results"][0]
    assert row["ids"] == [images[0]["id"], images[3]["id"]]
    # Its row is the newest picked image's.
    assert row["id"] == images[3]["id"]
    assert (row["first"], row["last"]) == (14, 16)


def test_seeds_the_gallery_or_a_live_job_has_are_stepped_over(client, fake, store):
    images = batch_of(client, seed=200, num_images=1, runs=2)
    # The same settings at 202, made by another run: not made again.
    plant(
        store, images[0]["prompt"], seed=202, run_id="elsewhere", mu=images[0]["mu"],
        guidance_scale=images[0]["guidance_scale"],
    )
    row = preview(client, ids=[images[0]["id"]], count=2)["results"][0]
    assert (row["first"], row["last"], row["skipped"]) == (203, 204, 1)
    # A first press still running holds its seeds: a second continues past.
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        first = queue(client, ids=[images[0]["id"]], count=2)
        second = preview(client, ids=[images[0]["id"]], count=2)["results"][0]
        assert (second["first"], second["last"]) == (205, 206)
    finally:
        release.set()
    assert stream_job(client, first["job_id"])[-1]["status"] == "done"


def test_a_batch_whose_batch_was_not_recorded_runs_at_the_pages(client, fake, store, monkeypatch):
    # Under the server's quantizer, as one run's images all are.
    quantization = current_settings(settings).quantization
    old = [plant(store, "an old batch", seed=seed, run_id="old-run", quantization=quantization) for seed in (1, 2, 3)]
    row = preview(client, ids=[old[0].id], count=5, batch=3)["results"][0]
    assert (row["batch"], row["recorded"]) == (3, False)
    assert row["notes"] == ["its Batch was not recorded: the panel's 3"]
    assert (row["first"], row["last"]) == (4, 8)
    # Held to the pixel budget: two 256x256 images at a time.
    monkeypatch.setattr(more_seeds, "settings", dataclasses.replace(settings, max_batch_pixels=2 * 256 * 256))
    answer = preview(client, ids=[old[0].id], count=5, batch=3)
    row = answer["results"][0]
    assert row["batch"] == 2 and answer["unrecorded"] == 1
    assert row["notes"][-1] == "Batch 3 is past the pixel budget at 256×256: 2 at a time"
    fake.seeds.clear()
    accepted = queue(client, ids=[old[0].id], count=5, batch=3)
    final = stream_job(client, accepted["job_id"])[-1]
    assert fake.seeds == [4, 6, 8] and [r.num_images for r in fake.requests[-3:]] == [2, 2, 1]
    assert {image["run_id"] for image in final["images"]} == {"old-run"}
    assert [image["batch"] for image in final["images"]] == [2, 2, 2, 2, 1]
    # Its Batch is recorded now, and the call that ended it does not lower it.
    monkeypatch.setattr(more_seeds, "settings", settings)
    row = preview(client, ids=[old[0].id], count=2, batch=5)["results"][0]
    assert (row["batch"], row["recorded"], row["first"]) == (2, True, 9)


def test_a_grids_cells_are_batches_of_one_job(client, fake):
    _, events = run_grid(client, seed=50, seeds=2, cfg_axis="1.5,2.5", steps=2)
    images = events[-1]["images"]
    low = [image for image in images if image["guidance_scale"] == 1.5]
    high = [image for image in images if image["guidance_scale"] == 2.5]
    answer = preview(client, ids=[low[0]["id"], high[1]["id"]], count=2)
    # Two batches - the grid's other cells are not one - in one run's job.
    assert (answer["batches"], answer["jobs"], answer["queued"]) == (2, 1, 4)
    assert {(r["seed_low"], r["seed_high"]) for r in answer["results"]} == {(50, 51)}
    fake.requests.clear()
    accepted = queue(client, ids=[low[0]["id"], high[1]["id"]], count=2)
    final = stream_job(client, accepted["job_id"])[-1]
    assert final["run_id"] == images[0]["run_id"]
    # Each at its Batch of one: a call per seed.
    assert sorted((r.guidance_scale, r.seed) for r in fake.requests) == [
        (1.5, 52), (1.5, 53), (2.5, 52), (2.5, 53),
    ]


def test_a_job_per_run_and_the_statuses_of_what_cannot_queue(client, fake, store):
    first = batch_of(client, seed=300, num_images=1, runs=1)
    second = batch_of(client, seed=400, num_images=1, runs=1, prompt="a harbour at dawn")
    foreign = plant(store, "elsewhere", seed=5, model_id="somebody/else")
    adapter = plant(store, "with an adapter", seed=6, lora="nowhere")
    ids = [first[0]["id"], second[0]["id"], foreign.id, adapter.id, 9999]
    answer = preview(client, ids=ids, count=1)
    statuses = {row["id"]: row["status"] for row in answer["results"]}
    assert statuses == {
        first[0]["id"]: "queued", second[0]["id"]: "queued", foreign.id: "foreign",
        adapter.id: "lora_missing", 9999: "deleted",
    }
    assert answer["counts"] == {"queued": 2, "foreign": 1, "lora_missing": 1, "deleted": 1}
    assert (answer["batches"], answer["jobs"]) == (2, 2)
    assert [row["id"] for row in answer["results"]] == sorted(ids)
    accepted = queue(client, ids=ids, count=1, expected=answer["ids"])
    # One job per run, each stamped with the run it extends.
    jobs = [accepted, *accepted["more"]]
    assert [job["run_id"] for job in jobs] == [first[0]["run_id"], second[0]["run_id"]]
    for job in jobs:
        assert stream_job(client, job["job_id"])[-1]["status"] == "done"


def test_refusals_a_changed_selection_nothing_to_queue_and_too_many(client, fake, store, monkeypatch):
    images = batch_of(client, seed=1, num_images=1, runs=2)
    changed = client.post("/api/more-seeds", json={"ids": [images[0]["id"]], "expected": [images[1]["id"]]})
    assert changed.status_code == 409
    assert changed.json()["detail"] == "the selection changed since the check; review the batches again"
    foreign = plant(store, "elsewhere", seed=5, model_id="somebody/else")
    refused = client.post("/api/more-seeds", json={"ids": [foreign.id, 9999]})
    assert refused.status_code == 422
    assert refused.json()["detail"] == "no selected batch can have more seeds (1 foreign, 1 deleted)"
    other = batch_of(client, seed=900, num_images=1, runs=1, prompt="another prompt")
    too_many = client.post("/api/more-seeds/preview", json={
        "ids": [images[0]["id"], other[0]["id"]], "count": settings.max_grid_images,
    })
    assert too_many.status_code == 422
    assert too_many.json()["detail"].startswith(f"2 batches × {settings.max_grid_images} is ")


def test_the_filter_is_the_selection_without_ids(client, fake):
    batch_of(client, seed=70, num_images=1, runs=2, prompt="a lighthouse in fog")
    batch_of(client, seed=80, num_images=1, runs=1, prompt="a harbour at dawn")
    answer = preview(client, search="lighthouse", favorites=False, count=1)
    assert (answer["batches"], answer["queued"]) == (1, 1)
    assert answer["results"][0]["first"] == 72


def test_the_batch_survives_a_reindex(client, fake, store):
    images = batch_of(client, seed=5, num_images=3, runs=1)
    with Image.open(store.settings.outputs_dir / images[0]["filename"]) as png:
        chunk = json.loads(png.info["local-diffusion"])
    assert chunk["batch"] == 3
    record = store.index_file("copy.png", {**chunk, "batch": "3"}, 256, 256, store.datetime.now())
    # Garbage in a block written elsewhere is not a Batch.
    assert record.batch is None


def test_seeds_follow_on_past_a_binned_seed_and_a_binned_pick_says_so(client, fake):
    # The highest seed of the batch is in the bin: it was made, so the next
    # seeds start above it, and restoring it never duplicates one.
    images = batch_of(client, seed=200, num_images=1, runs=3)
    assert [image["seed"] for image in images] == [200, 201, 202]
    client.post("/api/images/bin", json={"ids": [images[2]["id"]]})
    answer = preview(client, ids=[images[0]["id"], images[2]["id"]], count=2)
    rows = {row["id"]: row for row in answer["results"]}
    gone = rows[images[2]["id"]]
    assert (gone["status"], gone["detail"]) == ("deleted", "in the recycle bin")
    row = rows[images[0]["id"]]
    assert row["status"] == "queued" and (row["first"], row["last"]) == (203, 204)
