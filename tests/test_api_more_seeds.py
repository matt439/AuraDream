"""More seeds over HTTP: the selection grouped into batches, each batch's
next seeds and Batch, the jobs that join the runs they extend, and the
per-batch statuses."""
from __future__ import annotations

import dataclasses
import json
import threading

from PIL import Image

from backend import gallery_reports, more_seeds
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
        chunk = json.loads(png.info["auradream"])
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


def test_a_run_scope_is_every_live_image_of_the_run_starred_or_not(client, fake):
    _, events = run_grid(client, seed=50, seeds=2, cfg_axis="1.5,2.5,3.5", steps=2)
    images = events[-1]["images"]
    run = images[0]["run_id"]
    by = {(image["guidance_scale"], image["seed"]): image for image in images}
    # Another run, which the scope leaves out.
    batch_of(client, seed=50, num_images=1, runs=1, prompt="a harbour at dawn")
    # Starred holds one image; the run scope ignores it.
    client.post(f"/api/images/{by[(1.5, 50)]['id']}/favorite")
    # A cell wholly in the bin is no batch; a binned top seed still holds its seed.
    client.post("/api/images/bin", json={"ids": [by[(3.5, 50)]["id"], by[(3.5, 51)]["id"], by[(2.5, 51)]["id"]]})
    live = sorted(by[key]["id"] for key in ((1.5, 50), (1.5, 51), (2.5, 50)))
    answer = preview(client, run=run, count=2)
    assert answer["ids"] == live
    assert (answer["batches"], answer["jobs"], answer["queued"]) == (2, 1, 4)
    assert {(r["seed_low"], r["seed_high"], r["first"], r["last"]) for r in answer["results"]} == {
        (50, 51, 52, 53),
    }
    assert answer["count_default"] == 1
    assert answer["headline"] == "2 more of each of 2 cells = 4 images · seeds 52–53 · Starred ignored"
    unticked = preview(client, run=run, count=2, favorites=False)
    assert unticked["headline"] == "2 more of each of 2 cells = 4 images · seeds 52–53"
    # The filter and its Starred give the one starred image instead.
    assert preview(client, count=2)["ids"] == [by[(1.5, 50)]["id"]]
    fake.requests.clear()
    accepted = queue(client, run=run, count=2, expected=answer["ids"])
    assert accepted["run_id"] == run and accepted["queued"] == 4
    final = stream_job(client, accepted["job_id"])[-1]
    assert final["status"] == "done"
    assert {image["run_id"] for image in final["images"]} == {run}
    assert sorted((r.guidance_scale, r.seed) for r in fake.requests) == [
        (1.5, 52), (1.5, 53), (2.5, 52), (2.5, 53),
    ]


def test_a_run_scope_alone_and_its_refusals(client, fake, store):
    images = batch_of(client, seed=10, num_images=2, runs=1)
    run = images[0]["run_id"]
    answer = preview(client, run=run, count=2)
    # One batch: its own Batch, and a headline for its one cell.
    assert answer["count_default"] == 2
    assert answer["headline"] == "2 more of its one cell = 2 images · seeds 12–13 · Starred ignored"
    # The page's other scopes get the count, never the headline.
    assert preview(client, ids=[images[0]["id"]], count=2)["headline"] is None
    for clash in ({"ids": [images[0]["id"]]}, {"search": "lighthouse"}):
        refused = client.post("/api/more-seeds/preview", json={"run": run, **clash})
        assert refused.status_code == 422, clash
    unknown = client.post("/api/more-seeds/preview", json={"run": "nope"})
    assert (unknown.status_code, unknown.json()["detail"]) == (404, "run nope is not in the gallery")
    stale = client.post("/api/more-seeds", json={"run": run, "expected": [images[0]["id"]]})
    assert stale.status_code == 409
    client.post("/api/images/bin", json={"ids": [image["id"] for image in images]})
    binned = client.post("/api/more-seeds/preview", json={"run": run})
    assert (binned.status_code, binned.json()["detail"]) == (404, f"every image of run {run} is in the recycle bin")


def test_a_run_scope_reports_another_checkpoints_batch_and_its_batches_disagreeing(client, fake, store):
    quantization = current_settings(settings).quantization
    mine = plant(store, "a lighthouse", seed=1, run_id="mixed", batch=2, quantization=quantization)
    other = plant(store, "a harbour", seed=1, run_id="mixed", batch=3, quantization=quantization)
    foreign = plant(store, "a lighthouse", seed=2, run_id="mixed", model_id="somebody/else")
    answer = preview(client, run="mixed", count=1)
    statuses = {row["id"]: row["status"] for row in answer["results"]}
    assert statuses == {mine.id: "queued", other.id: "queued", foreign.id: "foreign"}
    # Batch 2 and Batch 3: no one count is the run's.
    assert answer["count_default"] is None
    assert answer["headline"] == "1 more of each of 2 cells = 2 images · seed 2 · Starred ignored"


def test_a_run_past_max_grid_images_is_queued_in_few_batches(client, fake, monkeypatch):
    from backend import main

    _, events = run_grid(client, seed=50, seeds=2, cfg_axis="1.5,2.5", steps=2)
    run = events[-1]["images"][0]["run_id"]
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, max_grid_images=3))
    # Four images: the filter refuses them by count; the run takes them.
    assert client.post("/api/more-seeds/preview", json={"favorites": False}).status_code == 422
    answer = preview(client, run=run, count=1)
    assert len(answer["ids"]) == 4 and answer["queued"] == 2
    assert stream_job(client, queue(client, run=run, count=1)["job_id"])[-1]["status"] == "done"
    # The batches x count bound still holds one press.
    too_many = client.post("/api/more-seeds/preview", json={"run": run, "count": 2})
    assert too_many.status_code == 422
    assert too_many.json()["detail"] == "2 batches × 2 is 4 images; a press may queue 3. Ask for fewer."


def test_an_edit_batch_replays_its_intent_receipt_without_fresh_seeds(client, fake, store, monkeypatch):
    # More seeds of an instruction edit (stage 4): the same receipt answers a
    # retried request, so a lost reply never edits the source twice over.
    from tests.test_instruction_edit_queue import edited, open_gate, source

    open_gate(fake, monkeypatch)
    _, bound = source(store)
    edit = edited(store, bound)
    answer = preview(client, ids=[edit.id], count=2)
    assert answer["queued"] == 2 and answer["intent"]
    body = {"ids": [edit.id], "count": 2, "expected": answer["ids"], "intent": answer["intent"]}
    accepted = queue(client, **body)
    final = stream_job(client, accepted["job_id"])[-1]
    assert [image["seed"] for image in final["images"]] == [8, 9]
    assert queue(client, **body) == accepted
    assert [call["seed"] for call in fake.edit_calls] == [8, 9]
    assert {store.get_image(image["id"]).edit for image in final["images"]} == {bound}


# ------------------------------------------------- Then refine's pass carried

SPEC = {"scale": 1.5, "strength": 0.5}


def then_refine_batch(client, *, seed=500, prompt="a fox at dusk", width=256, spec=SPEC, num_images=2):
    """A Then refine Generate of `num_images` at `seed`: its final snapshot."""
    response = client.post("/api/generate", json={
        "prompt": prompt, "width": width, "height": width, "steps": 8, "seed": seed,
        "num_images": num_images, "then_refine": spec,
    })
    assert response.status_code == 200, response.text
    final = stream_job(client, response.json()["job_id"])[-1]
    assert final["status"] == "done", final
    return final


def rows(store):
    return sorted(store.list_images(limit=200)[0], key=lambda row: row.id)


def test_more_seeds_of_a_then_refine_run_queues_each_base_with_its_pass(client, fake, store, monkeypatch):
    made = then_refine_batch(client)
    run, pass_run = made["run_id"], made["pass_run_id"]
    bases = [row for row in rows(store) if row.refine is None]
    assert len(bases) == 2
    answer = preview(client, ids=[bases[0].id], count=2)
    (row,) = answer["results"]
    assert (answer["queued"], answer["passes"], answer["jobs"]) == (2, 2, 1)
    assert (answer["then_refine_batches"], answer["then_refine_hold"]) == (1, None)
    assert row["status"] == "queued" and row["notes"] == ["each then refined at 1.5× at 0.5"]
    # Priced with its passes, each at the pass size: 2 images x 8 steps x 1 s,
    # and two 384 passes of 4 denoised steps at 3 s each.
    monkeypatch.setattr(gallery_reports, "step_cost_ms", lambda w, h, *a: {256: 1000}.get(w))
    monkeypatch.setattr(gallery_reports, "refine_cost_ms", lambda w, h, *a, **kw: {4: 3000} if w == 384 else {})
    priced = preview(client, ids=[bases[0].id], count=2)
    bases_only = preview(client, ids=[bases[0].id], count=2, then_refine=False)
    assert (priced["seconds"], bases_only["seconds"]) == (22.0, 16.0)

    before = {row.id for row in rows(store)}
    accepted = queue(client, ids=[bases[0].id], count=2, expected=answer["ids"])
    assert (accepted["queued"], accepted["passes"], accepted["run_id"]) == (2, 2, run)
    final = stream_job(client, accepted["job_id"])[-1]
    assert final["status"] == "done" and final["pass_run_id"] == pass_run
    new = [row for row in rows(store) if row.id not in before]
    # Each new base in the base run, then its pass in the run's own pass run.
    assert [(row.refine is None, row.seed) for row in new] == [(True, 502), (True, 503), (False, 502), (False, 503)]
    assert {row.run_id for row in new[:2]} == {run}
    assert [row.refine for row in new[2:]] == [new[0].id, new[1].id]
    assert {(row.run_id, row.width, row.strength) for row in new[2:]} == {(pass_run, 384, 0.5)}


def test_then_refine_off_makes_bases_only_and_a_later_press_still_carries_it(client, fake, store):
    made = then_refine_batch(client, seed=600)
    bases = [row for row in rows(store) if row.refine is None]
    off = preview(client, ids=[bases[0].id], count=2, then_refine=False)
    # The tick still shows: the run carries a pass, which this press leaves out.
    assert (off["passes"], off["then_refine_batches"], off["results"][0]["notes"]) == (0, 1, [])
    before = {row.id for row in rows(store)}
    final = stream_job(client, queue(client, ids=[bases[0].id], count=2, then_refine=False)["job_id"])[-1]
    assert final["then_refine"] is None
    new = [row for row in rows(store) if row.id not in before]
    assert [row.refine for row in new] == [None, None]
    # Its template now a base of that job, which carried no spec: the
    # batch's mates' jobs still give the run's.
    again = preview(client, ids=[new[-1].id], count=2)
    assert again["passes"] == 2 and again["results"][0]["notes"] == ["each then refined at 1.5× at 0.5"]
    final = stream_job(client, queue(client, ids=[new[-1].id], count=2)["job_id"])[-1]
    assert final["pass_run_id"] == made["pass_run_id"]
    passes = [row for row in rows(store) if row.refine is not None and row.seed in (604, 605)]
    assert {row.run_id for row in passes} == {made["pass_run_id"]} and len(passes) == 2


def test_a_plain_run_and_a_pass_batch_are_unchanged(client, fake, store):
    plain = batch_of(client, seed=100, num_images=1, runs=1)
    answer = preview(client, ids=[plain[0]["id"]], count=1)
    assert (answer["passes"], answer["then_refine_batches"], answer["then_refine_hold"]) == (0, 0, None)
    assert answer["results"][0]["notes"] == []
    then_refine_batch(client, seed=700, num_images=1)
    (pass_row,) = [row for row in rows(store) if row.refine is not None]
    # A pass batch: more passes of its source at new seeds, as before; its
    # job's spec is no Then refine of a pass.
    answer = preview(client, ids=[pass_row.id], count=1)
    assert (answer["queued"], answer["passes"], answer["then_refine_batches"]) == (1, 0, 0)
    final = stream_job(client, queue(client, ids=[pass_row.id], count=1)["job_id"])[-1]
    assert final["then_refine"] is None
    (made,) = [row for row in rows(store) if row.seed == 701]
    assert made.refine == pass_row.refine and made.run_id == pass_row.run_id


def test_no_refine_pass_on_the_checkpoint_holds_the_tick_and_makes_bases(client, fake, store, monkeypatch):
    from backend import submission

    then_refine_batch(client, seed=750, num_images=1)
    (base,) = [row for row in rows(store) if row.refine is None]

    async def none(services):
        return "this checkpoint has no refine pass in this app"

    monkeypatch.setattr(submission, "refine_unavailable", none)
    answer = preview(client, ids=[base.id], count=1)
    assert answer["then_refine_hold"] == "this checkpoint has no refine pass in this app"
    assert (answer["then_refine_batches"], answer["queued"], answer["passes"]) == (1, 1, 0)
    assert answer["results"][0]["notes"] == ["bases only: this checkpoint has no refine pass"]
    final = stream_job(client, queue(client, ids=[base.id], count=1)["job_id"])[-1]
    assert final["status"] == "done" and final["then_refine"] is None


def test_mates_of_two_specs_are_refused_in_words_and_off_still_queues(client, fake, store):
    from tests.test_api_finish_run import cell, retain

    quantization = current_settings(settings).quantization
    for seed, (scale, strength) in ((1, (1.5, 0.4)), (2, (2.0, 0.35))):
        item = cell(seed, steps=8)
        retain(store, [item], run="two-specs", metadata={"then_refine": {
            "scale": scale, "strength": strength, "steps": None, "pass_run_id": f"pass-{seed}",
        }})
        job_id = store.retained_run_jobs("two-specs")[-1]["job_id"]
        store.save_image(Image.new("RGB", (256, 256)), item.request, seed=seed, duration_ms=1,
                         model_id=settings.model_id, quantization=quantization, run_id="two-specs",
                         job_id=job_id, batch=1)
    answer = preview(client, run="two-specs", count=1)
    (row,) = answer["results"]
    assert row["status"] == "then_refine" and answer["counts"] == {"then_refine": 1}
    assert row["detail"] == (
        "its images were refined at 1.5× at 0.4 and at 2× at 0.35; untick Refine each new image for bases only, "
        "and Mass refine them at the numbers wanted"
    )
    assert answer["then_refine_batches"] == 1 and answer["queued"] == 0
    refused = client.post("/api/more-seeds", json={"run": "two-specs", "count": 1})
    assert refused.status_code == 422 and refused.json()["detail"].endswith("(1 then refine)")
    # The run's More seeds action is the bases': the dialog can untick it.
    assert client.get("/api/runs/two-specs").json()["actions"]["seeds"]["available"]
    off = preview(client, run="two-specs", count=1, then_refine=False)
    assert (off["queued"], off["passes"], off["results"][0]["status"]) == (1, 0, "queued")


def test_a_batch_whose_pass_is_past_the_guard_is_refused_while_the_rest_queue(client, fake, store, monkeypatch):
    from backend import then_refine

    small = then_refine_batch(client, seed=800, num_images=1)
    large = then_refine_batch(client, seed=900, num_images=1, prompt="a harbour at dawn", width=384)
    picks = [row.id for row in rows(store) if row.refine is None]
    # A 384 base's 1.5x pass is 576: past a guard of 500 pixels a side.
    monkeypatch.setattr(then_refine, "too_big", lambda width, height: max(width, height) > 500)
    answer = preview(client, ids=picks, count=1)
    statuses = {row["id"]: (row["status"], row["detail"]) for row in answer["results"]}
    assert statuses[picks[0]] == ("queued", None)
    status, detail = statuses[picks[1]]
    assert status == "then_refine" and detail.startswith("its run's pass: 384×384 at 1.5× is 576×576, past the ")
    assert (answer["queued"], answer["passes"], answer["jobs"]) == (1, 1, 1)
    final = stream_job(client, queue(client, ids=picks, count=1)["job_id"])[-1]
    assert final["run_id"] == small["run_id"] and final["pass_run_id"] == small["pass_run_id"]
    assert large["run_id"] != small["run_id"]


def test_two_specs_in_one_press_make_two_jobs_each_in_its_pass_run(client, fake, store):
    first = then_refine_batch(client, seed=1000, num_images=1)
    second = then_refine_batch(client, seed=1100, num_images=1, prompt="a harbour at dawn",
                               spec={"scale": 1.25, "strength": 0.4})
    picks = [row.id for row in rows(store) if row.refine is None]
    answer = preview(client, ids=picks, count=1)
    assert (answer["queued"], answer["passes"], answer["jobs"]) == (2, 2, 2)
    notes = {row["id"]: row["notes"] for row in answer["results"]}
    assert notes == {picks[0]: ["each then refined at 1.5× at 0.5"], picks[1]: ["each then refined at 1.25× at 0.4"]}
    accepted = queue(client, ids=picks, count=1, expected=answer["ids"])
    assert accepted["passes"] == 2 and len(accepted["more"]) == 1
    finals = [stream_job(client, job["job_id"])[-1] for job in (accepted, *accepted["more"])]
    assert [(f["run_id"], f["pass_run_id"]) for f in finals] == [
        (first["run_id"], first["pass_run_id"]), (second["run_id"], second["pass_run_id"]),
    ]
    made = sorted(
        (row.width, row.run_id) for row in rows(store) if row.refine is not None and row.seed in (1001, 1101)
    )
    assert made == sorted([(384, first["pass_run_id"]), (320, second["pass_run_id"])])
