"""Queue time estimates (GEN#32): each job priced once at submission from
the gallery's matching history, the same figure its preview gives, stored
on the job and only reported by its snapshot."""
from __future__ import annotations

import io
import json

from PIL import Image

from backend import gallery_reports, grid, main, refine, storage
from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import run_one, stream_job

BASE = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, "steps": 8}


def plant(store, *, width=256, height=256, steps=8, duration_ms=4000, seed=1, **fields):
    """A timed row: what the history prices from. The fake renders in no
    time, so its own rows (duration 0) never count."""
    return store.save_image(
        Image.new("RGB", (width, height)),
        GenerateRequest(prompt="x", width=width, height=height, steps=steps, **fields),
        seed=seed, duration_ms=duration_ms, model_id=settings.model_id,
    )


def accepted(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    answer = response.json()
    stream_job(client, answer["job_id"])
    return answer


def stored(client, job_id):
    return client.get(f"/api/jobs/{job_id}").json()["estimate_seconds"]


def test_a_generate_with_batch_and_repeat_stores_its_previews_figure(client, store):
    plant(store)  # 500 ms a step at 256x256
    preview = client.post("/api/grid/preview", json={**BASE, "num_images": 2, "seeds": 3, "seed": 9}).json()
    answer = accepted(client, "/api/generate", {**BASE, "num_images": 2, "runs": 3, "seed": 9})
    # 0.5 s x 8 steps x 6 images, in the answer and on the job.
    assert preview["seconds"] == answer["estimate_seconds"] == stored(client, answer["job_id"]) == 24.0


def test_a_grid_stores_its_previews_figure(client, store):
    plant(store)
    body = {**BASE, "steps_axis": "8,16", "seeds": 2, "seed": 3}
    preview = client.post("/api/grid/preview", json=body).json()
    answer = accepted(client, "/api/grid", body)
    assert preview["seconds"] == answer["estimate_seconds"] == stored(client, answer["job_id"]) == 24.0


def test_then_refine_prices_the_base_and_each_pass(client, store):
    source = plant(store)
    # The stand-in pass: 384x384 at 1.5x, its own refine history there.
    plant(store, width=384, height=384, refine=source.id, strength=0.5, duration_ms=3000, seed=2)
    spec = {"scale": 1.5, "strength": 0.5}
    preview = client.post("/api/grid/preview", json={**BASE, "num_images": 2, "seed": 4, "then_refine": spec}).json()
    answer = accepted(client, "/api/generate", {**BASE, "num_images": 2, "seed": 4, "then_refine": spec})
    # Two 4 s images and two 3 s passes.
    assert preview["seconds"] == answer["estimate_seconds"] == 14.0


def test_a_refine_pass_and_an_inpaint_are_priced_from_their_own_histories(client, store):
    source = plant(store)
    plant(store, refine=source.id, strength=0.5, duration_ms=3000, seed=2)
    body = {**BASE, "refine": source.id, "strength": 0.5, "seed": 5}
    preview = client.post("/api/grid/preview", json=body).json()
    answer = accepted(client, "/api/generate", body)
    assert preview["seconds"] == answer["estimate_seconds"] == 3.0
    mask = Image.new("L", (256, 256), 0)
    mask.paste(255, (0, 0, 128, 64))
    out = io.BytesIO()
    mask.save(out, format="PNG")
    hashed = client.post(
        "/api/masks", params={"source": source.id}, content=out.getvalue(), headers={"content-type": "image/png"},
    ).json()["hash"]
    inpaint = {**body, "mask": hashed}
    # No inpaint has been timed: a refine pass's history does not price one.
    assert accepted(client, "/api/generate", inpaint)["estimate_seconds"] is None
    plant(store, refine=source.id, strength=0.5, duration_ms=5000, seed=3, mask=hashed)
    preview = client.post("/api/grid/preview", json=inpaint).json()
    answer = accepted(client, "/api/generate", {**inpaint, "seed": 6})
    assert preview["seconds"] == answer["estimate_seconds"] == 5.0


def test_an_outpaint_is_priced_as_the_stamped_cell_it_runs(client, store):
    source = plant(store)
    spec = {"left": 32, "right": 16, "band": 16}
    body = {**BASE, "width": 304, "refine": source.id, "strength": 0.9, "outpaint": spec, "seed": 7}
    assert accepted(client, "/api/generate", body)["estimate_seconds"] is None
    (cell,) = storage.stamp_outpaints([grid.Cell(request=GenerateRequest(**{**body, "prompt": "x"}), seed=1)])
    store.save_image(Image.new("RGB", (304, 256)), cell.request, seed=1, duration_ms=6000, model_id=settings.model_id)
    preview = client.post("/api/grid/preview", json=body).json()
    answer = accepted(client, "/api/generate", {**body, "seed": 8})
    assert preview["seconds"] == answer["estimate_seconds"] == 6.0


def test_mass_refine_and_more_seeds_store_their_own_previews_figures(client, fake, store):
    plant(store)
    source = plant(store, seed=2)
    plant(store, refine=source.id, strength=0.5, duration_ms=3000, seed=3)
    preview = client.post("/api/refine/batch/preview", json={"ids": [source.id], "strength": 0.5, "scale": 1.0}).json()
    answer = accepted(client, "/api/refine/batch", {"ids": [source.id], "strength": 0.5, "scale": 1.0})
    assert preview["seconds"] == answer["estimate_seconds"] == stored(client, answer["job_id"]) == 3.0
    _, events = run_one(client, **{**BASE, "seed": 40, "num_images": 2})
    picked = events[-1]["images"][0]["id"]
    preview = client.post("/api/more-seeds/preview", json={"ids": [picked], "count": 2}).json()
    answer = accepted(client, "/api/more-seeds", {"ids": [picked], "count": 2})
    assert preview["seconds"] == answer["estimate_seconds"] == stored(client, answer["job_id"]) == 8.0


def test_no_matching_history_an_upscaler_or_an_edit_is_no_estimate(client, store):
    plant(store)
    # One cell at a size nothing has run at makes the whole job unpriced.
    answer = accepted(client, "/api/grid", {**BASE, "size_axis": "256x256,320x256", "seed": 1})
    assert answer["estimate_seconds"] is None and stored(client, answer["job_id"]) is None
    source = plant(store, seed=2)
    plant(store, refine=source.id, strength=0.5, duration_ms=3000, seed=3)
    pass_ = GenerateRequest(**BASE, refine=source.id, strength=0.5)
    assert gallery_reports.job_seconds([grid.Cell(request=pass_, seed=1)], settings.model_id) == 3.0
    upscaled = pass_.model_copy(update={"upscaler": refine.ANIME6B})
    assert gallery_reports.job_seconds([grid.Cell(request=upscaled, seed=1)], settings.model_id) is None
    edit = GenerateRequest(**BASE).model_copy(update={"edit": object()})
    assert gallery_reports.job_seconds([grid.Cell(request=edit, seed=1)], settings.model_id) is None


def test_a_pricing_failure_still_queues_the_job_unpriced(client, store, monkeypatch, caplog):
    plant(store)

    def broken(*args, **kwargs):
        raise RuntimeError("the history is unreadable")

    monkeypatch.setattr(gallery_reports, "job_seconds", broken)
    answer = accepted(client, "/api/generate", {**BASE, "seed": 1})
    assert answer["estimate_seconds"] is None
    done = client.get(f"/api/jobs/{answer['job_id']}").json()
    assert done["status"] == "done" and done["estimate_seconds"] is None
    assert "no time estimate (the history is unreadable)" in caplog.text


def test_the_snapshot_reports_the_stored_figure_without_reading_the_gallery(client, store, monkeypatch):
    plant(store)
    answer = accepted(client, "/api/generate", {**BASE, "seed": 1})
    assert answer["estimate_seconds"] == 4.0

    def refuse(*args, **kwargs):
        raise AssertionError("the snapshot read the gallery")

    monkeypatch.setattr(storage, "_session", refuse)
    for reader in ("execution_estimates", "job_seconds", "step_cost_ms", "refine_cost_ms", "image_bytes"):
        monkeypatch.setattr(gallery_reports, reader, refuse)
    assert main.jobs._jobs[answer["job_id"]].snapshot()["estimate_seconds"] == 4.0
    listed = client.get("/api/jobs", params={"include_finished": True}).json()
    assert [job["estimate_seconds"] for job in listed if job["job_id"] == answer["job_id"]] == [4.0]
    assert stored(client, answer["job_id"]) == 4.0


def test_the_estimate_is_no_reproduction_field(client, store):
    body = {**BASE, "num_images": 2, "seed": 11}
    unpriced = accepted(client, "/api/generate", body)
    # Three timed rows: the unpriced job's two can carry a few ms of the
    # fake's own time, and the median must still be the planted rows'.
    for seed in (1, 2, 3):
        plant(store, seed=seed)
    priced = accepted(client, "/api/generate", body)
    assert (unpriced["estimate_seconds"], priced["estimate_seconds"]) == (None, 8.0)
    assert unpriced["run_id"] == priced["run_id"]
    jobs = [main.jobs._jobs[answer["job_id"]] for answer in (unpriced, priced)]
    assert [grid.cell_keys(cell) for cell in jobs[0].cells] == [grid.cell_keys(cell) for cell in jobs[1].cells]
    assert main.backend.applied(jobs[0].request) == main.backend.applied(jobs[1].request)
    assert "estimate_seconds" not in GenerateRequest.model_fields
    with storage._session() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
    assert not any("estimate" in column for column in columns)
    newest = client.get("/api/gallery").json()["items"][0]
    with Image.open(store.settings.outputs_dir / newest["filename"]) as png:
        chunk = json.loads(png.info["auradream"])
    assert not any("estimate" in key for key in chunk)


def test_swap_tags_answer_carries_its_jobs_figure(client, fake, store):
    plant(store)
    _, events = run_one(client, **{**BASE, "prompt": "red hair, smile"}, seed=50)
    picked = events[-1]["images"][0]["id"]
    answer = accepted(client, "/api/swap-tags", {"ids": [picked], "swaps": [{"find": "red hair", "replace": "silver hair"}]})
    assert answer["estimate_seconds"] == stored(client, answer["job_id"]) == 4.0
    assert answer["more"] == []
