"""The refine pass over HTTP: published in the status, refused at submit where
the checkpoint has none or the source is gone, an ordinary grid axis, and
recorded on the row at the strength it ran at."""
from __future__ import annotations

import threading

import pytest

from tests.api_helpers import run_grid, run_one, stream_job


def _first(client) -> dict:
    """An image to refine: the newest row of a gallery that has one."""
    run_one(client)
    return client.get("/api/gallery").json()["items"][0]


def test_status_publishes_the_capability(client, fake):
    body = client.get("/api/status").json()
    assert body["defaults"]["refine_supported"] is True
    live = body["hints"]["refine"]
    assert "Strength" in live

    fake.refine_supported = False
    body = client.get("/api/status").json()
    assert body["defaults"]["refine_supported"] is False
    # A dead control explains itself rather than repeating what a live one says.
    assert body["hints"]["refine"] != live
    assert "counterpart" in body["hints"]["refine"]


def test_before_a_load_the_profile_guesses(client, fake, monkeypatch):
    # `describe()` answers None until a pipeline is loaded - its class is the
    # answer - and the profile's guess from the id stands until then.
    for model_id, guess in (("black-forest-labs/FLUX.1-Krea-dev", True), ("krea/Krea-2-Turbo", False)):
        described = dict(fake.describe(), refine_supported=None, model_id=model_id)
        monkeypatch.setattr(fake, "describe", lambda described=described: described)
        assert client.get("/api/status").json()["defaults"]["refine_supported"] is guess


def test_a_refine_is_refused_at_submit_where_the_checkpoint_has_none(client, fake, store):
    # A night of refinements must not discover at the first cell that every
    # one of them is a plain text-to-image at the larger size - a new picture,
    # and the one thing the pass exists to avoid.
    first = _first(client)
    fake.refine_supported = False
    made = len(fake.requests)
    for route, extra in (("/api/generate", {}), ("/api/grid", {"seed": 1, "strength_axis": "0.2,0.4"})):
        response = client.post(
            route, json={"prompt": "x", "width": 384, "height": 384, "refine": first["id"], **extra}
        )
        assert response.status_code == 422 and "no refine pass" in response.json()["detail"]
    assert len(fake.requests) == made


def test_an_unknown_source_is_named(client, fake, store):
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "refine": 4242}
    )
    assert response.status_code == 422
    assert "no gallery image with id 4242" in response.json()["detail"]
    assert fake.requests == []


def test_a_refine_reaches_the_backend_and_the_row(client, fake, store):
    first = _first(client)
    _, events = run_one(
        client, refine=first["id"], strength=0.5, width=384, height=384, steps=10
    )
    assert (fake.requests[-1].refine, fake.requests[-1].strength) == (first["id"], 0.5)
    newest = client.get("/api/gallery").json()["items"][0]
    assert (newest["refine"], newest["strength"]) == (first["id"], 0.5)
    assert (newest["width"], newest["height"], newest["steps"]) == (384, 384, 10)
    # A text-to-image row says so by absence, in both.
    assert (first["refine"], first["strength"]) == (None, None)
    # The job counted the five steps the strength leaves of the ten, not ten.
    assert (events[-1]["step"], events[-1]["total_steps"]) == (5, 5)
    # And the refinements of an image are a search away.
    found = client.get("/api/gallery", params={"search": f"refine:{first['id']}"}).json()
    assert [item["id"] for item in found["items"]] == [newest["id"]]


def test_a_strength_sweep_is_an_ordinary_grid_axis(client, fake, store):
    first = _first(client)
    payload = {
        "prompt": "a fox", "width": 384, "height": 384, "steps": 10, "seed": 1,
        "refine": first["id"], "strength_axis": "0.2,0.4,0.6",
    }
    preview = client.post("/api/grid/preview", json=payload).json()
    assert preview["cells"] == 3 and preview["values"]["strength"] == [0.2, 0.4, 0.6]
    assert "3 strength" in preview["summary"]

    job_id, _ = run_grid(client, **payload)
    assert [r.strength for r in fake.requests[-3:]] == [0.2, 0.4, 0.6]
    rows = client.get("/api/gallery", params={"search": f"job:{job_id}"}).json()["items"]
    assert {row["strength"] for row in rows} == {0.2, 0.4, 0.6}
    assert {row["refine"] for row in rows} == {first["id"]}
    # One run of one seed: the strengths are what the sheet lays out.
    assert len({row["seed"] for row in rows}) == 1 and len({row["run_id"] for row in rows}) == 1

    # The key carries the source and the strength, so a resume finds all
    # three done - and one more strength is the only cell left.
    assert client.post("/api/grid", json={**payload, "resume": True}).status_code == 409
    wider = {**payload, "strength_axis": "0.2,0.4,0.6,0.8", "resume": True}
    preview = client.post("/api/grid/preview", json=wider).json()
    assert (preview["cells"], preview["skipped"]) == (1, 3)

    # With no source the strength is not a setting, and the axis is one cell.
    plain = {k: v for k, v in payload.items() if k != "refine"}
    assert client.post("/api/grid/preview", json=plain).json()["cells"] == 1


def test_the_preview_says_where_a_pass_starts(client, fake, store, monkeypatch):
    # The Refine dialog's "starts at N% noise": the backend's reading of the
    # loaded schedule, for the pass as the pipeline will run it, and only
    # where every cell starts at the same point.
    first = _first(client)
    asked = []
    monkeypatch.setattr(fake, "start_sigma", lambda request: asked.append(request) or 0.474)
    fake.cfg_threshold = 1.0  # a negative prompt below it is dropped, as FLUX.1 drops it
    pass_ = {
        "prompt": "a fox", "negative_prompt": "blurry", "guidance_scale": 1.0,
        "width": 384, "height": 384, "steps": 10, "seed": 1, "refine": first["id"], "strength": 0.3,
    }
    assert client.post("/api/grid/preview", json=pass_).json()["start_sigma"] == 0.474
    (request,) = asked
    assert (request.refine, request.strength, request.steps) == (first["id"], 0.3, 10)
    assert request.negative_prompt is None  # applied, not as typed

    # A sweep starts at several points, and a direct render at none: not asked.
    asked.clear()
    sweep = {**pass_, "strength_axis": "0.2,0.4"}
    assert client.post("/api/grid/preview", json=sweep).json()["start_sigma"] is None
    plain = {k: v for k, v in pass_.items() if k not in ("refine", "strength")}
    assert client.post("/api/grid/preview", json=plain).json()["start_sigma"] is None
    assert asked == []

    # A backend that cannot read one says nothing: the base class's answer.
    monkeypatch.delattr(fake, "start_sigma")
    assert client.post("/api/grid/preview", json=pass_).json()["start_sigma"] is None


def test_the_previews_start_sigma_is_the_loaded_schedules(client, fake, store, monkeypatch):
    # End to end with the real reading: a FLUX.1 pipeline's schedule (its
    # scheduler, nothing else) behind the route, 1 MP to 1.5x at the
    # profile's 0.1 - where the card measured sigma 0.474.
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler

    from backend.backends.diffusers_backend import DiffusersBackend

    from test_refine_schedules import FLUX1

    first = _first(client)
    real = DiffusersBackend()
    pipe = diffusers.FluxPipeline.__new__(diffusers.FluxPipeline)
    vars(pipe).update(scheduler=FlowMatchEulerDiscreteScheduler.from_config(FLUX1), vae_scale_factor=8)
    real._pipe = pipe
    monkeypatch.setattr(fake, "start_sigma", real.start_sigma)
    body = {"prompt": "a fox", "width": 1536, "height": 1536, "steps": 28, "refine": first["id"], "strength": 0.1}
    assert client.post("/api/grid/preview", json=body).json()["start_sigma"] == pytest.approx(0.474, abs=0.001)
    # Mu 1.15 is the whole shift: at 0.2 the pass starts at 0.463 instead.
    body.update(mu=1.15, strength=0.2)
    assert client.post("/api/grid/preview", json=body).json()["start_sigma"] == pytest.approx(0.463, abs=0.001)


def test_a_pass_has_a_size_guard_of_its_own(client, fake, store):
    # Past the 4 MP presets: a pass takes its composition from the source, so
    # its guard is any 2 MP preset at 2x. Checked in the preview, which runs
    # the submit's own validation without rendering an 8 MP image.
    first = _first(client)
    made = len(fake.requests)
    wide = {"prompt": "a fox", "width": 3600, "height": 256, "seed": 1, "steps": 10}
    response = client.post("/api/grid/preview", json={**wide, "refine": first["id"]})
    assert response.status_code == 200 and response.json()["cells"] == 1
    # A direct render is held to the presets' side, and told why.
    response = client.post("/api/generate", json=wide)
    assert response.status_code == 422
    assert "past the 3168 a direct render takes; only a refine pass goes to 4448" in response.text
    # The refine budget is pixels too: 4096x4096 fits the side, not the area.
    big = {**wide, "width": 4096, "height": 4096, "refine": first["id"]}
    response = client.post("/api/generate", json=big)
    assert response.status_code == 422
    assert "8,411,136 pixel budget of a refine pass" in response.json()["detail"]
    # And no batch budget: a pass is one call per image, so four 4 MP passes
    # hold one image's pixels at a time where four direct renders hold all.
    batch = {**wide, "width": 2048, "height": 2048, "num_images": 4}
    response = client.post("/api/grid/preview", json={**batch, "refine": first["id"]})
    assert response.status_code == 200 and response.json()["images"] == 4
    response = client.post("/api/grid/preview", json=batch)
    assert response.status_code == 422 and "in one batch exceeds" in response.json()["detail"]
    assert len(fake.requests) == made


def test_a_sweep_that_would_run_no_step_is_refused_by_name(client, fake, store):
    first = _first(client)
    made = len(fake.requests)
    grid = {
        "prompt": "a fox", "width": 384, "height": 384, "steps": 4, "seed": 1,
        "refine": first["id"], "strength": 0.5, "strength_axis": "0.1,0.5",
    }
    # One cell of the sweep: every cell is a request of its own, so the grid
    # is refused whole when it is built, naming the value.
    response = client.post("/api/grid", json=grid)
    assert response.status_code == 422
    assert "strength 0.1 of 4 steps is less than one step" in response.json()["detail"]
    # And the panel's own strength, checked after the capability refusal.
    response = client.post("/api/generate", json={**grid, "strength": 0.2})
    assert response.status_code == 422
    assert "strength 0.2 of 4 steps is less than one step" in response.json()["detail"]
    assert len(fake.requests) == made


def test_a_strength_past_1_is_refused_as_a_fraction(client, fake, store):
    # The Refine dialog shows the preview's refusal as it comes, so the words
    # say what a strength is, not pydantic's bound - for the box's single
    # value and for a value of a sweep alike.
    first = _first(client)
    base = {"prompt": "a fox", "width": 384, "height": 384, "seed": 1, "refine": first["id"]}
    for body in ({**base, "strength": 1.5}, {**base, "strength_axis": "0.5,1.5"}):
        response = client.post("/api/grid/preview", json=body)
        assert response.status_code == 422
        detail = response.json()["detail"]
        text = detail if isinstance(detail, str) else " ".join(e["msg"] for e in detail)
        assert "1.5 is not a fraction between 0 and 1" in text
        assert "Value error" not in text and "less than or equal" not in text


def test_one_images_record_by_id(client, fake, store):
    # What the panel's source chip asks for: the size the scale buttons
    # multiply, for an id that may be on no page the browser has loaded.
    first = _first(client)
    assert client.get(f"/api/images/{first['id']}").json() == first
    assert client.get("/api/images/4242").status_code == 404


def test_a_rerender_carries_the_source_and_the_strength(client, fake, store):
    first = _first(client)
    run_one(client, refine=first["id"], strength=0.5, width=384, height=384, steps=10)
    selection = {"search": f"refine:{first['id']}", "favorites": False, "width": 512, "height": 512}
    response = client.post("/api/rerender", json=selection)
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    # The same pass at the new size: the source and the strength are the
    # record's reproduction set as much as its seed.
    assert (ran.refine, ran.strength, ran.width, ran.steps) == (first["id"], 0.5, 512, 10)

    # A source that has been deleted since cannot be refined again, and the
    # selection is refused naming it rather than rendered as something else.
    assert client.delete(f"/api/images/{first['id']}").status_code == 200
    selection["width"] = selection["height"] = 640
    response = client.post("/api/rerender", json=selection)
    assert response.status_code == 422
    assert f"no gallery image with id {first['id']}" in response.json()["detail"]


def test_a_refine_with_no_strength_runs_at_the_checkpoints_own(client, fake, store):
    # Where a strength lands depends on the checkpoint's schedule, so the
    # starting value is the profile's, published like the steps and filled in
    # for a request that names none.
    from backend.schemas import default_strength

    first = _first(client)
    assert client.get("/api/status").json()["defaults"]["refine_strength"] == default_strength()
    run_one(client, refine=first["id"], width=384, height=384, steps=20)
    assert fake.requests[-1].strength == default_strength()
    newest = client.get("/api/gallery").json()["items"][0]
    assert (newest["refine"], newest["strength"]) == (first["id"], default_strength())


@pytest.mark.parametrize("base,axes", [
    ({"steps": 8}, {"strength_axis": "0.3,0.5"}),
    ({"steps": 8, "strength": 0.1}, {"strength_axis": "0.3,0.5"}),
    ({"steps": 4, "strength": 0.1}, {"steps_axis": "20,28"}),
])
def test_axes_replace_an_unused_invalid_base_pair(client, fake, base, axes):
    first = _first(client)
    payload = {"prompt": "a fox", "width": 256, "height": 256,
               "refine": first["id"], "seed": 1, **base, **axes}
    preview = client.post("/api/grid/preview", json=payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["cells"] == 2
    _, events = run_grid(client, **payload)
    assert events[-1]["status"] == "done"
    for request in fake.requests[-2:]:
        request.check_refine_steps()


@pytest.mark.parametrize("route", ["/api/generate", "/api/grid", "/api/grid/preview"])
def test_unsupported_refine_precedes_invalid_step_pair(client, fake, route):
    fake.refine_supported = False
    response = client.post(route, json={"prompt": "x", "refine": 4242,
                                       "steps": 4, "strength": 0.1})
    assert response.status_code == 422
    assert "has no refine pass" in response.json()["detail"]
    assert fake.requests == []


def test_live_unprofiled_refine_gets_the_fallback_strength(client, fake, monkeypatch):
    described = dict(fake.describe(), model_id="someone/unprofiled-sdxl", refine_supported=True)
    monkeypatch.setattr(fake, "describe", lambda: described)
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["refine_supported"] is True
    assert defaults["refine_strength"] == 0.4


@pytest.mark.parametrize("route,extra", [
    ("/api/generate", {}),
    ("/api/grid", {"strength_axis": "0.2,0.4", "seed": 1}),
])
def test_load_disproving_refine_guess_fails_before_any_render(client, fake, monkeypatch, route, extra):
    first = _first(client)
    made = len(fake.requests)
    fake.loaded = False
    describe = fake.describe
    load = fake.load

    def live_description():
        return dict(describe(), model_id="black-forest-labs/FLUX.1-Kontext-dev",
                    refine_supported=fake.refine_supported if fake.loaded else None)

    def unsupported_load():
        load()
        fake.refine_supported = False

    monkeypatch.setattr(fake, "describe", live_description)
    monkeypatch.setattr(fake, "load", unsupported_load)
    response = client.post(route, json={"prompt": "x", "refine": first["id"], **extra})
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["status"] == "error"
    assert "has no refine pass" in done["error"]
    assert done["images"] == []
    assert len(fake.requests) == made


def test_refine_progress_totals_exist_before_the_first_callback(client, fake, monkeypatch):
    from backend import main

    first = _first(client)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    original_generate = fake.generate
    original_run = main.jobs._run
    queued_totals = []
    calls = 0

    async def observe_queued(job):
        queued_totals.append(job.total_steps)
        await original_run(job)

    def held_generate(*args, **kwargs):
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        assert release[index].wait(10), "test did not release the callback gate"
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(main.jobs, "_run", observe_queued)
    monkeypatch.setattr(fake, "generate", held_generate)
    response = client.post("/api/grid", json={
        "prompt": "x", "width": 256, "height": 256, "refine": first["id"],
        "steps": 28, "strength_axis": "0.1,0.4", "num_images": 2, "seed": 1,
    })
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    try:
        for index, total in enumerate((6, 24)):
            assert entered[index].wait(10)
            snapshot = client.get(f"/api/jobs/{job_id}").json()
            assert (snapshot["run"], snapshot["step"], snapshot["total_steps"]) == (index + 1, 0, total)
            release[index].set()
        assert queued_totals == [6]
    finally:
        for event in release:
            event.set()
        stream_job(client, job_id)


def test_preview_prices_refine_from_matching_full_durations(client, fake, store):
    from PIL import Image

    from backend.schemas import GenerateRequest

    first = _first(client)
    payload = {"prompt": "x", "width": 384, "height": 384, "steps": 28, "seed": 1}
    for extra, duration in (({}, 101304), ({"refine": first["id"], "strength": 0.1}, 16900)):
        store.save_image(
            Image.new("RGB", (384, 384)), GenerateRequest(**payload, **extra),
            seed=1, duration_ms=duration, model_id=store.settings.model_id,
        )
    direct = client.post("/api/grid/preview", json=payload).json()
    refined = client.post("/api/grid/preview", json={
        **payload, "refine": first["id"], "strength": 0.1,
    }).json()
    unmeasured = client.post("/api/grid/preview", json={
        **payload, "refine": first["id"], "strength": 0.4,
    }).json()
    assert direct["seconds"] == 101.3
    assert refined["seconds"] == 16.9
    assert unmeasured["seconds"] is None
