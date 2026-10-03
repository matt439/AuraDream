"""The refine pass over HTTP: published in the status, refused at submit where
the checkpoint has none or the source is gone, an ordinary grid axis, and
recorded on the row at the strength it ran at."""
from __future__ import annotations

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
    assert "no image-to-image counterpart" in body["hints"]["refine"]


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
    # And the panel's own strength, which the schema refuses before any route.
    response = client.post("/api/generate", json={**grid, "strength": 0.2})
    assert response.status_code == 422
    assert "strength 0.2 of 4 steps is less than one step" in response.json()["detail"][0]["msg"]
    assert len(fake.requests) == made


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
