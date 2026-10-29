"""Re-render at a tier over HTTP: each picked record at its own aspect's
entry at one megapixel tier, one job per size, the preview that shows the
plan and the `expected` ids that hold the queue to it."""
from __future__ import annotations

import dataclasses

from PIL import Image

from backend import grid, main, schemas_generate, sizes
from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import stream_job


def plant(store, prompt, *, seed, width=256, height=256):
    return store.save_image(
        Image.new("RGB", (width, height)),
        GenerateRequest(prompt=prompt, width=width, height=height, steps=6, guidance_scale=1.5),
        seed=seed, duration_ms=1, model_id=settings.model_id,
    )


def mixed(store):
    """A square, a 3:2 landscape, another square and one between aspects."""
    square = plant(store, "square", seed=1)
    wide = plant(store, "wide", seed=2, width=384, height=256)
    again = plant(store, "again", seed=3, width=512, height=512)
    odd = plant(store, "odd", seed=4, width=1008, height=704)
    return square, wide, again, odd


def test_two_aspects_give_two_jobs_and_the_unmatched_one_is_named_not_queued(client, fake, store):
    square, wide, again, odd = mixed(store)
    ids = [odd.id, wide.id, square.id, again.id]
    preview = client.post("/api/rerender/preview", json={"ids": ids, "tier": 1})
    assert preview.status_code == 200, preview.text
    plan = preview.json()
    assert plan["ids"] == sorted(ids)
    one, wide_size = sizes.SIZES["1:1"][1], sizes.SIZES["3:2"][1]
    assert [(s["width"], s["height"], s["images"], s["queued"]) for s in plan["sizes"]] == [
        (*one, 2, 2), (*wide_size, 1, 1),
    ]
    assert plan["unmatched"] == [{"id": odd.id, "width": 1008, "height": 704}]
    assert (plan["queued"], plan["jobs"], plan["pruned"], plan["planned"]) == (3, 2, 0, 0)
    # The preview queues nothing.
    assert fake.requests == [] and main.jobs.active() == []

    accepted = client.post("/api/rerender", json={"ids": ids, "tier": 1, "expected": plan["ids"]})
    assert accepted.status_code == 200, accepted.text
    answer = accepted.json()
    assert len(answer["more"]) == 1 and answer["unmatched"] == plan["unmatched"]
    first = stream_job(client, answer["job_id"])[-1]
    second = stream_job(client, answer["more"][0]["job_id"])[-1]
    assert first["status"] == second["status"] == "done"
    assert [(r.prompt, r.seed, r.width, r.height) for r in fake.requests] == [
        ("square", 1, *one), ("again", 3, *one), ("wide", 2, *wide_size),
    ]
    # Each job a run of its own size, the shape every run id has.
    assert answer["run_id"] != answer["more"][0]["run_id"]
    assert {image["run_id"] for image in first["images"]} == {answer["run_id"]}
    # Done is done: again, the gallery holds them all at their tier's sizes.
    repeat = client.post("/api/rerender", json={"ids": ids, "tier": 1})
    assert repeat.status_code == 409
    assert repeat.json()["detail"] == "every selected image is already rendered at its 1 MP size"
    again_plan = client.post("/api/rerender/preview", json={"ids": ids, "tier": 1}).json()
    assert (again_plan["queued"], again_plan["jobs"], again_plan["pruned"]) == (0, 0, 3)


def test_all_unmatched_is_a_422_naming_each(client, fake, store):
    odd = plant(store, "odd", seed=4, width=1008, height=704)
    for path in ("/api/rerender/preview", "/api/rerender"):
        response = client.post(path, json={"ids": [odd.id], "tier": 2})
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == (
            f"none of the picked images is a preset's aspect, so none has a 2 MP size: #{odd.id} is 1008x704"
        )
    assert fake.requests == []


def test_one_sizes_budget_refuses_the_press_by_that_size(client, fake, store, monkeypatch):
    square, wide, _, _ = mixed(store)
    # Between 1:1's 1 MP entry (1,048,576) and 3:2's (1,071,872).
    monkeypatch.setattr(schemas_generate, "settings", dataclasses.replace(settings, max_pixels=1_060_000))
    for path in ("/api/rerender/preview", "/api/rerender"):
        response = client.post(path, json={"ids": [square.id, wide.id], "tier": 1})
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == "1264x848 exceeds the 1,060,000 pixel budget"
    assert fake.requests == [] and main.jobs.active() == []
    # The square alone fits.
    assert client.post("/api/rerender", json={"ids": [square.id], "tier": 1}).status_code == 200


def test_a_selection_changed_since_the_preview_is_refused(client, fake, store):
    square, wide, again, _ = mixed(store)
    plan = client.post("/api/rerender/preview", json={"ids": [square.id, wide.id, again.id], "tier": 1}).json()
    assert client.post("/api/images/bin", json={"ids": [wide.id]}).status_code == 200
    response = client.post(
        "/api/rerender", json={"ids": [square.id, wide.id, again.id], "tier": 1, "expected": plan["ids"]},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "the selection changed since the check; check Re-render again"
    assert fake.requests == [] and main.jobs.active() == []
    # The fixed size takes `expected` too.
    response = client.post(
        "/api/rerender", json={"ids": [square.id], "width": 512, "height": 512, "expected": [square.id, 999]},
    )
    assert response.status_code == 409


def test_the_request_takes_a_tier_or_a_size_and_a_tier_only_over_picks(client, store):
    square = plant(store, "square", seed=1)
    for body in (
        {"ids": [square.id]},
        {"ids": [square.id], "width": 512},
        {"ids": [square.id], "tier": 1, "width": 512, "height": 512},
        {"ids": [square.id], "tier": 3},
        {"tier": 1, "favorites": False},
    ):
        assert client.post("/api/rerender", json=body).status_code == 422, body
        assert client.post("/api/rerender/preview", json=body).status_code == 422, body
    # The size is still held to the 16 grid.
    response = client.post("/api/rerender", json={"ids": [square.id], "width": 1000, "height": 512})
    assert response.status_code == 422 and "1000 is not a multiple of 16: 992 or 1008" in response.text


def test_an_inpaint_is_refused_at_a_tier_that_is_not_its_own_size(client, fake, store):
    from tests.test_api_inpaint import _inpainted

    _, made, _ = _inpainted(client, fake)
    response = client.post("/api/rerender", json={"ids": [made["id"]], "tier": 1})
    assert response.status_code == 422
    assert f"#{made['id']} is an inpaint of" in response.json()["detail"]
    assert "cannot be re-rendered at 1024x1024" in response.json()["detail"]


def test_without_a_tier_the_run_id_is_the_one_size_re_renders_as_before(client, fake, store):
    square, wide, _, _ = mixed(store)
    records = [store.get_image(square.id), store.get_image(wide.id)]
    before = grid.run_id(
        grid.cells_from_records(records, width=512, height=512),
        main.services.model_id(), main.services.quantization(),
    )
    preview = client.post("/api/rerender/preview", json={"ids": [square.id, wide.id], "width": 512, "height": 512})
    assert preview.status_code == 200, preview.text
    assert [(s["width"], s["height"], s["images"]) for s in preview.json()["sizes"]] == [(512, 512, 2)]
    accepted = client.post("/api/rerender", json={"ids": [square.id, wide.id], "width": 512, "height": 512})
    assert accepted.status_code == 200, accepted.text
    answer = accepted.json()
    assert answer["run_id"] == before and answer["more"] == [] and answer["unmatched"] == []
    stream_job(client, answer["job_id"])
