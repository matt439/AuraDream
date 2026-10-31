"""Re-render over HTTP: a selection rendered again at another size, its
cells' variables kept, and another checkpoint's records refused in words."""
from __future__ import annotations

import dataclasses
import json

from PIL import Image

from backend import main
from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import LABELS, TEXTS, VARIANTS, run_grid, stream_job


def test_a_selection_can_be_rendered_again_at_another_size(client, fake, store):
    # Star two cells of a 256px grid and ask for them at 512: two cells, the
    # same prompt, steps, CFG, mu and seed, only the size changed, oldest
    # first - and a second ask finds them done.
    _, events = run_grid(client, steps_axis="6,8,10", cfg_axis="0,1", seed=5, mu=1.15)
    images = sorted(events[-1]["images"], key=lambda r: r["id"])
    for record in (images[1], images[4]):
        client.post(f"/api/images/{record['id']}/favorite?value=true")
    fake.requests.clear()

    accepted = client.post("/api/rerender", json={"width": 512, "height": 512}).json()
    done = stream_job(client, accepted["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 2
    asked = [(r.steps, r.guidance_scale, r.mu, r.seed, r.width) for r in fake.requests]
    assert asked == [
        (images[1]["steps"], images[1]["guidance_scale"], 1.15, 5, 512),
        (images[4]["steps"], images[4]["guidance_scale"], 1.15, 5, 512),
    ]
    assert accepted["seed"] == 5
    # A run of its own, distinct from the grid it came from, and resumable.
    assert accepted["run_id"] != images[0]["run_id"]
    assert client.post("/api/rerender", json={"width": 512, "height": 512}).status_code == 409
    # The selection is the gallery's own filter, taken server-side: the two
    # 8-step cells, starred or not.
    narrowed = client.post(
        "/api/rerender", json={"width": 768, "height": 768, "favorites": False, "search": "steps:8"}
    )
    assert narrowed.status_code == 200
    assert len(stream_job(client, narrowed.json()["job_id"])[-1]["images"]) == 2
    assert client.post(
        "/api/rerender", json={"width": 512, "height": 512, "search": "seed:999"}
    ).status_code == 404
    # The pixel budget applies, as it does to both other producers.
    assert client.post("/api/rerender", json={"width": 2736, "height": 2736}).status_code == 422


def test_a_rerender_keeps_each_cells_variables(client, fake, store):
    # A comparison's version names live in its variables alone, and a `{var}`
    # grid's assignment is what its sheet lays out; a re-render that dropped
    # them left the new run's sheet a flat Prompt axis in alphabetical order.
    run_grid(client, prompt_variants=VARIANTS, seed=3)
    run_grid(client, prompt="a {size} dog", prompt_axis="size = big | none", seed=4)
    accepted = client.post("/api/rerender", json={"width": 512, "height": 512, "favorites": False})
    assert accepted.status_code == 200, accepted.text
    images = stream_job(client, accepted.json()["job_id"])[-1]["images"]
    assert [(r["prompt"], r["variables"], r["width"]) for r in images] == [
        *((text, {"variant": label}, 512) for text, label in zip(TEXTS, LABELS)),
        ("a big dog", {"size": "big"}, 512),
        # `none` is an assignment too, and survives as one.
        ("a dog", {"size": ""}, 512),
    ]
    # The PNG's own record carries it as well, for a reindex to bring back.
    with Image.open(store.settings.outputs_dir / images[0]["filename"]) as png:
        assert json.loads(png.info["auradream"])["variables"] == {"variant": LABELS[0]}


def test_a_rerender_refuses_another_checkpoints_records(client, fake, store):
    # Steps, CFG and mu are checkpoint-specific; rendering FLUX's 28 / 3.5 on
    # Turbo would record a broken image under Turbo. The gallery's scope keeps
    # the other checkpoint's row out of the selection, so the refusal is only
    # reachable once All models has let it in.
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256),
        seed=1, duration_ms=1, model_id="somebody/else",
    )
    selection = {"width": 512, "height": 512, "favorites": False}
    assert client.post("/api/rerender", json=selection).status_code == 404
    response = client.post("/api/rerender", json={**selection, "all_models": True})
    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"1 of the selected images was made on somebody/else; the server is running "
        f"{settings.model_id}, whose steps, CFG and mu differ. Untick All models, or add "
        f"model:{settings.model_id} to the search."
    )
    # With one of this checkpoint's own beside it, the scoped selection is
    # that one alone, and the other row never reaches the pipeline.
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="mine", width=256, height=256),
        seed=2, duration_ms=1, model_id=settings.model_id,
    )
    accepted = client.post("/api/rerender", json=selection)
    assert accepted.status_code == 200, accepted.text
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert [r.prompt for r in fake.requests] == ["mine"] and len(done["images"]) == 1


def test_a_rerender_refusal_offers_only_a_search_that_leaves_the_others_out(
    client, fake, store, monkeypatch
):
    # Anima's refusal said "add model:Anima to the search", and `model:` is a
    # substring: Animagine stayed in, and following the advice gave the same
    # refusal. The whole id is offered, and only where it leaves out every
    # other checkpoint selected - else unticking All models alone, the exact
    # scope.
    monkeypatch.setattr(
        main, "settings", dataclasses.replace(main.settings, model_id="circlestone-labs/Anima")
    )

    def plant(model_id, seed):
        store.save_image(
            Image.new("RGB", (256, 256)),
            GenerateRequest(prompt="a smile", width=256, height=256),
            seed=seed, duration_ms=1, model_id=model_id,
        )

    plant("cagliostrolab/animagine-xl-4.0", 1)
    plant("circlestone-labs/Anima", 2)
    plant("circlestone-labs/Anima", 3)
    selection = {"width": 512, "height": 512, "favorites": False, "all_models": True}
    response = client.post("/api/rerender", json=selection)
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "1 of the selected images was made on cagliostrolab/animagine-xl-4.0; the server is "
        "running circlestone-labs/Anima, whose steps, CFG and mu differ. Untick All models, "
        "or add model:circlestone-labs/Anima to the search."
    )
    # Following the advice works.
    narrowed = {**selection, "search": "model:circlestone-labs/Anima"}
    followed = client.post("/api/rerender", json=narrowed)
    assert followed.status_code == 200, followed.text
    assert len(stream_job(client, followed.json()["job_id"])[-1]["images"]) == 2

    # An id that holds the running one whole: no `model:` would leave it out.
    plant("circlestone-labs/Anima-Preview", 4)
    plant("circlestone-labs/Anima-Preview", 5)
    response = client.post("/api/rerender", json=narrowed)
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "2 of the selected images were made on circlestone-labs/Anima-Preview; the server is "
        "running circlestone-labs/Anima, whose steps, CFG and mu differ. Untick All models."
    )


def test_a_rerender_takes_the_picked_cards_whatever_the_filter(client, fake, store):
    # Two keepers picked to render at full size: the filter (Starred, which
    # matches none of them) used to be the only scope, so it was every match
    # or nothing. With `ids`, exactly the picks, oldest first however they
    # were picked; one binned and one deleted since are left out and named,
    # as Swap tags names them, not a refusal of the rest.
    _, events = run_grid(client, steps_axis="6,8,10", cfg_axis="0,1", seed=5)
    images = sorted(events[-1]["images"], key=lambda r: r["id"])
    keep, also, binned, deleted = (images[i]["id"] for i in (4, 1, 2, 3))
    assert client.post("/api/images/bin", json={"ids": [binned]}).status_code == 200
    assert client.delete(f"/api/images/{deleted}").status_code == 200
    fake.requests.clear()

    picked = {"ids": [keep, binned, also, deleted], "width": 512, "height": 512}
    accepted = client.post("/api/rerender", json=picked)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["deleted"] == [
        {"id": binned, "detail": "in the recycle bin"},
        {"id": deleted, "detail": "no longer in the gallery"},
    ]
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 2
    assert [(r.steps, r.guidance_scale, r.seed, r.width) for r in fake.requests] == [
        (images[1]["steps"], images[1]["guidance_scale"], 5, 512),
        (images[4]["steps"], images[4]["guidance_scale"], 5, 512),
    ]
    # Done is done, as for a filter.
    assert client.post("/api/rerender", json=picked).status_code == 409
    # With nothing left of the picks, nothing to queue, each said.
    response = client.post("/api/rerender", json={**picked, "ids": [binned, deleted]})
    assert response.status_code == 404
    assert response.json()["detail"] == (
        f"none of the selected images is left: #{binned} is in the recycle bin; "
        f"#{deleted} is no longer in the gallery"
    )
    # A filter's answer names none.
    star = client.post(f"/api/images/{keep}/favorite?value=true")
    assert star.status_code == 200
    filtered = client.post("/api/rerender", json={"width": 768, "height": 768})
    assert filtered.status_code == 200 and filtered.json()["deleted"] == []
    # As every other action's `ids`: none, or an id below 1, is refused.
    assert client.post("/api/rerender", json={**picked, "ids": []}).status_code == 422
    assert client.post("/api/rerender", json={**picked, "ids": [0]}).status_code == 422


def test_a_rerender_of_picks_refuses_another_checkpoints_and_says_which(client, fake, store):
    # Picked cards are no filter: All models does not scope them, so another
    # checkpoint's pick is refused with or without it - and the way out is
    # unpicking it, by name, not a search or a tick that cannot reach it.
    def plant(model_id, seed):
        return store.save_image(
            Image.new("RGB", (256, 256)),
            GenerateRequest(prompt="x", width=256, height=256),
            seed=seed, duration_ms=1, model_id=model_id,
        ).id

    mine = plant(settings.model_id, 1)
    theirs = [plant("somebody/else", seed) for seed in range(2, 10)]
    response = client.post("/api/rerender", json={"ids": [mine, theirs[0]], "width": 512, "height": 512})
    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"1 of the selected images was made on somebody/else; the server is running "
        f"{settings.model_id}, whose steps, CFG and mu differ. Unpick #{theirs[0]}."
    )
    # Named up to six, then counted.
    response = client.post("/api/rerender", json={"ids": [mine, *theirs], "width": 512, "height": 512})
    names = ", ".join(f"#{image_id}" for image_id in theirs[:6])
    assert response.json()["detail"].endswith(f"differ. Unpick {names} and 2 more.")
    assert not fake.requests
