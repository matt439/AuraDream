"""The gallery's own routes: the checkpoint scope, and Delete unstarred, the
inverse of the per-card delete."""
from __future__ import annotations

from PIL import Image

from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid


def _star(client, records):
    for record in records:
        client.post(f"/api/images/{record['id']}/favorite?value=true")


def _plant(store, model_id, prompt="a lighthouse in fog", seed=1, run_id=None):
    return store.save_image(
        Image.new("RGB", (16, 16)),
        GenerateRequest(prompt=prompt, width=256, height=256),
        seed=seed, duration_ms=1, model_id=model_id, run_id=run_id, job_id=run_id,
    )


def test_the_gallery_shows_the_running_checkpoint_unless_asked_for_every_one(client, store):
    # A gallery that has seen three checkpoints: by default the page is the
    # running one's images, and the reply says how many the filter would have
    # matched elsewhere - so a run made on the other model is "12 on other
    # models", not an empty grid.
    mine = [_plant(store, settings.model_id, seed=s, run_id="mine") for s in (1, 2, 3)]
    _plant(store, "somebody/else", prompt="a fox", seed=4, run_id="theirs")
    _plant(store, "somebody/else", seed=5, run_id="theirs")
    _plant(store, "another/one", seed=6)

    page = client.get("/api/gallery").json()
    assert sorted(r["id"] for r in page["items"]) == [r.id for r in mine]
    assert (page["total"], page["other_models"]) == (3, 3)
    # The search's other terms still count what they hide: `run:theirs` is
    # nothing here and two there; free text the same, through FTS.
    theirs = client.get("/api/gallery?search=run:theirs").json()
    assert (theirs["total"], theirs["other_models"], theirs["items"]) == (0, 2, [])
    fox = client.get("/api/gallery?search=fox").json()
    assert (fox["total"], fox["other_models"]) == (0, 1)
    # And the LIKE fallback, which swaps the search predicate, keeps the scope.
    odd = client.get('/api/gallery?search=a 6" lighthouse').json()
    assert (odd["total"], odd["other_models"]) == (0, 0)

    everything = client.get("/api/gallery?all_models=true").json()
    assert (everything["total"], everything["other_models"]) == (6, 0)
    assert client.get("/api/gallery?all_models=true&search=run:theirs").json()["total"] == 2

    # The Runs picker is the same gallery's table of contents.
    assert [r["run_id"] for r in client.get("/api/runs").json()] == ["mine"]
    assert sorted(r["run_id"] for r in client.get("/api/runs?all_models=true").json()) == [
        "mine", "theirs"
    ]


def test_delete_unstarred_stays_within_the_checkpoint_scope(client, store):
    # The dialog's count is the count of what the page shows: with All models
    # off, another checkpoint's unstarred images are neither counted nor
    # swept, and a number confirmed for the scoped view cannot reach them.
    _plant(store, settings.model_id, seed=1)
    _plant(store, settings.model_id, seed=2)
    other = _plant(store, "somebody/else", seed=3)

    preview = client.get("/api/gallery/unstarred").json()
    assert (preview["unstarred"], preview["models"]) == (2, [settings.model_id])
    assert client.get("/api/gallery/unstarred?all_models=true").json()["unstarred"] == 3
    # A count taken with every model is not the scoped selection's count.
    assert client.delete("/api/gallery/unstarred?expected=3").status_code == 409

    response = client.delete("/api/gallery/unstarred?expected=2")
    assert response.status_code == 200, response.text
    assert other.id not in response.json()["deleted"]
    assert client.get("/api/gallery?all_models=true").json()["total"] == 1

    assert client.delete("/api/gallery/unstarred?all_models=true&expected=1").status_code == 200
    assert client.get("/api/gallery?all_models=true").json()["total"] == 0


def test_delete_unstarred_keeps_the_stars_and_names_the_rest(client, fake, store):
    # Hundreds made, a few starred: the gesture is "keep these", not "not
    # this one" two hundred times. The preview carries the confirmation's
    # numbers and a sample of what goes; the delete is pinned to that count.
    _, events = run_grid(client, steps_axis="6,8,10", cfg_axis="0,1", seed=5)
    images = sorted(events[-1]["images"], key=lambda r: r["id"])
    _star(client, images[:2])

    preview = client.get("/api/gallery/unstarred").json()
    assert (preview["records"], preview["favorites"], preview["unstarred"]) == (6, 2, 4)
    assert preview["prompts"] == ["a lighthouse in fog"]

    response = client.delete("/api/gallery/unstarred?expected=4")
    assert response.status_code == 200, response.text
    assert sorted(response.json()["deleted"]) == [r["id"] for r in images[2:]]
    assert response.json()["files"] == 4
    page = client.get("/api/gallery").json()
    assert [r["id"] for r in page["items"]] == [images[1]["id"], images[0]["id"]]
    outputs = store.settings.outputs_dir
    assert (outputs / images[0]["filename"]).exists()
    assert not (outputs / images[5]["filename"]).exists()


def test_delete_unstarred_is_scoped_by_the_search_box(client, fake):
    first, _ = run_grid(client, steps_axis="6,8", seed=1)
    second, events = run_grid(client, steps_axis="6,8", seed=2)
    _star(client, events[-1]["images"][:1])

    preview = client.get(f"/api/gallery/unstarred?search=job:{second}").json()
    assert (preview["unstarred"], preview["favorites"], preview["jobs"]) == (1, 1, [second])
    client.delete(f"/api/gallery/unstarred?search=job:{second}&expected=1")
    # The other run is untouched: an untriaged night is not the one being judged.
    assert client.get(f"/api/gallery?search=job:{first}").json()["total"] == 2
    assert client.get(f"/api/gallery?search=job:{second}").json()["total"] == 1


def test_delete_unstarred_refuses_a_changed_selection(client, fake):
    # The dialog said 2; a sweep landed two more while it was open. The count
    # is checked under the write lock and nothing goes.
    run_grid(client, steps_axis="6,8", seed=1)
    run_grid(client, steps_axis="6,8", seed=2)
    response = client.delete("/api/gallery/unstarred?expected=2")
    assert response.status_code == 409
    assert "4 unstarred images match now, not the 2 confirmed" in response.json()["detail"]
    assert client.get("/api/gallery").json()["total"] == 4
    # `expected` is not optional: the gate is the contract.
    assert client.delete("/api/gallery/unstarred").status_code == 422


def test_delete_unstarred_refuses_free_text(client, fake):
    # The gallery matched "lighthouse" through FTS and its LIKE fallback; a
    # destructive scope is not resolved through either.
    run_grid(client, steps_axis="6,8", seed=1)
    for method in ("get", "delete"):
        response = getattr(client, method)("/api/gallery/unstarred?search=lighthouse&expected=2")
        assert response.status_code == 422
        assert "free text" in response.json()["detail"]
    assert client.get("/api/gallery").json()["total"] == 2
