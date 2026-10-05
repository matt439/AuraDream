"""The gallery's own routes: the checkpoint scope, and Delete unstarred, the
inverse of the per-card delete."""
from __future__ import annotations

import threading

from PIL import Image

from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid, stream_job


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
    # Nothing is running, so nothing says a job may move the count again.
    assert "still running" not in response.json()["detail"]
    # `expected` is not optional: the gate is the contract.
    assert client.delete("/api/gallery/unstarred").status_code == 422


def test_delete_unstarred_says_when_any_job_is_running(client, fake, store, monkeypatch):
    # The dialog's "a job is still running" line and the 409's hint were the
    # asking page's own `busy`, so a job another tab started - the one about to
    # land images in scope - went unmentioned. The server says, for any job.
    _plant(store, settings.model_id, seed=1)
    _plant(store, settings.model_id, seed=2)
    assert client.get("/api/gallery/unstarred").json()["job_active"] is False

    running, release = threading.Event(), threading.Event()
    generate = fake.generate

    def held(*args, **kwargs):
        running.set()
        assert release.wait(10)
        return generate(*args, **kwargs)

    monkeypatch.setattr(fake, "generate", held)
    job_id = client.post(
        "/api/generate", json={"prompt": "a lighthouse in fog", "width": 256, "height": 256}
    ).json()["job_id"]
    try:
        assert running.wait(5)
        preview = client.get("/api/gallery/unstarred").json()
        assert (preview["unstarred"], preview["job_active"]) == (2, True)
        # A count that no longer holds, with the job still going: the refusal
        # says the count can move again.
        response = client.delete("/api/gallery/unstarred?expected=1")
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "2 unstarred images match now, not the 1 confirmed" in detail
        assert detail.endswith(
            " A job is still running and may add more images here: press Delete "
            "unstarred again, or let it finish first."
        )
    finally:
        release.set()
    assert stream_job(client, job_id)[-1]["status"] == "done"
    preview = client.get("/api/gallery/unstarred").json()
    assert (preview["unstarred"], preview["job_active"]) == (3, False)


def _made_from(store, seed, **conditioning):
    """A refine pass or a referenced render of gallery images, on this checkpoint."""
    return store.save_image(
        Image.new("RGB", (16, 16)),
        GenerateRequest(prompt="a lighthouse in fog", width=256, height=256, **conditioning),
        seed=seed, duration_ms=1, model_id=settings.model_id, run_id="later", job_id="later",
    )


def test_delete_unstarred_keeps_what_a_starred_image_was_made_from(client, store):
    # Refine a source, star the pass, sweep the rest: the source used to go
    # with the sweep, and the kept pass could never be made again. A star
    # keeps what its image was made from - the pass's source, the render's
    # references - and the count the dialog asks for leaves them out, so the
    # number typed back is still the number the delete finds.
    source = _plant(store, settings.model_id, seed=1, run_id="night")
    reference = _plant(store, settings.model_id, seed=2, run_id="night")
    loose = _plant(store, settings.model_id, seed=3, run_id="night")
    refined = _made_from(store, seed=4, refine=source.id, strength=0.5)
    rendered = _made_from(store, seed=5, reference=[reference.id])
    _star(client, [{"id": refined.id}, {"id": rendered.id}])

    preview = client.get("/api/gallery/unstarred").json()
    assert (preview["records"], preview["favorites"], preview["sources_kept"]) == (5, 2, 2)
    assert (preview["unstarred"], preview["sources"]) == (1, [source.id, reference.id])
    # What goes is one run's, made at one moment.
    assert (preview["runs"], preview["first"], preview["last"]) == (
        1, loose.created_at, loose.created_at
    )
    # The count without the sources is no longer the delete's: the pin holds.
    assert client.delete("/api/gallery/unstarred?expected=3").status_code == 409
    response = client.delete("/api/gallery/unstarred?expected=1")
    assert response.status_code == 200, response.text
    assert response.json()["deleted"] == [loose.id]
    assert (store.settings.outputs_dir / source.filename).exists()
    left = client.get("/api/gallery").json()["items"]
    assert sorted(r["id"] for r in left) == sorted([source.id, reference.id, refined.id, rendered.id])

    # Unstarred, the pass no longer holds its source: both are swept together.
    client.post(f"/api/images/{refined.id}/favorite?value=false")
    preview = client.get("/api/gallery/unstarred").json()
    assert (preview["unstarred"], preview["sources"]) == (2, [reference.id])
    assert client.delete("/api/gallery/unstarred?expected=2").status_code == 200
    assert sorted(r["id"] for r in client.get("/api/gallery").json()["items"]) == sorted(
        [reference.id, rendered.id]
    )


def test_delete_unstarred_counts_what_is_newer_than_the_page(client, store):
    # The dialog says how many of the count were made since the page last
    # loaded. The page counted them itself, leaving out only the few kept
    # sources the preview names - the oldest six - so a new image a starred
    # pass was made from was counted as one the sweep takes. The server
    # counts them by the delete's own predicate, in the same scope.
    for seed in range(1, 7):
        source = _plant(store, settings.model_id, seed=seed, run_id="old")
        _star(client, [{"id": _made_from(store, seed=100 + seed, refine=source.id, strength=0.5).id}])
    head = _plant(store, settings.model_id, seed=7, run_id="old").id
    loose = _plant(store, settings.model_id, seed=8, run_id="new")
    fresh_source = _plant(store, settings.model_id, seed=9, run_id="new")
    _star(client, [{"id": _made_from(store, seed=10, refine=fresh_source.id, strength=0.5).id}])
    _plant(store, "somebody/else", seed=11, run_id="new")

    preview = client.get(f"/api/gallery/unstarred?newer_than={head}").json()
    assert (preview["unstarred"], preview["sources_kept"]) == (2, 7)
    assert fresh_source.id not in preview["sources"]
    assert preview["newer"] == 1
    # Asked under the page's own scope: All models, a search.
    assert client.get(
        f"/api/gallery/unstarred?newer_than={head}&all_models=true"
    ).json()["newer"] == 2
    assert client.get(
        f"/api/gallery/unstarred?newer_than={head}&search=run:old"
    ).json()["newer"] == 0
    assert client.get(f"/api/gallery/unstarred?newer_than={loose.id}").json()["newer"] == 0
    # Not asked, not answered: a sheet is one run laid out whole.
    assert client.get("/api/gallery/unstarred").json()["newer"] is None
    assert client.get("/api/gallery/unstarred?newer_than=-1").status_code == 422


def test_a_pass_whose_source_has_gone_says_so(client, store):
    # The pass keeps its source's id - it is the reproduction set's - and the
    # record says the source is gone, on the page and by id alike.
    source = _plant(store, settings.model_id, seed=1)
    refined = _made_from(store, seed=2, refine=source.id, strength=0.5)
    plain = _plant(store, settings.model_id, seed=3)
    items = {r["id"]: r for r in client.get("/api/gallery").json()["items"]}
    assert not any(r["refine_missing"] for r in items.values())

    assert client.delete(f"/api/images/{source.id}").status_code == 200
    items = {r["id"]: r for r in client.get("/api/gallery").json()["items"]}
    assert (items[refined.id]["refine"], items[refined.id]["refine_missing"]) == (source.id, True)
    assert items[plain.id]["refine_missing"] is False
    record = client.get(f"/api/images/{refined.id}").json()
    assert record["refine_missing"] is True


def test_delete_unstarred_refuses_free_text(client, fake):
    # The gallery matched "lighthouse" through FTS and its LIKE fallback; a
    # destructive scope is not resolved through either.
    run_grid(client, steps_axis="6,8", seed=1)
    for method in ("get", "delete"):
        response = getattr(client, method)("/api/gallery/unstarred?search=lighthouse&expected=2")
        assert response.status_code == 422
        assert "free text" in response.json()["detail"]
    assert client.get("/api/gallery").json()["total"] == 2
