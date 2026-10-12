"""The gallery's own routes: the checkpoint scope, and Delete unstarred, the
inverse of the per-card delete."""
from __future__ import annotations

import threading

from PIL import Image

from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid, run_one, stream_job


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


def test_one_checkpoints_images_by_its_exact_id_whichever_is_running(client, store, monkeypatch):
    # The Models picker's strip: the chosen row's images, newest first, by the
    # gallery's own exact scope - Anima's are not Anima@aesthetic-v1.1's, as a
    # `model:` substring would make them - and a page as small as the strip.
    from backend import main

    base = [_plant(store, "circlestone-labs/Anima", seed=s) for s in (1, 2, 3)]
    aesthetic = _plant(store, "circlestone-labs/Anima@aesthetic-v1.1", seed=4)
    _plant(store, settings.model_id, seed=5)

    page = client.get("/api/gallery?model=circlestone-labs/Anima&limit=2").json()
    assert [r["id"] for r in page["items"]] == [base[2].id, base[1].id]
    assert (page["total"], page["has_more"]) == (3, True)
    # In place of the running checkpoint's scope and of all_models.
    both = client.get("/api/gallery?model=circlestone-labs/Anima&all_models=true").json()
    assert both["total"] == 3
    variant = client.get("/api/gallery", params={"model": aesthetic.model_id}).json()
    assert [r["id"] for r in variant["items"]] == [aesthetic.id]
    assert client.get("/api/gallery?model=nobody/made-this").json()["total"] == 0
    assert client.get("/api/gallery?model=").status_code == 422

    # A gallery read, not model work: answered during a switch, which refuses
    # every model-dependent route, and under a stale revision header.
    monkeypatch.setattr(main, "_switching", True)
    stale = {"X-Local-Diffusion-Model-Revision": "-1"}
    during = client.get("/api/gallery?model=circlestone-labs/Anima", headers=stale)
    assert during.status_code == 200, during.text
    assert during.json()["total"] == 3
    # Bundle maintenance holds every gallery read, this one included: a 409
    # the strip says quietly, and asks again later.
    monkeypatch.setattr(main.bundle_manager, "state", {**main.bundle_manager.state, "state": "running"})
    held = client.get("/api/gallery?model=circlestone-labs/Anima")
    assert held.status_code == 409 and "bundle" in held.json()["detail"]


def test_model_equals_in_the_search_is_the_whole_id(client, store):
    # See all's words for the same scope: `model:=<id>` matches the whole id,
    # where `model:<text>` stays a substring.
    _plant(store, "circlestone-labs/Anima", seed=1)
    _plant(store, "circlestone-labs/Anima@aesthetic-v1.1", seed=2)
    _plant(store, "cagliostrolab/animagine-xl-4.0", seed=3)

    def total(search):
        return client.get("/api/gallery", params={"all_models": "true", "search": search}).json()["total"]

    assert total("model:=circlestone-labs/Anima") == 1
    assert total("model:=circlestone-labs/Anima@aesthetic-v1.1") == 1
    assert total("model:circlestone-labs/Anima") == 2
    assert total("model:=Anima") == 0
    assert total("model:anima") == 3
    # A bare `=` names nothing, and says so rather than matching every row.
    refused = client.get("/api/gallery", params={"all_models": "true", "search": "model:="})
    assert refused.status_code == 422 and "names no value" in refused.json()["detail"]
    # The strip and the search agree on every checkpoint.
    for model_id in ("circlestone-labs/Anima", "circlestone-labs/Anima@aesthetic-v1.1"):
        strip = client.get("/api/gallery", params={"model": model_id}).json()
        assert strip["total"] == total(f"model:={model_id}")


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
    # Into the recycle bin, files untouched: Restore undoes it.
    assert response.json()["binned"] == [r["id"] for r in images[2:]]
    assert response.json()["deleted"] == response.json()["binned"]
    assert response.json()["files"] == 0
    page = client.get("/api/gallery").json()
    assert [r["id"] for r in page["items"]] == [images[1]["id"], images[0]["id"]]
    binned = client.get("/api/gallery/bin").json()
    assert sorted(r["id"] for r in binned["items"]) == [r["id"] for r in images[2:]]
    outputs = store.settings.outputs_dir
    assert (outputs / images[0]["filename"]).exists()
    assert (outputs / images[5]["filename"]).exists()


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


def _ladder_terms(record, *, by_run=False):
    # What the lightbox's CFG comparison asks the gallery (web/lightbox-cfg.js,
    # ladderQuery): the seed - or the run, for a seed past 2**53 - with the
    # size, steps, refine source and mu, in the checkpoint's exact scope.
    mu = "none" if record["mu"] is None else record["mu"]
    first = f"run:{record['run_id']}" if by_run else f"seed:{record['seed']}"
    return (f"{first} size:{record['width']}x{record['height']} steps:{record['steps']} "
            f"refine:{record['refine'] if record['refine'] is not None else 'none'} mu:{mu}")


def test_the_cfg_ladder_query_finds_every_cfg_value_at_one_seed_and_settings(client, fake, store):
    # A CFG x steps grid, and a separate Generate at the same seed and
    # settings: every CFG value at the cell's steps comes back, from either
    # run, and nothing at the other steps, another size or another seed.
    _, events = run_grid(client, cfg_axis="2,3.5,5", steps_axis="6,8", seed=78)
    cells = [e["image"] for e in events if e["type"] == "image"]
    assert len(cells) == 6
    _, plain = run_one(client, guidance_scale=7, steps=8, seed=78)
    run_one(client, guidance_scale=4, steps=8, seed=78, width=512, height=512)
    run_one(client, guidance_scale=4, steps=8, seed=79)
    cell = next(r for r in cells if r["steps"] == 8 and r["guidance_scale"] == 3.5)
    params = {"model": cell["model_id"], "limit": 200, "search": _ladder_terms(cell)}
    page = client.get("/api/gallery", params=params).json()
    assert sorted(r["guidance_scale"] for r in page["items"]) == [2, 3.5, 5, 7]
    assert {r["steps"] for r in page["items"]} == {8}
    assert page["has_more"] is False
    # A page shorter than the ladder says there is more, which the note
    # reads as "may be incomplete".
    short = client.get("/api/gallery", params={**params, "limit": 2}).json()
    assert (len(short["items"]), short["has_more"]) == (2, True)
    # The run instead of the seed (the fallback for a seed JavaScript cannot
    # say exactly) finds the grid's cells, not the separate Generate.
    by_run = client.get("/api/gallery", params={**params, "search": _ladder_terms(cell, by_run=True)}).json()
    assert sorted(r["guidance_scale"] for r in by_run["items"]) == [2, 3.5, 5]
    assert [e["image"]["run_id"] for e in plain if e["type"] == "image"] != [cell["run_id"]]


def test_the_cfg_ladder_query_is_one_checkpoints_and_skips_refine_passes(client, fake, store):
    # The `model=` scope is exact: the same seed and settings under another
    # checkpoint are another picture. A refine pass of a cell is asked for
    # by its own source (`refine:<id>`), and a direct render's query
    # (`refine:none`) leaves passes out.
    _, events = run_grid(client, cfg_axis="2,5", steps_axis="8", seed=78)
    cells = [e["image"] for e in events if e["type"] == "image"]
    elsewhere = store.save_image(
        Image.new("RGB", (256, 256)), GenerateRequest(prompt="a lighthouse in fog", width=256, height=256, steps=8),
        seed=78, duration_ms=1, model_id="somebody/else",
    )
    _, passed = run_one(client, guidance_scale=2, steps=8, seed=78, refine=cells[0]["id"], strength=0.3,
                        width=256, height=256)
    cell = cells[0]
    params = {"model": cell["model_id"], "limit": 200, "search": _ladder_terms(cell)}
    ids = [r["id"] for r in client.get("/api/gallery", params=params).json()["items"]]
    assert sorted(ids) == sorted(r["id"] for r in cells)
    everyone = client.get("/api/gallery", params={"all_models": "true", "search": "seed:78 size:256x256"}).json()
    assert elsewhere.id in [r["id"] for r in everyone["items"]]
    refined = next(e["image"] for e in passed if e["type"] == "image")
    own = client.get("/api/gallery", params={**params, "search": _ladder_terms(refined)}).json()
    assert [r["id"] for r in own["items"]] == [refined["id"]]


def test_the_cfg_ladder_query_pages_past_a_full_page_of_other_prompts(client, fake, store):
    # A seed kept fixed across many prompts at one size fills the ladder's
    # first page with images of other prompts; the lightbox reads on by the
    # id cursor (`before`, fetchLadder in web/lightbox-cfg.js) and finds the
    # older neighbour on a later page, in the same scope and search.
    _, low = run_one(client, prompt="an old mill", guidance_scale=2, steps=8, seed=42)
    _, high = run_one(client, prompt="an old mill", guidance_scale=5, steps=8, seed=42)
    mill = [e["image"] for e in low + high if e["type"] == "image"]
    for n in range(5):
        run_one(client, prompt=f"filler {n}", guidance_scale=3, steps=8, seed=42)
    cell = mill[0]
    params = {"model": cell["model_id"], "limit": 3, "search": _ladder_terms(cell)}
    first = client.get("/api/gallery", params=params).json()
    assert first["has_more"] is True
    assert not {r["id"] for r in mill} & {r["id"] for r in first["items"]}
    seen = list(first["items"])
    page = first
    while page["has_more"]:
        page = client.get("/api/gallery", params={**params, "before": seen[-1]["id"]}).json()
        seen.extend(page["items"])
    assert len(seen) == 7 == len({r["id"] for r in seen})
    assert {r["id"] for r in mill} <= {r["id"] for r in seen}


# ---------------------------------------------------------------- recycle bin


def test_binning_and_restoring_answer_per_id_and_hold_the_record_back(client, store):
    image = _plant(store, settings.model_id, seed=1)
    other = _plant(store, settings.model_id, seed=2)
    response = client.post("/api/images/bin", json={"ids": [image.id, 9999]})
    assert response.status_code == 200, response.text
    assert response.json() == {"binned": [image.id], "already": [], "missing": [9999]}
    # A second press, another tab: idempotent, and says so.
    assert client.post("/api/images/bin", json={"ids": [image.id]}).json()["already"] == [image.id]
    assert [r["id"] for r in client.get("/api/gallery").json()["items"]] == [other.id]

    # By id it is gone, with the reason after the words the page reads,
    # unless asked for the bin's.
    record = client.get(f"/api/images/{image.id}")
    assert record.status_code == 404
    assert record.json()["detail"] == f"unknown image (#{image.id} is in the recycle bin)"
    assert client.get(f"/api/images/{image.id}/thumb", follow_redirects=False).status_code == 404
    shown = client.get(f"/api/images/{image.id}?binned=true").json()
    assert shown["id"] == image.id and shown["binned_at"]
    assert client.get(
        f"/api/images/{image.id}/thumb?binned=true", follow_redirects=False
    ).status_code in (302, 307)
    assert client.get("/api/images/9999?binned=true").json()["detail"] == "unknown image"
    # Not starred while it is binned.
    assert client.post(f"/api/images/{image.id}/favorite?value=true").status_code == 404

    response = client.post("/api/images/restore", json={"ids": [image.id, other.id, 9999]})
    assert response.json() == {"restored": [image.id], "not_binned": [other.id], "missing": [9999]}
    assert client.get(f"/api/images/{image.id}").json()["binned_at"] is None
    assert client.post(f"/api/images/{image.id}/favorite?value=true").status_code == 200
    # An empty or oversize list is refused before storage is asked.
    assert client.post("/api/images/bin", json={"ids": []}).status_code == 422
    too_many = list(range(1, settings.max_grid_images + 2))
    assert client.post("/api/images/restore", json={"ids": too_many}).status_code == 422


def test_the_bin_lists_every_checkpoints_binned_images_and_its_summary(client, store):
    mine = _plant(store, settings.model_id, prompt="a fox", seed=1)
    theirs = _plant(store, "somebody/else", prompt="a fox", seed=2)
    live = _plant(store, settings.model_id, prompt="a fox", seed=3)
    used = _plant(store, settings.model_id, prompt="a hare", seed=4)
    _made_from(store, seed=5, refine=used.id, strength=0.5)
    client.post("/api/images/bin", json={"ids": [mine.id, theirs.id, used.id]})

    page = client.get("/api/gallery/bin").json()
    assert [r["id"] for r in page["items"]] == [used.id, theirs.id, mine.id]
    assert (page["total"], page["other_models"], page["has_more"]) == (3, 0, False)
    assert all(r["binned_at"] for r in page["items"])
    page = client.get("/api/gallery/bin?search=fox&limit=1").json()
    assert ([r["id"] for r in page["items"]], page["total"], page["has_more"]) == ([theirs.id], 2, True)
    page = client.get(f"/api/gallery/bin?search=fox&limit=1&before={theirs.id}").json()
    assert [r["id"] for r in page["items"]] == [mine.id]
    assert client.get("/api/gallery/bin?search=seed:many").status_code == 422
    assert live.id not in {r["id"] for r in client.get("/api/gallery/bin").json()["items"]}

    summary = client.get("/api/gallery/bin/summary").json()
    assert summary["records"] == 3 and summary["bytes"] > 0
    assert (summary["sources_in_use"], summary["job_active"]) == ([used.id], False)


def test_emptying_the_bin_is_pinned_to_its_count(client, store):
    outputs = store.settings.outputs_dir
    image = _plant(store, settings.model_id, seed=1)
    live = _plant(store, settings.model_id, seed=2)
    client.post("/api/images/bin", json={"ids": [image.id]})
    assert client.delete("/api/gallery/bin").status_code == 422
    response = client.delete("/api/gallery/bin?expected=2")
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "The recycle bin changed: 1 images are in it now, not the 2 confirmed. "
        "Nothing was deleted; look again."
    )
    assert (outputs / image.filename).exists()

    response = client.delete("/api/gallery/bin?expected=1")
    assert response.json() == {"deleted": [image.id], "files": 1}
    assert not (outputs / image.filename).exists() and (outputs / live.filename).exists()
    assert client.get(f"/api/images/{image.id}?binned=true").status_code == 404
    assert client.get("/api/gallery/bin/summary").json()["records"] == 0
    assert client.delete("/api/gallery/bin?expected=0").json() == {"deleted": [], "files": 0}


def test_emptying_the_bin_is_pinned_to_its_ids_not_only_their_count(client, store):
    # Another tab restores one and bins a different one while the dialog is
    # open: the count is unchanged, and the image binned after the summary
    # was read must not be deleted for good on a confirmation that never
    # covered it.
    outputs = store.settings.outputs_dir
    first, second, later = (_plant(store, settings.model_id, seed=s) for s in (1, 2, 3))
    client.post("/api/images/bin", json={"ids": [first.id, second.id]})
    summary = client.get("/api/gallery/bin/summary").json()
    assert summary["records"] == 2 and len(summary["pin"]) == 16
    client.post("/api/images/restore", json={"ids": [first.id]})
    client.post("/api/images/bin", json={"ids": [later.id]})

    response = client.delete(f"/api/gallery/bin?expected=2&pin={summary['pin']}")
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "The recycle bin changed: other images are in it now than the 2 confirmed. "
        "Nothing was deleted; look again."
    )
    assert (outputs / later.filename).exists() and (outputs / second.filename).exists()

    again = client.get("/api/gallery/bin/summary").json()
    assert again["pin"] != summary["pin"]
    response = client.delete(f"/api/gallery/bin?expected=2&pin={again['pin']}")
    assert response.json() == {"deleted": sorted([second.id, later.id]), "files": 2}
    assert (outputs / first.filename).exists()


def test_the_permanent_delete_takes_a_binned_image_too(client, store):
    image = _plant(store, settings.model_id, seed=1)
    client.post("/api/images/bin", json={"ids": [image.id]})
    assert client.delete(f"/api/images/{image.id}").json() == {"deleted": image.id}
    assert not (store.settings.outputs_dir / image.filename).exists()


def test_delete_unstarred_leaves_what_is_already_in_the_bin_out(client, fake):
    _, events = run_grid(client, steps_axis="6,8,10", seed=5)
    images = sorted(events[-1]["images"], key=lambda r: r["id"])
    client.post("/api/images/bin", json={"ids": [images[0]["id"]]})
    preview = client.get("/api/gallery/unstarred").json()
    assert (preview["records"], preview["unstarred"]) == (2, 2)
    response = client.delete("/api/gallery/unstarred?expected=2")
    assert response.status_code == 200, response.text
    assert response.json()["binned"] == [r["id"] for r in images[1:]]
    assert client.get("/api/gallery").json()["total"] == 0
    assert client.get("/api/gallery/bin").json()["total"] == 3
    response = client.delete("/api/gallery/unstarred?expected=1")
    assert response.status_code == 409 and "Nothing was moved" in response.json()["detail"]


def test_a_pass_whose_source_is_in_the_bin_says_so_until_it_is_restored(client, store):
    source = _plant(store, settings.model_id, seed=1)
    refined = _made_from(store, seed=2, refine=source.id, strength=0.5)
    client.post("/api/images/bin", json={"ids": [source.id]})
    items = {r["id"]: r for r in client.get("/api/gallery").json()["items"]}
    assert (items[refined.id]["refine_missing"], items[refined.id]["refine_binned"]) == (True, True)
    record = client.get(f"/api/images/{refined.id}").json()
    assert (record["refine_missing"], record["refine_binned"]) == (True, True)

    client.post("/api/images/restore", json={"ids": [source.id]})
    record = client.get(f"/api/images/{refined.id}").json()
    assert (record["refine_missing"], record["refine_binned"]) == (False, False)
    # Deleted for good: missing, and no Restore to offer.
    client.delete(f"/api/images/{source.id}")
    record = client.get(f"/api/images/{refined.id}").json()
    assert (record["refine_missing"], record["refine_binned"]) == (True, False)


# ---------------------------------------------------------------- newest id


def test_the_status_names_the_galleries_newest_id_binned_rows_included(client, store):
    # A page learns of images it did not watch being made - another tab's job
    # that ran between two polls - by this moving past what it has seen.
    assert client.get("/api/status").json()["newest_image_id"] == 0
    job_id, events = run_one(client, seed=7)
    made = events[-1]["images"][0]["id"]
    assert client.get("/api/status").json()["newest_image_id"] == made
    # Binned, still counted: only an insert raises it.
    assert client.post("/api/images/bin", json={"ids": [made]}).status_code == 200
    assert client.get("/api/status").json()["newest_image_id"] == made
    # Cleared, the table is empty and the next image's id is past the old one.
    store.clear_history()
    assert client.get("/api/status").json()["newest_image_id"] == 0
    planted = _plant(store, settings.model_id, seed=8)
    assert planted.id > made
    assert client.get("/api/status").json()["newest_image_id"] == planted.id


def test_a_gallery_page_carries_the_newest_id_read_with_it(client, store):
    # The page's starting point for the status's newest id: read under the
    # page's own reservation, every row's - another checkpoint's and the
    # bin's included, whatever the view's filter.
    page = client.get("/api/gallery").json()
    assert page["newest_image_id"] == 0
    mine = _plant(store, settings.model_id, seed=1)
    theirs = _plant(store, "somebody/else", prompt="a fox", seed=2)
    page = client.get("/api/gallery", params={"favorites": "true"}).json()
    assert page["items"] == [] and page["newest_image_id"] == theirs.id > mine.id
    assert client.post("/api/images/bin", json={"ids": [theirs.id]}).status_code == 200
    assert client.get("/api/gallery").json()["newest_image_id"] == theirs.id
    assert client.get("/api/gallery/bin").json()["newest_image_id"] == theirs.id
