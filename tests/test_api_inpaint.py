"""An inpaint over HTTP: the mask stored from its upload and served back by
its hash, refused at submit where it, its source or the checkpoint cannot
run, an ordinary strength grid over one mask, and recorded on the row, in
the PNG's chunk and with its own bytes embedded."""
from __future__ import annotations

import base64
import dataclasses
import io
import json

import pytest
from PIL import Image

from backend import grid, main
from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEY
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid, run_one, stream_job, wait_for
from tests.fakes import fake_fill


def _source(client, size: int = 256) -> dict:
    """An image to inpaint: a new render of `size` square, its record."""
    run_one(client, width=size, height=size)
    return client.get("/api/gallery").json()["items"][0]


def _png(size=(256, 256), box=(0, 0, 128, 64), mode="L", level=255) -> bytes:
    """A mask's PNG: black, with `box` painted at `level`."""
    image = Image.new(mode, size, 0)
    image.paste(level if mode == "L" else (level,) * len(mode), box)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _upload(client, source: int, data: bytes):
    return client.post(
        "/api/masks", params={"source": source}, content=data, headers={"content-type": "image/png"}
    )


def _stored(client, record: dict, **kwargs) -> str:
    response = _upload(client, record["id"], _png((record["width"], record["height"]), **kwargs))
    assert response.status_code == 200, response.text
    return response.json()["hash"]


def _inpaint(record: dict, mask: str, **extra) -> dict:
    return {
        "prompt": "a red scarf", "width": record["width"], "height": record["height"], "steps": 10,
        "refine": record["id"], "strength": 0.6, "mask": mask, **extra,
    }


# ------------------------------------------------------------------ the upload


def test_an_upload_is_kept_by_the_hash_of_its_pixels(client, store):
    record = _source(client)
    first = _upload(client, record["id"], _png())
    assert first.status_code == 200
    body = first.json()
    assert (body["width"], body["height"], body["created"]) == (256, 256, True)
    assert len(body["hash"]) == 64
    path = store.masks_dir() / f"{body['hash']}.png"
    assert path.is_file()
    with Image.open(path) as handle:
        assert handle.mode == "L" and handle.size == (256, 256)
        assert body["hash"] == store.mask_hash(handle.convert("L"))
    assert store.mask_size(body["hash"]) == (256, 256)
    # The same picture is the same hash, and nothing new is written - in
    # any mode the page might send it, since it is kept as one channel.
    again = _upload(client, record["id"], _png(mode="RGB")).json()
    assert (again["hash"], again["created"]) == (body["hash"], False)
    assert list(store.masks_dir().glob("*.png*")) == [path]
    # Another painting is another mask.
    other = _upload(client, record["id"], _png(box=(0, 0, 64, 64))).json()
    assert other["hash"] != body["hash"] and other["created"] is True
    # A mask is no gallery image: no row, no card.
    assert client.get("/api/gallery").json()["total"] == 1


def test_a_mask_is_served_back_by_its_hash(client, store):
    record = _source(client)
    digest = _stored(client, record)
    served = client.get(f"/api/masks/{digest}")
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"
    assert served.content == (store.masks_dir() / f"{digest}.png").read_bytes()
    assert client.get(f"/api/masks/{'0' * 64}").status_code == 404
    # A hash, never a path: anything else is no route parameter at all.
    assert client.get("/api/masks/..%2F..%2Fauradream.db").status_code in (404, 422)
    assert client.get(f"/api/masks/{'A' * 64}").status_code == 422


def test_an_upload_refuses_what_is_no_mask_over_its_image(client, store):
    record = _source(client)
    jpeg = io.BytesIO()
    Image.new("RGB", (256, 256)).save(jpeg, format="JPEG")
    for data, words in (
        (jpeg.getvalue(), "not one"),
        (b"\x89PNG\r\n\x1a\nnot really", "cannot be read"),
        (_png((384, 384)), "a mask covers its image exactly"),
        (_png(level=0), "Nothing is masked"),
    ):
        response = _upload(client, record["id"], data)
        assert response.status_code == 422 and words in response.json()["detail"], response.text
    # An all-white mask repaints the whole picture: a refine, and allowed.
    assert _upload(client, record["id"], _png(box=(0, 0, 256, 256))).status_code == 200
    assert len(list(store.masks_dir().glob("*.png"))) == 1


def test_an_upload_names_a_source_that_is_gone_or_binned(client, store):
    record = _source(client)
    missing = _upload(client, 4242, _png())
    assert missing.status_code == 404 and missing.json()["detail"] == "No gallery image with id 4242."
    assert client.post("/api/images/bin", json={"ids": [record["id"]]}).status_code == 200
    binned = _upload(client, record["id"], _png())
    assert binned.status_code == 422 and "recycle bin" in binned.json()["detail"]
    assert not store.masks_dir().exists() or list(store.masks_dir().iterdir()) == []


def test_an_upload_is_held_to_the_refine_guard(client, store, monkeypatch):
    record = _source(client)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, refine_max_pixels=60_000))
    too_many = _upload(client, record["id"], _png())
    assert too_many.status_code == 422 and "past the 60,000 pixels" in too_many.json()["detail"]
    # And the body itself, before a byte of it is decoded.
    huge = b"\x89PNG\r\n\x1a\n" + b"\0" * (60_000 * 4 + (1 << 20) + 1)
    assert _upload(client, record["id"], huge).status_code == 413


# ---------------------------------------------------------------- the inpaint


def test_an_inpaint_reaches_the_backend_the_row_and_the_png(client, fake, store):
    record = _source(client)
    digest = _stored(client, record)
    _, events = run_one(client, **_inpaint(record, digest, seed=5))
    assert events[-1]["status"] == "done"
    ran = fake.requests[-1]
    assert (ran.refine, ran.mask, ran.strength) == (record["id"], digest, 0.6)
    newest = client.get("/api/gallery").json()["items"][0]
    assert (newest["refine"], newest["strength"], newest["mask"]) == (record["id"], 0.6, digest)
    assert record["mask"] is None
    # The fake repaints only under the mask: the rest is the source's pixels.
    outputs = store.settings.outputs_dir
    with Image.open(outputs / record["filename"]) as handle:
        source = handle.convert("RGB")
    with Image.open(outputs / newest["filename"]) as handle:
        info = dict(handle.info)
        image = handle.convert("RGB")
    assert image.crop((0, 64, 256, 256)).tobytes() == source.crop((0, 64, 256, 256)).tobytes()
    assert image.getpixel((10, 10)) == fake_fill(5)
    # The chunk names the mask beside its source, and the blend it ran with.
    meta = json.loads(info[PNG_METADATA_KEY])
    assert meta["mask"] == digest and meta["mask_blend"] == "overlay"
    assert meta["mask_source"] == {"filename": f"{digest}.png", "hash": digest, "width": 256, "height": 256}
    assert f", Denoising strength: 0.6, Refine: {record['id']}, Inpaint mask: {digest[:16]}" in info["parameters"]
    # And the mask's own bytes travel with it.
    assert base64.b64decode(info[PNG_MASK_KEY]) == (store.masks_dir() / f"{digest}.png").read_bytes()
    # A text-to-image row and a refine pass say so by absence.
    with Image.open(outputs / record["filename"]) as handle:
        plain = dict(handle.info)
    assert PNG_MASK_KEY not in plain
    assert json.loads(plain[PNG_METADATA_KEY])["mask"] is None
    assert "Inpaint mask" not in plain["parameters"]


def test_an_inpaint_is_refused_at_submit_where_it_cannot_run(client, fake, store):
    record = _source(client)
    larger = _source(client, 384)
    digest = _stored(client, record)
    on_larger = _stored(client, larger)
    made = len(fake.requests)
    cases = [
        (_inpaint(record, "0" * 64), "no stored mask 0000000000000000"),
        (_inpaint(record, digest, width=384, height=384), "an inpaint runs at its source's size"),
        (_inpaint(record, on_larger), "is 384x384, but source image"),
        ({**_inpaint(record, digest), "refine": None}, "a mask needs the image it covers"),
    ]
    for body, words in cases:
        for route, extra in (("/api/generate", {}), ("/api/grid", {"seed": 1, "strength_axis": "0.4,0.6"})):
            response = client.post(route, json={**body, **extra})
            assert response.status_code == 422 and words in response.text, (route, response.text)
    # Its file gone from the masks directory.
    (store.masks_dir() / f"{digest}.png").unlink()
    response = client.post("/api/generate", json=_inpaint(record, digest))
    assert response.status_code == 422 and "missing from the masks directory" in response.json()["detail"]
    assert len(fake.requests) == made


def test_an_inpaint_over_a_binned_source_is_refused(client, fake, store):
    record = _source(client)
    digest = _stored(client, record)
    assert client.post("/api/images/bin", json={"ids": [record["id"]]}).status_code == 200
    response = client.post("/api/generate", json=_inpaint(record, digest))
    assert response.status_code == 422 and "recycle bin" in response.json()["detail"]


def test_a_checkpoint_without_an_inpaint_refuses_one(client, fake, store):
    record = _source(client)
    digest = _stored(client, record)
    fake.inpaint_supported = False
    made = len(fake.requests)
    response = client.post("/api/generate", json=_inpaint(record, digest))
    assert response.status_code == 422 and "has no inpaint in this app" in response.json()["detail"]
    # The refine pass is another question, and still answers.
    plain = {**_inpaint(record, digest), "mask": None}
    _, events = run_one(client, **plain)
    assert events[-1]["status"] == "done" and len(fake.requests) == made + 1
    # And the other way: an inpaint does not need an image-to-image class.
    fake.inpaint_supported, fake.refine_supported = True, False
    _, events = run_one(client, **_inpaint(record, digest))
    assert events[-1]["status"] == "done" and fake.requests[-1].mask == digest


def test_a_strength_sweep_runs_over_one_mask(client, fake, store):
    record = _source(client)
    digest = _stored(client, record)
    payload = {**_inpaint(record, digest), "seed": 1, "strength_axis": "0.4,0.6,0.8"}
    preview = client.post("/api/grid/preview", json=payload).json()
    assert preview["cells"] == 3 and preview["values"]["strength"] == [0.4, 0.6, 0.8]
    assert preview["duplicates"] == 0
    _, events = run_grid(client, **payload)
    assert events[-1]["status"] == "done"
    assert [(r.mask, r.strength) for r in fake.requests[-3:]] == [(digest, 0.4), (digest, 0.6), (digest, 0.8)]
    rows = client.get("/api/gallery", params={"search": f"refine:{record['id']}"}).json()["items"]
    assert sorted(row["strength"] for row in rows) == [0.4, 0.6, 0.8]
    assert {row["mask"] for row in rows} == {digest}
    # Run again with Resume, every cell is in the gallery under its mask...
    again = client.post("/api/grid", json={**payload, "resume": True})
    assert again.status_code == 409
    # ...and over another mask, none is.
    other = _stored(client, record, box=(0, 0, 64, 64))
    preview = client.post("/api/grid/preview", json={**payload, "mask": other}).json()
    assert preview["in_gallery"] == 0


def test_the_preview_says_where_an_inpaint_starts(client, fake, store, monkeypatch):
    # The Inpaint dialog's "starts at N% noise", as Refine's: asked with the
    # inpaint as the pipeline runs it, mask and all, and not for a sweep.
    record = _source(client)
    digest = _stored(client, record)
    asked = []
    monkeypatch.setattr(fake, "start_sigma", lambda request: asked.append(request) or 0.51)
    payload = {**_inpaint(record, digest), "seed": 1}
    assert client.post("/api/grid/preview", json=payload).json()["start_sigma"] == 0.51
    (request,) = asked
    assert request.inpainting and request.mask == digest
    asked.clear()
    sweep = {**payload, "strength_axis": "0.4,0.6"}
    assert client.post("/api/grid/preview", json=sweep).json()["start_sigma"] is None
    assert asked == []


def test_the_job_names_its_mask(client, fake, store):
    record = _source(client)
    digest = _stored(client, record)
    job_id, _ = run_one(client, **_inpaint(record, digest))
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert (snapshot["refine"], snapshot["mask"]) == (record["id"], digest)


def test_a_job_over_many_sources_is_named_by_none_of_them():
    from backend.grid import Cell
    from backend.jobs import Job

    mask = "ab" * 32
    inpaint = GenerateRequest(prompt="a", width=256, height=256, refine=3, strength=0.6, mask=mask)
    plain = GenerateRequest(prompt="b", width=256, height=256)
    mixed = Job(id="x", request=inpaint, cells=[Cell(request=inpaint, seed=1), Cell(request=plain, seed=2)],
                grid=True, swap="red hair → silver hair")
    assert (mixed.snapshot()["refine"], mixed.snapshot()["mask"]) == (None, None)
    # A sweep over one source - Refine's strength grid - still names it.
    swept = inpaint.model_copy(update={"strength": 0.8})
    one = Job(id="y", request=inpaint, cells=[Cell(request=inpaint, seed=1), Cell(request=swept, seed=1)],
              grid=True)
    assert (one.snapshot()["refine"], one.snapshot()["mask"]) == (3, mask)


def test_a_cell_whose_mask_went_fails_in_words(client, fake, store, monkeypatch):
    record = _source(client)
    digest = _stored(client, record)
    path = store.masks_dir() / f"{digest}.png"
    saved = store.save_image

    def save_then_lose_the_mask(*args, **kwargs):
        # Gone once the first cell's image is saved, before the second began.
        try:
            return saved(*args, **kwargs)
        finally:
            path.unlink(missing_ok=True)

    monkeypatch.setattr(store, "save_image", save_then_lose_the_mask)
    made = len(fake.requests)
    _, events = run_grid(client, **_inpaint(record, digest), seed=1, strength_axis="0.4,0.6")
    done = events[-1]
    assert done["status"] == "done" and done["failures"] == 1
    assert "the file for mask" in done["error"] and "missing from the masks directory" in done["error"]
    # The first was saved with its mask; the second never reached the backend.
    assert len(done["images"]) == 1 and done["images"][0]["mask"] == digest
    assert [r.strength for r in fake.requests[made:]] == [0.4]


def test_a_cancel_mid_inpaint_keeps_what_finished(client, fake, store):
    record = _source(client)
    digest = _stored(client, record)
    fake.step_seconds = 0.01
    job_id = client.post("/api/generate", json=_inpaint(record, digest, runs=8)).json()["job_id"]
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    done = stream_job(client, job_id)[-1]
    assert done["status"] == "cancelled"
    assert 1 <= len(done["images"]) < 8
    assert all(image["mask"] == digest for image in done["images"])


# ---------------------------------------------------------------- the key


def test_two_masks_over_one_source_are_two_pictures():
    base = dict(prompt="a fox", width=256, height=256, steps=10, refine=3, strength=0.6, seed=1)
    one, two = "ab" * 32, "cd" * 32
    cells = [grid.Cell(request=GenerateRequest(**base, mask=mask), seed=1) for mask in (one, two, one)]

    def applied(request):
        return request.applied(
            negative_prompt_supported=True, cfg_threshold=1.0, refine_supported=True, inpaint_supported=True
        )

    # The third repeats the first; the second is another region.
    assert grid.duplicate_count(cells, applied, "cagliostrolab/animagine-xl-4.0") == 1
    assert grid.duplicate_count(cells[:2], applied, "cagliostrolab/animagine-xl-4.0") == 0


@pytest.mark.parametrize("missing", ["mask", "source"])
def test_the_mask_check_reads_the_table_and_the_source(client, store, missing):
    record = _source(client)
    digest = _stored(client, record)
    if missing == "mask":
        with store._session() as conn:
            conn.execute("DELETE FROM masks")
        words = "no stored mask"
    else:
        assert client.delete(f"/api/images/{record['id']}").status_code == 200
        words = f"no gallery image with id {record['id']}"
    response = client.post("/api/generate", json=_inpaint(record, digest))
    assert response.status_code == 422 and words in response.json()["detail"]


# ------------------------------------------------- the gallery's own actions


def _inpainted(client, fake, prompt="1girl, red hair, long hair") -> tuple[dict, dict, str]:
    record = _source(client)
    digest = _stored(client, record)
    _, events = run_one(client, **_inpaint(record, digest, prompt=prompt, seed=9))
    assert events[-1]["status"] == "done"
    return record, events[-1]["images"][0], digest


def test_more_seeds_and_swap_tags_run_over_the_same_mask(client, fake, store):
    record, made, digest = _inpainted(client, fake)
    response = client.post("/api/more-seeds", json={"ids": [made["id"]], "count": 2})
    assert response.status_code == 200, response.text
    assert stream_job(client, response.json()["job_id"])[-1]["status"] == "done"
    assert [(r.refine, r.mask) for r in fake.requests[-2:]] == [(record["id"], digest)] * 2
    swap = [{"find": "red hair, long hair", "replace": "silver hair"}]
    response = client.post("/api/swap-tags", json={"swaps": swap, "ids": [made["id"]]})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    # The prompt rewritten over the same region: a regional edit.
    assert (ran.prompt, ran.refine, ran.mask) == ("1girl, silver hair", record["id"], digest)


def test_mass_refine_makes_the_inpaint_its_source_and_drops_the_mask(client, fake, store):
    record, made, digest = _inpainted(client, fake)
    response = client.post("/api/refine/batch", json={"ids": [made["id"]], "strength": 0.5, "scale": 1.5,
                                                       "skip_passes": False})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    assert (ran.refine, ran.mask, ran.width) == (made["id"], None, 384)


def test_a_re_render_of_an_inpaint_whose_mask_is_missing_names_it(client, fake, store):
    record, made, digest = _inpainted(client, fake)
    (store.masks_dir() / f"{digest}.png").unlink()
    made_before = len(fake.requests)
    response = client.post("/api/rerender", json={"ids": [made["id"]], "width": 256, "height": 256})
    assert response.status_code == 422
    assert f"the mask of #{made['id']} is missing" in response.json()["detail"]
    assert len(fake.requests) == made_before


def test_search_and_stats_find_an_inpaint_by_its_mask(client, fake, store):
    from backend import gallery_reports

    record, made, digest = _inpainted(client, fake)

    def ids(search):
        response = client.get("/api/gallery", params={"search": search})
        assert response.status_code == 200, response.text
        return {item["id"] for item in response.json()["items"]}

    assert ids("inpaint:yes") == ids(f"mask:{digest[:16]}") == {made["id"]}
    assert ids("inpaint:no") == ids("mask:none") == {record["id"]}
    assert client.get("/api/gallery", params={"search": "inpaint:maybe"}).status_code == 422
    rows = gallery_reports.keep_rates(by=["mask"])
    assert {row["mask"] for row in rows} == {None, digest[:16]}


def test_swap_tags_and_more_seeds_name_a_missing_mask_by_its_image(client, fake, store):
    record, made, digest = _inpainted(client, fake)
    (store.masks_dir() / f"{digest}.png").unlink()
    said = f"the mask of #{made['id']} is missing"
    swap = [{"find": "red hair", "replace": "silver hair"}]
    preview = client.post("/api/swap-tags/preview", json={"swaps": swap, "ids": [made["id"]]})
    assert preview.status_code == 200, preview.text
    (image,) = [i for i in preview.json()["images"] if i["id"] == made["id"]]
    assert image["status"] == "mask_missing" and said in image["detail"], image
    assert digest[:16] not in image["detail"] and "import the bundle" in image["detail"]
    preview = client.post("/api/more-seeds/preview", json={"ids": [made["id"]], "count": 2})
    assert preview.status_code == 200, preview.text
    (batch,) = preview.json()["results"]
    assert batch["status"] == "mask_missing" and said in batch["detail"], batch
    # And nothing can be queued from either.
    assert client.post("/api/swap-tags", json={"swaps": swap, "ids": [made["id"]]}).status_code == 422
    assert client.post("/api/more-seeds", json={"ids": [made["id"]], "count": 2}).status_code == 422


def test_a_re_render_of_an_inpaint_at_another_size_names_the_inpaint(client, fake, store):
    record, made, digest = _inpainted(client, fake)
    plain = _source(client)
    before = len(fake.requests)
    for ids in ([made["id"]], [plain["id"], made["id"]]):
        response = client.post("/api/rerender", json={"ids": ids, "width": 384, "height": 384})
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert f"#{made['id']} is an inpaint of #{record['id']} at 256x256" in detail, detail
        assert ("Unpick" in detail) is (len(ids) > 1)
    # From a filter, the way out is the search.
    response = client.post("/api/rerender", json={"favorites": False, "width": 384, "height": 384})
    assert response.status_code == 422 and "inpaint:no" in response.json()["detail"]
    assert len(fake.requests) == before and main.jobs.active() == []


def test_a_reused_mask_is_kept_from_the_sweep_without_an_upload(client, store):
    from backend import gallery_delete

    record = _source(client)
    digest = _stored(client, record)
    with store._session() as conn:
        conn.execute("UPDATE masks SET created_at = ? WHERE hash = ?", ("2000-01-01T00:00:00+00:00", digest))
    kept = client.post(f"/api/masks/{digest}/keep")
    assert kept.status_code == 200 and kept.json() == {
        "hash": digest, "width": 256, "height": 256, "created": False,
    }
    assert gallery_delete.sweep_masks(None) == 0
    assert (store.masks_dir() / f"{digest}.png").is_file()
    # Gone, it says so, and the dialog uploads the picture it holds instead.
    (store.masks_dir() / f"{digest}.png").unlink()
    assert client.post(f"/api/masks/{digest}/keep").status_code == 404
    assert client.post(f"/api/masks/{'0' * 64}/keep").status_code == 404


def test_an_upload_with_alpha_or_16_bit_levels_is_kept_as_painted(client, store):
    import numpy as np

    record = _source(client)
    plain = _stored(client, record)
    # White strokes on a transparent white layer: only the strokes.
    layer = Image.new("RGBA", (256, 256), (255, 255, 255, 0))
    layer.paste((255, 255, 255, 255), (0, 0, 128, 64))
    out = io.BytesIO()
    layer.save(out, format="PNG")
    assert _upload(client, record["id"], out.getvalue()).json()["hash"] == plain
    levels = np.zeros((256, 256), dtype=np.uint16)
    levels[:64, :128] = 65535
    out = io.BytesIO()
    Image.fromarray(levels).save(out, format="PNG")
    assert _upload(client, record["id"], out.getvalue()).json()["hash"] == plain


def test_load_settings_from_an_inpaint_png_carries_no_mask_source_or_size(client, fake, store):
    record, made, digest = _inpainted(client, fake)
    with Image.open(store.settings.outputs_dir / made["filename"]) as handle:
        chunks = {key: value for key, value in handle.info.items() if isinstance(value, str)}
    assert PNG_MASK_KEY in chunks
    # The page blanks the mask's chunk, as it is no setting (web/png-settings.js).
    chunks[PNG_MASK_KEY] = ""
    response = client.post("/api/png/settings", json={"name": "made.png", "chunks": chunks})
    assert response.status_code == 200, response.text
    body = response.json()
    assert any("It is an inpaint" in note for note in body["notes"]), body["notes"]
    settings = body["settings"]
    assert settings["mask"] is None and settings["refine"] is None and body["size"] is False
