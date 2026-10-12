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
    assert client.get("/api/masks/..%2F..%2Flocal-diffusion.db").status_code in (404, 422)
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


def test_the_job_names_its_mask(client, fake, store):
    record = _source(client)
    digest = _stored(client, record)
    job_id, _ = run_one(client, **_inpaint(record, digest))
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert (snapshot["refine"], snapshot["mask"]) == (record["id"], digest)


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
