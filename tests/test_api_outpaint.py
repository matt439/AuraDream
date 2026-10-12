"""An outpaint over HTTP: refused at submit where it cannot run, its mask
built and stored before the job is made (and held, and healed), run over
the padded source, recorded on the row, in the chunk and the A1111 line,
carried by Re-render, Swap tags and More seeds, dropped by Mass refine,
and the presets route."""
from __future__ import annotations

import base64
import dataclasses
import json

import numpy as np
import pytest
from PIL import Image

from backend import gallery_lock, grid, main, schemas_generate, storage, submission
from backend.config import settings
from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEY
from backend.schemas import GenerateRequest, Outpaint
from tests.api_helpers import run_grid, run_one, stream_job
from tests.fakes import fake_fill

SPEC = {"left": 32, "right": 16, "band": 16}


def _gradient(width: int, height: int) -> Image.Image:
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([x % 256, y % 256, ((x * 3 + y) % 200) + 20], axis=2).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def plant(store, prompt="1girl, red hair, long hair", *, seed=1, width=256, height=256, quantization=None,
          **fields):
    # The row takes the picture's own size, so an off-grid source is a
    # picture of that size under any request.
    return store.save_image(
        _gradient(width, height),
        GenerateRequest(prompt=prompt, width=width - width % 16, height=height - height % 16, steps=10, **fields),
        seed=seed, duration_ms=1, model_id=settings.model_id, quantization=quantization,
    )


def plant_outpaint(store, source, spec=SPEC, *, prompt="1girl, red hair, long hair", seed=2, quantization=None):
    """An outpaint's row as a run of one writes it: stamped, then saved."""
    shape = Outpaint(**spec)
    width = source.width + shape.left + shape.right
    height = source.height + shape.top + shape.bottom
    request = GenerateRequest(
        prompt=prompt, width=width, height=height, steps=10, refine=source.id, strength=0.9, outpaint=spec,
    )
    (cell,) = storage.stamp_outpaints([grid.Cell(request=request, seed=seed)])
    return store.save_image(
        Image.new("RGB", (width, height)), cell.request, seed=seed, duration_ms=1,
        model_id=settings.model_id, quantization=quantization,
    )


def body(source, spec=SPEC, **extra) -> dict:
    side = {name: spec.get(name, 0) for name in ("left", "top", "right", "bottom")}
    return {
        "prompt": "a wider view", "width": source.width + side["left"] + side["right"],
        "height": source.height + side["top"] + side["bottom"], "steps": 10, "refine": source.id,
        "outpaint": spec, **extra,
    }


# ------------------------------------------------------------------- the run


def test_an_outpaint_runs_over_its_padded_source_and_is_recorded(client, fake, store):
    source = plant(store)
    _, events = run_one(client, **body(source), seed=5)
    assert events[-1]["status"] == "done", events[-1]
    expected = storage.outpaint_mask_hash(304, 256, Outpaint(**SPEC))
    ran = fake.requests[-1]
    assert (ran.refine, ran.mask, ran.outpaint.key()) == (source.id, expected, "edge-box3-r16:32,0,16,0@16")
    newest = client.get("/api/gallery").json()["items"][0]
    assert (newest["width"], newest["height"], newest["refine"], newest["mask"]) == (304, 256, source.id, expected)
    assert newest["outpaint"] == {"fill": "edge-box3-r16", "left": 32, "top": 0, "right": 16, "bottom": 0, "band": 16}
    # The stored mask is the server's, at the canvas's size.
    assert storage.mask_size(expected) == (304, 256)
    outputs = store.settings.outputs_dir
    with Image.open(outputs / source.filename) as handle:
        original = handle.convert("RGB")
    with Image.open(outputs / newest["filename"]) as handle:
        info = dict(handle.info)
        image = handle.convert("RGB")
    # Inside the band's inner edges the source exactly; the border the fake's fill.
    assert image.crop((48, 0, 272, 256)).tobytes() == original.crop((16, 0, 240, 256)).tobytes()
    assert image.getpixel((0, 0)) == fake_fill(5) and image.getpixel((303, 255)) == fake_fill(5)
    meta = json.loads(info[PNG_METADATA_KEY])
    assert meta["outpaint"] == {"left": 32, "top": 0, "right": 16, "bottom": 0, "band": 16}
    assert meta["outpaint_fill"] == "edge-box3-r16"
    assert meta["mask"] == expected and meta["mask_blend"] == "overlay"
    assert meta["mask_source"]["width"] == 304
    assert meta["refine_source"]["width"] == 256
    assert ", Outpaint: 32/0/16/0, Outpaint band: 16, Outpaint fill: edge-box3-r16" in info["parameters"]
    assert base64.b64decode(info[PNG_MASK_KEY]) == (store.masks_dir() / f"{expected}.png").read_bytes()
    # A row that is no outpaint says so by absence.
    with Image.open(outputs / source.filename) as handle:
        plain = json.loads(handle.info[PNG_METADATA_KEY])
    assert (plain["outpaint"], plain["outpaint_fill"]) == (None, None)
    # Searchable, and every outpaint is an inpaint too.
    assert [r["id"] for r in client.get("/api/gallery", params={"search": "outpaint:yes"}).json()["items"]] == [newest["id"]]
    assert source.id in [r["id"] for r in client.get("/api/gallery", params={"search": "outpaint:no"}).json()["items"]]
    assert client.get("/api/gallery", params={"search": "inpaint:yes outpaint:no"}).json()["total"] == 0


def test_the_mask_is_stamped_before_the_job_and_held_under_its_hash(client, fake, store, monkeypatch):
    source = plant(store)
    held = []
    hold = storage.hold_masks

    def spy(job_id, masks):
        held.append(set(masks))
        return hold(job_id, masks)

    monkeypatch.setattr(storage, "hold_masks", spy)
    unstamped = [grid.Cell(request=GenerateRequest(**body(source), seed=7), seed=7 + i) for i in range(2)]
    response = client.post("/api/generate", json={**body(source), "seed": 7, "runs": 2})
    assert response.status_code == 200, response.text
    job = main.jobs._jobs[response.json()["job_id"]]
    expected = storage.outpaint_mask_hash(304, 256, Outpaint(**SPEC))
    assert [cell.request.mask for cell in job.cells] == [expected, expected]
    assert job.request.mask == expected and job.request.outpaint is not None
    assert held == [{expected}]
    # The run id keys an outpaint by its spec, so stamping changes nothing.
    assert response.json()["run_id"] == grid.run_id(unstamped, main.services.model_id(), main.services.quantization())
    assert job.snapshot()["outpaint"]["left"] == 32 and job.snapshot()["refine"] == source.id
    stream_job(client, job.id)


def test_a_stamp_that_cannot_store_is_a_worded_503(client, fake, store, monkeypatch):
    source = plant(store)

    def busy(_mask):
        raise gallery_lock.GalleryBusy("another operation holds the gallery")

    monkeypatch.setattr(storage, "store_mask", busy)
    response = client.post("/api/generate", json=body(source))
    assert response.status_code == 503
    assert response.json()["detail"] == "The gallery is busy with another operation; submit the outpaint again."

    def full(_mask):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(storage, "store_mask", full)
    response = client.post("/api/generate", json=body(source))
    assert response.status_code == 503
    assert response.json()["detail"] == "The outpaint's mask could not be written: No space left on device."
    assert main.jobs.active() == [] and not fake.requests


def test_a_lost_mask_file_is_healed_by_the_next_submit(client, fake, store):
    source = plant(store)
    _, events = run_one(client, **body(source), seed=1)
    digest = events[-1]["images"][0]["mask"]
    path = store.masks_dir() / f"{digest}.png"
    path.unlink()
    # No outpaint row is ever reported as missing its mask: it is rebuilt.
    assert submission.missing_masks(store.get_images([events[-1]["images"][0]["id"]])) == []
    _, events = run_one(client, **body(source), seed=2)
    assert events[-1]["status"] == "done" and path.is_file()


def test_a_mask_gone_before_its_hold_says_submit_it_again(client, fake, store, monkeypatch):
    source = plant(store)
    monkeypatch.setattr(storage, "hold_masks", lambda job_id, masks: sorted(masks))
    response = client.post("/api/generate", json=body(source))
    assert response.status_code == 422
    assert response.json()["detail"].endswith("has just gone from this gallery; submit it again.")


def test_a_cell_whose_mask_went_says_submit_it_again(client, fake, store, monkeypatch):
    source = plant(store)
    saved = store.save_image
    expected = storage.outpaint_mask_hash(304, 256, Outpaint(**SPEC))

    def save_then_lose_the_mask(*args, **kwargs):
        try:
            return saved(*args, **kwargs)
        finally:
            (store.masks_dir() / f"{expected}.png").unlink(missing_ok=True)

    monkeypatch.setattr(store, "save_image", save_then_lose_the_mask)
    _, events = run_one(client, **body(source), seed=1, runs=2)
    done = events[-1]
    assert done["failures"] == 1
    assert "the outpaint's mask has gone from this gallery; submit it again" in done["error"]


# ---------------------------------------------------------------- refusals


def test_an_outpaint_is_refused_at_submit_where_it_cannot_run(client, fake, store):
    source = plant(store)
    cases = [
        ({**body(source), "width": 320}, f"an outpaint of #{source.id} (256x256) by +32/+0/+16/+0 is 304x256, "
                                         "not the 320x256 asked for"),
        (body(source, {"left": 32, "right": 32, "band": 128}),
         f"a band of 128 px on the left and right sides leaves 0 px of #{source.id}'s width unrepainted; "
         "at least 16 must be kept, so narrow the band"),
        # One padded side: the band comes off that side alone.
        (body(source, {"left": 16, "band": 248}),
         f"a band of 248 px on the left side leaves 8 px of #{source.id}'s width unrepainted; "
         "at least 16 must be kept, so narrow the band"),
        (body(source, {**SPEC, "fill": "edge-box5-r8"}),
         "this build has no outpaint fill 'edge-box5-r8'; it fills with edge-box3-r16"),
        (body(source, {**SPEC, "band": 264}), "less than or equal to 256"),
        (body(source, {"left": 50}), "50 is not a multiple of 16: 48 or 64"),
        (body(source, {"left": 0}), "an outpaint extends at least one side"),
        ({**body(source), "mask": "a" * 64}, "an outpaint builds its own mask; send none with it"),
        ({**body(source), "refine": None}, "an outpaint needs the image it extends"),
        ({**body(source), "refine": 9999}, "no gallery image with id 9999"),
    ]
    for payload, words in cases:
        for route, extra in (("/api/generate", {}), ("/api/grid", {"seed": 1, "strength_axis": "0.6,0.9"})):
            response = client.post(route, json={**payload, **extra})
            assert response.status_code == 422 and words in response.text, (route, words, response.text)
    assert not fake.requests and not store.masks_dir().exists()


def test_an_outpaint_past_the_refine_budget_is_refused(client, fake, store, monkeypatch):
    source = plant(store)
    monkeypatch.setattr(schemas_generate, "settings", dataclasses.replace(schemas_generate.settings,
                                                                          refine_max_pixels=70_000))
    response = client.post("/api/generate", json=body(source))
    assert response.status_code == 422 and "exceeds the 70,000 pixel budget of a refine pass" in response.text


def test_an_outpaint_over_a_binned_source_is_refused(client, fake, store):
    source = plant(store)
    assert client.post("/api/images/bin", json={"ids": [source.id]}).status_code == 200
    response = client.post("/api/generate", json=body(source))
    assert response.status_code == 422 and "recycle bin" in response.json()["detail"]


def test_a_checkpoint_without_an_inpaint_class_refuses_an_outpaint_in_its_words(client, fake, store):
    source = plant(store)
    fake.inpaint_supported = False
    response = client.post("/api/generate", json=body(source))
    assert response.status_code == 422
    assert "has no inpaint in this app" in response.text and "which an outpaint runs through" in response.text
    # An inpaint class is what it needs, not an image-to-image one.
    fake.inpaint_supported, fake.refine_supported = True, False
    _, events = run_one(client, **body(source))
    assert events[-1]["status"] == "done" and fake.requests[-1].outpaint is not None


def test_an_off_grid_source_is_named(client, fake, store):
    source = plant(store, width=264, height=264)
    # Over HTTP the canvas's own validator is the backstop.
    response = client.post("/api/generate", json=body(source))
    assert response.status_code == 422 and "is not a multiple of 16" in response.text
    # The submit check names the source.
    request = GenerateRequest(**body(source, width=304, height=256)).model_copy(update={"width": 312, "height": 264})
    assert submission.outpaint_unusable(request) == (
        f"#{source.id} is 264 x 264, off the 16-px grid, so no outpaint of it is either"
    )


# ------------------------------------------------------------- the presets


def test_the_presets_route_answers_the_servers_numbers(client, fake, store):
    source = plant(store, width=1024, height=1024)
    answer = client.get(f"/api/images/{source.id}/outpaint-presets")
    assert answer.status_code == 200, answer.text
    answer = answer.json()
    assert (answer["width"], answer["height"], answer["off_grid"]) == (1024, 1024, None)
    assert (answer["fill"], answer["band"], answer["max_band"], answer["min_kept"]) == ("edge-box3-r16", 32, 256, 16)
    presets = {p["name"]: p for p in answer["presets"]}
    assert [presets["16:9"][k] for k in ("left", "top", "right", "bottom", "width")] == [400, 0, 400, 0, 1824]
    assert presets["16:9"]["held"] is None
    assert presets["1:1"]["held"] == f"#{source.id} is already 1:1 - the 1:1 preset at 1 MP"
    off = plant(store, width=1016, height=1016)
    held = client.get(f"/api/images/{off.id}/outpaint-presets").json()
    assert held["off_grid"] == f"#{off.id} is 1016 x 1016, off the 16-px grid, so no outpaint of it is either"
    assert {p["held"] for p in held["presets"]} == {held["off_grid"]}
    gone = client.get("/api/images/9999/outpaint-presets")
    assert gone.status_code == 404 and gone.json()["detail"] == "no gallery image with id 9999"
    assert client.post("/api/images/bin", json={"ids": [source.id]}).status_code == 200
    # A binned source says so, so the page offers a Restore, never "deleted".
    binned = client.get(f"/api/images/{source.id}/outpaint-presets")
    assert binned.status_code == 404
    assert binned.json()["detail"] == f"no gallery image with id {source.id} (#{source.id} is in the recycle bin)"


# ------------------------------------------------------------ grid and resume


def test_a_strength_sweep_runs_over_one_outpaint_and_resumes_by_its_spec(client, fake, store):
    source = plant(store)
    payload = {**body(source), "seed": 1, "strength_axis": "0.6,0.9"}
    preview = client.post("/api/grid/preview", json=payload).json()
    assert preview["cells"] == 2 and preview["duplicates"] == 0
    # Nothing is stored for a preview: it keys the outpaint by its spec.
    assert not store.masks_dir().exists()
    _, events = run_grid(client, **payload)
    assert events[-1]["status"] == "done"
    assert [(r.outpaint.key(), r.strength) for r in fake.requests[-2:]] == [
        ("edge-box3-r16:32,0,16,0@16", 0.6), ("edge-box3-r16:32,0,16,0@16", 0.9),
    ]
    assert client.post("/api/grid", json={**payload, "resume": True}).status_code == 409
    assert client.post("/api/grid/preview", json=payload).json()["in_gallery"] == 2
    # Another band, or another padding, is another picture.
    wider = {**payload, "outpaint": {**SPEC, "band": 24}}
    assert client.post("/api/grid/preview", json=wider).json()["in_gallery"] == 0


# ---------------------------------------------------------- gallery actions


def test_rerender_carries_the_outpaint_and_refuses_another_size(client, fake, store):
    source = plant(store)
    # Another quantizer's row, so re-rendering it here is no duplicate.
    row = plant_outpaint(store, source, quantization="elsewhere")
    response = client.post("/api/rerender", json={"ids": [row.id], "width": 304, "height": 256})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    assert ran.outpaint.key() == row.outpaint.key() and ran.mask == row.mask
    refused = client.post("/api/rerender", json={"ids": [row.id], "width": 512, "height": 512})
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert f"#{row.id} is an outpaint of #{source.id} at 304x256" in detail
    assert "an outpaint runs at its source's size plus its padding" in detail


def _set_outpaint(store, row, key, mask=None) -> None:
    with store._session() as conn:
        conn.execute("UPDATE images SET outpaint = ?, mask = COALESCE(?, mask) WHERE id = ?", (key, mask, row.id))


def test_rerender_checks_each_outpaints_fill_apart_under_one_mask(client, fake, store):
    source = plant(store)
    good = plant_outpaint(store, source, quantization="elsewhere")
    other = plant_outpaint(store, source, seed=3, quantization="elsewhere")
    # The same sides and band, so the same mask: only the fill tells them apart.
    _set_outpaint(store, other, "edge-zz9:32,0,16,0@16")
    response = client.post("/api/rerender", json={"ids": [good.id, other.id], "width": 304, "height": 256})
    assert response.status_code == 422, response.text
    assert "this build has no outpaint fill 'edge-zz9'" in response.json()["detail"]
    assert not fake.requests


def test_an_outpaint_whose_spec_does_not_build_its_mask_is_refused_by_every_rerun(client, fake, store):
    source = plant(store, "a hill")
    row = plant_outpaint(store, source, quantization="elsewhere")
    # As a bundle's `masks_missing` row or a reindexed chunk can be: the
    # spec's band is not the one its mask was built with.
    _set_outpaint(store, row, "edge-box3-r16:32,0,16,0@24")
    words = f"the outpaint of #{row.id} was made under a mask this build does not build from its spec"
    response = client.post("/api/rerender", json={"ids": [row.id], "width": 304, "height": 256})
    assert response.status_code == 422 and words in response.json()["detail"], response.text
    swaps = [{"find": "red hair, long hair", "replace": "silver hair"}]
    answer = client.post("/api/swap-tags/preview", json={"swaps": swaps, "ids": [row.id]}).json()
    (image,) = answer["images"]
    assert (image["status"], words in image["detail"]) == ("outpaint", True)
    answer = client.post("/api/more-seeds/preview", json={"ids": [row.id], "count": 2}).json()
    (result,) = answer["results"]
    assert (result["status"], words in result["detail"]) == ("outpaint", True)
    assert not fake.requests


@pytest.mark.parametrize("key, words", [
    ("edge-box3-r16:0,0,0,0@16", "an outpaint extends at least one side"),
    ("edge-box3-r16:32,0,16,0@264", "less than or equal to 256"),
])
def test_rerender_of_a_row_no_request_could_make_is_a_worded_422(client, fake, store, key, words):
    source = plant(store)
    row = plant_outpaint(store, source, quantization="elsewhere")
    # Its mask the one its spec builds, so only the request model refuses it.
    spec = storage.OutpaintSpec.parse_key(key)
    _set_outpaint(store, row, key, storage.outpaint_mask_hash(row.width, row.height, spec))
    response = client.post("/api/rerender", json={"ids": [row.id], "width": row.width, "height": row.height})
    assert response.status_code == 422 and words in response.json()["detail"], response.text
    assert not fake.requests


def test_a_record_runs_again_as_its_outpaint_with_no_mask_of_its_own():
    record = storage.ImageRecord(
        id=3, filename="x.png", url="/outputs/x.png", prompt="x", negative_prompt=None, width=304, height=256, steps=10,
        guidance_scale=1.0, seed=4, model_id=settings.model_id, duration_ms=1, created_at="now",
        refine=2, strength=0.9, mask="b" * 64,
        outpaint=storage.OutpaintSpec.parse_key("edge-box3-r16:32,0,16,0@16"),
    )
    cell = grid.cell_from_record(record)
    assert cell.request.outpaint.key() == "edge-box3-r16:32,0,16,0@16" and cell.request.mask is None
    # Its key is the record's resume key, by its spec.
    assert grid.cell_key(cell)[9] == "outpaint:edge-box3-r16:32,0,16,0@16"


def test_swap_tags_checks_an_outpaint_apart_from_a_refine_of_its_source(client, fake, store):
    source = plant(store, "a hill")
    unknown = plant_outpaint(store, source, {**SPEC, "fill": "edge-box5-r8"}, seed=3)
    refined = plant(store, seed=4, width=304, height=256, refine=source.id, strength=0.5)
    good = plant_outpaint(store, source, seed=5)
    swaps = [{"find": "red hair, long hair", "replace": "silver hair"}]
    answer = client.post("/api/swap-tags/preview", json={"swaps": swaps, "ids": [refined.id, unknown.id, good.id]})
    assert answer.status_code == 200, answer.text
    images = {image["id"]: image for image in answer.json()["images"]}
    assert images[refined.id]["status"] == "queued"
    assert images[unknown.id]["status"] == "outpaint"
    assert images[unknown.id]["detail"] == "this build has no outpaint fill 'edge-box5-r8'; it fills with edge-box3-r16"
    assert images[good.id]["status"] == "queued"
    response = client.post("/api/swap-tags", json={
        "swaps": swaps, "ids": [good.id], "expected": [good.id],
    })
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    assert ran.prompt == "1girl, silver hair" and ran.outpaint.key() == good.outpaint.key()


def test_more_seeds_checks_and_carries_an_outpaint(client, fake, store):
    source = plant(store, "a hill")
    unknown = plant_outpaint(store, source, {**SPEC, "fill": "edge-box5-r8"}, seed=3)
    good = plant_outpaint(store, source, {**SPEC, "band": 8}, seed=5)
    answer = client.post("/api/more-seeds/preview", json={"ids": [unknown.id, good.id], "count": 2})
    assert answer.status_code == 200, answer.text
    results = {row["id"]: row for row in answer.json()["results"]}
    assert results[unknown.id]["status"] == "outpaint"
    assert results[good.id]["status"] == "queued"
    response = client.post("/api/more-seeds", json={"ids": [good.id], "count": 2, "expected": [good.id]})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    assert [r.outpaint.key() for r in fake.requests[-2:]] == [good.outpaint.key()] * 2
    assert [r.seed for r in fake.requests[-2:]] != [good.seed] * 2


def test_mass_refine_drops_the_outpaint(client, fake, store):
    source = plant(store)
    row = plant_outpaint(store, source)
    answer = client.post("/api/refine/batch/preview", json={"ids": [row.id], "strength": 0.5, "scale": 1.0})
    images = {image["id"]: image for image in answer.json()["images"]}
    assert images[row.id]["status"] == "pass"
    assert images[row.id]["detail"].startswith(f"an outpaint of #{source.id}")
    response = client.post("/api/refine/batch", json={
        "ids": [row.id], "strength": 0.5, "scale": 1.0, "skip_passes": False, "expected": [row.id],
    })
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    assert (ran.refine, ran.mask, ran.outpaint) == (row.id, None, None)


def test_load_settings_from_an_outpaints_png_says_so_and_takes_no_size(client, fake, store):
    source = plant(store)
    row = plant_outpaint(store, source)
    with Image.open(store.settings.outputs_dir / row.filename) as image:
        chunks = {key: value for key, value in image.info.items() if isinstance(value, str)}
    answer = client.post("/api/png/settings", json={"name": "out.png", "chunks": chunks})
    assert answer.status_code == 200, answer.text
    answer = answer.json()
    assert answer["size"] is False
    assert any(note.startswith("It is an outpaint, which is made again from its source") for note in answer["notes"])
    assert answer["settings"]["outpaint"] is None and answer["settings"]["refine"] is None


@pytest.mark.parametrize("status_type", ["SwapStatus", "MoreSeedsStatus"])
def test_the_status_types_accept_an_outpaint(status_type):
    from typing import get_args

    from backend import schemas_rerun

    assert "outpaint" in get_args(getattr(schemas_rerun, status_type))
    assert submission.UNUSABLE_STATUS["outpaint"] == "outpaint"
