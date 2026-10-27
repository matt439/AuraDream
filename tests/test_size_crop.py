"""Paired canvas grids and source crops share one reproduction extension."""
from __future__ import annotations

import dataclasses
import json
import threading

import pytest
from PIL import Image

from backend import gallery_reports, grid, refine, schemas_generate, then_refine
from backend.schemas import GenerateRequest, ThenRefine
from backend.seeds import MAX_SEED
from tests.api_helpers import stream_job


def request(**changes):
    return GenerateRequest(**{
        "prompt": "a fox", "width": 256, "height": 256, "steps": 8,
        "guidance_scale": 2.0, "mu": None, "seed": 7, **changes,
    })


@pytest.mark.parametrize("text,expected", [
    ("", []), ("  ", []), ("256x512", [(256, 512)]),
    (" 256 x 512 , 512X256,256x512 ", [(256, 512), (512, 256), (256, 512)]),
])
def test_sizes_are_paired_ordered_and_repeated(text, expected):
    assert grid.parse_sizes(text) == expected


@pytest.mark.parametrize("text", ["256", "256x", "256x256 512x512", "256-512x256", "256x256,", ",256x256", "1.5x256"])
def test_size_syntax_has_no_implicit_product_or_range(text):
    with pytest.raises(ValueError, match="Size:"):
        grid.parse_sizes(text)


def test_size_axis_is_bounded_and_included_in_the_preallocation_ceiling(monkeypatch):
    with pytest.raises(ValueError, match="Size: 65 values"):
        grid.parse_sizes(",".join(["256x256"] * 65))
    monkeypatch.setattr(grid, "settings", dataclasses.replace(grid.settings, max_grid_images=3))
    with pytest.raises(ValueError, match="4 images"):
        grid.expand(request(), size="256x256,512x256", steps="4,8")


@pytest.mark.parametrize("size,said", [
    ("240x256", "greater than or equal"), ("257x256", "multiple of 16"),
    ("4464x256", "less than or equal"), ("2208x2208", "pixel budget"),
])
def test_every_concrete_size_passes_the_request_guards(size, said):
    with pytest.raises(ValueError, match=said):
        grid.expand(request(), size=size)


def test_every_concrete_size_has_its_own_batch_guard(monkeypatch):
    monkeypatch.setattr(schemas_generate, "settings", dataclasses.replace(
        schemas_generate.settings, max_batch_pixels=256 * 512,
    ))
    with pytest.raises(ValueError, match="in one batch"):
        grid.expand(request(num_images=2), size="256x256,512x256")
    # Random batches are separate one-image requests before this guard.
    cells = grid.expand(request(prompt="{animal}", num_images=2), size="256x256,512x256",
                        prompt="animal = fox | wolf", prompt_mode="random")
    assert len(cells) == 4 and all(cell.request.num_images == 1 for cell in cells)


def test_sizes_cross_inside_each_seed_and_prompt_with_the_same_noise():
    cells = grid.expand(request(prompt="{animal}", seed=MAX_SEED, num_images=2),
                        prompt="animal = fox | wolf", size="256x512,512x256", steps="4,8", seeds=2)
    assert len(cells) == 16
    assert [cell.seed for cell in cells] == [MAX_SEED] * 8 + [1] * 8
    assert [(cell.request.prompt, cell.request.width, cell.request.height, cell.request.steps) for cell in cells[:8]] == [
        (animal, w, h, steps) for animal in ("fox", "wolf")
        for w, h in ((256, 512), (512, 256)) for steps in (4, 8)
    ]


def test_random_assignments_follow_each_image_across_sizes_and_listed_seed_wrap():
    cells = grid.expand(request(prompt="{animal}", num_images=2), prompt="animal = fox | wolf",
                        prompt_mode="random", size="256x256,512x256", seed_list=[MAX_SEED, 5])
    assert [cell.seed for cell in cells] == [MAX_SEED, MAX_SEED, 0, 0, 5, 5, 6, 6]
    for a, b in zip(cells[::2], cells[1::2]):
        assert a.variables == b.variables and a.request.prompt == b.request.prompt


def test_mixed_size_hash_is_order_independent_and_uniform_center_stays_legacy():
    plain = grid.expand(GenerateRequest(prompt="a fox", width=256, height=256, seed=1), steps="4,8")
    assert grid.run_id(plain, "m", "fp8") == "490ea78ed76e"
    mixed = grid.expand(request(), size="256x512,512x256", steps="4,8")
    reverse = grid.expand(request(), size="512x256,256x512", steps="8,4")
    assert grid.run_id(mixed, "m", "fp8") == grid.run_id(reverse, "m", "fp8")
    assert grid.run_id(mixed, "m", "fp8") != grid.run_id([mixed[0]], "m", "fp8")
    centered = grid.expand(request(refine=3, strength=0.5))
    corner = grid.expand(request(refine=3, strength=0.5, crop_anchor="top-left"))
    assert grid.run_id(centered, "m", "fp8") != grid.run_id(corner, "m", "fp8")
    assert grid.cell_key(corner[0])[8] == "3@0.5"
    assert grid.cell_key(corner[0])[-2:] == ("256x256@top-left", 7)


def test_repeated_sizes_are_counted_and_warned_about_without_collapsing():
    cells = grid.expand(request(), size="256x256,512x256,256x256")
    assert len(cells) == 3 and grid.duplicate_count(cells, lambda r: r) == 1
    values = grid.axis_values(request(), "", "", "", size="256x256,512x256,256x256")
    assert values["size"] == ["256x256", "512x256", "256x256"]
    assert "3 sizes" in grid.describe(values, 1)


@pytest.mark.parametrize("anchor", refine.CROP_ANCHORS)
@pytest.mark.parametrize("source,target", [((9, 5), (4, 6)), ((5, 9), (6, 4)), ((3, 5), (2, 2)), ((7, 7), (7, 7))])
def test_anchor_pixels_keep_cover_rounding_and_exact_edges(anchor, source, target):
    image = Image.new("RGB", source)
    image.putdata([(x * 23, y * 23, (x + y) * 11) for y in range(source[1]) for x in range(source[0])])
    w, h = target
    cover = max(w / image.width, h / image.height)
    resized = image.resize((max(w, round(image.width * cover)), max(h, round(image.height * cover))), Image.Resampling.LANCZOS)
    dx, dy = resized.width - w, resized.height - h
    left = 0 if "left" in anchor else dx if "right" in anchor else dx // 2
    top = 0 if "top" in anchor else dy if "bottom" in anchor else dy // 2
    expected = resized.crop((left, top, left + w, top + h))
    actual = refine.fitted(image, w, h, anchor)
    assert actual.tobytes() == expected.tobytes()
    if source == target:
        assert actual is image


def test_gan_prepared_source_applies_the_same_anchor_after_preparation(store, monkeypatch):
    image = Image.new("RGB", (256, 512))
    image.paste("red", (0, 0, 256, 128))
    image.paste("blue", (0, 384, 256, 512))
    record = store.save_image(image, request(height=512), seed=1, duration_ms=10, model_id="m")
    monkeypatch.setattr(refine, "anime6b_upscale", lambda image, *a, **kw: image.resize((1024, 2048)))
    bottom = refine.source_image(record.id, 256, 256, upscaler=refine.ANIME6B, crop_anchor="bottom")
    assert bottom.getpixel((128, 255)) == (0, 0, 255)
    assert bottom.getpixel((128, 0)) == (0, 0, 0)


@pytest.mark.parametrize("changes", [
    {}, {"refine": 1, "mask": "ab" * 32}, {"refine": 1, "outpaint": {"left": 16}},
])
def test_noncenter_crop_needs_an_ordinary_source(changes):
    with pytest.raises(ValueError, match="crop anchor"):
        request(crop_anchor="top", **changes)


def test_anchor_is_reset_when_applied_drops_its_source():
    crop = request(refine=1, crop_anchor="bottom-right")
    assert crop.applied(negative_prompt_supported=True, cfg_threshold=1, refine_supported=True).crop_anchor == "bottom-right"
    dropped = crop.applied(negative_prompt_supported=True, cfg_threshold=1, refine_supported=False)
    assert dropped.refine is None and dropped.crop_anchor == "center"


BODY = {"prompt": "a fox", "width": 256, "height": 256, "seed": 7, "steps": 8,
        "guidance_scale": 2.0, "mu": None, "size_axis": "256x256,512x256", "seeds": 2}


def test_mixed_grid_resume_reads_every_actual_size_and_retains_sizes(client, fake, store):
    accepted = client.post("/api/grid", json=BODY).json()
    done = stream_job(client, accepted["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 4
    assert [(r.width, r.height) for r in fake.requests] == [(256, 256), (512, 256)] * 2
    preview = client.post("/api/grid/preview", json={**BODY, "resume": True}).json()
    assert (preview["cells"], preview["skipped"]) == (0, 4)
    assert client.post("/api/grid", json={**BODY, "resume": True}).status_code == 409
    details = client.get(f"/api/runs/{accepted['run_id']}").json()
    assert details["sizes"] == [[256, 256], [512, 256]] and details["size_count"] == 2
    summary = next(run for run in client.get("/api/runs").json() if run["run_id"] == accepted["run_id"])
    assert summary["sizes"] == details["sizes"] and summary["size_count"] == 2
    # A permanently deleted cell is eligible again at precisely its own size.
    lost = next(image for image in done["images"] if image["width"] == 512)
    assert client.delete(f"/api/images/{lost['id']}").status_code == 200
    preview = client.post("/api/grid/preview", json={**BODY, "resume": True}).json()
    assert (preview["cells"], preview["skipped"]) == (1, 3)


def test_mixed_grid_resume_sees_planned_keys_before_gallery_awaits(client, fake):
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        accepted = client.post("/api/grid", json=BODY).json()
        preview = client.post("/api/grid/preview", json={**BODY, "resume": True}).json()
        assert preview["cells"] == 0 and preview["queued"] + preview["skipped"] == 4
    finally:
        release.set()
    assert stream_job(client, accepted["job_id"])[-1]["status"] == "done"


def test_mixed_retained_run_recovers_its_sizes_and_finishes_deleted_cells(client, fake):
    accepted = client.post("/api/grid", json=BODY).json()
    images = stream_job(client, accepted["job_id"])[-1]["images"]
    for image in images:
        assert client.delete(f"/api/images/{image['id']}").status_code == 200
    recovered = next(run for run in client.get("/api/runs").json() if run["run_id"] == accepted["run_id"])
    assert recovered["images"] == 0 and recovered["sizes"] == [[256, 256], [512, 256]]
    preview = client.post(f"/api/runs/{accepted['run_id']}/finish/preview").json()
    assert preview["available"] and preview["cells"] == 4
    finished = client.post(f"/api/runs/{accepted['run_id']}/finish", json={"expected": preview["pin"]}).json()
    assert len(stream_job(client, finished["job_id"])[-1]["images"]) == 4
    assert [(r.width, r.height) for r in fake.requests[-4:]] == [(256, 256), (512, 256)] * 2


def test_resume_keeps_a_partial_batch_at_its_actual_size(client, fake, store):
    for width, seed in [(256, 7), (512, 7), (512, 8)]:
        store.save_image(Image.new("RGB", (width, 256)), request(width=width, seed=seed),
                         seed=seed, duration_ms=10, model_id=store.settings.model_id,
                         quantization=store.settings.quantization)
    preview = client.post("/api/grid/preview", json={**BODY, "seeds": 1, "num_images": 2, "resume": True}).json()
    assert (preview["cells"], preview["images"], preview["skipped"]) == (1, 2, 1)


def test_size_axis_cannot_resize_a_mask_or_outpaint(client, store):
    source = store.save_image(Image.new("RGB", (256, 256)), request(), seed=1, duration_ms=10, model_id="m")
    mask, _ = store.store_mask(Image.new("L", (256, 256), 255))
    body = {**BODY, "seeds": 1, "refine": source.id, "strength": 0.5, "mask": mask}
    answer = client.post("/api/grid/preview", json=body)
    assert answer.status_code == 422 and "source's size" in answer.json()["detail"]
    body.update(mask=None, outpaint={"left": 32, "right": 16, "band": 16}, size_axis="304x256,320x256")
    answer = client.post("/api/grid/preview", json=body)
    assert answer.status_code == 422 and "304x256" in answer.json()["detail"]


def test_preview_prices_each_size_and_each_then_refine_pass(client, monkeypatch):
    monkeypatch.setattr(gallery_reports, "step_cost_ms", lambda w, h, *a: {256: 1000, 512: 2000}.get(w))
    monkeypatch.setattr(gallery_reports, "refine_cost_ms", lambda w, h, *a, **kw: {4: {384: 3000, 768: 7000}.get(w)} if w in (384, 768) else {})
    monkeypatch.setattr(gallery_reports, "image_bytes", lambda w, h, *a: w)
    body = {**BODY, "seeds": 1, "budget_minutes": 10 / 60,
            "then_refine": {"scale": 1.5, "strength": 0.5}}
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["pass_sizes"] == [[384, 384], [768, 384]]
    assert preview["pass_width"] is None and preview["pass_height"] is None
    assert preview["seconds"] == 34 and preview["bytes"] == 256 + 512 + 384 + 768
    assert preview["budget_seeds"] == 0
    monkeypatch.setattr(gallery_reports, "step_cost_ms", lambda w, h, *a: 1000 if w == 256 else None)
    monkeypatch.setattr(gallery_reports, "image_bytes", lambda w, h, *a: 256 if w == 256 else None)
    unknown = client.post("/api/grid/preview", json=body).json()
    assert unknown["seconds"] is None and unknown["budget_seeds"] is None and unknown["bytes"] is None


def test_then_refine_rejects_a_later_oversized_canvas():
    bases = [request(), request(width=2208, height=1024)]
    spec = ThenRefine(scale=2, strength=0.5)
    assert "2208×1024" in then_refine.refusal(bases, spec, None)


def test_crop_resume_does_not_confuse_anchors(client, fake, store):
    source = store.save_image(Image.new("RGB", (512, 256)), request(width=512), seed=1, duration_ms=10, model_id="m")
    body = {**BODY, "seeds": 1, "size_axis": "", "refine": source.id, "strength": 0.5, "crop_anchor": "right"}
    accepted = client.post("/api/grid", json=body).json()
    assert stream_job(client, accepted["job_id"])[-1]["status"] == "done"
    right = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    left = client.post("/api/grid/preview", json={**body, "resume": True, "crop_anchor": "left"}).json()
    assert (right["cells"], right["skipped"]) == (0, 1)
    assert (left["cells"], left["skipped"]) == (1, 0)
    with Image.open(store.settings.outputs_dir / store.list_images()[0][0].filename) as png:
        assert json.loads(png.info["auradream"])["crop_anchor"] == "right"
