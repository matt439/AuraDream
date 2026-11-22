"""Spatial generation's recipe travels through saved results and retained jobs."""
from __future__ import annotations

from contextlib import closing

import pytest
from PIL import Image

from backend import generate, generate_remote, grid, model_selection, more_seeds, refine_batch, then_refine
from backend.controlnet_contract import WEIGHTS, ControlNetSpec
from backend.schemas import GenerateRequest, ThenRefine
from backend.tiled_refine_contract import MODEL_ID, QUANTIZATION, ROOT_SIZE
from tests.test_api_finish_run import finish, preview, retain


def _regional_request(color: str) -> GenerateRequest:
    return GenerateRequest(
        prompt="a coat on a clothesline", width=ROOT_SIZE[0], height=ROOT_SIZE[1],
        steps=28, guidance_scale=5, seed=42,
        spatial_conditioning={
            "kind": "regional_prompts", "method": "sdxl-attention-v1",
            "regions": [{"left": 0, "top": 0, "right": 50, "bottom": 100, "prompt": f"{color} coat"}],
        },
    )


def test_saved_and_queued_spatial_recipes_compare_at_the_same_key(store):
    red = grid.Cell(_regional_request("red"), 42)
    blue = grid.Cell(_regional_request("blue"), 42)
    plain = grid.Cell(red.request.model_copy(update={"spatial_conditioning": None}), 42)
    assert len({grid.cell_key(cell) for cell in (red, blue, plain)}) == 3
    assert len({grid.run_id([cell], MODEL_ID, QUANTIZATION) for cell in (red, blue, plain)}) == 3
    run = grid.run_id([red, blue], MODEL_ID, QUANTIZATION)
    assert run == grid.run_id([blue, red], MODEL_ID, QUANTIZATION)

    retain(store, [red, blue], run, model=MODEL_ID, quant=QUANTIZATION, origin="generate")
    retained = store.retained_run_jobs(run)[0]["cells"]
    assert [grid.cell_key(cell) for cell in retained] == [grid.cell_key(red), grid.cell_key(blue)]

    with closing(Image.new("RGB", ROOT_SIZE, "red")) as pixels:
        row = store.save_image(pixels, red.request, seed=42, duration_ms=1, model_id=MODEL_ID,
                               quantization=QUANTIZATION, run_id=run)
    stored = store.get_image(row.id)
    assert stored.spatial_conditioning == red.request.spatial_conditioning
    assert grid.cell_key(grid.cell_from_record(stored)) == grid.cell_key(red)
    with_negative = stored.model_copy(update={"negative_prompt": "blurry", "true_cfg": 5.0})
    swapped = grid.cell_from_record(with_negative, negative_prompt="")
    assert swapped.request.true_cfg is None
    assert swapped.request.spatial_conditioning == red.request.spatial_conditioning
    assert more_seeds.settings_key(stored) == grid.cell_key(red)[:-1]
    done = store.completed_cells([red.request.prompt], *ROOT_SIZE, MODEL_ID, QUANTIZATION)
    assert done == {grid.cell_key(red)}
    assert grid.prune_done([red, blue, plain], done) == [blue, plain]

    batch, = more_seeds.group([stored])
    more_seeds.plan(batch, count=2, fallback=4, applied=lambda request: request, held=done)
    assert batch.status == more_seeds.QUEUED
    assert batch.seeds == [43, 44]
    assert [cell.request.spatial_conditioning for cell in batch.cells] == [red.request.spatial_conditioning] * 2
    assert [cell.request.num_images for cell in batch.cells] == [1, 1]

    follow = ThenRefine(scale=1, strength=0.35)
    assert "spatial conditioning" in (then_refine.refusal([red.request], follow, None) or "")
    planned = refine_batch.plan(
        stored, running=MODEL_ID, profile={}, scale=1.0, strength=0.35,
        steps=None, include_foreign=False, skip_passes=False,
    )
    assert planned.cell.request.refine == row.id
    assert planned.cell.request.spatial_conditioning is None


def test_controlnet_keys_name_the_proven_guide_pixels_across_gallery_remaps():
    original = ControlNetSpec(
        mode="depth", source=9, weight_revision=WEIGHTS["depth"].revision, strength=0.8,
        guide={"filename": "old.png", "width": 1024, "height": 1024, "hash": "2a" * 32},
    )
    request = _regional_request("red").model_dump()
    remapped_guide = original.guide.model_copy(update={"filename": "copied.png"})
    remapped = original.model_copy(update={"source": 47, "guide": remapped_guide})
    unresolved = original.model_copy(update={"source": None})
    different_guide = original.guide.model_copy(update={"hash": "c7" * 32})
    changed = original.model_copy(update={"guide": different_guide})
    cells = [grid.Cell(GenerateRequest(**{**request, "spatial_conditioning": spec}), 42)
             for spec in (original, remapped, unresolved, changed)]
    keys = [grid.cell_key(cell) for cell in cells]
    assert keys[0] == keys[1] == keys[2] != keys[3]
    assert isinstance(keys[0][-3], str)
    runs = [grid.run_id([cell], MODEL_ID, QUANTIZATION) for cell in cells]
    assert runs[0] == runs[1] == runs[2] != runs[3]


def test_finish_checks_qualification_then_replays_only_the_missing_regional_recipe(client, store, fake):
    model_selection.select(MODEL_ID, quantization=QUANTIZATION, offload="model")
    fake.loaded = False
    red, blue = [grid.Cell(_regional_request(color), 42) for color in ("red", "blue")]
    run = retain(store, [red, blue], model=MODEL_ID, quant=QUANTIZATION, origin="generate")
    with closing(Image.new("RGB", ROOT_SIZE, "red")) as pixels:
        store.save_image(pixels, red.request, seed=42, duration_ms=1, model_id=MODEL_ID,
                         quantization=QUANTIZATION, run_id=run, batch=1)

    refused = preview(client, run)
    assert not refused["available"]
    assert "Regional prompts" in refused["reason"]
    assert not fake.loaded

    fake.regional_prompts_qualified = True
    answer = preview(client, run)
    assert answer["available"] and answer["rendered"] == answer["cells"] == 1
    _, done = finish(client, run, answer)
    assert len(done["images"]) == 1
    assert done["images"][0]["spatial_conditioning"] == blue.request.spatial_conditioning.model_dump(mode="json")


def test_remote_cli_queues_the_pngs_spatial_recipe_only_on_a_qualified_server(store, monkeypatch, capsys):
    request = _regional_request("red")
    with closing(Image.new("RGB", ROOT_SIZE, "red")) as pixels:
        row = store.save_image(pixels, request, seed=42, duration_ms=1, model_id=MODEL_ID,
                               quantization=QUANTIZATION, batch=1)
    png = store.settings.outputs_dir / row.filename
    requests = []
    defaults = {"regional_prompts": False, "regional_prompts_reason": "Spatial card trial pending."}

    def server(_base, path, **kwargs):
        if path == "/api/status":
            return {"model_id": MODEL_ID, "quantization": QUANTIZATION, "offload": "model",
                    "model_revision": 6, "defaults": defaults}
        if path == "/api/generate":
            requests.append(kwargs)
            return {"job_id": "spatial-test"}
        assert path == "/api/jobs/spatial-test"
        return {"status": "done"}

    monkeypatch.setattr(generate_remote, "_server_json", server)
    monkeypatch.setattr(generate, "_project", lambda: pytest.fail("--via-server read its local gallery"))
    args = ["--from-image", str(png), "--via-server", "-q"]
    assert generate.main(args) == 2
    assert "Spatial card trial pending." in capsys.readouterr().err
    assert not requests

    defaults["regional_prompts"] = True
    assert generate.main(args) == 0
    submission, = requests
    assert submission["body"]["spatial_conditioning"] == request.spatial_conditioning.model_dump(mode="json")
    assert submission["headers"]["X-AuraDream-Model-Revision"] == "6"
