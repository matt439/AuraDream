"""Private Tiled Refine carries one identity through storage and replay holds."""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import pytest
from PIL import Image

from backend import grid, main, metadata, png_settings
from backend import tiled_refine_contract as tiled
from backend.schemas import GenerateRequest
from tests.test_api_finish_run import retain


def _request():
    spec = tiled.StoredTiledRefine(
        source=17, recipe=tiled.RECIPE_NAME, preparation=tiled.PREPARATION,
        root_source={"filename": "root.png", "width": 1024, "height": 1024, "hash": "ad" * 32},
        prepared_rgb_sha256="be" * 32,
    )
    return GenerateRequest(
        prompt="a bridge by the dock", negative_prompt="blurry", width=2048, height=2048,
        steps=28, guidance_scale=5.0, strength=0.35, mu=None, seed=42, tiled_refine=spec,
    ).applied(negative_prompt_supported=True, cfg_threshold=1.0)


def _save(store, request, *, run="tiled-candidate"):
    with closing(Image.new("RGB", (request.width, request.height), "skyblue")) as image:
        return store.save_image(
            image, request, seed=42, duration_ms=200, model_id=tiled.MODEL_ID,
            quantization=tiled.QUANTIZATION, run_id=run, batch=1,
        )


def test_private_recipe_is_distinct_from_an_ordinary_cell_and_survives_sql_png_and_jobs(store):
    request = _request()
    cell = grid.Cell(request, 42)
    conventional = grid.Cell(request.model_copy(update={"tiled_refine": None}), 42)
    assert grid.cell_key(cell) != grid.cell_key(conventional)
    assert grid.run_id([cell], tiled.MODEL_ID, "fp8") != grid.run_id([conventional], tiled.MODEL_ID, "fp8")

    run = retain(store, [cell], model=tiled.MODEL_ID, quant=tiled.QUANTIZATION)
    result = _save(store, request, run=run)
    assert store.get_image(result.id).tiled_refine == request.tiled_refine
    assert store.retained_run_jobs(run)[0]["cells"] == [cell]
    replay = grid.cell_from_record(result)
    assert grid.cell_key(replay) == grid.cell_key(cell)
    assert store.completed_cells([request.prompt], 2048, 2048, tiled.MODEL_ID, "fp8") == {grid.cell_key(cell)}

    with store._session() as conn:
        row = dict(conn.execute("SELECT * FROM images WHERE id = ?", (result.id,)).fetchone())
    path = store.settings.outputs_dir / result.filename
    with Image.open(path) as image:
        assert metadata.tiled_refine_from_info(row, image.info) == request.tiled_refine
        with pytest.raises(ValueError, match="disagree"):
            metadata.tiled_refine_from_info({**row, "tiled_refine": None}, image.info)
        with pytest.raises(ValueError, match="sampling recipe"):
            metadata.tiled_refine_from_info({**row, "true_cfg": 4}, image.info)
        with pytest.raises(png_settings.PngSettingsError, match="Tiled Refine"):
            png_settings.parse_parameters(image.info["parameters"])
    with pytest.raises(png_settings.PngSettingsError, match="Tiled Refine"):
        png_settings.image_settings(str(path))


def test_public_generation_and_replay_hold_the_private_pass(client, store, fake, monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, model_id=tiled.MODEL_ID))
    request = _request()
    for route in ("/api/generate", "/api/grid", "/api/grid/preview"):
        refused = client.post(route, json=request.model_dump(mode="json"))
        assert refused.status_code == 422 and tiled.HELD in refused.text

    result = _save(store, request)
    ordinary = _save(store, GenerateRequest(prompt="an ordinary image"))
    refused = client.post(
        "/api/rerender/preview", json={"ids": [ordinary.id, result.id], "width": 1024, "height": 1024},
    )
    assert refused.status_code == 422 and refused.json()["detail"] == tiled.HELD
    checked = client.post(f"/api/runs/{result.run_id}/finish/preview")
    assert checked.status_code == 200 and checked.json()["reason"] == tiled.HELD
    refused = client.post(f"/api/runs/{result.run_id}/finish", json={})
    assert refused.status_code == 422 and refused.json()["detail"] == tiled.HELD
    assert fake.seeds == []
