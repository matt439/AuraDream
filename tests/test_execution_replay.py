"""Stored execution metadata must never silently replay as ordinary eager work."""
from __future__ import annotations

import argparse
import io
import json
import time

import pytest
from PIL import Image, PngImagePlugin

from backend import generate, grid, main, more_seeds, png_settings
from backend.config import settings
from backend.execution import ExecutionSpec
from backend.metadata import PNG_METADATA_KEY
from backend.schemas import GenerateRequest


RECIPES = [
    {"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"},
    {"step_cache": "fbc-005-v1"},
    {"compilation": "regional-v1", "attention_backend": "cudnn-v1", "step_cache": "fbc-005-v1"},
]


def _metadata(recipe):
    return {
        "prompt": "1girl, red hair", "negative_prompt": None,
        "steps": 20, "guidance_scale": 1.5, "mu": None,
        "seed": 7, "size": "256x256", "model": settings.model_id,
        "execution": recipe,
    }


def _plant(store, recipe, seed=7):
    # A synthetic archived result, not a request admitted to generation.
    request = GenerateRequest(prompt="1girl, red hair", width=256, height=256, steps=20)
    request = request.model_copy(update={"execution": ExecutionSpec(**recipe) if recipe else None})
    return store.save_image(
        Image.new("RGB", (256, 256)), request, seed=seed, duration_ms=1,
        model_id=settings.model_id, quantization=settings.quantization,
        run_id="synthetic-execution-run", job_id="synthetic-execution-job", batch=1,
    )


@pytest.mark.parametrize("recipe", RECIPES)
def test_png_record_preserves_known_unavailable_recipe_as_metadata(recipe):
    meta = _metadata(recipe)
    assert png_settings.record({PNG_METADATA_KEY: json.dumps(meta)}, "sample.png") == meta


@pytest.mark.parametrize("recipe", [{"compilation": "future"}, {"unknown": None}, [], "regional-v1"])
def test_png_record_rejects_unknown_recipe_instead_of_falling_back(recipe):
    chunks = {PNG_METADATA_KEY: json.dumps(_metadata(recipe)), "parameters": "a different image"}
    with pytest.raises(png_settings.PngSettingsError, match="metadata"):
        png_settings.record(chunks, "sample.png")


@pytest.mark.parametrize("recipe", RECIPES)
def test_png_panel_refuses_unavailable_recipe_without_applying_eager_settings(client, fake, store, recipe):
    response = client.post("/api/png/settings", json={
        "name": "sample.png", "chunks": {
            PNG_METADATA_KEY: json.dumps(_metadata(recipe)),
            "parameters": "fallback\nSteps: 8, CFG scale: 1, Seed: 9, Size: 256x256",
        },
    })
    assert response.status_code == 422, response.text
    assert "not available" in response.json()["detail"]
    assert fake.requests == [] and fake.loads == 0
    assert store.list_images()[1] == 0


@pytest.mark.parametrize("recipe", RECIPES)
def test_cli_png_refuses_before_environment_sources_or_backend_work(recipe, tmp_path, monkeypatch, capsys):
    path = tmp_path / "sample.png"
    chunks = PngImagePlugin.PngInfo()
    chunks.add_text(PNG_METADATA_KEY, json.dumps(_metadata(recipe)))
    Image.new("RGB", (16, 16)).save(path, pnginfo=chunks)

    def forbidden(*args, **kwargs):
        raise AssertionError("unavailable execution reached CLI setup")

    monkeypatch.setattr(generate, "apply_env", forbidden)
    monkeypatch.setattr(generate, "_project", forbidden)
    with pytest.raises(SystemExit) as caught:
        generate.main(["--from-image", str(path), "--dry-run", "--model", "other/model", "replacement prompt"])
    assert caught.value.code == 2
    assert "not available" in capsys.readouterr().err


def test_cli_image_helper_keeps_legacy_metadata(tmp_path):
    path = tmp_path / "legacy.png"
    meta = _metadata(None)
    del meta["execution"]
    chunks = PngImagePlugin.PngInfo()
    chunks.add_text(PNG_METADATA_KEY, json.dumps(meta))
    Image.new("RGB", (16, 16)).save(path, pnginfo=chunks)
    assert generate._image_settings(str(path), argparse.ArgumentParser()) == meta


@pytest.mark.parametrize("key", [PNG_METADATA_KEY, "krea"])
@pytest.mark.parametrize("with_header_record", [False, True])
def test_file_settings_refuse_late_owned_metadata_instead_of_missing_its_recipe(
    tmp_path, key, with_header_record,
):
    chunks = PngImagePlugin.PngInfo()
    if with_header_record:
        chunks.add_text(key, json.dumps(_metadata(None)))
    chunks.add_text(key, json.dumps(_metadata({"compilation": "regional-v1"})))
    stream = io.BytesIO()
    Image.new("RGB", (16, 16)).save(stream, format="PNG", pnginfo=chunks)
    raw, at, parts = stream.getvalue(), 8, []
    while at < len(raw):
        size = int.from_bytes(raw[at:at + 4], "big") + 12
        parts.append(raw[at:at + size])
        at += size
    owned = [index for index, chunk in enumerate(parts) if chunk[4:8] == b"tEXt"]
    late = parts.pop(owned[-1])
    path = tmp_path / "late.png"
    path.write_bytes(raw[:8] + b"".join([*parts[:-1], late, parts[-1]]))
    # Pillow's header-only info misses the tail, or sees only the eager
    # first record. Reading the file must refuse before treating it as eager.
    with pytest.raises(png_settings.PngSettingsError, match="metadata"):
        png_settings.image_settings(str(path))
    with pytest.raises(SystemExit) as caught:
        generate._image_settings(str(path), argparse.ArgumentParser())
    assert caught.value.code == 2


@pytest.mark.parametrize("recipe", RECIPES)
def test_record_reruns_refuse_recipe_instead_of_losing_it(client, store, fake, recipe):
    record = _plant(store, recipe)
    response = client.post("/api/rerender", json={"ids": [record.id], "width": 384, "height": 256})
    assert response.status_code == 422, response.text
    assert "not available" in response.json()["detail"]
    for route, body, rows in [
        ("/api/swap-tags/preview", {"ids": [record.id], "swaps": [{"find": "red hair", "replace": "silver hair"}]}, "images"),
        ("/api/refine/batch/preview", {"ids": [record.id], "scale": 1.5}, "images"),
        ("/api/more-seeds/preview", {"ids": [record.id], "count": 2}, "results"),
    ]:
        response = client.post(route, json=body)
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["queued"] == 0
        assert answer[rows][0]["status"] == "invalid", answer
        assert "not available" in answer[rows][0]["detail"]
    assert fake.requests == [] and fake.loads == 0 and not main.jobs.active()
    assert store.get_image(record.id).execution.model_dump(exclude_none=True) == recipe


def test_more_seeds_keeps_distinct_execution_batches_separate(store):
    records = [_plant(store, {}, seed=1)]
    records += [_plant(store, recipe, seed=index + 2) for index, recipe in enumerate(RECIPES)]
    assert len({more_seeds.settings_key(record) for record in records}) == len(records)
    assert len(more_seeds.group(records)) == len(records)


@pytest.mark.parametrize("recipe", RECIPES)
def test_retained_run_recovery_cannot_erase_unavailable_recipe(client, fake, store, recipe):
    request = GenerateRequest(prompt="retained", width=256, height=256, steps=1)
    request = request.model_copy(update={"execution": ExecutionSpec(**recipe)})
    store.retain_run_job({
        "job_id": "retained-execution", "run_id": "retained-execution-run", "created_at": time.time(),
        "origin": "grid", "model_id": settings.model_id, "quantization": settings.quantization,
        "width": 256, "height": 256, "batch": 1, "comparison": False, "metadata": {"grid": True},
    }, [grid.Cell(request=request, seed=7)])
    with pytest.raises(ValueError, match="cannot be read"):
        store.retained_run_jobs("retained-execution-run")
    for suffix in ("/finish/preview", "/finish"):
        response = client.post(f"/api/runs/retained-execution-run{suffix}", json={})
        assert response.status_code == 422, response.text
    assert fake.requests == [] and fake.loads == 0 and not main.jobs.active()
