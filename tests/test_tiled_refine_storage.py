"""Tiled Refine keeps its original source through storage and public replay."""
from __future__ import annotations

import threading
from contextlib import closing
from dataclasses import replace

import pytest
from PIL import Image

from backend import config, generate, generate_remote, grid, main, metadata, png_settings, tiled_refine_admission
from backend import tiled_refine_contract as tiled
from backend.schemas import GenerateRequest
from tests.api_helpers import stream_job
from tests.cli_helpers import cli, isolated_execution_environment  # noqa: F401 - fixtures by name
from tests.test_api_finish_run import finish, preview, retain


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


def _source(store):
    with closing(Image.new("RGB", tiled.ROOT_SIZE, "navy")) as image:
        row = store.save_image(image, GenerateRequest(prompt="original dock", width=1024, height=1024),
                               seed=17, duration_ms=20,
                               model_id=tiled.MODEL_ID, quantization=tiled.QUANTIZATION)
    return row, tiled_refine_admission.bind_source(tiled.TiledRefineSpec(source=row.id, recipe=tiled.RECIPE_NAME))


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
    changed = grid.cell_from_record(result, negative_prompt="")
    assert changed.request.true_cfg is None and changed.request.tiled_refine == result.tiled_refine
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
        with pytest.raises(png_settings.PngSettingsError, match="Open Reuse"):
            png_settings.record(image.info, result.filename)
    assert png_settings.image_settings(str(path))["tiled_refine"] == request.tiled_refine.model_dump(mode="json")


def test_unqualified_backend_grid_and_resize_hold_tiled_intent(client, store, fake, monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, model_id=tiled.MODEL_ID, quantization=tiled.QUANTIZATION))
    request = _request()
    for route in ("/api/generate", "/api/grid", "/api/grid/preview"):
        refused = client.post(route, json=request.model_dump(mode="json"))
        assert refused.status_code == 422 and "Tiled Refine" in refused.text

    result = _save(store, request)
    ordinary = _save(store, GenerateRequest(prompt="an ordinary image"))
    refused = client.post(
        "/api/rerender/preview", json={"ids": [ordinary.id, result.id], "width": 1024, "height": 1024},
    )
    assert refused.status_code == 422 and "fixed 2048x2048" in refused.json()["detail"]
    checked = client.post(f"/api/runs/{result.run_id}/finish/preview")
    assert checked.status_code == 200 and "No Tiled Refine run plan" in checked.json()["reason"]
    refused = client.post(f"/api/runs/{result.run_id}/finish", json={})
    assert refused.status_code == 409 and "No Tiled Refine run plan" in refused.json()["detail"]
    assert fake.seeds == []


def test_finish_and_more_seeds_keep_the_bound_original_source(client, store, fake, monkeypatch):
    main.model_selection.select(tiled.MODEL_ID, quantization=tiled.QUANTIZATION, offload=tiled.OFFLOAD)
    monkeypatch.setattr(fake, "tiled_refine_refusal", lambda *_args, **_kwargs: None)
    root, spec = _source(store)
    fake.hold = threading.Event()
    fake.hold_when = lambda _request: len(fake.requests) == 2
    try:
        queued = client.post("/api/generate", json={
            **_request().model_dump(mode="json"), "true_cfg": None, "num_images": 2,
            "tiled_refine": {"source": root.id, "recipe": tiled.RECIPE_NAME},
        })
        assert queued.status_code == 200, queued.text
        run, job_id = queued.json()["run_id"], queued.json()["job_id"]
        assert fake.holding.wait(10), "the second tiled cell did not start"
        assert client.post(f"/api/jobs/{job_id}/cancel").is_success
        stopped = stream_job(client, job_id)[-1]
        assert stopped["status"] == "cancelled" and [row["seed"] for row in stopped["images"]] == [42]
        saved = store.get_image(stopped["images"][0]["id"])
        assert saved.tiled_refine == spec and saved.true_cfg == 5
    finally:
        fake.hold.set()

    with Image.open(store.settings.outputs_dir / saved.filename) as image:
        chunks = {key: value for key, value in image.info.items() if isinstance(value, str)}
    png_refusal = client.post("/api/png/settings", json={"name": saved.filename, "chunks": chunks})
    assert png_refusal.status_code == 422
    assert png_refusal.json()["detail"] == (
        f"The AuraDream metadata in {saved.filename} has a Tiled Refine recipe tied to its original gallery source. "
        "Open Reuse on the saved image or use --from-image on the command line."
    )

    store.bin_images([root.id])
    blocked = preview(client, run)
    assert not blocked["available"] and "recycle bin" in blocked["reason"]
    blocked_more = client.post("/api/more-seeds/preview", json={"ids": [saved.id], "count": 1})
    assert blocked_more.status_code == 200 and "recycle bin" in blocked_more.text
    assert fake.seeds == [42, 43]

    store.restore_images([root.id])
    ready = preview(client, run)
    assert (ready["available"], ready["rendered"], ready["cells"]) == (True, 1, 1)
    _, done = finish(client, run, ready)
    assert done["images"][0]["tiled_refine"] == spec.model_dump(mode="json")
    more = client.post("/api/more-seeds", json={"ids": [saved.id], "count": 2, "then_refine": True})
    assert more.status_code == 200, more.text
    done = stream_job(client, more.json()["job_id"])[-1]
    assert done["status"] == "done" and [row["seed"] for row in done["images"]] == [44, 45]
    assert all(row["tiled_refine"] == spec.model_dump(mode="json") for row in done["images"])
    assert fake.seeds == [42, 43, 43, 44, 45]
    assert all(req.tiled_refine == spec and req.num_images == 1 for req in fake.requests)


def test_cli_renders_and_replays_tiled_recipe_and_refuses_mismatched_original(cli, store, monkeypatch, capsys):
    monkeypatch.setattr(config, "settings", replace(store.settings, model_id=tiled.MODEL_ID,
                                                     quantization=tiled.QUANTIZATION))
    monkeypatch.setattr(cli, "tiled_refine_refusal", lambda *_args, **_kwargs: None)
    root, spec = _source(store)
    flags = ["a dock", "--negative", "blur", "--tiled-refine", str(root.id), "--seed", "80", "--batch", "2",
             "--runs", "2", "--aspect", "1:1", "--mp", "4"]
    assert generate.main([*flags, "--dry-run"]) == 0
    assert "four tiles, 36 denoising steps per image" in capsys.readouterr().out
    assert cli.loads == 0 and not cli.requests
    assert generate.main([*flags, "-q"]) == 0
    assert cli.seeds == [80, 81, 82, 83]
    assert all(req.tiled_refine == spec and req.num_images == 1 for req in cli.requests)
    assert generate.main([*flags, "--resume", "-q"]) == 0 and len(cli.requests) == 4

    result = store.list_images()[0][0]
    png = store.settings.outputs_dir / result.filename
    assert generate.main(["--from-image", str(png), "-q"]) == 0
    assert cli.seeds[-1] == result.seed and cli.requests[-1].tiled_refine == spec
    assert generate.main(["--from-image", str(png), "--negative", "", "-q"]) == 0
    changed_result = store.list_images()[0][0]
    assert changed_result.negative_prompt is None and changed_result.true_cfg is None
    with closing(Image.new("RGB", tiled.ROOT_SIZE, "gold")) as changed:
        changed.save(store.settings.outputs_dir / root.filename)
    assert generate.main(["--from-image", str(png), "-q"]) == 2
    assert "provenance" in capsys.readouterr().err
    assert len(cli.requests) == 6


def test_via_server_carries_the_same_tiled_spec_and_revision_without_local_project(store, monkeypatch):
    _, spec = _source(store)
    png = store.settings.outputs_dir / _save(store, _request().model_copy(update={"tiled_refine": spec})).filename
    requests = []

    def server(_base, path, **kwargs):
        requests.append((path, kwargs))
        if path == "/api/status":
            return {"model_id": tiled.MODEL_ID, "quantization": tiled.QUANTIZATION,
                    "offload": tiled.OFFLOAD, "model_revision": 6}
        if path == "/api/generate":
            return {"job_id": "tiled-test"}
        assert path == "/api/jobs/tiled-test"
        return {"status": "done"}

    monkeypatch.setattr(generate_remote, "_server_json", server)
    monkeypatch.setattr(generate, "_project", lambda: pytest.fail("--via-server must not read its local gallery"))
    assert generate.main(["--from-image", str(png), "--via-server", "-q"]) == 0
    _, submission = requests[1]
    assert submission["body"]["tiled_refine"] == spec.model_dump(mode="json")
    assert submission["headers"]["X-AuraDream-Model-Revision"] == "6"


def test_tiled_source_availability_and_search_keep_pass_identity(store):
    root, spec = _source(store)
    request = _request().model_copy(update={"tiled_refine": spec})
    result = _save(store, request)
    unresolved = _save(store, request, run="unresolved-source")
    with store._session() as conn:
        proof = tiled.stored_json(spec.model_copy(update={"source": None}))
        conn.execute("UPDATE images SET tiled_refine = ? WHERE id = ?", (proof, unresolved.id))
        conn.commit()
    refine_search = f"refine:{root.id}"

    assert [row.id for row in store.list_images(search=refine_search)[0]] == [result.id]
    assert [row.id for row in store.list_images(search="refine:none")[0]] == [root.id]
    assert store.get_image(unresolved.id).tiled_refine_missing
    assert not store.get_image(result.id).tiled_refine_missing

    store.bin_images([root.id])
    [pass_row] = store.list_images(search=refine_search)[0]
    assert pass_row.tiled_refine_missing and pass_row.tiled_refine_binned
    store.restore_images([root.id])
    assert not store.get_image(result.id).tiled_refine_missing

    # Identical RGB inside a different format still fails the explicit
    # proof. A gallery page's cheap availability hint need not decode it.
    with closing(Image.new("RGB", tiled.ROOT_SIZE, "navy")) as image:
        image.save(store.settings.outputs_dir / root.filename, format="BMP")
    [pass_row] = store.list_images(search=refine_search)[0]
    assert not pass_row.tiled_refine_missing
    changed = store.get_image(result.id)
    assert changed.tiled_refine_missing and not changed.tiled_refine_binned
    assert changed.tiled_refine == spec
