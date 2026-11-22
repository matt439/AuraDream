"""ControlNet preview, queue and portable gallery source proof without weights."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundles, controlnet_admission, gallery_delete, model_selection, reindex
from backend.backends import controlnet_runner
from backend.controlnet_contract import SIZE, WEIGHTS, ControlNetPreview, ControlNetSpec
from backend.schemas import GenerateRequest
from backend.tiled_refine_contract import MODEL_ID, OFFLOAD, QUANTIZATION
from tests.api_helpers import stream_job
from tests.bundle_helpers import png_metadata, save_source, use_destination


def save_guide(store, gray=False):
    with Image.new("L" if gray else "RGB", SIZE, 73 if gray else (32, 8, 128)) as guide:
        return store.save_image(guide, GenerateRequest(prompt="Guide", width=1024, height=1024),
                                3, 0, MODEL_ID, quantization=QUANTIZATION)


def generate_payload():
    return {
        "prompt": "the final image", "width": 1024, "height": 1024, "steps": 28,
        "guidance_scale": 5, "num_images": 1, "seed": 42,
    }


def save_conditioned(store, spec):
    request = GenerateRequest(**generate_payload(), spatial_conditioning=spec)
    with Image.new("RGB", SIZE, (12, 34, 56)) as image:
        return store.save_image(image, request, 42, 1, MODEL_ID, quantization=QUANTIZATION)


@pytest.mark.parametrize("guide_state", ["proved", "wrong_pixels", "unresolved"])
def test_starred_result_spares_only_a_proved_guide_and_bin_warns_when_it_is_used(store, guide_state):
    with Image.new("RGB", SIZE, "gold") as pixels:
        guide = store.save_imported_source(pixels, "pose-map.png")
    spec = controlnet_admission.preview(ControlNetPreview(source=guide.id, mode="pose"))
    result = save_conditioned(store, spec.model_copy(update={"source": None}) if guide_state == "unresolved" else spec)
    store.set_favorite(result.id, True)
    if guide_state == "wrong_pixels":
        with Image.new("RGB", SIZE, "navy") as pixels:
            pixels.save(store.settings.outputs_dir / guide.filename)

    proved = guide_state == "proved"
    kept = [guide.id] if proved else []
    options = {"model_id": MODEL_ID, "include_unscoped_sources": True}
    preview = gallery_delete.matching_preview(None, live_only=True, **options)
    assert preview["sources"] == kept
    assert preview["sources_kept"] == int(proved)
    moved = gallery_delete.delete_matching(None, keep_favorites=True, to_bin=True, **options)
    assert moved["ids"] == ([] if proved else [guide.id])
    if proved:
        assert gallery_delete.clear_history(keep_favorites=True)["records"] == 0
        gallery_delete.bin_images([guide.id])
    assert gallery_delete.bin_summary()["sources_in_use"] == kept


@pytest.mark.parametrize("mode", ["pose", "depth"])
def test_controlnet_preview_generate_and_refuse_a_changed_guide(client, fake, store, monkeypatch, mode):
    model_selection.select(MODEL_ID, quantization=QUANTIZATION, offload=OFFLOAD)
    guide = save_guide(store, gray=mode == "depth")
    draft = {"source": guide.id, "mode": mode, "strength": 0.8}
    route = "/api/controlnet/preview"
    assert client.post(route, json=draft).status_code == 422
    monkeypatch.setattr(fake, "controlnet_qualified", lambda candidate: candidate == mode)
    status = client.get("/api/status").json()
    assert status["controlnet"][mode]["repo_id"] == WEIGHTS[mode].repo_id
    assert status["controlnet"][mode]["reason"] is None
    with monkeypatch.context() as held:
        reason = "Reload the pinned Animagine XL 4.0 checkpoint before using ControlNet."
        held.setattr(fake, "controlnet_owner_refusal", lambda: reason)
        assert client.get("/api/status").json()["controlnet"][mode]["reason"] == reason
        assert client.post(route, json=draft).status_code == 422
    assert client.post(route, json=draft, headers={
        "X-AuraDream-Model-Revision": str(status["model_revision"] + 1),
    }).status_code == 409
    response = client.post(route, json=draft)
    assert response.status_code == 200, response.text
    proof = response.json()["spatial_conditioning"]
    assert (proof["guide"]["filename"], proof["weight_revision"]) == (guide.filename, WEIGHTS[mode].revision)

    payload = {**generate_payload(), "spatial_conditioning": proof}
    refused = client.post("/api/generate", json={**payload, "num_images": 2})
    assert refused.status_code == 422 and not fake.requests
    queued = client.post("/api/generate", json=payload)
    assert queued.status_code == 200, queued.text
    final = stream_job(client, queued.json()["job_id"])[-1]
    assert final["status"] == "done"
    assert final["images"][0]["spatial_conditioning"] == proof
    record = store.get_image(final["images"][0]["id"])
    assert record.spatial_conditioning == ControlNetSpec.model_validate(proof)
    assert png_metadata(store.settings.outputs_dir / record.filename)["spatial_conditioning"] == proof

    with Image.new("RGB", SIZE, "brown") as changed:
        changed.save(store.settings.outputs_dir / guide.filename)
    refused = client.post("/api/generate", json=payload)
    assert refused.status_code == 422 and "gallery pixels" in refused.text
    assert len(fake.requests) == 1


@pytest.mark.parametrize("include_guide", [False, True])
def test_controlnet_bundle_remaps_only_proven_guide_ids(store, tmp_path, monkeypatch, include_guide):
    guide = save_guide(store)
    proof = controlnet_admission.preview(ControlNetPreview(source=guide.id, mode="pose"))
    result = save_conditioned(store, proof)
    ids = [guide.id, result.id] if include_guide else [result.id]
    packed = bundles.export_bundle(keep_gallery=True, ids=ids)
    assert packed["links_outside"] == int(not include_guide)
    archive = Path(packed["path"])
    with ZipFile(archive) as file:
        assert json.loads(file.read("manifest.json"))["version"] == 4
    destination = use_destination(store, monkeypatch, tmp_path)
    assert save_source("unrelated local image").id == guide.id
    unpacked = bundles.import_bundle(archive)
    assert unpacked["unresolved_references"] == int(not include_guide)
    by_name = {row.filename: row for row in store.list_images()[0]}
    bound = by_name[result.filename].spatial_conditioning
    mapped = by_name[guide.filename].id if include_guide else None
    assert bound == proof.model_copy(update={"source": mapped})
    assert png_metadata(destination.outputs_dir / result.filename)["spatial_conditioning"] == bound.model_dump(mode="json")


@pytest.mark.parametrize("use_db", [False, True])
def test_controlnet_reindex_follows_full_rgb_proof(store, tmp_path, monkeypatch, use_db):
    source = store.settings
    guide = save_guide(store)
    proof = controlnet_admission.preview(ControlNetPreview(source=guide.id, mode="depth"))
    result = save_conditioned(store, proof)
    destination = replace(source, db_path=tmp_path / "new.sqlite3", outputs_dir=tmp_path / "recovered")
    monkeypatch.setattr(store, "settings", destination)
    monkeypatch.setattr(reindex, "settings", destination)
    store.init_db()
    assert save_source("unrelated local image").id == guide.id

    rebuilt = reindex.reindex(source.outputs_dir, database=source.db_path if use_db else None)
    assert rebuilt["unresolved_controlnet_guides"] == 0
    by_name = {row.filename: row for row in store.list_images()[0]}
    mapped = by_name[guide.filename].id
    bound = by_name[result.filename].spatial_conditioning
    assert mapped != guide.id and bound == proof.model_copy(update={"source": mapped})
    assert png_metadata(destination.outputs_dir / result.filename)["spatial_conditioning"] == bound.model_dump(mode="json")


def test_controlnet_cleanup_drops_auxiliary_weights_before_base_offload():
    events = []
    hook = SimpleNamespace(remove=lambda: events.append("remove"))
    controlnet = SimpleNamespace(to=lambda device: events.append(device))
    owner = SimpleNamespace(maybe_free_model_hooks=lambda: events.append("base offload"))
    controlnet_runner._cleanup(owner, controlnet, hook, None)
    assert events == ["remove", "meta", "base offload"]


def test_controlnet_admission_refuses_low_windows_commit_with_available_ram(monkeypatch):
    gib = 1024**3
    cuda = SimpleNamespace(empty_cache=lambda: None, mem_get_info=lambda _: (12 * gib, 16 * gib))
    torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setattr(controlnet_runner, "windows_memory_status", lambda: (64 * gib, 24 * gib, 12 * gib))
    with pytest.raises(ValueError, match="13 GiB of available Windows commit"):
        controlnet_runner._require_headroom(torch)
    monkeypatch.setattr(controlnet_runner, "windows_memory_status", lambda: (64 * gib, 24 * gib, 13 * gib))
    controlnet_runner._require_headroom(torch)
