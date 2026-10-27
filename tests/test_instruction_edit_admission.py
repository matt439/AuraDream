"""Unavailable instruction edits keep identity without starting model work."""
from __future__ import annotations

import json
import time

import pytest
from PIL import Image

from backend import generate, grid, instruction_edit_admission, main, model_selection
from backend.instruction_edit import KONTEXT_V1, StoredInstructionEdit, descriptor_from_image
from backend.metadata import PNG_METADATA_KEY
from backend.schemas import GenerateRequest


def source(store, *, width=1024, height=1024):
    image = Image.new("RGB", (width, height), (11, 32, 47))
    rec = store.save_image(
        image, GenerateRequest(prompt="original", width=width, height=height, mu=None),
        seed=3, duration_ms=1, model_id="source/checkpoint", quantization="fp8",
    )
    proof = descriptor_from_image(rec.filename, image)
    return rec, StoredInstructionEdit(source=rec.id, recipe="kontext-v1", edit_source=proof)


def request(bound, **changes):
    return GenerateRequest(**{
        "prompt": "make it night", "edit": bound, "width": 512, "height": 512, "seed": 7,
        **changes,
    })


def child(store, bound, *, run_id="instruction-edit-run"):
    req = request(bound)
    rec = store.save_image(
        Image.new("RGB", (512, 512), (50, 30, 10)), req,
        seed=7, duration_ms=1, model_id=KONTEXT_V1.model_id, quantization="fp8",
        run_id=run_id, batch=1,
    )
    return rec


def select_editor(monkeypatch):
    # An explicit synthetic selection exercises a hand-entered startup id;
    # this does not make a catalogue option, download or load the editor.
    monkeypatch.setattr(model_selection, "_selection", (KONTEXT_V1.model_id, 1))


def test_binding_rechecks_copied_mapping_and_freezes_actual_source(store):
    original, bound = source(store)
    raw = request(bound).model_copy(update={"edit": {"source": original.id, "recipe": "kontext-v1"}})
    prepared = instruction_edit_admission.bind_request(raw, KONTEXT_V1.model_id)
    assert prepared.edit == bound and raw.edit == {"source": original.id, "recipe": "kontext-v1"}
    stored = request(bound).model_copy(update={"edit": bound.model_dump()})
    assert instruction_edit_admission.bind_request(stored, KONTEXT_V1.model_id).edit == bound


def test_replay_binding_refuses_replaced_pixels_and_forged_proof(store):
    original, bound = source(store)
    forged = bound.model_copy(update={"edit_source": bound.edit_source.model_copy(update={"hash": "cd" * 32})})
    with pytest.raises(ValueError, match="recorded pixels"):
        instruction_edit_admission.bind_request(request(forged), KONTEXT_V1.model_id)
    Image.new("RGB", (1024, 1024), (0, 1, 2)).save(store.settings.outputs_dir / original.filename)
    with pytest.raises(ValueError, match="recorded pixels"):
        instruction_edit_admission.bind_request(request(bound), KONTEXT_V1.model_id)


@pytest.mark.parametrize("mode", ["missing", "binned", "unresolved"])
def test_binding_never_substitutes_an_unusable_source(store, mode):
    original, bound = source(store)
    if mode == "binned":
        store.bin_images([original.id])
    elif mode == "missing":
        bound = bound.model_copy(update={"source": original.id + 999})
    else:
        bound = bound.model_copy(update={"source": None})
    with pytest.raises(ValueError):
        instruction_edit_admission.bind_source(bound)


def test_preview_is_cold_source_only_and_names_exact_geometry(client, fake, store):
    original, bound = source(store)
    response = client.post("/api/instruction-edit/preview", json={"source": original.id, "recipe": "kontext-v1"})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["edit"] == bound.model_dump(mode="json")
    assert (answer["width"], answer["height"], answer["available"]) == (512, 512, False)
    assert answer["model_id"] == KONTEXT_V1.model_id and "not available" in answer["reason"]
    assert fake.loads == 0 and fake.requests == [] and main.jobs.active() == []
    assert store.list_images()[1] == 1


def test_preview_rejects_unknown_fields_and_unsupported_aspect_without_model_work(client, fake, store):
    original, _ = source(store, width=512, height=256)
    for payload in (
        {"source": original.id, "recipe": "kontext-v1"},
        {"source": original.id, "recipe": "kontext-v1", "path": "outside.png"},
        {"source": True, "recipe": "kontext-v1"},
    ):
        response = client.post("/api/instruction-edit/preview", json=payload)
        assert response.status_code == 422, response.text
    assert fake.loads == 0 and fake.requests == []


def test_preview_is_held_by_stale_model_revision_before_source_reads(client, fake, store, monkeypatch):
    original, _ = source(store)

    def forbidden(*args, **kwargs):
        pytest.fail("a stale instruction-edit preview reached source decoding")

    monkeypatch.setattr(instruction_edit_admission, "bind_source", forbidden)
    response = client.post("/api/instruction-edit/preview", json={
        "source": original.id, "recipe": "kontext-v1",
    }, headers={"X-Local-Diffusion-Model-Revision": "999999"})
    assert response.status_code == 409, response.text
    assert fake.loads == 0 and fake.requests == []


@pytest.mark.parametrize("route", ["/api/generate", "/api/grid", "/api/grid/preview"])
def test_public_submission_refuses_edit_before_load_or_queue(client, fake, store, monkeypatch, route):
    _, bound = source(store)
    select_editor(monkeypatch)
    response = client.post(route, json=request(bound).model_dump(mode="json"))
    assert response.status_code == 422, response.text
    assert "instruction edit" in response.text.lower()
    assert fake.loads == 0 and fake.requests == [] and main.jobs.active() == []
    assert store.list_images()[1] == 1


def test_edit_only_selection_cannot_generate_without_a_source_or_load(client, fake, monkeypatch):
    select_editor(monkeypatch)
    for route, payload in (
        ("/api/generate", {"prompt": "make it night", "width": 512, "height": 512}),
        ("/api/model/load", {}),
    ):
        response = client.post(route, json=payload)
        assert response.status_code == 422, response.text
        assert "not available" in response.text
    assert fake.loads == 0 and fake.requests == [] and main.jobs.active() == []


def test_gallery_replay_actions_never_turn_edit_instruction_into_a_prompt(client, fake, store, monkeypatch):
    _, bound = source(store)
    edited = child(store, bound)
    select_editor(monkeypatch)
    response = client.post("/api/rerender", json={"ids": [edited.id], "width": 512, "height": 512})
    assert response.status_code == 422 and "not available" in response.text, response.text
    for route, payload, rows in (
        ("/api/more-seeds/preview", {"ids": [edited.id], "count": 2}, "results"),
        ("/api/swap-tags/preview", {"ids": [edited.id], "swaps": [{"find": "night", "replace": "day"}]}, "images"),
    ):
        response = client.post(route, json=payload)
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["queued"] == 0 and answer[rows][0]["status"] == "invalid", answer
        assert "instruction edit" in answer[rows][0]["detail"].lower()
    assert fake.loads == 0 and fake.requests == [] and main.jobs.active() == []
    assert store.get_image(edited.id).edit == bound


def test_finish_keeps_retained_edit_inspectable_and_unavailable(client, fake, store, monkeypatch):
    _, bound = source(store)
    select_editor(monkeypatch)
    store.retain_run_job({
        "job_id": "retained-edit", "run_id": "retained-edit-run", "created_at": time.time(),
        "origin": "generate", "model_id": KONTEXT_V1.model_id, "quantization": "fp8",
        "width": 512, "height": 512, "batch": 1, "comparison": False, "metadata": {},
    }, [grid.Cell(request(bound), seed=7)])
    response = client.post("/api/runs/retained-edit-run/finish/preview", json={})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert not answer["available"] and "not available" in answer["reason"]
    assert (answer["total"], answer["cells"], answer["images"]) == (1, 0, 0)
    response = client.get("/api/runs/retained-edit-run")
    assert response.status_code == 200 and response.json()["sizes"] == [[512, 512]], response.text
    response = client.post("/api/runs/retained-edit-run/finish", json={})
    assert response.status_code == 422 and "not available" in response.text, response.text
    assert fake.loads == 0 and fake.requests == [] and main.jobs.active() == []
    assert store.retained_run_jobs("retained-edit-run")[0]["cells"][0].request.edit == bound


def test_png_and_cli_refuse_edit_metadata_without_environment_or_model_work(client, fake, store, monkeypatch, capsys):
    _, bound = source(store)
    edited = child(store, bound)
    path = store.settings.outputs_dir / edited.filename
    with Image.open(path) as image:
        chunks = {key: value for key, value in image.info.items() if isinstance(value, str)}
    assert json.loads(chunks[PNG_METADATA_KEY])["edit"] == bound.model_dump(mode="json")
    response = client.post("/api/png/settings", json={"name": edited.filename, "chunks": chunks})
    assert response.status_code == 422 and "gallery image" in response.text, response.text

    def forbidden(*args, **kwargs):
        pytest.fail("instruction-edit PNG reached CLI environment/model setup")

    monkeypatch.setattr(generate, "apply_env", forbidden)
    monkeypatch.setattr(generate, "_project", forbidden)
    with pytest.raises(SystemExit) as caught:
        generate.main(["--from-image", str(path), "--dry-run", "replacement prompt"])
    assert caught.value.code == 2 and "gallery image" in capsys.readouterr().err
    assert fake.loads == 0 and fake.requests == []


def test_direct_fake_paths_share_unavailable_guard_even_without_cfg_threshold(fake, store):
    _, bound = source(store)
    fake.cfg_threshold = None
    req = request(bound)
    for operation in (lambda: fake.applied(req), lambda: fake.generate(req, 7)):
        with pytest.raises(ValueError, match="not available"):
            operation()
    assert fake.loads == 0 and fake.requests == []


@pytest.mark.parametrize("method", ["is_symlink", "is_junction"])
def test_preview_refuses_linked_source_files_before_decode(client, store, monkeypatch, method):
    original, _ = source(store)
    path = store.settings.outputs_dir / original.filename
    monkeypatch.setattr(type(path), method, lambda self: self == path)
    response = client.post("/api/instruction-edit/preview", json={"source": original.id, "recipe": "kontext-v1"})
    assert response.status_code == 422 and "linked file" in response.text


def test_preview_turns_image_bomb_refusal_into_a_worded_error(client, store, monkeypatch):
    original, _ = source(store)

    def refuse(*args, **kwargs):
        raise Image.DecompressionBombError("synthetic oversized source")

    monkeypatch.setattr(Image, "open", refuse)
    response = client.post("/api/instruction-edit/preview", json={"source": original.id, "recipe": "kontext-v1"})
    assert response.status_code == 422 and "cannot be decoded" in response.text
