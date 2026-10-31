"""The checkpoint catalogue over HTTP: GET /api/models and the status it
shares, describing unloaded choices, draft loader options and this
gallery's timing per model without changing the selected checkpoint."""
from __future__ import annotations

import pytest
from PIL import Image

from backend import main
from backend.schemas import GenerateRequest
from tests.model_switch_helpers import (  # noqa: F401  (fixtures by name)
    ANIMAGINE,
    PAYLOAD,
    fresh_selection,
)


def test_model_inventory_uses_catalogue_labels_and_current_loading_options(client, fake):
    reply = client.get("/api/models")
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["models"] == [
        {"model_id": item.repo, "label": item.name} for item in main.CHECKPOINTS
    ]
    status = client.get("/api/status").json()
    assert (body["quantization"], body["offload"]) == (
        status["quantization"], status["offload"],
    )
    assert fake.loads == 0  # Listing the choices is cold.


def test_catalogue_details_describe_unloaded_choices_without_changing_model(client, fake):
    initial = client.get("/api/status").json()
    body = client.get("/api/models?quantization=int8").json()
    assert set(body["details"]) == {entry["model_id"] for entry in body["models"]}
    anime = body["details"][ANIMAGINE]
    assert (anime["style"], anime["refine"], anime["megapixels"]) == ("Anime", "Yes", "1 MP")
    assert anime["megapixels_label"] == "Max recommended"
    assert "Author-recommended" in anime["megapixels_note"]
    assert anime["favourite_id"] == ANIMAGINE
    assert anime["url"] == f"https://huggingface.co/{ANIMAGINE}"
    flux = body["details"]["black-forest-labs/FLUX.1-Krea-dev"]
    assert (flux["megapixels_label"], flux["megapixels"]) == ("Max recommended", "1.64 MP")
    assert client.get("/api/status").json()["model_revision"] == initial["model_revision"]
    assert fake.loads == 0


def test_catalogue_speed_uses_draft_loading_options_without_applying_them(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    calls = []

    def details(model_id, *, quantization, offload, edit_reason):
        calls.append((model_id, quantization, offload))
        return {"speed_label": "Unmeasured"}

    monkeypatch.setattr(main, "details_for", details)
    reply = client.get("/api/models?quantization=int8&offload=sequential")
    assert reply.status_code == 200
    assert calls and all(quant == "int8" and offload == "sequential" for _, quant, offload in calls)
    current = client.get("/api/status").json()
    assert (current["quantization"], current["offload"], current["model_revision"]) == (
        initial["quantization"], initial["offload"], initial["model_revision"],
    )
    assert fake.loads == 0


def test_catalogue_adds_this_gallerys_timing_per_model_without_changing_the_speed(client, fake, store):
    quantization = client.get("/api/status").json()["quantization"]
    before = client.get("/api/models").json()["details"]
    assert all(detail["gallery_timing"] is None for detail in before.values())
    model_id = client.get("/api/status").json()["model_id"]
    for duration in (4_000, 9_000, 5_000):
        store.save_image(
            Image.new("RGB", (256, 256)), GenerateRequest(**PAYLOAD), seed=1,
            duration_ms=duration, model_id=model_id, quantization=quantization,
        )
    after = client.get("/api/models").json()["details"]
    assert set(after) == set(before)
    assert after[model_id]["gallery_timing"] == {
        "median_ms": 5_000, "width": 256, "height": 256,
        "steps": GenerateRequest(**PAYLOAD).steps, "images": 3,
    }
    assert all(after[other]["gallery_timing"] is None for other in after if other != model_id)
    # History never sets the category or its words.
    for key in ("speed_label", "speed_summary", "speed_basis", "speed_note"):
        assert {m: d.get(key) for m, d in after.items()} == {m: d.get(key) for m, d in before.items()}
    # Another quantization's catalogue reads none of it.
    other = next(q for q in ("int8", "none", "fp8") if q != quantization)
    drafted = client.get(f"/api/models?quantization={other}").json()["details"]
    assert drafted[model_id]["gallery_timing"] is None
    assert fake.loads == 0


def test_a_busy_gallery_leaves_the_timing_out_of_the_catalogue(client, fake, monkeypatch):
    def busy(quantization):
        raise main.gallery_lock.GalleryBusy("busy")

    monkeypatch.setattr(main.gallery_reports, "gallery_timing", busy)
    reply = client.get("/api/models")
    assert reply.status_code == 200, reply.text
    assert all(detail["gallery_timing"] is None for detail in reply.json()["details"].values())


def test_a_reserved_gallery_leaves_the_timing_out_without_stalling_the_catalogue(client, fake, store):
    # The real reservation, held by another process as a bundle or a clear
    # holds it: the catalogue answers at once, without the line.
    import time

    from tests.bundle_helpers import _holding_child

    status = client.get("/api/status").json()
    store.save_image(
        Image.new("RGB", (256, 256)), GenerateRequest(**PAYLOAD), seed=1,
        duration_ms=4_000, model_id=status["model_id"], quantization=status["quantization"],
    )
    child = _holding_child(store.settings.db_path)
    try:
        started = time.monotonic()
        reply = client.get("/api/models")
        assert time.monotonic() - started < main.gallery_reports.TIMING_WAIT + 5
    finally:
        child.stdin.close()
        child.wait(timeout=15)
    assert reply.status_code == 200, reply.text
    assert all(detail["gallery_timing"] is None for detail in reply.json()["details"].values())
    assert client.get("/api/models").json()["details"][status["model_id"]]["gallery_timing"]["images"] == 1


def test_the_editors_row_carries_the_backends_answer_for_the_draft_loader(client, fake, monkeypatch):
    # The listed editor's row gets this instance's own answer for the loader
    # being chosen; every other row keeps the closed gate's words. Nothing
    # is selected or loaded.
    from backend.instruction_edit import INSTRUCTION_EDIT_UNAVAILABLE, KONTEXT_V1

    seen = {}

    def details(model_id, *, quantization, offload, edit_reason):
        seen[model_id] = edit_reason
        return {}

    monkeypatch.setattr(main, "details_for", details)
    initial = client.get("/api/status").json()

    def answer(query=""):
        seen.clear()
        assert client.get(f"/api/models{query}").status_code == 200
        others = {reason for model_id, reason in seen.items() if model_id != KONTEXT_V1.model_id}
        assert others == {INSTRUCTION_EDIT_UNAVAILABLE}
        return seen[KONTEXT_V1.model_id]

    assert answer() == INSTRUCTION_EDIT_UNAVAILABLE
    fake.edit_qualified = True
    assert answer("?quantization=fp8&offload=model") is None
    assert "sequential offload" in answer("?quantization=fp8&offload=sequential")
    assert "int8" in answer("?quantization=int8&offload=model")
    # A draft recipe is the loader being chosen too, and is held by name.
    assert "regional-v1 compilation" in answer("?quantization=fp8&offload=model&compilation=regional-v1")
    assert "cudnn-v1 attention" in answer("?quantization=fp8&offload=model&attention=cudnn-v1")
    assert answer("?quantization=fp8&offload=model&compilation=eager&attention=native") is None
    current = client.get("/api/status").json()
    assert (current["model_id"], current["model_revision"]) == (initial["model_id"], initial["model_revision"])
    assert fake.loads == 0


def test_the_editors_row_holds_its_unmeasured_loader_options_by_name(client, fake):
    # The picker holds each quantization and offload the editor never
    # measured, in the envelope's words for that choice alone, whatever the
    # draft or the gate; every other row has none. Nothing selected or loaded.
    from backend.instruction_edit import INSTRUCTION_EDIT_LOADER_UNQUALIFIED, KONTEXT_V1

    initial = client.get("/api/status").json()
    for query, qualified in [("", False), ("?quantization=int8&offload=none", False), ("?quantization=fp8", True)]:
        fake.edit_qualified = qualified
        body = client.get(f"/api/models{query}").json()
        quantizations = {o["value"] for o in body["quantizations"]}
        offloads = {o["value"] for o in body["offloads"]}
        holds = body["details"][KONTEXT_V1.model_id]["loader_holds"]
        assert set(holds["quantization"]) == quantizations - {"fp8"}
        assert set(holds["offload"]) == offloads - {"model"}
        assert holds["quantization"]["int8"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="int8 quantization")
        assert holds["quantization"]["none"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="BF16 (no quantization)")
        assert holds["offload"]["sequential"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="sequential offload")
        assert holds["offload"]["none"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="no offload")
        others = {model_id: info["loader_holds"] for model_id, info in body["details"].items() if model_id != KONTEXT_V1.model_id}
        assert others and all(value == {} for value in others.values())
    current = client.get("/api/status").json()
    assert (current["model_id"], current["model_revision"]) == (initial["model_id"], initial["model_revision"])
    assert fake.loads == 0


@pytest.mark.parametrize("query", ["quantization=garbage", "format=garbage", "offload=unknown"])
def test_catalogue_refuses_unknown_loading_options(client, fake, monkeypatch, query):
    calls = []
    monkeypatch.setattr(main, "details_for", lambda *args, **kwargs: calls.append(args))
    assert client.get(f"/api/models?{query}").status_code == 422
    assert not calls and fake.loads == 0


def test_status_and_catalogue_report_this_starts_overrides(client, monkeypatch):
    assert client.get("/api/status").json()["startup_overrides"] == {}
    overrides = {"model_id": "--model", "quantization": "AURADREAM_QUANT"}
    monkeypatch.setattr(main, "_startup_overrides", overrides)
    assert client.get("/api/status").json()["startup_overrides"] == overrides
    assert client.get("/api/models").json()["startup_overrides"] == overrides
