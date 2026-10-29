"""Runtime checkpoint changes keep one loader, one queue and one gallery."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import asyncio
import dataclasses
import json
import threading
import time
from types import SimpleNamespace

import pytest
import httpx
from PIL import Image

from backend import main, model_selection, startup_preferences, submission
from backend.model_defaults import defaults_for, guide_for
from backend.schemas import PromptGuide
from tests.api_helpers import ORIGIN, run_one, stream_job, wait_for


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
TURBO = "krea/Krea-2-Turbo"
PAYLOAD = {"prompt": "a lighthouse", "width": 256, "height": 256}


@pytest.fixture(autouse=True)
def fresh_selection(monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    monkeypatch.setattr(main, "_switching", False)
    monkeypatch.setattr(main, "_model_requests", 0)
    monkeypatch.setattr(main, "_model_request_paths", {}, raising=False)
    monkeypatch.setattr(main, "_switch_draining", False, raising=False)
    # A refusal behind a held request comes after the drain; keep it short.
    monkeypatch.setattr(main, "SWITCH_DRAIN_SECONDS", 0.3, raising=False)


def _settled(client, model_id, state="ready"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = client.get("/api/status").json()
        if body["model_id"] == model_id and body["state"] == state and not body["switching"]:
            return body
        time.sleep(0.01)
    raise AssertionError(f"checkpoint did not settle as {model_id} / {state}: {body}")


def _switch(client, model_id):
    reply = client.post("/api/model/switch", json={"model_id": model_id})
    assert reply.status_code == 200, reply.text
    return _settled(client, model_id)


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


def test_save_without_load_and_preload_toggle_preserve_the_selected_settings(client, fake):
    initial = client.get("/api/status").json()
    reply = client.post("/api/model/switch", json={
        "model_id": ANIMAGINE, "quantization": "int8", "offload": "sequential",
        "load": False, "load_on_startup": True,
    })
    assert reply.status_code == 200, reply.text
    selected = _settled(client, ANIMAGINE, "unloaded")
    assert (selected["quantization"], selected["offload"], selected["load_on_startup"]) == (
        "int8", "sequential", True,
    )
    assert fake.loads == 0 and not fake.loaded
    assert main.settings.quantization == initial["quantization"]  # Frozen startup is unchanged.
    saved = startup_preferences.load(main.settings.db_path)
    assert saved == startup_preferences.StartupPreferences(ANIMAGINE, "int8", "sequential", True)
    assert client.post("/api/model/load").status_code == 200
    _settled(client, ANIMAGINE)
    loads = fake.loads
    reply = client.post("/api/model/switch", json={
        "model_id": ANIMAGINE, "load": False, "load_on_startup": False,
    })
    assert reply.status_code == 200
    assert reply.json()["state"] == "ready" and not reply.json()["load_on_startup"]
    assert reply.json()["model_revision"] == selected["model_revision"]
    assert fake.loads == loads and fake.loaded
    assert not startup_preferences.load(main.settings.db_path).load_on_startup


def test_same_checkpoint_new_quantization_reloads_and_records_actual_quantizer(client, fake, store):
    initial = client.get("/api/status").json()
    _, before = run_one(client, seed=41, steps=2)
    reply = client.post("/api/model/switch", json={
        "model_id": initial["model_id"], "quantization": "int8", "offload": "none",
    })
    assert reply.status_code == 200
    changed = _settled(client, initial["model_id"])
    assert changed["model_revision"] == initial["model_revision"] + 1
    assert (changed["quantization"], changed["offload"]) == ("int8", "none")
    _, after = run_one(client, seed=41, steps=2)
    first, last = before[-1]["images"][0], after[-1]["images"][0]
    assert first["run_id"] != last["run_id"]
    assert last["quantization"] == "int8"
    with Image.open(store.settings.outputs_dir / last["filename"]) as image:
        assert json.loads(image.info["auradream"])["quantization"] == "int8"
    assert fake.loads == 1


def test_failed_save_leaves_working_weights_and_previous_preferences(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    previous = startup_preferences.StartupPreferences(initial["model_id"], "fp8", "model", False)
    startup_preferences.save(main.settings.db_path, previous)

    def refuse(*args):
        raise OSError("disk is full")

    monkeypatch.setattr(startup_preferences, "save", refuse)
    reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE, "load_on_startup": True})
    assert reply.status_code == 500 and "disk is full" in reply.json()["detail"]
    current = client.get("/api/status").json()
    assert current["model_id"] == initial["model_id"]
    assert current["model_revision"] == initial["model_revision"]
    assert not current["switching"] and not current["load_on_startup"]
    assert fake.loaded and fake.loads == 0
    assert startup_preferences.load(main.settings.db_path) == previous


@pytest.mark.parametrize("changes", [
    {"quantization": "garbage"}, {"offload": "garbage"}, {"format": "garbage"},
])
def test_invalid_loader_settings_never_replace_saved_selection(client, fake, changes):
    reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE, **changes})
    assert reply.status_code == 422
    assert fake.loaded and fake.loads == 0
    assert startup_preferences.load(main.settings.db_path) is None


def test_switch_updates_defaults_saved_metadata_and_exact_gallery_scope(client, fake, store):
    initial = client.get("/api/status").json()
    old_job, old_events = run_one(client, seed=42, steps=2)
    old_record = old_events[-1]["images"][0]
    changed = _switch(client, ANIMAGINE)
    assert changed["started_at"] == initial["started_at"]
    assert changed["model_revision"] == initial["model_revision"] + 1
    assert changed["guide"] == PromptGuide.model_validate(guide_for(ANIMAGINE)).model_dump(mode="json")
    profile = defaults_for(ANIMAGINE)
    for key in ("steps", "guidance_scale", "mu", "refine_strength"):
        assert changed["defaults"][key] == profile[key]
    assert client.get("/api/gallery").json()["total"] == 0

    new_job, new_events = run_one(client, seed=42, mu=3, negative_prompt="blurry")
    assert new_events[-1]["status"] == "done"
    record = new_events[-1]["images"][0]
    assert (record["model_id"], record["steps"], record["guidance_scale"], record["mu"]) == (
        ANIMAGINE, profile["steps"], profile["guidance_scale"], None,
    )
    assert record["negative_prompt"] == "blurry"
    assert record["true_cfg"] == profile["guidance_scale"]
    assert record["quantization"] == initial["quantization"]
    assert record["run_id"] != old_record["run_id"]
    with Image.open(store.settings.outputs_dir / record["filename"]) as image:
        metadata = json.loads(image.info["auradream"])
    assert (metadata["model"], metadata["steps"], metadata["mu"]) == (
        ANIMAGINE, profile["steps"], None,
    )
    gallery = client.get("/api/gallery").json()
    assert (gallery["total"], gallery["other_models"]) == (1, 1)
    assert [item["id"] for item in gallery["items"]] == [record["id"]]
    assert {item["id"] for item in client.get("/api/gallery?all_models=true").json()["items"]} == {
        old_record["id"], record["id"],
    }
    assert [item["run_id"] for item in client.get("/api/runs").json()] == [record["run_id"]]
    assert client.get(f"/api/jobs/{old_job}").json()["images"][0]["model_id"] == initial["model_id"]
    assert client.get(f"/api/jobs/{new_job}").json()["images"][0]["model_id"] == ANIMAGINE

    restored = _switch(client, initial["model_id"])
    assert restored["model_revision"] == initial["model_revision"] + 2
    assert client.get("/api/gallery").json()["items"][0]["id"] == old_record["id"]


@pytest.mark.parametrize("model_id", ["not-a-catalogue/model", "", "../../a-local-checkpoint"])
def test_unknown_model_is_refused_before_unloading(client, fake, monkeypatch, model_id):
    initial = client.get("/api/status").json()
    unloads = []
    unload = fake.unload

    def watched_unload():
        unloads.append(True)
        unload()

    monkeypatch.setattr(fake, "unload", watched_unload)
    reply = client.post("/api/model/switch", json={"model_id": model_id})
    assert reply.status_code == 422, reply.text
    current = client.get("/api/status").json()
    assert (current["model_id"], current["model_revision"]) == (
        initial["model_id"], initial["model_revision"],
    )
    assert fake.loaded and not unloads and fake.loads == 0


def test_selecting_the_ready_model_is_a_noop(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    unloads = []
    monkeypatch.setattr(fake, "try_unload", lambda: unloads.append(True))
    reply = client.post("/api/model/switch", json={"model_id": initial["model_id"]})
    assert reply.status_code == 200, reply.text
    assert reply.json()["model_revision"] == initial["model_revision"]
    assert not unloads and fake.loads == 0 and fake.loaded


def test_switch_refuses_running_queued_and_paused_jobs(client, fake):
    initial = client.get("/api/status").json()
    fake.step_seconds = 0.05
    ahead = client.post("/api/generate", json={**PAYLOAD, "steps": 10, "runs": 5}).json()["job_id"]
    behind = client.post("/api/generate", json={**PAYLOAD, "steps": 2}).json()["job_id"]
    try:
        wait_for(client, ahead, lambda job: job["status"] == "running")
        assert client.get(f"/api/jobs/{behind}").json()["status"] == "queued"
        assert client.post("/api/model/switch", json={"model_id": ANIMAGINE}).status_code == 409
        client.post(f"/api/jobs/{ahead}/pause")
        wait_for(client, ahead, lambda job: job["paused"])
        assert client.post("/api/model/switch", json={"model_id": ANIMAGINE}).status_code == 409
        current = client.get("/api/status").json()
        assert current["model_id"] == initial["model_id"] and not current["switching"]
        assert fake.loaded and fake.loads == 0
    finally:
        client.post(f"/api/jobs/{behind}/cancel")
        client.post(f"/api/jobs/{ahead}/cancel")
        stream_job(client, ahead)


def test_switch_refuses_an_existing_load(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    fake.loaded = False
    started, release = threading.Event(), threading.Event()
    load = fake.load

    def waiting_load():
        started.set()
        assert release.wait(10)
        load()

    monkeypatch.setattr(fake, "load", waiting_load)
    try:
        assert client.post("/api/model/load").status_code == 200
        assert started.wait(5)
        reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
        assert reply.status_code == 409, reply.text
        assert client.get("/api/status").json()["model_id"] == initial["model_id"]
    finally:
        release.set()
    _settled(client, initial["model_id"])


@pytest.mark.parametrize("holder", ["critic", "expand"])
def test_switch_refuses_an_editor_action_holding_the_card(client, fake, monkeypatch, holder):
    initial = client.get("/api/status").json()
    started, release = threading.Event(), threading.Event()
    if holder == "critic":
        work = fake.critic.critique

        def waiting_work(*args):
            started.set()
            assert release.wait(10)
            return work(*args)

        monkeypatch.setattr(fake.critic, "critique", waiting_work)
        route = "/api/prompt/critique"
    else:
        fake.prompt_expander_supported = True
        work = fake.expand_prompt

        def waiting_work(*args):
            started.set()
            assert release.wait(10)
            return work(*args)

        monkeypatch.setattr(fake, "expand_prompt", waiting_work)
        route = "/api/prompt/expand"
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(client.post, route, json=PAYLOAD)
            assert started.wait(5)
            assert client.get("/api/status").json()["card_holder"] == holder
            reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
            assert reply.status_code == 409, reply.text
            assert client.get("/api/status").json()["model_id"] == initial["model_id"]
            assert fake.loaded
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200


def test_switch_load_failure_can_be_retried_and_another_model_selected(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    load = fake.load

    def failed_load():
        raise RuntimeError("the requested model could not load")

    monkeypatch.setattr(fake, "load", failed_load)
    reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
    assert reply.status_code == 200, reply.text
    failed = _settled(client, ANIMAGINE, "error")
    assert "the requested model could not load" in failed["detail"]
    assert failed["model_revision"] == initial["model_revision"] + 1
    assert not fake.loaded
    monkeypatch.setattr(fake, "load", load)
    assert client.post("/api/model/load").status_code == 200
    recovered = _settled(client, ANIMAGINE)
    assert recovered["detail"] is None
    assert recovered["model_revision"] == failed["model_revision"]
    _switch(client, initial["model_id"])


@pytest.mark.parametrize("start", ["error", "unloaded"])
def test_switch_and_load_of_the_selected_model_retries_its_load(client, fake, monkeypatch, start):
    load = fake.load
    if start == "error":
        def failed_load():
            raise RuntimeError("the requested model could not load")

        monkeypatch.setattr(fake, "load", failed_load)
        assert client.post("/api/model/switch", json={"model_id": ANIMAGINE}).status_code == 200
        before = _settled(client, ANIMAGINE, "error")
        monkeypatch.setattr(fake, "load", load)
    else:
        _switch(client, ANIMAGINE)
        assert client.post("/api/model/unload").status_code == 200
        before = _settled(client, ANIMAGINE, "unloaded")
    loads = fake.loads
    # What Switch & load sends for the checkpoint already selected.
    reply = client.post("/api/model/switch", json={
        "model_id": ANIMAGINE, "quantization": before["quantization"],
        "offload": before["offload"], "load": True,
    })
    assert reply.status_code == 200, reply.text
    after = _settled(client, ANIMAGINE)
    assert after["model_revision"] == before["model_revision"]
    assert after["detail"] is None and fake.loads == loads + 1
    assert startup_preferences.load(main.settings.db_path) == startup_preferences.StartupPreferences(
        ANIMAGINE, before["quantization"], before["offload"], before["load_on_startup"],
    )


@pytest.mark.parametrize("preference", [None, ANIMAGINE])
@pytest.mark.parametrize("helper", ["critic", "translator", "weights"])
def test_failed_teardown_keeps_the_running_model_loaded_selected_and_saved(
    client, fake, helper, preference,
):
    initial = client.get("/api/status").json()
    db_path = main.settings.db_path
    if preference is not None:
        # A choice saved earlier, beside the running one, survives intact.
        earlier = startup_preferences.StartupPreferences(preference, "int8", "sequential", False)
        startup_preferences.save(db_path, earlier)
    loads = fake.loads

    def failed_cleanup():
        if helper == "critic":
            raise main.CriticError("the critic process did not stop")
        raise RuntimeError(f"the {helper} did not let go")

    if helper == "weights":
        target, name, stand_in = fake, "try_unload", lambda: False
    else:
        target = fake.critic if helper == "critic" else fake.translator
        name, stand_in = "unload", failed_cleanup
    original = getattr(target, name)
    setattr(target, name, stand_in)
    try:
        reply = client.post("/api/model/switch", json={"model_id": TURBO, "load_on_startup": True})
        assert reply.status_code == 200, reply.text
        failed = _settled(client, initial["model_id"], initial["state"])
        reason = {
            "critic": "the prompt critic would not stop",
            "translator": "the translator did not let go",
            "weights": "the weights are still loading",
        }[helper]
        label = initial["model_label"] or initial["model_id"]
        assert failed["switch_error"].startswith(f"Could not switch to Krea 2 Turbo: {reason}")
        assert failed["switch_error"].endswith(f"{label} is still selected.")
        assert failed["detail"] is None
        assert failed["model_revision"] == initial["model_revision"]
        assert fake.loaded and fake.loads == loads
        # Neither the saved file nor this process's auto-load says the failed choice.
        saved = startup_preferences.load(db_path)
        assert saved == (None if preference is None else earlier)
        assert failed["load_on_startup"] == initial["load_on_startup"]
    finally:
        setattr(target, name, original)
    assert _switch(client, ANIMAGINE)["switch_error"] is None


def test_switch_reserves_teardown_and_loading_against_new_model_work(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    unloading, release_unload = threading.Event(), threading.Event()
    loading, release_load = threading.Event(), threading.Event()
    unload, load = fake.try_unload, fake.load

    def waiting_unload():
        unloading.set()
        assert release_unload.wait(10)
        return unload()

    def waiting_load():
        loading.set()
        assert release_load.wait(10)
        load()

    monkeypatch.setattr(fake, "try_unload", waiting_unload)
    monkeypatch.setattr(fake, "load", waiting_load)
    try:
        reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
        assert reply.status_code == 200, reply.text
        assert unloading.wait(5)
        status = client.get("/api/status").json()
        assert status["switching"] and status["model_id"] == initial["model_id"]
        for route in (
            "/api/generate", "/api/grid", "/api/grid/preview", "/api/rerender",
            "/api/swap-tags", "/api/refine/batch", "/api/more-seeds", "/api/model/load", "/api/model/unload",
            "/api/prompt/analyze",
            "/api/prompt/critique", "/api/prompt/expand",
        ):
            refused = client.post(route, json=PAYLOAD)
            assert refused.status_code == 409, (route, refused.text)
        assert client.post("/api/model/switch", json={"model_id": TURBO}).status_code == 409
        assert not fake.requests and fake.loaded
        release_unload.set()
        assert loading.wait(5)
        # Selected: the rest is an ordinary load, as after Apply & save then
        # Load. Cold work answers and a Generate queues behind the weights;
        # another switch and an unload are still refused.
        status = client.get("/api/status").json()
        assert (status["model_id"], status["switching"], status["state"]) == (
            ANIMAGINE, False, "loading",
        )
        analyzed = client.post("/api/prompt/analyze", json=PAYLOAD)
        assert analyzed.status_code == 200, analyzed.text
        assert client.get("/api/loras").status_code == 200
        queued = client.post("/api/generate", json={**PAYLOAD, "steps": 2})
        assert queued.status_code == 200, queued.text
        assert client.post("/api/model/switch", json={"model_id": TURBO}).status_code == 409
        assert client.post("/api/model/unload").status_code == 409
        assert not fake.requests
    finally:
        release_unload.set()
        release_load.set()
    _settled(client, ANIMAGINE)
    assert fake.loads == 1
    done = stream_job(client, queued.json()["job_id"])[-1]
    assert done["status"] == "done", done
    assert done["images"][0]["model_id"] == ANIMAGINE
    run_one(client, steps=2)


def test_switch_refuses_a_request_between_validation_and_job_submission(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    validating, release = threading.Event(), threading.Event()
    validate = submission.refuse_unusable_inputs

    async def waiting_validation(services, request):
        validating.set()
        assert await asyncio.to_thread(release.wait, 10)
        await validate(services, request)

    monkeypatch.setattr(submission, "refuse_unusable_inputs", waiting_validation)
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(client.post, "/api/generate", json=PAYLOAD)
            assert validating.wait(5)
            assert not main.jobs.active()  # Admission, before any queue entry.
            refused = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
            assert refused.status_code == 409, refused.text
            assert client.get("/api/status").json()["model_id"] == initial["model_id"]
        finally:
            release.set()
        accepted = pending.result(timeout=5)
    assert accepted.status_code == 200, accepted.text
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert done["images"][0]["model_id"] == initial["model_id"]
    _switch(client, ANIMAGINE)


def _held_measure(monkeypatch):
    measuring, release = threading.Event(), threading.Event()

    def measure(*args, **kwargs):
        measuring.set()
        assert release.wait(10)
        return [], [], None

    monkeypatch.setattr(main.prompt_encoding, "measure", measure)
    return measuring, release


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def test_switch_waits_out_a_model_request_already_in_flight(client, fake, monkeypatch):
    # The page's own cold reads (adapter list, token count, tag check) go out
    # as a load lands; a switch pressed then must not be refused for them.
    monkeypatch.setattr(main, "SWITCH_DRAIN_SECONDS", 5.0)
    measuring, release = _held_measure(monkeypatch)
    with ThreadPoolExecutor(2) as pool:
        try:
            pending = pool.submit(client.post, "/api/prompt/measure", json={"prompt": "a lighthouse"})
            assert measuring.wait(5)
            switching = pool.submit(client.post, "/api/model/switch", json={"model_id": ANIMAGINE})
            _until(lambda: main._switch_draining)
            # New readers are refused meanwhile, so the wait ends.
            refused = client.get("/api/loras")
            assert refused.status_code == 409, refused.text
            assert client.post("/api/model/switch", json={"model_id": TURBO}).status_code == 409
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
        reply = switching.result(timeout=5)
    assert reply.status_code == 200, reply.text
    _settled(client, ANIMAGINE)
    assert not main._switch_draining and main._model_requests == 0


def test_switch_names_a_model_request_that_outlasts_its_wait(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    measuring, release = _held_measure(monkeypatch)
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(client.post, "/api/prompt/measure", json={"prompt": "a lighthouse"})
            assert measuring.wait(5)
            refused = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
            assert refused.status_code == 409, refused.text
            assert "/api/prompt/measure" in refused.json()["detail"]
            assert client.get("/api/status").json()["model_id"] == initial["model_id"]
            # The refusal ends the wait: readers are admitted again.
            assert not main._switch_draining
            assert client.get("/api/loras").status_code == 200
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    _switch(client, ANIMAGINE)


@pytest.mark.parametrize("header", ["X-AuraDream-Model-Revision", "X-Local-Diffusion-Model-Revision"])
def test_old_browser_revision_cannot_submit_settings_to_the_new_model(client, fake, header):
    # The second name is a page loaded before the rename to AuraDream.
    initial = client.get("/api/status").json()
    current = _switch(client, ANIMAGINE)
    stale = client.post(
        "/api/generate", json=PAYLOAD, headers={header: str(initial["model_revision"])},
    )
    assert stale.status_code == 409, stale.text
    malformed = client.post("/api/generate", json=PAYLOAD, headers={header: "not-a-number"})
    assert malformed.status_code == 422 and not fake.requests and not main.jobs.active()
    accepted = client.post(
        "/api/generate", json=PAYLOAD, headers={header: str(current["model_revision"])},
    )
    assert accepted.status_code == 200, accepted.text
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert done["images"][0]["model_id"] == ANIMAGINE


@pytest.mark.parametrize("revision", ["stale", "not-a-number"])
def test_routes_that_read_no_checkpoint_ignore_the_revision_header(client, fake, revision):
    # The page sends its revision on every /api/ request; only the server's
    # model-dependent routes may refuse it.
    initial = client.get("/api/status").json()
    _switch(client, ANIMAGINE)
    value = str(initial["model_revision"]) if revision == "stale" else revision
    headers = {"X-AuraDream-Model-Revision": value}
    for path in (
        "/api/status", "/api/models", "/api/gallery", "/api/runs", "/api/jobs",
        "/api/gallery/bundles/status",
    ):
        reply = client.get(path, headers=headers)
        assert reply.status_code == 200, (path, reply.text)
    guarded = client.post("/api/model/load", headers=headers)
    assert guarded.status_code == (409 if revision == "stale" else 422), guarded.text


def test_status_retries_a_description_that_crossed_a_switch(client, fake, monkeypatch):
    initial = client.get("/api/status").json()
    describing, release = threading.Event(), threading.Event()
    describe = fake.describe
    first = True

    def waiting_description():
        nonlocal first
        captured = describe()
        if first:
            first = False
            describing.set()
            assert release.wait(10)
        return captured

    monkeypatch.setattr(fake, "describe", waiting_description)
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(client.get, "/api/status")
            assert describing.wait(5)
            changed = _switch(client, ANIMAGINE)
        finally:
            release.set()
        status = pending.result(timeout=5).json()
    assert status["model_id"] == ANIMAGINE
    assert status["model_revision"] == changed["model_revision"] == initial["model_revision"] + 1
    assert status["defaults"]["steps"] == defaults_for(ANIMAGINE)["steps"]
    assert not status["switching"]


def test_status_retries_a_newest_id_read_that_crossed_a_switch(client, fake, monkeypatch):
    # The gallery's newest id is read off the event loop too; a switch that
    # ran to its end meanwhile once came back as the old checkpoint and
    # revision beside `switching` false, which the page took for a switch
    # with no revision of its own to come.
    initial = client.get("/api/status").json()
    reading, release = threading.Event(), threading.Event()
    newest = main.storage.newest_image_id
    first = True

    def waiting_newest():
        nonlocal first
        captured = newest()
        if first:
            first = False
            reading.set()
            assert release.wait(10)
        return captured

    monkeypatch.setattr(main.storage, "newest_image_id", waiting_newest)
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(client.get, "/api/status")
            assert reading.wait(5)
            changed = _switch(client, ANIMAGINE)
        finally:
            release.set()
        status = pending.result(timeout=5).json()
    assert status["model_id"] == ANIMAGINE
    assert status["model_revision"] == changed["model_revision"] == initial["model_revision"] + 1
    assert status["defaults"]["steps"] == defaults_for(ANIMAGINE)["steps"]
    assert not status["switching"]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_nunchaku_inventory_and_refusal_keep_the_startup_runtime(client, fake, monkeypatch, platform):
    first = main.CHECKPOINTS[0]
    selected = first.lite_repo("int4")
    monkeypatch.setattr(main, "settings", dataclasses.replace(
        main.settings, model_id=selected, quantization="nunchaku",
    ))
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform=platform))
    choices = client.get("/api/models").json()
    assert choices["quantization"] == "nunchaku"
    if platform == "linux":
        assert choices["models"] == [
            {"model_id": item.lite_repo("int4"), "label": item.name}
            for item in main.CHECKPOINTS if item.lite
        ]
    else:
        assert choices["models"] == []
    unloads = []
    monkeypatch.setattr(fake, "try_unload", lambda: unloads.append(True))
    rejected = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
    assert rejected.status_code == 422, rejected.text
    assert not unloads and fake.loaded


def test_stale_gallery_confirmation_and_switch_do_not_act_on_the_new_selection(client):
    initial = client.get("/api/status").json()
    run_one(client, steps=2)
    assert client.get("/api/gallery/unstarred").json()["unstarred"] == 1
    current = _switch(client, ANIMAGINE)
    run_one(client, steps=2)
    # The old and new gallery happen to have the same count: count checking
    # alone would accept this confirmation against a different image.
    assert client.get("/api/gallery/unstarred").json()["unstarred"] == 1
    headers = {"X-AuraDream-Model-Revision": str(initial["model_revision"])}
    deleted = client.delete("/api/gallery/unstarred?expected=1", headers=headers)
    assert deleted.status_code == 409, deleted.text
    switched = client.post("/api/model/switch", json={"model_id": TURBO}, headers=headers)
    assert switched.status_code == 409, switched.text
    assert client.get("/api/gallery?all_models=true").json()["total"] == 2
    assert client.get("/api/status").json()["model_revision"] == current["model_revision"]


def test_switch_rechecks_revision_after_a_slow_request_body(fake):
    # Header validation runs before the body is parsed. A request whose
    # upload stalls there must not undo a selection another tab completes
    # while that original request is still arriving.
    async def scenario():
        body_started, release_body = asyncio.Event(), asyncio.Event()

        async def slow_body():
            yield b'{"model_id": "'
            body_started.set()
            await release_body.wait()
            yield (TURBO + '"}').encode()

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as client:
            initial = (await client.get("/api/status")).json()
            pending = asyncio.create_task(client.post(
                "/api/model/switch",
                content=slow_body(),
                headers={
                    "Content-Type": "application/json",
                    "X-AuraDream-Model-Revision": str(initial["model_revision"]),
                },
            ))
            try:
                await asyncio.wait_for(body_started.wait(), timeout=5)
                reply = await client.post("/api/model/switch", json={"model_id": ANIMAGINE})
                assert reply.status_code == 200, reply.text
                deadline = time.monotonic() + 5
                while True:
                    changed = (await client.get("/api/status")).json()
                    if changed["model_id"] == ANIMAGINE and not changed["switching"]:
                        break
                    assert time.monotonic() < deadline, "the other tab's switch did not finish"
                    await asyncio.sleep(0.01)
                assert changed["state"] == "ready"
                assert changed["model_revision"] == initial["model_revision"] + 1
            finally:
                release_body.set()
            refused = await asyncio.wait_for(pending, timeout=5)
            assert refused.status_code == 409, refused.text
            current = (await client.get("/api/status")).json()
            assert (current["model_id"], current["model_revision"], current["switching"]) == (
                ANIMAGINE, changed["model_revision"], False,
            )
            assert fake.loads == 1
        await main.jobs.shutdown()

    asyncio.run(scenario())


def test_real_backend_rechecks_lora_fit_after_model_selection_changes(monkeypatch):
    from backend.backends.diffusers_backend import DiffusersBackend
    from backend import loras

    backend = DiffusersBackend()
    flux = loras.Denoiser(object, "transformer", "FluxTransformer", {"attention": (4, 8)})
    sdxl = loras.Denoiser(object, "unet", "SDXLUNet", {"conv": (4, 16)})
    configurations = iter((flux, sdxl))
    monkeypatch.setattr(backend, "_configured_denoiser", lambda: next(configurations))
    seen = []

    def mismatch(name, denoiser):
        seen.append(denoiser)
        return None if denoiser is flux else "this adapter does not fit the selected UNet"

    monkeypatch.setattr(loras, "mismatch", mismatch)
    assert backend.lora_mismatch("style.safetensors") is None
    assert backend.lora_mismatch("style.safetensors") is None
    model_selection.select(ANIMAGINE)
    assert backend.lora_mismatch("style.safetensors") == "this adapter does not fit the selected UNet"
    assert seen == [flux, flux, sdxl]


def _forbid_hub(monkeypatch):
    from backend import hub_access

    asked = []
    monkeypatch.setattr(hub_access, "check", lambda model_id: asked.append(model_id))
    return asked


def _saved_bytes():
    path = startup_preferences.path_for(main.settings.db_path)
    return path.read_bytes() if path.exists() else None


def test_switch_to_kontext_refused_while_gated_before_hub_save_or_unload(client, fake, monkeypatch):
    from backend.instruction_edit import INSTRUCTION_EDIT_UNAVAILABLE, KONTEXT_V1

    assert client.post("/api/model/load").status_code == 200
    ready = _settled(client, main.settings.model_id)
    asked = _forbid_hub(monkeypatch)
    loads = fake.loads
    before = _saved_bytes()
    # Listed, last, as any catalogue entry; listed is not loadable.
    assert client.get("/api/models").json()["models"][-1] == {
        "model_id": KONTEXT_V1.model_id, "label": "FLUX.1 Kontext [dev]",
    }
    # The fake's gate is closed (the real backend's envelope is
    # test_instruction_edit_release's): Switch & load and Apply & save
    # alike, by the exact id or a case variant, before anything is touched.
    for model_id in (KONTEXT_V1.model_id, KONTEXT_V1.model_id.upper()):
        for load in (True, False):
            reply = client.post("/api/model/switch", json={"model_id": model_id, "load": load})
            assert reply.status_code == 422, reply.text
            assert reply.json()["detail"] == INSTRUCTION_EDIT_UNAVAILABLE
    after = client.get("/api/status").json()
    assert (after["model_id"], after["state"], after["model_revision"]) == (
        ready["model_id"], "ready", ready["model_revision"],
    )
    assert asked == [] and _saved_bytes() == before and fake.loads == loads and fake.is_ready()


def test_switch_to_kontext_refuses_unmeasured_loader_before_hub(client, fake, monkeypatch):
    from backend.instruction_edit import INSTRUCTION_EDIT_EXACT_ID, INSTRUCTION_EDIT_LOADER_UNQUALIFIED, KONTEXT_V1

    fake.edit_qualified = True  # a stand-in gate, open as the real one is
    assert client.post("/api/model/load").status_code == 200
    ready = _settled(client, main.settings.model_id)
    asked = _forbid_hub(monkeypatch)
    loads = fake.loads
    before = _saved_bytes()
    for options, detail in (
        ({"offload": "sequential"}, INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="sequential offload")),
        ({"quantization": "int8"}, INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="int8 quantization")),
        ({"quantization": "none", "offload": "none"},
         INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="BF16 (no quantization)")),
    ):
        for load in (True, False):
            reply = client.post("/api/model/switch", json={"model_id": KONTEXT_V1.model_id, "load": load, **options})
            assert reply.status_code == 422, reply.text
            assert reply.json()["detail"] == detail
    # The host as well as the selection: off Windows nothing was measured.
    fake.edit_host = ("linux", "0.40.0")
    reply = client.post("/api/model/switch", json={"model_id": KONTEXT_V1.model_id})
    assert reply.status_code == 422
    assert reply.json()["detail"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="Linux (WSL)")
    reply = client.post("/api/model/switch", json={"model_id": KONTEXT_V1.model_id.lower()})
    assert reply.status_code == 422 and reply.json()["detail"] == INSTRUCTION_EDIT_EXACT_ID
    after = client.get("/api/status").json()
    assert (after["model_id"], after["state"], after["model_revision"]) == (
        ready["model_id"], "ready", ready["model_revision"],
    )
    assert asked == [] and _saved_bytes() == before and fake.loads == loads


def test_switch_to_kontext_envelope_asks_the_hub(client, fake, monkeypatch):
    from backend import hub_access
    from backend.instruction_edit import KONTEXT_V1

    # Started on the editor, with Krea 2 Turbo selected since.
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, model_id=KONTEXT_V1.model_id))
    monkeypatch.setattr(model_selection, "_selection", (TURBO, 1))
    fake.edit_qualified = True
    assert client.post("/api/model/load").status_code == 200
    ready = _settled(client, TURBO)
    asked = []

    def refused(model_id):
        asked.append(model_id)
        return hub_access.Refusal("gated", "Request access on the model page.", f"https://huggingface.co/{model_id}")

    monkeypatch.setattr(hub_access, "check", refused)
    before, loads = _saved_bytes(), fake.loads
    reply = client.post("/api/model/switch", json={"model_id": KONTEXT_V1.model_id})
    assert reply.status_code == 403, reply.text
    assert reply.json()["detail"].startswith("Request access on the model page.")
    after = client.get("/api/status").json()
    assert (after["model_id"], after["state"], after["model_revision"]) == (TURBO, "ready", ready["model_revision"])
    assert asked == [KONTEXT_V1.model_id] and _saved_bytes() == before and fake.loads == loads
