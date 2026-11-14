"""A checkpoint delete through the app (`GEN#72`, stage 1 of
docs/maintainer/plan-model-delete.md): the plan's GET on scratch caches, the
guards as one reusable function, and `_deleting_model` wired inert into
every reader of `_switching` (the delete that sets it is stage 3).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import HTTPException
from PIL import Image

from backend import main, model_cache_delete as delete, model_cache_storage, model_info, model_selection, prompt_encoding, startup_preferences
from backend.encoder_table import TokenizerSource
from backend.schemas import GenerateRequest
from tests.model_cache_fixtures import scratch_caches

ERNIE = "baidu/ERNIE-Image-Turbo"
KREA2 = "krea/Krea-2-Turbo"
ILLUSTRIOUS = "OnomaAIResearch/Illustrious-XL-v2.0"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
PLAN = "/api/models/storage/plan"
DELETING = "a checkpoint's files are being deleted; wait for the delete to finish"


@pytest.fixture
def caches(tmp_path, monkeypatch):
    return scratch_caches(tmp_path, monkeypatch, "copy")


@pytest.fixture(autouse=True)
def idle(monkeypatch):
    monkeypatch.setattr(main, "_deleting_model", None)
    monkeypatch.setattr(main, "_switching", False)
    monkeypatch.setattr(main, "_switch_draining", False)
    monkeypatch.setattr(main, "_model_requests", 0)
    monkeypatch.setattr(model_selection, "_selection", (None, 0))


def _image(store, model_id: str, seed: int = 1):
    return store.save_image(Image.new("RGB", (64, 64)), GenerateRequest(prompt="a lighthouse", width=256, height=256),
                            seed=seed, duration_ms=1, model_id=model_id)


# ---------------------------------------------------------------- the plan


def test_the_plan_answers_on_a_daemon_thread_with_the_gallery_count(client, caches, store, monkeypatch):
    caches.plant(ERNIE)
    _image(store, ERNIE, 1)
    _image(store, ERNIE, 2)
    binned = _image(store, ERNIE, 3)
    store.bin_images([binned.id])
    _image(store, KREA2, 4)
    threads = []
    planner = delete.plan

    def watched(model_id, context=None, cache_dir=None):
        threads.append(threading.current_thread())
        return planner(model_id, context, cache_dir)

    monkeypatch.setattr(delete, "plan", watched)
    response = client.get(PLAN, params={"model_id": ERNIE})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert (answer["entry"], answer["label"]) == (ERNIE, "ERNIE-Image Turbo")
    assert [(part["kind"], part["repo"], part["whole"]) for part in answer["parts"]] == [("hub", ERNIE, True)]
    assert answer["frees_bytes"] == answer["parts"][0]["bytes"] > 0
    assert answer["held"] is None and answer["downloading"] == []
    assert answer["gallery_images"] == 2  # live images only, this entry only
    assert (answer["selected"], answer["loaded"], answer["load_on_startup"]) == (False, False, False)
    assert len(answer["pin"]) == 64
    assert len(threads) == 1 and threads[0].daemon and threads[0].name == "model-cache-plan"
    # Read-only: the inventory's DELETE stays a 405.
    assert client.delete("/api/models/storage").status_code == 405


def test_a_lite_ids_images_count_for_its_entry(client, caches, store):
    entry = model_info.catalogue_entry(KREA2)
    assert entry is not None and entry.lite_repo("int4")
    caches.plant(KREA2)
    _image(store, KREA2, 1)
    _image(store, entry.lite_repo("int4") or "", 2)
    assert client.get(PLAN, params={"model_id": KREA2}).json()["gallery_images"] == 2


def test_unknown_and_absent_entries_are_the_same_404(client, caches):
    caches.plant(ERNIE)
    answers = [client.get(PLAN, params={"model_id": model}) for model in ("nobody/nothing", KREA2)]
    assert [answer.status_code for answer in answers] == [404, 404]
    assert answers[0].json() == answers[1].json() == {
        "detail": "No catalogue checkpoint by that id has files in this server's cache.",
        "code": "model_not_on_disk"}


def test_the_plan_reports_selected_loaded_and_saved_for_the_next_start(client, caches, fake, monkeypatch):
    caches.plant(ERNIE)
    monkeypatch.setattr(model_selection, "_selection", (ERNIE, 7))
    fake.loaded = True
    answer = client.get(PLAN, params={"model_id": ERNIE}).json()
    assert (answer["selected"], answer["loaded"], answer["load_on_startup"]) == (True, True, False)
    # Saved to load on the next start, though this run selects another.
    monkeypatch.setattr(model_selection, "_selection", (KREA2, 8))
    startup_preferences.save(main.settings.db_path, startup_preferences.StartupPreferences(
        model_id=ERNIE, quantization="fp8", offload="model", load_on_startup=True))
    answer = client.get(PLAN, params={"model_id": ERNIE}).json()
    assert (answer["selected"], answer["loaded"], answer["load_on_startup"]) == (False, False, True)


def test_a_running_scan_holds_the_plan_by_code(client, caches):
    caches.plant(ERNIE)
    model_cache_storage._scan_lock.acquire()
    try:
        response = client.get(PLAN, params={"model_id": ERNIE})
    finally:
        model_cache_storage._scan_lock.release()
    assert response.status_code == 409 and response.json()["code"] == "model_delete_busy"
    assert client.get(PLAN, params={"model_id": ERNIE}).status_code == 200


def test_an_unreadable_cache_says_so_without_its_paths(client, monkeypatch):
    def unreadable(*args, **kwargs):
        raise OSError("private path and environment text")

    monkeypatch.setattr(delete, "plan", unreadable)
    response = client.get(PLAN, params={"model_id": ERNIE})
    assert response.status_code == 503
    assert "private" not in response.text and "drive" in response.json()["detail"]


def test_no_answer_names_an_absolute_path(client, caches, tmp_path):
    caches.plant(ILLUSTRIOUS)
    caches.convert(ILLUSTRIOUS)
    partial = caches.folder(ILLUSTRIOUS) / "blobs" / "partial.incomplete"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"half")
    answer = client.get(PLAN, params={"model_id": ILLUSTRIOUS}).json()
    assert answer["held"] and answer["downloading"] and answer["also_taken_shared"]
    text = json.dumps(answer)
    for root in (tmp_path, caches.home, caches.hub, caches.converted):
        assert str(root) not in text and root.as_posix() not in text and json.dumps(str(root))[1:-1] not in text


# -------------------------------------------------------------- the guards


def _plan(**fields) -> delete.Plan:
    defaults = dict(entry=ERNIE, label="ERNIE-Image Turbo", model_ids=(ERNIE,), context=delete.PlanContext(),
                    repositories=frozenset({ERNIE}))
    return delete.Plan(**{**defaults, **fields})


def _code(plan: delete.Plan, saved: str | None = None) -> str | None:
    refused = main._model_delete_refusal(plan, main._model_delete_context(saved))
    return None if refused is None else refused.code


def test_an_idle_server_passes_every_guard(fake):
    assert _code(_plan()) is None


@pytest.mark.parametrize("state", ["selected", "loaded"])
def test_the_selected_or_loaded_checkpoint_is_refused(fake, monkeypatch, state):
    monkeypatch.setattr(model_selection, "_selection", (ERNIE, 3))
    fake.loaded = state == "loaded"
    refused = main._model_delete_refusal(_plan(), main._model_delete_context(None))
    assert refused is not None and refused.code == "model_selected"
    assert refused.detail == "ERNIE-Image Turbo is the selected checkpoint. Choose another checkpoint first."


def test_a_lite_id_selected_counts_as_its_entry(fake, monkeypatch):
    entry = model_info.catalogue_entry(KREA2)
    assert entry is not None
    monkeypatch.setattr(model_selection, "_selection", (entry.lite_repo("nvfp4"), 3))
    assert _code(_plan(entry=KREA2, label="Krea 2 Turbo", model_ids=(KREA2,))) == "model_selected"


def test_the_checkpoint_saved_for_the_next_start_is_refused_in_its_own_words(fake):
    refused = main._model_delete_refusal(_plan(), main._model_delete_context(ERNIE))
    assert refused is not None and refused.code == "model_selected"
    assert "saved to load automatically on the next start" in str(refused.detail)


def test_the_saved_start_entry_is_read_from_the_settings_file(fake, store):
    assert main._saved_start_entry() is None
    prefs = startup_preferences.StartupPreferences(model_id=ERNIE, quantization="fp8", offload="model",
                                                   load_on_startup=False)
    startup_preferences.save(main.settings.db_path, prefs)
    assert main._saved_start_entry() is None  # saved, but not to load on startup
    startup_preferences.save(main.settings.db_path, startup_preferences.StartupPreferences(
        model_id=ERNIE, quantization="fp8", offload="model", load_on_startup=True))
    assert main._saved_start_entry() == ERNIE


class _Task:
    def __init__(self, done: bool) -> None:
        self._done = done

    def done(self) -> bool:
        return self._done


@pytest.mark.parametrize("busy", ["jobs", "card", "load lock", "loading", "switching", "draining", "switch task",
                                  "unload task", "counted request", "scan", "another delete"])
def test_each_busy_state_refuses_by_code(fake, monkeypatch, busy):
    if busy == "jobs":
        monkeypatch.setattr(main.jobs, "busy", lambda: True)
    elif busy == "card":
        monkeypatch.setattr(main.jobs, "card", lambda: ("critic", "ticket"))
    elif busy == "load lock":
        monkeypatch.setattr(main._load_lock, "locked", lambda: True)
    elif busy == "loading":
        # `POST /api/model/load` reserves "loading" before it takes the lock.
        monkeypatch.setitem(main._load_state, "state", "loading")
    elif busy == "switching":
        monkeypatch.setattr(main, "_switching", True)
    elif busy == "draining":
        monkeypatch.setattr(main, "_switch_draining", True)
    elif busy == "switch task":
        monkeypatch.setattr(main, "_switch_task", _Task(False))
    elif busy == "unload task":
        monkeypatch.setattr(main, "_unload_task", _Task(False))
    elif busy == "counted request":
        monkeypatch.setattr(main, "_model_requests", 1)
    elif busy == "scan":
        model_cache_storage._scan_lock.acquire()
    else:
        monkeypatch.setattr(main, "_deleting_model", "Anima Aesthetic v1.1")
    try:
        refused = main._model_delete_refusal(_plan(), main._model_delete_context(None))
    finally:
        if busy == "scan":
            model_cache_storage._scan_lock.release()
    assert refused is not None and refused.code == "model_delete_busy" and refused.status_code == 409
    assert "delete ERNIE-Image Turbo once it is done" in str(refused.detail)


def test_finished_tasks_hold_nothing(fake, monkeypatch):
    monkeypatch.setattr(main, "_switch_task", _Task(True))
    monkeypatch.setattr(main, "_unload_task", _Task(True))
    assert _code(_plan()) is None


def test_a_tokenizer_fetch_holds_only_a_delete_it_reads_from(fake, monkeypatch):
    sdxl = TokenizerSource(SDXL, "tokenizer")
    monkeypatch.setattr(prompt_encoding, "pending_sources", lambda: frozenset({sdxl}))
    illustrious = _plan(entry=ILLUSTRIOUS, label="Illustrious XL v2.0-STABLE", model_ids=(ILLUSTRIOUS,),
                        repositories=frozenset({ILLUSTRIOUS, SDXL}))
    assert _code(illustrious) == "model_delete_busy"
    assert _code(_plan()) is None
    # A fetch from the entry's converted tree holds it too.
    tree = TokenizerSource(SDXL, "tokenizer", tree=ERNIE)
    monkeypatch.setattr(prompt_encoding, "pending_sources", lambda: frozenset({tree}))
    assert _code(_plan()) == "model_delete_busy"


def test_a_partial_download_and_a_held_part_refuse_by_their_codes(fake):
    downloading = _plan(downloading=[{"repo": ERNIE, "file": "models--baidu--ERNIE-Image-Turbo/blobs/x.incomplete"}])
    refused = main._model_delete_refusal(downloading, main._model_delete_context(None))
    assert refused is not None and refused.code == "model_downloading"
    assert "models--baidu--ERNIE-Image-Turbo/blobs/x.incomplete" in str(refused.detail)
    assert "prune" not in str(refused.detail)
    held = _plan(held={"code": "model_delete_held", "detail": "Not built yet."})
    refused = main._model_delete_refusal(held, main._model_delete_context(None))
    assert refused is not None and (refused.code, refused.detail) == ("model_delete_held", "Not built yet.")


# ----------------------------------------------- _deleting_model, inert


def test_a_delete_in_progress_refuses_model_work_switch_unload_and_bundles(client, monkeypatch):
    monkeypatch.setattr(main, "_deleting_model", "Anima Aesthetic v1.1")
    payload = {"prompt": "a lighthouse", "width": 256, "height": 256}
    for path, body in (("/api/generate", payload), ("/api/model/load", None), ("/api/model/unload", None),
                       ("/api/model/switch", {"model_id": KREA2})):
        response = client.post(path, json=body)
        assert response.status_code == 409 and response.json()["detail"] == DELETING, (path, response.text)
    assert client.post("/api/gallery/bundles/export", json={"expected": 0}).status_code == 409
    assert main.bundle_manager.state["state"] == "idle"
    status = client.get("/api/status").json()
    assert status["model_delete"] == "Anima Aesthetic v1.1" and status["switching"] is False


def test_the_routes_own_checks_refuse_too(fake, monkeypatch):
    # Behind the middleware: a switch's busy check and the unload's own.
    monkeypatch.setattr(main, "_deleting_model", "Anima Aesthetic v1.1")
    with pytest.raises(HTTPException) as switch:
        main._refuse_busy_switch()
    assert switch.value.status_code == 409 and switch.value.detail == DELETING
    with pytest.raises(HTTPException) as unload:
        asyncio.run(main._unload_weights())
    assert unload.value.status_code == 409 and unload.value.detail == DELETING
    assert main._switching is False


def test_a_delete_keeps_the_pc_awake_and_the_idle_release_away(fake, monkeypatch):
    assert main._busy_deleting in main.KEEP_AWAKE_PROVIDERS
    assert main._busy_deleting() is None
    assert main._idle_unload_busy() is False
    monkeypatch.setattr(main, "_deleting_model", "Anima Aesthetic v1.1")
    assert main._busy_deleting() == "deleting a checkpoint"
    assert main._idle_unload_busy() is True


def test_status_says_no_delete_when_none_runs(client):
    assert client.get("/api/status").json()["model_delete"] is None


# ------------------------------------------------- Check's fetches in flight


def test_a_fetch_in_flight_is_pending_until_its_thread_returns(fake, monkeypatch):
    monkeypatch.setattr(prompt_encoding, "_running", set())
    release = threading.Event()
    source = TokenizerSource(SDXL, "tokenizer")

    def slow(asked):
        release.wait(10)
        raise FileNotFoundError(asked.label)

    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", slow)
    prompt_encoding._start(source)
    assert prompt_encoding.pending_sources() == {source}
    # Unload forgets the fetch; it still reads the files until it returns.
    prompt_encoding.unload()
    assert prompt_encoding.pending_sources() == {source}
    release.set()
    deadline = time.monotonic() + 10
    while prompt_encoding.pending_sources() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert prompt_encoding.pending_sources() == frozenset()
