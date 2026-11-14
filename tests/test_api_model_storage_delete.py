"""A checkpoint delete through the app (`GEN#72`, docs/maintainer/plan-model-delete.md):
the plan's GET on scratch caches (stage 1), the guards as one reusable
function, `_deleting_model` in every reader of `_switching`, and the delete's
POST (stage 3): every guard by its code, the reservation while the worker
runs, a caller who leaves, the cache facts' revision, no absolute path; and
files inside a repository other entries share (stage 4) through the POST.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from backend import main, model_cache_delete as delete, model_cache_storage, model_info, model_selection, prompt_encoding, schemas, startup_preferences
from backend.encoder_table import TokenizerSource
from backend.schemas import GenerateRequest
from tests.api_helpers import ORIGIN
from tests.model_cache_fixtures import scratch_caches

ERNIE = "baidu/ERNIE-Image-Turbo"
KREA2 = "krea/Krea-2-Turbo"
ILLUSTRIOUS = "OnomaAIResearch/Illustrious-XL-v2.0"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
ANIMA, AESTHETIC = "circlestone-labs/Anima", "circlestone-labs/Anima@aesthetic-v1.1"
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
    # A cache-wide shared Xet store holds a delete, never an absent entry's 404.
    store = caches.hub / "blobs"
    store.mkdir()
    (store / delete.SHARED_STORE_MARKER).write_text("1\n")
    absent = client.get(PLAN, params={"model_id": KREA2})
    assert absent.status_code == 404 and absent.json()["code"] == "model_not_on_disk"
    assert client.get(PLAN, params={"model_id": ERNIE}).json()["held"]["code"] == "model_delete_held"


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
    # A cache-wide shared Xet store: a hold in words too.
    (caches.hub / "blobs").mkdir()
    (caches.hub / "blobs" / delete.SHARED_STORE_MARKER).write_text("1\n")
    answer = client.get(PLAN, params={"model_id": ILLUSTRIOUS}).json()
    assert answer["held"] and answer["downloading"] and answer["also_taken_shared"]
    text = json.dumps(answer)
    for root in (tmp_path, caches.home, caches.hub, caches.converted):
        assert str(root) not in text and root.as_posix() not in text and json.dumps(str(root))[1:-1] not in text


def test_the_inventory_holds_each_row_its_plan_holds(client, caches, monkeypatch):
    # With the shared-repository switch off, Illustrious alone, the last
    # holder of SDXL base's shared files, has a held plan, so its row says
    # so as the cache is read, before any press, in the plan's own words.
    # ERNIE's and a lite sub-line's are null.
    monkeypatch.setattr(delete, "SHARED_REPOSITORY_DELETE", False)
    caches.plant(ILLUSTRIOUS)
    caches.plant(ERNIE)
    entry = model_info.catalogue_entry(KREA2)
    assert entry is not None and (lite := entry.lite_repo("nvfp4"))
    caches.plant(lite)
    threads = []
    snapshot = model_cache_storage.snapshot

    def watched(*args, **kwargs):
        threads.append(threading.current_thread())
        return snapshot(*args, **kwargs)

    monkeypatch.setattr(model_cache_storage, "snapshot", watched)
    response = client.get("/api/models/storage")
    assert response.status_code == 200, response.text
    rows = {row["model_id"]: row for row in response.json()["models"]}
    assert all("delete_held" in row for row in rows.values())
    planned = client.get(PLAN, params={"model_id": ILLUSTRIOUS}).json()
    assert planned["held"] and rows[ILLUSTRIOUS]["delete_held"] == planned["held"]["detail"]
    assert SDXL in rows[ILLUSTRIOUS]["delete_held"]
    assert rows[ERNIE]["delete_held"] is None and rows[lite]["delete_held"] is None
    assert len(threads) == 1 and threads[0].daemon and threads[0].name == "model-cache-storage"
    assert not model_cache_storage._scan_lock.locked()
    # With WAI on disk the shared files are kept: Illustrious's row is free.
    caches.plant("LyliaEngine/waiIllustriousSDXL_v170")
    rows = {row["model_id"]: row for row in client.get("/api/models/storage").json()["models"]}
    assert rows[ILLUSTRIOUS]["delete_held"] is None
    # Switched on (stage 4), the last holder's row is free as well.
    monkeypatch.setattr(delete, "SHARED_REPOSITORY_DELETE", True)
    shutil.rmtree(caches.folder("LyliaEngine/waiIllustriousSDXL_v170"))
    rows = {row["model_id"]: row for row in client.get("/api/models/storage").json()["models"]}
    assert rows[ILLUSTRIOUS]["delete_held"] is None


def test_a_plan_that_fails_leaves_its_row_unheld_and_the_inventory_answering(client, caches, monkeypatch):
    caches.plant(ERNIE)
    planner = delete._plan

    def failing(entry_id, *args, **kwargs):
        if entry_id == ERNIE:
            raise OSError("private path")
        return planner(entry_id, *args, **kwargs)

    monkeypatch.setattr(delete, "_plan", failing)
    response = client.get("/api/models/storage")
    assert response.status_code == 200 and "private" not in response.text
    assert next(row for row in response.json()["models"] if row["model_id"] == ERNIE)["delete_held"] is None


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


def test_an_id_outside_the_catalogue_in_the_plans_repositories_is_refused(fake, monkeypatch):
    # Z-Anime's recipe takes Tongyi-MAI/Z-Image whole; selected or loaded by
    # its own id (AURADREAM_MODEL_ID, any case), it is the plan's too.
    zanime = _plan(entry="SeeSee21/Z-Anime", label="Z-Anime", model_ids=("SeeSee21/Z-Anime",),
                   repositories=frozenset({"SeeSee21/Z-Anime", "Tongyi-MAI/Z-Image"}))
    for selected in ("Tongyi-MAI/Z-Image", "tongyi-mai/z-image"):
        monkeypatch.setattr(model_selection, "_selection", (selected, 3))
        fake.loaded = True
        refused = main._model_delete_refusal(zanime, main._model_delete_context(None))
        assert refused is not None and refused.code == "model_selected"
        assert refused.detail == (f"Z-Anime's delete takes files of {selected}, which is the selected "
                                  "checkpoint. Choose another checkpoint first.")
    monkeypatch.setattr(model_selection, "_selection", ("someone/else", 4))
    assert main._model_delete_refusal(zanime, main._model_delete_context(None)) is None
    saved = main._model_delete_refusal(zanime, main._model_delete_context("Tongyi-MAI/Z-Image"))
    assert saved is not None and saved.code == "model_selected"
    assert saved.detail == ("Z-Anime's delete takes files of Tongyi-MAI/Z-Image, which is saved to load "
                            "automatically on the next start. Choose another checkpoint first.")


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


# ------------------------------------------- _deleting_model, its readers


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


# ------------------------------------------------- the delete (stage 3)

DELETE = "/api/models/storage/delete"


def _pin(client, model_id: str) -> str:
    response = client.get(PLAN, params={"model_id": model_id})
    assert response.status_code == 200, response.text
    return response.json()["pin"]


def _delete(client, model_id: str, pin: str, **headers):
    return client.post(DELETE, json={"model_id": model_id, "pin": pin}, headers=headers or None)


def test_a_delete_answers_what_it_freed_and_releases_everything(client, caches, monkeypatch):
    caches.plant(ERNIE)
    caches.plant(KREA2)
    pin = _pin(client, ERNIE)
    planned = client.get(PLAN, params={"model_id": ERNIE}).json()["frees_bytes"]
    revision = model_info.cache_revision()
    threads = []
    worker = delete.delete

    def watched(*args, **kwargs):
        threads.append(threading.current_thread())
        assert main._deleting_model == "ERNIE-Image Turbo"
        assert main._busy_deleting() == "deleting a checkpoint"
        return worker(*args, **kwargs)

    monkeypatch.setattr(delete, "delete", watched)
    response = _delete(client, ERNIE, pin)
    assert response.status_code == 200, response.text
    answer = response.json()
    assert set(answer) == {"label", "planned_bytes", "freed_bytes", "remaining"}
    assert (answer["label"], answer["planned_bytes"], answer["remaining"]) == ("ERNIE-Image Turbo", planned, [])
    assert isinstance(answer["freed_bytes"], int) and answer["freed_bytes"] >= 0
    assert len(threads) == 1 and threads[0].daemon and threads[0].name == "model-cache-delete"
    assert not caches.folder(ERNIE).exists() and caches.folder(KREA2).is_dir()
    assert main._deleting_model is None and main._delete_task is None
    assert not model_cache_storage._scan_lock.locked()
    # The Models cache facts follow (`download_progress` and the page key on it).
    assert model_info.cache_revision() > revision
    assert client.get("/api/status").json()["model_delete"] is None
    # Deleted, it is no longer on disk: the same 404 as an unknown id.
    gone = client.get(PLAN, params={"model_id": ERNIE})
    assert gone.status_code == 404 and gone.json()["code"] == "model_not_on_disk"


def test_one_of_two_sharers_goes_file_by_file_through_the_post(client, caches):
    # Stage 4: Anima Aesthetic's file sits in the repository Anima Turbo's
    # recipe names too, and the base both use is kept for Turbo, present.
    for model in (ANIMA, AESTHETIC):
        caches.plant(model)
    _, turbo_commit, _ = caches.own_file(ANIMA)
    _, commit, _ = caches.own_file(AESTHETIC)
    planned = client.get(PLAN, params={"model_id": AESTHETIC}).json()
    assert planned["held"] is None and [part["shared_repository"] for part in planned["parts"]] == [True]
    assert planned["kept_shared"][0]["with"] == ["Anima Turbo v1.1"]
    turbo = {key: value for key, value in caches.state().items() if turbo_commit in key or "Anima-Base" in key}
    response = _delete(client, AESTHETIC, planned["pin"])
    assert response.status_code == 200, response.text
    assert response.json()["remaining"] == [] and response.json()["planned_bytes"] == planned["frees_bytes"]
    assert not (caches.folder(ANIMA) / "snapshots" / commit).exists()
    assert {key: value for key, value in caches.state().items()
            if turbo_commit in key or "Anima-Base" in key} == turbo
    rows = {row["model_id"]: row for row in client.get("/api/models/storage").json()["models"]}
    assert rows[AESTHETIC]["exclusive_bytes"] == 0 and rows[ANIMA]["exclusive_bytes"] > 0
    assert rows[ANIMA]["delete_held"] is None
    assert main._deleting_model is None and not model_cache_storage._scan_lock.locked()


def test_the_body_is_exact(client, caches):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)
    for body in ({"model_id": ERNIE}, {"pin": pin}, {"model_id": ERNIE, "pin": pin, "force": True},
                 {"model_id": ERNIE, "pin": 7}, {"model_id": "", "pin": pin}):
        assert client.post(DELETE, json=body).status_code == 422, body
    assert caches.folder(ERNIE).is_dir()


def test_a_cross_site_post_is_refused_before_anything(client, caches):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)
    response = _delete(client, ERNIE, pin, Origin="https://elsewhere.example")
    assert response.status_code == 403
    assert caches.folder(ERNIE).is_dir() and main._deleting_model is None


def test_a_changed_cache_and_an_absent_entry_refuse_by_code(client, caches):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)
    caches.put(ERNIE, "7" * 40, "added.json")
    before = caches.state()
    changed = _delete(client, ERNIE, pin)
    assert changed.status_code == 409 and changed.json()["code"] == "model_cache_changed"
    assert caches.state() == before
    for model in ("nobody/nothing", KREA2):
        absent = _delete(client, model, pin)
        assert absent.status_code == 404 and absent.json() == {
            "detail": "No catalogue checkpoint by that id has files in this server's cache.",
            "code": "model_not_on_disk"}
    assert main._deleting_model is None


def test_held_downloading_and_in_use_refuse_by_code_with_nothing_changed(client, caches, monkeypatch):
    # Held: Anima Turbo's file at a commit no record names, which Anima
    # Aesthetic could also own (one repository, two entries).
    caches.put(ANIMA, "8" * 40, "split_files/diffusion_models/anima-turbo-v1.0.safetensors")
    held = _delete(client, ANIMA, _pin(client, ANIMA))
    assert held.status_code == 409 and held.json()["code"] == "model_delete_held"
    assert "could also own" in held.json()["detail"]
    caches.plant(ERNIE)
    partial = caches.folder(ERNIE) / "blobs" / "0123abcd.incomplete"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"half")
    downloading = _delete(client, ERNIE, _pin(client, ERNIE))
    assert downloading.status_code == 409 and downloading.json()["code"] == "model_downloading"
    assert "0123abcd.incomplete" in downloading.json()["detail"]
    partial.unlink()
    before = caches.state()
    pin = _pin(client, ERNIE)

    def refuse(source, destination, *args, **kwargs):
        raise PermissionError(13, "in use", str(source), 32)

    monkeypatch.setattr(delete.os, "rename", refuse)
    in_use = _delete(client, ERNIE, pin)
    assert in_use.status_code == 409 and in_use.json()["code"] == "model_files_in_use"
    assert caches.state() == before and main._deleting_model is None and main._delete_task is None


@pytest.mark.parametrize("state", ["selected", "loaded", "saved"])
def test_the_selected_loaded_or_saved_entry_is_refused_by_the_route(client, caches, fake, monkeypatch, state):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)
    if state == "saved":
        startup_preferences.save(main.settings.db_path, startup_preferences.StartupPreferences(
            model_id=ERNIE, quantization="fp8", offload="model", load_on_startup=True))
    else:
        monkeypatch.setattr(model_selection, "_selection", (ERNIE, 9))
        fake.loaded = state == "loaded"
    response = _delete(client, ERNIE, pin)
    assert response.status_code == 409 and response.json()["code"] == "model_selected"
    assert "Choose another checkpoint first" in response.json()["detail"]
    assert caches.folder(ERNIE).is_dir()


@pytest.mark.parametrize("busy", ["jobs", "card", "load lock", "loading", "switching", "counted request",
                                  "tokenizer", "scan", "another delete", "gallery maintenance"])
def test_each_busy_state_refuses_the_route_by_code(client, caches, monkeypatch, busy):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)
    code = "model_delete_busy"
    if busy == "jobs":
        monkeypatch.setattr(main.jobs, "busy", lambda: True)
    elif busy == "card":
        monkeypatch.setattr(main.jobs, "card", lambda: ("critic", "ticket"))
    elif busy == "load lock":
        monkeypatch.setattr(main._load_lock, "locked", lambda: True)
    elif busy == "loading":
        monkeypatch.setitem(main._load_state, "state", "loading")
    elif busy == "switching":
        monkeypatch.setattr(main, "_switching", True)
    elif busy == "counted request":
        monkeypatch.setattr(main, "_model_requests", 1)
    elif busy == "tokenizer":
        monkeypatch.setattr(prompt_encoding, "pending_sources",
                            lambda: frozenset({TokenizerSource(ERNIE, "tokenizer")}))
    elif busy == "scan":
        model_cache_storage._scan_lock.acquire()
    elif busy == "another delete":
        monkeypatch.setattr(main, "_deleting_model", "Anima Aesthetic v1.1")
    else:
        monkeypatch.setitem(main.bundle_manager.state, "state", "running")
        code = "gallery_maintenance"
    try:
        response = _delete(client, ERNIE, pin)
    finally:
        if busy == "scan":
            model_cache_storage._scan_lock.release()
    assert response.status_code == 409 and response.json()["code"] == code, response.text
    assert caches.folder(ERNIE).is_dir()
    if busy != "another delete":
        assert main._deleting_model is None


def test_while_a_delete_runs_model_work_switch_unload_bundles_and_scans_wait(client, caches, monkeypatch):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)
    entered, release = threading.Event(), threading.Event()
    worker = delete.delete

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return worker(*args, **kwargs)

    monkeypatch.setattr(delete, "delete", held)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(_delete, client, ERNIE, pin)
        try:
            assert entered.wait(30)
            assert main._deleting_model == "ERNIE-Image Turbo"
            payload = {"prompt": "a lighthouse", "width": 256, "height": 256}
            for path, body in (("/api/generate", payload), ("/api/model/load", None), ("/api/model/unload", None),
                               ("/api/model/switch", {"model_id": KREA2})):
                response = client.post(path, json=body)
                assert response.status_code == 409 and response.json()["detail"] == DELETING, (path, response.text)
            assert client.post("/api/gallery/bundles/export", json={"expected": 0}).status_code == 409
            status = client.get("/api/status").json()
            assert status["model_delete"] == "ERNIE-Image Turbo" and status["switching"] is False
            assert main._busy_deleting() == "deleting a checkpoint" and main._idle_unload_busy()
            again = _delete(client, ERNIE, pin)
            assert again.status_code == 409 and again.json()["code"] == "model_delete_busy"
        finally:
            release.set()
        assert future.result(timeout=30).status_code == 200
    assert main._deleting_model is None and not caches.folder(ERNIE).exists()


def test_a_caller_who_leaves_does_not_stop_the_delete(client, caches, monkeypatch):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)
    entered, release = threading.Event(), threading.Event()
    worker = delete.delete

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return worker(*args, **kwargs)

    monkeypatch.setattr(delete, "delete", held)
    request = schemas.ModelDeleteRequest(model_id=ERNIE, pin=pin)

    async def scenario():
        waiter = asyncio.create_task(main.delete_model_storage(request))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            owner = main._delete_task
            assert owner is not None
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            assert main._deleting_model == "ERNIE-Image Turbo" and not owner.done()
            response = await asyncio.to_thread(client.post, "/api/model/load")
            assert response.status_code == 409
        finally:
            release.set()
        outcome = await asyncio.wait_for(owner, 30)
        assert outcome.remaining == []
        assert main._deleting_model is None and main._delete_task is None

    client.portal.call(scenario)
    assert not caches.folder(ERNIE).exists()


def test_a_failure_nobody_expected_still_releases(client, caches, monkeypatch):
    caches.plant(ERNIE)
    pin = _pin(client, ERNIE)

    def broken(*args, **kwargs):
        raise RuntimeError("a private path and words")

    monkeypatch.setattr(delete, "delete", broken)
    failed = TestClient(main.app, base_url=ORIGIN, raise_server_exceptions=False)
    response = failed.post(DELETE, json={"model_id": ERNIE, "pin": pin})
    assert response.status_code == 500 and "private" not in response.text
    assert main._deleting_model is None and main._delete_task is None


def test_the_delete_answers_no_absolute_path(client, caches, tmp_path, monkeypatch):
    entry = model_info.catalogue_entry(KREA2)
    assert entry is not None
    nvfp4 = entry.lite_repo("nvfp4")
    assert nvfp4
    caches.plant(KREA2)
    caches.plant(nvfp4)
    pin = _pin(client, KREA2)
    rmtree = shutil.rmtree
    lite = "models--" + nvfp4.replace("/", "--")

    def stop(path, *args, **kwargs):
        if (Path(path) / lite).exists():
            raise PermissionError(13, "in use", str(path), 32)
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(delete.shutil, "rmtree", stop)
    response = _delete(client, KREA2, pin)
    assert response.status_code == 200 and response.json()["remaining"]
    assert response.json()["remaining"][0]["kind"] == "lite"
    text = response.text
    for root in (tmp_path, caches.home, caches.hub, caches.converted):
        assert str(root) not in text and root.as_posix() not in text and json.dumps(str(root))[1:-1] not in text
    # The leftover is out of the cache, named beside it, and swept next time.
    monkeypatch.setattr(delete.shutil, "rmtree", rmtree)
    warnings = client.get("/api/models/storage").json()["warnings"]
    assert any("hub.auradream-deleting" in warning for warning in warnings)
