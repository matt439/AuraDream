"""Saved performance intent never grants permission to execute it.

All lifecycle calls use the ordinary fake backend and an owned scratch gallery.
No experimental pipeline, framework, compiler or card is invoked here.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import threading
import time

import pytest

from backend import main, model_selection, startup_preferences
from backend.execution import ExecutionSpec, INVALID_LOADER_CONFIGURATION, UNAVAILABLE

KREA = "black-forest-labs/FLUX.1-Krea-dev"
TURBO = "krea/Krea-2-Turbo"
RECIPES = [{"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"}]


@pytest.fixture(autouse=True)
def idle_selection(fake, monkeypatch):
    monkeypatch.setattr(main, "_switching", False)
    monkeypatch.setattr(main, "_switch_draining", False)
    monkeypatch.setattr(main, "_switch_task", None)
    monkeypatch.setattr(main, "_model_requests", 0)
    monkeypatch.setattr(main, "_model_request_paths", {})
    monkeypatch.setattr(main, "_startup_preload", False)
    monkeypatch.setattr(main, "_switch_error", "previous outcome")
    model_selection.select(KREA, quantization="fp8", offload="model", loader_execution=None)


def save_current():
    current = model_selection.current_settings(main.settings)
    startup_preferences.save(main.settings.db_path, startup_preferences.StartupPreferences(
        current.model_id, current.quantization, current.offload, False,
        loader_execution=current.loader_execution,
    ))
    path = startup_preferences.path_for(main.settings.db_path)
    return path, path.read_bytes(), model_selection._selection


def settled(client, execution):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get("/api/status")
        assert response.status_code == 200, response.text
        body = response.json()
        if not body["switching"] and body["execution"] == execution:
            return body
        time.sleep(0.01)
    raise AssertionError(f"fake selection did not settle: {body}")


def forbidden(*_args, **_kwargs):
    pytest.fail("unavailable execution reached a forbidden effect")


@pytest.mark.parametrize("recipe", RECIPES)
def test_save_only_changes_loader_revision_and_unloads_without_granting_availability(client, fake, recipe):
    before = client.get("/api/status").json()
    response = client.post("/api/model/switch", json={"model_id": KREA, "execution": recipe, "load": False})
    assert response.status_code == 200, response.text
    body = settled(client, recipe)
    assert body["state"] == "unloaded" and not fake.loaded and fake.loads == 0
    assert body["model_revision"] == before["model_revision"] + 1
    assert body["execution_available"] is False and body["execution_unavailable_reason"] == UNAVAILABLE
    assert body["execution_configuration_reason"] is None
    saved = startup_preferences.load(main.settings.db_path)
    assert saved.loader_execution == ExecutionSpec.model_validate(recipe)
    assert saved.load_on_startup is False


@pytest.mark.parametrize("recipe", RECIPES)
def test_unavailable_save_only_checkpoint_change_does_not_ask_the_hub(client, fake, monkeypatch, recipe):
    model_selection.select(TURBO, quantization="fp8", offload="model", loader_execution=None)
    with monkeypatch.context() as guard:
        guard.setattr(main, "_refuse_hub_access", forbidden)
        response = client.post("/api/model/switch", json={"model_id": KREA, "execution": recipe, "load": False})
        assert response.status_code == 200, response.text
        assert settled(client, recipe)["model_id"] == KREA
    assert not fake.loaded and fake.loads == 0


@pytest.mark.parametrize("recipe", RECIPES)
def test_load_request_stops_before_hub_save_unload_or_state_changes(client, fake, monkeypatch, recipe):
    snapshot = save_current()
    main._load_state.update(state="ready", detail=None)
    before = dict(main._load_state)
    with monkeypatch.context() as guard:
        guard.setattr(main, "_refuse_hub_access", forbidden)
        guard.setattr(startup_preferences, "save", forbidden)
        guard.setattr(fake, "try_unload", forbidden)
        guard.setattr(fake, "load", forbidden)
        response = client.post("/api/model/switch", json={"model_id": KREA, "execution": recipe, "load": True})
    assert response.status_code == 422 and response.json()["detail"] == UNAVAILABLE
    assert main._load_state == before and main._switch_error == "previous outcome"
    assert snapshot[0].read_bytes() == snapshot[1] and model_selection._selection == snapshot[2]
    assert fake.loaded and fake.loads == 0


@pytest.mark.parametrize("recipe", RECIPES)
def test_selected_unavailable_loader_blocks_post_load_even_when_backend_is_ready(client, fake, monkeypatch, recipe):
    model_selection.select(KREA, loader_execution=recipe)
    snapshot = save_current()
    main._load_state.update(state="ready", detail=None)
    before = dict(main._load_state)
    with monkeypatch.context() as guard:
        guard.setattr(fake, "load", forbidden)
        response = client.post("/api/model/load")
    assert response.status_code == 422 and response.json()["detail"] == UNAVAILABLE
    assert main._load_state == before and main._switch_error == "previous outcome"
    assert snapshot[0].read_bytes() == snapshot[1] and model_selection._selection == snapshot[2]


@pytest.mark.parametrize("recipe", RECIPES)
def test_ensure_loaded_refuses_before_ready_check_or_generation_gate(fake, monkeypatch, recipe):
    model_selection.select(KREA, loader_execution=recipe)
    before = dict(main._load_state)
    monkeypatch.setattr(fake, "is_ready", forbidden)
    monkeypatch.setattr(fake, "load", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    with pytest.raises(ValueError, match="Execution recipes are not available"):
        asyncio.run(main.ensure_loaded())
    assert main._load_state == before and fake.loads == 0


@pytest.mark.parametrize("recipe", RECIPES)
def test_omission_preserves_selected_recipe_while_saving_only_startup_checkbox(client, fake, recipe):
    model_selection.select(KREA, loader_execution=recipe)
    before = model_selection._selection
    response = client.post("/api/model/switch", json={"model_id": KREA, "load": False, "load_on_startup": True})
    assert response.status_code == 200, response.text
    assert response.json()["execution"] == recipe
    assert model_selection._selection == before and fake.loads == 0
    saved = startup_preferences.load(main.settings.db_path)
    assert saved.loader_execution == ExecutionSpec.model_validate(recipe) and saved.load_on_startup
    refused = client.post("/api/model/switch", json={"model_id": KREA, "load": True})
    assert refused.status_code == 422 and refused.json()["detail"] == UNAVAILABLE
    assert model_selection._selection == before


@pytest.mark.parametrize("empty", [None, {}, {"compilation": None, "attention_backend": None, "step_cache": None}])
@pytest.mark.parametrize("load", [False, True])
def test_explicit_empty_can_select_eager_and_reload_from_unavailable_intent(client, fake, empty, load):
    model_selection.select(KREA, loader_execution={"compilation": "regional-v1"})
    revision = model_selection.revision()
    response = client.post("/api/model/switch", json={"model_id": KREA, "execution": empty, "load": load})
    assert response.status_code == 200, response.text
    body = settled(client, None)
    deadline = time.monotonic() + 10
    while body["state"] == "loading" and time.monotonic() < deadline:
        time.sleep(0.01)
        body = client.get("/api/status").json()
    assert body["state"] == ("ready" if load else "unloaded")
    assert body["execution_available"] and body["execution_unavailable_reason"] is None
    assert model_selection.revision() == revision + 1 and fake.loads == int(load)
    assert startup_preferences.load(main.settings.db_path).loader_execution is None


@pytest.mark.parametrize("changes", [
    {"model_id": TURBO}, {"quantization": "int8"}, {"offload": "sequential"},
    {"device": "cpu"}, {"execution": {"compilation": "regional-v1", "attention_backend": "cudnn-v1"}},
    {"execution": {"step_cache": "fbc-005-v1"}},
])
def test_save_only_refuses_structural_conflicts_without_saving_or_unloading(client, fake, monkeypatch, changes):
    snapshot = save_current()
    payload = {"model_id": KREA, "execution": {"compilation": "regional-v1"}, "load": False, **changes}
    if "device" in changes:
        monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, device=payload.pop("device")))
    with monkeypatch.context() as guard:
        guard.setattr(main, "_refuse_hub_access", forbidden)
        guard.setattr(startup_preferences, "save", forbidden)
        guard.setattr(fake, "try_unload", forbidden)
        response = client.post("/api/model/switch", json=payload)
    assert response.status_code == 422 and response.json()["detail"] != UNAVAILABLE
    assert snapshot[0].read_bytes() == snapshot[1] and model_selection._selection == snapshot[2]
    assert fake.loaded and fake.loads == 0


def test_bad_startup_diagnostic_is_visible_and_only_explicit_valid_save_repairs_it(client, fake, monkeypatch):
    # The original invalid input is never a status/error payload.
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, loader_execution_error="private-invalid-value"))
    monkeypatch.setattr(model_selection, "_selection", (None, 20))
    snapshot = save_current()
    response = client.get("/api/status")
    body = response.json()
    assert body["execution"] is None and body["execution_available"] is False
    assert body["execution_configuration_reason"] == INVALID_LOADER_CONFIGURATION
    assert "private-invalid-value" not in response.text
    assert client.post("/api/model/load").status_code == 422
    refused = client.post("/api/model/switch", json={"model_id": main.settings.model_id, "load": False})
    assert refused.status_code == 422 and refused.json()["detail"] == INVALID_LOADER_CONFIGURATION
    assert snapshot[0].read_bytes() == snapshot[1] and model_selection._selection == snapshot[2]
    repaired = client.post("/api/model/switch", json={"model_id": main.settings.model_id, "execution": None, "load": False})
    assert repaired.status_code == 200, repaired.text
    assert settled(client, None)["execution_available"] is True
    assert model_selection.current_settings(main.settings).loader_execution_error is None
    assert fake.loads == 0


def test_catalogue_describes_draft_configuration_without_hiding_choices_or_changing_selection(client):
    snapshot = save_current()
    response = client.get("/api/models?quantization=fp8&offload=model&compilation=regional-v1&attention=native")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["execution"] is None and body["execution_available"] is True
    krea = body["details"][KREA]
    assert krea["execution_configuration_reason"] is None
    assert body["details"][TURBO]["execution_configuration_reason"] is not None
    options = krea["execution_options"]
    assert [item["value"] for item in options["compilation"]] == [None, "regional-v1"]
    assert [item["value"] for item in options["attention"]] == [None, "cudnn-v1"]
    assert [item["value"] for item in options["step_cache"]] == [None, "fbc-005-v1"]
    assert options["compilation"][1]["configuration_reason"] is None
    assert options["compilation"][1]["unavailable_reason"] == UNAVAILABLE
    assert options["attention"][1]["configuration_reason"] is not None
    assert options["step_cache"][1]["configuration_reason"] is not None
    assert snapshot[0].read_bytes() == snapshot[1] and model_selection._selection == snapshot[2]


@pytest.mark.parametrize("prior", [b"{broken", b'{"version":99}', b'{"version":2,"loader_execution":{"unknown":"private-value"}}'])
def test_failed_repair_restores_exact_invalid_preferences_and_restart_diagnostic(client, fake, monkeypatch, prior):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, loader_execution_error=INVALID_LOADER_CONFIGURATION))
    monkeypatch.setattr(model_selection, "_selection", (None, 20))
    path = startup_preferences.path_for(main.settings.db_path)
    path.write_bytes(prior)
    before = model_selection._selection
    failed = threading.Event()

    def refused_cleanup():
        failed.set()
        raise RuntimeError("synthetic repair teardown refusal")

    with monkeypatch.context() as guard:
        guard.setattr(fake.critic, "unload", refused_cleanup)
        response = client.post("/api/model/switch", json={
            "model_id": main.settings.model_id, "execution": None, "load": False,
        })
        assert response.status_code == 200, response.text
        assert failed.wait(10)
        deadline = time.monotonic() + 10
        while main._switching and time.monotonic() < deadline:
            time.sleep(0.01)
        body = client.get("/api/status").json()
    assert not body["switching"] and "synthetic repair teardown refusal" in body["switch_error"]
    assert model_selection._selection == before and fake.loaded and fake.loads == 0
    assert path.read_bytes() == prior
    assert startup_preferences.load_result(main.settings.db_path).loader_execution_error == INVALID_LOADER_CONFIGURATION
    restarted = {"LOCAL_DIFFUSION_DB_PATH": str(main.settings.db_path)}
    startup_preferences.apply_saved_environment(restarted)
    assert restarted[startup_preferences.LOADER_EXECUTION_ERROR_VARIABLE] == INVALID_LOADER_CONFIGURATION
    assert body["execution_available"] is False


def test_catalogue_uses_one_frozen_execution_snapshot_across_description_worker(client, fake, monkeypatch):
    model_selection.select(KREA, loader_execution={"compilation": "regional-v1"})
    snapshot = save_current()
    describing, release = threading.Event(), threading.Event()
    describe = main.details_for
    first = True

    def held_description(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            describing.set()
            assert release.wait(15), "test did not release the synthetic catalogue read"
        return describe(*args, **kwargs)

    monkeypatch.setattr(main, "details_for", held_description)
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(client.get, "/api/models")
            assert describing.wait(10)
            model_selection.select(KREA, loader_execution={"attention_backend": "cudnn-v1"})
        finally:
            release.set()
        response = pending.result(timeout=10)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["execution"] == {"compilation": "regional-v1"}
    options = body["details"][KREA]["execution_options"]
    assert options["compilation"][1]["configuration_reason"] is None
    assert options["attention"][1]["configuration_reason"] is not None
    assert body["details"][KREA]["execution_configuration_reason"] is None
    assert model_selection.current_settings(main.settings).loader_execution == ExecutionSpec(attention_backend="cudnn-v1")
    assert snapshot[0].read_bytes() == snapshot[1] and fake.loads == 0
