"""Explicit switch recipes must refuse rather than silently switch as Eager."""
from __future__ import annotations

import pytest

from backend import main, model_selection, startup_preferences
from backend.execution import ExecutionSpec, UNAVAILABLE
from backend.schemas_model import ModelSwitchRequest

ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
NONEMPTY = [
    {"compilation": "regional-v1"},
    {"attention_backend": "cudnn-v1"},
    {"step_cache": "fbc-005-v1"},
    {"compilation": "regional-v1", "attention_backend": "cudnn-v1", "step_cache": "fbc-005-v1"},
]
MALFORMED = [
    {"unknown": None}, {"compilation": "unknown"}, {"attention_backend": "unknown"},
    {"step_cache": "unknown"}, {"compilation": False}, "regional-v1", [], False, 1,
]


@pytest.fixture(autouse=True)
def idle_switch(fake, monkeypatch):
    monkeypatch.setattr(main, "_switching", False)
    monkeypatch.setattr(main, "_switch_draining", False)
    monkeypatch.setattr(main, "_switch_task", None)
    monkeypatch.setattr(main, "_startup_preload", False)
    monkeypatch.setattr(main, "_switch_error", "earlier switch outcome")
    monkeypatch.setattr(model_selection, "_selection", (None, 0))


def saved_snapshot():
    previous = startup_preferences.StartupPreferences(main.settings.model_id, "int8", "sequential", True)
    startup_preferences.save(main.settings.db_path, previous)
    path = startup_preferences.path_for(main.settings.db_path)
    return path, path.read_bytes(), tuple(model_selection._selection), dict(main._load_state)


def assert_unchanged(snapshot, fake):
    path, data, selection, load_state = snapshot
    assert path.read_bytes() == data and model_selection._selection == selection
    assert main._load_state == load_state and fake.loaded and fake.loads == 0
    assert main._startup_preload is False and main._switch_error == "earlier switch outcome"
    assert not main._switching and not main._switch_draining and main._switch_task is None
    assert main.jobs.active() == []


def forbidden(label, effects):
    def unexpected(*_args, **_kwargs):
        effects.append(label)
        pytest.fail(f"Refused execution reached {label}")

    return unexpected


@pytest.mark.parametrize("recipe", NONEMPTY)
def test_switch_schema_preserves_known_explicit_recipe_for_selection_validation(recipe):
    request = ModelSwitchRequest(model_id=ANIMAGINE, execution=recipe)
    assert isinstance(request.execution, ExecutionSpec)
    assert request.execution.model_dump(exclude_none=True) == recipe


@pytest.mark.parametrize("recipe", [*NONEMPTY, *MALFORMED])
def test_direct_selection_refuses_before_resolving_any_loader_or_catalogue(monkeypatch, recipe):
    effects = []
    monkeypatch.setattr(model_selection, "current_settings", forbidden("current settings", effects))
    monkeypatch.setattr(model_selection, "available", forbidden("catalogue", effects))
    with pytest.raises(ValueError):
        model_selection.validate(main.settings, "not-a-catalogue-checkpoint", quantization="invalid",
                                  offload="invalid", execution=recipe)
    assert effects == []


@pytest.mark.parametrize("recipe", [
    ExecutionSpec(compilation="regional-v1"),
    ExecutionSpec().model_copy(update={"attention_backend": "cudnn-v1"}),
    ExecutionSpec.model_construct(step_cache="fbc-005-v1"),
    ExecutionSpec.model_construct(compilation="unknown"),
    ExecutionSpec().model_copy(update={"step_cache": "unknown"}),
])
def test_direct_copied_or_constructed_execution_cannot_bypass_central_guard(monkeypatch, recipe):
    effects = []
    monkeypatch.setattr(model_selection, "current_settings", forbidden("current settings", effects))
    with pytest.raises(ValueError):
        model_selection.validate(main.settings, ANIMAGINE, execution=recipe)
    assert effects == []


@pytest.mark.parametrize("recipe", NONEMPTY)
@pytest.mark.parametrize("same_checkpoint", [False, True])
def test_http_known_recipe_load_refuses_before_catalogue_hub_save_unload_or_selection(client, fake, monkeypatch, recipe, same_checkpoint):
    snapshot, effects = saved_snapshot(), []
    # Guard endpoint effects only. Lifespan shutdown legitimately unloads the
    # fake after this test's assertions, so restore before TestClient exits.
    with monkeypatch.context() as guard:
        guard.setattr(model_selection, "current_settings", forbidden("current settings", effects))
        guard.setattr(model_selection, "available", forbidden("catalogue", effects))
        guard.setattr(startup_preferences, "save", forbidden("preferences", effects))
        guard.setattr(main, "_refuse_hub_access", forbidden("Hub", effects))
        guard.setattr(fake, "try_unload", forbidden("unload", effects))
        guard.setattr(fake, "load", forbidden("load", effects))
        guard.setattr(model_selection, "select", forbidden("selection", effects))
        response = client.post("/api/model/switch", json={"model_id": main.settings.model_id if same_checkpoint else ANIMAGINE, "execution": recipe,
                                                         "load": True, "load_on_startup": True})
    expected = "Step caching is a per-render choice, not a model loading option." if "step_cache" in recipe else UNAVAILABLE
    assert response.status_code == 422 and response.json()["detail"] == expected
    assert effects == []
    assert_unchanged(snapshot, fake)


@pytest.mark.parametrize("load", [False, True])
@pytest.mark.parametrize("recipe", MALFORMED)
def test_http_malformed_recipe_refuses_before_endpoint_selection_validation(client, fake, monkeypatch, recipe, load):
    snapshot, effects = saved_snapshot(), []
    monkeypatch.setattr(model_selection, "validate", forbidden("selection validation", effects))
    response = client.post("/api/model/switch", json={"model_id": ANIMAGINE, "execution": recipe,
                                                     "load": load, "load_on_startup": True})
    assert response.status_code == 422 and effects == []
    assert_unchanged(snapshot, fake)


@pytest.mark.parametrize("payload", [
    {}, {"execution": None}, {"execution": {}},
    {"execution": {"compilation": None, "attention_backend": None, "step_cache": None}},
])
def test_http_eager_spellings_keep_ordinary_same_checkpoint_selection(client, fake, payload):
    before = client.get("/api/status").json()
    response = client.post("/api/model/switch", json={"model_id": before["model_id"], "load": False,
                                                     "load_on_startup": False, **payload})
    assert response.status_code == 200, response.text
    assert response.json()["model_revision"] == before["model_revision"]
    assert fake.loaded and fake.loads == 0
    saved = startup_preferences.load(main.settings.db_path)
    assert saved == startup_preferences.StartupPreferences(before["model_id"], before["quantization"], before["offload"], False)


def test_stale_header_still_refuses_switch_without_effects(client, fake):
    before = client.get("/api/status").json()
    snapshot = saved_snapshot()
    response = client.post("/api/model/switch", headers={"x-local-diffusion-model-revision": str(before["model_revision"] + 1)},
                           json={"model_id": ANIMAGINE, "execution": None, "load": False})
    assert response.status_code == 409
    assert_unchanged(snapshot, fake)
