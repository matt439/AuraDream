"""The load supervisor's cached ready state cannot hide a discarded pipeline.

These use the fake backend's readiness only. Public cached execution remains
unavailable; no private FBC path, model, compiler or GPU is exercised here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from backend import main, model_selection, startup_preferences


@pytest.fixture(autouse=True)
def idle_selection(fake, monkeypatch):
    monkeypatch.setattr(main, "_switching", False)
    monkeypatch.setattr(main, "_switch_draining", False)
    monkeypatch.setattr(main, "_switch_task", None)
    monkeypatch.setattr(main, "_model_requests", 0)
    monkeypatch.setattr(main, "_model_request_paths", {})
    monkeypatch.setattr(main, "_startup_preload", False)


def saved_snapshot():
    selected = model_selection.current_settings(main.settings)
    startup_preferences.save(main.settings.db_path, startup_preferences.StartupPreferences(
        selected.model_id, selected.quantization, selected.offload, False,
    ))
    path = startup_preferences.path_for(main.settings.db_path)
    return path, path.read_bytes(), model_selection._selection


def assert_selection_unchanged(snapshot):
    path, data, selection = snapshot
    assert path.read_bytes() == data
    assert model_selection._selection == selection
    assert not main._switching and main._switch_task is None
    assert main.jobs.active() == []


def settled(client):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get("/api/status")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["state"] == "ready" and not main._load_lock.locked():
            return body
        time.sleep(0.01)
    raise AssertionError(f"fake load did not finish: {body}")


@pytest.mark.parametrize("cached", ["unloaded", "ready"])
@pytest.mark.parametrize("ready", [False, True])
def test_status_reconciles_readiness_once_without_loading_or_mutating_cached_state(client, fake, monkeypatch, cached, ready):
    snapshot = saved_snapshot()
    main._load_state.update(state=cached)
    fake.loaded = ready
    before = dict(main._load_state)
    observations = []

    def actual_readiness():
        observations.append(ready)
        return ready

    monkeypatch.setattr(fake, "is_ready", actual_readiness)
    response = client.get("/api/status")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == ("ready" if ready else "unloaded")
    assert observations == [ready]
    assert main._load_state == before and fake.loads == 0
    assert_selection_unchanged(snapshot)


def test_explicit_load_leaves_an_actually_ready_backend_alone(client, fake):
    snapshot = saved_snapshot()
    main._load_state.update(state="ready", detail=None)
    assert fake.loaded
    response = client.post("/api/model/load")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "ready" and fake.loads == 0
    assert_selection_unchanged(snapshot)


@pytest.mark.parametrize("cached", ["loading", "error"])
@pytest.mark.parametrize("ready", [False, True])
def test_status_preserves_supervised_loading_or_error_and_its_details(client, fake, monkeypatch, cached, ready):
    snapshot = saved_snapshot()
    fake.loaded = ready
    main._load_state.update(state=cached, detail="the supervised outcome",
                            detail_kind="offline", detail_url="https://huggingface.co/example")
    before = dict(main._load_state)

    def unexpected_observation():
        pytest.fail("loading/error status must keep its supervised state")

    monkeypatch.setattr(fake, "is_ready", unexpected_observation)
    response = client.get("/api/status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["state"], body["detail"], body["detail_kind"], body["detail_url"]) == (
        cached, "the supervised outcome", "offline", "https://huggingface.co/example",
    )
    assert main._load_state == before and fake.loads == 0
    assert_selection_unchanged(snapshot)


@pytest.mark.parametrize("route", ["/api/model/load", "/api/model/switch"])
def test_explicit_load_recovers_lost_ready_without_a_preceding_status_get(client, fake, monkeypatch, route):
    snapshot = saved_snapshot()
    main._load_state.update(state="ready", detail=None)
    fake.loaded = False
    started, release = threading.Event(), threading.Event()
    load = fake.load

    def held_load():
        started.set()
        assert release.wait(15), "test did not release the synthetic load"
        load()

    monkeypatch.setattr(fake, "load", held_load)
    payload = {"model_id": main.settings.model_id, "load": True} if route.endswith("switch") else None
    try:
        response = client.post(route, json=payload)
        assert response.status_code == 200, response.text
        assert started.wait(10), "explicit Load did not start an ordinary load"
        assert response.json()["state"] == "loading"
        assert client.get("/api/status").json()["state"] == "loading"
        assert main._load_state["state"] == "loading" and not fake.loaded
    finally:
        release.set()
    body = settled(client)
    assert body["model_revision"] == snapshot[2][1] and fake.loads == 1 and fake.loaded
    assert_selection_unchanged(snapshot)


def test_concurrent_explicit_loads_share_one_backend_attempt_after_lost_ready(client, fake, monkeypatch):
    snapshot = saved_snapshot()
    main._load_state.update(state="ready", detail=None)
    fake.loaded = False
    started, release = threading.Event(), threading.Event()
    requests = threading.Barrier(2)
    attempts = []
    load = fake.load

    def held_load():
        attempts.append("load")
        started.set()
        assert release.wait(15), "test did not release the synthetic load"
        load()

    def post_load():
        requests.wait(timeout=10)
        return client.post("/api/model/load")

    monkeypatch.setattr(fake, "load", held_load)
    with ThreadPoolExecutor(2) as pool:
        try:
            pending = [pool.submit(post_load) for _ in range(2)]
            responses = [item.result(timeout=10) for item in pending]
            assert started.wait(10)
            assert all(response.status_code == 200 for response in responses)
            assert all(response.json()["state"] == "loading" for response in responses)
            assert client.get("/api/status").json()["state"] == "loading"
            assert attempts == ["load"] and fake.loads == 0
        finally:
            release.set()
    settled(client)
    assert attempts == ["load"] and fake.loads == 1
    assert_selection_unchanged(snapshot)


def test_explicit_reload_failure_retains_its_own_error_and_can_retry(client, fake, monkeypatch):
    snapshot = saved_snapshot()
    main._load_state.update(state="ready", detail=None)
    fake.loaded = False
    failed = threading.Event()

    def rejected_load():
        failed.set()
        raise RuntimeError("synthetic reload refusal")

    load = fake.load
    monkeypatch.setattr(fake, "load", rejected_load)
    assert client.post("/api/model/load").status_code == 200
    assert failed.wait(10)
    deadline = time.monotonic() + 10
    while main._load_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    body = client.get("/api/status").json()
    assert body["state"] == "error" and body["detail"] == "RuntimeError: synthetic reload refusal"
    assert not fake.loaded and fake.loads == 0
    monkeypatch.setattr(fake, "load", load)
    assert client.post("/api/model/load").status_code == 200
    assert settled(client)["detail"] is None
    assert fake.loads == 1
    assert_selection_unchanged(snapshot)
