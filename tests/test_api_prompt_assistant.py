"""Prompt-authoring HTTP admission, cancellation and post-inference verification."""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import httpx

from backend import api_prompt_assistant, critic_runtime, main, model_selection, prompt_assistant, tag_lookup
from backend.critic import CriticError, PromptCritic
from tests.fakes import FakeCritic
from tests.api_analysis_helpers import Vocabulary
from tests.api_helpers import ORIGIN, stream_job, until


def payload(**changes):
    return {
        "ticket": uuid.uuid4().hex, "mode": "create", "instruction": "A red fox in a forest",
        **changes,
    }


def test_fake_assistant_signature_matches_the_shared_runtime():
    def shape(method):
        return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()]

    assert shape(PromptCritic.assist) == shape(FakeCritic.assist)


@pytest.mark.parametrize("assistant_model,label", [
    ("default", "fake critic"),
    ("qwen3-8b", "Qwen3-8B Q4_K_M"),
    ("qwen3.5-4b", "Qwen3.5-4B Q8_0"),
])
@pytest.mark.parametrize("mode", ["create", "improve", "tags"])
def test_all_modes_work_cold_without_loading_image_weights(client, fake, monkeypatch, mode, assistant_model, label):
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary({"ahoge": "general"}))
    before = fake.loads
    response = client.post("/api/prompt/assist", json=payload(
        mode=mode, prompt="A red fox", assistant_model=assistant_model,
    ))
    assert response.status_code == 200, response.text
    body = response.json()
    assert fake.loads == before and fake.requests == []
    assert len(fake.critic.assists) == 1 and fake.critic.prepares == 1
    assert body["model"] == label
    assert fake.critic.assistant_preparations[0][0] == assistant_model
    assert fake.critic.assists[0][1].assistant_model == assistant_model
    if mode == "tags":
        assert body["drafts"] == []
        assert [tag["status"] for tag in body["tags"]] == ["valid", "not_found", "not_found"]
    else:
        assert len(body["drafts"]) == 2 and body["tags"] == []
    assert main.jobs.card() == (None, None)


def test_tag_changes_serialize_from_the_real_answer_validator(client, fake, monkeypatch):
    def assist(model_id, request, should_cancel, prepared=None):
        return prompt_assistant.validate_answer(json.dumps({
            "message": "Choose a direction.", "tags": [], "drafts": [
                {"title": "Night", "prompt": "1girl, park, night"},
                {"title": "Prose", "prompt": "A girl stands in a park at night."},
            ],
        }), model_id, request, model="fake critic", seconds=0.1)

    monkeypatch.setattr(fake.critic, "assist", assist)
    response = client.post("/api/prompt/assist", json=payload(mode="improve", prompt="1girl, park, day, sunny"))
    assert response.status_code == 200, response.text
    assert response.json()["draft_changes"] == [{"added": ["night"], "removed": ["day", "sunny"]}, None]
    assert fake.loads == 0 and fake.requests == []


def test_model_catalogue_reads_state_without_preparing_or_loading(client, fake):
    response = client.get("/api/prompt/assist/models")
    assert response.status_code == 200, response.text
    models = response.json()["models"]
    assert [model["id"] for model in models] == ["default", "qwen3-8b", "qwen3.5-4b"]
    assert [model["label"] for model in models] == ["fake critic", "Qwen3-8B Q4_K_M", "Qwen3.5-4B Q8_0"]
    assert all(not model["ready"] and model["fetch_bytes"] > 0 for model in models)
    assert all(model["qualification"] == "unmeasured" for model in models)
    assert fake.critic.prepares == 0 and fake.critic.assists == []
    assert fake.loads == 0 and fake.requests == [] and main.jobs.card() == (None, None)


@pytest.mark.parametrize("assistant_model", ["other-model", "", "../../local.gguf", "Qwen3-8B"])
def test_unknown_assistant_model_refuses_before_preparing(client, fake, assistant_model):
    response = client.post("/api/prompt/assist", json=payload(assistant_model=assistant_model))
    assert response.status_code == 422, response.text
    assert fake.critic.prepares == 0 and fake.critic.assists == []


def test_selected_model_becomes_ready_without_replacing_the_critic_default(client, fake):
    response = client.post("/api/prompt/assist", json=payload(assistant_model="qwen3-8b"))
    assert response.status_code == 200, response.text
    models = {model["id"]: model for model in client.get("/api/prompt/assist/models").json()["models"]}
    assert models["qwen3-8b"]["ready"] and models["qwen3-8b"]["fetch"] == []
    assert not models["default"]["ready"] and not models["qwen3.5-4b"]["ready"]
    assert models["default"]["fetch"] == ["fake critic"]
    assert models["qwen3.5-4b"]["fetch"] == ["Qwen3.5-4B Q8_0"]
    assert fake.critic.describe().label == "fake critic" and not fake.critic.fetched


def test_follow_up_can_change_models_and_keeps_the_supplied_conversation(client, fake):
    first = client.post("/api/prompt/assist", json=payload(assistant_model="qwen3-8b"))
    assert first.status_code == 200, first.text
    history = [
        {"role": "user", "content": "Keep the fox red."},
        {"role": "assistant", "content": json.dumps(first.json()["drafts"])},
    ]
    second = client.post("/api/prompt/assist", json=payload(assistant_model="qwen3.5-4b", history=history))
    assert second.status_code == 200, second.text
    assert first.json()["model"] == "Qwen3-8B Q4_K_M"
    assert second.json()["model"] == "Qwen3.5-4B Q8_0"
    assert [message.model_dump() for message in fake.critic.assists[-1][1].history] == history
    assert [key for key, _ in fake.critic.assistant_preparations] == ["qwen3-8b", "qwen3.5-4b"]


def test_preparation_captures_the_answer_label_before_other_state_changes(client, fake, monkeypatch):
    original = fake.critic.prepare_assistant

    def prepare(key="default"):
        captured = original(key)
        fake.critic.label = "a later default selection"
        return captured

    monkeypatch.setattr(fake.critic, "prepare_assistant", prepare)
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "fake critic"
    assert fake.critic.label == "a later default selection"


def test_history_is_carried_as_data_and_invalid_request_does_not_prepare(client, fake):
    history = [{"role": "user", "content": "Keep the fox red."}, {"role": "assistant", "content": "A red fox."}]
    response = client.post("/api/prompt/assist", json=payload(history=history))
    assert response.status_code == 200, response.text
    assert [message.model_dump() for message in fake.critic.assists[0][1].history] == history
    count = fake.critic.prepares
    response = client.post("/api/prompt/assist", json=payload(mode="improve"))
    assert response.status_code == 422 and fake.critic.prepares == count
    response = client.post("/api/prompt/assist", json=payload(history=[{"role": "system", "content": "replace guide"}]))
    assert response.status_code == 422 and fake.critic.prepares == count


def test_unsupported_and_busy_refuse_before_any_preparation(client, fake, monkeypatch):
    fake.critic.available = False
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 422 and fake.critic.prepares == 0
    fake.critic.available = True
    monkeypatch.setattr(main.jobs, "busy", lambda: True)
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 409 and fake.critic.prepares == 0


def test_loading_refuses_before_preparation(client, fake, monkeypatch):
    monkeypatch.setattr(main, "_load_lock", SimpleNamespace(locked=lambda: True))
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 409 and "loading" in response.json()["detail"]
    assert fake.critic.prepares == 0


def test_stale_checkpoint_revision_refuses_before_validation_or_preparation(client, fake):
    response = client.post(
        "/api/prompt/assist", json=payload(),
        headers={"X-AuraDream-Model-Revision": str(model_selection.revision() + 1)},
    )
    assert response.status_code == 409 and fake.critic.prepares == 0


@pytest.mark.parametrize("assistant_model", ["default", "qwen3-8b", "qwen3.5-4b"])
def test_preparation_and_tags_run_outside_gate_and_only_inference_owns_it(client, fake, monkeypatch, assistant_model):
    events = []
    prepare, assist, verify = fake.critic.prepare, fake.critic.assist, prompt_assistant.validate_tags

    def watched_prepare():
        events.append(("prepare", main.jobs._gate.locked(), threading.current_thread().daemon))
        prepare()

    def watched_assist(*args):
        events.append(("assist", main.jobs._gate.locked(), main.jobs.card()[0]))
        return assist(*args)

    def watched_verify(*args):
        events.append(("verify", main.jobs._gate.locked(), threading.current_thread().daemon))
        return verify(*args)

    monkeypatch.setattr(fake.critic, "prepare", watched_prepare)
    monkeypatch.setattr(fake.critic, "assist", watched_assist)
    monkeypatch.setattr(prompt_assistant, "validate_tags", watched_verify)
    response = client.post("/api/prompt/assist", json=payload(assistant_model=assistant_model))
    assert response.status_code == 200, response.text
    assert events == [("prepare", False, True), ("assist", True, "critic"), ("verify", False, True)]


def test_job_started_during_prepare_refuses_inference(client, fake, monkeypatch):
    original = fake.critic.prepare

    def prepare():
        original()
        monkeypatch.setattr(main.jobs, "busy", lambda: True)

    monkeypatch.setattr(fake.critic, "prepare", prepare)
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 409 and fake.critic.prepares == 1 and fake.critic.assists == []


def test_stop_before_post_prevents_preparation(client, fake):
    body = payload()
    assert client.post(f"/api/prompt/assist/{body['ticket']}/stop").json() == {"stopped": True, "running": False}
    response = client.post("/api/prompt/assist", json=body)
    assert response.status_code == 409 and fake.critic.prepares == 0


@pytest.mark.parametrize("assistant_model", ["default", "qwen3-8b", "qwen3.5-4b"])
def test_stop_during_prepare_never_starts_the_llm(client, fake, monkeypatch, assistant_model):
    entered, release = threading.Event(), threading.Event()
    original = fake.critic.prepare

    def prepare():
        entered.set()
        assert release.wait(30)
        original()

    monkeypatch.setattr(fake.critic, "prepare", prepare)
    body = payload(assistant_model=assistant_model)
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(client.post, "/api/prompt/assist", json=body)
        try:
            assert entered.wait(30)
            stop = client.post(f"/api/prompt/assist/{body['ticket']}/stop")
            assert stop.json() == {"stopped": True, "running": True}
            # The download thread can still be blocked. Stop forgets its
            # waiter at once and cannot leave a continuation that loads later.
            assert waiting.result(timeout=30).status_code == 409
        finally:
            release.set()
    assert fake.critic.assists == [] and main.jobs.card() == (None, None)


def test_duplicate_and_unrelated_stop_cannot_replace_or_cancel_the_owned_call(client, fake):
    fake.critic.delay = 30
    body = payload()
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(client.post, "/api/prompt/assist", json=body)
        try:
            assert fake.critic.started.wait(30)
            assert main.jobs.card() == ("critic", body["ticket"])
            duplicate = client.post("/api/prompt/assist", json=body)
            assert duplicate.status_code == 409 and len(fake.critic.assists) == 1
            unrelated = client.post(f"/api/prompt/assist/{uuid.uuid4().hex}/stop")
            assert unrelated.json()["running"] is False and not waiting.done()
        finally:
            client.post(f"/api/prompt/assist/{body['ticket']}/stop")
        assert waiting.result(timeout=30).status_code == 409
    assert fake.critic.events[-1] == "assist done" and main.jobs.card() == (None, None)


@pytest.mark.parametrize("phase", ["prepare", "inference"])
def test_disconnected_request_cannot_start_later_or_release_an_active_worker_early(fake, monkeypatch, phase):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = fake.critic.prepare
    if phase == "prepare":
        def prepare():
            entered.set()
            try:
                assert release.wait(30)
                original()
            finally:
                finished.set()

        monkeypatch.setattr(fake.critic, "prepare", prepare)
    else:
        fake.critic.delay = 30

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as client:
                request = asyncio.create_task(client.post("/api/prompt/assist", json=payload()))
                started = entered if phase == "prepare" else fake.critic.started
                assert await asyncio.to_thread(started.wait, 30)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(request, 30)
                assert main.jobs.card() == (None, None)
                if phase == "prepare":
                    assert fake.critic.assists == [] and not finished.is_set()
                    release.set()
                    assert await asyncio.to_thread(finished.wait, 30)
                    assert fake.critic.assists == []
                else:
                    assert fake.critic.events[-1] == "assist done"
        finally:
            release.set()
            await main.jobs.shutdown()

    asyncio.run(scenario())


def test_image_queued_behind_assistant_starts_only_after_stop_releases_gate(client, fake):
    fake.critic.delay = 30
    body = payload()
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(client.post, "/api/prompt/assist", json=body)
        try:
            assert fake.critic.started.wait(30)
            generated = client.post("/api/generate", json={"prompt": "a fox", "width": 256, "height": 256})
            assert generated.status_code == 200, generated.text
            job_id = generated.json()["job_id"]
            until(lambda: client.get(f"/api/jobs/{job_id}").json()["waiting_on"] == "critic")
            assert fake.requests == []
        finally:
            client.post(f"/api/prompt/assist/{body['ticket']}/stop")
        assert waiting.result(timeout=30).status_code == 409
    assert stream_job(client, job_id)[-1]["status"] == "done"
    assert len(fake.requests) == 1


def test_download_and_runtime_errors_keep_existing_failure_statuses(client, fake, monkeypatch):
    fake.critic.prepare_fail = CriticError("download checksum did not match")
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 502 and response.json()["detail"] == "download checksum did not match"
    assert fake.critic.assists == []
    fake.critic.prepare_fail = None
    fake.critic.fail = CriticError("the assistant returned malformed JSON")
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 502 and main.jobs.card() == (None, None)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, critic=False))
    fake.critic.available = False
    response = client.post("/api/prompt/assist", json=payload())
    assert response.status_code == 422 and "switched off" in response.json()["detail"]


def test_status_reads_only_recent_ticket_metadata_without_preparing(client, fake):
    assert client.get("/api/prompt/assist/unknown-status-ticket/status").status_code == 404
    body = payload(instruction="private instruction that must never appear in progress")
    assert client.post("/api/prompt/assist", json=body).status_code == 200
    status = client.get(f"/api/prompt/assist/{body['ticket']}/status")
    assert status.status_code == 200 and status.json() == {
        "phase": "complete", "running": False, "downloaded_bytes": None, "total_bytes": None,
    }
    prepares = fake.critic.prepares
    for _ in range(3):
        assert client.get(f"/api/prompt/assist/{body['ticket']}/status").json() == status.json()
    assert fake.critic.prepares == prepares
    assert "private" not in status.text
    assert client.post("/api/prompt/assist", json=body).status_code == 409


def test_progress_reports_actual_download_then_writing_and_owned_stop(client, fake, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = fake.critic.prepare_assistant

    def prepare(key="default"):
        critic_runtime.report_progress("downloading", 37, 100)
        entered.set()
        assert release.wait(30)
        return original(key)

    monkeypatch.setattr(fake.critic, "prepare_assistant", prepare)
    fake.critic.delay = 30
    body = payload()
    endpoint = f"/api/prompt/assist/{body['ticket']}/status"
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(client.post, "/api/prompt/assist", json=body)
        try:
            assert entered.wait(30)
            assert client.get(endpoint).json() == {
                "phase": "downloading", "running": True, "downloaded_bytes": 37, "total_bytes": 100,
            }
            assert client.get("/api/prompt/assist/unrelated-progress/status").status_code == 404
            release.set()
            assert fake.critic.started.wait(30)
            until(lambda: client.get(endpoint).json()["phase"] == "writing")
        finally:
            release.set()
            client.post(f"/api/prompt/assist/{body['ticket']}/stop")
        assert waiting.result(timeout=30).status_code == 409
    assert client.get(endpoint).json() == {
        "phase": "stopped", "running": False, "downloaded_bytes": None, "total_bytes": None,
    }
    assert main.jobs.card() == (None, None)


def test_verification_progress_is_after_child_releases_gate(client, fake, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def verify(model_id, response):
        assert main.jobs.card() == (None, None)
        entered.set()
        assert release.wait(30)
        return response

    monkeypatch.setattr(prompt_assistant, "validate_tags", verify)
    body = payload(mode="tags")
    endpoint = f"/api/prompt/assist/{body['ticket']}/status"
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(client.post, "/api/prompt/assist", json=body)
        try:
            assert entered.wait(30)
            assert client.get(endpoint).json()["phase"] == "verifying"
        finally:
            release.set()
        assert waiting.result(timeout=30).status_code == 200
    assert client.get(endpoint).json()["phase"] == "complete"


def test_failed_and_stop_before_post_statuses_remain_terminal(client, fake):
    failed = payload()
    fake.critic.fail = CriticError("synthetic runtime failure")
    assert client.post("/api/prompt/assist", json=failed).status_code == 502
    assert client.get(f"/api/prompt/assist/{failed['ticket']}/status").json()["phase"] == "failed"
    early = payload()
    client.post(f"/api/prompt/assist/{early['ticket']}/stop")
    assert client.get(f"/api/prompt/assist/{early['ticket']}/status").json() == {
        "phase": "stopped", "running": False, "downloaded_bytes": None, "total_bytes": None,
    }
    assert client.post("/api/prompt/assist", json=early).status_code == 409


def test_late_preparation_progress_cannot_overwrite_stopped_ticket_or_next_request(client, fake, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = fake.critic.prepare_assistant

    def prepare(key="default"):
        if not entered.is_set():
            entered.set()
            try:
                assert release.wait(30)
                critic_runtime.report_progress("downloading", 99, 100)
            finally:
                finished.set()
        return original(key)

    monkeypatch.setattr(fake.critic, "prepare_assistant", prepare)
    old, later = payload(), payload()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(client.post, "/api/prompt/assist", json=old)
        try:
            assert entered.wait(30)
            client.post(f"/api/prompt/assist/{old['ticket']}/stop")
            assert pending.result(timeout=30).status_code == 409
            assert client.post("/api/prompt/assist", json=later).status_code == 200
        finally:
            release.set()
        assert finished.wait(30)
    assert client.get(f"/api/prompt/assist/{old['ticket']}/status").json()["phase"] == "stopped"
    assert client.get(f"/api/prompt/assist/{later['ticket']}/status").json()["phase"] == "complete"


def test_progress_tombstones_are_bounded_and_expire(monkeypatch):
    from fastapi import HTTPException

    clock = [0.0]
    monkeypatch.setattr(api_prompt_assistant, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    router = api_prompt_assistant.create_router(main.services)
    stop = next(route.endpoint for route in router.routes if route.path.endswith("/stop"))
    status = next(route.endpoint for route in router.routes if route.path.endswith("/status"))

    async def scenario():
        for index in range(257):
            clock[0] += 0.1
            await stop(f"bounded-{index}")
        with pytest.raises(HTTPException) as absent:
            await status("bounded-0")
        assert absent.value.status_code == 404
        assert (await status("bounded-256"))["phase"] == "stopped"
        clock[0] += 301
        with pytest.raises(HTTPException) as expired:
            await status("bounded-256")
        assert expired.value.status_code == 404

    asyncio.run(scenario())


@pytest.mark.parametrize("reserved,value", [
    ("_switching", True), ("_switch_draining", True), ("_deleting_model", "another checkpoint"),
])
def test_ticket_status_is_model_independent_during_switch_and_stale_revision(client, fake, monkeypatch, reserved, value):
    body = payload()
    assert client.post("/api/prompt/assist", json=body).status_code == 200
    before = main._model_requests
    paths = main._model_request_paths.copy()
    snapshots = []
    original = api_prompt_assistant._AssistantRun.status

    def status(run, running):
        snapshots.append((main._model_requests, main._model_request_paths.copy()))
        return original(run, running)

    monkeypatch.setattr(api_prompt_assistant._AssistantRun, "status", status)
    monkeypatch.setattr(main, reserved, value)
    response = client.get(
        f"/api/prompt/assist/{body['ticket']}/status",
        headers={"x-auradream-model-revision": str(model_selection.revision() - 1)},
    )
    assert response.status_code == 200 and response.json()["phase"] == "complete"
    assert snapshots == [(before, paths)]
    assert main._model_requests == before and main._model_request_paths == paths
    # Exempt only the exact read-only status shape, never adjacent prompt
    # routes, model listing, a deeper URL or a state-changing request.
    assert client.get("/api/prompt/assist/models").status_code == 409
    assert client.get(f"/api/prompt/assist/{body['ticket']}/status/extra").status_code == 409
    assert client.post("/api/prompt/assist", json=payload()).status_code == 409
