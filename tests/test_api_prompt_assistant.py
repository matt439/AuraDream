"""Prompt-authoring HTTP admission, cancellation and post-inference verification."""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import httpx

from backend import main, model_selection, prompt_assistant, tag_lookup
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


@pytest.mark.parametrize("mode", ["create", "improve", "tags"])
def test_all_modes_work_cold_without_loading_image_weights(client, fake, monkeypatch, mode):
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary({"ahoge": "general"}))
    before = fake.loads
    response = client.post("/api/prompt/assist", json=payload(mode=mode, prompt="A red fox"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert fake.loads == before and fake.requests == []
    assert len(fake.critic.assists) == 1 and fake.critic.prepares == 1
    assert body["model"] == "fake critic"
    if mode == "tags":
        assert body["drafts"] == []
        assert [tag["status"] for tag in body["tags"]] == ["valid", "not_found", "not_found"]
    else:
        assert len(body["drafts"]) == 2 and body["tags"] == []
    assert main.jobs.card() == (None, None)


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


def test_preparation_and_tags_run_outside_gate_and_only_inference_owns_it(client, fake, monkeypatch):
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
    response = client.post("/api/prompt/assist", json=payload())
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


def test_stop_during_prepare_never_starts_the_llm(client, fake, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = fake.critic.prepare

    def prepare():
        entered.set()
        assert release.wait(30)
        original()

    monkeypatch.setattr(fake.critic, "prepare", prepare)
    body = payload()
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
