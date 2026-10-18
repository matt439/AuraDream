"""Stop abandons an answer, never the actual worker's GPU ownership."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend import api_prompt, main
from backend.schemas import PromptExpansionRequest
from tests.api_helpers import stream_job


def until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("worker did not reach its expected boundary")


@pytest.mark.parametrize("stage", ["download", "rewrite", "translation"])
@pytest.mark.parametrize("fail", [False, True])
def test_stop_returns_at_once_and_discards_each_stage_and_its_late_failure(
    client, fake, monkeypatch, stage, fail,
):
    fake.prompt_expander_supported = True
    entered, release = threading.Event(), threading.Event()
    expand, translate = fake.expand_prompt, fake.translator.translate
    ticket = f"stop-{stage}-{fail}"

    def pause():
        entered.set()
        assert release.wait(5)
        if fail:
            raise RuntimeError("opaque stage failed after Stop")

    def rewriting(prompt, width, height, should_cancel):
        if prompt != "first":
            return expand(prompt, width, height, should_cancel)
        if stage == "download":
            pause()
        result = expand(prompt, width, height, should_cancel)
        if stage == "rewrite":
            pause()
        return result

    def translating(text, should_cancel):
        if stage == "translation" and "first" in text:
            pause()
        return translate(text, should_cancel)

    monkeypatch.setattr(fake, "expand_prompt", rewriting)
    monkeypatch.setattr(fake.translator, "translate", translating)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(client.post, "/api/prompt/expand", json={"prompt": "first", "ticket": ticket})
        try:
            assert entered.wait(5)
            duplicate = client.post("/api/prompt/expand", json={"prompt": "duplicate", "ticket": ticket})
            assert duplicate.status_code == 409 and "already running" in duplicate.json()["detail"]
            assert client.post(f"/api/prompt/expand/{ticket}/stop").json() == {"stopped": True, "running": True}
            response = pending.result(timeout=2)
            assert response.status_code == 409 and response.json()["detail"] == "prompt expansion stopped"
            # The HTTP answer is gone already. The actual opaque GPU stage
            # still holds its ticket and a generation must wait behind it.
            if stage != "translation":
                assert client.get("/api/status").json()["card_ticket"] == ticket
                queued = client.post("/api/generate", json={"prompt": "queued", "width": 256, "height": 256}).json()
                assert fake.requests == []
            else:
                assert main.jobs.card() == (None, None)  # translation is CPU-only
        finally:
            release.set()
        if stage != "translation":
            stream_job(client, queued["job_id"])
        assert fake.translator.translations == []
        next_result = client.post("/api/prompt/expand", json={"prompt": "second", "ticket": ticket + "-next"})
        assert next_result.status_code == 200
        assert next_result.json()["expanded_prompt"].startswith("Expanded: second")


def test_stop_while_waiting_never_starts_that_rewrite(client, fake, monkeypatch):
    fake.prompt_expander_supported = True
    entered, release = threading.Event(), threading.Event()
    original = fake.expand_prompt

    def rewriting(prompt, width, height, should_cancel):
        if prompt == "first":
            entered.set()
            assert release.wait(5)
        return original(prompt, width, height, should_cancel)

    monkeypatch.setattr(fake, "expand_prompt", rewriting)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(client.post, "/api/prompt/expand", json={"prompt": "first", "ticket": "holder"})
        assert entered.wait(5)
        second = pool.submit(client.post, "/api/prompt/expand", json={"prompt": "waiting", "ticket": "waiter"})
        try:
            until(lambda: main._model_request_paths.get("/api/prompt/expand", 0) == 2)
            assert client.post("/api/prompt/expand/waiter/stop").status_code == 200
            assert second.result(timeout=2).status_code == 409
            assert main.jobs.card() == ("expand", "holder")
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
    assert [call[0] for call in fake.expansions] == ["first"]


def test_stop_can_overtake_the_post_and_tickets_cannot_be_reused(client, fake):
    fake.prompt_expander_supported = True
    assert client.post("/api/prompt/expand/not-arrived/stop").json() == {"stopped": True, "running": False}
    for _ in range(2):
        result = client.post("/api/prompt/expand", json={"prompt": "late", "ticket": "not-arrived"})
        assert result.status_code == 409 and result.json()["detail"] == "prompt expansion stopped"
    assert fake.expansions == [] and fake.loads == 0


def test_stop_during_the_model_load_keeps_the_load_lock(client, fake, monkeypatch):
    fake.prompt_expander_supported = True
    fake.loaded = False
    entered, release = threading.Event(), threading.Event()
    original = fake.load

    def load():
        entered.set()
        assert release.wait(5)
        original()

    monkeypatch.setattr(fake, "load", load)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(client.post, "/api/prompt/expand", json={"prompt": "load", "ticket": "loading"})
        try:
            assert entered.wait(5)
            assert client.post("/api/prompt/expand/loading/stop").status_code == 200
            assert pending.result(timeout=2).status_code == 409
            assert main._load_lock.locked()
            assert client.get("/api/status").json()["state"] == "loading"
        finally:
            release.set()
        until(lambda: not main._load_lock.locked())
    assert fake.expansions == [] and fake.loads == 1


def test_a_cancelled_request_keeps_its_detached_gpu_worker(fake):
    fake.prompt_expander_supported = True
    entered, release = threading.Event(), threading.Event()
    original = fake.expand_prompt

    def rewriting(prompt, width, height, should_cancel):
        entered.set()
        assert release.wait(5)
        return original(prompt, width, height, should_cancel)

    fake.expand_prompt = rewriting
    router = api_prompt.create_router(main.services)
    endpoint = next(route.endpoint for route in router.routes if getattr(route, "path", None) == "/api/prompt/expand")

    async def scenario():
        pending = asyncio.create_task(endpoint(PromptExpansionRequest(prompt="disconnect", ticket="disconnected")))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.01)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert main.jobs.card() == ("expand", "disconnected")
        finally:
            release.set()
        await main.jobs.wait_for_gate()
        await main.jobs.shutdown()

    asyncio.run(scenario())
    assert fake.expansions == [] and fake.translator.translations == []
