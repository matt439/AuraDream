"""Shared transient runtime, actual token counts and ticket cancellation, all stubbed."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pytest

from backend import critic, critic_server
from backend.backends.base import Cancelled
from backend.schemas_assistant import PromptAssistantRequest
from tests.critic_helpers import Server, prepared  # noqa: F401 - shared fixture


FLUX = "black-forest-labs/FLUX.1-Krea-dev"
ANSWER = {
    "message": "Here is a draft.",
    "drafts": [{"title": "Forest", "prompt": "A red fox in a forest at dawn.", "notes": []}],
    "tags": [],
}


def request() -> PromptAssistantRequest:
    return PromptAssistantRequest(ticket="runtime", mode="create", instruction="a red fox in a forest")


def test_assistant_uses_authenticated_local_runtime_and_releases_before_return(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    answer = prepared.assist(FLUX, request(), lambda: False)
    assert answer.drafts[0].prompt == ANSWER["drafts"][0]["prompt"]
    assert answer.model == critic.MODEL_LABEL
    completion = [call for call in server.calls if call[1] == "/v1/chat/completions"]
    assert len(completion) == 1 and completion[0][2] == server.keys[0]
    body = completion[0][3]
    assert body["response_format"]["json_schema"]["name"] == "prompt_assistant"
    assert 3048 <= body["max_tokens"] < 8192
    assert prepared._process is None
    assert server.process.terminated and server.process.poll() is not None


def test_assistant_cancellation_before_spawn_starts_nothing(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    with pytest.raises(Cancelled, match="stopped"):
        prepared.assist(FLUX, request(), lambda: True)
    assert server.commands == [] and server.calls == []


def test_assistant_cancelled_at_spawn_stops_its_process_without_sending_prompt(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    cancelled = threading.Event()
    spawn = critic_server._spawn

    def cancel_at_spawn(*args):
        process = spawn(*args)
        cancelled.set()
        return process

    monkeypatch.setattr(critic_server, "_spawn", cancel_at_spawn)
    with pytest.raises(Cancelled, match="stopped"):
        prepared.assist(FLUX, request(), cancelled.is_set)
    assert not any(call[2] for call in server.calls), "neither prompt nor key sent after Stop"
    assert server.process.terminated and prepared._process is None


def test_assistant_cancelled_during_count_never_submits_a_completion(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    cancelled = threading.Event()
    exchange = critic_server._request

    def cancel_on_count(method, url, key, body, timeout):
        response = exchange(method, url, key, body, timeout)
        if urlsplit(url).path == "/apply-template":
            cancelled.set()
        return response

    monkeypatch.setattr(critic_server, "_request", cancel_on_count)
    with pytest.raises(Cancelled, match="stopped"):
        prepared.assist(FLUX, request(), cancelled.is_set)
    assert not any(call[1] == "/v1/chat/completions" for call in server.calls)
    assert server.process.terminated and prepared._process is None


def test_stop_interrupts_the_blocked_completion_and_monitor_cannot_stop_the_next_call(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    cancelled, answering = threading.Event(), threading.Event()
    exchange = critic_server._request

    def held_answer(method, url, key, body, timeout):
        if urlsplit(url).path != "/v1/chat/completions":
            return exchange(method, url, key, body, timeout)
        answering.set()
        deadline = time.monotonic() + 30
        while not server.process.terminated:
            assert time.monotonic() < deadline, "Stop did not interrupt the blocked child request"
            time.sleep(0.01)
        raise ConnectionError("child stopped")

    monkeypatch.setattr(critic_server, "_request", held_answer)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(prepared.assist, FLUX, request(), cancelled.is_set)
        assert answering.wait(30)
        cancelled.set()
        with pytest.raises(Cancelled, match="stopped"):
            pending.result(timeout=30)
    assert server.process.terminated and prepared._process is None
    monkeypatch.setattr(critic_server, "_request", exchange)
    # The old predicate remains True: its watcher must have joined, and it
    # cannot stop a subsequent invocation sharing this same PromptCritic.
    assert prepared.assist(FLUX, request(), lambda: False).drafts
    assert len(server.commands) == 2


def test_assistant_refuses_the_actual_window_before_completion(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    server.count = lambda text: 7000
    with pytest.raises(critic.CriticError, match="conversation is too long"):
        prepared.assist(FLUX, request(), lambda: False)
    assert not any(call[1] == "/v1/chat/completions" for call in server.calls)
    assert server.process.terminated and prepared._process is None


def test_even_parseable_json_is_withheld_when_the_runtime_reports_truncation(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    exchange = critic_server._request

    def ran_out(method, url, key, body, timeout):
        status, response = exchange(method, url, key, body, timeout)
        if urlsplit(url).path == "/v1/chat/completions":
            response["choices"][0]["finish_reason"] = "length"
        return status, response

    monkeypatch.setattr(critic_server, "_request", ran_out)
    with pytest.raises(critic.CriticError, match="ran out of room"):
        prepared.assist(FLUX, request(), lambda: False)
    assert server.process.terminated and prepared._process is None
