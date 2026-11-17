"""Shared transient runtime, actual token counts and ticket cancellation, all stubbed."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pytest

from backend import assistant_models, critic, critic_runtime, critic_server
from backend.backends.base import Cancelled
from backend.schemas_assistant import PromptAssistantRequest
from tests.critic_helpers import Server, configure_critic, prepared  # noqa: F401 - shared fixture


FLUX = "black-forest-labs/FLUX.1-Krea-dev"
ANSWER = {
    "message": "Here is a draft.",
    "drafts": [{"title": "Forest", "prompt": "A red fox in a forest at dawn.", "notes": []}],
    "tags": [],
}


def request(**changes) -> PromptAssistantRequest:
    return PromptAssistantRequest.model_validate({
        "ticket": "runtime", "mode": "create", "instruction": "a red fox in a forest", **changes,
    })


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


@pytest.mark.parametrize("key", ["qwen3-8b", "qwen3.5-4b"])
def test_selected_preparation_keeps_the_critic_baseline_and_captures_verified_paths(prepared, monkeypatch, tmp_path, key):
    baseline = prepared._prepared
    profile = assistant_models.get_model(key)
    selected_model = tmp_path / f"{key}.gguf"
    selected_model.write_bytes(b"model fixture")
    downloads = []

    def download(model):
        downloads.append(model.key)
        return selected_model

    monkeypatch.setattr(type(profile), "download", download)
    captured = prepared.prepare_assistant(key)
    assert downloads == [key]
    assert captured.server == baseline[0] and captured.model == selected_model
    assert captured.label == profile.label and captured.profile is profile
    assert prepared._prepared == baseline


def test_default_preparation_captures_the_existing_override_label(prepared, monkeypatch):
    server, model = prepared._prepared
    configure_critic(monkeypatch, critic_model=model)
    captured = prepared.prepare_assistant()
    assert captured.server == server and captured.model == model
    assert captured.profile is None and captured.label == model.stem
    assert prepared._prepared == (server, model)


@pytest.mark.parametrize("key", ["qwen3-8b", "qwen3.5-4b"])
def test_selected_completion_uses_its_captured_file_and_profile_then_releases_it(prepared, monkeypatch, tmp_path, key):
    profile = assistant_models.get_model(key)
    server = Server(monkeypatch, ANSWER)
    selected_model = tmp_path / f"{key}.gguf"
    selected_model.write_bytes(b"model fixture")
    baseline = prepared._prepared
    captured = critic.PreparedAssistant(baseline[0], selected_model, profile.label, profile)
    exchange = critic_server._request
    counted_thinking = []

    def selected_template(method, url, api_key, body, timeout):
        if urlsplit(url).path == "/apply-template":
            counted_thinking.append(body["chat_template_kwargs"]["enable_thinking"])
            # The shared critic stub models a thinking-only template. Verify
            # this profile's actual request before adapting that fixture port.
            assert body["chat_template_kwargs"] == {"enable_thinking": profile.enable_thinking}
            body = {**body, "chat_template_kwargs": {"enable_thinking": True}}
        return exchange(method, url, api_key, body, timeout)

    monkeypatch.setattr(critic_server, "_request", selected_template)
    answer = prepared.assist(FLUX, request(assistant_model=key), lambda: False, captured)
    command = server.commands[0]
    assert command[command.index("-m") + 1] == str(selected_model.resolve())
    assert command[command.index("-c") + 1] == str(profile.context)
    assert command[command.index("--reasoning-budget") + 1] == str(profile.reasoning_budget)
    completion = next(call[3] for call in server.calls if call[1] == "/v1/chat/completions")
    assert completion["chat_template_kwargs"] == {"enable_thinking": profile.enable_thinking}
    assert {name: completion[name] for name in profile.sampling} == profile.sampling
    assert counted_thinking == [profile.enable_thinking]
    assert answer.model == profile.label and prepared._prepared == baseline
    assert server.process.terminated and prepared._process is None


@pytest.mark.parametrize("key,allowed", [("qwen3.5-4b", True), ("qwen3-8b", False)])
def test_actual_count_reserves_the_selected_models_thinking_budget(prepared, monkeypatch, tmp_path, key, allowed):
    profile = assistant_models.get_model(key)
    server = Server(monkeypatch, ANSWER)
    server.count = lambda text: 7000
    selected_model = tmp_path / f"{key}.gguf"
    selected_model.write_bytes(b"model fixture")
    captured = critic.PreparedAssistant(prepared._prepared[0], selected_model, profile.label, profile)
    exchange = critic_server._request

    def selected_template(method, url, api_key, body, timeout):
        if urlsplit(url).path == "/apply-template":
            assert body["chat_template_kwargs"] == {"enable_thinking": profile.enable_thinking}
            body = {**body, "chat_template_kwargs": {"enable_thinking": True}}
        return exchange(method, url, api_key, body, timeout)

    monkeypatch.setattr(critic_server, "_request", selected_template)
    if allowed:
        answer = prepared.assist(FLUX, request(assistant_model=key), lambda: False, captured)
        completion = next(call[3] for call in server.calls if call[1] == "/v1/chat/completions")
        # The non-thinking model keeps 1,000 answer tokens and can use this
        # long input; the remaining budget comes from the server's real count.
        assert answer.drafts and completion["max_tokens"] == 1192
    else:
        with pytest.raises(critic.CriticError, match="conversation is too long"):
            prepared.assist(FLUX, request(assistant_model=key), lambda: False, captured)
        assert not any(call[1] == "/v1/chat/completions" for call in server.calls)
    assert any(call[1] == "/tokenize" for call in server.calls)
    assert server.process.terminated and prepared._process is None


def test_assistant_catalogue_reads_only_local_file_state(prepared, monkeypatch, tmp_path):
    baseline = prepared._prepared
    monkeypatch.setattr(critic_runtime, "_model_cached", lambda: None)
    monkeypatch.setattr(type(assistant_models.MODELS[0]), "cached", lambda self: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("model listing attempted preparation, hashing or a child process")

    monkeypatch.setattr(prepared, "prepare", forbidden)
    monkeypatch.setattr(prepared, "prepare_assistant", forbidden)
    monkeypatch.setattr(critic_runtime, "_sha256", forbidden)
    monkeypatch.setattr(critic_server, "_spawn", forbidden)
    models = prepared.assistant_models()
    assert [model["id"] for model in models] == ["default", "qwen3-8b", "qwen3.5-4b"]
    assert not any(model["ready"] for model in models)
    assert all(model["fetch_bytes"] > 0 for model in models)
    assert prepared._prepared == baseline


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


def test_runtime_reports_starting_then_writing_without_changing_completion_or_cleanup(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)
    phases = []
    with critic_runtime.progress(lambda phase, done, total: phases.append((phase, done, total))):
        assert prepared.assist(FLUX, request(), lambda: False).drafts
    assert phases == [("starting", None, None), ("writing", None, None)]
    assert server.process.terminated and prepared._process is None
    critic_runtime.report_progress("starting")
    assert len(phases) == 2


def test_hub_progress_uses_real_transfer_counts_and_never_reaches_a_later_context(monkeypatch, tmp_path):
    import huggingface_hub

    target = tmp_path / "model.gguf"
    target.write_bytes(b"not real weights")
    monkeypatch.setattr(critic_runtime, "_model_cached", lambda *args: None)
    monkeypatch.setattr(critic_runtime, "verify_model", lambda *args, **kwargs: None)

    def fetch(**kwargs):
        with kwargs["tqdm_class"](total=10, initial=3, disable=True) as bar:
            bar.update(2)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    observations = []
    with critic_runtime.progress(lambda *value: observations.append(value)):
        assert critic_runtime.download_model(size=10) == target
    assert ("downloading", 3, 10) in observations and ("downloading", 5, 10) in observations
    assert observations[-1] == ("preparing", None, None)
    count = len(observations)
    critic_runtime.report_progress("writing")
    assert len(observations) == count


def test_broken_progress_observer_cannot_fail_assistant_or_process_cleanup(prepared, monkeypatch):
    server = Server(monkeypatch, ANSWER)

    def broken(phase, done, total):
        raise RuntimeError("presentation failed")

    with critic_runtime.progress(broken):
        assert prepared.assist(FLUX, request(), lambda: False).drafts
    assert server.process.terminated and prepared._process is None
