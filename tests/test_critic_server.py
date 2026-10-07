"""Critic server command, HTTP exchange, error handling, and cleanup."""
from __future__ import annotations

from pathlib import Path
import logging
import re

import pytest

from backend import config, critic, critic_context, critic_runtime, critic_server

from tests.critic_helpers import (
    FakeProcess,
    PROMPT,
    Server,
    _configure,
    _critique,
    _flag,
    _link_directory,
    prepared,
)


def test_server_command_binds_loopback_and_keeps_the_key_off_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = critic_server.server_command(Path("bin/llama-server.exe"), Path("m.gguf"), 5123)
    # Absolute: the server starts in its own directory, where a relative
    # model path would name another file.
    assert command[0] == str((tmp_path / "bin" / "llama-server.exe").resolve())
    assert _flag(command, "-m") == str((tmp_path / "m.gguf").resolve())
    assert _flag(command, "--host") == "127.0.0.1"
    assert _flag(command, "--port") == "5123"
    # Any local user can read a command line (/proc/<pid>/cmdline under WSL);
    # the key goes in the server's environment instead.
    assert "--api-key" not in command
    assert _flag(command, "-c") == "8192"
    assert _flag(command, "-np") == "1"
    assert _flag(command, "--fit") == "on"
    assert _flag(command, "--reasoning-budget") == str(critic_runtime.REASONING_BUDGET)
    for flag in ("--no-ui", "--jinja", "--offline"):
        assert flag in command


def test_server_env_drops_llama_settings_and_carries_the_key(monkeypatch):
    # A user's settings for their own servers, each of which would change
    # what the critic's does: shell and file tools, another chat template, a
    # key from a file, another model, a second slot, the web UI back on.
    for name, value in {
        "LLAMA_ARG_HOST": "0.0.0.0",
        "LLAMA_ARG_TOOLS": "all",
        "LLAMA_ARG_CHAT_TEMPLATE_FILE": "/tmp/template.jinja",
        "LLAMA_API_KEY_FILE": "/tmp/keys.txt",
        "LLAMA_ARG_MODEL": "/tmp/other.gguf",
        "LLAMA_ARG_N_PARALLEL": "4",
        "LLAMA_ARG_WEBUI": "1",
        "LLAMA_API_KEY": "shared",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LOCAL_DIFFUSION_KEEP", "1")
    env = critic_server._server_env(Path("bin/llama-server"), "k3y")
    assert [name for name in env if name.upper().startswith("LLAMA_")] == ["LLAMA_API_KEY"]
    # b11206 reads --api-key from LLAMA_API_KEY: this start's, never the user's.
    assert env["LLAMA_API_KEY"] == "k3y"
    assert env["LOCAL_DIFFUSION_KEEP"] == "1"


def test_configured_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_MODEL", "rel.gguf")
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_SERVER", "  ")
    model = config._env_path("LOCAL_DIFFUSION_CRITIC_MODEL")
    assert model.is_absolute() and model == tmp_path / "rel.gguf"
    assert config._env_path("LOCAL_DIFFUSION_CRITIC_SERVER") is None
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_SERVER", "~/llama-server")
    assert config._env_path("LOCAL_DIFFUSION_CRITIC_SERVER").is_absolute()

    # Absolute, not resolved: a path through a link stays the one given.
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "Qwen3-4B-Q8_0.gguf").write_bytes(b"gguf")
    _link_directory(tmp_path / "snapshot", blobs)
    given = tmp_path / "snapshot" / "Qwen3-4B-Q8_0.gguf"
    assert given.resolve() != given
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_MODEL", str(given))
    assert config._env_path("LOCAL_DIFFUSION_CRITIC_MODEL") == given


def test_a_linked_model_is_labelled_by_the_name_it_was_given(tmp_path, monkeypatch):
    # A Hub snapshot's entry links to a blob named by its sha256.
    blob = tmp_path / "blobs" / critic_runtime.MODEL_SHA256
    blob.parent.mkdir()
    blob.write_bytes(b"gguf")
    given = tmp_path / "Qwen3-4B-Q8_0.gguf"
    try:
        given.symlink_to(blob)
    except OSError:
        pytest.skip("this user can make no file symlink")
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_MODEL", str(given))
    _configure(monkeypatch, critic_model=config._env_path("LOCAL_DIFFUSION_CRITIC_MODEL"))
    assert critic.prompt_critic._label(critic.settings.critic_model) == "Qwen3-4B-Q8_0"


def test_a_critique_asks_a_keyed_server_and_stops_it(monkeypatch, prepared):
    server = Server(monkeypatch)
    result = _critique(prepared)

    key = server.keys[0]
    assert len(key) >= 32 and key not in server.commands[0]
    assert [(method, path) for method, path, _, _ in server.calls] == [
        ("GET", "/health"), ("GET", "/v1/models"),
        ("POST", "/apply-template"), ("POST", "/tokenize"), ("POST", "/tokenize"),
        ("POST", "/v1/chat/completions"),
    ]
    # The key-less probe must be refused; the port's listener is checked
    # next, before anything carries the key - the count, then the question.
    assert server.calls[1][2] is None
    assert [api_key for _, _, api_key, _ in server.calls[2:]] == [key] * 4
    assert server.checked_after == [2]
    # The messages as the server's template renders them, its control tokens
    # read as such; the prompt as the text the answer writes.
    rendered = server.calls[3][3]
    assert rendered["content"].endswith("<|im_start|>assistant\n")
    assert rendered["add_special"] and rendered["parse_special"]
    assert server.calls[4][3] == {"content": PROMPT, "add_special": False, "parse_special": False}
    body = server.calls[5][3]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert {k: body[k] for k in critic_context.SAMPLING} == critic_context.SAMPLING
    assert body["max_tokens"] == critic_context.MIN_TOKENS
    schema = body["response_format"]["json_schema"]["schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert "G3" in schema["properties"]["issues"]["items"]["properties"]["cite_id"]["enum"]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert PROMPT in body["messages"][1]["content"]
    assert server.calls[2][3]["messages"] == body["messages"]

    # Terminated and waited for, while its log was still there to read.
    assert server.process.events == ["terminate", "wait"] and prepared._process is None
    assert server.process.log_at_wait == [True] and not Path(server.logs[0]).exists()
    assert result.model == critic_runtime.MODEL_LABEL and result.seconds >= 0
    assert result.issues[0].cite_id == "G3" and result.issues[0].source == "guide"

    # A fresh key and a fresh server every time.
    _critique(prepared)
    assert server.keys[1] != key


def test_the_server_is_stopped_when_the_request_fails(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.post_error = TimeoutError("timed out")
    with pytest.raises(critic_runtime.CriticError, match="did not answer"):
        _critique(prepared)
    assert server.process.terminated and prepared._process is None


def test_an_early_exit_reports_the_server_output(monkeypatch, prepared):
    Server(monkeypatch, exits=1, output=b"load_model: failed\nerror: out of memory\n")
    with pytest.raises(critic_runtime.CriticError, match="code 1.*out of memory"):
        _critique(prepared)


def test_a_server_that_never_gets_healthy_is_stopped(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.health = 503
    monkeypatch.setattr(critic_server, "HEALTH_TIMEOUT", 0.0)
    with pytest.raises(critic_runtime.CriticError, match="not ready"):
        _critique(prepared)
    assert server.process.terminated
    assert all(path == "/health" for _, path, _, _ in server.calls)


def test_a_server_answering_without_its_key_is_not_asked(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.open_without_key = True
    with pytest.raises(critic_runtime.CriticError, match="without its API key"):
        _critique(prepared)
    assert server.process.terminated
    assert not any(path == "/v1/chat/completions" for _, path, _, _ in server.calls)


def test_critique_needs_prepare(monkeypatch):
    _configure(monkeypatch, critic=True)
    with pytest.raises(critic_runtime.CriticUnavailable, match="not prepared"):
        _critique(critic.PromptCritic())


def test_unload_stops_a_running_server():
    target = critic.PromptCritic()
    process = FakeProcess()
    target._process = process
    target.unload()
    assert process.terminated
    target._process = None
    target.unload()


def test_an_answer_cut_off_by_the_token_limit_says_so(monkeypatch, prepared):
    server = Server(monkeypatch, answer='{"parts": {"subject": "a wo')
    original = server.request

    def truncated(method, url, api_key, body, timeout):
        status, payload = original(method, url, api_key, body, timeout)
        if isinstance(payload, dict) and "choices" in payload:
            payload["choices"][0]["finish_reason"] = "length"
        return status, payload

    monkeypatch.setattr(critic_server, "_request", truncated)
    with pytest.raises(critic_runtime.CriticError, match="ran out of tokens.*shorter prompt"):
        _critique(prepared)
    assert server.process.terminated


def test_a_server_that_outlasts_terminate_is_killed_and_waited_for():
    process = FakeProcess(stubborn=1)
    critic_server._stop(process)
    assert process.events == ["terminate", "wait", "kill", "wait"]
    assert process.returncode == -9


def test_a_server_that_survives_its_kill_is_reported_and_forgotten(monkeypatch, prepared):
    server = Server(monkeypatch, stubborn=2)
    with pytest.raises(critic_runtime.CriticError, match="had not exited .* after it was killed"):
        _critique(prepared)
    assert server.process.events == ["terminate", "wait", "kill", "wait"]
    # The card may still be taken, which the error says; but the critic does
    # not keep the process as its own to stop, and the log goes regardless.
    assert prepared._process is None
    assert not Path(server.logs[0]).exists()


def _held_log(monkeypatch, attempts: list[str], free_after: int | None = None):
    """Path.unlink failing on the server's log as Windows fails it while
    anything has the file open, for `free_after` attempts (for good if None)."""
    unlink = Path.unlink

    def held(self, missing_ok=False):
        if self.name.startswith("llama-server-"):
            attempts.append(self.name)
            if free_after is None or len(attempts) <= free_after:
                raise PermissionError(32, "The process cannot access the file", str(self))
        return unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held)


def test_a_log_that_cannot_be_removed_does_not_cost_the_answer(monkeypatch, prepared, caplog):
    server = Server(monkeypatch)
    attempts: list[str] = []
    _held_log(monkeypatch, attempts)
    with caplog.at_level(logging.WARNING, logger=critic_server.log.name):
        result = _critique(prepared)
    monkeypatch.undo()
    assert result.issues[0].cite_id == "G3" and prepared._process is None
    assert "could not remove llama-server's log" in caplog.text and len(attempts) == 5
    Path(server.logs[0]).unlink()


def test_a_log_held_a_moment_is_removed_once_it_is_free(monkeypatch, prepared, caplog):
    # A scanner or an indexer opening the just-closed file, as `save_image`'s
    # rename meets it: tried again, the log goes.
    server = Server(monkeypatch)
    attempts: list[str] = []
    _held_log(monkeypatch, attempts, free_after=2)
    with caplog.at_level(logging.WARNING, logger=critic_server.log.name):
        _critique(prepared)
    assert len(attempts) == 3 and not Path(server.logs[0]).exists()
    assert "could not remove" not in caplog.text


def test_a_server_that_cannot_be_started_is_a_critic_error(prepared):
    # `prepared`'s server is a one-byte file, which no platform will run:
    # WinError 216 on Windows, a permission error elsewhere.
    with pytest.raises(critic_runtime.CriticError, match="could not be started") as caught:
        _critique(prepared)
    assert f"LOCAL_DIFFUSION_CRITIC_SERVER={critic.settings.critic_server}" in str(caught.value)
    assert prepared._process is None


@pytest.mark.parametrize(
    ("library", "package"),
    [("libgomp.so.1", "libgomp1"), ("libssl.so.3", "libssl3"), ("libcrypto.so.3", "libssl3")],
)
def test_a_missing_library_names_the_package_that_installs_it(
    monkeypatch, prepared, library, package
):
    Server(
        monkeypatch,
        exits=127,
        output=(
            f"/opt/llama/llama-server: error while loading shared libraries: {library}: "
            "cannot open shared object file: No such file or directory\n"
        ).encode(),
    )
    with pytest.raises(
        critic_runtime.CriticError, match=rf"{re.escape(library)} is missing.*sudo apt install {package}`"
    ):
        _critique(prepared)


def test_a_missing_library_without_a_known_package_is_named(monkeypatch, prepared):
    Server(
        monkeypatch,
        exits=127,
        output=(
            b"llama-server: error while loading shared libraries: libfoo.so.2: cannot open "
            b"shared object file: No such file or directory\n"
        ),
    )
    with pytest.raises(critic_runtime.CriticError, match=r"libfoo\.so\.2 is missing: .*No such") as caught:
        _critique(prepared)
    assert "apt" not in str(caught.value)


def test_a_port_another_process_listens_on_is_not_sent_the_key(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.owned = False
    with pytest.raises(critic_runtime.CriticError, match="not the llama-server this app started"):
        _critique(prepared)
    assert not any(path == "/v1/chat/completions" for _, path, _, _ in server.calls)
    assert server.process.terminated and prepared._process is None


def test_where_the_listener_cannot_be_seen_the_key_probe_is_the_guard(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.owned = None
    assert _critique(prepared).issues
    assert server.calls[-1][1] == "/v1/chat/completions"
