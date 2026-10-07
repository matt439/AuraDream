"""Shared fixtures and request helpers for the critic tests."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit
import dataclasses
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile

import pytest

from backend import (
    critic,
    critic_context,
    critic_runtime,
    critic_server,
    critic_validation,
    launch,
)
from backend.schemas import DocumentedOrder, GenerateRequest, PromptAnalysis


KLEIN = "black-forest-labs/FLUX.2-klein-4B"


PROMPT = (
    "neon city at night, a woman with a red umbrella walking past a shop with a sign "
    "saying open late, shot on film"
)


def _catalogue_repos() -> list[str]:
    return [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]


def _configure(monkeypatch, **changes) -> None:
    # `settings` is frozen, so point the module at a copy rather than mutating it.
    monkeypatch.setattr(critic, "settings", dataclasses.replace(critic.settings, **changes))


def _link_directory(link: Path, target: Path) -> None:
    """A directory link, as the Hub cache's snapshots are: a symlink where
    this user may make one, else (Windows without Developer Mode) a junction,
    which a resolve follows the same way."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("this user can make no directory link")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


@pytest.fixture
def root(tmp_path, monkeypatch):
    install = tmp_path / "hf" / "local-diffusion" / "llama.cpp"
    monkeypatch.setattr(critic_runtime, "install_root", lambda: install)
    return install


# --- Installing --------------------------------------------------------------


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


class Release:
    """A fake release: archives pinned by their real bytes, served by a fake fetch."""

    def __init__(self, monkeypatch, archives: dict[str, bytes], server: str = "llama-server.exe"):
        self.payloads = dict(archives)
        self.fetched: list[str] = []
        pins = tuple(
            critic_runtime.Archive(name, len(data), hashlib.sha256(data).hexdigest())
            for name, data in archives.items()
        )
        monkeypatch.setitem(critic_runtime.BUILDS, "test-x64", critic_runtime.Build(pins, server))
        monkeypatch.setattr(critic_runtime, "_fetch", self.fetch)

    def fetch(self, url, destination, size):
        name = url.rsplit("/", 1)[1]
        self.fetched.append(name)
        destination.write_bytes(self.payloads[name])


@pytest.fixture
def windows_release(monkeypatch):
    return Release(
        monkeypatch,
        {
            "llama-test.zip": _zip({"llama-server.exe": b"server", "ggml-cuda.dll": b"ggml"}),
            "cudart-test.zip": _zip({"cudart64_13.dll": b"cudart"}),
        },
    )


# --- The server --------------------------------------------------------------


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class FakeProcess:
    """A server process whose `wait` times out `stubborn` times before the
    signal it was last sent takes; `events` is what `_stop` did, in order,
    and `log_at_wait` whether the server's log was still there at each wait."""

    def __init__(self, returncode: int | None = None, *, stubborn: int = 0, log=None):
        self.returncode = returncode
        self.pid = 4242
        self.stubborn = stubborn
        self.log = log
        self.events: list[str] = []
        self.log_at_wait: list[bool] = []
        self._signal: int | None = None

    @property
    def terminated(self) -> bool:
        return "terminate" in self.events

    def pids(self):
        return {self.pid}

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append("terminate")
        self._signal = -15

    def kill(self):
        self.events.append("kill")
        self._signal = -9

    def wait(self, timeout=None):
        self.events.append("wait")
        if self.log is not None:
            self.log_at_wait.append(Path(self.log).exists())
        if self.returncode is None and self.stubborn:
            self.stubborn -= 1
            raise subprocess.TimeoutExpired("llama-server", timeout)
        if self.returncode is None:
            self.returncode = self._signal
        return self.returncode


def _answer(**changes) -> dict:
    answer = {
        "parts": {
            "subject": "a woman with a red umbrella",
            "action": "walking past a shop",
            "setting": "neon city at night",
            "style": None,
            "lighting": None,
            "camera": "shot on film",
            "text_to_render": "open late",
        },
        "issues": [
            {
                "kind": "missing",
                "problem": "The sign's words are not in quotation marks.",
                "quote": "sign saying open late",
                "cite_id": "G3",
                "fix": 'Write the sign as saying "OPEN LATE".',
            }
        ],
        "rewrite": (
            'A woman with a red umbrella walks past a shop in a neon city at night; its '
            'sign reads "OPEN LATE". Shot on film.'
        ),
        "reordered": None,
    }
    answer.update(changes)
    return answer


def _words(text: str, known: dict[str, int] | None = None) -> int:
    """A tokenizer for the fake server: a token per word or mark, as
    `fakes.fake_tokenizer` counts, but `known` texts at their given count."""
    total = 0
    for part, tokens in (known or {}).items():
        total += text.count(part) * tokens
        text = text.replace(part, " ")
    return total + len(re.findall(r"\w+|[^\w\s]", text))


class Server:
    """The seams around one critique: the spawn, every HTTP exchange, and
    whose the port's listener is (`owned`, as `_listener_is_ours` answers).
    It counts tokens with `count`, and not at all while `counts` is False."""

    def __init__(
        self, monkeypatch, answer=None, *, exits: int | None = None, output=b"", stubborn=0
    ):
        self.exits = exits
        self.stubborn = stubborn
        self.process: FakeProcess | None = None
        self.commands: list[list[str]] = []
        self.keys: list[str] = []
        self.calls: list[tuple[str, str, str | None, object]] = []
        self.answer = _answer() if answer is None else answer
        self.health = 200
        self.open_without_key = False
        self.post_error: BaseException | None = None
        self.output = output
        self.count = _words
        self.counts = True
        self.owned: bool | None = True
        # How many exchanges had happened at each listener check.
        self.checked_after: list[int] = []
        self.logs: list[str] = []
        mkstemp = critic.tempfile.mkstemp

        def record_log(*args, **kwargs):
            handle, name = mkstemp(*args, **kwargs)
            self.logs.append(name)
            return handle, name

        monkeypatch.setattr(critic.tempfile, "mkstemp", record_log)
        monkeypatch.setattr(critic_server, "_spawn", self.spawn)
        monkeypatch.setattr(critic_server, "_request", self.request)
        monkeypatch.setattr(critic_server, "_listener_is_ours", self.listener)
        monkeypatch.setattr(critic_server, "_HEALTH_POLL", 0.0)

    def spawn(self, command, log_file, api_key):
        self.commands.append(command)
        self.keys.append(api_key)
        log_file.write(self.output)
        log_file.flush()
        self.process = FakeProcess(self.exits, stubborn=self.stubborn, log=self.logs[-1])
        return self.process

    def listener(self, port, pids):
        assert port == int(_flag(self.commands[-1], "--port"))
        assert pids == self.process.pids()
        self.checked_after.append(len(self.calls))
        return self.owned

    def request(self, method, url, api_key, body, timeout):
        path = urlsplit(url).path
        self.calls.append((method, path, api_key, body))
        if path == "/health":
            return self.health, {}
        if path == "/v1/models":
            return (200, {}) if api_key or self.open_without_key else (401, {})
        if path in ("/apply-template", "/tokenize"):
            assert api_key == self.keys[-1]
            if not self.counts:
                return 404, {"error": {"message": "File Not Found"}}
            if path == "/apply-template":
                assert body["chat_template_kwargs"] == {"enable_thinking": True}
                rendered = "".join(
                    f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                    for message in body["messages"]
                )
                return 200, {"prompt": rendered + "<|im_start|>assistant\n"}
            return 200, {"tokens": list(range(self.count(body["content"])))}
        assert path == "/v1/chat/completions"
        if self.post_error is not None:
            raise self.post_error
        content = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
        return 200, {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "timings": {"prompt_n": 1200, "predicted_n": 1300, "predicted_per_second": 150.0},
        }


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    server, model = tmp_path / "llama-server.exe", tmp_path / "model.gguf"
    server.write_bytes(b"s")
    model.write_bytes(b"m")
    _configure(monkeypatch, critic=True, critic_server=server, critic_model=None)
    target = critic.PromptCritic()
    target._prepared = (server, model)
    return target


def _analysis(model_id: str = KLEIN, *, ordered: bool = True, findings=()) -> PromptAnalysis:
    order = (
        DocumentedOrder(
            steps=["subject", "action", "style", "context"],
            cite="Subject, then action, then style, then context",
        )
        if ordered
        else None
    )
    return PromptAnalysis(
        model_id=model_id, style="prose", words=20, findings=list(findings), order=order
    )


def _critique(target, model_id=KLEIN, prompt=PROMPT, *, negative=None, **kwargs):
    request = GenerateRequest(prompt=prompt, negative_prompt=negative)
    return target.critique(model_id, request, request, _analysis(model_id, **kwargs))


# --- Reading the answer ------------------------------------------------------


def _context(
    model_id: str = KLEIN, prompt: str = PROMPT, *, ordered: bool = True, negative=None,
    style: str = "prose",
):
    request = GenerateRequest(prompt=prompt, negative_prompt=negative)
    analysis = _analysis(model_id, ordered=ordered).model_copy(update={"style": style})
    return critic_context.build_context(model_id, request, request, analysis)


def _validate(
    answer, *, prompt=PROMPT, ordered=True, model_id=KLEIN, style="prose", findings=()
):
    request = GenerateRequest(prompt=prompt)
    analysis = _analysis(model_id, ordered=ordered, findings=findings).model_copy(
        update={"style": style}
    )
    context = critic_context.build_context(model_id, request, request, analysis)
    return critic_validation.validate(answer, context, prompt, model="test", seconds=1.234)


KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
