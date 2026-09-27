"""The prompt critic: its pinned runtime, the server's lifecycle, the grounding
it cites and what is kept of its answer. No network, no subprocess, no GPU -
the fetch, the spawn and the HTTP exchange are the module's seams."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import huggingface_hub
import pytest

from backend import config, critic, guides, launch, prompt_encoding, prompt_rules, prompt_sources
from backend.model_defaults import _profile_for
from backend.schemas import DocumentedOrder, GenerateRequest, PromptAnalysis, PromptFinding

KLEIN = "black-forest-labs/FLUX.2-klein-4B"
ANIMA = "circlestone-labs/Anima"
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


@pytest.fixture
def root(tmp_path, monkeypatch):
    install = tmp_path / "hf" / "local-diffusion" / "llama.cpp"
    monkeypatch.setattr(critic, "install_root", lambda: install)
    return install


# --- The pinned table --------------------------------------------------------


def test_pinned_builds_cover_windows_and_wsl():
    assert set(critic.BUILDS) == {"win-cuda-13.4-x64", "ubuntu-cuda-13.4-x64"}
    for key, build in critic.BUILDS.items():
        assert len(build.archives) == 2, key
        names = [archive.name for archive in build.archives]
        assert names[0].startswith(f"llama-{critic.RELEASE}-bin-") and key in names[0]
        assert names[1].startswith("cudart-llama-") and key in names[1]
        for archive in build.archives:
            assert re.fullmatch(r"[0-9a-f]{64}", archive.sha256), archive.name
            assert archive.size > 100_000_000, archive.name
            assert archive.url == (
                "https://github.com/ggml-org/llama.cpp/releases/download/b11206/" + archive.name
            )
    assert critic.BUILDS["win-cuda-13.4-x64"].server == "llama-server.exe"
    assert critic.BUILDS["ubuntu-cuda-13.4-x64"].server == "llama-server"
    assert critic.MODEL["repo_id"] == "Qwen/Qwen3-4B-GGUF"
    assert re.fullmatch(r"[0-9a-f]{40}", critic.MODEL["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", critic.MODEL_SHA256)
    assert critic.MODEL_SIZE == 4280404704


@pytest.mark.parametrize(
    ("system", "machine", "key"),
    [
        ("win32", "AMD64", "win-cuda-13.4-x64"),
        ("linux", "x86_64", "ubuntu-cuda-13.4-x64"),
        ("linux", "aarch64", None),
        ("darwin", "arm64", None),
    ],
)
def test_platform_key(monkeypatch, system, machine, key):
    monkeypatch.setattr(critic, "sys", SimpleNamespace(platform=system))
    monkeypatch.setattr(critic.platform, "machine", lambda: machine)
    assert critic.platform_key() == key


# --- Installing --------------------------------------------------------------


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def _tar(members: dict[str, bytes], top: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        if top is not None:
            directory = tarfile.TarInfo(top)
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            bundle.addfile(directory)
        for name, data in members.items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.size = len(data)
            info.mode = 0o755
            bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class Release:
    """A fake release: archives pinned by their real bytes, served by a fake fetch."""

    def __init__(self, monkeypatch, archives: dict[str, bytes], server: str = "llama-server.exe"):
        self.payloads = dict(archives)
        self.fetched: list[str] = []
        pins = tuple(
            critic.Archive(name, len(data), hashlib.sha256(data).hexdigest())
            for name, data in archives.items()
        )
        monkeypatch.setitem(critic.BUILDS, "test-x64", critic.Build(pins, server))
        monkeypatch.setattr(critic, "_fetch", self.fetch)

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


def test_install_verifies_extracts_and_publishes_once(root, windows_release):
    server = critic.install_server("test-x64")
    target = root / f"{critic.RELEASE}-test-x64"
    assert server == target / "llama-server.exe"
    assert server.read_bytes() == b"server"
    # Both archives land in one directory: the runtime beside the binary.
    assert (target / "ggml-cuda.dll").is_file() and (target / "cudart64_13.dll").is_file()
    assert json.loads((target / critic._INSTALL_MARKER).read_text())["release"] == critic.RELEASE
    assert critic.installed_server("test-x64") == server
    # The verified archives are not kept once installed, nor is any work dir.
    assert not list((root / "downloads").iterdir())
    assert sorted(path.name for path in root.iterdir() if path.is_dir()) == [
        "b11206-test-x64", "downloads",
    ]
    assert critic.install_server("test-x64") == server
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip"]


def test_a_bad_checksum_is_refused_and_retried(root, windows_release):
    good = windows_release.payloads["cudart-test.zip"]
    # Same size, other bytes: the size check passes and the hash must not.
    windows_release.payloads["cudart-test.zip"] = bytes(len(good))
    with pytest.raises(critic.CriticError, match="sha256"):
        critic.install_server("test-x64")
    assert critic.installed_server("test-x64") is None
    assert not (root / f"{critic.RELEASE}-test-x64").exists()
    assert not list((root / "downloads").glob("*.part"))
    # The first archive verified and is reused; only the bad one is fetched again.
    assert (root / "downloads" / "llama-test.zip").is_file()
    windows_release.payloads["cudart-test.zip"] = good
    assert critic.install_server("test-x64").read_bytes() == b"server"
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip", "cudart-test.zip"]


def test_a_wrong_size_is_refused(root, windows_release):
    windows_release.payloads["llama-test.zip"] += b"x"
    with pytest.raises(critic.CriticError, match="bytes, not the pinned"):
        critic.install_server("test-x64")
    assert critic.installed_server("test-x64") is None


@pytest.mark.parametrize("member", ["../evil.dll", "/evil.dll", "C:/evil.dll", "a/../../evil.dll"])
def test_a_zip_member_escaping_the_target_is_refused(root, monkeypatch, member):
    Release(monkeypatch, {"llama-test.zip": _zip({"llama-server.exe": b"s", member: b"evil"})})
    with pytest.raises(critic.CriticError, match="outside the install directory"):
        critic.install_server("test-x64")
    assert critic.installed_server("test-x64") is None
    assert not (root / f"{critic.RELEASE}-test-x64").exists()
    assert not list(root.parent.rglob("evil.dll"))


def test_a_tar_member_escaping_the_target_is_refused(root, monkeypatch):
    Release(
        monkeypatch,
        {"llama-test.tar.gz": _tar({"llama-server": b"s", "../evil.so": b"evil"})},
        server="llama-server",
    )
    with pytest.raises(critic.CriticError):
        critic.install_server("test-x64")
    assert critic.installed_server("test-x64") is None
    assert not list(root.parent.rglob("evil.so"))


def test_tarballs_are_flattened_into_one_directory(root, monkeypatch):
    Release(
        monkeypatch,
        {
            "llama-test.tar.gz": _tar(
                {"llama-server": b"server", "libggml-cuda.so": b"ggml"}, top="llama-b11206"
            ),
            "cudart-test.tar.gz": _tar({"libcudart.so.13": b"cudart"}, top="cudart-llama-test"),
        },
        server="llama-server",
    )
    server = critic.install_server("test-x64")
    assert server.read_bytes() == b"server"
    assert (server.parent / "libggml-cuda.so").is_file()
    assert (server.parent / "libcudart.so.13").is_file()
    if sys.platform != "win32":
        assert server.stat().st_mode & 0o111


def test_an_archive_without_the_server_is_not_installed(root, monkeypatch):
    Release(monkeypatch, {"llama-test.zip": _zip({"llama-cli.exe": b"cli"})})
    with pytest.raises(critic.CriticError, match="llama-server.exe"):
        critic.install_server("test-x64")
    assert critic.installed_server("test-x64") is None


def test_an_unmarked_tree_is_never_used_and_is_replaced(root, windows_release):
    target = root / f"{critic.RELEASE}-test-x64"
    target.mkdir(parents=True)
    (target / "llama-server.exe").write_bytes(b"half")
    assert critic.installed_server("test-x64") is None
    assert critic.install_server("test-x64").read_bytes() == b"server"
    # A marker from another recipe is as good as none.
    (target / critic._INSTALL_MARKER).write_text("{}")
    assert critic.installed_server("test-x64") is None


# --- supported, ready, prepare -----------------------------------------------


def test_supported_follows_the_switch_and_the_platform(monkeypatch, tmp_path):
    _configure(monkeypatch, critic=False, critic_server=None, critic_model=None)
    assert not critic.prompt_critic.supported()
    assert not critic.prompt_critic.ready()
    with pytest.raises(critic.CriticUnavailable, match="switched off"):
        critic.prompt_critic.prepare()

    _configure(monkeypatch, critic=True)
    monkeypatch.setattr(critic, "platform_key", lambda: "win-cuda-13.4-x64")
    assert critic.prompt_critic.supported()
    monkeypatch.setattr(critic, "platform_key", lambda: None)
    assert not critic.prompt_critic.supported()
    # A server of the user's own makes any platform supported.
    _configure(monkeypatch, critic_server=tmp_path / "llama-server")
    assert critic.prompt_critic.supported()


def test_ready_is_what_is_on_disk(monkeypatch, tmp_path, root, windows_release):
    monkeypatch.setattr(critic, "platform_key", lambda: "test-x64")
    _configure(monkeypatch, critic=True, critic_server=None, critic_model=None)
    cached: list[str | None] = [None]
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *args, **kwargs: cached[0]
    )
    assert critic.prompt_critic.supported() and not critic.prompt_critic.ready()
    critic.install_server("test-x64")
    assert not critic.prompt_critic.ready()  # the server alone is not enough
    gguf = tmp_path / "Qwen3-4B-Q8_0.gguf"
    gguf.write_bytes(b"gguf")
    cached[0] = str(gguf)
    assert critic.prompt_critic.ready()
    # Nothing was fetched to answer: ready() is a status-poll question.
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip"]

    own = tmp_path / "own-server"
    _configure(monkeypatch, critic_server=own, critic_model=tmp_path / "missing.gguf")
    assert not critic.prompt_critic.ready()
    own.write_bytes(b"s")
    assert not critic.prompt_critic.ready()
    _configure(monkeypatch, critic_model=gguf)
    assert critic.prompt_critic.ready()


def test_prepare_uses_configured_files_without_downloading(monkeypatch, tmp_path):
    server, model = tmp_path / "llama-server", tmp_path / "mine.gguf"
    server.write_bytes(b"s")
    model.write_bytes(b"m")
    _configure(monkeypatch, critic=True, critic_server=server, critic_model=model)
    monkeypatch.setattr(critic, "install_server", lambda key: pytest.fail("installed"))
    target = critic.PromptCritic()
    target.prepare()
    assert target._prepared == (server, model)
    assert target._label(model) == "mine"

    _configure(monkeypatch, critic_model=tmp_path / "missing.gguf")
    with pytest.raises(critic.CriticUnavailable, match="LOCAL_DIFFUSION_CRITIC_MODEL"):
        target.prepare()


def test_prepare_downloads_the_pinned_model_and_hashes_it_once(monkeypatch, tmp_path, root):
    gguf = tmp_path / "blob.gguf"
    gguf.write_bytes(b"pinned weights")
    monkeypatch.setattr(critic, "MODEL_SIZE", gguf.stat().st_size)
    monkeypatch.setattr(critic, "MODEL_SHA256", hashlib.sha256(b"pinned weights").hexdigest())
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(gguf)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    hashes = []
    real = critic._sha256
    monkeypatch.setattr(critic, "_sha256", lambda path: hashes.append(path) or real(path))
    server = tmp_path / "llama-server"
    server.write_bytes(b"s")
    _configure(monkeypatch, critic=True, critic_server=server, critic_model=None)

    target = critic.PromptCritic()
    target.prepare()
    target.prepare()
    assert downloads == [critic.MODEL, critic.MODEL]
    assert target._prepared == (server, gguf)
    assert target._label(gguf) == critic.MODEL_LABEL
    assert len(hashes) == 1


def test_a_model_off_its_pin_is_refused(monkeypatch, tmp_path, root):
    gguf = tmp_path / "blob.gguf"
    gguf.write_bytes(b"other weights!")
    monkeypatch.setattr(critic, "MODEL_SIZE", gguf.stat().st_size)
    with pytest.raises(critic.CriticError, match="sha256"):
        critic.verify_model(gguf)
    monkeypatch.setattr(critic, "MODEL_SIZE", 1)
    with pytest.raises(critic.CriticError, match="bytes"):
        critic.verify_model(gguf)


# --- The server --------------------------------------------------------------


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_server_command_binds_loopback_and_keeps_the_key_off_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = critic.server_command(Path("bin/llama-server.exe"), Path("m.gguf"), 5123)
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
    assert _flag(command, "--reasoning-budget") == str(critic.REASONING_BUDGET)
    for flag in ("--no-ui", "--jinja", "--offline"):
        assert flag in command


def test_server_env_drops_llama_settings_and_carries_the_key(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    monkeypatch.setenv("LLAMA_API_KEY", "shared")
    monkeypatch.setenv("LOCAL_DIFFUSION_KEEP", "1")
    env = critic._server_env(Path("bin/llama-server"), "k3y")
    assert "LLAMA_ARG_HOST" not in env
    # b11206 reads --api-key from LLAMA_API_KEY: this start's, never the user's.
    assert env["LLAMA_API_KEY"] == "k3y"
    assert env["LOCAL_DIFFUSION_KEEP"] == "1"


def test_configured_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_MODEL", "rel.gguf")
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_SERVER", "  ")
    model = config._env_path("LOCAL_DIFFUSION_CRITIC_MODEL")
    assert model.is_absolute() and model == (tmp_path / "rel.gguf").resolve()
    assert config._env_path("LOCAL_DIFFUSION_CRITIC_SERVER") is None
    monkeypatch.setenv("LOCAL_DIFFUSION_CRITIC_SERVER", "~/llama-server")
    assert config._env_path("LOCAL_DIFFUSION_CRITIC_SERVER").is_absolute()


class FakeProcess:
    def __init__(self, returncode: int | None = None):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
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


class Server:
    """The seams around one critique: the spawn and every HTTP exchange."""

    def __init__(self, monkeypatch, answer=None, *, exits: int | None = None, output=b""):
        self.exits = exits
        self.process: FakeProcess | None = None
        self.commands: list[list[str]] = []
        self.keys: list[str] = []
        self.calls: list[tuple[str, str, str | None, object]] = []
        self.answer = _answer() if answer is None else answer
        self.health = 200
        self.open_without_key = False
        self.post_error: BaseException | None = None
        self.output = output
        monkeypatch.setattr(critic, "_spawn", self.spawn)
        monkeypatch.setattr(critic, "_request", self.request)
        monkeypatch.setattr(critic, "_HEALTH_POLL", 0.0)

    def spawn(self, command, log_file, api_key):
        self.commands.append(command)
        self.keys.append(api_key)
        log_file.write(self.output)
        log_file.flush()
        self.process = FakeProcess(self.exits)
        return self.process

    def request(self, method, url, api_key, body, timeout):
        path = urlsplit(url).path
        self.calls.append((method, path, api_key, body))
        if path == "/health":
            return self.health, {}
        if path == "/v1/models":
            return (200, {}) if api_key or self.open_without_key else (401, {})
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


def test_a_critique_asks_a_keyed_server_and_stops_it(monkeypatch, prepared):
    server = Server(monkeypatch)
    result = _critique(prepared)

    key = server.keys[0]
    assert len(key) >= 32 and key not in server.commands[0]
    assert [(method, path) for method, path, _, _ in server.calls] == [
        ("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"),
    ]
    # The key-less probe must be refused, the question must carry the key.
    assert server.calls[1][2] is None and server.calls[2][2] == key
    body = server.calls[2][3]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert {k: body[k] for k in critic.SAMPLING} == critic.SAMPLING
    assert body["max_tokens"] == critic.MIN_TOKENS
    schema = body["response_format"]["json_schema"]["schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert "G3" in schema["properties"]["issues"]["items"]["properties"]["cite_id"]["enum"]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert PROMPT in body["messages"][1]["content"]

    assert server.process.terminated and prepared._process is None
    assert result.model == critic.MODEL_LABEL and result.seconds >= 0
    assert result.issues[0].cite_id == "G3" and result.issues[0].source == "guide"

    # A fresh key and a fresh server every time.
    _critique(prepared)
    assert server.keys[1] != key


def test_the_server_is_stopped_when_the_request_fails(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.post_error = TimeoutError("timed out")
    with pytest.raises(critic.CriticError, match="did not answer"):
        _critique(prepared)
    assert server.process.terminated and prepared._process is None


def test_an_early_exit_reports_the_server_output(monkeypatch, prepared):
    Server(monkeypatch, exits=1, output=b"load_model: failed\nerror: out of memory\n")
    with pytest.raises(critic.CriticError, match="code 1.*out of memory"):
        _critique(prepared)


def test_a_server_that_never_gets_healthy_is_stopped(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.health = 503
    monkeypatch.setattr(critic, "HEALTH_TIMEOUT", 0.0)
    with pytest.raises(critic.CriticError, match="not ready"):
        _critique(prepared)
    assert server.process.terminated
    assert all(path == "/health" for _, path, _, _ in server.calls)


def test_a_server_answering_without_its_key_is_not_asked(monkeypatch, prepared):
    server = Server(monkeypatch)
    server.open_without_key = True
    with pytest.raises(critic.CriticError, match="without its API key"):
        _critique(prepared)
    assert server.process.terminated
    assert not any(path == "/v1/chat/completions" for _, path, _, _ in server.calls)


def test_critique_needs_prepare(monkeypatch):
    _configure(monkeypatch, critic=True)
    with pytest.raises(critic.CriticUnavailable, match="not prepared"):
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

    monkeypatch.setattr(critic, "_request", truncated)
    with pytest.raises(critic.CriticError, match="ran out of tokens.*shorter prompt"):
        _critique(prepared)
    assert server.process.terminated


# Qwen3-4B's tokenizer on these texts, measured 2026-09-27 (text, repeats, tokens).
MEASURED = [
    (
        (
            "A tall sharp-featured man in an oversized charcoal wool coat, standing on a "
            "wet cobblestone street at night, fashion editorial, moody street lighting. "
        ),
        10, 321,
    ),
    (
        (
            "1girl, long_hair, blue_eyes, school_uniform, serafuku, looking_at_viewer, "
            "(masterpiece:1.2), absurdres, "
        ),
        10, 340,
    ),
    ("一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。", 10, 290),
    ("桜の木の下で微笑む少女、夕暮れの光、柔らかい風。", 10, 220),
    ("0123456789 3.1415926-53589793 ", 10, 300),
    ("😊🎉✨ ", 10, 40),
]


@pytest.mark.parametrize(("text", "repeats", "tokens"), MEASURED)
def test_the_token_estimate_errs_long(text, repeats, tokens):
    estimate = critic.estimate_tokens(text * repeats)
    # Within a tenth under at worst (digits), at most about half again over.
    assert 0.9 * tokens <= estimate <= 1.6 * tokens
    # Scaled down, a lower bound: what the refusal counts with.
    assert critic._ESTIMATE_FLOOR * estimate <= tokens


def test_a_long_prompt_is_given_room_for_its_answer(monkeypatch, prepared):
    server = Server(monkeypatch)
    # Near the request's 4000-character limit. A 4000-character English prompt
    # measured up to 966 tokens, and the answer repeats it three times after
    # up to REASONING_BUDGET of thought.
    prompt = (MEASURED[0][0] * 30)[:4000]
    _critique(prepared, prompt=prompt)
    budget = server.calls[-1][3]["max_tokens"]
    assert budget >= critic.REASONING_BUDGET + 3 * 966 + critic.ANSWER_OVERHEAD
    assert budget <= critic.CONTEXT
    # Unordered, the answer has no reordered copy, and a short prompt is
    # given the measured floor.
    _critique(prepared, prompt=prompt, ordered=False)
    assert server.calls[-1][3]["max_tokens"] < budget
    _critique(prepared, prompt="a cat")
    assert server.calls[-1][3]["max_tokens"] == critic.MIN_TOKENS


def test_a_prompt_whose_answer_cannot_fit_is_refused_before_a_server_starts(
    monkeypatch, prepared
):
    server = Server(monkeypatch)
    # 4000 characters of Chinese measured 2698 tokens: repeated twice with the
    # sources, and again as the negative prompt, it overflows 8192 tokens
    # before the critic has thought at all.
    chinese = (MEASURED[2][0] * 100)[:4000]
    with pytest.raises(critic.CriticError, match="too long"):
        _critique(prepared, "baidu/ERNIE-Image-Turbo", chinese, negative=chinese, ordered=False)
    assert server.commands == []
    # A shorter one is asked.
    _critique(prepared, "baidu/ERNIE-Image-Turbo", chinese[:1500], ordered=False)
    assert len(server.commands) == 1


# --- Reading the answer ------------------------------------------------------


def _context(
    model_id: str = KLEIN, prompt: str = PROMPT, *, ordered: bool = True, negative=None,
    style: str = "prose",
):
    request = GenerateRequest(prompt=prompt, negative_prompt=negative)
    analysis = _analysis(model_id, ordered=ordered).model_copy(update={"style": style})
    return critic.build_context(model_id, request, request, analysis)


def _validate(
    answer, *, prompt=PROMPT, ordered=True, model_id=KLEIN, style="prose", findings=()
):
    request = GenerateRequest(prompt=prompt)
    analysis = _analysis(model_id, ordered=ordered, findings=findings).model_copy(
        update={"style": style}
    )
    context = critic.build_context(model_id, request, request, analysis)
    return critic.validate(answer, context, prompt, model="test", seconds=1.234)


def test_citations_resolve_and_unknown_ones_are_dropped():
    sources = prompt_sources.sources_for(KLEIN)
    issue = _answer()["issues"][0]
    result = _validate(
        _answer(
            issues=[
                {**issue, "cite_id": "G99"},
                {**issue, "cite_id": "A1"},
                {**issue, "cite_id": "g1"},
                {**issue, "cite_id": "G2", "problem": "  "},
                "not an issue",
            ]
        )
    )
    assert result.dropped == 3
    assert [item.cite_id for item in result.issues] == ["A1", "G1"]
    author, guide = result.issues
    assert author.source == "author" and author.cite == sources.excerpts[0].text
    assert author.url == sources.excerpts[0].url
    assert guide.source == "guide" and guide.cite == sources.sentences[0].text
    assert guide.url == sources.sentences[0].url == sources.url
    assert result.seconds == 1.23


def test_quotes_are_located_in_the_prompt_or_nulled():
    issue = _answer()["issues"][0]
    result = _validate(
        _answer(
            issues=[
                {**issue, "quote": "A  RED\numbrella"},
                {**issue, "quote": '"sign saying open late."'},
                {**issue, "quote": "a purple elephant"},
                {**issue, "quote": "   "},
            ]
        )
    )
    real, quoted, invented, blank = result.issues
    assert real.quote == "a red umbrella"
    assert PROMPT[real.start:real.end] == real.quote
    assert quoted.quote == "sign saying open late"
    assert invented.quote is None and invented.start is None and invented.end is None
    assert blank.quote is None
    # A nulled quote costs the issue nothing: the citation is what is checked.
    assert result.dropped == 0


def test_reordered_is_kept_only_as_a_permutation_under_a_documented_order():
    prompt = "neon city at night, a woman with a red umbrella, shot on film"
    moved = "a woman with a red umbrella, neon city at night, shot on film"
    assert _validate(_answer(reordered=moved), prompt=prompt).reordered == moved
    # Case and punctuation are not words.
    assert _validate(_answer(reordered=moved.upper() + "."), prompt=prompt).reordered
    for bad in (
        moved + ", cinematic",  # a word added
        moved.replace("red ", ""),  # a word dropped
        moved.replace("woman", "girl"),  # a word changed
        prompt + ".",  # nothing moved
        "",
    ):
        assert _validate(_answer(reordered=bad), prompt=prompt).reordered is None, bad
    tags = "1girl, long_hair, smile"
    assert _validate(_answer(reordered="long hair, 1girl, smile"), prompt=tags).reordered is None
    # No documented order, no reordered variant, whatever the model says.
    assert _validate(_answer(reordered=moved), prompt=prompt, ordered=False).reordered is None


def _kept(reordered: str, prompt: str, style: str = "prose") -> bool:
    answer = _answer(reordered=reordered)
    return _validate(answer, prompt=prompt, style=style).reordered is not None


def test_a_reorder_moves_whole_phrases_and_keeps_rendered_text():
    city, woman, sign = "neon city at night", "a woman in a red dress and a blue hat", '"OPEN"'
    prompt = f"{city}, {woman}, sign that says {sign}"
    assert _kept(f"{woman}, {city}, sign that says {sign}", prompt)
    # Every word is still there, but the colours changed owner or the sign
    # lost its quotation marks: a different picture, not a different order.
    swapped = "a woman in a blue dress and a red hat"
    assert not _kept(f"{swapped}, {city}, sign that says OPEN", prompt)
    assert not _kept(f"{swapped}, {city}, sign that says {sign}", prompt)
    assert not _kept(f"{woman}, {city}, sign that says OPEN", prompt)
    assert not _kept(f'{woman}, {city}, sign that says "open"', prompt)
    # A tag list's items are its pieces.
    tags = "1girl, blue hair, red eyes"
    assert _kept("1girl, red eyes, blue hair", tags, "tags")
    assert not _kept("1girl, red hair, blue eyes", tags, "tags")
    # Sentences move whole; quoted text never splits, whatever it holds.
    sentences = "A neon city at night. A woman walks past."
    assert _kept("A woman walks past. A neon city at night.", sentences)
    quoted = 'sign that says "OPEN, LATE", a woman'
    assert _kept('a woman, sign that says "OPEN, LATE"', quoted)
    assert not _kept('LATE", a woman, sign that says "OPEN', quoted)
    # An apostrophe is not a quotation mark.
    possessive = "the animal's face, a dog's tail, 'OPEN' sign"
    assert _kept("a dog's tail, the animal's face, 'OPEN' sign", possessive)
    # Guillemets and „…“ are quotation marks as the rules read them: text in
    # them moves whole or not at all (a narrower copy of the pattern here once
    # let «Fresh bread, hot coffee» split across two pieces).
    for opening, closing in (("«", "»"), ("„", "“")):
        sign = f"a chalkboard sign reading {opening}Fresh bread, hot coffee{closing}"
        prompt = f"A bakery at dawn, {sign}, a cat on the step"
        assert _kept(f"{sign}, A bakery at dawn, a cat on the step", prompt)
        split = (f"A bakery at dawn, hot coffee{closing}, a chalkboard sign reading "
                 f"{opening}Fresh bread, a cat on the step")
        assert not _kept(split, prompt), opening
    assert critic.QUOTED is prompt_rules.QUOTED is prompt_encoding.QUOTED


def test_a_reorder_of_xml_moves_blocks_and_of_json_moves_keys():
    xml = guides.NEWBIE_GUIDE["examples"][0]
    first = xml[: xml.index("</character_1>") + len("</character_1>")]
    second = xml[xml.index("<character_2>") : xml.index("</character_2>") + len("</character_2>")]
    scene = xml[xml.index("<general_tags>") :]
    assert _kept(f"{scene}\n{first}\n  {second}", xml, "xml")
    # A tag moved from one character to the other is not an order.
    leaked = xml.replace("blue_hair, long_hair", "long_hair")
    leaked = leaked.replace("pink_hair,", "pink_hair, blue_hair,")
    assert not _kept(leaked, xml, "xml")
    scene, subjects = '"scene": "a street"', '"subjects": [{"type": "woman", "coat": "red"}]'
    prompt = f"{{{scene}, {subjects}}}"
    moved = '{"subjects": [{"coat": "red", "type": "woman"}], "scene": "a street"}'
    assert _kept(moved, prompt, "json")
    assert not _kept(prompt.replace("red", "blue"), prompt, "json")
    assert not _kept(f"{{{scene},  {subjects}}}", prompt, "json")  # nothing moved


def test_the_rewrite_is_dropped_when_it_changes_nothing():
    # Layer 1's lint of the rewrite is the route's (test_api_analysis), once
    # the gate is open; validate adds only its own checks.
    result = _validate(_answer(rewrite="  A calm street.  "))
    assert result.rewrite == "A calm street." and result.rewrite_findings == []
    same = _validate(_answer(rewrite="  " + PROMPT.replace(" ", "  ") + "\n"))
    assert same.rewrite is None and same.rewrite_findings == [] and same.set_aside == []
    assert _validate(_answer(rewrite="x" * (critic.MAX_TEXT + 1))).rewrite is None
    assert _validate(_answer(rewrite=None)).rewrite is None


def test_strings_are_capped_and_parts_normalised():
    issue = {**_answer()["issues"][0], "problem": "p" * 5000, "fix": "f" * 5000, "kind": "nonsense"}
    result = _validate(_answer(issues=[issue] * 8, parts={"subject": "  ", "camera": 3}))
    assert len(result.issues) == critic.MAX_ISSUES
    assert len(result.issues[0].problem) == critic.MAX_TEXT
    assert len(result.issues[0].fix) == critic.MAX_TEXT
    assert result.issues[0].kind == "other"
    assert result.parts == dict.fromkeys(critic.PARTS)


def test_parts_are_the_prompts_own_words_and_text_to_render_its_lettering():
    # Seen on the card: the whole prompt as text to render, "none" as a
    # camera, lighting the prompt never mentioned.
    parts = {
        "subject": "a woman, red umbrella",
        "action": "walking past a shop",
        "setting": "Neon  City at night",
        "style": "cinematic",
        "lighting": "golden hour light",
        "camera": "none",
        "text_to_render": PROMPT,
    }
    result = _validate(_answer(parts=parts))
    assert result.parts == {
        "subject": "a woman, red umbrella",
        "action": "walking past a shop",
        "setting": "Neon  City at night",
        "style": None,
        "lighting": None,
        "camera": None,
        "text_to_render": None,
    }
    # Text to render is what the prompt letters: its unquoted sign text, a
    # quoted stretch, or words running on from where the rules ended one.
    for text in ("open late", '"OPEN LATE"', "Open Late"):
        assert _validate(_answer(parts={"text_to_render": text})).parts["text_to_render"] == text
    quoted = 'a shop sign that says "Fish and Chips", a gull'
    for text, kept in (("Fish and Chips", True), ("a gull", False), ("chips", True)):
        answer = _answer(parts={"text_to_render": text})
        assert (_validate(answer, prompt=quoted).parts["text_to_render"] == text) is kept, text
    loose = "a shop sign that says fish and chips for sale, a gull"
    for text, kept in (("fish and chips", True), ("fish and chips for sale", True),
                       ("fish and chips for sale, a gull", False)):
        answer = _answer(parts={"text_to_render": text})
        assert (_validate(answer, prompt=loose).parts["text_to_render"] == text) is kept, text


def test_issues_restating_a_rule_finding_are_left_out_and_said_so():
    def issue(problem, fix="Fix it.", kind="other", cite="G1"):
        return {"kind": kind, "problem": problem, "quote": None, "cite_id": cite, "fix": fix}

    finding = PromptFinding(rule="guide.negation", severity="warn", message="m", source="guide")
    issues = [
        issue("Prompt is too short (7 words)", "Expand it to 30-80 words"),
        issue("Uses negation", "Rephrase the negative phrasing"),
        issue("The sign text is not in quotation marks"),
    ]
    # klein has a word range, so a length claim restates what the rules
    # measured; the negation is repeated only once it was reported.
    unreported = _validate(_answer(issues=issues))
    assert [i.problem for i in unreported.issues] == [
        "Uses negation", "The sign text is not in quotation marks",
    ]
    assert unreported.set_aside == ["1 issue left out: it repeated the rule findings above (word count)."]
    reported = _validate(_answer(issues=issues), findings=[finding])
    assert [i.problem for i in reported.issues] == ["The sign text is not in quotation marks"]
    assert reported.set_aside == [
        "2 issues left out: they repeated the rule findings above (word count, negation)."
    ]
    # Krea 2 states no word range: its critic may speak to length.
    krea2 = _validate(_answer(issues=issues[:1]), model_id="krea/Krea-2-Turbo", ordered=False)
    assert len(krea2.issues) == 1 and krea2.set_aside == []
    # An issue quoting the very words a finding points at restates it, in
    # whatever words it puts it ("no people" / "Replace with 'empty'").
    prompt = "A quiet street after the rain, no people, a shop sign that says Open Late."
    start = prompt.index("no people")
    negation = finding.model_copy(update={"start": start, "end": start + len("no people")})
    cut = PromptFinding(rule="encoder.cut", severity="info", message="m", source="encoder",
                        start=prompt.index("a shop"), end=len(prompt))
    quoting = [
        {"kind": "missing", "problem": "no people", "quote": "no people,", "cite_id": "A9",
         "fix": "Replace with 'empty', 'deserted' or 'solitary'"},
        # A cut's span is where reading stops, not what the words say.
        {"kind": "vague", "problem": "The shop is not described", "quote": "a shop sign",
         "cite_id": "A1", "fix": "Say what the shop sells"},
    ]
    result = _validate(
        _answer(issues=quoting, rewrite=None), prompt=prompt, findings=[negation, cut]
    )
    assert [i.problem for i in result.issues] == ["The shop is not described"]
    assert result.set_aside == ["1 issue left out: it repeated the rule findings above (negation)."]


def test_order_issues_that_ask_for_different_things_first_are_left_out():
    def order(problem, fix):
        return {"kind": "order", "problem": problem, "quote": None, "cite_id": "A1", "fix": fix}

    # Both from one answer about Krea's own "Moody night photography, ...".
    both = [
        order("Subject not first", "Move subject to the beginning"),
        order("Style not front-loaded", "Place style description first"),
        {**_answer()["issues"][0]},
    ]
    result = _validate(_answer(issues=both))
    assert [i.kind for i in result.issues] == ["missing"]
    assert result.set_aside == [
        "2 order issues left out: they asked for different things first (subject, style)."
    ]
    for fixes, wants in (
        (("Move subject to first position", "Place style description before subject"),
         "subject, style"),
        (("Start with the subject.", "Lead with the lighting"), "subject, lighting"),
    ):
        pair = [order("Out of order", fix) for fix in fixes]
        assert _validate(_answer(issues=pair)).set_aside == [
            f"2 order issues left out: they asked for different things first ({wants})."
        ], fixes
    # Ones that agree stand, whatever the prompt's words quoted in a fix say
    # and in whatever order a problem names the parts it describes.
    agreeing = [
        order("Subject is not first", "Start with 'a lone hiker in a red jacket'"),
        order("Environmental details precede subject", "Move 'misty light' after the subject"),
        order("Technical specs not prioritized", "Focus on image description, not camera details"),
        # Both from one answer on the card, which once read as subject against style.
        order("The subject is not placed first as required", "Move the subject to the beginning."),
        order("The style description is placed before the subject",
              "Place the subject first, then the style description."),
        # BFL's first step is "the image type and subject".
        order("Out of order", "Start with the image type and subject, then add the rest"),
    ]
    kept = _validate(_answer(issues=agreeing))
    assert len(kept.issues) == 6 and kept.set_aside == []


def test_a_reordered_version_that_changed_words_is_said_so():
    prompt = "neon city at night, a woman with a red umbrella, shot on film"
    changed = _validate(
        _answer(reordered="a woman with an umbrella, neon city at night, shot on film", rewrite=None),
        prompt=prompt,
    )
    assert changed.reordered is None
    assert changed.set_aside == [
        "The critic's reordered version was left out: it changed words, not only their "
        "order; a reordering of your own can go under Compare."
    ]
    # Nothing moved is not a change of words; nothing to say.
    unmoved = _validate(_answer(reordered=prompt, rewrite=None), prompt=prompt)
    assert unmoved.reordered is None and unmoved.set_aside == []


def test_the_rewrites_quotation_marks_are_held_to_the_prompts_lettering():
    prompt = "A quiet street after the rain, a small shop with a wooden sign that says Open Late."
    # Wrapped whole: unwrapped, and said so.
    wrapped = _validate(_answer(rewrite=f'"{prompt[:-1]}, empty and glistening."'), prompt=prompt)
    assert wrapped.rewrite == f"{prompt[:-1]}, empty and glistening."
    assert wrapped.set_aside[0].startswith("The rewrite came back wrapped in quotation marks")
    # Wrapped around a quotation of its own: the inner marks still pair up.
    inner = 'A quiet street after the rain, a wooden sign that says "Open Late"'
    assert _validate(_answer(rewrite=f"“{inner}”"), prompt=prompt).rewrite == inner
    # Two quotations are not one wrap, nor is an apostrophe a quotation.
    assert critic._unwrapped('"Open Late" on the sign, "Closed" on the door') is None
    assert critic._unwrapped("'Open' on the sign, 'Closed' on the door") is None
    assert critic._unwrapped("'A dog's tail'") == "A dog's tail"
    assert critic._unwrapped('"A sign that says "Open Late""') == 'A sign that says "Open Late"'
    assert critic._unwrapped("A plain rewrite.") is None
    # The sign's own words quoted: what the guide asks for.
    good = _validate(_answer(rewrite=inner), prompt=prompt)
    assert good.rewrite == inner and good.rewrite_findings == [] and good.set_aside == []
    # Lettering the prompt does not have - copied from an authors' example -
    # drops the rewrite.
    invented = _validate(
        _answer(rewrite=inner + ', a neon sign reading "Black Forest Labs"'), prompt=prompt
    )
    assert invented.rewrite is None
    assert invented.set_aside == [
        'The rewrite was left out: it puts "Black Forest Labs" in quotation marks - '
        "lettering the prompt does not ask for."
    ]
    # The prompt's own description in quotes is flagged where it sits.
    described = 'A "quiet street after the rain", a wooden sign that says "Open Late"'
    result = _validate(_answer(rewrite=described), prompt=prompt)
    assert result.rewrite == described
    (flag,) = result.rewrite_findings
    assert flag.rule == "heuristic.rewrite_quote" and flag.severity == "warn"
    assert flag.source == "heuristic" and flag.cite is None
    assert described[flag.start:flag.end] == '"quiet street after the rain"'


def test_a_rewrite_that_adds_content_says_so():
    # Seen on the card, told not to: Krea's tea party example came back with
    # a lens, light, greenery and sunlight it never asked for.
    prompt = "Photo of two people having a tea party in a lush garden"
    padded = (
        "Photo of two people having a tea party in a lush garden, wide-angle lens, shallow "
        "depth of field, soft natural lighting, cinematic composition, vibrant greenery"
    )
    (adds,) = _validate(_answer(rewrite=padded), prompt=prompt).rewrite_findings
    assert adds.rule == "heuristic.rewrite_adds" and adds.severity == "info"
    assert adds.source == "heuristic" and adds.start is None
    assert "adds 12 words the prompt does not have (wide-angle, lens, shallow, depth" in adds.message
    # Rewording is not new content: "no people" -> "empty", an inflection, a
    # quotation mark closing on a word.
    reworded = "A photo of two people who have a tea party in a lush, empty garden"
    assert _validate(_answer(rewrite=reworded), prompt=prompt).rewrite_findings == []
    sign = "A quiet street, no people, a sign that says Open Late."
    quoted = "A quiet street, empty, a sign that says 'Open Late'."
    assert _validate(_answer(rewrite=quoted), prompt=sign).rewrite_findings == []
    # Words run together in Chinese; nothing is counted there.
    chinese = "一只猫在沙发上"
    longer = "一只橘色的猫在阳光下的旧沙发上睡觉，背景是木地板和绿植，电影感光线"
    assert _validate(_answer(rewrite=longer), prompt=chinese).rewrite_findings == []


def test_the_answer_is_parsed_after_any_inline_thinking():
    content = "<think>the subject is {a woman}</think>\n" + json.dumps(_answer())
    assert _validate(content).issues[0].cite_id == "G3"
    for broken in ("no json here", '{"parts": ', "[1, 2]"):
        with pytest.raises(critic.CriticError):
            _validate(broken)


def test_a_think_tag_the_answer_quotes_does_not_cut_it():
    # A prompt pasted from another model's output can hold a </think>, and the
    # answer quotes the prompt back - in its parts ahead of the issues, or in
    # its rewrite at the end - whether or not the thinking came inline.
    prompt = PROMPT + ', a neon sign that reads "</think>"'
    parts = {**_answer()["parts"], "text_to_render": "</think>"}
    rewrite = 'A neon sign that reads "</think>" above a shop.'
    for answer in (_answer(parts=parts), _answer(rewrite=rewrite)):
        for content in (json.dumps(answer), "<think>{a woman}</think>\n" + json.dumps(answer)):
            result = _validate(content, prompt=prompt)
            assert [issue.cite_id for issue in result.issues] == ["G3"], content
            assert result.parts["subject"] == "a woman with a red umbrella"
            assert result.parts["text_to_render"] == answer["parts"]["text_to_render"]
            assert result.rewrite == answer["rewrite"]
    # Inline thinking can draft an answer of its own; the one after it counts.
    draft = json.dumps(_answer(rewrite="A draft the model threw away."))
    result = _validate(
        f"<think>First, {draft}</think>\n" + json.dumps(_answer(rewrite=rewrite)), prompt=prompt
    )
    assert result.rewrite == rewrite
    # An object with none of the answer's keys is not an empty answer.
    for other in (json.dumps(_answer()["issues"][0]), '<think>x</think> {"rewrit": "x"}'):
        with pytest.raises(critic.CriticError, match="not the JSON object it was asked for"):
            _validate(other)


# --- The question ------------------------------------------------------------


def test_schema_enumerates_exactly_the_numbered_sources():
    context = _context()
    ids = context.schema["properties"]["issues"]["items"]["properties"]["cite_id"]["enum"]
    assert ids == list(context.citations)
    sources = prompt_sources.sources_for(KLEIN)
    assert ids == [f"G{i}" for i in range(1, len(sources.sentences) + 1)] + [
        f"A{i}" for i in range(1, len(sources.excerpts) + 1)
    ]
    for cite_id in ids:
        assert f"\n{cite_id}: " in context.system
    assert context.schema["required"] == ["parts", "issues", "rewrite", "reordered"]
    assert context.schema["additionalProperties"] is False
    assert context.schema["properties"]["issues"]["maxItems"] == critic.MAX_ISSUES
    assert context.schema["properties"]["parts"]["required"] == list(critic.PARTS)
    assert context.schema["properties"]["reordered"] == {"type": ["string", "null"]}
    assert _context(ordered=False).schema["properties"]["reordered"] == {"type": "null"}


def test_the_messages_carry_the_request_as_it_runs():
    runs = _context(ANIMA, "1girl, smile", ordered=False, negative="lowres")
    assert "Negative prompt (it runs):\nlowres" in runs.user
    assert "No negative prompt runs" not in runs.system
    assert "reordered: null" in runs.system and "DOCUMENTED ORDER" not in runs.system
    # Anima's card does give a tag order - as A1 - and nothing may say it
    # does not: the app has no check of its own, the authors have an order.
    assert "document no order" not in runs.system
    assert "An order a source below states may still be an issue" in runs.system
    assert "\nA1: [quality/meta/year/safety tags] [1girl/1boy/1other etc]" in runs.system
    silent = _context(KLEIN)
    assert silent.user == f"Prompt:\n{PROMPT}"
    assert "No negative prompt runs" in silent.system
    assert "DOCUMENTED ORDER: subject > action > style > context" in silent.system
    # Quotation marks are asked for where the guide asks for them, not on a
    # tag checkpoint whose guide never mentions rendered text - and only for
    # lettering: the card's critic wrapped whole prompts on "text in quotes".
    lettering = "Put quotation marks only around words the image must show as lettering"
    assert lettering in silent.system and "never around the prompt" in silent.system
    assert lettering not in runs.system
    # The rewrite adds nothing and pads nothing; the examples lend their form.
    for context in (runs, silent):
        assert "Add no object, scenery, mood or text the prompt lacks" in context.system
        assert "not a minimum" in context.system
    assert "AUTHORS' EXAMPLE PROMPTS (their form only; never copy their subjects or text)" in (
        silent.system
    )


def test_the_flux1_family_is_told_how_bfl_reads_its_order():
    # On the card a critic held to a bare "subject first" called Krea's own
    # "Moody night photography, a small store..." out of order, told the same
    # prompt to put its style first, and cut camera details as a fault.
    for model_id in ("black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-dev",
                     "black-forest-labs/FLUX.1-schnell"):
        context = _context(model_id)
        assert "'Moody night photography') still puts the subject first" in context.system
        assert "Camera settings are part of BFL's template, not a fault" in context.system
        texts = {citation.text for citation in context.citations.values()}
        assert "Start with the image type and subject." in texts
        assert any("not a strict formula" in text and "[CAMERA SETTINGS]" in text for text in texts)


def test_krea_dev_is_told_its_sources_are_bfls():
    context = _context("black-forest-labs/FLUX.1-Krea-dev")
    assert "Krea published no prompting rules" in context.system
    # A shared note is written once and pointed back to.
    note = "(BFL's guide for the whole FLUX family; Krea published no prompting rules of its own)"
    assert context.system.count(note) == 1
    assert "(as A1)" in context.system


@pytest.mark.parametrize("model_id", sorted(set(_catalogue_repos())))
def test_the_system_message_fits_its_budget(model_id):
    findings = [
        PromptFinding(rule="encoder.cut", severity="warn", message="m" * 400, source="encoder")
        for _ in range(20)
    ]
    request = GenerateRequest(prompt="a cat", negative_prompt="blurry")
    analysis = PromptAnalysis(
        model_id=model_id,
        style="prose",
        words=2,
        findings=findings,
        order=DocumentedOrder(steps=["a", "b", "c", "d", "e"], cite="c" * 400),
    )
    context = critic.build_context(model_id, request, request, analysis)
    assert len(context.system) <= critic.SYSTEM_BUDGET_CHARS, len(context.system)
    assert context.system.count("\n- encoder.cut") == critic._MAX_FINDINGS


# --- The grounding -----------------------------------------------------------


def test_every_guide_constant_is_grounded():
    assert set(prompt_sources.GROUNDING) == set(prompt_sources.guide_constants())


@pytest.mark.parametrize("model_id", _catalogue_repos())
def test_every_catalogue_checkpoint_resolves_to_its_sources(model_id):
    sources = prompt_sources.sources_for(model_id)
    assert sources.guide is _profile_for(model_id)["guide"]
    assert sources.guide_name != "FALLBACK_GUIDE"
    assert sources.sentences
    # The page the rule checks cite, so the two layers cannot diverge.
    assert sources.url == sources.guide["analysis"]["url"]
    assert sources.url.startswith("https://")
    own = dict(prompt_sources.GROUNDING[sources.guide_name].sentence_urls)
    for sentence in sources.sentences:
        assert sentence.url == own.get(sentence.text, sources.url), sentence.text
    assert sources.excerpts


def test_an_unknown_checkpoint_has_only_the_fallback():
    sources = prompt_sources.sources_for("nobody/whatever")
    assert sources.guide_name == "FALLBACK_GUIDE" and sources.guide is guides.FALLBACK_GUIDE
    assert sources.excerpts == () and sources.url is None
    assert "no author sources" in sources.caveat
    assert _context("nobody/whatever", ordered=False).citations


def test_unsourced_sentences_are_not_citable():
    for name, grounding in prompt_sources.GROUNDING.items():
        sentences = prompt_sources.guide_sentences(prompt_sources.guide_constants()[name]["prompt"])
        # Verbatim sentences of the guide, or a guide edit has left them behind.
        for sentence in grounding.unsourced:
            assert sentence in sentences, (name, sentence)
        for sentence, url in grounding.sentence_urls:
            assert sentence in sentences and url.startswith("https://"), (name, sentence)
    krea = prompt_sources.sources_for("black-forest-labs/FLUX.1-Krea-dev")
    texts = [sentence.text for sentence in krea.sentences]
    assert not any("photo language" in text or "give it room" in text for text in texts)
    # Its prompting links to BFL's family guide; only the settings to Krea.
    assert krea.sentences[0].url == krea.url and krea.url.startswith("https://docs.bfl.ai/")
    assert krea.sentences[-1].url == "https://github.com/krea-ai/flux-krea"
    context = _context("black-forest-labs/FLUX.1-Krea-dev")
    assert "photo language" not in context.system and "give it room" not in context.system
    # Illustrious's sentences link to the report its rules come from, not
    # the v2.0 card, which has no prompting text.
    illustrious = prompt_sources.sources_for("OnomaAIResearch/Illustrious-XL-v2.0")
    assert {sentence.url for sentence in illustrious.sentences} == {
        "https://arxiv.org/html/2409.19946v1"
    }


def test_the_sentences_are_the_guide_prose():
    for name, guide in prompt_sources.guide_constants().items():
        sentences = prompt_sources.guide_sentences(guide["prompt"])
        assert " ".join(sentences) == " ".join(guide["prompt"].split()), name
        # The splitter takes ". " for a sentence end; an abbreviation would
        # cut a sentence in two.
        assert not re.search(r"\b(e\.g|i\.e|etc|vs)\. ", guide["prompt"]), name


def test_excerpts_are_short_sourced_and_labelled():
    for name, grounding in prompt_sources.GROUNDING.items():
        ids = [excerpt.id for excerpt in grounding.excerpts]
        assert len(ids) == len(set(ids)), name
        for excerpt in grounding.excerpts:
            assert excerpt.url.startswith("https://"), excerpt.id
            assert 0 < len(excerpt.text) <= 600, excerpt.id
            assert "{" not in excerpt.text and "civitai.com" not in excerpt.url, excerpt.id
            if excerpt.original is not None:
                # A translation says so, and keeps the words it translates.
                assert "translation" in (excerpt.note or ""), excerpt.id
                assert re.search(r"[\u4e00-\u9fff]", excerpt.original), excerpt.id
            else:
                assert not re.search(r"[\u4e00-\u9fff]", excerpt.text), excerpt.id


def test_author_orders_are_marked_only_where_the_guide_states_none():
    # Shown under Order as the authors' - never "no order is documented" -
    # and only where the app checks none of its own.
    marked = set()
    for name, grounding in prompt_sources.GROUNDING.items():
        ordered = [excerpt.id for excerpt in grounding.excerpts if excerpt.order]
        if ordered:
            assert "order" not in prompt_sources.guide_constants()[name].get("analysis", {}), name
        marked.update(ordered)
    assert marked == {
        "anima.order", "neta.sequence", "neta.prose_order", "ernie.sections", "qwen.portrait_flow",
    }


def test_borrowed_sources_say_whose_they_are():
    grounding = prompt_sources.GROUNDING
    assert "Krea published no prompting rules" in grounding["KREA_DEV_GUIDE"].caveat
    assert all("Krea published no" in e.note for e in grounding["KREA_DEV_GUIDE"].excerpts
               if e.url.startswith("https://docs.bfl.ai/guides/prompting_unified"))
    assert all("inherited" in e.note for e in grounding["NETAYUME_GUIDE"].excerpts)
    assert "system prefix" in grounding["NETAYUME_GUIDE"].caveat
    flux2 = [e for e in grounding["KLEIN_GUIDE"].excerpts if e.url.endswith("prompting_guide_flux2")]
    assert flux2 and all("klein by family" in e.note for e in flux2)
    assert "4-step" in grounding["Z_ANIME_GUIDE"].caveat
