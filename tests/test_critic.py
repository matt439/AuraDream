"""Critic runtime pins, platform support, preparation, and verified model cache."""
from __future__ import annotations

from types import SimpleNamespace
import dataclasses
import hashlib
import os
import re
import sys

import huggingface_hub
import pytest

from backend import critic, critic_runtime, main

from tests.critic_helpers import (
    _configure,
    _link_directory,
    root,
    windows_release,
)


# --- The pinned table --------------------------------------------------------


def test_pinned_builds_cover_windows_and_wsl():
    assert set(critic_runtime.BUILDS) == {"win-cuda-13.4-x64", "ubuntu-cuda-13.4-x64"}
    for key, build in critic_runtime.BUILDS.items():
        assert len(build.archives) == 2, key
        names = [archive.name for archive in build.archives]
        assert names[0].startswith(f"llama-{critic_runtime.RELEASE}-bin-") and key in names[0]
        assert names[1].startswith("cudart-llama-") and key in names[1]
        for archive in build.archives:
            assert re.fullmatch(r"[0-9a-f]{64}", archive.sha256), archive.name
            assert archive.size > 100_000_000, archive.name
            assert archive.url == (
                "https://github.com/ggml-org/llama.cpp/releases/download/b11206/" + archive.name
            )
    assert critic_runtime.BUILDS["win-cuda-13.4-x64"].server == "llama-server.exe"
    assert critic_runtime.BUILDS["ubuntu-cuda-13.4-x64"].server == "llama-server"
    assert critic_runtime.MODEL["repo_id"] == "Qwen/Qwen3-4B-GGUF"
    assert re.fullmatch(r"[0-9a-f]{40}", critic_runtime.MODEL["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", critic_runtime.MODEL_SHA256)
    assert critic_runtime.MODEL_SIZE == 4280404704


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
    monkeypatch.setattr(critic_runtime, "sys", SimpleNamespace(platform=system))
    monkeypatch.setattr(critic_runtime.platform, "machine", lambda: machine)
    assert critic_runtime.platform_key() == key


# --- supported, ready, prepare -----------------------------------------------


def test_supported_follows_the_switch_and_the_platform(monkeypatch, tmp_path):
    _configure(monkeypatch, critic=False, critic_server=None, critic_model=None)
    assert not critic.prompt_critic.supported()
    assert not critic.prompt_critic.ready()
    with pytest.raises(critic_runtime.CriticUnavailable, match="switched off"):
        critic.prompt_critic.prepare()

    _configure(monkeypatch, critic=True)
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: "win-cuda-13.4-x64")
    assert critic.prompt_critic.supported()
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: None)
    assert not critic.prompt_critic.supported()
    # A server of the user's own makes any platform supported.
    _configure(monkeypatch, critic_server=tmp_path / "llama-server")
    assert critic.prompt_critic.supported()


UBUNTU = "ubuntu-cuda-13.4-x64"


def _no_downloads(monkeypatch) -> None:
    monkeypatch.setattr(critic_runtime, "install_server", lambda key: pytest.fail("installed"))
    monkeypatch.setattr(critic_runtime, "download_model", lambda: pytest.fail("downloaded"))


def test_the_linux_build_is_offered_only_on_a_glibc_that_runs_it(monkeypatch, tmp_path):
    # The probes are this test's: nothing depends on the host being Linux.
    _configure(monkeypatch, critic=True, critic_server=None, critic_model=None)
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: UBUNTU)
    monkeypatch.setattr(critic_runtime, "missing_libraries", lambda names: pytest.fail("probed"))
    _no_downloads(monkeypatch)
    # Ubuntu 22.04, Debian 12, and a C library that is not glibc (musl).
    for old in [(2, 35), (2, 36), None]:
        monkeypatch.setattr(critic_runtime, "glibc_version", lambda old=old: old)
        assert not critic.prompt_critic.supported()
        with pytest.raises(critic_runtime.CriticUnavailable, match=r"needs glibc 2\.38 or newer"):
            critic.PromptCritic().prepare()
    # 2.38 itself (Ubuntu 23.10) runs it, and anything newer.
    for new in [(2, 38), (2, 39), (3, 0)]:
        monkeypatch.setattr(critic_runtime, "glibc_version", lambda new=new: new)
        assert critic.prompt_critic.supported(), new
        assert critic_runtime.pinned_build_refusal() is None
    # A server of one's own is not held to the pinned build's needs.
    monkeypatch.setattr(critic_runtime, "glibc_version", lambda: (2, 35))
    _configure(monkeypatch, critic_server=tmp_path / "llama-server")
    assert critic.prompt_critic.supported()
    # Windows' build asks nothing of the system.
    monkeypatch.setattr(critic_runtime, "glibc_version", lambda: pytest.fail("asked"))
    critic_runtime.preflight("win-cuda-13.4-x64")


def test_the_critique_route_says_the_glibc_is_too_old(client, monkeypatch):
    # Ubuntu 22.04 under WSL: a build is published for the platform, and this
    # system cannot run it - not that there is none.
    _configure(monkeypatch, critic=True, critic_server=None, critic_model=None)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, critic=True))
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: UBUNTU)
    monkeypatch.setattr(critic_runtime, "glibc_version", lambda: (2, 35))
    _no_downloads(monkeypatch)
    monkeypatch.setattr(main, "prompt_critic", critic.PromptCritic())
    assert client.get("/api/status").json()["prompt_critique_supported"] is False
    reply = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert reply.status_code == 422
    detail = reply.json()["detail"]
    assert "needs glibc 2.38 or newer" in detail and "has glibc 2.35" in detail
    assert "LOCAL_DIFFUSION_CRITIC_SERVER" in detail
    # A platform with no build at all is still told so.
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: None)
    reply = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert reply.status_code == 422
    assert reply.json()["detail"].startswith("llama.cpp publishes no pinned llama-server build")


def test_glibc_version_reads_the_c_librarys_own_answer(monkeypatch):
    answers = {"glibc 2.39": (2, 39), "glibc 2.38.9000": (2, 38), "musl 1.2": None, "": None}
    for answer, version in answers.items():
        monkeypatch.setattr(critic_runtime.os, "confstr", lambda name, a=answer: a, raising=False)
        assert critic_runtime.glibc_version() == version, answer

    def unknown(name):
        raise ValueError("unrecognized configuration name")

    monkeypatch.setattr(critic_runtime.os, "confstr", unknown, raising=False)
    assert critic_runtime.glibc_version() is None
    monkeypatch.delattr(critic_runtime.os, "confstr")  # Windows has none
    assert critic_runtime.glibc_version() is None


def test_a_missing_library_is_named_with_its_package_before_any_download(
    monkeypatch, tmp_path
):
    _configure(monkeypatch, critic=True, critic_server=None, critic_model=None)
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: UBUNTU)
    monkeypatch.setattr(critic_runtime, "glibc_version", lambda: (2, 39))
    missing: list[str] = ["libgomp.so.1"]
    asked: list[tuple[str, ...]] = []
    monkeypatch.setattr(critic_runtime, "missing_libraries", lambda names: asked.append(names) or missing)
    _no_downloads(monkeypatch)
    target = critic.PromptCritic()
    # One `apt install` away, so offered - and prepare() says which.
    assert target.supported()
    with pytest.raises(critic_runtime.CriticUnavailable, match=r"libgomp\.so\.1.*sudo apt install libgomp1$"):
        target.prepare()
    assert asked == [("libgomp.so.1", "libssl.so.3", "libcrypto.so.3")]
    missing[:] = ["libssl.so.3", "libcrypto.so.3"]
    with pytest.raises(critic_runtime.CriticUnavailable, match=r"sudo apt install libssl3$"):
        target.prepare()
    # Installed since: the next prepare goes on, with no restart.
    missing[:] = []
    server, gguf = tmp_path / "llama-server", tmp_path / "model.gguf"
    server.write_bytes(b"s")
    gguf.write_bytes(b"m")
    monkeypatch.setattr(critic_runtime, "install_server", lambda key: server)
    monkeypatch.setattr(critic_runtime, "download_model", lambda: gguf)
    target.prepare()
    assert target._prepared == (server, gguf)
    # A server of one's own is not probed for the pinned build's libraries.
    monkeypatch.setattr(critic_runtime, "missing_libraries", lambda names: pytest.fail("probed"))
    _configure(monkeypatch, critic_server=server)
    target.prepare()


def test_the_library_probe_asks_the_loader_in_a_child(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="libgomp.so.1\n", returncode=0)

    monkeypatch.setattr(critic_runtime.subprocess, "run", run)
    assert critic_runtime.missing_libraries(("libgomp.so.1", "libssl.so.3")) == ["libgomp.so.1"]
    [(command, kwargs)] = calls
    assert command[:4] == [sys.executable, "-I", "-S", "-c"]
    assert command[5:] == ["libgomp.so.1", "libssl.so.3"]
    assert kwargs["check"] and kwargs["timeout"]

    def fail(command, **kwargs):
        raise OSError("no interpreter")

    # A probe that cannot run checks nothing: the server's exit will say.
    monkeypatch.setattr(critic_runtime.subprocess, "run", fail)
    assert critic_runtime.missing_libraries(("libgomp.so.1",)) == []


def test_the_library_probe_prints_what_the_loader_cannot_find(monkeypatch, capsys):
    # The child's script itself, run here on names that load nothing new.
    present = "kernel32" if sys.platform == "win32" else "libc.so.6"
    monkeypatch.setattr(sys, "argv", ["-c", "libnot-a-library.so.0", present])
    exec(critic_runtime._LIBRARY_PROBE, {})
    assert capsys.readouterr().out.split() == ["libnot-a-library.so.0"]


def test_ready_is_what_is_on_disk(monkeypatch, tmp_path, root, windows_release):
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: "test-x64")
    _configure(monkeypatch, critic=True, critic_server=None, critic_model=None)
    cached: list[str | None] = [None]
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *args, **kwargs: cached[0]
    )
    assert critic.prompt_critic.supported() and not critic.prompt_critic.ready()
    critic_runtime.install_server("test-x64")
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


def test_describe_names_the_model_and_only_what_a_first_use_fetches(
    monkeypatch, tmp_path, root, windows_release
):
    # The line under the critic's button: the model as the answer's footer
    # names it, whether it is the pinned pair the page's time was measured
    # on, and what a first use would download - an override's file never.
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: "test-x64")
    _configure(monkeypatch, critic=True, critic_server=None, critic_model=None)
    cached: list[str | None] = [None]
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *args, **kwargs: cached[0]
    )
    server_size = sum(archive.size for archive in critic_runtime.BUILDS["test-x64"].archives)
    target = critic.PromptCritic()
    fresh = target.describe()
    assert (fresh.label, fresh.pinned) == (critic_runtime.MODEL_LABEL, True)
    assert fresh.fetch == [f"llama.cpp {critic_runtime.RELEASE}", critic_runtime.MODEL_LABEL]
    assert fresh.fetch_bytes == server_size + critic_runtime.MODEL_SIZE
    critic_runtime.install_server("test-x64")
    # From here, a status-poll question: nothing fetched, nothing hashed.
    fetched = list(windows_release.fetched)
    monkeypatch.setattr(critic_runtime, "_sha256", lambda path: pytest.fail("hashed"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: pytest.fail("fetched"))
    installed = target.describe()
    assert (installed.fetch, installed.fetch_bytes) == ([critic_runtime.MODEL_LABEL], critic_runtime.MODEL_SIZE)
    gguf = tmp_path / "Qwen3-4B-Q8_0.gguf"
    gguf.write_bytes(b"gguf")
    cached[0] = str(gguf)
    assert target.describe().fetch == [] and target.ready()

    # A GGUF of one's own beside the pinned server: its name, no time.
    own_model = tmp_path / "gemma-3-12b-Q4.gguf"
    _configure(monkeypatch, critic_model=own_model)
    mine = target.describe()
    assert (mine.label, mine.pinned, mine.fetch) == ("gemma-3-12b-Q4", False, [])
    assert mine.label == target._label(own_model)  # the footer's name
    # A server of one's own, the pinned model not cached: no llama.cpp.
    cached[0] = None
    _configure(monkeypatch, critic_server=tmp_path / "llama-server", critic_model=None)
    theirs = target.describe()
    assert (theirs.label, theirs.pinned) == (critic_runtime.MODEL_LABEL, False)
    assert (theirs.fetch, theirs.fetch_bytes) == ([critic_runtime.MODEL_LABEL], critic_runtime.MODEL_SIZE)
    # Both of one's own, neither there: never a download - prepare() says why.
    _configure(monkeypatch, critic_model=own_model)
    assert target.describe().fetch == [] and not target.ready()
    assert windows_release.fetched == fetched


def test_prepare_uses_configured_files_without_downloading(monkeypatch, tmp_path):
    server, model = tmp_path / "llama-server", tmp_path / "mine.gguf"
    server.write_bytes(b"s")
    model.write_bytes(b"m")
    _configure(monkeypatch, critic=True, critic_server=server, critic_model=model)
    monkeypatch.setattr(critic_runtime, "install_server", lambda key: pytest.fail("installed"))
    target = critic.PromptCritic()
    target.prepare()
    assert target._prepared == (server, model)
    assert target._label(model) == "mine"

    _configure(monkeypatch, critic_model=tmp_path / "missing.gguf")
    with pytest.raises(critic_runtime.CriticUnavailable, match="LOCAL_DIFFUSION_CRITIC_MODEL"):
        target.prepare()


def test_prepare_downloads_the_pinned_model_and_hashes_it_once(monkeypatch, tmp_path, root):
    gguf = tmp_path / "blob.gguf"
    gguf.write_bytes(b"pinned weights")
    monkeypatch.setattr(critic_runtime, "MODEL_SIZE", gguf.stat().st_size)
    monkeypatch.setattr(critic_runtime, "MODEL_SHA256", hashlib.sha256(b"pinned weights").hexdigest())
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(gguf)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    hashes = []
    real = critic_runtime._sha256
    monkeypatch.setattr(critic_runtime, "_sha256", lambda path: hashes.append(path) or real(path))
    server = tmp_path / "llama-server"
    server.write_bytes(b"s")
    _configure(monkeypatch, critic=True, critic_server=server, critic_model=None)

    target = critic.PromptCritic()
    target.prepare()
    target.prepare()
    assert downloads == [critic_runtime.MODEL, critic_runtime.MODEL]
    assert target._prepared == (server, gguf)
    assert target._label(gguf) == critic_runtime.MODEL_LABEL
    assert len(hashes) == 1


def test_a_model_off_its_pin_is_refused(monkeypatch, tmp_path, root):
    gguf = tmp_path / "blob.gguf"
    gguf.write_bytes(b"other weights!")
    monkeypatch.setattr(critic_runtime, "MODEL_SIZE", gguf.stat().st_size)
    with pytest.raises(critic_runtime.CriticError, match="sha256"):
        critic_runtime.verify_model(gguf)
    monkeypatch.setattr(critic_runtime, "MODEL_SIZE", 1)
    with pytest.raises(critic_runtime.CriticError, match="bytes"):
        critic_runtime.verify_model(gguf)


GOOD_WEIGHTS = b"pinned weights"


def _pin(monkeypatch, data: bytes = GOOD_WEIGHTS) -> None:
    monkeypatch.setattr(critic_runtime, "MODEL_SIZE", len(data))
    monkeypatch.setattr(critic_runtime, "MODEL_SHA256", hashlib.sha256(data).hexdigest())


def test_the_stamp_spares_only_the_file_it_was_written_for(monkeypatch, tmp_path, root):
    _pin(monkeypatch)
    stamped = tmp_path / "stamped.gguf"
    stamped.write_bytes(GOOD_WEIGHTS)
    critic_runtime.verify_model(stamped)
    hashes = []
    real = critic_runtime._sha256
    monkeypatch.setattr(critic_runtime, "_sha256", lambda path: hashes.append(path.name) or real(path))
    critic_runtime.verify_model(stamped)
    assert hashes == []
    before = stamped.stat()
    # Another file of the same size and mtime: the stamp is not its.
    other = tmp_path / "other.gguf"
    other.write_bytes(bytes(len(GOOD_WEIGHTS)))
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(critic_runtime.CriticError, match="sha256"):
        critic_runtime.verify_model(other)
    # The stamped file rewritten in place at its size, so only its mtime moved.
    stamped.write_bytes(bytes(len(GOOD_WEIGHTS)))
    os.utime(stamped, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    with pytest.raises(critic_runtime.CriticError, match="sha256"):
        critic_runtime.verify_model(stamped)
    assert hashes == ["other.gguf", "stamped.gguf"]


def test_a_cached_model_off_its_pin_is_downloaded_again_once(monkeypatch, tmp_path, root):
    _pin(monkeypatch)
    blob = tmp_path / "hub" / "blobs" / critic_runtime.MODEL_SHA256
    entry = tmp_path / "hub" / "snapshots" / "rev" / "Qwen3-4B-Q8_0.gguf"
    blob.parent.mkdir(parents=True)
    entry.parent.mkdir(parents=True)
    blob.write_bytes(bytes(len(GOOD_WEIGHTS)))  # corrupted at the pinned size
    entry.write_bytes(blob.read_bytes())
    forced: list[bool] = []

    # The Hub cache as hf_hub_download keeps it: a missing snapshot entry is
    # made from the blob already held (a link, or a copy where there are no
    # symlinks), only a forced download replaces the blob, and an entry that
    # is there is left alone, forced or not.
    def download(force_download=False, **kwargs):
        assert kwargs == critic_runtime.MODEL
        forced.append(force_download)
        if force_download or not blob.is_file():
            blob.write_bytes(GOOD_WEIGHTS)
        if not entry.is_file():
            entry.write_bytes(blob.read_bytes())
        return str(entry)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    assert critic_runtime.download_model() == entry
    assert entry.read_bytes() == GOOD_WEIGHTS
    assert forced == [False, True]
    # Verified and stamped: the next prepare neither fetches nor hashes.
    monkeypatch.setattr(critic_runtime, "_sha256", lambda path: pytest.fail("hashed"))
    assert critic_runtime.download_model() == entry
    assert forced == [False, True, False]


def test_a_model_still_off_its_pin_names_the_file_holding_it(monkeypatch, tmp_path, root):
    _pin(monkeypatch)
    blobs = tmp_path / "hub" / "blobs"
    blobs.mkdir(parents=True)
    held = blobs / "Qwen3-4B-Q8_0.gguf"
    held.write_bytes(bytes(len(GOOD_WEIGHTS)))
    # The path handed out goes through a link, as a Hub snapshot's entry does:
    # the file that holds the bytes is the one it resolves to.
    _link_directory(tmp_path / "hub" / "snapshot", blobs)
    entry = tmp_path / "hub" / "snapshot" / "Qwen3-4B-Q8_0.gguf"
    forced: list[bool] = []

    def download(force_download=False, **kwargs):
        forced.append(force_download)
        if not entry.is_file():
            entry.write_bytes(bytes(len(GOOD_WEIGHTS)))  # fetched, and bad again
        return str(entry)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    with pytest.raises(critic_runtime.CriticError, match="sha256") as caught:
        critic_runtime.download_model()
    assert forced == [False, True]
    # Not "delete it and retry": the retry has just been made.
    message = str(caught.value)
    assert message.startswith(f"{held} has sha256")
    assert message.endswith(
        "even freshly downloaded at the pinned revision: the pin or the download path is wrong"
    )
    assert "retry" not in message


@pytest.mark.parametrize(
    "failure, reason",
    [
        # hf_hub_download's own words when the forced download's request
        # fails, with the request's error as the cause; and with no cause.
        (
            ValueError("Force download failed due to the above error."),
            "ConnectionError: connection refused",
        ),
        (OSError("disk full"), "OSError: disk full"),
    ],
)
def test_a_model_that_cannot_be_downloaded_again_says_why(
    monkeypatch, tmp_path, root, failure, reason
):
    _pin(monkeypatch)
    entry = tmp_path / "Qwen3-4B-Q8_0.gguf"
    entry.write_bytes(bytes(len(GOOD_WEIGHTS)))
    if "above error" in str(failure):
        failure.__cause__ = ConnectionError("connection refused")

    def download(force_download=False, **kwargs):
        if force_download:
            raise failure
        return str(entry)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    # A CriticError, so the route's 502 with the reason, not a bare 500.
    with pytest.raises(critic_runtime.CriticError) as caught:
        critic_runtime.download_model()
    assert str(caught.value) == (
        "Qwen3-4B-Q8_0.gguf in the Hub cache is off its pin and could not be "
        f"downloaded again: {reason}"
    )
    assert caught.value.__cause__ is failure
