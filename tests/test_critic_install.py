"""Pinned critic runtime installation and atomic cache publication."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import io
import json
import shutil
import sys
import tarfile
import threading

import huggingface_hub
import pytest

from backend import critic, critic_runtime, main

from tests.critic_helpers import (
    Release,
    configure_critic,
    link_directory,
    zip_bytes,
    root,
    windows_release,
)


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
            # Not executable, the server included: the install must set that.
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_install_verifies_extracts_and_publishes_once(root, windows_release):
    server = critic_runtime.install_server("test-x64")
    target = root / f"{critic_runtime.RELEASE}-test-x64"
    assert server == target / "llama-server.exe"
    assert server.read_bytes() == b"server"
    # Both archives land in one directory: the runtime beside the binary.
    assert (target / "ggml-cuda.dll").is_file() and (target / "cudart64_13.dll").is_file()
    assert json.loads((target / critic_runtime._INSTALL_MARKER).read_text())["release"] == critic_runtime.RELEASE
    assert critic_runtime.installed_server("test-x64") == server
    # The verified archives are not kept once installed, nor is any work dir.
    assert not list((root / "downloads").iterdir())
    assert sorted(path.name for path in root.iterdir() if path.is_dir()) == [
        "b11206-test-x64", "downloads",
    ]
    assert critic_runtime.install_server("test-x64") == server
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip"]


def test_a_bad_checksum_is_refused_and_retried(root, windows_release):
    good = windows_release.payloads["cudart-test.zip"]
    # Same size, other bytes: the size check passes and the hash must not.
    windows_release.payloads["cudart-test.zip"] = bytes(len(good))
    with pytest.raises(critic_runtime.CriticError, match="sha256"):
        critic_runtime.install_server("test-x64")
    assert critic_runtime.installed_server("test-x64") is None
    assert not (root / f"{critic_runtime.RELEASE}-test-x64").exists()
    assert not list((root / "downloads").glob("*.part"))
    # The first archive verified and is reused; only the bad one is fetched again.
    assert (root / "downloads" / "llama-test.zip").is_file()
    windows_release.payloads["cudart-test.zip"] = good
    assert critic_runtime.install_server("test-x64").read_bytes() == b"server"
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip", "cudart-test.zip"]


def test_a_wrong_size_is_refused(root, windows_release):
    windows_release.payloads["llama-test.zip"] += b"x"
    with pytest.raises(critic_runtime.CriticError, match="bytes, not the pinned"):
        critic_runtime.install_server("test-x64")
    assert critic_runtime.installed_server("test-x64") is None


def test_an_archive_already_in_downloads_is_held_to_its_pin(root, windows_release):
    # Left there by anyone: the right size, other bytes. Reused only verified.
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    good = windows_release.payloads["llama-test.zip"]
    (downloads / "llama-test.zip").write_bytes(bytes(len(good)))
    assert critic_runtime.install_server("test-x64").read_bytes() == b"server"
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip"]


@pytest.mark.parametrize("member", ["../evil.dll", "/evil.dll", "C:/evil.dll", "a/../../evil.dll"])
def test_a_zip_member_escaping_the_target_is_refused(root, monkeypatch, member):
    Release(monkeypatch, {"llama-test.zip": zip_bytes({"llama-server.exe": b"s", member: b"evil"})})
    with pytest.raises(critic_runtime.CriticError, match="outside the install directory"):
        critic_runtime.install_server("test-x64")
    assert critic_runtime.installed_server("test-x64") is None
    assert not (root / f"{critic_runtime.RELEASE}-test-x64").exists()
    assert not list(root.parent.rglob("evil.dll"))


def test_a_tar_member_escaping_the_target_is_refused(root, monkeypatch):
    Release(
        monkeypatch,
        {"llama-test.tar.gz": _tar({"llama-server": b"s", "../evil.so": b"evil"})},
        server="llama-server",
    )
    with pytest.raises(critic_runtime.CriticError):
        critic_runtime.install_server("test-x64")
    assert critic_runtime.installed_server("test-x64") is None
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
    # The tarballs are Linux's; the chmod is seen on any host, and its effect
    # where the file system keeps the bits.
    monkeypatch.setattr(critic_runtime, "sys", SimpleNamespace(platform="linux"))
    chmods: list[tuple[str, int]] = []
    real_chmod = Path.chmod

    def chmod(path, mode, **kwargs):
        chmods.append((path.name, mode))
        real_chmod(path, mode, **kwargs)

    monkeypatch.setattr(Path, "chmod", chmod)
    server = critic_runtime.install_server("test-x64")
    assert server.read_bytes() == b"server"
    assert (server.parent / "libggml-cuda.so").is_file()
    assert (server.parent / "libcudart.so.13").is_file()
    assert [(name, mode & 0o111) for name, mode in chmods] == [("llama-server", 0o111)]
    if sys.platform != "win32":
        assert server.stat().st_mode & 0o111 == 0o111


def test_an_archive_without_the_server_is_not_installed(root, monkeypatch):
    Release(monkeypatch, {"llama-test.zip": zip_bytes({"llama-cli.exe": b"cli"})})
    with pytest.raises(critic_runtime.CriticError, match="llama-server.exe"):
        critic_runtime.install_server("test-x64")
    assert critic_runtime.installed_server("test-x64") is None


def test_an_unmarked_tree_is_never_used_and_is_replaced(root, windows_release):
    target = root / f"{critic_runtime.RELEASE}-test-x64"
    target.mkdir(parents=True)
    (target / "llama-server.exe").write_bytes(b"half")
    assert critic_runtime.installed_server("test-x64") is None
    assert critic_runtime.install_server("test-x64").read_bytes() == b"server"
    # A marker from another recipe is as good as none.
    (target / critic_runtime._INSTALL_MARKER).write_text("{}")
    assert critic_runtime.installed_server("test-x64") is None


def test_an_undecodable_marker_is_no_install(root, windows_release, monkeypatch, tmp_path, client):
    server = critic_runtime.install_server("test-x64")
    marker = server.parent / critic_runtime._INSTALL_MARKER
    # The recipe, re-saved the way PowerShell 5.1's Out-File saves: UTF-16.
    marker.write_bytes(critic_runtime._recipe("test-x64").encode("utf-16"))
    assert critic_runtime.installed_server("test-x64") is None

    # Not ready, and the status poll that asks still answers.
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: "test-x64")
    configure_critic(monkeypatch, critic=True, critic_server=None, critic_model=None)
    gguf = tmp_path / "Qwen3-4B-Q8_0.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(gguf))
    monkeypatch.setattr(main, "prompt_critic", critic.PromptCritic())
    assert not main.prompt_critic.ready()
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["prompt_critique_supported"] is True
    assert status.json()["prompt_critique_ready"] is False

    # The install replaces the tree rather than failing on it.
    assert critic_runtime.install_server("test-x64") == server
    assert marker.read_text(encoding="utf-8") == critic_runtime._recipe("test-x64")
    assert main.prompt_critic.ready()
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip"] * 2


def test_an_interrupted_installs_work_directories_are_swept(root, windows_release, monkeypatch):
    target = root / f"{critic_runtime.RELEASE}-test-x64"
    stale = root / f"{target.name}-k3j9x0q1"
    (stale / "tree").mkdir(parents=True)
    (stale / "tree" / "llama-server.exe").write_bytes(b"half")
    # Beside it, what is not an install's work, though named like one:
    # another build's own directory, whose key is this one's and eight
    # characters tempfile could write - empty, so only its name tells it
    # apart - and a file.
    monkeypatch.setitem(critic_runtime.BUILDS, "test-x64-avx2cuda", critic_runtime.BUILDS["test-x64"])
    other = root / f"{target.name}-avx2cuda"
    other.mkdir()
    note = root / f"{target.name}-notes123"
    note.write_text("mine")
    server = critic_runtime.install_server("test-x64")
    assert not stale.exists()
    assert other.is_dir() and note.read_text() == "mine"
    # Also once installed: what an earlier interrupted attempt left is found
    # by the next prepare, which only checks the install.
    stale.mkdir()
    assert critic_runtime.install_server("test-x64") == server
    assert not stale.exists()
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip"]


def test_the_sweep_takes_only_what_an_install_leaves(root, windows_release):
    # What an interruption leaves: the tree half-extracted, the archives'
    # scratch directories, or a work directory created and nothing more.
    target = root / f"{critic_runtime.RELEASE}-test-x64"
    left = {
        "tree": root / f"{target.name}-k3j9x0q1",
        "scratch": root / f"{target.name}-_a8b9c0d",
        "empty": root / f"{target.name}-zz00zz00",
    }
    (left["tree"] / "tree").mkdir(parents=True)
    (left["tree"] / "tree" / "ggml-cuda.dll").write_bytes(b"half")
    (left["tree"] / "archive-0").mkdir()
    (left["scratch"] / "archive-1" / "llama-b11206").mkdir(parents=True)
    left["empty"].mkdir()
    # And a user's, beside it: a backup of the install by name, a name
    # tempfile could make holding a copy of the install, work-shaped
    # contents under names seven and nine characters long or in capitals,
    # which tempfile never writes, and tempfile-shaped names holding entries
    # that only start like an install's: its entries are whole names.
    kept = {
        "backup": root / f"{target.name}-backup",
        "copy": root / f"{target.name}-backup01",
        "short": root / f"{target.name}-k3j9x0q",
        "long": root / f"{target.name}-k3j9x0q12",
        "capitals": root / f"{target.name}-QX7RB2KP",
        "tree.bak": root / f"{target.name}-bak_0001",
        "archive-0.zip": root / f"{target.name}-zip_0001",
    }
    for path in kept.values():
        (path / "tree").mkdir(parents=True)
    (kept["copy"] / "llama-server.exe").write_bytes(b"mine")
    (kept["copy"] / critic_runtime._INSTALL_MARKER).write_text("{}")
    (kept["tree.bak"] / "tree.bak").mkdir()
    (kept["archive-0.zip"] / "archive-0.zip").write_bytes(b"mine")
    critic_runtime.install_server("test-x64")
    for name, path in left.items():
        assert not path.exists(), name
    for name, path in kept.items():
        assert (path / "tree").is_dir(), name
    assert (kept["copy"] / "llama-server.exe").read_bytes() == b"mine"


def test_the_sweep_never_takes_a_link_named_like_work(root, windows_release, tmp_path, caplog):
    # A link or junction whose target holds exactly what a work directory
    # would is still no install's: what it points at is someone else's
    # directory, never the sweep's to remove or to warn about on every use.
    target = root / f"{critic_runtime.RELEASE}-test-x64"
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "tree").mkdir(parents=True)
    (elsewhere / "tree" / "keep.txt").write_text("mine")
    root.mkdir(parents=True)
    link = root / f"{target.name}-linked01"
    link_directory(link, elsewhere)
    assert not critic_runtime._is_work(link, target)
    with caplog.at_level("WARNING", logger=critic_runtime.log.name):
        critic_runtime.install_server("test-x64")
    assert link.is_dir() and (elsewhere / "tree" / "keep.txt").read_text() == "mine"
    assert "interrupted llama.cpp install" not in caplog.text


def test_the_sweep_never_fails_an_install(root, windows_release, monkeypatch, caplog):
    # Work with a file another process still holds, and work whose listing is
    # refused: neither stops the install, neither is reported gone, and the
    # sweep still takes what it can beside them. The held one is tried again
    # by the next install.
    target = root / f"{critic_runtime.RELEASE}-test-x64"
    stale = root / f"{target.name}-k3j9x0q1"
    (stale / "tree").mkdir(parents=True)
    held = root / f"{target.name}-held_0a1"
    (held / "tree").mkdir(parents=True)
    unlisted = root / f"{target.name}-unlist_1"
    (unlisted / "tree").mkdir(parents=True)
    refused = {unlisted}
    holding = {held}
    rmtree, iterdir = shutil.rmtree, Path.iterdir

    def held_rmtree(path, *args, ignore_errors=False, **kwargs):
        if Path(path) not in holding:
            return rmtree(path, *args, ignore_errors=ignore_errors, **kwargs)
        if not ignore_errors:
            raise PermissionError(32, "The process cannot access the file", str(held / "tree"))

    def refusing_iterdir(self):
        if self in refused:
            raise PermissionError(13, "Access is denied", str(self))
        return iterdir(self)

    monkeypatch.setattr(shutil, "rmtree", held_rmtree)
    monkeypatch.setattr(Path, "iterdir", refusing_iterdir)
    with caplog.at_level("WARNING", logger=critic_runtime.log.name):
        server = critic_runtime.install_server("test-x64")
    assert server.read_bytes() == b"server"
    assert not stale.exists()
    assert held.is_dir() and unlisted.is_dir()
    assert f"could not remove an interrupted llama.cpp install yet: {held}" in caplog.text
    assert f"removed an interrupted llama.cpp install: {held}" not in caplog.text
    # Let go of, it goes on the next install, which finds the build in place.
    holding.clear()
    caplog.clear()
    with caplog.at_level("WARNING", logger=critic_runtime.log.name):
        assert critic_runtime.install_server("test-x64") == server
    assert not held.exists() and unlisted.is_dir()
    assert f"removed an interrupted llama.cpp install: {held}" in caplog.text
    # Nor does an install directory that cannot be listed.
    refused.add(root)
    assert critic_runtime.install_server("test-x64") == server
    assert windows_release.fetched == ["llama-test.zip", "cudart-test.zip"]


def test_the_sweep_waits_for_an_install_holding_the_lock(root, windows_release):
    # Another process's install, mid-extraction: it holds the lock, and its
    # work directory is live until it lets go.
    import threading

    from huggingface_hub.utils import WeakFileLock

    server = critic_runtime.install_server("test-x64")
    target = server.parent
    live = root / f"{target.name}-live0001"
    answer: list[Path] = []
    with WeakFileLock(root / f"{target.name}.lock"):
        live.mkdir()
        worker = threading.Thread(target=lambda: answer.append(critic_runtime.install_server("test-x64")))
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and live.is_dir()
    worker.join(10)
    assert answer == [server]
    assert not live.exists()
