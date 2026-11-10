"""The checkout's short commit (`build_info.read_commit`) and the running
server's own, asked on loopback (`server_probe.running_commit`). git is a
fake but for one read of this checkout; the server is a tiny local one."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import build_info, server_probe


def _git(stdout="abc1234\n", returncode=0, calls=None):
    def run(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return run


def test_the_commit_is_gits_short_one_asked_in_the_folder(tmp_path):
    (tmp_path / ".git").mkdir()
    calls: list = []
    assert build_info.read_commit(tmp_path, run=_git(calls=calls)) == "abc1234"
    [(argv, kwargs)] = calls
    assert argv == ["git", "-C", str(tmp_path), "rev-parse", "--short", "HEAD"]
    # Bounded, and never waiting on a prompt.
    assert kwargs["timeout"] == build_info.GIT_SECONDS and kwargs["stdin"] is subprocess.DEVNULL


def test_a_worktrees_git_file_counts_as_a_checkout(tmp_path):
    (tmp_path / ".git").write_text("gitdir: elsewhere\n", encoding="ascii")
    assert build_info.read_commit(tmp_path, run=_git()) == "abc1234"


def test_no_git_folder_is_none_without_asking_git(tmp_path):
    # A tree unpacked inside another repository is not that repository's commit.
    def never(*args, **kwargs):
        raise AssertionError("git was asked")
    assert build_info.read_commit(tmp_path, run=never) is None


@pytest.mark.parametrize("failure", [
    FileNotFoundError("git"),                              # not installed, or not on the PATH
    PermissionError("git"),
    subprocess.TimeoutExpired("git", build_info.GIT_SECONDS),
])
def test_git_that_does_not_run_is_none(tmp_path, failure):
    (tmp_path / ".git").mkdir()

    def run(*args, **kwargs):
        raise failure
    assert build_info.read_commit(tmp_path, run=run) is None


@pytest.mark.parametrize(("stdout", "returncode"), [
    ("abc1234\n", 128),                                    # git refused the folder
    ("", 0),
    ("HEAD\n", 0),                                         # an unborn branch echoes the name
    ("fatal: not a git repository\n", 0),
    ("abc1234 extra\n", 0),
])
def test_an_answer_that_is_not_a_commit_is_none(tmp_path, stdout, returncode):
    (tmp_path / ".git").mkdir()
    assert build_info.read_commit(tmp_path, run=_git(stdout, returncode)) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_this_checkouts_commit_is_read_for_real():
    if not (build_info.ROOT / ".git").exists():
        pytest.skip("not a Git checkout")
    commit = build_info.read_commit()
    if commit is None:
        pytest.skip("git refused this folder")
    assert re.fullmatch(r"[0-9a-f]{4,40}", commit)


def test_short_commits_of_different_lengths_are_the_same_commit():
    assert build_info.same_commit("abc1234", "abc1234")
    assert build_info.same_commit("abc1234", "abc1234e9") and build_info.same_commit("abc1234e9", "abc1234")
    assert not build_info.same_commit("abc1234", "def5678")


def test_build_info_is_standard_library_only():
    code = ("import sys, backend.build_info; "
            "print(any(m in sys.modules for m in ('backend.config', 'torch', 'diffusers')))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# --- The running server's own ----------------------------------------------------

@pytest.fixture
def status_server():
    """A loopback server answering `/api/status` with the given body; returns
    its page URL and the paths asked."""
    servers: list[ThreadingHTTPServer] = []

    def start(body: bytes, code: int = 200) -> tuple[str, list[str]]:
        asked: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                asked.append(self.path)
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/", asked

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_the_running_commit_is_the_statuss_own(status_server):
    url, asked = status_server(json.dumps({"state": "unloaded", "commit": "abc1234"}).encode())
    assert server_probe.running_commit(url) == "abc1234"
    assert asked == ["/api/status"]


@pytest.mark.parametrize("body", [
    json.dumps({"state": "unloaded", "commit": None}).encode(),   # that server could not tell
    json.dumps({"state": "unloaded"}).encode(),                   # one from before the field
    json.dumps({"commit": 7}).encode(),
    json.dumps({"commit": ""}).encode(),
    json.dumps(["commit"]).encode(),
    b"<title>Another program</title>",
])
def test_an_answer_without_a_commit_is_none(status_server, body):
    url, _ = status_server(body)
    assert server_probe.running_commit(url) is None


def test_a_refusal_or_no_answer_is_none(status_server):
    url, _ = status_server(b"{}", code=404)
    assert server_probe.running_commit(url) is None
    assert server_probe.running_commit("http://127.0.0.1:9/") is None
