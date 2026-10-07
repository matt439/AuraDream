"""Real child process trees, listener ownership, and loopback proxy bypass."""
from __future__ import annotations

from pathlib import Path
import http.server
import importlib.util
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

from backend import critic, critic_runtime, critic_server

from tests.critic_helpers import (
    _answer,
    _configure,
    _critique,
)


# A wrapper's server: a child that starts a grandchild, which listens on a
# port it prints and, like a server freeing the card, takes a moment to go
# when terminated (a job object's kill cannot be delayed). On Windows the
# venv's python.exe is itself a launcher, so the tree is four processes deep.
_GRANDCHILD = (
    "import signal, socket, sys, time\n"
    "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.5), sys.exit(0)))\n"
    "listener = socket.socket()\n"
    "listener.bind(('127.0.0.1', 0))\n"
    "listener.listen()\n"
    "print(listener.getsockname()[1], flush=True)\n"
    "time.sleep(120)\n"
)


# On POSIX the grandchild takes a process group of its own, as a shell with
# job control (`set -m`) puts its jobs: still in the child's session.
_WRAPPER = (
    "import os, subprocess, sys, time\n"
    "group = {'process_group': 0} if os.name == 'posix' else {}\n"
    f"subprocess.Popen([sys.executable, '-c', {_GRANDCHILD!r}], **group)\n"
    "time.sleep(120)\n"
)


def _listening(port: int) -> bool:
    try:
        # A short timeout: Windows retries a refused loopback connect for 2 s.
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture
def wrapped_listener():
    process = critic_server._ServerProcess(
        [sys.executable, "-c", _WRAPPER], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE
    )
    try:
        yield process, int(process.stdout.readline())
    finally:
        critic_server._stop(process)
        process.stdout.close()


def test_stop_ends_the_whole_tree(wrapped_listener):
    process, port = wrapped_listener
    assert _listening(port)
    started = time.monotonic()
    critic_server._stop(process)
    # Terminated with the rest, not left for the kill `STOP_TIMEOUT` later.
    assert time.monotonic() - started < critic_server.STOP_TIMEOUT / 2
    assert process.poll() is not None
    assert not _listening(port)


@pytest.mark.skipif(sys.platform != "win32", reason="a job object")
def test_the_child_joins_its_job_before_it_runs(monkeypatch):
    # However long the job takes, the child has started nothing outside it
    # meanwhile: it is created suspended and resumed once inside.
    new_job = critic_server._new_job

    def slow(process):
        time.sleep(2)
        return new_job(process)

    monkeypatch.setattr(critic_server, "_new_job", slow)
    process = critic_server._ServerProcess(
        [sys.executable, "-c", _WRAPPER], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE
    )
    try:
        port = int(process.stdout.readline())
    finally:
        critic_server._stop(process)
        process.stdout.close()
    assert not _listening(port)


@pytest.mark.skipif(sys.platform != "win32", reason="a job object's process list")
def test_the_tree_is_waited_for_until_its_handles_are_closed(wrapped_listener, monkeypatch):
    process, port = wrapped_listener
    # Each process is held once seen in the job; then the job's list is
    # empty while the rest of the tree still runs - as it is, for a moment,
    # after each exits and before its handles (the log, the listening
    # socket) are closed.
    assert len(process._hold()) > 1
    monkeypatch.setattr(critic_server, "_job_pids", lambda job: set())
    subprocess.Popen.kill(process)  # the child alone
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=1)
    assert _listening(port)


# A process whose first thread exits alone (the exit syscall, not exit_group)
# while another keeps its listening socket for a while: /proc shows it a
# zombie, as it shows a multithreaded server between its threads' exits.
_HALF_EXITED = (
    "import ctypes, socket, sys, threading, time\n"
    "listener = socket.socket()\n"
    "listener.bind(('127.0.0.1', 0))\n"
    "listener.listen()\n"
    "print(listener.getsockname()[1], flush=True)\n"
    "threading.Thread(target=time.sleep, args=(3,)).start()\n"
    "ctypes.CDLL(None).syscall(int(sys.argv[1]))\n"
)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "aarch64"),
    reason="Linux's /proc, and the exit syscall's number",
)
def test_a_process_is_waited_for_until_its_last_thread_exits():
    exit_call = {"x86_64": 60, "aarch64": 93}[platform.machine()]
    process = critic_server._ServerProcess(
        [sys.executable, "-c", _HALF_EXITED, str(exit_call)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
    )
    try:
        port = int(process.stdout.readline())
        deadline = time.monotonic() + 2
        while Path(f"/proc/{process.pid}/stat").read_text().rpartition(")")[2].split()[0] != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert _listening(port)
        assert critic_server._session_pids(process.pid) == {process.pid}
    finally:
        critic_server._stop(process)
        process.stdout.close()
    assert not _listening(port) and critic_server._session_pids(process.pid) == set()


def test_a_tree_is_waited_for_after_its_child(monkeypatch):
    process = critic_server._ServerProcess([sys.executable, "-c", "pass"])
    # The child gone, the rest of the tree for two more looks.
    alive = iter([True, True, False])
    monkeypatch.setattr(process, "_others_alive", lambda: next(alive))
    assert process.wait(timeout=10) == 0
    assert next(alive, None) is None
    # And never past the timeout.
    monkeypatch.setattr(process, "_others_alive", lambda: True)
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=0.2)


@pytest.mark.skipif(
    not (sys.platform == "win32" or sys.platform.startswith("linux")),
    reason="the listener's owner is read from Windows' TCP table or Linux's /proc",
)
def test_the_listener_check_tells_the_tree_from_any_other_process(wrapped_listener):
    process, port = wrapped_listener
    pids = process.pids()
    assert process.pid in pids and len(pids) > 1
    assert critic_server._listener_is_ours(port, pids) is True
    assert critic_server._listener_is_ours(port, {os.getpid()}) is False
    with socket.socket() as own:
        own.bind(("127.0.0.1", 0))
        own.listen()
        port = own.getsockname()[1]
        assert critic_server._listener_is_ours(port, pids) is False
        assert critic_server._listener_is_ours(port, {os.getpid()}) is True
    # Nothing to be seen listening is no proof of anything.
    assert critic_server._listener_is_ours(critic_server._free_port(), pids) is False


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux: on Windows a listener on every address may raise a firewall prompt, and a "
    "mapped one is in the IPv4 table",
)
@pytest.mark.parametrize(
    ("family", "address"), [(socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::ffff:127.0.0.1")]
)
def test_a_listener_beside_loopback_is_seen(wrapped_listener, family, address):
    # Where else a connection to 127.0.0.1 lands: a listener on every
    # address, or an IPv6 one with IPv4 mapped in (the IPv6 table).
    process, _ = wrapped_listener
    with socket.socket(family) as other:
        if family == socket.AF_INET6:
            other.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        other.bind((address, 0))
        other.listen()
        port = other.getsockname()[1]
        assert critic_server._listener_is_ours(port, {os.getpid()}) is True
        assert critic_server._listener_is_ours(port, process.pids()) is False


# A llama-server stand-in: its port from the command line, its key from the
# environment, /v1/models, the counts and the chat behind the key,
# `answer.json` beside it. `FAKE_LLAMA_SERVER_WARMUP` seconds of 503 first.
_FAKE_LLAMA_SERVER = """\
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A run that fails to stop it must not leave it listening for good.
threading.Timer(60, os._exit, (0,)).start()
ready = time.monotonic() + float(os.environ.get("FAKE_LLAMA_SERVER_WARMUP", "0"))
port = int(sys.argv[sys.argv.index("--port") + 1])
key = os.environ["LLAMA_API_KEY"]
with open(os.path.join(os.path.dirname(__file__), "answer.json"), encoding="utf-8") as source:
    answer = source.read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def refused(self):
        if self.headers.get("Authorization") == "Bearer " + key:
            return False
        self.reply(401, {"error": {"message": "Invalid API Key"}})
        return True

    def do_GET(self):
        if self.path == "/health":
            self.reply(200 if time.monotonic() >= ready else 503, {})
        elif not self.refused():
            self.reply(200, {"data": []})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.refused():
            return
        if self.path == "/apply-template":
            text = "".join(message["content"] + " " for message in body["messages"])
            self.reply(200, {"prompt": text})
        elif self.path == "/tokenize":
            self.reply(200, {"tokens": list(range(len(body["content"].split())))})
        else:
            message = {"content": answer}
            self.reply(200, {"choices": [{"message": message, "finish_reason": "stop"}]})


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


# A launcher that starts the stand-in and exits once it listens, leaving it
# running - as a wrapper that backgrounds its server does.
_LAUNCHER = """\
import os, socket, subprocess, sys, time

port = int(sys.argv[sys.argv.index("--port") + 1])
server = os.path.join(os.path.dirname(__file__), "fake_llama_server.py")
subprocess.Popen([sys.executable, server, *sys.argv[1:]])
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        break
    except OSError:
        time.sleep(0.05)
"""


def _wrapped(monkeypatch, tmp_path, script: str):
    """The critic set up to run `script` through a wrapper, which the setting
    names: a .bat on Windows, a shell script without `exec` elsewhere. The
    ports it is given and its logs, recorded."""
    (tmp_path / "fake_llama_server.py").write_text(_FAKE_LLAMA_SERVER, encoding="utf-8")
    (tmp_path / "launcher.py").write_text(_LAUNCHER, encoding="utf-8")
    (tmp_path / "answer.json").write_text(json.dumps(_answer()), encoding="utf-8")
    if sys.platform == "win32":
        wrapper = tmp_path / "llama-server.bat"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = tmp_path / "llama-server"
        wrapper.write_text(
            f'#!/bin/sh\n"{sys.executable}" "$(dirname "$0")/{script}" "$@"\n', encoding="utf-8"
        )
        wrapper.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"m")
    _configure(monkeypatch, critic=True, critic_server=wrapper, critic_model=None)
    target = critic.PromptCritic()
    target._prepared = (wrapper, model)
    ports, logs = [], []
    free_port, mkstemp = critic_server._free_port, critic.tempfile.mkstemp
    monkeypatch.setattr(critic_server, "_free_port", lambda: ports.append(free_port()) or ports[-1])

    def record_log(*args, **kwargs):
        handle, name = mkstemp(*args, **kwargs)
        logs.append(name)
        return handle, name

    monkeypatch.setattr(critic.tempfile, "mkstemp", record_log)
    return target, ports, logs


def test_a_wrapped_server_is_asked_and_stopped_whole(monkeypatch, tmp_path, caplog):
    target, ports, logs = _wrapped(monkeypatch, tmp_path, "fake_llama_server.py")
    with caplog.at_level(logging.WARNING, logger=critic.log.name):
        result = _critique(target)
    assert result.issues[0].cite_id == "G3" and target._process is None
    # Counted by the server, over HTTP, before it was asked.
    assert "sized by the estimate" not in caplog.text
    # The server the wrapper started went with it, and its log with them:
    # nothing still holds the file once the stop returns.
    assert not _listening(ports[0])
    assert not Path(logs[0]).exists() and "could not remove" not in caplog.text


def test_a_server_its_wrapper_left_running_is_stopped(monkeypatch, tmp_path):
    # The wrapper exits once its server listens, before the server is ready:
    # the critic sees its child gone and gives up, and the stop must still
    # reach the server, since the child's exit is not the tree's.
    monkeypatch.setenv("FAKE_LLAMA_SERVER_WARMUP", "3")
    target, ports, _ = _wrapped(monkeypatch, tmp_path, "launcher.py")
    # Held, so that on Windows the job's closing cannot stand in for the stop.
    spawned = []
    spawn = critic_server._spawn
    monkeypatch.setattr(critic_server, "_spawn", lambda *args: spawned.append(spawn(*args)) or spawned[-1])
    with pytest.raises(critic_runtime.CriticError, match="exited with code 0 before it was ready"):
        _critique(target)
    assert spawned and not _listening(ports[0])


class _Recorder(http.server.BaseHTTPRequestHandler):
    """Records each request - method, path as sent (a proxy is sent the whole
    URL), Authorization - and answers `status`."""

    status = 200
    seen: list[tuple[str, str, str | None]] = []

    def log_message(self, *args):
        pass

    def _record(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.seen.append((self.command, self.path, self.headers.get("Authorization")))
        data = json.dumps({"path": self.path}).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _record


@pytest.fixture
def recording_server():
    servers = []

    def start(status: int):
        handler = type("Handler", (_Recorder,), {"status": status, "seen": []})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        servers.append(server)
        return server, handler.seen

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_loopback_requests_never_go_through_an_environment_proxy(monkeypatch, recording_server):
    target, at_target = recording_server(200)
    proxy, at_proxy = recording_server(502)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_port}")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    # The module as an app started under that proxy imports it, and urlopen's
    # shared opener rebuilt from the same environment.
    spec = importlib.util.spec_from_file_location("backend._critic_under_proxy", critic_server.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fresh)
    spec.loader.exec_module(fresh)
    monkeypatch.setattr(urllib.request, "_opener", None)

    base = f"http://127.0.0.1:{target.server_port}"
    assert fresh._request("GET", base + "/health", None, None, 5.0) == (200, {"path": "/health"})
    status, _ = fresh._request("POST", base + "/v1/chat/completions", "k3y", {"q": 1}, 5.0)
    assert status == 200
    assert at_target == [
        ("GET", "/health", None), ("POST", "/v1/chat/completions", "Bearer k3y"),
    ]
    assert at_proxy == []
