"""bin/auradream-server.ps1, run for real by Windows PowerShell against a stub
ssh on PATH (Windows only; the static checks run everywhere).

The stub is `ssh.cmd` handing its arguments to a Python script, so each
call's argv is logged exactly as ssh.exe would receive it from Windows
PowerShell 5.1. `ssh -G` answers 127.0.0.1 and a port the test chooses: a
loopback listener there is an awake server, a closed port a sleeping one.
No network beyond loopback; nothing is woken by Wake-on-LAN.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bin" / "auradream-server.ps1"
SYSTEM_ROOT = os.environ.get("SystemRoot", r"C:\Windows")
on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell script")

STUB = r'''
import json, os, sys
args = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\n")
if args[:1] == ["-G"]:
    print("user me")
    print("hostname 127.0.0.1")
    print("port " + os.environ["STUB_SSH_PORT"])
    sys.exit(0)
sys.stdout.write(os.environ.get("STUB_SSH_OUTPUT", "remote ran\n"))
sys.exit(int(os.environ.get("STUB_SSH_EXIT", "0")))
'''


# --- Static checks (everywhere) --------------------------------------------------

WINDOWS_SCRIPTS = sorted([*(REPO / "bin").glob("*.ps1"), *(REPO / "bin").glob("*.cmd"),
                          *REPO.glob("*.ps1"), *REPO.glob("*.cmd")])


def test_the_wrapper_pair_is_there():
    names = {p.name for p in WINDOWS_SCRIPTS}
    assert {"auradream-server.ps1", "auradream-server.cmd"} <= names


@pytest.mark.parametrize("script", WINDOWS_SCRIPTS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_windows_scripts_are_ascii_and_crlf(script):
    data = script.read_bytes()
    data.decode("ascii")                               # Windows PowerShell 5.1 reads ANSI
    assert b"\n" not in data.replace(b"\r\n", b"")     # .gitattributes: eol=crlf


def test_no_server_address_is_committed():
    # The repository is public: only the placeholders and the broadcast
    # address, never a real machine's address or MAC.
    for path in (SCRIPT, REPO / "linux" / "control.sh"):
        text = path.read_text(encoding="ascii")
        assert set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)) <= {
            "192.168.1.5", "192.168.1.20", "255.255.255.255", "127.0.0.1"}, path.name
        assert set(re.findall(r"\b[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}\b", text)) <= {"aa:bb:cc:dd:ee:ff"}, path.name


# --- The sandbox ---------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "ssh_stub.py").write_text(STUB, encoding="utf-8")
    (stubs / "ssh.cmd").write_text(
        f'@"{sys.executable}" "%~dp0ssh_stub.py" %*\r\n@exit /b %ERRORLEVEL%\r\n', encoding="ascii")
    log = tmp_path / "ssh.log"
    env = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("AURADREAM_", "PYTHON")) and key.upper() != "PATH"
    }
    env.update({
        "PATH": ";".join([str(stubs), rf"{SYSTEM_ROOT}\System32", SYSTEM_ROOT,
                          rf"{SYSTEM_ROOT}\System32\WindowsPowerShell\v1.0"]),
        "STUB_LOG": str(log),
        "AURADREAM_SERVER": "me@my-box",
        "AURADREAM_INSTALL_TEST_NO_BROWSER": "1",
    })
    return {"env": env, "log": log, "tmp": tmp_path}


@pytest.fixture
def awake(sandbox):
    """A loopback listener standing in for the server's sshd."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    sandbox["env"]["STUB_SSH_PORT"] = str(listener.getsockname()[1])
    yield listener
    listener.close()


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _run(sandbox, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(SCRIPT), *args],
        env=sandbox["env"], capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, stdin=subprocess.DEVNULL,
    )


def _calls(sandbox) -> list[list[str]]:
    log = sandbox["log"]
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return [json.loads(line) for line in lines]


def _remote(sandbox) -> list[str]:
    """The one ssh call that ran a command (not ssh -G)."""
    [call] = [c for c in _calls(sandbox) if c[:1] != ["-G"]]
    return call


BATCH = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
         "-o", "ServerAliveCountMax=6"]


def _control(words: str, directory: str = "~/'AuraDream'", script: str = "control.sh") -> str:
    return (f"cd {directory} || exit 2; [ -f linux/{script} ] || {{ echo 'auradream-server: this checkout "
            f"has no linux/{script}' >&2; exit 127; }}; bash linux/{script} {words}")


# --- Settings ---------------------------------------------------------------------

@on_windows
def test_a_missing_server_is_named(sandbox):
    del sandbox["env"]["AURADREAM_SERVER"]
    result = _run(sandbox, "status")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "AURADREAM_SERVER is not set" in result.stderr and "me@my-box" in result.stderr
    assert _calls(sandbox) == []


@on_windows
@pytest.mark.parametrize("wake", ["my-box", "ftp://192.168.1.5/__wake", "aa:bb:cc:dd:ee", "aa:bb-cc:dd:ee:ff"])
def test_a_bad_wake_setting_is_refused(sandbox, wake):
    sandbox["env"]["AURADREAM_SERVER_WAKE"] = wake
    result = _run(sandbox, "status")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "AURADREAM_SERVER_WAKE is neither an http(s) URL nor a MAC address" in result.stderr
    assert _calls(sandbox) == []


@on_windows
def test_help_and_an_unknown_command(sandbox):
    del sandbox["env"]["AURADREAM_SERVER"]               # help needs no server
    result = _run(sandbox, "help")
    assert result.returncode == 0 and "AURADREAM_SERVER_WAKE" in result.stdout
    result = _run(sandbox, "reboot")
    assert result.returncode == 2 and "Unknown command: reboot" in result.stderr
    assert _calls(sandbox) == []


@on_windows
@pytest.mark.parametrize("args", [("-h",), ("--help",), ("help", "--help")])
def test_dash_h_and_dash_dash_help_are_help(sandbox, args):
    """They name no parameter, so PowerShell hands them over as leftovers."""
    del sandbox["env"]["AURADREAM_SERVER"]
    result = _run(sandbox, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AURADREAM_SERVER_WAKE" in result.stdout
    assert _calls(sandbox) == []


@on_windows
def test_logs_takes_f_and_n_like_control_sh(sandbox, awake):
    """-f is not ambiguous between -Force and -Follow: it is -Follow's alias."""
    result = _run(sandbox, "logs", "-f", "-n", "5")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _remote(sandbox) == ["-t", *BATCH, "me@my-box", _control("'logs' '--follow' '--lines' '5'")]


@on_windows
@pytest.mark.parametrize("args", [("status", "-Force"), ("logs", "-DryRun"), ("stop", "-Lines", "5"),
                                  ("status", "-Command", "ls"), ("status", "extra")])
def test_an_option_for_another_command_is_refused(sandbox, args):
    result = _run(sandbox, *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert _calls(sandbox) == []


# --- The ssh command line -----------------------------------------------------------

@on_windows
@pytest.mark.parametrize(("args", "words"), [
    (("status",), "'status'"),
    (("start",), "'start'"),
    (("stop",), "'stop'"),
    (("stop", "-Force"), "'stop' '--force'"),
    (("restart", "-Force"), "'restart' '--force'"),
    (("logs",), "'logs'"),
    (("logs", "-Lines", "20"), "'logs' '--lines' '20'"),
    (("doctor",), "'doctor'"),
])
def test_each_command_runs_control_sh_in_the_checkout(sandbox, awake, args, words):
    result = _run(sandbox, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "remote ran" in result.stdout
    assert _calls(sandbox)[0] == ["-G", "me@my-box"]
    assert _remote(sandbox) == [*BATCH, "me@my-box", _control(words)]


@on_windows
def test_logs_follow_asks_for_a_terminal(sandbox, awake):
    assert _run(sandbox, "logs", "-Follow", "-Lines", "5").returncode == 0
    assert _remote(sandbox) == ["-t", *BATCH, "me@my-box", _control("'logs' '--follow' '--lines' '5'")]


@on_windows
@pytest.mark.parametrize(("args", "flag"), [(("update",), "'--restart'"),
                                           (("update", "-DryRun"), "'--restart' '--dry-run'")])
def test_update_runs_update_sh(sandbox, awake, args, flag):
    assert _run(sandbox, *args).returncode == 0
    assert _remote(sandbox) == [*BATCH, "me@my-box", _control(flag, script="update.sh")]


@on_windows
@pytest.mark.parametrize(("directory", "cd"), [
    ("/srv/Aura Dream", "'/srv/Aura Dream'"),
    ("~/apps/it's here", "~/'apps/it'\\''s here'"),
    ("~", "~"),
])
def test_the_checkout_folder_is_quoted_for_bash(sandbox, awake, directory, cd):
    sandbox["env"]["AURADREAM_SERVER_DIR"] = directory
    assert _run(sandbox, "status").returncode == 0
    assert _remote(sandbox)[-1] == _control("'status'", directory=cd)


@on_windows
def test_connect_opens_a_shell_or_runs_one_command(sandbox, awake):
    assert _run(sandbox, "connect").returncode == 0
    call = _remote(sandbox)
    assert call[0] == "-t" and "BatchMode=yes" not in call      # a password prompt may be wanted
    assert call[-2:] == ["me@my-box", "cd ~/'AuraDream' && exec bash -l"]
    sandbox["log"].unlink()
    assert _run(sandbox, "connect", "-Command", "echo 'ok' && uptime").returncode == 0
    assert _remote(sandbox)[-1] == "cd ~/'AuraDream' && echo 'ok' && uptime"


@on_windows
@pytest.mark.parametrize("args", [("connect", "-Command", 'echo "hi"'), ("connect", "-Command", "ls \\")])
def test_words_ssh_would_mangle_are_refused(sandbox, awake, args):
    result = _run(sandbox, *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Windows PowerShell" in result.stderr
    assert [c for c in _calls(sandbox) if c[:1] != ["-G"]] == []


@on_windows
def test_a_double_quote_in_the_folder_is_refused(sandbox, awake):
    sandbox["env"]["AURADREAM_SERVER_DIR"] = '~/Aura"Dream'
    result = _run(sandbox, "status")
    assert result.returncode == 2 and "double quotes" in result.stderr
    assert _calls(sandbox) == []


# --- Exit codes and hints ---------------------------------------------------------------

@on_windows
def test_the_remote_exit_code_is_the_scripts(sandbox, awake):
    sandbox["env"]["STUB_SSH_EXIT"] = "1"
    sandbox["env"]["STUB_SSH_OUTPUT"] = "  [stop] The AuraDream service is busy: 1 job(s) queued or running.\n"
    result = _run(sandbox, "stop")
    assert result.returncode == 1
    assert "The AuraDream service is busy" in result.stdout


@on_windows
def test_ssh_failing_gives_a_hint(sandbox, awake):
    sandbox["env"]["STUB_SSH_EXIT"] = "255"
    result = _run(sandbox, "status")
    assert result.returncode == 255
    assert "ssh could not reach me@my-box (exit 255)" in result.stderr and "authorized_keys" in result.stderr
    assert "BatchMode" in result.stderr


@on_windows
def test_connect_failing_does_not_blame_batch_mode(sandbox, awake):
    sandbox["env"]["STUB_SSH_EXIT"] = "255"
    result = _run(sandbox, "connect")
    assert result.returncode == 255
    assert "ssh could not reach me@my-box (exit 255)" in result.stderr and "authorized_keys" in result.stderr
    assert "BatchMode" not in result.stderr and "password" in result.stderr


@on_windows
def test_an_older_checkout_is_told_update_brings_control_sh(sandbox, awake):
    sandbox["env"]["STUB_SSH_EXIT"] = "127"
    result = _run(sandbox, "status")
    assert result.returncode == 127
    assert "has no linux/control.sh" in result.stderr and "auradream-server update brings it" in result.stderr


# --- Waking ---------------------------------------------------------------------------

@on_windows
def test_a_sleeping_server_without_a_wake_setting_is_tried_anyway(sandbox):
    sandbox["env"]["STUB_SSH_PORT"] = str(_closed_port())
    result = _run(sandbox, "status")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "is not answering" in result.stdout and "AURADREAM_SERVER_WAKE is not set" in result.stdout
    assert _remote(sandbox)[-1] == _control("'status'")


@on_windows
def test_a_wake_url_is_posted_and_the_server_waited_for(sandbox):
    # The wake address "starts" the server: after the POST, its sshd listens.
    port = _closed_port()
    sandbox["env"]["STUB_SSH_PORT"] = str(port)
    posts = []
    sshd = socket.socket()

    class Gate(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's name
            posts.append(self.path)
            if len(posts) == 1:
                sshd.bind(("127.0.0.1", port))
                sshd.listen(4)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    gate = ThreadingHTTPServer(("127.0.0.1", 0), Gate)
    threading.Thread(target=gate.serve_forever, daemon=True).start()
    sandbox["env"]["AURADREAM_SERVER_WAKE"] = f"http://127.0.0.1:{gate.server_address[1]}/__wake"
    try:
        result = _run(sandbox, "status")
    finally:
        gate.shutdown()
        gate.server_close()
        sshd.close()
    assert result.returncode == 0, result.stdout + result.stderr
    assert posts == ["/__wake"]
    assert "waking it" in result.stdout and "The server is awake." in result.stdout
    assert _remote(sandbox)[-1] == _control("'status'")


@on_windows
def test_open_uses_the_host_and_port(sandbox, awake):
    sandbox["env"]["AURADREAM_SERVER_PORT"] = "7999"
    result = _run(sandbox, "open")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Opening http://127.0.0.1:7999/" in result.stdout      # ssh -G's host name
    assert [c for c in _calls(sandbox) if c[:1] != ["-G"]] == []


@on_windows
@pytest.mark.parametrize("mac", ["aa:bb:cc:dd:ee:ff", "AA-BB-CC-DD-EE-FF"])
def test_the_magic_packet(mac):
    # Dot-sourced, the script defines its functions and runs nothing.
    command = (f". '{SCRIPT}'; $p = New-MagicPacket '{mac}'; "
               "Write-Output $p.Length; Write-Output ([BitConverter]::ToString($p))")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-Command", command], capture_output=True, text=True, timeout=60,
                            stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stdout + result.stderr
    length, hex_bytes = result.stdout.split()
    assert length == "102"
    assert hex_bytes == "-".join(["FF"] * 6 + ["AA", "BB", "CC", "DD", "EE", "FF"] * 16)
