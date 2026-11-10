"""linux/setup.sh, install-service.sh and update.sh, run for real by bash
against stub nvidia-smi/uv/git/systemctl/loginctl/hostname/id scripts in a
temporary folder (Linux only; the static checks run everywhere).

`AURADREAM_INSTALL_TEST_ROOT` points the scripts at a temporary project
folder and `AURADREAM_INSTALL_TEST_PROC_VERSION` at a stand-in /proc/version,
as linux/common.sh documents. PATH holds the stubs, then /usr/bin and /bin:
every stub shadows the machine's own tool, so no real systemctl --user, git
or uv ever runs. HOME is temporary and XDG_CONFIG_HOME unset unless a test
sets it, so the unit and environment file land there. No network.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LINUX = REPO / "linux"
SCRIPTS = sorted(LINUX.glob("*.sh"))
BASH = shutil.which("bash")
on_linux = pytest.mark.skipif(sys.platform != "linux" or BASH is None, reason="bash scripts for native Linux")


# --- Static checks (everywhere) --------------------------------------------------

def test_the_script_set_is_there():
    assert {p.name for p in SCRIPTS} >= {"common.sh", "setup.sh", "install-service.sh", "update.sh", "control.sh"}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_are_ascii_lf_and_strict(script):
    data = script.read_bytes()
    assert b"\r" not in data                      # .gitattributes: *.sh eol=lf
    data.decode("ascii")
    if script.name != "common.sh":                # sourced: the caller sets the options
        assert data.startswith(b"#!/usr/bin/env bash\n")
        assert b"\nset -euo pipefail\n" in data


def test_sudo_is_only_ever_printed():
    # Every line naming sudo is a message (say/note/stop/what_to_do/...), a
    # comment or inside a here-document of text; none runs it.
    for script in SCRIPTS:
        for number, line in enumerate(script.read_text(encoding="ascii").splitlines(), 1):
            code = line.strip()
            if "sudo" not in code or code.startswith("#"):
                continue
            assert ("'" in code or '"' in code) and not code.startswith("sudo"), f"{script.name}:{number}: {line}"


def test_gitattributes_keeps_the_scripts_lf():
    rules = (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "*.sh text eol=lf" in rules


@on_linux
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_parses_every_script(script):
    result = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- The sandbox ---------------------------------------------------------------

def _stub(stubs: Path, name: str, body: str, log: bool = True) -> None:
    path = stubs / name
    lines = ["#!/bin/bash"]
    if log:
        lines.append(f'printf "%s\\n" "{name} $*" >>"$STUB_LOG"')
    lines.append(body)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    path.chmod(0o755)


def _install_stubs(stubs: Path) -> None:
    _stub(stubs, "nvidia-smi", (
        'case "$1" in --query-gpu=*) printf "%s\\n" "$STUB_SMI_QUERY"; exit 0 ;; esac\n'
        'printf "%s\\n" "$STUB_SMI_HEADER"'
    ))
    _stub(stubs, "uv", 'exit "${STUB_UV_EXIT:-0}"')
    _stub(stubs, "git", (
        # STUB_PULL_WAIT: a file the pull waits for, so a test can act mid-update.
        'if [ "$3" = pull ]; then\n'
        '  if [ -n "${STUB_PULL_WAIT:-}" ]; then\n'
        '    for _ in $(seq 600); do [ -e "$STUB_PULL_WAIT" ] && break; sleep 0.05; done\n'
        '  fi\n'
        '  printf "%s\\n" "${STUB_PULL_OUTPUT:-Already up to date.}"; exit "${STUB_PULL_EXIT:-0}"\n'
        'fi\n'
        # What control.sh status reads of a checkout.
        'case "$3" in\n'
        # The commit update.sh names the database copy by.
        '  rev-parse) [ "$4" = --short ] && { printf "%s\\n" "${STUB_COMMIT-abc1234}"; exit 0; }\n'
        '    [ "$5" = HEAD ] && printf "main\\n"; [ "$5" = --symbolic-full-name ] && printf "origin/main\\n"; exit 0 ;;\n'
        '  log) printf "%s\\n" "abc1234 Keep awake (2 days ago)"; exit 0 ;;\n'
        '  fetch) [ -z "${STUB_FETCH_EXIT:-}" ] || { echo "fatal: unable to access the remote" >&2; exit "$STUB_FETCH_EXIT"; }; exit 0 ;;\n'
        '  rev-list) printf "%s\\n" "${STUB_AHEAD_BEHIND:-0 0}"; exit 0 ;;\n'
        'esac\n'
        'exit 0'
    ))
    # The service's state is a file: is-active answers from it, stop removes
    # it, start (or restart) makes it, STUB_START_DELAY seconds later when
    # set; a "failed" file beside it is a failed unit, an "auto-restart" file
    # one systemd is restarting. STUB_CRASH_AFTER seconds after a start,
    # NRestarts (the "restarts" file, else 0) grows by one.
    _stub(stubs, "systemctl", (
        'state="$STUB_STATE/active"\n'
        'begin() {\n'
        '  if [ -n "${STUB_START_DELAY:-}" ]; then (sleep "$STUB_START_DELAY"; : >"$state") >/dev/null 2>&1 &\n'
        '  else : >"$state"; fi\n'
        '  if [ -n "${STUB_CRASH_AFTER:-}" ]; then\n'
        '    (sleep "$STUB_CRASH_AFTER"; echo 1 >"$STUB_STATE/restarts") >/dev/null 2>&1 &\n'
        '  fi\n'
        '}\n'
        'case "$2" in\n'
        '  is-active) [ -e "$state" ]; exit ;;\n'
        '  is-failed) [ -e "$STUB_STATE/failed" ]; exit ;;\n'
        '  stop) rm -f "$state" ;;\n'
        '  start) begin ;;\n'
        '  restart) rm -f "$state"; begin ;;\n'
        '  show)\n'
        '    case " $* " in *" ControlGroup "*)\n'
        '      [ -e "$STUB_STATE/cgroup" ] && cat "$STUB_STATE/cgroup"; exit 0 ;; esac\n'
        '    if [ -e "$STUB_STATE/failed" ]; then a=failed s=failed\n'
        '    elif [ -e "$STUB_STATE/auto-restart" ]; then a=activating s=auto-restart\n'
        '    elif [ -e "$state" ]; then a=active s=running\n'
        '    else a=inactive s=dead; fi\n'
        '    n=0; [ -e "$STUB_STATE/restarts" ] && n=$(cat "$STUB_STATE/restarts")\n'
        '    printf "LoadState=loaded\\nActiveState=%s\\nSubState=%s\\nNRestarts=%s\\n" "$a" "$s" "$n"\n'
        '    printf "StateChangeTimestamp=Tue 2026-11-03 10:00:00 AEDT\\nUnitFileState=enabled\\n" ;;\n'
        'esac\n'
        'exit 0'
    ))
    _stub(stubs, "journalctl", 'printf "%s\\n" "Nov 03 10:00:01 my-box uv[1234]: INFO: Uvicorn running"')
    _stub(stubs, "systemd-inhibit", (
        'printf "%s\\n" "WHO       UID  USER   PID  COMM            WHAT       WHY                 MODE"\n'
        'printf "%s\\n" "AuraDream 1000 tester 4321 systemd-inhibit sleep:idle AuraDream is in use block"\n'
        'printf "%s\\n" "UPower    0    root   99   upowerd         sleep      Pause device polling delay"'
    ), log=False)
    _stub(stubs, "loginctl", 'printf "Linger=%s\\n" "${STUB_LINGER:-yes}"')
    _stub(stubs, "hostname", (
        'if [ "$1" = -I ]; then printf "%s\\n" "192.168.1.20 10.0.0.5 fe80::1234:5678 "; '
        'else printf "%s\\n" "My-Box"; fi'
    ), log=False)
    _stub(stubs, "id", (
        'case "$1" in -u) printf "%s\\n" "${STUB_UID:-1000}" ;; -un) printf "tester\\n" ;; '
        '*) exec /usr/bin/id "$@" ;; esac'
    ), log=False)


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "AuraDream"
    (root / "backend").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "auradream"\n')
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _install_stubs(stubs)
    home = tmp_path / "home"
    home.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    proc_version = tmp_path / "proc-version"
    proc_version.write_text("Linux version 6.17.0-5-generic (buildd@ubuntu) #5-Ubuntu SMP\n")
    log = tmp_path / "calls.log"
    env = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("AURADREAM_", "HF_", "UV_", "XDG_", "DBUS_", "VIRTUAL_ENV"))
        and key not in ("PATH", "HOME", "USER")
    }
    env.update({
        "PATH": f"{stubs}:/usr/bin:/bin",
        "HOME": str(home),
        "USER": "tester",
        "AURADREAM_INSTALL_TEST_ROOT": str(root),
        "AURADREAM_INSTALL_TEST_PROC_VERSION": str(proc_version),
        "STUB_LOG": str(log),
        "STUB_STATE": str(state),
        "STUB_SMI_HEADER": "| NVIDIA-SMI 595.99.02   Driver Version: 595.99.02   CUDA Version: 13.2 |",
        "STUB_SMI_QUERY": "NVIDIA GeForce RTX 3070, 8.6, 8192 MiB",
        # Nothing listens on port 1: a server running on this machine's 7860
        # never answers for the sandbox. Tests that want one set their own.
        "AURADREAM_PORT": "1",
    })
    return {"root": root, "stubs": stubs, "home": home, "state": state, "log": log,
            "env": env, "tmp": tmp_path, "proc_version": proc_version}


def _run(sandbox, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(LINUX / script), *args], env=sandbox["env"], capture_output=True, text=True,
        timeout=120, stdin=subprocess.DEVNULL, cwd=sandbox["tmp"],
    )


def _calls(sandbox) -> list[str]:
    log = sandbox["log"]
    return log.read_text().splitlines() if log.exists() else []


def _snapshot(*folders: Path) -> dict[str, str]:
    found = {}
    for folder in folders:
        for path in sorted(folder.rglob("*")):
            found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
    return found


def _venv(sandbox) -> Path:
    """A .venv whose python is this interpreter, beside the real server_probe."""
    bin_dir = sandbox["root"] / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    shutil.copy(REPO / "backend" / "server_probe.py", sandbox["root"] / "backend" / "server_probe.py")
    return bin_dir / "python"


def _service_active(sandbox) -> None:
    (sandbox["state"] / "active").touch()


def _systemctl(sandbox) -> list[str]:
    return [c for c in _calls(sandbox) if c.startswith("systemctl")]


# --- setup.sh -------------------------------------------------------------------

@on_linux
def test_setup_dry_run_on_an_8gb_ampere_card(sandbox):
    before = _snapshot(sandbox["root"], sandbox["home"])
    result = _run(sandbox, "setup.sh", "--dry-run")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "The driver supports CUDA 13.2" in out
    assert "Card: NVIDIA GeForce RTX 3070" in out
    assert "Compute capability 8.6 is supported" in out
    assert "int4" in out and "nvfp4" in out
    assert "8.0 GB of video memory" in out and "on an 8 GB RTX 3070 Animagine XL 4.0 at fp8 with model offload fits" in out
    assert "would run in" in out and "uv sync" in out and "never model weights" in out
    assert "backend.doctor" in out
    assert str(sandbox["home"] / ".cache" / "huggingface") in out and "HF_HOME" in out
    assert "[stop]" not in out and "Dry run finished. Nothing was changed." in out
    assert _snapshot(sandbox["root"], sandbox["home"]) == before
    assert [c for c in _calls(sandbox) if not c.startswith(("nvidia-smi", "systemctl --user is-active"))] == []


@on_linux
def test_setup_dry_run_refuses_an_old_card_and_driver_and_shows_the_rest(sandbox):
    sandbox["env"]["STUB_SMI_HEADER"] = "NVIDIA-SMI 551.86   Driver Version: 551.86   CUDA Version: 12.4"
    sandbox["env"]["STUB_SMI_QUERY"] = "NVIDIA GeForce GTX 1080, 6.1, 8192 MiB"
    (sandbox["stubs"] / "uv").unlink()                # and no uv: the pinned installer
    result = _run(sandbox, "setup.sh", "--dry-run")
    out = result.stdout
    assert result.returncode == 1, out + result.stderr
    assert "The driver supports CUDA 12.4" in out and "sudo apt install nvidia-driver-595-open" in out
    assert "Do not install the CUDA Toolkit" in out
    assert "compute capability 6.1" in out and "not supported" in out
    assert "int4" not in out                          # no Nunchaku note for a refused card
    assert "would install uv with uv's own installer, version 0.12.7" in out
    assert "https://astral.sh/uv/0.12.7/install.sh" in out
    assert "A real run would stop at:" in out
    assert not (sandbox["home"] / ".local").exists()
    assert [c for c in _calls(sandbox) if not c.startswith(("nvidia-smi", "systemctl --user is-active"))] == []


@on_linux
def test_setup_refuses_under_wsl(sandbox):
    sandbox["proc_version"].write_text("Linux version 6.6.87.2-microsoft-standard-WSL2 (root@host)\n")
    result = _run(sandbox, "setup.sh")                # a real run, not a dry one
    assert result.returncode == 1, result.stdout + result.stderr
    assert "This is WSL" in result.stdout and "bash wsl/setup.sh" in result.stdout
    assert _calls(sandbox) == []


@on_linux
def test_setup_runs_the_install_steps_in_order(sandbox):
    result = _run(sandbox, "setup.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    uv = [c for c in _calls(sandbox) if c.startswith("uv")]
    assert uv == ["uv python install 3.13", "uv sync", "uv run python -m backend.doctor"]
    assert "bash linux/install-service.sh" in result.stdout and "uv run hf auth login" in result.stdout
    assert "uv run python -m backend.main" in result.stdout
    assert not (sandbox["root"] / "auradream.settings.json").exists()


@on_linux
def test_setup_refuses_while_the_service_runs(sandbox):
    _service_active(sandbox)
    result = _run(sandbox, "setup.sh")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "systemctl --user stop auradream" in result.stdout
    assert [c for c in _calls(sandbox) if c.startswith("uv")] == []


@on_linux
def test_setup_refuses_while_a_venv_program_runs(sandbox):
    python = _venv(sandbox)
    child = subprocess.Popen([str(python), "-c", "import time; time.sleep(60)"])
    try:
        result = _run(sandbox, "setup.sh")
    finally:
        child.kill()
        child.wait()
    assert result.returncode == 1, result.stdout + result.stderr
    assert "A program from this folder's .venv is still running" in result.stdout
    assert f"(process {child.pid})" in result.stdout
    assert "AuraDream is not running" not in result.stdout
    assert [c for c in _calls(sandbox) if c.startswith("uv")] == []
    assert "/proc/" not in result.stderr              # unreadable processes are skipped quietly


@on_linux
def test_setup_probes_the_port_in_the_service_env_file(sandbox, auradream_listening):
    _venv(sandbox)
    sandbox["env"]["AURADREAM_PORT"] = "1"            # this shell's; the service's differs
    env_file = _env_file(sandbox)
    env_file.parent.mkdir(parents=True)
    env_file.write_text(f"AURADREAM_PORT={auradream_listening['port']}\n")
    result = _run(sandbox, "setup.sh")
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"AuraDream is running at http://127.0.0.1:{auradream_listening['port']}/" in result.stdout
    assert [c for c in _calls(sandbox) if c.startswith("uv")] == []


# --- install-service.sh -----------------------------------------------------------

def _unit(sandbox, config: Path | None = None) -> Path:
    return (config or sandbox["home"] / ".config") / "systemd" / "user" / "auradream.service"


def _env_file(sandbox, config: Path | None = None) -> Path:
    return (config or sandbox["home"] / ".config") / "auradream" / "auradream.env"


def _allowed_hosts(text: str) -> str:
    [line] = [line for line in text.splitlines() if line.startswith("AURADREAM_ALLOWED_HOSTS=")]
    return line.split("=", 1)[1]


@on_linux
def test_install_service_dry_run_prints_and_writes_nothing(sandbox):
    (sandbox["root"] / ".venv").mkdir()
    before = _snapshot(sandbox["root"], sandbox["home"])
    result = _run(sandbox, "install-service.sh", "--dry-run")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "[Unit]" in out and "[Install]" in out and "WantedBy=default.target" in out
    assert f'ExecStart="{sandbox["stubs"]}/uv" run --no-sync python -m backend.main' in out
    assert "AURADREAM_HOST=0.0.0.0" in out
    assert "would run: systemctl --user daemon-reload" in out
    assert "Dry run finished. Nothing was changed." in out
    assert _snapshot(sandbox["root"], sandbox["home"]) == before
    assert _systemctl(sandbox) == []


@on_linux
def test_install_service_writes_the_unit_and_env_and_enables_it(sandbox):
    (sandbox["root"] / ".venv").mkdir()
    sandbox["env"]["STUB_LINGER"] = "no"
    result = _run(sandbox, "install-service.sh")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    unit = _unit(sandbox).read_text()
    env_file = _env_file(sandbox)
    assert f"WorkingDirectory={sandbox['root']}\n" in unit
    assert f"EnvironmentFile={env_file}\n" in unit
    assert f'ExecStart="{sandbox["stubs"]}/uv" run --no-sync python -m backend.main\n' in unit
    assert Path(str(sandbox["stubs"] / "uv")).is_absolute()
    for line in ("Environment=PYTHONUTF8=1", "Restart=on-failure", "RestartSec=10",
                 "StartLimitIntervalSec=900", "StartLimitBurst=3", "TimeoutStopSec=120",
                 "UMask=0077", "WantedBy=default.target"):
        assert line + "\n" in unit
    settings = [line for line in unit.splitlines() if not line.startswith("#")]
    assert not [line for line in settings if "--open" in line or "systemd-inhibit" in line]
    assert not [line for line in settings if line.startswith("KillMode=")]   # the default, control-group
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    env_text = env_file.read_text()
    assert "AURADREAM_HOST=0.0.0.0\n" in env_text
    hosts = _allowed_hosts(env_text)
    assert hosts == "127.0.0.1,localhost,my-box,my-box.local,192.168.1.20,10.0.0.5"
    assert "*" not in hosts and "fe80" not in hosts
    assert "HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false" in env_text
    assert "HF_XET_FIXED_DOWNLOAD_CONCURRENCY=32" in env_text
    for example in ("#HF_HOME=", "#AURADREAM_PORT=", "#AURADREAM_KEEP_AWAKE=true",
                    "#AURADREAM_KEEP_AWAKE_IDLE_MINUTES=20"):
        assert example in env_text
    env_text.encode("ascii")
    assert _systemctl(sandbox) == [
        "systemctl --user is-active --quiet auradream.service",
        "systemctl --user daemon-reload",
        "systemctl --user enable --now auradream.service",
    ]
    assert "sudo loginctl enable-linger tester" in out
    assert "http://my-box:7860/" in out and "http://192.168.1.20:7860/" in out
    assert "http://127.0.0.1:7860/" in out            # this machine, not in the list
    assert "journalctl --user -u auradream -f" in out


@on_linux
def test_install_service_keeps_an_edited_env_file(sandbox):
    (sandbox["root"] / ".venv").mkdir()
    assert _run(sandbox, "install-service.sh").returncode == 0
    env_file = _env_file(sandbox)
    edited = env_file.read_text().replace("AURADREAM_ALLOWED_HOSTS=", "AURADREAM_ALLOWED_HOSTS=127.0.0.1,localhost,studio,") \
        + "AURADREAM_PORT=7999\n"
    env_file.write_text(edited)
    result = _run(sandbox, "install-service.sh", "--no-start")
    assert result.returncode == 0, result.stdout + result.stderr
    assert env_file.read_text() == edited
    assert "Kept as it is" in result.stdout
    assert "http://studio:7999/" in result.stdout
    assert _systemctl(sandbox)[-1] == "systemctl --user enable auradream.service"   # --no-start


@on_linux
def test_install_service_honours_xdg_config_home(sandbox):
    (sandbox["root"] / ".venv").mkdir()
    config = sandbox["tmp"] / "xdg"
    sandbox["env"]["XDG_CONFIG_HOME"] = str(config)
    result = _run(sandbox, "install-service.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _unit(sandbox, config).is_file() and _env_file(sandbox, config).is_file()
    assert not (sandbox["home"] / ".config").exists()


@on_linux
def test_install_service_uninstall_keeps_the_env_file(sandbox):
    (sandbox["root"] / ".venv").mkdir()
    assert _run(sandbox, "install-service.sh").returncode == 0
    result = _run(sandbox, "install-service.sh", "--uninstall")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not _unit(sandbox).exists()
    assert _env_file(sandbox).is_file()
    assert "Kept the environment file" in result.stdout
    assert "systemctl --user disable --now auradream.service" in _systemctl(sandbox)


@on_linux
def test_install_service_needs_a_venv(sandbox):
    result = _run(sandbox, "install-service.sh")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "no .venv yet" in result.stdout and "bash linux/setup.sh first" in result.stdout
    assert not (sandbox["home"] / ".config").exists()
    assert _systemctl(sandbox) == []


@on_linux
@pytest.mark.parametrize("script", ["setup.sh", "install-service.sh", "update.sh"])
def test_every_script_refuses_root(sandbox, script):
    (sandbox["root"] / ".venv").mkdir()
    (sandbox["root"] / ".git").mkdir()
    sandbox["env"]["STUB_UID"] = "0"
    before = _snapshot(sandbox["root"], sandbox["home"])
    result = _run(sandbox, script)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "as root" in result.stdout
    assert _snapshot(sandbox["root"], sandbox["home"]) == before
    assert _calls(sandbox) == []


# --- update.sh ----------------------------------------------------------------------

def _checkout_with_gallery(sandbox, name="auradream.db") -> None:
    (sandbox["root"] / ".git").mkdir()
    (sandbox["root"] / name).write_bytes(b"gallery")


def _no_git_or_uv(sandbox) -> bool:
    return not [c for c in _calls(sandbox) if c.startswith(("git", "uv"))]


@pytest.fixture
def auradream_listening(sandbox):
    """A loopback AuraDream on an ephemeral port: it answers while the stub
    service is active (or always, without one), its page with the title and
    its status with what `busy` says.

    With `gate["passcode"]` set it asks as `backend.passcode` does: signed
    out, `/` is the real sign-in page and everything else the coded 401; a
    request carrying that passcode as `Authorization: Bearer` is answered;
    while `gate["locked"]`, one carrying any passcode is a 429. `gate["sent"]`
    counts the requests that carried one."""
    busy: dict = {"status": {"state": "ready", "active_jobs": [], "switching": False, "card_holder": None},
                  "bundles": {"state": "idle"}}
    track = {"service": False}
    gate: dict = {"passcode": None, "locked": False, "sent": 0}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind="application/json", **headers):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server's name
            if track["service"] and not (sandbox["state"] / "active").exists():
                self.send_response(404)                # a tracked service that is stopped
                self.end_headers()
                return
            sent = self.headers.get("Authorization")
            gate["sent"] += sent is not None
            if gate["passcode"] is not None and sent != f"Bearer {gate['passcode']}":
                if sent is not None and gate["locked"]:
                    self._send(429, b'{"detail": "Too many wrong passcodes. Wait 7 seconds and try again."}',
                               Retry_After="7")
                elif self.path == "/":
                    self._send(200, (REPO / "web" / "passcode.html").read_bytes(), "text/html; charset=utf-8")
                else:
                    self._send(401, json.dumps({
                        "detail": "This server asks for a passcode. Reload the page and sign in.",
                        "code": "passcode_required",
                    }).encode(), WWW_Authenticate="Bearer")
                return
            if gate["passcode"] is not None and gate["locked"]:
                self._send(429, b'{"detail": "Too many wrong passcodes. Wait 7 seconds and try again."}',
                           Retry_After="7")
                return
            if self.path == "/api/status":
                body, kind = json.dumps(busy["status"]).encode(), "application/json"
            elif self.path == "/api/gallery/bundles/status":
                body, kind = json.dumps(busy["bundles"]).encode(), "application/json"
            else:
                body, kind = b"<!doctype html><title>AuraDream</title>", "text/html"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sandbox["env"]["AURADREAM_PORT"] = str(server.server_address[1])
    yield {"port": server.server_address[1], "busy": busy, "track": track, "gate": gate}
    server.shutdown()
    server.server_close()


@on_linux
@pytest.mark.parametrize("probe", ["venv-python", "curl"])
def test_update_refuses_while_auradream_is_running(sandbox, auradream_listening, probe):
    _checkout_with_gallery(sandbox)
    if probe == "venv-python":
        _venv(sandbox)
    elif not shutil.which("curl", path="/usr/bin:/bin"):
        pytest.skip("no curl here")
    before = _snapshot(sandbox["root"])
    result = _run(sandbox, "update.sh")               # a real run, not a dry one
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"AuraDream is running at http://127.0.0.1:{auradream_listening['port']}/" in result.stdout
    assert "Ctrl-C" in result.stdout
    assert _snapshot(sandbox["root"]) == before       # no database copy
    assert _no_git_or_uv(sandbox)


@on_linux
def test_update_refuses_while_the_service_is_active(sandbox):
    _checkout_with_gallery(sandbox)
    _service_active(sandbox)
    before = _snapshot(sandbox["root"])
    result = _run(sandbox, "update.sh")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "The AuraDream service is running" in result.stdout
    assert "systemctl --user stop auradream" in result.stdout and "--restart" in result.stdout
    assert _snapshot(sandbox["root"]) == before
    assert _no_git_or_uv(sandbox)
    assert _systemctl(sandbox) == ["systemctl --user is-active --quiet auradream.service"]


@on_linux
def test_update_refuses_while_a_venv_program_runs(sandbox):
    _checkout_with_gallery(sandbox)
    python = _venv(sandbox)
    # Started by the .venv's path, as uv run starts the server.
    child = subprocess.Popen([str(python), "-c", "import time; time.sleep(60)"])
    try:
        result = _run(sandbox, "update.sh")
    finally:
        child.kill()
        child.wait()
    assert result.returncode == 1, result.stdout + result.stderr
    assert "A program from this folder's .venv is still running" in result.stdout
    assert f"(process {child.pid})" in result.stdout
    assert _no_git_or_uv(sandbox)


@on_linux
def test_update_restart_dry_run_leaves_out_the_services_own_python(sandbox, auradream_listening):
    # The live server's first `auradream-server update -DryRun` refused over
    # the service's own python: a dry run does not stop it, the real run does.
    _checkout_with_gallery(sandbox)
    python = _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    service = subprocess.Popen([str(python), "-c", "import time; time.sleep(60)"])
    stray = subprocess.Popen([str(python), "-c", "import time; time.sleep(60)"])
    group = sandbox["tmp"] / "cgroup" / "user.slice" / "auradream.service"
    group.mkdir(parents=True)
    (group / "cgroup.procs").write_text(f"{service.pid}\n")
    (sandbox["state"] / "cgroup").write_text("/user.slice/auradream.service\n")
    sandbox["env"]["AURADREAM_INSTALL_TEST_CGROUP_ROOT"] = str(sandbox["tmp"] / "cgroup")
    try:
        lone = _run(sandbox, "update.sh", "--restart", "--dry-run")
        stray.kill()
        stray.wait()
        clean = _run(sandbox, "update.sh", "--restart", "--dry-run")
    finally:
        for child in (service, stray):
            child.kill()
            child.wait()
    # Another program from the .venv still stops it; the service's own does not.
    assert lone.returncode == 1 and f"(process {stray.pid})" in lone.stdout, lone.stdout
    assert f"(process {service.pid})" not in lone.stdout
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "still running" not in clean.stdout
    assert "would run: systemctl --user stop auradream" in clean.stdout


@on_linux
def test_update_dry_run_names_the_legacy_copy(sandbox):
    _checkout_with_gallery(sandbox, "local-diffusion.db")
    before = _snapshot(sandbox["root"])
    result = _run(sandbox, "update.sh", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "local-diffusion-before-update-abc1234.db" in result.stdout
    assert "would run in" in result.stdout and "git pull --ff-only" in result.stdout
    assert _snapshot(sandbox["root"]) == before
    assert [c for c in _calls(sandbox) if " pull" in c or c.startswith("uv")] == []


@on_linux
def test_update_copies_the_gallery_then_pulls_and_syncs(sandbox):
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    stale = root / "auradream-before-update-abc1234.db-wal"
    stale.write_bytes(b"old journal")
    (root / "auradream-before-update-abc1234.db-shm").write_bytes(b"old index")
    (root / "auradream.db-wal").write_bytes(b"live journal")
    (root / "auradream.db-shm").write_bytes(b"live index")
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update-abc1234.db").read_bytes() == b"gallery"
    assert (root / "auradream-before-update-abc1234.db-wal").read_bytes() == b"live journal"
    assert not (root / "auradream-before-update-abc1234.db-shm").exists()   # never copied
    calls = [c for c in _calls(sandbox) if c.startswith(("git", "uv"))]
    pulls = [i for i, c in enumerate(calls) if c.endswith("pull --ff-only")]
    assert pulls and calls.index("uv sync") > pulls[0]
    assert not any(word in c for c in calls for word in ("reset", "checkout", "clean", "stash"))
    assert "systemctl --user start auradream" in result.stdout
    assert "/proc/" not in result.stderr              # unreadable processes are skipped quietly


@on_linux
def test_update_removes_a_stale_copy_journal_when_the_gallery_has_none(sandbox):
    _checkout_with_gallery(sandbox)
    stale = sandbox["root"] / "auradream-before-update-abc1234.db-wal"
    stale.write_bytes(b"old journal")
    (sandbox["root"] / "auradream.db-wal").write_bytes(b"")          # empty: not copied
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not stale.exists()


@on_linux
def test_update_keeps_the_copy_an_earlier_update_made(sandbox):
    # The second update runs after the new version has changed the database:
    # the copy from before the first pull is still the way back.
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    sandbox["env"]["STUB_COMMIT"] = "aaa1111"
    assert _run(sandbox, "update.sh").returncode == 0
    (root / "auradream.db").write_bytes(b"migrated")
    sandbox["env"]["STUB_COMMIT"] = "bbb2222"
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update-aaa1111.db").read_bytes() == b"gallery"
    assert (root / "auradream-before-update-bbb2222.db").read_bytes() == b"migrated"
    assert not (root / "auradream-before-update.db").exists()


@on_linux
def test_update_leaves_a_copy_from_the_same_commit_alone(sandbox):
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copy = root / "auradream-before-update-abc1234.db"
    assert _run(sandbox, "update.sh").returncode == 0
    (copy.parent / (copy.name + "-wal")).write_bytes(b"the copy's journal")
    (root / "auradream.db").write_bytes(b"changed since")
    dry = _run(sandbox, "update.sh", "--dry-run")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert f"would leave {copy} as it is" in dry.stdout
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "left alone" in result.stdout
    assert copy.read_bytes() == b"gallery"
    assert (copy.parent / (copy.name + "-wal")).read_bytes() == b"the copy's journal"


def _older_copies(root: Path) -> dict[str, Path]:
    """Four named copies, a1 the oldest, and what never counts as one."""
    copies = {}
    for age, commit in enumerate(("a4a4a4a", "a3a3a3a", "a2a2a2a", "a1a1a1a"), start=1):
        path = root / f"auradream-before-update-{commit}.db"
        path.write_bytes(commit.encode())
        os.utime(path, (1_700_000_000 - age * 3600, 1_700_000_000 - age * 3600))
        copies[commit] = path
    (root / "auradream-before-update-a1a1a1a.db-wal").write_bytes(b"journal")
    (root / "auradream-before-update.db").write_bytes(b"the old fixed name")
    (root / "auradream-before-update-notes.db").write_bytes(b"not a commit")
    (root / "other-before-update-a0a0a0a.db").write_bytes(b"another database's")
    return copies


@on_linux
def test_update_keeps_the_newest_three_copies(sandbox):
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copies = _older_copies(root)
    before = _snapshot(root)
    dry = _run(sandbox, "update.sh", "--dry-run")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    for commit in ("a2a2a2a", "a1a1a1a"):
        assert f"would delete the older copy {copies[commit]}" in dry.stdout
    assert "a3a3a3a" not in dry.stdout and "a4a4a4a" not in dry.stdout
    assert _snapshot(root) == before
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update-abc1234.db").read_bytes() == b"gallery"
    assert copies["a4a4a4a"].exists() and copies["a3a3a3a"].exists()
    assert not copies["a2a2a2a"].exists() and not copies["a1a1a1a"].exists()
    assert not (root / "auradream-before-update-a1a1a1a.db-wal").exists()
    # Only this database's commit-named copies are counted or deleted.
    assert (root / "auradream-before-update.db").read_bytes() == b"the old fixed name"
    assert (root / "auradream-before-update-notes.db").exists()
    assert (root / "other-before-update-a0a0a0a.db").exists()


@on_linux
def test_update_counts_a_copy_it_left_alone_among_the_three(sandbox):
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copies = _older_copies(root)
    sandbox["env"]["STUB_COMMIT"] = "a1a1a1a"         # the oldest copy is this commit's
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert copies["a1a1a1a"].read_bytes() == b"a1a1a1a"
    assert copies["a4a4a4a"].exists() and copies["a3a3a3a"].exists()
    assert not copies["a2a2a2a"].exists()


@on_linux
def test_update_orders_a_copy_by_when_it_was_made(sandbox):
    # A gallery untouched for years: its copy is still the newest one, never
    # the first deleted because it carries the database's own time.
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    os.utime(root / "auradream.db", (1_000_000_000, 1_000_000_000))
    sandbox["env"]["STUB_COMMIT"] = "aaa1111"
    assert _run(sandbox, "update.sh").returncode == 0
    planted = []
    for age, commit in enumerate(("a2a2a2a", "a1a1a1a"), start=1):
        path = root / f"auradream-before-update-{commit}.db"
        path.write_bytes(commit.encode())
        os.utime(path, (1_700_000_000 - age * 3600, 1_700_000_000 - age * 3600))
        planted.append(path)
    sandbox["env"]["STUB_COMMIT"] = "bbb2222"
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update-aaa1111.db").read_bytes() == b"gallery"
    assert (root / "auradream-before-update-bbb2222.db").exists()
    assert planted[0].exists() and not planted[1].exists()


@on_linux
def test_update_without_a_commit_uses_the_fixed_name(sandbox):
    # Git answers no commit (an unborn branch): the one name, replaced.
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copies = _older_copies(root)
    sandbox["env"]["STUB_COMMIT"] = ""
    result = _run(sandbox, "update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update.db").read_bytes() == b"gallery"
    assert all(path.exists() for path in copies.values())   # nothing counted, nothing deleted


@on_linux
def test_update_reads_the_service_env_file(sandbox):
    # The service's AURADREAM_DB_PATH, not the checkout's default.
    _checkout_with_gallery(sandbox)
    data = sandbox["tmp"] / "data"
    data.mkdir()
    (data / "gallery.db").write_bytes(b"elsewhere")
    env_file = sandbox["home"] / ".config" / "auradream" / "auradream.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(f"# settings\nAURADREAM_DB_PATH={data / 'gallery.db'}\nAURADREAM_PORT=1\n")
    result = _run(sandbox, "update.sh", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"copy {data / 'gallery.db'} to {data / 'gallery-before-update-abc1234.db'}" in result.stdout
    assert "auradream-before-update" not in result.stdout


@on_linux
def test_update_explains_a_refused_pull_and_discards_nothing(sandbox):
    _checkout_with_gallery(sandbox)
    sandbox["env"]["STUB_PULL_EXIT"] = "128"
    sandbox["env"]["STUB_PULL_OUTPUT"] = "fatal: Not possible to fast-forward, aborting."
    result = _run(sandbox, "update.sh")
    assert result.returncode == 1
    assert "commits of its own" in result.stdout and "Nothing was discarded" in result.stdout
    calls = _calls(sandbox)
    assert not any(word in c for c in calls for word in ("reset", "checkout", "clean", "stash"))
    assert "uv sync" not in calls


@on_linux
def test_update_explains_untracked_files_in_the_way(sandbox):
    _checkout_with_gallery(sandbox)
    sandbox["env"]["STUB_PULL_EXIT"] = "1"
    sandbox["env"]["STUB_PULL_OUTPUT"] = \
        "error: The following untracked working tree files would be overwritten by merge:"
    result = _run(sandbox, "update.sh")
    assert result.returncode == 1
    assert "not part of the code" in result.stdout and "Move or rename them" in result.stdout
    assert "git restore" not in result.stdout
    assert "uv sync" not in _calls(sandbox)


@on_linux
def test_update_restart_stops_an_idle_service_and_starts_it_again(sandbox, auradream_listening):
    _checkout_with_gallery(sandbox)
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    result = _run(sandbox, "update.sh", "--restart")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [c for c in _calls(sandbox) if not c.startswith("systemctl --user is-active")]
    order = [c for c in calls if c in ("systemctl --user stop auradream.service", "uv sync",
                                       "systemctl --user start auradream.service")
             or c.endswith("pull --ff-only")]
    assert order[0] == "systemctl --user stop auradream.service"
    assert order[1].endswith("pull --ff-only") and order[2] == "uv sync"
    assert order[3] == "systemctl --user start auradream.service"
    assert "started again" in result.stdout
    assert (sandbox["state"] / "active").exists()


@on_linux
@pytest.mark.parametrize("busy", [
    ("status", "active_jobs", ["job-1"], "1 job(s) queued or running"),
    ("status", "state", "loading", "a model is loading"),
    ("bundles", "state", "running", "a gallery bundle operation is running"),
])
def test_update_restart_refuses_a_busy_service(sandbox, auradream_listening, busy):
    _checkout_with_gallery(sandbox)
    _venv(sandbox)
    _service_active(sandbox)
    part, key, value, words = busy
    auradream_listening["busy"][part][key] = value
    before = _snapshot(sandbox["root"])
    result = _run(sandbox, "update.sh", "--restart")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "The AuraDream service is busy" in result.stdout and words in result.stdout
    assert _snapshot(sandbox["root"]) == before
    assert _no_git_or_uv(sandbox)
    assert _systemctl(sandbox) == ["systemctl --user is-active --quiet auradream.service"]


@on_linux
def test_update_restart_starts_the_service_again_after_a_failed_update(sandbox, auradream_listening):
    _checkout_with_gallery(sandbox)
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    sandbox["env"]["STUB_PULL_EXIT"] = "128"
    sandbox["env"]["STUB_PULL_OUTPUT"] = "fatal: unable to access 'https://github.com/': Could not resolve host"
    result = _run(sandbox, "update.sh", "--restart")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Git could not reach GitHub" in result.stdout
    assert "The update did not finish; starting the AuraDream service again" in result.stdout
    assert _systemctl(sandbox)[-1] == "systemctl --user start auradream.service"
    assert "uv sync" not in _calls(sandbox)


@on_linux
def test_update_restart_after_a_failed_sync_says_the_code_moved(sandbox, auradream_listening):
    _checkout_with_gallery(sandbox)
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    sandbox["env"]["STUB_UV_EXIT"] = "1"
    result = _run(sandbox, "update.sh", "--restart")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "The code is updated but uv sync did not finish; starting the service on it anyway" in result.stdout
    assert "as it was" not in result.stdout
    assert _systemctl(sandbox)[-1] == "systemctl --user start auradream.service"
    assert (sandbox["state"] / "active").exists()


@on_linux
@pytest.mark.parametrize("gone", ["closed-pipe", "hangup"])
def test_update_restart_starts_the_service_again_when_its_output_is_gone(sandbox, auradream_listening, gone):
    # Output through a pipe whose reader ends mid-update (tee after Ctrl-C),
    # or a hang-up as well (an ssh session that dropped): the writes fail,
    # and the service must still be started again.
    _checkout_with_gallery(sandbox)
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    go = sandbox["tmp"] / "go"
    sandbox["env"]["STUB_PULL_WAIT"] = str(go)
    sandbox["env"]["STUB_PULL_EXIT"] = "128"
    sandbox["env"]["STUB_PULL_OUTPUT"] = "fatal: unable to access 'https://github.com/': Could not resolve host"
    errors = (sandbox["tmp"] / "stderr").open("w")
    child = subprocess.Popen(
        [BASH, str(LINUX / "update.sh"), "--restart"], env=sandbox["env"], stdout=subprocess.PIPE,
        stderr=errors, stdin=subprocess.DEVNULL, cwd=sandbox["tmp"],
    )
    try:
        assert child.stdout is not None
        for line in child.stdout:
            if b"git pull --ff-only" in line:
                break
        child.stdout.close()                         # the reader is gone
        if gone == "hangup":
            child.send_signal(signal.SIGHUP)
        go.touch()
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        errors.close()
    assert child.returncode != 0
    assert _systemctl(sandbox)[-1] == "systemctl --user start auradream.service"
    assert (sandbox["state"] / "active").exists()
    assert "uv sync" not in _calls(sandbox)


# --- control.sh ---------------------------------------------------------------------

def _changes(sandbox) -> list[str]:
    """The systemctl calls that change the service (not its reads)."""
    return [c for c in _systemctl(sandbox) if c.split()[2] in ("start", "stop", "restart")]


@on_linux
def test_control_status_of_an_active_answering_service(sandbox, auradream_listening):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    auradream_listening["busy"]["status"].update({
        "model_id": "cagliostrolab/animagine-xl-4.0", "model_label": "Animagine XL 4.0",
        "quantization": "fp8", "offload": "model", "state": "ready",
        "vram_used_mb": 4289, "vram_total_mb": 7842, "device_name": "NVIDIA GeForce RTX 3070",
        "disk_free_mb": 819200, "commit": "abc1234",
        "keep_awake": {"kind": "systemd-inhibit", "holding": True, "wanted": True,
                       "reasons": ["in use (a request in the last 20 minutes)"], "detail": None},
    })
    sandbox["env"]["STUB_AHEAD_BEHIND"] = "0\t3"
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "active (running) since Tue 2026-11-03 10:00:00 AEDT" in out
    assert "Enabled" in out and "Lingering is on" in out
    assert "Branch main at abc1234 Keep awake (2 days ago)" in out
    assert "3 commit(s) behind origin/main" in out
    assert "It answers." in out
    assert "Commit:       abc1234\n" in out              # the process's own, the checkout's too
    assert "Animagine XL 4.0 (cagliostrolab/animagine-xl-4.0)" in out and "fp8, model offload" in out
    assert "Jobs:         none" in out and "Card:         free" in out
    assert "systemd-inhibit, holding: in use (a request in the last 20 minutes)" in out
    assert "4289 of 7842 MiB in use (NVIDIA GeForce RTX 3070)" in out and "800.0 GiB" in out
    assert "AuraDream 1000 tester 4321 systemd-inhibit" in out and "UPower" not in out
    assert "The service is running and AuraDream answers." in out
    assert any(c.endswith("fetch --quiet") for c in _calls(sandbox))
    assert _changes(sandbox) == []


@on_linux
@pytest.mark.parametrize(("awake", "said"), [
    # Idle is not off: the first status on the live server called a released
    # hold "off" because only `wanted` was read.
    ({"kind": "systemd-inhibit", "holding": False, "wanted": False, "reasons": []},
     "systemd-inhibit, not holding: idle, so the PC may sleep"),
    ({"kind": "off", "holding": False, "wanted": False, "reasons": []},
     "off (AURADREAM_KEEP_AWAKE=false)"),
])
def test_control_status_tells_idle_from_switched_off(sandbox, auradream_listening, awake, said):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    auradream_listening["busy"]["status"]["keep_awake"] = {**awake, "detail": None, "idle_minutes": 20}
    result = _run(sandbox, "control.sh", "status")
    assert f"Keep awake:   {said}" in result.stdout, result.stdout + result.stderr


@on_linux
@pytest.mark.parametrize(("running", "said"), [
    # A pull without a restart: the checkout moved, the process did not.
    ("def5678", "Commit:       def5678, and the checkout is at abc1234: restart it to run the checkout's code"),
    ("abc1234e9", "Commit:       abc1234e9\n"),          # the same commit, abbreviated longer
    (None, None),                                         # a server that could not tell: no line
])
def test_control_status_names_the_running_commit(sandbox, auradream_listening, running, said):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    auradream_listening["busy"]["status"]["commit"] = running
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    if said is None:
        assert "Commit:" not in out
    else:
        assert said in out, out


@on_linux
def test_control_status_of_a_stopped_service_still_reports(sandbox, auradream_listening):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    auradream_listening["track"]["service"] = True        # stopped: nothing answers
    sandbox["env"]["STUB_FETCH_EXIT"] = "128"             # and no network: noted, not fatal
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 1, out + result.stderr
    assert "inactive (dead)" in out
    assert "git fetch failed, so that is as of the last fetch: fatal: unable to access the remote" in out
    assert "It does not answer" in out
    assert "== Sleep holds" in out                         # everything it can is still printed
    assert "The service is not running, or AuraDream does not answer" in out
    assert _changes(sandbox) == []


@on_linux
def test_control_stop_refuses_a_busy_server_and_force_stops(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["busy"]["status"]["active_jobs"] = ["job-1"]
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "The AuraDream service is busy: 1 job(s) queued or running." in result.stdout
    assert "--force" in result.stdout
    assert _changes(sandbox) == []
    assert (sandbox["state"] / "active").exists()

    forced = _run(sandbox, "control.sh", "stop", "--force")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert "stopping it although it is busy: 1 job(s) queued or running" in forced.stdout
    assert _changes(sandbox) == ["systemctl --user stop auradream.service"]
    assert not (sandbox["state"] / "active").exists()


@on_linux
def test_control_stop_of_an_idle_server(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "It is idle." in result.stdout and "is stopped" in result.stdout
    assert _changes(sandbox) == ["systemctl --user stop auradream.service"]


@on_linux
def test_control_restart_waits_for_the_page(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    sandbox["env"]["STUB_START_DELAY"] = "2"               # the page answers 2 s after the restart
    result = _run(sandbox, "control.sh", "restart")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "It is idle." in out and "Waiting for the page" in out
    assert f"AuraDream answers at http://127.0.0.1:{auradream_listening['port']}/" in out
    assert _changes(sandbox) == ["systemctl --user restart auradream.service"]


@on_linux
def test_control_restart_refuses_a_busy_server(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["busy"]["bundles"]["state"] = "running"
    result = _run(sandbox, "control.sh", "restart")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "a gallery bundle operation is running" in result.stdout
    assert _changes(sandbox) == []


@on_linux
def test_control_start_says_when_the_page_never_answers(sandbox):
    _venv(sandbox)
    sandbox["env"]["AURADREAM_INSTALL_TEST_WAIT"] = "2"    # nothing listens on port 1
    result = _run(sandbox, "control.sh", "start")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Started." in result.stdout and "did not answer within 2 s" in result.stdout
    assert _changes(sandbox) == ["systemctl --user start auradream.service"]


@on_linux
def test_control_start_stops_waiting_for_a_failed_unit(sandbox):
    _venv(sandbox)
    (sandbox["state"] / "failed").touch()
    result = _run(sandbox, "control.sh", "start")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "The AuraDream service failed while starting." in result.stdout


@on_linux
def test_control_start_stops_waiting_for_an_auto_restarting_unit(sandbox):
    """Restart=on-failure: a crash while starting is auto-restart, not failed."""
    _venv(sandbox)
    (sandbox["state"] / "auto-restart").touch()
    result = _run(sandbox, "control.sh", "start")       # the default 60 s wait
    assert result.returncode == 1, result.stdout + result.stderr
    assert "it is crashing, not slow" in result.stdout
    assert "did not answer within" not in result.stdout


@on_linux
def test_control_restart_stops_waiting_when_the_unit_restarts(sandbox):
    """NRestarts growing since the start is a crash too, whatever SubState says."""
    _venv(sandbox)
    (sandbox["state"] / "active").touch()
    sandbox["env"]["STUB_CRASH_AFTER"] = "1"
    result = _run(sandbox, "control.sh", "restart", "--force")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "it is crashing, not slow" in result.stdout
    assert "did not answer within" not in result.stdout


@on_linux
@pytest.mark.parametrize(("args", "expected"), [
    ((), "journalctl --user -u auradream.service -n 100 --no-pager"),
    (("-n", "20"), "journalctl --user -u auradream.service -n 20 --no-pager"),
    (("--lines", "5", "--follow"), "journalctl --user -u auradream.service -n 5 -f"),
    (("-f",), "journalctl --user -u auradream.service -n 100 -f"),
])
def test_control_logs_passes_lines_and_follow_through(sandbox, args, expected):
    result = _run(sandbox, "control.sh", "logs", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Uvicorn running" in result.stdout
    assert [c for c in _calls(sandbox) if c.startswith("journalctl")] == [expected]


@on_linux
def test_control_doctor_runs_with_the_service_settings(sandbox):
    env_file = _env_file(sandbox)
    env_file.parent.mkdir(parents=True)
    env_file.write_text("AURADREAM_HOST=0.0.0.0\nAURADREAM_PORT=7999\n")
    _stub(sandbox["stubs"], "uv", 'printf "doctor sees %s:%s in %s\\n" "$AURADREAM_HOST" "$AURADREAM_PORT" "$PWD"')
    result = _run(sandbox, "control.sh", "doctor")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"doctor sees 0.0.0.0:7999 in {sandbox['root']}" in result.stdout
    assert [c for c in _calls(sandbox) if c.startswith("uv")] == ["uv run --no-sync python -m backend.doctor"]


@on_linux
@pytest.mark.parametrize("args", [("bogus",), ("logs", "--lines", "many"), ("stop", "--now")])
def test_control_refuses_an_unknown_command_or_option(sandbox, args):
    result = _run(sandbox, "control.sh", *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Unknown" in result.stderr
    assert _changes(sandbox) == []


@on_linux
def test_control_help_and_root(sandbox):
    result = _run(sandbox, "control.sh")
    assert result.returncode == 0 and "bash linux/control.sh status" in result.stdout
    sandbox["env"]["STUB_UID"] = "0"
    result = _run(sandbox, "control.sh", "stop", "--force")
    assert result.returncode == 1 and "as root" in result.stdout
    assert _calls(sandbox) == []


# --- The passcode -------------------------------------------------------------------
#
# A server that asks (docs/design-notes.md, "A passcode beyond loopback"): the
# scripts read AURADREAM_PASSCODE from the service's environment file and send
# it as a header; a 401 and a 429 are each said in their own words, never as
# a server that is down or idle; and a passcode bash would read differently
# from systemd is refused before the file is sourced. The value is never
# printed.

SECRET = "Horse.Battery_9-+@%:,/"                 # every character the file may hold


def _passcode_file(sandbox, value: str, more: str = "") -> Path:
    env_file = _env_file(sandbox)
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_bytes(f"# settings\nAURADREAM_HOST=0.0.0.0\nAURADREAM_PASSCODE={value}\n{more}".encode())
    return env_file


def _asking(sandbox, listening, *, file: str | None = SECRET, served: str = SECRET) -> None:
    """A running service whose server asks for `served`; the environment
    file holds `file` (None: no such line)."""
    _venv(sandbox)
    _service_active(sandbox)
    listening["gate"]["passcode"] = served
    if file is not None:
        _passcode_file(sandbox, file)


@on_linux
def test_install_service_names_the_passcode_in_a_commented_line(sandbox):
    (sandbox["root"] / ".venv").mkdir()
    assert _run(sandbox, "install-service.sh").returncode == 0
    text = _env_file(sandbox).read_text()
    assert "\n#AURADREAM_PASSCODE=\n" in text
    assert "letters, digits and . _ - + @ % : , /" in text and "8 characters or more" in text
    assert "never reuse a password" in text
    # Commented: nothing is set, and the file as written passes its own check.
    assert not [line for line in text.splitlines() if line.startswith("AURADREAM_PASSCODE")]
    again = _run(sandbox, "install-service.sh", "--no-start")
    assert again.returncode == 0 and "[stop]" not in again.stdout, again.stdout + again.stderr


@on_linux
def test_control_status_reads_through_the_passcode(sandbox, auradream_listening):
    (sandbox["root"] / ".git").mkdir()
    _asking(sandbox, auradream_listening)
    auradream_listening["busy"]["status"].update({"model_id": "m/checkpoint", "commit": "abc1234"})
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "It answers." in out and "Checkpoint:   m/checkpoint" in out and "Jobs:         none" in out
    assert "could not be read" not in out
    assert auradream_listening["gate"]["sent"] == 1
    assert SECRET not in out + result.stderr


@on_linux
def test_stop_and_update_ask_whether_it_is_busy_through_the_passcode(sandbox, auradream_listening):
    _checkout_with_gallery(sandbox)
    _asking(sandbox, auradream_listening)
    auradream_listening["busy"]["status"]["active_jobs"] = ["job-1"]
    busy = _run(sandbox, "control.sh", "stop")
    assert busy.returncode == 1, busy.stdout + busy.stderr
    assert "The AuraDream service is busy: 1 job(s) queued or running." in busy.stdout
    updating = _run(sandbox, "update.sh", "--restart")
    assert updating.returncode == 1 and "The AuraDream service is busy: 1 job(s)" in updating.stdout
    assert _changes(sandbox) == [] and _no_git_or_uv(sandbox)

    auradream_listening["busy"]["status"]["active_jobs"] = []
    idle = _run(sandbox, "control.sh", "stop")
    assert idle.returncode == 0, idle.stdout + idle.stderr
    assert "It is idle." in idle.stdout
    assert _changes(sandbox) == ["systemctl --user stop auradream.service"]
    for result in (busy, updating, idle):
        assert SECRET not in result.stdout + result.stderr


@on_linux
@pytest.mark.parametrize(("file", "locked", "words"), [
    # A 401: no passcode in the file, or one the server does not accept.
    (None, False, "it asks for a passcode, and AURADREAM_PASSCODE is missing from the service's environment file"),
    ("another-passcode", False, "it did not accept the AURADREAM_PASSCODE these scripts read"),
    # A 429: the lock on wrong passcodes, the right one refused too.
    (SECRET, True, "it is refusing every passcode for now, after too many wrong ones just now: wait 7 s"),
])
def test_a_refused_passcode_is_said_in_its_own_words_and_never_taken_for_idle(
        sandbox, auradream_listening, file, locked, words):
    _checkout_with_gallery(sandbox)
    _asking(sandbox, auradream_listening, file=file)
    auradream_listening["gate"]["locked"] = locked
    before = _snapshot(sandbox["root"])

    stop = _run(sandbox, "control.sh", "stop")
    assert stop.returncode == 1, stop.stdout + stop.stderr
    assert "Whether the AuraDream service is busy could not be asked" in stop.stdout and words in stop.stdout
    assert "--force" in stop.stdout
    restart = _run(sandbox, "control.sh", "restart")
    assert restart.returncode == 1 and words in restart.stdout
    update = _run(sandbox, "update.sh", "--restart")
    assert update.returncode == 1, update.stdout + update.stderr
    assert "whether it is busy could not be asked" in update.stdout and words in update.stdout
    # Nothing stopped, nothing pulled, no database copy.
    assert _changes(sandbox) == [] and _no_git_or_uv(sandbox)
    assert (sandbox["state"] / "active").exists() and _snapshot(sandbox["root"]) == before

    # status: the server is up; only its status is closed.
    status = _run(sandbox, "control.sh", "status")
    assert status.returncode == 0, status.stdout + status.stderr
    assert "It answers." in status.stdout and f"Its status could not be read: {words}" in status.stdout
    assert "It does not answer" not in status.stdout
    assert "The service is running and AuraDream answers." in status.stdout

    for result in (stop, restart, update, status):
        said = result.stdout + result.stderr
        assert SECRET not in said and "another-passcode" not in said


@on_linux
def test_a_forced_restart_without_the_passcode_still_sees_the_page_come_up(sandbox, auradream_listening):
    _asking(sandbox, auradream_listening, file=None)
    auradream_listening["track"]["service"] = True
    sandbox["env"]["AURADREAM_INSTALL_TEST_WAIT"] = "20"
    result = _run(sandbox, "control.sh", "restart", "--force")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "whether it is busy could not be asked" in out and "restarting it anyway" in out
    assert f"AuraDream answers at http://127.0.0.1:{auradream_listening['port']}/" in out
    assert "These scripts cannot ask it anything more: it asks for a passcode" in out
    assert "did not answer within" not in out
    assert _changes(sandbox) == ["systemctl --user restart auradream.service"]


@on_linux
@pytest.mark.parametrize("probe", ["venv-python", "curl"])
def test_the_title_probe_reads_the_sign_in_page_as_auradream(sandbox, auradream_listening, probe):
    # No passcode anywhere here: the probe needs none.
    _checkout_with_gallery(sandbox)
    if probe == "venv-python":
        _venv(sandbox)
    elif not shutil.which("curl", path="/usr/bin:/bin"):
        pytest.skip("no curl here")
    auradream_listening["gate"]["passcode"] = SECRET
    result = _run(sandbox, "update.sh")
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"AuraDream is running at http://127.0.0.1:{auradream_listening['port']}/" in result.stdout
    assert auradream_listening["gate"]["sent"] == 0 and _no_git_or_uv(sandbox)


# What bash, sourcing the file, would read differently from systemd. The
# first would set the passcode to "correct" and run a program named horse.
# Each is (the value, a part of it that must never be printed).
_UNSAFE = [
    ("correct horse", "correct"), ("pa$$word-12", "word-12"), ("$(touch {marker})-12345", "12345"),
    ("`touch {marker}`-12345", "12345"), ('say"quoted"12', "quoted"), ("it's-a-passcode", "a-passcode"),
    ("semi;colon-12; touch {marker}", "colon-12"), ("hash #tag-1234", "tag-1234"), ("back\\slash-12", "slash-12"),
    ("amp&ersand-12", "ersand"), ("pipe|touch {marker}", "pipe|"), ("star*glob-123", "glob-123"),
    ("tilde~home-12", "tilde"), ("bang!bang-123", "bang-123"), ("less<more>-12", "more"),
    ("(parens)-1234", "parens"), ("Windows-line-end\r", "Windows-line-end"),
]


@on_linux
@pytest.mark.parametrize(("value", "part"), _UNSAFE)
@pytest.mark.parametrize(("script", "args"), [
    ("control.sh", ("status",)), ("control.sh", ("stop", "--force")), ("control.sh", ("doctor",)),
    ("update.sh", ("--dry-run",)), ("update.sh", ("--restart",)), ("setup.sh", ("--dry-run",)),
    ("install-service.sh", ()), ("install-service.sh", ("--dry-run",)),
])
def test_a_passcode_the_file_cannot_hold_is_refused_before_the_file_is_sourced(sandbox, script, args, value, part):
    marker = sandbox["tmp"] / "sourced"
    value = value.replace("{marker}", str(marker))
    _checkout_with_gallery(sandbox)
    (sandbox["root"] / ".venv").mkdir()
    _service_active(sandbox)
    # Programs of those names are on PATH: sourcing the file would run one.
    for name in ("horse", "touch"):
        _stub(sandbox["stubs"], name, f': >"{marker}"')
    env_file = _passcode_file(sandbox, value, more="AURADREAM_PORT=1\n")
    before = _snapshot(sandbox["root"], sandbox["home"])
    result = _run(sandbox, script, *args)
    said = result.stdout + result.stderr
    assert result.returncode == 1, said
    assert f"[stop] The AURADREAM_PASSCODE line in {env_file} has a character" in result.stdout
    assert "letters, digits and . _ - + @ % : , /" in result.stdout
    assert not marker.exists()
    assert not [c for c in _calls(sandbox) if c.startswith(("horse", "touch"))]
    # The value is never printed, whole or in part.
    assert value.strip() not in said and part not in said
    assert _snapshot(sandbox["root"], sandbox["home"]) == before
    assert _changes(sandbox) == [] and _no_git_or_uv(sandbox)


@on_linux
@pytest.mark.parametrize(("line", "sent"), [
    (f"AURADREAM_PASSCODE={SECRET}", 2), (f"  AURADREAM_PASSCODE={SECRET}  ", 2), ("AURADREAM_PASSCODE=", 0),
    # A comment is no setting, whatever it holds.
    ("#AURADREAM_PASSCODE=correct horse", 0), ("# AURADREAM_PASSCODE=pa$$word", 0),
    ("OTHER_AURADREAM_PASSCODE=1", 0),
])
def test_a_plain_passcode_or_none_is_read(sandbox, auradream_listening, line, sent):
    _venv(sandbox)
    _service_active(sandbox)
    env_file = _env_file(sandbox)
    env_file.parent.mkdir(parents=True)
    env_file.write_text(f"AURADREAM_HOST=0.0.0.0\n{line}\n")
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[stop]" not in result.stdout and "It is idle." in result.stdout
    # Sent (with both of the busy check's reads) only where the file sets one.
    assert auradream_listening["gate"]["sent"] == sent


@on_linux
def test_the_passcode_travels_in_the_environment_never_as_an_argument(sandbox, auradream_listening):
    # The python the scripts ask with is replaced by one that records its
    # arguments and then runs the real one.
    real = _venv(sandbox)
    target = os.readlink(real)
    seen = sandbox["tmp"] / "argv"
    real.unlink()
    real.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >>"{seen}"\nexec "{target}" "$@"\n')
    real.chmod(0o755)
    _service_active(sandbox)
    auradream_listening["gate"]["passcode"] = SECRET
    _passcode_file(sandbox, SECRET)
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert auradream_listening["gate"]["sent"] == 2 and seen.exists()
    assert SECRET not in seen.read_text()
