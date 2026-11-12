"""What the `linux/*.sh` tests share: the stub sandbox and the loopback
AuraDream the scripts probe.

`sandbox` runs linux/setup.sh, install-service.sh, update.sh and control.sh
for real under bash against stub nvidia-smi/uv/git/systemctl/journalctl/
systemd-inhibit/loginctl/hostname/id scripts in a temporary folder (Linux
only; `on_linux` marks those tests). `AURADREAM_INSTALL_TEST_ROOT` points the
scripts at a temporary project folder and `AURADREAM_INSTALL_TEST_PROC_VERSION`
at a stand-in /proc/version, as linux/common.sh documents. PATH holds the
stubs, then /usr/bin and /bin: every stub shadows the machine's own tool, so
no real systemctl --user, git or uv ever runs. HOME is temporary and
XDG_CONFIG_HOME unset unless a test sets it, so the unit and environment file
land there. No network: `auradream_listening` is a server on loopback.

`test_linux_scripts.py`, `test_linux_control.py` and `test_linux_passcode.py`
import these by name, the fixtures included.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.tls_helpers import loopback_certificate

REPO = Path(__file__).resolve().parents[1]
LINUX = REPO / "linux"
SCRIPTS = sorted(LINUX.glob("*.sh"))
BASH = shutil.which("bash")
on_linux = pytest.mark.skipif(sys.platform != "linux" or BASH is None, reason="bash scripts for native Linux")


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


def _unit(sandbox, config: Path | None = None) -> Path:
    return (config or sandbox["home"] / ".config") / "systemd" / "user" / "auradream.service"


def _env_file(sandbox, config: Path | None = None) -> Path:
    return (config or sandbox["home"] / ".config") / "auradream" / "auradream.env"


def _checkout_with_gallery(sandbox, name="auradream.db") -> None:
    (sandbox["root"] / ".git").mkdir()
    (sandbox["root"] / name).write_bytes(b"gallery")


def _no_git_or_uv(sandbox) -> bool:
    return not [c for c in _calls(sandbox) if c.startswith(("git", "uv"))]


@pytest.fixture
def auradream_listening(sandbox, request):
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
            if gate.get("redirect"):                   # given the passcode, it points elsewhere
                self._send(307, b"", Location=gate["redirect"] + self.path)
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
    if getattr(request, "param", "http") == "https":
        certfile, keyfile = loopback_certificate(sandbox["tmp"])
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile, keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        sandbox["env"].update(AURADREAM_SSL_CERTFILE=str(certfile), AURADREAM_SSL_KEYFILE=str(keyfile),
                              SSL_CERT_FILE=str(certfile), CURL_CA_BUNDLE=str(certfile))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sandbox["env"]["AURADREAM_PORT"] = str(server.server_address[1])
    yield {"port": server.server_address[1], "busy": busy, "track": track, "gate": gate}
    server.shutdown()
    server.server_close()


def _changes(sandbox) -> list[str]:
    """The systemctl calls that change the service (not its reads)."""
    return [c for c in _systemctl(sandbox) if c.split()[2] in ("start", "stop", "restart")]


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
