"""install.ps1 and update.ps1, run for real by Windows PowerShell against stub
nvidia-smi/winget/uv/git scripts in a temporary folder (Windows only).

`AURADREAM_INSTALL_TEST_ROOT` points the scripts at a temporary project
folder; PATH holds the stubs and the Windows system folders only, so the
machine's own uv, Git and winget are not found (its nvidia-smi is in
System32, behind the stub). USERPROFILE, APPDATA and LOCALAPPDATA are
temporary too, so shortcut and cache paths land there. No network.
"""
from __future__ import annotations

import hashlib
import os
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.tls_helpers import loopback_certificate

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell scripts")

REPO = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = os.environ.get("SystemRoot", r"C:\Windows")


def _write_cmd(path: Path, lines: list[str]) -> None:
    path.write_text("\r\n".join(["@echo off", *lines, ""]), encoding="ascii")


def _nvidia_smi(stubs: Path, header: str, query: str) -> None:
    _write_cmd(stubs / "nvidia-smi.cmd", [
        f'if /i "%~1"=="--query-gpu" (echo {query}& exit /b 0)',
        f"echo {header}",
        "exit /b 0",
    ])


def _git(stubs: Path, pull_exit: int = 0, pull_output: str = "Already up to date.") -> None:
    _write_cmd(stubs / "git.cmd", [
        'echo git %*>>"%STUB_LOG%"',
        'if "%~1"=="--version" (echo git version 2.99.0.stub& exit /b 0)',
        f'if "%~3"=="pull" (echo pulled>"%STUB_LOG%.pulled"& echo {pull_output}& exit /b {pull_exit})',
        'if "%~3"=="remote" (echo https://github.com/matt439/AuraDream.git& exit /b 0)',
        # The commit update.ps1 names the database copy by.
        'if "%~4"=="--short" (',
        '  if defined STUB_AFTER_COMMIT if exist "%STUB_LOG%.pulled" (echo %STUB_AFTER_COMMIT%& exit /b 0)',
        '  echo %STUB_COMMIT%& exit /b 0',
        ')',
        'if "%~3"=="rev-list" (echo 3& exit /b 0)',
        'if "%~3"=="rev-parse" (echo main& exit /b 0)',
        "exit /b 0",
    ])


def _uv(stubs: Path) -> None:
    _write_cmd(stubs / "uv.cmd", [
        'echo uv %*>>"%STUB_LOG%"',
        'if "%~1"=="--version" (echo uv 0.12.7 stub& exit /b 0)',
        "exit /b 0",
    ])


def _winget(stubs: Path) -> None:
    _write_cmd(stubs / "winget.cmd", ['echo winget %*>>"%STUB_LOG%"', "exit /b 0"])


@pytest.fixture
def sandbox(tmp_path):
    """A fake project folder, a stub folder and a temporary home."""
    root = tmp_path / "AuraDream"
    (root / "backend").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "auradream"\n')
    (root / "backend" / "main.py").write_text("")
    (root / "bin" / "auradream.cmd").write_text("@echo off\r\n")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    home = tmp_path / "home"
    for sub in ("Roaming", "Local", "Desktop"):
        (home / sub).mkdir(parents=True)
    log = tmp_path / "calls.log"
    env = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("AURADREAM_", "HF_", "UV_", "ONEDRIVE"))
        and key.upper() not in ("PATH", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOME")
    }
    env.update({
        "PATH": ";".join([str(stubs), rf"{SYSTEM_ROOT}\System32", SYSTEM_ROOT,
                          rf"{SYSTEM_ROOT}\System32\WindowsPowerShell\v1.0", rf"{SYSTEM_ROOT}\System32\Wbem"]),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "Roaming"),
        "LOCALAPPDATA": str(home / "Local"),
        "AURADREAM_INSTALL_TEST_ROOT": str(root),
        "STUB_LOG": str(log),
        "STUB_COMMIT": "abc1234",
        # Nothing listens on port 1: a server running on this machine's 7860
        # never answers for the sandbox. Tests that want one set their own.
        "AURADREAM_PORT": "1",
    })
    return {"root": root, "stubs": stubs, "home": home, "log": log, "env": env, "tmp": tmp_path}


def _run(sandbox, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(REPO / script), *args],
        env=sandbox["env"], capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180, stdin=subprocess.DEVNULL,
    )


def _snapshot(*folders: Path) -> dict[str, str]:
    """Every file under `folders` with its content hash (a .lnk included)."""
    found = {}
    for folder in folders:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                found[str(path)] = "dir"
    return found


def _calls(sandbox) -> list[str]:
    log = sandbox["log"]
    return log.read_text().splitlines() if log.exists() else []


def test_install_dry_run_on_an_old_machine_from_a_zip(sandbox):
    # Driver too old, a GTX 1080, no uv, no winget, a ZIP folder (no .git).
    _nvidia_smi(sandbox["stubs"], "NVIDIA-SMI 551.86   Driver Version: 551.86   CUDA Version: 12.4",
                "NVIDIA GeForce GTX 1080, 6.1, 8192")
    _git(sandbox["stubs"])
    before = _snapshot(sandbox["root"], sandbox["home"] / "Roaming", sandbox["home"] / "Desktop")
    result = _run(sandbox, "install.ps1", "-DryRun", "-Yes")
    out = result.stdout
    assert result.returncode == 1, out + result.stderr        # a real run would stop
    assert "The driver supports CUDA 12.4" in out and "nvidia.com/Download" in out
    assert "compute capability 6.1" in out and "not supported" in out
    assert "would install uv with uv's own installer, version 0.12.7" in out
    assert "https://astral.sh/uv/0.12.7/install.ps1" in out
    assert "no .git: it came from a ZIP download" in out
    assert "would run: git init; git remote add origin https://github.com/matt439/AuraDream.git" in out
    assert "Folder for model downloads" in out and str(sandbox["home"] / ".cache" / "huggingface") in out
    assert "would run in" in out and "uv sync" in out
    assert "would create the shortcut " + str(sandbox["home"] / "Roaming" / "Microsoft" / "Windows"
                                             / "Start Menu" / "Programs" / "AuraDream.lnk") in out
    assert str(sandbox["home"] / "Desktop" / "AuraDream.lnk") in out
    assert "--model cagliostrolab/animagine-xl-4.0 --open" in out       # no sign-in: an open checkpoint
    assert f"open PowerShell in '{sandbox['root']}' and run: uv run hf auth login" in out
    assert "double-click install.cmd in that folder again and answer y" in out
    assert 'press Enter for "Log in with your browser"' in out
    assert '"Paste an access token" takes a Read token; pasted input stays hidden' in out
    assert "A real run would stop at:" in out
    # Nothing changed: the folder, the shortcuts' places, no HF cache, no git or uv run.
    assert _snapshot(sandbox["root"], sandbox["home"] / "Roaming", sandbox["home"] / "Desktop") == before
    assert not (sandbox["home"] / ".cache").exists()
    assert not list(sandbox["home"].rglob("*.lnk"))
    assert [call for call in _calls(sandbox) if call != "git --version"] == []


def test_install_dry_run_on_a_supported_checkout(sandbox):
    # An unreadable CUDA version only warns; winget is used for uv; a checkout
    # needs no conversion; nothing would stop.
    (sandbox["root"] / ".git").mkdir()
    _nvidia_smi(sandbox["stubs"], "NVIDIA-SMI something new", "NVIDIA GeForce RTX 4060, 8.9, 8188")
    _git(sandbox["stubs"])
    _winget(sandbox["stubs"])
    result = _run(sandbox, "install.ps1", "-DryRun", "-Yes")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "its CUDA version could not be read" in out and "[warn]" in out
    assert "Compute capability 8.9 is supported" in out
    assert "8 GB of video memory" in out and "8 GB RTX 3070 under native Linux" in out
    assert "would install uv: winget install --id astral-sh.uv -e --accept-source-agreements" in out
    assert "is a Git checkout (branch main" in out
    assert "[stop]" not in out and "Dry run finished. Nothing was changed." in out
    assert not [c for c in _calls(sandbox) if c.startswith(("winget", "uv"))]   # winget never ran
    assert not list(sandbox["home"].rglob("*.lnk"))


def test_install_dry_run_re_run_keeps_saved_choices_and_sees_hf_token(sandbox):
    # Saved Models choices: no --model, which would set them aside. HF_TOKEN
    # counts as signed in, as the doctor counts it. Git through winget is a
    # per-user install, which asks for no administrator approval.
    _nvidia_smi(sandbox["stubs"], "NVIDIA-SMI 580.00   CUDA Version: 13.0", "NVIDIA GeForce RTX 5080, 12.0, 16303")
    _winget(sandbox["stubs"])
    _uv(sandbox["stubs"])
    (sandbox["root"] / "auradream.settings.json").write_text("{}")
    sandbox["env"]["HF_TOKEN"] = "hf_not_a_real_token"
    result = _run(sandbox, "install.ps1", "-DryRun", "-Yes")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "would install Git: winget install --id Git.Git -e --scope user" in out
    assert "a token is set in HF_TOKEN" in out
    assert "--model" not in out
    assert "restores the choices saved in the Models panel" in out
    assert "hf_not_a_real_token" not in out


def test_install_dry_run_without_a_token_still_keeps_saved_choices(sandbox):
    _nvidia_smi(sandbox["stubs"], "NVIDIA-SMI 580.00   CUDA Version: 13.0", "NVIDIA GeForce RTX 5080, 12.0, 16303")
    _uv(sandbox["stubs"])
    _git(sandbox["stubs"])
    (sandbox["root"] / ".git").mkdir()
    (sandbox["root"] / "local-diffusion.settings.json").write_text("{}")   # the pre-rename name
    result = _run(sandbox, "install.ps1", "-DryRun", "-Yes")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "starts on the choices saved in its Models panel" in out
    assert "--model" not in out


@pytest.mark.parametrize("token_file", [False, True])
@pytest.mark.parametrize(("memory", "vram", "gated"), [(64, 16303, True), (32, 16303, False), (64, 8188, False)])
def test_a_present_token_asks_for_the_first_checkpoint_with_a_memory_based_default(sandbox, token_file, memory, vram, gated):
    _nvidia_smi(sandbox["stubs"], "NVIDIA-SMI CUDA Version: 13.0", f"NVIDIA RTX, 12.0, {vram}")
    _git(sandbox["stubs"])
    _uv(sandbox["stubs"])
    (sandbox["root"] / ".git").mkdir()
    if token_file:
        path = sandbox["tmp"] / "token"
        path.write_text("hf_never_read_this")
        sandbox["env"]["HF_TOKEN_PATH"] = str(path)
    else:
        sandbox["env"]["HF_TOKEN"] = "hf_never_print_this"
    script = str(REPO / "install.ps1").replace("'", "''")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
         "function Get-CimInstance { param($ClassName) if ($ClassName -eq 'Win32_ComputerSystem') { "
         f"[pscustomobject]@{{TotalPhysicalMemory={memory}GB}} "
         f"}} }}; & '{script}' -DryRun -Yes"],
        env=sandbox["env"], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Start on FLUX.1 Krea [dev] instead of Animagine XL 4.0?" in result.stdout
    assert "A token does not prove access" in result.stdout
    assert ("--model cagliostrolab/animagine-xl-4.0 --open" in result.stdout) is not gated
    assert "hf_never_read_this" not in result.stdout + result.stderr
    assert "hf_never_print_this" not in result.stdout + result.stderr


def test_install_refuses_while_auradream_is_running(sandbox, auradream_listening):
    _nvidia_smi(sandbox["stubs"], "NVIDIA-SMI 580.00   CUDA Version: 13.0", "NVIDIA GeForce RTX 5080, 12.0, 16303")
    _uv(sandbox["stubs"])
    _git(sandbox["stubs"])
    (sandbox["root"] / ".git").mkdir()
    sandbox["env"]["AURADREAM_PORT"] = str(auradream_listening)
    before = _snapshot(sandbox["root"])
    result = _run(sandbox, "install.ps1", "-Yes")     # a real run, not a dry one
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"AuraDream is running at http://127.0.0.1:{auradream_listening}/" in result.stdout
    assert "Install stopped." in result.stdout
    assert _snapshot(sandbox["root"]) == before
    assert [c for c in _calls(sandbox) if not c.endswith("--version")] == []   # no uv sync, no git


@pytest.mark.parametrize("name", ["install", "update"])
def test_wrapper_survives_being_replaced_while_it_runs(tmp_path, name):
    # git pull (update) or the ZIP conversion (install) can rewrite the .cmd
    # while cmd.exe is running it; the wrapper must not read on in the new
    # file, and must pass the script's exit code through.
    (tmp_path / f"{name}.cmd").write_bytes((REPO / f"{name}.cmd").read_bytes())
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(b"@echo off\r\n" + b"echo REPLACEMENT LINE THAT MUST NOT RUN\r\n" * 20
                            + b"exit /b 77\r\n")
    (tmp_path / f"{name}.ps1").write_text(
        f"Copy-Item -LiteralPath '{replacement}' -Destination '{tmp_path / (name + '.cmd')}' -Force\n"
        "Write-Output 'script ran'\nexit 5\n", encoding="ascii")
    env = dict(os.environ, AURADREAM_INSTALL_NO_PAUSE="1")
    result = subprocess.run(["cmd.exe", "/d", "/c", str(tmp_path / f"{name}.cmd")], env=env,
                            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert "script ran" in result.stdout
    assert "REPLACEMENT" not in result.stdout and not result.stderr.strip(), result.stdout + result.stderr
    assert result.returncode == 5


def test_install_wrapper_explains_a_missing_companion_from_a_zip(tmp_path):
    wrapper = tmp_path / "install.cmd"
    wrapper.write_bytes((REPO / "install.cmd").read_bytes())
    result = subprocess.run(["cmd.exe", "/d", "/c", str(wrapper)],
                            env=dict(os.environ, AURADREAM_INSTALL_NO_PAUSE="1"),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "Extract the whole ZIP first" in result.stdout
    assert "-File" not in result.stderr


@pytest.mark.parametrize("script", ["install.ps1", "update.ps1"])
def test_untrusted_https_is_refused_before_an_install_or_update(sandbox, script):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib name
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<title>AuraDream</title>")

        def log_message(self, *args):
            pass

    certfile, keyfile = loopback_certificate(sandbox["tmp"])
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sandbox["env"].update(AURADREAM_SSL_CERTFILE=str(certfile), AURADREAM_SSL_KEYFILE=str(keyfile),
                          AURADREAM_PORT=str(server.server_address[1]))
    before = _snapshot(sandbox["root"])
    try:
        result = _run(sandbox, script, *(["-Yes"] if script == "install.ps1" else []))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"HTTPS listener at https://127.0.0.1:{server.server_address[1]}/" in result.stdout
    assert "could not be verified" in result.stdout and "Certificate verification stays on" in result.stdout
    assert _snapshot(sandbox["root"]) == before and _calls(sandbox) == []


def test_install_unblocks_only_its_named_powershell_scripts(sandbox):
    _nvidia_smi(sandbox["stubs"], "NVIDIA-SMI CUDA Version: 13.0", "NVIDIA RTX, 12.0, 16303")
    _git(sandbox["stubs"])
    _uv(sandbox["stubs"])
    (sandbox["root"] / ".git").mkdir()
    own = sandbox["root"] / "bin" / "auradream.ps1"
    personal = sandbox["root"] / "personal.ps1"
    for file in (own, personal):
        file.write_text("# inert file\n")
        Path(str(file) + ":Zone.Identifier").write_text("[ZoneTransfer]\nZoneId=3\n")
    dry = _run(sandbox, "install.ps1", "-DryRun", "-Yes")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "Unblock-File" in dry.stdout
    assert Path(str(own) + ":Zone.Identifier").exists()
    result = _run(sandbox, "install.ps1", "-Yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(str(own) + ":Zone.Identifier").exists()
    assert Path(str(personal) + ":Zone.Identifier").exists()


@pytest.fixture(params=["the page", "the sign-in page"])
def auradream_listening(request):
    """A loopback page titled AuraDream on an ephemeral port: the app's own,
    or what a server that asks for a passcode answers a signed-out `/` with
    (`web/passcode.html`, a 200 under the same title), which the scripts'
    title probe must read as AuraDream running. Any other path there is the
    passcode's 401."""
    asks = request.param == "the sign-in page"
    sign_in = (REPO / "web" / "passcode.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            if asks and self.path != "/":
                body = b'{"detail": "This server asks for a passcode.", "code": "passcode_required"}'
                self.send_response(401)
                self.send_header("WWW-Authenticate", "Bearer")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = sign_in if asks else b"<!doctype html><title>AuraDream</title>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _checkout_with_gallery(sandbox, name="auradream.db"):
    (sandbox["root"] / ".git").mkdir()
    (sandbox["root"] / name).write_bytes(b"gallery")
    _git(sandbox["stubs"])
    _uv(sandbox["stubs"])


def test_update_refuses_while_auradream_is_running(sandbox, auradream_listening):
    _checkout_with_gallery(sandbox)
    sandbox["env"]["AURADREAM_PORT"] = str(auradream_listening)
    before = _snapshot(sandbox["root"])
    result = _run(sandbox, "update.ps1")              # a real run, not a dry one
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"AuraDream is running at http://127.0.0.1:{auradream_listening}/" in result.stdout
    assert "Ctrl-C" in result.stdout
    assert _snapshot(sandbox["root"]) == before       # no database copy
    assert _calls(sandbox) == []                      # no git pull, no uv sync


def test_update_dry_run_names_the_legacy_copy(sandbox):
    _checkout_with_gallery(sandbox, "local-diffusion.db")
    sandbox["env"]["AURADREAM_PORT"] = "1"            # nothing listens there
    before = _snapshot(sandbox["root"])
    result = _run(sandbox, "update.ps1", "-DryRun")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "local-diffusion-before-update-abc1234.db" in result.stdout
    assert "would run in" in result.stdout and "git pull --ff-only" in result.stdout
    assert _snapshot(sandbox["root"]) == before
    assert [c for c in _calls(sandbox) if " pull" in c or c.startswith("uv")] == []


def test_update_copies_the_gallery_then_pulls_and_syncs(sandbox):
    _checkout_with_gallery(sandbox)
    stale = sandbox["root"] / "auradream-before-update-abc1234.db-wal"
    stale.write_bytes(b"old journal")
    sandbox["env"]["AURADREAM_PORT"] = "1"
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (sandbox["root"] / "auradream-before-update-abc1234.db").read_bytes() == b"gallery"
    assert not stale.exists()                         # never applied to the new copy
    calls = _calls(sandbox)
    assert any(c.endswith("pull --ff-only") for c in calls) and "uv sync" in calls
    assert "Already up to date (abc1234)." in result.stdout


def test_update_reports_the_old_and_new_commits_and_count(sandbox):
    _checkout_with_gallery(sandbox)
    sandbox["env"]["STUB_AFTER_COMMIT"] = "def5678"
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Updated from abc1234 to def5678 (3 changes)." in result.stdout
    assert (sandbox["root"] / "auradream-before-update-abc1234.db").read_bytes() == b"gallery"


def test_update_names_a_blocking_process_command_line(sandbox):
    executable = str(sandbox["root"] / ".venv" / "Scripts" / "python.exe").replace("'", "''")
    script = str(REPO / "update.ps1").replace("'", "''")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
         "function Get-CimInstance { param($ClassName) [pscustomobject]@{Name='python.exe';ProcessId=123;"
         f"ExecutablePath='{executable}';CommandLine='python -m backend.generate --seed 42'}} }}; & '{script}'"],
        env=sandbox["env"], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "process 123): python -m backend.generate --seed 42" in result.stdout
    assert _calls(sandbox) == []


def test_update_keeps_the_copy_an_earlier_update_made(sandbox):
    # The second update runs after the new version has changed the database:
    # the copy from before the first pull is still the way back.
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    sandbox["env"]["STUB_COMMIT"] = "aaa1111"
    assert _run(sandbox, "update.ps1").returncode == 0
    (root / "auradream.db").write_bytes(b"migrated")
    sandbox["env"]["STUB_COMMIT"] = "bbb2222"
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update-aaa1111.db").read_bytes() == b"gallery"
    assert (root / "auradream-before-update-bbb2222.db").read_bytes() == b"migrated"
    assert not (root / "auradream-before-update.db").exists()


def test_update_leaves_a_copy_from_the_same_commit_alone(sandbox):
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copy = root / "auradream-before-update-abc1234.db"
    assert _run(sandbox, "update.ps1").returncode == 0
    (copy.parent / (copy.name + "-wal")).write_bytes(b"the copy's journal")
    (root / "auradream.db").write_bytes(b"changed since")
    dry = _run(sandbox, "update.ps1", "-DryRun")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert f"would leave {copy} as it is" in dry.stdout
    result = _run(sandbox, "update.ps1")
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


def test_update_keeps_the_newest_three_copies(sandbox):
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copies = _older_copies(root)
    before = _snapshot(root)
    dry = _run(sandbox, "update.ps1", "-DryRun")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    for commit in ("a2a2a2a", "a1a1a1a"):
        assert f"would delete the older copy {copies[commit]}" in dry.stdout
    assert "a3a3a3a" not in dry.stdout and "a4a4a4a" not in dry.stdout
    assert _snapshot(root) == before
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update-abc1234.db").read_bytes() == b"gallery"
    assert copies["a4a4a4a"].exists() and copies["a3a3a3a"].exists()
    assert not copies["a2a2a2a"].exists() and not copies["a1a1a1a"].exists()
    assert not (root / "auradream-before-update-a1a1a1a.db-wal").exists()
    # Only this database's commit-named copies are counted or deleted.
    assert (root / "auradream-before-update.db").read_bytes() == b"the old fixed name"
    assert (root / "auradream-before-update-notes.db").exists()
    assert (root / "other-before-update-a0a0a0a.db").exists()


def test_update_counts_a_copy_it_left_alone_among_the_three(sandbox):
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copies = _older_copies(root)
    sandbox["env"]["STUB_COMMIT"] = "a1a1a1a"         # the oldest copy is this commit's
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert copies["a1a1a1a"].read_bytes() == b"a1a1a1a"
    assert copies["a4a4a4a"].exists() and copies["a3a3a3a"].exists()
    assert not copies["a2a2a2a"].exists()


def test_update_orders_a_copy_by_when_it_was_made(sandbox):
    # A gallery untouched for years: its copy is still the newest one, never
    # the first deleted because it carries the database's own time.
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    os.utime(root / "auradream.db", (1_000_000_000, 1_000_000_000))
    sandbox["env"]["STUB_COMMIT"] = "aaa1111"
    assert _run(sandbox, "update.ps1").returncode == 0
    planted = []
    for age, commit in enumerate(("a2a2a2a", "a1a1a1a"), start=1):
        path = root / f"auradream-before-update-{commit}.db"
        path.write_bytes(commit.encode())
        os.utime(path, (1_700_000_000 - age * 3600, 1_700_000_000 - age * 3600))
        planted.append(path)
    sandbox["env"]["STUB_COMMIT"] = "bbb2222"
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update-aaa1111.db").read_bytes() == b"gallery"
    assert (root / "auradream-before-update-bbb2222.db").exists()
    assert planted[0].exists() and not planted[1].exists()


def test_update_without_a_commit_uses_the_fixed_name(sandbox):
    # Git answers no commit (an unborn branch): the one name, replaced.
    _checkout_with_gallery(sandbox)
    root = sandbox["root"]
    copies = _older_copies(root)
    sandbox["env"]["STUB_COMMIT"] = ""
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "auradream-before-update.db").read_bytes() == b"gallery"
    assert all(path.exists() for path in copies.values())   # nothing counted, nothing deleted


def test_update_explains_a_refused_pull_and_discards_nothing(sandbox):
    _checkout_with_gallery(sandbox)
    _git(sandbox["stubs"], pull_exit=128, pull_output="fatal: Not possible to fast-forward, aborting.")
    sandbox["env"]["AURADREAM_PORT"] = "1"
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 1
    assert "commits of its own" in result.stdout and "Nothing was discarded" in result.stdout
    calls = _calls(sandbox)
    assert not any(word in c for c in calls for word in ("reset", "checkout", "clean", "stash"))
    assert "uv sync" not in calls


def test_update_explains_untracked_files_in_the_way(sandbox):
    # Untracked files (a ZIP folder's leftovers) are not restored by git
    # restore: the advice is to move them.
    _checkout_with_gallery(sandbox)
    _git(sandbox["stubs"], pull_exit=1,
         pull_output="error: The following untracked working tree files would be overwritten by merge:")
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 1
    assert "not part of the code" in result.stdout and "Move or rename them" in result.stdout
    assert "git restore" not in result.stdout
    assert "uv sync" not in _calls(sandbox)
