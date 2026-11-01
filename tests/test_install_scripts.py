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
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

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
        f'if "%~3"=="pull" (echo {pull_output}& exit /b {pull_exit})',
        'if "%~3"=="remote" (echo https://github.com/matt439/AuraDream.git& exit /b 0)',
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
    assert "8 GB of video memory" in out and "Only a 16 GB card has been measured" in out
    assert "would install uv: winget install --id astral-sh.uv -e --accept-source-agreements" in out
    assert "is a Git checkout (branch main" in out
    assert "[stop]" not in out and "Dry run finished. Nothing was changed." in out
    assert not [c for c in _calls(sandbox) if c.startswith(("winget", "uv"))]   # winget never ran
    assert not list(sandbox["home"].rglob("*.lnk"))


@pytest.fixture
def auradream_listening():
    """A loopback page titled AuraDream on an ephemeral port."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            body = b"<!doctype html><title>AuraDream</title>"
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
    assert "local-diffusion-before-update.db" in result.stdout
    assert "would run in" in result.stdout and "git pull --ff-only" in result.stdout
    assert _snapshot(sandbox["root"]) == before
    assert [c for c in _calls(sandbox) if " pull" in c or c.startswith("uv")] == []


def test_update_copies_the_gallery_then_pulls_and_syncs(sandbox):
    _checkout_with_gallery(sandbox)
    stale = sandbox["root"] / "auradream-before-update.db-wal"
    stale.write_bytes(b"old journal")
    sandbox["env"]["AURADREAM_PORT"] = "1"
    result = _run(sandbox, "update.ps1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (sandbox["root"] / "auradream-before-update.db").read_bytes() == b"gallery"
    assert not stale.exists()                         # never applied to the new copy
    calls = _calls(sandbox)
    assert any(c.endswith("pull --ff-only") for c in calls) and "uv sync" in calls


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
