"""linux/setup.sh, install-service.sh and update.sh, run for real by bash in
the stub sandbox `tests/linux_script_helpers.py` describes (Linux only; the
static checks run everywhere). control.sh's tests are
`test_linux_control.py`, and the passcode's `test_linux_passcode.py`.
"""
from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
from pathlib import Path

import pytest

from tests.linux_script_helpers import (  # noqa: F401  (fixtures by name)
    BASH,
    LINUX,
    REPO,
    SCRIPTS,
    SECRET,
    on_linux,
    _asking,
    _calls,
    _changes,
    _checkout_with_gallery,
    _env_file,
    _no_git_or_uv,
    _run,
    _service_active,
    _snapshot,
    _systemctl,
    _unit,
    _venv,
    auradream_listening,
    sandbox,
)


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
                 "RestartPreventExitStatus=78",   # main.SETTINGS_REFUSED_STATUS
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
@pytest.mark.parametrize("auradream_listening", ["https"], indirect=True)
@pytest.mark.parametrize("probe", ["venv-python", "curl"])
@pytest.mark.parametrize("trusted", [False, True])
def test_https_update_probe_preserves_verification(sandbox, auradream_listening, probe, trusted):
    _checkout_with_gallery(sandbox)
    if probe == "venv-python":
        _venv(sandbox)
    elif not shutil.which("curl", path="/usr/bin:/bin"):
        pytest.skip("no curl here")
    if not trusted:
        for key in ("SSL_CERT_FILE", "CURL_CA_BUNDLE"):
            sandbox["env"].pop(key, None)
    result = _run(sandbox, "update.sh")
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"https://127.0.0.1:{auradream_listening['port']}/" in result.stdout
    assert ("AuraDream is running at" in result.stdout) is trusted
    assert ("could not be verified" in result.stdout) is not trusted
    assert _no_git_or_uv(sandbox)
    assert not list(sandbox["root"].glob("*-before-update*.db"))


@on_linux
@pytest.mark.parametrize("auradream_listening", ["https"], indirect=True)
def test_https_service_status_and_idle_check_send_the_passcode_only_after_verification(sandbox, auradream_listening):
    _asking(sandbox, auradream_listening)
    status = _run(sandbox, "control.sh", "status")
    assert status.returncode == 0, status.stdout + status.stderr
    assert "== The server at https://127.0.0.1:" in status.stdout and "It answers." in status.stdout
    assert auradream_listening["gate"]["sent"] == 1
    sandbox["env"].pop("SSL_CERT_FILE")
    stopped = _run(sandbox, "control.sh", "stop")
    assert stopped.returncode == 1, stopped.stdout + stopped.stderr
    assert "could not be asked at https://127.0.0.1:" in stopped.stdout
    assert "CERTIFICATE_VERIFY_FAILED" in stopped.stdout
    assert auradream_listening["gate"]["sent"] == 1
    assert _changes(sandbox) == []
    assert SECRET not in status.stdout + status.stderr + stopped.stdout + stopped.stderr


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
