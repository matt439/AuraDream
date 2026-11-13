"""Startup refusals survive the shortcut window, using scratch files only.

The batch tests run its real cmd.exe control flow against Python renamed to
uv.exe: Python executes our inert `run` file instead of any app or uv code.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from backend import log_file, main

ROOT = Path(__file__).resolve().parents[1]


def _never(*args, **kwargs):
    pytest.fail("a refused startup reached a server or a network probe")


@pytest.mark.parametrize("level", ["INFO", "CRITICAL"])
@pytest.mark.parametrize("reason", ["passcode", "port"])
def test_a_startup_refusal_is_printed_once_and_kept_even_at_critical(fake, monkeypatch, capsys, level, reason):
    secret = "p7-Qx"
    monkeypatch.setattr(main, "settings", dataclasses.replace(
        main.settings, port=7860, log_level=level, passcode=secret if reason == "passcode" else None,
    ))
    monkeypatch.setattr(main, "Server", _never)
    if reason == "passcode":
        monkeypatch.setattr(main, "_port_busy", _never)
        monkeypatch.setattr(main, "_auradream_at", _never)
        expected = "AURADREAM_PASSCODE is shorter than 8 characters"
    else:
        monkeypatch.setattr(main, "_port_busy", lambda host, port: True)
        monkeypatch.setattr(main, "_auradream_at", lambda url: False)
        expected = "Port 7860 is in use by another program"
    root = logging.getLogger()
    handlers = list(root.handlers)
    levels = [handler.level for handler in handlers]
    with pytest.raises(SystemExit) as stopped:
        main.run()
    # A settings refusal is EX_CONFIG, which the Linux unit does not restart;
    # a held port can clear, so it stays 1.
    assert stopped.value.code == (main.SETTINGS_REFUSED_STATUS if reason == "passcode" else 1)
    said = capsys.readouterr()
    assert said.err.count(expected) == 1 and said.out == ""
    path = log_file.path_for(main.settings.db_path)
    written = path.read_text(encoding="utf-8")
    assert written.count(expected) == 1 and "ERROR" in written
    assert secret not in written + said.out + said.err
    assert root.handlers == handlers and [handler.level for handler in handlers] == levels
    assert not any(isinstance(handler, RotatingFileHandler) for handler in root.handlers)
    path.unlink()  # Windows proves the refusal closed its handle.


def test_a_duplicate_launch_leaves_an_existing_log_and_its_backups_alone(fake, monkeypatch, capsys):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, port=7860))
    monkeypatch.setattr(main, "_port_busy", lambda host, port: True)
    monkeypatch.setattr(main, "_auradream_at", lambda url: True)
    monkeypatch.setattr(main, "Server", _never)
    path = log_file.path_for(main.settings.db_path)
    previous = Path(f"{path}.1")
    path.write_bytes(b"running server log\n")
    previous.write_bytes(b"previous log\n")
    monkeypatch.setattr(main.log_file, "install", _never)
    with pytest.raises(SystemExit) as stopped:
        main.run()
    assert stopped.value.code == 0
    assert "AuraDream is already running" in capsys.readouterr().out
    assert path.read_bytes() == b"running server log\n" and previous.read_bytes() == b"previous log\n"


def test_an_unwritable_startup_log_does_not_hide_the_original_refusal(fake, monkeypatch, capsys):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, port=7860))
    monkeypatch.setattr(main, "_port_busy", lambda host, port: True)
    monkeypatch.setattr(main, "_auradream_at", lambda url: False)
    monkeypatch.setattr(main, "Server", _never)

    def cannot_open(*args, **kwargs):
        raise PermissionError("read-only scratch directory")

    monkeypatch.setattr(log_file, "_Handler", cannot_open)
    with pytest.raises(SystemExit) as stopped:
        main.run()
    assert stopped.value.code == 1
    assert "Port 7860 is in use" in capsys.readouterr().err
    assert not log_file.path_for(main.settings.db_path).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher")
@pytest.mark.parametrize(("code", "pauses"), [(0, False), (1, True), (7, True), (130, False), (-1073741510, False)])
@pytest.mark.parametrize("tool_location", ["path", "user", "winget"])
def test_the_launcher_keeps_errors_visible_and_preserves_exit_and_arguments(tmp_path, code, pauses, tool_location):
    project = tmp_path / "AuraDream with spaces"
    (project / "bin").mkdir(parents=True)
    launcher = project / "bin" / "auradream.cmd"
    launcher.write_bytes((ROOT / "bin" / "auradream.cmd").read_bytes())
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    # An executable named uv runs the inert `run` file at the project root.
    # PYTHONHOME makes this copied interpreter use its real standard library.
    user_home = tmp_path / "user home"
    local_data = user_home / "Local"
    if tool_location == "path":
        tool_dir = stubs
    elif tool_location == "user":
        tool_dir = user_home / ".local" / "bin"
    else:
        tool_dir = local_data / "Microsoft" / "WinGet" / "Links"
    tool_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys._base_executable, tool_dir / "uv.exe")
    (project / "run").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['STUB_LOG']).write_text(json.dumps([os.getcwd(), sys.argv[1:]]))\n"
        "print('inert uv finished', flush=True)\n"
        "sys.exit(int(os.environ['STUB_EXIT']))\n",
        encoding="utf-8",
    )
    calls = tmp_path / "calls.json"
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    cmd = str(Path(system_root) / "System32" / "cmd.exe")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("PYTHON", "AURADREAM_", "HF_", "UV_"))
        and key.upper() != "PATH"
    }
    environment.update({
        "PATH": os.pathsep.join([str(stubs), sys.base_prefix, str(Path(system_root) / "System32")]),
        "PYTHONHOME": sys.base_prefix,
        "USERPROFILE": str(user_home), "LOCALAPPDATA": str(local_data),
        "STUB_LOG": str(calls), "STUB_EXIT": str(code),
    })
    result = subprocess.run(
        f'"{cmd}" /d /s /c ""{launcher}" --model stub/model --open"',
        cwd=tmp_path, env=environment, input="\n", capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == code % (2 ** 32), result.stdout + result.stderr
    directory, arguments = json.loads(calls.read_text())
    assert Path(directory) == project
    assert arguments == ["python", "-m", "backend.main", "--model", "stub/model", "--open"]
    assert "inert uv finished" in result.stdout
    assert ("Press any key to close this window." in result.stdout) == pauses
    assert ("Read the message above." in result.stdout) == pauses


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell launcher")
@pytest.mark.parametrize("code", [0, 7])
@pytest.mark.parametrize("tool_location", ["path", "user", "winget", "missing"])
def test_the_powershell_launcher_finds_uv_as_its_cmd_does(tmp_path, code, tool_location):
    # DEV#67: auradream.ps1, which PowerShell picks over the .cmd, ran a bare
    # `uv` and exited 0 when there was none.
    project = tmp_path / "AuraDream with spaces"
    (project / "bin").mkdir(parents=True)
    launcher = project / "bin" / "auradream.ps1"
    launcher.write_bytes((ROOT / "bin" / "auradream.ps1").read_bytes())
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    user_home = tmp_path / "user home"
    local_data = user_home / "Local"
    tool_dir = {"path": stubs, "user": user_home / ".local" / "bin",
                "winget": local_data / "Microsoft" / "WinGet" / "Links", "missing": None}[tool_location]
    if tool_dir is not None:
        tool_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sys._base_executable, tool_dir / "uv.exe")
    (project / "run").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['STUB_LOG']).write_text(json.dumps([os.getcwd(), sys.argv[1:]]))\n"
        "print('inert uv finished', flush=True)\n"
        "sys.exit(int(os.environ['STUB_EXIT']))\n",
        encoding="utf-8",
    )
    calls = tmp_path / "calls.json"
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    powershell = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("PYTHON", "AURADREAM_", "HF_", "UV_"))
        and key.upper() != "PATH"
    }
    environment.update({
        "PATH": os.pathsep.join([str(stubs), sys.base_prefix, str(Path(system_root) / "System32")]),
        "PYTHONHOME": sys.base_prefix,
        "USERPROFILE": str(user_home), "LOCALAPPDATA": str(local_data),
        "STUB_LOG": str(calls), "STUB_EXIT": str(code),
    })
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(launcher), "--model", "stub/model", "--open"],
        cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=60,
    )
    if tool_location == "missing":
        assert result.returncode == 1, result.stdout + result.stderr
        assert "uv was not found. Run install.cmd" in result.stdout
        assert "CommandNotFoundException" not in result.stdout + result.stderr
        assert not calls.exists()
        return
    assert result.returncode == code, result.stdout + result.stderr
    directory, arguments = json.loads(calls.read_text())
    assert Path(directory) == project
    assert arguments == ["python", "-m", "backend.main", "--model", "stub/model", "--open"]
    assert "inert uv finished" in result.stdout
