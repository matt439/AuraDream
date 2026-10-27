"""The type gate cannot pass by checking an empty scope or another environment."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import typecheck


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "one.py").write_text("value: int = 1\n", encoding="utf-8")
    config = tmp_path / "pyrightconfig.json"
    config.write_text(json.dumps({"include": ["source"]}), encoding="utf-8")
    bundled = tmp_path / "pyright" / "dist"
    bundled.mkdir(parents=True)
    (bundled / "index.js").write_text("// synthetic bundled checker", encoding="utf-8")
    (bundled / "package.json").write_text(json.dumps({"version": typecheck.PYRIGHT_VERSION}), encoding="utf-8")
    node = tmp_path / "node.exe"
    node.write_bytes(b"mocked Node runtime")
    monkeypatch.setattr(typecheck.shutil, "which", lambda name: str(node))
    monkeypatch.setattr(typecheck.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(bundled.parent / "__init__.py")))
    monkeypatch.setattr(typecheck.importlib.metadata, "version", lambda name: typecheck.PYRIGHT_VERSION)
    return config


def _checker(monkeypatch, *, code=0, report=None, version=None):
    calls = []
    if report is None:
        report = {"generalDiagnostics": [], "summary": {"filesAnalyzed": 1, "errorCount": 0}}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[:3] == [sys.executable, "-m", "pyright"]
        if command[3:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, version or f"pyright {typecheck.PYRIGHT_VERSION}\n", "")
        return subprocess.CompletedProcess(command, code, json.dumps(report) if not isinstance(report, str) else report, "checker stderr\n")

    monkeypatch.setattr(typecheck.subprocess, "run", run)
    return calls


def test_missing_node_never_invokes_the_wrapper(project, monkeypatch, capsys):
    calls = _checker(monkeypatch)
    monkeypatch.setattr(typecheck.shutil, "which", lambda name: None)
    assert typecheck.main(project) == 2
    assert "Node is required" in capsys.readouterr().err
    assert calls == []


def test_missing_checker_never_invokes_an_installer(project, monkeypatch, capsys):
    calls = _checker(monkeypatch)
    monkeypatch.setattr(typecheck.importlib.util, "find_spec", lambda name: None)
    assert typecheck.main(project) == 2
    assert "not installed in this Python environment" in capsys.readouterr().err
    assert calls == []


def test_windows_command_shim_cannot_trigger_node_provisioning(project, monkeypatch):
    calls = _checker(monkeypatch)
    shim = project.parent / "node.cmd"
    shim.write_text("@echo mock", encoding="utf-8")
    queried = []

    def which(name):
        queried.append(name)
        return str(shim) if name == "node" else None

    monkeypatch.setattr(typecheck.sys, "platform", "win32")
    monkeypatch.setattr(typecheck.shutil, "which", which)
    assert typecheck.main(project) == 2
    assert queried == ["node.exe"] and calls == []


def test_node_must_be_a_file_before_the_wrapper_runs(project, monkeypatch):
    calls = _checker(monkeypatch)
    directory = project.parent / "node-directory"
    directory.mkdir()
    monkeypatch.setattr(typecheck.shutil, "which", lambda name: str(directory))
    assert typecheck.main(project) == 2
    assert calls == []


def test_wrong_version_fails_before_checking(project, monkeypatch, capsys):
    calls = _checker(monkeypatch, version="pyright 1.0.0\n")
    assert typecheck.main(project) == 2
    assert len(calls) == 1
    assert f"requires Pyright {typecheck.PYRIGHT_VERSION}" in capsys.readouterr().err


def test_wrong_python_package_version_cannot_select_an_npm_download(project, monkeypatch):
    calls = _checker(monkeypatch)
    monkeypatch.setattr(typecheck.importlib.metadata, "version", lambda name: "1.0.0")
    assert typecheck.main(project) == 2
    assert calls == []


@pytest.mark.parametrize("member", ["index.js", "package.json"])
def test_missing_bundled_checker_cannot_trigger_installation(project, monkeypatch, member):
    calls = _checker(monkeypatch)
    (project.parent / "pyright" / "dist" / member).unlink()
    assert typecheck.main(project) == 2
    assert calls == []


def test_wrong_bundled_version_cannot_trigger_installation(project, monkeypatch):
    calls = _checker(monkeypatch)
    (project.parent / "pyright" / "dist" / "package.json").write_text('{"version": "1.0.0"}')
    assert typecheck.main(project) == 2
    assert calls == []


def test_malformed_bundled_metadata_cannot_trigger_installation(project, monkeypatch):
    calls = _checker(monkeypatch)
    (project.parent / "pyright" / "dist" / "package.json").write_text("[]")
    assert typecheck.main(project) == 2
    assert calls == []


@pytest.mark.parametrize("config", [{}, {"include": []}, {"include": [""]}, {"include": [None]}, {"include": ["missing"]}])
def test_missing_or_empty_configured_scope_fails(project, monkeypatch, config, capsys):
    calls = _checker(monkeypatch)
    project.write_text(json.dumps(config), encoding="utf-8")
    assert typecheck.main(project) == 2
    assert "include" in capsys.readouterr().err
    assert calls == []


def test_existing_but_empty_include_directory_fails(project, monkeypatch):
    calls = _checker(monkeypatch)
    (project.parent / "source" / "one.py").unlink()
    assert typecheck.main(project) == 2
    assert calls == []


@pytest.mark.parametrize("report", [
    "not JSON", {}, {"summary": {}, "generalDiagnostics": []},
    {"summary": {"filesAnalyzed": 0, "errorCount": 0}, "generalDiagnostics": []},
    {"summary": {"filesAnalyzed": "1", "errorCount": 0}, "generalDiagnostics": []},
    {"summary": {"filesAnalyzed": 1, "errorCount": 0}, "generalDiagnostics": "invalid"},
])
def test_malformed_or_actual_empty_checker_output_cannot_pass(project, monkeypatch, report, capsys):
    _checker(monkeypatch, report=report)
    assert typecheck.main(project) == 2
    assert "malformed output or analyzed no files" in capsys.readouterr().err


@pytest.mark.parametrize("code", [1, 42])
def test_failure_exit_and_diagnostics_are_preserved(project, monkeypatch, capsys, code):
    report = {"generalDiagnostics": [{"message": "source/one.py: invalid argument"}],
              "summary": {"filesAnalyzed": 1, "errorCount": 1}}
    _checker(monkeypatch, code=code, report=report)
    assert typecheck.main(project) == code
    output = capsys.readouterr()
    assert "source/one.py: invalid argument" in output.out
    assert "checker stderr" in output.err


def test_malformed_failure_still_preserves_the_checker_exit(project, monkeypatch):
    _checker(monkeypatch, code=7, report="launcher failed")
    assert typecheck.main(project) == 7


def test_success_uses_the_same_python_and_explicit_config(project, monkeypatch):
    calls = _checker(monkeypatch)
    assert typecheck.main(project) == 0
    assert len(calls) == 2
    command, options = calls[-1]
    assert command == [sys.executable, "-m", "pyright", "--project", str(project), "--outputjson"]
    assert options["cwd"] == project.parent


def test_wrapper_overrides_cannot_request_updates_or_downloads(project, monkeypatch):
    for name, value in {
        "PYRIGHT_PYTHON_FORCE_VERSION": "latest", "PYRIGHT_PYTHON_PYLANCE_VERSION": "latest",
        "PYRIGHT_PYTHON_USE_BUNDLED_PYRIGHT": "0", "PYRIGHT_PYTHON_IGNORE_WARNINGS": "0",
        "PYRIGHT_PYTHON_GLOBAL_NODE": "0", "PYRIGHT_PYTHON_NODEJS_WHEEL": "1",
    }.items():
        monkeypatch.setenv(name, value)
    calls = _checker(monkeypatch)
    assert typecheck.main(project) == 0
    for _, options in calls:
        env = options["env"]
        assert env["PYRIGHT_PYTHON_FORCE_VERSION"] == typecheck.PYRIGHT_VERSION
        assert env["PYRIGHT_PYTHON_USE_BUNDLED_PYRIGHT"] == "1"
        assert env["PYRIGHT_PYTHON_IGNORE_WARNINGS"] == "1"
        assert env["PYRIGHT_PYTHON_GLOBAL_NODE"] == "1"
        assert env["PYRIGHT_PYTHON_NODEJS_WHEEL"] == "0"
        assert "PYRIGHT_PYTHON_PYLANCE_VERSION" not in env


def test_reported_errors_cannot_pass_with_a_zero_checker_status(project, monkeypatch):
    _checker(monkeypatch, report={"generalDiagnostics": [], "summary": {"filesAnalyzed": 1, "errorCount": 1}})
    assert typecheck.main(project) == 1


def test_launch_failure_is_clear(project, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("checker could not launch")

    monkeypatch.setattr(typecheck.subprocess, "run", fail)
    assert typecheck.main(project) == 2
    assert "checker could not launch" in capsys.readouterr().err
