"""Cold external-attention diagnostics; no framework import or backend policy."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

WARNING = (
    "DIFFUSERS_ATTN_BACKEND differs from Diffusers' native default. "
    "This external setting is not recorded as an AuraDream execution recipe."
)
KEY = "DIFFUSERS_ATTN_BACKEND"
ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import importlib.abc
import json
import os
import sys

class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "diffusers"}:
            raise AssertionError("diagnostics must stay cold: " + fullname)
        return None

sys.meta_path.insert(0, ColdOnly())
key = "DIFFUSERS_ATTN_BACKEND"
before_present, before_value = key in os.environ, os.environ.get(key)
mode = sys.argv[1]
if mode == "server":
    from backend import launch
    launch.configure(["--defaults", "--model", "example/diagnostic", "--quant", "int8", "--offload", "sequential"])
    assert "backend.config" not in sys.modules
    from backend import config
elif mode == "generate":
    from backend import generate
    assert "backend.config" not in sys.modules
    class ProjectReached(Exception):
        pass
    def project():
        from backend import config
        raise ProjectReached(config)
    generate._project = project
    try:
        generate.main(["diagnostic prompt", "--dry-run", "--ignore-server", "--model", "example/diagnostic",
                       "--quant", "int8", "--offload", "sequential"])
    except ProjectReached as stopped:
        config = stopped.args[0]
    else:
        raise AssertionError("the CLI did not reach its normal config boundary")
else:
    from backend import config

assert before_present == (key in os.environ) and before_value == os.environ.get(key)
warning = config.ATTENTION_ENVIRONMENT_WARNING
# A later environment edit cannot relabel an import-time dispatcher snapshot.
os.environ[key] = "post-import-change"
assert config.ATTENTION_ENVIRONMENT_WARNING == warning
if before_present:
    os.environ[key] = before_value
else:
    del os.environ[key]
assert not any(name.split(".", 1)[0] in {"torch", "diffusers"} for name in sys.modules)
print(json.dumps({"warning": warning, "cold": True, "environment_unchanged": True,
                  "selection": [config.settings.model_id, config.settings.quantization, config.settings.offload]}))
'''


def cold_child(tmp_path: Path, value: str | None, mode: str = "config"):
    environment = {name: item for name, item in os.environ.items() if not name.startswith("AURADREAM_")}
    environment.pop(KEY, None)
    if value is not None:
        environment[KEY] = value
    environment.update({"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
                        "AURADREAM_DB_PATH": str(tmp_path / "never-created.db"),
                        "AURADREAM_OUTPUTS_DIR": str(tmp_path / "never-created-outputs")})
    result = subprocess.run([sys.executable, "-B", "-c", CHILD, mode], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    (tmp_path / "cold.stdout").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "cold.stderr").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert not (tmp_path / "never-created.db").exists()
    assert not (tmp_path / "never-created-outputs").exists()
    return json.loads(result.stdout.strip().splitlines()[-1]), result.stderr


def test_installed_diffusers_uses_verbatim_getenv_with_native_only_for_absence():
    # Read the pinned source, never import Diffusers (its import graph can query
    # or initialize devices). A library semantic change needs a fresh audit.
    source = distribution("diffusers").locate_file("diffusers/utils/constants.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == KEY for target in node.targets))
    call = assignment.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert isinstance(call.func.value, ast.Name) and call.func.value.id == "os" and call.func.attr == "getenv"
    assert [ast.literal_eval(argument) for argument in call.args] == [KEY, "native"]
    assert call.keywords == []


@pytest.mark.parametrize("value, warned", [
    (None, False), ("", True), ("native", False), ("NATIVE", True),
    ("_native_cudnn", True), (" native ", True), ("private-value-<>\n" * 1000, True),
], ids=["absent", "empty", "native", "uppercase", "cudnn", "whitespace", "hostile-long"])
def test_cold_warning_matches_snapshot_without_echo_or_environment_mutation(tmp_path, value, warned):
    receipt, stderr = cold_child(tmp_path, value)
    assert receipt["cold"] and receipt["environment_unchanged"]
    assert receipt["warning"] == (WARNING if warned else None)
    assert stderr.count(WARNING) == int(warned)
    assert stderr == (WARNING + "\n" if warned else "")
    assert len(stderr) < 512
    assert "private-value" not in repr(receipt) + stderr


@pytest.mark.parametrize("mode", ["server", "generate"])
def test_cli_warning_uses_normal_logging_after_selection_and_remains_cold(tmp_path, mode):
    receipt, stderr = cold_child(tmp_path, "_native_cudnn", mode)
    assert receipt["warning"] == WARNING and stderr.count(WARNING) == 1
    assert receipt["selection"] == ["example/diagnostic", "int8", "sequential"]
    assert receipt["cold"] and receipt["environment_unchanged"]


@pytest.mark.parametrize("warning", [None, WARNING], ids=["default", "external"])
def test_status_exposes_fixed_warning_without_changing_selection_or_rereading_env(client, monkeypatch, warning):
    from backend import main, model_selection
    from backend.config import settings

    before = model_selection.current_settings(settings)
    overrides = dict(main._startup_overrides)
    monkeypatch.setattr(main, "ATTENTION_ENVIRONMENT_WARNING", warning)
    monkeypatch.setenv(KEY, "private-value-must-not-be-read-by-status")
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["attention_environment_warning"] == warning
    assert "private-value" not in response.text
    assert model_selection.current_settings(settings) == before
    assert main._startup_overrides == overrides
