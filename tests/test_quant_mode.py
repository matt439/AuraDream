"""KREA_QUANT=nunchaku: nothing is quantized at load, and the mode refuses early.

A Nunchaku Lite checkpoint arrives pre-quantized and names its quantizer in
`transformer/config.json`; a torchao config on top of that would be wrong, and
pointing the mode at a bf16 repo would download 25 GB and fail at the first
image. Both are decided before any weights move, so both run without a GPU.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import pytest

from backend.backends import diffusers_backend as module
from backend.backends.diffusers_backend import DiffusersBackend

MODEL = "lite-infer/some-checkpoint"
PREQUANTIZED = {"quantization_config": {"quant_method": "nunchaku_lite"}}


@pytest.fixture
def nunchaku(monkeypatch, tmp_path):
    """The happy path: Linux, both packages present, a pre-quantized repo.

    Returns the stub so a test can swap in another config.json.
    """
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(module.settings, quantization="nunchaku", model_id=MODEL),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(module, "_importable", lambda name: True)

    def hub_file(model_id, filename, config=PREQUANTIZED):
        assert (model_id, filename) == (MODEL, "transformer/config.json")
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(module, "_hub_file", hub_file)
    return hub_file


def test_prequantized_repo_gets_no_torchao_config(nunchaku):
    assert DiffusersBackend()._quantization_config() is None


def test_fp8_refuses_a_prequantized_repo_before_the_download(monkeypatch, tmp_path):
    # The mirror image of the Nunchaku preflight: `KREA_QUANT=fp8` on a lite
    # repo used to download ~9 GB and die inside transformers.
    monkeypatch.setattr(
        module, "settings", dataclasses.replace(module.settings, quantization="fp8", model_id=MODEL)
    )
    files = {
        "model_index.json": {"transformer": ["diffusers", "Krea2Transformer2DModel"]},
        "transformer/config.json": PREQUANTIZED,
    }

    def hub_file(model_id, filename):
        assert model_id == MODEL
        path = tmp_path / filename.replace("/", "_")
        path.write_text(json.dumps(files[filename]), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(module, "_hub_file", hub_file)
    with pytest.raises(RuntimeError, match="already quantized"):
        DiffusersBackend()._quantization_config()


def test_refuses_off_linux(nunchaku, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="WSL 2"):
        DiffusersBackend()._quantization_config()


def test_refuses_without_the_packages(nunchaku, monkeypatch):
    monkeypatch.setattr(module, "_importable", lambda name: name != "kernels")
    with pytest.raises(RuntimeError, match="kernels"):
        DiffusersBackend()._quantization_config()


def test_refuses_a_bf16_repo(nunchaku, monkeypatch):
    def plain(model_id, filename):
        return nunchaku(model_id, filename, config={"_class_name": "Krea2Transformer2DModel"})

    monkeypatch.setattr(module, "_hub_file", plain)
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite checkpoint") as err:
        DiffusersBackend()._quantization_config()
    assert MODEL in str(err.value)


_IMPORT_SETTINGS = (
    "import os; from backend.config import settings; "
    "print(settings.outputs_dir); print(settings.db_path); "
    "print(os.environ.get('DIFFUSERS_TRUST_REMOTE_KERNELS', ''))"
)


def _run_import(env: dict[str, str]) -> subprocess.CompletedProcess:
    # `settings` reads the environment at import, so a fresh interpreter is
    # the only honest way to test the variables.
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("KREA_", "DIFFUSERS_"))}
    return subprocess.run(
        [sys.executable, "-c", _IMPORT_SETTINGS],
        env={**clean, **env},
        capture_output=True,
        text=True,
    )


def _settings_in_subprocess(env: dict[str, str]) -> dict[str, str]:
    result = _run_import(env)
    assert result.returncode == 0, result.stderr
    out = result.stdout.splitlines()
    return {"outputs_dir": out[0], "db_path": out[1], "trust": out[2]}


def test_data_paths_come_from_the_environment(tmp_path):
    got = _settings_in_subprocess(
        {"KREA_OUTPUTS_DIR": str(tmp_path / "out"), "KREA_DB_PATH": str(tmp_path / "g.db")}
    )
    assert got["outputs_dir"] == str(tmp_path / "out")
    assert got["db_path"] == str(tmp_path / "g.db")
    # Created by storage when something is about to be written, not at import:
    # `clear.py --dry-run` promises to change nothing.
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "env, name",
    [({"KREA_QUANT": "fp16"}, "KREA_QUANT"), ({"KREA_OFFLOAD": "cpu"}, "KREA_OFFLOAD")],
)
def test_an_unknown_mode_is_refused_at_import(tmp_path, env, name):
    # Either used to select the heaviest path silently and fail at the first
    # image, with /api/status echoing the typo back.
    base = {"KREA_OUTPUTS_DIR": str(tmp_path / "out"), "KREA_DB_PATH": str(tmp_path / "g.db")}
    result = _run_import({**base, **env})
    assert result.returncode != 0
    assert name in result.stderr and "is not one of" in result.stderr


def test_nunchaku_opts_into_remote_kernels_and_nothing_else_does(tmp_path):
    base = {"KREA_OUTPUTS_DIR": str(tmp_path / "out"), "KREA_DB_PATH": str(tmp_path / "g.db")}
    assert _settings_in_subprocess({**base, "KREA_QUANT": "nunchaku"})["trust"] == "true"
    assert _settings_in_subprocess({**base, "KREA_QUANT": "fp8"})["trust"] == ""
    # An explicit refusal in the environment is respected.
    explicit = {**base, "KREA_QUANT": "nunchaku", "DIFFUSERS_TRUST_REMOTE_KERNELS": "false"}
    assert _settings_in_subprocess(explicit)["trust"] == "false"
