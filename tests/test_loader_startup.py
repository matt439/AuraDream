"""Cold loader preferences/configuration; no framework, compiler or model use."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend import config, launch, startup_preferences as preferences

MODEL = "black-forest-labs/FLUX.1-Krea-dev"
OTHER = "krea/Krea-2-Turbo"
VARIABLES = ("AURADREAM_MODEL_ID", "AURADREAM_QUANT", "AURADREAM_OFFLOAD",
             "AURADREAM_PRELOAD", "AURADREAM_COMPILATION", "AURADREAM_ATTENTION",
             preferences.OVERRIDES_VARIABLE, preferences.LOADER_EXECUTION_ERROR_VARIABLE)


@pytest.fixture(autouse=True)
def clean_options(monkeypatch, tmp_path):
    for name in VARIABLES:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("AURADREAM_DB_PATH", str(tmp_path / "gallery.db"))


def selected(recipe=None, *, preload=False):
    return preferences.StartupPreferences(MODEL, "fp8", "model", preload, recipe)


def database():
    return Path(os.environ["AURADREAM_DB_PATH"])


def test_valid_v1_is_eager_and_reads_without_upgrade_or_new_environment_keys():
    path = preferences.path_for(database())
    original = {"version": 1, "model_id": MODEL, "quantization": "fp8", "offload": "model", "load_on_startup": True}
    path.write_text(json.dumps(original), encoding="utf-8")
    before = path.read_bytes()
    loaded = preferences.load_result(database())
    assert loaded.preferences == selected(preload=True) and loaded.loader_execution_error is None
    assert set(loaded.preferences.environ()) == set(VARIABLES[:4])
    assert path.read_bytes() == before and not database().exists()


@pytest.mark.parametrize("field", ["loader_execution", "execution", "compilation", "attention_backend", "step_cache"])
def test_v1_envelope_with_explicit_execution_intent_cannot_silently_fall_back(field):
    payload = {"version": 1, "model_id": MODEL, "quantization": "fp8", "offload": "model", "load_on_startup": True,
               field: {"compilation": "regional-v1"}}
    preferences.path_for(database()).write_text(json.dumps(payload), encoding="utf-8")
    result = preferences.load_result(database())
    assert result.preferences is None and result.loader_execution_error == preferences.LOADER_EXECUTION_ERROR


def test_unsupported_saved_nunchaku_cannot_discard_a_nonempty_recipe(monkeypatch):
    choice = preferences.StartupPreferences("lite-infer/synthetic-model", "nunchaku", "model", True,
                                            {"compilation": "regional-v1"})
    preferences.save(database(), choice)
    monkeypatch.setattr(preferences.sys, "platform", "win32")
    env = {"AURADREAM_DB_PATH": str(database())}
    assert preferences.apply_saved_environment(env) is None
    assert env[preferences.LOADER_EXECUTION_ERROR_VARIABLE] == preferences.LOADER_EXECUTION_ERROR
    assert "AURADREAM_PRELOAD" not in env


@pytest.mark.parametrize("recipe", [None, {}, {"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"}])
def test_v2_canonical_loader_recipe_roundtrips_without_step_cache(recipe):
    choice = selected(recipe)
    preferences.save(database(), choice)
    payload = json.loads(preferences.path_for(database()).read_bytes())
    expected = choice.loader_execution.model_dump(exclude_none=True) if choice.loader_execution else None
    assert payload == {"version": 2, "model_id": MODEL, "quantization": "fp8", "offload": "model",
                       "load_on_startup": False, "loader_execution": expected}
    assert preferences.load(database()) == choice
    assert "step_cache" not in (payload["loader_execution"] or {})


@pytest.mark.parametrize("recipe", [{"step_cache": "fbc-005-v1"}, {"compilation": "unknown"}, {"future": "recipe"}])
def test_malformed_v2_cannot_become_an_eager_preload(recipe):
    path = preferences.path_for(database())
    payload = {"version": 2, "model_id": MODEL, "quantization": "fp8", "offload": "model",
               "load_on_startup": True, "loader_execution": recipe}
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    env = {"AURADREAM_DB_PATH": str(database()), "AURADREAM_PRELOAD": "true"}
    assert preferences.apply_saved_environment(env) is None
    assert env[preferences.LOADER_EXECUTION_ERROR_VARIABLE] == preferences.LOADER_EXECUTION_ERROR
    assert "AURADREAM_MODEL_ID" not in env and "AURADREAM_COMPILATION" not in env
    assert path.read_bytes() == before


@pytest.mark.parametrize("raw", [b"{", b"\xff", b"[]", b'{"version":999}', b'{"version":2,"version":1}',
                                  b'{"version":NaN}', b"x" * (65536 + 1), b"[" * 2000 + b"]" * 2000],
                         ids=["invalid-json", "invalid-utf8", "non-object", "future-version", "duplicate-version",
                              "nonfinite", "oversized", "deep-nesting"])
def test_corrupt_or_future_file_has_fixed_pure_diagnostic(raw):
    path = preferences.path_for(database())
    path.write_bytes(raw)
    before = dict(os.environ)
    result = preferences.load_result(database())
    assert result.preferences is None and result.loader_execution_error == preferences.LOADER_EXECUTION_ERROR
    assert dict(os.environ) == before and path.read_bytes() == raw


def test_missing_preferences_remain_read_only_without_diagnostic(tmp_path):
    absent = tmp_path / "not-created" / "gallery.db"
    assert preferences.load_result(absent) == preferences.PreferenceLoad(None)
    assert not absent.parent.exists()


@pytest.mark.parametrize("original", [b"{", b"\xff", b'{"version":999,"future":"recipe"}', b"", b"\n"],
                         ids=["invalid-json", "invalid-utf8", "future-json", "empty-file", "newline-file"])
def test_snapshot_restores_exact_invalid_bytes_and_their_restart_diagnostic(original):
    path = preferences.path_for(database())
    path.write_bytes(original)
    snapshot = preferences.capture_snapshot(database())
    assert snapshot.data == original and "recipe" not in repr(snapshot)
    preferences.save(database(), selected())
    assert preferences.load_result(database()).loader_execution_error is None
    preferences.restore_snapshot(database(), snapshot)
    assert path.read_bytes() == original
    assert preferences.load_result(database()).loader_execution_error == preferences.LOADER_EXECUTION_ERROR


def test_snapshot_restores_exact_absence_without_creating_missing_directories(tmp_path):
    absent = tmp_path / "not-created" / "gallery.db"
    missing = preferences.capture_snapshot(absent)
    assert missing == preferences.PreferenceSnapshot(None) and not absent.parent.exists()
    preferences.restore_snapshot(absent, missing)
    assert not absent.parent.exists()
    preferences.save(absent, selected())
    preferences.restore_snapshot(absent, missing)
    assert not preferences.path_for(absent).exists()
    assert preferences.load_result(absent) == preferences.PreferenceLoad(None)


def test_oversized_snapshot_refuses_capture_without_modifying_original():
    path = preferences.path_for(database())
    original = b"x" * (preferences._MAX_BYTES + 1)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="byte bound"):
        preferences.capture_snapshot(database())
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".auradream.settings.*.tmp"))


def test_unreadable_snapshot_refuses_capture_without_modifying_original(monkeypatch):
    path = preferences.path_for(database())
    original = b'{"version":999}'
    path.write_bytes(original)
    real_open = Path.open

    def refuse(target, *args, **kwargs):
        if target == path:
            raise PermissionError("injected snapshot read refusal")
        return real_open(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", refuse)
        with pytest.raises(PermissionError, match="injected"):
            preferences.capture_snapshot(database())
    assert path.read_bytes() == original


def test_failed_atomic_snapshot_restore_keeps_the_saved_file_and_cleans_owned_temporary(monkeypatch):
    snapshot = preferences.PreferenceSnapshot(b'{"version":999}')
    preferences.save(database(), selected())
    path = preferences.path_for(database())
    current = path.read_bytes()

    def refuse(*_args):
        raise OSError("injected rollback publication failure")

    monkeypatch.setattr(preferences.os, "replace", refuse)
    with pytest.raises(OSError, match="injected"):
        preferences.restore_snapshot(database(), snapshot)
    assert path.read_bytes() == current
    assert not list(path.parent.glob(".auradream.settings.*.tmp"))


def test_per_component_flags_and_environment_override_without_clearing_other_saved_component(monkeypatch):
    preferences.save(database(), selected({"compilation": "regional-v1"}))
    monkeypatch.setenv("AURADREAM_ATTENTION", "cudnn-v1")
    launch.configure(["--compilation", "eager"], ask=lambda _p: pytest.fail("asked terminal question"))
    assert os.environ["AURADREAM_COMPILATION"] == "eager"
    assert os.environ["AURADREAM_ATTENTION"] == "cudnn-v1"
    spec, error = config._loader_execution_environment()
    assert spec.compilation is None and spec.attention_backend == "cudnn-v1" and error is None
    assert preferences.startup_overrides() == {"compilation": "--compilation", "attention_backend": "AURADREAM_ATTENTION"}


def test_explicit_one_component_keeps_other_saved_component_for_cold_conflict_reporting(monkeypatch):
    preferences.save(database(), selected({"attention_backend": "cudnn-v1"}))
    monkeypatch.setenv("AURADREAM_COMPILATION", "regional-v1")
    launch.configure([], ask=lambda _p: pytest.fail("asked terminal question"))
    assert os.environ["AURADREAM_COMPILATION"] == "regional-v1"
    assert os.environ["AURADREAM_ATTENTION"] == "cudnn-v1"
    spec, error = config._loader_execution_environment()
    assert spec.compilation == "regional-v1" and spec.attention_backend == "cudnn-v1" and error is None


def test_defaults_skips_bad_saved_file_and_clears_only_old_diagnostic(monkeypatch):
    preferences.path_for(database()).write_bytes(b"{")
    monkeypatch.setenv(preferences.LOADER_EXECUTION_ERROR_VARIABLE, "untrusted old diagnostic")
    monkeypatch.setenv("AURADREAM_ATTENTION", "cudnn-v1")
    monkeypatch.setattr(preferences, "load_result", lambda _p: pytest.fail("--defaults read preferences"))
    launch.configure(["--defaults", "--compilation", "eager"])
    assert preferences.LOADER_EXECUTION_ERROR_VARIABLE not in os.environ
    assert os.environ["AURADREAM_ATTENTION"] == "cudnn-v1"
    assert os.environ["AURADREAM_COMPILATION"] == "eager"
    assert preferences.startup_overrides()["compilation"] == "--compilation"


def test_other_explicit_checkpoint_sets_aside_entire_v2_loader_choice():
    preferences.save(database(), selected({"compilation": "regional-v1"}, preload=True))
    launch.configure(["--model", OTHER])
    assert os.environ["AURADREAM_MODEL_ID"] == OTHER
    assert all(name not in os.environ for name in VARIABLES[1:6])


@pytest.mark.parametrize("explicit_attention", [False, True])
def test_menu_other_checkpoint_drops_only_saved_recipe_not_explicit_component(monkeypatch, explicit_attention):
    preferences.save(database(), selected({"compilation": "regional-v1"}, preload=True))
    args = ["--menu"]
    if explicit_attention:
        args += ["--attention", "cudnn-v1"]
    number = str(1 + [entry.repo for entry in launch.CHECKPOINTS].index(OTHER))
    answers = iter(["", number, ""])
    launch.configure(args, ask=lambda _p: next(answers), out=io.StringIO())
    assert "AURADREAM_COMPILATION" not in os.environ
    assert os.environ.get("AURADREAM_ATTENTION") == ("cudnn-v1" if explicit_attention else None)
    assert "AURADREAM_PRELOAD" not in os.environ


@pytest.mark.parametrize("variable,value", [("AURADREAM_COMPILATION", ""), ("AURADREAM_COMPILATION", "future"),
                                           ("AURADREAM_ATTENTION", " native "), ("AURADREAM_ATTENTION", "hostile" * 1000)],
                         ids=["empty-compilation", "unknown-compilation", "whitespace-attention", "long-attention"])
def test_invalid_explicit_execution_keeps_cold_browser_configuration_but_blocks_loader(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    before = dict(os.environ)
    spec, error = config._loader_execution_environment()
    assert spec is None and error == preferences.LOADER_EXECUTION_ERROR and (not value or value not in error)
    assert dict(os.environ) == before


def test_internal_diagnostic_is_fixed_not_raw_environment_text(monkeypatch):
    monkeypatch.setenv(preferences.LOADER_EXECUTION_ERROR_VARIABLE, "secret raw text")
    spec, error = config._loader_execution_environment()
    assert spec is None and error == preferences.LOADER_EXECUTION_ERROR and "secret raw text" not in error


def test_server_and_standalone_cli_share_explicit_recipe_flags():
    from backend import generate

    parsed = generate._build_parser().parse_args(["fixture", "--compilation", "regional-v1", "--attention", "native"])
    launch.apply_env(parsed)
    assert os.environ["AURADREAM_COMPILATION"] == "regional-v1"
    assert os.environ["AURADREAM_ATTENTION"] == "native"


def test_config_import_is_frozen_cold_and_does_not_bind_compiler_caches(tmp_path):
    script = '''
import importlib.abc, json, os, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'diffusers', 'triton'}:
            raise AssertionError('framework import during cold configuration')
sys.meta_path.insert(0, Refuse())
before = dict(os.environ)
from backend.config import settings
assert dict(os.environ) == before
assert not any(name.startswith(('TORCHINDUCTOR_', 'TRITON_CACHE_')) for name in os.environ)
print(json.dumps({'execution': settings.loader_execution.model_dump(exclude_none=True), 'error': settings.loader_execution_error}))
'''
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("AURADREAM_", "TORCHINDUCTOR_", "TRITON_CACHE_"))}
    env.update(AURADREAM_MODEL_ID=MODEL, AURADREAM_QUANT="fp8", AURADREAM_OFFLOAD="model",
               AURADREAM_COMPILATION="regional-v1", AURADREAM_ATTENTION="native",
               AURADREAM_DB_PATH=str(tmp_path / "unused.db"))
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"execution": {"compilation": "regional-v1"}, "error": None}
    assert list(tmp_path.iterdir()) == []
