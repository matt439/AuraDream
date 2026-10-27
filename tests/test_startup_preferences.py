"""Browser choices survive a server restart without overriding explicit startup."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from backend import launch, startup_preferences as preferences

MODEL = "black-forest-labs/FLUX.1-Krea-dev"
OTHER = "krea/Krea-2-Turbo"


@pytest.fixture
def remembered(tmp_path, monkeypatch):
    path = tmp_path / "gallery" / "local-diffusion.db"
    choice = preferences.StartupPreferences(OTHER, "int8", "sequential", True)
    preferences.save(path, choice)
    for name in [*choice.environ(), preferences.OVERRIDES_VARIABLE, preferences.LOADER_EXECUTION_ERROR_VARIABLE,
                 "LOCAL_DIFFUSION_COMPILATION", "LOCAL_DIFFUSION_ATTENTION"]:
        # Register restoration even for variables the launcher creates.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("LOCAL_DIFFUSION_DB_PATH", str(path))
    return path, choice


def never_ask(_prompt):
    pytest.fail("normal server startup must not ask terminal questions")


def test_round_trip_preserves_only_known_preferences(remembered):
    db_path, choice = remembered
    assert preferences.load(db_path) == choice
    payload = json.loads(preferences.path_for(db_path).read_text(encoding="utf-8"))
    assert payload == {"version": 2, **asdict(choice)}
    assert not db_path.exists(), "preferences must not create a database"
    assert list(db_path.parent.iterdir()) == [preferences.path_for(db_path)]


@pytest.mark.parametrize("interactive", [False, True])
def test_bare_start_restores_preferences_even_without_a_terminal(remembered, interactive):
    _, choice = remembered
    assert launch.configure([], ask=never_ask, interactive=interactive) is None
    assert {name: os.environ[name] for name in choice.environ()} == choice.environ()


def test_flags_then_environment_then_saved_settings_win(remembered, monkeypatch):
    _, choice = remembered
    monkeypatch.setenv("LOCAL_DIFFUSION_QUANT", "none")
    monkeypatch.setenv("LOCAL_DIFFUSION_PRELOAD", "false")
    launch.configure(["--model", choice.model_id], ask=never_ask)
    assert os.environ["LOCAL_DIFFUSION_MODEL_ID"] == choice.model_id
    assert os.environ["LOCAL_DIFFUSION_QUANT"] == "none"
    assert os.environ["LOCAL_DIFFUSION_OFFLOAD"] == "sequential"
    assert os.environ["LOCAL_DIFFUSION_PRELOAD"] == "false"
    assert preferences.startup_overrides() == {
        "model_id": "--model", "quantization": "LOCAL_DIFFUSION_QUANT",
        "load_on_startup": "LOCAL_DIFFUSION_PRELOAD",
    }


@pytest.mark.parametrize("by", ["flag", "environment"])
def test_another_explicit_model_sets_the_whole_saved_choice_aside(remembered, monkeypatch, caplog, by):
    # Saved loader options were chosen for their own checkpoint: BF16 with no
    # offload for a small model must not auto-load a large one that way.
    db_path, _ = remembered
    preferences.save(db_path, preferences.StartupPreferences(
        "black-forest-labs/FLUX.2-klein-4B", "none", "none", True,
    ))
    if by == "flag":
        launch.configure(["--model", OTHER], ask=never_ask)
    else:
        monkeypatch.setenv("LOCAL_DIFFUSION_MODEL_ID", OTHER)
        launch.configure([], ask=never_ask)
    assert os.environ["LOCAL_DIFFUSION_MODEL_ID"] == OTHER
    assert all(name not in os.environ for name in (
        "LOCAL_DIFFUSION_QUANT", "LOCAL_DIFFUSION_OFFLOAD", "LOCAL_DIFFUSION_PRELOAD",
    ))
    assert "Ignoring saved startup settings for black-forest-labs/FLUX.2-klein-4B" in caplog.text
    source = "--model" if by == "flag" else "LOCAL_DIFFUSION_MODEL_ID"
    assert preferences.startup_overrides() == {"model_id": source}


def test_a_menu_pick_of_another_model_sets_the_saved_auto_load_aside(remembered):
    # Anima saved with BF16, no offload and auto-load; the menu's Enters keep
    # BF16 and no offload, and its pick is FLUX.1 Krea dev. Starting that at
    # once was what an explicit --model for it never does.
    db_path, _ = remembered
    preferences.save(db_path, preferences.StartupPreferences("circlestone-labs/Anima", "none", "none", True))
    number = str(1 + [entry.repo for entry in launch.CHECKPOINTS].index(MODEL))
    asked = iter(["", number, ""])
    choice = launch.configure(["--menu"], ask=lambda _prompt: next(asked), out=io.StringIO())
    assert choice.model_id == MODEL
    assert os.environ.get("LOCAL_DIFFUSION_PRELOAD") != "true"
    assert preferences.startup_overrides() == {
        "model_id": "--menu", "quantization": "--menu", "offload": "--menu", "load_on_startup": "--menu",
    }


def test_a_menu_pick_of_the_saved_model_keeps_its_auto_load(remembered):
    _, choice = remembered
    asked = iter(["", "", ""])
    picked = launch.configure(["--menu"], ask=lambda _prompt: next(asked), out=io.StringIO())
    assert picked.model_id == choice.model_id
    assert os.environ["LOCAL_DIFFUSION_PRELOAD"] == "true"
    assert "load_on_startup" not in preferences.startup_overrides()


def test_explicit_loader_options_still_apply_with_another_model(remembered):
    launch.configure(["--model", MODEL, "--quant", "fp8", "--offload", "model"], ask=never_ask)
    assert (os.environ["LOCAL_DIFFUSION_QUANT"], os.environ["LOCAL_DIFFUSION_OFFLOAD"]) == ("fp8", "model")
    assert "LOCAL_DIFFUSION_PRELOAD" not in os.environ
    assert preferences.startup_overrides() == {
        "model_id": "--model", "quantization": "--quant", "offload": "--offload",
    }


def test_a_bare_start_records_no_overrides(remembered):
    launch.configure([], ask=never_ask)
    assert preferences.startup_overrides() == {}


@pytest.mark.parametrize("recorded", ["", "{", "[]", '{"model_id": 3, "host": "--x"}'])
def test_unusable_override_record_reads_as_none(recorded):
    assert preferences.startup_overrides({preferences.OVERRIDES_VARIABLE: recorded}) == {}


def test_defaults_bypasses_saved_choice_and_preserves_explicit_environment(remembered, monkeypatch):
    monkeypatch.setenv("LOCAL_DIFFUSION_MODEL_ID", MODEL)
    launch.configure(["--defaults", "--quant", "fp8"], ask=never_ask)
    assert os.environ["LOCAL_DIFFUSION_MODEL_ID"] == MODEL
    assert os.environ["LOCAL_DIFFUSION_QUANT"] == "fp8"
    assert "LOCAL_DIFFUSION_OFFLOAD" not in os.environ
    assert "LOCAL_DIFFUSION_PRELOAD" not in os.environ
    assert preferences.startup_overrides() == {
        "model_id": "LOCAL_DIFFUSION_MODEL_ID", "quantization": "--quant",
        "offload": "--defaults", "load_on_startup": "--defaults",
        "compilation": "--defaults", "attention_backend": "--defaults",
    }
    db_path, choice = remembered
    assert preferences.load(db_path) == choice, "one launch must not overwrite browser preferences"


def test_missing_preferences_are_read_only(tmp_path):
    db_path = tmp_path / "absent" / "gallery.db"
    assert preferences.load(db_path) is None
    assert not db_path.parent.exists()


@pytest.mark.parametrize("change", [
    {"version": 3}, {"version": True}, {"model_id": ""}, {"model_id": " bad/id"},
    {"model_id": "bad\nid"}, {"model_id": 123}, {"quantization": "fp4"},
    {"offload": "gpu"}, {"load_on_startup": "false"}, {"load_on_startup": 1},
    {"host": "0.0.0.0"},
])
def test_invalid_preferences_never_block_startup(remembered, change, caplog):
    db_path, choice = remembered
    path = preferences.path_for(db_path)
    path.write_text(json.dumps({"version": 2, **asdict(choice), **change}), encoding="utf-8")
    launch.configure([], ask=never_ask, interactive=True)
    assert "Ignoring saved startup settings" in caplog.text
    assert all(name not in os.environ for name in choice.environ())


@pytest.mark.parametrize("contents", [b"{", b"[]", b"null", b"\xff"])
def test_corrupt_or_non_object_file_is_ignored(tmp_path, contents, caplog):
    db_path = tmp_path / "gallery.db"
    preferences.path_for(db_path).write_bytes(contents)
    assert preferences.load(db_path) is None
    assert "Ignoring saved startup settings" in caplog.text


def test_failed_atomic_replace_keeps_previous_choice_and_cleans_temporary(remembered, monkeypatch):
    db_path, choice = remembered

    def refuse_replace(*args):
        raise PermissionError("read-only settings")

    monkeypatch.setattr(preferences.os, "replace", refuse_replace)
    with pytest.raises(PermissionError, match="read-only"):
        preferences.save(db_path, preferences.StartupPreferences(MODEL, "fp8", "model", False))
    assert preferences.load(db_path) == choice
    assert list(db_path.parent.iterdir()) == [preferences.path_for(db_path)]


def test_saved_nunchaku_on_windows_is_ignored_as_one_choice(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "gallery.db"
    choice = preferences.StartupPreferences("lite-infer/model-nvfp4", "nunchaku", "model", True)
    preferences.save(db_path, choice)
    monkeypatch.setattr(preferences.sys, "platform", "win32")
    env = {"LOCAL_DIFFUSION_DB_PATH": str(db_path)}
    preferences.apply_saved_environment(env)
    assert env == {"LOCAL_DIFFUSION_DB_PATH": str(db_path)}
    assert "requires Linux/WSL" in caplog.text


def test_saved_nunchaku_on_linux_retains_checkpoint_format(tmp_path, monkeypatch):
    db_path = tmp_path / "gallery.db"
    choice = preferences.StartupPreferences("lite-infer/model-int4", "nunchaku", "none", False)
    preferences.save(db_path, choice)
    monkeypatch.setattr(preferences.sys, "platform", "linux")
    env = {"LOCAL_DIFFUSION_DB_PATH": str(db_path)}
    preferences.apply_saved_environment(env)
    assert all(env[name] == value for name, value in choice.environ().items())


@pytest.mark.parametrize("explicit", [None, "false"])
def test_linux_server_can_choose_nunchaku_later_but_respects_explicit_kernel_setting(remembered, monkeypatch, explicit):
    monkeypatch.setattr(launch.sys, "platform", "linux")
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "")
    if explicit is None:
        monkeypatch.delenv("DIFFUSERS_TRUST_REMOTE_KERNELS")
    else:
        monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", explicit)
    launch.configure([], ask=never_ask)
    assert os.environ["DIFFUSERS_TRUST_REMOTE_KERNELS"] == (explicit or "true")


def test_windows_server_does_not_enable_nunchaku_kernels(remembered, monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "win32")
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "")
    monkeypatch.delenv("DIFFUSERS_TRUST_REMOTE_KERNELS")
    launch.configure([], ask=never_ask)
    assert "DIFFUSERS_TRUST_REMOTE_KERNELS" not in os.environ


def test_browser_choice_overrides_only_unchanged_wsl_script_defaults(remembered):
    db_path, choice = remembered
    env = {
        "LOCAL_DIFFUSION_DB_PATH": str(db_path),
        "LOCAL_DIFFUSION_QUANT": "nunchaku",
        "LOCAL_DIFFUSION_WSL_DEFAULT_QUANT": "nunchaku",
        "LOCAL_DIFFUSION_MODEL_ID": "lite-infer/wsl-default",
        "LOCAL_DIFFUSION_WSL_DEFAULT_MODEL_ID": "lite-infer/wsl-default",
    }
    assert preferences.explicit_sources(env) == {}
    preferences.apply_saved_environment(env)
    assert env["LOCAL_DIFFUSION_QUANT"] == choice.quantization
    assert env["LOCAL_DIFFUSION_MODEL_ID"] == choice.model_id
    # Changing a supplied value makes it an explicit override, even if the
    # shell still carries env.sh's earlier marker.
    env.update(LOCAL_DIFFUSION_QUANT="none", LOCAL_DIFFUSION_MODEL_ID=MODEL)
    assert preferences.explicit_sources(env) == {
        "model_id": "LOCAL_DIFFUSION_MODEL_ID", "quantization": "LOCAL_DIFFUSION_QUANT",
        "offload": "LOCAL_DIFFUSION_OFFLOAD", "load_on_startup": "LOCAL_DIFFUSION_PRELOAD",
    }
    preferences.apply_saved_environment(env)
    assert env["LOCAL_DIFFUSION_QUANT"] == "none"
    assert env["LOCAL_DIFFUSION_MODEL_ID"] == MODEL


@pytest.mark.parametrize("preload", [False, True])
def test_api_preference_survives_fresh_process_and_controls_lifespan_loading(client, fake, tmp_path, preload):
    from backend import main

    model = "cagliostrolab/animagine-xl-4.0"
    response = client.post("/api/model/switch", json={
        "model_id": model, "quantization": "int8", "offload": "sequential",
        "load": False, "load_on_startup": preload,
    })
    assert response.status_code == 200, response.text
    deadline = time.monotonic() + 5
    while client.get("/api/status").json()["switching"]:
        assert time.monotonic() < deadline, "the settings did not settle"
        time.sleep(0.01)
    assert fake.loads == 0, "saving a startup choice must not load immediately"

    # A new interpreter is important: config's dataclass defaults are frozen
    # at import, so reusing this test process could hide an ordering regression.
    env = {key: value for key, value in os.environ.items() if not key.startswith("LOCAL_DIFFUSION_")}
    env.update(
        LOCAL_DIFFUSION_DB_PATH=str(main.settings.db_path),
        LOCAL_DIFFUSION_OUTPUTS_DIR=str(tmp_path / "restarted-outputs"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    script = '''
import asyncio
import io
import json
from backend.launch import configure
configure([], ask=lambda _: (_ for _ in ()).throw(AssertionError("terminal question")))
from backend import main
from backend.jobs import JobManager
from tests.fakes import FakeBackend, FakeCritic, FakeTranslator
fake = FakeBackend(loaded=False)
main.backend = fake
main.prompt_critic = FakeCritic()
main.prompt_translator = FakeTranslator()
main.jobs = JobManager(fake, main.ensure_loaded)
async def probe():
    async with main.lifespan(main.app):
        if main.settings.load_on_startup:
            async def loaded():
                while main._load_state["state"] != "ready":
                    await asyncio.sleep(0.005)
            await asyncio.wait_for(loaded(), timeout=5)
        else:
            await asyncio.sleep(0)
        result = {
            "model_id": main.settings.model_id,
            "quantization": main.settings.quantization,
            "offload": main.settings.offload,
            "preload": main.settings.load_on_startup,
            "loads": fake.loads,
            "state": main._load_state["state"],
        }
    print(json.dumps(result))
asyncio.run(probe())
'''
    run = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout.splitlines()[-1]) == {
        "model_id": model, "quantization": "int8", "offload": "sequential",
        "preload": preload, "loads": int(preload), "state": "ready" if preload else "unloaded",
    }
