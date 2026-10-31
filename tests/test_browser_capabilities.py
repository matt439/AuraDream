"""Cold capability/fatal metadata checks; never launch a browser or use CDP."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
CAPABILITIES_FIXTURE = Path(__file__).resolve().parent / "browser_capabilities_fixture.mjs"
FATAL_FIXTURE = Path(__file__).resolve().parent / "browser_fatal_fixture.mjs"
pytestmark = pytest.mark.usefixtures("node")


def run_cold(tmp_path, masks, *, supported=False, probe=None):
    script = tmp_path / "must-not-read.json"
    script.write_text('[{"shot":"must-not-create.png"}]', encoding="utf-8")
    output = tmp_path / "must-not-create-output"
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    sentinel = "sensitive-global-fixture-value"
    # Patch the builtin exports consumed by the real harness, rather than
    # extracting/mirroring its guard. Any forbidden downstream step fails 99.
    request = dict(masks=masks, supported=supported, harness=str(ROOT / "tools" / "browser.mjs"),
                   script=str(script), output=str(output), probe=probe,
                   url=(ROOT / "tools" / "browser.mjs").as_uri())
    env = {**os.environ, "AURADREAM_SCREENSHOT_DIR": str(output),
           "AURADREAM_BROWSER": str(tmp_path / "fake-browser.exe"),
           "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    done = subprocess.run(["node", str(CAPABILITIES_FIXTURE), json.dumps(request)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == '[{"shot":"must-not-create.png"}]'
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["must-not-read.json", "owned-temp"]
    assert sentinel not in done.stdout + done.stderr
    return done


@pytest.mark.parametrize("name", ["fetch", "WebSocket"])
@pytest.mark.parametrize("mask", [
    {"remove": True}, {"value": None}, {"value": False}, {"value": 0},
    {"value": "sensitive-global-fixture-value"}, {"value": {"private": "sensitive-global-fixture-value"}},
], ids=["absent", "null", "false", "zero", "string", "object"])
def test_missing_or_noncallable_api_refuses_before_any_downstream_effect(tmp_path, name, mask):
    done = run_cold(tmp_path, {name: mask})
    assert done.returncode == 2
    assert done.stdout == ""
    assert done.stderr == (f"browser.mjs requires callable native Node APIs: {name}. "
                           "Use a Node runtime providing both fetch and WebSocket.\n")


def test_both_missing_apis_are_named_in_fixed_order_without_values(tmp_path):
    done = run_cold(tmp_path, {"WebSocket": {"remove": True}, "fetch": {"value": None}})
    assert done.returncode == 2 and done.stdout == ""
    assert done.stderr == ("browser.mjs requires callable native Node APIs: fetch, WebSocket. "
                           "Use a Node runtime providing both fetch and WebSocket.\n")


def test_callable_apis_parse_steps_then_reach_browser_lookup_without_invoking_them(tmp_path):
    done = run_cold(tmp_path, {}, supported=True)
    assert done.returncode == 2 and done.stdout == ""
    assert done.stderr == "no Edge or Chrome found; set AURADREAM_BROWSER to the executable\n"


@pytest.mark.parametrize("probe", ["input_read", "module_read", "output_write", "spawn", "fetch", "websocket"])
def test_cold_fixture_still_refuses_application_effects_after_source_capture(tmp_path, probe):
    done = run_cold(tmp_path, {}, supported=True, probe=probe)
    assert done.returncode == 99 and done.stdout == ""
    assert done.stderr == "UNEXPECTED_DOWNSTREAM_EFFECT\n"

def run_fatal_cold(tmp_path, steps, *, probe=None):
    """Exercise real fatal metadata preflight with every later effect poisoned."""
    script = tmp_path / "fatal-steps.json"
    script.write_text(json.dumps(steps), encoding="utf-8")
    output = tmp_path / "must-not-create-output"
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    request = dict(harness=str(ROOT / "tools/browser.mjs"), script=str(script),
                   url=(ROOT / "tools/browser.mjs").as_uri(), output=str(output),
                   temporary=str(temporary), probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_SCREENSHOT_DIR": str(output), "TMPDIR": str(temporary),
                   "TEMP": str(temporary), "TMP": str(temporary)}
    done = subprocess.run(["node", str(FATAL_FIXTURE), json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert json.loads(script.read_text(encoding="utf-8")) == steps
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fatal-steps.json", "owned-temp"]
    assert "private-fatal-fixture-value" not in done.stdout + done.stderr
    return done


@pytest.mark.parametrize("step", [
    {"expect": "true", "fatal": None}, {"expect": "true", "fatal": 0},
    {"expect": "true", "fatal": 1}, {"expect": "true", "fatal": "private-fatal-fixture-value"},
    {"expect": "true", "fatal": {}}, {"expect": "true", "fatal": []},
    {"fatal": True}, {"fatal": False}, {"eval": "true", "fatal": True},
    {"expect": "", "fatal": True}, {"expect": " \n\t ", "fatal": False},
    {"expect": None, "fatal": True}, {"expect": 1, "fatal": False},
    {"expect": False, "fatal": True}, {"expect": [], "fatal": True},
    {"expect": "true", "fatal": True, "eval": "false"},
    {"expect": "true", "fatal": False, "navigate": "about:blank"},
    {"expect": "true", "fatal": True, "wait": 1},
    {"expect": "true", "fatal": False, "shot": "must-not-create.png"},
    {"expect": "true", "fatal": True, "fullshot": "must-not-create.png"},
    {"expect": "true", "fatal": True, "elshot": ["body", "must-not-create.png"]},
    {"expect": "true", "fatal": True, "log": "'private-fatal-fixture-value'"},
    {"expect": "true", "fatal": True, "key": "Enter"},
    {"expect": "true", "fatal": True, "private-fatal-fixture-value": True},
])
def test_invalid_fatal_metadata_refuses_before_output_profile_spawn_or_network(tmp_path, step):
    # Even an earlier valid screenshot step must create nothing before a later
    # malformed prerequisite is rejected. Presence of fatal:false also validates.
    done = run_fatal_cold(tmp_path, [{"shot": "must-not-create.png"}, step])
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == ("step 2: fatal must be a boolean on an expect-only step "
                           "with a nonempty expression\n")


def test_fatal_metadata_reports_its_actual_first_step_index(tmp_path):
    done = run_fatal_cold(tmp_path, [{"expect": "true", "fatal": None}])
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == ("step 1: fatal must be a boolean on an expect-only step "
                           "with a nonempty expression\n")


@pytest.mark.parametrize("step", [
    {"expect": "Promise.resolve(true)", "fatal": True},
    {"expect": "false", "fatal": False}, {"expect": "true"},
    {"eval": "throw Error('ordinary eval is unchanged')"},
    {"expect": "true", "eval": "false"},
])
def test_valid_fatal_or_ordinary_steps_reach_the_existing_profile_boundary(tmp_path, step):
    done = run_fatal_cold(tmp_path, [step])
    # This is only preflight reachability: the poisoned scan prevents any
    # profile, browser, CDP or expression evaluation from taking place.
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_FATAL_PREFLIGHT_EFFECT\n"


@pytest.mark.parametrize("probe", ["source_read", "output_write", "profile_scan", "spawn", "fetch", "websocket"])
def test_fatal_cold_fixture_poisoned_effects_are_active(tmp_path, probe):
    done = run_fatal_cold(tmp_path, [{"expect": "true", "fatal": None}], probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_FATAL_PREFLIGHT_EFFECT\n"
