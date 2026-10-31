"""Cold step-container refusal; install under repository tests before running."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.browser_helpers import run_harness_cold

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "browser_containers_fixture.mjs"
pytestmark = pytest.mark.usefixtures("node")

ROOT_MESSAGE = "browser steps must be a JSON array"
STEP_MESSAGE = "step must be a JSON object"


def run_container_cold(tmp_path, value, *, probe: str | None = None):
    """Run the actual module until refusal or a poisoned browser lookup."""
    return run_harness_cold(
        tmp_path, FIXTURE, "container-steps.json", json.dumps(value), request=dict(probe=probe),
        environment={"AURADREAM_PORT_OFFSET": "0"}, secrets=("private-container-fixture-value",),
    )


@pytest.mark.parametrize("value", [
    None, False, True, 0, 1, 1.5, "private-container-fixture-value", {}, {"steps": []},
])
def test_nonarray_document_refuses_before_browser_lookup_or_later_effects(tmp_path, value):
    done = run_container_cold(tmp_path, value)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == ROOT_MESSAGE + "\nFIXTURE_READS:1\n"


@pytest.mark.parametrize("entry", [
    None, False, True, 0, 1, -2, 1.5, "", "private-container-fixture-value", [], [{}],
])
def test_nonobject_step_refuses_before_any_earlier_action_or_output(tmp_path, entry):
    # The malformed entry must be found before either earlier action executes.
    steps = [{"shot": "must-not-create.png"}, {"eval": "throw Error('must not evaluate')"}, entry]
    done = run_container_cold(tmp_path, steps)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "step 3: " + STEP_MESSAGE + "\nFIXTURE_READS:1\n"


def test_every_bad_entry_is_named_in_input_order_without_its_values(tmp_path):
    done = run_container_cold(tmp_path, [None, {}, "private-container-fixture-value", [], {"wait": 0}, False])
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == ("step 1: " + STEP_MESSAGE + "\nstep 3: " + STEP_MESSAGE
                           + "\nstep 4: " + STEP_MESSAGE + "\nstep 6: " + STEP_MESSAGE
                           + "\nFIXTURE_READS:1\n")


@pytest.mark.parametrize("steps", [
    [], [{}], [{"wait": 0}], [{"expect": "true"}],
    [{"expect": "Promise.resolve(true)", "fatal": True}],
    [{"expect": "true", "eval": "false"}],
    [{"unknown": "private-container-fixture-value"}],
    [{"wait": 0, "label": "named metadata"}],
    [{"unknown": None, "nested": [False, 0, None]}],
])
def test_existing_object_forms_reach_browser_lookup_without_launch(tmp_path, steps):
    done = run_container_cold(tmp_path, steps)
    # This proves only preflight reachability. The first browser lookup is a
    # poisoned boundary: no executable, output, profile or network is used.
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_CONTAINER_PREFLIGHT_EFFECT\nFIXTURE_READS:1\n"


def test_repository_smoke_records_reach_the_same_cold_lookup_boundary(tmp_path):
    # Each fixture owns its helper names. Concatenating files can make a
    # synchronous helper look async because another fixture defines that name.
    smokes = sorted((ROOT / "tools").glob("smoke*.json"))
    assert smokes
    for script in smokes:
        value = json.loads(script.read_text(encoding="utf-8"))
        case = tmp_path / script.stem
        case.mkdir()
        done = run_container_cold(case, value)
        details = script.name + ": " + done.stdout + done.stderr
        assert done.returncode == 99 and done.stdout == "", details
        assert done.stderr == "UNEXPECTED_CONTAINER_PREFLIGHT_EFFECT\nFIXTURE_READS:1\n", details


@pytest.mark.parametrize("probe", [
    "source_read", "browser_lookup", "output_write", "output_mkdir",
    "profile_scan", "spawn", "fetch", "websocket",
])
def test_container_fixture_poisoned_effect_ports_are_active(tmp_path, probe):
    done = run_container_cold(tmp_path, None, probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_CONTAINER_PREFLIGHT_EFFECT\nFIXTURE_READS:0\n"
