"""Viewport, mobile and inject step refusal before browser lookup or output/profile/network effects."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.browser_helpers import run_harness_cold

FIXTURE = Path(__file__).resolve().parent / "browser_viewport_fixture.mjs"
pytestmark = pytest.mark.usefixtures("node")

MESSAGE = ("viewport needs [width, height] or [width, height, dpr] "
           "with integer dimensions from 0 to 10000000 and a finite nonnegative dpr")


def run_viewport_cold(tmp_path, text: str, *, probe: str | None = None):
    """Execute the real harness module with exactly one fixture read permitted."""
    return run_harness_cold(
        tmp_path, FIXTURE, "viewport-steps.json", text, request=dict(probe=probe),
        environment={"AURADREAM_PORT_OFFSET": "0"}, secrets=("private-viewport-fixture-value",),
    )


@pytest.mark.parametrize("value", [
    None, False, True, 0, 1, "private-viewport-fixture-value", {},
    {"width": 1400, "height": 900}, [], [1400], [1400, 900, 1, 2],
    [None, 900], [1400, None], [True, 900], [1400, False],
    ["1400", 900], [1400, "900"], [10000001, 900], [1400, 10000001],
    [-1, 900], [1400, -1], [1400.5, 900], [1400, 900.5],
    [1400, 900, None], [1400, 900, False],
    [1400, 900, "1.25"], [1400, 900, -0.5],
])
def test_invalid_viewport_refuses_before_browser_lookup_or_any_later_effect(tmp_path, value):
    # Earlier screenshot steps must not create output before a later refusal.
    text = json.dumps([{"shot": "must-not-create.png"}, {"viewport": value}])
    done = run_viewport_cold(tmp_path, text)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 2: {MESSAGE}\n"


@pytest.mark.parametrize("values", ["1e309,900", "1400,1e309", "1400,900,1e309"])
def test_overflowed_json_numbers_refuse_as_nonfinite_viewport_values(tmp_path, values):
    # JSON's numeric grammar permits this literal; JS parses it to Infinity.
    done = run_viewport_cold(tmp_path, '[{"viewport":[' + values + ']}]')
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 1: {MESSAGE}\n"


def test_every_invalid_viewport_is_named_in_input_order_without_its_values(tmp_path):
    text = json.dumps([{"viewport": {}}, {"expect": "true"},
                       {"viewport": "private-viewport-fixture-value"}])
    done = run_viewport_cold(tmp_path, text)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 1: {MESSAGE}\nstep 3: {MESSAGE}\n"


@pytest.mark.parametrize("step", [
    {"viewport": [1400, 900]}, {"viewport": [600, 900, 1]},
    {"viewport": [2048, 1040, 1.25]}, {"viewport": [1536, 864, 2.75]},
    {"viewport": [100, 200, 0.5]}, {"expect": "true"},
    {"viewport": [0, 900]}, {"viewport": [1400, 0]}, {"viewport": [0, 0]},
    {"viewport": [1400, 900, 0]}, {"viewport": [0, 0, 0]},
    {"viewport": [10000000, 10000000, 0]},
])
def test_valid_documented_viewports_reach_browser_lookup_without_launch(tmp_path, step):
    done = run_viewport_cold(tmp_path, json.dumps([step]))
    # The first poisoned browser lookup stops valid input. No launch or CDP
    # takes place; omitted/fractional/zero DPR handling remains in the runtime
    # path. Maximum dimensions are never sent to an actual browser here.
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_VIEWPORT_PREFLIGHT_EFFECT\n"


@pytest.mark.parametrize("probe", [
    "source_read", "browser_lookup", "output_write", "output_mkdir",
    "profile_scan", "spawn", "fetch", "websocket",
])
def test_viewport_fixture_poisoned_effect_boundaries_are_active(tmp_path, probe):
    done = run_viewport_cold(tmp_path, '[{"viewport":{}}]', probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_VIEWPORT_PREFLIGHT_EFFECT\n"


MOBILE = "mobile needs true or false on a viewport step"
INJECT = "inject needs a relative path to a .js file in or below the folder of the steps file"


@pytest.mark.parametrize("step", [
    {"viewport": [393, 852, 3], "mobile": None}, {"viewport": [393, 852, 3], "mobile": 1},
    {"viewport": [393, 852, 3], "mobile": "true"}, {"viewport": [393, 852], "mobile": []},
    {"mobile": True}, {"mobile": False}, {"expect": "true", "mobile": True},
    {"touch": True, "mobile": True},
])
def test_mobile_is_refused_off_a_viewport_step_or_as_no_boolean(tmp_path, step):
    done = run_viewport_cold(tmp_path, json.dumps([{"shot": "must-not-create.png"}, step]))
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 2: {MOBILE}\n"


def test_a_bad_viewport_with_a_bad_mobile_names_both_in_order(tmp_path):
    done = run_viewport_cold(tmp_path, json.dumps([{"viewport": {}, "mobile": "private-viewport-fixture-value"}]))
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 1: {MESSAGE}\nstep 1: {MOBILE}\n"


@pytest.mark.parametrize("value", [
    None, True, 0, [], {}, "", " ", "phone_audit", "phone_audit.mjs", "phone_audit.js ",
    "/phone_audit.js", "C:/phone_audit.js", "../phone_audit.js", "a/../phone_audit.js",
    "a\\phone_audit.js", "a//phone_audit.js", "private-viewport-fixture-value",
])
def test_inject_needs_a_relative_js_path_inside_the_steps_folder(tmp_path, value):
    done = run_viewport_cold(tmp_path, json.dumps([{"shot": "must-not-create.png"}, {"inject": value}]))
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 2: {INJECT}\n"


@pytest.mark.parametrize("step", [
    {"viewport": [393, 852, 3], "mobile": True}, {"viewport": [1400, 900], "mobile": False},
    {"viewport": [852, 393, 3], "mobile": True},
    {"inject": "phone_audit.js"}, {"inject": "sub/dir/x.js"}, {"inject": "./x.js"},
])
def test_mobile_viewports_and_injects_reach_browser_lookup_without_launch(tmp_path, step):
    # The file an inject names is read only when its step runs, in the browser.
    done = run_viewport_cold(tmp_path, json.dumps([step]))
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_VIEWPORT_PREFLIGHT_EFFECT\n"
