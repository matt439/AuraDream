"""Viewport step refusal before browser lookup or output/profile/network effects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "browser_viewport_fixture.mjs"
pytestmark = pytest.mark.usefixtures("node")

MESSAGE = ("viewport needs [width, height] or [width, height, dpr] "
           "with integer dimensions from 0 to 10000000 and a finite nonnegative dpr")


def run_viewport_cold(tmp_path, text: str, *, probe: str | None = None):
    """Execute the real harness module with exactly one fixture read permitted."""
    script = tmp_path / "viewport-steps.json"
    script.write_text(text, encoding="utf-8")
    output = tmp_path / "must-not-create-output"
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    request = dict(harness=str(ROOT / "tools" / "browser.mjs"), script=str(script),
                   url=(ROOT / "tools" / "browser.mjs").as_uri(), output=str(output),
                   temporary=str(temporary), probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_SCREENSHOT_DIR": str(output), "AURADREAM_PORT_OFFSET": "0",
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    done = subprocess.run(["node", str(FIXTURE), json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == text
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["owned-temp", "viewport-steps.json"]
    assert "private-viewport-fixture-value" not in done.stdout + done.stderr
    return done


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
