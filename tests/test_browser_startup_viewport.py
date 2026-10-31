"""Cold startup geometry refusal and actual parsed-value observation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "browser_startup_viewport_fixture.mjs"
pytestmark = pytest.mark.usefixtures("node")

STARTUP_MESSAGE = ("AURADREAM_VIEWPORT needs WIDTHxHEIGHT with integer dimensions in 1..10000000; "
                   "AURADREAM_DPR needs a finite positive number")


def run_startup_cold(tmp_path, viewport: str | None, dpr: str | None, *,
                     capture: bool = True, probe: str | None = None):
    """Run real preflight with one read and an observation before discovery."""
    script = tmp_path / "startup-steps.json"
    text = '[{"shot":"must-not-create.png"},{"expect":"true"}]'
    script.write_text(text, encoding="utf-8")
    output = tmp_path / "must-not-create-output"
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    request = dict(harness=str(ROOT / "tools" / "browser.mjs"), script=str(script),
                   url=(ROOT / "tools" / "browser.mjs").as_uri(), output=str(output),
                   temporary=str(temporary), capture=capture, probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_SCREENSHOT_DIR": str(output), "AURADREAM_PORT_OFFSET": "0",
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    for name, value in (("AURADREAM_VIEWPORT", viewport), ("AURADREAM_DPR", dpr)):
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    done = subprocess.run(["node", str(FIXTURE), json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == text
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["owned-temp", "startup-steps.json"]
    assert "private-startup-fixture-value" not in done.stdout + done.stderr
    return done


@pytest.mark.parametrize("viewport", [
    "Infinityx900", "1400xInfinity", "1e309x900", "1400x1e309",
    "1400.5x900", "1400x900.5", "0.5x900", "1400x0.5",
    "10000001x900", "1400x10000001", "9007199254740992x900", "1400x9007199254740992",
    "1400x900xextra", "1400x900x", "1400x900x100", "1400x900xx",
])
def test_previously_permissive_dimensions_refuse_before_browser_or_output(tmp_path, viewport):
    done = run_startup_cold(tmp_path, viewport, "1")
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == STARTUP_MESSAGE + "\nFIXTURE_READS:1\n"


@pytest.mark.parametrize("dpr", ["Infinity", "+Infinity", "1e309"])
def test_nonfinite_positive_dpr_refuses_before_browser_or_output(tmp_path, dpr):
    done = run_startup_cold(tmp_path, "1400x900", dpr)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == STARTUP_MESSAGE + "\nFIXTURE_READS:1\n"


@pytest.mark.parametrize(("viewport", "dpr", "expected"), [
    (None, None, [1400, 900, 1]), ("", "", [1400, 900, 1]),
    (None, "1.25", [1400, 900, 1.25]), ("2560x1305", None, [2560, 1305, 1]),
    ("2048x1040", "1.25", [2048, 1040, 1.25]),
    ("1536x864", "2.75", [1536, 864, 2.75]), ("1x1", "0.5", [1, 1, 0.5]),
    ("10000000x10000000", "0.25", [10000000, 10000000, 0.25]),
    ("10000000x1", "1", [10000000, 1, 1]), ("1x10000000", "1", [1, 10000000, 1]),
    ("1.4e3x9e2", "1e0", [1400, 900, 1]),
    ("1400.0x900.0", "1.0", [1400, 900, 1]),
    (" 1400 x 900 ", " 1.25 ", [1400, 900, 1.25]),
    ("+1400x+900", "+1.25", [1400, 900, 1.25]),
    ("01400x00900", "0x1", [1400, 900, 1]),
    ("1e7x1e0", "1e300", [10000000, 1, 1e300]),
])
def test_supported_forms_preserve_actual_parsed_startup_metrics(tmp_path, viewport, dpr, expected):
    done = run_startup_cold(tmp_path, viewport, dpr)
    assert done.returncode == 97 and done.stdout == "", done.stdout + done.stderr
    lines = done.stderr.splitlines()
    assert len(lines) == 2 and lines[0].startswith("STARTUP_METRICS:")
    assert json.loads(lines[0].removeprefix("STARTUP_METRICS:")) == dict(
        zip(("width", "height", "deviceScaleFactor"), expected))
    assert lines[1] == "FIXTURE_READS:1"


def test_valid_configuration_reaches_actual_lookup_without_observation_insertion(tmp_path):
    done = run_startup_cold(tmp_path, "2048x1040", "1.25", capture=False)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_STARTUP_VIEWPORT_EFFECT:browser_lookup\nFIXTURE_READS:1\n"


@pytest.mark.parametrize("probe", [
    "source_read", "browser_lookup", "output_write", "output_mkdir",
    "profile_scan", "spawn", "fetch", "websocket",
])
def test_startup_fixture_effect_poison_ports_are_active(tmp_path, probe):
    done = run_startup_cold(tmp_path, "Infinityx900", "Infinity", probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"UNEXPECTED_STARTUP_VIEWPORT_EFFECT:{probe}\nFIXTURE_READS:0\n"
