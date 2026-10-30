"""Direct-harness port refusal before browser/output/profile/network effects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "browser_port_fixture.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")

OFFSET_MESSAGE = "AURADREAM_PORT_OFFSET must be a finite safe integer"
PORT_MESSAGE = "effective local ports must be integers in 1..65535 after AURADREAM_PORT_OFFSET"


def run_port_cold(tmp_path, steps: list[dict], offset: str | None, *, probe: str | None = None):
    """Load captured real modules with fixture I/O allowed and all effects inert."""
    script = tmp_path / "port-steps.json"
    text = json.dumps(steps)
    script.write_text(text, encoding="utf-8")
    output = tmp_path / "must-not-create-output"
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    request = dict(harness=str(ROOT / "tools" / "browser.mjs"), script=str(script),
                   url=(ROOT / "tools" / "browser.mjs").as_uri(), output=str(output),
                   temporary=str(temporary), probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_SCREENSHOT_DIR": str(output),
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    if offset is None:
        environment.pop("AURADREAM_PORT_OFFSET", None)
    else:
        environment["AURADREAM_PORT_OFFSET"] = offset
    done = subprocess.run(["node", str(FIXTURE), json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == text
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["owned-temp", "port-steps.json"]
    assert "private-offset-fixture-value" not in done.stdout + done.stderr
    return done


@pytest.mark.parametrize("offset", [
    "private-offset-fixture-value", "NaN", "Infinity", "-Infinity", "1e309", "-1e309",
    "0.5", "-0.5", "9007199254740992", "-9007199254740992",
])
def test_bad_offset_refuses_before_fixture_read_or_any_later_effect(tmp_path, offset):
    done = run_port_cold(tmp_path, [{"shot": "must-not-create.png"},
                                    {"navigate": "http://127.0.0.1:7860/"}], offset)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == OFFSET_MESSAGE + "\nFIXTURE_READS:0\n"


def test_invalid_offset_refuses_even_without_a_local_url(tmp_path):
    done = run_port_cold(tmp_path, [{"expect": "true"}], "private-offset-fixture-value")
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == OFFSET_MESSAGE + "\nFIXTURE_READS:0\n"


@pytest.mark.parametrize(("host", "port", "offset"), [
    ("127.0.0.1", "7860", "-7860"), ("localhost", "7860", "-7861"),
    ("localhost", "65535", "1"), ("127.0.0.1", "00", "0"),
    ("localhost", "65536", "0"), ("localhost", "99999", "0"),
    ("127.0.0.1", "7860", "9007199254740991"),
    ("localhost", "7860", "-9007199254740991"),
])
def test_invalid_effective_local_port_refuses_before_browser_or_output(tmp_path, host, port, offset):
    # A later bad URL must be found before an earlier screenshot or request.
    steps = [{"shot": "must-not-create.png"},
             {"eval": "fetch('http://" + host + ":" + port + "/api/status')"}]
    done = run_port_cold(tmp_path, steps, offset)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == PORT_MESSAGE + "\nFIXTURE_READS:1\n"


@pytest.mark.parametrize(("offset", "before", "after"), [
    (None, "http://127.0.0.1:7860/", "http://127.0.0.1:7860/"),
    ("", "http://localhost:7860/", "http://localhost:7860/"),
    ("0", "http://localhost:00080/", "http://localhost:00080/"),
    ("  ", "http://localhost:80/", "http://localhost:80/"),
    ("-0", "http://127.0.0.1:80/", "http://127.0.0.1:80/"),
    ("100", "http://127.0.0.1:7860/", "http://127.0.0.1:7960/"),
    ("-100", "http://localhost:7860/", "http://localhost:7760/"),
    ("+100", "http://localhost:7860/", "http://localhost:7960/"),
    ("1e2", "http://localhost:7860/", "http://localhost:7960/"),
    ("0x64", "http://localhost:7860/", "http://localhost:7960/"),
    ("1.0", "http://localhost:7860/", "http://localhost:7861/"),
    ("-9", "http://localhost:10/", "http://localhost:1/"),
    ("1", "http://localhost:00/", "http://localhost:1/"),
    ("1", "http://localhost:65534/", "http://localhost:65535/"),
    ("-1", "http://localhost:65536/", "http://localhost:65535/"),
    ("100", "https://example.invalid:7860/", "https://example.invalid:7860/"),
    ("100", "http://192.0.2.1:7860/", "http://192.0.2.1:7860/"),
    ("100", "http://localhost/api/status", "http://localhost/api/status"),
    # These do not match the existing two-to-five-digit substitution grammar.
    # A whole-URL validator or broadened matching policy is outside this repair.
    ("1", "http://localhost:9/", "http://localhost:9/"),
    ("1", "http://localhost:100000/", "http://localhost:100000/"),
])
def test_valid_offsets_transform_the_actual_parsed_fixture_then_stop_cold(tmp_path, offset, before, after):
    done = run_port_cold(tmp_path, [{"navigate": before}], offset)
    assert done.returncode == 97 and done.stdout == "", done.stdout + done.stderr
    lines = done.stderr.splitlines()
    assert len(lines) == 2 and lines[0].startswith("STOP_BROWSER_LOOKUP:")
    assert json.loads(lines[0].removeprefix("STOP_BROWSER_LOOKUP:")) == [{"navigate": after}]
    assert lines[1] == "FIXTURE_READS:1"


def test_all_matching_local_ports_in_eval_and_navigation_are_transformed(tmp_path):
    steps = [{"navigate": "http://127.0.0.1:7860/"},
             {"eval": "fetch('http://localhost:99/api/status'); fetch('http://127.0.0.1:80/')"}]
    done = run_port_cold(tmp_path, steps, "1")
    assert done.returncode == 97 and done.stdout == "", done.stdout + done.stderr
    parsed = json.loads(done.stderr.splitlines()[0].removeprefix("STOP_BROWSER_LOOKUP:"))
    assert parsed == [{"navigate": "http://127.0.0.1:7861/"},
                      {"eval": "fetch('http://localhost:100/api/status'); fetch('http://127.0.0.1:81/')"}]


def test_zero_offset_preserves_leading_zero_text_inside_actual_eval_data(tmp_path):
    steps = [{"eval": "const note = 'localhost:00080'; const value = '127.0.0.1:00009';"}]
    done = run_port_cold(tmp_path, steps, "-0")
    assert done.returncode == 97 and done.stdout == "", done.stdout + done.stderr
    parsed = json.loads(done.stderr.splitlines()[0].removeprefix("STOP_BROWSER_LOOKUP:"))
    assert parsed == steps


@pytest.mark.parametrize("probe", [
    "source_read", "browser_lookup", "output_write", "output_mkdir",
    "profile_scan", "spawn", "fetch", "websocket",
])
def test_port_fixture_forbidden_effect_ports_are_active(tmp_path, probe):
    done = run_port_cold(tmp_path, [{"expect": "true"}], "private-offset-fixture-value", probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_PORT_PREFLIGHT_EFFECT\nFIXTURE_READS:0\n"
