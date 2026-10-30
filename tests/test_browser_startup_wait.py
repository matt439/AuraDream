"""Bounded startup polling in the entire harness, with no browser or real I/O."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "browser_startup_fixture.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def run_startup_cold(tmp_path, mode, *, probe=None):
    script = tmp_path / "startup-steps.json"
    script.write_text("[]", encoding="utf-8")
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    output = tmp_path / "must-not-create-output"
    harness = ROOT / "tools" / "browser.mjs"
    request = dict(harness=str(harness), url=harness.as_uri(), script=str(script),
                   temporary=str(temporary), output=str(output),
                   browser="owned-synthetic-browser", mode=mode, probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": request["browser"],
                   "AURADREAM_SCREENSHOT_DIR": str(output), "AURADREAM_PORT_OFFSET": "0",
                   "AURADREAM_VIEWPORT": "1400x900", "AURADREAM_DPR": "1",
                   "AURADREAM_BROWSER_DIAGNOSTICS": "1",
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    done = subprocess.run(["node", str(FIXTURE), json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == "[]"
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["owned-temp", "startup-steps.json"]
    return done


def startup_ports(done):
    assert done.returncode == 0 and done.stderr == "", done.stdout + done.stderr
    assert done.stdout.startswith("STARTUP_PORTS:") and done.stdout.count("\n") == 1
    ports = json.loads(done.stdout.removeprefix("STARTUP_PORTS:"))
    assert ports["reads"] == ports["lookups"] == ports["inventories"] == 1
    assert ports["stderrEncoding"] == ["utf8"]
    assert len(ports["spawns"]) == len(ports["removals"]) == ports["cleanupChecks"] == 1
    launch = ports["spawns"][0]
    assert launch["executable"] == "owned-synthetic-browser"
    # The repair changes only the polling allowance, never launch/security flags.
    assert launch["args"] == [
        "--headless=new", "--remote-debugging-port=0", "--window-size=1400,900",
        "--no-first-run", "--no-default-browser-check", "--disable-sync", "--disable-gpu",
        "--disable-component-update", "--disable-extensions", "--disable-background-networking",
        "--user-data-dir=" + launch["profile"], "about:blank",
    ]
    assert launch["options"] == {"stdio": ["ignore", "ignore", "pipe"], "detached": os.name != "nt"}
    assert ports["removals"] == [{"path": launch["profile"], "options": {
        # Retried by browser.mjs against a deadline: rmSync's own retries skip
        # Windows' EPERM. A removal that succeeds is one call and one check.
        "recursive": True, "force": True,
    }}]
    return ports


def test_startup_still_accepts_devtools_after_the_previous_75_poll_limit(tmp_path):
    ports = startup_ports(run_startup_cold(tmp_path, "ready_after_75"))
    assert ports["outcome"] == "reached_websocket"
    assert (ports["polls"], ports["sleeps"], ports["elapsed"]) == (76, 75, 15000)
    assert ports["fetches"] == ["http://127.0.0.1:49876/json"]
    assert ports["sockets"] == ["ws://127.0.0.1:49876/devtools/page/owned-blank"]
    assert ports["exitEmissions"] == 0


def test_unavailable_devtools_still_stops_after_150_polls_and_cleans_up(tmp_path):
    ports = startup_ports(run_startup_cold(tmp_path, "unavailable"))
    assert ports["outcome"] == "startup_timeout"
    assert (ports["polls"], ports["sleeps"], ports["elapsed"]) == (150, 150, 30000)
    assert ports["fetches"] == ports["sockets"] == []
    assert ports["exitEmissions"] == 0


def test_diagnostic_exit_still_refuses_without_waiting_for_the_poll_allowance(tmp_path):
    ports = startup_ports(run_startup_cold(tmp_path, "early_exit"))
    assert ports["outcome"] == "early_exit"
    assert (ports["polls"], ports["sleeps"], ports["elapsed"]) == (1, 1, 200)
    assert ports["fetches"] == ports["sockets"] == []
    assert ports["exitEmissions"] == 1


@pytest.mark.parametrize("probe", [
    "source_read", "output_write", "output_mkdir", "inventory", "spawn", "kill", "fetch", "websocket",
])
def test_startup_fixture_forbids_ungranted_effects_after_source_capture(tmp_path, probe):
    done = run_startup_cold(tmp_path, "unavailable", probe=probe)
    assert done.returncode == 99 and done.stdout == ""
    assert done.stderr == "UNEXPECTED_STARTUP_EFFECT\n"
