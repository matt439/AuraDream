"""Cold step-container refusal; install under repository tests before running."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")

ROOT_MESSAGE = "browser steps must be a JSON array"
STEP_MESSAGE = "step must be a JSON object"


def run_container_cold(tmp_path, value, *, probe: str | None = None):
    """Run the actual module until refusal or a poisoned browser lookup."""
    script = tmp_path / "container-steps.json"
    text = json.dumps(value)
    script.write_text(text, encoding="utf-8")
    output = tmp_path / "must-not-create-output"
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    bootstrap = r'''
import fs from 'node:fs';
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
const input = JSON.parse(process.argv[1]);
const asModule = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const realRead = fs.readFileSync;
const outputModule = asModule(realRead(new URL('./browser-output.mjs', input.url), 'utf8'));
const source = realRead(input.harness, 'utf8');
const dependency = "'./browser-output.mjs'";
if (source.split(dependency).length !== 2) throw Error('expected one browser-output dependency');
const harnessModule = asModule(source.replace(dependency, JSON.stringify(outputModule)));
function forbidden() { console.error('UNEXPECTED_CONTAINER_PREFLIGHT_EFFECT'); process.exit(99); }
let reads = 0;
fs.readFileSync = (path, ...args) => String(path) === input.script && reads++ === 0
  ? realRead(path, ...args) : forbidden();
for (const name of ['existsSync', 'readdirSync', 'statSync', 'mkdirSync', 'mkdtempSync', 'rmSync', 'writeFileSync']) {
  fs[name] = forbidden;
}
childProcess.spawn = childProcess.spawnSync = forbidden;
syncBuiltinESMExports();
globalThis.fetch = forbidden;
globalThis.WebSocket = forbidden;
process.argv = [process.execPath, input.harness, input.script];
process.on('exit', () => console.error('FIXTURE_READS:' + reads));
const probes = {
  source_read: () => fs.readFileSync(input.harness),
  browser_lookup: () => fs.existsSync('must-not-discover-browser'),
  output_write: () => fs.writeFileSync(input.output, 'must-not-write'),
  output_mkdir: () => fs.mkdirSync(input.output),
  profile_scan: () => fs.readdirSync(input.temporary),
  spawn: () => childProcess.spawn('must-not-run'),
  fetch: () => globalThis.fetch('http://must-not-connect.invalid'),
  websocket: () => new globalThis.WebSocket('ws://must-not-connect.invalid'),
};
if (input.probe) probes[input.probe]();
await import(harnessModule);
'''
    request = dict(harness=str(ROOT / "tools" / "browser.mjs"), script=str(script),
                   url=(ROOT / "tools" / "browser.mjs").as_uri(), output=str(output),
                   temporary=str(temporary), probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_SCREENSHOT_DIR": str(output), "AURADREAM_PORT_OFFSET": "0",
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    done = subprocess.run(["node", "--input-type=module", "-e", bootstrap, json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == text
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["container-steps.json", "owned-temp"]
    assert "private-container-fixture-value" not in done.stdout + done.stderr
    return done


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
    # Execute the real whole-list check over the current fixture inventory;
    # do not mirror that check or run any fixture expression/action here.
    smokes = sorted((ROOT / "tools").glob("smoke*.json"))
    assert smokes
    steps = []
    for script in smokes:
        value = json.loads(script.read_text(encoding="utf-8"))
        # Represent a malformed document as a bad aggregate entry rather than
        # letting a dict become an apparently valid ordinary step or disappear.
        # Dedicated cases above exercise the harness's actual top-level refusal.
        steps.extend(value if isinstance(value, list) else [None])
    done = run_container_cold(tmp_path, steps)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_CONTAINER_PREFLIGHT_EFFECT\nFIXTURE_READS:1\n"


@pytest.mark.parametrize("probe", [
    "source_read", "browser_lookup", "output_write", "output_mkdir",
    "profile_scan", "spawn", "fetch", "websocket",
])
def test_container_fixture_poisoned_effect_ports_are_active(tmp_path, probe):
    done = run_container_cold(tmp_path, None, probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == "UNEXPECTED_CONTAINER_PREFLIGHT_EFFECT\nFIXTURE_READS:0\n"
