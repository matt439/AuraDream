"""Existing script/startup refusals run before browser lookup or output effects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")

NEEDS_KEY = "keycombo needs modifiers followed by a key"
NEEDS_MODIFIERS = "keycombo modifiers must be known and unique"


def run_keycombo_cold(tmp_path, steps: list[dict], *, allow_lookup: bool = False,
                      probe: str | None = None, environment_updates: dict | None = None):
    """Capture real modules, allow one fixture read, poison every later effect."""
    script = tmp_path / "keycombo-steps.json"
    text = json.dumps(steps)
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
const ports = { reads: 0, lookups: 0, mkdir: 0 };
const forbidden = name => function () {
  console.error('UNEXPECTED_KEYCOMBO_PREFLIGHT_EFFECT:' + name);
  process.exit(99);
};
fs.readFileSync = (path, ...args) => String(path) === input.script && ports.reads++ === 0
  ? realRead(path, ...args) : forbidden('source_read')();
for (const [name, label] of Object.entries({
  readdirSync: 'profile_scan', statSync: 'profile_stat', mkdtempSync: 'profile_create',
  rmSync: 'profile_remove', writeFileSync: 'output_write',
})) fs[name] = forbidden(label);
fs.existsSync = () => {
  ports.lookups++;
  // This is an inert existing-browser result, not an executable/file lookup.
  // It exposes the old screenshotRoot ordering while all real effects stay poisoned.
  return input.allow_lookup ? true : forbidden('browser_lookup')();
};
fs.mkdirSync = () => { ports.mkdir++; return forbidden('output_mkdir')(); };
childProcess.spawn = childProcess.spawnSync = forbidden('spawn');
syncBuiltinESMExports();
globalThis.fetch = forbidden('fetch');
globalThis.WebSocket = forbidden('websocket');
process.argv = [process.execPath, input.harness, input.script];
process.on('exit', () => console.error('PORTS:' + JSON.stringify(ports)));
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
                   temporary=str(temporary), allow_lookup=allow_lookup, probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_SCREENSHOT_DIR": str(output), "AURADREAM_PORT_OFFSET": "0",
                   "AURADREAM_VIEWPORT": "1400x900", "AURADREAM_DPR": "1",
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    environment.update(environment_updates or {})
    done = subprocess.run(["node", "--input-type=module", "-e", bootstrap, json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == text
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keycombo-steps.json", "owned-temp"]
    assert "private-keycombo-fixture-value" not in done.stdout + done.stderr
    assert "private-startup-fixture-value" not in done.stdout + done.stderr
    return done


@pytest.mark.parametrize(("combo", "message"), [
    (None, NEEDS_KEY), (False, NEEDS_KEY), (True, NEEDS_KEY), (0, NEEDS_KEY),
    ("private-keycombo-fixture-value", NEEDS_KEY), ({}, NEEDS_KEY),
    ([], NEEDS_KEY), (["Enter"], NEEDS_KEY), (["Control", None], NEEDS_KEY),
    (["Control", 0], NEEDS_KEY), (["Control", False], NEEDS_KEY), (["Control", ""], NEEDS_KEY),
    (["control", "Enter"], NEEDS_MODIFIERS), (["Hyper", "Enter"], NEEDS_MODIFIERS),
    ([1, "Enter"], NEEDS_MODIFIERS), ([None, "Enter"], NEEDS_MODIFIERS),
    (["Control", "Control", "Enter"], NEEDS_MODIFIERS),
    (["Alt", "Shift", "Alt", "x"], NEEDS_MODIFIERS),
])
def test_existing_invalid_chords_refuse_before_browser_lookup_or_output(tmp_path, combo, message):
    steps = [{"shot": "must-not-create.png"}, {"keycombo": combo}]
    done = run_keycombo_cold(tmp_path, steps)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 2: {message}\nPORTS:{{\"reads\":1,\"lookups\":0,\"mkdir\":0}}\n"


def test_bad_chord_is_refused_before_screenshot_mkdir_even_if_a_browser_would_exist(tmp_path):
    done = run_keycombo_cold(tmp_path, [{"shot": "must-not-create.png"},
                                       {"keycombo": ["Control", "Control", "Enter"]}],
                            allow_lookup=True)
    # The original source accepts the inert browser lookup and reaches the
    # poisoned mkdir port here; the relocated validator must never look it up.
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f"step 2: {NEEDS_MODIFIERS}\nPORTS:{{\"reads\":1,\"lookups\":0,\"mkdir\":0}}\n"


def test_existing_diagnostics_remain_aggregated_in_step_order_without_values(tmp_path):
    steps = [{"keycombo": None}, {"expect": "true"},
             {"keycombo": ["private-keycombo-fixture-value", "Enter"]}]
    done = run_keycombo_cold(tmp_path, steps)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == (f"step 1: {NEEDS_KEY}\nstep 3: {NEEDS_MODIFIERS}\n"
                           'PORTS:{"reads":1,"lookups":0,"mkdir":0}\n')


@pytest.mark.parametrize("step", [
    {"keycombo": ["Control", "Enter"]}, {"keycombo": ["Meta", "Enter"]},
    {"keycombo": ["Alt", "x"]}, {"keycombo": ["Shift", "x"]},
    {"keycombo": ["Control", "Shift", "Enter"]},
    {"keycombo": ["Shift", "Control", "x"]},
    {"keycombo": ["Alt", "Control", "Meta", "Shift", "Enter"]},
    {"keycombo": ["Control", " "]},
    {"keycombo": ["Control", "Control"]},
    {"expect": "true"}, {"shot": "still-cold.png"},
])
def test_accepted_existing_chords_and_ordinary_steps_reach_cold_browser_boundary(tmp_path, step):
    done = run_keycombo_cold(tmp_path, [step])
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == ('UNEXPECTED_KEYCOMBO_PREFLIGHT_EFFECT:browser_lookup\n'
                           'PORTS:{"reads":1,"lookups":1,"mkdir":0}\n')


@pytest.mark.parametrize("probe", [
    "source_read", "browser_lookup", "output_write", "output_mkdir",
    "profile_scan", "spawn", "fetch", "websocket",
])
def test_keycombo_fixture_effect_poison_ports_are_active(tmp_path, probe):
    done = run_keycombo_cold(tmp_path, [{"keycombo": None}], probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    ports = dict(reads=0, lookups=int(probe == "browser_lookup"), mkdir=int(probe == "output_mkdir"))
    assert done.stderr == (f"UNEXPECTED_KEYCOMBO_PREFLIGHT_EFFECT:{probe}\nPORTS:"
                           + json.dumps(ports, separators=(",", ":")) + "\n")


@pytest.mark.parametrize(("setup", "expression", "helper"), [
    ([], "__until(() => true) && true", "__until"),
    ([], "__until(() => true) || true", "__until"),
    ([], "!__until(() => true)", "__until"),
    ([], "true && __until(() => true)", "__until"),
    ([{"eval": "window.__wait = async () => true"}], "__wait() && true", "__wait"),
])
def test_existing_unawaited_helper_refusals_precede_browser_and_output(tmp_path, setup, expression, helper):
    steps = [{"shot": "must-not-create.png"}, *setup, {"expect": expression}]
    done = run_keycombo_cold(tmp_path, steps)
    message = f"step {len(steps)}: {helper}(...) joined with && or || is never awaited"
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == message + '\nPORTS:{"reads":1,"lookups":0,"mkdir":0}\n'


STARTUP_MESSAGE = ("AURADREAM_VIEWPORT needs WIDTHxHEIGHT with integer dimensions in 1..10000000; "
                   "AURADREAM_DPR needs a finite positive number")


@pytest.mark.parametrize("environment", [
    {"AURADREAM_VIEWPORT": "0x900"}, {"AURADREAM_VIEWPORT": "-1x900"},
    {"AURADREAM_VIEWPORT": "1400x0"}, {"AURADREAM_VIEWPORT": "1400x-1"},
    {"AURADREAM_VIEWPORT": "1400"},
    {"AURADREAM_VIEWPORT": "private-startup-fixture-value"},
    {"AURADREAM_VIEWPORT": "1400xbad"},
    {"AURADREAM_DPR": "0"}, {"AURADREAM_DPR": "-1"}, {"AURADREAM_DPR": "bad"},
])
def test_existing_nonpositive_or_missing_startup_values_refuse_before_preparation(tmp_path, environment):
    done = run_keycombo_cold(tmp_path, [{"shot": "must-not-create.png"}],
                            environment_updates=environment)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == STARTUP_MESSAGE + '\nPORTS:{"reads":1,"lookups":0,"mkdir":0}\n'


@pytest.mark.parametrize("environment", [
    {}, {"AURADREAM_VIEWPORT": "", "AURADREAM_DPR": ""},
    {"AURADREAM_VIEWPORT": "2048x1040", "AURADREAM_DPR": "1.25"},
    {"AURADREAM_VIEWPORT": "1x1", "AURADREAM_DPR": "0.5"},
])
def test_startup_guard_acceptance_is_unchanged_without_launching(tmp_path, environment):
    done = run_keycombo_cold(tmp_path, [{"expect": "true"}], environment_updates=environment)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == ('UNEXPECTED_KEYCOMBO_PREFLIGHT_EFFECT:browser_lookup\n'
                           'PORTS:{"reads":1,"lookups":1,"mkdir":0}\n')


@pytest.mark.parametrize("environment", [
    # DEV17 preserved permissive numeric parsing to isolate relocation. DEV18
    # now refuses these configurations while retaining the same early boundary.
    {"AURADREAM_VIEWPORT": "1400.5x900.5"},
    {"AURADREAM_VIEWPORT": "Infinityx900", "AURADREAM_DPR": "Infinity"},
    {"AURADREAM_VIEWPORT": "1400x900xextra"},
])
def test_previously_permissive_startup_inputs_now_refuse_before_preparation(tmp_path, environment):
    done = run_keycombo_cold(tmp_path, [{"expect": "true"}], environment_updates=environment)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == STARTUP_MESSAGE + '\nPORTS:{"reads":1,"lookups":0,"mkdir":0}\n'


@pytest.mark.parametrize(("setup", "expression"), [
    ([], "__until(() => true && true)"),
    ([], "(async () => await __until(() => true))()"),
    ([], "(async () => true && await __until(() => true))()"),
    ([{"eval": "window.__wait = async () => true"}], "__wait()"),
])
def test_existing_awaited_helper_forms_reach_browser_boundary(tmp_path, setup, expression):
    done = run_keycombo_cold(tmp_path, [*setup, {"expect": expression}])
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == ('UNEXPECTED_KEYCOMBO_PREFLIGHT_EFFECT:browser_lookup\n'
                           'PORTS:{"reads":1,"lookups":1,"mkdir":0}\n')


@pytest.mark.parametrize(("steps", "environment", "message"), [
    ([{"shot": "must-not-create.png"}, {"expect": "__until(() => true) && true"}], {},
     "step 2: __until(...) joined with && or || is never awaited"),
    ([{"shot": "must-not-create.png"}], {"AURADREAM_DPR": "0"}, STARTUP_MESSAGE),
])
def test_all_late_refusals_precede_screenshot_mkdir_even_with_inert_browser(tmp_path, steps, environment, message):
    done = run_keycombo_cold(tmp_path, steps, allow_lookup=True, environment_updates=environment)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == message + '\nPORTS:{"reads":1,"lookups":0,"mkdir":0}\n'


@pytest.mark.parametrize(("steps", "message"), [
    ([{"keycombo": ["Control", "Control", "x"]}, {"expect": "__until(() => true) && true"}],
     "step 1: " + NEEDS_MODIFIERS),
    ([{"keycombo": ["Control", "Enter"]}, {"expect": "__until(() => true) && true"}],
     "step 2: __until(...) joined with && or || is never awaited"),
])
def test_existing_combo_async_startup_refusal_priority_is_preserved(tmp_path, steps, message):
    done = run_keycombo_cold(tmp_path, steps, environment_updates={"AURADREAM_DPR": "0"})
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == message + '\nPORTS:{"reads":1,"lookups":0,"mkdir":0}\n'
