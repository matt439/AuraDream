"""Fixture read/syntax refusals use exit 2 before browser or output effects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
READ_MESSAGE = "could not read browser steps file; check its path and read access"
JSON_MESSAGE = "browser steps file must contain valid JSON"


def run_input_cold(tmp_path, text: str, *, read_error: str | None = None,
                   parse_error: str | None = None, guard: str | None = None,
                   offset: str = "0", probe: str | None = None):
    """Capture real modules, then count/poison ports and relay import errors."""
    script = tmp_path / "input-steps.json"
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
const ports = { reads: 0, parses: 0 };
function forbidden() { console.error('UNEXPECTED_INPUT_PREFLIGHT_EFFECT'); process.exit(99); }
fs.readFileSync = (path, ...args) => {
  if (String(path) !== input.script || ports.reads++ !== 0) return forbidden();
  if (input.read_error) {
    // Simulate a named OS error at the permitted port; no denied path is read.
    const error = new Error('private-input-fixture-value');
    error.code = input.read_error;
    throw error;
  }
  return realRead(path, ...args);
};
for (const name of ['existsSync', 'readdirSync', 'statSync', 'mkdirSync', 'mkdtempSync', 'rmSync', 'writeFileSync']) {
  fs[name] = forbidden;
}
childProcess.spawn = childProcess.spawnSync = forbidden;
syncBuiltinESMExports();
globalThis.fetch = forbidden;
globalThis.WebSocket = forbidden;
if (input.guard === 'missing_api') globalThis.fetch = undefined;
process.argv = [process.execPath, input.harness, ...(input.guard === 'missing_arg' ? [] : [input.script])];
const realParse = JSON.parse;
JSON.parse = (...args) => {
  ports.parses++;
  if (input.parse_error) {
    const kinds = { Error, RangeError, TypeError };
    throw new kinds[input.parse_error]('private-input-fixture-value');
  }
  return realParse(...args);
};
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
try {
  await import(harnessModule);
} catch (error) {
  // Relay the real module's unhandled error without printing a data-URL stack
  // or fixture body. Exit 1 remains distinct from a deliberate script refusal.
  console.error('UNHANDLED_FIXTURE_EXCEPTION:' + error.name);
  process.exitCode = 1;
}
'''
    request = dict(harness=str(ROOT / "tools" / "browser.mjs"), script=str(script),
                   url=(ROOT / "tools" / "browser.mjs").as_uri(), output=str(output),
                   temporary=str(temporary), read_error=read_error, parse_error=parse_error,
                   guard=guard, probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_SCREENSHOT_DIR": str(output), "AURADREAM_PORT_OFFSET": offset,
                   "AURADREAM_VIEWPORT": "1400x900", "AURADREAM_DPR": "1",
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    done = subprocess.run(["node", "--input-type=module", "-e", bootstrap, json.dumps(request)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == text
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input-steps.json", "owned-temp"]
    assert "private-input-fixture-value" not in done.stdout + done.stderr
    return done


@pytest.mark.parametrize("code", ["ENOENT", "EACCES", "EISDIR", "EPERM", "EIO"])
def test_input_read_errors_are_fixed_refusals_before_parsing_or_later_effects(tmp_path, code):
    # The owned fixture also has bad JSON: the earlier read error must win.
    done = run_input_cold(tmp_path, '["private-input-fixture-value",', read_error=code)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == READ_MESSAGE + '\nPORTS:{"reads":1,"parses":0}\n'


@pytest.mark.parametrize("text", [
    "", " \n\t", "{", "{]", '[{"expect":"true"},]', "[{'expect':'true'}]",
    '[{"type":"private-input-fixture-value}',
    '[{"expect":"true"}] private-input-fixture-value',
    '\ufeff[]', '[{"type":"line\nbreak"}]', "NaN", "Infinity",
])
def test_malformed_json_is_fixed_refusal_without_fixture_body_or_later_effects(tmp_path, text):
    done = run_input_cold(tmp_path, text)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == JSON_MESSAGE + '\nPORTS:{"reads":1,"parses":1}\n'


@pytest.mark.parametrize("text", [
    "[]", "[{}]", '[{"expect":"true"}]', '[{"shot":"must-not-create.png"}]',
    ' \n[{"expect":"true","fatal":false}]\t',
])
def test_valid_documents_reach_existing_browser_boundary_without_effects(tmp_path, text):
    done = run_input_cold(tmp_path, text)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == 'UNEXPECTED_INPUT_PREFLIGHT_EFFECT\nPORTS:{"reads":1,"parses":1}\n'


@pytest.mark.parametrize("kind", ["Error", "RangeError", "TypeError"])
def test_non_syntax_parser_faults_are_not_reclassified_as_invalid_json(tmp_path, kind):
    done = run_input_cold(tmp_path, "[]", parse_error=kind)
    assert done.returncode == 1 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == f'UNHANDLED_FIXTURE_EXCEPTION:{kind}\nPORTS:{{"reads":1,"parses":1}}\n'


@pytest.mark.parametrize(("text", "arguments", "message", "reads", "parses"), [
    ("[", {"guard": "missing_api"},
     "browser.mjs requires callable native Node APIs: fetch. Use a Node runtime providing both fetch and WebSocket.", 0, 0),
    ("[", {"guard": "missing_arg"}, "usage: node tools/browser.mjs <steps.json>", 0, 0),
    ("[", {"offset": "private-input-fixture-value"}, "AURADREAM_PORT_OFFSET must be a finite safe integer", 0, 0),
    ('[{"navigate":"http://localhost:65535/"},]', {"offset": "1"},
     "effective local ports must be integers in 1..65535 after AURADREAM_PORT_OFFSET", 1, 0),
    ("{}", {}, "browser steps must be a JSON array", 1, 1),
    ('[{"viewport":{}}]', {},
     "step 1: viewport needs [width, height] or [width, height, dpr] with integer dimensions from 0 to 10000000 and a finite nonnegative dpr", 1, 1),
    ('[{"expect":"true","fatal":"private-input-fixture-value"}]', {},
     "step 1: fatal must be a boolean on an expect-only step with a nonempty expression", 1, 1),
])
def test_existing_earlier_and_later_refusal_order_and_diagnostics_remain(tmp_path, text, arguments, message, reads, parses):
    done = run_input_cold(tmp_path, text, **arguments)
    assert done.returncode == 2 and done.stdout == "", done.stdout + done.stderr
    counts = json.dumps(dict(reads=reads, parses=parses), separators=(",", ":"))
    assert done.stderr == message + "\nPORTS:" + counts + "\n"


@pytest.mark.parametrize("probe", [
    "source_read", "browser_lookup", "output_write", "output_mkdir",
    "profile_scan", "spawn", "fetch", "websocket",
])
def test_input_fixture_poisoned_effect_ports_are_active(tmp_path, probe):
    done = run_input_cold(tmp_path, "[]", probe=probe)
    assert done.returncode == 99 and done.stdout == "", done.stdout + done.stderr
    assert done.stderr == 'UNEXPECTED_INPUT_PREFLIGHT_EFFECT\nPORTS:{"reads":0,"parses":0}\n'
