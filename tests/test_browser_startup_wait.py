"""Bounded startup polling in the entire harness, with no browser or real I/O."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def run_startup_cold(tmp_path, mode, *, probe=None):
    script = tmp_path / "startup-steps.json"
    script.write_text("[]", encoding="utf-8")
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    output = tmp_path / "must-not-create-output"
    bootstrap = r'''
import fs from 'node:fs';
import childProcess from 'node:child_process';
import { EventEmitter } from 'node:events';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
const input = JSON.parse(process.argv[1]);
// Capture both actual modules before poisoning application effects. Data URLs
// also avoid Node 22's loader reads being mistaken for harness input reads.
const realRead = fs.readFileSync;
const asModule = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const outputModule = asModule(realRead(new URL('./browser-output.mjs', input.url), 'utf8'));
const source = realRead(input.harness, 'utf8');
const dependency = "'./browser-output.mjs'";
if (source.split(dependency).length !== 2) throw Error('expected one browser-output dependency');
const harnessModule = asModule(source.replace(dependency, JSON.stringify(outputModule)));
const ports = {
  reads: 0, lookups: 0, inventories: 0, spawns: [], polls: 0, sleeps: 0,
  elapsed: 0, fetches: [], sockets: [], removals: [], cleanupChecks: 0,
  stderrEncoding: [], exitEmissions: 0, outcome: 'not_completed',
};
function forbidden() {
  console.error('UNEXPECTED_STARTUP_EFFECT');
  process.exit(99);
}
const proc = new EventEmitter();
proc.pid = undefined; // The real cleanup must never kill a host process.
proc.exitCode = null;
proc.signalCode = null;
proc.stderr = new EventEmitter();
proc.stderr.setEncoding = encoding => { ports.stderrEncoding.push(encoding); };
let profile;
const devtoolsPort = 49876;
const websocketUrl = `ws://127.0.0.1:${devtoolsPort}/devtools/page/owned-blank`;
const reachedWebSocket = new Error('OWNED_WEBSOCKET_SENTINEL');
for (const name of [
  'readFile', 'statSync', 'stat', 'mkdirSync', 'mkdir', 'mkdtempSync', 'mkdtemp',
  'writeFileSync', 'writeFile', 'appendFileSync', 'appendFile', 'openSync', 'open',
  'createWriteStream', 'createReadStream', 'unlinkSync', 'unlink', 'renameSync',
  'rename', 'copyFileSync', 'copyFile', 'rmdirSync', 'rmdir', 'rm',
]) fs[name] = forbidden;
fs.readFileSync = (path, encoding) => {
  if (String(path) === input.script && encoding === 'utf8' && ports.reads++ === 0) {
    return realRead(path, encoding);
  }
  if (!profile || String(path) !== `${profile}/DevToolsActivePort` || encoding !== 'utf8') {
    return forbidden();
  }
  ports.polls++;
  if (input.mode === 'ready_after_75' && ports.polls > 75) {
    return `${devtoolsPort}\n/devtools/browser/owned\n`;
  }
  // Synthetic absence at an inert port: no real missing/denied path is read.
  throw Object.assign(new Error('owned DevTools port not ready'), { code: 'ENOENT' });
};
fs.existsSync = path => {
  if (String(path) === input.browser && ports.lookups++ === 0) return true;
  if (profile && String(path) === profile) { ports.cleanupChecks++; return false; }
  return forbidden();
};
fs.readdirSync = path => {
  if (String(path) !== tmpdir() || ports.inventories++ !== 0) return forbidden();
  return []; // Never enumerate the real temp directory or stale profiles.
};
fs.rmSync = (path, options) => {
  if (!profile || String(path) !== profile) return forbidden();
  ports.removals.push({ path: String(path), options }); // No real deletion.
};
for (const name of ['spawnSync', 'exec', 'execSync', 'execFile', 'execFileSync', 'fork']) {
  childProcess[name] = forbidden;
}
childProcess.spawn = (executable, args, options) => {
  if (executable !== input.browser || ports.spawns.length || !Array.isArray(args)) return forbidden();
  const profiles = args.filter(value => value.startsWith('--user-data-dir='));
  if (profiles.length !== 1) return forbidden();
  profile = profiles[0].slice('--user-data-dir='.length);
  if (!profile.startsWith(`${tmpdir()}/auradream-browser-${process.pid}-`)) return forbidden();
  ports.spawns.push({ executable, args, options, profile });
  return proc;
};
process.kill = forbidden;
syncBuiltinESMExports();
const started = 1700000000000;
Date.now = () => started + ports.elapsed;
globalThis.setTimeout = (callback, delay, ...args) => {
  if (typeof callback !== 'function' || delay !== 200 || args.length || ports.sleeps >= 200) {
    return forbidden();
  }
  ports.sleeps++;
  ports.elapsed += delay;
  Promise.resolve().then(() => {
    if (input.mode === 'early_exit' && ports.sleeps === 1) {
      proc.exitCode = 7;
      ports.exitEmissions++;
      proc.emit('exit', 7, null); // Exercise the real installed exit listener.
    }
    callback();
  });
  return ports.sleeps;
};
globalThis.clearTimeout = forbidden; // CDP timers must never be reached.
globalThis.fetch = async url => {
  if (input.mode !== 'ready_after_75' || ports.polls <= 75
      || url !== `http://127.0.0.1:${devtoolsPort}/json`) return forbidden();
  ports.fetches.push(url);
  return { json: async () => [
    { type: 'service_worker', url: 'about:blank', webSocketDebuggerUrl: 'must-not-select-worker' },
    { type: 'page', url: 'chrome://sync/', webSocketDebuggerUrl: 'must-not-select-sync' },
    { type: 'page', url: 'about:blank', webSocketDebuggerUrl: websocketUrl },
  ] };
};
globalThis.WebSocket = class {
  constructor(url) {
    if (url !== websocketUrl || ports.fetches.length !== 1) return forbidden();
    ports.sockets.push(url);
    throw reachedWebSocket; // Stop actual module execution before any CDP work.
  }
};
process.argv = [process.execPath, input.harness, input.script];
const probes = {
  source_read: () => fs.readFileSync(input.harness, 'utf8'),
  output_write: () => fs.writeFileSync(input.output, 'must-not-write'),
  output_mkdir: () => fs.mkdirSync(input.output),
  inventory: () => fs.readdirSync(input.output),
  spawn: () => childProcess.spawn('must-not-run'),
  kill: () => process.kill(12345),
  fetch: () => globalThis.fetch('http://must-not-connect.invalid'),
  websocket: () => new globalThis.WebSocket('ws://must-not-connect.invalid'),
};
if (input.probe) probes[input.probe]();
try {
  await import(harnessModule);
  ports.outcome = 'unexpected_completion';
  process.exitCode = 1;
} catch (error) {
  if (error === reachedWebSocket) ports.outcome = 'reached_websocket';
  else if (error.message === 'the browser did not come up') ports.outcome = 'startup_timeout';
  else if (error.message === 'the browser exited before DevTools was ready') ports.outcome = 'early_exit';
  else { ports.outcome = 'unexpected_error:' + error.name; process.exitCode = 1; }
}
// The actual module registered cleanup first. Report after that exit handler
// has run, so a missing cleanup is observable rather than invoked by the test.
process.on('exit', () => console.log('STARTUP_PORTS:' + JSON.stringify(ports)));
'''
    harness = ROOT / "tools" / "browser.mjs"
    request = dict(harness=str(harness), url=harness.as_uri(), script=str(script),
                   temporary=str(temporary), output=str(output),
                   browser="owned-synthetic-browser", mode=mode, probe=probe)
    environment = {**os.environ, "AURADREAM_BROWSER": request["browser"],
                   "AURADREAM_SCREENSHOT_DIR": str(output), "AURADREAM_PORT_OFFSET": "0",
                   "AURADREAM_VIEWPORT": "1400x900", "AURADREAM_DPR": "1",
                   "AURADREAM_BROWSER_DIAGNOSTICS": "1",
                   "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    done = subprocess.run(["node", "--input-type=module", "-e", bootstrap, json.dumps(request)],
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
        "recursive": True, "force": True, "maxRetries": 20, "retryDelay": 250,
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
