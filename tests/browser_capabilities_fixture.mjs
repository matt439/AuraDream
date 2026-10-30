// Fixture for tests/test_browser_capabilities.py, run as `node <this file> <json>`. It is a
// file rather than `node -e` source because antivirus heuristics flag a long
// inline script that patches child_process (Defender: Trojan:Win32/SuspExec.SE).
import fs from 'node:fs';
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
const input = JSON.parse(process.argv[2]);
// Node 22's ESM loader itself calls fs.readFileSync. Capture both real source
// files before instrumenting application effects, then load them as data URLs.
// Only the local dependency specifier changes; the real preflight/body executes.
const asModule = (source) => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const outputModule = asModule(fs.readFileSync(new URL('./browser-output.mjs', input.url), 'utf8'));
const source = fs.readFileSync(input.harness, 'utf8');
const realRead = fs.readFileSync;
const dependency = "'./browser-output.mjs'";
if (source.split(dependency).length !== 2) throw Error('expected one browser-output dependency');
const harnessModule = asModule(source.replace(dependency, JSON.stringify(outputModule)));
function forbidden() { console.error('UNEXPECTED_DOWNSTREAM_EFFECT'); process.exit(99); }
for (const name of ['readFileSync', 'readdirSync', 'statSync', 'mkdirSync', 'mkdtempSync', 'rmSync', 'writeFileSync']) {
  fs[name] = forbidden;
}
// Supported APIs now permit exactly the script read needed for viewport
// preflight before browser lookup. Explicit poison probes remain forbidden.
let inputReads = 0;
fs.readFileSync = (path, ...args) => input.supported && !input.probe
  && String(path) === input.script && inputReads++ === 0 ? realRead(path, ...args) : forbidden();
fs.existsSync = input.supported ? () => false : forbidden;
childProcess.spawn = childProcess.spawnSync = forbidden;
syncBuiltinESMExports();
if (input.supported) {
  // Exercise the callable-supported branch without depending on this host's
  // Node release. Invoking either API would be an unexpected network action.
  globalThis.fetch = forbidden;
  globalThis.WebSocket = forbidden;
} else {
  for (const name of ['fetch', 'WebSocket']) {
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value: forbidden });
  }
  for (const [name, mask] of Object.entries(input.masks)) {
    if (mask.remove) delete globalThis[name];
    else globalThis[name] = mask.value;
  }
}
process.argv = [process.execPath, input.harness, input.script];
const probes = {
  input_read: () => fs.readFileSync(input.script),
  module_read: () => fs.readFileSync(input.harness),
  output_write: () => fs.writeFileSync(input.output, 'must-not-write'),
  spawn: () => childProcess.spawn('must-not-run'),
  fetch: () => globalThis.fetch('http://must-not-connect.invalid'),
  websocket: () => new globalThis.WebSocket('ws://must-not-connect.invalid'),
};
if (input.probe) probes[input.probe]();
await import(harnessModule);
