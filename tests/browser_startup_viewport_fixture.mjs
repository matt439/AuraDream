// Fixture for tests/test_browser_startup_viewport.py, run as `node <this file> <json>`. It is a
// file rather than `node -e` source because antivirus heuristics flag a long
// inline script that patches child_process (Defender: Trojan:Win32/SuspExec.SE).
import fs from 'node:fs';
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
const input = JSON.parse(process.argv[2]);
const asModule = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const realRead = fs.readFileSync;
const outputModule = asModule(realRead(new URL('./browser-output.mjs', input.url), 'utf8'));
let source = realRead(input.harness, 'utf8');
const dependency = "'./browser-output.mjs'";
if (source.split(dependency).length !== 2) throw Error('expected one browser-output dependency');
source = source.replace(dependency, JSON.stringify(outputModule));
if (input.capture) {
  // Observe actual parsed module locals after all preflight, before the first
  // existing lookup. The predicates and parsed expressions are not replaced.
  const boundary = 'const browser = CANDIDATES.find((p) => existsSync(p));';
  if (source.split(boundary).length !== 2) throw Error('expected one discovery boundary');
  source = source.replace(boundary, 'globalThis.__captureStartupMetrics(VW, VH, DPR);\n' + boundary);
}
const harnessModule = asModule(source);
function forbidden(name) {
  return function () { console.error('UNEXPECTED_STARTUP_VIEWPORT_EFFECT:' + name); process.exit(99); };
}
let reads = 0;
fs.readFileSync = (path, ...args) => String(path) === input.script && reads++ === 0
  ? realRead(path, ...args) : forbidden('source_read')();
for (const [name, label] of Object.entries({
  existsSync: 'browser_lookup', readdirSync: 'profile_scan', statSync: 'profile_stat',
  mkdirSync: 'output_mkdir', mkdtempSync: 'profile_create', rmSync: 'profile_remove',
  writeFileSync: 'output_write',
})) fs[name] = forbidden(label);
childProcess.spawn = childProcess.spawnSync = forbidden('spawn');
syncBuiltinESMExports();
globalThis.fetch = forbidden('fetch');
globalThis.WebSocket = forbidden('websocket');
globalThis.__captureStartupMetrics = (width, height, deviceScaleFactor) => {
  console.error('STARTUP_METRICS:' + JSON.stringify({ width, height, deviceScaleFactor }));
  process.exit(97);
};
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
