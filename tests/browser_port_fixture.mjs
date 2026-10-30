// Fixture for tests/test_browser_port_preflight.py, run as `node <this file> <json>`. It is a
// file rather than `node -e` source because antivirus heuristics flag a long
// inline script that patches child_process (Defender: Trojan:Win32/SuspExec.SE).
import fs from 'node:fs';
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
const input = JSON.parse(process.argv[2]);
const asModule = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const realRead = fs.readFileSync;
const outputModule = asModule(realRead(new URL('./browser-output.mjs', input.url), 'utf8'));
const source = realRead(input.harness, 'utf8');
const dependency = "'./browser-output.mjs'";
if (source.split(dependency).length !== 2) throw Error('expected one browser-output dependency');
const harnessModule = asModule(source.replace(dependency, JSON.stringify(outputModule)));
function forbidden() { console.error('UNEXPECTED_PORT_PREFLIGHT_EFFECT'); process.exit(99); }
let reads = 0;
fs.readFileSync = (path, ...args) => String(path) === input.script && reads++ === 0
  ? realRead(path, ...args) : forbidden();
for (const name of ['existsSync', 'readdirSync', 'statSync', 'mkdirSync', 'mkdtempSync', 'rmSync', 'writeFileSync']) {
  fs[name] = forbidden;
}
childProcess.spawn = childProcess.spawnSync = forbidden;
globalThis.fetch = forbidden;
globalThis.WebSocket = forbidden;
process.argv = [process.execPath, input.harness, input.script];
let parsedSteps;
const realParse = JSON.parse;
JSON.parse = (...args) => { parsedSteps = realParse(...args); return parsedSteps; };
if (!input.probe) {
  // A valid fixture reaches this first existing boundary. Observe only its
  // already-parsed data; no executable/browser/output/network is accessed.
  fs.existsSync = () => {
    console.error('STOP_BROWSER_LOOKUP:' + JSON.stringify(parsedSteps));
    process.exit(97);
  };
}
syncBuiltinESMExports();
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
