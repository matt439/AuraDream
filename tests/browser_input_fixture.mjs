// Fixture for tests/test_browser_input_refusals.py, run as `node <this file> <json>`. It is a
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
