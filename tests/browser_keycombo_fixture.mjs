// Fixture for tests/test_browser_keycombo_preflight.py, run as `node <this file> <json>`. It is a
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
