// Fixture for tests/test_browser_startup_wait.py: runs tools/browser.mjs's real
// startup with every effect replaced by an owned stand-in. It is a file rather
// than `node -e` source because antivirus heuristics flag a long inline script
// that patches child_process (Defender: Trojan:Win32/SuspExec.SE).
import fs from 'node:fs';
import childProcess from 'node:child_process';
import { EventEmitter } from 'node:events';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
const input = JSON.parse(process.argv[2]);
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
