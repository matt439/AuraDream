// Fixture for tests/test_browser_diagnostics.py, run as `node <this file> <json>`. It is a
// file rather than `node -e` source because antivirus heuristics flag a long
// inline script that patches child_process (Defender: Trojan:Win32/SuspExec.SE).
import fs from 'node:fs';
import childProcess from 'node:child_process';
import {EventEmitter} from 'node:events';
import {PassThrough} from 'node:stream';
import {syncBuiltinESMExports} from 'node:module';
const input = JSON.parse(process.argv[2]);
const exists = fs.existsSync;
fs.existsSync = (path) => String(path) === 'owned-synthetic-browser' || exists(path);
childProcess.spawn = (_browser, _args, options) => {
  if (JSON.stringify(options.stdio) !== JSON.stringify(['ignore', 'ignore', 'pipe'])) throw Error('stderr not piped');
  const child = new EventEmitter();
  child.stderr = new PassThrough();
  child.exitCode = null;
  child.signalCode = null;
  setTimeout(() => {
    child.stderr.end('clipped-fixture-prefix' + 'x'.repeat(40000) + 'final-owned-stderr');
    child.exitCode = 7;
    child.emit('exit', 7);
  }, 0);
  return child; // No pid: there is no process to kill or inspect.
};
syncBuiltinESMExports();
globalThis.fetch = () => { throw Error('unexpected network'); };
globalThis.WebSocket = function () { throw Error('unexpected socket'); };
process.argv = [process.execPath, input.harness, input.steps];
await import(input.url);
