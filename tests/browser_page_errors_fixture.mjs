// Fixture for tests/test_browser_page_errors.py, run as `node <this file> <json>`. It is a
// file rather than `node -e` source because antivirus heuristics flag a long
// inline script that patches child_process (Defender: Trojan:Win32/SuspExec.SE).
//
// A browser that is only a scripted DevTools socket: no process, no network.
// Every call is answered at once; a Runtime.evaluate whose expression is a
// key of `input.script` first delivers that entry's `events` (protocol
// messages with no id, as the browser sends them) and answers with its
// `reply` (default: the value true).
import fs from 'node:fs';
import childProcess from 'node:child_process';
import {EventEmitter} from 'node:events';
import {syncBuiltinESMExports} from 'node:module';
const input = JSON.parse(process.argv[2]);
const exists = fs.existsSync;
fs.existsSync = (path) => String(path) === 'owned-synthetic-browser' || exists(path);
const read = fs.readFileSync;
fs.readFileSync = (path, ...rest) => String(path).endsWith('/DevToolsActivePort') ? '9\n/devtools/browser/fixture' : read(path, ...rest);
const child = new EventEmitter();
child.exitCode = null;
child.signalCode = null;
childProcess.spawn = () => child; // No pid: there is no process to kill or inspect.
childProcess.spawnSync = () => { throw Error('unexpected process'); };
syncBuiltinESMExports();
globalThis.fetch = async (url) => {
  if (url !== 'http://127.0.0.1:9/json') throw Error('unexpected network');
  return {json: async () => [{type: 'page', url: 'about:blank', webSocketDebuggerUrl: 'ws://fixture.invalid/page'}]};
};
globalThis.WebSocket = class {
  constructor(url) {
    if (url !== 'ws://fixture.invalid/page') throw Error('unexpected socket');
    setTimeout(() => this.onopen?.(), 0);
  }
  send(text) {
    const call = JSON.parse(text);
    if (call.method === 'Browser.close') { setTimeout(() => child.emit('exit', 0), 0); return; }
    const entry = call.method === 'Runtime.evaluate' ? input.script[call.params.expression] : undefined;
    setTimeout(() => {
      for (const event of entry?.events ?? []) this.onmessage({data: JSON.stringify(event)});
      this.onmessage({data: JSON.stringify({id: call.id, result: entry?.reply ?? {result: {value: true}}})});
    }, 0);
  }
  close() {}
};
process.argv = [process.execPath, input.harness, input.steps];
await import(input.url);
