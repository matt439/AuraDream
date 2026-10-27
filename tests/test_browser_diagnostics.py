"""Opt-in owned-browser diagnostics are bounded; no browser or network runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def test_failed_owned_browser_reports_bounded_stderr_tail(tmp_path):
    node = shutil.which("node")
    assert node is not None, "Node is required for the browser harness"
    steps = tmp_path / "steps.json"
    steps.write_text('[{"expect":"true"}]', encoding="utf-8")
    bootstrap = r'''
import fs from 'node:fs';
import childProcess from 'node:child_process';
import {EventEmitter} from 'node:events';
import {PassThrough} from 'node:stream';
import {syncBuiltinESMExports} from 'node:module';
const input = JSON.parse(process.argv[1]);
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
'''
    request = dict(harness=str(ROOT / "tools/browser.mjs"), steps=str(steps),
                   url=(ROOT / "tools/browser.mjs").as_uri())
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_BROWSER_DIAGNOSTICS": "1",
                   "TEMP": str(tmp_path), "TMP": str(tmp_path), "TMPDIR": str(tmp_path)}
    done = subprocess.run([node, "--input-type=module", "-e", bootstrap, json.dumps(request)],
                          cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "the browser exited before DevTools was ready" in done.stderr
    assert "browser diagnostics: exit=7 signal=none" in done.stderr
    tail = done.stderr.split("browser stderr tail:\n", 1)[1].split("\nbrowser stderr tail [end]", 1)[0]
    assert len(tail) == 16384 and tail.endswith("final-owned-stderr")
    assert "clipped-fixture-prefix" not in done.stderr
    assert "unexpected network" not in done.stderr and "unexpected socket" not in done.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["steps.json"]
