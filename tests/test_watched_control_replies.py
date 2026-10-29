"""Deferred watched-action replies cannot write over a successor or newer press.

These exercise the actual classic-script functions with inert presentation ports.
They check ownership/order only; the fake-browser smoke still qualifies the UI.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = process.argv[1];
const scenario = process.argv[2];

function button(text) {
  const attributes = new Map([['aria-disabled', 'false']]);
  return { textContent: text, title: '', getAttribute: key => attributes.get(key),
    setAttribute: (key, value) => attributes.set(key, String(value)) };
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function fixture() {
  const calls = { errors: [], progress: [], mirrors: 0, cells: 0, requests: [] };
  const sandbox = vm.createContext({
    busy: true, currentJob: 'first', serverStartedAt: 10,
    document: { title: 'AuraDream' },
    els: { cancel: button('Cancel'), pause: button('Pause') },
    cancelledHere: new Set(),
    api: (url, options) => {
      const request = { ...deferred(), url, options };
      calls.requests.push(request);
      return request.promise;
    },
    showError: message => calls.errors.push(message),
    showJobNote: message => calls.errors.push(message),
    unitWord: () => 'image',
    calls,
  });
  for (const name of ['job-progress.js', 'job-stream.js']) {
    new vm.Script(fs.readFileSync(path.join(root, 'web', name), 'utf8'), { filename: name })
      .runInContext(sandbox);
  }
  // These are presentation dependencies, outside the asynchronous functions
  // under test. No page boot, real DOM, network or server process is executed.
  vm.runInContext(`
    setProgressText = text => calls.progress.push(text);
    mirrorJob = () => { calls.mirrors++; };
    showCell = () => { calls.cells++; };
  `, sandbox);
  return {
    sandbox, calls,
    cancel: () => vm.runInContext('cancelJob()', sandbox),
    pause: () => vm.runInContext('togglePause()', sandbox),
    asking: () => vm.runInContext('pauseAsking', sandbox),
    // This is setBusy's observed boundary, not a replacement implementation:
    // the successor owns new progress/control slots while old POSTs can wait.
    successor(id = 'second') {
      sandbox.currentJob = id;
      sandbox.busy = true;
      sandbox.els.cancel.textContent = 'Cancel';
      sandbox.els.cancel.setAttribute('aria-disabled', 'false');
      sandbox.els.pause.textContent = 'Pause';
      sandbox.els.pause.setAttribute('aria-disabled', 'false');
    },
  };
}
const held = (f, control) => f.sandbox.els[control].getAttribute('aria-disabled');

async function main() {
  const f = fixture();
  const { sandbox: s, calls: c } = f;
  if (scenario === 'cancel-current-success') {
    const work = f.cancel();
    assert.equal(held(f, 'cancel'), 'true');
    assert.equal(s.els.cancel.textContent, 'Stopping\u2026');
    assert.equal(c.requests[0].url, '/api/jobs/first/cancel');
    assert.equal(c.requests[0].options.method, 'POST');
    c.requests[0].resolve({});
    await work;
    assert.deepEqual(c.progress, ['finishing this step\u2026']);
    assert.equal(c.mirrors, 1);
    assert.equal(s.cancelledHere.has('first'), true);
  } else if (scenario === 'cancel-current-error') {
    const work = f.cancel();
    c.requests[0].reject(Error('current cancel failed'));
    await work;
    assert.equal(held(f, 'cancel'), 'false');
    assert.equal(s.els.cancel.textContent, 'Cancel');
    assert.deepEqual(c.errors, ['current cancel failed']);
    assert.equal(s.cancelledHere.has('first'), false);
  } else if (scenario === 'cancel-duplicate-held') {
    const work = f.cancel();
    await f.cancel();
    assert.equal(c.requests.length, 1);
    c.requests[0].resolve({});
    await work;
  } else if (scenario.startsWith('cancel-successor-')) {
    const old = f.cancel();
    f.successor();
    const newer = f.cancel();
    c.errors.push('newer job message');
    c.progress.push('newer job progress');
    if (scenario.endsWith('error')) c.requests[0].reject(Error('old cancel failed'));
    else c.requests[0].resolve({});
    await old;
    assert.equal(s.els.cancel.textContent, 'Stopping\u2026');
    assert.equal(held(f, 'cancel'), 'true');
    assert.deepEqual(c.progress, ['newer job progress']);
    assert.deepEqual(c.errors, ['newer job message']);
    assert.equal(c.mirrors, 0);
    assert.equal(s.cancelledHere.has('second'), true);
    c.requests[1].resolve({});
    await newer;
    assert.equal(c.requests.length, 2);
  } else if (scenario.startsWith('cancel-idle-')) {
    const old = f.cancel();
    s.currentJob = null;
    s.busy = false;
    s.els.cancel.textContent = 'Cancel';
    c.progress.push('finished successor');
    if (scenario.endsWith('error')) c.requests[0].reject(Error('old cancel failed'));
    else c.requests[0].resolve({});
    await old;
    assert.deepEqual(c.progress, ['finished successor']);
    assert.deepEqual(c.errors, []);
    assert.equal(s.els.cancel.textContent, 'Cancel');
    assert.equal(c.mirrors, 0);
  } else if (scenario.startsWith('cancel-restart-')) {
    const old = f.cancel();
    s.serverStartedAt = 20;
    c.progress.push('new server progress');
    if (scenario.endsWith('error')) c.requests[0].reject(Error('old server refused'));
    else c.requests[0].resolve({});
    await old;
    assert.deepEqual(c.progress, ['new server progress']);
    assert.deepEqual(c.errors, []);
    assert.equal(held(f, 'cancel'), 'true');
    assert.equal(c.mirrors, 0);
  } else if (scenario === 'cancel-newer-same-id-settled') {
    const old = f.cancel();
    s.serverStartedAt = 20;
    f.successor('first');
    const newer = f.cancel();
    c.requests[1].resolve({});
    await newer;
    c.requests[0].reject(Error('old server refused'));
    await old;
    assert.equal(s.cancelledHere.has('first'), true);
    assert.deepEqual(c.errors, []);
    assert.equal(held(f, 'cancel'), 'true');
    assert.equal(c.mirrors, 1);
  } else if (scenario === 'cancel-first-known-server') {
    s.serverStartedAt = null;
    const work = f.cancel();
    s.serverStartedAt = 10;
    c.requests[0].resolve({});
    await work;
    assert.deepEqual(c.progress, ['finishing this step\u2026']);
  } else if (scenario === 'pause-current-error') {
    const work = f.pause();
    assert.equal(s.els.pause.textContent, 'Pausing\u2026');
    assert.equal(c.requests[0].url, '/api/jobs/first/pause?value=true');
    c.requests[0].reject(Error('current pause failed'));
    await work;
    assert.equal(s.els.pause.textContent, 'Pause');
    assert.deepEqual(c.errors, ['current pause failed']);
    assert.equal(f.asking(), null);
    assert.equal(c.mirrors, 2);
  } else if (scenario === 'pause-current-success') {
    const work = f.pause();
    c.requests[0].resolve({});
    await work;
    assert.equal(f.asking(), null);
    assert.equal(s.els.pause.textContent, 'Pausing\u2026');
    assert.deepEqual(c.errors, []);
    assert.equal(c.mirrors, 2);
  } else if (scenario === 'pause-successor-error') {
    const old = f.pause();
    f.successor();
    const newer = f.pause();
    const owner = f.asking();
    const mirrors = c.mirrors;
    c.errors.push('newer job message');
    c.requests[0].reject(Error('old pause failed'));
    await old;
    assert.equal(f.asking(), owner);
    assert.equal(s.els.pause.textContent, 'Pausing\u2026');
    assert.deepEqual(c.errors, ['newer job message']);
    assert.equal(c.mirrors, mirrors);
    c.requests[1].resolve({});
    await newer;
    assert.equal(f.asking(), null);
  } else if (scenario === 'pause-newer-pending' || scenario === 'pause-newer-settled') {
    const old = f.pause();
    const newer = f.pause();
    const owner = f.asking();
    assert.equal(owner.value, false);
    assert.equal(c.requests[1].url, '/api/jobs/first/pause?value=false');
    if (scenario.endsWith('settled')) {
      c.requests[1].resolve({});
      await newer;
    }
    const mirrors = c.mirrors;
    c.requests[0].reject(Error('old pause failed'));
    await old;
    assert.equal(s.els.pause.textContent, 'Pause');
    assert.deepEqual(c.errors, []);
    assert.equal(c.mirrors, mirrors);
    assert.equal(f.asking(), scenario.endsWith('settled') ? null : owner);
    if (scenario.endsWith('pending')) {
      c.requests[1].resolve({});
      await newer;
    }
  } else if (scenario === 'pause-restart-error') {
    const old = f.pause();
    s.serverStartedAt = 20;
    s.els.pause.textContent = 'Resume';
    const mirrors = c.mirrors;
    c.requests[0].reject(Error('old server refused'));
    await old;
    assert.equal(s.els.pause.textContent, 'Resume');
    assert.deepEqual(c.errors, []);
    assert.equal(c.mirrors, mirrors);
    assert.equal(f.asking(), null);
  } else if (scenario === 'pause-restart-snapshot') {
    const old = f.pause();
    s.serverStartedAt = 20;
    vm.runInContext(`syncPause({ job_id: 'first', status: 'queued', pause_requested: false, paused: false })`, s);
    assert.equal(s.els.pause.textContent, 'Pause');
    c.requests[0].resolve({});
    await old;
  } else if (scenario === 'held-and-idle-refuse') {
    s.currentJob = null;
    s.busy = false;
    await f.cancel();
    await f.pause();
    f.successor();
    s.els.cancel.setAttribute('aria-disabled', 'true');
    s.els.pause.setAttribute('aria-disabled', 'true');
    s.els.pause.title = 'No pause point';
    await f.cancel();
    await f.pause();
    assert.equal(c.requests.length, 0);
    assert.deepEqual(c.errors, ['No pause point.']);
  } else throw Error('unknown scenario: ' + scenario);
  console.log('passed: ' + scenario);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("scenario", [
    "cancel-current-success", "cancel-current-error", "cancel-duplicate-held",
    "cancel-successor-success", "cancel-successor-error",
    "cancel-idle-success", "cancel-idle-error",
    "cancel-restart-success", "cancel-restart-error",
    "cancel-newer-same-id-settled", "cancel-first-known-server",
    "pause-current-success", "pause-current-error", "pause-successor-error",
    "pause-newer-pending", "pause-newer-settled", "pause-restart-error",
    "pause-restart-snapshot", "held-and-idle-refuse",
])
def test_watched_action_reply_ownership(scenario: str):
    node = shutil.which("node")
    assert node is not None, "Node is required for the watched-action ownership checks"
    done = subprocess.run(
        [node, "-e", SCRIPT, str(ROOT), scenario], capture_output=True,
        text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"passed: {scenario}" in done.stdout
