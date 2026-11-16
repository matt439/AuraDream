"""The supplied-keyboard fixture waits for its owners before intentional focus."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.usefixtures("node")


def test_keyboard_idle_drains_reads_redraws_viewport_and_changing_geometry():
    # Execute the real injected helper with controlled frames. No browser,
    # server, timer or source rewrite: each unfinished owner must keep it held.
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let now = 0, height = 300;
const frames = [], select = {}, entry = {signature: 'old'};
const dialog = {
  id: 'setups-dialog', open: true, scrollTop: 0, scrollHeight: 900,
  contains: (node) => node === select,
  querySelector: () => ({}),
  getBoundingClientRect: () => ({height, toJSON: () => ({height})}),
};
const context = {
  Date: {now: () => now}, requestAnimationFrame: (fn) => frames.push(fn),
  document: {activeElement: {id: 'setups-close'}},
  getComputedStyle: () => ({position: 'static'}),
  isPhone: () => true, backGoing: false, backDepth: 1,
  sheetOpened: {dialog}, sheetViewportFrame: 1,
  setupsReady: false, setupsPending: 'read',
  phoneLists: new Map([[select, entry]]), phoneListSignature: () => 'new',
};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
context.__keyboardMetrics = [48, 69];
let done = false;
const waiting = context.__keyboardIdle(dialog, true).then(() => {done = true;});
async function frame() {
  assert.equal(frames.length, 1);
  now += 16;
  frames.shift()(now);
  await Promise.resolve();
  await Promise.resolve();
}
async function held() {
  for (let i = 0; i < 3; i++) await frame();
  assert.equal(done, false);
}
(async () => {
  await held(); // unfinished list read
  context.setupsReady = true; context.setupsPending = null;
  await held(); // phone rows have not caught up with the list
  entry.signature = 'new';
  await held(); // a reveal frame is still queued
  context.sheetViewportFrame = 0;
  await held(); // rendered height is still the preceding viewport's
  height = 48;
  for (let i = 0; i < 3; i++) {
    dialog.scrollTop += 1;
    await frame();
    assert.equal(done, false); // fit alone is insufficient during layout
  }
  await frame();
  assert.equal(done, false);
  await frame();
  await waiting;
  assert.equal(done, true);
})().catch((error) => {console.error(error); process.exitCode = 1;});
"""
    done = subprocess.run(
        ["node", "-e", script, str(ROOT / "tools" / "phone_keyboard_audit.js")],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stdout + done.stderr
