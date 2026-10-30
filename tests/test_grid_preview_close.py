"""Closing Grid retires its pending preview, with real CFG presentation logic.

The actual declaration-only modules run on inert DOM/API ports. Native toggle,
focus, geometry and actual fake-preview responses belong to the separate smoke.
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
const root = process.argv[1], scenario = process.argv[2];
function element(value = '') {
  const classes = new Set();
  return { value, disabled: false, hidden: false, checked: false, title: '',
    textContent: '', children: [],
    classList: { add: name => classes.add(name), remove: name => classes.delete(name),
      contains: name => classes.has(name), toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
    replaceChildren(...items) {
      this.children = items;
      this.textContent = items.map(item => typeof item === 'string' ? item : item.textContent).join('');
    } };
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function fixture() {
  const calls = { requests: [], reveals: 0, analysis: 0, toggles: new Map() };
  const els = {};
  for (const name of ['prompt', 'negative', 'cfg', 'trueCfg', 'trueCfgField', 'negField', 'negNote', 'negEnable',
    'gridPreview', 'gridPrompt', 'gridPromptMode', 'gridSteps', 'gridSize', 'gridCfg', 'gridMu',
    'gridLoraScale', 'gridSeeds', 'gridSeedList', 'gridResume', 'gridBudget', 'runs', 'lora']) els[name] = element();
  Object.assign(els.prompt, { value: 'a {coat} fox' });
  els.negative.value = '  keep this negative  ';
  els.cfg.value = '0'; els.trueCfg.value = ''; els.runs.value = '1';
  els.gridPrompt.value = 'coat = red | blue'; els.gridPromptMode.value = 'cross';
  els.gridCfg.value = '0,2'; els.gridSeeds.value = '1';
  els.gridSection = { open: true, addEventListener: (event, handler) => calls.toggles.set(event, handler) };
  const sandbox = vm.createContext({ els, calls, trueCfgSupported: false, negativeSupported: true, cfgThreshold: 0,
    diskFreeMb: null, busy: false, elsewhereBusy: false, queuedIds: [], held: false,
    GRID_PLACEHOLDER: /\{[A-Za-z_]\w*\}/,
    document: { createElement: () => element() },
    collect: () => ({ prompt: els.prompt.value, negative_prompt: els.negative.value, num_images: 1,
      width: 256, height: 256, steps: 8, guidance_scale: +els.cfg.value, seed: 42 }),
    thenRefinePayload: () => null, modelChangeHold: () => sandbox.held,
    syncAnalysisStale: () => { calls.analysis++; }, revealGridRun: () => { calls.reveals++; },
    hideNegativeSuggestions: () => {},
    muStatic: () => null, humanBytes: value => String(value),
    api: (url, options) => { const request = { ...deferred(), url, options }; calls.requests.push(request); return request.promise; },
  });
  for (const name of ['sizes.js', 'grid-preview.js']) {
    new vm.Script(fs.readFileSync(path.join(root, 'web', name), 'utf8'), { filename: name }).runInContext(sandbox);
  }
  const wiring = "els.gridSection.addEventListener('toggle', previewGrid);";
  const boot = fs.readFileSync(path.join(root, 'web', 'app.js'), 'utf8');
  assert.equal(boot.split(wiring).length - 1, 1, 'actual boot must wire the real preview function to toggle');
  vm.runInContext(wiring, sandbox); // Capture that exact registration, without executing page boot.
  vm.runInContext('syncCfg()', sandbox);
  const evaluate = text => vm.runInContext(text, sandbox);
  const snapshot = () => JSON.stringify({ reach: evaluate('gridMaxCfg'), negative: els.negative.disabled,
    note: els.negNote.textContent, text: els.gridPreview.textContent,
    warning: els.gridPreview.classList.contains('warn'), reveals: calls.reveals,
    raw: [els.prompt.value, els.negative.value, els.gridPrompt.value, els.gridCfg.value] });
  return { sandbox, els, calls, evaluate, snapshot,
    toggle(open) { els.gridSection.open = open; return calls.toggles.get('toggle')({ type: 'toggle' }); },
    preview: () => evaluate('previewGrid()'),
    reply(index, max = 2) { calls.requests[index].resolve({ images: 4, summary: '2 prompts x 2 CFG x 1 seed',
      values: { prompt: ['a red fox', 'a blue fox'], guidance_scale: [0, max], steps: [8], mu: [1.15], seed: [42] },
      skipped: 0, queued: 0, duplicates: 0, in_gallery: 0, seconds: null, bytes: null }); },
  };
}
function closed(f) {
  assert.equal(f.evaluate('gridMaxCfg'), null);
  assert.equal(f.els.negative.disabled, true);
  assert.equal(f.els.negField.classList.contains('disabled'), true);
  assert.equal(f.els.negNote.textContent.includes('for grid cells'), false);
  assert.equal(f.evaluate('negativeSent(els.negative)'), null);
}
async function main() {
  const f = fixture();
  if (scenario === 'late-success-after-close' || scenario === 'late-error-after-close') {
    const old = f.preview();
    await f.toggle(false);
    closed(f);
    const before = f.snapshot();
    if (scenario.startsWith('late-error')) f.calls.requests[0].reject(Error('obsolete preview error'));
    else f.reply(0);
    await old;
    closed(f);
    assert.equal(f.snapshot(), before, 'obsolete reply changed closed composer/preview state');
  } else if (scenario === 'two-previews-and-repeated-close') {
    const first = f.preview(), second = f.preview();
    await f.toggle(false); await f.toggle(false);
    const before = f.snapshot();
    f.reply(1); await second; f.reply(0, 7); await first;
    closed(f);
    assert.equal(f.snapshot(), before);
  } else if (scenario === 'reopen-current-before-old' || scenario === 'reopen-old-before-current') {
    const old = f.preview();
    await f.toggle(false);
    const current = f.toggle(true);
    if (scenario.endsWith('old-before-current')) { f.reply(0, 7); await old; closed(f); }
    f.reply(1, 1); await current;
    const before = f.snapshot();
    if (scenario.endsWith('current-before-old')) { f.reply(0, 7); await old; }
    assert.equal(f.snapshot(), before);
    assert.equal(f.evaluate('gridMaxCfg'), 1);
    assert.equal(f.els.negative.disabled, false);
  } else if (scenario === 'current-success-keeps-raw-input-and-payload') {
    const raw = [f.els.prompt.value, f.els.negative.value, f.els.gridPrompt.value, f.els.gridCfg.value];
    const work = f.preview();
    assert.equal(f.calls.requests[0].url, '/api/grid/preview');
    assert.equal(f.calls.requests[0].options.method, 'POST');
    const body = JSON.parse(f.calls.requests[0].options.body);
    assert.equal(body.prompt_axis, 'coat = red | blue'); assert.equal(body.cfg_axis, '0,2'); assert.equal(body.runs, 1);
    f.reply(0); await work;
    assert.equal(f.evaluate('gridMaxCfg'), 2);
    assert.equal(f.els.negative.disabled, false);
    assert.equal(f.els.negNote.textContent, '(for grid cells above CFG 0)');
    assert.equal(f.evaluate('negativeSent(els.negative)'), 'keep this negative');
    assert.equal(f.els.gridPreview.children[0].className, 'totals');
    assert.match(f.els.gridPreview.textContent, /4 images/);
    assert.match(f.els.gridPreview.textContent, /a red fox/);
    assert.deepEqual([f.els.prompt.value, f.els.negative.value, f.els.gridPrompt.value, f.els.gridCfg.value], raw);
  } else if (scenario === 'current-error-still-visible') {
    const work = f.preview(); f.calls.requests[0].reject(Error('current bad axis')); await work;
    assert.equal(f.els.gridPreview.textContent, 'current bad axis');
    assert.equal(f.els.gridPreview.classList.contains('warn'), true); closed(f);
  } else if (scenario === 'model-hold-retires-old-preview') {
    const old = f.preview(); f.sandbox.held = true; await f.preview();
    const before = f.snapshot(); f.reply(0); await old;
    assert.equal(f.snapshot(), before); assert.equal(f.calls.requests.length, 1);
  } else if (scenario === 'already-closed-clears-reach-without-request') {
    f.evaluate('gridMaxCfg = 2; syncCfg(); gridRevealPending = true');
    await f.toggle(false); closed(f);
    assert.equal(f.calls.requests.length, 0); assert.equal(f.evaluate('gridRevealPending'), false);
    assert.equal(f.els.negative.value, '  keep this negative  ');
  } else if (scenario === 'blank-prompt-retires-old-preview') {
    const old = f.preview(); f.els.prompt.value = ''; await f.preview();
    const before = f.snapshot(); f.reply(0); await old;
    assert.equal(f.snapshot(), before); assert.match(f.els.gridPreview.textContent, /Enter a prompt/);
    assert.equal(f.calls.requests.length, 1);
  } else throw Error('unknown scenario: ' + scenario);
  console.log('passed: ' + scenario);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("scenario", [
    "late-success-after-close", "late-error-after-close", "two-previews-and-repeated-close",
    "reopen-current-before-old", "reopen-old-before-current", "current-success-keeps-raw-input-and-payload",
    "current-error-still-visible", "model-hold-retires-old-preview",
    "already-closed-clears-reach-without-request", "blank-prompt-retires-old-preview",
])
def test_grid_preview_close_ownership(scenario: str):
    node = shutil.which("node")
    assert node is not None, "Node is required for grid-preview ownership checks"
    done = subprocess.run(
        [node, "-e", SCRIPT, str(ROOT), scenario], capture_output=True,
        text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"passed: {scenario}" in done.stdout
