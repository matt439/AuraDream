"""Real copy/menu/lightbox functions on inert DOM and synthetic clipboard ports.

No page boot, browser, server, native clipboard, model or queue is exercised.
The separate focused smoke owns physical UI/navigation evidence.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = [
    "current-prompt-press", "current-seed-press", "late-success-after-walk",
    "late-error-after-walk", "walk-away-and-back", "late-error-after-close",
    "reopen-same-object", "native-close-reopen-before-event", "same-object-redraw",
    "latest-success-wins", "latest-error-ignored", "settled-note-clears",
    "newer-different-message-survives", "newer-identical-message-survives",
    "card-error-does-not-mirror", "detached-card", "reused-host-id",
    "held-copy-does-not-write", "current-error-local", "result-success",
    "stale-menu-record", "same-id-new-object-retires",
]

SCRIPT = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = process.argv[1], scenario = process.argv[2];
function element(tag = 'div') {
  const classes = new Set(), attrs = new Map();
  const el = { tagName: tag.toUpperCase(), dataset: {}, children: [], parentElement: null,
    hidden: false, disabled: false, isConnected: true, title: '', open: false,
    classList: { add: name => classes.add(name), remove: name => classes.delete(name),
      contains: name => classes.has(name), toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
    setAttribute(name, value) { attrs.set(name, String(value)); },
    getAttribute(name) { return attrs.get(name) ?? null; }, removeAttribute(name) { attrs.delete(name); },
    append(...children) { for (const child of children) { child.parentElement = this; this.children.push(child); } },
    replaceChildren(...children) { this.children = []; this.append(...children); },
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this); this.isConnected = false; },
    closest(selector) {
      if (selector === '[data-id]' && this.dataset.id !== undefined) return this;
      if (selector === '.gone' && classes.has('gone')) return this;
      if (selector.startsWith('dialog') && this.tagName === 'DIALOG' && this.open) return this;
      if (selector === 'figure' && this.tagName === 'FIGURE') return this;
      return this.parentElement?.closest(selector) ?? null;
    },
    contains(other) { return other === this || this.children.some(child => child.contains?.(other)); },
    querySelector(selector) { return selector === ':scope > .card-note' ? this.children.find(child => child.className?.startsWith('card-note')) ?? null : null; },
    querySelectorAll() { return []; }, focus() {},
    close() { this.open = false; }, showModal() { this.open = true; },
  };
  Object.defineProperty(el, 'firstChild', { get() { return this.children[0] ?? null; } });
  Object.defineProperty(el, 'textContent', {
    get() { return this.children.map(child => child.nodeType === 3 ? child.data : child.textContent).join(''); },
    set(text) { // DOM textContent replaces the child Text node, even for equal words.
      this.children = String(text) ? [{ nodeType: 3, data: String(text), parentElement: this }] : [];
    },
  });
  return el;
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function exactSlice(source, start, end) {
  assert.equal(source.split(start).length - 1, 1, 'missing or ambiguous real function start');
  assert.equal(source.split(end).length - 1, 1, 'missing or ambiguous real function end');
  return source.slice(source.indexOf(start), source.indexOf(end));
}
function fixture() {
  const calls = [], poisoned = [], timers = [];
  const els = new Proxy({ lightbox: element('dialog'), lbStage: element(), contextMenu: element('ul') }, {
    get(target, key) { if (!(key in target)) target[key] = element(); return target[key]; },
  });
  els.modelPicker.hidden = true;
  const a = { id: 11, seed: 912, prompt: '  first prompt\nwith exact spacing  ', width: 256, height: 256, url: '/a.png', model_id: 'cold' };
  const b = { id: 12, seed: 913, prompt: 'second prompt', width: 256, height: 256, url: '/b.png', model_id: 'cold' };
  const poison = name => function () { poisoned.push(name); throw Error('forbidden native/side-effect port: ' + name); };
  const clipboard = Object.freeze({
    writeText(text) { const call = { text, ...deferred() }; calls.push(call); return call.promise; },
    write: poison('image clipboard write'), read: poison('clipboard read'), readText: poison('clipboard readText'),
  });
  const noop = () => {};
  const sandbox = vm.createContext({ els, window: { isSecureContext: true }, navigator: { clipboard },
    lbList: [], lbIndex: -1, galleryRecords: [a,b], galleryMore: false, total: 2,
    viewerRecords: [], lbPickedPair: [], lbCfgShown: {}, refineSupported: true,
    document: { activeElement: element('body'), body: element('body'), querySelector: () => null,
      createElement: tag => element(tag), execCommand: poison('execCommand') },
    requestAnimationFrame: callback => callback(), setTimeout: callback => { timers.push(callback); return timers.length; },
    clearTimeout: noop, fetch: poison('fetch'), ClipboardItem: poison('native ClipboardItem'),
    referenceAt: () => -1, useSeed: poison('useSeed'), isShown: () => true, starTitle: () => 'Star',
    settingsLine: () => 'cold', promptSearchLine: () => element(), factsList: () => element(),
    syncReuseButton: noop, syncMoreSeedsFor: noop, syncOutpaintTool: noop, syncEditTool: noop,
    syncReferenceTool: noop, syncLightboxInpaint: noop, syncLightboxOutpaint: noop,
    forgetLadders: noop, showModelPicker: poison('model picker'), OFFLINE_MESSAGE: 'offline',
    selection: new Map(), lightboxPickHeld: () => '', lightboxPickTitle: () => 'Pick',
    pickFromLightbox: poison('pickFromLightbox'),
  });
  const read = name => fs.readFileSync(path.join(root, 'web', name), 'utf8').replace(/\r\n/g, '\n');
  new vm.Script(exactSlice(read('state.js'), 'const offlineSlots = new Map();',
    '// The server answers again:'), { filename: 'actual-state-showError.js' }).runInContext(sandbox);
  new vm.Script(exactSlice(read('gallery.js'), 'function refuseInCard(div, message, short) {',
    '// The card the keyboard is on,'), { filename: 'actual-card-feedback.js' }).runInContext(sandbox);
  for (const name of ['lightbox.js', 'context-menu.js']) {
    new vm.Script(read(name), { filename: name }).runInContext(sandbox);
  }
  // Keep actual open/render/step/close and menu/feedback functions. These
  // unrelated geometry, comparison and model controls are inert fixture ports.
  vm.runInContext(`askLadder = () => {}; layoutLightbox = () => {}; syncCompare = () => {};
    syncLightboxFavoriteHold = () => {}; syncLightboxBinWriteHold = () => {};
    clampScale = (_rec, scale) => scale; passesLine = () => document.createElement('p');`, sandbox);
  const evaluate = text => vm.runInContext(text, sandbox);
  const open = (rec = a) => sandbox.openLightbox(rec, [a,b]);
  const note = text => sandbox.showError(text, true, { slot: els.lbError, mirror: false });
  const copy = (label = 'Copy prompt', kind = 'lightbox', rec = a, host = els.lbStage) =>
    sandbox.imageItems({ kind, rec, host }).find(item => item?.label === label);
  const card = (rec = a, result = false) => {
    const host = element(); host.dataset.id = String(rec.id);
    if (result) { const figure = element('figure'); figure.dataset.id = String(rec.id); delete host.dataset.id; figure.append(host); }
    return host;
  };
  return { a,b,els,calls,poisoned,sandbox,evaluate,open,note,copy,card,
    resolve: index => calls[index].resolve(), reject: index => calls[index].reject(Error('synthetic copy refusal')),
    state: () => JSON.stringify({ text: els.lbError.textContent, hidden: els.lbError.hidden, gallery: els.galleryError.textContent }),
  };
}
async function main() {
  const f = fixture(); f.open();
  const { a,b,els } = f;
  if (scenario === 'current-prompt-press' || scenario === 'current-seed-press') {
    const seed = scenario.includes('seed'), item = f.copy(seed ? 'Copy seed' : 'Copy prompt');
    f.copy(seed ? 'Copy prompt' : 'Copy seed'); // Construction alone must not steal this press's ownership.
    els.contextMenu.hidden = false;
    f.evaluate('ctxMenu.host = els.lbStage');
    let pending;
    f.sandbox.chooseContextItem({ ctxRun: () => { pending = item.run(); return pending; } });
    assert.equal(els.contextMenu.hidden, true);
    assert.equal(f.calls.length, 1, 'write must start synchronously in the actual menu choice');
    assert.equal(f.calls[0].text, seed ? String(a.seed) : a.prompt);
    assert.equal(f.evaluate('ctxMenu.host'), null, 'normal menu close must not invalidate the copy');
    f.resolve(0); await pending;
    assert.equal(els.lbError.textContent, seed ? 'Seed copied' : 'Prompt copied');
    assert.equal(els.lbError.hidden, false);
  } else if (scenario === 'late-success-after-walk' || scenario === 'late-error-after-walk') {
    const pending = f.copy().run(); await f.sandbox.stepLightbox(1); f.note('B newer message');
    const before = f.state(), node = els.lbError.firstChild;
    if (scenario.includes('error')) f.reject(0); else f.resolve(0);
    await pending; assert.equal(f.state(), before); assert.equal(els.lbError.firstChild, node);
    assert.equal(f.calls[0].text, a.prompt);
  } else if (scenario === 'walk-away-and-back') {
    const pending = f.copy('Copy seed').run(); await f.sandbox.stepLightbox(1); await f.sandbox.stepLightbox(-1);
    f.note('A newer inspection'); f.reject(0); await pending;
    assert.equal(els.lbError.textContent, 'A newer inspection');
  } else if (scenario === 'late-error-after-close') {
    const pending = f.copy().run(); f.sandbox.closeLightbox(); const before = f.state();
    f.reject(0); await pending; assert.equal(f.state(), before); assert.equal(els.lightbox.open, false);
  } else if (scenario === 'reopen-same-object' || scenario === 'native-close-reopen-before-event') {
    const pending = f.copy().run();
    if (scenario.startsWith('native')) els.lightbox.close(); else f.sandbox.closeLightbox();
    f.open(a); f.note('new same-object inspection'); f.resolve(0); await pending;
    assert.equal(els.lbError.textContent, 'new same-object inspection');
  } else if (scenario === 'same-object-redraw') {
    const pending = f.copy('Copy seed').run(); f.sandbox.renderLightbox(); f.resolve(0); await pending;
    assert.equal(els.lbError.textContent, 'Seed copied');
  } else if (scenario === 'latest-success-wins' || scenario === 'latest-error-ignored') {
    const old = f.copy('Copy seed').run(), current = f.copy().run();
    assert.equal(f.calls.length, 2); f.resolve(1); await current;
    const node = els.lbError.firstChild;
    if (scenario.includes('error')) f.reject(0); else f.resolve(0);
    await old; assert.equal(els.lbError.textContent, 'Prompt copied'); assert.equal(els.lbError.firstChild, node);
  } else if (['settled-note-clears','newer-different-message-survives','newer-identical-message-survives'].includes(scenario)) {
    const pending = f.copy().run(); f.resolve(0); await pending;
    assert.equal(els.lbError.textContent, 'Prompt copied'); const original = els.lbError.firstChild;
    if (scenario !== 'settled-note-clears') {
      f.note(scenario.includes('identical') ? 'Prompt copied' : 'newer unrelated message');
      assert.notEqual(els.lbError.firstChild, original, 'actual showError must replace the owned text node');
    }
    const before = f.state(), node = els.lbError.firstChild;
    await f.sandbox.stepLightbox(1);
    if (scenario === 'settled-note-clears') { assert.equal(els.lbError.hidden, true); assert.equal(els.lbError.textContent, ''); }
    else { assert.equal(f.state(), before); assert.equal(els.lbError.firstChild, node); }
  } else if (scenario === 'card-error-does-not-mirror') {
    f.sandbox.closeLightbox(); const host = f.card(); const pending = f.copy('Copy seed','card',a,host).run();
    f.open(b); f.note('B existing refusal'); const node = els.lbError.firstChild;
    f.reject(0); await pending;
    assert.equal(els.lbError.textContent, 'B existing refusal'); assert.equal(els.lbError.firstChild, node);
    assert.equal(els.galleryError.textContent, 'Could not copy: synthetic copy refusal.');
    const local = host.querySelector(':scope > .card-note');
    assert.equal(local.textContent, 'Copy failed'); assert.equal(local.getAttribute('aria-hidden'), 'true');
    assert.equal(local.getAttribute('role'), null, 'only the gallery sentence should announce failure');
  } else if (scenario === 'detached-card' || scenario === 'reused-host-id') {
    f.sandbox.closeLightbox(); const host = f.card(); const pending = f.copy('Copy prompt','card',a,host).run();
    if (scenario === 'detached-card') host.isConnected = false; else host.dataset.id = String(b.id);
    f.reject(0); await pending; assert.equal(els.galleryError.textContent, ''); assert.equal(host.children.length, 0);
  } else if (scenario === 'held-copy-does-not-write') {
    f.sandbox.window.isSecureContext = false; await f.copy().run();
    assert.equal(f.calls.length, 0); assert.match(els.lbError.textContent, /browser opens the clipboard/);
    assert.equal(els.galleryError.textContent, '');
  } else if (scenario === 'current-error-local') {
    const pending = f.copy().run(); f.reject(0); await pending;
    assert.equal(els.lbError.textContent, 'Could not copy: synthetic copy refusal.');
    assert.equal(els.galleryError.textContent, '', 'lightbox failure must stay local');
    await f.sandbox.stepLightbox(1); assert.equal(els.lbError.textContent, '');
  } else if (scenario === 'result-success') {
    f.sandbox.closeLightbox(); const host = f.card(a,true); const pending = f.copy('Copy seed','result',a,host).run();
    assert.equal(f.calls[0].text, String(a.seed)); f.resolve(0); await pending;
    assert.equal(host.querySelector(':scope > .card-note').textContent, 'Seed copied');
    assert.equal(els.galleryError.textContent, '');
  } else if (scenario === 'stale-menu-record') {
    const stale = f.copy(); await f.sandbox.stepLightbox(1); f.note('B newer menu context');
    const pending = stale.run(); assert.equal(f.calls[0].text, a.prompt);
    f.resolve(0); await pending; assert.equal(els.lbError.textContent, 'B newer menu context');
  } else if (scenario === 'same-id-new-object-retires') {
    const pending = f.copy().run(); f.sandbox.openLightbox({ ...a }, null); f.note('refreshed record object');
    f.resolve(0); await pending; assert.equal(els.lbError.textContent, 'refreshed record object');
  } else throw Error('unknown scenario: ' + scenario);
  assert.deepEqual(f.poisoned, []); assert.equal(f.calls.length, new Set(f.calls).size);
  assert.equal(a.prompt, '  first prompt\nwith exact spacing  '); assert.equal(a.seed, 912);
  console.log('passed: ' + scenario);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_copy_feedback_ownership(scenario: str, tmp_path: Path):
    node = shutil.which("node")
    assert node is not None, "Node is required for copy-feedback ownership checks"
    done = subprocess.run(
        [node, "-e", SCRIPT, str(ROOT), scenario], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"passed: {scenario}" in done.stdout
