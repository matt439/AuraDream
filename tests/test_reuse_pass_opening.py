"""Deferred pass-source reads cannot replace a newer panel or Refine intent.

ROOT selects the actual adjacent repository web sources.
Presentation ports are inert; this does not qualify browser focus or layout.
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

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function record(id, prompt, extra = {}) {
  return { id, prompt, negative_prompt: 'negative ' + id, seed: id * 10,
    width: 512, height: 512, steps: 20, guidance_scale: 3.5, mu: 1.15,
    model_id: 'model-A', reference: [], ...extra };
}
const sourceA = record(1, 'source A');
const sourceB = record(2, 'source B');
const passA = record(101, 'pass A prompt', { refine: 1, width: 640, height: 768, strength: 0.65, seed: 111 });
const passB = record(102, 'pass B prompt', { refine: 2, width: 768, height: 640, strength: 0.45, seed: 222 });
const direct = record(3, 'direct replacement', { seed: 333, width: 1024, height: 1024 });

function fixture() {
  const calls = { requests: [], errors: [], writes: [], timers: 0, closedLightbox: 0 };
  const store = new Map();
  const document = { activeElement: null };
  class InertElement {
    constructor(id, tagName = 'INPUT') {
      this.id = id;
      this.tagName = tagName;
      this.dataset = {};
      this.hidden = false;
      this.open = false;
      this.checked = false;
      this.disabled = false;
      this.validity = { badInput: false, valid: true };
      this.isConnected = true;
      this.options = [];
      this.children = [];
      this.textContent = '';
      this.attributes = new Map();
      this.classList = { add() {}, remove() {}, contains() { return false; }, toggle() {} };
      this.style = { setProperty() {} };
      let value = '';
      Object.defineProperty(this, 'value', { get: () => value, set: next => { value = String(next ?? ''); } });
    }
    focus() { document.activeElement = this; }
    select() { this.selected = true; }
    remove() { this.isConnected = false; }
    append(...items) {
      for (const item of items) {
        this.children.push(item);
        this.textContent += typeof item === 'string' ? item : item.textContent;
      }
    }
    getAttribute(key) { return this.attributes.get(key) ?? null; }
    setAttribute(key, value) { this.attributes.set(key, String(value)); }
    removeAttribute(key) { this.attributes.delete(key); }
    closest(selector) {
      const host = this.host ?? this;
      if (selector === '#viewer') return host.area === 'viewer' ? viewer : null;
      if (selector.includes('#gallery .card[data-id]') && host.area === 'gallery') return host;
      if (selector.includes('#viewer figure[data-id]') && host.area === 'viewer') return host;
      return null;
    }
    contains(other) { return other === this || other?.host === this; }
    querySelectorAll() { return []; }
    querySelector() { return null; }
  }
  const elements = new Map();
  const els = new Proxy({}, { get(_target, name) {
    if (!elements.has(name)) elements.set(name, new InertElement(String(name)));
    return elements.get(name);
  } });
  const gallery = new InertElement('gallery', 'SECTION');
  const viewer = new InertElement('viewer', 'SECTION');
  const dialogNames = ['lightbox', 'refineDialog', 'inpaintDialog', 'outpaintDialog', 'msDialog'];
  for (const name of dialogNames) {
    els[name].tagName = 'DIALOG';
    els[name].showModal = () => { els[name].open = true; };
  }
  document.querySelectorAll = selector => selector === 'dialog[open]'
    ? dialogNames.map(name => els[name]).filter(dialog => dialog.open) : [];
  document.querySelector = selector => selector === '#viewer' ? viewer : selector === '#gallery' ? gallery : null;
  document.createElement = tag => new InertElement('created-' + tag, tag.toUpperCase());
  document.contains = element => !!element?.isConnected;
  const origin = (rec, area = 'gallery') => {
    const host = new InertElement(area + '-' + rec.id, area === 'viewer' ? 'FIGURE' : 'DIV');
    host.area = area;
    host.dataset.id = String(rec.id);
    const button = new InertElement('reuse-' + rec.id, 'BUTTON');
    button.host = host;
    return { host, button };
  };
  const origins = { A: origin(passA), B: origin(passB), viewer: origin(passA, 'viewer') };
  const initial = { prompt: 'draft before reuse', negative: 'draft negative', seed: '777',
    width: '800', height: '640', steps: '28', cfg: '3.5', mu: '1.15', trueCfg: '',
    batch: '1', runs: '1', lora: '', loraScale: '1', gridPromptMode: 'cross',
    gridSeeds: '1', stepCache: '' };
  for (const [name, value] of Object.entries(initial)) els[name].value = value;
  document.activeElement = origins.A.button;
  const sandbox = vm.createContext({
    document, els, AbortController, AbortSignal,
    Element: InertElement, HTMLElement: InertElement,
    window: { matchMedia: () => ({ matches: false }) },
    loadedModel: 'model-A', modelRevision: 1, serverStartedAt: 10,
    binMode: false, sheetMode: false,
    viewerRecords: [record(101, 'viewer A', { job_id: 'viewer-job-A' })],
    lbList: [passA, passB], lbIndex: 0,
    referenceIds: [], pendingLora: '', translatedSource: null,
    owned: { steps: true, cfg: true, mu: true, trueCfg: false },
    trScale: 1.5, invalidExecutionDraft: null, selectedExecution: null,
    performanceChoiceValue: control => control.value,
    refineSupported: true, trueCfgSupported: false, muSupported: true,
    loraSupported: false, referenceSupported: false,
    refineSource: null, refineReturn: null, refineLora: null,
    lastDefaults: { steps: 28, guidance_scale: 3.5, mu: 1.15, refine_strength: 0.5 },
    ASPECTS: ['1:1'], SIZES: { '1:1': { 1: [1024, 1024], 2: [1536, 1536] } },
    REFINE_SCALES: [0.5, 1, 1.5, 2], REFINE_SCALE: 1.5, REFINE_STRENGTH: 0.5,
    REFINE_NONE: 'Refine unavailable', pickedAspect: '1:1', pickedTier: 1,
    promptUndo: null, innerHeight: 900, scrollY: 0,
    remembered: key => store.get(key) ?? null,
    remember: (key, value) => store.set(key, value), forget: key => store.delete(key),
    api: (url, options) => {
      const request = { ...deferred(), url, options };
      calls.requests.push(request);
      // Deliberately ignore AbortSignal. The app must also reject stale results.
      return request.promise;
    },
    isShown: element => !!element?.isConnected && !element.hidden,
    modelsDiffer: id => !!id && id !== sandbox.loadedModel,
    checkpointName: id => id || '',
    sizeIsTight: () => false,
    lightboxPlace: () => els.lightbox.open ? { id: sandbox.lbList[sandbox.lbIndex]?.id } : null,
    closeLightbox: () => { calls.closedLightbox++; els.lightbox.open = false; },
    writeBox: (element, text) => { calls.writes.push([element.id, text]); element.value = text; },
    showError: (message, _note, options = {}) => {
      const slot = options.slot ?? els.error;
      calls.errors.push(String(message ?? ''));
      slot.textContent = String(message ?? ''); slot.hidden = !message;
    },
    setStepCache: value => { els.stepCache.value = value; }, stepCacheValue: () => els.stepCache.value,
    setTimeout: () => ++calls.timers, clearTimeout() {},
    calls,
  });
  sandbox.ownControls = () => { for (const key of Object.keys(sandbox.owned)) sandbox.owned[key] = true; };
  sandbox.offerPromptUndo = () => { const entry = new InertElement('reset-undo', 'BUTTON'); sandbox.promptUndo = entry; return entry; };
  sandbox.dropPromptUndo = () => { sandbox.promptUndo = null; };
  const noEffects = ['promptReplaced', 'syncSize', 'syncCfg', 'syncConditioning', 'syncRuns',
    'revealActiveSetup', 'syncExecutionControls', 'syncSubmitButtons', 'schedulePreview',
    'scheduleTagValidation', 'syncSetupUndo', 'clearAnalysis', 'forgetInpaint', 'forgetOutpaint',
    'refineFailed', 'sayRefineFromPanel', 'unmarkRefineFields', 'renderRefineCheck',
    'hideRefineSuggestions', 'sizeRefinePrompt', 'scheduleRefineCheck', 'scheduleRefineTags'];
  for (const name of noEffects) sandbox[name] = () => {};
  // Same order as the actual HTML. No source extraction or copied action logic.
  for (const name of ['generation.js', 'panel.js', 'reuse.js', 'refine.js', 'generation-presets.js']) {
    new vm.Script(fs.readFileSync(path.join(root, 'web', name), 'utf8'), { filename: name }).runInContext(sandbox);
  }
  // Loading declaration-only collaborators can define the presentation ports
  // listed above. Reapply those inert ports, keeping pure snapshots and all
  // action/entry functions real, including setupComposerState/panelNumberInputs.
  for (const name of noEffects) sandbox[name] = () => {};
  // Rendering, layout and scheduled follow-up reads are outside this race.
  // Keep actual reusePass/reuse/reuseIntoPanel/fillPanel/panelState/resetPanel,
  // clearReuseNote/syncReuseUndo/openRefine/refineValues and their entry points.
  vm.runInContext(`
    revealReused = () => {};
    syncRefine = () => {};
    scheduleRefinePreview = () => {};
    loadRefinePasses = () => Promise.resolve();
  `, sandbox);
  const read = expression => vm.runInContext(expression, sandbox);
  const call = (expression, rec) => { sandbox.inputRecord = rec; return read(expression); };
  return { sandbox, els, document, calls, origins,
    begin(rec, where = rec === passB ? origins.B : origins.A) {
      where.button.focus();
      return call('reusePass(inputRecord)', rec);
    },
    open(rec) { return call('openRefine(inputRecord)', rec); },
    reuse(rec) { return call('reuse(inputRecord)', rec); },
    reusePanel(rec) { return call('reuseIntoPanel(inputRecord, true)', rec); },
    reset() { read('resetPanel()'); }, persist() { read('persistPanel()'); },
    raw: () => read('setupComposerState()'),
    sourceId: () => read('refineSource?.id ?? null'),
    refine: () => read('JSON.stringify(refineValues())'),
    opening: () => read('typeof reusePassOpening === "undefined" ? null : reusePassOpening'),
  };
}
const missing = () => Object.assign(Error('unknown image #1'), { status: 404 });
const failure = () => Object.assign(Error('deliberate source read failure'), { status: 500 });
function unchanged(f, raw, focus) {
  assert.equal(f.raw(), raw, 'a stale source result rewrote the composer draft');
  assert.equal(f.els.refineDialog.open, false, 'a stale source result opened Refine');
  assert.equal(f.document.activeElement, focus, 'a stale result moved the inert focus port');
  assert.equal(f.els.error.textContent, 'newer work message', 'a stale result replaced the current message');
}

async function main() {
  const f = fixture(), { sandbox: s, els, calls: c } = f;
  const initial = f.raw();
  if (['current-success', 'current-viewer-origin', 'current-closed-lightbox-proxy',
    'current-disconnected-menu', 'first-server-identity'].includes(scenario)) {
    if (scenario === 'first-server-identity') s.serverStartedAt = null;
    let where = f.origins.A;
    if (scenario === 'current-viewer-origin') where = f.origins.viewer;
    if (scenario === 'current-closed-lightbox-proxy') {
      els.lightbox.open = false;
      els.lbReuse.tagName = 'BUTTON';
      where = { button: els.lbReuse };
    }
    if (scenario === 'current-disconnected-menu') {
      const menu = f.document.createElement('button');
      menu.isConnected = false;
      where = { button: menu };
    }
    const work = f.begin(passA, where);
    if (scenario === 'first-server-identity') s.serverStartedAt = 10;
    assert.equal(c.requests[0].url, '/api/images/1');
    c.requests[0].resolve(sourceA);
    await work;
    assert.equal(f.sourceId(), sourceA.id);
    assert.equal(els.refineDialog.open, true);
    assert.equal(els.refinePrompt.value, passA.prompt);
    assert.equal(els.refineSeed.value, String(passA.seed));
    assert.equal(els.refineWidth.value, String(passA.width));
    assert.equal(els.refineStrength.value, String(passA.strength));
    assert.equal(f.raw(), initial);
    assert.equal(f.document.activeElement, els.refineStrength);
    assert.equal(f.opening(), null);
  } else if (scenario === 'page-seed-write-success') {
    // A grid's drawn seed written back by the page is not a newer draft.
    const work = f.begin(passA);
    s.inputSeed = '4242';
    vm.runInContext('keepReusePassDraft(() => { els.seed.value = inputSeed; }); persistPanel();', s);
    assert.equal(c.requests[0].options.signal.aborted, false, 'a page write cancelled the current lookup');
    c.requests[0].resolve(sourceA);
    await work;
    assert.equal(f.sourceId(), sourceA.id, 'a page write dropped the current Reuse');
    assert.equal(els.refineDialog.open, true);
    assert.equal(els.seed.value, '4242');
    assert.equal(f.opening(), null);
  } else if (scenario === 'current-missing-fallback') {
    const work = f.begin(passA);
    c.requests[0].reject(missing());
    await work;
    assert.equal(els.refineDialog.open, false);
    assert.equal(els.prompt.value, passA.prompt);
    assert.equal(els.seed.value, String(passA.seed));
    assert.equal(els.width.value, '800');
    assert.equal(els.height.value, '640');
    assert.match(els.error.textContent, /source this was refined from.*no longer in the gallery/);
    assert.equal(f.document.activeElement, els.prompt);
    assert.equal(f.opening(), null);
  } else if (scenario === 'current-error') {
    const work = f.begin(passA);
    c.requests[0].reject(failure());
    await work;
    assert.equal(els.refineDialog.open, false);
    assert.equal(f.raw(), initial);
    assert.equal(els.error.textContent, 'deliberate source read failure');
    assert.equal(f.opening(), null);
  } else if (scenario === 'direct-refine-new-draft') {
    const old = f.begin(passA);
    f.open(sourceB);
    els.refinePrompt.value = 'B draft edited after the direct open';
    els.refineSeed.value = '1e';
    els.refineWidth.value = '1024.';
    els.refinePrompt.focus();
    const draft = f.refine(), focus = f.document.activeElement;
    c.requests[0].resolve(sourceA);
    await old;
    // Assert visible data first: original source fails here, not on a new name.
    assert.equal(f.sourceId(), sourceB.id, 'old Reuse A replaced direct Refine B');
    assert.equal(f.refine(), draft, 'old Reuse A erased B raw draft');
    assert.equal(f.document.activeElement, focus);
    assert.equal(c.requests[0].options.signal.aborted, true);
  } else if (scenario.startsWith('newer-pass-')) {
    const old = f.begin(passA);
    const newer = f.begin(passB);
    const owner = f.opening();
    els.error.textContent = 'newer work message';
    if (scenario.endsWith('error')) c.requests[0].reject(failure());
    else c.requests[0].resolve(sourceA);
    await old;
    assert.equal(els.refineDialog.open, false, 'old pass opened before newer request settled');
    assert.equal(els.error.textContent, 'newer work message', 'old pass error leaked over newer intent');
    assert.equal(f.opening(), owner, 'old finally released the newer request owner');
    assert.equal(c.requests[1].options.signal.aborted, false);
    c.requests[1].resolve(sourceB);
    await newer;
    assert.equal(f.sourceId(), sourceB.id);
    assert.equal(els.refinePrompt.value, passB.prompt);
    assert.equal(els.refineSeed.value, String(passB.seed));
    assert.equal(f.opening(), null);
  } else {
    let resetRaw = null;
    if (scenario === 'empty-reset-missing') { f.reset(); resetRaw = f.raw(); }
    if (scenario === 'bad-input-seed-missing') els.seed.value = '';
    if (scenario === 'lightbox-changed') { els.lightbox.open = true; s.lbIndex = 0; }
    const where = scenario === 'viewer-replaced' ? f.origins.viewer : f.origins.A;
    const old = f.begin(passA, where);
    let reject = false;
    if (scenario === 'prompt-edit-missing') {
      els.prompt.value = 'new raw prompt'; els.prompt.focus(); f.persist(); reject = true;
    } else if (scenario === 'edit-reverted-missing') {
      const before = els.prompt.value;
      els.prompt.value = 'a deliberate superseding edit'; els.prompt.focus(); f.persist();
      els.prompt.value = before; f.persist(); reject = true;
    } else if (scenario === 'seed-edit-missing') {
      els.seed.value = '1e'; els.seed.focus(); f.persist(); reject = true;
    } else if (scenario === 'resume-change-success' || scenario === 'resume-change-missing') {
      els.gridResume.checked = true; f.persist();
      reject = scenario.endsWith('missing');
    } else if (scenario === 'then-refine-change-missing') {
      els.thenRefine.checked = true;
      els.thenRefineStrength.value = '0.1,';
      els.thenRefineSteps.value = '1e';
      s.trScale = 2;
      els.thenRefineStrength.focus(); f.persist(); reject = true;
    } else if (scenario === 'performance-intent-change') {
      els.modelCompilation.value = 'raw-test-compilation';
      s.invalidExecutionDraft = { compilation: 'unavailable raw intent' };
      s.selectedExecution = { compilation: 'raw-test-compilation' };
      f.persist();
    } else if (scenario === 'bad-input-seed-missing') {
      const valueBefore = els.seed.value;
      els.seed.validity.badInput = true;
      assert.equal(els.seed.value, valueBefore, 'native badInput changes without a value change');
      els.seed.focus(); f.persist(); reject = true;
    } else if (scenario === 'reset-missing') {
      f.reset(); reject = true;
      assert.equal(els.prompt.value, ''); assert.equal(els.seed.value, '');
    } else if (scenario === 'empty-reset-missing') {
      f.reset(); reject = true;
      assert.equal(f.raw(), resetRaw, 'empty Reset must leave the captured raw panel identical');
    } else if (scenario === 'direct-reuse') {
      f.reuse(direct);
      assert.equal(els.prompt.value, direct.prompt);
    } else if (scenario === 'reuse-into-panel-missing') {
      f.reusePanel(direct); reject = true;
      assert.equal(els.prompt.value, direct.prompt);
    } else if (scenario === 'model-revision') s.modelRevision = 2;
    else if (scenario === 'model-name') s.loadedModel = 'model-B';
    else if (scenario === 'scope-search') els.search.value = 'run:newer';
    else if (scenario === 'origin-detached') f.origins.A.host.isConnected = false;
    else if (scenario === 'lightbox-changed') { s.lbIndex = 1; els.lbNext.focus(); }
    else if (scenario === 'dialog-opened-missing') { els.msDialog.open = true; els.msCount.focus(); reject = true; }
    else if (scenario === 'viewer-replaced') s.viewerRecords = [record(200, 'new viewer', { job_id: 'viewer-job-B' })];
    else if (scenario === 'known-server-restart') s.serverStartedAt = 20;
    else throw Error('unknown scenario ' + scenario);
    els.error.textContent = 'newer work message';
    const raw = f.raw(), focus = f.document.activeElement;
    if (reject) c.requests[0].reject(missing()); else c.requests[0].resolve(sourceA);
    await old;
    unchanged(f, raw, focus);
    if (['prompt-edit-missing', 'edit-reverted-missing', 'seed-edit-missing', 'reset-missing',
      'empty-reset-missing', 'direct-reuse', 'reuse-into-panel-missing',
      'resume-change-success', 'resume-change-missing', 'then-refine-change-missing',
      'performance-intent-change', 'bad-input-seed-missing'].includes(scenario)) {
      assert.equal(c.requests[0].options.signal.aborted, true);
    }
    if (scenario.startsWith('resume-change-')) assert.equal(els.gridResume.checked, true);
    if (scenario === 'then-refine-change-missing') {
      assert.equal(els.thenRefine.checked, true);
      assert.equal(els.thenRefineStrength.value, '0.1,');
      assert.equal(els.thenRefineSteps.value, '1e');
      assert.equal(s.trScale, 2);
    }
    if (scenario === 'bad-input-seed-missing') {
      assert.equal(els.seed.value, '');
      assert.equal(els.seed.validity.badInput, true);
    }
    if (scenario === 'lightbox-changed') { assert.equal(els.lightbox.open, true); assert.equal(s.lbIndex, 1); }
    if (scenario === 'dialog-opened-missing') assert.equal(els.msDialog.open, true);
    assert.equal(f.opening(), null);
  }
  assert.equal(c.requests.length, scenario.startsWith('newer-pass-') ? 2 : 1, 'unexpected unrelated API read');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("scenario", [
    "current-success", "current-viewer-origin", "current-closed-lightbox-proxy",
    "current-disconnected-menu", "current-missing-fallback", "current-error",
    "direct-refine-new-draft", "prompt-edit-missing", "edit-reverted-missing", "seed-edit-missing",
    "reset-missing", "empty-reset-missing", "newer-pass-success", "newer-pass-error", "direct-reuse",
    "reuse-into-panel-missing", "model-revision", "model-name", "scope-search",
    "origin-detached", "lightbox-changed", "dialog-opened-missing", "viewer-replaced",
    "known-server-restart", "first-server-identity",
    "resume-change-success", "resume-change-missing", "then-refine-change-missing",
    "performance-intent-change", "bad-input-seed-missing", "page-seed-write-success",
])
def test_reuse_pass_opening_owns_its_actual_source_result(scenario: str):
    node = shutil.which("node")
    assert node is not None, "Node is required for the Reuse pass opening checks"
    done = subprocess.run([node, "-e", SCRIPT, str(ROOT), scenario],
                          capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert done.returncode == 0, done.stdout + done.stderr
