"""Instruction-edit Reuse and lineage in Node VM with inert page stubs.

These are source-level behavior checks only. They do not boot the app,
launch a browser or qualify layout, focus, accessibility or real rendering
(the browser smoke `smoke-instruction-edit` does).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ assert });
const load = (name) => vm.runInContext(fs.readFileSync(`web/${name}`, 'utf8'), context, { filename: name });
const run = (source) => vm.runInContext(source, context);
run(`
  class InertElement {
    constructor(value = '') {
      this.value = value; this.dataset = {}; this.attributes = new Map();
      this.title = ''; this.textContent = ''; this.disabled = false;
      this.classes = new Set();
      this.classList = {
        add: (name) => this.classes.add(name),
        remove: (name) => this.classes.delete(name),
        toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name),
      };
    }
    setAttribute(key, value) { this.attributes.set(key, value); }
    getAttribute(key) { return this.attributes.get(key) ?? null; }
    removeAttribute(key) { this.attributes.delete(key); }
  }
  const buttons = [];
  const document = { createElement: () => new InertElement(), querySelectorAll: (selector) => {
    assert.equal(selector, '.reuse-settings'); return buttons;
  } };
  const els = {
    galleryError: new InertElement(), error: new InertElement(),
    prompt: new InertElement('unfinished draft'), negative: new InertElement('own negative'),
    seed: new InertElement('123'), width: new InertElement('1024'), height: new InertElement('768'),
    steps: new InertElement('31'), cfg: new InertElement('4'), mu: new InertElement('1.2'),
    trueCfg: new InertElement('2'), lora: new InertElement('own-adapter'), loraScale: new InertElement('0.7'),
  };
  const errors = [];
  const forbiddenCalls = [];
  function forbidden(name) { forbiddenCalls.push(name); throw new Error('unexpected side effect: ' + name); }
  function api() { return forbidden('source/network lookup'); }
  function writeBox() { return forbidden('prompt write'); }
  function dropPromptUndo() { return forbidden('drop prompt Undo'); }
  function panelState() { return forbidden('composer replacement'); }
  function setReferences() { return forbidden('reference replacement'); }
  function openRefine() { return forbidden('open Refine'); }
  function reuseInpaint() { return forbidden('open Inpaint'); }
  function reuseOutpaint() { return forbidden('open Outpaint'); }
  // Edit by instruction's Reuse (instruction-edit.js), recorded: it opens the
  // dialog and never the panel.
  const reusedEdits = [];
  async function reuseEdit(rec) { reusedEdits.push(rec); }
  function showError(reason, quiet, options) {
    assert.equal(quiet, false); assert.equal(options.slot, els.galleryError);
    errors.push(reason);
  }
  let loadedModel = 'running/checkpoint';
  function modelsDiffer(modelId) { return !!modelId && modelId !== loadedModel; }
  function checkpointName(modelId) { return modelId || ''; }
  function parseOutpaintKey() { return null; }
  let moreSeedsRefreshes = 0;
  function syncMoreSeedsTools() { moreSeedsRefreshes += 1; }
  const owned = { steps: true, cfg: true };
  let referenceIds = [8, 9];
  let pendingLora = 'own-pending-adapter';
  function editRecord(extra = {}) {
    return { id: 21, prompt: 'make it night', negative_prompt: null, seed: 7,
      width: 512, height: 512, model_id: 'editor/checkpoint',
      edit: { source: 17, recipe: 'kontext-v1', edit_source: { hash: 'ab'.repeat(32) } },
      edit_missing: false, edit_binned: false, execution: null, ...extra };
  }
`);
load('reuse.js');
load('gallery.js');
run(`
  const priorUndo = { remove: () => forbidden('remove existing Reuse Undo') };
  const priorNote = { lead: 'existing note', before: { draft: 'keep' }, after: { draft: 'keep' }, button: priorUndo };
  reuseNote = priorNote;
  reuseSaid = 'existing note';
  els.error.textContent = 'existing note with Undo';
  const originalDraft = JSON.stringify({ els, owned, referenceIds, pendingLora });
  function assertDraftIntact() {
    assert.equal(JSON.stringify({ els, owned, referenceIds, pendingLora }), originalDraft);
    assert.equal(reuseNote, priorNote); assert.equal(reuseNote.button, priorUndo);
    assert.equal(reuseSaid, 'existing note');
    assert.equal(forbiddenCalls.length, 0);
  }
`);
"""

# Lineage: facts.js, viewer.js, gallery.js and instruction-edit.js against
# elements that keep what is appended, so the rows read as text.
LINEAGE = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ assert });
const load = (name) => vm.runInContext(fs.readFileSync(`web/${name}`, 'utf8'), context, { filename: name });
const run = (source) => vm.runInContext(source, context);
run(`
  class Node {
    constructor(tag) {
      this.tag = tag; this.kids = []; this.own = ''; this.className = ''; this.title = '';
      this.dataset = {}; this.attributes = new Map(); this.style = {}; this.classes = new Set();
      this.classList = {
        add: (name) => this.classes.add(name),
        remove: (name) => this.classes.delete(name),
        toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name),
        contains: (name) => this.classes.has(name),
      };
    }
    append(...items) { this.kids.push(...items); }
    get textContent() { return this.own + this.kids.map((k) => (typeof k === 'string' ? k : k.textContent)).join(''); }
    set textContent(text) { this.kids = []; this.own = String(text); }
    setAttribute(key, value) { this.attributes.set(key, value); }
    getAttribute(key) { return this.attributes.get(key) ?? null; }
    buttons() { return this.kids.flatMap((k) => (typeof k === 'string' ? [] : [...(k.tag === 'button' ? [k] : []), ...k.buttons()])); }
  }
  const document = { createElement: (tag) => new Node(tag) };
  const els = {};
  let loadedModel = 'black-forest-labs/FLUX.1-Kontext-dev';
  const lbList = []; const galleryRecords = []; const viewerRecords = [];
  const restored = []; const opened = [];
  function restoreImages(ids) { restored.push(...ids); }
  function openLightbox(rec, list) { opened.push([rec.id, list.map((r) => r.id)]); }
  function checkpointName(id) { return (id ?? '').split('/').pop(); }
  function edited(extra = {}) {
    return { id: 21, prompt: 'make it night', negative_prompt: null, seed: 7, width: 512, height: 512,
      steps: 28, guidance_scale: 2.5, true_cfg: null, mu: null, reference: null, lora: null, refine: null,
      model_id: 'black-forest-labs/FLUX.1-Kontext-dev', quantization: 'fp8', duration_ms: 1200,
      created_at: '2026-10-27T08:00:00+00:00', run_id: 'abc', job_id: 'j1', batch: 1,
      edit: { source: 17, recipe: 'kontext-v1', edit_source: { hash: 'ab'.repeat(32) } },
      edit_missing: false, edit_binned: false, ...extra };
  }
`);
load('gallery.js');
load('facts.js');
load('viewer.js');
load('instruction-edit.js');
"""


def _node(source: str, harness: str = HARNESS) -> None:
    node = shutil.which("node")
    assert node is not None, "Node is required for classic-script source checks"
    done = subprocess.run(
        [node, "-e", harness + source], cwd=ROOT, capture_output=True,
        text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_reuse_of_a_resolved_edit_opens_the_dialog_and_never_writes_the_composer():
    _node(r"""
run(`
  const rec = editRecord();
  assert.equal(executionReuseReason(rec), '');
  assert.equal(refuseExecutionReuse(rec), false);
  reuse(rec);
  assert.deepEqual(reusedEdits, [rec]);
  // The panel never takes an edit, by any road: its instruction is about its source.
  reuseIntoPanel(rec, true);
  reuseIntoPanel(rec, false, 'imported PNG', { name: 'saved.png', sampling: false });
  assert.equal(errors.length, 2);
  assert.ok(errors.every((reason) => /never goes into the generation panel/.test(reason)));
  assert.equal(reusedEdits.length, 1);
  assertDraftIntact();
`);
""")


def test_unresolved_missing_and_binned_edits_are_held_with_words_before_any_lookup():
    _node(r"""
run(`
  const held = [
    [editRecord({ edit: { source: null, recipe: 'kontext-v1' } }), /never resolved to a gallery image/],
    [editRecord({ edit_missing: true }), /#17, the source this was edited from, has been deleted/],
    [editRecord({ edit_missing: true, edit_binned: true }), /#17, the source this was edited from, is in the recycle bin; restore it/],
  ];
  for (const [rec, words] of held) {
    assert.match(executionReuseReason(rec), words);
    assert.equal(refuseExecutionReuse(rec), true);
    reuse(rec);
    reuseIntoPanel(rec, true);
    assertDraftIntact();
  }
  assert.equal(reusedEdits.length, 0);
  assert.equal(errors.length, 9);
  assert.match(errors[0], /never resolved/);
`);
""")


def test_edit_reuse_buttons_follow_their_source_across_refreshes():
    _node(r"""
run(`
  for (const compact of [false, true]) {
    const button = reuseButton(editRecord(), compact);
    buttons.push(button);
    assert.equal(button.dataset.reuseEdit, 'true');
    assert.equal(button.dataset.reuseEditSource, 17);
    assert.equal(button.getAttribute('aria-disabled'), null);
    assert.equal(button.getAttribute('aria-label'), 'Reuse instruction edit');
    assert.match(button.title, /Open Edit by instruction… on #17/);
    button.onclick();
  }
  assert.equal(reusedEdits.length, 2);
  // Another checkpoint loaded changes nothing: the dialog says what runs it.
  for (const model of ['running/checkpoint', 'editor/checkpoint', null]) {
    loadedModel = model;
    syncReuseButtons();
    for (const button of buttons) {
      assert.equal(button.getAttribute('aria-disabled'), null);
      assert.equal(button.getAttribute('aria-label'), 'Reuse instruction edit');
      assert.equal(button.classes.has('model-mismatch'), false);
    }
  }
  // Its source binned (forgetImages writes the data attributes): held, saying
  // why; restored: live again.
  for (const button of buttons) { button.dataset.reuseEditGone = 'true'; button.dataset.reuseEditBinned = 'true'; }
  syncReuseButtons();
  for (const button of buttons) {
    assert.equal(button.getAttribute('aria-disabled'), 'true');
    assert.equal(button.getAttribute('aria-label'), 'Reuse instruction edit (unavailable)');
    assert.equal(button.disabled, false);
    assert.match(button.title, /recycle bin; restore it to reopen the edit/);
  }
  for (const button of buttons) { button.dataset.reuseEditGone = 'false'; button.dataset.reuseEditBinned = 'false'; }
  syncReuseButtons();
  for (const button of buttons) assert.equal(button.getAttribute('aria-disabled'), null);
  assert.equal(moreSeedsRefreshes, 5);
  assert.equal(errors.length, 0);
  assertDraftIntact();
`);
""")


def test_an_unresolved_edit_stays_held_through_a_refresh_and_absent_edit_keeps_legacy_reasons():
    _node(r"""
run(`
  for (const execution of [null, undefined, {}, { compilation: null, attention_backend: null, step_cache: null }]) {
    const rec = editRecord({ edit: null, execution });
    assert.equal(executionReuseReason(rec), '');
    assert.equal(refuseExecutionReuse(rec), false);
  }
  assert.match(executionReuseReason(editRecord({ edit: null, execution: { step_cache: 'fbc-005-v1' } })),
    /execution recipe that is unavailable/);
  const unresolved = reuseButton(editRecord({ edit: { source: null, recipe: 'kontext-v1' } }));
  buttons.push(unresolved);
  syncReuseButtons();
  assert.equal(unresolved.dataset.reuseEditSource, '');
  assert.equal(unresolved.getAttribute('aria-disabled'), 'true');
  assert.match(unresolved.title, /never resolved/);
  const button = new InertElement();
  syncReuseButton(button, editRecord({ edit_missing: true }));
  assert.equal(button.getAttribute('aria-disabled'), 'true');
  syncReuseButton(button, editRecord({ edit: null, model_id: loadedModel }));
  buttons.push(button);
  assert.equal(button.dataset.reuseEdit, 'false');
  syncReuseButtons();
  assert.equal(button.getAttribute('aria-disabled'), null);
  assert.equal(button.getAttribute('aria-label'), 'Reuse settings');
  assert.equal(errors.length, 0);
  assertDraftIntact();
`);
""")


def test_facts_viewer_and_titles_name_the_edit_source_gone_or_not():
    _node(r"""
run(`
  // Plain substrings: a backslash in this template literal would be eaten.
  const has = (text, part) => assert.ok(text.includes(part), JSON.stringify(text) + ' lacks ' + JSON.stringify(part));
  const rec = edited();
  const facts = factsList(rec);
  has(facts.textContent, 'Edited from#17Recipe');
  has(facts.textContent, 'Sampling28 steps · guidance 2.5Edited');
  has(facts.textContent, 'Recipekontext-v1 · FLUX.1 Kontext [dev]');
  assert.ok(!facts.textContent.includes('no Mu') && !facts.textContent.includes('CFG'));
  const link = facts.buttons().find((b) => b.textContent === '#17');
  assert.ok(link, 'the source is a button');
  galleryRecords.push({ id: 17 });
  link.onclick();
  assert.deepEqual(opened, [[17, [17, 21]]]);
  has(settingsLine(rec), 'edit of #17, seed 7');
  assert.equal(passLineage(rec).textContent, '✐ Instruction edit of #17');
  assert.equal(cardTitle(rec), '#21 · 512×512 · seed 7 · edit of #17' + String.fromCharCode(10) + 'make it night');

  const binned = edited({ edit_missing: true, edit_binned: true });
  const binnedFacts = factsList(binned);
  has(binnedFacts.textContent, 'Edited from#17 (in the recycle bin) · Restore');
  binnedFacts.buttons().find((b) => b.textContent === 'Restore').onclick();
  assert.deepEqual(restored, [17]);
  has(settingsLine(binned), 'edit of #17 (in the recycle bin), ');
  assert.equal(passLineage(binned).textContent, '✐ Instruction edit of #17 (in the recycle bin)');

  const deleted = edited({ edit_missing: true });
  const deletedFacts = factsList(deleted);
  has(deletedFacts.textContent, 'Edited from#17 (deleted)Recipe');
  assert.equal(deletedFacts.buttons().some((b) => b.textContent === 'Restore' || b.textContent === '#17'), false);
  has(cardTitle(deleted), ' · edit of #17 (deleted)');
  has(factsList(edited({ edit: { source: null, recipe: 'kontext-v1' } })).textContent, 'Edited froman unresolved source');
`);
""", harness=LINEAGE)
