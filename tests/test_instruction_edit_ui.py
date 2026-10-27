"""Instruction-edit Reuse holds in Node VM with inert page stubs.

These are source-level behavior checks only. They do not boot the app,
launch a browser or qualify layout, focus, accessibility or real rendering.
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
      execution: null, ...extra };
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


def _node(source: str) -> None:
    node = shutil.which("node")
    assert node is not None, "Node is required for classic-script source checks"
    done = subprocess.run(
        [node, "-e", HARNESS + source], cwd=ROOT, capture_output=True,
        text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_reuse_entry_holds_edits_before_composer_undo_or_source_lookup():
    _node(r"""
run(`
  for (const rec of [editRecord(), editRecord({ refine: 17 }),
      editRecord({ refine: 17, mask: 'ab'.repeat(32) }),
      editRecord({ refine: 17, outpaint: {} }),
      editRecord({ edit: { source: null, recipe: 'kontext-v1' } }),
      editRecord({ edit: {}, execution: { attention_backend: 'cudnn-v1' } })]) {
    const reason = executionReuseReason(rec);
    assert.match(reason, /Instruction editing is unavailable/);
    assert.equal(refuseExecutionReuse(rec), true);
    reuse(rec);
    reuseIntoPanel(rec, true);
    reuseIntoPanel(rec, false, 'imported PNG', { name: 'saved.png', sampling: false });
    assertDraftIntact();
  }
  assert.equal(errors.length, 24);
  assert.ok(errors.every((reason) => /source-dependent instruction/.test(reason)));
`);
""")


def test_button_dataset_refresh_keeps_edit_hold_across_model_and_execution_updates():
    _node(r"""
run(`
  for (const compact of [false, true]) {
    const button = reuseButton(editRecord(), compact);
    buttons.push(button);
    assert.equal(button.dataset.reuseEdit, 'true');
    assert.equal(button.dataset.reuseCompact, String(compact));
    assert.equal(button.getAttribute('aria-disabled'), 'true');
    assert.equal(button.getAttribute('aria-label'), 'Reuse instruction edit (unavailable)');
    assert.equal(button.disabled, false);
    assert.match(button.title, /source-dependent instruction/);
    button.onclick();
  }
  for (const model of ['running/checkpoint', 'editor/checkpoint', null]) {
    loadedModel = model;
    syncReuseButtons();
    for (const button of buttons) {
      assert.equal(button.dataset.reuseEdit, 'true');
      assert.equal(button.getAttribute('aria-disabled'), 'true');
      assert.equal(button.getAttribute('aria-label'), 'Reuse instruction edit (unavailable)');
      assert.equal(button.disabled, false);
      assert.match(button.title, /Instruction editing is unavailable/);
    }
  }
  assert.equal(moreSeedsRefreshes, 3);
  assert.equal(errors.length, 2);
  assertDraftIntact();
`);
""")


def test_absent_edit_keeps_legacy_reuse_reasons_and_refresh_can_clear_a_previous_hold():
    _node(r"""
run(`
  for (const execution of [null, undefined, {}, { compilation: null, attention_backend: null, step_cache: null }]) {
    const rec = editRecord({ edit: null, execution });
    assert.equal(executionReuseReason(rec), '');
    assert.equal(refuseExecutionReuse(rec), false);
  }
  assert.match(executionReuseReason(editRecord({ edit: null, execution: { step_cache: 'fbc-005-v1' } })),
    /execution recipe that is unavailable/);
  const button = new InertElement();
  syncReuseButton(button, editRecord());
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
