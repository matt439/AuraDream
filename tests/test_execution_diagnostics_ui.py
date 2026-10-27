"""Execution observations in the classic page, using Node VM and inert DOM stubs.

These exercise source contracts only. They do not boot the page, open a browser,
start a server, or qualify layout, accessibility, rendering or performance.
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
    constructor() { this.children = []; this.attributes = new Map(); this.title = ''; this.className = ''; this.ownText = ''; }
    get childElementCount() { return this.children.length; }
    get textContent() { return this.children.length ? this.children.map((node) => typeof node === 'string' ? node : node.textContent).join('') : this.ownText; }
    set textContent(value) { this.ownText = value; this.children = []; }
    replaceChildren(...nodes) { this.children = nodes; this.ownText = ''; }
    append(...nodes) { this.children.push(...nodes); }
    querySelector(selector) {
      const match = selector.match(/\\.([a-z-]+)$/);
      return match ? this.children.find((node) => node.className === match[1]) ?? null : null;
    }
    setAttribute(key, value) { this.attributes.set(key, value); }
    getAttribute(key) { return this.attributes.get(key) ?? null; }
    removeAttribute(key) { this.attributes.delete(key); }
    contains() { return false; }
    focus() {}
  }
  const document = { title: 'AuraDream', activeElement: null, createElement: () => new InertElement() };
  const els = { progressText: new InertElement() };
  let currentJob = 'job-1';
  let serverOffline = false;
  let modelState = 'ready';
  let modelRuntimeSettings = { model_id: 'actual/model' };
  let loadedModel = 'actual/model';
  let modelRevision = 7;
  let serverStartedAt = 100;
  function modelLabel(id) { return id; }
  function progress() {
    return { spoken: els.progressText.querySelector('.said')?.textContent ?? els.progressText.textContent,
      shown: els.progressText.querySelector('.seen')?.textContent ?? null };
  }
  function diagnostic(overrides = {}) {
    return { load_id: 1, recipe: 'regional-v1', registration: 'registered',
      first_call: 'pending', failure_stage: null, ...overrides };
  }
  function status(overrides = {}) {
    return { model_id: 'actual/model', model_revision: 7, started_at: 100,
      state: 'ready', execution_diagnostics: diagnostic(), ...overrides };
  }
`);
load('job-progress.js');
load('execution.js');
"""


def _node(source: str) -> None:
    node = shutil.which("node")
    assert node is not None, "Node is required for classic-script source checks"
    done = subprocess.run(
        [node, "-e", HARNESS + source],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_observations_accept_only_bounded_fixed_scalar_values():
    _node(r"""
run(`
  const valid = readExecutionDiagnostics(diagnostic());
  assert.equal(Object.isFrozen(valid), true);
  assert.equal(JSON.stringify(valid), JSON.stringify(diagnostic()));
  for (const first_call of ['pending', 'running', 'returned']) {
    assert.equal(readExecutionDiagnostics(diagnostic({ first_call })).first_call, first_call);
    assert.equal(readExecutionDiagnostics(diagnostic({ first_call, failure_stage: 'forward' })), null);
  }
  for (const first_call of ['failed', 'cancelled']) {
    assert.equal(readExecutionDiagnostics(diagnostic({ first_call })), null);
    for (const failure_stage of ['forward', 'verification', 'policy_exit']) {
      assert.equal(readExecutionDiagnostics(diagnostic({ first_call, failure_stage })).failure_stage, failure_stage);
    }
  }
  for (const value of [null, false, 'registered', [], {},
    diagnostic({ load_id: 0 }), diagnostic({ load_id: -1 }), diagnostic({ load_id: 1.5 }),
    diagnostic({ load_id: Number.MAX_SAFE_INTEGER + 1 }), diagnostic({ load_id: '1' }),
    diagnostic({ recipe: '<script>' }), diagnostic({ registration: 'warm' }),
    diagnostic({ first_call: 'warm' }), diagnostic({ extra: 'private path or traceback' }),
    diagnostic({ first_call: 'failed', failure_stage: 'arbitrary error' })]) {
    assert.equal(readExecutionDiagnostics(value), null);
    assert.equal(executionDiagnosticCaption(value), null);
  }
`);
""")


def test_captions_separate_registration_and_first_call_without_qualification():
    _node(r"""
run(`
  for (const first_call of ['pending', 'running', 'returned', 'failed', 'cancelled']) {
    const value = diagnostic({ first_call,
      failure_stage: ['failed', 'cancelled'].includes(first_call) ? 'forward' : null });
    const caption = executionDiagnosticCaption(value);
    assert.match(caption.short, new RegExp(first_call));
    assert.match(caption.full, /Regional registered/);
    assert.match(caption.full, /Warm kernels, cache hits and performance are unverified/);
    assert.doesNotMatch(caption.full, /ready to render|qualified|cache hit confirmed|warmed up/i);
  }
  assert.match(executionDiagnosticCaption(diagnostic()).full, /no forward attempt has been observed/);
  assert.match(executionDiagnosticCaption(diagnostic({ first_call: 'running' })).full, /callback may be delayed/);
  assert.match(executionDiagnosticCaption(diagnostic({ first_call: 'returned' })).full,
    /after ownership checks and policy exit/);
  for (const [failure_stage, label] of [
    ['forward', 'pipeline call'], ['verification', 'ownership verification'], ['policy_exit', 'compiler policy exit']]) {
    const full = executionDiagnosticCaption(diagnostic({ first_call: 'failed', failure_stage })).full;
    assert.ok(full.includes(label));
    assert.match(full, /Select Eager/);
  }
`);
""")


def test_one_attempt_never_regresses_but_new_loads_and_missing_data_replace_it():
    _node(r"""
run(`
  const running = diagnostic({ first_call: 'running' });
  assert.equal(mergeExecutionDiagnostics(running, diagnostic()).first_call, 'running');
  for (const first_call of ['returned', 'failed', 'cancelled']) {
    const terminal = diagnostic({ first_call,
      failure_stage: first_call === 'returned' ? null : 'verification' });
    for (const later of [diagnostic(), running, diagnostic({ first_call: 'returned' })]) {
      const kept = mergeExecutionDiagnostics(terminal, later);
      assert.equal(kept.first_call, first_call);
      assert.equal(kept.failure_stage, terminal.failure_stage);
    }
    // A load id is an identity, never a monotonically increasing clock.
    assert.equal(mergeExecutionDiagnostics(terminal, diagnostic({ load_id: 2 })).first_call, 'pending');
    assert.equal(mergeExecutionDiagnostics({ ...terminal, load_id: 50 }, diagnostic()).load_id, 1);
    assert.equal(mergeExecutionDiagnostics(terminal, null), null);
    assert.equal(mergeExecutionDiagnostics(terminal, { unexpected: 'payload' }), null);
  }
`);
""")


def test_models_use_loaded_identity_and_ignore_delayed_observations():
    _node(r"""
run(`
  receiveModelExecutionDiagnostics(status({ execution_diagnostics: diagnostic({ first_call: 'returned' }) }), 200);
  assert.ok(modelExecutionNote().startsWith('Loaded actual/model:'));
  assert.match(modelExecutionNote(), /First call returned/);
  // A newer unsaved picker choice is not the loaded checkpoint.
  const modelChoice = 'draft/other';
  assert.equal(modelExecutionNote().includes('draft/other'), false);
  receiveModelExecutionDiagnostics(status(), 199);
  receiveModelExecutionDiagnostics(status({ execution_diagnostics: null }), NaN);
  assert.match(modelExecutionNote(), /First call returned/);
  receiveModelExecutionDiagnostics(status(), 201);
  assert.match(modelExecutionNote(), /First call returned/);
  modelRevision = 8;
  assert.equal(modelExecutionNote(), '');
  receiveModelExecutionDiagnostics(status({ model_revision: 8 }), 202);
  assert.match(modelExecutionNote(), /no forward attempt has been observed/);
  serverStartedAt = 101;
  assert.equal(modelExecutionNote(), '');
  receiveModelExecutionDiagnostics(status({ model_revision: 8, started_at: 101 }), 203);
  assert.match(modelExecutionNote(), /no forward attempt has been observed/);
  modelRuntimeSettings = { model_id: 'another/loaded' };
  assert.equal(modelExecutionNote(), '');
  modelRuntimeSettings = { model_id: 'actual/model' };
  serverOffline = true;
  assert.equal(modelExecutionNote(), '');
  serverOffline = false;
  modelState = 'unloaded';
  assert.equal(modelExecutionNote(), '');
`);
""")


def test_models_clear_invalid_absent_unloaded_and_eager_observations():
    _node(r"""
run(`
  for (const next of [
    status({ execution_diagnostics: null }), status({ execution_diagnostics: undefined }),
    status({ execution_diagnostics: diagnostic({ private_error: 'untrusted' }) }),
    status({ state: 'unloaded' }), status({ state: 'loading' }), status({ state: 'error' })]) {
    modelExecutionAsked = -Infinity;
    receiveModelExecutionDiagnostics(status(), 1);
    assert.notEqual(modelExecutionNote(), '');
    receiveModelExecutionDiagnostics(next, 2);
    assert.equal(modelExecutionNote(), '');
  }
  // Observations cannot open the existing execution-availability gate.
  syncExecutionControls = () => {};
  receiveExecutionStatus(status({ execution: { compilation: 'regional-v1' },
    execution_available: false, execution_unavailable_reason: 'Unqualified recipe' }), 3);
  assert.equal(selectedExecutionAvailable, false);
  assert.equal(selectedExecutionReason, 'Unqualified recipe');
  receiveExecutionStatus(status({ execution: null, execution_diagnostics: null,
    execution_available: true }), 4);
  assert.equal(selectedExecution, null);
  assert.equal(selectedExecutionAvailable, true);
  assert.equal(modelExecutionNote(), '');
`);
""")


def test_progress_preserves_counters_and_ignores_another_jobs_observation():
    _node(r"""
run(`
  setExecutionProgress('job-1', diagnostic({ first_call: 'running' }), 'image 2/4');
  assert.ok(progress().spoken.includes('image 2/4'));
  assert.match(progress().spoken, /first call running/);
  assert.equal(progress().shown, null);
  const title = els.progressText.title;
  assert.equal(els.progressText.getAttribute('aria-description'), title);
  setExecutionProgress('old-job', diagnostic({ first_call: 'failed', failure_stage: 'forward' }), 'old');
  assert.ok(progress().spoken.includes('image 2/4'));
  assert.equal(els.progressText.title, title);
  setExecutionProgress('job-1', diagnostic({ first_call: 'running' }), 'denoising', '3/12');
  assert.ok(progress().shown.includes('3/12'));
  assert.match(progress().shown, /first call running/);
  setExecutionProgress('job-1', null, 'image 3/4');
  assert.equal(progress().spoken, 'image 3/4');
  assert.equal(els.progressText.title, '');
  assert.equal(els.progressText.getAttribute('aria-description'), null);
  setExecutionProgress('job-1', diagnostic({ unknown: true }), 'denoising', '4/12');
  assert.equal(progress().shown, '4/12');
  assert.equal(els.progressText.title, '');
  setExecutionProgress('job-1', diagnostic(), 'denoising');
  setProgressText('finishing this step');
  assert.equal(els.progressText.title, '');
  assert.equal(els.progressText.getAttribute('aria-description'), null);
`);
""")


STREAM_FIXTURE = r"""
run(`
  const watchedJobs = new Set(), submissions = new Map(), jobPlaceholders = new Map();
  const jobStreams = new Map(), jobsToShow = new Set();
  let finishStale = false, finishedData = null;
  const ANNOUNCED_KEY = 'announced';
  const sources = [];
  class EventSource {
    constructor() { sources.push(this); }
    close() { this.closed = true; }
    send(data) { this.onmessage({ data: JSON.stringify(data) }); }
  }
  const syncFinishLead = () => {}, isForeign = () => true;
  const clearInterval = () => {}, viewerWaiting = () => {};
  setProgress = () => {}; setIndeterminate = () => {}; setStep = () => {}; mirrorJob = () => {};
  showCell = () => {}; syncPause = () => {}; syncNow = () => {};
  const noteViewer = () => {}, forgetJob = () => {}, setBusy = () => {};
  const remember = () => {}, watchNext = () => {};
  const finishJob = (_id, data) => { finishedData = data; };
`);
load('job-stream.js');
"""


def test_stream_uses_its_own_job_snapshot_and_terminal_data():
    _node(STREAM_FIXTURE + r"""
run(`
  // Models describes another load. Its result must never leak into this job.
  receiveModelExecutionDiagnostics(status({ execution_diagnostics:
    diagnostic({ load_id: 90, first_call: 'failed', failure_stage: 'forward' }) }), 1);
  streamJob('job-1');
  sources[0].send({ type: 'status', status: 'running', run: 2, total_runs: 4,
    execution_diagnostics: diagnostic({ first_call: 'running' }) });
  assert.ok(progress().spoken.includes('image 2/4'));
  assert.match(progress().spoken, /first call running/);
  assert.doesNotMatch(els.progressText.title, /failed/);
  sources[0].send({ type: 'progress', run: 2, total_runs: 4, step: 3, total_steps: 12,
    execution_diagnostics: diagnostic({ first_call: 'returned' }) });
  sources[0].send({ type: 'status', status: 'running', run: 3, total_runs: 4,
    execution_diagnostics: diagnostic() });
  assert.ok(progress().spoken.includes('image 3/4'));
  assert.match(progress().spoken, /first call returned/);
  sources[0].send({ type: 'done', status: 'done', execution_diagnostics: diagnostic() });
  assert.equal(finishedData.execution_diagnostics.first_call, 'returned');
  assert.equal(finishedData.execution_diagnostics.load_id, 1);
  assert.equal(sources[0].closed, true);
`);
""")


def test_stream_clears_missing_observations_and_separates_new_job_load_identity():
    _node(STREAM_FIXTURE + r"""
run(`
  streamJob('job-1');
  sources[0].send({ type: 'status', status: 'running',
    execution_diagnostics: diagnostic({ first_call: 'running' }) });
  assert.match(els.progressText.title, /Regional registered/);
  sources[0].send({ type: 'progress', step: 1, total_steps: 12 });
  assert.equal(els.progressText.title, '');
  assert.equal(progress().shown, '1/12');
  sources[0].send({ type: 'done', status: 'done' });
  assert.equal(finishedData.execution_diagnostics, null);
  currentJob = 'job-2';
  streamJob('job-2');
  sources[1].send({ type: 'status', status: 'running',
    execution_diagnostics: diagnostic({ load_id: 2 }) });
  assert.match(progress().spoken, /first call pending/);
  assert.match(els.progressText.title, /no forward attempt has been observed/);
  sources[1].send({ type: 'done', status: 'done',
    execution_diagnostics: diagnostic({ load_id: 2, first_call: 'returned' }) });
  assert.equal(finishedData.execution_diagnostics.load_id, 2);
`);
""")


def test_finished_job_keeps_its_frozen_observation_until_own_finish_is_replaced():
    _node(r"""
run(`
  els.finishNote = new InertElement();
  els.error = { textContent: '', classList: { toggle() {} } };
  els.generate = new InertElement();
  const ownImageIds = new Set(), cancelledHere = new Set();
  const ANOTHER_TABS_JOB = 'Another tab job';
  let busy = false, elsewhereCatchUp = false, lastSaid = '';
  const showError = () => {}, sayJob = (text) => { lastSaid = text; };
  holdTitle = () => {}; notifyJobDone = () => {};
  const refreshStatus = () => {}, invalidateFinishRuns = () => {}, loadRuns = () => {};
`);
load('job-results.js');
run(`
  const done = { status: 'done', images: [], image_count: 0, started_at: 1, finished_at: 2,
    execution_diagnostics: diagnostic({ load_id: 3, first_call: 'returned' }) };
  finishJob('job-1', done, { sheet: false });
  assert.match(els.finishNote.textContent, /first call returned/);
  assert.match(els.finishNote.title, /after ownership checks and policy exit/);
  assert.equal(els.finishNote.getAttribute('aria-description'), els.finishNote.title);
  assert.match(lastSaid, /first call returned/);
  const previous = els.finishNote.title;
  // Polling another load cannot rewrite a completed job's result.
  receiveModelExecutionDiagnostics(status({ execution_diagnostics: diagnostic() }), 5);
  assert.equal(els.finishNote.title, previous);
  finishJob('foreign-job', { ...done, execution_diagnostics:
    diagnostic({ load_id: 4, first_call: 'failed', failure_stage: 'forward' }) }, { sheet: false, foreign: true });
  assert.equal(els.finishNote.title, previous);
  assert.match(els.finishNote.textContent, /first call returned/);
  finishJob('job-2', { ...done, execution_diagnostics: null }, { sheet: false });
  assert.equal(els.finishNote.title, '');
  assert.equal(els.finishNote.getAttribute('aria-description'), null);
  assert.doesNotMatch(els.finishNote.textContent, /first call/);
  setFinish(['replacement']);
  assert.equal(els.finishNote.textContent, 'replacement');
`);
""")
