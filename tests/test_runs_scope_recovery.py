"""Actual Runs handlers with deferred API reads and inert presentation ports."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")

HARNESS = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ assert, URLSearchParams });
vm.runInContext(`
class TestNode {
  constructor() { this.children = []; this.parentElement = null; this.dataset = {}; }
  append(...children) {
    for (const child of children) { child.parentElement = this; this.children.push(child); }
    const select = this.selectRoot();
    if (select && !select.chosen) select.chosen = select.options[0] ?? null;
  }
  selectRoot() { let node = this; while (node && !(node instanceof TestSelect)) node = node.parentElement; return node; }
  remove() {
    const select = this.selectRoot();
    if (this.parentElement) this.parentElement.children = this.parentElement.children.filter((child) => child !== this);
    this.parentElement = null;
    if (select?.chosen === this) select.chosen = select.options[0] ?? null;
  }
}
class TestOption extends TestNode {
  constructor(text, value) { super(); this.text = text; this.value = String(value); this.disabled = false; this.hidden = false; }
  get textContent() { return this.text; }
  set textContent(value) { this.text = value; }
}
class TestSelect extends TestNode {
  constructor() { super(); this.chosen = null; this.disabled = true; this.title = ''; }
  get options() {
    const flatten = (node) => node.children.flatMap((child) => child instanceof TestOption ? [child] : flatten(child));
    return flatten(this);
  }
  get selectedOptions() { return this.options.includes(this.chosen) ? [this.chosen] : []; }
  get selectedIndex() { return this.options.indexOf(this.chosen); }
  set selectedIndex(index) { this.chosen = this.options[index] ?? null; }
  get value() { return this.selectedOptions[0]?.value ?? ''; }
  set value(value) { this.chosen = this.options.find((option) => option.value === String(value)) ?? null; }
  set innerHTML(_value) { this.children = []; this.chosen = null; }
  get innerHTML() { return this.options.map((option) => option.text).join('|'); }
}
const Option = TestOption;
const els = { runList: new TestSelect(), allModels: { checked: false }, search: { value: '' } };
const document = { createElement: () => new TestNode(), querySelectorAll: () => [] };
let loadedModel = 'current/model', modelRevision = 1, serverStartedAt = 11, binMode = false;
const BIN_HELD = 'Not in the recycle bin - leave it with Bin';
const requests = [], opened = [], finishRefreshes = [], keyTimers = [];
function api(path) {
  return new Promise((resolve, reject) => { requests.push({ path, resolve, reject }); });
}
// These unchanged eligibility helpers are inert ports; the real handlers
// below must consult them. Native keyboard behavior is a separate smoke.
function keyboardEventHeld(event) { return !!event.held; }
function keyboardCommandModifier(event) { return !!(event.ctrlKey || event.metaKey || event.altKey); }
function setTimeout(callback) { keyTimers.push(callback); return keyTimers.length; }
function flushKeys() { while (keyTimers.length) keyTimers.shift()(); }
function clip(value) { return String(value); }
function promptGist(value) { return String(value); }
function plural(value, word) { return value + ' ' + word + (value === 1 ? '' : 's'); }
function row(run_id) {
  return { run_id, images: 2, jobs: 1, favorites: 0, prompts: 1, axes: {},
    sizes: [[256, 256]], size_count: 1, width: 256, height: 256,
    started_at: '2026-10-28T12:00:00Z', prompt: 'Run ' + run_id, seed: 7 };
}
function key(value, extra = {}) {
  return { key: value, repeat: false, ...extra, prevented: false,
    preventDefault() { this.prevented = true; } };
}
const realOptions = () => [...els.runList.options].filter((option) => option.value && !option.disabled && option.dataset.runsRetry !== 'true');
const optionFor = (id) => realOptions().find((option) => option.value === id);
const retryOption = () => [...els.runList.options].find((option) => option.dataset.runsRetry === 'true');
function selectOption(option) { els.runList.selectedIndex = [...els.runList.options].indexOf(option); }
function browse(option) {
  runListKey(key('ArrowDown'));
  selectOption(option);
  runListChange();
  flushKeys();
}
async function publish(rows = [row('alpha'), row('beta')]) {
  const pending = loadRuns(); requests.at(-1).resolve(rows); await pending;
}
async function failedNewScope() {
  els.allModels.checked = true;
  await publish([row('foreign')]);
  els.allModels.checked = false;
  const pending = loadRuns(); requests.at(-1).reject(Error('controlled current read refusal')); await pending;
  assert.ok(retryOption(), 'a failed current scope needs a deliberate Retry option');
  return retryOption();
}
async function failedSameScope() {
  await publish();
  const pending = loadRuns(); requests.at(-1).reject(Error('controlled current refresh refusal')); await pending;
  assert.ok(retryOption(), 'a failed refresh needs a deliberate Retry option');
  return retryOption();
}
async function turn() { await Promise.resolve(); await Promise.resolve(); }
`, context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context, { filename: 'web/runs.js' });
// Sheet opening and unrelated retained-run tool refresh are observation ports.
// The actual load/sync/key/change/commit handlers remain unmodified.
vm.runInContext(`showRun = (id) => { opened.push(id); }; refreshFinishRuns = (force) => { finishRefreshes.push(force); };`, context);
(async () => { await vm.runInContext(process.argv[2], context); })().catch((error) => {
  console.error(error.stack || error); process.exitCode = 1;
});
'''

CASES = {
    "ordinary_current_grouping_and_pick": r'''(async () => {
      await publish();
      assert.equal(els.runList.disabled, false);
      assert.equal(els.runList.options[0].text, 'Runs\u2026');
      assert.ok(els.runList.options[0].disabled && els.runList.options[0].hidden);
      assert.deepEqual(realOptions().map((option) => option.value), ['alpha', 'beta']);
      selectOption(optionFor('beta')); runListChange();
      assert.deepEqual(opened, ['beta']); assert.deepEqual(finishRefreshes, [true]);
    })()''',
    "new_scope_clears_foreign_options_while_read_is_pending": r'''(async () => {
      els.allModels.checked = true; await publish([row('foreign')]);
      els.allModels.checked = false; const pending = loadRuns();
      assert.deepEqual(realOptions().map((option) => option.value), [], 'old-scope runs were still offered while current read waited');
      assert.equal(els.runList.disabled, true); assert.equal(els.runList.options[0].text, 'Loading runs…');
      requests.at(-1).resolve([]); await pending;
      assert.equal(els.runList.options[0].text, 'No runs yet'); assert.equal(els.runList.disabled, true);
    })()''',
    "current_scope_failure_does_not_offer_foreign_runs": r'''(async () => {
      await failedNewScope();
      assert.deepEqual(realOptions().map((option) => option.value), []);
      assert.equal(els.runList.options[0].text, 'Runs unavailable');
      assert.equal(els.runList.disabled, false); assert.match(els.runList.title, /Retry loading runs/);
      assert.equal(requests.length, 2); assert.deepEqual(opened, []);
    })()''',
    "scope_boundary_retires_real_keyboard_candidate_before_blur": r'''(async () => {
      await publish(); browse(optionFor('beta')); assert.equal(runPending, 'beta');
      els.allModels.checked = true; const pending = loadRuns();
      commitRunList();
      assert.deepEqual(opened, [], 'blur opened the old-scope keyboard candidate');
      assert.equal(runPending, null);
      requests.at(-1).resolve([row('new')]); await pending;
    })()''',
    "synthetic_stale_change_during_pending_scope_cannot_open_sheet": r'''(async () => {
      await publish(); const old = optionFor('alpha');
      els.allModels.checked = true; const pending = loadRuns();
      // An inert stale UI event must not bypass the published-scope guard.
      els.runList.append(old); selectOption(old); runListChange(); commitRunList();
      assert.deepEqual(opened, []);
      requests.at(-1).resolve([]); await pending;
    })()''',
    "latest_success_survives_superseded_failure": r'''(async () => {
      await publish(); const old = loadRuns(), oldRequest = requests.at(-1);
      els.allModels.checked = true; const current = loadRuns(); requests.at(-1).resolve([row('current')]); await current;
      oldRequest.reject(Error('obsolete refusal')); await old;
      assert.deepEqual(realOptions().map((option) => option.value), ['current']);
      assert.equal(retryOption(), undefined); assert.equal(els.runList.disabled, false);
    })()''',
    "latest_success_survives_superseded_success": r'''(async () => {
      const old = loadRuns(), oldRequest = requests.at(-1);
      els.allModels.checked = true; const current = loadRuns(); requests.at(-1).resolve([row('current')]); await current;
      oldRequest.resolve([row('obsolete')]); await old;
      assert.deepEqual(realOptions().map((option) => option.value), ['current']);
    })()''',
    "scope_round_trip_does_not_revive_cached_old_options": r'''(async () => {
      await publish(); els.allModels.checked = true; const middle = loadRuns(), middleRequest = requests.at(-1);
      els.allModels.checked = false; const latest = loadRuns();
      assert.deepEqual(realOptions().map((option) => option.value), []);
      requests.at(-1).resolve([row('latest')]); await latest;
      middleRequest.resolve([row('middle')]); await middle;
      assert.deepEqual(realOptions().map((option) => option.value), ['latest']);
    })()''',
    "model_change_clears_options_before_await": r'''(async () => {
      await publish(); loadedModel = 'other/model'; const pending = loadRuns();
      assert.deepEqual(realOptions().map((option) => option.value), []);
      requests.at(-1).resolve([]); await pending;
    })()''',
    "revision_change_clears_options_before_await": r'''(async () => {
      await publish(); modelRevision++; const pending = loadRuns();
      assert.deepEqual(realOptions().map((option) => option.value), []);
      requests.at(-1).resolve([]); await pending;
    })()''',
    "known_restart_clears_options_with_same_model_and_revision": r'''(async () => {
      await publish(); serverStartedAt = 12; const pending = loadRuns();
      assert.deepEqual(realOptions().map((option) => option.value), []);
      requests.at(-1).reject(Error('new server unavailable')); await pending;
      assert.ok(retryOption()); assert.deepEqual(realOptions().map((option) => option.value), []);
    })()''',
    "first_identity_discovery_can_settle_then_is_bound": r'''(async () => {
      serverStartedAt = null; const pending = loadRuns();
      serverStartedAt = 11; requests.at(-1).resolve([row('learned')]); await pending;
      assert.ok(optionFor('learned'), 'first identity discovery stranded a useful current reply');
      serverStartedAt = 12; const restarted = loadRuns();
      assert.deepEqual(realOptions().map((option) => option.value), [], 'unknown capture remained a permanent server wildcard');
      requests.at(-1).resolve([]); await restarted;
    })()''',
    "known_old_identity_reply_is_ignored_without_new_request": r'''(async () => {
      const pending = loadRuns(); serverStartedAt = 12;
      requests.at(-1).resolve([row('old-server')]); await pending;
      assert.deepEqual(realOptions().map((option) => option.value), []);
    })()''',
    "same_scope_pending_refresh_keeps_real_keyboard_candidate": r'''(async () => {
      await publish(); const original = optionFor('beta'); browse(original);
      const pending = loadRuns();
      assert.equal(optionFor('beta'), original); assert.equal(runPending, 'beta');
      requests.at(-1).resolve([row('alpha'), row('beta')]); await pending;
      assert.equal(runPending, 'beta'); commitRunList(); assert.deepEqual(opened, ['beta']);
    })()''',
    "same_scope_failure_appends_retry_without_rebuilding_candidate": r'''(async () => {
      await publish(); const original = optionFor('beta'); browse(original);
      const pending = loadRuns(); requests.at(-1).reject(Error('refresh failed')); await pending;
      assert.equal(optionFor('beta'), original); assert.equal(runPending, 'beta'); assert.ok(retryOption());
      commitRunList(); assert.deepEqual(opened, ['beta']);
    })()''',
    "repeated_current_failures_keep_one_retry_option": r'''(async () => {
      await failedSameScope();
      const pending = loadRuns(); requests.at(-1).reject(Error('second failure')); await pending;
      assert.equal([...els.runList.options].filter((option) => option.dataset.runsRetry === 'true').length, 1);
      assert.deepEqual(realOptions().map((option) => option.value), ['alpha', 'beta']);
    })()''',
    "retry_browsing_then_enter_and_blur_issues_one_read": r'''(async () => {
      const retry = await failedNewScope(); const count = requests.length;
      browse(retry); assert.equal(requests.length, count); assert.deepEqual(opened, []);
      const enter = key('Enter'); runListKey(enter); assert.equal(enter.prevented, true);
      assert.equal(requests.length, count + 1); commitRunList(); runListKey(key('Enter', { repeat: true }));
      assert.equal(requests.length, count + 1); assert.deepEqual(opened, []);
      requests.at(-1).resolve([]); await turn(); assert.equal(els.runList.options[0].text, 'No runs yet');
    })()''',
    "retry_browsing_then_blur_issues_one_read": r'''(async () => {
      const retry = await failedNewScope(); const count = requests.length;
      browse(retry); commitRunList(); commitRunList();
      assert.equal(requests.length, count + 1); assert.deepEqual(opened, []);
      requests.at(-1).resolve([]); await turn();
    })()''',
    "retry_escape_cancels_candidate_without_read": r'''(async () => {
      const retry = await failedNewScope(); const count = requests.length;
      browse(retry); const escape = key('Escape'); runListKey(escape); commitRunList();
      assert.equal(escape.prevented, true); assert.equal(requests.length, count); assert.deepEqual(opened, []);
      assert.equal(els.runList.value, ''); assert.ok(retryOption());
    })()''',
    "held_modified_and_repeat_enter_do_not_retry": r'''(async () => {
      const retry = await failedNewScope(); browse(retry); const count = requests.length;
      for (const flags of [{ held: true }, { ctrlKey: true }, { metaKey: true }, { altKey: true }, { shiftKey: true }, { repeat: true }]) {
        runListKey(key('Enter', flags)); assert.equal(requests.length, count);
      }
      runListKey(key('Enter')); assert.equal(requests.length, count + 1);
      requests.at(-1).resolve([]); await turn();
    })()''',
    "direct_retry_selection_never_routes_to_sheet": r'''(async () => {
      const retry = await failedNewScope(); const count = requests.length;
      selectOption(retry); runListChange(); commitRunList();
      assert.equal(requests.length, count + 1); assert.deepEqual(opened, []);
      requests.at(-1).resolve([]); await turn();
    })()''',
    "real_run_value_collision_and_retry_action_are_distinct": r'''(async () => {
      await publish([row('__runs_retry__')]);
      const pending = loadRuns(); requests.at(-1).reject(Error('refresh failed')); await pending;
      const actual = optionFor('__runs_retry__'), retry = retryOption(); assert.ok(actual && retry);
      assert.equal(actual.value, retry.value);
      selectOption(actual); runListChange(); assert.deepEqual(opened, ['__runs_retry__']);
      const count = requests.length; browse(retry);
      assert.equal(els.runList.selectedOptions[0], retry, 'retry selection was retargeted by value collision');
      runListKey(key('Enter')); assert.equal(requests.length, count + 1);
      assert.deepEqual(opened, ['__runs_retry__']); requests.at(-1).resolve([]); await turn();
    })()''',
    "empty_retry_value_is_owned_by_selected_option_marker": r'''(async () => {
      const retry = await failedNewScope(); retry.value = ''; const count = requests.length;
      browse(retry); assert.equal(els.runList.selectedOptions[0], retry);
      runListKey(key('Enter')); assert.equal(requests.length, count + 1); assert.deepEqual(opened, []);
      requests.at(-1).resolve([]); await turn();
    })()''',
    "background_success_retires_pending_retry": r'''(async () => {
      const retry = await failedSameScope(); browse(retry);
      const current = loadRuns(); requests.at(-1).resolve([row('alpha')]); await current;
      const count = requests.length; assert.equal(retryOption(), undefined);
      runListKey(key('Enter')); commitRunList();
      assert.equal(requests.length, count); assert.deepEqual(opened, []);
    })()''',
    "background_success_retires_disappeared_real_run_candidate": r'''(async () => {
      await publish(); browse(optionFor('beta'));
      const pending = loadRuns(); requests.at(-1).resolve([row('alpha')]); await pending;
      commitRunList(); assert.deepEqual(opened, [], 'blur opened a run absent from the new list');
      assert.equal(runPending, null);
    })()''',
    "bin_entry_retires_real_candidate_before_blur": r'''(async () => {
      await publish(); browse(optionFor('beta')); binMode = true; syncRunList(); commitRunList();
      assert.deepEqual(opened, []); assert.equal(runPending, null); assert.equal(els.runList.title, BIN_HELD);
      binMode = false; syncRunList(); commitRunList(); assert.deepEqual(opened, []);
    })()''',
    "bin_holds_retry_candidate_and_direct_change": r'''(async () => {
      const retry = await failedNewScope(); browse(retry); const count = requests.length;
      binMode = true; syncRunList(); commitRunList(); selectOption(retry); runListChange();
      assert.equal(requests.length, count); assert.deepEqual(opened, []); assert.equal(els.runList.title, BIN_HELD);
      binMode = false; syncRunList(); commitRunList(); assert.equal(requests.length, count);
    })()''',
    "direct_commit_checks_live_scope_without_request_side_effects": r'''(async () => {
      await publish(); browse(optionFor('beta')); const count = requests.length;
      loadedModel = 'changed/model'; commitRunList();
      assert.deepEqual(opened, []); assert.equal(requests.length, count); assert.equal(runPending, null);
    })()''',
    "search_and_sheet_open_ports_are_untouched_by_recovery": r'''(async () => {
      els.search.value = 'harbor cfg:3'; const before = els.search.value;
      const retry = await failedNewScope(); browse(retry); runListKey(key('Escape'));
      selectOption(retry); runListChange(); requests.at(-1).resolve([]); await turn();
      assert.equal(els.search.value, before); assert.deepEqual(opened, []);
    })()''',
}


@pytest.mark.parametrize("case", CASES)
def test_actual_runs_scope_and_recovery_handlers(case):
    done = subprocess.run(
        ["node", "-e", HARNESS, str(ROOT / "web/runs.js"), CASES[case]],
        capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stdout + done.stderr
