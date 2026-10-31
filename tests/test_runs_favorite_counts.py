"""Favorite callers refresh Runs once after confirmation, using the actual script.

The DOM/API/aggregate reader are observation ports. Native controls and the
actual server's aggregate values are qualified separately by the fresh smoke.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.usefixtures("node")

HARNESS = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ assert });
vm.runInContext(`
const galleryRecords = [], viewerRecords = [], sessionRecords = [], lbList = [];
const selection = new Map();
let stashedSelection = { picks: new Map() }, lbIndex = 0, binMode = false;
let loadedModel = 'current/model', modelRevision = 1, serverStartedAt = 11;
const els = { lightbox: { open: false }, galleryError: {}, allModels: { checked: false } };
const document = { activeElement: null, querySelectorAll: () => [] };
const PICKED_NONE = 'Pick images first';
const writes = [], reads = [], notes = [], forgotten = [], spoken = [], holdStates = [];
function api(path, options) {
  return new Promise((resolve, reject) => writes.push({ path, options, resolve, reject }));
}
function loadRuns() {
  const scope = { model: loadedModel, revision: modelRevision,
    allModels: els.allModels.checked, server: serverStartedAt };
  const applied = {
    favorites: [galleryRecords, viewerRecords, sessionRecords, lbList,
      [...selection.values()], [...stashedSelection.picks.values()]].flat()
      .map(({ id, favorite }) => ({ id, favorite })),
    spoken: spoken.map((note) => ({ ...note })), notes: notes.map((note) => note.message),
    forgotten: JSON.parse(JSON.stringify(forgotten)),
  };
  return new Promise((resolve, reject) => reads.push({ scope, applied, resolve, reject }));
}
function refuseBinned() { return binMode; }
function syncPickedButton() { holdStates.push([...favoriteWriteIds].sort((a, b) => a - b)); }
function selectedIds() { return [...selection.keys()]; }
function showError(message, informational, slot) { notes.push({ message, informational, slot }); }
function imageNames(ids) { return ids.map((id) => '#' + id).join(', '); }
function plural(number, word) { return number + ' ' + word + (number === 1 ? '' : 's'); }
function records(id) {
  const copies = Array.from({ length: 6 }, () => ({ id, favorite: false }));
  galleryRecords.push(copies[0]); viewerRecords.push(copies[1]); sessionRecords.push(copies[2]);
  selection.set(id, copies[3]); stashedSelection.picks.set(id, copies[4]); lbList.push(copies[5]);
  return copies;
}
const result = (extra = {}) => ({ changed: [], already: [], binned: [], missing: [], ...extra });
async function turn() { for (let i = 0; i < 6; i++) await Promise.resolve(); }
function released() { assert.equal(favoriteWriteIds.size, 0); assert.deepEqual(holdStates.at(-1), []); }
`, context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context, { filename: 'web/gallery-actions.js' });
// Unrelated deletion presentation and live announcements are observation ports.
// Favorite admission, API catches, propagation, refresh call sites and finally
// cleanup remain the complete actual module implementations.
vm.runInContext(`
forgetImages = (ids, options = {}) => { forgotten.push({ ids, options }); return ''; };
sayStar = (record, value) => { spoken.push({ id: record.id, value }); };
`, context);
let completed = false;
(async () => { await vm.runInContext(process.argv[2], context); })().then(() => {
  completed = true;
}, (error) => {
  completed = true; console.error(error.stack || error); process.exitCode = 1;
});
process.once('beforeExit', () => {
  if (!completed) { console.error('favorite case did not settle'); process.exitCode = 1; }
});
'''

CASES = {
    "single_confirmed_star_refreshes_once_after_propagation": r'''(async () => {
      const copies = records(1); const pending = setFavorite(copies[0], true, { say: true });
      assert.equal(writes.length, 1); assert.equal(reads.length, 0);
      assert.ok(copies.every((record) => !record.favorite));
      writes[0].resolve({ id: 1, favorite: true }); await pending;
      assert.ok(copies.every((record) => record.favorite));
      assert.deepEqual(spoken, [{ id: 1, value: true }]);
      assert.equal(reads.length, 1); released();
      assert.ok(reads[0].applied.favorites.every((record) => record.favorite));
      assert.deepEqual(reads[0].applied.spoken, [{ id: 1, value: true }]);
      assert.equal(writes[0].path, '/api/images/1/favorite?value=true');
    })()''',
    "single_confirmed_unstar_refreshes_once": r'''(async () => {
      const copies = records(1); copies.forEach((record) => { record.favorite = true; });
      const pending = setFavorite(copies[0], false);
      writes[0].resolve({ id: 1, favorite: false }); await pending;
      assert.ok(copies.every((record) => !record.favorite));
      assert.equal(reads.length, 1); released();
    })()''',
    "single_claim_blocks_overlap_while_write_waits_without_read": r'''(async () => {
      const copies = records(1); const pending = setFavorite(copies[0], true);
      await setFavorite(copies[0], false); await starPicked(false);
      assert.equal(writes.length, 1); assert.equal(reads.length, 0);
      assert.deepEqual([...favoriteWriteIds], [1]);
      assert.ok(notes.every((note) => /Wait for the current star change/.test(note.message)));
      writes[0].resolve({ id: 1, favorite: true }); await pending;
      assert.equal(reads.length, 1); released();
    })()''',
    "ordinary_single_failure_does_not_refresh_or_apply": r'''(async () => {
      const copies = records(1); const pending = setFavorite(copies[0], true);
      writes[0].reject(Error('controlled pre-dispatch refusal')); await pending;
      assert.equal(reads.length, 0); assert.ok(copies.every((record) => !record.favorite));
      assert.match(notes.at(-1).message, /controlled pre-dispatch refusal/); released();
    })()''',
    "unknown_single_keeps_its_existing_one_refresh": r'''(async () => {
      const copies = records(1); const pending = setFavorite(copies[0], true);
      writes[0].reject(Error('unknown image')); await pending;
      assert.equal(reads.length, 1); assert.deepEqual(forgotten, [{ ids: [1], options: {} }]);
      assert.ok(copies.every((record) => !record.favorite)); released();
    })()''',
    "bin_guard_does_not_write_or_refresh": r'''(async () => {
      const copies = records(1); binMode = true; await setFavorite(copies[0], true);
      assert.equal(writes.length, 0); assert.equal(reads.length, 0);
      assert.equal(favoriteWriteIds.size, 0);
    })()''',
    "bulk_multiple_images_refresh_once_not_per_image": r'''(async () => {
      const copies = [records(1), records(2), records(3)]; const ids = [...selection.keys()];
      const pending = starPicked(true);
      assert.equal(writes.length, 1); assert.equal(reads.length, 0);
      assert.deepEqual(JSON.parse(writes[0].options.body), { ids, value: true });
      writes[0].resolve(result({ changed: [1, 2], already: [3] })); await pending;
      assert.ok(copies.flat().every((record) => record.favorite));
      assert.equal(reads.length, 1); assert.deepEqual([...selection.keys()], ids); released();
      assert.ok(reads[0].applied.favorites.every((record) => record.favorite));
      assert.ok(reads[0].applied.notes.some((message) => message.includes('Starred')));
    })()''',
    "bulk_already_only_still_refreshes_once": r'''(async () => {
      records(1); records(2); const pending = starPicked(false);
      writes[0].resolve(result({ already: [1, 2] })); await pending;
      assert.equal(reads.length, 1); assert.equal(notes.at(-1).message, '#1 and #2 were not starred.'); released();
    })()''',
    "bulk_partial_result_keeps_one_refresh_and_forgetting": r'''(async () => {
      const a = records(1), b = records(2); records(3); records(4);
      const pending = starPicked(true);
      writes[0].resolve(result({ changed: [1], already: [2], binned: [3], missing: [4] })); await pending;
      assert.ok([...a, ...b].every((record) => record.favorite));
      assert.equal(reads.length, 1);
      assert.deepEqual(forgotten, [{ ids: [3], options: { binned: true } }, { ids: [4], options: {} }]);
      assert.deepEqual(reads[0].applied.forgotten, forgotten);
      released();
    })()''',
    "empty_bulk_guard_does_not_write_or_refresh": r'''(async () => {
      await starPicked(true); assert.equal(writes.length, 0); assert.equal(reads.length, 0);
      assert.match(notes.at(-1).message, /Pick images first/); assert.equal(favoriteWriteIds.size, 0);
    })()''',
    "bulk_claim_blocks_whole_overlap_and_still_allows_disjoint_single": r'''(async () => {
      const a = records(1), b = records(2); const pending = starPicked(true);
      await setFavorite(a[0], false); assert.equal(writes.length, 1); assert.equal(reads.length, 0);
      const c = records(3); const other = setFavorite(c[0], true);
      assert.equal(writes.length, 2); assert.equal(reads.length, 0);
      writes[1].resolve({ id: 3, favorite: true }); await other; assert.equal(reads.length, 1);
      assert.deepEqual([...favoriteWriteIds].sort((x, y) => x - y), [1, 2]);
      writes[0].resolve(result({ changed: [1, 2] })); await pending;
      assert.equal(reads.length, 2); assert.ok([...a, ...b].every((record) => record.favorite)); released();
    })()''',
    "ordinary_bulk_failure_does_not_refresh_or_apply": r'''(async () => {
      const copies = [records(1), records(2)]; const pending = starPicked(true);
      writes[0].reject(Error('controlled bulk refusal')); await pending;
      assert.equal(reads.length, 0); assert.ok(copies.flat().every((record) => !record.favorite));
      assert.match(notes.at(-1).message, /controlled bulk refusal/); released();
    })()''',
    "held_runs_read_does_not_extend_single_write_ownership": r'''(async () => {
      const copies = records(1); let settled = false;
      const pending = setFavorite(copies[0], true).then(() => { settled = true; });
      writes[0].resolve({ id: 1, favorite: true }); await turn();
      assert.equal(reads.length, 1); assert.equal(settled, true); released();
      // Aggregate read is deliberately still unresolved. A later deliberate
      // opposite write must be admitted after the confirmed favorite settles.
      const opposite = setFavorite(copies[0], false); assert.equal(writes.length, 2);
      writes[1].resolve({ id: 1, favorite: false }); await opposite; await pending;
      assert.equal(reads.length, 2); released();
    })()''',
    "held_runs_read_does_not_extend_bulk_write_ownership": r'''(async () => {
      records(1); records(2); let settled = false;
      const pending = starPicked(true).then(() => { settled = true; });
      writes[0].resolve(result({ changed: [1, 2] })); await turn();
      assert.equal(reads.length, 1); assert.equal(settled, true); released(); await pending;
    })()''',
    "delayed_disjoint_failed_reply_does_not_add_a_refresh": r'''(async () => {
      const a = records(1), b = records(2);
      const older = setFavorite(a[0], true), newer = setFavorite(b[0], true);
      writes[1].resolve({ id: 2, favorite: true }); await newer;
      assert.equal(reads.length, 1); assert.ok(b.every((record) => record.favorite));
      writes[0].reject(Error('controlled nonforwarded refusal')); await older;
      assert.equal(reads.length, 1); assert.ok(a.every((record) => !record.favorite)); released();
    })()''',
    "confirmed_reply_refreshes_current_scope_after_context_change": r'''(async () => {
      const copies = records(1); const pending = setFavorite(copies[0], true);
      loadedModel = 'new/model'; modelRevision = 2; els.allModels.checked = true; serverStartedAt = 12;
      writes[0].resolve({ id: 1, favorite: true }); await pending;
      assert.deepEqual(reads.map((read) => read.scope), [
        { model: 'new/model', revision: 2, allModels: true, server: 12 },
      ]); released();
    })()''',
    "direct_per_record_propagation_never_reads_runs": r'''(async () => {
      records(1); records(2); markFavorite(1, true); markFavorite(2, false);
      assert.equal(writes.length, 0); assert.equal(reads.length, 0);
    })()''',
}


@pytest.mark.parametrize("case", CASES)
def test_confirmed_favorite_callers_refresh_runs_once(case):
    done = subprocess.run(
        ["node", "-e", HARNESS, str(ROOT / "web/gallery-actions.js"), CASES[case]],
        capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stdout + done.stderr
