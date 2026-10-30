// Runs web/gallery-select.js in a VM with stubbed page globals and drives
// pickRun (UI#37) through one case named on the command line, printing the
// outcome as JSON: the picks, the gallery line and the reads asked for.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const [root, scenario] = process.argv.slice(2);
const said = [];
const asked = [];
let release = null;
const gate = new Promise((go) => { release = go; });
const element = () => ({
  classList: { toggle() {}, contains: () => false }, setAttribute() {}, removeAttribute() {},
  getAttribute: () => null, querySelectorAll: () => [], querySelector: () => null,
  replaceChildren() {}, append() {}, focus() {}, dataset: {}, title: '', textContent: '', hidden: false,
});
const run = (n, start) => Array.from({ length: n }, (_, i) => ({ id: start + i, run_id: 'r1', seed: i }));
const rows = run(6, 10);
const sandbox = vm.createContext({
  els: new Proxy({}, { get: (t, k) => (t[k] ??= element()) }),
  URLSearchParams, sheetPage: 1024, sheetMode: false, binMode: false, binnedIds: new Set(), deletedIds: new Set(),
  galleryRecords: [], galleryMore: false, total: 0, document: { activeElement: null },
  plural: (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`,
  showError: (text) => { said.push(text); },
  syncPickedTier() {}, favoriteWriteReason: () => '', binWriteReason: () => '',
  sizePresetMatch: () => true, pickedTier: 2, BIN_HELD: 'held',
  api: async (url) => {
    asked.push(url);
    if (scenario === 'fails') throw new Error('fixture refusal');
    if (scenario !== 'plain' && scenario !== 'cap' && scenario !== 'none') await gate;
    const limit = +new URL(url, 'http://x').searchParams.get('limit');
    return { items: rows.slice(0, limit), total: rows.length };
  },
});
const source = fs.readFileSync(path.join(root, 'web', 'gallery-select.js'), 'utf8');
// The picks' marks, count and listeners are the browser's (smoke-select).
new vm.Script(`${source}\n;syncSelection = () => {};\nglobalThis.__sel = { selection, pickRun, clearSelection, stashSelection, setCap: (n) => { selectionCap = () => n; } };`,
  { filename: 'gallery-select.js' }).runInContext(sandbox);
const { selection, pickRun, clearSelection, stashSelection, setCap } = sandbox.__sel;
selection.set(1, { id: 1 });
let pending;
if (scenario === 'none') pending = pickRun({ id: 5, run_id: null, job_id: null });
else {
  if (scenario === 'cap') setCap(4);
  pending = pickRun({ id: 10, run_id: 'r1' });
  await Promise.resolve();
  if (scenario === 'cleared') clearSelection();
  if (scenario === 'bin') { stashSelection(); sandbox.binMode = true; }
  if (scenario === 'forgotten') sandbox.binnedIds.add(12);
  release();
}
await pending;
process.stdout.write(JSON.stringify({ picks: [...selection.keys()], said, asked }));
