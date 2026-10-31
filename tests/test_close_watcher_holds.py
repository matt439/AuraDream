"""A forced dialog close keeps write and import holds (UI#35).

The actual declaration-only snippet and saved-setup modules run on inert DOM
and API ports. A repeated Escape's real close, focus and layout belong to
smoke-snippet-portability and smoke-library-write-recovery.
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
function element(id = '', tagName = 'DIV') {
  const classes = new Set(), attributes = new Map(), listeners = new Map();
  return { id, tagName, value: '', disabled: false, hidden: false, checked: false, open: false, title: '',
    textContent: '', className: '', children: [], dataset: {}, selectedIndex: 0, maxLength: -1,
    validity: { badInput: false, valid: true }, isConnected: true,
    classList: { add: name => classes.add(name), remove: name => classes.delete(name),
      contains: name => classes.has(name), toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
    setAttribute(name, value) { attributes.set(name, String(value)); },
    getAttribute(name) { return attributes.has(name) ? attributes.get(name) : null; },
    addEventListener(type, handler) { listeners.set(type, [...(listeners.get(type) || []), handler]); },
    fire(type, event = {}) { for (const handler of listeners.get(type) || []) handler({ type, preventDefault() {}, ...event }); },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    querySelectorAll(selector) {
      const tags = selector.split(',').map(part => part.trim().toUpperCase());
      const found = [];
      const walk = node => { for (const child of node.children || []) { if (tags.includes(child.tagName)) found.push(child); walk(child); } };
      walk(this);
      return found;
    },
    focus() {}, close() { this.open = false; }, showModal() { this.open = true; } };
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const settle = () => new Promise(resolve => setImmediate(resolve));
function fixture() {
  const calls = { requests: [] };
  const byId = new Map();
  const elements = {};
  const els = new Proxy(elements, { get: (target, key) => (target[key] ??= element(String(key))) });
  const document = {
    activeElement: null,
    getElementById: id => { if (!byId.has(id)) byId.set(id, element(id)); return byId.get(id); },
    createElement: tag => element('', tag.toUpperCase()),
    querySelector: () => element('panel'),
  };
  const sandbox = vm.createContext({ els, document, calls, Option: function (text, value) { return element(value, 'OPTION'); },
    api: (url, options) => { const request = { ...deferred(), url, method: options?.method || 'GET', options };
      calls.requests.push(request); return request.promise; },
    readableDetail: value => String(value), checkpointName: id => id, isShown: () => true,
    executionSubmitHold: () => '', performanceChoiceValue: () => '', panelNumberInputs: () => [],
    selectedExecution: null, EXECUTION_UNAVAILABLE: 'unavailable', loadedModel: 'model-A', lastDefaults: {},
    modelSwitching: false, pendingLora: '', referenceIds: [], owned: {}, modelRevision: 1,
    panelState: () => ({}), reuseFields: () => ({}), invalidExecutionDraft: null, serverStartedAt: 0, trScale: 1,
  });
  for (const name of ['prompt-snippets.js', 'import-merge.js', 'snippet-import.js', 'generation-presets.js', 'setup-import.js']) {
    new vm.Script(fs.readFileSync(path.join(root, 'web', name), 'utf8'), { filename: name }).runInContext(sandbox);
  }
  const evaluate = text => vm.runInContext(text, sandbox);
  evaluate('initSnippets(); initSnippetImport(); initSetups(); initSetupImport();');
  const importDialog = document.getElementById('snippet-import-dialog');
  // A forced close: the dialog shuts and its close event fires, whatever the held cancel said.
  const forceClose = dialog => { dialog.open = false; dialog.fire('close'); };
  const writes = (method, url) => calls.requests.filter(request => request.method === method && request.url === url).length;
  return { sandbox, els, calls, evaluate, importDialog, forceClose, writes };
}
const snippet = (id, name, revision = 1) => ({ id, name, text: 'text ' + name, target: 'prompt', revision });
const setup = (id, name) => ({ id, name, model_id: 'model-A', revision: 1, payload: { mode: 'generate', width: 512, height: 512,
  seed: null, num_images: 1, runs: 1, steps: null, guidance_scale: null, mu: null, true_cfg: null,
  prompt: 'p', negative_prompt: '', grid: null } });
function importRow(index, name, action, matches) {
  return { index, action, snippet: { name, text: 'incoming', target: 'prompt' }, matches, exact_ids: [], incoming_collisions: [] };
}
function shown(rows) {
  return { rows, add_count: rows.filter(row => ['add', 'copy'].includes(row.action)).length, replace_count: 0,
    skip_count: rows.filter(row => row.action === 'skip').length, can_import: true, reason: '',
    payload_digest: 'd', collection_revision: 'r' };
}
async function main() {
  const f = fixture();
  const { els, evaluate } = f;
  if (scenario === 'snippet-forced-close-keeps-write') {
    els.snippetsDialog.open = true;
    evaluate("snippetsReady = true; snippetsItems = []");
    els.snippetsName.value = 'Created once'; els.snippetsText.value = 'text';
    const first = evaluate('saveSnippet()');
    assert.equal(f.writes('POST', '/api/snippets'), 1);
    f.forceClose(els.snippetsDialog);
    assert.equal(evaluate('snippetsPending'), 'write', 'a forced close dropped the write hold');
    els.snippetsDialog.open = true;
    await evaluate('readSnippets({ opening: true })');
    assert.equal(f.calls.requests.length, 1, 'the reopening read ran during the write');
    assert.equal(els.snippetsSave.getAttribute('aria-disabled'), 'true');
    assert.equal(els.snippetsSave.title, 'Wait for the snippet to finish saving.');
    await evaluate('saveSnippet()');
    assert.equal(f.writes('POST', '/api/snippets'), 1, 'a second Save sent another POST');
    f.calls.requests[0].resolve({ items: [snippet('a', 'Created once')] });
    await first;
    assert.equal(evaluate('snippetsPending'), null);
    assert.equal(evaluate('snippetsSelected.id'), 'a');
  } else if (scenario === 'snippet-read-cleared-by-close') {
    els.snippetsDialog.open = true;
    const read = evaluate('readSnippets({ opening: true })');
    assert.equal(evaluate('snippetsPending'), 'read');
    f.forceClose(els.snippetsDialog);
    assert.equal(evaluate('snippetsPending'), null, 'a closed dialog kept its read pending');
    f.calls.requests[0].resolve({ items: [snippet('a', 'Late')] });
    await read;
    assert.equal(evaluate('snippetsItems.length'), 0, 'a retired read replaced the list');
  } else if (scenario.startsWith('snippet-import-forced-close')) {
    els.snippetsDialog.open = true; f.importDialog.open = true;
    evaluate("snippetsReady = true; snippetImportRaw = '{}'; snippetImportChoices = []");
    evaluate("snippetImportShown = " + JSON.stringify(shown([])));
    const commit = evaluate('commitSnippetImport()');
    const ticket = evaluate('snippetImportTicket');
    f.forceClose(f.importDialog);
    assert.equal(evaluate('snippetImportTicket'), ticket, 'a forced close retired the commit');
    assert.equal(evaluate('snippetImportBusy'), 'commit');
    assert.equal(evaluate('snippetsPending'), 'write');
    if (scenario.endsWith('success')) {
      f.calls.requests[0].resolve({ items: [snippet('a', 'Imported')] });
      await commit;
      assert.equal(evaluate('snippetsItems.length'), 1);
      assert.match(els.snippetsStatus.textContent, /^Imported/);
    } else {
      f.calls.requests[0].reject(new TypeError('lost reply'));
      await commit;
      assert.match(els.snippetsStatus.textContent, /uncertain/);
      assert.equal(els.snippetsStatus.classList.contains('error'), true);
      assert.equal(evaluate('snippetImportNeedsReload'), true);
    }
    assert.equal(evaluate('snippetsPending'), null, 'the reply left Snippets frozen in write');
    assert.equal(evaluate('snippetImportBusy'), null);
  } else if (scenario === 'snippet-import-close-retires-preview') {
    els.snippetsDialog.open = true; f.importDialog.open = true;
    evaluate("snippetImportRaw = '{}'");
    const preview = evaluate('previewSnippetImport()');
    const ticket = evaluate('snippetImportTicket');
    f.forceClose(f.importDialog);
    assert.equal(evaluate('snippetImportTicket'), ticket + 1);
    assert.equal(evaluate('snippetImportBusy'), null);
    f.calls.requests[0].resolve(shown([importRow(0, 'Late', 'add', [])]));
    await preview;
    assert.equal(evaluate('snippetImportShown'), null);
  } else if (scenario === 'snippet-copy-row-keeps-file-match') {
    els.snippetsDialog.open = true; f.importDialog.open = true;
    const local = snippet('a', 'Harbour');
    evaluate('renderSnippetImport(' + JSON.stringify(shown([importRow(0, 'Harbour', 'skip', [local])])) + ')');
    const copy = importRow(0, 'Harbour 2', 'copy', []);
    evaluate('renderSnippetImport(' + JSON.stringify(shown([copy])) + ')');
    const section = document_rows(f);
    assert.equal(section.children[0].textContent, '#1 Harbour (Prompt)');
    const options = section.querySelectorAll('option');
    assert.equal(options.find(option => option.value === 'replace').disabled, false);
    assert.equal(options.find(option => option.value === 'add').disabled, true);
    assert.equal(evaluate('snippetImportChoices[0].name'), 'Harbour 2');
    evaluate("snippetImportChoices[0] = { index: 0, action: 'add' }");
    assert.match(evaluate('snippetImportChoiceReason()'), /saved name match/);
    evaluate("snippetImportChoices[0] = { index: 0, action: 'replace', item_id: 'a', revision: 1 }");
    assert.equal(evaluate('snippetImportChoiceReason()'), '');
  } else if (scenario === 'snippet-refresh-preview-keeps-changed-choice') {
    els.snippetsDialog.open = true; f.importDialog.open = true;
    evaluate("snippetImportRaw = '{}'; snippetImportNeedsReload = true; snippetImportChoices = [{ index: 0, action: 'add' }]");
    const preview = evaluate('previewSnippetImport()');
    f.calls.requests[0].resolve({ items: [] }); await settle();
    f.calls.requests[1].resolve(shown([importRow(0, 'Fresh', 'add', [])])); await settle();
    assert.equal(f.calls.requests.length, 3);
    const controls = document_rows(f).querySelectorAll('select');
    assert.ok(controls.length > 0 && controls.every(control => control.disabled), 'redrawn rows were editable during the preview');
    evaluate("snippetImportChoices[0].action = 'skip'");
    f.calls.requests[2].resolve(shown([importRow(0, 'Fresh', 'add', [])]));
    await preview;
    assert.equal(evaluate('snippetImportChoices[0].action'), 'skip', 'the reply overwrote a changed choice');
    assert.equal(evaluate('snippetImportShown'), null);
    assert.equal(f.sandbox.document.getElementById('snippet-import-status').textContent, 'Choices changed. Preview again before importing.');
  } else if (scenario === 'setup-forced-close-keeps-write') {
    els.setupsDialog.open = true;
    evaluate('setupsReady = true; setupsItems = []');
    els.setupsName.value = 'Saved once';
    const first = evaluate("saveSetup('new')");
    assert.equal(f.writes('POST', '/api/presets'), 1);
    f.forceClose(els.setupsDialog);
    assert.equal(evaluate('setupsPending'), 'write', 'a forced close dropped the write hold');
    els.setupsDialog.open = true;
    await evaluate('reloadSetups()');
    assert.equal(f.calls.requests.length, 1, 'the reopening read ran during the write');
    assert.equal(els.setupsSave.getAttribute('aria-disabled'), 'true');
    assert.equal(els.setupsReload.title, 'Wait for the saved setup request.');
    await evaluate("saveSetup('new')");
    assert.equal(f.writes('POST', '/api/presets'), 1, 'a second Save current sent another POST');
    f.forceClose(els.setupsDialog);
    f.calls.requests[0].resolve({ items: [setup('s', 'Saved once')] });
    await first;
    assert.equal(evaluate('setupsPending'), null, 'a closed dialog was left validating');
    assert.equal(evaluate('setupsSelected.id'), 's');
    assert.equal(f.calls.requests.length, 1, 'a reply after close started a validation');
    els.setupsDialog.open = true;
    const reload = evaluate('reloadSetups()');
    assert.equal(f.calls.requests.at(-1).url, '/api/presets', 'the reopened dialog could not Reload');
    f.calls.requests.at(-1).resolve({ items: [setup('s', 'Saved once')] });
    await reload;
    assert.equal(evaluate('setupsPending'), null);
  } else if (scenario === 'setup-close-retires-validation') {
    els.setupsDialog.open = true;
    evaluate("setupsReady = true; setupsItems = [" + JSON.stringify(setup('s', 'Checked')) + "]");
    evaluate("selectSetup('s')");
    assert.equal(evaluate('setupsPending'), 'validate');
    assert.match(f.calls.requests[0].url, /\/validate$/);
    f.forceClose(els.setupsDialog);
    assert.equal(evaluate('setupsPending'), null, 'a closed dialog kept its validation pending');
  } else throw Error('unknown scenario: ' + scenario);
  console.log('passed: ' + scenario);
}
function document_rows(f) {
  return f.sandbox.document.getElementById('snippet-import-rows').children[0];
}
main().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("scenario", [
    "snippet-forced-close-keeps-write", "snippet-read-cleared-by-close",
    "snippet-import-forced-close-success", "snippet-import-forced-close-lost-reply",
    "snippet-import-close-retires-preview", "snippet-copy-row-keeps-file-match",
    "snippet-refresh-preview-keeps-changed-choice", "setup-forced-close-keeps-write",
    "setup-close-retires-validation",
])
def test_close_watcher_holds(scenario: str):
    node = shutil.which("node")
    assert node is not None, "Node is required for close-watcher hold checks"
    done = subprocess.run(
        [node, "-e", SCRIPT, str(ROOT), scenario], capture_output=True,
        text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"passed: {scenario}" in done.stdout
