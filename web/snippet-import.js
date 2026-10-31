// Portable files are data. Keep their original JSON until the server has
// checked duplicate keys; parsing and reserializing here would erase them.
// The merge rows, reasons and holds are import-merge.js's; this file keeps the
// dialog's state, preview, commit and uncertain-import flow.
'use strict';

const SNIPPET_FILE_LIMIT = 8 * 1024 * 1024;
let snippetImportEls = null;
let snippetImportFile = null;
let snippetImportRaw = null;
let snippetImportChoices = [];
let snippetImportRows = [];
// Per file index, the latest server row that describes the file item under its
// own name (import-merge.js's noteImportIncoming).
let snippetImportIncoming = new Map();
let snippetImportShown = null;
let snippetImportBusy = null;
let snippetImportTicket = 0;
let snippetImportNeedsReload = false;
let snippetImportCopyConfirm = new Set();
let snippetExportPending = false;

// How the merge rows show a snippet, and the words that name one.
const SNIPPET_IMPORT_KIND = {
  prefix: 'snippet', noun: 'snippet', sameAs: 'target', targetLabel: 'Saved snippet to replace',
  item: (row) => row.snippet,
  collides: (n) => `Snippet #${n} now has a saved name match. Choose Skip, Add as copy or Replace.`,
  heading: (row) => `#${row.index + 1} ${row.snippet.name} (${row.snippet.target === 'negative' ? 'Negative' : 'Prompt'})`,
  incoming: (row) => ({ text: row.snippet.text, role: 'incoming-text', label: `Incoming text for item ${row.index + 1}` }),
  matchSummary: (local, i) => `Saved match #${i + 1}: ${local.name}`,
  local: (local) => ({ text: local.text, role: 'local-text' }),
  matchOption: (local, i) => `Saved match #${i + 1}: ${local.name}`,
};

function snippetImportSay(text, error = false) {
  snippetImportEls.status.textContent = text;
  snippetImportEls.status.classList.toggle('error', error);
}

function snippetSelectedReason() {
  if (!snippetsSelected) return '';
  const saved = snippetsItems.find((item) => item.id === snippetsSelected.id);
  if (!saved) return 'That saved snippet was removed. Choose New or select another saved snippet before saving or deleting.';
  return saved.revision !== snippetsSelected.revision
    ? 'The selected snippet changed. Select it again to read its latest version before saving or deleting.' : '';
}

function snippetPortabilityReason() {
  if (snippetsPending) return 'Wait for the current snippets request.';
  if (snippetExportPending) return 'Wait for the snippet export.';
  return '';
}

function snippetImportChoiceReason() {
  return importChoiceReason(SNIPPET_IMPORT_KIND, snippetImportChoices, snippetImportCopyConfirm, snippetImportIncomingRow);
}

function snippetImportIncomingRow(index) {
  return snippetImportIncoming.get(index) ?? snippetImportRows.find((entry) => entry.index === index);
}

function snippetImportReason() {
  return importReason(SNIPPET_IMPORT_KIND, snippetImportBusy, snippetImportRaw, snippetImportShown, snippetImportChoiceReason);
}

function syncSnippetPortability() {
  if (!snippetImportEls) return;
  importHold(snippetImportEls.export, snippetPortabilityReason(), 'Download saved snippets as a file containing personal prompt text');
  importHold(snippetImportEls.open, snippetPortabilityReason(), 'Choose a snippet file to preview; nothing is saved yet');
  syncImportHolds(SNIPPET_IMPORT_KIND, snippetImportEls, {
    busy: snippetImportBusy, raw: snippetImportRaw, needsReload: snippetImportNeedsReload,
    reason: snippetImportReason(), choiceReason: snippetImportChoiceReason,
  });
}

function invalidateSnippetImport() {
  snippetImportShown = null;
  snippetImportEls.summary.textContent = 'Choices changed. Preview again before importing.';
  snippetImportSay('');
  syncSnippetPortability();
}

function snippetImportEnvelope(extra = {}) {
  return importEnvelope(snippetImportRaw, snippetImportChoices, extra);
}

function renderSnippetImport(shown, keepChoices = false) {
  snippetImportRows = shown.rows;
  noteImportIncoming(snippetImportIncoming, shown.rows);
  if (!keepChoices) snippetImportChoices = importChoices(SNIPPET_IMPORT_KIND, shown.rows);
  renderImportRows(SNIPPET_IMPORT_KIND, shown, keepChoices, {
    els: snippetImportEls, choices: snippetImportChoices, copyConfirm: snippetImportCopyConfirm,
    incomingRow: snippetImportIncomingRow, invalidate: invalidateSnippetImport,
  });
}

async function previewSnippetImport() {
  if (snippetImportBusy || snippetImportRaw === null) { snippetImportSay(snippetImportEls.preview.title, true); return; }
  if (!snippetImportNeedsReload && snippetImportChoiceReason()) { snippetImportSay(snippetImportChoiceReason(), true); return; }
  const ticket = ++snippetImportTicket;
  snippetImportBusy = 'preview'; snippetImportShown = null;
  snippetImportSay('Reading the saved list and previewing the merge.'); syncSnippetPortability();
  const current = () => ticket === snippetImportTicket && snippetImportEls.dialog.open;
  try {
    if (snippetImportNeedsReload) {
      const saved = await api('/api/snippets');
      if (!current()) return;
      snippetsItems = saved.items; snippetsReady = true; renderSnippets();
      const defaults = await api('/api/snippets/import/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: '{"payload":' + snippetImportRaw + ',"decisions":[]}',
      });
      if (!current()) return;
      // The redrawn rows start enabled: hold them again until the reply.
      renderSnippetImport(defaults, true); snippetImportNeedsReload = false; syncSnippetPortability();
      if (snippetImportChoiceReason()) { snippetImportSay(snippetImportChoiceReason(), true); return; }
    }
    const body = snippetImportEnvelope();
    const shown = await api('/api/snippets/import/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
    });
    if (!current()) return;
    // Never let a reply overwrite choices made after its request was sent.
    if (snippetImportEnvelope() !== body) { snippetImportSay('Choices changed. Preview again before importing.', true); return; }
    snippetImportShown = shown; renderSnippetImport(shown);
    snippetImportSay(shown.can_import ? 'Review every choice. Import saves this merge; your prompt boxes stay unchanged.' : shown.reason);
  } catch (err) {
    if (!current()) return;
    if (err.status === 409) snippetImportNeedsReload = true;
    snippetImportSay(`Preview could not be made: ${err.message}`, true);
  } finally {
    if (current()) { snippetImportBusy = null; syncSnippetPortability(); }
  }
}

async function openSnippetImport(file) {
  if (snippetImportBusy === 'commit' || !file || !els.snippetsDialog.open) return;
  const ticket = ++snippetImportTicket;
  snippetImportFile = file; snippetImportRaw = null; snippetImportChoices = []; snippetImportRows = []; snippetImportIncoming = new Map();
  snippetImportCopyConfirm.clear();
  snippetImportShown = null; snippetImportNeedsReload = false; snippetImportBusy = 'file';
  snippetImportEls.filename.textContent = file.name;
  snippetImportEls.rows.replaceChildren(); snippetImportEls.summary.textContent = 'Nothing has been imported.';
  snippetImportSay('Reading the selected file.');
  if (!snippetImportEls.dialog.open) snippetImportEls.dialog.showModal();
  syncSnippetPortability(); snippetImportEls.close.focus({ preventScroll: true });
  try {
    const raw = await readImportFile(file, SNIPPET_FILE_LIMIT, 'snippet');
    if (ticket !== snippetImportTicket || !snippetImportEls.dialog.open) return;
    snippetImportRaw = raw; snippetImportBusy = null;
    await previewSnippetImport();
  } catch (err) {
    if (ticket !== snippetImportTicket || !snippetImportEls.dialog.open) return;
    snippetImportBusy = null; snippetImportSay(`File could not be read: ${err.message}`, true); syncSnippetPortability();
  }
}

function chooseSnippetImport() {
  const reason = snippetPortabilityReason();
  if (reason) { snippetSay(reason, true); return; }
  snippetImportEls.file.value = ''; snippetImportEls.file.click();
}

function closeSnippetImport() {
  if (snippetImportBusy === 'commit') { snippetImportSay('Wait for the import to finish.'); return; }
  ++snippetImportTicket; snippetImportBusy = null; snippetImportEls.dialog.close();
}

async function commitSnippetImport() {
  const reason = snippetImportReason();
  if (reason) { snippetImportSay(reason, true); return; }
  const shown = snippetImportShown;
  const ticket = ++snippetImportTicket;
  snippetImportBusy = 'commit'; snippetsPending = 'write';
  snippetImportSay('Importing the previewed merge.'); syncSnippets(); syncSnippetPortability();
  try {
    const result = await api('/api/snippets/import', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: snippetImportEnvelope({ payload_digest: shown.payload_digest, collection_revision: shown.collection_revision }),
    });
    if (ticket !== snippetImportTicket) return;
    snippetsItems = result.items; snippetsReady = true; snippetsPending = null;
    renderSnippets(); snippetImportBusy = null; snippetImportShown = null;
    snippetSay(`Imported ${shown.add_count} added, ${shown.replace_count} replaced; ${shown.skip_count} skipped. Your editor is unchanged.`);
    closeSnippetImport();
  } catch (err) {
    if (ticket !== snippetImportTicket) return;
    snippetImportShown = null; snippetImportNeedsReload = true; snippetsReady = false;
    if (!err.status || err.status >= 500) {
      for (const choice of snippetImportChoices) if (choice.action === 'copy') snippetImportCopyConfirm.add(choice.index);
    }
    const refused = err.status && err.status < 500;
    snippetImportSay(refused ? `Import was refused: ${err.message} Preview again before importing.`
      : 'Import result is uncertain. Preview again to reload the saved list and inspect what was saved; do not resend this import.', true);
    // A forced close leaves the commit running; its answer goes to Snippets.
    if (!snippetImportEls.dialog.open) {
      snippetSay(refused ? `Import was refused: ${err.message} Nothing was imported.`
        : 'Import result is uncertain. Reload the list to inspect what was saved; do not resend that import.', true);
    }
  } finally {
    if (ticket === snippetImportTicket) { snippetImportBusy = null; snippetsPending = null; syncSnippets(); syncSnippetPortability(); }
  }
}

async function exportSnippets() {
  const reason = snippetPortabilityReason();
  if (reason) { snippetSay(reason, true); return; }
  const ticket = snippetsDialogEpoch;
  snippetExportPending = true; syncSnippetPortability();
  try {
    const response = await fetch('/api/snippets/export', { cache: 'no-store' });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail || detail; } catch { /* fixed HTTP fallback */ }
      throw new Error(readableDetail(detail));
    }
    const blob = await response.blob();
    if (ticket !== snippetsDialogEpoch || !els.snippetsDialog.open) return;
    if (blob.size > SNIPPET_FILE_LIMIT) throw new Error('The exported file exceeds 8 MiB.');
    const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'auradream-snippets.json';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
    snippetSay('Saved snippets downloaded. The file contains personal prompt text; your editor is unchanged.');
  } catch (err) {
    if (ticket === snippetsDialogEpoch && els.snippetsDialog.open) snippetSay(`Snippets could not be exported: ${err.message}`, true);
  } finally { snippetExportPending = false; syncSnippetPortability(); }
}

function initSnippetImport() {
  const at = (id) => document.getElementById(id);
  snippetImportEls = {
    export: at('snippets-export'), open: at('snippets-import'), file: at('snippets-import-file'),
    dialog: at('snippet-import-dialog'), filename: at('snippet-import-filename'), rows: at('snippet-import-rows'),
    summary: at('snippet-import-summary'), status: at('snippet-import-status'),
    preview: at('snippet-import-preview'), run: at('snippet-import-run'), close: at('snippet-import-close'),
  };
  snippetImportEls.export.onclick = () => { void exportSnippets(); };
  snippetImportEls.open.onclick = chooseSnippetImport;
  snippetImportEls.file.onchange = () => { const file = snippetImportEls.file.files?.[0]; if (file) void openSnippetImport(file); };
  snippetImportEls.preview.onclick = () => { void previewSnippetImport(); };
  snippetImportEls.run.onclick = () => { void commitSnippetImport(); };
  snippetImportEls.close.onclick = closeSnippetImport;
  snippetImportEls.dialog.addEventListener('cancel', (event) => {
    if (snippetImportBusy === 'commit') { event.preventDefault(); snippetImportSay('Wait for the import to finish.'); }
  });
  snippetImportEls.dialog.addEventListener('close', () => {
    if (snippetImportEls.dialog.open) return;
    // A repeated Escape may close the dialog despite the held cancel (Chrome's
    // close-watcher rule). A pending commit keeps its ticket: its reply still
    // updates the list, clears the parent's write hold or records uncertainty.
    if (snippetImportBusy !== 'commit') { ++snippetImportTicket; snippetImportBusy = null; }
    if (els.snippetsDialog.open) snippetImportEls.open.focus({ preventScroll: true });
    syncSnippetPortability();
  });
  syncSnippetPortability();
}
