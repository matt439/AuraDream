// Portable saved-setup files are data. Keep their original JSON until the
// server has checked duplicate keys; parsing and reserializing here would
// erase them. Import rows render file text through textContent only and never
// call setupSummary, which writes the editor's summary and reads the composer.
// The merge rows, reasons and holds are import-merge.js's; this file keeps the
// dialog's state, preview, commit and uncertain-import flow.
'use strict';

const SETUP_FILE_LIMIT = 8 * 1024 * 1024;
const SETUP_EXPORT_NAME = 'auradream-generation-setups.json';
let setupImportEls = null;
let setupImportRaw = null;
let setupImportChoices = [];
let setupImportRows = [];
// Per file index, the latest server row that describes the file item under its
// own name (import-merge.js's noteImportIncoming).
let setupImportIncoming = new Map();
let setupImportShown = null;
let setupImportBusy = null;
let setupImportTicket = 0;
let setupImportNeedsReload = false;
let setupImportCopyConfirm = new Set();
let setupExportPending = false;
// One Saved setups dialog session; Reload's list ticket never retires it.
let setupPortabilitySession = 0;

function setupImportSay(text, error = false) {
  setupImportEls.status.textContent = text;
  setupImportEls.status.classList.toggle('error', error);
}

// A selection whose saved revision was replaced or removed holds every action
// that uses it until the user selects it again or chooses New.
function setupSelectedReason() {
  if (!setupsSelected || !setupsReady) return '';
  const saved = setupsItems.find((item) => item.id === setupsSelected.id);
  if (!saved) return 'That saved setup was removed. Choose New or select another saved setup before changing it.';
  return saved.revision !== setupsSelected.revision
    ? 'The selected setup changed. Select it again to read its latest version before changing or restoring it.' : '';
}

function setupPortabilityReason() {
  if (setupsPending) return 'Wait for the saved setup request.';
  if (setupExportPending) return 'Wait for the setup export.';
  return '';
}

function setupImportChoiceReason() {
  return importChoiceReason(SETUP_IMPORT_KIND, setupImportChoices, setupImportCopyConfirm, setupImportIncomingRow);
}

function setupImportIncomingRow(index) {
  return setupImportIncoming.get(index) ?? setupImportRows.find((entry) => entry.index === index);
}

function setupImportReason() {
  return importReason(SETUP_IMPORT_KIND, setupImportBusy, setupImportRaw, setupImportShown, setupImportChoiceReason);
}

function syncSetupPortability() {
  if (!setupImportEls) return;
  importHold(setupImportEls.export, setupPortabilityReason(), 'Download every checkpoint\'s saved setups; the file contains prompt text');
  importHold(setupImportEls.open, setupPortabilityReason(), 'Choose a setup file to preview; nothing is saved yet');
  syncImportHolds(SETUP_IMPORT_KIND, setupImportEls, {
    busy: setupImportBusy, raw: setupImportRaw, needsReload: setupImportNeedsReload,
    reason: setupImportReason(), choiceReason: setupImportChoiceReason,
  });
}

function invalidateSetupImport() {
  setupImportShown = null;
  setupImportEls.summary.textContent = 'Choices changed. Preview again before importing.';
  setupImportSay('');
  syncSetupPortability();
}

function setupImportEnvelope(extra = {}) {
  return importEnvelope(setupImportRaw, setupImportChoices, extra);
}

function setupImportKind(payload) {
  return payload.mode === 'grid' ? 'Grid' : 'Generate';
}

// The raw saved draft only: no profile, effective value or composer reads.
function setupImportEssentials(payload) {
  const saved = (value) => value ?? 'profile default';
  const lines = [
    `${payload.width} × ${payload.height} · Seed: ${payload.seed ?? 'random at submission'} · Batch: ${payload.num_images} · Repeat: ${payload.runs}`,
    `Steps: ${saved(payload.steps)} · CFG: ${saved(payload.guidance_scale)} · Mu: ${saved(payload.mu)} · True CFG: ${saved(payload.true_cfg)}`,
    `Prompt:\n${payload.prompt}`,
    `Negative:\n${payload.negative_prompt || '(empty)'}`,
  ];
  const grid = payload.grid;
  if (grid) {
    lines.push(`Variables (${grid.prompt_mode}):\n${grid.prompt_axis || '(none)'}`,
      `Axes: Steps ${grid.steps_axis || '(base)'}; CFG ${grid.cfg_axis || '(base)'}; Mu ${grid.mu_axis || '(base)'}; Size ${grid.size_axis || '(base)'}`,
      `Seeds: ${grid.seed_axis || grid.seeds} · Budget: ${grid.budget_minutes ?? 'none'} minutes`);
  }
  return lines.join('\n');
}

function setupImportCheckpoint(modelId) {
  const name = checkpointName(modelId);
  const where = modelId === loadedModel ? 'selected checkpoint' : 'another checkpoint';
  return `${name === modelId ? modelId : `${name} (${modelId})`} · ${where}`;
}

// How the merge rows show a setup, and the words that name one: the raw
// saved draft only (setupImportEssentials), never setupSummary.
const SETUP_IMPORT_KIND = {
  prefix: 'setup', noun: 'setup', sameAs: 'checkpoint',
  // Matches share the incoming exact checkpoint id by construction.
  targetLabel: 'Saved setup to replace (same checkpoint)',
  item: (row) => row.preset,
  collides: (n) => `Setup #${n} now has a saved name match on its checkpoint. Choose Skip, Add as copy or Replace.`,
  heading: (row) => `#${row.index + 1} ${row.preset.name}`,
  meta: (row) => `${setupImportKind(row.preset.payload)} · ${setupImportCheckpoint(row.preset.model_id)}`,
  incoming: (row) => ({ text: setupImportEssentials(row.preset.payload), role: 'incoming-payload', label: `Incoming setup ${row.index + 1}` }),
  matchSummary: (local, i) => `Saved match #${i + 1}: ${local.name} · ${setupImportKind(local.payload)} · revision ${local.revision}`,
  local: (local) => ({ text: setupImportEssentials(local.payload), role: 'local-payload' }),
  matchOption: (local, i) => `Saved match #${i + 1}: ${local.name} (revision ${local.revision})`,
};

function renderSetupImport(shown, keepChoices = false) {
  setupImportRows = shown.rows;
  noteImportIncoming(setupImportIncoming, shown.rows);
  if (!keepChoices) setupImportChoices = importChoices(SETUP_IMPORT_KIND, shown.rows);
  renderImportRows(SETUP_IMPORT_KIND, shown, keepChoices, {
    els: setupImportEls, choices: setupImportChoices, copyConfirm: setupImportCopyConfirm,
    incomingRow: setupImportIncomingRow, invalidate: invalidateSetupImport,
  });
}

async function previewSetupImport() {
  if (setupImportBusy || setupImportRaw === null) { setupImportSay(setupImportEls.preview.title, true); return; }
  if (!setupImportNeedsReload && setupImportChoiceReason()) { setupImportSay(setupImportChoiceReason(), true); return; }
  const ticket = ++setupImportTicket;
  setupImportBusy = 'preview'; setupImportShown = null;
  setupImportSay('Reading the saved list and previewing the merge.'); syncSetupPortability();
  const current = () => ticket === setupImportTicket && setupImportEls.dialog.open;
  try {
    if (setupImportNeedsReload) {
      // A fresh read lets the user inspect what is saved; it never
      // acknowledges an uncertain write or replaces the editor.
      const saved = await api('/api/presets');
      if (!current()) return;
      setupsItems = saved.items; setupsReady = true;
      if (setupsWriteRecovery) setupsWriteRecovery.needsReload = false;
      renderSetups();
      const defaults = await api('/api/presets/import/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: '{"payload":' + setupImportRaw + ',"decisions":[]}',
      });
      if (!current()) return;
      // The redrawn rows start enabled: hold them again until the reply.
      renderSetupImport(defaults, true); setupImportNeedsReload = false; syncSetupPortability();
      if (setupImportChoiceReason()) { setupImportSay(setupImportChoiceReason(), true); return; }
    }
    const body = setupImportEnvelope();
    const shown = await api('/api/presets/import/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
    });
    if (!current()) return;
    // Never let a reply overwrite choices made after its request was sent.
    if (setupImportEnvelope() !== body) { setupImportSay('Choices changed. Preview again before importing.', true); return; }
    setupImportShown = shown; renderSetupImport(shown);
    setupImportSay(shown.can_import ? 'Review every choice. Import saves this merge; your composer and setup name stay unchanged.' : shown.reason);
  } catch (err) {
    if (!current()) return;
    if (err.status === 409) setupImportNeedsReload = true;
    setupImportSay(`Preview could not be made: ${err.message}`, true);
  } finally {
    if (current()) { setupImportBusy = null; syncSetupPortability(); }
  }
}

async function openSetupImport(file) {
  if (setupImportBusy === 'commit' || !file || !els.setupsDialog.open) return;
  const ticket = ++setupImportTicket;
  setupImportRaw = null; setupImportChoices = []; setupImportRows = []; setupImportIncoming = new Map();
  setupImportCopyConfirm.clear();
  setupImportShown = null; setupImportNeedsReload = false; setupImportBusy = 'file';
  setupImportEls.filename.textContent = file.name;
  setupImportEls.rows.replaceChildren(); setupImportEls.summary.textContent = 'Nothing has been imported.';
  setupImportSay('Reading the selected file.');
  if (!setupImportEls.dialog.open) setupImportEls.dialog.showModal();
  syncSetupPortability(); setupImportEls.close.focus({ preventScroll: true });
  try {
    const raw = await readImportFile(file, SETUP_FILE_LIMIT, 'setup');
    if (ticket !== setupImportTicket || !setupImportEls.dialog.open) return;
    setupImportRaw = raw; setupImportBusy = null;
    await previewSetupImport();
  } catch (err) {
    if (ticket !== setupImportTicket || !setupImportEls.dialog.open) return;
    setupImportBusy = null; setupImportSay(`File could not be read: ${err.message}`, true); syncSetupPortability();
  }
}

function chooseSetupImport() {
  const reason = setupPortabilityReason();
  if (reason) { setupSay(reason, true); return; }
  setupImportEls.file.value = ''; setupImportEls.file.click();
}

function closeSetupImport() {
  if (setupImportBusy === 'commit') { setupImportSay('Wait for the import to finish.'); return; }
  ++setupImportTicket; setupImportBusy = null; setupImportEls.dialog.close();
}

async function commitSetupImport() {
  const reason = setupImportReason();
  if (reason) { setupImportSay(reason, true); return; }
  const shown = setupImportShown;
  const ticket = ++setupImportTicket;
  // Retire any late list/validation reply; the commit answers the whole list.
  ++setupsTicket;
  setupImportBusy = 'commit'; setupsPending = 'write';
  setupImportSay('Importing the previewed merge.'); syncSetups(); syncSetupPortability();
  try {
    const result = await api('/api/presets/import', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: setupImportEnvelope({ payload_digest: shown.payload_digest, collection_revision: shown.collection_revision }),
    });
    if (!Array.isArray(result?.items)) throw new Error('The import reply could not be read.');
    if (ticket !== setupImportTicket) return;
    // The list only: the editor name, its selected revision, the dirty draft,
    // the composer and its Undo stay. An earlier uncertain write stays held.
    setupsItems = result.items; setupsReady = true; setupsPending = null;
    if (setupsWriteRecovery) setupsWriteRecovery.needsReload = false;
    renderSetups(); setupImportBusy = null; setupImportShown = null;
    const held = setupSelectedReason();
    setupSay(`Imported ${shown.add_count} added, ${shown.replace_count} replaced; ${shown.skip_count} skipped. Your name and composer are unchanged.${held ? ' ' + held : ''}`);
    closeSetupImport();
  } catch (err) {
    if (ticket !== setupImportTicket) return;
    setupImportShown = null;
    if (!err.status || err.status >= 500) {
      // The server may have saved it. Never resend: a fresh list and preview
      // come first, replacements stay for review and copies need reconfirming.
      setupImportNeedsReload = true; setupsReady = false;
      setupsWriteRecovery = { needsReload: true };
      for (const choice of setupImportChoices) if (choice.action === 'copy') setupImportCopyConfirm.add(choice.index);
      setupImportSay('Import result is uncertain. Preview again to reload the saved list and inspect what was saved; do not resend this import.', true);
      if (!setupImportEls.dialog.open) setupSay('Import result is uncertain. Reload the list to inspect what was saved; do not resend that import.', true);
    } else if (err.status === 409) {
      setupImportNeedsReload = true;
      setupImportSay(`Import was refused: ${err.message} Preview again to refresh the choices.`, true);
    } else {
      setupImportSay(`Import was refused: ${err.message} Correct the choices, then Preview again.`, true);
    }
    if (setupImportEls.dialog.open || !err.status || err.status >= 500) return;
    setupSay(`Import was refused: ${err.message} Nothing was imported.`, true);
  } finally {
    if (ticket === setupImportTicket) { setupImportBusy = null; setupsPending = null; syncSetups(); syncSetupPortability(); }
  }
}

async function exportSetups() {
  const reason = setupPortabilityReason();
  if (reason) { setupSay(reason, true); return; }
  const session = setupPortabilitySession;
  const live = () => session === setupPortabilitySession && els.setupsDialog.open;
  setupExportPending = true; syncSetupPortability();
  try {
    const response = await fetch('/api/presets/export', { cache: 'no-store' });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail || detail; } catch { /* fixed HTTP fallback */ }
      throw new Error(readableDetail(detail));
    }
    const blob = await response.blob();
    if (!live()) return;
    if (blob.size > SETUP_FILE_LIMIT) throw new Error('The exported file exceeds 8 MiB.');
    const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = SETUP_EXPORT_NAME;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
    setupSay('Saved setups for every checkpoint downloaded. The file contains prompt text; your name and composer are unchanged.');
  } catch (err) {
    if (live()) setupSay(`Setups could not be exported: ${err.message}`, true);
  } finally {
    if (session === setupPortabilitySession) setupExportPending = false;
    syncSetupPortability();
  }
}

function initSetupImport() {
  const at = (id) => document.getElementById(id);
  setupImportEls = {
    export: at('setups-export'), open: at('setups-import'), file: at('setups-import-file'),
    dialog: at('setup-import-dialog'), filename: at('setup-import-filename'), rows: at('setup-import-rows'),
    summary: at('setup-import-summary'), status: at('setup-import-status'),
    preview: at('setup-import-preview'), run: at('setup-import-run'), close: at('setup-import-close'),
  };
  setupImportEls.export.onclick = () => { void exportSetups(); };
  setupImportEls.open.onclick = chooseSetupImport;
  setupImportEls.file.onchange = () => { const file = setupImportEls.file.files?.[0]; if (file) void openSetupImport(file); };
  setupImportEls.preview.onclick = () => { void previewSetupImport(); };
  setupImportEls.run.onclick = () => { void commitSetupImport(); };
  setupImportEls.close.onclick = closeSetupImport;
  setupImportEls.dialog.addEventListener('cancel', (event) => {
    if (setupImportBusy === 'commit') { event.preventDefault(); setupImportSay('Wait for the import to finish.'); }
  });
  setupImportEls.dialog.addEventListener('close', () => {
    if (setupImportEls.dialog.open) return;
    // A repeated Escape may close the dialog despite the held cancel (Chrome's
    // close-watcher rule). A pending commit keeps its ticket: its reply still
    // updates the list, clears the parent's write hold or records uncertainty.
    if (setupImportBusy !== 'commit') { ++setupImportTicket; setupImportBusy = null; }
    if (els.setupsDialog.open) setupImportEls.open.focus({ preventScroll: true });
    syncSetupPortability();
  });
  els.setupsDialog.addEventListener('close', () => {
    if (els.setupsDialog.open) return;
    // A late export reply from this session never downloads after Close.
    ++setupPortabilitySession; setupExportPending = false;
    if (setupImportEls.dialog.open) closeSetupImport();
    syncSetupPortability();
  });
  syncSetupPortability();
}
