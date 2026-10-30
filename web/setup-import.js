// Portable saved-setup files are data. Keep their original JSON until the
// server has checked duplicate keys; parsing and reserializing here would
// erase them. Import rows render file text through textContent only and never
// call setupSummary, which writes the editor's summary and reads the composer.
'use strict';

const SETUP_FILE_LIMIT = 8 * 1024 * 1024;
const SETUP_EXPORT_NAME = 'auradream-generation-setups.json';
let setupImportEls = null;
let setupImportRaw = null;
let setupImportChoices = [];
let setupImportRows = [];
// Per file index, the latest server row that describes the file item under its
// own name. A previewed Add as copy row is matched under its copy name, so it
// never replaces this: Add/Replace, saved matches and the heading keep the
// file item's real collision.
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
  for (const choice of setupImportChoices) {
    const item = `file item #${choice.index + 1}`;
    if (setupImportCopyConfirm.has(choice.index)) return `Choose the action again for ${item}; its previous Add as copy may already have been saved.`;
    if (!['add', 'copy', 'replace', 'skip'].includes(choice.action)) return `Choose an action for ${item}.`;
    const row = setupImportIncomingRow(choice.index);
    if (choice.action === 'add' && row?.matches.length) return `Setup #${choice.index + 1} now has a saved name match on its checkpoint. Choose Skip, Add as copy or Replace.`;
    if (choice.action === 'replace' && !row?.matches.some((local) => local.id === choice.item_id && local.revision === choice.revision)) {
      return `Choose the saved setup to replace for ${item}.`;
    }
    if (choice.action === 'copy' && (!choice.name?.trim() || Array.from(choice.name).length > 80)) return `Give copy #${choice.index + 1} a name of 1-80 characters.`;
  }
  return '';
}

function setupImportIncomingRow(index) {
  return setupImportIncoming.get(index) ?? setupImportRows.find((entry) => entry.index === index);
}

function setupImportReason() {
  if (setupImportBusy) return setupImportBusy === 'commit' ? 'Wait for the import to finish.' : 'Wait for the file preview.';
  if (setupImportRaw === null) return 'Choose a valid setup file first.';
  return setupImportChoiceReason() || (!setupImportShown ? 'Preview these choices before importing.'
    : !setupImportShown.can_import ? setupImportShown.reason || 'The selected import changes nothing.' : '');
}

function syncSetupPortability() {
  if (!setupImportEls) return;
  const hold = (button, reason, title) => { button.setAttribute('aria-disabled', String(!!reason)); button.title = reason || title; };
  hold(setupImportEls.export, setupPortabilityReason(), 'Download every checkpoint\'s saved setups; the file contains prompt text');
  hold(setupImportEls.open, setupPortabilityReason(), 'Choose a setup file to preview; nothing is saved yet');
  hold(setupImportEls.run, setupImportReason(), 'Commit exactly the previewed merge');
  hold(setupImportEls.preview, setupImportBusy ? 'Wait for the current import request.'
    : setupImportRaw === null ? 'Choose a valid setup file first.'
      : setupImportNeedsReload ? '' : setupImportChoiceReason(), 'Preview the chosen merge against the saved list');
  hold(setupImportEls.close, setupImportBusy === 'commit' ? 'Wait for the import to finish.' : '', 'Cancel without importing');
  for (const control of setupImportEls.rows.querySelectorAll('input, select')) control.disabled = !!setupImportBusy;
}

function invalidateSetupImport() {
  setupImportShown = null;
  setupImportEls.summary.textContent = 'Choices changed. Preview again before importing.';
  setupImportSay('');
  syncSetupPortability();
}

function setupImportEnvelope(extra = {}) {
  return '{"payload":' + setupImportRaw + ',"decisions":' + JSON.stringify(setupImportChoices)
    + Object.entries(extra).map(([key, value]) => ',' + JSON.stringify(key) + ':' + JSON.stringify(value)).join('') + '}';
}

function setupImportNode(tag, text, className = '') {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
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

function renderSetupImport(shown, keepChoices = false) {
  setupImportRows = shown.rows;
  for (const row of shown.rows) if (row.action !== 'copy' || !setupImportIncoming.has(row.index)) setupImportIncoming.set(row.index, row);
  if (!keepChoices) setupImportChoices = shown.rows.map((row) => ({
    index: row.index, action: row.action,
    ...(row.action === 'copy' ? { name: row.preset.name } : {}),
    ...(row.action === 'replace' ? { item_id: row.item_id, revision: row.revision } : {}),
  }));
  setupImportEls.rows.replaceChildren();
  for (const shownRow of shown.rows) {
    // Same index, payload and checkpoint; only the copy name may differ.
    const row = { ...setupImportIncomingRow(shownRow.index), incoming_collisions: shownRow.incoming_collisions };
    const choice = setupImportChoices.find((entry) => entry.index === row.index);
    const section = setupImportNode('section', undefined, 'setup-import-row');
    section.dataset.index = String(row.index);
    const heading = setupImportNode('h3', `#${row.index + 1} ${row.preset.name}`);
    heading.id = `setup-import-row-${row.index}`;
    section.setAttribute('aria-labelledby', heading.id);
    section.append(heading);
    const meta = setupImportNode('p', `${setupImportKind(row.preset.payload)} · ${setupImportCheckpoint(row.preset.model_id)}`, 'setup-import-meta');
    meta.dataset.role = 'meta';
    section.append(meta);
    if (row.exact_ids.length) section.append(setupImportNode('p', `Exact saved match: ${row.exact_ids.map((id) => '#' + (row.matches.findIndex((local) => local.id === id) + 1)).join(', ')}`, 'hint'));
    if (row.incoming_collisions.length) section.append(setupImportNode('p', `Same incoming name and checkpoint at ${row.incoming_collisions.map((index) => '#' + (index + 1)).join(', ')}. Each file item has its own choice.`, 'hint'));
    const payload = setupImportNode('pre', setupImportEssentials(row.preset.payload));
    payload.dataset.role = 'incoming-payload';
    payload.setAttribute('aria-label', `Incoming setup ${row.index + 1}`);
    section.append(payload);
    for (const [candidateIndex, local] of row.matches.entries()) {
      const details = document.createElement('details');
      details.open = !row.exact_ids.includes(local.id);
      details.append(setupImportNode('summary', `Saved match #${candidateIndex + 1}: ${local.name} · ${setupImportKind(local.payload)} · revision ${local.revision}`));
      const localPayload = setupImportNode('pre', setupImportEssentials(local.payload));
      localPayload.dataset.role = 'local-payload'; localPayload.dataset.itemId = local.id;
      details.append(localPayload); section.append(details);
    }
    const controls = setupImportNode('div', undefined, 'setup-import-choice');
    const actionLabel = setupImportNode('label', 'Action');
    const action = document.createElement('select'); action.dataset.role = 'action';
    if (setupImportCopyConfirm.has(row.index)) {
      const confirm = setupImportNode('option', 'Choose action again (previous Add as copy)');
      confirm.value = ''; confirm.disabled = true; action.append(confirm);
    }
    for (const [value, label] of [['skip', 'Skip'], ['add', row.matches.length ? 'Add (name now collides)' : 'Add'],
      ['copy', 'Add as copy'], ['replace', 'Replace a saved setup']]) {
      const option = setupImportNode('option', label); option.value = value;
      option.disabled = value === 'add' && row.matches.length > 0 || value === 'replace' && !row.matches.length;
      action.append(option);
    }
    action.value = setupImportCopyConfirm.has(row.index) ? '' : choice.action;
    actionLabel.append(action); controls.append(actionLabel);
    const copyLabel = setupImportNode('label', 'Copy name');
    const name = document.createElement('input'); name.type = 'text'; name.maxLength = 160; name.autocomplete = 'off';
    name.dataset.role = 'copy-name'; name.value = choice.name ?? row.preset.name;
    copyLabel.hidden = choice.action !== 'copy'; copyLabel.append(name); controls.append(copyLabel);
    // Matches share the incoming exact checkpoint id by construction.
    const targetLabel = setupImportNode('label', 'Saved setup to replace (same checkpoint)');
    const target = document.createElement('select'); target.dataset.role = 'replacement';
    const blank = setupImportNode('option', 'Choose a saved setup'); blank.value = ''; target.append(blank);
    for (const [candidateIndex, local] of row.matches.entries()) {
      const option = setupImportNode('option', `Saved match #${candidateIndex + 1}: ${local.name} (revision ${local.revision})`);
      option.value = local.id; target.append(option);
    }
    target.value = row.matches.some((local) => local.id === choice.item_id && local.revision === choice.revision) ? choice.item_id : '';
    targetLabel.hidden = choice.action !== 'replace'; targetLabel.append(target); controls.append(targetLabel);
    action.onchange = () => {
      setupImportCopyConfirm.delete(row.index);
      choice.action = action.value; delete choice.item_id; delete choice.revision; delete choice.name;
      if (choice.action === 'copy') choice.name = name.value;
      target.value = ''; copyLabel.hidden = choice.action !== 'copy'; targetLabel.hidden = choice.action !== 'replace';
      invalidateSetupImport();
    };
    name.oninput = () => { choice.name = name.value; invalidateSetupImport(); };
    target.onchange = () => {
      const local = row.matches.find((entry) => entry.id === target.value);
      delete choice.item_id; delete choice.revision;
      if (local) { choice.item_id = local.id; choice.revision = local.revision; }
      invalidateSetupImport();
    };
    section.append(controls); setupImportEls.rows.append(section);
  }
  if (!shown.rows.length) setupImportEls.rows.append(setupImportNode('p', 'This file contains no setups. Nothing will be changed.', 'hint'));
  setupImportEls.summary.textContent = keepChoices ? 'Saved list refreshed. Check your choices, then Preview.'
    : `${shown.add_count} added; ${shown.replace_count} replaced; ${shown.skip_count} skipped`;
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
    if (file.size > SETUP_FILE_LIMIT) throw new Error('The setup file exceeds 8 MiB.');
    const raw = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer());
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
