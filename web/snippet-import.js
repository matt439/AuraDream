// Portable files are data. Keep their original JSON until the server has
// checked duplicate keys; parsing and reserializing here would erase them.
'use strict';

const SNIPPET_FILE_LIMIT = 8 * 1024 * 1024;
let snippetImportEls = null;
let snippetImportFile = null;
let snippetImportRaw = null;
let snippetImportChoices = [];
let snippetImportRows = [];
let snippetImportShown = null;
let snippetImportBusy = null;
let snippetImportTicket = 0;
let snippetImportNeedsReload = false;
let snippetImportCopyConfirm = new Set();
let snippetExportPending = false;

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
  for (const choice of snippetImportChoices) {
    if (snippetImportCopyConfirm.has(choice.index)) return `Choose the action again for file item #${choice.index + 1}; its previous Add as copy may already have been saved.`;
    if (!['add', 'copy', 'replace', 'skip'].includes(choice.action)) return `Choose an action for file item #${choice.index + 1}.`;
    const row = snippetImportRows.find((entry) => entry.index === choice.index);
    if (choice.action === 'add' && row?.matches.length) return `Snippet #${choice.index + 1} now has a saved name match. Choose Skip, Add as copy or Replace.`;
    if (choice.action === 'replace' && !row?.matches.some((item) => item.id === choice.item_id && item.revision === choice.revision)) {
      return `Choose the saved snippet to replace for file item #${choice.index + 1}.`;
    }
    if (choice.action === 'copy' && (!choice.name?.trim() || Array.from(choice.name).length > 80)) return `Give copy #${choice.index + 1} a name of 1-80 characters.`;
  }
  return '';
}

function snippetImportReason() {
  if (snippetImportBusy) return snippetImportBusy === 'commit' ? 'Wait for the import to finish.' : 'Wait for the file preview.';
  if (snippetImportRaw === null) return 'Choose a valid snippet file first.';
  return snippetImportChoiceReason() || (!snippetImportShown ? 'Preview these choices before importing.'
    : !snippetImportShown.can_import ? snippetImportShown.reason || 'The selected import changes nothing.' : '');
}

function syncSnippetPortability() {
  if (!snippetImportEls) return;
  const hold = (button, reason, title) => { button.setAttribute('aria-disabled', String(!!reason)); button.title = reason || title; };
  hold(snippetImportEls.export, snippetPortabilityReason(), 'Download saved snippets as a file containing personal prompt text');
  hold(snippetImportEls.open, snippetPortabilityReason(), 'Choose a snippet file to preview; nothing is saved yet');
  hold(snippetImportEls.run, snippetImportReason(), 'Commit exactly the previewed merge');
  hold(snippetImportEls.preview, snippetImportBusy ? 'Wait for the current import request.'
    : snippetImportRaw === null ? 'Choose a valid snippet file first.'
      : snippetImportNeedsReload ? '' : snippetImportChoiceReason(), 'Preview the chosen merge against the saved list');
  hold(snippetImportEls.close, snippetImportBusy === 'commit' ? 'Wait for the import to finish.' : '', 'Cancel without importing');
  for (const control of snippetImportEls.rows.querySelectorAll('input, select')) control.disabled = !!snippetImportBusy;
}

function invalidateSnippetImport() {
  snippetImportShown = null;
  snippetImportEls.summary.textContent = 'Choices changed. Preview again before importing.';
  snippetImportSay('');
  syncSnippetPortability();
}

function snippetImportEnvelope(extra = {}) {
  return '{"payload":' + snippetImportRaw + ',"decisions":' + JSON.stringify(snippetImportChoices)
    + Object.entries(extra).map(([key, value]) => ',' + JSON.stringify(key) + ':' + JSON.stringify(value)).join('') + '}';
}

function snippetImportNode(tag, text, className = '') {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderSnippetImport(shown, keepChoices = false) {
  snippetImportRows = shown.rows;
  if (!keepChoices) snippetImportChoices = shown.rows.map((row) => ({
    index: row.index, action: row.action,
    ...(row.action === 'copy' ? { name: row.snippet.name } : {}),
    ...(row.action === 'replace' ? { item_id: row.item_id, revision: row.revision } : {}),
  }));
  snippetImportEls.rows.replaceChildren();
  for (const row of shown.rows) {
    const choice = snippetImportChoices.find((entry) => entry.index === row.index);
    const section = snippetImportNode('section', undefined, 'snippet-import-row');
    section.dataset.index = String(row.index);
    const heading = snippetImportNode('h3', `#${row.index + 1} ${row.snippet.name} (${row.snippet.target === 'negative' ? 'Negative' : 'Prompt'})`);
    heading.id = `snippet-import-row-${row.index}`;
    section.setAttribute('aria-labelledby', heading.id);
    section.append(heading);
    if (row.exact_ids.length) section.append(snippetImportNode('p', `Exact saved match: ${row.exact_ids.map((id) => '#' + (row.matches.findIndex((item) => item.id === id) + 1)).join(', ')}`, 'hint'));
    if (row.incoming_collisions.length) section.append(snippetImportNode('p', `Same incoming name and target at ${row.incoming_collisions.map((index) => '#' + (index + 1)).join(', ')}. Each file item has its own choice.`, 'hint'));
    const text = snippetImportNode('pre', row.snippet.text);
    text.dataset.role = 'incoming-text';
    text.setAttribute('aria-label', `Incoming text for item ${row.index + 1}`);
    section.append(text);
    for (const [candidateIndex, local] of row.matches.entries()) {
      const details = document.createElement('details');
      details.open = !row.exact_ids.includes(local.id);
      details.append(snippetImportNode('summary', `Saved match #${candidateIndex + 1}: ${local.name}`));
      const localText = snippetImportNode('pre', local.text);
      localText.dataset.role = 'local-text'; localText.dataset.itemId = local.id;
      details.append(localText); section.append(details);
    }
    const controls = snippetImportNode('div', undefined, 'snippet-import-choice');
    const actionLabel = snippetImportNode('label', 'Action');
    const action = document.createElement('select'); action.dataset.role = 'action';
    if (snippetImportCopyConfirm.has(row.index)) {
      const confirm = snippetImportNode('option', 'Choose action again (previous Add as copy)');
      confirm.value = ''; confirm.disabled = true; action.append(confirm);
    }
    for (const [value, label] of [['skip', 'Skip'], ['add', row.matches.length ? 'Add (name now collides)' : 'Add'], ['copy', 'Add as copy'], ['replace', 'Replace a saved snippet']]) {
      const option = snippetImportNode('option', label); option.value = value;
      option.disabled = value === 'add' && row.matches.length > 0 || value === 'replace' && !row.matches.length;
      action.append(option);
    }
    action.value = snippetImportCopyConfirm.has(row.index) ? '' : choice.action; actionLabel.append(action); controls.append(actionLabel);
    const copyLabel = snippetImportNode('label', 'Copy name');
    const name = document.createElement('input'); name.type = 'text'; name.maxLength = 160; name.autocomplete = 'off';
    name.dataset.role = 'copy-name'; name.value = choice.name ?? row.snippet.name;
    copyLabel.hidden = choice.action !== 'copy'; copyLabel.append(name); controls.append(copyLabel);
    const targetLabel = snippetImportNode('label', 'Saved snippet to replace');
    const target = document.createElement('select'); target.dataset.role = 'replacement';
    const blank = snippetImportNode('option', 'Choose a saved snippet'); blank.value = ''; target.append(blank);
    for (const [candidateIndex, local] of row.matches.entries()) {
      const option = snippetImportNode('option', `Saved match #${candidateIndex + 1}: ${local.name}`);
      option.value = local.id; target.append(option);
    }
    target.value = row.matches.some((local) => local.id === choice.item_id && local.revision === choice.revision) ? choice.item_id : '';
    targetLabel.hidden = choice.action !== 'replace'; targetLabel.append(target); controls.append(targetLabel);
    action.onchange = () => {
      snippetImportCopyConfirm.delete(row.index);
      choice.action = action.value; delete choice.item_id; delete choice.revision; delete choice.name;
      if (choice.action === 'copy') choice.name = name.value;
      target.value = ''; copyLabel.hidden = choice.action !== 'copy'; targetLabel.hidden = choice.action !== 'replace';
      invalidateSnippetImport();
    };
    name.oninput = () => { choice.name = name.value; invalidateSnippetImport(); };
    target.onchange = () => {
      const local = row.matches.find((item) => item.id === target.value);
      delete choice.item_id; delete choice.revision;
      if (local) { choice.item_id = local.id; choice.revision = local.revision; }
      invalidateSnippetImport();
    };
    section.append(controls); snippetImportEls.rows.append(section);
  }
  if (!shown.rows.length) snippetImportEls.rows.append(snippetImportNode('p', 'This file contains no snippets. Nothing will be changed.', 'hint'));
  snippetImportEls.summary.textContent = keepChoices ? 'Saved list refreshed. Check your choices, then Preview.'
    : `${shown.add_count} added; ${shown.replace_count} replaced; ${shown.skip_count} skipped`;
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
      renderSnippetImport(defaults, true); snippetImportNeedsReload = false;
      if (snippetImportChoiceReason()) { snippetImportSay(snippetImportChoiceReason(), true); return; }
    }
    const shown = await api('/api/snippets/import/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: snippetImportEnvelope(),
    });
    if (!current()) return;
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
  snippetImportFile = file; snippetImportRaw = null; snippetImportChoices = []; snippetImportRows = [];
  snippetImportCopyConfirm.clear();
  snippetImportShown = null; snippetImportNeedsReload = false; snippetImportBusy = 'file';
  snippetImportEls.filename.textContent = file.name;
  snippetImportEls.rows.replaceChildren(); snippetImportEls.summary.textContent = 'Nothing has been imported.';
  snippetImportSay('Reading the selected file.');
  if (!snippetImportEls.dialog.open) snippetImportEls.dialog.showModal();
  syncSnippetPortability(); snippetImportEls.close.focus({ preventScroll: true });
  try {
    if (file.size > SNIPPET_FILE_LIMIT) throw new Error('The snippet file exceeds 8 MiB.');
    const raw = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer());
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
    snippetImportSay(err.status && err.status < 500 ? `Import was refused: ${err.message} Preview again before importing.`
      : 'Import result is uncertain. Preview again to reload the saved list and inspect what was saved; do not resend this import.', true);
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
    ++snippetImportTicket; snippetImportBusy = null;
    if (els.snippetsDialog.open) snippetImportEls.open.focus({ preventScroll: true });
    syncSnippetPortability();
  });
  syncSnippetPortability();
}
