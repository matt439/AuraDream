// Personal snippets stay on the server. Only app.js calls initSnippets;
// opening the dialog is the first read, so startup needs no extra request.
'use strict';

let snippetsItems = [];
let snippetsSelected = null;
let snippetsReady = false;
let snippetsPending = null;
let snippetsReadTicket = 0;
let snippetsDialogEpoch = 0;
let snippetsOrigin = null;
let snippetsDirty = false;
let snippetsEditEpoch = 0;
let snippetsWriteRecovery = null;

function snippetSay(text, error = false) {
  els.snippetsStatus.textContent = text;
  els.snippetsStatus.classList.toggle('error', error);
}

function snippetFields() {
  return {
    name: els.snippetsName.value.trim(), text: els.snippetsText.value,
    target: els.snippetsTarget.value,
  };
}

function snippetWriteRecoveryReason() {
  if (!snippetsWriteRecovery) return '';
  return snippetsWriteRecovery.needsReload
    ? 'The last snippet write is uncertain. Reload the saved list before choosing how to continue.'
    : 'Saved list reloaded. Select a saved snippet or choose New before saving or deleting.';
}

function snippetDeleteReason() {
  if (snippetsPending) return 'Wait for the current snippets request.';
  return snippetWriteRecoveryReason() || (!snippetsReady ? 'Reload the saved list before changing it.'
    : !snippetsSelected ? 'Select a saved snippet to delete.' : snippetSelectedReason());
}

function snippetSaveReason() {
  if (snippetsPending) return snippetsPending === 'read' ? 'Wait for the saved list to arrive.' : 'Wait for the snippet to finish saving.';
  if (snippetWriteRecoveryReason()) return snippetWriteRecoveryReason();
  if (!snippetsReady) return 'Reload the saved list before saving.';
  if (snippetSelectedReason()) return snippetSelectedReason();
  const { name, text } = snippetFields();
  if (!name) return 'Give the snippet a name.';
  if (!text.trim()) return 'Give the snippet some text.';
  if (!els.snippetsName.validity.valid || !els.snippetsText.validity.valid) return 'The name or text is longer than its limit.';
  return '';
}

function snippetInsertReason() {
  if (snippetsPending === 'write') return 'Wait for the snippet to finish saving.';
  const { text, target } = snippetFields();
  if (!text.trim()) return 'Choose a snippet or enter some text first.';
  if (target === 'negative' && (els.negative.disabled || els.negField.classList.contains('unsupported'))) {
    return `The negative prompt is unavailable: ${els.negNote.textContent || 'this checkpoint takes no negative prompt at these settings.'}`;
  }
  const box = target === 'negative' ? els.negative : els.prompt;
  const length = text.length + box.value.length + (box.value ? 1 : 0);
  if (box.maxLength >= 0 && length > box.maxLength) return `The inserted text would exceed this box's ${box.maxLength}-character limit.`;
  return '';
}

function syncSnippets() {
  const held = (button, reason, title) => {
    button.setAttribute('aria-disabled', String(!!reason));
    button.title = reason || title;
  };
  held(els.snippetsSave, snippetSaveReason(), snippetsSelected ? 'Save changes to this snippet' : 'Save a new snippet');
  held(els.snippetsDelete, snippetDeleteReason(), 'Delete only the selected saved snippet');
  held(els.snippetsReload, snippetsPending ? 'Wait for the current snippets request.' : '',
    'Reread the saved list without replacing the editor; select an item to read its latest version');
  held(els.snippetsInsert, snippetInsertReason(), 'Insert the text into the chosen prompt; Ctrl+Z restores it');
  held(els.snippetsNew, snippetsPending === 'write' ? 'Wait for the snippet to finish saving.' : '', 'Start a new snippet');
  held(els.snippetsClose, snippetsPending === 'write' ? 'Wait for the snippet to finish saving.' : '', 'Close Snippets');
  for (const field of [els.snippetsList, els.snippetsName, els.snippetsText, els.snippetsTarget, els.snippetsPlacement]) {
    field.disabled = snippetsPending === 'write';
  }
  syncSnippetPortability();
}

function renderSnippets() {
  const list = els.snippetsList;
  list.replaceChildren();
  if (!snippetsItems.length) {
    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = 'No saved snippets yet';
    list.append(empty);
  }
  for (const item of snippetsItems) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${item.name} (${item.target === 'negative' ? 'Negative' : 'Prompt'})`;
    list.append(option);
  }
  list.value = snippetsSelected?.id || '';
  if (snippetSelectedReason() || snippetsWriteRecovery) list.selectedIndex = -1;
  syncSnippets();
}

function selectSnippet(id, userChoice = true) {
  if (snippetsPending === 'write') { snippetSay('Wait for the snippet to finish saving.'); return; }
  const item = snippetsItems.find((entry) => entry.id === id);
  if (!item) return;
  ++snippetsEditEpoch;
  if (userChoice && snippetsReady && !snippetsPending && snippetsWriteRecovery && !snippetsWriteRecovery.needsReload) {
    snippetsWriteRecovery = null;
  }
  snippetsSelected = { ...item };
  els.snippetsList.value = item.id;
  els.snippetsName.value = item.name;
  els.snippetsText.value = item.text;
  els.snippetsTarget.value = item.target;
  snippetsDirty = false;
  snippetSay(snippetWriteRecoveryReason());
  syncSnippets();
}

function newSnippet(userChoice = true) {
  if (snippetsPending === 'write') { snippetSay('Wait for the snippet to finish saving.'); return; }
  ++snippetsEditEpoch;
  if (userChoice && snippetsReady && !snippetsPending && snippetsWriteRecovery && !snippetsWriteRecovery.needsReload) {
    snippetsWriteRecovery = null;
  }
  snippetsSelected = null;
  snippetsDirty = false;
  els.snippetsList.value = '';
  els.snippetsName.value = '';
  els.snippetsText.value = '';
  els.snippetsTarget.value = 'prompt';
  snippetSay(['New snippet.', snippetWriteRecoveryReason()].filter(Boolean).join(' '));
  syncSnippets();
  els.snippetsName.focus({ preventScroll: true });
}

async function readSnippets({ opening = false } = {}) {
  if (!els.snippetsDialog.open) return;
  if (snippetsPending === 'write' || (!opening && snippetsPending)) {
    snippetSay('Wait for the current snippets request.'); return;
  }
  const epoch = snippetsEditEpoch;
  const ticket = ++snippetsReadTicket;
  snippetsReady = false;
  if (snippetsWriteRecovery) snippetsWriteRecovery.needsReload = true;
  snippetsPending = 'read';
  snippetSay('Reading saved snippets.');
  syncSnippets();
  try {
    const result = await api('/api/snippets');
    if (ticket !== snippetsReadTicket || !els.snippetsDialog.open) return;
    snippetsItems = result.items;
    snippetsReady = true;
    snippetsPending = null;
    if (snippetsWriteRecovery) snippetsWriteRecovery.needsReload = false;
    renderSnippets();
    // Reading a list never acknowledges an uncertain write. Only an explicit
    // selection or New after this read may authorize the next write.
    if (opening && !snippetsDirty && epoch === snippetsEditEpoch && !snippetsWriteRecovery) {
      const selected = snippetsItems.find((item) => item.id === snippetsSelected?.id);
      if (selected) selectSnippet(selected.id, false);
      else if (snippetsItems.length) selectSnippet(snippetsItems[0].id, false);
      else newSnippet(false);
    } else if (snippetsDirty || snippetsWriteRecovery) els.snippetsList.selectedIndex = -1;
    snippetSay(snippetWriteRecoveryReason() || (!opening
      ? 'Saved list reloaded. Your editor is unchanged; select a snippet to read its latest version.'
      : snippetsItems.length ? '' : 'Save your own tags, prefixes or negative prompts here.'));
    syncSnippets();
  } catch (err) {
    if (ticket !== snippetsReadTicket || !els.snippetsDialog.open) return;
    snippetsPending = null;
    snippetSay(`Snippets could not be read: ${err.message}`, true);
    syncSnippets();
  }
}

async function reloadSnippets() {
  await readSnippets();
}

async function openSnippets() {
  if (els.snippetsDialog.open) return;
  snippetsOrigin = document.activeElement;
  ++snippetsDialogEpoch;
  els.snippetsDialog.showModal();
  await readSnippets({ opening: true });
}

async function saveSnippet() {
  const reason = snippetSaveReason();
  if (reason) { snippetSay(reason, true); return; }
  const selected = snippetsSelected;
  const fields = snippetFields();
  const body = selected ? { ...fields, revision: selected.revision } : fields;
  snippetsPending = 'write';
  snippetSay('Saving snippet.');
  syncSnippets();
  try {
    const result = await api(selected ? `/api/snippets/${selected.id}` : '/api/snippets', {
      method: selected ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    snippetsItems = result.items;
    snippetsPending = null;
    renderSnippets();
    const saved = selected ? snippetsItems.find((item) => item.id === selected.id) : snippetsItems.at(-1);
    if (saved) {
      selectSnippet(saved.id, false);
      els.snippetsList.value = saved.id;
    }
    snippetSay('Snippet saved.');
  } catch (err) {
    snippetsPending = null;
    if (!err.status || err.status >= 500) {
      snippetsReady = false;
      snippetsWriteRecovery = { needsReload: true };
      snippetSay('Snippet save result is uncertain. Reload the list, then select a saved snippet or choose New. Your editor is unchanged.', true);
    } else {
      if (err.status === 409 || err.status === 404) snippetsReady = false;
      snippetSay(`Snippet was not saved: ${err.message}`, true);
    }
    syncSnippets();
  }
}

async function deleteSnippet() {
  const reason = snippetDeleteReason();
  if (reason) { snippetSay(reason, true); return; }
  const selected = snippetsSelected;
  snippetsPending = 'write';
  snippetSay('Deleting snippet.');
  syncSnippets();
  try {
    const result = await api(`/api/snippets/${selected.id}?revision=${selected.revision}`, { method: 'DELETE' });
    snippetsItems = result.items;
    snippetsPending = null;
    snippetsSelected = null;
    renderSnippets();
    if (snippetsItems.length) {
      selectSnippet(snippetsItems[0].id, false);
      els.snippetsList.value = snippetsItems[0].id;
    } else newSnippet(false);
    snippetSay('Snippet deleted. Text already inserted in a prompt is kept.');
  } catch (err) {
    snippetsPending = null;
    if (!err.status || err.status >= 500) {
      snippetsReady = false;
      snippetsWriteRecovery = { needsReload: true };
      snippetSay('Snippet delete result is uncertain. Reload the list, then select a saved snippet or choose New. Your editor is unchanged.', true);
    } else {
      if (err.status === 409 || err.status === 404) snippetsReady = false;
      snippetSay(`Snippet was not deleted: ${err.message}`, true);
    }
    syncSnippets();
  }
}

function closeSnippets() {
  if (snippetsPending === 'write') { snippetSay('Wait for the snippet to finish saving.'); return; }
  els.snippetsDialog.close();
}

function snippetsClosed() {
  // close fires asynchronously: an earlier close must not invalidate a
  // read started by a dialog that has already opened again.
  if (els.snippetsDialog.open) return;
  ++snippetsDialogEpoch;
  ++snippetsReadTicket;
  snippetsPending = null;
  const origin = snippetsOrigin;
  snippetsOrigin = null;
  if (origin?.isConnected) origin.focus({ preventScroll: true });
}

function insertSnippet() {
  const reason = snippetInsertReason();
  if (reason) { snippetSay(reason, true); return; }
  const { text, target } = snippetFields();
  const box = target === 'negative' ? els.negative : els.prompt;
  const before = box.value;
  const prepend = els.snippetsPlacement.value === 'prepend';
  const after = before ? (prepend ? `${text}\n${before}` : `${before}\n${text}`) : text;
  // Native modal dialogs make the page inert. Close before rewriting, then
  // focus without scrolling before the undo-aware write focuses the box.
  snippetsOrigin = null;
  els.snippetsDialog.close();
  rewriteBox(box, after);
  if (target === 'prompt') {
    promptReplaced();
    syncTagSelection();
    scheduleTagValidation(0);
  } else syncAnalysisStale();
  schedulePreview();
  persistPanel();
  box.focus({ preventScroll: true });
  const end = prepend ? text.length : after.length;
  box.setSelectionRange(end, end);
}

// Built with the Examples menu, after its author examples. The menu closes
// first so the dialog remembers its connected button as the focus origin.
function appendSnippetsMenuItem() {
  const item = document.createElement('li');
  item.id = 'snippets-menu-item';
  item.className = 'snippets-menu-item';
  item.setAttribute('role', 'menuitem');
  item.setAttribute('aria-haspopup', 'dialog');
  item.setAttribute('aria-controls', 'snippets-dialog');
  item.tabIndex = -1;
  item.textContent = 'Personal snippets…';
  const open = () => { showExamples(false); void openSnippets(); };
  item.onclick = open;
  item.onkeydown = (event) => {
    if (keyboardEventHeld(event) || keyboardCommandModifier(event) || event.shiftKey) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (!event.repeat) open();
    }
  };
  els.examplesMenu.appendChild(item);
}

function initSnippets() {
  els.snippetsClose.onclick = closeSnippets;
  els.snippetsDialog.addEventListener('close', snippetsClosed);
  els.snippetsDialog.addEventListener('cancel', (event) => {
    if (snippetsPending === 'write') { event.preventDefault(); snippetSay('Wait for the snippet to finish saving.'); }
  });
  els.snippetsList.onchange = () => selectSnippet(els.snippetsList.value);
  els.snippetsNew.onclick = () => newSnippet();
  els.snippetsReload.onclick = () => { void reloadSnippets(); };
  els.snippetsSave.onclick = () => { void saveSnippet(); };
  els.snippetsDelete.onclick = () => { void deleteSnippet(); };
  els.snippetsInsert.onclick = insertSnippet;
  for (const field of [els.snippetsName, els.snippetsText, els.snippetsTarget]) {
    field.addEventListener('input', () => { ++snippetsEditEpoch; snippetsDirty = true; syncSnippets(); });
    field.addEventListener('change', () => { ++snippetsEditEpoch; snippetsDirty = true; syncSnippets(); });
  }
}
