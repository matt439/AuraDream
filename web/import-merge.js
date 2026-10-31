// AuraDream's page script: what the two file-index merge dialogs share -
// snippet-import.js's personal snippets and setup-import.js's saved setups:
// the file read, the request envelope, the choice bookkeeping and reasons,
// the held buttons and the rows drawn from a server preview. Stateless: each
// dialog keeps its own state, preview, commit, reload and uncertain-import
// flow, and passes what these read. A `kind` names the dialog's words and
// how its rows show an item (SNIPPET_IMPORT_KIND, SETUP_IMPORT_KIND).
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// Portable files are data. Keep their original JSON until the server has
// checked duplicate keys; parsing and reserializing here would erase them.
async function readImportFile(file, limit, noun) {
  if (file.size > limit) throw new Error(`The ${noun} file exceeds 8 MiB.`);
  return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer());
}

function importEnvelope(raw, choices, extra = {}) {
  return '{"payload":' + raw + ',"decisions":' + JSON.stringify(choices)
    + Object.entries(extra).map(([key, value]) => ',' + JSON.stringify(key) + ':' + JSON.stringify(value)).join('') + '}';
}

function importNode(tag, text, className = '') {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function importHold(button, reason, title) {
  button.setAttribute('aria-disabled', String(!!reason));
  button.title = reason || title;
}

// Per file index, the latest server row that describes the file item under its
// own name. A previewed Add as copy row is matched under its copy name, so it
// never replaces this: Add/Replace, saved matches and the heading keep the
// file item's real collision.
function noteImportIncoming(incoming, rows) {
  for (const row of rows) if (row.action !== 'copy' || !incoming.has(row.index)) incoming.set(row.index, row);
}

// The server's default choice for each row, as a decision it accepts back.
function importChoices(kind, rows) {
  return rows.map((row) => ({
    index: row.index, action: row.action,
    ...(row.action === 'copy' ? { name: kind.item(row).name } : {}),
    ...(row.action === 'replace' ? { item_id: row.item_id, revision: row.revision } : {}),
  }));
}

function importChoiceReason(kind, choices, copyConfirm, incomingRow) {
  for (const choice of choices) {
    const item = `file item #${choice.index + 1}`;
    if (copyConfirm.has(choice.index)) return `Choose the action again for ${item}; its previous Add as copy may already have been saved.`;
    if (!['add', 'copy', 'replace', 'skip'].includes(choice.action)) return `Choose an action for ${item}.`;
    const row = incomingRow(choice.index);
    if (choice.action === 'add' && row?.matches.length) return kind.collides(choice.index + 1);
    if (choice.action === 'replace' && !row?.matches.some((local) => local.id === choice.item_id && local.revision === choice.revision)) {
      return `Choose the saved ${kind.noun} to replace for ${item}.`;
    }
    if (choice.action === 'copy' && (!choice.name?.trim() || Array.from(choice.name).length > 80)) return `Give copy #${choice.index + 1} a name of 1-80 characters.`;
  }
  return '';
}

function importReason(kind, busy, raw, shown, choiceReason) {
  if (busy) return busy === 'commit' ? 'Wait for the import to finish.' : 'Wait for the file preview.';
  if (raw === null) return `Choose a valid ${kind.noun} file first.`;
  return choiceReason() || (!shown ? 'Preview these choices before importing.'
    : !shown.can_import ? shown.reason || 'The selected import changes nothing.' : '');
}

// The merge dialog's own buttons and row controls; the parent dialog's
// Export… and Import… are each file's.
function syncImportHolds(kind, importEls, { busy, raw, needsReload, reason, choiceReason }) {
  importHold(importEls.run, reason, 'Commit exactly the previewed merge');
  importHold(importEls.preview, busy ? 'Wait for the current import request.'
    : raw === null ? `Choose a valid ${kind.noun} file first.`
      : needsReload ? '' : choiceReason(), 'Preview the chosen merge against the saved list');
  importHold(importEls.close, busy === 'commit' ? 'Wait for the import to finish.' : '', 'Cancel without importing');
  for (const control of importEls.rows.querySelectorAll('input, select')) control.disabled = !!busy;
}

// The rows of a preview, each file item's choice wired to `merge.choices` (the
// dialog's own array, its objects changed in place) and every change calling
// `merge.invalidate`. Import rows render file text through textContent only.
function renderImportRows(kind, shown, keepChoices, merge) {
  const { els: importEls, choices, copyConfirm } = merge;
  importEls.rows.replaceChildren();
  for (const shownRow of shown.rows) {
    // Same index, item and its saved place; only the copy name may differ.
    const row = { ...merge.incomingRow(shownRow.index), incoming_collisions: shownRow.incoming_collisions };
    const choice = choices.find((entry) => entry.index === row.index);
    const incoming = kind.item(row);
    const section = importNode('section', undefined, `${kind.prefix}-import-row import-row`);
    section.dataset.index = String(row.index);
    const heading = importNode('h3', kind.heading(row));
    heading.id = `${kind.prefix}-import-row-${row.index}`;
    section.setAttribute('aria-labelledby', heading.id);
    section.append(heading);
    if (kind.meta) {
      const meta = importNode('p', kind.meta(row), `${kind.prefix}-import-meta`);
      meta.dataset.role = 'meta';
      section.append(meta);
    }
    if (row.exact_ids.length) section.append(importNode('p', `Exact saved match: ${row.exact_ids.map((id) => '#' + (row.matches.findIndex((local) => local.id === id) + 1)).join(', ')}`, 'hint'));
    if (row.incoming_collisions.length) section.append(importNode('p', `Same incoming name and ${kind.sameAs} at ${row.incoming_collisions.map((index) => '#' + (index + 1)).join(', ')}. Each file item has its own choice.`, 'hint'));
    const shownItem = kind.incoming(row);
    const text = importNode('pre', shownItem.text);
    text.dataset.role = shownItem.role;
    text.setAttribute('aria-label', shownItem.label);
    section.append(text);
    for (const [candidateIndex, local] of row.matches.entries()) {
      const details = document.createElement('details');
      details.open = !row.exact_ids.includes(local.id);
      details.append(importNode('summary', kind.matchSummary(local, candidateIndex)));
      const shownLocal = kind.local(local);
      const localText = importNode('pre', shownLocal.text);
      localText.dataset.role = shownLocal.role; localText.dataset.itemId = local.id;
      details.append(localText); section.append(details);
    }
    const controls = importNode('div', undefined, `${kind.prefix}-import-choice import-choice`);
    const actionLabel = importNode('label', 'Action');
    const action = document.createElement('select'); action.dataset.role = 'action';
    if (copyConfirm.has(row.index)) {
      const confirm = importNode('option', 'Choose action again (previous Add as copy)');
      confirm.value = ''; confirm.disabled = true; action.append(confirm);
    }
    for (const [value, label] of [['skip', 'Skip'], ['add', row.matches.length ? 'Add (name now collides)' : 'Add'],
      ['copy', 'Add as copy'], ['replace', `Replace a saved ${kind.noun}`]]) {
      const option = importNode('option', label); option.value = value;
      option.disabled = value === 'add' && row.matches.length > 0 || value === 'replace' && !row.matches.length;
      action.append(option);
    }
    action.value = copyConfirm.has(row.index) ? '' : choice.action;
    actionLabel.append(action); controls.append(actionLabel);
    const copyLabel = importNode('label', 'Copy name');
    const name = document.createElement('input'); name.type = 'text'; name.maxLength = 160; name.autocomplete = 'off';
    name.dataset.role = 'copy-name'; name.value = choice.name ?? incoming.name;
    copyLabel.hidden = choice.action !== 'copy'; copyLabel.append(name); controls.append(copyLabel);
    const targetLabel = importNode('label', kind.targetLabel);
    const target = document.createElement('select'); target.dataset.role = 'replacement';
    const blank = importNode('option', `Choose a saved ${kind.noun}`); blank.value = ''; target.append(blank);
    for (const [candidateIndex, local] of row.matches.entries()) {
      const option = importNode('option', kind.matchOption(local, candidateIndex));
      option.value = local.id; target.append(option);
    }
    target.value = row.matches.some((local) => local.id === choice.item_id && local.revision === choice.revision) ? choice.item_id : '';
    targetLabel.hidden = choice.action !== 'replace'; targetLabel.append(target); controls.append(targetLabel);
    action.onchange = () => {
      copyConfirm.delete(row.index);
      choice.action = action.value; delete choice.item_id; delete choice.revision; delete choice.name;
      if (choice.action === 'copy') choice.name = name.value;
      target.value = ''; copyLabel.hidden = choice.action !== 'copy'; targetLabel.hidden = choice.action !== 'replace';
      merge.invalidate();
    };
    name.oninput = () => { choice.name = name.value; merge.invalidate(); };
    target.onchange = () => {
      const local = row.matches.find((entry) => entry.id === target.value);
      delete choice.item_id; delete choice.revision;
      if (local) { choice.item_id = local.id; choice.revision = local.revision; }
      merge.invalidate();
    };
    section.append(controls); importEls.rows.append(section);
  }
  if (!shown.rows.length) importEls.rows.append(importNode('p', `This file contains no ${kind.noun}s. Nothing will be changed.`, 'hint'));
  importEls.summary.textContent = keepChoices ? 'Saved list refreshed. Check your choices, then Preview.'
    : `${shown.add_count} added; ${shown.replace_count} replaced; ${shown.skip_count} skipped`;
}
