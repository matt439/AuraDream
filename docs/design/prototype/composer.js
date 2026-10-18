'use strict';

// Declarations only: index.html installs handlers and starts the concept.
// These are the three illustrative aspects represented by this prototype,
// copied from backend/sizes.py (the application owns the full preset table).
const composerSizes = {
  '1:1': {'1': [1024, 1024], '1.5': [1248, 1248], '2': [1440, 1440], '4': [2048, 2048]},
  '7:9': {'1': [896, 1152], '1.5': [1120, 1440], '2': [1264, 1632], '4': [1808, 2320]},
  '9:7': {'1': [1152, 896], '1.5': [1440, 1120], '2': [1632, 1264], '4': [2320, 1808]},
};

// Represented-field subset of web/panel.js resetPanel(). The sample model's
// defaults come from backend/model_defaults.py's exact WAI v17 profile.
// Real Reset also clears conditioning, grid, editing drafts and saved panel
// state; those operations do not exist in this disconnected prototype.
const composerResetDefaults = {
  prompt: '', negative: '', aspect: '1:1', mp: '1', seed: '',
  batch: '1', repeat: '1', steps: '28', cfg: '6', mu: '',
};

function readDraft() {
  const values = {};
  for (const id of ['prompt', 'negative', 'aspect', 'mp', 'seed', 'batch', 'repeat', 'steps', 'cfg', 'mu']) {
    values[id] = byId(id).value;
  }
  return values;
}

// Whole-box writes retain the browser's text Undo. setRangeText is only a
// fallback for a browser refusing insertText; the explicit draft Undo still
// restores the complete snapshot in either case.
function replaceBoxText(field, text) {
  if (field.value === text) return;
  field.setSelectionRange(0, field.value.length);
  if (!document.execCommand('insertText', false, text)) {
    field.setRangeText(text, 0, field.value.length, 'end');
  }
}

function rewriteBox(field, text) {
  field.focus({preventScroll: true});
  replaceBoxText(field, text);
}

function restoreDraft(values) {
  // Programmatic insertText emits input. Keep the shared settings snapshot
  // while index.html's ordinary-edit handler retires it during those writes.
  const previousUndo = draftUndo;
  const previousUndoAfter = draftUndoAfter;
  try {
    for (const id of ['prompt', 'negative']) {
      if (Object.hasOwn(values, id)) rewriteBox(byId(id), String(values[id] ?? ''));
    }
    for (const id of ['aspect', 'mp', 'seed', 'batch', 'repeat', 'steps', 'cfg', 'mu']) {
      if (Object.hasOwn(values, id)) byId(id).value = String(values[id] ?? '');
    }
    // Sample inspected records retain their original combined Size field.
    // Full snapshots use the independent aspect and tier fields above.
    if (values.size && !Object.hasOwn(values, 'aspect') && !Object.hasOwn(values, 'mp')) {
      const dimensions = String(values.size).match(/\d+/g)?.map(Number);
      for (const [aspect, tiers] of Object.entries(composerSizes)) {
        for (const [tier, [width, height]] of Object.entries(tiers)) {
          if (dimensions?.[0] === width && dimensions?.[1] === height) {
            byId('aspect').value = aspect;
            byId('mp').value = tier;
          }
        }
      }
    }
  } finally {
    draftUndo = previousUndo;
    draftUndoAfter = previousUndoAfter;
  }
  syncComposer();
}

function syncComposer() {
  const dimensions = composerSizes[byId('aspect').value]?.[byId('mp').value];
  byId('dimensions').textContent = dimensions ? `${dimensions[0]} × ${dimensions[1]} px` : 'Choose a represented preset';
  const batch = Number(byId('batch').value);
  const repeat = Number(byId('repeat').value);
  const validCount = Number.isInteger(batch) && batch >= 1 && batch <= 8 && Number.isInteger(repeat) && repeat >= 1;
  byId('output-plan').textContent = validCount
    ? `${batch * repeat} ${batch * repeat === 1 ? 'image' : 'images'} · ${repeat} ${repeat === 1 ? 'batch' : 'batches'} of ${batch}${repeat > 1 ? ', one after another' : ''} · ${byId('seed').value.trim() ? `start seed ${byId('seed').value.trim()}` : 'random start (Seed is empty)'}`
    : 'Enter whole Batch and Repeat counts to see the image total.';
  byId('sampling-summary').textContent = `Sampling · ${byId('steps').value || '—'} steps · CFG ${byId('cfg').value || '—'} · Mu unavailable for WAI`;
}

function initComposer() {
  for (const id of ['aspect', 'mp', 'seed', 'batch', 'repeat', 'steps', 'cfg', 'mu']) {
    byId(id).addEventListener('input', syncComposer);
    byId(id).addEventListener('change', syncComposer);
  }
  byId('reset').addEventListener('click', () => {
    const before = readDraft();
    restoreDraft(composerResetDefaults);
    draftUndo = before;
    draftUndoAfter = readDraft();
    byId('undo').textContent = 'Undo Reset';
    byId('undo').hidden = false;
    byId('draft-note').textContent = 'Panel reset. Seed is empty; enter your next seed. Undo Reset restores the previous draft.';
    byId('prompt').focus({preventScroll: true});
  });
  syncComposer();
}
