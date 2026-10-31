// AuraDream's page script: the note under the prompt and its Undo.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// What the note under the prompt holds (showPromptNote, below): an Example's
// or an expansion's write with its Undo, Reset's Undo, Expand's running line
// or a held expansion - or null.
let promptUndo = null;
let promptUndoTimer = null;

// The note under the prompt (#prompt-note), one at a time; docs/rules.md
// (the Examples menu's note under the prompt) has where it lies and what
// takes each kind. It holds `promptUndo`, {kind, after, undo, discard, onDrop, again}:
//   'write' - an Example's or an expansion's write, with its Undo (syncPromptUndo)
//   'info' - gone as 'write' is, with no button
//   'reset' - Reset's Undo (panel.js, syncResetUndo), lying in the emptied box
//   'running', 'held' - Expand's running line and a held expansion (prompt-expand.js)
// `host` puts it at the end of another box while shown, moved with
// `moveBefore` where the browser has it, which keeps the node's state: a live
// region removed and inserted again just before its words are written is
// often not read out. `again()` shows the same note anew - Expand offers back
// the one its running line took, where it fails.
const PROMPT_NOTE_HOME = els.promptNote.parentElement;
function showPromptNote(message, options = {}) {
  const {
    kind = 'write', undo = null, label = 'Undo', title = '', discard = null, ms = 15000, onDrop = null,
    host = PROMPT_NOTE_HOME, at = 'field',
  } = options;
  dropPromptUndo();
  if (els.promptNote.parentElement !== host) {
    let moved = false;
    if (typeof host.moveBefore === 'function') {
      try { host.moveBefore(els.promptNote, null); moved = true; } catch { /* inserted below */ }
    }
    if (!moved) host.append(els.promptNote);
  }
  const note = {
    kind, after: els.prompt.value, undo, discard, onDrop, again: () => showPromptNote(message, options),
  };
  promptUndo = note;
  // The kind is the stylesheet's too: Expand's running line and a held
  // expansion, which stay for minutes, take a line of the panel's flow
  // rather than lying over the next field.
  els.promptNote.dataset.kind = kind;
  // Where it lies (styles/fields.css): over the next field's caption, or in the
  // flow where that would cover something ('field'); in the box itself
  // ('box') for Reset's, whose box is empty while it shows - never in the
  // flow, which moved Generate as it came and went.
  els.promptNote.dataset.at = at;
  // Shown before it is written: it is a live region, and words written into
  // a hidden one are never read out.
  els.promptNote.hidden = false;
  els.promptNoteText.textContent = message;
  els.promptNoteUndo.hidden = !undo;
  els.promptNoteUndo.textContent = label;
  els.promptNoteUndo.title = title;
  els.promptNoteDiscard.hidden = !discard;
  els.promptExpandStop.hidden = !expandingPrompt;
  if (ms) {
    promptUndoTimer = setTimeout(() => {
      if (promptUndo === note && !els.promptNote.contains(document.activeElement)) dropPromptUndo();
    }, ms);
  }
  return note;
}

function offerPromptUndo(message, undo, options = {}) {
  return showPromptNote(message, { ...options, undo });
}

function dropPromptUndo() {
  clearTimeout(promptUndoTimer);
  const note = promptUndo;
  promptUndo = null;
  // Focus on its button goes back to the box the note is about, not <body>.
  if (els.promptNote.contains(document.activeElement)) els.prompt.focus({ preventScroll: true });
  els.promptNote.hidden = true;
  note?.onDrop?.();
  // Expand's Stop remains reachable after a Reset/Reuse or an expired note.
  // Wait until the current write has installed its own note before deciding
  // whether a plain running line is needed; never replace that write's Undo.
  queueMicrotask(syncExpandStop);
}

function syncPromptUndo() {
  if (promptUndo && ['write', 'info'].includes(promptUndo.kind) && els.prompt.value !== promptUndo.after) {
    dropPromptUndo();
  }
}

// The note's Undo (or Use it): a write's undo only while the box holds what
// the write left; Reset's and a held expansion's whatever the box holds.
function undoPromptWrite() {
  const pending = promptUndo;
  if (!pending?.undo) return;
  dropPromptUndo();
  if (pending.kind !== 'write' || els.prompt.value === pending.after) pending.undo();
}

function discardPromptNote() {
  const pending = promptUndo;
  dropPromptUndo();
  pending?.discard?.();
}
