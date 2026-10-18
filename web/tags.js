// Local Diffusion's page script: prompt writes, tag autocomplete and editing.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------------ tag state

// The chips' lookup asks about this many comma items at most: the request's
// own ceiling (schemas.TagValidationRequest), which the analyser's category
// lookup shares. The items past it are sent nowhere, stay unchecked, and the
// tray says how many were checked - a 65-item prompt used to grey out every
// chip, guide tags included, behind a 422.
const TAG_LOOKUP_CAP = 64;
// What the tag index last said about each tag, by `tagKey`, for the running
// checkpoint only: another's cutoff and guide tags give other answers. A
// pause in typing renders every chip it knows at once, so only a new or
// changed item waits on the lookup - every chip used to go grey and dashed at
// each keystroke and come back green a moment later.
let tagVerdicts = { model: null, results: new Map() };
// The last answer's order slots ({model, prompt, steps, slots}): each comma
// item of `prompt` with the step of the guide's documented order the
// server's order rule places it in - POST /api/tags/validate, given the
// prompt, places it as a Check of that prompt would (prompt_rules'
// `order_slots`). `slots` is null where the rules place nothing in it - a
// lone tag they do place - and `failed` is set where the request itself
// failed, so the Tags helper's caret line can tell a pending answer from one
// that will not come. The page places nothing itself: the order is the
// guide's, per checkpoint.
let tagSlots = null;
// The chip focus belongs to while it is in the tray - a number, or 'add' for
// the + tag button - which every rebuild of the tray puts back: a move, a
// delete and a validation landing each rebuild it, and focus fell to <body>
// after one Alt+Arrow. Set by a chip taking focus and by a write that hands
// focus back to the chip it moved (`tagRefocus`: the write took focus to the
// prompt); cleared when focus leaves the tray. A rebuild takes focus back only
// from <body>, where the rebuild itself left it, or from such a write - never
// from a box the user is typing in.
let tagFocusIndex = null;
let tagRefocus = false;
let tagRendering = false;
// The comma item being typed, {index, tag}, while the prompt has the focus
// and that item has changed since the last keystroke. An unknown verdict on
// it waits (`.typing`, no mark) until a comma, the caret moving out of it or
// 1.5 s without typing: the one red chip was the word still being written.
let tagTyping = null;
let tagTypingTimer = null;
// The tray's own count of the prompt: POST /api/prompt/measure, Check's
// encoder layer alone, after a pause in typing - {model, text, encoders,
// detail}. The label shows the last one while the next is counted.
let tagMeasured = null;
let tagMeasureTimer = null;
let tagMeasureEpoch = 0;
let tagHoverFrame = 0;
// Where the pointer rests over one of the tray's gaps, or null: a re-render -
// a lookup or a count landing - rebuilds the chips and with them the bar, which
// a pointer resting in the gap brought back only by moving again.
let tagHoverPoint = null;
// What the note under the prompt holds (showPromptNote, below): an Example's
// or an expansion's write with its Undo, Reset's Undo, Expand's running line
// or a held expansion - or null.
let promptUndo = null;
let promptUndoTimer = null;

// A tag's comparable form: what the index answers the same for. Case and
// spacing, not underscores - the guide's own tags match as typed, spaces and
// all.
const tagKey = (tag) => tag.trim().toLowerCase().replace(/\s+/g, ' ');
// A name as the box writes it and as the index does, compared: escapes and
// underscores folded too.
const tagName = (text) => tagKey(text.replace(/\\([()])/g, '$1').replace(/_/g, ' '));

// ------------------------------------------------------ writing the prompt

// replaceBoxText (analysis.js), for the tag tools, the Examples and Reuse
// (reuse.js's rewriteBox): the box rewritten the way typing writes it, so
// Ctrl+Z brings back what it held -
// assigning `.value` emptied the undo history, and in Chromium a later Ctrl+Z
// could splice two prompts together, which the panel then saved. The insert's
// `input` asks for suggestions and marks the caret's item as being typed, as
// typing does; a write is neither, and nothing opens over the new text.
function writeBox(el, text) {
  replaceBoxText(el, text);
  clearTimeout(tagSuggestionTimer);
  hideTagSuggestions();
  clearTimeout(tagTypingTimer);
  tagTyping = null;
}

// The note under the prompt (#prompt-note), one at a time, said where the
// press was: over the next field's caption on a two-column page - out of the
// panel's flow, so its coming and going moves nothing, Generate least - and
// in the flow in one column, while the negative box is greyed, and for the
// kinds with no timer, 'running' and 'held' (`data-kind`, style.css), which
// would otherwise cover a live negative box for minutes.
// What it holds is `promptUndo`, {kind, after, undo, discard, onDrop, again}:
//   'write' - a write the user did not type over their own words, an Example
//     picked over a prompt in progress or an expansion (app.js), with an Undo
//     that puts it back. It goes once the box holds anything else, `undo`
//     being about the text it replaced (syncPromptUndo); Ctrl+Z in the box
//     does the same for longer. 'info' goes the same way, with no button.
//   'reset' - Reset's Undo (panel.js), retired by Reset's own rule, the first
//     change of the panel (syncResetUndo), and after 20 s. It lies in the
//     emptied box itself (`at: 'box'`), as does the note its Undo leaves.
//   'running', 'held' - ERNIE's Expand while it runs, and an expansion the
//     user typed over meanwhile (Use it, Discard): typing takes neither.
// Each goes after `ms` (none for 0) unless focus is on its buttons. One Undo
// on offer at a time: the write that brings this one takes a Reuse's
// (reuse.js's syncReuseUndo), and a Reuse takes this. `onDrop` runs when the
// note goes, whatever takes it. `host` puts it at the end of another box for
// the time it is shown - an expansion's, in its English at the head of the
// results from 1800 px (app.js), where the panel would have pushed Generate
// below the window - and the next note brings it back under the prompt.
// Moved with `moveBefore` where the browser has it, which keeps the node's
// state: a live region removed and inserted again just before its words are
// written is often not read out. `again()` shows the same note anew - Expand
// offers back the one its running line took, where it fails.
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
  // Where it lies (style.css): over the next field's caption, or in the
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

// ------------------------------------------------------------ prompt tag helper

function promptTagParts() {
  return els.prompt.value.split(',').map((part) => part.trim()).filter(Boolean);
}

// Where each of promptTagParts() sits in the box, index for index: [start, end)
// of the trimmed item.
function promptTagSpans() {
  const spans = [];
  let from = 0;
  for (const part of els.prompt.value.split(',')) {
    const body = part.trim();
    const start = from + part.length - part.trimStart().length;
    if (body) spans.push({ start, end: start + body.length });
    from += part.length + 1;
  }
  return spans;
}

// The index of the comma item the caret is in, or null in an empty one - the
// moment after a comma.
function caretTagIndex() {
  const text = els.prompt.value;
  const caret = els.prompt.selectionStart ?? text.length;
  const start = text.lastIndexOf(',', caret - 1) + 1;
  const comma = text.indexOf(',', caret);
  const end = comma < 0 ? text.length : comma;
  const index = promptTagSpans().findIndex((span) => span.start >= start && span.end <= end);
  return index < 0 ? null : index;
}

// The comma item at the caret, trimmed: what a suggestion replaces. The
// panel's prompt, or the Refine dialog's (refine-prompt.js).
function currentTagRange(el = els.prompt) {
  const text = el.value;
  const caret = el.selectionStart ?? text.length;
  let start = text.lastIndexOf(',', Math.max(0, caret - 1)) + 1;
  let end = text.indexOf(',', caret);
  if (end < 0) end = text.length;
  while (start < end && /\s/.test(text[start])) start += 1;
  while (end > start && /\s/.test(text[end - 1])) end -= 1;
  return { start, end, query: text.slice(start, end).trim() };
}

function hideTagSuggestions() {
  tagSuggestionEpoch += 1;  // an in-flight response must not reopen a dismissed list
  tagSuggestions = [];
  tagSuggestionIndex = -1;
  els.tagSuggestions.hidden = true;
  els.tagSuggestions.replaceChildren();
  els.prompt.setAttribute('aria-expanded', 'false');
  els.prompt.removeAttribute('aria-activedescendant');
}

function humanCount(value) {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}m`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(value >= 1e5 ? 0 : 1)}k`;
  return String(value);
}

// The checkpoint guide's own tags come back as `model`; the chips call them
// guide tags, and so does the list.
const tagKind = (category) => (category === 'model' ? 'guide' : category);

// Whether the first suggestion is selected as the list opens, so Enter or Tab
// takes it: only once three characters are typed and it starts with them.
// Before that Tab moves on and Enter is a new line - "lo" then Tab, pressed
// to leave the box, wrote whatever came first. The server ranks the list
// (backend/tags.rank_suggestions); ArrowDown still selects from the top.
function preselects(query, suggestion) {
  const typed = tagName(query);
  return typed.length >= 3 && tagName(suggestion.text).startsWith(typed);
}

function selectTagSuggestion(index) {
  if (!tagSuggestions.length) return;
  tagSuggestionIndex = (index + tagSuggestions.length) % tagSuggestions.length;
  [...els.tagSuggestions.children].forEach((item, i) => {
    const active = i === tagSuggestionIndex;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
    if (active) {
      els.prompt.setAttribute('aria-activedescendant', item.id);
      item.scrollIntoView({ block: 'nearest' });
    }
  });
}

// The word at the caret replaced by an insert, as typing replaces a
// selection: one Ctrl+Z takes the suggestion back and leaves what was typed.
function applyTagSuggestion(suggestion) {
  const { start, end } = currentTagRange();
  els.prompt.focus();
  els.prompt.setSelectionRange(start, end);
  const inserted = document.execCommand('insertText', false, suggestion.text);
  if (!inserted) els.prompt.setRangeText(suggestion.text, start, end, 'end');
  // An accepted suggestion is a finished tag: nothing opens over it, and it
  // is not the item being typed.
  clearTimeout(tagSuggestionTimer);
  hideTagSuggestions();
  clearTimeout(tagTypingTimer);
  tagTyping = null;
  promptReplaced();
  syncTagSelection();
  schedulePreview();
  persistPanel();
  scheduleTagValidation(0);
}

function renderTagSuggestions(items, query = '') {
  tagSuggestions = items;
  tagSuggestionIndex = -1;
  els.tagSuggestions.replaceChildren();
  els.prompt.removeAttribute('aria-activedescendant');
  for (const [index, suggestion] of items.entries()) {
    const item = document.createElement('li');
    item.id = `tag-suggestion-${index}`;
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', 'false');
    const kind = document.createElement('span');
    kind.className = 'tag-kind';
    kind.textContent = tagKind(suggestion.category);
    const name = document.createElement('span');
    name.className = 'tag-name';
    name.textContent = suggestion.text;
    const count = document.createElement('span');
    count.className = 'tag-count';
    count.textContent = suggestion.post_count == null ? '' : humanCount(suggestion.post_count);
    item.append(kind, name, count);
    item.onmousedown = (event) => event.preventDefault();
    item.onclick = () => applyTagSuggestion(suggestion);
    els.tagSuggestions.appendChild(item);
  }
  els.tagSuggestions.hidden = items.length === 0;
  els.prompt.setAttribute('aria-expanded', String(items.length > 0));
  if (items.length && preselects(query, items[0])) selectTagSuggestion(0);
}

async function requestTagSuggestions() {
  const epoch = ++tagSuggestionEpoch;
  if (!tagVocabularyEnabled || document.activeElement !== els.prompt) {
    hideTagSuggestions();
    return;
  }
  const { query } = currentTagRange();
  if (!query) {
    hideTagSuggestions();
    return;
  }
  try {
    const body = await api(`/api/tags/suggest?q=${encodeURIComponent(query)}`);
    if (epoch !== tagSuggestionEpoch) return;
    renderTagSuggestions(body.suggestions || [], query);
  } catch {
    if (epoch === tagSuggestionEpoch) hideTagSuggestions();
  }
}

function scheduleTagSuggestions(delay = 160) {
  clearTimeout(tagSuggestionTimer);
  tagSuggestionTimer = setTimeout(requestTagSuggestions, delay);
}

// Each chip edit hands focus to the chip it leaves in place: the next one
// after a delete, the moved one after a move or a drop, the edited or
// inserted one after an edit.
function deletePromptTag(index) {
  const tags = promptTagParts();
  if (index < 0 || index >= tags.length) return;
  tags.splice(index, 1);
  writePromptTags(tags, null, Math.min(index, tags.length - 1));
}

function movePromptTag(from, to) {
  const tags = promptTagParts();
  if (from < 0 || from >= tags.length || to < 0 || to >= tags.length || from === to) return;
  const [tag] = tags.splice(from, 1);
  tags.splice(to, 0, tag);
  writePromptTags(tags, null, to);
}

function dropPromptTag(from, insertion) {
  const tags = promptTagParts();
  if (from < 0 || from >= tags.length) return;
  const [tag] = tags.splice(from, 1);
  const target = Math.max(0, Math.min(tags.length, insertion > from ? insertion - 1 : insertion));
  if (target === from) return;
  tags.splice(target, 0, tag);
  writePromptTags(tags, null, target);
}

function clearTagDropMarkers() {
  for (const chip of els.tagValidation.querySelectorAll('.tag-check')) {
    chip.classList.remove('drop-before', 'drop-after');
  }
}

// The gap a click on the tray would insert at, marked as a drop is: the same
// bar, on the chip either side of it. None off the rows, where a click does
// nothing.
function markTagGap(x, y) {
  clearTagDropMarkers();
  const chips = els.tagValidation.querySelectorAll('.tag-check');
  const index = tagInsertionIndexAtPoint(x, y, { gap: true });
  if (index === null) return;
  const before = els.tagValidation.querySelector(`.tag-check[data-index="${index}"]`);
  if (before) before.classList.add('drop-before');
  else chips[chips.length - 1].classList.add('drop-after');
}

// Escape or × in the editor: focus goes back to the chip it was editing, or
// beside the gap it was inserting into, rather than to <body> with the input.
function closeTagEditor() {
  const edited = tagEditAt;
  const inserted = tagInsertAt;
  const held = els.tagValidation.contains(document.activeElement);
  tagInsertAt = null;
  tagEditAt = null;
  tagEditorSuggestionEpoch += 1;
  clearTimeout(tagEditorSuggestionTimer);
  if (held && edited !== null) tagFocusIndex = edited;
  else if (held && inserted !== null) {
    tagFocusIndex = inserted >= lastTagValidationResults.length ? 'add' : Math.max(0, inserted - 1);
  }
  renderTagValidation(
    lastTagValidationResults, lastTagValidationState, lastTagValidationDetail,
  );
}

// The one exit from the editor that changes the prompt: a new item before
// `tagInsertAt`, or the item at `tagEditAt` replaced. An unchanged edit only
// closes.
function commitTagEdit(text) {
  const tag = text.trim();
  if (!tag || tag.includes(',')) return;
  const tags = promptTagParts();
  let at;
  if (tagEditAt !== null) {
    if (tagEditAt >= tags.length) return;
    if (tags[tagEditAt] === tag) { closeTagEditor(); return; }
    tags[tagEditAt] = tag;
    at = tagEditAt;
  } else if (tagInsertAt !== null) {
    at = Math.min(tagInsertAt, tags.length);
    tags.splice(at, 0, tag);
  } else {
    return;
  }
  writePromptTags(tags, null, at);
}

function selectEditorSuggestion(menu, index) {
  const suggestions = menu._suggestions || [];
  if (!suggestions.length) return;
  menu._index = (index + suggestions.length) % suggestions.length;
  [...menu.children].forEach((item, i) => {
    const active = i === menu._index;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
    if (active) {
      menu._input?.setAttribute('aria-activedescendant', item.id);
      item.scrollIntoView({ block: 'nearest' });
    }
  });
}

// The suggestion list of a box other than the panel's prompt: the chip
// editor's input, where a pick commits the edit, and the Refine dialog's
// prompt (refine-prompt.js), where it replaces the comma item at the caret - `pick`.
// `query` is what was looked up, which the preselect rule reads: the editor's
// whole text, the dialog's item at the caret. A click, or Enter or Tab on
// the selected item (the box's own keydown, through `menu._pick`), picks.
// `place`, given, positions the list once it is drawn, at its real height.
// `menu._query` is the item a drawn list answers.
function renderEditorSuggestions(
  input, menu, suggestions, { query = input.value, pick = (s) => commitTagEdit(s.text), place = null } = {},
) {
  menu._suggestions = suggestions;
  menu._query = suggestions.length ? query.trim() : null;
  menu._index = -1;
  menu._input = input;
  menu._pick = pick;
  menu.replaceChildren();
  input.removeAttribute('aria-activedescendant');
  for (const [index, suggestion] of suggestions.entries()) {
    const item = document.createElement('li');
    item.id = `${menu.id}-${index}`;
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', 'false');
    const kind = document.createElement('span');
    kind.className = 'tag-kind';
    kind.textContent = tagKind(suggestion.category);
    const name = document.createElement('span');
    name.className = 'tag-name';
    name.textContent = suggestion.text;
    const count = document.createElement('span');
    count.className = 'tag-count';
    count.textContent = suggestion.post_count == null ? '' : humanCount(suggestion.post_count);
    item.append(kind, name, count);
    item.onmousedown = (event) => event.preventDefault();
    item.onclick = () => pick(suggestion);
    menu.appendChild(item);
  }
  menu.hidden = suggestions.length === 0;
  input.setAttribute('aria-expanded', String(suggestions.length > 0));
  const editor = menu.parentElement.getBoundingClientRect();
  menu.classList.toggle('end', editor.left + 280 > window.innerWidth - 12);
  place?.(menu);
  // The prompt box's rule: "lo" then Tab in the editor wrote `low quality`.
  if (suggestions.length && preselects(query, suggestions[0])) selectEditorSuggestion(menu, 0);
}

// Nothing selected in a list left up: Enter and Tab are the box's own again.
function deselectEditorSuggestions(menu) {
  menu._index = -1;
  for (const item of menu.children) {
    item.classList.remove('active');
    item.setAttribute('aria-selected', 'false');
  }
  menu._input?.removeAttribute('aria-activedescendant');
}

// The list emptied at once and asked again after a pause, so Enter never
// takes a match for what was typed before. With `keep` (the Refine dialog's
// prompt) a list already up stays while it asks, nothing in it selected -
// the same guarantee, without the list blinking shut on every key, as the
// panel's prompt keeps its own. The rest of `options` is
// renderEditorSuggestions'.
function scheduleEditorSuggestions(input, menu, options = {}) {
  clearTimeout(tagEditorSuggestionTimer);
  const epoch = ++tagEditorSuggestionEpoch;
  const query = (options.query ?? input.value).trim();
  if (options.keep && query && !menu.hidden) deselectEditorSuggestions(menu);
  else renderEditorSuggestions(input, menu, [], options);
  if (!query) return;
  tagEditorSuggestionTimer = setTimeout(async () => {
    try {
      const body = await api(`/api/tags/suggest?q=${encodeURIComponent(query)}`);
      if (epoch !== tagEditorSuggestionEpoch || !input.isConnected) return;
      renderEditorSuggestions(input, menu, body.suggestions || [], options);
    } catch {
      if (epoch === tagEditorSuggestionEpoch) renderEditorSuggestions(input, menu, [], options);
    }
  }, 160);
}

// One editor for both jobs: empty in a gap to insert, or in a chip's place
// holding its text to rewrite it. Suggestions come only once the text is
// typed, so Enter on an untouched edit keeps the tag rather than taking the
// top match for it.
function buildTagEditor(initial = null) {
  const editing = initial !== null;
  const editor = document.createElement('span');
  editor.className = 'tag-editor';
  const input = document.createElement('input');
  input.type = 'text';
  input.maxLength = 200;
  input.placeholder = editing ? 'edit tag' : 'insert tag';
  input.value = initial ?? '';
  if (editing && initial.length > 24) input.style.width = `${Math.min(initial.length + 2, 64)}ch`;
  input.setAttribute('aria-label', editing ? `Edit ${initial}` : 'New tag');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-expanded', 'false');
  const menu = document.createElement('ul');
  menu.className = 'tag-editor-suggestions';
  menu.id = 'tag-editor-suggestions';
  menu.setAttribute('role', 'listbox');
  // Named as the prompt box's list and the Refine dialog's are; one editor is
  // open at a time, so the id - and its options' ids, which the input's
  // aria-activedescendant names - are the page's only ones.
  menu.setAttribute('aria-label', 'Tag suggestions');
  menu.hidden = true;
  input.setAttribute('aria-controls', menu.id);
  const accept = document.createElement('button');
  accept.type = 'button';
  accept.textContent = editing ? '✓' : '+';
  accept.disabled = !editing;
  accept.title = editing ? 'Apply edit' : 'Insert tag';
  accept.setAttribute('aria-label', accept.title);
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = '×';
  cancel.title = editing ? 'Cancel edit' : 'Cancel insertion';
  cancel.setAttribute('aria-label', cancel.title);
  input.oninput = () => {
    accept.disabled = !input.value.trim() || input.value.includes(',');
    scheduleEditorSuggestions(input, menu);
  };
  input.onkeydown = (event) => {
    if (event.key === 'ArrowDown' && !menu.hidden) {
      event.preventDefault();
      selectEditorSuggestion(menu, menu._index + 1);
    } else if (event.key === 'ArrowUp' && !menu.hidden) {
      event.preventDefault();
      selectEditorSuggestion(menu, menu._index - 1);
    } else if ((event.key === 'Enter' || event.key === 'Tab') && !menu.hidden && menu._index >= 0) {
      event.preventDefault();
      commitTagEdit(menu._suggestions[menu._index].text);
    } else if (event.key === 'Enter' && !accept.disabled) {
      event.preventDefault();
      commitTagEdit(input.value);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      closeTagEditor();
    }
  };
  accept.onclick = () => commitTagEdit(input.value);
  cancel.onclick = closeTagEditor;
  editor.append(input, accept, cancel, menu);
  setTimeout(() => { input.focus(); if (editing) input.select(); }, 0);
  return editor;
}

// Opening either editor drops a validation in flight: its render would rebuild
// the tray, and the editor with it, under the user's typing.
function openTagInsert(index) {
  clearTimeout(tagValidationTimer);
  tagValidationEpoch += 1;
  tagInsertAt = index;
  tagEditAt = null;
  renderTagValidation(
    lastTagValidationResults, lastTagValidationState, lastTagValidationDetail,
  );
}

function openTagEdit(index) {
  clearTimeout(tagValidationTimer);
  tagValidationEpoch += 1;
  tagInsertAt = null;
  tagEditAt = index;
  renderTagValidation(
    lastTagValidationResults, lastTagValidationState, lastTagValidationDetail,
  );
}

// The insertion index a point in the tray names: the nearest row of chips,
// then the first chip whose middle is right of the point. Asked for a gap
// (`gap`, a plain click and the pointer's marker), a point outside the rows -
// above, below or between them past half the row gap, before a row's first
// chip, or after the last row's + tag - names none: the label, the pinned hint
// and the foot under the chips are text, and a click on "click a gap or + tag
// to insert" opened an editor in the middle of the prompt. A right-click
// still takes the nearest place.
function tagInsertionIndexAtPoint(x, y, { gap = false } = {}) {
  const entries = [...els.tagValidation.querySelectorAll('.tag-check')].map((chip) => ({
    index: Number(chip.dataset.index),
    rect: chip.getBoundingClientRect(),
  }));
  if (!entries.length) return gap ? null : 0;

  const rows = [];
  for (const entry of entries) {
    let row = rows.find((candidate) => Math.abs(candidate.top - entry.rect.top) < 3);
    if (!row) {
      row = { top: entry.rect.top, bottom: entry.rect.bottom, entries: [] };
      rows.push(row);
    }
    row.bottom = Math.max(row.bottom, entry.rect.bottom);
    row.entries.push(entry);
  }
  const distance = (row) => y < row.top ? row.top - y : y > row.bottom ? y - row.bottom : 0;
  const last = rows[rows.length - 1];
  const row = rows.sort((a, b) => distance(a) - distance(b))[0];
  row.entries.sort((a, b) => a.rect.left - b.rect.left);
  if (gap) {
    const style = getComputedStyle(els.tagValidation);
    const columnGap = parseFloat(style.columnGap) || 0;
    const add = els.tagValidation.querySelector('.tag-add')?.getBoundingClientRect();
    const end = row === last ? Math.max(row.entries[row.entries.length - 1].rect.right, add?.right ?? 0) : Infinity;
    if (distance(row) > (parseFloat(style.rowGap) || 0) / 2
        || x < row.entries[0].rect.left - columnGap || x > end + columnGap) return null;
  }
  for (const entry of row.entries) {
    if (x < entry.rect.left + entry.rect.width / 2) return entry.index;
  }
  return row.entries[row.entries.length - 1].index + 1;
}
