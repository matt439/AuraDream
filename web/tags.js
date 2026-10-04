// Local Diffusion's page script: the prompt tag helper, autocomplete and validation chips. Declarations only; state.js has
// the load-order contract these files share.
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
// An Example's write over a prompt of the user's: the note's Undo puts it
// back while the box still holds what replaced it ({after, undo}).
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
// (gallery.js's rewriteBox): the box rewritten the way typing writes it, so
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

// The note under the prompt after a write the user did not type over their
// own words - an Example picked over a prompt in progress - with an Undo that
// puts it back. It goes once the box holds anything else, `undo` being about
// the text it replaced, and after a while unless its Undo has focus: it sits
// over the next field's caption. Ctrl+Z in the box does the same for longer.
// One Undo on offer at a time: the write that brings this one takes a
// Reuse's (gallery.js's syncReuseUndo), and a Reuse takes this.
function offerPromptUndo(message, undo) {
  promptUndo = { after: els.prompt.value, undo };
  els.promptNote.hidden = false;
  els.promptNoteText.textContent = message;
  clearTimeout(promptUndoTimer);
  promptUndoTimer = setTimeout(() => {
    if (document.activeElement !== els.promptNoteUndo) dropPromptUndo();
  }, 15000);
}

function dropPromptUndo() {
  clearTimeout(promptUndoTimer);
  promptUndo = null;
  els.promptNote.hidden = true;
}

function syncPromptUndo() {
  if (promptUndo && els.prompt.value !== promptUndo.after) dropPromptUndo();
}

function undoPromptWrite() {
  const pending = promptUndo;
  dropPromptUndo();
  if (pending && els.prompt.value === pending.after) pending.undo();
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

function currentTagRange() {
  const text = els.prompt.value;
  const caret = els.prompt.selectionStart ?? text.length;
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
    if (active) item.scrollIntoView({ block: 'nearest' });
  });
}

function renderEditorSuggestions(input, menu, suggestions) {
  menu._suggestions = suggestions;
  menu._index = -1;
  menu.replaceChildren();
  for (const suggestion of suggestions) {
    const item = document.createElement('li');
    item.setAttribute('role', 'option');
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
    item.onclick = () => commitTagEdit(suggestion.text);
    menu.appendChild(item);
  }
  menu.hidden = suggestions.length === 0;
  input.setAttribute('aria-expanded', String(suggestions.length > 0));
  const editor = menu.parentElement.getBoundingClientRect();
  menu.classList.toggle('end', editor.left + 280 > window.innerWidth - 12);
  // The prompt box's rule: "lo" then Tab in the editor wrote `low quality`.
  if (suggestions.length && preselects(input.value, suggestions[0])) selectEditorSuggestion(menu, 0);
}

function scheduleEditorSuggestions(input, menu) {
  clearTimeout(tagEditorSuggestionTimer);
  const epoch = ++tagEditorSuggestionEpoch;
  const query = input.value.trim();
  renderEditorSuggestions(input, menu, []);
  if (!query) return;
  tagEditorSuggestionTimer = setTimeout(async () => {
    try {
      const body = await api(`/api/tags/suggest?q=${encodeURIComponent(query)}`);
      if (epoch !== tagEditorSuggestionEpoch || !input.isConnected) return;
      renderEditorSuggestions(input, menu, body.suggestions || []);
    } catch {
      if (epoch === tagEditorSuggestionEpoch) renderEditorSuggestions(input, menu, []);
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

// --------------------------------------------------------------- the cut

// Where the encoders stop reading the prompt box, in its own units, with the
// encoders that stop there: from the tray's measure of this text, else a
// Check of it, else the last measure while the text up to its cut is
// unchanged - a tag typed at the end of a long list is past the cut before
// the next count lands. Null where nothing counted the text, or nothing that
// feeds the image's words cuts it (FLUX.1's CLIP-L feeds one pooled vector).
function tagCut() {
  const raw = els.prompt.value;
  const text = raw.trim();
  const measured = tagMeasured?.model === loadedModel ? tagMeasured : null;
  let counted = null;
  if (measured?.text === text) counted = measured;
  else if (analysisResult?.source === text) counted = { text, encoders: analysisResult.body.encoders };
  else if (measured) counted = measured;
  const cuts = (counted?.encoders ?? [])
    .filter((report) => report.field === 'prompt' && report.cut != null && !report.pooled_only);
  if (!cuts.length) return null;
  const at = Math.min(...cuts.map((report) => report.cut));
  const prefix = counted.text.slice(0, units(counted.text, at));
  if (!text.startsWith(prefix)) return null;
  const stop = cuts.filter((report) => report.cut === at);
  return {
    offset: raw.length - raw.trimStart().length + prefix.length,
    names: stop.map((report) => report.name).join(' and '),
    budget: stop[0].budget,
  };
}

// The count the label shows: the tightest encoder of the last measure, or
// "cut unknown" where an encoder went uncounted and none is known to cut -
// a tokenizer still loading is never "no cut".
function tagTokenLine() {
  if (!tagMeasured || tagMeasured.model !== loadedModel) return null;
  const reports = tagMeasured.encoders.filter((report) => report.field === 'prompt' && !report.pooled_only);
  const cut = reports.some((report) => report.cut != null);
  if (!reports.length || (tagMeasured.detail && !cut)) return { text: 'cut unknown', over: false };
  const tight = reports.reduce((a, b) => (b.tokens / b.budget > a.tokens / a.budget ? b : a));
  return { text: `${tight.tokens}/${tight.budget} tokens`, over: cut };
}

function scheduleTagMeasure(delay) {
  clearTimeout(tagMeasureTimer);
  const epoch = ++tagMeasureEpoch;
  const text = els.prompt.value.trim();
  if (!text || (tagMeasured?.text === text && tagMeasured.model === loadedModel)) return;
  tagMeasureTimer = setTimeout(async () => {
    const model = loadedModel;
    try {
      const body = await api('/api/prompt/measure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text }),
      });
      if (epoch !== tagMeasureEpoch) return;
      tagMeasured = { model, text, encoders: body.encoders || [], detail: body.encoders_detail || null };
    } catch (error) {
      if (epoch !== tagMeasureEpoch) return;
      tagMeasured = { model, text, encoders: [], detail: error.message };
    }
    // An open editor is not rebuilt under the user's typing: the label
    // changes now, the chips with the editor's close.
    if (tagInsertAt !== null || tagEditAt !== null) syncTagTray(lastTagValidationResults, lastTagValidationState);
    else if (!els.tagValidation.hidden) {
      renderTagValidation(lastTagValidationResults, lastTagValidationState, lastTagValidationDetail);
    }
  }, delay === 0 ? 0 : 600);
}

// ---------------------------------------------------------------- the chips

// The tray's label, its title and its state: what the lookup said (checked,
// still checking, the index unreachable - amber, never red, since nothing is
// invalid for it), how many were unknown - not the item still being typed -
// the lookup's cap, and the count.
function syncTagTray(results, state, detail = '', typing = null) {
  const unknown = results.filter((result, index) => result.valid === false && index !== typing).length;
  const label = [state === 'offline' ? 'Prompt tags · Danbooru unreachable, not checked' : 'Prompt tags · Danbooru'];
  if (state !== 'offline' && unknown) label.push(`${unknown} unknown`);
  if (state === 'pending') label.push('checking…');
  if (state !== 'offline' && results.length > TAG_LOOKUP_CAP) {
    label.push(`first ${TAG_LOOKUP_CAP} of ${results.length} checked`);
  }
  const tokens = tagTokenLine();
  if (tokens) label.push(tokens.text);
  els.tagValidation.dataset.label = label.join(' · ');
  els.tagValidation.dataset.state = state;
  els.tagValidation.toggleAttribute('data-over', !!tokens?.over);
  els.tagValidation.title = [
    state === 'pending' ? 'Checking Danbooru…'
      : state === 'offline'
        ? `Danbooru unreachable; unchecked tags are not treated as invalid${detail ? `: ${detail}` : ''}`
        : 'Each comma-separated item checked against Danbooru or checkpoint guide tags',
    'edge colour: blue general, green character, purple copyright, red artist, yellow meta, teal guide tag',
    'click or right-click a gap to insert there',
  ].join(' · ');
}

// The shared tip for an unknown chip: why, and that nothing is blocked. One
// element outside the tray, so a chip's text stays the tag's own.
function showTagTip(chip, lines) {
  const tip = els.tagTip;
  tip.textContent = lines.join('\n');
  tip.hidden = false;
  const box = chip.getBoundingClientRect();
  const below = box.bottom + 6 + tip.offsetHeight <= innerHeight - 8;
  tip.style.top = `${below ? box.bottom + 6 : box.top - 6 - tip.offsetHeight}px`;
  tip.style.left = `${Math.max(8, Math.min(box.left, innerWidth - tip.offsetWidth - 8))}px`;
}

function hideTagTip() {
  els.tagTip.hidden = true;
}

// The comma item being typed: the prompt's `input` says whether it was
// typing - a keystroke or a deletion, not a whole-box write, whose inserted
// text is the box - and the caret says which item.
function noteTagTyping(event) {
  clearTimeout(tagTypingTimer);
  const typed = event instanceof InputEvent
    && /^(insertText|insertCompositionText|delete)/.test(event.inputType)
    && !(event.data && event.data.length > 1 && event.data === els.prompt.value);
  const index = typed ? caretTagIndex() : null;
  tagTyping = index === null ? null : { index, tag: promptTagParts()[index] };
  if (tagTyping) tagTypingTimer = setTimeout(endTagTyping, 1500);
}

function endTagTyping() {
  clearTimeout(tagTypingTimer);
  const typing = tagTyping;
  tagTyping = null;
  if (!typing || tagInsertAt !== null || tagEditAt !== null || els.tagValidation.hidden) return;
  if (lastTagValidationResults[typing.index]?.valid === false) {
    renderTagValidation(lastTagValidationResults, lastTagValidationState, lastTagValidationDetail);
  }
}

// The caret moved by a click or a key: out of the item, it is typed.
function followTagCaret() {
  if (tagTyping && caretTagIndex() !== tagTyping.index) endTagTyping();
}

function renderTagValidation(results, state = 'checked', detail = '') {
  lastTagValidationResults = results.map((result) => ({ ...result }));
  lastTagValidationState = state;
  lastTagValidationDetail = detail;
  if (tagInsertAt !== null && tagInsertAt > results.length) tagInsertAt = results.length;
  if (tagEditAt !== null && tagEditAt >= results.length) tagEditAt = null;
  // Read only where the chips are the prompt's own items: a render of other
  // results (a smoke's) has no place in the box to read from.
  const spans = promptTagSpans();
  const cut = spans.length === results.length ? tagCut() : null;
  const typing = tagTyping && document.activeElement === els.prompt
    && results[tagTyping.index]?.tag === tagTyping.tag ? tagTyping.index : null;
  let focus = tagFocusIndex;
  if (typeof focus === 'number') focus = Math.min(focus, results.length - 1);
  const roving = [focus, ...results.keys()]
    .find((index) => typeof index === 'number' && index >= 0 && index !== tagEditAt);
  const seen = new Set();
  let past = 0;
  let unread = false;
  hideTagTip();
  // A rebuild is not the user leaving the tray, whatever the browser says
  // about the focused chip it removes.
  tagRendering = true;
  els.tagValidation.replaceChildren();
  tagRendering = false;
  for (const [index, result] of results.entries()) {
    if (tagInsertAt === index) els.tagValidation.appendChild(buildTagEditor());
    if (tagEditAt === index) {
      els.tagValidation.appendChild(buildTagEditor(result.tag));
      continue;
    }
    const chip = document.createElement('span');
    chip.className = 'tag-check';
    chip.draggable = true;
    chip.tabIndex = index === roving ? 0 : -1;
    chip.dataset.index = String(index);
    const label = document.createElement('span');
    label.className = 'tag-check-label';
    const mark = document.createElement('span');
    mark.className = 'tag-check-mark';
    const text = document.createElement('span');
    text.className = 'tag-check-text';
    text.textContent = result.tag;
    const notes = [];
    let status;
    const tip = [];
    if (result.valid === true) {
      chip.classList.add('valid');
      // The category the lookup gave, as the edge's colour: what a tag is,
      // never where it goes - the order is the guide's, per checkpoint, and
      // Check's to place.
      chip.dataset.category = result.source === 'model' ? 'guide' : result.category || 'general';
      mark.textContent = '✓';
      status = `known ${chip.dataset.category} tag`;
      notes.push(result.source === 'model'
        ? 'Checkpoint guide tag'
        : `Danbooru ${result.category} tag · ${humanCount(result.post_count)} posts`
          + (result.canonical !== result.tag ? ` · canonical: ${result.canonical}` : ''));
    } else if (result.valid === false && index === typing) {
      chip.classList.add('typing');
      status = 'being typed';
      notes.push('Being typed: checked once you move on');
    } else if (result.valid === false) {
      chip.classList.add('unknown');
      // A tag the checkpoint's cutoff came before is marked apart from one
      // the index never had: the fixes differ. Red either way.
      mark.textContent = result.code === 'after_cutoff' ? '⏲' : '?';
      status = result.code === 'after_cutoff' ? 'unknown, too new for this checkpoint' : 'unknown';
      tip.push(`${result.reason || 'not found in Danbooru'}; generation is still allowed.`, 'Double-click to fix it.');
    } else {
      chip.classList.add('unchecked');
      // "…" while the lookup is on its way, "–" where it will not come: the
      // index unreachable, or past the lookup's cap.
      const waiting = state === 'pending' && index < TAG_LOOKUP_CAP && !result.reason;
      mark.textContent = waiting ? '…' : '–';
      status = waiting ? 'not checked yet' : 'not checked';
      notes.push(waiting ? 'Checking Danbooru…' : result.reason || detail || 'Danbooru lookup unavailable');
    }
    if (cut && spans[index].start >= cut.offset) {
      chip.classList.add('past-cut');
      if (!unread) chip.classList.add('cut-start');
      unread = true;
      past += 1;
      notes.push(`not read: ${cut.names} stop at ${cut.budget} tokens`);
      status += ', not read';
    } else if (cut && spans[index].end > cut.offset) {
      chip.classList.add('part-cut');
      notes.push(`partly read: ${cut.names} stop at ${cut.budget} tokens inside it`);
      status += ', partly read';
    }
    const key = tagKey(result.canonical ?? result.tag);
    if (seen.has(key)) {
      chip.classList.add('duplicate');
      notes.push('repeated: adds nothing (heuristic)');
      status += ', repeated';
    }
    seen.add(key);
    label.append(mark, text);
    if (tip.length) {
      // The tip, not a title: a native tooltip would stack on it.
      tip.push(...notes);
      chip.setAttribute('aria-describedby', 'tag-tip');
      chip.onpointerenter = () => showTagTip(chip, tip);
      chip.onpointerleave = hideTagTip;
      chip.addEventListener('focus', () => showTagTip(chip, tip));
      chip.addEventListener('blur', hideTagTip);
    } else {
      chip.title = [...notes, 'double-click edits · drag or Alt+←/→ moves · right-click inserts after it · Delete removes']
        .join(' · ');
    }
    chip.setAttribute('aria-label',
      `${result.tag}, ${status}. Enter edits, Alt+Left or Right moves, the Menu key inserts after it, Delete removes.`);
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'tag-remove';
    remove.textContent = '×';
    remove.draggable = false;
    // Delete on the chip does the same: one tab stop per chip, not two.
    remove.tabIndex = -1;
    remove.title = `Delete ${result.tag}`;
    remove.setAttribute('aria-label', `Delete ${result.tag}`);
    remove.onclick = (event) => { event.stopPropagation(); deletePromptTag(index); };
    chip.append(label, remove);
    chip.addEventListener('focusin', () => {
      tagFocusIndex = index;
      for (const other of els.tagValidation.querySelectorAll('.tag-check')) other.tabIndex = other === chip ? 0 : -1;
    });
    chip.ondragstart = (event) => {
      draggedTagIndex = index;
      chip.classList.add('dragging');
      hideTagTip();
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', String(index));
      }
    };
    chip.ondragover = (event) => {
      if (draggedTagIndex === null) return;
      event.preventDefault();
      clearTagDropMarkers();
      const after = event.clientX >= chip.getBoundingClientRect().left + chip.offsetWidth / 2;
      chip.classList.add(after ? 'drop-after' : 'drop-before');
    };
    chip.ondrop = (event) => {
      if (draggedTagIndex === null) return;
      event.preventDefault();
      const after = event.clientX >= chip.getBoundingClientRect().left + chip.offsetWidth / 2;
      const insertion = index + (after ? 1 : 0);
      const from = draggedTagIndex;
      draggedTagIndex = null;
      clearTagDropMarkers();
      dropPromptTag(from, insertion);
    };
    chip.ondragend = () => {
      draggedTagIndex = null;
      chip.classList.remove('dragging');
      clearTagDropMarkers();
    };
    chip.oncontextmenu = (event) => {
      event.preventDefault();
      event.stopPropagation();
      openTagInsert(index + 1);
    };
    chip.ondblclick = (event) => {
      // A fast double click on × deletes one chip and lands on the next; that
      // is not a request to edit it.
      if (event.target.closest('.tag-remove')) return;
      event.preventDefault();
      openTagEdit(index);
    };
    // The tray is one tab stop: arrows, Home and End walk the chips (Alt
    // with an arrow moves the tag instead), the rest act on the one focused.
    chip.onkeydown = (event) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        deletePromptTag(index);
      } else if (event.key === 'Enter' || event.key === 'F2') {
        event.preventDefault();
        openTagEdit(index);
      } else if (event.altKey && event.key === 'ArrowLeft') {
        event.preventDefault();
        movePromptTag(index, index - 1);
      } else if (event.altKey && event.key === 'ArrowRight') {
        event.preventDefault();
        movePromptTag(index, index + 1);
      } else if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const chips = [...els.tagValidation.querySelectorAll('.tag-check')];
        const at = chips.indexOf(chip);
        const next = { ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: chips.length - 1 }[event.key];
        chips[Math.max(0, Math.min(chips.length - 1, next))].focus();
      }
    };
    els.tagValidation.appendChild(chip);
  }
  if (tagInsertAt === results.length) els.tagValidation.appendChild(buildTagEditor());
  else if (results.length) {
    // The visible way in for a pointer that does not right-click and for the
    // keyboard: a tag at the end. A gap takes a click or a right-click.
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'tag-add';
    add.textContent = '+ tag';
    add.title = 'Insert a tag at the end; click or right-click a gap to insert there';
    add.setAttribute('aria-label', 'Insert a tag at the end of the prompt');
    add.onclick = () => openTagInsert(results.length);
    add.addEventListener('focusin', () => { tagFocusIndex = 'add'; });
    els.tagValidation.appendChild(add);
  }
  syncTagTray(results, state, detail, typing);
  els.tagValidation.hidden = results.length === 0;
  // Read out once a lookup has answered, and only when it says something
  // new: the tray itself is no live region, since it is rebuilt at every
  // pause in typing and holds every chip.
  if (state !== 'pending' && results.length) {
    const known = results.filter((result) => result.valid === true).length;
    // Not the item still being typed, as the label does not count it: a
    // lookup landing mid-word would read out an unfinished word as unknown.
    const unknown = results.filter((result, index) => result.valid === false && index !== typing).length;
    const summary = (state === 'offline'
      ? `${results.length} prompt tags, not checked: Danbooru unreachable`
      : `${results.length} prompt tags: ${known} known${unknown ? `, ${unknown} unknown` : ''}`)
      + (past ? `, ${past} past the encoders' token limit` : '');
    if (els.tagSummary.textContent !== summary) els.tagSummary.textContent = summary;
  }
  const target = focus === 'add'
    ? els.tagValidation.querySelector('.tag-add')
    : typeof focus === 'number' && focus >= 0
      ? els.tagValidation.querySelector(`.tag-check[data-index="${focus}"]`)
      : null;
  if (target && document.activeElement !== target && (tagRefocus || document.activeElement === document.body)) {
    target.focus({ preventScroll: true });
  }
  tagRefocus = false;
}

function uncheckedTag(tag, index) {
  return index < TAG_LOOKUP_CAP
    ? { tag, valid: null }
    : { tag, valid: null, reason: `past the ${TAG_LOOKUP_CAP}-item lookup cap` };
}

function tagVerdictCache() {
  if (tagVerdicts.model !== loadedModel) tagVerdicts = { model: loadedModel, results: new Map() };
  return tagVerdicts.results;
}

function rememberTagVerdicts(results) {
  const cache = tagVerdictCache();
  for (const result of results) {
    if (result.valid === null) continue;
    const key = tagKey(result.tag);
    cache.delete(key);
    cache.set(key, result);
    if (cache.size > 2000) cache.delete(cache.keys().next().value);
  }
}

async function validatePromptTags(tags, epoch) {
  const rest = tags.slice(TAG_LOOKUP_CAP).map((tag) => uncheckedTag(tag, TAG_LOOKUP_CAP));
  try {
    const body = await api('/api/tags/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags: tags.slice(0, TAG_LOOKUP_CAP) }),
    });
    if (epoch !== tagValidationEpoch) return;
    rememberTagVerdicts(body.results);
    renderTagValidation([...body.results, ...rest], body.available ? 'checked' : 'offline', body.detail || '');
  } catch (error) {
    if (epoch !== tagValidationEpoch) return;
    renderTagValidation(tags.map(uncheckedTag), 'offline', error.message);
  }
}

// Every prompt change from outside the editor comes through here - the
// textarea, a drop, a restored panel - and closes it: its index no longer
// names the item it opened on.
function scheduleTagValidation(delay = 320) {
  clearTimeout(tagValidationTimer);
  const epoch = ++tagValidationEpoch;
  tagInsertAt = null;
  tagEditAt = null;
  syncPromptUndo();
  const hybridSentence = tagHelper?.hybrid_prompt && !els.prompt.value.includes(',');
  const tags = tagVocabularyEnabled && !hybridSentence ? promptTagParts() : [];
  if (!tags.length) {
    clearTimeout(tagMeasureTimer);
    tagMeasureEpoch += 1;
    tagFocusIndex = null;
    hideTagTip();
    els.tagValidation.hidden = true;
    els.tagValidation.replaceChildren();
    return;
  }
  // What the index said before shows at once; only what it has not seen
  // waits for the answer.
  const cache = tagVerdictCache();
  const results = tags.map((tag, index) => {
    const known = index < TAG_LOOKUP_CAP && cache.get(tagKey(tag));
    return known ? { ...known, tag } : uncheckedTag(tag, index);
  });
  const waiting = results.some((result, index) => result.valid === null && index < TAG_LOOKUP_CAP);
  renderTagValidation(results, waiting ? 'pending' : 'checked');
  tagValidationTimer = setTimeout(() => validatePromptTags(tags, epoch), delay);
  scheduleTagMeasure(delay);
}

function syncTagVocabulary() {
  tagVocabularyEnabled = tagHelper?.vocabulary === 'danbooru';
  els.prompt.spellcheck = !tagVocabularyEnabled || Boolean(tagHelper?.hybrid_prompt);
  if (!tagVocabularyEnabled) hideTagSuggestions();
  scheduleTagValidation(0);
}

// The chips and the prompt check go together, in that order: in the results
// column on a two-column page, where their height cannot move Generate in the
// sticky settings column - above the results, except that from 1800 px
// style.css docks an open check beside them (the chips and the viewer on the
// left, the check on the right; CSS only, this DOM order unchanged) - and back
// under the prompt at 900 px and below, where the page is one scrolling
// column anyway. The move keeps focus where it was - a
// window crossing 901 px under someone typing in Your version or on a chip
// otherwise sent it to <body> with the next keystroke: moveBefore keeps it
// whole where the browser has it, and elsewhere the focus and the selection
// are put back by hand.
function placeTagValidation() {
  const host = window.matchMedia('(min-width: 901px)').matches
    ? els.tagValidationWide
    : els.tagValidationHome;
  const focused = document.activeElement;
  const range = typeof focused?.selectionStart === 'number'
    ? [focused.selectionStart, focused.selectionEnd, focused.selectionDirection]
    : null;
  for (const node of [els.tagValidation, els.promptAnalysis]) {
    if (node.parentElement === host) continue;
    if (typeof host.moveBefore === 'function') host.moveBefore(node, null);
    else host.appendChild(node);
  }
  if (!focused || focused === document.activeElement || !focused.isConnected) return;
  focused.focus();
  if (range) focused.setSelectionRange(...range);
}

function tagCursorIndex() {
  const before = els.prompt.value.slice(0, els.prompt.selectionStart ?? els.prompt.value.length);
  return before.trim() ? before.split(',').length : 0;
}

// Every chip and Tags-helper edit writes the whole list back here, through
// `writeBox`, so Ctrl+Z restores what the box held. That takes focus to the
// prompt; it goes back to the chip `focusIndex` names, or to the helper's
// button that made the write. A chip's visit to the box is only in passing,
// so neither the panel nor the page moves to show it - a panel held by its
// bottom jumped 62 px at each Alt+Arrow in the other column - which is the
// press's exemption from `revealInPanel`, and no scroll on the focus.
function writePromptTags(parts, caret = null, focusIndex = null) {
  const from = document.activeElement;
  const passing = focusIndex !== null && focusIndex >= 0;
  const press = panelPos.press;
  if (passing) {
    panelPos.press = true;
    els.prompt.focus({ preventScroll: true });
  }
  try {
    writeBox(els.prompt, parts.join(', '));
  } finally {
    panelPos.press = press;
  }
  promptReplaced();
  const end = caret == null ? els.prompt.value.length : Math.min(caret, els.prompt.value.length);
  els.prompt.setSelectionRange(end, end);
  if (focusIndex !== null && focusIndex >= 0) [tagFocusIndex, tagRefocus] = [focusIndex, true];
  syncTagSelection();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  if (focusIndex === null && from?.isConnected && els.tagsMenu.contains(from)) from.focus({ preventScroll: true });
}

// Buttons are toggles. Exclusive groups replace an existing person-count or
// rating tag in place; otherwise insertion follows the textarea caret. Nothing
// silently reorders free-form character, series, pose or scene tags.
function togglePromptTag(tag, group) {
  let parts = promptTagParts();
  const wanted = tag.toLowerCase();
  const active = parts.findIndex((part) => part.toLowerCase() === wanted);
  if (active >= 0) {
    parts.splice(active, 1);
    writePromptTags(parts);
    return;
  }

  let insertAt = Math.min(tagCursorIndex(), parts.length);
  if (group.exclusive) {
    const choices = new Set(group.tags.map((choice) => choice.toLowerCase()));
    const old = parts.findIndex((part) => choices.has(part.toLowerCase()));
    parts = parts.filter((part) => !choices.has(part.toLowerCase()));
    if (old >= 0) insertAt = Math.min(old, parts.length);
  }
  parts.splice(insertAt, 0, tag);
  const caret = parts.slice(0, insertAt + 1).join(', ').length;
  writePromptTags(parts, caret);
}

function setYearTag(year) {
  let parts = promptTagParts();
  let insertAt = parts.findIndex((part) => /^year \d{4}$/i.test(part));
  if (insertAt < 0) insertAt = Math.min(tagCursorIndex(), parts.length);
  parts = parts.filter((part) => !/^year \d{4}$/i.test(part));
  insertAt = Math.min(insertAt, parts.length);
  parts.splice(insertAt, 0, `year ${year}`);
  writePromptTags(parts, parts.slice(0, insertAt + 1).join(', ').length);
}

function finishWithTags(suffix) {
  const ending = new Set(suffix.map((tag) => tag.toLowerCase()));
  const parts = promptTagParts().filter((part) => !ending.has(part.toLowerCase()));
  writePromptTags([...parts, ...suffix]);
}

// Where the next tag goes, named: the helper inserts at the prompt's caret,
// which it used to hide, and the usual caret - the end, after the quality
// tags - is not where most of its groups belong. Placing is Check's: its Use
// reordered puts the tags in the documented order. The page has no order of
// its own.
function tagInsertionNote() {
  const parts = promptTagParts();
  const at = Math.min(tagCursorIndex(), parts.length);
  const where = !parts.length ? 'A tag goes into the empty prompt'
    : at === 0 ? `A tag goes before "${parts[0]}", at the start`
      : at >= parts.length ? `A tag goes after "${parts[at - 1]}", at the end`
        : `A tag goes after "${parts[at - 1]}", before "${parts[at]}"`;
  return `${where} - click in the prompt to move its caret.`;
}

// The rest of how the buttons write, which stays put.
function tagHelperNote() {
  const exclusive = (tagHelper?.groups ?? []).filter((group) => group.exclusive)
    .map((group) => group.label.toLowerCase());
  const replaces = exclusive.length
    ? `${exclusive.join(' and ').replace(/^./, (c) => c.toUpperCase())} choices replace their current tag in place. `
    : '';
  return `${replaces}Select an active tag to remove it. Check's Use reordered puts the tags in the recommended order.`;
}

function syncTagSelection() {
  if (!els.tagsMenu || els.tagsMenu.hidden) return;
  const active = new Set(promptTagParts().map((tag) => tag.toLowerCase()));
  for (const button of els.tagsMenu.querySelectorAll('[data-tag]')) {
    const on = active.has(button.dataset.tag.toLowerCase());
    button.classList.toggle('active', on);
    button.setAttribute('aria-pressed', String(on));
  }
  const note = els.tagsMenu.querySelector('.tag-insert');
  if (note) note.textContent = tagInsertionNote();
}

function buildTagHelper() {
  els.tagsMenu.replaceChildren();
  if (!tagHelper) return;

  const order = document.createElement('p');
  order.className = 'tag-order';
  order.textContent = `Recommended order: ${tagHelper.order.join(' → ')}`;
  els.tagsMenu.appendChild(order);
  const insert = document.createElement('p');
  insert.className = 'tag-insert';
  els.tagsMenu.appendChild(insert);

  for (const group of tagHelper.groups) {
    const section = document.createElement('section');
    section.className = 'tag-group';
    const heading = document.createElement('strong');
    heading.textContent = group.label;
    const options = document.createElement('div');
    options.className = 'tag-options';
    for (const tag of group.tags) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'tag-option';
      button.textContent = tag;
      button.dataset.tag = tag;
      button.onclick = () => togglePromptTag(tag, group);
      options.appendChild(button);
    }
    section.append(heading, options);
    els.tagsMenu.appendChild(section);
  }

  if (tagHelper.year_tag) {
    const section = document.createElement('section');
    section.className = 'tag-group';
    const heading = document.createElement('strong');
    heading.textContent = 'Year';
    const row = document.createElement('div');
    row.className = 'tag-year';
    const input = document.createElement('input');
    input.type = 'text';
    input.inputMode = 'numeric';
    input.maxLength = 4;
    input.placeholder = '2023';
    input.setAttribute('aria-label', 'Four digit year tag');
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'ghost tiny';
    add.textContent = 'Insert year';
    add.disabled = true;
    input.oninput = () => { add.disabled = !/^\d{4}$/.test(input.value); };
    add.onclick = () => setYearTag(input.value);
    row.append(input, add);
    section.append(heading, row);
    els.tagsMenu.appendChild(section);
  }

  const actions = document.createElement('div');
  actions.className = 'tag-actions';
  if (tagHelper.recommended_suffix?.length) {
    const finish = document.createElement('button');
    finish.type = 'button';
    finish.className = 'ghost tiny';
    finish.dataset.action = 'recommended-suffix';
    finish.textContent = `Finish with ${tagHelper.recommended_suffix.join(', ')}`;
    finish.onclick = () => finishWithTags(tagHelper.recommended_suffix);
    actions.appendChild(finish);
  }
  if (tagHelper.negative_prompt) {
    const negative = document.createElement('button');
    negative.type = 'button';
    negative.className = 'ghost tiny';
    negative.dataset.action = 'recommended-negative';
    negative.textContent = 'Use the recommended negative prompt';
    // Written as typing writes it, so Ctrl+Z in the box brings back a
    // negative prompt of the user's own; focus stays in the helper.
    negative.onclick = () => {
      writeBox(els.negative, tagHelper.negative_prompt);
      negative.focus({ preventScroll: true });
      syncAnalysisStale();
      schedulePreview();
      persistPanel();
    };
    actions.appendChild(negative);
  }
  if (actions.children.length) els.tagsMenu.appendChild(actions);

  const note = document.createElement('p');
  note.className = 'tag-note';
  note.textContent = tagHelperNote();
  els.tagsMenu.appendChild(note);
  syncTagSelection();
}

// The helper drops under the prompt, over the fields below it, where the
// caret it inserts at stays in view and a click in the box moves it without
// closing the helper. Opened from the keyboard it takes focus - it follows
// the prompt in the page, four stops past its button - and closing hands
// focus back to the button rather than to <body>.
function showTagHelper(open, { focus = false } = {}) {
  if (open && !els.examplesMenu.hidden) showExamples(false);
  const inside = els.tagsMenu.contains(document.activeElement);
  els.tagsBtn.setAttribute('aria-expanded', String(open));
  els.tagsMenu.hidden = !open;
  if (open) {
    buildTagHelper();
    if (focus) els.tagsMenu.querySelector('button')?.focus({ preventScroll: true });
  } else if (inside) els.tagsBtn.focus();
}

function syncTagHelper() {
  const available = Boolean(tagHelper && Array.isArray(tagHelper.groups) && tagHelper.groups.length);
  els.tagHelper.hidden = !available;
  if (!available) showTagHelper(false);
  else if (!els.tagsMenu.hidden) buildTagHelper();
  syncTagVocabulary();
}
