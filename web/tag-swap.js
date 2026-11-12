// AuraDream's page script: Swap tags, a gallery selection run again with some tags swapped.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// "Redo with tags swapped": rows of comma items to find and what to put in
// their place, over the gallery's selection (gallery-select.js) where cards
// are picked, else what the filter matches - the same scope Re-render takes.
// Every image keeps its own seed, size and settings; only its prompt (or
// negative prompt) is rewritten. The server does the rewriting (tag_swap.py)
// and says, per image, what it would do: the page sends the rows, lists the
// answer and marks what was written in from the server's offsets, adding
// nothing of its own. Queue is held, with its reason, until the answer is the
// answer to what is typed, and the queueing request pins the ids the check
// resolved (`expected`), so a selection that changed in between is refused
// rather than queued unseen. A row with an empty Find is an Add row: its
// tags go in at the start or the end, as its toggle says, and the server
// places them in each prompt's own style (tag_swap.py, `_place`).

const SWAP_CHECK_DELAY = 400;
// The request's own limits (schemas.SwapTagsRequest, TagSwap): held on the
// page with the reason, never met as the server's validation text.
const SWAP_MAX_ROWS = 16;
const SWAP_MAX_TEXT = 1000;
const SWAP_STATUS_WORDS = {
  queued: 'queued', missing: 'missing a tag', unchanged: 'unchanged', rendered: 'already rendered', planned: 'already queued',
  invalid: 'invalid', source_deleted: 'source deleted', lora_missing: 'LoRA missing',
  mask_missing: 'mask missing', outpaint: 'outpaints that cannot run',
  mask_crop: 'masked areas that cannot run', duplicate: 'duplicates',
  by_hand: 'need editing by hand', structured: 'structured prompts',
  deleted: 'deleted', foreign: 'from another checkpoint',
};
const SWAP_CHIP = {
  queued: 'queued', missing: 'missing', unchanged: 'unchanged', rendered: 'already rendered', planned: 'already queued',
  invalid: 'invalid', source_deleted: 'source deleted', lora_missing: 'LoRA missing',
  mask_missing: 'mask missing', outpaint: 'outpaint refused', mask_crop: 'masked area refused',
  duplicate: 'duplicate', by_hand: 'by hand',
  structured: 'structured', deleted: 'deleted',
  foreign: 'other checkpoint',
};

// What the dialog acts on, fixed when it opens: `{ids}` or the filter. The
// modal page cannot change the selection under it; a delete that lands
// meanwhile (forgetImages) is shown as "deleted" by the next check.
let swapScope = null;
// The galleryActionTarget the dialog opened on, whose count the scope line
// says, and the gallery's count (`total`) the last check was asked under: a
// filter's matches grow as a job lands images, and the check is asked again
// (swapGalleryChanged) so a press pins what the filter holds now.
let swapScopeTarget = null;
let swapSeenTotal = null;
let swapRevision = null;
// The last check's answer and the payload it answered (as a string): Queue
// goes only while that is what the dialog says now.
let swapAnswer = null;
let swapAnswerKey = '';
// A check the server refused (a row two Finds share), said in the count line
// and held on Queue.
let swapProblem = '';
// The server refused a Queue press (the selection changed, or it is all in
// the gallery now): said in #swap-error until the next edit or press, though
// the check it asks again for may free Queue at once.
let swapRefused = false;
let swapAsking = null;
let swapTimer = null;
// A Queue press waiting on its check: a second press does nothing. Two
// waits aborted each other's check in a loop that never yielded.
let swapWaiting = false;
// The Replace boxes' unknown-tag lookup (scheduleSwapTagCheck): its timer, the
// epoch a late answer is dropped by, and the boxes' text it last answered.
let swapTagTimer = null;
let swapTagEpoch = 0;
let swapTagKey = '';

// The rows as the server's summary words them ("red hair, long hair → silver
// hair"), for the queue row until the server's own name lands.
function swapSummary(payload) {
  const rows = payload.swaps.map((row) => (row.add
    ? `+ ${swapTags(row.replace).join(', ')} (${row.add})`
    : `${row.find} → ${row.replace || '(removed)'}`)).join('; ');
  return payload.field === 'negative_prompt' ? `negative: ${rows}` : rows;
}

// An Add row's tags as `tag_swap.Row.describe` lists them: comma items,
// a tag typed twice once (by its text, near enough for a placeholder name).
function swapTags(text) {
  return [...new Set(text.split(',').map((tag) => tag.trim()).filter(Boolean))];
}

// What Swap tags would act on now (galleryActionTarget).
function swapTarget() {
  return galleryActionTarget('to choose the images to swap tags in', 'one swap');
}

// The button's state: held with the reason, or titled with what it takes.
// From syncRerender (every filter, count and size change) and every change
// of the selection.
function syncSwapButton() {
  const target = swapTarget();
  els.swap.setAttribute('aria-disabled', String(!target.scope));
  els.swap.title = target.scope
    ? `Run ${target.words} again with some tags swapped - each keeps its own seed, size and settings`
    : target.why;
  swapGalleryChanged();
}

// The gallery's count changed under an open dialog acting on the filter (a
// job landed matching images): the modal dialog holds the filter the
// gallery shows, so `total` is its count. The answer is dropped, so Queue
// reads as checking, and asked again - not a refusal at every press while
// a job keeps landing.
function swapGalleryChanged() {
  if (!els.swapDialog.open || !swapScope || swapScope.ids) return;
  if (total === swapSeenTotal) return;
  swapSeenTotal = total;
  swapAnswerKey = '';
  scheduleSwapCheck(0);
}

// The scope line, from the target the dialog opened on and, for a filter,
// the count of the latest answer.
function swapSayScope(answer) {
  if (!swapScopeTarget) return;
  if (!swapScope.ids && answer && typeof answer.total === 'number') swapScopeTarget = { ...swapScopeTarget, n: answer.total };
  const line = galleryActionScopeLine(swapScopeTarget, 'In');
  if (els.swapScope.textContent !== line) els.swapScope.textContent = line;
}

function openSwapDialog() {
  const target = swapTarget();
  if (!target.scope) { showError(target.why, true, { slot: els.galleryError }); return; }
  swapScope = target.scope;
  swapScopeTarget = target;
  swapSeenTotal = total;
  swapRevision = modelRevision;
  swapSayScope(null);
  // A checkpoint with no negative prompt has nothing to swap there.
  els.swapNegative.hidden = !negativeSupported;
  if (!negativeSupported) swapFieldInput('prompt').checked = true;
  collapseSwapRows();
  if (!els.swapRows.children.length) els.swapRows.append(swapRow());
  syncSwapRowNames();
  swapAnswer = null;
  swapAnswerKey = '';
  swapProblem = '';
  swapRefused = false;
  swapSay('');
  els.swapResults.replaceChildren();
  showError('', false, { slot: els.swapError });
  els.swapDialog.showModal();
  swapLiveRows()[0]?.querySelector('.swap-find').focus();
  scheduleSwapCheck(0);
  // Rows typed before are kept, and the checkpoint may have changed since.
  swapTagKey = '';
  scheduleSwapTagCheck(0);
}

function swapFieldInput(value) {
  return els.swapDialog.querySelector(`input[name="swap-field"][value="${value}"]`);
}

// One row: Find, Replace with, and × - which on the last row left empties
// it rather than leaving no row to type in. Between the boxes, the arrow,
// or with Find empty the Add row's start/end toggle in its place: one
// fixed-width slot, so neither box moves as Find empties or fills.
function swapRow(find = '', replace = '') {
  const row = document.createElement('div');
  row.className = 'swap-row';
  row.dataset.add = 'end';
  const findBox = document.createElement('input');
  findBox.type = 'text';
  findBox.className = 'swap-find';
  findBox.placeholder = 'Find: red hair (empty adds)';
  findBox.value = find;
  findBox.spellcheck = false;
  findBox.autocomplete = 'off';
  findBox.maxLength = SWAP_MAX_TEXT;
  swapListBox(findBox);
  const arrow = document.createElement('span');
  arrow.className = 'swap-arrow';
  arrow.setAttribute('aria-hidden', 'true');
  arrow.textContent = '→';
  const at = document.createElement('button');
  at.type = 'button';
  at.className = 'ghost tiny swap-at';
  at.onclick = () => {
    row.dataset.add = row.dataset.add === 'end' ? 'start' : 'end';
    syncSwapRowMode(row);
    swapEdited(0);
  };
  const mid = Object.assign(document.createElement('span'), { className: 'swap-mid' });
  mid.append(arrow, at);
  const replaceBox = document.createElement('input');
  replaceBox.type = 'text';
  replaceBox.className = 'swap-replace';
  replaceBox.placeholder = 'Replace with (empty removes)';
  replaceBox.value = replace;
  replaceBox.spellcheck = false;
  replaceBox.autocomplete = 'off';
  replaceBox.maxLength = SWAP_MAX_TEXT;
  swapListBox(replaceBox);
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'ghost tiny swap-remove';
  remove.textContent = '×';
  remove.onclick = () => removeSwapRow(row);
  row.append(findBox, mid, replaceBox, remove);
  syncSwapRowMode(row);
  return row;
}

// A row's Find empty or not: the arrow or the Add toggle, and what Replace
// with is for.
function syncSwapRowMode(row) {
  const adding = !row.querySelector('.swap-find').value.trim();
  const at = row.querySelector('.swap-at');
  row.querySelector('.swap-arrow').hidden = adding;
  at.hidden = !adding;
  const start = row.dataset.add === 'start';
  const words = start ? 'at start' : 'at end';
  if (at.textContent !== words) at.textContent = words;
  at.title = `Adds its tags at the ${start ? 'start' : 'end'} of each prompt; press to add at the ${start ? 'end' : 'start'}`;
  row.querySelector('.swap-replace').placeholder = adding ? 'Tags to add: night, rain' : 'Replace with (empty removes)';
}

// The rows that count: a removed row keeps its slot, empty and hidden
// (`.gone`), until the pointer leaves the rows (collapseSwapRows), so the ×
// of the row after it never slides under the pointer that pressed one - a
// double click removes one row, not two.
function swapLiveRows() {
  return [...els.swapRows.children].filter((row) => !row.classList.contains('gone'));
}

function collapseSwapRows() {
  els.swapRows.querySelectorAll('.swap-row.gone').forEach((row) => row.remove());
}

function syncSwapRowNames() {
  const rows = swapLiveRows();
  rows.forEach((row, i) => {
    const n = i + 1;
    row.querySelector('.swap-find').setAttribute('aria-label', `Find, row ${n}: comma-separated tags`);
    row.querySelector('.swap-replace').setAttribute('aria-label', `Replace with, row ${n}`);
    row.querySelector('.swap-at').setAttribute('aria-label', `Where row ${n} adds its tags`);
    const remove = row.querySelector('.swap-remove');
    const last = rows.length === 1;
    remove.setAttribute('aria-label', last ? 'Empty this row' : `Remove row ${n}`);
    remove.title = last ? 'Empty this row' : 'Remove this row';
  });
  const full = rows.length >= SWAP_MAX_ROWS;
  els.swapAdd.setAttribute('aria-disabled', String(full));
  els.swapAdd.title = full ? `At most ${SWAP_MAX_ROWS} rows in one swap` : 'Add a row of tags to find';
}

// The rows have a fixed height and scroll, so a row added never moves
// + Add row, the list or Queue; the new one scrolls into view inside them.
function addSwapRow() {
  if (swapLiveRows().length >= SWAP_MAX_ROWS) {
    showError(els.swapAdd.title, false, { slot: els.swapError });
    return;
  }
  collapseSwapRows();
  const row = swapRow();
  els.swapRows.append(row);
  syncSwapRowNames();
  swapEdited(0);
  row.querySelector('.swap-find').focus();
}

// Focus goes to the next row still there, else the one before.
function removeSwapRow(row) {
  const rows = swapLiveRows();
  if (!rows.includes(row)) return;
  if (rows.length === 1) {
    row.querySelectorAll('input').forEach((box) => { box.value = ''; });
    row.dataset.add = 'end';
    syncSwapRowMode(row);
    row.querySelector('.swap-find').focus();
  } else {
    const at = rows.indexOf(row);
    // Focus first: the row about to hide holds it.
    (rows[at + 1] ?? rows[at - 1]).querySelector('.swap-find').focus();
    row.querySelectorAll('input').forEach((box) => { box.value = ''; });
    row.classList.add('gone');
    row.setAttribute('aria-hidden', 'true');
  }
  syncSwapRowNames();
  swapEdited(0);
}

// An edit: a refusal said for the press before it is no longer the news.
function swapEdited(delay = SWAP_CHECK_DELAY) {
  if (swapRefused) {
    swapRefused = false;
    showError('', false, { slot: els.swapError });
  }
  scheduleSwapCheck(delay);
  scheduleSwapTagCheck(delay);
}

// ------------------------------------------------- the boxes' tag help

// On a tag checkpoint (tagVocabularyEnabled, as the prompt box decides) every
// Find and Replace box suggests tags for the comma item at the caret, as the
// negative box does: tag-lists.js's editor list, one for the dialog
// (#swap-tag-suggestions), placed under the box with the focus - the rows
// scroll, and a list inside one was cut off at their edge - and closed as
// anything in the dialog scrolls, the list's own scroll aside. Asked with
// `field=negative` while the swap is in the negative prompt. A pick replaces
// the item at the caret as typing would (one Ctrl+Z takes it back); Enter or
// Tab on a selected item picks it rather than pressing Queue or moving on.
function swapListBox(box) {
  box.setAttribute('aria-autocomplete', 'list');
  box.setAttribute('aria-controls', 'swap-tag-suggestions');
  box.setAttribute('aria-expanded', 'false');
}

function suggestSwapTags(event) {
  const box = event.target;
  const menu = els.swapTagSuggestions;
  if (!box.matches?.('.swap-find, .swap-replace')) return;
  if (!tagVocabularyEnabled || !els.swapDialog.open || document.activeElement !== box) {
    hideSwapSuggestions();
    return;
  }
  const range = currentTagRange(box);
  const { query } = range;
  const text = box.value;
  const model = loadedModel;
  const same = menu._input === box && !menu.hidden;
  if (event.type !== 'input' && same && menu._query === query) return;
  const negative = swapPayload().field === 'negative_prompt';
  scheduleEditorSuggestions(box, menu, {
    query,
    ...(negative ? { field: 'negative' } : {}),
    pick: (suggestion) => applySwapSuggestion(box, suggestion),
    place: placeSwapSuggestions,
    keep: same,
    current: () => loadedModel === model && tagVocabularyEnabled && els.swapDialog.open
      && document.activeElement === box && box.value === text
      && currentTagRange(box).start === range.start && currentTagRange(box).end === range.end,
  });
}

// Only a list that is up, as the negative box's (the lookup is shared).
function hideSwapSuggestions() {
  const menu = els.swapTagSuggestions;
  if (menu.hidden) return;
  clearTimeout(tagEditorSuggestionTimer);
  tagEditorSuggestionEpoch += 1;
  const box = menu._input;
  renderEditorSuggestions(box, menu, [], { pick: (suggestion) => applySwapSuggestion(box, suggestion) });
}

// Under the box, or over it where the window has more room above than the
// list needs below. Fixed to the window: the list is the dialog's, outside
// the rows' scroll, at the box's own left and width (280 px at the least,
// inside the window).
function placeSwapSuggestions(menu) {
  if (menu.hidden || !menu._input) return;
  const box = menu._input.getBoundingClientRect();
  const viewHeight = window.visualViewport?.height ?? window.innerHeight;
  const width = Math.min(Math.max(box.width, 280), window.innerWidth - 24);
  menu.style.width = `${width}px`;
  menu.style.left = `${Math.max(12, Math.min(box.left, window.innerWidth - 12 - width))}px`;
  const below = viewHeight - box.bottom;
  const up = menu.offsetHeight + 8 > below && box.top > below;
  menu.style.top = `${up ? box.top - 4 - menu.offsetHeight : box.bottom + 4}px`;
}

function applySwapSuggestion(box, suggestion) {
  if (!box.isConnected) { hideSwapSuggestions(); return; }
  const { start, end } = currentTagRange(box);
  box.focus();
  box.setSelectionRange(start, end);
  if (!document.execCommand('insertText', false, suggestion.text)) {
    box.setRangeText(suggestion.text, start, end, 'end');
    box.dispatchEvent(new Event('input', { bubbles: true }));
  }
  hideSwapSuggestions();
}

// The Replace boxes' tags the index does not know - in a replacing row and
// an Add row alike, since either writes them into every prompt - marked on
// the box (its title naming each and why) and said in the line under the
// rows: "Not a known tag: silver_hiar - Queue still puts it in." Advisory:
// Queue is never held for it, and silent when the lookup is unavailable or
// fails, as the chips never call an unchecked tag unknown. One lookup for
// every row, the rows' Replace text joined as one prompt so contextual prose
// on a hybrid checkpoint is told from a tag (as the Refine dialog's line);
// asked a moment after the typing stops, and not again for the same text.
function scheduleSwapTagCheck(delay = SWAP_CHECK_DELAY) {
  clearTimeout(swapTagTimer);
  const epoch = ++swapTagEpoch;
  const rows = swapLiveRows().map((row) => {
    const box = row.querySelector('.swap-replace');
    // The item at the caret of the box being typed in is not called unknown
    // until it is finished, as the chips' item being typed is not: the
    // check comes again as the box loses the focus.
    let typing = -1;
    if (document.activeElement === box) {
      const { start, query } = currentTagRange(box);
      if (query) typing = promptTagSpans(box.value).findIndex((span) => span.start === start);
    }
    return { box, parts: promptTagParts(box.value), typing };
  });
  const prompt = rows.map((row) => row.parts.join(', ')).filter(Boolean).join(', ');
  const tags = promptTagParts(prompt).slice(0, TAG_LOOKUP_CAP);
  const model = loadedModel;
  const key = JSON.stringify([model, prompt, rows.map((row) => row.typing)]);
  if (!els.swapDialog.open || !tagVocabularyEnabled || !tags.length) {
    swapTagKey = key;
    markSwapTags(rows, []);
    return;
  }
  if (key === swapTagKey) return;
  swapTagTimer = setTimeout(async () => {
    let unknown = [];
    try {
      const body = await api('/api/tags/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags, prompt }),
      });
      if (epoch !== swapTagEpoch || loadedModel !== model) return;
      if (body.available) {
        unknown = body.results.map((result, index) => ({ ...result, index })).filter((result) => result.valid === false
          && !body.parts?.some((part) => part.kind === 'prose' && result.index >= part.first && result.index <= part.last));
      }
    } catch { /* unchecked is not unknown */ }
    if (epoch !== swapTagEpoch) return;
    swapTagKey = key;
    markSwapTags(rows, unknown);
  }, delay);
}

// The verdict onto the rows it was asked for: each comma item's index is
// its row's by the rows' counts, in order.
function markSwapTags(rows, unknown) {
  let first = 0;
  const said = [];
  const titles = [];
  const several = rows.filter((row) => row.parts.length).length > 1;
  rows.forEach((row) => {
    const last = first + row.parts.length;
    const mine = unknown.filter((result) => result.index >= first && result.index < last
      && result.index !== first + row.typing);
    first = last;
    const names = [...new Set(mine.map((result) => result.tag))];
    row.box.classList.toggle('swap-unknown', names.length > 0);
    const n = swapLiveRows().indexOf(row.box.closest('.swap-row')) + 1;
    const reasons = names.map((tag) => `${tag}: ${mine.find((result) => result.tag === tag).reason || 'not a known tag'}`);
    row.box.title = names.length ? `Not a known tag - ${reasons.join('; ')}. Queue still puts ${names.length === 1 ? 'it' : 'them'} in.` : '';
    for (const tag of names) said.push(several ? `${tag} (row ${n})` : tag);
    titles.push(...reasons.map((reason) => (several ? `Row ${n}, ${reason}` : reason)));
  });
  const listed = said.length > 8 ? [...said.slice(0, 8), `${said.length - 8} more`] : said;
  const line = said.length
    ? `Not a known tag: ${listed.join(', ')} - Queue still puts ${said.length === 1 ? 'it' : 'them'} in.`
    : '';
  if (els.swapTags.textContent !== line) els.swapTags.textContent = line;
  els.swapTags.title = titles.join('\n');
}

// The tag helper came or went (tag-chips.js): an open dialog asks again.
function syncSwapTagVerdicts() {
  if (!els.swapDialog.open) return;
  swapTagKey = '';
  scheduleSwapTagCheck(0);
}

function swapPayload() {
  const swaps = swapLiveRows()
    .map((row) => {
      const find = row.querySelector('.swap-find').value.trim();
      const replace = row.querySelector('.swap-replace').value.trim();
      if (find) return { find, replace };
      return replace ? { find: '', replace, add: row.dataset.add === 'start' ? 'start' : 'end' } : null;
    })
    .filter(Boolean);
  const field = els.swapDialog.querySelector('input[name="swap-field"]:checked')?.value ?? 'prompt';
  return { ...swapScope, swaps, field };
}

// After each edit, a moment after the typing stops; at once on open, on a
// row removed and on a delete. A check still out is abandoned.
function scheduleSwapCheck(delay = SWAP_CHECK_DELAY) {
  clearTimeout(swapTimer);
  swapTimer = setTimeout(() => { void runSwapCheck(); }, delay);
  syncSwapQueue();
}

async function runSwapCheck() {
  if (!els.swapDialog.open || !swapScope) return;
  const payload = swapPayload();
  const key = JSON.stringify(payload);
  swapAsking?.abort();
  swapAsking = null;
  if (!payload.swaps.length) {
    swapAnswer = null;
    swapAnswerKey = key;
    swapProblem = '';
    swapSay('Type the tags to find, comma-separated, and what to put in their place - or leave Find empty to add tags.');
    els.swapResults.replaceChildren();
    syncSwapQueue();
    return;
  }
  const asking = new AbortController();
  swapAsking = asking;
  syncSwapQueue();
  try {
    const answer = await api('/api/swap-tags/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: asking.signal,
    });
    if (swapAsking !== asking) return;
    swapAnswer = answer;
    swapProblem = '';
    renderSwapResults(answer);
  } catch (err) {
    if (swapAsking !== asking || err.name === 'AbortError') return;
    swapAnswer = null;
    swapProblem = err.message;
    swapSay(err.message, true);
    els.swapResults.replaceChildren();
  }
  swapAnswerKey = key;
  swapAsking = null;
  syncSwapQueue();
}

// The count line: two lines kept, whatever it says, its whole text the title.
function swapSay(text, failed = false) {
  if (els.swapCount.textContent !== text) els.swapCount.textContent = text;
  els.swapCount.title = text;
  els.swapCount.classList.toggle('failed', failed);
}

function renderSwapResults(answer) {
  swapSayScope(answer);
  const parts = [`${answer.queued} will be queued`];
  // In one order whatever the server's, so the line reads alike check to check.
  for (const [status, words] of Object.entries(SWAP_STATUS_WORDS)) {
    if (status !== 'queued' && answer.counts[status]) parts.push(`${answer.counts[status]} ${words}`);
  }
  if (answer.jobs > 1) parts.push(`${answer.jobs} jobs, one per size`);
  if (answer.total > answer.images.length) parts.push(`the first ${answer.images.length} of ${answer.total} listed`);
  swapSay(parts.join(' · '));
  els.swapResults.replaceChildren(...answer.images.map(swapResultRow));
}

function swapResultRow(image) {
  const { li, body, head } = resultRow(image, 'swap', SWAP_CHIP);
  if (image.width && image.height) head.append(` ${image.width}×${image.height}`);
  if (image.detail && image.status !== 'queued') {
    head.append(Object.assign(document.createElement('span'), { className: 'swap-detail batch-detail', textContent: ` ${image.detail}` }));
  }
  if (image.text) body.append(swapMarked(image.text, image.spans));
  if (image.removed?.length) body.append(swapTookOut(image.removed));
  resultNotes(body, image, 'swap');
  return li;
}

// The text, with what the swap wrote in marked at the server's offsets.
function swapMarked(text, spans) {
  const p = document.createElement('p');
  p.className = 'swap-text';
  let at = 0;
  for (const [start, end] of spans ?? []) {
    if (start < at) continue;
    p.append(text.slice(at, start), Object.assign(document.createElement('mark'), { textContent: text.slice(start, end) }));
    at = end;
  }
  p.append(text.slice(at));
  return p;
}

// What the swap took out of the original, struck through, as the server
// names it: a removal or a many-to-one row's other finds leave nothing in
// the new text to mark.
function swapTookOut(removed) {
  const p = document.createElement('p');
  p.className = 'swap-removed';
  p.append('Took out: ');
  removed.forEach((item, i) => {
    if (i) p.append(', ');
    p.append(Object.assign(document.createElement('del'), { textContent: item }));
  });
  return p;
}

// Why Queue is held now, or '' when it may go.
function swapHold() {
  const payload = swapPayload();
  if (!payload.swaps.length) return 'Add a tag to find';
  if (swapAsking || JSON.stringify(payload) !== swapAnswerKey) return 'Checking the swaps…';
  if (swapProblem) return swapProblem;
  if (swapRevision !== modelRevision) return 'The checkpoint changed; close Swap tags and open it again';
  if (!swapAnswer || !swapAnswer.queued) {
    const counts = swapAnswer?.counts ?? {};
    if (counts.planned && Object.keys(counts).every((key) => key === 'planned' || key === 'rendered')) {
      return counts.rendered ? 'Every changed image is already rendered or queued' : 'Every changed image is already queued';
    }
    return 'No image would change';
  }
  return '';
}

function syncSwapQueue() {
  if (!els.swapDialog.open) return;
  const hold = swapHold();
  const answered = swapAnswer && !hold;
  els.swapQueue.textContent = answered
    ? `Queue ${plural(swapAnswer.queued, 'image')}${swapAnswer.jobs > 1 ? ` (${swapAnswer.jobs} jobs)` : ''}`
    : 'Queue';
  els.swapQueue.setAttribute('aria-disabled', String(Boolean(hold)));
  els.swapQueue.title = hold || `Queue ${plural(swapAnswer.queued, 'image')} with the tags swapped - each keeps its own seed, size and settings${swapAnswer.jobs > 1 ? `, one job per size` : ''}`;
  // A reason said for a press is about the state it was pressed in; a
  // refusal from the server stays until the next edit or press.
  if (!hold && !swapRefused && els.swapError.textContent) showError('', false, { slot: els.swapError });
}

// Queue: waits for the check of what the dialog says, as More seeds and Mass
// refine do, then queues - a press made before the typing's check had
// landed was said as "Checking the swaps…", wiped by that check, and lost.
async function queueSwap() {
  if (!els.swapDialog.open || submitting || swapWaiting) return;
  swapRefused = false;
  while (swapPayload().swaps.length && (swapAsking || JSON.stringify(swapPayload()) !== swapAnswerKey)) {
    clearTimeout(swapTimer);
    swapWaiting = true;
    try { await runSwapCheck(); } finally { swapWaiting = false; }
    if (!els.swapDialog.open || submitting) return;
  }
  const hold = swapHold();
  if (hold) { showError(hold, false, { slot: els.swapError }); return; }
  const payload = { ...swapPayload(), expected: swapAnswer.ids };
  void startJob('/api/swap-tags', payload, {
    slot: els.swapError,
    panel: false,
    onAccepted: () => { els.swapDialog.close(); },
    // Refused - the selection changed, the gallery holds them all now: the
    // dialog stays open with the reason, and the check is asked again, the
    // old answer dropped so Queue reads as checking until it lands.
    onRefused: () => {
      swapRefused = Boolean(els.swapError.textContent);
      swapAnswer = null;
      swapAnswerKey = '';
      scheduleSwapCheck(0);
    },
  });
}

// From forgetImages, the one place the page drops deleted ids: an open
// dialog that listed one checks again, and shows it as deleted.
function forgetSwapImages(gone) {
  if (!els.swapDialog.open || !swapAnswer) return;
  if (swapAnswer.ids.some((id) => gone.has(id))) scheduleSwapCheck(0);
}

function swapBackdrop(event) {
  if (event.target !== els.swapDialog) return;
  const box = els.swapDialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right
    && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) els.swapDialog.close();
}

function setupSwap() {
  els.swap.onclick = openSwapDialog;
  els.swapAdd.onclick = addSwapRow;
  els.swapCancel.onclick = () => els.swapDialog.close();
  els.swapQueue.onclick = () => { void queueSwap(); };
  els.swapDialog.addEventListener('click', swapBackdrop);
  els.swapRows.addEventListener('input', (e) => {
    const row = e.target.closest('.swap-row');
    if (row && e.target.matches('.swap-find')) syncSwapRowMode(row);
    swapEdited();
    suggestSwapTags(e);
  });
  els.swapRows.addEventListener('click', suggestSwapTags);
  els.swapRows.addEventListener('keyup', (e) => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) suggestSwapTags(e);
  });
  // The list's keys first (tag-lists.js, editorListKeydown): Enter or Tab on
  // a selected item picks it, and an Escape the list takes closes the list
  // alone - each prevented, so Queue's Enter below and the dialog leave it.
  els.swapRows.addEventListener('keydown', (e) => {
    if (e.target === els.swapTagSuggestions._input) editorListKeydown(e, els.swapTagSuggestions, hideSwapSuggestions);
  });
  // Out of a Replace box, the item it was typing is finished: checked at once.
  els.swapRows.addEventListener('focusout', (e) => {
    hideSwapSuggestions();
    if (e.target.matches('.swap-replace')) scheduleSwapTagCheck(0);
  });
  els.swapDialog.addEventListener('scroll', (e) => { if (e.target !== els.swapTagSuggestions) hideSwapSuggestions(); }, true);
  window.addEventListener('resize', hideSwapSuggestions);
  // Out of the rows, the slots removed rows kept can go: nothing in there is
  // under the pointer, and the rows' fixed height moves nothing outside.
  els.swapRows.addEventListener('pointerleave', collapseSwapRows);
  els.swapDialog.addEventListener('change', (e) => { if (e.target.name === 'swap-field') swapEdited(0); });
  // Enter in a box is Queue's press, as in any form.
  els.swapRows.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || keyboardEventHeld(e) || keyboardCommandModifier(e) || e.shiftKey) return;
    if (e.key === 'Enter' && e.target.matches('input')) {
      e.preventDefault();
      if (!e.repeat) void queueSwap();
    }
  });
  els.swapDialog.addEventListener('close', () => {
    clearTimeout(swapTimer);
    swapAsking?.abort();
    swapAsking = null;
    clearTimeout(swapTagTimer);
    swapTagEpoch += 1;
    hideSwapSuggestions();
  });
  onSelectionChange(syncSwapButton);
  syncSwapRowNames();
  syncSwapButton();
}
