// Local Diffusion's page script: Swap tags, a gallery selection run again with some tags swapped.
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
// rather than queued unseen.

const SWAP_CHECK_DELAY = 400;
// The request's own limits (schemas.SwapTagsRequest, TagSwap): held on the
// page with the reason, never met as the server's validation text.
const SWAP_MAX_ROWS = 16;
const SWAP_MAX_TEXT = 1000;
const SWAP_STATUS_WORDS = {
  queued: 'queued', missing: 'missing a tag', unchanged: 'unchanged', rendered: 'already rendered',
  invalid: 'invalid', source_deleted: 'source deleted', lora_missing: 'LoRA missing',
  mask_missing: 'mask missing', outpaint: 'outpaints that cannot run', duplicate: 'duplicates',
  by_hand: 'need editing by hand', structured: 'structured prompts',
  deleted: 'deleted', foreign: 'from another checkpoint',
};
const SWAP_CHIP = {
  queued: 'queued', missing: 'missing', unchanged: 'unchanged', rendered: 'already rendered',
  invalid: 'invalid', source_deleted: 'source deleted', lora_missing: 'LoRA missing',
  mask_missing: 'mask missing', outpaint: 'outpaint refused', duplicate: 'duplicate', by_hand: 'by hand',
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

// The rows as the server's summary words them ("red hair, long hair → silver
// hair"), for the queue row until the server's own name lands.
function swapSummary(payload) {
  const rows = payload.swaps.map((row) => `${row.find} → ${row.replace || '(removed)'}`).join('; ');
  return payload.field === 'negative_prompt' ? `negative: ${rows}` : rows;
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
}

function swapFieldInput(value) {
  return els.swapDialog.querySelector(`input[name="swap-field"][value="${value}"]`);
}

// One row: Find, Replace with, and × - which on the last row left empties
// it rather than leaving no row to type in.
function swapRow(find = '', replace = '') {
  const row = document.createElement('div');
  row.className = 'swap-row';
  const findBox = document.createElement('input');
  findBox.type = 'text';
  findBox.className = 'swap-find';
  findBox.placeholder = 'Find: red hair, long hair';
  findBox.value = find;
  findBox.spellcheck = false;
  findBox.autocomplete = 'off';
  findBox.maxLength = SWAP_MAX_TEXT;
  const arrow = document.createElement('span');
  arrow.className = 'swap-arrow';
  arrow.setAttribute('aria-hidden', 'true');
  arrow.textContent = '→';
  const replaceBox = document.createElement('input');
  replaceBox.type = 'text';
  replaceBox.className = 'swap-replace';
  replaceBox.placeholder = 'Replace with (empty removes)';
  replaceBox.value = replace;
  replaceBox.spellcheck = false;
  replaceBox.autocomplete = 'off';
  replaceBox.maxLength = SWAP_MAX_TEXT;
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'ghost tiny swap-remove';
  remove.textContent = '×';
  remove.onclick = () => removeSwapRow(row);
  row.append(findBox, arrow, replaceBox, remove);
  return row;
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
}

function swapPayload() {
  const swaps = swapLiveRows()
    .map((row) => ({ find: row.querySelector('.swap-find').value.trim(), replace: row.querySelector('.swap-replace').value.trim() }))
    .filter((row) => row.find);
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
    swapSay('Type the tags to find, comma-separated, and what to put in their place.');
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
  const li = document.createElement('li');
  li.className = 'swap-result';
  li.dataset.id = image.id;
  li.dataset.status = image.status;
  if (image.thumb_url) {
    const img = document.createElement('img');
    img.src = image.thumb_url;
    img.alt = '';
    img.loading = 'lazy';
    li.append(img);
  } else {
    li.append(Object.assign(document.createElement('span'), { className: 'swap-thumb-gone' }));
  }
  const body = document.createElement('div');
  body.className = 'swap-body';
  const head = document.createElement('p');
  head.className = 'swap-line';
  const chip = Object.assign(document.createElement('span'), { className: 'swap-chip', textContent: SWAP_CHIP[image.status] ?? image.status });
  head.append(`#${image.id} `, chip);
  if (image.width && image.height) head.append(` ${image.width}×${image.height}`);
  if (image.detail && image.status !== 'queued') {
    head.append(Object.assign(document.createElement('span'), { className: 'swap-detail', textContent: ` ${image.detail}` }));
  }
  body.append(head);
  if (image.text) body.append(swapMarked(image.text, image.spans));
  if (image.removed?.length) body.append(swapTookOut(image.removed));
  if (image.notes?.length) {
    body.append(Object.assign(document.createElement('p'), { className: 'swap-notes', textContent: image.notes.join(' · ') }));
  }
  li.append(body);
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
  if (!swapAnswer || !swapAnswer.queued) return 'No image would change';
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
  els.swapRows.addEventListener('input', () => swapEdited());
  // Out of the rows, the slots removed rows kept can go: nothing in there is
  // under the pointer, and the rows' fixed height moves nothing outside.
  els.swapRows.addEventListener('pointerleave', collapseSwapRows);
  els.swapDialog.addEventListener('change', (e) => { if (e.target.name === 'swap-field') swapEdited(0); });
  // Enter in a box is Queue's press, as in any form.
  els.swapRows.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.isComposing && e.target.matches('input')) { e.preventDefault(); void queueSwap(); }
  });
  els.swapDialog.addEventListener('close', () => {
    clearTimeout(swapTimer);
    swapAsking?.abort();
    swapAsking = null;
  });
  onSelectionChange(syncSwapButton);
  syncSwapRowNames();
  syncSwapButton();
}
