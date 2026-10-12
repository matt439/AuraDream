// Local Diffusion's page script: More seeds, more images of every batch the gallery selection holds.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// "More seeds…" in the gallery's footer row: N more images of each batch the
// selection holds (gallery-select.js), else of each batch the filter
// matches. A batch is what a card's hover shows (gallery-batch.js): one run's
// images that agree on everything but the seed, so two picked images of one
// batch ask for N more of it, not N of each. The server decides everything
// (`more_seeds.py`, `api_more_seeds.more_seeds_plan`): the batches, their next seeds
// - one past the highest the gallery holds of each, stepping over any
// already made or queued - the Batch they run at (the batch's own where its
// images recorded it, else the panel's), and what cannot run: another
// checkpoint's, an adapter or a source gone. The page lists the answer and
// adds nothing of its own. Queue is held, with its reason, until the answer
// is the answer to what the dialog says; the queueing request pins the ids
// that answer resolved (`expected`). One job per run the batches extend,
// stamped with that run's id, so the new images join it.

const MS_CHECK_DELAY = 250;
const MORE_SEEDS_KEY = 'local-diffusion.more-seeds.v1';
const MORE_SEEDS_COUNT = 4;
// In the count line, in this order whatever the server's.
const MS_STATUS_WORDS = {
  foreign: 'from other checkpoints', source_deleted: 'with their refine source deleted',
  lora_missing: 'with their LoRA missing', invalid: 'invalid', deleted: 'deleted',
};
const MS_CHIP = {
  queued: 'queued', foreign: 'other checkpoint', source_deleted: 'source deleted',
  lora_missing: 'LoRA missing', invalid: 'invalid', deleted: 'deleted',
};

// What the dialog acts on, fixed when it opens: `{ids}` or the filter, and
// the galleryActionTarget it came from, whose count the scope line says.
let msScope = null;
let msTarget = null;
// The gallery's count (`total`) the last check was asked under, as Mass
// refine keeps it (refineBatchGalleryChanged).
let msSeenTotal = null;
let msRevision = null;
// The last check's answer and the payload it answered (as a string).
let msAnswer = null;
let msAnswerKey = '';
// What the last check was refused with (too many images), or ''.
let msProblem = '';
// A Queue press the server refused: said until the next edit or press.
let msRefused = false;
let msAsking = null;
let msTimer = null;
// The POST is out: the dialog closes on its answer, and Escape and Cancel
// wait for it.
let msSending = false;
// A press waiting on its check: a second press does nothing. Two waits
// aborted each other's check in a loop that never yielded.
let msWaiting = false;
// An open waiting on its first check: a second press does nothing.
let msOpening = null;

function moreSeedsTarget() {
  return galleryActionTarget('to choose the batches to make more of', 'one press of More seeds');
}

// The button's state: held with the reason, or titled with what it takes.
// From syncRerender (every filter and count change) and every change of the
// selection.
function syncMoreSeedsButton() {
  const target = moreSeedsTarget();
  els.moreSeeds.setAttribute('aria-disabled', String(!target.scope));
  els.moreSeeds.title = target.scope
    ? `More images of each batch in ${target.words}, the seeds following on - each batch keeps its own prompt, size, settings and Batch`
    : target.why;
  moreSeedsGalleryChanged();
  syncMoreSeedsRun();
}

// The gallery's count changed under an open dialog acting on the filter:
// asked again, so the answer and the ids a press pins are the filter's now.
function moreSeedsGalleryChanged() {
  if (!els.moreSeedsDialog.open || !msScope || msScope.ids || msSending) return;
  if (total === msSeenTotal) return;
  msSeenTotal = total;
  msAnswerKey = '';
  scheduleMoreSeedsCheck(0);
}

function msSayScope(answer) {
  if (!msTarget) return;
  if (!msScope.ids && answer && typeof answer.total === 'number') {
    // The filter's images, not the answer's rows, which are batches.
    msTarget = { ...msTarget, n: answer.ids.length };
  }
  const line = galleryActionScopeLine(msTarget, 'From');
  if (els.moreSeedsScope.textContent !== line) els.moreSeedsScope.textContent = line;
}

function msCountText() { return els.moreSeedsCount.value.trim(); }

function msCountProblem() {
  const text = msCountText();
  return /^\d+$/.test(text) && +text >= 1 && +text <= sheetPage ? ''
    : `More of each batch is a whole number from 1 to ${sheetPage}`;
}

// The panel's Batch, for a batch whose own was not recorded: the box's
// number where it is one in the box's range, else 1, as the server would
// refuse anything else.
function msFallbackBatch() {
  const n = +els.batch.value;
  return Number.isInteger(n) && n >= 1 && n <= +els.batch.max ? n : 1;
}

function msPayload() {
  return { ...msScope, count: +msCountText(), batch: msFallbackBatch() };
}

async function askMoreSeeds(payload, signal) {
  return api('/api/more-seeds/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
}

// Opens on the first check's answer, not before, as Mass refine does: a
// failed check is said in the gallery's line rather than in a dialog with
// nothing to queue. All but the refusal by count (batches × count past a
// job's worth, "2 batches × 600 is…"): the count is the dialog's to lower,
// so it opens on that refusal, said and held as an edit's would be, the box
// marked. The count opens at the one last queued, else four, focused and
// selected.
async function openMoreSeeds() {
  if (msOpening || els.moreSeedsDialog.open) return;
  const target = moreSeedsTarget();
  if (!target.scope) { showError(target.why, true, { slot: els.galleryError }); return; }
  msScope = target.scope;
  msTarget = target;
  msSeenTotal = total;
  msRevision = modelRevision;
  const kept = +remembered(MORE_SEEDS_KEY);
  els.moreSeedsCount.value = String(Number.isInteger(kept) && kept >= 1 && kept <= sheetPage ? kept : MORE_SEEDS_COUNT);
  const opening = {};
  msOpening = opening;
  els.moreSeeds.setAttribute('aria-busy', 'true');
  showError('', false, { slot: els.galleryError });
  let answer = null;
  let overCap = '';
  try {
    answer = await askMoreSeeds(msPayload());
  } catch (err) {
    if (msOpening !== opening) return;
    if (err.status === 422 && /^\d+ batch/.test(err.message)) overCap = err.message;
    else { showError(err.message, true, { slot: els.galleryError }); return; }
  } finally {
    if (msOpening === opening) {
      msOpening = null;
      els.moreSeeds.removeAttribute('aria-busy');
    }
  }
  if (msRevision !== modelRevision) {
    showError('The checkpoint changed while More seeds was checking the batches; press More seeds… again.', true, { slot: els.galleryError });
    return;
  }
  msAnswer = answer;
  msAnswerKey = JSON.stringify(msPayload());
  msProblem = overCap;
  msRefused = false;
  els.moreSeedsCount.removeAttribute('aria-invalid');
  showError('', false, { slot: els.moreSeedsError });
  if (answer) renderMoreSeeds(answer);
  else {
    // Nothing listed: the refusal is the count's, said in the count line and
    // under it, and the title and list wait for a count that fits.
    msSayScope(null);
    els.moreSeedsTitle.textContent = 'More seeds';
    els.moreSeedsResults.replaceChildren();
    msSay(overCap, true);
  }
  els.moreSeedsDialog.showModal();
  syncMoreSeedsRun();
  if (overCap) {
    showError(overCap, false, { slot: els.moreSeedsError });
    els.moreSeedsCount.setAttribute('aria-invalid', 'true');
  }
  els.moreSeedsCount.focus();
  els.moreSeedsCount.select();
}

// An edit: a refusal said for the press before it is no longer the news.
function moreSeedsEdited(delay = MS_CHECK_DELAY) {
  els.moreSeedsCount.removeAttribute('aria-invalid');
  if (msRefused || els.moreSeedsError.textContent) {
    msRefused = false;
    showError('', false, { slot: els.moreSeedsError });
  }
  scheduleMoreSeedsCheck(delay);
}

function scheduleMoreSeedsCheck(delay = MS_CHECK_DELAY) {
  clearTimeout(msTimer);
  msTimer = setTimeout(() => { void runMoreSeedsCheck(); }, delay);
  syncMoreSeedsRun();
}

async function runMoreSeedsCheck() {
  if (!els.moreSeedsDialog.open || !msScope) return;
  const payload = msPayload();
  const key = JSON.stringify(payload);
  msAsking?.abort();
  msAsking = null;
  const field = msCountProblem();
  if (field) {
    // Said without asking; the list stays as the last answer left it.
    msAnswer = null;
    msAnswerKey = key;
    msProblem = field;
    msSay(field, true);
    syncMoreSeedsRun();
    return;
  }
  const asking = new AbortController();
  msAsking = asking;
  syncMoreSeedsRun();
  try {
    const answer = await askMoreSeeds(payload, asking.signal);
    if (msAsking !== asking) return;
    msAnswer = answer;
    msProblem = '';
    renderMoreSeeds(answer);
  } catch (err) {
    if (msAsking !== asking || err.name === 'AbortError') return;
    msAnswer = null;
    msProblem = err.message;
    msSay(err.message, true);
  }
  msAnswerKey = key;
  msAsking = null;
  syncMoreSeedsRun();
}

// The count line: two lines kept, whatever it says, its whole text the title.
function msSay(text, failed = false) {
  if (els.moreSeedsLine.textContent !== text) els.moreSeedsLine.textContent = text;
  els.moreSeedsLine.title = text;
  els.moreSeedsLine.classList.toggle('failed', failed);
}

const msBatches = (n) => `${n} batch${n === 1 ? '' : 'es'}`;

function renderMoreSeeds(answer) {
  msSayScope(answer);
  els.moreSeedsTitle.textContent = answer.queued ? plural(answer.queued, 'more image') : 'More seeds';
  const time = answer.seconds ? `about ${humanTime(answer.seconds)}`
    : answer.unpriced_sizes ? `no estimate: ${plural(answer.unpriced_sizes, 'size')} ${answer.unpriced_sizes === 1 ? 'has' : 'have'} no history`
      : '';
  const parts = [`${plural(answer.queued, 'image')} in ${msBatches(answer.batches)}`, time];
  if (answer.unrecorded) parts.push(`${msBatches(answer.unrecorded)} at the panel's Batch ${msFallbackBatch()}`);
  for (const [status, words] of Object.entries(MS_STATUS_WORDS)) {
    if (answer.counts[status]) parts.push(`${answer.counts[status]} ${words}`);
  }
  if (answer.jobs > 1) parts.push(`${answer.jobs} jobs, one per run`);
  if (answer.total > answer.results.length) parts.push(`the first ${answer.results.length} of ${answer.total} listed`);
  msSay(parts.filter(Boolean).join(' · '));
  els.moreSeedsResults.replaceChildren(...answer.results.map(moreSeedsRow));
}

const msSeeds = (low, high) => (low === high ? `seed ${low}` : `seeds ${low}–${high}`);

// A batch's row: its newest picked image, what the gallery holds of it, the
// seeds it gets and the Batch they run at - or why it gets none.
function moreSeedsRow(result) {
  const li = document.createElement('li');
  li.className = 'ms-result';
  li.dataset.id = result.id;
  li.dataset.status = result.status;
  if (result.thumb_url) {
    const img = document.createElement('img');
    img.src = result.thumb_url;
    img.alt = '';
    img.loading = 'lazy';
    li.append(img);
  } else {
    li.append(Object.assign(document.createElement('span'), { className: 'ms-thumb-gone' }));
  }
  const body = document.createElement('div');
  body.className = 'ms-body';
  const head = document.createElement('p');
  head.className = 'ms-line';
  const chip = Object.assign(document.createElement('span'), { className: 'ms-chip', textContent: MS_CHIP[result.status] ?? result.status });
  head.append(`#${result.id} `, chip);
  if (result.status !== 'deleted') {
    head.append(` ${plural(result.images, 'image')}, ${msSeeds(result.seed_low, result.seed_high)}`);
    if (result.queued) {
      head.append(` → ${msSeeds(result.first, result.last)}${result.skipped ? ` (${result.skipped} already made, stepped over)` : ''} · Batch ${result.batch}`);
    }
  }
  body.append(head);
  if (result.prompt) {
    body.append(Object.assign(document.createElement('p'), {
      className: 'ms-prompt', textContent: snippet(result.prompt, 140), title: result.prompt,
    }));
  }
  if (result.detail && result.status !== 'queued') {
    body.append(Object.assign(document.createElement('p'), { className: 'ms-detail', textContent: result.detail }));
  }
  if (result.notes?.length) {
    body.append(Object.assign(document.createElement('p'), { className: 'ms-notes', textContent: result.notes.join(' · ') }));
  }
  li.append(body);
  return li;
}

// Why Queue is held now, or '' when it may go.
function moreSeedsHold() {
  const held = submitHold();
  if (held) return held;
  const field = msCountProblem();
  if (field) return field;
  if (msAsking || JSON.stringify(msPayload()) !== msAnswerKey) return 'Checking the batches…';
  if (msProblem) return msProblem;
  if (msRevision !== modelRevision) return 'The checkpoint changed; close More seeds and open it again';
  if (!msAnswer?.queued) return 'No batch can have more seeds here; each one says why';
  return '';
}

function syncMoreSeedsRun() {
  if (!els.moreSeedsDialog.open) return;
  if (msSending) {
    els.moreSeedsRun.textContent = 'Sending…';
    els.moreSeedsRun.setAttribute('aria-disabled', 'true');
    els.moreSeedsRun.title = 'Waiting for the server to take the batches';
    return;
  }
  const hold = moreSeedsHold();
  const queue = busy ? ' (queue)' : '';
  els.moreSeedsRun.textContent = !hold && msAnswer
    ? `Queue ${plural(msAnswer.queued, 'image')}${msAnswer.jobs > 1 ? ` (${msAnswer.jobs} jobs)` : ''}${queue}`
    : `Queue${queue}`;
  els.moreSeedsRun.setAttribute('aria-disabled', String(Boolean(hold)));
  els.moreSeedsRun.title = hold
    || `Queue ${plural(msAnswer.queued, 'more image')} in ${msBatches(msAnswer.batches)} (Ctrl+Enter)`;
  if (!hold && !msRefused && els.moreSeedsError.textContent) showError('', false, { slot: els.moreSeedsError });
}

// Queue: waits for the check of what the dialog says, then queues. Held,
// the press says why in the error line and points at the count where that
// is what holds it.
async function runMoreSeeds() {
  if (!els.moreSeedsDialog.open || msSending || submitting || msWaiting) return;
  msRefused = false;
  while (!msCountProblem() && (msAsking || JSON.stringify(msPayload()) !== msAnswerKey)) {
    clearTimeout(msTimer);
    msWaiting = true;
    try { await runMoreSeedsCheck(); } finally { msWaiting = false; }
    if (!els.moreSeedsDialog.open || msSending || submitting) return;
  }
  const hold = moreSeedsHold();
  if (hold) {
    showError(hold, false, { slot: els.moreSeedsError });
    if (hold === msCountProblem() || /^\d+ batch/.test(hold)) {
      els.moreSeedsCount.setAttribute('aria-invalid', 'true');
      els.moreSeedsCount.focus();
    }
    if (serverOffline) void refreshStatus();
    return;
  }
  const payload = { ...msPayload(), expected: msAnswer.ids };
  msSending = true;
  syncMoreSeedsRun();
  let accepted = false;
  await startJob('/api/more-seeds', payload, {
    slot: els.moreSeedsError,
    panel: false,
    onAccepted: () => {
      accepted = true;
      msSending = false;
      remember(MORE_SEEDS_KEY, String(payload.count));
      els.moreSeedsDialog.close();
      // Focus goes back to More seeds… in the footer, which opened the dialog.
    },
    // Refused - the selection changed under it: the dialog stays open with
    // the reason, and checks again.
    onRefused: () => {
      msSending = false;
      msRefused = Boolean(els.moreSeedsError.textContent);
      msAnswer = null;
      msAnswerKey = '';
      scheduleMoreSeedsCheck(0);
    },
  });
  if (!accepted) {
    msSending = false;
    syncMoreSeedsRun();
  }
}

// Cancel, Escape and the backdrop, except while the POST is out.
function closeMoreSeeds() {
  if (!msSending) els.moreSeedsDialog.close();
}

// From forgetImages, the one place the page drops deleted ids: an open
// dialog that listed one checks again, and shows it as deleted.
function forgetMoreSeedsImages(gone) {
  if (!els.moreSeedsDialog.open || !msAnswer) return;
  if (msAnswer.ids.some((id) => gone.has(id))) scheduleMoreSeedsCheck(0);
}

// Keys inside the dialog: Enter in the box is Queue's press, as in any form;
// ArrowUp and ArrowDown step the count by one. Ctrl+Enter is the
// document's (app.js).
function moreSeedsKeydown(event) {
  if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === 'Enter' && !event.shiftKey && event.target === els.moreSeedsCount) {
    event.preventDefault();
    void runMoreSeeds();
    return;
  }
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && event.target === els.moreSeedsCount) {
    const text = msCountText();
    if (!/^\d+$/.test(text)) return;
    event.preventDefault();
    const next = +text + (event.key === 'ArrowUp' ? 1 : -1);
    els.moreSeedsCount.value = String(Math.min(sheetPage, Math.max(1, next)));
    moreSeedsEdited();
  }
}

function moreSeedsBackdrop(event) {
  if (event.target !== els.moreSeedsDialog) return;
  const box = els.moreSeedsDialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right
    && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) closeMoreSeeds();
}

function setupMoreSeeds() {
  els.moreSeeds.onclick = () => { void openMoreSeeds(); };
  els.moreSeedsCancel.onclick = closeMoreSeeds;
  els.moreSeedsRun.onclick = () => { void runMoreSeeds(); };
  els.moreSeedsDialog.addEventListener('click', moreSeedsBackdrop);
  els.moreSeedsDialog.addEventListener('keydown', moreSeedsKeydown);
  els.moreSeedsDialog.addEventListener('cancel', (e) => { if (msSending) e.preventDefault(); });
  els.moreSeedsCount.addEventListener('input', () => moreSeedsEdited());
  els.moreSeedsDialog.addEventListener('close', () => {
    clearTimeout(msTimer);
    msAsking?.abort();
    msAsking = null;
    msScope = null;
    msTarget = null;
  });
  onSelectionChange(syncMoreSeedsButton);
  syncMoreSeedsButton();
}
