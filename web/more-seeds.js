// Local Diffusion's page script: More seeds, more images of every batch the gallery selection holds.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// "More seeds…" in the gallery's footer row: N more images of each batch the
// selection holds (gallery-select.js), else of each batch the filter
// matches. A batch is one run's
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
//
// Where a run ends, the same dialog opens on a ready target rather than the
// footer's (openMoreSeeds' `target`): an image's batch (`{ids}` of one, from
// the menu, the lightbox and the result's column), a finished job's images
// (`{ids}`, the finish line), or a whole run (`{run}`: a grid's finish line,
// the banner, the sheet's run bar), whose images the server reads starred or
// not, and whose headline it writes. Those open at the count the press
// passes, else the server's `count_default` - the batch's own Batch - and
// never at the footer's remembered count, which only the footer reads and
// writes.

const MS_CHECK_DELAY = 250;
const MORE_SEEDS_KEY = 'local-diffusion.more-seeds.v1';
const MORE_SEEDS_COUNT = 4;
// In the count line, in this order whatever the server's.
const MS_STATUS_WORDS = {
  foreign: 'from other checkpoints', source_deleted: 'with their refine source deleted',
  lora_missing: 'with their LoRA missing', mask_missing: 'with their mask missing',
  outpaint: 'with an outpaint that cannot run', invalid: 'invalid', deleted: 'deleted',
};
const MS_CHIP = {
  queued: 'Ready', foreign: 'other checkpoint', source_deleted: 'source deleted',
  lora_missing: 'LoRA missing', mask_missing: 'mask missing', outpaint: 'outpaint refused', invalid: 'invalid',
  deleted: 'deleted',
};

// What the dialog acts on, fixed when it opens: `{ids}`, `{run}` or the
// filter, and the galleryActionTarget (or ready target) it came from, whose
// count the scope line says.
let msScope = null;
let msTarget = null;
// Opened from the footer: the one open that reads and writes the remembered
// count.
let msFooter = false;
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
// The fallback is chosen when the dialog opens, not by later composer edits.
let msBatch = 1;
let msFavorites = false;
// Once sent, this exact body and intent survive a lost response. A retry
// must recover that receipt before any new preview can allocate an intent.
let msSubmission = null;
let msUncertain = false;

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
// A run's scope is the run's, whatever the gallery shows.
function moreSeedsGalleryChanged() {
  if (!els.moreSeedsDialog.open || !msScope || msScope.ids || msScope.run || msSending || msSubmission) return;
  if (total === msSeenTotal) return;
  msSeenTotal = total;
  msAnswerKey = '';
  scheduleMoreSeedsCheck(0);
}

function msSayScope(answer) {
  if (!msTarget) return;
  if (!msScope.ids && !msScope.run && answer && typeof answer.total === 'number') {
    // The filter's images, not the answer's rows, which are batches.
    msTarget = { ...msTarget, n: answer.ids.length };
  }
  // A ready target names itself: an image's batch, a job's images, a run.
  const line = msTarget.line ?? galleryActionScopeLine(msTarget, 'From');
  if (els.moreSeedsScope.textContent !== line) els.moreSeedsScope.textContent = line;
}

function msCountText() { return els.moreSeedsCount.value.trim(); }

function msCountProblem() {
  const text = msCountText();
  return /^\d+$/.test(text) && +text >= 1 && +text <= sheetPage ? ''
    : `Additional images is a whole number from 1 to ${sheetPage}`;
}

// The panel's Batch, for a batch whose own was not recorded: the box's
// number where it is one in the box's range, else 1, as the server would
// refuse anything else.
function msFallbackBatch() {
  const n = +els.batch.value;
  return Number.isInteger(n) && n >= 1 && n <= +els.batch.max ? n : 1;
}

// A run scope ignores Starred, but says so in its headline only where
// Starred is ticked, so it carries the box's state; the filter's scope
// carries its own.
function msPayload() {
  const starred = msScope.run ? { favorites: msFavorites } : {};
  return { ...msScope, ...starred, count: +msCountText(), batch: msBatch };
}

async function askMoreSeeds(payload, signal) {
  return api('/api/more-seeds/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
}

const msCountFits = (n) => Number.isInteger(n) && n >= 1 && n <= sheetPage;

// Where a refusal while opening is said: the lightbox's own line while it is
// up - the modal lightbox covers the gallery's - the panel's for a press in
// the panel (the finish line, the banner), else the gallery's, and over the
// card or result tile pressed on (`host`: the menu's, the result column's),
// as its other refusals are (`refuseInCard`) - under the viewer the
// gallery's line can be a screen away.
function msOpenSlot(opener, host = null) {
  if (els.lightbox.open) return { slot: els.lbError, mirror: false };
  if (opener?.closest?.('.panel')) return { slot: els.error };
  return { slot: els.galleryError, card: host };
}

function msRefuse(message, said) {
  if (said.card) refuseInCard(said.card, message, 'Not opened');
  else showError(message, true, said);
}

// Opens on the first check's answer, not before, as Mass refine does: a
// failed check is said in the gallery's line (msOpenSlot) rather than in a
// dialog with nothing to queue. All but the refusal by count (batches ×
// count past a job's worth, "2 batches × 600 is…"): the count is the
// dialog's to lower, so it opens on that refusal, said and held as an edit's
// would be, the box marked. From the footer the count opens at the one last
// queued, else four; from a ready target at `count`, else at the server's
// count_default - asked again at it where the first check was not - else
// four. Focused and selected either way. `opener` is the button pressed,
// busy while the check is out - none for a menu item, whose menu has gone;
// `host` the card or tile a refusal is said over (msOpenSlot). Over the
// lightbox the dialog opens on top, and the lightbox stays.
async function openMoreSeeds(target = null, { count = null, opener = null, host = null } = {}) {
  if (msOpening || els.moreSeedsDialog.open) return;
  const footer = !target;
  const said = msOpenSlot(opener, host);
  if (footer) target = moreSeedsTarget();
  if (!target.scope) { msRefuse(target.why, said); return; }
  msScope = structuredClone(target.scope);
  msTarget = target;
  msFooter = footer;
  msSeenTotal = total;
  msRevision = modelRevision;
  msBatch = msFallbackBatch();
  msFavorites = els.favOnly.checked;
  msSubmission = null;
  msUncertain = false;
  els.moreSeedsCount.readOnly = false;
  els.moreSeedsDetails.open = false;
  els.moreSeedsCountLabel.textContent = 'Additional images';
  els.moreSeedsNotes.textContent = '';
  const kept = footer ? +remembered(MORE_SEEDS_KEY) : count;
  els.moreSeedsCount.value = String(msCountFits(kept) ? kept : MORE_SEEDS_COUNT);
  const controller = new AbortController();
  const lightboxId = els.lightbox.open ? lbList[lbIndex]?.id : null;
  const fromViewer = opener === els.viewerMoreSeeds || !!(host?.closest?.('#viewer') || opener?.closest?.('#viewer'));
  const viewerContext = viewerRecords[0]?.job_id ?? viewerRecords[0]?.id ?? null;
  const scopeKey = JSON.stringify(msScope);
  const openDialogs = [...document.querySelectorAll('dialog[open]')];
  const opening = { controller, valid: () => {
    if (msOpening !== opening || controller.signal.aborted) return false;
    if (opener && (!opener.isConnected || !isShown(opener))) return false;
    if (host && (!host.isConnected || !isShown(host))) return false;
    if ([...document.querySelectorAll('dialog[open]')].some((dialog) => !openDialogs.includes(dialog))) return false;
    if (lightboxId != null && (!els.lightbox.open || lbList[lbIndex]?.id !== lightboxId)) return false;
    if (fromViewer && (viewerRecords[0]?.job_id ?? viewerRecords[0]?.id ?? null) !== viewerContext) return false;
    if (footer && JSON.stringify(moreSeedsTarget().scope) !== scopeKey) return false;
    return true;
  } };
  msOpening = opening;
  opener?.setAttribute('aria-busy', 'true');
  syncMoreSeedsTools();
  showError('', false, said);
  let answer = null;
  let overCap = '';
  try {
    let first = await askMoreSeeds(msPayload(), controller.signal);
    if (!opening.valid()) return;
    const suggested = first.count_default;
    if (!footer && count == null && msCountFits(suggested) && suggested !== +msCountText()) {
      els.moreSeedsCount.value = String(suggested);
      first = null;
      first = await askMoreSeeds(msPayload(), controller.signal);
      if (!opening.valid()) return;
    }
    answer = first;
  } catch (err) {
    if (!opening.valid() || err.name === 'AbortError') return;
    if (err.status === 422 && /^\d+ batch/.test(err.message)) overCap = err.message;
    else { msRefuse(err.message, said); return; }
  } finally {
    if (msOpening === opening) {
      msOpening = null;
      opener?.removeAttribute('aria-busy');
      syncMoreSeedsTools();
    }
  }
  if (msRevision !== modelRevision) {
    msRefuse('The checkpoint changed while More seeds was checking the batches; press More seeds… again.', said);
    return;
  }
  msAnswer = answer;
  msAnswerKey = JSON.stringify(msPayload());
  msProblem = overCap;
  msRefused = false;
  els.moreSeedsCount.removeAttribute('aria-invalid');
  showError('', false, { slot: els.moreSeedsError });
  // Choose one fixed list size for this opening. Later previews, including
  // refusals and deletions, cannot move Queue or Cancel under the pointer.
  els.moreSeedsDialog.classList.toggle('ms-single', !!answer && answer.total === 1);
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
  syncMoreSeedsTools();
  if (overCap) {
    showError(overCap, false, { slot: els.moreSeedsError });
    els.moreSeedsCount.setAttribute('aria-invalid', 'true');
  }
  els.moreSeedsCount.focus();
  els.moreSeedsCount.select();
}

function cancelMoreSeedsOpening() {
  if (!msOpening) return;
  msOpening.controller.abort();
  msOpening = null;
  document.querySelectorAll('.more-seeds-tool[aria-busy], #btn-more-seeds[aria-busy]').forEach((button) => button.removeAttribute('aria-busy'));
  syncMoreSeedsTools();
}

// An edit: a refusal said for the press before it is no longer the news.
function moreSeedsEdited(delay = MS_CHECK_DELAY) {
  if (msSubmission) return;
  els.moreSeedsCount.removeAttribute('aria-invalid');
  if (msRefused || els.moreSeedsError.textContent) {
    msRefused = false;
    showError('', false, { slot: els.moreSeedsError });
  }
  scheduleMoreSeedsCheck(delay);
}

function scheduleMoreSeedsCheck(delay = MS_CHECK_DELAY) {
  if (msSubmission) return;
  clearTimeout(msTimer);
  msTimer = setTimeout(() => { void runMoreSeedsCheck(); }, delay);
  syncMoreSeedsRun();
}

async function runMoreSeedsCheck() {
  if (!els.moreSeedsDialog.open || !msScope || msSubmission) return;
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

const msBatches = (n) => `${n} settings group${n === 1 ? '' : 's'}`;

function renderMoreSeeds(answer) {
  msSayScope(answer);
  els.moreSeedsTitle.textContent = answer.queued ? plural(answer.queued, 'more image') : 'More seeds';
  const multiple = answer.total > 1;
  els.moreSeedsCountLabel.textContent = multiple ? 'Additional images per settings group' : 'Additional images';
  const time = answer.seconds ? `about ${humanTime(answer.seconds)}`
    : answer.unpriced_sizes ? `no estimate: ${plural(answer.unpriced_sizes, 'size')} ${answer.unpriced_sizes === 1 ? 'has' : 'have'} no history`
      : '';
  // A run's first words are the server's headline: "2 more of each of 6
  // cells = 12 images · seeds 8–9 · Starred ignored".
  const parts = [multiple
    ? `${msCountText()} per settings group × ${answer.batches} ready = ${plural(answer.queued, 'additional image')}`
    : `${plural(answer.queued, 'additional image')}`, time];
  if (answer.unrecorded) parts.push(`${msBatches(answer.unrecorded)} use panel Batch ${msBatch}`);
  for (const [status, words] of Object.entries(MS_STATUS_WORDS)) {
    if (answer.counts[status]) parts.push(`${answer.counts[status]} ${words}`);
  }
  if (answer.jobs > 1) parts.push(`${answer.jobs} jobs, one per run`);
  if (answer.total > answer.results.length) parts.push(`the first ${answer.results.length} of ${answer.total} listed`);
  msSay(parts.filter(Boolean).join(' · '));
  els.moreSeedsNotes.textContent = answer.headline || '';
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
  if (submitting && !msSending) return 'Wait for the last submission to be accepted; then queue these images';
  const held = submitHold();
  if (held) return held;
  if (msRevision !== modelRevision) return 'The checkpoint changed; close More seeds and open it again';
  if (msUncertain) return '';
  const field = msCountProblem();
  if (field) return field;
  if (msAsking || JSON.stringify(msPayload()) !== msAnswerKey) return 'Checking the settings groups…';
  if (msProblem) return msProblem;
  if (!msAnswer?.queued) return 'No settings group can have more seeds here; each one says why';
  if (!msAnswer.intent) return 'This preview has no submission receipt; close More seeds and open it again';
  return '';
}

function syncMoreSeedsRun() {
  if (!els.moreSeedsDialog.open) return;
  els.moreSeedsCount.readOnly = msSending || !!msSubmission;
  if (msSending) {
    els.moreSeedsRun.textContent = 'Sending…';
    els.moreSeedsRun.setAttribute('aria-disabled', 'true');
    els.moreSeedsRun.title = 'Waiting for the server to take the batches';
    els.moreSeedsHold.textContent = 'Sending this request. Wait for its reply before closing.';
    return;
  }
  const hold = moreSeedsHold();
  els.moreSeedsRun.textContent = msUncertain ? 'Retry same request'
    : !hold && msAnswer ? `Queue ${plural(msAnswer.queued, 'image')}${msAnswer.jobs > 1 ? ` (${msAnswer.jobs} jobs)` : ''}` : 'Queue';
  els.moreSeedsRun.setAttribute('aria-disabled', String(Boolean(hold)));
  els.moreSeedsRun.title = hold
    || (msUncertain ? 'Retry the original request and recover its original jobs; no extra extension is added'
      : `Queue ${plural(msAnswer.queued, 'more image')} in ${msBatches(msAnswer.batches)} (Ctrl+Enter)`);
  els.moreSeedsHold.textContent = hold || (msUncertain
    ? 'The reply was lost. Retry keeps this count and the same request. Closing abandons the reply; the jobs may already be queued.'
    : busy ? 'These images will queue after the current work.' : 'Ready to queue.');
  if (!hold && !msRefused && !msUncertain && els.moreSeedsError.textContent) showError('', false, { slot: els.moreSeedsError });
}

// Queue: waits for the check of what the dialog says, then queues. Held,
// the press says why in the error line and points at the count where that
// is what holds it.
async function runMoreSeeds() {
  if (!els.moreSeedsDialog.open || msSending || msWaiting) return;
  msRefused = false;
  while (!msSubmission && !submitting && !msCountProblem() && (msAsking || JSON.stringify(msPayload()) !== msAnswerKey)) {
    clearTimeout(msTimer);
    msWaiting = true;
    try { await runMoreSeedsCheck(); } finally { msWaiting = false; }
    if (!els.moreSeedsDialog.open || msSending) return;
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
  if (!msSubmission) msSubmission = {
    payload: structuredClone({ ...msPayload(), expected: msAnswer.ids, intent: msAnswer.intent }),
    model: loadedModel, revision: msRevision,
  };
  const { payload, model, revision } = msSubmission;
  msSending = true;
  syncMoreSeedsRun();
  let accepted = false;
  await startJob('/api/more-seeds', payload, {
    slot: els.moreSeedsError,
    panel: false,
    model, revision,
    onAccepted: (_, answer) => {
      accepted = true;
      msSending = false;
      if (answer.warning || answer.not_queued) {
        const jobs = [answer.job_id, ...(answer.more ?? []).map((job) => job.job_id)].join(', ');
        const slot = els.lightbox.open ? els.lbError : els.galleryError;
        showError(`Queued ${plural(answer.queued, 'image')} in the original job${answer.more?.length ? 's' : ''} ${jobs}. ${plural(answer.not_queued ?? 0, 'image')} in ${msBatches(answer.not_queued_batches ?? 0)} not queued. ${answer.warning ?? ''} No further images were added automatically.`, false, { slot, mirror: slot !== els.lbError });
      }
      if (msFooter) remember(MORE_SEEDS_KEY, String(payload.count));
      els.moreSeedsDialog.close();
      // Focus goes back to the button that opened the dialog.
    },
    // Refused - the selection changed under it: the dialog stays open with
    // the reason, and checks again.
    onRefused: (err) => {
      msSending = false;
      msRefused = Boolean(els.moreSeedsError.textContent);
      const receiptRefused = err.status === 409 && /intent|receipt|more seeds request/i.test(err.message);
      // Once a reply was lost, even a later explicit refusal can come from
      // middleware before the receipt is read (for example bundle admission).
      // It cannot resolve whether the original request already queued work.
      if (!receiptRefused && (msUncertain || !err.status || err.status >= 500)) {
        msUncertain = true;
        showError(`The server's reply is uncertain: ${err.message}. Retry this same request to recover its jobs.`, false, { slot: els.moreSeedsError });
        syncMoreSeedsRun();
        return;
      }
      msUncertain = false;
      // Unknown or expired receipts cannot silently acquire a fresh intent.
      if (receiptRefused) {
        msProblem = `${err.message} Review the queue before starting a new More seeds request.`;
        showError(msProblem, false, { slot: els.moreSeedsError });
        syncMoreSeedsRun();
        return;
      }
      msSubmission = null;
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
// dialog that listed one checks again, and shows it as deleted; a finish
// line's More seeds… whose every image has gone is held.
function forgetMoreSeedsImages(gone) {
  syncMoreSeedsTools();
  if (!els.moreSeedsDialog.open || !msAnswer) return;
  if (msAnswer.ids.some((id) => gone.has(id))) scheduleMoreSeedsCheck(0);
}

// Keys inside the dialog: Enter in the box is Queue's press, as in any form;
// ArrowUp and ArrowDown step the count by one. Ctrl+Enter is the
// document's (app.js).
function moreSeedsKeydown(event) {
  if (keyboardEventHeld(event) || keyboardCommandModifier(event)) return;
  if (event.key === 'Enter' && !event.shiftKey && event.target === els.moreSeedsCount) {
    event.preventDefault();
    if (!event.repeat) void runMoreSeeds();
    return;
  }
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && event.target === els.moreSeedsCount) {
    if (els.moreSeedsCount.readOnly) { event.preventDefault(); return; }
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
  els.lbMoreSeeds.setAttribute('aria-keyshortcuts', 'M');
  els.moreSeeds.onclick = () => { void openMoreSeeds(null, { opener: els.moreSeeds }); };
  els.viewerMoreSeeds.onclick = () => {
    syncViewerMoreSeeds();
    const records = viewerRecords.filter((rec) => !isGone(rec.id));
    const target = records.length === 1 ? moreSeedsImageTarget(records[0]) : {
      scope: { ids: records.map((rec) => rec.id) },
      line: `From the settings groups in these ${records.length} generated result images.`,
    };
    pressMoreSeedsTool(els.viewerMoreSeeds, target);
  };
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
    msSubmission = null;
    msUncertain = false;
    syncMoreSeedsTools();
  });
  // A preview has no modal yet. Escape still cancels that opening, and a
  // late reply is checked against its original lightbox/results context.
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !msOpening || keyboardEventHeld(event)
        || keyboardCommandModifier(event) || event.shiftKey || event.repeat) return;
    // Help/settings above the pending opener own Escape until they close.
    const dialog = event.target.closest?.('dialog[open]');
    if (els.shortcutsDialog.open || els.snippetsDialog.open || (dialog && dialog !== els.lightbox)) return;
    event.preventDefault();
    cancelMoreSeedsOpening();
  }, true);
  els.lbMoreSeeds.onclick = pressLightboxMoreSeeds;
  onSelectionChange(syncMoreSeedsButton);
  syncMoreSeedsButton();
}

// ------------------------------------------------------- where a run ends

// Why a More seeds… for an image, a job or a run is held now, or ''. `of` is
// what the button was synced for: the checkpoint its images were made on, an
// image in the recycle bin, a job's images (`ids`), held once every one has
// gone since. Everything else - one batch among several from another
// checkpoint, a source or an adapter gone - is the server's, per batch.
function moreSeedsToolHold({ model = null, binned = false, ids = null, unavailable = '' } = {}) {
  if (unavailable) return unavailable;
  if (ids?.length && ids.every(isGone)) {
    return ids.length === 1 ? 'Its image has been deleted since' : `All ${ids.length} of its images have been deleted since`;
  }
  if (binned) return 'In the recycle bin: restore it to make more of its batch';
  if (modelsDiffer(model)) return `Made on ${checkpointName(model)}; the server is running ${checkpointName(loadedModel)}`;
  if (serverOffline) return 'The server is not answering';
  if (els.moreSeedsDialog.open || msOpening) return 'More seeds is open already';
  return '';
}

// A More seeds… button as it is now: held with the reason as its title, else
// titled with what it makes. `of` stays on the button (`msOf`), so a change
// of the checkpoint, the server or the dialog syncs it again
// (syncMoreSeedsTools).
function syncMoreSeedsTool(button, of = button.msOf) {
  button.msOf = of ?? {};
  button.classList.add('more-seeds-tool');
  const action = button.dataset.run ? finishRuns.get(button.dataset.run)?.details?.actions?.seeds : null;
  const held = moreSeedsToolHold(button.msOf) || (action && !action.available ? action.reason : '');
  button.setAttribute('aria-disabled', String(Boolean(held)));
  button.title = held || button.msOf.title || '';
}

function syncMoreSeedsTools() {
  document.querySelectorAll('.more-seeds-tool').forEach((button) => syncMoreSeedsTool(button));
}

// For one image: its batch. The menu mirrors it (syncedItem).
function syncMoreSeedsFor(button, rec) {
  syncMoreSeedsTool(button, {
    model: rec.model_id,
    binned: !!rec.binned_at,
    title: `More images of #${rec.id}'s batch, the seeds following on - each at its own prompt, size, settings and Batch`,
  });
}

// A press: held, it says why where it was pressed (msOpenSlot) and opens
// nothing; else the dialog opens on `target`, a ready one ({scope, line}).
function pressMoreSeedsTool(button, target, options = {}) {
  syncMoreSeedsTool(button);
  const opener = options.opener ?? button;
  if (button.getAttribute('aria-disabled') === 'true') {
    msRefuse(button.title, msOpenSlot(opener, options.host));
    return;
  }
  void openMoreSeeds(target, { ...options, opener });
}

function moreSeedsImageTarget(rec) {
  return { scope: { ids: [rec.id] }, n: 1, line: `From #${rec.id}'s batch.` };
}

function moreSeedsRunTarget(runId) {
  return { scope: { run: runId }, line: `From every image of run ${runId}, starred or not.` };
}

// A More seeds… button for a run (the finish line, the banner, the run bar).
function moreSeedsRunButton(runId, { model = null, label = 'More seeds of this run…', className = 'ghost small' } = {}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = className;
  button.dataset.run = runId;
  button.textContent = label;
  button.setAttribute('aria-haspopup', 'dialog');
  syncMoreSeedsTool(button, {
    model,
    title: `More seeds of every cell of run ${runId}, starred or not, the seeds following on - each at its own settings and Batch`,
  });
  button.onclick = () => pressMoreSeedsTool(button, moreSeedsRunTarget(runId));
  return button;
}

// The menu's on a card or a result: as the result column's button, held the
// same way, the refusal said where the image is (refuseInCard).
function moreSeedsOfImage(rec, host) {
  const scratch = document.createElement('button');
  syncMoreSeedsFor(scratch, rec);
  if (scratch.getAttribute('aria-disabled') === 'true') {
    refuseInCard(host, `${scratch.title}.`, 'Held');
    return;
  }
  void openMoreSeeds(moreSeedsImageTarget(rec), { host });
}

// M always refers to the focused image's batch, independent of the picks or
// gallery filter. Reuse the existing holds and preview; this never queues.
function openMoreSeedsFromKey(target) {
  let rec, host = null;
  if (els.lightbox.open) {
    rec = lbList[lbIndex];
    host = els.lbStage;
  } else {
    const card = target.closest?.('#gallery .card[data-id]');
    const result = target.closest?.('#viewer figure[data-id]');
    const cell = target.closest?.('#sheet-grid .sheet-cell[data-id]');
    const recent = target.closest?.('#recent img[data-id]');
    if (card) { rec = galleryRecords.find((r) => r.id === +card.dataset.id); host = card; }
    else if (result) { rec = viewerRecords.find((r) => r.id === +result.dataset.id); host = result.querySelector('.tile'); }
    else if (cell) {
      rec = sheetRecords.find((r) => r.id === +cell.dataset.id)
        ?? (sheetLayout?.source?.id === +cell.dataset.id ? sheetLayout.source : null);
      host = cell;
    } else if (recent) rec = sessionRecords.find((r) => r.id === +recent.dataset.id);
  }
  if (!rec) return false;
  if (deletedIds.has(rec.id)) {
    msRefuse('Its image has been deleted since', msOpenSlot(target, host));
    return true;
  }
  const scratch = document.createElement('button');
  syncMoreSeedsFor(scratch, binnedIds.has(rec.id) && !rec.binned_at ? { ...rec, binned_at: true } : rec);
  if (scratch.getAttribute('aria-disabled') === 'true') {
    msRefuse(scratch.title, msOpenSlot(target, host));
    return true;
  }
  void openMoreSeeds(moreSeedsImageTarget(rec), { opener: target, host });
  return true;
}

function pressLightboxMoreSeeds() {
  const rec = lbList[lbIndex];
  if (!rec) return;
  syncMoreSeedsFor(els.lbMoreSeeds, rec);
  // At phone width the menu invokes this button while CSS hides it. Keep the
  // real visible launcher, or the opening's stale-context guard rejects it.
  const focused = document.activeElement;
  const opener = isShown(els.lbMoreSeeds) ? els.lbMoreSeeds
    : els.lightbox.contains(focused) && isShown(focused) ? focused : els.lbClose;
  pressMoreSeedsTool(els.lbMoreSeeds, moreSeedsImageTarget(rec), { opener });
}
