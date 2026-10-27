// AuraDream's page script: the job queue and which job the page watches.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

async function nameJob(jobId) {
  try { jobLabels.set(jobId, describeJob(await api(`/api/jobs/${jobId}`))); } catch { return; }
  syncQueueNote();
}

// The jobs waiting behind the watched one, from the ids their submissions
// returned - and, after a reload, from /api/jobs - one row each, named, with
// an × that cancels that one: "1 more job" named none of them, and Cancel
// queued was all or nothing. One waiting is one row, "Next:" under the Now
// line, with no heading over it: a heading and a row kept the queue below a
// 1040 px window. Rebuilt whole, with focus kept on the row it was on.
const queueCancelling = new Set();
// When each queued id was put in the list, so a status asked before then -
// whose listing could not have it - does not take it out (resyncQueue).
const queuedAt = new Map();
const queueStates = new Map();
let queueFirstId = null;
let queueMoving = null;
let queueOrderChangedAt = 0;

function rememberQueueState(job) {
  if (!job?.job_id) return;
  queueStates.set(job.job_id, job.status);
  if (job.job_id === currentJob && job.status === 'running' && queueFirstId === currentJob) {
    queueFirstId = queuedIds[0] ?? null;
    syncQueueNote();
  }
  syncQueueFront();
}

// The watched job can itself be queued behind another tab's work. It is not
// one of the queue rows, so its own Move action lives beside the progress.
function syncQueueFront() {
  const button = els.front;
  if (!button) return;
  const queued = busy && !!currentJob && queueStates.get(currentJob) === 'queued';
  // The progress parent hides at idle. Keep this slot throughout a job so
  // its admission to the worker cannot move Skip, Pause or Cancel.
  button.hidden = false;
  const reason = !busy || !currentJob ? 'No queued job is selected'
    : !queued ? 'Only a queued job can move to front'
    : queueMoving !== null ? 'Wait for the queue change to finish'
    : queueFirstId === null ? 'Checking the queue order'
    : queueFirstId === currentJob ? 'This job is already first in the queue' : '';
  button.setAttribute('aria-disabled', String(!!reason));
  button.title = reason || 'Move this job to the front of the queue; current work keeps running';
}

function moveCurrentJobFront() {
  if (!els.front || els.front.hidden || els.front.getAttribute('aria-disabled') === 'true') {
    if (busy) showJobNote(`${els.front?.title ?? 'No queued job is selected'}.`);
    return;
  }
  if (currentJob) void moveQueuedFront(currentJob);
}

// The queue replaced: a row whose job has gone keeps focus where it was - the
// row now in its place, else the one before, else Cancel - as an × press does.
function setQueue(ids) {
  const now = Date.now();
  for (const id of ids) if (!queuedAt.has(id)) queuedAt.set(id, now);
  for (const id of queuedAt.keys()) if (!ids.includes(id)) queuedAt.delete(id);
  const rows = [...els.queueList.children];
  const row = rows.findIndex((li) => li.contains(document.activeElement));
  const action = document.activeElement?.dataset.queueAction ?? 'cancel';
  const lost = row >= 0 && !ids.includes(rows[row].dataset.id);
  queuedIds = ids;
  syncQueueNote();
  // A paused job's cell line says what waits behind it.
  if (cellArgs) showCell(...cellArgs);
  if (!lost) return;
  const left = [...els.queueList.children];
  const replacement = left[row] ?? left[row - 1];
  (replacement?.querySelector(`[data-queue-action="${action}"]`)
    ?? replacement?.querySelector('button') ?? (busy ? els.cancel : els.generate)).focus({ preventScroll: true });
}

function syncQueueNote() {
  syncQueueFront();
  const n = queuedIds.length;
  els.queueNote.hidden = !(busy && n > 0);
  const waiting = queueStates.get(currentJob) === 'queued';
  els.queueText.textContent = n > 1 ? `${n} jobs queued${waiting ? '' : ' behind this one'}:` : '';
  els.queueText.hidden = n < 2;
  // Its × is the whole choice for one job; Cancel queued is for several.
  els.cancelQueued.hidden = n < 2;
  // A row of another tab's is that tab's work: Cancel queued says it takes
  // those too.
  els.cancelQueued.title = queuedIds.some((id) => !ownJobs.has(id))
    ? "Cancel every other queued job, another tab's included"
    : 'Cancel every other queued job';
  const focused = document.activeElement?.closest?.('#queue-list li')?.dataset.id;
  const action = document.activeElement?.dataset.queueAction ?? 'cancel';
  els.queueList.replaceChildren(...queuedIds.map((id) => queueRow(id, n === 1)));
  if (focused) els.queueList.querySelector(`li[data-id="${focused}"] [data-queue-action="${action}"]`)?.focus({ preventScroll: true });
}

// A row this page did not submit - another tab's job, listed from the
// server (resyncQueue), or one a reload's listing found - says whose it is
// before what it is, the prompt last for the ellipsis: its × cancels another
// tab's work. Unnamed, "a job submitted elsewhere" says it already.
function queueRow(jobId, alone) {
  const named = jobLabels.get(jobId);
  const foreign = named !== undefined && !ownJobs.has(jobId);
  const label = named ?? 'a job submitted elsewhere';
  const li = document.createElement('li');
  li.dataset.id = jobId;
  const text = document.createElement('span');
  const next = jobId === (queueFirstId ?? queuedIds[0]);
  text.textContent = foreign ? `${alone && next ? "Next, another tab's job" : ANOTHER_TABS_JOB}: ${label}`
    : alone ? `${next ? 'Next' : 'Queued'}: ${label}` : label;
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.dataset.queueAction = 'cancel';
  cancel.className = 'ghost tiny danger';
  cancel.textContent = '×';
  cancel.title = foreign ? "Cancel this job, another tab's, before it starts" : 'Cancel this job before it starts';
  cancel.setAttribute('aria-label', `Cancel ${foreign ? "another tab's " : ''}queued job: ${label}`);
  // Held while its POST runs, as Generate is during its own.
  cancel.setAttribute('aria-disabled', String(queueCancelling.has(jobId)));
  cancel.onclick = () => { void cancelQueuedJob(jobId); };
  const front = document.createElement('button');
  front.type = 'button';
  front.dataset.queueAction = 'front';
  front.className = 'ghost tiny';
  front.textContent = '↑';
  const reason = queueMoving !== null ? 'Wait for the queue change to finish'
    : next ? 'This job is already first in the queue' : '';
  front.title = reason || 'Move this job to the front of the queue; current work keeps running';
  front.setAttribute('aria-label', `Move ${foreign ? "another tab's " : ''}queued job to front: ${label}`);
  front.setAttribute('aria-disabled', String(!!reason));
  front.onclick = () => {
    if (reason) { showJobNote(`${reason}.`); return; }
    void moveQueuedFront(jobId);
  };
  li.append(text, cancel, front);
  return li;
}

async function moveQueuedFront(jobId) {
  if (queueMoving !== null) return;
  const watched = currentJob;
  const asked = Date.now();
  const before = new Set(queuedIds);
  queueMoving = jobId;
  queueOrderChangedAt = asked;
  syncQueueNote();
  try {
    const answer = await api(`/api/jobs/${jobId}/front`, { method: 'POST' });
    queueOrderChangedAt = Date.now();
    if (!busy || currentJob !== watched) return;
    const ids = answer.active_jobs ?? [];
    const ordered = ids.filter((id) => id !== watched && queueStates.get(id) !== 'running');
    // A Generate accepted while this reply was out is newer than its list.
    // Keep it at the tail, as a new server submission is appended there too.
    const newer = queuedIds.filter((id) => !ordered.includes(id)
      && ((queuedAt.get(id) ?? 0) > asked || !before.has(id)));
    queueFirstId = queueStates.get(jobId) === 'running' ? ordered[0] ?? null : jobId;
    queueWaiting = null;
    setQueue([...ordered, ...newer]);
    showJobNote('Moved to the front of the queue. Current work keeps running.');
  } catch (err) {
    showError(err.message);
  } finally {
    queueMoving = null;
    syncQueueNote();
    void refreshStatus();
  }
}

async function cancelQueuedJob(jobId) {
  if (queueCancelling.has(jobId)) return;
  queueCancelling.add(jobId);
  // Asked here: a page watching it says so as a cancel of its own.
  cancelledHere.add(jobId);
  syncQueueNote();
  try {
    await api(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
  } catch (err) {
    // Gone already is what was asked for; no answer is not.
    if (err.offline) {
      queueCancelling.delete(jobId);
      // Never sent: not this page's cancel should the job end cancelled.
      cancelledHere.delete(jobId);
      syncQueueNote();
      showError(err.message);
      return;
    }
  }
  queueCancelling.delete(jobId);
  forgetJob(jobId);
  // Through the queue's one writer, which keeps focus where the row was: the
  // next row's ×, else the one before, else Cancel.
  setQueue(queuedIds.filter((id) => id !== jobId));
}

// Once the watched job ends, the next one in the server's queue - oldest
// first, since that is the one about to run - becomes the watched job.
async function watchNext() {
  if (busy || jobListing) return;
  const jobId = await adoptListedJob('next in the queue…');
  if (jobId) await jobStreams.get(jobId);
}

// Only an accepted request still held by this page can owe an outcome.
// ownJobs survives reloads and includes old jobs; it is not a replay list.
function unwatchedSubmission(jobId) {
  return submissions.has(jobId) && !watchedJobs.has(jobId) && !cancelledHere.has(jobId);
}

// A page between jobs asks the server which to watch: the one holding the
// GPU, not the newest - with two live jobs the newest is the queued one, and
// attaching to it made Cancel stop the wrong job while the running grid kept
// the card with no reachable id - failing a running one the *oldest* queued,
// the one about to run, with the rest as its queue. It opens that job's
// stream itself, so whatever waits on the listing finds the job watched
// already (`watchedJobs`, `jobStreams`), whichever of them hears the answer
// first. Resolves to the job's id (null for nothing live, undefined for no
// answer, or a page that threw on it - which resumeActiveJob's own catch
// used to take, at load). While it is out it is `jobListing`, which a
// submission waits on (startJob) rather than taking the page beside it, and
// the page is checked again once it answers.
function adoptListedJob(text) {
  const listing = (async () => {
    try {
      let active;
      try { active = await api('/api/jobs'); } catch { return undefined; }
      if (busy) return null;
      // A short job can end between the previous stream and this listing.
      // Read the bounded server history only for accepted, unseen receipts
      // missing from the active list. Replay one at a time, in receipt order,
      // through the ordinary stream/finish path (including recovery tools).
      const pending = [...submissions.keys()].filter(unwatchedSubmission);
      let finished = null;
      if (pending.some((id) => !active.some((job) => job.job_id === id))) {
        let listed;
        try { listed = await api('/api/jobs?include_finished=true'); } catch { /* retry on the next poll */ }
        if (busy) return null;
        if (listed) {
          const byId = new Map(listed.map((job) => [job.job_id, job]));
          active = listed.filter((job) => ['queued', 'running'].includes(job.status));
          for (const id of pending) {
            if (!unwatchedSubmission(id)) continue;
            const job = byId.get(id);
            // Explicitly cancelled rows have been forgotten. A receipt the
            // server lost to a restart/retention cannot be streamed either.
            if (!job) { forgetJob(id); continue; }
            if (!finished && ['done', 'error', 'cancelled'].includes(job.status)) finished = job;
          }
        }
      }
      if (!finished && !active.length) { setQueue([]); return null; }
      const job = finished ?? active.find((j) => j.status === 'running') ?? active[0];
      learnJobs(active);
      setQueue(active.filter((j) => j.job_id !== job.job_id).map((j) => j.job_id));
      setBusy(true);
      currentJob = job.job_id;
      setProgressText(text);
      syncQueueNote();
      void streamJob(job.job_id);
      return job.job_id;
    } catch {
      return undefined;
    }
  })();
  jobListing = listing;
  void listing.finally(() => { if (jobListing === listing) jobListing = null; });
  return listing;
}

// Names for the jobs a listing returns that this page did not submit - a
// reload's, another tab's - keeping a submission's own where there is one.
function learnJobs(listed) {
  for (const job of listed) {
    rememberQueueState(job);
    if (!jobLabels.has(job.job_id)) jobLabels.set(job.job_id, describeJob(job));
  }
  queueFirstId = listed.find((job) => job.status === 'queued')?.job_id ?? null;
  syncQueueFront();
}

// What the status poll says is live on the server (`ModelStatus.active_jobs`,
// oldest first) against what this page watches. A page watching nothing
// learned of another tab's job only by a reload, or by its own next job: it
// said nothing was running, offered Unload and the critic, and its gallery
// missed that job's images. One watching its own listed only the jobs it
// had queued itself, a row cancelled elsewhere included. `asked` is when the
// status was asked for: a job this page listed since cannot be in it.
function followServerJobs(ids, asked) {
  if (busy) { void resyncQueue(ids, asked); return; }
  // Between jobs the listing that picks the next one decides, and a POST
  // out is this page's own job on its way.
  if (jobListing || submitting) return;
  // A transient listing failure must not strand an accepted receipt at idle.
  if ([...submissions.keys()].some(unwatchedSubmission)) { void watchNext(); return; }
  setElsewhereBusy(ids.some((id) => !watchedJobs.has(id) && !submissions.has(id)));
}

// Another tab's job on the card, of which this page watches nothing: held,
// with that reason, are Unload, the critic and Expand (syncUnload,
// syncCritic, syncPromptExpansion), and once it has gone the gallery and
// the Runs picker take in what it made, as they do when the server answers
// again (refreshStatus). Not attached to: watching it would take this page's
// viewer (isForeign is for one picked up after a job of this page's own).
// `quiet` where this page's own job takes over - its end catches up.
let elsewhereNote = null;
// Another tab's job has ended (setElsewhereBusy): taken in by the same
// status's followGalleryHead, which has the gallery's newest id to size the
// page it asks for.
let elsewhereEnded = false;
function setElsewhereBusy(on, { quiet = false } = {}) {
  if (elsewhereBusy === on) return;
  elsewhereBusy = on;
  syncPromptExpansion();
  // The grid's budget line counts the job ahead.
  schedulePreview();
  if (on) return;
  // The topbar's word for it (refreshStatus writes it while it lasts).
  els.statusAside.textContent = '';
  els.statusAside.hidden = true;
  // What a press on a held button said was about that job.
  if (elsewhereNote !== null && els.error.textContent === elsewhereNote) showError('');
  elsewhereNote = null;
  // Where this page's own job takes over, its end catches up instead - owed,
  // whatever that job makes (finishJob): a job of this page's cancelled
  // before its first image left the other tab's images out until a reload.
  if (quiet) { elsewhereCatchUp = true; return; }
  // Taken in by the same status's followGalleryHead (catchUpGallery), sized
  // by how far the gallery's newest id moved meanwhile.
  elsewhereEnded = true;
}

// Images this page did not watch being made are in: another tab's job ended,
// or one ran between two polls, or the CLI's. Every CFG ladder the lightbox
// holds that was read before them is asked again (noteUnseenImages),
// whatever the view takes in - a
// sheet, the bin, Starred or a search leave them out of `fresh` and
// joinFinished's noteLadderImages. Not under a sheet: a sheet is a run's or a
// search's, laid out again whole by a reset - every cell and control rebuilt
// - at a moment no one here chose, after a job this page never started.
// Sheet off resets the gallery, which takes the other tab's images in then.
// `made`: at most how many rows came in (the newest id's move), so the page
// prependFinished asks for reaches back past them to the head this page
// holds - an import of a hundred rows otherwise missed it and reset the
// gallery, emptying the grid under a user paged deep into it. `newest`: the
// status's newest id, which a ladder read since has in it already.
function catchUpGallery(made = 0, newest = null) {
  noteUnseenImages(newest);
  if (!sheetMode) void prependFinished(made, { upTo: newest });
  loadRuns();
}

// Another tab's images this page has yet to take in: its job ended while a
// job of this page's own was taking over (setElsewhereBusy's `quiet`), and
// that job's end takes them in - with its own, or alone where it made none.
let elsewhereCatchUp = false;

// The gallery's newest id on a status (`ModelStatus.newest_image_id`), against
// the last one said. `active_jobs` alone missed a job queued, run and ended
// between two polls - a turbo checkpoint's single image, a scripted fetch -
// which never set elsewhereBusy, so its images stayed out of this gallery and
// its CFG ladders until something else reset them. A move up that this page's
// own watched jobs do not explain (`ownImageIds`, every id between) is taken
// in as another tab's job's end is (catchUpGallery: prependFinished with its
// rules - nothing under a sheet or in the bin; cards held while the pointer
// rests on the gallery's column; no duplicates, joinFinished - the ladders,
// the Runs picker). While a job of this page's runs it is owed to that job's
// end (`elsewhereCatchUp`). While another tab's runs, its images come in as
// the status counts them: waiting for its end held a chain of them - a
// queue of Generates, Mass refine or More seeds split into several jobs -
// out until the last ended, hours for a grid, and a job that ended between
// two polls with another of that tab's queued behind it. `settled`:
// this status took everything in already - a restart, the server back, a
// checkpoint change. The first gallery page sets the starting point (no
// refresh); statuses before it change nothing. Unknown (None: a bundle
// operation or a clear holds the gallery) keeps what was said. A fall - a
// permanent delete of the newest, a clear - is only remembered: the next
// insert lands past the old value either way.
function followGalleryHead(newest, settled) {
  const known = Number.isInteger(newest);
  const seen = newestImageSeen;
  // Another tab's job ended (setElsewhereBusy).
  if (elsewhereEnded) {
    elsewhereEnded = false;
    if (!settled) catchUpGallery(known && seen !== null && newest > seen ? newest - seen : 0, known ? newest : null);
    if (known && seen !== null) keepGalleryHead(newest);
    return;
  }
  // Before the first gallery page: that page sets the starting point, read
  // with it (loadGallery, `GalleryPage.newest_image_id`) - this status's,
  // a separate read, could count an image saved after the page's SELECT.
  if (!known || seen === null) return;
  if (settled || newest <= seen) { keepGalleryHead(newest); return; }
  // This page's own job on its way, or the listing that picks the next one:
  // decided once they settle.
  if (submitting || jobListing) return;
  let explained = newest - seen <= 4096;
  for (let id = seen + 1; explained && id <= newest; id += 1) explained = ownImageIds.has(id);
  keepGalleryHead(newest);
  if (explained) return;
  // This page's own job's end takes the gallery in (`elsewhereCatchUp`); the
  // ladders are asked now, the images being saved.
  if (busy) { elsewhereCatchUp = true; noteUnseenImages(newest); return; }
  catchUpGallery(newest - seen, newest);
}

function keepGalleryHead(newest) {
  newestImageSeen = newest;
  for (const id of ownImageIds) if (id <= newest) ownImageIds.delete(id);
}

// The queue behind the watched job as the server has it: a row cancelled in
// another tab goes, and a job queued from elsewhere comes in under the
// server's name for it - the paused tab showed nothing waiting behind its
// pause. Only the jobs after the watched one: one ahead of it is not its
// queue. A job this page queued after the status was asked stays.
let queueResyncing = false;
async function resyncQueue(ids, asked) {
  if (queueResyncing || !currentJob || submitting || jobListing) return;
  if (queueMoving !== null || asked < queueOrderChangedAt) return;
  const watched = currentJob;
  const previousFirst = queueFirstId;
  const at = ids.indexOf(watched);
  // Ended: its stream says so, and the next listing picks the queue up.
  if (at < 0) { queueWaiting = null; return; }
  if (at > 0 || ids.some((id) => !jobLabels.has(id))) {
    queueResyncing = true;
    try {
      const listed = await api('/api/jobs');
      // A move or a different watched job may have superseded this request.
      // Do not let its old statuses or first queued id leak past the guard.
      if (!busy || submitting || jobListing || currentJob !== watched
        || queueMoving !== null || asked < queueOrderChangedAt) return;
      learnJobs(listed);
      ids = listed.map((job) => job.job_id);
    } catch {
      return;
    } finally {
      queueResyncing = false;
    }
    if (!busy || submitting || jobListing || currentJob !== watched
      || queueMoving !== null || asked < queueOrderChangedAt) return;
  }
  // A page can watch a queued job while an editor or another tab runs.
  // Moving another queued job ahead must keep it visible, rather than drop
  // everything before the watched id as the old insertion-order queue did.
  const behind = ids.filter((id) => id !== watched && queueStates.get(id) !== 'running');
  const first = ids.find((id) => queueStates.get(id) !== 'running');
  queueFirstId = first ?? null;
  syncQueueFront();
  const mine = queuedIds.filter((id) => !behind.includes(id) && (queuedAt.get(id) ?? 0) > asked);
  const next = [...behind, ...mine];
  // Back where it is shown, a change held for the pointer is gone with it:
  // replayed, a job queued and cancelled elsewhere came in as the pointer
  // left, its × a cancelled job's, and went at the next poll.
  if (next.length === queuedIds.length && next.every((id, i) => id === queuedIds[i])) {
    queueWaiting = null;
    if (previousFirst !== queueFirstId) syncQueueNote();
    return;
  }
  // Under a resting pointer it waits, asked again as it leaves (queuePointer).
  if (queuePointer) { queueWaiting = { ids, asked }; return; }
  for (const id of queuedIds) if (!next.includes(id) && !unwatchedSubmission(id)) forgetJob(id);
  setQueue(next);
}

// A row the status poll puts in or takes out - another tab's job queued or
// cancelled - changes the list's height, and with it moves everything under
// it: in the panel the grid fold and the rest, and in one column, under 901
// px, the whole page below. At 2560x1305 the Grid search summary went 138 px
// down from under a pointer resting on it, and what was under the pointer
// then was the new row's ×, which cancels the other tab's job. So while the
// pointer is on the list or anything it would move (`queuePointer`, from
// the page's pointerover, as `panelPos.pointer` and `galleryPointer` are,
// never `:hover`) the poll's change waits (`queueWaiting`, the latest one),
// and is asked again once the pointer leaves (releaseQueue). This page's
// own rows - a submission, an × - are its own presses and go in at once.
let queuePointer = false;
let queueWaiting = null;
function pointQueue(target) {
  const on = movedByQueue(target);
  if (on) { queuePointer = true; return; }
  if (queuePointer) releaseQueue();
}

function movedByQueue(target) {
  if (!(target instanceof Element)) return false;
  const after = Node.DOCUMENT_POSITION_FOLLOWING | Node.DOCUMENT_POSITION_CONTAINED_BY;
  if (!(els.queueNote.compareDocumentPosition(target) & after)) return false;
  // Beside the stage only the panel's foot moves: the panel holds its
  // place while the pointer is on it (panelPos.pointer).
  const panel = els.queueNote.closest('.panel');
  return getComputedStyle(panel).position !== 'sticky' || panel.contains(target);
}

function releaseQueue() {
  queuePointer = false;
  const waiting = queueWaiting;
  queueWaiting = null;
  if (waiting) void resyncQueue(waiting.ids, waiting.asked);
}

async function cancelQueued() {
  const ids = queuedIds.slice();
  for (const id of ids) {
    cancelledHere.add(id);
    try {
      await api(`/api/jobs/${id}/cancel`, { method: 'POST' });
    } catch (err) {
      // Gone already is what was asked for. Not asked at all, it is not this
      // page's cancel should it end cancelled.
      if (err.offline) cancelledHere.delete(id);
    }
    forgetJob(id);
  }
  setQueue(queuedIds.filter((id) => !ids.includes(id)));
}
