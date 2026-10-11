// Local Diffusion's page script: the job queue and which job the page watches.
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

// The queue replaced: a row whose job has gone keeps focus where it was - the
// row now in its place, else the one before, else Cancel - as an × press does.
function setQueue(ids) {
  const now = Date.now();
  for (const id of ids) if (!queuedAt.has(id)) queuedAt.set(id, now);
  for (const id of queuedAt.keys()) if (!ids.includes(id)) queuedAt.delete(id);
  const rows = [...els.queueList.children];
  const row = rows.findIndex((li) => li.contains(document.activeElement));
  const lost = row >= 0 && !ids.includes(rows[row].dataset.id);
  queuedIds = ids;
  syncQueueNote();
  // A paused job's cell line says what waits behind it.
  if (cellArgs) showCell(...cellArgs);
  if (!lost) return;
  const left = els.queueList.querySelectorAll('li button');
  (left[row] ?? left[row - 1] ?? (busy ? els.cancel : els.generate)).focus();
}

function syncQueueNote() {
  const n = queuedIds.length;
  els.queueNote.hidden = !(busy && n > 0);
  els.queueText.textContent = n > 1 ? `${n} jobs queued behind this one:` : '';
  els.queueText.hidden = n < 2;
  // Its × is the whole choice for one job; Cancel queued is for several.
  els.cancelQueued.hidden = n < 2;
  // A row of another tab's is that tab's work: Cancel queued says it takes
  // those too.
  els.cancelQueued.title = queuedIds.some((id) => !ownJobs.has(id))
    ? "Cancel every job waiting behind this one, another tab's included"
    : 'Cancel every job waiting behind this one';
  const focused = document.activeElement?.closest?.('#queue-list li')?.dataset.id;
  els.queueList.replaceChildren(...queuedIds.map((id) => queueRow(id, n === 1)));
  if (focused) els.queueList.querySelector(`li[data-id="${focused}"] button`)?.focus();
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
  text.textContent = foreign ? `${alone ? "Next, another tab's job" : ANOTHER_TABS_JOB}: ${label}`
    : alone ? `Next: ${label}` : label;
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'ghost tiny danger';
  cancel.textContent = '×';
  cancel.title = foreign ? "Cancel this job, another tab's, before it starts" : 'Cancel this job before it starts';
  cancel.setAttribute('aria-label', `Cancel ${foreign ? "another tab's " : ''}queued job: ${label}`);
  // Held while its POST runs, as Generate is during its own.
  cancel.setAttribute('aria-disabled', String(queueCancelling.has(jobId)));
  cancel.onclick = () => { void cancelQueuedJob(jobId); };
  li.append(text, cancel);
  return li;
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
      if (!active.length) { setQueue([]); return null; }
      const job = active.find((j) => j.status === 'running') ?? active[0];
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
  for (const job of listed) if (!jobLabels.has(job.job_id)) jobLabels.set(job.job_id, describeJob(job));
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
  // Not under a sheet: a sheet is a run's or a search's, laid out again
  // whole by a reset - every cell and control rebuilt - at a moment no one
  // here chose, after a job this page never started. Sheet off resets the
  // gallery, which takes the other tab's images in then.
  if (!sheetMode) void prependFinished(0);
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
// in as setElsewhereBusy's end takes another tab's job: prependFinished(0)
// with its rules (nothing under a sheet or in the bin; cards held while the
// pointer rests on the gallery's column; no duplicates, joinFinished), and
// the Runs picker. While a job of this page's runs it is owed to that job's
// end (`elsewhereCatchUp`); while another tab's runs, its end catches up and
// this waits for that. `settled`: this status took everything in already -
// the first one (the baseline, no refresh), a restart, the server back, a
// checkpoint change, another tab's job just ended. Unknown (None: a bundle
// operation or a clear holds the gallery) keeps what was said. A fall - a
// permanent delete of the newest, a clear - is only remembered: the next
// insert lands past the old value either way.
function followGalleryHead(newest, settled) {
  if (!Number.isInteger(newest)) return;
  const seen = newestImageSeen;
  if (seen === null || settled || newest <= seen) { keepGalleryHead(newest); return; }
  // This page's own job on its way, or the listing that picks the next one:
  // decided once they settle. Another tab's job running: its end catches up.
  if (submitting || jobListing || elsewhereBusy) return;
  let explained = newest - seen <= 4096;
  for (let id = seen + 1; explained && id <= newest; id += 1) explained = ownImageIds.has(id);
  keepGalleryHead(newest);
  if (explained) return;
  if (busy) { elsewhereCatchUp = true; return; }
  if (!sheetMode) void prependFinished(0);
  loadRuns();
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
  const watched = currentJob;
  const at = ids.indexOf(watched);
  // Ended: its stream says so, and the next listing picks the queue up.
  if (at < 0) return;
  const behind = ids.slice(at + 1);
  if (behind.some((id) => !jobLabels.has(id))) {
    queueResyncing = true;
    try {
      learnJobs(await api('/api/jobs'));
    } catch {
      return;
    } finally {
      queueResyncing = false;
    }
    if (!busy || submitting || jobListing || currentJob !== watched) return;
  }
  const mine = queuedIds.filter((id) => !behind.includes(id) && (queuedAt.get(id) ?? 0) > asked);
  const next = [...behind, ...mine];
  if (next.length === queuedIds.length && next.every((id, i) => id === queuedIds[i])) return;
  for (const id of queuedIds) if (!next.includes(id)) forgetJob(id);
  setQueue(next);
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
