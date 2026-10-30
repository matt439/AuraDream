// AuraDream's page script: the watched job's progress, mirrors and Pause.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// #progress-text is a status: every change to it is read out. `spoken` is
// the coarse state - queued, denoising, image 2/3, cell 4/12, finishing -
// written only when it changes, since a rewrite with the same words is read
// out again. `shown`, where the eye gets something else in its place - a
// single image's step count, "3/12", rewritten every step and read out every
// step - goes into an aria-hidden half beside a visually hidden spoken one
// (style.css), as the load's clock does (streamJob's tick). The mirrors and
// the title take what is shown (progressShown).
function setProgressText(spoken, shown = null) {
  const el = els.progressText;
  el.title = '';
  el.removeAttribute('aria-description');
  if (shown === null) {
    if (el.childElementCount || el.textContent !== spoken) el.textContent = spoken;
    return;
  }
  let said = el.querySelector(':scope > .said');
  let seen = el.querySelector(':scope > .seen');
  if (!said || !seen) {
    said = document.createElement('span');
    said.className = 'said';
    seen = document.createElement('span');
    seen.className = 'seen';
    seen.setAttribute('aria-hidden', 'true');
    el.replaceChildren(said, seen);
  }
  if (said.textContent !== spoken) said.textContent = spoken;
  if (seen.textContent !== shown) seen.textContent = shown;
}

function progressShown() {
  return els.progressText.querySelector(':scope > .seen')?.textContent ?? els.progressText.textContent;
}

// The watched job's fraction and time left, as the bars and the cell line
// last showed them, for the mirrors: the tab's title and the topbar.
let jobFraction = 0;
let jobEta = '';
// The same estimate in seconds (the budget's where it comes sooner), or null:
// what the queue's "all done in" adds its rows to (syncQueueTotal).
let jobEtaSeconds = null;

function setProgress(fraction) {
  jobFraction = fraction;
  els.fill.style.width = els.topbarFill.style.width = `${fraction * 100}%`;
  els.fill.parentElement.setAttribute('aria-valuenow', String(Math.round(fraction * 100)));
}

// While the weights load there is no fraction to show: a stretch of the bar
// slides (style.css) rather than sitting at 0% for minutes, which looks like
// a hang, and a screen reader is told no value.
function setIndeterminate(on) {
  const bar = els.fill.parentElement;
  if (bar.classList.contains('indeterminate') === on) return;
  bar.classList.toggle('indeterminate', on);
  els.topbarLine.classList.toggle('indeterminate', on);
  if (!on) { setProgress(jobFraction); return; }
  bar.removeAttribute('aria-valuenow');
  els.fill.style.width = els.topbarFill.style.width = '';
}

// The watched job where the eye is when it is not on the panel: the tab's
// title (a tab in the background said nothing about a job running or done)
// and a line in the topbar, which is in view at any scroll. The topbar's is
// aria-hidden - #progress-text is the status a screen reader hears.
const PAGE_TITLE = document.title;
function setTitle(text) {
  document.title = text ? `${text} — ${PAGE_TITLE}` : PAGE_TITLE;
}

// While the watched job's stream reconnects (streamJob's onerror): the topbar
// says it already, so its mirror says nothing more, and the tab says the same.
let streamLost = false;
function mirrorJob() {
  els.jobMirror.classList.toggle('load-wait', busy && !streamLost && !!els.progressText.querySelector('.clock'));
  if (!busy) { els.jobMirror.textContent = ''; setTitle(''); return; }
  if (streamLost) {
    els.jobMirror.textContent = '';
    setTitle(`${LOST_PHRASE[0].toUpperCase()}${LOST_PHRASE.slice(1)}`);
    return;
  }
  const text = progressShown();
  const steps = els.stepRow.hidden ? '' : els.stepText.textContent;
  // Another tab's job, picked up when this page's own ended, says whose it
  // is wherever it is shown (isForeign).
  const whose = foreignJobs.has(currentJob) ? ANOTHER_TABS_JOB : '';
  els.jobMirror.textContent = [whose, text, steps, jobEta].filter(Boolean).join(' · ');
  const measured = jobFraction > 0 && !els.fill.parentElement.classList.contains('indeterminate');
  const shown = [whose, text].filter(Boolean).join(' · ');
  const held = els.pause.textContent === 'Resume' ? pausedFor(cellArgs?.[1].job?.paused_at) : '';
  setTitle(els.pause.textContent === 'Resume' ? `⏸ Paused${held ? ` ${held}` : ''}${whose ? ` · ${whose}` : ''}`
    : measured ? `${Math.round(jobFraction * 100)}% · ${shown}` : shown);
}

// A finished job's outcome stays in the title until the page is looked at
// again - the tab focused, a press or a key in it - so a tab left in the
// background says the night is over, and how it ended. `titleHeld` is that
// outcome while it waits to be seen: another tab's job picked up meanwhile
// shows its progress there, and its end gives the title back (finishJob).
let titleHeld = null;
function holdTitle(text) {
  setTitle(text);
  titleHeld = text;
  const back = () => {
    window.removeEventListener('focus', back);
    document.removeEventListener('pointerdown', back, true);
    document.removeEventListener('keydown', back, true);
    titleHeld = null;
    if (!busy) setTitle('');
  };
  window.addEventListener('focus', back);
  document.addEventListener('pointerdown', back, true);
  document.addEventListener('keydown', back, true);
}

// What a job's units are called wherever its words reach the user: a grid's
// are cells; a plain job's repeats are images, or batches where each makes
// several. "cell 2/3" for three plain images named the grid's word, and
// "run 2/3" a fourth meaning of run.
function unitWord(job) {
  if (job?.grid) return 'cell';
  // Then refine's passes are no part of a batch: half the planned images,
  // after Finish run's missing passes (one image each) are taken off.
  const sources = job?.pass_sources?.length ?? 0;
  const made = ((job?.planned_images ?? 0) - sources) / (job?.then_refine ? 2 : 1);
  return made > ((job?.total_runs || 1) - sources || 1) ? 'batch' : 'image';
}

// The cell line, under the bars: a pause on its way or taken, which cell of a
// grid is running, the time so far and an estimate of what is left, and the
// cells skipped so far. A sweep sends no label - its settings never change -
// so only the timing shows there. The estimate is the job's own: time over
// the fraction of cells finished (the current cell's steps counting as a
// fraction of one), which tracks a run that has slowed where the gallery's
// per-step median would not. `elapsed` is the server's clock, so the two
// clocks need not agree - and so is a budget's end: `budget` is seconds from
// the job's start to its deadline, both the server's, and a budget that
// stops the job sooner than the estimate is the time left.
let cellArgs = null;
// Where a job's estimate is measured from (`jobId`, `elapsed`, `fraction`):
// its first progress event this page saw. The first step carries what no
// later one does - a pass's VAE encode of its source, the denoiser's move
// onto the card under offload - so extrapolated from the job's start, a WAI
// pass said 28 s at step 1 for 9.5 s, and FLUX.1's first image ~13 min. The
// estimate waits for a second progress event and runs from the first one's
// end; a one-step cell's next event is the next cell's, whose own encode
// every later cell will carry too. A one-step job has nothing to estimate.
let etaFrom = null;
// The job whose own-pace estimate the cell line has shown: from then on the
// history figure (`~N expected`) does not come back for it, not between
// cells nor at the last one, where there is nothing left to extrapolate.
let paceShown = null;
// How long a pause has held the gate, from the worker's `paused_at` (the
// server's Unix seconds, read against the server's clock as the last status
// gave it - a WSL server's drifts from the host's): "3 min", or
// nothing in its first minute. A paused job keeps the gate, and "⏸ Paused"
// said nothing of whether that was a minute or the night. Its clock is the
// status poll's (tickPauseClock), so whole minutes, not seconds.
function pausedFor(at) {
  if (typeof at !== 'number') return '';
  const s = Date.now() / 1000 + serverClockOffset - at;
  if (s < 60) return '';
  return s < 90 * 60 ? `${Math.floor(s / 60)} min` : humanTime(s);
}

// The status poll moves a pause's clock - this page's job paused, or one
// it waits behind - in the cell line and the tab's title.
function tickPauseClock() {
  const job = cellArgs?.[1].job;
  if (!busy || !(job?.paused || job?.ahead_paused_at != null)) return;
  showCell(...cellArgs);
  mirrorJob();
}

// `note` is what a queued job waits on, in words (the stream's status
// branch: the job ahead paused, the first load): kept in `cellArgs` with
// the rest, so a redraw - the queue resynced, a Pause pressed - says it
// again rather than wiping it under a progress line still saying "waiting".
// A function is asked again at each redraw, for a note with a clock in it.
function showCell(data, { job = null, budget = null, note = '', jobId = null } = {}) {
  cellArgs = [data, { job, budget, note, jobId }];
  rememberQueueState(job);
  syncSkip(job);
  const unit = unitWord(job);
  const parts = [];
  if (job?.paused) {
    const held = pausedFor(job.paused_at);
    const lasted = held ? ` for ${held}` : '';
    parts.push(job.run
      ? `Paused after ${unit} ${job.run}/${job.total_runs}${lasted} · GPU handed back`
      : `Paused before the first ${unit}${lasted} · GPU handed back`);
    // The card's memory is handed back, but the queue does not move on: a
    // job behind a pause - one another tab queued included - waits for
    // Resume or Cancel here.
    if (queuedIds.length) {
      parts.push(`${queuedIds.length === 1 ? '1 job waits' : `${queuedIds.length} jobs wait`} behind it`);
    }
  } else if (els.pause.textContent === 'Pausing…') {
    // A queued job has no image under way: its pause holds it before its
    // first one, once its turn comes (and it keeps the queue there).
    parts.push(job?.status === 'queued' ? `will hold before its first ${unit}` : `pausing after this ${unit}…`);
  }
  const said = typeof note === 'function' ? note(job) : note;
  if (said) parts.push(said);
  if (data.label) parts.push(`cell ${data.run}/${data.total_runs} · ${data.label}`);
  jobEta = '';
  jobEtaSeconds = null;
  const runs = data.total_runs || 1;
  const within = data.total_steps ? (data.step || 0) / data.total_steps : 0;
  const fraction = (((data.run || 1) - 1) + within) / runs;
  if (jobId !== null && data.step && typeof data.elapsed === 'number' && etaFrom?.jobId !== jobId) {
    etaFrom = { jobId, elapsed: data.elapsed, fraction };
  }
  if (typeof data.elapsed === 'number' && data.elapsed >= 1) {
    const from = etaFrom?.jobId === jobId ? etaFrom : null;
    let timing = `${humanTime(data.elapsed)} elapsed`;
    if (from && fraction > from.fraction && data.elapsed > from.elapsed && fraction > 0.02 && fraction < 1) {
      const eta = (data.elapsed - from.elapsed) * (1 - fraction) / (fraction - from.fraction);
      const left = budget === null ? Infinity : Math.max(0, budget - data.elapsed);
      jobEta = left < eta ? `~${humanTime(left)} left (budget)` : `~${humanTime(eta)} left`;
      jobEtaSeconds = Math.min(left, eta);
      paceShown = jobId;
      timing += ` · ${jobEta}`;
    }
    parts.push(timing);
  }
  // Until the job's own pace can be measured, the gallery's history for the
  // whole job, labelled as that - never mixed into the own-pace figure.
  const expected = !jobEta && data.step && jobId !== null && paceShown !== jobId ? queueFigure(jobId) : null;
  if (expected) parts.push(`${expected.text} expected`);
  els.cellNote.textContent = parts.join(' · ');
  // A grid's skipped cells as they happen, in the failure colour: the line
  // used to go from cell 4 to cell 6 with no mark.
  const skipped = (job?.failures ?? 0) + (job?.skipped ?? 0);
  if (skipped) {
    const span = document.createElement('span');
    span.className = 'skipped';
    span.textContent = `${parts.length ? ' · ' : ''}${skipped} skipped`;
    els.cellNote.append(span);
  }
  // The queue's total follows the live ETA it adds to.
  syncQueueTotal();
}

// Skip names the exact active call, not the job's next cell. Keep the token
// while its POST is out, and never enable a stale snapshot over that press.
let skipAsking = null;
let skipSnapshot = null;
// A queue redraw can replay cellArgs from before the POST's acknowledgement.
// Keep the accepted token held until a new call or the between-call frame.
let skipAccepted = null;

function initQueueControls() {
  for (const button of [els.skip, els.topbarSkip]) if (button) button.onclick = skipCurrentCell;
  if (els.front) els.front.onclick = moveCurrentJobFront;
  syncSkip(null);
  syncQueueFront();
}

function syncSkip(job) {
  skipSnapshot = job?.job_id === currentJob ? job : null;
  if (!skipSnapshot || skipAccepted?.jobId !== currentJob
    || skipAccepted.call !== skipSnapshot.current_call) skipAccepted = null;
  if (skipSnapshot?.skip_requested && skipSnapshot.current_call != null) {
    skipAccepted = { jobId: currentJob, call: skipSnapshot.current_call };
  }
  const pending = (skipAsking?.jobId === currentJob
    && skipAsking.call === skipSnapshot?.current_call)
    || (skipAccepted?.jobId === currentJob && skipAccepted.call === skipSnapshot?.current_call);
  const reason = pending ? 'Skipping this cell after the current step'
    : !busy || !skipSnapshot ? 'No cell is running'
    : skipSnapshot.skip_reason || (skipSnapshot.current_call == null ? 'No cell is running' : '');
  for (const button of [els.skip, els.topbarSkip]) {
    if (!button) continue;
    button.setAttribute('aria-disabled', String(!!reason));
    button.title = reason || `Skip this ${unitWord(job)} after the current step; Finish run can make it later`;
    // The short label stays one width so progress controls do not move.
    button.textContent = 'Skip';
  }
}

async function skipCurrentCell() {
  const jobId = currentJob;
  const call = skipSnapshot?.current_call;
  const button = els.skip ?? els.topbarSkip;
  if (!jobId || !button || button.getAttribute('aria-disabled') === 'true' || call == null) {
    if (busy) showJobNote(`${button?.title ?? 'No cell is running'}.`);
    return;
  }
  const asking = { jobId, call };
  skipAsking = asking;
  syncSkip(skipSnapshot);
  showJobNote(`Skipping this ${unitWord(skipSnapshot)} after the current step.`);
  try {
    const answer = await api(`/api/jobs/${jobId}/skip?call=${call}`, { method: 'POST' });
    if (currentJob === jobId && skipSnapshot?.current_call === call) syncSkip(answer);
  } catch (err) {
    if (currentJob === jobId) showJobNote(err.message);
  } finally {
    if (skipAsking === asking) skipAsking = null;
    if (currentJob === jobId) syncSkip(skipSnapshot);
  }
}

// The second bar: where the current run is within its own steps. Hidden for a
// single run, where it would only repeat the bar above it.
function setStep(show, fraction = 0, text = '') {
  els.stepRow.hidden = !show;
  els.stepFill.style.width = `${fraction * 100}%`;
  els.stepText.textContent = text;
}

// Pause holds the watched job at its next cell boundary and hands the card
// back; the same button resumes it. The worker's `paused` acknowledgement,
// not the click, is what flips the label to Resume - until it lands a denoise
// may still be running - but the click is answered at once: "Pausing…", and
// the cell line says the same, since on the card the boundary can be tens of
// seconds away and a button that kept saying Pause invited a second press,
// or Cancel. A press on "Pausing…" withdraws the request, and one on Resume
// reads Pause at once: the worker carries on within its quarter-second poll.
// The label is the state, with no aria-pressed beside it: a toggle's label
// must not change, and "Resume, pressed" read as resumed. A running job of
// one cell has no boundary to pause at, and its Pause says so (syncPause).
// While the request is out, a status frame says what the server had before
// it - the stream's opening snapshot of a job just started lands after a
// press made at once - and left to relabel the button it read Pause again
// over a pause on its way, so the next press asked for the pause a second
// time instead of withdrawing it, and the job stayed paused (`pauseAsking`:
// the job and the `pause_requested` asked for, until the server answers).
let pauseAsking = null;

// A response belongs to the watched job and the known server it was sent to.
// Learning the first server identity is allowed; a confirmed restart is not.
function watchedActionCurrent(asking) {
  return busy && currentJob === asking.jobId
    && (asking.startedAt == null || serverStartedAt == null || asking.startedAt === serverStartedAt);
}

async function togglePause() {
  if (!busy || !currentJob) return;
  if (els.pause.getAttribute('aria-disabled') === 'true') {
    if (busy) showJobNote(`${els.pause.title}.`);
    return;
  }
  const jobId = currentJob;
  const was = els.pause.textContent;
  const pausing = was === 'Pause';
  els.pause.textContent = pausing ? 'Pausing…' : 'Pause';
  if (cellArgs) showCell(...cellArgs);
  mirrorJob();
  const asking = { jobId, value: pausing, startedAt: serverStartedAt };
  pauseAsking = asking;
  try {
    await api(`/api/jobs/${jobId}/pause?value=${pausing}`, { method: 'POST' });
  } catch (err) {
    // A later press can withdraw this pause before its reply arrives. Only
    // the latest press for this watched context may restore the old label.
    if (pauseAsking === asking && watchedActionCurrent(asking)) {
      els.pause.textContent = was;
      showError(err.message);
    }
  } finally {
    if (pauseAsking === asking) {
      pauseAsking = null;
      if (watchedActionCurrent(asking)) {
        if (cellArgs) showCell(...cellArgs);
        mirrorJob();
      }
    }
  }
}

// The label from the worker's own account, every status frame - so a page
// that reattached shows a pause already asked for - and the button held, with
// the reason, where the job is one cell that has started. Not the label from
// a frame that contradicts this page's own press still on its way
// (togglePause): the press set it, and the frame predates the request. One
// that agrees with it - the worker's `paused` landing first - is taken.
function syncPause(job) {
  const unit = unitWord(job);
  const stale = pauseAsking?.jobId === job.job_id && watchedActionCurrent(pauseAsking)
    && !!job.pause_requested !== pauseAsking.value;
  if (!stale) {
    els.pause.textContent = job.paused ? 'Resume' : job.pause_requested ? 'Pausing…' : 'Pause';
  }
  const single = job.status === 'running' && (job.total_runs || 1) === 1 && !job.paused;
  els.pause.setAttribute('aria-disabled', String(single));
  els.pause.title = single ? `A single ${unit} has no pause point — Cancel stops it`
    : job.status === 'queued' ? 'Hold this job before it starts; Resume lets it run'
    : `Hold after the current ${unit} and hand the GPU back; Resume carries on`;
}

// The job Cancel stops, named over the bars and in Cancel's own title: with a
// queue, the prompt box held another job's text, and Cancel stopped the one
// it did not show. One line, beside Pause and Cancel, which the panel's width
// may cut: the line's title has it whole, an XML prompt's markup and all.
function syncNow(job) {
  const what = describeJob(job);
  const foreign = foreignJobs.has(job.job_id);
  const lead = foreign ? `${ANOTHER_TABS_JOB}:` : 'Now:';
  els.progressNow.textContent = what ? `${lead} ${what}` : '';
  els.progressNow.title = what ? `${lead} ${describeJob(job, 120, { markup: true })}` : '';
  const name = job.edit != null ? `the instruction edit of #${job.edit.source ?? '?'}`
    : job.refine_batch != null ? 'this batch refine'
    : job.more_seeds != null ? 'these more seeds'
    : job.swap != null ? 'this swap'
    : job.refine != null ? `the ${job.outpaint ? 'outpaint' : job.mask ? 'inpaint' : 'refine'} of #${job.refine}`
    : job.grid ? (job.prompt_template != null ? 'this grid' : job.swap != null ? 'this swap' : 'this re-render')
    : `"${snippet(job.prompt ?? '', 40)}"`;
  els.cancel.title = `Stop ${name}${foreign ? " - another tab's job -" : ''} after the current step`;
}

// Desktop notifications are a separate browser preference, never a panel
// setting or a reason to ask permission at load. A terminal frame is consumed
// even when visible or opted out: replaying it after hiding the tab must not
// turn an already seen result into a new notification.
const NOTIFY_DONE_KEY = 'auradream.notify-done';
const NOTIFIED_JOBS_KEY = 'auradream.notified-jobs';
const notifiedJobs = new Set();
let notifyDoneEnabled = false;
let notifyPermissionPending = false;

function jobNotificationHold() {
  if (typeof window.Notification !== 'function' || typeof window.Notification.requestPermission !== 'function') {
    return 'Desktop notifications are unavailable in this browser';
  }
  if (!window.isSecureContext) return 'Desktop notifications need a secure page, such as http://127.0.0.1';
  if (window.Notification.permission === 'denied') return 'Notifications are blocked; allow them in the browser\'s site settings first';
  return notifyPermissionPending ? 'Waiting for notification permission' : '';
}

function syncJobNotifications(note = '') {
  const why = jobNotificationHold();
  if (notifyDoneEnabled && (why || window.Notification.permission !== 'granted')) {
    notifyDoneEnabled = false;
    remember(NOTIFY_DONE_KEY, 'false');
  }
  els.notifyDone.checked = notifyDoneEnabled || notifyPermissionPending;
  els.notifyDone.setAttribute('aria-disabled', String(!!why));
  els.notifyDone.setAttribute('aria-busy', String(notifyPermissionPending));
  const words = why || note || 'Hidden tabs only';
  if (els.notifyDoneNote.textContent !== words) els.notifyDoneNote.textContent = words;
  els.notifyDoneNote.title = words;
  els.notifyDoneLabel.title = why || 'Notify when this tab is hidden and its watched job ends';
  els.notifyDone.title = els.notifyDoneLabel.title;
}

function restoreJobNotifications() {
  notifyDoneEnabled = remembered(NOTIFY_DONE_KEY) === 'true';
  // Permission may have been revoked since the preference was saved. Never
  // ask again without the opt-in gesture, or leave a checked but silent tick.
  syncJobNotifications();
}

async function toggleJobNotifications() {
  if (notifyPermissionPending) { syncJobNotifications(); return; }
  if (!els.notifyDone.checked) {
    notifyDoneEnabled = false;
    remember(NOTIFY_DONE_KEY, 'false');
    syncJobNotifications();
    return;
  }
  const why = jobNotificationHold();
  if (why) {
    notifyDoneEnabled = false;
    remember(NOTIFY_DONE_KEY, 'false');
    syncJobNotifications();
    return;
  }
  let permission = window.Notification.permission;
  let note = '';
  if (permission !== 'granted') {
    notifyPermissionPending = true;
    syncJobNotifications();
    try {
      // Called before the first await, from the checkbox's own gesture.
      permission = await window.Notification.requestPermission();
      if (permission !== 'granted') note = 'Permission was not granted; tick again to ask';
    } catch {
      note = 'Notification permission could not be requested; try again in the browser';
    } finally {
      notifyPermissionPending = false;
    }
  }
  notifyDoneEnabled = permission === 'granted';
  remember(NOTIFY_DONE_KEY, String(notifyDoneEnabled));
  syncJobNotifications(note);
}

function jobNotificationWords(data, { atLoad = false, cancelledElsewhere = false } = {}) {
  const made = data.image_count ?? data.images?.length ?? 0;
  const skipped = data.skipped ? `${plural(data.skipped, unitWord(data))} skipped by request.` : '';
  const kept = `${plural(made, 'image')} made${data.deleted_count ? `; ${plural(data.deleted_count, 'image')} since deleted` : ''}.${skipped ? ` ${skipped}` : ''}`;
  if (data.stopped_by === 'budget') return ['Time budget reached', `${kept} Finish run checks what remains.`];
  if (data.stopped_by === 'shutdown') return ['Server stopped', kept];
  if (data.status === 'cancelled') return [cancelledElsewhere ? 'Cancelled elsewhere' : 'Cancelled', kept];
  if (data.status === 'error') return [atLoad ? 'Load failed' : 'Job stopped', `${kept} Open AuraDream for the error.`];
  if (!made) {
    const missing = [data.failures ? `${plural(data.failures, unitWord(data))} failed.` : '', skipped].filter(Boolean).join(' ');
    return ['Nothing made', missing || 'No images were made.'];
  }
  if (data.failures || data.error) return ['Finished with failures', `${kept} ${data.failures ? `${plural(data.failures, unitWord(data))} failed.` : 'Some work failed.'}`];
  return [`Finished ${plural(made, 'image')}`, kept];
}

function notifyJobDone(jobId, data, options = {}) {
  if (typeof jobId !== 'string' || !['done', 'error', 'cancelled'].includes(data.status)) return;
  // Read again for another tab that watched the same job. The browser's tag
  // also replaces the same job's notification where two tabs finish together.
  try {
    const saved = JSON.parse(remembered(NOTIFIED_JOBS_KEY) ?? '[]');
    if (Array.isArray(saved)) for (const id of saved) if (typeof id === 'string') notifiedJobs.add(id);
  } catch { /* malformed or unavailable store */ }
  if (notifiedJobs.has(jobId)) return;
  notifiedJobs.add(jobId);
  remember(NOTIFIED_JOBS_KEY, JSON.stringify([...notifiedJobs].slice(-50)));
  syncJobNotifications();
  if (!document.hidden || !notifyDoneEnabled || jobNotificationHold()) return;
  const [outcome, body] = jobNotificationWords(data, options);
  try {
    const notification = new window.Notification(`${PAGE_TITLE}: ${outcome}`, {
      body, tag: `auradream-job-${jobId}`,
    });
    notification.onclick = () => { window.focus(); notification.close(); };
  } catch {
    notifyDoneEnabled = false;
    remember(NOTIFY_DONE_KEY, 'false');
    syncJobNotifications('Notifications could not be shown in this browser');
  }
}
