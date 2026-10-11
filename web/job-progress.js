// Local Diffusion's page script: the watched job's progress, mirrors and Pause.
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
  setTitle(els.pause.textContent === 'Resume' ? `⏸ Paused${whose ? ` · ${whose}` : ''}`
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
  return (job?.planned_images ?? 0) > (job?.total_runs || 1) ? 'batch' : 'image';
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
// `note` is what a queued job waits on, in words (the stream's status
// branch: the job ahead paused, the first load): kept in `cellArgs` with
// the rest, so a redraw - the queue resynced, a Pause pressed - says it
// again rather than wiping it under a progress line still saying "waiting".
function showCell(data, { job = null, budget = null, note = '', jobId = null } = {}) {
  cellArgs = [data, { job, budget, note, jobId }];
  const unit = unitWord(job);
  const parts = [];
  if (job?.paused) {
    parts.push(job.run
      ? `Paused after ${unit} ${job.run}/${job.total_runs} · GPU handed back`
      : `Paused before the first ${unit} · GPU handed back`);
    // The card's memory is handed back, but the queue does not move on: a
    // job behind a pause - one another tab queued included - waits for
    // Resume or Cancel here.
    if (queuedIds.length) {
      parts.push(`${queuedIds.length === 1 ? '1 job waits' : `${queuedIds.length} jobs wait`} behind it`);
    }
  } else if (els.pause.textContent === 'Pausing…') {
    parts.push(`pausing after this ${unit}…`);
  }
  if (note) parts.push(note);
  if (data.label) parts.push(`cell ${data.run}/${data.total_runs} · ${data.label}`);
  jobEta = '';
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
      timing += ` · ${jobEta}`;
    }
    parts.push(timing);
  }
  els.cellNote.textContent = parts.join(' · ');
  // A grid's skipped cells as they happen, in the failure colour: the line
  // used to go from cell 4 to cell 6 with no mark.
  const skipped = job?.failures ?? 0;
  if (skipped) {
    const span = document.createElement('span');
    span.className = 'skipped';
    span.textContent = `${parts.length ? ' · ' : ''}${skipped} skipped`;
    els.cellNote.append(span);
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
async function togglePause() {
  if (!currentJob) return;
  if (els.pause.getAttribute('aria-disabled') === 'true') {
    if (busy) showJobNote(`${els.pause.title}.`);
    return;
  }
  const was = els.pause.textContent;
  const pausing = was === 'Pause';
  els.pause.textContent = pausing ? 'Pausing…' : 'Pause';
  if (cellArgs) showCell(...cellArgs);
  mirrorJob();
  try {
    await api(`/api/jobs/${currentJob}/pause?value=${pausing}`, { method: 'POST' });
  } catch (err) {
    els.pause.textContent = was;
    showError(err.message);
  }
  if (cellArgs) showCell(...cellArgs);
  mirrorJob();
}

// The label from the worker's own account, every status frame - so a page
// that reattached shows a pause already asked for - and the button held, with
// the reason, where the job is one cell that has started.
function syncPause(job) {
  const unit = unitWord(job);
  els.pause.textContent = job.paused ? 'Resume' : job.pause_requested ? 'Pausing…' : 'Pause';
  const single = job.status === 'running' && (job.total_runs || 1) === 1 && !job.paused;
  els.pause.setAttribute('aria-disabled', String(single));
  els.pause.title = single ? `A single ${unit} has no pause point — Cancel stops it`
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
  const name = job.refine_batch != null ? 'this batch refine'
    : job.more_seeds != null ? 'these more seeds'
    : job.refine != null ? `the refine of #${job.refine}`
    : job.grid ? (job.prompt_template != null ? 'this grid' : job.swap != null ? 'this swap' : 'this re-render')
    : `"${snippet(job.prompt ?? '', 40)}"`;
  els.cancel.title = `Stop ${name}${foreign ? " - another tab's job -" : ''} after the current step`;
}
