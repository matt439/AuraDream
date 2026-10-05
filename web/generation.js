// Local Diffusion's page script: submitting jobs, the grid preview and the SSE stream. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// --------------------------------------------------------------- generation

function collect() {
  const seed = els.seed.value.trim();
  // An empty box is "not given", which the server fills from the checkpoint's
  // profile - not 0, which `+''` produces and which the schema accepts for
  // CFG, so a cleared box used to generate unguided. Size stays coerced: the
  // presets always fill it, and a 422 beats a silent 1024.
  const num = (el) => (el.value.trim() === '' ? undefined : +el.value);
  return {
    prompt: els.prompt.value.trim(),
    // A greyed-out field has been told it cannot apply; sending its contents
    // anyway put a negative prompt the pipeline never saw into every record.
    negative_prompt: els.negative.disabled ? null : els.negative.value.trim() || null,
    width: +els.width.value,
    height: +els.height.value,
    steps: num(els.steps),
    guidance_scale: num(els.cfg),
    // Blank is off; a greyed-out box is a knob this checkpoint has not got.
    true_cfg: els.trueCfg.disabled || els.trueCfg.value.trim() === '' ? null : +els.trueCfg.value,
    // Blank is the pipeline's own shift; a greyed box is a knob this
    // checkpoint's schedule has not got, whatever a Reuse or a restore left.
    mu: els.mu.disabled || els.mu.value === '' ? null : +els.mu.value,
    // A greyed-out control has been told it cannot apply, exactly as with the
    // negative prompt: sending its contents anyway would put an adapter or a
    // reference the pipeline never saw into every record.
    reference: referenceSupported ? referenceIds : [],
    lora: els.lora.disabled ? null : els.lora.value || null,
    lora_scale: els.loraScale.value.trim() === '' ? 1 : +els.loraScale.value,
    // Never a refine pass: that is the Refine dialog's (refine.js), so
    // Generate, the grid and the comparison always render from the prompt.
    refine: null,
    seed: seed === '' ? null : +seed,
    num_images: +els.batch.value,
    runs: +els.runs.value,
  };
}

function setBusy(on) {
  // A job ending with focus on Pause, Cancel or a queued row's × - a keyboard
  // user who pressed Cancel and let the queue run out - lost it to <body> as
  // #progress and the queue hid. Generate holds it until the finish line is
  // drawn, which takes it on to its first button (setFinish).
  finishTakesFocus = !on && (els.progress.contains(document.activeElement)
    || els.queueNote.contains(document.activeElement));
  if (finishTakesFocus) els.generate.focus({ preventScroll: true });
  busy = on;
  streamLost = false;
  // A job of this page's own: whatever another tab runs, this page now
  // watches the queue it is part of (followServerJobs).
  if (on) setElsewhereBusy(false, { quiet: true });
  // The submit buttons stay live while a job runs: a second Generate or Run
  // grid queues behind it on the server. They are only held during the POST
  // itself (see startJob), as the double-submit guard.
  // Unloading mid-job frees nothing - the worker holds its own reference to the
  // pipeline - and the server refuses it with a 409 anyway (syncUnload).
  syncUnload();
  els.generate.textContent = on ? 'Generate (queue)' : 'Generate';
  els.gridRun.textContent = on ? 'Run grid (queue)' : 'Run grid';
  els.refineRun.textContent = on ? 'Refine (queue)' : 'Refine';
  els.progress.hidden = !on;
  els.cancel.setAttribute('aria-disabled', String(!on));
  els.cancel.textContent = 'Cancel';
  els.cancel.title = 'Stop after the current step';
  els.pause.setAttribute('aria-disabled', String(!on));
  els.pause.textContent = 'Pause';
  els.pause.title = 'Hold after the current cell and hand the GPU back; Resume carries on';
  setIndeterminate(false);
  setProgress(0);
  setStep(false);
  els.cellNote.textContent = '';
  els.progressNow.textContent = '';
  els.progressNow.title = '';
  jobEta = '';
  cellArgs = null;
  // The finished job's line stays - only the next job's end replaces it -
  // and says whose it is while another runs.
  syncFinishLead();
  // A job this page watches supersedes the morning after's banner: hidden,
  // not dismissed - the done branch remembers a job watched to its end - so
  // "Last run finished" no longer sits under a newer job's own finish line,
  // its Show run opening the older run. New image stays: the viewer still
  // holds what it points at until this job's first frame clears it (landed)
  // - and another tab's job, picked up after this page's, never does.
  if (on) {
    els.lastRun.hidden = true;
  } else {
    setProgressText('');
    currentJob = null;
    // What a press on a held Pause or Unload said was about this job.
    if (jobNote !== null && els.error.textContent === jobNote) showError('');
    jobNote = null;
  }
  els.jobMirror.hidden = els.topbarLine.hidden = !on;
  mirrorJob();
  syncQueueNote();
  syncPromptExpansion();
  // The grid preview reads both ends of a job: a budget's queue ahead, and
  // the cells in the gallery, which now hold a finished job's.
  schedulePreview();
}

// A job in a few words, from its snapshot or a submission shaped like one:
// how many images, at what size, and what it renders - the prompt's start,
// or the image a pass refines. The line over the bars and the queue's rows;
// the page used to name neither the job Cancel stops nor those waiting. The
// count and the size first: they are what tells two jobs apart, and the Now
// line is cut at its end, where the prompt - whole in its title - can go:
// put last, the count was the first thing cut, even at 2560. A grid's cells
// differ in their prompts, so it is named by its own, its {variables}
// unfilled - the first cell's named the whole night by a cell that had long
// finished - and a re-render's cells are its records', with no one prompt to
// name.
function describeJob(job, length = 60, { markup = false } = {}) {
  const what = job.refine != null ? `refine of #${job.refine}`
    : !job.grid ? snippet(job.prompt ?? '', length, { markup })
    : job.prompt_template != null ? `grid · ${snippet(job.prompt_template, length, { markup })}`
    : 're-render';
  return [
    job.planned_images ? plural(job.planned_images, 'image') : '',
    job.width && job.height ? `${job.width}×${job.height}` : '',
    what,
  ].filter(Boolean).join(' · ');
}

// A prompt's words on one line, without the markup of an XML prompt
// (NewBie's): its opening tags are the same in every prompt, so jobs and runs
// that differed in an attribute - blue_hair, green_hair - were all named
// "<character_1> <n>hatsune_miku</n> <gender>…", and a queued row's × did not
// say which job it cancelled. A tag is `<name>` or `</name>` and nothing else,
// so prose, Qwen's `<|…|>` control text and `<lora:…>` forms stay as written;
// a prompt of nothing but tags keeps them.
const MARKUP_TAG = /<\/?[A-Za-z_][\w.-]*>/g;
function promptGist(text, { markup = false } = {}) {
  const line = (s) => s.trim().split(/\s+/).join(' ');
  return (!markup && line(text.replace(MARKUP_TAG, ' '))) || line(text);
}

// At a word's end where there is one in the second half: "under a pine a…"
// named a word the prompt does not have. `markup` keeps an XML prompt's tags,
// for a title that has the prompt as written.
function snippet(text, length = 60, { markup = false } = {}) {
  const line = promptGist(text, { markup });
  if (line.length <= length) return line;
  const cut = line.slice(0, length - 1);
  const space = cut.lastIndexOf(' ');
  return `${(space > length / 2 ? cut.slice(0, space) : cut).replace(/[\s,;:.]+$/, '')}…`;
}

// The queue note's name for a job this page has just submitted, until the
// server's snapshot names it (nameJob): a re-render carries no prompt of its
// own, and a grid's image count is the server's to work out.
function submissionLabel(path, payload) {
  if (path === '/api/rerender') {
    return `re-render of the gallery's selection at ${payload.width}×${payload.height}`;
  }
  const grid = path === '/api/grid';
  return describeJob({
    ...payload,
    grid,
    prompt_template: grid ? payload.prompt : null,
    planned_images: grid ? null : payload.num_images * (payload.runs || 1),
  });
}

async function nameJob(jobId) {
  try { jobLabels.set(jobId, describeJob(await api(`/api/jobs/${jobId}`))); } catch { return; }
  syncQueueNote();
}

// Everything the page kept about a job that has ended or gone.
function forgetJob(jobId) {
  submissions.delete(jobId);
  jobLabels.delete(jobId);
  jobPlaceholders.delete(jobId);
  jobsToShow.delete(jobId);
  foreignJobs.delete(jobId);
}

// A job this page watches that it did not submit, while its viewer holds
// results of its own: another tab's, picked up through watchNext when this
// page's job ended. Its first frame used to clear the image just made from
// the viewer, and its finish line, Use seed and held title spoke of it as
// this page's. It is shown in the progress row only, named as another tab's
// there, in the mirrors and at its end, and its images join the gallery. A
// reload's attach, with nothing of this page's in the viewer, is not one:
// the page shows it as it always has. A POST of this page's still out may
// be the job listed, so nothing is decided then.
const foreignJobs = new Set();
const ANOTHER_TABS_JOB = "Another tab's job";
function isForeign(jobId) {
  if (foreignJobs.has(jobId)) return true;
  if (ownJobs.has(jobId) || submissions.has(jobId) || submitting) return false;
  if (!viewerRecords.some((rec) => ownJobs.has(rec.job_id))) return false;
  foreignJobs.add(jobId);
  return true;
}

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

// The hidden line a screen reader hears a job's end on (#job-said), and a
// job joining the queue: written each time, the same words included, since
// each is an event of its own.
function sayJob(text) {
  els.jobSaid.textContent = text;
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

// Generate's and Run grid's hold, one writer for both: the POST in flight
// (the double-submit guard the old `if (busy) return` used to be), Expand or
// the critic on the card, and a server that does not answer. aria-disabled,
// not disabled: a disabled element cannot hold focus, and the Enter that
// pressed Generate dropped a keyboard user's place; a press says why. The
// reason is the title, in syncUnload's order - held by Expand, the title was
// still "Generate (Ctrl+Enter)" and only a press said why - except for the
// POST's own moment, which keeps the plain one. Another tab's job holds
// neither: the server queues behind it.
const GENERATE_TITLE = els.generate.title;
const GRID_RUN_TITLE = els.gridRun.title;
function submitHold() {
  return serverOffline ? 'The server is not answering'
    : expandingPrompt ? 'Wait for prompt expansion to finish'
    : critiquing ? 'Wait for the critic to finish'
    : '';
}

function syncSubmitButtons() {
  const why = submitHold();
  const held = String(submitting || !!why);
  els.generate.setAttribute('aria-disabled', held);
  els.gridRun.setAttribute('aria-disabled', held);
  els.generate.title = why || GENERATE_TITLE;
  els.gridRun.title = why || GRID_RUN_TITLE;
  // The Refine dialog's Refine says the same while the server is gone, and
  // the comparison's Run as Generate does.
  syncRefineRun();
  syncCompareRun();
}

function setSubmitting(on) {
  submitting = on;
  syncSubmitButtons();
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
// so only the timing shows there. The estimate is the job's own: elapsed over
// the fraction of cells finished (the current cell's steps counting as a
// fraction of one), which tracks a run that has slowed where the gallery's
// per-step median would not. `elapsed` is the server's clock, so the two
// clocks need not agree - and so is a budget's end: `budget` is seconds from
// the job's start to its deadline, both the server's, and a budget that
// stops the job sooner than the estimate is the time left.
let cellArgs = null;
// `note` is what a queued job waits on, in words (the stream's status
// branch: the job ahead paused, the first load): kept in `cellArgs` with
// the rest, so a redraw - the queue resynced, a Pause pressed - says it
// again rather than wiping it under a progress line still saying "waiting".
function showCell(data, { job = null, budget = null, note = '' } = {}) {
  cellArgs = [data, { job, budget, note }];
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
  if (typeof data.elapsed === 'number' && data.elapsed >= 1) {
    const runs = data.total_runs || 1;
    const within = data.total_steps ? (data.step || 0) / data.total_steps : 0;
    const fraction = (((data.run || 1) - 1) + within) / runs;
    let timing = `${humanTime(data.elapsed)} elapsed`;
    if (fraction > 0.02 && fraction < 1) {
      const eta = data.elapsed * (1 - fraction) / fraction;
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

// The page moving focus itself where a job it submitted took it from - the
// Refine dialog closed on its run, the strip a job's first frame empties -
// to `to`, without moving the page. A sheet a job opens at its end takes
// focus along from the control pressed (sheet.js, focusSheet), so that
// control becomes `to` where focus was on it or it has gone: a sweep run
// from the viewer's ⤢ left focus on Generate, a screen above its sheet in
// one column, and the next Tab took the page back up to it.
function handJobFocus(to) {
  const from = document.activeElement;
  to.focus({ preventScroll: true });
  for (const submitted of submissions.values()) {
    if (submitted.asker && (submitted.asker === from || !isShown(submitted.asker))) submitted.asker = to;
  }
}

// Every submit button queues one job and then watches the same stream; only
// the endpoint and the payload differ. `sheet` opens the job as a sheet once
// its stream reports done - the prompt check's comparison and a refine
// strength sweep, whose answer is the sheet rather than the strip. `panel`
// says the payload is the panel's own, whose Seed box a grid writes back to.
// `slot` is where a refusal is said: Re-render's, from the gallery's head,
// in the gallery's line; a pass's, in the Refine dialog. `onAccepted(jobId)`
// runs once the server has taken the job, before startJob watches it -
// though a listing between jobs (`jobListing`) may be streaming it already -
// the dialog closes there, and a refusal leaves it open - and `placeholder`
// is what the viewer shows of it until its first frame (streamJob, which
// reads it again at each step until it is there), keyed to the job, so a
// pass queued behind another cannot paint over that one. Resolves on a
// refusal, once the job has ended, or - for one queued behind the watched
// job - once the server has taken it.
async function startJob(path, payload, {
  sheet = false, panel = true, slot = els.error, placeholder = null, onAccepted = null,
} = {}) {
  if (submitting) return;
  // Held while the status poll gets no answer: the press says why, and the
  // poll asks again at once, so a server already back lets the next through.
  if (serverOffline) { showError(OFFLINE_MESSAGE, false, { slot }); void refreshStatus(); return; }
  if (expandingPrompt) { showError('Wait for prompt expansion to finish.', false, { slot }); return; }
  // The critic's server has the card until it answers; a job would queue
  // behind it on the server anyway, but a click that seems to do nothing for
  // fifteen seconds reads as a lost one.
  if (critiquing) { showError('Wait for the critic to finish.', false, { slot }); return; }
  // A re-render carries no prompt of its own - every cell brings its record's.
  if ('prompt' in payload && !payload.prompt) { showError('Enter a prompt first.'); return; }
  // The control pressed, for a sheet the job opens at its end (finishJob).
  const asker = document.activeElement;
  showError('');
  if (slot !== els.error) showError('', false, { slot });
  // Already watching a job: this one queues behind it on the server, and the
  // page picks it up when the watched one ends. The panel stays as it is.
  // So it does while the listing that picks the next job is out: what the
  // page watches is that listing's to say, and this job may be in it.
  const queueing = busy || jobListing !== null;
  setSubmitting(true);
  if (!queueing) {
    setBusy(true);
    setProgressText('queued…');
    mirrorJob();
  }

  try {
    const { job_id, seed } = await api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setSubmitting(false);
    ownJobs.add(job_id);
    // The topbar's "loading…" now, not at the next poll, beside the panel's
    // own "loading weights".
    void refreshStatus();
    submissions.set(job_id, { path, payload, seed, sheet, asker, at: Date.now() });
    jobLabels.set(job_id, submissionLabel(path, payload));
    if (placeholder) jobPlaceholders.set(job_id, placeholder);
    if (sheet) jobsToShow.add(job_id);
    onAccepted?.(job_id);
    // A grid started from a blank Seed box drew its own base, and every image
    // it produces carries a *derived* seed - so the base is written back here,
    // where Resume will need it. Only for a grid: a plain Generate left blank is
    // asking for a fresh seed every click.
    // Not for a listed seed axis: the list is the record, and the first seed
    // of it written into the box would only mislead the next plain Generate.
    if (panel && path === '/api/grid' && !payload.seed_axis && els.seed.value.trim() === '' && seed != null) {
      els.seed.value = seed;
      pageSeed = els.seed.value;
      syncRuns();
      persistPanel();
    }
    if (queueing) {
      // What was true before the POST may not be by its answer: the watched
      // job can end meanwhile. A listing still out decides first - it may
      // have made this job the watched one - and a page then left watching
      // nothing takes this job on. Queued regardless, it ran unwatched, with
      // no progress and no finish line, its id kept to be listed as "Next:"
      // behind a later job; and one the listing had made the watched job
      // was listed as its own "Next:" row, whose × cancelled it.
      while (jobListing) await jobListing;
      if (watchedJobs.has(job_id)) {
        // The listing made it the watched job: its stream is the page's, and
        // this resolves at its end, as for a job the page took itself - an
        // `await startJob(...)` back to back with another resolved with the
        // job still running. One watched to its end before this answer
        // landed (finishJob had no submission to read) leaves nothing to
        // read what the answer recorded, which goes with it.
        const stream = jobStreams.get(job_id);
        if (stream) await stream;
        else forgetJob(job_id);
        return;
      }
      if (busy) {
        if (!queuedIds.includes(job_id)) {
          setQueue([...queuedIds, job_id]);
          void nameJob(job_id);
        } else {
          // Listed already by a status poll (resyncQueue), under the
          // server's name for it until now.
          syncQueueNote();
        }
        // The queue's rows are no live region: a screen reader hears that
        // the press queued something, and what.
        sayJob(`Queued: ${jobLabels.get(job_id)}`);
        return;
      }
      setBusy(true);
      setProgressText('queued…');
      mirrorJob();
    }
    currentJob = job_id;
    await streamJob(job_id);
  } catch (err) {
    setSubmitting(false);
    showError(err.message, false, { slot });
    if (!queueing) setBusy(false);
    await dropRefusedReferences(err, path, payload, slot);
  }
}

// A submission refused for a reference the gallery no longer has - deleted
// in another tab, where nothing here had marked it, or before a reload - is
// refused at every press until the chip goes: so it goes, by forgetImages,
// once the server says the image is unknown (dropGoneReferences), and the
// refusal says so in the page's words. One more press is then enough.
async function dropRefusedReferences(err, path, payload, slot) {
  const named = /^no gallery image with id ([\d, ]+)$/.exec(err.message);
  if (!named) return;
  const ids = named[1].split(',').map((id) => +id.trim()).filter((id) => payload.reference?.includes(id));
  const dead = await dropGoneReferences(ids);
  if (!dead.length) return;
  forgetImages(dead);
  // Unless something else has been said there meanwhile.
  if (slot.textContent !== err.message) return;
  const one = dead.length === 1;
  // The comparison's Run sends a grid of its versions: named as what was pressed.
  const again = payload.prompt_variants?.length ? 'Run the comparison again'
    : path === '/api/grid' ? 'Run the grid again' : 'Generate again';
  showError(`Nothing was queued: reference ${imageNames(dead)} ${one ? 'has' : 'have'} been deleted, `
    + `so ${one ? 'it was' : 'they were'} taken out of the references. ${again} to run without ${one ? 'it' : 'them'}.`,
  false, { slot });
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
  const name = job.refine != null ? `the refine of #${job.refine}`
    : job.grid ? (job.prompt_template != null ? 'this grid' : 'this re-render')
    : `"${snippet(job.prompt ?? '', 40)}"`;
  els.cancel.title = `Stop ${name}${foreign ? " - another tab's job -" : ''} after the current step`;
}

// Generate takes the prompt as typed, braces and all - a `{tag}` can be text or
// emphasis pasted from elsewhere, so the server never refuses one. But with
// the grid open and its variables filled in, a `{name}` in either prompt is
// the grid's, and Generate would render it literally as one image: that is
// Run grid's press. The same pattern as grid.PLACEHOLDER.
const GRID_PLACEHOLDER = /\{[A-Za-z_]\w*\}/;
function generate() {
  const body = collect();
  if (els.gridSection.open && els.gridPrompt.value.trim()
      && GRID_PLACEHOLDER.test(`${body.prompt}\n${body.negative_prompt ?? ''}`)) {
    showError('Run grid fills the {placeholders}; Generate would render the braces.');
    return;
  }
  return startJob('/api/generate', body);
}

// The gallery's current selection - the search box, the Starred tick and
// the checkpoint scope - rendered again at the panel's size. Taken
// server-side, since the browser holds only the page it has scrolled to; the
// count in the button is the server's total for the filter, so it is also
// the confirmation.
function rerenderSelection() {
  return startJob('/api/rerender', {
    search: els.search.value.trim() || null,
    favorites: els.favOnly.checked,
    all_models: els.allModels.checked,
    width: +els.width.value,
    height: +els.height.value,
  }, { slot: els.galleryError });
}

function syncRerender() {
  const filtered = els.favOnly.checked || els.search.value.trim() !== '';
  const show = filtered && total > 0 && total <= sheetPage;
  els.rerender.hidden = !show;
  if (show) {
    els.rerender.textContent =
      `Re-render ${total} at ${els.width.value}×${els.height.value}`;
  }
}

// ------------------------------------------------------------- grid search

// The axes sit on top of the panel's own settings, so a blank one contributes
// the value already there. `runs` is not a grid axis - the seed count is - and
// the server refuses a grid that sets both.
function gridPayload() {
  return {
    ...collect(),
    runs: 1,
    prompt_axis: els.gridPrompt.value.trim(),
    steps_axis: els.gridSteps.value.trim(),
    cfg_axis: els.gridCfg.value.trim(),
    mu_axis: els.gridMu.disabled ? '' : els.gridMu.value.trim(),
    lora_scale_axis: els.gridLoraScale.disabled ? '' : els.gridLoraScale.value.trim(),
    seeds: +els.gridSeeds.value || 1,
    seed_axis: els.gridSeedList.value.trim(),
    resume: els.gridResume.checked,
    budget_minutes: els.gridBudget.value.trim() === '' ? null : +els.gridBudget.value,
  };
}

function humanTime(seconds) {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${minutes} min`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

const axisText = (values) => values.map((v) => (v === null ? 'none' : v)).join(', ');
// Mu's none is the scheduler's own shift, which on a fixed-shift checkpoint
// is a number: said, as the sheet's heads say it.
const muAxisText = (values) => {
  const fixed = muStatic();
  return values.map((v) => (v !== null ? v : fixed === null ? 'none' : `none (${muStaticText(fixed)})`)).join(', ');
};

// The server's summary names every axis, the ones of one value included, and
// multiplies with an ASCII x: it is the CLI's log line too (grid.describe, held
// to that by its tests). The axes line under it lists every value anyway, so
// here a factor of one goes - all but the seeds, the count a night is planned
// around - and the sign is the comparison preview's ×.
function gridFactors(summary) {
  const kept = summary.split(' x ').filter((factor) => !factor.startsWith('1 ') || /seeds?$/.test(factor));
  return kept.join(' × ') || summary;
}

// The prompts a variable block resolved to, a few of them. Substitution is the
// half of a prompt grid that goes wrong quietly - a value that reads fine in the
// box lands badly in the middle of a sentence - so the panel shows the finished
// text rather than the variables that produced it. One prompt too, whenever the
// variables box has text: a block that failed to vary - `size = big, small` -
// is a grid of one prompt, and that prompt is where it shows.
// A prompt written over lines - NewBie's XML, 29 of them - is listed by the
// lines a {variable} is on, whole, so each value is read in its line: three
// copies of the whole prompt, one word apart, grew the preview to 2,260 px at
// QHD and put Run grid 2,200 px under the window. Its line numbers say the
// rest of the prompt is there. Values are one line each, and a dropped one
// leaves its line in place (grid.substitute), so line n of every resolved
// prompt is line n of the prompt typed; a prompt that is not - no line holds
// a variable, the variables are the negative's - is listed whole. Whatever
// the listing, it is a box of a few lines that scrolls inside itself
// (style.css): a nested scroller, never the panel, whose menus and tips hang
// out past its edge.
const PREVIEW_PROMPTS = 3;
function promptListing(prompts, template) {
  if (!prompts || !prompts.length) return null;
  const shown = prompts.slice(0, PREVIEW_PROMPTS);
  const lines = template.split('\n');
  const marked = lines.flatMap((line, i) => (GRID_PLACEHOLDER.test(line) ? [i] : []));
  const byLine = lines.length > 1 && marked.length > 0 && marked.length < lines.length
    && shown.every((text) => text.split('\n').length === lines.length);
  let rows;
  if (byLine) {
    const where = `${marked.length === 1 ? 'line' : 'lines'} ${marked.map((i) => i + 1).join(', ')} of ${lines.length}`;
    // Continuation lines are indented with no-break spaces, which pre-line
    // keeps; each line's own indentation is the prompt's, not the listing's.
    rows = [`${prompts.length === 1 ? 'prompt' : 'prompts'}, ${where}:`, ...shown.flatMap((text) => {
      const own = text.split('\n');
      return marked.map((i, k) => `${k ? '  ' : '· '}${own[i].trim()}`);
    })];
  } else {
    rows = prompts.length === 1 ? [`prompt: ${prompts[0]}`] : shown.map((text) => `· ${text}`);
  }
  if (prompts.length > PREVIEW_PROMPTS) rows.push(`· … and ${prompts.length - PREVIEW_PROMPTS} more`);
  const box = document.createElement('span');
  box.className = 'prompts';
  box.textContent = rows.join('\n');
  return box;
}

// A variable of one value never varies, which in a grid is always a slip - and
// most often commas, where every other box in the fold splits on them. Read
// off the box the way grid.parse_variables reads it (the first `=`, values
// split on `|`, blanks and repeats dropped), but only to warn: the server's
// parse is the one the run uses, and its errors are the ones shown.
function singleValueLines(text) {
  const lines = [];
  for (const line of text.split('\n')) {
    const match = /^\s*([A-Za-z_]\w*)\s*=(.*)$/.exec(line);
    if (!match) continue;
    const values = new Set(match[2].split('|').map((v) => v.trim().split(/\s+/).join(' ')).filter(Boolean));
    if (values.size !== 1) continue;
    const [value] = values;
    lines.push(value.includes(',')
      ? `${match[1]} has one value '${value}' - values split on |, not commas`
      : `${match[1]} has one value: it never varies`);
  }
  return lines;
}

// The preview is the server's own expansion rather than a count worked out in
// here: the syntax has one parser, and the message a bad axis produces is the
// one the run itself would have refused with.
// Preview replies can overtake each other across the server's one await (the
// duration lookup), so a slow reply for the boxes as they *were* could land
// after the reply for the boxes as they are. Same token the gallery uses.
let previewEpoch = 0;

// Opening the fold by hand brings Run grid into view on the next frame
// (revealGridRun), and the preview's reply lands after that, under the same
// pointer, where the panel holds still (panelResized): its lines pushed Run
// grid back below the window the opening had just brought it into. The
// summary's click sets this, and the first preview shown after it reveals
// Run grid again, as part of the same opening.
let gridRevealPending = false;
function previewShown() {
  if (!gridRevealPending) return;
  gridRevealPending = false;
  revealGridRun();
}

async function previewGrid() {
  if (!els.gridSection.open) {
    gridRevealPending = false;
    if (gridMaxCfg !== null) { gridMaxCfg = null; syncCfg(); }
    return;
  }
  const epoch = ++previewEpoch;
  // Nothing to preview before there is something to render: the server needs a
  // prompt to validate the request at all, and an untouched panel is not a
  // mistake to be reported.
  if (!els.prompt.value.trim()) {
    els.gridPreview.classList.remove('warn');
    els.gridPreview.textContent = 'Enter a prompt above to see what this grid would run.';
    previewShown();
    return;
  }
  const body = gridPayload();
  try {
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (epoch !== previewEpoch) return;  // superseded while in flight
    els.gridPreview.classList.remove('warn');
    // The axis decides the negative prompt's reach - see syncCfg.
    const reach = Math.max(...p.values.guidance_scale.filter((v) => typeof v === 'number'));
    gridMaxCfg = Number.isFinite(reach) ? reach : null;
    syncCfg();
    // Totals on the first line, the resolved axes on the second: what a range
    // actually expanded to is the half of it that is easy to get wrong.
    // The disk is the other thing a night runs out of; a size estimate past
    // the free space is the warning colour, since the night would die at the
    // first save that does not fit.
    const short = p.bytes && diskFreeMb !== null && p.bytes > diskFreeMb * 1024 * 1024;
    // Under the prompts, what would make the night not the one planned: cells
    // the gallery has already (Resume skips them; Run grid as it stands draws
    // them again), a variable that never varies, a budget that stops short,
    // and the panel's Repeat, which a grid does not take.
    const notes = [];
    if (p.in_gallery) {
      const n = p.in_gallery * body.num_images;
      notes.push(`${n} of these ${n === 1 ? 'is' : 'are'} already in the gallery - tick Resume to skip them`);
    }
    const single = body.prompt_axis ? singleValueLines(body.prompt_axis) : [];
    notes.push(...single);
    // A seed's share of the estimate, against the budget: every seed runs the
    // same cells (seeds are the outer loop), so the share is exact under the
    // estimate - and there is no line without one. The budget counts from Run
    // grid, so the load and any job ahead of it in the queue spend it too,
    // which the estimate does not count: said while the topbar's dot (the
    // status's state) is not ready, and while this page watches a job or
    // another tab's runs (elsewhereBusy). Not under Resume, whose seeds are
    // no longer alike.
    let budgetShort = false;
    if (body.budget_minutes && p.seconds && !body.resume) {
      const seeds = p.values.seed?.length || body.seeds;
      const done = Math.min(seeds, Math.floor(body.budget_minutes * 60 / (p.seconds / seeds)));
      budgetShort = done === 0;
      const ahead = (busy || elsewhereBusy ? 1 : 0) + queuedIds.length;
      const also = [
        els.dot.classList.contains('ready') ? '' : 'the load',
        ahead === 1 ? 'the job ahead' : ahead ? `the ${ahead} jobs ahead` : '',
      ].filter(Boolean);
      notes.push(`budget ${body.budget_minutes} min ≈ ${done} of ${seeds} seed${seeds === 1 ? '' : 's'} complete` +
        (also.length ? ` (${also.join(' and ')} count${also.length > 1 || ahead > 1 ? '' : 's'} too)` : ''));
    }
    if (+els.runs.value > 1) notes.push(`Repeat (${+els.runs.value}) does not apply to a grid: Seeds is its count`);
    // Cells that would come out identical - an axis this checkpoint ignores -
    // are the other reason to fix the grid before the night rather than after.
    if (short || p.duplicates || p.in_gallery || single.length || budgetShort) {
      els.gridPreview.classList.add('warn');
    }
    const totals = document.createElement('span');
    totals.className = 'totals';
    totals.textContent =
      `${p.images} image${p.images === 1 ? '' : 's'} · ${gridFactors(p.summary)}` +
        (p.skipped ? ` · ${p.skipped} already done` : '') +
        (p.duplicates ? ` · ${p.duplicates} would be identical to another cell` : '') +
        (p.seconds ? ` · about ${humanTime(p.seconds)}` : '') +
        (p.bytes ? ` · ~${humanBytes(p.bytes)}${short ? ` — more than the ${humanBytes(diskFreeMb * 1024 * 1024)} free` : ''}` : '');
    // .hint.pre renders the breaks. The prompts are a box of their own
    // between the axes and the notes (promptListing), on its own line.
    const listing = body.prompt_axis ? promptListing(p.values.prompt, body.prompt) : null;
    els.gridPreview.replaceChildren(
      totals,
      `\nsteps ${axisText(p.values.steps)}  ·  CFG ${axisText(p.values.guidance_scale)}` +
        `  ·  mu ${muAxisText(p.values.mu)}` +
        // Only when it is an axis: every other run would carry a constant.
        // Named, since a sweep's weights mean nothing without the adapter.
        (p.values.lora_scale && p.values.lora_scale.length > 1
          ? `  ·  LoRA ${axisText(p.values.lora_scale)} (${els.lora.value.replace(/\.safetensors$/, '')})` : '') +
        (p.values.seed && p.values.seed.length ? `  ·  seeds ${axisText(p.values.seed)}` : ''),
      ...(listing ? ['\n', listing] : []),
      ...(notes.length ? [`\n${notes.join('\n')}`] : []),
    );
  } catch (err) {
    if (epoch !== previewEpoch) return;
    els.gridPreview.classList.add('warn');
    els.gridPreview.textContent = err.message;
  }
  previewShown();
}

// The prompt check's comparison is a grid on the same panel, so whatever moves
// this preview's totals - the size, the steps, the seed - moves its too.
let gridTimer;
function schedulePreview() {
  clearTimeout(gridTimer);
  gridTimer = setTimeout(previewGrid, 250);
  scheduleComparePreview();
}

// The server keeps whatever finished before the cancel landed, so this reports
// a count rather than treating the run as lost.
async function cancelJob() {
  if (!currentJob || els.cancel.getAttribute('aria-disabled') === 'true') return;
  els.cancel.setAttribute('aria-disabled', 'true');
  els.cancel.textContent = 'Stopping…';
  // This page's own press: its end is said as a plain cancel (finishJob).
  const jobId = currentJob;
  cancelledHere.add(jobId);
  try {
    await api(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
    setProgressText('finishing this step…');
    mirrorJob();
  } catch (err) {
    cancelledHere.delete(jobId);
    showError(err.message);
    els.cancel.setAttribute('aria-disabled', 'false');
    els.cancel.textContent = 'Cancel';
  }
}

// What a queued job waits behind (`Job.waiting_on`), in the counter's words
// and, where it is not a job ahead running, the cell line's: a paused job
// keeps the gate, so a job queued from another tab read "waiting for the job
// ahead…" for as long as the pause lasted, and one submitted during another
// tab's critique named a job that did not exist. This page's own Expand and
// critic hold Generate, so either is another tab's.
const WAITING = {
  queue: { text: 'waiting for the job ahead…' },
  paused: {
    text: 'waiting: job ahead paused',
    note: 'The job ahead is paused: this one runs once it is resumed or cancelled, in the tab or window that started it.',
  },
  critic: { text: 'waiting for the critic…', note: "Another tab's prompt critic has the card; this job starts once it answers." },
  expand: { text: 'waiting for Expand…', note: "Another tab's prompt expansion has the card; this job starts once it is done." },
};

// `reattached` counts the times a stream that gave up found its job still
// running and opened again (gaveUp), so a stream that keeps failing on a live
// job stops trying and says so.
function streamJob(jobId, { reattached = 0 } = {}) {
  // The finish line on show is a job's before this one from here on, until
  // this one's own end replaces it - or for good, if its stream gives up.
  // Another tab's job never replaces it (finishJob), and gives back what it
  // was at its end.
  const staleBefore = finishStale;
  finishStale = true;
  syncFinishLead();
  // Watched: a queued submission's answer that lands after this does not
  // queue it, or take the page with it, again (startJob), and waits on this
  // stream instead (`jobStreams`, the first stream opened for the job, which
  // a reattached one's end resolves).
  watchedJobs.add(jobId);
  // Another tab's job, picked up over this page's own results: the progress
  // row only (isForeign).
  const foreign = isForeign(jobId);
  const stream = new Promise((resolve) => {
    const source = new EventSource(`/api/jobs/${jobId}/stream`);
    // Results replace the viewer only once the first image or frame of this
    // job lands, so the previous one stays up through the queue wait and the
    // model load - and the strip is then laid out for the whole job, the
    // frame in the box its image will take (panel.js, viewerTile), so the
    // gallery under it does not move when the image replaces it.
    let cleared = false;
    let failures = 0;
    // Every snapshot carries the whole accumulated list, and a stream that
    // reattached missed the individual events, so images arrive by both routes.
    const shown = new Set();
    // The live preview: one tile at the end of the strip, its picture's src
    // replaced as frames arrive, gone the moment the real image lands. Never
    // a record. Or, for a pass, its stand-in (startJob's `placeholder`): the
    // source, dimmed, at the result's shape, captioned with the pass and its
    // step, from the job's first step until a frame or the result replaces
    // it - SDXL sends no frames, and the viewer showed an unrelated earlier
    // image for the whole of a pass.
    let previewEl = null;
    let caption = null;
    // Read again at each step until it is there: a listing can make a
    // submission the watched job before its POST's answer has registered the
    // stand-in, which then comes up at the first step after that answer.
    let placeholder = jobPlaceholders.get(jobId) ?? null;
    // The last status frame: what the job is (a grid's cells or a request's
    // repeats, its size, its budget) and what it waits on.
    let job = null;
    // The load's wait on the page's own clock, from the submission: the
    // server's `elapsed` starts only once the job runs, and minutes of a 0%
    // bar under "loading weights…" looked the same as a hang.
    let ticker = 0;
    const waitedFrom = submissions.get(jobId)?.at ?? Date.now();
    // Focus on the preview's tile (goToNewImage) goes with it to what takes
    // its place - the image, or the preview that follows a pass's stand-in -
    // and to Generate where nothing does, never to <body>.
    let previewFocused = false;
    const dropPreview = () => {
      if (previewEl) {
        previewFocused ||= previewEl.contains(document.activeElement);
        previewEl.remove();
        previewEl = null;
        settleStrip(false);
      }
      caption = null;
    };
    const focusReplacement = (el) => {
      if (!previewFocused) return;
      previewFocused = false;
      focusTile(el ?? els.generate);
    };
    const landed = () => {
      if (cleared || foreign) return;
      // What New image pointed at goes with the strip: offered again below
      // for what takes its place, where that is out of sight too.
      withdrawNewImage();
      clearResults(job);
      cleared = true;
      offerNewImage();
    };
    const show = (rec) => {
      markGoneSource(rec);
      if (shown.has(rec.id)) return;
      shown.add(rec.id);
      if (foreign) return;
      landed();
      dropPreview();
      appendResult(rec);
      focusReplacement(els.viewer.querySelector(`img[data-id="${rec.id}"]`));
    };
    // The clock in a part a screen reader is not told about: #progress-text
    // is a status, and a clock rewritten in it every second was read out
    // every second for the minutes a load takes. "loading weights" is said
    // once; the topbar's mirror and the tab's title take the whole text.
    const tick = () => {
      const s = Math.max(0, Math.floor((Date.now() - waitedFrom) / 1000));
      let clock = els.progressText.querySelector('.clock');
      if (!clock) {
        els.progressText.textContent = 'loading weights';
        clock = document.createElement('span');
        clock.className = 'clock';
        clock.setAttribute('aria-hidden', 'true');
        els.progressText.append(clock);
      }
      clock.textContent = ` · ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
      // And in the viewer's line while the strip is empty (panel.js).
      viewerWaiting(clock.textContent);
      mirrorJob();
    };
    const stopTicker = () => { clearInterval(ticker); ticker = 0; viewerWaiting(null); };
    const note = (data, extra = '') => showCell(data, {
      job,
      budget: job?.deadline && job.started_at ? job.deadline - job.started_at : null,
      note: extra,
    });
    const standIn = () => {
      landed();
      const { figure, tile } = viewerTile('pending', placeholder.width, placeholder.height);
      const img = document.createElement('img');
      img.src = placeholder.src;
      img.alt = '';
      // Where the result's tools will be.
      caption = document.createElement('p');
      caption.className = 'foot';
      tile.append(img, caption);
      previewEl = figure;
      stripAppend(figure);
    };

    source.onmessage = (event) => {
      // Back after a reconnect: the strip's note is its own again.
      if (streamLost) { streamLost = false; noteViewer(); }
      failures = 0;
      const data = JSON.parse(event.data);
      const runs = data.total_runs || 1;
      if (data.type === 'progress') {
        stopTicker();
        setIndeterminate(false);
        const done = (data.run || 1) - 1;
        const within = data.step / data.total_steps;
        setProgress((done + within) / runs);
        // A single image's step count is shown, not said: "1/12" to
        // "12/12" was read out a step at a time. The run counter is said
        // as it changes, and the bar's aria-valuenow has the fraction.
        if (runs > 1) setProgressText(`${unitWord(job)} ${data.run}/${runs}`);
        else setProgressText('denoising…', `${data.step}/${data.total_steps}`);
        setStep(runs > 1, within, `${data.step}/${data.total_steps}`);
        note(data);
        placeholder ??= jobPlaceholders.get(jobId) ?? null;
        if (placeholder && !cleared && !previewEl) standIn();
        if (caption) {
          caption.textContent = `${placeholder.label} · step ${data.step}/${data.total_steps}`
            + (jobEta ? ` · ${jobEta.replace(/^~/, '')}` : '');
        }
        mirrorJob();
      } else if (data.type === 'image') {
        show(data.image);
      } else if (data.type === 'preview') {
        // A rough picture of the cell so far, from the latents: blurry, but
        // the composition is there from a few steps in, which is soon enough
        // to cancel a bad prompt rather than wait the render out. Not over
        // this page's own results for another tab's job.
        if (foreign) return;
        if (caption) dropPreview();
        landed();
        if (!previewEl) {
          previewEl = previewTile(job?.width, job?.height);
          // At the strip's end, which a long run's scrolling strip follows
          // from the preview on (panel.js).
          stripAppend(previewEl);
          focusReplacement(previewEl);
        }
        const frame = previewEl.querySelector('img');
        frame.src = data.image;
        frame.title = `live preview · step ${data.step}/${data.total_steps}`;
        previewEl.querySelector('.foot').textContent = `Live preview · step ${data.step}/${data.total_steps}`;
      } else if (data.type === 'status') {
        // Either the snapshot every stream opens with or the start of a run;
        // both carry whatever landed while nobody was listening.
        job = data;
        if (data.images) data.images.forEach(show);
        // A queued job says what it is behind: the weights on their way, or
        // another job. The first read "waiting for the job ahead…" for the
        // whole of a 32 GB download, under a top bar saying the opposite.
        const loading = data.status === 'queued' && data.waiting_on === 'load';
        const unit = unitWord(data);
        setIndeterminate(loading);
        // Paused, the counter stays the counter - "paused — GPU handed back"
        // in its place squeezed the bar to 5 px; the cell line says it.
        // Loading, the ticker below writes it, and leaves "loading weights"
        // as it is when it is there already.
        // Behind the gate, what holds it: a job ahead, one ahead that is
        // paused - which keeps the gate until it is resumed or cancelled,
        // where it was started - or another tab's critic or Expand, with no
        // job ahead at all (`waiting_on`).
        const waiting = data.status !== 'queued' ? null
          : data.waiting_on == null ? { text: 'queued…' } : WAITING[data.waiting_on] ?? WAITING.queue;
        if (!loading) {
          if (data.paused) setProgressText(runs > 1 && data.run ? `${unit} ${data.run}/${runs}` : 'paused');
          else if (waiting) setProgressText(waiting.text);
          else if (runs > 1 && data.run) setProgressText(`${unit} ${data.run}/${runs}`);
          // A single image's counter is the step count from here on, shown
          // beside these words rather than said (the progress branch).
          else setProgressText('denoising…', 'denoising…');
        }
        syncPause(data);
        syncNow(data);
        // A run boundary, so the steps bar starts over rather than sitting full
        // until the first callback of the next run lands.
        setStep(runs > 1 && data.status !== 'queued' && !!data.run);
        // What it waits on, in words, carried with the cell's own so a
        // redraw says it again (showCell).
        note(data, loading ? 'The first load of a session takes a few minutes.' : waiting?.note ?? '');
        if (loading) {
          if (!ticker) ticker = setInterval(tick, 1000);
          tick();
        } else {
          stopTicker();
        }
        mirrorJob();
      } else if (data.type === 'done') {
        source.close();
        stopTicker();
        const submitted = submissions.get(jobId);
        const sheet = jobsToShow.has(jobId);
        forgetJob(jobId);
        setBusy(false);
        dropPreview();
        if (data.images) data.images.forEach(show);
        focusReplacement(null);
        // Cancelled or failed after its first frame, before any image: the
        // frame that offered New image has gone, and the button offered an
        // image there was none of. A job that never reached the strip left
        // it as it was, and what New image points at with it.
        if (cleared && !shown.size) withdrawNewImage();
        // Watched to its end, its finish line has said it: a reload's banner
        // would only say it again.
        remember(ANNOUNCED_KEY, jobId);
        finishJob(jobId, data, { submitted, sheet, foreign, staleBefore });
        resolve();
        watchNext();
      }
    };

    // The stream has given up - five reconnects, or an answer that was no
    // stream - and whether the job went with it is the server's to say: "the
    // run carries on - reload to pick it up" was false when the server had
    // died, and after a restart a reload found nothing at all.
    const gaveUp = async () => {
      dropPreview();
      focusReplacement(null);
      let alive = null;
      let failure = null;
      try { alive = await api(`/api/jobs/${jobId}`); } catch (err) { failure = err; }
      // Still there: watch it again - events() opens with `done` for one
      // that ended meanwhile.
      if (alive && reattached < 3) return streamJob(jobId, { reattached: reattached + 1 });
      const grid = job?.grid;
      const submitted = submissions.get(jobId);
      forgetJob(jobId);
      setBusy(false);
      // The strip's note was the reconnect's line; it is the strip's again.
      noteViewer();
      const made = shown.size;
      if (cleared && !made) withdrawNewImage();
      const saved = made
        ? `The ${plural(made, 'image')} finished so far ${made === 1 ? 'is' : 'are'} saved.`
        : 'Nothing of it had finished.';
      // What the viewer says where nothing of the job reached it (lostRun).
      let lost = null;
      if (alive) {
        showError('This job is still running, but its progress keeps getting lost: '
          + `reload the page to follow it. ${saved}`);
      } else if (failure.offline) {
        // The status poll's offline state takes it from here, and its first
        // answer takes this back.
        const text = `Can't reach the server — it may have stopped. ${saved} Start it again; the page will notice.`;
        showError(text, false, { offline: true });
        lost = { heading: 'Lost the server', text };
        void refreshStatus();
      } else if (failure.status === 404) {
        const planned = job?.planned_images;
        const text = made
          ? `The server restarted, and this job ended with it after ${made}${planned ? ` of ${planned}` : ''} `
            + `image${(planned ?? made) === 1 ? '' : 's'}. They are in the gallery; `
            + `${grid ? 'run the grid again with Resume ticked' : 'Generate again'} for the rest.`
          : 'The server restarted, and this job ended with it before any of it had finished.';
        showError(text);
        lost = { heading: 'The server restarted', text };
        void refreshStatus();
      } else {
        const text = `Lost this job's progress: ${failure.message}`;
        showError(text);
        lost = { heading: 'Lost this job', text };
      }
      // In the topbar's words where the server is what went.
      holdTitle(failure?.offline ? 'Server not answering' : 'Lost the job');
      // Its first frame had cleared the strip and nothing of it landed: the
      // viewer was an empty box, the only words red in the panel.
      if (lost && cleared && !made) showLostRun(lost, submitted);
      // With another tab's images owed since this job took over (finishJob);
      // offline, the poll's first answer takes everything in (refreshStatus).
      if (!failure?.offline && (made || elsewhereCatchUp)) {
        elsewhereCatchUp = false;
        void prependFinished(made);
        loadRuns();
      }
    };

    source.onerror = () => {
      // Whatever the stream does next, it is not the load ticking on: the
      // ticker wrote "loading weights" over "reconnecting…" every second
      // while the topbar said the server was not answering. The snapshot a
      // reconnect opens with starts it again if the load goes on.
      stopTicker();
      // EventSource reconnects on its own, and the server replays a snapshot to
      // whoever attaches; close() cancels that retry for good, which is what
      // used to strand a running sweep behind an idle-looking page. Ride out a
      // few failures before asking the server what became of the job.
      if (source.readyState === EventSource.CONNECTING && ++failures <= 5) {
        // In the topbar's words, and nothing beside them that is no longer
        // true: the progress row said "reconnecting… (2)" under the topbar's
        // "server not answering", the step count and "~19s left" frozen
        // beside it, and the tab a third thing. The next frame that lands
        // brings the counts back (`showCell`, `setStep`, `noteViewer`).
        setProgressText(`${LOST_PHRASE} (${failures})`);
        setStep(false);
        jobEta = '';
        els.cellNote.textContent = '';
        if (failures === 1) {
          els.viewerNote.replaceChildren(`${LOST_PHRASE[0].toUpperCase()}${LOST_PHRASE.slice(1)}…`);
          void refreshStatus();
        }
        streamLost = true;
        mirrorJob();
        return;
      }
      source.close();
      void gaveUp().then(resolve);
    };
  });
  if (!jobStreams.has(jobId)) {
    jobStreams.set(jobId, stream);
    void stream.finally(() => { if (jobStreams.get(jobId) === stream) jobStreams.delete(jobId); });
  }
  return stream;
}

// The finished job's line, #finish-note: its outcome and what to do next -
// the seed and Use seed, Show sheet, a Retry or a Resume. A line of its own,
// which only the next job's end replaces: in the cell line, a job queued
// behind cleared it the moment the watched one ended, and a grid's Retry
// with it - in the overnight queue the button exists for. While another job
// runs, or after one whose stream gave up, it says whose it is. Focus on it
// when it is replaced moves to the new line's first button, else Generate,
// rather than to the page's body.
let finishStale = false;
// Set by setBusy where the ending job's Pause or Cancel had the focus, which
// Generate holds until this line is drawn.
let finishTakesFocus = false;
function setFinish(nodes = []) {
  const focused = els.finishNote.contains(document.activeElement)
    || (finishTakesFocus && document.activeElement === els.generate);
  finishTakesFocus = false;
  const lead = document.createElement('span');
  lead.className = 'lead';
  els.finishNote.replaceChildren(...(nodes.length ? [lead, ...nodes] : []));
  finishStale = false;
  syncFinishLead();
  if (focused) (els.finishNote.querySelector('button') ?? els.generate).focus({ preventScroll: true });
}

function syncFinishLead() {
  const lead = els.finishNote.querySelector('.lead');
  if (lead) lead.textContent = busy || finishStale ? 'Previous job: ' : '';
  // The finished job's sentence in the panel's line says whose it is too.
  els.error.classList.toggle('previous', busy && jobOutcome !== null && els.error.textContent === jobOutcome);
}

// The panel's line (#error) about the watched job and about the finished one.
// A press on a held Pause or Unload says why (`jobNote`), which goes when the
// job ends. A finished job's outcome - a cancel, its time budget, an error,
// the cells it skipped (`jobOutcome`) - lasts as its finish line does: led by
// "Previous job:" while another runs (#error.previous, style.css), and gone
// once the next job's end is drawn. Both used to stay until the next press of
// Generate, and under a queued job's own finish line "Cancelled before
// anything finished." read as that job's outcome.
let jobNote = null;
let jobOutcome = null;

function showJobNote(text) {
  showError(text, true);
  jobNote = text;
}

function showOutcome(text, note = false) {
  showError(text, note);
  jobOutcome = text;
}

// The finished job's outcome: the finish line, the panel's line for what
// went wrong, the tab's title, and the gallery. `submitted` is the
// submission this page made, which a Retry sends again; `sheet` a job that
// opens as its sheet; `foreign` another tab's job this page picked up
// (isForeign), whose end is said as that - no Use seed, no title held.
function finishJob(jobId, data, { submitted, sheet, foreign = false, staleBefore = false }) {
  const images = data.images ?? [];
  const made = data.image_count ?? images.length;
  // The job before this one's outcome goes with its finish line.
  if (jobOutcome !== null && els.error.textContent === jobOutcome) showError('');
  jobOutcome = null;
  // A job that ended at the load is the status's to say - the viewer's card,
  // the topbar's Retry load - which it says once, with the steps; the panel
  // line used to give the same six lines again.
  const atLoad = data.status === 'error' && !data.started_at;
  const line = [];
  // Its finish line says it was not made, though: the press had queued an
  // image, and the panel went back to Generate with no word of what became
  // of it. The reason is the viewer's card, or the panel's line under this
  // where the viewer holds images (syncLoadFailure).
  // Marked, so the status can reword it once the load is no longer failed
  // (syncLoadFailure): the card and the line it pointed at go then.
  if (atLoad) {
    const notMade = document.createElement('span');
    notMade.className = 'not-made';
    notMade.dataset.lead = foreign ? `${ANOTHER_TABS_JOB} was not made` : 'Not made';
    notMade.textContent = notMadeText(notMade.dataset.lead, 'error');
    line.push(notMade);
  }
  // Nothing made - every cell skipped - leaves the line to the skipped cells.
  if (data.status === 'done' && data.started_at && data.finished_at && made) {
    // Fewer distinct pictures than images is an axis that changed nothing,
    // said in one number rather than found on the sheet.
    const distinct = typeof data.distinct_count === 'number' && made > 1 && data.distinct_count < made
      ? ` Only ${data.distinct_count} distinct.` : '';
    // What is left of it: the count is what the job made, which a delete
    // since does not undo.
    const deleted = data.deleted_count ? `, ${data.deleted_count} since deleted,` : '';
    // A plain job's seeds, which a blank Seed box drew and nothing on the
    // page showed short of the lightbox: Use seed keeps the composition for
    // the next prompt, and writes nothing else.
    const plain = !data.grid && data.refine == null && images.length;
    const seeds = !plain ? ''
      : images.length === 1 ? `, seed ${images[0].seed}`
      : `, seeds ${images[0].seed}–${images.at(-1).seed}`;
    const took = humanTime(data.finished_at - data.started_at);
    if (foreign) {
      // Its seeds are another tab's to use; its images join the gallery.
      line.push(`${ANOTHER_TABS_JOB} finished: ${plural(made, 'image')}${deleted.replace(/,$/, '')}.`);
    } else {
      line.push(`Finished ${plural(made, 'image')}${deleted} in ${took}${seeds}.${distinct}`);
      if (plain) {
        line.push(noteButton('Use seed', () => useSeed(images[0].seed), `Put ${images[0].seed} in the Seed box`));
      }
    }
  }
  // A grid's sheet is its payoff, and the way to it was the Runs picker or a
  // card's ⊞. Offered, never opened: a night ending under a user browsing
  // the gallery must not take the gallery away.
  if (data.grid && images.length && !sheet) {
    line.push(noteButton('Show sheet', () => (data.run_id ? showRun(data.run_id) : showJob(jobId)),
      'Lay this run out as a contact sheet'));
  }
  // The cells a grid skipped, or did not reach: its own payload again with
  // Resume and the base seed it ran from - never the panel, which may have
  // changed since - so only the missing cells run. Resume needs a fixed seed:
  // the payload's own where it gave one, else the base the POST drew. Not the
  // POST's answer then, which is the first cell the job had to run: past the
  // base when the job was itself a Resume and its first seeds were done. A
  // listed seed axis is its own.
  const skipped = data.status === 'done' ? data.failures : 0;
  // Stopped by the rule: every cell but the ones that rendered.
  const remaining = data.stopped_by === 'failures' ? data.total_runs - (data.run - data.failures) : 0;
  if (submitted?.path === '/api/grid' && (skipped || remaining)) {
    const count = skipped || remaining;
    const button = noteButton(
      skipped ? `Retry the ${plural(skipped, 'skipped cell')}` : `Resume the remaining ${plural(remaining, 'cell')}`,
      () => {
        if (button.getAttribute('aria-disabled') === 'true') return;
        void startJob('/api/grid', {
          ...submitted.payload,
          resume: true,
          ...(submitted.payload.seed_axis ? {} : { seed: submitted.payload.seed ?? submitted.seed }),
        }, {
          panel: false,
          sheet: submitted.sheet,
          // Sent once: a second press would queue the same cells again. The
          // button stays where the press left focus, saying so, until the
          // job it sent ends and its own line replaces this one.
          onAccepted: () => {
            button.setAttribute('aria-disabled', 'true');
            button.textContent = skipped ? 'Retry sent' : 'Resume sent';
            button.title = 'Sent: it runs as a job of its own';
          },
        });
      },
      `Run the grid again with Resume: only the ${plural(count, 'missing cell')}`,
    );
    line.push(button);
  }
  // Another tab's job keeps out of this page's finish line: that line is
  // this page's own last job, Use seed and all, and the foreign job's end
  // is said once, on the hidden line, as the title is given back below.
  // The line the foreign job's stream marked as a previous job's is this
  // page's latest again.
  if (foreign) {
    finishTakesFocus = false;
    finishStale = staleBefore;
    syncFinishLead();
  } else {
    setFinish(line);
  }

  // A cancel this page did not ask for - pressed in another tab or window,
  // which watched the same job - was reported as this page's own.
  const cancelledElsewhere = !cancelledHere.has(jobId);
  cancelledHere.delete(jobId);
  const whose = foreign ? `${ANOTHER_TABS_JOB}: ` : '';
  if (data.status === 'error') {
    if (!atLoad) showOutcome(`${whose}${data.error}`);
  } else if (data.status === 'cancelled') {
    // A stop at the time budget is the plan working, not a cancel: say
    // which, and that a Resume finishes the rest. `run` is the cells
    // finished there - the budget is checked before a cell starts. A stop
    // at the server's shutdown is no one's cancel either.
    const kept = `kept ${plural(images.length, 'image')}`;
    const by = cancelledElsewhere ? ' from another tab or window' : '';
    showOutcome(whose + (data.stopped_by === 'budget'
      ? `Time budget reached after ${data.run}/${data.total_runs} cells — ${kept}. Resume finishes the rest.`
      : data.stopped_by === 'shutdown'
        ? `Stopped: the server is shutting down — ${images.length ? kept : 'nothing had finished'}.`
      : data.waiting_on === 'load' ? `Cancelled${by} — the weights keep loading, so the next Generate starts sooner.`
      : images.length ? `Cancelled${by} — ${kept}.` : `Cancelled${by} before anything finished.`), true);
  } else if (data.error) {
    // Done, but not every cell: the grid carried on past a failure and says
    // which, rather than dropping the rest of the night.
    showOutcome(`${whose}${skippedCells(data)}`);
  }
  // Said once to a screen reader where the panel's alert line has not said
  // the outcome already: the finish line's sentence, without its buttons. A
  // job that ended at the load has its reason there only where the viewer
  // holds images (syncLoadFailure); over the load card, which is no live
  // region, its "Not made" sentence is said here.
  const inAlert = data.status === 'cancelled'
    || (atLoad ? !!els.viewer.querySelector('img, figure, .tomb') : data.status === 'error' || !!data.error);
  if (!inAlert) {
    sayJob(line.filter((node) => node.nodeName !== 'BUTTON')
      .map((node) => (typeof node === 'string' ? node : node.textContent)).join('').trim());
  }
  // Another tab's job holds no title: the tab's own last result is what
  // its title was about, and gets it back where it has not been seen yet -
  // the foreign job's progress wrote over it (setBusy cleared it again).
  if (foreign) {
    if (titleHeld !== null) setTitle(titleHeld);
  } else {
    holdTitle(data.status === 'done' ? `✓ Finished ${plural(made, 'image')}`
      : data.stopped_by === 'budget' ? 'Time budget reached'
      : data.stopped_by === 'shutdown' ? 'Server stopped'
      : data.status === 'cancelled' ? (cancelledElsewhere ? 'Cancelled elsewhere' : 'Cancelled')
      : atLoad ? 'Load failed' : 'Stopped');
  }
  // The status says how the load went, at once rather than at the next poll.
  // Said again, where it goes to the panel's line: the press that queued
  // this job cleared that line, and a detail already posted once was
  // otherwise never posted again (syncLoadFailure).
  if (atLoad) shownLoadDetail = null;
  void refreshStatus();
  // A comparison opens this submission alone. Run ids name rendered
  // cells, so reversing original/yours on the same seeds shares a run
  // but must not mix the two meanings of either version label.
  // Another tab's images owed since this job took over (setElsewhereBusy)
  // come in with this job's, or alone where it made none.
  const owed = elsewhereCatchUp;
  elsewhereCatchUp = false;
  if (sheet && images.length) { showJob(jobId, { fresh: true, asker: submitted?.asker ?? null }); loadRuns(); }
  else if (images.length || owed) {
    void Promise.resolve(prependFinished(made)).then((coming) => noteOutsideView(images, coming));
    loadRuns();
  }
}

// The finish line's sentence for a job that ended at the load, by where the
// load stands now: failed, it points at the reason - the viewer's card with
// Retry load, or the panel's line where the viewer holds images; loading
// again, or loaded since, that card and line have gone, and it said "why,
// and Retry load, are in the viewer" over a ready topbar (syncLoadFailure);
// so it did after a restart of the server, which starts unloaded.
function notMadeText(lead, state) {
  const now = state === 'error'
    ? `- ${els.viewer.querySelector('img, figure, .tomb') ? 'why is said below' : 'why, and Retry load, are in the viewer'}`
    : state === 'loading' ? 'then; it is loading now'
    : state === 'unloaded' ? 'then; it is not loaded now'
    : 'then; it has loaded since';
  return `${lead}: the model could not load ${now}.`;
}

// A finished job's images the gallery's filter leaves out - a comparison's
// or a sweep's sheet still scoped to its job:, a search, Starred - said
// under the gallery's head with a way to them: Generate after a comparison
// left the 12 images of its sheet on screen, the count unchanged, as though
// nothing had been saved. Counted across jobs while the filter is the same
// (`outsideView`); the gallery's line clears with the next reset.
let outsideView = null;
function viewFilter() {
  return [els.search.value.trim(), els.favOnly.checked, els.allModels.checked, sheetMode].join('|');
}

// `coming`: the ids prependFinished's page holds, whose cards may still be
// waiting for the pointer to leave the gallery's column (joinFinished) - in
// this view all the same, and said at once.
function noteOutsideView(images, coming) {
  if (!els.search.value.trim() && !els.favOnly.checked) { outsideView = null; return; }
  const have = new Set([...galleryRecords.map((r) => r.id), ...(Array.isArray(coming) ? coming : [])]);
  const missing = images.filter((r) => !have.has(r.id)).map((r) => r.id);
  if (!missing.length) return;
  if (outsideView?.filter !== viewFilter()) outsideView = { filter: viewFilter(), ids: new Set() };
  missing.forEach((id) => outsideView.ids.add(id));
  const n = outsideView.ids.size;
  showError(`${plural(n, 'new image')} outside this view`, true, { slot: els.galleryError, mirror: false });
  const show = document.createElement('button');
  show.type = 'button';
  show.className = 'ghost tiny';
  show.textContent = 'Show all';
  show.title = 'Clear the search and Starred, and close the sheet, to see every image';
  show.onclick = showAllImages;
  els.galleryError.append(' · ', show);
}

// The gallery unfiltered: the search box emptied, Starred off, the sheet
// closed - the escape the scope note offers.
function showAllImages() {
  outsideView = null;
  els.search.value = '';
  els.favOnly.checked = false;
  sheetReturn = null;
  if (sheetMode) setSheet(false);
  else resetGallery();
  loadRuns();
}

// The cells a grid skipped, by name: which settings, which seed, and the
// error - "1 cell(s) failed" left the hole in the sheet to be found by eye.
function skippedCells(data) {
  const cells = data.failed_cells ?? [];
  if (!cells.length) return data.error;
  const named = cells.slice(0, 3).map((c) => [c.label, `seed ${c.seed}`].filter(Boolean).join(', '));
  const more = cells.length > 3 ? `; and ${cells.length - 3} more` : '';
  const last = cells.at(-1).error;
  return `${cells.length} of ${data.total_runs} cells ${cells.length === 1 ? 'was' : 'were'} skipped: `
    + `${named.join('; ')}${more} (${cells.length === 1 ? '' : 'last: '}${last})`;
}

function noteButton(text, onClick, title) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost tiny';
  button.textContent = text;
  button.title = title;
  button.onclick = onClick;
  return button;
}

// The finished image's seed into the Seed box, and nothing else: Reuse also
// rewrites the prompt and the size, and "same seed, another prompt" is the
// loop once something good turns up. The run hint then says it is fixed.
function useSeed(seed) {
  els.seed.value = seed;
  pageSeed = els.seed.value;
  syncRuns();
  schedulePreview();
  persistPanel();
}

// A finished job's images at the head of the gallery the page holds, under
// the same filters, rather than the gallery reset: that scrolled a user
// browsing old images during the wait back to the top, and dropped every page
// they had scrolled in, at every job's end. Asked once any page in flight has
// landed, and dropped if the filters changed meanwhile - the reset that
// changed them fetches the new images itself. One page reaches back past the
// job (`made` plus a page, the sheet's ceiling at most); where it does not
// join up with what the page holds, or the gallery is empty, the gallery is
// reset as before. Never in sheet mode, whose sheet lays itself out from the
// whole run. The paging anchor - the last record - is untouched. The head's
// count is drawn as the page lands and the cards by joinFinished, which can
// wait for the pointer; answers the ids that page holds (noteOutsideView).
// `now` puts them in without that wait: Delete unstarred brings the grid up
// to what it would sweep before asking. Its press was on the head's own
// button (or a key), not on the cards, and its modal question follows
// within one round trip, so no card moves under a pointer about to press it.
function prependFinished(made, { now = false } = {}) {
  if (sheetMode || !galleryRecords.length) return resetGallery();
  const epoch = galleryEpoch;
  return Promise.resolve(galleryPending).then(async () => {
    if (epoch !== galleryEpoch || sheetMode) return;
    const params = new URLSearchParams({ limit: Math.min(made + PAGE, sheetPage) });
    if (els.search.value.trim()) params.set('search', els.search.value.trim());
    if (els.favOnly.checked) params.set('favorites', 'true');
    if (els.allModels.checked) params.set('all_models', 'true');
    let page;
    try {
      page = await api(`/api/gallery?${params}`);
    } catch (err) {
      // No server, no gallery to reload: what the page holds stays, and the
      // poll's first answer asks again (refreshStatus).
      return err.offline ? undefined : resetGallery();
    }
    if (epoch !== galleryEpoch || sheetMode) return;
    if (!joinsUp(page)) return resetGallery();
    // What the head says, at once: it is above the cards, and with the
    // gallery beside the viewer above the column they scroll in.
    total = page.total;
    els.count.textContent = total ? `(${total})` : '';
    els.otherModels.textContent = page.other_models ? ` (+${page.other_models})` : '';
    galleryOthers = page.other_models || 0;
    syncRerender();
    syncDeleteUnstarred();
    if (els.lightbox.open && lbList === galleryRecords) renderLightbox();
    void joinFinished(page, epoch, now);
    // For noteOutsideView: the images this page holds are the gallery's, in
    // view or about to be.
    return page.items.map((r) => r.id);
  });
}

// Whether a page of the gallery's newest joins up with the cards it holds.
function joinsUp(page) {
  const have = new Set(galleryRecords.map((r) => r.id));
  return galleryRecords.length > 0 && page.items.some((r) => have.has(r.id));
}

// A finished job's cards, from the page prependFinished fetched, put in ahead
// of the gallery's. Beside the viewer, a pointer resting on the gallery's
// column keeps the cards where they are until it leaves (galleryPointerGone):
// one image ahead of them moves every card a place along the rows, which no
// scroll can take back. Only where there is something to put in.
async function joinFinished(page, epoch, now = false) {
  const newer = () => page.items.some((r) => r.id > galleryRecords[0].id && !galleryRecords.some((g) => g.id === r.id));
  if (!now && galleryRecords.length && newer()) await galleryPointerGone();
  if (epoch !== galleryEpoch || sheetMode) return;
  // Deleted from meanwhile, the gallery may no longer meet this page.
  if (!joinsUp(page)) { void resetGallery(); return; }
  const have = new Set(galleryRecords.map((r) => r.id));
  const head = galleryRecords[0];
  const fresh = page.items.filter((r) => r.id > head.id && !have.has(r.id));
  // In place: every card's lightbox walks this array. One open on it keeps
  // its image as the head grows.
  galleryRecords.unshift(...fresh);
  if (lbList === galleryRecords && fresh.length) lbIndex += fresh.length;
  // With the day's lines as loadGallery draws them - over the first card
  // where its day is not today's, between two new cards, and between the
  // last of them and the old head (a job that ends after midnight under a
  // page loaded the day before). The old head is the grid's first card as
  // the page shows it, a tombstone included - its record has gone from
  // galleryRecords, and the line worked out from the records stood a
  // second time over it - unless the grid opens with a line already, which
  // is the old head's.
  const top = els.gallery.firstElementChild;
  const oldHead = top?.matches('.gallery-break') ? null : top;
  const firstLive = els.gallery.querySelector('.card[data-id]');
  const nodes = [];
  fresh.forEach((rec, i) => {
    if (i === 0 && !isToday(rec)) nodes.push(dayBreak(rec));
    nodes.push(card(rec));
    const next = fresh[i + 1];
    if (next ? localDay(next) !== localDay(rec) : oldHead && cardDay(oldHead) !== localDay(rec)) {
      nodes.push(dayBreak(next ?? { created_at: oldHead.dataset.made }));
    }
  });
  holdGalleryView(() => els.gallery.prepend(...nodes));
  // The grid keeps one way in from the keyboard (gallery.js): the newest
  // image, where the stop was the grid's default or the old head with focus
  // elsewhere; a card the user chose further down keeps it.
  const current = currentCard();
  if (fresh.length && (!galleryChosen || (current === firstLive && !els.gallery.contains(document.activeElement)))) {
    makeCurrent(els.gallery.querySelector('.card[data-id]'));
  } else {
    keepCurrentCard();
  }
  syncGalleryEmpty();
  if (els.lightbox.open && lbList === galleryRecords) renderLightbox();
}

// A job's first image or frame landing in a viewer out of sight - above the
// window, with a user browsing the gallery during the wait, or below it, in
// one column, where the viewer sits under the whole panel and a Generate
// pressed at the page's top put the image 150-800 px under the window's foot
// - is said with a press that goes to it: "New image ↑" or "↓". Never a
// scroll of the page's own, which would take the gallery from under the
// pointer. In one column the button is a pill fixed under the topbar
// (style.css): in the static panel it was as far out of view as the viewer.
// Gone once the viewer is in view, however it got there; its arrow follows a
// scroll that jumps past the viewer without showing it (End, Home).
let viewerWatch = null;
function viewerOutOfSight() {
  const top = document.querySelector('.topbar').getBoundingClientRect().bottom;
  const box = els.viewer.getBoundingClientRect();
  return box.bottom < top + 48 ? 'above' : box.top > innerHeight - 48 ? 'below' : null;
}

const NEW_IMAGE_WORDS = {
  above: ['New image ↑', 'New image: scroll up to it in the viewer'],
  below: ['New image ↓', 'New image: scroll down to it in the viewer'],
};
function syncNewImage() {
  if (els.newImage.hidden) return;
  const where = viewerOutOfSight();
  if (!where) { withdrawNewImage(newImageTarget()); return; }
  const [text, label] = NEW_IMAGE_WORDS[where];
  if (els.newImage.textContent === text) return;
  els.newImage.textContent = text;
  els.newImage.setAttribute('aria-label', label);
}

function offerNewImage() {
  if (!viewerOutOfSight()) return;
  els.newImage.hidden = false;
  syncNewImage();
  if (!viewerWatch) {
    viewerWatch = new IntersectionObserver(syncNewImage, { threshold: [0, 0.1, 0.25, 0.5, 1] });
    window.addEventListener('scroll', syncNewImage, { passive: true });
    const panel = document.querySelector('.panel');
    panel.addEventListener('pointerdown', newImageBackToPanel, true);
    panel.addEventListener('input', newImageBackToPanel, true);
  }
  viewerWatch.observe(els.viewer);
}

// The pill (one column) lies over whatever scrolls under the topbar - the
// prompt's first line, Check - once the page has moved at all, and a press
// meant for that went to the viewer. While the job runs it stays: the image
// is still coming. Once the job has ended, the user's own press in the panel
// or typing in one of its boxes is a return to the panel, and withdraws it;
// the finish line still names the result. Not a focus arriving: the finish
// line takes focus itself, and a Tab towards the pill passes through the
// panel. In two columns the button is in the panel's flow, over nothing.
function newImageBackToPanel(event) {
  if (busy || els.newImage.hidden || els.newImage.contains(event.target)) return;
  if (getComputedStyle(els.newImage).position !== 'fixed') return;
  withdrawNewImage();
}

// The button goes - the viewer in view, a new job, or a job that ended with
// nothing of it in the strip - and focus on it goes with it to `to`, else
// Generate: hidden with the focus, it fell to <body>.
function withdrawNewImage(to = null) {
  if (els.newImage.hidden) return;
  if (document.activeElement === els.newImage) focusTile(to ?? els.generate);
  els.newImage.hidden = true;
}

// A job whose stream gave up with nothing of it in the strip: its first frame
// had cleared the strip, the live preview went with the stream, and the
// viewer was an empty box with the only words red in the panel - still empty
// once the server was back. A card says it where the work is, as a failed
// load's does (status.js), with the same request a press away where this
// page sent it; its refusal - the server still gone - is said in the card.
// The next job's first frame clears it; an answering server rewords it
// (`lostRunBack`).
function showLostRun({ heading, text }, submitted) {
  const card = document.createElement('div');
  card.className = 'lost-run';
  card.setAttribute('role', 'group');
  card.setAttribute('aria-labelledby', 'lost-run-head');
  const head = document.createElement('h2');
  head.id = 'lost-run-head';
  head.textContent = heading;
  const line = document.createElement('p');
  line.textContent = text;
  card.append(head, line);
  if (submitted) {
    const refused = document.createElement('p');
    refused.className = 'error';
    refused.setAttribute('role', 'alert');
    refused.hidden = true;
    const again = document.createElement('button');
    again.type = 'button';
    again.className = 'primary';
    again.textContent = 'Run it again';
    again.title = 'Send the same request again';
    again.onclick = () => {
      void startJob(submitted.path, submitted.payload, { sheet: submitted.sheet, panel: false, slot: refused });
    };
    card.append(again, refused);
  }
  els.viewer.replaceChildren(card);
  els.viewer.classList.remove('multi', 'many', 'fitted');
  els.viewer.style.minHeight = els.viewer.style.maxHeight = '';
  els.viewerNote.replaceChildren();
}

// The server answers again: the card says so rather than that it is gone.
function lostRunBack(restarted, s) {
  const card = els.viewer.querySelector(':scope > .lost-run');
  if (!card) return;
  const at = typeof s.started_at === 'number'
    ? new Date(s.started_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  card.querySelector('h2').textContent = restarted ? `The server restarted${at ? ` at ${at}` : ''}` : 'The server is back';
  card.querySelector('p').textContent = 'Nothing of the last run finished'
    + (card.querySelector('button') ? ': Run it again sends the same request.' : '.');
  const refused = card.querySelector('.error');
  if (refused) { refused.hidden = true; refused.textContent = ''; }
}

// What New image goes to: the newest image; before any has landed - the
// first frame is what offered it - the live preview's tile, or a pass's
// stand-in, which streamJob hands on to the image that replaces it; where
// the viewer holds neither (an image deleted elsewhere), its held line.
// Null where the viewer is empty: a stream that reconnects.
function newImageTarget() {
  const images = els.viewer.querySelectorAll('img[data-id]');
  return images[images.length - 1]
    ?? els.viewer.querySelector(':scope > figure.live, :scope > figure.pending')
    ?? els.viewer.querySelector(':scope > .placeholder.held:not([hidden]), :scope > .tomb');
}

// Focus on an image or a button, or on a tile, a line or the viewer that
// takes it only for this (tabindex -1). The viewer's own is let go with the
// focus: kept, a click on its empty space would focus it.
function focusTile(el) {
  if (el === els.viewer) {
    el.tabIndex = -1;
    el.addEventListener('blur', () => el.removeAttribute('tabindex'), { once: true });
  } else if (!el.matches('img, button')) {
    el.tabIndex = -1;
  }
  el.focus({ preventScroll: true });
}

// To the newest image, focused - Enter opens it - since the button goes the
// moment the viewer is in view: focus left on it fell to <body> as the
// pressed button hid. An empty viewer takes the focus itself.
function goToNewImage() {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  els.viewer.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  focusTile(newImageTarget() ?? els.viewer);
}

// A job outlives the page that started it: the id lives on the server, so a
// reload - or a stream that gave up - can find its way back to a running sweep
// instead of leaving it unreachable until the process is killed.
async function resumeActiveJob() {
  if (busy || jobListing) return;
  // No answer is a server not up yet: nothing to say.
  const jobId = await adoptListedJob('reattaching…');
  if (jobId) { await jobStreams.get(jobId); return; }
  // Nothing live: say what the last one did instead, if the server still
  // knows - not where a Generate pressed meanwhile has taken the page.
  if (jobId === null && !busy) announceLastRun();
}
