// Local Diffusion's page script: generation requests, progress and queue controls.
// Declarations only; state.js has the shared-global load-order contract.
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

// The page moving focus itself where a job it submitted took it from - the
// Refine dialog closed on its run, the strip a job's first frame empties -
// to `to`, without moving the page. A sheet a job opens at its end takes
// focus along from the control pressed (sheet-render.js, focusSheet), so that
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
// pass queued behind another cannot paint over that one. `onRefused(err)`
// runs after a refusal has been said in `slot`, for the caller to say it in
// its own words. Resolves on a refusal, once the job has ended, or - for one
// queued behind the watched job - once the server has taken it.
async function startJob(path, payload, {
  sheet = false, panel = true, slot = els.error, placeholder = null, onAccepted = null, onRefused = null,
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
    onRefused?.(err);
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
