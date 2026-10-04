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
    mu: els.mu.value === '' ? null : +els.mu.value,
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
  // its Show run opening the older run.
  if (on) {
    els.lastRun.hidden = true;
    els.newImage.hidden = true;
  } else {
    els.progressText.textContent = '';
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
function describeJob(job, length = 60) {
  const what = job.refine != null ? `refine of #${job.refine}`
    : !job.grid ? snippet(job.prompt ?? '', length)
    : job.prompt_template != null ? `grid · ${snippet(job.prompt_template, length)}`
    : 're-render';
  return [
    job.planned_images ? plural(job.planned_images, 'image') : '',
    job.width && job.height ? `${job.width}×${job.height}` : '',
    what,
  ].filter(Boolean).join(' · ');
}

// At a word's end where there is one in the second half: "under a pine a…"
// named a word the prompt does not have.
function snippet(text, length = 60) {
  const line = text.trim().split(/\s+/).join(' ');
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
}

// The jobs waiting behind the watched one, from the ids their submissions
// returned - and, after a reload, from /api/jobs - one row each, named, with
// an × that cancels that one: "1 more job" named none of them, and Cancel
// queued was all or nothing. One waiting is one row, "Next:" under the Now
// line, with no heading over it: a heading and a row kept the queue below a
// 1040 px window. Rebuilt whole, with focus kept on the row it was on.
const queueCancelling = new Set();
function syncQueueNote() {
  const n = queuedIds.length;
  els.queueNote.hidden = !(busy && n > 0);
  els.queueText.textContent = n > 1 ? `${n} jobs queued behind this one:` : '';
  els.queueText.hidden = n < 2;
  // Its × is the whole choice for one job; Cancel queued is for several.
  els.cancelQueued.hidden = n < 2;
  const focused = document.activeElement?.closest?.('#queue-list li')?.dataset.id;
  els.queueList.replaceChildren(...queuedIds.map((id) => queueRow(id, n === 1)));
  if (focused) els.queueList.querySelector(`li[data-id="${focused}"] button`)?.focus();
}

function queueRow(jobId, alone) {
  const label = jobLabels.get(jobId) ?? 'a job submitted elsewhere';
  const li = document.createElement('li');
  li.dataset.id = jobId;
  const text = document.createElement('span');
  text.textContent = alone ? `Next: ${label}` : label;
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'ghost tiny danger';
  cancel.textContent = '×';
  cancel.title = 'Cancel this job before it starts';
  cancel.setAttribute('aria-label', `Cancel queued job: ${label}`);
  // Held while its POST runs, as Generate is during its own.
  cancel.setAttribute('aria-disabled', String(queueCancelling.has(jobId)));
  cancel.onclick = () => { void cancelQueuedJob(jobId); };
  li.append(text, cancel);
  return li;
}

async function cancelQueuedJob(jobId) {
  if (queueCancelling.has(jobId)) return;
  const row = [...els.queueList.children].findIndex((li) => li.dataset.id === jobId);
  const focused = els.queueList.children[row]?.contains(document.activeElement);
  queueCancelling.add(jobId);
  syncQueueNote();
  try {
    await api(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
  } catch (err) {
    // Gone already is what was asked for; no answer is not.
    if (err.offline) {
      queueCancelling.delete(jobId);
      syncQueueNote();
      showError(err.message);
      return;
    }
  }
  queueCancelling.delete(jobId);
  queuedIds = queuedIds.filter((id) => id !== jobId);
  forgetJob(jobId);
  syncQueueNote();
  // Focus where the row was: the next row's ×, else the one before, else Cancel.
  if (!focused) return;
  const left = els.queueList.querySelectorAll('li button');
  (left[row] ?? left[row - 1] ?? (busy ? els.cancel : els.generate)).focus();
}

// Generate's and Run grid's hold, one writer for both: the POST in flight
// (the double-submit guard the old `if (busy) return` used to be), Expand or
// the critic on the card, and a server that does not answer. aria-disabled,
// not disabled: a disabled element cannot hold focus, and the Enter that
// pressed Generate dropped a keyboard user's place; a press says why.
const GENERATE_TITLE = els.generate.title;
const GRID_RUN_TITLE = els.gridRun.title;
function syncSubmitButtons() {
  const held = String(submitting || expandingPrompt || critiquing || serverOffline);
  els.generate.setAttribute('aria-disabled', held);
  els.gridRun.setAttribute('aria-disabled', held);
  els.generate.title = serverOffline ? 'The server is not answering' : GENERATE_TITLE;
  els.gridRun.title = serverOffline ? 'The server is not answering' : GRID_RUN_TITLE;
}

function setSubmitting(on) {
  submitting = on;
  syncSubmitButtons();
  syncCompareRun();
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

function mirrorJob() {
  if (!busy) { els.jobMirror.textContent = ''; setTitle(''); return; }
  const text = els.progressText.textContent;
  const steps = els.stepRow.hidden ? '' : els.stepText.textContent;
  els.jobMirror.textContent = [text, steps, jobEta].filter(Boolean).join(' · ');
  const measured = jobFraction > 0 && !els.fill.parentElement.classList.contains('indeterminate');
  setTitle(els.pause.textContent === 'Resume' ? '⏸ Paused'
    : measured ? `${Math.round(jobFraction * 100)}% · ${text}` : text);
}

// A finished job's outcome stays in the title until the page is looked at
// again - the tab focused, a press or a key in it - so a tab left in the
// background says the night is over, and how it ended.
function holdTitle(text) {
  setTitle(text);
  const back = () => {
    window.removeEventListener('focus', back);
    document.removeEventListener('pointerdown', back, true);
    document.removeEventListener('keydown', back, true);
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
function showCell(data, { job = null, budget = null } = {}) {
  cellArgs = [data, { job, budget }];
  const unit = unitWord(job);
  const parts = [];
  if (job?.paused) {
    parts.push(job.run
      ? `Paused after ${unit} ${job.run}/${job.total_runs} · GPU handed back`
      : `Paused before the first ${unit} · GPU handed back`);
  } else if (els.pause.textContent === 'Pausing…') {
    parts.push(`pausing after this ${unit}…`);
  }
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

// Every submit button queues one job and then watches the same stream; only
// the endpoint and the payload differ. `sheet` opens the job as a sheet once
// its stream reports done - the prompt check's comparison and a refine
// strength sweep, whose answer is the sheet rather than the strip. `panel`
// says the payload is the panel's own, whose Seed box a grid writes back to.
// `slot` is where a refusal is said: Re-render's, from the gallery's head,
// in the gallery's line; a pass's, in the Refine dialog. `onAccepted(jobId)`
// runs once the server has taken the job, before it is watched - the dialog
// closes there, and a refusal leaves it open - and `placeholder` is what the
// viewer shows of it until its first frame (streamJob), keyed to the job, so
// a pass queued behind another cannot paint over that one.
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
  showError('');
  if (slot !== els.error) showError('', false, { slot });
  // Already watching a job: this one queues behind it on the server, and the
  // page picks it up when the watched one ends. The panel stays as it is.
  const queueing = busy;
  setSubmitting(true);
  if (!queueing) {
    setBusy(true);
    els.progressText.textContent = 'queued…';
    mirrorJob();
  }

  try {
    const { job_id, seed } = await api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setSubmitting(false);
    // The topbar's "loading…" now, not at the next poll, beside the panel's
    // own "loading weights".
    void refreshStatus();
    submissions.set(job_id, { path, payload, seed, sheet, at: Date.now() });
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
      queuedIds.push(job_id);
      syncQueueNote();
      void nameJob(job_id);
      return;
    }
    currentJob = job_id;
    await streamJob(job_id);
  } catch (err) {
    setSubmitting(false);
    showError(err.message, false, { slot });
    if (!queueing) setBusy(false);
  }
}

// Once the watched job ends, the next one in the server's queue - oldest
// first, since that is the one about to run - becomes the watched job.
async function watchNext() {
  if (busy) return;
  let active;
  try { active = await api('/api/jobs'); } catch { return; }
  if (!active.length) { queuedIds = []; syncQueueNote(); return; }
  const job = active.find((j) => j.status === 'running') ?? active[0];
  learnJobs(active);
  queuedIds = active.filter((j) => j.job_id !== job.job_id).map((j) => j.job_id);
  setBusy(true);
  currentJob = job.job_id;
  els.progressText.textContent = 'next in the queue…';
  syncQueueNote();
  await streamJob(job.job_id);
}

// Names for the jobs a listing returns that this page did not submit - a
// reload's, another tab's - keeping a submission's own where there is one.
function learnJobs(listed) {
  for (const job of listed) if (!jobLabels.has(job.job_id)) jobLabels.set(job.job_id, describeJob(job));
}

async function cancelQueued() {
  const ids = queuedIds.slice();
  for (const id of ids) {
    try { await api(`/api/jobs/${id}/cancel`, { method: 'POST' }); } catch { /* already gone */ }
    forgetJob(id);
  }
  queuedIds = queuedIds.filter((id) => !ids.includes(id));
  syncQueueNote();
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
    if (busy) showJobNote(els.pause.title);
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
// may cut: the line's title has it whole.
function syncNow(job) {
  const what = describeJob(job);
  els.progressNow.textContent = what ? `Now: ${what}` : '';
  els.progressNow.title = what ? `Now: ${describeJob(job, 120)}` : '';
  const name = job.refine != null ? `the refine of #${job.refine}`
    : job.grid ? (job.prompt_template != null ? 'this grid' : 'this re-render')
    : `"${snippet(job.prompt ?? '', 40)}"`;
  els.cancel.title = `Stop ${name} after the current step`;
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
    mu_axis: els.gridMu.value.trim(),
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
const PREVIEW_PROMPTS = 3;
function promptLines(prompts) {
  if (!prompts || !prompts.length) return [];
  if (prompts.length === 1) return [`prompt: ${prompts[0]}`];
  const lines = prompts.slice(0, PREVIEW_PROMPTS).map((text) => `· ${text}`);
  if (prompts.length > PREVIEW_PROMPTS) {
    lines.push(`· … and ${prompts.length - PREVIEW_PROMPTS} more`);
  }
  return lines;
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
    // status's state) is not ready, and while this page watches a job. Not
    // under Resume, whose seeds are no longer alike.
    let budgetShort = false;
    if (body.budget_minutes && p.seconds && !body.resume) {
      const seeds = p.values.seed?.length || body.seeds;
      const done = Math.min(seeds, Math.floor(body.budget_minutes * 60 / (p.seconds / seeds)));
      budgetShort = done === 0;
      const ahead = (busy ? 1 : 0) + queuedIds.length;
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
    els.gridPreview.replaceChildren(totals, [
      '',
      `steps ${axisText(p.values.steps)}  ·  CFG ${axisText(p.values.guidance_scale)}` +
        `  ·  mu ${axisText(p.values.mu)}` +
        // Only when it is an axis: every other run would carry a constant.
        // Named, since a sweep's weights mean nothing without the adapter.
        (p.values.lora_scale && p.values.lora_scale.length > 1
          ? `  ·  LoRA ${axisText(p.values.lora_scale)} (${els.lora.value.replace(/\.safetensors$/, '')})` : '') +
        (p.values.seed && p.values.seed.length ? `  ·  seeds ${axisText(p.values.seed)}` : ''),
      ...(body.prompt_axis ? promptLines(p.values.prompt) : []),
      ...notes,
    ].join('\n'));  // .hint.pre renders the breaks
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
  try {
    await api(`/api/jobs/${currentJob}/cancel`, { method: 'POST' });
    els.progressText.textContent = 'finishing this step…';
    mirrorJob();
  } catch (err) {
    showError(err.message);
    els.cancel.setAttribute('aria-disabled', 'false');
    els.cancel.textContent = 'Cancel';
  }
}

// `reattached` counts the times a stream that gave up found its job still
// running and opened again (gaveUp), so a stream that keeps failing on a live
// job stops trying and says so.
function streamJob(jobId, { reattached = 0 } = {}) {
  // The finish line on show is a job's before this one from here on, until
  // this one's own end replaces it - or for good, if its stream gives up.
  finishStale = true;
  syncFinishLead();
  return new Promise((resolve) => {
    const source = new EventSource(`/api/jobs/${jobId}/stream`);
    // Results replace the viewer only once the first image of this job lands,
    // so the previous one stays up through the queue wait and the model load.
    let cleared = false;
    let failures = 0;
    // Every snapshot carries the whole accumulated list, and a stream that
    // reattached missed the individual events, so images arrive by both routes.
    const shown = new Set();
    // The live preview: one <img> at the end of the strip, its src replaced
    // as frames arrive, gone the moment the real image lands. Never a record.
    // Or, for a pass, its stand-in (startJob's `placeholder`): the source,
    // dimmed, at the result's shape, captioned with the pass and its step,
    // from the job's first step until a frame or the result replaces it -
    // SDXL sends no frames, and the viewer showed an unrelated earlier image
    // for the whole of a pass.
    let previewEl = null;
    let caption = null;
    const placeholder = jobPlaceholders.get(jobId) ?? null;
    // The last status frame: what the job is (a grid's cells or a request's
    // repeats, its size, its budget) and what it waits on.
    let job = null;
    // The load's wait on the page's own clock, from the submission: the
    // server's `elapsed` starts only once the job runs, and minutes of a 0%
    // bar under "loading weights…" looked the same as a hang.
    let ticker = 0;
    const waitedFrom = submissions.get(jobId)?.at ?? Date.now();
    const dropPreview = () => {
      if (previewEl) { previewEl.remove(); previewEl = null; }
      caption = null;
    };
    const landed = () => {
      if (cleared) return;
      clearResults();
      cleared = true;
      offerNewImage();
    };
    const show = (rec) => {
      if (shown.has(rec.id)) return;
      shown.add(rec.id);
      landed();
      dropPreview();
      appendResult(rec);
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
      mirrorJob();
    };
    const stopTicker = () => { clearInterval(ticker); ticker = 0; };
    const note = (data) => showCell(data, {
      job,
      budget: job?.deadline && job.started_at ? job.deadline - job.started_at : null,
    });
    const standIn = () => {
      landed();
      const figure = document.createElement('figure');
      figure.className = 'pending';
      figure.style.setProperty('--w', placeholder.width);
      figure.style.setProperty('--h', placeholder.height);
      const img = document.createElement('img');
      img.src = placeholder.src;
      img.alt = '';
      caption = document.createElement('figcaption');
      figure.append(img, caption);
      els.viewer.appendChild(figure);
      previewEl = figure;
    };

    source.onmessage = (event) => {
      failures = 0;
      const data = JSON.parse(event.data);
      const runs = data.total_runs || 1;
      if (data.type === 'progress') {
        stopTicker();
        setIndeterminate(false);
        const done = (data.run || 1) - 1;
        const within = data.step / data.total_steps;
        setProgress((done + within) / runs);
        els.progressText.textContent = runs > 1
          ? `${unitWord(job)} ${data.run}/${runs}`
          : `${data.step}/${data.total_steps}`;
        setStep(runs > 1, within, `${data.step}/${data.total_steps}`);
        note(data);
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
        // to cancel a bad prompt rather than wait the render out.
        if (caption) dropPreview();
        landed();
        if (!previewEl) {
          previewEl = document.createElement('img');
          previewEl.className = 'preview';
          previewEl.alt = 'Live preview of the image being generated';
          els.viewer.appendChild(previewEl);
        }
        previewEl.src = data.image;
        previewEl.title = `live preview · step ${data.step}/${data.total_steps}`;
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
        if (!loading) {
          els.progressText.textContent =
            data.paused ? (runs > 1 && data.run ? `${unit} ${data.run}/${runs}` : 'paused')
            : data.status === 'queued' ? 'waiting for the job ahead…'
            : runs > 1 && data.run ? `${unit} ${data.run}/${runs}`
            : 'denoising…';
        }
        syncPause(data);
        syncNow(data);
        // A run boundary, so the steps bar starts over rather than sitting full
        // until the first callback of the next run lands.
        setStep(runs > 1 && data.status !== 'queued' && !!data.run);
        note(data);
        if (loading) {
          if (!ticker) ticker = setInterval(tick, 1000);
          tick();
          els.cellNote.textContent = 'The first load of a session takes a few minutes.';
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
        // Watched to its end, its finish line has said it: a reload's banner
        // would only say it again.
        remember(ANNOUNCED_KEY, jobId);
        finishJob(jobId, data, { submitted, sheet });
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
      let alive = null;
      let failure = null;
      try { alive = await api(`/api/jobs/${jobId}`); } catch (err) { failure = err; }
      // Still there: watch it again - events() opens with `done` for one
      // that ended meanwhile.
      if (alive && reattached < 3) return streamJob(jobId, { reattached: reattached + 1 });
      const grid = job?.grid;
      forgetJob(jobId);
      setBusy(false);
      const made = shown.size;
      const saved = made
        ? `The ${plural(made, 'image')} finished so far ${made === 1 ? 'is' : 'are'} saved.`
        : 'Nothing of it had finished.';
      if (alive) {
        showError('This job is still running, but its progress keeps getting lost: '
          + `reload the page to follow it. ${saved}`);
      } else if (failure.offline) {
        // The status poll's offline state takes it from here, and its first
        // answer takes this back.
        showError(`Can't reach the server — it may have stopped. ${saved} Start it again; the page will notice.`,
          false, { offline: true });
        void refreshStatus();
      } else if (failure.status === 404) {
        const planned = job?.planned_images;
        showError(made
          ? `The server restarted, and this job ended with it after ${made}${planned ? ` of ${planned}` : ''} `
            + `image${(planned ?? made) === 1 ? '' : 's'}. They are in the gallery; `
            + `${grid ? 'run the grid again with Resume ticked' : 'Generate again'} for the rest.`
          : 'The server restarted, and this job ended with it before any of it had finished.');
        void refreshStatus();
      } else {
        showError(`Lost this job's progress: ${failure.message}`);
      }
      holdTitle('Lost the job');
      if (made && !failure?.offline) { void prependFinished(made); loadRuns(); }
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
        els.progressText.textContent = `reconnecting… (${failures})`;
        mirrorJob();
        return;
      }
      source.close();
      void gaveUp().then(resolve);
    };
  });
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
// opens as its sheet.
function finishJob(jobId, data, { submitted, sheet }) {
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
    line.push(`Finished ${plural(made, 'image')}${deleted} in ${took}${seeds}.${distinct}`);
    if (plain) {
      line.push(noteButton('Use seed', () => useSeed(images[0].seed), `Put ${images[0].seed} in the Seed box`));
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
  // changed since - so only the missing cells run. Resume needs a fixed seed,
  // which the POST returned; a listed seed axis is its own.
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
          ...(submitted.payload.seed_axis ? {} : { seed: submitted.seed }),
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
  setFinish(line);

  if (data.status === 'error') {
    if (!atLoad) showOutcome(data.error);
  } else if (data.status === 'cancelled') {
    // A stop at the time budget is the plan working, not a cancel: say
    // which, and that a Resume finishes the rest. `run` is the cells
    // finished there - the budget is checked before a cell starts.
    const kept = `kept ${plural(images.length, 'image')}`;
    showOutcome(data.stopped_by === 'budget'
      ? `Time budget reached after ${data.run}/${data.total_runs} cells — ${kept}. Resume finishes the rest.`
      : data.waiting_on === 'load' ? 'Cancelled — the weights keep loading, so the next Generate starts sooner.'
      : images.length ? `Cancelled — ${kept}.` : 'Cancelled before anything finished.', true);
  } else if (data.error) {
    // Done, but not every cell: the grid carried on past a failure and says
    // which, rather than dropping the rest of the night.
    showOutcome(skippedCells(data));
  }
  holdTitle(data.status === 'done' ? `✓ Finished ${plural(made, 'image')}`
    : data.stopped_by === 'budget' ? 'Time budget reached'
    : data.status === 'cancelled' ? 'Cancelled'
    : atLoad ? 'Load failed' : 'Stopped');
  // The status says how the load went, at once rather than at the next poll.
  // Said again, where it goes to the panel's line: the press that queued
  // this job cleared that line, and a detail already posted once was
  // otherwise never posted again (syncLoadFailure).
  if (atLoad) shownLoadDetail = null;
  void refreshStatus();
  // A comparison opens this submission alone. Run ids name rendered
  // cells, so reversing original/yours on the same seeds shares a run
  // but must not mix the two meanings of either version label.
  if (sheet && images.length) { showJob(jobId, { fresh: true }); loadRuns(); }
  else if (images.length) { void prependFinished(made); loadRuns(); }
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
// whole run. The paging anchor - the last record - is untouched.
function prependFinished(made) {
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
    const have = new Set(galleryRecords.map((r) => r.id));
    if (!galleryRecords.length || !page.items.some((r) => have.has(r.id))) return resetGallery();
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
    els.gallery.prepend(...nodes);
    // The grid keeps one way in from the keyboard (gallery.js): the newest
    // image, where the stop was the grid's default or the old head with focus
    // elsewhere; a card the user chose further down keeps it.
    const current = currentCard();
    if (fresh.length && (!galleryChosen || (current === firstLive && !els.gallery.contains(document.activeElement)))) {
      makeCurrent(els.gallery.querySelector('.card[data-id]'));
    } else {
      keepCurrentCard();
    }
    total = page.total;
    els.count.textContent = total ? `(${total})` : '';
    els.otherModels.textContent = page.other_models ? ` (+${page.other_models})` : '';
    syncRerender();
    syncDeleteUnstarred();
    if (els.lightbox.open && lbList === galleryRecords) renderLightbox();
  });
}

// A job's first image or frame landing in a viewer the page is scrolled past -
// a user browsing the gallery during the wait - is said beside the progress,
// with a press that goes up to it. Never a scroll of the page's own, which
// would take the gallery from under the pointer. Gone once the viewer is in
// view, however it got there.
let viewerWatch = null;
function viewerOutOfSight() {
  const top = document.querySelector('.topbar').getBoundingClientRect().bottom;
  return els.viewer.getBoundingClientRect().bottom < top + 48;
}

function offerNewImage() {
  if (!viewerOutOfSight()) return;
  els.newImage.hidden = false;
  viewerWatch ??= new IntersectionObserver(() => {
    if (!els.newImage.hidden && !viewerOutOfSight()) els.newImage.hidden = true;
  }, { threshold: [0, 0.1, 0.25, 0.5, 1] });
  viewerWatch.observe(els.viewer);
}

// To the newest image, focused - Enter opens it - since the button goes the
// moment the viewer is in view.
function goToNewImage() {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  els.viewer.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  const images = els.viewer.querySelectorAll('img[data-id]');
  images[images.length - 1]?.focus({ preventScroll: true });
}

// A job outlives the page that started it: the id lives on the server, so a
// reload - or a stream that gave up - can find its way back to a running sweep
// instead of leaving it unreachable until the process is killed.
async function resumeActiveJob() {
  if (busy) return;
  try {
    const active = await api('/api/jobs');
    // The one holding the GPU, not the newest: with two live jobs the newest is
    // the queued one, and attaching to it made Cancel stop the wrong job while
    // the running grid kept the card with no reachable id. Failing a running
    // one, the *oldest* queued - the one about to run.
    const job = active.find((j) => j.status === 'running') ?? active[0];
    // Nothing live: say what the last one did instead, if the server still knows.
    if (!job) { announceLastRun(); return; }
    learnJobs(active);
    setBusy(true);
    currentJob = job.job_id;
    queuedIds = active.filter((j) => j.job_id !== job.job_id).map((j) => j.job_id);
    els.progressText.textContent = 'reattaching…';
    syncQueueNote();
    await streamJob(job.job_id);
  } catch { /* nothing running, or the server is not up yet */ }
}
