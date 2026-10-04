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
  busy = on;
  // The submit buttons stay live while a job runs: a second Generate or Run
  // grid queues behind it on the server. They are only held during the POST
  // itself (see startJob), as the double-submit guard.
  // Unloading mid-job frees nothing - the worker holds its own reference to the
  // pipeline - and the server refuses it with a 409 anyway.
  els.unload.disabled = on;
  els.generate.textContent = on ? 'Generate (queue)' : 'Generate';
  els.gridRun.textContent = on ? 'Run grid (queue)' : 'Run grid';
  els.refineRun.textContent = on ? 'Refine (queue)' : 'Refine';
  els.progress.hidden = !on;
  els.cancel.setAttribute('aria-disabled', String(!on));
  els.cancel.textContent = 'Cancel';
  els.pause.setAttribute('aria-disabled', String(!on));
  els.pause.textContent = 'Pause';
  setProgress(0);
  setStep(false);
  els.cellNote.textContent = '';
  if (!on) { els.progressText.textContent = ''; currentJob = null; }
  syncQueueNote();
  syncPromptExpansion();
  // The grid preview reads both ends of a job: a budget's queue ahead, and
  // the cells in the gallery, which now hold a finished job's.
  schedulePreview();
}

// The jobs waiting behind the watched one, from the ids their submissions
// returned - and, after a reload, from /api/jobs.
function syncQueueNote() {
  const n = queuedIds.length;
  els.queueNote.hidden = !(busy && n > 0);
  els.queueText.textContent = n ? `${n} more job${n === 1 ? '' : 's'} queued behind this one.` : '';
}

// Held during a POST, as the double-submit guard the old `if (busy) return`
// used to be. aria-disabled, not disabled: a disabled element cannot hold
// focus, and the Enter that pressed Generate dropped a keyboard user's place.
function setSubmitting(on) {
  submitting = on;
  els.generate.setAttribute('aria-disabled', String(on || expandingPrompt || critiquing));
  els.gridRun.setAttribute('aria-disabled', String(on || expandingPrompt || critiquing));
  syncCompareRun();
}

function setProgress(fraction) {
  els.fill.style.width = `${fraction * 100}%`;
  els.fill.parentElement.setAttribute('aria-valuenow', String(Math.round(fraction * 100)));
}

// Which cell of a grid is running, under the bars, with the time so far and an
// estimate of what is left. A sweep sends no label - its settings never change
// - so only the timing shows there. The estimate is the job's own: elapsed
// over the fraction of cells finished (the current cell's steps counting as a
// fraction of one), which tracks a run that has slowed where the gallery's
// per-step median would not. `elapsed` is the server's clock, so the two
// clocks need not agree.
function showCell(data) {
  const parts = [];
  if (data.label) parts.push(`cell ${data.run}/${data.total_runs} · ${data.label}`);
  if (typeof data.elapsed === 'number' && data.elapsed >= 1) {
    const runs = data.total_runs || 1;
    const within = data.total_steps ? (data.step || 0) / data.total_steps : 0;
    const fraction = (((data.run || 1) - 1) + within) / runs;
    let timing = `${humanTime(data.elapsed)} elapsed`;
    if (fraction > 0.02 && fraction < 1) {
      timing += ` · ~${humanTime(data.elapsed * (1 - fraction) / fraction)} left`;
    }
    parts.push(timing);
  }
  els.cellNote.textContent = parts.join(' · ');
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
async function startJob(path, payload, { sheet = false, panel = true } = {}) {
  if (submitting) return;
  if (expandingPrompt) { showError('Wait for prompt expansion to finish.'); return; }
  // The critic's server has the card until it answers; a job would queue
  // behind it on the server anyway, but a click that seems to do nothing for
  // fifteen seconds reads as a lost one.
  if (critiquing) { showError('Wait for the critic to finish.'); return; }
  // A re-render carries no prompt of its own - every cell brings its record's.
  if ('prompt' in payload && !payload.prompt) { showError('Enter a prompt first.'); return; }
  showError('');
  // Already watching a job: this one queues behind it on the server, and the
  // page picks it up when the watched one ends. The panel stays as it is.
  const queueing = busy;
  setSubmitting(true);
  if (!queueing) {
    setBusy(true);
    els.progressText.textContent = 'queued…';
  }

  try {
    const { job_id, seed } = await api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setSubmitting(false);
    if (sheet) jobsToShow.add(job_id);
    // A grid started from a blank Seed box drew its own base, and every image
    // it produces carries a *derived* seed - so the base is written back here,
    // where Resume will need it. Only for a grid: a plain Generate left blank is
    // asking for a fresh seed every click.
    // Not for a listed seed axis: the list is the record, and the first seed
    // of it written into the box would only mislead the next plain Generate.
    if (panel && path === '/api/grid' && !payload.seed_axis && els.seed.value.trim() === '' && seed != null) {
      els.seed.value = seed;
      syncRuns();
      persistPanel();
    }
    if (queueing) {
      queuedIds.push(job_id);
      syncQueueNote();
      return;
    }
    currentJob = job_id;
    await streamJob(job_id);
  } catch (err) {
    setSubmitting(false);
    showError(err.message);
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
  queuedIds = active.filter((j) => j.job_id !== job.job_id).map((j) => j.job_id);
  setBusy(true);
  currentJob = job.job_id;
  els.progressText.textContent = 'next in the queue…';
  syncQueueNote();
  await streamJob(job.job_id);
}

async function cancelQueued() {
  const ids = queuedIds.slice();
  for (const id of ids) {
    try { await api(`/api/jobs/${id}/cancel`, { method: 'POST' }); } catch { /* already gone */ }
  }
  queuedIds = queuedIds.filter((id) => !ids.includes(id));
  syncQueueNote();
}

// Pause holds the watched job at its next cell boundary and hands the card
// back; the same button resumes it. The worker's `paused` acknowledgement,
// not the click, is what flips the label - until it lands a denoise may
// still be running.
async function togglePause() {
  if (!currentJob || els.pause.getAttribute('aria-disabled') === 'true') return;
  const pausing = els.pause.textContent === 'Pause';
  try {
    const snap = await api(`/api/jobs/${currentJob}/pause?value=${pausing}`, { method: 'POST' });
    if (pausing && !snap.paused) els.progressText.textContent = 'pausing after this cell…';
    if (!pausing) els.pause.textContent = 'Pause';
  } catch (err) {
    showError(err.message);
  }
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

// The gallery's current selection - the search box, the Favorites tick and
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
  });
}

function syncRerender() {
  const filtered = els.favOnly.checked || els.search.value.trim() !== '';
  const show = filtered && total > 0 && total <= sheetPage;
  els.rerender.hidden = !show;
  if (show) {
    els.rerender.textContent =
      `↻ Re-render ${total} at ${els.width.value}×${els.height.value}`;
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
  } catch (err) {
    showError(err.message);
    els.cancel.setAttribute('aria-disabled', 'false');
    els.cancel.textContent = 'Cancel';
  }
}

function streamJob(jobId) {
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
    let previewEl = null;
    const dropPreview = () => { if (previewEl) { previewEl.remove(); previewEl = null; } };
    const show = (rec) => {
      if (shown.has(rec.id)) return;
      shown.add(rec.id);
      if (!cleared) { clearResults(); cleared = true; }
      dropPreview();
      appendResult(rec);
    };

    source.onmessage = (event) => {
      failures = 0;
      const data = JSON.parse(event.data);
      const runs = data.total_runs || 1;
      if (data.type === 'progress') {
        const done = (data.run || 1) - 1;
        const within = data.step / data.total_steps;
        setProgress((done + within) / runs);
        els.progressText.textContent = runs > 1
          ? `run ${data.run}/${runs}`
          : `${data.step}/${data.total_steps}`;
        setStep(runs > 1, within, `${data.step}/${data.total_steps}`);
        showCell(data);
      } else if (data.type === 'image') {
        show(data.image);
      } else if (data.type === 'preview') {
        // A rough picture of the cell so far, from the latents: blurry, but
        // the composition is there from a few steps in, which is soon enough
        // to cancel a bad prompt rather than wait the render out.
        if (!cleared) { clearResults(); cleared = true; }
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
        if (data.images) data.images.forEach(show);
        // A queued job says what it is behind: the weights on their way, or
        // another job. The first read "waiting for the job ahead…" for the
        // whole of a 32 GB download, under a top bar saying the opposite.
        els.progressText.textContent =
          data.paused ? 'paused — GPU handed back'
          : data.status === 'queued'
            ? (data.waiting_on === 'load' ? 'loading weights…' : 'waiting for the job ahead…')
          : runs > 1 && data.run ? `run ${data.run}/${runs}`
          : 'denoising…';
        els.pause.textContent = data.paused ? 'Resume' : 'Pause';
        // A run boundary, so the steps bar starts over rather than sitting full
        // until the first callback of the next run lands.
        setStep(runs > 1 && data.status !== 'queued' && !!data.run);
        showCell(data);
      } else if (data.type === 'done') {
        source.close();
        setBusy(false);
        dropPreview();
        if (data.images) data.images.forEach(show);
        // A finished run used to leave the panel indistinguishable from an
        // idle page. The summary goes in the cell line, which setBusy has
        // just cleared and which is outside the element it hides.
        if (data.status === 'done' && data.started_at && data.finished_at) {
          const made = data.images ? data.images.length : data.image_count;
          // Fewer distinct pictures than images is an axis that changed
          // nothing, said in one number rather than found on the sheet.
          const distinct = typeof data.distinct_count === 'number' && made > 1 && data.distinct_count < made
            ? ` Only ${data.distinct_count} distinct.` : '';
          els.cellNote.textContent =
            `Finished ${made} image${made === 1 ? '' : 's'} in ${humanTime(data.finished_at - data.started_at)}.${distinct}`;
        }
        if (data.status === 'error') showError(data.error);
        else if (data.status === 'cancelled') {
          // A stop at the time budget is the plan working, not a cancel: say
          // which, and that a Resume finishes the rest.
          const kept = `kept ${data.images.length} image${data.images.length === 1 ? '' : 's'}`;
          showError(data.stopped_by === 'budget'
            ? `Time budget reached after ${data.run - 1 || 0}/${data.total_runs} cells — ${kept}. Resume finishes the rest.`
            : data.images.length ? `Cancelled — ${kept}.` : 'Cancelled before anything finished.', true);
        } else if (data.error) {
          // Done, but not every cell: the grid carried on past a failure and
          // says so, rather than dropping the rest of the night.
          showError(data.error);
        }
        // A comparison opens this submission alone. Run ids name rendered
        // cells, so reversing original/yours on the same seeds shares a run
        // but must not mix the two meanings of either version label.
        const showSheet = jobsToShow.delete(jobId);
        if (showSheet && data.images.length) { showJob(jobId, { fresh: true }); loadRuns(); }
        else if (data.images.length) { resetGallery(); loadRuns(); }
        resolve();
        watchNext();
      }
    };

    source.onerror = () => {
      // EventSource reconnects on its own, and the server replays a snapshot to
      // whoever attaches; close() cancels that retry for good, which is what
      // used to strand a running sweep behind an idle-looking page. Ride out a
      // few failures before giving up - and even then the job is still going,
      // so a reload finds it again through /api/jobs.
      if (source.readyState === EventSource.CONNECTING && ++failures <= 5) {
        els.progressText.textContent = `reconnecting… (${failures})`;
        return;
      }
      source.close();
      setBusy(false);
      showError('Lost the generation stream. The run carries on — reload to pick it up.');
      resolve();
    };
  });
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
    setBusy(true);
    currentJob = job.job_id;
    queuedIds = active.filter((j) => j.job_id !== job.job_id).map((j) => j.job_id);
    els.progressText.textContent = 'reattaching…';
    syncQueueNote();
    await streamJob(job.job_id);
  } catch { /* nothing running, or the server is not up yet */ }
}
