// Local Diffusion's page script: generation requests and the job lifecycle.
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
    negative_prompt: negativeSent(els.negative),
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
  syncRefineBatchRun();
  syncMoreSeedsRun();
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
// name. A swap's are its records' too, and it is named by its rows; a
// batch refine's are passes of as many images, named by the scale and the
// strength they share; more seeds' continue batches, named by the first
// one's prompt.
function describeJob(job, length = 60, { markup = false } = {}) {
  const what = job.refine_batch != null ? `refine ×${job.refine_batch} at strength ${job.strength}`
    : job.more_seeds != null ? `more seeds · ${snippet(job.prompt ?? '', length, { markup })}`
    : job.swap != null ? `swap · ${snippet(job.swap, length, { markup: true })}`
    : job.refine != null ? `${job.outpaint ? 'outpaint' : job.mask ? 'inpaint' : 'refine'} of #${job.refine}`
      + (job.outpaint ? ` ${outpaintSidesText(job.outpaint)}` : '')
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
    const what = payload.ids ? plural(payload.ids.length, 'selected image') : "the gallery's selection";
    return `re-render of ${what} at ${payload.width}×${payload.height}`;
  }
  if (path === '/api/swap-tags') {
    return describeJob({ grid: true, swap: swapSummary(payload) });
  }
  if (path === '/api/refine/batch') {
    return describeJob({ grid: true, refine_batch: payload.scale, strength: payload.strength ?? refineDefaultStrength() });
  }
  if (path === '/api/more-seeds') {
    return `${plural(payload.count, 'more image')} of each batch in the gallery's selection`;
  }
  const grid = path === '/api/grid';
  return describeJob({
    ...payload,
    grid,
    prompt_template: grid ? payload.prompt : null,
    planned_images: grid ? null : payload.num_images * (payload.runs || 1),
  });
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

// The hidden line a screen reader hears a job's end on (#job-said), and a
// job joining the queue: written each time, the same words included, since
// each is an event of its own.
function sayJob(text) {
  els.jobSaid.textContent = text;
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
    : bundleHold() ? bundleHold()
    : modelChangeHold() ? modelChangeHold()
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
  // The Refine dialogs' Refine say the same while the server is gone, and
  // the comparison's Run as Generate does.
  syncRefineRun();
  syncInpaintRun();
  syncOutpaintRun();
  syncRefineBatchRun();
  syncMoreSeedsRun();
  // And the More seeds… buttons where a run ends, held while it is gone.
  syncMoreSeedsTools();
  syncCompareRun();
  syncModelSwitch();
  clearModelChangeHolds();
}

function setSubmitting(on) {
  submitting = on;
  syncSubmitButtons();
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
// in the gallery's line; a pass's, in the Refine dialog. `onAccepted(jobId,
// answer)` - the server's whole answer, where Re-render's names the picked
// images it left out - runs once the server has taken the job, before
// startJob watches it -
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
  model = loadedModel, revision = modelRevision,
} = {}) {
  if (submitting) return;
  if (modelChangeHold()) { sayModelChangeHold(slot); return; }
  const modelHold = submissionModelHold({ model, revision });
  if (modelHold) { showError(modelHold, false, { slot }); return; }
  // A saved request can run again after switching back to its checkpoint.
  // Bind this new submission to the current selection so another tab cannot
  // replace it between this check and the server accepting the request.
  if (model === loadedModel) revision = modelRevision;
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
  // A double holds every integer only up to 2^53 - 1: a Seed typed past it -
  // 9007199254740993, a 64-bit seed pasted from another tool - was sent as
  // its rounded neighbour, which then ran and was recorded while the box
  // still showed the seed typed. The server takes up to 2^63 - 1; the CLI
  // sends those exactly.
  if (Number.isInteger(payload.seed) && !Number.isSafeInteger(payload.seed)) {
    showError(`Seeds above ${Number.MAX_SAFE_INTEGER} cannot be sent exactly from the browser; use a smaller seed, or the CLI.`, false, { slot });
    return;
  }
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
    const answer = await api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(revision === null ? {} : { 'X-Local-Diffusion-Model-Revision': String(revision) }) },
      body: JSON.stringify(payload),
    });
    const { job_id, seed, more = [] } = answer;
    setSubmitting(false);
    ownJobs.add(job_id);
    // The topbar's "loading…" now, not at the next poll, beside the panel's
    // own "loading weights".
    void refreshStatus();
    // A submission the server split into several jobs (Swap tags: one per
    // size) answers the first as every other does and the rest in `more`:
    // this page's own, queued behind the first, each named by the server.
    // `split` is all of them in order, for each one's finish line to say
    // which part of the press it was (finishJob).
    const split = more.length ? [job_id, ...more.map((extra) => extra.job_id)] : null;
    submissions.set(job_id, { path, payload, seed, sheet, asker, model, revision, split, at: Date.now() });
    jobLabels.set(job_id, submissionLabel(path, payload));
    for (const extra of more) {
      ownJobs.add(extra.job_id);
      submissions.set(extra.job_id, { path, payload, seed: extra.seed, sheet: false, asker, model, revision, split, at: Date.now() });
      jobLabels.set(extra.job_id, submissionLabel(path, payload));
      void nameJob(extra.job_id);
    }
    // Read when they join the queue, which may have changed by then.
    const extras = () => more.map((extra) => extra.job_id).filter((id) => !queuedIds.includes(id));
    if (placeholder) jobPlaceholders.set(job_id, placeholder);
    if (sheet) jobsToShow.add(job_id);
    onAccepted?.(job_id, answer);
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
          setQueue([...queuedIds, job_id, ...extras()]);
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
    if (more.length) setQueue([...queuedIds, ...extras()]);
    await streamJob(job_id);
  } catch (err) {
    setSubmitting(false);
    showError(err.message, false, { slot });
    if (!queueing) setBusy(false);
    await dropRefusedReferences(err, path, payload, slot);
    // The Refine dialog's adapter is its own (dropRefineLora).
    if (slot !== els.refineError) await dropRefusedLora(err, path, payload, slot);
    onRefused?.(err);
  }
}

// What a refusal that took something out of the panel asks for next, named
// as what was pressed: the comparison's Run sends a grid of its versions.
function pressAgain(path, payload) {
  return payload.prompt_variants?.length ? 'Run the comparison again'
    : path === '/api/grid' ? 'Run the grid again' : 'Generate again';
}

// A submission refused for a reference the gallery no longer has - deleted
// or binned in another tab, where nothing here had marked it, or before a
// reload - is refused at every press until the chip goes: so it goes, by
// forgetImages, once the server says the image is unknown
// (dropGoneReferences), and the refusal says so in the page's words. One
// more press is then enough. The server names a binned one "reference image
// N", N the gallery id, not the chip's number: said as "#N" here.
async function dropRefusedReferences(err, path, payload, slot) {
  const named = /^no gallery image with id ([\d, ]+)$/.exec(err.message)
    ?? /^reference image ([\d, ]+) (?:is|are) in the recycle bin; /.exec(err.message);
  if (!named) return;
  const ids = named[1].split(',').map((id) => +id.trim()).filter((id) => payload.reference?.includes(id));
  const found = await dropGoneReferences(ids);
  const dead = found.ids;
  if (!dead.length) return;
  forgetReferences(found);
  // Unless something else has been said there meanwhile.
  if (slot.textContent !== err.message) return;
  const one = dead.length === 1;
  // Binned meanwhile, every one of them: in the recycle bin, not deleted.
  const went = found.binned.length === dead.length ? 'been moved to the recycle bin' : 'been deleted';
  showError(`Nothing was queued: reference ${imageNames(dead)} ${one ? 'has' : 'have'} ${went}, `
    + `so ${one ? 'it was' : 'they were'} taken out of the references. ${pressAgain(path, payload)} to run without ${one ? 'it' : 'them'}.`,
  false, { slot });
}

// The same for the panel's adapter: a file renamed or removed while the page
// was open stayed in LoRA, and every press was refused with the server's
// sentence for a developer (the directory's full path, the variable that
// moves it). A refusal that names the chosen adapter reads the list again
// (refreshLoras), which drops a name no longer offered; then the refusal
// says so. One the list still offers was refused for something else, and
// its sentence stands.
async function dropRefusedLora(err, path, payload, slot) {
  const name = payload.lora;
  if (!name || name !== els.lora.value || !err.message.includes(name)) return;
  const body = await refreshLoras();
  if (!body || els.lora.value === name) return;
  persistPanel();
  // Unless something else has been said there meanwhile.
  if (slot.textContent !== err.message) return;
  // Still in the directory, for another checkpoint: the tip says why.
  const why = (body.loras ?? []).some((e) => e.name === name)
    ? 'no longer fits this checkpoint (the LoRA tip says why)' : 'is no longer in the LoRA folder';
  showError(`Nothing was queued: ${name} ${why}, so it was taken out of LoRA. `
    + `${pressAgain(path, payload)} to run without it.`, false, { slot });
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

// The gallery's picked cards, else its current filter - the search box, the
// Starred tick and the checkpoint scope - rendered again at the panel's size
// (galleryActionTarget, as Swap tags…, Refine… and More seeds… take it). A
// filter is taken server-side, since the browser holds only the page it has
// scrolled to; the count in the button is the picks', or the server's total
// for the filter, so it is also the confirmation. A picked image binned or
// deleted since it was picked is left out by the server and named in its
// answer (`deleted`), said here and forgotten as every other gone id is.
function rerenderTarget() {
  return galleryActionTarget('to choose the images to render again', 'one re-render');
}
// The button's title on the filter (index.html's); on picks it names them.
const RERENDER_FILTER_TITLE = 'Render every image the filter matches again, at the size set in the panel - same prompt, settings and seed';

function rerenderSelection() {
  const target = rerenderTarget();
  const slot = { slot: els.galleryError };
  if (!target.scope) { showError(target.why, true, slot); return; }
  return startJob('/api/rerender', {
    ...target.scope, width: +els.width.value, height: +els.height.value,
  }, { slot: els.galleryError, onAccepted: (_, answer) => {
    sayRerenderGone(answer.deleted ?? []);
    const skipped = [
      answer.pruned ? `${answer.pruned} already rendered` : '',
      answer.planned ? `${answer.planned} already queued` : '',
    ].filter(Boolean).join(', ');
    if (skipped) {
      const gone = els.galleryError.textContent;
      showError([gone, `Re-render skipped ${skipped}.`].filter(Boolean).join(' '), true, { slot: els.galleryError });
    }
  } });
}

// The picks Re-render left out: "#7 is in the recycle bin", "#9 is no longer
// in the gallery" - the server's words - in the gallery's line, the job
// going ahead with the rest; and out of the page, as forgetImages takes any
// image the server has let go of.
function sayRerenderGone(gone) {
  if (!gone.length) return;
  const binned = gone.filter((item) => /recycle bin/.test(item.detail)).map((item) => item.id);
  const deleted = gone.filter((item) => !binned.includes(item.id)).map((item) => item.id);
  const also = (binned.length ? forgetImages(binned, { binned: true }) : '') + (deleted.length ? forgetImages(deleted) : '');
  const are = (ids) => (ids.length === 1 ? 'is' : 'are');
  const names = (ids) => (ids.length > 6 ? plural(ids.length, 'image') : imageNames(ids));
  const said = [
    binned.length ? `${names(binned)} ${are(binned)} in the recycle bin` : '',
    deleted.length ? `${names(deleted)} ${are(deleted)} no longer in the gallery` : '',
  ].filter(Boolean).join(', and ');
  showError(`Re-rendering the rest: ${said}.${also}`, true, { slot: els.galleryError });
}

// Coming and going with the pointer on the head's controls waits for it to
// leave them (showInHead); `now` for the bin's way in and out.
function syncRerender(now = false) {
  // Not in the recycle bin, whose count is not a render's (bin.js).
  const target = rerenderTarget();
  const show = !!target.scope;
  const size = `${els.width.value}×${els.height.value}`;
  const label = !show ? els.rerender.textContent
    : target.scope.ids ? `Re-render ${target.n} selected at ${size}` : `Re-render ${target.n} at ${size}`;
  // Its label is the confirmation, so it is never held back: under a
  // resting pointer the box keeps its width instead and the label ends in
  // an ellipsis there (gallery.css). "Re-render 3" to "Re-render 24" wrapped
  // Delete unstarred onto a row of its own at 1400 and slid Starred 130 px.
  const held = headPointer && !now;
  if (!held) els.rerender.style.width = '';
  else if (label !== els.rerender.textContent && !els.rerender.style.width
      && (!els.rerender.hidden || els.rerender.classList.contains('kept'))) {
    els.rerender.style.width = `${els.rerender.getBoundingClientRect().width}px`;
  }
  showInHead(els.rerender, show, now);
  if (show) {
    els.rerender.textContent = label;
    els.rerender.title = target.scope.ids
      ? `Render ${target.words} again, at the size set in the panel - same prompt, settings and seed`
      : RERENDER_FILTER_TITLE;
  }
  // Swap tags, Mass refine and More seeds take the same filter where nothing
  // is picked (tag-swap.js, refine-batch.js, more-seeds.js).
  syncSwapButton();
  syncRefineBatchButton();
  syncMoreSeedsButton();
}
