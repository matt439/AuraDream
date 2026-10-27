// Auradream's page script: the last-run banner, run picker and opening a sheet.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------ the morning after

// localStorage is a per-browser convenience here and nothing more: it can be
// absent (a private window), blocked, or throw on access, and the page has to
// work the same without it.
function remembered(key) { try { return localStorage.getItem(key); } catch { return null; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch { /* unavailable */ } }
function forget(key) { try { localStorage.removeItem(key); } catch { /* unavailable */ } }

// Retained runs are the server's definition, independent of the composer or
// this tab's submission history. Every surface shares these availability
// answers and one submission path. A response belongs to its server/model
// context and request entry; a late answer never revives an old control.
const finishRuns = new Map();
const finishPresses = new Set();
let finishServerStamp = '';
const finishIdentity = () => JSON.stringify([serverStartedAt, modelRevision, loadedModel]);
const finishContext = () => JSON.stringify([finishIdentity(), finishServerStamp]);
const finishTools = () => [...document.querySelectorAll('button.finish-run-tool[data-run]')];

function finishRunWords(run, preview) {
  if (!preview?.available) return preview?.reason || 'Checking Finish run.';
  const count = `${plural(preview.images, 'image')} in ${plural(preview.cells, 'missing cell')}`;
  const redraw = preview.redrawn ? `; ${plural(preview.redrawn, 'permanently deleted cell')} will be redrawn` : '';
  const budget = preview.budget_minutes ? `; original ${preview.budget_minutes}-minute budget` : '';
  return `Finish ${count} of run ${run}${redraw}${budget}`;
}

function finishRunHold(button, state = finishRuns.get(button.dataset.run)) {
  return button.dataset.finishSent ? 'Sent: it runs as a job of its own'
    : binMode ? BIN_HELD
    : submitHold() || (submitting ? 'A job is being submitted' : '')
    || (finishPresses.has(button.dataset.run) ? 'Checking Finish run.' : '')
    || (state?.pending || !state ? 'Checking Finish run.' : '')
    || state.error || (!state.preview?.available ? state.preview?.reason || 'Finish run is unavailable' : '');
}

function syncFinishRunButton(button) {
  const state = finishRuns.get(button.dataset.run);
  if (button.dataset.finishSent && !els.finishNote.contains(button) && state?.details && !state.pending
    && !state.preview?.reason?.startsWith('the rest is queued in job ')) {
    // A sheet control survives its job's finish line. Once that job ends,
    // allow the server's next answer (including another stop) to decide it.
    delete button.dataset.finishSent;
  }
  const why = finishRunHold(button, state);
  button.setAttribute('aria-disabled', String(!!why));
  button.title = why || finishRunWords(button.dataset.run, state.preview);
  button.setAttribute('aria-description', button.title);
  if (!button.dataset.finishSent) button.textContent = 'Finish run';
}

function syncFinishRunButtons() {
  finishTools().forEach(syncFinishRunButton);
  syncMoreSeedsTools();
  // Counts follow the buttons, on the sheet's existing last row, so a
  // changing answer never moves a control under the pointer.
  for (const line of els.sheetControls.querySelectorAll('.sheet-run-status[data-run]')) {
    const state = finishRuns.get(line.dataset.run);
    const preview = state?.preview;
    const extra = preview?.available
      ? `${plural(preview.cells, 'missing cell')}${preview.redrawn ? `; ${preview.redrawn} deleted to redraw` : ''}`
      : state?.pending ? 'checking Finish run' : preview?.reason || state?.error || '';
    line.dataset.base = sheetRunStatus(line.dataset.run);
    line.textContent = `${line.dataset.base}${extra ? ` · ${extra}` : ''}`;
    line.title = line.textContent;
  }
}

function configureFinishRun(button, run, { slot = els.galleryError } = {}) {
  button.type = 'button';
  button.classList.add('finish-run-tool');
  button.dataset.run = run;
  delete button.dataset.finishSent;
  button.onclick = () => { void pressFinishRun(button, { slot }); };
  syncFinishRunButton(button);
  void readFinishRun(run);
  return button;
}

function finishRunButton(run, { className = 'ghost small', slot = els.galleryError } = {}) {
  const button = document.createElement('button');
  button.className = className;
  return configureFinishRun(button, run, { slot });
}

async function readFinishRun(run, { force = false } = {}) {
  const context = finishContext();
  const previous = finishRuns.get(run);
  if (previous?.context === context && previous.pending) return previous.pending;
  if (!force && previous?.context === context) return previous.pending ?? previous;
  const state = { context, pending: null, details: null, preview: null, error: '' };
  finishRuns.set(run, state);
  state.pending = (async () => {
    try {
      const details = await api(`/api/runs/${encodeURIComponent(run)}`);
      const action = details.actions.finish;
      const preview = action.available
        ? await api(`/api/runs/${encodeURIComponent(run)}/finish/preview`, { method: 'POST' })
        : { available: false, reason: action.reason, cells: action.count, images: 0 };
      if (finishRuns.get(run) !== state || finishContext() !== context) return null;
      state.details = details;
      state.preview = preview;
      return state;
    } catch (err) {
      if (finishRuns.get(run) !== state || finishContext() !== context) return null;
      state.error = err.message;
      return state;
    } finally {
      state.pending = null;
      if (finishRuns.get(run) === state && finishContext() === context) syncFinishRunButtons();
    }
  })();
  syncFinishRunButtons();
  return state.pending;
}

function refreshFinishRuns(force = false) {
  const runs = new Set(finishTools().filter((button) => !button.closest('[hidden]')).map((button) => button.dataset.run));
  for (const run of runs) void readFinishRun(run, { force });
  syncFinishRunButtons();
}

function invalidateFinishRuns() {
  finishRuns.clear();
  refreshFinishRuns(true);
}

function receiveFinishStatus(status) {
  const stamp = JSON.stringify([status.started_at, status.model_revision, status.quantization,
    status.active_jobs, status.newest_image_id]);
  if (stamp !== finishServerStamp) {
    finishServerStamp = stamp;
    invalidateFinishRuns();
  } else syncFinishRunButtons();
}

async function pressFinishRun(button, { slot = els.galleryError } = {}) {
  const run = button.dataset.run;
  if (!run || button.dataset.finishSent || finishPresses.has(run)) return;
  const local = binMode ? BIN_HELD : submitHold() || (submitting ? 'A job is being submitted' : '');
  if (local) { showError(local, false, { slot }); return; }
  const identity = finishIdentity();
  let context = finishContext();
  const banner = !els.lastRun.hidden && els.lastRun.contains(button);
  finishPresses.add(run);
  button.setAttribute('aria-busy', 'true');
  syncFinishRunButtons();
  try {
    let state = await readFinishRun(run, { force: true });
    // A finishing job's next status poll can invalidate its newly drawn
    // control while this check is out. Read the current answer once more;
    // a changed server or checkpoint still needs a new deliberate press.
    if (!state && finishIdentity() === identity && button.isConnected && button.dataset.run === run) {
      context = finishContext();
      state = await readFinishRun(run, { force: true });
    }
    if (!button.isConnected || button.dataset.run !== run) return;
    if (!state || finishContext() !== context) {
      showError('The run or checkpoint changed while checking. Press Finish run again.', false, { slot });
      return;
    }
    if (state.error || !state.preview?.available) {
      showError(state.error || state.preview?.reason || 'Finish run is unavailable', false, { slot });
      return;
    }
    const preview = state.preview;
    const why = binMode ? BIN_HELD : submitHold() || (submitting ? 'A job is being submitted' : '');
    if (why) { showError(why, false, { slot }); return; }
    await startJob(`/api/runs/${encodeURIComponent(run)}/finish`, { expected: preview.pin }, {
      panel: false, slot,
      onAccepted: () => {
        finishPresses.delete(run);
        button.dataset.finishSent = 'true';
        button.textContent = 'Finish sent';
        button.removeAttribute('aria-busy');
        invalidateFinishRuns();
      },
      onRefused: () => {
        // setBusy hides the banner before sending; a refusal keeps its
        // recovery control and the server's reason available for another press.
        if (banner && els.lastRun.dataset.run === run) els.lastRun.hidden = false;
        invalidateFinishRuns();
      },
    });
  } finally {
    finishPresses.delete(run);
    button.removeAttribute('aria-busy');
    syncFinishRunButtons();
  }
}

const ANNOUNCED_KEY = 'auradream.announced-job';
const LEGACY_ANNOUNCED_KEYS = ['krea.announced-job'];
const ANNOUNCE_WINDOW_MS = 24 * 3600 * 1000;
const when = (ms) => new Date(ms).toLocaleString([], {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
});

function announcedJob() {
  let value = remembered(ANNOUNCED_KEY);
  if (value) return value;
  for (const key of LEGACY_ANNOUNCED_KEYS) {
    value = remembered(key);
    if (!value) continue;
    remember(ANNOUNCED_KEY, value);
    forget(key);
    return value;
  }
  return null;
}

// The newest finished job, while the server still holds it. A 480-cell grid
// that stopped at cell 300 used to leave an idle Generate button and a gallery
// with 299 more images in it, and nothing anywhere saying when or why. Once per
// job per browser, and only within a day of its end.
async function announceLastRun() {
  let jobs;
  try { jobs = await api('/api/jobs?include_finished=true'); } catch { return; }
  const last = jobs
    .filter((j) => j.finished_at && ['done', 'error', 'cancelled'].includes(j.status))
    .sort((a, b) => b.finished_at - a.finished_at)[0];
  if (!last) return;
  if (Date.now() - last.finished_at * 1000 > ANNOUNCE_WINDOW_MS) return;
  if (announcedJob() === last.job_id) return;
  // A job that ended at the load is the status's to say while the load still
  // fails - the viewer's card, with the steps - and the banner said it again.
  if (last.status === 'error' && !last.started_at) {
    let status = null;
    try { status = await api('/api/status'); } catch { /* announced below */ }
    if (status?.state === 'error' && status.detail === last.error) return;
  }
  // A job of this page's own started while the listings were out: it hides
  // the banner (setBusy), which must not come up over it afterwards.
  if (busy) return;

  const images = `${last.image_count} image${last.image_count === 1 ? '' : 's'}` +
    (typeof last.distinct_count === 'number' && last.image_count > 1 && last.distinct_count < last.image_count
      ? `, only ${last.distinct_count} distinct` : '');
  // Where a stopped job got to, as that: "(6/12 cells)" read as six done for
  // a job that stopped at the sixth with three images. A budget stop's count
  // is its finished cells, and a failure stop's error says where already.
  // In the job's own units (unitWord): a grid's cells, a plain job's images
  // or batches - "cell" is a grid's word.
  const stopped = last.status !== 'done' && last.stopped_by !== 'budget';
  const unit = unitWord(last);
  const units = unit === 'batch' ? 'batches' : `${unit}s`;
  const cells = stopped && !last.run ? ', never started'
    : last.stopped_by === 'failures' || last.total_runs <= 1 ? ''
    : stopped ? `, stopped at ${unit} ${last.run} of ${last.total_runs}`
    : ` (${last.run}/${last.total_runs} ${units})`;
  const outcome = last.stopped_by === 'budget' ? 'stopped at its time budget'
    : last.status === 'cancelled' ? 'was cancelled'
    : last.status === 'error' ? 'failed'
    : last.error ? 'finished with failures'
    : 'finished';
  // What is left of it: the count is what the job made, and the images
  // deleted since used to be counted and offered as though still there -
  // Show run opened a sheet of one for "6 images". All of them gone, there
  // is no run to show.
  const deleted = Math.min(last.deleted_count ?? 0, last.image_count);
  const left = !deleted ? '' : deleted === last.image_count ? ', all since deleted' : `, ${deleted} since deleted`;
  els.lastRunText.textContent =
    `Last run ${outcome} at ${when(last.finished_at * 1000)}: ${images}${cells}${left}.` +
    (last.error ? ` ${last.error}` : '') +
    (last.skipped ? ` ${last.skipped} ${last.skipped === 1 ? unit : units} skipped.` : '');
  els.lastRun.classList.toggle('bad', last.status === 'error');
  els.lastRun.dataset.job = last.job_id;
  els.lastRun.dataset.run = last.run_id || '';
  els.lastRunShow.hidden = !last.image_count || deleted === last.image_count;
  // A run that finished: more seeds of every cell of it, beside Show run.
  // The checkpoint and its batches are the server's to judge (`foreign`).
  els.lastRunMore.hidden = !(last.status === 'done' && last.run_id && last.image_count > deleted);
  syncMoreSeedsTool(els.lastRunMore, {
    title: `More seeds of every cell of run ${last.run_id}, starred or not, the seeds following on - each at its own settings and Batch`,
  });
  // Availability is the server's, including Repeat, comparisons and jobs
  // with no surviving images. The composer is never a run definition.
  els.lastRunResume.hidden = !last.run_id;
  if (last.run_id) configureFinishRun(els.lastRunResume, last.run_id, { slot: els.error });
  els.lastRun.hidden = false;
}

function dismissLastRun() {
  remember(ANNOUNCED_KEY, els.lastRun.dataset.job || '');
  for (const key of LEGACY_ANNOUNCED_KEYS) forget(key);
  els.lastRun.hidden = true;
}

function resumeLastRun() {
  void pressFinishRun(els.lastRunResume, { slot: els.error });
}

// The banner's More seeds…: the dialog on the run. The banner stays until
// the job it queues hides it (setBusy), as a Cancel there leaves it.
function moreSeedsLastRun() {
  const run = els.lastRun.dataset.run;
  if (run) pressMoreSeedsTool(els.lastRunMore, moreSeedsRunTarget(run));
}

// One line per run in the gallery, newest first. Picking one is the card's ⊞
// button without having to find a card: it searches for the run and lays it
// out as a sheet.
//
// What a sheet is for is a grid or a batch, and every plain Generate is a run
// of its own: an afternoon's twenty of them pushed last night's grid off a
// list of twenty, under twenty lines that differed in the minute alone. So
// the server is asked for 200, the runs of more than one image (or one
// resumed) come first, and after them the newest single images, eight at
// most, each by its second and its seed - what tells two of them apart.
const SINGLE_RUNS = 8;
const whenExactly = (ms) => new Date(ms).toLocaleString([], {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
});

// What a run varied, ahead of its prompt in the picker (/api/runs' `axes`):
// "steps×CFG 3×2", "steps ×8 · 5 seeds", "strength 0.05-0.15 of #37",
// "comparison · 3 versions", "4 seeds". Every line of a session read
// "6 img · A weathered…", and a steps × CFG grid, a batch and a refine sweep
// told apart by count and time.
const RUN_AXIS_NAMES = {
  steps: 'steps', cfg: 'CFG', mu: 'Mu', true_cfg: 'True CFG', lora_scale: 'LoRA', reference: 'refs', crop_anchor: 'crop anchors',
};
const runSizesWords = (run) => run.sizes?.length > 1
  ? `${run.sizes.length} sizes (${run.sizes.map(([w, h]) => `${w}\u00d7${h}`).join(', ')})` : '';
function runAxes(run) {
  const axes = run.axes ?? {};
  const sizes = runSizesWords(run);
  if (axes.variant) return [`comparison \u00b7 ${axes.variant} versions`, sizes].filter(Boolean).join(' \u00b7 ');
  if (run.refine != null) {
    const [low, high] = run.strengths ?? [];
    const at = low == null ? '' : low === high ? `strength ${low} ` : `strength ${low}-${high} `;
    return [`${at}of #${run.refine}`, sizes, axes.crop_anchor > 1 ? `${axes.crop_anchor} crop anchors` : ''].filter(Boolean).join(' \u00b7 ');
  }
  const varied = Object.keys(RUN_AXIS_NAMES).filter((key) => axes[key] > 1);
  const seeds = axes.seed > 1 ? `${axes.seed} seeds` : '';
  if (!varied.length) return [sizes, seeds].filter(Boolean).join(' \u00b7 ');
  const names = varied.map((key) => RUN_AXIS_NAMES[key]);
  const counts = varied.map((key) => axes[key]);
  const grid = varied.length === 1 ? `${names[0]} ×${counts[0]}` : `${names.join('×')} ${counts.join('×')}`;
  return [grid, sizes, seeds].filter(Boolean).join(' \u00b7 ');
}

let runsEpoch = 0;
async function loadRuns() {
  const epoch = ++runsEpoch;
  const model = loadedModel, revision = modelRevision, allModels = els.allModels.checked;
  let runs;
  // Under the gallery's scope: a run picked here opens as a sheet, and a run
  // the grid would not show is an empty one.
  const params = new URLSearchParams({ limit: 200 });
  if (els.allModels.checked) params.set('all_models', 'true');
  try { runs = await api(`/api/runs?${params}`); } catch { return; }
  if (epoch !== runsEpoch || revision !== modelRevision || model !== loadedModel || allModels !== els.allModels.checked) return;
  const several = (run) => run.images !== 1 || run.jobs > 1;
  const singles = runs.filter((run) => !several(run));
  els.runList.innerHTML = '';
  // What the picker shows while the gallery shows no run; not a run to pick.
  const head = new Option(runs.length ? 'Runs…' : 'No runs yet', '');
  head.disabled = head.hidden = true;
  els.runList.append(head);
  const group = (label, list, line) => {
    if (!list.length) return;
    const box = document.createElement('optgroup');
    box.label = label;
    for (const run of list) box.append(new Option(line(run), run.run_id));
    els.runList.append(box);
  };
  const stars = (run) => (run.favorites ? ` · ${run.favorites}★` : '');
  // Its words on one line, an XML prompt's without the markup every run of
  // it opens with (promptGist, generation.js): three NewBie runs that
  // differed in an attribute read "<character_1>\n<n>hatsune_miku</n>\n<gen…".
  const words = (run) => clip(promptGist(run.prompt ?? ''));
  group('Grids and batches', runs.filter(several), (run) => {
    // A run that made fewer pictures than images says so in the list.
    const distinct = typeof run.distinct === 'number' && run.distinct < run.images
      ? ` (${run.distinct} distinct)` : '';
    const prompts = run.prompts > 1 && !run.axes?.variant ? ` · ${run.prompts} prompts` : '';
    const axes = runAxes(run);
    return `${when(Date.parse(run.started_at))} · ${run.images} img${distinct}${prompts}${stars(run)}`
      + `${axes ? ` · ${axes}` : ''} · ${words(run)}`;
  });
  group(singles.length > SINGLE_RUNS ? `Single images (the newest ${SINGLE_RUNS} of ${singles.length})` : 'Single images',
    singles.slice(0, SINGLE_RUNS),
    (run) => `${whenExactly(Date.parse(run.started_at))} · seed ${run.seed}${stars(run)} · ${words(run)}`);
  els.runList.disabled = !runs.length;
  syncRunList();
  refreshFinishRuns(true);
}

// The run the search box shows, if it shows one.
const shownRun = () => /^run:(\S+)$/.exec(els.search.value.trim())?.[1] ?? null;

// The picker names the run the gallery is showing - the search box's
// `run:<id>`, however it got there: a pick here, a card's ⊞, Show run, the
// sheet's return, typing - and "Runs…" for anything else. It used to go back
// to "Runs…" the moment a run was picked, leaving a hex id in the box as the
// only label of what was on screen.
function syncRunList() {
  // A run browsed to from the keyboard and not yet opened (runListKey) stays
  // where the arrows left it through a rebuilt list.
  const shown = runPending ?? shownRun();
  els.runList.value = shown && [...els.runList.options].some((o) => o.value === shown) ? shown : '';
  // The closed picker is narrower than its labels (style.css, --runs-w): the
  // picked run's is whole in the title.
  els.runList.title = binMode ? BIN_HELD : els.runList.value ? els.runList.selectedOptions[0].text : 'Recent runs';
}
// A pick opens its run at once: a click in the open list, or Alt+Down and
// Enter, which change the picker with no key on it first. Browsed from the
// keyboard while closed - the arrows, Home and End, Page Up and Down, a
// letter's type-ahead - the picker changes at every key, and each change
// opened that run's sheet, asking the server for the whole run and
// scrolling the page: reaching the tenth run opened nine. A change one of
// those keys made is noted (`runPending`) and opened by Enter or by leaving
// the picker; Escape puts back the run the gallery shows. The key marks the
// change it causes alone - the change comes inside the key's own default -
// and the mark is gone by the next task.
let runKeyed = false;
let runPending = null;
const RUN_BROWSE_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown']);
function runListKey(e) {
  if (keyboardEventHeld(e) || keyboardCommandModifier(e) || e.shiftKey) return;
  if (e.key === 'Enter') {
    if (runPending !== null) { e.preventDefault(); if (!e.repeat) commitRunList(); }
    return;
  }
  if (e.key === 'Escape') {
    if (runPending !== null) { e.preventDefault(); if (!e.repeat) { runPending = null; syncRunList(); } }
    return;
  }
  if (RUN_BROWSE_KEYS.has(e.key) || (e.key.length === 1 && e.key !== ' ')) {
    runKeyed = true;
    setTimeout(() => { runKeyed = false; });
  }
}

function runListChange() {
  if (runKeyed) {
    runPending = els.runList.value;
    syncRunList();
    return;
  }
  runPending = null;
  void showRun(els.runList.value);
}

function commitRunList() {
  const run = runPending;
  runPending = null;
  if (run && run !== shownRun()) void showRun(run);
  else syncRunList();
}

// `fresh` is a comparison this page has just made: nothing in it is
// starred yet, so under the Starred tick its sheet opened empty. The tick
// comes off rather than being bypassed - the gallery's filters are the
// sheet's, and Re-render sends the tick with the search, so a sheet of
// unstarred images under a ticked box would re-render none of them - and the
// sheet says so.
function showRun(runId, { fresh = false, asker } = {}) {
  if (!runId) return;
  return showSheetScope(`run:${runId}`, { fresh, asker });
}

function showJob(jobId, { fresh = false, asker } = {}) {
  if (!jobId) return;
  return showSheetScope(`job:${jobId}`, { fresh, asker });
}

// Every sheet the page opens comes through here: a comparison, a refine sweep,
// the Runs picker, a card's ⊞, Show run. It keeps the search it replaces for
// Sheet off to put back (sheetReturn) - back to back, one comparison's sheet
// after another's, still the search from before the first - and `notice` is a
// line for the note while this query is the one shown.
//
// Then, once setSheet has resolved (after renderSheet) and only if this is
// still the sheet shown, the page goes to it: a finished comparison's sheet
// opened at y = 1650 on a 1305 px screen, under the prompt check and the
// viewer, with nothing in view changed but a line in the panel. To the gallery
// head rather than the sheet, so the search, Starred and Sheet - the sheet's
// own filters - stay in view above it. Not for the Sheet button (`scroll:
// false`), pressed at the gallery head already - unless the gallery stood
// beside the viewer, where the sheet opens under it (setSheet). The scroll
// hides nothing - the check and the viewer, where a running job lands, are a
// scroll up - and takes focus along only where the control that asked
// (`asker`: the one focused now, unless the caller knows better - a job's
// sheet, the one its submission was pressed on) is left out of sight, or
// focus has fallen out of the page (sheet-render.js's focusSheet).
function showSheetScope(search, { fresh = false, notice = null, scroll = true, asker = document.activeElement } = {}) {
  // A sheet is the gallery's: opened from the recycle bin, the bin closes.
  leaveBin();
  const unticked = fresh && els.favOnly.checked;
  if (unticked) els.favOnly.checked = false;
  if (!sheetMode || !sheetReturn || els.search.value.trim() !== sheetReturn.wrote) {
    sheetReturn = { search: els.search.value, wrote: search };
  } else {
    sheetReturn.wrote = search;
  }
  els.search.value = search;
  const opened = setSheet(true, { land: scroll, asker });
  // setSheet has begun the query the notice belongs to; another one drops it.
  const epoch = galleryEpoch;
  const text = unticked ? 'Starred unticked: nothing in a new comparison is starred yet' : notice;
  if (text) sheetNotice = { epoch, text };
  return opened;
}

// Where a sheet is landed on, once drawn (setSheet, only if it is still the
// sheet shown): the gallery head under the topbar, room reserved under the
// sheet for it to get there - and, once the scroll is over and if the sheet
// is still the one shown, the focus (focusSheet).
function landOnSheet(asker = null) {
  const head = document.querySelector('.gallery-head');
  reserveSheetRoom(head);
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  head.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  const epoch = galleryEpoch;
  void scrollSettled().then(() => { if (sheetMode && epoch === galleryEpoch) focusSheet(asker); });
}

// Room under a sheet for its head to come up to the topbar: a short sheet
// under a tall viewer - a refine sweep's three passes at QHD - ended the page
// before the head could get there, and the scroll stopped 43 px down with the
// sheet's cells at the window's foot. A margin, outside the sheet's box,
// which fitSheet measures; setSheet takes it off again.
function reserveSheetRoom(head) {
  els.sheet.style.marginBottom = '';
  const landing = head.getBoundingClientRect().top + scrollY - parseFloat(getComputedStyle(head).scrollMarginTop);
  // From where the page's content ends, not the document's scroll height,
  // which is never less than the window: a page shorter than the window got
  // a margin short by the difference, and the scroll stopped 85 px short,
  // the viewer's tools row cut in half under the topbar above the head.
  const end = document.querySelector('main').getBoundingClientRect().bottom + scrollY;
  const short = landing + innerHeight - end;
  if (short > 0) els.sheet.style.marginBottom = `${Math.ceil(short)}px`;
}
