'use strict';

const $ = (id) => document.getElementById(id);
const PAGE = 48;
// A sheet wants the whole run in one go - it is a matrix, and a matrix with a
// page boundary through it is a puzzle. The rows are small and the thumbnails
// load lazily, so a full grid is cheaper than it sounds. The size follows the
// server's grid ceiling (/api/status), so a raised KREA_MAX_GRID_IMAGES does
// not silently drop the earliest seeds off a sheet.
let sheetPage = 1024;

const els = {
  prompt: $('prompt'), negative: $('negative'), negField: $('negative-field'),
  negNote: $('neg-note'),
  width: $('width'), height: $('height'), steps: $('steps'), cfg: $('cfg'),
  mu: $('mu'), seed: $('seed'), batch: $('batch'), runs: $('runs'),
  stepsTip: $('steps-tip'), cfgTip: $('cfg-tip'), muTip: $('mu-tip'),
  generate: $('btn-generate'), dice: $('btn-dice'), unload: $('btn-unload'), load: $('btn-load'),
  cancel: $('btn-cancel'), pause: $('btn-pause'),
  queueNote: $('queue-note'), queueText: $('queue-text'), cancelQueued: $('btn-cancel-queued'),
  progress: $('progress'), fill: $('progress-fill'), progressText: $('progress-text'),
  stepRow: $('step-row'), stepFill: $('step-fill'), stepText: $('step-text'),
  cellNote: $('cell-note'),
  gridSection: $('grid-section'), gridPrompt: $('grid-prompt'),
  gridSteps: $('grid-steps'), gridCfg: $('grid-cfg'),
  gridMu: $('grid-mu'), gridSeeds: $('grid-seeds'), gridSeedList: $('grid-seed-list'),
  gridBudget: $('grid-budget'), gridPreview: $('grid-preview'),
  gridRun: $('btn-grid'),
  error: $('error'), viewer: $('viewer'), viewerNote: $('viewer-note'), gallery: $('gallery'),
  count: $('gallery-count'), search: $('search'), favOnly: $('fav-only'),
  more: $('btn-more'), presets: $('presets'), tiers: $('tiers'), sizeHint: $('size-hint'),
  gridResume: $('grid-resume'),
  sheetBtn: $('btn-sheet'), rerender: $('btn-rerender'),
  sheet: $('sheet'), sheetControls: $('sheet-controls'),
  sheetGrid: $('sheet-grid'), sheetNote: $('sheet-note'),
  runHint: $('run-hint'),
  lastRun: $('last-run'), lastRunText: $('last-run-text'),
  lastRunShow: $('last-run-show'), lastRunDismiss: $('last-run-dismiss'),
  runList: $('run-list'),
  dot: $('status-dot'), statusText: $('status-text'), vram: $('vram'),
  lightbox: $('lightbox'), lbImg: $('lb-img'), lbMeta: $('lb-meta'), lbClose: $('lb-close'),
  lbError: $('lb-error'), lbStage: $('lb-stage'), lbZoom: $('lb-zoom'),
  lbFit: $('lb-fit'), lbOne: $('lb-one'),
  lbReuse: $('lb-reuse'), lbPrev: $('lb-prev'), lbNext: $('lb-next'),
  lbFav: $('lb-fav'), lbDelete: $('lb-delete'),
};

// Aspect × megapixel tier. Every side is a multiple of 16 (the VAE downsamples by
// 16) and every entry lands within a few percent of its tier; 2 and 4 MP hit their
// ratios exactly. The whole 4 MP row is within 0.3% of the old 2K preset's pixel
// count, so it costs the same VRAM whichever framing you pick.
const SIZES = {
  '1:1':  { 1: [1024, 1024], 2: [1440, 1440], 4: [2048, 2048] },
  '3:4':  { 1: [896, 1200],  2: [1248, 1664], 4: [1776, 2368] },
  '4:3':  { 1: [1200, 896],  2: [1664, 1248], 4: [2368, 1776] },
  '16:9': { 1: [1360, 768],  2: [1920, 1088], 4: [2736, 1536] },
  '9:16': { 1: [768, 1360],  2: [1088, 1920], 4: [1536, 2736] },
};
const ASPECTS = Object.keys(SIZES);
const TIERS = [1, 2, 4];

// The two axes are picked independently - changing one holds the other - so the
// same framing at three sizes is three clicks in the same place.
let pickedAspect = '1:1';
let pickedTier = 1;

// Paging is by id, not by offset: the gallery is a table a sweep is inserting
// into, and an offset over it repeats the tail of one page at the head of the
// next and skips as many older rows for good. The anchor is the smallest id on
// the page (the last record - they arrive newest first), and the server says
// whether anything older exists, exactly.
let galleryMore = false;
let total = 0;
let busy = false;
// Which sampling controls the user has taken off the checkpoint's profile.
// Per control, not one flag for all three: touching Seed before the first
// /api/status used to freeze steps, CFG and mu at the page's fallback numbers,
// and on Turbo that is 28 steps at CFG 3.5 for a model trained for 8 at 0.
const owned = { steps: false, cfg: false, mu: false };
// The checkpoint the server is running, once /api/status has said. Reuse
// compares a record against it: steps, CFG and mu are checkpoint-specific.
let loadedModel = null;
let statusPending = false;
// The job whose stream the page is watching, and the ids queued behind it.
// The server was written for a queue - jobs wait on its gate, a queued one
// can be cancelled on the spot - and the browser used to refuse a second
// submission outright, so the queue was unreachable from the UI.
let currentJob = null;
let queuedIds = [];
let submitting = false;
// One gallery page in flight at a time, and a token so a page that was already
// on its way when the filters changed can be dropped rather than appended.
let galleryPending = null;
let galleryEpoch = 0;
// The CFG above which a negative prompt does anything: 1 on FLUX, where
// guidance_scale is a conditioning value baked into the model and only
// true_cfg_scale runs an uncond pass, 0 on Krea 2, which guides from 0 up. Not
// a control, so unlike the seeded defaults it follows the server all session.
let cfgThreshold = 1;
// FLUX.2 klein's pipeline takes no negative prompt at any CFG; the field greys
// out for the checkpoint rather than for the value.
let negativeSupported = true;
// Free space on the images volume, from the last status poll; null if unknown.
let diskFreeMb = null;

const humanBytes = (n) => {
  let size = n;
  for (const unit of ['B', 'KB', 'MB', 'GB']) {
    if (size < 1024 || unit === 'GB') return `${size.toFixed(1)} ${unit}`;
    size /= 1024;
  }
  return `${size.toFixed(1)} GB`;
};

// The two strips the lightbox can walk: what this job produced, and what the
// gallery has paged in so far. Both are kept in the order they are displayed.
let viewerRecords = [];
let galleryRecords = [];
let lbList = [];
let lbIndex = -1;

// Contact sheet. `sheetAxes` is what the user picked; null means "work it out
// from the images", which is what happens on the first render of a new set.
let sheetMode = false;
let sheetAxes = { rows: null, cols: null, pins: {} };
let sheetRecords = [];

// ------------------------------------------------------------------ helpers

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON body */ }
    throw new Error(readableDetail(detail));
  }
  return res.status === 204 ? null : res.json();
}

// FastAPI's own 422s carry a list of {loc, msg, ctx, input} objects, which as
// JSON.stringify is a wall of braces in the error slot. Ours are plain strings.
function readableDetail(detail) {
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return JSON.stringify(detail);
  return detail
    .map((e) => {
      const field = (e.loc ?? []).filter((part) => part !== 'body').join('.');
      return field ? `${field}: ${e.msg}` : e.msg;
    })
    .join('; ');
}

// `note` styles the same slot as muted rather than red, for outcomes that are
// not failures - a cancelled run in particular.
function showError(message, note = false) {
  // Unhide *before* writing: the slot is role=alert, and text written into a
  // display:none node is never announced.
  els.error.hidden = !message;
  els.error.textContent = message || '';
  els.error.classList.toggle('note', note);
  // The lightbox covers the panel's slot; mirror into its own while it is up.
  const mirror = els.lightbox.open;
  els.lbError.hidden = !(mirror && message);
  els.lbError.textContent = mirror ? message || '' : '';
  els.lbError.classList.toggle('note', note);
}

// -------------------------------------------------------------------- sizes

function buildPresets() {
  ASPECTS.forEach((aspect) => {
    els.presets.appendChild(presetButton(aspect, () => choose(aspect, pickedTier)));
  });
  TIERS.forEach((tier) => {
    els.tiers.appendChild(presetButton(`${tier} MP`, () => choose(pickedAspect, tier)));
  });
}

function presetButton(label, onClick) {
  const b = document.createElement('button');
  b.textContent = label;
  b.onclick = onClick;
  return b;
}

function choose(aspect, tier) {
  [els.width.value, els.height.value] = SIZES[aspect][tier];
  syncSize();
  // Assigning .value fires no `input`, so the grid preview - whose duration
  // estimate is priced per size - would keep the old numbers, and the
  // remembered panel would keep the old size.
  schedulePreview();
  persistPanel();
}

function syncSize() {
  const w = +els.width.value, h = +els.height.value;
  syncRerender();
  // Recover which preset this is, so a reused or hand-typed size lights the
  // buttons up too, and the next click off it starts from the right place.
  const match = ASPECTS.flatMap((a) => TIERS.map((t) => [a, t]))
    .find(([a, t]) => SIZES[a][t][0] === w && SIZES[a][t][1] === h);
  if (match) [pickedAspect, pickedTier] = match;

  [...els.presets.children].forEach((b, i) => {
    b.classList.toggle('active', !!match && ASPECTS[i] === match[0]);
  });
  [...els.tiers.children].forEach((b, i) => {
    b.classList.toggle('active', !!match && TIERS[i] === match[1]);
    // Spell out what the tier means for the framing currently in play.
    b.title = SIZES[pickedAspect][TIERS[i]].join(' × ');
  });

  const tight = w * h > 1600 * 1600;
  // 4 MP on a 16GB card is the edge of what fits once the DiT is resident.
  els.sizeHint.textContent = `${((w * h) / 1e6).toFixed(1)} MP` +
    (tight ? ' — may run tight on 16GB VRAM' : '');
  els.sizeHint.classList.toggle('warn', tight);
}

// The highest CFG the open grid's axis reaches, from the last preview; null when
// the grid section is closed or the axis is blank. A grid decides guidance per
// cell, so the negative prompt has to be offered when *any* cell would apply
// it, not only when the panel's single value would.
let gridMaxCfg = null;

function syncCfg() {
  const reach = gridMaxCfg === null ? +els.cfg.value : Math.max(+els.cfg.value, gridMaxCfg);
  const off = !negativeSupported || reach <= cfgThreshold;
  els.negField.classList.toggle('disabled', off);
  els.negative.disabled = off;
  els.negNote.textContent = !negativeSupported
    ? '(this checkpoint takes no negative prompt)'
    : gridMaxCfg !== null && +els.cfg.value <= cfgThreshold && reach > cfgThreshold
      ? `(applies to the grid cells above CFG ${cfgThreshold} — ~2× slower there)`
      : `(needs CFG > ${cfgThreshold} — ~2× slower)`;
}

// Batch holds every image in VRAM at once, which is what stops a 2K batch from
// fitting. Runs trade that for wall-clock: one pipeline call each, seed stepped
// by the batch size so nothing in the sweep repeats.
function syncRuns() {
  const runs = Math.max(1, +els.runs.value || 1);
  const batch = Math.max(1, +els.batch.value || 1);
  const seed = els.seed.value.trim();
  els.runHint.textContent = runs === 1
    ? (batch > 1 ? `${batch} images in one pass — all held in VRAM together.` : '')
    : `${runs * batch} images over ${runs} passes, seed +${batch} each, ` +
      (seed === '' ? 'from a random start.' : `from ${seed}.`);
}

// ------------------------------------------------------------------- status

// Seed the controls from the server's profile for the loaded checkpoint. Turbo
// wants 8 steps / CFG 0 / mu 1.15; FLUX wants 28 / 3.5 and no mu at all, since
// it derives its own resolution-dependent shift. Each control follows the
// profile until the user touches *that* control; the checkpoint cannot change
// under a running server, so re-applying to an untouched box is idempotent.
function applyDefaults(d) {
  if (!d || d.steps === undefined) return;
  if (!owned.steps) els.steps.value = d.steps;
  if (!owned.cfg) els.cfg.value = d.guidance_scale;
  if (!owned.mu) els.mu.value = d.mu ?? '';
  syncCfg();
}

// What steps/CFG/mu mean depends on the checkpoint - 8 steps at CFG 0 is right
// for Turbo and ruinous for FLUX - so the tooltips come from the same server
// profile as the values, and unlike those values they keep following it: the
// user owning the controls does not make the explanation stale.
function applyHints(hints) {
  if (!hints) return;
  const tips = { steps: els.stepsTip, guidance_scale: els.cfgTip, mu: els.muTip };
  for (const [key, el] of Object.entries(tips)) {
    if (hints[key]) el.textContent = hints[key];
  }
}

// Anything the user picks - by hand or through Reuse - takes that control off
// the profile. /api/status reads CUDA memory, which stalls behind a model load,
// so its reply can land minutes late and used to wipe steps/CFG/mu long after
// the page had settled.
function own(...keys) { keys.forEach((key) => { owned[key] = true; }); }
function ownControls() { own('steps', 'cfg', 'mu'); }

async function refreshStatus() {
  if (statusPending) return;  // a stalled poll must not queue more behind it
  statusPending = true;
  try {
    const s = await api('/api/status');
    loadedModel = s.model_id ?? loadedModel;
    applyDefaults(s.defaults);
    // After the profile, so a remembered value set on this checkpoint wins
    // over it and a value set on another is left to the profile.
    if (loadedModel) restoreSampling(loadedModel);
    applyHints(s.hints);
    if (s.defaults && typeof s.defaults.cfg_threshold === 'number'
        && s.defaults.cfg_threshold !== cfgThreshold) {
      cfgThreshold = s.defaults.cfg_threshold;
      syncCfg();
    }
    if (s.defaults && typeof s.defaults.negative_prompt_supported === 'boolean'
        && s.defaults.negative_prompt_supported !== negativeSupported) {
      negativeSupported = s.defaults.negative_prompt_supported;
      syncCfg();
    }
    if (typeof s.max_grid_images === 'number') sheetPage = s.max_grid_images;
    els.dot.className = `dot ${s.state}`;
    // A way to start - or retry, after a gated-repo 403 - the load without
    // queueing an image to do it.
    els.load.hidden = !(s.state === 'unloaded' || s.state === 'error');
    els.statusText.textContent =
      s.state === 'ready' ? s.model_id.split('/').pop()
      : s.state === 'loading' ? 'loading weights…'
      : s.state === 'error' ? 'load failed'
      : 'not loaded';
    if (s.state === 'error' && s.detail) showError(s.detail);
    // Free space beside the VRAM: the disk is the other thing a night can
    // run out of, and the grid preview compares its size estimate against it.
    diskFreeMb = typeof s.disk_free_mb === 'number' ? s.disk_free_mb : null;
    const disk = diskFreeMb !== null ? `${(diskFreeMb / 1024).toFixed(1)} GB free` : '';
    els.vram.textContent = [
      s.vram_total_mb
        ? `${s.device_name ?? ''} · ${(s.vram_used_mb / 1024).toFixed(1)}/${(s.vram_total_mb / 1024).toFixed(1)} GB · ${s.quantization}`
        : '',
      disk,
    ].filter(Boolean).join(' · ');
  } catch { /* server not up yet */ } finally { statusPending = false; }
}

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
    mu: els.mu.value === '' ? null : +els.mu.value,
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
  els.generate.setAttribute('aria-disabled', String(on));
  els.gridRun.setAttribute('aria-disabled', String(on));
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

// Both buttons queue one job and then watch the same stream; only the endpoint
// and the payload differ.
async function startJob(path, payload) {
  if (submitting) return;
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
    // A grid started from a blank Seed box drew its own base, and every image
    // it produces carries a *derived* seed - so the base is written back here,
    // where Resume will need it. Only for a grid: a plain Generate left blank is
    // asking for a fresh seed every click.
    // Not for a listed seed axis: the list is the record, and the first seed
    // of it written into the box would only mislead the next plain Generate.
    if (path === '/api/grid' && !payload.seed_axis && els.seed.value.trim() === '' && seed != null) {
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

function generate() { return startJob('/api/generate', collect()); }

// The gallery's current selection - the search box and the Favorites tick -
// rendered again at the panel's size. Taken server-side, since the browser
// holds only the page it has scrolled to; the count in the button is the
// server's total for the filter, so it is also the confirmation.
function rerenderSelection() {
  return startJob('/api/rerender', {
    search: els.search.value.trim() || null,
    favorites: els.favOnly.checked,
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

// The prompts a variable block resolved to, a few of them. Substitution is the
// half of a prompt grid that goes wrong quietly - a value that reads fine in the
// box lands badly in the middle of a sentence - so the panel shows the finished
// text rather than the variables that produced it.
const PREVIEW_PROMPTS = 3;
function promptLines(prompts) {
  if (!prompts || prompts.length < 2) return [];
  const lines = prompts.slice(0, PREVIEW_PROMPTS).map((text) => `· ${text}`);
  if (prompts.length > PREVIEW_PROMPTS) {
    lines.push(`· … and ${prompts.length - PREVIEW_PROMPTS} more`);
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

async function previewGrid() {
  if (!els.gridSection.open) {
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
    return;
  }
  try {
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(gridPayload()),
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
    if (short) els.gridPreview.classList.add('warn');
    els.gridPreview.textContent = [
      `${p.images} image${p.images === 1 ? '' : 's'} · ${p.summary}` +
        (p.skipped ? ` · ${p.skipped} already done` : '') +
        (p.seconds ? ` · about ${humanTime(p.seconds)}` : '') +
        (p.bytes ? ` · ~${humanBytes(p.bytes)}${short ? ` — more than the ${humanBytes(diskFreeMb * 1024 * 1024)} free` : ''}` : ''),
      `steps ${axisText(p.values.steps)}  ·  CFG ${axisText(p.values.guidance_scale)}` +
        `  ·  mu ${axisText(p.values.mu)}` +
        (p.values.seed && p.values.seed.length ? `  ·  seeds ${axisText(p.values.seed)}` : ''),
      ...promptLines(p.values.prompt),
    ].join('\n');  // .hint.pre renders the breaks
  } catch (err) {
    if (epoch !== previewEpoch) return;
    els.gridPreview.classList.add('warn');
    els.gridPreview.textContent = err.message;
  }
}

let gridTimer;
function schedulePreview() { clearTimeout(gridTimer); gridTimer = setTimeout(previewGrid, 250); }

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
    const show = (rec) => {
      if (shown.has(rec.id)) return;
      shown.add(rec.id);
      if (!cleared) { clearResults(); cleared = true; }
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
        if (data.images) data.images.forEach(show);
        // A finished run used to leave the panel indistinguishable from an
        // idle page. The summary goes in the cell line, which setBusy has
        // just cleared and which is outside the element it hides.
        if (data.status === 'done' && data.started_at && data.finished_at) {
          const made = data.images ? data.images.length : data.image_count;
          els.cellNote.textContent =
            `Finished ${made} image${made === 1 ? '' : 's'} in ${humanTime(data.finished_at - data.started_at)}.`;
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
        if (data.images.length) { resetGallery(); loadRuns(); }
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

// ------------------------------------------------------------------ the panel

// The panel remembers itself per browser. The grid spec used to live only in
// the DOM: continuing last night meant retyping a variable block, three axis
// strings and a seed exactly, and an axis typed slightly differently is a
// different grid that resume cannot match. Two phases - the model-independent
// fields synchronously at start-up, ahead of the sync functions that light up
// from them, and steps/CFG/mu once /api/status has named the checkpoint, and
// only when it is the one they were set on. Resume is never remembered: a box
// that came back ticked the next evening would silently skip cells of a grid
// the user did not mean to continue.
const PANEL_KEY = 'krea.panel.v1';
const PANEL_FIELDS = [
  'prompt', 'negative', 'width', 'height', 'seed', 'batch', 'runs',
  'gridPrompt', 'gridSteps', 'gridCfg', 'gridMu', 'gridSeeds', 'gridSeedList', 'gridBudget',
];
let panelTimer;

function persistPanel() {
  clearTimeout(panelTimer);
  panelTimer = setTimeout(() => {
    const state = { gridOpen: els.gridSection.open };
    for (const key of PANEL_FIELDS) state[key] = els[key].value;
    // Sampling values travel with the checkpoint they were set on, and only
    // the ones the user took off the profile - the profile itself is the
    // server's to give back.
    state.sampling = {
      model: loadedModel,
      steps: owned.steps ? els.steps.value : null,
      cfg: owned.cfg ? els.cfg.value : null,
      mu: owned.mu ? els.mu.value : null,
    };
    remember(PANEL_KEY, JSON.stringify(state));
  }, 300);
}

function storedPanel() {
  try { return JSON.parse(remembered(PANEL_KEY) || 'null'); } catch { return null; }
}

function restorePanel() {
  const state = storedPanel();
  if (!state) return;
  for (const key of PANEL_FIELDS) {
    if (typeof state[key] === 'string') els[key].value = state[key];
  }
  els.gridSection.open = !!state.gridOpen;
  els.gridSeeds.disabled = els.gridSeedList.value.trim() !== '';
}

// The second phase, from refreshStatus once the checkpoint is known.
let samplingRestored = false;
function restoreSampling(modelId) {
  if (samplingRestored) return;
  samplingRestored = true;
  const sampling = storedPanel()?.sampling;
  if (!sampling || sampling.model !== modelId) return;
  for (const [key, el] of [['steps', els.steps], ['cfg', els.cfg], ['mu', els.mu]]) {
    if (typeof sampling[key] === 'string') { el.value = sampling[key]; own(key); }
  }
  syncCfg();
}

// ------------------------------------------------------------ the morning after

// localStorage is a per-browser convenience here and nothing more: it can be
// absent (a private window), blocked, or throw on access, and the page has to
// work the same without it.
function remembered(key) { try { return localStorage.getItem(key); } catch { return null; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch { /* unavailable */ } }

const ANNOUNCED_KEY = 'krea.announced-job';
const ANNOUNCE_WINDOW_MS = 24 * 3600 * 1000;
const when = (ms) => new Date(ms).toLocaleString([], {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
});

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
  if (remembered(ANNOUNCED_KEY) === last.job_id) return;

  const images = `${last.image_count} image${last.image_count === 1 ? '' : 's'}`;
  const cells = last.total_runs > 1 ? ` (${last.run}/${last.total_runs} cells)` : '';
  const outcome = last.status === 'cancelled' ? 'was cancelled'
    : last.status === 'error' ? 'failed'
    : last.error ? 'finished with failures'
    : 'finished';
  els.lastRunText.textContent =
    `Last run ${outcome} at ${when(last.finished_at * 1000)}: ${images}${cells}.` +
    (last.error ? ` ${last.error}` : '');
  els.lastRun.classList.toggle('bad', last.status === 'error');
  els.lastRun.dataset.job = last.job_id;
  els.lastRun.dataset.run = last.run_id || '';
  els.lastRunShow.hidden = !last.image_count;
  els.lastRun.hidden = false;
}

function dismissLastRun() {
  remember(ANNOUNCED_KEY, els.lastRun.dataset.job || '');
  els.lastRun.hidden = true;
}

// One line per run in the gallery, newest first. Picking one is the card's ⊞
// button without having to find a card: it searches for the run and lays it
// out as a sheet.
async function loadRuns() {
  let runs;
  try { runs = await api('/api/runs?limit=20'); } catch { return; }
  els.runList.innerHTML = '';
  els.runList.append(new Option(runs.length ? 'Runs…' : 'No runs yet', ''));
  for (const run of runs) {
    const stars = run.favorites ? ` · ${run.favorites}★` : '';
    const label = `${when(Date.parse(run.started_at))} · ${run.images} img${stars} · ${clip(run.prompt)}`;
    els.runList.append(new Option(label, run.run_id));
  }
  els.runList.disabled = !runs.length;
}

function showRun(runId) {
  if (!runId) return;
  els.search.value = `run:${runId}`;
  setSheet(true);
}

function clearResults() {
  viewerRecords = [];
  els.viewer.innerHTML = '';
  els.viewer.classList.remove('multi', 'many');
  els.viewerNote.textContent = '';
}

// The most full-resolution images the strip keeps in the DOM. A 1024-cell
// night at 4 MP is ~5 GB of PNG; the strip is for watching images land, and
// the lightbox still walks every record of the run through `viewerRecords`.
const VIEWER_MAX = 48;

function appendResult(rec) {
  viewerRecords.push(rec);
  // Two size tiers, so a long sweep does not push the gallery a screen further
  // down with every image it finishes. The lightbox is where they get looked at.
  els.viewer.classList.toggle('multi', viewerRecords.length > 1);
  els.viewer.classList.toggle('many', viewerRecords.length > 4);
  const img = thumb(rec, viewerRecords, rec.url);
  els.viewer.appendChild(img);
  // Lazy on its own is not enough - the strip sets no height, so unloaded
  // images stack inside the viewport and all become eligible at once. The cap
  // is what bounds the fetches; the note lives outside the strip so it survives.
  while (els.viewer.children.length > VIEWER_MAX) els.viewer.firstElementChild.remove();
  const hidden = viewerRecords.length - els.viewer.children.length;
  els.viewerNote.textContent = hidden > 0
    ? `… and ${hidden} earlier image${hidden === 1 ? '' : 's'} this run — open one to walk them all`
    : '';
}

// One <img> for every place a record is shown: the viewer strip, a gallery
// card, a sheet cell. Lazy and async-decoded, since all three can hold hundreds
// of them; a thumbnail derivative where one exists, the full PNG otherwise.
// Focusable and activated by Enter or Space, so the lightbox - the only place
// the full image and its metadata are shown - is reachable without a mouse.
function thumb(rec, list, src = rec.thumb_url ?? rec.url) {
  const img = document.createElement('img');
  img.src = src;
  img.alt = rec.prompt;
  img.loading = 'lazy';
  img.decoding = 'async';
  img.dataset.id = rec.id;
  img.tabIndex = 0;
  img.setAttribute('role', 'button');
  img.setAttribute('aria-label', `Open: ${rec.prompt}`);
  const open = () => openLightbox(rec, list);
  img.onclick = open;
  img.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
  };
  return img;
}

// -------------------------------------------------------------------gallery

function resetGallery() {
  // Anything already in flight was fetched under the old filters and against
  // the grid this is about to empty; the token tells it to drop itself.
  galleryEpoch += 1;
  galleryMore = false;
  galleryRecords = [];
  els.gallery.innerHTML = '';
  // The fresh query has to be issued *after* an in-flight page settles rather
  // than sharing its promise: that page carries the old epoch and drops
  // itself, so sharing it left the gallery empty with nothing outstanding and
  // nothing to retry. Its `.finally` nulls `galleryPending` first, so the
  // chained call finds a free slot.
  const pending = galleryPending;
  return pending ? pending.then(loadGallery) : loadGallery();
}

function loadGallery() {
  // One page at a time. Nothing else serialises this - a double-clicked "Load
  // more", or an arrow key held down in the lightbox - and two calls reading
  // the same anchor append the same 48 cards twice.
  if (galleryPending) return galleryPending;
  const epoch = galleryEpoch;
  const params = new URLSearchParams({ limit: sheetMode ? sheetPage : PAGE });
  // The sheet asks for the whole run at once, so only the grid ever pages.
  const last = galleryRecords[galleryRecords.length - 1];
  if (!sheetMode && last) params.set('before', last.id);
  if (els.search.value.trim()) params.set('search', els.search.value.trim());
  if (els.favOnly.checked) params.set('favorites', 'true');
  els.more.disabled = true;
  els.lbNext.disabled = true;

  galleryPending = api(`/api/gallery?${params}`)
    .then((page) => {
      if (epoch !== galleryEpoch) return;  // superseded by a reset
      total = page.total;
      els.count.textContent = total ? `(${total})` : '';
      syncRerender();
      // Keyset paging cannot repeat a row; the guard stays for a stream that
      // reattaches and a page that overlap through the viewer's records.
      const have = new Set(galleryRecords.map((r) => r.id));
      page.items.forEach((rec) => {
        if (have.has(rec.id)) return;
        galleryRecords.push(rec);
        if (!sheetMode) els.gallery.appendChild(card(rec));
      });
      galleryMore = !!page.has_more;
      els.more.hidden = sheetMode || !galleryMore;
      if (sheetMode) renderSheet();
    })
    .catch((err) => {
      // A filter term that is not a number, most often: report it rather than
      // leaving the gallery silently showing the last good page.
      if (epoch === galleryEpoch) showError(err.message);
    })
    .finally(() => {
      galleryPending = null;
      els.more.disabled = false;
      if (els.lightbox.open) renderLightbox();  // re-derives the nav buttons
    });
  return galleryPending;
}

function card(rec) {
  const div = document.createElement('div');
  div.className = `card${rec.favorite ? ' fav' : ''}`;
  // Tagged by id so an action taken elsewhere - the lightbox, mostly - can
  // find this card and keep it in step.
  div.dataset.id = rec.id;

  const img = thumb(rec, galleryRecords);

  const star = tool(rec.favorite ? '★' : '☆', 'Favorite', () => setFavorite(rec, !rec.favorite));
  star.className = 'star';

  const tools = document.createElement('div');
  tools.className = 'tools';
  tools.append(
    tool('↻', 'Reuse settings', () => reuse(rec)),
    tool('⊞', 'Show this run as a sheet', () => {
      if (!rec.run_id && !rec.job_id) {
        showError('That image predates runs being grouped, so it belongs to no run.', true);
        return;
      }
      // The run id spans every job that worked on the grid - a night that died
      // and was resumed is one run - where the job id names only one of them.
      els.search.value = rec.run_id ? `run:${rec.run_id}` : `job:${rec.job_id}`;
      setSheet(true);
    }),
    star,
    tool('⌦', 'Delete', () => deleteImage(rec)),
  );

  div.append(img, tools);
  return div;
}

function tool(label, title, onClick) {
  const b = document.createElement('button');
  b.textContent = label;
  b.title = title;
  // A button is named by its contents, and `title` is never exposed - so these
  // announced as their glyph, or as a bare "button" for a glyph the screen
  // reader has no word for. The fourth of them deletes the image.
  b.setAttribute('aria-label', title);
  b.onclick = onClick;
  return b;
}

// The same image can be on the page more than once - in the viewer, in the
// gallery grid, on the sheet, in the lightbox - as separate record objects
// fetched at different times. So both actions apply by id: to every record
// that carries it, and to every element tagged with it.
function recordsWithId(id) {
  return [...galleryRecords, ...viewerRecords, ...lbList].filter((r) => r.id === id);
}

async function setFavorite(rec, value) {
  try {
    await api(`/api/images/${rec.id}/favorite?value=${value}`, { method: 'POST' });
  } catch (err) {
    showError(err.message);
    return;
  }
  recordsWithId(rec.id).forEach((r) => { r.favorite = value; });
  document.querySelectorAll(`.card[data-id="${rec.id}"]`).forEach((div) => {
    div.classList.toggle('fav', value);
    const star = div.querySelector('.star');
    star.textContent = value ? '★' : '☆';
    star.setAttribute('aria-label', value ? 'Unfavorite' : 'Favorite');
  });
  // The sheet's cell too - by class, never by re-rendering the sheet, which
  // rebuilds up to a thousand <img> nodes and resets the scroll per keypress.
  document.querySelectorAll(`.sheet-cell[data-id="${rec.id}"]`).forEach((cell) => {
    cell.classList.toggle('fav', value);
  });
  if (els.lightbox.open) renderLightbox();
}

// The delete removes the PNG as well as the row, and on a card the button sits
// a few pixels from the star. Say which image, so a slip is caught here.
function confirmDelete(rec) {
  const snippet = rec.prompt.length > 60 ? `${rec.prompt.slice(0, 57)}…` : rec.prompt;
  const what = `${rec.favorite ? 'favourite, ' : ''}seed ${rec.seed}`;
  return window.confirm(`Delete this image (${what})?\n\n"${snippet}"\n\nThe file is removed from disk.`);
}

async function deleteImage(rec) {
  if (!confirmDelete(rec)) return;
  try {
    await api(`/api/images/${rec.id}`, { method: 'DELETE' });
  } catch (err) {
    showError(err.message);
    return;
  }
  document.querySelectorAll(`[data-id="${rec.id}"]`).forEach((el) => el.remove());
  // Only the grid's counters mirror the server's: the viewer holds whatever
  // this session made, and the sheet lays itself out from the grid's records.
  const inGallery = galleryRecords.some((r) => r.id === rec.id);
  for (const list of new Set([galleryRecords, viewerRecords, lbList])) {
    const at = list.findIndex((r) => r.id === rec.id);
    if (at >= 0) list.splice(at, 1);
  }
  // The badge follows the server's count; the page anchor is an id, which a
  // delete does not move.
  if (inGallery) {
    total -= 1;
    els.count.textContent = total ? `(${total})` : '';
    syncRerender();
  }
  els.viewer.classList.toggle('multi', viewerRecords.length > 1);
  els.viewer.classList.toggle('many', viewerRecords.length > 4);
  if (sheetMode) renderSheet();
  if (!els.lightbox.open) return;
  // The strip has closed up under the viewer: the next image already sits at
  // this index, unless this was the last one.
  if (!lbList.length) closeLightbox();
  else { lbIndex = Math.min(lbIndex, lbList.length - 1); renderLightbox(); }
}

function reuse(rec) {
  ownControls();
  els.prompt.value = rec.prompt;
  els.negative.value = rec.negative_prompt ?? '';
  els.width.value = rec.width;
  els.height.value = rec.height;
  els.steps.value = rec.steps;
  els.cfg.value = rec.guidance_scale;
  // Turbo's flow shift. Records from a pipeline that derives its own carry null.
  els.mu.value = rec.mu ?? '';
  els.seed.value = rec.seed;
  syncSize();
  syncCfg();
  syncRuns();
  schedulePreview();
  persistPanel();
  // Steps, CFG and mu are checkpoint-specific: Turbo's 8 / 0 / 1.15 on FLUX is
  // a visibly broken image recorded under FLUX. The checkpoint is part of the
  // reproduction set as much as the seed, so say so - as a note, not a failure.
  const short = (id) => (id || '').split('/').pop();
  if (loadedModel && rec.model_id !== loadedModel) {
    showError(
      `These settings were made on ${short(rec.model_id)}; the server is running ` +
      `${short(loadedModel)}, whose steps, CFG and mu differ. Prompt, size and seed carry ` +
      'over; check the three sampling boxes before generating.',
      true,
    );
  } else {
    showError('');
  }
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
}

// ------------------------------------------------------------ contact sheet

// Everything a grid can vary. `of` reads one off a record; mu's null is a real
// value - "no shift of ours" - not a missing one, so it prints as `none`. The
// prompt is the one axis whose values are prose, which is why it is labelled
// rather than printed - see promptLabels.
const SHEET_AXES = [
  { key: 'prompt', label: 'Prompt', of: (r) => r.prompt },
  // A `{name}` placeholder may live in the negative prompt alone, which is a
  // legal axis the grid renders; without it here such a run collapsed to one
  // cell, every other axis having a single value.
  { key: 'negative', label: 'Negative', of: (r) => r.negative_prompt || null },
  { key: 'steps', label: 'Steps', of: (r) => r.steps },
  { key: 'cfg', label: 'CFG', of: (r) => r.guidance_scale },
  { key: 'mu', label: 'Mu', of: (r) => r.mu },
  { key: 'seed', label: 'Seed', of: (r) => r.seed },
];
// The axes in play for the current records. Records from a prompt grid carry
// the variable assignment that made them, and then each variable is an axis
// of its own - a `size` x `ground` night is a 2x2, not a 4-wide strip of
// finished sentences - and the flat prompt and negative axes are *suppressed*,
// not kept as fallbacks: a 4-value prompt axis would beat a 2-value variable
// axis in chooseAxes and turn a single-variable run into a diagonal.
let sheetAxesList = SHEET_AXES;
function axesFor(records) {
  const names = [];
  for (const rec of records) {
    for (const name of Object.keys(rec.variables || {})) if (!names.includes(name)) names.push(name);
  }
  if (!names.length) return SHEET_AXES;
  const variableAxes = names.map((name) => ({
    key: `var:${name}`,
    label: name,
    // `""` is the deliberately empty value - the word left out - and reads as
    // `none`; a record with no variables at all (a mixed search) reads as null.
    of: (r) => {
      const value = r.variables ? r.variables[name] : undefined;
      return value === undefined ? null : (value === '' ? 'none' : value);
    },
  }));
  return [...variableAxes, ...SHEET_AXES.filter((a) => a.key !== 'prompt' && a.key !== 'negative')];
}
const axisByKey = (key) => sheetAxesList.find((a) => a.key === key);
const axisValue = (axis, rec) => axis.of(rec) ?? null;

// Full prompt -> the words that make it different from its neighbours, filled
// in by sheetValues. Only the prompt axis is in here, so anything else that is
// a string prints as itself.
let promptShort = new Map();
const valueText = (v) => {
  if (v === null) return 'none';
  if (typeof v === 'string') return promptShort.get(v) ?? clip(v);
  return String(v);
};

const LABEL_CHARS = 44;
const clip = (text) => (text.length > LABEL_CHARS ? `${text.slice(0, LABEL_CHARS - 1)}…` : text);

// A header that repeats the forty words every prompt shares is unreadable, and
// the two or three that differ are the whole comparison - so the label keeps
// what varies and drops what does not.
//
// Same word count throughout means the varying positions can be picked out
// exactly, which is what a two-variable grid produces. A value that was set to
// `none` makes its prompt shorter than the others, and then the shared head and
// tail come off instead - `big dog on grass` against `dog on grass` labels
// itself `big` and `none`.
function promptLabels(prompts) {
  const words = prompts.map((p) => p.trim().split(/\s+/));
  const width = words[0].length;
  let kept;
  if (words.every((w) => w.length === width)) {
    const varies = [];
    for (let i = 0; i < width; i++) {
      if (!words.every((w) => w[i] === words[0][i])) varies.push(i);
    }
    kept = words.map((w) => varies.map((i) => w[i]));
  } else {
    let head = 0;
    while (words.every((w) => head < w.length && w[head] === words[0][head])) head += 1;
    let tail = 0;
    const same = (w) => w[w.length - 1 - tail] === words[0][words[0].length - 1 - tail];
    while (words.every((w) => tail < w.length - head && same(w))) tail += 1;
    kept = words.map((w) => w.slice(head, w.length - tail));
  }
  // Nothing left means this cell is the one that dropped the word the others
  // carry, which is a value in its own right and reads as one.
  return kept.map((w) => clip(w.join(' ')) || 'none');
}

// The distinct values each axis takes across the loaded images, ascending.
function sheetValues(records) {
  const values = {};
  for (const axis of sheetAxesList) {
    const seen = [];
    for (const rec of records) {
      const v = axisValue(axis, rec);
      if (!seen.includes(v)) seen.push(v);
    }
    seen.sort((a, b) => {
      if (a === null) return -1;
      if (b === null) return 1;
      return typeof a === 'string' ? a.localeCompare(b) : a - b;
    });
    values[axis.key] = seen;
  }
  // No prompt axis when variables carry the variation; nothing to label.
  const prompts = values.prompt || [];
  const labels = prompts.length > 1 ? promptLabels(prompts) : prompts.map(clip);
  promptShort = new Map(prompts.map((text, i) => [text, labels[i]]));
  return values;
}

// Columns and rows default to the two axes that actually vary, widest first.
// A third varying axis becomes a pin rather than a third dimension: a flat
// sheet that crosses everything at once is a wall nobody can read.
function chooseAxes(values, count) {
  let varying = sheetAxesList.filter((a) => values[a.key].length > 1);
  // An axis with one value per image is a list, not a dimension - the seed of
  // a mixed gallery, mostly. It is dropped whenever anything else varies; a
  // seeds-only run keeps it, since it is then the only dimension there is.
  const listlike = varying.filter((a) => values[a.key].length === count);
  if (listlike.length < varying.length) varying = varying.filter((a) => !listlike.includes(a));
  varying.sort((a, b) => values[b.key].length - values[a.key].length);
  return {
    cols: varying[0] ? varying[0].key : null,
    rows: varying[1] ? varying[1].key : null,
    pins: {},
  };
}

// Every varying axis is exactly one of columns, rows or a pin, and every pin
// holds a value that is actually present. Run on every render: promoting a
// pinned axis to Columns used to leave it in the pins too, so only one of its
// columns could ever fill; and switching the search to another run of the same
// shape kept a pin on a value the new records never had - a blank sheet under
// a dropdown showing a different value than the one filtering it.
function reconcileAxes(values) {
  const pins = {};
  for (const axis of sheetAxesList) {
    const key = axis.key;
    if (values[key].length < 2 || key === sheetAxes.cols || key === sheetAxes.rows) continue;
    const kept = sheetAxes.pins[key];
    pins[key] = values[key].includes(kept) ? kept : values[key][0];
  }
  sheetAxes.pins = pins;
}

// Picking for one side the axis already on the other swaps them rather than
// collapsing the sheet to one dimension twice; reconcileAxes then re-pins
// whatever the swap displaced.
function setAxis(side, key) {
  const other = side === 'cols' ? 'rows' : 'cols';
  if (key && key === sheetAxes[other]) sheetAxes[other] = sheetAxes[side];
  sheetAxes[side] = key || null;
  renderSheet();
}

function headCell(text, className) {
  const div = document.createElement('div');
  div.className = className || 'head';
  div.textContent = text;
  return div;
}

function labelled(text, control) {
  const wrap = document.createElement('label');
  wrap.className = 'sheet-pick';
  wrap.append(document.createTextNode(text), control);
  return wrap;
}

function picker(options, selected, onPick) {
  const select = document.createElement('select');
  options.forEach(([value, text]) => {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = text;
    option.selected = String(value) === String(selected);
    select.appendChild(option);
  });
  select.onchange = () => onPick(select.value);
  return select;
}

// Past this many cells the layout is a list, not a matrix - and each cell used
// to cost a scan of every record, so the Sheet button over an unfiltered
// gallery froze the tab for seconds and laid out a grid 100,000 px wide.
const MAX_SHEET_CELLS = 4096;
function renderSheet() {
  sheetRecords = galleryRecords;
  sheetAxesList = axesFor(sheetRecords);
  const values = sheetValues(sheetRecords);
  // Re-derive columns and rows when the image set changes shape, and only
  // then, so switching filters lands on something sensible while a hand-picked
  // axis survives a redraw. Pins are reconciled every time - see above. The
  // axis names are part of the shape: a variable run and a plain one with the
  // same cardinalities are different sheets.
  const signature = sheetAxesList.map((a) => `${a.key}:${values[a.key].length}`).join('/');
  if (sheetAxes.signature !== signature) {
    sheetAxes = { ...chooseAxes(values, sheetRecords.length), signature };
  }
  reconcileAxes(values);

  const axisOptions = [['', '—']].concat(
    sheetAxesList.filter((a) => values[a.key].length > 1).map((a) => [a.key, a.label]),
  );
  els.sheetControls.innerHTML = '';
  els.sheetControls.append(
    labelled('Columns', picker(axisOptions, sheetAxes.cols ?? '', (key) => setAxis('cols', key))),
    labelled('Rows', picker(axisOptions, sheetAxes.rows ?? '', (key) => setAxis('rows', key))),
  );
  for (const [key, pinned] of Object.entries(sheetAxes.pins)) {
    const axis = axisByKey(key);
    els.sheetControls.append(
      // By position, not by the text on the option: two prompts shorten to two
      // different labels, but nothing guarantees a label round-trips back to
      // the value it came from.
      labelled(axis.label, picker(
        values[key].map((v, index) => [index, valueText(v)]),
        values[key].indexOf(pinned),
        (picked) => {
          sheetAxes.pins[key] = values[key][Number(picked)] ?? null;
          renderSheet();
        },
      )),
    );
  }

  const colAxis = sheetAxes.cols ? axisByKey(sheetAxes.cols) : null;
  const rowAxis = sheetAxes.rows ? axisByKey(sheetAxes.rows) : null;
  const colValues = colAxis ? values[colAxis.key] : [null];
  const rowValues = rowAxis ? values[rowAxis.key] : [null];
  els.sheetGrid.innerHTML = '';
  if (colValues.length * rowValues.length > MAX_SHEET_CELLS) {
    els.sheetNote.textContent =
      `${colValues.length} × ${rowValues.length} cells is a list, not a sheet. ` +
      'Narrow the search to one run — the ⊞ button on a card does that.';
    return;
  }

  // One pass over the records into a map keyed by (row, column), rather than a
  // scan of every record per cell, so the sheet costs O(records + cells). The
  // first record wins a cell; they arrive newest first.
  const pins = Object.entries(sheetAxes.pins).map(([key, v]) => [axisByKey(key), v]);
  const rowIndex = new Map(rowValues.map((v, i) => [v, i]));
  const colIndex = new Map(colValues.map((v, i) => [v, i]));
  const cellOf = new Map();
  for (const rec of sheetRecords) {
    if (pins.some(([axis, v]) => axisValue(axis, rec) !== v)) continue;
    const r = rowAxis ? rowIndex.get(axisValue(rowAxis, rec)) : 0;
    const c = colAxis ? colIndex.get(axisValue(colAxis, rec)) : 0;
    const at = r * colValues.length + c;
    if (!cellOf.has(at)) cellOf.set(at, rec);
  }

  const shown = [];
  els.sheetGrid.style.gridTemplateColumns =
    `auto repeat(${colValues.length}, minmax(96px, 1fr))`;
  // One aspect ratio for the whole sheet - size is not a sheet axis, and a
  // free-text search can mix sizes - so the commonest one wins and the odd
  // cell letterboxes inside it rather than being cropped.
  const ratios = new Map();
  for (const rec of sheetRecords) {
    const key = `${rec.width} / ${rec.height}`;
    ratios.set(key, (ratios.get(key) || 0) + 1);
  }
  const [aspect] = [...ratios.entries()].sort((a, b) => b[1] - a[1])[0] || ['1'];
  els.sheetGrid.style.setProperty('--cell-aspect', aspect);
  els.sheetGrid.appendChild(headCell(''));
  // A prompt label is prose rather than a number, so its header is allowed to
  // wrap; nowrap would push one column out to the width of the whole phrase.
  // A variable's values are prose too.
  const prose = (axis) => (
    axis && (axis.key === 'prompt' || axis.key === 'negative' || axis.key.startsWith('var:')) ? ' text' : ''
  );
  colValues.forEach((v) => {
    els.sheetGrid.appendChild(
      headCell(colAxis ? `${colAxis.label} ${valueText(v)}` : '', `head${prose(colAxis)}`),
    );
  });

  rowValues.forEach((rowValue, r) => {
    els.sheetGrid.appendChild(
      headCell(
        rowAxis ? `${rowAxis.label} ${valueText(rowValue)}` : '',
        `rowhead${prose(rowAxis)}`,
      ),
    );
    colValues.forEach((_, c) => {
      const rec = cellOf.get(r * colValues.length + c);
      const cell = document.createElement('div');
      cell.className = rec ? `sheet-cell${rec.favorite ? ' fav' : ''}` : 'sheet-cell empty';
      if (rec) {
        shown.push(rec);
        // `shown` is still filling, but it is the same array by the time
        // anything can be clicked - so the arrows walk the sheet in reading
        // order rather than the gallery's.
        cell.appendChild(thumb(rec, shown));
        cell.dataset.id = rec.id;
      }
      els.sheetGrid.appendChild(cell);
    });
  });

  const pinned = pins
    .map(([axis, v]) => `${axis.label.toLowerCase()} ${valueText(v)}`)
    .join(', ');
  els.sheetNote.textContent = sheetRecords.length === 0
    ? 'Nothing matches. The filters above the sheet are the gallery search.'
    : `${shown.length} of ${sheetRecords.length} loaded` + (pinned ? ` · pinned to ${pinned}` : '') +
      (total > sheetRecords.length ? ` · ${total} match the search` : '');
}

function setSheet(on) {
  sheetMode = on;
  els.sheetBtn.classList.toggle('active', on);
  els.sheet.hidden = !on;
  els.gallery.hidden = on;
  // Off is a real escape hatch: whatever layout and pins were picked go with it.
  if (!on) sheetAxes = { rows: null, cols: null, pins: {} };
  return resetGallery();
}

// ----------------------------------------------------------------- lightbox

function openLightbox(rec, list) {
  lbList = list && list.includes(rec) ? list : [rec];
  lbIndex = lbList.indexOf(rec);
  renderLightbox();
}

// The view: a scale relative to the fit size (1 = fit, never below), and the
// image centre's offset from the stage centre in CSS pixels. It is keyed on the
// image's dimensions and kept across records that share them, so the arrow keys
// compare the identical crop at the identical magnification across the cells of
// a sweep - the whole point of zooming in on a steps or mu axis. A record of
// another size starts at fit; so does every fresh open.
const lbView = { scale: 1, x: 0, y: 0, key: null };
const ZOOM_STEP = 1.25;

// Fit is "contain": the largest size that shows the whole image inside the
// stage, taken from the record's own dimensions rather than the decoded image,
// so the layout is right before the PNG has arrived.
function fitSize(rec) {
  const k = Math.min(els.lbStage.clientWidth / rec.width, els.lbStage.clientHeight / rec.height);
  return [rec.width * k, rec.height * k];
}

// The scale at which one image pixel is one *device* pixel. Dividing by the
// devicePixelRatio is load-bearing: at 125-150% Windows scaling a plain
// natural-size image is a 1.25-1.5x resample, which is precisely the smoothing
// a sharpening sweep is being judged on.
function oneToOneScale(rec) {
  return (rec.width / window.devicePixelRatio) / fitSize(rec)[0];
}

function layoutLightbox() {
  const rec = lbList[lbIndex];
  if (!rec || !els.lightbox.open) return;
  const [fw, fh] = fitSize(rec);
  const w = fw * lbView.scale, h = fh * lbView.scale;
  const sw = els.lbStage.clientWidth, sh = els.lbStage.clientHeight;
  // An image smaller than the stage on an axis is centred on it; one larger
  // can be panned until its edge meets the stage's, and no further.
  const maxX = Math.max(0, (w - sw) / 2), maxY = Math.max(0, (h - sh) / 2);
  lbView.x = Math.min(maxX, Math.max(-maxX, lbView.x));
  lbView.y = Math.min(maxY, Math.max(-maxY, lbView.y));
  Object.assign(els.lbImg.style, {
    width: `${w}px`, height: `${h}px`,
    left: `${(sw - w) / 2 + lbView.x}px`, top: `${(sh - h) / 2 + lbView.y}px`,
  });
  const one = oneToOneScale(rec);
  const percent = Math.round((lbView.scale / one) * 100);
  els.lbZoom.textContent = lbView.scale <= 1 ? `fit · ${percent}%`
    : Math.abs(lbView.scale - one) < 1e-3 ? '1:1' : `${percent}%`;
  els.lbStage.classList.toggle('zoomed', lbView.scale > 1);
}

// Zoom by a factor about a point (cx, cy) given relative to the stage centre,
// keeping the image under that point where it is - the wheel zooms about the
// pointer, the keys about the middle. Never below fit; capped at 8x device
// pixels, past which the picture is squares.
function zoomLightbox(factor, cx = 0, cy = 0) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  const ceiling = Math.max(1, oneToOneScale(rec) * 8);
  const next = Math.min(ceiling, Math.max(1, lbView.scale * factor));
  const ratio = next / lbView.scale;
  lbView.x = cx - ratio * (cx - lbView.x);
  lbView.y = cy - ratio * (cy - lbView.y);
  lbView.scale = next;
  if (next <= 1) lbView.x = lbView.y = 0;
  layoutLightbox();
}

function zoomFit() { lbView.scale = 1; lbView.x = lbView.y = 0; layoutLightbox(); }
function zoomOneToOne(cx = 0, cy = 0) {
  const rec = lbList[lbIndex];
  if (rec) zoomLightbox(oneToOneScale(rec) / lbView.scale, cx, cy);
}

// Pointer position relative to the stage centre, for zooming about it.
function stagePoint(e) {
  const box = els.lbStage.getBoundingClientRect();
  return [e.clientX - box.left - box.width / 2, e.clientY - box.top - box.height / 2];
}

function renderLightbox() {
  const rec = lbList[lbIndex];
  if (!rec) { closeLightbox(); return; }
  const key = `${rec.width}x${rec.height}`;
  if (lbView.key !== key) {
    lbView.scale = 1; lbView.x = lbView.y = 0; lbView.key = key;
  }
  els.lbImg.src = rec.url;
  els.lbImg.alt = rec.prompt;
  els.lbMeta.textContent =
    // The checkpoint is part of the reproduction set as much as the seed is -
    // the same prompt and settings on Turbo and on FLUX are different images.
    `${rec.prompt} — ${rec.model_id.split('/').pop()}, ` +
    `${rec.width}×${rec.height}, ${rec.steps} steps, ` +
    `CFG ${rec.guidance_scale}, ${rec.mu == null ? 'no mu' : `mu ${rec.mu}`}, ` +
    `seed ${rec.seed}, ${(rec.duration_ms / 1000).toFixed(1)}s` +
    (lbList.length > 1 ? ` · ${lbIndex + 1}/${lbList.length}` : '');
  els.lbPrev.disabled = lbIndex === 0;
  els.lbNext.disabled = lbIndex === lbList.length - 1 && !canPage();
  els.lbFav.textContent = rec.favorite ? '★ Favourite' : '☆ Favourite';
  els.lbFav.classList.toggle('active', !!rec.favorite);
  // Open first: the stage has no size while the dialog is display: none.
  if (!els.lightbox.open) els.lightbox.showModal();
  layoutLightbox();
}

// Only the gallery strip has more behind it, and only while the server says so.
function canPage() { return lbList === galleryRecords && galleryMore; }

async function stepLightbox(delta) {
  if (!els.lightbox.open) return;
  const current = lbList[lbIndex];
  if (lbIndex + delta < 0) return;
  if (lbIndex + delta >= lbList.length) {
    // Walking off the end of the grid pulls the next page rather than stopping.
    if (!canPage()) return;
    await loadGallery();
    // The user may have moved on during the await - ArrowLeft is synchronous -
    // and this step then belongs to an image no longer in view: drop it rather
    // than re-imposing its own start and landing two images ahead.
    if (lbList[lbIndex] !== current) return;
    // The strip can have moved during the await - a delete, another page - so
    // step from where this image is now rather than from a stale index.
    const at = lbList.indexOf(current);
    if (at >= 0) lbIndex = at;
    if (lbIndex + delta >= lbList.length) return;
  }
  lbIndex += delta;
  renderLightbox();
}

function closeLightbox() {
  // close() fires the dialog's `close` event, which is wired back here - the
  // guard is what keeps that from recursing, and what makes a native Escape
  // (which closes the dialog first) still reset the strip.
  if (els.lightbox.open) els.lightbox.close();
  els.lbImg.src = '';
  els.lbError.hidden = true;
  lbList = [];
  lbIndex = -1;
  // A fresh open starts at fit; the view is only carried between images of a
  // walk, not between sessions of looking.
  lbView.key = null;
}
els.lightbox.addEventListener('close', closeLightbox);

// Wheel: zoom about the pointer. Non-passive, or the page behind the dialog
// scrolls with it.
els.lbStage.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoomLightbox(Math.exp(-e.deltaY * 0.002), ...stagePoint(e));
}, { passive: false });

// Drag to pan. A press that never moved is a click: on the stage beside the
// image, it closes the lightbox - but only at the fit size, where nothing is
// being panned and the stage is the backdrop it used to be. Zoomed in, the
// whole stage is a pan surface and only Escape or the cross close it.
let lbDrag = null;
els.lbStage.onpointerdown = (e) => {
  if (e.button !== 0) return;
  lbDrag = { x: e.clientX, y: e.clientY, moved: false };
  els.lbStage.setPointerCapture(e.pointerId);
};
els.lbStage.onpointermove = (e) => {
  if (!lbDrag) return;
  const dx = e.clientX - lbDrag.x, dy = e.clientY - lbDrag.y;
  if (!lbDrag.moved && Math.hypot(dx, dy) < 4) return;  // a jittery click is a click
  lbDrag.moved = true;
  els.lbStage.classList.add('dragging');
  lbView.x += dx; lbView.y += dy;
  lbDrag.x = e.clientX; lbDrag.y = e.clientY;
  layoutLightbox();
};
els.lbStage.onpointerup = (e) => {
  const drag = lbDrag;
  lbDrag = null;
  els.lbStage.classList.remove('dragging');
  if (!drag || drag.moved) return;
  if (e.target === els.lbStage && lbView.scale <= 1) closeLightbox();
};
els.lbStage.onpointercancel = () => { lbDrag = null; els.lbStage.classList.remove('dragging'); };
// Double-click: 1:1 about the point, or back to fit from anywhere else.
els.lbImg.ondblclick = (e) => {
  if (lbView.scale > 1) zoomFit();
  else zoomOneToOne(...stagePoint(e));
};
els.lbFit.onclick = zoomFit;
els.lbOne.onclick = () => zoomOneToOne();
window.addEventListener('resize', layoutLightbox);

// --------------------------------------------------------------------- wire

let searchTimer;
els.generate.onclick = generate;
els.cancel.onclick = cancelJob;
els.pause.onclick = togglePause;
els.cancelQueued.onclick = cancelQueued;
els.dice.onclick = () => { els.seed.value = ''; syncRuns(); schedulePreview(); persistPanel(); };
els.unload.onclick = async () => {
  try {
    await api('/api/model/unload', { method: 'POST' });
    showError('');
  } catch (err) {
    showError(err.message);  // 409 while a job is still running
  }
  refreshStatus();
};
els.more.onclick = () => loadGallery();
els.gridRun.onclick = () => startJob('/api/grid', gridPayload());
// The preview reflects the panel as much as the axes: a blank axis is the
// control's own value, and both the size and the batch move the totals.
// The negative prompt is in the list because a `{name}` placeholder may live
// there - the server validates it - and Resume because the whole "N already
// done" half of the preview is what ticking it is for.
for (const el of [els.gridPrompt, els.gridSteps, els.gridCfg, els.gridMu, els.gridSeeds,
                  els.gridSeedList, els.gridBudget, els.prompt, els.negative, els.steps,
                  els.cfg, els.mu, els.batch, els.width, els.height, els.seed]) {
  el.addEventListener('input', schedulePreview);
}
// A listed seed axis makes the count and the base seed moot; say so in place.
els.gridSeedList.addEventListener('input', () => {
  els.gridSeeds.disabled = els.gridSeedList.value.trim() !== '';
});
els.load.onclick = async () => {
  try {
    await api('/api/model/load', { method: 'POST' });
    showError('');
  } catch (err) {
    showError(err.message);
  }
  refreshStatus();
};
els.gridResume.addEventListener('change', previewGrid);
els.gridSection.addEventListener('toggle', previewGrid);
els.width.oninput = els.height.oninput = syncSize;
// These run before the panel's delegated persist handler on the same event -
// the target's listener fires before its ancestors' - so the `owned` flags
// are current by the time the panel is written.
els.steps.oninput = () => own('steps');
els.mu.oninput = () => own('mu');
// The seed is not a sampling value and has no business freezing the profile.
els.seed.oninput = syncRuns;
els.batch.oninput = els.runs.oninput = syncRuns;
els.cfg.oninput = () => { own('cfg'); syncCfg(); };
els.favOnly.onchange = resetGallery;
els.sheetBtn.onclick = () => setSheet(!sheetMode);
els.rerender.onclick = rerenderSelection;
els.runList.onchange = () => { showRun(els.runList.value); els.runList.value = ''; };
els.lastRunShow.onclick = () => {
  const { run, job } = els.lastRun.dataset;
  if (run) showRun(run);
  else if (job) { els.search.value = `job:${job}`; setSheet(true); }
  dismissLastRun();
};
els.lastRunDismiss.onclick = dismissLastRun;
els.search.oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(resetGallery, 250); };
els.lbClose.onclick = closeLightbox;
els.lbPrev.onclick = () => stepLightbox(-1);
els.lbNext.onclick = () => stepLightbox(1);
// The just-generated image only exists in the viewer, so reuse has to be
// reachable from the lightbox as well as from a gallery card.
els.lbReuse.onclick = () => { const rec = lbList[lbIndex]; if (rec) { reuse(rec); closeLightbox(); } };
els.lbFav.onclick = () => { const rec = lbList[lbIndex]; if (rec) setFavorite(rec, !rec.favorite); };
els.lbDelete.onclick = () => { const rec = lbList[lbIndex]; if (rec) deleteImage(rec); };
els.lightbox.onclick = (e) => { if (e.target === els.lightbox) closeLightbox(); };

document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd+Enter generates from anywhere, including inside the prompt box.
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { generate(); return; }
  if (!els.lightbox.open) return;
  // Arrows walk whichever strip the lightbox was opened from. Nothing else on
  // the page reads them while it is up, so the caret keys are free to take.
  if (e.key === 'Escape') { closeLightbox(); return; }
  if (e.key === 'ArrowLeft') { e.preventDefault(); stepLightbox(-1); return; }
  if (e.key === 'ArrowRight') { e.preventDefault(); stepLightbox(1); return; }
  // F, Delete and the zoom keys act on the image in view - unless a text field
  // still holds the caret, where they are typing.
  if (e.target.matches('input, textarea')) return;
  if (e.key === 'f' || e.key === 'F') els.lbFav.click();
  else if (e.key === 'Delete') els.lbDelete.click();
  else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomLightbox(ZOOM_STEP); }
  else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomLightbox(1 / ZOOM_STEP); }
  else if (e.key === '0') zoomFit();
  else if (e.key === '1') zoomOneToOne();
});

buildPresets();
// Before the sync functions, which light the presets, the CFG note and the
// runs hint up from whatever is in the boxes.
restorePanel();
syncSize();
syncCfg();
syncRuns();
// Every keystroke and tick in the panel, plus the programmatic writes below
// - presets, Reuse, the dice, the seed written back after Run grid - which
// fire no input event.
document.querySelector('.panel').addEventListener('input', persistPanel);
document.querySelector('.panel').addEventListener('change', persistPanel);
els.gridSection.addEventListener('toggle', persistPanel);
refreshStatus();
loadGallery();
loadRuns();
resumeActiveJob();
setInterval(refreshStatus, 3000);
