// Local Diffusion's page script: panel persistence, the last-run banner and the results viewer. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

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
const PANEL_KEY = 'local-diffusion.panel.v1';
const LEGACY_PANEL_KEYS = ['krea.panel.v1'];
const PANEL_FIELDS = [
  'prompt', 'negative', 'width', 'height', 'seed', 'batch', 'runs', 'lora', 'loraScale',
  'gridPrompt', 'gridSteps', 'gridCfg', 'gridMu', 'gridSeeds', 'gridSeedList', 'gridBudget',
  'gridLoraScale',
];
let panelTimer;

function persistPanel() {
  clearTimeout(panelTimer);
  panelTimer = setTimeout(() => {
    const state = { gridOpen: els.gridSection.open, reference: referenceIds };
    for (const key of PANEL_FIELDS) state[key] = els[key].value;
    // Sampling values travel with the checkpoint they were set on, and only
    // the ones the user took off the profile - the profile itself is the
    // server's to give back.
    state.sampling = {
      model: loadedModel,
      steps: owned.steps ? els.steps.value : null,
      cfg: owned.cfg ? els.cfg.value : null,
      mu: owned.mu ? els.mu.value : null,
      trueCfg: owned.trueCfg ? els.trueCfg.value : null,
    };
    remember(PANEL_KEY, JSON.stringify(state));
  }, 300);
}

function storedPanel() {
  let raw = remembered(PANEL_KEY);
  if (!raw) {
    for (const key of LEGACY_PANEL_KEYS) {
      raw = remembered(key);
      if (!raw) continue;
      remember(PANEL_KEY, raw);
      forget(key);
      break;
    }
  }
  try { return JSON.parse(raw || 'null'); } catch { return null; }
}

function restorePanel() {
  const state = storedPanel();
  if (!state) return;
  for (const key of PANEL_FIELDS) {
    if (typeof state[key] === 'string') els[key].value = state[key];
  }
  // Set again by refreshLoras once the options exist.
  pendingLora = typeof state.lora === 'string' ? state.lora : '';
  els.gridSection.open = !!state.gridOpen;
  els.gridSeeds.disabled = els.gridSeedList.value.trim() !== '';
  // The ids are restored whatever the checkpoint; `collect()` sends none while
  // the field is greyed, so a klein selection survives a look at FLUX and back.
  if (Array.isArray(state.reference)) referenceIds = state.reference.filter(Number.isInteger);
  syncConditioning();
}

// Reset: every box back to what a first visit shows - the checkpoint's own
// steps/CFG/mu, the 1 MP square, a random seed, no conditioning, an empty and
// folded grid - and the browser forgets the remembered panel. It is the one
// way to empty the panel short of wiping the site's data: clearing the gallery
// is a server-side deletion that never reaches this store, so the old prompt
// used to come back on every open after it.
function resetPanel() {
  clearTimeout(panelTimer);
  forget(PANEL_KEY);
  for (const key of LEGACY_PANEL_KEYS) forget(key);
  // The Refine dialog's scale, versions and per-checkpoint strengths too.
  forget(REFINE_KEY);
  for (const key of Object.keys(owned)) owned[key] = false;
  els.prompt.value = '';
  promptReplaced();
  // A first visit has no check open, and one about the prompt just emptied
  // would only be stale.
  clearAnalysis();
  els.negative.value = '';
  [pickedAspect, pickedTier] = ['1:1', 1];
  [els.width.value, els.height.value] = SIZES[pickedAspect][pickedTier];
  els.steps.value = lastDefaults?.steps ?? 28;
  els.cfg.value = lastDefaults?.guidance_scale ?? 3.5;
  els.mu.value = lastDefaults?.mu ?? '';
  els.trueCfg.value = '';
  referenceIds = [];
  els.lora.value = '';
  els.loraScale.value = 1;
  els.seed.value = '';
  els.batch.value = 1;
  els.runs.value = 1;
  for (const key of ['gridPrompt', 'gridSteps', 'gridCfg', 'gridMu', 'gridSeedList', 'gridBudget',
                     'gridLoraScale']) {
    els[key].value = '';
  }
  els.gridSeeds.value = 1;
  els.gridSeeds.disabled = false;
  els.gridResume.checked = false;
  els.gridSection.open = false;
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  schedulePreview();
  scheduleTagValidation(0);
  els.prompt.focus();
}

// The second phase, from refreshStatus once the checkpoint is known.
let samplingRestored = false;
function restoreSampling(modelId) {
  if (samplingRestored) return;
  samplingRestored = true;
  const sampling = storedPanel()?.sampling;
  if (!sampling || sampling.model !== modelId) return;
  for (const [key, el] of [['steps', els.steps], ['cfg', els.cfg], ['mu', els.mu], ['trueCfg', els.trueCfg]]) {
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
function forget(key) { try { localStorage.removeItem(key); } catch { /* unavailable */ } }

const ANNOUNCED_KEY = 'local-diffusion.announced-job';
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

  const images = `${last.image_count} image${last.image_count === 1 ? '' : 's'}` +
    (typeof last.distinct_count === 'number' && last.image_count > 1 && last.distinct_count < last.image_count
      ? `, only ${last.distinct_count} distinct` : '');
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
  for (const key of LEGACY_ANNOUNCED_KEYS) forget(key);
  els.lastRun.hidden = true;
}

// One line per run in the gallery, newest first. Picking one is the card's ⊞
// button without having to find a card: it searches for the run and lays it
// out as a sheet.
async function loadRuns() {
  let runs;
  // Under the gallery's scope: a run picked here opens as a sheet, and a run
  // the grid would not show is an empty one.
  const params = new URLSearchParams({ limit: 20 });
  if (els.allModels.checked) params.set('all_models', 'true');
  try { runs = await api(`/api/runs?${params}`); } catch { return; }
  els.runList.innerHTML = '';
  els.runList.append(new Option(runs.length ? 'Runs…' : 'No runs yet', ''));
  for (const run of runs) {
    const stars = run.favorites ? ` · ${run.favorites}★` : '';
    // A run that made fewer pictures than images says so in the list.
    const distinct = typeof run.distinct === 'number' && run.distinct < run.images
      ? ` (${run.distinct} distinct)` : '';
    const label = `${when(Date.parse(run.started_at))} · ${run.images} img${distinct}${stars} · ${clip(run.prompt)}`;
    els.runList.append(new Option(label, run.run_id));
  }
  els.runList.disabled = !runs.length;
}

// `fresh` is a comparison this page has just made: nothing in it is
// starred yet, so under the Favorites tick its sheet opened empty. The tick
// comes off rather than being bypassed - the gallery's filters are the
// sheet's, and Re-render sends the tick with the search, so a sheet of
// unstarred images under a ticked box would re-render none of them - and the
// sheet says so.
function showRun(runId, { fresh = false } = {}) {
  if (!runId) return;
  showSheetScope(`run:${runId}`, { fresh });
}

function showJob(jobId, { fresh = false } = {}) {
  if (!jobId) return;
  showSheetScope(`job:${jobId}`, { fresh });
}

function showSheetScope(search, { fresh = false } = {}) {
  const unticked = fresh && els.favOnly.checked;
  if (unticked) els.favOnly.checked = false;
  els.search.value = search;
  setSheet(true);
  // setSheet has begun the query the notice belongs to; another one drops it.
  if (unticked) sheetNotice = { epoch: galleryEpoch, text: 'Favorites unticked: nothing in a new comparison is starred yet' };
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
