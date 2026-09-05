'use strict';

const $ = (id) => document.getElementById(id);
const PAGE = 48;
// A sheet wants the whole run in one go - it is a matrix, and a matrix with a
// page boundary through it is a puzzle. The rows are small and the thumbnails
// load lazily, so a full grid is cheaper than it sounds.
const SHEET_PAGE = 1024;

const els = {
  prompt: $('prompt'), negative: $('negative'), negField: $('negative-field'),
  negNote: $('neg-note'),
  width: $('width'), height: $('height'), steps: $('steps'), cfg: $('cfg'),
  mu: $('mu'), seed: $('seed'), batch: $('batch'), runs: $('runs'),
  stepsTip: $('steps-tip'), cfgTip: $('cfg-tip'), muTip: $('mu-tip'),
  generate: $('btn-generate'), dice: $('btn-dice'), unload: $('btn-unload'),
  cancel: $('btn-cancel'),
  progress: $('progress'), fill: $('progress-fill'), progressText: $('progress-text'),
  stepRow: $('step-row'), stepFill: $('step-fill'), stepText: $('step-text'),
  cellNote: $('cell-note'),
  gridSection: $('grid-section'), gridSteps: $('grid-steps'), gridCfg: $('grid-cfg'),
  gridMu: $('grid-mu'), gridSeeds: $('grid-seeds'), gridPreview: $('grid-preview'),
  gridRun: $('btn-grid'),
  error: $('error'), viewer: $('viewer'), gallery: $('gallery'),
  count: $('gallery-count'), search: $('search'), favOnly: $('fav-only'),
  more: $('btn-more'), presets: $('presets'), tiers: $('tiers'), sizeHint: $('size-hint'),
  gridResume: $('grid-resume'),
  sheetBtn: $('btn-sheet'), sheet: $('sheet'), sheetControls: $('sheet-controls'),
  sheetGrid: $('sheet-grid'), sheetNote: $('sheet-note'),
  runHint: $('run-hint'),
  dot: $('status-dot'), statusText: $('status-text'), vram: $('vram'),
  lightbox: $('lightbox'), lbImg: $('lb-img'), lbMeta: $('lb-meta'), lbClose: $('lb-close'),
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

let offset = 0;
let total = 0;
let busy = false;
let defaultsApplied = false;
let statusPending = false;
let currentJob = null;
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
  els.error.textContent = message || '';
  els.error.hidden = !message;
  els.error.classList.toggle('note', note);
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
}

function syncSize() {
  const w = +els.width.value, h = +els.height.value;
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

function syncCfg() {
  const off = !negativeSupported || +els.cfg.value <= cfgThreshold;
  els.negField.classList.toggle('disabled', off);
  els.negative.disabled = off;
  els.negNote.textContent = negativeSupported
    ? `(needs CFG > ${cfgThreshold} — ~2× slower)`
    : '(this checkpoint takes no negative prompt)';
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
// it derives its own resolution-dependent shift. Applied once, before the user
// has had a chance to touch anything.
function applyDefaults(d) {
  if (defaultsApplied || !d || d.steps === undefined) return;
  defaultsApplied = true;
  els.steps.value = d.steps;
  els.cfg.value = d.guidance_scale;
  els.mu.value = d.mu ?? '';
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

// Anything the user picks - by hand or through Reuse - takes the controls off
// the profile. /api/status reads CUDA memory, which stalls behind a model load,
// so its reply can land minutes late and used to wipe steps/CFG/mu long after
// the page had settled.
function ownControls() { defaultsApplied = true; }

async function refreshStatus() {
  if (statusPending) return;  // a stalled poll must not queue more behind it
  statusPending = true;
  try {
    const s = await api('/api/status');
    applyDefaults(s.defaults);
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
    els.dot.className = `dot ${s.state}`;
    els.statusText.textContent =
      s.state === 'ready' ? s.model_id.split('/').pop()
      : s.state === 'loading' ? 'loading weights…'
      : s.state === 'error' ? 'load failed'
      : 'not loaded';
    if (s.state === 'error' && s.detail) showError(s.detail);
    els.vram.textContent = s.vram_total_mb
      ? `${s.device_name ?? ''} · ${(s.vram_used_mb / 1024).toFixed(1)}/${(s.vram_total_mb / 1024).toFixed(1)} GB · ${s.quantization}`
      : '';
  } catch { /* server not up yet */ } finally { statusPending = false; }
}

// --------------------------------------------------------------- generation

function collect() {
  const seed = els.seed.value.trim();
  return {
    prompt: els.prompt.value.trim(),
    negative_prompt: els.negative.value.trim() || null,
    width: +els.width.value,
    height: +els.height.value,
    steps: +els.steps.value,
    guidance_scale: +els.cfg.value,
    mu: els.mu.value === '' ? null : +els.mu.value,
    seed: seed === '' ? null : +seed,
    num_images: +els.batch.value,
    runs: +els.runs.value,
  };
}

function setBusy(on) {
  busy = on;
  els.generate.disabled = on;
  els.gridRun.disabled = on;
  // Unloading mid-job frees nothing - the worker holds its own reference to the
  // pipeline - and the server refuses it with a 409 anyway.
  els.unload.disabled = on;
  els.generate.textContent = on ? 'Generating…' : 'Generate';
  els.progress.hidden = !on;
  els.cancel.disabled = !on;
  els.cancel.textContent = 'Cancel';
  els.fill.style.width = '0%';
  setStep(false);
  els.cellNote.textContent = '';
  if (!on) { els.progressText.textContent = ''; currentJob = null; }
}

// Which cell of a grid is running, under the bars. A sweep sends no label -
// its settings never change - and the line stays empty rather than repeating
// the counters that are already there.
function showCell(data) {
  els.cellNote.textContent = data.label ? `cell ${data.run}/${data.total_runs} · ${data.label}` : '';
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
  if (busy) return;
  if (!payload.prompt) { showError('Enter a prompt first.'); return; }
  showError('');
  setBusy(true);
  els.progressText.textContent = 'queued…';

  try {
    const { job_id } = await api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    currentJob = job_id;
    await streamJob(job_id);
  } catch (err) {
    showError(err.message);
    setBusy(false);
  }
}

function generate() { return startJob('/api/generate', collect()); }

// ------------------------------------------------------------- grid search

// The axes sit on top of the panel's own settings, so a blank one contributes
// the value already there. `runs` is not a grid axis - the seed count is - and
// the server refuses a grid that sets both.
function gridPayload() {
  return {
    ...collect(),
    runs: 1,
    steps_axis: els.gridSteps.value.trim(),
    cfg_axis: els.gridCfg.value.trim(),
    mu_axis: els.gridMu.value.trim(),
    seeds: +els.gridSeeds.value || 1,
    resume: els.gridResume.checked,
  };
}

function humanTime(seconds) {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${minutes} min`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

const axisText = (values) => values.map((v) => (v === null ? 'none' : v)).join(', ');

// The preview is the server's own expansion rather than a count worked out in
// here: the syntax has one parser, and the message a bad axis produces is the
// one the run itself would have refused with.
async function previewGrid() {
  if (!els.gridSection.open) return;
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
    els.gridPreview.classList.remove('warn');
    // Totals on the first line, the resolved axes on the second: what a range
    // actually expanded to is the half of it that is easy to get wrong.
    els.gridPreview.textContent = [
      `${p.images} image${p.images === 1 ? '' : 's'} · ${p.summary}` +
        (p.skipped ? ` · ${p.skipped} already done` : '') +
        (p.seconds ? ` · about ${humanTime(p.seconds)}` : ''),
      `steps ${axisText(p.values.steps)}  ·  CFG ${axisText(p.values.guidance_scale)}` +
        `  ·  mu ${axisText(p.values.mu)}`,
    ].join('\n');  // .hint.pre renders the break
  } catch (err) {
    els.gridPreview.classList.add('warn');
    els.gridPreview.textContent = err.message;
  }
}

let gridTimer;
function schedulePreview() { clearTimeout(gridTimer); gridTimer = setTimeout(previewGrid, 250); }

// The server keeps whatever finished before the cancel landed, so this reports
// a count rather than treating the run as lost.
async function cancelJob() {
  if (!currentJob) return;
  els.cancel.disabled = true;
  els.cancel.textContent = 'Stopping…';
  try {
    await api(`/api/jobs/${currentJob}/cancel`, { method: 'POST' });
    els.progressText.textContent = 'finishing this step…';
  } catch (err) {
    showError(err.message);
    els.cancel.disabled = false;
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
        els.fill.style.width = `${((done + within) / runs) * 100}%`;
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
        els.progressText.textContent =
          data.status === 'queued' ? 'waiting for the job ahead…'
          : runs > 1 && data.run ? `run ${data.run}/${runs}`
          : 'denoising…';
        // A run boundary, so the steps bar starts over rather than sitting full
        // until the first callback of the next run lands.
        setStep(runs > 1 && data.status !== 'queued' && !!data.run);
        showCell(data);
      } else if (data.type === 'done') {
        source.close();
        setBusy(false);
        if (data.images) data.images.forEach(show);
        if (data.status === 'error') showError(data.error);
        else if (data.status === 'cancelled') {
          showError(data.images.length
            ? `Cancelled — kept ${data.images.length} image${data.images.length === 1 ? '' : 's'}.`
            : 'Cancelled before anything finished.', true);
        }
        if (data.images.length) resetGallery();
        resolve();
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
    const job = active[active.length - 1];
    if (!job) return;
    setBusy(true);
    currentJob = job.job_id;
    els.progressText.textContent = 'reattaching…';
    await streamJob(job.job_id);
  } catch { /* nothing running, or the server is not up yet */ }
}

function clearResults() {
  viewerRecords = [];
  els.viewer.innerHTML = '';
  els.viewer.classList.remove('multi', 'many');
}

function appendResult(rec) {
  viewerRecords.push(rec);
  // Two size tiers, so a long sweep does not push the gallery a screen further
  // down with every image it finishes. The lightbox is where they get looked at.
  els.viewer.classList.toggle('multi', viewerRecords.length > 1);
  els.viewer.classList.toggle('many', viewerRecords.length > 4);
  const img = document.createElement('img');
  img.src = rec.url;
  img.alt = rec.prompt;
  img.dataset.id = rec.id;
  img.onclick = () => openLightbox(rec, viewerRecords);
  els.viewer.appendChild(img);
}

// -------------------------------------------------------------------gallery

function resetGallery() {
  // Anything already in flight was fetched under the old filters and against
  // the grid this is about to empty; the token tells it to drop itself.
  galleryEpoch += 1;
  offset = 0;
  galleryRecords = [];
  els.gallery.innerHTML = '';
  return loadGallery();
}

function loadGallery() {
  // One page at a time. Nothing else serialises this - a double-clicked "Load
  // more", or an arrow key held down in the lightbox - and two calls reading
  // the same `offset` append the same 48 cards twice, then advance past a page
  // that is never fetched.
  if (galleryPending) return galleryPending;
  const epoch = galleryEpoch;
  const params = new URLSearchParams({ limit: sheetMode ? SHEET_PAGE : PAGE, offset });
  if (els.search.value.trim()) params.set('search', els.search.value.trim());
  if (els.favOnly.checked) params.set('favorites', 'true');
  els.more.disabled = true;
  els.lbNext.disabled = true;

  galleryPending = api(`/api/gallery?${params}`)
    .then((page) => {
      if (epoch !== galleryEpoch) return;  // superseded by a reset
      total = page.total;
      els.count.textContent = total ? `(${total})` : '';
      page.items.forEach((rec) => {
        galleryRecords.push(rec);
        if (!sheetMode) els.gallery.appendChild(card(rec));
      });
      offset += page.items.length;
      els.more.hidden = sheetMode || offset >= total;
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
      if (!els.lightbox.hidden) renderLightbox();  // re-derives the nav buttons
    });
  return galleryPending;
}

function card(rec) {
  const div = document.createElement('div');
  div.className = `card${rec.favorite ? ' fav' : ''}`;
  // Tagged by id so an action taken elsewhere - the lightbox, mostly - can
  // find this card and keep it in step.
  div.dataset.id = rec.id;

  const img = document.createElement('img');
  img.src = rec.url;
  img.alt = rec.prompt;
  img.loading = 'lazy';
  img.onclick = () => openLightbox(rec, galleryRecords);

  const star = tool(rec.favorite ? '★' : '☆', 'Favorite', () => setFavorite(rec, !rec.favorite));
  star.className = 'star';

  const tools = document.createElement('div');
  tools.className = 'tools';
  tools.append(
    tool('↻', 'Reuse settings', () => reuse(rec)),
    tool('⊞', 'Show this run as a sheet', () => {
      if (!rec.job_id) {
        showError('That image predates runs being grouped, so it belongs to no run.', true);
        return;
      }
      els.search.value = `job:${rec.job_id}`;
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
    div.querySelector('.star').textContent = value ? '★' : '☆';
  });
  if (!els.lightbox.hidden) renderLightbox();
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
  if (inGallery) {
    total -= 1;
    offset -= 1;
    els.count.textContent = total ? `(${total})` : '';
  }
  els.viewer.classList.toggle('multi', viewerRecords.length > 1);
  els.viewer.classList.toggle('many', viewerRecords.length > 4);
  if (sheetMode) renderSheet();
  if (els.lightbox.hidden) return;
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
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ------------------------------------------------------------ contact sheet

// Everything a grid can vary. `of` reads one off a record; mu's null is a real
// value - "no shift of ours" - not a missing one, so it prints as `none`.
const SHEET_AXES = [
  { key: 'steps', label: 'Steps', of: (r) => r.steps },
  { key: 'cfg', label: 'CFG', of: (r) => r.guidance_scale },
  { key: 'mu', label: 'Mu', of: (r) => r.mu },
  { key: 'seed', label: 'Seed', of: (r) => r.seed },
];
const axisByKey = (key) => SHEET_AXES.find((a) => a.key === key);
const axisValue = (axis, rec) => axis.of(rec) ?? null;
const valueText = (v) => (v === null ? 'none' : String(v));

// The distinct values each axis takes across the loaded images, ascending.
function sheetValues(records) {
  const values = {};
  for (const axis of SHEET_AXES) {
    const seen = [];
    for (const rec of records) {
      const v = axisValue(axis, rec);
      if (!seen.includes(v)) seen.push(v);
    }
    seen.sort((a, b) => (a === null ? -1 : b === null ? 1 : a - b));
    values[axis.key] = seen;
  }
  return values;
}

// Columns and rows default to the two axes that actually vary, widest first.
// A third varying axis becomes a pin rather than a third dimension: a flat
// sheet that crosses everything at once is a wall nobody can read.
function chooseAxes(values) {
  const varying = SHEET_AXES
    .filter((a) => values[a.key].length > 1)
    .sort((a, b) => values[b.key].length - values[a.key].length);
  const pins = {};
  varying.slice(2).forEach((a) => { pins[a.key] = values[a.key][0]; });
  return {
    cols: varying[0] ? varying[0].key : null,
    rows: varying[1] ? varying[1].key : null,
    pins,
  };
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

function renderSheet() {
  sheetRecords = galleryRecords;
  const values = sheetValues(sheetRecords);
  // Re-derive the layout when the image set changes shape, and only then, so
  // switching filters lands on something sensible while a hand-picked axis
  // survives a redraw.
  const signature = SHEET_AXES.map((a) => values[a.key].length).join('/');
  if (sheetAxes.signature !== signature) sheetAxes = { ...chooseAxes(values), signature };

  const axisOptions = [['', '—']].concat(
    SHEET_AXES.filter((a) => values[a.key].length > 1).map((a) => [a.key, a.label]),
  );
  els.sheetControls.innerHTML = '';
  els.sheetControls.append(
    labelled('Columns', picker(axisOptions, sheetAxes.cols ?? '', (key) => {
      // Swap rather than collapse: picking the axis already on the other side
      // would otherwise leave the sheet with one dimension twice.
      if (key && key === sheetAxes.rows) sheetAxes.rows = sheetAxes.cols;
      sheetAxes.cols = key || null;
      renderSheet();
    })),
    labelled('Rows', picker(axisOptions, sheetAxes.rows ?? '', (key) => {
      if (key && key === sheetAxes.cols) sheetAxes.cols = sheetAxes.rows;
      sheetAxes.rows = key || null;
      renderSheet();
    })),
  );
  for (const [key, pinned] of Object.entries(sheetAxes.pins)) {
    const axis = axisByKey(key);
    els.sheetControls.append(
      labelled(axis.label, picker(
        values[key].map((v) => [valueText(v), valueText(v)]),
        valueText(pinned),
        (picked) => {
          sheetAxes.pins[key] = values[key].find((v) => valueText(v) === picked) ?? null;
          renderSheet();
        },
      )),
    );
  }

  const colValues = sheetAxes.cols ? values[sheetAxes.cols] : [null];
  const rowValues = sheetAxes.rows ? values[sheetAxes.rows] : [null];
  const shown = [];
  const matches = (rec, rowValue, colValue) => {
    for (const [key, pinned] of Object.entries(sheetAxes.pins)) {
      if (axisValue(axisByKey(key), rec) !== pinned) return false;
    }
    if (sheetAxes.rows && axisValue(axisByKey(sheetAxes.rows), rec) !== rowValue) return false;
    if (sheetAxes.cols && axisValue(axisByKey(sheetAxes.cols), rec) !== colValue) return false;
    return true;
  };

  els.sheetGrid.innerHTML = '';
  els.sheetGrid.style.gridTemplateColumns =
    `auto repeat(${colValues.length}, minmax(96px, 1fr))`;
  els.sheetGrid.appendChild(headCell(''));
  const colAxis = sheetAxes.cols ? axisByKey(sheetAxes.cols) : null;
  colValues.forEach((v) => {
    els.sheetGrid.appendChild(headCell(colAxis ? `${colAxis.label} ${valueText(v)}` : ''));
  });

  const rowAxis = sheetAxes.rows ? axisByKey(sheetAxes.rows) : null;
  rowValues.forEach((rowValue) => {
    els.sheetGrid.appendChild(
      headCell(rowAxis ? `${rowAxis.label} ${valueText(rowValue)}` : '', 'rowhead'),
    );
    colValues.forEach((colValue) => {
      const rec = sheetRecords.find((r) => matches(r, rowValue, colValue));
      const cell = document.createElement('div');
      cell.className = rec ? 'sheet-cell' : 'sheet-cell empty';
      if (rec) {
        shown.push(rec);
        const img = document.createElement('img');
        img.src = rec.url;
        img.alt = rec.prompt;
        img.loading = 'lazy';
        // `shown` is still filling, but it is the same array by the time
        // anything can be clicked - so the arrows walk the sheet in reading
        // order rather than the gallery's.
        img.onclick = () => openLightbox(rec, shown);
        cell.appendChild(img);
      }
      els.sheetGrid.appendChild(cell);
    });
  });

  const pinned = Object.entries(sheetAxes.pins)
    .map(([key, v]) => `${axisByKey(key).label.toLowerCase()} ${valueText(v)}`)
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
  return resetGallery();
}

// ----------------------------------------------------------------- lightbox

function openLightbox(rec, list) {
  lbList = list && list.includes(rec) ? list : [rec];
  lbIndex = lbList.indexOf(rec);
  renderLightbox();
}

function renderLightbox() {
  const rec = lbList[lbIndex];
  if (!rec) { closeLightbox(); return; }
  els.lbImg.src = rec.url;
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
  els.lightbox.hidden = false;
}

// Only the gallery strip has more behind it, and only while a page is unread.
function canPage() { return lbList === galleryRecords && offset < total; }

async function stepLightbox(delta) {
  if (els.lightbox.hidden) return;
  const current = lbList[lbIndex];
  if (lbIndex + delta < 0) return;
  if (lbIndex + delta >= lbList.length) {
    // Walking off the end of the grid pulls the next page rather than stopping.
    if (!canPage()) return;
    await loadGallery();
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
  els.lightbox.hidden = true;
  els.lbImg.src = '';
  lbList = [];
  lbIndex = -1;
}

// --------------------------------------------------------------------- wire

let searchTimer;
els.generate.onclick = generate;
els.cancel.onclick = cancelJob;
els.dice.onclick = () => { els.seed.value = ''; syncRuns(); };
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
for (const el of [els.gridSteps, els.gridCfg, els.gridMu, els.gridSeeds, els.prompt,
                  els.steps, els.cfg, els.mu, els.batch, els.width, els.height, els.seed]) {
  el.addEventListener('input', schedulePreview);
}
els.gridSection.addEventListener('toggle', previewGrid);
els.width.oninput = els.height.oninput = syncSize;
els.steps.oninput = els.mu.oninput = ownControls;
els.seed.oninput = () => { ownControls(); syncRuns(); };
els.batch.oninput = els.runs.oninput = syncRuns;
els.cfg.oninput = () => { ownControls(); syncCfg(); };
els.favOnly.onchange = resetGallery;
els.sheetBtn.onclick = () => setSheet(!sheetMode);
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
  if (els.lightbox.hidden) return;
  // Arrows walk whichever strip the lightbox was opened from. Nothing else on
  // the page reads them while it is up, so the caret keys are free to take.
  if (e.key === 'Escape') { closeLightbox(); return; }
  if (e.key === 'ArrowLeft') { e.preventDefault(); stepLightbox(-1); return; }
  if (e.key === 'ArrowRight') { e.preventDefault(); stepLightbox(1); return; }
  // F and Delete act on the image in view - unless a text field still holds
  // the caret, where they are typing.
  if (e.target.matches('input, textarea')) return;
  if (e.key === 'f' || e.key === 'F') els.lbFav.click();
  else if (e.key === 'Delete') els.lbDelete.click();
});

buildPresets();
syncSize();
syncCfg();
syncRuns();
refreshStatus();
loadGallery();
resumeActiveJob();
setInterval(refreshStatus, 3000);
