// Local Diffusion's page script: panel persistence and its sticky top, the last-run banner and the results viewer. Declarations only; state.js has
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

// The panel as it is remembered: every field, the fold, the references, and
// the sampling values taken off the profile. Reuse keeps one too, for its Undo.
function panelState() {
  const state = { gridOpen: els.gridSection.open, reference: [...referenceIds] };
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
  return state;
}

function persistPanel() {
  clearTimeout(panelTimer);
  panelTimer = setTimeout(() => remember(PANEL_KEY, JSON.stringify(panelState())), 300);
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

// ------------------------------------------------------------ the sticky top

// The panel's sticky top (style.css has why it is not the stylesheet's alone),
// kept here in px and written inline. While the whole panel fits the window it
// rests under the topbar, at the stylesheet's top. While it does not, the top
// lies anywhere from there down to the top that holds its bottom a gutter above
// the window's - the gutter the page ends with - and moves within that range
// four ways:
//   - with the user's own scrolling (`followScroll`), as far as the page moves:
//     down the gallery the panel goes up until its bottom (Generate, Pause and
//     Cancel, the error line, Run grid) is in view and holds there, and back up
//     it comes down until its top (the prompt) is. A scroll the browser makes
//     on its own moves nothing: scroll anchoring holding the gallery still as
//     the viewer above it fills or empties, or the page shortening under a
//     reloaded gallery, would otherwise slide the panel - at a run's start,
//     Generate from under the pointer that just pressed it.
//   - with its height or the window's (`placePanel`), keeping the bottom where
//     it was while the bottom was in view, else the top. While the pointer is
//     over the panel nothing moves (`panelResized`): the progress row, the
//     queue note and the error line appear below Generate, and a panel held by
//     its bottom would lift Generate by as much, putting Pause and Cancel under
//     the pointer that pressed it - a second press meant to queue another run
//     stopped the first. The placement waits for the pointer to leave.
//   - with focus (`revealInPanel`): a control focused from the keyboard or by
//     the page, out of sight above the window, under the topbar or below the
//     window, is brought into view by moving the panel, not the page. A
//     focused control left out of sight (WCAG 2.4.11) was the cost of holding
//     the panel by its bottom: the browser scrolls the page to a focused
//     control, and a page scroll does not move a stuck panel.
//   - with the grid fold opened by hand (`revealGridRun`).
const panelPos = {
  top: null,          // the inline top in px; null until placed, and while the panel is static
  rest: 0,            // the stylesheet's top, --sticky-top, in px
  gap: 0,             // between a held panel's bottom and the window's: main's padding
  height: 0,          // the panel's, at the last placement
  window: 0,          // innerHeight, at the last placement
  deferred: false,    // a placement waiting for the pointer to leave the panel
  pointer: false,     // the pointer is over the panel
  press: false,       // a press in the panel is moving focus: the control is where it was pointed at
  scrollY: 0,         // at the last scroll event
  scrolling: false,   // a scroll of the user's is under way
  dragging: false,    // the page's scrollbar is held
  timer: 0,
};

// The range the top moves in: down to the one that holds the panel's bottom a
// gutter above the window's (never below resting, for a panel that fits), up
// to resting.
function setPanelTop(top) {
  const low = Math.min(panelPos.rest, innerHeight - panelPos.height - panelPos.gap);
  panelPos.top = Math.min(panelPos.rest, Math.max(low, top));
  document.querySelector('.panel').style.top = `${panelPos.top}px`;
}

// A change of the panel's height or the window's. The stylesheet's top is
// read with the inline one cleared, so a tier that changes --sticky-top or the
// gutter needs nothing here.
function placePanel() {
  const panel = document.querySelector('.panel');
  panel.style.top = '';
  const css = getComputedStyle(panel);
  panelPos.deferred = false;
  if (css.position !== 'sticky') { panelPos.top = null; return; }
  if (panelPos.top === null) panelPos.scrollY = scrollY;
  const keepBottom = panelPos.top === null
    || panelPos.top + panelPos.height <= panelPos.window - panelPos.gap + 0.5;
  panelPos.rest = parseFloat(css.top);
  panelPos.gap = parseFloat(getComputedStyle(panel.parentElement).paddingBottom);
  panelPos.height = panel.getBoundingClientRect().height;
  panelPos.window = innerHeight;
  setPanelTop(keepBottom ? -Infinity : panelPos.top);
}

// The ResizeObserver's: the panel grew or shrank.
function panelResized() {
  if (panelPos.pointer && panelPos.top !== null) panelPos.deferred = true;
  else placePanel();
}

// What the user does to scroll: the wheel, a touch, the keys that scroll the
// page from outside any control, a press on the page's scrollbar (Chrome sends
// it to the root element) or the middle button's autoscroll - not Space on
// Generate, whose run empties the viewer and has the page anchored by as much
// within the window below. Each opens a window that every scroll event of the
// movement keeps open, and `scrollend` or a quiet 400 ms closes, unless the
// scrollbar is still held.
const SCROLL_KEYS = new Set(['PageDown', 'PageUp', 'ArrowDown', 'ArrowUp', ' ', 'Home', 'End']);
function userScrolls() {
  panelPos.scrolling = true;
  clearTimeout(panelPos.timer);
  panelPos.timer = setTimeout(() => { panelPos.scrolling = panelPos.dragging; }, 400);
}

function followScroll() {
  const moved = scrollY - panelPos.scrollY;
  panelPos.scrollY = scrollY;
  if (!panelPos.scrolling || panelPos.top === null || !moved) return;
  userScrolls();
  if (panelPos.deferred) placePanel();
  setPanelTop(panelPos.top - moved);
}

// The control's block - its field, its row, the button - is brought between
// where the first block sits while the panel rests and where the last sits
// while it is held, as far as the control itself stays in view. A panel still
// in the page's flow, near the top of the page, does not move for its top
// alone; there the top is only lowered far enough that the page's scroll to
// the control carries the panel up the window, rather than sticking it on the
// way.
//
// That scroll has happened by the time focusin fires: the browser scrolls the
// page to a control before it dispatches the focus events, measuring a stuck
// panel as though it moved with the page, so the page jumped and the control
// stayed out of sight. Its scroll event comes only at the next frame, so the
// position the panel last saw is the one before the focus. The scroll is taken
// back - clicking a finding in the prompt check should not carry the check
// away - and made again only if moving the panel left the control out of sight:
// a panel still in the page's flow, or held at the end of the page.
function revealInPanel(el) {
  if (panelPos.top === null) return;
  if (panelPos.deferred) placePanel();
  const jumped = scrollY - panelPos.scrollY;
  if (jumped) scrollTo(scrollX, panelPos.scrollY);
  const panel = document.querySelector('.panel');
  const css = getComputedStyle(panel);
  const top = panelPos.rest + parseFloat(css.borderTopWidth) + parseFloat(css.paddingTop);
  const bottom = innerHeight - panelPos.gap - parseFloat(css.borderBottomWidth) - parseFloat(css.paddingBottom);
  const box = panel.getBoundingClientRect();
  const block = (el.closest('.panel > *, .section > *') ?? el).getBoundingClientRect();
  const own = el.getBoundingClientRect();
  const down = block.top < top ? Math.min(top - block.top, Math.max(0, bottom - own.bottom)) : 0;
  const up = !down && own.bottom > bottom ? Math.min(own.bottom - bottom, own.top - top) : 0;
  if (down > 0) setPanelTop(Math.max(panelPos.top, box.top + down));
  else if (up > 0) setPanelTop(Math.min(panelPos.top, box.top - up));
  if (!jumped) return;
  const now = el.getBoundingClientRect();
  if (now.top < document.querySelector('.topbar').getBoundingClientRect().bottom || now.bottom > innerHeight) {
    scrollTo(scrollX, panelPos.scrollY + jumped);
  }
}

// Opening the grid fold by hand brings Run grid into view: the fold opens
// downwards, past the window's bottom on most screens, and nothing in sight
// says it did. The panel is placed first - the fold has just made it taller,
// and the ResizeObserver would report that only after the frame, and then
// under the pointer that opened it - and the page scrolls where the panel is
// still in its flow, at the top of the page.
function revealGridRun() {
  if (!els.gridSection.open) return;
  placePanel();
  revealInPanel(els.gridRun);
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  els.gridRun.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' });
}

// A help's tip opens under its badge, and over it where under would run past
// the window's bottom and over has more room: the grid fold's last rows sit
// at the window's bottom while the panel is held there, and their tips ran
// off it. Measured as the tip opens, shown for the measurement, since it is
// display: none until the badge is hovered or focused.
function placeTip(help) {
  const tip = help.querySelector('.tip');
  if (!tip) return;
  help.classList.remove('up');
  tip.style.display = 'block';
  const badge = help.getBoundingClientRect();
  const shown = tip.getBoundingClientRect();
  tip.style.display = '';
  const above = badge.top - (shown.top - badge.bottom) - document.querySelector('.topbar').getBoundingClientRect().bottom;
  if (shown.bottom > innerHeight && above > innerHeight - shown.top) help.classList.add('up');
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
  img.onerror = () => { void forgetIfGone(rec.id); };
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
