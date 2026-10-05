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
// What else is remembered beside the fields, merged key by key as they are.
const PANEL_KEYS = [...PANEL_FIELDS, 'gridOpen', 'reference', 'translation'];
let panelTimer;

// The panel as it is remembered: every field, the fold, the references, the
// English an expansion came with (ERNIE's, with the prompt it translates),
// and the sampling values taken off the profile. Reuse and Reset keep one too,
// for their Undo.
function panelState() {
  const state = {
    gridOpen: els.gridSection.open,
    reference: [...referenceIds],
    translation: translatedSource !== null ? { source: translatedSource, text: els.promptTranslation.value } : null,
  };
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
  // A change since a Reuse or a Reset retires its Undo (gallery.js, below).
  syncReuseUndo();
  syncResetUndo();
  clearTimeout(panelTimer);
  panelTimer = setTimeout(writePanel, 300);
}

// Every tab of the page shares the one store, and each wrote its whole panel
// on every change: a second tab, opened earlier, that changed only its Seed
// wrote its own empty prompt, grid and steps over the first tab's, which were
// gone at that tab's next load. So a write puts back only what this page
// changed since it last read or wrote the store (`panelBase`), key by key -
// each field, the fold, the references, the translation, each sampling value
// - onto what the store holds now. An empty store - a first visit, another
// tab's Reset - takes the whole panel, sampling's checkpoint with it: written
// key by key into nothing, the sampling values lost the model they belong to.
// Nothing listens to `storage`: what another tab wrote stays the store's
// until this page changes the same key.
let panelBase = null;
// The panel as a Reset left it, until it changes: a write of that is no
// change, and the store stays forgotten - the prompt's blur after a Reset
// fires `change`, and wrote the reset panel back, which another tab then
// merged its own changes over rather than taking whole.
let resetLeft = null;
function writePanel() {
  const now = panelState();
  if (resetLeft !== null) {
    if (JSON.stringify(now) === resetLeft) return;
    resetLeft = null;
  }
  const stored = storedPanel();
  let out = now;
  if (stored && panelBase) {
    const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
    out = { ...stored };
    for (const key of PANEL_KEYS) {
      if (!same(now[key], panelBase[key])) out[key] = now[key];
    }
    out.sampling = { ...(stored.sampling ?? {}) };
    for (const key of Object.keys(now.sampling)) {
      if (!same(now.sampling[key], panelBase.sampling?.[key])) out.sampling[key] = now.sampling[key];
    }
  }
  remember(PANEL_KEY, JSON.stringify(out));
  panelBase = now;
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
  // What a later write compares this page against: the store as read - the
  // sampling half is applied once the checkpoint is known (restoreSampling),
  // and the panel's own values before then would make it look changed - or
  // the page as it opens, where nothing was stored.
  panelBase = state ?? panelState();
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
  // An expansion's English, while the prompt is still the one it translates:
  // shown once the status says this checkpoint expands prompts (status.js).
  const t = state.translation;
  if (t && typeof t.source === 'string' && typeof t.text === 'string' && t.source === els.prompt.value) {
    pendingTranslation = t;
  }
  syncConditioning();
}

// Reset: every box back to what a first visit shows - the checkpoint's own
// steps/CFG/mu, the 1 MP square, a random seed, no conditioning, an empty and
// folded grid - and the browser forgets the remembered panel. It is the one
// way to empty the panel short of wiping the site's data: clearing the gallery
// is a server-side deletion that never reaches this store, so the old prompt
// used to come back on every open after it.
//
// One press took the prompt and a grid spec that has to be retyped exactly
// for Resume to match, with nothing to bring them back: Ctrl+Z restored
// nothing and a reload kept the reset. So the panel as it was is kept in
// memory - from the page, not the store, which is 300 ms behind and absent
// in a private window - and an Undo follows the reset for a while
// (`offerResetUndo`). Nothing of it survives a reload, and Resume is never
// put back ticked.
function resetPanel() {
  const before = {
    state: panelState(),
    values: { steps: els.steps.value, cfg: els.cfg.value, mu: els.mu.value, trueCfg: els.trueCfg.value },
    owned: { ...owned },
    size: [pickedAspect, pickedTier],
    refine: remembered(REFINE_KEY),
    unsent: unsentRefine,
  };
  // The Refine dialog's scale, versions and per-checkpoint strengths too, and
  // anything left unsent in it.
  forgetRefine();
  for (const key of Object.keys(owned)) owned[key] = false;
  // Emptied as typing would empty them (`rewriteBox`), so Ctrl+Z in the
  // prompt, which Reset focuses, still gives its text back once the Undo has
  // gone: assigning `.value` emptied the box's undo history.
  rewriteBox(els.negative, '');
  rewriteBox(els.prompt, '');
  promptReplaced();
  // A first visit has no check open, and one about the prompt just emptied
  // would only be stale.
  clearAnalysis();
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
  // Not persisted, but a change all the same: a Reuse's Undo goes, as an
  // Example's does (scheduleTagValidation).
  syncReuseUndo();
  // Forgotten last: the boxes' writes above fire `input` and `change`, which
  // persist the panel (app.js) and would write the store back 300 ms on.
  clearTimeout(panelTimer);
  forget(PANEL_KEY);
  for (const key of LEGACY_PANEL_KEYS) forget(key);
  // And kept forgotten until the panel changes (writePanel): the prompt's
  // `change` comes at its next blur, after this. The panel as reset is what
  // a forgotten store means, so it is what a later write compares against -
  // never the panel before the Reset, whose every key would look changed
  // and go over what another tab wrote meanwhile.
  panelBase = panelState();
  resetLeft = JSON.stringify(panelBase);
  offerResetUndo(before);
  els.prompt.focus();
}

// The note a Reset leaves, with its Undo, where Reset was pressed: under the
// prompt (tags.js, showPromptNote). In the panel's line under Generate it
// was below the window at 1400x900 and in one column, and gone after 20
// seconds unseen. On offer while the panel is as the Reset left it - the
// first change takes it, as a Reuse's Undo goes - and for 20 seconds at most.
let resetNote = null;
const RESET_UNDO_MS = 20000;

function offerResetUndo(before) {
  const note = { before, after: JSON.stringify(panelState()), entry: null, undone: false };
  note.entry = offerPromptUndo('Panel reset.', () => undoReset(note), {
    kind: 'reset',
    title: 'Put the panel back as it was before Reset',
    ms: RESET_UNDO_MS,
    onDrop: () => { if (resetNote === note) resetNote = null; },
  });
  resetNote = note;
}

function retireResetUndo(note) {
  if (resetNote !== note) return;
  resetNote = null;
  if (promptUndo === note.entry) dropPromptUndo();
}

// Asked by persistPanel, which every change of the panel reaches.
function syncResetUndo() {
  if (resetNote && JSON.stringify(panelState()) !== resetNote.after) retireResetUndo(resetNote);
}

// The panel as Reset found it: the boxes through `rewriteBox`, so Ctrl+Z in
// either walks back to the reset, then every field, the fold, the references,
// the sampling values with whose they were, the size's preset, what the
// Refine dialog remembered, and an expansion's English where the prompt is
// the one it translates. The sync functions light the panel up from them.
function undoReset(note) {
  if (note.undone) return;
  note.undone = true;
  retireResetUndo(note);
  const { state, values } = note.before;
  rewriteBox(els.negative, state.negative);
  rewriteBox(els.prompt, state.prompt);
  for (const key of PANEL_FIELDS) {
    if (key !== 'prompt' && key !== 'negative') els[key].value = state[key];
  }
  els.gridSection.open = state.gridOpen;
  els.gridSeeds.disabled = els.gridSeedList.value.trim() !== '';
  referenceIds = state.reference;
  Object.assign(owned, note.before.owned);
  els.steps.value = values.steps;
  els.cfg.value = values.cfg;
  els.mu.value = values.mu;
  els.trueCfg.value = values.trueCfg;
  [pickedAspect, pickedTier] = note.before.size;
  if (note.before.refine !== null) remember(REFINE_KEY, note.before.refine);
  unsentRefine = note.before.unsent;
  promptReplaced();
  const t = state.translation;
  if (t && els.prompt.value === t.source) showPromptTranslation(t.source, t.text);
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  showPromptNote('Reset undone: the panel is as it was.', { kind: 'info' });
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
// Sideways, it moves in by as much as it would run past either edge of the
// window (TIP_EDGE short of it), whichever way the stylesheet hangs it: the
// Search tip opened rightward past a 600 px window and widened the page, and
// the grid fold's LoRA weight tip, hung leftward (.help.end) for the panel's
// sake, ran past the window's left edge at 1400 and 2560. A shift of its own,
// written inline and cleared at each placement, never the markup's `end`,
// whose badges must keep hanging left inside the panel.
const TIP_EDGE = 8;
function placeTip(help) {
  const tip = help.querySelector('.tip');
  if (!tip) return;
  help.classList.remove('up');
  tip.style.removeProperty('translate');
  tip.style.display = 'block';
  const badge = help.getBoundingClientRect();
  const shown = tip.getBoundingClientRect();
  tip.style.display = '';
  const above = badge.top - (shown.top - badge.bottom) - document.querySelector('.topbar').getBoundingClientRect().bottom;
  if (shown.bottom > innerHeight && above > innerHeight - shown.top) help.classList.add('up');
  const right = document.documentElement.clientWidth - TIP_EDGE;
  const past = shown.right > right ? right - shown.right : 0;
  const shift = shown.left + past < TIP_EDGE ? TIP_EDGE - shown.left : past;
  // Rounded away from the edge it clears: rounded to nearest, a 17.4 px shift
  // left a tip 7.6 px from the window's left edge.
  if (Math.abs(shift) >= 1) tip.style.translate = `${shift > 0 ? Math.ceil(shift) : Math.floor(shift)}px 0`;
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
    (last.error ? ` ${last.error}` : '');
  els.lastRun.classList.toggle('bad', last.status === 'error');
  els.lastRun.dataset.job = last.job_id;
  els.lastRun.dataset.run = last.run_id || '';
  els.lastRunShow.hidden = !last.image_count || deleted === last.image_count;
  // A grid the night did not finish - cancelled, out of time, stopped by its
  // failures - is finished by Resume. Not a re-render's, a refine sweep's or
  // a comparison's, which the panel's grid does not make (the sheet's Resume
  // this run refuses a comparison too), and not one with no image of its own
  // left: nothing of it for the preview to skip, and the preview then said
  // the panel's grid was another.
  lastRunJob = last;
  els.lastRunResume.hidden = !(last.grid && last.prompt_template != null && last.refine == null
    && !last.comparison && last.image_count > deleted
    && (last.status === 'cancelled' || last.stopped_by === 'failures'));
  els.lastRun.hidden = false;
}

// The job the banner names, for its Resume.
let lastRunJob = null;

function dismissLastRun() {
  remember(ANNOUNCED_KEY, els.lastRun.dataset.job || '');
  for (const key of LEGACY_ANNOUNCED_KEYS) forget(key);
  els.lastRun.hidden = true;
}

// The banner's Resume: the sheet's Resume this run (sheet.js) - the panel's
// grid made ready, and its preview saying what is left - with the job's size.
// Not its seed, which is the first cell it still had to run: a job that was
// itself a Resume, the run's first seeds done, carries one past the base, and
// that grid is another, with another run id.
function resumeLastRun() {
  const job = lastRunJob;
  if (!job) return;
  dismissLastRun();
  void resumeRun(job.run_id, { width: job.width, height: job.height });
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
  steps: 'steps', cfg: 'CFG', mu: 'Mu', true_cfg: 'True CFG', lora_scale: 'LoRA', reference: 'refs',
};
function runAxes(run) {
  const axes = run.axes ?? {};
  if (axes.variant) return `comparison · ${axes.variant} versions`;
  if (run.refine != null) {
    const [low, high] = run.strengths ?? [];
    const at = low == null ? '' : low === high ? `strength ${low} ` : `strength ${low}-${high} `;
    return `${at}of #${run.refine}`;
  }
  const varied = Object.keys(RUN_AXIS_NAMES).filter((key) => axes[key] > 1);
  const seeds = axes.seed > 1 ? `${axes.seed} seeds` : '';
  if (!varied.length) return seeds;
  const names = varied.map((key) => RUN_AXIS_NAMES[key]);
  const counts = varied.map((key) => axes[key]);
  const grid = varied.length === 1 ? `${names[0]} ×${counts[0]}` : `${names.join('×')} ${counts.join('×')}`;
  return [grid, seeds].filter(Boolean).join(' · ');
}

async function loadRuns() {
  let runs;
  // Under the gallery's scope: a run picked here opens as a sheet, and a run
  // the grid would not show is an empty one.
  const params = new URLSearchParams({ limit: 200 });
  if (els.allModels.checked) params.set('all_models', 'true');
  try { runs = await api(`/api/runs?${params}`); } catch { return; }
  const several = (run) => run.images > 1 || run.jobs > 1;
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
  els.runList.title = els.runList.value ? els.runList.selectedOptions[0].text : 'Recent runs';
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
  if (e.key === 'Enter') {
    if (runPending !== null) { e.preventDefault(); commitRunList(); }
    return;
  }
  if (e.key === 'Escape') {
    if (runPending !== null) { e.preventDefault(); runPending = null; syncRunList(); }
    return;
  }
  if (e.altKey || e.ctrlKey || e.metaKey) return;
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
// focus has fallen out of the page (sheet.js's focusSheet).
function showSheetScope(search, { fresh = false, notice = null, scroll = true, asker = document.activeElement } = {}) {
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

// ------------------------------------------------------------- the viewer

// The most full-resolution images the strip keeps in the DOM. A 1024-cell
// night at 4 MP is ~5 GB of PNG; the strip is for watching images land, and
// the lightbox still walks every record of the run through `viewerRecords`.
const VIEWER_MAX = 48;
// This page's earlier results beside a single image (#recent): how many it
// shows, and how many it keeps for the lightbox to walk on from them.
const RECENT_SHOWN = 6;
const SESSION_MAX = VIEWER_MAX;

// The job the strip is laid out for: how many images it makes and at what
// size, from its status frame as its first frame or image lands - so the
// tier is the whole job's from the start, and the live preview takes the box
// the result will.
const viewerJob = { expected: 0, width: 0, height: 0 };

// A job's first frame or image clears the strip, which until then kept the
// last job's up through the queue wait and the load. What it held becomes
// the newest of this page's earlier results (#recent) - unless it is this
// job's own, which a stream that reattached shows again. `job` is the job's
// last status frame.
function clearResults(job = null) {
  const earlier = viewerRecords.filter((r) => !job || r.job_id !== job.job_id).reverse();
  const kept = new Set(earlier.map((r) => r.id));
  sessionRecords = [...earlier, ...sessionRecords.filter((r) => !kept.has(r.id))].slice(0, SESSION_MAX);
  viewerRecords = [];
  Object.assign(viewerJob, { expected: job?.planned_images ?? 0, width: job?.width ?? 0, height: job?.height ?? 0 });
  // Focus on what goes - a result's tool the Refine dialog handed it back
  // to, its picture - goes to Generate rather than to <body>, and a sheet the
  // job opens takes it on from there (handJobFocus).
  const focused = els.viewer.contains(document.activeElement);
  if (focused) handJobFocus(els.generate);
  els.viewer.innerHTML = '';
  els.viewer.scrollTop = 0;
  els.viewer.style.maxHeight = '';
  els.viewer.style.minHeight = '';
  viewerTier();
  els.viewerNote.replaceChildren();
  // The gallery's cards of the run before are not what the strip shows now.
  els.gallery.querySelectorAll('.card.this-run').forEach((div) => div.classList.remove('this-run'));
}

// Three layouts by how many images the job makes, so a long sweep does not
// push the gallery a screen further down with every image it finishes: one
// at 68vh, with its column beside it from 1800 px; two to four as large as
// that height allows (`fitViewer`); more, a strip of small tiles. From the
// job's count, not the images so far: by the running count the strip was
// laid out three times in a sweep's first five images, and the gallery
// under it jumped each time. #recent goes with the single tier, and before
// the fit, which measures the width it leaves.
function viewerTier() {
  const n = Math.max(viewerJob.expected, viewerRecords.length);
  els.viewer.classList.toggle('multi', n > 1);
  const many = els.viewer.classList.toggle('many', n > 4);
  // A long run's tiles are one tab stop each, their pictures, as before the
  // tools came: four more a tile put a 48-image strip 240 presses deep. The
  // lightbox a picture opens has every action; the pointer has the row.
  els.viewer.querySelectorAll('.tools button').forEach((b) => { b.tabIndex = many ? -1 : 0; });
  renderRecent();
  fitViewer();
}

// Two to four images, best fit: the column count whose tile is largest
// within the strip's width and the single image's height, the rows' gaps and
// tools rows counted - two squares side by side at 68vh, four 16:9 images two
// by two. A fixed 34vh used 42% of a QHD strip for a batch of two. The tile
// is the job's shape; a strip whose images are not all of it wraps instead.
// Every row the job will fill is there from its first frame (`--rows`): a
// 2x2 held one row while the preview denoised, and grew by the second as
// the images landed, the gallery under it pushed 466 px down at QHD at the
// end of every such Generate.
let viewerWidth = 0;
function fitViewer() {
  const v = els.viewer;
  const aspect = viewerJob.width && viewerJob.height ? viewerJob.width / viewerJob.height
    : viewerRecords.length ? viewerRecords[0].width / viewerRecords[0].height : 0;
  const alike = viewerRecords.every((r) => Math.abs(r.width / r.height - aspect) / aspect < 0.01);
  const fits = v.classList.contains('multi') && !v.classList.contains('many') && aspect > 0 && alike;
  v.classList.toggle('fitted', fits);
  if (!fits) {
    v.style.removeProperty('--cols');
    v.style.removeProperty('--rows');
    v.style.removeProperty('--tile-h');
    return;
  }
  const css = getComputedStyle(v);
  const width = v.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight);
  viewerWidth = v.clientWidth;
  const gap = parseFloat(css.columnGap) || 0;
  // A tile's tools row, under its picture (style.css).
  const foot = parseFloat(css.getPropertyValue('--h-tool')) + parseFloat(css.getPropertyValue('--sp-1'));
  const height = innerHeight * 0.68 + foot;
  const n = Math.min(4, Math.max(viewerJob.expected, viewerRecords.length));
  let best = null;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const w = Math.min((width - (cols - 1) * gap) / cols, ((height - (rows - 1) * gap) / rows - foot) * aspect);
    if (!best || w > best.w) best = { cols, w, rows };
  }
  v.style.setProperty('--cols', best.cols);
  v.style.setProperty('--rows', best.rows);
  v.style.setProperty('--tile-h', `${Math.max(48, Math.floor(best.w / aspect))}px`);
}

// The strip's ResizeObserver: its width moved - the window, the panel, the
// prompt check docked beside it - and the best fit with it, on the next
// frame: refitted inside the observer's own callback, the strip's new height
// was a resize the same frame could not deliver. #recent beside a docked
// check likewise (fitRecent); its own coming leaves the image as it was, so
// the refit it causes ends there.
function viewerResized() {
  if (!els.recent.hidden) requestAnimationFrame(fitRecent);
  // A long run's reserve, at the column count the new width has.
  if (els.viewer.style.minHeight && els.viewer.clientWidth !== reservedWidth) {
    requestAnimationFrame(() => { els.viewer.style.minHeight = ''; reserveStrip(); });
  }
  if (els.viewer.clientWidth === viewerWidth || !els.viewer.classList.contains('fitted')) return;
  requestAnimationFrame(fitViewer);
}

// A tile of the strip, at its picture's shape and its tier's size: a result,
// the live preview, a pass's stand-in. All three are the one box (style.css)
// - the picture, and under it the tools row or a line of the same height -
// so whichever replaces another leaves the gallery where it was: the
// preview was a fixed 320 px, and the strip went from a 917 px result down
// to it and back up at every Generate.
function viewerTile(className, width, height) {
  const figure = document.createElement('figure');
  figure.className = className;
  figure.style.setProperty('--w', width || 1);
  figure.style.setProperty('--h', height || 1);
  const tile = document.createElement('div');
  tile.className = 'tile';
  figure.append(tile);
  return { figure, tile };
}

// The live preview's tile: its frames in place of the picture, the step
// where the tools will be. Never a record; the result replaces it.
function previewTile(width, height) {
  const { figure, tile } = viewerTile('live', width, height);
  const img = document.createElement('img');
  img.className = 'preview';
  img.alt = 'Live preview of the image being generated';
  const foot = document.createElement('p');
  foot.className = 'foot';
  tile.append(img, foot);
  return figure;
}

// The many tier scrolls inside itself from 2000 px (style.css): a strip at
// its end stays there as tiles come, one scrolled back up stays where it is.
// Asked at every tile, the live preview's and a pass's stand-in's too
// (`stripAppend`, generation.js): the preview's tile starts a row of its own
// a few steps before its image, and a strip asked only as images landed was
// by then a row short of its end - and so not at it, for the rest of the run.
function stripAtEnd() {
  const strip = els.viewer;
  return strip.scrollTop + strip.clientHeight >= strip.scrollHeight - 2;
}

function stripAppend(el) {
  const atEnd = stripAtEnd();
  els.viewer.appendChild(el);
  settleStrip(atEnd);
  // A docked check's #recent, beside the box the image will take.
  fitRecent();
}

// After the strip's tiles changed: back to its end if it was there. And a
// strip that holds every row but for its own bottom padding shows them all
// rather than scrolling: 28 images at QHD came to 3 px past the 560 px cap,
// and the strip drew a scrollbar for those 3 px while its note said to scroll
// it. Its scrollbar's room is kept whether it scrolls or not (style.css), so
// a scrollbar coming or going does not change how many tiles a row holds.
function settleStrip(atEnd) {
  const strip = els.viewer;
  strip.style.maxHeight = '';
  reserveStrip();
  if (getComputedStyle(strip).overflowY !== 'visible') {
    const over = strip.scrollHeight - strip.clientHeight;
    if (over > 0 && over <= parseFloat(getComputedStyle(strip).paddingBottom)) {
      strip.style.maxHeight = `${strip.offsetHeight + over}px`;
    }
  }
  if (atEnd) strip.scrollTop = strip.scrollHeight;
}

// A long run's strip from 2000 px, where it has a cap of its own: every row
// the job will fill, from its first tile (`min-height`, measured on that
// tile, at the column count the strip has), up to the cap - so the gallery
// under it does not move down a row at a time as the images land. The tier
// and the tile size were the job's from its first frame; now the height is
// too. Read before settleStrip's own max-height, and let go by clearResults.
let reservedWidth = 0;
function reserveStrip() {
  const strip = els.viewer;
  if (strip.style.minHeight || !strip.classList.contains('many') || viewerJob.expected <= 1) return;
  const css = getComputedStyle(strip);
  if (css.display !== 'grid' || css.overflowY === 'visible') return;
  const first = strip.querySelector(':scope > figure');
  if (!first) return;
  const cols = css.gridTemplateColumns.split(' ').filter(Boolean).length || 1;
  const rows = Math.ceil(Math.min(viewerJob.expected, VIEWER_MAX) / cols);
  const edge = (side) => parseFloat(css[`padding${side}`]) + parseFloat(css[`border${side}Width`]);
  const tall = rows * first.getBoundingClientRect().height + (rows - 1) * (parseFloat(css.rowGap) || 0)
    + edge('Top') + edge('Bottom');
  const cap = parseFloat(css.maxHeight);
  // Past the cap by no more than the bottom padding, the strip is let grow
  // by that rather than scroll (settleStrip), so that much is reserved too.
  const height = !(cap > 0) || tall - cap <= parseFloat(css.paddingBottom) ? tall : cap;
  // Never under the least the stylesheet gives it: beside a docked check
  // that is the check's room (style.css), which the inline value overrides.
  const least = parseFloat(css.minHeight) || 0;
  strip.style.minHeight = `${Math.ceil(Math.max(height, least))}px`;
  reservedWidth = strip.clientWidth;
}

function appendResult(rec) {
  const strip = els.viewer;
  const atEnd = stripAtEnd();
  viewerRecords.push(rec);
  viewerTier();
  const figure = resultFigure(rec);
  strip.appendChild(figure);
  // The column beside the image only while it is the strip's one.
  strip.querySelectorAll('.result-side').forEach((side) => side.remove());
  if (viewerRecords.length === 1 && viewerJob.expected <= 1) figure.append(resultSide(rec, figure));
  // Lazy on its own is not enough - the strip sets no height, so unloaded
  // images stack inside the viewport and all become eligible at once. The cap
  // is what bounds the fetches; the note lives outside the strip so it survives.
  // Images only: a deleted tile's tombstone holds a slot, and shows none.
  const tiles = () => strip.querySelectorAll('img[data-id]').length;
  while (tiles() > VIEWER_MAX) strip.firstElementChild.remove();
  settleStrip(atEnd);
  // A docked check's #recent, measured beside the image now in the strip.
  fitRecent();
  noteViewer();
  // The run's cards in the gallery say so (`this-run`, set in card() too),
  // so the strip and the gallery's first rows read as the same images.
  els.gallery.querySelector(`.card[data-id="${rec.id}"]`)?.classList.add('this-run');
}

// One result: its picture - thumb's, which opens the lightbox - and under
// it the gallery card's own tools, by the card's own helpers (gallery.js):
// Reuse, Refine, Star and Delete. The image just made had none: starring or
// refining it took the lightbox, or the same image found among the cards. A
// figure, never a .card - the gallery's counts and keyboard are the cards' -
// tagged by id, so setFavorite keeps its stars in step and forgetImages
// leaves a tombstone in its place.
function resultFigure(rec) {
  const { figure, tile } = viewerTile('result', rec.width, rec.height);
  figure.dataset.id = rec.id;
  // Named "#N" as a card is (card()), so its Star, Refine and the rest are
  // heard as this image's.
  figure.setAttribute('role', 'group');
  figure.setAttribute('aria-label', `#${rec.id}`);
  figure.classList.toggle('fav', !!rec.favorite);
  const img = thumb(rec, viewerRecords, rec.url, () => openViewerImage(rec));
  img.onerror = () => { void forgetIfGone(rec.id); };
  const star = tool('☆', '', () => setFavorite(rec, !rec.favorite), 'Star');
  star.classList.add('star');
  syncStar(star, !!rec.favorite);
  const refine = tool('⤢', '', () => refineResult(rec, tile));
  refine.classList.add('refine-tool');
  syncRefineTool(refine);
  // A card's ❐, where the checkpoint takes references (syncCardTools keeps
  // it to the status): on klein a reference had to come from a card.
  const reference = tool('❐', '', () => toggleReference(tile, rec));
  reference.classList.add('ref-tool');
  syncReferenceTool(reference, rec.id);
  reference.hidden = !referenceSupported;
  const del = tool('⌦', `Delete #${rec.id}`, () => deleteImage(rec), 'Delete');
  del.classList.add('del');
  const tools = document.createElement('div');
  tools.className = 'tools';
  tools.append(reuseButton(rec, true), refine, reference, star, del);
  // Out of the tab order in a long run's strip (viewerTier).
  if (els.viewer.classList.contains('many')) tools.querySelectorAll('button').forEach((b) => { b.tabIndex = -1; });
  tile.append(img, tools);
  return figure;
}

// The card's ⤢ on a result: on a checkpoint without a pass it says so under
// the gallery's head and over the tile (`refuseInCard`), and opens nothing.
function refineResult(rec, tile) {
  if (!refineSupported) { refuseInCard(tile, REFINE_NONE, 'No refine pass here'); return; }
  openRefine(rec);
}

// The image just made, while the strip holds it alone, with what it is and
// what to do next in the width the picture leaves - shown from 1800 px
// (style.css), where a square result left 1,270 px of the strip empty: which
// image it refines, Star, Refine…, Reuse, a pass's Compare with source and
// Use seed in words, the prompt, and the lightbox's own facts (factsList,
// without its Use seed: the row has one). In the figure, so a star and a
// delete reach it as they reach the tile; Delete stays the tile's.
function resultSide(rec, figure) {
  const side = document.createElement('div');
  side.className = 'result-side';
  if (rec.refine != null) side.append(passLineage(rec));
  const action = (button) => { button.classList.add('ghost', 'small'); return button; };
  // Its word drawn after the glyph (`data-word`): syncStar writes the glyph.
  const star = action(tool('☆', '', () => setFavorite(rec, !rec.favorite)));
  star.classList.add('star');
  star.dataset.word = 'Star';
  syncStar(star, !!rec.favorite);
  const refine = action(tool('⤢ Refine…', '', () => refineResult(rec, figure.querySelector('.tile'))));
  refine.classList.add('refine-tool');
  syncRefineTool(refine);
  const reference = action(tool('❐ Reference', '', () => toggleReference(figure.querySelector('.tile'), rec)));
  reference.classList.add('ref-tool');
  syncReferenceTool(reference, rec.id);
  reference.hidden = !referenceSupported;
  // Named by its words, as the finish line's is; the title has the seed.
  const seed = document.createElement('button');
  seed.type = 'button';
  seed.className = 'ghost small';
  seed.textContent = 'Use seed';
  seed.title = `Put ${rec.seed} in the Seed box`;
  seed.onclick = () => useSeed(rec.seed);
  // What the facts say, in a line, where the row stands in for the column
  // under a docked check (style.css); the settings line whole in its title.
  const brief = document.createElement('span');
  brief.className = 'result-brief';
  brief.textContent = `seed ${rec.seed} · ${rec.width}×${rec.height} · ${rec.steps} steps · CFG ${rec.guidance_scale}`;
  brief.title = settingsLine(rec);
  // Delete in words at the row's end, red under the pointer, as the page's
  // other destructive buttons: a lone ⌦ under the picture's corner was the
  // one tool the column did not have.
  const del = action(tool('Delete', `Delete #${rec.id}`, () => deleteImage(rec)));
  del.classList.add('del', 'danger');
  const actions = document.createElement('div');
  actions.className = 'result-actions';
  actions.append(star, refine, action(reuseButton(rec)), reference);
  if (rec.refine != null) actions.append(compareButton(rec, figure));
  actions.append(seed, brief, del);
  const prompt = document.createElement('p');
  prompt.className = 'result-prompt';
  prompt.textContent = rec.prompt;
  side.append(actions, prompt, factsList(rec));
  return side;
}

// A pass's Compare with source: the lightbox on the pass, as its picture
// opens it (openViewerImage, so a close comes back here), with the
// comparison already on - the question a pass that has just landed asks, a
// press and an S away before. Dead where the source has gone, saying why
// (`syncCompareResult`), as the lightbox's own is. The column's alone: the
// row under the picture that stands in for it with the check docked leaves
// it out (style.css), as it cut a portrait pass's row to ellipses.
function compareButton(rec, figure) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost small compare-source';
  button.textContent = 'Compare with source';
  button.onclick = () => {
    if (button.getAttribute('aria-disabled') === 'true') {
      refuseInCard(figure.querySelector('.tile'), `${button.title}.`, 'Source deleted');
      return;
    }
    openViewerImage(rec);
    void toggleCompare();
  };
  syncCompareResult(button, rec);
  return button;
}

function syncCompareResult(button, rec) {
  button.setAttribute('aria-disabled', String(!!rec.refine_missing));
  button.title = rec.refine_missing ? `#${rec.refine}, the image this was refined from, has been deleted`
    : `Compare this pass with #${rec.refine}, the image it was refined from, in the lightbox`;
}

// The column's first line for a pass: which image it refines, and how far.
function passLineage(rec) {
  const line = document.createElement('p');
  line.className = 'lineage';
  line.append('⤢ Refine pass of ', sourceLink(rec), ` at strength ${rec.strength}`);
  return line;
}

// A pass's source as the page names it: a press opens it in the lightbox
// with the pass after it (lightbox.js's openRefineSource), unless it has gone.
function sourceLink(rec) {
  if (rec.refine_missing) return `#${rec.refine} (deleted)`;
  return factButton(`#${rec.refine}`, `Open #${rec.refine}, the image this was refined from; → comes back to the pass`,
    () => { void openRefineSource(rec); });
}

// A pass's source deleted while the pass is in the strip (forgetImages, once
// its records say `refine_missing`): the note and the column's lineage and
// facts say "#N (deleted)" rather than keep a button whose press said only,
// in the panel's line a screen away, that #N had gone.
function viewerSourcesGone() {
  noteViewer();
  els.viewer.querySelectorAll('.result > .result-side').forEach((side) => {
    const rec = viewerRecords.find((r) => r.id === +side.parentElement.dataset.id);
    if (!rec?.refine_missing) return;
    side.querySelector('.lineage')?.replaceWith(passLineage(rec));
    side.querySelector('.facts')?.replaceWith(factsList(rec));
    const compare = side.querySelector('.compare-source');
    if (compare) syncCompareResult(compare, rec);
  });
}

// Under the strip: where a pass came from - "Refined from #10 at 0.5 ·
// 1344×1728", the source a press away (an empty note had said nothing of
// what the image was) - and how much of a long run the strip does not show.
// The image just made, alone and made from the prompt, is named as a card's
// caption names it, its seed with Use seed (`seed-line`): under 1800 px,
// where nothing beside the image said its seed but its title; from there
// the column beside it says it (style.css hides the line).
function noteViewer() {
  const parts = [];
  const first = viewerRecords[0];
  if (first?.refine != null && viewerRecords.every((r) => r.refine === first.refine)) {
    const strengths = [...new Set(viewerRecords.map((r) => r.strength))].join(', ');
    const line = document.createElement('span');
    line.className = 'lineage';
    line.append('Refined from ', sourceLink(first), ` at ${strengths} · ${first.width}×${first.height}`);
    parts.push(line);
  } else if (first && viewerRecords.length === 1 && viewerJob.expected <= 1) {
    const line = document.createElement('span');
    line.className = 'seed-line';
    line.append(`#${first.id} · ${first.width}×${first.height} · seed ${first.seed} · `,
      factButton('Use seed', `Put ${first.seed} in the Seed box`, () => useSeed(first.seed)));
    parts.push(line);
  }
  const shown = els.viewer.querySelectorAll('img[data-id]').length;
  const hidden = viewerRecords.length - shown;
  const scrolls = getComputedStyle(els.viewer).overflowY !== 'visible'
    && els.viewer.scrollHeight > els.viewer.clientHeight + 1;
  if (hidden > 0) {
    parts.push(`… and ${hidden} earlier image${hidden === 1 ? '' : 's'} this run — open one to walk them all`);
  } else if (scrolls) {
    parts.push(`${viewerRecords.length} images this run — scroll the strip, or open one to walk them all`);
  }
  els.viewerNote.replaceChildren(...parts.flatMap((part, i) => (i ? [' · ', part] : [part])));
}

// The strip's image opens the lightbox on the strip - and, where it is the
// only one, on the gallery from it, so → walks back through what came before
// it: alone, it opened a lightbox of one, with both arrows dead. Not in the
// gallery (a search, a sheet), it walks this page's earlier results.
function openViewerImage(rec) {
  if (viewerRecords.length === 1 && viewerRecords[0] === rec) {
    const copy = !sheetMode && galleryRecords.find((r) => r.id === rec.id);
    if (copy) { openLightbox(copy, galleryRecords, 'viewer'); return; }
    if (sessionRecords.length) { openLightbox(rec, [rec, ...sessionRecords], 'viewer'); return; }
  }
  openLightbox(rec, viewerRecords, 'viewer');
}

// #recent: beside a single image from 2000 px (style.css), up to six of this
// page's earlier results, each opening the lightbox on the image in the strip
// and those, newest first. Iterating on a prompt showed one picture at a time
// in a strip 2,000 px wide, the earlier ones a screen down among the cards -
// below the fold whenever the prompt check was open. Rebuilt only when what
// it shows changes, so a focused thumbnail keeps its focus.
let recentShown = null;
function renderRecent() {
  const single = !els.viewer.classList.contains('multi');
  const shown = single ? sessionRecords.slice(0, RECENT_SHOWN) : [];
  const current = single && viewerRecords.length === 1 ? viewerRecords[0] : null;
  els.recent.hidden = !shown.length;
  const key = [current?.id, ...shown.map((r) => r.id)].join();
  if (key !== recentShown) {
    recentShown = key;
    const walk = current ? [current, ...sessionRecords] : sessionRecords.slice();
    els.recentList.replaceChildren(...shown.map((rec) => thumb(rec, walk, undefined, () => openLightbox(rec, walk, 'recent'))));
  }
  fitRecent();
}

// With the prompt check docked (style.css), #recent stays beside the image
// in the width the image leaves at its own size - so neither the image nor
// the gallery under it moves as the check opens - and goes where that is
// under --recent-min. Measured with #recent out of the docked grid, the
// viewer then the whole of its column. Again whenever the strip's width
// moves (viewerResized): the check opening or closing, the window.
function fitRecent() {
  const stage = els.recent.parentElement;
  stage.classList.remove('recent-docked');
  // Docked in the stage, which it is not with the gallery beside the viewer:
  // the check takes the gallery's column there (style.css, `:not(.beside)`).
  const docked = matchMedia('(min-width: 2000px)').matches && !els.promptAnalysis.hidden
    && els.promptAnalysis.parentElement === els.tagValidationWide && !besideOn;
  // The image, or the preview in its box.
  const tile = els.viewer.querySelector(':scope > figure > .tile');
  if (els.recent.hidden || !docked || !tile) return;
  const css = getComputedStyle(els.viewer);
  const room = els.viewer.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight)
    - tile.getBoundingClientRect().width - parseFloat(getComputedStyle(stage).columnGap);
  if (room < parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--recent-min'))) return;
  stage.style.setProperty('--recent-room', `${Math.floor(room)}px`);
  stage.classList.add('recent-docked');
}

// While a job waits on the load and the strip is empty - its one idle line,
// or the line a delete left - the line says so, with the clock the progress
// row ticks (generation.js's ticker): after Generate the eye is on the
// viewer, where "Nothing generated this session" stayed for the minutes a
// first load takes. The clock is aria-hidden, as the progress row's is,
// which says "loading weights" once; the line stays one line (style.css).
// `clock` null puts the line's own words back: the wait over, cancelled or
// failed, or the stream lost. The first frame or image clears the strip, the
// line with it. Not over a line holding the focus (a first run's Examples).
let viewerWait = null;
function viewerWaiting(clock) {
  const line = els.viewer.querySelector(':scope > .placeholder');
  if (viewerWait && (clock == null || viewerWait.line !== line)) {
    if (viewerWait.line.isConnected) {
      viewerWait.line.replaceChildren(...viewerWait.words);
      viewerWait.line.classList.remove('waiting');
    }
    viewerWait = null;
  }
  if (clock == null || !line) return;
  if (!viewerWait) {
    if (line.contains(document.activeElement)) return;
    viewerWait = { line, words: [...line.childNodes] };
    const time = document.createElement('span');
    time.className = 'clock';
    time.setAttribute('aria-hidden', 'true');
    line.replaceChildren('Loading weights', time, ' - the first load of a session takes a few minutes.');
    line.classList.add('waiting');
  }
  line.querySelector('.clock').textContent = clock;
}

// A gallery with nothing in it at all - no image on any checkpoint, nothing
// filtered out - is a first run: the empty viewer stays the tall box, with
// the authors' examples a press away (`viewerExamples`), rather than the one
// line every later visit gets. From each unfiltered gallery page.
function syncEmptyInstall(page) {
  if (els.search.value.trim() || els.favOnly.checked) return;
  els.viewer.classList.toggle('empty-install', page.total === 0 && !page.other_models);
}

// The first run's Examples: the panel's own menu, open on its first item.
// After the click is through, whose document handler closes an open menu.
function viewerExamples() {
  requestAnimationFrame(() => {
    showExamples(true);
    els.examplesMenu.querySelector('[role="menuitem"]')?.focus();
  });
}

// One <img> for every place a record is shown: the viewer strip, a gallery
// card, a sheet cell, #recent. Lazy and async-decoded, since all of them can
// hold hundreds; a thumbnail derivative where one exists, the full PNG
// otherwise. Focusable and activated by Enter or Space, so the lightbox -
// where the full image and its metadata are shown - is reachable without a
// mouse; `open` is what a press does, the lightbox on `list` unless said.
function thumb(rec, list, src = rec.thumb_url ?? rec.url, open = () => openLightbox(rec, list)) {
  const img = document.createElement('img');
  img.src = src;
  img.alt = rec.prompt;
  img.loading = 'lazy';
  img.decoding = 'async';
  img.dataset.id = rec.id;
  img.tabIndex = 0;
  img.setAttribute('role', 'button');
  // Named as the page names images, by gallery id - the cells of a sweep
  // share a prompt, and "Open: …" named none of them apart - and titled with
  // what tells it from the one beside it under the pointer (cardTitle): its
  // size, its seed and a pass's source, then the prompt. A sheet's cell is
  // named by what the sheet lays out instead (renderSheet).
  img.setAttribute('aria-label', `Open #${rec.id}: ${rec.prompt}`);
  titleImage(img, rec);
  img.onclick = open;
  // A plain Enter or Space only: Ctrl+Enter is Generate's from anywhere
  // (app.js), and on a focused image - where closing the lightbox leaves the
  // focus - it opened the lightbox and queued a job behind it.
  img.onkeydown = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
  };
  return img;
}

// An image's title, cardTitle's facts and then its prompt, for the pointer,
// and its description, the facts alone: the title was the description too,
// so a long prompt was read twice an image, once in the name ("Open #N:
// <prompt>") and again after "#N · W×H · seed N". `aria-description` takes
// the title's place there, less any fact the name says already: a sheet's
// cell is named by its settings ("Open #16 · Steps 2 · Seed 7", renderSheet,
// which calls this again once it has named it), and its id and seed were
// read twice. Again where a deleted source changes the facts (forgetImages).
function titleImage(img, rec) {
  const title = cardTitle(rec);
  img.title = title;
  const named = new Set((img.getAttribute('aria-label') ?? '').replace(/^Open /, '').toLowerCase().split(' · '));
  const facts = title.split('\n')[0].split(' · ').filter((fact) => !named.has(fact.toLowerCase()));
  img.setAttribute('aria-description', facts.join(' · '));
}
