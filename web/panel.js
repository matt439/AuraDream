// AuraDream's page script: panel persistence, Reset, sticky placement and tips.
// Declarations only; state.js has the shared-global load-order contract.
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
const PANEL_KEY = 'auradream.panel.v1';
const LEGACY_PANEL_KEYS = ['krea.panel.v1'];
const PANEL_FIELDS = [
  'prompt', 'negative', 'width', 'height', 'seed', 'batch', 'runs', 'lora', 'loraScale',
  'gridPrompt', 'gridPromptMode', 'gridSteps', 'gridCfg', 'gridMu', 'gridSeeds', 'gridSeedList', 'gridBudget',
  'gridLoraScale', 'gridSize', 'stepCache',
];
// What else is remembered beside the fields, merged key by key as they are.
const PANEL_KEYS = [...PANEL_FIELDS, 'gridOpen', 'reference', 'translation'];
let panelTimer;
const SAMPLING_AXES = ['gridSteps', 'gridCfg', 'gridMu', 'gridLoraScale'];
let samplingByModel = {};

function currentSampling(model = loadedModel) {
  return {
    model,
    ...Object.fromEntries(Object.keys(owned).map((key) => [key, owned[key] ? els[key].value : null])),
    ...Object.fromEntries(SAMPLING_AXES.map((key) => [key, els[key].value])),
  };
}

function keepSampling(model = loadedModel) {
  if (model) samplingByModel[model] = currentSampling(model);
}

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
  state.stepCache = stepCacheValue();
  state.lora = els.lora.value || pendingLora;
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
  state.samplingByModel = { ...samplingByModel };
  if (loadedModel) state.samplingByModel[loadedModel] = currentSampling();
  return state;
}

function persistPanel() {
  // A change since a Reuse or a Reset retires its Undo (reuse.js and below).
  syncReuseUndo();
  syncResetUndo();
  syncSetupUndo();
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
    out.samplingByModel = { ...(stored.samplingByModel ?? {}) };
    for (const [model, sampling] of Object.entries(now.samplingByModel)) {
      const before = panelBase.samplingByModel?.[model];
      const values = { ...(out.samplingByModel[model] ?? {}) };
      for (const [key, value] of Object.entries(sampling)) {
        if (!before || !same(value, before[key])) values[key] = value;
      }
      out.samplingByModel[model] = values;
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
  samplingByModel = state?.samplingByModel && typeof state.samplingByModel === 'object'
    ? { ...state.samplingByModel } : {};
  if (state?.sampling?.model && !samplingByModel[state.sampling.model]) {
    samplingByModel[state.sampling.model] = state.sampling;
  }
  // What a later write compares this page against: the store as read - the
  // sampling half is applied once the checkpoint is known (restoreSampling),
  // and the panel's own values before then would make it look changed - or
  // the page as it opens, where nothing was stored.
  panelBase = state ?? panelState();
  if (!state) return;
  for (const key of PANEL_FIELDS) {
    if (key !== 'stepCache' && typeof state[key] === 'string') els[key].value = state[key];
  }
  setStepCache(Object.hasOwn(state, 'stepCache') ? state.stepCache : '');
  // Older panels have no mode; a malformed stored select must not send an
  // empty mode and turn every preview into a schema error.
  if (!['cross', 'random'].includes(els.gridPromptMode.value)) els.gridPromptMode.value = 'cross';
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
  revealActiveSetup();
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
  forgetInpaint();
  forgetOutpaint();
  samplingByModel = {};
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
  setStepCache('');
  referenceIds = [];
  els.lora.value = '';
  pendingLora = '';
  els.loraScale.value = 1;
  els.seed.value = '';
  els.batch.value = 1;
  els.runs.value = 1;
  for (const key of ['gridPrompt', 'gridSteps', 'gridCfg', 'gridMu', 'gridSeedList', 'gridBudget',
                     'gridLoraScale', 'gridSize']) {
    els[key].value = '';
  }
  els.gridSeeds.value = 1;
  els.gridPromptMode.value = 'cross';
  els.gridSeeds.disabled = false;
  els.gridResume.checked = false;
  els.gridSection.open = false;
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  syncExecutionControls();
  syncSubmitButtons();
  schedulePreview();
  scheduleTagValidation(0);
  // Not persisted, but a change all the same: a Reuse's Undo goes, as an
  // Example's does (scheduleTagValidation) - and its words with it, which
  // describe the panel just emptied.
  clearReuseNote();
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

// The note a Reset leaves, with its Undo, where Reset was pressed: in the top
// right of the prompt box it emptied, under Reset (tags.js, showPromptNote,
// `at: 'box'`). In the panel's line under Generate it was below the window at
// 1400x900 and in one column, and gone after 20 seconds unseen; in the
// panel's flow under the prompt it moved Generate down a line, and back up
// under the pointer when it went. On offer while the panel is as the Reset
// left it - the first change takes it, as a Reuse's Undo goes - and for 20
// seconds at most.
let resetNote = null;
const RESET_UNDO_MS = 20000;
const RESET_UNDONE_MS = 6000;

function offerResetUndo(before) {
  const note = { before, after: JSON.stringify(panelState()), entry: null, undone: false };
  note.entry = offerPromptUndo('Panel reset.', () => undoReset(note), {
    kind: 'reset',
    at: 'box',
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
  samplingByModel = { ...(state.samplingByModel ?? {}) };
  rewriteBox(els.negative, state.negative);
  rewriteBox(els.prompt, state.prompt);
  for (const key of PANEL_FIELDS) {
    if (!['prompt', 'negative', 'stepCache'].includes(key)) els[key].value = state[key];
  }
  setStepCache(Object.hasOwn(state, 'stepCache') ? state.stepCache : '');
  pendingLora = state.lora && !els.lora.value ? state.lora : '';
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
  revealActiveSetup();
  syncRuns();
  syncExecutionControls();
  syncSubmitButtons();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  // Where the Undo was pressed, over the box, a few seconds: nothing moves as
  // it comes and goes, and typing takes it at once (an 'info' note).
  showPromptNote('Reset undone: the panel is as it was.', { kind: 'info', at: 'box', ms: RESET_UNDONE_MS });
}

// The second phase, from refreshStatus once the checkpoint is known.
let samplingRestored = null;
function restoreSampling(modelId) {
  if (samplingRestored === modelId) return;
  samplingRestored = modelId;
  const sampling = samplingByModel[modelId] ?? storedPanel()?.sampling;
  if (!sampling || sampling.model !== modelId) return;
  for (const [key, el] of [['steps', els.steps], ['cfg', els.cfg], ['mu', els.mu], ['trueCfg', els.trueCfg]]) {
    if (typeof sampling[key] === 'string') { el.value = sampling[key]; own(key); }
  }
  for (const key of SAMPLING_AXES) {
    if (typeof sampling[key] === 'string') els[key].value = sampling[key];
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
//     down the gallery the panel goes up until its bottom - the LoRA row, under
//     References and the grid fold, with Generate, its progress and Run grid
//     above them - is in view and holds there, and back up
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
// the window's bottom and over has more room (and where neither holds it,
// under and risen into the window): the grid fold's last rows sit
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
  for (const side of ['left', 'top', 'right', 'bottom']) tip.style.removeProperty(side);
  // A badge inside a box that scrolls by itself - the prompt check's, above
  // the results and docked beside them - hangs its tip from the window
  // instead (`floatTip`): placed inside, the comparison's Batch tip was cut
  // off by the check's edge and lay under its sticky head.
  const clip = tipClipper(help);
  tip.classList.toggle('floating', !!clip);
  if (clip) { floatTip(help, tip); return; }
  floatingHelp = null;
  tip.style.display = 'block';
  const badge = help.getBoundingClientRect();
  const shown = tip.getBoundingClientRect();
  tip.style.display = '';
  const topbar = document.querySelector('.topbar').getBoundingClientRect().bottom;
  const above = badge.top - (shown.top - badge.bottom) - topbar;
  const up = shown.bottom > innerHeight && above > innerHeight - shown.top;
  if (up) help.classList.add('up');
  // Too tall for either side of a badge mid-window (the Search tip at 600x900,
  // once it listed lora: and quant:): it stays under, and rises by what it
  // would run past the window's foot, never above the topbar. It may then lie
  // over its own badge, which keeps it open: the tip is inside the .help.
  const rise = up ? 0 : Math.max(0, Math.min(shown.bottom - (innerHeight - TIP_EDGE), shown.top - topbar - TIP_EDGE));
  const right = document.documentElement.clientWidth - TIP_EDGE;
  const past = shown.right > right ? right - shown.right : 0;
  const shift = shown.left + past < TIP_EDGE ? TIP_EDGE - shown.left : past;
  // Rounded away from the edge it clears: rounded to nearest, a 17.4 px shift
  // left a tip 7.6 px from the window's left edge.
  const x = Math.abs(shift) >= 1 ? (shift > 0 ? Math.ceil(shift) : Math.floor(shift)) : 0;
  if (x || rise >= 1) tip.style.translate = `${x}px ${-Math.ceil(rise)}px`;
}

// The box round a badge that clips what overflows it, short of the page.
function tipClipper(help) {
  for (let up = help.parentElement; up && up !== document.body; up = up.parentElement) {
    const style = getComputedStyle(up);
    if (style.display === 'contents') continue;
    if (style.overflowX !== 'visible' || style.overflowY !== 'visible') return up;
  }
  return null;
}

// A tip fixed to the window by its badge (`.tip.floating`, style.css): under
// the badge where it fits, over it where there is more room there - neither
// past the topbar nor the window's foot - and moved in from either side as
// placeTip moves one. Kept with its badge while that box scrolls (app.js's
// scroll listener calls this again), and hidden (`.help.clipped`) while the
// badge itself is scrolled out of the box's view.
let floatingHelp = null;
function floatTip(help, tip = help.querySelector('.tip')) {
  floatingHelp = help;
  tip.style.display = 'block';
  const badge = help.getBoundingClientRect();
  const { width, height } = tip.getBoundingClientRect();
  tip.style.display = '';
  const view = tipClipper(help)?.getBoundingClientRect();
  help.classList.toggle('clipped', !!view && (badge.bottom <= view.top || badge.top >= view.bottom
    || badge.right <= view.left || badge.left >= view.right));
  const topbar = document.querySelector('.topbar').getBoundingClientRect().bottom;
  const below = badge.bottom + 7;
  const above = badge.top - 7 - height;
  const up = below + height > innerHeight && badge.top - 7 - topbar > innerHeight - below;
  const end = help.classList.contains('end');
  const right = document.documentElement.clientWidth - TIP_EDGE;
  let left = end ? badge.right + 6 - width : badge.left - 6;
  left = Math.max(TIP_EDGE, Math.min(left, right - width));
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(up ? above : below)}px`;
  tip.style.right = 'auto';
  tip.style.bottom = 'auto';
}

// The open floating tip follows its badge as anything scrolls; gone once
// neither the pointer nor the focus holds the badge.
function followFloatingTip() {
  const help = floatingHelp;
  if (!help) return;
  if (!help.isConnected || !help.matches(':hover, :focus')) { floatingHelp = null; return; }
  floatTip(help);
}
