// The Models picker's strip of the chosen checkpoint's images, loaded right
// after model-switch.js, whose selection, labels and shared helpers it reads.
// Declarations only; app.js wires the controls after every script has loaded.
'use strict';

// ------------------------------------------- the chosen checkpoint's images
// The strip under the table, a column beside it from 1280 px wide
// (model-picker.css): the newest images the chosen row's checkpoint has made
// and their count - what its style is, answered by its own
// pictures. Which images those are is the server's (`/api/gallery?model=`,
// the gallery's exact scope); the strip asks for only as many as it shows.
// A gallery read, not model work: it answers while the server is cold,
// loading or switching. A choice settles a moment before it is asked for, so
// arrowing through the rows asks once; a newer choice aborts the older
// request and an epoch drops its reply.
const MODEL_SHOTS_MAX = 24;
const MODEL_SHOTS_WAIT = 180;
// What the strip shows: the checkpoint (null until an opening asks), its
// records and count, and its state - `pending` (asked; the last pictures
// stay, dimmed and inert), `ready`, `failed` (the reason said in the box)
// or `none` (no row chosen).
const modelShots = { id: null, records: [], total: 0, state: 'none', reason: '' };
let modelShotsEpoch = 0;
let modelShotsTimer = 0;
let modelShotsAbort = null;
let modelShotsFailedAt = 0;
// The picture that is the strip's one tab stop, by image id.
let modelShotsStop = null;

// As many pictures as the grid holds whole, up to 24: its columns times the
// rows its height takes - one under the table, as many as the picker's
// height allows in the side column (from 1280 px), never a row cut in half.
// Measured as the picker stands when the request goes, and again on a
// resize while it is open (refitModelShots).
function modelShotsFit() {
  const style = getComputedStyle(els.modelShotsList);
  const columns = style.gridTemplateColumns;
  const tracks = /repeat|none/.test(columns) ? [] : columns.trim().split(/\s+/);
  if (!tracks.length) return MODEL_SHOTS_MAX;
  // Square pictures: a row is a column's width.
  const size = parseFloat(tracks[0]);
  const gap = parseFloat(style.rowGap) || 0;
  const rows = size > 0 ? Math.floor((els.modelShotsList.clientHeight + gap) / (size + gap)) : 1;
  return Math.max(1, Math.min(MODEL_SHOTS_MAX, tracks.length * Math.max(1, rows)));
}

// See all writes the id as one search word, `model:=<id>`, the same exact
// scope; an id with whitespace (an off-catalogue path) cannot be one.
function modelShotsSearch(id) {
  return id && !/\s/.test(id) ? `model:=${id}` : '';
}

// Ask again for the chosen row: on a new choice (after a pause, the old
// pictures dimmed meanwhile), or at once and in place (`fresh`: an opening,
// a delete, a retry). A no-op while the picker is closed or the choice is
// what the strip already shows.
function syncModelShots({ fresh = false } = {}) {
  if (els.modelPicker.hidden) return;
  const id = modelChoice();
  if (!fresh && id === modelShots.id) return;
  const changed = id !== modelShots.id;
  clearTimeout(modelShotsTimer);
  modelShotsAbort?.abort();
  modelShotsAbort = null;
  const epoch = ++modelShotsEpoch;
  const previous = modelShots.id;
  modelShots.id = id;
  if (!id) {
    // Before the first list, nothing yet; after it, no row is chosen.
    Object.assign(modelShots, { records: [], total: 0, state: modelCatalogueLoaded ? 'none' : 'pending', reason: '' });
  } else if (changed || modelShots.state !== 'ready') {
    modelShots.state = 'pending';
  }
  drawModelShots();
  // A move from one row to another waits for the choice to settle.
  const wait = changed && previous && !fresh ? MODEL_SHOTS_WAIT : 0;
  if (id) modelShotsTimer = setTimeout(() => { void fetchModelShots(id, epoch); }, wait);
}

// A closed picker asks for nothing; the next opening asks afresh.
function stopModelShots() {
  clearTimeout(modelShotsTimer);
  clearTimeout(modelShotsRefitTimer);
  modelShotsAbort?.abort();
  modelShotsAbort = null;
  ++modelShotsEpoch;
  if (modelShots.state === 'pending' && modelShots.id) modelShots.id = null;
}

async function fetchModelShots(id, epoch) {
  const abort = new AbortController();
  modelShotsAbort = abort;
  const query = new URLSearchParams({ model: id, limit: String(modelShotsFit()) });
  try {
    const page = await api(`/api/gallery?${query}`, { signal: abort.signal });
    if (epoch !== modelShotsEpoch) return;
    modelShotsFailedAt = 0;
    // A delete this page made may not have reached the reply's read.
    const records = page.items.filter((rec) => !isGone(rec.id));
    Object.assign(modelShots, {
      records, total: Math.max(records.length, page.total - (page.items.length - records.length)),
      state: 'ready', reason: '',
    });
  } catch (err) {
    if (epoch !== modelShotsEpoch || abort.signal.aborted) return;
    // Quietly, in the strip's own box: a gallery busy with a bundle, a
    // server not answering. A status poll asks again (retryModelShots).
    modelShotsFailedAt = Date.now();
    Object.assign(modelShots, { records: [], total: 0, state: 'failed', reason: err.message });
  } finally {
    if (modelShotsAbort === abort) modelShotsAbort = null;
  }
  drawModelShots();
}

// A failed strip asks again while the picker is open, from the status polls,
// every few seconds at most.
function retryModelShots() {
  if (modelShots.state === 'failed' && !els.modelPicker.hidden && Date.now() - modelShotsFailedAt > 5000) {
    syncModelShots({ fresh: true });
  }
}

// A resize while the picker is open (app.js), across the column's
// breakpoint or to a shorter or taller window: the pictures follow the fit.
// Fewer drop the oldest at once - newest first, so what stays is what a
// request would answer - and leave no row cut in half or hidden in the
// arrow keys' walk; more ask again in place (the pictures shown stay
// undimmed) once the resize has rested.
let modelShotsRefitTimer = 0;
function refitModelShots() {
  clearTimeout(modelShotsRefitTimer);
  if (els.modelPicker.hidden || modelShots.state !== 'ready') return;
  const fit = modelShotsFit();
  if (modelShots.records.length > fit) {
    modelShots.records = modelShots.records.slice(0, fit);
    drawModelShots();
  } else if (modelShots.records.length < Math.min(fit, modelShots.total)) {
    const id = modelShots.id;
    modelShotsRefitTimer = setTimeout(() => {
      if (modelShots.state === 'ready' && modelShots.id === id) syncModelShots({ fresh: true });
    }, MODEL_SHOTS_WAIT);
  }
}

function modelShotButton(rec) {
  const item = document.createElement('li');
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'model-shot';
  button.dataset.image = String(rec.id);
  const prompt = (rec.prompt ?? '').trim();
  button.title = prompt ? (prompt.length > 200 ? `${prompt.slice(0, 199)}…` : prompt) : `Image #${rec.id}`;
  const img = document.createElement('img');
  img.src = rec.thumb_url || rec.url;
  img.alt = prompt || `Image #${rec.id}`;
  img.decoding = 'async';
  img.draggable = false;
  button.append(img);
  item.append(button);
  return item;
}

function modelShotButtons() {
  return [...els.modelShotsList.querySelectorAll('.model-shot')];
}

// The pictures are one tab stop, the last one focused (else the newest);
// arrow keys, Home and End walk them (moveModelShotFocus).
function syncModelShotStops() {
  const buttons = modelShotButtons();
  const stop = buttons.find((b) => b.dataset.image === String(modelShotsStop)) ?? buttons[0];
  for (const button of buttons) setModelAttribute(button, 'tabindex', button === stop ? '0' : '-1');
}

function moveModelShotFocus(event) {
  const buttons = modelShotButtons();
  const at = buttons.indexOf(document.activeElement);
  if (at < 0) return;
  const to = { ArrowRight: at + 1, ArrowDown: at + 1, ArrowLeft: at - 1, ArrowUp: at - 1, Home: 0, End: buttons.length - 1 }[event.key];
  if (to === undefined) return;
  event.preventDefault();
  const target = buttons[Math.max(0, Math.min(buttons.length - 1, to))];
  modelShotsStop = target.dataset.image;
  syncModelShotStops();
  target.focus();
}

function drawModelShots() {
  const { id, records, total, state, reason } = modelShots;
  const name = id ? modelLabel(id) : '';
  const hadFocus = els.modelShots.contains(document.activeElement);
  const label = !id ? 'Made with the chosen checkpoint'
    : `Made with ${name}${state === 'ready' && total ? ` (${total})` : ''}`;
  if (els.modelShotsLabel.textContent !== label) els.modelShotsLabel.textContent = label;
  // The pictures shown are the reply's; a pending choice keeps the last
  // ones, dimmed and out of reach, until its own arrive.
  const key = JSON.stringify(records.map((rec) => [rec.id, rec.thumb_url || rec.url, rec.prompt]));
  if (state !== 'pending' && modelOptionKeys.get(els.modelShotsList) !== key) {
    els.modelShotsList.replaceChildren(...records.map(modelShotButton));
    modelOptionKeys.set(els.modelShotsList, key);
  }
  const stale = state === 'pending';
  if (els.modelShots.classList.contains('model-shots-stale') !== stale) els.modelShots.classList.toggle('model-shots-stale', stale);
  setModelAttribute(els.modelShotsList, 'aria-busy', String(stale));
  if (els.modelShotsList.inert !== stale) els.modelShotsList.inert = stale;
  syncModelShotStops();
  const empty = state === 'none' ? 'Choose a checkpoint to see what it has made.'
    : state === 'failed' ? `${name}'s images cannot be shown now: ${reason}`
    : state === 'ready' && !total ? `No images made with ${name} yet.` : '';
  if (els.modelShotsEmpty.textContent !== empty) els.modelShotsEmpty.textContent = empty;
  if (els.modelShotsEmpty.hidden !== !empty) els.modelShotsEmpty.hidden = !empty;
  // Shown only with something to see; the empty line says why not.
  const search = modelShotsSearch(id);
  const hideAll = !(state === 'ready' && total > 0 && search);
  if (els.modelShotsAll.hidden !== hideAll) els.modelShotsAll.hidden = hideAll;
  setModelAttribute(els.modelShotsAll, 'title', `Close this and show the gallery's ${total === 1 ? 'image' : `${total} images`} made with ${name}: every model's, searched for ${search}`);
  setModelAttribute(els.modelShotsAll, 'aria-label', `See all ${total} made with ${name}`);
  // A control that went from under the focus hands it on inside the picker,
  // whose focusout would otherwise close it.
  if (hadFocus && !els.modelShots.contains(document.activeElement)) {
    const target = (stale ? null : modelShotButtons().find((b) => b.tabIndex === 0))
      ?? modelRadios().find((radio) => radio.checked) ?? els.modelPickerBody;
    target.focus({ preventScroll: true });
  }
}

// A delete anywhere (forgetImages): the strip's pictures go, and an open
// strip asks again for its count and the pictures behind them.
function forgetModelShots(gone) {
  if (!modelShots.records.some((rec) => gone.has(rec.id)) && els.modelPicker.hidden) return;
  modelShots.records = modelShots.records.filter((rec) => !gone.has(rec.id));
  if (els.modelPicker.hidden) { modelShots.id = null; return; }
  syncModelShots({ fresh: true });
}

// The picture, in the lightbox over the picker, walking the strip. Its own
// close (dismissLightbox) hands the focus back to the picture.
function openModelShot(button) {
  const rec = modelShots.records.find((r) => String(r.id) === button.dataset.image);
  if (!rec || modelShots.state === 'pending') return;
  modelShotsStop = button.dataset.image;
  syncModelShotStops();
  openLightbox(rec, modelShots.records, 'models');
}

// The picture the lightbox ended on, else the strip's stop, else the chosen
// row: the focus back inside the still-open picker.
function focusModelShot(id) {
  const buttons = modelShotButtons();
  const target = buttons.find((b) => b.dataset.image === String(id))
    ?? buttons.find((b) => b.tabIndex === 0) ?? modelRadios().find((radio) => radio.checked) ?? els.modelPickerBody;
  if (target.dataset.image) { modelShotsStop = target.dataset.image; syncModelShotStops(); }
  target.focus({ preventScroll: true });
}

// See all: the picker closes and the gallery is pointed at this checkpoint -
// every model's images searched for `model:=<id>`, Starred off and the
// sheet closed, so the grid holds what the count said - then the search box
// takes the focus where the results are (showGallerySearch's way).
function showModelShotsGallery() {
  const search = modelShotsSearch(modelShots.id);
  if (!search || modelShots.state !== 'ready' || !modelShots.total) return;
  showModelPicker(false, { focus: false });
  outsideView = null;
  leaveBin();
  els.search.value = search;
  els.favOnly.checked = false;
  els.allModels.checked = true;
  sheetReturn = null;
  if (sheetMode) setSheet(false);
  else resetGallery();
  loadRuns();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelector('.gallery-head').scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  els.search.focus({ preventScroll: true });
}
