// AuraDream's page script: the Models table's favourites - the stars, kept by
// the server for the whole gallery (`/api/models/favourites`), this page's
// presses until the server confirms them, and the one-time offer to import
// the stars a browser kept before the server held them. The table, its
// grouping and the selection are model-switch.js's, loaded before this
// file; each reaches the other only at call time (renderModelOptions and
// syncModelSwitch from here, syncModelStars and modelFavouriteId from there).
// Declarations only; app.js wires the stars and the offer's buttons, and
// state.js has the shared-global load-order contract.
'use strict';

// The stars are the gallery's, on the server (`/api/models/favourites`,
// backend/model_favourites.py): one revisioned set every browser, address
// and device reaching this server shares. `modelFavouritesRevision` is the
// newest set this page has read (-1: none yet), so an older answer arriving
// late never undoes a newer one. `modelFavouriteChanges` holds this page's
// presses the server has not yet confirmed (id -> starred), laid over every
// set read, and those it refused, kept for this page with the reason.
let modelFavourites = new Set();
let modelFavouritesRevision = -1;
let modelFavouriteMessage = null;
let modelFavouriteChanges = new Map();
// Before the server kept them, the stars lived in each origin's
// localStorage under this key. It is only read now, for the one-time
// offer to import that set (syncModelFavouriteOffer); the import's own key
// remembers that it was taken or declined, so it is never offered again.
const MODEL_FAVOURITES_KEY = 'auradream.model-favourites.v1';
const MODEL_FAVOURITES_IMPORT_KEY = 'auradream.model-favourites-import.v1';
// The page's own memory of the answer when the store refuses to keep it.
let modelFavouriteOfferAnswered = false;
let modelFavouriteImporting = false;

// The earlier per-browser set, as the server would take it: trimmed text of
// at most 512 characters without control characters, each once.
function parseModelFavourites(raw) {
  try {
    const saved = JSON.parse(raw ?? '[]');
    if (!Array.isArray(saved)) return [];
    return [...new Set(saved.filter((id) => typeof id === 'string')
      .map((id) => id.trim())
      .filter((id) => id && id.length <= 512 && !/\p{Cc}/u.test(id)))];
  } catch { return []; }
}

function applyUnsavedModelFavourites() {
  for (const [id, on] of modelFavouriteChanges) {
    if (on) modelFavourites.add(id);
    else modelFavourites.delete(id);
  }
}

// A set the server answered: taken unless an answer already read is newer.
// True for the page's first set.
function receiveModelFavouriteSet(body) {
  if (!body || !Number.isInteger(body.revision) || !Array.isArray(body.items)) return false;
  if (body.revision < modelFavouritesRevision) return false;
  const first = modelFavouritesRevision < 0;
  modelFavouritesRevision = body.revision;
  modelFavourites = new Set(body.items.filter((id) => typeof id === 'string'));
  applyUnsavedModelFavourites();
  return first;
}

// Read at the page's start and as Models opens, never on a timer. The
// stars change in place; the grouping waits for the next opening, even for
// the page's first set while Models is open (a finger, the focus or an open
// list may be on a row that regrouping would move).
async function refreshModelFavourites() {
  try {
    const first = receiveModelFavouriteSet(await api('/api/models/favourites'));
    if (!modelFavouriteChanges.size) modelFavouriteMessage = null;
    if (first && modelCatalogueLoaded && els.modelPicker.hidden) {
      modelTableGrouping = new Set(modelFavourites);
      renderModelOptions(modelChoice());
    } else syncModelStars();
    if (first && els.modelPicker.hidden) syncModelFavouriteOffer();
  } catch (err) {
    // Offline, or signed out (the gate's 401), the topbar already says so.
    if (!err.offline && err.status !== 401) {
      modelFavouriteMessage = `The favourites could not be read: ${err.message}`;
    }
  }
  syncModelSwitch();
}

function modelFavouriteOfferAnsweredHere() {
  if (modelFavouriteOfferAnswered) return true;
  try { return localStorage.getItem(MODEL_FAVOURITES_IMPORT_KEY) !== null; }
  catch { return true; } // A blocked store has no earlier set to offer.
}

function answerModelFavouriteOffer(answer) {
  modelFavouriteOfferAnswered = true;
  try { localStorage.setItem(MODEL_FAVOURITES_IMPORT_KEY, answer); }
  catch { /* This page remembers; the next may ask again. */ }
}

// The earlier set's ids the server's set lacks.
function modelFavouritesToImport() {
  let saved = [];
  try { saved = parseModelFavourites(localStorage.getItem(MODEL_FAVOURITES_KEY)); }
  catch { return []; }
  return saved.filter((id) => !modelFavourites.has(modelFavouriteId(id)));
}

// The offer above the table: decided while the picker is hidden or as it
// opens, never while it stays open, so it never appears or goes under the
// pointer. A press keeps its line's height until the next opening.
function syncModelFavouriteOffer() {
  const offer = els.modelFavouritesOffer;
  offer.style.minHeight = '';
  const ids = modelFavouritesRevision < 0 || modelFavouriteOfferAnsweredHere() ? [] : modelFavouritesToImport();
  offer.hidden = !ids.length;
  if (!ids.length) return;
  const count = ids.length === 1 ? 'one model favourite' : `${ids.length} model favourites`;
  els.modelFavouritesOfferText.textContent = `This browser kept ${count} from before the server held them. `
    + `Import ${ids.length === 1 ? 'it' : 'them'} into this gallery's favourites, which every device shares?`;
  els.btnModelFavouritesImport.hidden = false;
  els.btnModelFavouritesDismiss.hidden = false;
}

function settleModelFavouriteOffer(words) {
  const offer = els.modelFavouritesOffer;
  offer.style.minHeight = `${offer.offsetHeight}px`;
  els.btnModelFavouritesImport.hidden = true;
  els.btnModelFavouritesDismiss.hidden = true;
  els.modelFavouritesOfferText.textContent = words;
  // The pressed button is gone: the focus moves to the line saying why.
  offer.focus({ preventScroll: true });
}

async function importModelFavourites() {
  if (modelFavouriteImporting) return;
  const ids = modelFavouritesToImport();
  if (!ids.length) {
    answerModelFavouriteOffer('imported');
    settleModelFavouriteOffer('These favourites are already in this gallery.');
    return;
  }
  modelFavouriteImporting = true;
  els.btnModelFavouritesImport.setAttribute('aria-disabled', 'true');
  els.btnModelFavouritesImport.title = 'Importing this browser\'s favourites…';
  els.btnModelFavouritesDismiss.setAttribute('aria-disabled', 'true');
  els.btnModelFavouritesDismiss.title = 'Wait for the import to finish';
  try {
    const body = await api('/api/models/favourites/import', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_ids: ids }),
    });
    receiveModelFavouriteSet(body);
    answerModelFavouriteOffer('imported');
    const added = Number.isInteger(body?.imported) ? body.imported : ids.length;
    settleModelFavouriteOffer(added === 0 ? 'These favourites are already in this gallery.'
      : added === 1 ? 'Imported one favourite. It is listed first the next time Models opens.'
        : `Imported ${added} favourites. They are listed first the next time Models opens.`);
    syncModelStars();
  } catch (err) {
    els.modelFavouritesOfferText.textContent = `The favourites were not imported: ${err.message} `;
  } finally {
    modelFavouriteImporting = false;
    els.btnModelFavouritesImport.removeAttribute('aria-disabled');
    els.btnModelFavouritesImport.title = "Add this browser's earlier favourites to this gallery's";
    els.btnModelFavouritesDismiss.removeAttribute('aria-disabled');
    els.btnModelFavouritesDismiss.title = "Keep this gallery's favourites as they are; this browser won't ask again";
  }
}

function dismissModelFavouriteOffer() {
  if (modelFavouriteImporting) return;
  answerModelFavouriteOffer('declined');
  settleModelFavouriteOffer("Not imported. This browser won't ask again.");
}

function modelFavouriteId(modelId) {
  return modelCatalogueDetails[modelId]?.favourite_id || modelId;
}

function syncModelStars() {
  for (const star of els.modelTable.querySelectorAll('.model-star')) {
    const id = star.closest('.model-row').dataset.model;
    const on = modelFavourites.has(modelFavouriteId(id));
    const name = modelLabel(id);
    const glyph = on ? '★' : '☆';
    if (star.textContent !== glyph) star.textContent = glyph;
    setModelAttribute(star, 'aria-pressed', String(on));
    setModelAttribute(star, 'title', on ? `${name} is a favourite — click to remove`
      : `Add ${name} to this gallery's favourites`);
  }
}

// A press stars or unstars at once, in place; the server answers the whole
// set at its new revision. Each press sends only its own star (a set
// operation), so it never undoes another device's. One star's writes go one
// at a time: a press while its star's write is on its way only records the
// wish, and that write's answer sends the newest wish if it differs, so the
// server never takes two presses of one star out of order. A refused write
// keeps the press for this page and says why in the note.
const modelFavouriteSending = new Set();
async function toggleModelFavourite(modelId) {
  const id = modelFavouriteId(modelId);
  const on = !modelFavourites.has(id);
  if (on) modelFavourites.add(id);
  else modelFavourites.delete(id);
  modelFavouriteChanges.set(id, on);
  // The star changes in place; the choice and pending loading settings stay.
  syncModelStars();
  syncModelSwitch();
  if (modelFavouriteSending.has(id)) return;
  modelFavouriteSending.add(id);
  try {
    while (modelFavouriteChanges.has(id)) {
      const wanted = modelFavouriteChanges.get(id);
      let body;
      try {
        body = await api('/api/models/favourites', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: id, favourite: wanted }),
        });
      } catch (err) {
        modelFavouriteMessage = `Favourites changed for this page, but the server could not save them: ${err.message}`;
        break;
      }
      // A later press of the same star sends next, from this loop.
      if (modelFavouriteChanges.get(id) === wanted) modelFavouriteChanges.delete(id);
      receiveModelFavouriteSet(body);
      if (!modelFavouriteChanges.size) modelFavouriteMessage = null;
    }
  } finally {
    modelFavouriteSending.delete(id);
  }
  syncModelStars();
  syncModelSwitch();
}
