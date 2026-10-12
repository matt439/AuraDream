// Local Diffusion's page script: the recycle bin.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The page's delete - a card's ⌦, the viewer's, the lightbox's Delete and
// Del, Delete unstarred - moves images to the recycle bin rather than off the
// disk (`POST /api/images/bin`), so it asks nothing: the note it leaves under
// the gallery's head has Restore, and the bin itself is a view of the gallery
// (Bin, in the footer row) where an image is restored, or deleted for good
// after a question (Delete permanently, Empty recycle bin). An image in the
// bin is gone from everything live - the gallery, references, refine sources,
// stars, runs - and the server says so; its PNG stays where it was until the
// bin is emptied.
//
// The bin view is the gallery's grid on `/api/gallery/bin` (galleryEndpoint):
// every checkpoint's, newest id first, under the search box. The gallery's
// controls that have no meaning there - Sheet, Starred, Runs, Gallery beside,
// Re-render, Swap tags…, Refine…, More seeds… - are held with BIN_HELD as
// their title, never disabled; Delete unstarred gives its place to Restore
// picked and Empty recycle bin. Not remembered: a reload is the gallery.
let binMode = false;
const BIN_HELD = 'Not in the recycle bin — leave it with Bin';
// The bin's size as the server last said it (`/api/gallery/bin/summary`):
// the footer's count, Empty recycle bin's hold.
let binCount = 0;
// The records this page binned, by id, as they were on the page - the grid's,
// the viewer's and #recent's copies, and the grid's query (`galleryEpoch`) at
// the time: a Restore puts each back where it was, while that is still the
// query shown (unforgetImages).
const binnedRecs = new Map();

function keepBinned(rec) {
  const find = (list) => list.find((r) => r.id === rec.id) ?? null;
  binnedRecs.set(rec.id, {
    gallery: find(galleryRecords), viewer: find(viewerRecords), session: find(sessionRecords), epoch: galleryEpoch,
  });
}

// What the server would refuse an image in the bin - a star, a refine - said
// before asking, where a gallery control says things. True where refused.
function refuseBinned(rec, what) {
  if (!rec?.binned_at) return false;
  showError(`#${rec.id} is in the recycle bin; restore it to ${what} it.`, true, { slot: els.galleryError });
  return true;
}

// The footer's count and title, and Empty recycle bin's hold, from the
// server's summary. Asked at load and after everything that moves the bin.
let binSyncEpoch = 0;
async function syncBin() {
  const epoch = ++binSyncEpoch;
  let found;
  try { found = await api('/api/gallery/bin/summary'); } catch { return null; }
  if (epoch === binSyncEpoch) showBinCount(found);
  return found;
}

function showBinCount(found) {
  binCount = found.records;
  const count = els.bin.querySelector('.count');
  const text = binCount ? ` ${binCount}` : '';
  if (count.textContent !== text) count.textContent = text;
  els.bin.setAttribute('aria-label', binCount ? `Recycle bin, ${plural(binCount, 'image')}` : 'Recycle bin');
  els.bin.title = binCount
    ? `Recycle bin: ${plural(binCount, 'image')} (${humanBytes(found.bytes)}). Restore them, or empty it to delete from disk`
    : 'Recycle bin: empty. Deleted images wait here until it is emptied';
  syncEmptyBin();
}

function syncEmptyBin() {
  const why = binCount ? '' : 'The recycle bin is empty';
  els.emptyBin.setAttribute('aria-disabled', String(!!why));
  els.emptyBin.title = why || `Delete all ${plural(binCount, 'image')} in the recycle bin from disk, after a confirmation`;
}

function syncRestorePicked() {
  const n = selectedIds().length;
  els.restorePicked.setAttribute('aria-disabled', String(!n));
  els.restorePicked.title = n ? `Move the ${n === 1 ? 'picked image' : `${n} picked images`} back to the gallery`
    : 'Pick images with Select to restore them';
}

// Everything that reads differently in the bin: Bin pressed, the head's
// name, Delete unstarred's place, and the held controls with their reason.
// A checkbox's title is its label's.
function syncBinControls() {
  els.bin.setAttribute('aria-pressed', String(binMode));
  els.bin.classList.toggle('active', binMode);
  els.galleryTitle.textContent = binMode ? 'Recycle bin' : 'Gallery';
  els.emptyBin.hidden = els.restorePicked.hidden = !binMode;
  for (const el of [els.sheetBtn, els.favOnly]) {
    const titled = el.type === 'checkbox' ? el.closest('label') : el;
    if (binMode) {
      titled.dataset.binTitle ??= titled.title;
      titled.title = BIN_HELD;
    } else if (titled.dataset.binTitle !== undefined) {
      titled.title = titled.dataset.binTitle;
      delete titled.dataset.binTitle;
    }
  }
  for (const el of [els.sheetBtn, els.favOnly, els.runList, els.besideTick]) {
    if (binMode) el.setAttribute('aria-disabled', 'true');
    else el.removeAttribute('aria-disabled');
  }
  // Their own titles follow the bin (runs.js, gallery.js).
  syncRunList();
  syncBesideTitle();
  syncDeleteUnstarred();
  // Re-render, and Swap tags, Refine… and More seeds… through it.
  syncRerender();
  syncEmptyBin();
  syncRestorePicked();
}

// A press on a held control: nothing changes, and the gallery's line says
// why. Taken before the control's own handlers - and a checkbox's tick, a
// select's list - can act.
function refuseInBin(e) {
  if (!binMode) return;
  e.preventDefault();
  e.stopImmediatePropagation();
  showError(`${BIN_HELD}.`, true, { slot: els.galleryError });
}

// The gallery's grid on the bin: the selection is the gallery's, and is set
// aside until the bin is left (the bin's picks are its own, for Restore
// picked); an open sheet closes, putting back the search it replaced; the
// search box stays, and narrows the bin as it does the gallery.
function enterBin() {
  if (binMode) return;
  binMode = true;
  outsideView = null;
  stashSelection();
  syncBinControls();
  if (sheetMode) void setSheet(false);
  else void resetGallery();
  void syncBin();
}

// Back to the gallery's own state, for the caller to query: Bin pressed
// again, Show all, a sheet opened, a search from the lightbox, the Models
// picker's See all. The bin's picks go and the gallery's come back, less any
// binned or deleted meanwhile; one restored from the bin is not added.
// Answers whether it was in the bin.
function leaveBin() {
  if (!binMode) return false;
  binMode = false;
  // Read before the gallery's own records replace the bin's: a pick another
  // tab binned is a card this view showed.
  unstashSelection(new Set(galleryRecords.filter((r) => r.binned_at).map((r) => r.id)));
  syncBinControls();
  return true;
}

function toggleBin() {
  if (binMode) { leaveBin(); void resetGallery(); } else enterBin();
}

// The bin's card: its picture, which opens the lightbox (or picks it in
// select mode), and two tools - ↶ Restore, and ⌦ Delete permanently, last,
// which asks first. No star, Reuse, ❐ or ⊞: the image is gone from all
// they act on until it is restored.
function binCard(div, img, rec) {
  div.classList.add('binned');
  img.removeAttribute('aria-keyshortcuts');
  const restore = tool('↶', `Restore #${rec.id}`, () => { void restoreImages([rec.id]); }, 'Restore');
  restore.classList.add('restore');
  const del = tool('⌦', `Delete #${rec.id} permanently`, () => deleteImage(rec), 'Delete permanently');
  del.classList.add('del');
  const tools = document.createElement('div');
  tools.className = 'tools';
  Object.assign(tools.dataset, { n: rec.id, size: `${rec.width}×${rec.height}`, seed: rec.seed });
  tools.append(restore, del);
  div.append(img, tools);
  cardStops(div, rec.id === galleryCurrent);
  syncCardSelected(div);
  return div;
}

// The undo for the page's delete: "Moved #12 to the recycle bin." and a
// Restore, under the gallery's head (and over the lightbox, where it is
// noteLightboxDelete's). A run of deletes adds to the one note while it is
// still the one shown; any other message ends it.
let binNote = null;
function noteBinned(recs, { also = '', fromLightbox = false } = {}) {
  const fresh = recs.map((r) => r.id);
  const still = binNote && !els.galleryError.hidden && els.galleryError.textContent === binNote.shown;
  const ids = still ? [...binNote.ids, ...fresh.filter((id) => !binNote.ids.includes(id))] : fresh;
  const names = ids.length > 6 ? plural(ids.length, 'image') : imageNames(ids);
  const text = `Moved ${names} to the recycle bin.${also}`;
  const decorate = (slot) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ghost tiny';
    button.textContent = 'Restore';
    button.title = `Move ${names} back to the gallery`;
    button.dataset.binRestore = '';
    button.onclick = () => { void restoreImages(ids); };
    slot.append(' ', button);
  };
  if (fromLightbox) noteLightboxDelete(text, decorate);
  else {
    showError(text, true, { slot: els.galleryError });
    decorate(els.galleryError);
  }
  binNote = { ids, shown: els.galleryError.textContent };
}

// Out of the bin and back in the gallery, usable again at once. In the bin
// view each card leaves (a tombstone, "Restored · #N"); a card or tile this
// page binned comes back in its own box (unforgetImages). Answers the
// server's answer, or null where it refused.
async function restoreImages(ids, { fromLightbox = els.lightbox.open } = {}) {
  const slot = { slot: els.galleryError };
  let result;
  try {
    result = await api('/api/images/restore', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids }),
    });
  } catch (err) {
    showError(err.message, false, slot);
    return null;
  }
  const { restored, not_binned: notBinned, missing } = result;
  // Out of the bin either way: restored here, or elsewhere first, or gone.
  if (binMode) {
    dropFromBinView([...restored, ...notBinned], 'Restored');
    dropFromBinView(missing, 'Deleted');
  }
  // Restored elsewhere first is live all the same; emptied elsewhere is gone
  // for good, which the passes of it and the tombstones now say ("(deleted)").
  unforgetImages([...restored, ...notBinned]);
  const also = missing.length ? forgetImages(missing) : '';
  const are = (list) => (list.length === 1 ? 'was' : 'were');
  // Named up to six, as noteBinned does, else counted: a Restore picked over
  // hundreds was one alert reading out every id.
  const names = (list) => (list.length > 6 ? plural(list.length, 'image') : imageNames(list));
  const said = [
    restored.length ? `Restored ${names(restored)}.` : '',
    notBinned.length ? `${names(notBinned)} ${are(notBinned)} not in the recycle bin.` : '',
    missing.length ? `${names(missing)} ${are(missing)} already deleted.` : '',
  ].filter(Boolean).join(' ') + also;
  if (fromLightbox && els.lightbox.open) noteLightboxDelete(said);
  else showError(said, true, slot);
  // A run comes back to the Runs picker with its images.
  loadRuns();
  void syncBin();
  return result;
}

// The bin view's cards for these ids gone from it, each a tombstone in its
// box, as forgetImages leaves the gallery's: the records, the count, the
// selection, the keyboard's card and focus, and the lightbox walking them.
function dropFromBinView(ids, word) {
  const out = new Set(ids);
  if (!out.size) return;
  const focused = document.activeElement;
  const current = currentCard();
  let lost = null;
  els.gallery.querySelectorAll('.card[data-id]').forEach((el) => {
    if (!out.has(+el.dataset.id)) return;
    const held = el.contains(focused);
    tombstone(el, word);
    if (held) lost = el;
  });
  const before = galleryRecords.length;
  for (let at = galleryRecords.length - 1; at >= 0; at--) if (out.has(galleryRecords[at].id)) galleryRecords.splice(at, 1);
  if (galleryRecords.length !== before) {
    total -= before - galleryRecords.length;
    els.count.textContent = total ? `(${total})` : '';
    syncGalleryEmpty();
  }
  forgetSelected(out, { stashed: false });
  if (current?.classList.contains('gone')) makeCurrent(liveNeighbour(current));
  if (lost) {
    const next = liveNeighbour(lost);
    if (next) next.querySelector('img')?.focus();
  }
  if (lbList === galleryRecords) lightboxAfterGone();
}

// Delete permanently, the bin's ⌦ and the lightbox's Delete on an image in
// the bin: after its question (confirmDelete), the row and the PNG for good.
async function deletePermanently(rec) {
  // Asked before the confirm, so it can name the passes - on any checkpoint.
  // A failed look asks without them; the delete then says what is wrong.
  let passes = null;
  try {
    passes = await api(`/api/gallery?limit=3&all_models=true&search=refine:${rec.id}`);
  } catch { /* asked without */ }
  if (!await confirmDelete(rec, passes)) return;
  const fromLightbox = els.lightbox.open;
  let already = false;
  try {
    await api(`/api/images/${rec.id}`, { method: 'DELETE' });
  } catch (err) {
    if (!/unknown image/.test(err.message)) {
      showError(err.message, false, { slot: els.galleryError });
      return;
    }
    already = true;
  }
  const also = forgetImages([rec.id]);
  const said = already ? `#${rec.id} was already deleted.` : `Deleted #${rec.id} (seed ${rec.seed}) from disk.`;
  if (fromLightbox && els.lightbox.open) noteLightboxDelete(said + also);
  else if (already || also) showError((already ? said + also : also).trim(), true, { slot: els.galleryError });
  void syncBin();
}

// Empty recycle bin: every image in it, every checkpoint's and whatever the
// search shows, deleted from disk after a question that says how many and
// how much, what of the live gallery leans on them, and whether a job is
// running. The count is pinned (`expected`): an image binned meanwhile - in
// another tab - refuses the whole delete.
async function emptyBin() {
  const slot = { slot: els.galleryError };
  let found;
  try {
    found = await api('/api/gallery/bin/summary');
  } catch (err) {
    showError(err.message, false, slot);
    return;
  }
  showBinCount(found);
  if (!found.records) { showError('The recycle bin is empty.', true, slot); return; }
  const n = found.records;
  const inUse = found.sources_in_use.length;
  const message = [
    `Permanently delete all ${plural(n, 'image')} in the recycle bin (${humanBytes(found.bytes)})?`,
    ...(els.search.value.trim() ? ['That is every one of them, not only those this search shows.'] : []),
    ...(inUse ? [`${inUse} of them ${inUse === 1 ? 'is' : 'are'} the source or a reference of an image outside the bin; `
      + `${inUse === 1 ? 'that' : 'those'} can't be made again.`] : []),
    ...(found.job_active ? ['A job is running; a cell that uses one of them will fail.'] : []),
    'This cannot be undone.',
  ].join('\n');
  if (!await askConfirm(message, { okLabel: 'Empty bin' })) return;
  let result;
  try {
    result = await api(`/api/gallery/bin?expected=${n}`, { method: 'DELETE' });
  } catch (err) {
    // The bin changed since the question: shown again as it is now.
    if (err.status === 409 && binMode) void resetGallery().then(() => showError(err.message, false, slot));
    else showError(err.message, false, slot);
    void syncBin();
    return;
  }
  const also = forgetImages(result.deleted);
  // Tombstones over an emptied bin say less than the bin's empty line.
  if (binMode) void resetGallery();
  const went = result.deleted.length;
  showError(`Deleted ${plural(went, 'image')} from the recycle bin.${also}`, true, slot);
  loadRuns();
  void syncBin();
}

function restorePicked() {
  const ids = selectedIds();
  if (!ids.length) { showError('Pick images with Select to restore them.', true, { slot: els.galleryError }); return; }
  void restoreImages(ids);
}

function setupBin() {
  els.bin.onclick = toggleBin;
  els.emptyBin.onclick = () => {
    if (!binCount) { showError('The recycle bin is empty.', true, { slot: els.galleryError }); return; }
    void emptyBin();
  };
  els.restorePicked.onclick = restorePicked;
  onSelectionChange(syncRestorePicked);
  for (const el of [els.sheetBtn, els.favOnly, els.besideTick]) el.addEventListener('click', refuseInBin, true);
  els.runList.addEventListener('mousedown', refuseInBin, true);
  els.runList.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab' && e.key !== 'Escape' && e.key !== 'Shift') refuseInBin(e);
  }, true);
  els.lbRestore.onclick = () => { const rec = lbList[lbIndex]; if (rec) void restoreImages([rec.id]); };
  syncBinControls();
  void syncBin();
}
