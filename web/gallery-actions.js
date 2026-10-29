// AuraDream's page script: starring, deleting (into the recycle bin) and
// forgetting gallery images - one, or the picked cards together - and taking
// them back once restored.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// An ordinary same-page favorite request owns these ids until its response is
// applied. A refusal/error releases the hold; it proves no cross-tab ordering
// or unknown-outcome recovery and never triggers an automatic retry.
const favoriteWriteIds = new Set();

function favoriteWriteReason(ids) {
  return ids.some((id) => favoriteWriteIds.has(id))
    ? 'Wait for the current star change to finish' : '';
}

function syncFavoriteWriteHolds(ids) {
  for (const id of ids) {
    const shown = `.card[data-id="${id}"], .sheet-cell[data-id="${id}"], .result[data-id="${id}"]`;
    document.querySelectorAll(shown).forEach((element) => {
      element.querySelectorAll('.star').forEach((star) => {
        // Use the displayed confirmed value: a sheet source can be absent
        // from recordsWithId until starCell resumes after setFavorite.
        syncStar(star, star.getAttribute('aria-pressed') === 'true', id);
      });
    });
  }
  if (els.lightbox.open && ids.includes(lbList[lbIndex]?.id)) syncLightboxFavoriteHold();
  syncPickedButton();
}

// The same image can be on the page more than once - in the viewer, in the
// gallery grid, on the sheet, in the lightbox - as separate record objects
// fetched at different times, or retained as a pick after its card leaves
// the filter (also while the bin sets those picks aside). Apply by id: to every record
// that carries it, and to every element tagged with it.
function recordsWithId(id) {
  return [
    ...galleryRecords, ...viewerRecords, ...sessionRecords, ...lbList,
    ...selection.values(), ...(stashedSelection?.picks.values() ?? []),
  ].filter((r) => r.id === id);
}

// `say` is F's: the key stars an image from wherever focus is in its card,
// sheet cell or lightbox, and only a press on the star itself changes the
// state of what has focus - so otherwise the outcome is said (sayStar).
async function setFavorite(rec, value, { say = false } = {}) {
  // The server refuses a star in the recycle bin as an unknown image, which
  // below would forget the card; said instead (bin.js).
  if (refuseBinned(rec, 'star')) return;
  const ids = [rec.id];
  const held = favoriteWriteReason(ids);
  if (held) { showError(held + '.', true, { slot: els.galleryError }); return; }
  for (const id of ids) favoriteWriteIds.add(id);
  try {
    syncFavoriteWriteHolds(ids);
    try {
      await api(`/api/images/${rec.id}/favorite?value=${value}`, { method: 'POST' });
    } catch (err) {
      // A star is a gallery control, and says what went wrong where they do -
      // not in the panel's line a column away (the lightbox's line mirrors it).
      if (!/unknown image/.test(err.message)) {
        showError(err.message, false, { slot: els.galleryError });
        return;
      }
      // Deleted elsewhere - another tab - while this page still showed it: said
      // as ⌦ says it in the same case, and forgotten everywhere (forgetImages)
      // rather than left on the page to be starred again. "unknown image" in
      // the panel's line said neither, and stayed on under the next image.
      const fromLightbox = els.lightbox.open;
      const said = `#${rec.id} was already deleted.${forgetImages([rec.id])}`;
      if (fromLightbox && els.lightbox.open) noteLightboxDelete(said);
      else showError(said, true, { slot: els.galleryError });
      // The Runs picker counts images, as after ⌦.
      loadRuns();
      return;
    }
    markFavorite(rec.id, value);
    // A star with focus says its own new state (aria-pressed); anywhere else
    // F changed a button nobody was on, and nothing was heard.
    if (say && !document.activeElement?.matches?.('.star, #lb-fav')) sayStar(rec, value);
    void loadRuns();
  } finally {
    for (const id of ids) favoriteWriteIds.delete(id);
    syncFavoriteWriteHolds(ids);
  }
}

// A star the server has taken, on every record that carries the id and every
// element showing the image - a card, a sheet cell, a result beside the
// viewer - by class and by its own star button, never by re-rendering: the
// sheet rebuilds up to a thousand <img> nodes and resets the scroll per
// keypress.
function markFavorite(id, value) {
  recordsWithId(id).forEach((r) => { r.favorite = value; });
  const shown = `.card[data-id="${id}"], .sheet-cell[data-id="${id}"], .result[data-id="${id}"]`;
  document.querySelectorAll(shown).forEach((el) => {
    el.classList.toggle('fav', value);
    el.querySelectorAll('.star').forEach((star) => syncStar(star, value, id));
  });
  if (els.lightbox.open && lbList[lbIndex]?.id === id) renderLightbox();
}

// Ids as a note names them: up to six by number, more by count, as the
// recycle bin's notes do (noteBinned, restoreImages).
const namesOrCount = (ids) => (ids.length > 6 ? plural(ids.length, 'image') : imageNames(ids));
const isAre = (ids) => (ids.length === 1 ? 'is' : 'are');
const wasWere = (ids) => (ids.length === 1 ? 'was' : 'were');

// Star picked and Unstar picked (the footer's Picked menu): the gallery's
// picked cards in one request (`POST /api/images/favorite`), each mark set
// as a card's star sets it. The server answers per id, as the bin does: one
// binned or deleted meanwhile - another tab - is said and forgotten
// (forgetImages), and the rest are starred all the same. F stays the card's
// own, in select mode too (gallery-select.js). The picks stay picked.
async function starPicked(value) {
  const slot = { slot: els.galleryError };
  const ids = selectedIds();
  if (!ids.length) { showError(`${PICKED_NONE}.`, true, slot); return; }
  const held = favoriteWriteReason(ids);
  if (held) { showError(held + '.', true, slot); return; }
  for (const id of ids) favoriteWriteIds.add(id);
  try {
    syncFavoriteWriteHolds(ids);
    let result;
    try {
      result = await api('/api/images/favorite', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids, value }),
      });
    } catch (err) {
      showError(err.message, false, slot);
      return;
    }
    const { changed, already, binned, missing } = result;
    for (const id of [...changed, ...already]) markFavorite(id, value);
    const also = (binned.length ? forgetImages(binned, { binned: true }) : '')
      + (missing.length ? forgetImages(missing) : '');
    const said = [
      changed.length ? `${value ? 'Starred' : 'Unstarred'} ${namesOrCount(changed)}.` : '',
      already.length ? `${namesOrCount(already)} ${wasWere(already)} ${value ? 'already starred' : 'not starred'}.` : '',
      binned.length ? `${namesOrCount(binned)} ${isAre(binned)} in the recycle bin.` : '',
      missing.length ? `${namesOrCount(missing)} ${wasWere(missing)} already deleted.` : '',
    ].filter(Boolean).join(' ') + also;
    showError(said, true, slot);
    void loadRuns();
  } finally {
    for (const id of ids) favoriteWriteIds.delete(id);
    syncFavoriteWriteHolds(ids);
  }
}

// Bin picked (the Picked menu): the picked cards into the recycle bin at
// once, asking nothing, as a card's ⌦ does - the note it leaves has Restore
// for all of them (noteBinned). Through forgetImages, which takes them out
// of the selection and leaves each card a tombstone in its place, so nothing
// slides under the pointer. Already in the bin (another tab) or deleted is
// said after the note, and forgotten as well.
async function binPicked() {
  const slot = { slot: els.galleryError };
  const recs = selectedRecords();
  if (!recs.length) { showError(`${PICKED_NONE}.`, true, slot); return; }
  const ids = recs.map((r) => r.id);
  const held = binWriteReason(ids);
  if (held) { showError(held + '.', true, slot); return; }
  for (const id of ids) binWriteIds.add(id);
  try {
    syncBinWriteHolds(ids);
    let result;
    try {
      result = await api('/api/images/bin', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids }),
      });
    } catch (err) {
      showError(err.message, false, slot);
      return;
    }
    const { binned, already, missing } = result;
    // Kept before they are forgotten, so a Restore can put their cards back.
    const moved = recs.filter((r) => binned.includes(r.id)).sort((a, b) => a.id - b.id);
    for (const rec of moved) keepBinned(rec);
    let also = forgetImages([...binned, ...already], { binned: true })
      + (missing.length ? forgetImages(missing) : '');
    also += [
      already.length ? ` ${namesOrCount(already)} ${wasWere(already)} already in the recycle bin.` : '',
      missing.length ? ` ${namesOrCount(missing)} ${wasWere(missing)} already deleted.` : '',
    ].join('');
    if (moved.length) noteBinned(moved, { also });
    else showError(also.trim(), true, slot);
    loadRuns();
    void syncBin();
  } finally {
    for (const id of ids) binWriteIds.delete(id);
    syncBinWriteHolds(ids);
  }
}

// What F did, for a screen reader: "Starred #12", "Unstarred #12", in a
// visually hidden polite line - the lightbox's own while it is up, since the
// page behind a modal dialog is inert, live regions and all. Emptied first
// and written on the next frame, so the same words again are heard again,
// and emptied after a while, so browse mode does not come on a stale one.
let starSaidTimer = 0;
function sayGalleryAction(text) {
  const line = els.lightbox.open ? els.lbStarSaid : els.starSaid;
  els.starSaid.textContent = '';
  els.lbStarSaid.textContent = '';
  clearTimeout(starSaidTimer);
  requestAnimationFrame(() => { line.textContent = text; });
  starSaidTimer = setTimeout(() => { if (line.textContent === text) line.textContent = ''; }, 5000);
}

function sayStar(rec, value) {
  sayGalleryAction(`${value ? 'Starred' : 'Unstarred'} #${rec.id}`);
}

// Gallery ids as the page names images: "#8", "#8 and #4", "#8, #4 and #2".
function imageNames(ids) {
  const names = ids.map((id) => `#${id}`);
  return names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names.at(-1)}` : names.join('');
}

// The delete's two questions, in a dialog of the page's own (#confirm-dialog)
// beside what was pressed: the browser's confirm and prompt were a light box
// at the top of the window, far from the card whose ⌦ asked, and held the
// whole page while they were up - a headless run waited on one for good.
// `message` is the question's lines as they used to be shown: the first is
// the title, a count's last is its box's label, the rest its paragraphs.
// confirmDialog answers true for Delete and false for Cancel, Escape or a
// press outside it; countDialog the typed text, or null. `okLabel` is the
// OK button's words, "Delete" unless the question names its own (Empty
// recycle bin's "Empty bin"); `destroy` false paints it as an ordinary
// primary, for a question whose OK queues work rather than deletes (Re-render
// picked at a tier). The smokes stand in for them through
// `askConfirm` and `askCount`, and put these back.
let askAnswer = null;
function openAsk(message, { count = false, okLabel = 'Delete', destroy = true } = {}) {
  // One question at a time: an earlier one still up is answered Cancel.
  if (els.confirmDialog.open) {
    const earlier = askAnswer;
    askAnswer = null;
    els.confirmDialog.close('');
    earlier?.('');
  }
  const lines = message.split('\n');
  els.confirmTitle.textContent = lines.shift();
  const label = count ? lines.pop() : '';
  els.confirmText.replaceChildren(...lines.filter((line) => line.trim()).map((line) => {
    const p = document.createElement('p');
    p.textContent = line.trim();
    return p;
  }));
  els.confirmCountField.hidden = !count;
  els.confirmCountLabel.textContent = label;
  els.confirmCount.value = '';
  els.confirmOk.textContent = okLabel;
  els.confirmOk.classList.toggle('destroy', destroy);
  const anchor = document.activeElement !== document.body ? document.activeElement : null;
  return new Promise((resolve) => {
    askAnswer = resolve;
    els.confirmDialog.returnValue = '';
    els.confirmDialog.showModal();
    placeAsk(anchor);
    (count ? els.confirmCount : els.confirmCancel).focus();
  });
}

function confirmDialog(message, { okLabel, destroy } = {}) {
  return openAsk(message, { okLabel, destroy }).then((answer) => answer === 'ok');
}

function countDialog(message) {
  return openAsk(message, { count: true }).then((answer) => (answer === 'ok' ? els.confirmCount.value : null));
}

let askConfirm = confirmDialog;
let askCount = countDialog;

// Beside what asked, clear of the image it asks about - its card or tile, not
// just the button, which sits on the picture: under it, over it, then to its
// right or left, whichever the window has room for, else as near as fits -
// and inside the window. Centred where nothing on the page has focus.
function placeAsk(anchor) {
  const dialog = els.confirmDialog;
  for (const side of ['margin', 'top', 'left', 'right', 'bottom']) dialog.style[side] = '';
  const near = anchor?.closest?.('.card, .viewer > figure, .sheet-cell, .lb-actions') ?? anchor;
  const at = near?.getBoundingClientRect();
  if (!at || (!at.width && !at.height)) return;
  const box = dialog.getBoundingClientRect();
  const room = 8;
  // The window less its scrollbars: innerWidth counts the page's, and a
  // dialog placed for a card in the right-hand column ran 7 px under it.
  const { clientWidth: width, clientHeight: height } = document.documentElement;
  const clampX = (x) => Math.max(room, Math.min(x, width - box.width - room));
  const clampY = (y) => Math.max(room, Math.min(y, height - box.height - room));
  const centreX = at.left + at.width / 2 - box.width / 2;
  const centreY = at.top + at.height / 2 - box.height / 2;
  const places = [
    [centreX, at.bottom + room, at.bottom + room + box.height <= height - room],
    [centreX, at.top - room - box.height, at.top - room - box.height >= room],
    [at.right + room, centreY, at.right + room + box.width <= width - room],
    [at.left - room - box.width, centreY, at.left - room - box.width >= room],
  ];
  const [x, y] = places.find(([, , fits]) => fits) ?? places[0];
  Object.assign(dialog.style, {
    margin: '0', top: `${clampY(y)}px`, left: `${clampX(x)}px`, right: 'auto', bottom: 'auto',
  });
}

function askClosed() {
  // The close of a question another has replaced, arriving once that one is up.
  if (els.confirmDialog.open) return;
  const answer = askAnswer;
  askAnswer = null;
  answer?.(els.confirmDialog.returnValue);
}

// A press outside the dialog's box - on its backdrop - is Cancel.
function askBackdrop(event) {
  if (event.target !== els.confirmDialog) return;
  const box = els.confirmDialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right
    && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) els.confirmDialog.close('');
}

// The recycle bin's Delete permanently (bin.js) removes the PNG as well as the
// row, for good. Say which image, so a slip is caught here: the cells of a
// sweep share a prompt and a seed, and a pass shares both with its source, so
// the settings line is what tells them apart. And say what else it costs -
// `passes` is the gallery's answer to `refine:<id>`, the live passes made
// from this image, which could not be made again without it. The page's
// ordinary delete asks nothing: it moves the image to the bin, and the note
// it leaves has Restore.
function confirmDelete(rec, passes) {
  const snippet = rec.prompt.length > 60 ? `${rec.prompt.slice(0, 57)}…` : rec.prompt;
  const costs = [];
  if (passes?.total === 1) {
    costs.push(`#${passes.items[0].id} was refined from this one; once it is gone, that pass cannot be made again.`);
  } else if (passes?.total > 1) {
    const named = passes.items.map((r) => `#${r.id}`).join(', ') + (passes.total > passes.items.length ? ', …' : '');
    costs.push(`${passes.total} refine passes were made from this one (${named}); once it is gone, they cannot be made again.`);
  }
  const at = referenceIds.indexOf(rec.id);
  if (at >= 0) costs.push(`It is reference image ${at + 1} in the panel, and leaves the references with it.`);
  return askConfirm([
    `Delete permanently #${rec.id}${rec.favorite ? ' (starred)' : ''}?`,
    '',
    settingsLine(rec),
    `"${snippet}"`,
    ...(costs.length ? ['', ...costs] : []),
    '',
    'The PNG is deleted from disk; this cannot be undone.',
  ].join('\n'));
}

// The ids whose delete is under way, from the press to the server's answer:
// a second press in that time - a double-click on ⌦ - sent a second request
// for the same image, and Delete permanently asked about it twice.
const deleting = new Set();

async function deleteImage(rec) {
  const ids = [rec.id];
  const held = binWriteReason(ids);
  if (held) { showError(held + '.', true, { slot: els.galleryError }); return; }
  for (const id of ids) binWriteIds.add(id);
  deleting.add(rec.id);
  try {
    syncBinWriteHolds(ids);
    await deleteOne(rec);
  } finally {
    deleting.delete(rec.id);
    for (const id of ids) binWriteIds.delete(id);
    syncBinWriteHolds(ids);
  }
}

// ⌦, the lightbox's Delete and Del: into the recycle bin, at once and asking
// nothing - the note it leaves has Restore. An image already in the bin (the
// bin's own cards, the lightbox walking them) is deleted for good instead,
// after its question (deletePermanently, bin.js).
async function deleteOne(rec) {
  if (rec.binned_at) { await deletePermanently(rec); return; }
  const fromLightbox = els.lightbox.open;
  let result;
  try {
    result = await api('/api/images/bin', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids: [rec.id] }),
    });
  } catch (err) {
    showError(err.message, false, { slot: els.galleryError });
    return;
  }
  // Gone already - from another tab, or a record a reattached job had
  // replayed: what the delete was for has happened, so it is a note.
  const missing = result.missing.includes(rec.id);
  const already = result.already.includes(rec.id);
  // Kept before it is forgotten, so a Restore can put its card back.
  if (!missing) keepBinned(rec);
  const also = forgetImages([rec.id], { binned: !missing });
  if (missing || already) {
    const said = `#${rec.id} was already ${missing ? 'deleted' : 'in the recycle bin'}.${also}`;
    if (fromLightbox && els.lightbox.open) noteLightboxDelete(said);
    else showError(said, true, { slot: els.galleryError });
  } else {
    // A card's tombstone says it in place, and the note under the head
    // offers it back; the lightbox has moved on to the next image, so its
    // note says which one went, until the next step.
    noteBinned([rec], { also, fromLightbox: fromLightbox && els.lightbox.open });
  }
  // The Runs picker counts images; it is the gallery's table of contents.
  loadRuns();
  void syncBin();
}

// After a delete in the lightbox the next image is in view at the same zoom,
// and on a sweep's near-identical cells nothing showed that anything had
// happened. Said in the gallery's slot, which mirrors into the lightbox's
// role=alert line, and taken back at the next key or press in the lightbox.
// `decorate(slot)` adds to each copy what text alone cannot carry - the
// recycle bin's Restore (noteBinned), whose own press and keys leave the
// note up, as do Del and the Delete button's own press, whose next note adds
// to it (noteBinned) or replaces it.
let lightboxDeleteNote = null;
function noteLightboxDelete(text, decorate = null) {
  showError(text, true, { slot: els.galleryError });
  if (decorate) {
    decorate(els.galleryError);
    if (els.lightbox.open) decorate(els.lbError);
  }
  const shown = els.galleryError.textContent;
  const mirrored = els.lbError.textContent;
  // Each copy only while it still says this: a job ending meanwhile resets
  // the gallery's line and leaves the lightbox's. The key and the press are
  // one listener pair, both taken off by whichever comes first - and by the
  // next delete's note - so a run of deletes leaves none behind.
  lightboxDeleteNote?.abort();
  const note = new AbortController();
  lightboxDeleteNote = note;
  const clear = (e) => {
    if (e?.target?.closest?.('[data-bin-restore], #lb-delete') || e?.key === 'Delete') return;
    note.abort();
    if (lightboxDeleteNote === note) lightboxDeleteNote = null;
    if (els.galleryError.textContent === shown) showError('', false, { slot: els.galleryError });
    else if (els.lbError.textContent === mirrored) { els.lbError.hidden = true; els.lbError.textContent = ''; }
  };
  els.lightbox.addEventListener('keydown', clear, { signal: note.signal });
  els.lightbox.addEventListener('pointerdown', clear, { signal: note.signal });
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    els.lbImg.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 180 });
  }
}

// Every id this page has seen deleted (forgetImages). A pass that arrives
// after its source went - its cell read the source as it began and was saved
// after the delete, or its job's stream sent it late or replays it - is
// marked as forgetImages marked the records already here (markGoneSource,
// from the stream's show): it read "refine of #N" with no "(deleted)" and
// offered the Refine dialog on a source that was gone, until a reload. And
// every id it has seen moved to the recycle bin (`binnedIds`), which is gone
// as far as the gallery goes - a pass of one says "(in the recycle bin)"
// (`refine_binned`) - until a Restore takes it out of both (unforgetImages).
const deletedIds = new Set();
const binnedIds = new Set();
const isGone = (id) => deletedIds.has(id) || binnedIds.has(id);
function markGoneSource(rec) {
  if (rec.refine != null && isGone(rec.refine)) {
    rec.refine_missing = true;
    rec.refine_binned = binnedIds.has(rec.refine);
  }
  // An instruction edit's source likewise (`edit_missing`, `edit_binned`).
  if (rec.edit?.source != null && isGone(rec.edit.source)) {
    rec.edit_missing = true;
    rec.edit_binned = binnedIds.has(rec.edit.source);
  }
  return rec;
}

// Every trace of these ids on the page, once the server has let them go: the
// cards, the sheet cells, the viewer tiles, the panel's references and the
// records behind them. Shared by the card's delete, the lightbox's, Delete
// unstarred, the recycle bin's deletes and a tile whose picture has gone. A
// card and a viewer tile keep their place as a tombstone: removed, the next
// card slid under the pointer that pressed ⌦ - its own ⌦ lit - and an
// emptied viewer shrank the page by half a screen. `binned` is a move to the
// recycle bin rather than off the disk, which the tombstones and notes say.
// Answers a sentence for the caller's note when references went with them,
// else ''.
function forgetImages(ids, { binned = false } = {}) {
  const gone = new Set(ids);
  for (const id of gone) {
    (binned ? binnedIds : deletedIds).add(id);
    (binned ? deletedIds : binnedIds).delete(id);
  }
  // Out of the lightbox's CFG ladders, and a CFG comparison with one of them
  // ends, saying so (lightbox-cfg.js) - before the strips close up under it.
  forgetLadderImages(gone);
  // The Models picker's strip asks again for what is left (model-shots.js).
  forgetModelShots(gone);
  const focused = document.activeElement;
  // Where the sheet's focus was, before its cells go below: the redraw puts
  // it back on the cell now in that place (sheet-render.js, renderSheet).
  const sheetPlace = sheetMode ? sheetFocusPlace() : null;
  const current = currentCard();
  let lost = null;
  document.querySelectorAll('[data-id]').forEach((el) => {
    if (!gone.has(+el.dataset.id)) return;
    if (!el.classList.contains('card') && el.parentElement !== els.viewer) { el.remove(); return; }
    const held = el.contains(focused);
    const kept = tombstone(el, binned ? 'In the recycle bin' : 'Deleted');
    if (held) lost = kept;
  });
  // Gone for good out of the recycle bin (Empty, Delete permanently, another
  // tab): the tombstones and the viewer's held line that said "in the
  // recycle bin" say "Deleted" now, and a Restore has nothing to put back.
  if (!binned) {
    for (const id of gone) binnedRecs.delete(id);
    forgetInpaintImages(gone);
    forgetOutpaintImages(gone);
    document.querySelectorAll('[data-tomb-id]').forEach((el) => {
      const tomb = el.matches('.tomb') ? el : el.querySelector('.tomb');
      if (tomb && gone.has(+el.dataset.tombId) && tomb.textContent.startsWith('In the recycle bin')) {
        tomb.textContent = `Deleted · ${imageNames([+el.dataset.tombId])}`;
      }
    });
    const held = els.viewer.querySelector(':scope > .placeholder.held[data-binned-ids]');
    if (held && held.dataset.binnedIds.split(',').every((id) => deletedIds.has(+id))) {
      held.textContent = held.textContent.replace('moved to the recycle bin', 'deleted');
      delete held.dataset.binnedIds;
    }
  }
  // Only the grid's counters mirror the server's: the viewer holds whatever
  // this session made, and the sheet lays itself out from the grid's records.
  const inGallery = galleryRecords.filter((r) => gone.has(r.id)).length;
  const fromViewer = viewerRecords.filter((r) => gone.has(r.id)).map((r) => r.id);
  for (const list of new Set([galleryRecords, viewerRecords, sessionRecords, lbList])) {
    for (let at = list.length - 1; at >= 0; at--) if (gone.has(list[at].id)) list.splice(at, 1);
  }
  // The badge follows the server's count; the page anchor is an id, which a
  // delete does not move.
  if (inGallery) {
    total -= inGallery;
    els.count.textContent = total ? `(${total})` : '';
    syncRerender();
    syncDeleteUnstarred();
    syncGalleryEmpty();
  }
  // Nor is a deleted image picked any longer (gallery-select.js), and an
  // open Swap tags, Mass refine or More seeds that listed one checks again
  // (tag-swap.js, refine-batch.js, more-seeds.js).
  forgetSelected(gone);
  forgetSwapImages(gone);
  forgetRefineBatchImages(gone);
  forgetMoreSeedsImages(gone);
  // Keyboard focus was on what went: it moves on to the next image in the
  // same strip, or back to the one before, rather than dropping to <body>.
  // The keyboard's way into the grid goes to the same card.
  if (current?.classList.contains('gone')) makeCurrent(liveNeighbour(current));
  if (lost) {
    const strip = [...lost.parentElement.children]
      .filter((el) => !el.matches('.gone, .tomb, .placeholder, .live, .pending, .gallery-break'));
    const after = strip.find((el) => lost.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
    const next = after ?? strip.at(-1);
    const image = next?.matches('img') ? next : next?.querySelector('img');
    if (image) image.focus();
    // The viewer's last image, deleted by its own ⌦: nothing beside it to
    // move on to, so its tombstone holds the focus - or the line that takes
    // the strip's place below.
    else if (lost.parentElement === els.viewer) { lost.tabIndex = -1; lost.focus({ preventScroll: true }); }
  }
  // The note under the strip named what it held: the image just made's seed
  // is not a deleted image's to offer.
  if (fromViewer.length) noteViewer();
  // A strip with images left keeps its tombstones in their slots. One whose
  // last image has gone, with no job to fill it, says what it is for again -
  // in place of the tombstones, which beside it read as leftovers - at the
  // height the strip had, so the gallery below does not jump; the next run's
  // first image clears it, height and all.
  if (fromViewer.length && !viewerRecords.length && !busy) {
    const css = getComputedStyle(els.viewer);
    const placeholder = document.createElement('p');
    placeholder.className = 'placeholder held';
    placeholder.style.minHeight = `${els.viewer.clientHeight - parseFloat(css.paddingTop) - parseFloat(css.paddingBottom)}px`;
    const names = fromViewer.length > 3 ? `${fromViewer.length} images` : imageNames(fromViewer);
    placeholder.textContent = `${names} ${fromViewer.length === 1 ? 'was' : 'were'} `
      + `${binned ? 'moved to the recycle bin' : 'deleted'}. Generated images appear here.`;
    // Which ids it names, for the line to say "deleted" once they all are.
    if (binned) placeholder.dataset.binnedIds = fromViewer.join(',');
    const focusHere = els.viewer.contains(document.activeElement);
    els.viewer.replaceChildren(placeholder);
    els.viewer.classList.remove('multi', 'many');
    els.viewerNoteText.textContent = '';
    syncViewerMoreSeeds();
    // Saying what went, where the focus was rather than on <body>.
    if (focusHere) { placeholder.tabIndex = -1; placeholder.focus({ preventScroll: true }); }
  }
  // This page's earlier results (#recent), with what went taken out.
  renderRecent();
  for (const id of gone) if (sheetSources.has(id)) sheetSources.set(id, null);
  for (const id of gone) outsideView?.ids.delete(id);
  // A pass whose source went with them cannot be made again: its records say
  // so, as a reload's would (`refine_missing`), and so do every image's title
  // (thumb's, on a card, in the viewer, #recent and the sheet), the viewer's
  // note and column, every Reuse that would have opened the dialog on the
  // source, and the lightbox's facts and comparison - "(in the recycle bin)"
  // for a source binned (`refine_binned`), and again when it goes for good.
  const orphans = new Set([...galleryRecords, ...viewerRecords, ...sessionRecords, ...lbList]
    .filter((r) => r.refine != null && gone.has(r.refine) && (!r.refine_missing || !!r.refine_binned !== binned)));
  for (const r of orphans) {
    r.refine_missing = true;
    r.refine_binned = binned;
    document.querySelectorAll(`img[data-id="${r.id}"]`).forEach((img) => { titleImage(img, r); });
  }
  // An instruction edit whose source went says so the same way
  // (`edit_missing`, `edit_binned`), and its Reuse is held until a Restore.
  const edits = [...galleryRecords, ...viewerRecords, ...sessionRecords, ...lbList]
    .filter((r) => r.edit?.source != null && gone.has(r.edit.source) && (!r.edit_missing || !!r.edit_binned !== binned));
  for (const r of edits) {
    r.edit_missing = true;
    r.edit_binned = binned;
    orphans.add(r);
    document.querySelectorAll(`img[data-id="${r.id}"]`).forEach((img) => { titleImage(img, r); });
  }
  // An open Edit by instruction… on one of them asks its preview again.
  editSourcesMoved(gone, { deleted: !binned });
  if (orphans.size) {
    document.querySelectorAll('.reuse-settings').forEach((b) => {
      if (b.dataset.reuseEditSource && gone.has(+b.dataset.reuseEditSource)) {
        b.dataset.reuseEditGone = 'true';
        b.dataset.reuseEditBinned = String(binned);
      }
      if (!b.dataset.reusePass || !gone.has(+b.dataset.reusePass)) return;
      b.dataset.reusePassGone = 'true';
      b.dataset.reusePassBinned = String(binned);
    });
    syncReuseButtons();
    viewerSourcesGone();
    if (orphans.has(lbList[lbIndex])) lbDrawn = null;
  }
  for (const id of gone) if (lbSources.has(id)) lbSources.set(id, null);
  if (sheetMode) renderSheet(sheetPlace);
  invalidateFinishRuns();
  // A Reuse's Undo may not put a gone reference back; and one dropped from
  // what the Reuse left is no change that retires the Undo (`syncReuseUndo`,
  // which the drop below reaches), so both of its copies lose it first.
  if (reuseNote) {
    reuseNote.before.state.reference = reuseNote.before.state.reference.filter((id) => !gone.has(id));
    reuseNote.after.reference = reuseNote.after.reference.filter((id) => !gone.has(id));
  }
  if (setupsUndo) {
    setupsUndo.before.state.reference = setupsUndo.before.state.reference.filter((id) => !gone.has(id));
  }
  // A reference that has gone would be refused at submit, naming an id the
  // chips never showed.
  const dropped = referenceIds.filter((id) => gone.has(id));
  if (dropped.length) setReferences(referenceIds.filter((id) => !gone.has(id)));
  const note = !dropped.length ? ''
    : dropped.length === 1 ? ` ${imageNames(dropped)} was a reference image; it is no longer in the references.`
      : ` ${imageNames(dropped)} were reference images; they are no longer in the references.`;
  lightboxAfterGone();
  return note;
}

// The lightbox once images have left the strip it walks - deleted, binned,
// or restored out of the recycle bin's (bin.js).
function lightboxAfterGone() {
  if (!els.lightbox.open) return;
  // The strip has closed up under the viewer: the next image already sits at
  // this index - unless this was the last one loaded and the server has more,
  // when the next is on its next page, not the image before.
  if (lbIndex >= lbList.length && canPage()) pageInBehind();
  else if (!lbList.length) closeLightboxAfterDelete();
  else { lbIndex = Math.min(lbIndex, lbList.length - 1); renderLightbox(); }
}

// forgetImages' sibling, for images restored out of the recycle bin
// (bin.js's restoreImages): no longer gone, and a pass of one can be made
// again - its records, titles, every Reuse, the viewer's lineage and the
// lightbox's facts say so. A card or a viewer tile this page binned, still a
// tombstone in its own box, is drawn again there and nothing else moves; its
// record goes back into the grid's records where the grid is still the query
// it was binned from (`binnedRecs`), and into the viewer's and #recent's - a
// lightbox walking any of them keeps the image in view. The panel's
// references are not put back: only setReferences writes them.
function unforgetImages(ids) {
  const back = new Set(ids);
  if (!back.size) return;
  for (const id of back) { binnedIds.delete(id); deletedIds.delete(id); }
  const passes = new Set([...galleryRecords, ...viewerRecords, ...sessionRecords, ...lbList]
    .filter((r) => r.refine != null && back.has(r.refine) && r.refine_missing));
  for (const r of passes) {
    r.refine_missing = false;
    r.refine_binned = false;
    document.querySelectorAll(`img[data-id="${r.id}"]`).forEach((img) => { titleImage(img, r); });
  }
  // An instruction edit's source back: its facts, titles and Reuse say so.
  const edits = [...galleryRecords, ...viewerRecords, ...sessionRecords, ...lbList]
    .filter((r) => r.edit?.source != null && back.has(r.edit.source) && r.edit_missing);
  for (const r of edits) {
    r.edit_missing = false;
    r.edit_binned = false;
    document.querySelectorAll(`img[data-id="${r.id}"]`).forEach((img) => { titleImage(img, r); });
  }
  editSourcesMoved(back);
  document.querySelectorAll('.reuse-settings').forEach((b) => {
    if (b.dataset.reuseEditSource && back.has(+b.dataset.reuseEditSource)) {
      b.dataset.reuseEditGone = 'false';
      b.dataset.reuseEditBinned = 'false';
    }
    if (!b.dataset.reusePass || !back.has(+b.dataset.reusePass)) return;
    b.dataset.reusePassGone = 'false';
    b.dataset.reusePassBinned = 'false';
  });
  for (const id of back) { sheetSources.delete(id); lbSources.delete(id); }
  let toGallery = 0;
  for (const id of back) {
    const kept = binnedRecs.get(id);
    if (!kept) continue;
    binnedRecs.delete(id);
    // Into the grid's records in id order, newest first - the lightbox
    // walking them keeps its image - and its card into its tombstone.
    if (kept.gallery && kept.epoch === galleryEpoch && !binMode && !galleryRecords.some((r) => r.id === id)) {
      const found = galleryRecords.findIndex((r) => r.id < id);
      const at = found < 0 ? galleryRecords.length : found;
      galleryRecords.splice(at, 0, kept.gallery);
      if (lbList === galleryRecords && els.lightbox.open && at <= lbIndex) lbIndex += 1;
      toGallery += 1;
      const tomb = els.gallery.querySelector(`.card.gone[data-tomb-id="${id}"]`);
      if (tomb) tomb.replaceWith(card(kept.gallery));
    }
    // The viewer's tile, where its tombstone still holds the place: not where
    // the strip's placeholder has taken them all.
    const tomb = kept.viewer && els.viewer.querySelector(`:scope > .tomb[data-tomb-id="${id}"]`);
    if (tomb && !viewerRecords.some((r) => r.id === id)) {
      const figure = resultFigure(kept.viewer);
      tomb.replaceWith(figure);
      // Its index among the strip's records: the ones the strip has let go
      // of (VIEWER_MAX) and then the tiles before it.
      const hidden = viewerRecords.length - els.viewer.querySelectorAll(':scope > [data-id]').length + 1;
      const before = [...els.viewer.children].slice(0, [...els.viewer.children].indexOf(figure))
        .filter((el) => el.matches('[data-id]')).length;
      const at = Math.max(0, hidden) + before;
      viewerRecords.splice(at, 0, kept.viewer);
      // The lightbox walking the strip keeps its image, as on the grid's.
      if (lbList === viewerRecords && els.lightbox.open && at <= lbIndex) lbIndex += 1;
      if (viewerRecords.length === 1 && viewerJob.expected <= 1) figure.append(resultSide(kept.viewer, figure));
    }
    if (kept.session && !sessionRecords.some((r) => r.id === id)) {
      const shown = lbList === sessionRecords && els.lightbox.open ? lbList[lbIndex] : null;
      sessionRecords.push(kept.session);
      sessionRecords.sort((a, b) => b.id - a.id);
      if (shown) lbIndex = sessionRecords.indexOf(shown);
    }
  }
  if (toGallery) {
    total += toGallery;
    els.count.textContent = total ? `(${total})` : '';
    syncRerender();
    syncDeleteUnstarred();
    syncGalleryEmpty();
    keepCurrentCard();
    if (sheetMode) renderSheet();
  }
  syncReuseButtons();
  noteViewer();
  viewerSourcesGone();
  renderRecent();
  if (els.lightbox.open && lbList[lbIndex]) { lbDrawn = null; renderLightbox(); }
}

// The lightbox's only image went - a strip of one, a gallery or a search of
// one - and the dialog with it, its focused Delete too, which left focus on
// <body>: the viewer's held line or tombstone was not focused (focus had been
// in the lightbox, not the viewer), and closeLightbox restores nothing. It
// goes where the walk came from - the viewer's line, the grid's way in -
// else the search box, as dismissLightbox's last resort.
function closeLightboxAfterDelete() {
  const fromViewer = lbList === viewerRecords || lbOrigin === 'viewer' || lbOrigin === 'recent';
  closeLightbox();
  requestAnimationFrame(() => {
    const held = els.viewer.querySelector(':scope > .placeholder.held, :scope > .tomb');
    const card = currentCard()?.querySelector('img');
    const target = (fromViewer ? held ?? card : card ?? held) ?? els.search;
    if (target === held) target.tabIndex = -1;
    target.focus({ preventScroll: true });
  });
}

// A card or a viewer tile that has gone, kept as a box of its size that says
// so. Out of the id-tagged set and of the pointer's way: nothing acts on it.
// Named as the page names images, by gallery id: the cells of a sweep share a
// seed, and "Deleted · seed …" on each of them said nothing of which went.
// `word` is what happened to it: "Deleted", "In the recycle bin", or in the
// bin "Restored". The id it held stays as `data-tomb-id`, which nothing
// that acts on images reads: a Restore puts the card back in its own box
// (unforgetImages). Answers the element that holds the place now.
function tombstone(el, word = 'Deleted') {
  const tomb = document.createElement('div');
  tomb.className = 'tomb';
  const id = +el.dataset.id;
  tomb.textContent = `${word} · ${imageNames([id])}`;
  if (el.classList.contains('card')) {
    el.dataset.tombId = id;
    el.replaceChildren(tomb);
    // Nor any mark of what the image was: a star, a reference, a pass, a
    // pick - nor the group (card()), which has no tools left to name; its
    // text says it.
    el.classList.remove('fav', 'ref', 'refined', 'inpainted', 'outpainted', 'selected');
    el.classList.add('gone');
    el.removeAttribute('role');
    el.removeAttribute('aria-label');
    delete el.dataset.id;
    delete el.dataset.refN;
    return el;
  }
  const box = el.getBoundingClientRect();
  tomb.style.width = `${box.width}px`;
  tomb.style.height = `${box.height}px`;
  tomb.dataset.tombId = id;
  el.replaceWith(tomb);
  return tomb;
}

// The lightbox's last loaded image went while the server has more: the page
// after it is fetched and the lightbox steps onto its first image. Meanwhile
// the deleted image is off the screen, and with the index past the strip's
// end its buttons act on nothing (each asks `lbList[lbIndex]` first) - not
// disabled, which would drop the focus of the one just pressed to <body>.
function pageInBehind() {
  els.lbImg.removeAttribute('src');
  els.lbImg.alt = '';
  els.lbPrompt.textContent = 'Loading next…';
  els.lbSettings.textContent = '';
  els.lbCount.textContent = '';
  loadGallery().catch(() => {}).then(() => {
    if (!els.lightbox.open) return;
    // A page that failed, or brought nothing, leaves the one before.
    lbIndex = Math.min(lbIndex, lbList.length - 1);
    if (lbIndex >= 0) renderLightbox();
    else closeLightbox();
  });
}

// A viewer tile whose picture would not load: deleted from another tab, or
// replayed by a job from before its delete - or the server is only out of
// reach. Asked, and only "unknown image" forgets it - as binned where the
// server says it is in the recycle bin.
async function forgetIfGone(id) {
  try {
    await api(`/api/images/${id}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) return;
    const note = forgetImages([id], { binned: /recycle bin/.test(err.message) });
    if (note) showError(note.trim(), true, { slot: els.galleryError });
  }
}

// The inverse of the card's delete, for the night that made hundreds and
// earned a few stars: star those, then move everything else the filter
// matches to the recycle bin in one gesture. Counted server-side, as Re-render's selection is -
// the browser holds one page - and gated on typing that count back: the
// hazard is no longer "the wrong image" but "the wrong scope", a run not yet
// looked at, and the number is what tells them apart. The delete is pinned
// to it, so images a running job lands meanwhile are refused, not swept.
// A starred image keeps what it was made from - a pass its source, a render
// its references - and the dialog names those. Everything it says, it says
// under the gallery's head, where the button is.
async function deleteUnstarred() {
  const slot = { slot: els.galleryError };
  const held = binBroadWriteReason();
  if (held) { showError(held + '.', true, slot); return; }
  const owner = {};
  binBroadOwner = owner;
  try {
    syncBinWriteHolds();
    const revision = modelRevision;
    const allModels = els.allModels.checked;
    // The grid brought up to what the sweep would reach first: images another
    // tab made since this page's last look were counted and deleted unseen -
    // the newest ones, the likeliest still to be starred - and the count to
    // type back was not the grid's. Under the same filters, as a finished job's
    // are (prependFinished); not on a sheet, which is one run laid out whole.
    const head = galleryRecords[0]?.id;
    if (!sheetMode && head != null) await prependFinished(0, { now: true });
    const search = els.search.value.trim();
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (allModels) params.set('all_models', 'true');
    // How many of the count are newer than the grid's head was is the
    // server's to say: it keeps a starred image's sources the page cannot see.
    const ask = new URLSearchParams(params);
    if (!sheetMode && head != null) ask.set('newer_than', head);
    let found;
    try {
      found = await api(`/api/gallery/unstarred?${ask}`);
    } catch (err) {
      showError(err.message, false, slot);
      keepUnstarredFocus();
      return;
    }
    // The checkpoint scope is part of what is confirmed: with All models off
    // the sweep stops at this checkpoint's images, and the dialog says so by
    // the server's own name for it. With it on, the wider sweep names every
    // checkpoint it reaches (the server lists six at most).
    const named = (id) => (id || '').split('/').pop();
    let scope = search ? `matching "${search}"` : 'in the gallery';
    if (!allModels) {
      const only = named(found.models[0] ?? loadedModel);
      if (only) scope += ` (${only} only)`;
    } else if (found.models.length) {
      const many = found.models.length >= 6;
      const count = many ? '6 or more' : found.models.length;
      scope += ` across ${count} checkpoint${found.models.length === 1 ? '' : 's'} `
        + `(${found.models.map(named).join(', ')}${many ? ', …' : ''})`;
    }
    const sources = found.sources_kept ?? 0;
    if (!found.unstarred) {
      // None left at all - another tab deleted them since this page looked -
      // is not "every image is starred".
      const kept = sources ? ' or one a starred image was made from' : '';
      showError(found.records ? `Nothing to delete: every image ${scope} is starred${kept}.`
        : `Nothing to delete: no image is left ${scope}.`, true, slot);
      keepUnstarredFocus();
      return;
    }
    const plural = found.unstarred === 1 ? '' : 's';
    const sourceNames = found.sources.map((id) => `#${id}`).join(', ') + (sources > found.sources.length ? ', …' : '');
    const keeping = `keeping ${found.favorites} starred`
      + (sources === 1 ? ` and ${sourceNames}, which a starred image was made from`
        : sources > 1 ? ` and ${sources} images starred ones were made from (${sourceNames})` : '');
    // How much of the gallery's history goes: the sample prompts are the first
    // three alphabetically and say nothing of how many runs that is, or when.
    const from = found.first ? when(Date.parse(found.first)) : '';
    const to = found.last ? when(Date.parse(found.last)) : '';
    const span = [
      found.runs ? `From ${found.runs} run${found.runs === 1 ? '' : 's'}` : '',
      from && (from === to ? `made ${from}` : `made ${from} to ${to}`),
    ].filter(Boolean).join(', ');
    const prompts = found.prompts.slice(0, 3).map((p) => `  "${p.length > 60 ? `${p.slice(0, 57)}…` : p}"`);
    if (found.prompts.length > 3) prompts.push('  …');
    const message = [
      `Delete ${found.unstarred} unstarred image${plural} ${scope}, ${keeping}?`,
      ...(span ? [`${span}.`] : []),
      ...freshLine(head, found.newer),
      '',
      ...prompts,
      '',
      // The count is pinned: an image a running job lands in scope before the
      // answer refuses the whole sweep, and that is better known before typing.
      // The server's answer, since the job may be another tab's.
      ...(found.job_active ? ['A job is still running; if it adds an image here before you confirm, nothing is deleted.'] : []),
      `They move to the recycle bin; Empty recycle bin removes them from disk. Type ${found.unstarred} to continue:`,
    ].join('\n');
    const typed = await askCount(message);
    if (typed === null) return;
    if (revision !== modelRevision) {
      showError('Nothing moved: the checkpoint changed. Review the new gallery before deleting.', true, slot);
      return;
    }
    if (typed.trim() !== String(found.unstarred)) {
      showError(`Nothing moved: "${typed.trim()}" is not ${found.unstarred}.`, true, slot);
      return;
    }
    if (binBroadOwner !== owner) {
      showError('Nothing moved: the gallery action changed. Review the gallery before deleting.', true, slot);
      return;
    }
    params.set('expected', found.unstarred);
    let result;
    try {
      result = await api(`/api/gallery/unstarred?${params}`, {
        method: 'DELETE',
        headers: revision === null ? {} : { 'X-AuraDream-Model-Revision': String(revision) },
      });
    } catch (err) {
      if (!/selection changed/i.test(err.message)) { showError(err.message, false, slot); return; }
      // The server counts something else now - images a job landed in scope,
      // most often. The page shows the server's count rather than the one the
      // message has just called stale. Whether a job is still running is in
      // the server's sentence: it may be another tab's.
      await resetGallery();
      keepUnstarredFocus();
      loadRuns();
      showError(err.message, false, slot);
      return;
    }
    // Into the recycle bin; `deleted` is the same ids, from a server before it.
    const moved = result.binned ?? result.deleted;
    const also = forgetImages(moved, { binned: true });
    // The page is now mostly holes; ask for it again from the top, which also
    // brings the badge back to the server's count - and clears the slot, so
    // the outcome is written after it.
    await resetGallery();
    keepUnstarredFocus();
    loadRuns();
    const went = moved.length;
    showError(`Moved ${went} image${went === 1 ? '' : 's'} to the recycle bin, ${keeping}.${also}`, true, slot);
    void syncBin();
  } finally {
    if (binBroadOwner === owner) binBroadOwner = null;
    syncBinWriteHolds();
  }
}

// A reload that leaves nothing in scope hides the button (syncDeleteUnstarred)
// and, where focus was on it, dropped it to <body> - a sweep that took every
// image, or one another tab had already emptied. The search box instead,
// which says what the scope was, as the other delete paths keep focus off
// <body>.
function keepUnstarredFocus() {
  const at = document.activeElement;
  if (!els.deleteUnstarred.hidden || (at && at !== document.body && at !== els.deleteUnstarred)) return;
  els.search.focus({ preventScroll: true });
}

// The dialog's line for what the grid has just been brought up to: `n`, the
// server's count of the images the sweep takes that are newer than the
// grid's head before the look (`head`, sent as `newer_than`). A page that
// did not join up was loaded again from the top (prependFinished), and where
// every card on it is newer and there is more, only the newest are on it.
function freshLine(head, n) {
  if (!n) return [];
  // At the top of the grid: a page scrolled down keeps its view
  // (holdGalleryView), so they may be above it rather than in sight.
  const some = n > 1 && galleryMore && galleryRecords.at(-1)?.id > head;
  return [`${n} of them ${n === 1 ? 'was' : 'were'} made since this page last loaded; `
    + `${some ? 'the newest are' : n === 1 ? 'it is' : 'they are'} at the top of the grid now.`];
}

// The head's buttons that come and go with the filter - Re-render, Delete
// unstarred - stand after the search, Starred and All models, which the free
// space before the search keeps to the row's right: one going moved every
// control before it along the row, Starred 137 px at 1400, and a second
// click on Starred landed in the search box. While the pointer rests on the
// head's controls (`headPointer`, from the head's pointerover and leave in
// app.js, as `galleryPointer` is, never `:hover`) one that goes keeps its
// place, empty (`.kept`: hidden, its box kept), and one that comes waits;
// both are put right once the pointer leaves them (releaseHead). The footer
// row under them is not theirs: Bin there changes the head as a whole, and
// its way in and out of the bin is `now`. All models' count of the filter's
// matches elsewhere, " (+3)", is that tick's width, before them too: a
// search that left none moved the search box and Starred 24 px, so its
// change waits with them (`headOthers`). What still changes there - the
// gallery's count, Re-render's label (syncRerender) - must not break the
// row either: with both buttons shown the controls fill it at 1400, and
// "(3)" to "(24)" wrapped Delete unstarred onto a row of its own and slid
// Starred 130 px. So while the pointer rests on a head whose controls are
// one row, the search gives way to them (`.held`: its basis and floor
// gone, so the row it is on never breaks), its right edge where it was.
let headPointer = false;
let headOthers = null;
function pointHead(on) {
  if (on && !headPointer) {
    headPointer = true;
    const head = els.search.closest('.gallery-head');
    // The flex rows' items: not the footer's row, nor #star-said, out of them.
    const mids = [...head.children].filter((el) => !el.classList.contains('gallery-footer')
        && el.getClientRects().length && getComputedStyle(el).position !== 'absolute')
      .map((el) => { const r = el.getBoundingClientRect(); return r.top + r.height / 2; });
    head.classList.toggle('held', Math.max(...mids) - Math.min(...mids) < 1);
  } else if (!on && headPointer) releaseHead();
}

function showInHead(el, show, now = false) {
  if (now || !headPointer) {
    el.hidden = !show;
    el.classList.remove('kept');
  } else if (!show && !el.hidden) {
    el.hidden = true;
    el.classList.add('kept');
  } else if (show && el.classList.contains('kept')) {
    // Back in its own place.
    el.hidden = false;
    el.classList.remove('kept');
  }
}

function showOtherModels(n) {
  const text = n ? ` (+${n})` : '';
  headOthers = headPointer && text !== els.otherModels.textContent ? text : null;
  if (headOthers === null) els.otherModels.textContent = text;
}

function releaseHead() {
  headPointer = false;
  els.search.closest('.gallery-head').classList.remove('held');
  if (headOthers !== null) els.otherModels.textContent = headOthers;
  headOthers = null;
  syncRerender();
  syncDeleteUnstarred();
}

// Shown while there is something it could act on: the filter matches images
// and the Starred tick is off, since with it on every match is a keeper.
// Whether the search box reads as a strict scope is the server's call, at
// the click. Not in the recycle bin, where Empty recycle bin has its place.
// While cards are picked it may give its place to Re-render of them
// (`unstarredGivesWay`, syncHeadForPicks).
function syncDeleteUnstarred(now = false) {
  const givesWay = unstarredGivesWay && (selectedIds().length > 0 || unstarredHeld === viewFilter());
  showInHead(els.deleteUnstarred, !(binMode || els.favOnly.checked || total <= 0 || givesWay), now);
  syncDeleteUnstarredHold();
}

// Ownership affects availability only: do not remeasure or move the head
// merely because a lifecycle write starts or settles.
function syncDeleteUnstarredHold() {
  els.deleteUnstarred.dataset.binBroadTitle ??= els.deleteUnstarred.title;
  const held = binBroadWriteReason();
  els.deleteUnstarred.setAttribute('aria-disabled', String(!!held));
  els.deleteUnstarred.title = held || els.deleteUnstarred.dataset.binBroadTitle;
}

// A change of the picks (onSelectionChange), with the pointer most often on
// the card just picked: Re-render takes the picks, shown whatever the filter
// and its label longer ("Re-render 8 selected at…"), and the head's controls
// keep the rows they had. At 1400 px Re-render and Delete unstarred side by
// side take two rows where either alone takes one: a first pick that brought
// Re-render in beside it wrapped the row and moved the grid 38 px under the
// pointer, and on a filter that showed both on one row the longer label
// wrapped it alike. So Delete unstarred stays where the head keeps its
// height with it, and gives Re-render its place where it does not - measured
// within the one task, so nothing is painted between: what fits is the
// widths' (the labels, the search, the window, Gallery beside), as `.held`'s
// one row is (pointHead). Where neither keeps it, Delete unstarred stays as
// it was. Only for a change of the picks: a filter typed, a tick or a job
// landing lays the head out as it always has, and the sweep is back the
// moment nothing is picked - where that keeps the rows. A filter typed while
// cards were picked keeps Delete unstarred given way, and its own Re-render
// beside it took two rows at 1400 where the picks' one took one: Clear then
// moved Select and the grid 38 px under the pointer on it. So emptying the
// picks is measured too, and where bringing Delete unstarred back would
// change the head's height it stays given way (`unstarredHeld`, the filter it
// was held on) until the pointer leaves the gallery - head, footer and grid,
// `libraryPointer` from #library's enter and leave in app.js, not `:hover` -
// or the filter changes. With the pointer elsewhere it comes back at once.
let unstarredGivesWay = false;
let unstarredHeld = null;
let libraryPointer = false;
let headPicked = false;
function syncHeadForPicks() {
  const picked = selectedIds().length > 0;
  const first = picked && !headPicked;
  const wasGiven = unstarredGivesWay;
  headPicked = picked;
  if (!picked || first) unstarredGivesWay = false;
  unstarredHeld = null;
  const head = els.search.closest('.gallery-head');
  const before = head.getBoundingClientRect().height;
  syncRerender();
  syncDeleteUnstarred();
  if (head.getBoundingClientRect().height === before) return;
  if (!picked) {
    if (!wasGiven || !libraryPointer) return;
    unstarredGivesWay = true;
    unstarredHeld = viewFilter();
    syncDeleteUnstarred();
    if (head.getBoundingClientRect().height === before) return;
    unstarredGivesWay = false;
    unstarredHeld = null;
    syncDeleteUnstarred();
    return;
  }
  unstarredGivesWay = !unstarredGivesWay;
  syncDeleteUnstarred();
  if (head.getBoundingClientRect().height === before) return;
  unstarredGivesWay = !unstarredGivesWay;
  syncDeleteUnstarred();
}

// The pointer has left the gallery: Delete unstarred, held given way after
// the picks were emptied, comes back - unless cards were picked again
// meanwhile, which measured the head afresh.
function libraryPointerLeft() {
  libraryPointer = false;
  if (unstarredHeld === null) return;
  unstarredHeld = null;
  unstarredGivesWay = false;
  syncDeleteUnstarred();
}
