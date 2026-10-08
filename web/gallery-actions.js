// Local Diffusion's page script: starring, deleting and forgetting gallery images.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The same image can be on the page more than once - in the viewer, in the
// gallery grid, on the sheet, in the lightbox - as separate record objects
// fetched at different times. So both actions apply by id: to every record
// that carries it, and to every element tagged with it.
function recordsWithId(id) {
  return [...galleryRecords, ...viewerRecords, ...sessionRecords, ...lbList].filter((r) => r.id === id);
}

// `say` is F's: the key stars an image from wherever focus is in its card,
// sheet cell or lightbox, and only a press on the star itself changes the
// state of what has focus - so otherwise the outcome is said (sayStar).
async function setFavorite(rec, value, { say = false } = {}) {
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
  recordsWithId(rec.id).forEach((r) => { r.favorite = value; });
  // Every element showing the image - a card, a sheet cell, a result beside
  // the viewer - by class and by its own star button, never by re-rendering:
  // the sheet rebuilds up to a thousand <img> nodes and resets the scroll per
  // keypress.
  const shown = `.card[data-id="${rec.id}"], .sheet-cell[data-id="${rec.id}"], .result[data-id="${rec.id}"]`;
  document.querySelectorAll(shown).forEach((el) => {
    el.classList.toggle('fav', value);
    el.querySelectorAll('.star').forEach((star) => syncStar(star, value));
  });
  if (els.lightbox.open && lbList[lbIndex]?.id === rec.id) renderLightbox();
  // A star with focus says its own new state (aria-pressed); anywhere else
  // F changed a button nobody was on, and nothing was heard.
  if (say && !document.activeElement?.matches?.('.star, #lb-fav')) sayStar(rec, value);
}

// What F did, for a screen reader: "Starred #12", "Unstarred #12", in a
// visually hidden polite line - the lightbox's own while it is up, since the
// page behind a modal dialog is inert, live regions and all. Emptied first
// and written on the next frame, so the same words again are heard again,
// and emptied after a while, so browse mode does not come on a stale one.
let starSaidTimer = 0;
function sayStar(rec, value) {
  const line = els.lightbox.open ? els.lbStarSaid : els.starSaid;
  const text = `${value ? 'Starred' : 'Unstarred'} #${rec.id}`;
  els.starSaid.textContent = '';
  els.lbStarSaid.textContent = '';
  clearTimeout(starSaidTimer);
  requestAnimationFrame(() => { line.textContent = text; });
  starSaidTimer = setTimeout(() => { if (line.textContent === text) line.textContent = ''; }, 5000);
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
// press outside it; countDialog the typed text, or null. The smokes stand in
// for them through `askConfirm` and `askCount`, and put these back.
let askAnswer = null;
function openAsk(message, { count = false } = {}) {
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
  const anchor = document.activeElement !== document.body ? document.activeElement : null;
  return new Promise((resolve) => {
    askAnswer = resolve;
    els.confirmDialog.returnValue = '';
    els.confirmDialog.showModal();
    placeAsk(anchor);
    (count ? els.confirmCount : els.confirmCancel).focus();
  });
}

function confirmDialog(message) {
  return openAsk(message).then((answer) => answer === 'ok');
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

// The delete removes the PNG as well as the row, and on a card the button sits
// a few pixels from the star. Say which image, so a slip is caught here: the
// cells of a sweep share a prompt and a seed, and a pass shares both with its
// source, so the settings line is what tells them apart. And say what else it
// costs - `passes` is the gallery's answer to `refine:<id>`, the passes made
// from this image, which could not be made again without it.
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
    `Delete #${rec.id}${rec.favorite ? ' (starred)' : ''}?`,
    '',
    settingsLine(rec),
    `"${snippet}"`,
    ...(costs.length ? ['', ...costs] : []),
    '',
    'The PNG is deleted from disk, not moved to a recycle bin.',
  ].join('\n'));
}

// The ids whose delete is under way, from the press to the server's answer.
// The confirm waits on the question about passes now, and a second press in
// that time - a double-click on ⌦, the lightbox's Delete held down - asked
// about the same image twice and sent two deletes.
const deleting = new Set();

async function deleteImage(rec) {
  if (deleting.has(rec.id)) return;
  deleting.add(rec.id);
  try {
    await deleteOne(rec);
  } finally {
    deleting.delete(rec.id);
  }
}

async function deleteOne(rec) {
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
    // Gone already - from another tab, or a record a reattached job had
    // replayed: what the delete was for has happened, so it is a note.
    already = true;
  }
  const also = forgetImages([rec.id]);
  const said = already ? `#${rec.id} was already deleted.` : `Deleted #${rec.id} (seed ${rec.seed}).`;
  // A card's tombstone says it in place. The lightbox has moved on to the
  // next image, so it says which one went, until the next step.
  if (fromLightbox && els.lightbox.open) noteLightboxDelete(said + also);
  else if (already || also) showError((already ? said + also : also).trim(), true, { slot: els.galleryError });
  // The Runs picker counts images; it is the gallery's table of contents.
  loadRuns();
}

// After a delete in the lightbox the next image is in view at the same zoom,
// and on a sweep's near-identical cells nothing showed that anything had
// happened. Said in the gallery's slot, which mirrors into the lightbox's
// role=alert line, and taken back at the next key or press in the lightbox.
let lightboxDeleteNote = null;
function noteLightboxDelete(text) {
  showError(text, true, { slot: els.galleryError });
  // Each copy only while it still says this: a job ending meanwhile resets
  // the gallery's line and leaves the lightbox's. The key and the press are
  // one listener pair, both taken off by whichever comes first - and by the
  // next delete's note - so a run of deletes leaves none behind.
  lightboxDeleteNote?.abort();
  const note = new AbortController();
  lightboxDeleteNote = note;
  const clear = () => {
    note.abort();
    if (lightboxDeleteNote === note) lightboxDeleteNote = null;
    if (els.galleryError.textContent === text) showError('', false, { slot: els.galleryError });
    else if (els.lbError.textContent === text) { els.lbError.hidden = true; els.lbError.textContent = ''; }
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
// offered the Refine dialog on a source that was gone, until a reload.
const deletedIds = new Set();
function markGoneSource(rec) {
  if (rec.refine != null && deletedIds.has(rec.refine)) rec.refine_missing = true;
  return rec;
}

// Every trace of these ids on the page, once the server has let them go: the
// cards, the sheet cells, the viewer tiles, the panel's references and the
// records behind them. Shared by the card's delete, the lightbox's, Delete
// unstarred and a tile whose picture has gone. A card and a viewer tile keep
// their place as a tombstone: removed, the next card slid under the pointer
// that pressed ⌦ - its own ⌦ lit - and an emptied viewer shrank the page by
// half a screen. Answers a sentence for the caller's note when references
// went with them, else ''.
function forgetImages(ids) {
  const gone = new Set(ids);
  for (const id of gone) deletedIds.add(id);
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
    const kept = tombstone(el);
    if (held) lost = kept;
  });
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
    placeholder.textContent = `${names} ${fromViewer.length === 1 ? 'was' : 'were'} deleted. Generated images appear here.`;
    const focusHere = els.viewer.contains(document.activeElement);
    els.viewer.replaceChildren(placeholder);
    els.viewer.classList.remove('multi', 'many');
    els.viewerNote.textContent = '';
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
  // source, and the lightbox's facts and comparison.
  const orphans = new Set([...galleryRecords, ...viewerRecords, ...sessionRecords, ...lbList]
    .filter((r) => r.refine != null && gone.has(r.refine) && !r.refine_missing));
  for (const r of orphans) {
    r.refine_missing = true;
    document.querySelectorAll(`img[data-id="${r.id}"]`).forEach((img) => { titleImage(img, r); });
  }
  if (orphans.size) {
    document.querySelectorAll('.reuse-settings').forEach((b) => {
      if (b.dataset.reusePass && gone.has(+b.dataset.reusePass)) b.dataset.reusePassGone = 'true';
    });
    syncReuseButtons();
    viewerSourcesGone();
    if (orphans.has(lbList[lbIndex])) lbDrawn = null;
  }
  for (const id of gone) if (lbSources.has(id)) lbSources.set(id, null);
  if (sheetMode) renderSheet(sheetPlace);
  // A Reuse's Undo may not put a gone reference back; and one dropped from
  // what the Reuse left is no change that retires the Undo (`syncReuseUndo`,
  // which the drop below reaches), so both of its copies lose it first.
  if (reuseNote) {
    reuseNote.before.state.reference = reuseNote.before.state.reference.filter((id) => !gone.has(id));
    reuseNote.after.reference = reuseNote.after.reference.filter((id) => !gone.has(id));
  }
  // A reference that has gone would be refused at submit, naming an id the
  // chips never showed.
  const dropped = referenceIds.filter((id) => gone.has(id));
  if (dropped.length) setReferences(referenceIds.filter((id) => !gone.has(id)));
  const note = !dropped.length ? ''
    : dropped.length === 1 ? ` ${imageNames(dropped)} was a reference image; it is no longer in the references.`
      : ` ${imageNames(dropped)} were reference images; they are no longer in the references.`;
  if (!els.lightbox.open) return note;
  // The strip has closed up under the viewer: the next image already sits at
  // this index - unless this was the last one loaded and the server has more,
  // when the next is on its next page, not the image before.
  if (lbIndex >= lbList.length && canPage()) pageInBehind();
  else if (!lbList.length) closeLightboxAfterDelete();
  else { lbIndex = Math.min(lbIndex, lbList.length - 1); renderLightbox(); }
  return note;
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
// Answers the element that holds the place now.
function tombstone(el) {
  const tomb = document.createElement('div');
  tomb.className = 'tomb';
  tomb.textContent = `Deleted · ${imageNames([+el.dataset.id])}`;
  if (el.classList.contains('card')) {
    el.replaceChildren(tomb);
    // Nor any mark of what the image was: a star, a reference, a pass - nor
    // the group (card()), which has no tools left to name; its text says it.
    el.classList.remove('fav', 'ref', 'refined');
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
// reach. Asked, and only "unknown image" forgets it.
async function forgetIfGone(id) {
  try {
    await api(`/api/images/${id}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) return;
    const note = forgetImages([id]);
    if (note) showError(note.trim(), true, { slot: els.galleryError });
  }
}

// The inverse of the card's delete, for the night that made hundreds and
// earned a few stars: star those, then remove everything else the filter
// matches in one gesture. Counted server-side, as Re-render's selection is -
// the browser holds one page - and gated on typing that count back: the
// hazard is no longer "the wrong image" but "the wrong scope", a run not yet
// looked at, and the number is what tells them apart. The delete is pinned
// to it, so images a running job lands meanwhile are refused, not swept.
// A starred image keeps what it was made from - a pass its source, a render
// its references - and the dialog names those. Everything it says, it says
// under the gallery's head, where the button is.
async function deleteUnstarred() {
  const slot = { slot: els.galleryError };
  const revision = modelRevision;
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
  if (els.allModels.checked) params.set('all_models', 'true');
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
  if (!els.allModels.checked) {
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
    `The files are removed from disk. Type ${found.unstarred} to continue:`,
  ].join('\n');
  const typed = await askCount(message);
  if (typed === null) return;
  if (revision !== modelRevision) {
    showError('Nothing deleted: the checkpoint changed. Review the new gallery before deleting.', true, slot);
    return;
  }
  if (typed.trim() !== String(found.unstarred)) {
    showError(`Nothing deleted: "${typed.trim()}" is not ${found.unstarred}.`, true, slot);
    return;
  }
  params.set('expected', found.unstarred);
  let result;
  try {
    result = await api(`/api/gallery/unstarred?${params}`, {
      method: 'DELETE',
      headers: revision === null ? {} : { 'X-Local-Diffusion-Model-Revision': String(revision) },
    });
  } catch (err) {
    if (!/selection changed/i.test(err.message)) { showError(err.message, false, slot); return; }
    // The server counts something else now - images a job landed in scope,
    // most often. The page shows the server's count rather than the one the
    // message has just called stale. Whether a job is still running is in
    // the server's sentence: it may be another tab's.
    void resetGallery().then(keepUnstarredFocus);
    loadRuns();
    showError(err.message, false, slot);
    return;
  }
  const also = forgetImages(result.deleted);
  // The page is now mostly holes; ask for it again from the top, which also
  // brings the badge back to the server's count - and clears the slot, so
  // the outcome is written after it.
  void resetGallery().then(keepUnstarredFocus);
  loadRuns();
  const went = result.deleted.length;
  showError(`Deleted ${went} image${went === 1 ? '' : 's'}, ${keeping}.${also}`, true, slot);
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

// Shown while there is something it could act on: the filter matches images
// and the Starred tick is off, since with it on every match is a keeper.
// Whether the search box reads as a strict scope is the server's call, at
// the click.
function syncDeleteUnstarred() {
  els.deleteUnstarred.hidden = els.favOnly.checked || total <= 0;
}
