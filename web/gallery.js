// Local Diffusion's page script: the gallery grid, its cards and Reuse. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// -------------------------------------------------------------------gallery

function resetGallery() {
  // Anything already in flight was fetched under the old filters and against
  // the grid this is about to empty; the token tells it to drop itself.
  galleryEpoch += 1;
  galleryMore = false;
  galleryRecords = [];
  // The keyboard's card is kept where it is still loaded, but as the grid's
  // default, not a card chosen in this one (prependFinished).
  galleryChosen = false;
  els.gallery.innerHTML = '';
  els.galleryEmpty.hidden = true;
  // The count of a finished job's images outside the view is the filter's
  // (noteOutsideView): another filter starts it again.
  if (outsideView && outsideView.filter !== viewFilter()) outsideView = null;
  // What the gallery said was about the grid this empties. Not the lightbox's
  // line: a job ending under an open lightbox resets the gallery too.
  showError('', false, { slot: els.galleryError, mirror: false });
  // Every new query comes through here, so the Runs picker follows the box.
  syncRunList();
  // The fresh query has to be issued *after* an in-flight page settles rather
  // than sharing its promise: that page carries the old epoch and drops
  // itself, so sharing it left the gallery empty with nothing outstanding and
  // nothing to retry. Its `.finally` nulls `galleryPending` first, so the
  // chained call finds a free slot.
  const pending = galleryPending;
  return pending ? pending.then(loadGallery) : loadGallery();
}

function loadGallery() {
  // One page at a time. Nothing else serialises this - a double-clicked "Load
  // more", or an arrow key held down in the lightbox - and two calls reading
  // the same anchor append the same 48 cards twice.
  if (galleryPending) return galleryPending;
  const epoch = galleryEpoch;
  const params = new URLSearchParams({ limit: sheetMode ? sheetPage : PAGE });
  // The sheet asks for the whole run at once, so only the grid ever pages.
  const last = galleryRecords[galleryRecords.length - 1];
  if (!sheetMode && last) params.set('before', last.id);
  if (els.search.value.trim()) params.set('search', els.search.value.trim());
  if (els.favOnly.checked) params.set('favorites', 'true');
  if (els.allModels.checked) params.set('all_models', 'true');
  els.more.disabled = true;
  els.more.textContent = 'Loading…';
  els.lbNext.disabled = true;

  galleryPending = api(`/api/gallery?${params}`)
    .then((page) => {
      if (epoch !== galleryEpoch) return;  // superseded by a reset
      total = page.total;
      els.count.textContent = total ? `(${total})` : '';
      // What the tick would add: the same filter's matches on other
      // checkpoints, so a run made on the other model reads as "+12" on the
      // tick rather than as an empty grid. 0 once it is on.
      els.otherModels.textContent = page.other_models ? ` (+${page.other_models})` : '';
      galleryOthers = page.other_models || 0;
      syncEmptyInstall(page);
      syncRerender();
      syncDeleteUnstarred();
      // Keyset paging cannot repeat a row; the guard stays for a stream that
      // reattaches and a page that overlap through the viewer's records.
      const have = new Set(galleryRecords.map((r) => r.id));
      page.items.forEach((rec) => {
        if (have.has(rec.id)) return;
        galleryRecords.push(rec);
        if (sheetMode) return;
        // The day's line where it changes from the card before - a tombstone
        // keeps its day - and over the first card where that is not today's.
        const before = els.gallery.lastElementChild;
        if (before ? cardDay(before) !== localDay(rec) : !isToday(rec)) els.gallery.appendChild(dayBreak(rec));
        els.gallery.appendChild(card(rec));
      });
      if (!sheetMode) keepCurrentCard();
      syncGalleryEmpty();
      galleryMore = !!page.has_more;
      els.more.hidden = sheetMode || !galleryMore;
      watchMore();
      // Under a filter the sheet waits for its run's order - learnSheetOrder.
      if (sheetMode) return learnSheetOrder(epoch).then(() => { if (epoch === galleryEpoch) renderSheet(); });
    })
    .catch((err) => {
      // A filter term that is not a number, most often: report it rather than
      // leaving the gallery silently showing the last good page - and where
      // the empty grid is, without the count of the filter before it.
      if (epoch !== galleryEpoch) return;
      // The button is the retry now: a trigger made again (armMore) waits
      // for a page that lands rather than fetching from this server again.
      moreArmed = false;
      showError(err.message, false, { slot: els.galleryError });
      if (!galleryRecords.length) els.count.textContent = els.otherModels.textContent = '';
    })
    .finally(() => {
      galleryPending = null;
      els.more.disabled = false;
      els.more.textContent = 'Load more';
      // Re-derives the nav buttons - while there is an image to show: a
      // delete paging in behind the last one (`pageInBehind`) places itself.
      if (els.lightbox.open && lbList[lbIndex]) renderLightbox();
    });
  return galleryPending;
}

// The gallery with nothing to show, said under its head: on a first visit it
// was a blank screen beside a disabled "No runs yet", with nothing saying it
// fills as images are made. What fills it, and the authors' examples a press
// away (the first run's own Examples, `viewerExamples`); under a filter,
// that nothing matches; either way what the All models tick would add. Not
// over the tombstones of a gallery deleted down to none - they say it - nor
// in sheet mode, whose note does. From every page that lands and every count
// a delete or a finished job moves.
let galleryOthers = 0;
function syncGalleryEmpty() {
  const empty = !sheetMode && total <= 0 && !galleryRecords.length && !els.gallery.querySelector('.card');
  els.galleryEmpty.hidden = !empty;
  if (!empty) return;
  const search = els.search.value.trim();
  const others = galleryOthers && !els.allModels.checked
    ? ` ${plural(galleryOthers, 'image')} on other checkpoints: tick All models.` : '';
  if (search || els.favOnly.checked) {
    els.galleryEmpty.replaceChildren(
      `${search ? 'No image matches this search' : 'No starred image here yet'}`
      + `${search && els.favOnly.checked ? ' among the starred' : ''}.${others}`);
    return;
  }
  const examples = document.createElement('button');
  examples.type = 'button';
  examples.className = 'ghost tiny';
  examples.textContent = 'Examples';
  examples.title = 'Example prompts for this checkpoint, from its authors';
  examples.onclick = viewerExamples;
  els.galleryEmpty.replaceChildren('No images yet - Generate, or start from one of the authors\' ',
    examples, `.${others}`);
}

// Re-arms the scroll trigger once a page has landed. The observer reports a
// *change* in whether the button is in view, and a page that does not push it
// past the fold - a wide grid, the tail of a short run - leaves it in view with
// nothing left to fire; observing afresh always reports the current state once.
// Called on success only: after a failed page the button is the retry, and
// re-arming would fetch again from the server that just refused.
function watchMore() {
  moreArmed = true;
  if (!moreWatcher) return;
  moreWatcher.unobserve(els.more);
  moreWatcher.observe(els.more);
}

// How the cards are drawn from 1800 px (style.css; the pairs in the head
// are shown only there): `density` S, smaller cards and more to a row, or M,
// the wide tier's own; `fit` the whole picture letterboxed (contain, the
// default) or the card filled and the picture cut to its middle (cover).
// This browser's choice, remembered apart from the panel - Reset is the
// panel's and leaves it - every access in try/catch (remembered/remember).
const CARD_VIEW_KEY = 'local-diffusion.gallery.v1';
const CARD_VIEW = { density: ['m', 's'], fit: ['contain', 'cover'] };
const cardView = { density: 'm', fit: 'contain' };

function restoreCardView() {
  let stored = null;
  try { stored = JSON.parse(remembered(CARD_VIEW_KEY) ?? 'null'); } catch { /* unreadable: the defaults */ }
  for (const [key, values] of Object.entries(CARD_VIEW)) {
    if (values.includes(stored?.[key])) cardView[key] = stored[key];
  }
  syncCardView();
}

function setCardView(key, value) {
  if (!CARD_VIEW[key]?.includes(value)) return;
  cardView[key] = value;
  remember(CARD_VIEW_KEY, JSON.stringify(cardView));
  syncCardView();
}

// The grid's data attributes, which the stylesheet reads, and each pair as a
// size strip is: the chosen one pressed and the pair's one tab stop.
function syncCardView() {
  els.gallery.dataset.density = cardView.density;
  els.gallery.dataset.fit = cardView.fit;
  for (const button of els.cardView.querySelectorAll('button')) {
    const key = 'density' in button.dataset ? 'density' : 'fit';
    const on = button.dataset[key] === cardView[key];
    button.setAttribute('aria-pressed', String(on));
    button.tabIndex = on ? 0 : -1;
  }
}

function cardViewClick(e) {
  const button = e.target.closest('button');
  if (!button) return;
  const key = 'density' in button.dataset ? 'density' : 'fit';
  setCardView(key, button.dataset[key]);
}

// The trigger itself, rooted where the gallery scrolls: the window, or the
// gallery's own column beside the viewer (syncBeside). Rooted on the window,
// a button in that column's box counted only once the box showed it, and
// the 400 px of warning were gone. Made again whenever that changes - and
// observing afresh reports once, so a button already near the new root's
// edge fetches - unless the last page failed (`moreArmed`), where the button
// is the retry, as in watchMore.
let moreArmed = true;
function armMore() {
  moreWatcher?.disconnect();
  moreWatcher = new IntersectionObserver((entries) => {
    if (galleryMore && !sheetMode && entries.some((e) => e.isIntersecting)) loadGallery();
  }, { root: besideOn ? els.libraryScroll : null, rootMargin: '0px 0px 400px 0px' });
  if (moreArmed) moreWatcher.observe(els.more);
}

// ------------------------------------------------------------ gallery beside

// From 2000 px the gallery can stand in a column of its own at the window's
// right (style.css, `.stage.beside`), scrolling by itself, so its cards stay
// in view however tall the viewer is: stacked under it, a single image left
// one row of cards in view at QHD and a long run none. A tick in the gallery
// head, off by default and remembered per browser - not part of the panel,
// so Reset leaves it. In effect only while the window is that wide and the
// contact sheet is closed: a sheet is laid out for the stage's whole width,
// so the page goes back to one column for it and lands on it (setSheet).
// The prompt check, open, takes the column's place (style.css).
const BESIDE_KEY = 'local-diffusion.gallery-beside';
const BESIDE_MEDIA = '(min-width: 2000px)';
// Whether it is in effect, which the stage's class says to the stylesheet.
let besideOn = false;

function syncBeside() {
  const on = els.besideTick.checked && matchMedia(BESIDE_MEDIA).matches && !sheetMode;
  syncBesideTitle();
  if (on === besideOn) return;
  besideOn = on;
  els.stage.classList.toggle('beside', on);
  armMore();
  // An open check docks in the stage or takes the gallery's column, and
  // #recent's room beside it (fitRecent) goes with that.
  fitRecent();
  // Back under the viewer, the page's own scroll anchoring has the gallery.
  if (!on) releaseGallery();
}

// While a sheet is open the tick changes nothing on screen - the page is one
// column for the sheet - so its title says when it counts.
function syncBesideTitle() {
  const label = els.besideTick.closest('label');
  label.dataset.title ??= label.title;
  label.title = sheetMode
    ? `${label.dataset.title}. With the sheet open, as now, it takes effect when the sheet closes`
    : label.dataset.title;
}

// A finished job's images join the gallery beside the viewer
// (prependFinished) once the pointer is off its scrolling column - its
// cards, the gaps between them, its scrollbar and Load more: one image ahead
// of them moves every card a place along the rows, and a card about to be
// pressed was another by the press. Resolved at once where the gallery is
// not beside the viewer, the pointer is elsewhere or the column is hidden
// under the prompt check; else when the pointer leaves or the gallery goes
// back under the viewer (syncBeside). Where the pointer is comes from its
// enter and leave (app.js), as the panel's does (`panelPos.pointer`), not
// from `:hover`: a tap leaves `:hover` on what it touched with no pointer
// there, and a job that ended after one waited for the next tap in the
// column.
let galleryWaiting = [];
let galleryPointer = false;
function galleryPointerGone() {
  if (!besideOn || !galleryPointer || !isShown(els.libraryScroll)) return Promise.resolve();
  return new Promise((resolve) => galleryWaiting.push(resolve));
}

function galleryPointerLeft() {
  galleryPointer = false;
  releaseGallery();
}

function releaseGallery() {
  const waiting = galleryWaiting;
  galleryWaiting = [];
  waiting.forEach((resolve) => resolve());
}

// Cards put in ahead of the ones the gallery's own column shows while it is
// scrolled down keep those where they were, as the page's anchoring keeps a
// gallery it is scrolled into: the column has none of its own (style.css), so
// the first card or line in its view is measured across the change and the
// column scrolled by what it moved. At the column's top the new cards come
// in at the top, in view.
function holdGalleryView(change) {
  const box = els.libraryScroll;
  if (!besideOn || box.scrollTop <= 0) { change(); return; }
  const edge = box.getBoundingClientRect().top;
  const anchor = [...els.gallery.children].find((el) => el.getBoundingClientRect().bottom > edge);
  const before = anchor?.getBoundingClientRect().top;
  change();
  if (anchor?.isConnected) box.scrollTop += anchor.getBoundingClientRect().top - before;
}

function toggleBeside() {
  if (els.besideTick.checked) remember(BESIDE_KEY, '1');
  else forget(BESIDE_KEY);
  syncBeside();
}

function restoreBeside() {
  els.besideTick.checked = remembered(BESIDE_KEY) === '1';
  syncBeside();
}

// The local day a record was made on - the day `date:` searches by.
function localDay(rec) {
  return new Date(rec.created_at).toDateString();
}

const isToday = (rec) => localDay(rec) === new Date().toDateString();

// A card's day from the card itself (`data-made`), which its tombstone keeps:
// the record behind a deleted card has gone from galleryRecords, and a day's
// line worked out from the records stood a second time over a tombstone.
function cardDay(el) {
  return el?.dataset.made ? localDay({ created_at: el.dataset.made }) : null;
}

// A line across the grid where the local day changes between one card and
// the next: a night's grid and the next morning's renders ran together across
// the rows, and the only way to find where one ended was to open images. And
// one over the first card where the newest images are not today's: drawn only
// between days, the top group had no heading and read as today's, while the
// first line on the page named the older day under it. It carries no
// data-id, so paging, the deletes and the lightbox never see it; a day whose
// cards are all deleted keeps it over their tombstones, which hold their
// slots so that nothing moves under the pointer.
function dayBreak(rec) {
  const line = document.createElement('h3');
  line.className = 'gallery-break';
  const made = new Date(rec.created_at);
  const now = new Date();
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).toDateString();
  const day = made.toLocaleDateString([], {
    weekday: 'long', day: 'numeric', month: 'long',
    ...(made.getFullYear() === now.getFullYear() ? {} : { year: 'numeric' }),
  });
  line.textContent = made.toDateString() === yesterday ? `Yesterday · ${day}` : day;
  return line;
}

// The refusals a capability's absence gives, one sentence each wherever the
// page says it: a card's ⤢ and ❐, the lightbox's Refine (openRefine) and its
// Use these references (syncUseReferences).
const REFINE_NONE = 'This checkpoint has no refine pass compatible with this app. '
  + 'The FLUX.1, SDXL, Z-Image and Qwen-Image checkpoints have one.';
const REFERENCES_NONE = "This checkpoint can't use reference images; FLUX.2 [klein] can.";

function card(rec) {
  const div = document.createElement('div');
  div.className = 'card';
  div.classList.toggle('fav', !!rec.favorite);
  // Tagged by id so an action taken elsewhere - the lightbox, mostly - can
  // find this card and keep it in step; and by when it was made, which its
  // tombstone keeps for the day's lines (cardDay).
  div.dataset.id = rec.id;
  div.dataset.made = rec.created_at;
  // A group named as the page names the image, so its star and tools - each
  // named for what it does ("Star", "Reuse settings", "Refine: …"), the
  // toggles by a name that never changes - are heard as #N's: focus landing on
  // ⤢ back from the Refine dialog named no image. Not a list or a grid: the
  // day's lines and the tombstones are the cards' siblings, and a grid role
  // would want cells and put a screen reader in focus mode.
  div.setAttribute('role', 'group');
  div.setAttribute('aria-label', `#${rec.id}`);
  // A refine pass says so, and at what strength (the badge is the class's
  // ::before): it keeps its source's picture by design, and the two sat side
  // by side as the same card.
  if (rec.refine != null) {
    div.classList.add('refined');
    div.dataset.strength = rec.strength;
  }
  // One of the images the viewer shows now (panel.js).
  div.classList.toggle('this-run', viewerRecords.some((r) => r.id === rec.id));

  // Named "#N" and titled by thumb(), as every image of the page is.
  const img = thumb(rec, galleryRecords);
  img.setAttribute('aria-keyshortcuts', 'F');

  // The star is a button of its own over the picture's corner, where a
  // starred card's badge always was - a click on that badge went through to
  // the lightbox, while the real toggle sat 4 px from Delete.
  const star = document.createElement('button');
  star.type = 'button';
  star.className = 'star fav-badge';
  star.setAttribute('aria-keyshortcuts', 'F');
  star.onclick = () => setFavorite(rec, !rec.favorite);
  syncStar(star, !!rec.favorite);

  const refine = tool('⤢', '', () => {
    // Said here rather than in openRefine, so the card can say it where it
    // was pressed as well.
    if (!refineSupported) { refuseInCard(div, REFINE_NONE, 'No refine pass here'); return; }
    openRefine(rec);
  });
  refine.classList.add('refine-tool');
  syncRefineTool(refine);
  const reference = tool('❐', '', () => toggleReference(div, rec));
  reference.classList.add('ref-tool');
  syncReferenceTool(reference, rec.id);
  const del = tool('⌦', `Delete #${rec.id}`, () => deleteImage(rec), 'Delete');
  del.classList.add('del');

  const tools = document.createElement('div');
  tools.className = 'tools';
  // What the card is, just over its tools on hover and keyboard focus -
  // "#12 · 1024×1024 · seed 5" - which style.css draws from these, as it
  // does the tools' names, with empty alt text: the image's label and title
  // say it already. A pseudo-element, so the row's children stay its tools.
  Object.assign(tools.dataset, { n: rec.id, size: `${rec.width}×${rec.height}`, seed: rec.seed });
  tools.append(
    reuseButton(rec, true),
    refine,
    reference,
    tool('⊞', 'Show this run as a sheet', () => {
      if (!rec.run_id && !rec.job_id) {
        showError('That image predates runs being grouped, so it belongs to no run.', true, { slot: els.galleryError });
        return;
      }
      // The run id spans every job that worked on the grid - a night that died
      // and was resumed is one run - where the job id names only one of them.
      showSheetScope(rec.run_id ? `run:${rec.run_id}` : `job:${rec.job_id}`);
    }, 'Sheet'),
    del,
  );

  div.append(img, star, tools);
  markReference(div, referenceAt(rec.id));
  cardStops(div, rec.id === galleryCurrent);
  return div;
}

// `tip` is the short name the row shows on hover and keyboard focus
// (style.css draws it from data-tip): the title is the long one, and a native
// title came a second late or not at all.
function tool(label, title, onClick, tip = '') {
  const b = document.createElement('button');
  b.textContent = label;
  b.title = title;
  // A button is named by its contents, and `title` is never exposed - so these
  // announced as their glyph, or as a bare "button" for a glyph the screen
  // reader has no word for. The last of them deletes the image.
  b.setAttribute('aria-label', title);
  if (tip) b.dataset.tip = tip;
  b.onclick = onClick;
  return b;
}

// A star button anywhere the page draws one over a picture - the card's
// badge, the result beside the viewer - from the state it shows. A toggle:
// its name is "Star" whatever the state, which `aria-pressed` and the glyph
// carry - "Unstar, pressed" said the opposite of what it meant, as "Resume,
// pressed" did on Pause, and the lightbox's star said the same state in
// other words. The title (starTitle) is where the state is put in words.
function syncStar(button, on) {
  button.textContent = on ? '★' : '☆';
  button.setAttribute('aria-pressed', String(on));
  button.setAttribute('aria-label', 'Star');
  button.title = starTitle(on, button);
}

// A star's title, the card's and the lightbox's alike. It names F only where
// the button declares the key (aria-keyshortcuts), as both do: a title
// promising F where nothing handles it did nothing.
function starTitle(on, button) {
  const key = button.getAttribute('aria-keyshortcuts') === 'F' ? ' (F)' : '';
  return on ? `Starred - click to unstar${key}` : `Star${key}`;
}

// The card's tools that need something of the checkpoint get the lightbox
// Refine's treatment where it has not got it: aria-disabled with the reason
// as their title, never disabled, so each keeps its tab stop and a press
// still says why. They looked live and explained themselves only once
// pressed, in a line a screen away. Set as a card is built and again where
// the status flips the capability (`syncCardTools`), since cards can be on
// the page before the first status has answered.
function syncRefineTool(button) {
  button.setAttribute('aria-disabled', String(!refineSupported));
  const title = refineSupported ? 'Refine: scale this image up and re-denoise it part of the way'
    : 'Refine - this checkpoint has no refine pass';
  button.title = title;
  button.setAttribute('aria-label', title);
  button.dataset.tip = refineSupported ? 'Refine…' : 'No refine pass here';
}

// ❐ is a toggle: pressed on an image already among the references. Re-adding
// one used to change nothing and say nothing. Named "Reference image" in
// every state, as the star is "Star": the state is `aria-pressed`, and the
// number the prompt calls it, or why it is dead (at four the others' are),
// is its description and title: a name that changed with the state beside
// aria-pressed misread, as Pause's first cut did.
function syncReferenceTool(button, id) {
  const at = referenceAt(id);
  const full = referenceIds.length >= 4;
  const title = !referenceSupported ? 'Use as a reference image - this checkpoint takes none'
    : at >= 0 ? `Reference image ${at + 1} - click to remove`
      : full ? 'Use as a reference image - four is the most here; remove one first'
        : 'Use as a reference image';
  button.setAttribute('aria-disabled', String(!referenceSupported || (full && at < 0)));
  button.setAttribute('aria-pressed', String(at >= 0));
  button.title = title;
  button.setAttribute('aria-label', 'Reference image');
  button.setAttribute('aria-description', title);
  button.dataset.tip = !referenceSupported ? 'No references here'
    : at >= 0 ? `Reference ${at + 1} - remove` : full ? 'Four is the most' : 'Reference';
}

function syncCardTools() {
  els.gallery.querySelectorAll('.card[data-id]').forEach((div) => {
    syncRefineTool(div.querySelector('.refine-tool'));
    syncReferenceTool(div.querySelector('.ref-tool'), +div.dataset.id);
  });
  // The viewer's results have a Refine and a ❐ of their own - the ❐ only
  // where the checkpoint takes references.
  els.viewer.querySelectorAll('.refine-tool').forEach(syncRefineTool);
  els.viewer.querySelectorAll('.ref-tool').forEach((button) => {
    syncReferenceTool(button, +button.closest('[data-id]').dataset.id);
    button.hidden = !referenceSupported;
  });
}

// Where an image is among the references the panel sends: its index, or -1 -
// and -1 for all of them where the checkpoint takes none, since `collect()`
// then sends none.
function referenceAt(id) {
  return referenceSupported ? referenceIds.indexOf(id) : -1;
}

// A card or a sheet cell that is a reference: ringed and numbered as its chip
// is, so the gallery says which images are in use - the chips could be 1,600
// px from the card. `at` is referenceAt's.
function markReference(el, at) {
  el.classList.toggle('ref', at >= 0);
  let badge = el.querySelector(':scope > .ref-n');
  if (at < 0) {
    delete el.dataset.refN;
    badge?.remove();
    return;
  }
  el.dataset.refN = at + 1;
  if (!badge) {
    badge = document.createElement('span');
    badge.className = 'ref-n';
    badge.setAttribute('aria-hidden', 'true');
    el.append(badge);
  }
  badge.textContent = at + 1;
}

// After every change of the references (renderReferences calls it): the
// marks, and the card tools that follow them.
function syncReferenceMarks() {
  document.querySelectorAll('.card[data-id], .sheet-cell[data-id]').forEach((el) => {
    markReference(el, referenceAt(+el.dataset.id));
  });
  syncCardTools();
}

// The card's ❐. References still change only through setReferences.
function toggleReference(div, rec) {
  if (!referenceSupported) {
    refuseInCard(div, REFERENCES_NONE, 'No references here');
    return;
  }
  if (referenceIds.includes(rec.id)) {
    setReferences(referenceIds.filter((id) => id !== rec.id));
    cardNote(div, 'Removed');
    return;
  }
  if (referenceIds.length >= 4) {
    refuseInCard(div, 'Four reference images is the most this panel sends; remove one first.', 'Four is the most');
    return;
  }
  setReferences([...referenceIds, rec.id]);
  cardNote(div, `Added as image ${referenceIds.length}`);
}

// A card tool's refusal: in the gallery's line, which stays and is the one
// that is announced, and for a moment where it was pressed too - its gist,
// which has one line of a card at its narrowest.
function refuseInCard(div, message, short) {
  showError(message, true, { slot: els.galleryError });
  cardNote(div, short, { quiet: true });
}

// What a card tool did, over the row it was pressed in, for two seconds: the
// panel's chips, and the line under the gallery's head, can be a screen away
// from the card. A status for what is said nowhere else; `quiet` (hidden from
// screen readers) where the gallery's line says it already.
function cardNote(div, text, { quiet = false } = {}) {
  div.querySelector(':scope > .card-note')?.remove();
  const note = document.createElement('div');
  note.className = `card-note${quiet ? ' refused' : ''}`;
  if (quiet) note.setAttribute('aria-hidden', 'true');
  else note.setAttribute('role', 'status');
  div.append(note);
  // A status is written once it is in the page: a live region added with its
  // text in it already is not announced.
  if (quiet) note.textContent = text;
  else requestAnimationFrame(() => { note.textContent = text; });
  setTimeout(() => note.remove(), 2000);
}

// The card the keyboard is on, by id: its image and its tools are the
// gallery's tab stops, and every other card's are not - seven stops a card
// put the 30th image some 270 Tab presses down the page. The arrow keys move
// it (`galleryKey`), as does focus arriving any other way: a click, the
// lightbox handing focus back. An id rather than the element, so a gallery
// reset or queried again keeps it where the record is still loaded.
let galleryCurrent = null;
// Whether the user put it there - focus arrived in a card since the last
// reset - or it is the grid's default, its first card. A job's images
// prepended ahead of a default first card take the stop (prependFinished):
// kept on the old head, Tab into the grid skipped everything just made.
let galleryChosen = false;

function currentCard() {
  return galleryCurrent == null ? null : els.gallery.querySelector(`.card[data-id="${galleryCurrent}"]`);
}

function cardStops(div, on) {
  div.querySelectorAll('img, button').forEach((el) => { el.tabIndex = on ? 0 : -1; });
}

function makeCurrent(div) {
  if (!div) return;
  const was = currentCard();
  if (was === div) return;
  if (was) cardStops(was, false);
  galleryCurrent = +div.dataset.id;
  cardStops(div, true);
}

// After a page lands: the grid keeps one way in from the keyboard.
function keepCurrentCard() {
  if (!currentCard()) makeCurrent(els.gallery.querySelector('.card[data-id]'));
}

// The next card along that is still an image, else the one before.
function liveNeighbour(el) {
  for (let n = el.nextElementSibling; n; n = n.nextElementSibling) if (n.matches('.card[data-id]')) return n;
  for (let n = el.previousElementSibling; n; n = n.previousElementSibling) if (n.matches('.card[data-id]')) return n;
  return null;
}

function galleryFocus(e) {
  const div = e.target.closest?.('.card[data-id]');
  if (!div) return;
  makeCurrent(div);
  galleryChosen = true;
}

// F stars the card focus is in, from its image, its star or a tool, as in the
// lightbox and on the sheet - the star's title promises it. On the image the
// arrows walk the grid and Home and End go to the first and last card loaded;
// on a tool they are left to the tool.
function galleryKey(e) {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  const div = e.target.closest?.('.card[data-id]');
  if (!div) return;
  if (e.key === 'f' || e.key === 'F') {
    e.preventDefault();
    const rec = galleryRecords.find((r) => r.id === +div.dataset.id);
    if (rec) void setFavorite(rec, !rec.favorite, { say: true });
    return;
  }
  if (e.target.parentElement !== div || e.target.tagName !== 'IMG') return;
  const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: 'up', ArrowDown: 'down', Home: 'first', End: 'last' }[e.key];
  if (step === undefined) return;
  e.preventDefault();
  void moveCard(div, step);
}

async function moveCard(from, step) {
  const cards = [...els.gallery.querySelectorAll('.card[data-id]')];
  const at = cards.indexOf(from);
  const to = step === 'first' ? cards[0] : step === 'last' ? cards.at(-1)
    : typeof step === 'number' ? cards[at + step] : rowNeighbour(cards, at, step === 'down');
  if (!to) {
    // Past the last card loaded, the next page, as scrolling there fetches
    // it - and only if it brought cards, and focus is still where it was.
    if (!(step === 1 || step === 'down') || !galleryMore || sheetMode) return;
    await loadGallery().catch(() => {});
    const img = from.querySelector('img');
    if (!img || document.activeElement !== img) return;
    if (els.gallery.querySelectorAll('.card[data-id]').length > cards.length) void moveCard(from, step);
    return;
  }
  to.querySelector('img').focus({ preventScroll: true });
  to.scrollIntoView({ block: 'nearest' });
}

// The card in the row above or below, the nearest across. Found from where
// the cards are rather than from the grid's column count: a day's line starts
// a row of its own, and a short row before it put "a row on" by the count
// that many cards short.
function rowNeighbour(cards, at, down) {
  const box = cards[at].getBoundingClientRect();
  const centre = box.left + box.width / 2;
  const dir = down ? 1 : -1;
  let row = null;
  let best = null;
  for (let i = at + dir; i >= 0 && i < cards.length; i += dir) {
    const r = cards[i].getBoundingClientRect();
    if (Math.abs(r.top - box.top) < 1) continue;
    if (row !== null && Math.abs(r.top - row) >= 1) break;
    row = r.top;
    const off = Math.abs(r.left + r.width / 2 - centre);
    if (!best || off < best.off) best = { card: cards[i], off };
  }
  return best?.card;
}

function modelsDiffer(modelId) {
  return !!loadedModel && !!modelId && modelId !== loadedModel;
}

// Warn before the click, not only after the panel has already been replaced.
// Gallery cards use a compact glyph; the lightbox has room for the full label.
// The data attributes let the first status response update buttons that were
// rendered while the server was still answering. `rec` is the record, or as
// much of it as the button keeps: its checkpoint, the source of a pass, its
// references and adapter.
function syncReuseButton(button, rec, compact = false) {
  const modelId = rec.model_id;
  const pass = rec.refine ?? null;
  button.dataset.reuseModel = modelId || '';
  button.dataset.reusePass = pass ?? '';
  button.dataset.reuseReference = (rec.reference ?? []).join(',');
  button.dataset.reuseLora = rec.lora || '';
  button.dataset.reuseCompact = String(compact);
  button.classList.add('reuse-settings');
  const mismatch = modelsDiffer(modelId);
  const from = (modelId || '').split('/').pop();
  const current = (loadedModel || '').split('/').pop();
  const warning = `generated with ${from}; the current model is ${current}`;
  button.classList.toggle('model-mismatch', mismatch);
  // A refined record's Reuse opens the Refine dialog on its source with the
  // pass as it ran (`reusePass`), where "Load these settings into the panel"
  // promised the panel and a dialog opened on another image.
  // A pass whose source has been deleted cannot be made again: its Reuse
  // fills the panel (reusePass), and says so before the click rather than
  // promising a dialog on an image that is gone.
  const gone = pass != null && !!rec.refine_missing;
  button.dataset.reusePassGone = String(gone);
  if (pass != null && !gone) {
    button.textContent = compact ? (mismatch ? '⚠' : '↻') : `${mismatch ? '⚠ ' : ''}Reuse pass…`;
    button.title = mismatch ? `Warning: ${warning}`
      : `Open the Refine dialog on #${pass} with this pass's size, strength and seed`;
    // Led by the words shown ("Reuse pass…"), so a voice command that says
    // what it reads finds the button.
    button.setAttribute('aria-label', `Reuse pass in the Refine dialog${mismatch ? ` (warning: ${warning})` : ''}`);
    button.dataset.tip = 'Reuse pass…';
    return;
  }
  const lost = mismatch ? notCarried(rec) : '';
  const why = gone ? `; #${pass}, its source, has been deleted, so the pass cannot be made again` : '';
  button.textContent = compact ? (mismatch ? '⚠' : '↻') : `${mismatch ? '⚠ ' : ''}Reuse settings`;
  button.title = mismatch ? `Warning: ${warning}${lost}${why}` : `Load these settings into the panel${why}`;
  button.setAttribute('aria-label', mismatch ? `Reuse settings (warning: ${warning}${lost}${why})` : 'Reuse settings');
  button.dataset.tip = mismatch ? 'Reuse - other model' : 'Reuse';
}

// A card's image title: the facts that tell it from the card beside it, a
// pass's source and strength among them - "(deleted)" where the source has
// gone, as the lightbox's facts say it.
function cardTitle(rec) {
  const pass = rec.refine != null
    ? ` · refine of #${rec.refine}${rec.refine_missing ? ' (deleted)' : ''} @ ${rec.strength}` : '';
  return `#${rec.id} · ${rec.width}×${rec.height} · seed ${rec.seed}${pass}\n${rec.prompt}`;
}

// What a Reuse here leaves behind, as fillPanel decides it - references where
// the checkpoint takes none, an adapter the LoRA box does not offer - for the
// warning to name before the click; the note after it says so again.
function notCarried(rec) {
  const references = rec.reference?.length && !referenceSupported;
  const adapter = rec.lora && !(loraSupported && [...els.lora.options].some((o) => o.value === rec.lora));
  const what = [
    references && `its reference image${rec.reference.length === 1 ? '' : 's'}`,
    adapter && 'its adapter',
  ].filter(Boolean);
  return what.length ? `; ${what.join(' and ')} will not carry over` : '';
}

function reuseButton(rec, compact = false) {
  const button = tool('', '', () => reuse(rec));
  syncReuseButton(button, rec, compact);
  return button;
}

function syncReuseButtons() {
  document.querySelectorAll('.reuse-settings').forEach((button) => {
    const kept = button.dataset;
    syncReuseButton(button, {
      model_id: kept.reuseModel,
      refine: kept.reusePass ? +kept.reusePass : null,
      refine_missing: kept.reusePassGone === 'true',
      reference: kept.reuseReference ? kept.reuseReference.split(',').map(Number) : [],
      lora: kept.reuseLora || null,
    }, kept.reuseCompact === 'true');
  });
}

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
  const focused = document.activeElement;
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
    document.querySelectorAll(`img[data-id="${r.id}"]`).forEach((img) => { img.title = cardTitle(r); });
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
  if (sheetMode) renderSheet();
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
  let found;
  try {
    found = await api(`/api/gallery/unstarred?${params}`);
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
    ...freshLine(head, found),
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
  if (typed.trim() !== String(found.unstarred)) {
    showError(`Nothing deleted: "${typed.trim()}" is not ${found.unstarred}.`, true, slot);
    return;
  }
  params.set('expected', found.unstarred);
  let result;
  try {
    result = await api(`/api/gallery/unstarred?${params}`, { method: 'DELETE' });
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

// The dialog's line for what the grid has just been brought up to: the
// unstarred images newer than its head before the look (`head`), which the
// sweep counts. A page that did not join up was loaded again from the top
// (prependFinished), and where every card on it is newer and there is more,
// the count is a floor. The few sources the preview names stay out of it:
// the sweep keeps them.
function freshLine(head, found) {
  if (head == null || sheetMode) return [];
  const kept = new Set(found.sources);
  const n = Math.min(found.unstarred,
    galleryRecords.filter((r) => r.id > head && !r.favorite && !kept.has(r.id)).length);
  if (!n) return [];
  // At the top of the grid: a page scrolled down keeps its view
  // (holdGalleryView), so they may be above it rather than in sight.
  const was = n === 1 ? 'was' : 'were';
  if (galleryMore && galleryRecords.at(-1).id > head) {
    return [`At least ${n} of them ${was} made since this page last loaded; the newest are at the top of the grid now.`];
  }
  return [`${n} of them ${was} made since this page last loaded; `
    + `${n === 1 ? 'it is' : 'they are'} at the top of the grid now.`];
}

// Shown while there is something it could act on: the filter matches images
// and the Starred tick is off, since with it on every match is a keeper.
// Whether the search box reads as a strict scope is the server's call, at
// the click.
function syncDeleteUnstarred() {
  els.deleteUnstarred.hidden = els.favOnly.checked || total <= 0;
}

// What made a record, into the panel: the prompt pair and the seed, and with
// `sampling` its steps, CFG, true CFG, mu and conditioning too. The size is
// not written here: a refine pass whose source has gone keeps the panel's.
// The two boxes are written as typing writes them (`replaceBoxText`), so
// Ctrl+Z gives back what they held: assigning `.value` emptied their undo
// history, and in Chromium a later Ctrl+Z could then splice the old prompt
// into the new one, which the panel went on to remember. The negative first,
// so the caret ends in the prompt. Synchronous, all of it: the references are
// in place when this returns.
function fillPanel(rec, sampling) {
  rewriteBox(els.negative, rec.negative_prompt ?? '');
  rewriteBox(els.prompt, rec.prompt);
  promptReplaced();
  els.seed.value = rec.seed;
  if (!sampling) return;
  ownControls();
  els.steps.value = rec.steps;
  els.cfg.value = rec.guidance_scale;
  // The scale the negative prompt ran at. Only where the box is a knob of its
  // own: on a checkpoint whose CFG is real CFG the record's value *is* the
  // CFG box, and the greyed box would only echo it.
  els.trueCfg.value = trueCfgSupported && rec.true_cfg != null ? rec.true_cfg : '';
  // Turbo's flow shift. Records from a pipeline that derives its own carry null.
  els.mu.value = rec.mu ?? '';
  // The conditioning that made it, where this server can still apply it: the
  // references are ids in this same gallery, and the adapter has to be a file
  // the select is offering or the request would be refused at submit.
  referenceIds = referenceSupported ? [...(rec.reference ?? [])] : [];
  const listed = [...els.lora.options].some((o) => o.value === rec.lora);
  els.lora.value = loraSupported && rec.lora && listed ? rec.lora : '';
  els.loraScale.value = els.lora.value ? (rec.lora_scale ?? 1) : 1;
}

// tags.js's `writeBox` - the page's one whole-box write: as typing writes it,
// with no suggestion list or typing state over the new text - with the box
// focused first without the browser's scroll to it. Writing as typing does
// takes the focus, and the browser then scrolled the page to the box as
// though a stuck panel moved with it: half a screen of the gallery lost at
// the page's end, nothing at all for a box focused already. Where the page
// goes after a Reuse is `revealReused`'s, and after its Undo, pressed at the
// panel's foot, nowhere.
function rewriteBox(el, text) {
  el.focus({ preventScroll: true });
  writeBox(el, text);
}

// Reuse restores what made a record. A direct render's settings go into the
// panel; a refine pass is the Refine dialog's, so a refined record opens the
// dialog on its source with the pass as it ran (`reusePass`).
function reuse(rec) {
  if (rec.refine != null) { void reusePass(rec); return; }
  reuseIntoPanel(rec, true);
}

// The note the last Reuse left in the panel - its sentences, the panel as it
// was before and as the Reuse left it, and the Undo button that puts it
// back - or null once there is no Undo to offer.
let reuseNote = null;

// What a Reuse's Undo puts back, as the panel holds it now.
function reuseFields() {
  return {
    prompt: els.prompt.value, negative: els.negative.value, seed: els.seed.value,
    width: els.width.value, height: els.height.value,
    steps: els.steps.value, cfg: els.cfg.value, mu: els.mu.value, trueCfg: els.trueCfg.value,
    owned: { ...owned }, reference: [...referenceIds], lora: els.lora.value, loraScale: els.loraScale.value,
  };
}

// The Examples' rule (tags.js's `syncPromptUndo`) for the whole of what the
// Undo restores: it is on offer only while the panel holds what the Reuse
// left there. It used to stay through any later edit, and pressing it then
// threw away the typing, the seed or the Example picked since, and said the
// panel was as it was. So the first change takes the button, and the note's
// sentences stay. Asked by `persistPanel`, which every change of the panel
// reaches - typing, a preset, the dice, an Example (whose own Undo then
// stands alone).
function syncReuseUndo() {
  const note = reuseNote;
  if (!note || JSON.stringify(reuseFields()) === JSON.stringify(note.after)) return;
  reuseNote = null;
  note.button?.remove();
}

function reuseIntoPanel(rec, withSize, note = '') {
  // One Undo on offer at a time: an Example's, for the text this replaces,
  // goes with it.
  dropPromptUndo();
  // Ctrl+Z covers the two text boxes; Reuse also replaces the seed, the size,
  // steps/CFG/mu and which of those the user had taken off the profile, so
  // the note's Undo restores those from this.
  const before = {
    state: panelState(),
    values: { steps: els.steps.value, cfg: els.cfg.value, mu: els.mu.value, trueCfg: els.trueCfg.value },
    owned: { ...owned },
  };
  fillPanel(rec, true);
  if (withSize) {
    els.width.value = rec.width;
    els.height.value = rec.height;
  }
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  // Reuse used to replace a half-written prompt with no word, and the work
  // was gone; the note says so whenever there was other text to replace. Its
  // first sentence, the Undo after it, then whatever else there is to say:
  // the note is at the panel's foot, and where the window shows the prompt
  // and only the start of the note, the start is what is in sight.
  const replaced = before.state.prompt.trim() !== '' && before.state.prompt !== rec.prompt;
  const lead = `Settings from #${rec.id} loaded${replaced ? ' - your prompt was replaced' : ''}.`;
  const notes = [note];
  // Steps, CFG and mu are checkpoint-specific: Turbo's 8 / 0 / 1.15 on FLUX is
  // a visibly broken image recorded under FLUX. The checkpoint is part of the
  // reproduction set as much as the seed, so say so - as a note, not a failure.
  const short = (id) => (id || '').split('/').pop();
  if (loadedModel && rec.model_id !== loadedModel) {
    notes.push(
      `These settings were made on ${short(rec.model_id)}; the server is running ` +
      `${short(loadedModel)}, whose steps, CFG and mu differ. Prompt` +
      `${withSize ? ', size' : ''} and seed carry over; check the steps, CFG and mu ` +
      'boxes before generating.',
    );
  }
  // What fillPanel could not carry, said: a dropped reference renders a plain
  // text-to-image that looks like a success.
  const references = rec.reference ?? [];
  if (references.length && !referenceSupported) {
    const one = references.length === 1;
    notes.push(
      `Its ${one ? '' : `${references.length} `}reference image${one ? '' : 's'} (${imageNames(references)}) `
      + `${one ? 'is' : 'are'} not used: ${short(loadedModel) || 'this checkpoint'} takes none, `
      + 'so this renders from the prompt alone.',
    );
  }
  reuseNote = { lead, parts: notes, before, after: reuseFields(), id: rec.id, button: null };
  showReuseNote();
  revealReused(reuseNote);
  void checkReused(reuseNote, rec);
}

// The Reuse note: its first sentence, the Undo, the rest. Written in the
// panel's slot - Reuse sends the user to the panel - and rewritten when a
// follow-up has more to say.
function showReuseNote() {
  const note = reuseNote;
  showError(note.lead, true);
  const undo = document.createElement('button');
  undo.type = 'button';
  undo.className = 'ghost tiny undo';
  undo.textContent = 'Undo';
  undo.title = 'Put the panel back as it was before this Reuse';
  undo.onclick = () => undoReuse(note);
  els.error.append(' ', undo);
  const rest = note.parts.filter(Boolean).join(' ');
  if (rest) els.error.append(` ${rest}`);
  note.button = undo;
}

// Where a Reuse leaves the page: the prompt in sight below the topbar, its
// label with it, and the note's Undo too where the window has room for both.
// Not the jump to the top of the page there used to be, which beside the
// gallery lost the user's place for nothing.
function revealReused(note) {
  const panel = document.querySelector('.panel');
  // The note has just made the panel taller: placed now rather than after
  // the frame, so what is measured below is where the panel stays.
  placePanel();
  // Beside the gallery the panel moves, not the page (panel.js's sticky top).
  revealInPanel(els.prompt);
  if (promptInSight()) { showUndo(note); return; }
  // In the page's flow - a one-column page, or a panel at the page's end,
  // which a sticky top cannot push down - the page moves, as far as puts the
  // panel where it rests: under the topbar as at the top of the page, or at
  // its sticky top.
  const rest = panelPos.top === null
    ? document.querySelector('.topbar').getBoundingClientRect().bottom + parseFloat(getComputedStyle(panel.parentElement).paddingTop)
    : panelPos.rest;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  scrollTo({ top: scrollY + panel.getBoundingClientRect().top - rest, behavior: reduce ? 'auto' : 'smooth' });
  if (reduce) showUndo(note);
  else addEventListener('scrollend', () => showUndo(note), { once: true });
}

// The prompt's field, label and all, between the topbar and the window's foot.
function promptInSight() {
  const field = els.prompt.closest('.field').getBoundingClientRect();
  return field.top >= document.querySelector('.topbar').getBoundingClientRect().bottom - 0.5
    && els.prompt.getBoundingClientRect().bottom <= innerHeight + 0.5;
}

// The Undo, after the note's first sentence: a stuck panel goes up to bring
// it in sight - a gutter clear of the window's foot where there is room for
// that - only as far as the prompt's field stays clear of the topbar. On a
// window too short for both, the prompt is what Reuse is about, and Ctrl+Z
// in it still gives the text back.
function showUndo(note) {
  if (panelPos.top === null || reuseNote !== note || !note.button.isConnected) return;
  const room = els.prompt.closest('.field').getBoundingClientRect().top
    - document.querySelector('.topbar').getBoundingClientRect().bottom;
  const bottom = note.button.getBoundingClientRect().bottom;
  const lift = Math.min(bottom - (innerHeight - panelPos.gap), room);
  if (lift > 0 && bottom - lift <= innerHeight) setPanelTop(panelPos.top - lift);
}

// What a Reuse could only find out by asking, once the panel is filled: a
// restored reference that has since been deleted - dropped, and only on the
// server's "unknown image", so a passing failure never takes a live one out -
// and why an adapter the record ran with is not in the LoRA box.
async function checkReused(note, rec) {
  const extra = [];
  const restored = referenceSupported ? (rec.reference ?? []) : [];
  const answers = await Promise.all(restored.map((id) => api(`/api/images/${id}`)
    .then(() => null, (err) => (/unknown image/.test(err.message) ? id : null))));
  const dead = answers.filter((id) => id != null);
  if (dead.length) {
    // Part of what the Reuse left, so not a change that retires its Undo.
    if (reuseNote === note) note.after.reference = note.after.reference.filter((id) => !dead.includes(id));
    setReferences(referenceIds.filter((id) => !dead.includes(id)));
    const which = dead.map((id) => `reference image ${restored.indexOf(id) + 1} (gallery #${id})`).join(' and ');
    extra.push(`Its ${which} ${dead.length === 1 ? 'is' : 'are'} no longer in the gallery, so `
      + `${dead.length === 1 ? 'it was' : 'they were'} left out.`);
  }
  if (rec.lora && els.lora.value !== rec.lora) {
    let reason = 'the file is no longer in the adapter directory';
    try {
      const body = await api('/api/loras');
      const entry = (body.loras ?? []).find((e) => e.name === rec.lora);
      if (!body.supported) reason = body.detail || 'this installation cannot load adapters';
      else if (entry?.detail) reason = entry.detail;
      else if (entry) reason = 'the adapter list had not loaded yet; pick it in the LoRA box';
    } catch { /* the plain reason stands */ }
    extra.push(`Its adapter ${rec.lora} @ ${rec.lora_scale ?? 1} is not applied here - ${reason}.`);
  }
  if (!extra.length) return;
  // Onto the note while it is still the one shown; on its own where nothing
  // has replaced it with something else to say.
  if (reuseNote === note && note.button?.isConnected) {
    note.parts.push(...extra);
    showReuseNote();
  } else if (els.error.hidden) showError(extra.join(' '), true);
}

// The panel as the Reuse found it: the boxes through `replaceBoxText` again,
// so Ctrl+Z still walks back through both, and the seed, the size, the
// sampling values with whose they were, the references and the adapter.
function undoReuse(note) {
  const { state, values } = note.before;
  if (reuseNote === note) reuseNote = null;
  rewriteBox(els.negative, state.negative);
  rewriteBox(els.prompt, state.prompt);
  els.seed.value = state.seed;
  els.width.value = state.width;
  els.height.value = state.height;
  Object.assign(owned, note.before.owned);
  els.steps.value = values.steps;
  els.cfg.value = values.cfg;
  els.mu.value = values.mu;
  els.trueCfg.value = values.trueCfg;
  referenceIds = state.reference;
  els.lora.value = state.lora;
  els.loraScale.value = state.loraScale;
  promptReplaced();
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  showError(`Reuse of #${note.id} undone: the panel is as it was.`, true);
}

// A refined record holds its source's id, not the source: asked for, so the
// dialog has the picture and the size to scale. A source that has been
// deleted cannot be refined again - the prompt and settings still go into
// the panel, at the panel's own size, since the pass's may be past what a
// direct render takes.
async function reusePass(rec) {
  let source;
  try {
    source = await api(`/api/images/${rec.refine}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) { showError(err.message); return; }
    reuseIntoPanel(rec, false,
      `#${rec.refine}, the source this was refined from, is no longer in the gallery, `
      + 'so the pass cannot be made again. The size is left as it was.');
    return;
  }
  openRefine(source, rec);
}
