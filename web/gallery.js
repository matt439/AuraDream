// Local Diffusion's page script: gallery paging, cards and keyboard navigation.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// -------------------------------------------------------------------gallery

function resetGallery() {
  // Anything already in flight was fetched under the old filters and against
  // the grid this is about to empty; the token tells it to drop itself.
  galleryEpoch += 1;
  galleryMore = false;
  galleryRecords = [];
  galleryReadHead = null;
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
  // The recycle bin is every checkpoint's, and Starred is held there.
  if (els.favOnly.checked && !binMode) params.set('favorites', 'true');
  if (els.allModels.checked && !binMode) params.set('all_models', 'true');
  els.more.disabled = true;
  els.more.textContent = 'Loading…';
  els.lbNext.disabled = true;

  galleryPending = api(`${galleryEndpoint()}?${params}`)
    .then((page) => {
      if (epoch !== galleryEpoch) return;  // superseded by a reset
      // The starting point for the status's newest id (followGalleryHead),
      // read with this page: nothing saved after it is counted as seen.
      if (newestImageSeen === null && !params.has('before') && Number.isInteger(page.newest_image_id)) {
        keepGalleryHead(page.newest_image_id);
      }
      if (!params.has('before') && Number.isInteger(page.newest_image_id)) galleryReadHead = page.newest_image_id;
      total = page.total;
      els.count.textContent = total ? `(${total})` : '';
      // What the tick would add: the same filter's matches on other
      // checkpoints, so a run made on the other model reads as "+12" on the
      // tick rather than as an empty grid. 0 once it is on.
      showOtherModels(page.other_models);
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
      // The selection's count of what this view does not show, and Select
      // held or not as the sheet is open (gallery-select.js).
      syncSelection();
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
      if (!galleryRecords.length) {
        els.count.textContent = '';
        showOtherModels(0);
      }
      // The grid resetGallery emptied shows none of the selection now.
      syncSelection();
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

// Where the grid's pages come from: the gallery, or in the recycle bin
// (bin.js) the bin, paged and searched the same way.
function galleryEndpoint() {
  return binMode ? '/api/gallery/bin' : '/api/gallery';
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
  if (binMode) {
    els.galleryEmpty.replaceChildren(search ? 'No image in the recycle bin matches this search.' : 'The recycle bin is empty.');
    return;
  }
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
  // The cards have moved from under a still pointer either way, and it is
  // on the head's tick or at the window's edge: what waited for it comes in.
  releaseGallery();
}

// While a sheet is open the tick changes nothing on screen - the page is one
// column for the sheet - so its title says when it counts.
function syncBesideTitle() {
  const label = els.besideTick.closest('label');
  label.dataset.title ??= label.title;
  label.title = binMode ? BIN_HELD : sheetMode
    ? `${label.dataset.title}. With the sheet open, as now, it takes effect when the sheet closes`
    : label.dataset.title;
}

// A finished job's images join the gallery (prependFinished) once the
// pointer is off its cards - with Gallery beside, its scrolling column: the
// cards, the gaps between them, its scrollbar and Load more. One image ahead
// of them moves every card a place along the rows, and a card about to be
// pressed was another by the press - its ⌦ bins without asking. Under the
// viewer too: the page's scroll anchoring keeps the rows where they were,
// not which card is in each place along them. Resolved at once where the
// pointer is elsewhere or the cards are not shown (the column hidden under
// the prompt check); else when the pointer leaves or the gallery changes
// columns (syncBeside). Where the pointer is comes from #library-scroll's
// enter and leave (app.js) - a box of its own beside the viewer, none under
// it, where its children's boxes are what the pointer enters - as the
// panel's does (`panelPos.pointer`), not from `:hover`: a tap leaves
// `:hover` on what it touched with no pointer there, and a job that ended
// after one waited for the next tap in the column.
let galleryWaiting = [];
let galleryPointer = false;
function galleryPointerGone() {
  if (!galleryPointer || !isShown(els.gallery)) return Promise.resolve();
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
    // An inpaint is a refine pass with a mask, and its badge says which.
    div.classList.toggle('inpainted', !!rec.mask);
    // And an outpaint an inpaint whose mask the server built (outpaint.js).
    div.classList.toggle('outpainted', !!rec.outpaint);
    div.dataset.strength = rec.strength;
  }
  // One of the images the viewer shows now (viewer.js).
  div.classList.toggle('this-run', viewerRecords.some((r) => r.id === rec.id));

  // Named "#N" and titled by thumb(), as every image of the page is.
  // Its click opens it; Ctrl/Cmd or Select picks it (cardPress,
  // gallery-select.js); syncCardSelected names the current action.
  const img = thumb(rec, galleryRecords, undefined, (e) => cardPress(div, rec, e));
  // A Shift+click picks a range out of select mode too, where the grid's
  // text is selectable: the press must not drag a text selection across it,
  // nor a double click select the words beside it.
  img.addEventListener('mousedown', (e) => { if (e.button === 0 && (e.shiftKey || e.detail > 1)) e.preventDefault(); });
  // A card of the recycle bin's has its own tools (bin.js).
  if (rec.binned_at) return binCard(div, img, rec);

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
  // Inpaint (inpaint.js): held with the reason where the checkpoint has none.
  const inpaint = tool('✎', '', () => inpaintFromCard(rec, div));
  inpaint.classList.add('inpaint-tool');
  syncInpaintTool(inpaint);
  const reference = tool('❐', '', () => toggleReference(div, rec));
  reference.classList.add('ref-tool');
  syncReferenceTool(reference, rec.id);
  const del = tool('⌦', `Move #${rec.id} to the recycle bin`, () => deleteImage(rec), 'Move to recycle bin');
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
    tool('⊞', 'Show this run as a sheet', () => showRecordRun(rec), 'Sheet'),
    // Last before Delete, so the tools before it keep their places.
    inpaint,
    del,
  );

  div.append(img, star, tools);
  markReference(div, referenceAt(rec.id));
  cardStops(div, rec.id === galleryCurrent);
  // Picked, or in select mode, as the selection says (gallery-select.js).
  syncCardSelected(div);
  return div;
}

// The card's ⊞, and the right-click menu's Show run as sheet wherever the
// image is (context-menu.js): the lightbox closes for the sheet, as the
// facts' ⊞ Show this run closes it.
function showRecordRun(rec) {
  if (!rec.run_id && !rec.job_id) {
    showError('That image predates runs being grouped, so it belongs to no run.', true, { slot: els.galleryError });
    return;
  }
  if (els.lightbox.open) closeLightbox();
  // The run id spans every job that worked on the grid - a night that died
  // and was resumed is one run - where the job id names only one of them.
  showSheetScope(rec.run_id ? `run:${rec.run_id}` : `job:${rec.job_id}`);
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
  // reader has no word for. The last of them moves the image to the recycle bin.
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
  // Not on the recycle bin's cards, which have neither (binCard).
  els.gallery.querySelectorAll('.card[data-id]:not(.binned)').forEach((div) => {
    syncRefineTool(div.querySelector('.refine-tool'));
    syncReferenceTool(div.querySelector('.ref-tool'), +div.dataset.id);
  });
  // Inpaint's, on the cards and the viewer's results alike.
  syncInpaintTools();
  // The viewer's results have a Refine and a ❐ of their own - the ❐ only
  // where the checkpoint takes references.
  els.viewer.querySelectorAll('.refine-tool').forEach(syncRefineTool);
  els.viewer.querySelectorAll('.ref-tool').forEach((button) => {
    syncReferenceTool(button, +button.closest('[data-id]').dataset.id);
    button.hidden = !referenceSupported;
  });
  // And the footer's Refine…, held with the same reason (refine-batch.js).
  syncRefineBatchButton();
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
  // The right-click menu over the lightbox (context-menu.js) passes its
  // stage: the note goes in the lightbox's line, and a refusal's gist not at
  // all - showError mirrored the sentence there already.
  if (div === els.lbStage) {
    if (!quiet) showError(text, true, { slot: els.lbError, mirror: false });
    return;
  }
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
  button.dataset.reuseMask = rec.mask ?? '';
  // An outpaint's spec by its key (`fill:L,T,R,B@band`), so a status
  // refresh rebuilds an outpaint's button rather than an inpaint's.
  button.dataset.reuseOutpaint = rec.outpaint ? outpaintKey(rec.outpaint) : '';
  button.dataset.reuseReference = (rec.reference ?? []).join(',');
  button.dataset.reuseLora = rec.lora || '';
  button.dataset.reuseCompact = String(compact);
  button.classList.add('reuse-settings');
  const mismatch = modelsDiffer(modelId);
  // The running checkpoint by the name the topbar gives it (checkpointName):
  // NetaYume v4's repository is named for its parent model.
  const from = checkpointName(modelId);
  const current = checkpointName(loadedModel);
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
  button.dataset.reusePassBinned = String(gone && !!rec.refine_binned);
  // An outpaint's opens the Outpaint dialog on its source with its sides,
  // band and fill (`reuseOutpaint`) - first, since it has a mask as well.
  if (pass != null && !gone && rec.outpaint) {
    button.textContent = compact ? (mismatch ? '⚠' : '↻') : `${mismatch ? '⚠ ' : ''}Reuse outpaint…`;
    button.title = mismatch ? `Warning: ${warning}`
      : `Open the Outpaint dialog on #${pass} with this outpaint's sides, band, strength and seed`;
    button.setAttribute('aria-label', `Reuse outpaint in the Outpaint dialog${mismatch ? ` (warning: ${warning})` : ''}`);
    button.dataset.tip = 'Reuse outpaint…';
    return;
  }
  // An inpaint's opens the Inpaint dialog on its source with its mask
  // (`reuseInpaint`).
  if (pass != null && !gone && rec.mask) {
    button.textContent = compact ? (mismatch ? '⚠' : '↻') : `${mismatch ? '⚠ ' : ''}Reuse inpaint…`;
    button.title = mismatch ? `Warning: ${warning}`
      : `Open the Inpaint dialog on #${pass} with this inpaint's mask, strength and seed`;
    button.setAttribute('aria-label', `Reuse inpaint in the Inpaint dialog${mismatch ? ` (warning: ${warning})` : ''}`);
    button.dataset.tip = 'Reuse inpaint…';
    return;
  }
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
  const why = !gone ? '' : rec.refine_binned
    ? `; #${pass}, its source, is in the recycle bin - restore it to make the pass again`
    : `; #${pass}, its source, has been deleted, so the pass cannot be made again`;
  button.textContent = compact ? (mismatch ? '⚠' : '↻') : `${mismatch ? '⚠ ' : ''}Reuse settings`;
  button.title = mismatch ? `Warning: ${warning}${lost}${why}` : `Load these settings into the panel${why}`;
  button.setAttribute('aria-label', mismatch ? `Reuse settings (warning: ${warning}${lost}${why})` : 'Reuse settings');
  button.dataset.tip = mismatch ? 'Reuse - other model' : 'Reuse';
}

// A card's image title: the facts that tell it from the card beside it, a
// pass's source and strength among them - "(deleted)" where the source has
// gone, "(in the recycle bin)" where it waits there (goneWord), as the
// lightbox's facts say it.
function cardTitle(rec) {
  const pass = rec.refine != null
    ? ` · ${passWord(rec)} of #${rec.refine}${outpaintOf(rec)}${rec.refine_missing ? ` (${goneWord(rec)})` : ''} @ ${rec.strength}` : '';
  return `#${rec.id} · ${rec.width}×${rec.height} · seed ${rec.seed}${pass}\n${rec.prompt}`;
}

// What a pass is called - "outpaint", "inpaint" or "refine" - by what its
// row carries: an outpaint has a mask as well, built by the server.
function passWord(rec) {
  return rec.outpaint ? 'outpaint' : rec.mask ? 'inpaint' : 'refine';
}

// An outpaint's padding after its source, " +400/+0/+400/+0", else ''.
function outpaintOf(rec) {
  return rec.outpaint ? ` ${outpaintSidesText(rec.outpaint)}` : '';
}

// Where a pass's gone source went (`refine_missing`): "deleted", or "in the
// recycle bin" (`refine_binned`), from which a Restore brings it back.
function goneWord(rec) {
  return rec.refine_binned ? 'in the recycle bin' : 'deleted';
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
      mask: kept.reuseMask || null,
      outpaint: parseOutpaintKey(kept.reuseOutpaint),
      refine_missing: kept.reusePassGone === 'true',
      refine_binned: kept.reusePassBinned === 'true',
      reference: kept.reuseReference ? kept.reuseReference.split(',').map(Number) : [],
      lora: kept.reuseLora || null,
    }, kept.reuseCompact === 'true');
  });
  // More seeds… is held on another checkpoint's images the same way.
  syncMoreSeedsTools();
}
