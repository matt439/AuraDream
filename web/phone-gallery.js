// AuraDream's page script: the gallery's head on a phone (styles/phone-
// gallery.css draws it) - one row, the search, Filters, Select and ⋯ - and
// the two sheets those open from the screen's foot: Filters holding Starred,
// All models, the Runs picker, Sheet and the search's terms; ⋯ the picks'
// count, Clear and Picked, Re-render, Swap tags…, Refine…, More seeds…,
// Saved searches…, Bundles…, the card size and Fit or Crop, the recycle bin
// and its own buttons, and Delete unstarred last. Each is the head's own node, moved whole under PHONE_MEDIA
// with phone.js's phoneMove and home again off it - never a copy, so `els`,
// every sync and every hold keep the one node. A bulk action says in its own
// row what it would take (its scope, or why it is held). While Select is on,
// the selection bar holds some of them instead (phone-select.js). The cards'
// size on the phone is this file's too (phoneCardSize), and the recycle bin
// as a screen of its own: its name under a ‹ back, only its own rows in the
// sheets, Back leaving it (touch.js). The runs - Filters' Runs row, their
// sheet, the run chip and a run sheet's foot - are phone-runs.js's. And the
// empty gallery's first visit as a screen (phoneGalleryStart) and the place
// hung under the head while the page scrolls (syncGalleryPlace).
// Declarations only;
// phone.js calls initPhoneGallery from initPhone and syncPhoneGallery from
// syncPhoneHosts.
'use strict';

// The search's help badge, the `.help` after the search: moved into the
// Filters sheet's fold, so it is kept here rather than looked up by place.
let phoneSearchHelp = null;

// What each sheet holds, host by host, in the order drawn.
const galleryFilterPlaces = () => [
  ['gallery-filter-chips', [els.favOnly.closest('label'), $('all-models-label')]],
  ['gallery-filter-run', [els.runList]],
  ['gallery-filter-view', [els.sheetBtn]],
  ['search-help-fold', [phoneSearchHelp]],
];
const galleryMorePlaces = () => [
  ['gallery-more-picks', [els.selectCount, els.selectClear, els.picked]],
  ['gallery-more-act', [els.rerender, els.swap, els.refineBatch, els.moreSeeds]],
  ['gallery-more-library', [els.savedSearches, els.bundles, els.bin]],
  ['phone-fit-row', [$('card-fit')]],
  ['gallery-more-bin', [els.restorePicked, els.purgePicked, els.binAge, els.emptyBin]],
  ['gallery-more-danger', [els.deleteUnstarred]],
];
const phoneGalleryMoved = () => [
  ...[...galleryFilterPlaces(), ...galleryMorePlaces()].flatMap(([, nodes]) => nodes), els.select,
];

// The rows that say under their name what they take: a bulk action's scope
// (the picks, else what the filter matches), or the reason it is held.
const gallerySubRows = () => [
  els.pickedBtn, els.swap, els.refineBatch, els.moreSeeds, els.savedSearches, els.bundles, els.bin,
  els.restorePicked, els.purgePicked, els.emptyBin, els.deleteUnstarred, els.sheetBtn,
];

// From syncPhoneHosts: at boot and at every change of tier.
function syncPhoneGallery() {
  const head = els.search.closest('.gallery-head');
  if (isPhone()) {
    for (const [id, nodes] of [...galleryFilterPlaces(), ...galleryMorePlaces()]) {
      const host = $(id);
      for (const node of nodes) phoneMove(node, host);
    }
    phoneMove(els.select, head, $('btn-gallery-more'));
  } else {
    for (const sheet of gallerySheets()) if (sheet.open) sheet.close();
    for (const node of [els.galleryError, ...phoneGalleryMoved()]) phoneHome(node);
    for (const row of gallerySubRows()) row.querySelector(':scope > .row-sub')?.remove();
    head.removeAttribute('data-bin');
    for (const sheet of gallerySheets()) sheet.removeAttribute('data-bin');
    $('btn-bin-back').hidden = true;
    // The runs' rows, chip and foot put away (phone-runs.js).
    syncPhoneRuns();
    delete $('gallery-place').dataset.on;
  }
  syncPhoneGalleryHead();
  // An empty gallery's words are the tier's: a phone's screen, a PC's line.
  if (!els.galleryEmpty.hidden) syncGalleryEmpty();
}

// The empty gallery's first visit on a phone (syncGalleryEmpty), a screen
// of its own under the head rather than a line with a button in its middle:
// what the gallery is, Create an image its one solid button - the Create
// view, the prompt given the focus - then the authors' examples (the first
// run's own Examples, viewerExamples) and, while the All models tick is off,
// the images on other checkpoints with a button that ticks it. A search or
// Starred that matched nothing, and the bin, keep their lines.
function phoneGalleryStart(elsewhere) {
  const box = els.galleryEmpty;
  box.dataset.start = '';
  const piece = (tag, className, text) => {
    const el = document.createElement(tag);
    el.className = className;
    el.textContent = text;
    return el;
  };
  const press = (className, text, title, onPress) => {
    const button = piece('button', className, text);
    button.type = 'button';
    button.title = title;
    button.addEventListener('click', onPress);
    return button;
  };
  const glyph = document.createElement('span');
  glyph.className = 'gallery-start-glyph';
  glyph.setAttribute('aria-hidden', 'true');
  glyph.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 2.5l2.2 6.3 6.3 2.2-6.3 2.2L12 19.5l-2.2-6.3L3.5 11l6.3-2.2z"/></svg>';
  const nodes = [
    glyph,
    piece('span', 'gallery-start-title', 'No images yet'),
    piece('span', 'gallery-start-note', 'What you make is kept here, newest first.'),
    press('primary gallery-start-create', 'Create an image', 'Write a prompt on Create', () => {
      showCreateView();
      els.prompt.focus();
    }),
    press('ghost gallery-start-examples', 'Start from an example', 'Example prompts for this checkpoint, from its authors', viewerExamples),
  ];
  if (elsewhere) {
    nodes.push(
      piece('span', 'gallery-start-others', `${plural(elsewhere, 'image')} on other checkpoints.`),
      press('ghost gallery-start-all', 'Show all models', "Tick All models: every checkpoint's images", () => {
        if (!els.allModels.checked) els.allModels.click();
      }),
    );
  }
  box.replaceChildren(...nodes);
}

// Where in the gallery the cards under the head are (P28), on a phone: an
// endless page of pictures three across said nothing of how far down it a
// finger had come, the day's lines scrolling away with the cards. Hung under
// the sticky head while the page scrolls - the day of the top card ("Today",
// "Yesterday", "Thu 5 Nov") and its place in what the filter matches ("52
// of 160"; the bin's place only, its cards' days not when they were binned)
// - and gone a moment after it stops. A picture over the cards, which a
// press goes through (styles/phone-gallery.css): nothing moves, nothing
// scrolls. Not over the top of the page, nor a run's sheet.
let galleryPlaceFrame = 0;
let galleryPlaceTimer = 0;

function queueGalleryPlace() {
  if (galleryPlaceFrame || !isPhone()) return;
  galleryPlaceFrame = requestAnimationFrame(() => { galleryPlaceFrame = 0; syncGalleryPlace(); });
}

// The first of the grid's children still in sight under `line`: the grid's
// rows run down the page in the children's order, so a halving search.
function galleryChildAt(line) {
  const kids = els.gallery.children;
  let lo = 0;
  let hi = kids.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (kids[mid].getBoundingClientRect().bottom > line) hi = mid;
    else lo = mid + 1;
  }
  return kids[lo] ?? null;
}

function placeDay(made) {
  const day = new Date(made);
  const now = new Date();
  if (day.toDateString() === now.toDateString()) return 'Today';
  if (day.toDateString() === new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).toDateString()) return 'Yesterday';
  return day.toLocaleDateString([], {
    weekday: 'short', day: 'numeric', month: 'short',
    ...(day.getFullYear() === now.getFullYear() ? {} : { year: 'numeric' }),
  });
}

// The phone names the newest group too: without its own heading, the first
// older day's line looked as though it named the newest pictures above it.
// Keep this one extra heading here, so paging, finished-image prepends and
// tombstones retain the gallery's own date groups, and a mouse gets its
// original top group back. Bin groups still name creation dates (the order
// and date: search are by those); say so beside their separate binned ages.
let phoneFirstDay = null;
function syncPhoneGalleryDays() {
  const first = els.gallery.firstElementChild;
  const top = first && first === phoneFirstDay ? first.nextElementSibling : first;
  // A finished-image prepend treats the existing leading heading as the
  // old group's boundary. If it is now a different day, leave that heading
  // there as an ordinary boundary instead of moving it away with the phone's
  // new first-group heading (a page left open across midnight).
  const before = phoneFirstDay?.previousElementSibling;
  if (phoneFirstDay?.isConnected && before?.dataset.made
      && localDay({ created_at: phoneFirstDay.dataset.phoneMade }) !== localDay({ created_at: before.dataset.made })) {
    phoneFirstDay.classList.remove('phone-first-day');
    phoneFirstDay.dataset.phoneDay = dayBreak({ created_at: phoneFirstDay.dataset.phoneMade }).textContent;
    phoneFirstDay = null;
  }
  if (isPhone() && !sheetMode && top?.dataset.made) {
    if (!phoneFirstDay) {
      phoneFirstDay = document.createElement('h3');
      phoneFirstDay.className = 'gallery-break phone-first-day';
    }
    const rec = { created_at: top.dataset.made };
    const words = `${isToday(rec) ? 'Today · ' : ''}${dayBreak(rec).textContent}`;
    phoneFirstDay.dataset.phoneMade = top.dataset.made;
    if (phoneFirstDay.dataset.phoneDay !== words) phoneFirstDay.dataset.phoneDay = words;
    if (top.previousElementSibling !== phoneFirstDay) top.before(phoneFirstDay);
  } else {
    phoneFirstDay?.remove();
  }
  for (const heading of els.gallery.querySelectorAll(':scope > .gallery-break')) {
    if (!('phoneDay' in heading.dataset)) heading.dataset.phoneDay = heading.textContent;
    const words = `${isPhone() && binMode ? 'Created ' : ''}${heading.dataset.phoneDay}`;
    if (heading.textContent !== words) heading.textContent = words;
  }
}

function syncGalleryPlace() {
  const label = $('gallery-place');
  const line = els.search.closest('.gallery-head').getBoundingClientRect().bottom;
  const first = els.gallery.firstElementChild;
  let at = isPhone() && phoneView === 'gallery' && !sheetMode && isShown(els.gallery)
    && first && first.getBoundingClientRect().top < line - 1 ? galleryChildAt(line) : null;
  while (at && !at.matches('.card[data-id]')) at = at.nextElementSibling;
  const index = at ? galleryRecords.findIndex((rec) => String(rec.id) === at.dataset.id) : -1;
  clearTimeout(galleryPlaceTimer);
  delete label.dataset.crossing;
  if (index < 0) {
    delete label.dataset.on;
    return;
  }
  const place = `${index + 1} of ${Math.max(total, galleryRecords.length)}`;
  const words = binMode || !at.dataset.made ? place : `${placeDay(at.dataset.made)} · ${place}`;
  if (label.textContent !== words) label.textContent = words;
  // A day line crossing under the fixed pill otherwise shows two different
  // dates on top of each other. Hide the pill immediately, with its box kept
  // for the next scroll's geometry; the heading stays readable and no card
  // moves. The pill returns once the line has passed it.
  const box = label.getBoundingClientRect();
  const crossing = [...els.gallery.querySelectorAll(':scope > .gallery-break')].some((heading) => {
    const day = heading.getBoundingClientRect();
    return day.bottom > box.top && day.top < box.bottom;
  });
  if (crossing) {
    delete label.dataset.on;
    label.dataset.crossing = '';
    return;
  }
  label.dataset.on = '';
  galleryPlaceTimer = setTimeout(() => { delete label.dataset.on; }, 1500);
}

// The gallery's sheets on the phone: Filters, ⋯ and the runs'.
const gallerySheets = () => [$('gallery-filter-sheet'), $('gallery-more-sheet'), $('runs-sheet')];

let phoneGallerySyncQueued = false;
function queuePhoneGallerySync() {
  if (phoneGallerySyncQueued) return;
  phoneGallerySyncQueued = true;
  queueMicrotask(() => { phoneGallerySyncQueued = false; syncPhoneGalleryHead(); });
}

// The head's state as the phone draws it: the search's words (the count it
// searches), Filters' badge (how many of Starred and All models are on),
// ⋯'s (how many images are picked), the bin's title row, each sheet's rows'
// lines, and the Filters sheet's count. The page's own writes are watched
// (initPhoneGallery), so this follows whatever changed them.
function syncPhoneGalleryHead() {
  // The selection bar (phone-select.js) first: what it holds is out of the
  // sheets, and the bin's rows are placed after it.
  syncPhoneSelectBar();
  syncPhoneGalleryDays();
  if (!isPhone()) return;
  syncSearchPlaceholder();
  const head = els.search.closest('.gallery-head');
  head.toggleAttribute('data-bin', binMode);
  // The bin's own screen: its name under a ‹ back, the sheets holding only
  // its own rows (styles/phone-gallery.css, `[data-bin]`).
  for (const sheet of gallerySheets()) sheet.toggleAttribute('data-bin', binMode);
  const moreTitle = binMode ? 'Bin actions' : 'Gallery actions';
  if ($('gallery-more-title').textContent !== moreTitle) $('gallery-more-title').textContent = moreTitle;
  $('btn-bin-back').hidden = !binMode;
  placeBinGroup();
  syncPhoneRuns();
  const on = [els.favOnly.checked && !binMode ? 'Starred' : '', els.allModels.checked && !binMode ? 'all models' : ''].filter(Boolean);
  const filters = $('btn-gallery-filters');
  badge($('gallery-filters-count'), on.length);
  filters.classList.toggle('on', on.length > 0);
  filters.setAttribute('aria-label', on.length ? `Filters: ${on.join(', ')}` : 'Filters');
  const n = selection.size;
  badge($('gallery-more-count'), n);
  $('btn-gallery-more').setAttribute('aria-label', n ? `More gallery actions: ${n} picked` : 'More gallery actions');
  const count = $('gallery-filter-count');
  const words = galleryShown();
  if (count.textContent !== words) count.textContent = words;
  const scope = $('gallery-more-scope');
  const target = galleryActionTarget('to choose the images', 'one press');
  const said = target.scope ? `On ${target.words}` : phoneWords(target.why);
  if (scope.textContent !== said) scope.textContent = said;
  scope.classList.toggle('held', !target.scope);
  for (const row of gallerySubRows()) syncGallerySub(row, target);
  // The card size and Fit or Crop are the cards': a run's sheet draws its
  // cells its own way (as #card-view is hidden there off the phone).
  $('gallery-more-view').hidden = sheetMode;
}

// A held reason in a phone's words: the bin is left by the ‹ before its
// name. (How
// cards are picked is already a touch screen's, gallery-select.js's
// pickCardsWords, in the title and in what a held press says.)
const phoneWords = (why) => (why || '').replace('leave it with Bin', 'leave it with ‹ beside its name');

// In the recycle bin its own rows come first in ⋯ - Restore picked,
// Delete picked permanently, which images Empty takes and Empty recycle
// bin - and the gallery's are not drawn (Recycle bin itself is the ‹ by
// the bin's name); out of it, the bin's rows before Delete unstarred.
// The group moves, never while the sheet is open (the bin is entered by
// the sheet's own row, which shuts it first).
function placeBinGroup() {
  const sheet = $('gallery-more-sheet');
  if (sheet.open) return;
  const group = $('gallery-more-bin');
  const body = group.parentNode;
  const before = binMode ? body.firstElementChild : $('gallery-more-danger');
  if (group !== before && group.nextElementSibling !== before) body.insertBefore(group, before);
  phoneMove(els.bin, $('gallery-more-library'));
  // Empty's choice's caption, straight over it.
  if (els.binAge.parentNode === group) phoneMove($('bin-age-caption'), group, els.binAge);
}

function badge(el, n) {
  el.hidden = !n;
  const text = n > 99 ? '99+' : String(n);
  if (el.textContent !== text) el.textContent = text;
}

// The gallery's count (`total`, which the head's title says as "(9)"), in
// words: what the filters show.
function galleryShown() {
  return binMode ? `${plural(total, 'image')} in the bin` : plural(total, 'image');
}

// The search's placeholder on the phone: what it searches, the count the
// head's title would say ("Search 9 images"); the touch tier's own words
// elsewhere (touch.js, syncTier).
function phoneSearchWords() {
  if (binMode) return 'Search the bin';
  return total > 0 ? `Search ${plural(total, 'image')}` : 'Search prompts';
}

// A row's second line: the held reason (its title, the press's own words),
// else what it would take. Swap tags…, Refine… and More seeds… share their
// scope, said once over them (#gallery-more-scope): each line says what the
// action does to it, or a reason of its own (no refine pass on this
// checkpoint) where it is held for one.
function syncGallerySub(row, target) {
  let sub = row.querySelector(':scope > .row-sub');
  if (!sub) {
    sub = document.createElement('span');
    sub.className = 'row-sub';
    row.append(sub);
  }
  const words = gallerySubText(row, target);
  if (sub.textContent !== words) sub.textContent = words;
  sub.hidden = !words;
}

const GALLERY_ACT_WORDS = new Map();

function gallerySubText(row, target) {
  const n = selection.size;
  const them = n === 1 ? 'the picked image' : `the ${n} picked images`;
  const held = row.getAttribute('aria-disabled') === 'true';
  if (GALLERY_ACT_WORDS.size === 0) {
    GALLERY_ACT_WORDS.set(els.swap, 'Run them again with some tags swapped');
    GALLERY_ACT_WORDS.set(els.refineBatch, 'A refine pass of each, at one strength and scale');
    GALLERY_ACT_WORDS.set(els.moreSeeds, 'More images of each batch, the seeds following on');
  }
  if (GALLERY_ACT_WORDS.has(row)) {
    // Held for a reason of its own: Refine… on a checkpoint with no refine
    // pass. Held for the shared one, the heading says it.
    const own = held && (target.scope || (row === els.refineBatch && !refineSupported));
    return own ? phoneWords(row.title) : GALLERY_ACT_WORDS.get(row);
  }
  if (held) {
    if (row === els.pickedBtn && !binMode && !n) return 'Pick images with Select first';
    return phoneWords(row.title);
  }
  switch (row) {
    case els.pickedBtn: return `Star, compare, re-render or bin ${them}`;
    case els.restorePicked: return `Back to the gallery: ${them}`;
    case els.purgePicked: return `From disk, after a question: ${them}`;
    case els.emptyBin: return els.binAge.value
      ? `From disk: those binned over ${els.binAge.value} days ago, after a question`
      : 'From disk: everything in it, after a question';
    case els.deleteUnstarred: return 'To the recycle bin: every unstarred image the filter matches, after a question';
    case els.bin: return binMode ? 'Back to the gallery'
      : binCount ? `${plural(binCount, 'image')}: restore them, or empty it` : 'Empty';
    case els.savedSearches: return 'Save, manage and apply named filters';
    case els.bundles: return 'Back up, merge, or bundle the gallery and clear it';
    case els.sheetBtn: return sheetMode ? 'Back to the cards' : 'These images laid out by their settings';
    default: return '';
  }
}

// --------------------------------------------------------------- the sheets

// A sheet opens from its button, the gallery's line (#gallery-error) moved
// into its foot, so a held press there says why in the sheet; it closes by
// its ✕, Escape, Back, a press on the dimmed page above it, or the press of
// one of its actions. Shut without an action, a held press's words go with
// it; an action's press takes the line home first, so what the action says
// after it (a refusal, a sweep's outcome) is the gallery's to show.
let gallerySheetSaid = null;
function openGallerySheet(sheet) {
  if (!isPhone() || sheet.open) return;
  gallerySheetSaid = els.galleryError.hidden ? '' : els.galleryError.textContent;
  syncPhoneGalleryHead();
  sheet.showModal();
  placeGalleryLine();
  sheet.scrollTop = 0;
}

function gallerySheetClosed() {
  if (gallerySheets().some((s) => s.open)) return;
  const said = els.galleryError.hidden ? '' : els.galleryError.textContent;
  if (gallerySheetSaid !== null && said && said !== gallerySheetSaid) showError('', false, { slot: els.galleryError, mirror: false });
  gallerySheetSaid = null;
  placeGalleryLine();
  // What the selection bar holds waited for the sheet (phone-select.js).
  queuePhoneGallerySync();
}

// An action pressed in a sheet: the line home, then the sheet shut, before
// the action's own handler runs (the press's capture).
function closeGallerySheetFor(sheet) {
  gallerySheetSaid = null;
  placeGalleryLine(sheet);
  sheet.close();
}

// Where the gallery's line is on the phone: in the foot of an open sheet
// (`closing` aside; the top one), else at the selection bar's top while it
// is up, else over a run sheet's foot while that is up, else under the
// head.
function placeGalleryLine(closing = null) {
  const open = gallerySheets().filter((s) => s.open && s !== closing).at(-1);
  if (open) phoneMove(els.galleryError, open.querySelector('.gallery-sheet-foot'));
  else if (isPhone() && !$('phone-select-bar').hidden) phoneMove(els.galleryError, $('phone-select-bar'), $('select-bar-head'));
  else if (isPhone() && !$('sheet-foot').hidden) phoneMove(els.galleryError, $('sheet-foot'), $('sheet-foot').firstChild);
  else phoneHome(els.galleryError);
}

// A press on the dimmed page beside the sheet closes it (its box is the
// dialog's; the backdrop's press is the dialog's own, outside its box).
function gallerySheetBackdrop(e) {
  const sheet = e.currentTarget;
  if (e.target !== sheet) return;
  const r = sheet.getBoundingClientRect();
  if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) sheet.close();
}

// ⋯'s rows: an action's press closes the sheet first, in the same turn, so
// the dialog it opens (Swap tags, Bundles, a question) or the view it
// changes (the bin, a comparison) is what the screen shows, and Back's one
// entry goes with it. Picked opens its rows inside the sheet; a held row
// keeps the sheet open, its reason in the foot.
function galleryMorePress(e) {
  const sheet = $('gallery-more-sheet');
  const control = e.target.closest('button, [role="menuitem"]');
  if (!control || !sheet.contains(control) || control.closest('.gallery-sheet-head')) return;
  if (control === els.pickedBtn || control.getAttribute('aria-disabled') === 'true') return;
  // The card size and Fit or Crop: settings, the sheet left open.
  if (control.closest('#gallery-more-view')) return;
  if (control.matches('[role="menuitem"]') && els.pickedBtn.getAttribute('aria-disabled') === 'true') return;
  closeGallerySheetFor(sheet);
}

// Filters' rows: Starred and All models leave it open (the count in its
// title follows them); Sheet and a run's choice show the gallery they make.
function galleryFilterPress(e) {
  const sheet = $('gallery-filter-sheet');
  if (e.target.closest('button') === els.sheetBtn && els.sheetBtn.getAttribute('aria-disabled') !== 'true') closeGallerySheetFor(sheet);
}

// ---------------------------------------------------------- the card size

// The cards on a phone: three across the screen upright and six on its side
// ('s', the default), or two and four ('l') - each filling its cell, 4 px
// apart (styles/phone-gallery.css). The whole picture or the card filled is
// #card-view's Fit or Crop, its pair moved into ⋯ on the phone; the PC's S
// and M are a wide window's and change nothing here. This browser's choice,
// kept apart from the PC's (auradream.gallery.v1), every access in
// try/catch (remembered/remember).
const PHONE_CARD_KEY = 'auradream.gallery.phone.v1';
const PHONE_CARD_SIZES = ['s', 'l'];
let phoneCardSize = 's';

function restorePhoneCardSize() {
  const stored = remembered(PHONE_CARD_KEY);
  if (PHONE_CARD_SIZES.includes(stored)) phoneCardSize = stored;
  syncPhoneCardSize();
}

function setPhoneCardSize(size) {
  if (!PHONE_CARD_SIZES.includes(size)) return;
  phoneCardSize = size;
  remember(PHONE_CARD_KEY, size);
  syncPhoneCardSize();
}

// The grid's data attribute, which the phone's stylesheet reads (a PC's
// reads none), and the pair as #card-view's are: the chosen one pressed and
// the pair's one tab stop.
function syncPhoneCardSize() {
  els.gallery.dataset.phoneSize = phoneCardSize;
  for (const button of $('phone-card-size').querySelectorAll('button')) {
    const on = button.dataset.phoneSize === phoneCardSize;
    button.setAttribute('aria-pressed', String(on));
    button.tabIndex = on ? 0 : -1;
  }
}

// A press in the sheet's card rows: the size, or Fit or Crop (whose own
// listener is #card-view's, which the moved pair has left). The sheet stays
// open, the cards changing behind it.
function galleryViewPress(e) {
  const button = e.target.closest('button');
  if (!button) return;
  if ('phoneSize' in button.dataset) setPhoneCardSize(button.dataset.phoneSize);
  else if ($('card-fit').contains(button)) cardViewClick(e);
}

function initPhoneGallery() {
  phoneSearchHelp = els.search.parentNode.querySelector(':scope > .help');
  for (const node of [els.galleryError, ...phoneGalleryMoved()]) {
    const mark = document.createComment(` ${node.id || node.className}'s place off the phone (phone-gallery.js) `);
    node.before(mark);
    phoneHomes.set(node, mark);
  }
  const filterSheet = $('gallery-filter-sheet');
  const moreSheet = $('gallery-more-sheet');
  $('btn-gallery-filters').addEventListener('click', () => openGallerySheet(filterSheet));
  $('btn-gallery-more').addEventListener('click', () => openGallerySheet(moreSheet));
  $('gallery-filter-close').addEventListener('click', () => filterSheet.close());
  $('gallery-more-close').addEventListener('click', () => moreSheet.close());
  for (const sheet of [filterSheet, moreSheet]) {
    sheet.addEventListener('close', gallerySheetClosed);
    sheet.addEventListener('click', gallerySheetBackdrop);
  }
  moreSheet.addEventListener('click', galleryMorePress, true);
  $('gallery-more-view').addEventListener('click', galleryViewPress);
  $('phone-card-size').addEventListener('keydown', (e) => stepPreset($('phone-card-size'), e));
  restorePhoneCardSize();
  // The selection bar (phone-select.js), once the homes above are marked,
  // and the runs (phone-runs.js).
  initPhoneSelect();
  initPhoneRuns();
  // The bin's ‹: the gallery again, as Recycle bin's own press leaves it.
  $('btn-bin-back').addEventListener('click', () => { if (binMode) toggleBin(); });
  filterSheet.addEventListener('click', galleryFilterPress, true);
  els.runList.addEventListener('change', () => { if (filterSheet.open && els.runList.value) closeGallerySheetFor(filterSheet); });
  // The gallery's line under the head, over the cards: a tap puts it away.
  els.galleryError.addEventListener('click', () => {
    if (isPhone() && !filterSheet.contains(els.galleryError) && !moreSheet.contains(els.galleryError)) {
      showError('', false, { slot: els.galleryError, mirror: false });
    }
  });
  // What the head and the rows read, as the page's own code changes it.
  const watch = new MutationObserver(queuePhoneGallerySync);
  watch.observe(els.count, { childList: true, characterData: true, subtree: true });
  watch.observe(els.gallery, { childList: true });
  for (const row of gallerySubRows()) {
    watch.observe(row, { attributes: true, attributeFilter: ['title', 'aria-disabled', 'aria-pressed', 'aria-label', 'class'] });
  }
  for (const el of [els.favOnly, els.allModels, els.binAge]) el.addEventListener('change', queuePhoneGallerySync);
  onSelectionChange(queuePhoneGallerySync);
  addEventListener('scroll', queueGalleryPlace, { passive: true });
}
