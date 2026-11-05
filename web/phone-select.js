// AuraDream's page script: picking gallery images on a phone (styles/phone-
// gallery.css draws it). While Select is on in the Gallery view, a bar in
// the tab bar's place holds what the picks take, in the thumb's reach: Done
// (Select off, the picks kept, as Select's own press), the picks' count,
// Clear and Select all over a row of Star or Unstar and Bin (the Picked
// menu's own items, pressed for you), Refine…, More seeds… and Swap tags…
// (the footer's own buttons, moved in while the bar is up and back into ⋯'s
// sheet after) and ⋯, that sheet; in the recycle bin, Restore picked and
// Delete picked permanently. The gallery's line is at the bar's top, so a
// held press says why where it was made. Back leaves Select (touch.js's
// layers). A long press's menu has Select too (context-menu.js,
// gallery-select.js's selectFromCard). Declarations only; phone-gallery.js's
// syncPhoneGalleryHead calls syncPhoneSelectBar, initPhoneGallery
// initPhoneSelect.
'use strict';

// The bar is up: on the phone, in the Gallery view, Select on, no run's
// sheet over the cards (Select is held there).
const phoneSelecting = () => isPhone() && selectMode && !sheetMode && phoneView === 'gallery';

// What the bar holds while it is up, by where in it: the count and Clear
// between Done and Select all; the actions before Bin (the gallery's) or
// alone (the bin's).
const selectBarHeadNodes = () => [els.selectCount, els.selectClear];
const selectBarGalleryActs = () => [els.refineBatch, els.moreSeeds, els.swap];
const selectBarBinActs = () => [els.restorePicked, els.purgePicked];

// The nodes the bar holds now: none while it is down.
function selectBarHolds() {
  if (!phoneSelecting()) return new Set();
  return new Set([...selectBarHeadNodes(), ...(binMode ? selectBarBinActs() : selectBarGalleryActs())]);
}

// From syncPhoneGalleryHead (a change of the selection, of Select, of the
// bin or of the tier, a page landing) and setPhoneView: the bar up or down,
// `body[data-selecting]` with it (the stylesheet's hook for the tab bar's
// place, the page's foot and --phone-bottom-chrome), its nodes moved in or
// back into ⋯'s sheet, and its own buttons' state. Nothing moves while a
// gallery sheet is open: the sheet holds some of them, and nothing there
// changes Select.
let phoneSelectReady = false;
function syncPhoneSelectBar() {
  // Not before the moved nodes' homes are marked (initPhoneGallery).
  if (!phoneSelectReady) return;
  const bar = $('phone-select-bar');
  const up = phoneSelecting();
  // Done (or a key) pressed in the bar: the focus to Select, not <body>.
  if (!up && !bar.hidden && bar.contains(document.activeElement) && isShown(els.select)) els.select.focus({ preventScroll: true });
  if (up) document.body.dataset.selecting = '';
  else delete document.body.dataset.selecting;
  bar.hidden = !up;
  bar.toggleAttribute('data-bin', binMode);
  if (!isPhone()) return;
  if ($('gallery-filter-sheet').open || $('gallery-more-sheet').open) return;
  const holds = selectBarHolds();
  // Back into ⋯'s sheet first, so the order the sheet keeps is its own.
  for (const node of [...selectBarHeadNodes(), ...selectBarGalleryActs(), ...selectBarBinActs()]) {
    if (!holds.has(node)) placeInGallerySheet(node);
  }
  if (up) {
    for (const node of selectBarHeadNodes()) phoneMove(node, $('select-bar-head'), $('select-all'));
    const acts = $('select-bar-acts');
    if (binMode) for (const node of selectBarBinActs()) phoneMove(node, acts, $('select-star'));
    else for (const node of selectBarGalleryActs()) phoneMove(node, acts, $('select-bin'));
    syncSelectBarButtons();
  }
  // The line with it: up, at its top; down, where it was, or what it said
  // there goes with the bar.
  const inBar = els.galleryError.parentNode === bar;
  placeGalleryLine();
  if (inBar && !up && !els.galleryError.hidden) showError('', false, { slot: els.galleryError, mirror: false });
}

// A node back in its place in ⋯'s sheet (phone-gallery.js, galleryMorePlaces):
// before the next of its group's nodes the group holds, else at its end.
function placeInGallerySheet(node) {
  for (const [id, nodes] of galleryMorePlaces()) {
    const at = nodes.indexOf(node);
    if (at < 0) continue;
    const host = $(id);
    if (node.parentNode === host) return;
    phoneMove(node, host, nodes.slice(at + 1).find((n) => n.parentNode === host) ?? null);
    return;
  }
}

// Star or Unstar (Unstar where every pick is starred already) and Bin: the
// Picked menu's items, each held and titled as that item is now, so a held
// press says the item's own reason. The gallery's items: hidden in the bin,
// whose picks are Restore picked's.
function syncSelectBarButtons() {
  const records = selectedRecords();
  const unstar = records.length > 0 && records.every((r) => r.favorite);
  mirrorPickedItem($('select-star'), unstar ? 'unstar' : 'star', unstar ? 'Unstar' : 'Star');
  mirrorPickedItem($('select-bin'), 'bin', 'Bin');
  for (const id of ['select-star', 'select-bin', 'select-more']) $(id).hidden = binMode;
}

function mirrorPickedItem(button, act, word) {
  const item = els.pickedMenu.querySelector(`[data-picked="${act}"]`);
  button.dataset.pickedAct = act;
  const held = item.getAttribute('aria-disabled') === 'true';
  if (button.getAttribute('aria-disabled') !== String(held)) button.setAttribute('aria-disabled', String(held));
  if (button.title !== item.title) button.title = item.title;
  const label = button.querySelector('.select-act-word');
  if (label.textContent !== word) label.textContent = word;
}

let selectBarSyncQueued = false;
function queueSelectBarSync() {
  if (selectBarSyncQueued) return;
  selectBarSyncQueued = true;
  queueMicrotask(() => {
    selectBarSyncQueued = false;
    if (phoneSelecting()) syncSelectBarButtons();
  });
}

function initPhoneSelect() {
  phoneSelectReady = true;
  $('select-done').addEventListener('click', () => setSelectMode(false));
  $('select-all').addEventListener('click', pickLoadedCards);
  for (const id of ['select-star', 'select-bin']) {
    $(id).addEventListener('click', (e) => {
      pickedChoose(els.pickedMenu.querySelector(`[data-picked="${e.currentTarget.dataset.pickedAct}"]`));
    });
  }
  $('select-more').addEventListener('click', () => openGallerySheet($('gallery-more-sheet')));
  // The Picked items' holds and titles as the page's own code changes them
  // (a star's write pending, the picks changing), and the stars they read.
  const watch = new MutationObserver(queueSelectBarSync);
  watch.observe(els.pickedMenu, { subtree: true, attributes: true, attributeFilter: ['aria-disabled', 'title'] });
}
