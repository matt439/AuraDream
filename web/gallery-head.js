// AuraDream's page script: the gallery head's pointer-stable layout - its
// buttons that come and go with the filter and the picks, and the search
// giving way while the pointer rests on them.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The head's buttons that come and go with the filter - Re-render, Delete
// unstarred - stand after the search, Starred and All models, which the free
// space before the search keeps to the row's right, so one going would move
// every control before it along the row (the measurements:
// docs/design-notes.md, Re-render). While the pointer rests on the head's
// controls (`headPointer`, from the head's pointerover and leave in
// app.js, as `galleryPointer` is, never `:hover`) one that goes keeps its
// place, empty (`.kept`: hidden, its box kept), and one that comes waits;
// both are put right once the pointer leaves them (releaseHead). The footer
// row under them is not theirs: Bin there changes the head as a whole, and
// its way in and out of the bin is `now`. All models' count of the filter's
// matches elsewhere, " (+3)", is that tick's width, before them too, so its
// change waits with them (`headOthers`). What still changes there - the
// gallery's count, Re-render's label (syncRerender) - must not break the
// row either, and with both buttons shown the controls fill it at 1400. So
// while the pointer rests on a head whose controls are one row, the search
// gives way to them (`.held`: its basis and floor gone, so the row it is on
// never breaks), its right edge where it was.
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
// keep the rows they had (docs/design-notes.md, Re-render, "A pick keeps the
// head's rows"): Re-render and Delete unstarred side by side can take two
// rows where either alone takes one. So Delete unstarred stays where the
// head keeps its height with it, and gives Re-render its place where it does
// not - measured within the one task, so nothing is painted between: what
// fits is the widths' (the labels, the search, the window, Gallery beside),
// as `.held`'s one row is (pointHead). Where neither keeps it, Delete unstarred stays as
// it was. Only for a change of the picks: a filter typed, a tick or a job
// landing lays the head out as it always has, and the sweep is back the
// moment nothing is picked - where that keeps the rows. A filter typed while
// cards were picked keeps Delete unstarred given way, so emptying the
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
