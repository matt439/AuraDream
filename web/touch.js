// AuraDream's page script: the phone's two views (styles/touch.css has the
// layouts; state.js the media queries they share) and, on any touch screen,
// Back closing the top overlay; phone.js has the phone's own hosts, which
// syncTier places. Declarations only; app.js calls initTouchLayout() once
// at boot.
'use strict';

const PHONE_VIEW_KEY = 'auradream.phoneView';
// The view the phone shows ('create' or 'gallery') and each view's own
// scroll, so a switch back lands where the user left it.
let phoneView = 'create';
const phoneViewScroll = { create: 0, gallery: 0 };

const phoneQuery = matchMedia(PHONE_MEDIA);
const touchQuery = matchMedia(TOUCH_MEDIA);
const isPhone = () => phoneQuery.matches;

// The tier, as `body[data-tier]`: 'phone' (PHONE_MEDIA), 'tablet' (the rest
// of the touch tier: the iPad's two columns either way up) or none (a PC,
// whatever its window). A hook for the phone's rules to key on beside the
// media query (styles/touch.css), set at boot and again whenever a query's
// answer changes - an iPad's Split View pane dragged across 700 px, a touch
// screen's pointer changing - so the page never keeps the tier it opened in.
// Undefined until boot's sync.
let pageTier;
const pageTierNow = () => (isPhone() ? 'phone' : touchQuery.matches ? 'tablet' : null);
// The search's placeholder as index.html has it, "/ to focus" and all.
let searchPlaceholder = '';

function syncTier() {
  const was = pageTier;
  pageTier = pageTierNow();
  if (pageTier) document.body.dataset.tier = pageTier;
  else delete document.body.dataset.tier;
  syncSearchPlaceholder();
  if (was === pageTier) return;
  if (pageTier !== 'phone') els.error.classList.remove('at-generate');
  // The phone's hosts (phone.js): Generate into the action bar and the
  // results under the settings, or home again and the queue's sheet shut.
  syncPhoneHosts();
  // The held titles that say how to pick cards (gallery-select.js,
  // pickCardsWords) are a touch screen's or a mouse's: re-read them.
  if (was !== undefined) syncSelection();
  // Back on the phone, the view shows where it was left, as a tab's press
  // does: the page between was the whole of both views, scrolled elsewhere.
  // The right-click menu is docked at the phone's foot and placed at the
  // pointer elsewhere: an open one closes as the phone comes or goes.
  if (was !== undefined && (was === 'phone' || pageTier === 'phone')) closeContextMenu(false);
  if (was !== undefined && pageTier === 'phone') restorePhoneScroll(phoneViewScroll[phoneView]);
  // The bar's state and the keyboard's hold read the page as it is now.
  syncPhoneTabsBusy();
  syncTyping();
}

// The search's placeholder: "/ to focus" is a keyboard's; on a phone it
// says the count it searches, the head's title being out of sight there
// (phone-gallery.js, phoneSearchWords).
function syncSearchPlaceholder() {
  const words = isPhone() ? phoneSearchWords() : touchQuery.matches ? 'Search prompts' : searchPlaceholder;
  if (els.search.placeholder !== words) els.search.placeholder = words;
}

// Show one of the phone's views. Only the user's press, or an action of
// theirs that belongs to the other view (Reuse writes the panel), calls this:
// nothing switches by itself when a job starts or ends.
function setPhoneView(view, { remember = true } = {}) {
  if (view !== 'create' && view !== 'gallery') return;
  if (view !== phoneView && isPhone()) phoneViewScroll[phoneView] = scrollY;
  const changed = view !== phoneView;
  phoneView = view;
  document.body.dataset.view = view;
  for (const tab of document.querySelectorAll('#phone-tabs [data-view]')) {
    const on = tab.dataset.view === view;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
  }
  if (remember) {
    try { sessionStorage.setItem(PHONE_VIEW_KEY, view); } catch { /* storage off: the view is this page's */ }
  }
  if (changed && isPhone()) scrollTo({ top: phoneViewScroll[view], behavior: 'instant' });
  // The gallery's selection bar is the Gallery view's (phone-select.js).
  if (changed) queuePhoneGallerySync();
}

// A press of the tab already chosen: its view back at its top, as a phone's
// own apps have it - the Gallery's newest cards, Create's prompt. The
// user's press, so the page's one scroll of its own a tab makes; smooth
// unless the system asks for less motion. The focus stays on the tab.
function phoneViewToTop() {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  scrollTo({ top: 0, behavior: reduce ? 'instant' : 'smooth' });
}

// A refusal of Generate's press (pressGenerate), on the phone: the panel's
// line, under Generate's own place in the panel, was a screen below the
// button riding on the tab bar, and the press seemed to do nothing. The
// line, the same node, rides above Generate instead (styles/touch.css),
// until its next write (showError) or a tap on it, which leaves it in its
// place; a press of a button inside it is the button's. Not a PC's or a
// tablet's: there the line is beside the button pressed.
function sayAtGenerate() {
  if (isPhone() && !els.error.hidden && els.error.textContent) els.error.classList.add('at-generate');
}

// Reuse, Load settings and the like write the panel: on a phone looking at
// the gallery, the panel is the other view, so they bring it.
function showCreateView() {
  if (isPhone() && phoneView !== 'create') setPhoneView('create');
}

// The running job, on the tab bar: Create's dot and its words in the tab's
// name, and the topbar's line copied along the bar's top edge
// (job-progress.js writes the topbar's).
function syncPhoneTabsBusy() {
  const line = $('phone-tabs-line');
  const fill = $('phone-tabs-fill');
  const busy = !els.topbarLine.hidden;
  $('tab-busy').hidden = !busy;
  $('tab-busy-said').hidden = !busy;
  line.hidden = !busy;
  line.classList.toggle('indeterminate', els.topbarLine.classList.contains('indeterminate'));
  fill.style.width = els.topbarFill.style.width;
}

// A box that takes the keyboard: on a phone the tab bar steps aside while it
// has the focus (styles/touch.css). Not a select, whose wheel is no keyboard,
// nor a checkbox's or a button's focus.
const TYPING_FIELD = 'textarea, [contenteditable="true"], input:not([type="checkbox"], [type="radio"], [type="range"], [type="color"], [type="file"], [type="button"], [type="submit"], [type="reset"])';

function syncTyping() {
  const at = document.activeElement;
  if (at?.matches?.(TYPING_FIELD) && !at.readOnly && !at.disabled) document.body.dataset.typing = '';
  else delete document.body.dataset.typing;
}

// The phone view's scroll, kept as it moves, so a tier's change has the
// place the user left. A resize's own scroll (clamped or anchored to the new
// layout) is told before the media query's change: leaving the phone,
// `isPhone()`, read live, already says no; coming back, `pageTier` does
// until syncTier, which then puts the kept place back.
function notePhoneScroll() {
  if (pageTier === 'phone' && isPhone() && !phoneScrollSettling) phoneViewScroll[phoneView] = scrollY;
}

// The kept place put back as the phone comes. At the media query's change
// the page can still be laid out at the sizes the tier had (Chrome, an
// iPad's pane dragged across), shorter than it is about to be - the More
// options sheet holds most of the panel - which clamped the scroll short of
// it: put back again as the page grows, for a second at most, unless a
// finger takes the page first. What the clamp scrolled is not noted.
let phoneScrollSettling = 0;
function restorePhoneScroll(top) {
  scrollTo({ top, behavior: 'instant' });
  if (Math.abs(scrollY - top) < 1) return;
  const settle = ++phoneScrollSettling;
  let timer = 0;
  const grown = new ResizeObserver(() => {
    if (settle !== phoneScrollSettling || !isPhone()) { done(); return; }
    scrollTo({ top, behavior: 'instant' });
    if (Math.abs(scrollY - top) < 1) done();
  });
  const done = () => {
    grown.disconnect();
    clearTimeout(timer);
    removeEventListener('touchstart', done);
    if (settle === phoneScrollSettling) phoneScrollSettling = 0;
  };
  timer = setTimeout(done, 1000);
  addEventListener('touchstart', done, { passive: true });
  grown.observe(document.body);
}

// Back closes the top overlay. On a finger - the touch tier, the tablet's
// too - each overlay open holds one history entry (`pushState`, the URL as
// it is): Android's Back and iOS's edge swipe went back a page from inside
// the lightbox, since nothing else stood behind it (the phone's selection
// bar is one too: Back leaves Select; so is the recycle bin, which Back
// leaves). Back (`popstate`)
// closes the overlays above the entry it lands on, top first, each by the
// close its own button makes; any other close - its ✕, Escape, a swipe
// down, Reuse, a light dismiss - takes its entry back with `history.go`.
// The stack is read from the page itself (each overlay's `open` or
// `hidden`, watched), so a close and an open in one turn (Refine closing
// the lightbox for its dialog) keep one entry, a close the overlay refuses
// (a dialog still sending) leaves it open and its entry is pushed again,
// and a dialog over the lightbox comes off first. A mouse's overlays hold
// none: a PC's history is untouched.
const BACK_KEY = 'auradreamLayer';
// This page's entries, told from those of an earlier load of the same tab,
// which the back never counts on.
const backPage = `${Date.now().toString(36)}.${Math.random().toString(36).slice(2)}`;
// The overlays, in the order two opened in one turn stack (the context menu
// is always on top): each with what it is shown by and the user's own close.
let backLayers = [];
// The open overlays that hold an entry, oldest first, and those opened with
// a mouse, which hold none.
let backStack = [];
const backUnheld = new Set();
// The entries of this page the current one stands on (0 at the page's own).
let backDepth = 0;
// A `history.go` of ours whose `popstate` has not come: nothing is pushed
// meanwhile, and its arrival closes nothing.
let backGoing = false;
let backGoingTimer = 0;
let backSyncQueued = false;
// The page's own scroll restoration while entries stand on it: 'manual'
// then, or a Back (or a ✕'s go) restored the scroll each entry was pushed
// at - a tab switched under the Models sheet scrolled back on its close.
let backScrollMode = null;

const backDepthOf = (state) => {
  const mark = state?.[BACK_KEY];
  return mark?.page === backPage && Number.isInteger(mark.depth) ? mark.depth : 0;
};

function backLayerList() {
  const dialogs = [...document.querySelectorAll('dialog')].filter((d) => d !== els.lightbox);
  const selectBar = $('phone-select-bar');
  return [
    // The recycle bin, shown in the gallery's place: Back leaves it, as its
    // own button does (on the phone the ‹ by its name). Its Empty button is
    // shown exactly while it is. Under Select's bar, opened inside it.
    { el: els.emptyBin, open: () => binMode, close: () => { if (binMode) toggleBin(); } },
    // The phone's selection bar (phone-select.js): Back leaves Select, the
    // picks kept, as Done does. Under every overlay opened from it.
    { el: selectBar, open: () => !selectBar.hidden, close: () => setSelectMode(false) },
    { el: els.sheet, open: () => sheetMode, close: () => { void setSheet(false); } },
    { el: els.modelPicker, open: () => !els.modelPicker.hidden, close: () => showModelPicker(false) },
    { el: els.lightbox, open: () => els.lightbox.open, close: dismissLightbox },
    // Escape's way: the dialog's `cancel` first, which a dialog busy sending
    // refuses, or which closes it by its own path (Saved searches).
    ...dialogs.map((d) => ({
      el: d,
      open: () => d.open,
      close: () => { if (d.dispatchEvent(new Event('cancel', { cancelable: true })) && d.open) d.close(); },
    })),
    { el: els.contextMenu, open: () => !els.contextMenu.hidden, close: () => closeContextMenu() },
  ];
}

function queueBackSync() {
  if (backSyncQueued) return;
  backSyncQueued = true;
  queueMicrotask(syncBackLayers);
}

// The entries made to match the overlays open now: one pushed for each
// opened, the closed ones' gone back over in one `go`. Never a step below
// this page's own entry, so the page is never left.
function syncBackLayers() {
  backSyncQueued = false;
  backStack = backStack.filter((layer) => layer.open());
  for (const layer of backUnheld) if (!layer.open()) backUnheld.delete(layer);
  for (const layer of backLayers) {
    if (!layer.open() || backStack.includes(layer) || backUnheld.has(layer)) continue;
    if (touchQuery.matches) backStack.push(layer);
    else backUnheld.add(layer);
  }
  if (backGoing) return;
  const want = backStack.length;
  if (backDepth < want) {
    if (backScrollMode === null) {
      backScrollMode = history.scrollRestoration;
      history.scrollRestoration = 'manual';
    }
    // Safari refuses a page's hundredth push in 30 seconds (a SecurityError):
    // the overlay then has no entry, and Back leaves the page as before.
    try {
      while (backDepth < want) {
        history.pushState({ [BACK_KEY]: { page: backPage, depth: backDepth + 1 } }, '');
        backDepth += 1;
      }
    } catch {
      if (backDepth === 0) {
        history.scrollRestoration = backScrollMode;
        backScrollMode = null;
      }
    }
  } else if (backDepth > want) {
    backGoing = true;
    // A go that never arrives (the page hidden mid-way) frees the next one.
    backGoingTimer = setTimeout(() => {
      backGoing = false;
      backDepth = backDepthOf(history.state);
      queueBackSync();
    }, 3000);
    history.go(want - backDepth);
  }
}

function backPopped(event) {
  const ours = backGoing;
  backGoing = false;
  clearTimeout(backGoingTimer);
  backDepth = backDepthOf(event.state);
  if (backDepth === 0 && backScrollMode !== null) {
    history.scrollRestoration = backScrollMode;
    backScrollMode = null;
  }
  // The user's Back (or a step back in the browser's list): the overlays
  // above the entry it reached close, the top one first.
  if (!ours) for (const layer of backStack.slice(backDepth).reverse()) layer.close();
  syncBackLayers();
}

function initBackLayers() {
  // A reload, or a page opened again in the tab, starts at its own entry:
  // a mark left by the last load is no layer of this one.
  if (history.state?.[BACK_KEY]) {
    history.replaceState(null, '');
    history.scrollRestoration = 'auto';
  }
  backLayers = backLayerList();
  const watch = new MutationObserver(queueBackSync);
  for (const layer of backLayers) watch.observe(layer.el, { attributes: true, attributeFilter: ['open', 'hidden'] });
  addEventListener('popstate', backPopped);
  // One already open (the Models picker offered at startup) holds its entry.
  queueBackSync();
}

function initTouchLayout() {
  let saved = null;
  try { saved = sessionStorage.getItem(PHONE_VIEW_KEY); } catch { /* storage off */ }
  setPhoneView(saved === 'gallery' ? 'gallery' : 'create', { remember: false });
  searchPlaceholder = els.search.placeholder;
  const tabs = $('phone-tabs');
  tabs.addEventListener('click', (e) => {
    const tab = e.target.closest('[data-view]');
    if (!tab) return;
    if (tab.dataset.view === phoneView && isPhone()) phoneViewToTop();
    else setPhoneView(tab.dataset.view);
  });
  // The tabs pattern's keys: the arrows move between the two and select.
  tabs.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const next = phoneView === 'create' ? 'gallery' : 'create';
    setPhoneView(next);
    tabs.querySelector(`[data-view="${next}"]`).focus();
  });
  const watch = new MutationObserver(syncPhoneTabsBusy);
  watch.observe(els.topbarLine, { attributes: true, attributeFilter: ['hidden', 'class'] });
  watch.observe(els.topbarFill, { attributes: true, attributeFilter: ['style'] });
  els.error.addEventListener('click', (e) => {
    if (!e.target.closest('button')) els.error.classList.remove('at-generate');
  });
  // iOS Safari applies :active - the press's state layer (styles/touch.css)
  // - only where a touchstart listener is; passive, it does nothing else.
  document.addEventListener('touchstart', () => {}, { passive: true });
  document.addEventListener('focusin', syncTyping);
  document.addEventListener('focusout', () => queueMicrotask(syncTyping));
  addEventListener('scroll', notePhoneScroll, { passive: true });
  // The phone's hosts' marks and watchers, before the tier first places
  // their nodes.
  initPhone();
  // The tier, the bar's job state and the keyboard's hold (a box may have
  // the focus before this runs: the prompt's, at boot).
  syncTier();
  phoneQuery.addEventListener('change', syncTier);
  touchQuery.addEventListener('change', syncTier);
  initBackLayers();
}
