// AuraDream's page script: the phone's two views (styles/touch.css has the
// layouts; state.js the media queries they share). Declarations only;
// app.js calls initTouchLayout() once at boot.
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
  // "/ to focus" is a keyboard's.
  els.search.placeholder = touchQuery.matches ? 'Search prompts' : searchPlaceholder;
  if (was === pageTier) return;
  // Back on the phone, the view shows where it was left, as a tab's press
  // does: the page between was the whole of both views, scrolled elsewhere.
  // (Phone-only layers - sheets, hosts - close or move back here as they
  // come; today the phone has none.)
  if (was !== undefined && pageTier === 'phone') scrollTo({ top: phoneViewScroll[phoneView], behavior: 'instant' });
  // The bar's state and the keyboard's hold read the page as it is now.
  syncPhoneTabsBusy();
  syncTyping();
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
}

// Reuse, Load settings and the like write the panel: on a phone looking at
// the gallery, the panel is the other view, so they bring it.
function showCreateView() {
  if (isPhone() && phoneView !== 'create') setPhoneView('create');
}

// The running job, on the tab bar: Create's dot, and the topbar's line
// copied along the bar's top edge (job-progress.js writes the topbar's).
function syncPhoneTabsBusy() {
  const line = $('phone-tabs-line');
  const fill = $('phone-tabs-fill');
  const busy = !els.topbarLine.hidden;
  $('tab-busy').hidden = !busy;
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
  if (pageTier === 'phone' && isPhone()) phoneViewScroll[phoneView] = scrollY;
}

function initTouchLayout() {
  let saved = null;
  try { saved = sessionStorage.getItem(PHONE_VIEW_KEY); } catch { /* storage off */ }
  setPhoneView(saved === 'gallery' ? 'gallery' : 'create', { remember: false });
  searchPlaceholder = els.search.placeholder;
  const tabs = $('phone-tabs');
  tabs.addEventListener('click', (e) => {
    const tab = e.target.closest('[data-view]');
    if (tab) setPhoneView(tab.dataset.view);
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
  document.addEventListener('focusin', syncTyping);
  document.addEventListener('focusout', () => queueMicrotask(syncTyping));
  addEventListener('scroll', notePhoneScroll, { passive: true });
  // The tier, the bar's job state and the keyboard's hold (a box may have
  // the focus before this runs: the prompt's, at boot).
  syncTier();
  phoneQuery.addEventListener('change', syncTier);
  touchQuery.addEventListener('change', syncTier);
}
