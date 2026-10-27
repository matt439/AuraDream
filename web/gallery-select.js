// Auradream's page script: picking gallery cards for an action on several images.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The gallery's selection: cards picked one by one for an action that takes
// several images at once (Re-render, Swap tags, Mass refine, More seeds, and
// the Picked menu's star, unstar and bin), where the filter alone was
// the only scope - a search that matched exactly the right images was often
// not to be had. A card's image opens on a click; Ctrl/Cmd toggles a pick
// and Ctrl/Cmd+Shift adds a range (cardPress). The keyboard keeps Enter
// for opening and Space for picking. The head's Select provides modifier-free
// touch picking: every press toggles. F stars a card and its tools
// are its own either way; a selection stays marked and counted until Clear,
// Escape or a click between the cards.
//
// By id, in the order picked, with the record each id was picked from: the
// selection outlives the cards. A new search, a filter, a sheet or a page
// paged in keeps it, and the count says how many of it this view does not
// show; only forgetImages, the one place the page drops deleted ids, takes
// ids out of it unasked (forgetSelected). The recycle bin sets it aside while
// it is shown and puts it back after (stashSelection, unstashSelection).
//
// What other features read:
//   selectedIds()          the ids in the order they were picked (a copy)
//   selectedRecords()      their records, in the same order: the loaded
//                          page's copy where there is one, else the record
//                          as picked (a snapshot; the ids are what to send)
//   isSelected(id)
//   onSelectionChange(fn)  fn(ids) after every change of the selection or of
//                          Select, and whenever a page lands (syncSelection)
//   clearSelection()
let selectMode = false;
const selection = new Map();
// The card last toggled, by id: a Shift pick runs from it. Opening leaves it.
let selectAnchor = null;
const selectionListeners = [];

function selectedIds() { return [...selection.keys()]; }
// Each id's record as the gallery holds it now where a loaded page has it
// (a star or a pass since is on that copy), else as it was when picked. The
// ids are what an action sends; the server reads the rows.
function selectedRecords() {
  const now = new Map(galleryRecords.map((r) => [r.id, r]));
  return [...selection].map(([id, rec]) => now.get(id) ?? rec);
}
function isSelected(id) { return selection.has(id); }
function onSelectionChange(fn) { selectionListeners.push(fn); }

function clearSelection() {
  selectAnchor = null;
  if (!selection.size) return;
  selection.clear();
  syncSelection();
}

// The gallery's picks while the recycle bin is shown (bin.js), which picks
// its own for Restore picked: set aside on the way in and back on the way
// out - a look into the bin for one keeper is no request to throw away
// forty picks. forgetSelected keeps it as it keeps the selection, so an id
// binned or deleted meanwhile does not come back.
let stashedSelection = null;

function stashSelection() {
  stashedSelection = { picks: new Map(selection), anchor: selectAnchor };
  clearSelection();
}

// `binned`: ids the bin's view showed, binned elsewhere meanwhile, which
// forgetImages never heard of; they stay out.
function unstashSelection(binned = new Set()) {
  const stash = stashedSelection;
  stashedSelection = null;
  selection.clear();
  selectAnchor = null;
  if (stash) {
    for (const [id, rec] of stash.picks) if (!binned.has(id)) selection.set(id, rec);
    selectAnchor = binned.has(stash.anchor) ? null : stash.anchor;
  }
  syncSelection();
}

// The most one action takes: the server's grid ceiling (max_grid_images, from
// /api/status), which the actions' `ids` are held to. Read as the sheet's
// page is, since that is the same number.
function selectionCap() { return sheetPage; }

// On a sheet the cards are hidden and its cells are not picked: Select is
// held there with the reason, a selection made before kept and counted.
const SELECT_ON_SHEET = 'The sheet lays its cells out its own way; close it to pick cards. A selection made before is kept.';

function setSelectMode(on) {
  if (on && sheetMode) {
    showError(SELECT_ON_SHEET, true, { slot: els.galleryError });
    return;
  }
  selectMode = on;
  if (!on) selectAnchor = null;
  syncSelection();
}

// Everything the selection shows, from the state: each card's mark, Select
// and the count, Clear - and then whoever listens. Called after every change,
// from every page that lands (loadGallery) and from setSheet.
function syncSelection() {
  els.gallery.classList.toggle('selecting', selectMode);
  const shown = new Set();
  els.gallery.querySelectorAll('.card[data-id]').forEach((div) => {
    syncCardSelected(div);
    shown.add(+div.dataset.id);
  });
  // Held on a sheet, Select does not read as pressed there, nor does the
  // count invite picks the sheet's cells cannot take.
  const picking = selectMode && !sheetMode;
  els.select.setAttribute('aria-pressed', String(picking));
  els.select.classList.toggle('active', picking);
  els.select.setAttribute('aria-disabled', String(sheetMode));
  els.select.dataset.title ??= els.select.title;
  els.select.title = sheetMode ? SELECT_ON_SHEET : els.select.dataset.title;
  els.select.setAttribute('aria-keyshortcuts', sheetMode ? 'Escape' : 'Escape Control+A Meta+A');
  const n = selection.size;
  const away = n - selectedIds().filter((id) => shown.has(id)).length;
  const text = n ? `${n} selected${away ? ` · ${away} not shown` : ''}`
    : picking ? 'Click cards to pick them' : '';
  // Written only when it changes: a page landing re-syncs, and a status
  // repeated is read out again.
  if (els.selectCount.textContent !== text) els.selectCount.replaceChildren(...countParts(n, away, text));
  els.selectCount.title = away ? `${away} of the selected images are not among the cards this search and its pages show; they stay selected` : text;
  // Clear goes with an empty selection. Focus on it moves to Select first,
  // so it does not fall to <body>.
  if (!n && document.activeElement === els.selectClear) els.select.focus({ preventScroll: true });
  els.selectClear.hidden = !n;
  syncPickedButton();
  const ids = selectedIds();
  for (const fn of selectionListeners) {
    try { fn(ids); } catch (err) { console.error(err); }
  }
}

// The count's text in parts the footer can narrow (gallery.css): the number
// is never cut, " selected" gives way to a ✓ before it where the row is
// short (Gallery beside, a phone), and only then does the "N not shown"
// tail - the warning - end in an ellipsis. Its words are the text as ever.
function countParts(n, away, text) {
  const part = (cls, words) => Object.assign(document.createElement('span'), { className: cls, textContent: words });
  if (!n) return text ? [part('tail', text)] : [];
  const num = part('num', String(n));
  num.append(part('word', ' selected'));
  return away ? [num, part('tail', ` · ${away} not shown`)] : [num];
}

// One card's mark. Outside Select the image opens; in Select it is a toggle.
// The group's name still says whether this card belongs to the picks.
function syncCardSelected(div) {
  const id = +div.dataset.id;
  const on = selection.has(id);
  div.classList.toggle('selected', on);
  div.setAttribute('aria-label', `#${id}${on ? ' (selected)' : ''}`);
  const img = div.querySelector(':scope > img');
  if (!img) return;
  if (selectMode) img.setAttribute('aria-pressed', String(on));
  else img.removeAttribute('aria-pressed');
  const label = img.getAttribute('aria-label') ?? '';
  img.setAttribute('aria-label', label.replace(/^(Open|Select) /, selectMode ? 'Select ' : 'Open '));
  // A recycle bin card has no star, so no F (binCard).
  const star = div.classList.contains('binned') ? '' : 'F ';
  img.setAttribute('aria-keyshortcuts', `Enter ${star}M Space Shift+Space Escape Control+A Meta+A`);
}

// A card image's press (thumb's `open`, from card()): a plain click opens
// without changing the picks or their anchor; Ctrl/Cmd toggles one pick.
// Shift picks the range alone and Ctrl/Cmd+Shift adds it. Enter opens,
// Space toggles, Shift+Space adds the range. Select mode is the
// path with no modifier keys - a touch screen's - where every click toggles
// as a Ctrl click does and Shift adds.
function cardPress(div, rec, e) {
  if (e?.type === 'keydown') {
    if (e.key === 'Enter') { openLightbox(rec, galleryRecords); return; }
    if (e.shiftKey && selectRange(div)) return;
    toggleSelected(rec);
    return;
  }
  if (e?.button != null && e.button !== 0) return;
  // Opening needs only the first click. In Select every press toggles,
  // quick ones too, and nothing opens.
  if (e?.detail > 1 && !selectMode) return;
  const add = selectMode || e?.ctrlKey || e?.metaKey;
  if (e?.shiftKey && selectRange(div, { add })) return;
  if (add) toggleSelected(rec);
  else if (e?.shiftKey) selectOnly(rec);
  else if (!e?.altKey) openLightbox(rec, galleryRecords);
}

// A Shift range with no loaded anchor starts by picking just this card.
function selectOnly(rec) {
  selection.clear();
  selection.set(rec.id, rec);
  selectAnchor = rec.id;
  syncSelection();
}

// A click on the grid between cards (on a day's line, past the last card)
// empties the selection, as a file manager's does, so the actions go back to
// the filter. Not in select mode, the path for touch, where a tap between
// cards is too easily made to cost the picks; not with a modifier held, nor
// at the end of a drag selecting a day line's text.
function galleryBlankClick(e) {
  if (selectMode || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
  if (!(getSelection()?.isCollapsed ?? true)) return;
  if (e.target.closest?.('.card, button, a, input, select, textarea, summary, [role="button"], [tabindex]')) return;
  clearSelection();
}

// The anchor moves only with a toggle that happened: a pick refused at the
// cap leaves it where the last real one was, so the next Shift range starts
// from a card that is picked.
function toggleSelected(rec) {
  if (selection.has(rec.id)) selection.delete(rec.id);
  else if (refuseOverCap(1)) return;
  else selection.set(rec.id, rec);
  selectAnchor = rec.id;
  syncSelection();
}

// The loaded cards from the anchor to this one, in the gallery's order, all
// picked - beside the rest with `add`, else in their place (a plain Shift
// click), the anchor kept. False where there is no anchor among them - none
// yet, or one this view does not show - and the press picks the one card
// instead.
function selectRange(div, { add = true } = {}) {
  const cards = [...els.gallery.querySelectorAll('.card[data-id]')];
  const from = cards.findIndex((c) => +c.dataset.id === selectAnchor);
  const to = cards.indexOf(div);
  if (from < 0 || to < 0) return false;
  if (!add) selection.clear();
  const span = cards.slice(Math.min(from, to), Math.max(from, to) + 1);
  // In the order the range runs, from the anchor out.
  if (to < from) span.reverse();
  const fresh = span.map((c) => +c.dataset.id).filter((id) => !selection.has(id));
  const room = Math.max(0, selectionCap() - selection.size);
  refuseOverCap(fresh.length);
  for (const id of fresh.slice(0, room)) {
    const rec = galleryRecords.find((r) => r.id === id);
    if (rec) selection.set(id, rec);
  }
  syncSelection();
  return true;
}

// Said where the gallery says things, before the pick: true where `more`
// would take the selection past the cap (and only what fits is picked).
function refuseOverCap(more) {
  const cap = selectionCap();
  if (selection.size + more <= cap) return false;
  const left = Math.max(0, cap - selection.size);
  showError(`${cap} images is the most one action takes; ${left ? `only ${left} more were picked` : 'nothing more was picked'}.`,
    true, { slot: els.galleryError });
  return true;
}

// Escape in the gallery - on a card, on Select or Clear: a selection is
// emptied first, then select mode left. False where it had nothing to do,
// so the key goes on to the page's own Escapes (the prompt check's).
function selectEscape() {
  if (selection.size) { clearSelection(); return true; }
  if (selectMode) { setSelectMode(false); return true; }
  return false;
}

function selectKey(e) {
  if (keyboardEventHeld(e)) return;
  if (!sheetMode && (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey
      && e.key.toLowerCase() === 'a'
      && e.target.closest?.('#gallery .card, #btn-select')
      && !keyboardEditingTarget(e.target)) {
    e.preventDefault();
    if (!e.repeat) pickLoadedCards();
    return;
  }
  if (e.key !== 'Escape' || keyboardCommandModifier(e) || e.shiftKey || keyboardEditingTarget(e.target)) return;
  // On a card or on the selection's own controls only: the search box's
  // Escape empties it, the Runs picker's is its own, and the sheet's cells
  // are not picked. The Picked menu's, open, closes it first (pickedKey).
  if (!e.target.closest?.('#gallery .card, #btn-select, #btn-select-clear, #btn-picked')) return;
  if (e.repeat) { e.preventDefault(); return; }
  if (selectEscape()) e.preventDefault();
}

// Add the cards this view has loaded, retaining picks from other filters.
// No fetch: Ctrl+A never claims that unloaded search results were picked.
function pickLoadedCards() {
  const records = new Map(galleryRecords.map((rec) => [rec.id, rec]));
  const fresh = [...els.gallery.querySelectorAll('.card[data-id]')]
    .map((card) => records.get(+card.dataset.id))
    .filter((rec) => rec && !selection.has(rec.id));
  const room = Math.max(0, selectionCap() - selection.size);
  const capped = refuseOverCap(fresh.length);
  const added = fresh.slice(0, room);
  for (const rec of added) selection.set(rec.id, rec);
  syncSelection();
  if (!capped) {
    showError(`Added ${plural(added.length, 'loaded image')}; ${plural(selection.size, 'image')} selected.${galleryMore ? ' Unloaded images were not picked.' : ''}`,
      true, { slot: els.galleryError });
  }
}

// From forgetImages, the one place the page drops deleted ids: out of the
// selection and out of the gallery's picks set aside while the bin is shown.
// `stashed: false` is the bin's own cards leaving its view (a Restore),
// which were never among the gallery's picks.
function forgetSelected(gone, { stashed = true } = {}) {
  let changed = false;
  for (const id of gone) changed = selection.delete(id) || changed;
  if (gone.has(selectAnchor)) selectAnchor = null;
  if (stashed && stashedSelection) {
    for (const id of gone) stashedSelection.picks.delete(id);
    if (gone.has(stashedSelection.anchor)) stashedSelection.anchor = null;
  }
  if (changed) syncSelection();
}

// The Picked menu (#btn-picked, between Clear and Select): what the picked
// cards take together that is no job - Star picked, Unstar picked, Bin picked
// (starPicked, binPicked: gallery-actions.js). Picks only, never the filter:
// a filter's stars are a card's F each, and its bin is Delete unstarred. A
// menu, not three buttons: those were some 320 px, and with a card picked the
// row had 22 px to spare in a 600 px window and 95 beside the viewer at QHD
// (where it is short, gallery.css leaves the button its arrow and Clear a
// ✕, so the count keeps its number and its "not shown"). Always shown,
// held with its reason where nothing is picked and in the recycle bin (whose
// picks are Restore picked's), so the row never changes under the pointer.
// It is a role=menu and keeps that promise as Examples does: ArrowDown on the
// button opens it on its first item, the arrows walk it, Escape or a choice
// closes it with focus back on the button, and leaving it closes it.
const PICKED_NONE = 'Pick cards (Ctrl/Cmd+click or Select) to star, unstar, bin or use their seeds in a grid';

// Seeds only: the picked order is the investigation's order, while another
// image of the same seed contributes no second entry. Leave grid limits to
// its existing server preview; never silently trim the selected seeds.
function pickedGridSeeds() {
  if (binMode) return { why: BIN_HELD };
  const records = selectedRecords();
  if (!records.length) return { why: PICKED_NONE };
  const unsafe = records.find((rec) => !Number.isSafeInteger(rec.seed) || rec.seed < 0);
  if (unsafe) return { why: `#${unsafe.id} has a seed the browser cannot copy exactly; use whole seeds from 0 to ${Number.MAX_SAFE_INTEGER}` };
  return { seeds: [...new Set(records.map((rec) => rec.seed))] };
}

function seedsToGrid() {
  const target = pickedGridSeeds();
  if (target.why) {
    showError(target.why + '.', true, { slot: els.galleryError });
    return;
  }
  els.gridSeedList.value = target.seeds.join(', ');
  els.gridSection.open = true;
  // The normal input path disables the count, persists the field and asks
  // the server to preview the complete grid, with every other value kept.
  els.gridSeedList.dispatchEvent(new Event('input', { bubbles: true }));
  els.gridSeedList.focus();
  els.gridSeedList.select();
  showError(`Seed list now has ${plural(target.seeds.length, 'distinct picked seed')} in picked order.`, true, { slot: els.galleryError });
}

function pickedComparison() {
  if (binMode) return { why: BIN_HELD };
  const records = selectedRecords();
  if (records.length !== 2) return { why: `Pick exactly 2 images to compare; ${records.length} selected` };
  const [a, b] = records;
  if (a.width * b.height !== b.width * a.height) {
    return { why: `Compare picked needs the same aspect ratio; #${a.id} is ${a.width}x${a.height} and #${b.id} is ${b.width}x${b.height}` };
  }
  return { records };
}

function comparePicked() {
  const target = pickedComparison();
  if (target.why) { showError(`${target.why}.`, true, { slot: els.galleryError }); return; }
  const pair = target.records;
  openLightbox(pair[0], pair);
  lbPickedPair = pair;
  comparePickedPair(pair[0]);
}

function syncPickedButton() {
  const n = selection.size;
  const why = binMode ? BIN_HELD : n ? '' : PICKED_NONE;
  els.pickedBtn.setAttribute('aria-disabled', String(!!why));
  const them = n === 1 ? 'the picked image' : `the ${n} picked images`;
  els.pickedBtn.title = why || `Actions on ${them}: star, unstar, compare, bin or copy seeds to the grid`;
  const grid = pickedGridSeeds();
  const compare = pickedComparison();
  const titles = {
    star: `Star ${them}`,
    unstar: `Unstar ${them}`,
    bin: `Move ${them} to the recycle bin; the note it leaves has Restore`,
    seeds: grid.why || `Fill Seed list with ${plural(grid.seeds.length, 'distinct picked seed')} in picked order; other grid settings are kept`,
    compare: compare.why || `Compare #${compare.records[0].id} with #${compare.records[1].id}; the first picked image stays on the left`,
  };
  for (const item of els.pickedMenu.querySelectorAll('[data-picked]')) {
    const held = why || (item.dataset.picked === 'seeds' ? grid.why : item.dataset.picked === 'compare' ? compare.why : '');
    item.setAttribute('aria-disabled', String(!!held));
    item.title = held || titles[item.dataset.picked];
  }
  if (why && !els.pickedMenu.hidden) showPickedMenu(false);
}

// Over the button where the window has no room for it under the footer -
// the gallery's head low under a tall viewer - rather than out of sight.
function showPickedMenu(open) {
  const inside = els.pickedMenu.contains(document.activeElement);
  els.pickedBtn.setAttribute('aria-expanded', String(open));
  els.pickedMenu.hidden = !open;
  if (!open) {
    if (inside) els.pickedBtn.focus({ preventScroll: true });
    return;
  }
  els.pickedMenu.classList.remove('up');
  const box = els.pickedMenu.getBoundingClientRect();
  const top = document.querySelector('.topbar').getBoundingClientRect().bottom;
  if (box.bottom > innerHeight && els.pickedBtn.getBoundingClientRect().top - box.height - 6 >= top) {
    els.pickedMenu.classList.add('up');
  }
}

// A press: the menu, or the reason it is held. From the keyboard (a click of
// detail 0, Enter or Space) focus goes in on the first item, as ArrowDown's
// does: the items are no Tab stops.
function pickedPress(e) {
  if (els.pickedBtn.getAttribute('aria-disabled') === 'true') {
    showError(`${els.pickedBtn.title}.`, true, { slot: els.galleryError });
    return;
  }
  const open = els.pickedMenu.hidden;
  showPickedMenu(open);
  if (open && e.detail === 0) els.pickedMenu.querySelector('[role="menuitem"]')?.focus();
}

function pickedKey(e) {
  if (keyboardEventHeld(e) || keyboardCommandModifier(e) || keyboardEditingTarget(e.target) || e.shiftKey) return;
  if (e.repeat && ['Enter', ' ', 'Escape'].includes(e.key)) { e.preventDefault(); return; }
  if (e.key === 'Escape' && !els.pickedMenu.hidden) { e.preventDefault(); showPickedMenu(false); return; }
  if (e.target === els.pickedBtn) {
    if (e.key !== 'ArrowDown' || els.pickedBtn.getAttribute('aria-disabled') === 'true') return;
    e.preventDefault();
    showPickedMenu(true);
    els.pickedMenu.querySelector('[role="menuitem"]')?.focus();
    return;
  }
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); pickedChoose(e.target); return; }
  const items = [...els.pickedMenu.querySelectorAll('[role="menuitem"]')];
  const at = items.indexOf(document.activeElement);
  const next = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: items.length - 1 }[e.key];
  if (next === undefined) return;
  e.preventDefault();
  items[(next + items.length) % items.length].focus();
}

function pickedChoose(item) {
  const target = item?.closest?.('[data-picked]');
  const act = target?.dataset.picked;
  if (!act) return;
  syncPickedButton();
  showPickedMenu(false);
  if (target.getAttribute('aria-disabled') === 'true') {
    showError(target.title + '.', true, { slot: els.galleryError });
    return;
  }
  if (act === 'bin') void binPicked();
  else if (act === 'seeds') seedsToGrid();
  else if (act === 'compare') comparePicked();
  else void starPicked(act === 'star');
}

// What an action on several images (Swap tags, Mass refine) would act on
// now: the picked cards, else the filter where it narrows the gallery to at
// most a job's worth, else nothing - with why. `purpose` ends the sentence
// that says how to choose ("to choose the images to swap tags in"), `one`
// names a single run of the action ("one swap"). `scope` is what the
// request sends: `{ids}` or the filter.
function galleryActionTarget(purpose, one) {
  // The recycle bin's images are no scope for an action (bin.js).
  if (binMode) return { why: BIN_HELD };
  const ids = selectedIds();
  if (ids.length) {
    return { scope: { ids }, n: ids.length, words: `the ${ids.length === 1 ? 'selected image' : `${ids.length} selected images`}` };
  }
  const search = els.search.value.trim();
  const filtered = els.favOnly.checked || search !== '';
  if (!filtered) {
    return { why: `Pick cards (Ctrl/Cmd+click or Select), or narrow the gallery with a search or Starred, ${purpose}` };
  }
  if (total === 0) return { why: 'Nothing in the gallery matches the filter' };
  if (total > sheetPage) {
    return { why: `${total} images match; ${one} takes at most ${sheetPage}. Narrow the search, or pick cards` };
  }
  return {
    scope: { search: search || null, favorites: els.favOnly.checked, all_models: els.allModels.checked },
    n: total,
    words: `the ${total === 1 ? 'image' : `${total} images`} the filter matches`,
  };
}

// The dialog's line naming what it acts on, fixed when it opens: the picked
// cards (and how many of them the gallery's view does not show), or the
// filter's terms and scope. `lead` starts it ("In", "Refining").
function galleryActionScopeLine(target, lead) {
  if (target.scope.ids) {
    const shown = new Set(galleryRecords.map((r) => r.id));
    const away = target.scope.ids.filter((id) => !shown.has(id)).length;
    return `${lead} ${target.words}${away ? ` (${away} not in the gallery's view)` : ''}.`;
  }
  const { search, favorites, all_models: everywhere } = target.scope;
  const parts = [search ? `matching "${search}"` : '', favorites ? 'starred' : ''].filter(Boolean);
  return `${lead} the ${plural(target.n, 'image')} ${parts.join(', ')}${everywhere ? ', across checkpoints' : ', this checkpoint only'}.`;
}
