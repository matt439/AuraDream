// Local Diffusion's page script: picking gallery cards for an action on several images.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The gallery's selection: cards picked one by one for an action that takes
// several images at once (Swap tags, Mass refine), where the filter alone was
// the only scope - a search that matched exactly the right images was often
// not to be had. The head's Select (in its footer row, beside Bundles…) turns
// a card's press into picking it; out of that mode a card does what it always
// did - its image opens the lightbox, F stars it, its tools are its own - and
// a selection already made stays marked and counted until Clear or Escape.
//
// By id, in the order picked, with the record each id was picked from: the
// selection outlives the cards. A new search, a filter, a sheet or a page
// paged in keeps it, and the count says how many of it this view does not
// show; only forgetImages, the one place the page drops deleted ids, takes
// ids out of it unasked (forgetSelected).
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
// The card a plain pick was last made on, by id: a Shift pick runs from it.
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

// One card's mark. Picking is the image's press in select mode, so the image
// is the toggle there (aria-pressed, named "Select #N"); out of it, the press
// opens the lightbox as ever and the group's name says the card is selected.
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
  img.setAttribute('aria-keyshortcuts', selectMode ? 'F Space Shift+Space Escape' : 'F');
}

// A card image's press (thumb's `open`, from card()): the lightbox, as
// always, out of select mode - and in it on Enter, so the keyboard can still
// look before picking. In select mode a click or Space picks or unpicks it,
// and with Shift picks every loaded card from the last one picked to it.
function cardPress(div, rec, e) {
  if (!selectMode || e?.key === 'Enter') { openLightbox(rec, galleryRecords); return; }
  if (e?.shiftKey && selectRange(div)) return;
  toggleSelected(rec);
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
// picked. False where there is no anchor among them - none yet, or one this
// view does not show - and the press picks the one card instead.
function selectRange(div) {
  const cards = [...els.gallery.querySelectorAll('.card[data-id]')];
  const from = cards.findIndex((c) => +c.dataset.id === selectAnchor);
  const to = cards.indexOf(div);
  if (from < 0 || to < 0) return false;
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
  if (e.key !== 'Escape' || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
  // On a card or on the selection's own controls only: the search box's
  // Escape empties it, the Runs picker's is its own, and the sheet's cells
  // are not picked.
  if (!e.target.closest?.('#gallery .card, #btn-select, #btn-select-clear')) return;
  if (selectEscape()) e.preventDefault();
}

// From forgetImages, the one place the page drops deleted ids.
function forgetSelected(gone) {
  let changed = false;
  for (const id of gone) changed = selection.delete(id) || changed;
  if (gone.has(selectAnchor)) selectAnchor = null;
  if (changed) syncSelection();
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
    return { why: `Pick cards with Select, or narrow the gallery with a search or Starred, ${purpose}` };
  }
  if (total === 0) return { why: 'Nothing in the gallery matches the filter' };
  if (total > sheetPage) {
    return { why: `${total} images match; ${one} takes at most ${sheetPage}. Narrow the search, or pick cards with Select` };
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
