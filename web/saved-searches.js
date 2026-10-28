// Named gallery filters. Declarations only; app.js owns initialization.
'use strict';

let savedSearchItems = [];
let savedSearchSelected = null;
let savedSearchReady = false;
let savedSearchPending = null;
let savedSearchReadTicket = 0;
let savedSearchEditEpoch = 0;
let savedSearchDirty = false;
let savedSearchOrigin = null;
let savedSearchConflict = null;

function savedSearchSay(text, error = false) {
  els.savedSearchStatus.textContent = text;
  els.savedSearchStatus.classList.toggle('error', error);
}

function savedSearchFields() {
  return {
    name: els.savedSearchName.value.trim(), query: els.savedSearchQuery.value,
    favorites: els.savedSearchFavorites.checked, all_models: els.savedSearchAllModels.checked,
  };
}

function savedSearchQueryReason() {
  const query = els.savedSearchQuery.value;
  if ([...query].length > 4000) return 'Keep the query within 4,000 characters.';
  if (!query.isWellFormed()) return 'The query must be valid Unicode.';
  return '';
}

function savedSearchSaveReason() {
  if (savedSearchPending) return 'Wait for the saved searches request to finish.';
  if (!savedSearchReady) return 'Reload the saved list before saving.';
  const name = els.savedSearchName.value.trim();
  if (!name || /\p{Cc}/u.test(name)) return 'Give the search a name without control characters.';
  if ([...name].length > 80) return 'Keep the name within 80 characters.';
  if (!name.isWellFormed()) return 'The name must be valid Unicode.';
  return savedSearchQueryReason();
}

function syncSavedSearches() {
  const held = (button, reason, title) => {
    button.setAttribute('aria-disabled', String(!!reason));
    button.title = reason || title;
  };
  const wait = savedSearchPending ? 'Wait for the saved searches request to finish.' : '';
  const write = savedSearchPending === 'write' ? wait : '';
  held(els.savedSearchSave, savedSearchSaveReason(), savedSearchSelected ? 'Save changes to this search' : 'Save a new named search');
  held(els.savedSearchDelete, wait || (!savedSearchReady ? 'Reload the saved list before deleting.'
    : !savedSearchSelected ? 'Select a saved search to delete.' : ''), 'Delete only this saved definition; keep the gallery and its images');
  held(els.savedSearchApply, wait || savedSearchQueryReason(), 'Apply the editor’s query, Starred and All models to the gallery; keep existing picks');
  held(els.savedSearchCurrent, write || (binMode ? 'Leave the recycle bin to copy a live gallery filter.' : ''), 'Copy the current gallery filter into this editor without saving');
  held(els.savedSearchReload, wait, 'Read the latest saved list; keep unsaved edits until you explicitly select an entry');
  held(els.savedSearchNew, write, 'Start a new saved search');
  held(els.savedSearchClose, write, 'Close Saved searches');
  els.savedSearchReplace.hidden = !savedSearchConflict;
  held(els.savedSearchReplace, savedSearchSaveReason(), 'Replace the named search with these fields using its latest read revision');
  for (const field of [els.savedSearchList, els.savedSearchName, els.savedSearchQuery, els.savedSearchFavorites, els.savedSearchAllModels]) {
    field.disabled = savedSearchPending === 'write';
  }
}

function renderSavedSearches() {
  els.savedSearchList.replaceChildren();
  for (const item of savedSearchItems) {
    const option = document.createElement('option');
    option.value = item.id;
    const scope = `${item.favorites ? 'Starred · ' : ''}${item.all_models ? 'All models' : 'Current checkpoint'}`;
    option.textContent = `${item.name} — ${scope}${item.query ? ` · ${item.query.replace(/\s+/gu, ' ').slice(0, 100)}` : ''}`;
    option.title = `${item.name}\n${scope}\n${item.query}`;
    els.savedSearchList.append(option);
  }
  if (!savedSearchItems.length) {
    const empty = new Option('No saved searches yet', '');
    els.savedSearchList.append(empty);
  }
  els.savedSearchList.value = savedSearchSelected?.id || '';
  syncSavedSearches();
}

function selectSavedSearch(id) {
  if (savedSearchPending === 'write') return;
  const item = savedSearchItems.find((entry) => entry.id === id);
  if (!item) return;
  ++savedSearchEditEpoch;
  savedSearchSelected = { ...item };
  savedSearchDirty = false;
  savedSearchConflict = null;
  els.savedSearchList.value = id;
  els.savedSearchName.value = item.name;
  els.savedSearchQuery.value = item.query;
  els.savedSearchFavorites.checked = item.favorites;
  els.savedSearchAllModels.checked = item.all_models;
  savedSearchSay('Selected for editing. Apply to gallery opens this filter.');
  syncSavedSearches();
}

function newSavedSearch({ focus = true } = {}) {
  if (savedSearchPending === 'write') { savedSearchSay('Wait for the saved search to finish saving.', true); return; }
  ++savedSearchEditEpoch;
  savedSearchSelected = null;
  savedSearchDirty = true;
  savedSearchConflict = null;
  els.savedSearchList.value = '';
  els.savedSearchName.value = '';
  els.savedSearchQuery.value = '';
  els.savedSearchFavorites.checked = false;
  els.savedSearchAllModels.checked = false;
  savedSearchSay('New search. Use current filter to copy the gallery’s query and ticks.');
  syncSavedSearches();
  if (focus) els.savedSearchName.focus({ preventScroll: true });
}

function useCurrentSavedSearch() {
  if (savedSearchPending === 'write' || binMode) { savedSearchSay(els.savedSearchCurrent.title, true); return; }
  els.savedSearchQuery.value = els.search.value;
  els.savedSearchFavorites.checked = els.favOnly.checked;
  els.savedSearchAllModels.checked = els.allModels.checked;
  savedSearchEdited();
  savedSearchSay('Current gallery filter copied into the editor. Save keeps it under a name.');
}

function savedSearchEdited() {
  ++savedSearchEditEpoch;
  savedSearchDirty = true;
  savedSearchConflict = null;
  syncSavedSearches();
}

async function reloadSavedSearches({ opening = false } = {}) {
  if (savedSearchPending === 'write') { savedSearchSay('Wait for the saved search to finish saving.', true); return; }
  const ticket = ++savedSearchReadTicket;
  const edit = savedSearchEditEpoch;
  savedSearchPending = 'read';
  savedSearchReady = false;
  savedSearchConflict = null;
  savedSearchSay('Reading saved searches.');
  syncSavedSearches();
  try {
    const result = await api('/api/gallery/saved-searches');
    if (ticket !== savedSearchReadTicket || !els.savedSearchDialog.open) return;
    savedSearchItems = result.items;
    savedSearchReady = true;
    savedSearchPending = null;
    renderSavedSearches();
    // Reload never adopts a new revision for a draft. An opening may fill a
    // clean editor only while no selection or typing happened during its read.
    if (opening && !savedSearchDirty && edit === savedSearchEditEpoch) {
      const selected = savedSearchItems.find((item) => item.id === savedSearchSelected?.id) || savedSearchItems[0];
      if (selected) selectSavedSearch(selected.id);
      else newSavedSearch({ focus: false });
    } else els.savedSearchList.selectedIndex = -1;
    savedSearchSay(!opening || savedSearchDirty ? 'List reloaded. Your editor is unchanged; select an entry to read its latest revision.'
      : savedSearchItems.length ? 'Select a search, then Apply to gallery.' : 'Save a name for a useful gallery filter.');
  } catch (err) {
    if (ticket !== savedSearchReadTicket || !els.savedSearchDialog.open) return;
    savedSearchPending = null;
    savedSearchSay(`Saved searches could not be read: ${err.message}`, true);
    syncSavedSearches();
  }
}

async function openSavedSearches() {
  if (els.savedSearchDialog.open) return;
  savedSearchOrigin = document.activeElement;
  els.savedSearchDialog.showModal();
  await reloadSavedSearches({ opening: true });
}

async function saveSavedSearch({ replace = false } = {}) {
  const reason = savedSearchSaveReason();
  if (reason) { savedSearchSay(reason, true); return; }
  const selected = replace ? savedSearchConflict : savedSearchSelected;
  if (replace && !selected) { savedSearchSay('Save first to check whether this name is already in use.', true); return; }
  const fields = savedSearchFields();
  const edit = savedSearchEditEpoch;
  savedSearchPending = 'write';
  savedSearchConflict = null;
  savedSearchSay('Saving search.');
  syncSavedSearches();
  try {
    const result = await api(selected ? `/api/gallery/saved-searches/${selected.id}` : '/api/gallery/saved-searches', {
      method: selected ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(selected ? { ...fields, revision: selected.revision } : fields),
    });
    savedSearchItems = result.items;
    savedSearchPending = null;
    renderSavedSearches();
    const saved = selected ? savedSearchItems.find((item) => item.id === selected.id) : savedSearchItems.at(-1);
    if (saved && edit === savedSearchEditEpoch) selectSavedSearch(saved.id);
    savedSearchSay('Search saved. The gallery filter is unchanged.');
  } catch (err) {
    // A duplicate is identified by the server's NFC/casefold comparison,
    // never by a second, approximate browser spelling rule.
    const conflictId = err.body?.conflict_id;
    if (err.status === 409 || err.status === 404) savedSearchReady = false;
    if (conflictId) {
      try {
        const fresh = await api('/api/gallery/saved-searches');
        savedSearchItems = fresh.items;
        savedSearchReady = true;
        savedSearchConflict = savedSearchItems.find((item) => item.id === conflictId) || null;
      } catch { /* The editor and prior revision stay available for Reload. */ }
    }
    savedSearchPending = null;
    renderSavedSearches();
    savedSearchSay(savedSearchConflict
      ? 'That name is already saved. Update named search replaces it with this editor’s fields, or choose another name.'
      : `Search was not saved: ${err.message}`, true);
    syncSavedSearches();
  }
}

async function deleteSavedSearch() {
  if (els.savedSearchDelete.getAttribute('aria-disabled') === 'true') { savedSearchSay(els.savedSearchDelete.title, true); return; }
  const selected = savedSearchSelected;
  savedSearchPending = 'write';
  savedSearchSay('Deleting saved search.');
  syncSavedSearches();
  try {
    const result = await api(`/api/gallery/saved-searches/${selected.id}?revision=${selected.revision}`, { method: 'DELETE' });
    savedSearchItems = result.items;
    savedSearchPending = null;
    savedSearchSelected = null;
    savedSearchDirty = true;
    savedSearchConflict = null;
    renderSavedSearches();
    savedSearchSay('Saved definition deleted. Your editor and gallery filter are unchanged.');
  } catch (err) {
    savedSearchPending = null;
    if (err.status === 409 || err.status === 404) savedSearchReady = false;
    savedSearchSay(`Search was not deleted: ${err.message}`, true);
    syncSavedSearches();
  }
}

// One transition owns all three controls, including the sheet/bin escape.
// Close the old view without reading it, then replace its epoch once. A sheet
// never gets to put back the search it replaced after this explicit apply.
function applyGalleryFilter({ query, favorites, all_models }) {
  clearTimeout(searchTimer);
  leaveBin();
  sheetReturn = null;
  if (sheetMode) void setSheet(false, { reload: false });
  els.search.value = query;
  els.favOnly.checked = favorites;
  els.allModels.checked = all_models;
  const opened = resetGallery();
  void loadRuns();
  return opened;
}

function applySavedSearch() {
  const reason = savedSearchPending ? 'Wait for the saved searches request to finish.' : savedSearchQueryReason();
  if (reason) { savedSearchSay(reason, true); return; }
  const fields = savedSearchFields();
  closeSavedSearches();
  const opened = applyGalleryFilter(fields);
  const picks = selectedIds().length;
  showError(`Gallery filter applied.${picks ? ` ${plural(picks, 'image')} ${picks === 1 ? 'remains' : 'remain'} selected, including picks outside this view.` : ''}`,
    true, { slot: els.galleryError, mirror: false });
  return opened;
}

function closeSavedSearches() {
  if (savedSearchPending === 'write') { savedSearchSay('Wait for the saved search to finish saving.', true); return; }
  ++savedSearchReadTicket;
  savedSearchPending = null;
  els.savedSearchDialog.close();
}

function savedSearchesClosed() {
  if (els.savedSearchDialog.open) return;
  ++savedSearchReadTicket;
  savedSearchPending = null;
  const origin = savedSearchOrigin;
  savedSearchOrigin = null;
  if (origin?.isConnected) origin.focus({ preventScroll: true });
}

function initSavedSearches() {
  els.savedSearches.onclick = () => { void openSavedSearches(); };
  els.savedSearchClose.onclick = closeSavedSearches;
  els.savedSearchDialog.addEventListener('close', savedSearchesClosed);
  els.savedSearchDialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    closeSavedSearches();
  });
  els.savedSearchList.onchange = () => selectSavedSearch(els.savedSearchList.value);
  els.savedSearchNew.onclick = () => newSavedSearch();
  els.savedSearchCurrent.onclick = useCurrentSavedSearch;
  els.savedSearchReload.onclick = () => { void reloadSavedSearches(); };
  els.savedSearchSave.onclick = () => { void saveSavedSearch(); };
  els.savedSearchReplace.onclick = () => { void saveSavedSearch({ replace: true }); };
  els.savedSearchDelete.onclick = () => { void deleteSavedSearch(); };
  els.savedSearchApply.onclick = applySavedSearch;
  for (const field of [els.savedSearchName, els.savedSearchQuery, els.savedSearchFavorites, els.savedSearchAllModels]) {
    field.addEventListener('input', savedSearchEdited);
    field.addEventListener('change', savedSearchEdited);
  }
}
