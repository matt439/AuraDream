// Checkpoint selection and the state that belongs to one model selection.
// Declarations only; app.js wires the controls after every script has loaded.
'use strict';

let modelRevision = null;
let modelSwitchSending = false;
let modelSwitching = false;
let modelCataloguePending = false;
let modelCatalogueLoaded = false;
let modelCatalogueEpoch = 0;
let modelCatalogueKind = '';
let modelCatalogueDetails = {};
let modelCatalogueEntries = [];
let modelStatusEpoch = 0;
let modelSwitchError = null;
// An outcome of this page's own press (saved, switched); a load failure or a
// failed switch replaces it.
let modelPickerMessage = null;
// A press on a held action: its reason, said until that action's hold
// changes or the picker reopens.
let modelPressReason = null;
// The settings this page's switch asked for, until the revision it makes
// arrives: that reset keeps the outcome rather than wiping it.
let modelOwnSwitch = null;
// From status: a failed load's detail, a switch that stopped before
// selecting, and what this start's flags or environment set aside.
let modelLoadDetail = null;
let modelServerSwitchError = null;
let modelStartupOverrides = {};
// The picker fields changed here and not yet submitted (model_id,
// quantization, offload, format, load_on_startup). Polls and catalogue
// replies still write every other field.
const modelPickerEditedKeys = new Set();
// The checkpoint last chosen on purpose (a row, the server's selection, a
// switch) and its canonical id. A catalogue for other loading options that
// lacks it chooses it again, or the same checkpoint's build, when it can.
let modelChoiceIntent = null;
let modelPickerStartupSeen = false;
let modelRuntimeSettings = null;
let modelPickerAppliedSettings = null;
let modelPickerDeferredReset = null;
let modelCatalogueRequested = null;
let modelCatalogueFailedAt = 0;
const modelOptionKeys = new WeakMap();
const MODEL_FAVOURITES_KEY = 'local-diffusion.model-favourites.v1';
let modelFavourites = new Set();
let modelFavouriteMessage = null;
let modelFavouriteChanges = new Map();
// The favourites the table is grouped by. A star toggles in place while the
// picker is open (moving its row would move a control under the pointer);
// the next opening groups by the stars as they are then.
let modelTableGrouping = null;

function parseModelFavourites(raw) {
  try {
    const saved = JSON.parse(raw ?? '[]');
    return new Set(Array.isArray(saved) ? saved.filter((id) => typeof id === 'string' && id.trim() && id.length <= 512) : []);
  } catch { return new Set(); }
}

function restoreModelFavourites() {
  try { modelFavourites = parseModelFavourites(localStorage.getItem(MODEL_FAVOURITES_KEY)); }
  catch { /* A blocked store still allows favourites for this page. */ }
  applyUnsavedModelFavourites();
}

function applyUnsavedModelFavourites() {
  for (const [id, on] of modelFavouriteChanges) {
    if (on) modelFavourites.add(id);
    else modelFavourites.delete(id);
  }
}

function modelFavouriteId(modelId) {
  return modelCatalogueDetails[modelId]?.favourite_id || modelId;
}

// The loader selects draw their lists in the page (base-select). Rebuilding or
// reselecting an open one moves its rows under a still pointer, so status
// writes to it and catalogue replies wait for every list to close.
function modelListOpen() {
  if (els.modelPicker.hidden) return false;
  try {
    return [els.modelQuantization, els.modelOffload, els.modelFormat]
      .some((select) => select.matches(':open'));
  } catch { return false; } // No :open: a native popup the page cannot see.
}

function whenModelListsClosed() {
  return new Promise((resolve) => {
    const check = () => (modelListOpen() ? setTimeout(check, 150) : resolve());
    check();
  });
}

function setModelAttribute(element, name, value) {
  if (element.getAttribute(name) !== value) element.setAttribute(name, value);
}

// Replace a list's options only when they differ, so an unchanged catalogue
// reply leaves the rows, and any highlight in them, where they were.
function setModelOptions(control, key, build) {
  if (modelOptionKeys.get(control) === key) return;
  control.replaceChildren(...build());
  modelOptionKeys.set(control, key);
}

function selectModelValue(control, value) {
  if (control.value !== value) control.value = value;
}

// The checkpoint table: one radio per model, with its facts beside it.
function modelRadios() {
  return [...els.modelTable.querySelectorAll('input[name="model-choice"]')];
}

function modelChoice() {
  return modelRadios().find((radio) => radio.checked)?.value ?? '';
}

function chooseModel(id) {
  for (const radio of modelRadios()) {
    const on = radio.value === id;
    if (radio.checked !== on) radio.checked = on;
  }
  syncModelTabStops();
}

function rememberModelChoice(id) {
  if (id) modelChoiceIntent = { id, canonical: modelFavouriteId(id) };
}

// Scroll only the picker's body (never the page) so a control's row shows
// below the sticky heading row.
function revealModelControl(control) {
  const row = control?.closest?.('.model-row');
  if (!row || !els.modelTable.contains(row)) return;
  const body = els.modelPickerBody;
  const inner = body.getBoundingClientRect().top + body.clientTop;
  const top = inner + els.modelTableHead.offsetHeight;
  const bottom = inner + body.clientHeight;
  const box = row.getBoundingClientRect();
  if (box.top < top) body.scrollTop -= top - box.top;
  else if (box.bottom > bottom) body.scrollTop += Math.min(box.bottom - bottom, box.top - top);
}

function modelLabel(id) {
  return modelCatalogueEntries.find((entry) => entry.model_id === id)?.label || id;
}

function syncModelStars() {
  for (const star of els.modelTable.querySelectorAll('.model-star')) {
    const id = star.closest('.model-row').dataset.model;
    const on = modelFavourites.has(modelFavouriteId(id));
    const name = modelLabel(id);
    const glyph = on ? '★' : '☆';
    if (star.textContent !== glyph) star.textContent = glyph;
    setModelAttribute(star, 'aria-pressed', String(on));
    setModelAttribute(star, 'title', on ? `${name} is a favourite — click to remove`
      : `Add ${name} to your favourites in this browser`);
  }
}

function modelFactCell(label, value, detail, note, { muted = false } = {}) {
  const cell = document.createElement('span');
  cell.className = muted ? 'model-fact model-fact-muted' : 'model-fact';
  if (note) cell.title = note;
  const hidden = document.createElement('span');
  hidden.className = 'sr-only';
  hidden.textContent = `${label}: `;
  cell.append(hidden, value || '—');
  if (detail) {
    const small = document.createElement('small');
    small.textContent = detail;
    cell.append(small);
  }
  return cell;
}

// The name cell's second line: the access label (its note as the title) and
// a state badge that only the server's current checkpoint fills
// (syncModelRowStates). The line is always there, so a badge never makes a
// row taller.
function modelNameCell(entry, info, favourite) {
  const name = document.createElement('span');
  name.className = 'model-name';
  const radio = document.createElement('input');
  radio.type = 'radio';
  radio.name = 'model-choice';
  radio.value = entry.model_id;
  const title = document.createElement('span');
  title.className = 'model-title';
  if (favourite) {
    // From the grouping, as the rows are drawn: a star toggled in an open
    // picker does not rename the radio under the user.
    const hidden = document.createElement('span');
    hidden.className = 'sr-only';
    hidden.textContent = 'Favourite, ';
    title.append(hidden);
  }
  title.append(entry.label);
  const line = document.createElement('small');
  if (info.access) {
    const access = document.createElement('span');
    access.className = 'model-access';
    access.dataset.access = info.access;
    access.textContent = info.access;
    if (info.access_note) access.title = info.access_note;
    line.append(access);
  }
  const state = document.createElement('span');
  state.className = 'model-state';
  const dot = document.createElement('span');
  dot.className = 'dot';
  const word = document.createElement('span');
  word.className = 'model-state-word';
  state.append(dot, word);
  line.append(state);
  title.append(line);
  name.append(radio, title);
  return name;
}

function modelTableRow(entry, { favourite = false, index = 0 } = {}) {
  const id = entry.model_id;
  const info = modelCatalogueDetails[id] ?? {};
  const row = document.createElement('div');
  row.className = 'model-row';
  row.dataset.model = id;
  const star = document.createElement('button');
  star.type = 'button';
  star.className = 'ghost model-star';
  star.setAttribute('aria-label', `Star ${entry.label}`);
  const choice = document.createElement('label');
  choice.className = 'model-row-choice';
  const name = modelNameCell(entry, info, favourite);
  const speedDetail = [info.speed_summary, info.speed_basis].filter(Boolean).join(' · ');
  choice.append(
    name,
    modelFactCell('Style', info.style),
    modelFactCell('Refine', info.refine, '', info.refine_note),
    modelFactCell('Size', info.megapixels, info.megapixels_label, info.megapixels_note),
    // Only a category timed with these options reads at full strength.
    modelFactCell('Speed', info.speed_label || 'Unmeasured', speedDetail, info.speed_note,
      { muted: !['Fast', 'Medium', 'Slow'].includes(info.speed_label) }),
  );
  row.append(star, choice);
  if (info.url) {
    const link = document.createElement('a');
    link.className = 'model-link';
    link.href = info.url;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = '↗';
    link.title = `Open ${entry.label}'s model page in a new tab`;
    link.setAttribute('aria-label', `${entry.label} model page (opens in a new tab)`);
    row.append(link);
  }
  // The cells' title notes, for a keyboard or a screen reader too: the radio
  // is described by them (a title on a span inside its label is not).
  const notes = [['Access', info.access_note], ['Refine', info.refine_note],
    ['Size', info.megapixels_note], ['Speed', info.speed_note]]
    .filter(([, note]) => note).map(([label, note]) => `${label}: ${note}`).join(' ');
  if (notes) {
    const described = document.createElement('span');
    described.className = 'sr-only';
    described.id = `model-notes-${index}`;
    described.textContent = notes;
    name.querySelector('input').setAttribute('aria-describedby', described.id);
    row.append(described);
  }
  return row;
}

// One tab stop for the table's stars and links, as for its radios: the
// chosen row's (the first row's while none is), and the row holding the
// focus until it leaves, so a star pressed by keyboard stays reachable.
function syncModelTabStops() {
  const rows = [...els.modelTable.querySelectorAll('.model-row[data-model]')];
  const home = rows.find((row) => row.querySelector('input')?.checked) ?? rows[0];
  const focused = els.modelTable.contains(document.activeElement) ? document.activeElement.closest('.model-row') : null;
  for (const row of rows) {
    const stop = row === home || row === focused ? '0' : '-1';
    for (const control of row.querySelectorAll('.model-star, .model-link')) setModelAttribute(control, 'tabindex', stop);
  }
}

const MODEL_ROW_STATES = {
  ready: ['Loaded', 'The server has this checkpoint loaded'],
  loading: ['Loading…', 'The server is loading this checkpoint'],
  error: ['Load failed', "The server's checkpoint; its last load failed"],
  unloaded: ['Current', "The server's checkpoint; not loaded"],
};

// The server's current checkpoint wears its state in its row. Written in
// place, outside the rebuild key: a poll changes one word, never the rows.
function syncModelRowStates() {
  const current = modelRuntimeSettings?.model_id ?? loadedModel;
  const state = MODEL_ROW_STATES[modelState] ? modelState : 'unloaded';
  const [word, title] = MODEL_ROW_STATES[state];
  for (const row of els.modelTable.querySelectorAll('.model-row[data-model]')) {
    if (row.dataset.model !== current) {
      if (row.hasAttribute('data-state')) row.removeAttribute('data-state');
      continue;
    }
    setModelAttribute(row, 'data-state', state);
    const badge = row.querySelector('.model-state');
    if (!badge) continue;
    setModelAttribute(badge, 'title', title);
    setModelAttribute(badge.querySelector('.dot'), 'class', `dot ${state === 'unloaded' ? '' : state}`.trim());
    const text = badge.querySelector('.model-state-word');
    if (text.textContent !== word) text.textContent = word;
  }
}

function renderModelOptions(wanted) {
  if (els.modelPicker.hidden || !modelTableGrouping) modelTableGrouping = new Set(modelFavourites);
  const grouped = (id) => modelTableGrouping.has(modelFavouriteId(id));
  const favourites = modelCatalogueEntries.filter((entry) => grouped(entry.model_id));
  const others = modelCatalogueEntries.filter((entry) => !grouped(entry.model_id));
  const key = JSON.stringify([
    favourites.map((entry) => entry.model_id), modelCatalogueEntries,
    modelCatalogueEntries.map((entry) => modelCatalogueDetails[entry.model_id] ?? null),
  ]);
  if (modelOptionKeys.get(els.modelTable) !== key) {
    // A rebuild (new loading options, a regrouped opening) keeps the keyboard
    // focus on the same model's control: losing it would close the picker.
    // The body holds it on a first opening, before any row exists.
    const focused = els.modelTable.contains(document.activeElement)
      || document.activeElement === els.modelPickerBody ? document.activeElement : null;
    const focusModel = focused?.closest('.model-row')?.dataset.model;
    const focusKind = focused?.classList.contains('model-star') ? '.model-star'
      : focused?.classList.contains('model-link') ? '.model-link' : 'input';
    const children = [els.modelTableHead];
    let index = 0;
    for (const [label, entries] of favourites.length ? [['Favourites', favourites], ['Other models', others]] : [['', others]]) {
      if (!entries.length) continue;
      if (label) {
        const heading = document.createElement('div');
        heading.className = 'model-group';
        heading.textContent = label;
        children.push(heading);
      }
      const favourite = label === 'Favourites';
      children.push(...entries.map((entry) => modelTableRow(entry, { favourite, index: index++ })));
    }
    els.modelTable.replaceChildren(...children);
    modelOptionKeys.set(els.modelTable, key);
    // Checked before the focus lands: on a first opening the body held it,
    // and the chosen row, not the first, takes it.
    chooseModel(wanted);
    if (focused) {
      const row = [...els.modelTable.querySelectorAll('.model-row')].find((r) => r.dataset.model === focusModel);
      const target = row?.querySelector(focusKind) ?? row?.querySelector('input')
        ?? modelRadios().find((radio) => radio.checked) ?? modelRadios()[0] ?? els.modelPickerBody;
      target.focus({ preventScroll: true });
      // A pointer resting on the body keeps its rows where they are.
      if (!els.modelPickerBody.matches(':hover')) revealModelControl(target);
    }
  }
  // A different weight format can expose a different set of model ids. Then
  // no row is chosen, and the actions say to choose one, rather than silently
  // selecting the head of the list.
  chooseModel(wanted);
  syncModelStars();
  syncModelRowStates();
}

function toggleModelFavourite(modelId) {
  // Merge against storage at the time of the click: another tab may have
  // saved a star before its asynchronous storage event reaches this one.
  restoreModelFavourites();
  const id = modelFavouriteId(modelId);
  if (modelFavourites.has(id)) modelFavourites.delete(id);
  else modelFavourites.add(id);
  modelFavouriteChanges.set(id, modelFavourites.has(id));
  try {
    localStorage.setItem(MODEL_FAVOURITES_KEY, JSON.stringify([...modelFavourites]));
    modelFavouriteChanges.clear();
    modelFavouriteMessage = null;
  } catch {
    modelFavouriteMessage = 'Favourites changed for this page, but this browser could not save them for next time.';
  }
  // The star changes in place; the choice and pending loading settings stay.
  syncModelStars();
  syncModelSwitch();
}

function receiveModelFavourites(event) {
  if (event.key !== MODEL_FAVOURITES_KEY && event.key !== null) return;
  try { if (event.storageArea !== localStorage) return; }
  catch { return; }
  // Read the latest store, rather than an event that may precede this tab's
  // own recent click. Keep session-only changes after a refused write.
  restoreModelFavourites();
  if (!modelFavouriteChanges.size) modelFavouriteMessage = null;
  if (modelCatalogueLoaded && els.modelPicker.hidden) renderModelOptions(modelChoice());
  else syncModelStars();
  syncModelSwitch();
}

function modelChangeHold() {
  return modelSwitchSending || modelSwitching ? 'Wait for the model settings change to finish' : '';
}

function modelPickerSettings() {
  return {
    model_id: modelChoice(),
    quantization: els.modelQuantization.value,
    offload: els.modelOffload.value,
    format: els.modelFormat.value || undefined,
    load_on_startup: els.modelLoadOnStartup.checked,
  };
}

function modelLoaderChanged() {
  const selected = modelPickerSettings();
  return !modelRuntimeSettings || ['model_id', 'quantization', 'offload'].some((key) => selected[key] !== modelRuntimeSettings[key])
    || (selected.quantization === 'nunchaku' && selected.format !== modelRuntimeSettings.format);
}

function modelCatalogueKey(selected) {
  return `${selected.quantization}|${selected.quantization === 'nunchaku' ? selected.format : ''}|${selected.offload}`;
}

// Work in this page a switch waits for; the picker never opens by itself
// over it either.
function modelWorkHold() {
  return modelChangeHold()
    || (submitting ? 'Wait for the job submission to finish' : '')
    || (busy || queuedIds.length ? 'A job is queued or running; switch once the image queue finishes' : '')
    || (elsewhereBusy ? 'A job started in another tab or window is running; switch once it ends' : '')
    || (expandingPrompt ? 'Wait for prompt expansion to finish' : '')
    || (critiquing ? 'Wait for the critic to finish' : '')
    || (cardHolder ? 'Wait for the current GPU work to finish' : '');
}

// The last request failed and none has succeeded since, its retry included.
function modelCatalogueFailed() {
  return !modelCatalogueLoaded && modelCatalogueFailedAt > 0;
}

function modelSwitchHold({ load = true } = {}) {
  return serverOffline ? 'The server is not answering'
    : modelWorkHold()
    || (modelState === 'loading' ? 'The weights are still loading' : '')
    || (modelCatalogueFailed() ? 'The checkpoint list could not be loaded; retrying' : '')
    || (!modelCatalogueLoaded || modelCataloguePending ? 'The checkpoint list is still loading' : '')
    || (!modelChoice() ? 'Choose a checkpoint compatible with these settings' : '')
    || (load && !modelLoaderChanged() && modelState === 'ready' ? 'These model settings are already loaded; use Apply & save to remember them' : '');
}

function modelOptionLabel(select, value) {
  return [...select.options].find((option) => option.value === value)?.textContent || value || '';
}

// The server's checkpoint and loading options, as the note names them.
function modelRuntimeLabel() {
  const current = modelRuntimeSettings;
  return [modelLabel(current.model_id), modelOptionLabel(els.modelQuantization, current.quantization),
    ...(current.quantization === 'nunchaku' ? [modelOptionLabel(els.modelFormat, current.format)] : []),
    modelOptionLabel(els.modelOffload, current.offload)].filter(Boolean).join(' · ');
}

// Edits that differ from the server's settings, which Revert puts back.
function modelPickerUnsaved() {
  return modelPickerEditedKeys.size > 0 && !!modelRuntimeSettings
    && (modelLoaderChanged() || els.modelLoadOnStartup.checked !== modelRuntimeSettings.load_on_startup);
}

const MODEL_PREFERENCE_NAMES = {
  model_id: 'checkpoint', quantization: 'quantization', offload: 'offload', load_on_startup: 'startup load',
};

function joinWords(words) {
  return words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} and ${words.at(-1)}`;
}

// The server's record of what this start's flags or environment set aside,
// as a sentence: a saved choice does not apply at a start like this one.
function modelOverridesSentence() {
  const entries = Object.entries(modelStartupOverrides ?? {}).filter(([key]) => MODEL_PREFERENCE_NAMES[key]);
  if (!entries.length) return '';
  const sources = [...new Set(entries.map(([, source]) => source))];
  if (sources.includes('--defaults')) {
    return 'This server was started with --defaults, which ignores saved settings; they apply at a start without it.';
  }
  const names = Object.keys(MODEL_PREFERENCE_NAMES).filter((key) => entries.some(([k]) => k === key));
  const model = entries.find(([key]) => key === 'model_id')?.[1];
  return `This server was started with ${joinWords(sources)}, which ${sources.length > 1 ? 'override' : 'overrides'} `
    + `the saved ${joinWords(names.map((key) => MODEL_PREFERENCE_NAMES[key]))} at a start like this one`
    + (model ? `; when ${model} names another checkpoint, its saved loading options are set aside too.` : '.');
}

// What an accepted press did, by name, with the startup choice.
function modelSavedMessage(selected, before, { load, wasReady, startup }) {
  const name = modelLabel(selected.model_id);
  const changed = !before || ['model_id', 'quantization', 'offload'].some((key) => selected[key] !== before[key])
    || (selected.quantization === 'nunchaku' && selected.format !== before.format);
  // A load of the same settings is the state line's to follow.
  const done = load && changed ? `Switched to ${name}; settings saved.`
    : !changed ? `Settings saved.${wasReady && !load ? ` ${name} stays loaded.` : ''}`
    : `Settings saved; ${wasReady ? `${modelLabel(before.model_id)} was unloaded. ${name}` : name} loads on Load, Switch & load or Generate.`;
  return [done, startup ? 'They will load automatically on startup.' : 'Next startup will wait for you to load or generate.',
    modelOverridesSentence()].filter(Boolean).join(' ');
}

// The note: what went wrong or what a press did, then the server's state or
// the unsaved edits (with Revert). Holds stay in the actions' titles until
// one is pressed.
function modelPickerNote() {
  const current = modelLabel(modelRuntimeSettings?.model_id ?? loadedModel);
  const lead = modelSwitchError || (modelPressReason ? `${modelPressReason.why}.` : '')
    || modelServerSwitchError
    || (modelState === 'error' && modelLoadDetail ? `Couldn't load ${current}: ${modelLoadDetail}` : '')
    || modelFavouriteMessage || modelPickerMessage || '';
  const firstLoad = 'A first load may download weights and take several minutes.';
  const unsaved = modelState !== 'loading' && modelPickerUnsaved();
  const state = modelState === 'loading' ? `Loading ${current}… ${firstLoad}`
    : unsaved ? `Unsaved changes. ${modelState === 'ready' ? 'Loaded' : "Server's choice"}: ${modelRuntimeLabel()}.`
      + (modelLoaderChanged() ? ` ${firstLoad}` : '')
    : lead || modelState === 'error' ? ''
    : modelState === 'ready' ? `${current} is loaded.`
    : `${current} is not loaded. ${firstLoad}`;
  return { text: [lead, state].filter(Boolean).join(' '), unsaved };
}

function syncModelSwitch() {
  // A refusal goes once its action's reason changes.
  if (modelPressReason && modelSwitchHold({ load: modelPressReason.load }) !== modelPressReason.why) modelPressReason = null;
  const why = modelSwitchHold();
  const saveWhy = modelSwitchHold({ load: false });
  syncModelRowStates();
  syncModelTabStops();
  // Every poll runs this more than once; write only what changed.
  setModelAttribute(els.modelSwitch, 'aria-disabled', String(!!why));
  setModelAttribute(els.modelSwitch, 'title', why || 'Remember these settings and load the selected checkpoint');
  // Apply & save releases a loaded model whose settings it changes.
  const unloads = modelState === 'ready' && !!modelRuntimeSettings && modelChoice() !== '' && modelLoaderChanged();
  const current = modelLabel(modelRuntimeSettings?.model_id ?? loadedModel);
  setModelAttribute(els.modelSave, 'aria-disabled', String(!!saveWhy));
  setModelAttribute(els.modelSave, 'title', saveWhy || (unloads
    ? `Save and unload ${current} now; ${modelLabel(modelChoice())} loads on Load, Switch & load or Generate`
    : 'Remember these settings without starting a new load'));
  const help = unloads
    ? `Apply & save saves these settings and unloads ${current} now; Switch & load loads ${modelLabel(modelChoice())} now.`
    : 'Apply & save saves these settings without loading them; Switch & load loads them now.';
  if (els.modelPickerHelp.textContent !== help) els.modelPickerHelp.textContent = help;
  // A list that could not be fetched stays, dimmed, rather than moving rows.
  const stale = modelCatalogueFailed();
  if (els.modelTable.classList.contains('model-table-stale') !== stale) els.modelTable.classList.toggle('model-table-stale', stale);
  setModelAttribute(els.modelTable, 'aria-busy', String(stale));
  const formatHidden = els.modelQuantization.value !== 'nunchaku';
  if (els.modelFormatField.hidden !== formatHidden) els.modelFormatField.hidden = formatHidden;
  for (const select of [els.modelQuantization, els.modelOffload, els.modelFormat]) {
    const option = select.selectedOptions[0];
    setModelAttribute(select, 'title', option?.dataset.description || option?.textContent || '');
  }
  for (const [select, help] of [[els.modelQuantization, els.modelQuantizationHelp], [els.modelOffload, els.modelOffloadHelp]]) {
    const text = select.selectedOptions[0]?.dataset.description || '';
    if (help.textContent !== text) help.textContent = text;
  }
  // Which card each Nunchaku format is for, all at once: the choice is the
  // GPU generation's, not a trade-off one option's text explains.
  const formats = [...els.modelFormat.options]
    .map((option) => `${option.textContent}: ${option.dataset.description || option.textContent}`).join('; ');
  if (els.modelFormatHelp.textContent !== formats) els.modelFormatHelp.textContent = formats;
  if (!els.modelPicker.hidden) {
    const { text, unsaved } = modelPickerNote();
    if (els.modelPickerNoteText.textContent !== text) els.modelPickerNoteText.textContent = text;
    if (els.modelRevert.hidden === unsaved) els.modelRevert.hidden = !unsaved;
  }
}

// Revert puts the server's settings back in every field. The focus, which
// the hidden button gives up, goes to the chosen row.
function revertModelPicker() {
  modelPickerEditedKeys.clear();
  modelPressReason = null;
  applyModelSettings(true);
  syncModelSwitch();
  const target = modelRadios().find((radio) => radio.checked) ?? els.modelPickerBody;
  target.focus({ preventScroll: true });
  revealModelControl(target);
}

function modelPickerEdited(key, { catalogue = false } = {}) {
  modelPickerEditedKeys.add(key);
  if (key === 'model_id') rememberModelChoice(modelChoice());
  modelSwitchError = null;
  modelPickerMessage = null;
  syncModelSwitch();
  if (catalogue) void refreshModelCatalogue();
}

// Polls update the current settings without erasing unsubmitted picker edits.
// A changed server revision retires those edits along with the old checkpoint.
function receiveModelSettings(status, { reset = false } = {}) {
  const previous = modelRuntimeSettings;
  modelRuntimeSettings = {
    model_id: status.model_id ?? loadedModel,
    quantization: status.quantization ?? modelRuntimeSettings?.quantization,
    offload: status.offload ?? modelRuntimeSettings?.offload,
    format: status.format ?? modelRuntimeSettings?.format,
    load_on_startup: status.load_on_startup ?? modelRuntimeSettings?.load_on_startup ?? false,
  };
  if (status.startup_overrides) modelStartupOverrides = status.startup_overrides;
  // A new failure takes the place of an outcome that said all went well.
  const failed = status.state === 'error' ? status.detail ?? null : null;
  const switchError = status.switch_error ?? null;
  if ((failed && failed !== modelLoadDetail) || (switchError && switchError !== modelServerSwitchError)) {
    modelPickerMessage = null;
    modelOwnSwitch = null;
  }
  modelLoadDetail = failed;
  modelServerSwitchError = switchError;
  if (!modelSwitchSending && previous && previous.load_on_startup !== modelRuntimeSettings.load_on_startup) {
    modelPickerMessage = null;
  }
  if (reset) {
    modelPickerEditedKeys.clear();
    // The revision this page's own switch made keeps the note saying so.
    const own = modelOwnSwitch && ['model_id', 'quantization', 'offload']
      .every((key) => modelOwnSwitch[key] === modelRuntimeSettings[key]);
    if (!own) modelPickerMessage = null;
    modelOwnSwitch = null;
  }
  if (!modelCatalogueLoaded && !modelCataloguePending && !els.modelPicker.hidden
      && Date.now() - modelCatalogueFailedAt > 10000) {
    void refreshModelCatalogue(); // A failed list retries while the picker is open.
  }
  applyModelSettings(reset);
}

// Show the server's settings in every field the user has not edited. Only
// fields that changed since the last apply are written, so an unchanged poll
// leaves the controls alone. An open list keeps its own value until it closes; the other
// controls take theirs at once, so a pick made in that list still sits on the
// server's new loading options.
function applyModelSettings(reset) {
  if (!modelCatalogueLoaded) return;
  const edited = (key) => modelPickerEditedKeys.has(key);
  const selected = modelRuntimeSettings;
  const applied = modelPickerAppliedSettings;
  if (!edited('model_id') && (reset || !applied || applied.model_id !== selected.model_id)) {
    chooseModel(selected.model_id);
    rememberModelChoice(selected.model_id);
  }
  let held = false;
  for (const [control, key] of [
    [els.modelQuantization, 'quantization'], [els.modelOffload, 'offload'], [els.modelFormat, 'format'],
  ]) {
    if (edited(key) || (!reset && applied && applied[key] === selected[key])) continue;
    if (modelListOpen() && control.matches(':open')) held = true;
    else selectModelValue(control, selected[key] ?? '');
  }
  if (held && !modelPickerDeferredReset) {
    // On close, unless the user picked from that list, the held value lands.
    modelPickerDeferredReset = true;
    void whenModelListsClosed().then(() => {
      modelPickerDeferredReset = null;
      applyModelSettings(true);
      syncModelSwitch();
    });
  }
  if (!edited('load_on_startup') && (reset || !applied || applied.load_on_startup !== selected.load_on_startup)) {
    els.modelLoadOnStartup.checked = selected.load_on_startup;
  }
  modelPickerAppliedSettings = { ...selected };
  // Edited loading options keep the catalogue their own change asked for.
  if (['quantization', 'offload', 'format'].some(edited)) return;
  // A runtime id missing from its own catalogue (an off-catalogue launch id
  // under other loading options) is asked for once, not on every poll.
  if (modelCatalogueKind !== modelCatalogueKey(selected)
      || (!modelChoice() && modelCatalogueRequested !== `${modelCatalogueKind}|${selected.model_id}`)) {
    void refreshModelCatalogue({ current: true });
  }
}

// The first usable status decides once. A later unload or recovery must not
// reopen a picker the user dismissed, and an automatic open never takes the caret.
function offerModelPicker(status) {
  if (modelPickerStartupSeen) return;
  modelPickerStartupSeen = true;
  if (!['unloaded', 'error'].includes(status.state) || modelWorkHold() || status.active_jobs?.length) return;
  showModelPicker(true, { focus: false });
}

function showModelPicker(on, { focus = true } = {}) {
  modelPickerStartupSeen = true;
  els.modelPicker.hidden = !on;
  els.models.setAttribute('aria-expanded', String(on));
  if (on) {
    // An opening starts from the state, not an earlier press's words.
    modelPressReason = null;
    modelPickerMessage = null;
    // Each opening groups the table by the stars as they are now.
    modelTableGrouping = new Set(modelFavourites);
    if (modelCatalogueLoaded) renderModelOptions(modelChoice() || modelRuntimeSettings?.model_id);
    syncModelSwitch();
    void refreshModelCatalogue();
    if (focus) {
      const target = modelRadios().find((radio) => radio.checked) ?? modelRadios()[0] ?? els.modelPickerBody;
      target.focus({ preventScroll: true });
      // Opening scrolls the picker's own body to the focused row, not the page.
      revealModelControl(target);
    }
  } else if (focus && els.modelPicker.contains(document.activeElement)) {
    els.models.focus({ preventScroll: true });
  }
}

async function refreshModelCatalogue({ current = false } = {}) {
  const epoch = ++modelCatalogueEpoch;
  modelCataloguePending = true;
  syncModelSwitch();
  // A failed catalogue request must not make its retry use the old runtime
  // quantization while the controls still show the user's edited choice.
  const selected = !els.modelQuantization.options.length || current ? modelRuntimeSettings : modelPickerSettings();
  const query = new URLSearchParams();
  if (selected?.quantization) query.set('quantization', selected.quantization);
  if (selected?.format) query.set('format', selected.format);
  if (selected?.offload) query.set('offload', selected.offload);
  try {
    const body = await api(`/api/models${query.size ? `?${query}` : ''}`);
    if (epoch !== modelCatalogueEpoch) return;
    if (modelListOpen()) {
      await whenModelListsClosed();
      if (epoch !== modelCatalogueEpoch) return;
    }
    const wanted = { ...(selected ?? body) };
    // Keep edits made while this request was in flight or a list was open,
    // including a checkpoint picked during a poll's own refresh and the
    // startup checkbox, neither of which asks for a catalogue of its own.
    const edits = modelPickerSettings();
    for (const key of modelPickerEditedKeys) {
      if (edits[key] !== '' && edits[key] !== undefined) wanted[key] = edits[key];
    }
    modelCatalogueEntries = body.models ?? [];
    modelCatalogueDetails = body.details ?? {};
    const wantedModel = wanted.model_id ?? loadedModel;
    // Only the server decides which checkpoint/loader combinations are valid.
    // A list for other loading options that lacks the choice takes the last
    // deliberate one, else the same checkpoint's build it lists, else none.
    const listed = (id) => modelCatalogueEntries.some((entry) => entry.model_id === id);
    let chosen = wantedModel;
    if (!current && !listed(chosen) && modelChoiceIntent) {
      const { id, canonical } = modelChoiceIntent;
      chosen = listed(id) ? id
        : modelCatalogueEntries.find((entry) => modelFavouriteId(entry.model_id) === canonical)?.model_id ?? chosen;
      if (chosen !== wantedModel) modelPickerEditedKeys.add('model_id');
    }
    renderModelOptions(chosen);
    for (const [control, key, choices] of [
      [els.modelQuantization, 'quantization', 'quantizations'],
      [els.modelOffload, 'offload', 'offloads'],
      [els.modelFormat, 'format', 'formats'],
    ]) {
      const entries = body[choices] ?? [];
      setModelOptions(control, JSON.stringify(entries), () => entries.map((entry) => {
        const option = new Option(entry.label, entry.value);
        option.dataset.description = entry.description || entry.label;
        option.title = option.dataset.description;
        return option;
      }));
      const value = wanted[key] ?? body[key];
      if ([...control.options].some((o) => o.value === value)) selectModelValue(control, value);
    }
    // Unless edited, the server's checkbox as it is now: a poll may have
    // brought another tab's save while this request was in flight.
    els.modelLoadOnStartup.checked = (modelPickerEditedKeys.has('load_on_startup') ? wanted.load_on_startup
      : modelRuntimeSettings?.load_on_startup ?? body.load_on_startup) ?? false;
    // Remember what this reply installed. If the server changed while it was
    // in flight, the next status still reconciles that change. A checkpoint the
    // list lacks is recorded as asked for, so polls stop rewriting it.
    modelPickerAppliedSettings = { ...modelPickerSettings(), model_id: modelChoice() || wantedModel };
    modelCatalogueKind = modelCatalogueKey(modelPickerSettings());
    modelCatalogueRequested = `${modelCatalogueKind}|${wantedModel}`;
    modelCatalogueLoaded = true;
    modelSwitchError = null;
  } catch (err) {
    if (epoch !== modelCatalogueEpoch) return;
    modelSwitchError = err.message;
    modelCatalogueLoaded = false;
    modelCatalogueFailedAt = Date.now();
  } finally {
    if (epoch === modelCatalogueEpoch) {
      modelCataloguePending = false;
      syncModelSwitch();
    }
  }
}

async function switchModel({ load = true } = {}) {
  const why = modelSwitchHold({ load });
  if (why) {
    modelPressReason = { why, load };
    syncModelSwitch();
    if (serverOffline) void refreshStatus();
    return;
  }
  const selected = modelPickerSettings();
  const before = modelRuntimeSettings ? { ...modelRuntimeSettings } : null;
  const wasReady = modelState === 'ready';
  modelSwitchError = null;
  modelPickerMessage = null;
  modelPressReason = null;
  modelOwnSwitch = selected;
  // Flush this checkpoint's edits while its identity still describes them.
  clearTimeout(panelTimer);
  writePanel();
  modelSwitchSending = true;
  ++modelStatusEpoch; // A status already in flight describes the old selection.
  syncPromptExpansion();
  try {
    const status = await api('/api/model/switch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...selected, load }),
    });
    modelPickerEditedKeys.clear();
    rememberModelChoice(selected.model_id);
    receiveStatus(status, Date.now());
    // No switch in the background: no revision of its own will come.
    if (!status.switching) modelOwnSwitch = null;
    modelPickerMessage = modelSavedMessage(selected, before, { load, wasReady, startup: status.load_on_startup });
    syncModelSwitch();
  } catch (err) {
    modelOwnSwitch = null;
    modelSwitchError = err.message;
    showError(err.message);
  } finally {
    modelSwitchSending = false;
    syncPromptExpansion();
    syncModelSwitch();
    void refreshStatus();
  }
}

// Also used for a switch made in another tab, including A -> B -> A between
// polls. Epochs retire CPU answers and adapter lists that began before it.
function checkpointChanged(previousModel) {
  modelSwitchError = null;
  if (previousModel) {
    if (els.confirmDialog.open) els.confirmDialog.close('');
    keepSampling(previousModel);
    for (const key of Object.keys(owned)) owned[key] = false;
    samplingRestored = null;
    els.trueCfg.value = '';
    for (const key of SAMPLING_AXES) els[key].value = '';
    clearReuseNote();
    if (resetNote) retireResetUndo(resetNote);
    unsentRefine = null;
    if (refineSource) {
      // A pass prepared under the old schedule cannot be submitted unchanged.
      refineSource = null;
      refineReturn = null;
      els.refineDialog.close();
      showError('The checkpoint changed. Reopen Refine to use its new sampling settings.', true);
    }
    els.gridResume.checked = false;
    pendingLora = '';
    els.lora.replaceChildren(new Option('none', ''));
  }
  clearAnalysis();
  ++comparePreviewEpoch;
  tagMeasured = null;
  tagSlots = null;
  tagVerdicts = { model: null, results: new Map() };
  ++tagMeasureEpoch;
  ++tagValidationEpoch;
  ++tagSuggestionEpoch;
  ++tagEditorSuggestionEpoch;
  hideTagSuggestions();
  ++refineEpoch;
  ++refineCheckEpoch;
  ++refineTagEpoch;
  ++previewEpoch;
  ++loraEpoch;
  lorasLoaded = false;
  lorasLive = false;
  loraRefusal = null;
  if (previousModel) {
    showExamples(false);
    showTagHelper(false);
  }
}

function submissionModelHold(submitted) {
  return submitted.model && submitted.model !== loadedModel
    ? `This request was prepared for ${checkpointName(submitted.model)}. Switch back to that checkpoint before running it again.`
    : !submitted.model && submitted.revision !== undefined && submitted.revision !== modelRevision
      ? 'The checkpoint selection changed since this request was prepared. Use the panel to submit a new request.'
    : '';
}
