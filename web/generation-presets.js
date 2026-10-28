// Named raw drafts. Declarations only; app.js wires the dialog, with no startup read.
'use strict';

let setupsItems = [];
let setupsSelected = null;
let setupsReady = false;
let setupsWriteRecovery = null;
let setupsPending = null;
let setupsTicket = 0;
let setupsOrigin = null;
let setupsDirty = false;
let setupsCheck = null;
let setupsCheckReason = '';
let setupsUndo = null;
let setupsComposerEpoch = 0;

function setupSay(text, error = false) {
  els.setupsStatus.textContent = text;
  els.setupsStatus.classList.toggle('error', error);
}

function setupWriteRecoveryReason() {
  if (!setupsWriteRecovery) return '';
  return setupsWriteRecovery.needsReload
    ? 'The previous setup write result is uncertain. Reload list before selecting a saved setup or choosing New.'
    : 'The previous setup write result is uncertain. Select a saved setup or choose New before writing again.';
}

function chooseSetupWriteRecovery(userChoice) {
  if (!setupsWriteRecovery) return true;
  if (!userChoice || setupsWriteRecovery.needsReload || !setupsReady) {
    setupSay(setupWriteRecoveryReason(), true);
    syncSetups();
    return false;
  }
  setupsWriteRecovery = null;
  return true;
}

function uncertainSetupWrite(err, operation) {
  if (err.status && err.status < 500) return false;
  setupsWriteRecovery = { needsReload: true };
  setupsReady = false;
  setupSay(`Setup ${operation} result is uncertain. Your name and composer are kept. Reload list, then select a saved setup or choose New before writing again.`, true);
  return true;
}

function setupPerformanceReason() {
  return executionSubmitHold() || (selectedExecution || performanceChoiceValue(els.modelCompilation)
    || performanceChoiceValue(els.modelAttention) ? EXECUTION_UNAVAILABLE : '');
}

function setupCaptureReason() {
  const held = setupPerformanceReason();
  if (held) return held;
  if (!loadedModel || !lastDefaults || modelSwitching) return 'Wait for the selected checkpoint settings.';
  const omitted = [];
  if (els.lora.value || pendingLora) omitted.push('LoRA');
  if (referenceIds.length) omitted.push('reference images');
  if (els.gridSection.open && els.gridLoraScale.value.trim()) omitted.push('LoRA weight axis');
  if (els.thenRefine.checked) omitted.push('Then refine');
  if (omitted.length) return `This setup cannot save ${omitted.join(', ')}. Turn them off before saving.`;
  const numbers = [...panelNumberInputs(), [els.gridSeeds, 'Grid seeds'], [els.gridBudget, 'Budget']];
  const bad = numbers.find(([el]) => el.validity.badInput);
  if (bad) return `${bad[1]} has an incomplete number. Finish typing it before saving.`;
  return '';
}

// Do not use collect(): it trims text and drops inactive inputs. The saved
// draft owns only sampling the user took off the profile, including zero.
function captureSetup() {
  const override = (key) => owned[key] && els[key].value.trim() !== '' ? +els[key].value : null;
  const nullable = (el) => el.value.trim() === '' ? null : +el.value;
  return {
    mode: els.gridSection.open ? 'grid' : 'generate',
    prompt: els.prompt.value, negative_prompt: els.negative.value,
    width: +els.width.value, height: +els.height.value,
    steps: override('steps'), guidance_scale: override('cfg'),
    mu: override('mu'), true_cfg: override('trueCfg'),
    seed: nullable(els.seed), num_images: +els.batch.value, runs: +els.runs.value,
    grid: els.gridSection.open ? {
      prompt_axis: els.gridPrompt.value, prompt_mode: els.gridPromptMode.value,
      steps_axis: els.gridSteps.value, cfg_axis: els.gridCfg.value, mu_axis: els.gridMu.value,
      size_axis: els.gridSize.value, seeds: +els.gridSeeds.value,
      seed_axis: els.gridSeedList.value, budget_minutes: nullable(els.gridBudget),
    } : null,
  };
}

function setupThenRefineState() {
  return { checked: els.thenRefine.checked, scale: trScale,
    strength: els.thenRefineStrength.value, steps: els.thenRefineSteps.value };
}

// Include displayed numbers, ownership and unsupported/raw performance intent;
// none of these snapshots is serialized as a saved setup.
function setupComposerState() {
  return JSON.stringify({ state: panelState(), values: reuseFields(),
    resume: els.gridResume.checked, then: setupThenRefineState(),
    compilation: performanceChoiceValue(els.modelCompilation), attention: performanceChoiceValue(els.modelAttention),
    invalid: invalidExecutionDraft, execution: selectedExecution,
    model: loadedModel, revision: modelRevision, server: serverStartedAt });
}

function setupRestoreReason() {
  if (setupsPending) return 'Wait for the saved setup request.';
  if (setupWriteRecoveryReason()) return setupWriteRecoveryReason();
  if (!setupsReady || !setupsSelected) return 'Select a saved setup from the current list.';
  if (setupsSelected.model_id !== loadedModel) return `This setup belongs to ${checkpointName(setupsSelected.model_id)}. Select that checkpoint in Models first.`;
  return setupPerformanceReason() || setupsCheckReason;
}

function syncSetups() {
  if (!els.setupsDialog.open) return;
  const hold = (button, reason, title) => {
    button.setAttribute('aria-disabled', String(!!reason));
    button.title = reason || title;
  };
  const pending = setupsPending ? 'Wait for the saved setup request.' : '';
  const nameValue = els.setupsName.value.trim();
  const name = !nameValue ? 'Give the setup a name.'
    : [...nameValue].length > 80 || !els.setupsName.validity.valid ? 'The setup name is longer than 80 characters.' : '';
  const ready = setupWriteRecoveryReason() || (!setupsReady ? 'Reload the saved list before changing it.' : '');
  const selection = !setupsSelected ? 'Select a saved setup first.' : '';
  const mismatch = setupsSelected && setupsSelected.model_id !== loadedModel
    ? `Select ${checkpointName(setupsSelected.model_id)} in Models first.` : '';
  hold(els.setupsNew, pending || (setupsWriteRecovery?.needsReload ? setupWriteRecoveryReason() : ''), 'Prepare to save the current composer as a new setup');
  hold(els.setupsSave, pending || ready || name || setupCaptureReason(), 'Save current composer as a new setup');
  hold(els.setupsUpdate, pending || ready || selection || mismatch || name || setupCaptureReason(), 'Replace this setup with the current composer');
  hold(els.setupsRename, pending || ready || selection || mismatch || name || setupPerformanceReason(), 'Save this name, keeping the saved settings');
  hold(els.setupsDelete, pending || ready || selection, 'Delete only this saved setup');
  hold(els.setupsRestore, setupRestoreReason(), 'Validate and restore this setup with Undo; no work is queued');
  hold(els.setupsReload, pending, 'Reread the list; select an item again to replace the editor');
  hold(els.setupsClose, setupsPending === 'write' ? pending : '', 'Close Saved setups');
  hold(els.setupsModels, pending, 'Open Models without switching or loading');
  for (const field of [els.setupsList, els.setupsName, els.setupsAll]) field.disabled = setupsPending === 'write';
  els.setupsModels.hidden = !mismatch;
}

function renderSetups() {
  els.setupsList.replaceChildren();
  const items = setupsItems.filter((item) => els.setupsAll.checked || item.model_id === loadedModel)
    .sort((a, b) => Number(b.model_id === loadedModel) - Number(a.model_id === loadedModel));
  for (const item of items) {
    const option = new Option(`${item.name} · ${checkpointName(item.model_id)}`, item.id);
    els.setupsList.append(option);
  }
  if (!items.length) els.setupsList.append(new Option('No saved setups in this view', ''));
  els.setupsList.value = setupsSelected?.id ?? '';
  if (setupsWriteRecovery) els.setupsList.selectedIndex = -1;
  syncSetups();
}

function setupSummary(payload, checked = null) {
  const clears = [];
  if (els.lora.value || pendingLora) clears.push('LoRA');
  if (referenceIds.length) clears.push(`${referenceIds.length} reference image(s)`);
  if (els.gridLoraScale.value) clears.push('LoRA weight axis');
  if (els.thenRefine.checked) clears.push('Then refine');
  const lines = [`${payload.mode === 'grid' ? 'Grid' : 'Generate'} · ${payload.width} × ${payload.height}`,
    `Restore clears conditioning and turns Then refine off${clears.length ? `: ${clears.join(', ')}.` : '.'}`,
    'Undo restores the previous whole draft. Restore queues no work.',
    `Seed: ${payload.seed ?? 'random at submission'} · Batch: ${payload.num_images} · Repeat: ${payload.runs}`];
  for (const [key, label] of [['steps', 'Steps'], ['guidance_scale', 'CFG'], ['mu', 'Mu'], ['true_cfg', 'True CFG']]) {
    lines.push(`${label}: saved ${payload[key] ?? 'profile default'}${checked ? `; effective ${checked.effective[key] ?? 'inactive'}` : ''}`);
  }
  lines.push(`Prompt:\n${payload.prompt}`, `Negative (kept even when inactive):\n${payload.negative_prompt || '(empty)'}`);
  if (payload.grid) {
    const grid = payload.grid;
    lines.push(`Variables (${grid.prompt_mode}):\n${grid.prompt_axis || '(none)'}`,
      `Axes — Steps: ${grid.steps_axis || '(base)'}; CFG: ${grid.cfg_axis || '(base)'}; Mu: ${grid.mu_axis || '(base)'}; Size: ${grid.size_axis || '(base)'}`,
      `Seeds: ${grid.seed_axis || grid.seeds}; Budget: ${grid.budget_minutes ?? 'none'} minutes. Request Repeat: 1; Resume: off.`);
  }
  if (checked) {
    const labels = { prompt: 'Prompt', negative_prompt: 'Negative', steps: 'Steps', guidance_scale: 'CFG',
      mu: 'Mu', true_cfg: 'True CFG', size: 'Size', seed: 'Seeds', lora_scale: 'LoRA weight' };
    const shown = (values) => Object.entries(values).filter(([key]) => key !== 'lora_scale')
      .map(([key, entries]) => `${labels[key] ?? key}: ${entries.map((value) => value ?? 'inactive').join(' | ') || '(base seed)'}`).join('\n');
    lines.push(`Effective: ${checked.cells} cells, ${checked.images} images.`);
    if (checked.inactive.length) lines.push(`Inactive in at least one cell: ${checked.inactive.map((key) => labels[key] ?? key).join(', ')}. Saved inputs are kept.`);
    if (Object.keys(checked.values).length) lines.push(`Requested grid values:\n${shown(checked.values)}\n\nEffective grid values:\n${shown(checked.effective_values)}`);
    if (checked.prompt_sample) lines.push('Random-grid prompts are examples only; no preview seed is adopted.');
  }
  els.setupsSummary.textContent = lines.join('\n\n');
}

function setupValidationBody(item) {
  return { revision: item.revision, model_revision: modelRevision, step_cache: null };
}

async function checkSetup(item, restore = false) {
  const ticket = ++setupsTicket;
  const fingerprint = setupComposerState();
  const epoch = setupsComposerEpoch;
  const name = els.setupsName.value;
  const revision = modelRevision;
  setupsPending = 'validate';
  setupSay(restore ? 'Validating before Restore.' : 'Checking saved and effective settings.');
  syncSetups();
  try {
    const checked = await api(`/api/presets/${item.id}/validate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(setupValidationBody(item)),
    });
    if (ticket !== setupsTicket || !els.setupsDialog.open) return;
    setupsPending = null;
    if (fingerprint !== setupComposerState() || epoch !== setupsComposerEpoch || name !== els.setupsName.value
        || setupsSelected?.id !== item.id || setupsSelected?.revision !== item.revision
        || checked.item_id !== item.id || checked.revision !== item.revision
        || checked.model_id !== loadedModel || checked.model_revision !== revision || modelRevision !== revision) {
      setupsCheck = null;
      setupSay('The draft, setup or checkpoint changed during validation. Press Restore again to check the current draft.', true);
      syncSetups();
      return;
    }
    if (setupPerformanceReason()) { setupSay(setupPerformanceReason(), true); syncSetups(); return; }
    setupsCheck = checked;
    setupsCheckReason = '';
    setupSummary(item.payload, checked);
    if (restore) applySetup(item, checked);
    else { setupSay('Checked. Restore rechecks before changing the composer.'); syncSetups(); }
  } catch (err) {
    if (ticket !== setupsTicket || !els.setupsDialog.open) return;
    setupsPending = null;
    setupsCheck = null;
    setupsCheckReason = err.message;
    if (err.status === 409 || err.status === 404) setupsReady = false;
    setupSay(`Setup was not restored: ${err.message} Reload the list to check again.`, true);
    syncSetups();
  }
}

function selectSetup(id, userChoice = true) {
  if (setupsPending === 'write') return;
  const item = setupsItems.find((entry) => entry.id === id);
  if (!item) return;
  if (!chooseSetupWriteRecovery(userChoice)) { els.setupsList.selectedIndex = -1; return; }
  ++setupsTicket;
  setupsPending = null;
  setupsSelected = item;
  setupsCheck = null;
  setupsCheckReason = '';
  setupsDirty = false;
  els.setupsList.value = id;
  els.setupsName.value = item.name;
  setupSummary(item.payload);
  const reason = setupRestoreReason();
  if (reason) { setupSay(reason); syncSetups(); }
  else void checkSetup(item);
}

function newSetup(userChoice = true) {
  if (setupsPending) { setupSay('Wait for the saved setup request.'); return; }
  if (!chooseSetupWriteRecovery(userChoice)) return;
  setupsSelected = null;
  setupsCheck = null;
  setupsCheckReason = '';
  setupsDirty = false;
  els.setupsList.value = '';
  els.setupsName.value = '';
  setupSummary(captureSetup());
  setupSay(setupCaptureReason() || 'Name this setup, then Save current.');
  syncSetups();
  els.setupsName.focus({ preventScroll: true });
}

async function reloadSetups() {
  if (setupsPending) { setupSay('Wait for the saved setup request.'); return; }
  const ticket = ++setupsTicket;
  setupsPending = 'read';
  setupsReady = false;
  if (setupsWriteRecovery) setupsWriteRecovery.needsReload = true;
  setupSay('Reading saved setups.');
  syncSetups();
  try {
    const result = await api('/api/presets');
    if (ticket !== setupsTicket || !els.setupsDialog.open) return;
    setupsItems = result.items;
    setupsReady = true;
    setupsPending = null;
    if (setupsWriteRecovery) setupsWriteRecovery.needsReload = false;
    renderSetups();
    // Rereading never discards an unsaved name or updates an editor revision.
    // Selecting from the refreshed list is the explicit reread of that item.
    if (setupsWriteRecovery) {
      els.setupsList.selectedIndex = -1;
      setupSay(`List reloaded; your name and composer are unchanged. ${setupWriteRecoveryReason()}`);
      syncSetups();
    } else if (setupsSelected || setupsDirty) setupSay('List reloaded. Select a setup again to read its latest saved revision; your name is kept until then.');
    else newSetup(false);
  } catch (err) {
    if (ticket !== setupsTicket || !els.setupsDialog.open) return;
    setupsPending = null;
    setupSay(`Setups could not be read: ${err.message}`, true);
    syncSetups();
  }
}

async function openSetups() {
  if (els.setupsDialog.open) return;
  setupsOrigin = document.activeElement;
  els.setupsDialog.showModal();
  await reloadSetups();
}

async function saveSetup(action) {
  const button = action === 'rename' ? els.setupsRename : action === 'update' ? els.setupsUpdate : els.setupsSave;
  syncSetups();
  if (button.getAttribute('aria-disabled') === 'true') { setupSay(button.title, true); return; }
  const selected = action === 'new' ? null : setupsSelected;
  const payload = action === 'rename' ? selected.payload : captureSetup();
  const body = { name: els.setupsName.value.trim(), model_id: loadedModel, payload,
    model_revision: modelRevision, step_cache: null,
    ...(selected ? { revision: selected.revision } : {}) };
  setupsPending = 'write';
  setupSay('Saving setup.');
  syncSetups();
  try {
    const result = await api(selected ? `/api/presets/${selected.id}` : '/api/presets', {
      method: selected ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    setupsItems = result.items;
    setupsPending = null;
    setupsDirty = false;
    renderSetups();
    const saved = selected ? setupsItems.find((item) => item.id === selected.id) : setupsItems.at(-1);
    if (saved) selectSetup(saved.id, false);
  } catch (err) {
    setupsPending = null;
    if (err.status === 409 || err.status === 404) setupsReady = false;
    if (!uncertainSetupWrite(err, 'save')) {
      setupSay(`Setup was not saved: ${err.message} Your name is kept; Reload list before selecting its latest revision.`, true);
    }
    syncSetups();
  }
}

async function deleteSetup() {
  syncSetups();
  if (els.setupsDelete.getAttribute('aria-disabled') === 'true') { setupSay(els.setupsDelete.title, true); return; }
  const selected = setupsSelected;
  setupsPending = 'write';
  setupSay('Deleting setup.');
  syncSetups();
  try {
    const result = await api(`/api/presets/${selected.id}?revision=${selected.revision}`, { method: 'DELETE' });
    setupsItems = result.items;
    setupsPending = null;
    setupsSelected = null;
    renderSetups();
    newSetup(false);
    setupSay('Setup deleted. The composer is kept.');
  } catch (err) {
    setupsPending = null;
    if (err.status === 409 || err.status === 404) setupsReady = false;
    if (!uncertainSetupWrite(err, 'delete')) {
      setupSay(`Setup was not deleted: ${err.message} Reload list before selecting its latest revision.`, true);
    }
    syncSetups();
  }
}

function restoreSetup() {
  const reason = setupRestoreReason();
  if (reason) { setupSay(reason, true); syncSetups(); return; }
  void checkSetup(setupsSelected, true);
}

function syncSetupComposer() {
  promptReplaced();
  syncSize();
  syncCfg();
  syncConditioning();
  revealActiveSetup();
  syncRuns();
  syncExecutionControls();
  syncSubmitButtons();
  syncThenRefine();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
}

function applySetup(item, checked) {
  const before = { state: panelState(), values: reuseFields(), owned: { ...owned },
    size: [pickedAspect, pickedTier], then: setupThenRefineState() };
  setupsOrigin = null;
  els.setupsDialog.close();
  clearReuseNote();
  reuseNote = null;
  dropPromptUndo();
  const payload = checked.saved;
  rewriteBox(els.negative, payload.negative_prompt);
  rewriteBox(els.prompt, payload.prompt);
  for (const [field, key] of [['width', 'width'], ['height', 'height'], ['seed', 'seed'], ['batch', 'num_images'], ['runs', 'runs']]) {
    els[field].value = payload[key] ?? '';
  }
  for (const [field, key] of [['steps', 'steps'], ['cfg', 'guidance_scale'], ['mu', 'mu'], ['trueCfg', 'true_cfg']]) {
    owned[field] = payload[key] !== null;
    // Use the profile for an unowned box; normalized inactive values must
    // never erase an explicit saved override.
    els[field].value = payload[key] ?? lastDefaults[key] ?? '';
  }
  setReferences([]);
  els.lora.value = '';
  pendingLora = '';
  els.loraScale.value = 1;
  els.gridLoraScale.value = '';
  const grid = payload.grid;
  for (const [field, key] of [['gridPrompt', 'prompt_axis'], ['gridPromptMode', 'prompt_mode'], ['gridSteps', 'steps_axis'],
    ['gridCfg', 'cfg_axis'], ['gridMu', 'mu_axis'], ['gridSize', 'size_axis'], ['gridSeeds', 'seeds'],
    ['gridSeedList', 'seed_axis'], ['gridBudget', 'budget_minutes']]) {
    els[field].value = grid?.[key] ?? (field === 'gridSeeds' ? 1 : field === 'gridPromptMode' ? 'cross' : '');
  }
  els.gridSection.open = payload.mode === 'grid';
  els.gridResume.checked = false;
  els.gridSeeds.disabled = !!els.gridSeedList.value.trim();
  els.thenRefine.checked = false;
  syncSetupComposer();
  const note = { before, after: setupComposerState(), entry: null, model: loadedModel, revision: modelRevision };
  note.entry = offerPromptUndo(`Setup “${item.name}” restored.`, () => undoSetup(note), {
    kind: 'setup', at: 'box', ms: RESET_UNDO_MS, title: 'Restore the whole previous composer draft',
    onDrop: () => { if (setupsUndo === note) setupsUndo = null; },
  });
  setupsUndo = note;
}

function syncSetupUndo() {
  if (setupsUndo && setupsUndo.after !== setupComposerState()) {
    const note = setupsUndo;
    setupsUndo = null;
    if (promptUndo === note.entry) dropPromptUndo();
  }
}

function undoSetup(note) {
  // undoPromptWrite drops the visible note (and calls onDrop) before invoking
  // its callback. The captured state and one-use flag protect this callback.
  if (note.undone) return;
  if (note.after !== setupComposerState()) { syncSetupUndo(); return; }
  note.undone = true;
  setupsUndo = null;
  dropPromptUndo();
  const { state, values } = note.before;
  rewriteBox(els.negative, state.negative);
  rewriteBox(els.prompt, state.prompt);
  for (const key of PANEL_FIELDS) {
    if (!['prompt', 'negative', 'stepCache'].includes(key)) els[key].value = state[key];
  }
  pendingLora = state.lora && !els.lora.value ? state.lora : '';
  setReferences(state.reference);
  samplingByModel = { ...state.samplingByModel };
  Object.assign(owned, note.before.owned);
  for (const key of Object.keys(owned)) els[key].value = values[key];
  [pickedAspect, pickedTier] = note.before.size;
  els.gridSection.open = state.gridOpen;
  els.gridResume.checked = false;
  els.gridSeeds.disabled = !!els.gridSeedList.value.trim();
  els.thenRefine.checked = note.before.then.checked;
  trScale = note.before.then.scale;
  els.thenRefineStrength.value = note.before.then.strength;
  els.thenRefineSteps.value = note.before.then.steps;
  syncSetupComposer();
  const translation = state.translation;
  if (translation && els.prompt.value === translation.source) showPromptTranslation(translation.source, translation.text);
  persistPanel();
  showPromptNote('Setup restore undone.', { kind: 'info', at: 'box', ms: RESET_UNDONE_MS });
}

function closeSetups() {
  if (setupsPending === 'write') { setupSay('Wait for the setup to finish saving.'); return; }
  els.setupsDialog.close();
}

function appendSetupsMenuItem() {
  const item = document.createElement('li');
  item.id = 'setups-menu-item';
  item.className = 'setups-menu-item';
  item.setAttribute('role', 'menuitem');
  item.setAttribute('aria-haspopup', 'dialog');
  item.setAttribute('aria-controls', 'setups-dialog');
  item.tabIndex = -1;
  item.textContent = 'Saved setups…';
  const open = () => { showExamples(false); void openSetups(); };
  item.onclick = open;
  item.onkeydown = (event) => {
    if (keyboardEventHeld(event) || keyboardCommandModifier(event) || event.shiftKey) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (!event.repeat) open();
    }
  };
  els.examplesMenu.append(item);
}

function initSetups() {
  els.setupsClose.onclick = closeSetups;
  els.setupsDialog.addEventListener('cancel', (event) => {
    if (setupsPending === 'write') { event.preventDefault(); setupSay('Wait for the setup to finish saving.'); }
  });
  els.setupsDialog.addEventListener('close', () => {
    if (els.setupsDialog.open) return;
    ++setupsTicket;
    setupsPending = null;
    const origin = setupsOrigin;
    setupsOrigin = null;
    if (origin?.isConnected && isShown(origin)) origin.focus({ preventScroll: true });
  });
  els.setupsList.onchange = () => selectSetup(els.setupsList.value);
  els.setupsAll.onchange = renderSetups;
  els.setupsName.oninput = () => { setupsDirty = true; syncSetups(); };
  els.setupsNew.onclick = () => newSetup();
  els.setupsReload.onclick = () => { void reloadSetups(); };
  els.setupsSave.onclick = () => { void saveSetup('new'); };
  els.setupsUpdate.onclick = () => { void saveSetup('update'); };
  els.setupsRename.onclick = () => { void saveSetup('rename'); };
  els.setupsDelete.onclick = () => { void deleteSetup(); };
  els.setupsRestore.onclick = restoreSetup;
  els.setupsModels.onclick = () => {
    if (setupsPending) { setupSay('Wait for the saved setup request.'); return; }
    setupsOrigin = null;
    closeSetups();
    showModelPicker(true);
  };
  const changed = () => { ++setupsComposerEpoch; syncSetupUndo(); syncSetups(); };
  document.querySelector('.panel').addEventListener('input', changed);
  document.querySelector('.panel').addEventListener('change', changed);
  els.thenRefineDialog.addEventListener('input', changed);
  els.thenRefineDialog.addEventListener('change', changed);
  els.thenRefineScales.addEventListener('click', changed);
  for (const field of [els.modelCompilation, els.modelAttention]) {
    field.addEventListener('input', changed);
    field.addEventListener('change', changed);
  }
}
