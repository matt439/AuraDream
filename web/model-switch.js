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
let modelInfoKey = null;
let modelStatusEpoch = 0;
let modelSwitchError = null;
let modelPickerMessage = null;
let modelPickerDirty = false;
let modelRuntimeSettings = null;

function modelChangeHold() {
  return modelSwitchSending || modelSwitching ? 'Wait for the model settings change to finish' : '';
}

function modelPickerSettings() {
  return {
    model_id: els.modelSelect.value,
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
  return `${selected.quantization}|${selected.quantization === 'nunchaku' ? selected.format : ''}`;
}

function modelSwitchHold({ load = true } = {}) {
  return serverOffline ? 'The server is not answering'
    : modelChangeHold()
    || (submitting ? 'Wait for the job submission to finish' : '')
    || (busy || queuedIds.length ? 'A job is queued or running; switch once the image queue finishes' : '')
    || (elsewhereBusy ? 'A job started in another tab or window is running; switch once it ends' : '')
    || (expandingPrompt ? 'Wait for prompt expansion to finish' : '')
    || (critiquing ? 'Wait for the critic to finish' : '')
    || (cardHolder ? 'Wait for the current GPU work to finish' : '')
    || (modelState === 'loading' ? 'The weights are still loading' : '')
    || (!modelCatalogueLoaded || modelCataloguePending ? 'The checkpoint list is still loading' : '')
    || (!els.modelSelect.value ? 'Choose a checkpoint compatible with these settings' : '')
    || (load && !modelLoaderChanged() && modelState === 'ready' ? 'These model settings are already loaded; use Apply & save to remember them' : '');
}

function syncModelSwitch() {
  const why = modelSwitchHold();
  const saveWhy = modelSwitchHold({ load: false });
  els.modelSwitch.setAttribute('aria-disabled', String(!!why));
  els.modelSwitch.title = why || 'Remember these settings and load the selected checkpoint';
  els.modelSave.setAttribute('aria-disabled', String(!!saveWhy));
  els.modelSave.title = saveWhy || 'Remember these settings without starting a new load';
  els.modelFormatField.hidden = els.modelQuantization.value !== 'nunchaku';
  for (const select of [els.modelSelect, els.modelQuantization, els.modelOffload, els.modelFormat]) {
    select.title = select.selectedOptions[0]?.textContent ?? '';
  }
  if (!els.modelPicker.hidden) {
    const note = modelSwitchError || saveWhy || modelPickerMessage || 'The first load may download weights and take several minutes.';
    if (els.modelPickerNote.textContent !== note) els.modelPickerNote.textContent = note;
  }
}

function syncModelInfo() {
  const info = modelCatalogueDetails[els.modelSelect.value];
  const key = `${els.modelSelect.value}|${JSON.stringify(info)}`;
  if (key === modelInfoKey) return;
  modelInfoKey = key;
  els.modelInfo.hidden = !info;
  const name = els.modelSelect.selectedOptions[0]?.textContent || 'this model';
  els.modelInfoSummary.textContent = `About ${name}`;
  els.modelInfoBody.replaceChildren();
  if (!info) return;
  for (const [key, label] of [['description', ''], ['access', 'Access'], ['sampling', 'Starting settings'], ['prompt', 'Prompting'], ['size', 'Size']]) {
    if (!info[key]) continue;
    const p = document.createElement('p');
    if (label) {
      const strong = document.createElement('strong');
      strong.textContent = `${label}: `;
      p.append(strong);
    }
    p.append(document.createTextNode(info[key]));
    els.modelInfoBody.append(p);
  }
  if (info.url) {
    const link = document.createElement('a');
    link.href = info.url;
    link.textContent = 'Model page ↗';
    link.target = '_blank';
    link.rel = 'noopener';
    els.modelInfoBody.append(link);
  }
}

function modelPickerEdited({ catalogue = false } = {}) {
  modelPickerDirty = true;
  modelSwitchError = null;
  modelPickerMessage = null;
  syncModelSwitch();
  if (catalogue) void refreshModelCatalogue();
  else syncModelInfo();
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
  if (!modelSwitchSending && previous && previous.load_on_startup !== modelRuntimeSettings.load_on_startup) {
    modelPickerMessage = null;
  }
  if (reset) {
    modelPickerDirty = false;
    modelPickerMessage = null;
  }
  if (modelCatalogueLoaded && !modelPickerDirty) {
    const selected = modelRuntimeSettings;
    els.modelSelect.value = selected.model_id;
    els.modelQuantization.value = selected.quantization;
    els.modelOffload.value = selected.offload;
    els.modelFormat.value = selected.format;
    els.modelLoadOnStartup.checked = selected.load_on_startup;
    if (!els.modelSelect.value || modelCatalogueKind !== modelCatalogueKey(selected)) {
      void refreshModelCatalogue({ current: true });
    } else syncModelInfo();
  }
}

function showModelPicker(on, { focus = true } = {}) {
  els.modelPicker.hidden = !on;
  els.models.setAttribute('aria-expanded', String(on));
  if (on) {
    syncModelSwitch();
    void refreshModelCatalogue();
    if (focus) els.modelSelect.focus({ preventScroll: true });
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
  try {
    const body = await api(`/api/models${query.size ? `?${query}` : ''}`);
    if (epoch !== modelCatalogueEpoch) return;
    const wanted = { ...(selected ?? body) };
    if (modelPickerDirty && !current) {
      // Keep edits made while this catalogue request was in flight, including
      // the startup checkbox, which does not need a new catalogue of its own.
      for (const [key, value] of Object.entries(modelPickerSettings())) {
        if (value !== '' && value !== undefined) wanted[key] = value;
      }
    }
    els.modelSelect.replaceChildren();
    for (const entry of body.models ?? []) {
      const option = document.createElement('option');
      option.value = entry.model_id;
      option.textContent = entry.label;
      els.modelSelect.append(option);
    }
    // Only the server decides which checkpoint/loader combinations are valid.
    const model = wanted.model_id ?? loadedModel;
    if ([...els.modelSelect.options].some((o) => o.value === model)) els.modelSelect.value = model;
    else {
      // A different weight format can expose a different set of model ids.
      // Make the fresh choice explicit instead of silently selecting its head.
      els.modelSelect.prepend(new Option('Choose a checkpoint…', ''));
      els.modelSelect.value = '';
    }
    for (const [control, key, choices] of [
      [els.modelQuantization, 'quantization', 'quantizations'],
      [els.modelOffload, 'offload', 'offloads'],
      [els.modelFormat, 'format', 'formats'],
    ]) {
      control.replaceChildren(...(body[choices] ?? []).map((entry) => new Option(entry.label, entry.value)));
      const value = wanted[key] ?? body[key];
      if ([...control.options].some((o) => o.value === value)) control.value = value;
    }
    els.modelLoadOnStartup.checked = wanted.load_on_startup ?? body.load_on_startup ?? false;
    modelCatalogueKind = modelCatalogueKey(modelPickerSettings());
    modelCatalogueDetails = body.details ?? {};
    modelCatalogueLoaded = true;
    modelSwitchError = null;
    syncModelInfo();
  } catch (err) {
    if (epoch !== modelCatalogueEpoch) return;
    modelSwitchError = err.message;
    modelCatalogueLoaded = false;
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
    modelPickerMessage = `${why}.`;
    els.modelPickerNote.textContent = modelPickerMessage;
    if (serverOffline) void refreshStatus();
    return;
  }
  const selected = modelPickerSettings();
  modelSwitchError = null;
  modelPickerMessage = null;
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
    modelPickerDirty = false;
    receiveStatus(status, Date.now());
    modelPickerMessage = `Settings saved. ${status.load_on_startup ? 'They will load automatically on startup.' : 'Next startup will wait for you to load or generate.'}`;
  } catch (err) {
    modelSwitchError = err.message;
    showError(err.message);
  } finally {
    modelSwitchSending = false;
    syncPromptExpansion();
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
