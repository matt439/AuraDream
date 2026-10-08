// Checkpoint selection and the state that belongs to one model selection.
// Declarations only; app.js wires the controls after every script has loaded.
'use strict';

let modelRevision = null;
let modelSwitchSending = false;
let modelSwitching = false;
let modelCataloguePending = false;
let modelCatalogueLoaded = false;
let modelStatusEpoch = 0;
let modelSwitchError = null;

function modelChangeHold() {
  return modelSwitchSending || modelSwitching ? 'Wait for the checkpoint switch to finish' : '';
}

function modelSwitchHold() {
  return serverOffline ? 'The server is not answering'
    : modelChangeHold()
    || (submitting ? 'Wait for the job submission to finish' : '')
    || (busy || queuedIds.length ? 'A job is queued or running; switch once the image queue finishes' : '')
    || (elsewhereBusy ? 'A job started in another tab or window is running; switch once it ends' : '')
    || (expandingPrompt ? 'Wait for prompt expansion to finish' : '')
    || (critiquing ? 'Wait for the critic to finish' : '')
    || (cardHolder ? 'Wait for the current GPU work to finish' : '')
    || (modelState === 'loading' ? 'The weights are still loading' : '')
    || (!modelCatalogueLoaded ? 'The checkpoint list is still loading' : '')
    || (!els.modelSelect.value || els.modelSelect.value === loadedModel ? 'Choose a different checkpoint' : '');
}

function syncModelSwitch() {
  const why = modelSwitchHold();
  els.modelSwitch.setAttribute('aria-disabled', String(!!why));
  els.modelSwitch.title = why || 'Free the current weights and load the selected checkpoint';
  if (!els.modelPicker.hidden) {
    els.modelPickerNote.textContent = modelSwitchError || why || 'The first load may download weights and take several minutes.';
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

async function refreshModelCatalogue() {
  if (modelCataloguePending) return;
  modelCataloguePending = true;
  try {
    const body = await api('/api/models');
    const selected = els.modelSelect.value || loadedModel;
    els.modelSelect.replaceChildren();
    for (const entry of body.models ?? []) {
      const option = document.createElement('option');
      option.value = entry.model_id;
      option.textContent = entry.label;
      els.modelSelect.append(option);
    }
    // An explicitly configured off-catalogue checkpoint can still be current.
    if (loadedModel && ![...els.modelSelect.options].some((o) => o.value === loadedModel)) {
      els.modelSelect.add(new Option(checkpointName(loadedModel), loadedModel));
    }
    els.modelSelect.value = [...els.modelSelect.options].some((o) => o.value === selected) ? selected : loadedModel;
    modelCatalogueLoaded = true;
    syncModelSwitch();
  } catch (err) {
    modelSwitchError = err.message;
    syncModelSwitch();
  } finally { modelCataloguePending = false; }
}

async function switchModel() {
  const why = modelSwitchHold();
  if (why) {
    els.modelPickerNote.textContent = `${why}.`;
    if (serverOffline) void refreshStatus();
    return;
  }
  const model = els.modelSelect.value;
  modelSwitchError = null;
  // Flush this checkpoint's edits while its identity still describes them.
  clearTimeout(panelTimer);
  writePanel();
  modelSwitchSending = true;
  ++modelStatusEpoch; // A status already in flight describes the old selection.
  syncPromptExpansion();
  try {
    const status = await api('/api/model/switch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: model }),
    });
    receiveStatus(status, Date.now());
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
