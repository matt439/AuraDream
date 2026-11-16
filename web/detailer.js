// Face and hand region review. The server detects, orders and freezes the
// regions; the page chooses which of those regions to queue, without writing
// the composer. Declarations only; app.js wires the controls.
'use strict';

let detailerSource = null;
let detailerStatus = null;
let detailerPreview = null;
let detailerSelected = new Set();
let detailerWorking = false;
let detailerSending = false;
let detailerRan = false;
let detailerReturn = null;
let detailerOpener = null;
let detailerEpoch = 0;
let detailerRevision = null;
let detailerLora = null;
let detailerProblem = '';
let detailerDetectorSelection = '';
let detailerNeedsReopen = false;

function syncDetailerTool(button, rec) {
  const binned = !!rec?.binned_at;
  button.setAttribute('aria-disabled', String(binned));
  button.title = binned ? 'In the recycle bin: restore it to detail it'
    : 'Find face or hand regions, review them, then queue an inpaint of each selected region';
}

function detailerFromCard(rec) {
  if (!rec || refuseBinned(rec, 'detail')) return;
  void openDetailer(rec);
}

async function openDetailer(rec) {
  if (!rec || refuseBinned(rec, 'detail')) return false;
  detailerOpener = document.activeElement;
  detailerReturn = lightboxPlace();
  if (els.lightbox.open) closeLightbox();
  detailerEpoch++;
  detailerSource = rec;
  detailerStatus = null;
  detailerPreview = null;
  detailerSelected = new Set();
  detailerWorking = false;
  detailerSending = false;
  detailerRan = false;
  detailerProblem = '';
  detailerNeedsReopen = false;
  detailerRevision = modelRevision;
  const foreign = modelsDiffer(rec.model_id);
  detailerLora = !foreign && loraSupported && rec.lora
    && [...els.lora.options].some((option) => option.value === rec.lora)
    ? { name: rec.lora, scale: rec.lora_scale ?? 1 } : null;
  els.detailerTitle.textContent = `Detail faces/hands in #${rec.id}`;
  els.detailerSourceLine.textContent = `#${rec.id} · ${rec.width} × ${rec.height} · ${checkpointName(loadedModel)}`;
  els.detailerImage.src = rec.url ?? rec.thumb_url;
  els.detailerImage.alt = `#${rec.id}, the source for the detail passes`;
  els.detailerStage.style.setProperty('--ip-ratio', String(rec.width / rec.height));
  replaceBoxText(els.detailerPrompt, rec.prompt ?? '');
  replaceBoxText(els.detailerNegative, rec.negative_prompt ?? '');
  els.detailerSteps.value = String(foreign ? (lastDefaults?.steps ?? 28) : rec.steps);
  els.detailerGuidance.value = String(foreign ? (lastDefaults?.guidance_scale ?? 5) : rec.guidance_scale);
  els.detailerStrength.value = String(inpaintStartStrength());
  els.detailerSeed.value = String(rec.seed);
  els.detailerRecipe.open = false;
  els.detailerCarried.textContent = detailerLora
    ? `LoRA ${detailerLora.name} at ${detailerLora.scale}, as the source was made.`
    : rec.lora ? `The source's LoRA ${rec.lora} is not applied: it is not available on this checkpoint.` : '';
  els.detailerDetectors.replaceChildren();
  renderDetailerRegions();
  detailerFailed('');
  syncDetailerDialog();
  els.detailerDialog.showModal();
  els.detailerDialog.scrollTop = 0;
  els.detailerDetect.focus({ preventScroll: true });
  await refreshDetailerStatus();
  return true;
}

async function refreshDetailerStatus() {
  if (!detailerSource) return;
  if (detailerWorking || detailerSending) { detailerFailed('Wait for the current request to finish.'); return; }
  const epoch = detailerEpoch;
  detailerWorking = true;
  syncDetailerDialog();
  try {
    const status = await api('/api/detailer/status');
    if (epoch !== detailerEpoch) return;
    detailerStatus = status;
    const selected = els.detailerDetectors.value;
    const options = (status.detectors ?? []).map((detector) => {
      const option = document.createElement('option');
      option.value = detector.id;
      option.textContent = detector.label;
      return option;
    });
    els.detailerDetectors.replaceChildren(...options);
    if (options.some((option) => option.value === selected)) els.detailerDetectors.value = selected;
    detailerDetectorSelection = els.detailerDetectors.value;
  } catch (err) {
    if (epoch === detailerEpoch) detailerFailed(err.message);
  } finally {
    if (epoch === detailerEpoch) { detailerWorking = false; syncDetailerDialog(); }
  }
}

function chosenDetailerDetector() {
  return detailerStatus?.detectors?.find((detector) => detector.id === els.detailerDetectors.value) ?? null;
}

function detailerCanPrepare(detector) {
  return !!detector && (detector.prepare_available === true
    || detector.prepare_available === undefined && (detector.available || detector.download_bytes > 0));
}

function detailerChangedDetector() {
  if (detailerWorking || detailerSending) {
    els.detailerDetectors.value = detailerDetectorSelection;
    detailerFailed('Wait for the current request to finish.');
    return;
  }
  detailerDetectorSelection = els.detailerDetectors.value;
  detailerPreview = null;
  detailerSelected.clear();
  renderDetailerRegions();
  detailerFailed('');
  syncDetailerDialog();
}

async function prepareDetailer() {
  if (!detailerSource) return;
  if (detailerWorking || detailerSending) { detailerFailed('Wait for the current request to finish.'); return; }
  const detector = chosenDetailerDetector();
  if (!detailerCanPrepare(detector)) {
    detailerFailed(detector?.reason || 'This detector cannot be prepared here.');
    return;
  }
  const epoch = detailerEpoch;
  detailerWorking = true;
  detailerFailed('');
  syncDetailerDialog();
  try {
    await api('/api/detailer/prepare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detector: detector.id }),
    });
  } catch (err) {
    if (epoch === detailerEpoch) detailerFailed(err.message);
  } finally {
    if (epoch === detailerEpoch) {
      detailerWorking = false;
      await refreshDetailerStatus();
    }
  }
}

async function detectDetailer() {
  if (!detailerSource) return;
  const reason = detailerDetectHold();
  if (reason) { detailerFailed(reason); return; }
  const epoch = detailerEpoch;
  const revision = modelRevision;
  detailerWorking = true;
  detailerPreview = null;
  detailerSelected.clear();
  detailerFailed('');
  renderDetailerRegions();
  syncDetailerDialog();
  try {
    const preview = await api('/api/detailer/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: detailerSource.id, detector: els.detailerDetectors.value,
        model_revision: revision }),
    });
    if (epoch !== detailerEpoch) return;
    if (revision !== modelRevision) throw new Error('The checkpoint changed. Detect regions again on the selected checkpoint.');
    detailerRevision = revision;
    detailerPreview = preview;
    const limit = detailerStatus?.max_regions;
    detailerSelected = new Set(typeof limit === 'number' && preview.regions.length > limit
      ? [] : preview.regions.map((region) => region.id));
    renderDetailerRegions();
  } catch (err) {
    if (epoch === detailerEpoch) detailerFailed(err.message);
  } finally {
    if (epoch === detailerEpoch) { detailerWorking = false; syncDetailerDialog(); }
  }
}

function detailerDetectHold() {
  if (detailerNeedsReopen) return 'The checkpoint changed. Close this dialog and reopen it on the current checkpoint.';
  if (detailerWorking) return 'Wait for the detector request to finish.';
  if (detailerSending || submitting) return `${PASS_IN_FLIGHT}.`;
  if (serverOffline) return OFFLINE_MESSAGE;
  if (busy || elsewhereBusy) return 'Wait for image jobs to finish before detecting regions.';
  if (!detailerStatus) return 'Read the detector status first.';
  if (!detailerStatus.supported) return detailerStatus.reason || 'Detail passes are unavailable on this checkpoint.';
  const detector = chosenDetailerDetector();
  if (!detector) return 'Choose a detector first.';
  if (!detector.available) return detector.reason || 'Download this detector before detecting regions.';
  return '';
}

function detailerNumberProblem() {
  const checks = [
    [els.detailerSteps, 'Steps', 1, 100, true],
    [els.detailerGuidance, 'CFG', 0, 20, false],
    [els.detailerStrength, 'Strength', 0.01, 1, false],
    [els.detailerSeed, 'Seed', 0, Number.MAX_SAFE_INTEGER, true],
  ];
  for (const [el, name, min, max, integer] of checks) {
    if (el === els.detailerSeed && el.value.trim() === '' && !el.validity.badInput) continue;
    const value = Number(el.value);
    if (el.validity.badInput || !el.value.trim() || !Number.isFinite(value) || value < min || value > max || integer && !Number.isInteger(value)) {
      return { el, text: `${name} needs ${integer ? 'a whole number' : 'a number'} from ${min} to ${max}.` };
    }
  }
  return null;
}

function detailerRunHold() {
  if (detailerNeedsReopen) return 'The checkpoint changed. Close this dialog and reopen it on the current checkpoint.';
  if (detailerSending || submitting) return `${PASS_IN_FLIGHT}.`;
  if (detailerWorking) return 'Wait for the detector request to finish.';
  if (serverOffline) return OFFLINE_MESSAGE;
  const held = submitHold();
  if (held) return held;
  if (!detailerPreview) return 'Detect regions, then review them before queuing.';
  if (detailerRevision !== modelRevision) return 'The checkpoint changed. Detect regions again.';
  if (!detailerSelected.size) return detailerPreview.regions.length
    ? 'Select at least one region to queue.' : 'No regions were detected. Nothing will be queued.';
  if (typeof detailerStatus?.max_regions === 'number' && detailerSelected.size > detailerStatus.max_regions) {
    return `Choose at most ${detailerStatus.max_regions} regions for one detail operation.`;
  }
  if (!els.detailerPrompt.value.trim()) return 'Write a prompt for the inpaint passes.';
  return detailerNumberProblem()?.text || '';
}

function syncDetailerDialog() {
  if (!detailerSource) return;
  const detector = chosenDetailerDetector();
  const detectReason = detailerDetectHold();
  const canPrepare = detailerCanPrepare(detector);
  const prepareReason = detailerWorking ? 'Wait for the detector request to finish.'
    : detailerSending ? `${PASS_IN_FLIGHT}.`
      : !canPrepare ? detector?.reason || 'This detector cannot be prepared here.' : '';
  const preparingReady = !!detector?.available;
  const prepareTitle = preparingReady
    ? `Verify cached detector assets; changed assets may be redownloaded (up to ${(detector.asset_bytes / 1e6).toFixed(1)} MB)`
    : `Download the pinned detector assets (${((detector?.download_bytes ?? 0) / 1e6).toFixed(1)} MB)`;
  els.detailerPrepare.textContent = preparingReady ? 'Verify/repair detector' : 'Download detector';
  for (const [button, reason] of [[els.detailerDetect, detectReason], [els.detailerPrepare, prepareReason]]) {
    button.setAttribute('aria-disabled', String(!!reason));
    button.title = reason || (button === els.detailerDetect ? 'Detect regions on the CPU; no image is generated'
      : prepareTitle);
  }
  els.detailerDetectors.setAttribute('aria-disabled', String(detailerWorking || detailerSending));
  els.detailerDetectors.title = detailerWorking || detailerSending ? 'Wait for the current request to finish.' : 'Choose what to detect';
  const detectorNote = [detailerStatus?.reason,
    detector?.reason || (detector ? `${detector.label} · ${detector.available ? 'ready on the CPU' : 'download required'}` : 'Reading detector status…'),
    detector?.note].filter(Boolean).join(' ');
  els.detailerDetectorNote.textContent = detectorNote;
  els.detailerDetectorNote.title = detectorNote;
  els.detailerOrderNote.textContent = (typeof detailerStatus?.feather === 'number'
    ? `Masks feather ${detailerStatus.feather}px within each marked area. `
    : 'Masks have feathered edges within each marked area. ')
    + 'Passes run in list order; later passes see earlier edits. Each pass is saved.';
  els.detailerBusy.textContent = detailerWorking ? 'Detector request in progress…' : '';
  const refreshReason = detailerWorking ? 'Wait for the detector request to finish.'
    : detailerSending ? `${PASS_IN_FLIGHT}.` : '';
  els.detailerRefresh.setAttribute('aria-disabled', String(!!refreshReason));
  els.detailerRefresh.title = refreshReason || 'Read detector availability without downloading or detecting';
  const runReason = detailerRunHold();
  els.detailerRun.setAttribute('aria-disabled', String(!!runReason));
  els.detailerRun.title = runReason || `Queue ${detailerSelected.size} ordered inpaint passes (Ctrl+Enter)`;
  els.detailerRun.textContent = detailerSending ? 'Sending…' : 'Queue detail passes';
  for (const button of [els.detailerCancel, els.detailerClose]) {
    button.setAttribute('aria-disabled', String(detailerSending));
    button.title = detailerSending ? `${PASS_IN_FLIGHT}.` : 'Close without queuing detail passes';
  }
  const count = detailerPreview ? detailerSelected.size : 0;
  els.detailerSummary.textContent = detailerPreview
    ? detailerPreview.regions.length ? `${count} of ${detailerPreview.regions.length} regions selected · ${count} inpaint passes`
      + (detailerPreview.regions.length > detailerStatus?.max_regions ? ` · choose up to ${detailerStatus.max_regions}` : ' · time estimate unavailable')
      : 'No regions detected. Nothing will be queued.'
    : 'Detection does not decide whether a face or hand needs changing. Review each region.';
}

function renderDetailerRegions() {
  els.detailerRegions.replaceChildren();
  els.detailerBoxes.replaceChildren();
  const numbers = new Map((detailerPreview?.regions ?? []).map((region, index) => [region.id, index + 1]));
  for (const [index, region] of (detailerPreview?.regions ?? []).entries()) {
    const selected = detailerSelected.has(region.id);
    const name = `${index + 1}. ${region.kind === 'hand' ? 'Hand' : 'Face'}`;
    const row = document.createElement('li');
    const label = document.createElement('label');
    const tick = document.createElement('input');
    tick.type = 'checkbox'; tick.checked = selected; tick.dataset.region = String(region.id);
    const text = document.createElement('span');
    const shared = (region.overlaps ?? []).map((id) => numbers.get(id)).filter((number) => number !== undefined);
    const overlaps = shared.length ? ` · overlaps region${shared.length === 1 ? '' : 's'} ${shared.join(', ')}` : '';
    text.textContent = `${name}${overlaps}`;
    label.append(tick, text); row.append(label); els.detailerRegions.append(row);
    const box = document.createElement('button');
    box.type = 'button'; box.className = 'detailer-box'; box.dataset.region = String(region.id);
    box.textContent = String(index + 1); box.setAttribute('aria-label', `${name} region`);
    box.setAttribute('aria-pressed', String(selected));
    const [left, top, right, bottom] = region.mask_box ?? region.box;
    box.style.left = `${left / detailerPreview.width * 100}%`;
    box.style.top = `${top / detailerPreview.height * 100}%`;
    box.style.width = `${(right - left) / detailerPreview.width * 100}%`;
    box.style.height = `${(bottom - top) / detailerPreview.height * 100}%`;
    els.detailerBoxes.append(box);
  }
}

function chooseDetailerRegion(event) {
  const target = event.target.closest('[data-region]');
  if (!target) return;
  const region = detailerPreview?.regions.find((item) => String(item.id) === target.dataset.region);
  if (!region) return;
  if (detailerSending || detailerWorking) {
    if (target.matches('input')) target.checked = detailerSelected.has(region.id);
    detailerFailed('Wait for the current request to finish.');
    return;
  }
  if (detailerSelected.has(region.id)) detailerSelected.delete(region.id);
  else detailerSelected.add(region.id);
  // Update in place: the focused checkbox/button remains in the document.
  for (const item of [...els.detailerDialog.querySelectorAll('[data-region]')]
    .filter((node) => node.dataset.region === String(region.id))) {
    if (item.matches('input')) item.checked = detailerSelected.has(region.id);
    else item.setAttribute('aria-pressed', String(detailerSelected.has(region.id)));
  }
  detailerFailed('');
  syncDetailerDialog();
}

function detailerChanged() {
  for (const el of [els.detailerSeed, els.detailerSteps, els.detailerGuidance, els.detailerStrength]) el.removeAttribute('aria-invalid');
  detailerFailed('');
  syncDetailerDialog();
}

function detailerKeydown(event) {
  if (keyboardEventHeld(event) || keyboardCommandModifier(event)) return;
  if (event.key === 'Enter' && !event.shiftKey && event.target.matches?.('#detailer-dialog input[type="number"]')) {
    event.preventDefault();
    if (!event.repeat) void runDetailer();
  }
}

function detailerFailed(message) {
  detailerProblem = message || '';
  els.detailerError.textContent = detailerProblem;
  els.detailerError.hidden = !detailerProblem;
  els.detailerSummary.hidden = !!detailerProblem;
}

function detailerReceiptGone(err) {
  return err?.code === 'detailer_receipt' || err?.code === 'detailer_stale';
}

async function runDetailer() {
  if (!detailerSource || !els.detailerDialog.open) return;
  const held = detailerRunHold();
  if (held) {
    detailerFailed(held);
    const bad = detailerNumberProblem();
    if (bad && held === bad.text) { bad.el.setAttribute('aria-invalid', 'true'); bad.el.focus({ preventScroll: true }); }
    return;
  }
  const source = detailerSource;
  const recipe = {
    prompt: els.detailerPrompt.value.trim(), negative_prompt: els.detailerNegative.value,
    steps: +els.detailerSteps.value, guidance_scale: +els.detailerGuidance.value,
    strength: +els.detailerStrength.value, seed: els.detailerSeed.value.trim() === '' ? null : +els.detailerSeed.value,
    ...(detailerLora ? { lora: detailerLora.name, lora_scale: detailerLora.scale } : {}),
  };
  detailerSending = true;
  detailerFailed('');
  syncDetailerDialog();
  let refusal = null;
  const sent = startJob('/api/detailer', {
    receipt: detailerPreview.receipt,
    accepted: detailerPreview.regions.filter((region) => detailerSelected.has(region.id)).map((region) => region.id),
    recipe,
  }, {
    panel: false, slot: els.detailerError,
    placeholder: { src: source.url ?? source.thumb_url, width: source.width, height: source.height,
      label: `Detailing #${source.id}` },
    onRefused: (err) => { refusal = err; },
    onAccepted: () => {
      detailerSending = false;
      detailerRan = true;
      if (detailerSource === source && els.detailerDialog.open) els.detailerDialog.close();
    },
  });
  syncDetailerDialog();
  await sent;
  detailerSending = false;
  if (refusal) {
    if (imageGone(refusal) || detailerReceiptGone(refusal)) {
      detailerPreview = null;
      detailerSelected.clear();
      renderDetailerRegions();
    }
    detailerFailed(refusal.message);
    if (!els.detailerDialog.open) showError(refusal.message);
  }
  syncDetailerDialog();
}

function closeDetailer() {
  if (detailerSending) { detailerFailed(`${PASS_IN_FLIGHT}.`); return; }
  els.detailerDialog.close();
}
function detailerCancelEvent(event) { if (detailerSending) event.preventDefault(); }

function detailerClosed() {
  const back = detailerReturn;
  const opener = detailerOpener;
  const ran = detailerRan || detailerSending;
  const source = detailerSource;
  detailerEpoch++;
  detailerSource = null;
  detailerStatus = null;
  detailerPreview = null;
  detailerSelected.clear();
  detailerReturn = null;
  detailerOpener = null;
  if (!ran && back && !els.lightbox.open) { returnToLightbox(back); return; }
  if (!ran && opener?.isConnected && isShown(opener)) { opener.focus({ preventScroll: true }); return; }
  const card = source && [...document.querySelectorAll(`#gallery .card[data-id="${source.id}"] img`)].find(isShown);
  handJobFocus(card ?? els.generate);
}

function detailerSourcesMoved(ids) {
  if (!detailerSource || !new Set(ids).has(detailerSource.id)) return;
  detailerEpoch++;
  detailerWorking = false;
  detailerPreview = null;
  detailerSelected.clear();
  renderDetailerRegions();
  detailerFailed('The source changed. Restore it if needed, then detect regions again.');
  syncDetailerDialog();
}

function detailerCheckpointChanged() {
  if (!detailerSource) return;
  detailerEpoch++;
  detailerWorking = false;
  detailerNeedsReopen = true;
  detailerPreview = null;
  detailerSelected.clear();
  renderDetailerRegions();
  detailerFailed('The checkpoint changed. Close this dialog and reopen it to use the current checkpoint.');
  syncDetailerDialog();
}
