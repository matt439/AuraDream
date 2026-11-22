// Pointer and keyboard editing for one region in the open Detailer review.
'use strict';

let detailerEdit = null;
let detailerDrag = null;

function detailerDrawHold() {
  if (detailerNeedsReopen) return 'The checkpoint changed. Reopen the dialog to draw a region.';
  if (detailerWorking || detailerSending || submitting) return 'Wait for the current request to finish.';
  if (serverOffline) return OFFLINE_MESSAGE;
  if (!detailerStatus) return 'Read the detector status first.';
  if (!detailerStatus.supported) return detailerStatus.reason || 'Detail passes are unavailable on this checkpoint.';
  if (!detailerPreview && (busy || elsewhereBusy)) return 'Wait for image jobs to finish before starting a region review.';
  return '';
}

function startDetailerEdit(kind, region, trigger, keyboard) {
  const held = detailerDrawHold() || (detailerEdit ? 'Apply or cancel the current box first.' : '');
  if (held) { detailerFailed(held); return; }
  detailerEdit = { kind, region, trigger };
  els.detailerBounds.value = region?.box.join(', ') ?? '';
  drawDetailerDraft();
  detailerFailed('');
  syncDetailerDialog();
  if (keyboard) els.detailerBounds.focus({ preventScroll: true });
}

function cancelDetailerEdit(returnFocus = false) {
  if (returnFocus && detailerWorking) { detailerFailed('Wait for the region review to finish.'); return; }
  const trigger = detailerEdit?.trigger;
  detailerDrag = null;
  detailerEdit = null;
  drawDetailerDraft();
  if (!returnFocus) return;
  detailerFailed('');
  syncDetailerDialog();
  if (trigger?.isConnected) trigger.focus({ preventScroll: true });
}

function detailerPixelBox() {
  const bounds = els.detailerBounds.value.trim();
  if (!/^\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+$/.test(bounds)) return null;
  const [left, top, right, bottom] = bounds.split(',').map(Number);
  const { width, height } = detailerPreview ?? detailerSource;
  return left < right && top < bottom && right <= width && bottom <= height ? [left, top, right, bottom] : null;
}

function drawDetailerDraft() {
  const box = detailerEdit ? detailerPixelBox() : null;
  els.detailerDraftBox.hidden = !box;
  if (box) placeDetailerBox(els.detailerDraftBox, box, detailerPreview ?? detailerSource);
}

function detailerStagePoint(event) {
  const frame = els.detailerStage.getBoundingClientRect();
  const { width, height } = detailerPreview ?? detailerSource;
  return [
    Math.min(width, Math.max(0, (event.clientX - frame.left) / frame.width * width)),
    Math.min(height, Math.max(0, (event.clientY - frame.top) / frame.height * height)),
  ];
}

function startDetailerDrag(event) {
  if (!detailerEdit || detailerDrag || detailerWorking || event.button !== 0) return;
  event.preventDefault();
  els.detailerStage.setPointerCapture(event.pointerId);
  detailerDrag = { pointer: event.pointerId, start: detailerStagePoint(event), previous: els.detailerBounds.value };
}

function moveDetailerDrag(event) {
  if (detailerDrag?.pointer !== event.pointerId) return;
  const [x, y] = detailerStagePoint(event);
  const [a, b] = detailerDrag.start;
  if (x === a || y === b) return;
  els.detailerBounds.value = [
    Math.floor(Math.min(a, x)), Math.floor(Math.min(b, y)),
    Math.ceil(Math.max(a, x)), Math.ceil(Math.max(b, y)),
  ].join(', ');
  drawDetailerDraft();
}

function finishDetailerDrag(event) {
  if (detailerDrag?.pointer !== event.pointerId) return;
  moveDetailerDrag(event);
  detailerDrag = null;
}

function cancelDetailerDrag(event) {
  if (detailerDrag?.pointer !== event.pointerId) return;
  els.detailerBounds.value = detailerDrag.previous;
  detailerDrag = null;
  drawDetailerDraft();
}

function saveDetailerBox() {
  if (!detailerEdit) return;
  const box = detailerPixelBox();
  if (!box) {
    detailerFailed('Drag a box on the image or enter four pixel bounds inside the image.');
    els.detailerBounds.focus({ preventScroll: true });
    return;
  }
  const { kind, region, trigger } = detailerEdit;
  void changeDetailerReview({ box, ...(region ? { region_id: region.id } : { kind }) }, trigger);
}

function pressDetailerRegionEdit(event) {
  const target = event.target.closest('button[data-edit-region]');
  if (!target) return;
  const region = detailerPreview?.regions.find((item) => item.id === target.dataset.editRegion);
  if (!region) return;
  if (target.dataset.editAction === 'redraw') startDetailerEdit(region.kind, region, target, event.detail === 0);
  else if (detailerEdit) detailerFailed('Apply or cancel the current box first.');
  else void changeDetailerReview({ region_id: region.id, box: null }, target);
}

async function changeDetailerReview(change, trigger) {
  const held = detailerDrawHold();
  if (held) { detailerFailed(held); return; }
  const source = detailerSource;
  const epoch = detailerEpoch;
  const revision = modelRevision;
  const selectedBefore = new Set(detailerSelected);
  const current = () => epoch === detailerEpoch && els.detailerDialog.open;
  detailerWorking = true;
  detailerFailed('');
  syncDetailerDialog();
  try {
    const options = (body) => ({
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const preview = detailerPreview ?? await api('/api/detailer/preview', options({
      source: source.id, model_revision: revision,
    }));
    if (!current()) return;
    const revised = await api('/api/detailer/review', options({ receipt: preview.receipt, ...change }));
    if (!current()) return;
    const added = revised.regions.find((region) => !preview.regions.some((old) => old.id === region.id));
    detailerSelected = new Set(revised.regions.filter((region) =>
      selectedBefore.has(region.id) || region.id === added?.id).map((region) => region.id));
    detailerPreview = revised;
    detailerRevision = revision;
    cancelDetailerEdit();
    renderDetailerRegions();
    const changed = [...els.detailerRegions.querySelectorAll('button[data-edit-region]')]
      .find((button) => button.dataset.editRegion === change.region_id);
    const focus = changed ?? (trigger.isConnected ? trigger : els.detailerDrawFace);
    focus.focus({ preventScroll: true });
  } catch (err) {
    if (!current()) return;
    if (imageGone(err) || detailerReceiptGone(err)) {
      cancelDetailerEdit();
      detailerPreview = null;
      detailerSelected.clear();
      renderDetailerRegions();
    }
    detailerFailed(err.message);
  } finally {
    if (current()) { detailerWorking = false; syncDetailerDialog(); }
  }
}
