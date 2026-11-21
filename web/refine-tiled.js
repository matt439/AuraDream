// The fixed Tiled Refine method and its server-owned source eligibility.
// Declarations only; app.js wires the Refine dialog after these scripts.
'use strict';

let refineTiled = false;
let refineTiledReplay = null;
let refineTiledCheck = null;

function tiledRefineCapabilityReason() {
  return lastDefaults?.tiled_refine_reason
    || (lastDefaults?.tiled_refine ? '' : 'This installation does not offer Tiled Refine.');
}

function tiledRefineSourceReason() {
  const id = refineSource?.id;
  if (binnedIds.has(id)) return sourceGoneWords(refineSource, 'binned', 'refine');
  if (deletedIds.has(id)) return sourceGoneWords(refineSource, 'deleted', 'refine');
  return '';
}

function tiledRefineChoiceReason() {
  return tiledRefineCapabilityReason() || tiledRefineSourceReason()
    || (refineTiledCheck?.answer ? refineTiledCheck.answer.reason : 'Checking this source for Tiled Refine…');
}

// Ask once per source and checkpoint revision. The submit checks the source
// again, since other tabs or a restore can change it after the dialog opens.
function checkTiledRefineSource() {
  if (tiledRefineCapabilityReason() || tiledRefineSourceReason()) {
    refineTiledCheck = null;
    return;
  }
  if (refineTiledCheck?.revision === modelRevision && refineTiledCheck.startedAt === serverStartedAt) return;
  const source = refineSource;
  const current = { revision: modelRevision, startedAt: serverStartedAt, answer: null, failed: false };
  refineTiledCheck = current;
  const body = { source: source.id, tiled_refine: refineTiledReplay ?? undefined };
  const settle = (answer, error = null) => {
    if (refineTiledCheck !== current || refineSource !== source || current.revision !== modelRevision
        || current.startedAt !== serverStartedAt) return;
    current.failed = !!error;
    if (error) answer = { reason: refineSourceGone(error, source) };
    else if (answer.code) answer.reason = refineSourceGone({ code: answer.code, body: answer, message: answer.reason }, source);
    current.answer = answer;
    syncRefine();
    if (refineTiled) scheduleRefinePreview(0);
  };
  void api('/api/tiled-refine/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(settle, (err) => settle(null, err));
}

function syncRefineMethod() {
  checkTiledRefineSource();
  const reason = tiledRefineChoiceReason();
  const plan = lastDefaults?.tiled_refine;
  els.refineWhole.classList.toggle('active', !refineTiled);
  els.refineTiled.classList.toggle('active', refineTiled);
  els.refineWhole.setAttribute('aria-pressed', String(!refineTiled));
  els.refineTiled.setAttribute('aria-pressed', String(refineTiled));
  els.refineTiled.setAttribute('aria-disabled', String(!!reason));
  els.refineTiled.title = reason || 'Refine this source in overlapping tiles';
  els.refineTiledHint.hidden = !refineTiled;
  const hint = reason || (plan
    ? `${plan.source_width}×${plan.source_height} → ${plan.width}×${plan.height} · strength ${plan.strength} · ${plan.steps} steps · CFG ${plan.guidance_scale}. Overlaps may show seams, and fine detail can change.`
    : '');
  if (refineTiled && els.refineTiledHint.textContent !== hint) els.refineTiledHint.textContent = hint;
  els.refineTiledHint.classList.toggle('warn', !!reason);
  els.refineSizeField.hidden = refineTiled;
  els.refineStrengthField.hidden = refineTiled;
  els.refineSamplingFields.hidden = refineTiled;
  els.refineCarried.hidden = refineTiled;
  els.refineMoreNote.hidden = refineTiled;
  const moreLabel = refineTiled ? 'Prompt' : 'Prompt and sampling';
  if (els.refineMoreLabel.textContent !== moreLabel) els.refineMoreLabel.textContent = moreLabel;
}

function chooseRefineMethod(tiled) {
  if (refineSending) { refineFailed(PASS_IN_FLIGHT); return; }
  const reason = tiled ? tiledRefineChoiceReason() : '';
  if (reason) {
    refineFailed(reason);
    if (refineTiledCheck?.failed) { refineTiledCheck = null; syncRefine(); }
    return;
  }
  if (refineTiled === tiled) return;
  refineTiled = tiled;
  refineEpoch++;
  refineProblem = null;
  refinePreviewed = null;
  els.refinePreview.textContent = '';
  refineChanged();
  refineCheckChanged();
}
