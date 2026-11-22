// ControlNet's draft and its server-bound guide proof.
'use strict';

let controlNetOptions = {};
let controlNetSpec = null;
let controlNetPreviewRequest = null;
let controlNetError = '';

const controlNetElement = (name) => document.getElementById(`controlnet-${name}`);
const controlNetSelected = () => Boolean(controlNetElement('mode')?.value);

function composerControlNet() {
  return controlNetSelected() ? controlNetSpec : null;
}

function controlNetDraft() {
  return {
    mode: controlNetElement('mode').value,
    source: controlNetElement('source').value,
    strength: controlNetElement('strength').value,
    proof: controlNetSpec,
  };
}

function controlNetReuseReason(spec) {
  if (!spec?.source) return 'The recorded ControlNet guide could not be resolved in this gallery.';
  const mode = controlNetOptions[spec.mode];
  if (!mode) return 'The selected checkpoint does not offer that ControlNet mode.';
  if (mode.reason) return mode.reason;
  return spec.weight_revision === mode.revision ? '' : 'The recorded ControlNet weight revision is unavailable.';
}

function controlNetSubmitHold() {
  if (!controlNetSelected()) return '';
  const mode = controlNetElement('mode').value;
  const option = controlNetOptions[mode];
  if (!option) return 'The selected checkpoint does not offer this ControlNet mode.';
  if (option.reason) return option.reason;
  if (controlNetError) return controlNetError;
  if (!controlNetSpec) return 'Preview a ControlNet guide before generating.';
  const unavailable = controlNetReuseReason(controlNetSpec);
  if (unavailable) return unavailable;
  const sizeMatches = +els.width.value === 1024 && +els.height.value === 1024;
  const stepsMatch = !els.steps.value || +els.steps.value === 28;
  const cfgMatches = !els.cfg.value || +els.cfg.value === 5;
  if (!sizeMatches || !stepsMatch || !cfgMatches || +els.batch.value !== 1) {
    return 'ControlNet needs 1024 × 1024, 28 steps, CFG 5 and batch 1.';
  }
  return els.lora.value || referenceIds.length ? 'Clear the LoRA and references before generating with ControlNet.' : '';
}

function controlNetPreviewHold() {
  if (controlNetPreviewRequest) return 'The ControlNet guide is being checked.';
  const mode = controlNetOptions[controlNetElement('mode').value];
  if (!mode) return 'Choose a supported ControlNet mode.';
  if (mode.reason) return mode.reason;
  const source = +controlNetElement('source').value;
  if (!Number.isSafeInteger(source) || source < 1) return 'Enter the guide’s gallery ID.';
  const strength = controlNetElement('strength');
  if (!strength.value || !strength.validity.valid) return 'Enter a strength from 0.1 to 1.';
  return '';
}

function syncControlNetControls() {
  const section = controlNetElement('section');
  if (!section) return;
  const active = controlNetSelected();
  section.hidden = !active && loadedModel !== 'cagliostrolab/animagine-xl-4.0';
  controlNetElement('fields').hidden = !active;
  const mode = controlNetElement('mode');
  const weights = controlNetOptions[mode.value];
  controlNetElement('summary').textContent = active ? mode.selectedOptions[0].textContent : 'off';
  for (const choice of mode.options) {
    choice.title = controlNetOptions[choice.value]?.reason ?? '';
  }
  const note = controlNetElement('note');
  const held = active ? controlNetSubmitHold() : '';
  let words = '';
  if (active) {
    const source = mode.value === 'pose' ? 'Import a prepared 1024 × 1024 OpenPose map and enter its gallery ID.'
      : 'Import a prepared 1024 × 1024 depth map with nearer areas brighter, then enter its gallery ID.';
    const model = weights ? ` ${weights.repo_id} · ${Math.round(weights.size / 1_000_000)} MB on first use.` : '';
    words = `${source}${model}${held ? ` ${held}` : ''}`;
  }
  if (note.textContent !== words) note.textContent = words;
  note.hidden = !words;
  const preview = controlNetElement('preview');
  const blocked = active ? controlNetPreviewHold() : '';
  preview.title = blocked;
  preview.setAttribute('aria-disabled', String(Boolean(blocked)));
  const image = controlNetElement('image');
  const url = active && controlNetSpec?.source ? `/api/images/${controlNetSpec.source}/thumb` : '';
  if (url && image.getAttribute('src') !== url) image.src = url;
  if (!url && image.hasAttribute('src')) image.removeAttribute('src');
  image.hidden = !url;
}

function syncControlNetStatus(options) {
  if (JSON.stringify(options) === JSON.stringify(controlNetOptions)) return;
  controlNetOptions = options;
  syncSubmitButtons();
  syncReuseButtons();
}

function restoreControlNet(spec, reveal = false) {
  controlNetPreviewRequest?.abort();
  controlNetPreviewRequest = null;
  controlNetSpec = spec?.kind === 'controlnet' ? spec : null;
  controlNetElement('mode').value = controlNetSpec?.mode ?? '';
  controlNetElement('source').value = controlNetSpec?.source ?? '';
  controlNetElement('strength').value = controlNetSpec?.strength ?? 0.8;
  controlNetError = '';
  if (reveal) controlNetElement('section').open = Boolean(controlNetSpec);
  syncControlNetControls();
}

function restoreControlNetDraft(draft, reveal) {
  if (!draft || draft.kind === 'controlnet') {
    restoreControlNet(draft, reveal);
    return;
  }
  const { mode, source, strength, proof } = draft;
  const matched = proof?.kind === 'controlnet' && proof.mode === mode
    && proof.source === (source === '' ? null : +source) && proof.strength === +strength;
  restoreControlNet(matched ? proof : null);
  controlNetElement('mode').value = mode ?? '';
  controlNetElement('source').value = source ?? '';
  controlNetElement('strength').value = strength ?? '0.8';
  if (reveal) controlNetElement('section').open = controlNetSelected();
  syncControlNetControls();
}

function controlNetChanged() {
  controlNetPreviewRequest?.abort();
  controlNetPreviewRequest = null;
  controlNetSpec = null;
  controlNetError = '';
  syncSubmitButtons();
  persistPanel();
  syncMoreOptions();
}

async function previewControlNet() {
  const held = controlNetPreviewHold();
  if (held) {
    sayPhoneHeld(controlNetElement('preview'), held);
    return;
  }
  const request = new AbortController();
  controlNetPreviewRequest = request;
  const body = {
    mode: controlNetElement('mode').value,
    source: +controlNetElement('source').value,
    strength: +controlNetElement('strength').value,
  };
  const revision = modelRevision;
  const server = serverStartedAt;
  controlNetSpec = null;
  controlNetError = '';
  syncSubmitButtons();
  try {
    const reply = await api('/api/controlnet/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: request.signal,
    });
    if (controlNetPreviewRequest === request && revision === modelRevision && server === serverStartedAt) {
      controlNetSpec = reply.spatial_conditioning;
      persistPanel();
    }
  } catch (err) {
    if (controlNetPreviewRequest === request) controlNetError = err.message;
  } finally {
    if (controlNetPreviewRequest === request) {
      controlNetPreviewRequest = null;
      syncSubmitButtons();
    }
  }
}

function setupControlNet() {
  controlNetElement('mode').addEventListener('change', controlNetChanged);
  for (const name of ['source', 'strength']) controlNetElement(name).addEventListener('input', controlNetChanged);
  controlNetElement('preview').addEventListener('click', previewControlNet);
}
