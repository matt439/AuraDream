// AuraDream's region editor and the composer's spatial conditioning draft.
'use strict';

const REGIONAL_METHOD = 'sdxl-attention-v1';
const REGIONAL_MAX = 3;
const REGIONAL_EDGES = [['left', 'Left'], ['top', 'Top'], ['right', 'Right'], ['bottom', 'Bottom']];

function regionalPromptsCapabilityReason() {
  return lastDefaults?.regional_prompts === true ? '' : lastDefaults?.regional_prompts_reason
    || 'Region prompts are unavailable for the current checkpoint and loader.';
}

function regionalPromptsRows() {
  return [...$('regional-prompts-rows').children];
}

function regionalPromptsDraft() {
  return regionalPromptsRows().map((row) => Object.fromEntries(
    [...row.querySelectorAll('input,textarea')].map((input) => [input.name, input.value]),
  ));
}

function regionalPromptsBounds(draft) {
  const bounds = {};
  for (const [key] of REGIONAL_EDGES) {
    const raw = draft[key];
    if (raw === '' || raw == null) return null;
    const value = Number(raw);
    if (!Number.isInteger(value) || value < 0 || value > 100) return null;
    bounds[key] = value;
  }
  const { left, top, right, bottom } = bounds;
  return right - left >= 5 && bottom - top >= 5 ? bounds : null;
}

function regionalPromptsReason() {
  const rows = regionalPromptsDraft();
  if (!rows.length) return '';
  const unavailable = regionalPromptsCapabilityReason();
  if (unavailable) return unavailable;
  if (+els.width.value !== 1024 || +els.height.value !== 1024) return 'Region prompts need a 1024 × 1024 canvas.';
  const steps = els.steps.value.trim() === '' ? lastDefaults.steps : +els.steps.value;
  const cfg = els.cfg.value.trim() === '' ? lastDefaults.guidance_scale : +els.cfg.value;
  if (steps !== 28 || cfg !== 5) return 'Region prompts need 28 steps and CFG 5.';
  if (+els.batch.value !== 1 || +els.runs.value !== 1) return 'Region prompts need Batch 1 and Repeat 1.';
  if ((!els.lora.disabled && els.lora.value) || (referenceSupported && referenceIds.length)) {
    return 'Turn off LoRA and reference images to use region prompts.';
  }
  for (const [index, row] of rows.entries()) {
    const label = `Region ${index + 1}`;
    const length = [...row.prompt.trim()].length;
    if (!length || length > 1000) return `${label} needs a prompt of 1 to 1,000 characters.`;
    if (!regionalPromptsBounds(row)) {
      return `${label} needs whole percentages from 0 to 100, spanning at least 5% across and down.`;
    }
  }
  return '';
}

function composerRegionalPrompts() {
  const draft = regionalPromptsDraft();
  if (!draft.length || regionalPromptsReason()) return null;
  return {
    kind: 'regional_prompts', method: REGIONAL_METHOD,
    regions: draft.map((row) => ({ prompt: row.prompt.trim(), ...regionalPromptsBounds(row) })),
  };
}

function regionalPromptsRow(region) {
  const row = document.createElement('fieldset');
  row.className = 'regional-prompt-row';
  const legend = document.createElement('legend');
  const promptLabel = document.createElement('label');
  const prompt = document.createElement('textarea');
  prompt.name = 'prompt';
  prompt.rows = 2;
  prompt.maxLength = 2000; // At most 1,000 Unicode characters, each up to two UTF-16 code units.
  prompt.defaultValue = region.prompt;
  promptLabel.append('Prompt', prompt);
  const coordinates = document.createElement('div');
  coordinates.className = 'regional-prompt-coordinates';
  for (const [name, word] of REGIONAL_EDGES) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'number';
    input.inputMode = 'numeric';
    input.name = name;
    input.min = 0;
    input.max = 100;
    input.step = 1;
    input.value = region[name];
    label.append(`${word} %`, input);
    coordinates.append(label);
  }
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'ghost small';
  remove.textContent = 'Remove';
  remove.addEventListener('click', () => {
    const next = row.nextElementSibling ?? row.previousElementSibling;
    row.remove();
    regionalPromptsChanged();
    (next?.querySelector('textarea') ?? $('regional-prompts-add')).focus({ preventScroll: true });
  });
  row.append(legend, promptLabel, coordinates, remove);
  return row;
}

function renderRegionalPromptGeometry() {
  const rows = regionalPromptsRows();
  const boxes = [];
  for (const [index, row] of rows.entries()) {
    row.querySelector('legend').textContent = `Region ${index + 1}`;
    row.querySelector('button').setAttribute('aria-label', `Remove region ${index + 1}`);
    const bounds = regionalPromptsBounds(Object.fromEntries(
      [...row.querySelectorAll('input')].map((input) => [input.name, input.value]),
    ));
    if (!bounds) continue;
    const marker = document.createElement('span');
    marker.textContent = index + 1;
    marker.style.inset = `${bounds.top}% ${100 - bounds.right}% ${100 - bounds.bottom}% ${bounds.left}%`;
    boxes.push(marker);
  }
  const preview = $('regional-prompts-map');
  preview.replaceChildren(...boxes);
  preview.hidden = rows.length === 0;
}

function restoreRegionalPrompts(regions, reveal) {
  const valid = Array.isArray(regions) ? regions.slice(0, REGIONAL_MAX).filter((region) => region
    && typeof region.prompt === 'string'
    && REGIONAL_EDGES.every(([key]) => ['number', 'string'].includes(typeof region[key]))) : [];
  $('regional-prompts-rows').replaceChildren(...valid.map(regionalPromptsRow));
  if (reveal) $('regional-prompts-section').open = valid.length > 0;
  renderRegionalPromptGeometry();
}

function syncRegionalPrompts() {
  const section = $('regional-prompts-section');
  const count = regionalPromptsRows().length;
  const unavailable = regionalPromptsCapabilityReason();
  section.hidden = !!unavailable && !section.open && count === 0;
  const summary = $('regional-prompts-summary');
  const text = count ? `${count} ${count === 1 ? 'region' : 'regions'}` : 'off';
  if (summary.textContent !== text) summary.textContent = text;
  const add = $('regional-prompts-add');
  const addReason = unavailable || (count >= REGIONAL_MAX ? 'Up to three regions.' : '');
  add.setAttribute('aria-disabled', String(!!addReason));
  add.title = addReason || 'Add a prompt for part of the image';
  const reason = $('regional-prompts-reason');
  const words = unavailable || regionalPromptsReason();
  if (reason.textContent !== words) reason.textContent = words;
  reason.hidden = !words;
}

function spatialConditioningDraft() {
  return { regions: regionalPromptsDraft(), controlnet: controlNetDraft() };
}

function spatialConditioningSelected() {
  return regionalPromptsRows().length > 0 || controlNetSelected();
}

function composerSpatialConditioning() {
  return composerRegionalPrompts() ?? composerControlNet();
}

function restoreSpatialConditioning(spec) {
  restoreRegionalPrompts(spec?.kind === 'regional_prompts' ? spec.regions : [], true);
  restoreControlNet(spec?.kind === 'controlnet' ? spec : null, true);
}

function restoreSpatialDraft(draft, reveal = true) {
  restoreRegionalPrompts(draft?.regions, reveal);
  restoreControlNetDraft(draft?.controlnet, reveal);
}

function spatialConditioningSubmitHold() {
  if (regionalPromptsRows().length && controlNetSelected()) return 'Choose either ControlNet or region prompts for this run.';
  const why = regionalPromptsReason() || controlNetSubmitHold();
  if (why) return why;
  return spatialConditioningSelected() && els.thenRefine.checked
    ? 'Turn off Then refine to run with spatial conditioning.' : '';
}

function spatialConditioningGridHold() {
  return spatialConditioningSelected() ? 'Turn off spatial conditioning to run a grid or prompt comparison.' : '';
}

function spatialConditioningReuseReason(spec) {
  if (spec?.kind === 'regional_prompts') {
    if (spec.method !== REGIONAL_METHOD) return 'This image uses a regional prompt method unavailable in this build.';
    return regionalPromptsCapabilityReason();
  }
  return spec?.kind === 'controlnet' ? controlNetReuseReason(spec) : '';
}

function syncSpatialConditioning() {
  syncRegionalPrompts();
  syncControlNetControls();
}

function regionalPromptsChanged() {
  renderRegionalPromptGeometry();
  syncSubmitButtons();
  persistPanel();
  syncMoreOptions();
}

function initRegionalPrompts() {
  $('regional-prompts-section').addEventListener('input', regionalPromptsChanged);
  $('regional-prompts-add').addEventListener('click', () => {
    const add = $('regional-prompts-add');
    if (add.getAttribute('aria-disabled') === 'true') { showError(add.title); return; }
    const count = regionalPromptsRows().length;
    const row = regionalPromptsRow({ prompt: '', left: count ? 50 : 0, top: 0, right: count ? 100 : 50, bottom: 100 });
    $('regional-prompts-rows').append(row);
    regionalPromptsChanged();
    row.querySelector('textarea').focus({ preventScroll: true });
  });
}
