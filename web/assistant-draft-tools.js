// Optional actions on current and earlier assistant drafts. Declarations only;
// app.js initializes these after every classic script has been declared.
'use strict';

let assistantDraftCheckEpoch = 0;
let assistantDraftCopyEpoch = 0;
let assistantComparisonEpoch = 0;
let assistantComparisonPrepared = null;
const assistantDraftToolOrigins = new Map();
const assistantCompareDrafts = new Map();

function assistantToolElement(tag, text = '', className = '') {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function assistantDraftToolReason(model) {
  return model !== loadedModel ? 'Choose this draft\u2019s image checkpoint before using this action.'
    : modelChangeHold() ? `${modelChangeHold()}.`
    : assistantRun ? 'Wait for the current answer, or press Stop.' : '';
}

function assistantToolFeedback(host, words, warn = false) {
  let line = host.querySelector(':scope > .assistant-tool-feedback');
  if (!line) {
    line = assistantToolElement('p', '', 'assistant-tool-feedback hint');
    line.setAttribute('role', 'status');
    host.append(line);
  }
  line.textContent = words;
  line.classList.toggle('warn', warn);
}

function assistantToolButton(menu, action, words, run) {
  const button = assistantToolElement('button', words, 'ghost small');
  button.type = 'button';
  button.dataset.assistantAction = action;
  button.onclick = run;
  menu.append(button);
  return button;
}

function renderAssistantProseDiff(menu, changes) {
  if (!changes?.segments?.length) return;
  const detail = assistantToolElement('details', '', 'assistant-prose-diff');
  detail.dataset.assistantProseDiff = '';
  detail.append(assistantToolElement('summary', 'Compare wording'));
  const explanation = changes.coarse
    ? 'Broad comparison with the prompt used for this reply. The replaced passage is shown in full.'
    : 'Compared with the prompt used for this reply. Added text is underlined; removed text is struck through.';
  detail.append(assistantToolElement('p', explanation, 'hint'));
  const text = assistantToolElement('p', '', 'assistant-prose-diff-text');
  for (const segment of changes.segments) {
    if (segment.kind === 'equal') text.append(document.createTextNode(segment.text));
    else if (['added', 'removed'].includes(segment.kind)) {
      const mark = assistantToolElement(segment.kind === 'added' ? 'ins' : 'del', segment.text);
      mark.title = segment.kind === 'added' ? 'Added' : 'Removed';
      text.append(mark);
    }
  }
  detail.append(text);
  menu.append(detail);
}

function renderAssistantDraftTools(article, body, index, { interactive = false } = {}) {
  const draft = body.drafts?.[index];
  if (!draft) return;
  const detail = assistantToolElement('details', '', 'assistant-draft-tools');
  const summary = assistantToolElement('summary', 'More');
  summary.setAttribute('aria-label', `More actions for ${draft.title || `draft ${index + 1}`}`);
  detail.append(summary);
  const menu = assistantToolElement('div', '', 'assistant-draft-tool-menu');
  const model = body.model_id;
  const key = `${model}\n${draft.prompt}`;
  const copy = assistantToolButton(menu, 'copy', 'Copy prompt', async () => {
    const epoch = ++assistantDraftCopyEpoch;
    if (!navigator.clipboard || !window.isSecureContext) {
      assistantToolFeedback(menu, 'The browser opens the clipboard only on a secure connection or localhost.', true);
      return;
    }
    try {
      // Keep the actual clipboard write inside the user\u2019s press.
      await navigator.clipboard.writeText(draft.prompt);
      if (epoch === assistantDraftCopyEpoch && article.isConnected) assistantToolFeedback(menu, 'Prompt copied.');
    } catch (err) {
      if (epoch === assistantDraftCopyEpoch && article.isConnected) {
        assistantToolFeedback(menu, `Could not copy: ${err.message || err}.`, true);
      }
    }
  });
  assistantHeld(copy, !navigator.clipboard || !window.isSecureContext
    ? 'The browser opens the clipboard only on a secure connection or localhost.' : '', 'Copy this exact draft without changing the editor');
  assistantToolButton(menu, 'save', 'Save as snippet\u2026', () => { void saveAssistantDraftSnippet(draft, menu); });
  const refine = assistantToolButton(menu, 'refine', 'Refine this option', () => {
    const why = assistantDraftToolReason(model);
    if (why) { assistantToolFeedback(menu, why, true); return; }
    detail.open = false;
    focusAssistantDraft(draft.prompt, draft.title || `Draft ${index + 1}`);
  });
  refine.dataset.assistantToolModel = model;
  const check = assistantToolButton(menu, 'check', 'Check this draft\u2026', () => {
    const why = assistantDraftToolReason(model) || (serverOffline ? 'The server is not answering.' : '');
    if (why) { assistantToolFeedback(menu, why, true); return; }
    detail.open = false;
    void checkAssistantDraft(draft, model, summary);
  });
  check.dataset.assistantToolModel = model;
  const compare = assistantToolButton(menu, 'compare', 'Add to comparison', () => {
    const why = assistantDraftToolReason(model);
    if (why) { assistantToolFeedback(menu, why, true); return; }
    if (assistantCompareDrafts.has(key)) assistantCompareDrafts.delete(key);
    else {
      if (assistantCompareDrafts.size >= 8) {
        assistantToolFeedback(menu, 'A comparison takes at most eight different prompts.', true);
        return;
      }
      assistantCompareDrafts.set(key, { model, prompt: draft.prompt, title: draft.title || `Draft ${index + 1}` });
    }
    syncAssistantDraftTools();
    assistantToolFeedback(menu, `${assistantCompareDrafts.size} ${assistantCompareDrafts.size === 1 ? 'option' : 'options'} selected for comparison.`);
  });
  compare.dataset.assistantCompareKey = key;
  compare.dataset.assistantToolModel = model;
  compare.setAttribute('aria-pressed', String(assistantCompareDrafts.has(key)));
  renderAssistantProseDiff(menu, body.prose_changes?.[index]);
  detail.append(menu);
  detail.addEventListener('toggle', () => {
    if (!detail.open) return;
    for (const other of els.assistantSidebar.querySelectorAll('.assistant-draft-tools[open]')) {
      if (other !== detail) other.open = false;
    }
    const bodyRect = els.assistantSidebar.querySelector('.assistant-body').getBoundingClientRect();
    const rect = detail.getBoundingClientRect();
    const above = rect.top - bodyRect.top, below = bodyRect.bottom - rect.bottom;
    menu.classList.toggle('assistant-menu-up', above > below);
    menu.style.maxHeight = `${Math.max(96, Math.max(above, below) - 8)}px`;
  });
  detail.addEventListener('keydown', (event) => {
    if (!detail.open || keyboardEventHeld(event)) return;
    if (event.key === 'Escape') {
      event.preventDefault(); event.stopPropagation();
      detail.open = false;
      summary.focus({ preventScroll: true });
    } else if (['1', '2', '3'].includes(event.key)) {
      // An open action menu owns its keys; digits must not apply a draft
      // through the sidebar\u2019s document-level shortcut handler underneath it.
      event.preventDefault(); event.stopPropagation();
    }
  });
  const head = article.querySelector('.assistant-draft-head');
  if (head) head.append(detail);
  else article.append(detail);
  article.dataset.assistantInteractive = String(interactive);
}

async function saveAssistantDraftSnippet(draft, host) {
  if (snippetsPending === 'write') {
    assistantToolFeedback(host, 'Wait for the current snippet to finish saving.', true);
    return;
  }
  if (els.snippetsDialog.open) return;
  const dirty = snippetsDirty;
  // Prefill before the list read settles. Its edit epoch and dirty flag then
  // protect this draft, and no late read may overwrite a newer dialog/editor.
  const opening = openSnippets();
  if (!els.snippetsDialog.open) return;
  if (dirty) {
    const dialogEpoch = snippetsDialogEpoch, editEpoch = snippetsEditEpoch;
    snippetSay('Your unsaved snippet text is kept. Save it or choose New before bringing in another draft.', true);
    await opening;
    if (els.snippetsDialog.open && snippetsDialogEpoch === dialogEpoch && snippetsEditEpoch === editEpoch) {
      snippetSay('Your unsaved snippet text is kept. Save it or choose New before bringing in another draft.', true);
    }
    return;
  }
  newSnippet();
  els.snippetsName.value = [...(draft.title || 'Assistant draft')].slice(0, 80).join('');
  els.snippetsText.value = draft.prompt;
  els.snippetsTarget.value = 'prompt';
  ++snippetsEditEpoch;
  snippetsDirty = true;
  const dialogEpoch = snippetsDialogEpoch, editEpoch = snippetsEditEpoch;
  snippetSay('Assistant draft ready. Edit its name, then choose Save. Nothing has been saved yet.');
  syncSnippets();
  els.snippetsName.focus({ preventScroll: true });
  await opening;
  if (els.snippetsDialog.open && snippetsDialogEpoch === dialogEpoch && snippetsEditEpoch === editEpoch) {
    snippetSay(snippetWriteRecoveryReason() || (snippetsReady
      ? 'Assistant draft ready. Edit its name, then choose Save. Nothing has been saved yet.'
      : 'Assistant draft ready. Reload the saved list before saving; your draft is kept.'), !snippetsReady);
  }
}

function assistantToolDialog(id, title, closeId) {
  const dialog = assistantToolElement('dialog', '', 'assistant-tools-dialog');
  dialog.id = id;
  dialog.setAttribute('aria-labelledby', `${id}-title`);
  const head = assistantToolElement('div', '', 'dialog-head');
  const heading = assistantToolElement('h2', title);
  heading.id = `${id}-title`;
  const close = assistantToolElement('button', 'Close', 'ghost small');
  close.id = closeId;
  close.type = 'button';
  close.onclick = () => dialog.close();
  head.append(heading, close);
  dialog.append(head);
  dialog.addEventListener('close', () => {
    if (dialog.open) return;
    if (id === 'assistant-draft-check-dialog') ++assistantDraftCheckEpoch;
    else { ++assistantComparisonEpoch; assistantComparisonPrepared = null; }
    const origin = assistantDraftToolOrigins.get(id);
    assistantDraftToolOrigins.delete(id);
    if (origin?.isConnected && isShown(origin)) origin.focus({ preventScroll: true });
  });
  document.body.append(dialog);
  return dialog;
}

async function checkAssistantDraft(draft, model, opener) {
  const dialog = $('assistant-draft-check-dialog');
  if (dialog.open) return;
  assistantDraftToolOrigins.set(dialog.id, opener);
  const epoch = ++assistantDraftCheckEpoch;
  const revision = modelRevision;
  const payload = { ...collect(), prompt: draft.prompt, num_images: 1, runs: 1 };
  $('assistant-draft-check-prompt').textContent = draft.prompt;
  $('assistant-draft-check-model').textContent = `Checked for ${checkpointName(model)} with the current generation settings. The editor is unchanged.`;
  const results = $('assistant-draft-check-results');
  results.replaceChildren();
  $('assistant-draft-check-status').textContent = 'Checking the draft with the CPU prompt analyser\u2026';
  dialog.showModal();
  try {
    const body = await api('/api/prompt/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (epoch !== assistantDraftCheckEpoch || !dialog.open) return;
    if (model !== loadedModel || revision !== modelRevision || body.model_id !== model) {
      $('assistant-draft-check-status').textContent = 'The image checkpoint changed. Close this check and check the draft again.';
      return;
    }
    renderFindings(results, body.findings || [], { empty: 'No findings from the available rules. This does not measure image quality.' });
    const encoders = assistantToolElement('div', '', 'assistant-draft-encoders');
    for (const report of body.encoders || []) {
      const field = report.field === 'negative_prompt' ? ' negative prompt' : '';
      const cut = report.cut == null ? '' : report.pooled_only ? ' \u00b7 pooled encoder truncates' : ' \u00b7 text is truncated';
      encoders.append(assistantToolElement('p', `${report.name}${field}: ${report.tokens} / ${report.budget} tokens${cut}`, report.cut != null && !report.pooled_only ? 'warn' : 'hint'));
    }
    if (body.encoders_detail || !body.encoders?.length) {
      encoders.append(assistantToolElement('p', body.encoders_detail || 'Tokenizer counts are unavailable; the token budget is unchecked.', 'hint'));
    }
    results.append(encoders);
    $('assistant-draft-check-status').textContent = 'Draft checked. Nothing was applied or generated.';
  } catch (err) {
    if (epoch === assistantDraftCheckEpoch && dialog.open) {
      $('assistant-draft-check-status').textContent = `The draft could not be checked: ${err.message}`;
    }
  }
}

function assistantComparisonReason() {
  return assistantExternalHold() || (assistantCompareDrafts.size < 2 ? 'Select at least two different draft prompts from More on their cards.' : '');
}

function syncAssistantDraftTools() {
  const prepare = $('assistant-compare');
  if (prepare) {
    prepare.hidden = assistantCompareDrafts.size === 0;
    assistantHeld(prepare, assistantComparisonReason(), 'Review the selected prompts on the same seeds; nothing runs until Generate comparison');
    prepare.textContent = assistantCompareDrafts.size ? `Compare options (${assistantCompareDrafts.size})` : 'Compare options';
  }
  for (const button of document.querySelectorAll('[data-assistant-tool-model]')) {
    const why = assistantDraftToolReason(button.dataset.assistantToolModel)
      || (button.dataset.assistantAction === 'check' && serverOffline ? 'The server is not answering.' : '');
    assistantHeld(button, why, button.dataset.assistantAction === 'refine'
      ? 'Use this option as the focus of your next message without applying it' : '');
    if (button.dataset.assistantAction === 'compare') {
      const selected = assistantCompareDrafts.has(button.dataset.assistantCompareKey);
      button.setAttribute('aria-pressed', String(selected));
      button.textContent = selected ? 'Remove from comparison' : 'Add to comparison';
    }
  }
  const generate = $('assistant-comparison-generate');
  if (generate) assistantHeld(generate, assistantPreparedComparisonReason(), 'Generate the reviewed prompts on the same captured settings and seeds');
}

function assistantPreparedComparisonReason() {
  const prepared = assistantComparisonPrepared;
  return !prepared ? 'Prepare a comparison first.'
    : prepared.invalidReason ? prepared.invalidReason
    : prepared.invalid || prepared.model !== loadedModel || prepared.revision !== modelRevision ? 'The image checkpoint changed. Prepare the comparison again.'
    : prepared.submitting ? 'The comparison is being submitted.'
    : !prepared.ready ? 'Wait for the server to check the comparison.'
    : assistantExternalHold() || submitHold({ composer: true }) || '';
}

async function prepareAssistantComparison() {
  const why = assistantComparisonReason();
  if (why) { assistantSay(why, true); return; }
  if (refuseBadNumber(panelNumberInputs({ batch: false, runs: false }))) return;
  const variants = [...assistantCompareDrafts.values()];
  if (variants.some((item) => item.model !== loadedModel)) {
    assistantSay('Select drafts from the current image checkpoint for this comparison.', true);
    return;
  }
  const unfilled = compareUnfilled(variants);
  if (unfilled) { assistantSay(unfilled, true); return; }
  const base = collect();
  if (base.seed != null && (!Number.isSafeInteger(base.seed) || base.seed < 0 || base.seed > Number.MAX_SAFE_INTEGER - 3)) {
    assistantSay(`Choose a whole base seed from 0 to ${Number.MAX_SAFE_INTEGER - 3}, or leave Seed empty, so all four comparison seeds are exact.`, true);
    return;
  }
  const epoch = ++assistantComparisonEpoch;
  const payload = JSON.parse(JSON.stringify({
    ...base, num_images: 1, runs: 1,
    prompt: variants[0].prompt, prompt_axis: '', prompt_mode: 'cross',
    steps_axis: '', size_axis: '', cfg_axis: '', mu_axis: '', lora_scale_axis: '', strength_axis: '',
    seeds: 4, seed_axis: '', resume: false, budget_minutes: null,
    prompt_variants: variants.map((item, index) => ({ label: `option-${index + 1}`, prompt: item.prompt })),
  }));
  const prepared = { payload, model: loadedModel, revision: modelRevision, ready: false, epoch };
  assistantComparisonPrepared = prepared;
  const dialog = $('assistant-comparison-dialog');
  assistantDraftToolOrigins.set(dialog.id, document.activeElement);
  const list = $('assistant-comparison-options');
  list.replaceChildren();
  for (const [index, variant] of variants.entries()) {
    const item = assistantToolElement('details');
    item.append(assistantToolElement('summary', `${index + 1}. ${variant.title}`), assistantToolElement('p', variant.prompt, 'assistant-draft-text'));
    list.append(item);
  }
  const seed = payload.seed == null ? 'a random base seed chosen by the server' : `base seed ${payload.seed}`;
  $('assistant-comparison-settings').textContent = `${checkpointName(prepared.model)} \u00b7 ${payload.width} \u00d7 ${payload.height} \u00b7 four shared seeds from ${seed}. Each option uses one image per seed with the same generation settings, negative prompt, references and LoRA. Repeat, grid axes and Then refine do not apply. Your prompt and grid fields stay unchanged.`;
  $('assistant-comparison-status').textContent = 'Checking the comparison\u2026';
  syncAssistantDraftTools();
  dialog.showModal();
  try {
    const preview = await api('/api/grid/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (epoch !== assistantComparisonEpoch || assistantComparisonPrepared !== prepared || !dialog.open) return;
    prepared.ready = true;
    const estimate = preview.seconds ? ` \u00b7 about ${humanTime(preview.seconds)}` : '';
    const duplicates = preview.duplicates ? ` \u00b7 ${preview.duplicates} identical cells` : '';
    $('assistant-comparison-status').textContent = `${preview.images} images \u00b7 ${variants.length} options \u00d7 4 shared seeds${estimate}${duplicates}. Settings are captured; later editor changes do not alter this comparison. Choose Generate comparison to run it.`;
  } catch (err) {
    if (epoch !== assistantComparisonEpoch || assistantComparisonPrepared !== prepared || !dialog.open) return;
    prepared.invalidReason = `The comparison could not be prepared: ${err.message}. Close this dialog and prepare it again.`;
    $('assistant-comparison-status').textContent = prepared.invalidReason;
  }
  syncAssistantDraftTools();
}

function generateAssistantComparison() {
  const why = assistantPreparedComparisonReason();
  if (why) { $('assistant-comparison-status').textContent = why; return; }
  const prepared = assistantComparisonPrepared;
  prepared.submitting = true;
  syncAssistantDraftTools();
  void Promise.resolve(startJob('/api/grid', prepared.payload, {
    sheet: true, panel: false, model: prepared.model, revision: prepared.revision,
    slot: $('assistant-comparison-status'),
    onAccepted: () => {
      if (assistantComparisonPrepared === prepared) $('assistant-comparison-dialog').close();
    },
    onRefused: (err) => {
      if (assistantComparisonPrepared !== prepared || (!imageGone(err) && !loraRefused(err))) return;
      prepared.invalidReason = imageGone(err)
        ? 'A captured reference is unavailable. Close this dialog and prepare the comparison again with the current references.'
        : 'The captured LoRA is unavailable. Close this dialog and prepare the comparison again with the current LoRA choice.';
      $('assistant-comparison-status').textContent = prepared.invalidReason;
    },
  })).finally(() => {
    if (assistantComparisonPrepared === prepared) {
      prepared.submitting = false;
      syncAssistantDraftTools();
    }
  });
}

function resetAssistantDraftTools(reason = 'The image checkpoint changed.') {
  assistantCompareDrafts.clear();
  ++assistantDraftCopyEpoch;
  ++assistantDraftCheckEpoch;
  if (assistantComparisonPrepared) {
    assistantComparisonPrepared.invalid = true;
    $('assistant-comparison-status').textContent = `${reason} Close this dialog and prepare the comparison again.`;
  }
  const check = $('assistant-draft-check-dialog');
  if (check?.open) $('assistant-draft-check-status').textContent = `${reason} Close this check and check the draft again.`;
  syncAssistantDraftTools();
}

function initAssistantDraftTools() {
  const check = assistantToolDialog('assistant-draft-check-dialog', 'Check assistant draft', 'assistant-draft-check-close');
  const model = assistantToolElement('p', '', 'hint'); model.id = 'assistant-draft-check-model';
  const prompt = assistantToolElement('p', '', 'assistant-draft-text'); prompt.id = 'assistant-draft-check-prompt';
  const status = assistantToolElement('p'); status.id = 'assistant-draft-check-status'; status.setAttribute('role', 'status');
  const results = assistantToolElement('div'); results.id = 'assistant-draft-check-results';
  check.append(model, prompt, status, results);
  const compare = assistantToolDialog('assistant-comparison-dialog', 'Compare assistant options', 'assistant-comparison-close');
  const settings = assistantToolElement('p', '', 'hint'); settings.id = 'assistant-comparison-settings';
  const options = assistantToolElement('div'); options.id = 'assistant-comparison-options';
  const said = assistantToolElement('p'); said.id = 'assistant-comparison-status'; said.setAttribute('role', 'status');
  const generate = assistantToolElement('button', 'Generate comparison', 'primary'); generate.type = 'button'; generate.id = 'assistant-comparison-generate';
  generate.onclick = generateAssistantComparison;
  compare.append(settings, options, said, generate);
  const prepare = $('assistant-compare');
  if (prepare) prepare.onclick = () => { void prepareAssistantComparison(); };
  syncAssistantDraftTools();
}
