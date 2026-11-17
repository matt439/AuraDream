// Explicit conversation context, continuation and request-owned progress.
// Declarations only; initPromptAssistant wires these controls.
'use strict';

let assistantBrief = '';
let assistantContextStart = 0;
let assistantFocus = null;
let assistantDraftsStale = false;
let assistantUnread = false;
let assistantContinuationSource = null;
let assistantContinuationOrigin = null;
let assistantContinuationFocus = null;

function assistantContextTurns() { return assistantTurns.slice(assistantContextStart); }

function assistantRequirements() {
  return $('assistant-requirements').value.split('\n').map((text) => text.trim()).filter(Boolean);
}

function assistantRequirementsReason() {
  const pins = assistantRequirements();
  return pins.length > 8 ? 'Keep at most eight pinned requirements.'
    : pins.some((text) => [...text].length > 200) ? 'Keep each pinned requirement to at most 200 characters.' : '';
}

function assistantRequirementsChanged() {
  return !!assistantResultData && JSON.stringify(assistantRequirements())
    !== JSON.stringify(assistantResultData.requestRequirements || []);
}

function assistantContextSize(instruction = els.assistantInstruction.value.trim()) {
  const source = assistantSource();
  return [source.prompt, source.negative, instruction, assistantBrief, assistantFocus?.prompt || '',
    ...assistantRequirements(), ...assistantContextTurns().map((turn) => turn.content)]
    .reduce((sum, text) => sum + [...text].length, 0);
}

function assistantTaskWords() {
  const mode = assistantMode;
  els.assistantInstruction.placeholder = mode === 'improve' ? 'What would you like to change?'
    : mode === 'tags' ? 'Describe the detail you need tags for\u2026' : 'Describe your idea\u2026';
  $('assistant-intro').querySelector('h3').textContent = mode === 'improve' ? 'What would you like to change?'
    : mode === 'tags' ? 'Which detail needs a tag?' : 'What would you like to create?';
  $('assistant-intro').querySelector('p').textContent = mode === 'improve'
    ? 'Start with your prompt in the editor, or choose Refine this option on a draft.'
    : mode === 'tags' ? 'Describe a detail to find candidate tags checked against the Danbooru index.'
      : 'Describe an idea and develop it into a prompt with follow-up messages.';
}

function syncAssistantWorkflow() {
  if (!$('assistant-context')) return;
  const turns = assistantContextTurns();
  const left = Math.max(0, Math.floor((ASSISTANT_HISTORY_LIMIT - turns.length) / 2) + 1);
  const near = left <= 1 || assistantContextSize() >= 9000 || turns.some((turn) => [...turn.content].length >= 4500);
  const full = !!assistantHistoryReason();
  const words = full ? 'Chat full \u00b7 continue with a brief' : near ? 'Chat nearly full' : `Room for ${left} ${left === 1 ? 'reply' : 'replies'}`;
  const meter = $('assistant-context');
  meter.textContent = words;
  meter.title = 'Conversation space is bounded by message count and text length. Token limits also depend on the selected local model.';
  meter.classList.toggle('warn', near || full);
  const next = $('assistant-continue');
  next.hidden = !assistantTurns.length && !assistantSourceChanged() && !assistantBrief;
  assistantHeld(next, assistantRun ? 'Wait for the answer, or press Stop first.' : '', 'Review a brief and continue with the current editor text; earlier messages remain visible');
  const pins = assistantRequirements();
  $('assistant-pins-label').textContent = pins.length ? `Requirements (${pins.length})` : 'Requirements';
  $('assistant-requirements-note').textContent = assistantRequirementsReason()
    || 'One requirement per line, up to eight. Exact phrase checks are advisory; review meaning and image results yourself.';
  $('assistant-focus').hidden = !assistantFocus;
  $('assistant-focus-text').textContent = assistantFocus ? `Refining: ${assistantFocus.title}` : '';
  assistantHeld($('assistant-focus-clear'), assistantRun ? 'Wait for the current answer, or press Stop.' : '', 'Use the main prompt as the next message\u2019s source');
  for (const article of els.assistantSidebar.querySelectorAll('[data-assistant-prompt]')) {
    const current = article.dataset.assistantPrompt === els.prompt.value;
    if (current) article.setAttribute('aria-current', 'true'); else article.removeAttribute('aria-current');
    article.querySelector('.assistant-applied').hidden = !current;
  }
  const model = assistantModelStatus();
  const readiness = $('assistant-readiness');
  readiness.textContent = model.ready ? 'Ready locally' : model.fetch?.length
    ? `Download ${humanBytes(model.fetch_bytes)}` : assistantModelsFailed ? 'Download details unavailable' : 'Checking local files\u2026';
  readiness.title = `${model.label}. ${readiness.textContent}${model.ready ? '' : ' on first use'}. Download size is not a memory measurement.`;
  syncAssistantDraftTools();
}

function focusAssistantDraft(prompt, title) {
  if (assistantRun) { assistantSay('Wait for the current answer, or press Stop.', true); return; }
  if (assistantSourceChanged()) assistantDraftsStale = true;
  assistantSnapshot = assistantSource();
  assistantFocus = { prompt, title };
  assistantMode = 'improve'; els.assistantMode.value = 'improve';
  assistantTaskWords();
  assistantSay(`Your next message refines ${title}. The main prompt is unchanged.`);
  syncPromptAssistant();
  els.assistantInstruction.focus({ preventScroll: true });
}

function openAssistantContinuation() {
  if (assistantRun) { assistantSay('Wait for the answer, or press Stop first.', true); return; }
  const dialog = $('assistant-continuation-dialog');
  if (dialog.open) return;
  assistantContinuationOrigin = document.activeElement;
  assistantContinuationSource = assistantSource();
  assistantContinuationFocus = assistantFocus;
  const last = [...assistantTurns].reverse().find((turn) => turn.role === 'user')?.content;
  const suggested = [assistantBrief, last ? `Latest direction: ${last}` : ''].filter(Boolean).join('\n');
  $('assistant-brief').value = [...suggested].length <= 2000 ? suggested : assistantBrief;
  $('assistant-continuation-source').textContent = assistantContinuationSource.prompt || '(The main prompt is empty.)';
  $('assistant-continuation-focus-row').hidden = !assistantContinuationFocus;
  $('assistant-continuation-focus-keep').checked = !!assistantContinuationFocus;
  $('assistant-continuation-option').hidden = !assistantContinuationFocus;
  $('assistant-continuation-option-text').textContent = assistantContinuationFocus?.prompt || '';
  $('assistant-continuation-status').textContent = 'Review the brief. Earlier messages stay visible but are no longer sent; requirements remain pinned. Old options stay held until a new answer.';
  dialog.showModal();
}

function applyAssistantContinuation() {
  const source = assistantSource();
  const why = assistantRun ? 'Wait for the answer to stop first.'
    : !assistantContinuationSource || Object.keys(source).some((key) => source[key] !== assistantContinuationSource[key])
      ? 'The editor or checkpoint changed while this brief was open. Close and reopen it to review the current prompt.'
      : [...$('assistant-brief').value].length > 2000 ? 'Keep the brief to at most 2000 characters.' : '';
  if (why) { $('assistant-continuation-status').textContent = why; return; }
  assistantBrief = $('assistant-brief').value;
  assistantContextStart = assistantTurns.length;
  assistantSnapshot = source;
  assistantFocus = $('assistant-continuation-focus-keep').checked ? assistantContinuationFocus : null;
  assistantDraftsStale = !!assistantResultData;
  $('assistant-continuation-dialog').close();
  assistantSay('Continuing with the current prompt and your reviewed brief. Ask for a new answer.');
  syncPromptAssistant();
}

function renderAssistantRequirements(article, checks) {
  if (!checks?.length) return;
  const detail = document.createElement('details'); detail.className = 'assistant-requirement-checks';
  const summary = document.createElement('summary');
  const missing = checks.filter((item) => !item.matched).length;
  summary.textContent = missing ? `Review ${missing} pinned ${missing === 1 ? 'requirement' : 'requirements'}` : 'Pinned phrases present';
  if (missing) summary.className = 'warn';
  detail.append(summary);
  for (const check of checks) {
    const line = document.createElement('p');
    line.textContent = `${check.matched ? 'Exact phrase present' : 'Review requirement'}: ${check.text}`;
    detail.append(line);
  }
  const note = document.createElement('p'); note.className = 'hint';
  note.textContent = 'Phrase presence checks do not establish meaning or what an image will contain.';
  detail.append(note); article.append(detail);
}

function assistantProgressWords(state) {
  const phases = { preparing: 'Preparing the local model\u2026', downloading: 'Downloading the local model\u2026',
    waiting: 'Waiting for the GPU to be idle\u2026', starting: 'Starting the local model\u2026',
    writing: 'Writing an answer\u2026', verifying: 'Verifying proposed tag names\u2026' };
  let words = phases[state.phase];
  if (words && state.phase === 'downloading' && Number.isFinite(state.downloaded_bytes)) {
    words += ` ${humanBytes(state.downloaded_bytes)}`;
    if (Number.isFinite(state.total_bytes) && state.total_bytes > 0) words += ` of ${humanBytes(state.total_bytes)}`;
  }
  return words;
}

async function pollAssistantProgress(run) {
  if (assistantRun !== run || run.stopped || !assistantSidebarOpen()) return;
  try {
    const state = await api(`/api/prompt/assist/${encodeURIComponent(run.ticket)}/status`, polledRead());
    if (assistantRun !== run || run.stopped || run.epoch !== assistantEpoch || !assistantSidebarOpen()) return;
    const words = assistantProgressWords(state);
    if (words) assistantSay(words);
  } catch { /* POST owns errors; missing early status and old servers keep the local clock. */ }
  if (assistantRun === run && !run.stopped && assistantSidebarOpen()) {
    run.progressTimer = setTimeout(() => { void pollAssistantProgress(run); }, 750);
  }
}

function resetAssistantContext() {
  assistantBrief = ''; assistantContextStart = 0; assistantFocus = null;
  assistantDraftsStale = false; assistantUnread = false;
  $('assistant-latest').hidden = true;
}

function initAssistantConversation() {
  $('assistant-continue').onclick = openAssistantContinuation;
  $('assistant-continuation-apply').onclick = applyAssistantContinuation;
  $('assistant-continuation-cancel').onclick = () => $('assistant-continuation-dialog').close();
  $('assistant-continuation-dialog').addEventListener('close', () => {
    if ($('assistant-continuation-dialog').open) return;
    assistantContinuationSource = null;
    assistantContinuationFocus = null;
    const origin = assistantContinuationOrigin; assistantContinuationOrigin = null;
    if (origin?.isConnected && isShown(origin)) origin.focus({ preventScroll: true });
  });
  $('assistant-focus-clear').onclick = () => {
    if (assistantRun) { assistantSay('Wait for the answer, or press Stop.', true); return; }
    assistantFocus = null; syncPromptAssistant();
    assistantSay('Your next message uses the current main prompt.');
  };
  $('assistant-requirements').addEventListener('input', syncPromptAssistant);
  $('assistant-expand').onclick = () => {
    const frame = els.assistantInstruction.closest('.assistant-input');
    const on = frame.classList.toggle('expanded');
    $('assistant-expand').textContent = on ? 'Collapse' : 'Expand';
    $('assistant-expand').setAttribute('aria-expanded', String(on));
    els.assistantInstruction.focus({ preventScroll: true });
  };
  $('assistant-latest').onclick = () => {
    assistantUnread = false; $('assistant-latest').hidden = true;
    scrollAssistantConversation();
  };
  els.assistantSidebar.querySelector('.assistant-body').addEventListener('scroll', () => {
    if (assistantScrollPosition().follow) { assistantUnread = false; $('assistant-latest').hidden = true; }
  });
}
