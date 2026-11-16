// Local prompt writing, revision and verified tag proposals. Declarations only;
// app.js wires the dialog after the classic scripts have all been declared.
'use strict';

let assistantRun = null;
let assistantResultData = null;
let assistantSnapshot = null;
let assistantTurns = [];
let assistantEpoch = 0;
let assistantOrigin = null;
let assistantMode = 'create';
let assistantSelectedTags = new Set();
let assistantClockTimer = null;

const ASSISTANT_INPUT_LIMIT = 12000;
const ASSISTANT_HISTORY_LIMIT = 8;

function assistantSource() {
  return { prompt: els.prompt.value, negative: els.negative.value, model: loadedModel, revision: modelRevision };
}

function assistantSourceChanged() {
  if (!assistantSnapshot) return false;
  const now = assistantSource();
  return Object.keys(now).some((key) => now[key] !== assistantSnapshot[key]);
}

function assistantSay(words, warn = false) {
  if (els.assistantStatus.textContent !== words) els.assistantStatus.textContent = words;
  els.assistantStatus.classList.toggle('warn', warn);
}

function assistantExternalHold() {
  return serverOffline ? 'The server is not answering.'
    : !promptCritiqueSupported ? 'The local prompt model is unavailable on this server.'
    : modelChangeHold() ? `${modelChangeHold()}.`
    : modelState === 'loading' ? 'Wait for the image model to finish loading.'
    : busy || elsewhereBusy ? 'An image job is queued or running. Wait for the image queue to finish.'
    : critiquing ? 'Wait for prompt improvements to finish.'
    : expandingPrompt ? 'Wait for prompt expansion to finish.'
    : cardHolder && cardTicket !== assistantRun?.ticket ? 'Wait for the current GPU work to finish.'
    : '';
}

function assistantHistoryReason(instruction = els.assistantInstruction.value.trim()) {
  if (assistantTurns.length > ASSISTANT_HISTORY_LIMIT
      || assistantTurns.some((turn) => [...turn.content].length > 6000)) {
    return 'This conversation has reached its context limit. Start a new conversation; the drafts stay unchanged until you choose one.';
  }
  const source = assistantSource();
  const length = [source.prompt, source.negative, instruction, ...assistantTurns.map((turn) => turn.content)]
    .reduce((sum, text) => sum + [...text].length, 0);
  return length > ASSISTANT_INPUT_LIMIT
    ? 'This conversation is too long for the local model. Shorten your instruction or start a new conversation.' : '';
}

function assistantAskReason() {
  const instruction = els.assistantInstruction.value.trim();
  const source = assistantSource();
  return assistantRun ? (assistantRun.stopped ? 'Wait for the local answer to stop.' : 'Wait for the current answer, or press Stop.')
    : assistantExternalHold()
    || (assistantSourceChanged() ? 'The prompt, negative prompt or checkpoint changed. Start a new conversation for the current text.' : '')
    || (!instruction ? 'Describe what you want help with first.' : '')
    || ([...instruction].length > 2000 ? 'Use an instruction of at most 2000 characters.' : '')
    || ([...source.prompt].length > 4000 || [...source.negative].length > 4000 ? 'Shorten the prompt and negative prompt to at most 4000 characters each.' : '')
    || (assistantMode === 'improve' && !source.prompt.trim() ? 'Enter a prompt in the main editor before choosing Improve.' : '')
    || assistantHistoryReason(instruction);
}

function assistantApplyReason() {
  return assistantRun ? 'Wait for the current answer to finish stopping or thinking.'
    : serverOffline ? 'The server is not answering. Reconnect before applying a draft to confirm the current checkpoint.'
    : modelChangeHold() ? `${modelChangeHold()}.`
    : assistantSourceChanged() ? 'The prompt, negative prompt or checkpoint changed. Start a new conversation before applying a draft.'
    : !assistantResultData ? 'Ask the assistant for a draft first.' : '';
}

function assistantHeld(button, reason, title) {
  button.setAttribute('aria-disabled', String(!!reason));
  button.title = reason || title;
}

function syncPromptAssistant() {
  if (!els.assistantDialog) return;
  assistantHeld(els.assistantAsk, assistantAskReason(), 'Ask the local model; nothing is applied automatically');
  assistantHeld(els.assistantStop, !assistantRun ? 'There is no answer running.'
    : assistantRun.stopping ? 'The stop request is being sent.' : '', 'Stop this answer and release its local process');
  assistantHeld(els.assistantReset, assistantRun ? 'Stop the current answer before starting a new conversation.' : '',
    'Clear this conversation; the main prompt stays unchanged');
  els.assistantAsk.textContent = assistantRun ? 'Thinking…' : 'Ask';
  els.assistantMode.setAttribute('aria-disabled', String(!!assistantRun));
  els.assistantMode.title = assistantRun ? 'Stop the current answer before changing the task.' : 'Choose the kind of prompt help';
  const applyReason = assistantApplyReason();
  for (const button of els.assistantResult.querySelectorAll('[data-assistant-draft]')) {
    assistantHeld(button, applyReason, 'Replace the main prompt with this draft; Ctrl+Z restores it');
  }
  for (const tick of els.assistantResult.querySelectorAll('[data-assistant-tag]')) {
    const tag = assistantResultData?.tags?.[Number(tick.dataset.assistantTag)];
    const invalid = !['valid', 'alias'].includes(tag?.status);
    assistantHeld(tick, applyReason || (invalid ? (tag?.detail || 'This proposed tag is not verified for use.') : ''),
      'Select this verified tag for insertion');
  }
  const insert = $('assistant-insert-tags');
  if (insert) assistantHeld(insert, applyReason || (!assistantSelectedTags.size ? 'Select at least one verified tag first.' : ''),
    'Append the selected verified tags to the prompt; Ctrl+Z restores it');
  els.assistantStale.hidden = !assistantSourceChanged();
  const critic = criticStatus ?? { label: 'local model', fetch: [], fetch_bytes: 0 };
  const fetch = !promptCritiqueReady && critic.fetch.length
    ? `First use downloads ${critic.fetch.join(' and ')} (about ${humanBytes(critic.fetch_bytes)}) once. This can take a few minutes.` : '';
  const cost = assistantRun
    ? assistantRun.stopped ? 'Stopping the local answer; waiting for its process to exit.'
      : fetch ? `${fetch} Then the local model writes an answer.` : `Waiting for ${critic.label} to finish its answer.`
    : assistantExternalHold() || fetch || `Uses local ${critic.label}; waits for image jobs.`;
  if (els.assistantCost.textContent !== cost) els.assistantCost.textContent = cost;
}

function assistantHistoryText(body) {
  // Keep the actual drafts in the next turn, not only the model's summary.
  // A long answer stays visible, but the next Ask is held rather than silently
  // truncating earlier instructions or the prompt the user wants to revise.
  return JSON.stringify({ message: body.message, drafts: body.drafts, tags: body.tags });
}

function renderAssistantHistory() {
  els.assistantHistory.replaceChildren();
  els.assistantHistoryWrap.hidden = !assistantTurns.length;
  for (const [index, turn] of assistantTurns.entries()) {
    const item = document.createElement('p');
    const who = document.createElement('strong');
    who.textContent = turn.role === 'user' ? 'You: ' : 'Assistant: ';
    item.append(who, document.createTextNode(turn.display ?? turn.content));
    item.dataset.turn = String(index);
    els.assistantHistory.append(item);
  }
}

function renderAssistantResult() {
  const host = els.assistantResult;
  host.replaceChildren();
  host.hidden = !assistantResultData;
  if (!assistantResultData) { syncPromptAssistant(); return; }
  const body = assistantResultData;
  const message = document.createElement('p');
  message.className = 'assistant-message';
  message.textContent = body.message;
  host.append(message);
  for (const warning of body.warnings || []) {
    const note = document.createElement('p');
    note.className = 'warn'; note.textContent = warning; host.append(note);
  }
  for (const [index, draft] of (body.drafts || []).entries()) {
    const article = document.createElement('article');
    article.className = 'assistant-draft';
    const title = document.createElement('h3'); title.textContent = draft.title || `Draft ${index + 1}`;
    const prompt = document.createElement('p'); prompt.className = 'assistant-draft-text'; prompt.textContent = draft.prompt;
    article.append(title, prompt);
    for (const words of draft.notes || []) {
      const note = document.createElement('p'); note.className = 'hint'; note.textContent = words; article.append(note);
    }
    const use = document.createElement('button'); use.type = 'button'; use.className = 'ghost small';
    use.textContent = 'Use prompt'; use.dataset.assistantDraft = String(index);
    use.onclick = () => applyAssistantDraft(index);
    article.append(use); host.append(article);
  }
  if (body.tags?.length) {
    const list = document.createElement('div'); list.className = 'assistant-tags';
    const note = document.createElement('p'); note.className = 'hint';
    note.textContent = 'The explanations below are model suggestions. The tag index decides which names are verified; only verified names can be selected.';
    list.append(note);
    for (const [index, tag] of body.tags.entries()) {
      const row = document.createElement('div'); row.className = 'assistant-tag';
      const label = document.createElement('label'); label.className = 'check';
      const tick = document.createElement('input'); tick.type = 'checkbox'; tick.dataset.assistantTag = String(index);
      tick.checked = assistantSelectedTags.has(index);
      const name = tag.status === 'alias' ? (tag.canonical || tag.text) : tag.text;
      const words = document.createElement('span'); words.textContent = name;
      label.append(tick, words); row.append(label);
      const verdict = document.createElement('p'); verdict.className = 'assistant-tag-verdict';
      verdict.textContent = `${tag.status === 'alias' ? `Alias of ${name}` : tag.status === 'valid' ? 'Verified tag' : 'Unverified tag'}${tag.detail ? `: ${tag.detail}` : ''}`;
      row.append(verdict);
      if (tag.reason) {
        const explanation = document.createElement('p'); explanation.className = 'hint';
        explanation.textContent = `Suggested meaning: ${tag.reason}`; row.append(explanation);
      }
      tick.addEventListener('click', (event) => {
        const why = tick.getAttribute('aria-disabled') === 'true' ? tick.title : '';
        if (why) { event.preventDefault(); assistantSay(why, true); return; }
        if (tick.checked) assistantSelectedTags.add(index); else assistantSelectedTags.delete(index);
        syncPromptAssistant();
      });
      list.append(row);
    }
    const insert = document.createElement('button'); insert.type = 'button'; insert.id = 'assistant-insert-tags';
    insert.className = 'ghost small'; insert.textContent = 'Insert selected tags'; insert.onclick = insertAssistantTags;
    list.append(insert); host.append(list);
  }
  const footer = document.createElement('p'); footer.className = 'hint';
  footer.textContent = `${body.model} · ${Math.round(body.elapsedSeconds ?? body.seconds)} s total. Review the result before using it.`;
  host.append(footer);
  syncPromptAssistant();
}

function resetAssistantConversation({ clearInstruction = true } = {}) {
  if (assistantRun) { assistantSay('Stop the current answer before starting a new conversation.', true); return; }
  ++assistantEpoch;
  assistantTurns = [];
  assistantResultData = null;
  assistantSnapshot = null;
  assistantSelectedTags.clear();
  if (clearInstruction) els.assistantInstruction.value = '';
  renderAssistantHistory(); renderAssistantResult();
  assistantSay('Describe your idea or the change you want. Nothing is applied automatically.');
}

function openPromptAssistant(mode = null) {
  if (els.assistantDialog.open) return;
  assistantOrigin = document.activeElement;
  if (mode && mode !== assistantMode && !assistantRun) {
    assistantMode = mode; els.assistantMode.value = mode;
    resetAssistantConversation({ clearInstruction: false });
  }
  syncPromptAssistant();
  els.assistantDialog.showModal();
  // The phone's sheet setup keeps the keyboard away until the user chooses a
  // text box. Desktop authors can type straight away.
  if (!isPhone()) els.assistantInstruction.focus({ preventScroll: true });
}

function assistantClosed() {
  if (els.assistantDialog.open) return;
  if (assistantRun) void stopPromptAssistant();
  const origin = assistantOrigin; assistantOrigin = null;
  if (origin?.isConnected) origin.focus({ preventScroll: true });
}

function assistantClock() {
  els.assistantClock.textContent = assistantRun ? ` · ${Math.floor((Date.now() - assistantRun.started) / 1000)} s` : '';
}

async function askPromptAssistant() {
  const why = assistantAskReason();
  if (why) { assistantSay(why, true); if (serverOffline) void refreshStatus(); return; }
  const source = assistantSource();
  const instruction = els.assistantInstruction.value.trim();
  const run = { epoch: ++assistantEpoch, ticket: expandTicket(), mode: assistantMode, stopped: false, stopping: false, started: Date.now() };
  const payload = {
    ticket: run.ticket, mode: run.mode, instruction, prompt: source.prompt,
    negative_prompt: source.negative || null,
    history: assistantTurns.map(({ role, content }) => ({ role, content })),
  };
  assistantRun = run;
  assistantSnapshot = source;
  assistantSay('Asking the local model…');
  clearInterval(assistantClockTimer); assistantClock();
  assistantClockTimer = setInterval(assistantClock, 1000);
  syncPromptExpansion();
  void refreshStatus();
  try {
    const body = await api('/api/prompt/assist', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    promptCritiqueReady = true;
    if (run.epoch !== assistantEpoch || run.stopped || !els.assistantDialog.open) return;
    if (body.model_id !== source.model) {
      assistantSay('The checkpoint changed while this answer was prepared. Start a new conversation.', true); return;
    }
    assistantResultData = { ...body, elapsedSeconds: (Date.now() - run.started) / 1000 };
    assistantSelectedTags.clear();
    assistantTurns.push({ role: 'user', content: instruction },
      { role: 'assistant', content: assistantHistoryText(body), display: body.message });
    if (els.assistantInstruction.value.trim() === instruction) els.assistantInstruction.value = '';
    els.assistantInstruction.placeholder = 'Refine the answer, for example: keep the pose and make the lighting softer';
    renderAssistantHistory(); renderAssistantResult();
    assistantSay(assistantSourceChanged()
      ? 'The source text changed while the model answered. Nothing can be applied; start a new conversation for the current text.'
      : 'Answer ready. Choose a draft or ask for a change; the main prompt is unchanged.', assistantSourceChanged());
    if (document.activeElement === els.assistantAsk) els.assistantResult.focus({ preventScroll: true });
  } catch (err) {
    if (run.epoch !== assistantEpoch || run.stopped || !els.assistantDialog.open) return;
    assistantSay(err.offline ? 'The server did not answer. The main prompt is unchanged.' : `The assistant did not answer: ${err.message}`, true);
  } finally {
    if (assistantRun === run) {
      assistantRun = null;
      clearInterval(assistantClockTimer); assistantClockTimer = null; assistantClock();
      if (run.stopped && run.epoch === assistantEpoch) assistantSay('Answer stopped. The main prompt is unchanged.');
      syncPromptExpansion();
      void refreshStatus();
    }
  }
}

async function stopPromptAssistant() {
  const run = assistantRun;
  if (!run || run.stopping) return;
  run.stopped = true;
  run.stopping = true;
  assistantSay('Stopping the local answer…');
  syncPromptAssistant();
  try {
    await api(`/api/prompt/assist/${encodeURIComponent(run.ticket)}/stop`, { method: 'POST' });
  } catch (err) {
    if (assistantRun === run) assistantSay(`The stop request did not finish: ${err.message}. Press Stop to try again.`, true);
  } finally {
    run.stopping = false;
    syncPromptAssistant();
    void refreshStatus();
  }
}

function applyAssistantText(text, message) {
  const why = assistantApplyReason();
  if (why) { assistantSay(why, true); return; }
  const before = els.prompt.value;
  assistantOrigin = null;
  els.assistantDialog.close();
  // A modal makes the main editor inert: close it before the undo-aware write.
  rewriteBox(els.prompt, text);
  promptReplaced(); syncTagSelection(); scheduleTagValidation(0); schedulePreview(); persistPanel();
  els.prompt.focus({ preventScroll: true });
  els.prompt.setSelectionRange(0, 0);
  offerPromptUndo(message, () => {
    rewriteBox(els.prompt, before); promptReplaced(); syncTagSelection(); scheduleTagValidation(0); schedulePreview(); persistPanel();
  });
}

function applyAssistantDraft(index) {
  const draft = assistantResultData?.drafts?.[index];
  if (draft) applyAssistantText(draft.prompt, 'A local assistant draft replaced your prompt.');
}

function insertAssistantTags() {
  const why = assistantApplyReason();
  if (why) { assistantSay(why, true); return; }
  const tags = [...assistantSelectedTags].map((index) => assistantResultData?.tags?.[index])
    .filter((tag) => tag && ['valid', 'alias'].includes(tag.status))
    .map((tag) => tag.status === 'alias' ? tag.canonical || tag.text : tag.text);
  const existing = new Set(promptTagParts().map(tagName));
  const fresh = [...new Set(tags)].filter((text) => !existing.has(tagName(text)));
  if (!fresh.length) { assistantSay(tags.length ? 'Those tags are already in the prompt.' : 'Select at least one verified tag first.', true); return; }
  const source = els.prompt.value.trimEnd();
  const join = source && !source.endsWith(',') ? ', ' : source ? ' ' : '';
  applyAssistantText(`${source}${join}${fresh.join(', ')}`, 'Selected assistant tags were appended to your prompt.');
}

function assistantCheckpointChanged() {
  if (assistantRun) { ++assistantEpoch; void stopPromptAssistant(); }
  assistantTurns = []; assistantResultData = null; assistantSnapshot = null; assistantSelectedTags.clear();
  renderAssistantHistory(); renderAssistantResult();
  if (els.assistantDialog.open) assistantSay('The checkpoint changed. This conversation was cleared; your instruction and the main prompt are unchanged.', true);
}

function appendAssistantMenuItem() {
  const item = document.createElement('li');
  item.id = 'prompt-assistant-menu-item'; item.setAttribute('role', 'menuitem'); item.tabIndex = -1;
  item.setAttribute('aria-haspopup', 'dialog'); item.setAttribute('aria-controls', 'prompt-assistant-dialog');
  item.textContent = 'Write a prompt…'; item.title = 'Create or improve a prompt, or find tags with a local model';
  const open = () => { showExamples(false); openPromptAssistant(); };
  item.onclick = open;
  item.onkeydown = (event) => {
    if (keyboardEventHeld(event) || keyboardCommandModifier(event) || event.shiftKey) return;
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); if (!event.repeat) open(); }
  };
  els.examplesMenu.append(item);
}

function initPromptAssistant() {
  els.assistantClose.onclick = () => els.assistantDialog.close();
  els.assistantDialog.addEventListener('close', assistantClosed);
  els.assistantAsk.onclick = () => { void askPromptAssistant(); };
  els.assistantStop.onclick = () => { void stopPromptAssistant(); };
  els.assistantReset.onclick = () => resetAssistantConversation();
  els.assistantMode.onchange = () => {
    if (assistantRun) { els.assistantMode.value = assistantMode; assistantSay('Stop the current answer before changing the task.', true); return; }
    assistantMode = els.assistantMode.value;
    resetAssistantConversation({ clearInstruction: false });
  };
  els.assistantInstruction.addEventListener('input', syncPromptAssistant);
  els.prompt.addEventListener('input', syncPromptAssistant);
  els.negative.addEventListener('input', syncPromptAssistant);
  els.assistantInstruction.addEventListener('keydown', (event) => {
    if (!keyboardSubmit(event)) return;
    event.preventDefault(); event.stopPropagation();
    if (!event.repeat) void askPromptAssistant();
  });
  syncPromptAssistant();
}
