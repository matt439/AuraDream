// Local prompt writing, revision and verified tag proposals. Declarations only;
// app.js wires the sidebar after the classic scripts have all been declared.
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
let assistantPendingText = '';
let assistantDraftPressed = null;
let assistantModel = 'default';
let assistantModels = [];
let assistantModelsEpoch = 0;
let assistantModelsFailed = false;
let assistantInstructionVersion = 0;

const ASSISTANT_INPUT_LIMIT = 12000;
const ASSISTANT_HISTORY_LIMIT = 8;

function assistantModelStatus() {
  const id = assistantRun?.model || assistantModel;
  if (id === 'default') {
    return { id, ready: promptCritiqueReady, ...(criticStatus ?? { label: 'Qwen3-4B Q8_0', fetch: [], fetch_bytes: 0 }) };
  }
  return assistantModels.find((model) => model.id === id)
    ?? { id, label: els.assistantModel.selectedOptions[0]?.textContent || 'local model', ready: false, fetch: [], fetch_bytes: 0 };
}

async function refreshAssistantModels() {
  const epoch = ++assistantModelsEpoch;
  try {
    const body = await api('/api/prompt/assist/models');
    if (epoch !== assistantModelsEpoch || !assistantSidebarOpen()) return;
    if (!Array.isArray(body.models) || !body.models.some((model) => model.id === 'default')) {
      throw new Error('The assistant model list is incomplete.');
    }
    assistantModels = body.models;
    assistantModelsFailed = false;
    for (const option of els.assistantModel.options) {
      const model = assistantModels.find((item) => item.id === option.value);
      if (model && option.textContent !== model.label) option.textContent = model.label;
    }
    syncPromptAssistant();
  } catch {
    if (epoch !== assistantModelsEpoch || !assistantSidebarOpen()) return;
    assistantModelsFailed = true;
    syncPromptAssistant();
  }
}

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
  els.assistantStatus.title = words;
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
  if (!els.assistantSidebar) return;
  assistantHeld(els.assistantAsk, assistantAskReason(), 'Ask the local model; nothing is applied automatically');
  assistantHeld(els.assistantStop, !assistantRun ? 'There is no answer running.'
    : assistantRun.stopping ? 'The stop request is being sent.' : '', 'Stop this answer and release its local process');
  assistantHeld(els.assistantReset, assistantRun ? 'Stop the current answer before starting a new conversation.' : '',
    'Clear this conversation; the main prompt stays unchanged');
  els.assistantAsk.textContent = assistantRun ? 'Thinking…' : 'Send';
  els.assistantMode.setAttribute('aria-disabled', String(!!assistantRun));
  els.assistantMode.title = assistantRun ? 'Stop the current answer before changing the task.' : 'Choose the kind of prompt help';
  els.assistantModel.setAttribute('aria-disabled', String(!!assistantRun));
  els.assistantModel.title = assistantRun ? 'Stop the current answer before changing the assistant model.'
    : 'Choose the local model for your next message; this conversation stays available';
  const applyReason = assistantApplyReason();
  for (const button of els.assistantResult.querySelectorAll('[data-assistant-draft]')) {
    const number = Number(button.dataset.assistantDraft) + 1;
    assistantHeld(button, applyReason, `Use prompt ${number}: press ${number} outside text fields, or Alt+${number} in the assistant. Ctrl+Z restores it.`);
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
  $('assistant-intro').hidden = !!assistantTurns.length || !!assistantResultData || !!assistantPendingText;
  $('assistant-pending').hidden = !assistantPendingText;
  $('assistant-pending').textContent = assistantPendingText;
  const model = assistantModelStatus();
  const fetch = !model.ready && model.fetch.length
    ? `First use downloads ${model.fetch.join(' and ')} (about ${humanBytes(model.fetch_bytes)}) once. This can take a few minutes.` : '';
  const cost = assistantRun
    ? assistantRun.stopped ? 'Stopping the local answer; waiting for its process to exit.'
      : fetch ? `${fetch} Then the local model writes an answer.` : `Waiting for ${model.label} to finish its answer.`
    : assistantExternalHold() || fetch || (assistantModelsFailed
      ? `Uses local ${model.label}. Download details are unavailable; Send will check the selected model.`
      : `Uses local ${model.label}; waits for image jobs.`);
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
    // The newest answer has the live Use controls below the history. Earlier
    // drafts remain readable in the conversation without offering stale Use.
    if (turn.role === 'assistant' && index === assistantTurns.length - 1) continue;
    const item = document.createElement('div');
    item.className = `assistant-turn assistant-turn-${turn.role}`;
    if (turn.role === 'assistant' && turn.answer) renderAssistantAnswer(item, turn.answer, false);
    else {
      const who = document.createElement('strong'); who.className = 'assistant-speaker';
      who.textContent = turn.role === 'user' ? 'You' : 'Assistant';
      const text = document.createElement('p'); text.textContent = turn.display ?? turn.content;
      item.append(who, text);
    }
    item.dataset.turn = String(index);
    els.assistantHistory.append(item);
  }
}

function renderAssistantResult() {
  const host = els.assistantResult;
  host.replaceChildren();
  host.hidden = !assistantResultData;
  if (!assistantResultData) { syncPromptAssistant(); return; }
  renderAssistantAnswer(host, assistantResultData, true);
  syncPromptAssistant();
}

function assistantMessageRepeatsDraft(body) {
  // Some model replies copy the structured drafts into their conversational
  // text too. Keep that text available, without showing a second prompt list.
  const normal = (text) => String(text ?? '').replace(/\s+/gu, ' ').trim();
  const message = normal(body.message);
  return (body.drafts || []).some((draft) => {
    const prompt = normal(draft.prompt);
    return prompt && message.includes(prompt);
  });
}

function renderAssistantTagChanges(article, changes) {
  if (!changes) return;
  const host = document.createElement('div'); host.className = 'assistant-tag-changes';
  host.title = 'Compared with the prompt used for this reply.';
  for (const [kind, label, prefix] of [['added', 'Added', '+'], ['removed', 'Removed', '−']]) {
    const tags = changes[kind] || [];
    if (!tags.length) continue;
    const row = document.createElement('p'); row.className = `assistant-tags-${kind}`;
    const name = document.createElement('strong'); name.textContent = `${label}:`; row.append(name);
    for (const tag of tags) {
      const item = document.createElement('span'); item.className = 'assistant-tag-change';
      item.textContent = `${prefix} ${tag}`; row.append(item);
    }
    host.append(row);
  }
  if (!host.childElementCount) {
    const note = document.createElement('p'); note.className = 'hint'; note.textContent = 'No tags added or removed.';
    host.append(note);
  }
  article.append(host);
}

function renderAssistantAnswer(host, body, interactive) {
  const speaker = document.createElement('strong'); speaker.className = 'assistant-speaker';
  speaker.textContent = 'Assistant'; host.append(speaker);
  const message = document.createElement('p');
  message.className = 'assistant-message';
  message.textContent = body.message;
  const repeatedMessage = assistantMessageRepeatsDraft(body);
  if (body.message?.trim() && !repeatedMessage) host.append(message);
  for (const warning of body.warnings || []) {
    const note = document.createElement('p');
    note.className = 'warn'; note.textContent = warning; host.append(note);
  }
  for (const [index, draft] of (body.drafts || []).entries()) {
    const article = document.createElement('article');
    article.className = 'assistant-draft';
    const head = document.createElement('div'); head.className = 'assistant-draft-head';
    const title = document.createElement('h3'); title.textContent = draft.title || `Draft ${index + 1}`;
    const prompt = document.createElement('p'); prompt.className = 'assistant-draft-text'; prompt.textContent = draft.prompt;
    head.append(title);
    if (interactive) {
      const number = document.createElement('span'); number.className = 'assistant-draft-number';
      number.textContent = String(index + 1); title.prepend(number, ' ');
      const use = document.createElement('button'); use.type = 'button'; use.className = 'ghost small';
      use.textContent = 'Use prompt'; use.dataset.assistantDraft = String(index);
      use.setAttribute('aria-label', `Use prompt ${index + 1}`);
      use.setAttribute('aria-keyshortcuts', `${index + 1} Alt+${index + 1}`);
      use.onclick = () => applyAssistantDraft(index); head.append(use);
    }
    article.append(head, prompt);
    renderAssistantTagChanges(article, body.draft_changes?.[index]);
    if (draft.notes?.length) {
      const notes = document.createElement('div'); notes.className = 'assistant-draft-notes';
      for (const words of draft.notes) {
        const note = document.createElement('p'); note.className = 'hint'; note.textContent = words; notes.append(note);
      }
      article.append(notes);
    }
    host.append(article);
  }
  if (repeatedMessage) {
    const note = document.createElement('details'); note.className = 'assistant-response-note';
    const summary = document.createElement('summary'); summary.textContent = 'Assistant note';
    note.append(summary, message); host.append(note);
  }
  if (body.tags?.length) {
    const list = document.createElement('div'); list.className = 'assistant-tags';
    const note = document.createElement('p'); note.className = 'hint';
    note.textContent = 'Only names verified by the tag index can be inserted. Meanings below are model suggestions.';
    list.append(note);
    for (const [index, tag] of body.tags.entries()) {
      const row = document.createElement('div'); row.className = 'assistant-tag';
      const label = document.createElement(interactive ? 'label' : 'div'); label.className = 'check';
      const tick = document.createElement('input'); tick.type = 'checkbox'; tick.dataset.assistantTag = String(index);
      tick.checked = assistantSelectedTags.has(index);
      const name = tag.status === 'alias' ? (tag.canonical || tag.text) : tag.text;
      const words = document.createElement('span'); words.textContent = name;
      if (interactive) label.append(tick);
      label.append(words); row.append(label);
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
    if (interactive) {
      const insert = document.createElement('button'); insert.type = 'button'; insert.id = 'assistant-insert-tags';
      insert.className = 'ghost small'; insert.textContent = 'Insert selected tags'; insert.onclick = insertAssistantTags;
      list.append(insert);
    }
    host.append(list);
  }
  const footer = document.createElement('p'); footer.className = 'hint';
  footer.textContent = `${body.model} · ${Math.round(body.elapsedSeconds ?? body.seconds)} s`;
  host.append(footer);
}

function resetAssistantConversation({ clearInstruction = true } = {}) {
  if (assistantRun) { assistantSay('Stop the current answer before starting a new conversation.', true); return; }
  ++assistantEpoch;
  assistantTurns = [];
  assistantResultData = null;
  assistantSnapshot = null;
  assistantPendingText = '';
  assistantSelectedTags.clear();
  if (clearInstruction) els.assistantInstruction.value = '';
  renderAssistantHistory(); renderAssistantResult();
  els.assistantInstruction.placeholder = assistantMode === 'improve' ? 'What would you like to change?' : assistantMode === 'tags' ? 'Describe the detail you need tags for…' : 'Describe your idea…';
  $('assistant-intro').querySelector('h3').textContent = assistantMode === 'improve' ? 'What would you like to change?' : assistantMode === 'tags' ? 'Which detail needs a tag?' : 'What would you like to create?';
  $('assistant-intro').querySelector('p').textContent = assistantMode === 'improve' ? 'Start with your prompt in the editor, then describe the change you want.' : assistantMode === 'tags' ? 'Describe a detail to find candidate tags checked against the Danbooru index.' : 'Describe an idea and develop it into a prompt with follow-up messages.';
  assistantSay('Describe your idea or the change you want.');
}

function assistantSidebarOpen() { return !els.assistantSidebar.hidden; }

function assistantScrollPosition() {
  const host = els.assistantSidebar.querySelector('.assistant-body');
  return { top: host.scrollTop, follow: host.scrollHeight - host.clientHeight - host.scrollTop <= 48 };
}

function scrollAssistantConversation(position = { follow: true, top: 0 }) {
  if (!assistantSidebarOpen()) return;
  const host = els.assistantSidebar.querySelector('.assistant-body');
  // Follow new messages at the end, but leave earlier messages where the
  // reader put them. Only this conversation moves, never the workspace.
  host.scrollTop = position.follow ? host.scrollHeight : position.top;
}

function openPromptAssistant(mode = null) {
  if (assistantSidebarOpen()) return;
  assistantOrigin = document.activeElement;
  if (mode && mode !== assistantMode && !assistantRun) {
    assistantMode = mode; els.assistantMode.value = mode;
    resetAssistantConversation({ clearInstruction: false });
  }
  syncPromptAssistant();
  els.assistantSidebar.hidden = false;
  document.body.classList.add('assistant-open');
  els.assistantToggle.setAttribute('aria-expanded', 'true');
  void refreshAssistantModels();
  syncBackLayers();
  if (!isPhone()) els.assistantInstruction.focus({ preventScroll: true });
  else els.assistantClose.focus({ preventScroll: true });
}

function closePromptAssistant() {
  if (!assistantSidebarOpen()) return;
  els.assistantSidebar.hidden = true;
  ++assistantModelsEpoch;
  document.body.classList.remove('assistant-open');
  els.assistantToggle.setAttribute('aria-expanded', 'false');
  if (assistantRun) void stopPromptAssistant();
  assistantPendingText = '';
  syncBackLayers();
  const origin = assistantOrigin; assistantOrigin = null;
  (origin?.isConnected && isShown(origin) ? origin : els.assistantToggle).focus({ preventScroll: true });
}

function assistantClock() {
  els.assistantClock.textContent = assistantRun ? ` · ${Math.floor((Date.now() - assistantRun.started) / 1000)} s` : '';
}

async function askPromptAssistant() {
  const why = assistantAskReason();
  if (why) { assistantSay(why, true); if (serverOffline) void refreshStatus(); return; }
  const source = assistantSource();
  const draft = els.assistantInstruction.value;
  const instruction = draft.trim();
  const instructionVersion = assistantInstructionVersion;
  let answered = false;
  const run = { epoch: ++assistantEpoch, ticket: expandTicket(), mode: assistantMode, model: assistantModel,
    stopped: false, stopping: false, started: Date.now() };
  const payload = {
    ticket: run.ticket, mode: run.mode, assistant_model: run.model, instruction, prompt: source.prompt,
    negative_prompt: source.negative || null,
    history: assistantTurns.map(({ role, content }) => ({ role, content })),
  };
  assistantRun = run;
  assistantSnapshot = source;
  assistantPendingText = instruction;
  els.assistantInstruction.value = '';
  const selected = assistantModelStatus();
  const download = !selected.ready && selected.fetch.length;
  assistantSay(download ? `First use downloads ${selected.fetch.join(' and ')} (about ${humanBytes(selected.fetch_bytes)}). Then the local model writes an answer.` : 'Asking the local model…');
  clearInterval(assistantClockTimer); assistantClock();
  assistantClockTimer = setInterval(assistantClock, 1000);
  syncPromptExpansion();
  scrollAssistantConversation();
  void refreshStatus();
  try {
    const body = await api('/api/prompt/assist', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (run.model === 'default') promptCritiqueReady = true;
    const prepared = assistantModels.find((model) => model.id === run.model);
    if (prepared) { prepared.ready = true; prepared.fetch = []; prepared.fetch_bytes = 0; }
    if (run.epoch !== assistantEpoch || run.stopped || !assistantSidebarOpen()) return;
    if (body.model_id !== source.model) {
      assistantSay('The checkpoint changed while this answer was prepared. Start a new conversation.', true); return;
    }
    const position = assistantScrollPosition();
    answered = true;
    assistantResultData = { ...body, elapsedSeconds: (Date.now() - run.started) / 1000 };
    assistantSelectedTags.clear();
    assistantTurns.push({ role: 'user', content: instruction },
      { role: 'assistant', content: assistantHistoryText(body), display: body.message, answer: assistantResultData });
    assistantPendingText = '';
    els.assistantInstruction.placeholder = 'Refine the answer, for example: keep the pose and make the lighting softer';
    renderAssistantHistory(); renderAssistantResult();
    scrollAssistantConversation(position);
    assistantSay(assistantSourceChanged()
      ? 'The source text changed while the model answered. Nothing can be applied; start a new conversation for the current text.'
      : 'Ready. Choose a draft or send a follow-up.', assistantSourceChanged());
  } catch (err) {
    if (run.epoch !== assistantEpoch || run.stopped || !assistantSidebarOpen()) return;
    assistantSay(err.offline ? 'The server did not answer. The main prompt is unchanged.' : `The assistant did not answer: ${err.message}`, true);
  } finally {
    if (assistantRun === run) {
      const position = assistantScrollPosition();
      assistantRun = null;
      assistantPendingText = '';
      // Failed or stopped sends remain retryable, without replacing anything
      // typed (even if subsequently erased) while the answer was pending.
      if (!answered && assistantInstructionVersion === instructionVersion && !els.assistantInstruction.value) {
        els.assistantInstruction.value = draft;
      }
      clearInterval(assistantClockTimer); assistantClockTimer = null; assistantClock();
      if (run.stopped && run.epoch === assistantEpoch) assistantSay('Answer stopped. The main prompt is unchanged.');
      syncPromptExpansion();
      scrollAssistantConversation(position);
      void refreshStatus();
      if (assistantSidebarOpen()) void refreshAssistantModels();
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
  const applyControl = document.activeElement;
  rewriteBox(els.prompt, text);
  // Our own explicit application becomes the next conversational source.
  // An external write or native Undo still invalidates it by exact text.
  assistantSnapshot = assistantSource();
  promptReplaced(); syncTagSelection(); scheduleTagValidation(0); schedulePreview(); persistPanel();
  const editorVisible = !isPhone() && els.prompt.getBoundingClientRect().right <= els.assistantSidebar.getBoundingClientRect().left;
  if (editorVisible) {
    els.prompt.focus({ preventScroll: true });
    els.prompt.setSelectionRange(0, 0);
  } else if (applyControl?.isConnected && els.assistantSidebar.contains(applyControl) && isShown(applyControl)) {
    // rewriteBox uses the native editor's undo path, which focuses it. A phone
    // covers that editor: keep the visible control and software keyboard state.
    applyControl.focus({ preventScroll: true });
  }
  assistantSay('Prompt updated. Keep refining it here, or undo in the editor.');
  syncPromptAssistant();
  offerPromptUndo(message, () => {
    rewriteBox(els.prompt, before); promptReplaced(); syncTagSelection(); scheduleTagValidation(0); schedulePreview(); persistPanel();
  });
}

function applyAssistantDraft(index) {
  const draft = assistantResultData?.drafts?.[index];
  if (draft) applyAssistantText(draft.prompt, 'A local assistant draft replaced your prompt.');
}

function assistantDraftKey(event) {
  if (keyboardEventHeld(event) || event.repeat || event.shiftKey || event.ctrlKey || event.metaKey
      || !assistantSidebarOpen() || !els.assistantSidebar.contains(event.target)
      || document.querySelector('dialog[open]') || !els.modelPicker.hidden
      || !els.contextMenu.hidden || !els.examplesMenu.hidden || !els.tagsMenu.hidden
      || !els.tagFinderMenu.hidden || !els.pickedMenu.hidden
      || (keyboardEditingTarget(event.target) && !event.altKey)) return false;
  const index = ['1', '2', '3'].indexOf(event.key);
  if (index < 0) return false;
  const use = els.assistantResult.querySelector(`[data-assistant-draft="${index}"]`);
  if (!use) return false;
  event.preventDefault();
  assistantDraftPressed = event.code || event.key;
  // The ordinary Use path owns stale-source, pending/offline holds and Undo.
  use.click();
  return true;
}

function assistantDraftRepeat(event) {
  if (!assistantDraftPressed || (event.code || event.key) !== assistantDraftPressed) return false;
  if (!event.repeat) { assistantDraftPressed = null; return false; }
  // Use focuses the main prompt. Its held number must not start typing there.
  event.preventDefault();
  return true;
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
  assistantPendingText = '';
  renderAssistantHistory(); renderAssistantResult();
  if (assistantSidebarOpen()) assistantSay('The checkpoint changed. This conversation was cleared; your instruction and the main prompt are unchanged.', true);
}

function initPromptAssistant() {
  try {
    const saved = localStorage.getItem('auradream.assistant-model');
    if ([...els.assistantModel.options].some((option) => option.value === saved)) assistantModel = saved;
  } catch { /* Storage may be unavailable; the choice still works for this page. */ }
  els.assistantModel.value = assistantModel;
  document.addEventListener('keyup', (event) => {
    if ((event.code || event.key) === assistantDraftPressed) assistantDraftPressed = null;
  }, true);
  window.addEventListener('blur', () => { assistantDraftPressed = null; });
  els.assistantToggle.onclick = () => {
    if (assistantSidebarOpen()) closePromptAssistant(); else openPromptAssistant();
  };
  els.assistantClose.onclick = closePromptAssistant;
  els.assistantAsk.onclick = () => { void askPromptAssistant(); };
  els.assistantStop.onclick = () => { void stopPromptAssistant(); };
  els.assistantReset.onclick = () => resetAssistantConversation();
  els.assistantMode.onchange = () => {
    if (assistantRun) { els.assistantMode.value = assistantMode; assistantSay('Stop the current answer before changing the task.', true); return; }
    assistantMode = els.assistantMode.value;
    resetAssistantConversation({ clearInstruction: false });
  };
  els.assistantModel.onchange = () => {
    if (assistantRun) {
      els.assistantModel.value = assistantModel;
      assistantSay('Stop the current answer before changing the assistant model.', true); return;
    }
    assistantModel = els.assistantModel.value;
    try { localStorage.setItem('auradream.assistant-model', assistantModel); } catch { /* Page-only choice. */ }
    syncPromptAssistant();
    assistantSay(`Your next message uses ${assistantModelStatus().label}. This conversation is kept.`);
  };
  for (const [control, reason] of [[els.assistantMode, 'task'], [els.assistantModel, 'assistant model']]) {
    const held = (event) => {
      if (!assistantRun || (event.type === 'keydown' && ![' ', 'Enter', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key))) return;
      event.preventDefault();
      assistantSay(`Stop the current answer before changing the ${reason}.`, true);
    };
    control.addEventListener('pointerdown', held);
    control.addEventListener('keydown', held);
  }
  els.assistantInstruction.addEventListener('input', () => {
    ++assistantInstructionVersion;
    syncPromptAssistant();
  });
  els.prompt.addEventListener('input', syncPromptAssistant);
  els.negative.addEventListener('input', syncPromptAssistant);
  els.assistantInstruction.addEventListener('keydown', (event) => {
    if (keyboardEventHeld(event) || !keyboardSubmit(event)) return;
    event.preventDefault(); event.stopPropagation();
    if (!event.repeat) void askPromptAssistant();
  });
  syncPromptAssistant();
}
