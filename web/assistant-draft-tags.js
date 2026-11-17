// Read-only draft verdicts from the same endpoint as the prompt editor.
// Declarations only; app.js initializes this after the classic scripts load.
'use strict';

const assistantDraftTagCache = new WeakMap();
const assistantDraftTagRuns = new Set();
let assistantDraftTagEpoch = 0;

function assistantDraftTagProse(record) {
  const indexes = new Set();
  const parts = record.answer?.parts;
  const spans = promptTagSpans(record.prompt);
  let next = 0;
  if (!Array.isArray(parts) || !parts.every((part) => {
    const fits = ['tag', 'prose'].includes(part.kind) && part.first === next
      && Number.isInteger(part.last) && part.last >= part.first
      && part.start === spans[part.first]?.start && part.end === spans[part.last]?.end;
    next = part.last + 1;
    return fits;
  }) || next !== spans.length) return indexes;
  for (const part of parts) {
    if (part.kind !== 'prose') continue;
    for (let index = part.first; index <= part.last; ++index) indexes.add(index);
  }
  return indexes;
}

function assistantDraftTagVerdict(record, index) {
  const tag = record.tags[index];
  if (isGridVariable(tag)) return { tag, valid: null, code: 'grid_variable' };
  if (index >= TAG_LOOKUP_CAP) return { tag, valid: null, reason: `Only the first ${TAG_LOOKUP_CAP} comma items are checked.` };
  const result = record.answer?.results?.[index];
  return result?.tag === tag ? result : { tag, valid: null, reason: record.pending
    ? 'Checking Danbooru\u2026' : record.answer?.detail || 'This tag has not been checked.' };
}

function assistantDraftTagState(result) {
  return result.code === 'grid_variable' ? 'grid-variable'
    : result.valid === true ? 'valid'
    : result.valid === false ? result.code === 'alias' ? 'alias' : 'unknown' : 'unchecked';
}

function assistantDraftTagWords(result) {
  const state = assistantDraftTagState(result);
  return state === 'valid' ? result.source === 'model' ? 'Checkpoint guide tag' : `Known Danbooru ${result.category || ''} tag`.replace(/  /g, ' ')
    : state === 'alias' ? `Alias; use ${result.canonical || 'the canonical spelling'} for the indexed tag`
    : state === 'grid-variable' ? 'Grid placeholder, not a tag'
    : state === 'unknown' ? `Unknown: ${result.reason || 'not found in Danbooru'}`
    : `Unchecked: ${result.reason || 'Danbooru is unavailable'}`;
}

function paintAssistantDraftTags(view) {
  const { record, article, text, summary, results, note, retry } = view;
  const prose = assistantDraftTagProse(record);
  const spans = promptTagSpans(record.prompt);
  const counts = { valid: 0, unknown: 0, alias: 0, unchecked: 0, 'grid-variable': 0 };
  const annotated = document.createDocumentFragment();
  const rows = document.createDocumentFragment();
  let offset = 0;
  for (const [index, span] of spans.entries()) {
    annotated.append(document.createTextNode(record.prompt.slice(offset, span.start)));
    const literal = record.prompt.slice(span.start, span.end);
    if (prose.has(index)) {
      const natural = document.createElement('span'); natural.className = 'assistant-draft-prose';
      natural.textContent = literal; natural.title = 'Natural-language passage, not a validated Danbooru tag';
      annotated.append(natural);
      const row = document.createElement('p'); row.className = 'assistant-draft-tag-verdict prose';
      row.textContent = `${literal} \u2014 Natural-language text, not counted as a validated tag. Dictionary lookup: ${assistantDraftTagWords(assistantDraftTagVerdict(record, index))}`;
      rows.append(row);
    }
    else {
      const result = assistantDraftTagVerdict(record, index);
      const state = assistantDraftTagState(result);
      counts[state] += 1;
      const words = assistantDraftTagWords(result);
      const mark = document.createElement('span'); mark.className = `assistant-draft-tag ${state}`;
      mark.dataset.tagIndex = String(index);
      mark.dataset.category = result.source === 'model' ? 'guide' : result.category || '';
      mark.title = `${result.tag}: ${words}`;
      mark.textContent = literal;
      annotated.append(mark);
      const row = document.createElement('p'); row.className = `assistant-draft-tag-verdict ${state}`;
      row.textContent = `${result.tag} \u2014 ${words}`;
      rows.append(row);
    }
    offset = span.end;
  }
  annotated.append(document.createTextNode(record.prompt.slice(offset)));
  text.replaceChildren(annotated);
  results.replaceChildren(rows);
  const label = Object.entries(counts).filter(([, count]) => count > 0)
    .map(([kind, count]) => `${count} ${kind === 'valid' ? 'known' : kind === 'grid-variable' ? 'placeholders' : kind === 'alias' ? count === 1 ? 'alias' : 'aliases' : kind}`);
  if (prose.size) label.push(`${prose.size} prose ${prose.size === 1 ? 'item' : 'items'}`);
  summary.textContent = record.pending ? 'Tag validation \u00b7 checking\u2026'
    : `Tag validation${label.length ? ` \u00b7 ${label.join(', ')}` : ' \u00b7 prose only'}`;
  article.dataset.tagState = record.pending ? 'pending' : record.answer?.available ? 'checked' : 'unchecked';
  note.textContent = record.pending ? 'Checking the same comma items as the prompt editor. Your draft stays unchanged.'
    : !record.answer ? 'Not checked yet. Choose Recheck tags when the server is available.'
      : record.answer.available === false ? `${record.answer.transport ? 'The tag check did not finish' : 'Danbooru could not be reached'}; unchecked tags are not invalid.${record.answer.detail ? ` ${record.answer.detail}` : ''}`
      : 'Known names and checkpoint guide tags are marked in the draft. Wavy underlines flag unknown names; dashed underlines flag aliases. Your draft stays unchanged.';
  if (prose.size) note.textContent += ' Natural-language passages remain ordinary text.';
  if (record.tags.length > TAG_LOOKUP_CAP) note.textContent += ` Items after ${TAG_LOOKUP_CAP} remain unchecked.`;
  assistantHeld(retry, record.pending ? 'This draft is already being checked.'
    : record.model !== loadedModel || record.revision !== modelRevision ? 'The image checkpoint changed.'
    : serverOffline ? 'The server is not answering.' : '', 'Check this exact draft again without changing the editor');
}

function paintAssistantDraftTagViews(record) {
  for (const view of record.views) {
    if (!view.article.isConnected) record.views.delete(view);
    else paintAssistantDraftTags(view);
  }
}

async function checkAssistantDraftTags(record, { force = false } = {}) {
  if (record.pending || (!force && record.attempted) || !assistantSidebarOpen()
      || !tagVocabularyEnabled || record.model !== loadedModel || record.revision !== modelRevision || serverOffline) return;
  const request = { epoch: assistantDraftTagEpoch, controller: new AbortController() };
  record.request = request; record.pending = true; record.attempted = true;
  assistantDraftTagRuns.add(record);
  paintAssistantDraftTagViews(record);
  const current = () => record.request === request && request.epoch === assistantDraftTagEpoch
    && assistantSidebarOpen() && record.model === loadedModel && record.revision === modelRevision;
  try {
    const answer = await api('/api/tags/validate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: request.controller.signal,
      body: JSON.stringify({ tags: record.tags.slice(0, TAG_LOOKUP_CAP), prompt: record.prompt }),
    });
    if (!current()) return;
    if (!Array.isArray(answer.results) || answer.results.length !== Math.min(record.tags.length, TAG_LOOKUP_CAP)
        || answer.results.some((result, index) => result.tag !== record.tags[index])) {
      throw new Error('The tag checker returned a different draft. Recheck this draft.');
    }
    record.answer = answer;
  } catch (err) {
    if (!current()) return;
    record.answer = { available: false, results: [], parts: record.answer?.parts || null, transport: true, detail: err.message };
  } finally {
    if (record.request === request) {
      record.pending = false;
      assistantDraftTagRuns.delete(record);
      if (current()) paintAssistantDraftTagViews(record);
    }
  }
}

function renderAssistantDraftTags(article, body, index) {
  if (!tagVocabularyEnabled || body.model_id !== loadedModel) return;
  const draft = body.drafts?.[index];
  const tags = draft ? promptTagParts(draft.prompt) : [];
  if (!tags.length) return;
  let records = assistantDraftTagCache.get(body);
  if (!records) { records = []; assistantDraftTagCache.set(body, records); }
  let record = records[index];
  if (!record || record.prompt !== draft.prompt || record.model !== loadedModel || record.revision !== modelRevision) {
    record = { model: loadedModel, revision: modelRevision, prompt: draft.prompt, tags, views: new Set(),
      pending: false, attempted: false, answer: null, request: null };
    records[index] = record;
  }
  const menu = article.querySelector('.assistant-draft-tool-menu');
  if (!menu) return;
  const detail = document.createElement('details'); detail.className = 'assistant-draft-tag-validation';
  const summary = document.createElement('summary');
  const note = document.createElement('p'); note.className = 'hint';
  const results = document.createElement('div'); results.className = 'assistant-draft-tag-results';
  const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'ghost small';
  retry.textContent = 'Recheck tags'; retry.dataset.assistantAction = 'recheck-tags';
  detail.append(summary, note, results, retry); menu.append(detail);
  const view = { record, article, text: article.querySelector('.assistant-draft-text'), summary, results, note, retry };
  for (const old of record.views) if (!old.article.isConnected) record.views.delete(old);
  record.views.add(view);
  article.dataset.assistantTags = '';
  article.assistantTagView = view;
  retry.onclick = () => {
    if (retry.getAttribute('aria-disabled') === 'true') { note.textContent = retry.title; return; }
    void checkAssistantDraftTags(record, { force: true });
  };
  paintAssistantDraftTags(view);
  // The renderer appends the card after this function returns. Start once
  // attached, so pruning retired views cannot remove this new subscriber.
  queueMicrotask(() => { if (article.isConnected) void checkAssistantDraftTags(record); });
}

function resumeAssistantDraftTags() {
  for (const article of els.assistantSidebar.querySelectorAll('[data-assistant-tags]')) {
    const view = article.assistantTagView;
    if (!view) continue;
    paintAssistantDraftTags(view);
    void checkAssistantDraftTags(view.record);
  }
}

function retireAssistantDraftTags() {
  ++assistantDraftTagEpoch;
  for (const record of assistantDraftTagRuns) {
    record.request?.controller.abort();
    record.request = null; record.pending = false; record.attempted = !!record.answer;
  }
  assistantDraftTagRuns.clear();
}

function initAssistantDraftTags() { resumeAssistantDraftTags(); }
