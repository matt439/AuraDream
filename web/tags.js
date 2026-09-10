// Krea Local's page script: the prompt tag helper, autocomplete and validation chips. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// --------------------------------------------------------- prompt tag helper

function promptTagParts() {
  return els.prompt.value.split(',').map((part) => part.trim()).filter(Boolean);
}

function currentTagRange() {
  const text = els.prompt.value;
  const caret = els.prompt.selectionStart ?? text.length;
  let start = text.lastIndexOf(',', Math.max(0, caret - 1)) + 1;
  let end = text.indexOf(',', caret);
  if (end < 0) end = text.length;
  while (start < end && /\s/.test(text[start])) start += 1;
  while (end > start && /\s/.test(text[end - 1])) end -= 1;
  return { start, end, query: text.slice(start, end).trim() };
}

function hideTagSuggestions() {
  tagSuggestionEpoch += 1;  // an in-flight response must not reopen a dismissed list
  tagSuggestions = [];
  tagSuggestionIndex = -1;
  els.tagSuggestions.hidden = true;
  els.tagSuggestions.replaceChildren();
  els.prompt.setAttribute('aria-expanded', 'false');
  els.prompt.removeAttribute('aria-activedescendant');
}

function humanCount(value) {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}m`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(value >= 1e5 ? 0 : 1)}k`;
  return String(value);
}

function selectTagSuggestion(index) {
  if (!tagSuggestions.length) return;
  tagSuggestionIndex = (index + tagSuggestions.length) % tagSuggestions.length;
  [...els.tagSuggestions.children].forEach((item, i) => {
    const active = i === tagSuggestionIndex;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
    if (active) {
      els.prompt.setAttribute('aria-activedescendant', item.id);
      item.scrollIntoView({ block: 'nearest' });
    }
  });
}

function applyTagSuggestion(suggestion) {
  const { start, end } = currentTagRange();
  const before = els.prompt.value.slice(0, start);
  const after = els.prompt.value.slice(end);
  els.prompt.value = before + suggestion.text + after;
  const caret = start + suggestion.text.length;
  els.prompt.setSelectionRange(caret, caret);
  clearPromptTranslation();
  syncTagSelection();
  schedulePreview();
  persistPanel();
  scheduleTagValidation(0);
  hideTagSuggestions();
}

function renderTagSuggestions(items) {
  tagSuggestions = items;
  tagSuggestionIndex = -1;
  els.tagSuggestions.replaceChildren();
  for (const [index, suggestion] of items.entries()) {
    const item = document.createElement('li');
    item.id = `tag-suggestion-${index}`;
    item.setAttribute('role', 'option');
    const kind = document.createElement('span');
    kind.className = 'tag-kind';
    kind.textContent = suggestion.category;
    const name = document.createElement('span');
    name.className = 'tag-name';
    name.textContent = suggestion.text;
    const count = document.createElement('span');
    count.className = 'tag-count';
    count.textContent = suggestion.post_count == null ? '' : humanCount(suggestion.post_count);
    item.append(kind, name, count);
    item.onmousedown = (event) => event.preventDefault();
    item.onclick = () => applyTagSuggestion(suggestion);
    els.tagSuggestions.appendChild(item);
  }
  els.tagSuggestions.hidden = items.length === 0;
  els.prompt.setAttribute('aria-expanded', String(items.length > 0));
  if (items.length) selectTagSuggestion(0);
}

async function requestTagSuggestions() {
  const epoch = ++tagSuggestionEpoch;
  if (!tagVocabularyEnabled || document.activeElement !== els.prompt) {
    hideTagSuggestions();
    return;
  }
  const { query } = currentTagRange();
  if (!query) {
    hideTagSuggestions();
    return;
  }
  try {
    const body = await api(`/api/tags/suggest?q=${encodeURIComponent(query)}`);
    if (epoch !== tagSuggestionEpoch) return;
    renderTagSuggestions(body.suggestions || []);
  } catch {
    if (epoch === tagSuggestionEpoch) hideTagSuggestions();
  }
}

function scheduleTagSuggestions(delay = 160) {
  clearTimeout(tagSuggestionTimer);
  tagSuggestionTimer = setTimeout(requestTagSuggestions, delay);
}

function deletePromptTag(index) {
  const tags = promptTagParts();
  if (index < 0 || index >= tags.length) return;
  tags.splice(index, 1);
  tagInsertAt = null;
  writePromptTags(tags);
}

function movePromptTag(from, to) {
  const tags = promptTagParts();
  if (from < 0 || from >= tags.length || to < 0 || to >= tags.length || from === to) return;
  const [tag] = tags.splice(from, 1);
  tags.splice(to, 0, tag);
  tagInsertAt = null;
  writePromptTags(tags);
}

function dropPromptTag(from, insertion) {
  const tags = promptTagParts();
  if (from < 0 || from >= tags.length) return;
  const [tag] = tags.splice(from, 1);
  const target = Math.max(0, Math.min(tags.length, insertion > from ? insertion - 1 : insertion));
  if (target === from) return;
  tags.splice(target, 0, tag);
  tagInsertAt = null;
  writePromptTags(tags);
}

function clearTagDropMarkers() {
  for (const chip of els.tagValidation.querySelectorAll('.tag-check')) {
    chip.classList.remove('drop-before', 'drop-after');
  }
}

function closeTagInsert() {
  tagInsertAt = null;
  tagInsertSuggestionEpoch += 1;
  clearTimeout(tagInsertSuggestionTimer);
  renderTagValidation(
    lastTagValidationResults, lastTagValidationAvailable, lastTagValidationDetail,
  );
}

function commitInsertedTag(text) {
  const tag = text.trim();
  if (!tag || tag.includes(',') || tagInsertAt === null) return;
  const tags = promptTagParts();
  tags.splice(Math.min(tagInsertAt, tags.length), 0, tag);
  tagInsertAt = null;
  writePromptTags(tags);
}

function selectInsertSuggestion(menu, index) {
  const suggestions = menu._suggestions || [];
  if (!suggestions.length) return;
  menu._index = (index + suggestions.length) % suggestions.length;
  [...menu.children].forEach((item, i) => {
    const active = i === menu._index;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
    if (active) item.scrollIntoView({ block: 'nearest' });
  });
}

function renderInsertSuggestions(input, menu, suggestions) {
  menu._suggestions = suggestions;
  menu._index = -1;
  menu.replaceChildren();
  for (const suggestion of suggestions) {
    const item = document.createElement('li');
    item.setAttribute('role', 'option');
    const kind = document.createElement('span');
    kind.className = 'tag-kind';
    kind.textContent = suggestion.category;
    const name = document.createElement('span');
    name.className = 'tag-name';
    name.textContent = suggestion.text;
    const count = document.createElement('span');
    count.className = 'tag-count';
    count.textContent = suggestion.post_count == null ? '' : humanCount(suggestion.post_count);
    item.append(kind, name, count);
    item.onmousedown = (event) => event.preventDefault();
    item.onclick = () => commitInsertedTag(suggestion.text);
    menu.appendChild(item);
  }
  menu.hidden = suggestions.length === 0;
  input.setAttribute('aria-expanded', String(suggestions.length > 0));
  const editor = menu.parentElement.getBoundingClientRect();
  menu.classList.toggle('end', editor.left + 280 > window.innerWidth - 12);
  if (suggestions.length) selectInsertSuggestion(menu, 0);
}

function scheduleInsertSuggestions(input, menu) {
  clearTimeout(tagInsertSuggestionTimer);
  const epoch = ++tagInsertSuggestionEpoch;
  const query = input.value.trim();
  renderInsertSuggestions(input, menu, []);
  if (!query) return;
  tagInsertSuggestionTimer = setTimeout(async () => {
    try {
      const body = await api(`/api/tags/suggest?q=${encodeURIComponent(query)}`);
      if (epoch !== tagInsertSuggestionEpoch || !input.isConnected) return;
      renderInsertSuggestions(input, menu, body.suggestions || []);
    } catch {
      if (epoch === tagInsertSuggestionEpoch) renderInsertSuggestions(input, menu, []);
    }
  }, 160);
}

function buildTagInsertEditor() {
  const editor = document.createElement('span');
  editor.className = 'tag-insert-editor';
  const input = document.createElement('input');
  input.type = 'text';
  input.maxLength = 200;
  input.placeholder = 'insert tag';
  input.setAttribute('aria-label', 'New tag');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-expanded', 'false');
  const menu = document.createElement('ul');
  menu.className = 'tag-insert-suggestions';
  menu.id = 'tag-insert-suggestions';
  menu.setAttribute('role', 'listbox');
  menu.hidden = true;
  input.setAttribute('aria-controls', menu.id);
  const accept = document.createElement('button');
  accept.type = 'button';
  accept.textContent = '+';
  accept.disabled = true;
  accept.title = 'Insert tag';
  accept.setAttribute('aria-label', 'Insert tag');
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = '×';
  cancel.title = 'Cancel insertion';
  cancel.setAttribute('aria-label', 'Cancel insertion');
  input.oninput = () => {
    accept.disabled = !input.value.trim() || input.value.includes(',');
    scheduleInsertSuggestions(input, menu);
  };
  input.onkeydown = (event) => {
    if (event.key === 'ArrowDown' && !menu.hidden) {
      event.preventDefault();
      selectInsertSuggestion(menu, menu._index + 1);
    } else if (event.key === 'ArrowUp' && !menu.hidden) {
      event.preventDefault();
      selectInsertSuggestion(menu, menu._index - 1);
    } else if ((event.key === 'Enter' || event.key === 'Tab') && !menu.hidden) {
      event.preventDefault();
      commitInsertedTag(menu._suggestions[menu._index].text);
    } else if (event.key === 'Enter' && !accept.disabled) {
      event.preventDefault();
      commitInsertedTag(input.value);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      closeTagInsert();
    }
  };
  accept.onclick = () => commitInsertedTag(input.value);
  cancel.onclick = closeTagInsert;
  editor.append(input, accept, cancel, menu);
  setTimeout(() => input.focus(), 0);
  return editor;
}

function openTagInsert(index) {
  clearTimeout(tagValidationTimer);
  tagValidationEpoch += 1;
  tagInsertAt = index;
  renderTagValidation(
    lastTagValidationResults, lastTagValidationAvailable, lastTagValidationDetail,
  );
}

function tagInsertionIndexAtPoint(x, y) {
  const entries = [...els.tagValidation.querySelectorAll('.tag-check')].map((chip) => ({
    index: Number(chip.dataset.index),
    rect: chip.getBoundingClientRect(),
  }));
  if (!entries.length) return 0;

  const rows = [];
  for (const entry of entries) {
    let row = rows.find((candidate) => Math.abs(candidate.top - entry.rect.top) < 3);
    if (!row) {
      row = { top: entry.rect.top, bottom: entry.rect.bottom, entries: [] };
      rows.push(row);
    }
    row.bottom = Math.max(row.bottom, entry.rect.bottom);
    row.entries.push(entry);
  }
  const distance = (row) => y < row.top ? row.top - y : y > row.bottom ? y - row.bottom : 0;
  const row = rows.sort((a, b) => distance(a) - distance(b))[0];
  row.entries.sort((a, b) => a.rect.left - b.rect.left);
  for (const entry of row.entries) {
    if (x < entry.rect.left + entry.rect.width / 2) return entry.index;
  }
  return row.entries[row.entries.length - 1].index + 1;
}

function renderTagValidation(results, available = true, detail = '') {
  lastTagValidationResults = results.map((result) => ({ ...result }));
  lastTagValidationAvailable = available;
  lastTagValidationDetail = detail;
  if (tagInsertAt !== null && tagInsertAt > results.length) tagInsertAt = results.length;
  els.tagValidation.replaceChildren();
  for (const [index, result] of results.entries()) {
    if (tagInsertAt === index) els.tagValidation.appendChild(buildTagInsertEditor());
    const chip = document.createElement('span');
    chip.className = 'tag-check';
    chip.draggable = true;
    chip.tabIndex = 0;
    chip.dataset.index = String(index);
    const label = document.createElement('span');
    label.className = 'tag-check-label';
    const mark = document.createElement('span');
    mark.className = 'tag-check-mark';
    const text = document.createElement('span');
    text.className = 'tag-check-text';
    text.textContent = result.tag;
    if (result.valid === true) {
      chip.classList.add('valid');
      mark.textContent = '✓';
      chip.title = result.source === 'model'
        ? 'Animagine model-card tag'
        : `Danbooru ${result.category} tag · ${humanCount(result.post_count)} posts` +
          (result.canonical !== result.tag ? ` · canonical: ${result.canonical}` : '');
    } else if (result.valid === false) {
      chip.classList.add('unknown');
      mark.textContent = '?';
      chip.title = `${result.reason}; generation is still allowed`;
    } else {
      chip.classList.add('unchecked');
      mark.textContent = '…';
      chip.title = detail || result.reason || 'Danbooru lookup unavailable';
    }
    label.append(mark, text);
    chip.title += ' · drag to reorder · right-click left/right half to insert · Delete removes';
    chip.setAttribute('aria-label', `${result.tag}. Drag to reorder; Delete removes it.`);
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'tag-remove';
    remove.textContent = '×';
    remove.draggable = false;
    remove.title = `Delete ${result.tag}`;
    remove.setAttribute('aria-label', `Delete ${result.tag}`);
    remove.onclick = (event) => { event.stopPropagation(); deletePromptTag(index); };
    chip.append(label, remove);
    chip.ondragstart = (event) => {
      draggedTagIndex = index;
      chip.classList.add('dragging');
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', String(index));
      }
    };
    chip.ondragover = (event) => {
      if (draggedTagIndex === null) return;
      event.preventDefault();
      clearTagDropMarkers();
      const after = event.clientX >= chip.getBoundingClientRect().left + chip.offsetWidth / 2;
      chip.classList.add(after ? 'drop-after' : 'drop-before');
    };
    chip.ondrop = (event) => {
      if (draggedTagIndex === null) return;
      event.preventDefault();
      const after = event.clientX >= chip.getBoundingClientRect().left + chip.offsetWidth / 2;
      const insertion = index + (after ? 1 : 0);
      const from = draggedTagIndex;
      draggedTagIndex = null;
      clearTagDropMarkers();
      dropPromptTag(from, insertion);
    };
    chip.ondragend = () => {
      draggedTagIndex = null;
      chip.classList.remove('dragging');
      clearTagDropMarkers();
    };
    chip.oncontextmenu = (event) => {
      event.preventDefault();
      event.stopPropagation();
      openTagInsert(index + 1);
    };
    chip.onkeydown = (event) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        deletePromptTag(index);
      } else if (event.altKey && event.key === 'ArrowLeft') {
        event.preventDefault();
        movePromptTag(index, index - 1);
      } else if (event.altKey && event.key === 'ArrowRight') {
        event.preventDefault();
        movePromptTag(index, index + 1);
      }
    };
    els.tagValidation.appendChild(chip);
  }
  if (tagInsertAt === results.length) els.tagValidation.appendChild(buildTagInsertEditor());
  els.tagValidation.title = available
    ? 'Each comma-separated item checked against Danbooru or Animagine special tags'
    : `Lookup unavailable; unchecked tags are not treated as invalid${detail ? `: ${detail}` : ''}`;
  els.tagValidation.hidden = results.length === 0;
}

async function validatePromptTags(tags, epoch) {
  try {
    const body = await api('/api/tags/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags }),
    });
    if (epoch !== tagValidationEpoch) return;
    renderTagValidation(body.results, body.available, body.detail || '');
  } catch (error) {
    if (epoch !== tagValidationEpoch) return;
    renderTagValidation(tags.map((tag) => ({ tag, valid: null })), false, error.message);
  }
}

function scheduleTagValidation(delay = 320) {
  clearTimeout(tagValidationTimer);
  const epoch = ++tagValidationEpoch;
  if (!tagVocabularyEnabled) {
    els.tagValidation.hidden = true;
    els.tagValidation.replaceChildren();
    return;
  }
  const tags = promptTagParts();
  if (!tags.length) {
    els.tagValidation.hidden = true;
    els.tagValidation.replaceChildren();
    return;
  }
  renderTagValidation(tags.map((tag) => ({ tag, valid: null })), false, 'Checking Danbooru…');
  tagValidationTimer = setTimeout(() => validatePromptTags(tags, epoch), delay);
}

function syncTagVocabulary() {
  tagVocabularyEnabled = tagHelper?.vocabulary === 'danbooru';
  els.prompt.spellcheck = !tagVocabularyEnabled;
  if (!tagVocabularyEnabled) hideTagSuggestions();
  scheduleTagValidation(0);
}

function placeTagValidation() {
  const host = window.matchMedia('(min-width: 901px)').matches
    ? els.tagValidationWide
    : els.tagValidationHome;
  if (els.tagValidation.parentElement !== host) host.appendChild(els.tagValidation);
}

function tagCursorIndex() {
  const before = els.prompt.value.slice(0, els.prompt.selectionStart ?? els.prompt.value.length);
  return before.trim() ? before.split(',').length : 0;
}

function writePromptTags(parts, caret = null) {
  els.prompt.value = parts.join(', ');
  clearPromptTranslation();
  const end = caret == null ? els.prompt.value.length : Math.min(caret, els.prompt.value.length);
  els.prompt.setSelectionRange(end, end);
  syncTagSelection();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
}

// Buttons are toggles. Exclusive groups replace an existing person-count or
// rating tag in place; otherwise insertion follows the textarea caret. Nothing
// silently reorders free-form character, series, pose or scene tags.
function togglePromptTag(tag, group) {
  let parts = promptTagParts();
  const wanted = tag.toLowerCase();
  const active = parts.findIndex((part) => part.toLowerCase() === wanted);
  if (active >= 0) {
    parts.splice(active, 1);
    writePromptTags(parts);
    return;
  }

  let insertAt = Math.min(tagCursorIndex(), parts.length);
  if (group.exclusive) {
    const choices = new Set(group.tags.map((choice) => choice.toLowerCase()));
    const old = parts.findIndex((part) => choices.has(part.toLowerCase()));
    parts = parts.filter((part) => !choices.has(part.toLowerCase()));
    if (old >= 0) insertAt = Math.min(old, parts.length);
  }
  parts.splice(insertAt, 0, tag);
  const caret = parts.slice(0, insertAt + 1).join(', ').length;
  writePromptTags(parts, caret);
}

function setYearTag(year) {
  let parts = promptTagParts();
  let insertAt = parts.findIndex((part) => /^year \d{4}$/i.test(part));
  if (insertAt < 0) insertAt = Math.min(tagCursorIndex(), parts.length);
  parts = parts.filter((part) => !/^year \d{4}$/i.test(part));
  insertAt = Math.min(insertAt, parts.length);
  parts.splice(insertAt, 0, `year ${year}`);
  writePromptTags(parts, parts.slice(0, insertAt + 1).join(', ').length);
}

function finishWithTags(suffix) {
  const ending = new Set(suffix.map((tag) => tag.toLowerCase()));
  const parts = promptTagParts().filter((part) => !ending.has(part.toLowerCase()));
  writePromptTags([...parts, ...suffix]);
}

function syncTagSelection() {
  if (!els.tagsMenu || els.tagsMenu.hidden) return;
  const active = new Set(promptTagParts().map((tag) => tag.toLowerCase()));
  for (const button of els.tagsMenu.querySelectorAll('[data-tag]')) {
    const on = active.has(button.dataset.tag.toLowerCase());
    button.classList.toggle('active', on);
    button.setAttribute('aria-pressed', String(on));
  }
}

function buildTagHelper() {
  els.tagsMenu.replaceChildren();
  if (!tagHelper) return;

  const order = document.createElement('p');
  order.className = 'tag-order';
  order.textContent = `Recommended order: ${tagHelper.order.join(' → ')}`;
  els.tagsMenu.appendChild(order);

  for (const group of tagHelper.groups) {
    const section = document.createElement('section');
    section.className = 'tag-group';
    const heading = document.createElement('strong');
    heading.textContent = group.label;
    const options = document.createElement('div');
    options.className = 'tag-options';
    for (const tag of group.tags) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'tag-option';
      button.textContent = tag;
      button.dataset.tag = tag;
      button.onclick = () => togglePromptTag(tag, group);
      options.appendChild(button);
    }
    section.append(heading, options);
    els.tagsMenu.appendChild(section);
  }

  if (tagHelper.year_tag) {
    const section = document.createElement('section');
    section.className = 'tag-group';
    const heading = document.createElement('strong');
    heading.textContent = 'Year';
    const row = document.createElement('div');
    row.className = 'tag-year';
    const input = document.createElement('input');
    input.type = 'text';
    input.inputMode = 'numeric';
    input.maxLength = 4;
    input.placeholder = '2023';
    input.setAttribute('aria-label', 'Four digit year tag');
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'ghost tiny';
    add.textContent = 'Insert year';
    add.disabled = true;
    input.oninput = () => { add.disabled = !/^\d{4}$/.test(input.value); };
    add.onclick = () => setYearTag(input.value);
    row.append(input, add);
    section.append(heading, row);
    els.tagsMenu.appendChild(section);
  }

  const actions = document.createElement('div');
  actions.className = 'tag-actions';
  const finish = document.createElement('button');
  finish.type = 'button';
  finish.className = 'ghost tiny';
  finish.dataset.action = 'recommended-suffix';
  finish.textContent = `Finish with ${tagHelper.recommended_suffix.join(', ')}`;
  finish.onclick = () => finishWithTags(tagHelper.recommended_suffix);
  actions.appendChild(finish);
  if (tagHelper.negative_prompt) {
    const negative = document.createElement('button');
    negative.type = 'button';
    negative.className = 'ghost tiny';
    negative.dataset.action = 'recommended-negative';
    negative.textContent = 'Use the recommended negative prompt';
    negative.onclick = () => {
      els.negative.value = tagHelper.negative_prompt;
      schedulePreview();
      persistPanel();
    };
    actions.appendChild(negative);
  }
  els.tagsMenu.appendChild(actions);

  const note = document.createElement('p');
  note.className = 'tag-note';
  note.textContent = 'Tag buttons insert at the prompt caret; select an active tag to remove it.';
  els.tagsMenu.appendChild(note);
  syncTagSelection();
}

function showTagHelper(open) {
  if (open && !els.examplesMenu.hidden) showExamples(false);
  els.tagsBtn.setAttribute('aria-expanded', String(open));
  els.tagsMenu.hidden = !open;
  if (open) buildTagHelper();
}

function syncTagHelper() {
  const available = Boolean(tagHelper && Array.isArray(tagHelper.groups) && tagHelper.groups.length);
  els.tagHelper.hidden = !available;
  if (!available) showTagHelper(false);
  else if (!els.tagsMenu.hidden) buildTagHelper();
  syncTagVocabulary();
}
