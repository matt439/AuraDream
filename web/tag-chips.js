// AuraDream's page script: tag validation chips.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ---------------------------------------------------------------- the chips

// The tray's label, its title and its state: what the lookup said (checked,
// still checking, the index unreachable - amber, never red, since nothing is
// invalid for it), how many were unknown - not the item still being typed -
// the lookup's cap, and the count.
function syncTagTray(results, state, detail = '', typing = null) {
  const prose = new Set(promptDisplayParts(results).filter((part) => part.kind === 'prose')
    .flatMap((part) => Array.from({ length: part.last - part.first + 1 }, (_, offset) => part.first + offset)));
  const unknown = results.filter((result, index) => !prose.has(index) && result.valid === false && index !== typing).length;
  const name = prose.size ? 'Prompt tags and prose' : 'Prompt tags';
  const label = [state === 'offline' ? `${name} · Danbooru unreachable, not checked` : `${name} · Danbooru`];
  if (state !== 'offline' && unknown) label.push(`${unknown} unknown`);
  if (state === 'pending') label.push('checking…');
  if (state !== 'offline' && results.length > TAG_LOOKUP_CAP) {
    label.push(`first ${TAG_LOOKUP_CAP} of ${results.length} checked`);
  }
  const tokens = tagTokenLine();
  if (tokens) label.push(tokens.text);
  els.tagValidation.dataset.label = label.join(' · ');
  els.tagValidation.dataset.state = state;
  els.tagValidation.setAttribute('aria-label', name);
  els.tagValidation.toggleAttribute('data-over', !!tokens?.over);
  // The caption's legend only where a chip carries one: the guide's steps,
  // which the order rule placed each tag in.
  const ordered = chipSlots(results).some(Boolean);
  els.tagValidation.title = [
    state === 'pending' ? 'Checking Danbooru…'
      : state === 'offline'
        ? `Danbooru unreachable; unchecked tags are not treated as invalid${detail ? `: ${detail}` : ''}`
        : prose.size ? 'Tags checked against Danbooru or checkpoint guide tags; natural language stays as text'
          : 'Each comma-separated item checked against Danbooru or checkpoint guide tags',
    'edge colour: blue general, green character, purple copyright, red artist, yellow meta, teal guide tag',
    ...(ordered
      ? [`caption: the step of the guide's documented order Check places the tag in (${tagSlots.steps.join(' → ')}); none where it cannot place it`]
      : []),
    'click or right-click a gap to insert there',
  ].join(' · ');
}

// Each chip's order slot, {name, n, of} or null, read off the server's last
// answer: for this very text, index for index (the comma items, as the chips
// split them). For a text not answered yet - every keystroke renders the tray,
// and the answer lands a pause later - the last answer's slot for an item
// written the same, which its own answer then confirms or takes away: every
// mark went out and came back at each pause. Nothing is worked out here; an
// item the last answer gave no slot, or did not hold, has none.
function chipSlots(results) {
  const answer = tagSlots?.model === loadedModel && tagSlots.slots ? tagSlots : null;
  if (!answer) return results.map(() => null);
  const slot = (name) => (name ? { name, n: answer.steps.indexOf(name) + 1, of: answer.steps.length } : null);
  if (answer.prompt === els.prompt.value.trim()) {
    return results.map((result, index) => (!promptProseAt(index) && answer.slots[index]?.tag === result.tag ? slot(answer.slots[index].slot) : null));
  }
  const byTag = new Map(answer.slots.map((item) => [item.tag, item.slot]));
  return results.map((result, index) => (promptProseAt(index) ? null : slot(byTag.get(result.tag))));
}

function prosePromptText(part, cut) {
  const span = document.createElement('span');
  span.className = 'prompt-prose';
  span.dataset.first = String(part.first);
  span.dataset.last = String(part.last);
  const raw = els.prompt.value;
  span.title = 'Natural-language prose; edit it in the prompt box';
  if (cut && part.end > cut.offset) {
    const start = Math.max(part.start, cut.offset);
    span.append(document.createTextNode(raw.slice(part.start, start)));
    const unread = document.createElement('span');
    unread.className = 'prose-past-cut';
    unread.textContent = raw.slice(start, part.end);
    span.append(unread);
    span.title += `; ${part.start >= cut.offset ? 'not read' : 'partly read'}: ${cut.names} stop at ${cut.budget} tokens`;
  } else span.textContent = raw.slice(part.start, part.end);
  return span;
}

// The shared tip for an unknown chip: why, and that nothing is blocked. One
// element outside the tray, so a chip's text stays the tag's own.
function showTagTip(chip, lines) {
  const tip = els.tagTip;
  tip.textContent = lines.join('\n');
  tip.hidden = false;
  const box = chip.getBoundingClientRect();
  const below = box.bottom + 6 + tip.offsetHeight <= innerHeight - 8;
  tip.style.top = `${below ? box.bottom + 6 : box.top - 6 - tip.offsetHeight}px`;
  tip.style.left = `${Math.max(8, Math.min(box.left, innerWidth - tip.offsetWidth - 8))}px`;
}

function hideTagTip() {
  els.tagTip.hidden = true;
}

// The comma item being typed: the prompt's `input` says whether it was
// typing - a keystroke or a deletion, not a whole-box write, whose inserted
// text is the box - and the caret says which item.
function noteTagTyping(event) {
  clearTimeout(tagTypingTimer);
  const typed = event instanceof InputEvent
    && /^(insertText|insertCompositionText|delete)/.test(event.inputType)
    && !(event.data && event.data.length > 1 && event.data === els.prompt.value);
  const index = typed ? caretTagIndex() : null;
  tagTyping = index === null ? null : { index, tag: promptTagParts()[index] };
  if (tagTyping) tagTypingTimer = setTimeout(endTagTyping, 1500);
}

function endTagTyping() {
  clearTimeout(tagTypingTimer);
  const typing = tagTyping;
  tagTyping = null;
  if (!typing || tagInsertAt !== null || tagEditAt !== null || els.tagValidation.hidden) return;
  if (lastTagValidationResults[typing.index]?.valid === false) {
    renderTagValidation(lastTagValidationResults, lastTagValidationState, lastTagValidationDetail);
  }
}

// The caret moved by a click or a key: out of the item, it is typed.
function followTagCaret() {
  if (tagTyping && caretTagIndex() !== tagTyping.index) endTagTyping();
}

function renderTagValidation(results, state = 'checked', detail = '') {
  const hadFocus = els.tagValidation.contains(document.activeElement);
  lastTagValidationResults = results.map((result) => ({ ...result }));
  lastTagValidationState = state;
  lastTagValidationDetail = detail;
  if (tagInsertAt !== null && tagInsertAt > results.length) tagInsertAt = results.length;
  if (tagEditAt !== null && tagEditAt >= results.length) tagEditAt = null;
  // Read only where the chips are the prompt's own items: a render of other
  // results (a smoke's) has no place in the box to read from.
  const spans = promptTagSpans();
  const cut = spans.length === results.length ? tagCut() : null;
  const parts = promptDisplayParts(results);
  const prose = new Set(parts.filter((part) => part.kind === 'prose')
    .flatMap((part) => Array.from({ length: part.last - part.first + 1 }, (_, offset) => part.first + offset)));
  const typing = tagTyping && document.activeElement === els.prompt
    && results[tagTyping.index]?.tag === tagTyping.tag ? tagTyping.index : null;
  let focus = tagFocusIndex;
  if (typeof focus === 'number') focus = Math.min(focus, results.length - 1);
  const roving = [focus, ...results.keys()]
    .find((index) => typeof index === 'number' && index >= 0 && !prose.has(index) && index !== tagEditAt);
  if (typeof focus === 'number' && prose.has(focus)) focus = roving ?? 'add';
  const seen = new Set();
  const slots = chipSlots(results);
  let past = 0;
  let unread = false;
  hideTagTip();
  // A rebuild is not the user leaving the tray, whatever the browser says
  // about the focused chip it removes.
  tagRendering = true;
  els.tagValidation.replaceChildren();
  tagRendering = false;
  for (const [index, result] of results.entries()) {
    if (tagInsertAt === index) els.tagValidation.appendChild(buildTagEditor());
    if (prose.has(index)) {
      const part = parts.find((part) => part.kind === 'prose' && part.first === index);
      if (part) els.tagValidation.appendChild(prosePromptText(part, cut));
      continue;
    }
    if (tagEditAt === index) {
      els.tagValidation.appendChild(buildTagEditor(result.tag));
      continue;
    }
    const chip = document.createElement('span');
    chip.className = 'tag-check';
    chip.draggable = true;
    chip.tabIndex = index === roving ? 0 : -1;
    chip.dataset.index = String(index);
    const label = document.createElement('span');
    label.className = 'tag-check-label';
    const mark = document.createElement('span');
    mark.className = 'tag-check-mark';
    const text = document.createElement('span');
    text.className = 'tag-check-text';
    text.textContent = result.tag;
    const notes = [];
    let status;
    const tip = [];
    if (result.valid === true) {
      chip.classList.add('valid');
      // The category the lookup gave, as the edge's colour: what a tag is.
      // Where it goes is the caption (data-slot), the server's order rule's
      // placement - never worked out here.
      chip.dataset.category = result.source === 'model' ? 'guide' : result.category || 'general';
      mark.textContent = '✓';
      status = `known ${chip.dataset.category} tag`;
      notes.push(result.source === 'model'
        ? 'Checkpoint guide tag'
        : `Danbooru ${result.category} tag · ${humanCount(result.post_count)} posts`
          // The index answers parentheses bare, as autocomplete writes them;
          // an escaped copy is the same tag, so no note for that alone.
          + (result.canonical !== result.tag.replace(/\\([()])/g, '$1') ? ` · canonical: ${result.canonical}` : ''));
    } else if (result.code === 'grid_variable') {
      // The grid's `{name}`, which no tag is spelled as: no verdict, and the
      // tip says what fills it - the open fold's values, each with the
      // verdict the cache holds - or that nothing will (gridVariableTip).
      chip.classList.add('grid-variable');
      mark.textContent = '▦';
      status = 'grid variable';
      tip.push(...gridVariableTip(result.tag));
    } else if (result.valid === false && index === typing) {
      chip.classList.add('typing');
      status = 'being typed';
      notes.push('Being typed: checked once you move on');
    } else if (result.valid === false) {
      chip.classList.add('unknown');
      // A tag the checkpoint's cutoff came before is marked apart from one
      // the index never had: the fixes differ. Red either way.
      mark.textContent = result.code === 'after_cutoff' ? '⏲' : '?';
      status = result.code === 'after_cutoff' ? 'unknown, too new for this checkpoint' : 'unknown';
      tip.push(`${result.reason || 'not found in Danbooru'}; generation is still allowed.`, 'Double-click to fix it.');
    } else {
      chip.classList.add('unchecked');
      // "…" while the lookup is on its way, "–" where it will not come: the
      // index unreachable, or past the lookup's cap.
      const waiting = state === 'pending' && index < TAG_LOOKUP_CAP && !result.reason;
      mark.textContent = waiting ? '…' : '–';
      status = waiting ? 'not checked yet' : 'not checked';
      notes.push(waiting ? 'Checking Danbooru…' : result.reason || detail || 'Danbooru lookup unavailable');
    }
    if (cut && spans[index].start >= cut.offset) {
      chip.classList.add('past-cut');
      if (!unread) chip.classList.add('cut-start');
      unread = true;
      past += 1;
      notes.push(`not read: ${cut.names} stop at ${cut.budget} tokens`);
      status += ', not read';
    } else if (cut && spans[index].end > cut.offset) {
      chip.classList.add('part-cut');
      notes.push(`partly read: ${cut.names} stop at ${cut.budget} tokens inside it`);
      status += ', partly read';
    }
    const key = tagKey(result.canonical ?? result.tag);
    if (seen.has(key)) {
      chip.classList.add('duplicate');
      notes.push('repeated: adds nothing (heuristic)');
      status += ', repeated';
    }
    seen.add(key);
    // Where the guide's documented order puts it, as the server's order rule
    // placed it - a mark drawn by the stylesheet from data-slot, so the
    // chip's text stays the tag's own.
    const slot = slots[index];
    if (slot) {
      chip.dataset.slot = slot.name;
      label.dataset.slot = slot.name;
      notes.push(`documented order: ${slot.name}, step ${slot.n} of ${slot.of}`);
      status += `, order step ${slot.n} of ${slot.of}, ${slot.name}`;
    }
    label.append(mark, text);
    if (tip.length) {
      // The tip, not a title: a native tooltip would stack on it.
      tip.push(...notes);
      chip.setAttribute('aria-describedby', 'tag-tip');
      chip.onpointerenter = () => showTagTip(chip, tip);
      chip.onpointerleave = hideTagTip;
      chip.addEventListener('focus', () => showTagTip(chip, tip));
      chip.addEventListener('blur', hideTagTip);
    } else {
      chip.title = [...notes, 'double-click edits · drag or Alt+←/→ moves · right-click inserts after it · Delete removes']
        .join(' · ');
    }
    chip.setAttribute('aria-label',
      `${result.tag}, ${status}. Enter edits, Alt+Left or Right moves, the Menu key inserts after it, Delete removes.`);
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'tag-remove';
    remove.textContent = '×';
    remove.draggable = false;
    // Delete on the chip does the same: one tab stop per chip, not two.
    remove.tabIndex = -1;
    remove.title = `Delete ${result.tag}`;
    remove.setAttribute('aria-label', `Delete ${result.tag}`);
    remove.onclick = (event) => { event.stopPropagation(); deletePromptTag(index); };
    chip.append(label, remove);
    chip.addEventListener('focusin', () => {
      tagFocusIndex = index;
      for (const other of els.tagValidation.querySelectorAll('.tag-check')) other.tabIndex = other === chip ? 0 : -1;
    });
    chip.ondragstart = (event) => {
      draggedTagIndex = index;
      chip.classList.add('dragging');
      hideTagTip();
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
    chip.ondblclick = (event) => {
      // A fast double click on × deletes one chip and lands on the next; that
      // is not a request to edit it.
      if (event.target.closest('.tag-remove')) return;
      event.preventDefault();
      openTagEdit(index);
    };
    // The tray is one tab stop: arrows, Home and End walk the chips (Alt
    // with an arrow moves the tag instead), the rest act on the one focused.
    chip.onkeydown = (event) => {
      if (keyboardEventHeld(event) || keyboardEditingTarget(event.target)) return;
      const insert = (!event.shiftKey && event.key === 'ContextMenu') || (event.shiftKey && event.key === 'F10');
      if (insert && !keyboardCommandModifier(event)) {
        event.preventDefault();
        if (!event.repeat) openTagInsert(index + 1);
        return;
      }
      const reorder = event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey
        && ['ArrowLeft', 'ArrowRight'].includes(event.key);
      if (reorder) {
        event.preventDefault();
        movePromptTag(index, index + (event.key === 'ArrowLeft' ? -1 : 1));
        return;
      }
      if (keyboardCommandModifier(event) || event.shiftKey) return;
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        if (!event.repeat) deletePromptTag(index);
      } else if (event.key === 'Enter' || event.key === 'F2') {
        event.preventDefault();
        if (!event.repeat) openTagEdit(index);
      } else if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const chips = [...els.tagValidation.querySelectorAll('.tag-check')];
        const at = chips.indexOf(chip);
        const next = { ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: chips.length - 1 }[event.key];
        chips[Math.max(0, Math.min(chips.length - 1, next))].focus();
      }
    };
    els.tagValidation.appendChild(chip);
  }
  if (tagInsertAt === results.length) els.tagValidation.appendChild(buildTagEditor());
  else if (results.length) {
    // The visible way in for a pointer that does not right-click and for the
    // keyboard: a tag at the end. A gap takes a click or a right-click.
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'tag-add';
    add.textContent = '+ tag';
    add.title = 'Insert a tag at the end; click or right-click a gap to insert there';
    add.setAttribute('aria-label', 'Insert a tag at the end of the prompt');
    add.onclick = () => openTagInsert(results.length);
    add.addEventListener('focusin', () => { tagFocusIndex = 'add'; });
    els.tagValidation.appendChild(add);
  }
  syncTagTray(results, state, detail, typing);
  // A pure paragraph remains in the prompt box, rather than being duplicated
  // in an annotation tray. Its encoder readout remains under the box.
  els.tagValidation.hidden = results.length === 0 || prose.size === results.length;
  syncPromptTokens();
  if (hadFocus && els.tagValidation.hidden) els.prompt.focus({ preventScroll: true });
  // Read out once a lookup has answered, and only when it says something
  // new: the tray itself is no live region, since it is rebuilt at every
  // pause in typing and holds every chip.
  if (state !== 'pending' && results.length && (!els.tagValidation.hidden || !promptTokensOver)) {
    const known = results.filter((result, index) => !prose.has(index) && result.valid === true).length;
    // Not the item still being typed, as the label does not count it: a
    // lookup landing mid-word would read out an unfinished word as unknown.
    const unknown = results.filter((result, index) => !prose.has(index) && result.valid === false && index !== typing).length;
    const variables = results.filter((result) => result.code === 'grid_variable').length;
    // A spoken line, so "1 prompt tag", never "1 prompt tags" (analysis.js's
    // `plural`, read at call time). A grid variable is no tag: counted apart.
    const tags = results.length - variables - prose.size;
    const summary = (!tags && prose.size ? 'Natural-language prose' : state === 'offline'
      ? `${plural(tags, 'prompt tag')}, not checked: Danbooru unreachable`
      : `${plural(tags, 'prompt tag')}: ${known} known${unknown ? `, ${unknown} unknown` : ''}`)
      + (variables ? `, ${plural(variables, 'grid variable')}` : '')
      + (prose.size && tags ? ', natural-language prose' : '')
      + (cut && parts.some((part) => part.kind === 'prose' && part.end > cut.offset) ? ', prose past the encoders\' token limit' : '')
      + (past ? `, ${past} past the encoders' token limit` : '');
    if (els.tagSummary.textContent !== summary) els.tagSummary.textContent = summary;
  }
  const target = els.tagValidation.hidden ? null : focus === 'add'
    ? els.tagValidation.querySelector('.tag-add')
    : typeof focus === 'number' && focus >= 0
      ? els.tagValidation.querySelector(`.tag-check[data-index="${focus}"]`)
      : null;
  if (target && document.activeElement !== target && (tagRefocus || document.activeElement === document.body)) {
    target.focus({ preventScroll: true });
  }
  tagRefocus = false;
  // Not under an open editor, which the bar would only point beside.
  if (tagHoverPoint && draggedTagIndex === null && tagInsertAt === null && tagEditAt === null) {
    markTagGap(tagHoverPoint.x, tagHoverPoint.y);
  }
}

function uncheckedTag(tag, index) {
  return index < TAG_LOOKUP_CAP
    ? { tag, valid: null }
    : { tag, valid: null, reason: `past the ${TAG_LOOKUP_CAP}-item lookup cap` };
}

// A comma item that is a grid placeholder whole, `{hair}`: the grid's
// pattern (GRID_PLACEHOLDER, grid.PLACEHOLDER), which the server reads alike
// (prompt_text.GRID_VARIABLE) and answers with no verdict. Drawn as the
// grid's at once, so it never waits on, or is sent as, a tag.
const isGridVariable = (tag) => new RegExp(`^${GRID_PLACEHOLDER.source}$`).test(tag.trim());
const gridVariableResult = (tag) => ({ tag, valid: null, code: 'grid_variable' });

// What a grid variable's chip says: the values the open fold declares for
// it, each with the verdict the lookup gave it (asked beside the chips,
// gridValueTags), or that nothing fills it - which Generate refuses only
// while the fold declares it.
function gridVariableTip(tag) {
  const name = tag.trim().slice(1, -1);
  const values = declaredGridVariables().get(name);
  if (!values) {
    return [`A grid placeholder: Run grid fills it once Grid search's Prompt variables declare ${name}; Generate renders it as typed.`];
  }
  const cache = tagVerdictCache();
  const verdict = (value) => {
    if (NONE_VALUES.has(value.toLowerCase())) return `${value}: dropped from the prompt`;
    const known = cache.get(tagKey(value));
    if (!known) return `${value}: not checked yet`;
    if (known.valid !== true) return `${value}: ${known.reason || 'not found in Danbooru'}`;
    return `${value}: ${known.source === 'model' ? 'checkpoint guide tag' : `Danbooru ${known.category} tag`}`;
  };
  return [`Grid variable: Run grid fills it with ${values.join(' | ')}.`, ...values.map(verdict)];
}

// grid.NONE_TOKENS (held equal by a test): the deliberately empty value,
// which is no tag.
const NONE_VALUES = new Set(['none', 'null', 'off', '-', '_']);

// Whether the tray holds a grid variable's chip, whose tip reads the grid's
// variables box (app.js asks the tray again as the box or the fold changes).
const gridVariableChips = () => lastTagValidationResults.some((result) => result.code === 'grid_variable');

// The declared values of the grid variables in `tags` the cache has no
// verdict on, to ask beside the chips (validatePromptTags).
function gridValueTags(tags) {
  const declared = declaredGridVariables();
  if (!declared.size) return [];
  const cache = tagVerdictCache();
  const wanted = new Set();
  for (const tag of tags) {
    if (!isGridVariable(tag)) continue;
    for (const value of declared.get(tag.trim().slice(1, -1)) || []) {
      if (!NONE_VALUES.has(value.toLowerCase()) && !cache.has(tagKey(value))) wanted.add(value);
    }
  }
  return [...wanted];
}

function tagVerdictCache() {
  if (tagVerdicts.model !== loadedModel) tagVerdicts = { model: loadedModel, results: new Map() };
  return tagVerdicts.results;
}

function rememberTagVerdicts(results) {
  const cache = tagVerdictCache();
  for (const result of results) {
    if (result.valid === null) continue;
    const key = tagKey(result.tag);
    cache.delete(key);
    cache.set(key, result);
    if (cache.size > 2000) cache.delete(cache.keys().next().value);
  }
}

// The verdicts on the first 64 items and, with the prompt they were split
// from, every item's order slot (`tagSlots`): one request, so the slots are
// placed with the categories the same lookup just gave.
async function validatePromptTags(tags, epoch, prompt) {
  if (modelChangeHold()) return;
  const rest = tags.slice(TAG_LOOKUP_CAP).map((tag) => uncheckedTag(tag, TAG_LOOKUP_CAP));
  const model = loadedModel;
  const chipped = tags.slice(0, TAG_LOOKUP_CAP);
  // A grid variable's values ride the same request, where the cap leaves
  // room: their verdicts go to the cache its chip's tip reads, not to chips.
  const values = gridValueTags(chipped).slice(0, TAG_LOOKUP_CAP - chipped.length);
  try {
    const body = await api('/api/tags/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags: [...chipped, ...values], prompt }),
    });
    if (epoch !== tagValidationEpoch) return;
    rememberTagVerdicts(body.results);
    keepPromptParts(body.parts, prompt, model);
    const caret = caretTagIndex();
    if (currentPromptParts()?.some((part) => part.kind === 'prose' && caret >= part.first && caret <= part.last)) hideTagSuggestions();
    tagSlots = { model, prompt: prompt.trim(), steps: body.order_steps || [], slots: body.slots || null };
    renderTagValidation([...body.results.slice(0, chipped.length), ...rest], body.available ? 'checked' : 'offline', body.detail || '');
    // The Tags helper's caret line reads these slots.
    syncTagSelection();
  } catch (error) {
    if (epoch !== tagValidationEpoch) return;
    tagSlots = { model, prompt: prompt.trim(), steps: [], slots: null, failed: true };
    renderTagValidation(tags.map((tag, index) => (isGridVariable(tag) ? gridVariableResult(tag) : uncheckedTag(tag, index))),
      'offline', error.message);
    syncTagSelection();
  }
}

// Every prompt change from outside the editor comes through here - the
// textarea, a drop, a restored panel - and closes it: its index no longer
// names the item it opened on.
function scheduleTagValidation(delay = 320) {
  clearTimeout(tagValidationTimer);
  const epoch = ++tagValidationEpoch;
  tagInsertAt = null;
  tagEditAt = null;
  syncPromptUndo();
  const tags = tagVocabularyEnabled ? promptTagParts() : [];
  if (!tags.length) {
    // Nothing is asked, so nothing will answer: the caret line reads this
    // prompt as one the server placed nothing in, not as one still pending.
    tagSlots = { model: loadedModel, prompt: els.prompt.value.trim(), steps: [], slots: null };
    tagFocusIndex = null;
    hideTagTip();
    els.tagValidation.hidden = true;
    els.tagValidation.replaceChildren();
    // No chips, but the encoders still stop somewhere: the line under the
    // prompt counts, on the same pause and the same route as the tray.
    scheduleTagMeasure(delay);
    syncPromptTokens();
    return;
  }
  // What the index said before shows at once; only what it has not seen
  // waits for the answer.
  const cache = tagVerdictCache();
  const results = tags.map((tag, index) => {
    if (isGridVariable(tag)) return gridVariableResult(tag);
    const known = index < TAG_LOOKUP_CAP && cache.get(tagKey(tag));
    return known ? { ...known, tag } : uncheckedTag(tag, index);
  });
  const waiting = (tagHelper?.hybrid_prompt && !currentPromptParts())
    || results.some((result, index) => result.valid === null && !result.code && index < TAG_LOOKUP_CAP);
  renderTagValidation(results, waiting ? 'pending' : 'checked');
  const prompt = els.prompt.value;
  tagValidationTimer = setTimeout(() => validatePromptTags(tags, epoch, prompt), delay);
  scheduleTagMeasure(delay);
}

function syncTagVocabulary() {
  tagVocabularyEnabled = tagHelper?.vocabulary === 'danbooru';
  els.prompt.spellcheck = !tagVocabularyEnabled || Boolean(tagHelper?.hybrid_prompt);
  if (!tagVocabularyEnabled) { hideTagSuggestions(); hideNegativeSuggestions(); }
  syncTagFinder();
  scheduleTagValidation(0);
}

// The chips and the prompt check go together, in that order: in the results
// column on a two-column page, where their height cannot move Generate in the
// sticky settings column - above the results, except that from 1800 px
// styles/analysis.css docks an open check beside them (the chips and the viewer on the
// left, the check on the right; CSS only, this DOM order unchanged) - and back
// under the prompt at 900 px and below, where the page is one scrolling
// column anyway. On a two-column page each has a host of its own, the chips'
// first (`#tag-chips-wide`, `#tag-validation-wide`): the docked grid moves
// the check's host to just before the gallery in its reading order, and a
// host shared by both was one item, read at the chips' place, which took Tab
// through the check before the image beside it. The move
// keeps focus where it was - a
// window crossing 901 px under someone typing in Your version or on a chip
// otherwise sent it to <body> with the next keystroke: moveBefore keeps it
// whole where the browser has it, and elsewhere the focus and the selection
// are put back by hand.
function placeTagValidation() {
  const wide = window.matchMedia('(min-width: 901px)').matches;
  const focused = document.activeElement;
  const range = typeof focused?.selectionStart === 'number'
    ? [focused.selectionStart, focused.selectionEnd, focused.selectionDirection]
    : null;
  for (const [node, host] of [
    [els.tagValidation, wide ? els.tagChipsWide : els.tagValidationHome],
    [els.promptAnalysis, wide ? els.tagValidationWide : els.tagValidationHome],
  ]) {
    if (node.parentElement === host) continue;
    // Under the prompt the chips go ahead of the check, wherever it is.
    const before = node === els.tagValidation && els.promptAnalysis.parentElement === host ? els.promptAnalysis : null;
    if (typeof host.moveBefore === 'function') host.moveBefore(node, before);
    else host.insertBefore(node, before);
  }
  if (!focused || focused === document.activeElement || !focused.isConnected) return;
  focused.focus();
  if (range) focused.setSelectionRange(...range);
}

// The chips' height, on the stage: above the results the prompt check starts
// under them - stacked below 1800 px, docked across both columns from it - and
// its cap leaves them out, so the image just made stays on the screen
// (styles/analysis.css, `--chips-h`). From a ResizeObserver (app.js); a hidden tray is
// 0, and in the panel under 901 px the check is not capped.
function chipsResized() {
  els.viewer.parentElement.style.setProperty('--chips-h', `${els.tagValidation.getBoundingClientRect().height}px`);
}

function tagCursorIndex() {
  const before = els.prompt.value.slice(0, els.prompt.selectionStart ?? els.prompt.value.length);
  return before.trim() ? before.split(',').length : 0;
}

// Every chip and Tags-helper edit writes the whole list back here, through
// `writeBox`, so Ctrl+Z restores what the box held. That takes focus to the
// prompt; it goes back to the chip `focusIndex` names, or to the helper's
// button that made the write. A chip's visit to the box is only in passing,
// so neither the panel nor the page moves to show it - a panel held by its
// bottom jumped 62 px at each Alt+Arrow in the other column - which is the
// press's exemption from `revealInPanel`, and no scroll on the focus.
function writePromptTags(parts, caret = null, focusIndex = null, layout = null) {
  const text = typeof parts === 'string' ? parts : parts.join(', ');
  if (layout) keepPromptParts(layout, text);
  const from = document.activeElement;
  const passing = focusIndex !== null && focusIndex >= 0;
  const press = panelPos.press;
  if (passing) {
    panelPos.press = true;
    els.prompt.focus({ preventScroll: true });
  }
  try {
    writeBox(els.prompt, text);
  } finally {
    panelPos.press = press;
  }
  promptReplaced();
  const end = caret == null ? els.prompt.value.length : Math.min(caret, els.prompt.value.length);
  els.prompt.setSelectionRange(end, end);
  if (focusIndex !== null && focusIndex >= 0) [tagFocusIndex, tagRefocus] = [focusIndex, true];
  syncTagSelection();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  // A press in the helper or Find tag stays there, so a second press takes
  // the tag back out rather than typing into the box.
  if (focusIndex === null && from?.isConnected && (els.tagsMenu.contains(from) || els.tagFinderMenu.contains(from))) {
    from.focus({ preventScroll: true });
  }
}
