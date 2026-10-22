// Local Diffusion's page script: tag validation chips and guide presets.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ---------------------------------------------------------------- the chips

// The tray's label, its title and its state: what the lookup said (checked,
// still checking, the index unreachable - amber, never red, since nothing is
// invalid for it), how many were unknown - not the item still being typed -
// the lookup's cap, and the count.
function syncTagTray(results, state, detail = '', typing = null) {
  const unknown = results.filter((result, index) => result.valid === false && index !== typing).length;
  const label = [state === 'offline' ? 'Prompt tags · Danbooru unreachable, not checked' : 'Prompt tags · Danbooru'];
  if (state !== 'offline' && unknown) label.push(`${unknown} unknown`);
  if (state === 'pending') label.push('checking…');
  if (state !== 'offline' && results.length > TAG_LOOKUP_CAP) {
    label.push(`first ${TAG_LOOKUP_CAP} of ${results.length} checked`);
  }
  const tokens = tagTokenLine();
  if (tokens) label.push(tokens.text);
  els.tagValidation.dataset.label = label.join(' · ');
  els.tagValidation.dataset.state = state;
  els.tagValidation.toggleAttribute('data-over', !!tokens?.over);
  // The caption's legend only where a chip carries one: the guide's steps,
  // which the order rule placed each tag in.
  const ordered = chipSlots(results).some(Boolean);
  els.tagValidation.title = [
    state === 'pending' ? 'Checking Danbooru…'
      : state === 'offline'
        ? `Danbooru unreachable; unchecked tags are not treated as invalid${detail ? `: ${detail}` : ''}`
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
    return results.map((result, index) => (answer.slots[index]?.tag === result.tag ? slot(answer.slots[index].slot) : null));
  }
  const byTag = new Map(answer.slots.map((item) => [item.tag, item.slot]));
  return results.map((result) => slot(byTag.get(result.tag)));
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
  lastTagValidationResults = results.map((result) => ({ ...result }));
  lastTagValidationState = state;
  lastTagValidationDetail = detail;
  if (tagInsertAt !== null && tagInsertAt > results.length) tagInsertAt = results.length;
  if (tagEditAt !== null && tagEditAt >= results.length) tagEditAt = null;
  // Read only where the chips are the prompt's own items: a render of other
  // results (a smoke's) has no place in the box to read from.
  const spans = promptTagSpans();
  const cut = spans.length === results.length ? tagCut() : null;
  const typing = tagTyping && document.activeElement === els.prompt
    && results[tagTyping.index]?.tag === tagTyping.tag ? tagTyping.index : null;
  let focus = tagFocusIndex;
  if (typeof focus === 'number') focus = Math.min(focus, results.length - 1);
  const roving = [focus, ...results.keys()]
    .find((index) => typeof index === 'number' && index >= 0 && index !== tagEditAt);
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
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        deletePromptTag(index);
      } else if (event.key === 'Enter' || event.key === 'F2') {
        event.preventDefault();
        openTagEdit(index);
      } else if (event.altKey && event.key === 'ArrowLeft') {
        event.preventDefault();
        movePromptTag(index, index - 1);
      } else if (event.altKey && event.key === 'ArrowRight') {
        event.preventDefault();
        movePromptTag(index, index + 1);
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
  els.tagValidation.hidden = results.length === 0;
  syncPromptTokens();
  // Read out once a lookup has answered, and only when it says something
  // new: the tray itself is no live region, since it is rebuilt at every
  // pause in typing and holds every chip.
  if (state !== 'pending' && results.length) {
    const known = results.filter((result) => result.valid === true).length;
    // Not the item still being typed, as the label does not count it: a
    // lookup landing mid-word would read out an unfinished word as unknown.
    const unknown = results.filter((result, index) => result.valid === false && index !== typing).length;
    const variables = results.filter((result) => result.code === 'grid_variable').length;
    // A spoken line, so "1 prompt tag", never "1 prompt tags" (analysis.js's
    // `plural`, read at call time). A grid variable is no tag: counted apart.
    const tags = results.length - variables;
    const summary = (state === 'offline'
      ? `${plural(tags, 'prompt tag')}, not checked: Danbooru unreachable`
      : `${plural(tags, 'prompt tag')}: ${known} known${unknown ? `, ${unknown} unknown` : ''}`)
      + (variables ? `, ${plural(variables, 'grid variable')}` : '')
      + (past ? `, ${past} past the encoders' token limit` : '');
    if (els.tagSummary.textContent !== summary) els.tagSummary.textContent = summary;
  }
  const target = focus === 'add'
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
    tagSlots = { model, prompt, steps: body.order_steps || [], slots: body.slots || null };
    renderTagValidation([...body.results.slice(0, chipped.length), ...rest], body.available ? 'checked' : 'offline', body.detail || '');
    // The Tags helper's caret line reads these slots.
    syncTagSelection();
  } catch (error) {
    if (epoch !== tagValidationEpoch) return;
    tagSlots = { model, prompt, steps: [], slots: null, failed: true };
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
  const hybridSentence = tagHelper?.hybrid_prompt && !els.prompt.value.includes(',');
  const tags = tagVocabularyEnabled && !hybridSentence ? promptTagParts() : [];
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
  const waiting = results.some((result, index) => result.valid === null && !result.code && index < TAG_LOOKUP_CAP);
  renderTagValidation(results, waiting ? 'pending' : 'checked');
  const prompt = els.prompt.value.trim();
  tagValidationTimer = setTimeout(() => validatePromptTags(tags, epoch, prompt), delay);
  scheduleTagMeasure(delay);
}

function syncTagVocabulary() {
  tagVocabularyEnabled = tagHelper?.vocabulary === 'danbooru';
  els.prompt.spellcheck = !tagVocabularyEnabled || Boolean(tagHelper?.hybrid_prompt);
  if (!tagVocabularyEnabled) hideTagSuggestions();
  syncTagFinder();
  scheduleTagValidation(0);
}

// The chips and the prompt check go together, in that order: in the results
// column on a two-column page, where their height cannot move Generate in the
// sticky settings column - above the results, except that from 1800 px
// style.css docks an open check beside them (the chips and the viewer on the
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
// (style.css, `--chips-h`). From a ResizeObserver (app.js); a hidden tray is
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
function writePromptTags(parts, caret = null, focusIndex = null) {
  const from = document.activeElement;
  const passing = focusIndex !== null && focusIndex >= 0;
  const press = panelPos.press;
  if (passing) {
    panelPos.press = true;
    els.prompt.focus({ preventScroll: true });
  }
  try {
    writeBox(els.prompt, parts.join(', '));
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

// The card's recommended positive prefix (Anima Turbo's), offered and never
// checked: the tags of it the prompt has not got, less any an exclusive group
// already holds another of - no safe beside sensitive, nsfw or explicit.
function prefixMissing(prefix) {
  const present = new Set(promptTagParts().map((part) => part.toLowerCase()));
  const held = (tag) => (tagHelper?.groups ?? []).some((group) => group.exclusive
    && group.tags.some((choice) => choice.toLowerCase() === tag.toLowerCase())
    && group.tags.some((choice) => present.has(choice.toLowerCase())));
  return prefix.filter((tag) => !present.has(tag.toLowerCase()) && !held(tag));
}

// The missing tags go ahead of the prompt, which is otherwise left as typed:
// sentences included, since the card puts its prefix ahead of prose too.
// Through writePromptTags, so Ctrl+Z in the box takes it back.
function startWithTags(prefix) {
  const missing = prefixMissing(prefix);
  if (!missing.length) return;
  const rest = els.prompt.value.trim();
  writePromptTags(rest ? [...missing, rest] : missing);
}

// Held, with the reason as its title, where it would add nothing; live, it
// says which tags it adds.
function syncStartWith() {
  const button = els.tagsMenu?.querySelector('[data-action="recommended-prefix"]');
  if (!button) return;
  const missing = prefixMissing(tagHelper.recommended_prefix);
  button.setAttribute('aria-disabled', String(!missing.length));
  button.title = missing.length
    ? `Adds ${missing.join(', ')} at the start of the prompt`
    : 'The prompt already has the recommended prefix, or a safety tag in place of safe';
}

// Where the next tag goes, in the documented order's own steps: the helper
// inserts at the prompt's caret, and the usual caret - the end, after the
// quality tags - is not where most of its groups belong. The steps a tag
// there would be in order at are read off the server's slots for the tags
// either side (`chipSlots`, Check's placement - a lone tag included): from
// the nearest placed tag before the caret to the nearest one after it - a
// tag Check gives no step is passed over, as its order rule passes it over.
// The page places nothing and has no order of its own. `answer` says what
// the server said of this very prompt: 'pending' while its answer has not
// landed, 'failed' where the request failed, 'unread' where it placed
// nothing in it (`places_items` false), 'placed' otherwise.
function tagCaretSteps(parts, at) {
  const steps = tagHelper?.order ?? [];
  if (!steps.length) return null;
  const slots = chipSlots(parts.map((tag) => ({ tag })))
    .map((slot) => (slot && slot.n > 0 ? slot : null));
  let before = -1;
  for (let i = at - 1; i >= 0; i--) if (slots[i]) { before = i; break; }
  let after = -1;
  for (let i = at; i < parts.length; i++) if (slots[i]) { after = i; break; }
  const lo = !parts.length ? 0 : before >= 0 ? slots[before].n - 1 : 0;
  const hi = !parts.length ? 0 : after >= 0 ? slots[after].n - 1 : steps.length - 1;
  const current = tagSlots?.model === loadedModel && tagSlots.prompt === els.prompt.value.trim();
  const answer = !current ? 'pending' : tagSlots.failed ? 'failed' : tagSlots.slots ? 'placed' : 'unread';
  return { steps, slots, lo, hi, placed: before >= 0 || after >= 0, answer };
}

function tagStepList(steps) {
  if (steps.length === 1) return `A ${steps[0]} tag`;
  if (steps.length <= 3) return `A ${steps.slice(0, -1).join(', ')} or ${steps.at(-1)} tag`;
  return `A tag from ${steps[0]} through ${steps.at(-1)}`;
}

function tagInsertionNote() {
  const parts = promptTagParts();
  const at = Math.min(tagCursorIndex(), parts.length);
  const fit = tagCaretSteps(parts, at);
  const named = (i) => `"${parts[i]}"${fit?.slots[i] ? ` (${fit.slots[i].name})` : ''}`;
  const where = !parts.length ? 'The prompt is empty'
    : at === 0 ? `The caret is at the start, before ${named(0)}`
      : at >= parts.length ? `The caret is at the end, after ${named(at - 1)}`
        : `The caret is after ${named(at - 1)}, before ${named(at)}`;
  const move = 'Click in the prompt to move the caret.';
  if (!fit) return `${where}; a tag goes in there. ${move}`;
  if (!parts.length) return `${where}: start with ${fit.steps[0]}.`;
  if (fit.lo > fit.hi) {
    return `${where}. The tags around it are already out of the recommended order - Check's Use reordered fixes that. ${move}`;
  }
  if (!fit.placed) {
    const order = `the recommended order is ${fit.steps.join(' → ')}`;
    if (fit.answer === 'pending') {
      return `${where}. The steps of the tags around it are still being looked up. ${move}`;
    }
    if (fit.answer === 'failed') {
      return `${where}. The tags could not be placed (the lookup failed), so ${order}. ${move}`;
    }
    if (fit.answer === 'unread') {
      return `${where}. Check reads no order in this prompt, so ${order}. ${move}`;
    }
    return `${where}. Check places none of the tags around it in one step, so ${order}. ${move}`;
  }
  return `${where}. ${tagStepList(fit.steps.slice(fit.lo, fit.hi + 1))} fits there. ${move}`;
}

// The rest of how the buttons write, which stays put.
function tagHelperNote() {
  const exclusive = (tagHelper?.groups ?? []).filter((group) => group.exclusive)
    .map((group) => group.label.toLowerCase());
  const replaces = exclusive.length
    ? `${exclusive.join(' and ').replace(/^./, (c) => c.toUpperCase())} choices replace their current tag in place. `
    : '';
  const reorder = tagHelper?.order?.length ? " Check's Use reordered puts the tags in the recommended order." : '';
  return `${replaces}Select an active tag to remove it.${reorder}`;
}

function syncTagSelection() {
  syncTagFinderSelection();
  if (!els.tagsMenu || els.tagsMenu.hidden) return;
  const parts = promptTagParts();
  const active = new Set(parts.map((tag) => tag.toLowerCase()));
  for (const button of els.tagsMenu.querySelectorAll('[data-tag]')) {
    const on = active.has(button.dataset.tag.toLowerCase());
    button.classList.toggle('active', on);
    button.setAttribute('aria-pressed', String(on));
  }
  syncStartWith();
  const note = els.tagsMenu.querySelector('.tag-insert');
  if (note) note.textContent = tagInsertionNote();
  // The steps the note names, marked in the order line and on the groups
  // whose tags go there - the note's words, shown; nothing else reads them.
  const fit = tagCaretSteps(parts, Math.min(tagCursorIndex(), parts.length));
  const fits = new Set(fit && fit.lo <= fit.hi && (fit.placed || !parts.length)
    ? fit.steps.slice(fit.lo, fit.hi + 1) : []);
  for (const el of els.tagsMenu.querySelectorAll('[data-step]')) {
    el.classList.toggle('fits', fits.has(el.dataset.step));
  }
}

// The order line, as Check's Order section shows it - both from the server:
// the documented order's steps, with its cite as the title, or where this app
// checks none, the authors' own orders Check lists there, with their notes.
function tagOrderLines() {
  if (tagHelper.order?.length) {
    const order = document.createElement('p');
    order.className = 'tag-order';
    if (tagHelper.order_cite) order.title = tagHelper.order_cite;
    order.append('Recommended order: ');
    tagHelper.order.forEach((step, index) => {
      if (index) order.append(' → ');
      const name = document.createElement('span');
      name.className = 'tag-step';
      name.dataset.step = step;
      name.textContent = step;
      order.appendChild(name);
    });
    return [order];
  }
  const authors = tagHelper.author_orders ?? [];
  const none = document.createElement('p');
  none.className = 'tag-order';
  none.textContent = authors.length
    ? "This app checks no tag order for this checkpoint. The authors' own texts give, as information:"
    : 'This app checks no tag order for this checkpoint: its guide states none.';
  return [none, ...authors.map((item) => {
    const line = document.createElement('p');
    line.className = 'tag-order tag-author-order';
    line.textContent = item.note ? `${item.text} (${item.note})` : item.text;
    return line;
  })];
}

// A group's heading names its step where the step is not its own name -
// Animagine's Score tags go in Quality, WAI's Rating in Subject and scene.
function tagGroupHeading(label, step) {
  const heading = document.createElement('strong');
  heading.textContent = label;
  if (step && step !== label) {
    const where = document.createElement('span');
    where.className = 'tag-group-step';
    where.textContent = ` · ${step} step`;
    heading.appendChild(where);
  }
  return heading;
}

function buildTagHelper() {
  els.tagsMenu.replaceChildren();
  if (!tagHelper) return;

  els.tagsMenu.append(...tagOrderLines());
  const insert = document.createElement('p');
  insert.className = 'tag-insert';
  els.tagsMenu.appendChild(insert);

  for (const group of tagHelper.groups) {
    const section = document.createElement('section');
    section.className = 'tag-group';
    const step = tagHelper.group_steps?.[group.label];
    if (step) section.dataset.step = step;
    const heading = tagGroupHeading(group.label, step);
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
    if (tagHelper.year_step) section.dataset.step = tagHelper.year_step;
    const heading = tagGroupHeading('Year', tagHelper.year_step);
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
  if (tagHelper.recommended_prefix?.length) {
    const start = document.createElement('button');
    start.type = 'button';
    start.className = 'ghost tiny';
    start.dataset.action = 'recommended-prefix';
    start.textContent = `Start with ${tagHelper.recommended_prefix.join(', ')}`;
    start.onclick = () => {
      if (start.getAttribute('aria-disabled') === 'true') { showError(start.title, true); return; }
      startWithTags(tagHelper.recommended_prefix);
    };
    actions.appendChild(start);
  }
  if (tagHelper.recommended_suffix?.length) {
    const finish = document.createElement('button');
    finish.type = 'button';
    finish.className = 'ghost tiny';
    finish.dataset.action = 'recommended-suffix';
    finish.textContent = `Finish with ${tagHelper.recommended_suffix.join(', ')}`;
    finish.onclick = () => finishWithTags(tagHelper.recommended_suffix);
    actions.appendChild(finish);
  }
  if (tagHelper.negative_prompt) {
    const negative = document.createElement('button');
    negative.type = 'button';
    negative.className = 'ghost tiny';
    negative.dataset.action = 'recommended-negative';
    negative.textContent = 'Use the recommended negative prompt';
    // Written as typing writes it, so Ctrl+Z in the box brings back a
    // negative prompt of the user's own; focus stays in the helper.
    negative.onclick = () => {
      writeBox(els.negative, tagHelper.negative_prompt);
      negative.focus({ preventScroll: true });
      syncAnalysisStale();
      schedulePreview();
      persistPanel();
    };
    actions.appendChild(negative);
  }
  if (actions.children.length) els.tagsMenu.appendChild(actions);

  const note = document.createElement('p');
  note.className = 'tag-note';
  note.textContent = tagHelperNote();
  els.tagsMenu.appendChild(note);
  syncTagSelection();
}

// The helper drops under the prompt, over the fields below it, where the
// caret it inserts at stays in view and a click in the box moves it without
// closing the helper. Opened from the keyboard it takes focus - it follows
// the prompt in the page, four stops past its button - and closing hands
// focus back to the button rather than to <body>.
function showTagHelper(open, { focus = false } = {}) {
  if (open && !els.examplesMenu.hidden) showExamples(false);
  if (open && !els.tagFinderMenu.hidden) showTagFinder(false);
  const inside = els.tagsMenu.contains(document.activeElement);
  els.tagsBtn.setAttribute('aria-expanded', String(open));
  els.tagsMenu.hidden = !open;
  if (open) {
    buildTagHelper();
    if (focus) els.tagsMenu.querySelector('button')?.focus({ preventScroll: true });
  } else if (inside) els.tagsBtn.focus();
}

function syncTagHelper() {
  const available = Boolean(tagHelper && Array.isArray(tagHelper.groups) && tagHelper.groups.length);
  els.tagHelper.hidden = !available;
  if (!available) showTagHelper(false);
  else if (!els.tagsMenu.hidden) buildTagHelper();
  syncTagVocabulary();
}
