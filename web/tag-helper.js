// AuraDream's page script: the Tags helper - its menu, toggles, Start with,
// Finish with and the caret line that presents the server's order and slots.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// Buttons are toggles. Exclusive groups replace an existing person-count or
// rating tag in place; otherwise insertion follows the textarea caret. Nothing
// silently reorders free-form character, series, pose or scene tags.
function togglePromptTag(tag, group) {
  let parts = promptEditingUnits();
  const wanted = tag.toLowerCase();
  const active = parts.findIndex((part) => part.kind === 'tag' && part.text.toLowerCase() === wanted);
  if (active >= 0) {
    parts.splice(active, 1);
    writePromptUnits(parts);
    return;
  }

  let insertAt = promptUnitInsertion(tagCursorIndex(), parts);
  if (group.exclusive) {
    const choices = new Set(group.tags.map((choice) => choice.toLowerCase()));
    const old = parts.findIndex((part) => part.kind === 'tag' && choices.has(part.text.toLowerCase()));
    parts = parts.filter((part) => part.kind !== 'tag' || !choices.has(part.text.toLowerCase()));
    if (old >= 0) insertAt = Math.min(old, parts.length);
  }
  parts.splice(insertAt, 0, { kind: 'tag', text: tag });
  const caret = parts.slice(0, insertAt + 1).map((part) => part.text).join(', ').length;
  writePromptUnits(parts, { caret });
}

function setYearTag(year) {
  let parts = promptEditingUnits();
  let insertAt = parts.findIndex((part) => part.kind === 'tag' && /^year \d{4}$/i.test(part.text));
  if (insertAt < 0) insertAt = promptUnitInsertion(tagCursorIndex(), parts);
  parts = parts.filter((part) => part.kind !== 'tag' || !/^year \d{4}$/i.test(part.text));
  insertAt = Math.min(insertAt, parts.length);
  parts.splice(insertAt, 0, { kind: 'tag', text: `year ${year}` });
  writePromptUnits(parts, { caret: parts.slice(0, insertAt + 1).map((part) => part.text).join(', ').length });
}

function finishWithTags(suffix) {
  const ending = new Set(suffix.map((tag) => tag.toLowerCase()));
  const parts = promptEditingUnits().filter((part) => part.kind !== 'tag' || !ending.has(part.text.toLowerCase()));
  writePromptUnits([...parts, ...suffix.map((text) => ({ kind: 'tag', text }))]);
}

// The card's recommended positive prefix (Anima Turbo's), offered and never
// checked: the tags of it the prompt has not got, less any an exclusive group
// already holds another of - no safe beside sensitive, nsfw or explicit.
function prefixMissing(prefix) {
  const present = new Set(promptEditingUnits().filter((part) => part.kind === 'tag').map((part) => part.text.toLowerCase()));
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
  if (!tagHelper?.hybrid_prompt) {
    const rest = els.prompt.value.trim();
    writePromptTags(rest ? [...missing, rest] : missing);
    return;
  }
  writePromptUnits([...missing.map((text) => ({ kind: 'tag', text })), ...promptEditingUnits()]);
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
  const active = new Set(promptEditingUnits().filter((part) => part.kind === 'tag').map((part) => part.text.toLowerCase()));
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
