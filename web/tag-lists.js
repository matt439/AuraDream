// AuraDream's page script: the editors' suggestion lists (the chip editor's
// and the Refine dialog's boxes) and the negative box's suggestions.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

function selectEditorSuggestion(menu, index) {
  const suggestions = menu._suggestions || [];
  if (!suggestions.length) return;
  menu._index = (index + suggestions.length) % suggestions.length;
  [...menu.children].forEach((item, i) => {
    const active = i === menu._index;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
    if (active) {
      menu._input?.setAttribute('aria-activedescendant', item.id);
      item.scrollIntoView({ block: 'nearest' });
    }
  });
}

// The suggestion list of a box other than the panel's prompt: the chip
// editor's input, where a pick commits the edit, and the Refine dialog's
// prompt (refine-prompt.js), where it replaces the comma item at the caret - `pick`.
// `query` is what was looked up, which the preselect rule reads: the editor's
// whole text, the dialog's item at the caret. A click, or Enter or Tab on
// the selected item (the box's own keydown, through `menu._pick`), picks.
// `place`, given, positions the list once it is drawn, at its real height.
// `menu._query` is the item a drawn list answers.
function renderEditorSuggestions(
  input, menu, suggestions, { query = input.value, pick = (s) => commitTagEdit(s.text), place = null } = {},
) {
  menu._suggestions = suggestions;
  menu._query = suggestions.length ? query.trim() : null;
  menu._index = -1;
  menu._input = input;
  menu._pick = pick;
  menu.replaceChildren();
  input.removeAttribute('aria-activedescendant');
  for (const [index, suggestion] of suggestions.entries()) {
    const item = document.createElement('li');
    item.id = `${menu.id}-${index}`;
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', 'false');
    const kind = document.createElement('span');
    kind.className = 'tag-kind';
    kind.textContent = tagKind(suggestion.category);
    const name = document.createElement('span');
    name.className = 'tag-name';
    name.textContent = suggestion.text;
    const count = document.createElement('span');
    count.className = 'tag-count';
    count.textContent = suggestion.post_count == null ? '' : humanCount(suggestion.post_count);
    item.append(kind, name, count);
    item.onmousedown = (event) => event.preventDefault();
    item.onclick = () => pick(suggestion);
    menu.appendChild(item);
  }
  menu.hidden = suggestions.length === 0;
  input.setAttribute('aria-expanded', String(suggestions.length > 0));
  const editor = menu.parentElement.getBoundingClientRect();
  menu.classList.toggle('end', editor.left + 280 > window.innerWidth - 12);
  place?.(menu);
  // The prompt box's rule: "lo" then Tab in the editor wrote `low quality`.
  if (suggestions.length && preselects(query, suggestions[0])) selectEditorSuggestion(menu, 0);
}

// Nothing selected in a list left up: Enter and Tab are the box's own again.
function deselectEditorSuggestions(menu) {
  menu._index = -1;
  for (const item of menu.children) {
    item.classList.remove('active');
    item.setAttribute('aria-selected', 'false');
  }
  menu._input?.removeAttribute('aria-activedescendant');
}

// The list emptied at once and asked again after a pause, so Enter never
// takes a match for what was typed before. With `keep` (the Refine dialog's
// prompt) a list already up stays while it asks, nothing in it selected -
// the same guarantee, without the list blinking shut on every key, as the
// panel's prompt keeps its own. `field` names the box to the server
// (`field=negative`, the negative box's); the rest of `options` is
// renderEditorSuggestions'.
function scheduleEditorSuggestions(input, menu, options = {}) {
  clearTimeout(tagEditorSuggestionTimer);
  const epoch = ++tagEditorSuggestionEpoch;
  const query = (options.query ?? input.value).trim();
  const value = input.value;
  const model = loadedModel;
  if (options.keep && query && !menu.hidden) deselectEditorSuggestions(menu);
  else renderEditorSuggestions(input, menu, [], options);
  if (!query) return;
  tagEditorSuggestionTimer = setTimeout(async () => {
    try {
      const field = options.field ? `&field=${encodeURIComponent(options.field)}` : '';
      const body = await api(`/api/tags/suggest?q=${encodeURIComponent(query)}${field}`);
      if (epoch !== tagEditorSuggestionEpoch || !input.isConnected || input.value !== value
        || model !== loadedModel || (options.current && !options.current())) return;
      renderEditorSuggestions(input, menu, body.suggestions || [], options);
    } catch {
      if (epoch === tagEditorSuggestionEpoch) renderEditorSuggestions(input, menu, [], options);
    }
  }, 160);
}

// The keys of an editor list left up under a box, on the box: ArrowDown and
// ArrowUp walk it, Enter or Tab takes the selected item, and an Escape the
// list takes closes the list alone (`hide`) - prevented, so nothing under it
// (a dialog's cancel, the prompt check's Escape) sees it; the next one does.
// Ctrl+Enter stays the document's. The Refine dialog's prompt and the panel's
// negative box.
function editorListKeydown(event, menu, hide) {
  if (menu.hidden || keyboardEventHeld(event) || keyboardCommandModifier(event) || event.shiftKey) return;
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    selectEditorSuggestion(menu, menu._index + (event.key === 'ArrowDown' ? 1 : -1));
  } else if ((event.key === 'Enter' || event.key === 'Tab') && menu._index >= 0) {
    event.preventDefault();
    if (!event.repeat) menu._pick(menu._suggestions[menu._index]);
  } else if (event.key === 'Escape') {
    event.preventDefault();
    hide();
  }
}

// ------------------------------------------------ the negative box's suggestions

// On a tag checkpoint the recommended negative prompt is a Danbooru tag list
// too, and the negative box suggests tags for the comma item at the caret as
// the Refine dialog's prompt does: the editor list under the box, its own
// (#negative-tag-suggestions, floating over the fields below, so it takes no
// room), the first item selected only from three typed characters it starts
// with (`preselects`). Asked with `field=negative`: the server keeps the
// guide's negative tags from the top of the prompt's list, and here they are
// the ones wanted. None of the prompt's own: no Find tag row (the finder
// writes into the prompt), no chips or typing state, and no prose guard -
// the negative has no classified layout. A greyed box is `disabled`, never
// focused, and asks nothing; one greyed with its list up closes it (syncCfg).
// Asked on typing, a click and the caret keys; a list already up stays while
// it asks (`keep`), and a click or caret key that stays in its item asks
// nothing. A whole-box write (`replacingBox`) is not typing.
function suggestNegativeTags(event) {
  const el = els.negative;
  const menu = els.negativeTagSuggestions;
  if (!tagVocabularyEnabled || el.disabled || replacingBox || document.activeElement !== el) {
    hideNegativeSuggestions();
    return;
  }
  const range = currentTagRange(el);
  const { query } = range;
  const text = el.value;
  const model = loadedModel;
  if (event?.type !== 'input' && !menu.hidden && menu._query === query) return;
  scheduleEditorSuggestions(el, menu, {
    query,
    field: 'negative',
    pick: applyNegativeSuggestion,
    keep: true,
    current: () => loadedModel === model && tagVocabularyEnabled && !el.disabled
      && document.activeElement === el && el.value === text
      && currentTagRange(el).start === range.start && currentTagRange(el).end === range.end,
  });
}

// Only a list that is up: the lookup is shared with the chip editor and the
// Refine dialog (state.js), and closing a list that is not there - a greyed
// box, a write, a blur - must not drop theirs. One the negative asked for and
// has not drawn yet is dropped by its own `current()`.
function hideNegativeSuggestions() {
  const menu = els.negativeTagSuggestions;
  if (menu.hidden) return;
  clearTimeout(tagEditorSuggestionTimer);
  tagEditorSuggestionEpoch += 1;
  renderEditorSuggestions(els.negative, menu, [], { pick: applyNegativeSuggestion });
}

// The comma item at the caret replaced by an insert, as typing replaces a
// selection: one Ctrl+Z takes the suggestion back and leaves what was typed.
// A caret-range edit, not a whole-box write. Its `input` is the box's own -
// the prompt check dims, the panel is remembered - and nothing opens over the
// accepted tag.
function applyNegativeSuggestion(suggestion) {
  const el = els.negative;
  const { start, end } = currentTagRange(el);
  el.focus();
  el.setSelectionRange(start, end);
  if (!document.execCommand('insertText', false, suggestion.text)) {
    el.setRangeText(suggestion.text, start, end, 'end');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }
  hideNegativeSuggestions();
}

function negativeCaret(event) {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) suggestNegativeTags(event);
}
