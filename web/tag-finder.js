// Local Diffusion's page script: Find tag - a tag found by what it shows.
// "cowlick sticking up" names no tag the suggestions could complete; the
// last row of the prompt box's list (tags.js, renderTagSuggestions) opens
// this dropdown with the words searched. The server searches Danbooru's wiki
// for them and answers general tags whose page mentions them, each with its
// first sentence, so a page that merely mentions the words reads as such.
// The first result pressed replaces the typed description, as a suggestion
// does, one Ctrl+Z from it; after that a press inserts at the caret, as the
// Tags helper's own tags do (togglePromptTag), and a second takes it out.
// It hangs where the helper does, never inside it: the helper already fills
// a 900 px window on Animagine, and the Prompt caption has no room left.

// The last search, kept while the finder closes and opens again, and
// dropped for another checkpoint's helper: its cutoff answers differently.
// `typed` is the description's place in the prompt while it is still there.
let tagFinder = { helper: null, query: '', results: [], status: '', typed: null };
let tagFinderEpoch = 0;

function tagFinderState() {
  if (tagFinder.helper !== tagHelper) {
    tagFinderEpoch += 1;
    tagFinder = { helper: tagHelper, query: '', results: [], status: '', typed: null };
  }
  return tagFinder;
}

// The suggestion row: the finder opened on the item at the caret, searched.
function openTagFinderFor(range) {
  const state = tagFinderState();
  hideTagSuggestions();
  state.typed = { start: range.start, end: range.end, text: els.prompt.value.slice(range.start, range.end) };
  state.query = range.query;
  showTagFinder(true);
  findTags(range.query);
}

// A result pressed: over the typed description while it is still where it
// was typed, else in or out at the caret.
function pickFoundTag(text, button) {
  const state = tagFinderState();
  const typed = state.typed;
  state.typed = null;
  // Still the whole comma item it was: "stupid hairs" is not "stupid hair".
  const prompt = els.prompt.value;
  const whole = typed && typed.text && prompt.slice(typed.start, typed.end) === typed.text
    && /(^|,)\s*$/.test(prompt.slice(0, typed.start)) && /^\s*(,|$)/.test(prompt.slice(typed.end));
  if (whole) {
    applyTagSuggestion({ text }, typed);
    button.focus({ preventScroll: true });
    return;
  }
  togglePromptTag(text, {});
}

function buildTagFinder() {
  const state = tagFinderState();
  els.tagFinderMenu.replaceChildren();
  const heading = document.createElement('strong');
  heading.className = 'tag-finder-heading';
  heading.id = 'tag-finder-heading';
  heading.textContent = 'Find a tag by what it shows';
  const form = document.createElement('form');
  form.className = 'tag-finder-row';
  form.setAttribute('role', 'search');
  form.setAttribute('aria-labelledby', heading.id);
  // Text, not search: Escape closes the finder, as it does the helper,
  // rather than first emptying the box.
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'tag-finder-input';
  input.maxLength = 100;
  input.placeholder = 'cowlick, stupid hair, sleeves over hands';
  input.setAttribute('aria-label', 'Describe what the tag shows');
  input.value = state.query;
  const find = document.createElement('button');
  find.type = 'submit';
  find.className = 'ghost tiny';
  find.textContent = 'Find';
  find.title = "Search Danbooru's tag wiki for these words";
  form.append(input, find);
  form.onsubmit = (event) => {
    event.preventDefault();
    findTags(input.value);
  };
  const status = document.createElement('p');
  status.className = 'tag-finder-status';
  status.setAttribute('role', 'status');
  const results = document.createElement('ul');
  results.className = 'tag-finder-results';
  results.setAttribute('aria-label', 'Tags found');
  const note = document.createElement('p');
  note.className = 'tag-note';
  note.textContent = "From Danbooru's tag wiki: general tags whose page mentions the words, "
    + 'most used first. Each one\'s first sentence says whether it is the one.';
  els.tagFinderMenu.append(heading, form, status, results, note);
  renderTagFinder();
}

function renderTagFinder() {
  const section = els.tagFinderMenu;
  const status = section.querySelector('.tag-finder-status');
  if (!status) return;
  const state = tagFinderState();
  status.textContent = state.status;
  const list = section.querySelector('.tag-finder-results');
  list.replaceChildren();
  for (const result of state.results) {
    const item = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tag-option tag-finder-option';
    button.dataset.tag = result.text;
    const name = document.createElement('span');
    name.className = 'tag-name';
    name.textContent = result.text;
    const count = document.createElement('span');
    count.className = 'tag-count';
    count.textContent = result.post_count == null ? '' : humanCount(result.post_count);
    button.append(name, count);
    button.onclick = () => pickFoundTag(result.text, button);
    const definition = document.createElement('span');
    definition.className = 'tag-finder-definition';
    definition.textContent = result.definition || 'Its wiki page opens with no sentence to show.';
    item.append(button, definition);
    list.appendChild(item);
  }
  list.hidden = !state.results.length;
  syncTagFinderSelection();
}

// A result already in the prompt shows pressed, as the helper's tags do.
function syncTagFinderSelection() {
  if (els.tagFinderMenu.hidden) return;
  const active = new Set(promptTagParts().map((tag) => tag.toLowerCase()));
  for (const button of els.tagFinderMenu.querySelectorAll('[data-tag]')) {
    const on = active.has(button.dataset.tag.toLowerCase());
    button.classList.toggle('active', on);
    button.setAttribute('aria-pressed', String(on));
  }
}

async function findTags(text) {
  const state = tagFinderState();
  const query = text.trim().replace(/\s+/g, ' ');
  const epoch = ++tagFinderEpoch;
  state.query = text;
  if (!query) {
    state.results = [];
    state.status = 'Type what the tag should show, such as cowlick.';
    renderTagFinder();
    return;
  }
  state.status = "Searching Danbooru's tag wiki…";
  renderTagFinder();
  let body;
  try {
    body = await api(`/api/tags/describe?q=${encodeURIComponent(query)}`);
  } catch (err) {
    body = { available: false, results: [], detail: err.message };
  }
  if (epoch !== tagFinderEpoch) return;
  state.results = body.results || [];
  if (!body.available) {
    state.status = `Danbooru unreachable, so nothing was found${body.detail ? `: ${body.detail}` : '.'}`;
  } else if (!state.results.length) {
    state.status = `No general tag's Danbooru wiki mentions “${query}”. Try fewer or other words.`;
  } else {
    const count = state.results.length === 1 ? 'One tag' : `${state.results.length} tags`;
    state.status = `${count} whose Danbooru wiki mentions “${query}”. Select one to insert it at the caret.`;
  }
  renderTagFinder();
}

// Under the prompt, as the helper is, and one at a time with it and the
// Examples menu. Opening puts the caret in its box, the words there to
// change; closing with focus inside hands it back to the prompt.
function showTagFinder(open) {
  if (open) {
    if (!els.tagsMenu.hidden) showTagHelper(false);
    if (!els.examplesMenu.hidden) showExamples(false);
  }
  const inside = els.tagFinderMenu.contains(document.activeElement);
  els.tagFinderMenu.hidden = !open;
  if (open) {
    buildTagFinder();
    els.tagFinderMenu.querySelector('.tag-finder-input').focus({ preventScroll: true });
  } else if (inside) els.prompt.focus({ preventScroll: true });
}

// Only where the guide names the Danbooru vocabulary (syncTagVocabulary).
function syncTagFinder() {
  if (!tagVocabularyEnabled) showTagFinder(false);
  else if (!els.tagFinderMenu.hidden) buildTagFinder();
}
