'use strict';

// The gallery's search from one image's facts: the lightbox's ⌕ beside a row
// adds that row's `key:value` terms to the search box, and "Search for this
// prompt" its words in place of the box's text, so finding the rest of a seed or a prompt is not a copy
// by hand. Only grammar the server parses (backend/search.py): a fact with no
// key - the negative prompt, the references, the variables - has no button.

// `search.FILTERS`' keys, held equal by tests/test_storage.py: a word whose
// key is one of these is a filter, anything else is prompt text.
const SEARCH_KEYS = new Set(['id', 'seed', 'steps', 'cfg', 'tcfg', 'mu', 'model', 'quant', 'lora', 'refine', 'strength', 'mask', 'inpaint', 'outpaint', 'edit', 'job', 'run', 'hash', 'date', 'size']);

// Explicit keyboard navigation, without changing the query, bin, run or picks.
// Reveal only when clipped; a visible search must not move the page.
function focusGallerySearch() {
  els.search.focus({ preventScroll: true });
  els.search.select();
  const box = els.search.getBoundingClientRect();
  if (box.top < document.querySelector('.topbar').getBoundingClientRect().bottom || box.bottom > innerHeight) {
    els.search.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'auto' });
  }
}

// The operand of an exclusion, by `search.exclusion`'s reading: one leading
// `-` and more after it that is not a `-` (`-blonde`, `-seed:1`), else null -
// a lone `-` and `--ar` stay free text.
function searchExclusion(word) {
  return word.length > 1 && word[0] === '-' && word[1] !== '-' ? word.slice(1) : null;
}

// The filter a search word is, by `split_search`'s reading: `key:value` with
// a known key and a value, else null (free text, `portrait:` included). It
// reads through one leading `-`: `-seed:1` is the seed filter, negated.
function searchKey(word) {
  const term = searchExclusion(word) ?? word;
  const at = term.indexOf(':');
  if (at < 1 || at === term.length - 1) return null;
  const key = term.slice(0, at).toLowerCase();
  return SEARCH_KEYS.has(key) ? key : null;
}

// `terms` added to the box's words. A filter replaces the box's terms with its
// key, negated or not, where the first of them stood: two seeds AND to
// nothing, so a second seed pressed is a change of seed, and `-seed:1` with
// seed:5 pressed is seed:5. A word already there (any case) is not added
// again, and a word pressed takes its own exclusion's place (`-fox` with fox
// pressed is fox); everything else the box held is kept, in its order.
function mergeSearch(current, terms) {
  const words = current.trim().split(/\s+/).filter(Boolean);
  for (const term of terms) {
    const key = searchKey(term);
    if (key) {
      const first = words.findIndex((w) => searchKey(w) === key);
      if (first < 0) { words.push(term); continue; }
      words[first] = term;
      for (let i = words.length - 1; i > first; i--) if (searchKey(words[i]) === key) words.splice(i, 1);
      continue;
    }
    const lower = term.toLowerCase();
    const negated = (w) => (searchExclusion(w) ?? '').toLowerCase() === lower;
    const there = words.some((w) => w.toLowerCase() === lower);
    const at = words.findIndex(negated);
    if (at >= 0 && !there) words[at] = term;
    for (let i = words.length - 1; i >= 0; i--) if (negated(words[i])) words.splice(i, 1);
    if (at < 0 && !there) words.push(term);
  }
  return words.join(' ');
}

// A prompt's words as the box's whole text: the box's filters and exclusions
// (`-key:value` and `-word`) kept, in their order, and every free word it held
// replaced - the prompt is one fact, as a seed is, so a second prompt pressed
// is a change of prompt, not both prompts' words ANDed to nothing. A word
// repeated in the prompt goes in once.
function mergePromptSearch(current, words) {
  const kept = current.trim().split(/\s+/).filter((w) => w && (searchKey(w) || searchExclusion(w) !== null));
  const seen = new Set();
  for (const word of words) {
    const lower = word.toLowerCase();
    if (!seen.has(lower)) { seen.add(lower); kept.push(word); }
  }
  return kept.join(' ');
}

// A prompt as the box's free text: its words, each a term the full-text index
// matches in any order. A word with no letter or digit (a lone comma, a "-")
// matches nothing of its own, and one that would read as a filter (`seed:3`
// in a prompt) is left out rather than turned into one. Leading hyphens are
// dropped, so `-style` is not written in as an exclusion; the index drops
// them anyway, so the word matches the rows the hyphenated one did.
function promptSearchWords(prompt) {
  return (prompt ?? '').split(/\s+/).map((w) => w.replace(/^-+/, ''))
    .filter((w) => w && /[\p{L}\p{N}]/u.test(w) && !searchKey(w));
}

// The image's local day as `date:` takes it - the day on this clock, as the
// server reads it (search.date_clause), not the UTC one in created_at.
function searchDay(rec) {
  const d = new Date(rec.created_at);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

// Each facts row's terms, by the row's name (factsList's), or none: a value
// with whitespace cannot be one search word, so it gets no button.
function factSearchTerms(rec) {
  const one = (key, value) => (value == null || /\s/.test(String(value)) ? null : [`${key}:${value}`]);
  const run = rec.run_id ? one('run', rec.run_id) : rec.job_id ? one('job', rec.job_id) : null;
  return {
    Model: one('model', rec.model_id),
    Size: [`size:${rec.width}x${rec.height}`],
    // As the row says them: True CFG only where it is a number of its own,
    // and "no Mu" as `mu:none` - `mu:0` is a different image.
    Sampling: [
      `steps:${rec.steps}`, `cfg:${rec.guidance_scale}`,
      ...(rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? [`tcfg:${rec.true_cfg}`] : []),
      rec.mu == null ? 'mu:none' : `mu:${rec.mu}`,
    ],
    LoRA: rec.lora ? one('lora', rec.lora) : null,
    // The other passes made from the same source.
    'Refined from': rec.refine != null ? [`refine:${rec.refine}`] : null,
    Seed: [`seed:${rec.seed}`],
    Created: [`date:${searchDay(rec)}`],
    Run: run,
  };
}

function searchButton(terms, text = '⌕', merge = mergeSearch) {
  const said = terms.join(' ');
  const b = document.createElement('button');
  b.type = 'button';
  b.className = text === '⌕' ? 'lb-search' : 'lb-link';
  b.textContent = text;
  b.title = `Add ${said} to the gallery search`;
  if (text === '⌕') b.setAttribute('aria-label', `Search the gallery for ${said}`);
  b.onclick = () => showGallerySearch(merge(els.search.value, terms));
  return b;
}

// The ⌕ at the end of each row that has terms (`factsList(rec, { search })`).
function addFactSearches(dl, rec) {
  const terms = factSearchTerms(rec);
  for (const div of dl.children) {
    const wanted = terms[div.firstChild.textContent];
    if (wanted?.length) div.lastChild.append(' ', searchButton(wanted));
  }
}

// The prompt's line above the facts: its words as the box's text.
function promptSearchLine(rec) {
  const line = document.createElement('p');
  line.className = 'lb-prompt-search';
  const words = promptSearchWords(rec.prompt);
  line.hidden = !words.length;
  if (words.length) {
    const b = searchButton(words, '⌕ Search for this prompt', mergePromptSearch);
    b.title = 'Search the gallery for the prompt\'s words in place of the search\'s text, its filters kept: images whose prompt has every one of them';
    line.append(b);
  }
  return line;
}

// The gallery under `search`, from a press in the lightbox: closed, the box
// written and the query asked again only where it changed, and the box
// focused where the results are - the passes link's way (showPasses).
function showGallerySearch(search) {
  if (els.lightbox.open) closeLightbox();
  // The search is the gallery's: from the recycle bin, back to it.
  const left = leaveBin();
  if (left || els.search.value.trim() !== search) {
    els.search.value = search;
    resetGallery();
  }
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelector('.gallery-head').scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  els.search.focus({ preventScroll: true });
}
