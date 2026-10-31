// AuraDream's page script: the prompt check, encoders, order and findings.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------- prompt check

// The check describes one text: the prompt as `collect()` sent it - trimmed -
// with the negative prompt beside it, whether it runs, and the references counted.
// Every offset in the answer counts from that text's first character, and
// every Use and Apply is a whole replacement worked out from it. So once
// either box holds something else, or the references change, the panel dims,
// the buttons that would write an answer about the old text over the new one
// go dead, and a finding's words are selected only while they are still there.
let analysisResult = null;  // { source, negative, effectiveNegative, reference, body: PromptAnalysis }
let analysisEpoch = 0;
// Dismissal retires reveal/focus only: accepted background findings and a
// critic pre-check keep their existing result and computation ownership.
let analysisDismissal = 0;
let analysisStale = false;
// The critic's answer is always about the checked text: asking checks the box
// again first when it has moved on, and a check of other text drops the answer.
let critiqueResult = null;  // { body: PromptCritique }
let critiqueEpoch = 0;
// Why the critic last failed to answer, shown in its block until the next
// ask or a check of other text. Never a critiqueResult: Use and Apply read
// that as an answer, and a failure has nothing to apply.
let critiqueFailure = null;
let criticClockTimer = 0;
// Which optional versions a comparison takes. An untick outlives the next
// check, so a rewrite that lands later does not tick itself back in.
const compareChoice = { reordered: true, rewrite: true, yours: true };
let comparePreviewTimer;
let comparePreviewEpoch = 0;

// What each finding rests on, by `PromptFinding.source`. The label is the
// sourcing rule made visible: Guide and Authors quote the checkpoint's authors,
// Encoder is what the pipeline's own tokenizer does, Heuristic is this app's
// pattern matching - its checks on the critic's rewrite included - and says so.
// A critic's cite of words not the authors' own carries its note (issueItem),
// and one of the fallback's url-less guide sentences reads General, not Guide.
const SOURCE_LABELS = {
  guide: 'Guide', author: 'Authors', encoder: 'Encoder', heuristic: 'Heuristic',
};
const STYLE_NAMES = { prose: 'prose', tags: 'a tag list', hybrid: 'tags and prose', xml: 'XML', json: 'JSON' };
const PART_LABELS = [
  ['subject', 'Subject'], ['action', 'Action'], ['setting', 'Setting'], ['style', 'Style'],
  ['lighting', 'Lighting'], ['camera', 'Camera'], ['text_to_render', 'Text to render'],
];

// The rules' word count in a reader's words. Each Han or kana character
// counts as one word there (`prompt_rules.word_count`: Chinese and Japanese
// space no words), and "300 words" read as a miscount of a 300-character
// Chinese paragraph; the server says how many of `words` are such characters.
function wordsLine(body) {
  const characters = body.characters ?? 0;
  const words = body.words - characters;
  if (!characters) return plural(body.words, 'word');
  if (!words) return plural(characters, 'character');
  const both = [plural(characters, 'character'), plural(words, 'word')];
  return (characters >= words ? both : both.reverse()).join(' and ');
}

function span(className, text = '') {
  const el = document.createElement('span');
  el.className = className;
  el.textContent = text;
  return el;
}

function note(text, className = 'analysis-note') {
  const p = document.createElement('p');
  p.className = className;
  p.textContent = text;
  return p;
}

// Sends what Generate would send: the negative prompt, the CFG and True CFG
// decide whether that negative ever runs, and the answer depends on it.
async function checkPrompt(payload = collect()) {
  if (modelChangeHold()) { sayModelChangeHold(); return null; }
  if (!payload.prompt) { showError('Enter a prompt first.'); els.prompt.focus(); return null; }
  const epoch = ++analysisEpoch;
  const dismissal = analysisDismissal;
  const effectiveNegative = analysisNegative(payload);
  // Instant once the tokenizers are cached; a first use fetches them.
  els.analyze.textContent = 'Checking…';
  els.analyze.setAttribute('aria-busy', 'true');
  try {
    const body = await api('/api/prompt/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (epoch !== analysisEpoch) return null;  // superseded, or the panel was cleared
    // The critic's answer is about the text checked last, negative prompt and
    // all, and so is a failure to answer.
    if (!analysed(payload, effectiveNegative)) critiqueResult = critiqueFailure = null;
    analysisResult = {
      source: payload.prompt, negative: payload.negative_prompt ?? '', effectiveNegative,
      reference: [...(payload.reference ?? [])], body,
    };
    showError('');
    analysisMessage('');
    renderAnalysis(dismissal === analysisDismissal);
    return analysisResult;
  } catch (err) {
    if (epoch === analysisEpoch) showError(err.message);
    return null;
  } finally {
    if (epoch === analysisEpoch) resetCheckButton();
  }
}

function resetCheckButton() {
  els.analyze.textContent = 'Check';
  els.analyze.removeAttribute('aria-busy');
}

// Check, pressed. On a two-column page the panel sits above the results (or
// beside them, from 1800 px), some forty Tab stops past the button, and only
// its summary is announced; so the answer takes focus, at its heading, while
// focus is still on the button. Under 901 px the panel is three stops away,
// and focus stays where it is.
async function checkFromButton() {
  const dismissal = analysisDismissal;
  const result = await checkPrompt();
  if (result && dismissal === analysisDismissal && !els.promptAnalysis.hidden
      && document.activeElement === els.analyze
      && els.promptAnalysis.parentElement === els.tagValidationWide) focusAnalysisHeading();
}

// The panel's heading (tabindex=-1): where focus goes when the button that had
// it is far from the answer, or gone. focus() scrolls it into view - the page;
// docked, the head sticks inside the panel's own scroll and so is always "in
// view" there, and the panel is put back at its top by hand, as focusing the
// heading did before the head stuck.
function focusAnalysisHeading() {
  els.promptAnalysis.scrollTop = 0;
  els.promptAnalysis.querySelector('.analysis-head h2').focus();
}

function renderAnalysis(reveal = true) {
  const { body } = analysisResult;
  // Keep a dismissed view hidden while accepting its background result.
  // First: the comparison's preview is only fetched for a panel on show.
  if (reveal) els.promptAnalysis.hidden = false;
  els.analysisModel.textContent = checkpointName(body.model_id);
  els.analysisModel.title = body.model_id;
  const warns = body.findings.filter((f) => f.severity === 'warn').length;
  const notes = body.findings.length - warns;
  // A documented order nothing here read must not sit beside "nothing to
  // flag" as if it had passed: a prose prompt with its subject last on FLUX
  // got the same panel as one with it first.
  const unchecked = !!body.order && !body.order_checked;
  // Nor may one the rules placed only partly - a tag with no category, never
  // looked up (offline, past the lookup's 64, a prose prompt) or unknown to
  // the tag index: a character after a general tag got the same panel as one
  // before it. The server's own predicate (`order_detail`), which the critic
  // reads too, never a guess of the page's.
  const partly = body.order_checked && !!body.order_detail;
  // Where the guide states no order, a category is read by one rule only -
  // the artist check (RouWei's "by "; `lookup_items`) - and it read none (the
  // index unreachable, or no usable reply: `categories_unread`) or not all
  // (past the lookup's 64). "Not every tag" understated an outage in which
  // no tag was checked.
  const artists = !body.order && !!body.categories_detail;
  const noArtists = artists && !!body.categories_unread;
  const counts = !body.findings.length
    ? (unchecked ? 'nothing flagged · order not checked' : partly || artists ? 'nothing flagged' : 'nothing to flag')
    : `${warns ? plural(warns, 'warning') : 'no warnings'}${notes ? ` · ${plural(notes, 'note')}` : ''}`;
  els.analysisSummary.textContent = `${wordsLine(body)}, ${STYLE_NAMES[body.style] ?? body.style} · ${counts}`
    + (partly ? ' · order partly checked' : noArtists ? ' · artists not checked' : artists ? ' · artists not all checked' : '');
  renderEncoders(body);
  renderOrder(body);
  renderFindings(els.analysisFindings, body.findings, {
    select: (f) => selectAnalysed(f.field, f.start, f.end),
    apply: applyFinding,
    // The artist case's reason is the note right under this line, so the
    // line does not say it a second time.
    empty: unchecked
      ? "Nothing flagged by the rules for this checkpoint's guide - but its word order was not checked (see Order)."
      : partly ? "Nothing flagged by the rules for this checkpoint's guide - but its word order was only partly checked (see Order)."
      : artists ? "Nothing flagged by the rules for this checkpoint's guide."
      : "Nothing to flag against this checkpoint's guide.",
  });
  if (artists) {
    els.analysisFindings.appendChild(note(
      `${noArtists ? 'No tag was checked' : 'Not every tag was checked'} for an artist name: ${body.categories_detail}.`));
  }
  renderCritic();
  renderCompare();
  // The box may have moved on while the request was out.
  syncAnalysisStale(true);
}

// Whether `payload` - collect()'s shape - is what the check read: the prompt,
// the negative prompt as it would run, and the reference images, in order:
// klein's guide counts them, to name the "image 3" a prompt never mentions.
// The grid may keep the negative editor enabled for its higher-CFG cells,
// while Generate, Check and Compare still use the panel's single CFG. Read
// that scale (or True CFG), not the editor's enabled state. Capture the result
// before awaiting a check or critique: a later settings change must not relabel
// the earlier answer as though its negative ran under the new settings.
// Keep the raw negative too: rules can distinguish typed but inactive text,
// and an Apply must never replace edits in the enabled grid-only editor.
function analysisNegative(payload) {
  const scale = trueCfgSupported ? payload.true_cfg : payload.guidance_scale ?? lastDefaults?.guidance_scale;
  return negativeSupported && scale != null && scale > cfgThreshold ? payload.negative_prompt ?? '' : '';
}

function analysed(payload, effectiveNegative = analysisNegative(payload)) {
  return promptsAnalysed(payload, effectiveNegative) && (payload.reference ?? []).join() === analysisResult.reference.join();
}

const promptsAnalysed = (payload, effectiveNegative = analysisNegative(payload)) =>
  !!analysisResult && payload.prompt === analysisResult.source
  && (payload.negative_prompt ?? '') === analysisResult.negative
  && effectiveNegative === analysisResult.effectiveNegative;

// Called on every change to either prompt box - typing, every programmatic
// write through promptReplaced(), and syncCfg() for the knobs that grey the
// negative one - and to the references, so it only compares, and touches the
// page when the answer flips, or when the badge's words do.
function syncAnalysisStale(force = false) {
  const payload = collect();
  const stale = !!analysisResult && !analysed(payload);
  // Whether the negative box is greyed changes without the check going stale
  // (an empty box sends nothing either way), so this runs every time.
  syncNegativeApply();
  // Which moved on, and the badge says so: the references alone would leave
  // "prompt changed" pointing at a prompt nobody touched.
  const badge = stale && promptsAnalysed(payload) ? 'references changed since' : 'prompt changed since';
  if (els.analysisStale.textContent !== badge) els.analysisStale.textContent = badge;
  if (stale === analysisStale && !force) return;
  analysisStale = stale;
  els.promptAnalysis.classList.toggle('stale', stale);
  els.analysisStale.hidden = !stale;
  // Check again hides once the check is current again; a keyboard user who
  // pressed it would be left on <body>, so the panel's heading takes focus.
  if (!stale && !els.promptAnalysis.hidden && document.activeElement === els.analysisAgain) focusAnalysisHeading();
  els.analysisAgain.hidden = !stale;
  for (const button of els.promptAnalysis.querySelectorAll('[data-writes]')) syncWriteButton(button);
  syncCompareRun();
}

// A negative prompt written into a greyed box never runs, and the check -
// which sends nothing from a greyed box - would answer the same again: so its
// Apply is dead while the box is, and says what would wake it.
function syncNegativeApply() {
  for (const button of els.promptAnalysis.querySelectorAll('[data-negative]')) {
    syncWriteButton(button);
    const off = button.parentElement.querySelector('.negative-off');
    off.hidden = !els.negative.disabled;
    off.textContent = `${NEGATIVE_GREYED}: it needs ${trueCfgSupported ? 'True CFG' : 'CFG'} above ${cfgThreshold}.`;
  }
  for (const show of els.promptAnalysis.querySelectorAll('[data-negative-show]')) syncNegativeShow(show);
}

// A negative prompt finding's Show, while the box is greyed: held, with the
// reason as its title - aria-disabled, not disabled, so it keeps focus and a
// press says why (selectIn), where it did nothing at all.
function syncNegativeShow(show) {
  if (els.negative.disabled) show.setAttribute('aria-disabled', 'true');
  else show.removeAttribute('aria-disabled');
  show.title = els.negative.disabled ? NEGATIVE_UNSHOWN : show.dataset.title;
}

const writeBlocked = (button) => analysisStale || ('negative' in button.dataset && els.negative.disabled);

// Why a write is dead, in its title: a dead Use rewrite kept "Replace the
// prompt with the rewrite", an offer it was refusing.
const STALE_REASON = 'The prompt or the references changed since this check; Check again first';
const NEGATIVE_GREYED = 'The negative prompt is greyed at these settings, so nothing written there would run';
const NEGATIVE_UNSHOWN = 'The negative prompt is greyed at these settings, so its words cannot be selected';

function syncWriteButton(button) {
  button.disabled = writeBlocked(button);
  button.title = analysisStale ? STALE_REASON
    : button.disabled ? NEGATIVE_GREYED
    : button.dataset.title;
}

// Close keeps what was found - the next Check replaces it anyway - and hands
// focus back to the button that opened the panel; to the prompt while that
// button waits out the critic's own check, since a disabled one takes no focus
// and it would fall to <body>.
function closeAnalysis() {
  analysisDismissal += 1;
  els.promptAnalysis.hidden = true;
  (els.analyze.disabled ? els.prompt : els.analyze).focus();
}

// Escape closes it too, once nothing nearer took the key (app.js's keydown:
// a dialog, a menu, a suggestion list, a search box emptying itself). Focus
// in the panel, which would fall to <body> with it, goes back as Close's
// does; focus anywhere else - the prompt being typed in - stays where it is.
function escapeAnalysis() {
  const focused = document.activeElement;
  if (!focused || focused === document.body || els.promptAnalysis.contains(focused)) closeAnalysis();
  else {
    analysisDismissal += 1;
    els.promptAnalysis.hidden = true;
  }
}

// Forget the check: another checkpoint's tokenizers and guide, or a Reset. An
// answer still on its way is dropped when it lands.
function clearAnalysis() {
  analysisEpoch += 1;
  critiqueEpoch += 1;
  analysisResult = null;
  critiqueResult = null;
  critiqueFailure = null;
  analysisStale = false;
  resetCheckButton();
  analysisMessage('');
  els.promptAnalysis.hidden = true;
  els.promptAnalysis.classList.remove('stale');
  for (const box of [els.analysisEncoders, els.analysisOrder, els.analysisFindings,
    els.criticResult, els.compareOptions]) box.replaceChildren();
  els.comparePreview.textContent = '';
  els.compareOwn.value = '';
  els.compareOwnNote.hidden = true;
  syncCompareRun();
}

// The panel's own messages - the words a finding names no longer where the
// check found them, a critic's answer set aside or never asked for - in the
// slot under Ask the critic, between the findings and the answer they are
// about. In #error they sat under Generate, a column away. The slot is a
// live region: unhidden before it is written, and never rewritten with the
// same words. Without the critic's section (no critic here, or the panel
// closed meanwhile) they go to #error as before.
function analysisMessage(text) {
  const slot = els.analysisStatus;
  if (!text) {
    slot.hidden = true;
    slot.textContent = '';
    return;
  }
  if (els.promptAnalysis.hidden || els.analysisCritic.hidden) { showError(text, true); return; }
  slot.hidden = false;
  if (slot.textContent !== text) slot.textContent = text;
}

// The analysed text of a field, which the offsets count into.
const analysedText = (field) =>
  (field === 'negative_prompt' ? analysisResult?.negative : analysisResult?.source) ?? '';

function selectAnalysed(field, start, end) {
  return selectIn(field === 'negative_prompt' ? els.negative : els.prompt, analysedText(field), start, end);
}

// Offsets count code points from the analysed text's first character; the box
// may hold it behind whitespace `collect()` trimmed off, or hold other text by
// now. The words are selected only while they are still where the check found
// them - and never in a greyed box, which takes no focus or selection: that
// says why, as words gone from their place do, where it said nothing.
function selectIn(el, text, start, end) {
  if (el.disabled) {
    analysisMessage(el === els.negative ? `${NEGATIVE_UNSHOWN}: it needs ${trueCfgSupported ? 'True CFG' : 'CFG'} above ${cfgThreshold}.`
      : 'That box is greyed, so its words cannot be selected.');
    return false;
  }
  const [from, to] = [units(text, start), units(text, end)];
  const raw = el.value;
  const lead = raw.length - raw.trimStart().length;
  if (raw.slice(lead + from, lead + to) !== text.slice(from, to)) {
    analysisMessage('Those words are no longer where the check found them; Check again.');
    return false;
  }
  el.focus();
  el.setSelectionRange(lead + from, lead + to);
  scrollToSelection(el);
  flashBox(el);
  return true;
}

// The box the words were selected in is outlined for a moment (600 ms, the
// stylesheet's `.box-flash`; not at all under reduced motion): a finding in
// the column beside the panel selected words a column away, and a selection
// in an unfocused-looking box was easy to miss. A reference chip a finding's
// "image N" goes to is outlined the same way. Restarted on every click.
let flashedBox = null;
let flashTimer = 0;
function flashBox(el) {
  clearTimeout(flashTimer);
  flashedBox?.classList.remove('box-flash');
  void el.offsetWidth;  // a restart, not a class already there
  el.classList.add('box-flash');
  flashedBox = el;
  flashTimer = setTimeout(() => { el.classList.remove('box-flash'); flashedBox = null; }, 600);
}

// A selection scrolled out of its box looks like a click that did nothing,
// and setSelectionRange scrolls nothing in a box that has focus already: a
// long prompt just typed sits scrolled to its end, and the words a finding
// names are near the start. So the box is scrolled to put them a third of
// the way down, measured on a hidden copy of the box laid out as it is - its
// width inside the scrollbar, its padding and font, its wrapping - holding
// the text up to the selection and a marker where the selection starts.
function scrollToSelection(el) {
  const css = getComputedStyle(el);
  const copy = document.createElement('div');
  for (const prop of ['paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft', 'fontFamily',
    'fontSize', 'fontWeight', 'fontStyle', 'lineHeight', 'letterSpacing', 'wordSpacing',
    'textTransform', 'tabSize', 'overflowWrap', 'wordBreak']) copy.style[prop] = css[prop];
  copy.style.boxSizing = 'content-box';
  copy.style.width = `${el.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight)}px`;
  copy.style.whiteSpace = 'pre-wrap';
  copy.style.position = 'absolute';
  copy.style.top = copy.style.left = '0';
  copy.style.visibility = 'hidden';
  copy.textContent = el.value.slice(0, el.selectionStart);
  const marker = document.createElement('span');
  marker.textContent = '\u200b';
  copy.appendChild(marker);
  document.body.appendChild(copy);
  el.scrollTop = Math.max(0, marker.offsetTop - el.clientHeight / 3);
  copy.remove();
}

// A button that writes an answer into the panel's fields. Each is worked out
// from the checked text, so every one goes dead while either box holds other
// text, its title saying so in place of what it offers.
function writeButton(label, title, onClick, { negative = false } = {}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost tiny';
  button.textContent = label;
  button.dataset.title = title;
  button.dataset.writes = '';
  if (negative) button.dataset.negative = '';
  syncWriteButton(button);
  button.onclick = onClick;
  return button;
}

// A whole box rewritten the way typing would write it, so Ctrl+Z brings back
// what it held. Assigning `.value` empties a textarea's undo history, and the
// check that follows makes the new text the comparison's original: what was
// typed would be nowhere on the page. The insert fires `input` like typing;
// a greyed box, or a browser that will not insert, gets the plain write.
// `replacingBox` is true while it writes, for a listener to tell the page's
// own whole-box writes - an Example, Reuse, Reset, a Use - from typing:
// ERNIE's Expand holds a result the user typed over, never one they replaced.
let replacingBox = false;
function replaceBoxText(el, text) {
  replacingBox = true;
  try {
    el.focus();
    el.select();
    const inserted = document.activeElement === el && document.execCommand('insertText', false, text);
    if (!inserted) el.value = text;
  } finally {
    replacingBox = false;
  }
}

// Use and Apply: the prompt replaced the way Expand replaces it - the box, its
// translation, the grid preview, the remembered panel, the chips, the caret -
// then checked again, which is instant, so the panel describes the new text
// rather than dimming over the old.
function usePromptText(text) {
  replaceBoxText(els.prompt, text);
  // The insert's `input` asked for suggestions at the caret, as typing does;
  // a replacement is not typing, and none open over the new text.
  clearTimeout(tagSuggestionTimer);
  hideTagSuggestions();
  promptReplaced();
  schedulePreview();
  persistPanel();
  scheduleTagValidation(0);
  els.prompt.focus();
  els.prompt.setSelectionRange(els.prompt.value.length, els.prompt.value.length);
  void checkPrompt();
}

// Use rewrite and Use reordered (the critic's, and Check's own reordering),
// and a rule finding's Apply on the prompt (applyFinding):
// a version of the prompt in its place, and the prompt it replaces in Your
// version while that box is empty - where the comparison takes it, so after
// the re-check the two run on the same seeds. Before, the re-check made the
// version the comparison's original, and what it replaced was on the page
// only as Ctrl+Z. A box with words of the user's own is left alone.
const KEEPS_PROMPT = ' The prompt it replaces goes into Your version, if that is empty, to compare on the same seeds; Ctrl+Z brings it back.';

function useVersion(text, button) {
  if (!els.compareOwn.value.trim()) {
    els.compareOwn.value = analysisResult.source;
    compareChoice.yours = true;
    els.compareOwnNote.textContent = `Your version holds the prompt before ${button} - Run comparison tests both on the same seeds.`;
    els.compareOwnNote.hidden = false;
  }
  usePromptText(text);
}

// A rule's suggestion is its field rewritten whole - quotes added, tags moved,
// the recommended negative prompt - so it applies in one click, to its field.
// On the prompt it is a version like Use rewrite's, and keeps the prompt it
// replaces the same way (useVersion); the negative prompt has no Your
// version to keep it in, and Ctrl+Z is its way back.
function applyFinding(finding) {
  if (finding.field !== 'negative_prompt') { useVersion(finding.suggestion, 'Apply'); return; }
  if (els.negative.disabled) return;  // its button is dead then; this is the backstop
  replaceBoxText(els.negative, finding.suggestion);
  syncAnalysisStale();
  schedulePreview();
  persistPanel();
  void checkPrompt();
}

// ----------------------------------------------------------------- encoders

// One row per encoder and field: the count against what the pipeline keeps,
// with the part it never reads drawn past the end of the part it does.
function renderEncoders(body) {
  const box = els.analysisEncoders;
  box.replaceChildren();
  for (const report of body.encoders) box.appendChild(encoderRow(report));
  if (body.encoders_detail || !body.encoders.length) {
    // An unmeasured budget is said so, never shown as one that passed.
    box.appendChild(note(body.encoders_detail || 'No tokenizer is known for this checkpoint, so nothing was counted.'));
  }
}

function encoderRow(report) {
  const row = document.createElement('div');
  row.className = 'encoder-row';
  const cut = report.cut != null;
  row.classList.toggle('cut', cut);
  // FLUX.1's CLIP-L feeds one pooled vector while T5 reads every word, so a
  // cut there costs the words far less: drawn as a caution, not a loss. The
  // server's call, as its finding's severity is - any other role's cut is a
  // loss like one without a role.
  row.classList.toggle('pooled', cut && report.pooled_only);
  const name = span('encoder-name', report.field === 'negative_prompt' ? `${report.name} · negative` : report.name);
  if (report.source) name.title = `Tokenizer: ${report.source}`;
  const count = span('encoder-count', `${report.tokens} / ${report.budget}`);
  const bar = span('encoder-bar');
  const scale = Math.max(report.tokens, report.budget, 1);
  const read = span('read');
  read.style.width = `${(Math.min(report.tokens, report.budget) / scale) * 100}%`;
  const lost = span('lost');
  lost.style.width = `${(Math.max(0, report.tokens - report.budget) / scale) * 100}%`;
  bar.append(read, lost);
  bar.setAttribute('role', 'img');
  bar.setAttribute('aria-label', cut
    ? `${report.tokens} tokens, ${report.tokens - report.budget} past the ${report.budget} it reads`
    : `${report.tokens} of ${report.budget} tokens`);
  row.append(name, count, bar);
  if (report.role || cut) {
    const line = document.createElement('p');
    line.className = 'encoder-note';
    if (report.role) line.append(report.role);
    if (cut) {
      if (report.role) line.append(' · ');
      line.append(cutButton(report));
    }
    row.appendChild(line);
  }
  return row;
}

// Selects everything the encoder never reads, from the cut to the end.
function cutButton(report) {
  const text = analysedText(report.field);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'analysis-link';
  button.textContent = `stops reading at "${excerpt(text.slice(units(text, report.cut)))}"`;
  button.title = `Select what ${report.name} never reads`;
  button.onclick = () => selectAnalysed(report.field, report.cut, [...text].length);
  return button;
}

// -------------------------------------------------------------------- order

function renderOrder(body) {
  const box = els.analysisOrder;
  box.replaceChildren();
  if (!body.order) {
    // Not "the authors document none": Neta Lumina's prompt book, NoobAI's
    // and RouWei's cards and the rewriter instructions give orders the guide leaves out.
    // What is missing is this app's check - and the authors' orders, where
    // they wrote some, are shown as theirs.
    // "Its guide states none" only where no author order follows: NoobAI's and
    // RouWei's ? text quotes the orders listed below, which the sentence then
    // contradicted. The Tags menu says the same (tag-chips.js, tagOrderLines).
    box.appendChild(note(body.author_orders?.length
      ? 'This app checks no word order for this checkpoint.'
      : 'This app checks no word order for this checkpoint: its guide states none.'));
    for (const order of body.author_orders ?? []) {
      const line = document.createElement('div');
      line.className = 'order-author';
      const text = note(order.text, 'order-author-text');
      line.append(text, sourceLabel('author', order.note, order.url));
      if (order.note) line.append(note(order.note, 'analysis-note sub'));
      box.appendChild(line);
    }
    box.appendChild(note(body.author_orders?.length
      ? "The authors' own texts give the order above - information, not a rule this app checks. To see whether it matters here, write a version in it under Compare and run it on your own seeds."
      : 'To see whether order matters here, write a version in another order under Compare and run it on your own seeds.'));
    return;
  }
  const line = document.createElement('div');
  line.className = 'order-line';
  const chain = document.createElement('ol');
  chain.className = 'order-chain';
  chain.title = body.order.cite;
  chain.setAttribute('aria-label', `Documented order: ${body.order.steps.join(', then ')}`);
  for (const step of body.order.steps) {
    const li = document.createElement('li');
    li.append(span('order-step', step));
    chain.appendChild(li);
  }
  line.append(chain, sourceLabel('guide', body.order.cite, body.order.url));
  box.appendChild(line);
  if (!body.order_checked) {
    // Only a tag checkpoint's comma tags take slots here; a sentence's order,
    // an XML prompt's blocks, are the guide's words and nothing reads them.
    // Said of the app, not of "Check" in the third person - and of a tag
    // checkpoint's tags, not of comma lists: a comma tag prompt on FLUX is
    // not placed, and a prose one on Animagine is.
    box.appendChild(note(`Not checked: this app checks word order only in a tag checkpoint's comma-separated tags. ${promptCritiqueSupported
      ? 'Ask the critic, or test'
      : 'Test'} a reordered version under Compare on your own seeds.`));
  } else if (body.order_detail) {
    // Placed, but an item with no category keeps every slot a category could
    // have put it in - which slots those are is the guide's (Animagine's order
    // has no artist) - so it can sit out of order unflagged. The server names
    // the items and why each has none.
    box.appendChild(note(`Partly checked: ${body.order_detail}. Any of those may sit out of order unflagged.`));
  }
  if (body.reordered_detail) box.appendChild(note(body.reordered_detail));
  // The prompt's own items moved into that order, nothing added or removed -
  // shown whole, since it is read before it is used.
  if (body.reordered) {
    const moved = document.createElement('div');
    moved.className = 'order-reordered';
    const quote = note(body.reordered, 'analysis-quote');
    quote.dataset.version = 'reordered';
    moved.append(span('analysis-note', 'In that order:'), quote,
      writeButton('Use reordered',
        `Replace the prompt with the same words in the documented order, then check it.${KEEPS_PROMPT}`,
        () => useVersion(body.reordered, 'Use reordered')));
    box.appendChild(moved);
  }
}

// ----------------------------------------------------------------- findings

function sourceLabel(source, cite, url) {
  const label = document.createElement(url ? 'a' : 'span');
  label.className = 'analysis-source';
  label.textContent = SOURCE_LABELS[source] ?? source;
  label.title = cite || (source === 'heuristic'
    ? "This app's own pattern check, not a rule from the guide" : '');
  if (url) {
    label.href = url;
    label.target = '_blank';
    label.rel = 'noopener noreferrer';
  }
  return label;
}

// Warnings first - the server sorts them so, and a stable sort keeps its order
// within each. `select` puts a finding's words in their box, which `where`
// names for its Show; `apply` writes its suggestion; a list without `apply`
// (the rewrite's) only reports, and `carried` names whether the prompt has the
// same finding.
const inPrompt = (f) => (f.field === 'negative_prompt' ? 'the negative prompt' : 'the prompt');

function renderFindings(box, findings, { select, apply, empty, carried = null, where = inPrompt }) {
  box.replaceChildren();
  if (!findings.length) { box.appendChild(note(empty)); return; }
  const list = document.createElement('ul');
  list.className = 'analysis-list';
  const ranked = [...findings].sort((a, b) => (a.severity === 'warn' ? 0 : 1) - (b.severity === 'warn' ? 0 : 1));
  for (const finding of ranked) list.appendChild(findingItem(finding, { select, apply, carried, where }));
  box.appendChild(list);
}

// A cite this short goes in the line, after the claim it supports, where the
// badge's tooltip alone hid it behind a hover; a longer one - an encoder's,
// a paragraph - stays the tooltip.
const INLINE_CITE = 80;

function findingItem(finding, { select, apply, carried, where }) {
  const li = document.createElement('li');
  li.className = `analysis-item ${finding.severity}`;
  const mark = span('analysis-mark', finding.severity === 'warn' ? '!' : 'i');
  mark.setAttribute('aria-hidden', 'true');
  const message = span('analysis-message');
  // What a finding says, with its severity in words: its Show's description.
  const said = span('analysis-said');
  said.append(span('analysis-sr', finding.severity === 'warn' ? 'Warning: ' : 'Note: '), findingText(finding));
  message.append(said);
  if ((finding.source === 'guide' || finding.source === 'author') && finding.cite
      && [...finding.cite].length < INLINE_CITE) {
    const cite = document.createElement('q');
    cite.className = 'analysis-cite';
    cite.textContent = finding.cite;
    message.append(' ', cite);
  }
  if (select && finding.start != null && finding.end != null) {
    message.append(makeSelectable(li, () => select(finding), said, where(finding),
      finding.field === 'negative_prompt'));
  }
  if (carried) message.append(span('analysis-carried', carried(finding)));
  li.append(mark, message, sourceLabel(finding.source, finding.cite, finding.url));
  // The encoder's suggestion is advice about the text; a rule's is the whole
  // field rewritten, which applies in one click - but not to a rewrite the
  // page only shows, nor twice: the order finding's suggestion is the Order
  // section's reordering, byte for byte, and one action gets one button.
  if (finding.suggestion && apply && finding.field !== 'negative_prompt'
      && finding.suggestion === analysisResult?.body.reordered) {
    li.appendChild(note('Same as Use reordered above.', 'analysis-note analysis-extra'));
  } else if (finding.suggestion && (finding.source === 'encoder' || apply)) {
    const extra = document.createElement('div');
    extra.className = 'analysis-extra';
    if (finding.source === 'encoder') extra.textContent = finding.suggestion;
    else {
      const negative = finding.field === 'negative_prompt';
      extra.append(writeButton(negative ? 'Apply to the negative prompt' : 'Apply',
        `Replace the ${negative ? 'negative prompt' : 'prompt'} with: ${finding.suggestion}${negative ? '' : `\n${KEEPS_PROMPT.trim()}`}`,
        () => apply(finding), { negative }));
      if (negative) {
        // Said beside the dead button while the box is greyed.
        const off = span('negative-off');
        off.hidden = true;
        extra.append(off);
      }
    }
    li.appendChild(extra);
  }
  return li;
}

// A finding's words. Klein's references rule names the images the prompt
// never mentions - "never says what to take from image 2, 3" - and each of
// those numbers is the reference chip of that number in the panel, which the
// words take you to: "image 2" and "3" are links, their words unchanged.
const REFERENCE_NAMES = /\bimage (\d+(?:, \d+)*)/;

function findingText(finding) {
  const match = finding.rule === 'guide.references' && REFERENCE_NAMES.exec(finding.message);
  if (!match) return finding.message;
  const text = document.createDocumentFragment();
  text.append(finding.message.slice(0, match.index));
  match[1].split(', ').forEach((n, i) => {
    if (i) text.append(', ');
    text.append(referenceLink(+n, i ? n : `image ${n}`));
  });
  text.append(finding.message.slice(match.index + match[0].length));
  return text;
}

// To reference chip N's ×, in the panel: focused, which brings it into view
// there (revealInPanel, or the page's own scroll in one column). The chips
// may have changed since the check - the panel is stale then, and says so -
// so a number past the last chip says that image is not attached now. A
// link that finds its chip takes back only that sentence of its own
// (`notAttachedSaid`): the slot may hold another one the user still needs,
// such as why the critic's answer cannot be applied.
let notAttachedSaid = '';

function referenceLink(n, label) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'analysis-link reference-link';
  button.textContent = label;
  button.title = `Go to reference image ${n} in the panel`;
  button.setAttribute('aria-label', `Reference image ${n}`);
  button.onclick = () => {
    if (n > referenceIds.length) {
      const have = referenceIds.length ? plural(referenceIds.length, 'reference image') : 'no reference images';
      notAttachedSaid = `Image ${n} is not attached: the panel has ${have} now.`;
      analysisMessage(notAttachedSaid);
      return;
    }
    if (notAttachedSaid && els.analysisStatus.textContent === notAttachedSaid) analysisMessage('');
    notAttachedSaid = '';
    focusReference(n - 1);
    // A pointer's press leaves the × no focus ring to find it by.
    flashBox(els.referenceList.querySelectorAll('.chip')[n - 1]);
  };
  return button;
}

// A finding's or a critic issue's words, selected in their box: its Show
// button - the Refine dialog's check's word for the same act - for the
// keyboard and a screen reader, described by what the item says (`said`), and
// a click anywhere else on the item for the pointer, though not one meant for
// its link or another button. The item itself is no tab stop: focusable, it
// was read as a list item titled "Select these words", whose Enter and Space
// did what no role said they would, and the items with nothing to select
// were stops that did nothing. A negative prompt finding's is held while that
// box is greyed (`negative`, syncNegativeShow). Returns the button, for the
// caller to place.
let saidIds = 0;

function makeSelectable(li, select, said, where, negative = false) {
  li.classList.add('selectable');
  li.onclick = (event) => { if (!event.target.closest('a, button')) select(); };
  if (!said.id) said.id = `analysis-said-${++saidIds}`;
  const show = document.createElement('button');
  show.type = 'button';
  show.className = 'ghost tiny analysis-show';
  show.textContent = 'Show';
  show.dataset.title = `Select these words in ${where}`;
  show.title = show.dataset.title;
  show.setAttribute('aria-describedby', said.id);
  show.onclick = select;
  if (negative) {
    show.dataset.negativeShow = '';
    syncNegativeShow(show);
  }
  return show;
}
