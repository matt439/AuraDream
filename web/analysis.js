// Local Diffusion's page script: the prompt check, the critic and the comparison grid. Declarations only; state.js has
// the load-order contract these files share.
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
let analysisStale = false;
// The critic's answer is always about the checked text: asking checks the box
// again first when it has moved on, and a check of other text drops the answer.
let critiqueResult = null;  // { body: PromptCritique }
let critiqueEpoch = 0;
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

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

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

// A few words from the start of `text`, cut at a word boundary: enough to say
// where an encoder stops without printing the rest of the prompt again.
// Counted in code points, as `units()` counts: CJK has no space to cut at, and
// a cut by UTF-16 units through an emoji left half of it, a broken glyph.
function excerpt(text, max = 36) {
  const flat = [...text.trim().replace(/\s+/g, ' ')];
  if (flat.length <= max) return flat.join('');
  const at = flat.lastIndexOf(' ', max);
  return `${flat.slice(0, at > max / 2 ? at : max).join('')}…`;
}

// Sends what Generate would send: the negative prompt, the CFG and True CFG
// decide whether that negative ever runs, and the answer depends on it.
async function checkPrompt(payload = collect()) {
  if (!payload.prompt) { showError('Enter a prompt first.'); els.prompt.focus(); return null; }
  const epoch = ++analysisEpoch;
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
    // The critic's answer is about the text checked last, negative prompt and all.
    if (critiqueResult && !analysed(payload, effectiveNegative)) critiqueResult = null;
    analysisResult = {
      source: payload.prompt, negative: payload.negative_prompt ?? '', effectiveNegative,
      reference: [...(payload.reference ?? [])], body,
    };
    showError('');
    renderAnalysis();
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

// Check, pressed. On a two-column page the panel sits above the results, some
// forty Tab stops past the button, and only its summary is announced; so the
// answer takes focus, at its heading, while focus is still on the button.
// Under 901 px the panel is three stops away, and focus stays where it is.
async function checkFromButton() {
  const result = await checkPrompt();
  if (result && document.activeElement === els.analyze
      && els.promptAnalysis.parentElement === els.tagValidationWide) focusAnalysisHeading();
}

// The panel's heading (tabindex=-1): where focus goes when the button that had
// it is far from the answer, or gone. focus() scrolls it into view.
function focusAnalysisHeading() {
  els.promptAnalysis.querySelector('.analysis-head h2').focus();
}

function renderAnalysis() {
  const { body } = analysisResult;
  // First: the comparison's preview is only fetched for a panel on show.
  els.promptAnalysis.hidden = false;
  els.analysisModel.textContent = body.model_id.split('/').pop();
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
  // Anima's @ before an artist name (`category_items`) - and it read none
  // (the index unreachable) or not all (past the lookup's 64).
  const artists = !body.order && !!body.categories_detail;
  const counts = !body.findings.length
    ? (unchecked ? 'nothing flagged · order not checked' : partly || artists ? 'nothing flagged' : 'nothing to flag')
    : `${warns ? plural(warns, 'warning') : 'no warnings'}${notes ? ` · ${plural(notes, 'note')}` : ''}`;
  els.analysisSummary.textContent =
    `${plural(body.words, 'word')}, ${STYLE_NAMES[body.style] ?? body.style} · ${counts}`
    + (partly ? ' · order partly checked' : artists ? ' · artists not all checked' : '');
  renderEncoders(body);
  renderOrder(body);
  renderFindings(els.analysisFindings, body.findings, {
    select: (f) => selectAnalysed(f.field, f.start, f.end),
    apply: applyFinding,
    empty: unchecked
      ? "Nothing flagged by the rules for this checkpoint's guide - but its word order was not checked (see Order)."
      : partly ? "Nothing flagged by the rules for this checkpoint's guide - but its word order was only partly checked (see Order)."
      : artists ? "Nothing flagged by the rules for this checkpoint's guide - but not every tag was checked for an artist name (below)."
      : "Nothing to flag against this checkpoint's guide.",
  });
  if (artists) {
    els.analysisFindings.appendChild(note(`Not every tag was checked for an artist name: ${body.categories_detail}.`));
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
  if (!stale && document.activeElement === els.analysisAgain) focusAnalysisHeading();
  els.analysisAgain.hidden = !stale;
  for (const button of els.promptAnalysis.querySelectorAll('[data-writes]')) button.disabled = writeBlocked(button);
  syncCompareRun();
}

// A negative prompt written into a greyed box never runs, and the check -
// which sends nothing from a greyed box - would answer the same again: so its
// Apply is dead while the box is, and says what would wake it.
function syncNegativeApply() {
  for (const button of els.promptAnalysis.querySelectorAll('[data-negative]')) {
    button.disabled = writeBlocked(button);
    const off = button.parentElement.querySelector('.negative-off');
    off.hidden = !els.negative.disabled;
    off.textContent = `The negative prompt is greyed at these settings, so nothing written there would run: it needs ${trueCfgSupported ? 'True CFG' : 'CFG'} above ${cfgThreshold}.`;
  }
}

const writeBlocked = (button) => analysisStale || ('negative' in button.dataset && els.negative.disabled);

// Close keeps what was found - the next Check replaces it anyway - and hands
// focus back to the button that opened the panel; to the prompt while that
// button waits out the critic's own check, since a disabled one takes no focus
// and it would fall to <body>.
function closeAnalysis() {
  els.promptAnalysis.hidden = true;
  (els.analyze.disabled ? els.prompt : els.analyze).focus();
}

// Forget the check: another checkpoint's tokenizers and guide, or a Reset. An
// answer still on its way is dropped when it lands.
function clearAnalysis() {
  analysisEpoch += 1;
  critiqueEpoch += 1;
  analysisResult = null;
  critiqueResult = null;
  analysisStale = false;
  resetCheckButton();
  els.promptAnalysis.hidden = true;
  els.promptAnalysis.classList.remove('stale');
  for (const box of [els.analysisEncoders, els.analysisOrder, els.analysisFindings,
    els.criticResult, els.compareOptions]) box.replaceChildren();
  els.comparePreview.textContent = '';
  els.compareOwn.value = '';
  syncCompareRun();
}

// The analysed text of a field, which the offsets count into.
const analysedText = (field) =>
  (field === 'negative_prompt' ? analysisResult?.negative : analysisResult?.source) ?? '';

// Every offset from the server counts code points, as Python indexes a str;
// the page's strings count UTF-16 units, where each character past the BMP -
// an emoji, a rare CJK ideograph - takes two. One offset into `text`, in the
// page's units.
const units = (text, points) => [...text].slice(0, points).join('').length;

function selectAnalysed(field, start, end) {
  return selectIn(field === 'negative_prompt' ? els.negative : els.prompt, analysedText(field), start, end);
}

// Offsets count code points from the analysed text's first character; the box
// may hold it behind whitespace `collect()` trimmed off, or hold other text by
// now. The words are selected only while they are still where the check found
// them.
function selectIn(el, text, start, end) {
  if (el.disabled) return false;
  const [from, to] = [units(text, start), units(text, end)];
  const raw = el.value;
  const lead = raw.length - raw.trimStart().length;
  if (raw.slice(lead + from, lead + to) !== text.slice(from, to)) {
    showError('Those words are no longer where the check found them; Check again.', true);
    return false;
  }
  el.focus();
  el.setSelectionRange(lead + from, lead + to);
  return true;
}

// A button that writes an answer into the panel's fields. Each is worked out
// from the checked text, so every one goes dead while either box holds other
// text.
function writeButton(label, title, onClick, { negative = false } = {}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost tiny';
  button.textContent = label;
  button.title = title;
  button.dataset.writes = '';
  if (negative) button.dataset.negative = '';
  button.disabled = writeBlocked(button);
  button.onclick = onClick;
  return button;
}

// A whole box rewritten the way typing would write it, so Ctrl+Z brings back
// what it held. Assigning `.value` empties a textarea's undo history, and the
// check that follows makes the new text the comparison's original: what was
// typed would be nowhere on the page. The insert fires `input` like typing;
// a greyed box, or a browser that will not insert, gets the plain write.
function replaceBoxText(el, text) {
  el.focus();
  el.select();
  const inserted = document.activeElement === el && document.execCommand('insertText', false, text);
  if (!inserted) el.value = text;
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

// A rule's suggestion is its field rewritten whole - quotes added, tags moved,
// the recommended negative prompt - so it applies in one click, to its field.
function applyFinding(finding) {
  if (finding.field !== 'negative_prompt') { usePromptText(finding.suggestion); return; }
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
    // Not "the authors document none": Anima's card, Neta Lumina's prompt
    // book and the rewriter instructions give orders the guide leaves out.
    // What is missing is this app's check - and the authors' orders, where
    // they wrote some, are shown as theirs.
    box.appendChild(note("This app checks no word order for this checkpoint: its guide states none."));
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
    box.appendChild(note(`Not checked: Check can only place a tag checkpoint's comma tags in its order. ${promptCritiqueSupported
      ? 'Ask the critic whether this prompt follows it, or write'
      : 'Write'} a reordered version under Compare to test it on your own seeds.`));
  } else if (body.order_detail) {
    // Placed, but an item with no category keeps every slot a category could
    // have put it in - which slots those are is the guide's (Animagine's order
    // has no artist) - so it can sit out of order unflagged. The server names
    // the items and why each has none.
    box.appendChild(note(`Partly checked: ${body.order_detail}. Any of those may sit out of order unflagged.`));
  }
  // The prompt's own items moved into that order, nothing added or removed -
  // shown, since it is read before it is used.
  if (body.reordered) {
    const moved = document.createElement('div');
    moved.className = 'order-reordered';
    moved.append(span('analysis-note', 'In that order:'), note(body.reordered, 'analysis-quote'),
      writeButton('Use reordered',
        `Replace the prompt with the same words in the documented order: ${body.reordered}`,
        () => usePromptText(body.reordered)));
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
// within each. `select` puts a finding's words in their box, `apply` writes
// its suggestion; a list without `apply` (the rewrite's) only reports.
function renderFindings(box, findings, { select, apply, empty }) {
  box.replaceChildren();
  if (!findings.length) { box.appendChild(note(empty)); return; }
  const list = document.createElement('ul');
  list.className = 'analysis-list';
  const ranked = [...findings].sort((a, b) => (a.severity === 'warn' ? 0 : 1) - (b.severity === 'warn' ? 0 : 1));
  for (const finding of ranked) list.appendChild(findingItem(finding, select, apply));
  box.appendChild(list);
}

function findingItem(finding, select, apply) {
  const li = document.createElement('li');
  li.className = `analysis-item ${finding.severity}`;
  li.tabIndex = 0;
  const mark = span('analysis-mark', finding.severity === 'warn' ? '!' : 'i');
  mark.setAttribute('aria-hidden', 'true');
  const message = span('analysis-message');
  message.append(span('analysis-sr', finding.severity === 'warn' ? 'Warning: ' : 'Note: '), finding.message);
  li.append(mark, message, sourceLabel(finding.source, finding.cite, finding.url));
  // The encoder's suggestion is advice about the text; a rule's is the whole
  // field rewritten, which applies in one click - but not to a rewrite the
  // page only shows.
  if (finding.suggestion && (finding.source === 'encoder' || apply)) {
    const extra = document.createElement('div');
    extra.className = 'analysis-extra';
    if (finding.source === 'encoder') extra.textContent = finding.suggestion;
    else {
      const negative = finding.field === 'negative_prompt';
      extra.append(writeButton(negative ? 'Apply to the negative prompt' : 'Apply',
        `Replace the ${negative ? 'negative prompt' : 'prompt'} with: ${finding.suggestion}`,
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
  if (select && finding.start != null && finding.end != null) makeSelectable(li, () => select(finding));
  return li;
}

// Clicking a finding (Enter or Space on a focused one) selects the words it is
// about - but not a click meant for its link or its button.
function makeSelectable(li, select) {
  li.classList.add('selectable');
  li.title = 'Select these words';
  li.onclick = (event) => { if (!event.target.closest('a, button')) select(); };
  li.onkeydown = (event) => {
    if ((event.key === 'Enter' || event.key === ' ') && event.target === li) {
      event.preventDefault();
      select();
    }
  };
}

// ------------------------------------------------------------------- critic

// The critic button and its line. Called from syncPromptExpansion, so every
// change to the queue, Expand or the critic itself passes through here.
function syncCritic() {
  els.analysisCritic.hidden = !promptCritiqueSupported;
  // aria-disabled, as Generate is held: disabled, the button just pressed lost
  // a keyboard user's focus to <body>. askCritic refuses each case itself.
  els.criticAsk.setAttribute('aria-disabled', String(critiquing || busy || expandingPrompt));
  els.criticAsk.textContent = critiquing ? 'Thinking…' : 'Ask the critic';
  els.criticAsk.title = busy
    ? 'An image job is queued or running. The critic needs the card to itself, so it waits for the queue to finish.'
    : expandingPrompt ? 'Wait for prompt expansion to finish.'
    : "A local model reads the prompt against this checkpoint's guide and its authors' own texts, cites what it goes by, and suggests a rewrite";
  // Named as the answer's footer names it, and a first use's download only
  // what it would fetch. The time and the GPU are the pinned pair's, measured
  // on the card; a server or a model of the user's own is promised neither.
  const critic = criticStatus ?? { label: 'model', pinned: false, fetch: [], fetch_bytes: 0 };
  const fetch = !promptCritiqueReady && critic.fetch.length
    ? `${critic.fetch.join(' and ')} (about ${humanBytes(critic.fetch_bytes)})` : '';
  const line = critiquing
    ? (fetch ? `Downloading ${fetch} once, then thinking…`
      : critic.pinned ? 'Thinking… about 15 seconds on the GPU.' : `Thinking… (${critic.label})`)
    : busy ? 'An image job has the card; the critic waits for the queue to finish.'
    : fetch ? `First use downloads ${fetch} once.`
    : critic.pinned ? `Local ${critic.label} on the GPU, about 15 seconds; waits for image jobs.`
    : `Local ${critic.label}, as configured; waits for image jobs.`;
  // The line is a live region: rewriting it with the same words would be
  // read out again on every queue change.
  if (els.criticLine.textContent !== line) els.criticLine.textContent = line;
  syncCompareRun();
}

async function askCritic() {
  if (critiquing) return;
  if (busy) { showError('Wait for the image queue to finish before asking the critic.'); return; }
  if (expandingPrompt) { showError('Wait for prompt expansion to finish.'); return; }
  const payload = collect();
  if (!payload.prompt) { showError('Enter a prompt first.'); els.prompt.focus(); return; }
  const effectiveNegative = analysisNegative(payload);
  const epoch = ++critiqueEpoch;
  // Held from here, the check below included: a job or an Expand started
  // while it ran went ahead, and the critique behind it came back a 409.
  critiquing = true;
  showError('');
  syncPromptExpansion();
  let answered = false;
  try {
    // The panel has to describe what the critic is asked about. A Check
    // pressed meanwhile would replace this check, and the critic was then
    // never asked, without a word: so Check waits for it.
    if (!analysed(payload, effectiveNegative)) {
      const before = analysisEpoch;
      els.analyze.disabled = els.analysisAgain.disabled = true;
      let checked;
      try {
        checked = await checkPrompt(payload);
      } finally {
        els.analyze.disabled = els.analysisAgain.disabled = false;
      }
      if (!checked) {
        // A failed check said why, and a cleared panel shows nothing pending
        // (a Reset, another checkpoint); one replaced says nothing itself.
        if (epoch === critiqueEpoch && analysisEpoch !== before + 1) {
          showError('The prompt check was replaced before the critic was asked, so nothing was sent. Ask again.', true);
        }
        return;
      }
    }
    critiqueResult = null;
    renderCritic();
    renderCompare();
    const body = await api('/api/prompt/critique', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (epoch !== critiqueEpoch) return;  // the panel was cleared meanwhile
    if (!analysed(payload, effectiveNegative)) {
      showError('The prompt was checked again while the critic ran, so its answer was set aside. Ask again.', true);
      return;
    }
    critiqueResult = { body };
    answered = true;
    // Shown, and dimmed with the rest of the panel, but never applied: its
    // rewrite would replace what was typed while it thought.
    if (!analysed(collect())) {
      showError('The prompt, the negative prompt or the references changed while the critic ran: its answer is about the earlier text, so nothing from it can be applied. Check again for the current one.', true);
    }
  } catch (err) {
    if (epoch === critiqueEpoch) showError(err.message);
  } finally {
    critiquing = false;
    syncPromptExpansion();
    if (epoch === critiqueEpoch) {
      renderCritic();
      renderCompare();
      // Nothing else announces the answer - the line under the button goes
      // back to the cost - so it takes focus, if focus is still waiting on
      // the button rather than on something typed meanwhile.
      if (answered && document.activeElement === els.criticAsk) els.criticResult.focus();
    }
  }
}

function renderCritic() {
  syncCritic();
  const box = els.criticResult;
  box.replaceChildren();
  if (!critiqueResult) return;
  const { body } = critiqueResult;

  const parts = PART_LABELS.filter(([key]) => body.parts?.[key]);
  if (parts.length) {
    const list = document.createElement('dl');
    list.className = 'critic-parts';
    for (const [key, label] of parts) {
      const term = document.createElement('dt');
      term.textContent = label;
      const value = document.createElement('dd');
      value.textContent = body.parts[key];
      list.append(term, value);
    }
    box.appendChild(list);
  }

  if (body.issues.length) {
    const list = document.createElement('ul');
    list.className = 'analysis-list';
    for (const issue of body.issues) list.appendChild(issueItem(issue));
    box.appendChild(list);
  } else {
    box.appendChild(note('No issue against the sources.'));
  }

  if (body.rewrite) {
    const block = document.createElement('div');
    block.className = 'critic-variant';
    const area = variantArea(body.rewrite, "The critic's rewrite");
    const head = document.createElement('div');
    head.className = 'critic-variant-head';
    head.append(span('critic-variant-name', 'Rewrite'));
    if (body.rewrite !== analysisResult.source) {
      head.append(writeButton('Use rewrite',
        'Replace the prompt with the rewrite, then check it', () => usePromptText(body.rewrite)));
    }
    const checked = document.createElement('div');
    checked.className = 'critic-checked';
    renderFindings(checked, body.rewrite_findings, {
      // The rewrite's words are in the rewrite; a negative prompt finding is
      // about the box it came from.
      select: (f) => (f.field === 'negative_prompt'
        ? selectAnalysed(f.field, f.start, f.end) : selectIn(area, body.rewrite, f.start, f.end)),
      apply: null,
      empty: 'The check finds nothing to flag in the rewrite.',
    });
    block.append(head, rewriteChanges(analysisResult.source, body.rewrite), area,
      note('The same check, run on the rewrite:', 'analysis-note sub'), checked);
    box.appendChild(block);
  } else {
    box.appendChild(note('No rewrite offered.', 'analysis-note critic-change-summary'));
  }

  if (body.reordered) {
    const block = document.createElement('div');
    block.className = 'critic-variant';
    const head = document.createElement('div');
    head.className = 'critic-variant-head';
    // The server holds it to the prompt's own pieces, moved whole, and where
    // Check placed the tags itself to Check's order too - and says which it
    // kept (`reordered_kind`): Check's own reordering ("check"), an order
    // Check's rule finds nothing wrong with where it could not place every
    // tag ("allowed"), or, where nothing here read its order, the critic's
    // reading ("unchecked"), which the check that follows places. Decided
    // with the critique's own analysis, not this page's last check, whose
    // tag lookup may have had other answers.
    const kind = body.reordered_kind;
    const title = kind === 'check'
      ? 'Replace the prompt with the same words in the documented order, as Check placed them, then check it'
      : kind === 'allowed'
        ? "Replace the prompt with the critic's reordering of the same words - an order Check finds nothing wrong with, though it could not place every tag - then check it"
        : "Replace the prompt with the critic's reordering of the same words - its reading of the order, not one this app has checked - then check it";
    head.append(span('critic-variant-name', 'Reordered'), writeButton('Use reordered', title,
      () => usePromptText(body.reordered)));
    block.append(head, variantArea(body.reordered,
      kind === 'check' ? "The critic's reordering, Check's own order" : "The critic's reordering"));
    box.appendChild(block);
  }

  // What the server took out of the answer, and why. A quote not found in the
  // prompt drops nothing - the issue only loses its quote - so it is not here.
  if (body.dropped > 0) {
    box.appendChild(note(`${body.dropped} suggestion${body.dropped === 1 ? '' : 's'} dropped: `
      + `${body.dropped === 1 ? 'it cited' : 'they cited'} no source the critic was given, or said nothing.`));
  }
  for (const line of body.set_aside ?? []) box.appendChild(note(line));
  box.appendChild(note(`${body.model} · ${body.seconds.toFixed(1)} s`, 'analysis-note sub'));
  for (const area of box.querySelectorAll('textarea')) fitArea(area);
}

// An exact text diff: words stay readable, but spaces, punctuation and each
// backslash remain separate tokens, so even an escape-only edit is visible.
// Prompts are capped at 4,000 characters. Trim the shared ends before the
// bounded longest-common-subsequence table, then coalesce adjacent edits.
function promptChanges(original, rewrite) {
  const tokenize = (text) => text.match(/[\p{L}\p{N}_]+|\s+|[^\p{L}\p{N}_\s]/gu) ?? [];
  const before = tokenize(original), after = tokenize(rewrite);
  let start = 0, endBefore = before.length, endAfter = after.length;
  while (start < endBefore && start < endAfter && before[start] === after[start]) start++;
  while (endBefore > start && endAfter > start && before[endBefore - 1] === after[endAfter - 1]) {
    endBefore--;
    endAfter--;
  }
  const a = before.slice(start, endBefore), b = after.slice(start, endAfter);
  const width = b.length + 1;
  const lengths = new Uint16Array((a.length + 1) * width);
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lengths[i * width + j] = a[i] === b[j]
        ? 1 + lengths[(i + 1) * width + j + 1]
        : Math.max(lengths[(i + 1) * width + j], lengths[i * width + j + 1]);
    }
  }
  const changes = [];
  const append = (kind, text) => {
    if (!text) return;
    if (changes.at(-1)?.kind === kind) changes.at(-1).text += text;
    else changes.push({ kind, text });
  };
  append('same', before.slice(0, start).join(''));
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      append('same', a[i++]);
      j++;
    } else if (i < a.length && (j === b.length
        || lengths[(i + 1) * width + j] >= lengths[i * width + j + 1])) {
      append('removed', a[i++]);
    } else {
      append('added', b[j++]);
    }
  }
  append('same', before.slice(endBefore).join(''));
  return changes;
}

function rewriteChanges(original, rewrite) {
  if (original === rewrite) {
    return note('No changes from original.', 'analysis-note critic-change-summary');
  }
  const changes = promptChanges(original, rewrite);
  const count = (kind) => changes.filter((part) => part.kind === kind)
    .reduce((total, part) => total + [...part.text].length, 0);
  const details = document.createElement('details');
  details.className = 'critic-changes';
  const summary = document.createElement('summary');
  summary.textContent = `Changed from original: ${plural(count('added'), 'character')} added, `
    + `${plural(count('removed'), 'character')} removed`;
  const diff = document.createElement('div');
  diff.className = 'critic-diff';
  diff.setAttribute('role', 'group');
  diff.setAttribute('aria-label', 'Changes from the original prompt');
  for (const { kind, text } of changes) {
    const part = document.createElement(kind === 'removed' ? 'del' : kind === 'added' ? 'ins' : 'span');
    part.textContent = text;
    diff.appendChild(part);
  }
  details.append(summary, note('Removed text is struck through; added text is underlined.', 'analysis-note sub'), diff);
  return details;
}

function variantArea(text, label) {
  const area = document.createElement('textarea');
  area.className = 'analysis-text';
  area.readOnly = true;
  area.rows = 2;
  area.value = text;
  area.setAttribute('aria-label', label);
  return area;
}

// A version is read before it is used, so its box shows it whole - up to
// about ten lines, past which it scrolls rather than taking the page. The
// stylesheet's `field-sizing` does that and follows a resize; this is the
// once-at-render fallback where the browser has not got it.
function fitArea(area) {
  if (CSS.supports('field-sizing', 'content')) return;
  if (!area.scrollHeight) return;  // a hidden panel has no layout to fit to
  area.style.height = `${Math.min(area.scrollHeight + 2, 220)}px`;
}

// One issue: its kind, the problem and the fix, and the numbered source it
// cites - resolved on the server, the cited words as the tooltip. A quote the
// server found in the prompt selects its words, like a finding's span. Words
// that are not the checkpoint's authors' own - a translation, a family or
// parent model's guide - say whose they are under it, as Order does; the
// fallback's sentences, with no page behind them, are this app's general
// advice and are not labelled Guide.
function issueItem(issue) {
  const li = document.createElement('li');
  li.className = 'analysis-item critic-issue';
  li.tabIndex = 0;
  const text = span('analysis-message');
  text.append(issue.problem);
  if (issue.quote) text.append(' ', span('analysis-quoted', `"${issue.quote}"`));
  text.append(span('critic-fix', `Fix: ${issue.fix}`));
  if (issue.note) text.append(span('analysis-note sub critic-note', issue.note));
  const cite = sourceLabel(issue.source, issue.cite, issue.url);
  const label = issue.source === 'guide' && !issue.url ? 'General' : cite.textContent;
  cite.textContent = `${issue.cite_id} · ${label}`;
  if (issue.note) {
    cite.title = `${issue.cite}\n(${issue.note})`;
    // A link's accessible name; a plain span takes none, and the note under
    // the Fix says it there.
    if (issue.url) cite.setAttribute('aria-label', `${issue.cite_id} · ${label}, ${issue.note}`);
  }
  li.append(span('critic-kind', issue.kind), text, cite);
  if (issue.start != null && issue.end != null) {
    makeSelectable(li, () => selectAnalysed('prompt', issue.start, issue.end));
  }
  return li;
}

// ------------------------------------------------------------------ compare

// The versions a comparison can take: the checked text always, a reordering -
// the rules' own, which is exact, ahead of the critic's - the critic's
// rewrite, and the user's own version. Whether the critic's reordering may
// stand is the server's to decide (`critic.validate`), and the page takes what
// it kept: where Check placed every tag itself, only Check's own order; where
// it could not place some tag (never looked up, or unknown to the tag index),
// any order the same rule finds nothing wrong with. The user's own version is
// what isolates order where neither layer can offer a reorder: a sentence on
// FLUX, where the critic's `reordered` rarely survives its check, or a
// checkpoint whose guide states no order. A version that is the original word
// for word, or another one's, is left out: the server would render it twice.
function compareVariants() {
  if (!analysisResult) return [];
  const { source, body } = analysisResult;
  const variants = [{ label: 'original', name: 'Original', prompt: source }];
  const seen = new Set([source]);
  for (const [label, name, text] of [
    ['reordered', 'Reordered', body.reordered ?? critiqueResult?.body.reordered],
    ['rewrite', 'Rewrite', critiqueResult?.body.rewrite],
    ['yours', 'Your version', els.compareOwn.value],
  ]) {
    const prompt = (text ?? '').trim();
    if (!prompt || seen.has(prompt)) continue;
    seen.add(prompt);
    variants.push({ label, name, prompt });
  }
  return variants;
}

const comparedVariants = () =>
  compareVariants().filter((v) => v.label === 'original' || compareChoice[v.label]);

// Typing a version of your own: the ticks at once, the server's count and
// estimate once the typing pauses.
function compareOwnChanged() {
  renderCompare(400);
}

// The checked prompt as the starting point for a version of your own - to
// move its clauses about, say - rather than typing it out again.
function compareOwnFromPrompt() {
  if (!analysisResult) return;
  els.compareOwn.value = analysisResult.source;
  els.compareOwn.focus();
  compareOwnChanged();
}

function renderCompare(previewDelay = 0) {
  const variants = compareVariants();
  els.compareOptions.replaceChildren();
  for (const variant of variants) {
    const label = document.createElement('label');
    label.className = 'check';
    label.title = variant.prompt;
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.dataset.variant = variant.label;
    box.checked = variant.label === 'original' || compareChoice[variant.label];
    // The row every other one is compared against.
    box.disabled = variant.label === 'original';
    box.onchange = () => {
      compareChoice[variant.label] = box.checked;
      syncCompareRun();
      scheduleComparePreview(0);
    };
    label.append(box, ` ${variant.name}`);
    els.compareOptions.appendChild(label);
  }
  syncCompareRun();
  scheduleComparePreview(previewDelay);
}

function syncCompareRun() {
  const count = comparedVariants().length;
  els.compareRun.disabled = count < 2 || analysisStale;
  // Held like Generate and Run grid while the editor has the card.
  els.compareRun.setAttribute('aria-disabled', String(expandingPrompt || critiquing || submitting));
  els.compareRun.title = analysisStale
    ? 'The prompt or the references changed since this check; Check again first'
    : count < 2 ? 'Needs a second version of the prompt to compare against'
    : 'One grid: each version of the prompt on the same seeds, then the sheet with a row per version';
}

// The grid's own payload - the panel's settings and seed - with every axis
// emptied: the versions are the one thing that varies, crossed with the
// seeds. Seeds is the total per version; the comparison's own Batch groups
// those seeds, so the grid's seed count is the number of batches. Every
// offered batch (1, 2, 4) divides both totals (4, 8). Read both selects here:
// a browser restoring form state sets them without a change event. Nothing is
// written into the prompt box or the grid's variables, which are the user's
// and remembered.
function comparePayload(variants) {
  const batch = +els.compareBatch.value;
  return {
    ...gridPayload(),
    num_images: batch,
    prompt: analysisResult.source,
    prompt_axis: '',
    steps_axis: '',
    cfg_axis: '',
    mu_axis: '',
    lora_scale_axis: '',
    seed_axis: '',
    seeds: +els.compareSeeds.value / batch,
    resume: false,
    budget_minutes: null,
    prompt_variants: variants.map(({ label, prompt }) => ({ label, prompt })),
  };
}

function scheduleComparePreview(delay = 250) {
  clearTimeout(comparePreviewTimer);
  if (!analysisResult || els.promptAnalysis.hidden) return;
  comparePreviewTimer = setTimeout(previewComparison, delay);
}

// The server's own count and estimate, as the grid section's preview is, with
// the same token against a slow reply landing over a newer one.
async function previewComparison() {
  const epoch = ++comparePreviewEpoch;
  const variants = comparedVariants();
  els.comparePreview.classList.remove('warn');
  if (variants.length < 2) {
    els.comparePreview.textContent = promptCritiqueSupported
      ? 'Only the original so far: write a version of your own above, or ask the critic for a rewrite.'
      : 'Only the original so far: write a version of your own above to compare it against.';
    return;
  }
  try {
    const payload = comparePayload(variants);
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (epoch !== comparePreviewEpoch) return;
    els.comparePreview.classList.toggle('warn', !!p.duplicates);
    // The grid's summary counts batch starts as seeds; the comparison's
    // Seeds select counts every image seed. Describe the request just sent,
    // so changing a select while it runs cannot relabel an earlier reply.
    els.comparePreview.textContent = `${plural(p.images, 'image')} · ${plural(variants.length, 'variant')}`
      + ` × ${plural(payload.seeds * payload.num_images, 'seed')} · batch ${payload.num_images}`
      + (p.seconds ? ` · about ${humanTime(p.seconds)}` : '')
      + (p.duplicates ? ` · ${p.duplicates} would be identical to another cell` : '');
  } catch (err) {
    if (epoch !== comparePreviewEpoch) return;
    els.comparePreview.classList.add('warn');
    els.comparePreview.textContent = err.message;
  }
}

// One grid, then its submission's sheet: startJob keeps the job id and opens
// it when the stream reports done, with one row per version as written.
function runComparison() {
  const variants = comparedVariants();
  if (!analysisResult || variants.length < 2 || analysisStale) return undefined;
  return startJob('/api/grid', comparePayload(variants), true);
}
