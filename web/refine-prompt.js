// Local Diffusion's page script: the Refine dialog prompt check and tag suggestions.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

let refineTagLayout = null;

function refineProseAtCaret() {
  if (!tagHelper?.hybrid_prompt || refineTagLayout?.model !== loadedModel
    || refineTagLayout.prompt !== els.refinePrompt.value) return false;
  const caret = els.refinePrompt.selectionStart ?? els.refinePrompt.value.length;
  return refineTagLayout.parts?.some((part) => part.kind === 'prose' && caret >= part.start && caret <= part.end) ?? false;
}

// ------------------------------------------------- the pass's own prompt

// The Prompt box grows with what it holds, to two fifths of the window:
// three rows showed 14 of a tag prompt's 41 tags. Only while the fold is
// open - a closed one has no height to measure.
function sizeRefinePrompt() {
  const el = els.refinePrompt;
  if (!els.refineMore.open) return;
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight + 2, Math.round(innerHeight * 0.4))}px`;
}

function scheduleRefineCheck(delay = 400) {
  clearTimeout(refineCheckTimer);
  refineCheckTimer = setTimeout(checkRefinePrompt, delay);
}

// The prompt check on the pass's own prompt - the panel's Check reads the
// panel's (`collect()`), which a pass never sends. The same route, cold and
// off the image queue, given the dialog's request; the strength left to the
// server, since what the encoders read does not depend on it and a refused
// one would cost the check. Its failures are silent: it is advice, and the
// preview says what refuses the pass. A first use fetches the tokenizers,
// which takes a while, and the line says it is checking; an instant answer
// does not flash that first.
async function checkRefinePrompt() {
  if (!refineSource || !els.refineDialog.open) return;
  const epoch = ++refineCheckEpoch;
  const { body } = refineRequest();
  if (!body.prompt) { refineChecked = null; renderRefineCheck(); return; }
  const asked = refineCheckAsked(body);
  const slow = setTimeout(() => {
    if (epoch !== refineCheckEpoch) return;
    els.refineCheck.replaceChildren(refineCheckLine('Checking how much of the prompt the encoders read…'));
    els.refineCheck.hidden = false;
  }, 300);
  let result = null;
  try {
    result = await api('/api/prompt/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, strength: undefined, strength_axis: undefined, seeds: undefined }),
    });
  } catch { /* advice only */ }
  clearTimeout(slow);
  if (epoch !== refineCheckEpoch) return;
  refineChecked = result ? { ...asked, body: result } : null;
  renderRefineCheck();
}

// What a check was asked about: the two boxes' text as the request sends it,
// and whether the negative runs - greying it changes what the rules read, as
// the panel's check counts it.
function refineCheckAsked(body) {
  return {
    prompt: body.prompt,
    negative: body.negative_prompt ?? '',
    negativeOff: els.refineNegative.disabled,
  };
}

function refineCheckLine(text, warn = false) {
  const line = document.createElement('p');
  line.className = warn ? 'hint warn' : 'hint';
  line.append(text);
  return line;
}

// `field` is the box the button acts on: one on the negative prompt is dead
// while that box is greyed (`syncRefineCheck`).
function refineCheckButton(label, title, field, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost tiny';
  button.textContent = label;
  button.title = title;
  button.dataset.title = title;
  button.dataset.field = field;
  button.onclick = onClick;
  return button;
}

// What the check found that the dialog can act on: where each encoder stops
// reading - SDXL's two CLIP towers stop together, one line for both - with
// Show selecting the rest, and each rule finding that rewrites its field
// whole, with Apply. Not a pooled-only cut (FLUX.1's CLIP-L, while T5 reads
// on), nor an encoder's advice, which is text and no rewrite. The lines are
// rebuilt on every reply; what a screen reader hears is their text, said
// again only when it changed (`#refine-check-said`), since a reply comes a
// moment after every pause in typing and is mostly the same as the last.
function renderRefineCheck() {
  const box = els.refineCheck;
  box.replaceChildren();
  const said = [];
  const checked = refineChecked;
  if (checked) {
    for (const [field, text] of [['prompt', checked.prompt], ['negative_prompt', checked.negative]]) {
      const cuts = new Map();
      for (const report of checked.body.encoders) {
        if (report.field !== field || report.cut == null || report.pooled_only) continue;
        cuts.set(report.cut, [...(cuts.get(report.cut) ?? []), report]);
      }
      for (const [cut, reports] of cuts) {
        const line = refineCutLine(field, text, cut, reports);
        said.push(line.firstChild.textContent.trim());
        box.append(line);
      }
    }
    for (const finding of checked.body.findings) {
      if (finding.source === 'encoder' || !finding.suggestion) continue;
      const negative = finding.field === 'negative_prompt';
      const line = refineCheckLine(`${finding.message} `, finding.severity === 'warn');
      line.append(refineCheckButton('Apply', `Rewrite the ${negative ? 'negative prompt' : 'prompt'} `
        + 'as the check suggests (Ctrl+Z undoes it)', finding.field, () => applyRefineFinding(finding)));
      // Why its Apply is dead while the negative box is greyed, where the
      // fold's own note may be out of sight.
      if (negative) line.append(Object.assign(document.createElement('span'), { className: 'rf-off', hidden: true }));
      said.push(finding.message);
      box.append(line);
    }
  }
  box.hidden = !box.children.length;
  const text = said.join(' ');
  if (els.refineCheckSaid.textContent !== text) els.refineCheckSaid.textContent = text;
  syncRefineCheck();
}

// "CLIP-L/CLIP-G read 75 of 132 tokens (22 of 42 tags); the rest, from
// "outstretched arm", is never read." Offsets count code points, as the
// server's do; a tag counts as read when the cut is past its end. `select`
// is what Show does: this dialog's box, or the Inpaint dialog's (inpaint.js).
function refineCutLine(field, text, cut, reports, select = selectRefineFrom) {
  const { tokens, budget } = reports[0];
  let tags = '';
  if (tagVocabularyEnabled && text.includes(',')) {
    let read = 0;
    let total = 0;
    let at = 0;
    for (const part of text.split(',')) {
      const end = at + [...part].length;
      if (part.trim()) { total += 1; if (end <= cut) read += 1; }
      at = end + 1;
    }
    tags = ` (${read} of ${total} tags)`;
  }
  const whose = field === 'negative_prompt' ? 'Negative prompt: ' : '';
  const line = refineCheckLine(
    `${whose}${reports.map((r) => r.name).join('/')} read ${Math.min(tokens, budget)} of ${tokens} tokens`
    + `${tags}; the rest, from "${excerpt(text.slice(units(text, cut)))}", is never read. `,
    true,
  );
  line.append(refineCheckButton('Show', 'Select what the encoders never read', field,
    () => select(field, text, cut)));
  return line;
}

// Show and Apply act on the text the check read: dead, and the line dimmed,
// while either box holds other text or the negative's activation moved - the
// next check is on its way. One on the negative prompt is dead too while that
// box is greyed, as the panel's is: written there, nothing would run.
function syncRefineCheck() {
  if (!refineSource) return;
  const { body } = refineRequest();
  const asked = refineCheckAsked(body);
  const stale = !refineChecked || ['prompt', 'negative', 'negativeOff'].some((key) => asked[key] !== refineChecked[key]);
  els.refineCheck.classList.toggle('stale', stale);
  const greyed = els.refineNegative.disabled;
  const off = `The negative prompt is greyed at these settings, so nothing written there would run: it needs ${
    trueCfgSupported ? 'True CFG' : 'CFG'} above ${cfgThreshold}.`;
  for (const button of els.refineCheck.querySelectorAll('button')) {
    const dead = greyed && button.dataset.field === 'negative_prompt';
    button.disabled = stale || dead;
    button.title = dead ? off : button.dataset.title;
  }
  for (const note of els.refineCheck.querySelectorAll('.rf-off')) {
    note.hidden = !greyed;
    note.textContent = greyed ? ` ${off}` : '';
  }
}

function refineCheckChanged() {
  syncRefineCheck();
  scheduleRefineCheck();
}

// The fold opened on the box, and everything from the cut to the end
// selected, as the panel's "stops reading at" does.
function selectRefineFrom(field, text, cut) {
  const el = field === 'negative_prompt' ? els.refineNegative : els.refinePrompt;
  const raw = el.value;
  const lead = raw.length - raw.trimStart().length;
  if (el.disabled || raw.slice(lead, lead + text.length) !== text) return;
  els.refineMore.open = true;
  sizeRefinePrompt();
  el.focus();
  el.setSelectionRange(lead + units(text, cut), lead + text.length);
}

// A rule's whole-field suggestion, written the way typing writes (Ctrl+Z
// brings back what the box held), into the dialog's box and never the
// panel's. The fold opens first: a box in a closed one takes no focus, and
// without it the write could not be undone.
function applyRefineFinding(finding) {
  const el = finding.field === 'negative_prompt' ? els.refineNegative : els.refinePrompt;
  if (el.disabled) return;
  els.refineMore.open = true;
  writeRefineBox(el, finding.suggestion);
  sizeRefinePrompt();
  refineChanged();
  scheduleRefineCheck(0);
  scheduleRefineTags(0);
}

// Every whole-box write into the dialog's Prompt or Negative prompt, as
// tags.js's writeBox is the panel's: replaceBoxText, so Ctrl+Z gives back
// what the box held, then no suggestion list over the new text - the
// insert's `input` asked for one, as typing does. Never a greyed box: woken
// for the write and greyed again straight after it, Chromium kept no undo
// step for it, and the text it held was gone for good (Apply and From panel
// leave it alone, and say so).
function writeRefineBox(el, text) {
  if (el.disabled) return;
  replaceBoxText(el, text);
  hideRefineSuggestions();
}

// From panel: the panel's prompt and negative prompt into the pass's boxes.
// A critic's rewrite, Check's reorder or a comparison's winner is put to use
// in the panel; a pass opens on its source's prompt, and this is the way from
// one to the other. Never the other way: the panel is not the dialog's to
// write. What a press would write: the prompt where it differs, and the
// negative where it differs and its box is live. A greyed negative is left
// as it is (writeRefineBox) - the pass does not send it - and `knob`
// says what would have it copied. A checkpoint with no negative prompt has
// nothing there to compare.
function refineFromPanelPlan() {
  const prompt = els.prompt.value;
  const negative = els.negative.value;
  const promptDiffers = prompt.trim() !== els.refinePrompt.value.trim();
  const negativeDiffers = negativeSupported && negative.trim() !== els.refineNegative.value.trim();
  const greyed = els.refineNegative.disabled;
  const knob = `set ${trueCfgSupported ? 'True CFG' : 'CFG'} above ${cfgThreshold}`;
  const problem = !prompt.trim() ? "The panel's prompt is empty: there is nothing to copy"
    : !promptDiffers && !negativeDiffers
      ? (negativeSupported ? "This pass's prompt and negative prompt are the panel's already"
        : "This pass's prompt is the panel's already")
      : !promptDiffers && greyed
        ? `This pass's prompt is the panel's already, and its negative prompt is greyed: ${knob} to copy the panel's`
        : null;
  return {
    problem,
    prompt: promptDiffers ? prompt : null,
    negative: negativeDiffers && !greyed ? negative : null,
    knob: negativeDiffers && greyed ? knob : null,
  };
}

// Held, with the reason as its title, while there is nothing it would write -
// as Compare 3 is - so it keeps focus and a press can say why.
function syncRefineFromPanel() {
  const plan = refineFromPanelPlan();
  els.refineFromPanel.setAttribute('aria-disabled', String(!!plan.problem));
  els.refineFromPanel.title = plan.problem
    ?? (plan.knob
      ? `Put the panel's prompt in this pass's box (Ctrl+Z in the box undoes it) - not its negative prompt, greyed here: ${plan.knob} to copy that too`
      : negativeSupported
        ? "Put the panel's prompt and negative prompt in this pass's boxes (Ctrl+Z in the box undoes it)"
        : "Put the panel's prompt in this pass's box (Ctrl+Z in the box undoes it)");
}

// What a press did not do, beside the button: the dialog's error line is at
// its foot, out of the window with the fold open on a 900 px screen, so a
// press there seemed to do nothing. A status, written into a line that is in
// the page already (empty, it has no height), and scrolled into the dialog's
// view where the button was its last line; the next change clears it
// (refineChanged), as do an open and a close.
function sayRefineFromPanel(message) {
  els.refineFromNote.textContent = message;
  if (message) els.refineFromNote.scrollIntoView({ block: 'nearest' });
}

// The negative first and the prompt last, so the prompt's Ctrl+Z is the top
// of the undo history and gives the source's back; focus ends in the box
// written last.
function copyRefineFromPanel() {
  const plan = refineFromPanelPlan();
  if (plan.problem) {
    sayRefineFromPanel(`${plan.problem}.`);
    return;
  }
  els.refineMore.open = true;
  if (plan.negative !== null) writeRefineBox(els.refineNegative, plan.negative);
  if (plan.prompt !== null) writeRefineBox(els.refinePrompt, plan.prompt);
  if (!els.refineDialog.contains(document.activeElement)) els.refineFromPanel.focus();
  sizeRefinePrompt();
  refineChanged();
  scheduleRefineCheck(0);
  scheduleRefineTags(0);
  // After refineChanged, which clears the line.
  if (plan.knob) {
    sayRefineFromPanel(`Copied the prompt. The negative prompt is greyed, so the panel's was not copied: ${plan.knob} and press From panel again.`);
  }
}

// ------------------------------------------------- the pass's tag suggestions

// On a tag checkpoint the Prompt box suggests tags as the panel's does, for
// the comma item at the caret: tags.js's editor list (`scheduleEditorSuggestions`)
// under the box, picked by a click, or Enter or Tab on the selected one - the
// first selected only from three typed characters it starts with (`preselects`).
// None of the panel's state: not its list, the chips' typing state or the
// panel's box. Asked on typing, a click and the caret keys, as the panel's;
// a list already up stays while it asks (`keep`), and a click or a caret key
// that stays in the item it lists asks nothing: before, the list blinked
// shut on every key and on a click in the box.
function suggestRefineTags(event) {
  if (!refineSource || !tagVocabularyEnabled || document.activeElement !== els.refinePrompt || refineProseAtCaret()) {
    hideRefineSuggestions();
    return;
  }
  const menu = els.refineTagSuggestions;
  const range = currentTagRange(els.refinePrompt);
  const { query } = range;
  const prompt = els.refinePrompt.value;
  const model = loadedModel;
  const source = refineSource;
  if (event?.type !== 'input' && !menu.hidden && menu._query === query) return;
  scheduleEditorSuggestions(els.refinePrompt, menu, {
    query,
    pick: applyRefineSuggestion,
    place: placeRefineSuggestions,
    keep: true,
    current: () => refineSource === source && els.refineDialog.open && loadedModel === model
      && document.activeElement === els.refinePrompt && els.refinePrompt.value === prompt
      && currentTagRange(els.refinePrompt).start === range.start
      && currentTagRange(els.refinePrompt).end === range.end && !refineProseAtCaret(),
  });
}

function hideRefineSuggestions() {
  clearTimeout(tagEditorSuggestionTimer);
  tagEditorSuggestionEpoch += 1;
  renderEditorSuggestions(els.refinePrompt, els.refineTagSuggestions, [], { pick: applyRefineSuggestion });
}

// Under the box, or over it where the part of the dialog in view has no room
// below for the list as drawn and more above: the dialog scrolls, and a list
// past its bottom edge was cut off there.
function placeRefineSuggestions(menu) {
  menu.classList.remove('up');
  if (menu.hidden) return;
  const box = els.refinePrompt.getBoundingClientRect();
  const frame = els.refineDialog.getBoundingClientRect();
  const below = frame.bottom - box.bottom;
  const above = box.top - frame.top;
  // The list's 4 px gap, the dialog's border and a scrollbar along its foot.
  menu.classList.toggle('up', menu.offsetHeight + 20 > below && above > below);
}

// The comma item at the caret replaced by an insert, as typing replaces a
// selection: one Ctrl+Z takes the suggestion back and leaves what was typed.
// An accepted suggestion is a finished tag: nothing opens over it, and the
// tag line checks it at once.
function applyRefineSuggestion(suggestion) {
  const el = els.refinePrompt;
  const { start, end } = currentTagRange(el);
  el.focus();
  el.setSelectionRange(start, end);
  if (!document.execCommand('insertText', false, suggestion.text)) {
    el.setRangeText(suggestion.text, start, end, 'end');
    el.dispatchEvent(new Event('input'));
  }
  hideRefineSuggestions();
  scheduleRefineTags(0);
}

// The list's keys, on the box: ArrowDown and ArrowUp walk it, Enter or Tab
// takes the selected item, and an Escape the list takes closes the list
// alone - prevented, so the dialog's own Escape (its cancel) never sees it;
// the next one closes the dialog. Ctrl+Enter stays the document's.
function refinePromptKeydown(event) {
  const menu = els.refineTagSuggestions;
  if (menu.hidden || event.isComposing || event.ctrlKey || event.metaKey) return;
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    selectEditorSuggestion(menu, menu._index + (event.key === 'ArrowDown' ? 1 : -1));
  } else if ((event.key === 'Enter' || event.key === 'Tab') && menu._index >= 0) {
    event.preventDefault();
    menu._pick(menu._suggestions[menu._index]);
  } else if (event.key === 'Escape') {
    event.preventDefault();
    hideRefineSuggestions();
  }
}

function refinePromptCaret(event) {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) suggestRefineTags(event);
}

// A tag checkpoint's unknown comma items, as the panel's chips mark them but
// as one line: "Not a known tag: smlie". Advisory - the pass still runs - and
// silent when the lookup is unavailable or fails, as the chips never call an
// unchecked tag unknown. Contextual prose is never a tag warning.
function scheduleRefineTags(delay = 400) {
  clearTimeout(refineTagTimer);
  const epoch = ++refineTagEpoch;
  const text = els.refinePrompt.value;
  const model = loadedModel;
  const tags = promptTagParts(text).slice(0, TAG_LOOKUP_CAP);
  if (!refineSource || !tagVocabularyEnabled || !tags.length) {
    refineTagLayout = null;
    els.refineTags.hidden = true;
    return;
  }
  refineTagTimer = setTimeout(async () => {
    let unknown = [];
    try {
      const body = await api('/api/tags/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags, prompt: text }),
      });
      if (epoch !== refineTagEpoch || loadedModel !== model || els.refinePrompt.value !== text) return;
      refineTagLayout = { model, prompt: text, parts: body.parts };
      if (refineProseAtCaret()) hideRefineSuggestions();
      if (body.available) unknown = body.results.filter((r, index) => r.valid === false
        && !body.parts?.some((part) => part.kind === 'prose' && index >= part.first && index <= part.last));
    } catch { /* unchecked is not unknown */ }
    if (epoch !== refineTagEpoch) return;
    // A dozen by name; a prompt the index knows little of is a count after that.
    const names = [...new Set(unknown.map((r) => r.tag))];
    const listed = names.length > 12 ? [...names.slice(0, 12), `${names.length - 12} more`] : names;
    els.refineTags.textContent = names.length
      ? `Not a known tag: ${listed.join(', ')} - the pass still runs with ${names.length === 1 ? 'it' : 'them'}.`
      : '';
    els.refineTags.title = names.map((tag) => `${tag}: ${unknown.find((r) => r.tag === tag).reason}`).join('\n');
    els.refineTags.hidden = !names.length;
  }, delay);
}
