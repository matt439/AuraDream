// Local Diffusion's page script: the Examples menu under the prompt.
// Declarations only; its handlers are wired in app.js, and state.js has the
// shared-global load-order contract.
'use strict';

// An edit of a picture the prompt takes to be there already - BFL's "Turn
// this cat into a dog" - which with no reference image renders from the words
// alone. prompt_rules.EDIT_PROMPT, the check's note on it, is the same
// pattern; tests/test_prompt_rules.py holds the two equal.
const EDIT_PROMPT = /^\s*(turn|change|make|replace|remove|put|give)\b.*\b(this|these|the (photo|image|picture)|image \d)\b/i;

// The Examples menu is built on open from the guide's current list, so a
// status that lands late (behind a load) is not stale by the first click. It
// is a role=menu and keeps that promise: its items are reached with the arrow
// keys (ArrowDown on the button opens it there), and closing it with focus
// inside - Escape, a pick - hands focus back to the button, not to <body>.
function showExamples(open) {
  if (open && !els.tagsMenu.hidden) showTagHelper(false);
  const inside = els.examplesMenu.contains(document.activeElement);
  els.examplesBtn.setAttribute('aria-expanded', String(open));
  els.examplesMenu.hidden = !open;
  if (!open) {
    if (inside) els.examplesBtn.focus();
    return;
  }
  els.examplesMenu.replaceChildren();
  if (!examples.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No examples for this checkpoint yet.';
    els.examplesMenu.appendChild(li);
    return;
  }
  // The example stays the authors' words; what it needs is said beside it.
  const unreferenced = referenceSupported && !referenceIds.length;
  const needs = (words) => {
    const span = document.createElement('span');
    span.className = 'example-note';
    span.textContent = words;
    return span;
  };
  examples.forEach((text) => {
    const li = document.createElement('li');
    li.setAttribute('role', 'menuitem');
    li.tabIndex = -1;
    const span = document.createElement('span');
    span.className = 'text';
    span.textContent = text;
    li.appendChild(span);
    if (exampleNotes[text]) li.appendChild(needs(exampleNotes[text]));
    if (unreferenced && EDIT_PROMPT.test(text)) li.appendChild(needs('needs a reference image'));
    const slots = [...new Set(templateSlots(text).map((slot) => slot.text))];
    if (slots.length) {
      li.appendChild(needs(slots.length === 1 ? `fill in ${slots[0]}`
        : slots.length === 2 ? `fill in ${slots[0]} and ${slots[1]}`
        : `fill in ${slots[0]} and ${slots.length - 1} more slots`));
    }
    li.title = text;
    li.onclick = () => pickExample(text);
    li.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickExample(text); } };
    els.examplesMenu.appendChild(li);
  });
}

// The authors' template slots in `text` - "[character]", "[location]" -
// in reading order, from the placeholder rule's own patterns (`guide.slots`).
function templateSlots(text) {
  return exampleSlots
    .flatMap((pattern) => [...text.matchAll(pattern)])
    .map((match) => ({ text: match[0], start: match.index, end: match.index + match[0].length }))
    .sort((a, b) => a.start - b.start);
}

// An example written the way typing writes it, so Ctrl+Z brings back what
// the box held - and, over a prompt of the user's own, a note under the box
// says so with an Undo: the menu opens over the prompt, and one stray click
// used to replace minutes of work for good, the panel saving it at once.
// An example that is a template has its first slot selected, so what is
// typed next replaces it: left at the end, Generate rendered "[character]"
// as words. Selecting writes nothing - the Undo and the panel are the same.
function pickExample(text) {
  const before = els.prompt.value;
  writeBox(els.prompt, text);
  promptReplaced();
  showExamples(false);
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  els.prompt.focus();
  const [slot] = templateSlots(els.prompt.value);
  if (slot) {
    els.prompt.setSelectionRange(slot.start, slot.end);
    scrollToSelection(els.prompt);
  }
  if (before.trim() && before !== text) {
    offerPromptUndo('An example replaced your prompt.', () => {
      writeBox(els.prompt, before);
      promptReplaced();
      schedulePreview();
      scheduleTagValidation(0);
      persistPanel();
    });
  }
}
