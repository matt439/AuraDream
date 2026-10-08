// Local Diffusion's page script, last of twelve: the event wiring and the boot.
// Everything called here is declared in the files index.html loads before
// it; state.js has the load-order contract.
'use strict';

// --------------------------------------------------------------------- wire

let searchTimer;
els.generate.onclick = generate;
els.cancel.onclick = cancelJob;
els.pause.onclick = togglePause;
els.cancelQueued.onclick = cancelQueued;
els.newImage.onclick = goToNewImage;
els.dice.onclick = () => { els.seed.value = ''; syncRuns(); schedulePreview(); persistPanel(); };
els.reset.onclick = resetPanel;
els.orientation.onclick = swapOrientation;

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

// The menu's keyboard: ArrowDown and ArrowUp walk it (round the ends), Home
// and End jump; leaving it - Tab, a click elsewhere - closes it, rather than
// leaving it open over the prompt.
els.examplesBtn.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowDown') return;
  e.preventDefault();
  if (els.examplesMenu.hidden) showExamples(true);
  els.examplesMenu.querySelector('[role="menuitem"]')?.focus();
});
els.examplesMenu.addEventListener('keydown', (e) => {
  const items = [...els.examplesMenu.querySelectorAll('[role="menuitem"]')];
  const at = items.indexOf(document.activeElement);
  const next = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: items.length - 1 }[e.key];
  if (next === undefined || !items.length) return;
  e.preventDefault();
  items[(next + items.length) % items.length].focus();
});
els.examplesBox.addEventListener('focusout', (e) => {
  if (!els.examplesMenu.hidden && !els.examplesBox.contains(e.relatedTarget)) showExamples(false);
});
els.promptNoteUndo.onclick = undoPromptWrite;
els.promptNoteDiscard.onclick = discardPromptNote;

// ERNIE's Expand: the prompt rewritten once by the checkpoint's own rewriter,
// visibly, then the English of it - the button's state is status.js's
// (syncExpandButton). What it does is said under the prompt, where it was
// pressed (tags.js, showPromptNote): while it runs, with a clock, what it
// waits on - the model's load, a first use's download - and when it lands,
// that it replaced the prompt, with an Undo (from 1800 px at the end of its
// English, at the head of the results). Its failures go in the panel's
// line, said as Expand's - under the prompt instead while that line holds a
// Reuse's Undo (`expandSay`). A first use can take minutes:
//   - typing meanwhile is never overwritten; the result is held under the
//     prompt (Use it, Discard) rather than thrown away, since it is sampled
//     and cannot be made again exactly.
//   - a Reset, a Reuse, an Example or a Use put another prompt there on
//     purpose, and the result goes without a word.
//   - nothing it says lands over an Undo another press left in the panel's
//     line (a Reuse's), which would take the button and the panel it puts
//     back (`expandSay`). Reset's Undo is under the prompt, where this one's
//     running line gives way to it; an Example's, or a held expansion, the
//     running line took is offered again where the expansion fails.
// Focus stays on the button through the request and after a failure, and
// goes to the prompt only with the text it brings.
let expandRun = null;
// What Expand last said - in the panel's line ({text}) or under the prompt
// ({note}) - a failure, a press while it ran: the next Expand, and the end of
// this one, take it back while it is still what is said there.
let expandLine = null;
// PromptExpansionRequest's `max_length` (tests/test_ernie.py holds them
// equal), checked here so a long prompt is refused in the panel's words, at
// the box, rather than in pydantic's.
const EXPAND_MAX_CHARS = 4000;

async function expandPrompt() {
  // Held: the press says why, as a held Unload's does.
  if (els.expandPrompt.getAttribute('aria-disabled') === 'true') {
    if (serverOffline) { showError(OFFLINE_MESSAGE); void refreshStatus(); }
    else if (expandingPrompt) {
      // Beside a Reuse's Undo it would be said under the prompt, where the
      // running line says as much already. What becomes of the result is
      // said as it will be decided now (expandOutcome): the box changed
      // meanwhile keeps the change.
      if (!(reuseNote?.button?.isConnected && promptUndo === expandRun?.note)) {
        expandSay({
          apply: 'Expanding… - the result goes into the prompt box when it lands.',
          hold: 'Expanding… - the prompt has been edited since, so the result will wait under it (Use it, Discard).',
          drop: 'Expanding… - the prompt was replaced since, so the result will be dropped.',
        }[expandOutcome(expandRun)], true);
      }
    } else if (busy) {
      // Said until this page's job ends (setBusy takes the line back), and
      // through expandSay, which puts it under the prompt rather than over a
      // Reuse's Undo in the panel's line.
      const said = expandSay(`${els.expandPrompt.title}.`, true);
      if (said.text) jobNote = said.text;
    }
    // Another tab's job (syncExpandButton): said until it ends, as Unload's
    // is - setElsewhereBusy takes back the line it left in the panel.
    else if (elsewhereBusy) {
      const said = expandSay(`${EXPAND_ELSEWHERE}.`, true);
      if (said.text) elsewhereNote = said.text;
    } else expandSay(`${els.expandPrompt.title}.`, true);
    return;
  }
  const source = els.prompt.value;
  const prompt = source.trim();
  if (!prompt) { showError('Enter a prompt to expand first.'); els.prompt.focus(); return; }
  // What the server would refuse in pydantic's words, said in the panel's, at
  // the box: a prompt past the request's length (in code points, as Python
  // counts), and the size ERNIE writes the expansion for - whole pixels,
  // inside the boxes' range.
  const length = [...prompt].length;
  if (length > EXPAND_MAX_CHARS) {
    showError(`Expand takes a prompt of at most ${EXPAND_MAX_CHARS} characters; this one has ${length}.`);
    els.prompt.focus();
    return;
  }
  const side = [els.width, els.height].find((box) => {
    const value = +box.value;
    return box.value.trim() === '' || !Number.isInteger(value) || value < +box.min || value > +box.max;
  });
  if (side) {
    showError(`Expand writes for the size in W and H: set both to whole pixels from ${els.width.min} to ${els.width.max} first.`);
    side.focus();
    return;
  }
  // The last expansion's failure is about that one.
  clearExpandLine();
  // The English of the prompt it replaces, for its Undo to bring back.
  const before = translatedSource === source ? { source, text: els.promptTranslation.value } : null;
  // Whether the box was rewritten whole meanwhile (replaceBoxText's flag),
  // rather than typed in. And the note the running line takes the place of -
  // an Example's Undo or a held expansion - to offer again where this fails.
  // `ticket` is this press's name to the server, which the status gives back
  // while the expansion holds the card: any other holder is one it waits for
  // (expandRunningText).
  const run = {
    source, replaced: false, text: '', note: null, clock: 0, from: Date.now(),
    displaced: ['write', 'held'].includes(promptUndo?.kind) ? promptUndo : null,
    ticket: expandTicket(),
  };
  const watch = () => { if (replacingBox) run.replaced = true; };
  els.prompt.addEventListener('input', watch);
  expandingPrompt = true;
  expandRun = run;
  syncPromptExpansion();
  showExpandRunning();
  // The topbar's "loading…" where the press starts the load, not at the
  // next poll, as a job's submission asks too.
  setTimeout(() => { void refreshStatus(); }, 250);
  try {
    let body;
    try {
      body = await api('/api/prompt/expand', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, width: +els.width.value, height: +els.height.value, ticket: run.ticket }),
      });
    } catch (err) {
      endExpandRunning(run);
      expandFailed(err);
      offerDisplaced(run);
      return;
    }
    endExpandRunning(run);
    // "Expanding…", said to a press while it ran, is over.
    clearExpandLine();
    const outcome = expandOutcome(run);
    if (outcome === 'hold') holdExpansion(body);
    if (outcome === 'apply') applyExpansion(body, source, before);
  } finally {
    els.prompt.removeEventListener('input', watch);
    clearInterval(run.clock);
    expandRun = null;
    expandingPrompt = false;
    syncPromptExpansion();
    // The topbar's "loading…" over, and the title's first-use cost with it,
    // now rather than at the next poll.
    void refreshStatus();
  }
}

// What a result does with the box as it is now: written in where the box
// still holds the prompt it was made from; held under it (Use it, Discard)
// where that was typed over; dropped where a Reset, a Reuse, an Example or a
// Use rewrote the box whole meanwhile (`replacingBox`), which was on purpose.
function expandOutcome(run) {
  if (els.prompt.value === run.source) return 'apply';
  return run.replaced ? 'drop' : 'hold';
}

// Expand's words in the panel's line - or, while a Reuse's Undo is on offer
// there, under the prompt, where Expand was pressed: written over, the Undo's
// button went with the text, and the panel it would put back with it; left
// unsaid, a failure looked like nothing happening. Under the prompt it goes
// as the panel line's words do not - once the box holds anything else, or
// after 20 s, since it sits over the next field's caption - and the note it
// takes the place of is the running line, or one Expand said itself: the
// Reuse that put that Undo there took an Example's. Answers where it went,
// {text} or {note}.
const EXPAND_NOTE_MS = 20000;
function expandSay(message, note = false, options = {}) {
  clearExpandLine();
  if (reuseNote?.button?.isConnected) {
    expandLine = { note: showPromptNote(message, { kind: 'info', ms: EXPAND_NOTE_MS }) };
  } else {
    showError(message, note, options);
    expandLine = { text: message };
  }
  return expandLine;
}

function clearExpandLine() {
  const said = expandLine;
  expandLine = null;
  if (said?.note) {
    if (promptUndo === said.note) dropPromptUndo();
  } else if (said && els.error.textContent === said.text) showError('');
}

// The note the running line took the place of, offered again where the
// expansion failed and nothing has taken its place since: an Example's Undo
// while the box still holds what the Example wrote, a held expansion
// whatever it holds - Ctrl+Z reached the one, and nothing the other. Not
// after a Reset, a Reuse or an Example meanwhile, which replaced it on purpose.
function offerDisplaced(run) {
  const was = run.displaced;
  if (!was || promptUndo || run.replaced) return;
  if (was.kind === 'write' && els.prompt.value !== was.after) return;
  was.again();
}

// The running line, with what the press waits on, in the order the server
// does it: the model's load, the rewriter's first download or its load, the
// expansion, the translator's first download, the translation. A clock
// counts beside it, aria-hidden, as the load's does: a first use is minutes
// with nothing else moving. The status's `prompt_expander` gives each
// download's step (`fetch[].step`), and refreshStatus asks for the line
// again as that and the model's state move (syncExpandRunning).
function expandRunningText() {
  const name = checkpointName(loadedModel) || 'the model';
  const s = expanderStatus;
  const fetch = s?.fetch ?? [];
  const download = (f) => `downloading ${f.name} (about ${humanBytes(f.bytes)}) once`;
  const ahead = [];
  // The card held by anything but this press: another tab's critic or
  // Expand, which the server lets finish first (run_when_idle) - the load
  // waits for it too. "Expanding…" was said for the whole of their run.
  const holder = cardHolder && !(cardHolder === 'expand' && cardTicket === expandRun?.ticket) ? cardHolder : null;
  if (holder) ahead.push(`waiting for the card, which ${CARD_HOLDERS[holder] ?? 'something else'} has`);
  if (modelState !== 'ready') ahead.push(`loading ${name}`);
  const rewriter = fetch.filter((f) => f.step === 'expand');
  if (rewriter.length) ahead.push(...rewriter.map(download));
  else if (s && !s.loaded) ahead.push("loading ERNIE's prompt rewriter");
  const after = fetch.filter((f) => f.step === 'translate').map(download);
  if (!ahead.length && !after.length) return 'Expanding with ERNIE, then translating locally…';
  const line = [...ahead, 'expanding', ...after, 'translating…'].join(', then ');
  return `${line[0].toUpperCase()}${line.slice(1)}`;
}

// Who has the card, as the running line names it: this page's own critic
// and jobs hold Expand, so a critic or an Expand holding it is another tab's
// - and a job, one that tab queued while this expansion's load ran.
const CARD_HOLDERS = {
  critic: "another tab's prompt critic",
  expand: "another tab's Expand",
  job: "another tab's image job",
};

// A press's name for the server (PromptExpansionRequest.ticket).
function expandTicket() {
  try { return crypto.randomUUID(); } catch { return `${Date.now()}-${Math.random().toString(36).slice(2)}`; }
}

function showExpandRunning() {
  const run = expandRun;
  if (!run) return;
  run.text = expandRunningText();
  run.note = showPromptNote(run.text, { kind: 'running', ms: 0 });
  const clock = document.createElement('span');
  clock.className = 'clock';
  clock.setAttribute('aria-hidden', 'true');
  els.promptNoteText.append(clock);
  const tick = () => {
    if (promptUndo !== run.note) return;
    const s = Math.max(0, Math.floor((Date.now() - run.from) / 1000));
    clock.textContent = ` · ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };
  tick();
  clearInterval(run.clock);
  run.clock = setInterval(tick, 1000);
}

// From refreshStatus, as the model's state moves: the line says what the
// expansion waits on now - while it is still the note shown.
function syncExpandRunning() {
  const run = expandRun;
  if (!run || promptUndo !== run.note) return;
  const text = expandRunningText();
  if (text === run.text) return;
  run.text = text;
  els.promptNoteText.firstChild.textContent = text;
}

function endExpandRunning(run) {
  clearInterval(run.clock);
  if (run.note && promptUndo === run.note) dropPromptUndo();
}

// The expansion into the prompt box, as typing writes it (Ctrl+Z gives back
// what it replaced), its English shown after the write, whose `input` would
// clear it - and the review from the top: the caret and both boxes' scroll at
// their start, where the caret at the end had scrolled a long expansion's
// opening lines out of the box. The note under the prompt has its Undo:
// the prompt it replaced, with that prompt's own English if it had one.
function applyExpansion(body, source, before) {
  rewriteBox(els.prompt, body.expanded_prompt);
  promptReplaced();
  if (body.translated_prompt) showPromptTranslation(body.expanded_prompt, body.translated_prompt);
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  els.prompt.setSelectionRange(0, 0);
  els.prompt.scrollTop = 0;
  els.promptTranslation.scrollTop = 0;
  // From 1800 px, where the English heads the results, the note is said
  // there, at the end of its box: under the prompt, in the panel's flow while
  // the negative is greyed - ERNIE's CFG 1 - it pushed Generate below a
  // 2048x1040 window at 125%.
  const beside = !!body.translated_prompt && TRANSLATION_WIDE.matches;
  const said = !body.translated_prompt ? 'Prompt expanded: review it before generating.'
    : beside ? 'Prompt expanded: review it and this translation before generating.'
    : 'Prompt expanded: review it and its English translation before generating.';
  offerPromptUndo(said, () => {
    rewriteBox(els.prompt, source);
    promptReplaced();
    if (before && els.prompt.value === before.source) showPromptTranslation(before.source, before.text);
    schedulePreview();
    scheduleTagValidation(0);
    persistPanel();
  }, {
    title: 'Put back the prompt Expand replaced',
    host: beside ? els.promptTranslationField : PROMPT_NOTE_HOME,
  });
  if (body.translation_detail) {
    expandSay(`Prompt expanded; its English translation is unavailable: ${trimStop(body.translation_detail)}.`, true);
  }
}

// An expansion the user typed over while it ran: kept under the prompt
// rather than thrown away - Use it writes it in as Expand would have, with
// the Undo back to what was typed; Discard, the next Expand, a Reset or a
// Reuse let it go.
function holdExpansion(body) {
  showPromptNote('Expansion ready - made from the prompt before your edits.', {
    kind: 'held',
    ms: 0,
    undo: () => applyExpansion(body, els.prompt.value, null),
    label: 'Use it',
    title: 'Replace the prompt with the expansion; Undo gives back what you typed',
    discard: () => {},
  });
}

// A failed expansion, said as Expand's, with the prompt as it was. A load that
// failed (a 503) is the status's to say: the viewer's card, where the viewer
// is empty, has the sentence and Retry load, so Expand only points there;
// beside images Expand says the sentence once, as the poll would - the poll
// then says it no more (shownLoadDetail) - its `command` marks made
// quotation marks. Retry load's success takes either back (loadModel), as its
// own messages: in the panel's line (`loadLine`) or under the prompt
// (`aboutLoad`).
function expandFailed(err) {
  if (err.status === 503) {
    void refreshStatus();
    const inViewer = loadFailureInViewer();
    const said = inViewer
      ? 'Expand needs the model, which could not load - why, and Retry load, are in the viewer.'
      : `Expand needs the model, which could not load: ${err.message.replace(/`([^`]*)`/g, '“$1”')}`;
    const at = expandSay(said, inViewer);
    if (!inViewer) shownLoadDetail = err.message;
    if (at.note) at.note.aboutLoad = true;
    else loadLine = said;
    return;
  }
  // A failure moves the prompt nowhere: the editor is as the user left it.
  expandSay(err.offline
    ? 'Expand did not finish: the server did not answer. The prompt is unchanged.'
    : `Expand did not finish: ${trimStop(err.message)}. The prompt is unchanged.`,
  false, { offline: !!err.offline });
}

const trimStop = (text) => String(text).trim().replace(/[.\s]+$/, '');

// A click from the keyboard (detail 0) - Enter or Space on the button - takes
// focus into the menu on its first item, as ArrowDown does: its items are no
// Tab stops, and Tab from the button went on to Tags, closing it unread.
els.examplesBtn.onclick = (e) => {
  const open = els.examplesMenu.hidden;
  showExamples(open);
  if (open && e.detail === 0) els.examplesMenu.querySelector('[role="menuitem"]')?.focus();
};
// A click from the keyboard takes focus into the helper too.
els.tagsBtn.onclick = (e) => showTagHelper(els.tagsMenu.hidden, { focus: e.detail === 0 });
els.expandPrompt.onclick = expandPrompt;
// The prompt check: Check and Check again run it on the prompt as it is now;
// the rest of its buttons are built with the results they act on.
els.analyze.onclick = () => { void checkFromButton(); };
els.analysisAgain.onclick = () => { void checkPrompt(); };
els.analysisClose.onclick = closeAnalysis;
els.criticAsk.onclick = askCritic;
els.compareSeeds.onchange = () => scheduleComparePreview(0);
els.compareBatch.onchange = () => scheduleComparePreview(0);
els.compareRun.onclick = runComparison;
els.compareOwn.addEventListener('input', compareOwnChanged);
els.compareOwnFill.onclick = compareOwnFromPrompt;
els.tagValidation.addEventListener('contextmenu', (event) => {
  // Keep the ordinary text-editing menu inside the inline input. Everywhere
  // else in the validation tray, including the flex gaps, is an insert point.
  if (event.target.closest('.tag-editor')) return;
  event.preventDefault();
  openTagInsert(tagInsertionIndexAtPoint(event.clientX, event.clientY));
});
// A gap takes a plain click too, and the pointer over one shows where the
// tag would go - the drop's bar, once a frame for the last position. Over the
// tray itself only, not a chip, the editor or + tag.
els.tagValidation.addEventListener('pointermove', (event) => {
  cancelAnimationFrame(tagHoverFrame);
  tagHoverFrame = 0;
  if (event.target !== els.tagValidation || draggedTagIndex !== null) {
    tagHoverPoint = null;
    clearTagDropMarkers();
    return;
  }
  const { clientX, clientY } = event;
  tagHoverPoint = { x: clientX, y: clientY };
  tagHoverFrame = requestAnimationFrame(() => { tagHoverFrame = 0; markTagGap(clientX, clientY); });
});
els.tagValidation.addEventListener('pointerleave', () => {
  cancelAnimationFrame(tagHoverFrame);
  tagHoverFrame = 0;
  tagHoverPoint = null;
  clearTagDropMarkers();
});
els.tagValidation.addEventListener('click', (event) => {
  if (event.target !== els.tagValidation) return;
  const index = tagInsertionIndexAtPoint(event.clientX, event.clientY, { gap: true });
  if (index === null) return;
  tagHoverPoint = null;
  clearTagDropMarkers();
  openTagInsert(index);
});
// Focus leaving the tray lets go of the chip it was on; a rebuild of the
// tray, which removes the focused chip, is not leaving it.
els.tagValidation.addEventListener('focusout', (event) => {
  if (tagRendering || els.tagValidation.contains(event.relatedTarget)) return;
  tagFocusIndex = null;
});
els.prompt.addEventListener('input', (event) => {
  if (translatedSource !== null && els.prompt.value !== translatedSource) clearPromptTranslation();
  syncAnalysisStale();
  noteTagTyping(event);
  syncTagSelection();
  scheduleTagSuggestions();
  scheduleTagValidation();
});
// The check read the negative prompt too; editing it dims the check as well.
els.negative.addEventListener('input', () => syncAnalysisStale());
// The caret moved: suggestions for the item it is in now, the item typed
// before it checked, and the Tags helper's note on where a tag goes.
els.prompt.addEventListener('click', () => { scheduleTagSuggestions(0); followTagCaret(); syncTagSelection(); });
els.prompt.addEventListener('keyup', (event) => {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) scheduleTagSuggestions(0);
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
    followTagCaret();
    syncTagSelection();
  }
});
// Leaving the box ends the typing, and the list goes with it: Tab, with
// nothing selected, moves on, and must not leave suggestions open behind.
els.prompt.addEventListener('blur', () => { hideTagSuggestions(); endTagTyping(); });
els.prompt.addEventListener('keydown', (event) => {
  if (els.tagSuggestions.hidden || event.ctrlKey || event.metaKey) return;
  if (event.key === 'ArrowDown') {
    event.preventDefault();
    selectTagSuggestion(tagSuggestionIndex + 1);
  } else if (event.key === 'ArrowUp') {
    event.preventDefault();
    selectTagSuggestion(tagSuggestionIndex - 1);
  } else if ((event.key === 'Enter' || event.key === 'Tab') && tagSuggestionIndex >= 0) {
    event.preventDefault();
    applyTagSuggestion(tagSuggestions[tagSuggestionIndex]);
  } else if (event.key === 'Escape') {
    event.preventDefault();
    hideTagSuggestions();
  }
});
document.addEventListener('click', (e) => {
  if (!els.examplesMenu.hidden && !e.target.closest('#examples')) showExamples(false);
  // The helper hangs under the prompt, which takes clicks to move its caret.
  if (!els.tagsMenu.hidden && !e.target.closest('#tag-helper, #tags-menu, #prompt')) showTagHelper(false);
  // The token line is in the editor (under the box) but no part of typing.
  if (!e.target.closest('.prompt-editor') || e.target === els.promptTokens) hideTagSuggestions();
});
els.unload.onclick = unloadModel;
els.more.onclick = () => loadGallery();
// One card at a time in the tab order, walked with the arrow keys (gallery.js).
els.gallery.addEventListener('keydown', galleryKey);
els.gallery.addEventListener('focusin', galleryFocus);
// The gallery beside the viewer, from 2000 px where ticked (gallery.js): the
// tick, the window crossing that width, and the tick as this browser left it.
els.besideTick.onchange = toggleBeside;
matchMedia(BESIDE_MEDIA).addEventListener('change', syncBeside);
// A finished job's images wait for the pointer to leave its column there.
els.libraryScroll.addEventListener('pointerenter', () => { galleryPointer = true; });
els.libraryScroll.addEventListener('pointerleave', galleryPointerLeft);
restoreBeside();
// Scrolling to the end of the gallery fetches the next page: the trigger is
// the button coming within a few hundred pixels of the viewport - or of the
// gallery's own column, beside the viewer - so the page is usually there
// before the button is. Hidden - no older rows, or the sheet - it never
// intersects. `loadGallery` serialises a click racing this.
if (!moreWatcher) armMore();
els.gridRun.onclick = () => startJob('/api/grid', gridPayload());
// The preview reflects the panel as much as the axes: a blank axis is the
// control's own value, and both the size and the batch move the totals.
// The negative prompt is in the list because a `{name}` placeholder may live
// there - the server validates it - and Resume because the whole "N already
// done" half of the preview is what ticking it is for. Repeat (`runs`) moves
// nothing a grid runs, but the preview says so while it is above one.
for (const el of [els.gridPrompt, els.gridSteps, els.gridCfg, els.gridMu, els.gridSeeds,
                  els.gridSeedList, els.gridBudget, els.prompt, els.negative, els.steps,
                  els.cfg, els.trueCfg, els.mu, els.batch, els.runs, els.width, els.height,
                  els.seed, els.gridLoraScale, els.loraScale]) {
  el.addEventListener('input', schedulePreview);
}
// A listed seed axis makes the count and the base seed moot; say so in place.
els.gridSeedList.addEventListener('input', () => {
  els.gridSeeds.disabled = els.gridSeedList.value.trim() !== '';
});
els.load.onclick = loadModel;
els.gridResume.addEventListener('change', previewGrid);
els.gridSection.addEventListener('toggle', previewGrid);
// On the summary's click (Enter and Space on it are clicks too), not on the
// toggle event, which restorePanel and resetPanel set off as well; the fold
// is open by the next frame. And again once its preview lands (previewShown).
els.gridSection.querySelector('summary').addEventListener('click', () => {
  gridRevealPending = true;
  requestAnimationFrame(revealGridRun);
});
els.width.oninput = els.height.oninput = syncSize;
// These run before the panel's delegated persist handler on the same event -
// the target's listener fires before its ancestors' - so the `owned` flags
// are current by the time the panel is written.
els.steps.oninput = () => own('steps');
els.mu.oninput = () => own('mu');
// The seed is not a sampling value and has no business freezing the profile.
els.seed.oninput = syncRuns;
els.batch.oninput = els.runs.oninput = syncRuns;
els.cfg.oninput = () => { own('cfg'); syncCfg(); };
// The weight boxes are dead without an adapter, so selecting one has to wake
// them - and clearing it has to stop a stale axis reaching the server.
els.lora.onchange = () => { syncConditioning(); schedulePreview(); };
els.trueCfg.oninput = () => { own('trueCfg'); syncCfg(); };
// The greyed negative box points at the box that wakes it.
els.negEnable.onclick = focusNegativeKnob;
els.favOnly.onchange = resetGallery;
// The Runs picker is the gallery's table of contents, so it follows the scope.
els.allModels.onchange = () => { resetGallery(); loadRuns(); };
els.sheetBtn.onclick = toggleSheet;
// How the cards are drawn (gallery.js): a press chooses; each pair is one
// tab stop whose arrows move focus along it, as a size strip's do.
els.cardView.addEventListener('click', cardViewClick);
for (const pair of els.cardView.querySelectorAll('[role="group"]')) pair.addEventListener('keydown', (e) => stepPreset(pair, e));
els.sheetGrid.addEventListener('keydown', (e) => { void sheetKey(e); });
// The sheet's fit is measured against the window, so it follows a resize.
window.addEventListener('resize', refitSheet);
els.rerender.onclick = rerenderSelection;
els.deleteUnstarred.onclick = deleteUnstarred;
// The picker then names the run on screen: resetGallery follows the search box.
// Browsed from the keyboard, a run opens on Enter or as focus leaves (runs.js).
els.runList.onchange = runListChange;
els.runList.addEventListener('keydown', runListKey);
els.runList.addEventListener('blur', commitRunList);
els.lastRunShow.onclick = () => {
  const { run, job } = els.lastRun.dataset;
  if (run) showRun(run);
  else if (job) showJob(job);
  dismissLastRun();
};
els.lastRunResume.onclick = resumeLastRun;
els.lastRunDismiss.onclick = dismissLastRun;
// A first run's empty viewer offers the checkpoint's examples.
els.viewerExamples.onclick = viewerExamples;
// A batch of two to four is laid out to fit the strip (viewer.js, fitViewer):
// again when its width moves, and with the window's height, which its 68vh is
// - as a long run's scrolling strip is let grow by its padding (settleStrip).
new ResizeObserver(viewerResized).observe(els.viewer);
window.addEventListener('resize', () => { fitViewer(); settleStrip(false); });
els.search.oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(resetGallery, 250); };
// The user's own closes - the cross, the backdrop, Escape below - hand focus
// to the image the walk ended on (dismissLightbox); every other close is
// closeLightbox alone.
els.lbClose.onclick = dismissLightbox;
els.lbPrev.onclick = () => stepLightbox(-1);
els.lbNext.onclick = () => stepLightbox(1);
// The just-generated image only exists in the viewer, so reuse has to be
// reachable from the lightbox as well as from a gallery card. Closed first:
// Reuse writes the prompt as typing would, which needs the focus a modal
// lightbox keeps to itself.
els.lbReuse.onclick = () => { const rec = lbList[lbIndex]; if (rec) { closeLightbox(); reuse(rec); } };
// Refine likewise: openRefine closes the lightbox for its dialog, and a close
// that runs nothing gives it back (refineClosed). Where the checkpoint has no
// such pass the lightbox stays open, since the note saying so is mirrored
// into it.
els.lbRefine.onclick = () => { const rec = lbList[lbIndex]; if (rec) openRefine(rec); };
els.lbFav.onclick = () => { const rec = lbList[lbIndex]; if (rec) setFavorite(rec, !rec.favorite); };
els.lbDelete.onclick = () => { const rec = lbList[lbIndex]; if (rec) deleteImage(rec); };
els.lightbox.onclick = (e) => { if (e.target === els.lightbox) dismissLightbox(); };
// A pass against its source: the button (and S) turns it on and off, a mode
// turns it on in that mode or switches to it.
els.lbCompareOn.onclick = () => { void toggleCompare(); };
els.lbSwipe.onclick = () => { void toggleCompare('swipe'); };
els.lbSideBySide.onclick = () => { void toggleCompare('side'); };
// Every close, the native Escape's included, resets the strip - unless the
// lightbox is open again by the time the event comes: the dialog's `close`
// is queued, not fired by close() itself, and a close followed at once by an
// open (a script's, a smoke's) had the late event shut the new lightbox, and
// reset its strip, under whatever it was doing - a delete in flight.
els.lightbox.addEventListener('close', () => { if (!els.lightbox.open) closeLightbox(); });
// The pictures: one that will not load says why, the next that loads takes it
// back, and a source that will not load ends the comparison.
els.lbImg.onerror = () => { void lbImageFailed(); };
els.lbImg.onload = lbImageLoaded;
els.lbSrcImg.onerror = lbSourceFailed;
// Zoom and pan: the wheel non-passive, or the page behind the dialog
// scrolls; a press, a drag, a release and a double-click on the stage; an
// arrow's press, for the stage to know a double-click begun on it.
els.lbMain.addEventListener('wheel', lbWheel, { passive: false });
for (const nav of [els.lbPrev, els.lbNext]) nav.addEventListener('pointerdown', lbArrowDown);
els.lbStage.onpointerdown = lbPointerDown;
els.lbStage.onpointermove = lbPointerMove;
els.lbStage.onpointerup = lbPointerUp;
els.lbStage.onpointercancel = lbPointerCancel;
els.lbStage.ondblclick = lbDoubleClick;
els.lbSplit.onkeydown = lbSplitKey;
els.lbFit.onclick = zoomFit;
els.lbOne.onclick = () => zoomOneToOne();
window.addEventListener('resize', layoutLightbox);
// And the stage's own size, whatever moves it: under 901 px a line in
// #lb-error (Use seed's, a delete's note) is a row of its own that took 30 px
// off the stage, and the picture laid out before kept its height, its foot
// cut off under a readout still saying "fit". Loop-safe: the layout writes
// the facts' room only when it changes (lbFactsRoom), from the stage as it
// would be with the facts at six lines.
new ResizeObserver(() => { if (els.lightbox.open) layoutLightbox(); }).observe(els.lbStage);

// The Refine dialog. Every box asks the server's preview again, as the grid's
// do; the scale buttons are built with the source they multiply. Keep shape
// moves the other side before the box's own refineChanged reads the pair.
els.refineRun.onclick = () => { void runRefine(); };
// Each close is held while the pass's POST is out (closeRefine), Escape's too.
els.refineCancel.onclick = els.refineClose.onclick = closeRefine;
els.refineDialog.addEventListener('cancel', refineCancelEvent);
els.refineDialog.addEventListener('close', refineClosed);
els.refineDialog.addEventListener('keydown', refineKeydown);
els.refineWidth.addEventListener('input', refineSideTyped);
els.refineHeight.addEventListener('input', refineSideTyped);
els.refineDice.onclick = () => { els.refineSeed.value = ''; refineChanged(); };
els.refineCompare.onclick = fillRefineCompare;
els.refineStartOver.onclick = startRefineOver;
els.refineMore.addEventListener('toggle', sizeRefinePrompt);
REFINE_WIDE.addEventListener('change', showRefineSource);
// The pass's own prompt check reads the text and whether the negative runs.
els.refinePrompt.addEventListener('input', () => { sizeRefinePrompt(); scheduleRefineTags(); });
// Its tag suggestions, on a tag checkpoint: typing, a click and the caret
// keys ask; leaving the box closes the list; the box's keys walk it, and an
// Escape it takes is not the dialog's.
els.refinePrompt.addEventListener('input', suggestRefineTags);
els.refinePrompt.addEventListener('click', suggestRefineTags);
els.refinePrompt.addEventListener('keyup', refinePromptCaret);
els.refinePrompt.addEventListener('keydown', refinePromptKeydown);
els.refinePrompt.addEventListener('blur', hideRefineSuggestions);
els.refineFromPanel.onclick = copyRefineFromPanel;
for (const el of [els.refinePrompt, els.refineNegative, els.refineCfg, els.refineTrueCfg]) {
  el.addEventListener('input', refineCheckChanged);
}
for (const el of [els.refineWidth, els.refineHeight, els.refineStrength, els.refineVersions,
                  els.refineSeed, els.refinePrompt, els.refineNegative, els.refineSteps,
                  els.refineCfg, els.refineMu, els.refineTrueCfg]) {
  el.addEventListener('input', refineChanged);
}
els.refineVersions.addEventListener('change', refineChanged);

// A delete's question (gallery-actions.js): Delete, Cancel, Enter in the count's box,
// a press outside it. Escape is the dialog's own cancel, answered as Cancel.
els.confirmOk.onclick = () => els.confirmDialog.close('ok');
els.confirmCancel.onclick = () => els.confirmDialog.close('');
els.confirmDialog.addEventListener('close', askClosed);
els.confirmDialog.addEventListener('click', askBackdrop);
els.confirmCount.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); els.confirmDialog.close('ok'); }
});

document.addEventListener('keydown', (e) => {
  // Nothing of the page's while a delete asks: Escape there is its Cancel,
  // which closed the lightbox under it as well, and Ctrl+Enter generated.
  if (els.confirmDialog.open) return;
  // Ctrl/Cmd+Enter submits whatever the focus is in: inside the Refine dialog
  // it runs the pass and inside the open grid fold it runs the grid, which is
  // what their boxes are for - from a grid box Generate would render the
  // prompt's {placeholders} literally. Anywhere else, the prompt box included,
  // it generates. Run grid's click, so its handler and guards are the press's.
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    // Taken, so the Enter goes nowhere else: the button a closing dialog
    // hands focus back to - the card's ⤢ - read it as a press of its own.
    e.preventDefault();
    if (els.refineDialog.open) void runRefine();
    else if (els.gridSection.open && e.target.closest?.('#grid-section')) els.gridRun.click();
    else generate();
    return;
  }
  // The lightbox's Escape before the menus': it is modal over a menu left
  // open under it - the Tags helper stays open when focus leaves it by
  // keyboard - and the menu's branch took the key, so the dialog's own
  // cancel closed the lightbox without dismissLightbox, focus going back to
  // the image it was opened on rather than the one the walk ended on. The
  // menu stays open under the page, as it was before the lightbox came up.
  if (e.key === 'Escape' && !e.defaultPrevented && els.lightbox.open) { dismissLightbox(); return; }
  // An Escape the prompt's suggestion list or a chip's editor took closes that
  // alone: the helper stays open under the box while its list comes and goes.
  if (e.key === 'Escape' && !e.defaultPrevented && !els.examplesMenu.hidden) { showExamples(false); return; }
  if (e.key === 'Escape' && !e.defaultPrevented && !els.tagsMenu.hidden) { showTagHelper(false); return; }
  // Then the prompt check, as its Close does - not under a dialog, whose
  // Escape is its own (the lightbox's below), nor in the middle of a
  // composition, nor in a search box with text, which Escape empties.
  if (e.key === 'Escape' && !e.defaultPrevented && !e.isComposing && !els.promptAnalysis.hidden
      && !els.lightbox.open && !els.refineDialog.open
      && !(e.target.matches?.('input[type="search"]') && e.target.value)) {
    escapeAnalysis();
    return;
  }
  if (!els.lightbox.open) return;
  // A key the lightbox's own control took - the swipe divider's arrows.
  if (e.defaultPrevented) return;
  // Arrows walk whichever strip the lightbox was opened from. Nothing else on
  // the page reads them while it is up, so the caret keys are free to take.
  // Comparing a pass, they swap to another pass of the same source where the
  // strip has one (`stepPass`), the comparison kept, and Shift+arrows move a
  // swipe's divider (`nudgeSplit`); otherwise they walk, as ever.
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();
    const delta = e.key === 'ArrowLeft' ? -1 : 1;
    if (e.shiftKey ? nudgeSplit(delta) : stepPass(delta)) return;
    void stepLightbox(delta);
    return;
  }
  // F, S, Delete and the zoom keys act on the image in view - unless a text
  // field still holds the caret, where they are typing, or a modifier makes
  // them the browser's: Ctrl+F is find, Ctrl+plus the page's zoom.
  if (e.target.matches('input, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
  // F does what the Star button's press does, and says so where focus is not
  // on that button (setFavorite's `say`): its state alone changed.
  if (e.key === 'f' || e.key === 'F') { const rec = lbList[lbIndex]; if (rec) void setFavorite(rec, !rec.favorite, { say: true }); }
  else if (e.key === 's' || e.key === 'S') void toggleCompare();
  else if (e.key === 'Delete') els.lbDelete.click();
  else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomLightbox(ZOOM_STEP); }
  else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomLightbox(1 / ZOOM_STEP); }
  else if (e.key === '0') zoomFit();
  else if (e.key === '1') zoomOneToOne();
});

// A file dragged over the page is dropped nowhere: the page takes no files,
// and the browser's own drop opened the file in the app's place.
for (const type of ['dragover', 'drop']) {
  document.addEventListener(type, (e) => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'none';
  });
}

window.addEventListener('resize', placeTagValidation);
placeTagValidation();
TRANSLATION_WIDE.addEventListener('change', placeTranslation);
placeTranslation();
new ResizeObserver(chipsResized).observe(els.tagValidation);
// The panel's sticky top (panel.js has the rules). It follows the panel's
// height - the grid fold, the progress, the last-run banner, the hints all
// change it - and the window's; the observer also reports once on observing,
// which places it at load. It moves with the user's scrolling, and for a
// control focused out of sight - but not for focus a press in the panel gave,
// where the control is under the pointer already. The press ends after the
// click, which is where a label hands its control focus.
new ResizeObserver(panelResized).observe(document.querySelector('.panel'));
window.addEventListener('resize', placePanel);
window.addEventListener('scroll', followScroll, { passive: true });
window.addEventListener('scrollend', () => { panelPos.scrolling = panelPos.dragging; });
window.addEventListener('wheel', userScrolls, { passive: true });
window.addEventListener('touchmove', userScrolls, { passive: true });
window.addEventListener('keydown', (e) => {
  if (e.defaultPrevented || !SCROLL_KEYS.has(e.key)) return;
  if (!e.target.closest('input, textarea, select, button, summary, a, [tabindex], [contenteditable]')) userScrolls();
});
window.addEventListener('pointerdown', (e) => {
  if (e.target !== document.documentElement && e.button !== 1) return;
  panelPos.dragging = e.target === document.documentElement;
  userScrolls();
});
document.querySelector('.panel').addEventListener('pointerenter', () => { panelPos.pointer = true; });
document.querySelector('.panel').addEventListener('pointerleave', () => {
  panelPos.pointer = false;
  if (panelPos.deferred) placePanel();
  promptTokensAimed = false;
  if (promptTokensHeld) syncPromptTokens();
});
// The token line under the prompt waits to show or go while the mouse is on a
// control it would move (tag-tokens.js, syncPromptTokens): one whose top is at or
// below the prompt box's bottom. Given back on leaving it and on a key.
const TOKENS_AIMED = 'button, a, input, select, textarea, summary, label, [role="option"], [role="menuitem"], [tabindex]';
document.querySelector('.panel').addEventListener('pointerover', (e) => {
  const control = e.pointerType === 'touch' ? null : e.target.closest(TOKENS_AIMED);
  promptTokensAimed = !!control && control !== els.prompt
    && control.getBoundingClientRect().top >= els.prompt.getBoundingClientRect().bottom - 0.5;
  if (promptTokensHeld && !promptTokensAimed) syncPromptTokens();
});
window.addEventListener('keydown', () => { if (promptTokensHeld) syncPromptTokens(true); }, true);
document.querySelector('.panel').addEventListener('pointerdown', () => { panelPos.press = true; });
window.addEventListener('pointerup', () => {
  panelPos.dragging = false;
  setTimeout(() => { panelPos.press = false; });
}, true);
window.addEventListener('pointercancel', () => { panelPos.press = panelPos.dragging = false; }, true);
document.querySelector('.panel').addEventListener('focusin', (e) => {
  if (!panelPos.press) revealInPanel(e.target);
});
// A tip is placed as it opens: entering its badge from outside, not moving
// between the badge and the tip it holds.
document.addEventListener('pointerover', (e) => {
  const help = e.target.closest?.('.help');
  if (help && !help.contains(e.relatedTarget)) placeTip(help);
});
document.addEventListener('focusin', (e) => { if (e.target.matches?.('.help')) placeTip(e.target); });
// A tip hung from the window by a badge in a scrolling box keeps up with it
// (panel.js, floatTip): any scroll, the box's own included.
document.addEventListener('scroll', followFloatingTip, { capture: true, passive: true });
buildPresets();
// Before the sync functions, which light the presets, the CFG note and the
// runs hint up from whatever is in the boxes.
restorePanel();
syncSize();
syncCfg();
syncRuns();
// Every keystroke and tick in the panel, plus the programmatic writes below
// - presets, Reuse, the dice, the seed written back after Run grid - which
// fire no input event.
document.querySelector('.panel').addEventListener('input', persistPanel);
document.querySelector('.panel').addEventListener('change', persistPanel);
els.gridSection.addEventListener('toggle', persistPanel);
restoreCardView();
refreshStatus();
loadGallery();
loadRuns();
resumeActiveJob();
setInterval(refreshStatus, 3000);
