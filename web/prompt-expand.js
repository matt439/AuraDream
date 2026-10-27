// AuraDream's page script: ERNIE's prompt Expand.
// Declarations only; its button is wired in app.js, and state.js has the
// shared-global load-order contract.
'use strict';

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
let expandEpoch = 0;
const stoppedExpansionTickets = new Set();
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
    revealSetupField(side);
    side.focus({ preventScroll: true });
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
    controller: new AbortController(), stopped: false,
    epoch: ++expandEpoch,
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
        signal: run.controller.signal,
      });
    } catch (err) {
      if (run.stopped || expandRun !== run) return;
      endExpandRunning(run);
      expandFailed(err);
      offerDisplaced(run);
      return;
    }
    if (run.stopped || expandRun !== run) return;
    endExpandRunning(run);
    // "Expanding…", said to a press while it ran, is over.
    clearExpandLine();
    const outcome = expandOutcome(run);
    if (outcome === 'hold') holdExpansion(body);
    if (outcome === 'apply') applyExpansion(body, source, before);
  } finally {
    els.prompt.removeEventListener('input', watch);
    clearInterval(run.clock);
    if (expandRun === run) {
      expandRun = null;
      expandingPrompt = false;
      els.promptExpandStop.hidden = true;
      syncPromptExpansion();
    }
    // The topbar's "loading…" over, and the title's first-use cost with it,
    // now rather than at the next poll.
    void refreshStatus();
  }
}

// Stop dismisses this answer immediately. The server's ticket owns the
// actual worker, which polls only between stages and keeps the GPU gate
// through an opaque download or rewrite. A new Expand has another ticket.
function stopExpandPrompt() {
  const run = expandRun;
  if (!run || run.stopped) return;
  run.stopped = true;
  stoppedExpansionTickets.add(run.ticket);
  if (stoppedExpansionTickets.size > 32) stoppedExpansionTickets.delete(stoppedExpansionTickets.values().next().value);
  run.controller.abort();
  expandRun = null;
  expandingPrompt = false;
  endExpandRunning(run);
  els.promptExpandStop.hidden = true;
  clearExpandLine();
  offerDisplaced(run);
  if (!promptUndo) {
    showPromptNote('Expansion abandoned. The prompt is unchanged; a download or model call already running can finish.',
      { kind: 'info', ms: EXPAND_NOTE_MS });
  }
  syncPromptExpansion();
  els.expandPrompt.focus({ preventScroll: true });
  void api(`/api/prompt/expand/${encodeURIComponent(run.ticket)}/stop`, { method: 'POST' }).catch((err) => {
    if (expandRun || run.epoch !== expandEpoch) return;
    expandSay(`Expansion abandoned here; the server could not confirm Stop: ${trimStop(err.message)}.`, true);
  });
  void refreshStatus();
}

function syncExpandStop() {
  els.promptExpandStop.hidden = !expandingPrompt;
  if (expandingPrompt && expandRun && !promptUndo) showExpandRunning();
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
  if (holder) {
    const whose = holder === 'expand' && stoppedExpansionTickets.has(cardTicket)
      ? 'the abandoned Expand' : CARD_HOLDERS[holder] ?? 'something else';
    ahead.push(`waiting for the card, which ${whose} has`);
  }
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
