// Local Diffusion's page script: the prompt critic and its rewrite.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------------- critic

// The critic button and its line. Called from syncPromptExpansion, so every
// change to the queue, Expand or the critic itself passes through here.
// Another tab's job holds it as this page's does (elsewhereBusy): the server
// refused it then, in words naming a job this page did not show.
const CRITIC_ELSEWHERE = 'A job started in another tab or window is running. The critic needs the card to itself, so it waits for that job to finish.';
// And what the server would refuse it under (main._refuse_critique_now) or
// could not be asked at all: Expand and Unload are held for both, and the
// critic was the one editor action left live - a press got "No answer from
// the Local Diffusion server — is it still running?." in the critic's block
// for good, or the route's 409 about the load.
const CRITIC_OFFLINE = 'The server is not answering.';
const CRITIC_LOADING = 'The image model is loading; the critic waits for the load.';
function criticHold() {
  return serverOffline ? CRITIC_OFFLINE
    : modelState === 'loading' ? CRITIC_LOADING
    : busy ? 'An image job is queued or running. The critic needs the card to itself, so it waits for the queue to finish.'
    : elsewhereBusy ? CRITIC_ELSEWHERE
    : expandingPrompt ? 'Wait for prompt expansion to finish.'
    : '';
}
function syncCritic() {
  els.analysisCritic.hidden = !promptCritiqueSupported;
  // aria-disabled, as Generate is held: disabled, the button just pressed lost
  // a keyboard user's focus to <body>. askCritic refuses each case itself.
  const hold = criticHold();
  els.criticAsk.setAttribute('aria-disabled', String(critiquing || !!hold));
  els.criticAsk.textContent = critiquing ? 'Thinking…' : 'Ask the critic';
  els.criticAsk.title = hold
    || "A local model reads the prompt against this checkpoint's guide and its authors' own texts, cites what it goes by, and suggests a rewrite";
  // Named as the answer's footer names it, and a first use's download only
  // what it would fetch. The time and the GPU are the pinned pair's, measured
  // on the card; a server or a model of the user's own is promised neither.
  const critic = criticStatus ?? { label: 'model', pinned: false, fetch: [], fetch_bytes: 0 };
  const fetch = !promptCritiqueReady && critic.fetch.length
    ? `${critic.fetch.join(' and ')} (about ${humanBytes(critic.fetch_bytes)})` : '';
  // A first use's download is minutes (114 s measured here, the bandwidth's
  // to say elsewhere), with nothing to show its progress: said, with the
  // clock beside the line counting.
  // Two halves, one shown at a time. What answers a press - the download,
  // the thinking - is a live region (#critic-progress), written only when
  // it changes, since the same words again are read out again. What a
  // press would cost, and the job that holds the card, is not
  // (#critic-cost): it changed at every job's start and end, read out about
  // a button nobody had pressed.
  const progress = !critiquing ? ''
    : fetch ? `Downloading ${fetch} once - this can take a few minutes - then thinking…`
    : critic.pinned ? 'Thinking… about 15 seconds on the GPU.' : `Thinking… (${critic.label})`;
  const cost = critiquing ? ''
    : serverOffline ? CRITIC_OFFLINE
    : modelState === 'loading' ? CRITIC_LOADING
    : busy ? 'An image job has the card; the critic waits for the queue to finish.'
    : elsewhereBusy ? "Another tab's image job has the card; the critic waits for it to finish."
    : fetch ? `First use downloads ${fetch} once.`
    : critic.pinned ? `Local ${critic.label} on the GPU, about 15 seconds; waits for image jobs.`
    : `Local ${critic.label}, as configured; waits for image jobs.`;
  if (els.criticProgress.textContent !== progress) els.criticProgress.textContent = progress;
  if (els.criticCost.textContent !== cost) els.criticCost.textContent = cost;
  syncCompareRun();
}

async function askCritic() {
  if (critiquing) return;
  // Held (criticHold): the press says why, each as its sibling buttons'
  // presses do. No answer is said in the offline words, which the poll's
  // first answer takes back, and the poll asks again at once.
  if (serverOffline) { showError(OFFLINE_MESSAGE); void refreshStatus(); return; }
  if (modelState === 'loading') { showError(CRITIC_LOADING, true); criticLoadNote = CRITIC_LOADING; return; }
  // A note, as Expand's and Unload's are, gone with that job (setBusy).
  if (busy) { showJobNote(els.criticAsk.title); return; }
  if (elsewhereBusy) {
    showError(CRITIC_ELSEWHERE, true);
    // Gone with that job (setElsewhereBusy).
    elsewhereNote = CRITIC_ELSEWHERE;
    return;
  }
  if (expandingPrompt) { showError('Wait for prompt expansion to finish.', true); return; }
  const payload = collect();
  if (!payload.prompt) { showError('Enter a prompt first.'); els.prompt.focus(); return; }
  const effectiveNegative = analysisNegative(payload);
  const epoch = ++critiqueEpoch;
  // Held from here, the check below included: a job or an Expand started
  // while it ran went ahead, and the critique behind it came back a 409.
  critiquing = true;
  critiqueFailure = null;
  critiqueOffline = false;
  showError('');
  analysisMessage('');
  startCriticClock();
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
          analysisMessage('The prompt check was replaced before the critic was asked, so nothing was sent. Ask again.');
        }
        return;
      }
    }
    critiqueResult = null;
    renderCritic();
    renderCompare();
    let body;
    try {
      body = await api('/api/prompt/critique', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      // The critique's own failure, not the check's before it: said in the
      // critic's block too, where after the wait the line had gone back to
      // its invitation as if nothing had been asked. No answer at all is
      // the server's, not the critic's, said in a sentence of its own that
      // the poll which answers again takes back (takeBackCriticOffline).
      if (epoch === critiqueEpoch) {
        critiqueOffline = !!err.offline;
        critiqueFailure = err.offline ? 'the server did not answer' : err.message;
      }
      throw err;
    }
    // An answer means the first use's download is done, whether or not it is
    // shown: the line under the button said it was still to come until the
    // next status poll.
    promptCritiqueReady = true;
    if (epoch !== critiqueEpoch) return;  // the panel was cleared meanwhile
    if (!analysed(payload, effectiveNegative)) {
      analysisMessage('The prompt was checked again while the critic ran, so its answer was set aside. Ask again.');
      return;
    }
    critiqueResult = { body };
    answered = true;
    // Shown, and dimmed with the rest of the panel, but never applied: its
    // rewrite would replace what was typed while it thought.
    if (!analysed(collect())) {
      analysisMessage('The prompt, the negative prompt or the references changed while the critic ran: its answer is about the earlier text, so nothing from it can be applied. Check again for the current one.');
    }
  } catch (err) {
    // The critique's own failure is said once: in the critic's block, which
    // takes focus as an answer does, and in #error only where that block is
    // not on show - the same sentence in both, each read out, was two problems.
    if (epoch !== critiqueEpoch) { /* the panel was cleared meanwhile */ }
    else if (!critiqueFailure) showError(err.message, false, { offline: !!err.offline });
    else if (els.promptAnalysis.hidden || els.analysisCritic.hidden) {
      showError(critiqueFailureText(critiqueFailure), false, { offline: critiqueOffline });
    }
  } finally {
    critiquing = false;
    stopCriticClock();
    syncPromptExpansion();
    if (epoch === critiqueEpoch) {
      renderCritic();
      renderCompare();
      // Nothing else announces the answer - the line under the button goes
      // back to the cost - so it takes focus, if focus is still waiting on
      // the button rather than on something typed meanwhile. A failure to
      // answer is an answer here.
      if ((answered || critiqueFailure) && document.activeElement === els.criticAsk) els.criticResult.focus();
    }
  }
}

// The seconds since Ask the critic, beside the line under it: a first use
// downloads for minutes and an answer takes seconds, and with nothing on the
// page moving, slow could not be told from stuck while every generation
// button was held. A measurement, not a promise, so it counts for a critic of
// the user's own too. aria-hidden, outside the line's live region.
function startCriticClock() {
  const start = performance.now();
  clearInterval(criticClockTimer);
  criticClockTimer = setInterval(() => {
    els.criticClock.textContent = `· ${Math.round((performance.now() - start) / 1000)} s`;
    els.criticClock.hidden = false;
  }, 1000);
}

function stopCriticClock() {
  clearInterval(criticClockTimer);
  els.criticClock.hidden = true;
  els.criticClock.textContent = '';
}

// A failed critique in the server's words, which say what to do where
// anything can be done - shorten the prompt, install a library, wait for the
// load - so nothing generic is added: "Ask again" contradicted a refusal
// that asking again only repeats. Said whose failure it is, in #error too
// where it goes there (the critic's block not on show): the bare reason sat
// under Generate with nothing naming the critic.
// Any closing stop, ? or ! of the reason's own gives way to the sentence's:
// "…still running?." came of keeping the question mark.
const critiqueFailureText = (reason) => `The critic did not answer: ${reason.replace(/[.?!\s]+$/, '')}.`;

// A critique that got no answer from the server: said as the server's, and
// taken back by the status poll that answers again (status.js), as Expand's
// and every offline message are - it stayed in the critic's block for good.
let critiqueOffline = false;
function takeBackCriticOffline() {
  if (!critiqueOffline) return;
  critiqueOffline = false;
  if (!critiqueFailure) return;
  critiqueFailure = null;
  renderCritic();
}

// What a press under the load said, until the load is over (status.js).
let criticLoadNote = null;
function takeBackCriticLoadNote() {
  if (criticLoadNote !== null && els.error.textContent === criticLoadNote) showError('');
  criticLoadNote = null;
}

// The rules whose finding on the prompt rests on the tag index's categories:
// the order's slots, the style a known name reads as, and which names are
// artists (`prompt_rules._place`, `_detect_style`, `_artist_prefix`). The
// prompt's check asks the index for a tag or hybrid prompt on a Danbooru
// checkpoint, and the artist rule fires only with its answer; the rewrite's
// check never asks (`main._lint`). Such a finding gone from the rewrite may
// be one its check could not see, so it is not called fixed - except the
// artist rule's where the server says the rewrite's artist check, with the
// analysis's own answer, read every artist the rewrite may name
// (`rewrite_artists_checked`): a rewrite that adds the @ has then fixed it,
// while one naming an artist the analysis never asked about, or keeping the
// name in a clause or a sentence, keeps the hedge.
const TAG_INDEX_RULES = new Set(['guide.order', 'guide.style', 'guide.artist_prefix']);
const readsTagIndex = (finding) => TAG_INDEX_RULES.has(finding.rule)
  && (finding.rule === 'guide.artist_prefix'
    ? !critiqueResult?.body?.rewrite_artists_checked
    : tagVocabularyEnabled && ['tags', 'hybrid'].includes(analysisResult.body.style));

function renderCritic() {
  syncCritic();
  const box = els.criticResult;
  box.replaceChildren();
  if (!critiqueResult) {
    // The server's reason, without the exception's class name; here alone
    // while the block is on show (askCritic).
    if (critiqueFailure) box.appendChild(note(critiqueFailureText(critiqueFailure), 'analysis-note critic-failure'));
    return;
  }
  const { body } = critiqueResult;

  const parts = PART_LABELS.filter(([key]) => body.parts?.[key]);
  if (parts.length) {
    // Captioned: unlabelled, the list read as part of the issue under it.
    const read = document.createElement('div');
    read.className = 'critic-read';
    const list = document.createElement('dl');
    list.className = 'critic-parts';
    for (const [key, label] of parts) {
      const term = document.createElement('dt');
      term.textContent = label;
      const value = document.createElement('dd');
      value.textContent = body.parts[key];
      list.append(term, value);
    }
    read.append(note('How the critic read it', 'analysis-note sub'), list);
    box.appendChild(read);
  }

  if (body.issues.length) {
    const list = document.createElement('ul');
    list.className = 'analysis-list';
    for (const issue of body.issues) list.appendChild(issueItem(issue));
    box.appendChild(list);
  } else {
    box.appendChild(note('No issue against the sources.'));
  }

  // The two versions, side by side where the panel is wide enough for both.
  const versions = document.createElement('div');
  versions.className = 'critic-variants';
  if (body.rewrite) {
    const block = document.createElement('div');
    block.className = 'critic-variant';
    const area = variantArea(body.rewrite, "The critic's rewrite", 'rewrite');
    const head = document.createElement('div');
    head.className = 'critic-variant-head';
    head.append(span('critic-variant-name', 'Rewrite'));
    if (body.rewrite !== analysisResult.source) {
      head.append(writeButton('Use rewrite', `Replace the prompt with the rewrite, then check it.${KEEPS_PROMPT}`,
        () => useVersion(body.rewrite, 'Use rewrite')));
    }
    // Whether the rewrite fixes what the prompt was flagged for: each of its
    // findings says whether the prompt has one under the same rule - which
    // need not be the same words (on Animagine the order finding names other
    // tags) - and the prompt's findings it no longer has follow. A finding
    // is matched by its rule and its field, never collapsed by them.
    const key = (f) => `${f.field} ${f.rule}`;
    const before = new Set(analysisResult.body.findings.map(key));
    const after = new Set(body.rewrite_findings.map(key));
    const checked = document.createElement('div');
    checked.className = 'critic-checked';
    renderFindings(checked, body.rewrite_findings, {
      // The rewrite's words are in the rewrite; a negative prompt finding is
      // about the box it came from.
      select: (f) => (f.field === 'negative_prompt'
        ? selectAnalysed(f.field, f.start, f.end) : selectIn(area, body.rewrite, f.start, f.end)),
      apply: null,
      empty: 'The check finds nothing to flag in the rewrite.',
      carried: (f) => (before.has(key(f)) ? 'also in your prompt' : 'new'),
      where: (f) => (f.field === 'negative_prompt' ? 'the negative prompt' : 'the rewrite'),
    });
    const gone = analysisResult.body.findings.filter((f) => !after.has(key(f)));
    const listed = (findings) => findings.map((f) => excerpt(f.message, 60)).join('; ');
    const fixed = gone.filter((f) => !readsTagIndex(f));
    const unseen = gone.filter(readsTagIndex);
    if (fixed.length) checked.appendChild(note(`No longer flagged: ${listed(fixed)}`, 'analysis-note sub'));
    if (unseen.length) {
      checked.appendChild(note('Not flagged on the rewrite, whose check asks the tag index nothing of its own - '
        + `these may still apply: ${listed(unseen)}`, 'analysis-note sub'));
    }
    block.append(head, rewriteChanges(analysisResult.source, body.rewrite), area,
      note('The same check, run on the rewrite:', 'analysis-note sub'), checked);
    versions.appendChild(block);
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
    head.append(span('critic-variant-name', 'Reordered'), writeButton('Use reordered', `${title}.${KEEPS_PROMPT}`,
      () => useVersion(body.reordered, 'Use reordered')));
    block.append(head, variantArea(body.reordered,
      kind === 'check' ? "The critic's reordering, Check's own order" : "The critic's reordering", 'reordered'));
    versions.appendChild(block);
  }
  if (versions.children.length) box.appendChild(versions);

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
  // The diff, then the gist of it after the <details> - a closed one shows
  // nothing but its summary - so what changed reads before any click.
  const shown = document.createDocumentFragment();
  shown.append(details, changeGist(changes));
  return shown;
}

// The first few changes, a few words of each - "Changed red → blue · added
// film grain · moved masterpiece" - marked as the diff marks them. Opened by
// default, the diff would print the whole rewrite a second time above its
// own box.
function changeGist(changes) {
  const items = gistItems(changeHunks(changes));
  // Not `sub`, the fields' flex column, which would stack its words.
  const line = note('', 'analysis-note critic-gist');
  const shown = items.slice(0, 3);
  const each = Math.max(12, Math.floor(96 / shown.length));
  // Unless the separators are all there is.
  const words = (tag, text, max) => {
    const el = document.createElement(tag);
    el.textContent = excerpt(gistCore(text) || text, max);
    return el;
  };
  shown.forEach((item, i) => {
    // The verb once for a run of the same kind: "Added ginger · curled".
    if (!i) line.append(`${item.verb[0].toUpperCase()}${item.verb.slice(1)} `);
    else line.append(item.verb === shown[i - 1].verb ? ' · ' : ` · ${item.verb} `);
    // Moved words are in both texts: neither struck through nor underlined.
    if (item.verb === 'moved') line.append(span('critic-moved', excerpt(item.text, each)));
    else if (!item.removed && !item.added) line.append('spacing');
    else if (!item.added) line.append(words('del', item.removed, each));
    else if (!item.removed) line.append(words('ins', item.added, each));
    else {
      const half = Math.max(12, Math.floor(each / 2));
      line.append(words('del', item.removed, half), ' → ', words('ins', item.added, half));
    }
  });
  if (items.length > 3) line.append(' · …');
  return line;
}

// The diff's runs as changes a reader would name. A real rewrite changes
// many words, and run by run the gist spent its three places on single ones:
// a word replaced was a removal and an addition, and two edits a space or a
// comma kept apart were two. So the runs are joined across what keeps them
// apart when that is only spaces and marks - kept on both sides of the
// change, since both texts have it - and each change holds what it took out
// and what it put in.
const GIST_GLUE = /^[\s\p{P}]*$/u;
// A change's words without the separators at their ends: "film grain" for
// ", film grain", "masterpiece" whichever side of it the comma moved with.
const gistCore = (text) => text.replace(/^[\s,;，、；]+|[\s,;，、；]+$/gu, '');

function changeHunks(changes) {
  const hunks = [];
  let hunk = null, glue = '';
  for (const { kind, text } of changes) {
    if (kind === 'same') {
      // Joins only what comes next; at the end it joins nothing.
      if (hunk && GIST_GLUE.test(text)) glue = text;
      else { hunk = null; glue = ''; }
      continue;
    }
    if (!hunk) hunks.push(hunk = { removed: '', added: '' });
    hunk.removed += glue;
    hunk.added += glue;
    glue = '';
    hunk[kind] += text;
  }
  return hunks.map(({ removed, added }) => ({ removed: removed.trim(), added: added.trim() }));
}

// Each change with its verb, words taken out in one place and put back in
// another being one move: on a tag checkpoint a rewrite mostly moves tags,
// and "removed masterpiece, · added , masterpiece" read as two edits.
function gistItems(hunks) {
  const items = [];
  const paired = new Set();
  hunks.forEach((hunk, i) => {
    if (paired.has(i)) return;
    const [from, to] = hunk.removed ? ['removed', 'added'] : ['added', 'removed'];
    const words = gistCore(hunk[from]);
    const other = words && !hunk[to]
      ? hunks.findIndex((h, j) => j > i && !paired.has(j) && !h[from] && gistCore(h[to]) === words) : -1;
    if (other >= 0) {
      paired.add(other);
      items.push({ verb: 'moved', text: words });
      return;
    }
    // Spaces alone, both sides empty once trimmed, are a change of spacing.
    const verb = hunk.removed && !hunk.added ? 'removed' : hunk.added && !hunk.removed ? 'added' : 'changed';
    items.push({ ...hunk, verb });
  });
  return items;
}

function variantArea(text, label, version) {
  const area = document.createElement('textarea');
  area.className = 'analysis-text';
  area.readOnly = true;
  area.rows = 2;
  area.value = text;
  area.setAttribute('aria-label', label);
  // The Compare row it stands for, outlined from there (`linkVersion`).
  area.dataset.version = version;
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
  const text = span('analysis-message');
  // The problem and its quote: what Show selects, and its description.
  const said = span('analysis-said');
  said.append(issue.problem);
  if (issue.quote) said.append(' ', span('analysis-quoted', `"${issue.quote}"`));
  text.append(said);
  if (issue.start != null && issue.end != null) {
    text.append(makeSelectable(li, () => selectAnalysed('prompt', issue.start, issue.end), said, 'the prompt'));
  }
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
  return li;
}
