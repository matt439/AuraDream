// Local Diffusion's page script: Mass refine, a refine pass of every image the gallery selection holds.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// "Refine…" in the gallery's footer row: a refine pass of each image the
// selection holds (gallery-select.js), else what the filter matches - each
// from its own pixels under its own prompt, negative prompt, seed, CFG, mu
// and LoRA, at one strength and one scale. The server decides, per image,
// what it would do (`refine_batch.py`, `api_refine_batch.refine_batch_plan`): queued, a
// refine pass skipped, another checkpoint's left out unless asked for, too
// large at this scale, no step left at this strength, its file or its LoRA
// gone, or the pass already in the gallery. The page lists the answer and
// adds nothing of its own. Refine is held, with its reason, until the answer
// is the answer to what the dialog says; the queueing request pins the ids
// that answer resolved (`expected`). One job per distinct size, as Swap tags.
//
// The single Refine dialog (refine.js) is one image's, with its prompt, size
// and versions; this one has none of those - the images differ in all of
// them - and shares only what is the checkpoint's: the remembered scale and
// strength (REFINE_KEY), the scale buttons' factors and the opening cap.

const RB_CHECK_DELAY = 250;
const RB_ONE_STRENGTH = "One strength for a batch. A sweep is one image's Refine.";
const RB_STEPS_PROBLEM = "Steps is a whole number from 1 to 100, or blank for each image's own";
// In the count line, in this order whatever the server's.
const RB_STATUS_WORDS = {
  pass: 'refine passes skipped', foreign: 'from other checkpoints', too_big: 'too large at this scale',
  no_step: 'left no step at this strength', file_missing: 'with their file missing',
  lora_missing: 'with their LoRA missing', rendered: 'already have this pass', deleted: 'deleted',
  invalid: 'invalid',
};
const RB_CHIP = {
  queued: 'queued', pass: 'refine pass', foreign: 'other checkpoint', too_big: 'too large',
  no_step: 'no step', file_missing: 'file missing', lora_missing: 'LoRA missing',
  rendered: 'already refined', deleted: 'deleted', invalid: 'invalid',
};

// What the dialog acts on, fixed when it opens: `{ids}` or the filter, and
// the galleryActionTarget it came from, whose count the scope line says.
let rbScope = null;
let rbTarget = null;
// The gallery's count (`total`) the last check was asked under: a filter's
// matches grow as a job lands images, and the check is asked again
// (refineBatchGalleryChanged) so the press pins what the filter holds now.
let rbSeenTotal = null;
let rbRevision = null;
// The scale the passes run at: one of REFINE_SCALES. `rbOpened` is the one
// the dialog opened at, and `rbCapped` why that is not the one asked for
// ({asked, remembered}), or null: the largest image would have gone past
// the guard or the tight line.
let rbScale = REFINE_SCALE;
let rbOpened = REFINE_SCALE;
let rbCapped = null;
// The last check's answer and the payload it answered (as a string).
let rbAnswer = null;
let rbAnswerKey = '';
// What the last check was refused with (a strength past 1), or ''.
let rbProblem = '';
// A Refine press the server refused: said until the next edit or press.
let rbRefused = false;
let rbAsking = null;
let rbTimer = null;
// The POST is out: the dialog closes on its answer, and Escape and Cancel
// wait for it, as the single dialog's do.
let rbSending = false;
// An open waiting on its first check: a second press does nothing.
let rbOpening = null;

// Why the button is held, or the target it acts on (galleryActionTarget).
function refineBatchTarget() {
  if (binMode) return { why: BIN_HELD };
  if (!refineSupported) {
    return {
      why: lastDefaults?.refine_settled ? REFINE_NONE
        : 'Refine is unavailable: as far as the server can tell, this checkpoint has no compatible '
          + 'image-to-image pipeline. A load answers for certain.',
    };
  }
  return galleryActionTarget('to choose the images to refine', 'one batch refine');
}

// The button's state: held with the reason, or titled with what it takes.
// From syncRerender (every filter and count change), every change of the
// selection, and syncCardTools (the checkpoint's refine pass came or went).
function syncRefineBatchButton() {
  const target = refineBatchTarget();
  els.refineBatch.setAttribute('aria-disabled', String(!target.scope));
  els.refineBatch.title = target.scope
    ? `Refine ${target.words} at one strength and scale - each keeps its own prompt, seed, CFG and mu`
    : target.why;
  refineBatchGalleryChanged();
  syncRefineBatchRun();
}

// The gallery's count changed under an open dialog acting on the filter: a
// job landed images the filter matches (or a delete elsewhere). The modal
// dialog holds the filter the gallery shows, so `total` is its count; the
// check is asked again so the answer, and the ids a press pins, are the
// filter's now - not a refusal at every press while a job keeps landing.
function refineBatchGalleryChanged() {
  if (!els.refineBatchDialog.open || !rbScope || rbScope.ids || rbSending) return;
  if (total === rbSeenTotal) return;
  rbSeenTotal = total;
  // Not the answer to what the filter holds now: a press waits for the next.
  rbAnswerKey = '';
  scheduleRefineBatchCheck(0);
}

// The scope line, from the target the dialog opened on and, for a filter,
// the count of the latest answer.
function rbSayScope(answer) {
  if (!rbTarget) return;
  if (!rbScope.ids && answer && typeof answer.total === 'number') rbTarget = { ...rbTarget, n: answer.total };
  const line = galleryActionScopeLine(rbTarget, 'Refining');
  if (els.refineBatchScope.textContent !== line) els.refineBatchScope.textContent = line;
}

function rbStrengthText() { return els.refineBatchStrength.value.trim(); }

// A field the page can refuse without asking: a list or a range in
// Strength, Steps not a whole number in range.
function rbFieldProblem() {
  const strength = rbStrengthText();
  if (strength !== '' && !SINGLE_STRENGTH.test(strength)) {
    return /[\d.]\s*[,;\s-]\s*[\d.]|x/i.test(strength) ? RB_ONE_STRENGTH : `"${strength}" is not a strength: a number between 0 and 1`;
  }
  const steps = els.refineBatchSteps.value.trim();
  if (steps !== '' && !(/^\d+$/.test(steps) && +steps >= 1 && +steps <= 100)) return RB_STEPS_PROBLEM;
  return '';
}

function rbPayload() {
  const strength = rbStrengthText();
  const steps = els.refineBatchSteps.value.trim();
  return {
    ...rbScope,
    scale: rbScale,
    strength: strength === '' ? null : +strength,
    steps: steps === '' ? null : +steps,
    skip_passes: els.refineBatchSkip.checked,
    include_foreign: els.refineBatchForeign.checked,
  };
}

async function askRefineBatch(payload, signal) {
  return api('/api/refine/batch/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
}

// Opens on the first check's answer, not before: the scale it opens at is
// capped by the largest image the selection holds - the server's to say, the
// gallery's page may not hold it - as the single dialog caps by its one
// source (openingRefineSize). Strength opens at what was last run on this
// checkpoint, else its own; Steps blank; refine passes skipped.
async function openRefineBatch() {
  if (rbOpening || els.refineBatchDialog.open) return;
  const target = refineBatchTarget();
  if (!target.scope) { showError(target.why, true, { slot: els.galleryError }); return; }
  rbScope = target.scope;
  rbTarget = target;
  rbSeenTotal = total;
  rbRevision = modelRevision;
  const prefs = refinePrefs();
  const kept = prefs.strength?.[loadedModel];
  els.refineBatchStrength.value = typeof kept === 'string' && SINGLE_STRENGTH.test(kept)
    ? kept : String(refineDefaultStrength());
  els.refineBatchSteps.value = '';
  els.refineBatchSkip.checked = true;
  els.refineBatchForeign.checked = false;
  const asked = REFINE_SCALES.includes(+prefs.scale) ? +prefs.scale : REFINE_SCALE;
  rbScale = asked;
  rbCapped = null;
  const opening = {};
  rbOpening = opening;
  els.refineBatch.setAttribute('aria-busy', 'true');
  showError('', false, { slot: els.galleryError });
  let answer;
  try {
    answer = await askRefineBatch(rbPayload());
    const largest = answer.largest && { width: answer.largest[0], height: answer.largest[1] };
    if (largest) {
      const { factor } = openingRefineSize(largest, asked);
      if (factor !== asked) {
        rbCapped = { asked, remembered: REFINE_SCALES.includes(+prefs.scale) };
        rbScale = factor;
        answer = await askRefineBatch(rbPayload());
      }
    }
  } catch (err) {
    if (rbOpening === opening) showError(err.message, true, { slot: els.galleryError });
    return;
  } finally {
    if (rbOpening === opening) {
      rbOpening = null;
      els.refineBatch.removeAttribute('aria-busy');
    }
  }
  if (rbRevision !== modelRevision) {
    showError('The checkpoint changed while Refine was checking the images; press Refine… again.', true, { slot: els.galleryError });
    return;
  }
  rbOpened = rbScale;
  rbAnswer = answer;
  rbAnswerKey = JSON.stringify(rbPayload());
  rbProblem = '';
  rbRefused = false;
  showError('', false, { slot: els.refineBatchError });
  renderRefineBatch(answer);
  els.refineBatchDialog.showModal();
  syncRefineBatchRun();
  els.refineBatchStrength.focus();
  els.refineBatchStrength.select();
}

// The four factors, each titled with what it makes of the largest image,
// the chosen one pressed.
function syncRefineBatchScales(largest) {
  if (!els.refineBatchScales.children.length) {
    for (const scale of REFINE_SCALES) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `${scale}×`;
      button.onclick = () => chooseRefineBatchScale(scale);
      els.refineBatchScales.append(button);
    }
  }
  const run = typeof lastDefaults?.refine_tight_pixels === 'number' ? lastHints?.refine_tight : null;
  [...els.refineBatchScales.children].forEach((button, i) => {
    const scale = REFINE_SCALES[i];
    const chosen = scale === rbScale;
    button.classList.toggle('active', chosen);
    button.setAttribute('aria-pressed', String(chosen));
    const capped = rbCapped && scale === rbCapped.asked
      ? (rbCapped.remembered ? ` (last run at ${scale}×)` : ' (the usual opening size)') : '';
    if (!largest) {
      button.title = `${scale} times each image's size${capped}`;
    } else {
      const size = refineScaled(largest[0], largest[1], scale);
      const fits = refineFits(size);
      const why = !fits ? ' - past what a refine pass renders; the images that would go past it are left out'
        : refineTight(...size) ? ` - may run tight on 16GB VRAM${run ? `, ${run}` : ''}` : '';
      button.title = `${scale}×: the largest image, ${largest[0]}×${largest[1]}, becomes ${size[0]}×${size[1]}${why}${capped}`;
    }
    button.setAttribute('aria-label', `${scale} times each image's size. ${button.title}`);
  });
}

function chooseRefineBatchScale(scale) {
  if (rbSending || scale === rbScale) return;
  rbScale = scale;
  syncRefineBatchScales(rbAnswer?.largest ?? null);
  refineBatchEdited(0);
}

// An edit: a refusal said for the press before it is no longer the news.
function refineBatchEdited(delay = RB_CHECK_DELAY) {
  els.refineBatchStrength.removeAttribute('aria-invalid');
  els.refineBatchSteps.removeAttribute('aria-invalid');
  if (rbRefused || els.refineBatchError.textContent) {
    rbRefused = false;
    showError('', false, { slot: els.refineBatchError });
  }
  scheduleRefineBatchCheck(delay);
}

function scheduleRefineBatchCheck(delay = RB_CHECK_DELAY) {
  clearTimeout(rbTimer);
  rbTimer = setTimeout(() => { void runRefineBatchCheck(); }, delay);
  syncRefineBatchRun();
}

async function runRefineBatchCheck() {
  if (!els.refineBatchDialog.open || !rbScope) return;
  const payload = rbPayload();
  const key = JSON.stringify(payload);
  rbAsking?.abort();
  rbAsking = null;
  const field = rbFieldProblem();
  if (field) {
    // Said without asking; the list stays as the last answer left it.
    rbAnswer = null;
    rbAnswerKey = key;
    rbProblem = field;
    rbSay(field, true);
    syncRefineBatchRun();
    return;
  }
  const asking = new AbortController();
  rbAsking = asking;
  syncRefineBatchRun();
  try {
    const answer = await askRefineBatch(payload, asking.signal);
    if (rbAsking !== asking) return;
    rbAnswer = answer;
    rbProblem = '';
    renderRefineBatch(answer);
  } catch (err) {
    if (rbAsking !== asking || err.name === 'AbortError') return;
    rbAnswer = null;
    rbProblem = err.message;
    rbSay(err.message, true);
  }
  rbAnswerKey = key;
  rbAsking = null;
  syncRefineBatchRun();
}

// The count line: two lines kept, whatever it says, its whole text the title.
function rbSay(text, failed = false) {
  if (els.refineBatchCount.textContent !== text) els.refineBatchCount.textContent = text;
  els.refineBatchCount.title = text;
  els.refineBatchCount.classList.toggle('failed', failed);
}

function renderRefineBatch(answer) {
  rbSayScope(answer);
  els.refineBatchTitle.textContent = answer.queued ? `Refine ${plural(answer.queued, 'image')}` : 'Refine';
  syncRefineBatchScales(answer.largest);
  // Offered only where the selection holds another checkpoint's images, and
  // kept while ticked.
  els.refineBatchForeignLabel.hidden = !answer.foreign && !els.refineBatchForeign.checked;
  els.refineBatchForeignLabel.title = answer.foreign
    ? `Refine ${plural(answer.foreign, 'image')} made on ${answer.models.join(', ')} too, at ${checkpointName(loadedModel)}'s steps, CFG and mu and without their LoRA`
    : '';
  els.refineBatchSkip.parentElement.title = answer.passes
    ? `${plural(answer.passes, 'image')} in the selection ${answer.passes === 1 ? 'is a refine pass' : 'are refine passes'}; untick to refine ${answer.passes === 1 ? 'it' : 'them'} again`
    : 'Leave out images that are themselves refine passes';
  const low = answer.start_sigma_low;
  const high = answer.start_sigma_high;
  const noise = typeof low === 'number' && typeof high === 'number'
    ? `starts at ${Math.round(low * 100) === Math.round(high * 100) ? `${Math.round(low * 100)}%`
      : `${Math.round(low * 100)}–${Math.round(high * 100)}%`} noise` : '';
  const time = answer.seconds ? `about ${humanTime(answer.seconds)}`
    : answer.unpriced_sizes ? `no estimate: ${plural(answer.unpriced_sizes, 'size')} ${answer.unpriced_sizes === 1 ? 'has' : 'have'} no refine history`
      : '';
  const parts = [`${answer.queued} will be refined`, `strength ${answer.strength}`, noise, time];
  for (const [status, words] of Object.entries(RB_STATUS_WORDS)) {
    if (answer.counts[status]) parts.push(`${answer.counts[status]} ${words}`);
  }
  if (answer.jobs > 1) parts.push(`${answer.jobs} jobs, one per size`);
  if (answer.total > answer.images.length) parts.push(`the first ${answer.images.length} of ${answer.total} listed`);
  rbSay(parts.filter(Boolean).join(' · '));
  els.refineBatchResults.replaceChildren(...answer.images.map(refineBatchRow));
}

function refineBatchRow(image) {
  const li = document.createElement('li');
  li.className = 'rb-result';
  li.dataset.id = image.id;
  li.dataset.status = image.status;
  if (image.thumb_url) {
    const img = document.createElement('img');
    img.src = image.thumb_url;
    img.alt = '';
    img.loading = 'lazy';
    li.append(img);
  } else {
    li.append(Object.assign(document.createElement('span'), { className: 'rb-thumb-gone' }));
  }
  const body = document.createElement('div');
  body.className = 'rb-body';
  const head = document.createElement('p');
  head.className = 'rb-line';
  const chip = Object.assign(document.createElement('span'), { className: 'rb-chip', textContent: RB_CHIP[image.status] ?? image.status });
  head.append(`#${image.id} `, chip);
  if (image.width && image.height) {
    head.append(` ${image.width}×${image.height}`);
    if (image.target_width && image.status !== 'pass' && image.status !== 'foreign') {
      head.append(` → ${image.target_width}×${image.target_height}`);
    }
  }
  body.append(head);
  if (image.detail && image.status !== 'queued') {
    body.append(Object.assign(document.createElement('p'), { className: 'rb-detail', textContent: image.detail }));
  }
  if (image.notes?.length) {
    body.append(Object.assign(document.createElement('p'), { className: 'rb-notes', textContent: image.notes.join(' · ') }));
  }
  li.append(body);
  return li;
}

// Why Refine is held now, or '' when it may go.
function refineBatchHold() {
  const held = submitHold();
  if (held) return held;
  if (!refineSupported) return 'This checkpoint has no refine pass';
  const field = rbFieldProblem();
  if (field) return field;
  if (rbAsking || JSON.stringify(rbPayload()) !== rbAnswerKey) return 'Checking the images…';
  if (rbProblem) return rbProblem;
  if (rbRevision !== modelRevision) return 'The checkpoint changed; close Refine and open it again';
  if (!rbAnswer?.queued) {
    const counts = rbAnswer?.counts ?? {};
    return counts.rendered && Object.keys(counts).length === 1
      ? 'Every image already has this pass at this strength and scale'
      : 'No image can be refined at these settings; each one says why';
  }
  return '';
}

function syncRefineBatchRun() {
  if (!els.refineBatchDialog.open) return;
  if (rbSending) {
    els.refineBatchRun.textContent = 'Sending…';
    els.refineBatchRun.setAttribute('aria-disabled', 'true');
    els.refineBatchRun.title = 'Waiting for the server to take the batch';
    return;
  }
  const hold = refineBatchHold();
  const queue = busy ? ' (queue)' : '';
  els.refineBatchRun.textContent = !hold && rbAnswer
    ? `Refine ${plural(rbAnswer.queued, 'image')}${rbAnswer.jobs > 1 ? ` (${rbAnswer.jobs} jobs)` : ''}${queue}`
    : `Refine${queue}`;
  els.refineBatchRun.setAttribute('aria-disabled', String(Boolean(hold)));
  els.refineBatchRun.title = hold
    || `Refine ${plural(rbAnswer.queued, 'image')} at strength ${rbAnswer.strength} and ${rbScale}× (Ctrl+Enter)`;
  if (!hold && !rbRefused && els.refineBatchError.textContent) showError('', false, { slot: els.refineBatchError });
}

// Refine: waits for the check of what the dialog says, as the single
// dialog's Refine does, then queues. Held, the press says why in the error
// line and points at the field it is about.
async function runRefineBatch() {
  if (!els.refineBatchDialog.open || rbSending || submitting) return;
  rbRefused = false;
  while (!rbFieldProblem() && (rbAsking || JSON.stringify(rbPayload()) !== rbAnswerKey)) {
    clearTimeout(rbTimer);
    await runRefineBatchCheck();
    if (!els.refineBatchDialog.open || rbSending || submitting) return;
  }
  const hold = refineBatchHold();
  if (hold) {
    showError(hold, false, { slot: els.refineBatchError });
    const field = /strength/i.test(hold) ? els.refineBatchStrength : /^Steps/.test(hold) ? els.refineBatchSteps : null;
    if (field) { field.setAttribute('aria-invalid', 'true'); field.focus(); }
    if (serverOffline) void refreshStatus();
    return;
  }
  const payload = { ...rbPayload(), expected: rbAnswer.ids };
  rbSending = true;
  syncRefineBatchRun();
  let accepted = false;
  await startJob('/api/refine/batch', payload, {
    slot: els.refineBatchError,
    panel: false,
    onAccepted: () => {
      accepted = true;
      rbSending = false;
      rememberRefineBatch();
      els.refineBatchDialog.close();
      // Focus goes back to Refine… in the footer, which opened the dialog.
    },
    // Refused - the selection changed, the gallery holds every pass now: the
    // dialog stays open with the reason, and checks again.
    onRefused: () => {
      rbSending = false;
      rbRefused = Boolean(els.refineBatchError.textContent);
      rbAnswer = null;
      rbAnswerKey = '';
      scheduleRefineBatchCheck(0);
    },
  });
  if (!accepted) {
    rbSending = false;
    syncRefineBatchRun();
  }
}

// The scale and strength, in the single dialog's REFINE_KEY: one place for
// what was last run on this checkpoint. Only a scale the user chose - where
// the dialog opened below the one asked for, that one run as it opened is
// the dialog's choice, as rememberRefine has it.
function rememberRefineBatch() {
  const prefs = refinePrefs();
  if (!rbCapped || rbScale !== rbOpened) prefs.scale = rbScale;
  const strengths = prefs.strength && typeof prefs.strength === 'object' ? prefs.strength : {};
  const text = rbStrengthText();
  if (loadedModel && SINGLE_STRENGTH.test(text)) {
    if (+text === refineDefaultStrength()) delete strengths[loadedModel];
    else strengths[loadedModel] = text;
  }
  prefs.strength = strengths;
  remember(REFINE_KEY, JSON.stringify(prefs));
}

// Cancel, Escape and the backdrop, except while the POST is out.
function closeRefineBatch() {
  if (!rbSending) els.refineBatchDialog.close();
}

// From forgetImages, the one place the page drops deleted ids: an open
// dialog that listed one checks again, and shows it as deleted.
function forgetRefineBatchImages(gone) {
  if (!els.refineBatchDialog.open || !rbAnswer) return;
  if (rbAnswer.ids.some((id) => gone.has(id))) scheduleRefineBatchCheck(0);
}

// Keys inside the dialog: Enter in a box is Refine's press, as in any form;
// ArrowUp and ArrowDown step Strength by 0.05, 0.01 with Shift, as the
// single dialog's do. Ctrl+Enter is the document's (app.js).
function refineBatchKeydown(event) {
  if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === 'Enter' && !event.shiftKey && event.target.matches('input:not([type="checkbox"])')) {
    event.preventDefault();
    void runRefineBatch();
    return;
  }
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && event.target === els.refineBatchStrength) {
    const text = rbStrengthText();
    if (!SINGLE_STRENGTH.test(text)) return;
    event.preventDefault();
    const step = (event.shiftKey ? 0.01 : 0.05) * (event.key === 'ArrowUp' ? 1 : -1);
    els.refineBatchStrength.value = String(Math.min(1, Math.max(0.01, +(+text + step).toFixed(2))));
    refineBatchEdited();
  }
}

function refineBatchBackdrop(event) {
  if (event.target !== els.refineBatchDialog) return;
  const box = els.refineBatchDialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right
    && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) closeRefineBatch();
}

function setupRefineBatch() {
  els.refineBatch.onclick = () => { void openRefineBatch(); };
  els.refineBatchCancel.onclick = closeRefineBatch;
  els.refineBatchRun.onclick = () => { void runRefineBatch(); };
  els.refineBatchDialog.addEventListener('click', refineBatchBackdrop);
  els.refineBatchDialog.addEventListener('keydown', refineBatchKeydown);
  els.refineBatchDialog.addEventListener('cancel', (e) => { if (rbSending) e.preventDefault(); });
  els.refineBatchStrength.addEventListener('input', () => refineBatchEdited());
  els.refineBatchSteps.addEventListener('input', () => refineBatchEdited());
  els.refineBatchSkip.addEventListener('change', () => refineBatchEdited(0));
  els.refineBatchForeign.addEventListener('change', () => refineBatchEdited(0));
  els.refineBatchDialog.addEventListener('close', () => {
    clearTimeout(rbTimer);
    rbAsking?.abort();
    rbAsking = null;
    rbScope = null;
    rbTarget = null;
  });
  onSelectionChange(syncRefineBatchButton);
  syncRefineBatchButton();
}
