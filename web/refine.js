// Local Diffusion's page script: the Refine dialog, submission and source passes.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------ the refine pass

// A pass used to be a mode of the panel: the ⤢ button rewrote the prompt,
// seed, steps and size, and from then on Generate, Batch, Repeat and the grid
// all ran passes until the source chip was removed - and a pass inherited a
// Seed, Batch and Repeat meant for direct renders. The dialog keeps the two
// apart: it opens on one image with that image's own prompt and sampling, has
// its own size, strength, versions and seed, and submits itself. The panel's
// requests never carry a source (`collect()`).

// The largest pass the server takes: config's `refine_max_side` and
// `refine_max_pixels`, any 2 MP preset at 2x, read off the same table as the
// presets. tests/test_profiles.py holds these two lines to the server's.
const REFINE_MAX_SIDE = 2 * Math.max(...ASPECTS.flatMap((a) => SIZES[a][2]));
const REFINE_MAX_PIXELS = 4 * Math.max(...ASPECTS.map((a) => SIZES[a][2][0] * SIZES[a][2][1]));

// What the dialog remembers between opens, per browser: the scale it last
// ran at, the number of versions, and per checkpoint a single strength the
// user ran instead of the profile's - 0.4 means one thing on SDXL and another
// on a shifted flow schedule. A sweep is not remembered, or the next open
// would start one unasked. Reset forgets all of it with the panel.
const REFINE_KEY = 'local-diffusion.refine.v1';

// One number, or a sweep in the grid's syntax (`0.3,0.4,0.5`, `0.2-0.5x0.1`)
// that the server parses: one strength is an ordinary request, several a grid
// with a strength axis.
const SINGLE_STRENGTH = /^(\d+(\.\d*)?|\.\d+)$/;

// Two panes from here (style.css has the layout): the source as large as the
// dialog allows, so the picture is the PNG itself. Under it the source is a
// 72 px thumbnail, which the PNG would only cost a decode for.
const REFINE_WIDE = window.matchMedia('(min-width: 1600px)');
// The passes the dialog shows of its source, newest first; the heading counts
// them all.
const REFINE_PASSES = 8;

let refineTimer;
let refineEpoch = 0;
// What the last preview was refused with, or null: what Refine would be
// refused with, so the button says so before the click rather than after the
// dialog has closed.
let refineProblem = null;
// The request the last preview answered for, as JSON: Refine waits for the
// preview of what the boxes hold before it reads the answer (`runRefine`).
let refinePreviewed = null;
let refineWaiting = false;
// The scale the opening size passed over - the one last run, or the 1.5x a
// first open takes - with the factor and size it opened at instead: its
// button says so, one click away, and the size line says why while the size
// is the one it opened at (`refineCapNote`). A default passed over said
// nothing: a 1.5 MP Z-Anime source opened at 1x, and the dialog never said
// why it would not upscale. {scale, remembered, opened, size}, or null.
// `refineCapped`: the dialog opened below the scale it was asked for, so that
// size, run as it opened, is not the user's choice.
let refineSkipped = null;
let refineCapped = false;
// What openRefine filled the boxes with, and whether the dialog closed by
// running: a close that was neither keeps what was typed, in this page only,
// for the next open of the same image on the same checkpoint -
// {id, model, values}. Never in REFINE_KEY: a sweep left there would start on
// another image unasked.
let refineFilled = null;
let refineRan = false;
// The pass's POST is out (runRefine): the dialog is held until it answers.
let refineSending = false;
let unsentRefine = null;
// The dialog's own prompt check (`checkRefinePrompt`), its tag lookup and its
// list of passes, each with an epoch of its own: `refineEpoch` moves with
// every preview, which would drop their replies.
let refineCheckTimer;
let refineCheckEpoch = 0;
let refineChecked = null;  // { prompt, negative, negativeOff, body: PromptAnalysis }
let refineTagTimer;
let refineTagEpoch = 0;
let refinePassesEpoch = 0;

function refinePrefs() {
  try {
    const prefs = JSON.parse(remembered(REFINE_KEY) || 'null');
    return prefs && typeof prefs === 'object' ? prefs : {};
  } catch { return {}; }
}

// A source's size times a factor, each side on the 16-pixel grid, half
// rounding up: backend/refine.py's `scaled`, so this dialog and the CLI's
// --scale land on the same size for the same image.
function refineScaled(width, height, factor) {
  const snap = (side) => Math.max(16, Math.floor(side * factor / 16 + 0.5) * 16);
  return [snap(width), snap(height)];
}

const refineFits = ([w, h]) =>
  w >= 256 && h >= 256 && w <= REFINE_MAX_SIDE && h <= REFINE_MAX_SIDE && w * h <= REFINE_MAX_PIXELS;

const refineDefaultStrength = () => lastDefaults?.refine_strength ?? REFINE_STRENGTH;

// Where a pass may run tight on the card: the profile's measured line where
// docs/verified.md has one (`defaults.refine_tight_pixels` - FLUX.1 fills
// the card at 2.4 MP, WAI ran 8.6), else the panel's. Past it the size line
// warns, naming the run it quotes (`hints.refine_tight`), and the dialog does
// not open there.
function refineTight(w, h) {
  const measured = lastDefaults?.refine_tight_pixels;
  return typeof measured === 'number' ? w * h > measured : sizeIsTight(w, h);
}

// The size the dialog opens at: the source's times `scale`, or the largest
// smaller factor the pass takes that is not past the tight line, else the
// source's own. Past the guard Refine would only refuse; past the line a
// 1.5 MP FLUX.1 source at 1.5x is 3.5 MP, where the card spills and the
// checkpoint's own strength, its shift grown with the size, repaints the
// subject. The buttons still go there in one click.
function openingRefineSize(rec, scale) {
  for (const factor of [scale, ...REFINE_SCALES.filter((s) => s < scale).reverse()]) {
    const size = refineScaled(rec.width, rec.height, factor);
    if (refineFits(size) && !refineTight(...size)) return { size, factor };
  }
  return { size: refineScaled(rec.width, rec.height, 1), factor: 1 };
}

// Compare 3: half the checkpoint's strength, it, and half as much again, on
// the 0.05 grid, less any that leaves no step of the Steps box or passes 1.
// model_defaults.strength_sweep is the same arithmetic in the same order, so
// the Strength tip's example and this button's sweep are the same numbers.
// Half rounds up on the quotient snapped to six places: Z-Anime's 0.15 makes
// ties of both neighbours, which float error floored to 0.05,0.15,0.2.
// `dropped` is each value left out and why, for the button's title: a source
// of few steps loses the half first, and its label then counts two.
function compareSweep(steps = refineStepsNow()) {
  const start = refineDefaultStrength();
  const values = [];
  const dropped = [];
  for (const factor of [0.5, 1, 1.5]) {
    const value = +(Math.floor(+(factor * start / 0.05).toFixed(6) + 0.5) * 0.05).toFixed(2);
    if (values.includes(value)) continue;
    const why = value <= 0 ? `half of ${start} rounds to 0`
      : value > 1 ? `${value} is past 1`
        : Math.floor(steps * value) < 1 ? `${value} leaves no step at ${steps} steps` : null;
    if (why) dropped.push(why);
    else values.push(value);
  }
  return { start, values, dropped };
}

function compareStrengths(steps = refineStepsNow()) {
  return compareSweep(steps).values;
}

function refineStepsNow() {
  return +els.refineSteps.value || lastDefaults?.steps || 28;
}

// The least Steps at which Compare 3 has two values to sweep: a source made
// at 2 steps - a 6-12x2 steps grid makes them - left none of 0.05, 0.1 and
// 0.15 a step, and the button wrote an empty Strength.
function compareStepsNeeded() {
  for (let steps = refineStepsNow() + 1; steps <= 1000; steps++) {
    if (compareStrengths(steps).length >= 2) return steps;
  }
  return null;
}

// What a pass at w x h keeps of a sw x sh source, as fractions of it:
// refine.fitted's arithmetic - scaled to cover, each side rounded as Python
// rounds (half to even), then the middle cropped out - so the frame drawn on
// the source and the size line's numbers are the pass's own.
function refineKept(sw, sh, w, h) {
  if (sw === w && sh === h) return { x: 0, y: 0, w: 1, h: 1 };
  const even = (v) => { const r = Math.round(v); return r - v === 0.5 && r % 2 ? r - 1 : r; };
  const cover = Math.max(w / sw, h / sh);
  const rw = Math.max(w, even(sw * cover));
  const rh = Math.max(h, even(sh * cover));
  return { x: Math.floor((rw - w) / 2) / rw, y: Math.floor((rh - h) / 2) / rh, w: w / rw, h: h / rh };
}

// Why the dialog opened below the scale it was asked for, named by the first
// scale above the one it opened at: "1.25× and larger go past the 1536×1536
// (2.36 MP) measured on Z-Anime at fp8". Pixels grow with the factor, so what
// holds for that scale holds for every larger one.
function refineCapNote(src, run) {
  const next = REFINE_SCALES.find((s) => s > refineSkipped.opened);
  if (next === undefined) return '';
  const size = refineScaled(src.width, src.height, next);
  const why = !refineFits(size) ? 'are past what a refine pass renders'
    : run ? `go ${run}` : 'may run tight on 16GB VRAM';
  return `${next}× and larger ${why}`;
}

function refineSizeProblem(w, h) {
  if (!(w >= 256 && h >= 256)) return 'each side needs at least 256 pixels';
  if (w % 16 || h % 16) return 'sides must be multiples of 16';
  if (!refineFits([w, h])) {
    return `past the ${REFINE_MAX_SIDE}-pixel side and ${(REFINE_MAX_PIXELS / 1e6).toFixed(1)} MP a refine pass takes`;
  }
  return null;
}

// Opens the dialog on `rec`, the image to refine: a card's ⤢ or the
// lightbox's Refine. `pass` is a refined record to start from instead - Reuse
// on one - with its size, strength, seed, prompt and sampling. Otherwise the
// source's own prompt, sampling and seed, the scale last run (1.5x the first
// time) unless that is past the pass's guard or the tight line, the versions
// last run and the strength last run on this checkpoint, else its profile's -
// or, after a close that left settings unsent on this image, those, which it
// says. Focus lands on Strength, its value selected: what the dialog is
// opened to set. Answers whether it opened.
function openRefine(rec, pass = null) {
  // The lightbox's Refine on a checkpoint with no pass explains itself in the
  // panel's line, which showError mirrors into the lightbox, and opens
  // nothing. A card's ⤢ never gets here: card() says it under the gallery's
  // head and over the card (refuseInCard). Reuse of a refined record opens
  // anyway: the pass is kept, with the reason, until a load says it can run.
  if (!refineSupported && !pass) {
    showError(REFINE_NONE, true);
    return false;
  }
  // Opened from the lightbox, a close that does not run goes back to it
  // (`refineClosed`): looking at the dialog used to cost the walk and the zoom.
  refineReturn = lightboxPlace();
  if (els.lightbox.open) closeLightbox();
  showError('');
  // Unsent settings are one image's, on one checkpoint, and only a plain open
  // puts them back: Reuse opens on the pass it names. Any other open forgets
  // them.
  const draft = !pass && unsentRefine?.id === rec.id && unsentRefine.model === loadedModel
    ? unsentRefine : null;
  unsentRefine = null;
  refineSource = rec;
  refineRan = false;
  const prefs = refinePrefs();
  const from = pass ?? rec;
  // An image from another checkpoint refines like any other - the pass starts
  // from pixels - but its steps, CFG, mu and strength are that checkpoint's,
  // so this one's profile stands in for them.
  const foreign = modelsDiffer(from.model_id);
  // Opened on the source either way, since the pass starts from its pixels;
  // the title and a line of its own say which pass is being run again.
  els.refineTitle.textContent = pass ? `Refine #${rec.id} as #${pass.id} was made` : `Refine #${rec.id}`;
  els.refineReused.hidden = !pass;
  showRefineSource();
  writeRefineSourceLine(rec);
  els.refinePromptLine.textContent = rec.prompt;
  // Rebuilt by syncRefine for this source's size.
  els.refineScales.textContent = '';
  refineSkipped = null;
  refineCapped = false;
  if (pass) {
    [els.refineWidth.value, els.refineHeight.value] = [pass.width, pass.height];
  } else {
    const scale = +prefs.scale || REFINE_SCALE;
    const { size, factor } = openingRefineSize(rec, scale);
    [els.refineWidth.value, els.refineHeight.value] = size;
    refineCapped = factor !== scale;
    if (refineCapped) refineSkipped = { scale, remembered: !!+prefs.scale, opened: factor, size };
  }
  const kept = prefs.strength?.[loadedModel];
  const keptOne = typeof kept === 'string' && SINGLE_STRENGTH.test(kept);
  const carried = pass && !foreign && pass.strength != null;
  els.refineStrength.value = carried ? pass.strength : keptOne ? kept : refineDefaultStrength();
  // A pass from another checkpoint ran at a strength of that checkpoint's
  // schedule, which reads otherwise on this one's: Strength starts at this
  // checkpoint's own, and the line says so - it named the old number beside a
  // box holding another, with nothing to say which would run.
  els.refineReused.textContent = !pass ? ''
    : carried || pass.strength == null
      ? `Reusing #${pass.id}: ${pass.width}×${pass.height}`
        + `${pass.strength != null ? ` at strength ${pass.strength}` : ''}, seed ${pass.seed}`
      : `Reusing #${pass.id}: ${pass.width}×${pass.height}, seed ${pass.seed} - its strength ${pass.strength} `
        + `was ${checkpointName(pass.model_id)}'s; Strength starts at `
        + (keptOne ? `${kept}, as last run on ${checkpointName(loadedModel)}`
          : lastDefaults?.refine_strength_profiled ? `${checkpointName(loadedModel)}'s ${els.refineStrength.value}`
            : `${els.refineStrength.value}, the general starting strength`);
  els.refineVersions.value = !pass && ['1', '2', '4'].includes(String(prefs.versions))
    ? String(prefs.versions) : '1';
  els.refineSeed.value = from.seed;
  els.refinePrompt.value = from.prompt;
  els.refineNegative.value = from.negative_prompt ?? '';
  const d = lastDefaults;
  els.refineSteps.value = foreign ? (d?.steps ?? 28) : from.steps;
  els.refineCfg.value = foreign ? (d?.guidance_scale ?? 3.5) : from.guidance_scale;
  els.refineMu.value = foreign ? (d?.mu ?? '') : (from.mu ?? '');
  // The scale the negative prompt ran at, where True CFG is a knob of its
  // own; on a checkpoint whose CFG is real CFG the record's value is the CFG.
  els.refineTrueCfg.value = !foreign && trueCfgSupported && from.true_cfg != null ? from.true_cfg : '';
  // The adapter carries over from this checkpoint's own records, and only one
  // the server still lists for it; anything else is said, not sent.
  const listed = [...els.lora.options].some((o) => o.value === from.lora);
  refineLora = !foreign && loraSupported && from.lora && listed
    ? { name: from.lora, scale: from.lora_scale ?? 1 } : null;
  els.refineCarried.textContent = refineLora
    ? `LoRA ${refineLora.name} at ${refineLora.scale}, as #${from.id} was made.`
    : from.lora
      ? `LoRA ${from.lora} is not applied: ${foreign ? 'it was for another checkpoint'
        : !loraSupported ? 'this installation cannot load adapters'
          : 'it is not in the adapter list now'}.`
      : '';
  // A pass's strength is replaced too (above); a source's never carries.
  // Mu named only where its box can be checked, as Reuse's note does
  // (reuseIntoPanel): it is greyed on every checkpoint without one.
  const knobs = ['steps', 'CFG', ...(muSupported ? ['mu'] : []), ...(pass ? ['strength'] : [])];
  els.refineMoreNote.textContent = foreign
    ? `made on ${checkpointName(from.model_id)}: ${knobs.slice(0, -1).join(', ')} and ${knobs.at(-1)} are `
      + `${checkpointName(loadedModel)}'s`
    : `as #${from.id} was made`;
  // Open where they could not carry over, so the substitution is seen.
  els.refineMore.open = foreign;
  refineFilled = refineValues();
  if (draft) applyRefineValues(draft.values);
  els.refineDraft.hidden = !draft;
  els.refineDraftText.textContent = draft ? `Your unsent settings for #${rec.id}, as you left them.` : '';
  refineProblem = null;
  refinePreviewed = null;
  refineFailed('');
  sayRefineFromPanel('');
  unmarkRefineFields();
  els.refinePreview.textContent = '';
  refineChecked = null;
  renderRefineCheck();
  els.refineTags.hidden = true;
  hideRefineSuggestions();
  syncRefine();
  if (!els.refineDialog.open) els.refineDialog.showModal();
  els.refineDialog.scrollTop = 0;
  els.refineStrength.focus();
  els.refineStrength.select();
  sizeRefinePrompt();
  scheduleRefinePreview(0);
  scheduleRefineCheck(0);
  scheduleRefineTags(0);
  void loadRefinePasses(rec);
  return true;
}

// The source's size, seed and checkpoint, and its own source where it is a
// pass - and, once the server has refused it as gone (`refineSourceGone`),
// that it is: "#28 (deleted)", as the viewer and the lightbox say it.
function writeRefineSourceLine(rec, gone = false) {
  els.refineSourceLine.textContent = [
    `${rec.width}×${rec.height}`, `seed ${rec.seed}`, checkpointName(rec.model_id),
    rec.refine != null ? `itself a refine of #${rec.refine}${rec.refine_missing ? ', since deleted' : ''}` : '',
    gone ? `#${rec.id} (deleted)` : '',
  ].filter(Boolean).join(' · ');
}

// A refusal of the source itself - deleted from another tab while the
// dialog was open - in the page's words, with the source line marked: the
// server's "no gallery image with id 28" read as a setting gone wrong, in a
// dialog still titled "Refine #28" over the picture. The dialog stays open
// with its edits, as for any refusal. Any other message is returned as it is.
function refineSourceGone(message, source = refineSource) {
  const match = /^no gallery image with id (\d+)$/.exec(message ?? '');
  if (!source || !match || +match[1] !== source.id) return message;
  if (refineSource === source) writeRefineSourceLine(source, true);
  return `#${source.id} has been deleted, so it cannot be refined.`;
}

// Every box openRefine fills, by name: what a close compares with what it
// filled, and what reopening puts back.
function refineBoxes() {
  return {
    strength: els.refineStrength, width: els.refineWidth, height: els.refineHeight,
    versions: els.refineVersions, seed: els.refineSeed, prompt: els.refinePrompt,
    negative: els.refineNegative, steps: els.refineSteps, cfg: els.refineCfg,
    mu: els.refineMu, trueCfg: els.refineTrueCfg,
  };
}

// The boxes' text, and whether the fold was open - kept with them, though
// opening it alone is not a setting left unsent.
function refineValues() {
  const values = Object.fromEntries(Object.entries(refineBoxes()).map(([key, el]) => [key, el.value]));
  return { ...values, more: els.refineMore.open };
}

function applyRefineValues(values) {
  for (const [key, el] of Object.entries(refineBoxes())) {
    if (typeof values[key] === 'string') el.value = values[key];
  }
  els.refineMore.open = !!values.more;
}

// Start over: the record's own settings again, as a first open fills them.
function startRefineOver() {
  const rec = refineSource;
  if (!rec) return;
  unsentRefine = null;
  openRefine(rec);
}

// Reset forgets the dialog with the panel: what it remembers per browser,
// and any settings left unsent in this page.
function forgetRefine() {
  forget(REFINE_KEY);
  unsentRefine = null;
}

// The source's picture: the PNG itself as the left pane, the thumbnail in one
// column - again when the window crosses the line with the dialog open. The
// figure takes the record's shape, so the frame's percentages are the picture's.
function showRefineSource() {
  const rec = refineSource;
  if (!rec) return;
  const src = REFINE_WIDE.matches ? rec.url : (rec.thumb_url ?? rec.url);
  if (els.refineThumb.getAttribute('src') !== src) els.refineThumb.src = src;
  els.refineThumb.alt = `#${rec.id}, the source`;
  els.refineFigure.style.setProperty('--rf-ratio', `${rec.width} / ${rec.height}`);
}

// Everything the dialog shows that follows its boxes and the server: the
// scale buttons, the size line and the kept region, the strength's example
// sweep, the seed line, the negative prompt's activation, Mu and True CFG
// where the checkpoint has them, and whether Refine can run. Status calls it
// too, since those answers can change.
function syncRefine() {
  const src = refineSource;
  if (!src) return;
  const w = +els.refineWidth.value;
  const h = +els.refineHeight.value;
  if (!els.refineScales.children.length) {
    for (const scale of REFINE_SCALES) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `${scale}×`;
      button.onclick = () => applyRefineScale(scale);
      els.refineScales.append(button);
    }
  }
  // The chosen scale is pressed, as the panel's size strips are: .active
  // alone was the look of it, and a screen reader heard four plain buttons.
  [...els.refineScales.children].forEach((button, i) => {
    const size = refineScaled(src.width, src.height, REFINE_SCALES[i]);
    const fits = refineFits(size);
    const chosen = fits && w === size[0] && h === size[1];
    button.disabled = !fits;
    button.classList.toggle('active', chosen);
    button.setAttribute('aria-pressed', String(chosen));
    const skipped = REFINE_SCALES[i] !== refineSkipped?.scale ? ''
      : refineSkipped.remembered ? ` (last run at ${refineSkipped.scale}×)`
        : ' (the usual opening size)';
    button.title = (fits ? `${size[0]} × ${size[1]}` : `${size[0]} × ${size[1]} is past what a refine pass renders`)
      + skipped;
    button.setAttribute('aria-label', `${REFINE_SCALES[i]} times the source's size, ${button.title}`);
    // Shown under the factor (style.css), so its text stays the factor.
    button.dataset.size = `${size[0]}×${size[1]}`;
  });

  // Past the tight line, the measured run it is, by name: a profile can be
  // shared, and FLUX.1 dev quotes Krea dev's. A size of another shape says
  // how much of the source it keeps, and frames it on the large source.
  const problem = refineSizeProblem(w, h);
  const run = typeof lastDefaults?.refine_tight_pixels === 'number' ? lastHints?.refine_tight : null;
  const tight = !problem && refineTight(w, h);
  const notes = problem ? [problem] : tight
    ? [run ? `may run tight on 16GB VRAM, ${run}` : 'may run tight on 16GB VRAM']
    : [];
  const fx = w / src.width;
  const fy = h / src.height;
  const sized = w > 0 && h > 0;
  const cropped = sized && Math.abs(fx - fy) > 0.02 * Math.max(fx, fy);
  const kept = sized ? refineKept(src.width, src.height, w, h) : null;
  const percent = (part) => `${Math.round(part * 100)}%`;
  const shape = !sized ? '' : !cropped ? `${+((fx + fy) / 2).toFixed(2)}× the source`
    : `the source cropped to this shape, keeping the middle ${percent(kept.w)} × ${percent(kept.h)}`;
  // At the size it opened at in place of the scale asked for, why: the next
  // scale up is past the line (or the guard), and so is every larger one.
  const capped = !notes.length && refineSkipped && w === refineSkipped.size[0] && h === refineSkipped.size[1]
    ? refineCapNote(src, run) : '';
  els.refineSizeHint.textContent = [sized ? `${(w * h / 1e6).toFixed(1)} MP` : '', shape, ...notes, capped]
    .filter(Boolean).join(' — ');
  els.refineSizeHint.classList.toggle('warn', notes.length > 0);
  els.refineKept.hidden = !cropped;
  if (cropped) {
    Object.assign(els.refineKept.style, {
      left: `${kept.x * 100}%`, top: `${kept.y * 100}%`, width: `${kept.w * 100}%`, height: `${kept.h * 100}%`,
    });
  }
  // What a larger pass does to a strength, where the checkpoint has words for
  // it (`hints.refine_large`, status.js writes them in): FLUX.1's measured runs.
  els.refineLarge.hidden = !tight || !els.refineLarge.textContent;

  // Strength's own help: the profile's number with the sweep around it, as
  // the placeholder, the Compare 3 button and a line saying how a sweep is
  // written - the same three numbers as the tip's example.
  // Steps low enough to leave fewer than two of them a step has no sweep to
  // offer: the button is held, saying the Steps that would, and the lines
  // name none. Two of them is a sweep of two: the label counts them, and the
  // title names the one left out and why - it promised "either side".
  const { start, values: sweepValues, dropped } = compareSweep();
  const compare = sweepValues.join(',');
  const swept = sweepValues.length >= 2;
  const need = swept ? null : compareStepsNeeded();
  els.refineStrength.placeholder = swept ? `${start} or ${compare}` : `${start}`;
  els.refineCompare.setAttribute('aria-disabled', String(!swept));
  // Written only when it changes: this runs on every status poll while the
  // dialog is open, and the button may hold focus.
  const compareLabel = `Compare ${sweepValues.length === 2 ? 2 : 3}`;
  if (els.refineCompare.textContent !== compareLabel) els.refineCompare.textContent = compareLabel;
  const other = sweepValues.find((v) => v !== start);
  const around = sweepValues.length === 3 ? `this checkpoint's ${start} and either side of it`
    : sweepValues.includes(start) ? `this checkpoint's ${start} and ${other} ${other < start ? 'below' : 'above'} it`
      : `around this checkpoint's ${start}`;
  els.refineCompare.title = swept
    ? `Write ${compare} into Strength: ${around}, run side by side as one sweep`
      + (dropped.length ? ` - ${dropped.join(', and ')}` : '')
    : `No sweep at ${refineStepsNow()} steps: fewer than two strengths around ${start} leave a step`
      + (need ? ` - raise Steps to ${need}` : '');
  els.refineSweepHint.textContent = swept
    ? `Several values (${compare}) run side by side and open as a sheet.`
    : 'Several values, comma-separated, run side by side and open as a sheet'
      + (need ? `; at ${refineStepsNow()} steps a sweep around ${start} needs Steps of ${need} or more.` : '.');

  // The seeds the versions take, counting up from the box like a Batch: one
  // request with that many images, or a sweep's seed count for every strength.
  const versions = +els.refineVersions.value || 1;
  const sweep = !SINGLE_STRENGTH.test(els.refineStrength.value.trim()) && els.refineStrength.value.trim() !== '';
  const seedText = els.refineSeed.value.trim();
  const seed = seedText === '' ? null : +seedText;
  const seeds = seed === null
    ? (versions > 1 ? `${versions} seeds counting up from a random one` : 'a random seed')
    : versions > 1 ? `seeds ${seed}-${seed + versions - 1}` : `seed ${seed}`;
  els.refineSeedHint.textContent = `${seeds}${seed === src.seed ? ", from the source's" : ''}`
    + (sweep ? ', at every strength' : '');

  // The negative prompt, Mu and True CFG by the panel's rules (syncCfg,
  // syncMu): greyed where they cannot apply, so nothing unused is sent.
  const trueCfg = els.refineTrueCfg.value.trim() === '' ? 0 : +els.refineTrueCfg.value;
  const reach = trueCfgSupported ? trueCfg : +els.refineCfg.value;
  const off = !negativeSupported || reach <= cfgThreshold;
  els.refineNegative.disabled = off;
  els.refineNegField.classList.toggle('disabled', off);
  // In the panel's words for the same state (sizes.js, syncCfg): where CFG
  // is the knob a live box costs nothing more, and only a greyed one says
  // what waking it costs.
  const knob = trueCfgSupported ? 'True CFG' : 'CFG';
  els.refineNegNote.textContent = !negativeSupported
    ? '(this checkpoint takes no negative prompt)'
    : !trueCfgSupported && !off
      ? `(runs at CFG > ${cfgThreshold}, no extra cost)`
      : `(needs ${knob} > ${cfgThreshold} — ~2× slower)`;
  // Blank is the pipeline's own shift, as the panel's box says (syncMu) -
  // with its value where the scheduler's shift is fixed; a dead box promises
  // nothing.
  els.refineMu.placeholder = muPlaceholder();
  els.refineMu.disabled = !muSupported;
  els.refineMuField.classList.toggle('disabled', !muSupported);
  if (!muSupported) els.refineMu.value = '';
  els.refineTrueCfg.disabled = !trueCfgSupported;
  els.refineTrueCfgField.classList.toggle('disabled', !trueCfgSupported);
  if (!trueCfgSupported) els.refineTrueCfg.value = '';

  // A guess - the profile's, before a load - is said as one, and the pass
  // waits here for the load to answer. An answer that is final - the loaded
  // pipeline's, or a pipeline of this repo's own, which diffusers has no
  // image-to-image class beside (`refine_settled`) - is said as the card's ⤢
  // says it: "a load answers" sent a user with Anima loaded off to load it.
  els.refineUnavailable.hidden = refineSupported;
  els.refineUnavailable.textContent = refineSupported ? ''
    : lastDefaults?.refine_settled ? `Refine is unavailable. ${REFINE_NONE}`
      : 'Refine is unavailable: as far as the server can tell, this checkpoint has no compatible '
        + 'image-to-image pipeline. A load answers for certain - the pass stays here until then.';
  syncRefineFromPanel();
  syncRefineRun();
  // After the negative's activation has moved: the check's Show and Apply
  // follow it.
  syncRefineCheck();
}

// Held, with the reason as its title, as Generate and Run grid are
// (syncSubmitButtons): first for what holds those - a server that does not
// answer, Expand or the critic running (submitHold), which startJob refuses
// anyway, but the button said nothing of it before the press - then no
// refine pass, a size past the guard, or the preview's refusal.
function syncRefineRun() {
  const problem = submitHold()
    || (!refineSupported ? 'This checkpoint has no refine pass'
      : refineSizeProblem(+els.refineWidth.value, +els.refineHeight.value) || refineProblem);
  els.refineRun.setAttribute('aria-disabled', String(!!problem));
  els.refineRun.title = problem || 'Run the pass (Ctrl+Enter)';
}

function applyRefineScale(scale) {
  if (!refineSource) return;
  [els.refineWidth.value, els.refineHeight.value] = refineScaled(refineSource.width, refineSource.height, scale);
  unmarkRefineFields();
  refineFailed('');
  syncRefine();
  scheduleRefinePreview(0);
}

// Every box: the lines follow at once, the server's preview a moment later.
function refineChanged() {
  unmarkRefineFields();
  refineFailed('');
  sayRefineFromPanel('');
  syncRefine();
  scheduleRefinePreview();
}

// W and H follow each other at the source's shape while Keep shape is ticked:
// typing one side is the natural way to ask for "2000 wide", and alone it
// reframed the picture. Typed input only - a script that sets both sides (a
// test, a restored draft) means the size it sets. Before refineChanged on
// the same event (app.js), so the lines read the pair.
function refineSideTyped(event) {
  if (!event.isTrusted || !els.refineLock.checked || !refineSource) return;
  const typed = +event.target.value;
  if (!(typed > 0)) return;
  const { width, height } = refineSource;
  const snap = (side) => Math.max(16, Math.floor(side / 16 + 0.5) * 16);
  if (event.target === els.refineWidth) els.refineHeight.value = snap(typed * height / width);
  else els.refineWidth.value = snap(typed * width / height);
}

// Keys inside the dialog. A plain Enter in a one-line box runs the pass, as a
// form's implicit submit would: Strength has focus on open, so "open, Enter"
// still runs at what the dialog opened with, and "0.3, Enter" at 0.3.
// ArrowUp and ArrowDown step a single Strength by 0.05, 0.01 with Shift, as
// a number box steps; a list or a range is text and left alone. Ctrl+Enter is
// the document's (app.js), and runs the pass from anywhere in the dialog.
function refineKeydown(event) {
  if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === 'Enter' && !event.shiftKey && event.target.matches('input:not([type="checkbox"])')) {
    event.preventDefault();
    void runRefine();
    return;
  }
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && event.target === els.refineStrength) {
    const text = els.refineStrength.value.trim();
    if (!SINGLE_STRENGTH.test(text)) return;
    event.preventDefault();
    const step = (event.shiftKey ? 0.01 : 0.05) * (event.key === 'ArrowUp' ? 1 : -1);
    els.refineStrength.value = String(Math.min(1, Math.max(0.01, +(+text + step).toFixed(2))));
    refineChanged();
  }
}

// Compare 3 pressed: the sweep written into Strength, which the preview then
// prices and Refine runs as one sheet. Never remembered, as no sweep is.
function fillRefineCompare() {
  // Held where Steps leaves no sweep: the press says why, and writes nothing
  // over the user's Strength.
  const values = compareStrengths();
  if (els.refineCompare.getAttribute('aria-disabled') === 'true' || values.length < 2) {
    refineFailed(`${els.refineCompare.title}.`);
    return;
  }
  els.refineStrength.value = values.join(',');
  refineChanged();
  els.refineStrength.focus();
}

function refineFailed(message) {
  els.refineError.hidden = !message;
  els.refineError.textContent = message || '';
}

// The request the dialog stands for. One strength is an ordinary request
// whose `num_images` is the versions - a pass is one pipeline call per image,
// so versions cost time, not VRAM. Several are a grid whose strength axis
// crosses that many seeds, and whose sheet opens when it is done. `preview`
// is the same as a grid, for /api/grid/preview.
function refineRequest() {
  const num = (el) => (el.value.trim() === '' ? undefined : +el.value);
  const versions = +els.refineVersions.value || 1;
  const strength = els.refineStrength.value.trim();
  const single = strength === '' || SINGLE_STRENGTH.test(strength);
  const seed = els.refineSeed.value.trim();
  const base = {
    prompt: els.refinePrompt.value.trim(),
    negative_prompt: els.refineNegative.disabled ? null : els.refineNegative.value.trim() || null,
    width: +els.refineWidth.value,
    height: +els.refineHeight.value,
    steps: num(els.refineSteps),
    guidance_scale: num(els.refineCfg),
    true_cfg: els.refineTrueCfg.disabled || els.refineTrueCfg.value.trim() === '' ? null : +els.refineTrueCfg.value,
    // Blank is "no shift of ours", as the panel's Mu box sends it.
    mu: els.refineMu.disabled || els.refineMu.value.trim() === '' ? null : +els.refineMu.value,
    lora: refineLora?.name ?? null,
    lora_scale: refineLora?.scale ?? 1,
    refine: refineSource.id,
    // Blank is "not given": the server fills in the checkpoint's own.
    strength: single && strength !== '' ? +strength : undefined,
    seed: seed === '' ? null : +seed,
    runs: 1,
  };
  if (single) {
    const body = { ...base, num_images: versions };
    return { path: '/api/generate', body, preview: { ...body, seeds: 1 }, sweep: false };
  }
  const body = { ...base, num_images: 1, seeds: versions, strength_axis: strength };
  return { path: '/api/grid', body, preview: body, sweep: true };
}

function scheduleRefinePreview(delay = 250) {
  clearTimeout(refineTimer);
  refineTimer = setTimeout(previewRefine, delay);
}

// The server's own reading of the pass: the image count, the strengths a
// sweep expanded to, and an estimate from passes already made at this size
// and step count - the preview the grid section asks for, so a strength that
// leaves no step, or a size past the guard, is refused here in the server's
// words. Replies that overtake each other are dropped by the epoch.
async function previewRefine() {
  if (!refineSource || !els.refineDialog.open) return;
  const epoch = ++refineEpoch;
  const { preview, sweep } = refineRequest();
  const asked = JSON.stringify(preview);
  els.refinePreview.classList.remove('warn');
  refineProblem = null;
  if (!refineSupported || refineSizeProblem(preview.width, preview.height)) {
    els.refinePreview.textContent = '';
    refinePreviewed = asked;
    syncRefineRun();
    return;
  }
  if (!preview.prompt) {
    refineProblem = 'Enter a prompt: the pass is denoised under one.';
    els.refinePreview.classList.add('warn');
    els.refinePreview.textContent = refineProblem;
    refinePreviewed = asked;
    syncRefineRun();
    return;
  }
  try {
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(preview),
    });
    if (epoch !== refineEpoch) return;
    const strengths = p.values.strength ?? [];
    const steps = p.values.steps?.[0];
    // About: the pipelines round differently (SDXL floors, the flow ones
    // round up), and the progress bar counts what actually runs.
    const run = !sweep && typeof steps === 'number' && strengths.length === 1
      ? ` · about ${Math.max(1, Math.round(steps * strengths[0]))} of ${steps} steps run` : '';
    // Description, not a refusal: the tip says 1 keeps nothing, and here
    // the number is about to.
    const single = !sweep && strengths.length === 1 ? strengths[0] : null;
    const keeps = single === null ? '' : single >= 1 ? ' · keeps nothing of the source'
      : single >= 0.9 ? ' · keeps almost nothing of the source' : '';
    // Where the pass starts, read off the loaded schedule for this size
    // (`start_sigma`): on FLUX.1 one strength starts a larger pass at more
    // noise, its shift growing with the size - 0.1 is 47% at 1.5x of 1 MP
    // and 75% at 2x. Nothing before a load, or where the schedule's sigma is
    // no noise fraction (SDXL).
    const noise = single !== null && typeof p.start_sigma === 'number'
      ? ` · starts at ${Math.round(p.start_sigma * 100)}% noise` : '';
    els.refinePreview.classList.toggle('warn', !!p.duplicates);
    els.refinePreview.textContent = plural(p.images, 'image')
      + ` · strength ${axisText(strengths)}${noise}${run}${keeps}`
      + (p.seconds ? ` · about ${humanTime(p.seconds)}` : '')
      + (p.duplicates ? ` · ${p.duplicates} would be identical to another` : '')
      + (sweep ? '\nOpens as a sheet when it is done.' : '');
  } catch (err) {
    if (epoch !== refineEpoch) return;
    refineProblem = refineSourceGone(err.message);
    els.refinePreview.classList.add('warn');
    els.refinePreview.textContent = refineProblem;
  }
  refinePreviewed = asked;
  syncRefineRun();
}

// Refine: the job queues like Generate's - behind a running one if there is
// one, its progress in the panel, its images in the strip - and the dialog
// closes once the server has taken it. A sweep opens as its sheet when done.
// Never the panel's Seed box: the dialog's seed is its own.
async function runRefine() {
  if (!refineSource || !els.refineDialog.open || submitting || refineWaiting) return;
  // The preview is what refuses a strength past 1 or one that leaves no step,
  // and it waits out the typing: Enter straight after "1.5" ran ahead of it,
  // closed the dialog and put the server's refusal in the panel, the typed
  // settings gone with it. So the preview of what the boxes hold answers
  // first - again if they changed while it was asked.
  while (refinePreviewed !== JSON.stringify(refineRequest().preview)) {
    clearTimeout(refineTimer);
    refineWaiting = true;
    try { await previewRefine(); } finally { refineWaiting = false; }
    if (!refineSource || !els.refineDialog.open || submitting) return;
  }
  if (els.refineRun.getAttribute('aria-disabled') === 'true') {
    // What holds Generate too, said as a sentence in the dialog, where it is
    // still up to say so; as Generate's press, the poll asks again at once
    // where the server did not answer.
    const held = submitHold();
    if (held) refineFailed(`${held}.`);
    else refuseRefine(els.refineRun.title);
    if (serverOffline) void refreshStatus();
    return;
  }
  const { path, body, sweep } = refineRequest();
  if (!body.prompt) { refineFailed('Enter a prompt first.'); return; }
  // The dialog closes once the server has taken the pass, not before: a
  // refusal at submit - a source deleted elsewhere - landed in the panel's
  // line with the dialog and its edits gone. It stays open now, the refusal
  // in its own line. Until the pass's first frame, the viewer shows its
  // source standing in for it (startJob's `placeholder`), captioned with
  // where the pass is: on SDXL, which sends no frames, it showed whatever
  // image was there before for the whole pass. The full image - the wide
  // dialog's source pane has it already - since the thumbnail, stretched to
  // the viewer, was a fourfold upscale at QHD.
  const source = refineSource;
  const strength = els.refineStrength.value.trim() || String(refineDefaultStrength());
  // Held while the POST is out (`refineSending`): an Escape or a Cancel then
  // looked like a cancel, put the lightbox back as though nothing had run,
  // and the pass the server had taken rendered behind it. The dialog says
  // so in its Refine button, and closes once the server answers.
  refineSending = true;
  let accepted = false;
  const settle = () => {
    refineSending = false;
    els.refineRun.textContent = busy ? 'Refine (queue)' : 'Refine';
  };
  // Resolves on a refusal, once an accepted job has ended, or - for one
  // queued behind the watched job - once the server has taken it (startJob).
  const sent = startJob(path, body, {
    sheet: sweep,
    panel: false,
    slot: els.refineError,
    placeholder: {
      src: source.url ?? source.thumb_url,
      width: body.width,
      height: body.height,
      label: `Refining #${source.id} → ${body.width}×${body.height} · strength ${strength}`,
    },
    onAccepted: () => {
      accepted = true;
      settle();
      // Unless a close the browser would not hold - a second Escape with no
      // other press between - closed it already: then refineClosed took it
      // as the run it is.
      if (refineSource !== source || !els.refineDialog.open) return;
      rememberRefine();
      refineRan = true;
      els.refineDialog.close();
      // The dialog gives focus back to what opened it: nothing, where a
      // script did (Reuse of a pass), or a button that has gone with the
      // lightbox. Then Generate - or New image, beside the progress (in one
      // column, a pill under the topbar), where the strip the pass shows in
      // is out of sight - rather than <body>.
      // refineClosed puts it on the source's card where the lightbox opened
      // the dialog. A sweep's sheet takes focus on from either (handJobFocus).
      requestAnimationFrame(() => {
        if (document.activeElement && document.activeElement !== document.body) return;
        handJobFocus(els.newImage.hidden ? els.generate : els.newImage);
      });
    },
  });
  // After startJob's own start, whose setBusy names it "Refine (queue)".
  if (refineSending) els.refineRun.textContent = 'Sending…';
  await sent;
  if (accepted) return;
  settle();
  // A source deleted since the last preview is refused at submit, in the
  // server's words, which startJob wrote into the dialog's line: said again
  // in the page's before the task ends, so the alert is read once, in these.
  const refused = els.refineError.textContent;
  const said = refineSourceGone(refused, source);
  if (said !== refused) refineFailed(said);
  // Refused with the dialog gone: its line is out of sight, so the panel's says it.
  if (!els.refineDialog.open && els.refineError.textContent) showError(els.refineError.textContent);
}

// Cancel, the cross and Escape, except while the pass's POST is out: the
// server may already have it, and the dialog closes on its answer.
function closeRefine() {
  if (!refineSending) els.refineDialog.close();
}

function refineCancelEvent(event) {
  if (refineSending) event.preventDefault();
}

// A press on a refused Refine says why in the dialog, once, and points at the
// field. The preview's own refusal moves to the error line - the same sentence
// twice, amber and red, read as two problems - in the preview's place, which
// it leaves hidden, and at its size (style.css), so Refine does not move under
// the pointer; Strength is focused and marked where the sentence is about it,
// and the next change brings the preview back. A size past the guard keeps
// the size line beside W and H, says it again by the button, and focuses W.
function refuseRefine(reason) {
  const size = refineSupported && refineSizeProblem(+els.refineWidth.value, +els.refineHeight.value);
  if (refineSupported && !size && reason === refineProblem && els.refinePreview.textContent === reason) {
    els.refinePreview.textContent = '';
    els.refinePreview.classList.remove('warn');
    els.refinePreview.hidden = true;
    refineFailed(reason);
    if (/strength/i.test(reason)) markRefineFields(els.refineStrength);
    return;
  }
  refineFailed(reason);
  if (size) markRefineFields(els.refineWidth, els.refineHeight);
}

function markRefineFields(...fields) {
  for (const el of fields) el.setAttribute('aria-invalid', 'true');
  fields[0].focus();
}

// A refusal's marks off, and the preview back in its place.
function unmarkRefineFields() {
  for (const el of [els.refineStrength, els.refineWidth, els.refineHeight]) el.removeAttribute('aria-invalid');
  els.refinePreview.hidden = false;
}

function rememberRefine() {
  const prefs = refinePrefs();
  const src = refineSource;
  const w = +els.refineWidth.value;
  const h = +els.refineHeight.value;
  const scale = REFINE_SCALES.find((s) => {
    const [sw, sh] = refineScaled(src.width, src.height, s);
    return sw === w && sh === h;
  });
  // Only a scale the user chose. Where the dialog opened below the one it
  // was asked for, that size run as it opened is the dialog's choice:
  // remembered, one Enter on a large source opened every later one at 1x.
  const chosen = !refineCapped || w !== +refineFilled?.width || h !== +refineFilled?.height;
  if (scale && chosen) prefs.scale = scale;
  prefs.versions = +els.refineVersions.value || 1;
  const strengths = prefs.strength && typeof prefs.strength === 'object' ? prefs.strength : {};
  const text = els.refineStrength.value.trim();
  if (loadedModel && SINGLE_STRENGTH.test(text)) {
    if (+text === refineDefaultStrength()) delete strengths[loadedModel];
    else strengths[loadedModel] = text;
  }
  prefs.strength = strengths;
  remember(REFINE_KEY, JSON.stringify(prefs));
}

// Closed by Refine, Cancel, the cross, Escape or a pass opened from the
// strip: nothing is open any more, and a reply still in flight lands on
// nothing. A close that did not run keeps what was typed for the next open
// of the same image - a sweep, a size and a prompt edit used to be lost to a
// look at the picture - where the boxes differ from what openRefine filled;
// and, opened from the lightbox, gives the lightbox back where it was, unless
// the strip has opened it on a pass already - or, where it ran, puts the
// focus on the source's card.
function refineClosed() {
  const src = refineSource;
  // A close the browser would not hold while the POST was out is a run too:
  // the server most likely has the pass already (runRefine).
  const ran = refineRan || refineSending;
  const values = src && !ran ? refineValues() : null;
  const unsent = values && Object.keys(refineBoxes()).some((key) => values[key] !== refineFilled?.[key]);
  unsentRefine = unsent ? { id: src.id, model: loadedModel, values } : null;
  refineSource = null;
  refineLora = null;
  refineEpoch++;
  refineCheckEpoch++;
  refineTagEpoch++;
  refinePassesEpoch++;
  clearTimeout(refineTimer);
  clearTimeout(refineCheckTimer);
  clearTimeout(refineTagTimer);
  hideRefineSuggestions();
  sayRefineFromPanel('');
  const back = refineReturn;
  refineReturn = null;
  if (!back || els.lightbox.open) return;
  if (!ran) { returnToLightbox(back); return; }
  // Run from a dialog the lightbox opened: the dialog gives focus back to the
  // lightbox's Refine, closed with it, which left it on <body>. It goes to
  // the source's card, where the walk was, else Generate - without moving the
  // page, whose progress is where the pass now shows. A card the prompt check
  // covers (Gallery beside) is not one focus can land on.
  const card = [...document.querySelectorAll(`#gallery .card[data-id="${back.rec.id}"] img`)].find(isShown);
  handJobFocus(card ?? els.generate);
}

// ------------------------------------------------- the source's passes

// The passes already made from this source, newest first - the gallery's own
// `refine:` search, on every checkpoint - so a pass is judged against what it
// was made from before another is run. One opens in the lightbox with the
// source and the other passes beside it and the comparison already on
// (judging the pass against its source is what it was opened for, and it
// took an S or a press more); ← and → then swap between the passes, still
// compared (`stepPass`). The dialog closes for it, keeping what was typed for
// the next open (`refineClosed`).
async function loadRefinePasses(rec) {
  const epoch = ++refinePassesEpoch;
  els.refinePasses.hidden = true;
  els.refinePassList.replaceChildren();
  let page;
  try {
    page = await api(`/api/gallery?limit=${REFINE_PASSES}&all_models=true`
      + `&search=${encodeURIComponent(`refine:${rec.id}`)}`);
  } catch { return; }  // a convenience: the dialog works without it
  if (epoch !== refinePassesEpoch || refineSource !== rec || !page.total) return;
  els.refinePassesHead.textContent = `Passes of #${rec.id} (${page.total})`;
  for (const pass of page.items) {
    const what = `#${pass.id}: ${pass.width}×${pass.height} at strength ${pass.strength}, seed ${pass.seed}`;
    const button = document.createElement('button');
    button.type = 'button';
    button.title = what;
    button.setAttribute('aria-label', `Open the pass ${what} in the lightbox, compared with #${rec.id}`);
    const img = document.createElement('img');
    img.src = pass.thumb_url ?? pass.url;
    img.alt = '';
    img.loading = 'lazy';
    button.append(img);
    button.onclick = () => {
      if (refineSending) return;
      els.refineDialog.close();
      openLightbox(pass, [rec, ...page.items]);
      void toggleCompare();
    };
    els.refinePassList.append(button);
  }
  els.refinePasses.hidden = false;
}
