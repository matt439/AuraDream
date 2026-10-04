// Local Diffusion's page script: the Refine dialog. Declarations only; state.js has
// the load-order contract these files share.
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
// The scale last run, where the opening size passed over it: its button says
// so, one click away. `refineCapped`: the dialog opened below the scale it
// was asked for, so that size, run as it opened, is not the user's choice.
let refineSkipped = null;
let refineCapped = false;
// What openRefine filled the boxes with, and whether the dialog closed by
// running: a close that was neither keeps what was typed, in this page only,
// for the next open of the same image on the same checkpoint -
// {id, model, values}. Never in REFINE_KEY: a sweep left there would start on
// another image unasked.
let refineFilled = null;
let refineRan = false;
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
function compareStrengths() {
  const start = refineDefaultStrength();
  const steps = +els.refineSteps.value || lastDefaults?.steps || 28;
  const out = [];
  for (const factor of [0.5, 1, 1.5]) {
    const value = +(Math.floor(factor * start / 0.05 + 0.5) * 0.05).toFixed(2);
    if (value > 0 && value <= 1 && Math.floor(steps * value) >= 1 && !out.includes(value)) out.push(value);
  }
  return out;
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
  // A ⤢ press on a checkpoint with no pass explains itself in place, in the
  // lightbox too (showError mirrors there). Reuse of a refined record opens
  // anyway: the pass is kept, with the reason, until a load says it can run.
  if (!refineSupported && !pass) {
    showError(
      'This checkpoint has no refine pass compatible with this app. '
      + 'The FLUX.1, SDXL, Z-Image and Qwen-Image checkpoints have one.',
      true,
    );
    return false;
  }
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
  const short = (id) => (id || '').split('/').pop();
  // Opened on the source either way, since the pass starts from its pixels;
  // the title and a line of its own say which pass is being run again.
  els.refineTitle.textContent = pass ? `Refine #${rec.id} as #${pass.id} was made` : `Refine #${rec.id}`;
  els.refineReused.hidden = !pass;
  els.refineReused.textContent = pass
    ? `Reusing #${pass.id}: ${pass.width}×${pass.height} at strength ${pass.strength}, seed ${pass.seed}` : '';
  showRefineSource();
  els.refineSourceLine.textContent = [
    `${rec.width}×${rec.height}`, `seed ${rec.seed}`, short(rec.model_id),
    rec.refine != null ? `itself a refine of #${rec.refine}${rec.refine_missing ? ', since deleted' : ''}` : '',
  ].filter(Boolean).join(' · ');
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
    if (prefs.scale && refineCapped) refineSkipped = scale;
  }
  const kept = prefs.strength?.[loadedModel];
  els.refineStrength.value = pass && !foreign && pass.strength != null
    ? pass.strength
    : typeof kept === 'string' && SINGLE_STRENGTH.test(kept) ? kept : refineDefaultStrength();
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
  els.refineMoreNote.textContent = foreign
    ? `made on ${short(from.model_id)}: steps, CFG and mu are ${short(loadedModel)}'s`
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
  unmarkRefineFields();
  els.refinePreview.textContent = '';
  refineChecked = null;
  renderRefineCheck();
  els.refineTags.hidden = true;
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
  [...els.refineScales.children].forEach((button, i) => {
    const size = refineScaled(src.width, src.height, REFINE_SCALES[i]);
    const fits = refineFits(size);
    button.disabled = !fits;
    button.classList.toggle('active', fits && w === size[0] && h === size[1]);
    button.title = (fits ? `${size[0]} × ${size[1]}` : `${size[0]} × ${size[1]} is past what a refine pass renders`)
      + (REFINE_SCALES[i] === refineSkipped ? ` (last run at ${refineSkipped}×)` : '');
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
  els.refineSizeHint.textContent = [sized ? `${(w * h / 1e6).toFixed(1)} MP` : '', shape, ...notes]
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
  const start = refineDefaultStrength();
  const compare = compareStrengths().join(',');
  els.refineStrength.placeholder = `${start} or ${compare}`;
  els.refineCompare.title = `Write ${compare} into Strength: this checkpoint's ${start} and either side of it, `
    + 'run side by side as one sweep';
  els.refineSweepHint.textContent = `Several values (${compare}) run side by side and open as a sheet.`;

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
  els.refineNegNote.textContent = !negativeSupported
    ? '(this checkpoint takes no negative prompt)'
    : `(needs ${trueCfgSupported ? 'True CFG' : 'CFG'} > ${cfgThreshold})`;
  els.refineMu.disabled = !muSupported;
  els.refineMuField.classList.toggle('disabled', !muSupported);
  if (!muSupported) els.refineMu.value = '';
  els.refineTrueCfg.disabled = !trueCfgSupported;
  els.refineTrueCfgField.classList.toggle('disabled', !trueCfgSupported);
  if (!trueCfgSupported) els.refineTrueCfg.value = '';

  els.refineUnavailable.hidden = refineSupported;
  els.refineUnavailable.textContent = refineSupported ? ''
    : 'Refine is unavailable: as far as the server can tell, this checkpoint has no compatible '
      + 'image-to-image pipeline. A load answers for certain - the pass stays here until then.';
  syncRefineRun();
  // After the negative's activation has moved: the check's Show and Apply
  // follow it.
  syncRefineCheck();
}

function syncRefineRun() {
  const problem = !refineSupported ? 'This checkpoint has no refine pass'
    : refineSizeProblem(+els.refineWidth.value, +els.refineHeight.value) || refineProblem;
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
  els.refineStrength.value = compareStrengths().join(',');
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
    els.refinePreview.classList.toggle('warn', !!p.duplicates);
    els.refinePreview.textContent = plural(p.images, 'image')
      + ` · strength ${axisText(strengths)}${run}${keeps}`
      + (p.seconds ? ` · about ${humanTime(p.seconds)}` : '')
      + (p.duplicates ? ` · ${p.duplicates} would be identical to another` : '')
      + (sweep ? '\nOpens as a sheet when it is done.' : '');
  } catch (err) {
    if (epoch !== refineEpoch) return;
    refineProblem = err.message;
    els.refinePreview.classList.add('warn');
    els.refinePreview.textContent = err.message;
  }
  refinePreviewed = asked;
  syncRefineRun();
}

// Refine: the dialog closes and the job queues like Generate's - behind a
// running one if there is one, its progress in the panel, its images in the
// strip. A sweep opens as its sheet when done. Never the panel's Seed box:
// the dialog's seed is its own.
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
    refuseRefine(els.refineRun.title);
    return;
  }
  // As startJob refuses them, but here, while the dialog is still up to say so.
  if (expandingPrompt || critiquing) {
    refineFailed(`Wait for ${critiquing ? 'the critic' : 'prompt expansion'} to finish.`);
    return;
  }
  const { path, body, sweep } = refineRequest();
  if (!body.prompt) { refineFailed('Enter a prompt first.'); return; }
  rememberRefine();
  refineRan = true;
  els.refineDialog.close();
  await startJob(path, body, { sheet: sweep, panel: false });
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
// look at the picture - where the boxes differ from what openRefine filled.
function refineClosed() {
  const src = refineSource;
  const values = src && !refineRan ? refineValues() : null;
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
// server's do; a tag counts as read when the cut is past its end.
function refineCutLine(field, text, cut, reports) {
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
    () => selectRefineFrom(field, text, cut)));
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
  replaceBoxText(el, finding.suggestion);
  sizeRefinePrompt();
  refineChanged();
  scheduleRefineCheck(0);
  scheduleRefineTags(0);
}

// A tag checkpoint's unknown comma items, as the panel's chips mark them but
// as one line: "Not a known tag: smlie". Advisory - the pass still runs - and
// silent when the lookup is unavailable or fails, as the chips never call an
// unchecked tag unknown. Not on a hybrid checkpoint's single sentence.
function scheduleRefineTags(delay = 400) {
  clearTimeout(refineTagTimer);
  const epoch = ++refineTagEpoch;
  const text = els.refinePrompt.value;
  // Each tag once: a repeated unknown one took the line's dozen names.
  const tags = [...new Set(text.split(',').map((part) => part.trim()).filter(Boolean))].slice(0, 64);
  const sentence = tagHelper?.hybrid_prompt && !text.includes(',');
  if (!refineSource || !tagVocabularyEnabled || sentence || !tags.length) {
    els.refineTags.hidden = true;
    return;
  }
  refineTagTimer = setTimeout(async () => {
    let unknown = [];
    try {
      const body = await api('/api/tags/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags }),
      });
      if (body.available) unknown = body.results.filter((r) => r.valid === false);
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

// ------------------------------------------------- the source's passes

// The passes already made from this source, newest first - the gallery's own
// `refine:` search, on every checkpoint - so a pass is judged against what it
// was made from before another is run. One opens in the lightbox with the
// source ahead of it, ← and → stepping between them; the dialog closes for
// it, keeping what was typed for the next open (`refineClosed`).
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
    button.setAttribute('aria-label', `Open the pass ${what} in the lightbox`);
    const img = document.createElement('img');
    img.src = pass.thumb_url ?? pass.url;
    img.alt = '';
    img.loading = 'lazy';
    button.append(img);
    button.onclick = () => {
      els.refineDialog.close();
      openLightbox(pass, [rec, ...page.items]);
    };
    els.refinePassList.append(button);
  }
  els.refinePasses.hidden = false;
}
