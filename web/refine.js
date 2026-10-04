// Local Diffusion's page script: the Refine dialog. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ------------------------------------------------------------ the refine pass

// A pass used to be a mode of the panel: the ⤢ button rewrote the prompt,
// seed, steps and size, and from then on Generate, Batch, Runs and the grid
// all ran passes until the source chip was removed - and a pass inherited a
// Seed, Batch and Runs meant for direct renders. The dialog keeps the two
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

let refineTimer;
let refineEpoch = 0;
// What the last preview was refused with, or null: what Refine would be
// refused with, so the button says so before the click rather than after the
// dialog has closed.
let refineProblem = null;

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

// The source's size times `scale`, or the largest smaller factor that fits:
// a 4 MP source cannot go to 1.5x, and opening at a size Refine refuses would
// only be a red line to correct.
function fittingRefineSize(rec, scale) {
  for (const factor of [scale, ...REFINE_SCALES.filter((s) => s < scale).reverse()]) {
    const size = refineScaled(rec.width, rec.height, factor);
    if (refineFits(size)) return size;
  }
  return refineScaled(rec.width, rec.height, 1);
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
// time, or the largest that fits), the versions last run and the strength
// last run on this checkpoint, else its profile's. Answers whether it opened.
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
  refineSource = rec;
  const prefs = refinePrefs();
  const from = pass ?? rec;
  // An image from another checkpoint refines like any other - the pass starts
  // from pixels - but its steps, CFG, mu and strength are that checkpoint's,
  // so this one's profile stands in for them.
  const foreign = modelsDiffer(from.model_id);
  const short = (id) => (id || '').split('/').pop();
  els.refineTitle.textContent = `Refine image ${rec.id}`;
  els.refineThumb.src = rec.thumb_url ?? rec.url;
  els.refineSourceLine.textContent = [
    `${rec.width}×${rec.height}`, `seed ${rec.seed}`, short(rec.model_id),
    rec.refine != null ? `itself a refine of image ${rec.refine}` : '',
  ].filter(Boolean).join(' · ');
  els.refinePromptLine.textContent = rec.prompt;
  // Rebuilt by syncRefine for this source's size.
  els.refineScales.textContent = '';
  [els.refineWidth.value, els.refineHeight.value] = pass
    ? [pass.width, pass.height]
    : fittingRefineSize(rec, +prefs.scale || REFINE_SCALE);
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
    ? `LoRA ${refineLora.name} at ${refineLora.scale}, as image ${from.id} was made.`
    : from.lora
      ? `LoRA ${from.lora} is not applied: ${foreign ? 'it was for another checkpoint'
        : !loraSupported ? 'this installation cannot load adapters'
          : 'it is not in the adapter list now'}.`
      : '';
  els.refineMoreNote.textContent = foreign
    ? `made on ${short(from.model_id)}: steps, CFG and mu are ${short(loadedModel)}'s`
    : `as image ${from.id} was made`;
  // Open where they could not carry over, so the substitution is seen.
  els.refineMore.open = foreign;
  refineProblem = null;
  refineFailed('');
  els.refinePreview.textContent = '';
  syncRefine();
  if (!els.refineDialog.open) els.refineDialog.showModal();
  scheduleRefinePreview(0);
  return true;
}

// Everything the dialog shows that follows its boxes and the server: the
// scale buttons, the size line, the seed line, the negative prompt's
// activation, Mu and True CFG where the checkpoint has them, and whether
// Refine can run. Status calls it too, since those answers can change.
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
    button.title = fits ? `${size[0]} × ${size[1]}` : `${size[0]} × ${size[1]} is past what a refine pass renders`;
    button.setAttribute('aria-label', `${REFINE_SCALES[i]} times the source's size, ${button.title}`);
  });

  const problem = refineSizeProblem(w, h);
  const notes = problem ? [problem] : sizeIsTight(w, h) ? ['may run tight on 16GB VRAM'] : [];
  const fx = w / src.width;
  const fy = h / src.height;
  const shape = Math.abs(fx - fy) <= 0.02 * Math.max(fx, fy)
    ? `${+((fx + fy) / 2).toFixed(2)}× the source` : 'the source cropped to this shape';
  els.refineSizeHint.textContent = [w > 0 && h > 0 ? `${(w * h / 1e6).toFixed(1)} MP` : '', shape, ...notes]
    .filter(Boolean).join(' — ');
  els.refineSizeHint.classList.toggle('warn', notes.length > 0);

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
  syncRefine();
  scheduleRefinePreview(0);
}

// Every box: the lines follow at once, the server's preview a moment later.
function refineChanged() {
  refineFailed('');
  syncRefine();
  scheduleRefinePreview();
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
  els.refinePreview.classList.remove('warn');
  refineProblem = null;
  if (!refineSupported || refineSizeProblem(preview.width, preview.height)) {
    els.refinePreview.textContent = '';
    syncRefineRun();
    return;
  }
  if (!preview.prompt) {
    refineProblem = 'Enter a prompt: the pass is denoised under one.';
    els.refinePreview.classList.add('warn');
    els.refinePreview.textContent = refineProblem;
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
    els.refinePreview.classList.toggle('warn', !!p.duplicates);
    els.refinePreview.textContent = plural(p.images, 'image')
      + ` · strength ${axisText(strengths)}${run}`
      + (p.seconds ? ` · about ${humanTime(p.seconds)}` : '')
      + (p.duplicates ? ` · ${p.duplicates} would be identical to another` : '')
      + (sweep ? '\nOpens as a sheet when it is done.' : '');
  } catch (err) {
    if (epoch !== refineEpoch) return;
    refineProblem = err.message;
    els.refinePreview.classList.add('warn');
    els.refinePreview.textContent = err.message;
  }
  syncRefineRun();
}

// Refine: the dialog closes and the job queues like Generate's - behind a
// running one if there is one, its progress in the panel, its images in the
// strip. A sweep opens as its sheet when done. Never the panel's Seed box:
// the dialog's seed is its own.
async function runRefine() {
  if (!refineSource || !els.refineDialog.open || submitting) return;
  if (els.refineRun.getAttribute('aria-disabled') === 'true') {
    refineFailed(els.refineRun.title);
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
  els.refineDialog.close();
  await startJob(path, body, { sheet: sweep, panel: false });
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
  if (scale) prefs.scale = scale;
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

// Closed by Refine, Cancel, the cross or Escape: nothing is open any more,
// and a preview still in flight lands on nothing.
function refineClosed() {
  refineSource = null;
  refineLora = null;
  refineEpoch++;
  clearTimeout(refineTimer);
}
