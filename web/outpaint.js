// Auradream's page script: the Outpaint dialog - the extend controls,
// the frame, the presets, the preview, submission, close and Reuse.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ----------------------------------------------------------------- outpainting

// An outpaint extends a gallery image's canvas past its edges and paints the
// new area: an inpaint over a padded copy of the source, run through the same
// inpaint pipeline (so the gate is `inpaint_supported`), composited over the
// padded source so every pixel of the original outside the band is kept. The
// server does what is not presentation: it fills the new area (the source's
// edge pixels stretched outward and blurred), builds the mask (the new border
// and a feathered band reaching into the original) as the job is submitted,
// and decides every preset's numbers (`GET /api/images/{id}/outpaint-presets`).
// The page sends four sides and a band; it draws no fill and paints no mask.
// It is a dialog of its own rather than a mode of the Inpaint dialog, whose
// stage and tools are all about painting; what is the same is inpaint.js's
// (the start strength and its note, the unavailable words, the sampling
// boxes' rules, the prompt check's lines), called rather than copied.

// The record the dialog is open on, null while it is closed; the outpainted
// record a Reuse opened on, whose sides, band, strength and seed it starts
// from; and the server's presets answer for the source.
let outpaintSource = null;
let outpaintFrom = null;
let outpaintPresets = null;
// The four sides and the band as last committed - rounded to their grids -
// which the frame, the preview and the submit read; the boxes may hold a
// number being typed until it is committed (change, Enter, blur).
let outpaintSides = { left: 0, top: 0, right: 0, bottom: 0 };
let outpaintBandPx = 0;
// The fill the request names: this build's, or a Reuse's (refused by name
// at the preview where this build has no such fill).
let outpaintFill = '';
// What the status line said last, until the next change: a rounding, a
// preset, or a held preset's reason.
let outpaintSaid = null;
let outpaintReturn = null;
let outpaintReturnKeep = null;
let outpaintLora = null;
let outpaintRan = false;
let outpaintSending = false;
// What holds Outpaint while a submit is in flight.
const OUTPAINT_IN_FLIGHT = 'Wait for the last submit to be taken';
let outpaintWaiting = false;
let outpaintFilled = null;
let outpaintStrengthStandIn = null;
let outpaintOpenEpoch = 0;
let outpaintLoraDrops = 0;
let outpaintTimer;
let outpaintEpoch = 0;
let outpaintProblem = null;
let outpaintPreviewed = null;
let outpaintCheckTimer;
let outpaintCheckEpoch = 0;
let outpaintChecked = null;

// What a close that ran nothing left on an image - the sides, the band and
// the boxes - in page memory only, for the next open of that image on the
// same checkpoint, as the Inpaint dialog keeps its strokes.
const outpaintKept = new Map();
const OUTPAINT_KEPT = 8;
const OUTPAINT_GRID = 16;
const OUTPAINT_BAND_GRID = 8;
const OUTPAINT_SIDES = ['left', 'top', 'right', 'bottom'];

function outpaintBox(side) {
  return { left: els.outpaintLeft, top: els.outpaintTop, right: els.outpaintRight, bottom: els.outpaintBottom }[side];
}

// "+400/+0/+400/+0", as the server and the CLI say a padding.
function outpaintSidesText(spec) {
  return OUTPAINT_SIDES.map((side) => `+${spec[side] ?? 0}`).join('/');
}

// A stored spec's key (`fill:L,T,R,B@band`, the Reuse button's data
// attribute) back to the spec, or null.
function parseOutpaintKey(text) {
  const match = /^([^:]+):(\d+),(\d+),(\d+),(\d+)@(\d+)$/.exec(text ?? '');
  if (!match) return null;
  const [, fill, left, top, right, bottom, band] = match;
  return { fill, left: +left, top: +top, right: +right, bottom: +bottom, band: +band };
}

function outpaintKey(spec) {
  return `${spec.fill}:${spec.left},${spec.top},${spec.right},${spec.bottom}@${spec.band}`;
}

// ------------------------------------------------------------- the entry points

// The status's answer, as it lands (syncInpaintCapability calls it): an
// open dialog takes the library's strength where it still holds the stand-in,
// and says what holds it now.
function syncOutpaintCapability() {
  if (!outpaintSource) return;
  if (outpaintStrengthStandIn !== null && inpaintLibraryStrength() !== null) {
    if (libraryStrengthInto(els.outpaintStrength, outpaintStrengthStandIn, outpaintFilled)) scheduleOutpaintPreview(0);
    outpaintStrengthStandIn = null;
  }
  syncOutpaint();
}

// A result's Outpaint…: held, with the reason as its title, where the
// checkpoint has no inpaint pipeline - never disabled, so a press says why.
function syncOutpaintTool(button) {
  button.setAttribute('aria-disabled', String(!inpaintSupported));
  const title = inpaintSupported ? 'Outpaint: extend the canvas past this image\'s edges and paint the new area'
    : `Outpaint - ${inpaintNone(true)}`;
  button.title = title;
  button.setAttribute('aria-label', title);
  button.dataset.tip = inpaintSupported ? 'Outpaint…' : 'No outpaint here';
}

function syncLightboxOutpaint(rec) {
  const binned = !!rec.binned_at;
  els.lbOutpaint.setAttribute('aria-disabled', String(!inpaintSupported || binned));
  els.lbOutpaint.title = binned ? 'In the recycle bin: restore it to outpaint it'
    : inpaintSupported ? 'Outpaint this image: extend the canvas past its edges and paint the new area'
      : inpaintNone(true);
}

// The lightbox's Outpaint…: a held one says why in the lightbox's line and
// opens nothing.
function pressLightboxOutpaint() {
  const rec = lbList[lbIndex];
  if (!rec || refuseBinned(rec, 'outpaint')) return;
  if (!inpaintSupported) { showError(inpaintNone(true), true); return; }
  void openOutpaint(rec);
}

// A result's Outpaint…: on a checkpoint without one it says so under the
// gallery's head and over the tile, and opens nothing.
function outpaintFromCard(rec, div) {
  if (!inpaintSupported) { refuseInCard(div, inpaintNone(true), 'No outpaint here'); return; }
  void openOutpaint(rec);
}

// ------------------------------------------------------------- opening

// Opens the dialog on `rec`, the image to extend. `from` is an outpainted
// record to start from instead - Reuse on one - with its sides, band, fill,
// strength, seed, prompt and sampling. Otherwise the source's own prompt,
// sampling and seed, no side extended, the server's band and the starting
// strength - or, after a close that left sides or settings unsent on this
// image, those, which it says. The presets are the server's, asked first:
// they carry the source's size check and the guard's limits as well. A
// `restart` (Start over, from inside the open dialog) skips the support
// gate: the dialog resets and its own unavailable line says why it cannot
// run. Answers whether it opened.
async function openOutpaint(rec, from = null, restart = false) {
  if (!inpaintSupported && !from && !restart) {
    showError(inpaintNone(true), true);
    return false;
  }
  const opened = ++outpaintOpenEpoch;
  const fromLightbox = els.lightbox.open;
  let presets;
  try {
    presets = await api(`/api/images/${rec.id}/outpaint-presets`);
  } catch (err) {
    if (opened !== outpaintOpenEpoch) return false;
    const message = err.message ?? '';
    const gone = /^no gallery image with id/.test(message);
    // The server says a binned source is in the bin: this page's copy of
    // the record may be older than that.
    const kind = /recycle bin/.test(message) || rec.binned_at ? 'binned' : 'deleted';
    const said = gone ? sourceGoneWords(rec, kind, 'outpaint') : message;
    if (outpaintMovedOn(rec, fromLightbox, restart)) return false;
    showError(said, true, els.lightbox.open ? undefined : { slot: els.galleryError });
    return false;
  }
  if (opened !== outpaintOpenEpoch || outpaintMovedOn(rec, fromLightbox, restart)) return false;
  outpaintReturn = lightboxPlace();
  if (els.lightbox.open) closeLightbox();
  showError('');
  clearTimeout(outpaintTimer);
  clearTimeout(outpaintCheckTimer);
  outpaintEpoch++;
  outpaintCheckEpoch++;
  const kept = !from ? outpaintKept.get(rec.id) : null;
  const draft = kept && kept.model === loadedModel ? kept : null;
  outpaintKept.delete(rec.id);
  outpaintSource = rec;
  outpaintFrom = from;
  outpaintPresets = presets;
  outpaintRan = false;
  outpaintSaid = null;
  const start = from ?? rec;
  const foreign = modelsDiffer(start.model_id);
  els.outpaintTitle.textContent = from ? `Outpaint #${rec.id} as #${from.id} was made` : `Outpaint #${rec.id}`;
  writeOutpaintSourceLine(rec);
  const spec = from?.outpaint ?? null;
  els.outpaintReused.hidden = !from;
  els.outpaintReused.textContent = from
    ? `Reusing #${from.id}'s ${spec ? outpaintSidesText(spec) : 'padding'}, band ${spec?.band ?? presets.band} px`
      + `${from.strength != null ? ` at strength ${from.strength}` : ''}, seed ${from.seed}`
      + (foreign && from.strength != null ? ` - its strength was ${checkpointName(from.model_id)}'s` : '')
    : '';
  // The frame's picture: the thumbnail is plenty at the frame's size.
  const src = rec.thumb_url ?? rec.url;
  if (els.outpaintImage.getAttribute('src') !== src) els.outpaintImage.src = src;
  outpaintFill = spec?.fill ?? presets.fill;
  setOutpaintSides(spec ?? { left: 0, top: 0, right: 0, bottom: 0 }, spec?.band ?? presets.band);
  fillOutpaintShapes(presets);
  outpaintStrengthStandIn = null;
  const carried = from && !foreign && from.strength != null;
  els.outpaintStrength.value = String(carried ? from.strength : inpaintStartStrength());
  if (!carried && inpaintLibraryStrength() === null) outpaintStrengthStandIn = +els.outpaintStrength.value;
  els.outpaintVersions.value = '1';
  els.outpaintSeed.value = start.seed;
  els.outpaintPrompt.value = start.prompt;
  els.outpaintNegative.value = start.negative_prompt ?? '';
  const d = lastDefaults;
  els.outpaintSteps.value = foreign ? (d?.steps ?? 28) : start.steps;
  els.outpaintCfg.value = foreign ? (d?.guidance_scale ?? 3.5) : start.guidance_scale;
  els.outpaintMu.value = foreign ? (d?.mu ?? '') : (start.mu ?? '');
  els.outpaintTrueCfg.value = !foreign && trueCfgSupported && start.true_cfg != null ? start.true_cfg : '';
  const listed = [...els.lora.options].some((o) => o.value === start.lora);
  outpaintLora = !foreign && loraSupported && start.lora && listed
    ? { name: start.lora, scale: start.lora_scale ?? 1 } : null;
  els.outpaintCarried.textContent = outpaintLora
    ? `LoRA ${outpaintLora.name} at ${outpaintLora.scale}, as #${start.id} was made.`
    : start.lora ? `LoRA ${start.lora} is not applied: ${foreign ? 'it was for another checkpoint'
      : !loraSupported ? 'this installation cannot load adapters' : 'it is not in the adapter list now'}.` : '';
  els.outpaintMoreNote.textContent = foreign
    ? `made on ${checkpointName(start.model_id)}: steps and CFG are ${checkpointName(loadedModel)}'s`
    : `as #${start.id} was made`;
  els.outpaintMore.open = foreign;
  outpaintFilled = outpaintValues();
  if (draft) {
    applyOutpaintValues(draft.values);
    setOutpaintSides(draft.sides, draft.band);
  }
  els.outpaintDraft.hidden = !draft;
  els.outpaintDraftText.textContent = draft ? `Your unsent sides and settings for #${rec.id}, as you left them.` : '';
  outpaintProblem = null;
  outpaintPreviewed = null;
  outpaintFailed('');
  els.outpaintPreview.textContent = '';
  els.outpaintPreview.hidden = false;
  els.outpaintStrength.removeAttribute('aria-invalid');
  outpaintChecked = null;
  renderOutpaintCheck();
  // Scrolled to the top only as it opens: Start over fills it in place.
  if (!els.outpaintDialog.open) {
    els.outpaintDialog.showModal();
    els.outpaintDialog.scrollTop = 0;
  }
  syncOutpaint();
  // Focus on the first side box, what the dialog is opened to set.
  els.outpaintTop.focus({ preventScroll: true });
  scheduleOutpaintPreview(0);
  scheduleOutpaintCheck(0);
  return true;
}

// Whether the user moved on while the presets came: the lightbox it was
// pressed in was dismissed or moved to another image, or another dialog has
// opened since. A late answer then opens nothing over it.
function outpaintMovedOn(rec, fromLightbox, restart) {
  if (restart) return false;
  if (fromLightbox && (!els.lightbox.open || lbList[lbIndex]?.id !== rec.id)) return true;
  return [...document.querySelectorAll('dialog[open]')]
    .some((dialog) => dialog !== els.lightbox && dialog !== els.outpaintDialog);
}

function writeOutpaintSourceLine(rec, gone = null) {
  els.outpaintSourceLine.textContent = [
    `${rec.width}×${rec.height}`, `seed ${rec.seed}`, checkpointName(rec.model_id),
    rec.refine != null ? `itself ${rec.outpaint ? 'an outpaint' : rec.mask ? 'an inpaint' : 'a refine'} of #${rec.refine}` : '',
    gone ? `#${rec.id} (${gone === 'binned' ? 'in the recycle bin' : 'deleted'})` : '',
  ].filter(Boolean).join(' · ');
}

function outpaintSourceGone(message, source = outpaintSource) {
  const gone = sourceGoneKind(message, source);
  if (!gone) return message;
  if (outpaintSource === source) writeOutpaintSourceLine(source, gone);
  return sourceGoneWords(source, gone, 'outpaint');
}

// The four sides and the band, as committed, into the boxes.
function setOutpaintSides(sides, band) {
  for (const side of OUTPAINT_SIDES) {
    outpaintSides[side] = Math.max(0, sides[side] ?? 0);
    outpaintBox(side).value = String(outpaintSides[side]);
  }
  outpaintBandPx = Math.max(0, band);
  els.outpaintBand.value = String(outpaintBandPx);
}

// The To shape… list: every aspect the server answered a preset for, past
// the three that have buttons of their own.
function fillOutpaintShapes(presets) {
  const select = els.outpaintShape;
  select.replaceChildren(new Option('To shape…', ''));
  for (const preset of presets.presets) {
    if (preset.name === '+25%') continue;
    const option = new Option(preset.held ? `${preset.label} (held)` : `${preset.label} · ${preset.width} × ${preset.height}`,
      preset.name);
    option.title = preset.held ?? `${outpaintSidesText(preset)}: ${preset.width} × ${preset.height}`;
    select.append(option);
  }
  select.value = '';
}

function outpaintBoxes() {
  return {
    strength: els.outpaintStrength, versions: els.outpaintVersions, seed: els.outpaintSeed,
    prompt: els.outpaintPrompt, negative: els.outpaintNegative, steps: els.outpaintSteps,
    cfg: els.outpaintCfg, mu: els.outpaintMu, trueCfg: els.outpaintTrueCfg,
  };
}

function outpaintValues() {
  const values = Object.fromEntries(Object.entries(outpaintBoxes()).map(([key, el]) => [key, el.value]));
  return { ...values, more: els.outpaintMore.open, sides: outpaintKey({ fill: '', ...outpaintSides, band: outpaintBandPx }) };
}

function applyOutpaintValues(values) {
  for (const [key, el] of Object.entries(outpaintBoxes())) {
    if (typeof values[key] === 'string') el.value = values[key];
  }
  els.outpaintMore.open = !!values.more;
}

// Start over: no side extended, the server's band and the record's own
// settings again (a Reuse's as it ran).
function startOutpaintOver() {
  const rec = outpaintSource;
  if (!rec || outpaintSending) return;
  outpaintKept.delete(rec.id);
  outpaintReturnKeep = outpaintReturn;
  void openOutpaint(rec, outpaintFrom, true).then(() => {
    outpaintReturn = outpaintReturnKeep;
    outpaintReturnKeep = null;
  });
}

// Reset forgets what the dialog keeps in this page, with the panel.
function forgetOutpaint() {
  outpaintKept.clear();
}

// Images gone for good (forgetImages): what a close kept on them goes too.
function forgetOutpaintImages(ids) {
  for (const id of ids) outpaintKept.delete(id);
}

// A checkpoint switch: what was set for one checkpoint's sampling is not
// submitted on another's unchanged, as the Inpaint dialog's is not.
function outpaintCheckpointChanged() {
  outpaintKept.clear();
  outpaintOpenEpoch++;
  if (!outpaintSource) return;
  outpaintSource = null;
  outpaintReturn = null;
  els.outpaintDialog.close();
  showError('The checkpoint changed. Reopen Outpaint to use its new sampling settings.', true);
}

// ------------------------------------------------------------- the extend controls

// A side box's number committed (change, Enter, blur): rounded to the
// nearest multiple of 16, half up, as the server's grid wants it, and the
// status line says so where it moved. Never while typing - a number already
// on the grid is taken as it is typed (an arrow's step lands on one).
function commitOutpaintSide(side, { typing = false } = {}) {
  const el = outpaintBox(side);
  const text = el.value.trim();
  const typed = text === '' ? 0 : Number(text);
  if (typing && !(Number.isInteger(typed) && typed >= 0 && typed % OUTPAINT_GRID === 0)) return;
  const rounded = Number.isFinite(typed) ? Math.max(0, Math.floor((typed + OUTPAINT_GRID / 2) / OUTPAINT_GRID) * OUTPAINT_GRID) : 0;
  if (!typing && String(rounded) !== text) {
    outpaintSaid = { text: `${side} ${text === '' ? 'empty' : text} → ${rounded}, a multiple of ${OUTPAINT_GRID}`, warn: false };
    el.value = String(rounded);
  } else if (!typing || outpaintSides[side] !== rounded) {
    // A typed side that moves the canvas makes what a preset or a rounding
    // said stale: the status says what the boxes now hold.
    outpaintSaid = null;
  }
  if (outpaintSides[side] === rounded && !outpaintSaid) { syncOutpaint(); return; }
  outpaintSides[side] = rounded;
  outpaintChanged();
}

// The band likewise, to 8 px (the latent grid it ends on) and within the
// server's bound.
function commitOutpaintBand({ typing = false } = {}) {
  const el = els.outpaintBand;
  const text = el.value.trim();
  const typed = text === '' ? 0 : Number(text);
  const most = outpaintPresets?.max_band ?? 256;
  if (typing && !(Number.isInteger(typed) && typed >= 0 && typed <= most && typed % OUTPAINT_BAND_GRID === 0)) return;
  let rounded = Number.isFinite(typed) ? Math.max(0, Math.floor((typed + OUTPAINT_BAND_GRID / 2) / OUTPAINT_BAND_GRID) * OUTPAINT_BAND_GRID) : 0;
  const capped = rounded > most;
  rounded = Math.min(rounded, most);
  if (!typing && String(rounded) !== text) {
    outpaintSaid = {
      text: `band ${text === '' ? 'empty' : text} → ${rounded}, ${capped ? `the most it takes` : `a multiple of ${OUTPAINT_BAND_GRID}`}`,
      warn: false,
    };
    el.value = String(rounded);
  } else if (!typing || outpaintBandPx !== rounded) {
    outpaintSaid = null;
  }
  if (outpaintBandPx === rounded && !outpaintSaid) { syncOutpaint(); return; }
  outpaintBandPx = rounded;
  outpaintChanged();
}

// Whether a box holds a number not committed yet: a press commits it first.
function commitOutpaintBoxes() {
  for (const side of OUTPAINT_SIDES) {
    if (outpaintBox(side).value.trim() !== String(outpaintSides[side])) commitOutpaintSide(side);
  }
  if (els.outpaintBand.value.trim() !== String(outpaintBandPx)) commitOutpaintBand();
}

function outpaintPreset(name) {
  return outpaintPresets?.presets.find((p) => p.name === name) ?? null;
}

// A preset replaces the four sides - never adds to them - and leaves the
// band alone. A held one says why on the status line and changes nothing.
function applyOutpaintPreset(name) {
  if (!outpaintSource || !name) return;
  const preset = outpaintPreset(name);
  if (!preset) return;
  if (preset.held) {
    outpaintSaid = { text: `${preset.held}.`, warn: true };
    syncOutpaint();
    return;
  }
  setOutpaintSides(preset, outpaintBandPx);
  outpaintSaid = { text: `${preset.label}: ${outpaintSidesText(preset)}, ${preset.width} × ${preset.height}`, warn: false };
  outpaintChanged();
}

// The canvas the four sides make over the source.
function outpaintCanvasSize() {
  const src = outpaintSource;
  return [src.width + outpaintSides.left + outpaintSides.right, src.height + outpaintSides.top + outpaintSides.bottom];
}

// What of the source the band leaves unrepainted, each way.
function outpaintKeptSize() {
  const src = outpaintSource;
  const b = outpaintBandPx;
  return [
    src.width - b * (outpaintSides.left > 0) - b * (outpaintSides.right > 0),
    src.height - b * (outpaintSides.top > 0) - b * (outpaintSides.bottom > 0),
  ];
}

// The frame: the new canvas letterboxed into a box of a fixed size, the
// source at its place on it and the band's inner edge dashed, all placed in
// fractions of the canvas, so only the content scales as a number changes.
function drawOutpaintFrame() {
  const src = outpaintSource;
  const [cw, ch] = outpaintCanvasSize();
  const fw = els.outpaintFrame.clientWidth;
  const fh = els.outpaintFrame.clientHeight;
  const scale = Math.min(fw / cw, fh / ch);
  Object.assign(els.outpaintCanvas.style, { width: `${Math.round(cw * scale)}px`, height: `${Math.round(ch * scale)}px` });
  const pct = (n, of) => `${(100 * n) / of}%`;
  Object.assign(els.outpaintImage.style, {
    left: pct(outpaintSides.left, cw), top: pct(outpaintSides.top, ch),
    width: pct(src.width, cw), height: pct(src.height, ch),
  });
  const b = outpaintBandPx;
  const [kw, kh] = outpaintKeptSize();
  const edge = els.outpaintBandEdge;
  const extended = OUTPAINT_SIDES.some((side) => outpaintSides[side] > 0);
  edge.hidden = !extended || b === 0 || kw <= 0 || kh <= 0;
  if (!edge.hidden) {
    Object.assign(edge.style, {
      left: pct(outpaintSides.left + (outpaintSides.left > 0 ? b : 0), cw),
      top: pct(outpaintSides.top + (outpaintSides.top > 0 ? b : 0), ch),
      width: pct(kw, cw), height: pct(kh, ch),
    });
  }
  els.outpaintFrameLine.textContent = `${cw} × ${ch} · #${src.id} ${src.width} × ${src.height} at ${outpaintSidesText(outpaintSides)}`;
}

// The presets' buttons, held with the server's reason as their title.
function syncOutpaintPresets() {
  for (const button of [els.outpaintGrow, els.outpaintWide, els.outpaintTall]) {
    const preset = outpaintPreset(button.dataset.preset);
    const held = !preset ? 'Wait for the presets to load' : preset.held;
    button.setAttribute('aria-disabled', String(!!held));
    button.title = held || `Extend to ${preset.width} × ${preset.height}: ${outpaintSidesText(preset)} (replaces the four sides)`;
  }
  const picked = els.outpaintShape.value ? outpaintPreset(els.outpaintShape.value) : null;
  const held = !els.outpaintShape.value ? 'Pick a shape first' : !picked ? 'Wait for the presets to load' : picked.held;
  els.outpaintShapeApply.setAttribute('aria-disabled', String(!!held));
  els.outpaintShapeApply.title = held
    || `Extend to ${picked.width} × ${picked.height}: ${outpaintSidesText(picked)} (replaces the four sides)`;
}

// Apply beside To shape…: the picked aspect's preset. Held, it says why in
// the status line, as a held preset button does.
function applyOutpaintShape() {
  if (!outpaintSource) return;
  if (!els.outpaintShape.value) {
    outpaintSaid = { text: 'Pick a shape in the list first, then Apply.', warn: true };
    syncOutpaint();
    return;
  }
  applyOutpaintPreset(els.outpaintShape.value);
}

// What holds Outpaint because of the sides, the band or the source, or ''.
function outpaintExtendProblem() {
  const src = outpaintSource;
  const presets = outpaintPresets;
  if (!src || !presets) return 'Wait for the presets to load';
  if (presets.off_grid) return presets.off_grid;
  if (!OUTPAINT_SIDES.some((side) => outpaintSides[side] > 0)) return 'Extend at least one side';
  const [cw, ch] = outpaintCanvasSize();
  if (Math.max(cw, ch) > presets.max_side) {
    return `${cw} × ${ch} is past the ${presets.max_side}-pixel side a refine pass takes`;
  }
  if (cw * ch > presets.max_pixels) {
    return `${cw} × ${ch} is past the ${presets.max_pixels.toLocaleString('en-US')} pixels a refine pass takes`;
  }
  return outpaintBandProblem() ?? '';
}

// Whether the band leaves too little kept: the server's least kept side
// (`min_kept`), in its words - the short axis and the padded sides the band
// comes off - or null.
function outpaintBandProblem() {
  const src = outpaintSource;
  const least = outpaintPresets?.min_kept ?? 16;
  const [kw, kh] = outpaintKeptSize();
  if (!src || (kw >= least && kh >= least)) return null;
  const width = kw < least;
  const sides = (width ? ['left', 'right'] : ['top', 'bottom']).filter((side) => outpaintSides[side] > 0);
  return `A band of ${outpaintBandPx} px on the ${sides.join(' and ')} side${sides.length > 1 ? 's' : ''} `
    + `leaves ${Math.max(width ? kw : kh, 0)} px of #${src.id}'s ${width ? 'width' : 'height'} unrepainted; `
    + `at least ${least} must be kept, so narrow the band`;
}

// The status line: what the last commit or preset said, else what the
// padding repaints and keeps.
function syncOutpaintStatus() {
  const src = outpaintSource;
  const said = outpaintSaid;
  let text;
  let warn = false;
  if (said) {
    ({ text, warn } = said);
  } else if (outpaintPresets?.off_grid) {
    text = `${outpaintPresets.off_grid}.`;
    warn = true;
  } else if (!OUTPAINT_SIDES.some((side) => outpaintSides[side] > 0)) {
    text = 'Nothing extended yet: set a side, or pick a preset.';
  } else {
    const [kw, kh] = outpaintKeptSize();
    const problem = outpaintBandProblem();
    text = problem ? `${problem}.`
      : `The new area and a ${outpaintBandPx}-px band are repainted; ${kw} × ${kh} of #${src.id} is kept exactly.`;
    warn = !!problem;
  }
  if (els.outpaintStatus.textContent !== text) els.outpaintStatus.textContent = text;
  els.outpaintStatus.classList.toggle('warn', warn);
  els.outpaintStatus.title = text;
}

// ------------------------------------------------------------- the dialog

// Everything that follows the boxes and the server: the frame and its line,
// the presets, the seed line, the strength's note, the sampling boxes, the
// unavailable line, and whether Outpaint can run.
function syncOutpaint() {
  const src = outpaintSource;
  if (!src) return;
  drawOutpaintFrame();
  syncOutpaintPresets();
  syncOutpaintStatus();
  const versions = +els.outpaintVersions.value || 1;
  const text = els.outpaintStrength.value.trim();
  const sweep = text !== '' && !SINGLE_STRENGTH.test(text);
  const seedText = els.outpaintSeed.value.trim();
  const seed = seedText === '' ? null : +seedText;
  const seeds = seed === null
    ? (versions > 1 ? `${versions} seeds counting up from a random one` : 'a random seed')
    : versions > 1 ? `seeds ${seed}-${seed + versions - 1}` : `seed ${seed}`;
  els.outpaintSeedHint.textContent = `${seeds}${seed === src.seed ? ", from the source's" : ''}`
    + (sweep ? ', at every strength' : '');
  els.outpaintStrength.placeholder = String(inpaintStartStrength());
  els.outpaintStrengthNote.textContent = inpaintStrengthNote();
  // The server's words for its starting band (outpaint.band_note).
  els.outpaintBandNote.textContent = `px into #${src.id} from each extended side; ${outpaintPresets?.band_note ?? ''}`;
  els.outpaintBandNote.title = els.outpaintBandNote.textContent;
  syncPassSampling(outpaintSampling());
  els.outpaintUnavailable.hidden = inpaintSupported;
  els.outpaintUnavailable.textContent = inpaintSupported ? '' : inpaintUnavailableText(true);
  syncOutpaintRun();
  syncOutpaintCheck();
}

function outpaintSampling() {
  return {
    negative: els.outpaintNegative, negField: els.outpaintNegField, negNote: els.outpaintNegNote,
    cfg: els.outpaintCfg, mu: els.outpaintMu, muField: els.outpaintMuField,
    trueCfg: els.outpaintTrueCfg, trueCfgField: els.outpaintTrueCfgField,
  };
}

// Held, with the reason as its title: first what holds Generate, then no
// inpaint pipeline here, then the sides, band and source, then the
// preview's refusal (its words, an unknown fill's included).
function syncOutpaintRun() {
  if (!els.outpaintRun || !outpaintSource) return;
  const problem = submitHold() || (submitting ? OUTPAINT_IN_FLIGHT : '')
    || (!inpaintSupported ? inpaintNone(true).replace(/\.$/, '').split('. ')[0]
      : outpaintExtendProblem() || outpaintProblem);
  els.outpaintRun.setAttribute('aria-disabled', String(!!problem));
  els.outpaintRun.title = problem || 'Run the outpaint (Ctrl+Enter)';
}

function outpaintChanged() {
  els.outpaintStrength.removeAttribute('aria-invalid');
  els.outpaintPreview.hidden = false;
  outpaintFailed('');
  syncOutpaint();
  scheduleOutpaintPreview();
}

// Enter in a side or band box commits it; in any other one-line box it
// runs, as the Inpaint dialog's. ArrowUp and ArrowDown step a single
// Strength by 0.05 (0.01 with Shift).
function outpaintKeydown(event) {
  if (keyboardEventHeld(event) || keyboardCommandModifier(event)) return;
  const side = OUTPAINT_SIDES.find((s) => outpaintBox(s) === event.target);
  if (event.key === 'Enter' && !event.shiftKey && event.target === els.outpaintShape) {
    event.preventDefault();
    if (!event.repeat) applyOutpaintShape();
    return;
  }
  if (event.key === 'Enter' && !event.shiftKey && (side || event.target === els.outpaintBand)) {
    event.preventDefault();
    if (!event.repeat) {
      if (side) commitOutpaintSide(side); else commitOutpaintBand();
    }
    return;
  }
  if (event.key === 'Enter' && !event.shiftKey && event.target.matches('input:not([type="checkbox"]):not([type="range"])')) {
    event.preventDefault();
    if (!event.repeat) void runOutpaint();
    return;
  }
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && event.target === els.outpaintStrength) {
    const text = els.outpaintStrength.value.trim();
    if (!SINGLE_STRENGTH.test(text)) return;
    event.preventDefault();
    const step = (event.shiftKey ? 0.01 : 0.05) * (event.key === 'ArrowUp' ? 1 : -1);
    els.outpaintStrength.value = String(Math.min(1, Math.max(0.01, +(+text + step).toFixed(2))));
    outpaintChanged();
  }
}

function outpaintFailed(message) {
  els.outpaintError.hidden = !message;
  els.outpaintError.textContent = message || '';
}

// The request the dialog stands for: the canvas (the source's size plus the
// padding), the source by its id and the outpaint's sides, band and fill -
// never a mask, which the server builds - one strength as an ordinary
// request of `versions` images, several as a grid whose strength axis
// crosses that many seeds.
function outpaintRequest() {
  const num = (el) => (el.value.trim() === '' ? undefined : +el.value);
  const versions = +els.outpaintVersions.value || 1;
  const strength = els.outpaintStrength.value.trim();
  const single = strength === '' || SINGLE_STRENGTH.test(strength);
  const seed = els.outpaintSeed.value.trim();
  const [width, height] = outpaintCanvasSize();
  const base = {
    prompt: els.outpaintPrompt.value.trim(),
    negative_prompt: negativeSent(els.outpaintNegative),
    width,
    height,
    steps: num(els.outpaintSteps),
    guidance_scale: num(els.outpaintCfg),
    true_cfg: els.outpaintTrueCfg.disabled || els.outpaintTrueCfg.value.trim() === '' ? null : +els.outpaintTrueCfg.value,
    mu: els.outpaintMu.disabled || els.outpaintMu.value.trim() === '' ? null : +els.outpaintMu.value,
    lora: outpaintLora?.name ?? null,
    lora_scale: outpaintLora?.scale ?? 1,
    refine: outpaintSource.id,
    outpaint: { fill: outpaintFill, ...outpaintSides, band: outpaintBandPx },
    strength: single && strength !== '' ? +strength : inpaintStartStrength(),
    seed: seed === '' ? null : +seed,
    runs: 1,
  };
  if (single) {
    const body = { ...base, num_images: versions };
    return { path: '/api/generate', body, preview: { ...body, seeds: 1 }, sweep: false };
  }
  const body = { ...base, strength: undefined, num_images: 1, seeds: versions, strength_axis: strength };
  return { path: '/api/grid', body, preview: body, sweep: true };
}

function scheduleOutpaintPreview(delay = 250) {
  clearTimeout(outpaintTimer);
  outpaintTimer = setTimeout(previewOutpaint, delay);
}

// The server's reading of the outpaint, as the Inpaint dialog's: the image
// count, the strengths a sweep expands to, an estimate - rarely one, since a
// canvas of a source plus a padding is seldom a size anything has run at -
// and a refusal in the server's words: a size not the source's plus the
// padding, a band too wide, a fill this build has not, a source gone. Nothing
// is asked while the sides themselves hold Outpaint: it says why.
async function previewOutpaint() {
  if (!outpaintSource || !els.outpaintDialog.open) return;
  const epoch = ++outpaintEpoch;
  const { preview, sweep } = outpaintRequest();
  const asked = JSON.stringify(preview);
  els.outpaintPreview.classList.remove('warn');
  outpaintProblem = null;
  if (!inpaintSupported || outpaintExtendProblem()) {
    els.outpaintPreview.textContent = '';
    outpaintPreviewed = asked;
    syncOutpaintRun();
    return;
  }
  if (!preview.prompt) {
    outpaintProblem = 'Enter a prompt: the new area is denoised under one.';
    els.outpaintPreview.classList.add('warn');
    els.outpaintPreview.textContent = outpaintProblem;
    outpaintPreviewed = asked;
    syncOutpaintRun();
    return;
  }
  try {
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(preview),
    });
    if (epoch !== outpaintEpoch) return;
    const strengths = p.values.strength ?? [];
    const steps = p.values.steps?.[0];
    const run = !sweep && typeof steps === 'number' && strengths.length === 1
      ? ` · about ${Math.max(1, Math.round(steps * strengths[0]))} of ${steps} steps run` : '';
    // Where the pass starts at the canvas's size, as the Inpaint dialog's.
    const noise = !sweep && strengths.length === 1 && typeof p.start_sigma === 'number'
      ? ` · starts at ${Math.round(p.start_sigma * 100)}% noise` : '';
    els.outpaintPreview.classList.toggle('warn', !!p.duplicates);
    els.outpaintPreview.textContent = plural(p.images, 'image')
      + ` · ${preview.width} × ${preview.height}`
      + ` · strength ${axisText(strengths)}${noise}${run}`
      + (p.seconds ? ` · about ${humanTime(p.seconds)}` : ` · no estimate: nothing has run at ${preview.width} × ${preview.height} here`)
      + (p.duplicates ? ` · ${p.duplicates} would be identical to another` : '')
      + (sweep ? '\nOpens as a sheet when it is done.' : '');
  } catch (err) {
    if (epoch !== outpaintEpoch) return;
    if (await dropOutpaintLora(err.message)) return previewOutpaint();
    if (epoch !== outpaintEpoch) return;
    outpaintProblem = outpaintSourceGone(err.message);
    els.outpaintPreview.classList.add('warn');
    els.outpaintPreview.textContent = outpaintProblem;
  }
  outpaintPreviewed = asked;
  syncOutpaintRun();
}

// An adapter removed or renamed since the dialog opened, named by a refusal:
// taken out of this outpaint and said, as the Inpaint dialog's. Answers
// whether it went.
async function dropOutpaintLora(message, source = outpaintSource) {
  const name = outpaintLora?.name;
  if (!name || !message?.includes(name)) return false;
  const body = await refreshLoras();
  if (!body || outpaintSource !== source || outpaintLora?.name !== name) return false;
  const entry = (body.loras ?? []).find((e) => e.name === name);
  if (entry && !entry.detail) return false;
  outpaintLora = null;
  outpaintLoraDrops++;
  els.outpaintCarried.textContent = `LoRA ${name} is not applied: it is not in the adapter list now.`;
  outpaintFailed(`${name} ${entry ? 'no longer fits this checkpoint' : 'is no longer in the LoRA folder'}, `
    + 'so it was taken out of this outpaint; Outpaint runs it without the adapter.');
  return true;
}

// Outpaint: the job queues like the Inpaint dialog's, and the dialog closes
// once the server has taken it. A sweep opens as its sheet when done.
async function runOutpaint() {
  if (!outpaintSource || !els.outpaintDialog.open || outpaintWaiting) return;
  // A submit in flight - this dialog's or the panel's - holds Outpaint; a
  // press says so rather than doing nothing.
  if (submitting) { outpaintFailed(`${OUTPAINT_IN_FLIGHT}.`); return; }
  commitOutpaintBoxes();
  const drops = outpaintLoraDrops;
  while (outpaintPreviewed !== JSON.stringify(outpaintRequest().preview)) {
    clearTimeout(outpaintTimer);
    outpaintWaiting = true;
    try { await previewOutpaint(); } finally { outpaintWaiting = false; }
    if (!outpaintSource || !els.outpaintDialog.open || submitting || outpaintLoraDrops !== drops) return;
  }
  if (els.outpaintRun.getAttribute('aria-disabled') === 'true') {
    const held = submitHold();
    if (held) outpaintFailed(`${held}.`);
    else refuseOutpaint(els.outpaintRun.title);
    if (serverOffline) void refreshStatus();
    return;
  }
  const { path, body, sweep } = outpaintRequest();
  const source = outpaintSource;
  const strength = els.outpaintStrength.value.trim() || String(inpaintStartStrength());
  outpaintSending = true;
  let accepted = false;
  const settle = () => {
    outpaintSending = false;
    els.outpaintRun.textContent = busy ? 'Outpaint (queue)' : 'Outpaint';
  };
  const sent = startJob(path, body, {
    sheet: sweep,
    panel: false,
    slot: els.outpaintError,
    placeholder: {
      src: source.url ?? source.thumb_url,
      width: body.width,
      height: body.height,
      label: `Outpainting #${source.id} ${outpaintSidesText(body.outpaint)} · strength ${strength}`,
    },
    onAccepted: () => {
      accepted = true;
      settle();
      if (outpaintSource !== source || !els.outpaintDialog.open) return;
      outpaintRan = true;
      els.outpaintDialog.close();
      requestAnimationFrame(() => {
        if (document.activeElement && document.activeElement !== document.body) return;
        handJobFocus(els.newImage.hidden ? els.generate : els.newImage);
      });
    },
  });
  if (outpaintSending) els.outpaintRun.textContent = 'Sending…';
  await sent;
  if (accepted) return;
  settle();
  const refused = els.outpaintError.textContent;
  const said = outpaintSourceGone(refused, source);
  if (said !== refused) outpaintFailed(said);
  else if (await dropOutpaintLora(refused, source)) scheduleOutpaintPreview(0);
  if (!els.outpaintDialog.open && els.outpaintError.textContent) showError(els.outpaintError.textContent);
}

// A press on a held Outpaint says why, in the preview's place and at its
// size, so the button does not move under the pointer; Strength is marked
// where the sentence is about it.
function refuseOutpaint(reason) {
  if (inpaintSupported && reason === outpaintProblem && els.outpaintPreview.textContent === reason) {
    els.outpaintPreview.textContent = '';
    els.outpaintPreview.classList.remove('warn');
    els.outpaintPreview.hidden = true;
    outpaintFailed(reason);
    if (/strength/i.test(reason)) {
      els.outpaintStrength.setAttribute('aria-invalid', 'true');
      els.outpaintStrength.focus({ preventScroll: true });
    }
    return;
  }
  // An empty preview line gives its place to the refusal.
  if (!els.outpaintPreview.textContent) els.outpaintPreview.hidden = true;
  outpaintFailed(/[.!?]$/.test(reason) ? reason : `${reason}.`);
}

function closeOutpaint() {
  if (!outpaintSending) els.outpaintDialog.close();
}

function outpaintCancelEvent(event) {
  if (outpaintSending) event.preventDefault();
}

// Closed by Outpaint, Cancel, the cross or Escape. A close that ran nothing
// keeps the sides, the band and the boxes for the next open of the same
// image on this checkpoint - in page memory - and, opened from the
// lightbox, gives the lightbox back where it was; where it ran, focus goes
// to the source's card.
function outpaintClosed() {
  const src = outpaintSource;
  const ran = outpaintRan || outpaintSending;
  if (src && !ran) {
    const values = outpaintValues();
    const edited = [...Object.keys(outpaintBoxes()), 'sides'].some((key) => values[key] !== outpaintFilled?.[key]);
    if (edited) {
      outpaintKept.set(src.id, { model: loadedModel, values, sides: { ...outpaintSides }, band: outpaintBandPx });
      while (outpaintKept.size > OUTPAINT_KEPT) outpaintKept.delete(outpaintKept.keys().next().value);
    }
  }
  outpaintSource = null;
  outpaintFrom = null;
  outpaintLora = null;
  outpaintPresets = null;
  outpaintEpoch++;
  outpaintCheckEpoch++;
  outpaintOpenEpoch++;
  clearTimeout(outpaintTimer);
  clearTimeout(outpaintCheckTimer);
  const back = outpaintReturn;
  outpaintReturn = null;
  if (!back || els.lightbox.open) return;
  if (!ran) { returnToLightbox(back); return; }
  const card = [...document.querySelectorAll(`#gallery .card[data-id="${back.rec.id}"] img`)].find(isShown);
  handJobFocus(card ?? els.generate);
}

// ------------------------------------------------------------- Reuse

// An outpainted record holds its source's id and its spec: the source asked
// for, so the dialog has its picture and size, and opened with the sides,
// band and fill. A source that has gone cannot be outpainted again - the
// prompt and settings go into the panel instead, at the panel's own size;
// one in the recycle bin is said as the Inpaint dialog says it.
async function reuseOutpaint(rec) {
  let source;
  try {
    // The bin's rows too, so a binned source is offered a Restore rather
    // than said to be gone for good.
    source = await api(`/api/images/${rec.refine}?binned=true`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) { showError(err.message); return; }
    reuseIntoPanel(rec, false,
      `#${rec.refine}, the source this was outpainted from, is no longer in the gallery, `
      + 'so the outpaint cannot be made again. The size is left as it was.');
    return;
  }
  if (source.binned_at) { showError(sourceGoneWords(source, 'binned', 'outpaint'), true); return; }
  await openOutpaint(source, rec);
}

// ------------------------------------------------------------- the prompt check

function scheduleOutpaintCheck(delay = 400) {
  clearTimeout(outpaintCheckTimer);
  outpaintCheckTimer = setTimeout(checkOutpaintPrompt, delay);
}

// The prompt check on the outpaint's own prompt, as the Inpaint dialog's:
// cold, off the image queue, advice only - its failures are silent.
async function checkOutpaintPrompt() {
  if (!outpaintSource || !els.outpaintDialog.open) return;
  const epoch = ++outpaintCheckEpoch;
  const { body } = outpaintRequest();
  if (!body.prompt) { outpaintChecked = null; renderOutpaintCheck(); return; }
  const asked = outpaintCheckAsked(body);
  let result = null;
  try {
    result = await api('/api/prompt/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...body, outpaint: undefined, strength: undefined, strength_axis: undefined, seeds: undefined,
      }),
    });
  } catch { /* advice only */ }
  if (epoch !== outpaintCheckEpoch) return;
  outpaintChecked = result ? { ...asked, body: result } : null;
  renderOutpaintCheck();
}

function outpaintCheckAsked(body) {
  return { prompt: body.prompt, negative: body.negative_prompt ?? '', negativeOff: els.outpaintNegative.disabled };
}

function renderOutpaintCheck() {
  renderPassCheck({
    box: els.outpaintCheck, said: els.outpaintCheckSaid, checked: outpaintChecked,
    select: selectOutpaintFrom, apply: applyOutpaintFinding,
  });
  syncOutpaintCheck();
}

function syncOutpaintCheck() {
  if (!outpaintSource) return;
  const { body } = outpaintRequest();
  syncPassCheck(els.outpaintCheck, outpaintChecked, outpaintCheckAsked(body), els.outpaintNegative.disabled);
}

function selectOutpaintFrom(field, text, cut) {
  selectPassFrom(els.outpaintPrompt, els.outpaintNegative, els.outpaintMore, field, text, cut);
}

function applyOutpaintFinding(finding) {
  const el = finding.field === 'negative_prompt' ? els.outpaintNegative : els.outpaintPrompt;
  if (el.disabled) return;
  els.outpaintMore.open = true;
  replaceBoxText(el, finding.suggestion);
  outpaintChanged();
  scheduleOutpaintCheck(0);
}
