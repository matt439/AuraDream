// Local Diffusion's page script: size presets, the CFG note and the conditioning controls. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// -------------------------------------------------------------------- sizes

function buildPresets() {
  // A portrait framing and its landscape inverse are one segmented box on the
  // strip: the table lists them adjacently, so two neighbours that are each
  // other's inverse share a container and anything else (1:1) stands alone.
  // Buttons stay in ASPECTS order in the document, which syncSize relies on.
  for (let i = 0; i < ASPECTS.length; i += 1) {
    const aspect = ASPECTS[i], next = ASPECTS[i + 1];
    const button = presetButton(aspect, () => choose(aspect, pickedTier));
    if (next && next === oppositeAspect(aspect)) {
      const pair = document.createElement('div');
      pair.className = 'pair';
      pair.append(button, presetButton(next, () => choose(next, pickedTier)));
      els.presets.appendChild(pair);
      i += 1;
    } else {
      els.presets.appendChild(button);
    }
  }
  TIERS.forEach((tier) => {
    els.tiers.appendChild(presetButton(`${tier} MP`, () => choose(pickedAspect, tier)));
  });
}

function presetButton(label, onClick) {
  const b = document.createElement('button');
  b.textContent = label;
  b.onclick = onClick;
  return b;
}

function choose(aspect, tier) {
  [els.width.value, els.height.value] = SIZES[aspect][tier];
  syncSize();
  // Assigning .value fires no `input`, so the grid preview - whose duration
  // estimate is priced per size - would keep the old numbers, and the
  // remembered panel would keep the old size.
  schedulePreview();
  persistPanel();
}

function oppositeAspect(aspect) {
  const [wide, high] = aspect.split(':');
  const opposite = `${high}:${wide}`;
  return opposite !== aspect && ASPECTS.includes(opposite) ? opposite : null;
}

function swapOrientation() {
  const opposite = oppositeAspect(pickedAspect);
  if (opposite) choose(opposite, pickedTier);
}

function syncOrientation() {
  const opposite = oppositeAspect(pickedAspect);
  els.orientation.disabled = !opposite;
  if (!opposite) {
    els.orientation.textContent = 'Orientation';
    els.orientation.title = 'Square has no portrait or landscape orientation';
    els.orientation.setAttribute('aria-label', els.orientation.title);
    return;
  }
  const [width, height] = SIZES[opposite][pickedTier];
  const target = width > height ? 'Landscape' : 'Portrait';
  els.orientation.textContent = target;
  els.orientation.title = `Switch ${pickedAspect} to ${target.toLowerCase()} (${opposite}, ${width} × ${height})`;
  els.orientation.setAttribute('aria-label', els.orientation.title);
}

function syncSize() {
  const w = +els.width.value, h = +els.height.value;
  syncRerender();
  // Recover which preset this is, so a reused or hand-typed size lights the
  // buttons up too, and the next click off it starts from the right place.
  const match = ASPECTS.flatMap((a) => TIERS.map((t) => [a, t]))
    .find(([a, t]) => SIZES[a][t][0] === w && SIZES[a][t][1] === h);
  if (match) [pickedAspect, pickedTier] = match;
  syncOrientation();

  [...els.presets.querySelectorAll('button')].forEach((b, i) => {
    const aspect = ASPECTS[i];
    const guided = presetIsInGuide(aspect, pickedTier);
    b.classList.toggle('active', !!match && aspect === match[0]);
    b.classList.toggle('recommended', guided);
    describePreset(b, aspect, pickedTier, guided);
  });
  [...els.tiers.children].forEach((b, i) => {
    const tier = TIERS[i];
    const guided = presetIsInGuide(pickedAspect, tier);
    b.classList.toggle('active', !!match && tier === match[1]);
    b.classList.toggle('recommended', guided);
    // Spell out what the tier means for the framing currently in play.
    describePreset(b, pickedAspect, tier, guided);
  });

  const mp = (w * h) / 1e6;
  const notes = [];
  // What the line says that is not a warning: a refine pass, and how the
  // size relates to its source.
  const info = [];
  const namedAspect = match?.[0] || null;
  const refining = refineSupported && !!refineSource;
  if (refining) {
    // The documented range and aspects are about a direct render, where a
    // size past them composes a new picture badly. A refine pass starts from
    // the picture - going past the range this way is the range's own advice.
    if (refineSource.width) {
      const fx = w / refineSource.width, fy = h / refineSource.height;
      info.push(Math.abs(fx - fy) <= 0.02 * Math.max(fx, fy)
        ? `refining image ${refineSource.id} at ${+((fx + fy) / 2).toFixed(2)}× its size`
        : `refining image ${refineSource.id}, cropped to this shape`);
    } else {
      info.push(`refining image ${refineSource.id}`);
    }
  } else {
    if (sizeAspects && !sizeMatchesGuideAspect(w, h, namedAspect)) {
      notes.push(namedAspect
        ? `${namedAspect} is not a recommended aspect for this checkpoint`
        : 'this is not a recommended aspect for this checkpoint');
    }
    // The checkpoint's own range first: a 4 MP request of a model its authors
    // validated at 1 MP is the more likely disappointment. The presets land
    // within a few percent of their tier, so a tenth either way is still "at".
    if (sizeRange && mp > sizeRange[1] * 1.1) {
      notes.push(`above the ${fmtMp(sizeRange[1])} MP this checkpoint is ${sizeRangeWord} for`);
    } else if (sizeRange && mp < sizeRange[0] * 0.9) {
      notes.push(`below the ${fmtMp(sizeRange[0])} MP this checkpoint is ${sizeRangeWord} for`);
    }
  }
  // 4 MP on a 16GB card is the edge of what fits once the DiT is resident.
  const tight = w * h > 1600 * 1600;
  if (tight) notes.push('may run tight on 16GB VRAM');
  els.sizeHint.textContent = `${mp.toFixed(1)} MP` + [...info, ...notes].map((n) => ` — ${n}`).join('');
  els.sizeHint.classList.toggle('warn', notes.length > 0);
  syncRefineScales();
}

function sizeMatchesGuideAspect(width, height, namedAspect) {
  if (!sizeAspects) return true;
  if (namedAspect) return sizeAspects.includes(namedAspect);
  return guideResolutions?.some(([wide, high]) => wide === width && high === height) || false;
}

// The buttons and the size line use one definition of "within range". Each
// strip previews the full preset that clicking it would choose: aspects at the
// current tier, and tiers at the current aspect.
function presetIsInGuide(aspect, tier) {
  if (!sizeRange) return false;
  if (sizeAspects && !sizeAspects.includes(aspect)) return false;
  const [w, h] = SIZES[aspect][tier];
  const mp = (w * h) / 1e6;
  return mp >= sizeRange[0] * 0.9 && mp <= sizeRange[1] * 1.1;
}

function describePreset(button, aspect, tier, guided) {
  const dimensions = SIZES[aspect][tier].join(' × ');
  const note = guided
    ? (sizeAspects
      ? ', a recommended aspect and size for this checkpoint'
      : `, within the size range this checkpoint is ${sizeRangeWord} for`)
    : '';
  button.title = dimensions + note;
  button.setAttribute('aria-label', button.textContent + note);
}

function fmtMp(mp) { return Number.isInteger(mp) ? String(mp) : mp.toFixed(1); }

// The highest CFG the open grid's axis reaches, from the last preview; null when
// the grid section is closed or the axis is blank. A grid decides guidance per
// cell, so the negative prompt has to be offered when *any* cell would apply
// it, not only when the panel's single value would.
let gridMaxCfg = null;

function syncCfg() {
  // The True CFG box is live only where it is a knob of its own; elsewhere
  // it is greyed with its badge saying so, and the CFG box decides.
  els.trueCfg.disabled = !trueCfgSupported;
  els.trueCfgField.classList.toggle('disabled', !trueCfgSupported);
  const cfgReach = gridMaxCfg === null ? +els.cfg.value : Math.max(+els.cfg.value, gridMaxCfg);
  const trueCfg = els.trueCfg.value.trim() === '' ? 0 : +els.trueCfg.value;
  const reach = trueCfgSupported ? trueCfg : cfgReach;
  const knob = trueCfgSupported ? 'True CFG' : 'CFG';
  const off = !negativeSupported || reach <= cfgThreshold;
  els.negField.classList.toggle('disabled', off);
  els.negative.disabled = off;
  els.negNote.textContent = !negativeSupported
    ? '(this checkpoint takes no negative prompt)'
    : !trueCfgSupported && gridMaxCfg !== null && +els.cfg.value <= cfgThreshold && reach > cfgThreshold
      ? `(applies to the grid cells above CFG ${cfgThreshold} — ~2× slower there)`
      : `(needs ${knob} > ${cfgThreshold} — ~2× slower)`;
  // A negative prompt greyed or woken here stops or starts running, and the
  // prompt check read it as it ran then.
  syncAnalysisStale();
}

// The conditioning controls, greyed for the checkpoint (references, the refine
// pass) or for the installation (adapters) rather than for a value - so a dead
// box always has a reason, and the reason is in the badge beside it. The
// strength boxes are dead without a source the way the weight boxes are
// without an adapter.
function syncConditioning() {
  els.referenceField.classList.toggle('disabled', !referenceSupported);
  els.lora.disabled = !loraSupported;
  els.loraScale.disabled = !loraSupported || !els.lora.value;
  els.loraField.classList.toggle('disabled', !loraSupported);
  els.gridLoraScale.disabled = !loraSupported || !els.lora.value;
  els.gridLoraField.classList.toggle('disabled', els.gridLoraScale.disabled);
  const refining = refineSupported && !!refineSource;
  els.refineField.classList.toggle('disabled', !refineSupported);
  els.strength.disabled = !refining;
  els.gridStrength.disabled = !refining;
  els.gridStrengthField.classList.toggle('disabled', !refining);
  renderReferences();
  renderRefine();
}

// The chips: one per selected image, in order, each removable. Rendered from
// `referenceIds` rather than from the DOM, because Reuse and the gallery's own
// button both write the list.
function renderReferences() {
  els.referenceList.textContent = '';
  els.referenceNote.textContent = !referenceSupported
    ? '(this checkpoint takes none)'
    : referenceIds.length
      ? `(${referenceIds.length}, each lengthening the sequence)`
      : '(add one with the ❐ button on a gallery image)';
  for (const id of referenceIds) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const img = document.createElement('img');
    img.src = `/api/images/${id}/thumb`;
    img.alt = '';
    const drop = document.createElement('button');
    drop.type = 'button';
    drop.className = 'ghost';
    drop.textContent = '×';
    drop.title = `Remove image ${id}`;
    drop.setAttribute('aria-label', `Remove reference image ${id}`);
    drop.onclick = () => setReferences(referenceIds.filter((other) => other !== id));
    chip.append(img, drop);
    els.referenceList.append(chip);
  }
}

function setReferences(ids) {
  referenceIds = ids.slice(0, 4);
  renderReferences();
  schedulePreview();
  persistPanel();
  // The prompt check counted them: klein's guide asks what each one is for.
  syncAnalysisStale();
}

// ----------------------------------------------------------- the refine pass

// The largest size the server takes: its `max_side` and `max_pixels`, which
// backend/config.py reads off the preset table this file's copy is held to.
const MAX_SIDE = Math.max(...ASPECTS.flatMap((a) => TIERS.flatMap((t) => SIZES[a][t])));
const MAX_PIXELS = Math.max(...ASPECTS.flatMap((a) => TIERS.map((t) => SIZES[a][t][0] * SIZES[a][t][1])));

// A source's size times a factor, each side on the 16-pixel grid, half
// rounding up: backend/refine.py's `scaled`, so this button and the CLI's
// --scale land on the same size for the same image.
function refineScaled(width, height, factor) {
  const snap = (side) => Math.max(16, Math.floor(side * factor / 16 + 0.5) * 16);
  return [snap(width), snap(height)];
}

const refineFits = ([w, h]) =>
  w >= 256 && h >= 256 && w <= MAX_SIDE && h <= MAX_SIDE && w * h <= MAX_PIXELS;

// The source chip and the note beside the caption. Rendered from
// `refineSource` rather than from the DOM, like the reference chips: the
// card's button, the lightbox's, Reuse, Reset and a restore all write it.
function renderRefine() {
  const source = refineSource;
  els.refineSource.textContent = '';
  els.refineNote.textContent = !refineSupported
    ? '(this checkpoint has no refine pass)'
    : !source
      ? '(pick an image with its ⤢ button)'
      : source.width ? `(image ${source.id}, ${source.width}×${source.height})` : `(image ${source.id})`;
  if (source) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const img = document.createElement('img');
    img.src = `/api/images/${source.id}/thumb`;
    img.alt = '';
    const drop = document.createElement('button');
    drop.type = 'button';
    drop.className = 'ghost';
    drop.textContent = '×';
    drop.title = `Stop refining image ${source.id}: Generate renders from the prompt alone again`;
    drop.setAttribute('aria-label', `Remove refine source image ${source.id}`);
    drop.onclick = () => setRefine(null);
    chip.append(img, drop);
    els.refineSource.append(chip);
  }
  syncRefineScales();
}

// The scale buttons: the source's size times each factor, lit when the Size
// boxes hold exactly that, dead with no source to multiply or where the
// result is past what the server renders.
function syncRefineScales() {
  const source = refineSupported && refineSource?.width ? refineSource : null;
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
    const size = source ? refineScaled(source.width, source.height, REFINE_SCALES[i]) : null;
    const fits = !!size && refineFits(size);
    button.disabled = !fits;
    button.classList.toggle('active', fits && +els.width.value === size[0] && +els.height.value === size[1]);
    button.title = !size
      ? 'Pick an image to refine first'
      : fits ? `${size[0]} × ${size[1]}` : `${size[0]} × ${size[1]} is outside the sizes this app renders`;
    button.setAttribute('aria-label', `${REFINE_SCALES[i]} times the source's size` + (size ? `, ${button.title}` : ''));
  });
}

function applyRefineScale(scale) {
  if (!refineSource?.width) return;
  [els.width.value, els.height.value] = refineScaled(refineSource.width, refineSource.height, scale);
  syncSize();
  // As `choose` tells them: assigning .value fires no `input`.
  schedulePreview();
  persistPanel();
}

// `source` is {id, width, height} - the size null until it is looked up - or
// null for none. `scale` also sets the Size boxes to the source's size times
// it, falling back through the smaller factors where that is too large.
function setRefine(source, scale = null) {
  refineSource = source;
  if (source?.width && scale) {
    const fitting = [scale, ...REFINE_SCALES.filter((s) => s < scale).reverse()]
      .map((s) => refineScaled(source.width, source.height, s)).find(refineFits);
    if (fitting) [els.width.value, els.height.value] = fitting;
  }
  syncConditioning();
  syncSize();
  schedulePreview();
  persistPanel();
}

// The source's record, for a source that arrived as an id alone - restored
// from the remembered panel, or from Reuse on a refined image - and to learn
// that it has been deleted since, before a Generate is refused over it.
async function loadRefineSource() {
  const id = refineSource?.id;
  if (!id) return;
  let record;
  try {
    record = await api(`/api/images/${id}`);
  } catch (err) {
    // Only a clear "unknown image" drops it: a server that is not up yet
    // says nothing about the gallery.
    if (refineSource?.id === id && /unknown image/.test(err.message)) {
      setRefine(null);
      showError(`Image ${id}, the refine source, is no longer in the gallery.`, true);
    }
    return;
  }
  if (refineSource?.id !== id) return;  // another was picked while this was in flight
  refineSource = { id, width: record.width, height: record.height };
  renderRefine();
  syncSize();
  persistPanel();
}

// Batch holds every image in VRAM at once, which is what stops a 2K batch from
// fitting. Runs trade that for wall-clock: one pipeline call each, seed stepped
// by the batch size so nothing in the sweep repeats.
function syncRuns() {
  const runs = Math.max(1, +els.runs.value || 1);
  const batch = Math.max(1, +els.batch.value || 1);
  const seed = els.seed.value.trim();
  els.runHint.textContent = runs === 1
    ? (batch > 1 ? `${batch} images in one pass — all held in VRAM together.` : '')
    : `${runs * batch} images over ${runs} passes, seed +${batch} each, ` +
      (seed === '' ? 'from a random start.' : `from ${seed}.`);
}
