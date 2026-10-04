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
  for (const strip of [els.presets, els.tiers]) strip.addEventListener('keydown', (e) => stepPreset(strip, e));
}

// Each strip is one tab stop, the chosen preset (syncSize keeps the roving
// tabindex), and the arrows move between its buttons - seventeen stops used to
// stand between the prompt and Generate. Focus only: each choice persists the
// panel and asks for a preview, so it waits for Enter or Space, as before.
// The buttons are queried, not the strip's children: a pair is a box of two.
function stepPreset(strip, e) {
  const buttons = [...strip.querySelectorAll('button')];
  const from = buttons.indexOf(e.target);
  if (from < 0) return;
  const to = { ArrowLeft: from - 1, ArrowRight: from + 1, Home: 0, End: buttons.length - 1 }[e.key];
  if (to === undefined || !buttons[to]) return;
  e.preventDefault();
  buttons[from].tabIndex = -1;
  buttons[to].tabIndex = 0;
  buttons[to].focus();
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
    markPreset(b, !!match && aspect === match[0], guided);
    describePreset(b, aspect, pickedTier, guided);
  });
  [...els.tiers.children].forEach((b, i) => {
    const tier = TIERS[i];
    const guided = presetIsInGuide(pickedAspect, tier);
    markPreset(b, !!match && tier === match[1], guided);
    // Spell out what the tier means for the framing currently in play.
    describePreset(b, pickedAspect, tier, guided);
  });
  for (const strip of [els.presets, els.tiers]) {
    const buttons = [...strip.querySelectorAll('button')];
    // A mark on every preset says nothing: the size line carries the range.
    strip.classList.toggle('all-guided', buttons.every((b) => b.classList.contains('recommended')));
    // The strip's one tab stop: the chosen preset, or the first where the
    // size is none of them.
    if (!buttons.some((b) => b.tabIndex === 0)) buttons[0].tabIndex = 0;
  }

  const mp = (w * h) / 1e6;
  const notes = [];
  const namedAspect = match?.[0] || null;
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
  if (sizeIsTight(w, h)) notes.push('may run tight on 16GB VRAM');
  els.sizeHint.textContent = `${mp.toFixed(1)} MP` + notes.map((n) => ` — ${n}`).join('');
  els.sizeHint.classList.toggle('warn', notes.length > 0);
}

// 4 MP on a 16GB card is the edge of what fits once the DiT is resident; the
// Refine dialog's size line says the same past it.
const sizeIsTight = (w, h) => w * h > 1600 * 1600;

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

// The chosen preset was a class alone, so a screen reader could not tell which
// one it was: aria-pressed says so, and it is the strip's tab stop.
function markPreset(button, active, guided) {
  button.classList.toggle('active', active);
  button.classList.toggle('recommended', guided);
  button.setAttribute('aria-pressed', String(active));
  button.tabIndex = active ? 0 : -1;
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
  // A checkpoint that takes none at all is its caption alone (style.css):
  // a greyed two-line box that nothing could wake took the room the
  // reference chips under it use. Its text is kept for one that does.
  els.negField.classList.toggle('unsupported', !negativeSupported);
  els.negative.disabled = off;
  els.negNote.textContent = !negativeSupported
    ? (els.negative.value.trim() ? '(this checkpoint takes none; the text is kept for one that does)'
      : '(this checkpoint takes no negative prompt)')
    : !trueCfgSupported && gridMaxCfg !== null && +els.cfg.value <= cfgThreshold && reach > cfgThreshold
      ? `(applies to the grid cells above CFG ${cfgThreshold} — ~2× slower there)`
      : `(needs ${knob} > ${cfgThreshold} — ~2× slower)`;
  // The way to the box that wakes it, which on FLUX sits a field's height
  // below and is named only in the note.
  els.negEnable.hidden = !(off && negativeSupported);
  els.negEnable.textContent = `Set ${knob}`;
  els.negEnable.title = `Go to the ${knob} box: above ${cfgThreshold} it runs the negative prompt`;
  // A negative prompt greyed or woken here stops or starts running, and the
  // prompt check read it as it ran then.
  syncAnalysisStale();
}

// The greyed negative box's button: focus the box that decides, its value
// selected, and write nothing - a value is the checkpoint's (Qwen-Image's
// guide asks for about 4, schnell's for none), and its tip says which.
function focusNegativeKnob() {
  if (!negativeSupported || !els.negative.disabled) return;
  const knob = trueCfgSupported ? els.trueCfg : els.cfg;
  knob.focus();
  // The press was in the panel, so its focusin moved nothing; the box it
  // focused is elsewhere in the panel, and is brought into view the way a
  // control focused from the keyboard is.
  revealInPanel(knob);
  knob.select();
}

// The conditioning controls, greyed for the checkpoint (references) or for
// the installation (adapters) rather than for a value - so a dead box always
// has a reason, and the reason is in the badge beside it.
function syncConditioning() {
  els.referenceField.classList.toggle('disabled', !referenceSupported);
  // The badge says what references do where they are taken, and why the
  // field is dead where they are not.
  els.referenceTip.textContent = referenceSupported
    ? 'Up to four gallery images the picture is conditioned on, in this order: drag a chip, '
      + 'or press Alt+Left/Right on its ×, to move it. With two or more, say in the prompt '
      + 'what to take from each by its number - "the jacket from image 1". Each is resized '
      + 'to about 1 MP on the way in.'
    : referenceTipDefault;
  els.lora.disabled = !loraSupported;
  els.loraScale.disabled = !loraSupported || !els.lora.value;
  els.loraField.classList.toggle('disabled', !loraSupported);
  // A disabled box shows no tooltip, so the reason is its wrapper's - and only
  // the value's: an installation with no adapters says why in the badge.
  els.loraScale.closest('.sub').title = loraSupported && !els.lora.value ? 'Pick an adapter first' : '';
  els.gridLoraScale.disabled = !loraSupported || !els.lora.value;
  els.gridLoraField.classList.toggle('disabled', els.gridLoraScale.disabled);
  // The sweep's box names the adapter it weighs, 570 px below the select -
  // whole in its title, since the box is narrow - and, with none picked, says
  // so in two words: "pick an adapter above" was cut to "…abov" at QHD. Where
  // to pick one is the wrapper's title, as the panel's Weight has it.
  const adapter = els.lora.value.replace(/\.safetensors$/i, '');
  els.gridLoraScale.placeholder = !loraSupported ? '0-1x0.25'
    : adapter ? `${adapter}: 0-1x0.25` : 'no adapter';
  els.gridLoraScale.title = adapter ? els.gridLoraScale.placeholder : '';
  els.gridLoraField.title = loraSupported && !adapter ? 'Pick an adapter in LoRA above first' : '';
  renderReferences();
}

// The chips: one per selected image, in order, each numbered as the prompt
// names it, movable and removable. Rendered from `referenceIds` rather than
// from the DOM, because Reuse and the gallery's own button both write the
// list - and every change, a move included, goes through setReferences on a
// copy, so the panel is remembered, the preview asked again and the prompt
// check marked stale.
let draggedReference = null;

function renderReferences() {
  els.referenceList.textContent = '';
  const n = referenceIds.length;
  // One reference needs no number in the prompt; two or more do, and the
  // chips' badges are the numbers.
  els.referenceNote.textContent = !referenceSupported ? '(this checkpoint takes none)'
    : n === 0 ? '(add one with the ❐ button on a gallery image)'
    : n === 1 ? '(1 of 4)'
    : `(${n} of 4 — name them ${n === 2 ? 'image 1 and image 2' : `image 1 to image ${n}`} in the prompt)`;
  referenceIds.forEach((id, i) => {
    const name = `Reference image ${i + 1} (gallery #${id})`;
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.title = name;
    chip.draggable = true;
    const img = document.createElement('img');
    img.src = `/api/images/${id}/thumb`;
    img.alt = name;
    img.draggable = false;
    // Deleted since it was picked, or remembered by this browser from before.
    img.onerror = () => {
      chip.classList.add('gone');
      chip.title = img.alt = `Reference image ${i + 1} (gallery #${id}, deleted)`;
    };
    const number = document.createElement('b');
    number.className = 'chip-n';
    number.textContent = i + 1;
    number.setAttribute('aria-hidden', 'true');
    const drop = document.createElement('button');
    drop.type = 'button';
    drop.className = 'ghost';
    drop.textContent = '×';
    drop.title = `Remove reference image ${i + 1} (#${id}) · Alt+Left/Right moves it`;
    drop.setAttribute('aria-label', `Remove reference image ${i + 1}`);
    drop.setAttribute('aria-keyshortcuts', 'Delete Alt+ArrowLeft Alt+ArrowRight');
    drop.onclick = () => removeReference(i);
    drop.onkeydown = (e) => {
      if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        e.preventDefault();
        const to = i + (e.key === 'ArrowLeft' ? -1 : 1);
        if (moveReference(i, to)) focusReference(to);
      } else if (e.key === 'Delete') {
        e.preventDefault();
        removeReference(i);
      }
    };
    chip.ondragstart = (e) => {
      draggedReference = i;
      chip.classList.add('dragging');
      if (e.dataTransfer) {
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(id));
      }
    };
    // Dropped on a chip's left half it goes before that chip, on its right half after.
    const after = (e) => e.clientX >= chip.getBoundingClientRect().left + chip.offsetWidth / 2;
    chip.ondragover = (e) => {
      if (draggedReference === null) return;
      e.preventDefault();
      clearReferenceDropMarkers();
      chip.classList.add(after(e) ? 'drop-after' : 'drop-before');
    };
    chip.ondrop = (e) => {
      if (draggedReference === null) return;
      e.preventDefault();
      const from = draggedReference;
      const insertion = i + (after(e) ? 1 : 0);
      draggedReference = null;
      clearReferenceDropMarkers();
      moveReference(from, insertion > from ? insertion - 1 : insertion);
    };
    chip.ondragend = () => {
      draggedReference = null;
      chip.classList.remove('dragging');
      clearReferenceDropMarkers();
    };
    chip.append(img, number, drop);
    els.referenceList.append(chip);
  });
  // The gallery's cards and the sheet's cells say which images these are.
  syncReferenceMarks();
}

function clearReferenceDropMarkers() {
  els.referenceList.querySelectorAll('.drop-before, .drop-after')
    .forEach((chip) => chip.classList.remove('drop-before', 'drop-after'));
}

// False where there is nowhere to go: the first chip moved left, the last right.
function moveReference(from, to) {
  if (to < 0 || to >= referenceIds.length || to === from) return false;
  const ids = referenceIds.slice();
  ids.splice(to, 0, ids.splice(from, 1)[0]);
  setReferences(ids);
  return true;
}

// Focus stays in the field rather than dropping to <body>: the next chip's ×,
// else the previous one's, else the emptied row itself. Not the field's
// badge: a focused badge opens its tip, which then covered LoRA and Seed for
// someone who had only clicked the last × away.
function removeReference(index) {
  setReferences(referenceIds.filter((_, i) => i !== index));
  if (referenceIds.length) focusReference(Math.min(index, referenceIds.length - 1));
  else els.referenceList.focus();
}

function focusReference(index) {
  els.referenceList.querySelectorAll('.chip button')[index]?.focus();
}

function setReferences(ids) {
  referenceIds = ids.slice(0, 4);
  renderReferences();
  schedulePreview();
  persistPanel();
  // The prompt check counted them: klein's guide asks what each one is for.
  syncAnalysisStale();
}

// Batch holds every image in VRAM at once, which is what stops a 2K batch from
// fitting. Repeat (the request's `runs`) trades that for wall-clock: one
// pipeline call each, seed stepped by the batch size so nothing in the sweep
// repeats. Batches, not passes: a pass is the Refine dialog's word.
function syncRuns() {
  const runs = Math.max(1, +els.runs.value || 1);
  const batch = Math.max(1, +els.batch.value || 1);
  const seed = els.seed.value.trim();
  const start = seed === '' ? 'from a random start.' : `from ${seed}.`;
  // A seed the page put in the box - a grid's base written back for Resume,
  // a finished image's through Use seed - says so while it is there: every
  // Generate after it repeats it, where a blank box draws afresh each press.
  const fixed = seed !== '' && seed === pageSeed ? `Seed fixed at ${seed}; 🎲 for a random one.` : '';
  els.runHint.textContent = runs === 1
    ? [batch > 1 ? `${batch} images at once — all held in VRAM together.` : '', fixed].filter(Boolean).join(' ')
    : batch === 1
      ? `${runs} images, one after another, seed +1 each, ${start}`
      : `${runs * batch} images: ${runs} batches of ${batch}, one after another, seed +${batch} each, ${start}`;
}
