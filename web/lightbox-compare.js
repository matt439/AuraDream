// Local Diffusion's page script: the lightbox's comparison - its controls,
// toggles, divider and pass stepping; across CFG is lightbox-cfg.js, and its
// geometry (layoutCompare) and state block (lbCompare, closeLightbox's resets,
// lbCfgShown's included) are lightbox.js's. Declarations only; state.js has the
// load-order contract these files share.
'use strict';

// An explicit pair belongs to this lightbox session, independent of later
// selection changes. Ordinary gallery opens still choose source or CFG.
let lbPickedPair = [];
const pickedDiffCache = new WeakMap();
function pickedPair(rec) {
  return rec && lbPickedPair.length === 2 && lbPickedPair.some((r) => r.id === rec.id)
    && lbPickedPair.every((r) => !isGone(r.id)) ? lbPickedPair : null;
}

// Read the same axes and prompt edit as the sheet. Extra reproduction facts
// which no sheet axis lays out still distinguish a pair from two models.
function pickedDifferences(pair) {
  if (pickedDiffCache.has(pair)) return pickedDiffCache.get(pair);
  const [a, b] = pair;
  const axes = [...new Map([...SHEET_AXES, ...axesFor(pair)].map((axis) => [axis.key, axis])).values()];
  const fields = axes.filter((axis) => axisValue(axis, a) !== axisValue(axis, b));
  const extras = [
    ['Model', (r) => r.model_id], ['Size', (r) => `${r.width}x${r.height}`],
    ['Quantization', (r) => r.quantization], ['Refined from', (r) => r.refine],
    ['Mask', (r) => r.mask], ['Outpaint', (r) => r.outpaint ? JSON.stringify(r.outpaint) : null],
  ].filter(([, of]) => of(a) !== of(b));
  const words = { short: new Map(), adapter: null, muStatic: null };
  const label = (r) => [
    ...fields.map((axis) => ['prompt', 'negative'].includes(axis.key) ? axis.label : headText(axis, axisValue(axis, r), words)),
    ...extras.map(([key, of]) => `${key} ${key === 'Model' ? checkpointName(of(r)) : of(r) ?? 'none'}`),
  ].join(' · ');
  const names = [...fields.map((axis) => axis.label), ...extras.map(([key]) => key)];
  const edits = ['prompt', 'negative_prompt'].filter((key) => a[key] !== b[key])
    .map((key) => `${key === 'prompt' ? 'Prompt' : 'Negative'}: ${versionEdit(a[key] ?? '', b[key] ?? '')}`);
  const detail = { label, summary: `Picked order: #${a.id} left, #${b.id} right · ${names.length ? `differs: ${names.join(', ')}` : 'same recorded settings'}`
    + (edits.length ? ` · ${edits.join(' · ')}` : '') };
  pickedDiffCache.set(pair, detail);
  return detail;
}

function pickedTag(rec, pair = pickedPair(rec)) {
  const detail = pair ? pickedDifferences(pair).label(rec) : '';
  return `#${rec.id}${detail ? ` · ${detail}` : ''}`;
}

function comparePickedPair(rec, mode = null) {
  const pair = pickedPair(rec);
  if (!pair) return;
  lbCompare.kind = 'picked';
  lbCompare.ladder = pair;
  lbCompare.source = pair.find((r) => r.id !== rec.id);
  setCompare(rec.id, mode ?? lbCompare.mode);
}

function syncPickedCompare(rec, on) {
  const pair = pickedPair(rec);
  const partner = pair.find((r) => r.id !== rec.id);
  const button = els.lbCompareCfg;
  button.hidden = false;
  button.textContent = 'Compare picked';
  button.setAttribute('aria-pressed', String(on));
  button.setAttribute('aria-disabled', 'false');
  button.setAttribute('aria-keyshortcuts', 'S');
  button.title = `Compare the two picked images at the same crop; #${pair[0].id} stays on the left (S)`;
  els.lbCfgSteps.hidden = true;
  els.lbAgainst.hidden = false;
  syncAgainst(rec, pair, partner, '', (r) => pickedTag(r, pair), 'The other picked image; the first picked stays on the left');
  const full = pickedDifferences(pair).summary;
  return { held: '', steps: true, shown: true, note: on ? (full.length > 200 ? `${full.slice(0, 197)}...` : full) : '', full };
}

// Both records remain at fixed sides; arrows change which one's facts are
// being read, carrying the same crop, divider and mode, without wrapping.
function stepPicked(delta) {
  const rec = lbList[lbIndex];
  const pair = rec && pickedPair(rec);
  if (!pair || lbCompare.id !== rec.id || lbCompare.kind !== 'picked') return false;
  const next = pair[pair.findIndex((r) => r.id === rec.id) + delta];
  if (!next) {
    lbCompareSaid = { id: rec.id, text: `#${rec.id} is the ${delta < 0 ? 'first' : 'last'} of the two picked images` };
    syncCompare(rec);
    return true;
  }
  const i = lbList.findIndex((r) => r.id === next.id);
  if (i < 0) return false;
  lbCompare.id = next.id;
  lbCompare.source = rec;
  lbIndex = i;
  renderLightbox();
  return true;
}

// ------------------------------------------------------- lightbox compare

// The comparison's controls, on every image: the shared part here - the two
// modes, the keys line, the divider's keys, the names and the note - and
// each kind's own in syncSourceCompare and syncCfgCompare (lightbox-cfg.js).
// A control with nothing to do is held, saying why.
function syncCompare(rec) {
  const pass = rec.refine != null;
  const on = lbCompare.id === rec.id;
  const pair = pickedPair(rec);
  const kind = on ? lbCompare.kind : pair ? 'picked' : pass ? 'source' : 'cfg';
  const sourceHeld = syncSourceCompare(rec, on && kind === 'source', !pair);
  const cfg = pair ? syncPickedCompare(rec, on && kind === 'picked')
    : syncCfgCompare(rec, on && kind === 'cfg', !pass, cfgShown(rec, on && kind === 'cfg'));
  // The box: on every image from 901 px, where it is the column's; under it
  // only where there is something in it (cfgShown), as the CFG part is.
  els.lbCompare.hidden = !pass && !cfg.shown;
  // A mode turns on the kind a press of S would, or switches the one on: held
  // where that has nothing to compare with, its press saying why.
  const held = on || pair ? '' : pass ? sourceHeld : cfg.held;
  for (const [button, mode] of [[els.lbSwipe, 'swipe'], [els.lbSideBySide, 'side']]) {
    button.setAttribute('aria-pressed', String(on && lbCompare.mode === mode));
    // A deleted source holds them on a pass whether or not CFG is on, as ever.
    const why = held || (pass && kind === 'source' ? sourceHeld : '');
    button.setAttribute('aria-disabled', String(!!why));
    // Held, the title is why - on every image with no CFG neighbour, most
    // plain Generates; otherwise what the mode shows, in the words of the
    // kind it would compare (Side by side's, which way the halves go, is
    // layoutLightbox's).
    const title = why || (mode === 'side' ? lbSidesTitle : kind === 'picked'
      ? 'The two picked images over each other, a divider between them to drag'
      : kind === 'cfg'
      ? 'The lower CFG and the higher over each other, a divider between them to drag'
      : 'The source and the pass over each other, a divider between them to drag');
    if (title && button.title !== title) button.title = title;
  }
  // Comparing, with another pass of the same source in the strip, the arrows
  // swap passes (`stepPass`), or with another CFG value of the image in it,
  // step along CFG (`stepCfg`); the divider takes Shift+arrows instead.
  const passes = on && lbSiblingPass(1) >= 0;
  const cfgSteps = on && kind === 'cfg' && cfg.steps;
  const pairSteps = on && kind === 'picked';
  const divider = on && lbCompare.mode === 'swipe';
  // The keys line names S, which does something on every image: it compares
  // a pass with its source, any other image across CFG, or says why it
  // cannot. In the title and the button's hint alone, it was found only by
  // hovering. Likewise what the arrows do while comparing.
  const keys = lbKeysLine({ passes, cfg: cfgSteps, picked: pairSteps, divider, binned: !!rec.binned_at });
  if (els.lbKeys.textContent !== keys) els.lbKeys.textContent = keys;
  if (passes || cfgSteps || pairSteps) {
    els.lbSplit.setAttribute('aria-keyshortcuts', 'Shift+ArrowLeft Shift+ArrowRight PageUp PageDown Home End');
    els.lbSplit.title = passes ? 'Drag, or Shift+← →, to move the divider; ← → switch to another pass of this source'
      : pairSteps ? 'Drag, or Shift+← →, to move the divider; ← → switch between the two picked images'
      : 'Drag, or Shift+← →, to move the divider; ← → step along CFG';
  } else {
    els.lbSplit.removeAttribute('aria-keyshortcuts');
    els.lbSplit.removeAttribute('title');
  }
  const source = on ? lbCompare.source : null;
  if (source && els.lbSrcImg.getAttribute('src') !== source.url) els.lbSrcImg.src = source.url;
  if (source && kind === 'source') {
    els.lbSrcImg.alt = `#${source.id}, the image this was refined from`;
    els.lbTagSrc.textContent = `Source #${source.id}`;
    els.lbTagPass.textContent = `Pass #${rec.id}`;
    els.lbSplit.setAttribute('aria-label', 'Divider: the source left of it, the pass right');
  } else if (source && kind === 'picked') {
    els.lbSrcImg.alt = pickedTag(source, pair);
    els.lbTagSrc.textContent = clip(pickedTag(source, pair));
    els.lbTagPass.textContent = clip(pickedTag(rec, pair));
    const [left, right] = pair;
    els.lbSplit.setAttribute('aria-label', `Divider: #${left.id} left of it, #${right.id} right`);
  } else if (source) {
    // Named by CFG, the lower on the left (layoutCompare) and said so.
    els.lbSrcImg.alt = `#${source.id}, ${cfgName(source)}`;
    els.lbTagSrc.textContent = cfgTag(source);
    els.lbTagPass.textContent = cfgTag(rec);
    const [low, high] = lbSwapped(rec) ? [rec, source] : [source, rec];
    els.lbSplit.setAttribute('aria-label', `Divider: ${cfgName(low)} left of it, ${cfgName(high)} right`);
  }
  const said = lbCompareSaid?.id === rec.id ? lbCompareSaid.text : '';
  const note = said || (kind === 'source' ? sourceNote(rec, source) : on ? cfg.note : '');
  if (els.lbCompareNote.textContent !== note) els.lbCompareNote.textContent = note;
  els.lbCompareNote.title = kind === 'picked' && on ? cfg.full : '';
  for (const tag of [els.lbTagSrc, els.lbTagPass]) tag.classList.toggle('picked', kind === 'picked');
}

// Whether the CFG part of the comparison is drawn. From 901 px always, held
// where there is nothing to compare: the box is the column's, pinned above
// the buttons, and an answer from the gallery changes no layout. Under 901 px
// the box is a row of the strip under the stage, and a row there is height
// the picture and the facts' letterbox lose (lbFactsRoom): a square picture
// at 600x900 kept 77 px of letterbox where it had 126, and lost its last
// facts. There it is drawn where the page's own records give the image a
// CFG neighbour, decided when the image is drawn and kept until another one
// is (`lbCfgShown`) - never on the gallery's later answer, which would move
// the strip under the pointer - and once a comparison across CFG is on (S
// works without it).
let lbCfgShown = { id: null, shown: false };
function cfgShown(rec, on) {
  if (lbCfgShown.id !== rec.id) lbCfgShown = { id: rec.id, shown: cfgLadder(rec).length > 1 };
  if (on) lbCfgShown.shown = true;
  return !lbNarrow() || lbCfgShown.shown;
}

// Compare with source, on a pass only: dead, saying why, where the source has
// gone. Answers that reason, or '' where it can be pressed.
function syncSourceCompare(rec, on, shortcut = true) {
  const pass = rec.refine != null;
  els.lbCompareOn.hidden = !pass;
  if (!pass) return '';
  const gone = rec.refine_missing || lbSources.get(rec.refine) === null;
  els.lbCompareOn.setAttribute('aria-pressed', String(on));
  els.lbCompareOn.setAttribute('aria-disabled', String(gone));
  const why = gone ? `#${rec.refine}, the image this was refined from, ${rec.refine_binned ? 'is in the recycle bin' : 'has been deleted'}` : '';
  if (shortcut) els.lbCompareOn.setAttribute('aria-keyshortcuts', 'S');
  else els.lbCompareOn.removeAttribute('aria-keyshortcuts');
  els.lbCompareOn.title = why || `Compare this pass with #${rec.refine}, the image it was refined from, at the same crop${shortcut ? ' (S)' : ''}`;
  return why;
}

// A pass's note: its source and how much of it the pass kept, or that the
// source has gone.
function sourceNote(rec, source) {
  if (rec.refine == null) return '';
  if (rec.refine_missing || lbSources.get(rec.refine) === null) return `source #${rec.refine} ${goneWord(rec)}`;
  if (!source) return '';
  const kept = refineKept(source.width, source.height, rec.width, rec.height);
  const cropped = kept.w < 1 || kept.h < 1;
  return `#${source.id}, ${source.width}×${source.height}${cropped ? ', cropped as the pass was' : ''}`;
}

// The lightbox's keys, under the comparison (index.html has the line as the
// first image will read it). Each key is bound to what it does - no-break
// spaces, and a word joiner in +/− - so the line breaks only after a ·: the
// longer line of a pass split "+/" from "−" at 1400 px. Comparing, the arrows
// read "passes" where they swap passes of the source (`passes`), "CFG" where
// they step along CFG (`cfg`), and Shift+arrows move a swipe's divider
// (`divider`). Del moves the image to the recycle bin, and deletes one
// already there for good (`binned`).
function lbKeysLine({ passes = false, cfg = false, picked = false, divider = false, binned = false } = {}) {
  const keys = [['← →', passes ? 'passes' : cfg ? 'CFG' : picked ? 'picked' : 'walk'], ...(divider ? [['Shift+← →', 'divider']] : []),
    ['F', 'star'], ['R', 'refine'], ['S', 'compare'], ['Del', binned ? 'delete permanently' : 'bin'],
    ['0', 'fit'], ['1', '1:1'], ['+/−', 'zoom'], ['Esc', 'close']];
  return keys.map(([key, does]) => `${key.replaceAll(' ', '\u00a0').replace('/', '/\u2060')}\u00a0${does}`)
    .join('\u00a0· ');
}

// S, the buttons and the two modes. Off, a mode turns on the comparison S
// would - a pass's with its source, any other image's across CFG - in that
// mode; on, S and the pressed kind's button turn it off, the other kind's
// button switches to that kind, and the other mode switches to it, keeping
// the kind. The source is fetched the first time, and a source the gallery
// no longer has leaves the controls dead with the reason; a press with
// nothing to compare says why (compareCfg).
async function toggleCompare(mode = null, kind = null) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  if (lbCompare.id === rec.id && (!kind || kind === lbCompare.kind)) {
    if (!mode) setCompare(null);
    else if (mode !== lbCompare.mode) setCompare(rec.id, mode);
    return;
  }
  kind ??= pickedPair(rec) ? 'picked' : rec.refine != null ? 'source' : 'cfg';
  if (kind === 'picked') { comparePickedPair(rec, mode); return; }
  if (kind === 'cfg') { compareCfg(rec, null, mode); return; }
  if (rec.refine == null) return;
  const source = await compareSource(rec);
  if (lbList[lbIndex] !== rec || !els.lightbox.open) return;
  if (!source) { syncCompare(rec); return; }
  lbCompare.kind = 'source';
  lbCompare.ladder = [];
  lbCompare.source = source;
  lbCompareSaid = null;
  setCompare(rec.id, mode ?? lbCompare.mode);
}

// A pass's source by id, once a session (`lbSources`).
async function compareSource(rec) {
  const id = rec.refine;
  if (rec.refine_missing) return null;
  if (lbSources.has(id)) return lbSources.get(id);
  // An image already in a strip - the Refine dialog opens a pass behind its
  // source, a sweep's sheet walks the source first - needs no request.
  let source = [...lbList, ...galleryRecords, ...viewerRecords].find((r) => r.id === id);
  if (!source) {
    try {
      source = await api(`/api/images/${id}`);
    } catch (err) {
      if (!/unknown image/.test(err.message)) { showError(err.message); return null; }
      source = null;
    }
  }
  lbSources.set(id, source);
  return source;
}

// On, off, or another mode, keeping what is in view: side by side fits the
// picture into half the stage, so a zoom is carried as the same magnification
// - a 1:1 stays 1:1 on the same spot - and only fit stays fit.
function setCompare(id, mode = lbCompare.mode) {
  const rec = lbList[lbIndex];
  const before = rec ? fitSize(rec)[0] : 0;
  const held = els.lbSplit.contains(document.activeElement);
  lbCompare.id = id;
  lbCompare.mode = mode;
  // A line said for the last state (an end of the ladder) is not this one's.
  lbCompareSaid = null;
  if (id === null) { lbCompare.source = null; lbCompare.ladder = []; }
  if (!rec) return;
  if (lbView.scale !== 1) {
    lbView.scale = clampScale(rec, lbView.scale * before / fitSize(rec)[0]);
    if (lbView.scale <= 1) lbView.x = lbView.y = 0;
  }
  syncCompare(rec);
  layoutLightbox();
  // S pressed on the divider hides it with the focus in it, which fell to the
  // top of the dialog or out of it: it goes to the toggle that brings the
  // divider back instead: the toggle of the kind that was on.
  if (held && els.lbSplit.hidden) {
    (['cfg', 'picked'].includes(lbCompare.kind) || els.lbCompareOn.hidden ? els.lbCompareCfg : els.lbCompareOn).focus({ preventScroll: true });
  }
}

// The divider, from a drag or its keys: a fraction of the picture on screen.
function moveSplit(clientX) {
  const [from, to] = lbCompare.range;
  const x = clientX - els.lbStage.getBoundingClientRect().left;
  lbCompare.split = to > from ? Math.min(1, Math.max(0, (x - from) / (to - from))) : 0.5;
  layoutLightbox();
}

// The partner's picture: the comparison ends, and says why. A CFG partner
// is asked for, as lbImageFailed asks for the image in view, to tell one
// deleted elsewhere - dropped from every ladder - from a file that failed.
async function lbSourceFailed() {
  const rec = lbList[lbIndex];
  if (!rec || lbCompare.id !== rec.id || !els.lbSrcImg.getAttribute('src')) return;
  const { kind, source } = lbCompare;
  setCompare(null);
  lbCompareSaid = { id: rec.id, text: `#${source.id} could not be loaded` };
  syncCompare(rec);
  if (!['cfg', 'picked'].includes(kind)) return;
  try {
    await api(`/api/images/${source.id}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) return;
    if (kind === 'picked') forgetImages([source.id]);
    else forgetLadderImages(new Set([source.id]));
    if (lbList[lbIndex] !== rec || lbCompareSaid?.id !== rec.id) return;
    lbCompareSaid = { id: rec.id, text: `#${source.id} is no longer in the gallery` };
    syncCompare(rec);
  }
}

// The divider's keys, while it has focus (app.js skips a key handled here).
// With no other pass of the source in the strip, the arrows move it rather
// than walking the strip. With one, the plain arrows are left to app.js,
// which swaps passes (`stepPass`) - comparing three passes against their
// source is what the arrows were wanted for, and a drag left the focus here -
// and likewise with another CFG value of a CFG comparison in the strip,
// which they step to (`stepCfg`, lightbox-cfg.js).
// Shift+arrows move it by the same 2% either way, as `nudgeSplit` does off
// it, so the step does not depend on where the focus is; PageUp/PageDown
// (10%), Home and End move it either way, as a slider's do.
function lbSplitKey(e) {
  const arrow = e.key === 'ArrowLeft' || e.key === 'ArrowRight';
  if (arrow && !e.shiftKey && (lbSiblingPass(1) >= 0 || cfgArrows() || lbCompare.kind === 'picked')) return;
  const to = {
    ArrowLeft: lbCompare.split - 0.02, ArrowRight: lbCompare.split + 0.02,
    PageDown: lbCompare.split - 0.1, PageUp: lbCompare.split + 0.1, Home: 0, End: 1,
  }[e.key];
  if (to === undefined) return;
  e.preventDefault();
  lbCompare.split = Math.min(1, Math.max(0, to));
  layoutLightbox();
}

// Shift+arrows anywhere in the lightbox while a swipe is up: the divider by
// the slider's small step, so it moves without tabbing to it. False where
// there is no divider, and the key walks as it always did.
function nudgeSplit(delta) {
  const rec = lbList[lbIndex];
  if (!rec || lbCompare.id !== rec.id || lbCompare.mode !== 'swipe' || els.lbSplit.hidden) return false;
  lbCompare.split = Math.min(1, Math.max(0, lbCompare.split + delta * 0.02));
  layoutLightbox();
  return true;
}

// Comparing a pass with its source, the index of the next pass of the same
// source in the strip in the direction of `delta`, wrapping round - the
// source and other images in between are skipped - or -1 where the strip
// holds no other, or the comparison is across CFG. Only the strip as
// loaded: a gallery page not yet fetched is not pulled for it.
function lbSiblingPass(delta) {
  const rec = lbList[lbIndex];
  if (!rec || rec.refine == null || lbCompare.id !== rec.id || lbCompare.kind !== 'source') return -1;
  const n = lbList.length;
  for (let k = 1; k < n; k++) {
    const i = (((lbIndex + delta * k) % n) + n) % n;
    if (lbList[i].refine === rec.refine && lbList[i].id !== rec.id) return i;
  }
  return -1;
}

// ← → while comparing a pass with another pass of its source in the strip:
// that pass, compared with the same source, in the same mode, at the same
// divider. False where there is none, and the arrows walk (ending it).
function stepPass(delta) {
  const i = lbSiblingPass(delta);
  if (i < 0) return false;
  lbCompare.id = lbList[i].id;
  lbIndex = i;
  renderLightbox();
  return true;
}
