// AuraDream's page script: the Inpaint dialog - opening, the mask's
// upload, the preview, submission and close.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ----------------------------------------------------------------- inpainting

// An inpaint is a refine pass with a mask: the same source (a gallery id),
// strength, prompt, seed and sampling, run through the inpaint pipeline
// diffusers ships beside the loaded one, at the source's own size, and
// composited back over the source with the mask by the server, so every
// unmasked pixel is the source's. The dialog is its own rather than a mode of
// Refine's: Refine's scale buttons, W and H and kept-region frame are all
// about resizing, which an inpaint never does - a reframing would misalign
// the mask. It shares what is the same: the prompt check (refine-prompt.js's
// lines), the strength box's rules (one value, or a list or range that is a
// sheet), Versions, Seed, the server's preview and how a refusal is said.
// The panel's requests never carry a source or a mask (`collect()`).

// Whether the checkpoint can run an inpaint (the status's `inpaint_supported`,
// the profile's guess before a load and the loaded class's answer after), and
// the record the dialog is open on, null while it is closed.
let inpaintSupported = false;
let inpaintSource = null;
// The inpainted record a Reuse opened on, whose mask, strength and seed the
// dialog starts from, or null.
let inpaintFrom = null;
// Where a dialog the lightbox opened goes back to on a close that ran nothing,
// and the same held across a Start over, which opens the dialog again.
let inpaintReturn = null;
let inpaintReturnKeep = null;
// The adapter carried over from the record, {name, scale}, or null.
let inpaintLora = null;
let inpaintRan = false;
let inpaintSending = false;
let inpaintWaiting = false;
// What openInpaint filled the boxes with, to tell a close's unsent edits.
let inpaintFilled = null;
// The strength the box was filled with where the library's own default was
// not known yet (no load): a load that answers puts the library's in place of
// it, where the box still holds it. Null otherwise.
let inpaintStrengthStandIn = null;

// The mask as the server stored it ({hash, width, height}), or null, and
// where its upload is: 'none' (nothing painted), 'pending' (exported and sent
// a moment after the last stroke), 'stored', 'empty' (an eraser took it all),
// 'refused' (the server's words in `inpaintMaskRefusal`) or 'loading' (a
// Reuse's stored mask on its way).
let inpaintMask = null;
let inpaintMaskState = 'none';
let inpaintMaskRefusal = '';
let inpaintUploadTimer;
let inpaintUploadEpoch = 0;
// When the server last stored or kept the mask (Date.now()), and the timer
// that keeps it while the dialog stays open on it: an unused mask is swept
// an hour after its last upload (`gallery_delete.sweep_masks`), and a dialog
// left open over lunch must not lose it. Well inside that hour.
let inpaintMaskKeptAt = 0;
let inpaintKeepTimer;
const INPAINT_KEEP_MS = 20 * 60 * 1000;
// Counts openInpaint's fills: a Reuse's mask that lands after the dialog was
// filled again (Start over, a close and another open) is not applied.
let inpaintOpenEpoch = 0;
// Counts adapters taken out (dropInpaintLora), so a press that waited on a
// preview that took one out stops there, as Refine's does.
let inpaintLoraDrops = 0;

let inpaintTimer;
let inpaintEpoch = 0;
// What the last preview refused, and the request it answered for (as JSON).
let inpaintProblem = null;
let inpaintPreviewed = null;
let inpaintCheckTimer;
let inpaintCheckEpoch = 0;
let inpaintChecked = null;

// What a close that ran nothing left on an image - its mask's strokes and the
// boxes - in page memory only, for the next open of that image on the same
// checkpoint. A few images' worth: a stored base is the source's size.
const inpaintKept = new Map();
const INPAINT_KEPT = 4;

// The families this app inpaints, for the sentence that says where it is
// not available.
const INPAINT_FAMILIES = 'The FLUX.1, SDXL and Z-Image checkpoints have it.';

// Why an inpaint cannot run here, one sentence wherever the page says it: no
// inpaint pipeline in the installed diffusers, or one the app has not enabled
// for this family yet (`inpaint_deferred`, the backend's "not enabled yet").
// An outpaint runs through the same inpaint pipeline (outpaint.js), and
// `outpainting` says so, as the server's refusal does.
function inpaintNone(outpainting = false) {
  const name = checkpointName(loadedModel) || 'This checkpoint';
  return lastDefaults?.inpaint_deferred
    ? `Inpainting${outpainting ? ', which an outpaint runs through,' : ''} is not enabled yet for ${name} in this app: `
      + `the installed diffusers has its inpaint pipeline, but its family waits on measurements. ${INPAINT_FAMILIES}`
    : `${name} has no inpaint pipeline in the installed diffusers${outpainting ? ', which an outpaint runs through' : ''}. `
      + INPAINT_FAMILIES;
}

// The same, as Refine says it: a guess before a load said as one, a final
// answer as final.
function inpaintUnavailableText(outpainting = false) {
  const what = outpainting ? 'Outpaint' : 'Inpaint';
  return lastDefaults?.inpaint_settled === false
    ? `${what} is unavailable: as far as the server can tell, this checkpoint has no inpaint pipeline. `
      + `A load answers for certain - the ${what.toLowerCase()} stays here until then.`
    : `${what} is unavailable. ${inpaintNone(outpainting)}`;
}

// The status's answer, as it lands: the tools that open the dialog follow it,
// and an open dialog asks its preview again.
// The reason moves too (a deferral, a checkpoint's name), and the titles with it.
let inpaintReasonShown = null;
function syncInpaintCapability(defaults) {
  if (typeof defaults?.inpaint_supported !== 'boolean') return;
  const changed = defaults.inpaint_supported !== inpaintSupported
    || (!defaults.inpaint_supported && inpaintNone() !== inpaintReasonShown);
  inpaintSupported = defaults.inpaint_supported;
  inpaintReasonShown = inpaintSupported ? null : inpaintNone();
  if (changed) {
    syncInpaintTools();
    if (inpaintSource) scheduleInpaintPreview();
    if (outpaintSource) scheduleOutpaintPreview();
    if (els.lightbox.open) renderLightbox();
  }
  if (inpaintSource) {
    takeLibraryStrength();
    syncInpaint();
  }
  // The Outpaint dialog's, which runs through the same class (outpaint.js).
  syncOutpaintCapability();
}

// A card's, a result's and the lightbox's Inpaint: held, with the reason as
// the title, where the checkpoint has none - never disabled, so each keeps its
// tab stop and a press says why.
function syncInpaintTool(button) {
  button.setAttribute('aria-disabled', String(!inpaintSupported));
  const title = inpaintSupported ? 'Inpaint: paint over part of this image and redraw only that'
    : `Inpaint - ${inpaintNone()}`;
  button.title = title;
  button.setAttribute('aria-label', title);
  button.dataset.tip = inpaintSupported ? 'Inpaint…' : 'No inpaint here';
}

function syncInpaintTools() {
  document.querySelectorAll('.inpaint-tool').forEach(syncInpaintTool);
  document.querySelectorAll('.outpaint-tool').forEach(syncOutpaintTool);
}

function syncLightboxInpaint(rec) {
  const binned = !!rec.binned_at;
  els.lbInpaint.setAttribute('aria-disabled', String(!inpaintSupported || binned));
  els.lbInpaint.title = binned ? 'In the recycle bin: restore it to inpaint it'
    : inpaintSupported ? 'Inpaint this image: paint over part of it and redraw only that'
      : inpaintNone();
}

// The lightbox's Inpaint…: a held one says why in the lightbox's line and
// opens nothing.
function pressLightboxInpaint() {
  const rec = lbList[lbIndex];
  if (!rec || refuseBinned(rec, 'inpaint')) return;
  if (!inpaintSupported) { showError(inpaintNone(), true); return; }
  openInpaint(rec);
}

// A card's or a result's ✎: on a checkpoint without one it says so under the
// gallery's head and over the tile, and opens nothing.
function inpaintFromCard(rec, div) {
  if (!inpaintSupported) { refuseInCard(div, inpaintNone(), 'No inpaint here'); return; }
  openInpaint(rec);
}

// Where the Strength box starts: the inpaint pipeline's own default, read off
// the installed diffusers at the load (`defaults.inpaint_strength`) - the
// library's number, since no checkpoint's authors give one for inpainting -
// and before a load, while that is unknown, the checkpoint's refine strength,
// which the note says. Where the profile measured its own start (FLUX.1's
// 0.95, `defaults.inpaint_start`) the server reports that instead, before a
// load as well, and the note says it is the checkpoint's, not the library's. Where that number was measured as the right start is
// the server's to say (`hints.inpaint_strength`), Outpaint's note too.
function inpaintLibraryStrength() {
  return typeof lastDefaults?.inpaint_strength === 'number' ? lastDefaults.inpaint_strength : null;
}

function inpaintStartStrength() {
  return inpaintLibraryStrength() ?? refineDefaultStrength();
}

function inpaintStrengthNote() {
  const library = inpaintLibraryStrength();
  const measured = typeof lastHints?.inpaint_strength === 'string' ? lastHints.inpaint_strength
    : `the library's number, not one measured for ${checkpointName(loadedModel)}`;
  // A profile's own measured start (`defaults.inpaint_start`, FLUX.1's) is
  // not the library's number, and the note must not call it that.
  const own = typeof lastDefaults?.inpaint_start === 'number' && lastDefaults.inpaint_start === library;
  if (own) return `${library} is this checkpoint's own start, in place of the installed diffusers' default for this inpaint pipeline - ${measured}.`;
  return library !== null
    ? `${library} is the installed diffusers' default for this inpaint pipeline - ${measured}.`
    : `The inpaint pipeline's own default is read when the model loads; until then Strength starts at `
      + `${refineDefaultStrength()}, this checkpoint's refine strength.`;
}

// A load that answers with the library's default, while the box still holds
// the stand-in it opened with: the library's takes its place.
function takeLibraryStrength() {
  if (inpaintStrengthStandIn === null || inpaintLibraryStrength() === null) return;
  if (libraryStrengthInto(els.inpaintStrength, inpaintStrengthStandIn, inpaintFilled)) scheduleInpaintPreview(0);
  inpaintStrengthStandIn = null;
}

// The same for any box of the inpaint class's (the Outpaint dialog's too):
// the library's number in place of the stand-in the box still holds, and in
// the values a close compares with. Answers whether it wrote.
function libraryStrengthInto(box, standIn, filled) {
  const library = inpaintLibraryStrength();
  if (standIn === null || library === null || box.value !== String(standIn)) return false;
  box.value = String(library);
  if (filled) filled.strength = String(library);
  return true;
}

// Opens the dialog on `rec`, the image to inpaint. `from` is an inpainted
// record to start from instead - Reuse on one - with its mask, strength,
// seed, prompt and sampling. Otherwise the source's own prompt, sampling and
// seed, an empty mask and the starting strength - or, after a close that left
// a mask or settings unsent on this image, those, which it says. Focus lands
// on the picture, which is what the dialog is opened to paint. Answers
// whether it opened.
function openInpaint(rec, from = null) {
  if (!inpaintSupported && !from) {
    showError(inpaintNone(), true);
    return false;
  }
  inpaintReturn = lightboxPlace();
  if (els.lightbox.open) closeLightbox();
  showError('');
  // Whatever the last fill had in flight - an upload, a preview, a check -
  // answers for a mask and boxes that are not these: Start over reopens on
  // the same image, where the source alone does not tell them apart.
  const opened = ++inpaintOpenEpoch;
  clearTimeout(inpaintUploadTimer);
  clearTimeout(inpaintTimer);
  clearTimeout(inpaintCheckTimer);
  clearInterval(inpaintKeepTimer);
  inpaintUploadEpoch++;
  inpaintEpoch++;
  inpaintCheckEpoch++;
  const kept = !from ? inpaintKept.get(rec.id) : null;
  const draft = kept && kept.model === loadedModel ? kept : null;
  inpaintKept.delete(rec.id);
  inpaintSource = rec;
  inpaintFrom = from;
  inpaintRan = false;
  const start = from ?? rec;
  const foreign = modelsDiffer(start.model_id);
  els.inpaintTitle.textContent = from ? `Inpaint #${rec.id} as #${from.id} was made` : `Inpaint #${rec.id}`;
  writeInpaintSourceLine(rec);
  els.inpaintReused.hidden = !from;
  els.inpaintReused.textContent = from
    ? `Reusing #${from.id}'s mask${from.strength != null ? ` at strength ${from.strength}` : ''}, seed ${from.seed}`
      + (foreign && from.strength != null ? ` - its strength was ${checkpointName(from.model_id)}'s` : '')
    : '';
  // The picture: the PNG itself, since the mask is painted on its pixels.
  els.inpaintStage.style.setProperty('--ip-ratio', `${rec.width} / ${rec.height}`);
  if (els.inpaintImage.getAttribute('src') !== rec.url) els.inpaintImage.src = rec.url;
  els.inpaintCanvas.setAttribute('aria-label', `Mask over #${rec.id}`);
  resetMask(rec.width, rec.height, draft?.mask ?? null);
  inpaintMask = null;
  inpaintMaskRefusal = '';
  inpaintMaskState = 'none';
  inpaintStrengthStandIn = null;
  const carried = from && !foreign && from.strength != null;
  els.inpaintStrength.value = String(carried ? from.strength : inpaintStartStrength());
  if (!carried && inpaintLibraryStrength() === null) inpaintStrengthStandIn = +els.inpaintStrength.value;
  els.inpaintVersions.value = '1';
  els.inpaintSeed.value = start.seed;
  els.inpaintPrompt.value = start.prompt;
  els.inpaintNegative.value = start.negative_prompt ?? '';
  const d = lastDefaults;
  els.inpaintSteps.value = foreign ? (d?.steps ?? 28) : start.steps;
  els.inpaintCfg.value = foreign ? (d?.guidance_scale ?? 3.5) : start.guidance_scale;
  els.inpaintMu.value = foreign ? (d?.mu ?? '') : (start.mu ?? '');
  els.inpaintTrueCfg.value = !foreign && trueCfgSupported && start.true_cfg != null ? start.true_cfg : '';
  // The adapter carries over from this checkpoint's own records, as Refine's.
  const listed = [...els.lora.options].some((o) => o.value === start.lora);
  inpaintLora = !foreign && loraSupported && start.lora && listed
    ? { name: start.lora, scale: start.lora_scale ?? 1 } : null;
  els.inpaintCarried.textContent = inpaintLora
    ? `LoRA ${inpaintLora.name} at ${inpaintLora.scale}, as #${start.id} was made.`
    : start.lora ? `LoRA ${start.lora} is not applied: ${foreign ? 'it was for another checkpoint'
      : !loraSupported ? 'this installation cannot load adapters' : 'it is not in the adapter list now'}.` : '';
  els.inpaintMoreNote.textContent = foreign
    ? `made on ${checkpointName(start.model_id)}: steps and CFG are ${checkpointName(loadedModel)}'s`
    : `as #${start.id} was made`;
  els.inpaintMore.open = foreign;
  inpaintFilled = inpaintValues();
  if (draft) applyInpaintValues(draft.values);
  els.inpaintDraft.hidden = !draft;
  els.inpaintDraftText.textContent = draft ? `Your unsent mask and settings for #${rec.id}, as you left them.` : '';
  inpaintProblem = null;
  inpaintPreviewed = null;
  inpaintFailed('');
  els.inpaintPreview.textContent = '';
  els.inpaintPreview.hidden = false;
  els.inpaintStrength.removeAttribute('aria-invalid');
  inpaintChecked = null;
  renderInpaintCheck();
  syncInpaint();
  // Scrolled to the top only as it opens: Start over fills it in place, and
  // jumping the dialog would move its controls from under the pointer.
  if (!els.inpaintDialog.open) {
    els.inpaintDialog.showModal();
    els.inpaintDialog.scrollTop = 0;
  }
  els.inpaintCanvas.focus({ preventScroll: true });
  inpaintKeepTimer = setInterval(() => { void keepInpaintMask(); }, INPAINT_KEEP_MS);
  // A Reuse's mask is the stored one: loaded as the base, and its hash is the
  // mask until a stroke changes it - the same picture, so no upload. Kept
  // from the sweep once (`/keep`): it may be days old, and nothing else of
  // a Reuse restarts its settle window.
  if (from?.mask) {
    inpaintMaskState = 'loading';
    syncInpaint();
    const hash = from.mask;
    void loadMaskBase(`/api/masks/${hash}`).then((loaded) => {
      if (opened !== inpaintOpenEpoch || inpaintSource !== rec || inpaintFrom !== from) return;
      if (!loaded) {
        inpaintMaskState = 'refused';
        inpaintMaskRefusal = `#${from.id}'s mask could not be read; paint the part to repaint again.`;
      } else if (!maskStrokes.length && !maskStroke && !maskFeatherPx) {
        inpaintMask = { hash, width: rec.width, height: rec.height };
        inpaintMaskState = 'stored';
        inpaintMaskKeptAt = 0;
        clearMaskWait();
        maskMeasured = { empty: false, coverage: maskMeasured.coverage };
        void measureStoredMask(rec);
        void keepInpaintMask();
      } else {
        // Painted on, or feathered, while it loaded: a new mask, sent.
        inpaintMaskChanged();
        return;
      }
      syncInpaint();
      scheduleInpaintPreview(0);
    });
  } else if (draft && maskTouched()) {
    inpaintMaskChanged();
  }
  scheduleInpaintPreview(0);
  scheduleInpaintCheck(0);
  return true;
}

// How much a Reuse's stored mask covers, for the status line: measured as an
// export would measure it, without sending anything.
async function measureStoredMask(rec) {
  await exportMask();
  if (inpaintSource === rec) syncInpaint();
}

function writeInpaintSourceLine(rec, gone = null) {
  els.inpaintSourceLine.textContent = [
    `${rec.width}×${rec.height}`, `seed ${rec.seed}`, checkpointName(rec.model_id),
    rec.refine != null ? `itself ${rec.mask ? 'an inpaint' : 'a refine'} of #${rec.refine}` : '',
    gone ? `#${rec.id} (${gone === 'binned' ? 'in the recycle bin' : 'deleted'})` : '',
  ].filter(Boolean).join(' · ');
}

// A refusal of the source itself - deleted or binned from another tab while
// the dialog was open - in the page's words, as Refine says it.
function inpaintSourceGone(message, source = inpaintSource) {
  const gone = sourceGoneKind(message, source);
  if (!gone) return message;
  if (inpaintSource === source) writeInpaintSourceLine(source, gone);
  return sourceGoneWords(source, gone, 'inpaint');
}

function inpaintBoxes() {
  return {
    strength: els.inpaintStrength, versions: els.inpaintVersions, seed: els.inpaintSeed,
    prompt: els.inpaintPrompt, negative: els.inpaintNegative, steps: els.inpaintSteps,
    cfg: els.inpaintCfg, mu: els.inpaintMu, trueCfg: els.inpaintTrueCfg,
  };
}

function inpaintValues() {
  const values = Object.fromEntries(Object.entries(inpaintBoxes()).map(([key, el]) => [key, el.value]));
  return { ...values, more: els.inpaintMore.open };
}

function applyInpaintValues(values) {
  for (const [key, el] of Object.entries(inpaintBoxes())) {
    if (typeof values[key] === 'string') el.value = values[key];
  }
  els.inpaintMore.open = !!values.more;
}

// Start over: an empty mask and the record's own settings again.
function startInpaintOver() {
  const rec = inpaintSource;
  if (!rec || inpaintSending) return;
  inpaintKept.delete(rec.id);
  const from = inpaintFrom;
  // The dialog stays open: openInpaint fills it again in place.
  inpaintReturnKeep = inpaintReturn;
  openInpaint(rec, from);
  inpaintReturn = inpaintReturnKeep;
  inpaintReturnKeep = null;
}

// Reset forgets what the dialog keeps in this page, with the panel.
function forgetInpaint() {
  inpaintKept.clear();
}

// Images gone for good (forgetImages): what a close kept on them goes too,
// a canvas the source's size with it.
function forgetInpaintImages(ids) {
  for (const id of ids) inpaintKept.delete(id);
}

// A checkpoint switch: what was painted for one checkpoint's sampling is not
// submitted on another's unchanged, as Refine's pass is not.
function inpaintCheckpointChanged() {
  inpaintKept.clear();
  if (!inpaintSource) return;
  inpaintSource = null;
  inpaintReturn = null;
  els.inpaintDialog.close();
  showError('The checkpoint changed. Reopen Inpaint to use its new sampling settings.', true);
}

// ------------------------------------------------------------- the mask's upload

// A stroke has begun: the mask stored before it is no longer the one shown,
// and nor is the preview's estimate of it.
function inpaintMaskEditing() {
  clearTimeout(inpaintUploadTimer);
  inpaintUploadEpoch++;
  inpaintMask = null;
  inpaintMaskState = 'pending';
  inpaintMaskRefusal = '';
  els.inpaintPreview.textContent = '';
  els.inpaintPreview.classList.remove('warn');
  syncInpaintRun();
}

// A "wait for the mask" refusal said by a press, once the mask has landed.
function clearMaskWait() {
  if (/^Wait for the mask to (upload|load)\.$/.test(els.inpaintError.textContent)) inpaintFailed('');
}

// Keep the stored mask from the sweep for another window, as an upload of
// it would: on a Reuse, every INPAINT_KEEP_MS while the dialog is open, and
// before a press where it has been longer. Gone already (another tab's Empty
// bin, a CLI clear), the picture the dialog still holds goes up again - the
// same picture, the same hash. Answers whether the mask is stored now.
async function keepInpaintMask() {
  const source = inpaintSource;
  const mask = inpaintMask;
  if (!source || !mask || inpaintMaskState !== 'stored') return false;
  try {
    await api(`/api/masks/${mask.hash}/keep`, { method: 'POST' });
    if (inpaintMask === mask) inpaintMaskKeptAt = Date.now();
    return true;
  } catch {
    if (inpaintSource !== source || inpaintMask !== mask) return false;
    return resendInpaintMask();
  }
}

// The stored mask has gone from the server while the dialog still shows it:
// sent again, and answers whether it is stored now.
async function resendInpaintMask() {
  const source = inpaintSource;
  if (!source || !maskTouched()) return false;
  inpaintMaskEditing();
  await uploadInpaintMask();
  return inpaintSource === source && inpaintMaskState === 'stored';
}

// Whether a refusal says the mask has gone (swept, or its file lost), in
// any of the server's words for it.
function maskGoneRefusal(message) {
  return /\bmask [0-9a-f]{16}\b.*(has just gone|is missing)|no stored mask [0-9a-f]{16}|the file for mask [0-9a-f]{16}/i
    .test(message ?? '');
}

// Every change of the mask (inpaint-mask.js's maskEdited, and the feather):
// exported and uploaded a moment after the last one, so a burst of strokes
// is one upload. The same picture answers the same hash, so painting back to
// a mask already stored costs nothing on the server.
function inpaintMaskChanged() {
  if (!inpaintSource) return;
  inpaintMaskEditing();
  inpaintUploadTimer = setTimeout(() => { void uploadInpaintMask(); }, 300);
}

async function uploadInpaintMask() {
  const source = inpaintSource;
  if (!source || !els.inpaintDialog.open) return;
  // Mid-stroke: its end uploads.
  if (maskStroke) return;
  const epoch = ++inpaintUploadEpoch;
  if (!maskTouched()) {
    inpaintMaskState = 'none';
    maskMeasured = { empty: true, coverage: 0 };
    syncMaskTools();
    syncInpaint();
    scheduleInpaintPreview(0);
    return;
  }
  const out = await exportMask();
  if (epoch !== inpaintUploadEpoch || inpaintSource !== source) return;
  if (out.empty) {
    inpaintMaskState = 'empty';
    syncInpaint();
    scheduleInpaintPreview(0);
    return;
  }
  try {
    const answer = await api(`/api/masks?source=${source.id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'image/png' },
      body: out.blob,
    });
    if (epoch !== inpaintUploadEpoch || inpaintSource !== source) return;
    inpaintMask = answer;
    inpaintMaskState = 'stored';
    inpaintMaskKeptAt = Date.now();
    clearMaskWait();
  } catch (err) {
    if (epoch !== inpaintUploadEpoch || inpaintSource !== source) return;
    inpaintMaskState = 'refused';
    inpaintMaskRefusal = inpaintSourceGone(err.message, source);
  }
  syncInpaint();
  scheduleInpaintPreview(0);
}

// What holds Inpaint because of the mask, or ''.
function inpaintMaskProblem() {
  const name = inpaintSource ? `#${inpaintSource.id}` : 'the picture';
  switch (inpaintMaskState) {
    case 'none': return `Paint the part of ${name} to repaint first`;
    case 'empty': return `Nothing is masked: paint the part of ${name} to repaint`;
    case 'pending': return 'Wait for the mask to upload';
    case 'loading': return 'Wait for the mask to load';
    case 'refused': return inpaintMaskRefusal.replace(/\.$/, '');
    default: return inpaintMask ? '' : 'Wait for the mask to upload';
  }
}

// ------------------------------------------------------------- the dialog

// Everything that follows the boxes and the server: the seed line, the
// strength's note, the negative prompt's activation, Mu and True CFG where
// the checkpoint has them, the unavailable line, and whether Inpaint can run.
function syncInpaint() {
  const src = inpaintSource;
  if (!src) return;
  const versions = +els.inpaintVersions.value || 1;
  const text = els.inpaintStrength.value.trim();
  const sweep = text !== '' && !SINGLE_STRENGTH.test(text);
  const seedText = els.inpaintSeed.value.trim();
  const seed = seedText === '' ? null : +seedText;
  const seeds = seed === null
    ? (versions > 1 ? `${versions} seeds counting up from a random one` : 'a random seed')
    : versions > 1 ? `seeds ${seed}-${seed + versions - 1}` : `seed ${seed}`;
  els.inpaintSeedHint.textContent = `${seeds}${seed === src.seed ? ", from the source's" : ''}`
    + (sweep ? ', at every strength' : '');
  els.inpaintStrength.placeholder = String(inpaintStartStrength());
  els.inpaintStrengthNote.textContent = inpaintStrengthNote();
  syncPassSampling(inpaintSampling());
  els.inpaintUnavailable.hidden = inpaintSupported;
  els.inpaintUnavailable.textContent = inpaintSupported ? '' : inpaintUnavailableText();
  syncMaskStatus();
  syncInpaintRun();
  syncInpaintCheck();
}

// The dialog's sampling boxes, for syncPassSampling.
function inpaintSampling() {
  return {
    negative: els.inpaintNegative, negField: els.inpaintNegField, negNote: els.inpaintNegNote,
    cfg: els.inpaintCfg, mu: els.inpaintMu, muField: els.inpaintMuField,
    trueCfg: els.inpaintTrueCfg, trueCfgField: els.inpaintTrueCfgField,
  };
}

// The negative prompt, Mu and True CFG by the panel's rules, as Refine's,
// in an inpaint class's dialog (this one, and the Outpaint dialog's).
function syncPassSampling(f) {
  const trueCfg = f.trueCfg.value.trim() === '' ? 0 : +f.trueCfg.value;
  const reach = trueCfgSupported ? trueCfg : +f.cfg.value;
  const off = !negativeSupported || reach <= cfgThreshold;
  f.negative.disabled = off;
  f.negField.classList.toggle('disabled', off);
  const knob = trueCfgSupported ? 'True CFG' : 'CFG';
  f.negNote.textContent = !negativeSupported ? '(this checkpoint takes no negative prompt)'
    : !trueCfgSupported && !off ? `(runs at CFG > ${cfgThreshold}, no extra cost)`
      : !off ? TRUE_CFG_LIVE_NOTE : `(needs ${knob} > ${cfgThreshold} — ~2× slower)`;
  f.mu.placeholder = muPlaceholder();
  f.mu.disabled = !muSupported;
  f.muField.classList.toggle('disabled', !muSupported);
  if (!muSupported) f.mu.value = '';
  f.trueCfg.disabled = !trueCfgSupported;
  f.trueCfgField.classList.toggle('disabled', !trueCfgSupported);
  if (!trueCfgSupported) f.trueCfg.value = '';
}

// Held, with the reason as its title: first what holds Generate, then no
// inpaint here, then the mask (nothing painted, still uploading, refused),
// then the preview's refusal.
function syncInpaintRun() {
  if (!els.inpaintRun) return;
  const problem = submitHold()
    || (!inpaintSupported ? inpaintNone().replace(/\.$/, '').split('. ')[0]
      : inpaintMaskProblem() || inpaintProblem);
  els.inpaintRun.setAttribute('aria-disabled', String(!!problem));
  els.inpaintRun.title = problem || 'Run the inpaint (Ctrl+Enter)';
}

function inpaintChanged() {
  els.inpaintStrength.removeAttribute('aria-invalid');
  els.inpaintPreview.hidden = false;
  inpaintFailed('');
  syncInpaint();
  scheduleInpaintPreview();
}

// Enter in a one-line box runs, as Refine's; ArrowUp and ArrowDown step a
// single Strength by 0.05 (0.01 with Shift). The canvas has keys of its own
// (maskKeydown), which never reach here as these.
function inpaintKeydown(event) {
  if (keyboardEventHeld(event) || keyboardCommandModifier(event)) return;
  if (event.target === els.inpaintCanvas) return;
  if (event.key === 'Enter' && !event.shiftKey && event.target.matches('input:not([type="checkbox"]):not([type="range"])')) {
    event.preventDefault();
    if (!event.repeat) void runInpaint();
    return;
  }
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && event.target === els.inpaintStrength) {
    const text = els.inpaintStrength.value.trim();
    if (!SINGLE_STRENGTH.test(text)) return;
    event.preventDefault();
    const step = (event.shiftKey ? 0.01 : 0.05) * (event.key === 'ArrowUp' ? 1 : -1);
    els.inpaintStrength.value = String(Math.min(1, Math.max(0.01, +(+text + step).toFixed(2))));
    inpaintChanged();
  }
}

function inpaintFailed(message) {
  els.inpaintError.hidden = !message;
  els.inpaintError.textContent = message || '';
}

// The request the dialog stands for: the source at its own size, the stored
// mask by its hash, one strength as an ordinary request of `versions` images,
// several as a grid whose strength axis crosses that many seeds.
function inpaintRequest() {
  const num = (el) => (el.value.trim() === '' ? undefined : +el.value);
  const versions = +els.inpaintVersions.value || 1;
  const strength = els.inpaintStrength.value.trim();
  const single = strength === '' || SINGLE_STRENGTH.test(strength);
  const seed = els.inpaintSeed.value.trim();
  const base = {
    prompt: els.inpaintPrompt.value.trim(),
    negative_prompt: negativeSent(els.inpaintNegative),
    width: inpaintSource.width,
    height: inpaintSource.height,
    steps: num(els.inpaintSteps),
    guidance_scale: num(els.inpaintCfg),
    true_cfg: els.inpaintTrueCfg.disabled || els.inpaintTrueCfg.value.trim() === '' ? null : +els.inpaintTrueCfg.value,
    mu: els.inpaintMu.disabled || els.inpaintMu.value.trim() === '' ? null : +els.inpaintMu.value,
    lora: inpaintLora?.name ?? null,
    lora_scale: inpaintLora?.scale ?? 1,
    refine: inpaintSource.id,
    mask: inpaintMask?.hash ?? null,
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

function scheduleInpaintPreview(delay = 250) {
  clearTimeout(inpaintTimer);
  inpaintTimer = setTimeout(previewInpaint, delay);
}

// The server's reading of the inpaint, as Refine's preview: the image count,
// the strengths a sweep expands to, an estimate, and a refusal in the
// server's words - a strength that leaves no step, the mask gone. Nothing is
// asked while the mask is not stored: Inpaint says why it waits.
async function previewInpaint() {
  if (!inpaintSource || !els.inpaintDialog.open) return;
  const epoch = ++inpaintEpoch;
  const { preview, sweep } = inpaintRequest();
  const asked = JSON.stringify(preview);
  els.inpaintPreview.classList.remove('warn');
  inpaintProblem = null;
  if (!inpaintSupported || !preview.mask) {
    els.inpaintPreview.textContent = '';
    inpaintPreviewed = asked;
    syncInpaintRun();
    return;
  }
  if (!preview.prompt) {
    inpaintProblem = 'Enter a prompt: the masked part is denoised under one.';
    els.inpaintPreview.classList.add('warn');
    els.inpaintPreview.textContent = inpaintProblem;
    inpaintPreviewed = asked;
    syncInpaintRun();
    return;
  }
  try {
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(preview),
    });
    if (epoch !== inpaintEpoch) return;
    const strengths = p.values.strength ?? [];
    const steps = p.values.steps?.[0];
    const run = !sweep && typeof steps === 'number' && strengths.length === 1
      ? ` · about ${Math.max(1, Math.round(steps * strengths[0]))} of ${steps} steps run` : '';
    const single = !sweep && strengths.length === 1 ? strengths[0] : null;
    const keeps = single === null ? '' : single >= 1 ? ' · keeps nothing of the masked part'
      : single >= 0.9 ? ' · keeps almost nothing of the masked part' : '';
    // Where the pass starts, read off the loaded schedule as Refine's is
    // (`start_sigma`): Z-Image's inpaint class answers; SDXL's schedule has
    // no noise fraction, and nothing answers before a load.
    const noise = single !== null && typeof p.start_sigma === 'number'
      ? ` · starts at ${Math.round(p.start_sigma * 100)}% noise` : '';
    els.inpaintPreview.classList.toggle('warn', !!p.duplicates);
    els.inpaintPreview.textContent = plural(p.images, 'image')
      + ` · strength ${axisText(strengths)}${noise}${run}${keeps}`
      + (p.seconds ? ` · about ${humanTime(p.seconds)}` : '')
      + (p.duplicates ? ` · ${p.duplicates} would be identical to another` : '')
      + (sweep ? '\nOpens as a sheet when it is done.' : '');
  } catch (err) {
    if (epoch !== inpaintEpoch) return;
    // The adapter gone: asked again without it, the refusal said instead.
    if (await dropInpaintLora(err.message)) return previewInpaint();
    // The mask swept while the dialog sat open: the picture it holds goes
    // up again, and that upload asks the preview again.
    if (maskGoneRefusal(err.message) && await resendInpaintMask()) return undefined;
    if (epoch !== inpaintEpoch) return;
    inpaintProblem = inpaintSourceGone(err.message);
    els.inpaintPreview.classList.add('warn');
    els.inpaintPreview.textContent = inpaintProblem;
  }
  inpaintPreviewed = asked;
  syncInpaintRun();
}

// An adapter removed or renamed since the dialog opened, named by a refusal:
// taken out of this inpaint, said, as Refine's dropRefineLora does. Answers
// whether it went.
async function dropInpaintLora(message, source = inpaintSource) {
  const name = inpaintLora?.name;
  if (!name || !message?.includes(name)) return false;
  const body = await refreshLoras();
  if (!body || inpaintSource !== source || inpaintLora?.name !== name) return false;
  const entry = (body.loras ?? []).find((e) => e.name === name);
  if (entry && !entry.detail) return false;
  inpaintLora = null;
  inpaintLoraDrops++;
  els.inpaintCarried.textContent = `LoRA ${name} is not applied: it is not in the adapter list now.`;
  inpaintFailed(`${name} ${entry ? 'no longer fits this checkpoint' : 'is no longer in the LoRA folder'}, `
    + 'so it was taken out of this inpaint; Inpaint runs it without the adapter.');
  return true;
}

// Inpaint: the job queues like Refine's, and the dialog closes once the
// server has taken it. A sweep opens as its sheet when done.
async function runInpaint() {
  if (!inpaintSource || !els.inpaintDialog.open || submitting || inpaintWaiting) return;
  // A mask stored long ago is kept first (or sent again, where it has gone):
  // a sweep would otherwise refuse it at submit.
  if (inpaintMaskState === 'stored' && Date.now() - inpaintMaskKeptAt > INPAINT_KEEP_MS) {
    inpaintWaiting = true;
    try { await keepInpaintMask(); } finally { inpaintWaiting = false; }
    if (!inpaintSource || !els.inpaintDialog.open || submitting) return;
  }
  // The preview of what the boxes hold answers first, as Refine's: it is
  // what refuses a strength past 1 or one that leaves no step. One that took
  // the adapter out (dropInpaintLora) has said so, and this press stops there.
  const drops = inpaintLoraDrops;
  while (inpaintPreviewed !== JSON.stringify(inpaintRequest().preview)) {
    clearTimeout(inpaintTimer);
    inpaintWaiting = true;
    try { await previewInpaint(); } finally { inpaintWaiting = false; }
    if (!inpaintSource || !els.inpaintDialog.open || submitting || inpaintLoraDrops !== drops) return;
  }
  if (els.inpaintRun.getAttribute('aria-disabled') === 'true') {
    const held = submitHold();
    if (held) inpaintFailed(`${held}.`);
    else refuseInpaint(els.inpaintRun.title);
    if (serverOffline) void refreshStatus();
    return;
  }
  const { path, body, sweep } = inpaintRequest();
  const source = inpaintSource;
  const strength = els.inpaintStrength.value.trim() || String(inpaintStartStrength());
  inpaintSending = true;
  let accepted = false;
  const settle = () => {
    inpaintSending = false;
    els.inpaintRun.textContent = busy ? 'Inpaint (queue)' : 'Inpaint';
  };
  const sent = startJob(path, body, {
    sheet: sweep,
    panel: false,
    slot: els.inpaintError,
    placeholder: {
      src: source.url ?? source.thumb_url,
      width: body.width,
      height: body.height,
      label: `Inpainting #${source.id} · strength ${strength}`,
    },
    onAccepted: () => {
      accepted = true;
      settle();
      if (inpaintSource !== source || !els.inpaintDialog.open) return;
      inpaintRan = true;
      els.inpaintDialog.close();
      requestAnimationFrame(() => {
        if (document.activeElement && document.activeElement !== document.body) return;
        handJobFocus(els.newImage.hidden ? els.generate : els.newImage);
      });
    },
  });
  if (inpaintSending) els.inpaintRun.textContent = 'Sending…';
  await sent;
  if (accepted) return;
  settle();
  const refused = els.inpaintError.textContent;
  const said = inpaintSourceGone(refused, source);
  if (said !== refused) inpaintFailed(said);
  // An adapter gone since the last preview: taken out, said, and the
  // preview asked again without it.
  else if (await dropInpaintLora(refused, source)) scheduleInpaintPreview(0);
  // The mask swept since: sent again, and the press is the user's to repeat.
  else if (inpaintSource === source && maskGoneRefusal(refused) && await resendInpaintMask()) {
    inpaintFailed('The mask had gone from the server and was sent again; press Inpaint to run it.');
  }
  if (!els.inpaintDialog.open && els.inpaintError.textContent) showError(els.inpaintError.textContent);
}

// A press on a held Inpaint says why, in the preview's place and at its size,
// so the button does not move under the pointer; Strength is marked where
// the sentence is about it. A mask that is not ready says so on the line too.
function refuseInpaint(reason) {
  if (inpaintSupported && reason === inpaintProblem && els.inpaintPreview.textContent === reason) {
    els.inpaintPreview.textContent = '';
    els.inpaintPreview.classList.remove('warn');
    els.inpaintPreview.hidden = true;
    inpaintFailed(reason);
    if (/strength/i.test(reason)) {
      els.inpaintStrength.setAttribute('aria-invalid', 'true');
      els.inpaintStrength.focus({ preventScroll: true });
    }
    return;
  }
  inpaintFailed(/[.!?]$/.test(reason) ? reason : `${reason}.`);
}

function closeInpaint() {
  if (!inpaintSending) els.inpaintDialog.close();
}

function inpaintCancelEvent(event) {
  if (inpaintSending) event.preventDefault();
}

// Closed by Inpaint, Cancel, the cross or Escape. A close that ran nothing
// keeps the mask's strokes and the boxes for the next open of the same image
// on this checkpoint - in page memory - and, opened from the lightbox, gives
// the lightbox back where it was; where it ran, focus goes to the source's
// card.
function inpaintClosed() {
  const src = inpaintSource;
  const ran = inpaintRan || inpaintSending;
  if (maskStroke) endStroke();
  if (src && !ran) {
    const values = inpaintValues();
    const edited = Object.keys(inpaintBoxes()).some((key) => values[key] !== inpaintFilled?.[key]);
    if (edited || maskStrokes.length || maskRedo.length) {
      inpaintKept.set(src.id, { model: loadedModel, values, mask: maskSnapshot() });
      while (inpaintKept.size > INPAINT_KEPT) inpaintKept.delete(inpaintKept.keys().next().value);
    }
  }
  inpaintSource = null;
  inpaintFrom = null;
  inpaintLora = null;
  inpaintEpoch++;
  inpaintCheckEpoch++;
  inpaintUploadEpoch++;
  inpaintOpenEpoch++;
  clearTimeout(inpaintTimer);
  clearTimeout(inpaintCheckTimer);
  clearTimeout(inpaintUploadTimer);
  clearInterval(inpaintKeepTimer);
  els.inpaintBrush.hidden = true;
  const back = inpaintReturn;
  inpaintReturn = null;
  if (!back || els.lightbox.open) return;
  if (!ran) { returnToLightbox(back); return; }
  const card = [...document.querySelectorAll(`#gallery .card[data-id="${back.rec.id}"] img`)].find(isShown);
  handJobFocus(card ?? els.generate);
}

// ------------------------------------------------------------- Reuse

// An inpainted record holds its source's id and its mask's hash: the source
// asked for, so the dialog has its picture, and opened with the mask loaded.
// A source that has gone cannot be inpainted again - the prompt and settings
// go into the panel instead, at the panel's own size.
async function reuseInpaint(rec) {
  let source;
  try {
    // The bin's rows too, so a binned source is offered a Restore rather
    // than said to be gone for good.
    source = await api(`/api/images/${rec.refine}?binned=true`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) { showError(err.message); return; }
    reuseIntoPanel(rec, false,
      `#${rec.refine}, the source this was inpainted from, is no longer in the gallery, `
      + 'so the inpaint cannot be made again. The size is left as it was.');
    return;
  }
  if (source.binned_at) { showError(sourceGoneWords(source, 'binned', 'inpaint'), true); return; }
  openInpaint(source, rec);
}

// ------------------------------------------------------------- the prompt check

function scheduleInpaintCheck(delay = 400) {
  clearTimeout(inpaintCheckTimer);
  inpaintCheckTimer = setTimeout(checkInpaintPrompt, delay);
}

// The prompt check on the inpaint's own prompt, as the Refine dialog's: cold,
// off the image queue, advice only - its failures are silent.
async function checkInpaintPrompt() {
  if (!inpaintSource || !els.inpaintDialog.open) return;
  const epoch = ++inpaintCheckEpoch;
  const { body } = inpaintRequest();
  if (!body.prompt) { inpaintChecked = null; renderInpaintCheck(); return; }
  const asked = inpaintCheckAsked(body);
  let result = null;
  try {
    result = await api('/api/prompt/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...body, mask: undefined, strength: undefined, strength_axis: undefined, seeds: undefined,
      }),
    });
  } catch { /* advice only */ }
  if (epoch !== inpaintCheckEpoch) return;
  inpaintChecked = result ? { ...asked, body: result } : null;
  renderInpaintCheck();
}

function inpaintCheckAsked(body) {
  return { prompt: body.prompt, negative: body.negative_prompt ?? '', negativeOff: els.inpaintNegative.disabled };
}

// Refine's lines (refine-prompt.js): where each encoder stops reading, with
// Show selecting the rest of this dialog's box, and a rule's whole-field
// rewrite with Apply.
function renderInpaintCheck() {
  renderPassCheck({
    box: els.inpaintCheck, said: els.inpaintCheckSaid, checked: inpaintChecked,
    select: selectInpaintFrom, apply: applyInpaintFinding,
  });
  syncInpaintCheck();
}

// Show and Apply are dead while a box holds other text than was checked, and
// on a greyed negative prompt, as Refine's.
function syncInpaintCheck() {
  if (!inpaintSource) return;
  const { body } = inpaintRequest();
  syncPassCheck(els.inpaintCheck, inpaintChecked, inpaintCheckAsked(body), els.inpaintNegative.disabled);
}

function selectInpaintFrom(field, text, cut) {
  selectPassFrom(els.inpaintPrompt, els.inpaintNegative, els.inpaintMore, field, text, cut);
}

// A rule's rewrite, written the way typing writes (replaceBoxText), so
// Ctrl+Z gives back what the box held; the fold opens first, since a box in
// a closed one takes no focus.
function applyInpaintFinding(finding) {
  const el = finding.field === 'negative_prompt' ? els.inpaintNegative : els.inpaintPrompt;
  if (el.disabled) return;
  els.inpaintMore.open = true;
  replaceBoxText(el, finding.suggestion);
  inpaintChanged();
  scheduleInpaintCheck(0);
}
