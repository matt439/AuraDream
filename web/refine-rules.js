// AuraDream's page script: the refine pass's shared rules - its bounds and
// sizes, the dialog's remembered settings, the strength, and a gone source's
// words - which the Refine, Inpaint, Outpaint and Edit dialogs, Mass refine,
// Then refine, the lightbox and the panel read - and what the Inpaint and
// Outpaint dialogs share: the unavailable words, the library start strength
// and its note, the sampling boxes' rules and the masked area's request and
// words.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

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
const REFINE_KEY = 'auradream.refine.v1';

// One number, or a sweep in the grid's syntax (`0.3,0.4,0.5`, `0.2-0.5x0.1`)
// that the server parses: one strength is an ordinary request, several a grid
// with a strength axis.
const SINGLE_STRENGTH = /^(\d+(\.\d*)?|\.\d+)$/;

// ArrowUp and ArrowDown in a pass dialog's Strength: a single strength
// (`text`, the box's) stepped by 0.05, 0.01 with Shift, held to 0.01-1; a
// sweep is left to the caret. Answers whether it stepped.
function stepStrength(event, box, text = box.value.trim()) {
  if (!SINGLE_STRENGTH.test(text)) return false;
  event.preventDefault();
  const step = (event.shiftKey ? 0.01 : 0.05) * (event.key === 'ArrowUp' ? 1 : -1);
  box.value = String(Math.min(1, Math.max(0.01, +(+text + step).toFixed(2))));
  return true;
}

// What holds a pass dialog's run while a submit is in flight - its own or
// the panel's: the title, and the words a press says.
const PASS_IN_FLIGHT = 'Wait for the last submit to be taken';

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

// What a pass at w x h keeps of a sw x sh source, as fractions of it:
// refine.fitted's arithmetic - scaled to cover, each side rounded as Python
// rounds (half to even), then the middle cropped out - so the frame drawn on
// the source and the size line's numbers are the pass's own.
function refineKept(sw, sh, w, h, anchor = 'center') {
  if (sw === w && sh === h) return { x: 0, y: 0, w: 1, h: 1 };
  const even = (v) => { const r = Math.round(v); return r - v === 0.5 && r % 2 ? r - 1 : r; };
  const cover = Math.max(w / sw, h / sh);
  const rw = Math.max(w, even(sw * cover));
  const rh = Math.max(h, even(sh * cover));
  anchor ||= 'center';
  const x = anchor.includes('left') ? 0 : anchor.includes('right') ? rw - w : Math.floor((rw - w) / 2);
  const y = anchor.includes('top') ? 0 : anchor.includes('bottom') ? rh - h : Math.floor((rh - h) / 2);
  return { x: x / rw, y: y / rh, w: w / rw, h: h / rh };
}

function refineSizeProblem(w, h) {
  if (!(w >= 256 && h >= 256)) return 'each side needs at least 256 pixels';
  if (w % 16 || h % 16) return 'sides must be multiples of 16';
  if (!refineFits([w, h])) {
    return `past the ${REFINE_MAX_SIDE}-pixel side and ${(REFINE_MAX_PIXELS / 1e6).toFixed(1)} MP a refine pass takes`;
  }
  return null;
}

// An adapter removed or renamed since a pass dialog opened, named by a
// refusal: taken out of the pass and said. `pass` is the dialog's own -
// `lora()`/`source()` read its lets as they are now, `drop()` clears the one
// and counts the drop, `carried` the line naming it, `failed` the error line,
// `noun` what the words call the pass and `run` its button. Answers whether
// it went.
async function dropPassLora(message, source, pass) {
  const name = pass.lora()?.name;
  if (!name || !message?.includes(name)) return false;
  const body = await refreshLoras();
  if (!body || pass.source() !== source || pass.lora()?.name !== name) return false;
  const entry = (body.loras ?? []).find((e) => e.name === name);
  if (entry && !entry.detail) return false;
  pass.drop();
  pass.carried.textContent = `LoRA ${name} is not applied: it is not in the adapter list now.`;
  pass.failed(`${name} ${entry ? 'no longer fits this checkpoint' : 'is no longer in the LoRA folder'}, `
    + `so it was taken out of this ${pass.noun}; ${pass.run} runs it without the adapter.`);
  return true;
}

// Whether a refusal says `source` has gone: 'deleted', 'binned' or null.
function sourceGoneKind(message, source) {
  const deleted = /^no gallery image with id (\d+)\.?$/.exec(message ?? '');
  const binned = /^source image (\d+) is in the recycle bin; /i.exec(message ?? '');
  const match = deleted ?? binned;
  if (!source || !match || +match[1] !== source.id) return null;
  return binned ? 'binned' : 'deleted';
}

// The page's words for it, by the dialog's verb ("refine", "inpaint",
// "outpaint", "edit"): "refine" is "refined", never "refineed".
function sourceGoneWords(source, gone, verb) {
  const done = verb.endsWith('e') ? `${verb}d` : `${verb}ed`;
  return gone === 'binned' ? `#${source.id} is in the recycle bin; restore it to ${verb} it.`
    : `#${source.id} has been deleted, so it cannot be ${done}.`;
}

// ---- what the inpaint class's dialogs share (Inpaint and Outpaint)

// The families this app inpaints, for the sentence that says where it is
// not available.
const INPAINT_FAMILIES = 'The FLUX.1, SDXL and Z-Image checkpoints have it.';

// Only the masked area (`mask_crop`, backend/mask_crop.py): the rule, the
// padding the box starts at and the working tier, as the server spells them.
// The box itself is the server's to work out (`/api/grid/preview`'s
// `mask_crop`); the page only draws it.
const MASK_CROP_RULE = 'bbox16';
const MASK_CROP_PAD = 32;
const MASK_CROP_TIER = 1;

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

// Any box of the inpaint class's (the Inpaint and Outpaint dialogs'): the
// library's number in place of the stand-in the box still holds, and in the
// values a close compares with. Answers whether it wrote.
function libraryStrengthInto(box, standIn, filled) {
  const library = inpaintLibraryStrength();
  if (standIn === null || library === null || box.value !== String(standIn)) return false;
  box.value = String(library);
  if (filled) filled.strength = String(library);
  return true;
}

// The `mask_crop` a dialog's tick and Padding box ask for, or null while
// unticked: the box's number, or the default padding while it is empty. The
// server checks the padding (0-256 in steps of 8) and says why it refuses.
function maskCropAsked(check, padBox) {
  if (!check.checked) return null;
  const text = padBox.value.trim();
  return { rule: MASK_CROP_RULE, pad: text === '' ? MASK_CROP_PAD : +text, tier: MASK_CROP_TIER };
}

// The preview's words for the box: its size with the padding in it, the
// working size and how much larger (or smaller) that is.
function maskCropText(area, pad) {
  const [x1, y1, x2, y2] = area.box;
  const [w, h] = area.working;
  const scale = area.scale >= 1 ? `${area.scale.toFixed(1)}× up` : `${area.scale.toFixed(2)}× down`;
  return ` · the masked area and its ${pad} px padding, ${x2 - x1}×${y2 - y1}, at ${w}×${h} (${scale}), pasted back`
    + ' · live preview shows the area';
}

// The negative prompt, Mu and True CFG by the panel's rules, as Refine's,
// in an inpaint class's dialog (the Inpaint and the Outpaint dialog's).
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
