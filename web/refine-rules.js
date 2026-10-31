// AuraDream's page script: the refine pass's shared rules - its bounds and
// sizes, the dialog's remembered settings, the strength, and a gone source's
// words - which the Refine, Inpaint, Outpaint and Edit dialogs, Mass refine,
// Then refine, the lightbox and the panel read.
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
