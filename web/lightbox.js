// Local Diffusion's page script: the lightbox - its view, zoom, layout, walk and
// gestures. A record's facts are facts.js's; the comparison's controls, divider
// and pass stepping lightbox-compare.js's, across CFG lightbox-cfg.js's.
// Declarations only; state.js has the load-order contract these files share.
'use strict';

// ----------------------------------------------------------------- lightbox

// `origin` is where an open from a closed lightbox came from, where it is not
// the strip it walks: the viewer's image walks the gallery from it, and
// #recent this page's earlier results (viewer.js) - a close hands focus back
// there (dismissLightbox), not to a card a screen down.
let lbOrigin = null;
function openLightbox(rec, list, origin = null) {
  if (!els.lightbox.open) lbOrigin = origin;
  lbList = list && list.includes(rec) ? list : [rec];
  lbIndex = lbList.indexOf(rec);
  renderLightbox();
}

// The view: a scale relative to the fit size (1 = fit) and the image centre's
// offset from its viewport's centre in CSS pixels. Never below fit or 1:1,
// whichever is smaller: on a QHD screen every preset up to about 2 MP fits
// larger than 1:1, and a floor at fit left the 1:1 a sharpening sweep is
// judged at out of reach there. It is kept across records whose shape agrees,
// so the arrow keys show the same part of the picture at the same size across
// the cells of a sweep - the whole point of zooming in on a steps or mu axis -
// and across a pass and the source it was scaled up from. Agrees within 1%,
// not as a rounded key: the 16-pixel grid puts 2048x1152 and 1360x768 apart
// in the third decimal. A record of another shape starts at fit; so does
// every fresh open.
const lbView = { scale: 1, x: 0, y: 0, aspect: null };
const ZOOM_STEP = 1.25;

// A refine pass beside the image it was made from: the source under the pass
// and shown left of a divider (`swipe`), or the two side by side (`side`), on
// the one view - the same crop at the same magnification, which no walk gave:
// the source was rarely next to its pass, and a step reset the zoom. The
// source is drawn as `refine.fitted` cropped it (`refineKept`), so a reshaped
// pass is compared with the part of the source it kept. On for the record it
// was turned on for (`id`); a step to another image ends it, and the mode is
// kept for the next. `kind` says which comparison it is: 'source', a pass
// against the image it was refined from, or 'cfg', any image against another
// that differs from it in CFG alone (lightbox-cfg.js), whose `ladder` is the
// set of them sorted by CFG, built once when it is turned on. `source` is the
// other image either way - the partner - drawn on the layer over this one.
const lbCompare = { id: null, kind: 'source', mode: 'swipe', split: 0.5, source: null, ladder: [], range: [0, 0] };
// A line for the comparison's note that the state does not say - a held
// press's reason, a partner that has gone - for the image it was said on
// ({id, text}), until another image is drawn or a comparison is turned on.
let lbCompareSaid = null;
// Side by side's title while it can be pressed (layoutLightbox's): which way
// the halves go. Held, the title is the reason instead.
let lbSidesTitle = '';
// Sources by id, fetched once a session: a record, or null for one the
// gallery no longer has.
const lbSources = new Map();
// Each image's count of passes made from it, asked once a session.
const lbPasses = new Map();
let lbPassesTimer;
// The record the side column's facts were drawn for; a re-render of the same
// one (a star, a status) keeps them, and the focus in them.
let lbDrawn = null;
// A click beside an image shown below fit closes a moment later, unless it was
// the first half of a double-click: the picture is smaller than fit there, so
// the double-click that gives fit back can start on the backdrop.
let lbCloseTimer;
// Where a Refine opened from the lightbox goes back to when it is closed
// without running (`refineClosed`): set by openRefine, which closes the
// lightbox for its dialog.
let refineReturn = null;
// The Seed row's Use seed note ({id, text}) while the lightbox's line holds it.
let lbSeedNote = null;
// A failed download belongs to the image it was requested for.
let lbDownloadError = null;
const LB_SIDE_GAP = 8;
// How far the arrows' circles reach in from the stage's edges: style.css's
// --lb-nav (40 px, 52 from 1800 px) and its 10 px inset, measured off the ›
// circle as drawn, so the two below follow the circle wherever its size is
// set - a constant for the 40 px one put side by side's halves under the
// wide tier's. The circle is in the same place as a strip's and alone
// (style.css), so either reads the same: 50 px, or 62.
function lbNavReach() {
  const circle = els.lbNext.firstElementChild.getBoundingClientRect();
  return Math.max(0, els.lbStage.getBoundingClientRect().right - circle.left);
}
// Side by side, the halves start past the circles and a gap: 58 px, or 70.
function lbNavRoom(reach = lbNavReach()) { return reach + 8; }

// The boxes the picture is fitted into, in the stage's pixels: the whole
// stage, or side by side its two halves, each showing the same crop - inside
// the arrows, whose circles sat over the source's left edge and the pass's
// right one there (alone, a picture is centred clear of them). Left and
// right, or the source above the pass where that fits the picture larger
// (`lbStacked`). `reach` is lbNavReach's, where the caller has read it
// already; `sw` and `sh` a stage of another size than the one drawn
// (lbFactsRoom); `stacked` the split, where the caller has decided it.
function lbPorts(reach, sw = els.lbStage.clientWidth, sh = els.lbStage.clientHeight, stacked = undefined) {
  if (lbCompare.id === null || lbCompare.mode !== 'side') return [{ x: 0, y: 0, w: sw, h: sh }];
  return sidePorts(reach, sw, sh, stacked ?? lbStacked());
}
function sidePorts(reach, sw, sh, stacked) {
  const inset = Math.min(lbNavRoom(reach), sw / 8);
  if (stacked) {
    const w = sw - 2 * inset, h = (sh - LB_SIDE_GAP) / 2;
    return [{ x: inset, y: 0, w, h }, { x: inset, y: h + LB_SIDE_GAP, w, h }];
  }
  const w = (sw - 2 * inset - LB_SIDE_GAP) / 2;
  return [{ x: inset, y: 0, w, h: sh }, { x: inset + w + LB_SIDE_GAP, y: 0, w, h: sh }];
}

// The stage as it is with the facts at six lines: under 901 px they take
// what the picture leaves of it (lbFactsRoom), and this is the stage before
// they did, whatever the picture before this one left them.
function lbBaseStage() {
  const sw = els.lbStage.clientWidth;
  let sh = els.lbStage.clientHeight;
  if (lbNarrow()) sh += Math.max(0, els.lbMeta.clientHeight - lbSixLines());
  return [sw, sh];
}
const lbNarrow = () => matchMedia('(max-width: 900px)').matches;
const lbSixLines = () => parseFloat(getComputedStyle(els.lbMeta).fontSize) * 6;

// Whether side by side's halves go one above the other: where the window is
// under 901 px, or the stage taller than it is wide, and the halves stacked
// fit `rec` at least LB_STACK_GAIN larger than left and right do. At 600x900
// left and right gave a square pass and its source 213 px each of a 549x626
// stage, a 7:4 one 213x121, where stacked gives them 309 and 433; a portrait
// pass keeps left and right, 213x372, where stacked would give it 177x309.
// The gain is asked because stacked halves take the stage's height, which
// was the facts' letterbox (lbFactsRoom): a square pass at 800x900 would
// gain 15 px a side, 313 to 328, and lose every fact under Size. A
// landscape stage from 901 px keeps left and right whatever the picture:
// the layout side by side was designed in. Decided on the stage with the
// facts at six lines (lbBaseStage), never on what the facts left of it:
// decided there, a landscape pass's room was worked out for stacked halves,
// the stage it left was wider than tall, and the halves were drawn left and
// right at 213 px over a band of nothing.
const LB_STACK_GAIN = 1.125;
function lbStacked(rec = lbList[lbIndex]) {
  if (!rec) return false;
  const [sw, sh] = lbBaseStage();
  if (!lbNarrow() && sh <= sw) return false;
  const reach = lbNavReach();
  const width = (stacked) => fitSize(rec, sidePorts(reach, sw, sh, stacked))[0];
  return width(true) >= LB_STACK_GAIN * width(false);
}

// Fit is "contain": the largest size that shows the whole image inside its
// viewport, taken from the record's own dimensions rather than the decoded
// image, so the layout is right before the PNG has arrived.
function fitSize(rec, ports = lbPorts()) {
  const { w, h } = ports[0];
  const k = Math.min(w / rec.width, h / rec.height);
  return [rec.width * k, rec.height * k];
}

// The scale at which one image pixel is one *device* pixel. Dividing by the
// devicePixelRatio is load-bearing: at 125-150% Windows scaling a plain
// natural-size image is a 1.25-1.5x resample, which is precisely the smoothing
// a sharpening sweep is being judged on.
function oneToOneScale(rec, ports) {
  return (rec.width / window.devicePixelRatio) / fitSize(rec, ports)[0];
}

// Between fit or 1:1, whichever is smaller, and 8x device pixels, past which
// the picture is squares.
function clampScale(rec, scale) {
  const one = oneToOneScale(rec);
  return Math.min(Math.max(1, one * 8), Math.max(Math.min(1, one), scale));
}

const placeBox = (el, box) => Object.assign(el.style, {
  width: `${box.w}px`, height: `${box.h}px`, left: `${box.left}px`, top: `${box.top}px`,
});

// A clip-path for an element at `box` showing only what lies inside `rect`
// (both in the stage's pixels): how a half, or the swipe's side, is cut. The
// clipped part takes no clicks either, so a click there is the backdrop's.
function clipBox(box, rect) {
  const t = Math.max(0, rect.top - box.top), l = Math.max(0, rect.left - box.left);
  const b = Math.max(0, box.top + box.h - rect.bottom), r = Math.max(0, box.left + box.w - rect.right);
  return `inset(${t}px ${r}px ${b}px ${l}px)`;
}

function layoutLightbox() {
  const rec = lbList[lbIndex];
  if (!rec || !els.lightbox.open) return;
  // First, as it moves the stage's edge: the facts under a one-column stage
  // take what the picture leaves of it. It writes only when that changes -
  // another image, another window - so what follows still holds:
  // everything read off the page is read here, before the first write, so a
  // wheel step, a pan move or a drag of the divider lays the page out once.
  // The split is decided before, on the stage as it is with the facts at
  // six lines, so the room and the halves agree (lbStacked).
  // Whether a press of Side by side would stack them, for its title, is
  // decided the same way before the comparison is on (read before any write).
  // Any image can be compared (CFG, lightbox-cfg.js), so the split is decided
  // for every one.
  const comparing = lbCompare.id === rec.id;
  const split = lbStacked(rec);
  const stacked = comparing && split;
  lbFactsRoom(rec, stacked);
  const reach = lbNavReach();
  const ports = lbPorts(reach, undefined, undefined, stacked);
  const [fw, fh] = fitSize(rec, ports);
  const one = oneToOneScale(rec, ports);
  const w = fw * lbView.scale, h = fh * lbView.scale;
  const { w: pw, h: ph } = ports[0];
  // An image smaller than its viewport on an axis is centred on it; one
  // larger can be panned until its edge meets the viewport's, and no further.
  const maxX = Math.max(0, (w - pw) / 2), maxY = Math.max(0, (h - ph) / 2);
  lbView.x = Math.min(maxX, Math.max(-maxX, lbView.x));
  lbView.y = Math.min(maxY, Math.max(-maxY, lbView.y));
  // The picture's box in a viewport; the pass is the last one's - or, in a
  // CFG comparison where this image has the lower CFG, the first one's
  // (`lbSwapped`): the lower CFG is always on the left.
  const boxIn = (port) => ({
    left: port.x + (port.w - w) / 2 + lbView.x, top: port.y + (port.h - h) / 2 + lbView.y, w, h,
  });
  const rectOf = (port) => ({ left: port.x, right: port.x + port.w, top: port.y, bottom: port.y + port.h });
  const minePort = lbSwapped(rec) ? ports[0] : ports.at(-1);
  const pass = boxIn(minePort);
  placeBox(els.lbImg, pass);
  els.lbImg.style.clipPath = ports.length > 1 ? clipBox(pass, rectOf(minePort)) : '';
  layoutCompare(rec, ports, pass, boxIn, rectOf);
  // The mode keeps its name where the halves are stacked; its title says
  // which way they go, or would on a press (lbStacked): before the press it
  // said "left of the pass" where the press stacked them. In the words of
  // the comparison on, or of the one a press would turn on.
  const cfg = comparing ? lbCompare.kind === 'cfg' : rec.refine == null;
  const [first, second] = cfg ? ['The lower CFG', 'the higher'] : ['The source', 'the pass'];
  const sides = split ? `${first} above ${second}, each fitted to half the stage: larger here than left and right`
    : `${first} left of ${second}, each fitted to half the stage`;
  // Held, its title is the reason instead (syncCompare), put back from here.
  lbSidesTitle = sides;
  if (els.lbSideBySide.getAttribute('aria-disabled') !== 'true' && els.lbSideBySide.title !== sides) {
    els.lbSideBySide.title = sides;
  }
  // Tested in this order: a wheel stop between 1:1 and fit is neither.
  const percent = Math.round((lbView.scale / one) * 100);
  writeZoom(Math.abs(lbView.scale - one) < 1e-3 ? '1:1' : lbView.scale === 1 ? `fit · ${percent}%` : `${percent}%`);
  els.lbStage.classList.toggle('zoomed', lbView.scale > 1);
  // The arrows: at fit, strips as wide as the backdrop beside the picture
  // (less a margin, 48-160 px) over the middle 60% of its height, so a press
  // a little off the circle still steps rather than closing - and never over
  // the picture, nor the corners, which still close. Anywhere else, or with
  // no backdrop beside it, just the circle: the stage beside the picture has
  // a gesture of its own there - zoomed in the pan, below fit the
  // double-click that gives fit back - and a strip would take it. A disabled
  // arrow is its circle alone too (style.css), so the end of the walk's
  // backdrop still closes.
  const gutter = (pw - w) / 2 - 8;
  // The least strip is the circle's reach less 2 px - 48 for the 40 px
  // circle, 60 for the wide tier's 52 - so the circle never reaches past the
  // margin onto the picture.
  const least = reach - 2;
  const strips = lbView.scale === 1 && gutter >= least;
  els.lbMain.classList.toggle('strips', strips);
  els.lbMain.style.setProperty('--lb-gutter', `${strips ? Math.min(160, gutter) : least}px`);
}

// The zoom readout, written only when its words change: rewritten on every
// layout, as it was, ten wheel notches queued ten announcements and a star
// rewrote "fit · 124%" over itself. The readout on screen is aria-hidden
// (index.html); what is said is its hidden copy, #lb-zoom-said, written
// 400 ms after the last change - a wheel's burst, a resize, a step to an
// image of another size - so a burst ends in one announcement, or at once
// for a zoom a key, a button or a double-click asked for (`sayZoom`).
let lbZoomTimer;
function writeZoom(text) {
  if (els.lbZoom.textContent === text) return;
  els.lbZoom.textContent = text;
  clearTimeout(lbZoomTimer);
  lbZoomTimer = setTimeout(sayZoom, 400);
}
function sayZoom() {
  clearTimeout(lbZoomTimer);
  const text = els.lbZoom.textContent;
  if (els.lbZoomSaid.textContent !== text) els.lbZoomSaid.textContent = text;
}

// Under a one-column stage (under 901 px) the facts are a scroller six lines
// tall (style.css, .lb-meta), which showed Model and Size and hid the seed,
// while a square picture at 600 px wide, or any landscape one, left the
// stage a band of letterbox above and below it. The band is the facts':
// --lb-room is how much taller than six lines they may grow - the stage's
// height less the height the picture is fitted to, both as they would be
// with the facts at six lines, so a tall picture keeps the whole stage and
// nothing carries over from the image before.
function lbFactsRoom(rec, stacked) {
  const meta = els.lbMeta;
  const was = meta.style.getPropertyValue('--lb-room');
  if (!lbNarrow()) {
    if (was) meta.style.removeProperty('--lb-room');
    return;
  }
  const [sw, sh] = lbBaseStage();
  const ports = lbPorts(undefined, sw, sh, stacked);
  const [, fh] = fitSize(rec, ports);
  // Stacked halves are two pictures and the gap between them.
  const used = ports.length > 1 && ports[1].y > 0 ? 2 * fh + LB_SIDE_GAP : fh;
  const room = `${Math.max(0, Math.floor(sh - used))}px`;
  if (room !== was) meta.style.setProperty('--lb-room', room);
}

// The partner's layer: over the pass and cut at the divider, or in its half
// - a source's full picture positioned so the part `refine.fitted` kept lands
// exactly on the pass's box, and clipped to that box - with the two named.
// The left side (or the upper half) is the source's, or the lower CFG's: in
// a CFG comparison where the image in view has the lower CFG (`lbSwapped`)
// the two trade places, the partner's layer cut to the right of the divider
// and its half the second, so the lower CFG is on the left whichever of the
// two is in view.
function layoutCompare(rec, ports, pass, boxIn, rectOf) {
  const source = lbCompare.id === rec.id ? lbCompare.source : null;
  const swipe = !!source && lbCompare.mode === 'swipe';
  els.lbSrcImg.hidden = els.lbTagSrc.hidden = els.lbTagPass.hidden = !source;
  els.lbSplit.hidden = !swipe;
  if (!source) return;
  const swapped = lbSwapped(rec);
  const halves = ports.length > 1;
  const partnerPort = halves && swapped ? ports.at(-1) : ports[0];
  const at = halves ? boxIn(partnerPort) : pass;
  // Two images of one size in a CFG comparison: the whole frame. An
  // outpaint's source is drawn at its place on the canvas, at the result's
  // scale - (left, top) and its own size - so the divider shows the original
  // against the kept rectangle pixel for pixel.
  const kept = lbCompare.kind !== 'source' ? { x: 0, y: 0, w: 1, h: 1 }
    : rec.outpaint ? {
      x: -rec.outpaint.left / source.width, y: -rec.outpaint.top / source.height,
      w: rec.width / source.width, h: rec.height / source.height,
    }
      : refineKept(source.width, source.height, rec.width, rec.height);
  const full = { w: at.w / kept.w, h: at.h / kept.h };
  const box = { left: at.left - kept.x * full.w, top: at.top - kept.y * full.h, ...full };
  placeBox(els.lbSrcImg, box);
  // What of a picture is on screen in a viewport: the divider's range, the
  // partner's cut, and where the names go.
  const seenIn = (pic, rect) => ({
    left: Math.max(pic.left, rect.left), right: Math.min(pic.left + pic.w, rect.right),
    top: Math.max(pic.top, rect.top), bottom: Math.min(pic.top + pic.h, rect.bottom),
  });
  const seen = seenIn(halves ? boxIn(ports[0]) : at, rectOf(ports[0]));
  let cut = halves ? seenIn(at, rectOf(partnerPort)) : seen;
  const [leftName, rightName] = compareNames(rec, source, swapped);
  if (swipe) {
    const x = seen.left + lbCompare.split * (seen.right - seen.left);
    cut = swapped ? { ...seen, left: x } : { ...seen, right: x };
    lbCompare.range = [seen.left, seen.right];
    Object.assign(els.lbSplit.style, { left: `${x}px`, top: `${seen.top}px`, height: `${seen.bottom - seen.top}px` });
    const percent = Math.round(lbCompare.split * 100);
    els.lbSplit.setAttribute('aria-valuenow', String(percent));
    els.lbSplit.setAttribute('aria-valuetext', `${leftName} on the left ${percent}%, ${rightName} on the rest`);
  }
  els.lbSrcImg.style.clipPath = clipBox(box, cut);
  // Side by side, the second half: beside the first, or under it.
  const secondSeen = swipe ? seen : seenIn(boxIn(ports.at(-1)), rectOf(ports.at(-1)));
  const [leftTag, rightTag] = swapped ? [els.lbTagPass, els.lbTagSrc] : [els.lbTagSrc, els.lbTagPass];
  leftTag.classList.remove('end');
  Object.assign(leftTag.style, { left: `${seen.left + 8}px`, top: `${seen.top + 8}px` });
  // Swipe puts the right-hand name at the far side, where that picture is.
  rightTag.classList.toggle('end', swipe);
  Object.assign(rightTag.style, {
    left: `${swipe ? secondSeen.right - 8 : secondSeen.left + 8}px`, top: `${secondSeen.top + 8}px`,
  });
}

// The two pictures by the names the divider's value says them in, left
// first: "#12" and "#14" for a pass and its source, as ever; "CFG 2 (#41)"
// for a CFG comparison.
function compareNames(rec, partner, swapped) {
  if (lbCompare.kind !== 'cfg') return [`#${partner.id}`, `#${rec.id}`];
  const name = (r) => `${cfgName(r)} (#${r.id})`;
  return swapped ? [name(rec), name(partner)] : [name(partner), name(rec)];
}

// Whether the image in view has the lower CFG of a CFG comparison, and so
// takes the left side (layoutCompare).
function lbSwapped(rec) {
  return lbCompare.id === rec.id && lbCompare.kind === 'cfg' && !!lbCompare.source
    && rec.guidance_scale < lbCompare.source.guidance_scale;
}

// Zoom by a factor about a point (cx, cy) given relative to the viewport's
// centre, keeping the image under that point where it is - the wheel zooms
// about the pointer, the keys about the middle. Between fit or 1:1, whichever
// is smaller, and 8x device pixels (`clampScale`); at or below fit the image
// is centred. `say`: the new zoom is said at once - a key's, a button's -
// rather than once the wheel's burst is over (`writeZoom`).
function zoomLightbox(factor, cx = 0, cy = 0, say = true) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  const next = clampScale(rec, lbView.scale * factor);
  const ratio = next / lbView.scale;
  lbView.x = cx - ratio * (cx - lbView.x);
  lbView.y = cy - ratio * (cy - lbView.y);
  lbView.scale = next;
  if (next <= 1) lbView.x = lbView.y = 0;
  layoutLightbox();
  if (say) sayZoom();
}

function zoomFit() { lbView.scale = 1; lbView.x = lbView.y = 0; layoutLightbox(); sayZoom(); }
function zoomOneToOne(cx = 0, cy = 0) {
  const rec = lbList[lbIndex];
  if (rec) zoomLightbox(oneToOneScale(rec) / lbView.scale, cx, cy);
}

// Pointer position relative to the centre of the viewport it is in - either
// half, side by side or stacked, since both show the same crop - for zooming
// about it.
function stagePoint(e) {
  const box = els.lbStage.getBoundingClientRect();
  const x = e.clientX - box.left, y = e.clientY - box.top;
  // The half whose centre is nearer: the halves are one size, so that is the
  // one the point is in, or the gap's nearer side.
  const offset = (p) => [x - p.x - p.w / 2, y - p.y - p.h / 2];
  const [port] = lbPorts().sort((a, b) => Math.hypot(...offset(a)) - Math.hypot(...offset(b)));
  return offset(port);
}

// Whether a point is on the picture - either picture, comparing - as drawn:
// the hit test honours the clip-paths, where the boxes would not.
function onPicture(e) {
  const hit = document.elementFromPoint(e.clientX, e.clientY);
  return hit === els.lbImg || hit === els.lbSrcImg;
}

function renderLightbox() {
  const rec = lbList[lbIndex];
  if (!rec) { closeLightbox(); return; }
  // Another image ends a comparison: it was that pass's with its source.
  // stepPass and stepCfg (lightbox-compare.js, lightbox-cfg.js) carry it by
  // setting `id` to the next image first.
  if (lbCompare.id !== null && lbCompare.id !== rec.id) setCompare(null);
  if (lbCompareSaid && lbCompareSaid.id !== rec.id) lbCompareSaid = null;
  // The gallery is asked for this image's other CFG values a moment after
  // it lands (lightbox-cfg.js), as for its passes.
  askLadder(rec);
  const aspect = rec.width / rec.height;
  if (lbView.aspect === null || Math.abs(aspect - lbView.aspect) / lbView.aspect >= 0.01) {
    lbView.scale = 1; lbView.x = lbView.y = 0;
  } else {
    // Carried, within what this image's own fit and 1:1 allow.
    lbView.scale = clampScale(rec, lbView.scale);
  }
  lbView.aspect = aspect;
  if (els.lbImg.getAttribute('src') !== rec.url) els.lbImg.src = rec.url;
  els.lbImg.alt = rec.prompt;
  els.lbPrompt.textContent = rec.prompt;
  els.lbDesc.textContent = settingsLine(rec);
  // Use seed's note names the seed of the image it was pressed on.
  if (lbSeedNote && lbSeedNote.id !== rec.id) {
    if (els.lbError.textContent === lbSeedNote.text) { els.lbError.hidden = true; els.lbError.textContent = ''; }
    lbSeedNote = null;
  }
  if (lbDownloadError && lbDownloadError.id !== rec.id) {
    if (els.lbError.textContent === lbDownloadError.text) showError('', false, { slot: els.lbError, mirror: false });
    lbDownloadError = null;
  }
  if (lbDrawn !== rec || !els.lbSettings.firstChild) {
    lbDrawn = rec;
    els.lbSettings.replaceChildren(promptSearchLine(rec), factsList(rec, { useSeed: true, search: true, megapixels: true }), passesLine(rec));
  }
  // A live region, written only when its words change: a star, a status or
  // a delete elsewhere re-renders the same image, and each rewrite said the
  // same words again (the zoom's likewise, writeZoom).
  const count = lightboxCount(rec);
  if (els.lbCount.textContent !== count) els.lbCount.textContent = count;
  els.lbPrev.disabled = lbIndex === 0;
  els.lbNext.disabled = lbIndex === lbList.length - 1 && !canPage();
  // A toggle whose words stay "Star" (its aria-label too), as the card's
  // star's name does: the state is the glyph, the amber and aria-pressed,
  // and in words the card star's title (starTitle), as this one names F too.
  els.lbFav.textContent = rec.favorite ? '★ Star' : '☆ Star';
  els.lbFav.title = starTitle(!!rec.favorite, els.lbFav);
  els.lbFav.setAttribute('aria-pressed', String(!!rec.favorite));
  els.lbFav.classList.toggle('active', !!rec.favorite);
  // An image in the recycle bin (bin.js): Restore, Delete permanently, and
  // Star and Refine held until it is restored - their presses say why.
  const binned = !!rec.binned_at;
  els.lbRestore.hidden = !binned;
  els.lbDelete.textContent = binned ? '⌦ Delete permanently' : '⌦ Delete';
  els.lbDelete.title = binned ? 'Delete permanently (Del)' : 'Move to the recycle bin (Del)';
  els.lbFav.setAttribute('aria-disabled', String(binned));
  if (binned) els.lbFav.title = 'In the recycle bin: restore it to star it';
  // Kept facts keep their buttons, whose state follows the checkpoint's.
  for (const use of els.lbSettings.querySelectorAll('.lb-use-refs')) syncUseReferences(use);
  // The record, not its model id: a pass's Reuse names the Refine dialog.
  syncReuseButton(els.lbReuse, rec);
  // Beside it, More seeds… of this image's batch: held, never hidden, so
  // Reuse keeps its width as the lightbox steps.
  syncMoreSeedsFor(els.lbMoreSeeds, rec);
  // aria-disabled, so the button keeps its place in the tab order and its
  // click can still say why: the reason is the checkpoint's, not the image's.
  els.lbRefine.setAttribute('aria-disabled', String(!refineSupported || binned));
  els.lbRefine.title = binned ? 'In the recycle bin: restore it to refine it'
    : refineSupported ? 'Refine this image: scale it up and re-denoise it part of the way (R)'
      : 'This checkpoint has no refine pass';
  syncLightboxInpaint(rec);
  syncLightboxOutpaint(rec);
  syncCompare(rec);
  // Open first: the stage has no size while the dialog is display: none.
  if (!els.lightbox.open) els.lightbox.showModal();
  layoutLightbox();
}

// Where this image is in what the arrows walk, and its name: "3/144 · #57".
// The gallery's count is the server's - the strip holds the pages loaded so
// far, and 41/60 read as the end of a gallery of 144 - with a "+" only while
// the server's count is behind what has been loaded.
function lightboxCount(rec) {
  if (lbList.length < 2 && !canPage()) return `#${rec.id}`;
  const gallery = lbList === galleryRecords;
  const of = gallery ? Math.max(total, lbList.length) : lbList.length;
  const more = gallery && canPage() && of === lbList.length ? '+' : '';
  return `${lbIndex + 1}/${of}${more} · #${rec.id}`;
}

// Keep the original PNG, including its settings chunks: never round-trip the
// image through a canvas. Capture the record at the press, so walking while
// its fetch is pending cannot give another image its filename.
function lightboxFileName(rec) {
  const model = String(rec.model_id).split('/').pop().replace(/[^\w.-]+/g, '-')
    .replace(/^[.-]+|[.-]+$/g, '').slice(0, 100) || 'model';
  return `${rec.id}-${model}-${rec.seed}.png`;
}

async function downloadLightboxImage() {
  const rec = lbList[lbIndex];
  if (!els.lightbox.open || !rec) return;
  try {
    const response = await fetch(rec.url);
    if (!response.ok) throw new Error(`the image answered ${response.status}`);
    const blob = await response.blob();
    if (lbDownloadError?.id === rec.id) {
      if (els.lbError.textContent === lbDownloadError.text) showError('', false, { slot: els.lbError, mirror: false });
      lbDownloadError = null;
    }
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = lightboxFileName(rec);
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  } catch (err) {
    // A failed download stays in the lightbox: never navigate to a 404 page
    // or overwrite another image's note after the walk has moved on.
    if (els.lightbox.open && lbList[lbIndex] === rec) {
      const text = `Could not download #${rec.id}: ${err?.message || err}.`;
      lbDownloadError = { id: rec.id, text };
      showError(text, false, { slot: els.lbError, mirror: false });
    }
  }
}

// The passes made from this image, as a search: shown once the gallery says
// there are any (its own `refine:` search, in the gallery's scope, asked
// a moment after the image lands, so a walk does not ask for every one).
function passesLine(rec) {
  const line = document.createElement('p');
  line.className = 'lb-passes';
  line.hidden = true;
  const draw = (count) => {
    if (!count) return;
    line.replaceChildren(factButton(`⤢ Passes made from #${rec.id} (${count})`,
      `Show the refine passes made from #${rec.id} in the gallery (refine:${rec.id})`, () => showPasses(rec)));
    line.hidden = false;
  };
  clearTimeout(lbPassesTimer);
  if (lbPasses.has(rec.id)) {
    draw(lbPasses.get(rec.id));
    return line;
  }
  lbPassesTimer = setTimeout(async () => {
    const params = new URLSearchParams({ limit: 1, search: `refine:${rec.id}` });
    if (els.favOnly.checked) params.set('favorites', 'true');
    if (els.allModels.checked) params.set('all_models', 'true');
    try {
      const page = await api(`/api/gallery?${params}`);
      lbPasses.set(rec.id, page.total);
      draw(page.total);
    } catch { /* a convenience: the lightbox works without it */ }
  }, 200);
  return line;
}

function showPasses(rec) {
  showGallerySearch(`refine:${rec.id}`);
}

// Only the gallery strip has more behind it, and only while the server says so.
function canPage() { return lbList === galleryRecords && galleryMore; }

async function stepLightbox(delta) {
  if (!els.lightbox.open) return;
  const current = lbList[lbIndex];
  if (lbIndex + delta < 0) return;
  if (lbIndex + delta >= lbList.length) {
    // Walking off the end of the grid pulls the next page rather than stopping.
    if (!canPage()) return;
    await loadGallery();
    // The user may have moved on during the await - ArrowLeft is synchronous -
    // and this step then belongs to an image no longer in view: drop it rather
    // than re-imposing its own start and landing two images ahead.
    if (lbList[lbIndex] !== current) return;
    // The strip can have moved during the await - a delete, another page - so
    // step from where this image is now rather than from a stale index.
    const at = lbList.indexOf(current);
    if (at >= 0) lbIndex = at;
    if (lbIndex + delta >= lbList.length) return;
  }
  lbIndex += delta;
  renderLightbox();
}

function closeLightbox({ keepPicker = false } = {}) {
  // Opened from the Models picker's strip, which stays open under it.
  const fromModels = lbOrigin === 'models';
  // The dialog's `close` event is queued, not fired by close(), and app.js
  // wires it back here only while the dialog is still closed when it comes -
  // which covers a close the browser made itself - so a lightbox opened
  // again at once keeps its strip. The guard only spares a second close().
  if (els.lightbox.open) els.lightbox.close();
  clearTimeout(lbCloseTimer);
  clearTimeout(lbPassesTimer);
  // The readouts written only when they change start empty, so the next
  // open says where it is and at what zoom, as a fresh open always did.
  clearTimeout(lbZoomTimer);
  els.lbCount.textContent = els.lbZoom.textContent = els.lbZoomSaid.textContent = '';
  els.lbImg.src = '';
  els.lbSrcImg.removeAttribute('src');
  els.lbError.hidden = true;
  lbList = [];
  lbIndex = -1;
  lbOrigin = null;
  lbDrawn = null;
  lbSeedNote = null;
  lbCompare.id = null;
  lbCompare.source = null;
  lbCompare.ladder = [];
  lbCompareSaid = null;
  lbCfgShown = { id: null, shown: false };  // lightbox-compare.js's cfgShown
  lbSources.clear();
  lbPasses.clear();
  forgetLadders();
  // A fresh open starts at fit; the view is only carried between images of a
  // walk, not between sessions of looking.
  lbView.aspect = null;
  // A close that hands the page to something else - Reuse, Refine, a
  // search, Show this run - takes the picker with it; the user's own close
  // (dismissLightbox) goes back to the strip.
  if (fromModels && !keepPicker && !els.modelPicker.hidden) showModelPicker(false, { focus: false });
}

// Closed by the user - Escape, the cross, a click on the backdrop or beside
// the picture - focus goes to where the walk ended: that image's card, tile
// or sheet cell, which focusing scrolls into view, rather than the one the
// lightbox was opened on, forty arrow presses and a screen away. Not for a
// close that hands the page to something else - Refine's dialog, Reuse's
// panel, a delete, Show this run - which call closeLightbox alone: a card
// focused there would scroll the page back from where they took it.
function dismissLightbox() {
  const last = lbList[lbIndex];
  const from = lbList;
  const origin = lbOrigin;
  if (origin === 'models') {
    // Back to the picture in the Models picker, still open under it.
    closeLightbox({ keepPicker: true });
    if (!els.modelPicker.hidden) focusModelShot(last?.id);
    return;
  }
  closeLightbox();
  if (!last) return;
  const strips = { viewer: `#viewer img[data-id="${last.id}"]`, recent: `#recent img[data-id="${last.id}"]`,
    gallery: `#gallery .card[data-id="${last.id}"] img`, sheet: `#sheet-grid .sheet-cell[data-id="${last.id}"] img` };
  const order = origin === 'recent' ? ['recent', 'viewer', 'gallery', 'sheet']
    : from === viewerRecords || origin === 'viewer' ? ['viewer', 'recent', 'gallery', 'sheet']
    : from === galleryRecords ? ['gallery', 'sheet', 'viewer', 'recent'] : ['sheet', 'gallery', 'viewer', 'recent'];
  const shown = (selector) => [...document.querySelectorAll(selector)].find(isShown);
  requestAnimationFrame(() => {
    for (const strip of order) {
      const img = shown(strips[strip]);
      if (img) { img.focus(); return; }
    }
    // The image is in no strip shown any more - the server came back under
    // another checkpoint and the gallery is another scope, it was deleted
    // elsewhere, or the prompt check covers the gallery's column (Gallery
    // beside) - and the card that opened the lightbox may have gone with it:
    // the grid's way in, else the search box, each where it is shown, else
    // the viewer's first image, else Generate, rather than <body>.
    const back = [currentCard()?.querySelector('img'), shown('#gallery .card[data-id] img'), els.search]
      .find((el) => el && isShown(el));
    (back ?? shown('#viewer img[data-id]') ?? els.generate).focus({ preventScroll: true });
  });
}

// Where the lightbox is - the image, the strip it is walking, the view and
// the comparison - for a Refine opened from it to go back to
// (`refineReturn`), and the way back after a close that ran nothing: the
// image by id in the strip as it is now - a job that landed meanwhile
// reloads the gallery, or starts a new viewer strip, into a new array - else
// in the list it was walked in, at that view.
function lightboxPlace() {
  const rec = lbList[lbIndex];
  if (!els.lightbox.open || !rec) return null;
  const compare = lbCompare.id === rec.id ? {
    kind: lbCompare.kind, mode: lbCompare.mode, split: lbCompare.split, source: lbCompare.source, ladder: lbCompare.ladder,
  } : null;
  return {
    rec, list: lbList, gallery: lbList === galleryRecords, viewer: lbList === viewerRecords,
    view: { ...lbView }, compare,
  };
}

function returnToLightbox(place) {
  const live = place.gallery ? galleryRecords : place.viewer ? viewerRecords : place.list;
  const list = live.some((r) => r.id === place.rec.id) ? live : place.list;
  openLightbox(list.find((r) => r.id === place.rec.id) ?? place.rec, list);
  const rec = lbList[lbIndex];
  // The comparison before the view: a scale is relative to the fit of the
  // boxes it was taken in, and side by side that is half the stage - put on
  // the whole of it, a 1:1 came back at 166%.
  // A CFG partner deleted meanwhile is not put back.
  if (place.compare && !(place.compare.kind === 'cfg' && deletedIds.has(place.compare.source.id))) {
    // A pass's source back in the cache; a CFG comparison's ladder as it was
    // frozen, less anything deleted meanwhile.
    if (place.compare.kind === 'source') lbSources.set(rec.refine, place.compare.source);
    lbCompare.kind = place.compare.kind;
    lbCompare.ladder = place.compare.ladder.filter((r) => !deletedIds.has(r.id));
    lbCompare.source = place.compare.source;
    lbCompare.split = place.compare.split;
    setCompare(rec.id, place.compare.mode);
  }
  if (lbView.aspect === place.view.aspect) {
    Object.assign(lbView, place.view);
    // Within this window's limits, which a resize meanwhile can have moved.
    lbView.scale = clampScale(rec, lbView.scale);
    layoutLightbox();
  }
}

// The picture failed to load: a broken image and nothing else said why. Most
// often the server is not answering; the record is asked for to tell that
// from an image deleted elsewhere. Said where the user is looking, and taken
// back by the next picture that loads. A server not answering is the page's
// one sentence for it (OFFLINE_MESSAGE), an offline message like the panel's:
// the first status poll the server answers takes it back (clearOffline) and
// asks for the picture again (`retryLightboxImage`). A src emptied by a close
// fires this too, and is no failure.
let lbLoadError = '';
async function lbImageFailed() {
  const rec = lbList[lbIndex];
  if (!els.lightbox.open || !rec || els.lbImg.getAttribute('src') !== rec.url) return;
  let offline = serverOffline;
  let said = OFFLINE_MESSAGE;
  if (!offline) {
    try {
      await api(`/api/images/${rec.id}`);
      said = `The file of #${rec.id} could not be loaded.`;
    } catch (err) {
      offline = !!err.offline;
      said = offline ? OFFLINE_MESSAGE
        : /unknown image/.test(err.message) ? `#${rec.id} is no longer in the gallery.`
          : 'This image could not be loaded.';
    }
  }
  if (lbList[lbIndex] !== rec || els.lbImg.complete && els.lbImg.naturalWidth) return;
  lbLoadError = said;
  showError(said, false, { slot: els.lbError, mirror: false, offline });
}

function lbImageLoaded() {
  if (lbLoadError && els.lbError.textContent === lbLoadError) showError('', false, { slot: els.lbError, mirror: false });
  lbLoadError = '';
}

// The server answers again: a picture that failed for its silence is asked
// for once more, rather than left broken under a line that has gone.
function retryLightboxImage() {
  const rec = lbList[lbIndex];
  if (!els.lightbox.open || !rec || lbLoadError !== OFFLINE_MESSAGE || els.lbImg.naturalWidth) return;
  lbLoadError = '';
  els.lbImg.src = rec.url;
}

// Wheel: zoom about the pointer. Wired non-passive (app.js), or the page
// behind the dialog scrolls with it, and on the stage's parent, so the
// arrows' strips over its edges zoom too: on the stage alone they were a
// band each side the wheel scrolled the page behind the dialog.
function lbWheel(e) {
  e.preventDefault();
  zoomLightbox(Math.exp(-e.deltaY * 0.002), ...stagePoint(e), false);
}

// Drag to pan. A press that never moved is a click: on the stage beside the
// image, it closes the lightbox - but only at or below the fit size, where
// nothing is being panned and the stage is the backdrop it used to be. Zoomed
// in, the whole stage is a pan surface and only Escape or the cross close it.
// Where the press landed is read here, on the way down: once the stage has
// pointer capture, every later event's target is the stage, image or not. A
// press on the swipe's divider drags the divider instead.
let lbDrag = null;
// A press on an arrow, for the stage to know the second half of a
// double-click begun on one: the step can take the strip from under it - the
// next picture fills the width - and that press is still the arrow's, neither
// a close nor the double-click's 1:1. Half a second, Windows' default
// double-click time.
let lbArrowPress = -Infinity;
let lbAfterArrow = false;
function lbArrowDown(e) { lbArrowPress = e.timeStamp; }

function lbPointerDown(e) {
  if (e.button !== 0) return;
  clearTimeout(lbCloseTimer);
  lbAfterArrow = e.timeStamp - lbArrowPress < 500;
  lbDrag = { x: e.clientX, y: e.clientY, moved: false, onStage: e.target === els.lbStage, split: e.target === els.lbSplit };
  els.lbStage.setPointerCapture(e.pointerId);
}

function lbPointerMove(e) {
  if (!lbDrag) return;
  if (lbDrag.split) { lbDrag.moved = true; moveSplit(e.clientX); return; }
  const dx = e.clientX - lbDrag.x, dy = e.clientY - lbDrag.y;
  if (!lbDrag.moved && Math.hypot(dx, dy) < 4) return;  // a jittery click is a click
  lbDrag.moved = true;
  els.lbStage.classList.add('dragging');
  lbView.x += dx; lbView.y += dy;
  lbDrag.x = e.clientX; lbDrag.y = e.clientY;
  layoutLightbox();
}

function lbPointerUp() {
  const drag = lbDrag;
  lbDrag = null;
  els.lbStage.classList.remove('dragging');
  if (!drag || drag.moved || drag.split || !drag.onStage || lbAfterArrow || lbView.scale > 1) return;
  if (lbView.scale === 1) dismissLightbox();
  else lbCloseTimer = setTimeout(dismissLightbox, 400);
}

function lbPointerCancel() { lbDrag = null; els.lbStage.classList.remove('dragging'); }

// Double-click: 1:1 about the point on the picture, or back to fit from any
// other scale, 1:1 below fit included. On the stage, not the image: the click
// events come out of the captured pointer, so they are the stage's too, and
// the picture is found by hit-testing the point instead. Beside the picture
// at fit the first click already closed.
function lbDoubleClick(e) {
  clearTimeout(lbCloseTimer);
  if (lbAfterArrow) return;
  if (lbView.scale !== 1) { zoomFit(); return; }
  if (onPicture(e)) zoomOneToOne(...stagePoint(e));
}
