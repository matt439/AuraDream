// Local Diffusion's page script: the lightbox and its zoom. Declarations only; state.js has
// the load-order contract these files share.
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
  if (els.lbSideBySide.title !== sides) els.lbSideBySide.title = sides;
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
  // Two images of one size in a CFG comparison: the whole frame.
  const kept = lbCompare.kind === 'source' ? refineKept(source.width, source.height, rec.width, rec.height)
    : { x: 0, y: 0, w: 1, h: 1 };
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
  // stepPass and stepCfg carry it by setting `id` to the next image first.
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
  // Kept facts keep their buttons, whose state follows the checkpoint's.
  for (const use of els.lbSettings.querySelectorAll('.lb-use-refs')) syncUseReferences(use);
  // The record, not its model id: a pass's Reuse names the Refine dialog.
  syncReuseButton(els.lbReuse, rec);
  // aria-disabled, so the button keeps its place in the tab order and its
  // click can still say why: the reason is the checkpoint's, not the image's.
  els.lbRefine.setAttribute('aria-disabled', String(!refineSupported));
  els.lbRefine.title = refineSupported
    ? 'Refine this image: scale it up and re-denoise it part of the way'
    : 'This checkpoint has no refine pass';
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

// What made a record, in one line: the lightbox's description for a screen
// reader, and what the delete confirm names an image by - the cells of a
// sweep share a prompt and a seed, and differ only here.
function settingsLine(rec) {
  return (
    // The checkpoint is part of the reproduction set as much as the seed is -
    // the same prompt and settings on Turbo and on FLUX are different images.
    `${rec.model_id.split('/').pop()}, ` +
    `${rec.width}×${rec.height}, ${rec.steps} steps, ` +
    `CFG ${rec.guidance_scale}, ` +
    // Only where it differs from the CFG: on Krea 2 it is the same number.
    // Named as the panel's boxes are.
    (rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? `True CFG ${rec.true_cfg}, ` : '') +
    `${rec.mu == null ? 'no Mu' : `Mu ${rec.mu}`}, ` +
    // Only when there was one: most records carry neither.
    (rec.lora ? `LoRA ${rec.lora} @ ${rec.lora_scale}, ` : '') +
    // Gallery images by the page's one name for them, "#N" - the Refine
    // dialog's title, the sheet's source column and the confirms say it so.
    (rec.reference?.length ? `ref ${rec.reference.map((id) => `#${id}`).join('+')}, ` : '') +
    // A refine pass: the image it started from and how far it was re-denoised,
    // and whether that image is still in the gallery to make it again from.
    (rec.refine != null ? `refine of #${rec.refine}${rec.refine_missing ? ' (deleted)' : ''} @ ${rec.strength}, ` : '') +
    `seed ${rec.seed}, ${(rec.duration_ms / 1000).toFixed(1)}s`
  );
}

// What made a record, as a list (`dl.facts`): the lightbox's side column, and
// whatever else shows one image with its settings. The one line it replaced
// was 12 px of commas that buried the seed mid-sentence and left out the
// negative prompt it gave the True CFG of, the date and the run. One row a
// fact, in the reproduction set's order; a row the record has no value for is
// left out. Its buttons open the image it was refined from (with this one
// after it, so → comes back), each reference, and the run as a sheet - and,
// where the view has no Use seed of its own (`useSeed`: the lightbox, where
// older images' seeds are browsed; not the result column, whose row of
// actions has one), put the seed in the panel's Seed box. `search` (the
// lightbox's) ends each row the search grammar can say with a ⌕ that adds it
// to the gallery's search (gallery-search.js). `megapixels` (the lightbox's,
// where a picture is looked at full size) follows the Size with its
// megapixels to one decimal, "1536×1024 (1.6 MP)", as the size hints say them.
function factsList(rec, { useSeed: seedButton = false, search = false, megapixels = false } = {}) {
  const dl = document.createElement('dl');
  dl.className = 'facts';
  const row = (term, ...value) => {
    const div = document.createElement('div');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = term;
    dd.append(...value);
    div.append(dt, dd);
    dl.append(div);
    return dd;
  };
  // The running checkpoint by the catalogue's name, as the topbar has it (checkpointName).
  row('Model', [checkpointName(rec.model_id), rec.quantization].filter(Boolean).join(' · ')).title = rec.model_id;
  row('Size', `${rec.width}×${rec.height}` + (megapixels ? ` (${(rec.width * rec.height / 1e6).toFixed(1)} MP)` : ''));
  row('Sampling', [
    `${rec.steps} steps`, `CFG ${rec.guidance_scale}`,
    // As the settings line has it: only where it is a number of its own.
    ...(rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? [`True CFG ${rec.true_cfg}`] : []),
    rec.mu == null ? 'no Mu' : `Mu ${rec.mu}`,
  ].join(' · '));
  if (rec.negative_prompt) row('Negative', rec.negative_prompt).className = 'text';
  if (rec.lora) row('LoRA', `${rec.lora} @ ${rec.lora_scale}`);
  if (rec.reference?.length) row('References', referenceThumbs(rec));
  if (rec.refine != null) {
    // A source that has gone cannot be opened: said, not linked.
    const from = rec.refine_missing ? `#${rec.refine} (deleted)`
      : factButton(`#${rec.refine}`, `Open #${rec.refine}, the image this was refined from; → comes back here`,
        () => { void openRefineSource(rec); });
    row('Refined from', from, ` @ ${rec.strength}`);
  }
  const vars = Object.entries(rec.variables ?? {});
  if (vars.length) {
    // A comparison's version by the name the sheet gives it.
    row('Variables', vars.map(([k, v]) => (k === 'variant' ? `version: ${VERSION_NAMES.get(v) ?? v}` : `${k} = ${v}`)).join(', '));
  }
  // The panel is under the dialog, so the press says what it did in the
  // lightbox's own line - not the panel's, where it would replace a job's
  // outcome - and the run hint says the rest once it closes ("Seed fixed
  // at …").
  row('Seed', String(rec.seed), ...(seedButton ? [' · ', factButton('Use seed', `Put ${rec.seed} in the Seed box`, () => {
    useSeed(rec.seed);
    const text = `Seed ${rec.seed} is in the Seed box.`;
    showError(text, true, { slot: els.lbError, mirror: false });
    // Until another image is drawn (renderLightbox), whose seed it is not.
    lbSeedNote = { id: rec.id, text };
  })] : []));
  const made = new Date(rec.created_at).toLocaleString([], {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
  row('Created', `${made} · took ${(rec.duration_ms / 1000).toFixed(1)}s`);
  // What the card's ⊞ does: the run id spans every job that worked on a grid,
  // a resumed night included; the job id names one of them.
  const run = rec.run_id ? `run:${rec.run_id}` : rec.job_id ? `job:${rec.job_id}` : null;
  row('Run', run
    ? factButton('⊞ Show this run', `Show the run that made this as a sheet (${run})`, () => {
      if (els.lightbox.open) closeLightbox();
      void showSheetScope(run);
    })
    : 'none: made before runs were grouped');
  if (search) addFactSearches(dl, rec);
  return dl;
}

function factButton(text, title, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'lb-link';
  b.textContent = text;
  b.title = title;
  b.onclick = onClick;
  return b;
}

// The images a render was conditioned on, numbered as the panel's chips are -
// "image 2" in a klein prompt is the second - each opening in the lightbox
// ahead of this one, and the set put back in the panel in one press, for a
// new prompt on the same inputs without a Reuse that replaces the panel.
function referenceThumbs(rec) {
  const ids = rec.reference;
  const block = document.createElement('div');
  block.className = 'lb-refs';
  const list = document.createElement('div');
  list.className = 'lb-ref-list';
  const use = document.createElement('button');
  // Each slot's "is it gone?" by id, for the Use press to ask again.
  const goneBy = new Map();
  ids.forEach((id, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lb-ref';
    b.dataset.id = id;
    b.title = `Open #${id}, reference image ${i + 1}; → comes back here`;
    const img = document.createElement('img');
    img.src = `/api/images/${id}/thumb`;
    img.alt = '';
    img.loading = 'lazy';
    // Only the server's "unknown image" is a deleted one: then the slot says
    // so and opens nothing.
    const gone = (err) => {
      if (!/unknown image/.test(err.message)) return false;
      if (img.isConnected) {
        img.replaceWith(Object.assign(document.createElement('span'), { className: 'gone', textContent: 'deleted' }));
      }
      b.setAttribute('aria-disabled', 'true');
      b.title = `#${id}, reference image ${i + 1}, has been deleted`;
      syncUseReferences(use);
      return true;
    };
    goneBy.set(id, gone);
    // A picture that will not load is asked about.
    img.onerror = () => { api(`/api/images/${id}`).catch(gone); };
    const caption = document.createElement('span');
    caption.textContent = `image ${i + 1} · #${id}`;
    b.append(img, caption);
    b.onclick = async () => {
      if (b.getAttribute('aria-disabled') === 'true') return;
      let other;
      try {
        other = await api(`/api/images/${id}`);
      } catch (err) {
        if (!gone(err)) showError(err.message);
        return;
      }
      openLightbox(other, [other, rec]);
    };
    list.append(b);
  });
  use.type = 'button';
  use.className = 'ghost small lb-use-refs';
  use.textContent = 'Use these references';
  // Every id is asked about again, as Reuse's are (`checkReused`): a slot
  // drawn from the cache says nothing of an image deleted since, and a
  // deleted one put in the panel was an empty chip and a refused Generate.
  // Only the server's "unknown image" leaves one out; anything else keeps it.
  use.onclick = async () => {
    syncUseReferences(use);
    if (use.getAttribute('aria-disabled') === 'true') { showError(use.title, true); return; }
    const kept = await Promise.all(ids.map((id) => api(`/api/images/${id}`).then(() => true, (err) => !goneBy.get(id)(err))));
    const live = ids.filter((_, i) => kept[i]);
    const dead = ids.filter((_, i) => !kept[i]);
    // Asked again after the answers: a load can change the checkpoint meanwhile.
    syncUseReferences(use);
    if (use.getAttribute('aria-disabled') === 'true') { showError(use.title, true); return; }
    setReferences(live);
    const put = live.length === 1 ? `Reference ${imageNames(live)} is in the panel`
      : `References ${imageNames(live)} are in the panel`;
    showError(!dead.length ? `${put}.` : `${put}; ${imageNames(dead)} ${dead.length === 1
      ? 'is no longer in the gallery, so it was' : 'are no longer in the gallery, so they were'} left out.`, true);
  };
  block.append(list, use);
  syncUseReferences(use);
  return block;
}

// Use these references, dead and saying why where the checkpoint takes none
// or every one of them has gone - from the checkpoint as it is now, so a
// render of the same facts (a status after a load) brings it up to date.
function syncUseReferences(use) {
  const slots = [...use.parentElement?.querySelectorAll('.lb-ref') ?? []];
  const ids = slots.map((b) => b.dataset.id);
  const why = !referenceSupported ? REFERENCES_NONE
    : slots.length && slots.every((b) => b.getAttribute('aria-disabled') === 'true')
      ? `${imageNames(ids)} ${ids.length === 1 ? 'is' : 'are'} no longer in the gallery.`
      : '';
  use.setAttribute('aria-disabled', String(!!why));
  use.title = why || 'Put these images in the panel as its references, in this order';
}

// The image a pass was refined from, by id: from a strip the page holds, else
// asked for. Opened with the pass after it, so → comes back.
async function openRefineSource(rec) {
  let source = [...lbList, ...galleryRecords, ...viewerRecords].find((r) => r.id === rec.refine);
  if (!source) {
    try {
      source = await api(`/api/images/${rec.refine}`);
    } catch (err) {
      showError(/unknown image/.test(err.message) ? `#${rec.refine} is no longer in the gallery.` : err.message);
      return;
    }
  }
  openLightbox(source, [source, rec]);
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

// The comparison's controls, on every image: the shared part here - the two
// modes, the keys line, the divider's keys, the names and the note - and
// each kind's own in syncSourceCompare and syncCfgCompare (lightbox-cfg.js).
// A control with nothing to do is held, saying why.
function syncCompare(rec) {
  const pass = rec.refine != null;
  const on = lbCompare.id === rec.id;
  const kind = on ? lbCompare.kind : pass ? 'source' : 'cfg';
  const sourceHeld = syncSourceCompare(rec, on && kind === 'source');
  const cfg = syncCfgCompare(rec, on && kind === 'cfg', !pass, cfgShown(rec, on && kind === 'cfg'));
  // The box: on every image from 901 px, where it is the column's; under it
  // only where there is something in it (cfgShown), as the CFG part is.
  els.lbCompare.hidden = !pass && !cfg.shown;
  // A mode turns on the kind a press of S would, or switches the one on: held
  // where that has nothing to compare with, its press saying why.
  const held = on ? '' : pass ? sourceHeld : cfg.held;
  for (const [button, mode] of [[els.lbSwipe, 'swipe'], [els.lbSideBySide, 'side']]) {
    button.setAttribute('aria-pressed', String(on && lbCompare.mode === mode));
    // A deleted source holds them on a pass whether or not CFG is on, as ever.
    const why = held || (pass && kind === 'source' ? sourceHeld : '');
    button.setAttribute('aria-disabled', String(!!why));
  }
  // Comparing, with another pass of the same source in the strip, the arrows
  // swap passes (`stepPass`), or with another CFG value of the image in it,
  // step along CFG (`stepCfg`); the divider takes Shift+arrows instead.
  const passes = on && lbSiblingPass(1) >= 0;
  const cfgSteps = on && kind === 'cfg' && cfg.steps;
  const divider = on && lbCompare.mode === 'swipe';
  // The keys line names S, which does something on every image: it compares
  // a pass with its source, any other image across CFG, or says why it
  // cannot. In the title and the button's hint alone, it was found only by
  // hovering. Likewise what the arrows do while comparing.
  const keys = lbKeysLine({ passes, cfg: cfgSteps, divider });
  if (els.lbKeys.textContent !== keys) els.lbKeys.textContent = keys;
  if (passes || cfgSteps) {
    els.lbSplit.setAttribute('aria-keyshortcuts', 'Shift+ArrowLeft Shift+ArrowRight PageUp PageDown Home End');
    els.lbSplit.title = passes ? 'Drag, or Shift+← →, to move the divider; ← → switch to another pass of this source'
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
function syncSourceCompare(rec, on) {
  const pass = rec.refine != null;
  els.lbCompareOn.hidden = !pass;
  if (!pass) return '';
  const gone = rec.refine_missing || lbSources.get(rec.refine) === null;
  els.lbCompareOn.setAttribute('aria-pressed', String(on));
  els.lbCompareOn.setAttribute('aria-disabled', String(gone));
  const why = gone ? `#${rec.refine}, the image this was refined from, has been deleted` : '';
  els.lbCompareOn.title = why || `Compare this pass with #${rec.refine}, the image it was refined from, at the same crop (S)`;
  return why;
}

// A pass's note: its source and how much of it the pass kept, or that the
// source has gone.
function sourceNote(rec, source) {
  if (rec.refine == null) return '';
  if (rec.refine_missing || lbSources.get(rec.refine) === null) return `source #${rec.refine} deleted`;
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
// (`divider`).
function lbKeysLine({ passes = false, cfg = false, divider = false } = {}) {
  const keys = [['← →', passes ? 'passes' : cfg ? 'CFG' : 'walk'], ...(divider ? [['Shift+← →', 'divider']] : []),
    ['F', 'star'], ['S', 'compare'], ['Del', 'delete'],
    ['0', 'fit'], ['1', '1:1'], ['+/−', 'zoom'], ['Esc', 'close']];
  return keys.map(([key, does]) => `${key.replaceAll(' ', ' ').replace('/', '/⁠')} ${does}`)
    .join(' · ');
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
  kind ??= rec.refine != null ? 'source' : 'cfg';
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
    (lbCompare.kind === 'cfg' || els.lbCompareOn.hidden ? els.lbCompareCfg : els.lbCompareOn).focus({ preventScroll: true });
  }
}

// The divider, from a drag or its keys: a fraction of the picture on screen.
function moveSplit(clientX) {
  const [from, to] = lbCompare.range;
  const x = clientX - els.lbStage.getBoundingClientRect().left;
  lbCompare.split = to > from ? Math.min(1, Math.max(0, (x - from) / (to - from))) : 0.5;
  layoutLightbox();
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
  lbCfgShown = { id: null, shown: false };
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
  if (kind !== 'cfg') return;
  try {
    await api(`/api/images/${source.id}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) return;
    forgetLadderImages(new Set([source.id]));
    if (lbList[lbIndex] !== rec || lbCompareSaid?.id !== rec.id) return;
    lbCompareSaid = { id: rec.id, text: `#${source.id} is no longer in the gallery` };
    syncCompare(rec);
  }
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
  if (arrow && !e.shiftKey && (lbSiblingPass(1) >= 0 || cfgArrows())) return;
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
