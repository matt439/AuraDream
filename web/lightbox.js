// Local Diffusion's page script: the lightbox and its zoom. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ----------------------------------------------------------------- lightbox

// `origin` is where an open from a closed lightbox came from, where it is not
// the strip it walks: the viewer's image walks the gallery from it, and
// #recent this page's earlier results (panel.js) - a close hands focus back
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
// kept for the next.
const lbCompare = { id: null, mode: 'swipe', split: 0.5, source: null, range: [0, 0] };
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
function lbNavRoom() { return lbNavReach() + 8; }

// The boxes the picture is fitted into, in the stage's pixels: the whole
// stage, or side by side its two halves, each showing the same crop - inside
// the arrows, whose circles sat over the source's left edge and the pass's
// right one there (alone, a picture is centred clear of them).
function lbPorts() {
  const sw = els.lbStage.clientWidth, sh = els.lbStage.clientHeight;
  if (lbCompare.id === null || lbCompare.mode !== 'side') return [{ x: 0, w: sw, h: sh }];
  const inset = Math.min(lbNavRoom(), sw / 8);
  const w = (sw - 2 * inset - LB_SIDE_GAP) / 2;
  return [{ x: inset, w, h: sh }, { x: inset + w + LB_SIDE_GAP, w, h: sh }];
}

// Fit is "contain": the largest size that shows the whole image inside its
// viewport, taken from the record's own dimensions rather than the decoded
// image, so the layout is right before the PNG has arrived.
function fitSize(rec) {
  const { w, h } = lbPorts()[0];
  const k = Math.min(w / rec.width, h / rec.height);
  return [rec.width * k, rec.height * k];
}

// The scale at which one image pixel is one *device* pixel. Dividing by the
// devicePixelRatio is load-bearing: at 125-150% Windows scaling a plain
// natural-size image is a 1.25-1.5x resample, which is precisely the smoothing
// a sharpening sweep is being judged on.
function oneToOneScale(rec) {
  return (rec.width / window.devicePixelRatio) / fitSize(rec)[0];
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
  const ports = lbPorts();
  const [fw, fh] = fitSize(rec);
  const w = fw * lbView.scale, h = fh * lbView.scale;
  const { w: pw, h: ph } = ports[0];
  // An image smaller than its viewport on an axis is centred on it; one
  // larger can be panned until its edge meets the viewport's, and no further.
  const maxX = Math.max(0, (w - pw) / 2), maxY = Math.max(0, (h - ph) / 2);
  lbView.x = Math.min(maxX, Math.max(-maxX, lbView.x));
  lbView.y = Math.min(maxY, Math.max(-maxY, lbView.y));
  // The picture's box in a viewport; the pass is the last one's.
  const boxIn = (port) => ({ left: port.x + (port.w - w) / 2 + lbView.x, top: (ph - h) / 2 + lbView.y, w, h });
  const rectOf = (port) => ({ left: port.x, right: port.x + port.w, top: 0, bottom: ph });
  const pass = boxIn(ports.at(-1));
  placeBox(els.lbImg, pass);
  els.lbImg.style.clipPath = ports.length > 1 ? clipBox(pass, rectOf(ports.at(-1))) : '';
  layoutCompare(rec, ports, pass, boxIn, rectOf);
  // Tested in this order: a wheel stop between 1:1 and fit is neither.
  const one = oneToOneScale(rec);
  const percent = Math.round((lbView.scale / one) * 100);
  els.lbZoom.textContent = Math.abs(lbView.scale - one) < 1e-3 ? '1:1'
    : lbView.scale === 1 ? `fit · ${percent}%` : `${percent}%`;
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
  const least = lbNavReach() - 2;
  const strips = lbView.scale === 1 && gutter >= least;
  els.lbMain.classList.toggle('strips', strips);
  els.lbMain.style.setProperty('--lb-gutter', `${strips ? Math.min(160, gutter) : least}px`);
}

// The source's layer: over the pass and cut at the divider, or in the left
// half - its full picture positioned so the part `refine.fitted` kept lands
// exactly on the pass's box, and clipped to that box - with the two named.
function layoutCompare(rec, ports, pass, boxIn, rectOf) {
  const source = lbCompare.id === rec.id ? lbCompare.source : null;
  const swipe = !!source && lbCompare.mode === 'swipe';
  els.lbSrcImg.hidden = els.lbTagSrc.hidden = els.lbTagPass.hidden = !source;
  els.lbSplit.hidden = !swipe;
  if (!source) return;
  const at = ports.length > 1 ? boxIn(ports[0]) : pass;
  const kept = refineKept(source.width, source.height, rec.width, rec.height);
  const full = { w: at.w / kept.w, h: at.h / kept.h };
  const box = { left: at.left - kept.x * full.w, top: at.top - kept.y * full.h, ...full };
  placeBox(els.lbSrcImg, box);
  // What of the picture is on screen in the source's viewport: the
  // divider's range, and where the names go.
  const port = rectOf(ports[0]);
  const seen = {
    left: Math.max(at.left, port.left), right: Math.min(at.left + at.w, port.right),
    top: Math.max(at.top, 0), bottom: Math.min(at.top + at.h, port.bottom),
  };
  let cut = seen;
  if (swipe) {
    const x = seen.left + lbCompare.split * (seen.right - seen.left);
    cut = { ...seen, right: x };
    lbCompare.range = [seen.left, seen.right];
    Object.assign(els.lbSplit.style, { left: `${x}px`, top: `${seen.top}px`, height: `${seen.bottom - seen.top}px` });
    const percent = Math.round(lbCompare.split * 100);
    els.lbSplit.setAttribute('aria-valuenow', String(percent));
    els.lbSplit.setAttribute('aria-valuetext', `#${source.id} on the left ${percent}%, #${rec.id} on the rest`);
  }
  els.lbSrcImg.style.clipPath = clipBox(box, cut);
  const passSeen = swipe ? seen : {
    left: Math.max(pass.left, ports.at(-1).x), right: Math.min(pass.left + pass.w, ports.at(-1).x + ports.at(-1).w),
    top: seen.top,
  };
  Object.assign(els.lbTagSrc.style, { left: `${seen.left + 8}px`, top: `${seen.top + 8}px` });
  // Swipe puts the pass's name at the far side, where the pass is.
  els.lbTagPass.classList.toggle('end', swipe);
  Object.assign(els.lbTagPass.style, {
    left: `${swipe ? passSeen.right - 8 : passSeen.left + 8}px`, top: `${passSeen.top + 8}px`,
  });
}

// Zoom by a factor about a point (cx, cy) given relative to the viewport's
// centre, keeping the image under that point where it is - the wheel zooms
// about the pointer, the keys about the middle. Between fit or 1:1, whichever
// is smaller, and 8x device pixels (`clampScale`); at or below fit the image
// is centred.
function zoomLightbox(factor, cx = 0, cy = 0) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  const next = clampScale(rec, lbView.scale * factor);
  const ratio = next / lbView.scale;
  lbView.x = cx - ratio * (cx - lbView.x);
  lbView.y = cy - ratio * (cy - lbView.y);
  lbView.scale = next;
  if (next <= 1) lbView.x = lbView.y = 0;
  layoutLightbox();
}

function zoomFit() { lbView.scale = 1; lbView.x = lbView.y = 0; layoutLightbox(); }
function zoomOneToOne(cx = 0, cy = 0) {
  const rec = lbList[lbIndex];
  if (rec) zoomLightbox(oneToOneScale(rec) / lbView.scale, cx, cy);
}

// Pointer position relative to the centre of the viewport it is in - either
// half, side by side, since both show the same crop - for zooming about it.
function stagePoint(e) {
  const box = els.lbStage.getBoundingClientRect();
  const x = e.clientX - box.left;
  const ports = lbPorts();
  const port = ports.find((p) => x < p.x + p.w + LB_SIDE_GAP / 2) ?? ports.at(-1);
  return [x - port.x - port.w / 2, e.clientY - box.top - box.height / 2];
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
  if (lbCompare.id !== null && lbCompare.id !== rec.id) setCompare(null);
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
  if (lbDrawn !== rec || !els.lbSettings.firstChild) {
    lbDrawn = rec;
    els.lbSettings.replaceChildren(factsList(rec, { useSeed: true }), passesLine(rec));
  }
  els.lbCount.textContent = lightboxCount(rec);
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
// actions has one), put the seed in the panel's Seed box.
function factsList(rec, { useSeed: seedButton = false } = {}) {
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
  row('Model', [rec.model_id.split('/').pop(), rec.quantization].filter(Boolean).join(' · ')).title = rec.model_id;
  row('Size', `${rec.width}×${rec.height}`);
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
    showError(`Seed ${rec.seed} is in the Seed box.`, true, { slot: els.lbError, mirror: false });
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
  closeLightbox();
  els.search.value = `refine:${rec.id}`;
  resetGallery();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelector('.gallery-head').scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  els.search.focus({ preventScroll: true });
}

// The comparison's controls, on a pass only: dead, saying why, where the
// source has gone.
function syncCompare(rec) {
  const pass = rec.refine != null;
  els.lbCompare.hidden = !pass;
  // The keys line names S where it does something: on a pass. In the title
  // and the button's hint alone, it was found only by hovering.
  els.lbKeys.textContent = lbKeysLine(pass);
  if (!pass) return;
  const on = lbCompare.id === rec.id;
  const gone = rec.refine_missing || lbSources.get(rec.refine) === null;
  els.lbCompareOn.setAttribute('aria-pressed', String(on));
  for (const [button, mode] of [[els.lbSwipe, 'swipe'], [els.lbSideBySide, 'side']]) {
    button.setAttribute('aria-pressed', String(on && lbCompare.mode === mode));
    button.setAttribute('aria-disabled', String(gone));
  }
  els.lbCompareOn.setAttribute('aria-disabled', String(gone));
  els.lbCompareOn.title = gone ? `#${rec.refine}, the image this was refined from, has been deleted`
    : `Compare this pass with #${rec.refine}, the image it was refined from, at the same crop (S)`;
  const source = on ? lbCompare.source : null;
  if (source && els.lbSrcImg.getAttribute('src') !== source.url) els.lbSrcImg.src = source.url;
  if (source) {
    els.lbSrcImg.alt = `#${source.id}, the image this was refined from`;
    els.lbTagSrc.textContent = `Source #${source.id}`;
    els.lbTagPass.textContent = `Pass #${rec.id}`;
  }
  const kept = source && refineKept(source.width, source.height, rec.width, rec.height);
  const cropped = kept && (kept.w < 1 || kept.h < 1);
  els.lbCompareNote.textContent = gone ? `source #${rec.refine} deleted`
    : source ? `#${source.id}, ${source.width}×${source.height}${cropped ? ', cropped as the pass was' : ''}` : '';
}

// The lightbox's keys, under the comparison (index.html has the line without
// S, as it reads on any image that is not a pass).
function lbKeysLine(pass) {
  return `← → walk · F star${pass ? ' · S compare' : ''} · Del delete · 0 fit · 1 1:1 · +/− zoom · Esc close`;
}

// S, the button and the two modes. Off, a mode turns it on in that mode; on,
// the button and S turn it off and the other mode switches to it. The source
// is fetched the first time, and a source the gallery no longer has leaves
// the controls dead with the reason.
async function toggleCompare(mode = null) {
  const rec = lbList[lbIndex];
  if (!rec || rec.refine == null) return;
  if (lbCompare.id === rec.id) {
    if (!mode) setCompare(null);
    else if (mode !== lbCompare.mode) setCompare(rec.id, mode);
    return;
  }
  const source = await compareSource(rec);
  if (lbList[lbIndex] !== rec || !els.lightbox.open) return;
  if (!source) { syncCompare(rec); return; }
  lbCompare.source = source;
  setCompare(rec.id, mode ?? lbCompare.mode);
}

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
  if (id === null) lbCompare.source = null;
  if (!rec) return;
  if (lbView.scale !== 1) {
    lbView.scale = clampScale(rec, lbView.scale * before / fitSize(rec)[0]);
    if (lbView.scale <= 1) lbView.x = lbView.y = 0;
  }
  syncCompare(rec);
  layoutLightbox();
  // S pressed on the divider hides it with the focus in it, which fell to the
  // top of the dialog or out of it: it goes to the toggle that brings the
  // divider back instead, or to the cross where this image has none.
  if (held && els.lbSplit.hidden) (els.lbCompare.hidden ? els.lbClose : els.lbCompareOn).focus({ preventScroll: true });
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

function closeLightbox() {
  // close() fires the dialog's `close` event, which is wired back here - the
  // guard is what keeps that from recursing, and what makes a native Escape
  // (which closes the dialog first) still reset the strip.
  if (els.lightbox.open) els.lightbox.close();
  clearTimeout(lbCloseTimer);
  clearTimeout(lbPassesTimer);
  els.lbImg.src = '';
  els.lbSrcImg.removeAttribute('src');
  els.lbError.hidden = true;
  lbList = [];
  lbIndex = -1;
  lbOrigin = null;
  lbDrawn = null;
  lbCompare.id = null;
  lbCompare.source = null;
  lbSources.clear();
  lbPasses.clear();
  // A fresh open starts at fit; the view is only carried between images of a
  // walk, not between sessions of looking.
  lbView.aspect = null;
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
  closeLightbox();
  if (!last) return;
  const strips = { viewer: `#viewer img[data-id="${last.id}"]`, recent: `#recent img[data-id="${last.id}"]`,
    gallery: `#gallery .card[data-id="${last.id}"] img`, sheet: `#sheet-grid .sheet-cell[data-id="${last.id}"] img` };
  const order = origin === 'recent' ? ['recent', 'viewer', 'gallery', 'sheet']
    : from === viewerRecords || origin === 'viewer' ? ['viewer', 'recent', 'gallery', 'sheet']
    : from === galleryRecords ? ['gallery', 'sheet', 'viewer', 'recent'] : ['sheet', 'gallery', 'viewer', 'recent'];
  const shown = (selector) => [...document.querySelectorAll(selector)].find((el) => el.checkVisibility?.() ?? true);
  requestAnimationFrame(() => {
    for (const strip of order) {
      const img = shown(strips[strip]);
      if (img) { img.focus(); return; }
    }
    // The image is in no strip any more - the server came back under another
    // checkpoint and the gallery is another scope, or it was deleted
    // elsewhere - and the card that opened the lightbox went with it: the
    // grid's way in, else the search box, rather than <body>.
    (currentCard()?.querySelector('img') ?? shown('#gallery .card[data-id] img') ?? els.search)
      .focus({ preventScroll: true });
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
  const compare = lbCompare.id === rec.id
    ? { mode: lbCompare.mode, split: lbCompare.split, source: lbCompare.source } : null;
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
  if (place.compare) {
    lbSources.set(rec.refine, place.compare.source);
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

// The source's picture: the comparison ends, and says why.
function lbSourceFailed() {
  const rec = lbList[lbIndex];
  if (!rec || lbCompare.id !== rec.id || !els.lbSrcImg.getAttribute('src')) return;
  setCompare(null);
  els.lbCompareNote.textContent = `#${rec.refine} could not be loaded`;
}

// Wheel: zoom about the pointer. Wired non-passive (app.js), or the page
// behind the dialog scrolls with it, and on the stage's parent, so the
// arrows' strips over its edges zoom too: on the stage alone they were a
// band each side the wheel scrolled the page behind the dialog.
function lbWheel(e) {
  e.preventDefault();
  zoomLightbox(Math.exp(-e.deltaY * 0.002), ...stagePoint(e));
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

// The divider's keys, while it has focus: the arrows move it rather than
// walking the strip (app.js skips a key handled here).
function lbSplitKey(e) {
  const step = e.shiftKey ? 0.1 : 0.02;
  const to = { ArrowLeft: lbCompare.split - step, ArrowRight: lbCompare.split + step, Home: 0, End: 1 }[e.key];
  if (to === undefined) return;
  e.preventDefault();
  lbCompare.split = Math.min(1, Math.max(0, to));
  layoutLightbox();
}
