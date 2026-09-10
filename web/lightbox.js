// Krea Local's page script: the lightbox and its zoom. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ----------------------------------------------------------------- lightbox

function openLightbox(rec, list) {
  lbList = list && list.includes(rec) ? list : [rec];
  lbIndex = lbList.indexOf(rec);
  renderLightbox();
}

// The view: a scale relative to the fit size (1 = fit, never below), and the
// image centre's offset from the stage centre in CSS pixels. It is keyed on the
// image's dimensions and kept across records that share them, so the arrow keys
// compare the identical crop at the identical magnification across the cells of
// a sweep - the whole point of zooming in on a steps or mu axis. A record of
// another size starts at fit; so does every fresh open.
const lbView = { scale: 1, x: 0, y: 0, key: null };
const ZOOM_STEP = 1.25;

// Fit is "contain": the largest size that shows the whole image inside the
// stage, taken from the record's own dimensions rather than the decoded image,
// so the layout is right before the PNG has arrived.
function fitSize(rec) {
  const k = Math.min(els.lbStage.clientWidth / rec.width, els.lbStage.clientHeight / rec.height);
  return [rec.width * k, rec.height * k];
}

// The scale at which one image pixel is one *device* pixel. Dividing by the
// devicePixelRatio is load-bearing: at 125-150% Windows scaling a plain
// natural-size image is a 1.25-1.5x resample, which is precisely the smoothing
// a sharpening sweep is being judged on.
function oneToOneScale(rec) {
  return (rec.width / window.devicePixelRatio) / fitSize(rec)[0];
}

function layoutLightbox() {
  const rec = lbList[lbIndex];
  if (!rec || !els.lightbox.open) return;
  const [fw, fh] = fitSize(rec);
  const w = fw * lbView.scale, h = fh * lbView.scale;
  const sw = els.lbStage.clientWidth, sh = els.lbStage.clientHeight;
  // An image smaller than the stage on an axis is centred on it; one larger
  // can be panned until its edge meets the stage's, and no further.
  const maxX = Math.max(0, (w - sw) / 2), maxY = Math.max(0, (h - sh) / 2);
  lbView.x = Math.min(maxX, Math.max(-maxX, lbView.x));
  lbView.y = Math.min(maxY, Math.max(-maxY, lbView.y));
  Object.assign(els.lbImg.style, {
    width: `${w}px`, height: `${h}px`,
    left: `${(sw - w) / 2 + lbView.x}px`, top: `${(sh - h) / 2 + lbView.y}px`,
  });
  const one = oneToOneScale(rec);
  const percent = Math.round((lbView.scale / one) * 100);
  els.lbZoom.textContent = lbView.scale <= 1 ? `fit · ${percent}%`
    : Math.abs(lbView.scale - one) < 1e-3 ? '1:1' : `${percent}%`;
  els.lbStage.classList.toggle('zoomed', lbView.scale > 1);
}

// Zoom by a factor about a point (cx, cy) given relative to the stage centre,
// keeping the image under that point where it is - the wheel zooms about the
// pointer, the keys about the middle. Never below fit; capped at 8x device
// pixels, past which the picture is squares.
function zoomLightbox(factor, cx = 0, cy = 0) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  const ceiling = Math.max(1, oneToOneScale(rec) * 8);
  const next = Math.min(ceiling, Math.max(1, lbView.scale * factor));
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

// Pointer position relative to the stage centre, for zooming about it.
function stagePoint(e) {
  const box = els.lbStage.getBoundingClientRect();
  return [e.clientX - box.left - box.width / 2, e.clientY - box.top - box.height / 2];
}

function renderLightbox() {
  const rec = lbList[lbIndex];
  if (!rec) { closeLightbox(); return; }
  const key = `${rec.width}x${rec.height}`;
  if (lbView.key !== key) {
    lbView.scale = 1; lbView.x = lbView.y = 0; lbView.key = key;
  }
  els.lbImg.src = rec.url;
  els.lbImg.alt = rec.prompt;
  els.lbMeta.textContent =
    // The checkpoint is part of the reproduction set as much as the seed is -
    // the same prompt and settings on Turbo and on FLUX are different images.
    `${rec.prompt} — ${rec.model_id.split('/').pop()}, ` +
    `${rec.width}×${rec.height}, ${rec.steps} steps, ` +
    `CFG ${rec.guidance_scale}, ` +
    // Only where it differs from the CFG: on Krea 2 it is the same number.
    (rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? `true CFG ${rec.true_cfg}, ` : '') +
    `${rec.mu == null ? 'no mu' : `mu ${rec.mu}`}, ` +
    // Only when there was one: most records carry neither.
    (rec.lora ? `LoRA ${rec.lora} @ ${rec.lora_scale}, ` : '') +
    (rec.reference?.length ? `ref ${rec.reference.join('+')}, ` : '') +
    `seed ${rec.seed}, ${(rec.duration_ms / 1000).toFixed(1)}s` +
    (lbList.length > 1 ? ` · ${lbIndex + 1}/${lbList.length}` : '');
  els.lbPrev.disabled = lbIndex === 0;
  els.lbNext.disabled = lbIndex === lbList.length - 1 && !canPage();
  els.lbFav.textContent = rec.favorite ? '★ Favourite' : '☆ Favourite';
  els.lbFav.classList.toggle('active', !!rec.favorite);
  // Open first: the stage has no size while the dialog is display: none.
  if (!els.lightbox.open) els.lightbox.showModal();
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
  els.lbImg.src = '';
  els.lbError.hidden = true;
  lbList = [];
  lbIndex = -1;
  // A fresh open starts at fit; the view is only carried between images of a
  // walk, not between sessions of looking.
  lbView.key = null;
}
els.lightbox.addEventListener('close', closeLightbox);

// Wheel: zoom about the pointer. Non-passive, or the page behind the dialog
// scrolls with it.
els.lbStage.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoomLightbox(Math.exp(-e.deltaY * 0.002), ...stagePoint(e));
}, { passive: false });

// Drag to pan. A press that never moved is a click: on the stage beside the
// image, it closes the lightbox - but only at the fit size, where nothing is
// being panned and the stage is the backdrop it used to be. Zoomed in, the
// whole stage is a pan surface and only Escape or the cross close it.
let lbDrag = null;
els.lbStage.onpointerdown = (e) => {
  if (e.button !== 0) return;
  lbDrag = { x: e.clientX, y: e.clientY, moved: false };
  els.lbStage.setPointerCapture(e.pointerId);
};
els.lbStage.onpointermove = (e) => {
  if (!lbDrag) return;
  const dx = e.clientX - lbDrag.x, dy = e.clientY - lbDrag.y;
  if (!lbDrag.moved && Math.hypot(dx, dy) < 4) return;  // a jittery click is a click
  lbDrag.moved = true;
  els.lbStage.classList.add('dragging');
  lbView.x += dx; lbView.y += dy;
  lbDrag.x = e.clientX; lbDrag.y = e.clientY;
  layoutLightbox();
};
els.lbStage.onpointerup = (e) => {
  const drag = lbDrag;
  lbDrag = null;
  els.lbStage.classList.remove('dragging');
  if (!drag || drag.moved) return;
  if (e.target === els.lbStage && lbView.scale <= 1) closeLightbox();
};
els.lbStage.onpointercancel = () => { lbDrag = null; els.lbStage.classList.remove('dragging'); };
// Double-click: 1:1 about the point, or back to fit from anywhere else.
els.lbImg.ondblclick = (e) => {
  if (lbView.scale > 1) zoomFit();
  else zoomOneToOne(...stagePoint(e));
};
els.lbFit.onclick = zoomFit;
els.lbOne.onclick = () => zoomOneToOne();
window.addEventListener('resize', layoutLightbox);
