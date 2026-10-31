// AuraDream's page script: contact-sheet zoom and pan.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// Zoom: a matrix's cells drawn larger than Fit or Fill draws them, for the
// detail a CFG or steps sweep differs in, the box scrolling under its stuck
// heads and a drag panning it (sheetPointerDown). A factor on the cells'
// width at zoom 1 (sheetZoomBase, which fitSheet measures with the zoom
// taken off), from 1 up to SHEET_ZOOM_MAX screen pixels to the image pixel,
// the lightbox's steps (ZOOM_STEP); `sizes` follows it, so a zoomed cell
// shows the PNG rather than the thumbnail. Not a preference: every sheet
// opens at 1 (setSheet), and a redraw of the same sheet - a page landing, a
// pin, a job writing to the run - keeps it. A wrapped row lays out like the
// gallery and is not zoomed. The export draws the unzoomed cells.
const SHEET_ZOOM_MAX = 4;
let sheetZoom = 1;
let sheetZoomBase = 0;
const ZOOM_WRAPPED = 'A wrapped row lays out like the gallery: zoom is for a matrix';
const ZOOM_EMPTY = 'Nothing on the sheet to zoom';

// The pixels across of the sheet's images, the commonest - its cells take
// the commonest shape (renderSheet) - and the zoom that draws them 1:1: a
// cell's picture is its width less the 1 px border each side.
function sheetImageWidth() {
  const widths = new Map();
  for (const rec of sheetLayout?.cells ?? []) widths.set(rec.width, (widths.get(rec.width) || 0) + 1);
  return [...widths.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
}
function sheetOneToOne() {
  const width = sheetImageWidth();
  return width && sheetZoomBase ? (width / devicePixelRatio + 2) / sheetZoomBase : 0;
}
function sheetZoomMax() {
  const width = sheetImageWidth();
  return width && sheetZoomBase
    ? Math.max(1, ((width * SHEET_ZOOM_MAX) / devicePixelRatio + 2) / sheetZoomBase) : 1;
}
function sheetZoomable() {
  return !!sheetLayout && !sheetLayout.wrap && sheetZoomBase > 0 && sheetLayout.shown > 0;
}

// The zoomed width on the grid itself, as the template's column (renderSheet:
// `minmax(var(--cell-min), var(--cell-w))`), over Fit's --cell-max and Fill's
// 1fr alike; at 1 the stylesheet's own.
function applySheetZoom() {
  const grid = els.sheetGrid;
  const on = sheetZoom > 1;
  els.sheet.classList.toggle('zoomed', on);
  if (!on) {
    grid.style.removeProperty('--cell-w');
    grid.style.removeProperty('--cell-min');
    return;
  }
  const width = `${(sheetZoomBase * sheetZoom).toFixed(2)}px`;
  grid.style.setProperty('--cell-w', width);
  grid.style.setProperty('--cell-min', width);
}

// Zoom to `zoom` keeping the picture under `point` (client coordinates)
// where it is - the wheel's pointer, a key's focused cell - or under the
// middle of what the box shows, for a button. Pinned to the cell under the
// point, or the nearest one where the point is a gap or a head, by where in
// that cell it falls: the heads keep their size as the cells grow, so the
// grid does not scale about any one point of it. `say`: the new zoom is
// said at once, a key's or a button's, rather than once the wheel's burst
// is over (writeSheetZoom).
function setSheetZoom(zoom, point = null, say = true) {
  if (!sheetZoomable()) return;
  const next = Math.min(sheetZoomMax(), Math.max(1, zoom));
  if (Math.abs(next - sheetZoom) < 1e-6) return;
  const scroller = els.sheetGrid.parentElement;
  const at = point ?? sheetMiddle(scroller);
  const cell = nearestSheetCell(at);
  const was = cell?.getBoundingClientRect();
  sheetZoom = Math.abs(next - 1) < 1e-3 ? 1 : next;
  applySheetZoom();
  if (cell) {
    const now = cell.getBoundingClientRect();
    scroller.scrollLeft += now.left + ((at.x - was.left) / was.width) * now.width - at.x;
    scroller.scrollTop += now.top + ((at.y - was.top) / was.height) * now.height - at.y;
  }
  sizeSheetImages();
  syncZoomControls();
  writeSheetZoom(say);
}
const zoomSheet = (factor, point = null, say = true) => setSheetZoom(sheetZoom * factor, point, say);

// The middle of what the box shows of the cells: clear of the stuck heads.
function sheetMiddle(scroller) {
  const box = scroller.getBoundingClientRect();
  const style = getComputedStyle(scroller);
  const left = box.left + scroller.clientLeft + (parseFloat(style.scrollPaddingLeft) || 0);
  const top = box.top + scroller.clientTop + (parseFloat(style.scrollPaddingTop) || 0);
  const right = box.left + scroller.clientLeft + scroller.clientWidth;
  const bottom = box.top + scroller.clientTop + scroller.clientHeight;
  return { x: (left + right) / 2, y: (top + bottom) / 2 };
}

function nearestSheetCell({ x, y }) {
  let best = null, bestDistance = Infinity;
  for (const cell of els.sheetGrid.querySelectorAll('.sheet-cell')) {
    const r = cell.getBoundingClientRect();
    const dx = Math.max(r.left - x, 0, x - r.right), dy = Math.max(r.top - y, 0, y - r.bottom);
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) { best = cell; bestDistance = distance; }
    if (!distance) break;
  }
  return best;
}

// Each image's `sizes`, the width its cell is drawn at - fitted or filled,
// the cells of a sheet are one width - which picks the thumbnail or the PNG.
// Zoomed, every cell shows its PNG outright, whatever srcset would pick: the
// thumbnail is a lossy 384 px WEBP, and a zoom is asked for to see the
// detail it loses. Lazy loading still fetches only the cells in the box's
// view; one swapped stays the PNG, a zoom back out included.
function sizeSheetImages() {
  const width = sheetCellWidth();
  if (!width) return;
  const sizes = `${Math.ceil(width)}px`;
  for (const img of els.sheetGrid.querySelectorAll('.sheet-cell img[data-full]')) {
    if (sheetZoom > 1 && img.getAttribute('src') !== img.dataset.full) {
      img.removeAttribute('srcset');
      img.removeAttribute('sizes');
      img.src = img.dataset.full;
    } else if (img.hasAttribute('srcset')) img.sizes = sizes;
  }
}

// The readout: the image's pixels to the screen's, as the lightbox says it -
// "fit · 33%" (or "fill · …") at 1, "1:1", "250%".
function sheetZoomText() {
  const width = sheetImageWidth();
  if (!sheetZoomable() || !width) return '';
  const scale = ((sheetZoomBase * sheetZoom - 2) * devicePixelRatio) / width;
  const percent = Math.round(scale * 100);
  if (Math.abs(scale - 1) < 5e-3) return '1:1';
  return sheetZoom === 1 ? `${sheetFill() ? 'fill' : 'fit'} · ${percent}%` : `${percent}%`;
}

// Written on screen at once and said - #sheet-zoom-said - at once for a key
// or a button, else once the wheel's burst is over: every wheel step said
// was a stream of percentages.
let sheetZoomTimer;
function writeSheetZoom(say) {
  const text = sheetZoomText();
  const readout = els.sheetControls.querySelector('.sheet-zoom-level');
  if (readout) readout.textContent = text;
  clearTimeout(sheetZoomTimer);
  const sayIt = () => { if (els.sheetZoomSaid.textContent !== text) els.sheetZoomSaid.textContent = text; };
  if (say) sayIt();
  else sheetZoomTimer = setTimeout(sayIt, 400);
}

// The zoom's buttons, held with their reason where they would do nothing.
function syncZoomControls() {
  const group = els.sheetControls.querySelector('.sheet-zoom');
  if (!group) return;
  const zoomable = sheetZoomable();
  const one = sheetOneToOne();
  const held = zoomable ? '' : sheetLayout?.wrap ? ZOOM_WRAPPED : ZOOM_EMPTY;
  const reasons = {
    out: held || (sheetZoom <= 1 ? 'The cells are at the sheet\'s own size: + or Ctrl+wheel zooms in' : ''),
    in: held || (sheetZoom >= sheetZoomMax() - 1e-6
      ? `At ${SHEET_ZOOM_MAX * 100}%, ${SHEET_ZOOM_MAX} screen pixels to the image pixel, the most the sheet zooms` : ''),
    one: held || (one <= 1 ? 'The cells already draw each image larger than its own pixels'
      : Math.abs(sheetZoom - one) < 1e-3 ? 'The cells already show each image 1:1' : ''),
    level: held || (sheetZoom <= 1 ? 'The cells are at the sheet\'s own size' : ''),
  };
  const titles = {
    out: 'Zoom out (−)',
    in: 'Zoom in (+); Ctrl+wheel zooms about the pointer, and a drag pans',
    one: 'One image pixel per screen pixel (1)',
    level: 'Back to the sheet\'s own size (0)',
  };
  for (const [key, reason] of Object.entries(reasons)) {
    const button = group.querySelector(`[data-zoom="${key}"]`);
    button.setAttribute('aria-disabled', String(!!reason));
    button.title = reason || titles[key];
  }
  group.querySelector('.sheet-zoom-level').textContent = sheetZoomText() || '—';
}

function zoomControls() {
  const group = document.createElement('span');
  group.className = 'sheet-zoom';
  group.setAttribute('role', 'group');
  group.setAttribute('aria-label', 'Zoom');
  const actions = {
    out: () => zoomSheet(1 / ZOOM_STEP),
    in: () => zoomSheet(ZOOM_STEP),
    one: () => setSheetZoom(sheetOneToOne()),
    level: () => setSheetZoom(1),
  };
  const labels = { out: ['−', 'Zoom out'], in: ['+', 'Zoom in'], one: ['1:1', 'One to one'], level: ['', 'Zoom level'] };
  for (const key of ['out', 'in', 'one', 'level']) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `ghost small${key === 'level' ? ' sheet-zoom-level' : ''}`;
    button.dataset.zoom = key;
    const [text, name] = labels[key];
    button.textContent = text;
    // The readout's name is its words, which change; the glyphs' are fixed.
    if (key !== 'level') button.setAttribute('aria-label', name);
    button.onclick = () => {
      if (button.getAttribute('aria-disabled') === 'true') {
        showError(`${button.title}.`, true, { slot: els.galleryError });
        return;
      }
      actions[key]();
    };
    group.appendChild(button);
  }
  // Held until fitSheet syncs them once the cells are drawn; a sheet with
  // nothing laid out never gets there.
  queueMicrotask(syncZoomControls);
  return group;
}

// The zoom keys on a focused cell, the lightbox's: + and − step, 0 is the
// sheet's own size, 1 is 1:1, each about the cell. Held, they say why as
// the buttons do.
function sheetZoomKey(key, cell) {
  const r = cell.getBoundingClientRect();
  const point = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  const which = key === '+' || key === '=' ? 'in' : key === '-' || key === '_' ? 'out' : key === '0' ? 'level' : 'one';
  const button = els.sheetControls.querySelector(`.sheet-zoom [data-zoom="${which}"]`);
  if (!button) return;
  if (button.getAttribute('aria-disabled') === 'true') {
    showError(`${button.title}.`, true, { slot: els.galleryError });
    return;
  }
  if (which === 'in') zoomSheet(ZOOM_STEP, point);
  else if (which === 'out') zoomSheet(1 / ZOOM_STEP, point);
  else if (which === 'level') setSheetZoom(1, point);
  else setSheetZoom(sheetOneToOne(), point);
}

// Ctrl+wheel - and a touchpad's pinch, which the browser sends as one -
// zooms about the pointer; a plain wheel still scrolls the box and the page.
// Wired non-passive (app.js), or the browser zooms the page with it.
function sheetWheel(e) {
  if (!e.ctrlKey || !sheetZoomable()) return;
  e.preventDefault();
  zoomSheet(Math.exp(-e.deltaY * 0.002), { x: e.clientX, y: e.clientY }, false);
}

// Drag to pan the box, wherever it scrolls - zoomed, or filled past the
// window's height. A press that never moved 4 px is a click, a cell's
// lightbox as before; one that did is no click (sheetClickAfterPan). Not on
// a button, which keeps its press, nor a touch, which scrolls the box
// itself. The image's own drag is the browser's otherwise, and takes the
// pointer from the pan (sheetDragStart).
let sheetPan = null;
let sheetPanned = false;
const sheetPannable = (box) => box.scrollWidth > box.clientWidth + 1 || box.scrollHeight > box.clientHeight + 1;

function sheetPointerDown(e) {
  const box = e.currentTarget;
  if (e.button !== 0 || e.pointerType === 'touch' || e.target.closest('button, select') || !sheetPannable(box)) return;
  sheetPan = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: false };
}

function sheetPointerMove(e) {
  if (!sheetPan || e.pointerId !== sheetPan.id) return;
  const box = e.currentTarget;
  // Released outside the box before the pan took the pointer: over.
  if (!(e.buttons & 1)) { sheetPointerCancel(e); return; }
  const dx = e.clientX - sheetPan.x, dy = e.clientY - sheetPan.y;
  if (!sheetPan.moved) {
    if (Math.hypot(dx, dy) < 4) return;  // a jittery click is a click
    sheetPan.moved = true;
    box.setPointerCapture(e.pointerId);
    box.classList.add('panning');
    getSelection()?.removeAllRanges();
  }
  box.scrollLeft -= dx;
  box.scrollTop -= dy;
  sheetPan.x = e.clientX;
  sheetPan.y = e.clientY;
}

function sheetPointerUp(e) {
  if (!sheetPan || e.pointerId !== sheetPan.id) return;
  sheetPanned = sheetPan.moved;
  sheetPan = null;
  e.currentTarget.classList.remove('panning');
  // The click this release makes comes before any timer.
  if (sheetPanned) setTimeout(() => { sheetPanned = false; });
}

function sheetPointerCancel(e) {
  sheetPan = null;
  e.currentTarget.classList.remove('panning');
}

// Capturing, on the box: a drag's release is no press of the cell under it.
function sheetClickAfterPan(e) {
  if (!sheetPanned) return;
  sheetPanned = false;
  e.preventDefault();
  e.stopPropagation();
}

function sheetDragStart(e) {
  if (sheetPannable(e.currentTarget)) e.preventDefault();
}
