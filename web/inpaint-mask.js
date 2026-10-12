// Local Diffusion's page script: the Inpaint dialog's mask - the canvas, its
// tools, the strokes and their undo, and the mask's export.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------------ the mask

// The mask is a picture of the source's own size: white is repainted, black
// kept, as the inpaint pipelines read it. The canvas's backing bitmap is that
// size too and the brush is in the source's pixels, drawn at whatever scale
// the dialog shows the picture at, so what is painted is where it lands
// whatever the window. Undo keeps strokes (their points, size and tool) and
// replays them rather than keeping pictures: at the refine guard's 8.4 MP a
// snapshot is about 34 MB of RGBA. Past MASK_UNDO strokes the oldest is drawn
// into the base for good, so the list stays bounded. The base is also where
// a Reuse's stored mask starts.
const MASK_UNDO = 200;
// [ and ] and Alt+wheel step the brush by this factor.
const MASK_SIZE_STEP = 1.2;
const MASK_SIZE_MIN = 2;

let maskWidth = 0;
let maskHeight = 0;
// What the strokes start from (a Reuse's mask, and strokes past MASK_UNDO),
// and what they make: white where repainted, its alpha the level. Both
// canvases of the source's size, never in the page.
let maskBase = null;
let maskLayer = null;
// Whether the base holds anything: a stored mask, or strokes drawn into it.
let maskBaseSet = false;
// A scratch canvas for the outline, kept so a frame does not allocate one.
let maskEdge = null;
// {tool: 'brush' | 'eraser' | 'invert' | 'clear', size, points: [[x, y], ...]}
let maskStrokes = [];
let maskRedo = [];
// The stroke being drawn, and what draws it: a pointer's id, or 'keys'.
let maskStroke = null;
let maskTool = 'brush';
let maskSizePx = 64;
let maskFeatherPx = 0;
// The keyboard brush's place, in the source's pixels, and whether it shows:
// it does once an arrow or Space has been used, while the canvas has focus.
let maskCursor = { x: 0, y: 0 };
let maskCursorShown = false;
let maskFrame = 0;
// What the last export measured: whether anything is masked, and how much.
let maskMeasured = { empty: true, coverage: 0 };

function canvas2d(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

// A new mask over a source of width x height: empty, or `kept` - what a close
// that ran nothing left on this image (maskSnapshot) - put back.
function resetMask(width, height, kept = null) {
  maskWidth = width;
  maskHeight = height;
  els.inpaintCanvas.width = width;
  els.inpaintCanvas.height = height;
  maskBase = kept?.base ?? canvas2d(width, height);
  maskBaseSet = !!kept?.baseSet;
  maskLayer = canvas2d(width, height);
  maskEdge = canvas2d(width, height);
  maskStrokes = kept ? [...kept.strokes] : [];
  maskRedo = kept ? [...kept.redo] : [];
  maskStroke = null;
  maskTool = kept?.tool ?? 'brush';
  // A brush a tenth of the picture's shorter side to start with: big enough
  // to see at once, small enough for a face.
  const largest = Math.max(64, Math.round(Math.max(width, height) / 2));
  els.inpaintSize.max = String(largest);
  maskSizePx = kept?.size ?? Math.max(8, Math.round(Math.min(width, height) / 10));
  maskFeatherPx = kept?.feather ?? 0;
  maskCursor = { x: Math.round(width / 2), y: Math.round(height / 2) };
  maskCursorShown = false;
  maskMeasured = kept?.measured ?? { empty: true, coverage: 0 };
  replayMask();
  syncMaskTools();
}

// What a close keeps of the mask for the next open of the same image: the
// strokes, the base they start from and the tool's settings, all in page
// memory. Never in localStorage: a mask is the image's, and large.
function maskSnapshot() {
  return {
    base: maskBase, baseSet: maskBaseSet, strokes: [...maskStrokes], redo: [...maskRedo],
    tool: maskTool, size: maskSizePx, feather: maskFeatherPx, measured: maskMeasured,
  };
}

// Whether anything has been painted since the mask was empty - a stroke, or a
// stored mask in the base. Not whether anything is masked now: an eraser can
// take it all back, which only the export can tell (maskMeasured).
function maskTouched() {
  return maskStrokes.length > 0 || maskBaseSet;
}

// A stored mask (an `L` PNG, white repainted) as the base, its levels as the
// layer's alpha: Reuse on an inpainted record. Answers whether it loaded.
async function loadMaskBase(url) {
  let bitmap;
  try {
    const res = await fetch(url);
    if (!res.ok) return false;
    bitmap = await createImageBitmap(await res.blob());
  } catch { return false; }
  if (bitmap.width !== maskWidth || bitmap.height !== maskHeight) { bitmap.close?.(); return false; }
  const scratch = canvas2d(maskWidth, maskHeight);
  const ctx = scratch.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close?.();
  const pixels = ctx.getImageData(0, 0, maskWidth, maskHeight);
  const data = pixels.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i + 3] = data[i];
    data[i] = data[i + 1] = data[i + 2] = 255;
  }
  maskBase.getContext('2d').putImageData(pixels, 0, 0);
  maskBaseSet = true;
  replayMask();
  syncMaskTools();
  return true;
}

// One stroke onto a context: a dab at its first point, then a segment to each
// next one - the same calls while it is drawn as when it is replayed, so an
// undo and a redo give back exactly the mask that was on screen.
function applyStroke(ctx, stroke) {
  ctx.save();
  if (stroke.tool === 'clear') {
    ctx.clearRect(0, 0, maskWidth, maskHeight);
  } else if (stroke.tool === 'invert') {
    // xor with opaque white leaves 1 - alpha: every level turned over.
    ctx.globalCompositeOperation = 'xor';
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, maskWidth, maskHeight);
  } else {
    ctx.globalCompositeOperation = stroke.tool === 'eraser' ? 'destination-out' : 'source-over';
    const [x, y] = stroke.points[0];
    dab(ctx, x, y, stroke.size);
    for (let i = 1; i < stroke.points.length; i++) segment(ctx, stroke.points[i - 1], stroke.points[i], stroke.size);
  }
  ctx.restore();
}

function dab(ctx, x, y, size) {
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(x, y, size / 2, 0, 2 * Math.PI);
  ctx.fill();
}

function segment(ctx, [x0, y0], [x1, y1], size) {
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = size;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// The layer from scratch: the base, then every stroke kept.
function replayMask() {
  if (!maskLayer) return;
  const ctx = maskLayer.getContext('2d');
  ctx.clearRect(0, 0, maskWidth, maskHeight);
  ctx.drawImage(maskBase, 0, 0);
  for (const stroke of maskStrokes) applyStroke(ctx, stroke);
  drawMaskSoon();
}

function drawMaskSoon() {
  if (!maskFrame) maskFrame = requestAnimationFrame(drawMask);
}

// The canvas the user sees: the mask as a tint, at the levels it will be sent
// with (the feather included), and an outline round its hard edge - nothing
// at all with Show mask off. The source is the picture under it (an <img>),
// so the canvas holds only the mask. Colours are the page's tokens.
function drawMask() {
  maskFrame = 0;
  const ctx = els.inpaintCanvas.getContext('2d');
  ctx.clearRect(0, 0, maskWidth, maskHeight);
  if (!maskLayer || !els.inpaintShow.checked) return;
  const style = getComputedStyle(els.inpaintDialog);
  const tint = style.getPropertyValue('--accent').trim() || '#7c5cff';
  const line = style.getPropertyValue('--accent-2').trim() || '#4ad6c8';
  ctx.save();
  ctx.filter = maskFeatherPx ? `blur(${maskFeatherPx / 2}px)` : 'none';
  ctx.drawImage(maskLayer, 0, 0);
  ctx.filter = 'none';
  ctx.globalCompositeOperation = 'source-in';
  ctx.globalAlpha = 0.45;
  ctx.fillStyle = tint;
  ctx.fillRect(0, 0, maskWidth, maskHeight);
  ctx.restore();
  // The outline: the mask grown by about 1.5 screen pixels, less the mask.
  const shown = els.inpaintCanvas.getBoundingClientRect().width || maskWidth;
  const k = Math.max(1, Math.round(1.5 * maskWidth / shown));
  const edge = maskEdge.getContext('2d');
  edge.save();
  edge.clearRect(0, 0, maskWidth, maskHeight);
  for (const [dx, dy] of [[k, 0], [-k, 0], [0, k], [0, -k]]) edge.drawImage(maskLayer, dx, dy);
  edge.globalCompositeOperation = 'destination-out';
  edge.drawImage(maskLayer, 0, 0);
  edge.globalCompositeOperation = 'source-in';
  edge.fillStyle = line;
  edge.fillRect(0, 0, maskWidth, maskHeight);
  edge.restore();
  ctx.drawImage(maskEdge, 0, 0);
}

// ------------------------------------------------------------- strokes

function beginStroke(point, by) {
  if (maskStroke || !maskLayer) return;
  maskStroke = { tool: maskTool, size: maskSizePx, points: [point], by };
  const ctx = maskLayer.getContext('2d');
  ctx.save();
  ctx.globalCompositeOperation = maskTool === 'eraser' ? 'destination-out' : 'source-over';
  dab(ctx, point[0], point[1], maskSizePx);
  ctx.restore();
  drawMaskSoon();
  inpaintMaskEditing();
}

function extendStroke(point) {
  const stroke = maskStroke;
  if (!stroke) return;
  const last = stroke.points.at(-1);
  // Under a pixel apart adds nothing but points to replay.
  if (Math.hypot(point[0] - last[0], point[1] - last[1]) < 1) return;
  stroke.points.push(point);
  const ctx = maskLayer.getContext('2d');
  ctx.save();
  ctx.globalCompositeOperation = stroke.tool === 'eraser' ? 'destination-out' : 'source-over';
  segment(ctx, last, point, stroke.size);
  ctx.restore();
  drawMaskSoon();
}

function endStroke() {
  const stroke = maskStroke;
  if (!stroke) return;
  maskStroke = null;
  delete stroke.by;
  commitStroke(stroke);
}

// A stroke (or an Invert, or a Clear) kept: undoable, the redo list gone, the
// oldest past the cap drawn into the base for good.
function commitStroke(stroke) {
  maskStrokes.push(stroke);
  maskRedo = [];
  while (maskStrokes.length > MASK_UNDO) {
    applyStroke(maskBase.getContext('2d'), maskStrokes.shift());
    maskBaseSet = true;
  }
  maskEdited();
}

// Every change of what the mask is: the tools' state, and the dialog told,
// which exports and uploads it a moment after the last.
function maskEdited() {
  syncMaskTools();
  inpaintMaskChanged();
}

// --------------------------------------------------------------- tools

function setMaskTool(tool) {
  maskTool = tool;
  syncMaskTools();
}

function setMaskSize(px) {
  const largest = +els.inpaintSize.max || 512;
  maskSizePx = Math.round(Math.min(largest, Math.max(MASK_SIZE_MIN, px)));
  syncMaskTools();
  placeKeyBrush();
}

function stepMaskSize(up) {
  setMaskSize(up ? Math.max(maskSizePx + 1, maskSizePx * MASK_SIZE_STEP) : Math.min(maskSizePx - 1, maskSizePx / MASK_SIZE_STEP));
}

function setMaskFeather(px) {
  maskFeatherPx = Math.max(0, Math.round(px));
  syncMaskTools();
  drawMaskSoon();
  // Baked into the levels sent, so a new feather is a new mask.
  inpaintMaskChanged();
}

function invertMask() {
  if (!maskLayer || maskStroke) return;
  const stroke = { tool: 'invert', size: 0, points: [] };
  applyStroke(maskLayer.getContext('2d'), stroke);
  drawMaskSoon();
  commitStroke(stroke);
}

// Clear is a stroke too, so Undo brings the mask back.
function clearMask() {
  if (!maskLayer || maskStroke) return;
  if (maskButtonHeld(els.inpaintClear)) return;
  const stroke = { tool: 'clear', size: 0, points: [] };
  applyStroke(maskLayer.getContext('2d'), stroke);
  drawMaskSoon();
  commitStroke(stroke);
}

function undoMask() {
  if (maskStroke || maskButtonHeld(els.inpaintUndo)) return;
  maskRedo.push(maskStrokes.pop());
  replayMask();
  maskEdited();
}

function redoMask() {
  if (maskStroke || maskButtonHeld(els.inpaintRedo)) return;
  maskStrokes.push(maskRedo.pop());
  replayMask();
  maskEdited();
}

// A held tool's press says why, on the status line, and does nothing.
function maskButtonHeld(button) {
  if (button.getAttribute('aria-disabled') !== 'true') return false;
  els.inpaintStatus.textContent = `${button.title}.`;
  return true;
}

// The tools' state: the pressed tool, the sliders' readouts, Undo, Redo and
// Clear held with the reason where there is nothing for them, and the
// status line.
function syncMaskTools() {
  for (const [button, tool] of [[els.inpaintBrushTool, 'brush'], [els.inpaintEraserTool, 'eraser']]) {
    button.setAttribute('aria-pressed', String(maskTool === tool));
  }
  els.inpaintSize.value = String(maskSizePx);
  els.inpaintSizeOut.textContent = `${maskSizePx} px`;
  els.inpaintFeather.value = String(maskFeatherPx);
  els.inpaintFeatherOut.textContent = `${maskFeatherPx} px`;
  const hold = (button, why, title) => {
    button.setAttribute('aria-disabled', String(!!why));
    button.title = why || title;
  };
  hold(els.inpaintUndo, maskStrokes.length ? '' : 'Nothing to undo', 'Undo the last stroke (Ctrl+Z)');
  hold(els.inpaintRedo, maskRedo.length ? '' : 'Nothing to redo', 'Redo the stroke undone last (Ctrl+Shift+Z or Ctrl+Y)');
  hold(els.inpaintClear, maskTouched() && !maskMeasured.empty ? '' : 'Nothing is masked to clear',
    'Clear the mask (Undo brings it back)');
  syncMaskStatus();
}

// One line: the tool, its size and feather, and how much of the picture the
// mask covers - from the last export, so it lags the brush by a moment.
function syncMaskStatus() {
  const tool = maskTool === 'eraser' ? 'Eraser' : 'Brush';
  const covered = !maskTouched() ? 'nothing masked yet - paint the part to repaint'
    : maskMeasured.empty ? 'nothing masked'
      : `${maskMeasured.coverage < 1 ? '<1' : Math.round(maskMeasured.coverage)}% of the picture masked`;
  const text = `${tool} ${maskSizePx} px · feather ${maskFeatherPx} px · ${covered}`;
  if (els.inpaintStatus.textContent !== text) els.inpaintStatus.textContent = text;
}

// --------------------------------------------------------------- export

// The mask as it will be sent: one grey level per pixel (white repainted),
// the feather baked in as a blur of the levels, over black - the canvas as a
// PNG, which the server reads as one `L` channel (POST /api/masks). With
// whether anything is masked and how much (levels from 128 up, which is what
// the pipelines' binarised mask repaints).
async function exportMask() {
  const out = canvas2d(maskWidth, maskHeight);
  const ctx = out.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, maskWidth, maskHeight);
  ctx.filter = maskFeatherPx ? `blur(${maskFeatherPx / 2}px)` : 'none';
  ctx.drawImage(maskLayer, 0, 0);
  ctx.filter = 'none';
  const data = ctx.getImageData(0, 0, maskWidth, maskHeight).data;
  let any = false;
  let covered = 0;
  for (let i = 0; i < data.length; i += 4) {
    const level = data[i];
    if (level) any = true;
    if (level >= 128) covered += 1;
  }
  maskMeasured = { empty: !any, coverage: (100 * covered) / (maskWidth * maskHeight) };
  syncMaskTools();
  if (!any) return { blob: null, ...maskMeasured };
  const blob = await new Promise((resolve) => out.toBlob(resolve, 'image/png'));
  return { blob, ...maskMeasured };
}

// --------------------------------------------------------- pointer and keys

// A pointer's place in the source's pixels, from where the canvas is drawn
// now: the canvas never resizes under a stroke, but the dialog may have
// scrolled since the press.
function maskPoint(event) {
  const rect = els.inpaintCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * maskWidth;
  const y = ((event.clientY - rect.top) / rect.height) * maskHeight;
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}

function maskPointerDown(event) {
  if (event.button !== 0 || !inpaintSource || maskStroke) return;
  event.preventDefault();
  els.inpaintCanvas.focus({ preventScroll: true });
  // Captured, so the stroke goes on past the picture's edge and ends there;
  // a pointer the browser no longer has (a synthetic one) paints uncaptured.
  try { els.inpaintCanvas.setPointerCapture(event.pointerId); } catch { /* not an active pointer */ }
  maskCursorShown = false;
  beginStroke(maskPoint(event), event.pointerId);
  placeBrush(event.clientX, event.clientY);
}

function maskPointerMove(event) {
  placeBrush(event.clientX, event.clientY);
  if (maskStroke?.by !== event.pointerId) return;
  for (const e of event.getCoalescedEvents?.() ?? [event]) extendStroke(maskPoint(e));
}

function maskPointerUp(event) {
  if (maskStroke?.by === event.pointerId) endStroke();
}

function maskPointerLeave() {
  if (!maskStroke && !maskCursorShown) els.inpaintBrush.hidden = true;
}

// The brush's ring, centred on a point of the page, at the brush's size as
// the picture is shown.
function placeBrush(clientX, clientY) {
  const rect = els.inpaintCanvas.getBoundingClientRect();
  const stage = els.inpaintStage.getBoundingClientRect();
  if (!rect.width) return;
  const across = (maskSizePx * rect.width) / maskWidth;
  Object.assign(els.inpaintBrush.style, {
    left: `${clientX - stage.left}px`, top: `${clientY - stage.top}px`,
    width: `${across}px`, height: `${across}px`,
  });
  els.inpaintBrush.hidden = false;
}

// The keyboard brush's ring, where the arrow keys have put it.
function placeKeyBrush() {
  if (!maskCursorShown) return;
  const rect = els.inpaintCanvas.getBoundingClientRect();
  placeBrush(rect.left + (maskCursor.x / maskWidth) * rect.width, rect.top + (maskCursor.y / maskHeight) * rect.height);
}

// Alt+wheel over the picture sizes the brush; a plain wheel scrolls the
// dialog, as it does anywhere else.
function maskWheel(event) {
  if (!event.altKey || !event.deltaY) return;
  event.preventDefault();
  stepMaskSize(event.deltaY < 0);
}

// The canvas's keys, only while it has focus: B and E pick the tool, [ and ]
// size it, I inverts, Ctrl+Z undoes and Ctrl+Shift+Z or Ctrl+Y redoes; the
// arrow keys move a keyboard brush (Shift for four times the step) and Space
// paints with it, held down along the way it moves. Each is taken, so a key
// here never scrolls the dialog or reaches the page.
function maskKeydown(event) {
  if (event.isComposing) return;
  const key = event.key;
  if (event.ctrlKey || event.metaKey) {
    if (event.altKey) return;
    const lower = key.toLowerCase();
    if (lower === 'z' && !event.shiftKey) { event.preventDefault(); undoMask(); }
    else if ((lower === 'z' && event.shiftKey) || lower === 'y') { event.preventDefault(); redoMask(); }
    return;
  }
  if (event.altKey) return;
  const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  if (arrows[key]) {
    event.preventDefault();
    const step = Math.max(1, Math.round(maskSizePx / 4)) * (event.shiftKey ? 4 : 1);
    maskCursor = {
      x: Math.min(maskWidth, Math.max(0, maskCursor.x + arrows[key][0] * step)),
      y: Math.min(maskHeight, Math.max(0, maskCursor.y + arrows[key][1] * step)),
    };
    maskCursorShown = true;
    placeKeyBrush();
    if (maskStroke?.by === 'keys') extendStroke([maskCursor.x, maskCursor.y]);
    return;
  }
  if (key === ' ') {
    event.preventDefault();
    if (event.repeat) return;
    maskCursorShown = true;
    placeKeyBrush();
    beginStroke([maskCursor.x, maskCursor.y], 'keys');
    return;
  }
  const lower = key.toLowerCase();
  if (lower === 'b') { event.preventDefault(); setMaskTool('brush'); }
  else if (lower === 'e') { event.preventDefault(); setMaskTool('eraser'); }
  else if (lower === 'i') { event.preventDefault(); invertMask(); }
  else if (key === '[') { event.preventDefault(); stepMaskSize(false); }
  else if (key === ']') { event.preventDefault(); stepMaskSize(true); }
}

function maskKeyup(event) {
  if (event.key === ' ' && maskStroke?.by === 'keys') {
    event.preventDefault();
    endStroke();
  }
}

// Focus gone: a keyboard stroke ends where it is, and the keyboard's ring goes.
function maskBlur() {
  if (maskStroke?.by === 'keys') endStroke();
  if (maskCursorShown) {
    maskCursorShown = false;
    els.inpaintBrush.hidden = true;
  }
}
