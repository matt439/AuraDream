// Local Diffusion's page script: contact-sheet PNG export.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------ sheet export

// The one artefact worth keeping from a night is the matrix with its labels
// burnt in, and a screenshot is capped at the viewport and drops the pinned
// caption. Drawn client-side from the layout the sheet is already showing -
// a server route would duplicate a layout contract the client holds, the
// non-round-trippable prompt labels especially. Thumbnails, not the PNGs, on a
// fitted sheet: a contact sheet is for seeing which cell, and 384 px cells of
// a 32-wide grid are already a 12k-pixel image. A filled one (Fill width) is
// drawn at the larger of the device pixels its cells take on screen and the
// thumbnail's width - never smaller than the fitted export - from the PNG
// where that is wider than the thumbnail, as the sheet's own srcset is.
let sheetLayout = null;
const EXPORT_MAX_EDGE = 8000;
// Export all past the edge sends one file per slice, up to this many: the
// browser asks before a page downloads several, and a hundred files was
// nobody's intent.
const EXPORT_FILES_MAX = 24;
// The most a caption can take (the search, three lines of prompt, two of
// variables, two of Fixed, the checkpoint's): the room a stack keeps under
// its slices.
const EXPORT_CAPTION_MAX = 12 + 9 * 20;

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);  // a missing file is an empty cell, not a failed export
    img.src = src;
  });
}

// The sheet as an export draws it, taken when the export is asked for: the
// layout, the axes and the words its labels are made of (sheetWords), the
// search it shows and, on a filled matrix, the width its cells are drawn at
// and the device pixels to the CSS pixel - so a redraw while the images load,
// or while Export all writes its slices, changes none of it. Read live, a
// sheet redrawn on another run halfway through Export all threw on the next
// slice's caption.
function exportLayout() {
  if (!sheetLayout) return null;
  const fill = sheetFill() && !sheetLayout.wrap;
  return {
    ...sheetLayout, axes: sheetAxesList, words: sheetWords(), search: els.search.value.trim(),
    shown: fill ? sheetCellWidth() / sheetZoom : 0, dpr: window.devicePixelRatio || 1,
  };
}

// The measures of an exported sheet: the gap, the type, the caption under a
// wrapped cell, the column heads' row, the row heads' column, and the title
// over each slice where several are stacked.
function exportMetrics({ colAxis, rowAxis, wrap, source }, titled) {
  return {
    gap: 6, font: 14, labelH: wrap ? 22 : 0,
    headH: (colAxis || source) && !wrap ? 34 : 8, headW: rowAxis ? 150 : 8,
    titleH: titled ? 30 : 0,
  };
}

// Where each cell goes, in the sheet's reading order, given one slice's cells.
// A wrapped row wraps here too, into a block about as wide as it is tall, each
// cell captioned underneath: drawn as one row, the 96 px floor beat the edge
// limit and a 120-seed run came out 12,254 px wide. A refine sweep's source
// leads, a column of its own across the rows.
function exportGrid({ colAxis, colValues, rowValues, wrap, source, words }) {
  if (wrap) {
    const n = colValues.length + (source ? 1 : 0);
    const cols = Math.ceil(Math.sqrt(n));
    const place = (cellOf) => {
      const row = colValues.map((v, c) => ({ rec: cellOf.get(c), label: headText(colAxis, v, words) }));
      if (source) row.unshift({ rec: source, label: `Source #${source.id}` });
      return row.map((item, i) => ({ ...item, col: i % cols, row: Math.floor(i / cols) }));
    };
    return { cols, rows: Math.ceil(n / cols), lead: 0, place };
  }
  const lead = source ? 1 : 0;
  const place = (cellOf) => {
    const items = source ? [{ rec: source, col: 0, row: 0, span: true }] : [];
    rowValues.forEach((_, r) => colValues.forEach((__, c) => {
      items.push({ rec: cellOf.get(r * colValues.length + c), col: lead + c, row: r });
    }));
    return items;
  };
  return { cols: colValues.length + lead, rows: rowValues.length, lead, place };
}

// A cell's width in the export: the thumbnail's at most, fitted; filled, the
// device pixels the sheet draws its cells at, or the thumbnail's where that
// is more - either shrunk so the longer edge stays inside what a canvas can
// hold, and never past the sheet's 96 px floor. Filled at its CSS width, a
// 1400 px window's 228 px cells exported smaller than the fitted sheet's
// 384, and at 125% the file lost a fifth of what the screen showed.
function exportCellWidth(layout) {
  const { cols, rows } = exportGrid(layout);
  const most = layout.shown ? Math.max(THUMB_SIDE, Math.round(layout.shown * layout.dpr)) : THUMB_SIDE;
  return Math.max(SHEET_CELL_MIN, Math.min(most, Math.floor(EXPORT_MAX_EDGE / Math.max(cols, rows))));
}

// The cell width that stacks `n` slices inside the edge: one slice's width,
// or less, down to the 96 px floor; null where even that is too tall.
function stackedCellWidth(layout, n) {
  const { rows } = exportGrid(layout);
  const { gap, labelH, headH, titleH } = exportMetrics(layout, true);
  const height = (w) => n * (titleH + headH + rows * (Math.round(w / layout.aspect) + labelH + gap) + gap)
    + EXPORT_CAPTION_MAX;
  let w = exportCellWidth(layout);
  while (w >= SHEET_CELL_MIN && height(w) > EXPORT_MAX_EDGE) w -= 1;
  return w >= SHEET_CELL_MIN ? w : null;
}

// A slice's pins in words, "size big, ground grass": the caption's "pinned
// to", and a stacked slice's title.
const pinsText = (pins, words = sheetWords()) => pins
  .map(([axis, v]) => `${axis.label.toLowerCase()} ${clip(sheetText(axis, v, words))}`).join(', ');

// `slices` ({pins, cellOf} each) are drawn one under another, each titled by
// its pins; without them, the slice on screen. `cellW` is a stack's
// (stackedCellWidth).
async function buildSheetCanvas(slices = null, { layout = exportLayout(), cellW = null } = {}) {
  if (!layout) return null;
  const { colAxis, rowAxis, colValues, rowValues, values, wrap, source, records, aspect, axes, words } = layout;
  const head = (axis, v) => headText(axis, v, words);
  const blocks = slices ?? [{ pins: layout.pins, cellOf: layout.cellOf }];
  const titled = blocks.length > 1;
  const { cols, rows, lead, place } = exportGrid(layout);
  const cellWidth = cellW ?? exportCellWidth(layout);
  const cellH = Math.round(cellWidth / aspect);
  const { gap, font, labelH, headH, headW, titleH } = exportMetrics(layout, titled);
  const width = headW + cols * (cellWidth + gap) + gap;
  const blockH = titleH + headH + rows * (cellH + labelH + gap) + gap;
  const gridH = blocks.length * blockH;
  const typeface = `${font}px "Segoe UI", system-ui, sans-serif`;
  // Measured apart from the canvas, whose height waits on the caption's lines.
  const measure = document.createElement('canvas').getContext('2d');
  measure.font = typeface;
  const clipText = (text, max) => {
    let t = text;
    while (t.length > 1 && measure.measureText(t).width > max) t = `${t.slice(0, -2)}…`;
    return t;
  };
  // Prose over a few lines at most, the last one clipped.
  const wrapText = (text, max, most) => {
    const out = [''];
    for (const word of text.trim().split(/\s+/)) {
      const next = out.at(-1) ? `${out.at(-1)} ${word}` : word;
      if (out.at(-1) && measure.measureText(next).width > max) out.push(word);
      else out[out.length - 1] = next;
    }
    if (out.length > most) {
      out.length = most;
      out[most - 1] += ' …';
    }
    return out.map((line) => clipText(line, max));
  };

  // The caption is what a screenshot loses: which run, what was pinned - or,
  // stacked, which axes the slices run through - what prompt made it, with a
  // variable run's values as declared, since its template is not a prompt any
  // cell ran, what every image shares, and when the run was made: the oldest
  // image's time, not the export's.
  const placed = blocks.map((block) => place(block.cellOf));
  const first = records[0];
  const oldest = first && records.reduce((a, r) => (r.id < a.id ? r : a));
  const opening = placed.flat().find((item) => item.rec && item.rec !== source)?.rec;
  const prompt = values.prompt?.length === 1 ? `Prompt: ${values.prompt[0]}`
    : opening ? `Prompt of the first image: ${opening.prompt}` : '';
  const declared = axes.filter((a) => a.key.startsWith('var:'))
    .map((a) => `${a.label} = ${values[a.key].map((v) => sheetText(a, v, words)).join(' | ')}`).join('; ');
  const fixed = fixedSettings(values, axes, words);
  const max = width - 2 * gap;
  const scope = titled
    ? ` · a slice per ${blocks[0].pins.map(([axis]) => axis.label.toLowerCase()).join(' × ')}`
    : blocks[0].pins.length ? ` · pinned to ${pinsText(blocks[0].pins, words)}` : '';
  const lines = [
    clipText(`${layout.search || 'gallery'}${scope}`, max),
    ...(prompt ? wrapText(prompt, max, 3) : []),
    ...(declared ? wrapText(declared, max, 2) : []),
    ...(fixed ? wrapText(fixed, max, 2) : []),
    first ? clipText(`${first.model_id.split('/').pop()} · ${first.width}×${first.height}` +
      (first.run_id ? ` · run ${first.run_id}` : '') + ` · ${when(Date.parse(oldest.created_at))}`, max) : '',
  ].filter(Boolean);
  const caption = 12 + lines.length * 20;
  const height = gridH + caption;

  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b0b0e';
  ctx.fillRect(0, 0, width, height);
  ctx.font = typeface;
  ctx.textBaseline = 'middle';

  // A filled sheet's cells drawn wider than the thumbnail take the PNG.
  const src = (rec) => (layout.shown && rec.url && cellWidth > thumbWidthOf(rec) ? rec.url : rec.thumb_url ?? rec.url);
  // Every cell's image in flight at once; a hole stays a hole. The heads,
  // the titles and the captions under wrapped cells are drawn before any
  // image lands.
  const jobs = [];
  blocks.forEach((block, b) => {
    const top = b * blockH + titleH;
    // A stacked slice's title, in the accent's light violet over the white
    // heads, left where the eye starts each block, under a rule from the one
    // before.
    if (titled) {
      if (b) {
        ctx.fillStyle = '#26262f';
        ctx.fillRect(gap, b * blockH, width - 2 * gap, 1);
      }
      ctx.fillStyle = '#c4b5fd';
      ctx.textAlign = 'left';
      ctx.fillText(clipText(pinsText(block.pins, words), max), gap, b * blockH + titleH / 2 + 2);
    }
    ctx.fillStyle = '#e8e8ee';
    ctx.textAlign = 'center';
    if (source && !wrap) ctx.fillText(clipText(`Source #${source.id}`, cellWidth), headW + gap + cellWidth / 2, top + headH / 2);
    colValues.forEach((v, c) => {
      if (!colAxis || wrap) return;
      const x = headW + gap + (lead + c) * (cellWidth + gap) + cellWidth / 2;
      ctx.fillText(clipText(head(colAxis, v), cellWidth), x, top + headH / 2);
    });
    ctx.textAlign = 'right';
    rowValues.forEach((v, r) => {
      if (!rowAxis) return;
      const y = top + headH + gap + r * (cellH + gap) + cellH / 2;
      ctx.fillText(clipText(head(rowAxis, v), headW - 12), headW - 6, y);
    });

    for (const item of placed[b]) {
      const x = headW + gap + item.col * (cellWidth + gap);
      const y = top + (item.span
        ? headH + gap + (rows * (cellH + gap) - gap - cellH) / 2
        : headH + gap + item.row * (cellH + labelH + gap));
      ctx.fillStyle = '#1a1a21';
      ctx.fillRect(x, y, cellWidth, cellH);
      if (item.label) {
        ctx.fillStyle = '#e8e8ee';
        ctx.textAlign = 'center';
        ctx.fillText(clipText(item.label, cellWidth), x + cellWidth / 2, y + cellH + labelH / 2);
      }
      const { rec } = item;
      if (!rec) continue;
      jobs.push(loadImage(src(rec)).then((img) => {
        if (!img) return;
        // Contain, as the sheet does: the whole frame, letterboxed if it must be.
        const k = Math.min(cellWidth / img.naturalWidth, cellH / img.naturalHeight);
        const w = img.naturalWidth * k, h = img.naturalHeight * k;
        ctx.drawImage(img, x + (cellWidth - w) / 2, y + (cellH - h) / 2, w, h);
        if (rec.favorite) {
          ctx.fillStyle = '#fbbf24';
          ctx.textAlign = 'right';
          ctx.fillText('★', x + cellWidth - 6, y + 12);
        }
      }));
    }
  });
  await Promise.all(jobs);

  ctx.fillStyle = '#8b8b9a';
  ctx.textAlign = 'left';
  lines.forEach((line, i) => ctx.fillText(line, gap, gridH + 16 + i * 20));
  return canvas;
}

// Named by the run and the pins: every slice of a run used to download as
// sheet-<run>.png, told apart only by the browser's " (1)". A stack of every
// slice is named by the axes its slices run through instead
// (`sheet-<run>-every-cfg_seed.png`). An export names its files from its own
// layout (exportLayout); without one, the sheet on screen.
function sheetFileName(pins = sheetLayout?.pins ?? [], every = false, layout = null) {
  const first = (layout?.records ?? sheetRecords)[0];
  const words = layout?.words ?? sheetWords();
  const parts = every
    ? [`every-${pins.map(([axis]) => axis.key.replace(/^var:/, '')).join('_')}`]
    : pins.map(([axis, v]) => `${axis.key.replace(/^var:/, '')}-${clip(sheetText(axis, v, words))}`);
  const name = `sheet-${first?.run_id ?? 'gallery'}${parts.length ? `-${parts.join('_')}` : ''}`;
  // A value ending in a mark - "none (1.79)", a clipped label's "…" - leaves
  // no hyphen dangling before the next pin or the extension.
  return `${name.replace(/[^\w.-]+/g, '-').replace(/-+(?=_|$)/g, '')}.png`;
}

// One image to the browser's downloads: true once handed over, false where
// the browser could not encode it, which is said.
function saveCanvas(canvas, name) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        showError('The browser could not encode the sheet.', false, { slot: els.galleryError });
        resolve(false);
        return;
      }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      resolve(true);
    }, 'image/png');
  });
}

// An export under way holds both buttons, aria-disabled with the reason:
// Export all draws its slices one after another, and a second press would
// draw them all again beside the first.
let sheetExporting = false;
const EXPORT_BUSY = 'The sheet is still being exported';
function syncExportButton(button) {
  button.setAttribute('aria-disabled', String(sheetExporting));
  button.title = sheetExporting ? EXPORT_BUSY : button.dataset.title;
}
function exportButtonFor(text, title, run) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost small sheet-export';
  button.textContent = text;
  button.dataset.title = title;
  button.onclick = async () => {
    if (sheetExporting) { showError(`${EXPORT_BUSY}.`, true, { slot: els.galleryError }); return; }
    sheetExporting = true;
    els.sheetControls.querySelectorAll('.sheet-export').forEach(syncExportButton);
    try {
      await run();
    } catch (err) {
      // Said where the press was, never an unhandled rejection; what was
      // saved before it stays saved.
      showError(`The sheet could not be exported: ${err.message}.`, false, { slot: els.galleryError });
    } finally {
      sheetExporting = false;
      els.sheetControls.querySelectorAll('.sheet-export').forEach(syncExportButton);
    }
  };
  syncExportButton(button);
  return button;
}

async function exportSheet() {
  const layout = exportLayout();
  if (!layout) return;
  const name = sheetFileName(layout.pins, false, layout);
  const canvas = await buildSheetCanvas(null, { layout });
  if (canvas) await saveCanvas(canvas, name);
}

// Export all: every slice the pins hold, where Export PNG has the one on
// screen. Stacked in one image, each slice under a title naming its pins, at
// one slice's cell width or less - down to the 96 px floor - so the stack
// stays inside the edge; past that, one image per slice, each named by its
// pins as Export PNG names it, and told apart where two names come out alike.
// A slice no image is in is left out.
async function exportAllSheet() {
  const layout = exportLayout();
  if (!layout || !layout.pins.length) return;
  const slot = { slot: els.galleryError };
  const axes = layout.pins.map(([axis]) => axis);
  const count = axes.reduce((n, axis) => n * layout.values[axis.key].length, 1);
  if (count > MAX_SHEET_CELLS) {
    showError(`The pins hold ${count.toLocaleString('en')} slices, too many to export. `
      + 'Lay a pinned axis out as Columns or Rows, or narrow the search.', false, slot);
    return;
  }
  let combos = [[]];
  for (const axis of axes) combos = combos.flatMap((held) => layout.values[axis.key].map((v) => [...held, [axis, v]]));
  const slices = combos.map((pins) => ({ pins, cellOf: sliceCells(layout, pins, layout.records).cellOf }))
    .filter((slice) => slice.cellOf.size);
  if (slices.length < 2) {
    const [only] = slices;
    const name = sheetFileName(only?.pins ?? layout.pins, false, layout);
    const canvas = await buildSheetCanvas(only ? [only] : null, { layout });
    if (canvas) await saveCanvas(canvas, name);
    return;
  }
  const edge = EXPORT_MAX_EDGE.toLocaleString('en');
  const cellW = stackedCellWidth(layout, slices.length);
  if (cellW !== null) {
    const name = sheetFileName(layout.pins, true, layout);
    const canvas = await buildSheetCanvas(slices, { layout, cellW });
    if (canvas) await saveCanvas(canvas, name);
    return;
  }
  if (slices.length > EXPORT_FILES_MAX) {
    showError(`Stacked, the ${slices.length} slices would pass ${edge} px, and one image each is more than `
      + `${EXPORT_FILES_MAX} downloads. Lay a pinned axis out as Columns or Rows, or export the slices one at a time.`,
    false, slot);
    return;
  }
  const taken = new Set();
  const named = slices.map((slice) => {
    const base = sheetFileName(slice.pins, false, layout);
    let name = base;
    for (let i = 2; taken.has(name); i += 1) name = base.replace(/\.png$/, `-${i}.png`);
    taken.add(name);
    return { slice, name };
  });
  let saved = 0;
  for (const { slice, name } of named) {
    const canvas = await buildSheetCanvas([slice], { layout });
    if (!canvas || !(await saveCanvas(canvas, name))) return;
    saved += 1;
  }
  showError(`Exported ${saved} slices as ${saved} images: stacked, they would pass ${edge} px. `
    + 'The browser may ask before keeping several downloads.', true, slot);
}
