// Krea Local's page script: the contact sheet and its export. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ------------------------------------------------------------ contact sheet

// Everything a grid can vary. `of` reads one off a record; mu's null is a real
// value - "no shift of ours" - not a missing one, so it prints as `none`. The
// prompt is the one axis whose values are prose, which is why it is labelled
// rather than printed - see promptLabels.
const SHEET_AXES = [
  { key: 'prompt', label: 'Prompt', of: (r) => r.prompt },
  // A `{name}` placeholder may live in the negative prompt alone, which is a
  // legal axis the grid renders; without it here such a run collapsed to one
  // cell, every other axis having a single value.
  { key: 'negative', label: 'Negative', of: (r) => r.negative_prompt || null },
  { key: 'steps', label: 'Steps', of: (r) => r.steps },
  { key: 'cfg', label: 'CFG', of: (r) => r.guidance_scale },
  // The scale the negative prompt ran at. Dropped by axesFor wherever it only
  // echoes the CFG axis - every Krea 2 record - or a CFG sweep with a negative
  // prompt would lay itself out as a diagonal against its own shadow.
  { key: 'tcfg', label: 'True CFG', of: (r) => r.true_cfg },
  { key: 'mu', label: 'Mu', of: (r) => r.mu },
  // The adapter's weight - the comparison a LoRA is bought for, and a sweep
  // over it is exactly a sheet. Null where no adapter ran, so a mixed search
  // does not invent an axis out of one.
  { key: 'lora', label: 'LoRA', of: (r) => r.lora_scale },
  { key: 'seed', label: 'Seed', of: (r) => r.seed },
];
const trueCfgEchoesCfg = (records) =>
  records.every((r) => r.true_cfg == null || r.true_cfg === r.guidance_scale);
// The axes in play for the current records. Records from a prompt grid carry
// the variable assignment that made them, and then each variable is an axis
// of its own - a `size` x `ground` night is a 2x2, not a 4-wide strip of
// finished sentences - and the flat prompt and negative axes are *suppressed*,
// not kept as fallbacks: a 4-value prompt axis would beat a 2-value variable
// axis in chooseAxes and turn a single-variable run into a diagonal.
let sheetAxesList = SHEET_AXES;
function axesFor(records) {
  const names = [];
  for (const rec of records) {
    for (const name of Object.keys(rec.variables || {})) if (!names.includes(name)) names.push(name);
  }
  const flat = trueCfgEchoesCfg(records) ? SHEET_AXES.filter((a) => a.key !== 'tcfg') : SHEET_AXES;
  if (!names.length) return flat;
  const variableAxes = names.map((name) => ({
    key: `var:${name}`,
    label: name,
    // `""` is the deliberately empty value - the word left out - and reads as
    // `none`; a record with no variables at all (a mixed search) reads as null.
    of: (r) => {
      const value = r.variables ? r.variables[name] : undefined;
      return value === undefined ? null : (value === '' ? 'none' : value);
    },
  }));
  return [...variableAxes, ...flat.filter((a) => a.key !== 'prompt' && a.key !== 'negative')];
}
const axisByKey = (key) => sheetAxesList.find((a) => a.key === key);
const axisValue = (axis, rec) => axis.of(rec) ?? null;

// Full prompt -> the words that make it different from its neighbours, filled
// in by sheetValues. Only the prompt axis is in here, so anything else that is
// a string prints as itself.
let promptShort = new Map();
const valueText = (v) => {
  if (v === null) return 'none';
  if (typeof v === 'string') return promptShort.get(v) ?? clip(v);
  return String(v);
};

const LABEL_CHARS = 44;
const clip = (text) => (text.length > LABEL_CHARS ? `${text.slice(0, LABEL_CHARS - 1)}…` : text);

// A header that repeats the forty words every prompt shares is unreadable, and
// the two or three that differ are the whole comparison - so the label keeps
// what varies and drops what does not.
//
// Same word count throughout means the varying positions can be picked out
// exactly, which is what a two-variable grid produces. A value that was set to
// `none` makes its prompt shorter than the others, and then the shared head and
// tail come off instead - `big dog on grass` against `dog on grass` labels
// itself `big` and `none`.
function promptLabels(prompts) {
  const words = prompts.map((p) => p.trim().split(/\s+/));
  const width = words[0].length;
  let kept;
  if (words.every((w) => w.length === width)) {
    const varies = [];
    for (let i = 0; i < width; i++) {
      if (!words.every((w) => w[i] === words[0][i])) varies.push(i);
    }
    kept = words.map((w) => varies.map((i) => w[i]));
  } else {
    let head = 0;
    while (words.every((w) => head < w.length && w[head] === words[0][head])) head += 1;
    let tail = 0;
    const same = (w) => w[w.length - 1 - tail] === words[0][words[0].length - 1 - tail];
    while (words.every((w) => tail < w.length - head && same(w))) tail += 1;
    kept = words.map((w) => w.slice(head, w.length - tail));
  }
  // Nothing left means this cell is the one that dropped the word the others
  // carry, which is a value in its own right and reads as one.
  return kept.map((w) => clip(w.join(' ')) || 'none');
}

// The distinct values each axis takes across the loaded images, ascending.
function sheetValues(records) {
  const values = {};
  for (const axis of sheetAxesList) {
    const seen = [];
    for (const rec of records) {
      const v = axisValue(axis, rec);
      if (!seen.includes(v)) seen.push(v);
    }
    seen.sort((a, b) => {
      if (a === null) return -1;
      if (b === null) return 1;
      return typeof a === 'string' ? a.localeCompare(b) : a - b;
    });
    values[axis.key] = seen;
  }
  // No prompt axis when variables carry the variation; nothing to label.
  const prompts = values.prompt || [];
  const labels = prompts.length > 1 ? promptLabels(prompts) : prompts.map(clip);
  promptShort = new Map(prompts.map((text, i) => [text, labels[i]]));
  return values;
}

// Columns and rows default to the two axes that actually vary, widest first.
// A third varying axis becomes a pin rather than a third dimension: a flat
// sheet that crosses everything at once is a wall nobody can read.
function chooseAxes(values, count) {
  let varying = sheetAxesList.filter((a) => values[a.key].length > 1);
  // An axis with one value per image is a list, not a dimension - the seed of
  // a mixed gallery, mostly. It is dropped whenever anything else varies; a
  // seeds-only run keeps it, since it is then the only dimension there is.
  const listlike = varying.filter((a) => values[a.key].length === count);
  if (listlike.length < varying.length) varying = varying.filter((a) => !listlike.includes(a));
  varying.sort((a, b) => values[b.key].length - values[a.key].length);
  return {
    cols: varying[0] ? varying[0].key : null,
    rows: varying[1] ? varying[1].key : null,
    pins: {},
  };
}

// Every varying axis is exactly one of columns, rows or a pin, and every pin
// holds a value that is actually present. Run on every render: promoting a
// pinned axis to Columns used to leave it in the pins too, so only one of its
// columns could ever fill; and switching the search to another run of the same
// shape kept a pin on a value the new records never had - a blank sheet under
// a dropdown showing a different value than the one filtering it.
function reconcileAxes(values) {
  const pins = {};
  for (const axis of sheetAxesList) {
    const key = axis.key;
    if (values[key].length < 2 || key === sheetAxes.cols || key === sheetAxes.rows) continue;
    const kept = sheetAxes.pins[key];
    pins[key] = values[key].includes(kept) ? kept : values[key][0];
  }
  sheetAxes.pins = pins;
}

// Picking for one side the axis already on the other swaps them rather than
// collapsing the sheet to one dimension twice; reconcileAxes then re-pins
// whatever the swap displaced.
function setAxis(side, key) {
  const other = side === 'cols' ? 'rows' : 'cols';
  if (key && key === sheetAxes[other]) sheetAxes[other] = sheetAxes[side];
  sheetAxes[side] = key || null;
  renderSheet();
}

function headCell(text, className) {
  const div = document.createElement('div');
  div.className = className || 'head';
  div.textContent = text;
  return div;
}

function labelled(text, control) {
  const wrap = document.createElement('label');
  wrap.className = 'sheet-pick';
  wrap.append(document.createTextNode(text), control);
  return wrap;
}

function picker(options, selected, onPick) {
  const select = document.createElement('select');
  options.forEach(([value, text]) => {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = text;
    option.selected = String(value) === String(selected);
    select.appendChild(option);
  });
  select.onchange = () => onPick(select.value);
  return select;
}

// Past this many cells the layout is a list, not a matrix - and each cell used
// to cost a scan of every record, so the Sheet button over an unfiltered
// gallery froze the tab for seconds and laid out a grid 100,000 px wide.
const MAX_SHEET_CELLS = 4096;
function renderSheet() {
  sheetRecords = galleryRecords;
  sheetAxesList = axesFor(sheetRecords);
  const values = sheetValues(sheetRecords);
  // Re-derive columns and rows when the image set changes shape, and only
  // then, so switching filters lands on something sensible while a hand-picked
  // axis survives a redraw. Pins are reconciled every time - see above. The
  // axis names are part of the shape: a variable run and a plain one with the
  // same cardinalities are different sheets.
  const signature = sheetAxesList.map((a) => `${a.key}:${values[a.key].length}`).join('/');
  if (sheetAxes.signature !== signature) {
    sheetAxes = { ...chooseAxes(values, sheetRecords.length), signature };
  }
  reconcileAxes(values);

  const axisOptions = [['', '—']].concat(
    sheetAxesList.filter((a) => values[a.key].length > 1).map((a) => [a.key, a.label]),
  );
  els.sheetControls.innerHTML = '';
  els.sheetControls.append(
    labelled('Columns', picker(axisOptions, sheetAxes.cols ?? '', (key) => setAxis('cols', key))),
    labelled('Rows', picker(axisOptions, sheetAxes.rows ?? '', (key) => setAxis('rows', key))),
  );
  if (sheetRecords.length) {
    const exportButton = document.createElement('button');
    exportButton.className = 'ghost small';
    exportButton.textContent = 'Export PNG';
    exportButton.title = 'Download this sheet as one image, labels and caption included';
    exportButton.onclick = exportSheet;
    els.sheetControls.append(exportButton);
  }
  for (const [key, pinned] of Object.entries(sheetAxes.pins)) {
    const axis = axisByKey(key);
    els.sheetControls.append(
      // By position, not by the text on the option: two prompts shorten to two
      // different labels, but nothing guarantees a label round-trips back to
      // the value it came from.
      labelled(axis.label, picker(
        values[key].map((v, index) => [index, valueText(v)]),
        values[key].indexOf(pinned),
        (picked) => {
          sheetAxes.pins[key] = values[key][Number(picked)] ?? null;
          renderSheet();
        },
      )),
    );
  }

  const colAxis = sheetAxes.cols ? axisByKey(sheetAxes.cols) : null;
  const rowAxis = sheetAxes.rows ? axisByKey(sheetAxes.rows) : null;
  const colValues = colAxis ? values[colAxis.key] : [null];
  const rowValues = rowAxis ? values[rowAxis.key] : [null];
  els.sheetGrid.innerHTML = '';
  if (colValues.length * rowValues.length > MAX_SHEET_CELLS) {
    els.sheetNote.textContent =
      `${colValues.length} × ${rowValues.length} cells is a list, not a sheet. ` +
      'Narrow the search to one run — the ⊞ button on a card does that.';
    return;
  }

  // One pass over the records into a map keyed by (row, column), rather than a
  // scan of every record per cell, so the sheet costs O(records + cells). The
  // first record wins a cell; they arrive newest first.
  const pins = Object.entries(sheetAxes.pins).map(([key, v]) => [axisByKey(key), v]);
  const rowIndex = new Map(rowValues.map((v, i) => [v, i]));
  const colIndex = new Map(colValues.map((v, i) => [v, i]));
  const cellOf = new Map();
  for (const rec of sheetRecords) {
    if (pins.some(([axis, v]) => axisValue(axis, rec) !== v)) continue;
    const r = rowAxis ? rowIndex.get(axisValue(rowAxis, rec)) : 0;
    const c = colAxis ? colIndex.get(axisValue(colAxis, rec)) : 0;
    const at = r * colValues.length + c;
    if (!cellOf.has(at)) cellOf.set(at, rec);
  }

  // What the export draws: the same matrix, the same pins, the same labels.
  sheetLayout = { colAxis, rowAxis, colValues, rowValues, cellOf, pins };

  const shown = [];
  els.sheetGrid.style.gridTemplateColumns =
    `auto repeat(${colValues.length}, minmax(96px, 1fr))`;
  // One aspect ratio for the whole sheet - size is not a sheet axis, and a
  // free-text search can mix sizes - so the commonest one wins and the odd
  // cell letterboxes inside it rather than being cropped.
  const ratios = new Map();
  for (const rec of sheetRecords) {
    const key = `${rec.width} / ${rec.height}`;
    ratios.set(key, (ratios.get(key) || 0) + 1);
  }
  const [aspect] = [...ratios.entries()].sort((a, b) => b[1] - a[1])[0] || ['1'];
  els.sheetGrid.style.setProperty('--cell-aspect', aspect);
  els.sheetGrid.appendChild(headCell(''));
  // A prompt label is prose rather than a number, so its header is allowed to
  // wrap; nowrap would push one column out to the width of the whole phrase.
  // A variable's values are prose too.
  const prose = (axis) => (
    axis && (axis.key === 'prompt' || axis.key === 'negative' || axis.key.startsWith('var:')) ? ' text' : ''
  );
  colValues.forEach((v) => {
    els.sheetGrid.appendChild(
      headCell(colAxis ? `${colAxis.label} ${valueText(v)}` : '', `head${prose(colAxis)}`),
    );
  });

  rowValues.forEach((rowValue, r) => {
    els.sheetGrid.appendChild(
      headCell(
        rowAxis ? `${rowAxis.label} ${valueText(rowValue)}` : '',
        `rowhead${prose(rowAxis)}`,
      ),
    );
    colValues.forEach((_, c) => {
      const rec = cellOf.get(r * colValues.length + c);
      const cell = document.createElement('div');
      cell.className = rec ? `sheet-cell${rec.favorite ? ' fav' : ''}` : 'sheet-cell empty';
      if (rec) {
        shown.push(rec);
        // `shown` is still filling, but it is the same array by the time
        // anything can be clicked - so the arrows walk the sheet in reading
        // order rather than the gallery's.
        cell.appendChild(thumb(rec, shown));
        cell.dataset.id = rec.id;
      }
      els.sheetGrid.appendChild(cell);
    });
  });

  const pinned = pins
    .map(([axis, v]) => `${axis.label.toLowerCase()} ${valueText(v)}`)
    .join(', ');
  els.sheetNote.textContent = sheetRecords.length === 0
    ? 'Nothing matches. The filters above the sheet are the gallery search.'
    : `${shown.length} of ${sheetRecords.length} loaded` + (pinned ? ` · pinned to ${pinned}` : '') +
      (total > sheetRecords.length ? ` · ${total} match the search` : '');
}

// ------------------------------------------------------------ sheet export

// The one artefact worth keeping from a night is the matrix with its labels
// burnt in, and a screenshot is capped at the viewport and drops the pinned
// caption. Drawn client-side from the layout the sheet is already showing -
// a server route would duplicate a layout contract the client holds, the
// non-round-trippable prompt labels especially. Thumbnails, not the PNGs: a
// contact sheet is for seeing which cell, and 384 px cells of a 32-wide grid
// are already a 12k-pixel image.
let sheetLayout = null;
const EXPORT_MAX_EDGE = 8000;

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);  // a missing file is an empty cell, not a failed export
    img.src = src;
  });
}

async function buildSheetCanvas() {
  const layout = sheetLayout;
  if (!layout) return null;
  const { colAxis, rowAxis, colValues, rowValues, cellOf, pins } = layout;
  const aspect = (() => {
    const [w, h] = (els.sheetGrid.style.getPropertyValue('--cell-aspect') || '1 / 1').split('/').map(Number);
    return h ? w / h : 1;
  })();
  const cols = colValues.length, rows = rowValues.length;
  // Cells shrink so the longer edge stays inside what a canvas can hold.
  const cellW = Math.max(96, Math.min(384, Math.floor(EXPORT_MAX_EDGE / Math.max(cols, rows))));
  const cellH = Math.round(cellW / aspect);
  const gap = 6, font = 14, headH = colAxis ? 34 : 8, headW = rowAxis ? 150 : 8, caption = 52;
  const width = headW + cols * (cellW + gap) + gap;
  const height = headH + rows * (cellH + gap) + gap + caption;
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b0b0e';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#e8e8ee';
  ctx.font = `${font}px "Segoe UI", system-ui, sans-serif`;
  ctx.textBaseline = 'middle';

  const clipText = (text, max) => {
    let t = text;
    while (t.length > 1 && ctx.measureText(t).width > max) t = `${t.slice(0, -2)}…`;
    return t;
  };
  ctx.textAlign = 'center';
  colValues.forEach((v, c) => {
    if (!colAxis) return;
    const x = headW + gap + c * (cellW + gap) + cellW / 2;
    ctx.fillText(clipText(`${colAxis.label} ${valueText(v)}`, cellW), x, headH / 2);
  });
  ctx.textAlign = 'right';
  rowValues.forEach((v, r) => {
    if (!rowAxis) return;
    const y = headH + gap + r * (cellH + gap) + cellH / 2;
    ctx.fillText(clipText(`${rowAxis.label} ${valueText(v)}`, headW - 12), headW - 6, y);
  });

  // Every cell's thumbnail in flight at once; a hole stays a hole.
  const jobs = [];
  rowValues.forEach((_, r) => colValues.forEach((__, c) => {
    const rec = cellOf.get(r * cols + c);
    const x = headW + gap + c * (cellW + gap), y = headH + gap + r * (cellH + gap);
    ctx.fillStyle = '#1a1a21';
    ctx.fillRect(x, y, cellW, cellH);
    if (!rec) return;
    jobs.push(loadImage(rec.thumb_url ?? rec.url).then((img) => {
      if (!img) return;
      // Contain, as the sheet does: the whole frame, letterboxed if it must be.
      const k = Math.min(cellW / img.naturalWidth, cellH / img.naturalHeight);
      const w = img.naturalWidth * k, h = img.naturalHeight * k;
      ctx.drawImage(img, x + (cellW - w) / 2, y + (cellH - h) / 2, w, h);
      if (rec.favorite) {
        ctx.fillStyle = '#fbbf24';
        ctx.textAlign = 'right';
        ctx.fillText('★', x + cellW - 6, y + 12);
        ctx.fillStyle = '#e8e8ee';
      }
    }));
  }));
  await Promise.all(jobs);

  // The caption is what a screenshot loses: which run, what was pinned, when.
  const first = sheetRecords[0];
  const pinned = pins.map(([axis, v]) => `${axis.label.toLowerCase()} ${valueText(v)}`).join(', ');
  const lines = [
    `${els.search.value.trim() || 'gallery'}${pinned ? ` · pinned to ${pinned}` : ''}`,
    first ? `${first.model_id.split('/').pop()} · ${first.width}×${first.height}` +
      (first.run_id ? ` · run ${first.run_id}` : '') + ` · ${when(Date.now())}` : '',
  ];
  ctx.fillStyle = '#8b8b9a';
  ctx.textAlign = 'left';
  lines.forEach((line, i) => ctx.fillText(clipText(line, width - 2 * gap), gap, height - caption + 16 + i * 20));
  return canvas;
}

async function exportSheet() {
  const canvas = await buildSheetCanvas();
  if (!canvas) return;
  canvas.toBlob((blob) => {
    if (!blob) { showError('The browser could not encode the sheet.'); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const first = sheetRecords[0];
    a.download = `sheet-${first && first.run_id ? first.run_id : 'gallery'}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }, 'image/png');
}

function setSheet(on) {
  sheetMode = on;
  els.sheetBtn.classList.toggle('active', on);
  els.sheet.hidden = !on;
  els.gallery.hidden = on;
  // Off is a real escape hatch: whatever layout and pins were picked go with it.
  if (!on) sheetAxes = { rows: null, cols: null, pins: {} };
  return resetGallery();
}
