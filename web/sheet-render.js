// AuraDream's page script: contact-sheet rendering and navigation.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The cells of one slice of the matrix: the records that pass `pins`, each in
// its (row, column) cell. One pass over the records into a map keyed by the
// cell, rather than a scan of every record per cell, so the sheet costs
// O(records + cells). The first record wins a cell; they arrive newest
// first. The ones it turns away are counted for the note, where "1 of 3
// loaded" read as two images still on their way: a repeat of the cell's
// image - the same grid run again without Resume, which keeps its run id, or
// a Re-render - or one that differs in something no axis lays out, the size,
// the checkpoint. Export all asks it of every slice the pins hold.
function sliceCells({ colAxis, rowAxis, colValues, rowValues }, pins, records = sheetRecords) {
  const rowIndex = new Map(rowValues.map((v, i) => [v, i]));
  const colIndex = new Map(colValues.map((v, i) => [v, i]));
  const cellOf = new Map();
  let repeats = 0, crowded = 0;
  for (const rec of records) {
    if (pins.some(([axis, v]) => axisValue(axis, rec) !== v)) continue;
    const r = rowAxis ? rowIndex.get(axisValue(rowAxis, rec)) : 0;
    const c = colAxis ? colIndex.get(axisValue(colAxis, rec)) : 0;
    const at = r * colValues.length + c;
    if (!cellOf.has(at)) cellOf.set(at, rec);
    else if (sameSettings(rec, cellOf.get(at))) repeats += 1;
    else crowded += 1;
  }
  return { cellOf, repeats, crowded };
}

// A leading source belongs to this slice only: pins can change its crop, or
// leave several crops that no one source thumbnail could represent. Exports
// use the same decision with their captured source cache for every slice.
function sheetSourceView(records, sources = sheetSources) {
  const sourceId = sharedSource(records);
  const candidate = sourceId === null ? null : sources.get(sourceId) ?? null;
  const crops = candidate ? records.map((rec) => refineKept(
    candidate.width, candidate.height, rec.width, rec.height, rec.crop_anchor,
  )) : [];
  const commonCrop = crops.length > 0 && crops.every((crop) => JSON.stringify(crop) === JSON.stringify(crops[0]));
  const source = commonCrop ? candidate : null;
  return {
    sourceId, candidate, commonCrop, source, sourceCrop: source ? crops[0] : null,
    sourceAspect: source ? records[0].width / records[0].height : null,
  };
}

// Prompt iterations made on random seeds share no seed: a version stands on a
// seed of its own, so the sheet can only lay them out as a diagonal or pin the
// seed to one of them ("1 of 6 loaded"), which read as a broken sheet. The
// note says why, and where versions are set side by side on the same seeds.
const OWN_SEEDS = "each image has its own seed: Check's Compare on the same seeds puts versions side by side";
const ownSeeds = (values, records) => records.length > 1
  && (values.prompt?.length ?? 0) > 1 && values.seed.length === records.length;

// A star on every cell, the card's own (`.fav-badge`): always there on a
// starred cell, on another where the pointer or the focus is. Tab reaches
// the current cell's star after its image; the others stay out of Tab order,
// so its stops do not grow by a thousand. The value comes from the cell,
// whose class setFavorite keeps, and goes back to the record after: a refine
// sweep's source is no gallery record for setFavorite to find. `options` is
// setFavorite's: F says what it did (sheetKey).
async function starCell(cell, rec, options = {}) {
  await setFavorite(rec, !cell.classList.contains('fav'), options);
  const now = els.sheetGrid.querySelector(`.sheet-cell[data-id="${rec.id}"]`);
  if (now) rec.favorite = now.classList.contains('fav');
}

// The sheet laid out again from the records - a gallery page landing, a
// delete, a pin or an axis picked, a resize that wraps it - rebuilds every
// cell and control, and a keyboard's focus on one fell to <body>: F on a cell
// another tab had deleted, the end of another tab's job, an ArrowDown on the
// Columns select. `place` is where focus was (sheetFocusPlace), taken before
// anything was removed - forgetImages takes it before it drops the cells -
// and it goes back there: the same cell's image (or star) by its id, else
// the image now at its place in reading order, else the last; the control
// of the same name; the search box where the sheet has no cells left.
function renderSheet(place = sheetFocusPlace()) {
  const cells = [...els.sheetGrid.querySelectorAll('.sheet-cell[data-id]')];
  const previous = cells.findIndex((cell) => +cell.dataset.id === sheetCurrent);
  const index = previous < 0 ? sheetCurrentIndex : previous;
  drawSheet();
  keepCurrentSheet(index);
  restoreSheetFocus(place);
}

// What drawSheet drew (with the `shown` and `cells` it fills in): the
// export, Fill and the zoom read it.
let sheetLayout = null;

// One image is the sheet's way in from Tab. Its tools remain reachable from
// it; other cells are reached with arrows or the pointer. Keep the id across
// redraws, and the old reading-order place when a pin or delete removes it.
let sheetCurrent = null;
let sheetCurrentIndex = 0;
let sheetCurrentRow = 0;
const sheetCells = () => [...els.sheetGrid.querySelectorAll('.sheet-cell[data-id]')];
function sheetStops(cell, on) {
  cell.querySelectorAll('img, button').forEach((el) => { el.tabIndex = on ? 0 : -1; });
}

function makeSheetCurrent(cell, row = null) {
  if (!cell) return;
  const previous = els.sheetGrid.querySelector(`.sheet-cell[data-id="${sheetCurrent}"]`);
  if (previous && previous !== cell) sheetStops(previous, false);
  if (row !== null) sheetCurrentRow = row;
  else if (previous !== cell || !cell.classList.contains('source')) sheetCurrentRow = +cell.dataset.sheetRow;
  sheetCurrent = +cell.dataset.id;
  sheetCurrentIndex = sheetCells().indexOf(cell);
  sheetStops(cell, true);
}

function keepCurrentSheet(index = 0) {
  const cells = sheetCells();
  const current = cells.find((cell) => +cell.dataset.id === sheetCurrent);
  const cell = current ?? cells[index] ?? cells.at(-1);
  if (cell) {
    sheetCurrentRow = Math.min(sheetCurrentRow, (sheetLayout?.rowValues.length ?? 1) - 1);
    makeSheetCurrent(cell);
  } else {
    sheetCurrent = null;
    sheetCurrentIndex = 0;
    sheetCurrentRow = 0;
  }
}

function sheetFocus(e) {
  const cell = e.target.closest?.('.sheet-cell[data-id]');
  if (!cell) return;
  makeSheetCurrent(cell);
  if (!cell.classList.contains('source')) sheetSwipeFrom = +cell.dataset.id;
}

function sheetFocusPlace() {
  const at = document.activeElement;
  if (!at || at === document.body || !els.sheet.contains(at)) return null;
  const cell = at.closest('.sheet-cell[data-id]');
  if (cell && els.sheetGrid.contains(cell)) {
    const cells = [...els.sheetGrid.querySelectorAll('.sheet-cell[data-id]')];
    return { id: cell.dataset.id, index: cells.indexOf(cell), star: at.matches('.star') };
  }
  if (els.sheetControls.contains(at)) {
    const controls = [...els.sheetControls.querySelectorAll('select, button')];
    return { control: sheetControlName(at), index: controls.indexOf(at) };
  }
  return null;
}

// A control's name across a redraw: a select by its label's words (Columns,
// Rows, a pin's axis), a button by its own.
function sheetControlName(el) {
  const label = el.closest('label');
  return label ? `select:${label.firstChild?.textContent ?? ''}` : `button:${el.textContent}`;
}

function restoreSheetFocus(place) {
  if (!place) return;
  const now = document.activeElement;
  // Kept, or taken somewhere on purpose meanwhile.
  if (now && now !== document.body && now.isConnected) return;
  let to = null;
  if (place.id !== undefined) {
    const cell = els.sheetGrid.querySelector(`.sheet-cell[data-id="${place.id}"]`);
    if (cell) to = (place.star && cell.querySelector('.star')) || cell.querySelector('img');
    else {
      const images = [...els.sheetGrid.querySelectorAll('.sheet-cell img[data-id]')];
      to = images[Math.max(0, place.index)] ?? images.at(-1) ?? null;
    }
  } else {
    const controls = [...els.sheetControls.querySelectorAll('select, button')];
    to = controls.find((c) => sheetControlName(c) === place.control) ?? controls[place.index] ?? null;
  }
  (to ?? els.search).focus({ preventScroll: true });
}

// Swipe CFG and Swipe steps: the lightbox on a cell, compared with its
// neighbour along that axis by swipe, walking only that cell's ladder - the
// cells of its row, or column, that differ from it in that alone (ladderOf,
// lightbox-cfg.js) - so ← → step along it and nothing else. Each is drawn
// only where its axis is laid out (a column or row axis), as the axis
// pickers list only what varies; held, saying why, where no two cells differ
// in it alone. The cell is the one the keyboard was last on in the sheet
// (`sheetSwipeFrom`, app.js), or else the first in reading order with a lower
// value beside it.
let sheetSwipeFrom = null;
const CFG_SWIPE_NONE = 'No two cells here differ in CFG alone';
const SHEET_SWIPES = {
  cfg: {
    className: 'sheet-cfg-swipe', label: 'Swipe CFG', none: CFG_SWIPE_NONE,
    title: 'Open these cells as a swipe between neighbouring CFG values; ← → step along CFG, S turns it off',
  },
  steps: {
    className: 'sheet-steps-swipe', label: 'Swipe steps', none: 'No two cells here differ in steps alone',
    title: 'Open these cells as a swipe between neighbouring step counts; ← → step along steps, S turns it off',
  },
};
function sheetSwipeButton(kind) {
  const swipe = SHEET_SWIPES[kind];
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `ghost small ${swipe.className}`;
  button.textContent = swipe.label;
  button.setAttribute('aria-disabled', 'true');
  button.title = swipe.none;
  button.onclick = () => openSheetSwipe(kind, button);
  return button;
}

// One key for "agrees on everything but the axes in `skip`", for the hold's
// one pass over the cells: what sameExcept compares, as JSON.
const ladderGroupKey = (rec, skip = CFG_SKIP) => JSON.stringify([
  ...SHEET_AXES.filter((axis) => !skip.includes(axis.key)).map((axis) => axisValue(axis, rec)),
  rec.width, rec.height, rec.model_id, rec.quantization, rec.refine,
]);

function syncSheetSwipes() {
  for (const [kind, swipe] of Object.entries(SHEET_SWIPES)) {
    const button = els.sheetControls.querySelector(`.${swipe.className}`);
    if (!button) continue;
    const axis = LADDER_AXES[kind];
    const groups = new Map();
    for (const rec of sheetLayout?.cells ?? []) {
      const key = ladderGroupKey(rec, axis.skip);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(rec);
    }
    // CFG's key leaves True CFG out, as sameExcept does; the press's ladder
    // (ladderOf) keeps only the cell's True CFG where that is not an echo of
    // CFG, so a CFG x True CFG grid with cells missing can group two CFG
    // values no ladder holds. The hold asks the press's own question, within
    // each group rather than of the whole sheet a cell at a time.
    const some = [...groups.values()].some((group) => new Set(group.map(axis.of)).size > 1
      && group.some((r) => ladderOf(axis, r, group).length > 1));
    button.setAttribute('aria-disabled', String(!some));
    button.title = some ? swipe.title : swipe.none;
  }
}

function sheetSwipeStart(axis, cells) {
  const ladder = (rec) => ladderOf(axis, rec, cells);
  const from = cells.find((r) => r.id === sheetSwipeFrom);
  if (from && ladder(from).length > 1) return from;
  return cells.find((r) => {
    const own = ladder(r);
    return own.length > 1 && own[0].id !== r.id;
  }) ?? null;
}

function openSheetSwipe(kind, button) {
  const cells = sheetLayout?.cells ?? [];
  const axis = LADDER_AXES[kind];
  const rec = button.getAttribute('aria-disabled') === 'true' ? null : sheetSwipeStart(axis, cells);
  // Held, the title is the reason; otherwise the cells changed since the
  // hold was worked out, and the reason is the hold's own.
  if (!rec) {
    showError(button.getAttribute('aria-disabled') === 'true' ? button.title : SHEET_SWIPES[kind].none, true);
    return;
  }
  openLightbox(rec, ladderOf(axis, rec, cells), 'sheet');
  void toggleCompare('swipe', kind);
}

function drawSheet() {
  // Where the box and the page were scrolled, for fitSheet to put back:
  // emptying the grid, below, lets a layout clamp both - a zoomed sheet
  // panned to its far corner went back to the first cell at every redraw.
  const scrolled = sheetScrolled();
  sheetRecords = galleryRecords;
  sheetAxesList = axesFor(sheetRecords);
  const current = (held) => (held && held.epoch === galleryEpoch ? held : null);
  const whole = current(sheetWritten)?.records;
  const values = sheetValues(sheetRecords, whole);
  const adapters = new Set(sheetRecords.filter((r) => r.lora_scale != null).map((r) => r.lora));
  const [adapter] = adapters;
  sheetAdapter = adapters.size === 1 && adapter ? adapterName(adapter) : null;
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
    els.sheetControls.append(fillButton(), zoomControls(), exportButtonFor(
      'Export PNG', 'Download this sheet as one image, labels and caption included', exportSheet,
    ));
    // Every slice the pins hold, where something is pinned.
    if (Object.keys(sheetAxes.pins).length) {
      els.sheetControls.append(exportButtonFor(
        'Export all', 'Download every pinned slice: stacked in one image, titled by its pins, '
          + `or one image per slice where stacked they would pass ${EXPORT_MAX_EDGE.toLocaleString('en')} px`,
        exportAllSheet,
      ));
    }
    // A swipe between neighbouring CFG values, or step counts, where that
    // axis is laid out.
    for (const kind of Object.keys(SHEET_SWIPES)) {
      if (sheetAxes.cols === kind || sheetAxes.rows === kind) els.sheetControls.append(sheetSwipeButton(kind));
    }
  }
  for (const [key, pinned] of Object.entries(sheetAxes.pins)) {
    const axis = axisByKey(key);
    els.sheetControls.append(
      // By position, not by the text on the option: two prompts shorten to two
      // different labels, but nothing guarantees a label round-trips back to
      // the value it came from.
      labelled(axis.label, picker(
        values[key].map((v, index) => [index, clip(sheetText(axis, v))]),
        values[key].indexOf(pinned),
        (picked) => {
          sheetAxes.pins[key] = values[key][Number(picked)] ?? null;
          renderSheet();
        },
      )),
    );
  }
  // Finish uses the stored run, without touching an unrelated composer.
  // Both run actions stay visible with their reasons, even over no images.
  const barRun = sheetShownRun();
  if (barRun) els.sheetControls.append(
    finishRunButton(barRun), moreSeedsRunButton(barRun, { model: sheetRecords[0]?.model_id }),
  );
  if (sheetRecords.length) {
    const hint = document.createElement('span');
    hint.className = 'sheet-hint';
    hint.textContent = 'F stars the image in focus · Ctrl+wheel zooms, drag pans';
    els.sheetControls.append(hint);
  }
  // What the run is, after every button: its count changes while a job
  // writes to the run, and text before them would move them under the
  // pointer.
  if (barRun) {
    const status = document.createElement('span');
    status.className = 'sheet-run-status';
    status.dataset.run = barRun;
    status.dataset.base = sheetRunStatus(barRun);
    status.textContent = sheetRunStatus(barRun);
    status.title = status.textContent;
    els.sheetControls.append(status);
    syncFinishRunButtons();
  }
  els.sheetFixed.textContent = fixedSettings(values);
  els.sheetFixed.hidden = !els.sheetFixed.textContent;

  const colAxis = sheetAxes.cols ? axisByKey(sheetAxes.cols) : null;
  const rowAxis = sheetAxes.rows ? axisByKey(sheetAxes.rows) : null;
  const colValues = colAxis ? values[colAxis.key] : [null];
  const rowValues = rowAxis ? values[rowAxis.key] : [null];
  els.sheetGrid.innerHTML = '';
  // A table, or a list where it wraps, once something is drawn (below).
  els.sheetGrid.removeAttribute('role');
  els.sheetGrid.removeAttribute('aria-label');
  // Nothing to lay out, and the note says so. Drawn anyway - a corner, two
  // heads and one empty cell - the cell grew to the window's height (the
  // fit's --cell-max) and the sheet was a screen-tall grey box.
  if (!sheetRecords.length) {
    sheetLayout = null;
    els.sheet.classList.remove('wrapped');
    els.sheetNote.textContent = 'Nothing matches. The filters above the sheet are the gallery search.';
    return;
  }
  if (colValues.length * rowValues.length > MAX_SHEET_CELLS) {
    sheetLayout = null;
    els.sheetNote.textContent =
      `${colValues.length} × ${rowValues.length} cells is a list, not a sheet. ` +
      'Narrow the search to one run — the ⊞ button on a card does that.';
    return;
  }

  const pins = Object.entries(sheetAxes.pins).map(([key, v]) => [axisByKey(key), v]);
  const { cellOf, repeats, crowded } = sliceCells({ colAxis, rowAxis, colValues, rowValues }, pins);

  const displayed = [...cellOf.values()].filter(Boolean);
  const { sourceId, candidate, commonCrop, source, sourceCrop, sourceAspect } = sheetSourceView(displayed);
  const columns = colValues.length + (source ? 1 : 0);
  const wrap = sheetWraps(colAxis, rowAxis, columns);

  // One aspect ratio for the whole sheet - a size axis or a
  // free-text search can mix sizes - so the commonest one wins and the odd
  // cell letterboxes inside it rather than being cropped.
  const ratios = new Map();
  for (const rec of sheetRecords) {
    const key = `${rec.width} / ${rec.height}`;
    ratios.set(key, (ratios.get(key) || 0) + 1);
  }
  const [aspect] = [...ratios.entries()].sort((a, b) => b[1] - a[1])[0] || ['1'];
  const [aspectW, aspectH] = aspect.split('/').map(Number);

  // What the export draws: the same matrix, the same pins, the same labels,
  // over the records it was laid out from.
  sheetLayout = {
    colAxis, rowAxis, colValues, rowValues, cellOf, pins, values, wrap, source,
    records: sheetRecords, aspect: aspectH ? aspectW / aspectH : 1, sourceCrop, sourceAspect,
  };

  els.sheetGrid.style.setProperty('--cell-aspect', aspect);
  // What the stylesheet sizes a matrix's cells from (--cell-max): the
  // aspect as a number and the rows that share the window's height.
  els.sheetGrid.style.setProperty('--sheet-ar', String(sheetLayout.aspect));
  els.sheetGrid.style.setProperty('--sheet-rows', String(rowValues.length));
  els.sheet.classList.toggle('wrapped', wrap);
  els.sheet.classList.toggle('fill-width', sheetFill());
  const fill = els.sheetControls.querySelector('.sheet-fill');
  if (fill) syncFillButton(fill);
  // A matrix's columns are at most --cell-w wide: --cell-max, fitted, or 1fr
  // where the sheet fills (styles/sheet.css); zoomed, exactly the zoomed width
  // (applySheetZoom sets both on the grid).
  els.sheetGrid.style.gridTemplateColumns = wrap
    ? 'repeat(auto-fill, minmax(var(--thumb-min), 1fr))'
    : `auto repeat(${columns}, minmax(var(--cell-min, ${SHEET_CELL_MIN}px), var(--cell-w)))`;

  const shown = [];
  // What the lightbox walks from a cell: the sheet in reading order, after the
  // source where there is one, so the arrows flip the original against each
  // pass. Still filling, but it is the same array by the time anything can be
  // clicked - so the arrows walk the sheet rather than the gallery.
  const walk = source ? [source] : [];
  // A pass looks like its source by design; a cell says it is one, unless
  // the strength is what the sheet lays out already (an axis or a pin).
  const strengthShown = values.strength.length > 1;
  // Each cell is named by what the sheet lays out - "Open #16 · Steps 2 ·
  // Seed 7332…", a comparison's version by its name - where the name was
  // "Open #16: <the prompt>", the one thing every cell of a run shares, and
  // the setting each cell is there to show was nowhere in it. Where nothing
  // is laid out (one image), the name is thumb's. A pin's value is every
  // cell's, and the sheet's controls say it.
  const cellFor = (rec, caption, badge, heads = []) => {
    const cell = document.createElement('div');
    cell.className = 'sheet-cell';
    cell.setAttribute('role', wrap ? 'listitem' : 'cell');
    if (!rec) {
      cell.classList.add('empty');
      return cell;
    }
    cell.classList.toggle('fav', !!rec.favorite);
    cell.dataset.id = rec.id;
    const img = thumb(rec, walk);
    img.tabIndex = -1;
    if (heads.length) {
      img.setAttribute('aria-label', [`Open #${rec.id}`, ...heads].join(' · '));
      titleImage(img, rec);
    }
    // The full PNG beside the thumbnail, for a cell drawn wider than the
    // thumbnail is (fitSheet sets `sizes`): 384 px shown at 520 was soft
    // exactly where a CFG or steps sweep differs. Lazy loading still fetches
    // only the cells on screen.
    const thumbWidth = thumbWidthOf(rec);
    if (rec.thumb_url && rec.width > thumbWidth) {
      img.srcset = `${rec.thumb_url} ${thumbWidth}w, ${rec.url} ${rec.width}w`;
    }
    // What a zoomed sheet shows instead (sizeSheetImages).
    img.dataset.full = rec.url;
    img.setAttribute('aria-keyshortcuts',
      'ArrowLeft ArrowRight ArrowUp ArrowDown Home End Control+Home Control+End F M Delete Plus - 0 1');
    const star = document.createElement('button');
    star.type = 'button';
    star.className = 'star fav-badge';
    star.tabIndex = -1;
    star.setAttribute('aria-keyshortcuts', 'F');
    star.onclick = () => { void starCell(cell, rec); };
    syncStar(star, !!rec.favorite, rec.id);
    cell.append(img, star);
    markReference(cell, referenceAt(rec.id));
    if (badge && rec.refine != null) {
      cell.classList.add('refined');
      cell.dataset.strength = rec.strength;
    }
    if (caption) {
      const cap = document.createElement('span');
      cap.className = 'cap';
      cap.textContent = caption;
      cell.appendChild(cap);
    }
    return cell;
  };
  // A matrix is a table to a screen reader: each of its rows a role=row of
  // its own that lays nothing out (`.sheet-row`, display: contents), so the
  // heads and cells are still the grid's items, the heads are column and row
  // headers, and a cell is read with the setting above it and the one
  // beside it. A wrapped row is a list of captioned cells.
  els.sheetGrid.setAttribute('role', wrap ? 'list' : 'table');
  els.sheetGrid.setAttribute('aria-label', sheetSummary(wrap, colAxis, rowAxis, null, source));
  let into = els.sheetGrid;
  const startRow = () => {
    into = document.createElement('div');
    into.className = 'sheet-row';
    into.setAttribute('role', 'row');
    els.sheetGrid.appendChild(into);
  };
  const add = (el, role = null) => {
    if (role) el.setAttribute('role', role);
    return into.appendChild(el);
  };
  // A prompt label is prose rather than a number, so its row head is capped
  // in width and wraps; a variable's values are prose too.
  const prose = (axis) => (
    axis && (axis.key === 'prompt' || axis.key === 'negative' || axis.key.startsWith('var:')) ? ' text' : ''
  );
  const headTitle = (axis, v) => (axis.key === 'prompt' || axis.key === 'negative' ? v ?? 'none' : headText(axis, v));
  const versions = rowAxis?.key === VARIANT_AXIS ? versionPrompts([...sheetRecords, ...(whole ?? [])]) : null;
  const original = versions?.get('original');
  const rowHead = (v) => {
    if (!rowAxis) return headCell('', 'rowhead');
    const prompt = versions?.get(v);
    const head = headCell(headText(rowAxis, v), `rowhead${prose(rowAxis)}`, prompt ?? headTitle(rowAxis, v),
      typeof v === 'number');
    const edit = prompt && original && v !== 'original' ? versionEdit(original, prompt) : '';
    if (edit) {
      const line = document.createElement('span');
      line.className = 'rowhead-edit';
      line.textContent = edit.length > EDIT_CHARS ? `${edit.slice(0, EDIT_CHARS - 1)}…` : edit;
      head.classList.add('has-edit');
      head.appendChild(line);
    }
    return head;
  };

  if (!wrap) {
    startRow();
    add(headCell('', 'head corner'), 'cell');
    if (source) {
      add(headCell(`Source #${source.id}`, 'head',
        `#${source.id}, ${source.width}×${source.height}: the image every pass here was refined from`), 'columnheader');
    }
    colValues.forEach((v) => {
      add(colAxis
        ? headCell(headText(colAxis, v), `head${prose(colAxis)}`, headTitle(colAxis, v), typeof v === 'number')
        : headCell('', 'head'), 'columnheader');
    });
  }
  rowValues.forEach((rowValue, r) => {
    if (!wrap) {
      startRow();
      add(rowHead(rowValue), 'rowheader');
    }
    // The source spans every row: one image the passes are all set against.
    // Cropped as the passes were (styles/sheet.css): a 1:1 source letterboxed
    // beside 3:2 passes cut from it set two framings side by side, which
    // the eye compared as one. Said where the shapes differ.
    if (source && r === 0) {
      const passAspect = sourceAspect;
      const cropped = Math.abs(source.width / source.height - passAspect) / passAspect > 0.01;
      const caption = [wrap ? `Source #${source.id}` : '', cropped ? 'cropped as the passes were' : '']
        .filter(Boolean).join(' · ');
      const cell = add(cellFor(source, caption, true, ['Source']));
      cell.classList.add('source');
      const image = cell.querySelector('img');
      // The slice may have another shape than the common sheet cell. Keep
      // its crop in the same contained box as its passes, with letterboxing.
      const frame = document.createElement('div');
      frame.className = 'sheet-source-frame';
      const width = Math.min(1, sourceAspect / sheetLayout.aspect);
      const height = Math.min(1, sheetLayout.aspect / sourceAspect);
      Object.assign(frame.style, {
        position: 'absolute', overflow: 'hidden',
        width: `${100 * width}%`, height: `${100 * height}%`,
        left: `${50 * (1 - width)}%`, top: `${50 * (1 - height)}%`,
      });
      image.before(frame);
      frame.appendChild(image);
      Object.assign(image.style, {
        position: 'absolute', objectFit: 'fill', maxWidth: 'none',
        width: `${100 / sourceCrop.w}%`, height: `${100 / sourceCrop.h}%`,
        left: `${-100 * sourceCrop.x / sourceCrop.w}%`, top: `${-100 * sourceCrop.y / sourceCrop.h}%`,
      });
      cell.dataset.sheetRow = '0';
      cell.dataset.sheetCol = '0';
      if (!wrap) {
        cell.style.gridRow = `span ${rowValues.length}`;
        cell.setAttribute('aria-rowspan', String(rowValues.length));
      }
    }
    colValues.forEach((v, c) => {
      const rec = cellOf.get(r * colValues.length + c);
      if (rec) {
        shown.push(rec);
        walk.push(rec);
      }
      const heads = [colAxis && headText(colAxis, v), rowAxis && headText(rowAxis, rowValue)].filter(Boolean);
      const cell = add(cellFor(rec, wrap ? headText(colAxis, v) : '', !strengthShown, heads));
      cell.dataset.sheetRow = String(r);
      cell.dataset.sheetCol = String(c + (source ? 1 : 0));
    });
  });
  // What saySheet counts, and the cells Swipe CFG and Swipe steps open from.
  sheetLayout.shown = shown.length;
  sheetLayout.cells = shown;
  syncSheetSwipes();

  const pinned = pins
    .map(([axis, v]) => `${axis.label.toLowerCase()} ${clip(sheetText(axis, v))}`)
    .join(', ');
  const notice = current(sheetNotice);
  const turned = [
    repeats && `${repeats} ${repeats === 1 ? 'repeats' : 'repeat'} a shown image's settings `
      + '(the same grid run again, a Re-render)',
    crowded && `${crowded} `
      + `${crowded === 1 ? 'shares a cell with another image and differs' : 'share a cell with another image and differ'} `
      + 'in something the sheet does not lay out (size, checkpoint, quantization…)',
  ].filter(Boolean);
  const count = turned.length
    ? `${shown.length} of ${sheetRecords.length} shown · ${turned.join(' · ')}`
    : `${shown.length} of ${sheetRecords.length} loaded`;
  els.sheetNote.textContent = count + (pinned ? ` · pinned to ${pinned}` : '') +
    (total > sheetRecords.length ? ` · ${total} match the search` : '') +
    (sourceId !== null && sheetSources.get(sourceId) === null ? ` · source #${sourceId} (deleted)` : '') +
    (ownSeeds(values, sheetRecords) ? ` · ${OWN_SEEDS}` : '') +
    (candidate && !commonCrop ? ' \u00b7 Source crops vary; open an image to compare its source.' : '') +
    (notice ? ` · ${notice.text}` : '');
  fitSheet(scrolled);
}

// The tallest a matrix's cells may be: the window, less what sits above the
// grid once showSheetScope has landed on the gallery head - the topbar, the
// head, the controls, the Fixed line - and the note below it, and less the
// column heads, shared between the rows (the stylesheet's --cell-max). The
// box the grid scrolls in is no taller, so a matrix that still does not fit -
// a filled one, mostly (Fill width) - keeps its heads in view. Measured
// rather than guessed, so a tier that
// scales the controls needs nothing here, and `100vh` follows the window from
// there. At QHD a 4 x 3 used to be 524 px cells in a sheet 1,607 px tall.
// The heads' row and the row heads' column (the corner is as wide) are also
// the box's scroll padding: a cell focused from the keyboard is scrolled
// clear of the heads stuck over it, where it lined up with the box's edge and
// the heads hid its top and its focus ring. All of it is measured with the
// zoom taken off, the cells' width then being the zoom's base
// (sheetZoomBase); the zoom goes back on at the cells' drawn width - a new
// base, Fill width's or a resized window's, changes the factor rather than
// the picture's scale - clamped to what the new base allows. The box's
// scroll, and the page's where the zoom's absence shortened the page under
// it, go back where they were (`scrolled`, drawSheet's, from before it
// emptied the grid). Then each image's `sizes` (sizeSheetImages).
function fitSheet(scrolled = sheetScrolled()) {
  const scroller = els.sheetGrid.parentElement;
  const zoom = sheetZoom;
  const drawn = sheetZoomBase * sheetZoom;
  sheetZoom = 1;
  applySheetZoom();
  if (!els.sheet.classList.contains('wrapped')) {
    const head = document.querySelector('.gallery-head');
    const box = scroller.getBoundingClientRect();
    // The head's scroll margin is where it lands: the topbar and a gap.
    const above = parseFloat(getComputedStyle(head).scrollMarginTop) + box.top - head.getBoundingClientRect().top;
    const below = els.sheet.getBoundingClientRect().bottom - box.bottom
      + parseFloat(getComputedStyle(document.querySelector('main')).paddingBottom);
    els.sheet.style.setProperty('--sheet-chrome', `${Math.ceil(above + below)}px`);
  }
  const heads = [...els.sheetGrid.querySelectorAll('.head')];
  const row = Math.max(0, ...heads.map((h) => h.getBoundingClientRect().height));
  els.sheet.style.setProperty('--sheet-head', `${Math.ceil(row)}px`);
  els.sheet.style.setProperty('--sheet-rowhead', `${Math.ceil(heads[0]?.getBoundingClientRect().width ?? 0)}px`);
  sheetZoomBase = sheetLayout && !sheetLayout.wrap ? sheetCellWidth() : 0;
  const factor = zoom > 1 && drawn > 0 && sheetZoomBase > 0 ? drawn / sheetZoomBase : 1;
  sheetZoom = sheetZoomable() ? Math.min(Math.max(1, factor), sheetZoomMax()) : 1;
  applySheetZoom();
  scroller.scrollLeft = scrolled.left;
  scroller.scrollTop = scrolled.top;
  if (scrollY < scrolled.page) scrollTo({ top: scrolled.page, behavior: 'instant' });
  sizeSheetImages();
  syncZoomControls();
}

function sheetScrolled() {
  const box = els.sheetGrid.parentElement;
  return { left: box.scrollLeft, top: box.scrollTop, page: scrollY };
}

// The width a cell is drawn at, fitted or filled - the cells of a sheet are
// one width - or 0 with nothing drawn.
function sheetCellWidth() {
  return els.sheetGrid.querySelector('.sheet-cell')?.getBoundingClientRect().width ?? 0;
}

// The fit follows the window (app.js): a resize measures the chrome again -
// a narrower window wraps the gallery head onto more lines - and each image's
// `sizes`, and a row that now fits, or no longer does, is laid out again.
// Measured once per render, a QHD sheet dragged down to 1400 px kept the
// `sizes` of 339 px cells drawn at 204 and its note under the window's edge,
// and a row wrapped at 800 px stayed wrapped at QHD.
let refitTimer = null;
function refitSheet() {
  clearTimeout(refitTimer);
  refitTimer = setTimeout(() => {
    if (!sheetMode || !sheetLayout) return;
    const { colAxis, rowAxis, colValues, source, wrap } = sheetLayout;
    if (sheetWraps(colAxis, rowAxis, colValues.length + (source ? 1 : 0)) !== wrap) renderSheet();
    else fitSheet();
  }, 150);
}

// A populated cell's record: the sheet's, or a strength sweep's source,
// which is no gallery record of the sheet's.
function sheetCellRecord(cell) {
  const id = Number(cell.dataset.id);
  return sheetRecords.find((r) => r.id === id) ?? (sheetLayout?.source?.id === id ? sheetLayout.source : null);
}

// F stars the focused cell from its image or tools; navigation is image-only.
async function sheetKey(e) {
  if (keyboardEventHeld(e) || keyboardEditingTarget(e.target) || e.metaKey || e.altKey || els.lightbox.open) return;
  const cell = e.target.closest('.sheet-cell[data-id]');
  if (!cell) return;
  if (!e.ctrlKey && !e.shiftKey && (e.key === 'f' || e.key === 'F')) {
    e.preventDefault();
    if (e.repeat) return;
    const rec = sheetCellRecord(cell);
    if (!rec) return;
    await starCell(cell, rec, { say: true });
    return;
  }
  // Delete bins the cell's image, as its card's ⌦ does: at once, asking
  // nothing, with Restore in the note. forgetImages takes the focus's place
  // before the cell goes, and the redraw puts it on the cell now there.
  if (!e.ctrlKey && !e.shiftKey && e.key === 'Delete') {
    e.preventDefault();
    if (e.repeat) return;
    const rec = sheetCellRecord(cell);
    if (rec && !rec.binned_at) await deleteImage(rec);
    return;
  }
  // The zoom keys, the lightbox's; Ctrl with them is the browser's page zoom.
  if (!e.ctrlKey && ['+', '=', '-', '_', '0', '1'].includes(e.key)) {
    e.preventDefault();
    sheetZoomKey(e.key, cell);
    return;
  }
  if (e.shiftKey || e.target.tagName !== 'IMG' || (e.ctrlKey && e.key !== 'Home' && e.key !== 'End')) return;
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) return;
  e.preventDefault();
  moveSheetCell(cell, e.key, e.ctrlKey);
}

// A matrix skips holes along its row or column without wrapping at the edge;
// a wrapped list follows rendered rows, like the gallery. Home/End stay in
// the row; Control reaches the first/last populated cell. Buttons retain
// their own navigation, and Enter/Space remain thumb's own actions.
function moveSheetCell(from, key, corner = false) {
  if (!sheetLayout) return;
  const cells = sheetCells();
  const at = cells.indexOf(from);
  let row = from.classList.contains('source') ? sheetCurrentRow : +from.dataset.sheetRow;
  const col = +from.dataset.sheetCol;
  let to = null;
  if (corner) to = key === 'Home' ? cells[0] : cells.at(-1);
  else if (sheetLayout.wrap) {
    if (key === 'ArrowUp' || key === 'ArrowDown') to = rowNeighbour(cells, at, key === 'ArrowDown');
    else if (key === 'ArrowLeft' || key === 'ArrowRight') to = cells[at + (key === 'ArrowLeft' ? -1 : 1)];
    else {
      const top = from.getBoundingClientRect().top;
      const sameRow = cells.filter((cell) => Math.abs(cell.getBoundingClientRect().top - top) < 1);
      to = key === 'Home' ? sameRow[0] : sameRow.at(-1);
    }
  } else if (key === 'ArrowUp' || key === 'ArrowDown') {
    const column = cells.filter((cell) => +cell.dataset.sheetCol === col && cell !== from);
    const forward = key === 'ArrowDown';
    const next = column.filter((cell) => forward ? +cell.dataset.sheetRow > row : +cell.dataset.sheetRow < row);
    to = forward ? next[0] : next.at(-1);
  } else {
    const sameRow = cells.filter((cell) => cell.classList.contains('source') || +cell.dataset.sheetRow === row);
    if (key === 'Home' || key === 'End') to = key === 'Home' ? sameRow[0] : sameRow.at(-1);
    else {
      const next = sameRow.filter((cell) => key === 'ArrowRight' ? +cell.dataset.sheetCol > col : +cell.dataset.sheetCol < col);
      to = key === 'ArrowRight' ? next[0] : next.at(-1);
    }
  }
  if (!to) return;
  // A source spans every matrix row: Left/Home can enter it from any row,
  // and Right returns to that row rather than jumping to the first pass.
  if (!to.classList.contains('source')) row = +to.dataset.sheetRow;
  else if (corner) row = 0;
  makeSheetCurrent(to, row);
  to.querySelector('img').focus({ preventScroll: true });
  revealSheetCell(to);
}

// Reveal keyboard movement inside the sheet's own box, under sticky heads.
// scrollIntoView would scroll the page too. Redraw/restored focus does not
// call this, so it moves no scroll position.
function revealSheetCell(cell) {
  const scroller = els.sheetGrid.parentElement;
  const box = scroller.getBoundingClientRect();
  const style = getComputedStyle(scroller);
  const rect = cell.getBoundingClientRect();
  const top = box.top + scroller.clientTop + (parseFloat(style.scrollPaddingTop) || 0);
  const left = box.left + scroller.clientLeft + (parseFloat(style.scrollPaddingLeft) || 0);
  const bottom = box.top + scroller.clientTop + scroller.clientHeight;
  const right = box.left + scroller.clientLeft + scroller.clientWidth;
  if (rect.top < top) scroller.scrollTop += rect.top - top;
  else if (rect.bottom > bottom) scroller.scrollTop += Math.min(rect.top - top, rect.bottom - bottom);
  if (rect.left < left) scroller.scrollLeft += rect.left - left;
  else if (rect.right > right) scroller.scrollLeft += Math.min(rect.left - left, rect.right - right);
}

// The Sheet button. Over an empty search a gallery of several runs makes a
// poor matrix - each run's axes crossed with every other's, mostly holes - so
// the button opens the newest run, as picking it from Runs would, and says so;
// clearing the search then lays out the whole gallery. A search the user typed is
// theirs to lay out, however many runs it spans.
function toggleSheet() {
  if (sheetMode) { void setSheet(false); return; }
  const runs = new Set(galleryRecords.map((r) => r.run_id).filter(Boolean));
  if (!els.search.value.trim() && runs.size > 1) {
    const [newest] = runs;
    void showSheetScope(`run:${newest}`, {
      scroll: false, notice: 'opened on the newest run: clear the search to lay out the whole gallery',
    });
    return;
  }
  void setSheet(true);
}

// `land`: once the sheet is drawn, bring its head up under the topbar
// (landOnSheet) - asked by showSheetScope, and done regardless where the
// sheet takes the page out of the gallery's column beside the viewer, which
// is no width for a sheet: the page is one column again, and the sheet,
// under the viewer, would otherwise open out of sight of the press.
// `asker` is the control that asked for it, for the landing's focus
// (focusSheet). Every sheet opened is said (saySheet).
function setSheet(on, { land = false, asker = null, reload = true } = {}) {
  sheetMode = on;
  // Every sheet opens at its own size (sheet.js's zoom), said afresh.
  sheetZoom = 1;
  sheetZoomBase = 0;
  applySheetZoom();
  els.sheetZoomSaid.textContent = '';
  if (on && besideOn) land = true;
  syncBeside();
  els.sheetBtn.classList.toggle('active', on);
  els.sheet.hidden = !on;
  els.gallery.hidden = on;
  // The cards' size and fit (cardView) are the cards': a sheet's cells are
  // its own.
  els.cardView.hidden = on;
  // Select is held on a sheet, whose cells are not picked (gallery-select.js).
  syncSelection();
  // The room a scroll to the last sheet reserved under it (reserveSheetRoom).
  els.sheet.style.marginBottom = '';
  // Off is a real escape hatch: whatever layout and pins were picked go with
  // it, and so does a search the page wrote to open the sheet - a sweep's
  // three images under its job: search looked like a gallery whose other
  // images had gone.
  if (!on) {
    sheetAxes = { rows: null, cols: null, pins: {} };
    if (sheetReturn && els.search.value.trim() === sheetReturn.wrote) els.search.value = sheetReturn.search;
    sheetReturn = null;
  }
  // A compound filter transition closes the sheet before writing all its
  // fields, then starts the one replacement read itself.
  const opened = reload ? resetGallery() : Promise.resolve();
  if (!on) {
    els.sheetSaid.textContent = '';
    return opened;
  }
  // The query just begun: another one drops what follows with it.
  const epoch = galleryEpoch;
  return opened.then(() => {
    if (!sheetMode || epoch !== galleryEpoch) return;
    saySheet();
    if (land) landOnSheet(asker);
  });
}

// Resolves once the page has stopped scrolling - a smooth scroll's frames
// run out, or there were none - read as three frames at one scrollY, or a
// second and a half at most.
function scrollSettled() {
  return new Promise((resolve) => {
    const start = performance.now();
    let last = scrollY, still = 0;
    const tick = () => {
      if (scrollY === last) still += 1;
      else { still = 0; last = scrollY; }
      if (still >= 3 || performance.now() - start > 1500) resolve();
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

// Once a page-opened sheet has been landed on (landOnSheet), the focus. Left
// on what asked - Run comparison at every width, the finish line's Show
// sheet in one column - it was out of sight above the sheet, and the next
// Tab scrolled the page back away from it; where the sheet took it away -
// a card's ⊞, the gallery hidden under the sheet - it had fallen to <body>.
// Either goes to the sheet's first cell, without moving the page (else the
// search box, over a sheet with nothing in it). `asker` is the control that
// asked, or where the page itself put focus since (handJobFocus). Focus
// anywhere else stays: a comparison's sheet opens when its job ends, maybe
// while the prompt is being typed in.
function focusSheet(asker) {
  const now = document.activeElement;
  const lost = !now || now === document.body || !isShown(now);
  if (!lost && (now !== asker || wholeInSight(now))) return;
  (els.sheetGrid.querySelector('.sheet-cell img[tabindex="0"]') ?? els.search).focus({ preventScroll: true });
}

// Whether all of `el` can be seen: inside the window under the topbar and
// inside every box round it that clips what overflows it. The window alone
// is not enough: at 2560 the docked prompt check scrolls inside itself, and
// once the sheet was landed on, Run comparison lay in the window under the
// topbar and wholly outside the check's own box - out of sight, and the next
// Tab took the page back to it.
function wholeInSight(el) {
  const box = el.getBoundingClientRect();
  let top = document.querySelector('.topbar').getBoundingClientRect().bottom;
  let bottom = innerHeight, left = 0, right = innerWidth;
  for (let up = el.parentElement; up && up !== document.body; up = up.parentElement) {
    const style = getComputedStyle(up);
    if (style.display === 'contents' || (style.overflowX === 'visible' && style.overflowY === 'visible')) continue;
    const clip = up.getBoundingClientRect();
    top = Math.max(top, clip.top);
    bottom = Math.min(bottom, clip.bottom);
    left = Math.max(left, clip.left);
    right = Math.min(right, clip.right);
  }
  return box.top >= top - 0.5 && box.bottom <= bottom + 0.5 && box.left >= left - 0.5 && box.right <= right + 0.5;
}
