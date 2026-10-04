// Local Diffusion's page script: the contact sheet and its export. Declarations only; state.js has
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
  // Which adapter the weight belongs to: two adapters' sweeps at the same
  // weights shared their cells, one of each pair hidden.
  { key: 'adapter', label: 'Adapter', of: (r) => (r.lora ? adapterName(r.lora) : null) },
  // The reference images, by gallery id and in their order. With and without
  // a reference on one seed is the comparison klein is used for, and without
  // an axis its images all landed in one cell. Null where there were none.
  { key: 'ref', label: 'References', of: (r) => (r.reference?.length ? r.reference.map((id) => `#${id}`).join(' + ') : null) },
  // A refine pass's strength, null on a direct render for the same reason:
  // a strength sweep over one source is a row of how much of it was redrawn.
  { key: 'strength', label: 'Strength', of: (r) => r.strength },
  { key: 'seed', label: 'Seed', of: (r) => r.seed },
];
const trueCfgEchoesCfg = (records) =>
  records.every((r) => r.true_cfg == null || r.true_cfg === r.guidance_scale);
const adapterName = (file) => file.replace(/\.safetensors$/, '');

// A comparison's versions by the names the prompt check gives them, on the
// sheet and in its export; the stored labels - what `stats --by variant`
// groups by - stay as they are. A label this does not know (a user's own
// `variant` variable) reads as written.
const VERSION_NAMES = new Map([
  ['original', 'Original'], ['reordered', 'Reordered'], ['rewrite', 'Rewrite'], ['yours', 'Your version'],
]);

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
    label: `var:${name}` === VARIANT_AXIS ? 'Version' : name,
    // `""` is the deliberately empty value - the word left out - and reads as
    // `none`; a record with no variables at all (a mixed search) reads as null.
    of: (r) => {
      const value = r.variables ? r.variables[name] : undefined;
      return value === undefined ? null : (value === '' ? 'none' : value);
    },
  }));
  // Comparisons can share a run after original and yours are swapped.
  // A mixed search can do the same. Keep Prompt as an axis wherever the same
  // variable assignment names different text, instead of overwriting one of
  // those images in the (variant, seed) cell. Ordinary grids stay unchanged.
  const prompts = new Map();
  const keys = [...names].sort();
  const ambiguous = names.includes('variant') && records.some((rec) => {
    const assignment = JSON.stringify(keys.map((name) => rec.variables?.[name] ?? null));
    if (prompts.has(assignment)) return prompts.get(assignment) !== rec.prompt;
    prompts.set(assignment, rec.prompt);
    return false;
  });
  return [...variableAxes, ...flat.filter((a) => a.key !== 'negative' && (a.key !== 'prompt' || ambiguous))];
}
const axisByKey = (key) => sheetAxesList.find((a) => a.key === key);
const axisValue = (axis, rec) => axis.of(rec) ?? null;
// Two images that agree on every axis a sheet can have and on what none lays
// out - the size, the checkpoint and its quantization, the image a pass
// refined - are one picture asked for twice. Every axis, not the sheet's own
// list, which leaves True CFG out where it echoes CFG.
const sameSettings = (a, b) => SHEET_AXES.every((axis) => axisValue(axis, a) === axisValue(axis, b))
  && ['width', 'height', 'model_id', 'quantization', 'refine'].every((key) => a[key] === b[key]);

// Full prompt -> the words that make it different from its neighbours, filled
// in by sheetValues. Only the prompt axis is in here, so anything else that is
// a string prints as itself. Whole: a heading wraps, three lines at most with
// its full text as the title, and the 44 characters `clip` keeps are for a
// line of fixed width - a pin's option, the note, the export's canvas. A
// 540 px column used to show 44 characters across half its width.
let promptShort = new Map();
const valueText = (v) => {
  if (v === null) return 'none';
  if (typeof v === 'string') return promptShort.get(v) ?? v;
  return String(v);
};

const LABEL_CHARS = 44;
const clip = (text) => (text.length > LABEL_CHARS ? `${text.slice(0, LABEL_CHARS - 1)}…` : text);

// The adapter every LoRA weight on the sheet belongs to, while there is only
// one: a weight sweep's heads and its export then say whose weights they are,
// and sheets of several adapters' sweeps can be told apart afterwards.
let sheetAdapter = null;

// A value as its axis reads it - a version by its name - and a heading: the
// axis and the value, a known version by its name alone ("Rewrite", not
// "Version rewrite"), a weight with its adapter.
function sheetText(axis, v) {
  if (axis.key === VARIANT_AXIS && v !== null) return VERSION_NAMES.get(v) ?? v;
  return valueText(v);
}
function headText(axis, v) {
  if (axis.key === VARIANT_AXIS && VERSION_NAMES.has(v)) return VERSION_NAMES.get(v);
  if (axis.key === 'lora' && sheetAdapter && v !== null) return `LoRA ${sheetAdapter} ${v}`;
  return `${axis.label} ${sheetText(axis, v)}`;
}

// What one prompt changed against another, a run of edits at a time:
// `−dawn +dusk`, `+35mm photo,`. From promptChanges (analysis.js), the exact
// diff the critic's rewrite is shown with; edits only spaces apart read as
// one, and the runs are joined by an ellipsis for the words they kept.
function editRuns(base, text) {
  const runs = [];
  let run = null;
  for (const { kind, text: part } of promptChanges(base, text)) {
    if (kind === 'same' && !(run && /^\s+$/.test(part))) { run = null; continue; }
    if (!run) { run = { removed: '', added: '' }; runs.push(run); }
    if (kind === 'same') { run.removed += part; run.added += part; } else run[kind] += part;
  }
  const words = (part) => part.trim().replace(/\s+/g, ' ');
  return runs.map(({ removed, added }) => ({ removed: words(removed), added: words(added) }));
}
function promptEdit(base, text) {
  return editRuns(base, text).map(({ removed, added }) => [
    removed && `−${removed}`,
    added && `+${added}`,
  ].filter(Boolean).join(' ')).join(' … ');
}

// A version's edit against the original, under its row head. One that keeps
// the original's words and moves some - the critic's Reordered, mostly - read
// as each moved run removed in one place and added in another, the same words
// twice, a puzzle where "same words, new order" was meant; it says what moved
// once, from the added side.
function versionEdit(original, prompt) {
  const words = (text) => text.match(/[\p{L}\p{N}]+/gu) ?? [];
  const [before, after] = [words(original), words(prompt)];
  const sorted = (list) => [...list].sort().join(' ');
  if (before.join(' ') === after.join(' ') || sorted(before) !== sorted(after)) return promptEdit(original, prompt);
  const moved = editRuns(original, prompt)
    .map(({ added }) => added.replace(/^[\s,;:.]+|[\s,;:.]+$/g, ''))
    .filter(Boolean);
  return `same words, moved: ${moved.join(' … ')}`;
}

// A header that repeats the forty words every prompt shares is unreadable, and
// the two or three that differ are the whole comparison - so the label keeps
// what varies and drops what does not.
//
// Same word count throughout means the varying positions can be picked out
// exactly, which is what a two-variable grid produces. A value that was set to
// `none` makes its prompt shorter than the others, and then the shared head and
// tail come off instead - `big dog on grass` against `dog on grass` labels
// itself `big` and `none`.
//
// Successive edits of a prompt - a word added here, one swapped there - leave
// neither: no equal counts, and a word added at the front keeps every head and
// tail, so six iterations labelled themselves with three clipped openings.
// Where the labels come out repeated or longer than a line, each prompt is
// labelled by its edit against the oldest (`prompts` is in the order they were
// written - sheetValues), which keeps its own words. Not past EDIT_LABELS_MAX
// prompts: each edit is a diff, quadratic in the prompt's length, and a typed
// search over a mixed gallery - a thousand prompts of 250 words - froze the
// tab for 600 ms to label columns nobody reads across sixty-odd.
const EDIT_LABELS_MAX = 64;
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
  const labels = kept.map((w) => w.join(' ') || 'none');
  if (prompts.length > EDIT_LABELS_MAX) return labels;
  if (new Set(labels).size === labels.length && labels.every((l) => l.length <= LABEL_CHARS)) return labels;
  return prompts.map((p, i) => (i === 0 ? p.trim() : promptEdit(prompts[0], p) || p.trim()));
}
// The labels of the last prompt list. Every redraw of the same records - a
// pin, an axis, a star, a delete - asks for them again.
let promptLabelMemo = { key: null, labels: [] };

// The distinct values each axis takes across the loaded images, ascending - or,
// for a variable, in the order the grid wrote them: by the oldest image carrying
// each value. The grid renders a variable's values as declared and a prompt
// variant's in the order given, so the first image of each is in that order;
// alphabetical read a `size = small | big` grid as big, small. The prompt goes
// by its oldest image too: a search over a prompt's iterations reads in the
// order they were written, where alphabetical put the newest first. `whole` is
// the rest of the run when a filter hid some of it (learnSheetOrder), since the
// oldest *loaded* image of a value is only the oldest the filter kept.
function sheetValues(records, whole = []) {
  const values = {};
  for (const axis of sheetAxesList) {
    const seen = [];
    const written = axis.key.startsWith('var:') || axis.key === 'prompt';
    const oldest = new Map();
    for (const rec of records) {
      const v = axisValue(axis, rec);
      if (!seen.includes(v)) seen.push(v);
      if (written && (!oldest.has(v) || rec.id < oldest.get(v))) oldest.set(v, rec.id);
    }
    if (written) {
      for (const rec of whole) {
        const v = axisValue(axis, rec);
        if (oldest.has(v) && rec.id < oldest.get(v)) oldest.set(v, rec.id);
      }
    }
    seen.sort((a, b) => {
      if (a === null) return -1;
      if (b === null) return 1;
      if (written) return oldest.get(a) - oldest.get(b);
      return typeof a === 'string' ? a.localeCompare(b) : a - b;
    });
    values[axis.key] = seen;
  }
  // No prompt axis when variables carry the variation; nothing to label.
  const prompts = values.prompt || [];
  const key = prompts.join('\u0000');
  if (promptLabelMemo.key !== key) {
    promptLabelMemo = { key, labels: prompts.length > 1 ? promptLabels(prompts) : prompts };
  }
  const { labels } = promptLabelMemo;
  promptShort = new Map(prompts.map((text, i) => [text, labels[i]]));
  return values;
}

// The whole run behind a filtered sheet, read for its order alone. Under the
// Starred tick, or a search narrower than `run:<id>`, the oldest loaded image
// of a value is the oldest that survived: star the winners and a comparison
// read reordered, rewrite, original. So loadGallery asks once more, for the
// same run unfiltered, before it renders. A row the server no longer has
// (Delete unstarred) cannot be asked for; its value orders by what is left.
// Failing this, the loaded images order themselves, as they always did.
// learnSheetOrder asks for that and for a refine sweep's source (below) at
// once, and loadGallery renders when both have answered.
let sheetWritten = null;
function learnSheetOrder(epoch) {
  return Promise.all([learnRunOrder(epoch), learnSheetSource()]);
}

async function learnRunOrder(epoch) {
  sheetWritten = null;
  const runs = new Set(galleryRecords.map((r) => r.run_id));
  const [run] = runs;
  if (runs.size !== 1 || !run || !galleryRecords.some((r) => r.variables)) return;
  // A comparison's submission may reverse an earlier comparison's labels
  // within this run. Learn its order from that job, even under Starred.
  const job = galleryRecords[0]?.job_id;
  const jobScope = job && galleryRecords.every((r) => r.job_id === job)
    && els.search.value.trim().split(/\s+/).includes(`job:${job}`);
  const search = jobScope ? `job:${job}` : `run:${run}`;
  if (!els.favOnly.checked && els.search.value.trim() === search) return;
  const params = new URLSearchParams({ limit: sheetPage, search });
  if (els.allModels.checked) params.set('all_models', 'true');
  try {
    const page = await api(`/api/gallery?${params}`);
    if (epoch === galleryEpoch) sheetWritten = { epoch, records: page.items };
  } catch { /* unordered by the run, not failed */ }
}

// A refine sweep's source: the image every pass on the sheet was made from,
// drawn as a column of its own ahead of the passes (renderSheet), so a
// strength is chosen against the original rather than among the passes - in
// sheet mode the gallery that holds it is hidden. By id, asked for with each
// query the sheet shows and kept for the redraws between them; never a sheet
// record, so it is no axis value, no cell of the matrix and no count. Null
// for a source that has been deleted (forgetImages says so too).
const sheetSources = new Map();
const sharedSource = (records) => {
  const id = records[0]?.refine;
  return id != null && records.every((r) => r.refine === id) ? id : null;
};
async function learnSheetSource() {
  const id = sharedSource(galleryRecords);
  if (id === null) return;
  if (galleryRecords.some((r) => r.refine_missing)) { sheetSources.set(id, null); return; }
  try {
    sheetSources.set(id, await api(`/api/images/${id}`));
  } catch (err) {
    // Gone is an answer; anything else is asked again by the next render.
    if (/unknown image/.test(err.message)) sheetSources.set(id, null);
  }
}

// A line the sheet adds to its note while it shows the query it was opened
// on: what the page changed to open it (showRun's Starred tick, the Sheet
// button's newest run).
let sheetNotice = null;

// The search a sheet the page opened replaced, and what the page wrote over
// it (showSheetScope): Sheet off puts it back while the box still holds what
// was written. A search typed since is the user's own and stays.
let sheetReturn = null;

// The axis a prompt-variant grid records its labels under - `grid.VARIANT`,
// held to it by tests/test_grid.py.
const VARIANT_AXIS = 'var:variant';

// Columns and rows default to the two axes that actually vary, widest first.
// A third varying axis becomes a pin rather than a third dimension: a flat
// sheet that crosses everything at once is a wall nobody can read.
//
// Except that prompt variants are rows, whatever the widths: one row per
// version of the prompt, in the order written, the seeds across. Widest-first
// alone gave that only while the seeds outnumbered the variants - 3 x 3, 3 x 2
// or 8 x 4 turned the variants into columns, the tie going to the variable
// axes because axesFor lists them first. Alone, the variant axis stays a strip:
// a sheet with nothing across is one column of full-width images.
function chooseAxes(values, count) {
  const all = sheetAxesList.filter((a) => values[a.key].length > 1);
  const wider = (a, b) => values[b.key].length - values[a.key].length;
  let varying = all;
  // An axis with one value per image is a list, not a dimension - the seed of
  // a mixed gallery, mostly. It is dropped whenever anything else varies; a
  // seeds-only run keeps it, since it is then the only dimension there is.
  // The variant axis and the one across it are the exception, placed below.
  const listlike = varying.filter((a) => values[a.key].length === count);
  if (listlike.length < varying.length) varying = varying.filter((a) => !listlike.includes(a));
  varying.sort(wider);
  // The variants and what runs across them are placed from every varying
  // axis, not from what the list rule left: with one image per version loaded
  // - a star each under Starred, or what Delete unstarred kept - the rule
  // dropped the variants, or the seeds they cross, and the pin in their place
  // showed one image of the comparison.
  const variant = all.find((a) => a.key === VARIANT_AXIS);
  if (variant && all.length > 1) {
    const others = (axes) => axes.filter((a) => a !== variant);
    varying = [others(varying)[0] ?? others(all).sort(wider)[0], variant];
  }
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
//
// A new pin takes the value most images carry under the pins already held.
// The first value it used to take is `none` for every variable of a mixed
// gallery - the images from plain prompts carry none - and no image is `none`
// on all of them at once, so the Sheet button over a whole gallery filled not
// one cell. Over a full run every value has as many images and the first
// still wins.
function reconcileAxes(values) {
  const pins = {};
  const open = [];
  for (const axis of sheetAxesList) {
    const key = axis.key;
    if (values[key].length < 2 || key === sheetAxes.cols || key === sheetAxes.rows) continue;
    const kept = sheetAxes.pins[key];
    if (values[key].includes(kept)) pins[key] = kept;
    else open.push(axis);
  }
  for (const axis of open) {
    const held = Object.entries(pins).map(([key, v]) => [axisByKey(key), v]);
    const counts = new Map();
    for (const rec of sheetRecords) {
      if (held.some(([a, v]) => axisValue(a, rec) !== v)) continue;
      const v = axisValue(axis, rec);
      counts.set(v, (counts.get(v) || 0) + 1);
    }
    const count = (v) => counts.get(v) || 0;
    pins[axis.key] = values[axis.key].reduce((best, v) => (count(v) > count(best) ? v : best));
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

// A heading holds its label in a span of its own, which the stylesheet clamps
// to three lines while the heading - stuck to the grid's edge - covers its
// whole row or column; its title is the value whole, a prompt's full text
// where the label is the words that vary.
function headCell(text, className, title = text) {
  const div = document.createElement('div');
  div.className = className || 'head';
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = text;
  div.appendChild(label);
  if (title) div.title = title;
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
// The least a matrix's cell is drawn at, and the gap between cells
// (.sheet-grid's): more columns than fit scroll sideways rather than squeeze
// their thumbnails past it.
const SHEET_CELL_MIN = 96;
const SHEET_GAP = 6;
// backend/storage.py's THUMB_SIDE: the longer edge of the WEBP a cell shows.
const THUMB_SIDE = 384;

// Settings nothing on the sheet has - no adapter, no references, no pass -
// are left out of the Fixed line rather than listed as `none`.
const ABSENT_WHEN_NONE = new Set(['tcfg', 'lora', 'adapter', 'ref', 'strength']);

// The settings every image on the sheet shares. Under the Starred tick they
// are what made the keepers, and the tick takes them off every axis: two
// starred images read as two cells, with nothing saying they shared a seed,
// steps and CFG. A line of its own rather than part of the note.
function fixedSettings(values) {
  const fixed = [];
  for (const axis of sheetAxesList) {
    if (values[axis.key].length !== 1 || axis.key === 'prompt' || axis.key === 'negative') continue;
    const [v] = values[axis.key];
    if (v === null && (ABSENT_WHEN_NONE.has(axis.key) || axis.key.startsWith('var:'))) continue;
    fixed.push(`${axis.label} ${sheetText(axis, v)}`);
  }
  return fixed.length ? `Fixed: ${fixed.join(' · ')}` : '';
}

// Each version of a comparison by its prompt, where its label names one
// prompt only - axesFor keeps Prompt an axis where a label names several. A
// version's row head carries the whole prompt as its title and, under the
// name, what it changed against the original: four 400-character versions
// open with the same words.
function versionPrompts(records) {
  const prompts = new Map();
  for (const rec of records) {
    const label = rec.variables?.variant;
    if (label === undefined) continue;
    if (!prompts.has(label)) prompts.set(label, new Set());
    prompts.get(label).add(rec.prompt);
  }
  const single = [...prompts].filter(([, texts]) => texts.size === 1);
  return new Map(single.map(([label, texts]) => [label, [...texts][0]]));
}

// The run the sheet shows, where it shows one and was opened on `run:<id>` -
// the Runs picker, a card's ⊞, Show run: what Resume this run continues. Not
// a comparison's or a refine sweep's, which the panel's grid did not make,
// and not another checkpoint's (All models): resume is scoped to the
// checkpoint the server runs, so its preview could only skip nothing.
function sheetRun() {
  const runs = new Set(sheetRecords.map((r) => r.run_id));
  const [run] = runs;
  if (runs.size !== 1 || !run || !els.search.value.trim().split(/\s+/).includes(`run:${run}`)) return null;
  if (loadedModel && sheetRecords[0].model_id !== loadedModel) return null;
  return sheetRecords.some((r) => r.refine != null || r.variables?.variant !== undefined) ? null : run;
}

// A sheet with nothing down it and more across than fits - a seeds-only run,
// mostly, which is every Batch or Repeat over 1 - wraps into rows of
// gallery-sized cells, each captioned, rather than one ribbon of 96 px cells
// several screens wide. One that fits stays a row under its heads.
function sheetWraps(colAxis, rowAxis, columns) {
  return !!colAxis && !rowAxis
    && columns * (SHEET_CELL_MIN + SHEET_GAP) > els.sheetGrid.parentElement.clientWidth;
}

function renderSheet() {
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
        values[key].map((v, index) => [index, clip(sheetText(axis, v))]),
        values[key].indexOf(pinned),
        (picked) => {
          sheetAxes.pins[key] = values[key][Number(picked)] ?? null;
          renderSheet();
        },
      )),
    );
  }
  const run = sheetRun();
  if (run) {
    const resume = document.createElement('button');
    resume.className = 'ghost small';
    resume.textContent = 'Resume this run';
    resume.title = 'Put the seed this run started from and its size in the panel, tick Resume and preview '
      + 'what is left of it. The grid in the panel has to be the one that made the run';
    resume.onclick = () => { void resumeRun(run); };
    els.sheetControls.append(resume);
  }
  if (sheetRecords.length) {
    const hint = document.createElement('span');
    hint.className = 'sheet-hint';
    hint.textContent = 'F stars the image in focus';
    els.sheetControls.append(hint);
  }
  els.sheetFixed.textContent = fixedSettings(values);
  els.sheetFixed.hidden = !els.sheetFixed.textContent;

  const colAxis = sheetAxes.cols ? axisByKey(sheetAxes.cols) : null;
  const rowAxis = sheetAxes.rows ? axisByKey(sheetAxes.rows) : null;
  const colValues = colAxis ? values[colAxis.key] : [null];
  const rowValues = rowAxis ? values[rowAxis.key] : [null];
  els.sheetGrid.innerHTML = '';
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

  // One pass over the records into a map keyed by (row, column), rather than a
  // scan of every record per cell, so the sheet costs O(records + cells). The
  // first record wins a cell; they arrive newest first. The ones it turns
  // away are counted for the note, where "1 of 3 loaded" read as two images
  // still on their way: a repeat of the cell's image - the same grid run
  // again without Resume, which keeps its run id, or a Re-render - or one
  // that differs in something no axis lays out, the size, the checkpoint.
  const pins = Object.entries(sheetAxes.pins).map(([key, v]) => [axisByKey(key), v]);
  const rowIndex = new Map(rowValues.map((v, i) => [v, i]));
  const colIndex = new Map(colValues.map((v, i) => [v, i]));
  const cellOf = new Map();
  let repeats = 0, crowded = 0;
  for (const rec of sheetRecords) {
    if (pins.some(([axis, v]) => axisValue(axis, rec) !== v)) continue;
    const r = rowAxis ? rowIndex.get(axisValue(rowAxis, rec)) : 0;
    const c = colAxis ? colIndex.get(axisValue(colAxis, rec)) : 0;
    const at = r * colValues.length + c;
    if (!cellOf.has(at)) cellOf.set(at, rec);
    else if (sameSettings(rec, cellOf.get(at))) repeats += 1;
    else crowded += 1;
  }

  const sourceId = sharedSource(sheetRecords);
  const source = sourceId === null ? null : sheetSources.get(sourceId) ?? null;
  const columns = colValues.length + (source ? 1 : 0);
  const wrap = sheetWraps(colAxis, rowAxis, columns);

  // What the export draws: the same matrix, the same pins, the same labels.
  sheetLayout = { colAxis, rowAxis, colValues, rowValues, cellOf, pins, values, wrap, source };

  // One aspect ratio for the whole sheet - size is not a sheet axis, and a
  // free-text search can mix sizes - so the commonest one wins and the odd
  // cell letterboxes inside it rather than being cropped.
  const ratios = new Map();
  for (const rec of sheetRecords) {
    const key = `${rec.width} / ${rec.height}`;
    ratios.set(key, (ratios.get(key) || 0) + 1);
  }
  const [aspect] = [...ratios.entries()].sort((a, b) => b[1] - a[1])[0] || ['1'];
  const [aspectW, aspectH] = aspect.split('/').map(Number);
  els.sheetGrid.style.setProperty('--cell-aspect', aspect);
  // What the stylesheet sizes a matrix's cells from (--cell-max): the
  // aspect as a number and the rows that share the window's height.
  els.sheetGrid.style.setProperty('--sheet-ar', String(aspectH ? aspectW / aspectH : 1));
  els.sheetGrid.style.setProperty('--sheet-rows', String(rowValues.length));
  els.sheet.classList.toggle('wrapped', wrap);
  els.sheetGrid.style.gridTemplateColumns = wrap
    ? 'repeat(auto-fill, minmax(var(--thumb-min), 1fr))'
    : `auto repeat(${columns}, minmax(${SHEET_CELL_MIN}px, var(--cell-max)))`;

  const shown = [];
  // What the lightbox walks from a cell: the sheet in reading order, after the
  // source where there is one, so the arrows flip the original against each
  // pass. Still filling, but it is the same array by the time anything can be
  // clicked - so the arrows walk the sheet rather than the gallery.
  const walk = source ? [source] : [];
  // A pass looks like its source by design; a cell says it is one, unless
  // the strength is what the sheet lays out already (an axis or a pin).
  const strengthShown = values.strength.length > 1;
  const cellFor = (rec, caption, badge) => {
    const cell = document.createElement('div');
    cell.className = 'sheet-cell';
    if (!rec) {
      cell.classList.add('empty');
      return cell;
    }
    cell.classList.toggle('fav', !!rec.favorite);
    cell.dataset.id = rec.id;
    const img = thumb(rec, walk);
    // The full PNG beside the thumbnail, for a cell drawn wider than the
    // thumbnail is (fitSheet sets `sizes`): 384 px shown at 520 was soft
    // exactly where a CFG or steps sweep differs. Lazy loading still fetches
    // only the cells on screen.
    const thumbWidth = Math.round(rec.width * Math.min(1, THUMB_SIDE / Math.max(rec.width, rec.height)));
    if (rec.thumb_url && rec.width > thumbWidth) {
      img.srcset = `${rec.thumb_url} ${thumbWidth}w, ${rec.url} ${rec.width}w`;
    }
    img.setAttribute('aria-keyshortcuts', 'F');
    cell.appendChild(img);
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
  const add = (el) => els.sheetGrid.appendChild(el);
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
    const head = headCell(headText(rowAxis, v), `rowhead${prose(rowAxis)}`, prompt ?? headTitle(rowAxis, v));
    const edit = prompt && original && v !== 'original' ? versionEdit(original, prompt) : '';
    if (edit) {
      const line = document.createElement('span');
      line.className = 'rowhead-edit';
      line.textContent = edit.length > 80 ? `${edit.slice(0, 79)}…` : edit;
      head.classList.add('has-edit');
      head.appendChild(line);
    }
    return head;
  };

  if (!wrap) {
    add(headCell('', 'head corner'));
    if (source) {
      add(headCell(`Source #${source.id}`, 'head',
        `#${source.id}, ${source.width}×${source.height}: the image every pass here was refined from`));
    }
    colValues.forEach((v) => {
      add(colAxis
        ? headCell(headText(colAxis, v), `head${prose(colAxis)}`, headTitle(colAxis, v))
        : headCell('', 'head'));
    });
  }
  rowValues.forEach((rowValue, r) => {
    if (!wrap) add(rowHead(rowValue));
    // The source spans every row: one image the passes are all set against.
    if (source && r === 0) {
      const cell = add(cellFor(source, wrap ? `Source #${source.id}` : '', true));
      cell.classList.add('source');
      if (!wrap) cell.style.gridRow = `span ${rowValues.length}`;
    }
    colValues.forEach((v, c) => {
      const rec = cellOf.get(r * colValues.length + c);
      if (rec) {
        shown.push(rec);
        walk.push(rec);
      }
      add(cellFor(rec, wrap ? headText(colAxis, v) : '', !strengthShown));
    });
  });

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
    (notice ? ` · ${notice.text}` : '');
  fitSheet();
}

// The tallest a matrix's cells may be: the window, less what sits above the
// grid once showSheetScope has landed on the gallery head - the topbar, the
// head, the controls, the Fixed line - and the note below it, and less the
// column heads, shared between the rows (the stylesheet's --cell-max). The
// box the grid scrolls in is no taller, so a matrix that still does not fit
// keeps its heads in view. Measured rather than guessed, so a tier that
// scales the controls needs nothing here, and `100vh` follows the window from
// there. At QHD a 4 x 3 used to be 524 px cells in a sheet 1,607 px tall.
// The heads' row and the row heads' column (the corner is as wide) are also
// the box's scroll padding: a cell focused from the keyboard is scrolled
// clear of the heads stuck over it, where it lined up with the box's edge and
// the heads hid its top and its focus ring. Then each image's `sizes`, the
// width its cell is drawn at - the cells of a sheet are one width - which
// picks the thumbnail or the PNG.
function fitSheet() {
  const scroller = els.sheetGrid.parentElement;
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
  const cell = els.sheetGrid.querySelector('.sheet-cell');
  if (!cell) return;
  const sizes = `${Math.ceil(cell.getBoundingClientRect().width)}px`;
  for (const img of els.sheetGrid.querySelectorAll('img[srcset]')) img.sizes = sizes;
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

// F stars the image in focus, as it does in the lightbox, so keepers are
// marked with the sheet in view rather than one lightbox trip each - the side
// by side view is what the choice is made on. One listener on the grid (app.js)
// rather than a button per cell, which would be up to 1,024 more tab stops.
// The value comes from the cell, whose class setFavorite keeps, and goes back
// to the record after: a refine sweep's source is no gallery record for
// setFavorite to find.
async function sheetKey(e) {
  if ((e.key !== 'f' && e.key !== 'F') || e.ctrlKey || e.metaKey || e.altKey || els.lightbox.open) return;
  const cell = e.target.closest('.sheet-cell[data-id]');
  if (!cell) return;
  const id = Number(cell.dataset.id);
  const rec = sheetRecords.find((r) => r.id === id) ?? (sheetLayout?.source?.id === id ? sheetLayout.source : null);
  if (!rec) return;
  e.preventDefault();
  await setFavorite(rec, !cell.classList.contains('fav'));
  const now = els.sheetGrid.querySelector(`.sheet-cell[data-id="${id}"]`);
  if (now) rec.favorite = now.classList.contains('fav');
}

// Resume this run: the panel's grid made ready to finish the run on the
// sheet. Resume needs the seed the grid started from, which every image
// carries only derived from it; /api/runs reads it off the run's oldest image
// - the sheet's own oldest where the run is past the 200 newest and nothing
// filters it. The size comes from the images, all one size in a run: resume
// is scoped by it, and the panel's after any other render was the likelier
// one, so the preview skipped nothing and blamed the grid. Resume is ticked
// for this session only, never remembered. The grid's axes are not stored per
// run, so whether the panel still holds the grid that made it is the
// preview's to say: one that would skip nothing is a different grid, and the
// preview says so rather than Run grid rendering it. `known` is a job's own
// seed and size, the last-run banner's (panel.js): no lookup, and a seed of
// null - a listed seed axis, which is its own - leaves the Seed box alone.
async function resumeRun(runId, known = null) {
  // A refusal is said under the gallery's head, by the sheet whose button
  // this is - and beside the Starred tick one of them asks to untick.
  const slot = { slot: els.galleryError };
  let seed = known?.seed;
  if (!known) {
    const params = new URLSearchParams({ limit: 200 });
    if (els.allModels.checked) params.set('all_models', 'true');
    try {
      seed = (await api(`/api/runs?${params}`)).find((r) => r.run_id === runId)?.seed;
    } catch (err) {
      showError(err.message, false, slot);
      return;
    }
    if (seed == null && !els.favOnly.checked && sheetRecords.every((r) => r.run_id === runId)) {
      seed = sheetRecords.reduce((a, r) => (r.id < a.id ? r : a)).seed;
    }
    if (seed == null) {
      showError(`Run ${runId} is not among the 200 newest; untick Starred so the sheet holds its oldest image.`, false, slot);
      return;
    }
  }
  if (seed != null) els.seed.value = seed;
  const size = known ?? sheetRecords[0];
  els.width.value = size.width;
  els.height.value = size.height;
  syncSize();
  syncRuns();
  persistPanel();
  scheduleComparePreview();
  els.gridResume.checked = true;
  // The fold's toggle event previews on its own; the check below waits for it
  // and then asks again, so the last preview is the one it judges.
  if (!els.gridSection.open) {
    const toggled = new Promise((resolve) => els.gridSection.addEventListener('toggle', resolve, { once: true }));
    els.gridSection.open = true;
    await toggled;
  }
  const preview = previewGrid();
  const epoch = previewEpoch;
  let p = null;
  try {
    p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(gridPayload()),
    });
  } catch { /* the preview says why */ }
  await preview;
  if (p && !p.skipped && epoch === previewEpoch) {
    els.gridPreview.classList.add('warn');
    // Appended as a text node: the preview's first line is a .totals span,
    // which rewriting textContent flattened into the rest.
    els.gridPreview.append(`\nNone of these cells is in run ${runId}: the grid above is not the one that `
      + 'made it (its prompt, axes or other settings changed, or the server runs another quantization), '
      + 'so Run grid would render a different grid.');
  }
  // Last, once the preview has grown the fold to its height: brought into
  // view before it, Run grid ended a few lines under the window's edge.
  revealGridRun();
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
  const { colAxis, rowAxis, colValues, rowValues, cellOf, pins, values, wrap, source } = layout;
  const aspect = (() => {
    const [w, h] = (els.sheetGrid.style.getPropertyValue('--cell-aspect') || '1 / 1').split('/').map(Number);
    return h ? w / h : 1;
  })();
  // Where each cell goes, in the sheet's reading order. A wrapped row wraps
  // here too, into a block about as wide as it is tall, each cell captioned
  // underneath: drawn as one row, the 96 px floor beat the edge limit and a
  // 120-seed run came out 12,254 px wide. A refine sweep's source leads, a
  // column of its own across the rows.
  const items = [];
  const lead = source && !wrap ? 1 : 0;
  let cols, rows;
  if (wrap) {
    const row = colValues.map((v, c) => ({ rec: cellOf.get(c), label: headText(colAxis, v) }));
    if (source) row.unshift({ rec: source, label: `Source #${source.id}` });
    cols = Math.ceil(Math.sqrt(row.length));
    rows = Math.ceil(row.length / cols);
    row.forEach((item, i) => items.push({ ...item, col: i % cols, row: Math.floor(i / cols) }));
  } else {
    cols = colValues.length + lead;
    rows = rowValues.length;
    if (source) items.push({ rec: source, col: 0, row: 0, span: true });
    rowValues.forEach((_, r) => colValues.forEach((__, c) => {
      items.push({ rec: cellOf.get(r * colValues.length + c), col: lead + c, row: r });
    }));
  }
  // Cells shrink so the longer edge stays inside what a canvas can hold.
  const cellW = Math.max(96, Math.min(384, Math.floor(EXPORT_MAX_EDGE / Math.max(cols, rows))));
  const cellH = Math.round(cellW / aspect);
  const gap = 6, font = 14, labelH = wrap ? 22 : 0;
  const headH = (colAxis || source) && !wrap ? 34 : 8, headW = rowAxis ? 150 : 8;
  const width = headW + cols * (cellW + gap) + gap;
  const gridH = headH + rows * (cellH + labelH + gap) + gap;
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

  // The caption is what a screenshot loses: which run, what was pinned, what
  // prompt made it - with a variable run's values as declared, since its
  // template is not a prompt any cell ran - what every image shares, and when
  // the run was made: the oldest image's time, not the export's.
  const first = sheetRecords[0];
  const oldest = first && sheetRecords.reduce((a, r) => (r.id < a.id ? r : a));
  const pinned = pins.map(([axis, v]) => `${axis.label.toLowerCase()} ${clip(sheetText(axis, v))}`).join(', ');
  const opening = items.find((item) => item.rec && item.rec !== source)?.rec;
  const prompt = values.prompt?.length === 1 ? `Prompt: ${values.prompt[0]}`
    : opening ? `Prompt of the first image: ${opening.prompt}` : '';
  const declared = sheetAxesList.filter((a) => a.key.startsWith('var:'))
    .map((a) => `${a.label} = ${values[a.key].map((v) => sheetText(a, v)).join(' | ')}`).join('; ');
  const fixed = fixedSettings(values);
  const max = width - 2 * gap;
  const lines = [
    clipText(`${els.search.value.trim() || 'gallery'}${pinned ? ` · pinned to ${pinned}` : ''}`, max),
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
  ctx.fillStyle = '#e8e8ee';
  ctx.font = typeface;
  ctx.textBaseline = 'middle';

  ctx.textAlign = 'center';
  if (source && !wrap) ctx.fillText(clipText(`Source #${source.id}`, cellW), headW + gap + cellW / 2, headH / 2);
  colValues.forEach((v, c) => {
    if (!colAxis || wrap) return;
    const x = headW + gap + (lead + c) * (cellW + gap) + cellW / 2;
    ctx.fillText(clipText(headText(colAxis, v), cellW), x, headH / 2);
  });
  ctx.textAlign = 'right';
  rowValues.forEach((v, r) => {
    if (!rowAxis) return;
    const y = headH + gap + r * (cellH + gap) + cellH / 2;
    ctx.fillText(clipText(headText(rowAxis, v), headW - 12), headW - 6, y);
  });

  // Every cell's thumbnail in flight at once; a hole stays a hole. The
  // captions under wrapped cells are drawn here, before any image lands.
  const jobs = [];
  for (const item of items) {
    const x = headW + gap + item.col * (cellW + gap);
    const y = item.span
      ? headH + gap + (rows * (cellH + gap) - gap - cellH) / 2
      : headH + gap + item.row * (cellH + labelH + gap);
    ctx.fillStyle = '#1a1a21';
    ctx.fillRect(x, y, cellW, cellH);
    if (item.label) {
      ctx.fillStyle = '#e8e8ee';
      ctx.textAlign = 'center';
      ctx.fillText(clipText(item.label, cellW), x + cellW / 2, y + cellH + labelH / 2);
    }
    const { rec } = item;
    if (!rec) continue;
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
  }
  await Promise.all(jobs);

  ctx.fillStyle = '#8b8b9a';
  ctx.textAlign = 'left';
  lines.forEach((line, i) => ctx.fillText(line, gap, gridH + 16 + i * 20));
  return canvas;
}

// Named by the run and the pins: every slice of a run used to download as
// sheet-<run>.png, told apart only by the browser's " (1)".
function sheetFileName() {
  const first = sheetRecords[0];
  const pins = (sheetLayout?.pins ?? [])
    .map(([axis, v]) => `${axis.key.replace(/^var:/, '')}-${clip(sheetText(axis, v))}`);
  const name = `sheet-${first?.run_id ?? 'gallery'}${pins.length ? `-${pins.join('_')}` : ''}`;
  return `${name.replace(/[^\w.-]+/g, '-')}.png`;
}

async function exportSheet() {
  const name = sheetFileName();
  const canvas = await buildSheetCanvas();
  if (!canvas) return;
  canvas.toBlob((blob) => {
    if (!blob) { showError('The browser could not encode the sheet.', false, { slot: els.galleryError }); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }, 'image/png');
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

function setSheet(on) {
  sheetMode = on;
  els.sheetBtn.classList.toggle('active', on);
  els.sheet.hidden = !on;
  els.gallery.hidden = on;
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
  return resetGallery();
}
