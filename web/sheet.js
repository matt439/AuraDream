// Local Diffusion's page script: contact-sheet axes, labels and controls.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------ contact sheet

// Everything a grid can vary. `of` reads one off a record; mu's null is a real
// value - "no shift of ours" - not a missing one, so it prints as `none`, with
// the shift it left where that is a fixed number (sheetMuStatic). The
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
// list, which leaves True CFG out where it echoes CFG. `sameExcept` is the
// same test with some axes left out by key - the lightbox's CFG ladder
// (lightbox-cfg.js) skips `cfg` and `tcfg` - so the two cannot drift.
const sameExcept = (a, b, skip = []) => SHEET_AXES
  .every((axis) => skip.includes(axis.key) || axisValue(axis, a) === axisValue(axis, b))
  && ['width', 'height', 'model_id', 'quantization', 'refine'].every((key) => a[key] === b[key]);
const sameSettings = (a, b) => sameExcept(a, b);

// Full prompt -> the words that make it different from its neighbours, filled
// in by sheetValues. Only the prompt axis is in here, so anything else that is
// a string prints as itself. Whole: a heading wraps, three lines at most with
// its full text as the title, and the 44 characters `clip` keeps are for a
// line of fixed width - a pin's option, the note, the export's canvas. A
// 540 px column used to show 44 characters across half its width.
let promptShort = new Map();
// The words the sheet's text is made of - the prompts' short labels and the
// adapter (sheetAdapter, below) - as they stand now. An export takes them
// with its layout (exportLayout): Export all writes its slices one after
// another, and a sheet redrawn on another run meanwhile replaces both.
const sheetWords = () => ({ short: promptShort, adapter: sheetAdapter, muStatic: sheetMuStatic });
const valueText = (v, words = sheetWords()) => {
  if (v === null) return 'none';
  if (typeof v === 'string') return words.short.get(v) ?? v;
  return String(v);
};

const LABEL_CHARS = 44;
const clip = (text) => (text.length > LABEL_CHARS ? `${text.slice(0, LABEL_CHARS - 1)}…` : text);

// The adapter every LoRA weight on the sheet belongs to, while there is only
// one: a weight sweep's heads and its export then say whose weights they are,
// and sheets of several adapters' sweeps can be told apart afterwards.
let sheetAdapter = null;

// The shift Mu's none ran at, where it is a number: every image on the sheet
// the running checkpoint's, whose scheduler's shift is fixed (`muStatic`:
// Z-Anime's, NetaYume's and NewBie's mu 1.79). "Mu 1, none, 2.5" laid out as
// "Mu none | Mu 1 | Mu 2.5" there, the 1.79 column left of 1 with nothing to
// say so; it reads "Mu none (1.79)" and sorts at 1.79. Null elsewhere - FLUX
// derives its shift from the size - where none is no number and sorts first.
let sheetMuStatic = null;

// A value as its axis reads it - a version by its name - and a heading: the
// axis and the value, a known version by its name alone ("Rewrite", not
// "Version rewrite"), a weight with its adapter.
function sheetText(axis, v, words = sheetWords()) {
  if (axis.key === VARIANT_AXIS && v !== null) return VERSION_NAMES.get(v) ?? v;
  if (axis.key === 'mu' && v === null && words.muStatic != null) return `none (${muStaticText(words.muStatic)})`;
  return valueText(v, words);
}
function headText(axis, v, words = sheetWords()) {
  if (axis.key === VARIANT_AXIS && VERSION_NAMES.has(v)) return VERSION_NAMES.get(v);
  if (axis.key === 'lora' && words.adapter && v !== null) return `LoRA ${words.adapter} ${v}`;
  return `${axis.label} ${sheetText(axis, v, words)}`;
}

// What one prompt changed against another, a run of edits at a time:
// `−dawn +dusk`, `+35mm photo,`. From promptChanges (analysis-critic.js), the exact
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
  // Whole XML blocks in another order are named by their tags: as words,
  // NewBie's <character_2> moved first read "character_2 … kasane_teto …
  // pink_hair …", the markup's names and every tag inside it.
  const blocks = blockMove(original, prompt);
  if (blocks) return blocks.move ? `same blocks, moved: ${blocks.move}` : tagLine('same blocks, in the order', blocks.order);
  const words = (text) => text.match(/[\p{L}\p{N}]+/gu) ?? [];
  const [before, after] = [words(original), words(prompt)];
  const sorted = (list) => [...list].sort().join(' ');
  if (before.join(' ') === after.join(' ') || sorted(before) !== sorted(after)) return promptEdit(original, prompt);
  const moved = editRuns(original, prompt)
    .map(({ added }) => added.replace(/^[\s,;:.]+|[\s,;:.]+$/g, ''))
    .filter(Boolean);
  return `same words, moved: ${moved.join(' … ')}`;
}

// Tags after `lead` in at most EDIT_CHARS characters, cut between two tags
// with an ellipsis: the row head's line was cut at its 80th character, and
// four blocks' order ended inside a tag, "<character_" read as a block of
// its own. The row head's title is the whole prompt.
const EDIT_CHARS = 80;
function tagLine(lead, tags) {
  let line = lead;
  for (const [i, tag] of tags.entries()) {
    const more = i < tags.length - 1 ? ' …'.length : 0;
    if (line.length + 1 + tag.length + more > EDIT_CHARS) return `${line} …`;
    line += ` ${tag}`;
  }
  return line;
}

// A prompt's top-level XML blocks, as the critic finds them: critic.py's
// _XML_BLOCK, Python's Unicode \w written out (JavaScript's \w is ASCII
// alone), held equal by tests/test_critic.py. Each block's tag and its text,
// its spaces collapsed, and `gaps`, the text before, between and after them,
// each in its place: loose text that moves past a block is an edit too.
const XML_BLOCK = /<([A-Za-z_][\p{L}\p{N}_.-]*)>[\s\S]*?<\/\1>/gu;
function xmlBlocks(text) {
  const blocks = [];
  const gaps = [];
  const squeeze = (t) => t.replace(/\s+/g, ' ').trim();
  let at = 0;
  for (const m of text.matchAll(XML_BLOCK)) {
    gaps.push(squeeze(text.slice(at, m.index)));
    blocks.push({ tag: `<${m[1]}>`, text: squeeze(m[0]) });
    at = m.index + m[0].length;
  }
  gaps.push(squeeze(text.slice(at)));
  return { blocks, gaps };
}

// Where `prompt` is `original`'s top-level XML blocks in another order and
// nothing else - not a word in or between them changed - what moved, by the
// blocks' tags: one block taken out and put back elsewhere is `move`
// ("<general_tags> first", "<character_2> before <character_1>",
// "<character_1> last"), named as the one that went earlier where either of
// two could be; anything else is the new `order`, a list of the tags. Null
// for any other edit, which reads as words.
function blockMove(original, prompt) {
  const a = xmlBlocks(original), b = xmlBlocks(prompt);
  if (a.blocks.length < 2 || a.blocks.length !== b.blocks.length || a.gaps.join('\n') !== b.gaps.join('\n')) return null;
  const was = a.blocks.map((x) => x.text), now = b.blocks.map((x) => x.text);
  const same = (x, y) => x.join('\n') === y.join('\n');
  if (same(was, now) || !same([...was].sort(), [...now].sort())) return null;
  const without = (list, k) => list.filter((_, j) => j !== k);
  const tag = (i) => b.blocks[i].tag;
  for (let i = 0; i < now.length; i++) {
    const from = was.indexOf(now[i]);
    if (from > i && same(without(was, from), without(now, i))) {
      return { move: i === 0 ? `${tag(i)} first` : `${tag(i)} before ${tag(i + 1)}` };
    }
  }
  for (let i = now.length - 1; i >= 0; i--) {
    const from = was.indexOf(now[i]);
    if (from < i && same(without(was, from), without(now, i))) {
      return { move: i === now.length - 1 ? `${tag(i)} last` : `${tag(i)} after ${tag(i - 1)}` };
    }
  }
  return { order: b.blocks.map((x) => x.tag) };
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
  // XML prompts whose blocks only moved are labelled by the blocks' tags
  // (blockMove) - the oldest by its blocks' order where every other one is -
  // where the edit's words were the markup's names and every tag moved.
  const moves = prompts.map((p, i) => (i === 0 ? null : blockMove(prompts[0], p)));
  const blockLabel = (m) => (m.move ? m.move : `blocks ${m.order.join(' ')}`);
  const first = moves.slice(1).every(Boolean)
    ? `blocks ${xmlBlocks(prompts[0]).blocks.map((x) => x.tag).join(' ')}` : prompts[0].trim();
  return prompts.map((p, i) => {
    if (i === 0) return first;
    return moves[i] ? blockLabel(moves[i]) : promptEdit(prompts[0], p) || p.trim();
  });
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
  const fixedMu = muStatic();
  sheetMuStatic = fixedMu !== null && records.length && records.every((r) => r.model_id === loadedModel)
    ? fixedMu : null;
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
    // Mu's none at the shift it ran at, as its head prints it, ahead of an
    // explicit value equal to that: 1.79 typed from the placeholder sits
    // after "none (1.79)", where ln 6's 1.7918 put it before.
    const noneAt = axis.key === 'mu' && sheetMuStatic !== null ? +muStaticText(sheetMuStatic) : null;
    seen.sort((a, b) => {
      if (noneAt !== null) return ((a ?? noneAt) - (b ?? noneAt)) || (a === null ? -1 : b === null ? 1 : 0);
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
// where the label is the words that vary. A number's label (`numeric`: the
// value is a number - a seed, steps, a weight) breaks only between groups of
// four digits (`numberLabel`).
function headCell(text, className, title = text, numeric = false) {
  const div = document.createElement('div');
  div.className = className || 'head';
  const label = document.createElement('span');
  label.className = 'label';
  if (numeric) label.append(...numberLabel(text));
  else label.textContent = text;
  div.appendChild(label);
  if (title) div.title = title;
  return div;
}

// A head's 16-digit seed, broken by the label's `overflow-wrap: break-word`
// wherever a column was under 112 px - an 8-seed comparison at 1400x900 read
// "Seed 760152042044950" over "9", a number misread as another when typed
// back - goes to a line of its own whole (`.number`, an inline-block), and
// breaks only where the number alone is wider than its column, and then only
// between groups of four digits counted from its end: a <wbr> between them
// is a place to break, so a tail is never one to three digits. <wbr>s alone
// filled the first line after "Seed" and split numbers that had gone onto a
// line of their own whole, "Seed 760152042044" over "9509" in 112-143 px.
// The text is the number whole, so a copy has every digit.
function numberLabel(text) {
  return text.split(/(\d{5,})/).flatMap((part, i) => {
    if (i % 2 === 0) return part ? [part] : [];
    const number = document.createElement('span');
    number.className = 'number';
    part.match(/\d{1,4}(?=(?:\d{4})*$)/g).forEach((group, j) => {
      if (j) number.append(document.createElement('wbr'));
      number.append(group);
    });
    return [number];
  });
}

// What a sheet is, in words: the table's name ("Sheet: Steps × Seed"), and,
// with its count of images, what is said as it opens (`saySheet`).
function sheetSummary(wrap, colAxis, rowAxis, count, source = null) {
  const axes = [colAxis, rowAxis].filter(Boolean).map((a) => a.label).join(' × ');
  const parts = [
    count == null ? '' : `${count} image${count === 1 ? '' : 's'}`,
    axes && (wrap ? `by ${axes}` : axes),
    source ? `beside source #${source.id}` : '',
  ].filter(Boolean);
  return parts.length ? `Sheet: ${parts.join(', ')}` : 'Sheet';
}

// Said as a sheet opens, from the page or the Sheet button (setSheet): the
// page scrolled to it and nothing said what had happened. Written on the
// next frame after an empty line, so the same sheet opened again is said
// again; a sheet with nothing laid out says its note.
function saySheet() {
  const text = sheetLayout
    ? sheetSummary(sheetLayout.wrap, sheetLayout.colAxis, sheetLayout.rowAxis, sheetLayout.shown, sheetLayout.source)
    : `Sheet: ${els.sheetNote.textContent}`;
  els.sheetSaid.textContent = '';
  requestAnimationFrame(() => { if (sheetMode) els.sheetSaid.textContent = text; });
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
const thumbWidthOf = (rec) => Math.round(rec.width * Math.min(1, THUMB_SIDE / Math.max(rec.width, rec.height)));

// Settings nothing on the sheet has - no adapter, no references, no pass -
// are left out of the Fixed line rather than listed as `none`.
const ABSENT_WHEN_NONE = new Set(['tcfg', 'lora', 'adapter', 'ref', 'strength']);

// The settings every image on the sheet shares. Under the Starred tick they
// are what made the keepers, and the tick takes them off every axis: two
// starred images read as two cells, with nothing saying they shared a seed,
// steps and CFG. A line of its own rather than part of the note.
function fixedSettings(values, axes = sheetAxesList, words = sheetWords()) {
  const fixed = [];
  for (const axis of axes) {
    if (values[axis.key].length !== 1 || axis.key === 'prompt' || axis.key === 'negative') continue;
    const [v] = values[axis.key];
    if (v === null && (ABSENT_WHEN_NONE.has(axis.key) || axis.key.startsWith('var:'))) continue;
    fixed.push(`${axis.label} ${sheetText(axis, v, words)}`);
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

// Fit or Fill. Fitted, a matrix's cells are no taller than its rows can share
// of the window (--cell-max), so a 4 x 3 is one screen; Fill gives them the
// sheet's whole width (1fr) and lets the box they scroll in scroll, for a
// sheet read cell by cell rather than at a glance. A preference, remembered
// per browser (`remembered`/`remember` hold every access in try/catch, and
// without storage the sheet is fitted) and switched by the stylesheet's
// .sheet.fill-width alone (not `.fill`, the progress bar's), so the toggle
// redraws nothing; fitSheet's `sizes` and the export follow it. A wrapped row
// already takes the width.
const SHEET_KEY = 'local-diffusion.sheet.v1';
let sheetFillOn = null;
function sheetFill() {
  if (sheetFillOn === null) {
    let stored = null;
    try { stored = JSON.parse(remembered(SHEET_KEY) || 'null'); } catch { stored = null; }
    sheetFillOn = stored?.cells === 'fill';
  }
  return sheetFillOn;
}

const FILL_WRAPPED = 'A wrapped row already fills the width: Fill is for a matrix';
function syncFillButton(button) {
  const on = sheetFill();
  const wrapped = els.sheet.classList.contains('wrapped');
  button.setAttribute('aria-pressed', String(on));
  button.classList.toggle('active', on);
  button.setAttribute('aria-disabled', String(wrapped));
  button.title = wrapped ? FILL_WRAPPED
    : on ? 'Cells fill the width, and the sheet scrolls - press to fit the sheet to the window'
      : 'Cells fit the window - press to let them fill the width';
}

function setSheetFill(on) {
  sheetFillOn = on;
  remember(SHEET_KEY, JSON.stringify({ cells: on ? 'fill' : 'fit' }));
  els.sheet.classList.toggle('fill-width', on);
  const button = els.sheetControls.querySelector('.sheet-fill');
  if (button) syncFillButton(button);
  if (sheetMode && sheetLayout) fitSheet();
}

function fillButton() {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost small sheet-fill';
  button.textContent = 'Fill width';
  button.onclick = () => {
    if (button.getAttribute('aria-disabled') === 'true') {
      showError(`${FILL_WRAPPED}.`, true, { slot: els.galleryError });
      return;
    }
    setSheetFill(!sheetFill());
  };
  syncFillButton(button);
  return button;
}
