// Local Diffusion's page script: grid requests and their preview.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------- grid search

// The axes sit on top of the panel's own settings, so a blank one contributes
// the value already there. `runs` is not a grid axis - the seed count is - and
// the server refuses a grid that sets both.
function gridPayload() {
  return {
    ...collect(),
    runs: 1,
    prompt_axis: els.gridPrompt.value.trim(),
    prompt_mode: els.gridPromptMode.value,
    steps_axis: els.gridSteps.value.trim(),
    size_axis: els.gridSize.value.trim(),
    cfg_axis: els.gridCfg.value.trim(),
    mu_axis: els.gridMu.disabled ? '' : els.gridMu.value.trim(),
    lora_scale_axis: els.gridLoraScale.disabled ? '' : els.gridLoraScale.value.trim(),
    seeds: +els.gridSeeds.value || 1,
    seed_axis: els.gridSeedList.value.trim(),
    resume: els.gridResume.checked,
    budget_minutes: els.gridBudget.value.trim() === '' ? null : +els.gridBudget.value,
    then_refine: thenRefinePayload(),
  };
}

function humanTime(seconds) {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${minutes} min`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

const axisText = (values) => values.map((v) => (v === null ? 'none' : v)).join(', ');
// Mu's none is the scheduler's own shift, which on a fixed-shift checkpoint
// is a number: said, as the sheet's heads say it.
const muAxisText = (values) => {
  const fixed = muStatic();
  return values.map((v) => (v !== null ? v : fixed === null ? 'none' : `none (${muStaticText(fixed)})`)).join(', ');
};

// The server's summary names every axis, the ones of one value included, and
// multiplies with an ASCII x: it is the CLI's log line too (grid.describe, held
// to that by its tests). The axes line under it lists every value anyway, so
// here a factor of one goes - all but the seeds, the count a night is planned
// around - and the sign is the comparison preview's ×.
function gridFactors(summary) {
  const kept = summary.split(' x ').filter((factor) => !factor.startsWith('1 ') || /seeds?$/.test(factor));
  return kept.join(' × ') || summary;
}

// The prompts a variable block resolved to, a few of them. Substitution is the
// half of a prompt grid that goes wrong quietly - a value that reads fine in the
// box lands badly in the middle of a sentence - so the panel shows the finished
// text rather than the variables that produced it. One prompt too, whenever the
// variables box has text: a block that failed to vary - `size = big, small` -
// is a grid of one prompt, and that prompt is where it shows.
// A prompt written over lines - NewBie's XML, 29 of them - is listed by the
// lines a {variable} is on, whole, so each value is read in its line: three
// copies of the whole prompt, one word apart, grew the preview to 2,260 px at
// QHD and put Run grid 2,200 px under the window. Its line numbers say the
// rest of the prompt is there. Values are one line each, and a dropped one
// leaves its line in place (grid.substitute), so line n of every resolved
// prompt is line n of the prompt typed; a prompt that is not - no line holds
// a variable, the variables are the negative's - is listed whole. Whatever
// the listing, it is a box of a few lines that scrolls inside itself
// (style.css): a nested scroller, never the panel, whose menus and tips hang
// out past its edge.
const PREVIEW_PROMPTS = 3;
function promptListing(prompts, template, name = 'prompt') {
  if (!prompts || !prompts.length) return null;
  const shown = prompts.slice(0, PREVIEW_PROMPTS);
  const lines = template.split('\n');
  const marked = lines.flatMap((line, i) => (GRID_PLACEHOLDER.test(line) ? [i] : []));
  const byLine = lines.length > 1 && marked.length > 0 && marked.length < lines.length
    && shown.every((text) => text.split('\n').length === lines.length);
  let rows;
  if (byLine) {
    const where = `${marked.length === 1 ? 'line' : 'lines'} ${marked.map((i) => i + 1).join(', ')} of ${lines.length}`;
    // Continuation lines are indented with no-break spaces, which pre-line
    // keeps; each line's own indentation is the prompt's, not the listing's.
    rows = [`${name}${prompts.length === 1 ? '' : 's'}, ${where}:`, ...shown.flatMap((text) => {
      const own = text.split('\n');
      return marked.map((i, k) => `${k ? '  ' : '· '}${own[i].trim()}`);
    })];
  } else {
    rows = prompts.length === 1 ? [`${name}: ${prompts[0]}`] : shown.map((text) => `· ${text}`);
    if (prompts.length > 1 && name !== 'prompt') rows.unshift(`${name}s:`);
  }
  if (prompts.length > PREVIEW_PROMPTS) rows.push(`· … and ${prompts.length - PREVIEW_PROMPTS} more`);
  const box = document.createElement('span');
  box.className = 'prompts';
  box.textContent = rows.join('\n');
  return box;
}

// A variable of one value never varies, which in a grid is always a slip - and
// most often commas, where every other box in the fold splits on them. Read
// off the box the way grid.parse_variables reads it (the first `=`, values
// split on `|`, blanks and repeats dropped), but only to warn: the server's
// parse is the one the run uses, and its errors are the ones shown.
function singleValueLines(text) {
  const lines = [];
  for (const line of text.split('\n')) {
    const match = /^\s*([A-Za-z_]\w*)\s*=(.*)$/.exec(line);
    if (!match) continue;
    const values = new Set(match[2].split('|').map((v) => v.trim().split(/\s+/).join(' ')).filter(Boolean));
    if (values.size !== 1) continue;
    const [value] = values;
    lines.push(value.includes(',')
      ? `${match[1]} has one value '${value}' - values split on |, not commas`
      : `${match[1]} has one value: it never varies`);
  }
  return lines;
}

// The names the open fold's variables box declares, read as
// grid.parse_variables reads a line (a name, then `=`): what makes a `{name}`
// in a prompt the grid's rather than text. Empty with the fold shut, as
// Generate's guard reads it. Only to refuse a press the grid's own parse
// would not fill (the comparison's) and to name a chip (the tag tray).
function declaredGridVariables() {
  if (!els.gridSection.open) return new Map();
  const declared = new Map();
  for (const line of els.gridPrompt.value.split('\n')) {
    const match = /^\s*([A-Za-z_]\w*)\s*=(.*)$/.exec(line);
    if (match && !declared.has(match[1])) {
      declared.set(match[1], match[2].split('|').map((v) => v.trim().split(/\s+/).join(' ')).filter(Boolean));
    }
  }
  return declared;
}

// The preview is the server's own expansion rather than a count worked out in
// here: the syntax has one parser, and the message a bad axis produces is the
// one the run itself would have refused with.
// Preview replies can overtake each other across the server's one await (the
// duration lookup), so a slow reply for the boxes as they *were* could land
// after the reply for the boxes as they are. Same token the gallery uses.
let previewEpoch = 0;

// Opening the fold by hand brings Run grid into view on the next frame
// (revealGridRun), and the preview's reply lands after that, under the same
// pointer, where the panel holds still (panelResized): its lines pushed Run
// grid back below the window the opening had just brought it into. The
// summary's click sets this, and the first preview shown after it reveals
// Run grid again, as part of the same opening.
let gridRevealPending = false;
function previewShown() {
  if (!gridRevealPending) return;
  gridRevealPending = false;
  revealGridRun();
}

async function previewGrid() {
  if (modelChangeHold()) { ++previewEpoch; return; }
  if (!els.gridSection.open) {
    gridRevealPending = false;
    if (gridMaxCfg !== null) { gridMaxCfg = null; syncCfg(); }
    return;
  }
  const epoch = ++previewEpoch;
  // Nothing to preview before there is something to render: the server needs a
  // prompt to validate the request at all, and an untouched panel is not a
  // mistake to be reported.
  if (!els.prompt.value.trim()) {
    els.gridPreview.classList.remove('warn');
    els.gridPreview.textContent = 'Enter a prompt above to see what this grid would run.';
    previewShown();
    return;
  }
  const body = gridPayload();
  try {
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (epoch !== previewEpoch) return;  // superseded while in flight
    els.gridPreview.classList.remove('warn');
    // The axis decides the negative prompt's reach - see syncCfg.
    const reach = Math.max(...p.values.guidance_scale.filter((v) => typeof v === 'number'));
    gridMaxCfg = Number.isFinite(reach) ? reach : null;
    syncCfg();
    // Totals on the first line, the resolved axes on the second: what a range
    // actually expanded to is the half of it that is easy to get wrong.
    // The disk is the other thing a night runs out of; a size estimate past
    // the free space is the warning colour, since the night would die at the
    // first save that does not fit.
    const short = p.bytes && diskFreeMb !== null && p.bytes > diskFreeMb * 1024 * 1024;
    // Under the prompts, what would make the night not the one planned: cells
    // the gallery has already (Resume skips them; Run grid as it stands draws
    // them again), a variable that never varies, a budget that stops short,
    // and the panel's Repeat, which a grid does not take.
    const notes = [];
    if (body.prompt_mode === 'random') {
      notes.push(p.prompt_sample
        ? 'Prompts shown are examples; Run grid draws a new seed and choices. Images run separately.'
        : 'Prompt choices are fixed by each image seed. Images run separately.');
    }
    if (p.in_gallery) {
      // Random mode has already split Batch into one-image cells.
      const n = p.in_gallery * (body.prompt_mode === 'random' ? 1 : body.num_images);
      notes.push(`${n} of these ${n === 1 ? 'is' : 'are'} already in the gallery - tick Resume to skip them`);
    }
    const single = body.prompt_axis ? singleValueLines(body.prompt_axis) : [];
    notes.push(...single);
    // The seeds the budget completes, the server's count by the job's own
    // rule (`budget_seeds`): a cell starts while the deadline has not passed
    // and runs to its end, so the seed whose last cell starts in time
    // completes past the deadline - the budget over a seed's share of the
    // estimate said 4 of 6 for a grid that made 5. There is no line without
    // an estimate. The budget counts from Run grid, so the load and any job
    // ahead of it in the queue spend it too, which the estimate does not
    // count: said while the topbar's dot (the status's state) is not ready,
    // and while this page watches a job or another tab's runs
    // (elsewhereBusy). Not under Resume, whose seeds are no longer alike.
    let budgetShort = false;
    if (body.budget_minutes && p.budget_seeds != null && !body.resume) {
      const seeds = p.values.seed?.length || body.seeds;
      const done = p.budget_seeds;
      budgetShort = done === 0;
      const ahead = (busy || elsewhereBusy ? 1 : 0) + queuedIds.length;
      const also = [
        els.dot.classList.contains('ready') ? '' : 'the load',
        ahead === 1 ? 'the job ahead' : ahead ? `the ${ahead} jobs ahead` : '',
      ].filter(Boolean);
      notes.push(`budget ${body.budget_minutes} min ≈ ${done} of ${seeds} seed${seeds === 1 ? '' : 's'} complete` +
        (also.length ? ` (${also.join(' and ')} count${also.length > 1 || ahead > 1 ? '' : 's'} too)` : ''));
    }
    if (+els.runs.value > 1) notes.push(`Repeat (${+els.runs.value}) does not apply to a grid: Seeds is its count`);
    // Cells that would come out identical - an axis this checkpoint ignores -
    // are the other reason to fix the grid before the night rather than after.
    if (short || p.duplicates || p.in_gallery || single.length || budgetShort) {
      els.gridPreview.classList.add('warn');
    }
    const totals = document.createElement('span');
    totals.className = 'totals';
    totals.textContent =
      `${p.images} image${p.images === 1 ? '' : 's'} · ${gridFactors(p.summary)}` +
        // Then refine's passes, one per image, priced in the time and disk.
        (p.passes ? ` · + ${p.passes} refine pass${p.passes === 1 ? '' : 'es'} at ${(p.pass_sizes?.length ? p.pass_sizes : [[p.pass_width, p.pass_height]]).map(([w, h]) => `${w}×${h}`).join(", ")}` : '') +
        (p.skipped ? ` · ${p.skipped} already done` : '') +
        // Resume's other drop: cells a live job already plans, the same run
        // still queued - not "done", which is the gallery's.
        (p.queued ? ` · ${p.queued} queued` : '') +
        (p.duplicates ? ` · ${p.duplicates} would be identical to another cell` : '') +
        (p.seconds ? ` · about ${humanTime(p.seconds)}` : '') +
        (p.bytes ? ` · ~${humanBytes(p.bytes)}${short ? ` — more than the ${humanBytes(diskFreeMb * 1024 * 1024)} free` : ''}` : '');
    // .hint.pre renders the breaks. The prompts are a box of their own
    // between the axes and the notes (promptListing), on its own line.
    const listing = body.prompt_axis ? promptListing(p.values.prompt, body.prompt) : null;
    // Random draws can vary only the negative, or both boxes together. Show
    // that resolved text too; it otherwise disappears behind a constant prompt.
    const negatives = body.prompt_mode === 'random' && GRID_PLACEHOLDER.test(body.negative_prompt ?? '')
      ? promptListing(p.values.negative_prompt.map((text) => text ?? ''), body.negative_prompt, 'negative prompt')
      : null;
    els.gridPreview.replaceChildren(
      totals,
      `\nsteps ${axisText(p.values.steps)}  ·  CFG ${axisText(p.values.guidance_scale)}` +
        `  ·  mu ${muAxisText(p.values.mu)}` +
        (body.size_axis ? `  \u00b7  size ${axisText(p.values.size)}` : '') +
        // Only when it is an axis: every other run would carry a constant.
        // Named, since a sweep's weights mean nothing without the adapter.
        (p.values.lora_scale && p.values.lora_scale.length > 1
          ? `  ·  LoRA ${axisText(p.values.lora_scale)} (${els.lora.value.replace(/\.safetensors$/, '')})` : '') +
        (p.values.seed && p.values.seed.length ? `  ·  seeds ${axisText(p.values.seed)}` : ''),
      ...(listing ? ['\n', listing] : []),
      ...(negatives ? ['\n', negatives] : []),
      ...(notes.length ? [`\n${notes.join('\n')}`] : []),
    );
  } catch (err) {
    if (epoch !== previewEpoch) return;
    els.gridPreview.classList.add('warn');
    els.gridPreview.textContent = err.message;
  }
  previewShown();
}

// The prompt check's comparison is a grid on the same panel, so whatever moves
// this preview's totals - the size, the steps, the seed - moves its too.
let gridTimer;
function schedulePreview() {
  // Then refine's fold says the pass size of the size now in the panel.
  syncThenRefine();
  clearTimeout(gridTimer);
  gridTimer = setTimeout(previewGrid, 250);
  scheduleComparePreview();
}
