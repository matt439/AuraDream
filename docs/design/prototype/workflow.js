'use strict';

// Local interaction fixtures only. No application API, weights or gallery.
const studyJobs = [
  {id: 'J12', label: 'Refine keepers', state: 'Running', progress: '12/28 steps · 18s elapsed · ~24s left'},
  {id: 'J13', label: 'More of Run A', state: 'Queued', progress: 'Waiting for J12'},
];

function updateQueueCount() {
  byId('queue-count').textContent = studyJobs.filter(job => job.state !== 'Cancelled').length + sampleQueue.length;
}

function syncWatchedJob() {
  const job = studyJobs.find(item => item.id === byId('watch-job').value) || studyJobs[0];
  byId('job-readout').textContent = `${job.state} · ${job.progress} · sample`;
  const held = job.state === 'Cancelled';
  byId('cancel-job').setAttribute('aria-disabled', String(held));
  byId('cancel-job').title = held ? 'This sample job is already cancelled' : `Cancel ${job.id}: ${job.label}`;
  document.querySelector('.jobbar .progress span').style.width = job.state === 'Running' ? '40%' : '0%';
  updateQueueCount();
}

function cancelStudyJob(job) {
  if (job.state === 'Cancelled') return;
  job.state = 'Cancelled';
  job.progress = 'Saved results retained';
  syncWatchedJob();
  byId('workspace-note').textContent = `Sample ${job.id} cancelled. The draft and saved sample images are retained.`;
}

function openQueueStudy() {
  information('Simulated queue', ['Two independent job fixtures demonstrate watch and cancel. New sample requests are recorded below; no generation runs.']);
  for (const job of studyJobs) {
    const row = document.createElement('div'); row.className = 'job-entry';
    const label = document.createElement('p'); label.textContent = `${job.id} · ${job.label} · ${job.state}`;
    const actions = document.createElement('div'); actions.className = 'row';
    const watch = document.createElement('button'); watch.textContent = 'Watch';
    watch.addEventListener('click', () => {byId('watch-job').value = job.id; syncWatchedJob(); closeDialog('info-dialog');});
    const cancel = document.createElement('button'); cancel.textContent = 'Cancel'; cancel.dataset.cancelJob = job.id;
    cancel.setAttribute('aria-disabled', String(job.state === 'Cancelled'));
    cancel.addEventListener('click', () => {cancelStudyJob(job); label.textContent = `${job.id} · ${job.label} · ${job.state}`; cancel.setAttribute('aria-disabled', 'true');});
    actions.append(watch, cancel); row.append(label, actions); byId('info-body').append(row);
  }
  for (const text of sampleQueue) {const row = document.createElement('p'); row.className = 'demo-queue'; row.textContent = text; byId('info-body').append(row);}
  const more = document.createElement('details');
  const title = document.createElement('summary'); title.textContent = 'Secondary job controls';
  const note = document.createElement('p'); note.textContent = 'Pause/Resume remains available in the application. Any paused state must still be visible; reconnect and retries retain their existing rules.';
  more.append(title, note); byId('info-body').append(more);
}

function sampleScope() {
  const images = picks.size ? sampleImages.filter(image => picks.has(image.id)) : visibleImages();
  return {images, label: picks.size ? `${picks.size} picked images, including hidden picks` : `${images.length} images under the current filter`};
}

function openScopeStudy(action) {
  const scope = sampleScope();
  if (!scope.images.length) {byId('gallery-note').textContent = 'No images in this scope. Pick images or change the filter.'; return;}
  const explanation = {
    'Re-render': `Use each image's recorded settings at the draft size, ${byId('dimensions').textContent}; preserve seeds and preview exclusions.`,
    'Swap tags': 'Find/replace or remove tags in each recorded prompt or negative; preserve its seed and settings. Preview every included/excluded image before Queue.',
    'Refine': 'One scale and strength across the scope, keeping each image as its own source. Other checkpoints and existing passes remain opt-in secondary choices.',
    'Bin picked': 'Move the explicit picked set to the recycle bin and offer Restore; permanent removal remains a separate confirmed action.',
  };
  information(`${action} · ${scope.label}`, [explanation[action], 'This control demonstrates scope and placement. The real application provides the server preview, eligibility and execution; this concept does not run that planner or change files.']);
}

function renderSampleSheet() {
  const images = visibleImages();
  const matrix = byId('gallery-matrix');
  const table = document.createElement('table'); table.className = 'sample-matrix';
  const caption = document.createElement('caption');
  caption.textContent = 'Sample settings matrix · gallery query scope, independent of picks';
  const cfgValues = [...new Set(images.map(image => image.cfg))].sort((a, b) => a - b);
  const rows = [...new Set(images.map(image => `${image.model} · ${image.steps} steps`))];
  const head = document.createElement('thead'), headRow = document.createElement('tr');
  const corner = document.createElement('th'); corner.textContent = 'Model / Steps'; headRow.append(corner);
  for (const cfg of cfgValues) {const th = document.createElement('th'); th.scope = 'col'; th.textContent = `CFG ${cfg}`; headRow.append(th);}
  head.append(headRow);
  const body = document.createElement('tbody');
  for (const name of rows) {
    const row = document.createElement('tr');
    const title = document.createElement('th'); title.scope = 'row'; title.textContent = name; row.append(title);
    for (const cfg of cfgValues) {
      const cell = document.createElement('td');
      const matches = images.filter(image => `${image.model} · ${image.steps} steps` === name && image.cfg === cfg);
      for (const image of matches) {
        const button = document.createElement('button'); button.setAttribute('aria-label', `Inspect sheet image ${image.id}, seed ${image.seed}`);
        const art = document.createElement('div'); art.className = `art ${artClass(image)}`;
        const stamp = document.createElement('span'); stamp.textContent = `#${image.id} · Seed ${image.seed}`; art.append(stamp); button.append(art);
        button.addEventListener('click', () => inspect(image)); cell.append(button);
      }
      if (!matches.length) cell.textContent = '—';
      row.append(cell);
    }
    body.append(row);
  }
  table.append(caption, head, body); matrix.replaceChildren(table);
}

function openArchiveStudy() {
  information('Gallery bundles · entire gallery', ['Whole-gallery archiving covers every checkpoint and the bin, independently of the current filter and picks. These are design controls; no files are read, cleared or exported.']);
  const password = document.createElement('label'); password.className = 'subtle-field';
  const protect = document.createElement('input'); protect.type = 'checkbox'; protect.checked = true;
  password.append(protect, document.createTextNode('Password protection · sample choice'));
  byId('info-body').append(password);
  const actions = document.createElement('div'); actions.className = 'archive-actions';
  for (const action of ['Back up gallery', 'Bundle and clear gallery', 'Download saved bundle', 'Import bundle']) {
    const button = document.createElement('button'); button.textContent = action;
    button.addEventListener('click', () => {byId('archive-result').textContent = `${action}: design preview only. The real app verifies an archive before cleanup; passwords, counts and import eligibility are resolved there.`;});
    actions.append(button);
  }
  byId('info-body').append(actions);
  const result = document.createElement('p'); result.id = 'archive-result'; result.className = 'notice'; result.setAttribute('role', 'status');
  const other = document.createElement('details');
  const label = document.createElement('summary'); label.textContent = 'Other bundle scopes';
  const text = document.createElement('p'); text.textContent = 'Picked and Starred remain available with their existing inclusion rules. They are secondary in this personal layout.';
  other.append(label, text); byId('info-body').append(result, other);
}

function openModelStudy() {
  information('Models · favourites and guidance', ['WAI Illustrious v17 · anime · Refine available. Author guidance, own-gallery examples and retained timing references stay together; no new benchmark is implied.']);
  for (const text of ['Favourite this model · sample', 'Load automatically on startup · sample']) {
    const label = document.createElement('label'); label.className = 'subtle-field';
    const input = document.createElement('input'); input.type = 'checkbox'; input.checked = true;
    label.append(input, document.createTextNode(text)); byId('info-body').append(label);
  }
  const more = document.createElement('details'), summary = document.createElement('summary'); summary.textContent = 'Loading details and manual controls';
  const note = document.createElement('p'); note.textContent = 'Sample FP8 · Model offload. Quantization/offload, save without load and manual load/unload/retry stay secondary. A load failure still exposes its remedy. WSL/Nunchaku remains an occasional choice.';
  more.append(summary, note); byId('info-body').append(more);
}

function initWorkflow() {
  byId('compare').addEventListener('click', () => {if (picks.size === 2) byId('compare-title').textContent = 'Compare picked · proposed';});
  byId('watch-job').append(...studyJobs.map(job => new Option(`${job.id} · ${job.label}`, job.id)));
  byId('watch-job').addEventListener('change', syncWatchedJob);
  byId('cancel-job').addEventListener('click', () => cancelStudyJob(studyJobs.find(job => job.id === byId('watch-job').value)));
  byId('job-retry').addEventListener('click', () => {view('runs'); byId('workspace-note').textContent = 'Review the stopped run. Finish run is the accepted future recovery; current app Retry/Resume retains its documented behavior.';});
  byId('rerender-scope').addEventListener('click', () => openScopeStudy('Re-render'));
  byId('swap-scope').addEventListener('click', () => openScopeStudy('Swap tags'));
  byId('refine-scope').addEventListener('click', () => openScopeStudy('Refine'));
  byId('bin-picks').addEventListener('click', () => {if (!picks.size) {byId('gallery-note').textContent = 'Pick images before Bin picked.'; return;} openScopeStudy('Bin picked');});
  byId('star-picks').addEventListener('click', () => {if (!picks.size) {byId('gallery-note').textContent = 'Pick images before Star picked.'; return;} sampleImages.filter(image => picks.has(image.id)).forEach(image => image.starred = true); renderGallery(); byId('gallery-note').textContent = `${picks.size} picked sample images starred.`;});
  byId('sheet-toggle').addEventListener('click', () => {const on = byId('sheet-toggle').getAttribute('aria-pressed') !== 'true'; byId('sheet-toggle').setAttribute('aria-pressed', String(on)); byId('gallery-tiles').hidden = on; byId('gallery-matrix').hidden = !on; byId('sheet-fill').hidden = !on; if (on) renderSampleSheet();});
  byId('sheet-fill').addEventListener('click', () => {const on = byId('sheet-fill').getAttribute('aria-pressed') !== 'true'; byId('sheet-fill').setAttribute('aria-pressed', String(on)); byId('gallery-matrix').classList.toggle('fill', on);});
  byId('card-density').addEventListener('change', () => byId('gallery').dataset.density = byId('card-density').value);
  byId('card-crop').addEventListener('change', () => byId('gallery').dataset.crop = byId('card-crop').value);
  byId('use-seed').addEventListener('click', () => {byId('seed').value = String(inspected.seed); byId('seed').dispatchEvent(new Event('input')); byId('inspector-note').textContent = `Seed ${inspected.seed} set; other draft settings kept.`;});
  byId('refine-image').addEventListener('click', () => information(`Refine #${inspected.id}`, ['Scale, strength and versions stay direct in the dedicated Refine dialog, with source comparison after a pass.', 'This concept shows the route; the current application owns the editor, schedule and execution.']));
  byId('source-compare').addEventListener('click', () => {
    const source = sampleImages.find(image => image.id === inspected.sourceId);
    if (!source) {byId('inspector-note').textContent = 'This direct sample has no source pass. Inspect sample #107 to try source comparison.'; return;}
    byId('compare-title').textContent = 'Compare with source';
    byId('comparison-images').replaceChildren(imageTile(source, false), imageTile(inspected, false));
    byId('comparison-diff').textContent = `Source #${source.id} on the left · Pass #${inspected.id} on the right. Synthetic placeholders show context, not image quality.`;
    openDialog('compare-dialog');
  });
  for (const [id, field] of [['copy-prompt', 'prompt'], ['copy-seed', 'seed']]) byId(id).addEventListener('click', async () => {const text = String(inspected[field]); try {await navigator.clipboard.writeText(text); byId('inspector-note').textContent = `${field} copied.`;} catch {byId('inspector-note').textContent = `Copy ${field}: ${text}`;}});
  const help = {
    examples: ['Examples · beside the prompt', 'The application supplies checkpoint-author examples. This concept does not invent or substitute author prompts.'],
    'tags-help': ['Tags · beside the prompt', 'Checkpoint tags, autocomplete and editable/reorderable chips stay adjacent to authoring. Their vocabulary and validation remain the real application’s.'],
    'check-prompt': ['Check · beside the prompt', 'The application runs cold CPU tokenizers and sourced rules; this concept does not invent token counts or findings. Critic/rewrite/comparison remain contextual follow-ups.'],
    'prompt-guide': ['Prompting guide', 'Keep the selected checkpoint’s sourced guidance and control help reachable without leaving the draft.'],
    variables: ['Prompt variables', 'Frequent variable authoring has a direct entry here; occasional Steps/CFG/seed-count experiments live in Grid. The production server resolves the complete prompt list and count before Queue.'],
  };
  for (const [id, [title, text]] of Object.entries(help)) byId(id).addEventListener('click', () => information(title, [text, 'This is a placement study, with no model or application connection.']));
  syncWatchedJob();
  sampleImages[6].sourceId = 101;
}
