// Blind gallery reviews take a snapshot of the chosen images and their recipes.
// Declarations only; state.js has the classic-script load-order contract.
'use strict';

const blindEl = (name) => $(`blind-compare${name ? `-${name}` : ''}`);
let blindReview = null;

function blindSeed(record) {
  return record.seed_decimal ?? (Number.isSafeInteger(record.seed) ? String(record.seed) : null);
}

const blindSeedOrder = (a, b) => a.length - b.length || a.localeCompare(b);

function blindVariants(records) {
  if (records.some((rec) => rec.imported_name || rec.edit || rec.detailer || rec.mask || rec.outpaint || rec.tiled_refine)) return [];
  const distinct = records.some((rec) => rec.refine != null
    || rec.spatial_conditioning?.kind === 'regional_prompts'
    || rec.prompt !== records[0].prompt || rec.negative_prompt !== records[0].negative_prompt);
  if (!distinct) return [];
  const groups = new Map();
  const recipeAxes = SHEET_AXES.filter((axis) => axis.key !== 'seed');
  for (const rec of records) {
    const seed = blindSeed(rec);
    if (seed == null) return [];
    const axes = recipeAxes.map((axis) => axisValue(axis, rec));
    const key = JSON.stringify([rec.model_id, rec.quantization, rec.refine, rec.execution, rec.spatial_conditioning, axes]);
    if (!groups.has(key)) groups.set(key, { example: rec, seeds: new Map() });
    const seeds = groups.get(key).seeds;
    // A re-render can supply a second copy of a cell. Take the newest.
    if (!seeds.has(seed) || rec.id > seeds.get(seed).id) seeds.set(seed, rec);
  }
  return [...groups.values()];
}

function pickedBlindComparison() {
  const records = selectedRecords();
  if (records.length < 2) return { why: 'Pick two images or pick images from two prompt or Refine variants' };
  if (records.length === 2) return { records };
  const variants = blindVariants(records);
  return variants.length >= 2 ? { variants } : { why: 'Pick images from at least two different prompt or Refine variants' };
}

function blindSheetButton(records) {
  const variants = blindVariants(records);
  if (variants.length < 2) return null;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost small sheet-blind-compare';
  button.textContent = 'Blind comparison…';
  button.title = 'Compare prompt or Refine variants from the images loaded on this sheet';
  button.setAttribute('aria-haspopup', 'dialog');
  button.setAttribute('aria-controls', 'blind-compare');
  button.onclick = () => openBlindComparison({ variants });
  return button;
}

function blindVariantLabel(record) {
  const version = record.variables?.variant;
  return [
    version && (VERSION_NAMES.get(version) ?? version),
    record.refine != null && `Refine #${record.refine} at ${record.strength}`,
    checkpointName(record.model_id),
    `${record.width}×${record.height} · ${record.steps} steps · CFG ${record.guidance_scale}`,
    clip(record.prompt),
  ].filter(Boolean).join(' · ');
}

function blindPair(first, second, seed = null) {
  const sides = [{ record: first, variant: 0 }, { record: second, variant: 1 }];
  if (Math.random() < 0.5) sides.reverse();
  return { sides, seed, vote: null };
}

function openBlindComparison({ records, variants }) {
  const dialog = blindEl('');
  if (dialog.open) return;
  const sheet = $('gallery-more-sheet');
  if (sheet.open) closeGallerySheetFor(sheet);
  if (document.activeElement === document.body) {
    (isPhone() ? $(phoneSelecting() ? 'select-more' : 'btn-gallery-more') : els.pickedBtn).focus({ preventScroll: true });
  }
  const session = { variants, opener: document.activeElement, index: 0, load: 0, revealed: false };
  blindReview = session;
  blindEl('review').hidden = true;
  blindEl('choose').hidden = true;
  blindComparisonError('');
  dialog.showModal();
  if (records) {
    beginBlindPairs(session, [blindPair(...records)], []);
  } else if (variants.length === 2) {
    beginBlindVariants(session, 0, 1);
  } else {
    for (const [name, selected] of [['first', 0], ['second', 1]]) {
      const select = blindEl(name);
      select.replaceChildren(...variants.map(({ example }, index) =>
        new Option(`${index + 1}. ${blindVariantLabel(example)}`, index, false, index === selected)));
    }
    blindEl('choose').hidden = false;
  }
}

function beginBlindVariants(session, left, right) {
  if (left === right) {
    blindComparisonError('Choose two different variants.');
    return;
  }
  const [first, second] = [session.variants[left], session.variants[right]];
  const seeds = [...new Set([...first.seeds.keys(), ...second.seeds.keys()])].sort(blindSeedOrder);
  const pairs = [];
  const unmatched = [];
  for (const seed of seeds) {
    const a = first.seeds.get(seed);
    const b = second.seeds.get(seed);
    if (a && b) pairs.push(blindPair(a, b, seed));
    else unmatched.push(seed);
  }
  beginBlindPairs(session, pairs, unmatched);
}

function beginBlindPairs(session, pairs, unmatched) {
  session.pairs = pairs;
  blindEl('choose').hidden = true;
  blindEl('review').hidden = false;
  blindEl('results').hidden = true;
  blindEl('results').replaceChildren();
  blindEl('votes').hidden = !pairs.length;
  const select = blindEl('pair');
  select.replaceChildren(...pairs.map((pair, i) => new Option(String(i + 1), i)));
  select.parentElement.hidden = pairs.length < 2;
  blindEl('prev').hidden = blindEl('next').hidden = pairs.length < 2;
  const missing = blindEl('unmatched');
  missing.textContent = unmatched.length ? `Unmatched ${unmatched.length === 1 ? 'seed' : 'seeds'} — ${unmatched.join(', ')}` : '';
  missing.hidden = !unmatched.length;
  blindEl('close').focus({ preventScroll: true });
  drawBlindPair(session, 0);
}

function blindComparisonError(text) {
  const error = blindEl('error');
  error.hidden = !text;
  error.textContent = text;
}

function holdBlindButton(button, reason) {
  button.setAttribute('aria-disabled', String(!!reason));
  button.title = reason;
}

function syncBlindReview(session) {
  const pair = session.pairs[session.index];
  const recorded = session.pairs.filter(({ vote }) => vote != null).length;
  blindEl('status').textContent = pair
    ? `Pair ${session.index + 1} of ${session.pairs.length} · ${plural(recorded, 'choice')} recorded${session.loading ? ' · Loading images…' : ''}`
    : 'The two variants have no matching seeds.';
  holdBlindButton(blindEl('prev'), session.index === 0 ? 'This is the first pair' : '');
  holdBlindButton(blindEl('next'), session.index >= session.pairs.length - 1 ? 'This is the last pair' : '');
  for (const button of blindEl('votes').querySelectorAll('[data-blind-vote]')) {
    button.setAttribute('aria-pressed', String(pair?.vote === button.dataset.blindVote));
    holdBlindButton(button, session.ready ? '' : 'Both images must be visible before recording a choice');
  }
  holdBlindButton(blindEl('reveal'), recorded ? '' : 'Record a choice before revealing the settings');
}

function drawBlindPair(session, index) {
  const load = ++session.load;
  session.index = index;
  session.ready = false;
  session.loading = !!session.pairs[index];
  session.drag = null;
  session.zoom = 'fit';
  session.center = { x: 0.5, y: 0.5 };
  blindComparisonError('');
  blindEl('pair').value = String(index);
  syncBlindReview(session);
  const pair = session.pairs[index];
  const frame = blindEl('').querySelector('.blind-compare-images');
  frame.hidden = !pair;
  const zooms = blindEl('').querySelectorAll('[data-blind-zoom]');
  for (const button of zooms) button.hidden = !pair;
  if (!pair) return;
  if (session.revealed) drawBlindResults(session);

  const images = [...frame.querySelectorAll('img')];
  const current = () => blindReview === session && session.load === load && blindEl('').open;
  const decoded = images.map((image, side) => {
    image.hidden = true;
    image.src = pair.sides[side].record.url;
    return image.decode();
  });
  paintBlindPair(session);
  void Promise.all(decoded).then(() => {
    if (!current()) return;
    session.ready = true;
    session.loading = false;
    paintBlindPair(session);
    images.forEach((image) => { image.hidden = false; });
    syncBlindReview(session);
  }).catch(() => {
    if (!current()) return;
    session.loading = false;
    syncBlindReview(session);
    blindComparisonError('This pair could not be displayed.');
  });
}

const blindClamp = (value, half) => Math.max(half, Math.min(1 - half, value));

function paintBlindPair(session) {
  const pair = session.pairs?.[session.index];
  if (!pair) return;
  const stages = [...blindEl('').querySelectorAll('.blind-compare-stage')];
  const density = window.devicePixelRatio;
  const pixelScale = (session.zoom === 'double' ? 2 : 1) / density;
  const snap = (position) => Math.round(position * density) / density;
  const views = stages.map((stage, index) => {
    const { width, height } = pair.sides[index].record;
    const scale = session.zoom === 'fit'
      ? Math.min(stage.clientWidth / width, stage.clientHeight / height) : pixelScale;
    const w = width * scale;
    const h = height * scale;
    return { stage, w, h, x: w > stage.clientWidth, y: h > stage.clientHeight };
  });
  for (const { stage, w, h, x, y } of views) {
    const image = stage.querySelector('img');
    const halfX = x ? stage.clientWidth / w / 2 : 0.5;
    const halfY = y ? stage.clientHeight / h / 2 : 0.5;
    image.style.width = `${w}px`;
    image.style.height = `${h}px`;
    image.style.left = `${snap(stage.clientWidth / 2 - w * blindClamp(session.center.x, halfX))}px`;
    image.style.top = `${snap(stage.clientHeight / 2 - h * blindClamp(session.center.y, halfY))}px`;
    stage.dataset.pan = String(session.ready && views.some((view) => view.x || view.y));
  }
  for (const button of blindEl('').querySelectorAll('[data-blind-zoom]')) {
    button.setAttribute('aria-pressed', String(button.dataset.blindZoom === session.zoom));
  }
}

function blindPan(session, stage, origin, dx, dy) {
  const image = stage.querySelector('img');
  const other = [...stage.closest('.blind-compare-images').querySelectorAll('.blind-compare-stage')].find((member) => member !== stage);
  const partner = other.querySelector('img');
  const width = Math.max(image.width, partner.width);
  const height = Math.max(image.height, partner.height);
  const halfX = Math.min(stage.clientWidth / width / 2, 0.5);
  const halfY = Math.min(stage.clientHeight / height / 2, 0.5);
  session.center = {
    x: blindClamp(origin.x + dx / width, halfX),
    y: blindClamp(origin.y + dy / height, halfY),
  };
  paintBlindPair(session);
}

function drawBlindResults(session) {
  const headline = document.createElement('h3');
  headline.id = 'blind-compare-result-head';
  headline.tabIndex = -1;
  headline.textContent = 'Results';
  const wins = [0, 0];
  let ties = 0;
  let pending = 0;
  for (const pair of session.pairs) {
    if (pair.vote === 'tie') ties++;
    else if (pair.vote) wins[pair.sides[pair.vote === 'a' ? 0 : 1].variant]++;
    else pending++;
  }
  const totals = document.createElement('p');
  totals.textContent = `${wins.map((count, i) => `Variant ${i + 1} preferred ${count}`).join(' · ')} · ${plural(ties, 'tie')} · ${pending} unreviewed`;
  const facts = session.pairs[session.index].sides.map(({ record, variant }, index) => {
    const card = document.createElement('div');
    const label = document.createElement('h4');
    const prompt = document.createElement('p');
    const settings = document.createElement('p');
    label.textContent = `${index === 0 ? 'A' : 'B'} · Variant ${variant + 1} · #${record.id}`;
    prompt.textContent = record.prompt;
    settings.textContent = record.imported_name ? `Imported image · ${record.imported_name} · ${record.width}×${record.height}`
      : [settingsLine({ ...record, seed: blindSeed(record) ?? record.seed }), record.quantization].filter(Boolean).join(' · ');
    card.append(label, prompt);
    if (record.negative_prompt) {
      const negative = document.createElement('p');
      negative.textContent = `Negative prompt — ${negativeShown(record.negative_prompt)}`;
      card.append(negative);
    }
    card.append(settings);
    const spatial = record.spatial_conditioning;
    if (spatial?.kind === 'regional_prompts') {
      const regions = document.createElement('ul');
      for (const { left, top, right, bottom, prompt: localPrompt } of spatial.regions) {
        const item = document.createElement('li');
        item.textContent = `${left}%, ${top}% to ${right}%, ${bottom}% · ${localPrompt}`;
        regions.append(item);
      }
      card.append(regions);
    }
    return card;
  });
  const results = blindEl('results');
  const vote = session.pairs[session.index].vote;
  const choice = document.createElement('p');
  choice.textContent = !vote ? 'Current pair — Unreviewed' : vote === 'tie' ? 'Current pair — Tie' : `Current pair — Preferred ${vote.toUpperCase()}`;
  results.replaceChildren(headline, totals, choice, ...facts);
  results.hidden = false;
}

function initBlindComparison() {
  const dialog = blindEl('');
  blindEl('close').onclick = () => dialog.close();
  dialog.addEventListener('close', () => {
    if (dialog.open) return;
    const opener = blindReview?.opener;
    blindReview = null;
    dialog.querySelectorAll('img').forEach((image) => image.removeAttribute('src'));
    if (opener?.isConnected && document.activeElement === document.body) opener.focus({ preventScroll: true });
  });
  blindEl('start').onclick = () => beginBlindVariants(blindReview, +blindEl('first').value, +blindEl('second').value);
  blindEl('pair').onchange = (event) => drawBlindPair(blindReview, +event.target.value);
  for (const [name, move] of [['prev', -1], ['next', 1]]) {
    blindEl(name).onclick = () => {
      const index = blindReview.index + move;
      if (index >= 0 && index < blindReview.pairs.length) drawBlindPair(blindReview, index);
      else blindComparisonError(blindEl(name).title);
    };
  }
  for (const button of dialog.querySelectorAll('[data-blind-zoom]')) {
    button.onclick = () => { blindReview.zoom = button.dataset.blindZoom; paintBlindPair(blindReview); };
  }
  for (const button of dialog.querySelectorAll('[data-blind-vote]')) {
    button.onclick = () => {
      const session = blindReview;
      if (!session.ready || session.revealed) { blindComparisonError(button.title); return; }
      session.pairs[session.index].vote = button.dataset.blindVote;
      syncBlindReview(session);
    };
  }
  blindEl('reveal').onclick = () => {
    const session = blindReview;
    if (blindEl('reveal').getAttribute('aria-disabled') === 'true') {
      blindComparisonError(blindEl('reveal').title);
      return;
    }
    session.revealed = true;
    blindComparisonError('');
    blindEl('votes').hidden = true;
    drawBlindResults(session);
    $('blind-compare-result-head').focus({ preventScroll: true });
  };
  const stages = dialog.querySelectorAll('.blind-compare-stage');
  const resize = new ResizeObserver(() => { if (blindReview && dialog.open) paintBlindPair(blindReview); });
  const moves = { ArrowLeft: [-48, 0], ArrowRight: [48, 0], ArrowUp: [0, -48], ArrowDown: [0, 48] };
  for (const stage of stages) {
    resize.observe(stage);
    stage.addEventListener('keydown', (event) => {
      const move = moves[event.key];
      if (!move || stage.dataset.pan !== 'true' || keyboardCommandModifier(event)) return;
      event.preventDefault();
      blindPan(blindReview, stage, blindReview.center, ...move);
    });
    stage.addEventListener('pointerdown', (event) => {
      if (stage.dataset.pan !== 'true' || event.button !== 0 || blindReview.drag) return;
      blindReview.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, center: blindReview.center };
      stage.setPointerCapture(event.pointerId);
    });
    stage.addEventListener('pointermove', (event) => {
      const drag = blindReview?.drag;
      if (drag?.id === event.pointerId) blindPan(blindReview, stage, drag.center, drag.x - event.clientX, drag.y - event.clientY);
    });
    stage.addEventListener('lostpointercapture', (event) => {
      if (blindReview?.drag?.id === event.pointerId) blindReview.drag = null;
    });
  }
}
