// AuraDream's page script: the phone's compose controls (styles/phone.css
// draws them) - the topbar's model chip and its ⋯, the prompt's toolbar
// under the box and the check's sheet, and the size as Shape and Size -
// each over the page's own nodes, moved whole under PHONE_MEDIA with
// phone.js's phoneMove and home again off it, or drawn for the phone alone
// and taken out off it. Never a copy: `els` and every writer keep the one
// node. Declarations only; phone.js calls initPhoneCompose from initPhone
// and syncPhoneCompose from syncPhoneHosts.
'use strict';

// The nodes moved here, each home again off the phone: the status into
// Models (the chip), and the caption's Examples, Check, Tags and Expand
// into the toolbar, in that order - the authors' words, the check, then
// the checkpoint's own helpers. Reset is the topbar ⋯'s.
const phoneToolbarNodes = () => [$('examples'), els.analyze, els.tagHelper, els.expandPrompt];
const phoneComposeMoved = () => [$('status'), ...phoneToolbarNodes()];

// From syncPhoneHosts: at boot and at every change of tier.
function syncPhoneCompose() {
  if (isPhone()) {
    phoneMove($('status'), els.models, els.models.querySelector('.models-word'));
    const bar = $('prompt-toolbar');
    for (const node of phoneToolbarNodes()) phoneMove(node, bar);
    placePhoneSize(true);
    // A check open as the phone comes is shut, its line saying it is there:
    // its sheet opens on a press, never by itself.
    if (!els.promptAnalysis.hidden) dismissAnalysis();
  } else {
    if ($('check-sheet').open) $('check-sheet').close();
    for (const node of phoneComposeMoved()) phoneHome(node);
    placePhoneSize(false);
  }
  // The check's place - the sheet on the phone - is placeTagValidation's.
  placeTagValidation();
  syncPhoneCheckFolds();
  syncPhoneSize();
  syncCheckChip();
}

// ------------------------------------------------------------- the topbar

// The ⋯: an action sheet (the docked menu, context-menu.js) of what the
// phone's topbar does not draw - Load or Unload, the saved setups, the
// snippets, Load settings from PNG, and Reset last in --danger. Each row
// presses the page's own button, held and titled as it is, so a held one
// says why: over the bars, where the press was in sight (sayAtGenerate).
function openTopbarMenu(event) {
  const button = $('btn-topbar-more');
  const press = (label, own) => {
    const item = buttonItem(label, own);
    return item && { ...item, run: () => { own.click(); sayAtGenerate(); } };
  };
  const box = button.getBoundingClientRect();
  showContextMenu([
    press(els.load.textContent, els.load),
    press('Unload', els.unload),
    null,
    { label: 'Saved setups…', title: 'Restore, save and manage named setups of the panel', run: () => { void openSetups(); } },
    { label: 'Personal snippets…', title: 'Insert, save and manage your own prompt snippets', run: () => { void openSnippets(); } },
    press('Load settings from PNG…', els.pngButton),
    null,
    dangerItem(press('Reset', els.reset)),
  ], 'More', button, box.left, box.bottom, event.detail === 0);
}

// ---------------------------------------------------------- the check sheet

// The sheet is open while the check is shown on the phone: a Check's
// answer opens it, its ✕ (closeAnalysis), Escape and Back shut it, a
// model's change or Reset (clearAnalysis) too. Shut, what was found is
// kept (the next Check replaces it) and the line under the prompt says so.
function syncPhoneCheck() {
  const sheet = $('check-sheet');
  if (isPhone() && sheet.contains(els.promptAnalysis)) {
    if (!els.promptAnalysis.hidden && !sheet.open) sheet.showModal();
    else if (els.promptAnalysis.hidden && sheet.open) sheet.close();
  }
  syncCheckChip();
}

// Escape or Back shut the sheet: the check is shut as its ✕ shuts it, the
// focus on Check. Its ✕ shut it already; a focus the closing dropped goes
// to Check too.
function checkSheetClosed() {
  if (!isPhone()) return;
  if (!els.promptAnalysis.hidden) closeAnalysis();
  else if (!document.activeElement || document.activeElement === document.body) els.analyze.focus({ preventScroll: true });
  syncCheckChip();
}

// A finding's Show selects its words in the prompt or the negative, which
// the sheet covers: the sheet is left first, its line under the prompt
// opening it again. Off the phone, nothing.
function leavePhoneCheck() {
  const sheet = $('check-sheet');
  if (!isPhone() || !sheet.open) return;
  dismissAnalysis();
  sheet.close();
}

// The line under the toolbar while a check is kept and shut: "Check" and
// what it found - its findings' count, amber with a warning among them, and
// that the prompt has changed since - and a chevron into the sheet. A press
// shows the same check again; Check runs a new one.
function syncCheckChip() {
  const chip = $('check-chip');
  const body = analysisResult?.body;
  const show = isPhone() && !!body && els.promptAnalysis.hidden;
  if (chip.hidden !== !show) chip.hidden = !show;
  if (!show) return;
  const n = body.findings.length;
  const warns = body.findings.filter((f) => f.severity === 'warn').length;
  const found = n ? plural(n, 'finding') : 'nothing flagged';
  const stale = analysisStale ? ` · ${els.analysisStale.textContent}` : '';
  const words = chip.querySelector('.check-chip-words');
  if (words.textContent !== found + stale) words.textContent = found + stale;
  chip.classList.toggle('warn', warns > 0);
  chip.title = 'Show the prompt check again';
  chip.setAttribute('aria-label', `Prompt check: ${found}${stale}. Show it again`);
}

function pressCheckChip() {
  if (!analysisResult) return;
  renderAnalysis(true);
}

// The check's sections are disclosures on a phone. Keep every content node
// in its original section: the critic's stale/status rules and the writers'
// own hidden states still apply. Only the heading's text moves into its
// disclosure button; a wider tier gets the plain heading back. A redraw or
// tier change keeps each section's choice, with Findings open initially.
const phoneCheckFolds = [];

function syncPhoneCheckFolds() {
  for (const fold of phoneCheckFolds) {
    const { section, heading, button, words } = fold;
    if (isPhone()) {
      if (button.parentNode !== heading) {
        button.append(...words);
        heading.append(button);
      }
      button.setAttribute('aria-expanded', String(fold.open));
      section.toggleAttribute('data-check-collapsed', !fold.open);
    } else {
      section.removeAttribute('data-check-collapsed');
      if (button.parentNode === heading) {
        heading.append(...words);
        button.remove();
      }
    }
  }
}

function initPhoneCheckFolds() {
  for (const [i, section] of [...els.promptAnalysis.querySelectorAll('.analysis-section')].entries()) {
    const heading = section.querySelector(':scope > h3');
    const content = [...section.children].filter((node) => node !== heading);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'phone-check-toggle';
    button.id = `phone-check-toggle-${i + 1}`;
    button.setAttribute('aria-controls', content.map((node, j) => {
      if (!node.id) node.id = `phone-check-content-${i + 1}-${j + 1}`;
      return node.id;
    }).join(' '));
    const fold = { section, heading, button, words: [...heading.childNodes], open: content.includes(els.analysisFindings) };
    button.addEventListener('click', () => {
      fold.open = !fold.open;
      syncPhoneCheckFolds();
    });
    phoneCheckFolds.push(fold);
  }
}

// ----------------------------------------------------------- Shape and Size

// The size on the phone: a 44 px segmented Shape - Square, Portrait,
// Landscape, each with the ratio it would take - and a Size chip on the
// caption's line, where the PC's eighteen pills stood in three rows (four
// at 320 px). Both press the strips' own buttons (sizes.js, `choose`),
// which stay in the page undrawn: the presets are the server's, this is
// their presentation. A Shape's ratio is the one last chosen in it, or the
// first of 3:4, 2:3, 7:9, 9:16, 4:7, 5:12 (their inverses on their side)
// in this checkpoint's guide at this size; a press on the chosen Portrait
// or Landscape opens its ratios as the docked menu's rows, the chip the
// megapixels and Custom dimensions… (More options). Made once, put in the
// Size field on the phone and taken out off it.
const SHAPES = [['square', 'Square'], ['portrait', 'Portrait'], ['landscape', 'Landscape']];
const PORTRAIT_FIRST = ['3:4', '2:3', '7:9', '9:16', '4:7', '5:12'];
const phoneShapeKept = { portrait: null, landscape: null };
let phoneShape = null;
let phoneSizeChip = null;

function aspectShape(aspect) {
  const [w, h] = aspect.split(':').map(Number);
  return w === h ? 'square' : h > w ? 'portrait' : 'landscape';
}

function shapeAspect(shape) {
  if (shape === 'square') return '1:1';
  const kept = phoneShapeKept[shape];
  if (kept && ASPECTS.includes(kept)) return kept;
  const order = PORTRAIT_FIRST.map((a) => (shape === 'portrait' ? a : oppositeAspect(a))).filter((a) => ASPECTS.includes(a));
  return order.find((a) => presetIsInGuide(a, pickedTier)) ?? order[0];
}

const presetFor = (aspect) => [...els.presets.querySelectorAll('button')][ASPECTS.indexOf(aspect)];
const sizeWords = (aspect, tier) => SIZES[aspect][tier].join(' × ');
const everyAspectGuided = () => ASPECTS.every((a) => presetIsInGuide(a, pickedTier));

function placePhoneSize(on) {
  if (!phoneShape) return;
  const field = els.presets.closest('.field');
  if (on) {
    if (phoneSizeChip.parentNode !== field) field.firstElementChild.after(phoneSizeChip);
    if (phoneShape.parentNode !== field) els.presets.before(phoneShape);
  } else {
    phoneSizeChip.remove();
    phoneShape.remove();
  }
}

// From syncSize (sizes.js) and the tier's sync: the chosen Shape (the
// width against the height, a custom size's too), each segment's ratio and
// size, the guide's dot where some aspects are and some are not, and the
// chip's megapixels (a custom size's to a tenth) and pixels.
function syncPhoneSize() {
  if (!phoneShape?.isConnected) return;
  const w = +els.width.value, h = +els.height.value;
  const match = sizePresetMatch(w, h);
  if (match && aspectShape(match[0]) !== 'square') phoneShapeKept[aspectShape(match[0])] = match[0];
  const now = !(w > 0 && h > 0) ? null : w === h ? 'square' : h > w ? 'portrait' : 'landscape';
  const marks = !everyAspectGuided();
  for (const b of phoneShape.children) {
    const shape = b.dataset.shape;
    const name = SHAPES.find(([s]) => s === shape)[1];
    const on = shape === now;
    const aspect = on ? match?.[0] ?? null : shapeAspect(shape);
    const opens = on && shape !== 'square';
    const ratio = aspect ?? 'custom';
    const dims = aspect ? sizeWords(aspect, pickedTier) : `${w} × ${h}`;
    b.querySelector('.shape-ratio').textContent = ratio;
    b.setAttribute('aria-pressed', String(on));
    b.classList.toggle('opens', opens);
    b.classList.toggle('recommended', marks && !!aspect && presetIsInGuide(aspect, pickedTier));
    if (opens) b.setAttribute('aria-haspopup', 'menu');
    else { b.removeAttribute('aria-haspopup'); b.removeAttribute('aria-expanded'); }
    b.title = opens ? `${name} ${ratio}, ${dims}: press for the other ${shape} ratios` : `${name} ${ratio}, ${dims}`;
    b.setAttribute('aria-label', opens ? `${name}, ${ratio}, ${dims}. Other ${shape} ratios` : `${name}, ${ratio}, ${dims}`);
  }
  const mp = match ? `${match[1]} MP` : `${((w * h) / 1e6).toFixed(1)} MP`;
  const px = w > 0 && h > 0 ? `${w} × ${h}` : 'check width and height';
  phoneSizeChip.querySelector('.size-chip-mp').textContent = mp;
  phoneSizeChip.querySelector('.size-chip-px').textContent = px;
  phoneSizeChip.setAttribute('aria-label', `Size: ${mp}, ${px}. Other sizes`);
  phoneSizeChip.classList.toggle('warn', els.sizeHint.classList.contains('warn'));
}

// A Shape's press: another Shape takes its ratio at this size (its preset's
// own press); the chosen Portrait or Landscape opens its ratios.
function pressPhoneShape(event) {
  const b = event.target.closest('[data-shape]');
  if (!b) return;
  const shape = b.dataset.shape;
  if (b.getAttribute('aria-pressed') !== 'true') { presetFor(shapeAspect(shape))?.click(); return; }
  if (shape === 'square') return;
  const match = sizePresetMatch();
  const name = SHAPES.find(([s]) => s === shape)[1];
  const marks = !everyAspectGuided();
  const items = ASPECTS.filter((a) => aspectShape(a) === shape).map((a) => ({
    label: `${a} · ${sizeWords(a, pickedTier)}${marks && presetIsInGuide(a, pickedTier) ? ' · recommended' : ''}`,
    checked: match?.[0] === a,
    title: presetFor(a)?.title ?? '',
    run: () => presetFor(a)?.click(),
  }));
  const box = b.getBoundingClientRect();
  showContextMenu([{ heading: `${name} at ${pickedTier} MP` }, ...items], `${name} ratios`, b, box.left, box.bottom, event.detail === 0);
}

// The chip: the four sizes of this ratio, each its tier's own press, then
// Custom dimensions… in More options' sheet, its fold opened and brought up
// under the title.
function pressPhoneSizeChip(event) {
  const match = sizePresetMatch();
  const aspect = match?.[0] ?? pickedAspect;
  const items = TIERS.map((tier, i) => ({
    label: `${tier} MP · ${sizeWords(aspect, tier)}${presetIsInGuide(aspect, tier) && !TIERS.every((t) => presetIsInGuide(aspect, t)) ? ' · recommended' : ''}`,
    checked: match?.[1] === tier,
    title: els.tiers.children[i]?.title ?? '',
    run: () => els.tiers.children[i]?.click(),
  }));
  const box = phoneSizeChip.getBoundingClientRect();
  showContextMenu([{ heading: `Size at ${aspect}` }, ...items, null, {
    label: 'Custom dimensions…', title: 'Width and height of your own, in More options',
    run: () => openCustomSize(),
  }], 'Size', phoneSizeChip, box.left, box.bottom, event.detail === 0);
}

function openCustomSize() {
  openMoreSheet();
  els.customSize.open = true;
  revealInMoreSheet(els.customSize);
}

function makePhoneSize() {
  phoneShape = document.createElement('div');
  phoneShape.className = 'phone-shape';
  phoneShape.setAttribute('role', 'group');
  phoneShape.setAttribute('aria-label', 'Shape');
  for (const [shape, name] of SHAPES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.shape = shape;
    const label = document.createElement('span');
    label.className = 'shape-name';
    label.textContent = name;
    const ratio = document.createElement('span');
    ratio.className = 'shape-ratio';
    b.append(label, ratio);
    phoneShape.append(b);
  }
  phoneShape.addEventListener('click', pressPhoneShape);
  phoneSizeChip = document.createElement('button');
  phoneSizeChip.type = 'button';
  phoneSizeChip.className = 'phone-size-chip';
  phoneSizeChip.id = 'phone-size-chip';
  phoneSizeChip.setAttribute('aria-haspopup', 'menu');
  phoneSizeChip.title = 'The size in megapixels, or custom dimensions';
  const mp = document.createElement('span');
  mp.className = 'size-chip-mp';
  const px = document.createElement('span');
  px.className = 'size-chip-px';
  phoneSizeChip.append(mp, px);
  phoneSizeChip.addEventListener('click', pressPhoneSizeChip);
}

function initPhoneCompose() {
  for (const node of phoneComposeMoved()) {
    const mark = document.createComment(` ${node.id || node.className}'s place off the phone (phone-compose.js) `);
    node.before(mark);
    phoneHomes.set(node, mark);
  }
  makePhoneSize();
  initPhoneCheckFolds();
  const chip = $('check-chip');
  const name = document.createElement('span');
  name.className = 'check-chip-name';
  name.textContent = 'Check:';
  const words = document.createElement('span');
  words.className = 'check-chip-words';
  chip.append(name, words);
  chip.addEventListener('click', pressCheckChip);
  $('btn-topbar-more').addEventListener('click', openTopbarMenu);
  $('check-sheet').addEventListener('close', checkSheetClosed);
  const watch = new MutationObserver(syncPhoneCheck);
  watch.observe(els.promptAnalysis, { attributes: true, attributeFilter: ['hidden', 'class'] });
  watch.observe(els.analysisSummary, { childList: true, characterData: true, subtree: true });
}
