// AuraDream's page script: the phone's own places for nodes the page holds
// once (styles/phone.css draws them) - Generate in the Create view's action
// bar over the tab bar, the results under the settings that make them, the
// running job and the queue in a sheet - each moved there whole under
// PHONE_MEDIA, never copied, and home again off it, so `els`, `setQueue`
// and every writer keep writing the one node; the bar's newest result and
// queue count; and More options - its row, its sheet holding the tuning,
// the negative prompt's place and Batch's and Repeat's − and +.
// Declarations only; touch.js calls initPhone at boot and syncPhoneHosts
// whenever the tier changes.
'use strict';

// Each moved node's home: a comment left where it stood in the markup.
const phoneHomes = new Map();

// `node` into `host` before `before` (its end when null), with `moveBefore`
// where the browser has it, which keeps the node's state - its focus, a
// live region's - and elsewhere the focus put back by hand, as
// placeTagValidation does.
function phoneMove(node, host, before = null) {
  if (node.parentNode === host && (before ? node.nextSibling === before : true)) return;
  const focused = document.activeElement;
  const inside = node.contains(focused);
  let moved = false;
  if (typeof host.moveBefore === 'function') {
    try { host.moveBefore(node, before); moved = true; } catch { /* inserted below */ }
  }
  if (!moved) host.insertBefore(node, before);
  if (inside && focused.isConnected && focused !== document.activeElement) focused.focus({ preventScroll: true });
}

function phoneHome(node) {
  const mark = phoneHomes.get(node);
  if (mark && mark.nextSibling !== node) phoneMove(node, mark.parentNode, mark.nextSibling);
}

// More options' sheet holds, in this order, the tuning the phone's first
// screen leaves out: Sampling, Seed (its dice after the box), Repeat,
// Custom dimensions, LoRA & references, Notify and Then refine, Step cache
// and Grid search - and, first, the negative prompt while it does not run
// (placePhoneNegative). Run grid is in its foot.
const moreSheetNodes = () => [
  $('sampling-caption').closest('.field'), els.seed.closest('.sub'), els.runs.closest('.sub'),
  els.customSize, els.conditioningSection, document.querySelector('.notification-choice'),
  document.querySelector('.execution-cache-row'), els.gridSection,
];
const moreSheetMoved = () => [...moreSheetNodes(), els.negField, els.dice, els.gridRun];

// The nodes that move, in the order each host holds them: the queue's sheet
// has the panel's line (#error) between the job and the queue, so a held
// press in the sheet - ↑, Pause, a row's menu - says why where it was
// pressed, not in a panel the sheet covers.
const phoneResultNodes = () => [els.viewer, els.viewerNote, els.recent];
const phoneQueueNodes = () => [els.progress, els.error, els.queueNote];

// On the phone, Generate in the bar before the queue's count, the results
// in their host and the tuning in More options' sheet (the dice after the
// Seed box, Run grid in the foot, the − and + after their boxes); off it,
// all home, the sheets shut, the queue's nodes home too and the − and +
// gone. From syncTier (touch.js): at boot and on every change.
function syncPhoneHosts() {
  if (isPhone()) {
    phoneMove(els.generate, $('phone-actions'), $('phone-queue'));
    for (const node of phoneResultNodes()) {
      if (node.parentNode !== $('phone-results')) phoneMove(node, $('phone-results'));
    }
    const body = $('more-sheet-body');
    for (const node of moreSheetNodes()) {
      if (node.parentNode !== body) phoneMove(node, body);
    }
    if (els.seed.nextElementSibling !== els.dice) phoneMove(els.dice, els.seed.parentNode, els.seed.nextSibling);
    phoneMove(els.gridRun, $('more-sheet-actions'));
    for (const s of phoneSteppers) {
      s.input.before(s.minus);
      s.input.after(s.plus);
      s.input.setAttribute('aria-label', s.name);
    }
    placePhoneNegative({ now: true });
  } else {
    for (const sheet of [$('queue-sheet'), $('more-sheet')]) if (sheet.open) sheet.close();
    for (const s of phoneSteppers) { s.minus.remove(); s.plus.remove(); s.input.removeAttribute('aria-label'); }
    for (const node of [els.generate, ...phoneResultNodes(), ...phoneQueueNodes(), ...moreSheetMoved()]) phoneHome(node);
    syncPhoneNegativeFold();
  }
  syncPhoneDock();
  syncPhoneQueue();
  syncMoreOptions();
  // The topbar's chip and ⋯, the prompt's toolbar and check, Shape and
  // Size (phone-compose.js).
  syncPhoneCompose();
  // The gallery's one-row head and its two sheets (phone-gallery.js).
  syncPhoneGallery();
  // The viewer's bars over the picture (phone-viewer.js).
  syncPhoneViewer();
  // The sheets' explanations folded, the keyboard's inset (phone-sheets.js).
  syncPhoneSheets();
}

// ------------------------------------------------------- the newest result

// What the bar's picture shows, and so what its press opens: 'live' for
// the running job's live preview, else the newest result's record, else
// null (nothing made on this page yet).
let phoneDockShows = null;

// The newest result: the viewer's last image not gone, else the newest of
// this page's earlier results (#recent's), with the list the lightbox
// walks from it.
function phoneNewest() {
  for (let i = viewerRecords.length - 1; i >= 0; i--) {
    if (!isGone(viewerRecords[i].id)) return { rec: viewerRecords[i], list: viewerRecords };
  }
  const rec = sessionRecords.find((r) => !isGone(r.id));
  return rec ? { rec, list: sessionRecords } : null;
}

// The picture, its ring and its words, from what the page shows already:
// the viewer's live preview and images (watched), and the topbar's job line
// and fraction, which job-progress.js writes (watched, as the tab bar's
// dot watches them).
function syncPhoneDock() {
  const dock = $('phone-dock');
  const img = $('phone-dock-img');
  const running = !els.topbarLine.hidden;
  const frame = els.viewer.querySelector(':scope > figure.live img.preview[src]');
  const newest = phoneNewest();
  phoneDockShows = frame ? 'live' : newest?.rec ?? null;
  const src = frame ? frame.getAttribute('src') : newest ? newest.rec.thumb_url ?? newest.rec.url : '';
  if (src && img.getAttribute('src') !== src) img.src = src;
  if (!src) img.removeAttribute('src');
  img.hidden = !src;
  dock.classList.toggle('running', running);
  dock.classList.toggle('live', !!frame);
  dock.classList.toggle('indeterminate', running && els.topbarLine.classList.contains('indeterminate'));
  const fraction = parseFloat(els.topbarFill.style.width) / 100;
  dock.style.setProperty('--ring', String(running && fraction > 0 ? Math.min(1, fraction) : 0));
  const step = running ? progressShown() : '';
  const job = running ? ` - a job running${step ? `, ${step}` : ''}` : '';
  let held = '';
  let title = '';
  if (phoneDockShows === 'live') title = 'Show the live preview in the results';
  else if (phoneDockShows) title = `Open #${phoneDockShows.id}, the newest result`;
  else held = running ? 'No image of this job yet' : 'Nothing generated on this page yet';
  dock.setAttribute('aria-disabled', String(!!held));
  dock.title = held || title;
  dock.setAttribute('aria-label', held ? `Newest result: ${held[0].toLowerCase()}${held.slice(1)}${job}` : `${title}${job}`);
}

// A press: the newest result in the lightbox, walking the viewer's strip
// (or this page's earlier results) from it, as the result's own picture
// does; the live preview's tile in the results, scrolled to by this press
// as New image's was (goToNewImage), while that is what the picture shows.
// Held, it says why above the bars. Never by itself.
function pressPhoneDock() {
  const dock = $('phone-dock');
  if (dock.getAttribute('aria-disabled') === 'true') {
    showError(`${dock.title}.`, true);
    sayAtGenerate();
    return;
  }
  if (phoneDockShows === 'live') { goToNewImage(); return; }
  const newest = phoneNewest();
  if (!newest) return;
  if (newest.list === viewerRecords) openViewerImage(newest.rec, 'dock');
  else openLightbox(newest.rec, sessionRecords, 'dock');
}

// --------------------------------------------------------------- the queue

// The count of jobs waiting behind the running one, on the bar's queue
// button, and its hold while nothing runs or waits.
function syncPhoneQueue() {
  const button = $('phone-queue');
  const count = $('phone-queue-count');
  const running = !els.progress.hidden;
  const waiting = els.queueNote.hidden ? 0 : els.queueList.children.length;
  count.hidden = !waiting;
  count.textContent = waiting > 99 ? '99+' : String(waiting);
  button.classList.toggle('running', running);
  const held = !running && !waiting;
  const words = [running ? 'a job running' : '', waiting ? `${waiting} waiting` : ''].filter(Boolean).join(', ');
  button.setAttribute('aria-disabled', String(held));
  button.title = held ? 'Nothing running or queued' : `Open the queue: ${words}`;
  button.setAttribute('aria-label', `Queue: ${held ? 'nothing running or queued' : words}`);
}

// The sheet holds the running job's progress (Pause, Skip, Cancel, the
// bars) and the queue's rows while it is open: the panel's own nodes, moved
// in and home again as it closes (by its ✕, Escape, Back, or the phone
// going). Held, the press says why above the bars.
function openQueueSheet() {
  const button = $('phone-queue');
  const sheet = $('queue-sheet');
  if (button.getAttribute('aria-disabled') === 'true') {
    showError(`${button.title}.`, true);
    sayAtGenerate();
    return;
  }
  if (!isPhone() || sheet.open) return;
  // A refusal riding over the bars (sayAtGenerate) is the panel's again.
  els.error.classList.remove('at-generate');
  for (const node of phoneQueueNodes()) phoneMove(node, $('queue-sheet-body'));
  sheet.showModal();
}

function queueSheetClosed() {
  for (const node of phoneQueueNodes()) phoneHome(node);
}

// A queued row's ⋯ (job-queue.js queueRow): its Move to front and its
// Cancel as the docked menu's rows (context-menu.js), in the sheet, the
// job named over them, Cancel last in --danger. Each row presses the row's
// own button as it is at the choice - the queue redraws its rows - so a
// held one says why, as its press would; a job gone from the queue while
// the menu was open (started, cancelled elsewhere) is said, never pressed.
function openQueueRowMenu(button, byKey) {
  const li = button.closest('li[data-id]');
  if (!li) return;
  const id = li.dataset.id;
  const live = (action) => [...els.queueList.children].find((row) => row.dataset.id === id)
    ?.querySelector(`[data-queue-action="${action}"]`);
  const item = (label, action) => {
    const own = live(action);
    if (!own) return null;
    return {
      label, held: own.getAttribute('aria-disabled') === 'true', title: own.title,
      run: () => {
        const now = live(action);
        if (now) now.click();
        else showJobNote('That job is no longer queued.');
      },
    };
  };
  const name = li.querySelector(':scope > span')?.textContent ?? '';
  const box = button.getBoundingClientRect();
  showContextMenu([{ heading: name }, item('↑ Move to front', 'front'), null, dangerItem(item('× Cancel this job', 'cancel'))],
    'Queued job', button, box.left, box.bottom, byKey);
}

// ------------------------------------------------------------ More options

// The negative prompt is under the prompt on the phone only while it runs -
// a checkpoint whose CFG (or True CFG) at its value applies it - and the
// first thing in More options otherwise, its caption saying what wakes it:
// on FLUX.1 at its defaults it was 150 px of the first screen, inert. Its
// place changes as the sheet closes (where True CFG is set), with a status
// or a Reuse, never while the sheet is open or the box has the focus, so
// nothing moves under a finger; `now` is the tier's own sync.
function placePhoneNegative({ now = false } = {}) {
  if (!isPhone()) return;
  if (!now && ($('more-sheet').open || els.negField.contains(document.activeElement))) return;
  const body = $('more-sheet-body');
  if (!els.negative.disabled) phoneHome(els.negField);
  else if (els.negField.parentNode !== body) phoneMove(els.negField, body, body.firstChild);
  syncPhoneNegativeFold();
}

// Under the prompt the negative is one 48 px fold row, shut at first: its
// name and its first words ("none" while empty), where its caption and
// two-row box were 200 px of an SDXL checkpoint's first screen and pushed
// More options under the bars. A press opens it to its note and box, the row
// staying as their head (a press there shuts it); a write of words into it
// - Reuse, Load settings from PNG, a setup's Restore, the tag helper's
// negative, a snippet, a finding's rewrite - opens it first, through
// `replaceBoxText` (analysis.js), so the write is typing's and the change is
// seen, and so does Check selecting words in it. In the sheet it is the
// field it always was. The row is the phone's alone: made once, put in the
// field while it is under the prompt on the phone, and taken out otherwise.
let phoneNegFold = null;
let phoneNegOpen = false;

function syncPhoneNegativeFold() {
  const fold = phoneNegFold;
  if (!fold) return;
  const field = els.negField;
  if (!isPhone() || field.parentNode === $('more-sheet-body')) {
    if (fold.isConnected) fold.remove();
    delete field.dataset.fold;
    return;
  }
  if (field.firstElementChild !== fold) field.prepend(fold);
  field.dataset.fold = phoneNegOpen ? 'open' : 'shut';
  fold.setAttribute('aria-expanded', String(phoneNegOpen));
  const text = els.negative.value.replace(/\s+/g, ' ').trim();
  const words = text ? (text.length > 120 ? `${text.slice(0, 120)}…` : text) : 'none';
  const summary = fold.querySelector('.neg-fold-summary');
  if (summary.textContent !== words) summary.textContent = words;
  fold.setAttribute('aria-label', phoneNegOpen ? 'Negative prompt' : `Negative prompt: ${words}`);
}

function togglePhoneNegative() {
  phoneNegOpen = !phoneNegOpen;
  if (!phoneNegOpen) hideNegativeSuggestions();
  syncPhoneNegativeFold();
}

// Opened for words about to be written into it (`text`, empty ones leave it
// shut) or for a selection in it (no `text`). Only from a press's own write.
function openPhoneNegative(text = null) {
  if (phoneNegOpen || !phoneNegFold?.isConnected) return;
  if (text !== null && !String(text).trim()) return;
  phoneNegOpen = true;
  syncPhoneNegativeFold();
}

// Batch's and Repeat's − and +, the phone's alone: − before the box and +
// after it, as a phone's steppers read (Repeat's <label> names its box by
// `for`, so a − ahead of the box in it is not the label's control, and the
// box takes its name from aria-label while they are in, or the label's
// content would name it "Repeat − +"); taken out off the phone. A press
// steps the box as its own arrows would and sends its input and change, so
// syncRuns and the remembered panel follow; at the box's least or most the
// button is held, titled with why, and a press says it.
const phoneSteppers = [];

function makePhoneStepper(input, name) {
  const s = { input, name };
  const button = (dir, glyph, label) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ghost phone-step';
    b.textContent = glyph;
    b.dataset.label = label;
    b.setAttribute('aria-label', label);
    b.addEventListener('click', () => stepPhoneNumber(s, dir, b));
    return b;
  };
  s.minus = button(-1, '−', `${name}: one fewer`);
  s.plus = button(1, '+', `${name}: one more`);
  phoneSteppers.push(s);
}

function stepPhoneNumber(s, dir, b) {
  if (b.getAttribute('aria-disabled') === 'true') { sayPhoneHeld(b, `${b.title}.`); return; }
  try {
    if (dir > 0) s.input.stepUp(); else s.input.stepDown();
  } catch { return; }
  s.input.dispatchEvent(new Event('input', { bubbles: true }));
  s.input.dispatchEvent(new Event('change', { bubbles: true }));
}

function syncPhoneSteppers() {
  for (const s of phoneSteppers) {
    const text = s.input.value.trim();
    const value = text === '' ? +s.input.min : +text;
    const at = (b, held) => {
      b.setAttribute('aria-disabled', String(!!held));
      b.title = held || b.dataset.label;
    };
    at(s.minus, Number.isFinite(value) && value <= +s.input.min ? `${s.name} is at its least, ${s.input.min}` : '');
    at(s.plus, Number.isFinite(value) && value >= +s.input.max ? `${s.name} is at its most, ${s.input.max}` : '');
  }
}

// A held press's why: in the sheet, in its foot's line, where the press
// was; in the panel, the panel's line above the bars (sayAtGenerate).
function sayPhoneHeld(at, words) {
  if ($('more-sheet').contains(at)) { setMoreNote(words); return; }
  showError(words, true);
  sayAtGenerate();
}

// The sheet's foot line: unhidden before it is written, so it is read out.
function setMoreNote(words) {
  const note = $('more-sheet-note');
  note.hidden = !words;
  note.textContent = words;
}

// The row's words - steps, CFG and the seed, then whatever else the sheet
// holds that is set - and its dot while anything there is off the
// checkpoint's defaults (the profile's steps, CFG and Mu; a seed, a second
// batch, a size of no preset, a LoRA or a reference, Then refine, the step
// cache, an open grid, or a negative prompt waiting in the sheet). Notify is
// the browser's, not the picture's, and counts for neither. Then the
// steppers' holds. From the panel's input and change, syncCfg and syncRuns
// (the page's own writes: a status, Reuse, Reset, the dice), the folds and
// Then refine's opener, and the sheet's close.
function syncMoreOptions() {
  if (!isPhone()) return;
  syncPhoneSteppers();
  syncPhoneNegativeFold();
  const d = lastDefaults;
  const val = (el) => el.value.trim();
  const num = (text) => (text === '' ? null : +text);
  const steps = val(els.steps) || (d ? String(d.steps) : '');
  const cfg = val(els.cfg) || (d ? String(d.guidance_scale) : '');
  const seed = val(els.seed);
  const runs = Math.max(1, +els.runs.value || 1);
  const refs = referenceIds.length;
  const negativeWaits = els.negField.parentNode === $('more-sheet-body') && val(els.negative) !== '';
  const extra = [
    runs > 1 ? `repeat ×${runs}` : '',
    trueCfgSupported && val(els.trueCfg) ? `True CFG ${val(els.trueCfg)}` : '',
    muSupported && val(els.mu) && num(val(els.mu)) !== (d?.mu ?? null) ? `mu ${val(els.mu)}` : '',
    sizePresetMatch() ? '' : `${els.width.value} × ${els.height.value}`,
    els.lora.value || pendingLora ? 'LoRA' : '',
    refs ? `${refs} reference${refs === 1 ? '' : 's'}` : '',
    thenRefineOn() ? 'then refine' : '',
    stepCacheValue() ? 'step cache' : '',
    els.gridSection.open ? 'grid' : '',
    negativeWaits ? 'negative' : '',
  ].filter(Boolean);
  const changed = extra.length > 0 || seed !== ''
    || (d && ((val(els.steps) !== '' && +val(els.steps) !== +d.steps)
      || (val(els.cfg) !== '' && +val(els.cfg) !== +d.guidance_scale)));
  const words = [steps ? `${steps} steps` : '', cfg ? `CFG ${cfg}` : '', `seed ${seed || 'random'}`, ...extra]
    .filter(Boolean).join(' · ');
  const summary = $('more-options-summary');
  if (summary.textContent !== words) summary.textContent = words;
  $('more-options-dot').hidden = !changed;
}

// The sheet: the screen across, as every dialog on the phone, its own
// history entry (Back closes it), Done or Escape too. Its controls are the
// panel's own, there for the phone's life, so it opens on what they hold.
function openMoreSheet() {
  const sheet = $('more-sheet');
  if (!isPhone() || sheet.open) return;
  setMoreNote('');
  sheet.showModal();
}

function moreSheetClosed() {
  setMoreNote('');
  placePhoneNegative();
  syncMoreOptions();
}

// Grid search opened by its row's press in the sheet (revealGridRun): its
// row is brought up under the sheet's title, its rows under it in sight -
// the row is the sheet's last, and opening it brings the foot with Run grid
// up over it. The press's own scroll, as on a PC, where it brings Run grid
// into view; never by itself.
function revealPhoneGrid() {
  revealInMoreSheet(els.gridSection);
}

// A fold in the open sheet brought up under its title by a press (Grid
// search's row, the Size chip's Custom dimensions…).
function revealInMoreSheet(el) {
  const sheet = $('more-sheet');
  if (!sheet.open || !sheet.contains(el)) return;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  el.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
}

// A press elsewhere that hands the focus to a control in the closed sheet -
// Generate's refusal of an unfinished number, Expand's of a size - opens
// it, the refusal's words in its foot, where the focus lands.
function openMoreOptionsFor(el, words = '') {
  if (!isPhone() || !$('more-sheet').contains(el)) return;
  // The box is the point: its focus, and the keyboard, stay (phone-sheets.js).
  keepSheetFocus($('more-sheet'));
  openMoreSheet();
  setMoreNote(words);
}

function initPhone() {
  for (const node of [els.generate, ...phoneResultNodes(), ...phoneQueueNodes(), ...moreSheetMoved()]) {
    const mark = document.createComment(` ${node.id || node.className}'s place off the phone (phone.js) `);
    node.before(mark);
    phoneHomes.set(node, mark);
  }
  initPhoneCompose();
  initPhoneGallery();
  initPhoneViewer();
  initPhoneSheets();
  makePhoneStepper(els.batch, 'Batch');
  makePhoneStepper(els.runs, 'Repeat');
  phoneNegFold = document.createElement('button');
  phoneNegFold.type = 'button';
  phoneNegFold.className = 'neg-fold';
  phoneNegFold.id = 'neg-fold';
  phoneNegFold.setAttribute('aria-controls', 'negative');
  const name = document.createElement('span');
  name.className = 'neg-fold-name';
  name.textContent = 'Negative prompt';
  const summary = document.createElement('span');
  summary.className = 'neg-fold-summary';
  phoneNegFold.append(name, summary);
  phoneNegFold.addEventListener('click', togglePhoneNegative);
  $('more-options').addEventListener('click', openMoreSheet);
  $('more-sheet-done').addEventListener('click', () => $('more-sheet').close());
  $('more-sheet').addEventListener('close', moreSheetClosed);
  const panel = document.querySelector('.panel');
  for (const type of ['input', 'change']) panel.addEventListener(type, syncMoreOptions);
  $('more-sheet-body').addEventListener('input', () => setMoreNote(''));
  for (const fold of [els.gridSection, els.customSize, els.conditioningSection]) fold.addEventListener('toggle', syncMoreOptions);
  new MutationObserver(syncMoreOptions).observe(els.thenRefineOpen, { childList: true, characterData: true, subtree: true });
  $('phone-dock').addEventListener('click', pressPhoneDock);
  $('phone-queue').addEventListener('click', openQueueSheet);
  $('queue-sheet-close').addEventListener('click', () => $('queue-sheet').close());
  $('queue-sheet').addEventListener('close', queueSheetClosed);
  const dockWatch = new MutationObserver(syncPhoneDock);
  dockWatch.observe(els.viewer, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
  dockWatch.observe(els.recent, { childList: true, subtree: true });
  dockWatch.observe(els.topbarLine, { attributes: true, attributeFilter: ['hidden', 'class'] });
  dockWatch.observe(els.topbarFill, { attributes: true, attributeFilter: ['style'] });
  const queueWatch = new MutationObserver(syncPhoneQueue);
  queueWatch.observe(els.progress, { attributes: true, attributeFilter: ['hidden'] });
  queueWatch.observe(els.queueNote, { attributes: true, attributeFilter: ['hidden'] });
  queueWatch.observe(els.queueList, { childList: true });
}
