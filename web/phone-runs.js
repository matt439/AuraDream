// AuraDream's page script: runs on a phone (styles/phone-gallery.css draws
// them). The Runs picker, a native select whose one line was cut to "5 Nov,
// 01:35 pm · 4 …", is a row in the Filters sheet opening a sheet of runs -
// each a 56 px row with its first image, when it ran, its count and what
// it varied - built from the select's own options, which stay the one list
// of what can be picked (runs.js; the select is not drawn, its press is
// runListChange's). The run the gallery shows - the search's `run:<id>`,
// however it got there - is a chip in the search's place, its ✕ the
// gallery as it was before. Over a run's sheet, its actions - Finish run and
// More seeds of this run…, the sheet's own buttons moved in whole - are in
// a foot over the tab bar, with a ⋯ for its swipes, Fill width and its
// exports (the sheet's own buttons pressed, not drawn there on the phone).
// Declarations only; phone-gallery.js's initPhoneGallery calls initPhoneRuns,
// syncPhoneGallery syncPhoneRuns; runs.js's syncRunList calls syncPhoneRuns,
// sheet-render.js's drawSheet dropPhoneSheetFoot and placePhoneSheetFoot.
'use strict';

let phoneRunsReady = false;

// From syncRunList (the runs' list, the shown run, the bin), the tier's
// sync, the head's sync and the search's focus: the Filters sheet's Runs
// row, the open runs sheet's rows and the run chip.
function syncPhoneRuns() {
  if (!phoneRunsReady) return;
  const sheet = $('runs-sheet');
  if (!isPhone()) {
    if (sheet.open) sheet.close();
    $('runs-sheet-body').replaceChildren();
    $('phone-run-chip').hidden = true;
    els.search.closest('.gallery-head').removeAttribute('data-run');
    syncPhoneSheetFoot();
    return;
  }
  syncRunsRow();
  syncRunChip();
  if (sheet.open) drawRunsSheet();
  // Back on the phone from a mouse's screen with a run's sheet shown: its
  // actions, home in the controls there, into the foot again, and the foot
  // up (drawSheet's own placing is the same, so this moves nothing then).
  const foot = $('sheet-foot');
  if (sheetRunActs(els.sheetControls).length || foot.hidden === phoneSheetFootUp()) placePhoneSheetFoot();
}

// The run's words for its row and chip: when it began, then how many
// images ("6 Nov, 03:48 · 4 images"); a single image's second and seed. A
// run of today is its time alone ("Today, 4:47 am"), as the chip says it,
// so a list of mostly recent runs reads by the time.
function runWhen(run) {
  const several = run.images !== 1 || run.jobs > 1;
  const at = Date.parse(run.started_at);
  const date = new Date(at);
  const said = date.toDateString() !== new Date().toDateString() ? (several ? when(at) : whenExactly(at))
    : `Today, ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', ...(several ? {} : { second: '2-digit' }) })}`;
  return several ? `${said} · ${plural(run.images, 'image')}` : `${said} · seed ${run.seed}`;
}

// ---------------------------------------------------------- the Runs row

// Filters' Runs row: its line the run shown, or what a pick does; held
// with the picker's own reason (the bin, the list loading or empty), which
// a press says in the sheet's foot.
function syncRunsRow() {
  const row = $('btn-phone-runs');
  const held = binMode || els.runList.disabled;
  const run = shownRun();
  const listed = run && runsListed.get(run);
  const words = held ? phoneWords(els.runList.title)
    : run ? (listed ? runWhen(listed) : `Run ${run}`)
    : 'Lay one run out by its settings';
  row.setAttribute('aria-disabled', String(held));
  if (row.title !== els.runList.title) row.title = els.runList.title;
  const line = row.querySelector('.runs-row-line');
  if (line.textContent !== words) line.textContent = words;
}

function pressRunsRow() {
  const row = $('btn-phone-runs');
  if (row.getAttribute('aria-disabled') === 'true') {
    showError(`${phoneWords(row.title)}.`, true, { slot: els.galleryError });
    return;
  }
  openRunsSheet();
}

// Over Filters' sheet (Back goes back to it), or alone from the run chip;
// its rows drawn before it shows.
function openRunsSheet() {
  drawRunsSheet();
  openGallerySheet($('runs-sheet'));
}

// ---------------------------------------------------------- the runs sheet

// Its rows, from the select's options in their order and groups: a group's
// name over its rows, Retry where the list did not load, else the
// picker's own head ("Loading runs…", "No runs yet"). Drawn afresh only
// when what it would draw changes, so a refresh of the same list moves
// nothing under a finger.
let runsSheetDrawn = '';
function drawRunsSheet() {
  const body = $('runs-sheet-body');
  const shown = shownRun();
  const options = [...els.runList.options];
  const signature = JSON.stringify([shown, binMode, options.map((o) => [o.value, o.text, o.disabled, o.parentElement.label ?? '']),
    [...runsListed.values()].map((r) => r.cover)]);
  const count = $('runs-sheet-count');
  const real = options.filter(realRunOption).length;
  const said = real ? plural(real, 'run') : '';
  if (count.textContent !== said) count.textContent = said;
  if (signature === runsSheetDrawn && body.childElementCount) return;
  runsSheetDrawn = signature;
  const rows = [];
  for (const child of els.runList.children) {
    if (child.tagName === 'OPTGROUP') {
      const head = document.createElement('p');
      head.className = 'gallery-sheet-scope runs-group';
      head.textContent = child.label;
      rows.push(head, ...[...child.children].filter(realRunOption).map((o) => runRow(o, shown)));
    } else if (child.dataset.runsRetry === 'true') {
      rows.push(retryRow(child));
    } else if (realRunOption(child)) {
      rows.push(runRow(child, shown));
    }
  }
  if (!rows.some((r) => r.tagName === 'BUTTON')) {
    const note = document.createElement('p');
    note.className = 'hint runs-empty';
    note.textContent = binMode ? phoneWords(BIN_HELD) : options[0]?.text || 'No runs yet';
    rows.unshift(note);
  }
  body.replaceChildren(...rows);
}

// A run's row: its first image (an empty-set marker for a run with none left),
// then when it ran and its count over what it varied and its prompt. The
// option's own words are its title; the run shown is marked.
function runRow(option, shown) {
  const run = runsListed.get(option.value);
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'run-row';
  row.dataset.run = option.value;
  row.title = option.text;
  if (option.value === shown) {
    row.setAttribute('aria-current', 'true');
    row.classList.add('active');
  }
  const face = document.createElement('span');
  face.className = 'run-face';
  face.setAttribute('aria-hidden', 'true');
  if (run?.cover != null) {
    const img = document.createElement('img');
    img.alt = '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.src = `/api/images/${run.cover}/thumb`;
    face.append(img);
  } else {
    face.classList.add('run-face-empty');
    face.textContent = '∅';
    face.title = run?.images === 0 ? 'No images in this run' : 'No preview image';
  }
  const text = document.createElement('span');
  text.className = 'run-text';
  const top = document.createElement('span');
  top.className = 'run-when';
  top.textContent = run ? runWhen(run) + (run.favorites ? ` · ${run.favorites}★` : '') : option.text;
  text.append(top);
  if (run) {
    const sub = document.createElement('span');
    sub.className = 'run-what';
    sub.textContent = [runAxes(run), promptGist(run.prompt ?? '')].filter(Boolean).join(' · ');
    text.append(sub);
  }
  row.append(face, text);
  return row;
}

function retryRow(option) {
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'run-row runs-retry';
  row.dataset.runsRetry = '';
  row.textContent = option.text;
  row.title = els.runList.title;
  return row;
}

// A row's press: the run picked as the select's own change picks it
// (runListChange: the bin, a list of another scope or a run gone from it
// change nothing), the sheets shut first, so what the screen shows is the
// run's sheet and Back's one entry is its. Retry loads the list again in
// the sheet.
function pressRunsSheet(e) {
  const row = e.target.closest('button.run-row');
  if (!row) return;
  const options = [...els.runList.options];
  if ('runsRetry' in row.dataset) {
    const retry = options.find((o) => o.dataset.runsRetry === 'true');
    if (!retry) return;
    els.runList.selectedIndex = options.indexOf(retry);
    runListChange();
    return;
  }
  if (binMode) { showError(`${phoneWords(BIN_HELD)}.`, true, { slot: els.galleryError }); return; }
  const option = options.find((o) => realRunOption(o) && o.value === row.dataset.run);
  if (!option) return;
  for (const sheet of [$('runs-sheet'), $('gallery-filter-sheet')]) if (sheet.open) closeGallerySheetFor(sheet);
  els.runList.value = option.value;
  runListChange();
}

// ---------------------------------------------------------- the run chip

// The run the gallery shows, in the search's place (the head's
// `data-run`): "Run · 6 Nov, 03:48 · 4 images" and a ✕. Not while the
// search has the focus - what is typed shows as typed - nor in the bin.
function syncRunChip() {
  const run = binMode ? null : shownRun();
  const head = els.search.closest('.gallery-head');
  const up = !!run && document.activeElement !== els.search;
  const chip = $('phone-run-chip');
  chip.hidden = !up;
  head.toggleAttribute('data-run', up);
  if (!up) return;
  const listed = runsListed.get(run);
  const [short, more] = listed ? runChipWords(listed) : [run, ''];
  const open = $('run-chip-open');
  const shortEl = open.querySelector('.run-chip-short');
  const moreEl = open.querySelector('.run-chip-more');
  if (shortEl.textContent !== short) shortEl.textContent = short;
  if (moreEl.textContent !== more) moreEl.textContent = more;
  const full = listed ? runWhen(listed) : run;
  open.setAttribute('aria-label', `Run ${full}: choose another run`);
  open.title = `Run ${full}`;
}

// The chip's words: what fits at 360 px beside "Run" and ✕ - the time of
// a run of today ("3:48 am"), else its day ("5 Nov") - then the rest,
// drawn where the chip has the room (styles/phone-gallery.css).
function runChipWords(run) {
  const at = new Date(Date.parse(run.started_at));
  const today = at.toDateString() === new Date().toDateString();
  const time = at.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  const day = at.toLocaleDateString([], { month: 'short', day: 'numeric' });
  const count = plural(run.images, 'image');
  return today ? [time, ` · ${count}`] : [day, `, ${time} · ${count}`];
}

// ✕: the gallery as it was before the run - the search the run replaced,
// where the page wrote it (a pick, a card's ⊞, Show run), else none - in
// the gallery's one filter transition, which closes the run's sheet.
function clearRunChip() {
  const search = els.search.value.trim();
  const query = sheetReturn && search === sheetReturn.wrote ? sheetReturn.search : '';
  void applyGalleryFilter({ query, favorites: els.favOnly.checked });
}

// ---------------------------------------------------------- the sheet's foot

// The foot is up on a phone, in the Gallery view, over a run's sheet.
const phoneSheetFootUp = () => isPhone() && sheetMode && !els.sheet.hidden && phoneView === 'gallery';

// The sheet's run actions: Finish run and More seeds of this run…, in that
// order (drawSheet's), each the sheet's own node.
const sheetRunActs = (host) => [...host.querySelectorAll(':scope > button[data-run]')];

// Before drawSheet empties the controls: the actions it drew last time,
// moved into the foot, go - it draws them again.
function dropPhoneSheetFoot() {
  for (const node of sheetRunActs($('sheet-foot'))) node.remove();
}

// After drawSheet: on a phone, its run actions into the foot.
function placePhoneSheetFoot() {
  if (!phoneRunsReady) return;
  if (isPhone()) {
    const foot = $('sheet-foot');
    for (const node of sheetRunActs(els.sheetControls)) phoneMove(node, foot, $('sheet-foot-more'));
  }
  syncPhoneSheetFoot();
}

// The foot up or down (`body[data-run-foot]`: the page's foot, its scroll
// padding and --phone-bottom-chrome count it), the gallery's line over it
// while it is up; off the phone, the actions home in the controls, before
// the keyboard's line and the run's.
function syncPhoneSheetFoot() {
  const foot = $('sheet-foot');
  if (!isPhone()) {
    const acts = sheetRunActs(foot);
    const before = els.sheetControls.querySelector(':scope > :is(.sheet-hint, .sheet-run-status)');
    for (const node of acts) phoneMove(node, els.sheetControls, before);
  }
  const up = phoneSheetFootUp();
  foot.hidden = !up;
  document.body.toggleAttribute('data-run-foot', up);
  foot.classList.toggle('alone', !sheetRunActs(foot).length);
  const more = $('sheet-foot-more');
  const none = !sheetFootItems().length;
  more.setAttribute('aria-disabled', String(none));
  more.title = none ? 'Nothing on the sheet to swipe, fit or export' : 'Swipe, Fill width and Export';
  placeGalleryLine();
}

// ⋯'s rows: the sheet's own swipes, Fill width and exports, each pressed
// for you, held and titled as it is - a held one's reason said over the
// foot rather than by its own press (a swipe's goes to the panel's line).
function sheetFootItems() {
  const own = (selector) => els.sheetControls.querySelector(`:scope > ${selector}`);
  const press = (label, button) => {
    const item = buttonItem(label, button);
    return item && {
      ...item,
      run: () => {
        if (button.getAttribute('aria-disabled') === 'true') showError(`${button.title}.`, true, { slot: els.galleryError });
        else button.click();
      },
    };
  };
  const fill = own('.sheet-fill');
  const fillItem = fill && press('Fill width', fill);
  return [
    ...Object.values(SHEET_SWIPES).map((swipe) => press(swipe.label, own(`.${swipe.className}`))),
    // Fill width ticked while it is on, as a phone's menus tick the choice
    // in force.
    fillItem && { ...fillItem, checked: fill.getAttribute('aria-pressed') === 'true' },
    ...[...els.sheetControls.querySelectorAll(':scope > .sheet-export')].map((b) => press(b.textContent, b)),
  ].filter(Boolean);
}

function openSheetFootMenu(event) {
  const button = $('sheet-foot-more');
  if (button.getAttribute('aria-disabled') === 'true') {
    showError(`${button.title}.`, true, { slot: els.galleryError });
    return;
  }
  const box = button.getBoundingClientRect();
  showContextMenu(sheetFootItems(), 'This sheet', button, box.left, box.top, event.detail === 0);
}

function initPhoneRuns() {
  phoneRunsReady = true;
  $('btn-phone-runs').addEventListener('click', pressRunsRow);
  const sheet = $('runs-sheet');
  $('runs-sheet-close').addEventListener('click', () => sheet.close());
  sheet.addEventListener('close', gallerySheetClosed);
  sheet.addEventListener('click', gallerySheetBackdrop);
  sheet.addEventListener('click', pressRunsSheet);
  $('run-chip-open').addEventListener('click', openRunsSheet);
  $('run-chip-clear').addEventListener('click', clearRunChip);
  $('sheet-foot-more').addEventListener('click', openSheetFootMenu);
  els.search.addEventListener('focus', syncRunChip);
  els.search.addEventListener('blur', () => queueMicrotask(syncPhoneRuns));
  // The foot follows the sheet, the view and the tier.
  const watch = new MutationObserver(syncPhoneSheetFoot);
  watch.observe(els.sheet, { attributes: true, attributeFilter: ['hidden'] });
  watch.observe(document.body, { attributes: true, attributeFilter: ['data-view'] });
}
