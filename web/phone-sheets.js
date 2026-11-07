// AuraDream's page script: the phone's sheets - what every dialog does as a
// sheet across the screen (styles/phone-sheets.css draws them), beyond the
// sheet's own markup: its head's ✕ where the dialog had only a Cancel at
// its foot, no keyboard thrown up as it opens, and the keyboard's inset,
// so a sheet's sticky action row rides above the keyboard rather than
// behind it; and on the phone the long explanations in folds of their own.
// Declarations only; phone.js calls initPhoneSheets at boot and
// syncPhoneSheets whenever the tier changes.
'use strict';

// A field that brings up the keyboard when it has the focus.
const KEYBOARD_FIELD = 'textarea, [contenteditable="true"], '
  + 'input:not([type="checkbox"], [type="radio"], [type="range"], [type="button"], [type="submit"], [type="file"], [type="color"])';

// A dialog just opened on the phone with the focus in a text box - its
// `autofocus`, or the opener's own `.focus()` in the same task (Refine's
// Strength, Swap tags' first Find, More seeds' count, a setup's name): the
// keyboard came up over half the sheet and its actions before anything was
// read. The focus goes to the head's ✕ instead (else the sheet itself); a
// press in the box brings the keyboard when it is wanted. On a PC and the
// tablet the focus stays where the dialog put it.
function settlePhoneSheetFocus(dialog) {
  if (!isPhone() || !dialog.open || dialog === els.lightbox || sheetFocusKept === dialog) return;
  const at = document.activeElement;
  if (!at || !dialog.contains(at) || !at.matches(KEYBOARD_FIELD)) return;
  const close = [...dialog.querySelectorAll(':scope > .dialog-head button')].find(isShown);
  if (close) { close.focus({ preventScroll: true }); return; }
  if (!dialog.hasAttribute('tabindex')) dialog.tabIndex = -1;
  dialog.focus({ preventScroll: true });
}

// A sheet opened to take the focus to a box on purpose - a refusal of what
// that box holds, which opens More options on it (openMoreOptionsFor) -
// keeps it there, with the keyboard, until it closes.
let sheetFocusKept = null;
function keepSheetFocus(dialog) { sheetFocusKept = dialog; }

// The same for an opener that focuses its box once its list has come
// (Saved setups' name, Saved searches', the snippets'): a text box given
// the focus by the page - no press or key since the sheet opened - within
// two seconds of the opening.
let sheetOpened = null;
let sheetInputAt = -Infinity;
function noteSheetInput() { sheetInputAt = performance.now(); }
function lateSheetFocus(event) {
  const opened = sheetOpened;
  if (!opened || !isPhone() || !event.target.matches?.(KEYBOARD_FIELD)) return;
  if (performance.now() - opened.at > 2000 || sheetInputAt >= opened.at) return;
  if (event.target.closest('dialog') !== opened.dialog) return;
  settlePhoneSheetFocus(opened.dialog);
}

// The head's ✕ (`data-sheet-close`, drawn on the phone alone) closes as
// Escape and Back do: the dialog's `cancel` first, which a dialog busy
// sending refuses, or which closes it by its own path.
function pressSheetClose(event) {
  const button = event.target.closest?.('[data-sheet-close]');
  const dialog = button?.closest('dialog');
  if (!dialog?.open) return;
  if (dialog.dispatchEvent(new Event('cancel', { cancelable: true })) && dialog.open) dialog.close();
}

// Fit touch sheets to the visible viewport itself. Safari can resize
// innerHeight AND pan a fixed dialog while the keyboard opens; subtracting
// offsetTop from a guessed keyboard height lost the phone's last 42 px.
// The top compensates for that pan, and the height is what the keys leave.
// Page pinch zoom keeps the ordinary layout rather than resizing its forms.
function syncSheetViewport() {
  const vv = window.visualViewport;
  const root = document.documentElement.style;
  for (const [name, value] of [['--sheet-viewport-top', vv?.offsetTop], ['--sheet-viewport-height', vv?.height]]) {
    if (!touchQuery.matches || !vv || vv.scale !== 1) { root.removeProperty(name); continue; }
    const pixels = `${value}px`;
    if (root.getPropertyValue(name) !== pixels) root.setProperty(name, pixels);
  }
}

// The long explanations a sheet opened on (P53): Swap tags' matching rules,
// Mass refine's what-each-image-keeps, Inpaint's and Outpaint's note on the
// strength - five lines before the controls. On the phone each is moved
// whole into a shut fold of its own in its place, "How it works" (a
// `details.section`, drawn as every fold on the phone), and home again off
// it; the words, and what writes them, are unchanged. A fold whose words
// are empty is not drawn.
const phoneWhyNodes = () => [$('swap-hint'), document.querySelector('.rb-hint'),
  els.inpaintStrengthNote, $('outpaint-strength-note')];
const phoneWhyFolds = new Map();

function syncPhoneWhy() {
  for (const [node, fold] of phoneWhyFolds) {
    if (isPhone()) { if (node.parentNode !== fold) phoneMove(node, fold); } else phoneHome(node);
    fold.hidden = !isPhone();
  }
}

function initPhoneWhy() {
  for (const node of phoneWhyNodes()) {
    const mark = document.createComment(` ${node.id || node.className}'s place off the phone (phone-sheets.js) `);
    node.before(mark);
    phoneHomes.set(node, mark);
    const fold = document.createElement('details');
    fold.className = 'section phone-why';
    fold.hidden = true;
    const summary = document.createElement('summary');
    summary.textContent = 'How it works';
    fold.append(summary);
    mark.after(fold);
    phoneWhyFolds.set(node, fold);
  }
}

// The Models sheet (P45): on the phone one scroller - the options, the
// checkpoints, their pictures and Load automatically - with a sticky foot
// holding what the sheet does: its note (what it says of a press, and of
// the load), Apply & save and Switch & load, and the topbar's Load and
// Unload, which the phone's topbar does not draw (its ⋯ presses them). The
// footer's half-screen of pictures and buttons had left the checkpoints a
// 439 px box of their own to scroll.
const modelsFootNodes = () => [els.modelPickerNote, document.querySelector('.model-actions'), els.load, els.unload];

function syncPhoneModels() {
  const foot = $('models-foot');
  if (isPhone()) {
    for (const node of modelsFootNodes()) if (node.parentNode !== foot) phoneMove(node, foot);
  } else {
    for (const node of modelsFootNodes()) phoneHome(node);
  }
}

// Saved setups', Saved searches' and Prompt snippets' lists (P48): a
// native list box six rows tall, which iOS draws as a one-line picker and
// which held its own scroll inside the sheet's. On the phone each is drawn
// as rows of a finger's 56 px from the select's own options - the select
// stays the list, hidden, what a row's press sets (its `change`, as a
// click in it on a PC), and what every writer writes - and redrawn when its
// options or its choice change.
const phoneListSelects = () => [els.setupsList, els.savedSearchList, els.snippetsList];
const phoneLists = new Map();

function phoneListSignature(select) {
  return [...select.options].map((o) => `${o.value}\u0001${o.textContent}\u0001${o.selected}\u0001${o.disabled}\u0001${o.title}`)
    .join('\u0002');
}

function drawPhoneList(select) {
  const entry = phoneLists.get(select);
  const signature = phoneListSignature(select);
  if (entry.signature === signature) return;
  entry.signature = signature;
  const rows = [...select.options].map((option) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'phone-list-row';
    row.setAttribute('role', 'option');
    // An option of no value is the list's line ("No saved setups in this
    // view"), never a choice.
    const line = option.disabled || option.value === '';
    row.setAttribute('aria-selected', String(option.selected && !line));
    if (line) row.setAttribute('aria-disabled', 'true');
    row.dataset.value = option.value;
    row.textContent = option.textContent;
    if (option.title) row.title = option.title;
    return row;
  });
  entry.list.replaceChildren(...rows);
}

function pressPhoneList(event) {
  const row = event.target.closest('.phone-list-row');
  if (!row || row.getAttribute('aria-disabled') === 'true') return;
  const select = [...phoneLists].find(([, e]) => e.list.contains(row))?.[0];
  if (!select || select.disabled) return;
  select.value = row.dataset.value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
  drawPhoneList(select);
}

function initPhoneLists() {
  for (const select of phoneListSelects()) {
    const list = document.createElement('div');
    list.className = 'phone-list';
    list.setAttribute('role', 'listbox');
    list.setAttribute('aria-label', select.getAttribute('aria-label') || 'List');
    list.addEventListener('click', pressPhoneList);
    (select.closest('label') ?? select).after(list);
    phoneLists.set(select, { list, signature: null });
    drawPhoneList(select);
    // Its options, and whatever its dialog writes as a choice changes (a
    // choice set by `.value` changes no attribute; the dialog's own lines
    // are written with it), looked at once per task.
    let queued = false;
    const look = () => { if (queued) return; queued = true; queueMicrotask(() => { queued = false; drawPhoneList(select); }); };
    new MutationObserver(look).observe(select.closest('dialog') ?? select,
      { childList: true, subtree: true, characterData: true, attributes: true });
    select.addEventListener('change', look);
  }
}

// From syncPhoneHosts (phone.js): at boot and on every change of the tier.
function syncPhoneSheets() {
  syncPhoneWhy();
  syncPhoneModels();
  syncSheetViewport();
}

function initPhoneSheets() {
  initPhoneWhy();
  initPhoneLists();
  for (const node of modelsFootNodes()) {
    const mark = document.createComment(` ${node.id || node.className}'s place off the phone (phone-sheets.js) `);
    node.before(mark);
    phoneHomes.set(node, mark);
  }
  document.addEventListener('click', pressSheetClose);
  // Each dialog's opening, after the task that opened it has put its focus.
  const watch = new MutationObserver((records) => {
    for (const r of records) {
      if (!r.target.open && sheetFocusKept === r.target) sheetFocusKept = null;
      if (r.target.open && r.oldValue === null) {
        const opened = { dialog: r.target, at: performance.now() };
        sheetOpened = opened;
        queueMicrotask(() => settlePhoneSheetFocus(r.target));
        // And a look every 100 ms of the two seconds, for a focus that
        // came with no focusin (a page without the window's focus).
        const look = () => {
          if (sheetOpened !== opened || sheetInputAt >= opened.at || performance.now() - opened.at > 2000) return;
          settlePhoneSheetFocus(opened.dialog);
          setTimeout(look, 100);
        };
        setTimeout(look, 100);
      }
    }
  });
  for (const dialog of document.querySelectorAll('dialog')) {
    watch.observe(dialog, { attributes: true, attributeFilter: ['open'], attributeOldValue: true });
  }
  document.addEventListener('pointerdown', noteSheetInput, true);
  document.addEventListener('keydown', noteSheetInput, true);
  document.addEventListener('focusin', lateSheetFocus);
  window.visualViewport?.addEventListener('resize', syncSheetViewport);
  window.visualViewport?.addEventListener('scroll', syncSheetViewport);
  syncSheetViewport();
}
