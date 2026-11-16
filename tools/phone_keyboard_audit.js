// Supplied viewport checks for image pass dialogs. These inspect layout and
// hit targets; they do not emulate a native software keyboard.
'use strict';

window.__keyboardUntil = async (fn, ms = 5000) => {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (fn()) return true;
    await new Promise((resolve) => setTimeout(resolve, 40));
  }
  return false;
};
// Opening a library form also reads its list, redraws the phone's rows and
// settles opening focus. A changed viewport queues its own reveal frame.
// Drain those owners and observe stable rendered geometry before simulating
// the user's focus; two unconditional frames can end in the middle of either.
window.__keyboardIdle = async (dialog, supplied = false) => {
  const end = Date.now() + 5000;
  let previous = '', stable = 0;
  while (Date.now() < end) {
    await new Promise(requestAnimationFrame);
    const libraryReady = dialog.id === 'setups-dialog' ? setupsReady && !setupsPending
      : dialog.id === 'snippets-dialog' ? snippetsReady && !snippetsPending
      : dialog.id === 'saved-searches-dialog' ? savedSearchReady && !savedSearchPending : true;
    const rowsReady = [...phoneLists].every(([select, entry]) => !dialog.contains(select)
      || entry.signature === phoneListSignature(select));
    const box = dialog.getBoundingClientRect();
    const ready = dialog.open && !backGoing && backDepth === 1 && libraryReady && rowsReady
      && !sheetViewportFrame && (!isPhone() || sheetOpened?.dialog === dialog)
      && (!supplied || Math.abs(box.height - window.__keyboardMetrics[0]) < 1);
    const geometry = JSON.stringify([box.toJSON(), dialog.scrollTop, dialog.scrollHeight,
      document.activeElement?.id, getComputedStyle(dialog.querySelector('.dialog-head')).position]);
    stable = ready && geometry === previous ? stable + 1 : 0;
    previous = geometry;
    if (stable >= 2) return true;
  }
  throw new Error(`${dialog.id} did not become idle before keyboard focus`);
};
window.__keyboardViewportDescriptor = Object.getOwnPropertyDescriptor(window, 'visualViewport');
window.__keyboardViewport = (height, offsetTop) => {
  Object.defineProperty(window, 'visualViewport', { configurable: true, value: { height, offsetTop, scale: 1 } });
  syncSheetViewport();
};
window.__keyboardRestore = () => {
  document.activeElement?.blur();
  Object.defineProperty(window, 'visualViewport', __keyboardViewportDescriptor);
  syncSheetViewport();
};
window.__keyboardField = (field) => {
  const d = field.closest('dialog'), box = field.getBoundingClientRect(), dialog = d.getBoundingClientRect();
  const head = d.querySelector('.dialog-head'), actions = d.querySelector('.dialog-actions').getBoundingClientRect();
  const top = getComputedStyle(head).position === 'sticky' ? Math.max(dialog.top, head.getBoundingClientRect().bottom) : dialog.top;
  const beside = box.right <= actions.left || box.left >= actions.right;
  const bottom = beside ? dialog.bottom : Math.min(dialog.bottom, actions.top);
  const hits = [[0.5, 0.5], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]].every(([x, y]) => document.elementFromPoint(box.left + box.width * x, box.top + box.height * y) === field);
  const vv = window.visualViewport;
  return {
    id: field.id, focused: document.activeElement === field,
    clear: document.activeElement === field && box.top >= top - 1 && box.bottom <= bottom + 1 && hits,
    actions: actions.top >= vv.offsetTop - 1 && actions.bottom <= vv.offsetTop + vv.height + 1,
    box: { left: box.left, top: box.top, right: box.right, bottom: box.bottom },
    band: { top, bottom }, hits,
  };
};
window.__keyboardPasses = async () => {
  const cases = [
    { open: openInpaint, dialog: 'inpaint-dialog', fields: ['inpaint-strength', 'inpaint-seed'] },
    { open: openOutpaint, dialog: 'outpaint-dialog', fields: ['outpaint-right', 'outpaint-strength', 'outpaint-seed'] },
    { open: openEdit, dialog: 'edit-dialog', fields: ['edit-seed', 'edit-guidance'] },
  ];
  const results = [];
  for (const test of cases) {
    __keyboardRestore();
    await test.open(galleryRecords[0]);
    const d = document.getElementById(test.dialog);
    if (!await __keyboardUntil(() => d.open && !backGoing)) throw new Error(`${test.dialog} did not open`);
    __keyboardViewport(...window.__keyboardMetrics);
    for (const id of test.fields) {
      const field = document.getElementById(id);
      noteSheetInput();
      field.focus({ preventScroll: true });
      syncSheetViewport();
      await __keyboardUntil(() => __keyboardField(field).clear && __keyboardField(field).actions, 1200);
      results.push(__keyboardField(field));
      // A late source/preview line must not cover a focused field that was
      // already clear. Its arrival is independent of a viewport resize.
      const late = document.createElement('p');
      late.style.height = '28px';
      (field.closest('.field') || field.parentElement).before(late);
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      await __keyboardUntil(() => __keyboardField(field).clear, 1200);
      results.push({ late: true, ...__keyboardField(field) });
      late.remove();
    }
    __keyboardRestore();
    d.querySelector('[id$="-cancel"]').click();
    if (!await __keyboardUntil(() => !d.open && !backGoing)) throw new Error(`${test.dialog} did not close`);
  }
  return results;
};

// Other phone forms can have many actions. In a tiny keyboard viewport the
// form must remain editable; its usual sticky controls return on dismissal.
window.__keyboardOtherForms = async () => {
  const cases = [
    // Steps is in the Sampling fold, shut the first time (UI#51): opened here.
    { open: () => { els.samplingSection.open = true; return openMoreSheet(); }, dialog: 'more-sheet', field: 'steps' },
    { open: openSetups, dialog: 'setups-dialog', field: 'setups-name' },
    { open: openSnippets, dialog: 'snippets-dialog', field: 'snippets-name' },
    { open: openSavedSearches, dialog: 'saved-searches-dialog', field: 'saved-searches-name' },
  ];
  const tick = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  const results = [];
  for (const test of cases) {
    __keyboardRestore();
    await test.open();
    const d = document.getElementById(test.dialog), field = document.getElementById(test.field);
    await __keyboardIdle(d);
    if (field.type === 'text') {
      field.value = 'Keep this draft';
      field.dispatchEvent(new Event('input', { bubbles: true }));
    }
    await __keyboardIdle(d);
    const controls = [...d.querySelectorAll('input, textarea, select, button')];
    const holds = () => controls.filter((c) => c.tagName === 'BUTTON')
      .map((c) => [c.id, c.disabled, c.getAttribute('aria-disabled'), c.hidden]);
    const originalHolds = JSON.stringify(holds()), value = field.value, page = [scrollX, scrollY];
    const clear = () => {
      const b = field.getBoundingClientRect(), r = d.getBoundingClientRect();
      return document.activeElement === field && b.top >= r.top - 1 && b.bottom <= r.bottom + 1
        && b.left >= r.left - 1 && b.right <= r.right + 1
        && [[0.5, 0.5], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
          .every(([x, y]) => document.elementFromPoint(b.left + b.width * x, b.top + b.height * y) === field);
    };
    const geometry = () => ({ active: document.activeElement.id, box: field.getBoundingClientRect().toJSON(),
      dialog: d.getBoundingClientRect().toJSON(), scrollTop: d.scrollTop, maxScroll: d.scrollHeight - d.clientHeight,
      viewportFrame: sheetViewportFrame,
      hit: document.elementFromPoint(field.getBoundingClientRect().left + 10, field.getBoundingClientRect().top + 10)?.id });
    const unchanged = () => field === document.getElementById(test.field) && field.value === value
      && controls.length === d.querySelectorAll('input, textarea, select, button').length
      && controls.every((c, i) => c === d.querySelectorAll('input, textarea, select, button')[i])
      && JSON.stringify(holds()) === originalHolds && scrollX === page[0] && scrollY === page[1];
    __keyboardViewport(...window.__keyboardMetrics);
    await __keyboardIdle(d, true);
    noteSheetInput();
    field.focus({ preventScroll: true });
    syncSheetViewport();
    const fitted = () => Math.abs(d.getBoundingClientRect().height - window.__keyboardMetrics[0]) < 1;
    await tick();
    await __keyboardUntil(() => fitted() && clear(), 1200);
    results.push({ field: test.field, clear: clear(), unchanged: unchanged(),
      supplied: [...window.__keyboardMetrics], height: d.getBoundingClientRect().height,
      headPosition: getComputedStyle(d.querySelector('.dialog-head')).position, geometry: geometry() });
    const late = document.createElement('p');
    late.style.height = '28px';
    (field.closest('.field') || field.parentElement).before(late);
    await tick();
    await __keyboardUntil(clear, 1200);
    const afterLate = { field: test.field, late: true, clear: fitted() && clear(), unchanged: unchanged(), geometry: geometry() };
    late.remove();
    __keyboardRestore();
    await tick();
    const head = d.querySelector('.dialog-head'), actions = d.querySelector('.dialog-actions');
    afterLate.restored = !document.documentElement.hasAttribute('data-short-sheet')
      && !document.documentElement.style.getPropertyValue('--sheet-viewport-height')?.includes('48px')
      && getComputedStyle(head).position === 'sticky' && getComputedStyle(actions).position === 'sticky'
      && field.value === value && JSON.stringify(holds()) === originalHolds;
    results.push(afterLate);
    d.close();
    if (!await __keyboardUntil(() => !d.open && !backGoing && backDepth === 0)) throw new Error(`${test.dialog} did not close`);
  }
  return results;
};

window.__keyboardCountConfirm = async () => {
  __keyboardRestore();
  let answer = 'pending';
  void countDialog('A synthetic count question\nType the count to confirm.').then((value) => { answer = value; });
  const d = els.confirmDialog, field = els.confirmCount;
  if (!await __keyboardUntil(() => d.open && backDepth === 1 && !backGoing)) throw new Error('Count question did not open');
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  field.value = 'not a count';
  field.dispatchEvent(new Event('input', { bubbles: true }));
  const buttons = [...d.querySelectorAll('button')];
  const holds = () => JSON.stringify(buttons.map((b) => [b.id, b.disabled, b.getAttribute('aria-disabled')]));
  const originalHolds = holds(), page = [scrollX, scrollY], results = [];
  const clear = () => {
    const b = field.getBoundingClientRect(), r = d.getBoundingClientRect();
    return document.activeElement === field && b.height >= 44 - 0.5 && b.top >= r.top - 1 && b.bottom <= r.bottom + 1
      && [[0.5, 0.5], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        .every(([x, y]) => document.elementFromPoint(b.left + b.width * x, b.top + b.height * y) === field);
  };
  __keyboardViewport(48, 69);
  noteSheetInput();
  field.focus({ preventScroll: true });
  syncSheetViewport();
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  await __keyboardUntil(clear, 1200);
  const geometry = () => ({ active: document.activeElement.id, box: field.getBoundingClientRect().toJSON(),
    dialog: d.getBoundingClientRect().toJSON(), scrollTop: d.scrollTop, maxScroll: d.scrollHeight - d.clientHeight,
    hit: document.elementFromPoint(field.getBoundingClientRect().left + 10, field.getBoundingClientRect().top + 10)?.id });
  results.push({ field: 'confirm-count', clear: clear(), height: d.getBoundingClientRect().height, geometry: geometry(),
    value: field.value, unchanged: field === els.confirmCount && holds() === originalHolds && scrollX === page[0] && scrollY === page[1] });
  const late = document.createElement('p');
  late.style.height = '28px';
  els.confirmCountField.before(late);
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  await __keyboardUntil(clear, 1200);
  const afterLate = { field: 'confirm-count', late: true, clear: clear(), value: field.value, geometry: geometry(),
    unchanged: field === els.confirmCount && holds() === originalHolds && scrollX === page[0] && scrollY === page[1] };
  late.remove();
  __keyboardRestore();
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  afterLate.restored = !document.documentElement.hasAttribute('data-short-sheet')
    && d.getBoundingClientRect().height > 48 && field.value === 'not a count' && holds() === originalHolds;
  els.confirmCancel.click();
  if (!await __keyboardUntil(() => !d.open && !backGoing && backDepth === 0 && answer !== 'pending')) throw new Error('Count Cancel did not settle');
  afterLate.cancelled = answer === null;
  results.push(afterLate);
  return results;
};
