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
    await __keyboardUntil(() => d.open && !backGoing);
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
    await __keyboardUntil(() => !d.open && !backGoing);
  }
  return results;
};
