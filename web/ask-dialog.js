// AuraDream's page script: the page's own questions (#confirm-dialog), a
// confirm or a typed count beside what asked. Asked by gallery-actions.js,
// loaded right after this file (Delete permanently's question, Delete
// unstarred's count), bin.js (Empty bin, Delete older, Delete permanently),
// generation.js (Re-render picked at a tier) and bundles.js (a saved
// bundle's delete). Declarations only; app.js wires the dialog's close and
// backdrop, and state.js has the shared-global load-order contract.
'use strict';

// The delete's two questions, in a dialog of the page's own (#confirm-dialog)
// beside what was pressed: the browser's confirm and prompt were a light box
// at the top of the window, far from the card whose ⌦ asked, and held the
// whole page while they were up - a headless run waited on one for good.
// `message` is the question's lines as they used to be shown: the first is
// the title, a count's last is its box's label, the rest its paragraphs.
// confirmDialog answers true for Delete and false for Cancel, Escape or a
// press outside it; countDialog the typed text, or null. `okLabel` is the
// OK button's words, "Delete" unless the question names its own (Empty
// recycle bin's "Empty bin"); `destroy` false paints it as an ordinary
// primary, for a question whose OK queues work rather than deletes (Re-render
// picked at a tier). The smokes stand in for them through
// `askConfirm` and `askCount`, and put these back.
let askAnswer = null;
function openAsk(message, { count = false, okLabel = 'Delete', destroy = true } = {}) {
  // One question at a time: an earlier one still up is answered Cancel.
  if (els.confirmDialog.open) {
    const earlier = askAnswer;
    askAnswer = null;
    els.confirmDialog.close('');
    earlier?.('');
  }
  const lines = message.split('\n');
  els.confirmTitle.textContent = lines.shift();
  const label = count ? lines.pop() : '';
  els.confirmText.replaceChildren(...lines.filter((line) => line.trim()).map((line) => {
    const p = document.createElement('p');
    p.textContent = line.trim();
    return p;
  }));
  els.confirmCountField.hidden = !count;
  els.confirmCountLabel.textContent = label;
  els.confirmCount.value = '';
  els.confirmOk.textContent = okLabel;
  els.confirmOk.classList.toggle('destroy', destroy);
  const anchor = document.activeElement !== document.body ? document.activeElement : null;
  return new Promise((resolve) => {
    askAnswer = resolve;
    els.confirmDialog.returnValue = '';
    els.confirmDialog.showModal();
    placeAsk(anchor);
    (count ? els.confirmCount : els.confirmCancel).focus();
  });
}

function confirmDialog(message, { okLabel, destroy } = {}) {
  return openAsk(message, { okLabel, destroy }).then((answer) => answer === 'ok');
}

function countDialog(message) {
  return openAsk(message, { count: true }).then((answer) => (answer === 'ok' ? els.confirmCount.value : null));
}

let askConfirm = confirmDialog;
let askCount = countDialog;

// Beside what asked, clear of the image it asks about - its card or tile, not
// just the button, which sits on the picture: under it, over it, then to its
// right or left, whichever the window has room for, else as near as fits -
// and inside the window. Centred where nothing on the page has focus. On a
// phone it is a sheet from the screen's foot (touch.css), placed by nothing.
function placeAsk(anchor) {
  const dialog = els.confirmDialog;
  for (const side of ['margin', 'top', 'left', 'right', 'bottom']) dialog.style[side] = '';
  if (matchMedia(PHONE_MEDIA).matches) return;
  const near = anchor?.closest?.('.card, .viewer > figure, .sheet-cell, .lb-actions') ?? anchor;
  const at = near?.getBoundingClientRect();
  if (!at || (!at.width && !at.height)) return;
  const box = dialog.getBoundingClientRect();
  const room = 8;
  // The window less its scrollbars: innerWidth counts the page's, and a
  // dialog placed for a card in the right-hand column ran 7 px under it.
  const { clientWidth: width, clientHeight: height } = document.documentElement;
  const clampX = (x) => Math.max(room, Math.min(x, width - box.width - room));
  const clampY = (y) => Math.max(room, Math.min(y, height - box.height - room));
  const centreX = at.left + at.width / 2 - box.width / 2;
  const centreY = at.top + at.height / 2 - box.height / 2;
  const places = [
    [centreX, at.bottom + room, at.bottom + room + box.height <= height - room],
    [centreX, at.top - room - box.height, at.top - room - box.height >= room],
    [at.right + room, centreY, at.right + room + box.width <= width - room],
    [at.left - room - box.width, centreY, at.left - room - box.width >= room],
  ];
  const [x, y] = places.find(([, , fits]) => fits) ?? places[0];
  Object.assign(dialog.style, {
    margin: '0', top: `${clampY(y)}px`, left: `${clampX(x)}px`, right: 'auto', bottom: 'auto',
  });
}

function askClosed() {
  // The close of a question another has replaced, arriving once that one is up.
  if (els.confirmDialog.open) return;
  const answer = askAnswer;
  askAnswer = null;
  answer?.(els.confirmDialog.returnValue);
}

// A press outside the dialog's box - on its backdrop - is Cancel.
function askBackdrop(event) {
  if (event.target !== els.confirmDialog) return;
  const box = els.confirmDialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right
    && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) els.confirmDialog.close('');
}
