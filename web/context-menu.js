'use strict';

// The right-click menu on an image: a gallery card, a result tile in the
// viewer, the lightbox's picture. Its items are the image's own buttons -
// pressed for you, held and titled as they are, so a held item still says
// why when chosen - plus what no button does: copy the prompt, the seed or
// the picture, and open the PNG in a tab. A picked card - in select mode or
// picked with Ctrl or Shift outside it - gets the picks' actions instead. Shift with the right button, and every text
// box, keep the browser's own menu; the Menu key or Shift+F10 opens this one
// on the image focus is in. One menu, moved into the lightbox while that is
// up: a modal dialog leaves the page under it inert.

const ctxMenu = { host: null, opener: null, keyAt: -Infinity };

// What a right-click or the Menu key at `target` is over: the lightbox's
// stage, a live gallery card, a viewer result. Null for anything else, the
// recycle bin's cards and tombstones included.
function contextTarget(target) {
  if (!target?.closest) return null;
  if (els.lightbox.open) {
    if (!target.closest('#lb-stage')) return null;
    const rec = lbList[lbIndex];
    return rec ? { kind: 'lightbox', rec, host: els.lbStage } : null;
  }
  const card = target.closest('#gallery .card[data-id]');
  if (card) {
    const rec = galleryRecords.find((r) => r.id === +card.dataset.id);
    return rec && !rec.binned_at && !card.classList.contains('gone') ? { kind: 'card', rec, host: card } : null;
  }
  const figure = target.closest('#viewer figure[data-id]');
  if (figure) {
    const rec = viewerRecords.find((r) => r.id === +figure.dataset.id);
    const tile = figure.querySelector('.tile');
    return rec && tile?.querySelector('img') ? { kind: 'result', rec, host: tile } : null;
  }
  return null;
}

// An item from a button of the page: its words here, its hold and reason
// from the button as they are now, and a choice is the button's press.
// Nothing for a button that is not there, or hidden on purpose.
function buttonItem(label, button, hint = '') {
  if (!button || button.hidden) return null;
  const held = button.getAttribute('aria-disabled') === 'true';
  return { label, hint, held, title: button.title, run: () => button.click() };
}

// An item held as a button built for it would be, by that button's own
// sync function (Outpaint… and ❐ where the place has no such button).
function syncedItem(label, sync, run) {
  const scratch = document.createElement('button');
  sync(scratch);
  return { label, held: scratch.getAttribute('aria-disabled') === 'true', title: scratch.title, run };
}

const CLIPBOARD_NONE = 'The browser opens the clipboard only to a page at 127.0.0.1 or localhost';

function copyItem(label, title, write, host, said) {
  const held = !navigator.clipboard || !window.isSecureContext;
  return {
    label, held, title: held ? CLIPBOARD_NONE : title,
    run: async () => {
      if (held) { refuseInCard(host, `${CLIPBOARD_NONE}.`, 'No clipboard here'); return; }
      try {
        await write();
        cardNote(host, said);
      } catch (err) {
        refuseInCard(host, `Could not copy: ${err?.message || err}.`, 'Copy failed');
      }
    },
  };
}

// The PNG as one clipboard item: the fetch is handed over as a promise, so
// the write starts inside the click that asked for it.
function copyImage(rec) {
  const png = fetch(rec.url).then((res) => {
    if (!res.ok) throw new Error(`the image answered ${res.status}`);
    return res.blob();
  }).then((blob) => (blob.type === 'image/png' ? blob : new Blob([blob], { type: 'image/png' })));
  return navigator.clipboard.write([new ClipboardItem({ 'image/png': png })]);
}

function imageItems({ kind, rec, host }) {
  const lightbox = kind === 'lightbox';
  const card = kind === 'card' ? host : null;
  // A result's buttons are its figure's: the tile's row, and the column
  // beside it from 1800 px.
  const own = lightbox ? null : (card ?? host.closest('figure'));
  const find = (selector) => own?.querySelector(selector);
  const starred = !!rec.favorite;
  const at = referenceAt(rec.id);
  return [
    !lightbox && {
      label: 'Open', title: `Open #${rec.id} in the lightbox`,
      run: () => (card ? openLightbox(rec, galleryRecords) : openViewerImage(rec)),
    },
    null,
    buttonItem(starred ? '★ Unstar' : '☆ Star', lightbox ? els.lbFav : find('.star'), kind === 'result' ? '' : 'F'),
    buttonItem('↻ Reuse settings', lightbox ? els.lbReuse : find('.reuse-settings')),
    {
      label: 'Use seed', title: `Put ${rec.seed} in the Seed box`,
      run: () => { useSeed(rec.seed); cardNote(host, lightbox ? `Seed ${rec.seed} is in the Seed box.` : 'Seed set'); },
    },
    null,
    buttonItem('⤢ Refine…', lightbox ? els.lbRefine : find('.refine-tool')),
    buttonItem('✎ Inpaint…', lightbox ? els.lbInpaint : find('.inpaint-tool')),
    lightbox ? buttonItem('⬚ Outpaint…', els.lbOutpaint)
      : syncedItem('⬚ Outpaint…', syncOutpaintTool, () => outpaintFromCard(rec, host)),
    syncedItem(at >= 0 ? `❐ Remove reference ${at + 1}` : '❐ Use as reference',
      (b) => syncReferenceTool(b, rec.id), () => toggleReference(host, rec)),
    { label: '⊞ Show run as sheet', title: 'Show the run that made this as a sheet', run: () => showRecordRun(rec) },
    null,
    copyItem('Copy prompt', 'Copy the prompt as text', () => navigator.clipboard.writeText(rec.prompt), host, 'Prompt copied'),
    copyItem('Copy seed', `Copy ${rec.seed}`, () => navigator.clipboard.writeText(String(rec.seed)), host, 'Seed copied'),
    copyItem('Copy image', 'Copy the full-size PNG as a picture', () => copyImage(rec), host, 'Image copied'),
    {
      label: 'Open PNG in new tab', title: 'The full-size PNG, with its settings in its text chunks',
      run: () => window.open(rec.url, '_blank', 'noopener'),
    },
    null,
    lightbox && buttonItem('↶ Restore', els.lbRestore),
    lightbox ? buttonItem(rec.binned_at ? '⌦ Delete permanently' : '⌦ Move to recycle bin', els.lbDelete, 'Del')
      : buttonItem('⌦ Move to recycle bin', find('.del')),
  ];
}

// A picked card in select mode: what the footer does with the picks, as the
// footer would - the Picked menu's items and the buttons after it.
function pickedItems() {
  const n = selection.size;
  const pick = (act) => {
    const li = els.pickedMenu.querySelector(`[data-picked="${act}"]`);
    const held = els.pickedBtn.getAttribute('aria-disabled') === 'true';
    return { label: li.textContent, held, title: held ? els.pickedBtn.title : li.title, run: () => pickedChoose(li) };
  };
  return [
    { heading: n === 1 ? '1 picked' : `${n} picked` },
    pick('star'), pick('unstar'), pick('bin'),
    null,
    buttonItem('Swap tags…', els.swap),
    buttonItem('Refine…', els.refineBatch),
    buttonItem('More seeds…', els.moreSeeds),
  ];
}

// The items in order, a null between groups; no group's line at either end
// or doubled where a group came out empty.
function renderContextMenu(items) {
  const menu = els.contextMenu;
  menu.replaceChildren();
  let gap = false;
  for (const item of items) {
    if (!item) { gap = menu.childElementCount > 0; continue; }
    if (gap) {
      const line = document.createElement('li');
      line.setAttribute('role', 'separator');
      menu.append(line);
      gap = false;
    }
    const li = document.createElement('li');
    if (item.heading) {
      li.className = 'heading';
      li.setAttribute('role', 'presentation');
      li.textContent = item.heading;
      menu.append(li);
      continue;
    }
    li.setAttribute('role', 'menuitem');
    li.tabIndex = -1;
    li.title = item.title ?? '';
    if (item.held) li.setAttribute('aria-disabled', 'true');
    const label = document.createElement('span');
    label.textContent = item.label;
    li.append(label);
    if (item.hint) {
      const key = document.createElement('kbd');
      key.textContent = item.hint;
      li.append(key);
      li.setAttribute('aria-keyshortcuts', item.hint === 'Del' ? 'Delete' : item.hint);
    }
    li.ctxRun = item.run;
    menu.append(li);
  }
}

// At the point, or for the keyboard at the focused element's corner; kept
// inside the window, turned up or left where it would run off it.
function openContextMenu(found, x, y, byKey) {
  const menu = els.contextMenu;
  closeContextMenu(false);
  const picked = found.kind === 'card' && selection.has(found.rec.id);
  renderContextMenu(picked ? pickedItems() : imageItems(found));
  menu.setAttribute('aria-label', picked ? 'Picked images' : `#${found.rec.id}`);
  const home = els.lightbox.open ? els.lightbox : document.body;
  if (menu.parentElement !== home) home.append(menu);
  ctxMenu.host = found.host;
  ctxMenu.opener = document.activeElement;
  menu.style.left = '0px';
  menu.style.top = '0px';
  menu.hidden = false;
  const { width, height } = menu.getBoundingClientRect();
  const left = x + width + 4 > innerWidth ? Math.max(4, x - width) : x;
  const top = y + height + 4 > innerHeight ? Math.max(4, y - height) : y;
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
  const first = menu.querySelector('[role="menuitem"]');
  if (byKey && first) first.focus({ preventScroll: true });
  else menu.focus({ preventScroll: true });
}

// `refocus`: back to where focus was when it opened - after Escape or a
// choice, not after a press somewhere else, which has put focus there.
function closeContextMenu(refocus = true) {
  const menu = els.contextMenu;
  if (menu.hidden) return;
  const inside = menu.contains(document.activeElement);
  menu.hidden = true;
  const opener = ctxMenu.opener;
  ctxMenu.host = null;
  ctxMenu.opener = null;
  if (refocus && inside && opener?.isConnected && opener !== document.body) opener.focus({ preventScroll: true });
}

function chooseContextItem(li) {
  const run = li?.ctxRun;
  if (!run) return;
  closeContextMenu();
  void run();
}

function contextMenuEvent(e) {
  // The menu the Menu key opened already (contextMenuKey) is this event's.
  if (performance.now() - ctxMenu.keyAt < 400) { e.preventDefault(); return; }
  if (e.target.closest?.('#context-menu')) { e.preventDefault(); return; }
  if (e.shiftKey || e.target.closest?.('input, textarea, select, [contenteditable="true"]')) return;
  const found = contextTarget(e.target);
  if (!found) { closeContextMenu(false); return; }
  e.preventDefault();
  openContextMenu(found, e.clientX, e.clientY, false);
}

function contextMenuKey(e) {
  const asked = e.key === 'ContextMenu' || (e.key === 'F10' && e.shiftKey);
  if (!asked || e.ctrlKey || e.altKey || e.metaKey) return;
  // In the lightbox, from any of its controls but a field: its picture is
  // what the menu is about, and the stage takes no focus of its own.
  const at = document.activeElement;
  const field = at?.closest?.('input, textarea, select, #context-menu');
  const found = contextTarget(els.lightbox.open && els.lightbox.contains(at) && !field ? els.lbStage : at);
  if (!found || field) return;
  e.preventDefault();
  ctxMenu.keyAt = performance.now();
  const box = (found.host.contains(at) ? at : found.host).getBoundingClientRect();
  openContextMenu(found, box.left + Math.min(16, box.width / 2), box.top + Math.min(16, box.height / 2), true);
}

// The menu's keys are a menu's, as the Picked menu's are, and go no further:
// its Escape is not the lightbox's.
function contextMenuKeydown(e) {
  if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeContextMenu(); return; }
  if (e.key === 'Tab') { e.preventDefault(); e.stopPropagation(); closeContextMenu(); return; }
  if ((e.key === 'Enter' || e.key === ' ') && e.target.matches('[role="menuitem"]')) {
    e.preventDefault(); e.stopPropagation(); chooseContextItem(e.target); return;
  }
  const items = [...els.contextMenu.querySelectorAll('[role="menuitem"]')];
  const at = items.indexOf(document.activeElement);
  const next = { ArrowDown: at + 1, ArrowUp: at < 0 ? items.length - 1 : at - 1, Home: 0, End: items.length - 1 }[e.key];
  e.stopPropagation();
  if (next === undefined || !items.length) return;
  e.preventDefault();
  items[(next + items.length) % items.length].focus({ preventScroll: true });
}

// A press anywhere else closes it. On the lightbox's stage the press is the
// menu's alone: a click there closes the lightbox, and dismissing the menu
// should not.
function contextMenuPointerDown(e) {
  if (els.contextMenu.hidden || els.contextMenu.contains(e.target)) return;
  const stage = ctxMenu.host === els.lbStage && e.button === 0 && els.lbStage.contains(e.target);
  closeContextMenu(false);
  if (stage) e.stopPropagation();
}

// Scrolling under it, the window resized or left: it closes, as a native
// one does. Its own scroll, where it is taller than the window, is its own.
function contextMenuAway(e) {
  if (!els.contextMenu.hidden && !els.contextMenu.contains(e.target)) closeContextMenu(false);
}
