// Keyboard help describes the existing handlers, grouped by where a key acts.
// Keep this list available before any gallery card or comparison is rendered.
// Native field editing and browser shortcuts keep their own meaning.
'use strict';

// Shared by local handlers and the document dispatcher. A key already owned
// by an editor/menu, or still composing, must never become a second command.
function keyboardEventHeld(event) {
  return event.defaultPrevented || event.isComposing || event.keyCode === 229;
}

function keyboardEditingTarget(target) {
  return !!target?.closest?.('input, textarea, select') || !!target?.isContentEditable;
}

function keyboardCommandModifier(event) {
  return event.ctrlKey || event.metaKey || event.altKey;
}

function keyboardSubmit(event) {
  return event.key === 'Enter' && event.ctrlKey !== event.metaKey
    && !event.altKey && !event.shiftKey;
}

const SHORTCUT_GROUPS = [
  {
    id: 'help',
    title: 'Help',
    rows: [[['?'], 'Open this help outside text fields; Escape closes it first.']],
  },
  {
    id: 'generate',
    title: 'Generate and passes',
    scope: 'Ctrl/Cmd+Enter follows the focused action dialog or Grid field. Help, settings, menus, Bundles and the lightbox never generate underneath.',
    rows: [
      [['Ctrl/Cmd+Enter'], 'Generate; in an open Grid field, run the grid. In Refine, Inpaint or Outpaint, run that pass; in Mass refine, Swap tags or More seeds, queue that action.'],
      [['Enter'], 'Run from a one-line Refine or Inpaint field. In Outpaint, padding and band fields commit their values, the shape field applies its preset, and other one-line fields run the pass.'],
      [['Up', 'Down', 'Shift+Up', 'Shift+Down'], 'In a pass\'s Strength field, step a single value by 0.05, or 0.01 with Shift.'],
      [['Escape'], 'Close the current dialog or menu first.'],
    ],
  },
  {
    id: 'gallery',
    title: 'Gallery and selection',
    scope: 'Click opens an image. Ctrl/Cmd+click toggles a pick; Ctrl/Cmd+Shift+click adds a range. Select makes clicks pick. These keys act on the focused card; Ctrl/Cmd+A also works on Select.',
    rows: [
      [['Tab'], 'Enter the current card and its controls.'],
      [['Left', 'Right', 'Up', 'Down', 'Home', 'End'], 'Move between card images; moving past the last loaded card can load more.'],
      [['Enter'], 'Open the focused image in the lightbox.'],
      [['/'], 'Focus gallery search from the workspace, leaving the query and filters unchanged. Fields, menus and dialogs keep their keys.'],
      [['M'], 'Open More seeds for the focused image\'s batch. Picks stay unchanged; nothing queues until Queue is pressed.'],
      [['Space', 'Shift+Space'], 'Pick or unpick the focused image; with Shift, add the range from the selection anchor.'],
      [['Ctrl/Cmd+A'], 'Add every loaded gallery card to the selection.'],
      [['Escape'], 'Clear the picks, then leave Select mode on the next press.'],
      [['F'], 'Star or unstar the focused card, except in the recycle bin.'],
    ],
  },
  {
    id: 'sheet',
    title: 'Contact sheet',
    scope: 'Navigation starts on a populated cell; zoom also works on its star.',
    rows: [
      [['Left', 'Right', 'Up', 'Down'], 'Move between populated cells.'],
      [['Home', 'End'], 'Move to the ends of the current row.'],
      [['Ctrl+Home', 'Ctrl+End'], 'Move to the first or last populated cell.'],
      [['Enter', 'Space'], 'Open the focused image in the lightbox.'],
      [['F'], 'Star or unstar the focused cell.'],
      [['M'], 'Open More seeds for the focused cell\'s batch; review and Queue separately.'],
      [['Delete'], 'Move the focused cell\'s image to the recycle bin at once; the note has Restore. Holding the key does not repeat.'],
      [['+', '-'], 'Zoom the sheet in or out.'],
      [['0', '1'], 'Reset the sheet zoom; show one image pixel per screen pixel.'],
    ],
  },
  {
    id: 'lightbox',
    title: 'Lightbox and comparison',
    scope: 'Letter and zoom keys do not replace typing or modified browser shortcuts. A select keeps its own arrows.',
    rows: [
      [['Left', 'Right'], 'Browse images, or step through the current picked pair, source passes or comparison values.'],
      [['F', 'R', 'S'], 'Star the image; open Refine; toggle the current comparison.'],
      [['X'], 'Pick or unpick this image for the gallery\'s actions. Not in the recycle bin or over a contact sheet.'],
      [['M'], 'Open More seeds for this image\'s batch. Opening does not queue images.'],
      [['Delete'], 'Move this image to the recycle bin. In the bin, ask to delete it permanently. Holding the key does not repeat.'],
      [['+', '-'], 'Zoom in or out.'],
      [['0', '1'], 'Fit the picture; show one image pixel per screen pixel.'],
      [['Shift+Left', 'Shift+Right'], 'Move a swipe divider without leaving the current comparison.'],
      [['PageUp', 'PageDown', 'Home', 'End'], 'With the divider focused, move it by 10% or to either edge.'],
      [['Escape'], 'Close the lightbox and return to the image.'],
    ],
  },
  {
    id: 'prompt',
    title: 'Prompts and tag chips',
    scope: 'Suggestions and chip editors take Escape before the helper or prompt check beneath them.',
    rows: [
      [['P'], 'Focus the prompt box from the workspace, with the caret at the end. Fields, menus and dialogs keep their keys.'],
      [['Ctrl/Cmd+Z'], 'Undo text inserted or replaced by Apply, Reuse, PNG settings, Expand, Examples, Snippets or a chip edit.'],
      [['Tab', 'Enter'], 'Accept a matching tag suggestion when its list offers one.'],
      [['Up', 'Down'], 'Move through the open suggestion list.'],
      [['Left', 'Right', 'Home', 'End'], 'Move between focused tag chips.'],
      [['Enter', 'F2'], 'Edit the focused chip.'],
      [['Delete', 'Backspace'], 'Remove the focused chip.'],
      [['Alt+Left', 'Alt+Right'], 'Move the focused chip within the prompt.'],
      [['Menu', 'Shift+F10'], 'Insert a tag after the focused chip.'],
      [['Escape'], 'Close the suggestion list or chip editor first, then the helper, then the prompt check.'],
    ],
  },
  {
    id: 'references',
    title: 'Reference images',
    scope: 'Focus a reference chip\'s remove button.',
    rows: [
      [['Delete'], 'Remove that reference.'],
      [['Alt+Left', 'Alt+Right'], 'Move the reference in the ordered list.'],
    ],
  },
  {
    id: 'inpaint',
    title: 'Inpaint picture',
    scope: 'Focus the picture to use these keys; text fields keep normal editing.',
    rows: [
      [['B', 'E', 'I'], 'Choose the brush; choose the eraser; invert the mask.'],
      [['[', ']'], 'Make the brush smaller or larger.'],
      [['Ctrl/Cmd+Z'], 'Undo a mask change.'],
      [['Ctrl/Cmd+Shift+Z', 'Ctrl/Cmd+Y'], 'Redo a mask change.'],
      [['Left', 'Right', 'Up', 'Down'], 'Move the brush; hold Shift to move farther.'],
      [['Space'], 'Paint at the brush; hold Space while moving to draw a stroke.'],
    ],
  },
  {
    id: 'queue',
    title: 'Job queue and recovery',
    scope: 'There are no dedicated queue accelerators. Use the visible current-job, queued-row and recovery buttons; each names what it affects.',
    rows: [
      [['Tab', 'Shift+Tab'], 'Move between the available job actions.'],
      [['Enter', 'Space'], 'Activate the focused button, including Cancel, Skip, Move to front or a recovery action. A held button explains its reason.'],
    ],
  },
  {
    id: 'bundles',
    title: 'Gallery bundles',
    scope: 'There are no dedicated bundle accelerators. Ctrl/Cmd+Enter never starts a bundle operation. Bundle & clear still requires the exact gallery-record count.',
    rows: [
      [['Tab', 'Shift+Tab'], 'Move between bundle scope, passwords, confirmation and actions.'],
      [['Enter', 'Space'], 'Activate the focused button; this does not bypass a held action or its safety checks.'],
      [['Escape'], 'Close Bundles; an operation already accepted by the server continues.'],
    ],
  },
  {
    id: 'menus',
    title: 'Menus and settings groups',
    rows: [
      [['Menu', 'Shift+F10'], 'Open actions for a focused gallery card, result image, contact-sheet cell or lightbox image.'],
      [['Up', 'Down', 'Home', 'End'], 'Move through an open menu.'],
      [['Enter', 'Space'], 'Choose the focused menu item or setting button.'],
      [['Escape'], 'Close the menu and return focus.'],
      [['Left', 'Right', 'Home', 'End'], 'Move focus along size presets or a card-view button group without choosing.'],
    ],
  },
];

let shortcutsOrigin = null;

// Read before showModal moves focus and closes an originating menu. Every
// group remains in the inventory, including contexts with no images yet.
function shortcutContext(target) {
  const at = target?.closest ? target : null;
  const context = (label, groups, note) => ({ label, groups, note });
  if (at?.closest('#context-menu')) {
    const source = shortcutContext(ctxMenu.host);
    return context('Image actions', ['menus', ...source.groups], 'Enter chooses a menu item. Image shortcuts apply outside the menu in their listed contexts.');
  }
  if (at?.closest('#examples-menu')) return context('Example prompts', ['menus', 'prompt'], 'Enter chooses an example; Escape returns to Examples.');
  if (at?.closest('#picked-menu')) return context('Picked images', ['menus', 'gallery'], 'Enter chooses a picked-image action; Escape returns to Picked.');
  if (at?.closest('#inpaint-canvas')) return context('Inpaint picture', ['inpaint', 'generate'], 'Picture keys act on the mask; text fields keep normal editing.');
  if (at?.closest('#bundles-dialog')) return context('Gallery bundles', ['bundles'], 'Use the focused controls. Ctrl/Cmd+Enter does not start an operation here.');
  const dialog = at?.closest('dialog[open]') || document.querySelector('dialog[open]:not(#shortcuts-dialog)');
  if (dialog === els.lightbox) return context('Lightbox', ['lightbox'], 'Image shortcuts leave text fields and selects alone. Ctrl/Cmd+Enter does not Generate here.');
  if ([els.refineDialog, els.inpaintDialog, els.outpaintDialog, els.editDialog, els.refineBatchDialog, els.swapDialog, els.moreSeedsDialog].includes(dialog)) {
    return context(dialog.querySelector('h2')?.textContent.trim() || 'Image action', ['generate'], 'Ctrl/Cmd+Enter runs this action. Help above it never submits underneath.');
  }
  if (dialog) return context(dialog.querySelector('h2')?.textContent.trim() || 'Settings', ['menus'], 'Use the focused controls. Ctrl/Cmd+Enter does not Generate underneath settings.');
  if (at?.closest('#reference-list')) return context('Reference images', ['references'], 'Reference keys act on the focused reference chip.');
  if (at?.closest('#tag-validation, #prompt-analysis, .prompt-editor, #negative-field, #examples, #tag-helper, #tags-menu, #tag-finder, #btn-analyze, #btn-reset, #btn-expand')) {
    return context('Prompt editing', ['prompt'], 'Typing stays in its field; suggestions and chip editors own their keys first.');
  }
  if (at?.closest('#grid-section')) return context('Grid search', ['generate', 'menus'], 'Ctrl/Cmd+Enter runs the grid from a focused field in the open Grid section.');
  if (at?.closest('#progress, #queue-note, #finish-note, #last-run, #btn-new-image')) {
    return context('Job queue and recovery', ['queue'], 'Tab reaches the visible job actions. No global key cancels or reorders jobs.');
  }
  if (at?.closest('#sheet')) return context('Contact sheet', ['sheet'], 'Navigation starts on a populated cell. Image actions are available after opening its lightbox.');
  if (at?.closest('#viewer')) return context('Result image', ['menus', 'gallery'], 'Enter or Space opens a focused result image; M opens More seeds for its batch. Menu or Shift+F10 opens its image actions. Gallery navigation keys belong to gallery cards.');
  if (at?.closest('#recent')) return context('Earlier session image', ['gallery'], 'Enter or Space opens this image; M opens More seeds for its batch. This strip has no app image-actions menu or gallery-card navigation; open its lightbox for more actions.');
  if (at?.closest('#library, .gallery-head')) {
    if (sheetMode) return context('Contact sheet', ['sheet'], 'Navigation starts on a populated cell. Image actions are available after opening its lightbox.');
    return context(binMode ? 'Recycle bin' : 'Gallery', ['gallery'], binMode ? 'These are the bin\'s picks. Starring and the app\'s image-actions menu are unavailable here.' : 'Card keys act on the focused image; fields keep their normal editing.');
  }
  if (sheetMode && (!at || at === els.shortcuts || at === document.body)) return context('Contact sheet', ['sheet'], 'Navigation starts on a populated cell. Image actions are available after opening its lightbox.');
  return context('Generation settings', ['generate'], 'Keys act where focus is. All shortcut groups remain available below.');
}

function shortcutReturnTargets(origin) {
  const targets = [origin];
  if (origin?.closest?.('#examples-menu')) targets.push(els.examplesBtn);
  if (origin?.closest?.('#picked-menu')) targets.push(els.pickedBtn);
  if (origin?.closest?.('#context-menu')) {
    targets.push(ctxMenu.opener, ctxMenu.host?.querySelector('img'));
    if (ctxMenu.host === els.lbStage) targets.push(els.lbClose);
  }
  const dialog = origin?.closest?.('dialog[open]');
  if (dialog) targets.push(dialog.querySelector('button:not([disabled])'));
  targets.push(els.shortcuts);
  return [...new Set(targets.filter(Boolean))];
}

function shortcutFocusReady(target) {
  return target?.isConnected && !target.disabled && !target.closest?.('[hidden], [inert]')
    && target.getClientRects().length > 0 && getComputedStyle(target).visibility === 'visible';
}

function renderShortcuts(context) {
  const order = new Map(context.groups.map((id, index) => [id, index]));
  const groups = [...SHORTCUT_GROUPS].sort((a, b) => (order.get(a.id) ?? 100) - (order.get(b.id) ?? 100));
  const note = document.createElement('p');
  note.id = 'shortcuts-context';
  note.className = 'shortcuts-context';
  note.textContent = `${context.label} shortcuts first. ${context.note} The full list is below.`;
  const sections = groups.map((group) => {
    const section = document.createElement('section');
    section.dataset.shortcutGroup = group.id;
    section.classList.toggle('shortcut-relevant', order.has(group.id));
    const heading = document.createElement('h3');
    heading.id = `shortcuts-group-${group.id}`;
    heading.textContent = group.title;
    section.setAttribute('aria-labelledby', heading.id);
    section.append(heading);
    if (group.scope) {
      const scope = document.createElement('p');
      scope.className = 'shortcuts-scope hint';
      scope.textContent = group.scope;
      section.append(scope);
    }
    const list = document.createElement('dl');
    for (const [keys, action] of group.rows) {
      const term = document.createElement('dt');
      for (const [at, key] of keys.entries()) {
        if (at) term.append(document.createTextNode(' / '));
        const label = document.createElement('kbd');
        label.textContent = key;
        term.append(label);
      }
      const description = document.createElement('dd');
      description.textContent = action;
      list.append(term, description);
    }
    section.append(list);
    return section;
  });
  els.shortcutsBody.replaceChildren(note, ...sections);
}

function openShortcuts() {
  if (els.shortcutsDialog.open) return;
  const origin = document.activeElement;
  const context = shortcutContext(origin);
  shortcutsOrigin = shortcutReturnTargets(origin);
  renderShortcuts(context);
  els.shortcutsDialog.setAttribute('aria-describedby', 'shortcuts-intro shortcuts-context');
  els.shortcutsDialog.scrollTop = 0;
  els.shortcutsDialog.showModal();
  els.shortcutsClose.focus({ preventScroll: true });
}

function closeShortcuts() {
  if (els.shortcutsDialog.open) els.shortcutsDialog.close();
}

function shortcutsClosed() {
  // A queued close event must not restore focus under a help opened again.
  if (els.shortcutsDialog.open) return;
  const targets = shortcutsOrigin ?? [];
  shortcutsOrigin = null;
  for (const target of targets) {
    if (!shortcutFocusReady(target)) continue;
    target.focus?.({ preventScroll: true });
    if (document.activeElement === target) break;
  }
}

function shortcutsBackdrop(event) {
  if (event.target !== els.shortcutsDialog) return;
  const box = els.shortcutsDialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right
    && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) closeShortcuts();
}

function shortcutKey(event) {
  if (els.shortcutsDialog.open) {
    // Close is help's only interactive control. Native Tab can otherwise
    // leave it for the browser chrome before cycling back into the dialog.
    if (event.key === 'Tab' && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      els.shortcutsClose.focus({ preventScroll: true });
    }
    return;
  }
  if (event.key !== '?' || keyboardEventHeld(event)
      || event.repeat || keyboardCommandModifier(event) || keyboardEditingTarget(event.target)) return;
  event.preventDefault();
  openShortcuts();
}
