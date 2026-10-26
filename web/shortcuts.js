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
    title: 'Help',
    rows: [[['?'], 'Open this help outside text fields; Escape closes it first.']],
  },
  {
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
    title: 'Gallery and selection',
    scope: 'Click opens an image. Ctrl/Cmd+click toggles a pick; Ctrl/Cmd+Shift+click adds a range. Select makes clicks pick. These keys act on the focused card; Ctrl/Cmd+A also works on Select.',
    rows: [
      [['Tab'], 'Enter the current card and its controls.'],
      [['Left', 'Right', 'Up', 'Down', 'Home', 'End'], 'Move between card images; moving past the last loaded card can load more.'],
      [['Enter'], 'Open the focused image in the lightbox.'],
      [['Space', 'Shift+Space'], 'Pick or unpick the focused image; with Shift, add the range from the selection anchor.'],
      [['Ctrl/Cmd+A'], 'Add every loaded gallery card to the selection.'],
      [['Escape'], 'Clear the picks, then leave Select mode on the next press.'],
      [['F'], 'Star or unstar the focused card, except in the recycle bin.'],
    ],
  },
  {
    title: 'Contact sheet',
    scope: 'Navigation starts on a populated cell; zoom also works on its star.',
    rows: [
      [['Left', 'Right', 'Up', 'Down'], 'Move between populated cells.'],
      [['Home', 'End'], 'Move to the ends of the current row.'],
      [['Ctrl+Home', 'Ctrl+End'], 'Move to the first or last populated cell.'],
      [['Enter', 'Space'], 'Open the focused image in the lightbox.'],
      [['F'], 'Star or unstar the focused cell.'],
      [['+', '-'], 'Zoom the sheet in or out.'],
      [['0', '1'], 'Reset the sheet zoom; show one image pixel per screen pixel.'],
    ],
  },
  {
    title: 'Lightbox and comparison',
    scope: 'Letter and zoom keys do not replace typing or modified browser shortcuts. A select keeps its own arrows.',
    rows: [
      [['Left', 'Right'], 'Browse images, or step through the current picked pair, source passes or comparison values.'],
      [['F', 'R', 'S'], 'Star the image; open Refine; toggle the current comparison.'],
      [['Delete'], 'Move this image to the recycle bin. In the bin, ask to delete it permanently. Holding the key does not repeat.'],
      [['+', '-'], 'Zoom in or out.'],
      [['0', '1'], 'Fit the picture; show one image pixel per screen pixel.'],
      [['Shift+Left', 'Shift+Right'], 'Move a swipe divider without leaving the current comparison.'],
      [['PageUp', 'PageDown', 'Home', 'End'], 'With the divider focused, move it by 10% or to either edge.'],
      [['Escape'], 'Close the lightbox and return to the image.'],
    ],
  },
  {
    title: 'Prompts and tag chips',
    scope: 'Suggestions and chip editors take Escape before the helper or prompt check beneath them.',
    rows: [
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
    title: 'Reference images',
    scope: 'Focus a reference chip\'s remove button.',
    rows: [
      [['Delete'], 'Remove that reference.'],
      [['Alt+Left', 'Alt+Right'], 'Move the reference in the ordered list.'],
    ],
  },
  {
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
    title: 'Menus and settings groups',
    rows: [
      [['Menu', 'Shift+F10'], 'Open the focused image\'s actions.'],
      [['Up', 'Down', 'Home', 'End'], 'Move through an open menu.'],
      [['Enter', 'Space'], 'Choose the focused menu item or setting button.'],
      [['Escape'], 'Close the menu and return focus.'],
      [['Left', 'Right', 'Home', 'End'], 'Move focus along size presets or a card-view button group without choosing.'],
    ],
  },
];

let shortcutsOrigin = null;

function renderShortcuts() {
  const sections = SHORTCUT_GROUPS.map((group, index) => {
    const section = document.createElement('section');
    const heading = document.createElement('h3');
    heading.id = `shortcuts-group-${index}`;
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
  els.shortcutsBody.replaceChildren(...sections);
}

function openShortcuts() {
  if (els.shortcutsDialog.open) return;
  shortcutsOrigin = document.activeElement;
  renderShortcuts();
  els.shortcutsDialog.showModal();
  els.shortcutsClose.focus({ preventScroll: true });
}

function closeShortcuts() {
  if (els.shortcutsDialog.open) els.shortcutsDialog.close();
}

function shortcutsClosed() {
  // A queued close event must not restore focus under a help opened again.
  if (els.shortcutsDialog.open) return;
  const origin = shortcutsOrigin;
  shortcutsOrigin = null;
  if (origin?.isConnected) origin.focus?.({ preventScroll: true });
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
