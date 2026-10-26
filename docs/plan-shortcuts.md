# Keyboard shortcut overlay

`UI#12`, from the [2026-10-24 review](review-2026-10-24.md#g9-keyboard-shortcut-overlay).

## Scope

Show the page's keyboard shortcuts in a native dialog, opened by `?` or a
Keys button in the top bar. The settings panel gains no height. Group the
keys by where they work and keep the labels consistent with the controls'
`aria-keyshortcuts` and existing handlers. This is help for the existing
keyboard behavior, not a change to those actions.

## Implementation

1. Add a named dialog with a visible Close button and a top-bar opener.
   Keep feature declarations in `web/shortcuts.js`, loaded before `app.js`;
   wire listeners only from `app.js`. Style it with the existing tokens.
2. Open on an unmodified `?`, ignoring repeats, composition, text editing
   and selects. Allow help above another dialog without closing it. While
   help is open, keys belong to help: in particular Ctrl/Cmd+Enter must not
   submit and Escape must close only help. Restore its connected opener
   without scrolling after Close, Escape or a backdrop press.
3. List the actual scoped actions, including generation, gallery/select,
   sheet, lightbox/comparison, prompt chips and inpaint. Say where typing or
   focus changes a key's meaning. Keep the usage guide and module map current.

## Verification

Add a browser smoke using a scratch fake server. Check the keyboard and
button openers, editable/modifier/composition/repeat guards, help over the
lightbox and a pass dialog, no submission from Ctrl+Enter, Close/Escape/
backdrop and focus restoration, and a narrow dialog without horizontal
overflow. Run the affected existing lightbox/inpaint smokes and the docs-map
checks. No GPU or checkpoint download is needed.

## Status

Implemented and checked 2026-10-25. `smoke-shortcuts` passes on Chrome,
including nested dialogs, typing/modifier/composition guards, Tab containment,
focus restoration and the 390 px layout. `smoke-lightbox` and `smoke-inpaint`
also pass, run serially on scratch fakes. The six docs-map checks and Node's
syntax checks for `shortcuts.js` and `app.js` pass.

The browser checks used an isolated copy of HEAD plus this feature, keeping
uncommitted work on other TODO items out of the result. Browser startup failed
inside the restricted process; the approved execution environment ran the
installed Chrome successfully. No model weights were loaded or downloaded.
