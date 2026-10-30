# Keyboard ownership, help and compact setup

Approved 2026-10-26 after the browser shortcut and decluttering audits. Matt
approved all three stages: fix existing conflicts, make help contextual and
visible, then add a small set of fast routes and group secondary setup.
This is a bounded change to the current app, not the separate workbench layout.

## Stages and checks

1. **Dispatch and focus.** One event has one owner. Handled/composing keys,
   extra modifiers and editing/select targets keep their meaning. The active
   modal owns commands; unknown settings dialogs cannot submit underneath.
   Discrete actions do not repeat, while navigation, zoom and numeric stepping
   do. Regress the observed Then refine/Picked Ctrl+Enter double actions,
   lightbox Alt+Arrow and F in a select, including delayed More seeds replies.
2. **Help and hints.** Show the relevant context first without losing the full
   inventory. Help and visible hints describe actual scoped bindings; menus,
   dialogs and field editing must agree. Preserve native focus return.
3. **Fast routes and setup groups.** `/` focuses gallery search outside editing
   and modals. `M` opens eligible focused-image/lightbox More seeds, never
   queues immediately. Group Custom dimensions and LoRA/reference setup with
   accurate current/active summaries; presets, current dimensions, warnings and
   active inputs remain visible. Comparison grouping is conditional on a safe,
   discoverable fast route and must preserve S and browser keys.

Each stage receives independent review, isolated fake-browser checks and a
scoped local commit. No real generation, weights, download, push or deployment.
Synthetic key flags establish application dispatch behavior, not acceptance on
every physical keyboard, IME, assistive technology or device.

## Status

Stage 1 completed and qualified 2026-10-26. Independent static reviews are
clear. Eight isolated browser suites passed: shortcut-dispatch, shortcuts,
context-menu, select, tags, inpaint, outpaint and model-switch. The dispatch
smoke includes trusted native extra-modifier and repeated Enter events on
Generate/Run grid, synthetic IME/handled-event probes, positive submission
controls, real modal focus return and a delayed More seeds/help reply.

The first dispatch run exposed native modified-button activation in addition
to the audited conflicts; the final run passed after its capture guard. Two
request-spy route errors were corrected without weakening expectations.
The browser harness now sends all requested modifiers and validates combos
before launch; the one legacy empty-modifier fixture uses native Enter.
Browser-profile cleanup and docs-map tests: 9 passed, 1 Windows skip. Ruff,
all frontend/harness syntax checks and whitespace checks passed.

Receipts: `%TEMP%/local-diffusion-shortcut-existing-20261026-i1.log`,
`%TEMP%/local-diffusion-shortcut-dispatch-20261026-i2.log` and
`%TEMP%/local-diffusion-shortcut-harness-20261026-i1.log`.

Stage 2 completed and qualified 2026-10-26. Relevant groups appear first and
all 11 groups remain available. Help snapshots menu context and visible focus
fallbacks before focusout; `?` works from image menus, including shifted `?`.
Queue/Bundles describe native controls and retained safeguards. Visible menu
hints distinguish menu activation from image-context shortcuts; R appears only
for the lightbox route. Recent and contact-sheet help avoid promising the image
menu where it is unavailable.

Independent reviews are clear. Contextual help, dispatch and context-menu browser
suites passed, plus six docs-map tests and syntax/whitespace checks. Screenshots
`shortcut-help-context-menu.png` and `shortcut-help-context-lightbox.png` were
visually inspected. The existing menu fixture assumed the old first child and
Star's fixed row number; it now asserts the presentation hint separately and
native-clicks the exact Star action after checking its hit target. No product
behavior was loosened to satisfy that fixture. Receipt:
`%TEMP%/local-diffusion-shortcut-help-20261026-j1.log` (help/dispatch),
`%TEMP%/local-diffusion-shortcut-help-menu-20261026-j2.log` (menu).
Disconnected-origin fallback is source-reviewed; hidden-menu-origin return is
browser-qualified. Physical screen-reader/touch acceptance remains UI#7.

Stage 3 completed and qualified 2026-10-26. `/` focuses gallery search without
changing query/scope/picks. M opens the focused image's existing More seeds
preview, independently of hidden picks, and queues nothing. Its holds, actual
visible opener and late-reply guards remain in force. Hints appear beside the
lightbox button, in its image menu and in contextual help. Comparison controls
remain visible with S; the compact legend retains its prior geometry.

Custom dimensions and LoRA/reference setup are native disclosures with current
summaries. Presets, dimensions, warnings, Generate, More seeds entry points and
queue/error controls remain direct. Restore, Reuse/Undo, new reference activation
and validation focus reveal relevant controls; ordinary polling changes words
only. Manual collapse retains every value. Pending LoRA choices are named
truthfully, cleared by Reset/replacement Reuse and preserved by Undo snapshots.

The browser sweep caught and corrected three related issues: a narrow lightbox
menu used its hidden More seeds button as opener, a record redraw could remove
or hide the focused lightbox control, and the new activation guard caught the
intentional Shift+Space range command on an ARIA image button. The final guard
limits native modified-button prevention to native controls. Native Source
activation now returns lost focus to Close without stealing a higher dialog's
focus. Native and synthetic Shift+Space regressions retain exact pick counts.
An intermediate M legend entry changed comparison-button positions; M remains
discoverable beside its action/menu/help while the original legend is retained.
These follow-ups qualify the final tree rather than implying the earlier Stage 1
commit alone had the final preservation coverage.

Independent reviews are clear. Seventeen distinct isolated browser suites passed
across the final qualification runs:

| Receipt under `%TEMP%` | Passing coverage |
|---|---|
| `local-diffusion-compact-existing-20261026-k1.log` | Conditioning, numeric inputs, control sizing (wide-layout assumptions were corrected later). |
| `local-diffusion-compact-preservation-20261026-k3.log` | Wide, grid, Bundles, More seeds, shortcuts, More seeds tabs, mixed prompts (the other five failures were rechecked below). |
| `local-diffusion-compact-rechecks-20261026-k4.log` | Lightbox, selection, CFG/steps comparison, PNG settings, size/crop, shortcut dispatch and compact setup: 7/7 passed. |

The 307-step compact suite covers real status replies, delayed catalogue replies,
Reset/Undo, focus, actual bin/restore, hidden picks, opening without queuing and
390/600/1400/2048 px layouts. Six docs-map tests, Ruff, frontend/harness syntax
and whitespace checks passed. Desktop/narrow proof screenshots were visually
inspected; `compact-setup-20261026.png` is the retained desktop proof.

Fixture repairs preserved outcomes: wide tests explicitly exercise overheight
panels when needed, PNG tests navigate the new disclosure, comparison tests use
native focused controls, and crop tests await actual close cleanup before a new
Reuse. Initial closed-details visibility probes were corrected to consult native
`checkVisibility`; bin tests set their intended query before entering the bin.
The earlier failing receipts remain available. Synthetic CPU/browser checks do
not establish physical IME, screen-reader, touch or GPU/model acceptance.

## Shared conventions

For the parent's DeckLab coordination: Escape owns the top context, primary
Ctrl/Cmd+Enter is contextual and exact, native editing/Undo and browser/OS chords
keep their meaning, and discrete actions do not repeat. Image's two new routes
are intentionally scoped: `/` searches the main gallery; M opens the focused
image's continuation preview. No extra global letter bindings were added.

The approved three stages are complete. No model was loaded or downloaded, and
nothing was pushed or deployed.

## Stage 4 prompt route

**Status:** Done 2026-10-30 as `UI#36` (added 2026-10-29 from the TODO
audit). `focusPromptBox` (`web/app.js`) is the third of the document
keydown handler's small routes, inside the same guard as `/` and also
`!modal`; the help's Prompts group has the P row and `#prompt` carries
`aria-keyshortcuts="P"`. `smoke-shortcut-dispatch` covers a sheet cell, a
gallery card (P and Caps Lock's P), the card the lightbox returns focus to,
the lightbox itself, the modifier/repeat/composing/handled probes, the search,
the negative, a select, a contenteditable, the Picked menu, the model picker,
help and Then refine, with the prompt's text, picks, Select mode, query,
sheet, Check, suggestions and `scrollY` unchanged and no box writer called;
`smoke-shortcuts` the help row and the attribute; `smoke-narrow` at 600x900
the box brought into view by one `scrollIntoView` from a card 1,000 px
down, and nothing moved when it is already on screen. `tests/test_prompt_route.py`
holds the wiring statically. Two choices the design left implicit were taken
while Matt was away (2026-10-29), revisit if wanted: the negative and the
select are probed with their hidden/disabled state lifted for the probe only
(FLUX greys the negative), and the route shares `/`'s `!modal` rather than a
wider guard.

**Evidence.** `docs/design/preferences.md` rates L6 (keyboard shortcuts and
navigation) and A1 (write or edit a prompt) Often. Stage 3 added `/` for
gallery search and M for More seeds, but there is no key back to the prompt
box. In `web/app.js` the document keydown handler's "two small routes"
(`app.js:700-712`) handle only `/` (`focusGallerySearch`, `gallery-search.js:15`)
and unshifted M. P is bound nowhere: the lightbox takes F/R/S/Delete
(`app.js:724`), cards and sheet cells take F (`gallery.js:741`,
`sheet-render.js:704`), and Select mode takes Ctrl/Cmd+A
(`gallery-select.js:269-277`). The help's Prompts group
(`shortcuts.js`, `id: 'prompt'`) lists only keys used inside the box.

### Design

- **Key and guards.** Unmodified, unshifted P (either case, so Caps Lock
  still works) under exactly the guards `/` uses: not repeated, not in a
  menu or listbox, not in a popup (`[role="dialog"]`, the model picker, an
  open Examples or Picked menu), not on an editing target
  (`keyboardEditingTarget`), no Ctrl/Cmd/Alt, and no open `dialog`
  (`!modal`). It runs after `keyboardEventHeld` and `shortcutKey`, like `/`,
  so a key an editor or menu already owns, or one still composing, is never
  a second command. The lightbox is a modal dialog, so P does nothing there;
  after the lightbox closes, P works from the card it returned focus to.
- **Effect.** `preventDefault()`, so no "p" reaches the box that has just
  been focused. Then `els.prompt.focus({ preventScroll: true })` and the caret
  at the end (`setSelectionRange(len, len)`). The route changes no text, so
  it does not use `writeBox`/`rewriteBox`/`replaceBoxText`, and it adds no
  Undo step. It does not run or dismiss Check, does not open tag
  suggestions (they open on input, a click or caret keys, `app.js:115-134`),
  and does not change picks, Select mode, the search, or the sheet.
- **Scrolling.** Above 900 px wide the panel is sticky (`style.css:399`),
  so at 1400x900 the box is already on screen and the page must not move.
  At 900 px and narrower the panel is static above the gallery
  (`style.css:419-422`); where that leaves the box off screen, bring it
  into view only as `focusGallerySearch` does for `/`: one `scrollIntoView({ block: 'center' })` when its rectangle is under
  the topbar or below the window. This follows from the key press; the page
  still never scrolls by itself. The textarea's own internal scroll to show
  the caret is the browser's.
- **Discoverability.** Add one row to the help's Prompts group
  (`[['P'], 'Focus the prompt box from the workspace, with the caret at the
  end. Fields, menus and dialogs keep their keys.']`) and
  `aria-keyshortcuts="P"` on `#prompt`. Stage 2's contextual ordering stays
  as it is. No visible hint is added to the panel: the prompt caption has no
  room at 1400x900 (`docs/rules.md`, Frontend), and nothing moves under
  1800 px.
- **Other keys.** P does nothing in the lightbox (a modal), so it reserves
  no lightbox meaning. The Shared conventions above still hold: this is
  the one further letter, scoped to the workspace like `/`.

### Non-goals

- No key to the negative prompt, Grid fields or a dialog's prompt.
- No change to `/`, M, `?` or any existing binding, and no global binding
  that works inside fields or modals.
- No new visible control or hint line in the panel.
- Physical IME, screen-reader and touch acceptance stays with `UI#7`.

### Stages and completion criteria

1. **Route and help (`UI#36`).** Build the route, the help row and
   `aria-keyshortcuts`. Closed by `smoke-shortcut-dispatch` and
   `smoke-shortcuts` steps on their fakes:
   - from a focused gallery card, a contact-sheet cell, and the card focus
     returns to after the lightbox closes, a native P focuses `#prompt` with
     `selectionStart === selectionEnd === value.length`, the value
     unchanged, no Undo step, picks, Select mode and the query unchanged,
     and `scrollY` unchanged at 1400x900;
   - negatives: P in the search box, the negative, a select, a
     contenteditable, an open menu, the model picker, the lightbox, the
     help and a settings dialog does not move focus; Shift+P, Ctrl/Alt/Meta+P,
     a repeated, composing or already-handled P do nothing;
   - the help's Prompts group shows the P row, and `#prompt` carries
     `aria-keyshortcuts="P"`;
   - `smoke-narrow` (or the compact setup smoke at 390/600 px) shows the
     box brought into view once when it was off screen, with no
     horizontal scroll.
   The existing shortcut, dispatch, context-menu, select and tags suites
   still pass. CPU and browser checks only; no card stage.
