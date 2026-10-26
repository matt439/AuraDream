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

Stages 2 and 3 remain open.
