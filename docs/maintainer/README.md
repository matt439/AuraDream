# Maintainer records

Feature plans, review findings, implementation checklists, testing runbooks
and audit history live here. Start current work from [TODO.md](../../TODO.md);
a dated finding is not an open work order.

## Working on the code

- [rules.md](../rules.md): every rule in full, the module map and what each smoke
  covers. [AGENTS.md](../../AGENTS.md) is its short version.
- [design-notes.md](../design-notes.md): why things are built as they are, model
  sources, and what went wrong before.
- [source-audit.md](source-audit.md): the file splits and why the long files
  stay whole, pass by pass.
- [ios-simulator.md](ios-simulator.md): the prepared MacBook connection,
  Windows fake server and SSH tunnel, retained native Safari test workflow,
  input pitfalls and cleanup.
- `plan-*.md`: feature plans. Code comments, the rules and
  [TODO.md](../../TODO.md) cite them as the owners of their designs; what was
  built differently is recorded in the design notes. A finished plan opens
  with **Status: done** and its date. Plans and reviews remain as provenance;
  Git history retains their original locations before the 2026-11-11 move.
  - Live, with open TODO rows or deferred phases:
    [Anima Refine](plan-anima-refine.md),
    [attention backends](plan-attention-backends.md),
    [card harness](plan-card-harness.md), [detailer](plan-detailer.md),
    [fp8 activations](plan-fp8-activations.md),
    [inpainting](plan-inpainting.md), [outpainting](plan-outpainting.md),
    [instruction editing](plan-instruction-editing.md) with its
    [carry](instruction-edit-carry.md) and
    [bundle](instruction-edit-bundles.md) checklists,
    [peak VRAM](plan-peak-vram.md),
    [regional compilation](plan-regional-compilation.md),
    [step caching](plan-step-caching.md) and
    [Then refine](plan-then-refine.md).
  - Done: [automatic backups](plan-automatic-backups.md) (built
    default-off, unattended use declined),
    [button hierarchy](plan-button-hierarchy.md),
    [continuing a run](plan-continue-run.md),
    [gallery area search](plan-gallery-area-search.md),
    [gallery date ranges](plan-gallery-date-ranges.md),
    [gallery orientation](plan-gallery-orientation.md),
    [generation presets](plan-generation-presets.md),
    [More seeds](plan-more-seeds-ux.md),
    [prompt snippets](plan-prompt-snippets.md),
    [queue controls](plan-queue-controls.md),
    [queue estimates](plan-queue-estimates.md),
    [random prompt variables](plan-random-prompt-variables.md),
    [saved searches](plan-saved-searches.md),
    [search exclusion](plan-search-exclusion.md),
    [setup portability](plan-setup-portability.md),
    [shortcut workflow](plan-shortcut-workflow.md),
    [shortcuts](plan-shortcuts.md),
    [size axis and crop anchor](plan-size-axis-crop-anchor.md),
    [snippet portability](plan-snippet-portability.md) and
    [tooling](plan-tooling.md).
- [design/](../design/README.md): the product design study (requirements,
  research, a prototype and trial material).
- `../images/`: the README's screenshots (2026-11-04): a fake server's gallery
  whose PNGs and thumbnails were replaced with drawn stand-in pictures, then
  Playwright at 1600x1000, an iPhone 15 Pro and an iPad Pro 11 in landscape.
- [testing-notes.md](testing-notes.md): observations from use, each marked
  with the commit that closed it; every entry so far is closed.

## Dated records

`review-*.md`, `review-*-findings.json` and the `review-2026-10-04-ui/`
folder are dated reviews of the code and the UI, written with AI assistance. Their findings were triaged into
[TODO.md](../../TODO.md) and fixed or deferred there; they are kept as the
provenance that commit messages and the docs cite, not as current
documentation. Paths in them to a "scratchpad" refer to the reviewer's
temporary folder, which is not part of the repository.

[typecheck-baseline-2026-10-26.md](typecheck-baseline-2026-10-26.md) is the
dated first Pyright baseline (`DEV#6`, since brought to zero errors); the
[tooling plan](plan-tooling.md#backend-type-checking) owns the gate.
