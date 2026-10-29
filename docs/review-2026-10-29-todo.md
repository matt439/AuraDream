# TODO audit, 2026-10-29

Matt asked for an audit of the backlog and, if no work was ready, a
brainstorm of what could be added. He then asked for every proposal to be
added. This audit writes owners and rows only: none of the features below is
built, no GPU run, model download or real gallery operation was performed,
and no decision was taken for Matt.

## Reconciliation

The starting index had 29 rows: 15 CARD, nine DEFERRED, three USER, one
DECISION and one BLOCKED. No row was OPEN, so NOW's aim ("keep useful work
ready") had nothing ready behind it. All local links resolved. Fixed in this
commit:

- NOW held finished history (the 2026-10-27 decisions, `GEN#22`'s card
  prerequisite, completed shortcut stages) and named three CARD rows as
  "next". It now lists the ready rows, the decisions and what needs Matt's
  hands or card, separately.
- Backup activation choices and the compilation opt-in were unsettled in
  their owners with no DECISION row; they are now `GAL#22` and `PERF#7`,
  with numbered questions in [backups](plan-automatic-backups.md#matts-choices-for-unattended-backups)
  and [compilation](plan-regional-compilation.md#matts-questions).
- `PERF#2` was the only speed row without a CPU planning slice. The planning
  slice is now `PERF#5` ([plan](plan-fp8-activations.md)); `PERF#2` keeps
  only the card measurement, waiting on `PERF#1`'s card qualification.
- Stale owner text: [setup portability](plan-setup-portability.md) still
  called `GAL#12` incomplete and `UI#20` open; the
  [Runs favorite counts](review-2026-10-29-runs-favorite-counts.md) status
  contradicted itself; the [CI browser startup](review-2026-10-29-ci-browser-startup.md)
  review still awaited hosted runs that had passed; [peak VRAM](plan-peak-vram.md)
  and [Then refine](plan-then-refine.md) did not say `GEN#22` met their RAM
  prerequisite; [requirements](design/requirements.md) described Pick this
  run as built.
- Review coverage now links the `GAL#8` and `UI#23` owners and the CI startup
  review, and is in date order.
- The five commits that closed `GAL#12`, `UI#28`, `GEN#31`, `GAL#19`,
  `UI#20`, `DEV#8` and `UI#25` are on a local branch with no hosted CI;
  Review coverage says so until they reach main.

## Method

One agent reconciled every row with its owner and the source. Five more
proposed work from different angles: composing and the queue; the gallery,
inspection and bundles; reruns, Refine and models; defects in recent code;
and tooling plus CPU slices of gated rows. A separate verifier then tried to
refute each lens's proposals: already built, decided against, wrong evidence
or a failure scenario that does not trace. Ten of 33 proposals were dropped
or merged. Each owner was written by one agent and checked against the
source by another, which corrected line citations and several behavioural
claims.

[Reported use](design/preferences.md) set the ranking: Often features first,
real defects first among equals.

## Added rows

| Row | Status | Owner |
|---|---|---|
| `GAL#21` search exclusion | Done 2026-10-30 | [plan-search-exclusion.md](plan-search-exclusion.md) |
| `GEN#32` queue estimates | Done 2026-10-30 | [plan-queue-estimates.md](plan-queue-estimates.md) |
| `PA#12` negative-box tag suggestions | Done 2026-10-30 | [review-2026-10-29-backlog.md](review-2026-10-29-backlog.md#pa12-tag-suggestions-in-the-negative-prompt) |
| `UI#32` sheet-cell menu and Delete | Done 2026-10-30 | [review-2026-10-29-backlog.md](review-2026-10-29-backlog.md#ui32-contact-sheet-cell-menu-and-delete) |
| `UI#33` pick from the lightbox | Done 2026-10-30 | [review-2026-10-29-backlog.md](review-2026-10-29-backlog.md#ui33-pick-from-the-lightbox) |
| `GEN#33` Re-render at a tier | Done 2026-10-30 | [review-2026-10-29-backlog.md](review-2026-10-29-backlog.md#gen33-re-render-at-a-tier) |
| `GEN#34` Swap tags Add rows | Done 2026-10-30 | [review-2026-10-29-backlog.md](review-2026-10-29-backlog.md#gen34-swap-tags-add-rows) |
| `GAL#23`, `GEN#36`, `GEN#38`, `UI#38` | OPEN | [review-2026-10-29-backlog.md](review-2026-10-29-backlog.md) |
| `GEN#35`, `GEN#37` Then refine follow-ons | OPEN | [plan-then-refine.md](plan-then-refine.md) |
| `UI#34` Compare scales | OPEN | [plan-size-axis-crop-anchor.md](plan-size-axis-crop-anchor.md#compare-scales-in-the-refine-dialog) |
| `UI#35` close-watcher holds | Done 2026-10-30 | [review-2026-10-29-close-watcher.md](review-2026-10-29-close-watcher.md) |
| `UI#36` prompt route key | OPEN | [plan-shortcut-workflow.md](plan-shortcut-workflow.md#stage-4-prompt-route) |
| `UI#37` Pick this run, `GAL#24` bin age | OPEN | [review-2026-10-12.md](review-2026-10-12.md) F19 and F5 follow-ups |
| `DEV#20`, `DEV#22`, `DEV#23` card-session tools | OPEN | [plan-card-harness.md](plan-card-harness.md) |
| `PERF#5` fp8-activation plan | OPEN | [plan-fp8-activations.md](plan-fp8-activations.md) |
| `PERF#6` fixed comparison cases | OPEN | [plan-step-caching.md](plan-step-caching.md#fixed-comparison-cases) |
| `DEV#21` trial kit refresh | OPEN | [trials README](design/trials/README.md#refresh-for-the-current-app-dev21) |
| `DEV#24`-`DEV#26` smoke reruns, timings, fake-knob map | OPEN | [plan-tooling.md](plan-tooling.md) |
| `GAL#22`, `PERF#7`, `GEN#39` | DECISION | backups, compilation, [Then refine](plan-then-refine.md#decision-several-strengths-per-image) |

`GEN#6`, `INP#9` and `INP#10` changed from DEFERRED to DECISION: each has a
CPU slice, and admitting it is Matt's call. Nothing about them was built.

## Dropped

- Refusing Generate on prompt variables while Grid is folded: the open-fold
  template scope is documented (design notes, the Ctrl/Cmd+Enter rule).
- Showing a tab's last result again after a reload: the design notes record
  that remembered ownership is not a replay list.
- Saved-bundle rows showing their record count: the date and Protected tag
  already show, and a count reads every manifest under leases.
- A bundle import preview: L-size on the staging path for an idempotent import
  that already reports its counts.
- A viewer Edit tool for instruction edits: waits on `INP#11`.
- More quantization/offload card checks and LoRA measurements: K4 and I1/I2
  are reported Never.
- Separate card-harness, snippet-port and forced-close rows: merged into
  `DEV#20` and `UI#35`.

## Validation

See the commit: Markdown links in changed files were checked locally, and
the docs-map tests and ruff were run. No code changed.
