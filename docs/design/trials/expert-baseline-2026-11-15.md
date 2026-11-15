# AD-2: current workbench journey baseline - 2026-11-15

Agent expert review, not a human trial or a decision to ship the concept.
Sources: `ac78e436` in an isolated worktree; only trial tooling/docs changed.
No production UI, Refine code, GPU, weights, remote service or live gallery
was touched. Existing October records remain historical evidence.

## Reproduction and scope

Use the [trial kit](README.md), fixture v2 variant A, a new owned scratch
folder and fresh browser profile. Wait until the owned fake answers
`/api/gallery/bundles` before starting the browser. Generate steps with
`tools/trial_walkthrough.py` and run `node tools/browser.mjs <steps.json>`.
This run used port 8345, Edge/HeadlessChrome 157, Windows, 1400x900 CSS,
DPR 1. The original checkout's existing Python environment was reused;
no packages or models were installed. Variant B is not qualified here.

The concept was served from `docs/design` on 8343 with Python's loopback
HTTP server. `AURADREAM_PORT_OFFSET=432 node tools/browser.mjs
docs/design/prototype/check.json` passed **54/54 expectations**, exit 0.
It verifies synthetic interactions, not production performance or backend
parity. Its simulated queue/archive/Finish cannot prove those workflows.

## Baseline repair, not a redesign

Two discarded diagnostic runs exposed prerequisites rather than valid task
measurements. One started before server readiness, lost page helpers and
continued after a failed scratch ownership assertion. A fresh ready run
then found the old driver clicking True CFG inside the now-closed Sampling
disclosure. It could type into the wrong focused field. Both runs were
stopped; neither is counted as a pass.

The driver now opens Sampling through its actual summary button and
asserts it opened. Ownership and initial/reload readiness are fatal
prerequisites, so invalid setup stops before subsequent actions. The extra
Sampling press is real navigation, not a hidden DOM shortcut. Tests lock
these ordering/metadata contracts. No production-layout change is justified
by a stale test alone.

## Current-app result

The repaired fresh-A run passed **208/208 expectations**, exit 0. The
existing eight task cards remained unchanged. Recorded trusted event totals:

| Task | Clicks | Keys | Context outcome |
|---|---:|---:|---|
| 1 Generate / Reset / Undo | 16 | 16 | Exact Undo; one extra Sampling press versus October's 15 clicks |
| 2 Queue / cancel / reload | 15 | 18 | Draft and survivor job retained; one Custom dimensions disclosure |
| 3 Inspect / source / sheet | 10 | 14 | Draft retained; close returned focus to child image |
| 4 Reuse / Undo / picked rerun | 12 | 19 | Two hidden picks and unfinished draft retained; More seeds did not queue |
| 5 Archive / restore | 11 | 12 | 77 records restored, one binned; draft and sidecars retained |
| 6 Saved setup / Undo | 6 | 2 | Exact Undo; nothing queued |
| 7 Saved search | 3 | 0 | Saved run restored, Starred off, focus to opener |
| 8 Then refine / source comparison | 11 | 10 | Base/pass linked; downward wheel and leaving Select mode required |

No draft-loss incident occurred in these checked paths. Counts for tasks 2-8
match the October record. This does not show unchanged usability in every
layout: task 8 still needs an explicit wheel detour and mode exit, and the
first task's extra disclosure is a measurable tradeoff of the compact panel.
The synthetic Then refine run is adjacency coverage, not qualification of
the separate Refine implementation pack.

Validation: `python -m pytest tests/test_trial_kit.py -q`: **16 passed**;
`git diff --check`: clean. One existing Starlette/httpx deprecation warning.

## Fixed practical comparison

These are task boundaries, not a race between unequal implementations.
Trusted automation clicks/keys include setup; they are not human effort or
timing. Programmatic scroll helpers do not establish human reachability.

| Task | Current app evidence / friction to measure | Concept contract and remaining gap |
|---|---|---|
| Make a controlled draft, Generate, Reset, Undo | Negative requires True CFG; fresh profile requires one Sampling disclosure press. Reset/Undo must recover exact draft. | Ready draft Generate is one action, but the static concept does not replace real model-specific validation. Preserve controls, do not hide their effective values. |
| Inspect child, compare source, star, return | Lightbox and source comparison are separate views; draft and return focus are checked. | Inspector keeps draft separate and source lineage visible. No evidence yet that another layout reduces human reorientation. |
| More seeds, dismiss, Reuse, Undo | Preview must queue nothing until confirmed; reuse is deliberate draft replacement with Undo. | Concept retains draft on inspection and More seeds; Reuse closes inspector, Undo restores draft. Same safety contract, not a new feature by itself. |
| Pick keeper and child across filters, rerender | Two search contexts plus pick mode; scope must retain both picks despite Starred/filter. | Concept explicitly labels picks versus filter/Starred scope and refuses Compare unless exactly two. Test interpretation before consolidating controls. |
| Return after reload and restore a saved setup | Reload has no guaranteed focused control; restoring a setup is a separate find/preview/restore flow. | Concept tests draft separation but not durable server recovery or actual setup validation. Keep these acceptance tests before migration. |

## Ranked independent next slices

1. **Keep the baseline trustworthy.** Run variant B and short-window views;
   make hidden-target/focus prerequisites fail closed before typing. Keep
   real disclosure presses in counts. This is small, independent of Refine.
2. **Scope comprehension trial.** Use the same two picked images and a
   contradictory Starred/search filter in both UIs. Ask what will run before
   pressing More seeds/rerender; count mistaken scope predictions, corrections,
   and trips back to selection controls. Do not call the concept faster merely
   because it has fewer controls.
3. **Draft-preservation / judge-return trial.** Keep an unfinished prompt,
   inspect a result and source, star, close, then resume editing. Record draft
   loss (must be zero), focus recovery and navigation back to the draft. Only
   then consider a bounded inspector/navigation change.
4. **Saved-setup discovery.** Compare current Examples -> Saved setups ->
   row -> Restore with a directly discoverable entry, retaining preview and
   Undo. Measure wrong routes and context re-entry rather than deleting the
   confirmation step for a lower click count.

Refine pack changes, GPU quality/performance qualification, complete
workbench replacement and human speed claims are explicitly out of scope.
