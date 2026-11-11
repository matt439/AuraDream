# Review and plan: explicit fatal smoke prerequisites, 2026-10-28

**Status:** `DEV#9` implemented and locally qualified through cold
preflight checks, ordinary driver probes and the converted scratch smokes. Existing ordinary Edge access remains restored as
recorded in [tooling](plan-tooling.md); no launch/flag/profile/security route
change is part of this work.

## Ranked shortlist

| Rank | Current evidence | Value, scope and risk |
|---|---|---|
| 1 | `browser.mjs` converts thrown/rejected page expressions to undefined; ordinary eval ignores that result and later steps continue. The initial favorite-order diagnostic failed its native pick prerequisite, then spent 195 seconds in downstream cascades. Fresh-data guards also currently use eval/throw. | Explicit prerequisites should stop later fixture actions and retain useful failure evidence. Small opt-in metadata/loop change, with cold validation and actual harmless driver probes. Selected; preserve ordinary aggregate expectations. |
| 2 | Desktop transport loss can leave no run summary while completed per-smoke receipts survive. | Existing receipts already enabled safe reconciliation. More runner state is lower priority than preventing avoidable cascades; do not add a generic framework. |
| 3 | Audit copy prompt/seed (`J1`, reported Often) for async false-success or stale clipboard data. | Refuted: context-menu starts the clipboard call synchronously, awaits success and catches rejection. Captured image identity is deliberate; existing stub coverage checks values. No clipboard change or real clipboard access is justified. |

The ranking uses current local source and observed owned test runs. Card/user
gates, explicit feature deferrals and actual denied targets retain their
existing owners. This step option proves no fake-server identity or permission.

## Minimal contract

Add optional boolean `fatal` to a step containing only a nonempty string
`expect`. Validate even `fatal:false`; ambiguous action keys or malformed
values are refused with exit 2 before screenshots, profile work or spawn.
Reuse the existing async-helper validation and evaluate semantics.

An unsuccessful fatal expectation, including false/zero/undefined, a throw
or a rejected promise, reports/counts failure and stops later steps. End the
step loop naturally so ordinary Browser.close and cleanup run; exit 1. A
successful fatal expectation continues. Ordinary eval and nonfatal expect
keep their current diagnostic/aggregate behavior. No new CLI mode, production
module, request mechanism, browser selection, launch flags or cleanup path.

Initially convert only three new owned scratch setup expressions: picked
favorites, saved-search write recovery and favorite write ordering. These
async expressions explicitly return true. Also mark favorite ordering's
existing initial pair-selection expectation fatal, directly preventing the
observed offscreen-pick cascade. Do not sweep all older smokes.

## Qualification

First use the unchanged ordinary driver on about:blank to record a harmless
throwing eval followed by a unique log marker. Its continuation is current
behavior, not a product-data mutation. Use a new owned temporary root; no
app/server/gallery, credentials or clipboard is touched.

Cold tests execute the real harness with instrumented builtins, following
existing capability/diagnostic tests. Malformed fatal metadata must refuse
before filesystem output/profile/spawn/network effects; valid true/false
metadata must pass preflight to an instrumented downstream sentinel without
launching. Keep real GUI browsers out of CPU pytest.

After implementation run tiny sequential ordinary Edge probes on about:blank:
false, throw and rejected promise with fatal true must exit 1 and omit their
later marker; successful fatal continues/exit 0; fatal false and ordinary
nonfatal false still continue/exit 1. Check each new owned profile is cleaned
through the unchanged path. A deliberately failing probe is expected evidence,
not a table smoke to register under --all. Stop any actually denied browser
operation, preserving independent cold/source work.

Run the three converted positive scratch smokes sequentially, targeted cold
tests, classic/docs maps, lint and independent review. Runtime probes establish
driver control flow/cleanup only; they add no app, GPU or release acceptance
and complete no older denied artifact reads.


## Baseline driver evidence

The unchanged ordinary Edge driver ran a throwing setup eval on about:blank,
then printed the unique later marker and exited **0**. Its owned profile was
cleaned (zero leftovers). This is the existing diagnostic-only eval contract;
it establishes why callers need an explicit prerequisite instead of assuming
throw aborts a smoke. No app/server/gallery or clipboard operation was involved.
The retained receipt points to `C:\Users\matsi\AppData\Local\Temp\auradream-fatal-baseline-1jqhpnbb`. Source/metadata changes
were applied only after this probe and independent review.


## Final implementation and qualification

The driver validates optional strict boolean fatal metadata before output,
profile or spawn effects. Only an expect-only step with a nonempty string
may carry it. A failed fatal expectation keeps the existing failure count,
prints its stopping step and breaks to the unchanged graceful close/cleanup
path. Ordinary eval/evaluate/nonfatal behavior, browser discovery, launch
arguments and profile handling are unchanged. No new module, CLI mode,
dependency or general step framework was introduced.

Three existing scratch setup expressions now explicitly return true from
fatal expectations; favorite ordering's initial picked-pair condition is also
fatal. Later behavioral assertions retain aggregate diagnostics. This makes
the fixture's prerequisite explicit without changing the tested app flow.

Actual current-repository cold/scope/docs checks passed **131 tests** (96
instrumented capability/metadata/output/diagnostic checks plus 35 classic/docs
checks), with Ruff, Node syntax and diff checks clear. The new negative
controls refuse filesystem output/profile/spawn/network operations, prove the
sentinels remain active, and keep actual GUI/CDP execution outside pytest.
These cold checks qualify preflight only.

Six sequential ordinary Edge probes on about:blank qualified runtime behavior:

| Probe | Exit | Later marker | Owned profile leftovers |
|---|---|---|---|
| Fatal false | 1 | Absent | 0 |
| Fatal throw | 1 | Absent | 0 |
| Fatal rejected promise | 1 | Absent | 0 |
| Successful awaited fatal | 0 | Present | 0 |
| Explicit nonfatal false | 1 | Present | 0 |
| Ordinary nonfatal false | 1 | Present | 0 |

Negative probe exits are expected evidence, not failed product acceptance.
The retained receipt points to `C:\Users\matsi\AppData\Local\Temp\auradream-fatal-final-rk1q7azx`. No app/server/gallery,
clipboard or credentials were used. All probes followed the existing browser
selection/flags and graceful cleanup, without a new access route.

All three converted positive scratch smokes passed sequentially (**3 of 3 in
66 seconds**): picked favorites (86 steps), saved-search write recovery (134)
and favorite ordering (305). Artifacts are under
`auradream-smoke-artifacts-jt1nsu8z`. Independent source/test reviews found no
blocker. No new visual behavior was introduced, so no additional app visual
acceptance is claimed. The older denied targets, user/card gates and runtime
qualification remain separate in their owners.
