# Closed Grid preview ownership

**Status:** `UI#29` implemented and locally qualified, 2026-10-29 UTC.
Independent source/VM/native reviews, original controls, positive targeted
checks, two existing browser regressions and three new screenshot inspections
are complete. No model or pixel acceptance is claimed.

## Evidence and ranked choice

1. **MED, selected: closing Grid leaves its pending preview valid.**
   Native Grid is `details#grid-section` in `web/index.html`; a summary click
   closes it and emits `toggle`. `web/app.js` registers the actual
   `previewGrid` function on that event. Its closed branch clears
   `gridMaxCfg` and calls `syncCfg`, then returns before advancing
   `previewEpoch`. A preview already awaiting `/api/grid/preview` therefore
   still owns the current epoch. Its delayed success restores old CFG reach;
   its delayed refusal can replace preview text with an obsolete error.
2. **LOW, separate and unselected: old results during the input debounce.**
   `schedulePreview` starts the next request after 250 ms without immediately
   retiring the former request. A brief old count can publish during that
   delay. This proposal does not change debounce, scheduling or edit semantics.
3. **No backend variable defect established.** Existing grid tests already
   cover frozen random vectors, variable independence, shared draws, seed wrap,
   limits and resume identity. No speculative expansion rewrite is proposed.

The reported-use evidence is prompt variables Often (`C7`) and steps, CFG,
consecutive seeds and crossed axes Sometimes (`C1/C2/C5/C8`) in
[preferences](design/preferences.md). The existing
[preview design](design-notes.md#grid-preview-and-the-per-step-estimate)
describes its server counts and resolved settings as planning feedback.

## Reachable presentation effect

The real `syncCfg` in `web/sizes.js` combines `gridMaxCfg` with the panel CFG.
With Krea 2 Turbo's supported negative, threshold zero and no separate True
CFG, a panel CFG of zero disables the negative. A grid containing CFG one
offers the negative for those cells and says `(for grid cells above CFG 0)`.
After a native close, a delayed old preview can restore that grid-only caption
and re-enable the negative while Grid is closed. The input text itself remains
present. `negativeSent` also observes the disabled flag; this establishes a
collection-affordance mismatch, not a measured backend normalization or image
change. No GPU, model execution or actual-pixel claim follows.

## Bounded repair

Advance `previewEpoch` at entry to the existing closed-fold branch, before
its current synchronous clear and return. All older success and error replies
then fail their existing epoch guard. The model-hold increment, current-open
request capture, empty-prompt behavior, rendering, grid reveal, payloads and
variable grammar remain unchanged. Reopening captures a fresh epoch and may
publish its own response normally. Repeated closes make no API request.

This adds no cancellation/transport layer, automatic retry, backend change,
model identity policy, layout control or debounce repair. The native toggle
is the ownership boundary; an event not yet delivered is not claimed to have
retired a request.

## CPU qualification scope

`tests/test_grid_preview_close.py` contains ten actual-module cases. The
original-source results below establish three stale-reply failures and seven
ordinary controls; all ten pass after the bounded source repair.

It loads the actual complete declaration-only `sizes.js` and `grid-preview.js`
modules in a Node VM. It checks and executes only the exact current boot line
registering `previewGrid` on toggle, then dispatches through that captured
callback using inert details/API/DOM ports. Real `syncCfg` and `negativeSent`
must retain the closed negative state and caption after late success/error.
Two pending previews and repeated closes exercise the latest ownership edge.
Reopen-before/after-old, current success/error, model hold, already-closed and
empty-prompt controls preserve current behavior. Current success checks the
actual grid payload helper's variables/axes and real preview rendering.
`collect` remains an inert input port, so these checks do not qualify the
whole composer serializer or request normalization.

The VM has no page boot, real native event, network, browser, server, live
gallery or model. Native event order, focus, layout and actual preview bodies
belong to the separate browser fixture.

## Native qualification scope

`tools/smoke-grid-preview-close.json` uses a fresh default Krea 2 Turbo fake
on new port 7948 through the ordinary `run_smoke.py` runner, one worker. It makes
no generation or gallery mutation. GET/HEAD and the existing read-only grid
preview/prompt-analysis POSTs are the only allowed requests; other mutation
attempts fail the fixture. It retains and checks the planted records and
requires no jobs, a cold empty bin and no JavaScript/guard errors.

Physical summary clicks exercise the real native `toggle`. A transparent
listener wrapper observes the exact original handler's Promise completion;
it adds no preview logic, and the original summary handler remains untouched.
The first actual owned preview response is held. After physical close and
completion of its closed handler, release the old success and require stable
closed CFG/negative/caption/text and raw drafts. This is the meaningful browser
negative on original source. Reopen with a real current preview to show ordinary
counts, resolved variables and negative reach. A separately declared controlled
HTTP500, never forwarded, then checks obsolete-error suppression after close.
A further real reopen succeeds. The three new wide/current/narrow shots
qualify only the visible portions of this independent run, described below.

The exact original and positive commands are:

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp <fresh-owned-temp> tests/test_grid_preview_close.py
.venv\Scripts\python.exe -B tools/run_smoke.py smoke-grid-preview-close --jobs 1
```

The root registered the new smoke before its original-source control, then
applied source only after meaningful VM/native failures and independent reviews.
Positive actual-module, classic-script and docs checks remain separate from
native event/response and visual evidence. The ordinary Edge harness uses
one worker and fresh owned fake data for the new and existing regression runs.
Shared documents are staged only as owned hunks; prior pending work and all
stopped/unknown operations remain separate.

## Original-source controls (2026-10-29 UTC)

The ten actual-module VM cases produce three intended failures and seven
ordinary controls passing in 0.75 seconds. Late success restores CFG reach
two after the closed branch cleared it; late rejection writes obsolete text
and a warning; repeated closes still let the second old preview publish.
No SyntaxError or ReferenceError occurs. One installed TestClient/httpx
deprecation warning is reported. Full output is retained in the owned workspace.

The independent native run reproduces the same problem with an actual
read-only fake preview and physical summary clicks. The closing original
toggle handler has completed, closed CFG state/focus checks pass, and only
then the held real success is released. Fatal step 12 observes the obsolete
reply restoring closed reach/caption/preview. It stops later steps. Browser
exit is zero without a signal; runner exit is one from the intended assertion,
after six seconds (browser three). Final guards/screenshots are not counted
as passed evidence for this negative run.

The new root is `auradream-smoke-artifacts-4k570bf0`, child
`smoke-grid-preview-close-08meym4f`. No old denied artifact was accessed or
recreated, and no unknown GAL12/UI28 browser fixture was repeated.

The completion observer reattaches the unchanged preview handler after the
later persistence listener. That listener preserves raw Undo/storage state
and changes no preview epoch or CFG reach; unchanged full listener order is
not claimed. Raw test setup is programmatic; summary actions/focus are native.


## Positive qualification (2026-10-29 UTC)

All 45 targeted checks pass in 5.01 seconds, with no skips: the ten actual-
module cases, classic-script syntax/shared-scope guard and docs maps. One
installed Starlette TestClient/httpx deprecation warning is reported.

The full 40-step native fixture passes in eight seconds (browser five).
Eight original toggle bodies cover physical close/reopen. Four preview
attempts comprise three real forwarded reads and one explicitly controlled,
nonforwarded HTTP500. Old success/error leave closed CFG reach, negative
state/caption, preview text and raw drafts intact. Current reopen displays
the actual four-image count, resolved red/blue prompts and grid-only negative
reach. Summary focus and the before/after geometry check pass. Final guards
require unchanged planted rows/draft, no jobs/bin work, no JavaScript errors,
no forbidden mutations and every response gate released.

`grid-preview-closed-wide.png`, `grid-preview-current-wide.png` and
`grid-preview-closed-narrow.png` were read and visually inspected from
`auradream-smoke-artifacts-7c_iz1ik/smoke-grid-preview-close-_z3iih8u`.
They show the negative field's closed/current caption and retained draft at
wide/narrow widths. The complete Grid preview lies below their framing;
native state/count/geometry assertions and the existing Grid regression
provide separate evidence for it. No full-fold screenshot acceptance is inferred.

Existing `smoke-grid` passes in 33 seconds and `smoke-random-prompts` in four,
both serially on fresh data (52 seconds total runner time). These retain
ordinary resolved variables/counts, errors, budgets, persistence and Reset
Undo behavior; their own fake-job setup does not qualify real model output.
No-cache Ruff passes. Pinned offline Pyright 1.1.407 checks
146 files with zero errors/warnings in 12.148 seconds.

Independent review is clear at source `6f1538c4`, tests `ad96d0ca`, fixture
`92bb86e0` and original owner `4b68f24c`; rebased documentation anchors are
also checked. The newly owned design paragraph uses its actual implementation
date, October 29. Application code adds one epoch increment only. No old
denied artifact or unknown GAL12/UI28 run was accessed, recreated or repeated.


### Staged text identity

`tools/smoke-grid-preview-close.json` retains executed/frozen raw hash `92bb86e0`; Git's exact CRLF-to-LF form is `92083e1e` (137 physical line endings). Exact normalized byte equality is verified, and the fixture's parsed JSON is identical. No content/assertion or application behavior changed; no rerun is inferred or needed for this serialization-only difference.
