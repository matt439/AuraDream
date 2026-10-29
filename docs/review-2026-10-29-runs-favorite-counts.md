# Review: Runs favorite counts after confirmed stars, 2026-10-29

**Status:** `GAL#19` implemented and functionally qualified, 2026-10-29 UTC.
Independent source/VM/native reviews, measured controls, positive actual-
module/native checks and three related browser regressions are complete.
New screenshot inspection remains open after the result artifact read was
denied. This does not claim complete visual acceptance.

## Source finding and selected scope

Ordinary `setFavorite` success applies the confirmed value through
`markFavorite` and the existing announcement, but never reloads Runs.
`starPicked` applies each `changed`/`already` id and reloads Runs only when
the result also contains `binned` or `missing` ids. A run's displayed star
count therefore remains stale after the ordinary successful star actions.
The partial-result and single unknown-image paths already ask again.

`gallery_reports.runs` already aggregates live stored favorites. The favorite
routes await `storage.set_favorite`/`set_favorites` before returning success;
these persistence functions update SQLite and read/classify rows without a
PNG rewrite. No backend change or client-side count arithmetic is needed.
This is a bounded correctness repair to the reported frequent gallery/star
workflow. Global read batching and favorite revision/receipt protocols would
add a different lifecycle and are outside this repair.

## Contract

- After propagating a successful single favorite reply and invoking any
  applicable existing announcement path, start one `void loadRuns()`.
- After propagating a successful bulk result, forgetting unavailable ids and
  invoking its existing note path, start one `void loadRuns()`, including
  `already` and partial outcomes. Replace
  the existing partial-only refresh, so a result still starts one read.
- Keep `markFavorite` free of network work: bulk calls it once per image.
  Keep the existing single unknown-image refresh at one. Empty, binned or
  overlapping admissions and ordinary failed POST replies add no read.
- Do not await the aggregate read or put it in `finally`. Favorite ownership
  releases after its own confirmed propagation even if the independent Runs
  read is still pending. A failed aggregate read cannot revoke a confirmed
  star and uses the existing Runs recovery control.
- The read captures the current model/revision/All-models/known-server context
  when the favorite reply settles. `GAL#18` owns current-context publication,
  latest read epochs, deliberate retry and preserved same-scope keyboard
  candidates. Keep the same select element and focus; successful refresh may
  replace Option nodes while retaining a still-published pending run id.
- Once means once per confirmed action. Disjoint successful actions may each
  start one read. `GAL#9` same-page overlap ownership and its error/unknown
  outcome and cross-client limits stay unchanged. A fresh read is not a write
  ordering receipt or proof that an earlier failed write never committed.

## Qualification plan and evidence boundary

`tests/test_runs_favorite_counts.py` loads the complete actual
`gallery-actions.js` into Node VM. Deferred API/aggregate reads and inert
presentation ports observe the real favorite admission, propagation, catches,
refresh call sites and `finally` cleanup. Its 17 cases cover confirmed star
and unstar, multi-id fanout, already/partial results, overlap/disjoint progress,
empty/bin guards, ordinary/unknown failures, held reads, a delayed disjoint
failed reply and current-context invocation. Existing `markFavorite` remains
real; unrelated deletion presentation and announcements are observation ports.
These checks do not qualify DOM layout, native keyboard behavior or actual
storage. The original-source results below measure ten missing-refresh
failures and seven existing-behavior controls; native results separately
observe actual server values and keyboard behavior.

`tools/smoke-runs-favorite-counts.json` is a new **42-step** fresh-data FLUX
fake fixture on **7949**, with 19 fatal checks and three screenshot steps.
It uses the actual two other-model planted rows/run, initially
one starred. Exact owned favorite POST grants and actual Runs GET bodies
establish one confirmed single action changing the count to two, then one
bulk unstar changing it to zero. Held successful replies prove no early read;
an explicitly nonforwarded delayed 500 on a disjoint star must add no read
after the successful summary publishes. Labeled DOM setup focuses the closed
select, then actual native ArrowDown establishes its pending run candidate.
That candidate and the same select node/focus must survive the single refresh;
Escape cancels that candidate before the fixture deliberately blurs it.
Native picks/menu drive the bulk action. Successful Option-node identity is
not asserted. No generation, model activation, bin, delete or broad write is
admitted. This is owned fake data, not real user-data qualification.

The fixture checks confirmed stored values separately from card and option
state. Any PNG byte check is a direct comparison of the owned fake files,
not a claim based only on the SQLite source contract. Native popup-open
selection behavior, other tabs, unknown favorite outcomes and hardware/model
execution remain outside these checks. No old unknown browser run or denied
artifact target is retried by this fixture.

The root measured the new VM/native controls before applying the two source
hunks, then ran positive checks and related regressions serially. The normal
Edge harness, launch settings and artifact policy stayed unchanged. Shared
documents are staged as owned hunks only. No acceptance is inferred from a
prepared packet or from inaccessible artifacts.

## Original-source controls (2026-10-29 UTC)

The 17 actual-module cases produce ten intended missing-refresh assertion
failures and seven existing admission/error controls passing in 0.93 seconds.
Confirmed ordinary single/bulk paths make zero reads where one is required;
unknown-image/partial and refusal controls retain existing behavior. There
are no SyntaxError or ReferenceError failures. One installed TestClient/httpx
deprecation warning is reported. Full output is retained in the owned workspace.

The original native fixture independently verifies the real favorite POST
committed two favorites while its response is held: a passive actual server
GET reports two, and the application still displays one. Before release,
duplicate same-id admission is held and a disjoint controlled error remains
unforwarded. After the successful response is released, fatal step 16 times
out waiting for the required application Runs refresh and two-star label.
It stops later checks. Browser exit is zero/no signal, runner exit one from
the intended assertion; runner 30 seconds, browser 23. Later final guards
and screenshots are not claimed as evidence for this negative control.

The new root is `auradream-smoke-artifacts-yjvnwdoj`, child
`smoke-runs-favorite-counts-u2ca_gv0`. No old denied artifact or unknown
GAL12/UI28 run was accessed, recreated or repeated.

Announcement ordering here means invoking the existing presentation path.
`sayStar` queues its polite text on the next frame; no screen-reader timing
or completed live-region delivery before the aggregate GET is claimed.


## Positive functional qualification (2026-10-29 UTC)

The initial focused run reports 80 passes and one failure in 6.47 seconds.
The failure is a new test's unreachable-on-original-source note assertion:
it expected `already not starred`, while the unchanged successful bulk note
is `#1 and #2 were not starred.` The root corrected only that assertion to
the exact existing message; the affected case passes in 0.24 seconds, with
16 cases deselected. Thus 81 distinct checks are qualified across those two
runs; this is not a claim of a second complete focused run. The original
ten/seven split remains valid because that case first failed at the missing
read assertion before reaching its note check. One installed TestClient/httpx
deprecation warning is reported. The frozen proposal remains unchanged.

All 42 native steps, including 19 fatal checks, pass in 13 seconds (browser
six). Actual application aggregate bodies report favorite counts one, two,
then zero. Two exact owned favorite POSTs reach the fake server; a third,
delayed 500 is explicitly not forwarded. Held success responses add no early
read; each confirmed single/bulk action adds exactly one aggregate read.
The actual Runs node, focus and native pending run candidate survive each
refresh. Picks, composer, search, records beyond the two favorite flags and
the owned PNG bytes remain unchanged. Final guards report zero forbidden
mutations, JavaScript errors, unintended run opens or jobs, and all declared
response gates settle. Keyboard setup and popup-open limits remain as above.

Existing `smoke-gallery`, `smoke-favorite-write-order` and
`smoke-runs-scope-recovery` all pass serially in 113 seconds total. Gallery
reports 64 seconds and Runs recovery eight; no individual favorite-order
duration is inferred from truncated output. These use their own fresh fake
data and existing synthetic job setup, not actual model/hardware execution.
No-cache Ruff passes. Pinned offline Pyright 1.1.407 checks 146 files with
zero errors or warnings; its measured duration is retained in the workspace
journal. Documentation maps are checked again after this evidence update.

Independent reviews are clear at application source `ce327c19`, frozen
tests `df10700a`, fixture `0e22c093` and original owner `1c89ee32`. The
reviewer independently confirmed the root's exact assertion-only correction
to tests `b88f614a`. No favorite admission, payload, write-order or failure
semantics were expanded.

## Remaining visual qualification and access boundary

The runner reports the new screenshots written under
`C:\Users\matsi\AppData\Local\Temp\auradream-smoke-artifacts-u67ch_a4\smoke-runs-favorite-counts-9fb6bxdk`.
The subsequent exact `result.json` read there returned **Access denied**,
followed by the shell's secondary not-found message. No artifact contents
were returned. Reads of this whole new artifact directory are stopped:
no screenshot read, alternate reader, copied artifact, elevated read or
duplicate focused run was attempted. This is an actual filesystem denial,
not an uncertain browser outcome: the completed runner independently reports
all steps passing and normal browser exit. Wide/narrow screenshot inspection
remains incomplete, so `GAL#19` remains in TODO for that exact action.

The bounded source implementation has actual-module and native functional
qualification; retaining this separate visual item does not qualify a layout
or actual OS/device acceptance. The earlier GAL12/UI28 unknown runs and all
previous denied artifacts remain untouched.
