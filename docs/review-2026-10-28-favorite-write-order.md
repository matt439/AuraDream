# Review and plan: overlapping favorite writes, 2026-10-28

**Status:** `GAL#9` reproduced, repaired and locally qualified for
ordinary confirmed same-page favorite requests.

This is the next ranked source investigation from the
[saved-search recovery review](review-2026-10-28-saved-search-recovery.md),
under Matt's autonomous-work request. Existing hardware, decision and denied
operation boundaries remain in their owners.

## Ranked shortlist

| Rank | Current source evidence | Value, scope and risk |
|---|---|---|
| 1 | `setFavorite` and `starPicked` independently send explicit values and apply that captured value on each successful reply. Opposite writes for overlapping ids are currently allowed. Storage serializes actual writes but promises no FIFO HTTP request order. | Starring is reported Often (`F5`). A controlled actual-write/reply-order diagnostic can prove whether the UI contradicts storage. Prefer a shared per-id in-flight hold for ordinary confirmed same-page requests, with disjoint ids usable. Bounded UI scope, moderate regression risk across cards/sheets/viewer/lightbox. Selected. |
| 2 | A lost favorite reply can leave an unknown outcome, and another tab can change the row later. There is no favorite revision/CAS or operation receipt. | A stronger general ordering contract needs a broader API/storage and recovery design. Neither a latest-issued response token nor a GET alone proves actual commit order/completion. Do not claim the bounded same-page fix establishes this. |
| 3 | The smoke runner can lack a final summary after desktop transport loss while completed per-smoke receipts survive. | Possible diagnostic hygiene, but existing receipts already enabled safe reconciliation. Keep below the reported common product task and avoid a speculative runner framework. |

Source facts: `web/gallery-actions.js`, both favorite routes in
`backend/api_gallery.py`, and `storage.set_favorite`/`set_favorites`. The
ranking uses current local source and recorded tasks, not competitor claims.

## Reproduction before implementation

Use the ordinary Windows Edge harness on 7940 with fresh scratch FLUX.1 data.
Create three owned synthetic images, then navigate afresh. All favorite/menu
actions are native; bounded direct GETs prove actual stored values.

1. Let batch true commit and hold only its response. A disjoint single-image
   star remains usable. Then commit/respond batch false and release the older
   true response; inspect actual storage separately from all visible records
   and pressed stars.
2. Repeat across single true and overlapping batch false, proving the paths
   need one shared policy rather than separate guards.
3. Delay the first request before dispatch, commit the later-issued opposite
   request, then dispatch the first. This controlled delivery case establishes
   that issue order and commit order differ; a latest-issued token is not a
   sufficient repair. It is not evidence of spontaneous backend scheduling.

The baseline intentionally logs/asserts current incorrect behavior where
reproduced. A successful diagnostic exit is not a fixed-behavior pass. Do not
apply a patch if earlier fixture or isolation prerequisites fail.

## Bounded repair and qualification

If reproduced, claim every affected id synchronously before the first await,
across individual and batch favorites. Refuse a whole overlapping batch with
a clear wait reason, retaining that claim until the response is applied.
Allow disjoint ids. Repeated F must be guarded at the action function, not
only through button styling, and must not queue a stale toggle value.

Use aria-disabled/title reasons while keeping the Star name, glyph,
aria-pressed, focus and geometry. New/detached card, sheet and viewer buttons
need explicit ids when synchronizing their pending state. Lightbox bin and
pending reasons compose without rerendering its view just for a hold.
Only Picked's Star/Unstar items are held on overlap: its opener, Compare,
Bin and Seeds retain their existing availability. Selection changes recompute
overlap. Preserve successful propagation to hidden/stashed picks, existing
binned/missing handling and announcements.

Define error/unknown-outcome cleanup precisely and keep its limits explicit;
do not call an error proof that the write did not happen or present a read as
a commit-order receipt. No automatic retry, server/API change or cross-tab
guarantee is part of this selected repair.

The final focused smoke must invert the overlap assertions, prove no second
overlapping request dispatch, disjoint progress, release and deliberate later
opposite writes, individual/batch/F consistency and current/newly rendered
star controls including the lightbox. Include success/error cleanup, hidden
picks, retained order/composer and no image generation beyond owned fake
fixture creation. Run affected gallery/selection/lightbox/sheet checks
sequentially, classic-scope/docs-map/lint checks, independent review and
relevant visual containment/state checks before a local commit.


## Controlled baseline evidence

The first 77-step attempt did not reproduce the product issue: its fixture
scrolled the grid so the footer's Select was above the viewport. No pair was
picked, batch requests never dispatched, and downstream assertions cascaded.
No production patch was applied from that run. Removing that scroll and
adding native hit/selection prerequisites corrected the fixture.

The corrected **83-step** diagnostic completed all intended current-bug
assertions on unpatched source after `c7d8df4` (16 seconds). Select and both
owned image hit targets were visible; the native pair was exactly ids 3/4.
Case A committed batch true, then disjoint id 5 true, then batch false;
delivering the first response last made UI records, picked snapshots and
pressed stars true while storage for ids 3/4 was false. Case B reproduced
the same contradiction across a held single true reply and a later confirmed
batch false. Picks/order/composer and the disjoint star were retained.

Case C held the earlier request before dispatch: the later-issued false
committed at observation ordinal 8 and the earlier-issued true at ordinal 9.
Storage and current UI then correctly ended true. A latest-issued token would
discard the reply describing the final actual mutation. This is a controlled
counterexample, not a claim about spontaneous scheduling. Fresh artifacts
are under `auradream-smoke-artifacts-amxbasc0`; the earlier fixture-only failed
run is `auradream-smoke-artifacts-dlsb4yy7`. Diagnostic success establishes
the current defects and repair constraint, not fixed-behavior acceptance.

The proposed hold is limited to requests currently owned by this page. Its
finally cleanup releases the UI after errors, keeping existing failure
wording and permitting a later deliberate action; that does not establish
whether a lost request committed or prevent another tab from changing the
row. No automatic retry or stronger API ordering guarantee is introduced.


## Final result and qualification

One shared in-flight id set now guards individual and batch favorites before
their first await. Claims remain through successful propagation; a whole
overlapping batch is refused before dispatch, while disjoint ids proceed.
The original API-only catches remain nested inside finally cleanup, so UI
errors cannot be mistaken for unknown-image HTTP failures. No request queue,
response token, API, storage format or automatic retry was added.

Star controls receive explicit ids even while detached. Claim/release updates
only held attributes/title using the displayed confirmed state, preserving
glyph, Star name, pressed state and focus. Lightbox hold updates avoid its
render/ladder/view work and preserve the bin reason. Only Picked Star/Unstar
items hold on overlap; its opener, Compare, Bin and Seeds retain their normal
availability, with overlap recomputed when selection changes.

The final **305-step** Windows regression passes on a new scratch fake:
native batch/single/F overlap refusal sends no second request, disjoint ids
finish independently, success releases claims and a later deliberate opposite
action works. Fresh gallery reads and current/new card, sheet, viewer and
lightbox controls retain correct confirmed values and held state. Known
pre-dispatch 422 and actual owned-image 404/bin cases release cleanly.
Hidden active/stashed picks retain confirmed stars and order, with independent
bin picks untouched and composer/page-error checks clear.

The last 33 steps explicitly exercise a separately cached sheet source with
`recordsWithId(sourceId)` empty. Actual owned A/B reads plus client-only
B-as-pass metadata construct that state; native source Star/F uses the real
fake favorite endpoint. Hold release preserves the newly confirmed DOM value,
then `starCell` adopts it into the cached source, and the opposite toggle
works too. Backend B's refine metadata remains null and no new job is created.
Detached viewer construction similarly qualifies pending-control constructors,
not a complete generation/refine workflow.

The focused run passed in **31 seconds**, with artifacts under
`auradream-smoke-artifacts-bi1fo1k1/smoke-favorite-write-order-azoi6dj0`.
All five existing gallery, sheet, lightbox, selection and hidden-pick smokes
passed sequentially on the same production source (**5 of 5 in 327 seconds**),
under `auradream-smoke-artifacts-mbqs8lia`. Only the final focused fixture was
extended after those passes; production source did not change. Its Reveal now
precedes geometry capture, avoiding an offscreen/scroll fixture false failure.

Classic-scope/docs-map checks pass **35 tests**, with Ruff/diff checks clear.
Independent source and final-fixture reviews found no blocker. Four final
screenshots were inspected: desktop gallery, newly opened pending lightbox,
400x650 sheet and 1400x600 sheet. Hold labels, focus and layout remain contained;
no styles changed. All writes/images were owned fake scratch data. Unknown
favorite outcomes, other tabs, real model/card behavior and older denied
artifact/download targets retain the limits described above.
