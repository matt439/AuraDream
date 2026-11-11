# Review and plan: targeted Bin/Restore ordering, 2026-10-28

**Status:** `GAL#10`'s targeted lifecycle repair is implemented, reviewed and
locally qualified. Both baseline defects were reproduced before the repair. This follows the current source/owner audit under Matt's
autonomous-work request; no real gallery operation is authorized by a test
fixture or performed here.

## Ranked shortlist and bounded stages

| Rank | Evidence | Value, scope and risk |
|---|---|---|
| 1 | `restoreImages` permits overlapping calls and applies restored/not-binned ids through `unforgetImages`. That function consumes the current `binnedRecs` entry, which a later Bin can replace. A normal fresh read can expose a restored row before its first reply returns. | Bin one/picked images (`J4/J5`) are reported Often; Restore (`J7`) Sometimes. Reproduce actual stored binned state versus UI and add shared targeted operation holds if confirmed. Selected phase 1; medium UI integration scope. |
| 2 | Delete unstarred and Empty bin have server-counted/pinned scopes rather than prospective exact id lists; Bundle & clear has a separate maintenance path. | Loaded-id approximations cannot provide complete coverage. A subsequent source phase must select/qualify broad mutual exclusion, preserving confirmations, pins and backup semantics. Keep this remaining boundary explicit rather than claiming universal lifecycle order from phase 1. |
| 3 | Neither storage locking nor request issue order supplies an operation version/CAS receipt. | A general cross-tab/unknown-outcome ordering protocol would be substantially wider API/storage work. No response-token or snapshot-identity shortcut is justified. |

Phase 1 covers all UI operations with captured exact ids: reversible single
Bin/Bin picked/Restore and confirmed single permanent deletion/picked purge.
The latter must participate because `deleteOne` chooses its path from cached
binned state and the backend DELETE accepts live or binned rows. Preserve
all existing confirmation and deletion semantics; tests may use only owned
temporary fake data. No real irreversible operation is part of this loop.

Broad operations are a separate source follow-up, not a user/card acceptance
gate. Their remaining race boundary stays recorded after phase 1. Bundle
maintenance is not silently included or claimed qualified. Unknown outcomes,
other tabs and independent API clients retain their limits as in the favorite
ordering review. No automatic retry or server version protocol is introduced.

## Source-backed failure paths

`bin.js` Restore applies both restored and already-live ids. `unforgetImages`
clears local gone sets/source missing markers, then reads the **current**
cached Bin snapshot and can reinsert its row/card at the matching gallery
epoch. `keepBinned` replaces that snapshot after a later successful Bin.
`deleting` guards only another `deleteImage`, not Restore or Bin picked.
Backend workers serialize actual mutations but return classifications without
delivery-order receipts. Neither a latest-issued token nor comparing only a
captured cache object establishes the final actual commit order.

## Controlled diagnostic

Use a new owned FLUX.1 fake on 7941 through the ordinary single-worker Edge
harness. The 74-step baseline creates three fake images, navigates afresh,
uses fatal isolation/native/gate prerequisites, and independently reads each
actual row with `binned=true`. All tested mutations are native reversible
Bin/Restore; DELETE/purge is blocked and unexpected confirmations cancelled.

1. Bin A; Restore1 commits but its reply is held; Restore2 responds already
   live and puts the card back; Bin A commits again; release Restore1. Log
   storage, local gone sets, cached snapshot and visible card before asserting
   the intended current defect.
2. Repeat with an ordinary native search reading the restored row instead of
   issuing Restore2, then Bin and release the old reply. This tests that a
   duplicate-Restore guard alone cannot repair the case.

A diagnostic pass proves the current defect, not repaired behavior. Do not
apply source changes if earlier fixture/isolation prerequisites fail. These
are controlled commit/delivery sequences, not spontaneous timing claims.

## Phase 1 repair and acceptance

If reproduced, synchronously claim captured ids before dispatch, across the
targeted entry points. Refuse a whole overlapping operation, allow disjoint
ids and retain ownership through all response application (including error
paths that unforget rows). Guard functions as well as controls. Confirmation
flows claim their captured ids before the first lookup/await and keep ownership
through the existing question, write and response application. Cancel/refusal
releases in finally. Recheck captured ownership and picked selection before a
confirmed write, without queuing an old cached-state action or adding a GET
that would silently change the existing confirmed-deletion semantics.

Newly read card/viewer/lightbox controls inherit the id hold. Restore notes
carry their whole id set; lineage Restore carries the source id. Keep buttons
focusable with aria-disabled/title reasons, preserve names, geometry, cached
records, ordering and composer references. Restore never re-adds references.
Release through finally with existing outcome feedback and no automatic retry.

The final focused regression must prove no second overlapping write dispatch,
disjoint progress, fresh-read control holds, success/error release, native
single/picked/note/lightbox/lineage funnels and preserved selection/composer.
Qualify permanent/purge coordination only on owned fake records with existing
confirmations. Run affected Bin, gallery, selection, lightbox/context checks
sequentially, targeted cold/source checks and independent/visual review before
a local commit. Preserve broad-operation and unknown/cross-tab limits in the
completion evidence and follow-up backlog.


## Baseline evidence and implementation checkpoint

The ordinary single-worker Windows Edge diagnostic passed all 74 steps on
2026-10-28 (15:05 UTC). This means both defects were reproduced: each later
Bin committed, but releasing the earlier Restore made the page's current
snapshot/card appear live while an independent actual row read was binned.
The second path used a native fresh search instead of a duplicate Restore.
The run used three owned fake PNGs, zero permanent requests/confirmations,
and no additional generation jobs. Artifacts:
`auradream-smoke-artifacts-fssmr7wl/smoke-bin-restore-order-9g4csaht`.

The repair adds a synchronous per-id lifecycle hold across every exact-id UI
Bin/Restore/single-delete/picked-purge funnel. Newly read controls, all ids of
Restore notes and lineage source ids inherit it; disjoint ids remain usable.
Ownership covers the original confirmation lifecycle and error application.
No API/storage/confirmation semantics, retry queue or server-version claim is
added. Broad sweeps remain the independent source follow-up above.

A refused Restore on a focused note keeps the whole note text, nodes, focus,
geometry and lightbox dismissal controller unchanged. Changing its leading
text would move its inline button. The existing polite gallery/lightbox action
channel announces the wait reason instead; Star announcements reuse the same
unchanged message/timer behavior through the extracted helper. Other refusal
feedback keeps the existing error slot. No stylesheet change was needed.

Initial qualification: 35 classic-script/shared-scope and documentation checks
passed. Ruff's default `.ruff_cache` temporary-file write was denied by Windows;
that cache action was not retried. The independent `ruff check --no-cache .`
validation passed without touching the denied cache. Final evidence follows.


## Final qualification

All five existing affected smokes passed sequentially through the ordinary
Windows Edge harness: Lightbox, Gallery, Select, Bin and Context menu. The
command exited 0; its attempted `summary.json` read at
`C:\Users\matsi\AppData\Local\Temp\auradream-smoke-artifacts-xd9plq2j\summary.json`
was denied and was not retried or recovered through another route. This does
not invalidate the completed live exit/individual pass evidence.

The first 344-step focused fixture stopped at a fatal native Restore hit-test
gate (9 seconds); it did not qualify the repair. Reading its retained
`C:\Users\matsi\AppData\Local\Temp\auradream-smoke-artifacts-51wve610\smoke-bin-restore-order-4vwwfc5p\browser.log`
was denied and was not retried/copied. That exact failed step's index remains
unproven because multiple gates shared the expression. A separate new 42-step
reversible diagnostic printed geometry to live stdout and passed at the
unchanged 1400x900 viewport; the first note was visible, with no clipping.
Clipping was a hypothesis, not an established cause. Native nearest scrolling
and per-target diagnostics were retained without removing hit-test gates.

Independent fixture review also corrected its bundle prefix to the actual
`/api/gallery/bundles`, and made unexpected enqueue POSTs throw before
dispatch (including Swap tags and run Finish). All three installed helper
copies were checked with inert local stubs: zero forwards on forbidden routes.
The deliberate six-image seed uses the explicit base API before the guarded
workflow. Full-note snapshots now retain every descendant node, container
rectangle, text and controller identity. Direct function checks exercise the
Restore funnel during the held initial delete lookup and open single/purge
confirmation; native actions remain the tested UI elsewhere.

The strengthened **420-step final regression passed in 34 seconds** (27
browser seconds), all fatal prerequisites intact. It verifies whole-overlap
refusal, disjoint progress, fresh-read/new controls, both earlier stale-reply
paths, complete focused note preservation and polite wait feedback, known
predispatch 422 cleanup, cancellation, actual confirmation behavior, and
scoped single permanent/picked purge response ordering. Actual row reads and
PNG 404 checks confirm only the three deliberately approved fake records were
deleted. Viewer/lineage constructor-only checks are labeled as such. No real
gallery, model execution, credentials, unexpected jobs or broad operations
were used; composer state stayed unchanged and all claims released.

Final artifacts:
`auradream-smoke-artifacts-cs0380fv/smoke-bin-restore-order-e7voxavk`.
All three new 1400x900 screenshots were read and visually inspected: pending
new gallery controls, pending lightbox Restore, and pending picked purge.
Focused controls and names remain visible, the layout is stable and the
pending reason is present. This is desktop fake-data evidence, not hands-on
assistive-technology/touch/card qualification. Independent source and fixture
reviews found no remaining blocker; 35 classic/docs checks and the subsequent
six updated documentation checks passed, with no-cache Ruff and diff checks
clear.

`GAL#11` now owns synchronous broad Delete unstarred/Empty coordination in
[the next review](review-2026-10-28-broad-bin-order.md). Bundle & clear remains
a separate subsequent source phase. These are source boundaries, not user or
hardware acceptance gates. Other-tab/client and unknown-outcome ordering
remain outside this page-local repair; no automatic retry is implied.
