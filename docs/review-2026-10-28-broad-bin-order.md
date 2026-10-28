# Review and plan: broad Bin/Restore ordering, 2026-10-28

**Status:** `GAL#11` selected for controlled reversible reproduction and a
bounded source repair after qualified [exact-id ordering](review-2026-10-28-bin-restore-order.md).

## Ranked next work

1. Coordinate Delete unstarred and Empty bin against each other and every
   exact-id lifecycle owner. Their full prospective ids are server-owned;
   a loaded-card approximation cannot exclude stale reply application. These
   common existing workflows justify a small page-local broad owner.
2. Bundle & clear has a long-running worker/status lifecycle and remains a
   separate source slice. A POST 202 is not terminal cleanup; ordinary Back up,
   import and automatic-backup semantics must remain intact.
3. A general version/CAS or unknown-outcome/cross-client protocol is wider API
   work. Do not claim either earlier page-local slice supplies that protocol.

## Evidence and bounded plan

`deleteUnstarred` and `emptyBin` currently bypass `binWriteIds`; a prior
Restore can apply after a later broad server mutation. Reproduce with owned
fresh fake rows: native Bin A, hold an actual committed Restore reply, fresh
search exposes A, native count-confirmed Delete unstarred bins A, then release
the old Restore. Independently read actual binned state and compare cached
snapshots/cards. Diagnostic pass means defect proof, not repair acceptance.
No permanent or bundle operation is needed for this baseline.

If confirmed, add one exclusive broad owner for just Delete unstarred/Empty.
Broad admission refuses any exact owner (even an unloaded id) or another
broad owner. Exact admission sees broad ownership first, while disjoint exact
operations remain concurrent when no broad owner exists. Claim before the
first lookup/await and retain through original confirmation, critical response
application and each explicit awaited `resetGallery`; release in finally on
every cancellation/refusal/error. Token identity prevents accidental release
of someone else's owner. Refresh only aria/title on existing controls, plus
the normal picked/broad sync helpers; preserve note and lightbox geometry.

Preserve request/confirmation semantics: Delete unstarred's server keepers,
count-pinned expected scope and model revision; Empty's expected count, exact
server pin and fixed age cutoff/before. Do not invent an immutable id receipt
for count-pinned sweeps. Capture `all_models` once for both Delete unstarred
request and preview wording; the current code re-reads the checkbox after an
await and can describe a different scope. Do not lock filters or add a new API.

## Qualification

Use a fresh single-worker ordinary Edge fake with explicit owned ids and fatal
isolation/native prerequisites. Prove targeted-held broad refusal with no
preview/confirm/delete dispatch; broad preview/confirm/reply-held exact and
second-broad refusal; new card/lightbox/note/lineage controls; cancellation,
wrong count, known refusal and actual success cleanup. Delay the broad's
explicit reset response and verify the hold lasts until it applies. Verify
stored state and preserved composer/picks/caches. Existing Delete/Bin tests
retain keeper, model/search scope, pin/same-count and age behavior. Permanent
Empty checks may touch only owned fake records and retain native confirmations.
Run appropriate existing smokes, classic/docs checks, no-cache lint and
independent/visual review before a local commit.

## Explicit remaining Bundle boundary

Future clear ownership must survive request completion/dialog closure through
terminal `receiveBundleStatus` deleted-id bookkeeping and awaited refresh.
`startBundleOperation.finally` is too early. A failed POST plus
`await pollBundleStatus()` does not prove a fresh status: a pending poll can
return immediately and request epochs can discard an older observation.
Known 409/422 pre-reservation refusal differs from unknown transport/5xx or
malformed success. Do not silently retry or release uncertain clear ownership.

Adopting an already-running export must hold new writes. An earlier local
target/sweep owner may still apply a reply, so a later design must drain those
owners before terminal clear bookkeeping, without overwriting their token or
prematurely marking the terminal id handled. Preserve first-page already-done
status as past, ordinary backup/import behavior and existing epochs/poll cleanup.
These are source design obligations, not a request for user/gallery activation.
