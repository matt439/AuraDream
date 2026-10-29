# Review and plan: broad Bin/Restore ordering, 2026-10-28

**Status:** `GAL#11`'s broad-operation repair is implemented, reviewed and
locally qualified after actual defect reproduction and
[exact-id ordering](review-2026-10-28-bin-restore-order.md).

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


## Actual baseline and source checkpoint

The initial 52-step run stopped at its fatal seed-isolation gate (2 browser
seconds); no lifecycle work or source repair was accepted from it. The fake
always plants two other-checkpoint rows, including a star. The fixture was
corrected to preserve those rows, using current-model emptiness plus global
bin emptiness and an exact snapshot of the planted pair. It does not assume
fixed owned ids. `steps:1` is the strict native search/deletion scope, validated
against every deliberately seeded row; free-text deletion is not used.

The corrected **53-step diagnostic passed in 12 seconds** (6 browser seconds)
on 2026-10-28, with actual mutations around 15:46 UTC. Native Bin A (#3)
committed; native Restore committed but its reply was held. A native fresh
search exposed all six owned rows, and native Delete unstarred/count 6 moved
ids 3-8 back to the bin while A's exact hold remained pending. Releasing the
old Restore cleared A from `binnedIds`/`deletedIds` and consumed its cached
snapshot, announcing `Restored #3.`. Independent actual reads still showed
all six owned rows binned, current-model live 0, all-models live 2, and bin 6.
The old gallery epoch kept the grid empty: no card resurrection is claimed.
The planted pair's serialized records were unchanged; all eight PNG URLs
returned 200, proving presence rather than byte equality. Exactly three
reversible writes, one seed job, zero blocked/unexpected writes and zero app
errors were recorded. No permanent/bundle operation or real gallery was used.

Diagnostic artifacts:
`auradream-smoke-artifacts-jig97li3/smoke-broad-bin-order-_05psf9p`.
This pass is defect evidence, not repaired behavior.

The reviewed repair adds one token for Delete unstarred/Empty. Exact and
broad entry guards synchronously refuse each other; a broad owner excludes
all ids, without estimating its scope from loaded rows. Exact-disjoint work
remains possible outside a broad owner. All four existing critical reset
paths are awaited before finally releases. Those reads settling does not
assert they succeeded or provide a universal background-read protocol.
Attribute-only availability sync preserves the head's layout, names, notes
and lightbox. The count/keeper/revision and Empty expected/pin/before payloads
are preserved. One captured All models value supplies request and wording.
Bundle clear and the stated unknown/cross-client limits remain separate.

Independent source reviews found no blocker. Initial current-source validation
passed 35 classic/shared-scope/docs tests and no-cache Ruff. Final evidence follows.


## Final qualification

All three existing affected smokes passed sequentially in 82 seconds:
Delete (25 browser seconds), Bin (14), and the 420-step exact-id ordering
regression (27). Artifacts: `auradream-smoke-artifacts-3g6cf83a`.

The first 346-step final fixture stopped at fatal gate 55 before a broad
write: a count-only wait accepted the old two-row result before a debounced
filter loaded the next two rows. Scope/toggle/reset prerequisites now require
fresh epochs, exact owned ids and the correct run/mode. No blind sleep or
selector substitution was used. Two subsequent 352-step runs passed their
ordering/state checks but failed the final API guard count. Added live path
logging identified two `/api/runs/<owned-run>/finish/preview` requests, with
no page errors or extra enqueues. Source inspection confirmed this route
returns a read-only plan; the separate `/finish` route submits jobs.

The final guard permits only exact encoder measure and the two exact encoded
owned-run preview paths. Nine isolated mock-API checks proved allowed/refused
cases, including foreign runs, query variants and actual Finish submissions.
API POST/DELETE guards remain scoped; no universal PUT/PATCH/raw-fetch guard
is claimed. The fixture invokes no raw mutation fetch. Initial owned seeding
uses its explicit base API; no real gallery or model execution occurs.

The corrected **352-step regression passed in 26 seconds** (19 browser
seconds), with all 111 fatal gates intact. It verifies hidden exact-owner
admission, whole broad exclusion through preview/question/reply/critical
reset, cancellation/wrong-count cleanup, injected predispatch 409 cleanup,
fresh controls/notes and captured All models request/wording despite a later
checkbox change. Actual stored state confirms the two swept rows were later
restored, C/D picks and stashes survived, and only the two explicitly approved
owned E/F records were permanently removed by the one-use native Empty.
Actual Empty expected/pin/before matched its server preview and bin ids.
Planted projected metadata stayed unchanged, composer stayed intact, all
owners released, and no unexpected jobs, forbidden writes or page errors
remained. Constructor/direct-function checks are labeled separately from
native UI actions.

Final artifacts:
`auradream-smoke-artifacts-4t3wqiwl/smoke-broad-bin-order-gnsnm7pc`.
Both new 1400x900 screenshots were read and visually reviewed: held sweep
reply with retained picks and held Empty reply with bin picks. Names/focusable
holds and stable geometry remain usable. This is desktop fake-data evidence,
not hands-on assistive-technology, touch or GPU qualification. Independent
source and fixture reviews found no blocker; classic/docs, no-cache lint and
diff checks were clear.

The next source slice is `GAL#12`, [Bundle & clear coordination](review-2026-10-28-bundle-clear-order.md).
No bundle worker/unknown-outcome/cross-client protocol is claimed here.
Existing actual access denials and user/card gates stay recorded in their
separate owners; none was retried for this qualification.
