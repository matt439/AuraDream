# GAL#20: gallery pagination binding bounds

**Status:** `GAL#20` implemented and locally qualified, 2026-10-29 UTC.
Independent review, original actual-SQL failure controls, all 52 targeted
API checks, six documentation checks, lint and pinned type checking pass.

## Problem and choice

Before this change, the live gallery and recycle-bin GET endpoints
accepted arbitrary-size Python integers for `offset` and `before`. Their
lower bounds were correct, but `storage._page` binds both directly into
SQLite; a value above signed 64-bit MAX raised OverflowError rather than
the ValueError handled by the routes.
The same overflow problem was already recognized for search integer terms.

The ranked audit chose this concrete small robustness repair over an
unverified embedded-quote FTS hypothesis. Gallery browsing is reported Often
(E1), but ordinary UI cursors already come from valid stored IDs, so exposure
is malformed/scripted API requests. No claim of observed common-user failures
or measured speed gain is made.

## Bounded repair

Only `backend/api_gallery.py` changes: reuse the existing `search.SQLITE_INT`
constant and add `le=SQLITE_INT[1]` to the two routes' `offset` and `before`
Query declarations. This is the storage engine's integer binding domain,
not a business cap or an arbitrary gallery-size limit. Both defaults and
lower bounds remain unchanged; signed MAX remains valid.

`limit`, keyset/offset composition, ordering, counts/has_more, search,
favorites, model/all-model scope, bin semantics and storage implementation
remain unchanged. No additional endpoint/ID validation, lifecycle operation,
generic protocol or generation behavior is added. No web source changes.

## Cold HTTP qualification scope

`tests/test_gallery_page_bounds.py` contains 14 actual TestClient cases:

- Four MAX+1 query refusals across both parameters and routes. A spy
  forwards the actual storage call into the owned fixture SQLite database,
  recording OverflowError, and a TestClient with server exceptions disabled
  exposes the original route's HTTP 500. The repaired route must answer 422
  without calling that spy. No fake storage answer establishes the result.
- Four existing lower-bound controls at offset -1 / before 0, also before
  storage.
- Four actual-SQL MAX controls: MAX offset produces an empty page with
  unchanged total; MAX cursor still permits all eligible stored rows.
- Two route regressions exercise ordinary offset/keyset pagination and
  counts/has_more. The live path also checks exact model/all-model/search and
  Starred scopes; the bin retains its all-checkpoint contents.

Owned scratch images/rows are created only through the existing test store;
only fixture-owned IDs receive favorite/bin fixture flags. Model loading,
generation and queue admission are poisoned. Before/after row bodies and
PNG hashes remain identical across GETs, and no job is queued. The original
source measurement below records four upper-bound failures and ten ordinary
controls passing; the repaired source passes all fourteen.

CPU/API qualification is sufficient for this HTTP-only change. It provides
no browser, model/card, real-gallery or runtime activation evidence. All
actual denials, unknown GEN31/GAL12/UI28 outcomes, stopped source ranges and
pending work remain untouched. Shared documentation is supplied only as
exact owned anchors; TODO is root-owned.


## Original-source measurement (2026-10-29 UTC)

All four MAX+1 requests (offset/before on live/bin) return actual HTTP 500
with a recorded `OverflowError` from the forwarded owned SQLite call.
The other ten controls pass: existing lower-bound refusal, legal signed-MAX
SQL binding and ordinary paging/scopes. The full run reports four intended
failures, ten passes in 2.44 seconds, with one installed TestClient/httpx
deprecation warning. Rows and PNG hashes are checked unchanged before each
status assertion; cold-operation guards and empty-jobs teardown also pass.
This is a measured database/HTTP failure, not a fabricated response or an
assertion against the new declaration alone. Full output is retained in the
owned workspace. No browser, actual gallery or model/card acceptance is claimed.


## Positive API checks and documentation maintenance (2026-10-29 UTC)

The combined run reports 57 passes and one documentation-inventory failure
in 8.05 seconds. All 52 API checks pass: the fourteen new cases and all 38
existing gallery/bin API regressions. Oversized paging now returns named
query HTTP 422 before storage, legal signed-MAX values still execute real
SQL, and ordinary paging/counts/scopes remain intact. New-test snapshots
confirm unchanged scratch row bodies/PNG hashes and no model/queue entry.
One installed TestClient/httpx deprecation warning is reported.

The remaining failure is the repository's documentation inventory: the
unchanged untracked GEN31 diagnostic/registration survived the disconnect
without a Tests description. Its actual outcome is still unknown and its
fake-server log read was denied. A narrowly added Tests paragraph records
those facts, the intended diagnostic scope and unapplied source status. No
GEN31 operation, input, source proposal or acceptance was changed. Final
documentation verification below remains distinct from the 52 API results;
no second full API run is inferred.

No-cache Ruff passes. Pinned offline Pyright 1.1.407 checks 146 files with
zero errors/warnings in 12.159 seconds. Source `50f70a2a`, v2 tests
`51ea4271`, original owner `53c5ea33` and six authored anchor proofs are
independently reviewed. Two initial selected-document patch dry checks
stopped before any source mutation; bounded trailing context and merged
overlapping hunks corrected that preparation. No stopped rule/source range
or whole-rule snapshot was used to resolve those checks.

Only the API module changes at runtime. No browser, model/card, actual
gallery or activation acceptance is claimed. All prior pending work and
stopped/unknown operations remain preserved.

Final documentation maps: all six pass in 0.23 seconds after the inventory
correction. These and the 52 API checks qualify 58 distinct cases across
the reported runs; no second full combined run is claimed.
