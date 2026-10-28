# Review: More seeds headline across seed wrap

**Status:** `GEN#30` implemented and locally qualified, 2026-10-28 UTC.
Whole-run More seeds headlines report every allocated seed's numeric extrema.
Independent review and targeted CPU/API qualification are complete.

## Evidence and choice

Batch, picked/filter and whole-run More seeds are reported Often (`G4`, `G5`,
`G6` in [reported use](design/preferences.md)). The numerical edge is uncommon,
but an existing server sentence must accurately describe the work planned.

`more_seeds.run_headline` describes the lowest/highest new seed span. It currently
takes each batch's first and last seeds as extrema. The existing allocation
contract wraps: after MAX_SEED-1, three new seeds are `[MAX_SEED, 0, 1]`.
Their endpoints are not the numeric minimum/maximum. A single wrapped batch
therefore produces a descending headline; combined with an ordinary batch,
the headline can omit the wrapped extremes entirely.

The original-source controls below reproduce this defect in the pure helper
and actual owned-record API preview. Existing `next_seeds` tests already
describe the valid wrapped allocation.

Ranked repair choices:

1. Compute numeric extrema from every actual allocated seed, selected. It
   fulfills the existing sentence's contract through a two-line computation
   change and retains ordinary wording.
2. Add wrap annotations or redesign individual call labels. This broadens
   formatting policy and is unnecessary to repair the aggregate range.

This is a display defect, not a seed-allocation or pixel error. The repair does
not change the count cap or admission policy.

## Bounded repair

Use per-batch nested min/max and then aggregate across the supplied batches.
The input contract remains queued batches with nonempty allocated seed lists.
There is no new eligibility filter, fallback or empty-batch policy. In
particular, an invalid empty batch is not silently discarded by flattening.

Preserve allocation order and wrap modulus, per-batch counts, call labels,
result `first`/`last`/`skipped`, statuses, Batch fallback, lineage grouping and
preview-issued receipt behavior. The numeric headline is a bounding span;
it does not assert that every intermediate number is queued. Existing ordinary,
single-seed, plural and Starred-ignored wording remains unchanged.

## Qualification scope

Six pure full-headline cases cover wrapped/mixed batches, ordinary gaps,
singletons and repeated values in different cells, with unchanged allocated
seed lists. An empty queued list retains its existing absent headline.

Three owned scratch Python TestClient previews cover the real planner's wrap,
ordinary unrecorded-Batch fallback with a foreign batch excluded, and an all-
foreign selection with no headline/intent. They assert unchanged first/last/
skipped fields, counts and fallback behavior, snapshot all rows/PNG hashes and
poison loading, generation and queue admission. Fixture PNGs are owned setup;
these preview cases enqueue no jobs or generate new model images. Python JSON
preserves MAX_SEED here; no browser entry or JavaScript handling of 19-digit
numbers is qualified. Existing API/receipt regressions retain their own owned
fake-job setup and submission coverage.

No web source changes, so no new browser or screenshot run is needed for
this server-string repair. Pending GAL12/UI28 work, unrelated proposals and
all denied or unknown operations remain separate. Shared documents are staged
only as exact owned hunks.

## Original-source negative control (2026-10-28 UTC)

The frozen ten-case module on unchanged source produces four intended full-
headline failures and six ordinary/no-eligible controls passing in 0.62 seconds.
Three pure wrap/mixed-batch ranges are descending or omit the true extrema.
The owned Python API preview returns 200 and correctly reports first MAX_SEED,
last 1, skipped zero and the expected counts, then fails on its descending
headline. Its rows/PNGs and model/queue guards remain controlled. One installed
Starlette TestClient/httpx deprecation warning is reported. No browser seed
entry, generated image or real gallery is involved.

## Positive results (2026-10-28 UTC)

All 59 targeted checks pass in 5.91 seconds, with no skips: the ten new
headline checks, existing More seeds planning/API tests, both intent/receipt
modules and docs maps. The new wrapped API preview keeps first MAX_SEED, last 1,
skipped zero and the existing counts while its headline reports 0 through
MAX_SEED. New preview row/PNG snapshots and model/queue guards pass; ordinary
wording, foreign exclusion, unrecorded Batch fallback and no-eligible behavior
remain intact. One installed Starlette TestClient/httpx deprecation warning
is reported.

No-cache Ruff and the pinned offline Pyright gate pass. Independent review
is clear at source `b03c325d`, tests `b775ca5d` and original owner `f1c9a5c3`.
Only nested headline extrema and their comment change in application code;
allocation, call labels, result ordering/fields, status filtering and receipt
paths are untouched. No real gallery, browser, model/card or denied operation
was used to qualify this repair.
