# Review: scientific scalar search terms

**Status:** `GAL#16` implemented and locally qualified, 2026-10-28 UTC.
Scientific float terms emitted by Sampling facts now remain exact scalar queries.
Independent source review and CPU/API checks are complete; web source is unchanged.

## Concrete problem and choice

A valid record can contain tiny guidance or Mu: `GenerateRequest` allows CFG
from 0 through 20 and Mu from 0 through 10, without a minimum precision.
Explicit values survive profile defaults and supported-field normalization.
The actual `factSearchTerms` helper in `web/gallery-search.js` writes those
recorded values directly into Sampling-row search terms. For a `1e-7` value,
it emits `cfg:1e-7` or `mu:1e-7`.

The previous numeric parser considers a hyphen range before trying the whole
scalar. It splits `1e-7` into a supposed low endpoint `1e` and high endpoint
`7`, then refuses `1e` as a number. The gallery route turns that named error
into HTTP422. Equivalent decimal exact terms and scientific comparator terms
already work. This breaks a query produced by the app's own facts button.
The original-source control below confirms the storage/HTTP failure and the
actual facts-generated query; browser interaction is outside this CPU slice.

Ranked repair choices:

1. Try the whole value as a float scalar before the existing range branch.
   This repairs the shared parser for facts, typed search and its other
   callers, with one bounded insertion and CPU-only qualification.
2. Expand scientific notation in the UI. This repairs only one caller and
   adds formatting/UI policy while leaving manual queries inconsistent.
3. Replace the range grammar. That introduces signed endpoint and error-policy
   decisions beyond the scalar defect. Neither 2 nor 3 is selected.

## Bounded implementation

Keep string, NULL and comparison handling in its existing order. Only when
the selected filter cast is `float`, try `_cast` on the entire raw value.
Success produces parameterized equality. ValueError falls through to the
unchanged first-hyphen range parser and its final scalar/error path.

Every integer path and SQLite bound stays unchanged. Existing float conversion
policy, including its nonfinite conversions, is preserved; this change does
not validate source fields or define SQLite matching for NaN. NULL remains
different from zero. Comparisons and valid existing range endpoints retain
their parameters. In particular `cfg:0-1e-7` already works and keeps working.
Unsupported forms such as `cfg:1e-7-1e-6` still fail on the first `1e` operand;
`cfg:-1-2` still fails on its whole value. No new signed-range grammar,
endpoint ordering, rounding, finiteness rule or frontend change is added.

## Qualification scope

The 44 new cases cover actual scalar/comparator/range/error/NULL/integer
contracts, including exponent signs, case, negative zero exponents, existing
negative scalar forms and parser-only nonfinite conversions. Valid normalized
requests become actual owned PNG/SQLite rows; scientific queries must select
the same tiny row as decimal/comparator controls while excluding zero/neighbors.

TestClient gallery reads and strict GET previews use the same predicate. Model
loading, generation and queue entry points are poisoned, and read snapshots
require unchanged scratch rows. The Node VM loads actual declaration-only
`gallery-search.js`, runs `factSearchTerms` on the serialized record and feeds
its actual Sampling terms back into storage. There is no DOM, app boot or
browser. Source fields/pixels are not qualified by the parser-only controls.

This verifies backend search and existing facts serialization. It does not
qualify a browser trial, model, card, live gallery, credential or denied artifact.
Pending GAL12/UI28 work and unknown browser outcomes remain separate. Shared
usage/design/rules edits are staged only as owned hunks.

## Original-source negative control (2026-10-28 UTC)

The actual unchanged repository source produces 22 intended scientific-scalar
failures and 22 preserved-grammar/control passes in 1.03 seconds. Equivalent
decimal and comparator gallery queries return 200 and select the tiny row;
the scientific exact query returns HTTP422 naming the false `1e` operand.
Actual Sampling terms from the declaration-only Node helper also fail when
fed into the original storage parser. All model/queue guards stay clear and
only owned scratch rows/PNGs are used. No browser or live gallery is involved.

## Positive results (2026-10-28 UTC)

All 309 targeted checks pass in 18.54 seconds, with no skips: new numeric
scalars, orientation, existing storage/gallery and API queries, saved searches
and their API, local dates/date ranges and docs maps. Actual facts-generated
scientific terms now select their scratch record; scientific gallery/strict
reads return 200 with exact counts. Model/queue guards remain untouched.
No-cache Ruff passes; pinned offline Pyright checks 146 files with zero
diagnostics. Independent review is clear at source `0c7a3d9e`, tests `8a0b70d2`
and the original owner packet `e28386f2`.

The original-source negative control preceded the four exact source/doc anchors.
No frontend edit, browser run, real gallery mutation or activation occurred,
and no stopped artifact or other denied operation was retried.
