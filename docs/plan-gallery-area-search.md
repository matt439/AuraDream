# Gallery area comparisons

**Status:** `GAL#17` implemented and locally qualified, 2026-10-28 UTC.
Comparison-only actual MP filters use the existing size key. Independent
source review and targeted CPU/storage/API qualification are complete.
No frontend edit, browser trial, GPU or live gallery is involved.

## Evidence and ranked choice

1. **Comparison-only actual area, selected.** Resolution tiers are reported
   Often (`A4`) and size/model/sampling searches Sometimes (`E7`); prompt search,
   search-as-sheet and filtered More seeds are Often (`E5`, `F9`, `G5`). An
   actual-area comparison finds larger or smaller results across orientations
   without knowing every exact pair of dimensions. These preferences support
   a bounded addition; they establish neither a measured speed gain nor an
   explicit request for this syntax.
2. **Nominal preset-tier matching.** Tiers are approximate categories, and
   stored dimensions do not retain which tier or custom control produced them.
   Inferring a tier would add ambiguity, especially for Refine and Outpaint.
3. **Bare/equality/range MP syntax or unit aliases.** Two comparisons already
   express an interval. Extending these other forms adds grammar and exact-label
   questions with less evidence of value, so they remain outside this slice.

[Reported use](design/preferences.md) is the local evidence. The current
[official PhotoPrism reference](https://docs.photoprism.app/user-guide/search/filters/)
also provides an MP resolution filter, checked 2026-10-28. AuraDream uses its
own existing size key and measurement convention.

## Actual area and nominal labels

The existing [size calculation](../web/sizes.js) and [Size facts](../web/facts.js)
use width multiplied by height divided by **1,000,000**. Facts and size hints
round for display; guidance bounds have their own formatting. Preset tiers in
[sizes.py](../backend/sizes.py) are nominal labels on dimensions aligned to the
VAE grid. Guidance tolerances do not define stored image identity.

For example, 1024x1024 is exactly 1.048576 MP, while 1536x640 is 0.983040 MP;
both are nominal 1 MP presets. Thus `size:>1mp` includes the former, and
`size:>=1mp` excludes the latter. Matching uses the unrounded stored area.

## Search contract

Use `size:<Nmp`, `size:<=Nmp`, `size:>Nmp` or `size:>=Nmp`. Magnitudes are
finite and nonnegative, including zero and signed zero. The pixel threshold
may not exceed SQLite's integer binding maximum: 9223372036854775807 pixels,
or exactly 9223372036854.775807 MP. Negative/nonfinite/invalid numbers receive a
named size refusal; out-of-range thresholds receive the existing named range
refusal. This bound comes from the engine rather than a profile or nominal tier.

Numeric spelling follows Decimal's finite values, including ordinary scientific
notation; the `mp` suffix and the size key are case-insensitive. Bare
`size:1mp`, equality, MP range forms and other unit spellings remain unsupported.
`size:>=1mp size:<2mp` uses existing AND composition. Free text remains gallery
text and is refused by strict scopes. Exact WxH, multiplication-sign spelling,
orientation values and their existing refusals retain their paths.

The new area branch selects positive stored SQLite INTEGER dimensions. It uses
explicit type/positivity predicates, without coercing REAL, text, NULL or
nonpositive legacy values. This adds no row validation, rewrite or migration to
the existing exact-size/orientation branches.

## Exact parsing and arithmetic

Parse into Decimal with a fresh explicit private context. Constructors preserve
all input digits; malformed-input flags belong to that private context. Check
finiteness before comparisons, then sign and the exact MP maximum before any
pixel-to-integer conversion. Normalize zero first so an extreme zero exponent
needs no shifted representation. Shift a nonzero Decimal tuple exponent by six
to obtain pixels exactly, without multiplication, scaleb or denominator powers.

Let T be that exact pixel threshold. For `>` and `<=`, bind B = floor(T).
For `>=` and `<`, bind B = ceil(T) minus one. Convert the rounded Decimal to a
Python integer **before** subtracting one. No ambient Decimal operation or
context update is required. Rounding outputs fit the 19-digit engine bound;
long fractional inputs retain their boundary even at caller precision two.
Huge positive exponents refuse before integer allocation, and tiny exponents
can round directly to zero/one without constructing huge powers.

For positive integer dimensions, lower bounds use `width > B / height` and
upper bounds `width <= B / height`, with SQLite integer division. B stays in
-1 through INTMAX. At zero, B=-1 still gives the correct all/none outcomes
because dimensions are positive. Products larger than INTMAX never have to be
formed. SQLite's [operator reference](https://www.sqlite.org/lang_expr.html#operators)
documents integer division and floating conversion on arithmetic overflow.
Python's [Decimal reference](https://docs.python.org/3.13/library/decimal.html#decimal.Decimal)
documents exact construction and explicit-context integral rounding. Both
primary references were checked 2026-10-28.

Root separately reported a SQLite 3.53.1 in-memory identity probe: 27 stored
rows and 52 predicates matched a Python arbitrary-integer/Fraction oracle,
including engine limits, overflow-sized products and invalid stored types.
That probe exercises the arithmetic design, not this application's parser or
new source, and does not replace the required controls and qualification.

## Existing frontend and shared scopes

The size key is already recognized by existing search helpers, and requests,
saved definitions and sheets pass the raw query. The facts Size button retains
its exact WxH terms and existing same-key replacement behavior. No web key,
control, route or activation is added. The Usage guide documents the new values.

Gallery/model/favorite/bin filters and pagination retain their existing scopes.
Strict preview/deletion and keeper rates consume the same parsed predicate;
their existing expected-count, star/source retention and confirmation behavior
remain in force. Saved searches retain raw text/ticks and redacted invalid-write
refusals. This work reads no live user gallery and changes no stored image.

## Qualification scope

The 94 new cases execute actual SQLite predicates: all four comparisons,
inclusive/exclusive one-pixel edges, nominal/actual examples, zero/signed-zero,
tiny exponents, adjacent engine limits and overflow-sized products. Caller
precision two and 400, traps and pre-set flags surround 256-plus-digit values
above/below boundaries, including INTMAX. Success and malformed/sNaN/exponent
refusals preserve caller state. Nonpositive, NULL, REAL, text and blob geometry
does not match the MP branch; existing exact-size/orientation paths are retained.

Owned scratch API checks cover current/all-model/favorite scopes, other-model
counts, paging, bin results and strict GET previews with row/PNG-hash snapshots.
Saved queries preserve raw intervals/ticks and redact invalid writes without
rewriting their sidecar. Keeper rates share the predicate and retain exclusions.
Model loading, generation and queue entry points are poisoned. No DELETE is
needed for these new checks; existing strict-scope lifecycle tests also run.

This backend-only feature needs no browser run. It grants no acceptance to
pending UI28/GAL12, hardware or denied artifacts. Only owned feature hunks
are staged in shared documents; all prior pending work is preserved.

## Original-source negative control (2026-10-28 UTC)

The frozen 94-case module against unchanged repository source produced
56 intended feature failures and 38 existing-behavior/refusal controls passing
in 2.53 seconds. Exact actual-area values were rejected as WIDTHxHEIGHT by
the original parser, including the scratch gallery, bin, saved-query and
keeper-rate paths. No model or queue work occurred. One warning was reported;
its body was lost in the truncated failure output, so it is not classified.
Positive qualification will retain complete output. The independent frozen
source/test review is clear; source and test hashes remain recorded in the
workspace run journal.

## Positive results (2026-10-28 UTC)

All 403 targeted checks pass in 28.51 seconds, with no skips: the new 94 cases,
existing storage/gallery and API queries, saved searches and API, local dates,
date ranges, orientation, scientific scalar queries and docs maps. Scratch rows
and PNG hashes remain unchanged by new reads, and model/queue guards stay clear.
The suite reports one installed Starlette TestClient/httpx deprecation warning;
there is no application failure. The original control's truncated warning body
is not retroactively classified from this separate result.

No-cache Ruff passes. Pinned offline Pyright 1.1.407 checks 146 files with zero
errors or warnings in 11.599 seconds. Independent review is clear at source
`d4b59097`, tests `117eee43` and original plan packet `05f152b5`; root also
reviewed the final source diff. All nine anchors match the frozen snapshots.

No web edit, browser launch, live gallery mutation, model/card operation or
activation occurred. No stopped artifact or other denied operation was retried.
