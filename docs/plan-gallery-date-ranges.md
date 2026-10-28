# Plan: explicit gallery date ranges

**Status:** `GAL#14` implemented and locally CPU/API-qualified, 2026-10-28.
Independent source/test review was clear. Six range-selection baseline cases
failed in 0.22 seconds before the additive parser branch was applied.

## Why this slice

The autonomous shortlist ranked (1) explicit multi-day gallery filters and
(2) structured keeper-rate CLI export. Current `search.date_clause` accepts
only one local day; separate `date:` terms AND together and cannot express a
multi-night interval. Current `stats.py` prints a human table with truncated
values and rounded percentages, but [reported use](design/preferences.md)
rates keeper-rate inspection Never (L3), while date/run/id searching is
Sometimes (E6), browsing is Often (E1), and scoped Delete unstarred is Often
(J6). The date range is a bounded convenience hypothesis, not an observed
task saving or a newly reported user requirement.

As a comparable capability, the [official Immich search guide](https://docs.immich.app/features/searching/)
lists a timeframe filter with start/end dates (checked 2026-10-28). This is
precedent for selecting a time bucket, not syntax compatibility or evidence
of AuraDream demand, speed or quality. No further comparison or remote API
access is necessary for this slice.

## Contract

`date:2026-10-26..2026-10-28` selects October 26, 27 and 28 in the server's
local timezone. Both endpoints are canonical absolute `YYYY-MM-DD` dates,
with the first no later than the last. A one-day range equals the existing
single-day query. Open ends, repeated/malformed separators, reverse order,
relative or comparator-marked endpoints are refused with a named range error.

The indexed SQL stays `created_at >= ? AND created_at < ?`: convert the first
local midnight and the local midnight after the final date independently to
UTC, preserving [the calendar-boundary repair](review-2026-10-28-local-date-bounds.md).
A 3-day interval crossing a clock change can span 71 or 73 hours. A final
date without a representable following midnight retains the named overflow
refusal. No arbitrary maximum duration, global clock or timezone setting is
introduced.

Existing single-day/`today`/`yesterday` behavior is unchanged. Separate terms
still intersect, known keys/free text retain their current parsing, and
free text stays invalid for a destructive strict scope. The explicit range
shares `search.date_clause` across the gallery, bin, statistics and strict
scope readers; counts, confirmations, stars, source retention and maintenance
admission are unchanged.

Saved searches store the raw query text and ticks without rewriting dates,
using their existing validation and version-1 sidecar. The frontend treats
date values opaquely, so no web module, control, layout or browser interaction
changes. No new browser acceptance is claimed.

## CPU and API qualification

`tests/test_search_date_ranges.py` supplies synthetic local-midnight offsets
without changing the OS timezone, environment, wall clock or tzdata. Actual
SQLite :memory: queries test inclusive endpoints, middle rows, exact cutoff,
ordinary and same-day spans, 71/73-hour clock-change spans, year/leap rollover
and intersections. Existing single-date spellings remain equivalent; malformed,
relative, reversed and unrepresentable ranges refuse.

Owned scratch gallery/API cases exercise live and bin filtering and GET-only
unstarred previews, preserving row states and using existing star counts.
One separate fresh-store TestClient integration checks scoped binning only:
an owned starred Refine pass keeps its source, a stale expected count returns
409 without changes, and the correct count bins only the loose in-range row
while every PNG and all outside/prior-bin rows stay present. This is an owned
unit fixture, not real-server cleanup or permission to retry any old run.
Saved-search CRUD checks exact whitespace/query/ticks and the existing fixed
redacted 422 for invalid ranges without changing the previous sidecar. Model
load/generate, queue submission and gate entry points are poisoned and jobs
remain empty. No real gallery, credential, GPU, network or browser run is
performed by these tests; only the explicitly owned binning fixture uses DELETE.

The new 33 range cases plus existing date, storage, gallery API/strict-scope,
saved-search and documentation checks passed **219 tests in 16.73 seconds**.
No-cache Ruff passed across the repository. Installed pinned Pyright 1.1.407
analyzed **146 files with zero diagnostics**, in 10.924 seconds. The six baseline
failures now select the specified calendar intervals. The owned binning fixture
verified a mismatched count leaves every row unchanged, then bins only the
one eligible row while stars, retained sources, outside dates and PNGs remain.

No web source changed and no new browser, model, OS credential, timezone or
real-gallery operation was performed. GAL12's dirty repair/unknown run, denied
artifact access, stopped equivalent searches and unapplied UI25 proposals remain
preserved. Only this feature's documentation hunks are qualified for staging.
