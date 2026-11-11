# Review and plan: exact local calendar-day search bounds

**Status:** `GAL#13` implemented and locally CPU-qualified, 2026-10-28.

## Evidence and selection

Independent search review ranked (1) wrong local-day ranges on clock-change
days, (2) unrepresentable date bounds escaping validation, and (3) nonfinite
numeric filters. Choose the first and its adjacent boundary refusal; the third
has no documented finite-filter policy and is not commissioned by this review.
The current contract explicitly says `date:` is the local calendar day, shared
by gallery browsing and strict scopes for deletion/statistics/clear.

Previously date_clause converted the first midnight to UTC and added 24 hours.
That includes the first hour of tomorrow on a 23-hour day and omits today's last
hour on a 25-hour day. Existing local-day tests use ordinary September dates and
the host timezone, so they do not establish these transition boundaries.
Creating the next day or converting an extreme midnight can also overflow,
escaping the routes' ValueError-to-422 contract.

A deterministic baseline of seventeen tests produced **7 failures and 10 passes
in 0.32 seconds**: short/long UTC bounds and SQLite membership failed, both
unrepresentable cases escaped, and the actual gallery route escaped its named
validation response. All clocks are synthetic date-specific midnight offsets;
SQLite is in memory and the HTTP fixture owns its scratch gallery. No OS timezone,
environment, clock, service, real gallery or credential state was changed.

## Bounded repair and acceptance

Construct both local calendar midnights before independently converting each to
UTC. Retain the same half-open SQL predicate, UTC ISO spelling and parameter
order, grammar, `today`/`yesterday` selection and strict-scope rules. Translate
only bound-construction/conversion OverflowError to a named date ValueError;
do not impose an arbitrary date cap or widen a failed scope. Other conversion
failure types and the host's timezone implementation remain outside this claim.

Seven frozen boundary pairs and seven real SQLite strict-scope selections cover
24/23/25-hour days, positive and half-hour offsets, year rollover and entering/
leaving leap day. Boundary rows include the last fall-back half-hour and the
first half-hour after spring-forward's next midnight. Two extreme cases and an
actual HTTP 422 probe check named refusals. The datetime subclass supplies only
midnight offsets and uses real aware-datetime UTC arithmetic. No ZoneInfo/tzdata
installation, OS timezone override or external source is needed.

Independent source/test review found no blocking issue. Run targeted date,
storage, gallery API/strict-scope and documentation checks, no-cache Ruff and the
installed pinned type gate. This proves the calendar algorithm with supplied
offsets, not every Windows timezone or actual user-data cleanup. Preserve all
pending GAL12 source and its unknown run/artifact record, stopped equivalent
searches/reads, setup API/UI proposals and user/card/decision gates.

## Implemented result

Both local calendar midnights now determine the UTC bounds independently, and
unrepresentable bounds produce the named date validation error. All seventeen
new cases passed alongside existing storage, gallery API, strict-scope and
documentation checks: **125 tests in 11.53 seconds**. The seven baseline failures
were corrected, including actual HTTP 422 for the terminal date. No-cache Ruff
passed across the repository. Installed pinned Pyright 1.1.407 analyzed **146
files with zero diagnostics**, in 12.006 seconds. Independent source/test review
was clear. No OS timezone, clock, real gallery, provider, browser, model or
previously blocked action was changed or exercised.
