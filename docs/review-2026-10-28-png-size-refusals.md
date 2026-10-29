# Review and plan: unreadable PNG dimensions

**Status:** `UI#27` implemented and locally CPU-qualified, 2026-10-28.

## Evidence and selection

This is the third ranked independent candidate in the
[PNG decoder review](review-2026-10-28-png-json-refusals.md). Native PNG settings
and A1111 parameters pass digit-only sizes directly to int before the ordinary
size-validation fallback. Python may refuse excessive decimal input with
ValueError; a 4301-digit side fits comfortably in the existing 1 MiB body cap
but currently escapes the route's normal response path. Four actual HTTP
baseline cases (either side and either metadata format) failed in 0.48 seconds.
The configured/default decimal conversion limit here is 4300; no interpreter
setting was changed. No browser or real gallery was used.

## Bounded behavior and acceptance

Treat a refused integer conversion like an unreadable size. Keep usable prompt,
negative text and sampling; retain the panel's size and say that the PNG's size
could not be read. Distinguish a present unreadable field from an absent field,
whose existing behavior stays unchanged. A Refine record keeps its existing
source-pass explanation instead of an extra direct-size note. Do not invent
substitute dimensions, impose a new numeric cap, change global digit limits,
map foreign samplers or alter the ordinary GenerateRequest budget path.

Eighteen HTTP cases cover either format/side, malformed and absent/null sizes,
Unicode decimal digits, ordinary leading zeros and zeros exactly at the current
conversion limit, plus Refine behavior. Real oversized conversion cases skip
only when no limit is configured or when an unusually high limit would exceed
the bounded fixture; report actual skips rather than calling them acceptance.
Retain existing PNG API/decoder cases and documentation maps, no-cache Ruff and
the installed pinned type gate. Source review found no blocking issue before
application. This is backend response qualification only; pending GAL12 browser
work, stopped reads/searches and setup API/UI proposals stay untouched.

## Implemented result

Only the HTTP PNG reader changed: failed decimal conversion returns unreadable,
and its two metadata formats now explain a named unreadable size honestly.
Missing-size behavior, ordinary numeric/budget validation and Refine source
semantics remain as specified. Independent source/test review was clear.

All eighteen new cases ran without skips. New size and decoder regressions,
existing PNG API checks and documentation maps passed **54 tests in 3.16
seconds**. No-cache Ruff passed across the repository. Installed pinned Pyright
1.1.407 analyzed **146 files with zero diagnostics**, in 11.257 seconds. All
four baseline failures now preserve usable settings and explain the size.
No interpreter limit, browser, model, real gallery or blocked operation changed.
