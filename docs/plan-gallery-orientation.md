# Gallery orientation searches

**Status:** `GAL#15` implemented and locally qualified, 2026-10-28 UTC.
The existing size filter accepts portrait, landscape and square values across
resolutions. CPU/API checks and source review are complete; no web source or
real gallery was changed for this slice.

## Evidence and ranked choice

1. **Orientation across resolution tiers, selected.** Aspect presets and
   resolution tiers are reported Often (`A3`, `A4`); size/model/sampling
   searches Sometimes (`E7`). Prompt search, settings-to-search, search-as-sheet
   and filtered More seeds are Often (`E5`, `E9`, `F9`, `G5`). The existing
   exact `size:WxH` filter requires a specific resolution. Three orientation
   values group similarly shaped images across tiers without knowing
   every pair of dimensions. These preferences support the choice; they do
   not establish a measured time saving or an explicit request for this syntax.
2. **Exact scientific sampling numbers, a separate correctness candidate.**
   Source inspection shows `filter_clause` partitions a hyphen as a range
   before casting a bare float, so a value such as `cfg:1e-7` appears to be
   misread while `cfg:>=1e-7` takes the comparison cast directly. The facts
   helper interpolates numeric values into search terms. This candidate is
   unexecuted and outside this slice.
3. **Independent width/height comparisons.** They would offer flexible numeric
   filters, but new keys require changing the frontend's `SEARCH_KEYS` list
   and its qualification. Custom dimensions/orientation swapping are reported
   Never (`A5`), so this has weaker evidence than grouping existing presets.

[Reported use](design/preferences.md) is the local evidence. A comparable photo
library also exposes landscape, portrait and square filters in its current
[official PhotoPrism filter reference](https://docs.photoprism.app/user-guide/search/filters/),
checked 2026-10-28. AuraDream's proposal uses the existing `size:` key and
stored width/height comparisons. No PhotoPrism implementation or exact matching
semantics are copied or claimed.

## Search contract

- `size:portrait`: stored width is less than height.
- `size:landscape`: stored width is greater than height.
- `size:square`: stored width equals height.

The three values are case-insensitive and span resolutions. Use direct integer
comparisons, without aspect-ratio division, products or tolerance thresholds.
No new columns, migrations, derived metadata or model work are needed.

The exact size path stays intact, including `x`, `X`, the multiplication sign,
leading zeroes, supported digit spelling and SQLite integer bounds. Other
invalid values retain the current `WIDTHxHEIGHT` refusal. Portrait does not
introduce a portrait-subject classifier or a rotation operation; it describes
the recorded dimensions.

Terms retain their existing AND behavior: repeated terms intersect, and
`size:portrait size:512x768` narrows to that exact size. Prompt text remains FTS
text for gallery reads and is refused by strict scopes. Quotes and negation
gain no new meaning: `size:"portrait"` and `size:!portrait` remain invalid
values; `-size:portrait`, `"size:portrait"` and `orientation:portrait` remain
unknown-key free text. Saved searches keep the raw query and ticks.

## Existing frontend and shared scopes

`web/gallery-search.js` already recognizes `size`; `searchKey` validates only
the key and the existence of a raw value. Gallery requests pass the full query
through `URLSearchParams`. Saved-search editing and apply preserve the raw
query, and a sheet uses the same gallery request. No web source or new control
is required. The Node supplement loads the actual declaration-only
search helper and checks those pure merge/classification behaviors with no DOM.

The facts Size button continues to emit exact `WxH`; its existing merge replaces
all prior `size:` terms, including orientation. Prompt-based merging retains
size filters. In-app Search help continues to show its existing exact-size
example; the Usage guide documents these additional typed values.

`storage.list_images` supplies gallery/sheet data, current/all-model scoping,
favorites, pagination and counts. The bin uses the same grammar while retaining
its existing all-model/no-Starred contract. Strict scopes feed unstarred
preview/deletion, CLI clear and keeper-rate statistics; the new values must
agree in each. Filtered rerun producers resolve through `list_images`, so their
existing selection, ceilings and confirmation rules remain in force. Recent
runs intentionally accepts only limit and model scope; it is not extended with
a query parameter by this work.

## Qualification scope

The 46 new cases exercise actual SQLite predicates, including adjacent 64-bit
integer dimensions, resolution tiers, value case, AND composition, exact-size
intersections and preserved spelling/refusals. The declaration-only Node VM
supplement checks actual existing search-key/merge functions without a DOM or
app boot. No web code changes are required.

Owned scratch HTTP checks cover current/all-model/favorite scopes, other-model
counts, keyset paging, punctuation-triggered LIKE fallback, bin results and
strict previews. Read requests preserve all rows and PNG hashes. Model loading,
generation and queue entry points are poisoned. One owned-gallery DELETE checks
stale expected-count refusal, a source protected by an out-of-scope starred pass,
stars/other shapes/models/bin exclusions and PNG-preserving binning. Saved
queries retain raw text/ticks and existing redacted refusal without rewriting
the sidecar. Keeper rates share the predicate and exclude binned/other-model rows.

This is backend grammar and CPU/API qualification with existing page functions;
it does not claim a browser trial, performance measurement, GPU behavior or
qualification of pending UI28/GAL12 work. No live gallery, weights, credential,
service activation or denied artifact operation was needed.

## Original-source negative control (2026-10-28 UTC)

The actual unchanged repository source produced 19 intended feature failures
and 27 ordinary/control passes in 2.62 seconds. Orientation SQL/HTTP queries
refused with the existing size error/422, including strict previews and saved
search admission; exact dimensions, malformed-input controls and the existing
frontend key/merge functions already passed. Model/queue entry guards stayed
clear. Only owned scratch records/PNGs were used; no live gallery, browser,
GPU, credentials or denied artifact was accessed.

## Positive results (2026-10-28 UTC)

All 265 targeted checks pass in 20.94 seconds, with no skips: new orientation
cases, existing storage/gallery and API queries, saved searches and their API,
local dates/date ranges and docs maps. This includes the existing frontend-key
parity guard. No-cache Ruff passes; pinned offline Pyright checks 146 files with
zero diagnostics. Independent review is clear at source `6b70a6f6`, tests
`d018bf7e` and the original plan packet `7f5666c7`.

The actual original-source negative control preceded the seven exact source/doc
anchors. The existing exact-size path, including the multiplication sign, is
unchanged. Shared documentation is staged only through this feature's hunks;
all earlier pending Reuse/GAL12 source and unknown outcomes stay preserved.
