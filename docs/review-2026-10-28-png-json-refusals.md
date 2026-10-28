# Review and plan: named PNG JSON refusals

**Status:** `UI#26` implemented and locally CPU-qualified, 2026-10-28.

## Evidence and selection

After the current source backlog reached independent acceptance/access gates,
ranked source-only candidates were: (1) malformed PNG JSON escaping the existing
named refusal contract, (2) smoke runner port plans reaching socket startup
without a range check, and (3) unreadable oversized PNG dimensions escaping the
ordinary size fallback. The first is a user-facing import defect with a two-site
repair and an actual HTTP/file regression; port planning is prepared separately.
No comparison-project claim or new product policy is needed for these defects.

The shared `png_settings.record` and HTTP-only workflow detector catch
`ValueError` from `json.loads`, but its excessive nesting raises `RecursionError`.
The nested document is a string inside the HTTP body, so it can pass the normal
1 MiB outer-body and schema caps. Six targeted baseline cases reproduced the
exception: current/legacy app chunks and the workflow/prompt detector. They
failed in 1.45 seconds; no browser, model or real gallery was used.

## Bounded repair and acceptance

Treat the decoder's `RecursionError` like its existing malformed-JSON refusal.
An unreadable higher-priority app/legacy record must still refuse by filename,
even when a usable lower-priority A1111 block exists. The workflow detector must
continue to the next keyword when the previous one cannot be decoded. Do not
introduce a new parser, fixed nesting policy, global recursion/digit settings,
metadata extraction, accepted-format mapping or fallback to different settings.

Exercise the actual shared reader, HTTP route and owned one-pixel PNG files;
retain existing PNG API and CLI refusal coverage, documentation maps, no-cache
Ruff and the pinned type gate. These CPU checks establish named backend refusals,
not browser layout, GPU behavior or acceptance of arbitrary malformed PNGs.
Preserve the six pending GAL12 files, its unknown run/artifact boundary, the
stopped equivalent searches and the unapplied setup API/UI proposals.

## Implemented result

Only the two existing decoder catches changed. Current and both legacy app
metadata keywords still take precedence over an A1111 fallback. An unreadable
workflow keyword no longer hides a later recognizable prompt graph. The shared
file reader and the HTTP route retain their existing named refusal wording.

Independent source/test review found no blocking issue. The twelve new real
JSON/HTTP/owned-PNG cases plus existing PNG API and stubbed CLI checks passed
**148 tests in 29.27 seconds**. Documentation maps passed **6 tests in 0.23
seconds**. No-cache Ruff passed across the repository. Installed pinned Pyright
1.1.407 analyzed **146 files with zero diagnostics**, in 10.83 seconds. The
baseline's six failures were corrected; no browser or GPU qualification is
claimed, and no previous blocked action was retried.
