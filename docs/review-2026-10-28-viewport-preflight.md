# Review and plan: viewport preflight

**Status:** `DEV#14` implemented and locally qualified, 2026-10-28.
Reviewed harness source `e504787d` passed the recorded cold boundary and
existing tooling checks. Browser resize and cleanup mechanisms are unchanged.

## Choice and evidence

Ranked independent source work:

1. Refuse malformed viewport steps before browser discovery or side effects.
   A newly owned watched-controls run passed 20 assertions, then destructuring
   an object viewport threw at browser.mjs's resize branch. The run had already
   launched a browser and performed earlier steps. This is direct evidence of
   a predictable fixture error being discovered too late. Scope and risk are
   small: one input shape, fixed diagnostics and cold effects-boundary checks.
2. Generalize all action validation or rewrite browser cleanup. Those may have
   value, but would expand policy or process-lifecycle behavior without the
   same bounded evidence. Leave them separate.

This takes the first and does not repeat the interrupted run, inspect old
artifacts or alter browser flags, selection, security settings or cleanup.

## Bounded source change

After native capability checks and the one JSON fixture read, validate every
declared viewport value before browser discovery, screenshot roots, profile
sweeping, network, CDP or process creation. Require the documented array of
two or three numbers, with integer dimensions in CDP's inclusive 0..10000000
range and optional finite nonnegative DPR. Explicit null, strings, booleans,
object viewports, wrong lengths, fractional/out-of-range dimensions, negative
DPR and JSON numeric overflow refuse
with exit 2 and a fixed step-indexed sentence that never echoes the value.

The existing runtime destructuring stays unchanged, including omitted DPR's
default of 1, fractional DPR and zero reset values. The protocol's dimension
cap is enforced; DPR has no invented upper bound beyond finiteness.
Environment viewport/DPR parsing retains its existing positive startup guard.
Unrelated actions, mixed-action
policy and malformed container policy are outside this slice.

Browser discovery moves after viewport and fatal metadata preflight. Thus valid step input
is read before a missing-browser refusal; malformed viewport input receives its
specific fixed diagnostic even on a host without a browser. The existing
missing-browser message is unchanged. Native capability refusal still precedes
the fixture read, and supported-API cold checks explicitly permit that one read.

The integration owner checked the official
[DevTools Protocol Emulation definition](https://github.com/ChromeDevTools/devtools-protocol/blob/master/pdl/domains/Emulation.pdl)
on 2026-10-28. `setDeviceMetricsOverride` defines integer width/height from zero
through 10000000 and says zero disables the override; `deviceScaleFactor` also
uses zero to disable its override. Retaining those reset values is a compatibility
requirement, rather than narrowing existing valid steps to positive sizes.

## Proposed qualification

The proposed test module executes the actual harness and screenshot helper as
data URLs after source capture, following the existing capability tests. It
permits exactly one fixture read and poisons browser lookup, further source
reads, output writes/mkdirs, profile scanning, process creation, fetch and
WebSocket. Direct probes prove each poisoned boundary is live. Invalid input
must exit 2 before any probe; valid arrays stop at the poisoned browser lookup.
An earlier screenshot step tests that refusal happens before artifact setup.

Cases include object-format reproduction, all wrong raw types, both dimensions,
DPR, wrong lengths, JSON overflow to Infinity, input-order diagnostics, default
and fractional DPR, zero resets, maximum dimensions and one past the maximum,
and ordinary steps without a viewport. These prove cold
preflight behavior, not actual resizing or browser cleanup.

Before integration, run failing negative cases against the original harness;
then apply source and run the focused viewport/capability/output tests, smoke
table contracts, no-cache Ruff and pinned type gate. Existing qualified browser
resize behavior is preserved by leaving the runtime branch unchanged; a fresh
browser run is not needed solely to prove this cold refusal boundary. Do not
read denied artifacts, inspect unknown processes or change trust settings.

## Actual qualification (2026-10-28)

The original-source baseline failed all 31 malformed-input refusal cases in
2.76 seconds: each reached the deliberately poisoned browser lookup and exited
99 instead of returning the fixed exit-2 diagnostic. The 12 valid-input cases
and eight direct poison probes passed. Within the instrumented harness, real
browser lookup, output/profile writes, child spawning and network calls stayed
blocked; the poison stopped each attempt. This is a meaningful negative control for the actual input boundary,
alongside the earlier watched-fixture runtime failure that motivated the work.

After applying the exact independently reviewed source and narrow capability
fixture adjustment, 187 focused tests passed in 7.63 seconds with no skips:
viewport preflight, native capabilities/fatal metadata, screenshot-output
boundaries, smoke-runner contracts and docs maps. The 51 new cases exercise
the actual module bodies. Invalid values receive fixed input-order diagnostics
without their contents; default/fractional/zero DPR, each zero dimension and
the inclusive maximum reach the same poisoned browser lookup. One beyond the
maximum is refused separately for each dimension. Maximum-size values are
never sent to an actual browser by these cold checks.

No-cache Ruff passed. Pinned Pyright checked 146 backend/tools files with zero
errors or warnings. Independent review confirmed both the effects boundary and
the corrected zero-reset compatibility. The runtime resize branch, browser
candidates/flags, startup environment guard, profiles and cleanup are unchanged.
No new browser run was needed for this cold input slice; the prior GEN29 browser
qualification remains evidence for its app/control behavior only. No hardware,
model, credential, real gallery, WSL, compiler-wheel or old artifact access was
performed. The six pending GAL12 files and their unknown result remain separate.
