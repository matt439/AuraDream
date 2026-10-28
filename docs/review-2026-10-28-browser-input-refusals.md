# Review and plan: browser input refusals

**Status:** `DEV#19` implemented and locally qualified, 2026-10-28 UTC.
Fixture read failures and JSON syntax errors receive fixed exit-2 refusals.
Independent review and cold checks are complete; no browser or denied-path
operation was performed for this slice.

## Evidence and ranked choice

1. The harness documents exit 2 for a script refused before browser startup,
   but `readFileSync(file, 'utf8')` and `JSON.parse(text)` are unhandled at
   the input boundary. A missing/unreadable fixture or malformed JSON therefore
   becomes an unhandled Node error with exit 1, potentially accompanied by a
   stack, path or fixture excerpt. It also becomes indistinguishable from a
   failed assertion in the runner's top-level result: `run_smoke.py` preserves
   child statuses and returns 2 if any child returned 2, otherwise 1 for
   failures. The original-source control below confirms this status mismatch before browser preparation.
2. More general CDP/runtime error presentation could improve failed-smoke
   diagnosis, but spans live transport, step errors and owned browser cleanup.
   Its impact and qualification are broader than this cold input boundary.
3. Numeric query/search additions can add product capability, but require
   their own filter semantics and query qualification. They
   do not resolve this already concrete refusal-contract mismatch.

Select the first: a bounded correction with an objective existing contract,
useful diagnostics and meaningful offline qualification. It is independent of
all denied paths and unknown browser outcomes. A simulated permission error
does not authorize retrying a real denied operation.

## Bounded change and preserved order

Catch only the fixture `readFileSync` operation at its existing position. Print
`could not read browser steps file; check its path and read access` and exit 2
on a read failure. Do not echo the raw path, fixture body, exception message or
OS diagnostic. No new read, fallback path or automatic retry is introduced.

Catch only `SyntaxError` around the existing `JSON.parse` call at its current
position. Print `browser steps file must contain valid JSON` and exit 2.
Unexpected non-syntax parser faults still propagate; a broad parser catch
could hide a programming or resource fault as bad JSON, so this repair does
not do that.

Capability/missing-argument/offset checks remain before the single fixture read.
Effective-port substitution/refusal stays after that read and before JSON
parsing. Existing parsed shape/action/startup checks, browser discovery/output
preparation and valid runtime flow keep their order and predicates. No BOM
stripping, JSON relaxation, schema, numerical policy, host matching, profile,
launch or cleanup behavior changes. A BOM-prefixed document remains invalid
JSON; only its refusal diagnostic/status changes.

## Qualification scope

The 40 new cases execute captured actual harness/helper module bodies with only
the dependency specifier changed. An owned read port and JSON parsing are
counted; all discovery/output/profile/process/network effects are poisoned.
Five named read errors are injected at that permitted port, and twelve actual
malformed documents exercise native JSON syntax failures. Neither paths nor
fixture bodies are printed. EACCES/EPERM cases are simulations, not filesystem
permission probes, access restorations or authorization for denied operations.

Five valid documents reach the first poisoned lookup; three unexpected parser
faults still propagate. Seven order controls cover native APIs, arguments,
offset/port, parsed shape, viewport and fatal metadata. Eight direct probes
verify the effect guards, including the constructible WebSocket poison.

The test-only outer import relay prints only the propagated error name and
retains exit 1. It cannot turn deliberate application exit 2 or poisoned-effect
exit 99 into a pass. This permits actual-module negative controls without a
large data-URL stack or fixture excerpt. Cold results do not qualify browser
layout, cleanup, hardware or pending UI28/GAL12 work.

## Original-source negative control (2026-10-28 UTC)

The actual unchanged harness produces 17 intended failures and 23 preserved-
behavior/active-poison passes in 2.06 seconds. Five simulated read errors and
twelve actual JSON syntax failures propagate as exit 1 through the test-only
name relay instead of the required fixed exit 2. Valid documents, unexpected
parser faults and existing guard ordering already pass. The read errors are
injected at an owned permitted port; no denied path or real permission failure
was probed. Browser/output/profile/process/network effects stay forbidden.

## Positive results (2026-10-28 UTC)

All 428 focused cold checks pass in 24.29 seconds, with no skips: new input
refusals, startup geometry, preparation order, per-file fixture inventory and
containers, port offsets, capability/fatal metadata, step viewports, screenshot-
output, runner and docs maps. No-cache Ruff passes; pinned offline Pyright
checks 146 files with zero diagnostics. Independent review is clear at source
`1fbf803c`, tests `c24512eb` and the original owner packet `f634ac8f`.

The actual original-source control preceded the two exact input anchors.
Valid flow, launch/profile/cleanup and existing guard order are preserved.
No actual browser, network, output/profile operation or real denied-path read
ran, and no earlier denied action or unknown browser run was retried.
