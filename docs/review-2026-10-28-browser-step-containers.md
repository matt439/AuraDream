# Review and plan: browser step containers

**Status:** `DEV#16` implemented and locally qualified, 2026-10-28.
Malformed step containers now refuse before browser discovery. Source/test review
and cold regression checks are complete; no browser run or artifact read occurred.

## Evidence and ranking

1. Reject malformed step entries before browser discovery. The documented
   format is a JSON list of step objects. Current number, boolean, string and
   nested-array entries survive the viewport/fatal checks, screenshot-name
   collection and the later action chain. Their action properties are absent,
   so the runtime silently skips them. Source traversal shows that an otherwise
   empty malformed fixture can reach ordinary zero-failure cleanup; a mixed
   fixture can run earlier actions while silently losing malformed steps. This
   is a source-derived consequence, not a measured browser run or exit status.
2. Normalize basic top-level shape diagnostics. Current valid JSON null,
   scalar or object documents pass the guarded viewport/fatal scans, then
   reach browser discovery. With a discoverable browser, `screenshotRoot`
   throws on `flatMap`; its existing catch reports exit 2 before output
   creation. Without a browser, the diagnostic instead says no browser was
   found. Null entries similarly fail in screenshot-name collection only
   after discovery. A shared early shape boundary resolves this alongside 1.
3. Normalize source-read and JSON-syntax failures. These currently terminate
   as unhandled module errors before discovery. A catch/diagnostic policy
   would be a separate concern, with file-path/error-detail choices. It is not
   included merely because the new shape check is nearby.

The first two share one small post-parse guard. They improve trust in fixture
coverage and produce a consistent refusal without changing browser controls.

## Bounded repair

Immediately after the existing `JSON.parse`, require an array. Scan every
entry for a non-null, non-array object, report every invalid step in input
order, and exit 2 before browser lookup, output preparation, profile scans,
spawn or network. Fixed diagnostics name the step position without echoing
its content. Existing native-API and offset refusals retain their earlier
ordering; malformed JSON and file-read exceptions retain their existing path.

Preserve empty arrays, empty objects, unknown keys, named metadata and existing
multi-action object precedence. This is a container check, not a complete
action schema, minimum-expectation policy or typo detector. Valid objects
continue to the existing viewport/fatal/keycombo/screenshot/action rules.
`browser-output.mjs` stays unchanged; it is an internal helper and the harness
owns validation of its input document.

## Qualification scope

The 39 new cases load the actual harness and screenshot-helper module bodies
after capturing them and poisoning effects. Only their dependency specifier is
substituted. Exactly one fixture read is permitted. Invalid documents and entries
must receive fixed exit-2 diagnostics before browser lookup, output, profile,
process or network ports; compatible object forms reach a poisoned first lookup.
Eight separate probes verify the effect guards themselves remain active.

The inventory case passes each current repository fixture independently through
the actual whole-document cold preflight. Each must reach the lookup boundary;
no fixture action runs and no browser acceptance is inferred. Empty fixtures remain permitted;
unknown actions and existing multi-action precedence retain their existing policy.
The new shape guard does not catch JSON syntax or source-read errors.

These checks qualify only preflight refusal/compatibility. Browser controls,
runtime dispatch, output helpers, cleanup and launch flags are unchanged. No
browser or screenshot inspection is required for this early shape boundary,
and no result is inferred for pending UI28 or GAL12 browser work.

## Original-source baseline (2026-10-28)

The actual repository harness before this insertion produced 21 intended
malformed-shape failures and 18 compatible/inventory/poison-control passes in
2.16 seconds. All malformed cases reached the first poisoned browser-discovery
boundary instead of the required fixed refusal. Earlier screenshot/eval steps
remained unexecuted. No browser, output, profile or network effect occurred.

## Positive results (2026-10-28)

All 275 focused checks pass in 12.24 seconds, with no skips: new container cases,
port offsets, capability/fatal metadata, viewport, screenshot-output, runner and
docs maps. No-cache Ruff passes; pinned offline Pyright checks 146 files with
zero diagnostics. Independent source/test review is clear; the final source
comment was clarified to distinguish null's prior exception from silently
skipped non-null primitives. Final source is `22a1d911`, tests `5828b897`.

The original-source negative control preceded the one-insertion source change.
All new cases run through instrumented real module bodies with actual browser,
profile, output and network effects forbidden. Existing artifact denials and
unknown browser outcomes were preserved; this work did not retry those actions.

## Per-file inventory scope during DEV17 integration (2026-10-28)

DEV17 moves existing async-helper checks before discovery. Its first positive
run passed 343 cases and failed the old concatenated inventory: helper names
`__small` and `__landed` from separate fixtures contaminated one synthetic global
scope. The inventory now sends each original JSON document independently to
the real instrumented harness in its own scratch subdirectory. No validator is
disabled and values are passed unchanged. The initial DEV16 combined result
above remains historical. The revised per-file inventory plus six docs-map
checks pass together in 5.47 seconds; no fixture action or browser is run.
