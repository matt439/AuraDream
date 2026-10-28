# Review and plan: direct browser port preflight

**Status:** `DEV#15` implemented and locally qualified, 2026-10-28.
The independent direct-harness preflight and its offline regression checks are
complete. No browser launch or artifact access was performed for this slice.

## Evidence and choice

Ranked concrete independent tooling defects:

1. The documented direct `node tools/browser.mjs steps.json` route converts
   `AURADREAM_PORT_OFFSET` with `Number(raw || 0)` and uses `if (OFFSET)` to
   decide whether to rewrite local endpoints. A mistyped nonnumeric value
   becomes falsy `NaN`, so the fixture silently keeps its default endpoints.
   A developer intending an isolated fake can consequently address another
   already running default-port server. Fractional/nonfinite offsets and
   out-of-range effective ports are instead written into the fixture and fail
   only after later setup. The original-source cold baseline below confirms these effects. `DEV#12` guards the smoke runner's own integer CLI/worker plan, whereas
   this independent direct-harness route bypasses that runner.
2. A malformed root step container or a null step can reach an incidental
   `flatMap`/`Object.hasOwn` exception instead of a deliberate fixed refusal.
   Useful diagnostics work, with broader step-schema policy and less impact.
3. Invalid `keycombo` metadata is checked after `screenshotRoot` can create
   output. Its refusal is already before browser launch, so moving it earlier
   has lower practical impact than endpoint targeting.

Choose the first as a bounded prevention of accidentally targeting the wrong
server. This is independent of UI28's denied artifact read and unknown browser
outcome, and of the preserved original GAL12 work. It needs no new browser run
or artifact access to establish its cold boundary.

## Bounded change

Preserve capability and missing-argument checks first. Convert the environment
using the existing `Number` operation and require `Number.isSafeInteger` before
the fixture is read. The fixed diagnostic names the environment setting without
echoing its value or fixture contents. Empty/unset, whitespace and signed zero
still mean zero. Existing valid Number forms such as `+100`, `1e2`, `0x64` and
`1.0` remain accepted; negative offsets are allowed.

After the single fixture read, apply the existing raw-global local-host matcher
unchanged: `\b(127\.0\.0\.1|localhost):(\d{2,5})\b`. Check each effective port
after addition is a safe integer in inclusive `1..65535`. Refuse with a fixed
diagnostic before JSON parsing, browser discovery, screenshots, profile work,
process creation or network. A zero offset validates matched ports while
returning their original matched text exactly, preserving leading-zero and
embedded eval spelling. Nonzero valid offsets retain the existing replacement.

This is not whole-URL validation or a change to the raw-text matching grammar.
Single-digit and six-or-more-digit port tokens remain unmatched; trailing
letters and host word boundaries retain their existing behavior. No claim is
made that these inputs are now validated or rewritten. Changing that grammar
would be a separately scoped repair. No host/scheme allowlist, browser flags,
runtime driver, profile policy or cleanup behavior changes.

## Qualification scope

The 49 new actual-module cases capture the harness and output-helper bodies
before poisoning effects, then load them through data modules. Only the
dependency specifier is substituted. Malformed offsets refuse before fixture
I/O; invalid matched effective ports refuse after exactly one read. Valid cases
stop at the first browser-discovery boundary, which records the actual parsed
fixture data. Separate probes verify that rereads, discovery, output write/mkdir,
profile scans, process creation, fetch and WebSocket are poisoned.

This verifies cold refusal and actual navigation/eval substitutions. It does
not establish browser launch, cleanup, layout or GPU behavior. No runtime driver,
profile cleanup or launch flags changed, so no fresh browser run or screenshot
inspection is required for this input-only boundary. UI28 and GAL12's separate
unknown browser outcomes and all denied artifact reads remain unchanged.

## Original-source baseline (2026-10-28)

The actual unchanged repository harness produced 19 intended failures and 30
ordinary/transformation/poison-control passes in 4.47 seconds. Bad environment
offsets and invalid effective ports reached the poisoned discovery boundary
instead of the required fixed refusal. The captured real parsed data showed
the unchanged default endpoint for nonnumeric offsets and malformed/overflowed
ports for other cases. No browser, profile, output or network operation ran.

## Positive results (2026-10-28)

All 236 focused checks pass in 25.71 seconds, with no skips: the new port suite,
existing capability/fatal metadata, viewport, screenshot-output and smoke-runner
checks, and docs maps. The 49 new cases include ordinary negative/zero/numeric
forms, inclusive effective port boundaries, unchanged matching limits, real
parsed substitutions and eight active poisoned-effect controls. No-cache Ruff
passes. The pinned offline Pyright gate checks 146 files with zero diagnostics.
Independent review is clear at source `aa28c94e` and tests `283db2f8`.

The original-source baseline ran before the one-block source patch. Browser,
network, output and profile effects were instrumented as forbidden throughout;
this is independent of and does not retry any denied artifact or browser action.
