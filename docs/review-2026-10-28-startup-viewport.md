# Review and plan: startup viewport numbers

**Status:** `DEV#18` implemented and locally qualified, 2026-10-28.
Startup geometry refuses invalid numeric/shape inputs before browser preparation.
Independent review and cold checks are complete; no browser run or artifact read
was performed for this slice.

## Evidence and choice

The startup environment currently splits `AURADREAM_VIEWPORT` on `x`, takes
only two values, coerces them and DPR through `Number`, and checks positivity.
Consequently positive infinities, fractional or oversized dimensions and extra
components survive preflight. Width/height then feed both `--window-size` and
`Emulation.setDeviceMetricsOverride`; DPR feeds that CDP call, without a
corresponding scale CLI flag. Nonfinite values become null when serialized.
The source's `send` resolves CDP error envelopes, and startup ignores the
viewport result. Thus invalid metrics need not cause a fixture failure and
may leave geometry different from the requested configuration. These are
source-derived paths, not measured browser behavior or an observed passing run.

Ranked bounded work:

1. Refuse malformed startup geometry before preparation. This has a specific
   invalid-input path, small scope and meaningful cold negative controls.
2. Handle generic CDP error responses. Useful separately, but it changes all
   runtime calls and is broader than configuration validation.
3. Add decimal-only spelling rules or heuristic DPR caps. No evidence supports
   those restrictions, so they are not included.

## Changed invalid-configuration contract

Require exactly two `x`-separated dimensions. Each parsed value must be an
integer from 1 through 10000000, and parsed DPR must be finite and positive.
Use one fixed diagnostic, without echoing environment values, and return exit
2 at the existing startup-check position before discovery/output/profiles,
spawning, network or CDP. DEV17's keycombo/async/startup ordering stays intact.

Preserve unset/empty defaults, `Number`-compatible whitespace, scientific and
integral decimal spellings, plus signs, positive fractional DPR and DPR's
existing compatible hexadecimal spelling. Do not introduce a DPR maximum
beyond finiteness. Startup zero remains refused. This differs intentionally
from a later viewport step, whose protocol-supported zero-reset values remain
accepted. No browser flag, CDP call, response handling or runtime branch changes.

The existing [DEV14 review](review-2026-10-28-viewport-preflight.md) records
the integration owner's 2026-10-28 check of the official
[Emulation protocol definition](https://github.com/ChromeDevTools/devtools-protocol/blob/master/pdl/domains/Emulation.pdl).
It supplies integer dimensions through 10000000 and zero-reset semantics.
The stricter positive lower bound here retains the startup environment's
existing policy. No new official fact or vendor behavior is claimed.

## Qualification scope

The 44 new cases execute captured real harness/helper bodies with one fixture
read and all actual browser/output/profile/process/network effects poisoned.
A declared observation immediately before the original discovery boundary
reports actual VW/VH/DPR locals; it neither replaces guards nor reconstructs
their parsing. A separate valid control omits that insertion and reaches the
actual poisoned lookup. Eight direct probes verify the effect guards, including
a constructible WebSocket poison. Maximum dimensions and huge finite DPR are
observed only as values and never applied to a browser.

The suite covers 19 previously permitted invalid configurations, 16 exact-value
compatibility cases, an unmodified-path control and eight probes. DEV17's
startup diagnostic constant is updated and three permissive controls remain
as strict-refusal checks. Its four other acceptance cases and all combination,
helper and preparation-order checks retain their purpose. Existing per-step
viewport zero resets remain covered independently.

These tests qualify cold refusal, actual parsed values and composition with
existing checks. They do not qualify browser geometry, cleanup or hardware.
Generic CDP error-response handling remains separate; no runtime dispatch,
browser flags or public feature gate changes.

## Original-source negative control (2026-10-28)

The unchanged harness produces 19 intended invalid-input failures and 25 parsed-
value/ordinary/active-poison passes in 2.46 seconds. Invalid dimensions and DPR
reach the observation point instead of refusing. The actual module locals show
fractional/oversized dimensions and silently ignored extra components; the
observer's JSON serialization turns infinities into null. No CDP call, browser,
profile, screenshot directory or network action occurs.

## Positive results (2026-10-28)

All 388 focused checks pass in 21.75 seconds, with no skips: startup values,
preparation order, per-file fixture inventory/containers, port offsets,
capability/fatal metadata, step viewports, screenshot-output, runner and docs
maps. No-cache Ruff passes; pinned offline Pyright checks 146 files with zero
diagnostics. Independent review is clear at source `6e124ba1`, new tests
`889426b8` and adjusted ordering tests `a565af8d`.

The actual original-source control ran before applying the three exact anchors.
No real browser, profile, screenshot root, network, model or GPU operation ran.
All pending UI28/GAL12 work and unknown outcomes remain preserved; no denied
artifact or other stopped action was retried.
