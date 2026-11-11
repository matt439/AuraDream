# Review and plan: existing script and startup refusals before preparation

**Status:** `DEV#17` implemented and locally qualified, 2026-10-28.
Existing static script/startup checks precede browser and screenshot preparation.
All 344 focused cases are qualified across the combined run and targeted inventory
repair check. No real browser or artifact operation was performed.

## Concrete effect and bounded value

The current harness validates `keycombo`, detects existing unawaited async-helper
expressions and checks startup viewport/DPR positivity before browser launch,
but after `CANDIDATES.find(existsSync)` and `screenshotRoot(steps)`. A fixture
with a screenshot and duplicate Control modifiers can therefore create an
output root before its existing fixed refusal. Unawaited-helper refusals and
nonpositive startup values have the same ordering. With no installed browser,
the unrelated no-browser error hides the actual diagnostic. Cold negative controls below confirm lookup and mkdir attempts without
performing browser operations.

Moving the common preparation block after these checks gives existing bad input
a deterministic cold refusal and leaves no output directory to mistake for a
started smoke. It fixes the shared boundary once rather than moving individual
validators in successive commits. The value is lower than the offset-targeting
fix, but the effect is concrete and needs no new schema, numeric policy,
browser activation or speculative feature. No current Python test mentions
`keycombo` in the targeted `tests/*.py` literal scan; native modifier dispatch
already exists in the runtime branch.

## Exact change and dependency review

Move `CANDIDATES`, browser discovery/refusal and the complete
`screenshotRoot`/screenshot-log block, without changing any of their code,
from before `KEY_MODIFIERS` to immediately after the existing startup
`VW`/`VH`/`DPR` positivity guard and before the profile section. Keep all
validation predicates and the validators' relative order unchanged. Add only
a comment describing the preparation boundary. Declarations keep their same
top-level lexical scope.

The moved block uses only `steps`, environment and imported helpers. The
intervening validators use only `steps`, environment and pure local helpers;
none depends on `browser` or `screenshots`. `browser` is first used by the
later spawn and `screenshots` by capture steps. A valid fixture retains the
same screenshot announcement before profile work/spawn; successful checks in
between do not log. Refusals now produce no screenshot announcement or root
allocation.

The existing combination contract remains: an array contains at least a
modifier and a nonempty string key; preceding modifiers must be known and
unique. Alt, Control, Meta and Shift keep their masks/order. A space as the
key and a modifier name as the last key retain their existing acceptance.
Native key events, focus, flags, profiles, capture and cleanup are unchanged.

At DEV17, startup's `> 0` and split/destructure policy stayed unchanged,
including Infinity, fractional dimensions and ignored extra `x` fields; cold
controls never sent them to CDP. The subsequent [DEV18 startup guard](review-2026-10-28-startup-viewport.md)
now refuses those inputs. The three controls remain as refusal checks and the
startup diagnostic is updated; the preparation-order boundary is retained. With several kinds of bad input, the same earlier
validator wins. Existing combo/async/startup refusals now precede missing-browser
or invalid-screenshot-path errors. This changed invalid-input error precedence
is intentional.

## Qualification scope

The 69 new cases execute captured actual harness/helper module bodies with only
the dependency specifier changed. One fixture read is permitted. Browser lookup,
output, profile, spawn, fetch and WebSocket effects are poisoned. Fixed messages
and relative combo/async/startup refusal priority are retained. Inert successful
lookup exposes the prior screenshot mkdir attempt without reading an executable
or creating an output directory. Accepted values stop at the first lookup.

The suite covers 18 malformed combinations, aggregate diagnostics, ordinary
modifiers/keys, eight active poison ports, existing unawaited-helper refusals,
startup positivity failures, preserved numeric/parser acceptance, ordinary
awaited-helper forms and refusal priority. These checks establish cold ordering
and preserved predicates; native event dispatch, layout, cleanup and hardware
remain outside their scope. Launch flags and valid-run preparation are unchanged.

## Initial baseline fixture correction (2026-10-28)

The first original-source run reported 40 failures and 29 passes in 3.67
seconds. Besides 39 intended ordering failures, the WebSocket poison probe
failed because its factory returned an arrow function and `new WebSocket`
could not construct it. An isolated diagnostic reproduced that TypeError
in 0.31 seconds. Only the fixture factory was changed to return an ordinary
function; all assertions, markers, counters and application source remain
unchanged. A corrected original-source baseline is required before applying
the source relocation. No browser or network operation ran.

## Corrected original-source baseline (2026-10-28)

After the one-line fixture repair, the actual unchanged harness produces the
39 intended ordering failures and 30 ordinary/active-poison passes in 3.42
seconds. Cases reach either the first poisoned discovery call or, with inert
successful lookup, the poisoned screenshot mkdir before their existing refusal.
The corrected WebSocket constructor probe reaches its named poison normally.
No browser, profile, output directory or network operation is performed.

## Positive qualification (2026-10-28)

The combined positive run passed 343 cases, including all 69 new ordering cases,
and failed one prior inventory test in 15.37 seconds. That test concatenated
unrelated fixtures, allowing async helper names from one file to contaminate
another. It now passes each original JSON document independently to the actual
cold harness, preserving all assertions and effect guards. Independent review
is clear at the revised inventory test `c27bb684`.

Only the changed inventory and docs maps were rerun: seven tests pass in 5.47
seconds. Together with the 343 unchanged passing cases, all 344 focused cases
are qualified against the same source; this is not a claim of a second full
combined run. No-cache Ruff passes, and pinned offline Pyright checks 146 files
with zero diagnostics. Source review is clear at `6b5ba54c`, the new tests at
`f0abb503`. The [container review](review-2026-10-28-browser-step-containers.md)
records the inventory scope correction and its historical aggregate result.

The original preparation block is unchanged byte for byte and remains before
profile/spawn work. Pending UI28 and GAL12 source and their unknown browser
outcomes remain preserved. No denied path or operation was retried.
