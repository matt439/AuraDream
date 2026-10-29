# UI#31: keep a dismissed Check hidden

**Status:** implemented and locally qualified, 2026-10-29 UTC. Original
failure controls, actual-module tests, focused native behavior, three new
screenshots, related analysis regression, Ruff and pinned type checking are
complete. Final documentation inventory also passes before the local commit.
This is the last slice under Matt's stop-after-next-commit instruction.

## Evidence and priority

[Reported use](design/preferences.md) rates Check rules/order Often (B7),
Apply/reorder and critic/rewrite Sometimes (B8-B10), and keyboard navigation
Often (L6). A prompt-operation audit ranked one bounded usability defect:
a pending Check or Check again could reopen the panel after Close/Escape;
main Check could also steal focus back to the late result's heading.

Close retains findings. Reset/checkpoint retirement through `clearAnalysis`
is stronger: it advances computation epochs and clears caches. Using that
for Close would break accepted behavior, including the critic pre-check
which awaits the accepted analysis result even if its panel is dismissed.

## Source contract

Only `web/analysis.js` changes at runtime. A separate dismissal counter is
captured by a pending check. Close/Escape advance it without changing
computation epochs, results, critique caches or finally/button handling.
An accepted late response updates retained findings and normal stale/critic
state, and returns to a waiting critic pre-check. It cannot reveal a panel
dismissed since that request began. Both heading-focus handoffs also
require an eligible visible panel. A later explicit Check can reopen it,
including the same prompt. Newer checks, Reset and model changes keep their
existing computation ownership. No editor writes or critic/API changes occur.

## Actual-module qualification

`tests/test_check_dismissal.py` contains seventeen cases against the actual
analysis and critic declarations. DOM/detail-rendering/API dependencies are
inert ports installed at their intended boundaries. These cover main/again
Close/Escape, retained accepted cache, returned focus/editor selection,
deliberate reopening, same-prompt critique retention, wide/narrow ordinary
focus, stale-write holds, newer-check/old-finally ownership, stronger clear
retirement, ordinary failure cleanup, and critic pre-check continuity and
hidden error routing. Unit clear cases do not claim physical Reset/model
switch acceptance, and critic promises invoke no HTTP/server/service.

On the original source, eight intended reveal/focus/error-routing assertions
fail and nine controls pass in 1.04 seconds. No SyntaxError/ReferenceError is
reported. The repaired combined targeted run passes all 52 cases in 5.18
seconds: seventeen new, 29 classic-script and six documentation checks.
One installed TestClient/httpx deprecation warning appears in these runs.

## Native qualification and fixture correction

The normal fresh FLUX fake on 7952 installs the tests' fake tokenizer and
FakeCritic before serving. It reaches no Hub, real tokenizer, critic service
or image model. The focused fixture rejects every critique request before
dispatch. Six exact one-use analyze bodies are forwarded once; each real
response is cloned for evidence, held after receipt and returned unchanged.
Four native prompt input events separately grant their exact prompt-only
measurement body once. Those real cold fake-tokenizer responses also pass
unchanged, and all four must settle with HTTP 200/current model/encoders.
Extra bodies, repeats, query variants and unrelated writes remain refused.

The original source fails fatal step 18 after native Close returns focus to
Check, computation epochs stay unchanged, and the held actual analyze reply
is released: the response reopens/refocuses the panel. Runner ten seconds,
browser four, exit one; browser exit zero/no signal. Later steps and shots
are unreached. This is independent of the original-source VM failures.

The first repaired native run passes behavior through step 45 but fails the
generic final refusal guard. A separate frozen v2 changes only diagnostic
text; it identifies four ordinary `/api/prompt/measure` requests from the
fixture's own native edits, with zero critic attempts or JavaScript errors.
Those refused measurement requests were not forwarded. Source tracing in
`tag-tokens.js` and `api_prompt.py` proves they are the normal debounced count
route. V3 adds only the exact trusted-edit grants and strict settlement
checks above. All six analyze grants, behavior predicates, critic refusal
and side-effect guards remain. No product source correction was needed for
this fixture mistake; neither earlier failed run is counted as acceptance.

V3 passes all 46 steps, including seventeen fatal expectation steps, in
thirteen runner seconds/six browser seconds. Physical Check/Check again,
Close/Escape, editor selection, two later deliberate reopenings, normal
focus and all final guards pass. Six analyze and four measure requests each
finish exactly once; no critique or unrelated mutation occurs. The actual
two planted row bodies, PNG hashes, empty jobs/queue and non-prompt composer
values remain unchanged. Optional comparison variants are unticked as
declared disposable-page setup; comparison behavior is not qualified here.

Root inspects these three new 1400 by 1000 screenshots under
`auradream-smoke-artifacts-pvjbg7nd/smoke-check-dismissal-g3aey59y`:

- `ui31-close-after-real-reply.png`: hidden panel and blue prompt;
- `ui31-escape-after-real-reply.png`: hidden panel, focused selected green text;
- `ui31-current-explicit-check.png`: current gold prompt and open findings.

No clipping/overlap is observed in the inspected states. Focus/cache/epoch
claims use the actual assertions; screenshots alone do not prove them.
No narrow-browser or real user/device acceptance is inferred.

## Regression, review and boundaries

The existing related `smoke-analysis` regression passes on its separate
fresh fake. No-cache Ruff passes; pinned offline Pyright 1.1.407 checks 146
files with zero errors/warnings. Final documentation results are recorded
below. Tests and browser runs are serial; no full-suite/GPU run is claimed.

Independent source/VM and source/native reviews are CLEAR. The final v3
measurement correction is separately reviewed against frozen v2: only the
log, setup/grants and final settlement checks differ. Executed source is
`0fa3f645`, new VM test `1a3da8f1`, native v1 `41a271e0`, diagnostic v2
`ca4a9e60` and qualified v3 `b5405b4e`. Frozen packets remain in the task
workspace; only owned documentation hunks and 7952 are staged.

All fourteen prior pending files remain preserved, including GAL12/UI28
and unapplied GEN31's fixture/7951 registration. GEN31 native 22364, GAL12
and UI28 older interrupted outcomes/cleanup remain unknown; their denied
artifact reads and stopped equivalent searches remain stopped. GAL19's
denied visual read and compiler/WSL/access boundaries are unchanged. No real
gallery, credentials, security/trust setting, service, card or release gate
is touched. After the one qualified local commit, this loop is stopped.


Final checks: the related analysis regression passes in 92 runner seconds
(86 browser); pinned Pyright's 146-file check takes 11.071 seconds. The final
documentation-only run passes all six cases in 0.23 seconds, with the same
installed TestClient/httpx deprecation warning. The six documentation cases
were also part of the earlier 52-case run; they are a targeted final rerun,
not six additional distinct tests or another complete suite execution.
