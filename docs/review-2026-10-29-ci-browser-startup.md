# CI: bounded browser startup qualification

**Status:** narrow source/test repair independently reviewed and applied,
2026-10-29 UTC. Local runtime qualification is unavailable after the denied
Python-to-Node launch; hosted qualification on the new head is pending. This does not restart the
autonomous implementation loop or publish any unfinished work.

## Hosted evidence

The first push of `b7b381a0b87cbd341172d50b1cffd6c7a617a6c9` starts
[CI run 40816328616](https://github.com/matt439/AuraDream/actions/runs/40816328616).
Lint and pinned type checking pass. The Ubuntu CPU job reports one failure,
7,488 passes and 83 skips in 252.39 seconds. The sole failure is
`tests/test_browser_profile.py::test_a_passing_run_removes_its_profile`:
`tools/browser.mjs` reports `the browser did not come up` before the page
assertion. Browser-table CI is a separate job; its result is not inferred
from the CPU run.

That failure does not establish whether Chromium was slow or exited. The
original target loop makes 75 attempts separated by 200 ms; profile tests
leave the existing startup diagnostics off. A failed run therefore supplies
neither an early-exit distinction nor its bounded browser stderr tail.
The repository already requires seconds of timing margin on slow CI runners.

## Bounded repair and controls

The startup poll allowance becomes 150 attempts, approximately thirty
seconds plus request time. Readiness still requires the owned profile's
DevTools port and a page returned by the same loopback discovery request.
No success condition, assertion, browser flag, profile/port isolation or
cleanup behavior changes. The profile-test environment enables the
existing explicit diagnostics, so an early exit is distinguished and a
failed browser's stderr tail is retained. Its outer 120-second process
timeout and the interruption test's separate thirty-second profile-ready
deadline remain in force.

`tests/test_browser_startup_wait.py` loads the entire actual harness and its
actual output dependency, changing only the local dependency specifier for
the controlled module load. Builtin filesystem/process ports and virtual
sleep are inert; no browser is launched, no real loopback request is sent
and no real browser profile is changed. Only owned fixture input/temp
files are created. A WebSocket sentinel ends the controlled
load before any CDP step. The cases cover readiness after the original
75-poll allowance, permanent absence bounded at the new allowance, and
diagnostic early exit. The controls are designed to observe the harness's real exit cleanup.

The attempted original local control run reports eleven failures in
0.97 seconds, all at Python's `subprocess` CreateProcess of `node` with
`PermissionError: [WinError 5] Access is denied`. None reaches JavaScript,
so none is a measured original regression or pass. The local Node launch
path is stopped: no retry, alternate executable, runtime or browser route.
The predicted two regression failures/nine controls remain unmeasured.
Actual hosted Linux checks, where the existing authorized CI already runs
Node, must qualify the new head before merge; this does not reroute or
repeat the denied MattPC process launch. Independent source/test review is
CLEAR; permitted Python-only static checks are recorded below.
Windows profile tests use fresh owned temporary directories and headless
profiles; they do not operate the signed-in GitHub browser or its login tab.
No Linux reproduction or explanation of the historical failure is claimed
without hosted evidence. Full latest-head CI remains the publication gate.

All fourteen prior unfinished files, denied operations, unknown earlier
jobs/artifacts, real-data and hardware qualification boundaries remain
preserved. No check is skipped or weakened and no permissions, credentials,
trust settings or browser security flags are changed.


## Existing footer smoke assertions

The same hosted browser job finishes 92 of 95 smokes in 851 seconds.
`smoke-swap` and `smoke-more-seeds` each fail their single old footer
adjacency assertion: Bin is no longer immediately followed by Bundles.
Current `index.html` places Saved searches between those buttons, and
`state.js` maps it as `els.savedSearches`. Each fixture changes exactly one
predicate to require Bin, Saved searches, then Bundles; every other clause,
step and behavioral assertion stays intact. Independent whole-file review
confirms this exact replacement. No product layout change is made.

## Synthetic model-hover probe

The third failure is the final off-catalogue probe in `smoke-model-hover`.
Its first settle predicate passes, its composite no-mutation/empty-choice
predicate fails, and real-status recovery passes. CI does not identify the
failed clause. Source tracing shows a concrete interference path: the
three-second status poll calls `receiveModelSettings`, replacing the
fixture's locally fabricated off-catalogue runtime with the server's real
FLUX settings between separate commands. This is source-derived evidence,
not a measured reconstruction of that historical failure. Observer queues
alone are not a diagnosis: the reset already drains `takeRecords()`.

One controlled async expectation holds only the model-settings application
boundary while testing the synthetic runtime. Status polling and the rest
of status handling continue; replies already in flight also encounter the
temporary receiver when delivered. Both original settle/idempotence
predicates and both actual repeated apply calls remain. A promise checkpoint
delivers any resulting observer mutations before their zero-count checks.
`finally` restores the exact receiver, runtime and tracked fetch on success,
false or throw. A new identity assertion verifies restoration, followed by
the unchanged actual refresh/recovery predicate and observer cleanup.
The visible fixture log labels the synthetic held scope. No product model,
status, API or polling behavior changes. Independent exact-file review is
CLEAR; fixture `20368e9a` remains unrun locally after the Node denial.


Publication authorization explicitly permits narrow independently reviewed
CI fixes to be committed/pushed for hosted qualification when local runtime
access is unavailable. Local qualification is never reported as a pass.


## Permitted local static checks

Python-only structural checks pass: actual HTML direct siblings and the
Saved searches binding, exact reviewed source/profile/footer identities,
both Python AST parses, all three fixture JSON shapes and frozen model-hover
and cold-test identities. Final no-cache Ruff passes. Documentation inventory
passes all six cases in 0.99 seconds, with the installed TestClient/httpx
deprecation warning. No Node, browser, JavaScript or local type-check runtime
was retried after the denial. Hosted exact-head CPU/type/browser checks are
still required before merge; these static results do not replace them.
