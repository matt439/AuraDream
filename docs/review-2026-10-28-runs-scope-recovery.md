# Review: Runs scope recovery

**Status:** `GAL#18` implemented and locally qualified, 2026-10-29 UTC.
Independent review, actual-handler controls, the new native browser fixture,
related regressions and inspection of three new screenshots are complete.

## Problem, evidence and priority

`loadRuns` currently checks epoch/model/revision/All models only after a
successful response, and its catch simply returns. Existing options remain
selectable while a new scope waits and after its failed response. For example,
All models shows a real foreign run; switching it off starts a current-model
request but still offers that foreign run. Choosing it opens a sheet under the
new gallery scope, where it does not belong. An earlier closed-picker keyboard
candidate also survives the scope boundary and can open on its later blur.
The original-source controls below reproduce the stale choices through
actual handlers and a native All models change.

The preferences report recent runs (`E4`), Starred (`E2`), single stars (`F5`)
and picked stars (`G9`) as Often. Ranked independent findings:

1. Stop prior-scope Runs options and keyboard candidates surviving a failed
   scope refresh; make deliberate recovery available. Bounded correctness
   repair with a concrete real caller, chosen for `GAL#18`.
2. Refresh Runs favorite counts after confirmed ordinary star changes. Current
   success paths update image surfaces but refresh Runs only for missing or
   binned ids. Smaller impact; leave this separate from scope recovery.
3. The split-submission extra can be adopted before a delayed Swap receipt
   re-adds it as its own queue row. Source evidence remains, but meaningful
   actual-entry race qualification is needed before choosing its repair.

The source repair is confined to `web/runs.js`. No gallery filter, sheet,
composer, favorite write, queue behavior, HTML, CSS or bin implementation changes.

## Request, publication and input ownership

Capture model, revision, All models and known server start identity at each
read. The latest epoch and live context own both success and failure. A known
server identity must match directionally; an unknown initial identity may
learn its first matching observation and is then bound, including on sync or
settlement. This is not physical-server identity proof. Existing status code
already reloads Runs after model changes and known restarts; no automatic
refetch or status-module change is added to simulate unusual test contexts.

A different context synchronously removes old options and clears ordinary and
retry keyboard candidates before awaiting the request. The picker shows
`Loading runs…` and is disabled. A current failure shows `Runs unavailable`
and an enabled `Retry loading runs` action option. This uses the existing
native select and its width; no new toolbar control is added.

For a same-context refresh with useful options, keep their identity and a valid
ordinary keyboard candidate while waiting. Failure appends one Retry option
without rebuilding the useful list. Success retains a candidate only if that
run still exists in the new list, and retires any old Retry candidate. Empty
successful results keep the existing `No runs yet` behavior. Grouping, labels,
shown-run selection and successful retained-run tool refresh stay unchanged.

The action belongs to its selected Option's dataset marker and membership,
not its value. A real imported run may collide with the action's value; a
Retry value may even be empty in a fixture. Selection by index preserves the
action identity in those cases. Ordinary `runPending` remains a run id, while
Retry has a separate candidate flag. Every change/commit checks live published
context and the actual current option. These are UI lifecycle guards, not a
claim that arbitrary hostile DOM injection is a supported security boundary.

Closed-picker arrows/typeahead only choose a candidate. Enter or blur commits
one deliberate Retry; Escape cancels it. Held/modifier/repeated Enter does not
start a retry through the local handler, and focus alone does not fetch.
Starting the read synchronously retires its action/candidate, preventing an
Enter/change/blur sequence from duplicating it. Retry never routes through
`showRun` or rewrites the search, sheet or composer. Same-context valid run
choices retain their existing keyboard behavior.

The existing bin mouse/key capture and aria hold remain in place. Its call to
`syncRunList` retires both pending choices, and direct change/commit handlers
also guard bin mode so that a stale blur cannot leave the bin or issue Retry.
Returning from the bin does not itself retry.

## Qualification scope

`tests/test_runs_scope_recovery.py` evaluates the full real Runs script and its actual load/sync/
key/change/commit handlers with an inert Select/Option model, deferred API
reads and observation ports for sheet opening and unrelated Finish-tool
refresh. Keyboard eligibility helpers are inert ports; native behavior belongs
to the browser fixture. Existing actual date formatters run normally. The
supplement covers scope waiting/failure, latest replies, model/revision/server
contexts and initial identity learning, ordinary candidate preservation and
retirement, Retry/empty/colliding values, Enter/blur/Escape/modifier/repeat,
same-scope option identity and bin/direct-handler holds. It does not establish
native select focus, geometry, accessibility or event ordering.

The new `smoke-runs-scope-recovery.json` uses fresh FLUX data at 7947 and the
ordinary existing Edge driver, one row at a time. The fake's two actual
other-model rows have `job_id=run_id=planted`; All models returns that real
two-image summary and current FLUX scope returns an empty list. Thus its setup
is GET-only: no jobs, model switch, image mutation or synthetic run is needed.
Exact one-use response gates first obtain and verify actual server bodies,
then hold unmodified success or supply clearly declared controlled HTTP500
read refusals. The wrapper observes original `loadRuns` promises without
replacing their logic. It refuses and counts foreign/non-GET fetches.

Its first original-source negative control physically toggles All models on,
then off, and requires the foreign option to disappear while the actual new
scope GET is held. That fatal public-DOM check precedes future helper checks.
The 79-step fixture has 40 fatal expectations and three new wide/narrow shots.
Later steps check native focus/arrow/Escape, Enter and repeated Enter, physical
Search-click blur, same-context keyboard preservation, late old replies and
bin holds. It preserves raw composer/search/sheet/picks, planted rows and the
gallery error slot except the ordinary explicitly exercised bin refusal.
Final guards require every gate settled, no unexpected fetch/JavaScript errors,
no sheet opens, no jobs/receipts/queue work and default-off backup metadata.

A separate pending-choice scope probe is explicitly programmatic: checkbox
`click()` invokes its actual change handler without first moving native focus,
then a physical Search click supplies blur. A physical checkbox click normally
blurs first and may legitimately commit its then-valid old run. No native
checkbox claim is made for that supplement. The driver does not map Tab, so
native blur probes use Search clicks rather than assuming bare Tab works.
Modified commands/model/revision guards belong to VM/source checks; the
browser does not press Ctrl+Enter and risk the global generation shortcut.

Qualification runs serially on fresh owned data/output. Actual original-source
VM and native controls precede source application. Positive actual-handler,
classic-script and docs checks are separate from native and visual acceptance.
The new fixture and existing gallery/shortcuts/bin regressions use the ordinary
restored Edge harness with one worker. New screenshots are inspected only for
their own run. No model weights, GPU or real gallery acceptance follows.

No old GAL12/UI28 fixture is repeated and no denied artifact is read/copied or
accessed through another route. Known unknown outcomes, pending source and all
other access/activation boundaries remain unchanged. Shared docs/AGENTS must
be applied and staged only as this proposal's owned anchors by the root.

## Original-source controls (2026-10-29 UTC)

The frozen actual-handler VM supplement produces 25 expected assertion
failures and four ordinary/previous-guard controls passing in 1.68 seconds.
The first failure directly observes the foreign option remaining while the
new-scope read waits; another observes a pending old run opening on later
commit. Other failures cover current read recovery and the new context/input
guards. No SyntaxError or ReferenceError occurs. Full output is retained in
the owned workspace. One installed TestClient/httpx deprecation warning occurs.

The separate ordinary Edge run confirms the first symptom through actual
native All models changes and the real fresh fake's planted Runs body. At
fatal step 10, the old foreign run remains selectable while the actual
current-scope GET is held. The fixture stops there, before any future-helper
assertion or later recovery step. Browser exit is zero with no signal; runner
exit is one from the intended failed assertion, after 11 seconds (browser two).
The initial checks confirm two planted foreign rows and no jobs, queue work,
receipts or active backup/credential configuration. Later final guards were
not reached and are not counted as evidence for this negative run.

The newly reported artifact root is `auradream-smoke-artifacts-w_zakjjb`
with child `smoke-runs-scope-recovery-wlq49llm`. No old denied artifact was
read or recreated, and no old GAL12/UI28 fixture was repeated.

## Positive qualification (2026-10-29 UTC)

All 64 targeted actual-handler, classic-script syntax/shared-scope and docs-map
checks pass in 6.34 seconds, with no skips. The VM's 29 named cases pass.
One installed Starlette TestClient/httpx deprecation warning is reported.

The complete 79-step native fixture passes in 14 seconds (browser eight),
including its 40 fatal prerequisites/checks. Fourteen original loadRuns calls
reach fourteen one-use real read-response gates. New-scope loading/failure,
native Retry through Enter and Search-click blur, Escape, repeated Enter,
same-context pending choices, obsolete replies and bin holds all pass.
Final checks require unchanged planted rows and raw composer/search/sheet/picks,
no jobs/receipts/queue work, no mutations or JavaScript errors, zero unintended
run opens, every gate settled and default-off backup/credential configuration.
The external programmatic-scope and direct-handler probes remain labeled
supplements; they are not claimed as native checkbox or hostile-DOM tests.

The new `runs-scope-unavailable-wide.png`, `runs-scope-unavailable-narrow.png`
and `runs-scope-recovered-wide.png` were read and visually inspected in
`auradream-smoke-artifacts-8sf5ucxw/smoke-runs-scope-recovery-h8tenj08`.
Closed picker labels, focus rings and neighboring controls fit at both widths;
these screenshots do not depict the operating system's open popup menu.

Existing regressions pass serially: `smoke-gallery` (64 seconds),
`smoke-bin` and `smoke-shortcuts` (9 seconds), with all three
passing in the runner's 100-second total. They retain ordinary native
run opening, keyboard browsing/commit, gallery/sheet behavior and bin holds.
No-cache Ruff passes; pinned offline Pyright 1.1.407 checks
146 files with zero errors/warnings in 11.938 seconds.
Independent frozen review is clear at source `6a125fda`, tests `863920a1`,
fixture `49e9410d` and original owner `fb1825c1`.

The three new screenshots were independently permitted. No denied old artifact
was read/copied/recreated, and no unknown GAL12/UI28 fixture was repeated.
Their existing uncommitted work and unknown qualification remain separate.


### Staged fixture identity

The executed native fixture and frozen packet retain raw hash `49e9410d`.
Git stages its 301 CRLF line endings as LF, yielding `6ba858b8`. An exact
byte comparison after that normalization and full parsed-JSON equality both
pass. The first commit guard stopped before committing on this raw/blob hash
mismatch; the verification now distinguishes those identities. No fixture
content, assertion, application code or test result changed, so no browser
rerun was needed for this serialization-only difference.
