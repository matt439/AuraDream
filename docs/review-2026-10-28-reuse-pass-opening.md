# Review and plan: Reuse pass opening ownership

**Status:** closed 2026-10-29. `UI#28`'s source, 31 actual-function VM cases
and the focused browser fixture pass; an independent review's finding is fixed
with its own negative control (see
[final qualification](#final-qualification-2026-10-29)).

## Evidence and choice

Ranked independently feasible work after the current source slices:

1. Protect newer drafts from a delayed Reuse-pass source lookup. Reuse and
   Refine are reported Often (`G1`, `H1`, `H2`, `H4` in
   [preferences](design/preferences.md)); the source sequence can discard a
   newer dialog draft. It has one bounded ownership problem and leaves the
   editing/generation semantics unchanged.
2. Refresh the Runs picker's star counts after confirmed single/bulk favorite
   changes. Current favorite actions update image surfaces but do not refresh
   normal-success run summaries. Useful, smaller visible impact; native-picker
   interaction needs care when its options are rebuilt.
3. Give a failed run-list scope refresh an explicit stale/error/retry state.
   `loadRuns` guards successful replies but leaves prior-scope options after a
   refusal. Useful, with a broader recovery interaction to decide carefully.

This slice selects the first. Before the change, `reusePass` awaits
`GET /api/images/{rec.refine}` and then calls `openRefine(source, rec)` without
checking what happened while it waited. A user can press Reuse on pass A, then
open Refine directly on B and edit B's prompt. `openRefine` writes its source
and every draft field even when its dialog is already open. A's late success
therefore replaces B's draft. A late missing-source response similarly calls
`reuseIntoPanel` after the user has typed or reset the composer; a late ordinary
error overwrites the current message. These are current-source event sequences,
not measured browser reproductions yet.

The existing More seeds opener supplies the ownership pattern: one exact
opening object, an abort signal and context validation before using a reply.
The preview-local `refineEpoch` is unsuitable as an opening owner because it
also advances with preview work and is not advanced synchronously at every
competing open.

## Bounded repair

`reusePassOpening` owns only the source lookup. Its raw draft key reuses the
existing pure `setupComposerState` UI snapshot, including transient Grid Resume,
Then refine and raw/unavailable performance intent. Model/revision/server are
removed from that key and checked separately so that learning the first server
identity remains allowed. Number-box `validity.badInput` flags supplement the
raw values: Chromium can display an unfinished edit while exposing the same
empty value as before it. No `collect`, numeric coercion or request validation
runs while capturing a half-written draft. The opening also captures the
loaded checkpoint/revision, the known server start identity, current gallery
scope and currently open dialogs/lightbox identity. A card/result host is bound
only when the actually focused element's closest image host carries this
record's id. Body, menu items and a lightbox's unrelated restored focus do not
become required hosts. Viewer identity matters only for a bound viewer origin.

Every newer Reuse starts by retiring the previous source lookup. A deliberate
panel replacement and a direct `openRefine` do the same. Reset already calls
`clearReuseNote`, which now retires a lookup even if Reset leaves an identical
empty raw draft. The existing `syncReuseUndo` call on `persistPanel` retires a
lookup on meaningful raw draft changes, so typing and undoing the typing does
not revive the old intent. No new panel event layer or layout control is added.

`api` receives the opening's abort signal. Its existing catch only translates
fetch `TypeError` into an offline-marked error; an `AbortError` is rethrown
without writing global offline/error state. Aborted or obsolete reply handling
is silent. A transport that ignores abort still cannot use the reply because
the exact owner and captured contexts are checked again. A known server change
invalidates a request; learning the first server identity is allowed.

The current owner is released before a deliberate successful `openRefine`,
missing-source fallback or current error. Those current actions therefore do
not cancel themselves. `finally` clears only the same opening object and
cannot clear a newer lookup. The existing missing-source fallback keeps its
prompt/sampling behavior, unchanged composer size, explanatory note and Undo;
ordinary current errors keep their existing wording.

The gallery scope key compares effective search, Starred, All models, bin and
sheet state, rather than the paging epoch. An automatic same-filter refresh
alone is not a new filter intent; a bound host that disappears or becomes hidden
still invalidates its lookup. New dialogs, changed captured lightbox identity,
viewer replacement for a viewer origin and model context changes invalidate
settlement. This is a page-local opening guard, not a durable intent journal,
request timeout or a history of every unrelated modal opened and closed again.

The change does not repair the separate asynchronous source lookups of Inpaint,
Outpaint, instruction edits or source-navigation links. A newer such Reuse
retires a pending plain pass through the common `reuse` entry, but their own
late-reply behavior is outside this slice. No deferred editor gate is changed.

## Required qualification

The proposed actual-function Node VM supplement is unrun. It loads the real
generation, panel, Reuse, Refine and named-setup UI scripts (the latter only
for its existing pure snapshot helpers), with inert presentation/storage/timer ports
and deferred API promises. Current success, missing-source fallback and error
must work against original source; stale success/error/fallback, direct newer
Refine, raw composer edits/Reset, reversed newer Reuse replies and exact-owner
cleanup should fail meaningfully before the repair. Context cases include
ordinary card/viewer/closed-lightbox origins, scope/model/server changes,
origin disappearance, new dialogs, viewer replacement, transient Resume/Then
refine/raw performance changes and a number's bad-input-only transition. Abort-insensitive
deferred replies must still be harmless.

`tests/test_reuse_pass_opening.py` proposes 30 named cases, including seven
ordinary cases expected to work on the original source. Its stale cases assert
visible draft/source/message effects before new token or signal names. The
scope test exercises search changes; the other scope-key fields are additionally
source-reviewed rather than exhaustively cross-product tested. Inert focus
ports do not establish native focus or layout behavior. Copy this proposed test
into the actual repository before taking a baseline: its source root is the
adjacent `web` directory, so running it in the packet would read proposals.

The fresh-data browser regression uses actual owned fake-server source/pass
records. Hold a real source-A GET after physical Reuse(pass A), open Refine B
physically and type B's draft natively, then release A and require B's source,
raw values and focus to remain. Include stale declared 404/error responses after
typing/Reset or a newer Reuse, followed by current missing-source fallback and
current ordinary error/retry. Clearly label controlled unforwarded responses;
do not delete an actual gallery image merely to make the fixture's 404.

One explicitly declared abort-insensitive response gate may preserve the first
real GET despite its caller's abort so that the browser checks reply ownership
as well as cancellation. It must preserve the actual source response body,
origin and exact approved path. Other requests should preserve normal options.
All mutation/setup prerequisites must be fatal, count only owned fake work and
finish with no active jobs, unresolved response gates or unexpected mutations.

Run the original-source negative controls before applying source, then the
focused VM and classic-script/shared-scope/docs checks, relevant existing
Refine/Reuse smokes serially, no-cache Ruff and the pinned type gate. Inspect
only new permitted screenshots. No real gallery, model weights, GPU,
credentials, old artifacts, WSL or trust/security changes are needed. Preserve
all pending GAL12 work and its previous unknown test result.

## Original-source negative controls (2026-10-28)

The actual browser baseline completed with exit 1 at step 32 after three real
owned jobs created four source/pass records. Physical Reuse A held its actual
source reply; physical Refine B opened and native typing wrote B's new draft.
Releasing A replaced B's source/raw fields/focus. The fatal check stopped later
steps. Runner 12 seconds, browser 6; the browser closed normally with exit 0.

The focused actual-function VM baseline then reproduced 23 stale-effect failures
with seven ordinary success/fallback/error/origin cases passing in 1.68 seconds.
The five real UI scripts and their entry/snapshot functions ran against inert
presentation ports and deferred API replies. This does not qualify browser
layout or hardware; the subsequent positive results still need recording.

## Positive checks and interrupted browser qualification (2026-10-28)

The 30 new actual-function cases, classic-script/shared-scope guards, docs maps
and instruction-edit UI checks pass together: 70 tests in 6.72 seconds. The
first combined attempt passed 69 tests and failed one existing lineage fixture
because that inert fixture lacked the already-used `syncBinWriteButton` port.
It does not load either changed Reuse/Refine script. The fixture now supplies
an unheld bin button with an assertion on its exact source id; all original
assertions and production facts/bin code remain unchanged.

The positive browser command was started once as session 10479. At the next
checkpoint, polling returned `Unknown process id 10479`. Reconciliation then
attempted `Get-ChildItem -LiteralPath` on only that new run's reported root
`C:\Users\matsi\AppData\Local\Temp\auradream-smoke-artifacts-33esgww9`
and its reported `smoke-reuse-pass-opening-7dizc15l` child. Both accesses returned
`Access to the path is denied`; no result, log or screenshot was read.

The command's outcome is therefore unknown. No positive browser pass, visual
inspection, browser cleanup result or absence of remaining owned work is
claimed. The denied paths and equivalent access routes are stopped. The focused
run will not be repeated to replace this unknown outcome. Source, fixture and
test changes remain uncommitted, separately from the older pending GAL12 work.
Independent permitted cold checks and other useful source work can continue.

## Final qualification (2026-10-29)

The rerun of the positive browser fixture reached every behavioural check:
stale success, stale 404/500 after typing, Reset and a newer Reuse, current
fallback, current error, and retry. It then failed only its final accounting
gate. The page's advisory `POST /api/prompt/measure`, the cold token count
sent after the fallback writes passA's prompt into the composer, was counted
as an unapproved mutation. The fixture now passes it through beside
`/api/prompt/analyze`, and the fixture passes in 9 s. Against the original
`reuse.js`/`refine.js`, it still fails fatally at step 32 ("obsolete Reuse
success replaced newer Refine source/draft/focus"), as in the baseline.

An independent review confirmed one defect. Two writes the page makes itself
changed the raw draft key and silently cancelled a current lookup: a grid's
drawn seed written back into a blank Seed box (`startJob`), and
`refreshLoras` dropping an adapter the server no longer offers (through
`dropRefusedLora`). Neither is a newer user intent. `keepReusePassDraft(write)`
now rebases the opening's draft across such a write, but only when the draft
still matched the panel just before the write. `valid()` reads
`opening.draft` rather than the captured constant. New VM case
`page-seed-write-success` fails without the rebase and passes with it (31
cases). The review's other UI#28 claim was refuted as the documented design:
an automatic rebuild of the bound host retires the opening.

The new wide/narrow screenshots were inspected. B's typed Refine draft
survives the old reply, and the current retry opens Refine #3 with Strength
focused, with no clipping or overlap. The topbar band in the full-page
captures is the sticky header in a stitched shot. Related `smoke-refine` and
`smoke-refine-sdxl` regressions and the full test run are recorded in the
final run of this session.
