# Review and plan: queued cancellation recovery

**Status:** `GEN#28` implemented and locally qualified, 2026-10-28.
The accepted source proposal is SHA `963eaf70`; the dedicated regression
fixture is SHA `8250f153`. Changes are local; no GPU or release acceptance follows.

The first ordinary Edge baseline launched and passed the cold scratch-data
prerequisites, then stopped at step 7 during fixture setup. The fixture required
32 hexadecimal job-id characters while the manager returns 12, so its first
accepted fake submission was rejected by the wrapper. It did not reach the
cancellation regression. That run completed with exit 1; the browser closed with
exit 0. Correct the fixture and rerun against the unchanged queue source before
claiming either regression evidence or acceptance.

## Evidence and choice

Reported use rates queued cancellation, queueing and reconnecting to work Often
(D4, D8, D10 in [preferences](design/preferences.md)). Source audit ranked:

1. Preserve queued cancellation receipts and rows across uncertain replies.
2. Guard watched Cancel/Pause replies against a successor watched job.

This slice takes the first. Watched controls in job-stream.js/job-progress.js
remain unchanged and are a separate follow-up. No new product control or layout
is introduced; all work is independent of the pending GAL12 repair and the
unapplied setup API/UI proposals.

Before this change, bulk Cancel queued called forgetJob after every request,
including an offline rejection, and removed all captured rows. forgetJob deletes
the accepted submission that GEN25's history recovery needs. A queued job that
finishes between polls can therefore lose its outcome. The row's x preserves an
offline receipt, but treats every other thrown HTTP/JSON error as success.

There is also an earlier loss window: cancelledHere is set before awaiting the
cancel reply, which makes unwatchedSubmission false. A status listing that omits
a now-terminal job can then forget it before cancellation later fails. The
adopter can drop its row at idle, and an old history reply arriving just after
failure can forget the newly eligible receipt. Initial source inspection found
these branches; the actual regression and qualification results are recorded below.

## Bounded repair

Use per-id owner tokens in queueCancelling. A row claims its id synchronously;
bulk claims every captured queued id before its first await, then sends the
existing serial sequence. Pressing an already-owned row or bulk control says its
hold reason and sends no duplicate POST. New submissions are outside the captured
batch. Each callback releases only its own token.

Keep existing decoded 2xx acknowledgement and explicit HTTP404 gone semantics.
Every other throw, including offline, HTTP refusal and JSON decoding failure,
withdraws cancelledHere attribution, retains the accepted request/row and surfaces
the error. Ordinary polling/history may subsequently show a known terminal
outcome once. No cancellation is automatically retried and no receipt is invented
for another tab's completed job.

Listing-derived queues retain owned pending rows and exclude the current or
incoming watched id. Server-list adoption defers pending cancellation ids while awaiting the
reply. Cancellation revision guards reject obsolete active/history answers and
Move queue-order snapshots;
status timestamps at or before the last transition are conservatively ignored.
Move error display also requires the captured cancellation revision, watched job
and known server context to remain current; its final release/refresh is unchanged.
A fresh ordinary poll can reconcile the queue afterwards. Removing an id from
its queue row because it became watched does not discard a still-owned cancel
receipt; ordinary watched terminal handling remains authoritative for cleanup.
This protects receipt ownership without changing the shared queue writer or
deliberately preserving old rows through a confirmed server restart.

Tokens bind to the known server start identity. Hidden old-server tokens do not
hold the new queue's bulk button, and a serial batch checks context before each
remaining POST as well as at settlement. Late old-context queued-cancel callbacks clean only
their token/attribution and do not change the new server's queue or error display.
The ownership is page-local; this introduces no durable cancellation journal,
request timeout or recovery guarantee across reload. Existing serial head-of-line
waiting is retained. An id can still become watched through its ordinary late
submission response; this packet does not rewrite startJob or watched Cancel/
Pause. Its self-row is omitted and the receipt stays until a confirmed queued
cancellation or ordinary watched terminal cleanup. This corner is source-reviewed;
the dedicated browser fixture need not invent a watched identity to claim coverage.

All row labels and fixed control slots stay the same. Held controls use
aria-disabled and title reasons; row removal/focus goes through setQueue. The
watched/front control and a row's Move hold for that exact cancellation owner;
other queue moves remain available, with old replies guarded by revision.

## Required qualification

The new dedicated owned fake/browser fixture checks the original failure and fixed
receipt lifetime: held cancel while active/history drops the terminal job; stale
history answered after failure; row/bulk duplicate presses; offline, HTTP500,
malformed JSON and404 outcomes; partial bulk success with a new submission; and
ordinary GEN25 once-only replay after uncertainty. Include a held old Move reply,
pending-to-watched exclusion and serial-batch server-context change where feasible.

Use fatal fixture prerequisites and count exact permitted mutations. Run relevant
existing queue/jobs/outcome smokes serially, classic-script syntax/shared-scope
checks, docs maps and no-cache Ruff. Keep any actual browser/access denial exact
and separate from source qualification. No real user gallery, credential, model,
GPU, service activation, trust/security setting or old artifact access is needed.
The results below record what the integration owner actually ran.

## Actual qualification (2026-10-28)

After the setup-only attempt above, the corrected fixture ran against unchanged
queue source and failed at step 17: an actual fake-server cancellation finished,
its HTTP reply stayed held, and a fresh active-list resync lost its queued row
and accepted receipt. The fatal prerequisite stopped subsequent steps. This
negative control completed with exit 1 in 13 seconds (browser 6 seconds).

With the independently reviewed source applied, all 90 steps passed in 19
seconds (browser 11 seconds). Seven accepted owned fake jobs, 26 explicitly
permitted mutations and 11 read-only POSTs completed with zero forbidden
requests, JavaScript errors or unused grants. The unmodified held history reply
was rejected after a newer cancellation transition; one fresh history read
recovered the real failed job exactly once. Bulk cancellation retained its failed
member and a newer submission, while confirmed success removed only its own
member. Delayed Move success and failure did not revive rows or overwrite the
current error. Malformed JSON preserved the receipt; an explicitly synthetic
missing id exercised the real 404 response. Final checks found no active jobs,
unsettled receipts or response gates, retained all planted/owned images and kept
the composer and default-off backup state unchanged.

The three new screenshots were read and visually inspected directly from
`auradream-smoke-artifacts-7153379x/smoke-queue-cancel-recovery-6atkh79s`:
`queue-cancel-pending-wide.png`, `queue-cancel-recovered-wide.png` and
`queue-cancel-partial-narrow.png`. Pending controls, recovered error/Finish
affordance, remaining rows and focus fit their wide/narrow layouts. These are
new permitted artifacts; no older denied artifact was accessed or recreated.

Existing queue controls passed in 37 seconds (browser 30); queue outcomes passed
in 16 seconds (browser 10). The job-lifecycle result and final static gates are
recorded below by the integration owner.

Classic-script/shared-scope and documentation-map checks: 35 passed in 5.19
seconds. No-cache Ruff passed. Pinned Pyright checked 146 backend/tools files
with zero errors. These checks use owned scratch data and the ordinary restored
Edge harness; they do not load models or qualify hardware, credentials, real
gallery recovery, WSL, compiler wheels or the separate unknown GAL12 run.

The existing queued-owner-to-watched attribution corner remains explicitly
outside this slice. The separate watched Cancel/Pause late-reply repair is next;
this result does not claim a shared durable cancellation journal or a timeout.

Job lifecycle: passed in 215 seconds (browser 211); no failed expectations.
