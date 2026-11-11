# Review and plan: watched action replies

**Status:** `GEN#29` implemented and locally qualified, 2026-10-28.
Source proposals `89216290`/`3e48dd27` and fixture `12f20db6` passed
the recorded focused and regression checks. No hardware or release acceptance follows.

## Evidence and choice

This follows queued cancellation recovery without expanding that slice. The
remaining independent defect is in the watched controls: `cancelJob` writes
progress or an error and re-enables Cancel after its awaited POST without
checking which job the page now watches. `togglePause` restores the label from
before a failed POST, then redraws progress and mirrors after every settlement.
An older pause failure can therefore undo a newer Pause/Resume press on the
same job as well as overwrite a successor job's control state.

The existing Skip path already guards its exact watched job/call and releases
only its own pending object. This bounded follow-up applies that ownership
principle to Cancel and Pause. It is preferable to a broad progress-state
rewrite: the source defect is concrete, the action semantics stay unchanged,
and it is independent of gallery ordering, setup import or device qualification.
No competitor feature claim or new product decision is needed.

## Bounded repair

`watchedActionCurrent` requires a busy watched job with the captured id and the
server start identity known when the request began. Learning the first server
identity is allowed; a known restart invalidates the old context. It is a
response guard rather than server-side cancellation of an in-flight request.

Cancel keeps per-id request objects while each POST is pending. A decoded
successful reply writes progress only for the matching owner/context. A failure
cleans only its own cancellation attribution, then restores controls and says
the error only when that context is still current. `finally` releases only that
exact object. A successor job remains independently actionable; even a same-id
job after a known restart cannot have its newer attribution removed by the old
request settling after the newer one.

Pause retains its existing immediate label and withdraw/resume semantics.
Only its exact latest request object and watched/server context may roll back a
failed optimistic label. Only the matching object clears pending state, and
only its still-current context redraws progress/mirrors. The status-frame hold
also checks server context, so an old request cannot suppress a new server's
snapshot solely because its id happens to match.

The change introduces no automatic retry, deadline, durable journal, reload
guarantee, backend action or new control. Ordinary stream frames remain the
worker's authority. The separate queued-owner-to-watched edge is unchanged;
this packet does not add another shared-control ownership system or alter
`startJob`, queue adoption, watched completion, Skip or layout.

## Required qualification

`tests/test_watched_control_replies.py` supplements the browser fixture.
It evaluates the actual two classic source files in a Node VM with inert
presentation ports and deferred API promises. Nineteen named cases cover
current success/failure, duplicate Cancel refusal, late Cancel success/error
after a successor or idle transition, known restarts, first identity discovery,
per-id attribution after a newer request settles, successor Pause, same-job
newer Pause pending/settled, context-aware status-frame holds and held/idle
refusal. It does not execute the page boot or claim DOM/browser acceptance.

Before applying source, run the meaningful failing cases against the original
source, then apply the proposal and run the focused tests plus classic-script
syntax/shared-scope checks. Keep the existing no-cache Ruff and pinned type
gates. No full suite or GPU qualification is required by this source slice.

A separate fresh-data browser fixture must still use actual owned fake jobs
and held exact action replies. Prove an old Cancel success and refusal after
the next real watched job begins; include a newer Cancel still pending so the
old catch cannot re-enable it. Prove old Pause refusal after a successor and
after a newer same-job withdraw/resume request. Count exact POSTs and use fatal
preconditions. Confirm current controls remain usable, unchanged labels, stable
focus, fresh-job progress and the ordinary current-action error path. New
screenshots, if taken, qualify only that run; do not read denied old artifacts.

## Actual qualification (2026-10-28)

The original-source VM baseline failed 12 of 19 named cases in 0.99 seconds;
the seven ordinary current-action cases passed. Failures covered obsolete Cancel
success/error after a successor, idle or known restart; newer same-id ownership;
and Pause successor/newer-request/restart/status-frame races. The actual browser
baseline independently failed at step 11: after A really finished cancelling and
B became watched and physically paused, releasing A's real successful reply
wrote "finishing this step…" into B's progress. The fatal check stopped later
steps. That run completed in 15 seconds (browser 9) with a normal browser exit.

With the exact reviewed source applied, all 54 focused VM, classic-script
syntax/shared-scope and docs-map checks passed in 5.10 seconds. Independent
static review found no blocking source/test issue; the design note scopes its
attribution wording to watched controls and preserves the separate shared-owner
corner described above.

The first positive browser attempt passed every assertion through step 20,
then stopped at step 21 because the fixture supplied an object where the driver
requires a viewport array. That partial run is unqualified. Only steps 21 and
24 were corrected to the documented arrays; boot and assertions stayed exact.
Its diagnostic printed the browser as running before synchronous exit cleanup
and retained-profile cleanup reported a warning. A read-only probe of only that
run's exact known DevTools endpoint confirmed it was already closed; no extra
close request, process inventory or profile deletion was performed. The retained
profile was left to the existing harness policy. This was a fixture TypeError,
not a newly observed access denial.

The corrected 27-step fresh-data run passed in 22 seconds (browser 14). Three
real owned jobs, 15 explicitly permitted mutation attempts and 10 forwarded
requests covered old Cancel success/error and Pause error after actual successor
adoption, an independently pending newer Cancel, current-request failures, and
old Resume failure while a newer same-job Pause owner remained pending. Final
checks found exactly the three terminal owned jobs, one stream/outcome per job,
no active jobs or pending control/response owners, zero forbidden mutations,
JavaScript errors or unused grants, unchanged planted image facts and composer,
an empty bin and default-off backup/credential state. Controlled HTTP500 replies
were never forwarded. The declared pre-start queued Pause is setup evidence;
it is not claimed as browser Pause acceptance.

The new `watched-action-pending-wide.png`, `watched-action-recovered-wide.png`
and `watched-action-pending-narrow.png` were read and visually inspected from
`auradream-smoke-artifacts-u_vkzb6l/smoke-watched-action-replies-cyuxu6jj`.
Pending/recovered control labels and current errors fit at both widths. The
focused fixture does not assert dynamic focus preservation; the existing job
smoke's ordinary focus cases remain separate evidence. No old denied screenshot
or artifact was read or recreated.

No-cache Ruff passed; pinned Pyright checked 146 backend/tools files with zero
errors or warnings. The existing regression results are recorded below. No
model weights, GPU run, real gallery, credentials, server restart, WSL or
compiler-wheel qualification was performed. GAL12 remains uncommitted with
its previous unknown result; its source and evidence were preserved.

`smoke-jobs`: passed in 215 seconds (browser 211); no failed expectations.

`smoke-queue-controls`: passed in 36 seconds (browser 30); no failed expectations.
