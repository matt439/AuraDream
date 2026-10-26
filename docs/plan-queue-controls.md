# Queue controls — GEN#18

Owner: [2026-10-24 review, G7](review-2026-10-24.md#g7-queue-move-to-front-and-skip-one-cell).

## Contract

- The manager owns an explicit ordered list of queued jobs. New submissions
  append; Move to front puts one queued job first without interrupting the
  running job, a paused job or an editor operation. `active()` and both HTTP
  job listings expose that same order, with the running job first.
- Loading stays outside the generation gate. Jobs may await the shared load
  concurrently, but the manager checks its admission order again after taking
  the gate; the lock's waiter order cannot decide which job generates next.
- `POST /api/jobs/{id}/front` moves a queued job. Moving the first queued job
  again is harmless. Unknown ids return 404; a job already running or finished
  returns 409. Its answer includes the authoritative `active_jobs` order.
- `POST /api/jobs/{id}/skip?call=N` requests a cooperative skip of the exact
  active cell call identified by `current_call` in its snapshot. The backend
  raises `Cancelled` at its next callback. The job keeps earlier images and
  continues to its next cell. A repeated request for that call is harmless;
  a stale call is 409 and must never skip the following cell.
- Intentional skips have their own count, details and retained outcome indices.
  They do not count toward the failure stop rule. They never enter
  `completed_indices`, so panel Resume and the server's Finish run see the
  absent images and can render them later. No reproduction key changes.
- Cancel wins over Skip. A skip arriving after the backend's final callback
  cannot discard a completed image or carry into the next call. Pause remains
  cooperative at the next cell boundary. Skip is held while queued, loading,
  paused, saving or otherwise between calls; there is no running cell there.
- Dynamic Then refine passes are not retained cells: Finish currently only
  compares the base cells. Skip is therefore held during that stage with a
  clear reason. Standalone Refine/inpaint/outpaint jobs remain ordinary cells
  and support Skip. Extending pass-only recovery is separate work.

## Implementation

Backend ownership: `backend/jobs.py`, `backend/api_jobs.py`, and focused tests
in the jobs/API jobs suites. The queue list and its change notification belong
to the event loop. The active call and skip token share a small thread lock;
the worker never awaits the loop to poll cancellation. Every terminal state
still goes through `_finish`, which also removes queued admission entries and
wakes waiters. Shutdown and cancellation during a load remain immediate.

Frontend ownership is separate: queue rows use `setQueue` only, keep focus on
the same job and action when reordered, and hold buttons with `aria-disabled`
and a reason. The Move to front answer supplies the authoritative order;
status polling and reload must retain it. Skip sends the last snapshot's call
token and treats stale/refused answers as a note. No automatic scrolling.

## Validation

CPU tests, using fakes only:

1. FIFO without a move; queued moves under a running job and under an editor
   operation; repeated moves; authoritative listing order; unknown/running/
   finished refusals; cancel/remove and shutdown wake the next waiter.
2. A move while the old front already awaits the gate still changes the next
   generated job. A delayed shared load neither permits overtaking nor makes
   a caller hold the gate while awaiting the load.
3. Skip stops one call, continues the next, retains earlier images, records
   the missing cell and allows Finish to recover it. More than three user
   skips do not trigger the failure policy.
4. Duplicate and stale skip tokens, cancel versus skip, late/decode/save-time
   requests, pause/between-call holds, a standalone refine call, and the Then
   refine hold. No request leaks into the following cell.
5. SSE full/compact snapshots and retained terminal outcomes expose the new
   fields consistently; existing load/cancel/pause/editor-gate tests remain.

Browser checks are run by the integration owner after backend wiring, using
an owned fake server and the existing browser harness. They cover actual next
job order, reload/another-tab order, focus and pointer stability, held reasons,
Skip acknowledgement and Finish recovery. No GPU, model download or image
quality claim is part of this feature.

## Status

Implemented 2026-10-25 and checked 2026-10-26. The queue review recorded 127
passing focused CPU tests, followed by 15 passing affected regressions after
the closure-binding lint fix; the integrated feature suites also reached the
storage tests with the queue/jobs checks passing. Ruff and changed-script
syntax checks pass.

After explicit local-browser test approval, `smoke-queue-controls`,
`smoke-jobs`, `smoke-results` and `smoke-narrow` passed serially on Chrome
and scratch fake servers. The queue smoke covers delayed reorder replies,
new submissions during a move, another tab's queued work, the watched job's
Front control, stable focus/geometry on admission, one-call Skip, held reasons
and Finish recovery. Adjacent assertions now use the click-to-open viewer
hint and Cancel queued's wording that includes other queued jobs. No real
model weights were loaded or downloaded.
