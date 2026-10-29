# Queue time estimates

**Status:** `GEN#32` done, 2026-10-30: stages 1 and 2 built together (see
[Result](#result-2026-10-30)). Planned from the 2026-10-29 TODO audit (its
item 3). Both stages are CPU, fake-server and browser work; neither needs the
card, and nothing here claims the history figure's accuracy on it.

## The ask and the evidence

Matt rates these as Often in [reported use](design/preferences.md):
watching progress, elapsed time and estimates (`D1`), queueing work while
another job runs (`D4`), Batch (`A12`) and Repeat (`A13`). Batch and Repeat
are how a plain Generate grows into minutes of work. While that work waits in
the queue, the page says nothing about how long it will take:

- A queued row is the job's description and its two buttons, nothing else
  (`web/job-queue.js:139-177`, `queueRow`). The line above several rows reads
  "N jobs queued behind this one:" (`web/job-queue.js:113-118`,
  `syncQueueNote`).
- The only time estimate on the page belongs to the watched job, and that job
  must be running. `showCell` measures it from the job's own pace
  (`web/job-progress.js:201-216`). It appears only after a second progress
  event (`etaFrom`, `web/job-progress.js:139-147`;
  [design notes](design-notes.md#grid-preview-and-the-per-step-estimate) and
  the progress section give the reason: extrapolating from the first step said
  28 s for a 9.5 s pass).
- A plain Generate has no preview, unlike Run grid, so nothing prices it
  before or after the press. `POST /api/generate` answers only `job_id`,
  `seed` and `run_id` (`backend/api_generate.py:421-440`, `JobAccepted` at
  `backend/schemas_generate.py:950-956`).

The pricing already exists and is shared. `gallery_reports.execution_estimates`
(`backend/gallery_reports.py:135-162`) reads matching history for each canvas
size. `grid.estimate_seconds` (`backend/grid.py:912-933`) sums the prices, or
returns None when any cell has no matching history. The grid preview uses both
(`backend/api_generate.py:613-668`), and so does the CLI
(`backend/generate.py:1808-1809`). More seeds (`backend/api_more_seeds.py:230`)
and Mass refine (`backend/api_refine_batch.py:158`) price their previews through
`estimate_seconds` too. Rule: no matching history means no estimate
([design notes](design-notes.md#grid-preview-and-the-per-step-estimate)).

## Design

### Priced once, at acceptance

`JobManager.submit_cells` (`backend/jobs.py:806`) prices every job, whatever
sent it: Generate, Run grid, Re-render, Swap tags, Mass refine, More seeds,
Finish run, refine, inpaint, outpaint. `submit` (`backend/jobs.py:784`) passes
through it as well. The price is stored on the `Job` in a new field,
`estimate_seconds: float | None`. Constraints:

- **The cells the worker runs.** Pricing reads the job's own cell list after
  execution is resolved and outpaints are stamped
  (`backend/jobs.py:854-873`), so the sizes and step counts are exactly what
  will run. Then refine adds each base cell's stand-in pass, in run order, the
  same way the grid preview does (`backend/api_generate.py:613-621`,
  `then_refine.stand_in`). That expansion moves into one helper that both the
  preview and `submit_cells` call, so the arithmetic is not written twice
  (`grid.estimate_seconds`'s own rule).
- **Off the event loop.** The history queries and the median run in
  `asyncio.to_thread`, the way the preview already runs
  `execution_estimates`. The queue needs only the time, so the call must not
  `stat()` sample PNGs for the disk figure (`image_bytes`,
  `backend/gallery_reports.py:104-132`). Either add a time-only parameter or
  split the function, and keep the preview's behaviour unchanged.
- **Never refuses a job.** If a database error, a busy gallery or any
  exception occurs while pricing, the job gets `estimate_seconds = None` and a
  log line, and it is queued as before. The estimate is presentation and may
  not create a new reason for a submission to fail.
- **The same rules as the preview.** History is matched on exact size and the
  selected checkpoint. Refine and inpaint passes are priced from their own
  histories. An upscaler pass returns None (`backend/grid.py:950-951`), and so
  does an instruction edit (`backend/gallery_reports.py:147-150`). One cell
  without history makes the whole job None; nothing is borrowed from another
  size.
- **Not repriced.** Moving a job to the front, a model switch (refused while a
  job is live), or new images added to the gallery do not change a stored
  estimate. Rows show the price the job had when it was accepted.
- **Not a reproduction field.** `estimate_seconds` stays out of `CellKey`,
  `run_id`, `applied()`, the images row, the PNG `auradream` chunk and bundles,
  the same as the peak-memory pair (AGENTS.md, "Reproduction and storage").
  It is not part of retained outcomes, so a restarted server has no estimates
  for its jobs.

### Reported, never computed, by the snapshot

`Job.snapshot()` (`backend/jobs.py:341`) adds `"estimate_seconds": self.estimate_seconds`.
`snapshot()` runs on the loop for every listing, attach and per-cell status
frame, so it reads only the stored value: no database read, no file access.
`JobAccepted` gains an optional `estimate_seconds`, so the page has the figure
for its own submission when the reply arrives. The field is additive and old
pages ignore it. Each route that builds a `JobAccepted` or one of its
subclasses (`RerenderAccepted`, `FinishRunAccepted`,
`backend/schemas_rerun.py:65,119`) copies the figure from the job. So do the
separate answers of Swap tags, Mass refine and More seeds
(`SwapTagsAccepted`, `RefineBatchAccepted`, `MoreSeedsAccepted`,
`backend/schemas_rerun.py:422-443`).

### The page

The server sets the price. The page formats and sums it, and adds nothing of
its own except the watched job's live ETA, which it already computes.

- **Per row.** A priced queued row starts with `~` and the figure formatted
  by `humanTime` (`web/grid-preview.js:51-56`: `40s` under 90 s, then
  `3 min`, then `1h 05m`), before the job's description:
  `~3 min · 4 images · 1024×1024 · prompt…`, or `Next: ~3 min · …` when the
  job is alone. A budgeted grid whose deadline (already in the snapshot,
  `backend/jobs.py:425`) comes sooner shows `~N min (budget)`, matching the
  running job's existing wording. An unpriced row adds no text. Its text span's
  title says "No time estimate: nothing has run at this size on this
  checkpoint here".
- **One write per row.** The estimate goes into the row in the same text
  write as the label. For the page's own jobs the reply already carries it
  (Swap tags' and Mass refine's extra jobs in `more`, a list of
  `JobAccepted`), so the row has it from its first render. The label writes
  stay the ones there are now: `submissionLabel` when the reply lands, then
  `nameJob`'s rename from the server's snapshot
  (`web/generation.js:412-417` and `:462`, `web/job-queue.js:5-8`), each
  carrying the same figure. For another tab's
  jobs, the listing snapshot that names them carries it
  (`web/job-queue.js:358`). An estimate must never arrive later as a second
  change to a row that the pointer may be over (AGENTS.md: never move a
  control under the pointer). `setQueue` remains the only writer of the rows.
- **The total.** When more than one job is queued, the existing `#queue-text`
  line carries the total. No line is added:
  - every row priced and the watched job running with a live ETA:
    `3 jobs queued behind this one · all done in ~14 min:`. The figure is the
    live ETA plus the rows' sum.
  - every row priced, with no live ETA (the first step has not happened, the
    weights are loading, the watched job or the job ahead is paused, or the
    watched job is itself queued): `3 jobs queued behind this one · ~12 min of
    queued work:`. A pause holds the gate for as long as it lasts, so
    "all done in" would be a promise the page cannot keep.
  - any row unpriced: `3 jobs queued behind this one · no estimate:`, with a
    title naming how many queued jobs have no history. A partial sum is never
    shown as the total.
  - The weights load is never priced. While `waiting_on` is `load`, the
    "of queued work" form applies.
- **No layout movement.** The total is rewritten only when its formatted
  text changes, at the redraws that already rewrite `#queue-text`
  (`syncQueueNote`), never on a timer of its own. At 1400×900, the narrowest panel the smokes use,
  `#queue-text` must stay one line with the longest figure (`1h 05m`). If it
  does not fit, the implementation drops "behind this one" whenever a total is
  shown. It must not add a line or move a row. No CSS changes under 1800 px
  (AGENTS.md, Frontend).
- **Holds stay as they are.** The rows' × and ↑ keep their `aria-disabled`
  reasons (`web/job-queue.js:155-170`). The estimate is text, not a control.

### Stage 2: a run hint after a press

Most presses are a plain Generate, which has no preview. The panel has no
spare height for a pre-press line (rows under Generate are full at 1400×900),
so the hint goes only into lines that already exist:

- Acceptance: when the reply returns while other work is ahead, the page
  today writes no visible line; the row appears, and the hidden live line
  `#job-said` says `Queued: <label>` (`sayJob`, `web/generation.js:257-259`,
  `:470`). Visibly, the row's own `~N` (stage 1) is the hint. The live line
  gains `· ~3 min, after ~11 min of work ahead`, so a screen reader hears
  it too. With no estimate it says `Queued: <label>`, as now. The panel's
  note line (`showJobNote`, `web/job-results.js:49-52`) is not used: it is
  `showError`'s line, for what went wrong or was refused.
- Before the live ETA: from the first progress event until `etaFrom` can
  measure a pace (`web/job-progress.js:139-147`), the cell line shows
  `~3 min expected` from the stored estimate. It is labelled as history and
  replaced by the job's own-pace `~N left` once that exists. The extrapolation
  rule that holds the live ETA back does not change, and the history figure is
  never mixed into it.

## Non-goals

- No pre-press estimate route for the plain Generate in either stage. A
  priced Generate preview would need its own debounced request and a place on
  the panel that does not exist; it gets its own row if Matt asks for it.
- No repricing, and no estimate learned from the running job's pace and
  applied to the jobs behind it.
- No pricing for the weights load, for instruction edits or for upscaler
  passes. All stay "no estimate" until they have measured history.
- No change to the grid preview's figures, the time budget's stop rule, the
  live ETA's formula or the CLI's estimate line.
- No estimate in `generate.py`'s queue (it has none) and no change to the
  retained-outcome format.
- No claim about accuracy on the card. A history median is what it is, and
  `docs/verified.md` is not touched by this work.

## Stages and completion criteria

### Stage 1: stored estimate, row values and the queue total

Backend (`backend/jobs.py`, `backend/gallery_reports.py`, `backend/grid.py`,
`backend/schemas_generate.py`, the accepting routes):

1. `pytest`: a job submitted with planted timed rows (`duration_ms`, as
   `tests/test_api_grid.py` plants them) stores the same `estimate_seconds`
   that `POST /api/grid/preview` returns for the same cells. This holds for
   Generate with Batch and Repeat, a grid, Then refine (base plus passes),
   a refine pass, an inpaint and an outpaint. A Mass refine and a More seeds
   job each store the figure that their own preview returns.
2. `pytest`: one cell at an unplanted size, an upscaler pass or an edit gives
   `None`. A pricing call that raises still queues the job, with `None`.
3. `pytest`: with `storage._session` and the `gallery_reports` readers
   patched to raise, `snapshot()` and `GET /api/jobs` still answer and carry
   the stored figure. This proves the snapshot does no database read.
4. `pytest`: `run_id`, cell keys and `applied()` are unchanged for a job with
   and without an estimate, and the field is not on saved rows or in the PNG
   chunk.
5. `ruff`, the pinned type gate and the full suite pass.

Frontend (`web/job-queue.js`, `web/generation.js`):

6. A new smoke, `smoke-queue-estimates`, on a fake with `--step 0.3` and fresh
   scratch `--data`. The implementing commit adds its row to the AGENTS.md
   smoke table and to `docs/rules.md` under Tests. It renders one image to
   create history, then queues three jobs behind a running one: two at the
   rendered size and one at a size never rendered. It checks the priced rows'
   `~N` values against `GET /api/jobs`, the unpriced row's absent value and
   its title, `no estimate` in `#queue-text`, and, after cancelling the
   unpriced job, the summed total in both forms ("of queued work" before the
   live ETA, "all done in" after). It also asserts that `#queue-text` is one
   line and that no row's `getBoundingClientRect` changes between the row
   being rendered and the total updating.
7. The existing `smoke-jobs`, `smoke-queue-controls`,
   `smoke-queue-cancel-recovery` and `smoke-split-extra-queue` pass unchanged.
   This shows the row text change kept their focus, order and cancel
   assertions.

### Stage 2: the run hint

8. `smoke-queue-estimates` gains the press: for a Generate accepted behind
   running work, `#job-said` reads `Queued: <label> · ~N, after ~M of work
   ahead`. For an unpriced one it reads `Queued: <label>`, unchanged. With `--step 0.3`, the cell line shows `~N expected` before
   the second progress event and the own-pace `~N left` after it.
9. `smoke-preview` and `smoke-results` still pass, so the live ETA's timing
   and wording are unchanged.

Neither stage needs the card or Matt's OK to run. How accurate the history
figure is on the card is a later observation, and this work does not
establish it.

## Result (2026-10-30)

Built as designed, both stages in one commit.

- Backend: `gallery_reports.execution_estimates(..., disk=False)` skips the
  PNG sample, and `gallery_reports.job_seconds` prices execution units with
  it through `grid.estimate_seconds`. `then_refine.execution_units` is the
  one Then refine expansion (base cell, then a stand-in pass per image), now
  used by the grid preview, the CLI's estimate line and the queue.
  `JobManager._price` runs in `asyncio.to_thread` after outpaints are
  stamped and before masks are held or the run is retained. Any exception
  is logged, gives `None` and still queues the job. `Job.estimate_seconds`
  is reported by `snapshot()` and copied into every accepting answer
  (`JobAccepted` and its subclasses, plus the top level and `more` of Swap
  tags', Mass refine's and More seeds' answers).
- Page: `jobEstimates` (state.js) holds each job's `{seconds, budget}` from
  the reply or the naming snapshot. `queueRow` writes `~N · <label>` in its
  one text write, and `syncQueueTotal` writes `#queue-text`. `queuedHint`
  adds the figure and the work ahead to `#job-said`, and `showCell` shows
  `~N expected` until the own-pace ETA exists.
- Tests: `tests/test_api_queue_estimates.py` covers criteria 1-4 (Generate
  with Batch and Repeat, a grid, Then refine, a refine pass, an inpaint, an
  outpaint, Mass refine, More seeds, Swap tags; the unpriced cases; a
  raising pricer; the snapshot with every gallery reader refused; run id,
  cell keys, `applied()`, row columns and the PNG chunk unchanged).
  `smoke-queue-estimates` (7954) covers criteria 6 and 8.
  `smoke-queue-controls`, `smoke-queue-cancel-recovery`,
  `smoke-split-extra-queue` and `smoke-results` pass unchanged. `smoke-jobs`
  and `smoke-preview` failed at first, because their fakes have 1024×1024
  history, so their rows are now priced. Their exact-text checks of a row,
  the `Queued:` line and `#queue-text` now read past the figure. Their
  focus, order, cancel and live-ETA checks are unchanged, and both pass
  (criteria 7 and 9). The review found `smoke-newbie` failing the same way
  (its queued rows compared with their Cancel labels); its rows are read
  past the figure too, and the whole smoke table passes.

Choices the plan left open, taken while Matt was away (2026-10-29), revisit
if wanted:

- **A queued budgeted grid's cap** is the budget's whole length
  (`deadline - created_at` from a snapshot, or the submitted
  `budget_minutes`), not the time left: a queued job has no start time yet,
  and a cap that shrank while it waited would rewrite the row on a clock.
- **The total follows the live ETA** at the watched job's own progress
  redraws (`showCell` calls `syncQueueTotal`), not only at queue redraws.
  The ETA it adds to changes there. The text is still written only when
  its words change, and never on a timer.
- **The longest total fits.** At 1400×900 both `… · all done in ~1h 05m:`
  and `… · ~1h 05m of queued work:` stay one line, so "behind this one"
  is kept. The smoke checks both strings.
- **`#job-said` when the work ahead is not fully priced** (an unpriced row
  ahead, or no live ETA yet) says `Queued: <label> · ~N`, without the
  "after" clause, rather than a partial sum.
- **The history figure does not return** for a job once its own-pace
  `~N left` has shown, including at the last cell, where nothing is left to
  extrapolate (`paceShown`).
- **The unpriced titles:** the row's is the plan's ("No time estimate:
  nothing has run at this size on this checkpoint here"). The total's
  counts them ("1 queued job has no time estimate: nothing like it has run
  on this checkpoint here"), because an upscaler pass or an edit is
  unpriced at any size.
- **Mass refine's and More seeds' previews** sum per-size figures that are
  rounded separately. A job is one size (Mass refine) or one run (More
  seeds), so each job's stored figure equals its preview's for a
  single-job selection. A multi-job preview's total can differ from the
  rows' sum by a rounding tenth.
