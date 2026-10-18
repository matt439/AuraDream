# Plan: continuing a run

**Status, 2026-10-18: Matt took every recommended answer to the open questions the same day (below). Phase 1 is built, 2026-10-18: 1a (the Resume double-render fix, `8097f1d`) and 1b (the `run` scope, `9c01d64`; the page, `c230b6e`); phase 2 persistence is built; Finish service/page integration remains.** Why it is built as it is is in `docs/design-notes.md`'s "More seeds where the run ends"; the rules in `docs/rules.md`. Where the build differs from the text below: the planned-keys drop is a function of its own, `submission.not_yet_planned`, chained after `not_yet_rendered` on Resume, rather than an argument to it; `count_default` is the Batch every batch that recorded one agrees on, answered for every scope (the finish line's open is an `{ids}` one), and the page asks again at it, one more preview; the headline reads "N more of each of C cells = T images" ("of its one cell" for one); the run bar's line is "Run X · N images · checkpoint quant", the images the gallery holds under the filter, since phase 1 stores no cell total for "70/96 cells"; the lightbox's More seeds… is hidden, not held, at 900 px wide or narrower, where the right-click menu has it, as `#lb-inpaint` is; and the lightbox's keys are ignored while the dialog is open over it, so Escape closes the dialog alone. Phase 2 persistence now keeps pre-stamp cells and budgets before admission, writes terminal outcomes, and excludes recovery definitions from bundles. The service/page stages below remain the design as written. It was written from four read-only reports on
the current code (More seeds, grid Resume, the places a finished run shows,
and the usage evidence), a user-research pass, three design proposals and
two judges' reviews of them (one on usability, one on engineering). Every
statement about today's behaviour cites the file and line it was read
from; claims a judge found false were dropped or corrected, and each
correction is noted where it applies. A later adversarial pass re-opened
every cite against the code (2026-10-18); what it changed is marked
"Corrected" or folded into the text. What only the card can say (timing,
memory) is out of scope: nothing here changes a pixel, a seed or a
memory cap.

The recommended design, in one paragraph: **phase 1** puts the two
continuations the server already runs - *More seeds* (new seeds, same
settings, joining the run) and *Resume the remaining N cells* (the missing
cells of a grid) - at every place a finished run is shown, adds a `run`
scope to `POST /api/more-seeds` so a run can be extended whole without the
gallery filter, gives a budget stop its button, and fixes a latent
double-render in Resume. **Phase 2** stores each job's resolved cells in the
database so a run can be finished after a reload or a server restart
without the panel holding its grid or Matt knowing its base seed. Widening
an axis and "load the run into the panel" wait on Matt's answers.

## Problem statement

Matt's ask, verbatim:

> sometimes a run completes and i want to continue it. at the moment i dont
> think theres a simple way to do it.

Matt asked for a formal design process: personas, user stories, scenarios,
requirements, options and their evaluation. This document is that process
in order, ending in a recommended design and the questions only Matt can
answer.

The closest earlier case is recorded in the design notes as "Do four more
of these" (2026-10-10): a Batch 4 that turned out well wanted four more like
it, which meant Reuse, add 4 to the seed by hand and Generate, and a seed
added wrong re-rendered an image the gallery already had or skipped four
(`docs/design-notes.md:5119-5122`). More seeds was built for it. The ask
today is that it is still not *simple* from where a run ends.

## Evidence and its limits

- There is one user, Matt. No interviews were held; the evidence is Matt's
  dated decisions in the design notes and `TODO.md`, the two reviews
  (`docs/review-2026-10-04-ui.md`, `docs/review-2026-10-12.md`),
  `docs/verified.md`, and the saved settings.
- The live gallery is empty (0 rows in `local-diffusion.db`, no PNGs in
  `outputs/`). The three `bundles/*.ldbundle` (2026-10-15, -17, -18) are
  behind the password layer (`backend/bundle_crypto.py`). So there are no
  numbers for batch sizes, grid shapes or how often runs are extended.
- Some quotes are reviewers', not Matt's ("Seeds are the commonest sweep",
  `docs/review-2026-10-04-ui.md:2447`; "Same seed, tweak the prompt",
  `:2420`). They are marked and weigh less.
- Matt bundles and clears every one to two days, so a run older than the
  last clear is not in the live gallery at all.

## Personas

These are modes of one person, not separate people; Matt moves between
them in a session.

### P1. The Night Runner

Before bed, Matt queues a large grid or sweep with a time budget and judges
it in the morning.

- **Goals:** every parameter combination for as many seeds as the night
  allows; nothing lost when the night ends early; the rest finished in one
  press; a good sweep widened without re-rendering what exists.
- **Context:** seeds are the outer loop so that "A night that ends early -
  a cancel, a crash, a power cut - has then finished every parameter
  combination for the seeds it reached" (`docs/design-notes.md:2802-2808`).
  The budget means "start no cell after", and what was not reached "is a
  resume, not a loss" (`backend/jobs.py:912-917`). The card is shared with
  Unreal Editor and the browsers (`docs/verified.md:5-8`).
- **Frustrations today:** a budget stop gets words and no button
  (`web/job-results.js:223-224`; `remaining` counts only the failure rule,
  `:142`). A job this page watched to its end never gets the banner
  (`web/job-stream.js:287-289`). A complete grid has no continue action, and
  the sheet's Resume this run on it ends in a 409 "every cell is already
  rendered" (`backend/api_generate.py:143-146`). Resume needs the panel to
  still hold the grid - "The grid's axes are not stored per run"
  (`web/sheet-render.js:580-583`).

### P2. The Batch Iterator

At the desk, mostly on anime SDXL (WAI Illustrious v17 is the saved
default), Matt generates a Batch of about 4, judges it, and goes again.

- **Goals:** when a batch is good, N more like it, with seeds that neither
  repeat nor skip; the batch's own Batch kept ("keep batch settings from the
  original run", Matt's words, `docs/design-notes.md:5158-5161`); staying
  where the results are.
- **Context:** Matt chose that N counts images per batch
  (`docs/design-notes.md:5137-5140`) and that new images join the run
  (`:5171`).
- **Frustrations today:** the finish line of a plain job offers only Use
  seed (`web/job-results.js:121-123`). The results strip and lightbox have
  no More seeds, and the single-image menu has none: More seeds is offered
  only on a picked card (`web/context-menu.js:192-193`, the picked menu's
  item at `:143`; `imageItems` at `:84-128` has none). The footer button
  is held until something is picked or filtered
  (`web/gallery-select.js:355-359`). A remembered count of 64 replaces the
  default 4 (`web/more-seeds.js:142-143`, `:379`).

### P3. The Curator

The morning after, Matt narrows a night on the sheet and in the lightbox,
stars, refines or inpaints keepers, then bundles and clears.

- **Goals:** ask for more around a keeper while looking at it; keep a run's
  images together as one set; export what is worth keeping.
- **Context:** "three cells of a sweep, a few keepers across two nights.
  The user chose click-to-select cards" (`docs/design-notes.md:5952-5956`).
- **Frustrations today:** the lightbox has no More seeds; its Run row "⊞
  Show this run" closes the lightbox and opens `run:` (`web/facts.js:125-133`),
  after which the footer's More seeds acts on every batch of the run, not the
  keeper's. Under Starred the `run:` filter takes only starred images, so
  only starred batches are extended (`web/gallery-select.js:364-368`).

A CLI persona was considered and dropped: `generate.py --resume` exists, but
nothing shows Matt continuing runs from the CLI. It is a parity surface.

## What "continue" means

No control or doc names an action "continue" today; Matt's word maps onto
Resume (five controls), More seeds and Re-render. Seven meanings were found:

| # | Meaning | Exists today as | Persona | Evidence |
|---|---|---|---|---|
| M1 | N more of the same batch (new seeds) | More seeds… | P2, P3 | Strong: Matt's 2026-10-10 case |
| M2 | More seeds on a finished grid | More seeds over `run:` (N per cell, joins the run); or panel grid + larger Seeds + Resume (new run id) | P1, P3 | Medium |
| M3 | More values on another axis | Panel only: edit the axis, keep the base Seed, tick Resume (new run id) | P1 | Weak: none from Matt |
| M4 | Finish an interrupted run (budget, cancel, crash, restart) | Banner Resume, sheet Resume this run, grid tick; finish line only for a failure-rule stop | P1 | Strong: the "morning after" design |
| M5 | Retry failed or skipped cells | Finish-line Retry, `/api/grid` only, page memory | P1 | Medium |
| M6 | Finish a stopped plain Repeat-N | None | P2 | Weak: inferred |
| M7 | Continue under a changed setting | Reuse + edit + Generate; Re-render; Swap tags | P2, P3 | Weak as "continue" |

"A run **completes** and i want to continue it" points at a finished run:
M1 and M2 are the likeliest meanings. M4 and M5 are runs that did *not*
complete, but they share the entry points and the vocabulary, so the
design covers them.

**Vocabulary today** (the root of US-4): "Resume" names five controls - the
Pause button's Resume inside a live job (`web/job-progress.js:287`), the
grid's Resume tick (`web/index.html:632`), the banner's Resume
(`web/index.html:529-530`), the finish line's "Resume the remaining N cells"
(`web/job-results.js:163`) and the sheet's "Resume this run"
(`web/sheet-render.js:244`). "Run" means a
definition hash, the Repeat count's internal name, a cell counter and a
job banner. "Batch" means the in-VRAM count, More seeds' seed group, and a
repeat unit.

## User stories

MoSCoW for this round. Criteria are Given/When/Then and solution-neutral.

### Must

**US-1 (P2). More of this batch from where I'm looking.** N more of a
just-finished batch from the finish line, a result tile or the lightbox,
without picking or filtering first.
- AC1: after a plain Batch 4 finishes, "more" from the finish line queues 4
  more with at most one confirmation and no pick or search.
- AC2: from a tile or the lightbox, the scope is that image's batch
  (`more_seeds.batch_key`, `backend/more_seeds.py:71`), not the run or the
  filter.
- AC3: seeds follow the batch's highest held seed and step over held and
  planned seeds, server-chosen, as today.
- AC4: when unavailable, the control is `aria-disabled` with the reason as
  its title, and a press says why.
- AC5: the images join the original run (`backend/api_more_seeds.py:244-250`).

**US-2 (P1). Finish a budget-stopped night with one press.**
- AC1: a grid stopped by its budget, watched on this page, gets a one-press
  Resume on the finish line.
- AC2: only the missing cells are queued, under the run's id.
- AC3: an edited panel does not change what is resumed.

**US-3 (P1). Extend a completed grid's seeds and keep it one run.**
- AC1: a `done` grid's sheet and finish line offer "more seeds".
- AC2: the dialog states images per cell and in total, and the 1024 cap,
  before a press.
- AC3: new images carry the original run id; nothing held is re-rendered.
- AC4: Starred does not quietly narrow the scope.

**US-4 (all). One clear name per meaning.** "Unpause", "finish the
missing cells" and "make N more" each have one label on the run surfaces,
and each is described once in `docs/usage.md`.

### Should

- **US-5 (P3).** More of a keeper's batch from the lightbox and the
  single-image menu; the lightbox stays open; no existing control moves.
- **US-6 (P2).** A sensible default count (the batch's own Batch), not a
  value remembered from last week.
- **US-7 (P1).** Finishing a run survives a reload, another tab and a server
  restart.
- **US-8 (P1).** Continuing a grid never depends on Matt knowing its base
  seed.
- **US-9 (P2).** Picks after queueing are cleared or visibly kept.

### Could

- **US-10 (P1).** Add values to a finished grid's axis, kept in the run.
- **US-11 (P2).** Finish a cancelled plain Repeat-N.
- **US-12 (P3).** Seeds to grid (`TODO.md:90`, UI#3).

### Won't (this round)

- **US-13.** Redraw a binned cell: binned rows count as rendered until Empty
  bin.
- **US-14.** Continue across a checkpoint switch: refused per batch as
  `foreign` (`backend/api_more_seeds.py:41-56`).
- **US-15.** Widen a prompt comparison's variant axis.

## Current-state journeys

Counts are presses at 1400x900, assuming the remembered More-seeds count is
already right.

### J1. A plain Batch 4 just finished: four more (P2, M1)

1. Finish line: "Finished 4 images in T, seeds a–b." and **Use seed**
   (`web/job-results.js:121-123`). Use seed puts the first image's seed in
   the Seed box and writes nothing else; the next Generate is a new
   submission.
2. Results strip and lightbox: no More seeds (`web/viewer.js:290-314`,
   `web/index.html:977-999`); a tile cannot be picked
   (`web/context-menu.js:192`).
3. Shortest route: Ctrl+click a card (hinted only in the Select button's
   title, `web/index.html:837`), right-click it → More seeds…, wait for the
   preview, Enter. **3 actions plus discovery**; picks stay afterwards
   (`web/more-seeds.js:376-381`).
4. The old trap still works: Reuse puts the image's own seed in the box
   (`web/reuse.js:18`), and Generate renders seeds the gallery already
   holds with no warning - `/api/generate` never asks the gallery
   (`backend/api_generate.py:117-132`). (Corrected: it is not always "a new
   run": with the batch's first seed and the same Batch the cells, and so
   `grid.run_id`, are the original's, `backend/jobs.py:630`.)

### J2. An overnight grid finished every cell; widen its seeds (P1, M2)

1. Finish line: Show sheet only; Retry/Resume need skipped or remaining
   cells (`web/job-results.js:139-143`).
2. After a reload, the banner offers no Resume for `done`
   (`web/runs.js:98-100`), and none at all for a job this page watched.
3. Sheet → Resume this run looks right but previews 0 images; Run grid
   then answers 409 (`backend/api_generate.py:143-146`).
4. Sheet → footer More seeds… works: N per cell, joins the run, but cells ×
   N over 1024 is refused and Starred narrows it silently.
5. Panel route: raise Seeds, keep the base Seed, tick Resume - the new
   cells get a new run id, because `grid.run_id` hashes the full grid
   (`backend/grid.py:883-921`).

### J3. A grid stopped by its budget, watched on this page (P1, M4)

1. Finish line: "Time budget reached after X/Y cells — kept N images.
   Resume finishes the rest." and no button (`web/job-results.js:223-224`).
2. No banner: the watched job is marked announced
   (`web/job-stream.js:287-289`).
3. Sheet → Resume this run → preview → Run grid: **4+ actions**, and only
   if the panel still holds the grid and the run's oldest image carries the
   base seed (`web/sheet-render.js:580-583`; `/api/runs` reads the seed off
   the oldest image, `backend/gallery_reports.py:250-258`).

### J4. The morning after, fresh tab or server restart (P1, M4/M5)

1. Finished jobs are held in server memory only (`backend/jobs.py:356-364`),
   so after a restart there is no banner.
2. Retry/Resume lived in the old page's `submissions` map
   (`web/state.js:326-330`): lost.
3. Only the sheet route of J3 remains, with the same preconditions.

### J5. Judging a keeper in the lightbox (P3, M1)

Close the lightbox, find the card, pick it, More seeds…, Queue: **4-5
actions**. Or the Reuse trap.

### J6. Widen a finished grid on another axis (P1, M3)

Panel only, the axes must still be there, and the new images get a new run
id, so `run:<old>` excludes them.

| Breakpoint | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| No action where the run ends | x | x | x | x | x | |
| Needs a pick or filter first | x | x | | | x | |
| Needs the panel to hold the grid | | x | x | x | | x |
| Needs the base seed | | x | x | x | | x |
| Splits the run's identity | | x | | | | x |
| Lost on reload or restart | | | | x | | |
| "Resume" ambiguity | | x | x | x | | |

## Options considered

Each option was scored 1-10 by both judges.

### Option A: "surface" - More seeds and Resume where a run ends

Build no new engine. Put **More seeds…** and **Resume the remaining N
cells** at every place a finished run shows, under the names they already
have. One server change: an optional `MoreSeedsRequest.run` scope that
resolves a run's live images without the gallery filter, so Starred and
the filter's 1024-image cap (`backend/submission.py:345-359`) do not apply.
(Corrected: the `ids` path is capped too - `MoreSeedsRequest.ids` has
`max_length=settings.max_grid_images`, `backend/schemas_rerun.py:268` - so
picking cannot reach a run past 1024 images either; only the new `run`
scope is uncapped.) Budget stops get the finish line's
existing Resume button. No migration. Size **M**, about a day.

- **Usability: 8.** Every Must story it can, at the lowest cost; keeps the
  names Matt knows; reuses the dialog Matt chose. Leaves the morning after
  (US-7, US-8) unsolved and US-4 only partly met.
- **Engineering: 8.** Claims check out; the `run` branch is thin; nothing
  touches `run_id`, `CellKey` or the cell loop.
- **Corrections both judges found:** it said the finish-line More seeds…
  would also show for refine, inpaint and outpaint jobs, but gated it on
  `seeded`, which is false for them (`web/job-results.js:100-103`). It did
  not say that a budget Resume re-sends the original `budget_minutes`.

### Option B: "verb" - one Continue run verb, server classifies

`POST /api/runs/{id}/continue[/preview]`. The server classifies a run as
`unfinished`, `grid`, `batch` or `cells` and decides what "continue" means.
A `run_jobs` table stores each job's resolved cells and its outcome. Every
run-level control becomes **Continue**. Size **L**.

- **Usability: 6.** Matt's own word, one control everywhere, but one label
  hides "finish 27 cells" and "make 120 new images", and eligibility is
  known only after a press.
- **Engineering: 6.** Storing resolved cells is the sturdiest basis for
  restart survival. But its grid extension (`grid.expand` with
  `seeds = S + N`, then keep the new rows) is false as written: `expand`
  refuses when the *whole* grid exceeds `max_grid_images`
  (`backend/grid.py:953-970`) and seeds are capped at `MAX_GRID_SEEDS = 256`
  (`backend/seeds.py:23`), so a night near 1024 images could never gain a
  row. It also changes Resume to skip deleted cells, and adds a write that
  can refuse every submission.

### Option C: "run-home" - the run as a home

A run bar on the sheet with **Finish run**, **More seeds**, **Widen** and
**Load into panel**; a `run_specs` table stores each run's definition (the
request, base seed resolved) and Finish re-expands it; all four "Resume"
labels and the panel tick are renamed. Size **L**.

- **Usability: 7.** Best mental model and clearest vocabulary; holds known
  before a press through a per-action `actions` answer. Largest, with Widen
  built for a Could story with no evidence, and label churn.
- **Engineering: 6.** Found a real latent bug (below). But Finish
  re-expands the spec through `grid.expand` and inherits the same full-grid
  cap, which its own More seeds bump can push past; the spec write sits in
  `save_image`'s transaction, the most rule-bound path; and it changes the
  encrypted bundle format.

### The bug found along the way

`submission.not_yet_rendered` reads `storage.completed_cells` and never
`jobs.planned_keys()` (`backend/submission.py:292-314`; `planned_keys` at
`backend/jobs.py:399`). A Resume pressed while the same run's cells are
still queued renders them twice. More seeds already subtracts planned keys
(`backend/api_more_seeds.py:82-89`); Resume does not. Both judges confirmed
it. `not_yet_rendered` has five callers, not one: grid Resume
(`expand_grid`, `backend/api_generate.py:75`), the grid preview's
`in_gallery` count without Resume (`:269-272`), Re-render (`:245`), Swap tags
(`backend/api_swap.py:136`) and Mass refine (`backend/api_refine_batch.py:95`).
The last three share the same latent double-render.

### Verdict

Both judges recommend the same shape: **A now, then the stored-definition
part of C/B as phase 2, never B's single verb.** They differ on what to
store: the usability judge took C's spec, the engineering judge B's
resolved cells. This plan takes **resolved cells**: they do not depend on
today's prompt resolution (a later fix to it, like the `none`-spacing one
in `docs/rules.md:823-828`, cannot make a stored run resolve to another
grid), and finishing from them never goes through `grid.expand`'s
full-grid cap.

## The recommended design

### Vocabulary

| Meaning | Label | Where |
|---|---|---|
| Hold and carry on a live job | **Pause / Resume** | progress row only, unchanged |
| Render the missing cells of a run | **Resume the remaining N cells** (phase 1); **Finish run** in phase 2 only if Matt approves the rename (Q7) | finish line, banner, run bar |
| New seeds, same settings | **More seeds of this batch…** / **More seeds of this run…** | everywhere a run or image shows |

The scope is always in the label; nothing is called a bare "Continue".
The panel's grid Resume tick keeps its name. `docs/usage.md` gains one
"Continuing a run" paragraph mapping Matt's word onto these.

### Phase 1: where the run ends

**Server.**

1. `MoreSeedsRequest` (`backend/schemas_rerun.py:254-281`) gains
   `run: str | None`, refused together with `ids` or `search` (422);
   `favorites` and `all_models` are ignored when it is set.
2. A third mode of `submission.resolve_selection`, a `run=` keyword
   beside `ids` and the filter - not a separate resolver: CLAUDE.md has
   every rerun's images come only through `resolve_selection`. It reads
   the run's rows from `storage.lineage_images([run_id], [])`, live ones
   (`binned_at` None) as the batches; 404 when none is live. Binned rows
   still step seeds, as `lineage_images` reads every row
   (`backend/storage.py:1630-1646`). `lineage_images` builds records
   without `_missing_sources`, so `refine_missing` is never set on them; a
   gone source is still caught later by `unusable_input`
   (`backend/api_more_seeds.py:112-120`). No image cap; the
   batches × count ≤ `max_grid_images` refusal
   (`backend/api_more_seeds.py:57-65`) still bounds one press.
3. `more_seeds_plan` passes `run` where it resolves the selection
   (`backend/api_more_seeds.py:41-44`); everything after - `foreign`,
   held and planned keys, outpaint mismatch, `unusable_input`, the plan
   through `grid.cell_from_record`, the run join - is unchanged.
4. The preview answers `count_default` (the single batch's recorded Batch
   through `more_seeds.batch_size`, `backend/more_seeds.py:126-133`) and a
   server-written headline for a run scope: "N more of every cell × C cells
   = T images · seeds a–b · Starred ignored". The page shows both.
5. **The double-render fix:** grid Resume's branch of `expand_grid`
   (`backend/api_generate.py:73-76`) also subtracts
   `services.jobs().planned_keys().get((width, height), set())` (a size
   with no live job has no entry), so the grid tick, the banner Resume and
   the new budget Resume cannot queue cells a live job already plans. Not
   inside `not_yet_rendered` unconditionally: that would turn the
   preview's "already in the gallery" note (`web/grid-preview.js:194-197`)
   and Re-render's 409 "every selected image is already rendered"
   (`backend/api_generate.py:246-250`) into claims about queued cells. An
   opt-in argument, used by Resume now and offered to Re-render, Swap tags
   and Mass refine as a follow-up, keeps each caller's words true. The
   grid preview then reports those cells as a count of its own ("N
   queued"), since `skipped` is shown as "already done"
   (`web/grid-preview.js:233`).

**Page.**

- **Finish line** (`web/job-results.js` `finishJob`):
  - **More seeds…** after Use seed, scope `{ids}` of the job's live images,
    for a plain job, a More-seeds job and - with the gate widened beyond
    `seeded` (`:100-103`) - a single refine, inpaint or outpaint pass, whose
    strength, mask or spec are in `settings_key`. Held when every image was
    deleted or the checkpoint is another.
  - **More seeds of this run…** after Show sheet for a `done` grid, scope
    `{run}`.
  - **Budget Resume:** `remaining` counts `stopped_by` `'failures'` or
    `'budget'` (`:142`). The budget is checked before a cell starts, so
    `data.run` is the cells reached (`backend/jobs.py:909-917`); the button,
    its one-send label and its deleted-source hold are reused. It re-sends
    the submission's own payload, including its `budget_minutes` (Q6).
- **Single-image menu** (`imageItems`, `web/context-menu.js`): **More seeds
  of this batch…** after Use seed, mirrored through `syncedItem`; on a
  picked card the picks still win (`:192-193`).
- **Lightbox:** **More seeds…** paired with Reuse in an `.lb-pair` row (the
  class exists, `web/index.html:990`; `web/styles/lightbox.css:198-199`),
  so the column grows no taller and nothing below Reuse moves. It is always
  present and held with `aria-disabled` and a title, never hidden, so
  Reuse keeps one width from image to image and nothing shifts under the
  pointer as the lightbox steps. The dialog opens over the lightbox, which
  stays open. No new key (Q10).
- **Result side column** (≥1800 px only): More seeds… after Use seed. No
  new tool in the compact tile row.
- **Run bar on the sheet** (grafted from C): beside the sheet's controls
  (built from `web/sheet-render.js:204`; Resume this run at `:240-249`), a
  status line - "Run abc · 70/96 cells · WAI fp8" - then Resume this run
  (as today) and **More seeds of this run…**. The status line sits on a row
  of its own, or after every button: its count changes while a job writes
  to the run, and text before the buttons would move them under the
  pointer. The latter's gate is looser than `sheetRun()`: one run
  under `run:`, refine and comparison runs allowed, held (not hidden) on
  another checkpoint.
- **Dead-end pointer:** when Resume this run's preview skips cells and
  renders none, it says "Nothing in run X is missing. More seeds of this
  run… adds seeds."
- **Banner:** for a `done` job with images, **More seeds…** with scope
  `{run}` beside Show run.
- `more-seeds.js`: `openMoreSeeds(target, {count, opener})` takes a ready
  target; `moreSeedsGalleryChanged` skips `run` scopes as it skips `ids`
  (`web/more-seeds.js:80-86`). A refusal while opening is said today in the
  gallery's line (`els.galleryError`, `web/more-seeds.js:137`, `:155`,
  `:163`),
  which a modal lightbox covers: opened from the lightbox, it is said in
  the lightbox's own line instead. The count opens at the passed or
  server-suggested value on a run-surface open; the remembered count stays
  for the footer only (Q3). `syncMoreSeedsFor(button, rec)` holds with a
  reason on the bin, another checkpoint, offline, or the dialog already
  open; everything else is the server's per batch.
- The footer path is unchanged.

**Click counts after phase 1:** J1 2 (More seeds…, Enter); J2 2 from the
finish line, 3 after a reload via the Runs picker; J3 1; J5 2 with the
lightbox open.

### Phase 2: runs keep their cells

**Storage.** A `run_jobs` table, created per connection like the masks
tables (a table, not an `images` column, so `_migrate`'s column list is
untouched; a test still asserts an old database gains it): `job_id`,
`run_id` (indexed), `created_at`, `origin`, model, quantization, width,
height, batch, `comparison`, `cells` (gzip JSON: each cell's raw request,
seed, label, variables, stored *before* the outpaint stamp so a missing
outpaint cell is stamped again), and `outcome` (cells reached, failed
indices, `stopped_by`; NULL if the job never reached `_finish`).

**Writes.**
- `submit_cells` (`backend/jobs.py:583`) records the row off the loop
  beside the existing mask-hold write (`storage.hold_masks`,
  `backend/jobs.py:644-658`), after `Job(...)` mints the id (`:625`) and
  before `self._jobs[job.id] = job` (`:659`), the precedent for a DB write
  that can refuse a submission. (Corrected: it cannot go "before
  `Job(...)`" in the outpaint stamp's thread, since the job id does not
  exist yet.) The cells it stores are the list as passed in, captured
  before `storage.stamp_outpaints` replaces it (`:614-623`). It waits on
  the gallery lock as `set_favorite` does
  (`backend/storage.py:1649`); what to do on a hard failure is Q11.
- `_finish` (`:710`), the one terminal transition, writes `outcome`. It
  runs on the event loop and is synchronous, so the write goes to a daemon
  thread, never awaited, as its mask release does (`:728-733`); a failure
  there is logged and leaves it NULL, which falls back to "held or not",
  today's Resume.
- Bundles leave the table out (an imported run can still take More seeds);
  `clear` empties it.

**Finish run.** `POST /api/runs/{run_id}/finish[/preview]` and per-action
availability on `GET /api/runs/{run_id}` (`actions: {finish, seeds}`, each
with `available`, `reason`, `count`), so holds show before a press. The
cells are the union of the run's stored cells in stored order, minus held
keys (through `backend.applied()`, as `not_yet_rendered` compares) and
planned keys, submitted with the original run id passed explicitly. No
`grid.expand`, so no full-grid cap on a finish. It replaces the page-memory
Retry/Resume and the sheet's panel-driven `resumeRun`
(`web/sheet-render.js:587`), and covers a stopped plain Repeat-N (US-11).
The panel's grid tick stays for hand-built grids.

**Rules:**
- Another checkpoint: held, as `foreign` is today.
- Another quantizer: Finish held ("made under fp8; every cell would render
  again"), since `completed_cells` is quantizer-scoped; More seeds keeps
  today's note.
- Run still queued: Finish held, "the rest is queued in job X".
- Drift guard: a non-empty run whose Finish would skip zero cells is held
  ("this build no longer reproduces this run's cells").
- Deleted images: redrawn by Finish, as today's Resume does, and the
  preview says how many (Q5). Binned: never redrawn.
- Runs from before `run_jobs`, reindexed or imported: Finish held with "made
  before runs kept their cells: use the panel's Resume tick"; More seeds
  still works.
- Refine source gone, LoRA missing, mask swept: refused through
  `unusable_input`, as every rerun is.
- Comparison: Finish allowed (the variants are in the stored cells), More
  seeds as today, widening refused (Q8).

**If a "new seed rows" or "widen" mode is ever built** on stored grids, it
must build only the new cells (`seed_list` = the new seeds only), never
`seeds = S + N`, so the `max_grid_images` and `MAX_GRID_SEEDS` checks apply
to what is queued.

### Edge cases and refusals, both phases

| Case | Behaviour |
|---|---|
| Another checkpoint | Held with "made on X; the server is running Y"; server `foreign` per batch |
| Model switch mid-check | While opening, the dialog never opens and the line says "press More seeds… again" (`web/more-seeds.js:162-165`); once open, Queue is held with "close More seeds and open it again" (`:323`) |
| batches × N > 1024 | 422; the dialog opens with the count marked |
| Run gains images between preview and Queue | 409 on `expected`; the dialog re-checks |
| Second press while the first is queued | Seeds step past `planned_keys` |
| Bin view or binned image | Held with the bin's reason |
| Deleted top seed (More seeds) | May come back, documented (`docs/design-notes.md:5153-5155`); phase 2 can count stored cells as taken seeds |
| Bundled and cleared runs | 404; re-import is the route |

## Tests and smokes

**Phase 1**
- `tests/test_api_more_seeds.py`: `run` resolves exactly the run's live
  images and ignores Starred; binned images are not batches but their seeds
  are skipped; `run` + `ids` / `search` is 422; unknown or fully deleted run
  is 404; mixed checkpoint gives per-batch `foreign`; a run of more than
  1024 images in few batches is queued; `expected` 409; new images carry the
  run id; `count_default` and the headline.
- A test that `not_yet_rendered` subtracts planned keys (grid Resume while
  the run is queued plans nothing twice).
- Smokes: `smoke-more-seeds` (7902: finish line, lightbox, menu on an
  unpicked card and a tile, holds, run bar with Starred ticked);
  `smoke-jobs` (7882: extend the budget block at lines 111-115 with the
  button, one press, "Resume sent"); `smoke-context-menu` (7908: one more
  item); `smoke-lightbox` (7884: the pair, nothing moves);
  `smoke-sheet` (7880: run bar and the dead-end pointer); `smoke-wide`
  (7876: side column). No new port.

**Phase 2**
- Storage: an old database gains `run_jobs`; `clear` empties it; a crash
  path leaves `outcome` NULL.
- Jobs: `_finish` writes the outcome for done, cancelled, budget and
  failures.
- HTTP: Finish after a budget stop queues exactly the rest under the same
  run id; after a reload and a restart; when the base seed's images were
  deleted or the run began as a Resume (US-8); a stopped Repeat-N; the
  quantizer, queued and drift holds; a legacy run held; stale revision
  refused (the route joins the model-dependent set, `backend/main.py:360`).
- A new `tools/smoke-run-continue.json` on **7909** (FLUX.1-Krea-dev, fresh
  scratch `--data`, `--step 0.3`); a job ends by its budget with no new
  fake-server knob, as `smoke-jobs` already does with a 0.06-minute budget
  over 20 seeds at `--step 0.3` (`tools/smoke-jobs.json:112-114`); its row
  added to the CLAUDE.md/AGENTS.md port table, which DEV#2's
  test holds to `tools/smoke*.json`.

**Docs, either phase:** `docs/usage.md` (entry points, "Continuing a run"),
`docs/design-notes.md` (a "More seeds where the run ends" section; later
"Runs keep their cells"), `docs/rules.md` (frontend mirroring rule, the
`run` scope, smoke coverage), `TODO.md`. Fix the stale hover wording in the
same pass: `docs/design-notes.md:5173-5174`, `docs/usage.md:499` and `:512`,
`backend/more_seeds.py:9`, `backend/api_more_seeds.py:228`,
`backend/schemas_rerun.py:259` all still name the removed batch hover.

## Phases

| Phase | Content | Size |
|---|---|---|
| 1a | `not_yet_rendered` planned-keys fix and its test | S |
| 1b | `run` scope, `count_default`, headline; finish line (More seeds ×2, budget Resume); menu; lightbox pair; side column; run bar and pointer; banner More seeds; smokes; docs | M |
| 2 | `run_jobs`, outcome writes, `GET /api/runs/{id}` actions, Finish run replacing page-memory Resume; new smoke | M-L |
| later | Widen, Load into panel, Seeds to grid, CLI `--continue` - only on Matt's answers | - |

Phase 1 needs no migration and nothing for `docs/verified.md`; each phase
ships alone.

### Phase 2 execution slices

**Persistence built and verified, 2026-10-18 (`GEN#11`); service/page work remains.** Keep
one TODO row for the feature. These are dependent stages, not three ready
features. All fourteen product questions are answered; neither task trials
for the whole-app layout nor a card measurement blocks this work.

| Stage | Scope | Evidence before moving on |
|---|---|---|
| Persistence | Create `run_jobs`; store the resolved pre-stamp cells before admission; write outcomes through `_finish`; clear the table with the gallery; leave bundles unchanged | Old-DB creation, raw outpaint round trip, disk-failure refusal without admitting a job, NULL outcome after interruption, and all terminal outcomes covered by storage/job tests |
| Finish service and API | Resolve stored cells in order, subtract completed/binned and planned keys, expose availability plus preview/submit with the original run id | HTTP tests for restart, stopped Repeat, partial completion, queued/model/quantizer/drift/legacy holds, stale model revision, missing inputs and preview/submit changes; use the queue-before-gallery-read ordering from grid Resume |
| Page integration | Replace page-memory recovery actions with Finish run availability; retain the panel's hand-built Resume and More seeds; add the 7909 smoke and update usage/rules/maps | Fresh-tab and restarted-server fake-backend scenarios work with an unrelated composer; focus/holds/counts are correct; no duplicate queued cells and no claim that legacy/imported runs have retained cells |

The storage/API stages may be reviewed separately. The feature stays open
until the page path and restart evidence above pass. The complete storage
contract, refusals and tests remain in [Phase 2](#phase-2-runs-keep-their-cells)
and [Tests and smokes](#tests-and-smokes); this breakdown does not change them.

**Persistence evidence, 2026-10-18.** `tests/test_run_persistence.py`
adds 24 cases for old databases, raw outpaint requests and metadata,
disk-failure refusal/mask release, every terminal path, interrupted NULL
outcomes, gallery clearing, export exclusion and import preservation.
The focused persistence/jobs/storage/outpaint/bundle/clear/rerun regression
command passed 245 tests. Outcomes also keep `completed_indices`, allowing
Finish to distinguish deleted successful cells from failures and unfinished
ones. Outcome writes capture their database path and update existing rows
only, so delayed daemon work cannot recreate cleared recovery definitions.

Persistence follow-up, 2026-10-18: mask touch/hold/release writers now take
the gallery reservation before the write lock, matching terminal outcome
writes. Three deterministic contention regressions and the combined
persistence/outpaint/CLI set passed (140 tests).

This night's execution forbids trusted instruction-file edits. Page/restart
checks therefore extend existing smoke files and may use a standalone
owned-fake-server check instead of adding a new mapped 7909 smoke; the
required scenarios and evidence remain the same.

Concurrent overlapping Finish submissions must not admit the same missing
cells twice, including cells awaiting mask/outpaint writes before admission.
Test admission coordination separately from completion during a gallery
scan; a queue snapshot alone does not cover simultaneous new submissions.

## Risks

1. **Per-cell "N more" may not read as "more seed rows"** for a grid; for a
   listed seed axis it continues from max+1, not a list of Matt's choosing.
   The server headline must say it (Q2).
2. **Dialog over the lightbox:** both are modal; focus return and Escape
   order need the smoke.
3. **The run identity split stays for the panel route.** A grid typed into
   the panel and Resumed with more seeds or values still gets a new run id
   (`backend/grid.py:883-921`); only the run's own actions keep it.
4. **A new write in `submit_cells`** (phase 2) is a failure point at 3 a.m.
   (Q11).
5. **Legacy runs** hold Finish, so the run bar behaves differently across
   one gallery until older runs are cleared - which, at Matt's
   bundle-and-clear rhythm, is days.
6. **Phase 1 leaves J4 fragile:** the budget Resume is still page memory
   until phase 2.

## Open questions for the user

All fourteen answered 2026-10-18: Matt took every recommended answer, so
each stands as written. Q7's rename waits for phase 2; Q13 defers Widen and
Load into panel.

1. **Does "continue a run that completes" mean mainly M1 (more of a batch)
   and M2 (more seeds on a finished grid)?** Recommended: yes; phase 1 is
   built around them.
2. **For a finished grid, does "4 more" mean 4 more per cell (4 more seed
   rows at Batch 1)?** Recommended: yes, as More seeds already does, with
   the total stated before Queue.
3. **Default count from a run surface: the batch's own Batch, the
   remembered count, or always 4?** Recommended: the batch's own Batch from
   run surfaces; the footer keeps its remembered count.
4. **May the trivial case (one batch, N = its Batch) skip the dialog?**
   Recommended: no; keep the dialog, opened with Enter ready, so Queue always
   matches a server answer.
5. **Should Finish redraw images you permanently deleted?** Recommended:
   yes, as Resume does today, with the preview saying how many.
6. **Should a budget Resume run under the same budget again?** Recommended:
   yes (it re-sends the night's own request), with the budget shown on the
   button's title.
7. **Rename the missing-cells controls to "Finish run"?** Recommended: yes in
   phase 2, when they stop depending on the panel; keep "Resume" in phase 1.
   Pause/Resume and the panel tick keep their names.
8. **Comparisons:** allow Finish from stored cells in phase 2?
   Recommended: yes; widening the variant axis stays refused.
9. **Should continuing a run ignore Starred** (the whole run, not only its
   starred batches)? Recommended: yes for the run scope; picks and the
   footer filter keep today's behaviour.
10. **A lightbox key for More seeds (e.g. M)?** Recommended: no; the button
    and the menu are enough for now.
11. **Phase 2: if recording a job's cells fails, refuse the submission or
    queue it without Finish?** Recommended: wait on the gallery lock, refuse
    only on a disk error, and say so in the error line.
12. **Store runs' cells at all (phase 2), and should extensions always join
    the run?** Recommended: yes to both - Matt chose joining for More seeds
    (`docs/design-notes.md:5171`).
13. **Are Widen (M3) and Load into panel (M7) wanted?** Recommended: defer
    until a night shows the need.
14. **Clear picks after a More seeds from the footer?** Recommended: leave
    as is; the new entry points never use picks.

## Non-goals

1. No change to pixels, seeds or how seeds are drawn; seeds stay
   server-chosen.
2. No change to how existing run ids are computed.
3. No continuing across checkpoints.
4. No redrawing of binned cells.
5. Nothing queues by itself; the page never scrolls or moves a control
   under the pointer.
6. No new GPU-memory policy; `max_batch_pixels` and `max_grid_images` stay.
7. No reaching runs removed by bundle-and-clear; re-import is the route.
8. No change under 1800 px for wide-only extras; the 600 px footer row gains
   nothing.
9. No CLI change in phase 1; `--resume` is parity only.
