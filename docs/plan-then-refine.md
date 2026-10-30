# Plan: generate, then refine, in one job

**Status, 2026-10-29: phases 1 and 2 built; phase 3's card check half
done. Its WAI v17 night ran - no drift in the alternation's peak, the pair
back to back no slower than apart, the same pixels either way
(`verified.md`, Refine pass, "Peak per image and Then refine, WAI v17");
its FLUX.1 Krea dev night is left (`GEN#16`). That night stopped at the
RAM guard during its load on 2026-10-25; `GEN#22`'s fixed loader has
since met that prerequisite (a cold FLUX.1 Krea dev fp8 load peaked at
27.9 GiB working set, neither guard tripping:
[verified.md](verified.md#ram-at-the-load-the-fixed-loader-2026-10-27)),
so the card run itself is the one remaining step, and it waits on Matt's
OK. The build items once filed under phase 3 are now rows of their own,
added 2026-10-29: [phase 4](#phase-4-finish-run-adds-missing-passes),
missing passes in Finish run (`GEN#35`, built 2026-10-30); [phase 5](#phase-5-more-seeds-carries-the-pass),
More seeds carrying the pass (`GEN#37`, built 2026-10-30); and
[several strengths per image](#decision-several-strengths-per-image)
(`GEN#39`, DECISION: Q2's "if wanted" is Matt's yes or no).**
Matt took every recommended answer to the open questions the same day ("i
like recommended"). Built as below except: a pass's failures have a streak
of their own, apart from the cells', since a base made between two failed
passes reset the shared count and three failed passes never ended a night;
the new module is `backend/then_refine.py` (the stand-in pass, the
refusals, the pass run id); the fold warns only past the guard, not at the
measured tight line; the run hint is the fold's own hint line; the control is
not the fold the plan drew but an opener at the Notify row's end and a
dialog, since the fold's summary row pushed Run grid below a 900 px window;
and `smoke-wide`/`smoke-narrow` were run unchanged rather than extended. Why it
is built so is in `docs/design-notes.md`'s "Generate, then refine".
Every statement about today's behaviour cites the file and line it was read
from on this date. What only the card can say (time, memory, how a pass
looks) is out of scope: the design reuses the refine pass unchanged, so no
pixel of a pass differs from one the Refine dialog or Mass refine makes.

The recommended design, in one paragraph: **the follow-on pass is a stage
inside the worker, not a cell planned at expand time.** A Generate or Run
grid submission may carry `then_refine` (scale, strength, steps - Mass
refine's three numbers). The job's cells stay the base cells, unchanged.
After each base cell's images are saved, the worker builds one pass per
saved record with `refine_batch.plan` - the function Mass refine already
builds its cells with - and runs those passes before the next base cell.
Each pass row is an ordinary refine row (`refine` = the just-saved base's
gallery id, the base's own seed, prompt pair, CFG, mu and LoRA), under a
run id of its own derived from the base run's. The base row, its cell key,
its run id and the seed loop are untouched.

## The ask and the evidence

Matt rates Refine at the current size, enlarging with Refine, trying several
strengths and Mass refine all "Often" (`docs/design/preferences.md:143-147`),
and "Refine and enlarge, strength/version comparison and Mass refine
prominent" is the stated priority for editing keepers (`:222`). The SDXL
anime authors' own recipe is a two-stage one: WAI's "1024x1344 followed by a
1.5x hires pass at denoising strength 0.35-0.5" (`docs/design-notes.md:1237-1238`;
the hires recipe again at `:8857-8858`). Today that takes two presses per
image or a Mass refine after the run: a card's Refine, or the gallery's
picked cards through Mass refine, which was built because the single-image
route was "twenty times over" (`docs/design-notes.md:5290-5292`).

## What exists that the design leans on

- **A pass is a request field naming a gallery id.** `GenerateRequest.refine`
  is an `int`, `ge=1` (`backend/schemas_generate.py:231`), and `strength`
  means nothing without it (`:219-233`; `applied()` resets it, `:514-515`).
  The rules: "References and refine sources are gallery ids, never paths"
  (AGENTS.md, Reproduction and storage).
- **Mass refine already turns one record into one pass cell.**
  `refine_batch.plan(record, running=, profile=, scale=, strength=, steps=,
  include_foreign=, skip_passes=)` (`backend/refine_batch.py:67-157`) sizes
  the pass with `refine.scaled` (`:52-55`), checks the guard (`:100-106`),
  builds the cell through `grid.cell_from_record` with `refine=record.id`,
  the strength, no references, mask, outpaint or upscaler (`:107-140`),
  then `check_budget` and `check_refine_steps` (`:141-157`). The cell keeps
  the record's seed (`grid.cell_from_record`, `backend/grid.py:771`), as the
  Refine dialog's seed box starts at the source's (`web/refine.js:291`).
- **The worker is one loop over fixed cells.** `JobManager._generate`
  (`backend/jobs.py:924`) checks every cell's refine capability against the
  loaded pipeline first (`:935-939`), then per cell: cancel, pause (`:993`),
  the budget's "start no cell after" (`:999-1005`), `backend.generate`
  (`:1033`), `_save` (`:1041`), `completed_indices` (`:1042-1043`), and the
  skip-and-continue failure rule (`:1049-1060`, `_too_many_failures` at
  `:921-922`). `_save` records `backend.applied(cell.request)` (`:1116`)
  under `job.run_id` (`:1132`). "A job's cells are fixed when it is made"
  is written into `planned_keys` (`:420-431`) and `one_source` (`:202-213`).
- **The saved PNG is in place when `_save` returns.** `save_image` writes
  `.png.part` and renames after the row commits (AGENTS.md), and a pass reads
  its source from the file by id (`refine.source_image(request.refine, ...)`,
  `backend/backends/diffusers_backend.py:815`).
- **Runs are retained.** `submit_cells` keeps the raw cells and a metadata
  dict (`backend/jobs.py:647`, `:705-716`, written at `:719`), and Finish
  re-submits a run from them, passing each metadata key back by name
  (`backend/api_generate.py:303-309`).
- **A run has one size.** `gallery_reports.runs` reads size, checkpoint and
  quantizer off a run's oldest image because "`run_id` hashes them"
  (`backend/gallery_reports.py:260-262`); `grid.run_id` hashes the first
  cell's size (`backend/grid.py:919-926`).

## The alternatives

**A. A second cell planned at expand time.** `grid.expand` (or the routes)
would append, after each base cell, a pass cell whose source is "the image
cell k will make". Its `refine` is unknown until that save, so the cell
cannot be a `GenerateRequest`: it needs a placeholder (a new field such as
`refine_of_cell`, or a sentinel id). That touches, in order: refine sources
being gallery ids; `GenerateRequest` as the reproduction set; `CellKey` and
`run_id` (the key's refine element is `id@strength`,
`backend/grid.py:140-147`, so a pass key cannot be formed - a resume could
never count a pass done, and `run_id` over placeholder keys would change
when resolved); `retain_run_job`'s raw cells (`backend/storage.py:1729-1754`)
and Finish, which would retain unresolvable requests;
`completed_cells`/`not_yet_rendered`, which scope by one size
(`backend/storage.py:1023-1029`) where the job would now hold two; the
duplicate count; the `check_refine_steps` on every expanded cell
(`backend/grid.py:1033`); `planned_keys`; and the CLI's copy. Every reader
of a cell would have to learn the placeholder. Rejected.

**B. A stage inside the worker after each base save (recommended).** The
job's cells stay the base cells, so `grid.expand`, `jobs.submit`'s inline
seed loop (`backend/jobs.py:604-618`), the CLI's copy and
`tests/test_cli.py`'s equality, `CellKey`, the base `run_id`,
`applied()`, retained cells, Resume and the duplicate count are untouched.
The pass cell is built from a saved record, so its source is a real id and
it is an ordinary refine cell to every reader. What it touches: the
worker loop (a stage per cell), "a job's cells are fixed when it is made"
(`planned_keys` and the snapshot's counts must add the passes), the
failure rule's arithmetic, and one more key in the retained metadata.

**C. A chained second job: Mass refine of the base job's images at its
end.** Touches the fewest invariants of all - a second ordinary
`submit_cells(refine_batch=...)` - but it is not what was asked. The passes
wait for the whole run (an overnight grid's passes all come at the end, and
a budget or a cancel leaves none); another job queued meanwhile runs
between the two; the page follows two jobs for one press; and the second
job's admission happens in `_finish`, which must stay the one synchronous
terminal transition (AGENTS.md, Jobs). Kept as the fallback answer for
recovering missing passes (below), not as the feature.

## The request

A new model `ThenRefine` in `backend/schemas_generate.py`: `scale` (1.0 to
2.0, as `RefineBatchRequest.scale`, `backend/schemas_rerun.py:231`),
`strength` (None = the checkpoint's own, `default_strength`, validated by
`_a_fraction` as Mass refine's is, `:244-247`) and `steps` (None = each
cell's own, as Mass refine's `:233`). No prompt, no upscaler: the pass
keeps the cell's prompt pair (Mass refine's rule) and Lanczos (Mass refine
"creates a new ordinary pass with Lanczos", `docs/design-notes.md:1661`).

It is **not** a `GenerateRequest` field. That model is the reproduction set
and is dumped into every cell (`grid.expand`'s `template`,
`backend/grid.py:985`) and every retained cell (`backend/storage.py:1740-1744`).
It goes on the two submission bodies only: a small `GenerateSubmission
(GenerateRequest)` with `then_refine: ThenRefine | None = None` that
`/api/generate` takes, and `GridRequest` derives from it instead of from
`GenerateRequest` (`backend/schemas_generate.py:564`). The routes pass it to
`submit_cells(then_refine=...)` beside `refine_batch=`/`more_seeds=`, the
pattern `submit_cells`' docstring describes for submission-only fields
(`backend/jobs.py:633-644`), and build cells from the request without it
(expand's `GenerateRequest(**template)` ignores unknown keys, as it already
ignores the grid's axis fields). `jobs.submit` gains the keyword and passes
it through, its seed loop unchanged.

**Refused at submit, in words, before anything is queued:**

- the checkpoint has no refine pass: `submission.refine_unavailable`'s own
  sentence (`backend/submission.py:113-124`) - Krea 2 Turbo and Raw, both
  klein profiles, ERNIE, NewBie, NetaYume and every Anima entry profile
  `refine_supported: False` (`backend/model_defaults.py:169`, `:204`,
  `:237`, `:274`, `:738`, `:767`, `:910-1068`);
- the base is itself a pass, inpaint or outpaint (`refine` set): "Then
  refine follows a direct render; this is already a pass of #N";
- the pass size, `refine.scaled(width, height, scale)`, past
  `refine_batch.too_big` (`backend/refine_batch.py:58-64`), worded as Mass
  refine's `too_big` (`:100-106`). One base size per job (a grid has no
  size axis), so this is one check;
- a strength that leaves no step of any cell's pass steps
  (`check_refine_steps`, `backend/schemas_generate.py:326-333`), checked on
  a stand-in pass of each distinct steps value - a grid's Steps axis makes
  several - and worded as Mass refine's `no_step` (`backend/refine_batch.py:150-156`).

The worker re-checks capability against the loaded pipeline with the
cells (`backend/jobs.py:935-939`): one stand-in pass request (any id; the
check reads `applied()`, never storage, `backend/backends/base.py:114-139`)
so a profile guess that a load contradicts fails before the first base
image, not after it.

## The worker

`Job` gains `then_refine: ThenRefine | None`, `pass_run_id: str | None`,
`stage: str | None` ("pass" while a pass runs) and `pending_passes:
list[Cell]`. In `_generate`, after a base cell's `_save` succeeds:

1. Build the passes: for each record `_save` just wrote, `refine_batch.plan
   (record, running=model, profile=defaults_for(model), scale, strength,
   steps, include_foreign=False, skip_passes=False)`. A status other than
   `queued` (the guard, a missing step) cannot happen after the submit
   checks; if it does, it is a pass failure with that `detail`.
2. Set `job.pending_passes` (so `planned_keys` includes them), `job.stage =
   "pass"`, `job.label = "pass of #N"`, `job.total_steps =
   progress_steps(pass)` and emit a status frame. `job.run` stays the base
   cell's index: the pair is one unit.
3. For each pass: `backend.generate(pass.request, seed=pass.seed, ...)` with
   the same callbacks, then `_save` under `run_id=job.pass_run_id` (a new
   `run_id` argument to `_save`, defaulting to `job.run_id`).
4. Clear `pending_passes` and `stage`.

**Cancel, pause and the budget.** A base cell and its passes are one unit.
Cancel is polled at every step as now (`callback_on_step_end`), so it stops
inside a pass too, leaving the base saved and its pass missing. Pause and
the budget are checked only before a base cell, as today
(`backend/jobs.py:991-1005`): a budget never leaves a finished base without
its pass, and `_hold`'s "hold after the current cell" keeps its meaning.

**Failure.** A base that fails has no pass (nothing to refine) and counts as
today. A pass that fails leaves its base row; it is reported in
`failed_cells` as `{"label": "pass of #N", "seed", "error"}`, counts in
`failures` and `consecutive`, and adds nothing to `failed_indices`, which
index `job.cells` and feed Finish (`backend/api_generate.py:172-174`). The
base's index stays in `completed_indices`: the base cell did complete. The
failure rule's denominator becomes base cells plus planned passes. Three
passes failing in a row (an OOM at a scale the card does not hold) ends
the night as three failed cells would (Q3).

**What `completed_indices` and the snapshot mean.** `completed_indices`
keeps meaning "base cell complete". The snapshot adds `then_refine` (the
spec), `stage` and `pass_run_id`; `planned_images` becomes base images
times (1 + passes per image), since the page's placeholders and "N of M"
read it (`backend/jobs.py:271`). `one_source` is unchanged: a job of base
cells has no shared source.

## Reproduction and lineage

- **The base row is unchanged**: same request, same `applied()`, same
  `CellKey`, same run id. A then-refine run and a plain one of the same
  settings share base rows' keys, so Resume of either counts the other's
  bases as done - correct, since they are the same images.
- **The pass row is a normal refine row**: `refine` = base id, `strength`,
  the base's seed, `batch` 1 (`cell_from_record` sets no `num_images`),
  size = base size times scale on the 16 grid. Its `CellKey` is exactly
  the one Mass refine would give the same pass, so Mass refine's
  `rendered` status, Re-render, Reuse (into the Refine dialog), Load
  settings from PNG and bundles' source remapping see nothing new.
- **The pass run id** is `grid.pass_run_id(base_run_id, spec, model,
  quant)`: the first 12 hex of a SHA-1 over `{"then_refine_of": base_run_id,
  "scale", "strength", "steps"}`, the strength as `applied()` fills it.
  Deterministic, so a Finish or Resume of the base run continues the same
  pass run. Its own run keeps "a run has one size" true for both runs; the
  shared id would put two sizes in one run, which `gallery_reports.runs`
  and `RunDetails` (`backend/api_generate.py:276-281`) read off one row.
  Run ids are no part of the reproduction set (docs/rules.md,
  Reproduction).
- **Retained metadata** gains `"then_refine": {scale, strength, steps,
  pass_run_id}` beside `refine_batch`/`more_seeds` (`backend/jobs.py:710-714`).
  `origin` is unchanged ("generate" or "grid"). Retained cells are the
  base cells only, unchanged in format.

## The other producers

- **Resume (the grid tick) and Finish run.** Both submit base cells; the
  passes follow the cells kept. Finish must pass `then_refine` back from the
  metadata (`backend/api_generate.py:304-309` names each key). A base that
  is complete but whose pass is missing (cancelled or failed between the
  stages) is not missing to either. Phase 1's answer is Mass refine over
  `run:<base run>` at the same numbers: its `rendered` status skips every
  base whose pass exists, and the rows it makes have the same keys (under
  Mass refine's run id). [Phase 4](#phase-4-finish-run-adds-missing-passes)
  adds those passes to Finish's plan (Q4, `GEN#35`, built 2026-10-30).
- **The pass run's own Finish.** It has no retained definition, so today's
  `finish_plan` would say "made before runs kept their cells: use the
  panel's Resume tick" (`backend/api_generate.py:179`), which is wrong.
  Its records' `job_id` names the base job, whose definition carries
  `pass_run_id`; `finish_plan` reads it (a new
  `storage.retained_run_job(job_id)`, `job_id` is the primary key,
  `backend/storage.py:224`) and says "refine passes of run X: finish that
  run". More seeds on it keeps its current meaning for any pass batch.
- **More seeds** of a base batch adds base images only: it builds cells
  from records (`backend/more_seeds.py:23-25`) and reads no retained
  metadata. Q5 answered: not yet; [phase 5](#phase-5-more-seeds-carries-the-pass)
  carries it (`GEN#37`, built 2026-10-30).
- **Re-render** of a base re-renders the base; of a pass, the pass of the
  same source at the new size - both as today (`grid.cell_from_record`,
  `backend/grid.py:715-783`).
- **Grids.** Every grid cell gets its passes. The grid's Strength axis stays
  what it is - an axis only beside a source, collapsing to one value
  otherwise (`backend/grid.py:512-518`) - so it never sweeps the follow-on's
  strength. Several follow-on strengths per image is Q2, open as
  [a decision](#decision-several-strengths-per-image) (`GEN#39`).
- **Duplicates.** `duplicate_count` counts base cells; a duplicate base has
  a duplicate pass, and the preview's duplicate line says "(and their
  passes)" where it is non-zero.

## Estimates, disk and the budget

`GridPreview` (`backend/schemas_generate.py:648-690`) gains `passes` (images
times one) and the pass size; `seconds` adds the passes' price,
`refine_cost_ms` at the pass size (`backend/gallery_reports.py:71-73`)
keyed by denoised steps (`grid.denoised_steps`, `backend/grid.py:795-802`),
and stays None where either has no history - as today, never extrapolated.
`bytes` adds `image_bytes` at the pass size. `seeds_within_budget` prices
each unit as base plus passes, so "the budget completes N seeds" matches
the worker's rule above (`backend/grid.py:864-886`). `start_sigma` is
answered for the pass, as the Refine dialog's preview is
(`backend/api_generate.py:533-537`). Plain Generate has no preview; its run
hint says the pass size in words.

## The CLI

`generate.py` gains `--then-refine-scale F`, `--then-refine-strength S` and
`--then-refine-steps N` (one flag turning it on; the other two optional),
refused with `--refine`, `--mask` and `--outpaint` as the parser's existing
alternatives are (`backend/generate.py:978-1013`). Its loop is its own copy
of the worker's (`:1690-1790`); after a cell's saves it calls the same
`refine_batch.plan` on each saved record and runs the passes under the
pass run id. The seed loop (`grid.expand`, `:1232`) is untouched, so
`tests/test_cli.py`'s equality holds as it is, and a new test holds the
CLI's pass rows equal to the server's for one cell. `--via-server` forwards
`then_refine` in the body.

## The page

**Placement.** The panel's order is fixed and nothing rarely used goes
between Sampling and Generate (`docs/rules.md:2278-2287`); the prompt area
has no spare height at 1400x900 (measured 2026-10-22). The control is a
fold of its own, `<details class="section" id="then-refine-section">`,
directly after `.notification-choice` (`web/index.html:545-549`) and before
the grid fold (`:552`): one summary line below Generate and everything a
job adds under it, so Generate, the bars and the elapsed line do not move.
It holds Mass refine's three controls and wording (`web/index.html:1525-1542`:
Strength, Scale presets, Steps "each image's own"), sharing
`refine-batch.js`' scale buttons and the single dialog's remembered
`REFINE_KEY` numbers (`docs/design-notes.md:5349-5352`), and one tick,
**Refine each image**.

**Visible while on.** A setting that doubles every press must not hide in a
closed fold: the summary reads "Then refine <em>1.5× at 0.4</em>" (off:
"<em>off</em>"); Generate's label becomes "Generate + refine" and Run
grid's "Run grid + refine", their titles naming scale and strength (no
size change: the buttons are full width); the run hint adds "each then
refined at 1.5× to 1536×1536". The progress line names the stage ("cell
3/12 · pass of #418").

**Held, never hidden.** Where `refine_supported` is false the tick is
`aria-disabled` with `refine_unavailable`'s words as its title, and a press
says them (AGENTS.md, Frontend); a checkpoint switch to one with no pass
holds a tick that was on and the labels go back. Where the pass would be
past the profile's tight line (`defaults.refine_tight_pixels`), the fold
warns in the Refine dialog's words (`refineCapNote`,
`docs/design-notes.md:1374-1377`); past the guard, the server's refusal.

**Results.** Pass images arrive as ordinary `image` events; the results
strip shows each pass after its base. Reuse of a pass opens the Refine
dialog as today.

## Tests and smokes

Python (no GPU): `tests/test_jobs.py` - a 2-cell batch-2 job writes 4 base
rows then, after each cell, its 2 passes in order; pass rows' refine,
seed, strength, size, `batch` and run id; cancel during a pass keeps the
base; pause and the budget act only between units; a pass failure is
reported, not in `failed_indices`, and three in a row stop the job;
`planned_keys` includes pending passes. `tests/test_api_grid.py` and
`tests/test_api_refine.py` - every refusal above, the preview's passes,
seconds, bytes and budget seeds. `tests/test_api_finish_run.py` - Finish
carries `then_refine`; the pass run's reason. `tests/test_refine_batch.py`
- the pass cell equals Mass refine's for the same record and numbers.
`tests/test_cli.py` - the CLI's pass rows equal the server's; the flag
refusals. `tests/test_run_persistence.py` - the metadata key round-trips.
A test that no `GenerateRequest` field was added (the reproduction set's
field list is pinned).

Smokes: a new `smoke-then-refine.json` on port 7912, FLUX.1-Krea-dev,
fresh scratch `--data`, `--step 0.3`: the fold, the tick, the labels, a
Generate of batch 2 producing two bases and two passes in the strip, the
lineage line on a pass, the stage in the progress line. The held tick on
the default Krea 2 Turbo fake joins `smoke-refine-unavailable`.
`smoke-wide` holds the panel's order with the fold; `smoke-narrow` that
Generate and a running batch's progress do not move with it open or shut.
AGENTS.md's port table and `docs/rules.md`'s smoke list gain the row.

## Phases

Effort on the review's scale: S a day or less, M a few days, L a project.

**Phase 1 - backend, CLI and reproduction (M).** `ThenRefine` and the
submission body; the submit refusals; `submit_cells(then_refine=)` and the
retained metadata; the worker stage, `_save`'s run id, `planned_keys`, the
snapshot; `grid.pass_run_id`; Finish carrying the spec and the pass run's
reason; the preview's passes, time, disk and budget; the CLI flags and
loop; the fakes; every Python test above. Done when `uv run --group dev
pytest` passes and a fake-server job with `then_refine` writes base and
pass rows a Mass refine of the same bases would report `rendered`.

**Phase 2 - the page (M).** The fold, the tick, the labels and run hint,
the held state, the tight-line warning, the progress stage, placeholders
for passes, the smokes, `docs/usage.md`, `docs/rules.md` (Layout, Rules,
Tests) and a "Generate, then refine" section in `docs/design-notes.md`.
Done when the new smoke, `smoke-wide`, `smoke-narrow`,
`smoke-refine-unavailable` and the existing refine smokes pass.

**Phase 3 - the card (S+card, `GEN#16`).** The card's word on a
then-refine night on WAI v17 and FLUX.1 Krea dev at 1.5x (one checkpoint
per session, stopping at the first that fills the card): peak memory of
the alternation, which moves the VAE and the sibling class every cell, and
whether the base and its pass back to back cost more than the two run
apart. WAI v17 is done (`verified.md`, "Peak per image and Then refine, WAI
v17"). FLUX.1 Krea dev is left: its 2026-10-25 attempt stopped at the RAM
guard during the load, and the fixed loader's 27.9 GiB peak
([verified.md](verified.md#ram-at-the-load-the-fixed-loader-2026-10-27))
clears that, so only the run remains. It needs Matt's OK for the session,
and RAM checked as well as the card (FLUX loads spike RAM; one load per
session). Done when `docs/verified.md` records that night's peak of the
alternation beside `nvidia-smi` and shared GPU memory, and the pair's time
back to back against apart, or the guard that stopped it. The phases'
earlier build items (several strengths, missing passes in Finish, More
seeds carrying the pass) are no longer part of phase 3: they are phases 4
and 5 and the decision below.

### Phase 4: Finish run adds missing passes

**Status: `GEN#35`, built 2026-10-30 (M, CPU/fake/browser only); added
2026-10-29.** Built as designed below; what differs and the evidence are
in [Phase 4 built](#phase-4-built) at the section's end.
Matt took Q4's recommendation on 2026-10-24 ("i like recommended"): a
later phase adds the missing passes to Finish's plan, and Mass refine over
`run:<id>` recovers them until then (Q4 below; the answers paragraph,
"missing passes in Finish in a later phase"). The status line had filed
it under "if wanted"; it is not conditional. Matt rates "Retry
skipped cells or resume the remainder" (D9) and Mass refine (H5) both
Often (`docs/design/preferences.md:88`, `:147`).

**Today, read 2026-10-29.** `finish_plan` (`backend/api_generate.py:187`)
counts a retained cell missing only where its own image keys are not all
in the gallery (`:275-293`), so a base saved before a cancel or a failed
pass is "rendered" and its pass is never looked for. A run whose every
base is saved but some passes are not answers "nothing in this run is
missing" (`:303`) and is held. The pass run's own Finish names its base
("the refine passes of run X: finish that run, and its passes follow",
`:257-267`), which then says nothing is missing: a dead end. The only
recovery is Mass refine over `run:<base run>` at the same numbers
(`docs/design-notes.md:12496-12499`), whose passes have the pass's
`CellKey` but land under Mass refine's own run id, not the pass run. Finish
passes back the first retained job's `then_refine` only
(`api_generate.py:402-414`, `_retained_then_refine` at `:179-184`).

**Design.**

- **Which passes are missing.** For each retained job of the run whose
  metadata carries `then_refine` (`backend/jobs.py:920-933`), its live base
  records - the run's rows from `storage.lineage_images([run_id], [])`
  (`api_generate.py:197`) with that `job_id`, `refine` None and
  `binned_at` None (a binned base is Matt's discard; its pass would be
  refused anyway as a binned source). Each base's pass is the one the
  worker makes: `refine_batch.plan(record, running=, profile=, scale=,
  strength=, steps=, include_foreign=False, skip_passes=False)`, the call in
  `JobManager._passes` (`jobs.py:1400-1408`). It is missing where
  `submission.not_yet_rendered` at the pass size says so
  (`backend/submission.py:346`, Mass refine's own check,
  `backend/api_refine_batch.py:89-113`) and `not_yet_planned` keeps it.
  So a pass Mass refine already made of that base counts as made, and
  nothing is made twice.
- **One spec per run.** A Resume tick with other numbers continues the
  same base run with another spec, and so another pass run. Where the
  run's retained jobs disagree on the spec, Finish refuses in words, as it
  refuses a run made on another checkpoint (`api_generate.py:268-271`):
  "this run's passes were made at 1.5× at 0.4 and at 2× at 0.35; Mass
  refine over run:X recovers either". Where they agree, that spec is what
  Finish passes back, so the passes land under the same
  `then_refine.pass_run_id(base_run, spec)` (`backend/then_refine.py:85-93`,
  deterministic) as the passes already made.
- **How the job runs them.** `Job` gains `pass_sources: list[int]`, the
  base gallery ids whose pass is missing, and `submit_cells` the matching
  keyword. The worker runs each as a unit of its own before the first
  base cell, through `_passes` with the base re-read by id
  (`storage.get_image`, `backend/storage.py:1826`) at its turn; a base
  binned or deleted since the preview is a failed pass with that reason,
  never a crash. Pause and the budget are checked before each such unit,
  as before a base cell; cancel stops inside one; its failures count under
  the pass streak and `_units` (`jobs.py:1379-1384`) adds them to the
  share. Their pass cells are in `planned_keys` from submit (as
  `pending_passes` are, `jobs.py:601`), so a second Finish or a Mass
  refine meanwhile does not queue them again. A Finish with missing passes
  and no missing base cell is a job with no cells: every reader of
  `job.cells[0]` or `job.request` (the stand-in capability check at
  `jobs.py:1210-1211`, the retained definition's size and batch at
  `:920-925`, the snapshot) reads the first pass instead. The retained
  definition keeps its cells' format (the missing base cells, possibly
  none) and adds `pass_sources` to the metadata; the pass run still keeps
  no definition of its own, so its Finish keeps naming its base.
- **The preview.** `FinishRunPreview` (`backend/schemas_rerun.py:88-100`)
  gains `passes`, the missing passes it would queue; `available` holds
  where cells or passes remain; "nothing in this run is missing" only when
  both are none, and "already queued" covers queued passes. The pin hashes
  the pass keys beside the cells' (`api_generate.py:326-329`), so a pass
  landing between preview and press is the existing 409. The page's words
  (`finishRunWords`, `web/runs.js:25-31`) add "and N missing passes"; the
  hold stays `finishRunHold`'s `aria-disabled` with its reason as the title
  (`web/runs.js:33-40`; AGENTS.md, Frontend). The text sits on the existing
  line: nothing moves under 1800 px.

**Constraints that apply.** The server decides what is missing; the page
only words it. Base rows, their `CellKey`, the base run id and `applied()`
are untouched (AGENTS.md, Schedule and profiles; Reproduction and
storage): a pass is still built only from a saved record through
`grid.cell_from_record`, inside `refine_batch.plan`. Every terminal
transition still goes through `_finish`; nothing new is queued from it.
Binned rows are read for keys but never refined.

**Non-goals.** The grid panel's Resume tick does not add missing passes (it
submits the panel's cells, not the gallery's); passes for a plain job that
continues the run; a pass of a binned base; reconciling two specs in one
Finish; moving passes Mass refine already made into the pass run.

**Completion criteria.** Done when, with `uv run --group dev pytest`
passing: `tests/test_api_finish_run.py` covers a then-refine run cancelled
during a pass (preview `passes` 1, `cells` 0, available; Finish queues one
pass under the pass run id with Mass refine's key; a second preview says
already queued), a binned base skipped, a pass Mass refine already made
counted, two specs refused in words, the pin changing when a pass lands,
and a source deleted between preview and its turn reported as a failed
pass; `tests/test_then_refine.py` covers pass-only units under pause, the
budget, cancel and the failure streak, and `planned_keys` including them;
`smoke-then-refine` gains a step - cancel a batch on its pass (the
`--step 0.3` fake), then the run's Finish run says one missing pass, and a
press adds the pass to the strip and the pass run; `smoke-wide` and
`smoke-narrow` pass unchanged. `docs/design-notes.md`'s "Generate, then
refine" drops "Finish does not yet add a missing pass", and
`docs/rules.md`'s Then refine entry names `pass_sources`. No card stage:
the pass is the refine pass already measured.

#### Phase 4 built

**Built, 2026-10-30.** `finish_plan`
(`backend/api_generate.py`) reads the spec of every retained job carrying
one, refuses two in words ("this run's passes were made at 1.5× at 0.4 and
at 2× at 0.35; Mass refine over run:X recovers either"), builds each live
base's pass with `then_refine.plan_passes` (now the one builder, which
`JobManager._passes` calls too) and keeps those `not_yet_rendered` and
`not_yet_planned` leave; the checks every kept cell gets (budget, steps,
`unusable_input`) run on them, and `refine_unavailable` where a spec is
carried. `FinishRunPreview` and `FinishRunAccepted` gain `passes`; the pin
hashes the pass keys; the held words gain "the missing passes are already
queued" and "the missing cells and passes are already queued". The route
passes `then_refine=plan.spec, pass_sources=[...]` to `submit_cells`, which
plans the sources' passes at submit (`Job.source_passes`, in `planned_keys`
until each unit has run) and retains `pass_sources` in the metadata. The
worker runs each source as a unit before the first cell
(`_stops_before` is the one boundary check for both kinds of unit;
`_source_pass` re-reads the base with `include_binned` so the failure says
"deleted" or "moved to the recycle bin"; `_pass_failed` is the one pass
failure path). `job.run`/`total_runs` count the pass units first; a job
of passes alone takes its request, seed, sizes and estimate from its first
pass and its `one_source` is none. The snapshot adds `pass_sources` and
counts them in `planned_images`; the page's `unitWord` takes them off
before deciding batch or image, and `finishRunWords` and the sheet's run
line add "N missing passes". Deviations: none in behaviour. Two wording
changes outside the plan: the Skip hold during a pass now reads "Then
refine passes cannot be skipped: Finish run adds a pass a stop left
missing" (it said Finish recovered base cells only), with
`smoke-queue-controls` and `docs/plan-queue-controls.md` following; and
the agreed spec is what Finish passes back even where the run's first
retained job was a plain one (the plan's "where they agree, that spec").
After review, one addition: a pass the run made and then permanently
deleted is redrawn as a deleted cell is, with words. Each job retains
`completed_passes` (the bases whose pass it saved) in its outcome; the
preview and the accepted answer gain `redrawn_passes`, and the page says
"N permanently deleted passes will be redrawn" (`finishRunWords`; the
sheet's run line counts them in "N deleted to redraw"). A binned pass
still counts as made; a job retained before this build has no list, so a
pass of its deleted is redrawn as a plain missing pass.
Evidence: `tests/test_api_finish_run.py` (nine tests: cancel on a pass,
binned base, Mass refine made and queued, two specs, the pin, a source
deleted before its turn, a permanently deleted pass redrawn with its
count, a stop's missing pass not counted as redrawn, and a plain first
job not hiding a later job's spec) and `tests/test_then_refine.py` (four: pause with
`planned_keys`, budget, cancel, three failures in a row); the new
`smoke-then-refine` step (a Generate cancelled on its pass, "Finish 1
missing pass of run …", the pass in the strip and the pass run, then that
pass permanently deleted and the words saying it will be redrawn), with
`smoke-wide`, `smoke-narrow`, `smoke-jobs`, `smoke-results` and
`smoke-queue-controls` passing unchanged in their layout.

### Phase 5: More seeds carries the pass

**Status: `GEN#37`, built 2026-10-30 (M, CPU/fake/browser only); added 2026-10-29.**
Built as designed below; what differs and the evidence are in
[Phase 5 built](#phase-5-built) at the section's end.
Matt took Q5's recommendation on 2026-10-24: More seeds makes bases only
for now; "Later, More seeds can read the retained spec." (Q5 below). Matt rates "Make more seeds of an
entire run" (G6) Often (`docs/design/preferences.md:134`).

**Today, read 2026-10-29.** More seeds builds each batch's cells from its
records through `grid.cell_from_record` (`backend/more_seeds.py:1-26`) and
queues one job per run and size with `submit_cells(cells,
run_id=group[0].run_id, more_seeds=count)` (`backend/api_more_seeds.py:314-317`,
grouping at `:162-170`), with no `then_refine`. So More seeds of a
then-refine run adds bases to the base run and no passes; the night's
new images need a Mass refine afterwards. A pass batch (its records'
`refine` set) gets more passes of the same source at new seeds, an
existing meaning this phase keeps.

**Design.**

- **Which batches carry it.** A queued base batch (its template's
  `refine` None) carries the spec of its template's job where that job's
  retained metadata has `then_refine` (`storage.retained_run_metadata`,
  `backend/storage.py:1906-1926`, read as `_retained_then_refine` reads it,
  `api_generate.py:179-184`, moved to a shared place). The template can be
  a base an earlier More seeds job made, and that job carried no spec, so
  the spec is read from the jobs of the batch's mates (`batch.mates`, every
  image of its run at its settings) where they agree, and a batch whose
  mates' jobs disagree is refused in words. A batch none of whose jobs
  carried one is unchanged.
- **Refusals in words, per batch.** `then_refine.refusal` on the batch's
  cells (`backend/then_refine.py:44-82`: a pass past the guard, the pixel
  budget at the Batch it runs at, a strength leaving no step) sets a batch
  status of its own with that detail, as the other per-batch statuses are
  set (`api_more_seeds.py:99-128`); the rest still queue.
- **One spec per job.** `_more_seeds_jobs`' key gains the spec, so each job
  carries one; `submit_cells(..., then_refine=spec)` derives
  `pass_run_id(group run id, spec)` (`jobs.py:892-893`), the same id the
  original passes used, so the new passes join the existing pass run.
- **The request and its replay.** `MoreSeedsRequest`
  (`backend/schemas_rerun.py:311-336`) gains `then_refine: bool = True`:
  carry the run's pass where a batch has one. `_intent_payload`
  (`api_more_seeds.py:184-198`) dumps every field, so a replayed intent
  keeps the choice. The preview's row notes say "each then refined at 1.5×
  at 0.4"; `queued` and `seconds` count the passes, priced at the pass size
  as the grid preview prices them; the dialog's tick **Refine each new
  image** shows where a resolved batch carries a spec, on by default
  (the run's own choice), `aria-disabled` with the reason as its title
  where the checkpoint has no pass. It sits in the More seeds dialog, not
  the panel, so no panel row moves.
- **The cap.** The `max_grid_images` refusal (`api_more_seeds.py:62-71`)
  counts base images, as the press asks for them.

**Constraints that apply.** The new base cells, their seeds (stepping over
taken seeds, `seed + i`) and run id are exactly today's (AGENTS.md,
Reproduction and storage; `seeds.py`); a pass is built only from a saved
record by the worker, never planned as a cell; the server decides which
batch carries a pass and the page only shows it.

**Non-goals.** Carrying a pass for a plain run; a pass of a pass; changing
the numbers in the More seeds dialog (they are the run's spec); a
then-refine choice for Re-render or Swap tags.

**Completion criteria.** Done when, with `uv run --group dev pytest`
passing: `tests/test_api_more_seeds.py` covers More seeds over a
then-refine run queueing bases and their passes under the original pass
run id, `then_refine: false` making bases only, a plain run unchanged, a
pass batch unchanged, a second More seeds (its template an earlier More
seeds base) still carrying the pass, mates of two specs refused in words, a batch past the guard refused in words while the
rest queue, and two specs in one press making two jobs;
`tests/test_more_seeds_intents.py` that a replayed intent keeps the
choice; `smoke-more-seeds` (or `smoke-then-refine`) gains a step - More
seeds of a then-refine run shows the note and the tick, and the strip gets
each new base followed by its pass. `docs/rules.md` and the design notes'
"Generate, then refine" say More seeds carries the pass. No card stage.

#### Phase 5 built

**Built, 2026-10-30.** `api_more_seeds._carry_then_refine` runs after
the per-batch statuses, over the queued base batches (template `refine`
and `edit` None): it reads `storage.retained_run_metadata` for every held
image's `job_id` and each job's spec through `then_refine.retained` (the
Finish reader, moved out of `api_generate.py`), so a template from an
earlier bases-only More seeds still finds the run's. With one spec and
`then_refine.refusal` passing on the batch's cells, `Batch.then_refine`
is set and the row notes "each then refined at 1.5× at 0.4"; two specs, or
a refusal, set status `then_refine` with the words ("its images were
refined at 1.5× at 0.4 and at 2× at 0.35; untick Refine each new image for
bases only, and Mass refine them at the numbers wanted", or "its run's
pass: " and the refusal), and the rest queue. `_more_seeds_jobs` keys by
`then_refine.spec_key` too, and each job is `submit_cells(...,
then_refine=spec)`, so its `pass_run_id` is the original pass run's.
`MoreSeedsRequest.then_refine` (default True) rides in the intent payload.
The preview prices the carried passes at their size
(`then_refine.execution_units`). Deviations: `queued` keeps counting the
new bases - the cap and the page's "N additional images" read it - and the
preview and the accepted answer gain `passes` beside it, as Finish's
preview did, rather than folding passes into `queued`; the preview also
answers `then_refine_batches` (batches whose run carries a spec, carried
or not: where the tick shows) and `then_refine_hold` (no refine pass on
the checkpoint: the tick held and unticked, those batches making bases
with the note "bases only: this checkpoint has no refine pass", rather
than each refused); and the run's `seeds` action (`RunDetails`) asks bases
only, since a batch refused for its pass can still have more with the
tick off. The dialog's tick sits beside the count on the same row, so
neither its showing nor a toggle moves anything; the count line and Queue
add "+ N refine passes".
Evidence: `tests/test_api_more_seeds.py` (seven tests: bases and passes
under the original pass run with the passes priced at the pass size;
`then_refine: false` making bases only and a press from that bases-only
template still carrying it; a plain run and a pass batch unchanged; no
pass on the checkpoint held with bases only; mates of two specs refused in
words, the run's More seeds action still available and `false` queueing;
a batch past a stubbed guard refused while the other queues; two specs in
one press making two jobs, each in its pass run) and
`tests/test_more_seeds_intents.py` (a replayed intent keeps the choice and
a body that would now carry the pass is refused); `smoke-then-refine`'s new
step (More seeds of the batch of two: the tick shown and on, the note, "+ 2
refine passes" in the count line and Queue, unticking dropping them with
the tick, count, Queue and list unmoved, and the job putting each new base
then its pass in the strip, the passes in the original pass run), its
screenshot looked at; `smoke-more-seeds` passing unchanged.

**Follow-up, 2026-10-31 (review).** Two gaps the review left are closed.
The bases-only note is the per-batch refusal in words this design asks
for, so it must show; it did not, since the page unticks its held tick
and asks again with `then_refine: false`, and `_carry_then_refine`
returned on `carry` off before noting it. The server now asks
`refine_unavailable` first and notes "bases only: this checkpoint has no
refine pass" whatever the tick says (the page's re-check answers it too;
`then_refine_hold` unchanged). A held tick's press also lost its words at
the next Queue sync (`syncMoreSeedsRun` cleared the error line whenever
Queue could go); it now keeps them while the tick is still held for that
reason. The fake gains `POST /api/fake/refine` (`supported`), which takes
the claimed checkpoint's pass away, so the no-pass case runs through the
real preview rather than a stubbed answer. Evidence:
`tests/test_api_more_seeds.py`'s no-pass test also asks with
`then_refine: false` and gets the note; `smoke-then-refine` gains two
steps - a press held on its POST, the tick `aria-disabled` (not
`disabled`) with "This request has been sent with its choice; close More
seeds to make another" as its title and, pressed, in the error line; and
the pass taken away, the tick shown held and off with the server's reason
as its title, the row's note, no pass counted, a press saying the reason,
nothing moving. Before the fixes the note and the press's words failed
there (the two expectations), the rest passed; after, the smoke passes,
its new screenshot looked at, and `smoke-more-seeds` passes.

## Decision: several strengths per image

**Decided 2026-10-29: no.** Matt declined a strength list; one strength stays one strength, and a sweep stays Refine's Compare or one Then refine run per strength. Kept for the record (`GEN#39` retired).

**Was: `GEN#39`, DECISION, added 2026-10-29: Matt's yes or no.** Matt
took Q2's recommendation on 2026-10-24: "one strength in phase 1, a list in
phase 3 if wanted - it triples a night's time" (Q2 below). Matt
rates "Try several strengths or versions" (H4) Often
(`docs/design/preferences.md:146`). Nothing is built until he says yes.

**What a yes means.** Then refine's Strength takes a list ("0.3,0.4,0.5")
and every image gets one pass per strength, in order, right after its
cell: a strength sheet of every base. **It multiplies a night's pass time
by the list's length**: three strengths are three passes of every image
where there is one today, and three times the passes' disk; a budget
night completes correspondingly fewer seeds. Each strength's passes would
keep the one-strength pass run id of that strength
(`then_refine.pass_run_id` hashes the strength, `backend/then_refine.py:85-93`),
so a list of one is today's run exactly and no existing run id changes.
Mass refine keeps refusing a list ("One strength for a batch",
`docs/design-notes.md:5809-5815`); this would be the follow-on's only
strength sweep.

**What a no means.** One strength stays one strength; a sweep stays one
image's Refine dialog (its Compare) or a Then refine run per strength.

**Completion criteria.** The decision closes when Matt answers. On a yes,
it becomes an OPEN row of its own with its plan here (the list's
validation and refusals, the worker's per-strength passes, the preview's
price, the opener's words), checked on the fake like phases 4 and 5.

## Risks

- **A long night's passes at a size the card does not hold.** A 1.5x pass
  of a 1.5 MP FLUX.1 base is past the measured 2.4 MP line; the fold warns,
  and three failed passes stop the job, but each failure costs its time.
- **Forgotten tick.** Mitigated by the labels and summary; Q1 decides
  whether it is remembered at all.
- **Two runs for one press.** The base run and the pass run are separate
  in the Runs picker; the run bar of each names the other.
- **The worker loop grows a stage.** The stage reuses `generate` and
  `_save` and adds no new terminal path; every terminal transition still
  goes through `_finish`.

## Open questions for Matt

Answered 2026-10-24: every recommendation taken - the numbers remembered
and the tick off on reload; one strength; pass failures under the cells'
failure rule; missing passes in Finish in a later phase, Mass refine over
`run:<id>` until then; More seeds makes bases only; steps but no prompt of
the pass's own; Lanczos only. 2026-10-29: Q4's later phase is
[phase 4](#phase-4-finish-run-adds-missing-passes) (`GEN#35`), Q5's is
[phase 5](#phase-5-more-seeds-carries-the-pass) (`GEN#37`), and Q2's "if
wanted" is [the open decision](#decision-several-strengths-per-image)
(`GEN#39`).

1. **Remembered or per session?** Is the tick saved with the panel, or off
   on every reload? Recommended: the numbers remembered (shared with the
   Refine dialog's), the tick off on reload, so a fresh page never doubles
   a press unasked.
2. **One strength or several per image?** Mass refine refuses a list
   ("One strength for a batch", `docs/design-notes.md:5809-5815`); a
   follow-on could run "0.3,0.4,0.5" as three passes per image, a strength
   sheet per base (H4, "Often"). Recommended: one strength in phase 1, a
   list in phase 3 if wanted - it triples a night's time.
3. **Do pass failures stop the night?** Recommended: yes, through the same
   three-in-a-row / a-third rule as cells, since a pass that fails three
   times at one size will fail all night; the bases already made are kept.
   The alternative is to drop the follow-on for the rest of the job and go
   on with bases alone.
4. **Missing passes in Finish run?** Recommended: phase 2 (or 3) adds them
   to Finish's plan; until then Mass refine over `run:<id>` recovers them
   with identical rows.
5. **More seeds of a then-refine run?** Recommended: not in phase 1 - More
   seeds stays "same settings, more seeds" and makes bases only; refine the
   new ones with Mass refine. Later, More seeds can read the retained spec.
6. **A pass prompt of its own?** The ask allowed "the pass's own
   steps/prompt defaulting to the cell's". Recommended: steps yes (Mass
   refine's box), prompt no for now - "Refine with a revised prompt" is
   "Sometimes" (`docs/design/preferences.md:145`) and belongs to one
   image's Refine dialog.
7. **The GAN upscaler on WAI v17 and RouWei?** Recommended: Lanczos only,
   as Mass refine; the opt-in GAN stays the single dialog's.
