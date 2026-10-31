# Size axis and Refine crop anchor (GEN#19)

**Status: done 2026-10-30** (`GEN#19` 2026-10-26, `UI#34` 2026-10-30).

Owner: [2026-10-24 review, G8](review-2026-10-24.md#g8-size-as-a-grid-axis-and-a-crop-anchor-for-refine).

## Scope and ordering

A grid may cross paired canvas sizes, and an ordinary Refine may choose which
part of a differently shaped source it keeps. Size and anchor join the cell's
reproduction identity together, in one key extension. This is one feature:
neither a size grid whose Resume confuses its sizes nor a crop control whose
position is lost by a rerun is complete.

Implement after GEN#17's random-variable core has been checked and committed;
both change the grid, request schema, API and CLI. Preserve concurrent work.
No checkpoint download, model loading or GPU run is needed for implementation
or fake/CPU verification. Real model quality, timing and memory claims still
belong in `verified.md` after an authorized card run.

## Request and execution contract

- `GridRequest.size_axis` is a bounded string of comma-separated `WIDTHxHEIGHT`
  pairs, with whitespace around a pair or `x` allowed. There are no implicit
  ranges or separate width/height products. Blank means the base request's
  size. Preview values use canonical `WIDTHxHEIGHT` strings under `size`.
  Keep pair order and count repeated pairs honestly; the duplicate advisory
  can explain repeated cells rather than silently changing the request.
- Every concrete cell is a validated `GenerateRequest`. All existing side,
  pixel, batch, Refine and inpaint/outpaint guards apply to each size. The
  whole-grid ceiling includes the size factor before materializing cells.
  A mask/outpaint cannot be resized by choosing another grid size: those
  remain bound to their source/spec and are refused with the existing reason.
- Seeds remain outermost. Within one image seed and resolved prompt, cross
  size with the sampling axes. Each size uses the same noise seed, and random
  variables draw the same assignment across sizes. GEN#17's per-image cells
  and seed-group budget contract remain intact.
- Add `GenerateRequest.crop_anchor`, a choice of `center`, `top`, `bottom`,
  `left`, `right`, `top-left`, `top-right`, `bottom-left`, `bottom-right`.
  `center` is the default and has exactly today's behavior. Noncenter values
  require an ordinary Refine source and are refused for masks/outpaint.
  When `applied()` drops a source, it resets its anchor too. The anchor is
  one selected setting, not another grid axis.
- Add `--grid-size` and `--crop-anchor`, including direct CLI and
  `--via-server`, and carry the anchor from a PNG whose Refine source can be
  reproduced. Do not reinterpret the panel's width/height as separate axes.

## One identity extension, preserving old runs

Append one canonical size/anchor element immediately before the seed in
`CellKey`; increment `_KEY_EXTENSIONS` once. Build that element through one
helper from both `grid.cell_key` and `storage.completed_cells`. The size is
always present in the comparison key. An absent/legacy anchor means center.
Keep existing element positions before the new extension, including the
refine element that the duplicate advisory replaces with a schedule index.

`grid.run_id` needs explicit legacy handling. For a uniform-size, centered
run, omit the new element from the cell's hashed name and retain the old
definition's `size` field byte for byte. Every pinned plain/conditioning/
Refine/mask/outpaint/upscaler run hash must survive. A noncenter anchor or
mixed sizes carries the new element in the hashed cells. Mixed-size runs use
an order-independent size definition rather than the first cell's size, so
reversing the size axis cannot rename the same set of cells. Do not use the
first size to scope an entire mixed run anywhere else.

Resume reads gallery keys per actual size, using
`submission.cells_by_size`, then compares every cell through `applied()`.
Keep the planned-key snapshot before awaited gallery reads. Live planned
keys already have size groups; retained Finish already groups its concrete
cells by size. Reuse those paths. Update direct CLI Resume likewise.

## Crop arithmetic and persistence

Extend `refine.fitted` and `source_image` with the anchor; both ordinary and
GAN-prepared source branches pass it. Keep today's resize-to-cover and
Python rounding exactly. Center uses the existing floor-half excess on each
axis; edges/corners select zero or the full excess. Do not replace this with
another image-fit routine whose rounding would change legacy center pixels.

Persist a nullable `crop_anchor` column, with `_migrate`, `_COLUMNS`, explicit
`save_image` and `index_file` values, and unguarded `_row_to_record` reading.
NULL/absent means center. Validate the finite choices and ordinary-source
relationship in saved/imported data. Carry the anchor in ImageRecord, PNG
metadata, retained cells, `grid.cell_from_record`, Re-render, Swap tags,
More seeds, CLI `--from-image`, Reuse, reindex and bundles. Preserve it when
rerunning the same source; reset it when Mass refine chooses a new source or
an unresolved/deleted provenance link is dropped. Load settings from PNG
leaves it out with the other source-dependent settings. Bundle row checks,
metadata rewrites, staging and remapping must agree. Legacy files need no
new anchor value. Existing width/height columns need no migration.

## Preview, runs and source views

Counts and labels include the size axis. Estimates may never price all cells
from the base request's size. Gather matching direct/refine/inpaint costs and
PNG-size samples per concrete size, then price execution units in their real
order. If any unit lacks matching history, its time/budget estimate remains
unknown; never borrow another size's measurement. Sum per-size disk estimates
only when all required sizes have samples. If a staged implementation cannot
yet price mixed sizes, explicitly return unknown estimates for those grids
until that step is completed.

Then refine already creates a pass from each concrete base request. Expose
all resulting pass sizes, keeping the old scalar `pass_width`/`pass_height`
only when they agree; do not advertise the first stand-in's size for every
pass. Include every pass's own size in cost and disk accounting.

Add the run's distinct size list/count to live and recovered RunSummary and
RunDetails. Existing scalar dimensions may remain a representative legacy
field, but updated UI text and decisions use the distinct sizes. Retained
cell requests, not `run_jobs`' scalar header, are the authoritative definition.
Make Size a contact-sheet axis, with captions/export headings that describe
mixed sizes. Keep the shared cell aspect ratio and letterboxing behavior;
one image must not hide another merely because its size differed.

Refine's kept-region frame and wording, source comparison in the lightbox,
and sheet source thumbnails all show the chosen crop. Extend the shared
`refineKept` arithmetic with the same anchor as Python and pass each record's
anchor at every source-view call site. The existing center path remains
pixel-aligned with the server. Facts and Reuse expose/carry the actual anchor.

Add the size box inside the existing grid fold and a crop control in the
Refine dialog. Use existing settings persistence and request/preview paths;
the server remains the authority for parsing and expansion. No new line in
the main prompt/Generate area is necessary.

## Phases and checks

1. Schema, paired-size parsing and shape-key helper; anchor arithmetic and
   persistence round trips. Preserve all legacy run hashes. A partial phase
   does not close GEN#19.
2. Multi-size expansion, API/CLI Resume, retained work and preview/run
   reporting. Update source views and UI controls against the settled API.
3. Focused tests, review, browser verification, docs and a scoped commit.
   Record what was verified in this plan/design notes, then remove TODO's
   row only when the complete contract is implemented and checked.

CPU tests should cover:

- Paired-size syntax, order, blanks, repeated values and every size guard;
  cross/random modes, seed lists/wrap, count ceilings and batch behavior.
- Pinned legacy run hashes; mixed-size order independence; distinct keys,
  duplicate counts and gallery/queue Resume across equal settings at
  different sizes or anchors; partial batch and retained Finish behavior.
- Synthetic color-band/checker images proving each crop position and odd
  resize rounding, equal-shape no-op and GAN source preparation.
- Anchor normalization/refusals; row/PNG/retained/rerun/reindex/bundle round
  trips; legacy defaults and provenance/source-loss cleanup.
- Per-size estimates and unknown-history handling; Then refine pass sizes;
  live/recovered run summaries and CLI direct/server parity.

Browser smokes should cover a mixed-size grid preview/run/Resume, mode and
size persistence, the Size sheet axis, Refine's chosen frame and Reuse, and
source comparison at that same crop. Reuse/extend the grid, Refine, sheet,
lightbox and More seeds smokes as appropriate; add any new smoke to the port
table and `rules.md` Tests map. Run checks serially to bound machine use.

## Status

Implemented 2026-10-25/26 after GEN#17's checked commit `e3cc9bf`.
Size/crop requests, reproduction keys, persistence, reruns, estimates, CLI
and browser wiring are covered by the full CPU suite: **3,721 passed,
5 skipped**, with Ruff clean. The dedicated `smoke-size-crop` passed using
isolated fake data: mixed-size preview, run and Resume; settings persistence;
Size/Crop sheet axes; noncenter and legacy-center Reuse; and crop/letterbox
geometry and synthetic pixel checks in displayed and exported source views.
Adjacent `smoke-grid`, `smoke-refine`, `smoke-sheet` and `smoke-lightbox`
also passed. The last two now assert the new Size and Crop anchor fields;
independent review confirmed those exact-label changes. Documentation and
smoke-runner checks passed (10 tests). No models were loaded or downloaded;
CPU/browser checks do not establish GPU pixels, timing or memory.

A static review found and corrected a repeated-import edge case: a child-only
selection drops its absent source and crop, and a later full archive must keep
that known row unchanged even when it includes the source. A regression for
selection then full import passed in the full CPU suite.

Static and browser review also corrected per-slice source views/exports,
source-aspect letterboxing, a duplicate picked Size label, and centered Reuse
after a remembered noncenter anchor. Tests retain the existing legacy run
hashes while asserting the new comparison keys.

The existing pinned legacy grid test keeps its original run-id assertion;
its comparison-key expectation now includes the size/center extension. This
updates the new key contract without changing the legacy hash expectation.

## Compare scales in the Refine dialog

**Status:** Done 2026-10-30 (UI#34, added 2026-10-29 from that day's TODO
audit, item 10): stages 1 and 2 are built and pass; stage 3, the card's
quality judgement at 2x, is not run and does not hold closure (below).

**What was built.** `web/refine.js`: `REFINE_COMPARE_SCALES = [1, 1.5, 2]`,
`compareScales` (each factor's `refineScaled` size, ascending, less any
`refineFits` refuses or `refineTight` puts past the line, each named),
the `refineComparing` mark, `toggleRefineScaleCompare` (held with the reason
under two factors, a press saying it with focus kept) and
`leaveRefineScaleCompare`; `refineRequest` sends `/api/grid` with
`size_axis` (and `strength_axis` for a list) over the boxes' base size;
`previewRefine` names the server's `values.size`; the run's placeholder
names the sizes; `rememberRefine` stores no scale from a comparison;
`openRefine` and `applyRefineScale` clear the mark, and `web/app.js` clears it
on a typed W or H. While comparing, the size line reads "Comparing 1×
1024×1024, 1.5× 1536×1536", no scale button is pressed and the seed line
says "at every size". The button is `#refine-compare-scales` in
`web/index.html`.

**Deviation.** The button sits in the Size field's caption, floated right
(`web/styles/refine.css`), not in the scale row: the four factor buttons
fill the row with their size lines, and a fifth would wrap it. Its block
margins hold the caption to its own line height, so the field and the
dialog keep their height (smoke-checked by hiding it); nothing under
1800 px moves. A second press also leaves the comparison (the button is
`aria-pressed`).

**Evidence.** `tests/test_profiles.py::test_compare_scales_sends_the_scale_buttons_own_sizes`
holds the factors to `REFINE_SCALES`, the page's arithmetic to
`refine.scaled` and the pairs to `grid.parse_sizes`;
`tests/test_api_refine.py::test_compare_scales_is_a_grid_over_the_scale_buttons_sizes`
sends `refine.scaled` pairs over an ordinary source to `/api/grid/preview`
and `/api/grid` (one cell per size and seed, six distinct `CellKey`s, a
resume finding all six, an unknown estimate, the pass's own pixel budget
past a render's) and is refused with a `mask` or an `outpaint` on both
routes. `smoke-refine` (FLUX.1 Krea dev fake: 2x named as past the measured
line, the same heights with the button hidden, the preview's sizes, a
strength list crossed, the sheet's Size axis, the last single scale on
reopen, a scale button, a typed W and a second press leaving it, Reuse of a
compared cell, held on a 1.5 MP source) and `smoke-refine-sdxl` (all three
inside Animagine's line; held on the generic line) pass, with
`smoke-button-states`, `smoke-grid` and `smoke-sheet`. Ruff, the type gate
and the full suite pass. No model was loaded: nothing here says how 2x
looks or what it costs on the card.

### Why scales cannot be compared today

`docs/design/preferences.md` rates H2 (enlarge an image using Refine) and H4
(try several strengths or versions) as Often. The dialog sweeps only one of
the two. A strength list becomes a grid's `strength_axis`
(`web/refine.js:757`,
`{ ...base, num_images: 1, seeds: versions, strength_axis: strength }`),
while the scale is a single choice: the buttons built in `syncRefine`
(`web/refine.js:475` onwards, from `REFINE_SCALES = [1, 1.25, 1.5, 2]` in
`web/state.js:534`) write one width and height into the boxes
(`applyRefineScale`). To see 1x beside 1.5x and 2x of one keeper, the user
runs the dialog once per scale and compares the results by hand.

The server side already exists. GEN#19 (this plan) added
`GridRequest.size_axis` (`backend/schemas_generate.py:826`), parsed by
`grid.parse_sizes` (`backend/grid.py:125`) into ordered `WIDTHxHEIGHT` pairs.
Each pair becomes a validated `GenerateRequest` under the refine guard
(`refine_max_side`/`refine_max_pixels`), has its own `shape_key` element in
`CellKey`, is priced per size, and appears as Size on the contact sheet.
`/api/grid` and `/api/grid/preview` both expand through `expand_grid`, which
hands it to `grid.expand` and `grid.axis_values`
(`backend/api_generate.py:88`, `:97`), then holds each size's group to its
inputs with `submission.refuse_unusable_inputs`. In `web/`, only `grid-preview.js`,
`generation-presets.js` and `setup-import.js` mention `size_axis` today; `refine.js` does not.

### Design

**Decided 2026-10-29:** factors past the tight line are left out, not
offered with a warning (Matt took the recommendation).

- **One button, no new row.** Add a "Compare scales" ghost button to the
  existing scale group's row (`#refine-scales`, `web/index.html:1161`), in
  the same style as Compare 3 (`#refine-compare`, `:1202`). A press marks
  the dialog as comparing. It writes nothing into Width, Height or Strength.
  The dialog's height must not change at 1400x900, and nothing under 1800 px
  may move beyond what `docs/rules.md` already allows. If the row has no
  room, the button goes inside the existing size-line caption. It never
  becomes a new line.
- **Sizes come from `refine.scaled`.** The compared factors are 1, 1.5 and 2,
  each turned into a size by `refineScaled` (`web/refine.js:93`), the page's
  copy of `backend/refine.py:466` `scaled`. Each pair is written as
  canonical `WIDTHxHEIGHT`, in ascending order. The page never invents a
  size that the scale buttons could not produce.
- **Only factors that pass the guard, and none past the tight line.** A
  factor is left out when `refineFits` (`web/refine.js:98`) refuses its size.
  It is also left out when `refineTight` says the size may run tight on the
  card, which matches how `openingRefineSize` (`:119`) never opens the
  dialog past that line. Each left-out factor and its reason go in the
  button's title, as `compareSweep`'s `dropped` does for Compare 3. With
  fewer than two factors left, the button is held: `aria-disabled`, with the
  reason as its title, and a press repeats it. This is the Compare 3
  pattern, not `disabled`; the scale buttons past the guard remain the
  documented `disabled` exception (`docs/rules.md`, the held-button and
  size-strip paragraphs).
- **The request.** While comparing, `refineRequest` (`:726`) returns the
  grid shape: `path: '/api/grid'`, `num_images: 1`, `seeds: versions`,
  `size_axis` set to the pairs, and `strength_axis` only when Strength holds
  a list. A strength sweep and a scale comparison may cross. The server
  expands, prices and refuses the cross; the page adds presentation only
  ("the server decides"). The base `width`/`height` stay at the boxes' size,
  so the request is valid on its own. `crop_anchor` is the dialog's single
  anchor for every size; it is never an axis.
- **Preview, run and sheet.** `previewRefine` (`:771`) posts the same body
  to `/api/grid/preview`. The preview line reads the sizes from the server's
  `values.size`, and its counts and estimate from the reply. An estimate
  stays unknown when any size lacks matching history (this plan's per-size
  pricing rule). The run uses the existing sweep path, `sheet: true`
  (`web/refine.js:891`), so the result opens as a contact sheet with Size as
  an axis, next to the source thumbnails the sheet already shows.
- **Leaving the comparison.** Pressing a scale button, editing Width or
  Height, or Reuse returns the dialog to a single size and clears the mark.
  The size line names the compared sizes while the mark is set. The kept
  frame continues to follow the boxes' size, since a uniform scale keeps the
  whole source apart from 16-pixel rounding.
- **Never remembered as the scale.** `rememberRefine` (`:983`) stores
  `prefs.scale` only from a single size the user chose. A compared list is
  not stored in `REFINE_KEY` or anywhere else, the same as a strength sweep
  ("A sweep is not remembered", `web/refine.js:24`, `:701-702`). Reusing a
  compared cell opens the dialog at that cell's own single size.

Rules that apply: identity and normalisation are the server's, through
`applied()` on each concrete cell. `CellKey` and `run_id` are those GEN#19
already defines (the `shape_key` extension); there is no new key element,
and no existing run id changes. A held button is `aria-disabled` with its
reason. No layout moves under 1800 px. All `localStorage` access stays in
the existing try/catch helpers. The dialog writes no prompt or negative, so
no `writeBox` path is involved.

### Non-goals

- No arbitrary scale entry and no factors beyond `REFINE_SCALES`. A custom
  size list is already available as the grid fold's Size box.
- No upscaler comparison (with versus without Anime6B) and no crop-anchor
  comparison. Each would be a separate axis decision.
- No change to Mass refine, Then refine or the inpaint and outpaint dialogs.
  A mask or outpaint stays bound to its source's size, and the server keeps
  refusing it.
- No claims about pixels, VRAM or timing from CPU or browser checks.

### Stages and completion criteria

1. **Page and CPU checks.** Add the button, the comparing mark, the factor
   filter with its held state, and the request shape. A pytest in the style
   of `tests/test_profiles.py`'s source-string checks holds the dialog's
   size list to `refineScaled`/`refine.scaled` and to the 1, 1.5, 2 factors.
   A `tests/test_api_refine.py` case sends `size_axis` pairs from
   `refine.scaled` with an ordinary refine source to `/api/grid/preview` and
   `/api/grid`. It checks one cell per size and seed, distinct `CellKey`s,
   the budget, and a refusal when the request also carries a `mask` or an
   `outpaint`. Ruff, the type
   gate and the full suite must pass.
2. **Browser.** Extend `smoke-refine` (port 7875, FLUX.1-Krea-dev fake) and,
   where SDXL sizes differ, `smoke-refine-sdxl`. Steps: the button's title
   names any left-out factor; on a source where fewer than two factors fit,
   the button is held, a press says why and focus stays; the preview lists
   the server's sizes; the run opens a sheet with a Size axis; reopening the
   dialog keeps the last single scale, not the list; a scale button press
   leaves the comparison; the dialog is the same height with and without the
   button. `smoke-button-states` must still pass. Close UI#34 when stages 1
   and 2 pass and this section records them.
3. **Card (needs Matt's OK).** On one FLUX.1 Krea dev keeper and one SDXL
   keeper, compare 1x, 1.5x and 2x at the profile strength. Record in
   `docs/verified.md` whether 2x keeps the composition, and the peak memory
   at each size. Use the existing RAM and VRAM guards, with one load per
   session. This stage does not hold UI#34's closure; if wanted, it becomes
   a CARD row of its own.
