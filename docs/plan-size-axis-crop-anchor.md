# Size axis and Refine crop anchor (GEN#19)

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
