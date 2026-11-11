# Random-choice prompt variables (GEN#17)

**Status: done 2026-10-25** (`GEN#17`).

## Contract

The existing `name = a | b` block has two modes: `cross` (the default,
unchanged Cartesian product) and `random` (one assignment per image seed).
`GridRequest.prompt_mode` and the CLI's `--grid-prompt-mode` carry the mode.
Random mode requires a nonempty variable block and cannot accompany whole
prompt variants. Existing placeholder/value validation and `none` substitution
apply in both modes.

Each variable is selected by SHA-256 of the UTF-8 JSON array
`["local-diffusion.prompt-choice.v1", seed, name]` (frozen from the app's
old name, since it feeds the draw), using compact separators and
`ensure_ascii=False`; its big-endian digest modulo the number of values
is the selected index. The variable name makes a draw independent of variable
line order, other variables, and sampling-axis order. The value order remains
meaningful. No process hash or global random generator participates.

Random mode expands Batch N into N single-image cells, at the ordinary
wrapped `base + offset` seeds. The seed axis still advances by N and remains
outermost; each image's complete parameter sweep follows before the next
image. This allows each cell's scalar prompt and variables to describe exactly
what the backend runs. The saved batch is therefore 1. The preview and CLI
say these images run separately. Time budgets still count complete seed-axis
groups, each containing N images times the sampling product.
The per-call batch pixel guard checks one image; the whole-grid image ceiling
still counts all N images and all sampling combinations.

## Identity, preview and retained work

Choices are resolved before submission. Existing `Cell.variables`, gallery
rows, PNG chunks and retained cells carry the chosen values; no column or key
extension is needed. Resume, queued-cell detection, duplicates and run IDs read
the concrete prompt pair and image seed. Cartesian cells and identities stay
unchanged. Re-render, Finish and More seeds keep a saved image's resolved
prompt/assignment, rather than recovering an absent variable template.

Expand once per API request. Random preview prompt lists come from the full
expanded cells, before Resume pruning, never from a second draw. If neither a
base seed nor a seed list is specified, `GridPreview.prompt_sample` is true:
the listed prompts are examples and Run grid draws another base. The UI must
say this explicitly; counts remain exact. Once queued, the ordinary accepted
seed provides the fixed base for Resume.

The random summary does not multiply by the count of sampled prompts. Image
totals sum cell batches, and random-mode gallery/queued/skipped cell counts
already count individual images. Variable products are never materialized in
random mode; the actual seed/batch/sampling product is checked before cells
are built.

## Implementation and checks

Backend/core ownership: `grid.py`, `schemas_generate.py`, `api_generate.py`,
`generate.py`, focused grid/API/CLI tests and this plan. The page owns the mode
control, payload/persistence, sample/separate-execution notice and presentation.
Queue implementation and gallery schema stay outside this change.

CPU checks cover fixed draw vectors, line-order independence, prompt/negative
agreement, XML/none, batch offsets and wrap, seed lists, limits, unresolved or
invalid variables, variants, preview counts/sample notice, duplicates after
`applied()`, gallery/queue Resume, saved row/PNG choices, CLI parity and
unchanged Cartesian identity. Extend the existing grid browser smoke for mode
selection/persistence and a fake run. Run focused pytest, correctness lint and
the grid smoke serially; no model download or GPU run is required.

## Status

Implemented and checked 2026-10-25. The focused grid/API/CLI suite passed
285 tests; correctness lint and JavaScript syntax checks passed. An isolated
copy of HEAD plus this feature passed all six docs-map checks and both
`smoke-random-prompts` and the existing `smoke-grid`, serially on Chrome with
scratch fake servers. No model weights or GPU work were used.
