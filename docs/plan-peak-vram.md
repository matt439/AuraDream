# Plan: peak GPU memory per image

**Status, 2026-10-24: phases 1 and 2 built (`GEN#15`); phase 3, the card
check, is Matt's. 2026-10-25: phase 3's SDXL half is done (WAI v17: the
recorded pair is the allocator's, equal to `verified.md`'s refine table,
and agrees across the record, the API and the PNG; `verified.md`, Refine
pass, "Peak per image and Then refine, WAI v17"); its FLUX.1 Krea [dev]
half is not, the session having stopped at the RAM guard during its
load.** 2026-10-29: that RAM-guard prerequisite is met: `GEN#22`'s fixed
loader brought the cold fp8 FLUX.1 Krea [dev] load to a 27.9 GiB working set,
down from 46+ (`docs/verified.md`, "RAM at the load, the fixed loader
(2026-10-27)"), so the one remaining step of `GEN#15` is phase 3's FLUX.1
Krea [dev] render on the card, which needs Matt's OK (one load per session,
GPU and RAM watched). That `GEN#22` session's own render read 12.00 GiB
allocated and 12.63 reserved with 14,947 MiB by `nvidia-smi`, a reference
for the comparison; its section does not record the pair read back from the
record, the API and the PNG, so it does not close phase 3. Matt decided the feature on 2026-10-24 and took every
proposed answer to the open questions at the end ("i like recommended").
Built as written, except: the body of `generate()` moved to
`_generate_images` under the name, the wrapper keeping the signature;
the peak keys are absent from the chunk when unmeasured rather than null,
as `upscaler` is, so an unmeasured chunk is byte-for-byte what it was; and
`bundle_images._validate_row` drops a bad value before validating, so a
bundle is never refused over it. Why it is built so is in
`docs/design-notes.md`'s "Peak memory per image". Every statement about today's behaviour cites the file and
line it was read from on 2026-10-24 (`525cb24`).

The design in one paragraph: the backend resets PyTorch's CUDA peak
counters at the top of each `generate()` call and reads
`max_memory_allocated` and `max_memory_reserved` when it returns; the job
worker and the CLI hand the pair to `storage.save_image`, which writes two
new nullable columns and two keys in the PNG's `auradream` chunk; the
facts' Created row shows it beside the duration. It is a *result*, like
`duration_ms`: no part of the reproduction set, `CellKey`, `run_id`,
`applied()` or duplicate detection, and not searchable. A row or PNG from
before it reads as unknown and shows nothing.

## What it measures, and what it does not

Say this wherever the number is shown or documented:

- `torch.cuda.max_memory_allocated` is the high-water mark of tensors live
  in PyTorch's caching allocator on that device since the last reset. It
  includes the weights already resident when the cell began (the reset sets
  the peak to what is allocated now), so it is "what this process's tensors
  held at most during the cell", which is what `docs/verified.md` already
  calls a peak: "'peak' is what the process held rather than what
  `nvidia-smi` showed" (`docs/verified.md:4-6`), and "'peak' is the process's
  own allocator" (`:627`).
- `max_memory_reserved` adds the allocator's cached-but-free blocks, so it
  is closer to what `nvidia-smi` charges the process. The refine table
  already records both ("3.89 GiB allocated, 4.70 reserved",
  `docs/verified.md:632`).
- Neither sees the CUDA context, cuBLAS/cuDNN workspaces that bypass the
  caching allocator, kernels' own allocations outside torch (Nunchaku's
  are unverified here), the desktop's share of the card, or another
  process (the critic's `llama-server` is one). Above all, neither sees
  Windows WDDM spill as such: an allocation the driver backs with shared
  system memory still counts as allocated, so a spilled run shows a
  plausible peak and a slow time. The head of `verified.md` is why that
  matters (a run within a gigabyte of the top "spills into system RAM and
  slows two to three times", `:7-10`).
- So the recorded peak is a per-image note for comparing runs of this app,
  not a fit test. A CARD run still watches whole-card memory and shared
  GPU memory (Task Manager or `nvidia-smi`) as it does today; the recorded
  number does not replace that, and `verified.md` rows still come from
  those runs.

## Where the reset and the read go

### In the backend, not the worker

The worker times a cell around `self.backend.generate(...)`
(`backend/jobs.py:1019`, `:1033-1040`) and the CLI does the same around its
deliberate copy (`backend/generate.py:1710`, `:1723-1728`, `:1750`). Neither
imports torch, and the fake must be able to report a value, so the
measurement belongs to the backend. Add one optional method to
`GenerationBackend` in `backend/backends/base.py`, in the style of
`start_sigma` (an optional answer, None by default, `:140-149`):

```python
def last_peak(self) -> tuple[int, int] | None:
    """(allocated, reserved) bytes at most during the last generate() that
    returned images, or None where this backend cannot say."""
    return None
```

A method rather than a changed return type: `generate()` returning
`list[Image]` is the contract the ABC states (`base.py:62-77`) and the
fakes, the worker, the CLI and the tests all index it as a list. One worker
thread calls `generate()` and then `_save()` in sequence (`jobs.py:1033-1041`),
and the CLI is single-threaded, so "the last call" is unambiguous. The value
is cleared to None at the start of every call and set only on a normal
return, so a cancelled or failed cell never leaves its predecessor's peak
to be saved by the next.

### In `DiffusersBackend.generate()`

Reset immediately after the not-loaded check (`diffusers_backend.py:657-662`)
and before anything that can allocate: the upscaler support check
(`:666-667`), the LoRA hot-swap (`:695`), the Anime6B pre-pass (`:809-813`)
and the pipeline call. Read after the last GPU work. Today the body has two
exits inside `with torch.no_grad()` (`:851-855`, the text-to-image path
returning from inside the block) and one after the composite (`:862`), so
the cleanest change is to move the body to `_generate()` unchanged and wrap
it:

```python
self._peak = None
device = self._peak_device()          # None: nothing to measure
if device is not None:
    torch.cuda.reset_peak_memory_stats(device)
images = self._generate(...)
if device is not None:
    self._peak = (torch.cuda.max_memory_allocated(device),
                  torch.cuda.max_memory_reserved(device))
return images
```

No `synchronize()` is needed: the allocator records an allocation on the
host when it is requested, and the images have already been copied to the
CPU by the time the pipeline returns PIL images. The read is in its own
`try/except` that leaves None, as `describe()`'s memory read is
(`:989-997`): a measurement must never fail a cell.

`_peak_device()` uses the rule `describe()` already uses for "the card the
weights are on" (`:989-993`): CUDA available and (`device.type == "cuda"` or
`offload != "none"`), index from the device or 0. That is the same rule
`load()` refuses on (`:125-131`), so a loaded pipeline with no such device
is the CPU-only case and records None.

### Batches: one peak per call, shared by its rows

A text-to-image batch is one pipeline call (`diffusers_backend.py:707-711`:
"A text-to-image batch is one call"), with one generator per image
(`:747-750`). Its images cannot be told apart in memory, so the peak is
the call's and every row of the batch gets the same pair. That differs
from the duration on purpose: `_save` divides the elapsed time by the
image count (`jobs.py:1129`; the CLI's `generate.py:1767`), but a peak is a
maximum and does not divide. The facts say "peak of its batch" when
`batch > 1` so the shared number is not read as each image's own.

A refine batch is one call *per image* (`_refine`, `:867-906`, the loop at
`:896`). Measuring each would mean resetting inside that loop and returning
a list. Not done: the cell is the unit the worker saves, the first image's
VAE encode leaves the VAE resident for the rest (`docs/verified.md:645-648`),
so per-image peaks would differ by call order rather than by image, and one
rule for every batch is easier to read. The refine batch's rows share the
cell's peak like any batch.

### The paths a cell can take

- **Text-to-image**, with or without references: the whole call.
- **Refine, inpaint and outpaint**: the same wrapper covers the sibling
  pipelines (`:684-689`), the composite (CPU), and the end-of-call
  `maybe_free_model_hooks` (`:902-905`).
- **Anime6B pre-pass**: runs inside `generate()` on the device
  (`refine.py:276`, tiles at `:293`) before the VAE, and its net and tiles
  are released before the pipeline runs (`refine.py:255-258`). It is inside
  the reset, so its tiles count if they peak higher than the denoise. That
  is the honest number for the cell; `verified.md` already notes a refine
  whose peak "is Lanczos's" (`:794`).
- **The LoRA swap** counts too (it is after the reset): a first cell under a
  new adapter can peak above its repeats. Documented, not corrected.
- **The first cell after a load**: the load's own allocations are before
  the reset and do not count; its residency does.

### Offload modes

With `model` offload the peak is the largest component on the card plus
activations; with `sequential` it is lower and the time longer; with
`none` it is everything. The recorded number is whatever the mode made,
which is why `offload` matters to reading it. The row has no offload
column today (`storage.py:142-172`) and this plan does not add one as a
reproduction field; see open question 2.

## Storage

### The row

Two columns, `vram_peak_allocated INTEGER` and `vram_peak_reserved INTEGER`,
in bytes (exact, what torch answers; SQLite's INTEGER holds 64 bits):

- `_SCHEMA` (`storage.py:142-172`), and an entry each in `_migrate`'s
  `added` tuple (`:257-296`) with no index - nothing filters on them.
- `_COLUMNS` (`:426-430`), so `_insert_row` writes them.
- `_row_to_record` reads both unguarded (`:364-420`; the rule in its
  docstring, `:369-375`, and `docs/rules.md:886-888`).
- `ImageRecord` gains `vram_peak_allocated: int | None = None` and
  `vram_peak_reserved: int | None = None` beside `duration_ms`
  (`schemas_gallery.py:98`).
- `save_image` takes `peak: tuple[int, int] | None = None` (signature
  `storage.py:484-495`) and writes both in the insert dict (`:626-657`).
- `storage.index_file` (reindex's writer, `:1176-1236`) builds its dict
  explicitly, so it gets both keys, read from the chunk through a helper
  like `_batch` (`:1268-1272`): `_peak(value)` answers a non-negative int
  or None, so a hand-written chunk never refuses a row over a result.

### The chunk

Two keys in `meta` (`storage.py:534-576`), `"vram_peak_allocated"` and
`"vram_peak_reserved"`, null when unknown, beside `"batch"` (`:566`), which
is the precedent for a non-reproduction field in the chunk. Not in the
A1111 `parameters` block (`:569-603`): it is settings for other tools, not
results. Note that `duration_ms` is in no chunk today, which is why a
PNG-only reindex records it as 0 (`storage.py:1187-1190`, `:1212`); the peak
is the first result the chunk carries, so a PNG-only reindex recovers it.

Load settings from PNG builds a `GenerateRequest` from named fields
(`api_png.py:116-136`, `:144-200`), so the new keys are ignored there by
construction; a test holds that.

### Reindex, merges and bundles

- `reindex._database_meta` (`reindex.py:333-377`) rebuilds the chunk for a
  copied file and `merge_database`'s `values` (`:503-532`) the row: both
  carry the pair through `storage._peak`. `_DB_REQUIRED` (`:80-84`) is not
  widened: an older source database simply has neither column.
- Bundles: the staging table gains missing columns by type, inferred from a
  hard-coded set (`bundles.py:773-776`: `INTEGER if column in {"refine",
  "batch"}`); the two new names join that set, or they would be added as
  TEXT. The merge copies `_COLUMNS` (`bundles.py:816`) and
  `bundle_images._validate_row` (`bundle_images.py:128-160`) validates
  through `_row_to_record`; both carry them once `_COLUMNS` does, with
  `_peak` applied to the values so a bad value is NULL rather than a
  refused bundle. `bundle_images._metadata` (`:201-234`) adds the two chunk
  keys beside `"batch"`.

### What it must not touch

`CellKey`, `grid.cell_key`/`cell_keys` (`grid.py:615`, `:680`), `run_id`,
`GenerateRequest.applied()`, `storage.completed_cells` (`storage.py:1023`),
`grid.cell_from_record`, Re-render/Swap/Mass refine/More seeds, the search
grammar (`search.py`; Matt rated a memory search Never) and
`gallery_reports`' estimates. None of them reads a column it is not named
in, so the work is to leave them alone and test that.

## The callers

- **Worker**: in `_run` after `generate()` returns, read
  `peak = self.backend.last_peak()` once (`jobs.py:1033-1041`) and pass it
  to `_save` (`:1104-1110`) and on to every `save_image` of the cell
  (`:1125-1136`).
- **CLI**: the same in `generate.py` after its `backend.generate`
  (`:1723-1728`), into its `save_image` (`:1764-1774`). `tests/test_cli.py`
  holds the two loops equal; it gains the argument. The CLI's printed line
  is not changed (open question 4).
- **Fake backend** (`tests/fakes.py:26`, `generate` at `:175-207`): an
  attribute `peak: tuple[int, int] | None = None` that `last_peak()`
  answers after a successful call, so every existing test and smoke sees
  None and nothing changes on screen.
- **Fake server** (`tools/fake_server.py:104-106`, `DrawingBackend`
  inherits it): a knob in the style of `/api/fake/lora` (`:519`),
  `POST /api/fake/peak {"allocated": n, "reserved": n}` (or `null`), so a
  smoke can show a number without every other smoke's Created row changing.
  The planted rows (`:421-428`) stay None.

## The facts

`factsList`'s Created row is `"<date> · took 12.3s"` (`web/facts.js:130`).
It becomes, when the record has a peak:

> Created  Oct 24, 2026, 10:12 · took 12.3s · 3.89 GiB peak

the number `vram_peak_allocated` in GiB to two places (verified.md's
units), wrapped in a span whose title gives the rest:
"Peak GPU memory PyTorch allocated during the call that made this (4.70 GiB
reserved). Not the whole card: the desktop, the driver and shared-memory
spill are not counted." With `batch > 1` it reads "3.89 GiB peak of its
batch". With none, nothing is added - no "unknown" text on every old row.
The search ⌕ on the Created row stays `date:` only (`docs/rules.md:3739-3745`).

Not in `settingsLine` (`facts.js:15`), which says what *made* an image for
the lightbox description, the delete confirm and the viewer's title
(`lightbox.js:464`, `gallery-actions.js:284`, `viewer.js:378`).

Room: the Created row already wraps at 1400x900 on long dates, and
`smoke-lightbox.json:555` holds Seed/Created/Run inside the meta box; the
added words must keep that passing (the 2026-10 lesson that the page has
no spare height). If it does not, the fallback is the number alone, title
unchanged.

## Docs

- `docs/rules.md`: beside the reproduction-set bullet (`:844-852`), a
  sibling: the peak pair is on the row and in the chunk, no part of the
  reproduction set, CellKey or run id, never searched; per call, shared by
  a batch's rows; None on CPU, before it existed, and from the fake. The
  `_migrate` bullet (`:884-888`) needs nothing new. Facts: the Created row's
  wording. Tests: the new tests and the smoke step.
- `docs/design-notes.md`: a section "Peak memory per image": why per call,
  why bytes, why not divided, the allocator caveat, why not searchable.
- `docs/usage.md`: one paragraph where the lightbox facts are described
  (near `:501-509`), with the caveat in plain words.
- `docs/verified.md`: one sentence under its head (`:4-10`): recorded peaks
  are the allocator's, as this file's are, and a CARD run cites them only
  beside its own whole-card and shared-memory readings.
- `AGENTS.md`: one line under Reproduction and storage ("the peak pair is
  a result, never in CellKey/run_id/applied").

## Tests and the smoke

- `tests/test_storage.py`: a database made with today's schema (no peak
  columns) gains both on connect, its rows read None; `save_image` with a
  peak round-trips row → record.
- PNG round trip (`tests/test_reindex.py`): the chunk carries both keys; a
  PNG-only reindex restores them; an old PNG with no keys and one with
  `"vram_peak_allocated": "lots"` both read None and are indexed.
- `tests/test_jobs.py` / `test_api_jobs.py`: a `FakeBackend` with `peak`
  set yields that pair on every row of a batch of 3; a failed cell after a
  good one saves nothing stale (the fake clears on failure); `peak=None`
  rows read None.
- Identity: `grid.cell_key` and `run_id` of a cell are equal with and
  without a peak on the source record (`tests/test_grid.py`); a Resume over
  rows differing only in peak skips them (`completed_cells`).
- `tests/test_cli.py`: the CLI saves the fake's peak, and the loop-equality
  test covers the new argument.
- Bundles (`tests/test_bundles_merge.py`): export and merge carry both; a
  bundle from before the columns merges with None; the staging table gets
  INTEGER columns.
- `tests/test_api_png.py`: a chunk with peak keys loads the same panel
  settings as one without.
- `DiffusersBackend`: a unit test with a stubbed `torch.cuda` (reset/read
  call order around a stub pipe; `is_available()` False gives None; a read
  that raises gives None and the images still return), as
  `tests/test_api_refine.py:196-201` already builds a real backend over a
  stub.
- Smoke: a step in `smoke-lightbox.json` (port 7884) that posts
  `/api/fake/peak`, generates, and expects the Created row to contain
  "GiB peak" and its title "reserved", plus a batch-of-2 image reading
  "of its batch", and the existing `:555` layout check still passing.

## Phases

Effort on the review's scale: S a day or less, M a few days, L a project.

**Phase 1 - record it (S).**

- [x] `last_peak()` on the ABC; the wrapper and `_peak_device()` in
      `DiffusersBackend`; the fake's `peak`.
- [x] Columns, `_migrate`, `_COLUMNS`, `_row_to_record`, `ImageRecord`,
      `save_image`, the chunk keys, `storage._peak`, `index_file`.
- [x] Worker and CLI pass it through.
- [x] Reindex (both paths) and bundles (`_metadata`, the staging types).
- [x] Every Python test above.

Done when `uv run --group dev pytest` passes, a fresh gallery's rows carry
the pair from the fake, an old database and old PNGs read None, and no
existing run id, `CellKey` or Resume count changes.

**Phase 2 - show it (S).**

- [x] The Created row's words and title in `facts.js`.
- [x] `/api/fake/peak` and the smoke step; `smoke-lightbox` passes whole.
- [x] `rules.md`, `design-notes.md`, `usage.md`, `verified.md`, `AGENTS.md`.

Done when the smoke passes at 1400x900 and nothing changes for a record
without a peak.

**Phase 3 - the card (S, Matt's).** One generation each on an SDXL profile
and FLUX.1 Krea [dev] (one load per session; GPU and RAM as the memory
notes ask), compared with a hand reading of `nvidia-smi` and shared GPU
memory. Done when the recorded allocated/reserved agree with the process's
own allocator figures in the style of `verified.md`'s refine table, and
the result is a row in `verified.md`. Nothing here is claimed until then.
SDXL done 2026-10-25 on WAI v17, against nvidia-smi and the Windows
shared-memory counters rather than a hand reading (the counters Task
Manager shows); FLUX.1 Krea [dev] not yet.

**Later, not designed:** the Refine size warning reading local history.
Today the tight line is a profile's measured pixel count from
`verified.md` (`defaults.refine_tight_pixels`) or the panel's own guard
(`web/refine.js:107-110`). Reading the gallery instead would need a
server-side lookup like `gallery_reports.step_cost_ms` keyed on checkpoint,
quantization, offload (not recorded today), pass kind and size; enough
rows near the asked size to fit a line; a rule for the allocator-vs-card
gap (the desktop's share is not in the number, `verified.md:5-8`); and a
decision about whether a history-based line may loosen the measured one or
only tighten it. That waits for phase 3 and for weeks of rows.

## Open questions for Matt

Answered 2026-10-24: every proposal taken - allocated in the facts with
reserved in the title, no offload column now, a refine batch's rows sharing
the cell's peak, and the CLI's line unchanged.

1. Allocated alone in the facts with reserved in the title (proposed), or
   both visible?
2. Record the offload mode beside the peak (a third result column, not a
   reproduction field), since the same image under `model` and `sequential`
   offload peaks differently? Proposed: not now; the later phase would need
   it.
3. Per-cell peak shared by a refine batch's rows (proposed), or per image
   for refine batches only?
4. Should the CLI print the peak on its per-image line? Proposed: no, to
   keep its output as it is.
