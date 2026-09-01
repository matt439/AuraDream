# Krea Local

Local text-to-image generator on an RTX 5080. FastAPI backend + vanilla JS
frontend, no build step. The default checkpoint is **FLUX.1 Krea [dev]**
(`black-forest-labs/FLUX.1-Krea-dev`), which is click-through gated; **Krea 2
Turbo** (`krea/Krea-2-Turbo`) is what the sizing notes below are written around
and needs manual approval on HuggingFace. Either runs unchanged - set
`KREA_MODEL_ID`.

## Run

```powershell
uv run python -m backend.main      # http://127.0.0.1:7860
uv run --group dev pytest          # no GPU, no weights, a few seconds
```

The model is *not* loaded at startup. The first `POST /api/generate` (or
`POST /api/model/load`) triggers a load that takes a few minutes.

## Layout

- `backend/config.py` — settings, most overridable by `KREA_*` env vars
- `backend/schemas.py` — pydantic request/response models
- `backend/model_defaults.py` — recommended steps/CFG/mu per checkpoint
- `backend/seeds.py` — the seed range and the wrap, shared by server and CLI
- `backend/backends/base.py` — the `GenerationBackend` ABC
- `backend/backends/diffusers_backend.py` — in-process diffusers implementation
- `backend/grid.py` — axis specs (`6-12x2`) and the cell expansion, shared
- `backend/jobs.py` — single-worker queue, SSE progress
- `backend/storage.py` — SQLite index + PNGs with embedded metadata, search filters
- `backend/main.py` — routes
- `backend/generate.py` — headless CLI; drives the backend directly, no server
- `backend/clear.py` — deletes rows, their PNGs and any orphans
- `web/` — served as-is from `/static`

## Constraints that shaped the code

- Krea 2 is ~13B and needs **~25 GB in bf16**, which does not fit 16 GB. The
  default path quantizes transformer + text encoder to float8 via torchao and
  uses `enable_model_cpu_offload()`. Peak VRAM lands near 13 GB.
- Quantization must go through `PipelineQuantizationConfig(quant_mapping=...)`.
  The shorthand `quant_backend="torchao"` form raises `ValueError` on this
  pipeline: the text encoder is a *transformers* model (Qwen3-VL) and its
  `TorchAoConfig.__init__` signature differs from the diffusers one. Both take a
  torchao `AOBaseConfig` **instance** (`Float8WeightOnlyConfig()`), not a string.
- **`sigmas` is an input to the shift, not the shift's result.** `__call__`
  takes `sigmas` and no `mu`, but it derives a mu of its own - 1.15 hard-coded
  for `is_distilled`, `calculate_shift(image_seq_len, ...)` otherwise - and
  passes it to `set_timesteps`, which shifts whatever sigmas it was handed
  whenever `use_dynamic_shifting` is set (it is, in both shipped
  `scheduler_config.json`s). The shift multiplies the odds `s/(1-s)` by `e^mu`,
  so two shifts compose into `mu_a + mu_b`. Pre-shifting the schedule therefore
  *added* to the pipeline's mu rather than replacing it, and every Turbo image
  made before that was fixed ran at 2.30 instead of 1.15. `DiffusersBackend`
  now overrides the value on its way to the scheduler (`_forced_mu`), and only
  falls back to an explicit pre-shifted schedule (`_shifted_sigmas`, the same
  formula: `sigma' = e^mu / (e^mu + (1/sigma - 1))`) for a scheduler that does
  not shift for itself. Assert on `scheduler.sigmas`, never on the list the
  backend computed - see `tests/test_schedule.py`.
- Turbo is distilled: 8 steps, `guidance_scale=0.0`. **Where guidance switches
  on is a property of the pipeline, not the constant 1.0**: `FluxPipeline` runs
  an uncond pass only at `true_cfg_scale > 1`, while `Krea2Pipeline` guides from
  0 up and substitutes `""` for a missing negative prompt - so the old `> 1.0`
  gate silently guided Turbo against an empty prompt in the band `0 < g <= 1`.
  The backend keys the threshold off `"true_cfg_scale" in self._call_params`,
  and `/api/status` publishes it as `defaults.cfg_threshold` for the UI and the
  CLI. Dropping the negative prompt below the threshold saves nothing - neither
  pipeline encodes it outside its guidance branch - it only keeps the recorded
  settings honest about what ran.
- Components are `text_encoder` (Qwen3-VL), `transformer` (Krea2Transformer2DModel),
  `vae` (AutoencoderKLQwenImage). Offload order is `text_encoder->transformer->vae`.
- Loading happens in exactly one place: `main.ensure_loaded()`, which holds an
  asyncio lock across the whole load. Jobs `await` it rather than calling
  `backend.load()` from the worker thread — doing both raced two concurrent
  loads per request and produced duplicate tracebacks. `generate()` raises
  rather than loading a missing pipeline, so that rule is enforced rather than
  merely documented, and `POST /api/model/unload` answers 409 while any job is
  live: unloading under a running job freed nothing (the worker holds its own
  reference) and had the next run of a sweep reload from the gen thread.
- The loader is model-agnostic: `DiffusionPipeline.from_pretrained` resolves the
  class from `model_index.json`, and `_component_libraries()` reads the same file
  to decide which components to quantize and with whose `TorchAoConfig`. FLUX
  layouts therefore pick up `text_encoder_2` (T5) automatically.
- `model_defaults.py` keys recommended steps/CFG/mu off the checkpoint id, not the
  pipeline signature: `Krea2Pipeline` reports 28 steps at CFG 4.5, which is the
  family default and wrong for Turbo. `/api/status` serves the profile and the UI
  seeds its controls from it once, on load.
- `mu` is Turbo-only. FLUX derives its own resolution-dependent shift, so forcing
  Turbo's constant on it degrades output; the profile leaves mu null there.
- On FLUX, `guidance_scale` is a distilled conditioning value, **not** CFG - a
  negative prompt is silently ignored unless `true_cfg_scale > 1.0`. The backend
  sets `true_cfg_scale` from the request's CFG whenever the pipeline exposes it.
- **Use `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline
  call.** Model offload moves components across devices inside `forward()`, and
  torchao's quantized tensor subclasses trip torch 2.13's storage-aliasing guard
  when that happens under inference_mode. The failure surfaces deep in
  `torchao/utils.py -> return_and_correct_aliasing` as "Attempted to set the
  storage of a tensor on device cuda:0 to a storage on different device cpu".
  This is also what the `Unable to import torchao Tensor objects` startup warning
  is pointing at.
- Seeds are drawn from `2**53 - 1` (minus the largest sweep), not `2**63 - 1`:
  JSON numbers are doubles, so a 19-digit seed came back from `res.json()`
  rounded and Reuse then generated a *different* image while claiming to
  reproduce one. The wrap modulus stays at `2**63 - 1`, the seed column's
  bound, so a hand-entered `2**60` is still itself. Both live in
  `backend/seeds.py` because the draw, the per-image offset and the generator
  must not drift apart.
- A batch gets **one generator per image**, seeded `seed + i`. diffusers draws
  each sample from its own generator and collapses a one-element list back to
  the scalar path, so image *i* is identical whether it came out of a batch or
  a solo run of its recorded seed. One shared generator made the per-image
  seeds in the gallery unreproducible.
- `mu` is part of the reproduction set, so `ImageRecord` carries it and the UI's
  Reuse restores it. `/api/status` defaults are seeded once and never after the
  user has touched a control — `describe()` reads CUDA memory and can answer
  minutes late from behind a load.
- Generation is serialized through a one-thread pool — the GPU cannot usefully
  run two at once.
- A job outlives the page that started it. Each SSE stream gets **its own**
  queue (`Job.subscribers`): one shared queue is destructive, so a reconnecting
  client overlapping the stream it replaces would split the events and only one
  reader would ever see `done`. The browser's automatic reconnect is left alone
  for the first few failures, and `GET /api/jobs` lists what is still live, so a
  reload reattaches instead of orphaning a sweep with no reachable id.
- `save_image` writes `<name>.png.part` and renames only after the row commits.
  `_write_lock` is per-process and `clear.py` is a separate one: mid-encode, a
  plain `.png` is an untracked file for its orphan sweep to delete.
- Cancellation is **cooperative**. A diffusion step cannot be interrupted from
  outside, so `POST /api/jobs/{id}/cancel` only sets `Job.cancel_requested`; the
  backend polls it from `callback_on_step_end` and raises `Cancelled`, which
  unwinds out of the denoising loop. A cancel therefore lands within one step,
  and everything already written to disk is kept and reported. A job still in
  the queue is finished off by the endpoint itself rather than waiting for the
  job ahead of it, so `_finish()` guards the terminal SSE event against being
  emitted twice.
- `request.runs` loops `jobs._generate` over N sequential pipeline calls with
  base seed `seed + i * num_images`. That is the *only* way to get many 2K
  images out of 16 GB: `num_images` holds the whole batch in VRAM at once, runs
  hold one. Each image is emitted as its own `image` SSE event as it lands, so a
  64-run sweep fills the page as it goes.
- A job runs a **list of cells**, not a request and a run count. A plain request
  expands to `runs` cells of identical settings; a grid (`POST /api/grid`) to
  the cartesian product of its axes. `_generate` cannot tell them apart, so
  cancellation, the progress events and the per-image seed maths are written
  once. `Job.total_steps` is the *cell's*, since a grid varies it between them.
- In a grid, **seeds are the outermost loop**. A night that ends early - a
  cancel, a crash, a power cut - has then finished every parameter combination
  for the seeds it reached, which is the comparison that was wanted; the other
  order leaves every seed part-done and nothing comparable. Cells are built as
  real `GenerateRequest`s, so an axis value outside the schema fails at submit
  with the schema's own message rather than sixty images in.
- `POST /api/grid/preview` runs the same expansion without queueing it, so the
  panel's count, its resolved axes and its error text are the run's own. Its
  duration estimate is per *step* (`storage.step_cost_ms`), measured from recent
  gallery rows at the same size: a grid's cells disagree about step count, so an
  average image duration would misstate most of them. No rows, no estimate.
- **Resume keys on settings, not on the job.** `storage.completed_cells()` reads
  back (steps, CFG, mu, seed) for a prompt at a size on a checkpoint, and
  `grid.prune_done()` subtracts them, so a night that died is continued by a
  *new* job and a cell rendered by hand earlier counts too. Both sides round to
  the six decimals `parse_axis` produces, so a float that has been through
  SQLite's REAL still matches. Resume **requires an explicit seed** — a random
  base would match none of the run it means to continue and would silently
  render the whole grid again — and an entirely-finished grid answers 409
  rather than queueing a job that reports "done" having drawn nothing.
- Every image records the `job_id` that made it: a browser job's id, or one
  minted per CLI invocation. That is what groups a night's work, and the
  gallery's `job:` filter is how you get it back. Rows written before the column
  existed have NULL, so `_row_to_record` and the UI both tolerate its absence -
  and `_migrate()` adds the column to an older database, because
  `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it was.
- The gallery search box is `key:value` terms plus free text: the terms become
  SQL predicates (`seed:42`, `cfg:<=0.5`, `steps:8-12`, `mu:none`, `model:flux`,
  `job:ab12cd`) and only what is left goes to the FTS index, so a prompt that
  happens to contain a colon still searches for itself. `mu:none` is `IS NULL` -
  no shift at all is a different image from `mu:0`, and SQL matches NULL with
  neither.
- The contact sheet (`web/app.js`, `renderSheet`) lays one run out as the grid
  that made it: the two widest-varying axes become columns and rows, any third
  becomes a pinned selector rather than a third dimension nobody can read. It
  asks for the whole run in one page (`SHEET_PAGE`) because a matrix with a page
  boundary through it is a puzzle.
- `generate.py` duplicates the sweep loop from `jobs._generate` rather than
  importing it: that method is welded to a `Job`, an event loop and an SSE
  queue, none of which exist in a CLI. What it no longer duplicates is *which
  images to make*: both call `grid.expand()`, which with no axes set yields
  exactly the old runs loop (`seed + index * num_images`, same wrap helper), so
  a cell in the browser and a cell in the terminal cannot drift apart. It still
  owns the per-image `seed + offset` and the `save_image` call. Nor does it
  duplicate settings:
  `--model/--quant/--offload` are written into `os.environ` *before* the first
  import of `backend.config`, because `settings` is a frozen dataclass whose
  fields are evaluated at import time.
- `events()` opens with a `done` event, not `status`, for a job that has already
  finished — otherwise a stream attached after the fact (a reload, or a queued
  job the cancel endpoint ended) leaves the client waiting on an event that will
  never come.

## Adding a backend

Implement `GenerationBackend` (`load`/`unload`/`is_ready`/`generate`/`describe`)
and register it in `backend/backends/__init__.py:get_backend`. `generate()` is
handed a `should_cancel` predicate and must poll it at every step boundary,
raising `Cancelled` when it answers True. Nothing in the HTTP layer or the UI
needs to change.

## Tests

`uv run --group dev pytest`. They stub the pipeline (`tests/test_api.py`'s
`FakeBackend`) and the scheduler (`tests/test_schedule.py`'s `StubPipeline`), so
nothing downloads weights or touches the GPU, and they run against a throwaway
database and outputs directory (`tests/conftest.py`). They cover the invariants
that are expensive to get wrong: the schedule the scheduler ends up with, seeds
surviving a JSON round trip, the search fallback keeping its filters, and the
HTTP guards.

## Environment

Python 3.13 (not 3.14 — torchao/diffusers wheels lag there).
torch 2.13.0+cu130, diffusers 0.40.0, torchao 0.18.0.
