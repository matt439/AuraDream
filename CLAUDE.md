# Krea Local

Local text-to-image generator on an RTX 5080. FastAPI backend + vanilla JS
frontend, no build step. The default checkpoint is **FLUX.1 Krea [dev]**
(`black-forest-labs/FLUX.1-Krea-dev`), which is click-through gated; **Krea 2
Turbo** (`krea/Krea-2-Turbo`) is what the sizing notes below are written around
and needs manual approval on HuggingFace. Either runs unchanged - set
`KREA_MODEL_ID`.

## Run

```powershell
uv run python -m backend.main      # http://127.0.0.1:7860; bare on a terminal, asks
uv run python -m backend.main --defaults   # KREA_* as set, no questions
uv run --group dev pytest          # no GPU, no weights, a few seconds
```

```bash
bash wsl/setup.sh                  # WSL 2, once: Linux venv + CUDA check
source wsl/env.sh && bash wsl/serve.sh   # 4-bit Nunchaku, same URL
```

The model is *not* loaded at startup. The first `POST /api/generate` (or
`POST /api/model/load`) triggers a load that takes a few minutes.

## Layout

- `backend/config.py` — settings, most overridable by `KREA_*` env vars
- `backend/launch.py` — the start-up menu and the `--model/--quant/--offload`
  flags, run before `config` is imported; the checkpoint catalogue lives here
- `backend/schemas.py` — pydantic request/response models
- `backend/model_defaults.py` — recommended steps/CFG/mu per checkpoint
- `backend/seeds.py` — the seed range and the wrap, shared by server and CLI
- `backend/sizes.py` — the preset table; the CLI's `--aspect/--mp`, config's
  `max_side`/`max_pixels`, `web/app.js`'s copy, `index.html`'s `max=` and the
  README table are all held to it by `tests/test_profiles.py`
- `backend/backends/base.py` — the `GenerationBackend` ABC
- `backend/backends/diffusers_backend.py` — in-process diffusers implementation
- `backend/grid.py` — axis specs (`6-12x2`) and the cell expansion, shared
- `backend/jobs.py` — single-worker queue, SSE progress
- `backend/storage.py` — SQLite index + PNGs with embedded metadata, search filters
- `backend/main.py` — routes
- `backend/generate.py` — headless CLI; drives the backend directly, no server
- `backend/clear.py` — deletes rows, their PNGs and any orphans
- `backend/stats.py` — the starred *rate* per setting (`storage.keep_rates`),
  scoped to a run by default
- `backend/reindex.py` — rows rebuilt from the PNGs' `krea` chunk, in place or
  copied in from another gallery; idempotent on the UNIQUE filename
- `web/` — served as-is from `/static`
- `wsl/` — `env.sh` (Linux-side paths and the Nunchaku defaults, to source),
  `setup.sh`, `serve.sh`: running the same checkout from WSL 2

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
  not shift *dynamically*. **A non-dynamic scheduler still shifts, statically**:
  `set_timesteps`'s else branch multiplies the odds by `self.shift` (1.0 on
  the shipped checkpoints, 3.0 on SD3/3.5, 6.0 on Lumina 2) on top of whatever
  sigmas it was handed, so the fallback pre-shifts by `mu - ln(shift)`
  (`_static_shift`, read off the instance) and the total is still `e^mu`. At
  `shift == 1.0` that is byte-identical, which is exactly why a test pinned to
  1.0 hid it. `shift_terminal` and the karras/exponential/beta options
  post-process in both branches and get a warning at load. Assert on
  `scheduler.sigmas`, never on the list the backend computed - see
  `tests/test_schedule.py`.
- Turbo is distilled: 8 steps, `guidance_scale=0.0`. **Where guidance switches
  on is a property of the pipeline, not the constant 1.0**: `FluxPipeline` runs
  an uncond pass only at `true_cfg_scale > 1`, while `Krea2Pipeline` guides from
  0 up and substitutes `""` for a missing negative prompt - so the old `> 1.0`
  gate silently guided Turbo against an empty prompt in the band `0 < g <= 1`.
  The backend keys the threshold off `"true_cfg_scale" in self._call_params`,
  and `/api/status` publishes it as `defaults.cfg_threshold` for the UI and the
  CLI. Dropping the negative prompt below the threshold saves nothing - neither
  pipeline encodes it outside its guidance branch - it exists to keep the
  recorded settings honest about what ran, **and that only holds if the drop
  reaches the row**: `GenerateRequest.applied()` is the one normalisation
  (negative prompt gone where unsupported or at/below the threshold, mu gone
  where the pipeline has no way to take one), `backend.applied()` answers it
  from the loaded parameter list or, unloaded, from the profile, and both save
  paths write `applied()` rather than the request. `main._expand` and the CLI
  run the same normalisation over the cells before `prune_done`, so a resume
  whose negative prompt the pipeline never saw still matches the rows it
  wrote. The UI's `collect()` sends null for a greyed-out field, and the field
  is gated on the highest CFG the open grid's axis reaches, not the panel's
  single value - a grid decides guidance per cell.
- `KREA_QUANT=nunchaku` loads a checkpoint that is *already* quantized (Nunchaku
  Lite, SVDQuant W4A4; `quant_method: nunchaku_lite` in the transformer's
  `config.json`, bitsandbytes NF4 in the text encoder's) and passes diffusers no
  quantization config at all - the repo's own configs name the quantizer, and
  torchao on top would be wrong. The CUDA ops arrive through the `kernels`
  package from `rootonchair/nunchaku-lite-kernels` (branch `v2`, pinned by
  diffusers), which publishes x86_64 **Linux** builds only - torch 2.11-2.13 for
  cu126/cu130/cu132 - so on this machine the mode means WSL 2. NVFP4 needs
  Blackwell (sm_120 qualifies); the `int4` repos are for older cards.
  `_nunchaku_preflight` refuses off Linux, without `kernels`/`bitsandbytes`, or
  on a repo whose transformer is not pre-quantized, *before* any download: the
  bf16 case otherwise loads fine and OOMs at the first image. diffusers reads
  `DIFFUSERS_TRUST_REMOTE_KERNELS` once, at import, so `config.py` sets it
  (`setdefault`) - it is imported before diffusers everywhere, the CLI included.
  Under WSL, `wsl/env.sh` moves the venv (`UV_PROJECT_ENVIRONMENT`), the gallery
  (`KREA_OUTPUTS_DIR`, `KREA_DB_PATH`) and the HF cache off `/mnt/c`: 9P is slow
  and SQLite cannot lock over it. The original `nunchaku` engine is a different
  package with its own transformer classes and no torch 2.13 wheel; not used.
- **torch 2.13 runs some CUDA ops as Triton kernels in eager mode** (`torch._native`
  overrides `aten::bmm` and others), and Triton compiles a C helper on first use.
  A fresh WSL Ubuntu has no C compiler, so the Qwen3-VL text encoder's rotary
  embedding died on the first step with "Failed to find C compiler". `wsl/env.sh`
  sets `TORCH_DISABLE_NATIVE_JIT=1`, which torch honours by declaring those
  overrides unavailable and dispatching to ATen. Verified 2026-09-05: Krea 2
  Turbo NVFP4 via Nunchaku on the 5080, 8 steps, model offload: 1 MP in 15 s at
  8.1 GiB peak, 2 MP in 24 s at 8.6 GiB. `KREA_OFFLOAD=none` does **not** fit
  this checkpoint: 14.7 GiB at 1 MP, 18.7 at 2 MP, and the WDDM driver spills
  the excess into system RAM through WSL just as it does natively - the images
  then took 26 s and 72 s. Offload's per-image weight transfer costs ~5 s and is
  the cheaper of the two. The same file pins the xet downloader's
  concurrency (`HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false`), because its
  adaptive controller collapsed to one connection and stalled a 12 GB checkpoint
  twice on this connection.
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
- **FLUX.2 [klein]** (`Flux2KleinPipeline`, in diffusers 0.40) loads through the
  same loader, but its ids contain `flux`, so `model_defaults.py` has `klein-base`
  and `klein` entries *ahead of* `flux`: distilled klein is 4 steps and ignores
  guidance (the pipeline warns above 1), base is 50 steps at real CFG 4.0. Its
  `__call__` has **no `negative_prompt`** - only embeds, and base's uncond pass
  runs against `""` - so the profile carries `negative_prompt_supported: False`,
  `describe()` reports `"negative_prompt" in _call_params` once loaded,
  `/api/status` merges it the way it merges `cfg_threshold`, and the UI greys the
  field for the checkpoint rather than for the value. `cfg_threshold()`:
  `true_cfg_scale` present -> 1.0 (FLUX.1); else `negative_prompt` present
  **and the pipeline class is in `_GUIDES_FROM_ZERO`** (`Krea2Pipeline`, the
  `ZImage*` family - the only nine of 131 diffusers pipelines whose
  `do_classifier_free_guidance` is `> 0`) -> 0.0; else 1.0 (klein's classic
  `guidance_scale > 1` gate, and SD3/SDXL/Chroma/Lumina 2's). "Has
  `negative_prompt`" alone used to read 0.0, which made loading SD3 make
  `/api/status` *less* accurate than the profile's guess. `hints_for` takes
  the live `negative_prompt_supported` as well as the threshold, so the CFG
  tooltip cannot contradict the greyed-out field. `load()` refuses a pipeline
  whose `__call__` lacks `generator` or `callback_on_step_end`
  (`_REQUIRED_PARAMS`): dropped silently, the first meant unseeded noise under
  a recorded seed and the second a Cancel that never lands. Verified 2026-09-05 via Nunchaku NVFP4 on the
  5080: 1 MP in 4.0 s at 2.9 GiB peak, 2 MP in 4.6 s at 3.4 GiB. Its pipeline also lacks
  `enable_vae_tiling`/`enable_vae_slicing`; the loader falls through to the same
  switches on `pipe.vae`. Its mu is empirical (sequence length and step count),
  so the profile leaves mu null and `_forced_mu` stays out of the way. Text
  encoder is a plain `Qwen3ForCausalLM` (transformers), so the fp8 path
  quantizes it like any other; the 4B's is 8 GB in bf16, bigger than its
  7.75 GB transformer.
- **A bare `python -m backend.main` on a terminal asks** which checkpoint,
  quantization and offload to use; *any* argument (`--model`, `--quant`,
  `--offload`, `--defaults`) or a non-tty stdin skips the menu, and what the
  flags do not name comes from `KREA_*` as before. `settings` is frozen at
  import, so `launch.configure()` runs from a `__name__ == "__main__"` guard
  placed *before* `main.py`'s own package imports - the same reason
  `generate.py` applies its flags before importing `backend.config`. Both entry
  points take the flags from `launch.add_model_arguments`, so they cannot
  drift. The menu asks quantization first because it decides the repo family:
  `nunchaku` (offered on Linux only) lists the lite-infer re-uploads for the
  families that have one and asks nvfp4 or int4; the environment's values are
  the marked defaults, so `wsl/env.sh` still gets what it set on a bare Enter.
  `wsl/serve.sh` forwards its arguments.
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
  Reuse restores it. `/api/status` defaults are applied **per control**: each
  of steps/CFG/mu follows the profile until the user touches *that* box
  (`owned` in `web/app.js`) — `describe()` reads CUDA memory and can answer
  minutes late from behind a load, and one flag for all three, wired to Seed
  as well, used to freeze the panel at the fallback's 28 / 3.5 on a Turbo
  server because the user pasted a seed in the first two seconds. Reuse owns
  all three, and warns (as a note) when the record's checkpoint is not the one
  the server is running: steps, CFG and mu are checkpoint-specific. An empty
  Steps or CFG box is sent as *absent*, which the schema fills from the
  profile; `+''` is 0, and 0 is a valid CFG that generates unguided.
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
  plain `.png` is an untracked file for its orphan sweep to delete. The encode
  sits *inside* the cleanup guard (a KeyboardInterrupt mid-encode is a
  BaseException Pillow does not tidy), the cleanup only runs while the row has
  **not** committed - past that the partial is the only copy of an image the
  database already names - and the rename is retried over a few hundred
  milliseconds, because on Windows a just-closed PNG is exactly what Defender
  or an indexer holds open. The orphan sweep and `summarise()` glob the
  directory *before* reading the rows (a file the scan saw has a row by then),
  and sweep `.part` files only past `PARTIAL_SETTLE_SECONDS` (an hour): a
  fresh one may be another process mid-encode. `_migrate`'s once-per-process
  flag is keyed on the database path, so a file restored under a running
  server gets its columns again.
- **Shutdown has to reach the sweep before uvicorn drains connections.**
  uvicorn closes listeners, waits for open connections, and only *then* runs
  the lifespan shutdown - and the browser's EventSource stays open until the
  job emits `done`. So `jobs.shutdown()` (the cancel) sat behind the stream
  that was waiting for it, and Ctrl-C printed "Waiting for connections to
  close" for the rest of a grid. `main.run()` subclasses `uvicorn.Server` and
  sets `cancel_requested` on every active job in `handle_exit`, i.e. on the
  signal itself, with `timeout_graceful_shutdown=30` for a client that stops
  reading. The load runs on a **daemon thread** (`main._detached`), not
  `asyncio.to_thread`: that pool is joined at interpreter exit, so a Ctrl-C
  four minutes into a 32 GB download waited the download out. Lifespan calls
  `backend.try_unload()` (non-blocking lock acquire) off the loop rather than
  `unload()`, which blocks behind a load in progress and then discards what it
  built; `POST /api/model/unload` answers 409 while a load is in flight for the
  same reason.
- **A cell's failure is the cell's, not the night's.** `_generate` catches
  `Exception` after `Cancelled`, records it on the job (`failures`, `error`),
  skips the cell and continues; three failures in a row, or more than a third
  of the grid, re-raise and end the job. A finished job with failures reports
  `done` with an `error` line the UI shows. Every job logs a line when queued,
  one per cell and one at the end, and carries `created_at`/`started_at`/
  `finished_at`/`last_progress_at` as float seconds in its snapshot
  (json-native; a datetime would raise mid-job on the SSE path). `_generate`
  returns whether it left the loop early, and the terminal status comes from
  that rather than from the
  cancel flag - a cancel that lands after the last step used to relabel a
  complete run "cancelled". Every terminal transition goes through `_finish`,
  which owns the `done_sent` guard for state *and* event: a queued job the
  endpoint had cancelled used to be rewritten to `error` when its load then
  failed. `generate.py` mirrors the per-cell rule and exits 1 when any cell
  was skipped. `Job.snapshot(images=False)` is what the per-cell status event
  and `/api/jobs` send: the per-cell event used to carry every record so far,
  258 MiB over a 1000-cell grid.
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
- **The prompt is an axis too**, through `{name}` placeholders and a block of
  `size = big | small` lines (`grid.parse_variables`). Values split on `|`, not
  on the comma the numeric axes use: `a big, fluffy dog` is one prompt fragment,
  and splitting it the other way turns a value into two cells of nonsense. The
  names cross with each other and with steps/CFG/mu, so two variables of two
  values is four prompts. Both halves of a mismatch are refused at submit rather
  than found in the morning: a variable no placeholder mentions multiplies the
  night by a value that changes nothing, and a placeholder with no variable
  renders the literal `{ground}` into every image of the run - **including
  when the variables box is empty**, which used to short-circuit past both
  checks (a plain prompt with no placeholders still expands to itself). The
  product of the variables is **counted before it is built**
  (`math.prod` against `max_grid_images`): `MAX_AXIS_VALUES` caps values per
  variable, nothing capped the variables, and eighteen binary lines built
  262,144 dicts on the event loop per preview keystroke. `none` is a value
  - the word is simply not said - and `substitute()` closes the gap it leaves,
  so `{size} dog` is `dog` rather than ` dog`. The prompt loop sits just inside
  the seed loop, for the reason seeds are outermost: a night that ends early has
  swept every setting for the prompts it reached, which is a comparison.
- The contact sheet's axes gained **Prompt**, whose labels are only the words
  that vary across the run (`promptLabels` in `web/app.js`): a column header
  repeating the forty words every cell shares is unreadable. Equal word counts
  let the differing positions be picked out exactly; a `none` makes one prompt
  shorter than its neighbours, and then the shared head and tail come off
  instead. Pins address their values by position, because a shortened label is
  not something that round-trips back to the prompt it came from.
- `POST /api/grid/preview` runs the same expansion without queueing it, so the
  panel's count, its resolved axes and its error text are the run's own. Its
  duration estimate is per *step* (`storage.step_cost_ms`), measured from recent
  gallery rows at the same size **on the same checkpoint** (klein's step is half
  of Turbo's at 1 MP): a grid's cells disagree about step count, so an average
  image duration would misstate most of them. No rows, no estimate.
- **Resume keys on settings, not on the job.** `storage.completed_cells()` reads
  back (prompt, negative prompt, steps, CFG, mu, seed) for the *distinct
  prompts* a grid resolved to, at a size on a checkpoint **under a quantizer**
  - a list, because a prompt grid renders several - and `grid.prune_done()`
  subtracts them, so a night that died is continued by a *new* job and a cell
  rendered by hand earlier counts too. A cell with `num_images > 1` is done
  only when **every** per-image seed is present (`grid.cell_keys`): the rows
  are per image, and one committed image of four used to mark the cell done.
  Quantization is part of the reproduction set (same seed, different quantizer,
  different pixels), so it is recorded on the row and in the PNG and scopes the
  lookup; rows from before the column match any quantizer. Both sides round to
  the six decimals `parse_axis` produces, so a float that has been through
  SQLite's REAL still matches. Resume **requires an explicit seed** — a random
  base would match none of the run it means to continue and would silently
  render the whole grid again — and an entirely-finished grid answers 409
  rather than queueing a job that reports "done" having drawn nothing. The seed
  is the one setting the client is never given back by the images: every row
  carries a *derived* seed (`base + seed_index * num_images`, plus the batch
  offset), so `JobAccepted` and `Job.snapshot()` publish the base, the UI
  writes it into a blank Seed box after `Run grid`, and the recovery
  instruction everywhere says the *oldest* image of the run, never "any".
- Every image records the `job_id` that made it - a browser job's id, or one
  minted per CLI invocation - **and the `run_id` of the grid it belongs to**
  (`grid.run_id`: a hash of the checkpoint, quantization, size, batch and every
  cell key of the *full* grid, computed in `main._expand` and `generate.py`
  before resume prunes anything). A resumed night therefore lands under a second
  job id but the same run id, and the gallery's `run:` filter - which is what
  the card's ⊞ button sets - shows the whole matrix where `job:` shows one
  submission. Rows written before either column existed have NULL, which the
  UI tolerates - and `_migrate()` adds the columns to an older database,
  because `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it
  was. `_row_to_record` reads every column **unguarded**: rows only ever come
  out of a connection `_connect` opened, and that ran `_migrate` first. A new
  column needs a `_migrate` entry, not a `"x" in row.keys()` read guard - the
  guard would let the SELECT succeed and every INSERT after it fail.
- **The lightbox zooms, and the view carries across cells.** `lbView` in
  `web/app.js` is a scale relative to fit (never below 1) plus a pan, keyed on
  the record's `width x height` and kept across records that share it, so the
  arrows compare the identical crop at the identical magnification down a
  steps or mu axis. The image is laid out with explicit width/height/left/top
  inside a `flex: 1 1 auto; min-height: 0` stage - not a CSS transform, which
  can composite a fit-size texture and scale it - and 1:1 divides by
  `devicePixelRatio`, or Windows scaling makes it a 1.25x resample, exactly
  the smoothing a sharpening sweep is judging. The wheel listener is
  non-passive; the zoom keys sit below the `input, textarea` guard. A click on
  the stage beside the image closes the lightbox only at fit: zoomed, the
  stage is a pan surface and only Escape or the cross close it.
- **A grid can have a time budget.** `GridRequest.budget_minutes` reaches the
  job through `submit_cells(budget_minutes=)` - GridRequest-only fields are
  gone by the time cells are built - as an absolute `Job.deadline` from
  submission (the load counts). `_generate` checks it at the top of every
  cell ("start no cell after"), sets `stopped_by = "budget"` and returns
  early through the *cancel* path: a fourth terminal status would have left
  `active()`/`busy()` holding on and unload answering 409 for the session.
  The UI reads `stopped_by` to say the plan worked and a Resume finishes the
  rest; the CLI's `--budget` mirrors it and exits 130 like any other stop
  short of the full count.
- **The variable assignment is recorded.** `Cell.variables` reaches
  `save_image(variables=)` from both writers, lands in a `variables TEXT`
  column (JSON; `_migrate`), the `krea` chunk and `ImageRecord.variables`.
  The sheet builds one axis per variable name seen (`axesFor` in `app.js`)
  and **suppresses** the flat prompt and negative axes when any record
  carries variables - kept as fallbacks, a 4-value prompt axis would beat a
  2-value variable axis in `chooseAxes` and turn a single-variable run into a
  diagonal. `""` (the `none` value) reads as `none`; a record without
  variables in a mixed search reads as null. The axis list is rebuilt per
  render and the names are part of the layout signature. `completed_cells`
  and `CellKey` are untouched: the resolved prompt is what ran.
- **Two writers, one INSERT.** `save_image` and `reindex.index_file` both go
  through `storage._insert_row` over `_COLUMNS`, so the column list exists
  once and the FTS trigger fires for both. The reindex reads
  `Image.open(p).info["krea"]`, parses `created_at` from the filename prefix
  as UTC (mtime failing that), copies a foreign file in `.part`-then-rename
  and never over a different file of the same name, records `duration_ms=0`
  (so `step_cost_ms`'s `> 0` ignores it) and loses `favorite`, which lives in
  SQLite alone. `_strict_scope` is the `key:value`-only reading of the search
  grammar for the paths where the LIKE fallback is a hazard: `clear --where`
  (`delete_matching`, `DELETE ... RETURNING filename` in one statement, no
  orphan sweep on that path) and `keep_rates`. `/api/status` carries
  `disk_free_mb` (one `shutil.disk_usage` syscall; the directory walk in
  `summarise()` deliberately does not ride the 3 s poll) and the preview
  carries `bytes` (`image_bytes`: median PNG size at this size on this
  checkpoint, times the images) under the same no-rows-no-estimate rule.
- **The CLI reads PNGs and probes the server.** `--from-image` reads
  `Image.open(p).info["krea"]` - not `.text`, which decodes every pixel - and
  fills what the flags left unset *before* `apply_env`, so the record's
  checkpoint reaches `KREA_MODEL_ID`; quantization is deliberately not
  carried (a Nunchaku PNG on Windows would ask for a mode the machine cannot
  run) and job/run ids are displayed, never reused. `_server_holds_gpu`
  probes `/api/status` after the `--dry-run` guard: `ready`/`loading` or a
  timeout refuses with exit 4, connection refused is the ordinary no-server
  case, `--ignore-server` overrides; the test fixture stubs it, or the suite
  would refuse to run beside the developer's own server. `keep_rates` groups
  on the same six-decimal rounding `completed_cells` uses and always includes
  size; free text in its scope is refused rather than sent through the
  fuzzy LIKE fallback.
- **The panel persists to `localStorage`** (`krea.panel.v1`) in two phases:
  the model-independent fields synchronously at start-up, *before* the sync
  functions that light the presets and hints up from them, and steps/CFG/mu
  from `refreshStatus` once the checkpoint is known - only the ones the user
  owned, and only when the stored `model` is the loaded one. Writes come from
  a delegated `input`/`change` listener on the panel plus explicit calls
  after the programmatic assignments that fire no event (`choose`, `reuse`,
  the dice, the seed written back after Run grid). Resume is never stored.
  Every access is in try/catch; a blocked store is a page that works without
  memory. The contact sheet's cells take the commonest aspect ratio on the
  sheet (`--cell-aspect`, `object-fit: contain`) and carry `.fav`, toggled by
  class from `setFavorite` - never by re-rendering, which rebuilds up to a
  thousand `<img>` nodes per keypress.
- **The seed axis is a count or a list.** `GridRequest.seed_axis` (the UI's
  Seed list box, the CLI's `--seed-list`, exclusive with `--grid-seeds`) goes
  through `grid.parse_seeds` - `parse_axis` with an int cast - and `expand()`
  takes `seed_list`, which wins over `seeds` and the base seed. The ceiling
  counts whichever applies. Listed values are **not** wrapped: the schema's
  `ge=0, le=MAX_SEED` on the cell is what turns a typo'd `-5` into the
  schema's own error rather than a valid 9.2-quintillion seed. A listed grid
  is determinate on its own, so `_resume_needs_a_fixed_seed` waives the base
  seed for it, and the UI does not write the first listed seed back into the
  Seed box. The preview's `values.seed` is the list, or `[]` for a count.
- **A queued job says what it waits on.** `Job.waiting_on` is `"load"` while
  the weights are on their way and `"queue"` behind another job - a field
  beside `status`, not a fourth state, because `cancel()` keys its instant
  path on `queued` and a new state would have turned Cancel-during-load into
  a multi-minute no-op. The label is set only when a load is actually
  needed. Progress events carry `elapsed` (server clock, since the job
  started running); the UI's ETA is `elapsed * (1 - f) / f` over the fraction
  of cells done, the job's own pace rather than the gallery's per-step
  median, so a run that has slowed shows it. The `done` summary goes into
  `#cell-note`, which sits outside the element `setBusy` hides.
- `date:` in the search box is a **local** day (`today`, `yesterday`,
  `YYYY-MM-DD`) as a half-open range on `created_at`, converted to UTC and
  formatted the way the rows are so the string comparison is the time
  comparison. Not a LIKE prefix: that cannot use the index and would need the
  UTC date, which is the wrong one for anything made in the evening.
- **`POST /api/rerender` is the third producer of cells**, beside
  `/api/generate` and `/api/grid`: the gallery's own filter (search plus the
  Favorites flag, taken server-side because the browser holds one page) at
  another size, one cell per record with the record's prompt, negative
  prompt, steps, CFG, mu and seed, oldest first. It calls `check_budget()`,
  refuses more than `max_grid_images` matches by count rather than truncating,
  refuses records from another checkpoint (steps/CFG/mu are the checkpoint's),
  prunes with `completed_cells` like a resume and 409s when nothing is left.
  Same seed at another shape is a *new sample* - the latents are drawn at the
  requested size - which the README says out loud.
- **The morning after has two sources.** `GET /api/runs` groups the gallery
  by `run_id` (`storage.runs`: images, favourites, distinct prompts, jobs,
  first/last timestamp, and the *oldest* image's prompt and seed - the base
  seed a resume needs); it outlives the process and answers "how many and
  when". `GET /api/jobs?include_finished=true` adds the terminal jobs from
  memory (`jobs.finished()`, newest-finished first, `image_count` in every
  snapshot) and is the only thing that answers "why" - a cancel, an OOM, the
  cell it stopped at - for as long as the server is up. The UI reads the
  first into the Runs picker over the gallery (pick one: `run:<id>` search
  plus the sheet) and the second into a banner under the panel when nothing
  is live, gated to a job finished within 24 h and announced once per job
  per browser (`localStorage`, every access in try/catch).
- The gallery pages **by id, not by offset**: `GET /api/gallery?before=<id>`
  returns rows older than that id and `has_more`, which the server knows
  exactly by fetching one row past the page. OFFSET over a table a sweep is
  inserting into repeated the tail of one page at the head of the next and
  skipped as many older rows for good; `offset` stays for scripts. The
  browser's anchor is the last record it holds, so a delete moves nothing.
- The offload hooks are handed `KREA_DEVICE` explicitly (when it is a CUDA
  device) and `describe()` reads that card's name and memory: given no device,
  diffusers resolves its own accelerator, which is always GPU 0, so `cuda:1`
  used to load onto cuda:0 while the status line described cuda:0 as if that
  were the setting.
- The gallery search box is `key:value` terms plus free text: the terms become
  SQL predicates (`seed:42`, `cfg:<=0.5`, `steps:8-12`, `mu:none`, `model:flux`,
  `run:ab12cd`, `job:ab12cd`) and only what is left goes to the FTS index, so a prompt that
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
  images to make*: the CLI and `/api/grid` both call `grid.expand()`, which
  with no axes set yields exactly the old runs loop (`seed + index *
  num_images`, same wrap helper), so a grid cell in the browser and one in the
  terminal cannot drift apart. **`jobs.submit()` - the plain `/api/generate`
  path - is the one remaining copy of that seed loop**, kept inline
  deliberately: routing it through `expand()` would apply the grid's
  placeholder check to an ordinary prompt with braces in it. Its arithmetic is
  asserted equal to the CLI's in `tests/test_cli.py`. The CLI still owns the
  per-image `seed + offset` and the `save_image` call, and shares
  `grid.estimate_seconds` with the preview. Nor does it duplicate settings:
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
