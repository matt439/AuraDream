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
- `backend/jobs.py` — single-worker queue, SSE progress
- `backend/storage.py` — SQLite index + PNGs with embedded metadata
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
- `generate.py` duplicates the sweep loop from `jobs._generate` rather than
  importing it: that method is welded to a `Job`, an event loop and an SSE
  queue, none of which exist in a CLI. The parts that *must* not drift — the
  `seed + index * num_images` step, the per-image `seed + offset`, the
  `save_image` call — are a dozen lines. What it does not duplicate is settings:
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
