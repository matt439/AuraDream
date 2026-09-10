# Design notes

Why the code is the way it is, one section per area. `CLAUDE.md` states the
rules; this file keeps the reasoning behind each, the history of what went
wrong before the rule existed, and the evidence, in the notes kept while the
code was written. Read the section for an area before changing it. The
measurements quoted here are indexed by date and checkpoint in `verified.md`.

## Why float8 and model offload

Krea 2 is ~13B and needs **~25 GB in bf16**, which does not fit 16 GB. The
default path quantizes transformer + text encoder to float8 via torchao and
uses `enable_model_cpu_offload()`. Peak VRAM lands near 13 GB.

## Quantization goes through quant_mapping

Quantization must go through `PipelineQuantizationConfig(quant_mapping=...)`.
The shorthand `quant_backend="torchao"` form raises `ValueError` on this
pipeline: the text encoder is a *transformers* model (Qwen3-VL) and its
`TorchAoConfig.__init__` signature differs from the diffusers one. Both take a
torchao `AOBaseConfig` **instance** (`Float8WeightOnlyConfig()`), not a string.

## mu, sigmas and the shift that lands

**`sigmas` is an input to the shift, not the shift's result.** `__call__`
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

## Where guidance switches on, and applied()

Turbo is distilled: 8 steps, `guidance_scale=0.0`. **Where guidance switches
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

## Nunchaku Lite: pre-quantized repos, Linux only

`KREA_QUANT=nunchaku` loads a checkpoint that is *already* quantized (Nunchaku
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

## torch 2.13's Triton kernels under WSL, and the xet downloader

**torch 2.13 runs some CUDA ops as Triton kernels in eager mode** (`torch._native`
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

## Krea 2's components and offload order

Components are `text_encoder` (Qwen3-VL), `transformer` (Krea2Transformer2DModel),
`vae` (AutoencoderKLQwenImage). Offload order is `text_encoder->transformer->vae`.

## Loading happens in one place

Loading happens in exactly one place: `main.ensure_loaded()`, which holds an
asyncio lock across the whole load. Jobs `await` it rather than calling
`backend.load()` from the worker thread — doing both raced two concurrent
loads per request and produced duplicate tracebacks. `generate()` raises
rather than loading a missing pipeline, so that rule is enforced rather than
merely documented, and `POST /api/model/unload` answers 409 while any job is
live: unloading under a running job freed nothing (the worker holds its own
reference) and had the next run of a sweep reload from the gen thread.

## The loader is model-agnostic

The loader is model-agnostic: `DiffusionPipeline.from_pretrained` resolves the
class from `model_index.json`, and `_component_libraries()` reads the same file
to decide which components to quantize and with whose `TorchAoConfig`. FLUX
layouts therefore pick up `text_encoder_2` (T5) automatically.

## Profiles key off the checkpoint id

`model_defaults.py` keys recommended steps/CFG/mu off the checkpoint id, not the
pipeline signature: `Krea2Pipeline` reports 28 steps at CFG 4.5, which is the
family default and wrong for Turbo. `/api/status` serves the profile and the UI
seeds its controls from it once, on load.

## mu is Turbo-only

`mu` is Turbo-only. FLUX derives its own resolution-dependent shift, so forcing
Turbo's constant on it degrades output; the profile leaves mu null there.

## True CFG is its own field

On FLUX, `guidance_scale` is a distilled conditioning value, **not** CFG - a
negative prompt is silently ignored unless `true_cfg_scale > 1.0`. **Real CFG
is its own request field there**, `GenerateRequest.true_cfg` (None = off):
the two used to be set from one number, so the only reachable states were
(3.5, off) and (g, g), and a negative prompt typed at 3.5 stacked guidance
twice. `describe()` publishes `true_cfg_supported` (`"true_cfg_scale" in
_call_params`; the profile guesses it from the id until then), `/api/status`
merges it like the threshold, and where it is True `applied()` compares the
threshold against `true_cfg`, not `guidance_scale`. **The recorded `true_cfg`
means one thing on every checkpoint: the scale the negative prompt was
applied at, or None** - on Krea 2/SD3/klein that is `guidance_scale` when
the negative prompt ran, so a column, a filter (`tcfg:`), a `_GROUPS` entry
and the resume key all read the same way, and a row from before the column
(NULL) is read through `schemas.effective_true_cfg` / `storage._TRUE_CFG_SQL`
as "guidance_scale if there was a negative prompt", which is what ran then.
`CellKey` gained the element before the seed; `grid.run_id` names a cell
by the old six-element key whenever `true_cfg` is None, so every existing
run id survives (pinned in `tests/test_grid.py`). The UI's True CFG box is
greyed - not hidden - where the knob does not exist, the negative prompt
follows whichever box decides, and Reuse fills it only on a checkpoint that
has it. Verified on the card 2026-09-08 (FLUX.1 Krea dev, fp8): a negative
prompt with no true CFG produces the byte-identical image to no negative
prompt; with true CFG 1.5 a different, reproducible one at ~1.65x the time.

## FLUX.2 klein: no negative prompt, and the threshold rule

**FLUX.2 [klein]** (`Flux2KleinPipeline`, in diffusers 0.40) loads through the
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

## Reference images, by gallery id

**Reference images are klein's and are addressed by id.** `Flux2KleinPipeline.__call__`
takes `image` first, resizes each reference to <=1 MP, VAE-encodes it and
concatenates the tokens onto the latent sequence every step; nothing else in
the catalogue has the parameter, so `describe()` reports
`reference_supported` (`"image" in _call_params`), the profile guesses it
from the id until then, `/api/status` merges it like the threshold and the
UI greys the field for the checkpoint. `GenerateRequest.reference` is a list
of **gallery ids**, never paths - a path joined onto `outputs_dir` is an
arbitrary-file read reachable from `/api/generate` - and
`storage.reference_paths` resolves them, naming an id that is missing and a
file that is gone. `applied()` drops them where the pipeline has none, and
`main._refuse_unusable_inputs` refuses at *submit* rather than at cell 1 of
64: unlike a negative prompt, a dropped reference renders a plain
text-to-image that looks like a success. The module imports `Image` as the
class for typing, so `_reference_images` opens through a local
`from PIL import Image as PILImage`.
**The 2-3x cost the review predicted did not appear.** Measured on the card
2026-09-08 (klein 4B, fp8, model offload, 1024x1024, 4 steps): 10.2 s with
no reference, 9.2 s and 8.2 s with one, 11.3 s with two. Each reference
roughly doubles the sequence and attention is quadratic in it, but under
model offload the per-image weight transfer dominates both, so the estimate
in the panel needs no reference-aware term. The conditioning itself is
unmistakable and reproducible: the same fox, cabin and treeline as the
referenced image, in the pose the prompt asked for, and the same hash twice.

## LoRA adapters are the environment's and are hot-swapped

**A LoRA is the environment's capability, not the checkpoint's.** All three
families mix a loader in, so what decides it is peft and the quantizer:
`lora_unavailable_reason()` refuses `KREA_QUANT=nunchaku` (already 4-bit
SVDQuant; the modules PEFT would wrap are not `nn.Linear`) and a missing
peft up front, the way `_nunchaku_preflight` refuses, and `describe()`
answers `lora_supported` with no pipeline loaded at all. Adapters are
**hot-swapped on the live pipeline** in `generate()`, never baked into
`load()`: diffusers strips and re-attaches the offload hooks around the
injection, so the "adapters must precede `enable_model_cpu_offload`" rule
does not hold, and a second load path driven from the job worker is the one
thing `main.ensure_loaded` exists to prevent. `_apply_lora` tracks
`(name, weight)` itself and a weight change is a `set_adapters`, which is
what makes `0-1x0.25` one load and four re-weightings; a failed injection
unloads and clears that state, because the alternative is every later cell
running under an adapter nothing recorded. The request carries a *name*
(`_bare_name` refuses separators) and `backend/loras.py` is the only place
one becomes a path, resolved under `loras_dir` and checked to still be
inside it. `.safetensors` only: a `.bin` LoRA is a pickle.
**torchao fp8 + PEFT was the unverified part, and it works**: measured on the
card 2026-09-08 (FLUX.1 Krea dev, fp8, model offload, a 172 MB FLUX.1-dev
LoRA) - the adapter changes the image and reproduces, `unload_lora_weights`
restores the base hash exactly, and **weight 0 is byte-identical to no
adapter**, which is what proves the weight reaches the maths rather than
merely being recorded. A three-weight sweep was 17.5/16.3/18.6 s against
15.4 s unadapted: the swap between cells costs nothing measurable. The row
keeps `lora` at weight 0 rather than nulling it - the adapter *was* attached,
the way CFG 0 is recorded as 0 - and `distinct_count` is what says the 0 cell
matched the baseline.

## CellKey and run_id

**Both widen `CellKey`, and `run_id` survives it.** The key is
(prompt, negative, steps, CFG, mu, true CFG, references, LoRA, seed), with
`grid.reference_key` (ids in order) and `grid.lora_key` (`name@weight`)
spelled once and built from a request in `grid` and from columns in
`storage.completed_cells`. `run_id` names a cell by the **shortest** key
carrying all its non-None optional elements - it drops true CFG, references
and LoRA from the right while they are None - so a grid using none of them
is still named by the six-element key its images have carried since before
any of them existed (pinned in `tests/test_conditioning.py`). A weight axis
with no adapter selected is not an axis: `applied()` puts `lora_scale` back
to 1.0 where `lora` is None, or two cells that differ on a number no pixel
depends on would split the resume key, and `_sampling_values` returns the
single value rather than the parsed list.

## The start-up menu

**A bare `python -m backend.main` on a terminal asks** which checkpoint,
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

## no_grad, never inference_mode

**Use `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline
call.** Model offload moves components across devices inside `forward()`, and
torchao's quantized tensor subclasses trip torch 2.13's storage-aliasing guard
when that happens under inference_mode. The failure surfaces deep in
`torchao/utils.py -> return_and_correct_aliasing` as "Attempted to set the
storage of a tensor on device cuda:0 to a storage on different device cpu".
This is also what the `Unable to import torchao Tensor objects` startup warning
is pointing at.

## Seeds fit in a double

Seeds are drawn from `2**53 - 1` (minus the largest sweep), not `2**63 - 1`:
JSON numbers are doubles, so a 19-digit seed came back from `res.json()`
rounded and Reuse then generated a *different* image while claiming to
reproduce one. The wrap modulus stays at `2**63 - 1`, the seed column's
bound, so a hand-entered `2**60` is still itself. Both live in
`backend/seeds.py` because the draw, the per-image offset and the generator
must not drift apart.

## One generator per image

A batch gets **one generator per image**, seeded `seed + i`. diffusers draws
each sample from its own generator and collapses a one-element list back to
the scalar path, so image *i* is identical whether it came out of a batch or
a solo run of its recorded seed. One shared generator made the per-image
seeds in the gallery unreproducible.

## Per-control defaults and Reuse

`mu` is part of the reproduction set, so `ImageRecord` carries it and the UI's
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

## One worker thread

Generation is serialized through a one-thread pool — the GPU cannot usefully
run two at once.

## A job outlives the page; per-stream queues

A job outlives the page that started it. Each SSE stream gets **its own**
queue (`Job.subscribers`): one shared queue is destructive, so a reconnecting
client overlapping the stream it replaces would split the events and only one
reader would ever see `done`. The browser's automatic reconnect is left alone
for the first few failures, and `GET /api/jobs` lists what is still live, so a
reload reattaches instead of orphaning a sweep with no reachable id.

## Atomic PNG writes

`save_image` writes `<name>.png.part` and renames only after the row commits.
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

## Shutdown has to reach the sweep first

**Shutdown has to reach the sweep before uvicorn drains connections.**
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

## A cell's failure is the cell's

**A cell's failure is the cell's, not the night's.** `_generate` catches
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

## Cooperative cancellation

Cancellation is **cooperative**. A diffusion step cannot be interrupted from
outside, so `POST /api/jobs/{id}/cancel` only sets `Job.cancel_requested`; the
backend polls it from `callback_on_step_end` and raises `Cancelled`, which
unwinds out of the denoising loop. A cancel therefore lands within one step,
and everything already written to disk is kept and reported. A job still in
the queue is finished off by the endpoint itself rather than waiting for the
job ahead of it, so `_finish()` guards the terminal SSE event against being
emitted twice.

## Runs, not batches, for many 2K images

`request.runs` loops `jobs._generate` over N sequential pipeline calls with
base seed `seed + i * num_images`. That is the *only* way to get many 2K
images out of 16 GB: `num_images` holds the whole batch in VRAM at once, runs
hold one. Each image is emitted as its own `image` SSE event as it lands, so a
64-run sweep fills the page as it goes.

The batch-wide pixel ceiling is a typo guard, not a VRAM formula: peak memory
varies substantially by checkpoint, quantizer and conditioning. Its default is
the larger of two maximum-size 4 MP presets and Batch 8 at the largest 1 MP
preset aspect. The original two-at-4-MP ceiling was 8,411,136 pixels, which
accidentally rejected eight 3:4 or 4:3 images from the nominal 1 MP row; a
measured batch of seven used only 10.7 GB of the 16 GB card. Large batches are
still refused, and a machine with different headroom can set
`KREA_MAX_BATCH_PIXELS` explicitly.

## A job is a list of cells

A job runs a **list of cells**, not a request and a run count. A plain request
expands to `runs` cells of identical settings; a grid (`POST /api/grid`) to
the cartesian product of its axes. `_generate` cannot tell them apart, so
cancellation, the progress events and the per-image seed maths are written
once. `Job.total_steps` is the *cell's*, since a grid varies it between them.

## Seeds are the outermost loop

In a grid, **seeds are the outermost loop**. A night that ends early - a
cancel, a crash, a power cut - has then finished every parameter combination
for the seeds it reached, which is the comparison that was wanted; the other
order leaves every seed part-done and nothing comparable. Cells are built as
real `GenerateRequest`s, so an axis value outside the schema fails at submit
with the schema's own message rather than sixty images in.

## The prompt is an axis: variables

**The prompt is an axis too**, through `{name}` placeholders and a block of
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

## Prompt labels on the sheet

The contact sheet's axes gained **Prompt**, whose labels are only the words
that vary across the run (`promptLabels` in `web/app.js`): a column header
repeating the forty words every cell shares is unreadable. Equal word counts
let the differing positions be picked out exactly; a `none` makes one prompt
shorter than its neighbours, and then the shared head and tail come off
instead. Pins address their values by position, because a shortened label is
not something that round-trips back to the prompt it came from.

## Grid preview and the per-step estimate

`POST /api/grid/preview` runs the same expansion without queueing it, so the
panel's count, its resolved axes and its error text are the run's own. Its
duration estimate is per *step* (`storage.step_cost_ms`), measured from recent
gallery rows at the same size **on the same checkpoint** (klein's step is half
of Turbo's at 1 MP): a grid's cells disagree about step count, so an average
image duration would misstate most of them. No rows, no estimate.

## Resume keys on settings

**Resume keys on settings, not on the job.** `storage.completed_cells()` reads
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

## job_id, run_id and _migrate

Every image records the `job_id` that made it - a browser job's id, or one
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

## The lightbox zooms

**The lightbox zooms, and the view carries across cells.** `lbView` in
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

## The sheet exports client-side

**The sheet exports client-side.** `renderSheet` leaves its matrix in
`sheetLayout`; `buildSheetCanvas` draws the same axes, pins and labels onto
a canvas from the *thumbnails* (cells shrink so the longer edge stays under
8000 px), with a caption naming the search, the pins, the checkpoint, the
run and the time, and `exportSheet` downloads it. Not a server route: that
would duplicate a layout contract the client already holds, the
non-round-trippable prompt labels especially.

## Pause

**Pause is cooperative, like cancel, and lands between cells.**
`POST /api/jobs/{id}/pause?value=` sets `pause_requested`; `_generate`
answers at the top of the next cell by setting `paused` (a worker-written
acknowledgement - the UI flips its label on *that*, never on the click,
because a denoise may still be running), calls `backend.free_cache()`
(torch's `empty_cache`: under model offload nearly all the card holds
between cells is allocator reserve) and polls `pause_requested` every
quarter second. Polled, never `Event.wait()`: a bare wait deadlocks
`shutdown()`, which cancels and then joins the thread, and makes `cancel()`
a no-op on a paused job. `busy()` is unchanged, so unload still 409s - the
weights are needed for the resume - which is the cheap half of the design;
a full unload-while-paused only matters for `KREA_OFFLOAD=none`.

## The browser's queue

**The browser can hold a queue.** `startJob` no longer refuses while busy:
a submission during a job is POSTed and its id kept in `queuedIds`, the
buttons are held (`aria-disabled`) only for the POST itself as the
double-submit guard, and `watchNext()` attaches to the next job when the
watched stream ends. `resumeActiveJob` attaches to the running job, else
the *oldest* queued (`active[0]` - `active()` is newest last), never the
newest. Cancel acts on the watched job; "Cancel queued" on the rest.

## Time budget

**A grid can have a time budget.** `GridRequest.budget_minutes` reaches the
job through `submit_cells(budget_minutes=)` - GridRequest-only fields are
gone by the time cells are built - as an absolute `Job.deadline` from
submission (the load counts). `_generate` checks it at the top of every
cell ("start no cell after"), sets `stopped_by = "budget"` and returns
early through the *cancel* path: a fourth terminal status would have left
`active()`/`busy()` holding on and unload answering 409 for the session.
The UI reads `stopped_by` to say the plan worked and a Resume finishes the
rest; the CLI's `--budget` mirrors it and exits 130 like any other stop
short of the full count.

## The variable assignment is recorded

**The variable assignment is recorded.** `Cell.variables` reaches
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

## Two writers, one INSERT; strict scope; disk estimates

**Two writers, one INSERT.** `save_image` and `reindex.index_file` both go
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

## The CLI reads PNGs and probes the server

**The CLI reads PNGs and probes the server.** `--from-image` reads
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

## Panel persistence

**The panel persists to `localStorage`** (`krea.panel.v1`) in two phases:
the model-independent fields synchronously at start-up, *before* the sync
functions that light the presets and hints up from them, and steps/CFG/mu
from `refreshStatus` once the checkpoint is known - only the ones the user
owned, and only when the stored `model` is the loaded one. Writes come from
a delegated `input`/`change` listener on the panel plus explicit calls
after the programmatic assignments that fire no event (`choose`, `reuse`,
the dice, the seed written back after Run grid). Resume is never stored.
Every access is in try/catch; a blocked store is a page that works without
memory. **Reset** (`resetPanel`, the button at the end of the Prompt caption)
is the one thing that empties it: the store is removed, every `owned` flag
dropped, and the boxes put back to what a first visit shows - the last
status's profile for steps/CFG/mu, the 1 MP square, a random seed, no
conditioning, an empty folded grid. It exists because `backend.clear` is a
server-side deletion that never reaches `localStorage`, so after clearing a
gallery the old prompt came back on every open, and the only other way out
was wiping the site's data. It sits away from Generate on purpose: it costs
the prompt, and a slip next to the button pressed all day would cost it
unasked. The contact sheet's cells take the commonest aspect ratio on the
sheet (`--cell-aspect`, `object-fit: contain`) and carry `.fav`, toggled by
class from `setFavorite` - never by re-rendering, which rebuilds up to a
thousand `<img>` nodes per keypress.

## The seed axis is a count or a list

**The seed axis is a count or a list.** `GridRequest.seed_axis` (the UI's
Seed list box, the CLI's `--seed-list`, exclusive with `--grid-seeds`) goes
through `grid.parse_seeds` - `parse_axis` with an int cast - and `expand()`
takes `seed_list`, which wins over `seeds` and the base seed. The ceiling
counts whichever applies. Listed values are **not** wrapped: the schema's
`ge=0, le=MAX_SEED` on the cell is what turns a typo'd `-5` into the
schema's own error rather than a valid 9.2-quintillion seed. A listed grid
is determinate on its own, so `_resume_needs_a_fixed_seed` waives the base
seed for it, and the UI does not write the first listed seed back into the
Seed box. The preview's `values.seed` is the list, or `[]` for a count.

## waiting_on and the ETA

**A queued job says what it waits on.** `Job.waiting_on` is `"load"` while
the weights are on their way and `"queue"` behind another job - a field
beside `status`, not a fourth state, because `cancel()` keys its instant
path on `queued` and a new state would have turned Cancel-during-load into
a multi-minute no-op. The label is set only when a load is actually
needed. Progress events carry `elapsed` (server clock, since the job
started running); the UI's ETA is `elapsed * (1 - f) / f` over the fraction
of cells done, the job's own pace rather than the gallery's per-step
median, so a run that has slowed shows it. The `done` summary goes into
`#cell-note`, which sits outside the element `setBusy` hides.

## date: is a local day

`date:` in the search box is a **local** day (`today`, `yesterday`,
`YYYY-MM-DD`) as a half-open range on `created_at`, converted to UTC and
formatted the way the rows are so the string comparison is the time
comparison. Not a LIKE prefix: that cannot use the index and would need the
UTC date, which is the wrong one for anything made in the evening.

## Re-render

**`POST /api/rerender` is the third producer of cells**, beside
`/api/generate` and `/api/grid`: the gallery's own filter (search plus the
Favorites flag, taken server-side because the browser holds one page) at
another size, one cell per record with the record's prompt, negative
prompt, steps, CFG, mu and seed, oldest first. It calls `check_budget()`,
refuses more than `max_grid_images` matches by count rather than truncating,
refuses records from another checkpoint (steps/CFG/mu are the checkpoint's),
prunes with `completed_cells` like a resume and 409s when nothing is left.
Same seed at another shape is a *new sample* - the latents are drawn at the
requested size - which the README says out loud.

## The morning after: runs and finished jobs

**The morning after has two sources.** `GET /api/runs` groups the gallery
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

## Paging by id

The gallery pages **by id, not by offset**: `GET /api/gallery?before=<id>`
returns rows older than that id and `has_more`, which the server knows
exactly by fetching one row past the page. OFFSET over a table a sweep is
inserting into repeated the tail of one page at the head of the next and
skipped as many older rows for good; `offset` stays for scripts. The
browser's anchor is the last record it holds, so a delete moves nothing.

## KREA_DEVICE reaches the offload hooks

The offload hooks are handed `KREA_DEVICE` explicitly (when it is a CUDA
device) and `describe()` reads that card's name and memory: given no device,
diffusers resolves its own accelerator, which is always GPU 0, so `cuda:1`
used to load onto cuda:0 while the status line described cuda:0 as if that
were the setting.

## The search grammar

The gallery search box is `key:value` terms plus free text: the terms become
SQL predicates (`seed:42`, `cfg:<=0.5`, `steps:8-12`, `mu:none`, `model:flux`,
`run:ab12cd`, `job:ab12cd`) and only what is left goes to the FTS index, so a prompt that
happens to contain a colon still searches for itself. `mu:none` is `IS NULL` -
no shift at all is a different image from `mu:0`, and SQL matches NULL with
neither.

## The contact sheet

The contact sheet (`web/app.js`, `renderSheet`) lays one run out as the grid
that made it: the two widest-varying axes become columns and rows, any third
becomes a pinned selector rather than a third dimension nobody can read. It
asks for the whole run in one page (`SHEET_PAGE`) because a matrix with a page
boundary through it is a puzzle.

## The CLI's loop, and jobs.submit's

`generate.py` duplicates the sweep loop from `jobs._generate` rather than
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

## events() opens with done

`events()` opens with a `done` event, not `status`, for a job that has already
finished — otherwise a stream attached after the fact (a reload, or a queued
job the cancel endpoint ended) leaves the client waiting on an event that will
never come.

## Content hashes and duplicate counts

**Every image records a hash of its pixels** (`storage.content_hash`: 16
hex of the sha256 of `image.tobytes()`; column, `krea` chunk key `hash`,
`ImageRecord.content_hash`, `hash:` filter, and the reindex recomputes it
from the decoded file). It is what lets "did this axis do anything" be a
count: `Job.snapshot()` carries `distinct_count`, `RunSummary.distinct` is
`COUNT(DISTINCT content_hash)` only when every image of the run is hashed,
and the UI and the CLI say "only N distinct" when it is fewer than made.
The premise - bit-determinism through torchao's fp8 kernels with model
offload moving components mid-forward - was measured on 2026-09-08 rather
than assumed: the same request twice on FLUX.1 Krea dev is the same hash,
as is a negative prompt the pipeline dropped against none. The companion is
`grid.duplicate_count`: `GridPreview.duplicates` and the CLI's dry run
count the cells whose *applied* key repeats an earlier one (a
negative-prompt-only variable on klein, a CFG axis under the threshold),
compared through `backend.applied()` so the capability flags decide; the
cells are still rendered as spelled - dropping them would leave a `{avoid}`
axis with one cell of three and a sheet unable to say why.

## The live preview

**The live preview is a linear map from the latents, keyed on the VAE.**
`_step_end` hands `cb_kwargs["latents"]` to `latent_preview`: unpacked
through the pipeline's own `_unpack_latents` (FLUX's 4-arg staticmethod,
Krea 2's 3-arg method returning 5-D; klein has none and gets no preview),
multiplied by a 16x3 matrix fitted by least squares on the card from
pipeline-normalised latents (`(z - shift) * scale` for FLUX's
`AutoencoderKL`, `(z - mean) / std` for Krea 2's `AutoencoderKLQwenImage`;
R² 0.96 / 0.95 on real images), with a gain of `1/(1 - sigma)` because the
callback's latents are x_t not x_0, a 3-tap blur against the leftover
noise, nothing at all above sigma 0.9, and never on the last step. At most
one frame per 0.4 s, a ~5 KB JPEG at twice latent resolution as a data URL
in a `preview` SSE event. The whole thing sits in a try/except that logs
once and switches itself off: an exception out of that callback is how
`Cancelled` ends a job. `jobs.on_preview` checks `job.subscribers` **per
frame**, not once per cell - the browser opens its stream after the POST
returns, by which time an idle server is inside the first cell, and the
once-per-cell check skipped every first cell (found on the card, fixed).
The matrices cannot be validated in CI; the fit script and its date are
in the module docstring, and the frames were looked at.

## FLUX.1 dev and schnell, Z-Image Turbo, Qwen-Image

**FLUX.1 [dev], FLUX.1 [schnell] and Z-Image Turbo are in the catalogue**
(`launch.py`), each with a lite-infer re-upload so the nunchaku menu offers
them. dev is the `flux` profile as it stands. schnell has its own entry
*ahead of* `flux`: 4 steps, guidance 0 - its transformer has
`guidance_embeds: false`, so the CFG box never reaches the model and only
True CFG does - and a scheduler shipped `use_dynamic_shifting: false,
shift: 1.0`, i.e. unshifted. Z-Image Turbo is `ZImagePipeline` (6B, Qwen3
text encoder): 8 steps, CFG 0, and it guides from zero like Krea 2, which
`_GUIDES_FROM_ZERO` already covered; `z-image-turbo` sits ahead of a
`z-image` entry carrying the pipeline's own 50 / 5.0. It ships the FLUX
`AutoencoderKL` and normalises latents the FLUX way, so the fitted preview
matrix applies - but its latents are never packed and it has no
`_unpack_latents`, so `latent_preview.unpack` passes a 4-D latent through
and still answers None for a 3-D packed sequence with no hook (klein). Its
scheduler is `use_dynamic_shifting: false, shift: 3.0`: the pipeline
computes a resolution-dependent mu and the scheduler ignores it, so a blank
mu is a fixed shift of 3 (mu = ln 3) and a typed one goes through the
static-shift fallback and lands exactly. `test_launch.py` derives the
"another repo id" option number from the catalogue's length. Measured
2026-09-08 under Nunchaku NVFP4 on the 5080, model offload, the CLI
in-process with torch's peak allocated (`~/krea-local/measure.py` in WSL):
Z-Image Turbo 1 MP in 8 s at 4.1 GiB, 2 MP in 14.5 s at 4.6 GiB, an 11 s
load, and preview frames at steps 2/4/6 of 8 in the right colours; FLUX.1
dev 1 MP in 30 s and 2 MP in 48 s at 7.4 / 7.8 GiB; schnell 1 MP in
16-23 s and 2 MP in 25 s at the same peaks - most of a schnell image is the
per-image cost that does not scale with steps (T5 under bitsandbytes NF4,
the weight transfer), so it is not the speed win Z-Image is. The desktop
holds 5-6 GB of the card while Unreal Editor and the browsers are open, and
a run that lands within a gigabyte of the top spills into system RAM
(schnell's 2 MP read 35-45 s that way and 25 s after the baseline fell to
1.3 GB). **Qwen-Image is in the catalogue too** (`QwenImagePipeline`, Krea
2's parent: same VAE, same unpack hook, FLUX-style `true_cfg_scale`), with
a `qwen-image` profile - 50 steps, guidance 1.0 which the transformer
ignores, true CFG separate at threshold 1.0 - and **the pipeline runs its
uncond pass only when a negative prompt is present**, the same gate FLUX.1
has; the True CFG hint says to type one. Measured the same way with true
CFG 4 and a negative prompt: 11.75 GiB peak at 1 MP and 12.1 GiB at 2 MP,
78 s per 1 MP image and 152 s at 2 MP once warm (210 s for the first of a
session), a 22 s load, and the rows record true CFG 4 and the negative
prompt. Its lite-infer re-upload is 12 GB of transformer at NVFP4.

## ERNIE-Image Turbo: the prompt rewriter, and a mu with no way in

**ERNIE-Image Turbo is in the catalogue** (`baidu/ERNIE-Image-Turbo`,
`ErnieImagePipeline` in diffusers 0.40; Baidu's 8B single-stream DiT, Apache
2.0), with lite-infer's re-upload - named without the `_r32` the other lite
ids carry, so the entry spells its id out instead of using `_LITE`. The
profile is `ernie-image-turbo` ahead of an `ernie-image` family entry at the
pipeline's own 50 / 4.0: 8 steps at guidance 1.0. Its
`do_classifier_free_guidance` is the classic `> 1.0`, so `cfg_threshold()`'s
else-branch answers 1.0 with no change to `_GUIDES_FROM_ZERO`; it takes a
`negative_prompt` and no `true_cfg_scale`, so to the controls it is klein
with the negative prompt back. **Its repo carries a prompt rewriter**: `pe`,
a 7.7 GB bf16 Ministral 3B with its own tokenizer, in the class's
`_optional_components`, and `__call__` defaults `use_pe=True`, under which
every prompt is rewritten by `pe.generate(do_sample=True, temperature=0.6)`
before it is encoded. A hidden rewrite would make the recorded prompt differ
from what the model saw and the image seed would not reproduce the whole
operation. `_UNUSED_COMPONENTS` therefore still maps the class name to the
two components and ordinary `load()` reads `model_index.json` ahead of the
weights (`_unused_components`) to pass `pe=None, pe_tokenizer=None` - **to the
concrete class named by the index**, not to `DiffusionPipeline`. The base
class resolves the same class from the same file, but `_get_signature_keys`
is a classmethod that reads `cls._optional_components`, which on the base is
empty: through it `pe` is not an expected module, `pe=None` becomes a pipeline
kwarg, `download()` never filters `pe/` out and the loop loads the model - seen
on the card (7.7 GB pulled at load, 11 GB of RAM taken, never called) before
the switch. Through `ErnieImagePipeline.from_pretrained` the two are passed
components: `download()` drops their folders from `allow_patterns` and
`load_module` skips them.

The explicit **Expand** editor action is the exception. `POST
/api/prompt/expand` waits for the normal model load, refuses to cut ahead of
an image job, then loads `pe` and `pe_tokenizer` from the component classes in
the same manifest on first use. They are registered on the pipeline and its
configured offload hooks are rebuilt so the large rewriter moves to the chosen
card only for its call. The selected width and height reach ERNIE's JSON user
message; the sampled result replaces the textarea only if the user has not
edited it while the load was in flight. Later expansions reuse the component
until Unload. `generate()` still sets `use_pe=False` wherever the switch
exists: the expanded text is visible, editable, and is exactly what the row and
PNG record. The rewriter's bundled chat template contains its Chinese system
instruction and does not render an added system-role message, so asking the
same model for English through `_enhance_prompt_with_pe(system_prompt=...)`
does nothing. After expansion, a separate
[`Helsinki-NLP/opus-mt-zh-en`](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
Marian model therefore translates the result. It is a 312 MB, CC-BY-4.0,
Chinese-to-English checkpoint loaded lazily through the Transformers and
SentencePiece packages already in the app, kept on CPU so it never disturbs
the image pipeline's offload chain, and dropped by Unload. The Chinese text
remains in the editor and is the only prompt sent and recorded; the English is
a read-only aid underneath, hidden on the first edit. Translation failure is
returned separately so a useful multi-GB expansion is not thrown away. The
index read is a try/except that yields nothing on failure - an optimisation,
not a gate - and the tests' conftest now refuses `_hub_file` for every test,
since a stubbed `from_pretrained` had been the only thing between a loader
test and the Hub. **mu has no way in**: the pipeline builds `linspace(1, 0)`
itself and hands it to a scheduler shipped `use_dynamic_shifting: false,
shift: 4.0`, and `__call__` takes no `sigmas` and no `mu`, so `_mu_supported`
answers False, `applied()` drops the value and the row records none; the
shift that lands is the constant 4 (mu = ln 4), which the hint says. Sides
are multiples of 16 (`AutoencoderKLFlux2`, 32 latent channels, 2x2-patchified
into the 128 channels the callback carries as a 4-D tensor), which every
preset already is; `factors_for` has no map for that VAE, so the preview
switches itself off once at DEBUG and the cell pays nothing for it - a fit
would need the card and the pipeline's `bn_mean`/`bn_std` normalisation.
Baidu lists 1 MP shapes only; 2 MP and 4 MP render. **Measured 2026-09-08
in the evening**, Nunchaku NVFP4 on the 5080 with model offload, the CLI
in-process (`~/krea-local/measure.py`), Z-Image Turbo re-run the same
evening for a like-for-like: ERNIE 1 MP in 9.6-10.2 s once warm (17-18 s
for the first image of a process, 22 s load) at 5.2 GiB, 2 MP in 24.3 s at
5.9 GiB, a 4 MP pair in one call in 48.6 s at 10.3 GiB allocated / 12.0 GiB
reserved; Z-Image 10.6 s, 12.8 s and 23.5 s at 4.1 / 4.6 / 7.4 GiB (11.8
reserved) - level at 1 MP, twice the time above it. **Under Nunchaku a seed
does not reproduce bit-for-bit**: the same request in two processes gave
two hashes on both checkpoints, Z-Image's pair 4.5/255 apart on average
with 17% of pixels off by more than 8 (a different seed: 36/255, 85%),
ERNIE's 14-24/255 and 58-75% (a different seed: 41/255, 94%) - visibly
different details on the 8-step model, not rounding. The fp8 path on
Windows reproduces (see `verified.md`), so a duplicate count or a re-render
hash is that path's guarantee only. One measurement trap: right after WSL
reclaimed memory (`Cached: 0`) Z-Image read 17-19 s per image, 10.6 s on the
next run; the card was idle and boosting fine. And the 7.7 GB the base-class
load had pulled was deleted from the cache before the concrete-class load
was verified, so that fetching nothing was observable.

## Animagine XL 4.0: an SDXL schedule is not a Mu

**Animagine XL 4.0 Opt is in the catalogue**
(`cagliostrolab/animagine-xl-4.0`, CreativeML Open RAIL++-M and open). The
repository publishes a normal `StableDiffusionXLPipeline`: two CLIP encoders,
an SDXL UNet, `AutoencoderKL` and an Euler scheduler. Its Diffusers component
weights total about 6.5 GiB. The 2025-02-13 Opt commit changed the component
tree's UNet as well as adding `animagine-xl-4.0-opt.safetensors` at the root,
so `DiffusionPipeline.from_pretrained` gets Opt without a single-file loader.
That commit also changed the manifest from `EulerAncestralDiscreteScheduler`
to `EulerDiscreteScheduler`. The model card still calls Euler Ancestral the
preferred sampler, but the detailed example does not replace the manifest's
scheduler. Follow the current checkpoint rather than overriding it from stale
prose; a sampler control would need to become part of `CellKey`, the PNG and
the database before it could be changed reproducibly.

**A `sigmas` parameter is not proof of flow matching.** Current
`StableDiffusionXLPipeline.__call__` takes custom sigmas, so the old generic
`_mu_supported` test would have treated Animagine's epsilon-prediction Euler
scheduler as flow matching and replaced its schedule with `_shifted_sigmas`.
Mu now requires a scheduler class from the `FlowMatch` family first. The
capability is part of the checkpoint profile before load and `describe()`
after load, so the main Mu box and its grid axis are disabled as well as a
stray request being dropped from the record. This also gives ERNIE's known
no-Mu pipeline the disabled control its tooltip had already promised.

**Use the checkpoint's tag vocabulary.** The model card says natural-language
input may not be effective. Its order is person count, character, series,
rating, the remaining subject/pose/scene tags, and quality tags last; both CLIP
tokenizers stop at 77 tokens. It recommends 25-28 steps (28), CFG 4-7 (5), a
specific negative prompt, and nine roughly-1-MP shapes. The six previously
missing aspects - 9:7, 7:4 and 12:5 plus their inverses - now use the card's
exact 1 MP dimensions. The existing 3:2 pair keeps ERNIE-Image's nearby sizes,
and every Animagine aspect is available across all four tiers. Its guide also
names those nine ratios in `aspects`: recommendation highlighting requires both
the named framing and the pixel range, so unrelated 1 MP presets such as 16:9
do not turn green merely because their area is close. `resolutions` carries the
card's exact nine dimension pairs for free-form W/H input; exact matching admits
the rounded 1216x832 "3:2" without loosely aliasing 16:9 to nearby 7:4.

**The Tags picker is guide data, not a hard-coded Animagine branch.**
`PromptGuide.tag_helper` carries the card's person-count, rating, quality and
score vocabularies, its `year {n}` form, recommended quality suffix and negative
prompt. The button appears only when that object is present. Each tag is an
explicit editor action at the current caret; active tags toggle off, exclusive
person/rating choices replace one another, and the suffix action moves that
known set to the end. Character, series, subject, pose and scene stay free text
because the card describes those slots but publishes no finite vocabulary for
them. The prompt remains visible, editable, persisted and recorded exactly as
before; the helper never rewrites on generation. Its documented limitations
are text, difficult anatomy, multiple characters, post-cutoff
characters and degradation at high resolutions such as 1536x1536. Sources,
checked 2026-09-10: the model card and repository
(<https://huggingface.co/cagliostrolab/animagine-xl-4.0>) and the Opt commit
(<https://huggingface.co/cagliostrolab/animagine-xl-4.0/commit/4688bebb86806957f6f83c39f2573161630e22db>).
The class signature, profile, guide and no-Mu application have automated
coverage. The user then confirmed on 2026-09-10 that the weights loaded and
the output looked good on this card; the exact quantizer and offload mode were
not recorded. Speed, VRAM, LoRA behavior, live-preview fit and torchao
quantization therefore remain unmeasured.

**The complete vocabulary is Danbooru's, not a list in the checkpoint.** The
maintainer explicitly says XL 4.0 ships no wildcard or character list because
it trained from nearly all Danbooru image data, and directs users to Danbooru's
counts. Danbooru's official `/tags.json` API supplies canonical names,
categories, post counts, normalized multi-name lookup and alias-aware prefix
search. `backend.tags` proxies that fixed HTTPS origin off the event loop and
caches prefixes, exact hits and misses; the browser sends only the active tag
prefix or the comma-separated tag names, never an image or generation setting.
Suggestions exclude empty/deprecated tags and anything created after the
checkpoint's 2025-01-07 knowledge cutoff. Model-card-only controls such as
`great score` and `year 2023` validate locally; finite special tags are also
merged into autocomplete ahead of remote results and remain available offline.
A green chip means a model tag
or a nonempty Danbooru tag at the cutoff; red means unknown, deprecated,
empty or too new. An unavailable API makes ordinary tags grey and unchecked,
not invalid, and never blocks generation: presence in the index is useful
evidence, not proof that a particular caption reached training. Sources,
checked 2026-09-10: the maintainer's answer
(<https://huggingface.co/spaces/Asahina2K/animagine-xl-4.0/discussions/4>) and
Danbooru's tag API documentation
(<https://safebooru.donmai.us/wiki_pages/api:tags>).
On a two-column page the single validation node is moved into a panel above the
wide results area, where every chip wraps without a nested scrollbar and its
height cannot move Generate in the settings column. At 900 px and below it
returns under the prompt in normal document flow, where the whole page is
already a single scrolling column.

**Those chips are a round-trip editor, not a second prompt.** HTML drag/drop
moves a comma item and immediately serialises the whole order back through
`writePromptTags`; Alt+Left/Right is the keyboard equivalent. Every chip has an
× button and accepts Delete/Backspace while focused. A context click on its
left or right half opens an inline insertion field before or after it. Its
visible listbox reuses the same cached tag-suggestion endpoint; a native
`datalist` did not reliably reopen when asynchronous results arrived. Click,
Enter or Tab accepts a suggestion, while + commits literal text and Escape or ×
cancels. Every edit goes through the same preview, validation and
panel-persistence path, so the visible textarea, generated prompt and chips
cannot become separate sources of truth.
Chip and suggestion text is 12 px. A valid chip uses high-contrast light text
on neutral charcoal, with green confined to its checkmark and outline; unknown
chips keep the corresponding red mark/outline instead of colour-on-colour text.

## The prompting guide

**Every profile carries a `guide`**: `prompt` and `size` prose for the ?
badges on the Prompt box and the Size presets, a `pixels` range in
megapixels with a `pixels_word`, and the authors' `examples`. The prose
rides in `/api/status`'s `hints` under `prompt` and `size`, beside the
control tooltips it is shaped like; the data is `ModelStatus.guide`
(`schemas.PromptGuide`), because the size line, Examples menu and optional tag
helper need numbers and lists, not a sentence. `hints_for` passes the guide's
prose through the same `.format` as the control hints, so the paragraphs
hold no braces. The checkpoints here disagree about nearly everything a
prompt can be - FLUX.1's CLIP reads 77 tokens and its T5 512 (256 on
schnell), Qwen-Image's encoder is prompted to *describe the image by
detailing the color, shape, size, texture, quantity, text, spatial
relationships*, ERNIE ships a rewriter this backend leaves out and so
expects the expanded form typed in - and about sizes: FLUX.1 is documented
for 0.1-2 MP, Krea 2 Turbo for 1k-2k, Krea 2 Raw *trained to 1k*, Baidu
lists 1 MP shapes only. A user who has read that "some are only validated
at 1 MP" was right to ask which.

**The word matters as much as the range.** A model card's "supports 0.1 to
2 MP", a README's "recommended 1024-1280" and a report's "trained to
1328x1328" are three different promises, so the size line quotes the
profile's word - `above the 1.6 MP this checkpoint is recommended for`,
`below the 1 MP this checkpoint is documented for` - rather than one
verb for all. The line warns a tenth outside the range: the 1 MP presets
land at 1.04-1.08 MP and every profile's range has to admit them
(`test_every_guide_admits_the_1_mp_presets`), or a fresh page would open
on a warning. The preset buttons use that same tolerance and turn green when
the full size they would choose is inside the range and, when a guide publishes
a finite `aspects` list, uses one of those framings: aspect buttons are judged
at the current megapixel tier, and tier buttons at the current aspect. The VRAM
warning stays and stacks after it.

**Orientation is the inverse preset, not a width/height swap.** The button
derives `b:a` from the selected `a:b`, looks it up in the same `SIZES` table and
routes the result through `choose()`, so dimensions, active/recommended states,
grid preview and panel persistence all move together at the current tier. Every
non-square entry is asserted to have an inverse with transposed dimensions.
For 1:1 there is no distinct inverse, so the button is disabled and greyed.

**FLUX.1 Krea [dev] has its own profile entry for the guide alone.**
`krea-dev` sits ahead of `flux` (its id carries that token) as a copy of
the flux profile with Krea's guide: Krea's README recommends 1024-1280 px
sides and 28-32 steps at CFG 3.5-5, which FLUX.1 [dev]'s card does not,
and the checkpoint's whole point is photographs without the oversaturated
look. `test_krea_dev_is_flux_to_the_controls_and_its_own_to_the_guide`
holds the numbers and the control hints identical so nothing about a
request changed. `test_every_catalogue_entry_has_a_prompting_guide`
refuses the fallback's "Unrecognised checkpoint" text under a checkpoint
the menu offers by name, on the official id and the lite-infer one.

**The Examples menu builds on open** from the list the last status
carried, so a status that lands minutes late behind a load is not stale by
the first click; a pick assigns the textarea's value, which fires no
`input`, so it calls `schedulePreview` and `persistPanel` the way `choose`
does. The examples are verbatim from the authors - a model card's demo
prompt, a prompting guide's, a blog caption - because a checkpoint's own
examples are the shortest way to learn its voice, and because an invented
one would be a claim about the model with nothing behind it.

Sources, checked 2026-09-10. FLUX.1: BFL's prompting basics
(<https://docs.bfl.ai/guides/prompting_unified_basics>), the launch post's
"0.1 and 2.0 megapixels" (<https://bfl.ai/blog/24-08-01-bfl>), the model
cards, the reference CLI's 1360x768 and its multiple-of-16 rounding
(<https://github.com/black-forest-labs/flux>), diffusers' 77/512/256
limits. Above 2 MP degrading is community evidence
(<https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions/32>), and
the guide says "users report". Krea [dev]: <https://github.com/krea-ai/flux-krea>
(1024-1280, 28-32 steps, CFG 3.5-5), <https://bfl.ai/blog/flux-1-krea-dev>
and <https://www.krea.ai/blog/flux-krea-open-source-release> ("opinionated",
the "AI look"). FLUX.2 [klein]: <https://docs.bfl.ai/guides/prompting_guide_flux2>
(subject-action-style-context, 30-80 words, quotes, no negative prompt,
JSON prompts, 64x64 to 4 MP with 2 MP recommended) - written for [pro]
and [max], applied to klein by family; the klein cards say only "prompt
following is heavily influenced by the prompting style", and every example
is 1024x1024. Krea 2: <https://github.com/krea-ai/krea-2> `docs/prompting.md`
("natural language", "long detailed prompts yield best results", quotes
for text, "up to 2k"), its README ("1k ~ 2k", Raw "trained to generate
upto 1k", "TRAIN on Raw and RUN on Turbo"), the technical report
(<https://www.krea.ai/blog/krea-2-technical-report>: 256/512/1024 px
stages, a text-rendering reward, an encoder chosen for multilingual
generalisation). The "validated at 1 MP only" claim was **not** found for
Turbo in any official source; the guide says 1k-2k documented and 1024 px
the top pretraining stage. Qwen-Image: the README's positive magic and
resolution table (<https://github.com/QwenLM/Qwen-Image>; the HF copy's
1472x1140 is not a multiple of 16 and GitHub's 1104 is used), the report's
1328x1328 final stage (<https://arxiv.org/abs/2508.02324>). Z-Image:
<https://huggingface.co/Tongyi-MAI/Z-Image-Turbo> ("512x512 to 2048x2048
total pixel area", bilingual text), the maintainers' "works best with long
and detailed prompts" and "does not use negative prompts at all"
(<https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8>), the
report's "up to the 1k-1.5k range" (<https://arxiv.org/abs/2511.22699>).
ERNIE-Image: the card's seven 1 MP shapes and 8 steps at guidance 1
(<https://huggingface.co/baidu/ERNIE-Image-Turbo>), the rewriter's system
prompt in `pe/chat_template.jinja` (expand a brief description into a
"content-rich, detail-sufficient visual description"), the report's
Ministral-3 encoder and 256/512/1024 stages (<https://arxiv.org/abs/2605.25347>);
its tokenizer's 2048 is the config's, not a card statement.

## uv.lock's hand-pinned torch hashes

**`uv.lock` carries hashes for the PyTorch wheels by hand.** The
`download.pytorch.org` index publishes none for torch 2.13's three wheels or
the Windows torchvision wheel, so `uv lock` cannot add them; the four were
downloaded, hashed and checked file-for-file (`RECORD`) against the Windows
and WSL venvs on 2026-09-08. `uv lock --check` and `--dry-run` accept the
edited entries; a `uv lock --upgrade` that moves torch will drop them.

## Verified on the card, 2026-09-08

Verified on the card 2026-09-08 beyond the above: a real Ctrl-C (SIGINT to
uvicorn's handler) 12 s into a load exits in 3 s with the lifespan run;
mid-sweep with a stream attached, `done` (cancelled) reaches the stream
0.2 s after the signal and the process is gone 3 s later; Pause is
acknowledged at the cell boundary and the card drops from 14.95 GB during
a 1 MP fp8 denoise to 2.1 GB held; `reindex --from` over the real gallery
reproduces every hash and true CFG. Not verified: `KREA_DEVICE=cuda:1` (one
card here).
