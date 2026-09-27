# Design notes

Why the code is the way it is, one section per area. `CLAUDE.md` states the
rules; this file keeps the reasoning behind each, the history of what went
wrong before the rule existed, and the evidence, in the notes kept while the
code was written. Read the section for an area before changing it. The
measurements quoted here are indexed by date and checkpoint in `verified.md`.

## The project-owned namespace follows Local Diffusion

**Only checkpoint and vendor names use Krea.** App-owned configuration uses
`LOCAL_DIFFUSION_*`; the default database is `local-diffusion.db`; WSL state
lives under `~/local-diffusion`; browser state uses `local-diffusion.*`; and
new PNGs carry their JSON in a `local-diffusion` text chunk. The PNG reader
still accepts the pre-rename chunk and the browser moves pre-rename state on
first read, so existing images and saved prompts survive without making the
old names the write path.

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
now overrides the value on its way to the scheduler (`schedule.forced_mu`), and only
falls back to an explicit pre-shifted schedule (`schedule.shifted_sigmas`, the same
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

`LOCAL_DIFFUSION_QUANT=nunchaku` loads a checkpoint that is *already* quantized (Nunchaku
Lite, SVDQuant W4A4; `quant_method: nunchaku_lite` in the transformer's
`config.json`, bitsandbytes NF4 in the text encoder's) and passes diffusers no
quantization config at all - the repo's own configs name the quantizer, and
torchao on top would be wrong. The CUDA ops arrive through the `kernels`
package from `rootonchair/nunchaku-lite-kernels` (branch `v2`, pinned by
diffusers), which publishes x86_64 **Linux** builds only - torch 2.11-2.13 for
cu126/cu130/cu132 - so on this machine the mode means WSL 2. NVFP4 needs
Blackwell (sm_120 qualifies); the `int4` repos are for older cards.
`loading.nunchaku_preflight` refuses off Linux, without `kernels`/`bitsandbytes`, or
on a repo whose transformer is not pre-quantized, *before* any download: the
bf16 case otherwise loads fine and OOMs at the first image. diffusers reads
`DIFFUSERS_TRUST_REMOTE_KERNELS` once, at import, so `config.py` sets it
(`setdefault`) - it is imported before diffusers everywhere, the CLI included.
Under WSL, `wsl/env.sh` moves the venv (`UV_PROJECT_ENVIRONMENT`), the gallery
(`LOCAL_DIFFUSION_OUTPUTS_DIR`, `LOCAL_DIFFUSION_DB_PATH`) and the HF cache off `/mnt/c`: 9P is slow
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
8.1 GiB peak, 2 MP in 24 s at 8.6 GiB. `LOCAL_DIFFUSION_OFFLOAD=none` does **not** fit
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

**A load waits out the generation gate before it moves the weights.** A
queued job awaits `ensure_loaded()` *before* it takes the gate, and the load
itself never took it, so a job submitted while the prompt critic held the
gate (its `llama-server` has ~5 GB of the card) started loading 13 GB beside
it - `tests/test_api_analysis.py` recorded `load` before `critique done`
with the fix removed. `ensure_loaded()` now awaits `jobs.wait_for_gate()`
inside `_load_lock`, once it knows a load is needed: the gate is acquired and
released at once, which waits out whatever holds it without keeping it. That
covers every load - a job's, `POST /api/model/load`'s, Expand's - and, being
inside the lock, the critique route's "the image model is loading" refusal
covers the wait too. `asyncio.Lock` serves waiters in order, so a critique
that has passed its second busy check either holds the gate already or is
queued ahead of any later load. It cannot deadlock because nothing that holds
the gate awaits the load lock: `_run` loads before it enters, and
`run_when_idle`'s operations are synchronous work that never calls
`ensure_loaded`. (Expand's first use does register ERNIE's rewriter inside
the gate, but through the backend, not the load lock.) The cost: a job that
waits out a critique reports `waiting_on="load"` while the status does not
yet say `loading`, for the critique's fifteen seconds or so.

## The loader is model-agnostic

The loader is model-agnostic: `DiffusionPipeline.from_pretrained` resolves the
class from `model_index.json`, and `_component_libraries()` reads the same file
to decide which components to quantize and with whose `TorchAoConfig`. FLUX
layouts therefore pick up `text_encoder_2` (T5) automatically.

## Anima Turbo: a standard interface over modular inference

**Anima Turbo v1.1** uses the official `circlestone-labs/Anima` selection.
`anima_loading.py` pins the full `anima-turbo-v1.1.safetensors` checkpoint
and its revision, and converts it once on CPU into an atomic cached component
tree. The official pinned `Anima-Base-v1.0-Diffusers` repository supplies
configs, tokenizers, Qwen3 encoder and Qwen-Image VAE only. Its Base denoiser
and text-conditioner weights are excluded: both learned components come from
the Turbo file. Turbo v1.1 uses the `model.diffusion_model.` prefix; split its
`llm_adapter` into `AnimaTextConditioner` and use Diffusers' Cosmos mapping
for the rest, with strict state-dictionary loading. Recipe changes invalidate
the cache. Settings and saved metadata retain the official repository id.
That id denotes this pinned Turbo recipe in this app; adding another Anima
variant must give it a distinct recorded identity.

Diffusers 0.40 exposes Anima through modular blocks, whose pipeline call
does not meet this app's generator/callback signature contract.
`AnimaTurboPipeline` registers the standard components on `DiffusionPipeline`
so the existing owned quantization configs and explicit device/offload path
still apply. It runs the upstream text encoding, conditioning, noise, schedule
and decode blocks. Its only denoising-loop extension invokes the app callback
after each completed scheduler step and lets cancellation propagate. Cleanup
offloads models in `finally`, including a cancelled call, and clears the
callback. CPU generator lists flow unchanged to the upstream noise sampler.
The offload chain is encoder, conditioner, transformer, VAE. The small text
conditioner and VAE remain bfloat16; the transformer and encoder use the
selected fp8/int8 configs. Nunchaku is refused before downloading.

Turbo defaults to 8 steps and CFG 1 (the author recommends 8-12 steps).
A fresh guider is built per call, enabled only above 1, matching recorded
negative-prompt normalization even at CFG 0. The scheduler is the official
Diffusers conversion's FlowMatch Euler with shift 3. Mu, reference images
and user LoRAs are unavailable in this integration. Five-dimensional image
latents are already in Qwen-VAE space; the preview adapter exposes them to
the existing fitted Qwen map. Author documentation covers 512 squared through
1536 squared pixels, tags and natural-language descriptions. ComfyUI sampler
alternatives and prompt-weight syntax are not promised by this adapter.

Sources: [official model card and Turbo files](https://huggingface.co/circlestone-labs/Anima),
[official shared component tree](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers),
[Diffusers conversion](https://github.com/huggingface/diffusers/blob/main/scripts/convert_anima_to_diffusers.py),
and the installed Diffusers 0.40 Anima modular blocks. Runtime evidence belongs
in `verified.md`; CPU stub tests and fake-browser checks do not measure quality
or VRAM.

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
so the profile leaves mu null and `schedule.forced_mu` stays out of the way. Text
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
`lora_unavailable_reason()` refuses `LOCAL_DIFFUSION_QUANT=nunchaku` (already 4-bit
SVDQuant; the modules PEFT would wrap are not `nn.Linear`) and a missing
peft up front, the way `loading.nunchaku_preflight` refuses, and `describe()`
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

**An adapter for another checkpoint is not offered, and is refused by name.**
The directory is the directory, but a LoRA is a set of deltas addressed to
modules by name, and PEFT injects it by looking those names up on the
denoiser: the one FLUX.1 adapter in `loras/` (Frosting Lane, keys under
`transformer.single_transformer_blocks.*`) was listed on Z-Anime, chosen, and
failed at the first cell with a PEFT traceback naming neither model
(2026-09-20). `loras.mismatch` is that lookup run ahead of time: `targets()`
reads what the file addresses through the pipeline class's own
`lora_state_dict` - so a kohya or ComfyUI layout is seen the way the injection
sees it - and finishes the names with `convert_unet_state_dict_to_peft`
exactly as `load_lora_adapter` does (SDXL's kohya conversion stops at
`attn1.processor.to_q_lora.down.weight`), then compares module names and
widths against a `loras.Denoiser`. A family table was considered and
rejected on the cases it gets wrong: FLUX.1 and FLUX.2 [klein] share every
block name and differ in the count, FLUX.1 and Krea 2 share the names and
differ in the width (3072 against 6144), and only the denoiser's own module
list knows either. The `Denoiser` is the live pipeline's once there is one
(`transformer_name`/`unet_name` names the attribute; a PEFT-wrapped target
is read through its `base_layer`) and before that `loading.denoiser` builds
the class from `model_index.json` and the denoiser's `config.json` under
`accelerate.init_empty_weights` - every name and width, no weights, tens of
milliseconds - which is what lets the listing answer at page open. Illustrious
is read from its cached conversion (`single_file.converted_tree`) or not at
all: a listing must never be what starts the 7 GB conversion. Where no
config can be read the fit goes *unchecked*, not refused - the load that
follows either answers it or fails with the real reason - and the page
fetches the list once more when the status turns ready. The listing carries
each adapter's `detail`; the page hides an entry on it and names it in the
tooltip with the reason, so a file dropped into the directory that never
appears is explained rather than missing; `/api/generate`, `/api/grid` and
`/api/rerender` answer 422 with the same reason and the CLI exits 2 with it.
Measured on the frosting-lane file against the cached checkpoints: fits
FLUX.1 Krea; 494 modules Z-Image has not got; 454 klein has not got; the
right names at the wrong width on Krea 2; no `transformer` on SDXL.

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
flags do not name comes from `LOCAL_DIFFUSION_*`. `settings` is frozen at
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
(`owned` in `web/state.js`) — `describe()` reads CUDA memory and can answer
minutes late from behind a load, and one flag for all three, wired to Seed
as well, used to freeze the panel at the fallback's 28 / 3.5 on a Turbo
server because the user pasted a seed in the first two seconds. Reuse owns
all three. Its button warns before the click, and the panel keeps an explanatory
note after the click, when the record's checkpoint is not the one the server is
running: steps, CFG and mu are checkpoint-specific. An empty
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
the larger of two maximum-size 4 MP presets and a full batch (`MAX_BATCH` in
`seeds.py`, the one number the schema, the Batch box and the budget read) at
the largest 1 MP preset aspect. The original two-at-4-MP ceiling was 8,411,136
pixels, which accidentally rejected eight 3:4 or 4:3 images from the nominal
1 MP row; a measured batch of seven used only 10.7 GB of the 16 GB card. The
cap itself was 8 until 2026-09-12, when eight 1 MP images on Animagine XL 4.0
read 9.6 GB of the 16 as read off the card; it is 12 now, which the same
reading puts at roughly 13 GB on that checkpoint. A heavier checkpoint at
Batch 12 is the user's call, as it was at 8: the ceiling was never a VRAM
model. Large batches are still refused, and a machine with different headroom
can set `LOCAL_DIFFUSION_MAX_BATCH_PIXELS` explicitly.

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
that vary across the run (`promptLabels` in `web/sheet.js`): a column header
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
`web/lightbox.js` is a scale relative to fit (never below 1) plus a pan, keyed on
the record's `width x height` and kept across records that share it, so the
arrows compare the identical crop at the identical magnification down a
steps or mu axis. The image is laid out with explicit width/height/left/top
inside a `flex: 1 1 auto; min-height: 0` stage - not a CSS transform, which
can composite a fit-size texture and scale it - and 1:1 divides by
`devicePixelRatio`, or Windows scaling makes it a 1.25x resample, exactly
the smoothing a sharpening sweep is judging. The wheel listener is
non-passive; the zoom keys sit below the `input, textarea` guard. A click on
the stage beside the image closes the lightbox only at fit: zoomed, the
stage is a pan surface and only Escape or the cross close it. Where a press
landed is read on `pointerdown` and kept in `lbDrag`, and the double-click
handler is the stage's and hit-tests the image's box: the stage takes
pointer capture on press, and from then on every event of that pointer -
`pointerup`, the `click` and the `dblclick` made from it - targets the
stage whether the mouse is over the image or not. Testing `e.target` on
the way up closed the lightbox on a single click of the image, and a
`dblclick` listener on the image never heard one; `tools/smoke.json` clicks
and double-clicks the image for both.

**The controls are beside the stage, not under it.** The dialog is a row:
`.lb-main` (the stage and its arrows) and a 280px `.lb-side` column with
the cross and the counter, the prompt and the settings line, the error
slot, the zoom row and the buttons. They used to sit under the stage as a
meta line and a row of buttons, which cost it around 150px of height on a
900px screen - and the screen is wider than it is tall while most images
are not, so height was the axis nearly every fit was bound by. A wide
16:9 image is the one that fits a little smaller now. In the column the
prompt and settings scroll (`flex: 1 1 auto; min-height: 0`) and the
controls are pinned to the bottom, so Favourite and Delete are where the
mouse left them from one image to the next however long the prompt is;
`#lb-close` carries `autofocus`, or `showModal()` would focus the first
control in the tree, which is now an arrow. The arrows are positioned in
`.lb-main`, over the stage's edges but *not* inside the stage: the stage
takes pointer capture on press, so a press on a child of it would have
been a pan, and at fit a close, rather than the button's click. Below
900px wide the column drops under the stage as a wrapping row. The
`tools/smoke.json` grid is 1152×896 for the same reason the layout changed:
at the driver's 1400×900 a 1024×768 image now fits at 104%, and 1:1 is
below fit, which the zoom never goes to.

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
a full unload-while-paused only matters for `LOCAL_DIFFUSION_OFFLOAD=none`.

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
column (JSON; `_migrate`), the `local-diffusion` chunk and `ImageRecord.variables`.
The sheet builds one axis per variable name seen (`axesFor` in `sheet.js`)
and **suppresses** the flat prompt and negative axes when any record
carries variables - kept as fallbacks, a 4-value prompt axis would beat a
2-value variable axis in `chooseAxes` and turn a single-variable run into a
diagonal. `""` (the `none` value) reads as `none`; a record without
variables in a mixed search reads as null. The axis list is rebuilt per
render and the names are part of the layout signature. `completed_cells`
and `CellKey` are untouched: the resolved prompt is what ran.

**A prompt variant records its label as the variable `variant`.**
`GridRequest.prompt_variants` - the prompt analyser's original, reordered and
rewritten versions of one idea - is a list of whole prompts, and each cell
records `variables={"variant": label}`, so a label rides the column, the chunk
and the sheet with no new field. That stretches `variables` from "what the
placeholders were set to" to "which version of the prompt this is"; a label is
not substituted anywhere. Variables could not carry the versions themselves:
`prompt_axis` stops at 2,000 characters where one prompt may be 4,000 (three
long Krea 2 prompts overrun it), `|` splits a value and parsing is a line per
variable, so NewBie's XML - all newlines - cannot be one value, and `none`
and its siblings become empty. A variant is therefore taken verbatim: nothing
is substituted into it, and a `{` in pasted JSON is text, not a placeholder
with no values. The request's own prompt is not rendered; its negative prompt
runs beside every variant, and a placeholder left in *that* is still refused,
since nothing would fill it. Labels are stored trimmed and a label of only
spaces is refused. Identical variant texts are not refused - counted as
duplicates in the preview, as `_expand` treats duplicates elsewhere - and the
page drops them before submitting. `CellKey`, `run_id`, resume and the
duplicate count read the variant's text, so renaming the variants changes
nothing about the run.

**The sheet lists a variable's values in the order the grid wrote them.**
`sheetValues` used to sort a `var:` axis with `localeCompare`, which put
`original` below `reordered` and `rewrite`. It now orders by the oldest image
carrying each value: the grid renders values as declared and variants in the
order given, seeds outermost, so the first image of each is in written order.
That also puts a `size = big | small` grid in declared order, which is what
`parse_variables` already promised; numeric and flat-prompt axes are
unchanged. `chooseAxes` gives a varying `var:variant` axis the rows whenever
anything else varies: widest-first alone made the variants rows only while
the seeds outnumbered them (3 x 3, 3 x 2 and 8 x 4 all turned them into
columns, the tie going to the `var:` axes because `axesFor` lists them first).
Alone, the variants stay a strip in the columns, because rows with nothing
across are one column of full-width images. A variable a user names `variant`
by hand gets the rows too - the one name the sheet recognises, held equal to
`grid.VARIANT` by `tests/test_grid.py`. Known gap: order by oldest id means a
first cell that failed (a job skips a failed cell) or a resume that renders it
later moves that value down for the run.

## Two writers, one INSERT; strict scope; disk estimates

**Two writers, one INSERT.** `save_image` and `reindex.index_file` both go
through `storage._insert_row` over `_COLUMNS`, so the column list exists
once and the FTS trigger fires for both. The reindex reads
the Local Diffusion PNG metadata, parses `created_at` from the filename prefix
as UTC (mtime failing that), copies a foreign file in `.part`-then-rename
and never over a different file of the same name, records `duration_ms=0`
(so `step_cost_ms`'s `> 0` ignores it) and loses `favorite`, which lives in
SQLite alone. `search.strict_scope` is the `key:value`-only reading of the search
grammar for the paths where the LIKE fallback is a hazard: `clear --where`
(`delete_matching`, `DELETE ... RETURNING` in one statement, no orphan
sweep on that path), the gallery's Delete unstarred (the same function,
see its section) and `keep_rates`. `/api/status` carries
`disk_free_mb` (one `shutil.disk_usage` syscall; the directory walk in
`summarise()` deliberately does not ride the 3 s poll) and the preview
carries `bytes` (`image_bytes`: median PNG size at this size on this
checkpoint, times the images) under the same no-rows-no-estimate rule.

**A database-backed import keeps the database-only state.** `reindex --from
<outputs> --database <local-diffusion.db>` reads the source database in read-only mode,
copies only readable PNGs, and inserts through the same `_insert_row` seam.
Unlike PNG-only recovery it preserves `favorite`, `duration_ms`, and every
newer reproduction column the source has. SQLite ids cannot be copied into a
non-empty gallery: source rows are therefore visited in ascending id order and
reference-image ids are translated through the source-to-destination map.
References necessarily point to images which already existed when the
conditioned image was made; a missing or conflicting predecessor is dropped
and counted rather than left pointing at an unrelated destination row. The
copied PNG's Local Diffusion text chunk is replaced with the remapped metadata by
rewriting PNG chunks, not pixels. Filename uniqueness remains the idempotence
key, and an unindexed destination file of the same name remains a conflict,
never an overwrite.

## The CLI reads PNGs and probes the server

**The CLI reads PNGs and probes the server.** `--from-image` reads
the app's PNG metadata from `.info` - not `.text`, which decodes every pixel - and
fills what the flags left unset *before* `apply_env`, so the record's
checkpoint reaches `LOCAL_DIFFUSION_MODEL_ID`; quantization is deliberately not
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

**The panel persists to `localStorage`** (`local-diffusion.panel.v1`) in two phases:
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
`/api/generate` and `/api/grid`: the gallery's own filter (search, the
Favorites flag and the checkpoint scope, taken server-side because the
browser holds one page) at another size, one cell per record with the
record's prompt, negative prompt, steps, CFG, mu and seed, oldest first. It
calls `check_budget()`, refuses more than `max_grid_images` matches by count
rather than truncating, refuses records from another checkpoint (steps/CFG/mu
are the checkpoint's - only `all_models` lets one into the selection),
prunes with `completed_cells` like a resume and 409s when nothing is left.
Same seed at another shape is a *new sample* - the latents are drawn at the
requested size - which the README says out loud.

## Delete unstarred

**The gallery's one bulk delete is the inverse of the card's.** A night
makes hundreds of images and a few of them are worth keeping, and deleting
the rest one card at a time - each with its own confirm - is a chore that
was being skipped, leaving galleries of hundreds. The star already existed
as the gesture made while looking (`F` in the lightbox), it is
non-destructive, and it scales with what is kept rather than what was made;
so the page offers `GET`/`DELETE /api/gallery/unstarred?search=` - the
search box read through `strict_scope`, plus `favorite = 0`, plus the
gallery's checkpoint scope unless `all_models=true` - which is
`clear --where ... --keep-favorites` over HTTP, and shares `delete_matching`
with it. Rows and their files only: no orphan sweep and no vacuum, for the
reason the CLI's `--where` path has none.

**The hazard is inverted with the gesture, and the gate is built for the
inverted one.** A per-card delete goes wrong by taking the wrong image; a
sweep goes wrong by taking the wrong *scope* - a run not yet triaged, or the
whole gallery when one run was meant. So the confirmation is not a yes/no:
`GET` returns the counts and the CLI preview's sample of prompts and jobs,
the dialog shows them, and asks for the number of images to be typed back.
Typing "405" when "12" was meant is the moment the wrong scope is noticed.
The `DELETE` then carries that number as `expected`, and
`storage.delete_matching` compares it against a `COUNT(*)` taken under the
write lock before the `DELETE ... RETURNING`: a sweep that lands more
unstarred images while the dialog is open (the queue keeps running while
one triages) is answered 409 with both numbers rather than swept up on a
count confirmed for fewer. `expected` is required, not optional - the gate
is the route's contract, not the page's manners.

**The scope is strict, judged at the click.** The button shows whenever
the filter matches something and the Favorites tick is off (with it on,
every match is a keeper). Whether the box is a strict scope is the
server's call: free text is refused with a 422 that says what to do,
rather than the page guessing at the grammar with a second copy of the
key list. An empty box is the whole gallery, which is why
`matching_preview` and `delete_matching(keep_favorites=True)` now accept
an empty scope and `clear.py` refuses a blank `--where` itself - the
"whole gallery is `clear` without --where" message is the CLI's manners,
not a storage invariant; the invariant that stayed is that a delete with
*no* predicate at all still raises.

**The page forgets by id.** The `DELETE` returns the ids it removed and
`forgetImages` - factored out of the card's delete, which now goes through
it too - drops them from every list and element on the page in one pass,
then `resetGallery` reloads the (now mostly empty) page from the top for
the server's count. The smoke script stubs `window.prompt` to drive the
dialog: a wrong number first, then the right one, then the free-text
refusal, checking that nothing went until the right number was typed.

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

## The gallery is the running checkpoint's

**`GET /api/gallery` is scoped to `settings.model_id` unless `all_models=true`**,
and so are `GET /api/runs`, `GET`/`DELETE /api/gallery/unstarred` and
`POST /api/rerender` - every route that reads "the gallery's filter". The
page's All models tick, beside Favorites and unpersisted like it, is the one
switch. Steps, CFG and mu are the checkpoint's, so under a gallery that has
seen three checkpoints the grid was mostly settings that could not be reused,
each under a Reuse button that warned, and the working set - this checkpoint's
images - had to be found with a `model:` term remembered every visit.

The default is the *server's* rather than a `model:` term the page adds for
two reasons. The page learns the checkpoint from `/api/status`, and
`describe()` reads CUDA memory, which behind a cold load answers minutes
late - a client-side scope would paint every checkpoint's images first and
hide them later, or show nothing until the load ended. And `model:` is a
LIKE substring, on purpose (`model:flux` for the family), where the scope
has to be exact: a full id matches every longer id that extends it, and
`krea/Krea-2-Turbo` is one suffix away from a catalogue entry that a
substring scope would show under it. So `storage.list_images` takes
`model_id` as a predicate of its own, kept out of the `where` list the FTS
fallback rewrites, and `main._model_scope` turns the flag into
`settings.model_id` or None.

**`other_models` is on every page.** `list_images` answers a third number:
how many rows the *rest* of the filter matches on other checkpoints, one
more `COUNT(*)` over the same predicates with the scope inverted, taken
after the FTS fallback has settled so it counts under the same text
predicate the page did. The tick reads `All models (+40)` from it, and it
is what tells "this run made nothing" from "this run was made on the other
model": a `run:` typed or picked for a run from another checkpoint shows an
empty grid under the scope, and the tick beside it saying `+96` is the whole
explanation. 0 once the tick is on.

**Delete unstarred takes the scope because the grid does.** Its dialog
counts server-side, and with the scope on the grid and off the count, an
empty box would have confirmed "delete 405 unstarred images in the gallery"
against a screen showing 40 - the wrong-scope hazard the typed-back number
exists to catch, built into the number itself. `matching_preview` and
`delete_matching` take `model_id` as one more exact predicate, and the
dialog names the scope from the server's own `models` list
(`... in the gallery (Krea-2-Turbo only)`). Re-render's refusal of another
checkpoint's records stays, now reachable only with All models on; the
Runs picker follows the tick because a run picked from it opens as a sheet,
and a run the grid would not show is an empty one.

`status.js` re-asks for the gallery and the runs when `model_id` *changes*
between status polls with the tick off: the scope is server-side, so a
server restarted under another checkpoint while the page stayed open would
otherwise keep showing the old one's images. The first answer is not a
change - the page loaded under this checkpoint's scope already. The
`tools/fake_server.py` gallery plants two images under a checkpoint it is
not claiming, one starred, so `tools/smoke.json` can watch the tick reveal
them, the badge count them, the Runs picker follow and the dialog name the
scope; `.gallery-head`'s search box has a 140px flex basis for the same
change, because a row break is decided from each item's basis before
anything shrinks and the tick otherwise pushed Delete unstarred onto a
second row at laptop width.

## Paging by id

The gallery pages **by id, not by offset**: `GET /api/gallery?before=<id>`
returns rows older than that id and `has_more`, which the server knows
exactly by fetching one row past the page. OFFSET over a table a sweep is
inserting into repeated the tail of one page at the head of the next and
skipped as many older rows for good; `offset` stays for scripts. The
browser's anchor is the last record it holds, so a delete moves nothing.

The grid pages itself: an `IntersectionObserver` on the "Load more" button
(`moreWatcher`, armed in `app.js`) calls `loadGallery` as the button comes
within 400 px of the viewport, so scrolling to the end keeps going and the
page is usually there before the button is. The observer reports a *change*
in visibility, and a page that does not push the button past the fold - a
wide grid, the tail of a short run - would leave it in view with nothing left
to fire, so `watchMore` re-observes the button after every page that lands:
observing afresh always reports the current state once. It is not re-armed
after a page that failed - re-arming there would fetch again from the server
that just refused - which is why the button stays: it is the retry, and the
visible sign that older rows exist. Hidden in the sheet and once `has_more`
is false, it never intersects.

## LOCAL_DIFFUSION_DEVICE reaches the offload hooks

The offload hooks are handed `LOCAL_DIFFUSION_DEVICE` explicitly (when it is a CUDA
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
neither. `model:` is a substring by design, for the family; the running
checkpoint's scope over the gallery is a separate, exact predicate (above),
not a term the page writes into the box.

## The contact sheet

The contact sheet (`web/sheet.js`, `renderSheet`) lays one run out as the grid
that made it: the two widest-varying axes become columns and rows, any third
becomes a pinned selector rather than a third dimension nobody can read. The
exception is a comparison's `var:variant` axis, which takes the rows whenever
another axis varies - including a variable a user's own prompt grid happens
to name `variant` (see "The variable assignment is recorded"). It
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
hex of the sha256 of `image.tobytes()`; column, metadata key `hash`,
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
in-process with torch's peak allocated (`~/local-diffusion/measure.py` in WSL):
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
operation. `loading.UNUSED_COMPONENTS` therefore still maps the class name to the
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
not a gate - and the tests' conftest now refuses `loading.hub_file` for every test,
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
in-process (`~/local-diffusion/measure.py`), Z-Image Turbo re-run the same
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
scheduler as flow matching and replaced its schedule with `schedule.shifted_sigmas`.
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
× button and accepts Delete/Backspace while focused. A context click in any
flex gap finds the nearest wrapped row and the boundary between the adjacent
chip centres; a context click on a chip inserts after it. Its
visible listbox reuses the same cached tag-suggestion endpoint; a native
`datalist` did not reliably reopen when asynchronous results arrived. Click,
Enter or Tab accepts a suggestion, while + commits literal text and Escape or ×
cancels. A double click on a chip (Enter or F2 while it has focus) opens the
same editor in the chip's place, holding its text, so a typo is fixed where it
sits rather than by deleting the chip and inserting a new one; suggestions
start with the first keystroke, not with the opening, so Enter on an untouched
edit keeps the tag instead of taking the top match for it, and an unchanged
commit only closes. One editor is open at a time, and any prompt change from
outside it - the textarea, a drop, a restored panel - closes it, because its
index no longer names the item it opened on. Every edit goes through the same
preview, validation and panel-persistence path, so the visible textarea,
generated prompt and chips cannot become separate sources of truth.
Chip and suggestion text is 12 px. A valid chip uses high-contrast light text
on neutral charcoal, with green confined to its checkmark and outline; unknown
chips keep the corresponding red mark/outline instead of colour-on-colour text.

## Illustrious XL v2.0-STABLE: a single file converted once

The official `OnomaAIResearch/Illustrious-XL-v2.0` repo has only the 6.94 GB
`Illustrious-XL-v2.0.safetensors`, not a `model_index.json`. Merely adding it
to the catalogue would fail before loading under the default FP8 component
scan. Diffusers 0.40's pipeline `from_single_file` also does not distribute a
`PipelineQuantizationConfig.quant_mapping` to its CLIP loaders. Rather than
invent a second quantization path, `backends/single_file.py` converts the
official checkpoint to an unquantized bfloat16 component tree once on CPU.
The ordinary loader then scans and quantizes that tree with the same
per-library TorchAoConfig instances as every other model.

Only this exact catalogue repo id takes that branch. The source is pinned to
`69459c1fe6f46db41ab31e6114f05acc0e06bcaa` (STABLE), and the SDXL base
descriptors to `462165984030d82259a11f4367a4eed129e94a7b`; the config download
allows JSON and tokenizer text only, never SDXL base model weights. The
conversion fixes epsilon prediction and Euler Ancestral, the sampler used
in the authors' report, and saves that scheduler in the tree. This is a
fixed model configuration, not a per-request sampler setting.

The cache is `$HF_HOME/local-diffusion/converted`, separate from the Hub's
managed snapshots. Its key hashes the source, config and versioned recipe.
A process lock covers conversion, and a temporary directory is renamed only
after `save_pretrained` and a recipe marker complete. Failure cannot publish
a partial tree. The same tree supports FP8, int8 and none; Nunchaku is refused
before downloading. Both original and converted weights remain on disk,
about 14 GB total. Only the loader's local settings copy sees the cache path:
status, profiles, cell keys and PNGs retain the official repo id.

The app's starting values are 28 steps and CFG 5. The report gives >20 steps
and CFG 5-7.5, not a unique default; the authors' v2.0 release example runs
28 steps at CFG 6.5 (written as 2.5 under DPM++ 2S Ancestral CFG++, with
their own note that it "should be 6.5 in recent implementations"), and the
control hints quote both. Its v1.0+ native range is 1-2.25 MP, and
up to 4 MP carries some loss. Figure 19's direct 2048x2048 result is explicitly
v1.0; it is family context, not evidence that STABLE was measured here. The
release post adds that v2.0 "supports 512~1536 resolutions", trained to 1:10
in multiples of 32, and then renders its own example at 1824x1248; the size
hint reports both as written.

The prompt hint opens with what the authors say the checkpoint is, because
that is why a bare prompt disappoints next to Animagine: the v0.1 card calls
the family "untuned BASE model ... not tuned for aesthetics / preferences"
and notes "The model does not have any default style. This is intended
behavior for the base model"; the v2.0 release post repeats it - "not
trained with aesthetic set", released "'untuned'" as a "merging / training"
base - and the report says the non-fine-tuned release is deliberate, with
styles associated with artist names. Animagine 4.0 is an aesthetic-tuned
finetune whose score tags pull on training it received; Illustrious has
nothing for such a tail to pull on, so the hint sends the style to an
artist or style tag (the Danbooru lookup already labels the artist category)
and passes on the v0.1 card's composition advice: one of `upper body`,
`cowboy shot`, `portrait` or `full body`, and not several conflicting ones.
The v2.0 examples demonstrate natural language, but the report also describes
every Illustrious version as tag-based and gives its caption order: person
count, character, rating, general tags, artist, quality and year. The shared
Danbooru autocomplete and chip editor therefore apply to Illustrious too, with
the archived v2.0 card's `2024-08` dataset update represented by an inclusive
`2024-08-31` vocabulary cutoff. The picker exposes the terms written in the
report - `1girl`/`1boy` from its example prompts, the six quality tiers in table
3 and the five year modifiers in table 4 - plus a Rating group holding
Danbooru's four rating names, `general`, `sensitive`, `questionable` and
`explicit`. That group exists because the rating slot cannot be filled any
other way: a Danbooru rating is a post field, not a tag, so the index carries
those words only as deprecated, empty placeholders that autocomplete drops and
validation refuses, and a local group is the one path that accepts them. The
report itself writes only `general` (its example prompts, and its note that
`general` in the prompt is a simple safety control); the other three are the
dataset's vocabulary, not the authors', and the Prompt hint says so (the
release post also lists `sfw`, `nsfw`, `r-18` and similar as working
alternatives; the hint does not repeat them). Animagine's `safe`/`nsfw`
spelling, quality suffix and numeric `year {n}` form are not copied over.
Because Illustrious also supports natural language, spellcheck remains
enabled and a single sentence is not rendered as one unknown tag chip;
comma-separated tag prompts get the complete editor. At the user's request,
the Prompt hint also turns the documented quality-then-year order into a
clearly qualified practical starting tail, `masterpiece, newest`; it stays
out of a Finish-with button because no author prompt ends that way (theirs
end in `masterpiece` alone, or put it mid-prompt).

The negative prompt was first left empty, on the reading that the authors had
published none for v2.0, with an app-invented baseline offered as
"experimental". That reading was wrong, and it hurt: the v0.1 model card runs
both of its examples with one negative prompt, verbatim `worst quality,
comic, multiple views, bad quality, low quality, lowres, displeasing, very
displeasing, bad anatomy, bad hands, scan artifacts, monochrome, greyscale,
signature, twitter username, jpeg artifacts, 2koma, 4koma, guro, extra
digits, fewer digits` (the second example appends `jaggy lines, unclear`),
and the v2.0 release example runs `worst quality, bad quality`. Since
2026-09-19 the helper's `negative_prompt` is the v0.1 card's list, so the
Tags menu offers the same "Use the recommended negative prompt" button
Animagine has, and the hint names both forms and their source rather than
an app recipe. The `displeasing`/`very displeasing` pair in it is the only
trace of aesthetic modifiers in any author text; the report says such
modifiers were used in later epochs and defers details, so no positive
aesthetic tag is suggested. The example menu now holds three author prompts
that fit in 77 tokens: the report's figure 3 prompt, the v0.1 card's
silhouette prompt and the release post's forest scenery; the post's
natural-language example is ~200 tokens and would be truncated silently, so
the hint quotes only how it ends (`persona 3, masterpiece`, tags after prose).
Community guides read for this - SeaArt's release-day user guide, which
fronts `masterpiece, best quality, amazing quality`, adds `very aesthetic,
newest`, puts CFG's sweet spot at 4.5-5 and says the model follows negatives
as closely as positives - are not authors' text and contribute nothing to
the hint. Their "Clip Skip 2" is diffusers' SDXL default already
(`hidden_states[-2]` in `encode_prompt`), so nothing needed changing there.

Sources checked 2026-09-17 and 2026-09-19:
- Official release/card and files: <https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0>.
  Its "ILXL Image Generation User Guide" link (`illustrious-xl.ai/updates/21`)
  and the site's blog and updates pages answered 404 on 2026-09-19.
- Archived v2.0 card with the dataset-through-2024-08 statement:
  <https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0/blob/f9885efc7cbd20d0291dbbb0d5fe810ea143911f/README.md>.
- The authors' v2.0 release post on Civitai - untuned base, natural-language
  example, 512-1536 and 1:10 in multiples of 32, the 1824x1248 example with
  its negative prompt and settings, the rating tokens - read through
  CivArchive because Civitai itself refuses this region:
  <https://civarchive.com/models/1369089?modelVersionId=1546777>.
- The v0.1 card: sampler, steps and CFG, the quality tiers, the composition
  advice, "no default style", and both example prompt/negative pairs:
  <https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0>.
- Authors' report, B.2 settings, B.1.4 language, figure 19 size, figure 3 prompt:
  <https://arxiv.org/html/2409.19946v1>.
- Diffusers single-file API: <https://huggingface.co/docs/diffusers/api/loaders/single_file>;
  the installed 0.40 `load_single_file_sub_model` confirms the CLIP quantization boundary.

Validation on 2026-09-17: 333 Python tests and the 15-check Illustrious browser
smoke passed. The 6,938,040,674-byte official checkpoint was verified against
SHA-256 `c2a1a3eaa13d4c107dc7e00c3fe830cab427aa026362740ea094745b3422a331`.
Conversion succeeded with the installed Diffusers 0.40 / Transformers stack,
including the checkpoint's extra CLIP `position_ids` buffer (the loader skips
non-state keys). The cached tree is 6,945,302,591 bytes. A second process reused
it without conversion and loaded the FP8 pipeline in 7 seconds. All three test
PNGs retained the official model id and requested settings, not the cache path.
GPU samples and their quality limits are recorded in `verified.md`. The
2026-09-19 prompting rewrite passed 335 Python tests and the 19-check
Illustrious smoke on the fake server; no GPU sample was taken for it, since
it changes text and one button, not the pipeline.

## WAI Illustrious v17: an author checkpoint through a pinned mirror

`LyliaEngine/waiIllustriousSDXL_v170` selects WAI0731's v17 full SDXL
checkpoint through a community transport mirror. The mirror is neither the
author's repository nor a LoRA, despite misleading Hub metadata. Its card
is not authoritative for the loading recipe or permissions. The primary
author source is the Tensor.Art v17 release page. The file's SHA-256 matches
the author's Civitai version 2883731 as recorded by CivArchive; direct
Civitai access returned HTTP 451 during integration.

The source is pinned to revision
`5ef4e2da7173a160ad04aebcaa2fdcd6d20ed792`, filename
`waiIllustriousSDXL_v170.safetensors`, 6,938,040,682 bytes, SHA-256
`f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04`.
`single_file.py` verifies that digest before converting. It shares the
Illustrious CPU conversion path and pinned SDXL base descriptors at
`462165984030d82259a11f4367a4eed129e94a7b`: only JSON and tokenizer text are
downloaded from the base, never replacement weights. The checkpoint's VAE
and both CLIP encoders are retained. The saved scheduler is epsilon-prediction
Euler Ancestral, matching the author's Euler a recommendation.

The existing lock, temporary tree, recipe marker and atomic rename protect
the conversion cache. WAI's distinct source and digest produce a separate
cache key; the existing Illustrious recipe remains unchanged. The cached
tree feeds normal per-component FP8, int8 or bfloat16 loading, and Nunchaku
is refused before conversion. The original and converted weights occupy
about 14 GB together. Only the loader sees the cache path: status, saved
metadata and resume identity retain `LyliaEngine/waiIllustriousSDXL_v170`.

The source contains NaNs in CLIP-L's final transformer layer (layer 11).
They are preserved as source weights, not silently replaced or sanitized.
Standard SDXL prompt conditioning selects this tower's penultimate hidden
states (`hidden_states[-2]`) and takes pooled conditioning from the second
text tower. Consequently the contaminated final CLIP-L output is not a
conditioning input. A focused regression preserves that distinction by
checking the consumed conditioning rather than rejecting an unused output.
Changes to hidden-state selection must revisit this assumption.

The exact-id profile starts at 28 steps and CFG 6, within the author's
15-30 and 5-7 ranges. These are starting settings, not a measured optimum.
Mu is unavailable because the scheduler is not flow matching; negative
prompts use ordinary CFG above 1. The guide does not inherit Illustrious
v2.0's untuned-base description, native size range, ratings or negative
prompt. WAI has its own author-supplied positive quality recipe and negative
prompt. The quality group is nonexclusive because all three positive tags
are intended together, while the four rating buttons are exclusive.
Examples inserts the author's verbatim quality recipe rather than an
invented scene, and the hint explains that the subject still needs adding.
The family release notes describe natural-language support, so the helper
keeps spellcheck and single-sentence prose behavior. Standard SDXL CLIP's
77-token context applies; the author's weighted-prompt syntax is not parsed.

WAI publishes no dataset cutoff. Its Danbooru helper therefore passes
`None` to the tag client, which omits only the creation-date filter and
still excludes deprecated and empty tags. Existing dated profiles keep
their limits, including when they share cached tag records with WAI.
WAI's quality and rating tokens validate locally; remote matches remain
advisory vocabulary membership, not evidence of checkpoint training.

The author recommends original dimensions above 1024x1024 and describes
1024x1344 example inputs followed by a separate 1.5x hires refinement pass:
20 hires steps, R-ESRGAN 4x+ Anime6B, denoising strength 0.35-0.5. No maximum
is published. The existing guide contract uses the documented reference
span, 1.048576-1.376256 MP, with `pixels_word="documented"`; the tooltip
explains that this is neither a maximum nor an equivalent to the refined
examples. This app renders directly at the requested size and does not
implement that hires pass. The fake-server smoke is `tools/smoke-wai.json`
on port 7873; actual card measurements belong in `verified.md`.

Sources checked 2026-09-26:
- [Author's v17 release card and permissions](https://tensor.art/models/990778216270553015/WAI-illustrious-v17).
- [Pinned community transport repository](https://huggingface.co/LyliaEngine/waiIllustriousSDXL_v170/tree/5ef4e2da7173a160ad04aebcaa2fdcd6d20ed792).
- [CivArchive's author-model record, including version 2883731](https://civarchive.com/models/827184/wai-nsfw-illustrious-sdxl).

## Z-Anime Base: a nested Diffusers tree

`SeeSee21/Z-Anime` is the next anime candidate after Illustrious. The author
publishes Base, distilled, GGUF and AIO weights in one repository, with the
Base component tree under `diffusers/`. The card's suggested pipeline-level
`subfolder="diffusers"` is not implemented by the installed Diffusers 0.40
`DiffusionPipeline.from_pretrained`/`download`: they still seek a root
`model_index.json`. Passing the nested local directory works through the
existing loader and its component-specific quantization path.

`loading.resolve_model` pins revision
`0f5fb51464638a2f7328a1d74590281e63e1fde2`, downloads only the manifest and
component/scheduler configs first, then lets the usual quantizer preflight
run. Only after that succeeds does it materialize the `diffusers/`
subtree, excluding its incorrectly packaged text-encoder weights (see below).
Both passes share the revision, so the preflight cannot inspect one
version and load another. Runtime settings, status, run keys and saved
metadata retain `SeeSee21/Z-Anime`; ordinary ids and local directories pass
through unchanged. Nunchaku is refused before weights because the denoiser is
unquantized, with no catalogue Lite counterpart.

**The supplied encoder is not loadable as declared.** The real first load
reported 253 unexpected `*.scale` tensors. Inspection of the cached
`text_encoder/model.safetensors` found 253 FP8 tensors and scalar BF16 scales
(for example, layer 0's Q projection scale is 0.0013275146484375), but its
config has no quantization declaration. Transformers drops those scales;
casting the raw FP8 values to BF16 or applying torchao cannot restore them.
The initial render run was stopped and is not verification evidence.

The card identifies the default text encoder as standard Z-Image Qwen3-4B,
the one the model was trained against. `loading.component_overrides` therefore
loads `Qwen3ForCausalLM` from the official `Tongyi-MAI/Z-Image` parent's
`text_encoder/`, pinned at `04cc4abb7c5069926f75c9bfde9ef43d49423021`, and
passes it into the pipeline. It receives the existing Transformers-owned
`quant_mapping["text_encoder"]`, or BF16 for `none`; the denoiser still uses
the normal Diffusers-owned config. This does not select the separately
advertised Engineer V4 alternative. The source encoder weights are excluded
from new downloads. The corrected combination totals about 20.7 GB (19.3 GiB)
in the HF cache; a machine that previously downloaded the malformed encoder
also retains that 4.8 GB until the owner clears it.

The profile uses the author's Diffusers example: 40 steps, CFG 4, Mu blank.
The real `ZImagePipeline` exposes generator and step callbacks, guides above
zero, and accepts negative prompts. The guide favours descriptive sentences
and carries the author's portrait prompt template in Examples. There is no
Danbooru tag helper. The card documents approximately 512x512 to 2048x2048,
so the guide says **documented** 0.26-4.2 MP rather than claiming a measured
training range. Its 1024x1536 and 1920x1088 suggestions are explained in the
size tooltip; the latter is already the 16:9, 2 MP preset.

The pinned scheduler is `FlowMatchEulerDiscreteScheduler`, static shift 6,
with beta sigmas and stochastic sampling disabled. Preserve it. The card's
ComfyUI recommendation (Euler Ancestral/beta) is a different sampling setup;
no equivalence or quality superiority is claimed. In particular, do not copy
Z-Image Turbo's shift-3 Mu hint or its 8-step/CFG-0 defaults. Mu remains
available as an explicit override under the existing flow-schedule rules.

Sources checked 2026-09-19: [author's model card](https://huggingface.co/SeeSee21/Z-Anime),
[pinned manifest](https://huggingface.co/SeeSee21/Z-Anime/blob/0f5fb51464638a2f7328a1d74590281e63e1fde2/diffusers/model_index.json),
[pinned scheduler](https://huggingface.co/SeeSee21/Z-Anime/blob/0f5fb51464638a2f7328a1d74590281e63e1fde2/diffusers/scheduler/scheduler_config.json),
[standard encoder config](https://huggingface.co/Tongyi-MAI/Z-Image/blob/04cc4abb7c5069926f75c9bfde9ef43d49423021/text_encoder/config.json),
and the installed Diffusers pipeline loader and `ZImagePipeline` source.
`tests/test_nested_models.py` checks preflight ordering, quantization, tree
selection, the pinned encoder override and the retained public id; `smoke-z-anime.json` checks the UI on
port 7868. Card measurements belong in `verified.md`.

## Kohaku XL Zeta: ordinary SDXL with its own hybrid prompt guide

`KBlueLeaf/Kohaku-XL-Zeta` publishes a complete Diffusers component tree, so
the catalogue entry uses the normal loader. Keep its supplied epsilon-prediction
`EulerDiscreteScheduler`; its accepted custom sigmas do not make it a flow
scheduler. The profile disables Mu before load, as the live scheduler does
afterwards. The app's 28 steps and CFG 5 are starting points within the author's
12-50 and 3.5-6.5 ranges, not a separately measured optimum. No Nunchaku
repository is mapped to this entry.

`KOHAKU_GUIDE` uses Zeta's documented tag vocabulary and the Epsilon-rev2
caption order that Zeta refers to. The general-tag slot can hold prose or a
mixture of prose and tags. `hybrid_prompt` preserves spellcheck and avoids
validating a single sentence as one unknown Danbooru tag. The shared tag
service uses the model's 2024-07-10 Danbooru cutoff; no frontend checkpoint
branch is required. Quality and date choices are exclusive. Rating choices
remain individual toggles because the author's explicit-rating caption
contains both `nsfw` and `explicit`. There is no fixed suffix or recommended
negative-prompt button: the Zeta card supplies neither recipe.

The author's size recommendation is 1024x1024 or a comparable area, so the
guide publishes one recommended area in MP and lets the common tolerance
admit nearby 1 MP presets. It does not turn the training bucket's maximum
side length into a promise of large square outputs. The card also describes
300-token training context, but both published tokenizer configs declare
`model_max_length: 77`. The normal SDXL pipeline truncates to those limits;
no long-prompt embedding or chunking path has been added. The prompt hint
states that distinction.

The Examples menu takes `DEFAULT_TAGS` verbatim from the author's TIPO demo,
whose `diff.py` selects Kohaku XL Zeta by default. The twelve Zeta model-card
JPEGs contain no prompt metadata. Hugging Face's auto-generated library
snippet is not an author example. The demo's prompt is a useful author input,
not a claim that this app reproduces its separate K-diffusion sampling and
prompt-expansion setup. Hardware checks and their exact settings belong in
`docs/verified.md`.

Sources checked 2026-09-26:

- [Zeta model card](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/README.md).
- [Epsilon-rev2 caption order](https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon-rev2#prompt-format)
  and [Delta's tag spelling](https://huggingface.co/KBlueLeaf/Kohaku-XL-Delta#tags).
- [Supplied scheduler](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/scheduler/scheduler_config.json),
  [first tokenizer](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/tokenizer/tokenizer_config.json),
  and [second tokenizer](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/tokenizer_2/tokenizer_config.json).
- Author's [TIPO demo input](https://huggingface.co/spaces/KBlueLeaf/TIPO-DEMO/blob/1ae60d3cbec4f5274ff349321b71817cfde60175/app.py)
  and [image backend](https://huggingface.co/spaces/KBlueLeaf/TIPO-DEMO/blob/1ae60d3cbec4f5274ff349321b71817cfde60175/diff.py).

## NetaYume Lumina v4: pinned denoiser and standard Lumina 2 components

`duongve/NetaYume-Lumina-Image-2.0` selects the author's full-precision
`Unet/v4/NetaYumev4_unet.safetensors`, pinned to
`69c29c441119d43405d8373c654b16827d3cf46e`. The repository also carries v2,
v2-plus, v3, v3.5-pretrained and all-in-one files; the menu explicitly names
v4 so the source recipe is unambiguous. Another release must have a distinct
recorded identity rather than changing this recipe's weights silently.

`netayume_loading.py` converts the 5.22 GB BF16 split checkpoint on CPU.
Diffusers' `convert_lumina2_to_diffusers` maps the original names and splits
QKV; it removes the original unused `norm_final.weight`. The converted state
must load strictly into the pinned parent transformer config. Header-only
validation of the published file found 400 source tensors and 459 converted
tensors, matching every expected key and shape. Missing, extra or wrongly
shaped weights fail instead of leaving random parameters.

The author identifies the encoder as pretrained Gemma-2-2b and the VAE as
pretrained FLUX.1-dev. Their descriptors, tokenizer and weights come from
the official `Alpha-VLLM/Lumina-Image-2.0` revision
`53504abd8178b30685b6c4c7a4cd181ff78b73e9`, together with the transformer
config and scheduler. The allowlist excludes all parent denoiser weights.
No community conversion or remote Python code is needed. A recipe hash,
process lock, temporary directory and final rename provide the same atomic
cache contract as Illustrious and Anima. FP8/int8 quantization applies only
after conversion through the normal per-library `quant_mapping`; full
bfloat16 uses the same tree. Nunchaku is refused before downloads.

The resulting pipeline is standard `Lumina2Pipeline`, which already takes
per-image generators and `callback_on_step_end`. The supplied scheduler is
FlowMatch Euler, static shift 6, with no terminal or beta post-processing.
Mu uses the existing whole-shift override. Negative prompting applies above
CFG 1, and saved True CFG is that effective scale. The pipeline retains
its default normalized CFG, full-step guidance and 256-token sequence.
The author's anime system instruction is passed explicitly on every call;
Lumina adds its `<Prompt Start>` separator. The user prompt remains the
stored image prompt. The fixed system instruction is part of the conversion
recipe, and is not duplicated in Examples or the editor. Negative text is
encoded as the user supplied it. These semantics and standard Euler are
not the author's ComfyUI Euler Ancestral/normal or
res_multistep/linear_quadratic workflow. The existing FLUX VAE preview map
applies to Lumina's unpacked 4-D latents. User LoRAs use the ordinary
compatibility check and hot-swap path; references are unavailable.

The exact-id profile starts at 40 steps and CFG 5. The author card recommends
40-50 and 4-7, but still describes v2, so the guide labels that family
guidance rather than inventing v4 tuning evidence. The only documented
sizes used by this guide are Neta Lumina's 1024-square example and the
author's v2 1024x1536 workflow. Their 1.05-1.57 MP span drives the advisory
highlighting; its prose explicitly says it is an example span, not a
supported maximum. The example prompt is the author's workflow image
prompt with only the system prefix and separator removed. No vocabulary
cutoff is published for v4, so no tag validator is guessed.

Sources:
- [Pinned author card](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/69c29c441119d43405d8373c654b16827d3cf46e/README.md).
- [Pinned v4 split file](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/69c29c441119d43405d8373c654b16827d3cf46e/Unet/v4/NetaYumev4_unet.safetensors).
- [Author v2 workflow](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/69c29c441119d43405d8373c654b16827d3cf46e/Lumina_image_v2_tensorart_workflow.json).
- [Pinned official Lumina 2 components](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/tree/53504abd8178b30685b6c4c7a4cd181ff78b73e9).
- [Neta Lumina base guidance](https://huggingface.co/neta-art/Neta-Lumina/blob/65c078e75781946e9f3423fe41457adae3eba3cd/README.md).
- Installed Diffusers 0.40 `pipeline_lumina2.py` and `convert_lumina2_to_diffusers`.

Runtime measurements belong in `verified.md`; conversion tests and browser
smoke checks alone do not measure output quality, speed or VRAM.

## NewBie Exp0.1: pinned components and an explicit sampling recipe

`NewBie-AI/NewBie-image-Exp0.1` selects the official release at
`4f7825a1a7e1ed5c1882a52e3c51d1119a43e298`. Diffusers 0.40 has no
`NewbiePipeline`, despite the card's quickstart, and the published tensors
need explicit mappings. `newbie_loading` strictly converts them on CPU into
an atomic cache, preserving the official id in status and saved metadata.
The recipe hash includes the source revision, scheduler, prompt prefix and
CFG normalization so changes to those cannot silently reuse an older tree.

`newbie_transformer` extends the installed Lumina 2 blocks with the release's
pooled-CLIP conditioning. Gemma 3 supplies penultimate hidden states. The
separate Jina text tower uses native Transformers layers and the author's
unnormalized masked mean; its merged `jina-clip-v2.safetensors` is distinct
from the larger file containing a vision tower and adapter parametrizations.
No remote Python or unused vision weights are needed. Both encoders use
normal forward calls so model-offload hooks work. The native Jina classes
require Transformers 5.16.1 or newer. Quantization is applied after conversion
through the ordinary per-component loader.

The exact-id profile starts at **28 steps and CFG 4**: 28 is the model-card
example, while the author's sampling script uses CFG 4 with midpoint and
18 steps. That script is evidence for a starting guidance value, not proof
that this app's sampler reproduces its output. The author-proposed Diffusers
pipeline instead defaults to CFG 5. These author sources disagree; none is
presented as a measured optimum here.

This app fixes one recipe: FlowMatch Euler with static shift 6, an empty
system prefix for both encoders, CFG throughout sampling above 1, and
whole-sample norm limiting at the conditional prediction's norm. The empty
prefix follows the author's Diffusers proposal. Their training/demo code
also supports system-prefixed Gemma captions but keeps Jina captions raw;
the community proposal's generic Lumina prefix on both encoders is not used.
The norm cap follows the original model, rather than the community
proposal's unconditional rescaling along only the final spatial dimension.
Transformer time is `1 - scheduler_t / num_train_timesteps`, with model
velocity negated for the scheduler, as in the author's Comfy adapter. The
author Diffusers proposal's unscaled timestep is not copied. Blank Mu keeps
shift 6; an explicit Mu goes through the existing whole-shift override.

`NewbiePipeline` exposes one generator per image and the step callback used
for progress, preview and cancellation, with offload cleanup on exit. Its
Gemma input has a 512-token limit and Jina has 2048, following the author's
encoder path and keeping the supplied XML example intact. References
and user LoRAs are disabled; there is no Nunchaku mapping. The author XML
example remains editable text, without comma-tag chip parsing. The size
guide highlights the card's 1024-square example area and explicitly says
it is not a minimum, maximum or quality limit.

Sources, checked 2026-09-26:

- [Pinned card and official files](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/tree/4f7825a1a7e1ed5c1882a52e3c51d1119a43e298).
- [Author sampling settings](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/709e9a482fb2b861ad6aa50dcafb0198cad66c1d/scripts/sample.sh),
  [original CFG implementation](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/709e9a482fb2b861ad6aa50dcafb0198cad66c1d/models/model.py),
  and [training caption paths](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/709e9a482fb2b861ad6aa50dcafb0198cad66c1d/finetune.py).
- [Author Diffusers proposal, PR 12789](https://github.com/E-Anlia/diffusers/blob/694e38842aed2b53cb6a0b248e0799aceaf89101/src/diffusers/pipelines/newbie/pipeline_newbie.py).
- [Author Comfy timestep adapter](https://github.com/E-Anlia/ComfyUI-NewBie/blob/29655ed6fa86551abbf1835511764ff151dbbbc5/comfy/model_base.py)
  and [shift configuration](https://github.com/E-Anlia/ComfyUI-NewBie/blob/29655ed6fa86551abbf1835511764ff151dbbbc5/comfy/supported_models.py).
- [Community Diffusers proposal, PR 12803](https://github.com/huggingface/diffusers/pull/12803),
  implementation reference pinned to `e061f982c6c1e162309f9df919662e6fa61e60ea`.
- [Native Jina architecture reference](https://huggingface.co/jinaai/jina-embeddings-v3/blob/343dbf534c76fe845f304fa5c2d1fd87e1e78918/config.json).

`tools/smoke-newbie.json` checks the profile, XML example selection and
persistence, size warning and saved settings using the fake server on 7872.
CPU and browser checks do not establish GPU performance or image quality;
hardware evidence belongs in `docs/verified.md`.

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
schnell in BFL's reference, not in this app: see the prompt analyser's open
questions), Qwen-Image's encoder is prompted to *describe the image by
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

**A pair of inverses is one box on the strip.** The thirteen aspects used to
sit in the order they were added - landscape/portrait pairs, but 16:9 ahead
of 3:2 and 9:7 behind both - and the strip, `Object.keys` of the table, showed
that. The table now lists 1:1 and then each portrait framing followed by its
landscape inverse, pairs from nearest-square to widest; `buildPresets` wraps
two adjacent inverses in a `.pair` container drawn as a segmented box, so
the pairing is the table's order and nothing else, and `syncSize` walks the
strip's buttons rather than its children. `test_the_aspects_are_in_strip_order`
holds the order, and the browser and README copy tests compare key order along
with the values. Portrait first in a pair is how the request was phrased.

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

**The guide is also data: `analysis`.** Each guide dict carries its
checkable statements as an `analysis` entry for the prompt analyser's rule
checks (`prompt_rules`) - a documented order's steps, a word range, "quote
the text", "describe what you want" - and every entry holds the words it
rests on as `cite`, a verbatim piece of that guide's own `prompt` prose (or,
for `author.placeholder`, of one of its examples), with a URL of its own or
the guide's. The prose stays the claim and the entry only points at it; the
next section says why that is enforced by a test. `analysis` is a key inside
the guide, not a profile key - a top-level one would reach
`ModelStatus.defaults` through `defaults_for` and fail its validation - and
is left out of `guide_for`, because the page shows findings, not the rules
behind them. `FALLBACK_GUIDE`'s is empty. Sources the entries cite beyond
those above, checked 2026-09-27: BFL's unified guide covers the whole FLUX
family (FLUX.1, Kontext, FLUX.2; <https://docs.bfl.ai/guides/prompting_summary>),
and its technical page (<https://docs.bfl.ai/guides/prompting_unified_technical>)
is where "working without negative prompts" lives, with a table of
replacements ("no people" to "empty", "deserted", "solitary") and "Front-load
the positive description — word order signals priority"; its building page
(<https://docs.bfl.ai/guides/prompting_unified_building>) carries the 30-80
word table and "Start with a clear subject". Krea 2's
<https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md> is the
natural-language, long-detailed and quotes guidance, and its companion
<https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt> - the system
prompt Krea publishes for an LLM of your choice to expand prompts with - is
where "one cohesive paragraph" comes from ("Write one cohesive paragraph
after the thinking block. No bullets, JSON, or markdown").

## The prompt analyser

The request was a tool that says whether a natural-language prompt is
structured right and in the right order, and how to improve it. The research
behind it (2026-09-27) found that "the right order" has an author's answer
on only some checkpoints: BFL's family guide puts the subject first for FLUX.1,
its FLUX.2 guide gives subject, action, style, context for klein, the tag
checkpoints' cards and Illustrious's report give tag orders, and NewBie's
captioning instruction gives XML field orders - while Krea 2's guide gives no
order, Qwen-Image's only in the 2512 rewriter's portrait flow, Z-Image's
rewriter a processing order rather than an output one, ERNIE's only in a
rewriter prompt its README does not link, Z-Anime's only for its 4-step
variant, NetaYume's only from its parent, and WAI's only its quality recipe
at the end (checked as a two-step order: subject and scene, then quality).
So there are three layers, each answering what the one before cannot, and
each held to the one before:

1. **Check** (`POST /api/prompt/analyze`) - what the checkpoint's own
   tokenizers do to the text (`prompt_encoding`) and what the guide says that
   a rule can check (`prompt_rules`). Deterministic, instant, on the CPU, and
   sourced by construction.
2. **The critic** (`POST /api/prompt/critique`) - what only reading can
   judge: whether the subject really comes first in a sentence, whether "shot
   on film" is a camera, what a better sentence would be. A local model held
   to numbered author sources, told the rule findings, and its rewrite put
   back through layer 1.
3. **Compare** (`GridRequest.prompt_variants`) - where the authors wrote no
   order, or an order is only a claim, the user's own seeds on this checkpoint
   are the evidence: original, reordered, rewrite and the user's own version
   as one grid, one sheet row each. A "reordered" row is a pure permutation
   of the original, so a difference between the two rows is the order and
   nothing else - which is why the user's own version exists: on a sentence
   prompt Check has no reordering and the critic's rarely survives
   `is_reorder` (next paragraphs), so the reorder the user writes is the only
   one there is.

**Check works cold.** It takes the body Generate would send
(`PromptAnalysisRequest`), because half of what a checker says depends on
the rest of it: a negative prompt at CFG 0 on Krea 2 Turbo, or without True
CFG on FLUX.1, never runs, and that is `applied()`'s answer, which
`backend.applied()` gives from the profile before a load. It runs off the
loop, never calls `ensure_loaded` and never takes the gate, so it answers in
a moment on a cold server and beside a running night. The negative prompt is
measured and checked only as `applied()` runs it: a budget for a negative the
pipeline drops is a number about nothing, and a negation in the prompt is a
warning only where no negative prompt runs.

**The encoder layer measures the string the pipeline builds.** No pipeline
here parses A1111 or ComfyUI weighting - diffusers has no prompt weighting,
and compel and sd_embed are not installed - and every overflow is cut
silently from the right: the SDXL negative prompt, every Qwen-encoded
pipeline, ERNIE and Anima never say so, and the others' warnings go to the
server console. `prompt_encoding` loads each checkpoint's own tokenizers on
the CPU and counts over the exact string: Krea 2 and Qwen-Image's 34-token
system turn, Z-Image's and klein's chat template with the pipeline's
`enable_thinking` (on and off respectively), NetaYume's system instruction and
`<Prompt Start>` (the negative gets none), and a boundary merge counted the
way the tokenizer counts it. The budgets that come out, which builder A
matched against each pipeline's own truncating tokenizer call: FLUX.1 CLIP 75
(one pooled vector - T5 reads on, so its cut is info) and T5 511; Krea 2 507,
its suffix appended after truncation and never cut; Qwen-Image 512 after its
34 system tokens, the template tail lost past 507; Z-Image and Z-Anime 509
(504 before the assistant turn goes); klein 509 (500); NetaYume about 237
(255 for the negative); NewBie Gemma 3 511 and Jina 2046; Anima Qwen 512 and
T5 ids 511; ERNIE 2047; SDXL 75 per CLIP. Past the clean budget Z-Image,
klein and Qwen-Image encode an unterminated user turn (`encoder.template_tail`).
Every template string, length and thinking switch copied into the module is
held equal to the installed diffusers source - and to this repo's Anima and
NewBie pipelines - by reading those files in `tests/test_prompt_encoding.py`,
so a diffusers upgrade that moves one fails a test rather than the counts.
The table is first-match on the id like `_PROFILES`, exact ids first,
reusing the loaders' pinned constants; its tokens are narrower than the
profiles' (`flux.1`, so FLUX.2 klein never reaches FLUX.1's CLIP and T5).
Control tokens are found by tokenizing, from each tokenizer's own special
tokens and the markers its chat template uses, not from a list: Gemma 2's and
3's `<s>` and `</s>` turned out to be ordinary HTML-strikethrough tokens
beside `<b>` and `<table>`, which the research's table had wrong, and
`<think>` counts only where the template uses it (klein, Z-Image, Z-Anime).
Tokenizers load from a directory - converted tree, then the Hub cache, then
the Hub - not a Hub id, because transformers 5.16 loading a fully cached
pinned tokenizer by id with `local_files_only` failed; the download patterns
are tokenizer files only (`*.json`, `*.txt`, `*.model`, `*.jinja`), since
NewBie's tokenizer folder sits beside 8 GB of weights. A load is cached under
a lock and kept only after it succeeds (`translation.py`'s pattern), so a
gated repo or a first use offline is retried next time; `measure` never
raises and says what it could not check in `encoders_detail`. It did raise
once: `json.loads` on 3,000 nested `[` throws `RecursionError`, well inside
the 4,000-character limit - `_is_json`, `detect_style` and `is_reorder` now
catch it. No status route loads a tokenizer, because a first load can be a
download; Unload and shutdown drop them with the pipeline.

**The rule layer makes "never invented" mechanical.** Each guide's
`analysis` entry (previous section) holds the words a rule rests on as a
verbatim piece of the prose the ? badge shows, and
`test_every_cite_is_the_guides_own_words` fails an entry the prose does not
contain - so a rule can only be a statement the page already makes, and a
`guide` finding carries that cite and its URL to the page. What rests on an
author's example rather than the prose is `author` (Z-Anime's `[character]`
and NewBie's `$character_1$` slots). Pattern matching with no author sentence
behind it - quality boilerplate on a prose checkpoint, a repeated tag - is
`heuristic`, carries no cite, and the page labels it Heuristic. The authors'
own example prompts raise no warning against their own guide (a test), apart
from those template slots; some raise info - Kohaku's own example departs
from its documented Epsilon order, two of Illustrious's put `masterpiece`
mid-prompt, FLUX.1's and klein's are under 30 words - which is why order and
length are advice, not warnings. A tag's Danbooru category (character,
series, artist) comes from the chips' own lookup (`_lookup_tags`, so the two
cannot disagree), only on a Danbooru-vocabulary guide - or one whose rules
read artists, Anima's - and a tag or hybrid prompt; without it the order
check places only what the guide's own groups
know and never guesses a slot, and any failure there - a truncated reply
raised `IncompleteRead`, which `tags._request_json` now also treats as
unavailable - costs the order check, never the answer. `reorder_tags` moves
the items and keeps their spelling and the text between them, a pure
permutation the tests assert, which is what the comparison's "reordered" row
needs. A rule's `suggestion` is always the whole field rewritten, so the page
can apply it in one click; advice that cannot be applied that way ("no
people" - BFL's table suggests "empty", "deserted", "solitary") stays in the
message. The review of the first build found the warnings that would have
cost trust: unquoted sign text ("reading NO PARKING") and romanised titles
("Sousou no Frieren") read as negations, and "A mother says goodbye" read as
text to render; negations now skip text-to-render spans and a lowercase
"no" between capitalised words, and "says", "labeled" and "spells" need
something that carries text earlier in the sentence. The final review found
the quote rule's one-click fix quoting half a sign: the text to render ended
at the first "and", "to", "with" or "for", so "says Fish and Chips" became
`"Fish" and Chips` and "reads Welcome to Paradise" `"Welcome" to Paradise`,
and the re-check then found nothing, so the user believed it fixed. A joining
word now carries the lettering on when a capital follows it (a name or a
title), a run of capitals before a lowercase one is the whole of it ("says
OPEN and a cat"), and where the end is a guess - "says fish and chips for
sale", or more than six words - the finding asks for the exact words in
quotation marks and offers no Apply. It also found the reported style
worked out without the tag categories while the rules used them, so the
summary read "tags and prose" beside an order finding only a tag list gets;
`detect_style` now takes the categories and `PromptAnalysis.style` is the
rules' own. "images 1 and 2", "images 1-3" and "images 1, 2 & 3" name every
image in them for the references rule. And `guide.artist_prefix` (Anima's
"put @ before an artist name") never fired: it needs an artist's category,
and Anima's guide has no tag helper, so the route never asked. The route now
asks the tag index for any guide whose rules read artists
(`wants_artist_categories`), with no cutoff and no special tags of its own;
Anima's card says it was trained on Danbooru-style tags.

**The order is shown as checked only where it was.** Only a tag
checkpoint's order has slots a comma item can take (`_OrderSpec`); FLUX.1's
"subject first", klein's four steps and NewBie's blocks are documented and
shown, but nothing reads a sentence's order. The final review found the
panel reading as a pass on the default checkpoint - a prose prompt with its
subject last and one with it first got identical panels, "nothing to flag"
beside the chain. `PromptAnalysis.order_checked` says whether the rules held
the text to the order at all, and the page says "Not checked" under the
chain and "order not checked" in the summary. A pattern check for a prose
prompt whose first clause is only light, camera or style was considered and
left out: BFL's own steps start with the image type, Krea's own example
opens "Moody night photography", and a rule that flags the authors' own
examples fails `test_the_authors_examples_raise_no_warning` for good
reason. Where a guide states no order, the page no longer says "the authors
document no word order": Anima's card, Neta Lumina's prompt book and the
ERNIE and Qwen rewriter instructions do give one, and the critic cites them.
It says the app checks none, and shows those excerpts (`Excerpt.order`,
`PromptAnalysis.author_orders`) as the authors', with their links, as
information.

**Deliberately not checked**, each for a reason: schnell's "about 180 words",
which rests on a 256-token limit this app does not apply (below); FLUX.1's
"English is the most precise language", which BFL's current basics page
contradicts (other languages "with the same level of quality"); Z-Image
base's "strongly recommends a negative prompt", which its card calls
optional, and which is not in the catalogue; the order of a sentence, an
XML prompt's blocks or a JSON prompt's keys (above: shown as not checked);
Anima's card tag order and prefix, which are not in the guide prose - the
critic has them as excerpts and the page shows the order as the authors'; an
Illustrious prompt "missing" a composition or artist tag, which the authors'
own examples lack; quality tags placed past CLIP's 77 tokens, which the
encoder cut already reports; Illustrious's `sfw`, `nsfw` and `r-18`, which
its release post accepts (only `safe` is corrected, to `general`); a single
klein reference with no "image 1" (BFL's own "Turn this cat into a dog"
names none - two or more are checked); and `\(`/`\)` escapes, which the
app's own autocomplete writes (below).

**The critic runs in llama.cpp, not in-process.** Measured on this card
(RTX 5080) with Qwen3-4B: llama.cpp b11206's CUDA build decodes Q8_0 at 170
tokens/s (`llama-bench`), about 151 end to end under a JSON grammar with
reasoning on - 8.8-9.0 s for 1,320-1,348 tokens at temperature 0, 2.8 s with
reasoning off - at about 5.2 GB of VRAM over the desktop's at an 8k context,
and the server loads in 1.7-1.9 s. That was the research's greedy decoding;
with the shipped sampling `verified.md` records 8.1-16.0 s per critique
(median 10.5), 19.3 s for a tag prompt, and 5,581-5,611 MiB (about 5.5 GiB)
over the baseline. In-process transformers decodes the same
model in bf16 at 33 tokens/s, so a reasoning-length answer takes about 40 s;
the image pipeline's own torchao fp8 encoder decodes at 9.4 (reusing it is
out); on the CPU, bf16 manages about 6. Constrained decoding in-process would
also need a new dependency. llama-cpp-python's CUDA wheels are built for
sm_75 to sm_90 only (Blackwell would JIT the PTX, cost unmeasured) with an
untuned CPU path, and a CUDA abort in-process takes the FastAPI server with
it; Ollama is a separate install with its own model store and a GPU scheduler
that knows nothing of the image jobs. The official release archives run as a
child process, so a crash is the child's. The model is Qwen's own Q8_0 GGUF
of Qwen3-4B (Apache-2.0) at a pinned revision - the research measured the
same weights converted from klein's text encoder, which is unmodified
Qwen3-4B, but a conversion needs tooling the app does not install and klein's
files exist only where klein does. Qwen3.5-9B had the strongest published
instruction-following scores the research found among Apache-licensed models
of 9B or less, but its speed on this build was not measured, so the pin is
the model that was.

**Reasoning stays on.** Without it the 4B models tried were unreliable in
ways that do harm: one called the camera missing when the prompt said "shot
on film", one applied the reference-image sentence to sign text, one advised
putting the negative phrase "don't make it look like a painting" in quotation
marks - which asks the model to letter it into the image - and rewrites kept
"no blur, no extra fingers" against the guide's negation rule. Krea 2's
Qwen3-VL and NewBie's Gemma 3, tried as critics, missed unquoted text and
negations too. With reasoning on, Qwen3-4B caught subject order, unquoted
text and negative phrasing and wrote a clean rewrite - and on one prompt
still kept the negatives in its rewrite, which is why the rewrite goes back
through the rules. The request uses Qwen3's recommended thinking sampling
(temperature 0.6, top-p 0.95, top-k 20; its card warns greedy decoding loops
in thinking mode), `--reasoning-budget 2048` so a long thought cannot run the
JSON off the end, and a `max_tokens` sized per request: the budget plus the
prompt's estimated tokens times the copies the answer repeats (parts, the
rewrite, the reordered version) plus 800, between 4,096 and 8,192. A fixed
4,096 failed review: a 4,000-character English prompt is 966 real tokens,
and three copies after 2,048 tokens of thought do not fit. A prompt whose
answer cannot fit the 8k window even with no thought (4,000 characters of
Chinese) is refused with a `CriticError` before a server starts. Measured by
builder C on the real b11206 server with the shipped messages: 12.7-18.6 s
per critique, about 2 s of it server start (a klein prose prompt, 985 prompt
tokens and 1,542 generated at 154 tokens/s; an Animagine tag prompt whose
thought ran to the 2,048 budget), VRAM back to baseline and no server left
after each - hence the page's "about 15 seconds". Card measurements belong
in `verified.md` under "Prompt analyser".

**One pinned server per critique, keyed, and stopped before the gate opens.**
`critic.BUILDS` pins release b11206's Windows CUDA 13.4 archives and the
Ubuntu ones for WSL (the default architecture list includes sm_120) by size
and the sha256 the GitHub release API publishes, and `MODEL` pins
`Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q8_0.gguf` by revision, size and sha256. The
binaries are a download like the weights and handled like them: each archive
to `.part`, verified, extracted into a sibling with zip members that escape
the target refused and tarballs through `filter="data"`, the marker written
last and the tree renamed into place under `HF_HOME/local-diffusion/llama.cpp`;
a tree without the marker is never used. The GGUF is hashed once, and a stamp
of its path, size and mtime spares every later critique hashing 4 GB again.
A first use is about 5 GB (576 MB of binaries and runtime on Windows, 4.3 GB
of model), so `prepare()` runs before the gate, on a daemon thread as the
model load does (a Ctrl-C must not wait out a download), and the route checks
the queue and the load again afterwards, since minutes may have passed. Only
`critique()` runs inside `jobs.run_when_idle`: `llama-server` on 127.0.0.1 and
a free port, `--offline`, `--no-ui`, `-c 8192`, `-np 1`, `--jinja`, `--fit on`
(b11206 adapts only unset arguments, so layer placement follows the VRAM
actually free while the context stays pinned by `-c`; hence the 180 s health
and 600 s request timeouts, for layers that land on the CPU), then one chat
completion, then terminate, wait, kill in `finally`, so the card is free
before an image job is let in. A resident server would hold ~5 GB against the
pipeline for nothing when a start costs two seconds; `--sleep-idle-seconds
120` is only the backstop for a server this process could not stop. Its CORS
allows every origin, so a web page in this browser could reach a keyless
one: each start gets a fresh random key, and before the prompt is sent a
keyless `GET /v1/models` must answer 401 (b11206 leaves only its health
endpoints public), or the server is refused. The key goes to the server as
`LLAMA_API_KEY` in its environment, which b11206 reads in place of
`--api-key` (its `--help` says so): on the command line, as first built,
any local user could read it (`/proc/<pid>/cmdline` under WSL). The child's
environment otherwise drops `LLAMA_*` - `LLAMA_ARG_TOOLS` would switch on its
shell and file tools - and loopback requests bypass any proxy.
`LOCAL_DIFFUSION_CRITIC=false` switches it off; `LOCAL_DIFFUSION_CRITIC_SERVER`
and `LOCAL_DIFFUSION_CRITIC_MODEL` name files already on disk in place of
either download (made absolute, because the server starts in its own
directory), and naming one is choosing it: neither is held to the pins, but
a server of one's own must take b11206's flags and its key from the
environment. The gap between `_free_port()` and the server's bind is guarded
by the 401 probe, not by the bind failing (b11206 sets `SO_REUSEADDR`), as
`_wait_until_healthy` says. Open, from review: a server orphaned by a hard
kill of the app frees its VRAM after the idle timeout but stays alive and
listening. The WSL tarballs' layout - one top-level directory each,
flattened on install so the CUDA runtime sits beside the binary - was read
from b11206's `release.yml` when this was built and verified on the card on
2026-09-27 (`ldd` resolves every library inside the tree; `verified.md`).

**The critic cites only what it is given, and what it says is checked
before it is shown.** `prompt_sources` gives it, per guide constant, the
guide's own `prompt` prose split into sentences (G1..Gn - the text the ?
badge shows) and short verbatim excerpts the checkpoint's authors put in
writing (A1..An); the answer's JSON schema makes `cite_id` an enum of
exactly those ids, so a citation cannot be invented, only misapplied, and an
id that names nothing drops its issue (`dropped`). A G sentence links to the
guide's own `analysis["url"]`, the page the rule checks cite, so the two
layers cannot point one sentence at two pages - unless the sentence's author
page is another one, which `Grounding.sentence_urls` names: Krea [dev]'s
settings sentence links to Krea's README and its camera sentence to BFL's
building page. A sentence the prose carries with no author behind it is
listed `unsourced` and never citable, and a test fails if one stops matching
the prose; Krea [dev]'s "photo language" and "give it room" sentences were
the two, until the final review found the critic, grounded in BFL's pages,
telling users to remove the camera details the badge beside it asked for -
they were taken out of the prose and replaced by what BFL's building page
does say (camera settings "when framing matters"). Excerpts, checked against copies
fetched 2026-09-27 - verbatim but for markdown emphasis, elisions marked
"…" and a few labelled reformattings (three rows of BFL's replacement
table, Neta's vague/specific pair, orders flattened from lists): BFL's family
pages for FLUX.1, schnell and Krea [dev]
(<https://docs.bfl.ai/guides/prompting_unified_basics>, `_building`,
`_technical`); for klein,
<https://docs.bfl.ai/guides/prompting_guide_flux2> (written for [pro] and
[max], klein by family) and the T2I upsampling instruction in
<https://github.com/black-forest-labs/flux2/blob/main/src/flux2/system_messages.py>,
which BFL's own CLI also runs for klein; Krea 2's `docs/prompting.md` and
`docs/expansion.txt`; Qwen's
<https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py>
(and the portrait flow from `prompt_utils_2512.py`, labelled same-family);
the Z-Image Turbo Space's rewriter
(<https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py>)
and Baidu's <https://github.com/baidu/ERNIE-Image/blob/main/src/pe_prompt.txt>,
both Chinese, so `text` is this app's translation, labelled, with the
original kept; the Neta Lumina prompt book
(<https://neta.art/blog/neta_lumina_prompt_book/>), labelled inherited from
NetaYume's parent; NewBie's card and the XML field rules of its captioning
instruction (`XML Prompt.json`), nothing of its adult-content captioning;
the Anima, Animagine, Z-Anime, Kohaku Zeta and Epsilon-rev2 cards; the
Illustrious report (<https://arxiv.org/html/2409.19946v1>); and WAI's v17
description through its archive
(<https://civarchive.com/models/827184/wai-nsfw-illustrious-sdxl?modelVersionId=2883731>).
A per-guide `caveat` says how far the sources reach: Krea published no
prompting rules for Krea [dev], so both lists are BFL's; on every FLUX.1
checkpoint, that BFL's steps start with the image type and subject - so
"Photo of" or "Moody night photography" still puts the subject first - and
that camera settings are part of BFL's template, not a fault (the excerpts
now carry "Start with the image type and subject" and the template with
"not a strict formula"; a critic held to a bare "subject first" called
Krea's own example out of order and told it to put its style first in the
same answer); NetaYume's excerpts are its parent's and the app adds the
system prefix itself; Z-Anime's "details early" is its 4-step variant's.
The system message also says the rewrite adds no object, scenery, mood or
text and never pads toward a word range - a range is the guide's band for
most images, not a minimum (almost every rewrite on the card padded a short
prompt, against the sources' own "Start short") - that quotation marks go
only around lettering, never the prompt, and that the authors' example
prompts lend their form, not their subjects or text. It stays under about
1.5k tokens (the phase-1 review measured 500-880 with the real Qwen3
tokenizer before findings; the budget test holds every checkpoint's under
6,000 characters). Then every answer is checked (`validate`): a quote is
looked for in the prompt (case and whitespace aside) and nulled when absent,
the span located on the server so the page selects the prompt's own
characters; a part is kept only if each of its comma pieces is in the
prompt, and "none" or "n/a" is no part, and `text_to_render` only if it is
lettering the prompt asks for (`prompt_rules.rendered_spans`: its quoted
stretches and unquoted sign text) - on the card it held the whole prompt in
most runs; an issue that repeats a rule finding the critic was told of (the
word count wherever the guide has a range, a reported negation, quotation
marks, tag order, weighting, quality boilerplate - by its wording, or by
quoting the very words a finding points at) is left out, since the model
did so in nearly every answer however it was told not to, and miscounted
words beside the rules' correct count; two `order` issues that
ask for different parts first are both left out; `reordered` survives only
where the app has a documented order and only as a pure reorder - the same
whole tags, clauses, sentences or XML blocks (for JSON, the same value
written differently), every quoted span unchanged (`is_reorder`), where
"quoted" is `prompt_rules.QUOTED`, the one pattern the rules, the encoder
layer and the critic share (the critic's own narrower copy let a «…» or
„…“ quotation split across two pieces); a rewrite identical to the prompt is
dropped, one wrapped whole in quotation marks is unwrapped, one that quotes
words the prompt does not have is dropped - lettering nobody asked for,
such as the sign example's "Black Forest Labs" - and one that quotes the
prompt's description gets a `heuristic.rewrite_quote` warning where it
sits: the rules skip quoted text, so none of these reached the layer-1
re-check. A rewrite with six or more content words the prompt has in no
form gets `heuristic.rewrite_adds` (info), naming them: the instruction not
to pad did not stop the card's critic from adding lenses, light and scenery
to short prompts, and a comparison against such a rewrite tests new content,
not only wording. What validation takes out besides uncited issues is said on the
page (`PromptCritique.set_aside`). Word multisets were the first reorder
check and the review broke it: "a woman in a red dress and a blue hat" came
back with the colours swapped and the sign text's quotes gone, and passed.
On the real server the model's `reordered` added "great score, absurdres"
and once copied a character tag from the example prompts; on FLUX prose it
survived in 1 of 10 runs, and in 0 of 20 after the final fixes, being the
rewrite again 17 times (`verified.md`).
So on tag checkpoints the rules' `reorder_tags` is the dependable reordered
version, and the page prefers it; on prose the user writes one (Compare).
The answer is read from the content after the reasoning, but not only after
the last `</think>`: a prompt pasted from another model can hold one, and
the answer quotes it back. The rewrite's layer-1 findings are the rules and
encoders run over it by the route (`_lint`, without the tag lookup), added
to `validate`'s own - after `run_when_idle` returns, not inside it: the lint
can load a tokenizer the analysis failed to, from the Hub, with no timeout,
and inside the gate that held image jobs back;
`test_the_rewrite_is_linted_once_the_gate_is_open` holds it. What survives
all this is still a 4B model's reading - it misapplies sources and pads
prompts - which is why the page shows every version to be read before it is
used, and the comparison exists.

**The comparison is a grid, and its statistics are the caveat.** The
variants are one `POST /api/grid` with every other axis empty, the panel's
seed (or a drawn one, written back as grids do), `num_images` 1 and 4 seeds
by default, 8 offered; the preview gives the count and the estimate before
anything runs, and the page opens the run as the sheet when its stream says
done. For a 1 MP 28-step Krea [dev] image in fp8, `verified.md` records
15.4 s on 2026-09-08 - three versions on four seeds in about three minutes -
and 41.6-42.6 s on 2026-09-27, cause uninvestigated - about 8.5 minutes; the
preview's estimate, from the measured per-step time, is the one to go by.
The seeds are shared, so the rows are pairs, and the test that fits is a
paired sign test on the seeds where the two rows differ: four of four the
same way is p = 0.125, so four seeds cannot tell even a change on every seed
from chance; six of six is 0.031 and eight of eight 0.008 (seven of eight,
0.070, is not enough). An unpaired two-sided Fisher exact test on the hit
counts (0/4 against 4/4 is p = 0.029; at eight, 2/8 against 7/8 0.041) assumes
independent images, which shared seeds are not. Hence the line under the
Seeds select: 4 seeds is a first look - even a change on every seed could be
chance - and 8 can tell a consistent change from chance. The first wording,
"4 seeds only shows all-or-nothing differences", took the unpaired figure.
The fourth version, the user's own (`yours`), is what makes a comparison of
order possible on a prose prompt: nothing else there is a pure reorder.
Scoring the images
automatically (a VLM asked whether each element appears) was left for later:
nothing about its accuracy on these checkpoints has been measured, and
published judges are weakest exactly on text rendering and style. Stars
(`stats --by prompt`) and the run's `distinct` count are the signals that
exist.

**The page.** Check sits in the Prompt caption after Tags and before Expand
and sends what Generate would (`collect()`); its panel moves with the chips
(`placeTagValidation`), above the results on a two-column page so its
length cannot push Generate down the sticky settings column. The panel
describes one text - the prompt and the negative prompt as `collect()` sent
them, a greyed negative counting as none - because every offset in the
answer counts from that text and every Use and Apply is a whole replacement
worked out from it. Any change to either box marks it stale: typing, each
programmatic write through the one `promptReplaced()` hook (an example,
Reuse, Reset, a chip, an accepted suggestion), `syncCfg()` greying or waking
the negative. Stale results stay visible, dimmed, with "Check again"; every
Use, Apply and Run comparison goes dead, and a finding's words are selected
only while they are still where the check found them. The first build
compared the prompt alone, and "Apply to the negative prompt" then
overwrote a negative typed after the check. The critic's answer is always
about the checked text: asking checks again first when either box moved on,
a check of other text drops the answer, and an answer that lands after an
edit is shown dimmed and applies nothing. A checkpoint change clears the
panel. Apply is offered on rule findings only - a rule's `suggestion` is the
field rewritten - while an encoder's suggestion (for `((masterpiece:1.2))`,
`Write "masterpiece" plainly; ...`) is advice and shows as text. Use and
Apply write through `replaceBoxText`, which selects the box and inserts with
`execCommand`, so
Ctrl+Z brings back what was typed; assigning `.value` emptied the undo
history, and the re-check that follows would have made the replacement the
comparison's original, leaving the typed prompt nowhere on the page. Server
offsets count code points, as Python indexes a str (`PromptFinding` says
so); the page's strings count UTF-16 units, so `units()` converts before
selecting - one emoji ahead of a finding selected the wrong words, and the
still-in-place guard, comparing two JS slices, could not notice. An encoder
row is drawn as a caution only when the server says `pooled_only` (FLUX.1's
CLIP-L); keyed on `role`, it drew NetaYume's only encoder's cut as mild. The
critic holds what Expand holds - Generate and Run grid aria-disabled,
Expand, Unload - and `startJob` refuses while it runs; its button is off
while a job is queued or running, with a title saying why. The comparison
sends `num_images: 1` (the panel's Batch 4 had made "4 seeds" sixteen),
never writes the prompt box or the grid's variables, which are the user's
and persisted, and leaves out a version identical to another. Its Your
version box is the user's too: kept across checks (a check of other text
still compares against it), cleared with the rest of the panel, never
persisted. The Order section says "Not checked" under a documented order
the rules did not read (`order_checked`), and the summary "nothing flagged
· order not checked" rather than "nothing to flag". A negative prompt
finding's Apply is dead while the negative box is greyed, with a line saying
what CFG would wake it: `collect()` sends nothing from a greyed box, so the
write changed nothing the re-check could see and the same finding came back
with the same button. The critic's `set_aside` lines are shown under its
answer, and the `dropped` line says what dropping means - a citation that
named nothing it was given, or an empty issue; an unfound quote only loses
its quote.

**Open questions the build turned up**, for decisions rather than code:

- The tag autocomplete writes Danbooru names with A1111 escapes -
  `tags.display_tag` turns `(` into `\(` - to match the authors' own examples
  (Animagine's has them). Diffusers never unescapes, so CLIP reads
  `firefly \ ( honkai : star rail \ )`, ten tokens where the name is eight.
  The analyser does not flag what the app itself writes; whether the chips
  should write bare parentheses is undecided.
- schnell: the guide prose and the README say its T5 stops at 256 tokens,
  "about 180 words". That is BFL's reference
  (`load_t5(..., max_length=256 if name == "flux-schnell" else 512)` in
  <https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py>);
  this app passes no `max_sequence_length`, so `FluxPipeline` encodes 512 on
  schnell too, and the encoder report says 511. Either pass 256 on schnell or
  reword the guide; the word rule stays out until then.
- Qwen-Image's guide says to sample "with True CFG 4 and a negative prompt -
  a single space if there is nothing to remove", but `generation.js`'s
  `collect()` sends `els.negative.value.trim() || null`, so a single space is
  sent as no negative prompt and no True CFG pass runs. The True CFG hint
  already says "a word will do".
- Resolved 2026-09-27: Krea [dev]'s "It answers best to photo language: a
  mood, a lens, a light, a film stock, a colour grade" and "Krea calls it
  opinionated - give it room rather than over-specifying" had no literal
  source in Krea's README, Krea's release post or BFL's, and the critic, held
  to BFL's pages, told users the opposite on the same page. They are out of
  the prose; what BFL's building page says about camera settings replaced
  them.
- Anima's guide prose leaves out the card's tag order - `[quality/meta/year/
  safety tags] [1girl/1boy/1other etc] [character] [series] [artist]
  [general tags]` - and its recommended prefix `masterpiece, best quality,
  score_7, safe, `, so Anima has no documented order in the app: the critic
  sees both as excerpts (A1, A3) and the page shows the order as the
  authors', unchecked. Added to the prose, they would be checkable.
  (`guide.artist_prefix`, which never fired as first shipped, now does: the
  route asks the tag index for Anima's artist categories.)
- NewBie's guide prose names a block's fields "name, gender, appearance, ...",
  but the authors' XML example and captioning instruction spell the first one
  `<n>`. The XML rule accepts both; whether the model reads `<name>` as it
  reads `<n>` has not been tried.

## The page script is eleven files, not one module

`web/app.js` reached 3,200 lines - every section of the page in one
file - and was split on 2026-09-10 along its own section comments into
ten scripts `index.html` loads in order: `state` (the element table, every
`let` the page shares, the fetch helper), `sizes`, `status`, `generation`,
`panel`, `gallery`, `sheet`, `lightbox`, `tags` and last `app`, the wiring
and the boot; the prompt analyser added an eleventh, `analysis` (the check,
the critic and the comparison), between `tags` and `app` on 2026-09-27. They
stay *classic* scripts rather than ES modules on purpose.
Classic scripts share one global scope: a top-level `let`, `const` or
function in any of them is visible to all the others, so nothing needed an
export or an import, and - the reason that matters - a smoke step's `eval`
(`tools/browser.mjs`) still reaches `els`, `galleryRecords`, `startJob`,
`analysisResult` and the rest by name, which a module's scope would hide. The
contract that makes the order safe is that ten of the eleven only *declare*:
a function in `sizes.js` may call one from `analysis.js` because every call
happens after all eleven have loaded, from a handler or from `app.js`'s
boot. A file that ran something at load would have to sit after everything
it touches; keep the wiring in `app.js` (the lightbox's own pointer handlers
are the exception, wired beside the code they belong to and touching nothing
outside it).

The same pass moved `tests/test_api.py` (1,300 lines) into `tests/fakes.py`,
the fixtures in `conftest.py`, `tests/api_helpers.py` and one `test_api*.py`
per area, and pulled `backend/search.py`, `backend/guides.py`,
`backend/backends/loading.py` and `backend/backends/schedule.py` out of the
three largest backend modules. `loading` takes `settings` as a parameter
rather than importing it: the tests replace the backend's `settings`, and a
second module-level copy would have gone on reading the real one.

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
reproduces every hash and true CFG. Not verified: `LOCAL_DIFFUSION_DEVICE=cuda:1` (one
card here).
