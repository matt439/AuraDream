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

The same file exports `LOCAL_DIFFUSION_QUANT=nunchaku` and a lite-infer model
id, and six tests written against the default checkpoint and fp8 failed under
WSL because of it: the suite read whatever the shell had. `tests/conftest.py`
now drops every `LOCAL_DIFFUSION_*` except `OUTPUTS_DIR`, `DB_PATH` and
`LORAS_DIR` before its first `backend` import - a fixture is too late, since
`settings` reads the environment once, on import - and asserts that nothing
imported `backend.config` ahead of it. The paths stay so that a test which
fails to redirect them writes to this environment's gallery rather than the
Windows one under `/mnt/c`.

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
Turbo's constant on it degrades output; the profile leaves mu null there. The
panel's Mu box says "auto" while blank, where Mu applies at all, rather than
sitting empty beside True CFG's "off"; the grid's Mu axis suggests
`none,<the profile's mu>` and nothing where the profile has none, instead of
Turbo's 1.15 on every checkpoint (U87, U106).

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
On the default checkpoint the negative box is therefore greyed on every fresh
page, with the box that wakes it a field's height below. While it is greyed and
the checkpoint takes a negative at all, a "Set True CFG" (or "Set CFG") button
after the note that names that box focuses and selects it and writes nothing
(`focusNegativeKnob`): a value is the checkpoint's, about 4 on Qwen-Image,
none on schnell. The field is a `<div>` with a `<label for="negative">`, as
the Prompt field is: inside a `<label>` the button became the label's control
and the textarea lost its name (U100). The button first sat in the box's
corner, to cost no height, with `pointer-events: none` on the greyed box so a
press on it reached the field and did the same. Both went: the button lay over
the box's own text, a carried-over negative being what someone looking at a
greyed box is reading, and the greyed box no longer took the wheel, so such a
negative could be read only to its second line. In the caption it costs
nothing where the note already wraps (the 330 px panel) and a line on a wide
one.

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

**The chips are numbered, and their order is editable.** klein's guide, its
prompts and the check name references "image 1", "image 2", so the order is
what the prompt means. Each chip carries its number at bottom left, its image
is named "Reference image N (gallery #id)", and it moves the way the tag chips
do - dragged, or Alt+Left/Right on its × - with Delete on the × dropping it and
focus going to the next chip's ×, else the previous one's, else the emptied
row (`tabindex="-1"`, a group named "Reference images"), rather than to
`<body>`. Not the field's badge: a focused badge opens its tip, which then lay
over LoRA and Seed for someone who had only clicked the last × away. Every move builds a new array and goes through
`setReferences`, the one writer, so the panel is remembered, the preview asked
again and the prompt check marked stale ("references changed since"). A chip
whose thumbnail does not load - an image deleted since, or an id this browser
remembered - is marked deleted rather than showing a broken image. The note
under the caption counts "1 of 4" and asks for numbers in the prompt only from
two references on (one needs none, as BFL's own example shows); it no longer
says each lengthens the sequence, a cost measured above as absent. Where the
field is greyed its empty chip row is hidden, the caption, badge and note
staying; the badge says what references do where they are taken, rather than
the unsupported checkpoint's sentence (U36, U62, U77, U95).

**A deleted reference leaves the chips.** An id the gallery no longer holds
used to stay a chip - with the browser's cached thumbnail in the session that
deleted it, a broken image after a reload - until Generate was refused with
"no gallery image with id 4", an id the chips never showed. `forgetImages`,
the one place the page drops deleted ids, drops them from `referenceIds`
too (through `setReferences`, the one writer) and its caller says so in the
gallery's line; the per-image confirm says "It is reference image N in the
panel" before the delete. Reuse restores a record's references
synchronously and then asks `GET /api/images/{id}` for each, dropping only
those answered "unknown image" (see "Per-control defaults and Reuse"). The
server still refuses a missing reference at submit (`storage.reference_paths`).
And a starred image keeps its references through Delete unstarred: they are
part of what made it (see "Delete unstarred").

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
appears is explained rather than missing - after the tooltip's own sentence
(the directory it reads, what Weight does), not in its place, and without the
file's name twice, since a `detail` starts with it (U118); `/api/generate`, `/api/grid` and
`/api/rerender` answer 422 with the same reason and the CLI exits 2 with it.
Measured on the frosting-lane file against the cached checkpoints: fits
FLUX.1 Krea; 494 modules Z-Image has not got; 454 klein has not got; the
right names at the wrong width on Krea 2; no `transformer` on SDXL.

## The refine pass

**A refine pass keeps the picture, which a direct render at a larger size
cannot.** The size guides had been saying so in the authors' words and the
app had no way to do it: FLUX.1's "render at 1-2 MP and upscale", and WAI's
1024x1344 followed by a 1.5x hires pass at denoising strength 0.35-0.5. What
the app had was Re-render, whose own notes say what it delivers - the latents
are drawn at the requested shape, so the same seed at another size is a new
sample - and the measurements say where a direct large render goes wrong
(Illustrious at 2048 rendering the shelves and omitting the doctor its 1536
kept). A pass is `GenerateRequest.refine`, a gallery id, and `strength`: the
source is resized to the request's width and height, encoded by the VAE,
noised to the point on the schedule the strength names and denoised from
there under the request's own prompt, negative prompt, steps, CFG and seed.

**It runs the sibling pipeline over the components already loaded.**
diffusers ships an image-to-image class beside most text-to-image ones, by
name (`FluxImg2ImgPipeline` beside `FluxPipeline`) and over the same
components, so no extra weights are downloaded or allocated. With model
offload, the source's VAE encode keeps the VAE resident beside the denoiser
through the pass until the final cleanup; the measured WAI and FLUX.1 peaks
are about 0.16 GiB above a direct render at the same size. Sharing weights
does not imply the same peak residency.
`refine.sibling(type(pipe))` is the lookup, and the whole capability: the
name with `Img2Img` in it, only for the class diffusers itself exports under
the pipeline's name (a local class that merely shares one has components
nobody checked), and only where the sibling's `__call__` takes
`refine.REQUIRED` - `image`, `strength`, and the `generator` and
`callback_on_step_end` `load()` insists on - and every argument in
`refine.CARRIED` that the text-to-image call takes. `CARRIED` is the list
`generate()` passes besides the size, held to that function's source by
`tests/test_refine.py`: a sibling that lacked `negative_prompt` or `sigmas`
would have it dropped by the backend's drop loop, at DEBUG, and render a
refine the row then describes wrongly. In diffusers 0.40 that gives FLUX.1
(Krea dev, dev, schnell), SDXL (Animagine, Kohaku, Illustrious, WAI),
Z-Image (Turbo, Z-Anime) and Qwen-Image a pass, and Krea 2, FLUX.2 [klein],
ERNIE-Image and Lumina 2 (NetaYume) none; Anima and NewBie run this repo's
own pipeline classes, which have no sibling. A table of families was not
written: the loader is model-agnostic, and so is this. Each profile carries
`refine_supported` as its guess before a load, `describe()` answers from the
class after it, `/api/status` merges the two like `reference_supported`, and
`tests/test_refine.py` holds every guess to the installed diffusers.
If the live class supports the pass but the profile has no strength, status
reports `refine.DEFAULT_STRENGTH`, matching request defaults and the tooltip.

**`refine.build`, not `from_pipe`.** `DiffusionPipeline.from_pipe` is the
library's own way to make one pipeline from another "without reallocating
additional memory", and it ends in `new_pipeline.to(dtype=dtype)` - float32
unless told otherwise - over modules the first pipeline is still using,
torchao-quantized ones among them. The sibling is constructed directly
instead: each constructor parameter from `pipe.components`, or from
`pipe.config` for the plain values (SDXL's `force_zeros_for_empty_prompt`),
and an error naming a required one the pipeline has not got. A constructor
only registers what it is handed. The offload hooks are the modules' own, so
they come along; the scheduler is the same object, which is what lets
`schedule.forced_mu` - patched onto the loaded pipeline's scheduler - reach
a pass, and a static scheduler is handed the pre-shifted `sigmas` exactly as
a text-to-image call is. The adapter goes into the denoiser through the
loaded pipeline (`_apply_lora`) and the sibling shares the denoiser. Built
on the first refine, kept, and dropped on unload - it is a second set of
references to the same modules, and kept past that an unload would free
nothing.

**One call per image, because the VAE's encoding is sampled.** Every other
batch here is one pipeline call with a list of generators, one per image, so
that image *i* is what a solo run of its recorded seed makes. An
image-to-image call encodes the source before it draws the noise, and that
encoding is a *sample* from the VAE's posterior: `FluxImg2ImgPipeline` loops
over the images it was given - one - and draws that sample from the first
generator alone, so image 2 of a batch got image 1's encoding and its own
noise, and its recorded seed reproduced something else. SDXL's sibling
samples per generator and would have been fine. `DiffusersBackend._refine`
makes one call per image with that image's generator (a scalar: the sample,
then the noise, from one stream, on every pipeline), which costs a text
encode and a VAE encode per image instead of per batch and makes the
invariant true by construction rather than per pipeline. Measured on the
card (`verified.md`): image 2 of a batch of two is byte-identical to a solo
run of its seed, on SDXL, FLUX.1 and Z-Image alike. Progress counts through the images
(`batch["index"] * steps + step`), so a batch's bar does not start over.

**The sibling has no hook list, so the loaded pipeline frees the card.**
`enable_model_cpu_offload` was called on the loaded pipeline;
`maybe_free_model_hooks`, which every call ends with, returns at once on a
pipeline that never had it called (`_all_hooks` is absent). After a refine
the last component used - the VAE - would stay on the card until the next
call's first forward. `_refine` calls the *loaded* pipeline's
`maybe_free_model_hooks()` in a `finally`, which resets the chain the way a
text-to-image call's own ending does, after a cancel or a failure too.
Measured: the device reads the same 1.4 GiB after a pass as after a
text-to-image call, and the source's own request renders the same hash after
the passes, a batch and a cancelled pass as before any of them.

**`steps` is the schedule's length; the strength decides how many run.**
The pipeline builds the full `num_inference_steps` schedule and starts part
of the way down it, and how the count rounds is the pipeline's own: SDXL
runs `int(steps * strength)`, the flow pipelines `steps - int(steps -
steps * strength)`, so 28 steps at 0.4 is 11 on SDXL and 12 on FLUX.1. The
row records `steps` as asked - it is what reproduces the pass. Progress and
estimates account for the sliced schedule: the step callback reads
`pipe.num_timesteps` for progress events, while the initial job total uses
`refine.steps_run` with the checkpoint's integer rounding. `refine.start_index`
also makes the duplicate advisory count strengths that begin at the same
step; reproduction keys retain the requested strength. `storage.step_cost_ms`
uses direct rows only. `storage.refine_cost_ms` keeps full per-image refine
durations, including encoding and decoding, grouped by actual denoised step
count at the same size and checkpoint. A refine estimate needs matching
history for every count; it does not extrapolate from direct renders or a
different count. SDXL's call raises below one step, so
`GenerateRequest.check_refine_steps()` refuses `int(steps * strength) < 1`
after the capability check, and for a grid checks each expanded cell by name
instead of unused base values. The preview reads its sigma at
`scheduler.begin_index + step + 1`:
a pass starts at the index its pipeline set on the scheduler, and read from
the top the gain was that of a far noisier step than the one just taken.

**The source is covered and cropped, never stretched.** `refine.fitted`
scales the source until it covers the requested size and crops the centre,
Lanczos. The scale buttons multiply the source's size (`refine.scaled`, each
side rounded to 16, half up - written out because `round()` rounds half to
even and the page's `Math.round` does not, and the scale buttons and the
CLI's `--scale` must land on the same size), where the crop is a few pixels
at most; a size of another shape is a reframing, and the size line says
"cropped to this shape" and how much it keeps ("keeping the middle 56% ×
100%"), from `refine.fitted`'s own arithmetic - cover, each side rounded
half to even as Python's `round()` does, the centre cut out - which the
page repeats (`refineKept`) so the number and the frame drawn on the large
source are the pass's. Keep shape, on by default, moves the other side to
the source's shape when W or H is typed: typing one side is how "2000 wide"
is asked for, and alone it reframed the picture. It acts on typed input
only, so a script that sets both sides means the size it sets; off, a
reframing is one keystroke. The resize is in the backend rather than left to
the pipeline because SDXL's image-to-image call takes its size from the
image and has no `width` or `height` at all.

**It opens at a size the checkpoint holds.** The dialog opens at the scale
last run, 1.5x the first time, or the largest smaller factor that the pass
takes and that is not past the tight line (below), else at the source's
own size; a remembered scale it passed over says so on its button ("(last
run at 2×)"), one click away. It used to open at 1.5x or the remembered
scale whatever that came to: a 1.5 MP FLUX.1 source opened at 3.5 MP, and
after one 2x pass every later source opened at 2x - past what the card
holds (it fills at 2.4 MP), and on FLUX.1 at a size where the profile's
own 0.1 starts near sigma 0.65-0.92 and repaints the subject. Open and
Enter, the fastest path, ran exactly that. The line is the profile's
measured one where `verified.md` has it, so WAI, which ran 8.6 MP, still
opens at 1.5x; an unmeasured SDXL checkpoint opens capped at the generic
line, one click from 1.5x or 2x. The guard is unchanged and is still the
only refusal. A run at the size the dialog opened at in place of the scale
it was asked for leaves the remembered scale alone (`refineCapped`): that
size was the dialog's choice, and remembered as the user's, one Enter on a
1.5 MP FLUX.1 source opened every later 1 MP source at 1x - the 2x trap
above, turned round.

**Everything else is the reference image's design.** The source is an id,
never a path, resolved through `storage.reference_paths` under its own noun;
refused at submit (`_refuse_unusable_inputs`, and the CLI before its load)
where the checkpoint has no pass or the image is gone, because a night of
refinements that rendered as plain text-to-image at the larger size would be
a night of new pictures. Jobs and the CLI check again after loading, before
any cell renders, and fail if the live capability would drop the requested
source. `applied()` still normalises an unsupported pass away,
with the strength back at its default - a strength with no source is not a
setting, as a LoRA weight with no adapter is not - and with any references,
since the sibling's `image` is the source. `refine` and `strength` are
columns (`_migrate`), keys of the PNG's chunk, `Denoising strength` and
`Refine` in the A1111 block, `refine:` and `strength:` in the search
grammar, `strength` and `refine` in `stats --by`, and the fourth optional
element of `CellKey` (`grid.refine_key`, `id@strength`). Re-render,
`--from-image` and Reuse carry both; a database merge remaps the source id
with the reference ids and drops one whose image did not come across.
PNG-only recovery uses `reference_sources` and `refine_source` descriptors
(filename, decoded-pixel hash and dimensions), recorded by `save_image` and
database merges. It imports each proven source first, then records its new id.
Legacy ids without provenance, mismatches and missing sources are cleared
and counted as unresolved; an unproven id is never used as a destination id.
Changed links are written back into in-place PNG metadata too, without
recompressing pixels, so `--from-image` sees the same links as the rebuilt row.
`GridRequest.strength_axis` (`--grid-strength`) is an axis only beside a
source; on the page it is the Refine dialog's Strength box holding more
than one value. Reuse on a refined record asks `GET /api/images/{id}` for
its source, which the record holds by id alone, and opens the dialog on it.
A source deleted while the dialog is open is the server's to refuse, by
name: in the dialog, where Refine waits for the preview of changed boxes
(the preview runs the submit's checks), and at submit otherwise.

**A strength is an index into the checkpoint's schedule, so where it starts
is the profile's.** The first version had one default, 0.4, inside the
0.35-0.5 WAI's author documents - and on the card that number was right for
WAI and wrong for FLUX.1 Krea dev, the default checkpoint: at 1.5x of a 1 MP
image the pass kept the composition and repainted the man's face, at 0.4 and
still at 0.2. The pipelines turn a strength into a *step index* and the
schedule's shift then decides the noise level at that index. SDXL's epsilon
schedule at 0.4 starts where the signal-to-noise ratio is that of a flow
sigma of about 0.50 (leading spacing, 28 steps). FLUX.1's shift is dynamic and grows with the size - mu
is 1.15 at 1 MP and 2.0 at the 2.4 MP a 1.5x pass renders - so its 28-step
schedule there spends nearly all its steps at high noise: strength 0.4
starts at sigma 0.85, 0.2 at 0.67, 0.15 at 0.62, 0.1 at 0.47 and 0.05 at
0.37. The 0.85 and 0.67 values are computed from the schedule; 0.62, 0.47 and
0.37 were read off `scheduler.sigmas` at `begin_index` in the logged sweep
and match the formula. The face held at 0.05 and 0.1 and began to change at 0.15.
So `refine_strength` is a profile key, where the Refine dialog's Strength
box starts on the checkpoint, filled in by the schema for a
request that names none (`schemas.default_strength`), and what `applied()`
puts a strength back to when there is no source. Each value starts the pass
at about half noise for a 1.5x pass of a 1 MP image:

| Profile | Start | Basis |
| --- | --- | --- |
| Animagine, Kohaku, Illustrious, WAI (SDXL) | 0.4 | WAI's author's 0.35-0.5; measured on WAI |
| FLUX.1 Krea dev, FLUX.1 dev | 0.1 | measured on Krea dev: sigma 0.47, 3 of 28 steps |
| Z-Anime Base | 0.15 | static shift 6: sigma 0.51, 6 of 40 steps; measured |
| Z-Image, Z-Image Turbo | 0.25 | static shift 3: sigma 0.5; arithmetic only |
| FLUX.1 schnell | 0.5 | unshifted 4-step schedule: 2 steps from sigma 0.5; arithmetic only |
| Qwen-Image | 0.3 | its published dynamic shift (max 0.9 over 8192 tokens) puts 0.3 near 0.53; arithmetic only, and from the config as published - this machine has no copy to check it against |

The rows marked arithmetic have not been run here: they are the installed
scheduler's formula applied to each checkpoint's shift, not a measurement,
and `verified.md` says which are which. Every value leaves at least one step
of its profile's own step count (`tests/test_refine.py`), or the Refine
dialog on a 4-step checkpoint would open on a request the schema refuses. A
checkpoint the table does not know has no number of its own and runs at
`refine.DEFAULT_STRENGTH`. Two consequences are FLUX.1's alone, and the
dialog says both under Strength (the profile's `refine_large` hint) once a
pass is past the tight line, where they bite: 2x redraws more than 1.5x at
one strength (at 4 MP mu is 3.2, and even the schedule's last step starts
near sigma 0.5; measured 2026-10-04, 0.1 at 2x starts at 0.752 and repaints
the man), and a Mu set for the pass moves the index back - measured, Mu 1.15
at strength 0.2 starts at sigma 0.46 with 6 steps where the native
schedule's 0.1 starts at 0.47 with 3, and at 2x it starts at 0.463 and keeps
him. The strength axis exists to put the candidates on one sheet.

**The strength's help is the checkpoint's own numbers.** The Strength tip
is two sentences - what it is, where it starts on this checkpoint - and a
sweep example, all format strings over the profile: the example is
`model_defaults.strength_sweep`, half the profile's strength, it, and half as
much again on the 0.05 grid, less any value that leaves no step (0.05,0.1,0.15
on FLUX.1, 0.2,0.4,0.6 on SDXL). The fixed example it replaced, 0.3,0.4,0.5,
repaints the subject three times on FLUX.1, and so did the old 0.4
placeholder. The dialog's Compare 3 writes the same three values into
Strength - the page's `compareStrengths`, the same arithmetic in the same
order so the floats round alike, at the Steps box's count - its placeholder
is the start and the sweep, and one visible line says how a sweep is
written. The button only fills the box; a sweep is never remembered. ArrowUp
and ArrowDown step a single strength by 0.05 (0.01 with Shift) - one fixed
step, since the profile's half would move SDXL's 0.4 past WAI's documented
0.35-0.5 in one press - and a list or a range is left alone. A strength out
of (0, 1] is refused in words that say what a strength is (`_a_fraction`, a
custom error, since pydantic's own bound read "Input should be less than or
equal to 1" in the dialog), and a single strength of 0.9 or more says in the
preview that the pass keeps (almost) nothing of the source: description, not
a refusal.

**No live start sigma.** The review asked for the preview to read the pass's
starting sigma off the loaded scheduler for the size. For an explicit Mu
that is a copy of the scheduler and one `set_timesteps`, but the case that
matters - FLUX.1 with no Mu, whose shift grows with the size - takes its mu
from the pipeline's own `calculate_shift` over the latent sequence length,
inside `__call__`. Reading it means re-deriving that outside the pipeline
(the VAE scale factor, the patch size, the scheduler's base and max shift),
which `schedule.forced_mu` exists to avoid, or running the pipeline, which
needs the gate and the card. So the dialog says what was measured instead:
the tight line, and on FLUX.1 the `refine_large` note.

**Not the author's recipe.** WAI's hires pass upscales with R-ESRGAN 4x+
Anime6B and has its own step count; this pass resizes with Lanczos and runs
the part of Steps the strength leaves. An upscaling model would be another
pinned download and another thing to verify; it is not here.

**A dialog, not a mode of the panel.** The pass first shipped as a field
under Size: the card's Refine button rewrote the panel - prompt, seed,
steps, size - and from then on Generate, Batch, Repeat, the grid and even a
prompt comparison all ran passes until the source chip was removed. That
was confusing in use, and it needed guards of its own: a remembered source
retained across reloads, Generate and Run grid blocked while a retained
source could not run, a comparison clearing a remembered strength axis, a
seed the panel shared between direct renders and passes. Since 2026-10-04
the pass is a modal dialog (`web/refine.js`), opened by a card's ⤢, the
lightbox's Refine and Reuse on a refined record, and `collect()` never sends
a source. The dialog starts from the record itself - its prompt, negative
prompt, steps, CFG, Mu, True CFG, adapter and seed; this checkpoint's
profile instead where the record is another checkpoint's, with the fold
that holds them open to say so - and has its own size (scale buttons and
the two boxes), Strength, Versions and Seed. Versions is `num_images`, on
consecutive seeds from the Seed box: a pass is one pipeline call per image,
so they cost time, not VRAM. A Strength box holding a list or a range is
the grid's strength axis instead, with Versions as its seed count, and the
job opens as a sheet when done. Each change asks `/api/grid/preview` with
the grid-shaped request - the same validation the submit runs - so a size
past the guard or a strength that leaves no step is refused in the dialog,
in the server's words, and Refine is aria-disabled with the reason as its
title. A ⤢ press on a checkpoint with no pass says so and opens nothing;
Reuse on a refined record opens the dialog anyway, with Refine dead and the
reason shown, until a status says the loaded pipeline has one. The dialog
remembers per browser (`local-diffusion.refine.v1`) the scale and versions
last run and, per checkpoint, a single strength the user ran over the
profile's - never a sweep, or the next open would start one; Reset forgets
it with the panel. Ctrl+Enter inside the dialog runs the pass, not
Generate, and its button shows the shortcut (a pseudo-element: `setBusy`
rewrites the label).

**Opened on the picture, at the controls it is opened for.** From 1600 px
the dialog is two panes: the source's own PNG as large as the window allows
on the left - a strength and a scale are judgements about this picture, and
the 72 px cover-cropped thumbnail showed the middle 56% of a 16:9 source -
with the kept region of a reshaped size framed on it and the rest dimmed,
and the controls on the right at the width they have in one column. Under
1600 px it is the one column, the thumbnail showing the whole picture. The
two wrappers are `display: contents` there, so the tab order is the tree's
at every width; the figure spans the grid's rows with the last row the
flexible one, so a tall picture lengthens that row and never spreads the
controls. It is anchored at a fixed top and grows downward: centred, every
change of height - the fold, a sweep's second preview line, a refusal -
moved the controls, the fold's own summary 136 px from under the pointer.
The top margin comes out of `max-height`, so it still scrolls on a short
screen. Focus lands on Strength with its value selected, so a typed number
replaces it, where it used to land on Refine, the last control, and the
first Tab left the dialog; a plain Enter in a one-line box runs the pass,
as a form's submit would, so "open, Enter" still runs what it opened with.
Refine - the button, Enter or Ctrl+Enter - first waits for the preview of
what the boxes hold (`refinePreviewed`, the request the last preview
answered, against the one they make now): the preview is what refuses a
strength past 1 or one that leaves no step, and it is debounced, so
"1.5, Enter" ran ahead of it, closed the dialog, put the server's refusal
in the panel and lost the typed settings with it. A press on a refused
Refine says why once: the preview's own refusal moves to the error line,
in the preview's place and at its size, with Strength focused and marked
`aria-invalid`; a size past the guard keeps its size line and focuses W -
two copies of one sentence read as two problems, and a line of another
height moved Refine from under the pointer. The dialog's column never
shrinks its items (`flex-shrink: 0`): too tall for the window, it squeezed
each to its min-height, a hint's 15 px, and a two-line preview ran over the
line under it.

**What is around the source.** Opened from Reuse on a pass, the title and a
line of their own name the pass being run again ("Refine #4 as #5 was
made", "Reusing #5: 1344×1728 at strength 0.2, seed 9"), since the dialog
opens on the source. Images are named "#N" as the gallery's ids. The passes
already made from the source are a row of thumbnails, newest first ("Passes
of #4 (3)", the gallery's own `refine:` search on every checkpoint); one
opens in the lightbox with the source ahead of it, so ← and → step between
the pass and what it was made from at the same crop, and S lays the one
over the other. A close that did not run - Escape,
Cancel, the cross, a pass opened from that row - keeps what was typed for
the next open of the same image on the same checkpoint, in this page's
memory only, and the reopened dialog says so with a Start over; Reuse, any
other image, Reset or a run forgets it. It is never written to
`local-diffusion.refine.v1`, where a sweep would start on another image.

**The pass's own prompt is checked too.** The panel's Check reads the
panel's prompt (`collect()`), which a pass never sends, so the dialog asks
`/api/prompt/analyze` with its own request - the same cold, off-the-gate
route, the strength left to the server - on open and a moment after the
prompt, the negative or what switches it changes, under an epoch of its own.
A line above the preview, outside the fold, says where each encoder stops
reading ("CLIP-L/CLIP-G read 75 of 132 tokens (22 of 42 tags); the rest,
from "outstretched arm", is never read") with Show selecting the rest, and
each rule finding that rewrites its field whole has an Apply that writes
the dialog's box through `replaceBoxText`, never the panel's: on SDXL a tag
added at the end of a long prompt is past the 77-token cut and does nothing,
which a hires-style pass is exactly where people add them. A pooled-only cut
(FLUX.1's CLIP-L) and an encoder's advice are not shown. It never holds
Refine, says it is checking only while a first use fetches the tokenizers,
and is silent when it fails. Show and Apply go dead while the boxes hold
other text than it read or the negative's activation has moved, and one on
the negative prompt is dead, saying why, while the dialog's negative box is
greyed - written there, nothing would run - as the panel's is. The lines
are rebuilt on every reply, a moment after every pause in typing, so they
are no live region: a screen reader re-read all of them each time, mostly
unchanged. A visually hidden line beside them carries their text and is
written only when it changes. On a tag checkpoint the Prompt box grows with
what it holds, and a line under it names the comma items the tag index does
not know, each once, from `/api/tags/validate` - advisory, and silent when
the lookup is unavailable, as the chips are. The source's prompt above is captioned:
the box may hold another by now.

**A pass has a size guard of its own.** `max_side` and `max_pixels` are the
largest preset because a direct render past a checkpoint's range composes
badly; a pass takes its composition from the source, so that reason does
not hold for it, and a 4 MP image could not be refined to anything larger.
`refine_max_side` and `refine_max_pixels` are any 2 MP preset at 2x (4448,
and 1776x1184 x 4 = 8,411,136 pixels), read off `sizes.py` like the others;
the page derives the same two numbers from its copy of the table, held to
the server's by `tests/test_profiles.py`. The schema's `width`/`height`
bound is the larger, and `_direct_renders_stay_in_the_presets` holds a
request without a source to the presets' side; `check_budget` holds a pass
to its own pixels and skips the batch budget, which bounds images resident
in one call. It is a typo guard, not a VRAM model, as the preset one is:
measured on 2026-10-04 (`verified.md`), WAI ran 8.6 MP at 5.5 GiB with the
card two-thirds used, Z-Anime filled the card at 4.2 MP and took 715 s at
8.4 MP while spilling into system memory, and FLUX.1 fills it at 2.4 MP.
What fits is quoted, not modelled: a profile's `refine_measured` is the
largest pass `verified.md`'s "Past 4 MP" table ran without spilling - the
row's checkpoint, its quantization and the size: 1536x1536 on FLUX.1 Krea
dev and on Z-Anime, 2560x3360 on WAI, past the guard, so no size the dialog
offers is tight there, all at fp8 - and `tests/test_profiles.py` holds each
to its row. The page gets its pixels as `defaults.refine_tight_pixels` and
its words as `hints.refine_tight`, so past it the size line says "may run
tight on 16GB VRAM, past the 1536×1536 (2.36 MP) measured on FLUX.1 Krea
dev at fp8". The run is named because a profile is shared: FLUX.1 dev and
the lite repos are the flux profile in every number and were never
measured - the same transformer at fp8 on the one, 4-bit under Nunchaku on
the others - and "measured on this checkpoint" said otherwise; so did one
decimal, "past the 2.4 MP measured" on a request it called 2.4 MP. Every
checkpoint whose profile has no row is None and keeps the panel's generic
line past 1600x1600: Animagine, Kohaku and Illustrious run WAI's pipeline,
but were not measured, and do not borrow its number. The time estimate,
from passes already made at that size, is still the honest per-checkpoint
answer.

## CellKey and run_id

**Three things widen `CellKey`, and `run_id` survives each.** The key is
(prompt, negative, steps, CFG, mu, true CFG, references, LoRA, refine,
seed), with `grid.reference_key` (ids in order), `grid.lora_key`
(`name@weight`) and `grid.refine_key` (`id@strength`)
spelled once and built from a request in `grid` and from columns in
`storage.completed_cells`. `run_id` names a cell by the **shortest** key
carrying all its non-None optional elements - it drops true CFG, references,
LoRA and refine from the right while they are None - so a grid using none of
them is still named by the six-element key its images have carried since
before any of them existed (pinned in `tests/test_conditioning.py`). A weight axis
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
profile; `+''` is 0, and 0 is a valid CFG that generates unguided. Reuse
does not carry a record's variables: it fills an editable panel, and what
Generate then makes is a new image from `/api/generate`, which records none.
A comparison version's name rides the two reproductions, Re-render and the
CLI's `--from-image`, not Reuse. A refine pass is the Refine dialog's, not
the panel's (see "The refine pass"): Reuse on a refined record asks
`GET /api/images/{id}` for its source and opens the dialog on it with the
pass as it ran, and only when the source has gone does it fill the panel -
its prompt and sampling, at the panel's own size, with a note saying the
pass cannot be made again. The dialog fills its own boxes; nothing of the
panel's `fillPanel` is shared with it.

**Reuse can be taken back.** It replaces the prompt, the negative, the seed,
the size, steps/CFG/mu and the references and adapter in one click, and it
used to do so over a half-written prompt with no word: the boxes were
assigned `.value`, which empties a textarea's undo history (and in Chromium
a later Ctrl+Z could splice the old text into the new, which `persistPanel`
then remembered). `fillPanel` now writes both boxes through `replaceBoxText`,
as Use and Apply do, so Ctrl+Z walks back through them, and still
synchronously - the references are in place when Reuse returns. Before it,
`reuseIntoPanel` keeps `panelState()` - the object the panel persists - with
the raw steps/CFG/mu/true-CFG values and the `owned` flags beside it, and its
note opens with "Settings from #N loaded - your prompt was replaced." (the
last clause only when there was other text) and an **Undo** right after that
first sentence, which puts all of that back, the boxes again through
`replaceBoxText`. The note stays in the panel's line, where Reuse sends the
user; whatever else it has to say - the checkpoint-mismatch sentence, kept,
and the rest below - follows the Undo, so the action is on the note's first
lines, the ones in sight where the window shows the prompt and only the
start of the note. The note also says what `fillPanel`
could not carry, since a dropped reference renders a plain text-to-image
that looks like a success: references on a checkpoint that takes none,
and - found by asking, after the synchronous fill - an adapter the LoRA box
does not offer (the reason is `/api/loras`' own: a misfit's `detail`, an
installation that cannot load adapters, or a file no longer in the
directory) and a restored reference the server answers "unknown image" for,
which is then dropped. Only that answer drops one: a passing failure must
not take a live reference out. A reference deleted from the gallery while
the Undo is still on offer leaves the Undo's copy of the panel too
(`forgetImages`), or the Undo would bring back a chip the submit refuses.
The Undo is on offer only while the panel holds what the Reuse left there
(`reuseNote.after`, compared by `syncReuseUndo` on every `persistPanel`):
the first change - typing, Ctrl+Z, a seed, a preset, an Example - takes the
button and leaves the note's sentences. It used to stay through anything, and
the integration review pressed it after typing, a new seed and an Example
pick: all three were thrown away, the seed beyond Ctrl+Z's reach, and the
note said the panel was as it was - while the Example's own Undo, which
goes once its box changes, showed beside it with the opposite meaning. So
the page has the Examples' rule for both and one Undo on offer at a time: an
Example retires a Reuse's, and a Reuse an Example's (`dropPromptUndo`). A
reference dropped from what the Reuse left - deleted, or answered "unknown
image" - is no change of the user's, and comes out of `after` as well.
Both boxes are written through tags.js's `writeBox`, the page's one
whole-box write (`rewriteBox` focuses the box without a scroll first).

**Where Reuse leaves the page** is `revealReused`'s: the prompt's field,
label and all, in sight under the topbar, and the user's place in the
gallery kept wherever that allows. It used to scroll to the top of the page,
which beside the gallery threw that place away for nothing; then to the
prompt with `scrollIntoView({ block: 'nearest' })` when the prompt was out
of sight, which left it under the sticky topbar on one column and, beside
the gallery, scrolled a page whose stuck panel does not follow a page
scroll - 600 to 750 px of gallery lost at 1400x900, the prompt still out of
sight. The browser's own scroll to the focused box was no better:
`replaceBoxText` focuses the box, and the browser scrolled the page as
though a stuck panel moved with it (half a screen at the page's end), or
not at all for a prompt focused already. So `fillPanel` and the Undo focus
each box with `preventScroll` first (`rewriteBox`), and once the note is
written - the panel at its final height, placed at once rather than after
the frame - the panel is moved, not the page (`revealInPanel`, see "The
settings panel's sticky top"). Only where it cannot be - a one-column page,
or a panel at the page's end, which a sticky top cannot push down - does the
page move, as far as puts the panel where it rests, smoothly unless reduced
motion is asked for. Then, if the Undo is below the window, a stuck panel
goes up to bring it into sight, but only as far as the prompt's field stays
clear of the topbar (`showUndo`, after the smooth scroll ends): at QHD and
at 125% both fit; at 1400x900, a four-line mismatch note included, they do
not, and the prompt is what Reuse is about - Ctrl+Z in it still gives the
text back. `smoke-delete.json` holds all four: the page unmoved
mid-gallery with the prompt focused already, the page's end, one column,
and the Undo in sight at 2048x1040 at 125%. The lightbox's Reuse closes the
lightbox first: a modal dialog keeps focus to itself, and `replaceBoxText`
needs the prompt focused.

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
close" for the rest of a grid. `main.Server`, the `uvicorn.Server` subclass
`main.run()` serves with, sets `cancel_requested` on every active job in
`handle_exit`, i.e. on the signal itself, and starts the critic's stop on a
thread of its own; `run()` sets `timeout_graceful_shutdown=30` for a client
that stops reading. `Server` sits at module level so
`tests/test_api_analysis.py` can call `handle_exit`; that file also pins the
lifespan's order: the critic's server, then `jobs.shutdown()`, then the
weights. Lifespan calls `backend.try_unload()` (non-blocking lock acquire)
off the loop rather than `unload()`, which blocks behind a load in progress
and then discards what it built; `POST /api/model/unload` answers 409 while
a load is in flight for the same reason.

The load runs on a **daemon thread** (`main._detached`), not
`asyncio.to_thread`: that pool is joined at interpreter exit, so a Ctrl-C
four minutes into a 32 GB download waited the download out. The critic's
`prepare()` runs on one too (`critic-prepare`), for the same reason: a first
use downloads about 5 GB. Since 2026-09-28 Check's analysis and the
critique's analysis and lint do as well (`prompt-check`), and each tokenizer
load runs on a daemon thread of its own (`prompt_encoding._fetch`), because a
first use's Hub fetch has no HTTP timeout; uvicorn still answers an open Check
before it stops, which `LOAD_WAIT` bounds.
`test_prepare_and_the_load_run_on_daemon_threads` drives the critique and the
load through `httpx.ASGITransport` on a loop in pytest's main thread, as
uvicorn runs its loop, not through TestClient: TestClient runs the app's loop
on a daemon thread, every thread that loop starts inherits the flag, and an
`asyncio.to_thread` or Starlette threadpool worker would pass for a daemon
there. After the calls it checks that each ran on a daemon thread that is not
in `concurrent.futures.thread._threads_queues` - the interpreter joins every
thread in that registry at exit whatever its flag, and a worker is added
only once it has started.

Unload stops the critic's server too, and all three stops go through
`main._stop_critic`. Unload's runs off the event loop - a stop waits up to
twice `STOP_TIMEOUT`, which on the loop would stall every progress stream
(`test_unload_says_unloaded_before_the_analyser_and_critic_let_go` holds it).
`prompt_critic.unload()` raises `CriticError` when a kill is not waited out.
Raised through the lifespan, that skipped `jobs.shutdown()`, `try_unload()`
and the tokenizers' unload; on the signal's thread it was an unhandled
thread exception; through `POST /api/model/unload` it made the route a 500
over weights already gone and left the tokenizers cached. It is logged and
the rest goes on. Unload still answers with the status, which stays true:
"unloaded" is the image model's state, and `vram_used_mb` is read off the
image model's card (`torch.cuda.mem_get_info` on `settings.device`,
device-wide), so on this app's one card, the memory `llama-server` still
holds shows there rather than being claimed free. `llama-server` is pinned to
no card, so on a second GPU only the ERROR log line, with its pid, says so.
No field of its own was added for a case that needs a kill to fail.

Once shutdown is complete, uvicorn re-raises the signal it captured, so a
Ctrl-C reaches `main.run()` as KeyboardInterrupt. `uvicorn.run()` swallows
it, but `run()` calls `Server(config).run()` directly for `handle_exit`, and
every Ctrl-C ended in a traceback after "Application shutdown complete".
`run()` exits 130 instead, as the start-up menu does on Ctrl-C;
`test_ctrl_c_ends_the_server_with_130_and_no_traceback` sends a real SIGINT
to a started server. Launched from a script without job control, a
background server has SIGINT ignored, and there the re-raise is a no-op that
exits 0.

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
can set `LOCAL_DIFFUSION_MAX_BATCH_PIXELS` explicitly. A refine pass is
outside it: it makes one pipeline call per image, so its batch never holds
more than one image at once ("A pass has a size guard of its own").

The panel labels the count **Repeat**; `request.runs`, the `runs` box id, its
remembered key and the CLI's `--runs` keep the name. "Run" is the
submission everywhere else on the page - the gallery's Runs picker lists one
Repeat-3 Generate as one run, `run:` searches for it, the last-run banner
reports it - and a panel box of the same name counted something inside it
(U88). Batch has a badge saying why Repeat exists, and the line under the
boxes says batches, not passes, which is the Refine dialog's word. In one
column (under 901 px) the badge's tip opens leftwards: there the badge is
600 px along an 800 px row, and a tip opening rightwards ran past the window
and widened the page.

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

The one mistake neither check catches is a variable that does not vary.
`size = big, small` is legal - one value, `big, small` - and the night runs
`a big, small fox` on every cell. The preview listed the resolved prompts
only from two up, so it went quiet exactly when the substitution failed. It
now lists the prompt whenever the variables box has text (one as `prompt:
...`), and the page marks any line of one value in the warning colour,
naming the comma when that is why. That warning is the page's alone and
advisory: the `|` split, and the parser that decides, stay the server's.

The placeholders are also why **Ctrl+Enter goes where the focus is**. It
used to generate from anywhere, so pressed in a grid box it rendered the
literal `{size}` as one image, and the night's grid never started. From a
box of the open fold it now presses Run grid (its click, so the guards are
the button's), in the Refine dialog it runs the pass, and anywhere else it
generates. Generate itself refuses, in the page only, while the fold is
open, its variables box has text and either prompt holds a `{name}`: the
server never refuses braces, which in a plain prompt can be text or pasted
`{tag}` emphasis. Both buttons show the shortcut as generated content with
empty alt text, since `setBusy` rewrites their labels as text.

## Prompt labels on the sheet

The contact sheet's axes gained **Prompt**, whose labels are only the words
that vary across the run (`promptLabels` in `web/sheet.js`): a column header
repeating the forty words every cell shares is unreadable. Equal word counts
let the differing positions be picked out exactly; a `none` makes one prompt
shorter than its neighbours, and then the shared head and tail come off
instead. Pins address their values by position, because a shortened label is
not something that round-trips back to the prompt it came from.

Successive edits of one prompt defeat both: different word counts, and a word
added at the front keeps every head and tail, so six iterations of a portrait
labelled themselves with three clipped openings, five of six columns alike.
Where the two paths leave labels repeated or longer than `LABEL_CHARS`,
each prompt is labelled by its edit against the oldest (`promptEdit`, over
`promptChanges`, the critic's exact diff): `+, film grain`, `−golden +blue`,
runs of edits joined by `…`; the oldest keeps its own words. That needs the
prompt axis in the order the prompts were written, so `sheetValues` orders it
by oldest image as it does a variable (alphabetical put `35mm photo, …`, the
newest, first). On screen a label is whole - a heading wraps to three lines
and its title is the full prompt - where a 540 px column used to show 44
characters; `clip` stays for a line of fixed width: a pin's option, the note,
the export's canvas. Each edit is a diff, quadratic in the prompt's length, so
the edits stop at `EDIT_LABELS_MAX` (64) prompts - past that the labels are the
varying words, as before - and the labels of the last prompt list are kept
(`promptLabelMemo`) for the redraws a pin, an axis, a star or a delete asks
for: a typed search over a thousand 250-word prompts took 600 ms a redraw.

A comparison's `var:variant` axis reads as the prompt check names its
versions (`VERSION_NAMES`: Original, Reordered, Rewrite, Your version) under
the axis label Version, on the sheet and in its export, where it read `variant
yours`. The stored labels are untouched - they are what `stats --by variant`
groups by - and a label the map does not know reads as written. Where each
label names one prompt (`versionPrompts`; where one names several, Prompt is
kept as an axis instead), a version's row head carries the whole prompt as its
title and, under the name, its edit against the original, up to 80
characters: four 400-character versions open with the same words. A version
that keeps the original's words and only moves some - the critic's Reordered -
says `same words, moved: soft light` (`versionEdit`), each moved run once from
the added side: as an edit it read `+soft light, … −, soft light`, every moved
run twice.

## Grid preview and the per-step estimate

`POST /api/grid/preview` runs the same expansion without queueing it, so the
panel's count, its resolved axes and its error text are the run's own. Its
direct-render duration estimate is per *step* (`storage.step_cost_ms`), measured
from recent direct gallery rows at the same size **on the same checkpoint** (klein's step is half
of Turbo's at 1 MP): a grid's cells disagree about step count, so an average
image duration would misstate most of them. Refine estimates instead use full
per-image durations for the matching denoised step count, as described in
the refine section. No matching history, no estimate.

`grid.describe` names every axis, ones included, with an ASCII `x`: it is
the CLI's log line as well, and its tests hold it to that. The page trims
the factors of one but the seeds from the preview's first line and writes
the sign as `×`, since the axes line under it lists every value anyway. That
first line - the count, the time, the disk - is what decides whether to
start a night, so it reads at body size; the axes and notes stay hint text.
Under the prompts come the notes: cells already in the gallery (below), a
variable that never varies, Repeat above one (a grid steps its own seeds,
so the panel's `runs` is sent as 1 and the preview says it does not apply),
and a time budget. The budget line divides the estimate by the seed count -
every seed runs the same cells, so a seed's share is exact under the
estimate - and says how many seeds the minutes complete, warning at none;
it needs the estimate, and is left out under Resume, whose seeds are no
longer alike. The budget runs from submission (`jobs.submit_cells` sets the
deadline), so the line says the load counts too while the model is not
loaded, and the jobs ahead while the page is watching one: a Run grid
pressed then queues, and its minutes go on the wait as well.

The preview is asked again whenever something it reads moves, and not all
of that is a box: `setBusy` asks on a job's start and end (the gallery
holds a finished job's cells, and the queue ahead has emptied), and
`refreshStatus` when the model becomes ready or stops being so. Before,
only the boxes asked, and the two lines added for after a night - the
cells already in the gallery, the load - stayed as they were before it.
The first reply after the fold is opened by hand reveals Run grid again
(`gridRevealPending`): the reply lands under the pointer that opened the
fold, where the panel holds still (`panelResized`), and its lines pushed
Run grid back below the window the opening had just brought it into.

Axis messages lead with the name of the box they came from - `Steps`,
`CFG`, `Mu`, `LoRA weight`, the Refine dialog's `Strength`, `Seed list` -
since the preview prints them under the whole fold. A range may be typed
with spaces (`6 - 12 x 2`): `parse_axis` closes up a `-` or an `x` between
two numbers before it splits on whitespace, which had cut `2 - 5` into a
`-` refused as a none token everywhere but Mu, where `0.8 - 1.2` silently
became three cells. Closed up, `0.8-1.2` is a range with no step, counted in
ones - the one value 0.8, which the trimmed totals would not show either -
so a range its ones never sweep is asked for its step (`needs a step, e.g.
0.8-1.2x0.1`), typed with the spaces or without. A `-` with no number on
both sides is still Mu's none, and `6..12` or `6 to 12` is answered with the
syntax this one uses.

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

Without Resume, the preview still asks the same lookup and reports what it
found as `GridPreview.in_gallery`, dropping nothing: Run grid again after a
night with a failed cell rendered the whole night again, with nothing
before the press saying that Resume would not. Only where the seeds are
fixed - an explicit base or a list - since a random base matches nothing
and a lookup for it would be a scan per keystroke for a zero.

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
`web/lightbox.js` is a scale relative to fit plus a pan, never below fit or
1:1, whichever is smaller, kept across records whose shape agrees, so the
arrows compare the identical crop at the identical magnification down a
steps or mu axis - and, since 2026-10-04, between a pass and the source it
was scaled up from, which share a shape and not a size. "Agrees" is a ratio
within 1% (`lbView.aspect`), not a key: the 16-pixel grid puts 2048x1152
and 1360x768 apart in the third decimal, where both rounded keys tried for
it broke; a carried scale is clamped to the new record's own floor and
ceiling. The floor used to be fit: on a 2560x1305 screen every preset up to
about 2 MP fits larger than 1:1 (a 1 MP square at 124%), so the 1 key, the
1:1 button and the double-click all did nothing on a default-size image,
and a sharpening sweep was judged through a 1.1-1.65x upscale. The zoom
line tests 1:1 first, then fit, then a plain percentage - a wheel stop
between the two is neither - and a double-click goes back to fit from any
other scale, a 1:1 below fit included. The image is laid out with explicit
width/height/left/top inside a `flex: 1 1 auto; min-height: 0` stage - not a
CSS transform, which can composite a fit-size texture and scale it - and
1:1 divides by `devicePixelRatio`, or Windows scaling makes it a 1.25x
resample, exactly the smoothing a sharpening sweep is judging. The wheel
listener is non-passive; the zoom keys sit below the `input, textarea`
guard, and below one for modifiers, so Ctrl+F and Ctrl+plus stay the
browser's. A click on the stage beside the image closes the lightbox only
at or below fit: zoomed, the stage is a pan surface and only Escape or the
cross close it. Below fit that close waits out a double-click
(`lbCloseTimer`): the picture is smaller there, and the double-click that
gives fit back can start on the backdrop. Where a press landed is read on
`pointerdown` and kept in `lbDrag`, and the double-click handler is the
stage's and hit-tests the picture (`elementFromPoint`, which honours the
comparison's clip-paths): the stage takes pointer capture on press, and from
then on every event of that pointer - `pointerup`, the `click` and the
`dblclick` made from it - targets the stage whether the mouse is over the
image or not. Testing `e.target` on the way up closed the lightbox on a
single click of the image, and a `dblclick` listener on the image never
heard one; `tools/smoke.json` clicks and double-clicks the image for both.

**A pass is compared with its source in place.** The only question a pass
asks is whether it improved its source, on detail, at a strength of 0.1,
and no route showed the two at one crop: the source was seldom next to its
pass in a strip, a step between them reset the zoom, and the viewer had
replaced it with the pass. On a record with `refine`, S (or Compare with
source, with its two mode buttons) lays the source under the pass and shows
it left of a divider (swipe), or beside it (side by side: each half fitted,
showing the same crop), on the one `lbView`, through the same explicit
geometry. The source is fetched by id once a session (`lbSources`; a strip
that holds it already is looked in first) and drawn as `refine.fitted`
cropped it - `refineKept`'s arithmetic places its whole picture so the kept
part lands on the pass's box, and a clip-path cuts it to that box, its half
and the divider - so a reshaped pass is compared with what it kept.
Switching between the one view and the halves keeps the magnification and
the centre (fit stays fit), so a 1:1 stays a 1:1. A step to another image
ends it, its mode kept; a re-render of the same image (a star, a status)
does not. Where the source has gone (`refine_missing`, or the fetch's
"unknown image") the controls stay, `aria-disabled`, saying so. The divider
is dragged (a press on it is not a pan) or moved by its arrow keys, which it
takes before the lightbox's walk does; hidden with the focus in it - S
pressed there - it hands the focus to the toggle, not to the top of the
dialog.

**The controls are beside the stage, not under it.** The dialog is a row:
`.lb-main` (the stage and its arrows) and a `.lb-side` column (280px;
340-400px from 1800 px, `--lb-side-w` with the wide tier's other widths) with
the cross and the counter, the prompt and the facts, the error slot, the
comparison, the keys line, the zoom row and the buttons. They used to sit
under the stage as a meta line and a row of buttons, which cost it around
150px of height on a 900px screen - and the screen is wider than it is
tall while most images are not, so height was the axis nearly every fit was
bound by; a square or portrait picture is still bound by the height, so the
wider column costs it nothing, and a 16:9 one about 3%. In the column the
prompt and facts scroll (`flex: 1 1 auto; min-height: 0`) and the controls
are pinned to the bottom, so Star and Delete are where the mouse left them
from one image to the next however long the prompt is; `#lb-close` carries
`autofocus`, or `showModal()` would focus the first control in the tree,
which is now an arrow. The facts are a list (`factsList`, a `dl.facts` any
view of one image can use): a row a fact, where a 12 px line of commas
buried the seed and left out the negative prompt its True CFG applied to,
the date and the run. The source, each reference (numbered as the panel's
chips are, with "Use these references") and the run are buttons: the first
two open that image with this one after it, so → comes back; the run opens
as the card's ⊞ does. Use these references asks for every id again, as
Reuse does, and leaves out and names any the gallery no longer has: put in
the panel, a deleted one was an empty chip and a refused Generate. It is
`aria-disabled` with the reason where the checkpoint takes no references or
every one has gone, and a render of the same facts brings that up to date
(`syncUseReferences`), since a status after a load can change the first. A source's passes are linked as the gallery's
`refine:` search once the gallery says there are any. The one-line
`settingsLine` stays as the dialog's description (`#lb-desc`, hidden) and
the delete confirm's. The counter names the image ("3/144 · #57") and
counts the gallery by the server's total, not the pages loaded. A picture
that will not load says why in the lightbox's own line, the record asked
for to tell an outage from a deletion.

**The arrows are strips, and a close returns to the walk.** The arrows are
positioned in `.lb-main`, over the stage's edges but *not* inside the
stage: the stage takes pointer capture on press, so a press on a child of
it would have been a pan, and at fit a close, rather than the button's
click. A 40 px circle there left 560 px of backdrop each side of a portrait
picture at QHD, and a press 10 px off it closed the lightbox; so at fit,
with backdrop beside the picture, each arrow is a strip as wide as that
backdrop less a margin (48-160 px, `--lb-gutter`) over the middle 60% of
the height, its circle where it always was. The corners stay backdrop,
which `tools/smoke.json`'s close at [30, 30] relies on. Anywhere else the
arrows are their circles again: zoomed in, a strip would take the pan
surface's drags, and below fit the double-click that gives fit back - a
strip there stepped on its first click instead. So is a disabled arrow (the
first image's ‹, the last's ›): a disabled button takes the press, and as a
strip it left that side's backdrop closing nothing. The wheel is the
stage's parent's, so it zooms over a strip as over the stage rather than
scrolling the page behind the dialog. A press on the stage within half a
second of one on an arrow is the second half of a double-click on the
arrow (`lbAfterArrow`), neither a close nor a 1:1: the step can take the
strip from under the pointer when the next picture fills the width. The
arrows' own clicks are not filtered by click count, which would have cost
a quick walk every other press. The backdrop shows the default cursor, the
picture the zoom one. A close of
the user's own - Escape, the cross, a click on the backdrop or beside the
picture (`dismissLightbox`) - focuses the image the walk ended on, its card,
tile or sheet cell, which scrolls it into view; a close that hands the page
to something else - Refine's dialog, Reuse's panel, a delete, Show this run
- is `closeLightbox` alone, since a focused card would scroll the page back
from where those took it. The Refine dialog closes the lightbox for itself;
closed without running, it gives the lightbox back on the same image of the
same strip, at the same view and comparison (`refineReturn`, resolved by
id, since a job landing meanwhile reloads the gallery into a new array; the
comparison put back before the view, whose scale is relative to the fit of
half the stage side by side - on the whole of it a 1:1 came back at 166%),
unless its passes row has opened the lightbox on a pass already. Below 900px wide the column
drops under the stage as a wrapping row, without the keys line. The
`tools/smoke.json` grid is 1152x896 for the same reason the layout changed:
at the driver's 1400x900 a 1024x768 image fits at 104%, and its 1:1 is
below fit; `tools/smoke-lightbox.json` has the rest.

## The sheet exports client-side

**The sheet exports client-side.** `renderSheet` leaves its matrix in
`sheetLayout`; `buildSheetCanvas` draws the same axes, pins and labels onto
a canvas from the *thumbnails* (cells shrink so the longer edge stays under
8000 px), with a caption naming the search, the pins, the checkpoint, the
run and the time, and `exportSheet` downloads it. Not a server route: that
would duplicate a layout contract the client already holds, the
non-round-trippable prompt labels especially.

The caption is what the image has to say on its own later: the prompt (the
run's one prompt, else the first image's, with a variable run's values as
declared - `size = big | small; ground = grass | dirt` - since a template is
not a prompt any cell ran), the Fixed line, and the time the run was made,
the oldest image's, where it used to print the moment of the export. The file
is named by the run and the pins (`sheet-<run>-cfg-2_seed-42.png`), since
every slice of a run used to download under one name. The export is still
the slice on screen, as the screen is: an "every pin" export was left out. A
wrapped single row (see "The contact sheet") wraps here too, `ceil(√n)`
across with each cell captioned under it - drawn as one row, the 96 px floor
beat the edge limit and a 120-seed run came out 12,254 px wide - and a refine
sweep's source is drawn as the first column, headed `Source #<id>`.

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
`save_image(variables=)` from both writers and lands in a `variables TEXT`
column (JSON; `_migrate`), the `local-diffusion` chunk and `ImageRecord.variables`.
Both reproductions - `/api/rerender`, and the CLI's `--from-image` while the
prompt pair is unchanged - hand a row's variables to the cells they build.
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
nothing about the run. In particular, swapping original and yours on the
same seeds shares a run, although the labels now mean different prompts.
The comparison therefore opens `job:<id>`, its own submission, and its
statistics recipe uses `--job`. This keeps existing run ids and partially
completed runs resumable. When a sheet reads several such submissions, the
flat Prompt axis remains available if a complete variable assignment names
more than one prompt: otherwise its (variant, seed) cell would silently keep
only the newest. Ordinary variable grids still suppress that redundant axis.

**The sheet lists a variable's values in the order the grid wrote them.**
`sheetValues` used to sort a `var:` axis with `localeCompare`, which read a
`size = small | big` grid as big, small - against the declared order
`parse_variables` promises. (The comparison's own labels - original,
reordered, rewrite, yours - happen to sort alphabetically in the order the
page sends them, so they never showed it; an earlier version of this note
said alphabetical put `original` below `reordered`, which it does not.) It
orders by the oldest image carrying each value: the grid renders values as
declared and variants in the order given, seeds outermost, so the first
image of each is in written order. Numeric and flat-prompt axes are
unchanged. The oldest *loaded* image is only the oldest a filter kept: star
the winners, tick Favorites, and the order was whichever survived (a
`size = small | large` grid over seeds 20 and 21, with s20-large and
s21-small starred, read large, small). So when Favorites is ticked, or the
search is anything but `run:<id>`, a sheet of one run with variables asks
once more for that run unfiltered before it renders (`learnSheetOrder`, from
`loadGallery`: the one extra request) and orders by the whole run's oldest
ids; if that request fails, the loaded images order themselves.
`chooseAxes` gives a varying `var:variant` axis the rows whenever anything
else varies: widest-first alone made the variants rows only while the seeds
outnumbered them (3 x 3, 3 x 2 and 8 x 4 all turned them into columns, the
tie going to the `var:` axes because `axesFor` lists them first). The
variants and the axis across them are placed from every varying axis, not
from what the listlike rule left. With one image per version loaded (a star
each under Favorites, or what Delete unstarred kept), that rule dropped the
variants when each version had one image, or the seeds when each image had
its own seed, and the pin that took their place showed one image of the
comparison. Alone, the variants stay a strip in the columns, because rows
with nothing across are one column of full-width images. A variable a user
names `variant` by hand gets the rows too - the one name the sheet
recognises, held equal to `grid.VARIANT` by `tests/test_grid.py`. Known gap:
ordering by oldest id still moves a value down for the run when its first
cell failed (a job skips a failed cell), when a resume rendered it later, or
when Delete unstarred removed the rows that carried it first, since the
server no longer has them to ask for. Recording each cell's declared index
would close all three; it would need a new column and is not done.

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

**`--from-image` carries the variables** (`_image_variables`), as
`/api/rerender` does, because they are the only place a comparison
version's name lives. They carry only while the prompt pair is the image's
own, compared as text after the fill so a retyped prompt keeps them, and
never beside a `--grid-prompt` that holds a variable (a blank one is no
axis): a variable may have filled either half, and a label on another text
names a version it is not. A comparison version (`{"variant": label}`) goes
back through `expand(variants=[...])` as a one-variant prompt axis, so its
prompt is taken verbatim and a `{word}` in it is text, not a placeholder
with no values. A `{var}` assignment is already substituted into the prompt
and is attached to each cell as it is, leading the cell's log label as a
grid's own does. So is a `variant` value that is no label a comparison
could have recorded, because a user's own grid may name its variable
`variant`: its `none` value is empty and a long one is past
`PromptVariant`'s 40 characters, and the schema refusing it is what picks
the attach route. Known gap, older than that carry: every CLI run goes
through `expand()`, whose placeholder check `/api/generate` skips (see "The
CLI's loop, and jobs.submit's"), so a plain prompt holding a literal
`{word}` - typed, or read back by `--from-image` from an image Generate
made - exits 2 with `prompt: {word} has no values`; only a carried
comparison version is taken verbatim.

**`stats --by variant`.** It groups on `json_extract(variables, '$.variant')`
(`grid.VARIANT`), a comparison's version by its label. `--by prompt`, the
recipe first written for a comparison, printed one identical row per
version: a cell holds 37 characters plus `...`, and reorders share their
opening words. An XML prompt's newlines also broke the table; `_cell` folds
whitespace to single spaces. Rows rank by rate, then count, then first
appearance (`MIN(id)`), so a tie - an untouched comparison at 0% everywhere -
reads in written order, not in the label-alphabetical order of the GROUP BY.
The ranking itself is the table's contract, so versions are not forced into
written order. A comparison is read with `--job`, since run identity ignores
labels. A broader scope that gives a complete variable assignment two prompt
texts is refused with advice to use `--by variant,prompt` or a narrower job.
The check canonicalizes variable key order and stays within each emitted
stats group, so an ordinary `variant` by `colour` grid can aggregate colours,
and an explicit seed or steps grouping can separate different meanings.

**The CLIs write UTF-8 when redirected.** `generate`, `clear`, `stats` and
`reindex` reconfigure stdout to UTF-8 (`errors="backslashreplace"`) at the
top of `main()`: on Windows always, elsewhere when stdout is not a terminal.
Redirected or piped on Windows, stdout encodes with the locale's code page
(cp1252); a CJK or emoji prompt, or a CJK `--from` folder name, was then a
UnicodeEncodeError partway through the output, and `stats ... > out.txt`
exited 1 with only its header in the file. `isatty()` cannot be the Windows
test: `> NUL` is a character device, answers True and still encodes with
the code page, while the console itself is UTF-8 already
(`_WindowsConsoleIO`), so reconfiguring it changes nothing. A stream with no
`reconfigure` is left alone - a StringIO, or pythonw's `None`, where a
scheduled `clear --yes` must still run; the first version called `isatty()`
before that check, which made it an AttributeError. `reindex`'s per-file
lines go to `out`, bound to the same stream when the module is imported, and
reconfigure changes that stream in place, so they are covered too.
`tests/test_cli.py` holds the four copies to one setup. Windows PowerShell's
own `>` is beyond it: that re-decodes a program's bytes with
`[Console]::OutputEncoding` before writing the file, and the README says how
to set that to UTF-8.

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
record's prompt, negative prompt, steps, CFG, true CFG, mu, references,
adapter and weight, refine source and strength, seed and variables, oldest
first. It calls
`check_budget()`, refuses more than `max_grid_images` matches by count
rather than truncating, refuses records from another checkpoint (steps/CFG/mu
are the checkpoint's - only `all_models` lets one into the selection),
prunes with `completed_cells` like a resume and 409s when nothing is left.
Same seed at another shape is a *new sample* - the latents are drawn at the
requested size - which the README says out loud; getting the *same picture*
larger is what the refine pass is for. A refined record is the exception
that proves it: it re-renders as a refine pass of its own source at the new
size, which is the same picture again. Until 2026-09-28 the cells
were built without `variables`: a re-rendered comparison lost its version
labels, which live nowhere else, and its sheet fell back to a flat,
alphabetical Prompt axis; a `{var}` grid lost its axes the same way. A
selection mixing records with variables and records without them lays out
like the same mixed filter over the originals - usually Favorites over a
`{var}` grid plus some one-off prompts: the variable axis replaces the Prompt
axis, the plain records sit under `none`, and where two land in one cell the
sheet shows the newest (`axesFor` and `renderSheet` in `web/sheet.js`).
Keeping the Prompt axis for such a mix is a sheet change, not a re-render
one.

## Delete unstarred

**The gallery's one bulk delete is the inverse of the card's.** A night
makes hundreds of images and a few of them are worth keeping, and deleting
the rest one card at a time - each with its own confirm - is a chore that
was being skipped, leaving galleries of hundreds. The star already existed
as the gesture made while looking (`F` in the lightbox), it is
non-destructive, and it scales with what is kept rather than what was made;
so the page offers `GET`/`DELETE /api/gallery/unstarred?search=` - the
search box read through `strict_scope`, plus `favorite = 0` less what a
starred image was made from, plus the gallery's checkpoint scope unless
`all_models=true` - which is
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

**A star keeps what its image was made from.** Refine a source, star the
pass, sweep the rest: the sweep used to delete the source's PNG, the kept
pass then read "refine of #63 @ 0.1" for good and could never be made again,
and nothing before the click said so. The source is part of the pass's
reproduction set, and a klein render's references are part of its, so
`keep_favorites` keeps them with the stars: one predicate in storage
(`_UNKEPT`: `favorite = 0` and not the `refine` or a `json_each(reference)`
value of any starred row, one level deep, neither arm able to yield the NULL
that would turn `NOT IN` into NULL), shared by `delete_matching`,
`clear_history` and the counts. The count gate moved with it, or every
sweep would have answered 409: `matching_preview` returns `sources_kept`
(and the first few as `sources`), the route's `unstarred` is `records -
favorites - sources_kept`, and the CLI's two "Would delete" lines and
`summarise()` take the same field. Sources stay ids, and the dialog names
them: "keeping 4 starred and #63, which a starred image was made from".
A per-image delete of a source is still the user's to make, but its confirm
first asks `refine:<id>` across every checkpoint and says which pass could
not be made again; a record whose source has gone carries `refine_missing`
(looked up by `list_images` and `GET /api/images/{id}`, not stored), and the
lightbox's facts read "Refined from #63 (deleted) @ 0.1", its comparison
dead, saying "source #63 deleted".

**The dialog describes the scope, not three prompts.** The sample prompts
are the first three alphabetically, and say nothing of how many runs are
about to go or from when, so `matching_preview` also counts the runs the
delete reaches into and the first and last creation times among what it
takes ("From 23 runs, made Oct 4, 04:32 PM to Oct 4, 04:40 PM"). With All
models ticked the scope names every checkpoint it reaches from the server's
own `models` (six at most, "6 or more" at the cap), as the narrow scope
names the one. While this page is watching a job, the dialog says a landed
image refuses the sweep; and when one does - the 409 with both numbers -
the page reloads the gallery and the Runs picker, so the count beside the
message is the server's new one, and says to press again or let the job
finish.

**Everything it says, it says by the gallery.** The outcomes - "Deleted 12
images, keeping 3 starred.", the wrong number, the free-text refusal, the
409 - go to `#gallery-error` under the gallery's head (`showError`'s `slot`),
as do a failed gallery page, Re-render's submit errors and a card's
refusals. At QHD they used to land in the panel's line, 2,400 px from the
button; a sighted user saw a click do nothing. One slot per message, since
both are role=alert; the lightbox mirror is the same. `resetGallery` clears
the slot, so the sweep writes its outcome after the reset it starts.

**The page forgets by id.** The `DELETE` returns the ids it removed and
`forgetImages` - shared by the card's delete, the lightbox's, the sweep and
a viewer tile whose image would not load - drops them from every list and
element on the page in one pass, the panel's references included, then the
sweep's `resetGallery` reloads the (now mostly empty) page from the top for
the server's count and `loadRuns` the picker's. A single delete keeps its
place: the card becomes a tombstone ("Deleted · #N", dashed, no
pointer events, no `data-id`; by id, as the page names images, since the
cells of a sweep share a seed) and a viewer tile one of the tile's frozen
size - removed, the next card slid under the pointer that had pressed ⌦,
its own ⌦ lit, and an emptied viewer shrank the page by 537 px - and focus
on what went moves to the next image in the strip, else the one before.
A viewer whose last image went, with no job to fill it, says what it is for
again ("#9 was deleted. Generated images appear here.") in place of its
tombstones - a frozen box with the placeholder beside it read as two
leftovers - held at the strip's height so the gallery still does not move,
until the next run's first image clears it; the results viewer counts its
live tiles, not its slots, for the 48-tile cap and the "and N earlier" note.
The per-image confirm names
the image by id with the lightbox's settings line (`settingsLine`), since
a sweep's cells share a prompt and a seed. Asking `refine:<id>` for the
passes made the confirm wait on a round trip, and a double-click asked
twice and sent two deletes, both answered 200 since `delete_image` looked
the row up outside the write lock; a delete under way now ignores a second
press of the same image (`deleting`), and `delete_image` looks up and
deletes under the one lock, so a second delete is "unknown image". In the lightbox a delete says
which image went in its line until the next key or press, and a delete of
the last image loaded, with more on the server, pages the next one in
(`pageInBehind`, "Loading next…" meanwhile) rather than stepping back onto
an image already judged. A delete answered "unknown image" - another tab
took it, or a reattached job had replayed it - forgets it with a note. The
server does its part: the delete routes call `jobs.forget_images`, and a
job's snapshot replays only its live records, with `deleted_count` beside
an unchanged `image_count` (what the night produced), so a reload during
a run no longer brings a deleted image back as a broken tile. The smokes
stub `window.prompt` to drive the dialog: a wrong number first, then the
right one, then the free-text refusal, checking that nothing went until
the right number was typed; `smoke-delete.json` takes the rest to the end,
a starred pass's source and a landed image's 409 included.

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
another axis varies, however few of its images are loaded (a star each under
Favorites) - including a variable a user's own prompt grid happens to name
`variant` (see "The variable assignment is recorded"). It asks for the whole
run in one page (`sheetPage`, the status's `max_grid_images`) because a
matrix with a page boundary through it is a puzzle.

**Every sheet the page opens comes through `showSheetScope`** - a comparison,
a refine sweep, the Runs picker, a card's ⊞, both branches of Show run - and
once `setSheet` resolves (after `renderSheet`), if that is still the sheet
shown, the page scrolls the gallery head under the topbar (its
`scroll-margin-top`; smooth unless reduced motion is asked for). A finished
comparison's sheet used to open at y = 1650 on a 1305 px screen, under the
prompt check and the viewer, with nothing in view changed but a line in the
panel. The landing is the gallery head rather than the sheet so the search,
Favorites and Sheet - the sheet's own filters - stay in view; the viewer is
not hidden (a running job lands there) nor the check folded, both a scroll
up, and scrolling moves no focus. The Sheet button does not scroll: it is
pressed at the gallery head.

**The cells fit the screen's height as well as its width.** Columns are
`minmax(96px, var(--cell-max))`, and `--cell-max` is the window less what
sits round the grid at that landing - topbar, gallery head, controls, the
Fixed line, the note - less the column heads, shared between the rows and
times the aspect. `fitSheet` measures that chrome rather than guessing it, so
a tier that scales the controls needs nothing, and `100vh` follows the window
from there. A 4 x 3 at QHD went from 524 px cells in a 1,607 px sheet to
339 px cells on one screen. The grid scrolls in a box of its own, no taller
than the same space, and the heads stick to its top and left edges on the
page's background, over their whole row or column, so a matrix still past
the 96 px floor keeps its labels beside the rows being read; the page itself
never scrolls sideways. Heads are at the body-small size in `--text` - 11 px
grey over 500 px cells read as an afterthought - each label clamped to three
lines with the whole value as the heading's title. The heads' row and the row
heads' column, plus the gap, are the box's `scroll-padding` (`fitSheet`
measures both into `--sheet-head` and `--sheet-rowhead` on `.sheet`): a cell
the keyboard focused while the box was scrolled was lined up with the box's
edge, under the stuck heads, its top and its focus ring hidden. A cell drawn
wider than its 384 px thumbnail asks for the PNG (`srcset`, with `sizes` the
measured cell width; lazy loading still fetches only what is on screen): the
upscaled WEBP was soft exactly where a CFG or steps sweep differs. All of it
is measured, so it is measured again when the window is resized
(`refitSheet`, debounced, from app.js), and a row that now fits or no longer
does is laid out again: once per render, a QHD sheet resized to 1400 px kept
the `sizes` of 339 px cells drawn at 204 and its note under the window's edge.

**A long single row wraps.** With no row axis and more columns than fit at
96 px - a seeds-only run, which every Batch or Repeat over 1 is - the cells lay
out like the gallery (`minmax(var(--thumb-min), 1fr)`), each captioned with
its value, rather than a ribbon several screens wide; one that fits stays a
row under its heads. Only a sheet with no row axis wraps, so a matrix and the
variant rows are never reshaped.

**Around the matrix.** A Fixed line names the settings every image shares -
under Favorites, what the keepers have in common, which the tick otherwise
takes off every axis - in an element of its own, since the note is read back
exactly. A starred cell carries the card's ring and 28 px badge (a 12 px
glyph in a 690 px cell was not seen) - the card's own rules, which name
`.sheet-cell.fav` beside `.card.fav` so the two cannot drift - and F on a
focused cell stars it, one
listener on the grid rather than a button per cell; the lightbox still walks
the sheet in reading order. A refine pass says so (`⤢ 0.1`, top left) unless
the strength is already an axis or a pin. Where every image is a pass of one
source, the source leads as a column of its own, headed `Source #<id>`,
spanning the rows and first in what the lightbox walks: it is fetched by id
(`sheetSources`, alongside `learnSheetOrder`) and never a sheet record - no
axis value, no cell of `cellOf`, no count - so a strength of null still means
a direct render; a deleted source is said in the note. The axes gained
References (`#8 + #10`) and Adapter, null where none applied as the LoRA
weight is, so with and without a reference at one seed is three cells, not
one; with one adapter the LoRA heads name it. Images that pass the pins but
find their cell taken are counted, where "1 of 3 loaded" read as images still
on their way, and in two kinds. One that has the cell's image's settings
throughout (`sameSettings`: every axis there is, the size, the checkpoint and
its quantization, the refined source) is a repeat - "2 of 4 shown · 2 repeat a
shown image's settings": the same grid run again without Resume keeps its run
id, and a repeat-seed check is exactly what a sheet is read for. The rest
"share a cell with another image and differ in something the sheet does not
lay out" - the size, mostly. Size stays off the axes.

**The way back, and on.** Sheet off restores the search a page-opened sheet
replaced (`sheetReturn`) while the box still holds what the page wrote, so a
sweep's three images do not look like a gallery whose rest had gone; back to
back sheets keep the search from before the first, and a search typed since
stays. Over an empty search the Sheet button opens the newest run, saying so
in the note, where a gallery of several runs gave a wall of `none` pins; a
typed search is laid out as typed. A new pin takes the value most images
carry under the pins already held (`reconcileAxes`): the first value is
`none` for every variable of a mixed gallery and no image was `none` on all
of them, so not one cell filled. Over a full run the counts tie and the first
value still wins. A sheet of one run opened on `run:<id>` offers Resume this
run: the run's base seed from `/api/runs` into the Seed box and its size into
Width and Height, Resume ticked for this session only, the grid fold opened
and Run grid brought into view. Resume is scoped by the size, the checkpoint
and the quantization: filling only the seed, the panel's size after any other
render made the preview skip nothing and blame the grid, so the size is
filled too, and a run of another checkpoint (All models) offers no button.
The axes are not stored per run, so whether the panel still holds that grid
is the preview's to say, and a preview that would skip nothing says the grid
- its prompt, axes or other settings, or the server's quantization - is a
different one. Not on a comparison or a refine sweep, which the panel's grid
did not make.

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
compared through `backend.applied()` so the capability flags decide. Refine
cells compare their integer schedule start, not the raw strength, only in
this advisory; resume and run ids keep the requested strength. The
cells are still rendered as spelled - dropping them would leave a `{avoid}`
axis with one cell of three and a sheet unable to say why.

## The running batch on wide screens

At 2000 CSS pixels and wider, the results viewer's `many` tier (more than
four completed images) uses eight equal columns. The previous wrapping flex
row sized images by `20vh`, which fit only seven square results across a
2560 x 1440 desktop. Eight columns also fit QHD at 125% display scaling;
portrait and landscape images keep their aspect ratio within the same slots.
The live preview shares the columns and their height limit. Smaller windows
and the single-image and two-to-four-image tiers keep their existing layout.

## The settings panel's sticky top

The settings panel is `position: sticky`, and its top is kept by
`web/panel.js` (`panelPos`) and written inline over the stylesheet's
`--sticky-top`. While the whole panel fits the window it rests under the
topbar at `--sticky-top` (68 px), as it always has. While it does not - the
grid fold open, the last-run banner, a short window, QHD at 125% scaling
(2048 x 1040 CSS px, where even the idle panel runs past the window) - its
top lies anywhere from resting down to the window's height less the panel's
and a gutter, a negative number at which the panel's bottom holds a gutter
above the window's (the gutter is read off `main`'s padding, the one the page
ends with, and the resting top off the stylesheet with the inline top
cleared, so a tier that changes either needs nothing in the script). Within
that range it moves four ways.

- **With the user's scrolling** (`followScroll`), by as much as the page
  moves: down the gallery it goes up until its bottom - Generate, Pause and
  Cancel, the error line, Run grid - is in view and holds there; back up, it
  comes down until its top, the prompt, is in view. The first version held the
  bottom and nothing else (4cb5a76), and the prompt came back only within
  about 180 px of the top of the page (580 px with the fold open), however
  far up the user scrolled; the review of that commit found it, along with
  the hidden focus and the moving Generate below. Only a scroll of the
  user's moves the panel: the wheel, a touch, the scroll keys from outside any
  control, a press on the page's scrollbar (Chrome dispatches it to the root
  element) or the middle button open a window that each scroll event of the
  movement keeps open and `scrollend` or a quiet 400 ms closes. The browser
  scrolls on its own too, and following those slid the panel for nothing the
  user did: scroll anchoring holds the gallery still when the viewer above it
  empties at a run's start or fills as images land, so a lifted panel crept
  up the page as images arrived and slid down at a run's start - Generate
  from under the pointer that pressed it - and the page shortening under a
  reloaded gallery did the same. Space on a button starts a run rather than
  scrolling, which is why the keys count only from outside a control. The
  panel moves a frame behind the page, since it follows scroll events on the
  main thread; that was preferred to a lag-free scheme, which a sticky box
  cannot give: holding its top at rest while the page scrolls up needs room
  above its place in its containing block, and a margin there is taken out
  of that room.
- **With its height or the window's** (`placePanel`, from a ResizeObserver
  and `resize`), keeping the bottom where it was while the bottom was in view
  (held, or resting with room to spare), else the top. While the pointer is
  over the panel a change of its height moves nothing (`panelResized`) and
  the placement waits for the pointer to leave. A held panel otherwise grows
  upwards: the progress row, the queue note and the error line appear below
  Generate, Generate rose 40 px under the pointer that pressed it, and Pause
  and Cancel took its place - with Generate still live as "Generate (queue)",
  a second press meant to queue a run paused or cancelled the first. While
  the placement waits, the panel grows downwards instead: what appears under
  Generate is in view (Pause and Cancel at 2048 x 1040), and what goes below
  the window is the fold's summary - with the fold open, Run grid - until the
  pointer leaves or the user scrolls.
- **With focus** (`revealInPanel`, on `focusin`): a control focused from the
  keyboard or by the page - Shift+Tab, a finding clicked in the prompt check,
  Use and Apply, an example picked - out of sight above the window, under the
  topbar or below the window, has its block (its field, row or button)
  brought between where the first block sits while the panel rests and where
  the last sits while it is held, by moving the panel and not the page. The
  browser scrolls the page to a focused control before it dispatches the
  focus events, measuring a stuck panel as if it moved with the page, so that
  scroll moved the page and left the control hidden (WCAG 2.4.11). Its scroll
  event comes at the next frame, so the last position the panel saw is the
  one before the focus: the scroll is taken back, and made again only if
  moving the panel could not show the control - a panel still in the page's
  flow near its top, or pinned by its containing block at the end of the
  page. A press in the panel moves nothing, through to the click, where a
  label hands its control focus: what was pointed at is in view already, and
  moving the panel between press and release would move it from under the
  pointer.
- **With the grid fold opened by hand** (`revealGridRun`, below).

A help's tip opens under its badge, and over it (`placeTip`, `.help.up`)
where under would run past the window's bottom and over has more room: the
fold's last rows - Resume, Budget, the seed list - sit at the window's
bottom while the panel is held there, and their tips ran off the window,
which scrolling could not fix since the held panel does not move with it.

It replaces `@media (max-height: 900px) { .panel { position: static } }`
(496bfb1), written because everything past the fold - Generate, Cancel, the
error line, Run grid - was unreachable until the very end of the gallery
released the panel. That rule never applied: it and the `max-width: 900px`
query sat above the equally specific `.panel` rule, which came later and so
won, and the panel stayed sticky at 1400 x 900 and in one column below 901
px alike. 900 px was a guess besides: the 2026-10-04 UI review (U1) measured
the panel at 1489 px with the grid open in a 1305 px QHD window and at 1089
px idle in 1040 px, both past it, while the gallery's paging moves the end of
the page away as it is reached. The `max-width` query now sits below `.panel`
and really unpins the one-column page. No height query is kept as a fallback
for a page without scripts, which cannot generate anyway, and one below the
rule would override the measurement on every window under its height.
`overflow-y: auto` on the panel was rejected: the Examples menu, Animagine's
tag menu and the right-hand tips hang out past the panel's edge, and a scroll
box clips them. The panel stays sticky at z-index 10, so its tips still paint
over the cards.

Opening the grid fold by hand - a click on its summary, as Enter and Space
on it are - brings Run grid into view (`revealGridRun`): the panel is placed
for its new height first, since the ResizeObserver reports only after the
frame (and then under the pointer that opened it), Run grid is revealed as a
focused control would be, and where the panel is still in the page's flow
the page scrolls to it (`block: 'nearest'`, smooth unless reduced motion is
asked for). It listens for the click rather than the toggle event, which
`restorePanel` and `resetPanel` set off as well: a fold restored at load
scrolls nothing. `#btn-grid`'s `scroll-margin-bottom` is the gutter, the
panel's bottom padding and its border, so the page's scroll leaves Run grid
where a panel held by its bottom keeps it, with room for the preview's reply
to add a line. `tools/smoke-wide.json` checks all of it at 2560 x 1305,
2048 x 1040 at 125% and 1400 x 900, and the page's scale at each (the wide
tier's, below). The wide tier makes the panel taller and lets the prompt box
grow with its text, by as much as seven lines: that is a change of the
panel's height like any other, placed as above. On the FLUX fake with a
one-image last-run banner, at 2560 x 1305 the idle panel is 1,132 px of the
1,216 it can rest in, and still rests; at 2048 x 1040 it was already held by
its bottom (1,071 px), and the short-window tier keeps it a little shorter
than it was (1,034 px) while 70 px wider.

## Design tokens

`web/style.css` opens with tokens on `:root` - seven type sizes, the four
greys between `--text` and `--muted`, the spacing ladder, control heights,
radii and the layout's widths - under a table saying what each is for. They
came in on 2026-10-04 (step 5 of that day's UI review plan, U92) at the
values the page already had, checked pixel for pixel against the commit
before them at 1400 x 900, 2560 x 1305, 2048 x 1040 at 125% and 800 x 900,
and by every element's computed style in thirteen states. Those shots are
of freshly loaded pages, compared viewport by viewport. Rows of the panel
sit on half pixels (the size presets are 26.5 px tall), so after a scroll or
a change of the panel's top the browser can raster them a device pixel off
from a fresh page's, with every box where it was; and a full-page or element
capture lays the page out at a passing 1 x 1 viewport, where the one-column
rule now applies and the panel goes static, so such captures of the panel
are not comparable across the move of that rule below `.panel`. The point is
the steps after it: a wide tier or a short-window tier is one media query
overriding `:root`, not a restyle of every rule (the panel stays unpinned by
measurement, never by a window-height query; see the sticky top above). So
the type scale keeps the page's seven steps (10 to 16 px) rather than the
review prototype's five: the prototype's size names where a size has one use
(`--fs-xs` to `--fs-xl`), and role names, `--fs-label` and `--fs-body-sm`,
for the two steps it does not have. A value one rule alone uses - a hairline, a
pill's padding, a chip's state colours - stays a literal, and a glyph sized
to a box of its own (the ? badge, a finding's mark, the stars, the close and
arrow buttons) keeps its size, since scaling the type should not grow an
icon inside a fixed circle. A control's height is its token: the vertical
padding is worked out from it, `(height - line - border) / 2`, which is
exactly the 7 px a ghost button always had, so a tier that raises a height
or a type token gets that height without restating a padding. `--warn` and
`--star` are the same amber with different meanings; `--accent-strong`, the
primary buttons' fill, came in as the accent itself and `--line-strong`, the
controls' borders (inputs, selects, ghost buttons, presets, the ? badges), as
`--line`, so that the step after could change them alone for contrast (below),
leaving `--line` on the panel, cards, viewer and folds. `--topbar-h` is the topbar's
set height rather than a floor under its content, so the sticky top and the
check heading's scroll margin, which read it, read the real height whatever
a tier does to the buttons in it.

**The wide tier** (step 6 of the plan; U14, U15, U84, U96, U103, U90) is
`@media (min-width: 1800px)` overriding `:root`, and a second query for a
window under 1150 px tall inside it. At 2560 x 1305 the page had been the
1400 px page stretched: a 330 px panel (12.9% of the width) of 10-12 px text
and 19 px caption buttons, a five-line prompt box showing half of a
guide-length prompt, twelve 171 px cards a row - smaller than at 1400. The
type grows with the window: `--ramp` is `(100vw - 1800px) / 760`, a length
from 0 at 1800 px to 1 px at 2560, and each size is today's plus its steps
times it, capped at 2560's - body 14 to 15.5 px, captions 12 to 13.5, sub-labels,
hints and tips 11 to 13, badges 10 to 12. So nothing jumps at 1800, and QHD at
125% (2048 CSS px) gets a third of the increase: Windows already draws its
11 px as 13.75 device px, and the review measured that panel past its window
before any change (U15's verifier would have gated the type at 2200 px for
that reason; the ramp is the plan's answer instead). The layout steps at
1800: the panel `clamp(400px, 18vw, 480px)` (461 px at 2560, 400 at 2048),
the cards `clamp(200px, 9vw, 260px)` (eight a row at 2560, seven at 2048),
the search box to 480 px, tips to 320/380 px, small counters to 72 px, the
controls taller (fields 44 px, the primary 48) and the topbar with them. In
the wide tier the prompt box grows with what is typed, `field-sizing:
content` from five lines to twelve (eight under 1150 px), capped in lines
rather than in the window's height: a box 40% of the window tall would add
400 px to a panel already past the fold at 125% (U14's rule tension). The cap
is the growth's, not the user's: dragging the grip writes the box's height
inline, and the rule applies only to a box without one
(`#prompt:not([style*="height"])`), so a dragged box keeps the height it was
dragged to, as under 1800 px. The drag was the way round a long prompt before
the tier, and a grip that stopped at eight lines took it away at 125%. The
short-window tier keeps the type and the widths with compact controls of its
own - about today's heights rather than the wide tier's (a field 40 px, the
primary 44, the small and tiny buttons 28 and 21, two pixels over today's for
the larger type in them), today's panel padding and topbar, and a 12 px gap
where today's is 14 - so at 2048 x 1040 the panel is about as tall as before
while 70 px wider. Under 1800 px every token is today's, which is where the
smokes run and why their computed-style checks hold.

Some of the same change is not scale but contrast and focus, and applies at
every width: `--line-strong` is `#666680`, 3.33:1 on the
panel and 3.11:1 on the fields where `--line` was 1.24 and 1.15 (U24; the
review plan's `#62627a` is 2.92:1 on a field), `--accent-strong` is `#6a4cf0`,
white on it 5.4:1 where the accent gave 4.35 (U89, U92), placeholders are
`--muted` (5.16:1, U89), and a disabled box in one of the panel's field rows
(`.panel .sub > input`, `select`) is dimmed like a greyed one (U94) - not
every disabled input in the panel, since under 901 px the prompt check sits
there and its comparison's "original" box is disabled because it always
runs. `color-scheme: dark` on `:root` turns the scrollbars,
checkboxes, spinners and select lists dark and `accent-color` ticks the boxes
in the accent (U21); it also turns Chromium's `outline: auto` focus ring near
black on this background, so the same change gives every control the page's
own ring, `:focus-visible` at the lowest specificity (2 px of `--accent-2`,
10.4:1), which any control with a focus style of its own outranks. Checkboxes
had none at all: the text boxes' `outline: none` caught them and their border
does not render (U54). A text box's focus is two pixels of the accent, since
one pixel of it beside the stronger border was a 1.3:1 change. The LoRA select
is drawn as the text boxes are, by its id alone: a rule for every select would
restyle the comparison's selects in the panel under 901 px (U21). Danger red
on hover is a `.danger` class on Cancel, Cancel queued and Delete unstarred,
not every small ghost button (U91), and a tip lets clicks through
(`pointer-events: none`, U139): one pinned open by a click lay over the Refine
dialog's scale buttons and swallowed the clicks meant for them. A tip's line
breaks are kept only where its text has them (`.tip.pre`, set by the code
that writes it), not on every wide tip, which broke a tip written in the
markup wherever its source lines fell.

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
holds about 5,200-6,100 MiB of the card while Unreal Editor and the browsers
are open, and a run that lands within a gigabyte of the top spills into system
RAM (schnell's 2 MP read 35-45 s that way and 25 s after the baseline fell to
1,270 MiB). **Qwen-Image is in the catalogue too** (`QwenImagePipeline`, Krea
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
them. The helper drops under the prompt box, the box's width, over the
negative prompt and the sizes: hung from its button it covered 209 of the
box's 296 px, caret included, so its own "insert at the prompt caret" pointed
at a caret nobody could see, and a click in the box to move it closed the
helper. A click in the box now moves the caret and leaves the helper open,
and its second line names the place - "A tag goes after "absurdres", at the
end" - recomputed on every insert and caret move. It does not slot tags
itself: the documented order differs per checkpoint (Kohaku's rating is
last, Illustrious's third, WAI's folded into subject and scene), the page
has no order of its own, and Check's Use reordered places them. The text is
the chips' 12 px, and the window is its only height cap: at that size
Animagine's helper is 573 px and outgrew the 520 px it had, its closing note
behind a scrollbar. Opened from the keyboard it takes focus; Escape gives focus
back to its button. Typing in the box with the helper open brings the
suggestion list over it, and an Escape there closes that list alone (the
document's Escape skips one already taken); the next closes the helper. The
prompt remains visible, editable, persisted and recorded exactly as
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
merged into autocomplete and remain available offline, where they are the
whole list. They used to be merged ahead of every remote result, which made
"gre" offer `great score` before `green hair` and "lo" offer `low quality`
first - the item Enter and Tab take. `tags.rank_suggestions` orders the
list: an exact match, then names matching from their start, then at a later
word, then alias matches; within that the index's post-count order, a guide
tag going ahead of it only once the query covers half of it ("high s" is
heading for `high score`); and a tag of the guide's recommended negative
prompt is never first unless typed whole. The page selects the first item
only from three typed characters and when it starts with them, in the box
and in the chip editor alike, so Tab after "lo" moves on and Enter is a
line break; ArrowDown still selects. The list calls the guide's own tags
`guide`, as the chips do.
A green chip means a model tag
or a nonempty Danbooru tag at the cutoff; red means unknown, deprecated,
empty or too new, with the reason as a `code` beside the sentence
(`not_found`, `deprecated`, `empty`, `no_date`, `after_cutoff`): a tag too
new for the checkpoint shows ⏲ where a typo shows ?, red either way, since
the fixes differ, and matching the sentence would break with its wording.
An unavailable API makes ordinary tags unchecked, not invalid, and never
blocks generation: presence in the index is useful
evidence, not proof that a particular caption reached training. The tray
says which of three states it is in (`data-state`): checking (… on the
chips waiting, title "Checking Danbooru…"), checked, or unreachable - the
label amber, "Danbooru unreachable, not checked", never red, guide tags
still green and the rest marked –. Checking and unreachable used to look
the same, and the checking title claimed an outage. The chips ask about the
first 64 items, the request's own ceiling, and mark the rest unchecked
"past the 64-item lookup cap", the label saying "first 64 of N checked": a
65-item prompt was one refused request that left every chip, guide tags
included, waiting for good. What the index said is kept per checkpoint (a
cutoff and guide tags change the answers), so a pause in typing renders
every known chip at once and only a new item waits: every chip used to go
grey and dashed at each pause and back to green a moment later. The item
being typed - a keystroke or a deletion in it, not a whole-box write - has
no verdict while it is unknown (`.typing`, no mark) until a comma, the caret
leaving it, a blur or 1.5 s: the one red chip was the word still being
written. A known one shows at once. Sources,
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
chip centres; a context click on a chip inserts after it, as the Menu key on
a focused chip does. Its
visible listbox reuses the same cached tag-suggestion endpoint; a native
`datalist` did not reliably reopen when asynchronous results arrived. Click,
Enter or Tab accepts a suggestion (the first is selected as the box's is, from
three characters it starts with), while + commits literal text and Escape or ×
cancels, focus going back to the chip or beside the gap. A plain click in a
gap inserts there too, the pointer over a gap shows where with the drop's
accent bar, and a "+ tag" button after the last chip inserts at the end -
the visible way in for a pointer that does not right-click and for the
keyboard, where right-click had only a 10 px hint that moved with the last
chip. The hint is pinned to the tray's foot above the results. A gap for a
plain click and the marker is within a row of chips, from its first chip to
the last row's + tag, and within half the row gap above or below it: the
label before the chips and the hint under them are text, and a click on
"click a gap or + tag to insert" opened an editor in the middle of the
prompt. A right-click still takes the nearest place. A double click on a chip (Enter or F2 while it has focus) opens the
same editor in the chip's place, holding its text, so a typo is fixed where it
sits rather than by deleting the chip and inserting a new one; suggestions
start with the first keystroke, not with the opening, so Enter on an untouched
edit keeps the tag instead of taking the top match for it, and an unchanged
commit only closes. One editor is open at a time, and any prompt change from
outside it - the textarea, a drop, a restored panel - closes it, because its
index no longer names the item it opened on. Every edit goes through the same
preview, validation and panel-persistence path, so the visible textarea,
generated prompt and chips cannot become separate sources of truth.
`writePromptTags` writes the box through `writeBox` - analysis.js's
`replaceBoxText`, then no suggestion list or typing state over the new text -
so Ctrl+Z restores what the chips changed: assigning `.value` emptied the
undo history, and in Chromium a later Ctrl+Z could splice two prompts into
one, which the panel then saved. An accepted suggestion replaces the word
at the caret with an insert, one undo step from what was typed, and the
recommended negative prompt is written the same way. The write takes focus
to the prompt; it goes back to the chip the edit leaves in place (the moved
one, the one after a deleted one, the edited or inserted one) or to the
helper's button that made it, and every rebuild of the tray - a pending
render, then the lookup's answer - puts focus back on its chip until focus
leaves the tray, taking it only from `<body>`, where the rebuild left it:
one Alt+Arrow used to drop it there for good. A chip's visit to the prompt
is only in passing, so it moves nothing: the focus scrolls nothing, and
the panel's `revealInPanel` skips it as it skips a press in the panel - a
panel held by its bottom at 1400x900 jumped 62 px at each Alt+Arrow in the
other column. The helper's writes, whose focus does come back to the
panel, still reveal the box. The tray is one tab stop: the
chip last focused (or the first) takes Tab, arrows, Home and End walk the
chips, × is out of the tab order (Delete does the same), and "+ tag" is the
second stop. A chip's label says what it is and how to act on it; the tray
is a labelled group, not a live region - it holds every chip and is rebuilt
at each pause in typing - and a hidden polite summary says "N prompt tags:
K known, U unknown" once a lookup answers, not counting the item still being
typed, as the label does not. A focused chip draws a 2 px teal ring of its
own after every state rule, at full opacity past the cut.
Chip and suggestion text is 12 px. A valid chip uses high-contrast light text
on neutral charcoal, with green confined to its checkmark and outline; unknown
chips keep the corresponding red mark/outline instead of colour-on-colour text.
A valid chip's left edge carries the category the lookup returned, in
Danbooru's colours (general blue, character green, copyright purple, artist
red, meta yellow) and teal for the guide's own tags: what a tag is, never an
order slot, which only the guide's `analysis.order` and Check may place. A
later copy of a tag (the same canonical name) has its text underlined in
amber dots, "repeated", as Check's heuristic says - on the text, not the
outline it first had, which overrode the focus ring, so the arrows landing
on a repeat showed nothing. An unknown chip's reason, with
"generation is still allowed", is in one shared tip outside the tray,
shown under the chip as it is hovered or focused and named by its
`aria-describedby`, so the chip's text stays the tag's own.

**The chips show where the encoders stop reading.** On these SDXL
checkpoints both CLIP tokenizers stop at 77 tokens, and the guide's own
quality suffix, written last, is the first thing a long list loses - while
it showed four green ticks. `POST /api/prompt/measure` is Check's encoder
layer alone (`prompt_encoding.measure`, no rules, no tag lookup: a Check's
64 category lookups at every pause would be a second lookup beside the
chips'), on a daemon thread like Check, never `ensure_loaded` or the gate.
The tray asks it 600 ms after typing stops, with its own epoch. A chip past
the earliest cut of an encoder that reads the words (not FLUX.1's pooled
CLIP-L) is dimmed and dashed (`.past-cut`), an amber bar where reading
stops; the one the cut splits keeps a dashed far edge, "partly read". The
count is in the tray's label - "101/75 tokens", amber when over - with no
element of its own, so every child of the tray is a chip, the editor or
"+ tag". An encoder left uncounted (a tokenizer still loading past
`LOAD_WAIT`, no network for a first fetch) is "cut unknown", never "no
cut". A Check of the same text serves until the count lands, and the last
count serves while the text up to its cut is unchanged, so a tag typed at
the end of a long list is past the cut at once. A first measure on a cold
server may fetch the checkpoint's tokenizer files before anyone presses
Check; that is the tray's to do, not `/api/status`'s, which still loads
none.

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
a partial tree. That temporary directory removes itself on the way out, but
not after a hard kill, nor on Windows whenever its cleanup meets a file still
open or still being deleted. Each such attempt could leave a partial tree as
large as the converted one (about 5 GB for Anima to 15 GB for NewBie), and
the cleanup's error was raised over the conversion's own, so a failed
conversion reported WinError 32 instead of why it failed. The work directory
is created with `ignore_cleanup_errors=True`, so the caller hears the
conversion's own error, and `single_file.sweep_work`, which all four
converters call first under the same lock, removes what is left on the next
load of that checkpoint, converted or not. A conversion holds that lock
until it is done with its work directory, so anything the sweep finds is
stale. It takes only the exact tree name, a hyphen and `tempfile`'s eight
characters (`[a-z0-9_]`), and only a real directory (not a link or junction)
holding nothing but `tree`: another checkpoint's work, a published tree and
a user's copy named like one are left alone. Removal is best-effort
(`ignore_errors`, no network); a directory the sweep cannot list is kept,
one it cannot remove yet is logged as such and tried again on the next load,
and nothing in the sweep fails a load (2026-09-28 review, F39). The same tree
supports FP8, int8 and none; Nunchaku is refused before downloading. Both
original and converted weights remain on disk, about 14 GB total. Only the
loader's local settings copy sees the cache path: status, profiles, cell keys
and PNGs retain the official repo id.

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
The two are separate sentences in the prose: the style one is the app's
reading of "no default style" and is `unsourced` for the critic
(2026-09-28), while the composition one is the v0.1 card's ("We suggest
using suitable composition tags like...") and links it.
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

The existing lock, temporary tree, recipe marker, atomic rename and the
sweep of an interrupted conversion's work directory protect the conversion
cache. WAI's distinct source and digest produce a separate cache key; the
existing Illustrious recipe remains unchanged. The cached tree feeds normal
per-component FP8, int8 or bfloat16 loading, and Nunchaku is refused before
conversion. The original and converted weights occupy about 14 GB together.
Only the loader sees the cache path: status, saved metadata and resume
identity retain `LyliaEngine/waiIllustriousSDXL_v170`.

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
examples. Generate renders directly at the requested size. Since 2026-10-03
the Refine pass (its own section above) is the nearest thing here to that
hires pass - 1.5x and a strength in the author's range, measured on this
checkpoint - with a Lanczos resize where the author uses the upscaling
model, which the size tooltip says. The fake-server smoke is `tools/smoke-wai.json`
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
process lock, temporary directory, final rename and the sweep of an
interrupted conversion's work directory (`single_file.sweep_work`) provide
the same atomic cache contract as Illustrious and Anima. FP8/int8
quantization applies only after conversion through the normal per-library
`quant_mapping`; full bfloat16 uses the same tree. Nunchaku is refused
before downloads.

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
schnell in BFL's reference, not in this app; schnell's guide says both),
Qwen-Image's encoder is prompted to *describe the image by detailing the
color, shape, size, texture, quantity, text, spatial relationships*, ERNIE
ships a rewriter this backend leaves out and so expects the expanded form
typed in - and about sizes: FLUX.1 is documented
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
on a warning. The preset buttons use that same tolerance and are marked
(`.recommended`) when the full size they would choose is inside the range and,
when a guide publishes a finite `aspects` list, uses one of those framings:
aspect buttons are judged at the current megapixel tier, and tier buttons at
the current aspect. The VRAM warning stays and stacks after it.

**The selection reads first; the range is a mark.** A marked preset used to be
filled green, and on Turbo all 17 buttons were, on FLUX 15 - the largest
saturated block in the panel - while the chosen one differed from its
neighbours by a 1.4:1 fill (the 2026-10-04 UI review, U81). Now the chosen
preset is the accent's fill with white text and `aria-pressed="true"`, and a
marked one has brighter text, a green edge and a dot in its corner, absolutely
placed so it widens nothing. A strip where every button is marked
(`all-guided`: FLUX's aspects at 1 MP, everything on Turbo) drops the marks,
which say nothing there; the size line still quotes the range. The classes and
`presetIsInGuide` are unchanged. Each strip is a `role="group"` with one tab
stop, the chosen preset (or the first where the size is none of them), and
the arrow keys, Home and End move focus along `strip.querySelectorAll('button')`
- the pairs are nested - without choosing: a choice persists the panel and asks
for a preview, so it waits for Enter, Space or a click. The 17 size stops had
put Generate 36 Tab presses from the prompt (U105).

**A tag checkpoint's empty prompt shows tags.** The static placeholder is
prose, which failed Animagine's own style check. `applyGuide` writes the start
of the authors' first example, cut at the last comma within 60 characters and
ended with ", …", where the guide's `tag_helper` is Danbooru vocabulary and
not `hybrid_prompt` (Kohaku, Illustrious and WAI document prose as well), and
puts the markup's placeholder back elsewhere (U133).

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
the first click. A pick writes the box through `writeBox`, as typing would,
so Ctrl+Z brings back what it held, and tells the preview, the chips and the
remembered panel the way `choose` does; over a prompt of the user's own it
also shows "An example replaced your prompt." with an Undo, until the box
holds anything else or 15 s pass. On a two-column page it lies over the next
field's caption, out of the sticky panel's flow, so nothing moves; on one
column the chips come straight under the box, and lying over their first
row it took their clicks, so there it takes a line of its own - and so it
does while the negative box is greyed, when that caption says why and holds
the button that wakes it: lying over it, the note left "Negative prompt
(ne" of the sentence. A Reuse takes it, and it takes a Reuse's Undo: one
Undo on offer at a time ("Per-control defaults and Reuse"). Assigning
`.value` had emptied the undo history: the menu opens over
the prompt, and one stray click replaced minutes of work for good, the
panel saving it at once. It is a `role=menu` and keeps that promise: ArrowDown,
Enter or Space on the button opens it on its first item (its items are no
Tab stops, so a Tab from the button went on to Tags and closed it unread),
the arrows walk it round the ends,
Home and End jump, Escape hands focus back to the button, and focus leaving
it closes it - it stayed open over the prompt after a Tab. Where the
checkpoint takes reference images and none is attached, an example that
edits a picture "this" prompt takes to be there (`prompt_rules.EDIT_PROMPT`,
copied as `EDIT_PROMPT` in app.js and held equal by a test) says "needs a
reference image" beside the authors' words, which stay as written. The
examples are verbatim from the authors - a model card's demo
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
("overcomes the oversaturated 'AI look'") and
<https://www.krea.ai/blog/flux-krea-open-source-release> ("opinionated",
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
generalisation) and "rich, carefully constructed captions that describe
images with dense visual detail". The "validated at 1 MP only" claim was
**not** found for Turbo in any official source; the guide says 1k-2k
documented and 1024 px the top pretraining stage. Qwen-Image: the README's
positive magic and resolution table (<https://github.com/QwenLM/Qwen-Image>;
the HF copy's 1472x1140 is not a multiple of 16 and GitHub's 1104 is used),
the report's 1328x1328 final stage (<https://arxiv.org/abs/2508.02324>). Z-Image:
<https://huggingface.co/Tongyi-MAI/Z-Image-Turbo> ("512x512 to 2048x2048
total pixel area", bilingual text), the maintainers' "works best with long
and detailed prompts" and "does not use negative prompts at all"
(<https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8>), the
report's "up to the 1k-1.5k range" (<https://arxiv.org/abs/2511.22699>).
The undistilled Z-Image: <https://huggingface.co/Tongyi-MAI/Z-Image>
(guidance 3.0-5.0, 28-50 steps, and a negative prompt "Optional, but would
be powerful when you want to remove some unwanted content"; checked
2026-09-28). ERNIE-Image: the card's seven 1 MP shapes and 8 steps at
guidance 1 (<https://huggingface.co/baidu/ERNIE-Image-Turbo>), the rewriter's
system prompt in `pe/chat_template.jinja` (expand a brief description into a
"content-rich, detail-sufficient visual description"), the report's
Ministral-3 encoder and 256/512/1024 stages (<https://arxiv.org/abs/2605.25347>);
its tokenizer's 2048 is the config's, not a card statement
(`model_max_length` in each repo's `tokenizer/tokenizer_config.json`, and the
report's own words are "we adopt Ministral-3(3B) ... as the text encoder";
checked 2026-09-28). The undistilled ERNIE-Image:
<https://huggingface.co/baidu/ERNIE-Image> (guidance 4.0, 50 steps; checked
2026-09-28).

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
   `is_reorder` (below), so the reorder the user writes is the only
   one there is.

**Check works cold.** It takes the body Generate would send
(`PromptAnalysisRequest`), because half of what a checker says depends on
the rest of it: a negative prompt at CFG 0 on Krea 2 Turbo, or without True
CFG on FLUX.1, never runs, and that is `applied()`'s answer, which
`backend.applied()` gives from the profile before a load. It runs off the
loop, on a daemon thread of its own as the model load does (`_detached`),
never calls `ensure_loaded` and never takes the gate, so it answers in
a moment on a cold server and beside a running night - a first use's tokenizer
download holds it `LOAD_WAIT` at most (below). The negative prompt is
measured and checked only as `applied()` runs it: a budget for a negative the
pipeline drops is a number about nothing, and a negation in the prompt is a
warning only where no negative prompt runs. `POST /api/prompt/measure` is its
encoder layer alone, for the tag chips' count and cut (under "Animagine XL
4.0"): the same `_detached` thread, no rules and no tag lookup.

**The encoder layer measures the string the pipeline builds.** No pipeline
here parses A1111 or ComfyUI weighting - diffusers has no prompt weighting,
and compel and sd_embed are not installed - and every overflow is cut silently
from the right: the SDXL negative prompt, every Qwen-encoded pipeline, ERNIE
and Anima never say so, and the others' warnings go to the server console.
`prompt_encoding` loads each checkpoint's own tokenizers on the CPU and counts
over the exact string: Krea 2 and Qwen-Image's 34-token system turn, Z-Image's
and klein's chat template with the pipeline's `enable_thinking` (on and off
respectively), NetaYume's system instruction and `<Prompt Start>` (the
negative gets none), and a boundary merge counted the way the tokenizer counts
it. The budgets that come out, which builder A matched against each pipeline's
own truncating tokenizer call: FLUX.1 CLIP 75 (one pooled vector - T5 reads
on, so its cut is info) and T5 511; Krea 2 507, its suffix appended after
truncation and never cut; Qwen-Image 512 after its 34 system tokens, the
template tail lost past 507; Z-Image and Z-Anime 509 (504 before the assistant
turn goes); klein 509 (500); NetaYume about 237 (255 for the negative, which
is sent bare, so its row carries no `role`; until the 2026-09-28 review it
said the negative shared the instruction's budget); NewBie Gemma 3 511 and
Jina 2046; Anima Qwen 512 and T5 ids 511; ERNIE 2047; SDXL 75 per CLIP. Past
the clean budget Z-Image, klein and Qwen-Image encode an unterminated user
turn (`encoder.template_tail`). The cut finding quotes the words from the cut
on, starting at the beginning of the word the cut splits. Chinese and Japanese
put no spaces between words, and until the 2026-09-28 review that walk back
ran to the prompt's first character, so the message named the opening as where
reading stops. Now Han and kana, and their 、，。, stand alone; Hangul does
not, since Korean spaces its words. A walk back longer than half the
40-character snippet (a URL, Thai, an unspaced Hangul run) quotes from the cut
instead. Every template string, length and thinking switch copied into the
module is held equal to the installed diffusers source - and to this repo's
Anima and NewBie pipelines - by reading those files in
`tests/test_prompt_encoding.py`, so a diffusers upgrade that moves one fails a
test rather than the counts. The table is first-match on the id like
`_PROFILES`, exact ids first, reusing the loaders' pinned constants; its
tokens are narrower than the profiles' (`flux.1`, so FLUX.2 klein never
reaches FLUX.1's CLIP and T5). Control tokens are found by tokenizing, from
each tokenizer's own special tokens and the markers its chat template uses,
not from a list: Gemma 2's and 3's `<s>` and `</s>` turned out to be ordinary
HTML-strikethrough tokens beside `<b>` and `<table>`, which the research's
table had wrong, and `<think>` counts only where the template uses it (klein,
Z-Image, Z-Anime). A typed control token is compared with the stored one
through the tokenizer's own normalizer. CLIP lowercases before it matches its
added tokens, so `<|EndOfText|>` is its EOS; until the 2026-09-28 review that
typing read as an unknown character, because EOS is also CLIP's unk id, and
`<|STARTOFTEXT|>` went unflagged. FLUX.1-Krea-dev's CLIP-L stores its EOS with
`normalized=False`, so there the same typing is text and is not flagged.

**A first load is a download, and nothing waits on it for long.** Tokenizers
load from a directory - converted tree, then the Hub cache, then the Hub - not
a Hub id, because transformers 5.16 loading a fully cached pinned tokenizer by
id with `local_files_only` failed; the download patterns are tokenizer files
only (`*.json`, `*.txt`, `*.model`, `*.jinja`), since NewBie's tokenizer
folder sits beside 8 GB of weights. A load runs on a daemon thread of its own
and is kept only after it succeeds, so a gated repo or a first use offline is
retried next time. Until the 2026-09-28 review a load was cached under the
lock `unload()` takes (`translation.py`'s pattern), and the Hub fetch inside
it held that lock: `snapshot_download`'s `repo_info` and `list_repo_tree`, on
huggingface_hub's shared httpx client with `timeout=None`. So Unload and
shutdown waited out a first Check's download, forever on a stalled connection,
while `/api/status` said ready over no weights. Now the lock guards only the
cache and the loads in flight (`_loading`), and concurrent checks share one
load. `measure` starts every encoder's load before waiting on any, and waits
`LOAD_WAIT` (30 s) for all of them at most; past that the encoder is reported
as still loading while the load goes on, and the next check finds it cached.
`unload()` forgets the loads in flight: one that lands afterwards is handed to
the check waiting on it but not cached, and the next check starts a fresh
load, so Unload is also how a stalled fetch gets retried. The bound is a wait,
not an HTTP timeout: the only way to give that client a timeout is
`set_client_factory`, which is process-wide and would bound the weight
downloads too, so a fetch stalled for good keeps its thread until the process
exits. The Unload route says "unloaded" as soon as `try_unload` succeeds,
before the translator, critic and tokenizers let go. `measure` never raises
and says what it could not check in `encoders_detail`, each unchecked encoder
under its own reason, so "check again in a moment" never speaks for a gated or
broken tokenizer beside it. It did raise once: `json.loads` on 3,000 nested
`[` throws `RecursionError`, well inside the 4,000-character limit -
`_is_json`, `detect_style` and `is_reorder` now catch it. No status route
loads a tokenizer, because a first load can be a download; Unload and shutdown
drop them with the pipeline.

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
unavailable - costs the order check, never the answer, and says so
(`categories_detail` and `order_detail`, below). `reorder_tags` moves
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

**A one-click quote only where the lettering's end is sure.** The 2026-09-28
review found the quote rule's one-click fix quoting description as lettering,
after which the re-check came back clean: "with the words printed across the
chest" quoted "printed", "a book of spells glowing" quoted "glowing", and "a
newspaper reading without glasses by the fire" hid its negation as lettering.
The first word after a trigger now cannot be one that describes the text
(`_DESCRIBES`): a participle such as printed or carved, a preposition such as
into, about, by or without, an article or a possessive, or an -ly adverb
leading into one - in lowercase only, so "reads Into the Woods" is still a
title. A trigger that can be a noun (an -ing form, or "spells") is one after
"of", "with", an article or a possessive, unless a capital follows
(`_NOUN_USE`: "a book of spells" and "with reading glasses" are nouns, "a sign
above her reading OPEN" is lettering); a finite reads, says or said never is a
noun, because "her" is also an object: "a sign above her reads NO ENTRY" is
lettering, not an absence. Lowercase words after a participle trigger (reading,
saying, spelling) are not exact, because the participle may belong to any
earlier noun ("a boy on a box reading comics"): the finding stays, the Apply
goes. Lowercase lettering after a finite verb keeps its one-click quote where a
stop word ends it ("a wooden sign that says open above the bakery door", which
the analysis smoke applies). Making every lowercase target inexact, as the
review first suggested, would have taken that away, and requiring the bearer
directly before the verb would have lost "a shop sign in the window says
CLOSED". The price is accepted: lowercase lettering that opens with a
describing word - "a sign that says by appointment only", "says the end", a
clause-final "says daily." - now gets no finding at all, where it was quoted
rightly before, and "says without limits" reads as an absence instead.
Capitalised ("says By Appointment Only") it is
still lettering. Phase-4 re-verification found the noun triggers still
quoting description in one click, since the describing words are a closed
list: "with the words shimmering across the chest" quoted "shimmering", and
blinking, hanging, dripping, scratched, piped and "set in bold type" went the
same way. The reason for sparing lowercase targets was "says open", a finite
verb, and it does not reach "with the words". After a noun trigger ("with the
words", "the text", "the slogan", "the headline" and the rest) only words in
capitals now get the Apply ("with the words HAPPY BIRTHDAY", "2024", "FISH and
CHIPS"). Anything else keeps the finding and loses the one click, as the
review asked: "Happy Birthday" and "coming soon" there are likely lettering,
but "set in bold type" has the same shape. A lowercase -ing or -ed word with a
preposition of place after it ("shimmering across", "piped in", "dripping
down", "spelled out") places the words rather than being them, so it is no
lettering at all, and the critic flags a rewrite that quotes it. The price,
like "says by appointment only", is that lowercase lettering of that shape
after a noun trigger ("with the words dancing in the rain") gets no finding
unless it is quoted or in capitals. The same re-verification found a negation
hidden further into a participle's lowercase words: "a boy on a box reading
comics without pictures" read "comics without pictures" as lettering. Those
words now stop at without, never, no or not, and the negation rule has the
rest. A review of that stop found it cutting lettering that opens with a
negation or holds "do not": "a red sign reading do not enter" became "do",
"please do not touch" "please do" and "no pain no gain" "no pain", and nothing
warned, since the negation began inside the words it had stopped. So the words
stop only where the negation rule's own reading finds an absence starting at
that word (`_absence_at`) - not inside a negation that starts earlier, nor at
one with nothing after it ("now or never") - and those are lettering whole
again. The cost is lettering that holds a negation further in ("a banner
reading love without limits", "a sign reading dogs not allowed"), which warns
as an absence until it is quoted or in capitals. That stop is the rules' own:
the lettering may hold the negation, so the critic should still take "love
without limits" whole as text to render and not flag a rewrite quoting it,
which is the fix the quote rule's finding asks for. A first fix left the stop
in `rendered_targets` with `ends` False, a guess the lettering may run on
past, and the review's final completeness check found the critic then
reading on to the clause's end: "a banner reading love without limits above
the stage" kept "love without limits above the stage" as text to render and
let a rewrite quote it, description passing as lettering. `rendered_targets`
now reads the words on through the negation to where they would stop
without it - "love without limits", as "love forever" stops at "above" - with
`ends` True, so what follows is description, as after any participle's
words. The quote and negation rules keep the stop at the negation. The
review of that fix found the critic then taking the negation's own words for
lettering: a rewrite quoting only "without pictures" in "a boy on a box
reading comics without pictures" - the absence the negation rule warns on -
passed as a piece of "comics without pictures", where it had been flagged
while the lettering stopped at the negation. Each target now carries
`head`, the rules' stop (its `end` wherever there is none), and the critic
takes as lettering only words that start before it, as its run-on past a
guess always did: "comics", "comics without" and "comics without pictures"
pass, and "without pictures" alone is flagged. That review also found the
stop still cutting "now or never" wherever description followed it: "a
poster reading now or never in bold type" quoted "now or" and warned on
"never in bold type" as an absence, and "better late than never on the
door" went the same way. A negation right after "or" or "than" closes the
phrase ("now or never", "better late than never", "ready or not", "yes or
no"), so it stops nothing either; a place word after the negation would not
have told them apart, since "comics not in color" are black and white. The
2026-09-28 review also found the negation rule warning on "a cat trying to
avoid the rain" and "the film Dr. No". "avoiding" is no longer a head, and
"avoid" or a capital "No" asks for nothing mid-clause: right after a word
("trying to avoid", "the film No") or an abbreviation that goes before a
name (Dr., Mr., Mrs., Ms., Mt., Prof.), spaced or not ("Dr. No", "Dr.No").
One that comes after a name ends a sentence as often: with St., Jr. and Sr.
on the list too, the final completeness check found "a photo taken on Main
St. No cars." and "Martin Luther King Jr. No crowds." silent, so those now
open a clause, like any other full stop.
Anything else opens a clause - the text's start, a line break, a full stop
(after a year, a short word or "a.m." too), a spaced dash, a bullet, a bar or
a slash - and so do "please" and "also". A first fix also read a capital "No"
before any title-case word as a name. Phase-4 re-verification found that
silencing real absences: "Kyoto. No English text.", "No Chinese characters",
"No Apple logo", "No People on the beach". A clause's own "No" is capitalised
by grammar whatever follows it, so that reading is gone. A capital "No" is a
name only mid-clause ("a poster for No Country for Old Men", "a No Smoking
sign"). The accepted cost is a title that opens a clause: "No Country for Old
Men poster on a wall" warns as an absence, its span taking in the whole name.
And a negated phrase is no longer cut inside a run of capitalised words
("without ropes on El Capitan", not "... on El").

English lettering can also continue through a word normally read as a
location or through punctuation. A capitalized stop word in `MADE IN JAPAN`
or `OPEN AT NOON`, marks within `9:00`, `$9.99` or `www.example.com`, and a
comma before another capital in `HELLO, WORLD` now make the end uncertain.
Check asks for explicit quotes and offers no Apply, instead of quoting only
`MADE`, `9` or `$9`. `rendered_targets` carries that uncertainty to the critic.
Ordinary lowercase location descriptions and terminal sentence punctuation
still give the existing one-click quote.

**Chinese lettering has no spaces to end it.** The same review found the
Chinese triggers (写着, 印有 and the rest) taking up to twelve characters of
whatever followed as exact: 一块写着欢迎光临的木质招牌挂在门口 quoted the sign
and the door along with the words, and split 门口. A Chinese target is now
exact only where whitespace, punctuation, a quotation mark, a bracket or the
end of the text ends it - not an ASCII mark with a letter or digit on both
sides (营业时间9:00-21:00, 3.14, www.shop.com), nor a space inside Latin
lettering (写着OPEN NOW的). A target stopped at 的 or 之 is a guess and gets
no Apply, unless it is all Latin letters and digits, where the change of
script ends it (写着OPEN的招牌); so is one stopped by the twelve-character cap
with more text after. A title in 《》 is the lettering, marks and all. With no
word segmentation, a run that punctuation ends could still hold description:
写着营业中挂在门口， quoted 营业中挂在门口 in one click, and 写有欢迎字样，
quoted 字样 along with 欢迎. 字样, or a count of the characters (两个大字),
now ends the lettering for certain, so 写有欢迎字样 becomes `写有"欢迎"字样`
in one click; a 等 before either is a guess, "and so on" in 欢迎光临等字样 but
the end of the word in 平等字样, where a first draft quoted "平". A verb that
puts the sign somewhere, followed by 在 or 于 (挂在, 贴在, 悬挂在, 站在, 位于
and the like), ends it as a guess, the way 的 does: no Apply, and nothing past
it is read as lettering. The exception is a verb the sign itself asks for,
with a request or a prohibition right before it - an adverb such as 随意 may
stand between them, or 将 or 把 with one or two characters (请站在黄线外,
禁止停放在此, 请勿随意停放在此, 请将车辆停放在指定区域). A first draft
exempted only a prohibition up to two characters back, and cut 请站在黄线外 to
请; a request anywhere earlier in the run would have quoted 请勿吸烟挂在墙上
and 请把门关好挂在门上 whole. 在 alone never ends it, so 正在营业, 现在 and
在线 keep their Apply. Two gaps are left: a description that starts with a
bare 在 (写着营业中在门口) keeps a wrong Apply, and a slogan with a placing
verb of its own and no request before it (我们站在一起) loses its Apply. Every
target, English or Chinese, carries two flags: `exact` says whether the Apply
is offered, and `ends` whether what follows is description (False where the
lettering may run on past it), which `rendered_targets` hands the critic. Past
的 or a placing verb, or after a participle, the Apply is withheld but what
follows is still description, so the critic's run-on fallback does not take
it. Where a negation stopped a participle's words, `rendered_targets` reads
them on through it to their ordinary stop first, and what follows that is
description too; its `head` keeps the stop, before which lettering starts.
Words not in capitals after a noun trigger lose the Apply and keep the
`ends` their stop gave them.

**A stretch of lettering is one item, and no rule rewrites it.** The
2026-09-28 review found three rules rewriting lettering in one click -
repeated items, Anima's lowercase and the underscore rule - since each rewrote
a comma item whole, lettering included, and a comma inside quoted lettering
splits it into items: a quoted `"Ho, ho, ho, Merry Christmas"` lost a "ho",
`"HELLO"` became `"hello"` and `"user_name"` became `"user name"`. Its
follow-up found four more failures of the chips' split, which cuts
`"hello, world"` in two at its comma. The order
rule placed the halves apart (`1girl, world", holding sign, "hello, smile`); a
rating or artist rule rewrote a piece of a quotation that happened to be a tag
(`"hi, safe, bye"` became `"hi, general, bye"`, or `"@nnn yryr"`); a tag
inside lettering counted as present (a quality tier, the recommended finish,
NewBie's `2girls`); and an inserted or dropped item shifted `, ` into a
quotation's `"hi,there"`. A stretch of lettering - quoted, or unquoted text to
render such as a 《title》 - is now one item with its commas (`_units`), to
every rule that reads the prompt's items and to `reorder_tags`. It is matched
as no tag (`_Context.norms` is None for it), the tag index is never asked
about it, and it is placed by its shape alone: prose-shaped, it goes in the
general slot whether or not the index was asked about anything else. The
lowercase and underscore rules leave the lettering inside an item as written
(`_Context.segments`), and its words do not make a tag beside it prose for the
underscore rule. A "，" inside the lettering splits nothing for the order
check; one between it and a tag (`"hi"，1girl`) still stands the order down,
since that tag would sit inside the lettering's item where nothing places it
(the follow-up's first draft blanked the lettering to spaces and read such a
prompt as placed). Standing the order down wherever a comma fell inside
lettering was the alternative, and would have left a tag prompt with a quoted
sign unchecked; as one item the prompt is placed, and `order_checked` stays
true to what the rule did. A move that would still change what `QUOTED` reads
is not offered - `reorder_tags` is None and the order finding has no Apply: a
stray inch mark (`12" figure`) moved ahead of a quotation would pair with its
opening mark. Two such marks that `QUOTED` reads as one quotation
(`5" doll, 1girl, 7" doll`) are lettering to every rule, 1girl included -
`QUOTED`'s reading, shared with the critic and the encoder layer. And FLUX.1's
style rule, which wants three items before it calls a prompt a tag list,
counts `1girl, "hi, there"` as two. The review also found NetaYume's
system-prefix rule locating the instruction in `text.lower()`, whose offsets
are not the text's: a leading "İ" lowers to two characters, and the Apply cut
into the tags. The rule now searches the same-length `ctx.bare`, with
lettering blanked, case-insensitively. A sign that explicitly says
`"<Prompt Start>"`, or quotes the system instruction itself, therefore keeps
its inscription; a real duplicate after it is still found. Apply tidies only
the cut's seam - one space, and one comma where there was one on either side -
instead of collapsing every run of whitespace in the prompt.

**The order is shown as checked only where it was.** Only a tag
checkpoint's order has slots a comma item can take (`_OrderSpec`); FLUX.1's
"subject first", klein's four steps and NewBie's blocks are documented and
shown, but nothing reads a sentence's order. The final review found the
panel reading as a pass on the default checkpoint - a prose prompt with its
subject last and one with it first got identical panels, "nothing to flag"
beside the chain. `PromptAnalysis.order_checked` says whether the rules held
the text to the order at all, and the page says "Not checked" under the
chain and, where nothing is flagged, "order not checked" in the summary. A
pattern check for a prose
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

**Where the rules placed nothing, or not every tag, the page says so.** The
2026-09-28 review found a list written one tag per line, or with fullwidth
commas, reading as tags while nothing in it was placed: `detect_style` splits
at commas, line breaks and "，", the chips and the order rule (`comma_items`)
at commas alone. The order still showed as checked, and the quality rules
called a present `masterpiece` or `best quality` missing (Animagine's suffix
rule appended it again). `_placeable` is now the one answer `order_checked`,
`reorder_tags` and the order rule share: a guide with slots, not XML or JSON,
at most 256 comma items, and no comma item that a line break or a fullwidth
comma splits (a stretch of lettering is one item with its commas, and one
piece of it). Prose written over several lines on a tag checkpoint is
therefore "Not checked" too, and `test_an_order_not_checked_is_never_flagged`
holds, on every catalogue checkpoint, that an order not checked is never
flagged. `_suffix` and `_quality_tier` count a tag on its own line, or after
"，", as present. The review also found the panel reading as a pass offline.
`_tag_categories` returned None for a failed lookup just as for nothing to
ask, and never asked about items past the chips' 64, so an out-of-order tag
prompt and an in-order one got identical answers with `order_checked` true.
The rules still place only what the guide's groups know, and a category is
never guessed; what changed is that `PromptAnalysis.categories_detail` says
why categories are missing or partial: the index unreachable (a connection
error, the cause under `TagLookupUnavailable`), no usable reply from it (an
HTTP error status, a cut or unreadable body), or the cap. The page then said
"Partly checked:" and that reason under the chain, and "order partly checked"
in the summary. Nor is a partly placed prompt's "nothing to move" a verdict,
so the server keeps a critic's reordering there that the order rule, placing
with the same categories, finds nothing wrong with (the critic's paragraph
below). The server reads "partly placed" off the placement itself
(`places_every_item`), so it also covers a tag the index answered for and did
not know, and a clause of a prose prompt a name could still be, which is never
looked up. Phase-4 re-verification of the review's fixes found the page
still keyed on `categories_detail`: for an OC's name the index did not know
(`zzkx oc, green hair` on Animagine) or a prose prompt's clauses, the Order
section drew the order as fully checked while the server, and the critic
block's "though it could not place every tag", treated it as partly placed,
and an out-of-order prompt and an in-order one again got the same "checked"
answer - F12's class. There is now one rule. `_analyse` computes the
placement once, `prompt_rules.unplaced_items` with the categories the rules
placed with (`places_every_item` is its emptiness, the critic's
`placed_partly`), and `PromptAnalysis.order_detail` is set exactly where
`order_checked` and some item went unplaced, naming the items (three, then
a count) under the reason each has no category: the lookup's own reason
(`categories_detail`, kept word for word), a tag the index does not know, a
prose prompt's clauses that are not whole sentences (prose is never looked
up, and a whole sentence sits in the general slot), and an item never asked
about (lettering, an @ name). The page shows "Partly checked:
<order_detail>" and the summary's "order partly checked" from that field
alone, wording it as a combination with `order_checked` rather than a second
copy of its conditions; `categories_detail` alone - a cap past which every
item was placed anyway, as prose - no longer makes an order "partly
checked". The note names no slot, because which slots an unplaced item keeps
is each guide's own. Phase-4 review of that fix found it said "Partly
checked" too widely, and one reason too narrowly. Every unplaced item
counted, so a one-tag or one-clause prompt, which no order can be wrong
about (the order rule and `reorder_tags` read fewer than two items not at
all), and an unknown tag every category would leave in order
(`1girl, zzkx oc, masterpiece`) or out of it (`zzkx oc, 1girl`, which is
flagged) read as partly checked, where before they had rightly shown as
checked - and the note told the user such a tag "may sit out of order
unflagged". And a tag the index knew but withheld a category for,
deprecated or newer than the checkpoint's cutoff, was said to be one the
index "does not know", beside chips giving the real reason. `unplaced_items`
now names an unplaced item only where its slot decides the order: some slot
it could take puts it out of order beside another item and another slot
does not (`_undecided`), a pair `_violation`, flagging only what is out of
order beyond doubt, passes. Where no item is, the rule's answer and
`reorder_tags`' order are the same whatever category each item has, and
`test_an_item_is_left_unplaced_exactly_where_its_category_decides` holds
that by enumerating the categories. So on Animagine `zzkx oc, green hair` is
checked (a character, a series or a general tag, it is in order ahead of
`green hair`), while `green hair, zzkx oc` is partly checked; on Kohaku,
whose meta tags follow quality, `1girl, zzkx oc, masterpiece` is partly
checked too. The review's final completeness check found a tag written
twice still partly checked: `1girl, zzkx oc, zzkx oc, masterpiece` named
both copies, since `_undecided` read each against the other as though their
slots were independent. Two items written the same, and both lettering or
neither, are placed alike under any categories (`_place` reads nothing
else), so they take one slot whatever it is and
are never out of order beside each other; that pair no longer counts, and
the prompt is checked. Beside another item a copy still counts (`green hair,
zzkx oc, zzkx oc` is partly checked, the tag named once). The critic's
`placed_partly` reads the same predicate, so a
reorder of the critic's where no missing category decides anything is held
to Check's own order, which is then the documented one (the critic's
paragraph below). The unknown tag's reason now reads "the tag index has no
usable Danbooru category for", which is true of all of them; the chips say
which. Where the guide states no order, the one rule that
reads a category is Anima's artist prefix (`category_items`); it reads none
when the lookup fails and not all past the cap, and the page says not every
tag was checked for an artist name.

**Deliberately not checked**, each for a reason: schnell's "about 180 words",
BFL's reference 256-token limit, which this app does not apply (below; the
prose says T5 reads 512 here and gives the 256 as BFL's, in a sentence the
critic is not given); a negative prompt on the undistilled Z-Image, which its
card calls "Optional, but would be powerful when you want to remove some
unwanted content" (the prose said Tongyi "strongly recommends" one until the
2026-09-28 review) and which is not in the catalogue; the order of a sentence,
an XML prompt's blocks or a JSON prompt's keys (above: shown as not checked);
Anima's card tag order and prefix, which are not in the guide prose - the
critic has them as excerpts and the page shows the order as the authors'; an
Illustrious prompt "missing" a composition or artist tag, which the authors'
own examples lack (the prose's "Give it a style" is the app's reading and
`unsourced`, so the critic cannot cite it to ask for an artist tag either; the
composition sentence is the v0.1 card's and stays citable); quality tags
placed past CLIP's 77 tokens, which the encoder cut already reports;
Illustrious's `sfw`, `nsfw` and `r-18`, which its release post accepts (only
`safe` is corrected, to `general`); a single klein reference with no "image 1"
(BFL's own "Turn this cat into a dog" names none - two or more are checked;
with none attached, an edit of "this" picture - `EDIT_PROMPT` - gets the
app's own note, `heuristic.no_reference`, since it renders from the words
alone, a plain text-to-image that looks like a success);
and `\(`/`\)` escapes, which the app's own autocomplete writes (below).
FLUX.1's "English is the most precise language" was on this list until the
2026-09-28 review took it out of the prose: BFL's current basics page says
other languages work "with the same level of quality", and while unchecked it
was still citable by the critic.

**The critic runs in llama.cpp, not in-process.** Measured on this card
(RTX 5080) with Qwen3-4B: llama.cpp b11206's CUDA build decodes Q8_0 at 170
tokens/s (`llama-bench`), about 151 end to end under a JSON grammar with
reasoning on - 8.8-9.0 s for 1,320-1,348 tokens at temperature 0, 2.8 s with
reasoning off - at about 5,160 MiB of VRAM over the desktop's (7,260 against
2,099 by `nvidia-smi`) at an 8k context, and the server loads in 1.7-1.9 s.
That was the research's greedy decoding; with the shipped sampling
`verified.md` records 8.1-16.0 s per critique (median 10.5), 19.3 s
for a tag prompt, and 5,581-5,611 MiB (about 5.5 GiB) over the
baseline. In-process transformers decodes the same model in bf16 at 33
tokens/s, so a reasoning-length answer takes about 40 s;
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
prompt's tokens as the running server counts them times the copies the answer
repeats (parts, the rewrite and, under a documented order, the reordered
version) plus 800, between 4,096 and 8,192. A fixed 4,096 failed review: a
4,000-character English prompt is 966 real tokens, and three copies after
2,048 tokens of thought do not fit.

**What cannot fit is refused before it is asked.** A question whose answer
cannot fit the 8k window even with no thought is refused with a `CriticError`
(`check_fits`), in two steps. Before a server starts, `critic.estimate_tokens`
scaled by 0.6 (`_ESTIMATE_FLOOR`) refuses only what cannot fit however the
text tokenizes - 4,000 characters of Chinese with as many again in a negative
prompt that runs, say. The estimate splits the text as Qwen2's pre-tokenizer
does after its NFC (Qwen3 keeps both), counts each piece a token at least, and
charges letters past ASCII at a per-script rate measured against Qwen3-4B's
own tokenizer: FLUX.2 [klein]'s `tokenizer.json`, whose ids equal b11206's
`llama-tokenize` on the pinned GGUF. It is a bound in neither direction. It
runs long on ordinary text, by a quarter to two fifths on English prose and at
most three fifths (Vietnamese), which the 0.6 covers - a bare list of
camel-case brands, two and a half times over, is the exception - and short on
vocabulary it cannot see: Turkish or Icelandic words and romanised names by a
fifth, Cantonese by two fifths, rare Han characters by half. So the server
counts too: once it is up and its listener checked, `_counted` asks its
`/apply-template` for the messages as it will read them and its `/tokenize`
for their count and the prompt's. A question that does not fit is refused
there, unasked - what it admits fits, unless the answer outgrows its copies of
the prompt and 800 tokens - and `max_tokens` is sized from the same count; on
the real server on 2026-09-28 the count of the messages equalled the prompt
tokens it then reported (1,072 of 1,072). A server that does not answer those
endpoints, such as one of the user's own, is sized by the estimate, with a
warning in the log. A refusal on the count costs a server start, 3 s on the
card and 28 s under WSL with the GGUF on /mnt/c; under WSL with it in the
Hugging Face cache on ext4, the default, a start and stop took about 3 s
while the file was in the page cache and 8-9 s after a drop (`verified.md`,
2026-09-29). Over 183 cases counted with
the real tokenizer (the 2026-09-28 review's 160, ten kinds of text from
Turkish to Cantonese and romanised names at 500-4,000 characters on klein and
Krea [dev]; English, brand-heavy, English under a 4,000-character negative and
Greek prompts on klein, Krea [dev] and Qwen-Image; Chinese on ERNIE), none
that fits was refused before a server started, none admitted failed to fit,
and no thought was cut while the window had room; 8 were refused before a
server started and 46 on the count. That review found the first estimate - a
token per digit and a third of one per other ASCII character or UTF-8 byte -
short by up to 45% on Greek, base64, Hindi and joined emoji: scaled by 0.6 it
admitted prompts needing 1.4 to 2.4 times the window, which ran to its end and
answered 502, and its `max_tokens` cut the thought short while the window had
room (F37). A second, per-script and unscaled, was still short on the
languages above and, long on the rest, refused prompts that fit.

**The page's "about 15 seconds" is the pinned pair's.** `verified.md` ("Prompt
analyser") has the times with the pinned GGUF and b11206: the 8.1-16.0 s above
over 19 FLUX.1 Krea [dev] critiques (median 10.5 s), each back at the baseline
VRAM a second later with no server left, and the Animagine tag prompt's
19.3 s, before the final review's fixes; 7.4-22.3 s over 20 after them (median
13.2 s), the card at its baseline and no server left after each run of ten.
The times include the server's start and stop; the page's figure lies within
both ranges and above both medians. The critiques generated 705-2,568 tokens,
thought included, which is what `MIN_TOKENS` (4,096) has to cover. An earlier
smoke by builder C, 12.7-18.6 s, once cited here as the basis, is history: it
ran the research's own `llama-quantize` conversion of klein's text encoder
(4,280,404,896 bytes against the pinned 4,280,404,704) with an earlier system
message, and is not in `verified.md` (review 2026-09-28, F79). Only the pinned
pair is promised it: `PromptCritic.describe()` (`ModelStatus.prompt_critic`)
gives the page the model's label - the name the answer's footer uses - and
what a first use would fetch, from the settings and file checks, and a server
or GGUF of the user's own gets neutral wording and no time. The review found
the fixed line naming Qwen3-4B and its time beside a footer naming another
GGUF, and promising a llama.cpp download that a configured server never makes
(F45). Card measurements belong in `verified.md` under "Prompt analyser".

**One pinned server per critique, keyed, and stopped before the gate opens.**
`critic.BUILDS` pins release b11206's Windows CUDA 13.4 archives and the
Ubuntu ones for WSL (the default architecture list includes sm_120) by size
and the sha256 the GitHub release API publishes, and `MODEL` pins
`Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q8_0.gguf` by revision, size and sha256. The
binaries are a download like the weights and handled like them: each archive
to `.part`, verified, extracted into a sibling with zip members that escape
the target refused and tarballs through `filter="data"`, the marker written
last and the tree renamed into place under `HF_HOME/local-diffusion/llama.cpp`;
a tree without the marker is never used, nor is one whose marker does not
decode (re-saved as UTF-16, say): that counts as none, so `ready()` answers
False rather than failing every `/api/status` poll, and the install replaces
the tree (review 2026-09-28, F18). The sibling is a
`tempfile.TemporaryDirectory`, which removes itself on the way out - but not
after a hard kill, nor on Windows when the interpreter exits with a file in it
still open on the prepare thread, and each such attempt left up to 744 MB
(F39). The lock holds through the extraction, so any such directory found
under it is stale, and `_sweep_work` removes them on every install call,
installed or not. The review's first fix took every `<target>-*` directory
that was not another build's target, which would have deleted a user's
`b11206-win-cuda-13.4-x64-backup` beside the install. The sweep takes only
what `tempfile` makes there (`_is_work`): the target's name, a hyphen and
eight characters of `tempfile`'s alphabet (`[a-z0-9_]`), as a real directory -
not a link or junction - and never another build's own (a key of this one's
and eight more characters would be named like work, and empty it would pass
the contents check), holding nothing but the `tree` and the `archive-<n>`
scratch directories an install extracts into. The contents check is what
spares a copy of an install named `...-backup01`, eight characters `tempfile`
could have written, since a copy holds the server and its marker. Removal is
best-effort and never fails an install: a file another process holds, a work
directory or an install root that cannot be listed are logged or passed over,
and what cannot go yet is tried again by the next
(`test_the_sweep_never_fails_an_install`, the converters'
`test_the_sweep_never_fails_a_load`). The model converters save into work
directories the same way and are swept by the same kind of check
(`single_file.sweep_work`). The GGUF is hashed
once, and a stamp of its path, size and mtime spares every later critique
hashing 4 GB again. A cached GGUF off its pin is downloaded again, once per
call, rather than left to "delete it and retry" (F40): in a Hub cache with
symlinks, deleting the snapshot entry leaves the blob, which `hf_hub_download`
relinks without downloading, and in one without them (Windows without
Developer Mode) the entry is the file itself, and `force_download` alone
replaces only the blob beside it - so the entry is removed and the download
forced. If the file is still off its pin, the error names the resolved file
and says it was freshly downloaded, so the pin or the download path is wrong,
rather than advising a retry that has just been made; the pin is the LFS
object's own sha256, so such bytes mean a wrong pin or corruption on the way,
a later click may legitimately succeed, and nothing is remembered. A forced
download that fails (the Hub unreachable, `HF_HUB_OFFLINE=1`) is a
`CriticError`, the route's 502, quoting the cause of `hf_hub_download`'s
error, whose own words for an unreachable Hub are only "the above error"; the
entry is gone by then, which is right, since it was off the pin. A first use
is about 5 GB (576 MB of binaries and runtime on Windows, 4.3 GB of model) -
the page sums the pins' sizes for only what is still missing (`describe()`)
and prints them in binary units, "about 4.5 GB" for both and "about 4.0 GB"
for the model alone - so `prepare()` runs before the gate, on a daemon thread
as the model load does (a Ctrl-C must not wait out a download), and the route
checks the queue and the load again afterwards, since minutes may have passed.
A cached `prepare()` is not quite offline: `hf_hub_download` builds
huggingface_hub's user agent before it looks in the cache, and that fetches
the Hub's `/api/agent-harnesses` list, best-effort and at most once a day,
unless `HF_HUB_OFFLINE` or `HF_HUB_DISABLE_TELEMETRY` is set - the library's
request, left as it is. Only `critique()` runs inside `jobs.run_when_idle`:
`llama-server` on 127.0.0.1 and a free port, `--offline`, `--no-ui`,
`-c 8192`, `-np 1`, `--jinja`, `--fit on` (b11206 adapts only unset arguments,
so layer placement follows the VRAM actually free while the context stays
pinned by `-c`; hence the 180 s health and 600 s request timeouts, for layers
that land on the CPU), then, after the listener check below, the question
counted by the server itself and one chat completion, then terminate, wait,
kill in `finally`, applied to the whole process tree, so the card is free
before an image job is let in. A resident server would hold ~5 GB against the
pipeline for nothing when a start costs two seconds;
`--sleep-idle-seconds 120` is only the backstop for a server this process
could not stop.

**The stop takes the server's whole process tree.**
`LOCAL_DIFFUSION_CRITIC_SERVER` may name a wrapper - a .bat, a venv launcher,
a script that does not `exec` - whose server is a grandchild. The 2026-09-28
review found that stopping the child alone left such a server listening, and
holding the card, after the gate opened, and that the log it still held turned
a finished critique into a 500 (F19). So `critic._ServerProcess` stops a tree,
even when the child itself has already exited, since a wrapper's server
outlives it. On Windows the tree is a job object with kill-on-close. The child
is created suspended and resumed only once it is in the job (by ntdll's
`NtResumeProcess`, since Popen keeps only the process handle), so nothing it
starts escapes; where Windows refuses a job, the child runs alone and only it
is stopped, with a warning in the log. The stop waits until each process's own
handle is signalled, not merely until the job's process list is empty: a
process leaves that list before its handles close, and when only the list was
polled, 17 of 30 wrapped critiques under full CPU load returned with the log
still held and the gate opening mid-teardown, against 0 of 30 with the handles
waited on. On POSIX the tree is a session of the child's own. On Linux every
live process in it is signalled and waited for, whichever process group it has
moved to (`set -m`) - the same set the listener check accepts - and a process
counts as gone only when its last thread has exited, since /proc shows a
multithreaded process as a zombie while its other threads still hold its
sockets; elsewhere on POSIX the child's process group is signalled. A kill
that is still not waited out is a `CriticError`, since the card may still be
taken, and the critic forgets the process either way. The log's removal is
retried for a few hundred milliseconds, as `save_image`'s rename is, and a log
still held after that is a warning, not a lost answer. Open, from review, on
POSIX only: a server can be orphaned by a hard kill of the app, or by closing
the terminal the app runs in. It frees its VRAM after the idle timeout but
stays alive and listening, since its session outlives the app: the terminal's
hang-up no longer reaches it, and the app does not take the hang-up as a
shutdown, because uvicorn handles only SIGINT and SIGTERM. Before the process
tree, the hang-up did reach a server in the app's own process group
(2026-09-28). Treating SIGHUP like SIGTERM in `main.run()` would close the
second case; `PR_SET_PDEATHSIG` was left out, since `preexec_fn` is unsafe in
a threaded process and would reach the child alone. On Windows the job's
kill-on-close takes the server down with an app killed outright (checked with
a stand-in server on 2026-09-28, not with llama-server).

**The key, and who listens on the port.** Its CORS allows every origin, so a
web page in this browser could reach a keyless server: each start gets a fresh
random key, and before the prompt is sent a keyless `GET /v1/models` must
answer 401 (b11206 leaves only its health endpoints public), or the server is
refused. The key goes to the server as `LLAMA_API_KEY` in its environment,
which b11206 reads in place of `--api-key` (its `--help` says so): on the
command line, as first built, any local user could read it
(`/proc/<pid>/cmdline` under WSL). The child's environment otherwise drops
`LLAMA_*` - `LLAMA_ARG_TOOLS` would switch on its shell and file tools - and
loopback requests bypass any proxy. The 2026-09-28 review found the 401 probe
proving less than it was taken to: a process that took the port first - it is
on the command line for any local user to see - and refused keyless requests
would have been sent the prompt and the key, and could have answered in the
server's place (F43). Neither the bind failing (b11206 sets `SO_REUSEADDR`)
nor the 401 probe (a stranger can answer 401 too) guards the gap between
`_free_port()` and the server's bind. So, last before the key goes out (the
count and the question both carry it), the port's listener must be the
server's own process tree (`critic._listener_is_ours`, as
`_wait_until_healthy`'s comment says): on Windows the TCP tables' owning pid
is checked against the job's processes, and on Linux, WSL included, the
listening socket's inode in /proc/net/tcp and tcp6 against the descriptors of
the session's processes, a listener on 127.0.0.1, on every address or
dual-stack counting alike. A listener that belongs to anyone else, or no
visible listener at all, is refused with a `CriticError`, and neither the
prompt nor the key is sent; the real b11206 server passed the check on Windows
and under WSL 2 on 2026-09-28. Where /proc/net/tcp is empty or incomplete
(WSL 1, gVisor) that refuses every critique, which is accepted: the app runs
on Windows and WSL 2. On other platforms, or where the tables cannot be read,
nothing reads the owner, and the key and the 401 probe are the only guard. The
check reads the owner once, just before the first keyed request; a listener
bound beside the server after that moment is not seen. A Unix socket in a
private directory, the review's other suggestion, was left out: b11206's
`--host` takes a `.sock` path, but urllib has no transport for one.

**A server or a model of one's own is the user's choice.**
`LOCAL_DIFFUSION_CRITIC=false` switches the critic off;
`LOCAL_DIFFUSION_CRITIC_SERVER` and `LOCAL_DIFFUSION_CRITIC_MODEL` name files
already on disk in place of either download (made absolute, because the server
starts in its own directory, but not resolved: a Hub snapshot's entry is a
link to a blob named by its hash, and the critique's footer shows the
configured file's name; `server_command` still resolves both for the command
line), and naming one is choosing it: neither is held to the pins, but a
server of one's own must take b11206's flags and its key from the environment,
and should answer `/apply-template` and `/tokenize` with the key - without
them the question is sized by the estimate. The server setting should name the
llama-server binary itself, not a script or launcher that starts it: a wrapper
is stopped with everything it starts - its job on Windows, its session on
Linux whichever process group it moves to, its process group elsewhere - but a
server that leaves the tree (on POSIX, into a session of its own: `setsid`, a
daemon's double fork) escapes the stop, and where the platform can tell, the
listener check refuses its port.

**The Linux build needs what its tree does not carry.** The WSL tarballs'
layout - one top-level directory each, flattened on install so the CUDA
runtime sits beside the binary - was read from b11206's `release.yml` when
this was built and verified on the card on 2026-09-27 (`ldd` resolves every
library inside the tree; `verified.md`). What the tree does not carry was read
off its ELF headers under WSL 2 on 2026-09-28: symbols up to GLIBC_2.38 and
GLIBCXX_3.4.32, libgomp.so.1 for the CPU backends, and OpenSSL 3's libssl.so.3
and libcrypto.so.3 (`Build.glibc`, `Build.libraries`). The 2026-09-27 `ldd`
ran with libgomp1 already installed (it came with gcc); a fresh WSL Ubuntu has
none, and the 2026-09-28 review found its first critique downloading about
4.9 GB to end in exit 127, with the button still offered (F44). `supported()`
is False below glibc 2.38, as on Ubuntu 22.04 and Debian 12, unless a server
of one's own is named; the version is `os.confstr`'s, a libc call cheap enough
for the status poll. A direct call to the critique route then gets a 422
saying so (`critic.pinned_build_refusal`, read by `main._critic_unsupported`),
not that no build is published for the platform. `critic.preflight`, called
from `prepare()` ahead of both downloads, asks the dynamic loader for the
three libraries in a child interpreter (`ctypes.CDLL` there, so no second
OpenMP runtime is loaded beside torch's) and raises `CriticUnavailable`, the
route's 503, naming the apt packages (libgomp1, libssl3; libssl3t64 provides
the latter on 24.04 and later). A missing library leaves the critic offered:
the fix is one `apt install`, and the 503 says which. A probe that cannot run
checks nothing, GLIBCXX is not checked separately, and a server that still
exits 127 on a missing library names its package (`_exited`). `wsl/setup.sh`
checks the same with `getconf` and `ldconfig -p` and prints the apt command
rather than running sudo; it installs none of them.

**The critic cites only what it is given, and what it says is checked
before it is shown.** `prompt_sources` gives it, per guide constant, the
guide's own `prompt` prose split into sentences (G1..Gn - the text the ?
badge shows) and short verbatim excerpts the checkpoint's authors put in
writing (A1..An); the answer's JSON schema makes `cite_id` an enum of
exactly those ids, so a citation cannot be invented, only misapplied, and an
id that names nothing drops its issue (`dropped`). A G sentence links to the
page of the `analysis` entry whose cite it carries - the page that entry's
rule finding links - so a rule finding and a critic cite of the same words
open the same page, and the prose gives each page's words a sentence of
their own: `test_a_sentence_links_the_page_its_rule_findings_do` fails on a
sentence carrying two pages' words. Until the 2026-09-28 review every
sentence without a `sentence_urls` entry took the guide's `analysis["url"]`,
and 15 rule findings in 7 guides (13 sentences) linked one page from the
rule and another from a critic cite; the opening summaries of FLUX.1's,
schnell's, Krea [dev]'s and Krea 2's prose, which folded several of BFL's or
Krea's pages into one sentence, were split one page to a sentence. A
sentence no entry cites links to the page `Grounding.sentence_urls` names
for it, else to the guide's `analysis["url"]`: Krea [dev]'s "AI look" to
BFL's launch post (<https://bfl.ai/blog/flux-1-krea-dev>), its settings
sentence to Krea's README and its camera sentence to BFL's building page;
Krea 2's report claims (dense captions, the text-rendering reward, the
encoder) to the technical report and its CFG and Raw sentences to the
README (<https://github.com/krea-ai/krea-2>); Illustrious's untuned-base,
natural-language-example and v2.0-negative sentences to the v2.0 release
post through CivArchive
(<https://civarchive.com/models/1369089?modelVersionId=1546777>) and its
v0.1 negative prompt to the v0.1 card; ERNIE's encoder to the report
(<https://arxiv.org/abs/2605.25347>) and its 2048 tokens to each repo's
`tokenizer/tokenizer_config.json`; NewBie's 512/2048 to the author's
`finetune.py`; the undistilled ERNIE-Image's and Z-Image's settings to their
own cards. `test_each_sentence_links_the_page_that_says_it` pins each
of those guides but Z-Image sentence by sentence, and FLUX.1's and
schnell's; Z-Image's one `sentence_urls` entry is held only by the
check that every entry is followed, not by which page it names.
Z-Image's "Long, detailed natural language" rule entry cites the
maintainers' discussion 8 in the Turbo repo, where that advice is;
other sentences stating how this app's pipeline reads the prompt
(a CLIP or T5 limit) still link the guide's page. A sentence the prose
carries with no author behind it - the app describing its own menus,
autocomplete and chips, the Expand button, the sampler it runs, what it does
not implement, or a starting point or reading of its own - is listed
`unsourced` and never citable, and no rule entry may cite one: the
2026-09-28 review listed nineteen across ten guides, among them
Illustrious's "masterpiece, newest is a practical starting tail" and "Give
it a style", Kohaku's "Keep prompts concise" (its card's context is 300
tokens) and schnell's BFL-reference 256-token sentence, which does not hold
here. How the pipeline reads the prompt - a token limit, where guidance
brings the negative prompt in, the system instruction NetaYume's loader
adds - stays citable, because it is what runs:
`test_no_citable_sentence_misstates_a_token_limit` holds every stated limit
to `prompt_encoding`'s table and
`test_no_citable_sentence_is_the_app_talking` keeps the UI markers out; a
test fails if a listed sentence stops matching the prose. Krea [dev]'s
"photo language" and "give it room" were the first two such sentences,
until the final review found the critic, grounded in BFL's pages, telling
users to remove the camera details the badge beside it asked for - they
were taken out of the prose and replaced by what BFL's building page does
say (camera settings "when framing matters"). The fallback guide's
sentences are the app's general advice: its system message names no
authors (`build_context` drops "from the checkpoint's authors" when
`Sources.url` is None), its citations carry no url, and its "Unrecognised
checkpoint." label is unsourced.

Excerpts, checked against copies fetched 2026-09-27 - verbatim but for
markdown emphasis, elisions marked "…" and a few labelled reformattings
(three rows of BFL's replacement table, Neta's vague/specific pair, orders
flattened from lists): BFL's family pages for FLUX.1, schnell and Krea [dev]
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
An excerpt's note travels with its citation to the page too
(`Citation.note` -> `CritiqueIssue.note`), shown under the issue and in the
cite's tooltip, as Order shows the same notes; until the 2026-09-28 review
it reached only the system message, so a cite of ERNIE's translated
rewriter instruction read as Baidu's words and NetaYume's inherited prompt
book as its own authors'. The fallback's cites carry this app's note
instead ("this app's general advice: no author sources are known for this
checkpoint"), and the page labels them General, not Guide.
The critic is also told that order templates position the parts actually
supplied, not require every slot. Generic and original subjects are valid;
character and series names are optional and must not be invented or flagged
as absent. The deterministic tag rules already work this way, and
Animagine's own examples include generic subjects without either name.

The system message also says the rewrite adds no object, scenery, mood or
text and never pads toward a word range - a range is the guide's band for
most images, not a minimum (almost every rewrite on the card padded a short
prompt, against the sources' own "Start short") - that quotation marks go
only around lettering, never the prompt, and that the authors' example
prompts lend their form, not their subjects or text. It stays under about
1.5k tokens (the phase-1 review measured 500-880 with the real Qwen3
tokenizer before findings; the budget test holds every checkpoint's under
6,000 characters).

Then every answer is checked (`validate`). A quote is looked for in the
prompt as whole words - "man" is not the end of "woman", "pen" not the
middle of "OPEN", "OPEN" not the start of "OPENING" - though Han, kana, Thai,
Lao, Myanmar and Khmer, written without spaces, match anywhere, as does Hangul
(Korean spaces its words, but a particle joins the word before it; `_SPACED`
builds on `QUOTED`'s `_WORD_NOT_CJK`), and a tag's underscore counts as a space
("hair" in "long_hair"). Case, runs of whitespace, and curly against straight
apostrophes, quotation marks and dashes are set aside (folded one character for
one, Word's non-breaking hyphen included, so the span is still the prompt's
own); the quote is nulled when absent, the span located on the server so the
page selects the prompt's own characters. A part is kept only if each of its
comma pieces is in the prompt, and "none" or "n/a" is no part, and
`text_to_render` only if it is lettering the prompt asks for
(`prompt_rules.rendered_targets`: its quoted stretches and unquoted sign
text) - on the card it held the whole prompt in most runs. Words may run on
within their clause past an end the rules had to guess; past one they are
sure of ("says OPEN in red letters") only words starting with a capital or
a digit run on - "OPEN IN 5 MINUTES", or "Cafe On Main", a name the rules
stop at its "On" - so lowercase description never passes as lettering.
Either way they start before the target's `head`: where lettering may hold
a negation the rules stop at ("reading comics without pictures"), the
negation's words alone are the absence the rules read, not lettering.

An issue that repeats a rule finding reported on the prompt (a negation,
quotation marks, tag order, weighting, quality boilerplate) is left out,
whether it repeats it in its wording or by a quote that takes in the words
the finding points at, since the model did so in nearly every answer
however it was told not to; a finding on the negative prompt counts only
for an issue that says it is about the negative prompt. Left out too is one
that speaks to the prompt's own length wherever the guide has a range and the
rules counted it (not XML or JSON), reported or not, since the model
miscounted words beside the rules' correct count; in range, `set_aside`
says the check measured the length (N words, inside the range) rather than
naming a finding that is not there. The 2026-09-28 review narrowed all
three checks. The length pattern had caught "a focal length of 85mm", "the
dress length is" and "add 3 words on its glow"; it now reads a sentence at
a time, and only claims about the prompt's length (its count, the guide's
band, a bound, "Too short"), not a number of words about something ("add at
least 5 words about the lighting") or lettering ("keep the sign text under
5 words"). Phase-4 re-verification found that narrowing held only for a
count followed by "on", "about", "of" or "describing": "add at least 5
words for the lighting", "add more than 3 words to the lighting", "only 2
words describe the setting", "describe the umbrella in more than 2 words",
"give the lighting at least 5 words", "the lighting gets only 3 words" and
"the ideal length of the raincoat" still went, under a note saying the
check had measured the prompt's length. Its fix made a count some part's
where "for", "to", "in", "covering" or "describe" followed it
(`_SOME_PART`), "in" came ahead of it, or a part was given it ahead ("give
the lighting", "the lighting gets"; `_PART_COUNT`), unless what followed
named the prompt, its guide or range, "it", a total or all, or a verb of
fit - and review of that fix found each exception wider than its purpose.
"Only 3 words of the prompt are about the lighting", which the first
narrowing had kept, went again, since "of the prompt" named the prompt;
"to match the lighting", "for all three figures", "the dynamic range" and
"the style guide" were the prompt's, since a fit verb, "all", or any range
or guide was; "the lighting has only 3 words" was still the prompt's
length; and the other way, "keep it in under 80 words", "the recommended
length of 30 words" and "too long a description" were no longer length
claims. Now a count of the prompt's words is a part's where the sentence
goes on to give them to one ("are about", "cover", "describe", "go to"); a
fit is the prompt's only where it fits the guide, its range, a limit or a
number ("to reach the range", "to stay under 80"); a range or guide only
bare or as the guide's, the model's, the recommended or suggested one or a
span of numbers; "all" only as "in all". A part named ahead takes its
count by give or devote, by has, is, are, gets, receives or uses, or by
cut, trim, shorten, expand, keep or limit to or under a bound
(`_PART_COUNT`); "in" or "with" ahead of a bound is how a part is to be
described unless "it", "this", the prompt or its description comes before
it (`_MANNER`); and a recommended length of so many words, of prompts or
of the description is the prompt's. Each of those words has a test row of
its own. It is still a reading of the critic's wording, so a phrasing
unlike these can go either way. "The prompt runs long" (reads, seems, a
little, slightly) counts as a length claim; "too long a shadow" does not,
"too long a description" or "a list" does. The last fix round found "keep
the lighting description under 5 words" still left out under the note that
the check had measured the prompt's length, since any description was the
prompt's: now the word just ahead of "description" decides. A determiner
("the description", "your description") or a word for the whole ("the
full description", "the image description"; `_WHOLE_WORDS`) makes it the
prompt's; any other word names a part ("the lighting description", "the
background description"), as an object other than the image or picture
does in "the description of the lighting" - in `_PART_COUNT` and in
`_MANNER` alike. It also found "the prompt reads as long" kept: a claim
about the prompt's length, like "reads long", it is now left out with the
measured note where the count was in range, and as repeating the finding
where one was reported. Review of that fix found any word the list did not
name taken for a part: "keep the revised description under 80 words", "the
new description has only 12 words" and the like, the critic's words for its
own version or for the whole, were kept where they had been left out. The
other way, "your lighting description", "a lighting description" and "the
soft warm lighting description" were still the prompt's in a count, whose
determiners lacked "your", "my", "our", "a" and "an" and which named a
part in two words at most, while a manner read them as parts; plural
descriptions were the prompt's in a count and a part's in a manner; and
"as" after any verb made "the prompt is as long as it needs to be, but the
lighting is vague" a length claim. Now one list of determiners
(`_DETERMINERS`) and one of words for the whole (`_WHOLE_WORDS`, the
critic's "revised", "new", "final", "short" and "detailed" among them)
decide both readings, singular or plural, a noun for the whole as a
possessive too ("the image's description"); a count takes a part named in
up to four words; "a" or "an" names a part only with its description, so
"keep a single paragraph under 80 words" is still the prompt's; and only
"reads as" is a length claim, never "as long as".
`CriticContext.reported` counted negative-prompt findings, so weighting in Animagine's negative prompt left
out "give the expression more emphasis".
A quote within 8 characters of any finding's span counted, so "a shop" and
"rain" beside "no people" went as restating the negation, and on FLUX.1
T5's `unknown_chars` spans, which cover nearly all of a Chinese prompt,
took every quoted issue. A quote now counts only if it takes in some of the
finding's words and runs at most 8 characters past them (none past a
single word, surrounding punctuation aside), and only the spans of
findings about meaning count: guide, author and heuristic rules and
weighting syntax, not a cut, a template tail, unknown characters or a
control token. Order issues are read as what each asks for: a part first,
or one part before another ("put the camera settings before the
lighting"); an instruction in the negative ("avoid placing the style before
the subject", "don't start with the style") asks for nothing it names. One
the documented order reverses is left out - a part first that the order
puts another part ahead of, or a pair it puts the other way: on FLUX.1
[dev], "lighting first" or lighting before camera; on Krea [dev], whose one
step is "subject first", anything but the subject asked for first or put
ahead of it. A step names every part in it, so WAI's "Subject and scene"
puts the scene first too. Then the order issues whose wants cannot all
hold together are all left out. The review found "Start with the subject"
and "Put the camera settings before the lighting" both left out as a
conflict, because the pair had been read as "camera first"; with the
documented order applied first, "Move subject to the beginning" beside
"Place style description first" on klein keeps the former, where both used
to go.

`reordered` survives only where the app has a documented order and only as
a pure reorder - the same whole tags, clauses, sentences or XML blocks (for
JSON, the same values written back with the keys sorted, each number as the
text writes it (`_json_forms`), with no key repeated, in another key
order), every quoted span unchanged, typed marks folded
(`is_reorder`). The movable pieces also protect every `rendered_targets`
span, including unquoted lettering such as `《你好, 世界》`; this protection
applies before either tag-order validation or the unchecked prose path.
Initials and common abbreviation periods stay inside their piece, so
`Mr. Green` cannot trade names with `Ms. Blue`, and `J. M. W. Turner`
cannot become `J. W. M. Turner`. An ambiguous abbreviation at a sentence end
conservatively prevents that split. Here "quoted" is `prompt_rules.QUOTED`,
the one pattern the rules, the encoder layer and the critic share (the
critic's own narrower copy let a «…» or „…“ quotation split across two pieces), whose
single-quote pair may touch Han, kana and Hangul but no other letter or
digit: Chinese writes 写着'营业中'的 with no space, and ERNIE's own
rewriter asks for single quotes. Until the 2026-09-28 review the pattern
refused that pair, so such lettering was nulled from `text_to_render` and a
rewrite that changed it was kept. Word multisets were the first reorder
check, and review broke it: "a woman in a red dress and a blue hat" came
back with the colours swapped and the sign text's quotes gone, and passed.
The 2026-09-28 review found the JSON check comparing values with Python's
1 == 1.0 == True, so a literal retyped as another type passed; its fix
compared `json.dumps(sort_keys=True)`, and phase-4 re-verification found
that still a comparison of values: 0.50 written 0.5 or 5e-1, 1.0 written
1.00 or 1e0, -0 written 0, passed beside a key moved, and the kept version,
the critic's text, gave the encoder other text than the user's. Numbers are
now read as written. What is kept (JSON aside, which keeps the critic's
text: its values are the prompt's writing, only the space between them the
critic's) has the prompt's own pieces, each as the prompt writes it, in
the critic's order, with the critic's space and punctuation between them: a
case-folded comparison once let an upper-cased reorder through, and T5 and
Qwen3 read case, while separators left where they lay turned "says EAT!"
into "says EAT." once the sentences moved. Where Check placed the tags
itself (`order_checked`), the critic's version stands only if its pieces
are the rules' `reorder_tags` result, and the text kept is that result: any
other permutation is not the documented order. That holds where every item
was placed. Where the rules could not place some item the order rule reads
(`prompt_rules.places_every_item` false, placing with the categories the
analysis used, `PromptAnalysis.categories`, kept off the response), that
item keeps every slot a category could give it, and Check's order - or its
"nothing to move" - is one the documented order allows, not the only one.
That covers an item never looked up (offline, an unusable reply, past the
64-tag cap) and one the index was asked about and had no usable category
for - an OC's name, a trigger word, a typo, a tag past the cutoff - where
that item's slot decides the order (below). There the critic's
version stands where the order rule, run again on it with the same
categories (`prompt_rules.order_finding`), flags nothing, and is set aside
in the rule's own words where it flags something. It must also keep the
items the order rule reads whole (`prompt_rules.order_items`) and be text
the rule reads at all: outside protected lettering, the critic's pieces
break at "、", ";" and line breaks too, so four tags joined by "、" are one
item to the rule, which no order can be wrong about. Lettering such as
"《你好, 世界》" is now protected directly during piece splitting, including
on prose checkpoints whose order the rules do not check. The first fix of
the 2026-09-28 review put this exception on the page: Compare offered the
critic's reordering where the lookup was partial (`categories_detail`), while the server still dropped
every such reorder, so the page's branch was reachable only by injection.
The follow-up moved the one rule to the server, first keyed on
`categories_detail`; its review found that key wrong - 64 tags the index
did not know were set aside with a note saying Check had placed them,
while 65 were kept - and the raw comma items it compared letting a critic
part 《…》 lettering at its comma. Both were fixed as described above.
Phase-4 re-verification found the page's "Partly checked" still on the old
key, so the critic's "allowed" reordering could sit under an order drawn as
fully checked; the page now reads `PromptAnalysis.order_detail`, set from
the same placement (`prompt_rules.unplaced_items`), and
`test_the_page_and_the_critic_read_partial_placement_alike` holds the two
to the same prompts. The review of that fix narrowed the predicate for both:
an unplaced item counts only where its slot decides the order. Where none
does, `reorder_tags`' order is in the documented order whatever the missing
categories are - a pair it swaps was out of order beyond doubt, and one it
keeps was in order beyond doubt, or is a tag and its copy, in order
whatever that tag is - so it is the documented
order itself, and a critic's other permutation is set aside as before. With
`green hair` known and `firefly` not, `1girl, firefly, green hair,
masterpiece` is in order whatever `firefly` is, and the critic's swap
putting `firefly` after `green hair` - out of order were it a character -
is now set aside instead of being kept as "allowed"; with the prompt the
other way round, `firefly`'s slot decides the order, and the swap putting it
first is kept.

On the real server the model's `reordered` added "great score, absurdres"
and once copied a character tag from the example prompts; on FLUX prose it
survived in 1 of 10 runs, and in 0 of 20 after the final fixes, being the
rewrite again 17 times (`verified.md`). So on tag checkpoints the rules'
`reorder_tags` is the dependable reordered version, and the page prefers
it; on prose the user writes one (Compare). The comparison takes the rules'
reordering, else the critic's as the server kept it, and adds no rule of
its own. The critic's Use reordered says which it is as the server says
(`PromptCritique.reordered_kind`: `check`, `allowed`, `unchecked`), not as
the page reads its own last check: Check's own order, one Check's rule
finds nothing wrong with though it could not place every tag, or - where
Check read no order - the critic's reading of the order, not one the app
checked. The critique route analyses the prompt again, and a lookup that
failed is not cached, so the page's last check may have placed the tags
with other categories than the ones the server decided with. On a tag or
hybrid checkpoint Check places a prose prompt too: a clause that
`_is_sentence` reads as a sentence takes the general slot, so there a
sentence order is tested under Your version, while a clause that a Danbooru
name could still be keeps every slot, since prose is never looked up, and a
critic's reorder of such clauses can stand as `allowed`. Where every clause
was placed, a critic's reorder of its sentences is set aside like any other
order but Check's; the note said Check placed "these tags" until phase-4
re-verification found it said so of sentences, and it now names tags only
on a prompt read as tags, and "this prompt" elsewhere. The last fix round
found its sibling, for a reorder that does not keep the comma items whole,
still saying Check would read "other tags" on a prose prompt ("a girl
standing in the rain. smiling softly, holding a red umbrella", whose items
`order_detail` calls clauses) and a hybrid one; it follows the same rule
now: "other tags" on a prompt read as tags, "other clauses" on prose and
"other tags and clauses" on a hybrid prompt.

Retained tag items keep their original parenthesis escaping. The critic had
returned tag prompts with only the backslashes in series names removed,
which looked like an improvement and became a redundant comparison row.
Validation restores the original spelling where a whole tag differs only
in parenthesis escaping, before the unchanged check; it does not normalize
prose, lettering, or other changes to a tag. The critic is instructed to
keep escapes and return the original when no useful edit is needed.

A rewrite that is the prompt as it stands (whitespace, a closing stop and
how its marks are typed aside) is dropped and said so, since the schema
requires a rewrite and an echo is the critic proposing none. One wrapped
whole in quotation marks, as four of the 19 FLUX rewrites on the card were
(`verified.md`), is unwrapped, sentence punctuation after the closing mark
included: a stop, spaced or not, stays on the text, closed up, and a comma,
semicolon or colon - the critic's own sentence going on - goes. The
2026-09-28 review found '"…sofa".' dropped as invented lettering, and the
fix set aside only a stop right after the mark; phase-4 re-verification
found '"…sofa" .', '"…sofa";' and '"…sofa",' still dropped that way. One
that quotes words the prompt does not have is dropped -
lettering nobody asked for, such as the sign example's "Black Forest
Labs" - and one that quotes the prompt's description gets a
`heuristic.rewrite_quote` warning where it sits: the rules skip quoted
text, so none of these reached the layer-1 re-check. Quoted words are held
to the prompt's lettering with the same folding as a quote, so “Joe’s
Diner” retyped with a straight apostrophe, or 24–7 with a hyphen, is the
same lettering; until the 2026-09-28 review such a rewrite was dropped as
inventing lettering. The review's optional warning that a rewrite changed
only its lettering's punctuation was not built. A rewrite with six or more
content words the prompt has in no form gets `heuristic.rewrite_adds`
(info), naming them: the instruction not to pad did not stop the card's
critic from adding lenses, light and scenery to short prompts, and a
comparison against such a rewrite tests new content, not only wording. What
validation takes out besides uncited issues is said on the page
(`PromptCritique.set_aside`): restated issues and ones speaking to a length
the check measured, order issues the documented order reverses or that
cannot all hold, parts the model named that are not the prompt's words or
its lettering, a rewrite or reordered version over the 4,000-character
limit, a rewrite that is the prompt unchanged, came back wrapped or quotes
words the prompt lacks, and a reordered version that changed words or that
Check's order does not take (above).

The answer is read from the content after the reasoning, but not only after
the last `</think>`: a prompt pasted from another model can hold one, and
the answer quotes it back. Qwen3's control and reasoning tokens as typed in
either message - `<|im_end|>` and every `<|...|>`, `<think>`,
`<tool_call>`, `<tool_response>` and their closers - reach llama-server
defused by a zero-width joiner after the "<"; one already there gets
another, so the escape is reversible. llama-server tokenizes the rendered
chat with special tokens parsed, so until the 2026-09-28 review a typed
`<|im_end|>` ended the user's turn and opened one of its own, and on klein,
Z-Image and Qwen-Image the control-token finding that quotes it ended the
app's system message early. `validate` takes the joiners out of the
answer's strings, so quotes and parts are looked for in the prompt as
typed. The rewrite's layer-1 findings are the rules and encoders run over
it by the route (`_lint`, without the tag lookup), added to `validate`'s
own - after `run_when_idle` returns, not inside it: the lint can load a
tokenizer the analysis failed to, from the Hub, waiting up to `LOAD_WAIT`
(30 s) for it, and inside the gate that held image jobs back;
`test_the_rewrite_is_linted_once_the_gate_is_open` holds it. What survives
all this is still a 4B model's reading - it misapplies sources and pads
prompts - which is why the page shows every version to be read before it is
used, and the comparison exists.

**The comparison is a grid, and its statistics are the caveat.** The
variants are one `POST /api/grid` with every other axis empty, the panel's
seed (or a drawn one, written back as grids do), and 4 seeds per version
by default, 8 offered. Its own Batch select offers 1 (default), 2 or 4:
`num_images` is Batch and grid `seeds` is the chosen total divided by Batch.
Each grid seed-axis step advances the base seed by Batch, so every version receives
the same 4 or 8 seeds and the image count stays versions times that total.
The preview gives the count, seeds per version, batch size and estimate before
anything runs, and the page opens that submission as the sheet when its
stream says done (`showJob(jobId, { fresh: true })`). A run can contain the
same prompts with original and yours swapped; `job:<id>` keeps each row's
meaning fixed. If Favorites is ticked, it comes off first and the sheet's
note says so: nothing in the new comparison is starred yet, so
under the tick the sheet opened empty. The tick is removed rather than
bypassed because the gallery's filters are the sheet's, and Re-render sends
the tick with the search - a sheet of unstarred images under it would
re-render none of them. At Batch 1, for a 1 MP 28-step Krea [dev] image in fp8,
`verified.md` records 41.6-42.6 s (2026-09-27), so three versions on four
seeds take about 8.5 minutes. The 15.4 s once set beside it was an 8-step
image, and the parent commit 7a9c651 rendered at the same ~42 s through the
CLI, so neither points to a slowdown. The preview's estimate, from the
measured per-step time, is the one to go by.
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
order dependable on a prose prompt: the rules never move one prose clause past
another, and on FLUX prose the critic's `reordered` rarely survived its check
(above). Scoring the images automatically (a VLM asked whether each element
appears) was left for later: nothing about its accuracy on these checkpoints
has been measured, and published judges are weakest exactly on text rendering
and style. Stars (`stats --job <id> --by variant`) and the run's `distinct`
count are the signals that exist.

**The page.** A retained rewrite shows **Changed from original**, with
character counts and an expandable exact diff. `promptChanges` aligns whole
words, whitespace and individual punctuation using a bounded common-subsequence
table after stripping common ends; it retains every character, including
escapes and emoji. Added and removed text is built with `textContent` in
`ins`/`del` elements, never HTML from the prompt. It compares against the
checked source even if the editor has since changed. An absent rewrite says
**No rewrite offered** alongside the existing set-aside explanations.
The diff stays closed by default - opened, it prints the whole rewrite a
second time directly above its own box - and a gist line after the
`<details>` (a closed one shows only its summary) gives the first three
changes, a few words of each, marked as the diff marks them ("Changed A
quiet → An empty · added film grain · moved masterpiece"). The 2026-10-04
review found the change hidden behind a click. The gist groups the diff's
runs first (`changeHunks`, `gistItems`): a word replaced is one change, not a
removal and an addition; edits only spaces and marks keep apart are one; and
words taken out in one place and put back in another, separators aside, are
one move. Run by run, a real rewrite's gist spent its three places on single
words ("Removed A · added An · removed quiet"). Each finding the check makes
on the rewrite says whether the prompt has one under the same rule and field
("also in your prompt") or not ("new"), and the prompt's findings the
rewrite no longer has follow as "No longer flagged: ..."; they are matched,
never collapsed, by rule, since the same rule can be another instance - on
Animagine the rewrite's order finding names other tags than the prompt's.
The rewrite is checked without the tag index (`main._lint`), so a prompt
finding that rests on it - the order, the style a known name reads as, an
artist's @ (`readsTagIndex`) - is never called fixed: gone from the rewrite,
it is listed as "Not flagged on the rewrite, whose check runs without the tag
index - these may still apply". Review found "No longer flagged" naming a
Series-before-Character order the rewrite kept word for word. (Linting the
rewrite with the analysis's own categories would let the page say more; that
is the route's to do.) The answer's parts list is captioned "How the critic
read it"; uncaptioned, it read as part of the issue under it.
`smoke-rewrite.json` checks the rendered diff, exact reconstruction,
escape-only changes, Unicode, markup as text, unchanged/absent rewrites, the
gist and the carried, no-longer-flagged and not-flagged marks;
`smoke-analysis-tags.json` the not-flagged one end to end on Animagine.

Check sits in the Prompt caption after Tags and before Expand
and sends what Generate would (`collect()`); its panel moves with the chips
(`placeTagValidation`), above the results on a two-column page - beside them
from 1800 px, docked by the stylesheet (below) - so its length cannot push
Generate down the sticky settings column. The move keeps
whatever had focus: `moveBefore` where the browser has it, elsewhere the
focused element and its selection put back by hand. The 2026-09-28 review
found the move re-appending the subtree, which blurred whoever was typing in
Your version or sitting on a chip when the window crossed 901 px, and the
next keystroke went to `<body>`. The panel describes one text - the prompt
and the negative prompt as `collect()` sent them, plus whether the panel's
CFG or True CFG applies that negative - because every offset in the answer
counts from that text and every Use and Apply is a whole replacement worked
out from it. Any change to either box marks it stale: typing, each
programmatic write through the one
`promptReplaced()` hook (an example, Reuse, Reset, a chip, an accepted
suggestion), `syncCfg()` greying or waking the negative, or changing its
effective activation while an open grid keeps the editor enabled for other
CFG values. The raw text and effective negative are captured before either
Check or the critic awaits a reply; later settings cannot relabel that answer.
Raw edits still invalidate a check when the enabled negative is inactive,
since some rules read what was typed and Apply must preserve edits. Adding or
removing a reference image also marks it stale (`setReferences`, or Reuse
through its `syncCfg()`): the check read those too, since klein's `guide.references`
counts them. Stale results stay visible, dimmed, with "Check again" and a
badge - "references changed since" where only the references moved, else
"prompt changed since", written only when its words change, not on every
keystroke; every Use, Apply and Run comparison goes dead, and a finding's
words are selected only while they are still where the check found them.
The first build compared the prompt alone, and "Apply to the negative
prompt" then overwrote a negative typed after the check; the 2026-09-28
review found the references still left out, so a third one left the panel
looking current, with no "image 3" note. The critic's answer is always
about the checked text: asking checks again first when either box or the
references moved on, a check of other text drops the answer, and an answer
that lands after an edit is shown dimmed and applies nothing. The critic's
hold starts at the click, that check included. The 2026-09-28 review found
it set after the check: Generate, Expand and Unload stayed live through it,
so a job started then got the critique a 409, and a Check pressed meanwhile
replaced the critic's check, so the critique was never sent, with no
message. Check and Check again are disabled while that check runs, and
re-enabled in a `finally`; a check replaced anyway says the critic was not
asked, while one a Reset or another checkpoint clears says nothing, since
the cleared panel shows nothing pending. Close pressed meanwhile hands focus
to the prompt: the disabled Check would drop it to `<body>`. A checkpoint
change clears the panel. Apply is offered on rule findings only - a rule's
`suggestion` is the field rewritten - while an encoder's suggestion (for
`((masterpiece:1.2))`, `Write "masterpiece" plainly; ...`) is advice and
shows as text. Use and Apply write through `replaceBoxText`, which selects
the box and inserts with `execCommand`, so Ctrl+Z brings back what was
typed; assigning `.value` emptied the undo history, and the re-check that
follows would have made the replacement the comparison's original, leaving
the typed prompt nowhere on the page. Use rewrite and Use reordered (the
critic's, and Check's own reordering) also put the prompt they replace into
Your version while that box is empty, with a note under it saying so: the
2026-10-04 review tried the rewrite and then compared it with what it had,
the obvious order, and found Compare offering only the new text and Run
comparison dead. A box holding words of the user's own is left alone, the
answer is still dropped by the re-check, and no new version label is added:
Your version already has the lifetime rules the replaced prompt needs. A
finding's Apply is not extended this way - it fixes one rule's complaint
rather than offering a version - and the order finding, whose suggestion is
the Order section's reordering byte for byte, says "Same as Use reordered
above" instead of a second button for the one action. Server offsets count
code points, as Python indexes a str (`PromptFinding` says so); the page's strings count
UTF-16 units, so `units()` converts before selecting - one emoji ahead of a
finding selected the wrong words, and the still-in-place guard, comparing
two JS slices, could not notice. The cut link's excerpt ("stops reading at
...") is cut in code points too: cut in UTF-16 units, as until the
2026-09-28 review, it could end in half an emoji wherever there was no space
to cut at, which on CJK is everywhere. An encoder row is drawn as a caution
only when the server says `pooled_only` (FLUX.1's CLIP-L); keyed on `role`,
it drew NetaYume's only encoder's cut as mild. The critic holds what Expand
holds - Generate and Run grid aria-disabled, Expand, Unload - and `startJob`
refuses while it runs. Its own button is aria-disabled while it runs
("Thinking…"), while Expand runs and while a job is queued or running, the
title saying why, and `askCritic` refuses each case; the 2026-09-28 review
found it disabled, which dropped a keyboard user's focus to `<body>` at the
click. An answer takes focus (`#critic-result`, tabindex=-1, a labelled
group) if focus is still on the button: nothing else announces it, since the
line under the button goes back to the cost. Check pressed on a two-column
page moves focus to the panel heading (tabindex=-1, its `scroll-margin-top`
clearing the sticky topbar) if focus is still on the button, because the
panel is about forty Tab stops away there; under 901 px it is three stops
away and focus stays. Check again hides itself once its check lands, so it
hands focus to the heading first. The comparison temporarily sent `num_images: 1`
because inheriting the panel's Batch 4 had made "4 seeds" sixteen. It now
uses its own Batch select and divides the grid seed count by that batch
size, preserving the total seeds per version. The main panel's Batch and
Repeat (the count once labelled Runs) still do not apply; that line and the
rest of the mechanics are a `?` badge after the comparison's Batch select,
opening upwards (from 1800 px the panel scrolls inside itself, and a tip
opening down from its last section was cut off), while the 4-or-8 caveat
stays in view beside it as the decision aid. It reads both comparison
selects directly for its preview and its run (a browser restoring form state sets a select
without a change event, and a copied variable had stayed at 4), never writes
the prompt box or the grid's variables, which are the user's and persisted,
and leaves out a version identical to another. Its Your version box is the
user's too: kept across checks (a check of other text still compares against
it), cleared with the rest of the panel, never persisted (autocomplete=off,
as are Seeds and comparison Batch, to discourage browser restoration).
`smoke-comparison-batch.json` checks all six Seeds/Batch pairings, preview
and request counts, select values changed without an event, and paired
output seeds, sheets and stale-run disabling. API tests cover paired saved
seeds, batch grouping and preview counts for all six pairings.
The Order section says "Not checked" under a documented order the rules did
not read (`order_checked`),
and the summary "nothing flagged · order not checked" rather than "nothing
to flag"; a prompt the rules placed only partly (`order_detail`: some tag
or clause kept every slot where that decides the order - the lookup failed
or stopped at its 64-tag cap, the index had no usable category for the tag,
a prose clause is never looked up) gets
"Partly checked: <why>" under the chain, naming those items, and the summary
adds "order partly checked" whatever else it counts. Until phase-4
re-verification the page keyed this on `categories_detail`, so an unknown
tag or a prose clause showed the order as checked beside a critic block
saying Check could not place every tag; the first fix then said it of a
lone tag or clause, and of an unknown tag no category could move, too. A negative prompt finding's Apply is dead while the negative box is
greyed, with a line saying what CFG would wake it: `collect()` sends nothing
from a greyed box, so the write changed nothing the re-check could see and
the same finding came back with the same button. The critic's `set_aside`
lines are shown under its answer, and the `dropped` line says what dropping
means - a citation that named nothing it was given, or an empty issue; an
unfound quote only loses its quote.

**Beside the results from 1800 px.** The 2026-10-04 review measured the
check at QHD: opened above the viewer it was 2,161 px wide and 669 px tall
after Check, 1,136 after the critic, and the image just made went from wholly
in view to 59% and then 61 px of it; at 2048x1040 at 125% none of it was left
after the answer - while about 1,270 px beside the image stood empty. So from
1800 px the stylesheet docks the open check in a column of its own beside the
viewer (`--dock-w`, 520-760 px): `.stage` becomes a grid while
`#prompt-analysis` is open in the wide host (`:has()`), the chips, the viewer
and its note on the left, the check on the right across their rows,
scrolling inside itself - and everything from the gallery head on spans both
columns in the rows after, in document order. The check takes the height of
the left column's rows rather than setting it (`contain: size`, stretched),
at most the window's height under the topbar; the viewer, while the check is
docked, is at least 36rem or that height, whichever is less, so beside an
empty viewer the column is still one to read. (The minimum sat on the column
first, and since the column spans the note's `1fr` row, Chromium grew that
row by what the minimum lacked over the empty viewer's 380 px even beside a
917 px image.) The integration review found the first version's
two faults with it. It was sticky under the topbar, and a sticky grid item
is held by the whole stage, not by its rows: as the page scrolled it slid
down over the gallery head, the cards and an open comparison sheet, hiding
the search box, Favorites and Sheet and taking their clicks. And it was
sized by its own content up to the window's height, so the critic's answer
always set the rows' height and put the gallery head past the fold at QHD,
beside an empty viewer as much as beside an image (1,313 px of 1,305); it
is now at 708 px beside the empty viewer and, beside a 1024x1024 image, at
1,049 with the check open or closed (866 of 1,040 at 2048x1040 at 125%).
The note's row is `1fr`, so a check taller than the viewer would have its
excess below the viewer rather than above it; with no
chips there is no row for them, since an empty one kept its gap and opening
the check moved the image down by it. The check's head - the title, the
stale badge, Check again, Close - sticks at the top of the column's own
scroll: the critic's answer takes focus and scrolls the column down to it,
and the head went with it, so a prompt edited after the answer left only
the light dimming to say the check no longer described it. Check, focusing
the heading, puts the column back at its top by hand, since a stuck heading
is always in view and `focus()` no longer did. With no chips the image stays where it was after
Check, after the answer and for a Generate pressed meanwhile, and Generate
does not move. On a tag checkpoint the chips move from the full width into
the left column, and the image moves down by exactly the lines they wrap
onto there. With the tray's own label, `+ tag` and pinned hint (the chip
package's), measured on the integrated page: none for six tags at QHD (one
line of chips there either way), 69 px for 40 tags at QHD, where the image's
bottom is then 97 px above the fold, and 65 px for 32 tags at 2048x1040 at
125%, where it is 25 px above. There a prompt of 40 tags - well past CLIP's
75 tokens - takes the image's bottom 8 px past the fold. Keeping the chips across both
columns would leave the image put, but the check would then start below them
and its bottom, sized for a column under the topbar, would sit below the
fold until the page scrolled. It is
CSS only: `placeTagValidation` and the DOM order - chips, then the check, in
the wide host - are unchanged, the focus rules still hold (the heading after
Check, the answer after the critic: `focus()` scrolls the panel's own box),
and without `:has()` the check stays above the viewer, as below 1800 px. The
cost is a `.many` batch's eight slots at about 160 px while the check is open;
leaving the dock for a batch would flip the layout as the batch starts. In the
column the sections stack - side by side, Order had 356 px - and full width
above the results, from 1400 px, Encoders keeps a 260-360 px track and Order
the rest, where `auto-fit` had made seven 286 px tracks of which two were
used. Text is capped at 120 characters' width (`.analysis-list`,
`.analysis-text`, `.critic-diff`): at QHD a finding's words ended 800-1,400 px
from the badge saying what they rest on, and a version ran 400 characters a
line. A guide or author cite under 80 characters is in the line, after the
claim, as `<q class="analysis-cite">`; it was only the badge's tooltip, which
nothing but a hover shows. The badge keeps its title and link. The critic's
Rewrite and Reordered sit side by side where both fit at 60 characters a
line: full width above the results from 1400 px; in the docked column,
narrower than two such lines, they stack.
Badges are sentence case at the notes' size; Ask the critic and Run comparison
are lit while they would act (never while `disabled` or `aria-disabled`),
since as ghost buttons enabled differed from disabled only by opacity; Close
is a 28 px target. The reordering under Order is shown whole, scrolling past
six lines - clamped at three, it cut off what Use reordered writes - and a
version's tick under Compare outlines the box holding its text while it is
pointed at or focused (`data-version`).

The panel's own messages - a finding's words no longer where the check found
them, an answer set aside or never asked for - are written to a status slot
under Ask the critic (`analysisMessage`: unhidden before it is written, never
rewritten with the same words, and `#error` where the critic's section is
hidden), between the findings and the answer they are about; in `#error` they
sat under Generate, a column away. The slot stays at full strength while the
rest of a stale panel is dimmed - what it says is mostly why the panel went
stale, and its amber at 0.55 opacity fell under 4.5:1 - so the dimming is on
its siblings, since a child cannot undo an ancestor's opacity. Those are
dimmed lightly, to 0.8 with the quiet text raised from `--muted` to
`--text-3` under it (about 5:1 on the panel), not to 0.55, where the notes
fell under 3:1: stale results are still the last answer and must still read,
and the amber badge, Check again and the dead buttons say they are stale
(U89). The Refine dialog's own check goes stale by the same rule, one
declaration for both; its lines had kept 0.55, about 2.5:1. The refusals
("Enter a prompt first.", "Wait for ...") stay in `#error`. A dead Use or
Apply says why in its title, as Run comparison does, rather than keeping the
offer it refuses. They stay `disabled`, not `aria-disabled` as Ask the critic
and Refine are: they are dead as a set, a dozen at a time, while the stale
badge and Check again - at full strength in the head - say why to every
reader, and what makes them stale is an edit somewhere else - a prompt box,
the references, the CFG - so focus is seldom on one when it dies. A critique
that fails leaves a row in the critic's block in the
server's words ("The critic did not answer: ...") and takes focus like an
answer. `#error` gets that sentence only where the critic's block is not on
show (the panel closed while the critic thought): it had it every time at
first, and the one failure was then read out twice, from two slots a column
apart from 1800 px. Nothing generic follows the reason: the server's messages say
what to do where anything can be done (shorten the prompt, install a
library), and "Ask again", added to every row at first, contradicted a
refusal that asking again only repeats. The failure is `critiqueFailure`,
never a `critiqueResult`, which Use and Apply read as an answer, and is
cleared by the next Ask, a check of other text and a cleared panel. The route
sends `CriticError`'s and `CriticUnavailable`'s messages without the class
name, which led every one of them; the log line keeps it.
While the critic runs, the seconds since the click count beside the line
(aria-hidden, outside its live region, so it is not read out every second),
and a first use's line says the download can take a few minutes: with nothing
moving, slow could not be told from stuck while every generation button was
held. A finding clicked, the cut link and a critic quote scroll the box to the
words they select (`scrollToSelection`, a hidden copy of the box laid out as
it is): `setSelectionRange` scrolls nothing in a box that has focus, and after
typing a long prompt the box sits at its end while the words are near its
start. The Order section's "Not checked" says the app checks word order only
in a tag checkpoint's comma-separated tags - of the app, not of "Check" in the
third person, and of a tag checkpoint's tags, since a comma tag prompt on FLUX
is not placed and a prose one on Animagine is. The comparison's preview counts
versions, the panel's word, not variants. `smoke-analysis.json` checks the
dock at 2560x1305 and 2048x1040 at 125%, and the stacking at 1400x900;
`smoke-analysis-tags.json` the dock on Animagine, where the image moves by
exactly the chips' extra lines.

**Open questions the build turned up**, for decisions rather than code:

- The tag autocomplete writes Danbooru names with A1111 escapes -
  `tags.display_tag` turns `(` into `\(` - to match the authors' own examples
  (Animagine's has them). Diffusers never unescapes, so CLIP reads
  `firefly \ ( honkai : star rail \ )`, ten tokens where the name is eight.
  The analyser does not flag what the app itself writes; whether the chips
  should write bare parentheses is undecided.
- Resolved 2026-09-28: schnell's guide said its T5 stops at 256 tokens,
  "about 180 words" - BFL's reference
  (`load_t5(..., max_length=256 if name == "flux-schnell" else 512)` in
  <https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py>) -
  while this app passes no `max_sequence_length`, so `FluxPipeline` encodes
  512 on schnell too and the encoder report says 511. The guide was reworded
  rather than the pipeline changed: "schnell's T5 encoder reads up to 512
  tokens in this app", and BFL's 256 is named as the reference in an
  `unsourced` sentence the critic is not given; the word rule stays out.
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

## The page script is twelve files, not one module

`web/app.js` reached 3,200 lines - every section of the page in one
file - and was split on 2026-09-10 along its own section comments into
ten scripts `index.html` loads in order: `state` (the element table, every
`let` the page shares, the fetch helper), `sizes`, `status`, `generation`,
`panel`, `gallery`, `sheet`, `lightbox`, `tags` and last `app`, the wiring
and the boot; the prompt analyser added an eleventh, `analysis` (the check,
the critic and the comparison), between `tags` and `app` on 2026-09-27, and
the Refine dialog a twelfth, `refine`, between `lightbox` and `tags` on
2026-10-04. They stay *classic* scripts rather than ES modules on purpose.
Classic scripts share one global scope: a top-level `let`, `const` or
function in any of them is visible to all the others, so nothing needed an
export or an import, and - the reason that matters - a smoke step's `eval`
(`tools/browser.mjs`) still reaches `els`, `galleryRecords`, `startJob`,
`analysisResult` and the rest by name, which a module's scope would hide. The
contract that makes the order safe is that eleven of the twelve only *declare*:
a function in `sizes.js` may call one from `analysis.js` because every call
happens after all twelve have loaded, from a handler or from `app.js`'s
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
acknowledged at the cell boundary, where a paused 512x512 grid held
2,046 MiB of the card (whole card, desktop included; a 1 MP fp8 denoise
reads 14,950 MiB); `reindex --from` over the real gallery
reproduces every hash and true CFG. Not verified: `LOCAL_DIFFUSION_DEVICE=cuda:1` (one
card here).
