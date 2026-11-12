# Anima Refine

**Status: stages 0 and 1 done (2026-11-11, 2026-11-12); card qualification
(stage 2) and release (stage 3) remain.** The public Refine is still held.
Matt selected Anima as the first family for `GEN#48` on 2026-11-11.
This closes the family choice in the [peer review](review-2026-11-10-peers.md#r1-refine-on-the-families-that-have-none).
It does not qualify a Refine pass or authorize a GPU run. No capability,
profile, loader or generation code changed in stage 0.

The pass starts from a gallery image, fits it to the requested size through
the existing Refine path, VAE-encodes it, adds the selected schedule's noise
and denoises the remaining steps. Existing `refine`, `strength`, seed,
prompt, size and checkpoint fields carry its reproduction intent. The
ordinary source proof, replay, PNG and bundle machinery remains applicable.

## Stage 0: source feasibility, completed

The installed version is Diffusers **0.42.1**, the version in `uv.lock`.
Its Anima implementation already contains image-to-image blocks. A new
callback-capable app interface can reuse those blocks; a copied denoising
pipeline or a custom noise approximation is unnecessary.

The inspected files are under
`.venv/Lib/site-packages/diffusers/modular_pipelines/anima/`:

| Source | Evidence in the installed implementation |
|---|---|
| `modular_blocks_anima.py:129` | `AnimaImg2ImgCoreDenoiseStep` sequences text conditioning, text/image input, strength-sliced timesteps, noisy source latents and the existing denoising loop. |
| `encoders.py:283,326` | `encode_vae_image` and `AnimaImg2ImgVaeEncoderStep` preprocess the image, encode `B,C,1,H,W` latents, and normalize them with the VAE's `latents_mean`/`latents_std`. The encoder uses posterior **mode**, its default `sample_mode="argmax"`, rather than sampling. |
| `before_denoise.py:493,553` | `get_timesteps` floors the skipped step count, slices the full scheduler timesteps and sets its begin index; the image-to-image timestep block first builds the full linear-sigma schedule on the supplied scheduler. |
| `before_denoise.py:635` | `AnimaImg2ImgPrepareLatentsStep` draws float32 noise using the supplied generator and calls the scheduler's `scale_noise` at the first sliced timestep. |
| `denoise.py:169` | `AnimaDenoiseLoopWrapper` runs the supplied timestep tail; the app's existing completed-step callback extension can be installed in this core too. |

The current [Anima adapter](../../backend/backends/anima_pipeline.py)
registers compatible components, but calls `AnimaCoreDenoiseStep` and
exposes neither `image` nor `strength`. Its Aesthetic and Preview interfaces
share this implementation. The generic [sibling resolver](../../backend/refine.py)
intentionally accepts only exact Diffusers-exported pipeline classes, so
merely adding a locally named `Img2ImgPipeline` would not enable Refine.

### Integration details that must be explicit

- Keep Anima's fixed FlowMatch Euler shift 3 and disabled Mu. Its timestep
  slicing nevertheless uses **flow rounding**: eight steps at strength 0.4
  run four steps. `refine.start_index` currently uses `mu_supported` to
  distinguish flow from SDXL rounding and would report three. Represent the
  Anima rounding separately from Mu support, with all existing families'
  counts unchanged.
- The current start-sigma reader expects a dynamic shift calculation from
  a standard sibling's module. Anima uses its fixed scheduler and modular
  timestep block instead. Read the result from a scheduler copy through the
  same Anima block, preserving the loaded scheduler's shift, and test that
  the preview and actual first denoise timestep agree. Do not enable Mu to
  make this path fit the existing reader.
- Preserve posterior-mode encoding and latent normalization exactly. Keep
  the backend's one-call-per-image seed contract, even though this encoder
  does not consume randomness for the posterior mode.
- The wrapper must report the sliced `num_timesteps` before its first
  callback, so progress, previews and cancellation use the actual tail.
  Keep `torch.no_grad`, the fresh per-call guider and callback clearing in
  `finally`; run the existing loaded-owner hook cleanup after a pass.
- Construct a sibling over the loaded components with `refine.build`,
  without `from_pipe`, casts, new downloads or new loader placement. The
  VAE is used before and after the transformer in a pass; shared modules
  do not establish equivalent offload residency or memory peaks.
- Keep the generic sibling identity check for every unrelated pipeline.
  Add a deliberate exact-class Anima extension only at release, verifying
  `REQUIRED` and every applicable `CARRIED` parameter as the ordinary path
  does. A local class merely borrowing an Anima name must remain refused.

### CPU probe retained as evidence

On 2026-11-11 a one-off CPU-only probe instantiated the installed
`AnimaImg2ImgSetTimestepsStep` and `AnimaImg2ImgPrepareLatentsStep` with a real
`FlowMatchEulerDiscreteScheduler(shift=3.0)`, `PipelineState`, a CPU generator
and tiny constant source/noise tensors. It loaded no learned component and
made no download. It asserted the slice count and that the noised latents
equal `(1 - sigma) * source + sigma * noise`, reading sigma from the real
scheduler at its begin index. Exit code 0:

| Full steps | Strength | Begin index | Steps run | Start sigma |
|---|---|---|---|---|
| 8 | 0.4 | 4 | 4 | 0.75 |
| 8 | 1.0 | 0 | 8 | 1.0 |
| 30 | 0.4 | 18 | 12 | 0.666667 |
| 30 | 0.1 | 27 | 3 | 0.25 |
| 8 | 0.125 | 7 | 1 | 0.3 |

The source SHA-256 values below identify the inspected installed files.
They are evidence, not runtime pins or a replacement for `uv.lock`:

| File | SHA-256 |
|---|---|
| `modular_blocks_anima.py` | `7e5cf39c557b2410d8cc4939cdc0011bb6eb4c9d0d75e2277023234eb6388bd7` |
| `before_denoise.py` | `e33eb4d118d6ba4d719174c01ebdcc991b5a9b17f742097df57737a8f4d149d9` |
| `encoders.py` | `fe009f435371edd0a3f65bf2f3432103839ad5fc2bb4b87d5b74a357fd8ac8a4` |
| `denoise.py` | `d94362491bf634ced936a2a4c2d3fd3ae1dfce462b1e8c0aa1e9fb39a4e6c0b3` |
| `decoders.py` | `0191bca50fa59f0cfb01a294ad969a9aec8b58e60812b7a0a2ef881398799406` |

This proves a usable source-level route and the schedule/noise arithmetic.
It does not exercise a real VAE, quantized denoiser, image quality or GPU
memory. The probe emitted the environment's existing torchao import warning;
no quantized load was attempted or qualified.

## Remaining stages and completion criteria

### 1. Private wrapper and CPU qualification

The next bounded slice is a private callback-capable Anima image-to-image
wrapper over the upstream VAE encoder, image-to-image core and decoder.
Keep the public resolver unregistered and Anima profiles' Refine answer
false while this is being built. Test the wrapper directly with tiny CPU
components, extending the existing Anima pipeline test fixtures with an
encoder whose posterior mode is distinguishable from a sampled posterior.

Verify the source affects the result; real installed scheduler sigmas and
begin index; fractional-strength boundaries; VAE normalization; positive
and negative conditioning at CFG 1 and above; exact component identity and
no casts/downloads; one-image seed reproduction; callback progress; and
cleanup after success, exception and cancellation. Check the preview on a
scheduler copy without changing the live scheduler. Existing direct Anima
samples from the CPU fixtures must retain their pixels and noise stream.

Exercise all three current local classes structurally. Turbo uses the
existing eight-step/CFG-1 defaults; Aesthetic and Preview use their existing
30-step/CFG-4 defaults. This shared source does not qualify all three
checkpoint weights, or make their Refine strength/size recommendations equal.
Inpaint, outpaint, references, user LoRAs and Nunchaku stay outside this slice.

Completion: focused Anima/refine/schedule/backend tests and applicable
lint/type checks pass, with public Refine still held. Record the commands
and results here. No new `refine_measured` result is invented.

#### Stage 1 result, 2026-11-12

Built on CPU only, with no weights, downloads or GPU:

- [`backend/backends/anima_refine.py`](../../backend/backends/anima_refine.py)
  holds `AnimaTurboImg2ImgPipeline`, `AnimaAestheticImg2ImgPipeline` and
  `AnimaPreviewImg2ImgPipeline` (the `IMG2IMG` map from each local text
  class). Each subclasses its family's text pipeline, keeps its exact
  `__init__` signature so `refine.build` hands over the loaded components,
  and runs the upstream `AnimaTextEncoderStep`, `AnimaImg2ImgVaeEncoderStep`,
  `AnimaImg2ImgCoreDenoiseStep` (its denoise sub-block replaced by the text
  pipeline's callback step) and `AnimaDecodeStep` directly, in the text
  call's `try`/`finally` (fresh guider enabled only above CFG 1, callback
  cleared, `maybe_free_model_hooks`). `num_timesteps` is the sliced count,
  set right after the `set_timesteps` block and so before the first
  callback. `strength` has no default; a strength outside (0, 1] or one
  whose slice is empty is refused before any component runs.
- Flow rounding is `anima_refine.start_index`/`steps_run`, the installed
  `get_timesteps` arithmetic in its own order. It is deliberately not yet
  wired into `refine.start_index`, which still answers None for Anima
  because the profiles keep `refine_supported` false; stage 3 carries it
  into that function (a rounding policy separate from `mu_supported`),
  progress, estimates and the duplicate advisory, leaving every existing
  family's count unchanged.
- `start_sigma(steps, strength)` runs the installed
  `AnimaImg2ImgSetTimestepsStep` on a scheduler built from the loaded one's
  config with its static shift and reads the sigma at the begin index; the
  live scheduler is not touched. Stage 3 connects it to the backend's
  `start_sigma` (Anima has no `START_RECIPES` entry and must not get Mu).
- Nothing imports the module outside its tests; `refine.sibling` still
  answers None for all three text classes and their image-to-image classes.

[`tests/test_anima_refine.py`](../../tests/test_anima_refine.py) extends the
existing Anima CPU fixtures with a VAE whose posterior sample differs from
its mode and whose latents mean/std are not the identity, and runs every
pass test against all three classes built through `refine.build`. It checks:
the resolver and the three profiles still refuse; the call takes `REQUIRED`
and every `CARRIED` parameter its text class takes, with Turbo's 8/CFG 1 and
Aesthetic/Preview's 30/CFG 4 defaults; the stage 0 probe table against the
installed block; `start_index`/`steps_run` against the block for ten step
counts by 108 strengths (fractional boundaries included), and that the SDXL
branch would answer three of eight where Anima runs four; the start sigma
equals the first callback's begin-index sigma and timestep, with the live
scheduler's timesteps, sigmas, begin index and shift unchanged; the first
denoiser input is exactly `sigma * noise + (1 - sigma) * normalised mode`,
the noise the generator's first draw; two sources give different pictures;
one image per call reproduces its seed (scalar or one-element list) and
another seed differs; the negative prompt is encoded and applied only above
CFG 1 (CFG 0, 1 and 4, against the first Euler step's arithmetic); every
component is the loaded pipeline's object, with no module or pipeline
`.to`, Hub download, dtype or device change; a direct text render keeps its
pixels and noise stream after a pass; the callback sees the sliced count,
scheduler step index and timesteps; cleanup after success, an exception and
a cancel (no decode, no "Error in block" log, callback cleared, the owner's
real CPU offload hooks rebuilt), and a clean pass after the cancel.

Commands and results, 2026-11-12 (CPU, `CUDA_VISIBLE_DEVICES` empty):
`uv run --group dev pytest tests/test_anima_refine.py tests/test_anima_pipeline.py`
78 passed; `uv run --group dev ruff check .` clean;
`uv run --group dev python tools/typecheck.py` 0 errors; the full suite
`uv run --group dev pytest -n 6` 9367 passed, 254 skipped. No smoke covers
this: nothing in the page or the HTTP answers changed.

Not done here and not claimed: a real VAE or quantized denoiser, image
quality, memory, speed, the backend's `_refine` loop over these classes,
inpaint, outpaint, references, user LoRAs and Nunchaku. The next step is
stage 2 on the card, separately authorized.

### 2. Card qualification, separately authorized

Use the existing [card guard](plan-card-harness.md#card-guard) and current
[measured Anima loading limits](../verified.md) on a quiet RTX 5080. Start
with the bare official Turbo selection (`circlestone-labs/Anima`), whose
existing identity is pinned Turbo v1.1, at fp8/model offload on Windows.
Use one load and the private runner; do not open an unqualified public route
to conduct the experiment.

First retain a 1024x1024 direct baseline and source. At that same size,
compare a small strength set (0.125, 0.25, 0.4) with the source, recording
the actual start sigma and step count. These are trial values, not an
author's recommendation or a chosen default. Strength 1 is a CPU schedule
control, not evidence that the picture should be preserved at that setting.
Only after the same-size pass fits, try one 1.25x target within measured
headroom; go no larger when the guard, RAM or card headroom refuses it.

Retain GPU allocated/reserved/whole-card peaks, process RAM and commit
headroom, load and pass time, image and metadata hashes, and visual judgement
of composition retention and added detail. Check repeat seed, a second image
of a batch against its solo seed, cancellation, final resident memory and
the original direct baseline after success and cancellation. A spill does
not qualify a size. Choose the opening strength and largest supported
measured size from this evidence, rather than copying SDXL's 0.4 or its
size line.

Aesthetic and the 40-layer Preview need their own one-load qualification
before either is released, even if the private wrapper is shared. Separate
sessions preserve the existing checkpoint identities and memory boundaries.
No 3070, WSL or alternative quantization/offload support is inferred.

Completion: dated receipts and measured limits in `verified.md`, bounded by
the exact checkpoint, loader, platform, size and dependency version tested.

### 3. Release the measured support through the existing flows

Register the exact qualified Anima classes deliberately, update the matching
profiles and hints with measured strength/size results, and keep any
unqualified variant held. Preserve before-load/live capability agreement.
Carry the new rounding policy through initial progress, estimates and the
duplicate advisory; retain requested strength in keys and metadata.

Qualify Refine, Mass refine, Then refine, rerun/recovery and the CLI through
their existing paths. Add a fake-browser check for the newly available
Refine controls, preview, size/strength holds and saved source/strength;
reuse the existing Anima smokes for unchanged unsupported features. Update
the [rules](../rules.md), [design notes](../design-notes.md#the-refine-pass),
usage and catalogue wording, regenerating any generated capability table
through its owner tool. Run the relevant automated gates and record their
bounded results here. Remove `GEN#48` only when its implemented and measured
release is complete; a selected family or private wrapper does not close it.
