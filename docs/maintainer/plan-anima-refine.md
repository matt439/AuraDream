# Anima Refine

**Status: stages 0 and 1 done (2026-11-11, 2026-11-12); stage 2's Turbo
experiment measured and agent-reviewed at 1024 square on Windows fp8/model
(2026-11-16), followed by independent Aesthetic and Preview trials on
2026-11-17. Bounded Turbo passed integrated public API/CLI-via-server
acceptance. The measured Aesthetic/Preview source extension passed its full
CPU/browser gates; their integrated API and standalone CLI checks remain.**
Matt selected Anima as the first family for `GEN#48` on 2026-11-11.
This closes the family choice in the [peer review](review-2026-11-10-peers.md#r1-refine-on-the-families-that-have-none).
That family choice alone did not qualify a Refine pass or authorize a GPU run. No capability,
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
  live scheduler is not touched. It refuses, with the call's own
  `ValueError` (`refuse_empty`), every strength the call refuses, so it
  never names the trailing 0.0 of an empty slice. Stage 3 connects it to the backend's
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
counts by 108 strengths (fractional boundaries included), and that
`refine.start_index`'s non-Mu branch, called for Animagine, answers three of
eight where Anima runs four; the start sigma equals the first callback's
begin-index sigma and timestep, with the live scheduler's timesteps, sigmas,
begin index and shift unchanged, a runtime `set_shift(5.0)` on the live
scheduler reaches the copy (the pass and the reading agree on shift 5's
sigma), and strengths the call refuses are refused by the reading too; the first
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


## Offline session pack (2026-11-15)

`tools/anima_refine_pack.py` makes stage 2 repeatable without opening the
public route. It has no runner, backend import, counters or network:

```powershell
uv run python tools/anima_refine_pack.py plan > "$env:TEMP/anima-refine-plan.json"
# After a separately authorized session, save the completed receipt object:
uv run python tools/anima_refine_pack.py check "$env:TEMP/anima-refine-receipt.json"
```

The JSON contains `receipt`, a **blank template**, not a measurement. Save
that object alone for `check`. All required measurements begin as null and
cases as `not-run`. Exit 1 means missing/inconsistent evidence; exit 0 means
only that the recorded fields are complete and internally consistent.
The checker does not inspect external receipt files, authenticate hashes,
judge pictures or qualify a model. It cannot enable public Refine.

### Fixed one-load Turbo sequence

Windows, official Turbo v1.1, fp8/model offload, eight steps, CFG 1, no Mu,
negative prompt, LoRA or references. The prompt is a synthetic composition
probe, not an author's example. It fixes a subject, coloured clothing and a
second object so source retention is easy to judge. Seeds are fixed before
viewing results. The sequence has a direct 1024-square baseline; strengths
0.125, 0.25 and 0.4; a repeated 0.4; consecutive batch members at seed and
seed+1 followed by solo seed+1; cancellation after the first completed step;
a recovery pass; a direct baseline again; and one optional 1280-square pass.
All passes use the original baseline pixels, never a previous refined image.
The batch is the app's consecutive one-image-call contract, not an unqualified
multi-image call. The direct-after check follows both success and cancellation.

The tail counts 1/2/4 and shift-3 start sigmas 0.3/0.5/0.75 are checked
against the existing private wrapper and installed scheduler in a CPU test.
The receipt must record the *actual* callback counts and sigmas, not copy
these expected figures. Cancel must record no decoded output; repeat,
batch/solo, recovery and direct-after comparisons use decoded-pixel hashes.
Hash definitions: `pixel_sha256` is the full SHA-256 of decoded RGB bytes
(at the recorded dimensions); `png_sha256` hashes saved PNG bytes;
`metadata_sha256` hashes the exact retained UTF-8 metadata receipt bytes.
`source_sha256` uses the same full RGB pixel hash as the baseline. These are
full proof hashes, not the gallery tool's abbreviated 16-character display.
Retain files, not just the digests. A source with a different size or mode
needs its own explicitly recorded conversion before hashing and use.

### Preconditions and bounded attempt

This pack is preparation, not permission for a GPU run. The future private
runner must wire the existing card guard before loading, retain its raw
samples and effective limits, and confirm a quiet RTX 5080 session. The
existing limits remain owned by `tools/card_guard.py`; do not copy or loosen
them here. Choose and record a checkpoint-specific load RAM minimum and
busy-card threshold from the current measured Anima record and live baseline.
State the RAM budget and reserve the machine before starting. Set a 30-minute
session cap, a ten-minute load cap and two-minute pass cap for this *attempt*;
these are stop budgets, not measured performance claims. A timeout is retained
as an incomplete session, never silently retried or converted to a pass.

Stop on OOM, spill/shared-memory trip, low free RAM or commit headroom,
missing required counters or source proof, cancellation that does not stop,
or the wall-time cap. No later case follows a safety trip. Check final
resident memory and guard cleanup even on failure. Refuse the optional
1280 step unless same-size evidence and guard projection permit it; record
`skipped` and the reason. A skipped enlargement establishes no supported
larger size. No attempted size above 1280, no second checkpoint, and no
3070/WSL/alternative-loader inference belongs in this session.

Each completed image needs elapsed time, allocated/reserved/whole-card peak,
shared-memory rises, process RAM peak, free-RAM and commit minima, resident
allocated memory, raw output/metadata hashes, and composition/detail/artefact
notes. Keep the session's version/loader, load receipt, guard log, and final
residency receipt. Manual review of those sources and pictures is still
required before any dated result goes into `verified.md` or stage 3 starts.

### Quality gate and CPU evidence

Review the baseline and each completed case side by side at matched display
size and at 100% crops. Record whether the woman, blue coat, red bicycle and
park remain identifiable; compare face/hands, object boundaries, textures
and colour stability. Reject visible seams, broken geometry, smeared detail
or an unexplained composition replacement. Higher strength need not look
better. A strength is a candidate only if the reviewer explicitly records
its benefit and trade-off; a complete receipt alone never makes it one.
If the baseline is unusable, preserve the failed probe rather than silently
changing prompt/seed midway through the one-load session. The optional
upscale must show useful detail rather than merely a larger saved canvas.

CPU verification: `pytest -q tests/test_anima_refine_pack.py
 tests/test_anima_refine.py tests/test_card_guard.py tests/test_docs_maps.py
 --timeout=120` passed 231 tests, with one existing Starlette deprecation
warning. `ruff check .` passed. The repository's pinned Pyright check passed
203 files with zero errors/warnings using a temporary config pointing at
the original checkout's existing virtualenv (no dependency installation).
The wider CPU regression run (`pytest -n 2 -q --timeout=120`) passed
10,034 tests, skipped 325, and emitted 83 warnings in 484.56 seconds. It
started before the final ten oversized-number metric cases were added;
the final focused 231-test run includes those cases and the final checker.
Tests used `CUDA_VISIBLE_DEVICES=''`; no weights, live gallery or GPU run.
Nothing in this pack is evidence of real VAE/denoiser quality, GPU fit or
speed.

### Private runner, 2026-11-16

`tools/anima_refine_card.py` runs this fixed Turbo sequence only in a separately
authorized quiet Windows RTX 5080 session. It never registers public Refine,
loads another checkpoint, opens a server or reads the production gallery.
The required RAM admission is a conservative session budget: the historical
Turbo record does not give a measured host-RAM peak. For the overnight attempt
the budget is 40 GiB free before the load and a 2,000 MiB idle-card ceiling;
the ordinary 4 GiB free-RAM/commit floors and device/shared limits stay intact.

```powershell
uv run --no-sync python tools/anima_refine_card.py --output work/anima-refine-2026-11-16 --need-ram-gib 40 --busy-card-mib 2000
```

The evidence directory must be new. A waiting child joins a kill-on-close job
before admission; only a successful mandatory-counter sample lets it import
the backend and load. The parent uses the existing guard sampler and decisions,
requires held PDH counters rather than the slower typeperf fallback, retains
every quarter-second raw sample and enforces the fixed phase/session caps.
An independent watchdog kills the child on those caps or a sampler heartbeat
gap over two seconds. CUDA visibility/order overrides are refused so CUDA 0
cannot silently differ from the guarded physical GPU 0.

The child forces offline Hub/Transformers operation, a scratch gallery and the
frozen loader choices; a missing converted Turbo tree refuses before any load,
conversion or download. Its `keep_awake.hold_while` scope requires an actual
Windows system power request before importing/loading weights, covers all
load/cases/unload work and releases in `finally` before reporting a failed
worker to the parent. `keep-awake.json` retains actual acquired/held/released
state and failures; a forced process exit closes Windows' handle. This uses
the CLI primitive and never reads the app's event-loop providers.
It records actual callback counts/sigmas, load/pass
allocator measurements, full RGB/PNG/metadata hashes and supporting source
hashes. Each phase is durably acknowledged before the next begins. Failure
retains the incomplete receipt and final after-exit device/host reading; no
later case follows a trip. The sibling's loaded owner frees model hooks after
each call, including cancellation, and sibling references go before unload.

The optional 1280-square tail is omitted by default. Even with
`--include-upscale`, a single measured area cannot supply a projection slope:
the runner records a skip and proceeds to cleanup. A later enlargement needs
its own defensible measured headroom/projection evidence. No guessed slope
qualifies a larger size.

`receipt.json` keeps visual reviews null until the actual retained images are
inspected. `anima_refine_pack.py check` should therefore refuse an otherwise
successful unreviewed run. Exit zero means the supervised sequence completed;
it neither authenticates pictures nor chooses a strength or enables release.

#### First attempt stopped at Windows IPC, 2026-11-16

The authorized overnight attempt in `work/anima-refine-2026-11-16/` retained
one Turbo load, the direct baseline and three same-size strength passes. The
parent then stopped the child because replacing the polled `ack.json` failed
with Windows error 5. This was a runner publication defect, not a card-limit
trip. The original incomplete receipt, source/images, raw samples and hashes
remain untouched. Repeat, batch/solo, cancellation/recovery, direct-after and
ordinary unload were not reached; visual reviews remain unset. These partial
observations do not qualify Anima Refine or establish an opening strength.

| Case | Completed steps / start sigma | Seconds | Allocated / reserved peak MiB | Whole-card peak MiB |
|---|---|---|---|---|
| Direct baseline, 1024 square | 8 / not applicable | 19.670 | 2223.37 / 2392 | 4145 |
| Strength 0.125 | 1 / 0.300000012 | 3.943 | 2470.98 / 2640 | 4393 |
| Strength 0.25 | 2 / 0.5 | 3.063 | 2471.48 / 2640 | 4396 |
| Strength 0.4 | 4 / 0.75 | 3.798 | 2471.48 / 2640 | 4391 |

The cached load took 6.406 seconds and sampled 7.984 GiB maximum process
RAM, with 35.896 GiB free RAM and 41.628 GiB commit headroom at its minima.
Across those completed image cases the shared-process rise was 82 MiB and
the adapter rise at most 84.715 MiB. The guard retained 185 samples with a
maximum achieved interval of 0.250649 seconds and no limit trip. After the
supervisor killed the child, its separate after-exit reading was 1,445 MiB
whole-card, 43.372 GiB free RAM and 51.436 GiB commit headroom. The worker's
power receipt retains its last acquired/held state, because the forced exit
could not write a normal release receipt; Windows closes its process handle.

The repair publishes immutable sequence-specific request/result/failure files
and sequence/action-specific acknowledgement files. A poller never opens a
target that a later handoff must replace. Mutable evidence receipts retry
only Windows access/sharing errors 5, 32 and 33, at most eight attempts and
0.28 seconds of delay; a persistent or unrelated failure still stops the
attempt. A real CPU-only Windows subprocess regression holds all thirteen
begin acknowledgements open without delete sharing, proves replacement is
refused, and completes the actual parent/child handshakes. It imports no
backend or torch and takes no power request or GPU counter reading. The
repaired runner needs a new evidence directory for a fresh one-load attempt;
it never resumes or rewrites the failed session into a successful one.

#### Turbo one-load result, 2026-11-16

The repaired runner completed a new one-load session in
`work/anima-refine-2026-11-16-ipc-fix/`, exit 0. Its compact committed
[measurement receipt](anima-refine-turbo-2026-11-16.json) retains the exact
recipe, source and output proofs, guard/load/case/final measurements,
agent visual review and limits. The original failed session above remains
unchanged. No public resolver, profile or generation flow was opened by this
experiment; stage 3 remains separate.

**Measured envelope only:** official Turbo v1.1
(`circlestone-labs/Anima`, revision
`f973fc41ec7545364ac9776c2440285f43ff2a30`), the pinned shared component tree,
Windows 11 build 26300, RTX 5080 16 GB, driver 616.92, torch 2.13.0+cu130,
diffusers 0.42.1 and torchao 0.18.0; fp8, model offload, Eager/native,
FlowMatch Euler shift 3, one image per CPU-generator call. Every image was
1024x1024 RGB, eight full steps, CFG 1, no Mu, LoRA or reference, the frozen
synthetic prompt and seeds 4471/4472. Negative was empty and unused at CFG 1.
1280 was skipped. There is no measured larger size, other aspect, Aesthetic,
Preview, WSL, 3070 or alternative-loader Refine result here.

| Case | Completed steps / start sigma | Seconds | Allocated / reserved peak MiB | Whole-card peak MiB |
|---|---|---|---|---|
| Direct baseline | 8 / not applicable | 14.168 | 2223.37 / 2392 | 4146 |
| Strength 0.125 | 1 / 0.300000012 | 2.733 | 2470.98 / 2640 | 4394 |
| Strength 0.25 | 2 / 0.5 | 3.088 | 2471.48 / 2640 | 4395 |
| Strength 0.4 and same-seed repeats/recovery | 4 / 0.75 | 3.928-4.712 | 2471.48 / 2640 | 4394-4398 |
| Batch second / solo second, seed 4472 | 4 / 0.75 | 4.301 / 4.235 | 2471.48 / 2640 | 4393 / 4392 |
| Cancel after first completed step | 1 / 0.75 | 3.410 | 2470.98 / 2640 | 4394 |
| Direct again after passes/cancel | 8 / not applicable | 5.584 | 2223.37 / 2392 | 4147 |

The direct baseline's full RGB hash is the direct-after hash. Strength 0.4,
its repeat, the batch's first image and post-cancel recovery have one full
RGB hash; batch second and solo seed 4472 have another. Every Refine and
cancel case proves the unchanged original baseline RGB as its source.
The cancel unwound at callback one with no decoded PNG or metadata output.
All ten PNG, decoded RGB and exact metadata-byte hashes were recomputed;
there are five unique RGB outputs. The raw direct-after record also carries
the baseline hash as available runner context even though its request source
is null and the text pipeline receives no image; that annotation is retained
and explained, not rewritten into an image-to-image request.

The cached load took 4.389 seconds and sampled 7.768 GiB maximum process RAM.
Session minima were 35.413 GiB free RAM and 38.765 GiB commit headroom;
shared-process and adapter rises peaked at 82 and 84.109 MiB. All required
counters were present, all 296 sampled verdicts were `ok`, the requested
period was 0.25 seconds and the largest achieved interval was 0.294277 seconds.
The stated 40 GiB load admission and 2,000 MiB idle-card ceiling were kept,
along with the ordinary 4 GiB RAM/commit floors and card/shared limits.
Normal cleanup recorded 32 MiB allocated and reserved before child exit,
not zero; after exit the whole card read 1,448 MiB against 1,445 at preflight.
The Windows power request was acquired and released with no holder error.

**Agent visual judgement:** the root agent inspected all five unique outputs
at 1024 display size; duplicates inherit findings only after full RGB equality.
The woman, blue coat, red bicycle and leafy park stay identifiable, with the
first seed's composition retained and no observed composite seam. The baseline
already has broken fine spokes/fingers and unwanted caption or watermark-like
generated text. Strength 0.125 mostly preserves those pixels with little clear
benefit. Strength 0.25 cleans some wheel/outline work with small line changes
while retaining face, pose and objects; anatomy and caption defects remain.
Strength 0.4 is sharper and more saturated but redraws more face/background/foot
detail and drifts the unwanted caption; the second seed also changes buttons,
shoes and hand detail without identity replacement. The provisional opening
candidate is **0.25 for the exact measured recipe**, an observation on this
synthetic source, not an author recommendation or quality guarantee. Matt has
not performed this visual review himself.

The successful raw `receipt.json` received only those visual-review strings.
Its byte-identical `receipt-unreviewed.json` and full pre-review SHA-256 remain
beside it; `visual-review.json` records who/when, viewed unique files, hash
links and notes. `anima_refine_pack.py check` then returned `receipt_complete:
true`, no problems, exit 0; that completeness check does not enable support.
Before the retry, 60 runner CPU tests passed, including thirteen real Windows
locked-reader handoffs; the runner/pack/doc-map/section set passed 178 tests,
Ruff passed and pinned Pyright checked 215 sources with zero errors.

**Next:** stage 3 may release only deliberately bounded Turbo support from
this evidence, with the ordinary source/reproduction paths and the listed
automated/browser checks. Aesthetic and Preview stay held until their own
one-load sessions. No enlargement recommendation follows from the skipped
1280 case. `GEN#48` remains open until its measured public release is complete.

#### Integrated CPU gate, 2026-11-17

After the full-suite Windows reader failure, the bounded read retry and
deterministic locked-reader/CRT regression passed all 72 runner tests.
The complete Windows CPU suite then passed **10,431 tests, 325 skipped**,
85 warnings, exit 0 in 367.16 seconds (`pytest -n 4 --timeout=600 -q`;
`work/overnight-2026-11-17-cpu.log`). The skips include uncached real
tokenizers and platform-specific cases. Ruff passed and pinned Pyright
checked 215 sources with zero errors/warnings. This gate checks the final
IO hardening as well as the private attention/FBC compatibility slices;
the successful card receipt retains its earlier exact runner hash rather
than pretending the later IO changes were used for that render.

#### Windows reader race found by the full CPU gate, 2026-11-17

The wider CPU run then exposed a second IPC gap: its real Windows child
exited after handoff nine. The parent retained the incomplete fixture receipt,
but the old fixture did not save the child's stderr as a file. An independent
private-card CPU probe reproduced access denial while reading a newly
published immutable message, including CRT `PermissionError(EACCES)` without
a `winerror`. Immutable names prevent replacement conflicts; they do not
guarantee that an exclusive scanner has already released a new file.

`read_json` now retries only Windows errors 5/32/33 or the Windows-only CRT
form above, at most eight attempts and 0.28 seconds of delay. Missing files
still answer immediately; persistent denial, unrelated I/O and malformed JSON
remain errors. Deterministic tests cover transient recovery, the finite stop
budget and those refusal boundaries. The actual thirteen-handoff Windows
fixture now also injects a CRT read denial at handoff nine and must recover;
its complete child stdout/stderr remain in fixture artifacts on success or
failure. A focused pass alone does not resolve the original full-gate result:
the full CPU gate must pass again after this repair. No card evidence or its
recorded historical tool-source hashes changed, and this I/O-only repair
requires no new GPU measurement.

## Bounded Turbo integration, 2026-11-17

Stage 3's first source slice registers only the exact Turbo counterpart.
`refine_qualification` freezes the supplied-fact envelope: exact official
Turbo id, RTX 5080, Windows CUDA 0, fp8/model offload, Eager/Native,
diffusers 0.42.1, 1024-square original source and output, 8 full steps and
CFG 1. Only the measured strengths 0.125/0.25/0.4 are admitted; the provisional
opening is 0.25. Unknown hardware, another loader/host, another strength,
enlargement, Aesthetic and Preview remain held with a specific reason.

Host discovery remains with existing card queries and explicit model loading.
The new qualification reader never imports a framework or probes a device;
successful Anima loading refreshes the name after preparation, so an earlier
unknown cache cannot hold it forever. Native checks compare startup/current
environment, already-imported defaults and the exact active Native entry.
Every Turbo forward verifies them before and after use without retry or
environment mutation. Ordinary text-to-image and other families keep their
existing behavior.

All expanded requests are checked before shared source verdicts. Queue
admission refuses the whole batch's execution/composer and pure qualification
errors before source proofs or mask work. Actual gallery PNGs must match the
1024-square record and its existing RGB provenance; queued changes are checked
again at each cell and decoded through one handle before use. The normal
source/key/PNG/bundle machinery is unchanged. Then refine uses its base's
dimensions for the future source, never the placeholder gallery id; Finish
and the CLI check those same future passes before queueing or loading.

The explicit flow-rounding policy feeds central progress/estimate/duplicate
logic without enabling Mu or changing requested-strength keys. Sigma advice
uses the existing Anima scheduler-copy reader. The backend frees the loaded
owner after each Turbo image call, matching the private serial protocol.
Status supplies typed measured suggestions and held reasons, while previews
return an exact common per-image step count. Polling preserves typed values
and updates a changed refusal even when capability remains false.

The source checks caught and fixed a same-size PIL ownership error before
card use: `refine.fitted` can return its input, so the returned image must stay
open. Independent review also identified queue-order, retained Then refine,
CLI loader-preflight and Native-drift gaps; regression checks cover each.
The browser fixture's immediate close/reopen also raced the queued native
dialog close event. It now waits for the real `refineSource === null`
transition before opening another source, with no fixed sleep.

### Remaining integrated card acceptance

CPU and fake-browser results do not qualify the newly connected real-backend
path. Run a separate guarded real-server session after integration,
with a scratch gallery and the retained baseline RGB. At matched seed 4471,
the 0.25 pass must reproduce the measured `726e7968…` pixels. A serial two-image
0.4 call must match `a3d6d770…` / `fc78687d…`, and a direct-then-Refine at
scale 1 / 0.25 must reproduce the baseline/pass pair. Observe actual progress,
cancel/recovery, saved metadata and owner-hook residency; keep held-path
checks and ordinary generation. No larger target belongs in that acceptance.

`GEN#48` remains tracked for this acceptance and independent Aesthetic/Preview
one-load qualifications. The original trial receipts and source hashes stay
historical evidence, not a claim that the integrated public path was measured.

### Local integration checks

The isolated worktree used the original checkout's existing environment,
without dependency changes or a second virtualenv.

- Full source suite: `python -m pytest -n 4 --durations=20 -q --tb=short
  --timeout=600` — **10,665 passed, 325 skipped**, 85 warnings, exit 0,
  334.33 seconds. The earlier full pass found two stale test expectations:
  the new accessibility-smoke registration and Turbo's old No capability.
  Both were corrected without weakening the audit and passed 57 focused
  checks before the fresh full run.
- Ruff passed. Pinned Pyright checked **217 backend/tool sources** with zero
  diagnostics, using a temporary relative-scope config pointed at the existing
  environment. An initial absolute-scope experiment was rejected as validation
  when Pyright reported that it ignored those paths; the corrected scope is
  the result above.
- The new API matrix includes 44 checks for all actual grid combinations,
  copied-request order, sources changed in size or pixels, hardware/loader
  changes, per-image Mass refine, Then refine stand-ins and retained Finish.
  Actual backend CPU tests cover private-wrapper equality, batch/solo seeds,
  per-image hook cleanup, cancellation/error recovery, living PIL ownership,
  Native drift and cached-unknown hardware refresh. The import-cold matrix
  passed 109 isolated checks and Native observations passed 18 isolated checks.
- Four distinct Anima browser smokes passed: Turbo, Aesthetic, Preview and the
  new measured Refine row. The final new row passed **103 steps** in 9.41 s.
  Its accessibility watch placement and terminal audit match the registered
  contract. Desktop 1400×900, narrow 800×900 and emulated phone 393×852 valid
  and held screenshots were inspected, with stable hit tests and readable
  choices/reasons. The phone fixture deliberately scrolls the modal to reveal
  Strength or the held reason; this is not physical-device acceptance.

Browser evidence is retained at
`%TEMP%/auradream-smoke-artifacts-nnrwrl38/summary.json` (final Refine row,
seven captures) and `auradream-smoke-artifacts-2p2zc4d_/summary.json` (the
other three passing rows and the original diagnosed fixture failure).
Owned fake/browser processes were cleaned up; the final row's port 19960
had no listener afterwards.

These runs used no checkpoint weights or model inference. The broad suite
did initialize one CUDA metadata/allocator context: Windows PowerShell removes
an environment variable assigned an empty string, so that assignment did not
hide the card. Existing CPU-stub peak-accounting paths can query CUDA when it
is visible. That worker exited with the suite. Future whole-suite commands
use `CUDA_VISIBLE_DEVICES=-1`; the strict installed CPU probes independently
forbid CUDA initialization, compilation and network access. No integrated
GPU fit, pixels, memory or speed result is claimed by these checks.

#### Aesthetic one-load result, 2026-11-17

The private Aesthetic v1.1 experiment completed one load, ten saved images and
one cancellation, exit 0, in `work/anima-aesthetic-refine-2026-11-17/`.
The committed measurement receipt is
`docs/maintainer/anima-refine-aesthetic-2026-11-17.json`. This is evidence for
the exact `circlestone-labs/Anima@aesthetic-v1.1` selection at revision
`594c27fea35648b87c86a9b4d5436a6024c820b5`, Windows 11 build 26300, RTX 5080,
driver 616.92, torch 2.13.0+cu130, diffusers 0.42.1 and torchao 0.18.0;
fp8/model offload, Eager/native, FlowMatch Euler shift 3, 1024x1024 RGB,
30 full steps, CFG 4, empty negative, no Mu/LoRA/reference. Seeds 4471 and
4472 ran as consecutive one-image CPU-generator calls over the immutable
direct baseline. No larger size or different aspect was attempted.

Strengths 0.125/0.25/0.4 actually ran 4/8/12 steps from begin indices
26/22/18 and sigmas 0.315789521/0.521739185/0.666666687, matching the
installed scheduler's CPU proof. Their first pass times were 5.131/7.229/
10.143 seconds; direct baseline/direct-after were 30.282/21.257 seconds.
The 0.4 repeat, batch first and post-cancel recovery are RGB-identical to
the first 0.4 pass; batch second equals solo seed 4472; direct-after equals
the baseline. Cancellation stopped at callback one without decoded output.
All ten PNG/full RGB/exact metadata hashes were recomputed; the ten images
have five unique RGB hashes, and every pass proves the same original source.

The cached load took 6.918 seconds and sampled 8.257 GiB peak process RAM.
Refine peaks reached 2472.98/2642 MiB allocated/reserved and 4982 MiB whole
card over a 2030 MiB preflight. Shared-process/adapter rises peaked at
82/81.719 MiB; session minima were 25.803 GiB free RAM and 18.663 GiB commit
headroom. The initial admission was 24 GiB free RAM and commit, with an
explicit 2500 MiB idle-card ceiling; continuous RAM/commit floors stayed
4 GiB, the process-RSS cap 16 GiB and the card/shared limits unchanged.
All 601 sampled verdicts were `ok`; requested period 0.25 seconds, largest
achieved interval 0.253483 seconds. Normal unload retained 32 MiB allocated
and reserved before exit, and the post-exit card read 2031 MiB. The Windows
power request was acquired and released without error.

The root agent inspected all five unique outputs. Woman, blue coat, red
bicycle and leafy background remain identifiable, with no observed seams
or composition replacement; fine bicycle/hand geometry was already imperfect.
At 0.125 pose/face are retained and some bicycle outlines are cleaner, at the
cost of flatter foliage and higher contrast. At seed 4471, 0.25/0.4 increasingly
posterize foliage and redraw small outlines without clear texture benefit.
Seed 4472 at 0.4 retains more leaf texture but changes face/hair, buttons and
shoes. The provisional conservative opening candidate is **0.125 for this
exact measured recipe**, not an author recipe, anatomy-improvement claim or
quality guarantee. Matt has not reviewed these images himself. The raw
receipt received only those visual-review strings; its byte-identical
`receipt-unreviewed.json` and review sidecar preserve the pre-review hash and
reviewer/time. Public resolver/profile/browser/CLI release remains separate;
Preview still requires its own conversion and one-load card session.

#### Preview preparation and one-load result, 2026-11-17

Preview required a separate offline CPU conversion before the private card
session. Its cached split file was exactly 5,843,204,206 bytes with SHA-256
`0b3020d1b906155f7eb30667622723e87160632c8c7a5f1c93bdce685f2a346d`;
the pinned shared snapshot was present. Normal
`anima_loading.resolve_model(PREVIEW_MODEL_ID, "fp8")` retained its locking
and atomic publication, with local-only Hub reads and refusal to remove any
old target/staging path. The successful CPU attempt took 10.583 seconds inside
the converter, 19.108 seconds under supervision, and sampled 7.555 GiB peak
RSS over 77 baseline/samples. CUDA stayed uninitialized, with zero backend
loads, inference calls or forbidden native/lazy/Tensor/Module CUDA attempts.
Final checks passed and the Windows power request was released.

The first CPU attempt failed before conversion/component loading: a
Transformers lazy Bloom import wrapped torchao's import-time CUDA capability
probe, which met the harness's hard lazy-init refusal. It did not establish a
missing dependency. A guarded import-only reproduction retained the complete
cause chain. The repaired private CPU context advertises CUDA unavailable/
zero devices with `CUDA_VISIBLE_DEVICES=-1`, preserving every hard CUDA and
inference refusal. Traceback/cause recording and late-failure outcome handling
were also corrected. The original failed folder remains unchanged, and the
successful conversion used a new evidence directory. Its configuration hashes
were subsequently matched exactly to the GPU load.

The private Preview v1 card experiment then completed one load, ten images
and one cancellation in `work/anima-preview-refine-2026-11-17/`, exit 0.
The combined committed receipt is
`docs/maintainer/anima-refine-preview-2026-11-17.json`. Its exact selection is
`Gazingstars123/Anima-2.9B`, revision
`9f9cb502dbae7a616c3cc5a530633427fe735665`, the 40-layer Preview and its own
conditioner; Windows 11 build 26300, RTX 5080, driver 616.92, torch
2.13.0+cu130, diffusers 0.42.1, torchao 0.18.0, fp8/model offload,
Eager/native, FlowMatch Euler shift 3, 1024x1024 RGB, 30 full steps, CFG 4,
empty negative and no Mu/LoRA/reference. Seeds 4471/4472 ran as consecutive
one-image CPU-generator calls over the immutable direct baseline.

Strengths 0.125/0.25/0.4 ran 4/8/12 steps from indices 26/22/18 and sigmas
0.315789521/0.521739185/0.666666687, matching the CPU schedule proof. Their
first pass times were 6.228/9.992/13.305 seconds; direct baseline/direct-after
were 30.634/29.872 seconds. The 0.4 repeat, batch first and post-cancel recovery
are RGB-identical to the first 0.4 pass; batch second equals solo seed 4472;
direct-after equals the baseline. Cancellation stopped at callback one without
decoded output. All ten PNG/full RGB/exact metadata hashes were recomputed;
there are five unique RGB outputs, and every pass proves the same source.

The GPU cached load took 6.440 seconds and sampled 10.399 GiB peak process
RAM. Refine peaks reached 3277.10/3446 MiB allocated/reserved and 5783 MiB
whole card over a 2022 MiB preflight. Shared-process/adapter rises peaked at
82/96.043 MiB; minimum free RAM/commit headroom were 23.814/18.222 GiB.
Initial admission stayed 24 GiB free RAM and commit, with an explicit
2500 MiB idle-card ceiling; continuous floors stayed 4 GiB, the process-RSS
cap 16 GiB and card/shared limits unchanged. All 723 sampled verdicts were
`ok`; the largest achieved interval was 0.250633 seconds at a requested
0.25-second period. Normal cleanup retained 32 MiB allocated and reserved
before exit; the post-exit card read 2028 MiB. Power hold/release passed.

The root agent inspected all five unique actual images. Subject, blue coat,
red bicycle and leafy park remain coherent against a soft painterly baseline;
the bicycle is inherently cropped and fine anatomy is imperfect. At 0.125,
subject/pose/face/bicycle/colours are preserved with sharper outlined/angular
foliage and changed texture. At seed 4471, 0.25/0.4 increasingly flatten and
saturate foliage and redraw small linework without clear texture benefit.
Seed 4472 at 0.4 retains painterly foliage but changes face/hair and slightly
changes basket/wheel geometry. No new composite seam or composition
replacement was observed. The provisional conservative opening is **0.125
for this exact measured recipe**, not author advice, a quality guarantee or
Matt acceptance. The original raw receipt bytes remain in
`receipt-unreviewed.json`; only visual-review strings were added to the live
receipt, with reviewer/time and hashes in its sidecar. Public integration is
stage 3; the measured envelope covers this exact 1024-square recipe.

### Integrated public card acceptance, 2026-11-17

Passed on `9341e575dc450b5867e3c88745e1849c40af47ce`: one cached Turbo
load through the real public server, Windows RTX 5080 CUDA 0, fp8/model,
Eager/Native, diffusers 0.42.1, 1024-square source/output, eight full steps
and CFG 1. Eleven declared outputs reproduce the private reference RGB
hashes: the 0.25 single/recovery/CLI pass, serial 0.4 batch and its solo
second seed, same-size Then refine's baseline/pass pair, Mass refine,
More seeds and retained Finish. Every saved PNG, full RGB and exact metadata
hash was independently recomputed against SQL and the captured job/API
records; allocated/reserved peaks agree with the process observation and
`last_peak`. All four CPU fixture PNGs stayed unchanged, and every actual
Refine source, including the new Then refine base, proves the original full
RGB. The [public acceptance receipt](anima-refine-turbo-public-2026-11-17.json)
retains the hashes, source manifest and complete scope.

Actual progress was two steps at start sigma 0.5 for strength 0.25 and four
at 0.75 for 0.4, eight callbacks for the serial batch. The loaded owner
released hooks after each image and cancellation, twelve observations in
all. Cancel was requested after real SSE step one and unwound at callback
two with no saved/decoded output and cleared `last_peak`; Finish then used
the checked retained SQL cell and preview pin in the same one-load session.
Five held probes returned 422 without queueing; same-size Re-render pruned
with 409 and enlargement stayed 422. The ordinary CLI used `--from-image
--via-server` without explicit source/strength flags; its actual retained
request carried the PNG's source id and 0.25 strength. This establishes
public API and CLI-via-server acceptance, not a native-browser GPU trial,
standalone CLI loading, a restart-recovery trial or another platform.

The load took 11.12 s, peaking at 8.65 GiB server-tree RSS and 10.63 GiB
private; session minima were 25.61 GiB free RAM and 19.92 GiB commit headroom.
Whole-card peak was 4,983 MiB over a 2,031 MiB baseline, shared rises
82.0 MiB process / 82.1 MiB adapter. The declared 24 GiB RAM/commit admission,
16 GiB RSS cap and unchanged card/shared/4 GiB floors held. Of 306 samples,
one 2.437 s startup gap preceded any load; the 238 samples from loading
onward averaged 0.249985 s with a 0.266288 s maximum gap. No guard tripped.
Power was acquired/released, unload answered `unloaded`, and later readback
found neither server PID 45424 nor a listener on 7993. All 193 source hashes
match at start, end and independent audit; the raw receipt was not mutated.
Times are descriptive. The original visual limits remain; this does not
qualify enlargement, Aesthetic/Preview, other loaders or cards.

#### Explicit nonempty negatives, 2026-11-17

Separate private one-load probes of Aesthetic v1.1 and Preview v1 completed
six calls each: the retained empty-negative 0.125 control, explicit
`low quality, blurry` at 0.125, its repeat, callback-one cancellation at 0.4,
recovery at 0.125, then an empty-negative direct render. Each probe reused
its original audited 1024-square RGB baseline and seed 4471, with 30 full
steps, CFG 4, fp8/model offload, Eager/native and one CPU generator per
one-image call. The combined receipt is
`docs/maintainer/anima-refine-negative-2026-11-17.json`; the source data head
was `091303a`, with runtime code unchanged from `9341e57`.

All twelve retained `*-call.json` files record the actual pipeline kwargs,
including the exact negative string. The saved image metadata and receipt
arguments agree. Both empty controls match their original 0.125 hashes;
both final direct images match their original baselines. Each nonempty image
differs from its empty control and is RGB-identical to its repeat and its
post-cancel recovery. The 0.125 passes actually ran four steps from sigma
0.315789521; nonempty cancellation stopped at callback one from strength
0.4/sigma 0.666666687 without decoding. Each run saved five images with three
unique RGB outputs, plus a retained call proof for cancellation. Every PNG,
full RGB, exact output metadata and call-metadata hash was recomputed; source
pixels and source PNGs remain unchanged.

| Measurement | Aesthetic | Preview |
|---|---|---|
| Cached load | 6.615 s | 7.843 s |
| Nonempty 0.125 / repeat / recovery | 5.548 / 4.823 / 4.902 s | 6.252 / 6.122 / 6.180 s |
| Nonempty cancel after callback one | 2.182 s | 2.622 s |
| Empty direct after | 21.303 s | 29.635 s |
| Peak allocated / reserved | 2472.99 / 2642 MiB | 3277.11 / 3446 MiB |
| Whole-card peak | 4978 MiB | 5738 MiB |
| Peak process RAM | 7.121 GiB | 9.098 GiB |
| Minimum free RAM / commit | 27.003 / 26.421 GiB | 23.554 / 23.713 GiB |
| Guard samples / total time | 257 / 64.390 s | 314 / 78.639 s |
| Largest achieved sample interval | 0.250569 s | 0.261135 s |
| Post-exit card / preflight | 1985 / 2028 MiB | 1983 / 1985 MiB |

All sampled verdicts were `ok`; the requested period was 0.25 seconds.
Initial admission stayed 24 GiB free RAM and commit with a 2500 MiB idle-card
ceiling; continuous RAM/commit floors stayed 4 GiB, process RSS cap 16 GiB,
and card/shared limits unchanged. Normal cleanup recorded 32 MiB allocated
and reserved before process exit. Both Windows power requests were acquired
and released without error. Complete recipe, source/configuration identity,
per-call metrics and raw evidence hashes remain in the combined receipt.

The root agent inspected `negative-0125.png` from both runs: subject, blue
coat, red bicycle and scene composition remain, with expected outline/texture
changes and no obvious new seam or identity replacement. Existing geometry
and anatomy imperfections remain. Other image findings are inherited only
after full RGB equality with earlier reviewed controls or this newly viewed
negative output. This is agent review, not Matt acceptance, author advice or
a universal quality claim. The tested string is qualification input. The
stable nonempty branch and its cancellation/recovery support removing the
temporary empty-negative admission condition for ordinary content inside the
existing measured envelope; no larger size, other loader or completed
nonempty 0.4 quality comparison is established. Original raw receipts remain
byte-identical in `receipt-unreviewed.json`; only visual-note strings were
added to the live receipts, with review sidecars recording who/when and hashes.

## Aesthetic and Preview integration, 2026-11-17

Independent one-load private trials have now supplied the missing Aesthetic
v1.1 and 40-layer 2.9B Preview data. The earlier holds were prerequisites,
not a decision to leave those variants unsupported after measurement. Their
source extension uses the same RTX 5080/Windows CUDA 0, fp8/model,
Eager/Native, diffusers 0.42.1 and 1024-square source/output envelope.

The pure immutable envelope table keeps identities, display names, steps,
CFG, allowed strengths and provisional openings distinct. Turbo retains
8 steps/CFG 1/opening 0.25. Aesthetic and Preview each use 30 steps/CFG 4 and
open at 0.125. Their allowed 0.125/0.25/0.4 passes run 4/8/12 steps from
indices 26/22/18, at measured start sigmas 0.315789521/0.521739185/0.666666687.
No strength, size, loader or platform boundary was widened from those data.

The resolver registers only each exact measured text class and its exact
counterpart, preserving constructor and `REQUIRED`/`CARRIED` contracts.
Profiles, measured-size notes, typed suggestions, API admission, actual
per-call/source checks and central rounding all consult the corresponding
variant. Tests retain unknown/impostor/subclass refusals and direct-render
behavior; no generation algorithm or dependency changed.

Separate guarded negative-conditioning probes also passed for each new
variant: known empty 0.125 control hashes, a nonempty `low quality, blurry`
negative that changed the pixels, exact repeats/recovery, cancellation without
a saved output, and each direct baseline unchanged afterwards. The temporary
empty-negative hold was removed before release; no user text is cleared.
Normal CFG-4 negatives retain their existing derived `true_cfg=4` in rows,
PNGs, keys and reruns. This does not enable a separate True CFG control or
guarantee arbitrary prompt quality. Turbo's CFG-1 negative behavior is unchanged.

The existing status-driven frontend needs no model-specific arithmetic:
it receives the 30-step measured suggestions and exact 4/8/12 counts.
Native Aesthetic/Preview smoke fixtures keep their existing guide, LoRA,
reference, Mu and negative-prompt coverage while adding measured Refine
opening, holds, queueing and source/strength carry. The Turbo fixture checks
switching to those independently measured recipes and back to Turbo's own.

The evidence remains separate: `anima-refine-aesthetic-2026-11-17.json` and
`anima-refine-preview-2026-11-17.json` own their private measurements; Turbo's
integrated acceptance owns its real API result. Aesthetic/Preview integrated
real-server/API acceptance follows this source extension before the complete
family is declared finished. Enlargement and every unmeasured loader/card
remain held.

### Family source validation

The completed source extension passed the full suite with
`CUDA_VISIBLE_DEVICES=-1`, the existing original virtualenv and at most four
workers: **11,061 passed, 325 skipped**, 85 warnings, exit 0, 366.03 seconds.
Ruff passed and pinned Pyright checked all **217 backend/tool sources** with
zero diagnostics, using the relative-scope worktree configuration.

Focused checks included 605 resolver/schedule/qualification tests, 117 actual
CPU-backend/Native-observation tests, 100 HTTP admission/recovery tests and
201 profile/CLI/legacy/docs/audit/capability checks. The negative tests cover
actual conditional output changes and repeats plus unchanged user text and
effective CFG 4 through rows, PNGs, More seeds and Finish; Turbo's ignored
negative behavior remains distinct.

All four Anima browser smokes passed. Aesthetic ran 85 steps, Preview 73 and
the shared Turbo Refine row 103, alongside the existing Turbo guide smoke.
The sole new fixture failure assumed an empty reference list instead of the
API's canonical null; it was corrected to assert null, without changing
production behavior or weakening the source/negative checks. Measured controls
and twelve-of-thirty preview captions were visually inspected. Owned fake and
browser processes/ports were cleaned up.

Browser receipts are retained at
`%TEMP%/auradream-smoke-artifacts-daym2goa/summary.json` (three passing rows and
the diagnosed Preview fixture) and
`auradream-smoke-artifacts-yzbb38wk/summary.json` (corrected Preview). These
source/fake-browser checks do not substitute for the final integrated
Aesthetic/Preview card sessions.
