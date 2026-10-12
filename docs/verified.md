# Verified on the card

What has been measured on the RTX 5080 (16 GB), and when. The reasoning
around each number is in `design-notes.md`; this is the lookup. Times are per
image with model offload, and "peak" is what the process held rather than
what `nvidia-smi` showed, unless said otherwise - the desktop holds about
5,200-6,100 MiB of the card while Unreal Editor and the browsers are open,
and a run that lands within a gigabyte of the top spills into system RAM and
slows two to three times (schnell's 2 MP read 35-45 s that way and 25 s at a
1,270 MiB baseline).

## Checkpoints

| Checkpoint | Quantization | Size | Steps | Time | Peak | Date |
| --- | --- | --- | --- | --- | --- | --- |
| Krea 2 Turbo | fp8, Windows | 1 MP | 8 | - | ~13 GB | 2026-09 |
| Krea 2 Turbo | NVFP4, WSL | 1 MP | 8 | 15 s | 8.1 GiB | 2026-09-05 |
| Krea 2 Turbo | NVFP4, WSL | 2 MP | 8 | 24 s | 8.6 GiB | 2026-09-05 |
| Krea 2 Turbo, `LOCAL_DIFFUSION_OFFLOAD=none` | NVFP4, WSL | 1 MP / 2 MP | 8 | 26 s / 72 s | 14.7 / 18.7 GiB, spilling | 2026-09-05 |
| FLUX.2 klein 4B | NVFP4, WSL | 1 MP | 4 | 4.0 s | 2.9 GiB | 2026-09-05 |
| FLUX.2 klein 4B | NVFP4, WSL | 2 MP | 4 | 4.6 s | 3.4 GiB | 2026-09-05 |
| FLUX.2 klein 4B | fp8, Windows | 1024x1024 | 4 | 10.2 s; 9.2 / 8.2 s with one reference, 11.3 s with two | - | 2026-09-08 |
| FLUX.1 Krea dev | fp8, Windows | 1 MP | 8 | - | 14,950 MiB whole-card by `nvidia-smi` mid-denoise, the desktop's 1,773 MiB included; 2,148 MiB with the pipeline loaded and idle | 2026-09-08 |
| FLUX.1 Krea dev | fp8, Windows | 1024x1024 | 8 | 15.4 s; 17.5 / 16.3 / 18.6 s under a LoRA at three weights | - | 2026-09-08 |
| FLUX.1 Krea dev, true CFG 1.5 | fp8, Windows | 768x768 | 12 | 25.9 / 25.4 s against 15.7 / 15.2 s without, ~1.65x | - | 2026-09-08 |
| Z-Image Turbo | NVFP4, WSL | 1 MP | 8 | 8 s (11 s load) | 4.1 GiB | 2026-09-08 |
| Z-Image Turbo | NVFP4, WSL | 2 MP | 8 | 14.5 s | 4.6 GiB | 2026-09-08 |
| FLUX.1 dev | NVFP4, WSL | 1 MP | 28 | 30 s | 7.4 GiB | 2026-09-08 |
| FLUX.1 dev | NVFP4, WSL | 2 MP | 28 | 48 s | 7.8 GiB | 2026-09-08 |
| FLUX.1 schnell | NVFP4, WSL | 1 MP | 4 | 16-23 s | 7.4 GiB | 2026-09-08 |
| FLUX.1 schnell | NVFP4, WSL | 2 MP | 4 | 25 s | 7.8 GiB | 2026-09-08 |
| Qwen-Image, true CFG 4 | NVFP4, WSL | 1 MP | 50 | 78 s once warm, 210 s for the first of a session (22 s load) | 11.75 GiB | 2026-09-08 |
| Qwen-Image, true CFG 4 | NVFP4, WSL | 2 MP | 50 | 152 s | 12.1 GiB | 2026-09-08 |
| ERNIE-Image Turbo | NVFP4, WSL | 1 MP | 8 | 9.6-10.2 s once warm, 17-18 s for the first of a process (22 s load) | 5.2 GiB | 2026-09-08 |
| ERNIE-Image Turbo | NVFP4, WSL | 2 MP | 8 | 24.3 s | 5.9 GiB | 2026-09-08 |
| ERNIE-Image Turbo | NVFP4, WSL | 4 MP, batch 2 | 8 | 48.6 s per pair | 10.3 GiB allocated, 12.0 GiB reserved | 2026-09-08 |
| Z-Image Turbo, the same evening | NVFP4, WSL | 1 MP / 2 MP | 8 | 10.6 s / 12.8 s | 4.1 / 4.6 GiB | 2026-09-08 |
| Z-Image Turbo, the same evening | NVFP4, WSL | 4 MP, batch 2 | 8 | 23.5 s per pair | 7.4 GiB allocated, 11.8 GiB reserved | 2026-09-08 |
| Illustrious XL v2.0-STABLE, CFG 5 | fp8, Windows | 1024x1024 | 28 | 15 s | not measured | 2026-09-17 |
| Illustrious XL v2.0-STABLE, CFG 5 | fp8, Windows | 1536x1536 | 28 | 18 s | not measured | 2026-09-17 |
| Illustrious XL v2.0-STABLE, CFG 5 | fp8, Windows | 2048x2048 | 28 | 35 s | not measured | 2026-09-17 |
| Z-Anime Base, CFG 4, first image after cancellation | fp8, Windows | 1024x1024 | 40 | 78.1 s | 10.35 GiB allocated | 2026-09-19 |
| Z-Anime Base, CFG 4 | fp8, Windows | 1024x1536 | 40 | 128.9 s | 7.25 GiB allocated | 2026-09-19 |
| Z-Anime Base, CFG 4 | fp8, Windows | 1920x1088 | 40 | 183.1 s | 7.70 GiB allocated | 2026-09-19 |
| Z-Anime Base, CFG 4 | fp8, Windows | 2048x2048 | 40 | 519.1 s | 9.58 GiB allocated, 13.95 GiB reserved | 2026-09-19 |
| Z-Anime Base, CFG 4, repeated seed | fp8, Windows | 1024x1024 | 40 | 79.2 s | 6.78 GiB allocated | 2026-09-19 |
| Anima Turbo v1.1, CFG 1 | fp8, Windows | 1024x1024 | 8 | 5.68 s after cancellation; 4.40 s repeat | 2.17 GiB allocated, 2.34 GiB reserved | 2026-09-26 |
| Kohaku XL Zeta, CFG 5 | fp8, Windows | 1024x1024 | 28 | 9.33 s after cancellation; 9.68 s repeat | 3.69 GiB allocated, 3.81 GiB reserved (maximum of the pair) | 2026-09-26 |
| NetaYume Lumina v4, CFG 5 | fp8, Windows | 1024x1024 | 40 | 38.46 s after cancellation; 38.74 s repeat | 5.76 / 3.30 GiB allocated, 5.92 / 4.38 GiB reserved | 2026-09-26 |
| NewBie Image Exp0.1, XML prompt, CFG 4 | fp8, Windows | 1024x1024 | 28 | 39.39 s after cancellation; 38.74 s repeat | 4.81 GiB allocated, 4.86 GiB reserved | 2026-09-26 |
| WAI Illustrious v17, CFG 6 | fp8, Windows | 1024x1344 | 28 | 9.25 s after cancellation; 9.64 s repeat | 3.69 GiB allocated, 4.37 GiB reserved (maximum of the pair) | 2026-09-26 |
| FLUX.1 Krea dev, CFG 3.5 | fp8, Windows | 1024x1024 | 28 | 41.6-42.6 s; 60.8 s for the first after a 38.2 s load; 41.9 s repeat; about 42 s for the second of two images through the CLI, by the two PNGs' one-second save stamps (1m31s for both) from a clean worktree of the analyser's parent commit, 7a9c651 | 14,615 MiB whole-card by `nvidia-smi`, the desktop's 1,746 MiB included; 1,688 of 16,303 MiB free | 2026-09-27 |
| Animagine XL 4.0, CFG 5 | fp8, Windows | 1024x1024 | 28 | 17.2 s, the first image of a process; 0.22 s a step after the first | 3.12 GiB allocated, 3.47 reserved; 6,203 MiB whole-card | 2026-10-05 |
| Illustrious XL v2.0-STABLE, CFG 5 | fp8, Windows | 1024x1024 | 28 | 11.1 s, the first of a process; 0.22 s a step | 3.12 / 4.37 GiB; 7,170 MiB whole-card | 2026-10-05 |
| FLUX.1 schnell | fp8, Windows | 1024x1024 | 4 | 27.3 s, the first of a process, 19.5 s of it the first step; 1.4 s a step after | 11.99 / 12.66 GiB; 15,433 MiB whole-card | 2026-10-05 |
| Z-Image Turbo | fp8, Windows | 1024x1024 | 8 | 14.3 s, the first of a process; 0.9 s a step after the first | 6.43 / 7.17 GiB; 9,499 MiB whole-card | 2026-10-05 |
| Z-Image (undistilled), CFG 5 | fp8, Windows | 1024x1024 | 50 | 102.5 s, the first of a process; 1.8 s a step | 6.77 / 8.44 GiB; 10,885 MiB whole-card | 2026-10-05 |
| RouWei 0.8 epsilon, CFG 6 | fp8, Windows | 1024x1024 | 28 | 7.638 s after cancellation; 7.882 s repeat | 3.686 GiB allocated, 4.371 GiB reserved (maximum of the pair) | 2026-10-07 |
| Anima Aesthetic v1.1, CFG 4 | fp8, Windows | 1024x1024 | 30 | 23.229 s after cancellation; 22.582 s repeat | 2.173 GiB allocated, 2.338 GiB reserved | 2026-10-07 |
| NoobAI-XL V-pred 1.0, CFG 5 | fp8, Windows | 1024x1024 | 28 | 8.078 s after cancellation; 8.490 s repeat | 3.686 GiB allocated, 4.371 GiB reserved (maximum of the pair) | 2026-10-07 |

The October 5 rows were each the source of a refine ladder ("Past 4 MP",
below), with another project's processes keeping the CPU near 75% busy:
their times are upper bounds.

NoobAI-XL V-pred 1.0 was verified on 2026-10-07 with Windows FP8, model
offload, batch 1 and seed 12345. The pinned official Diffusers tree loaded
in 8.912 seconds from cache; the initial download and load took 106.800
seconds. The live `StableDiffusionXLPipeline` used `EulerDiscreteScheduler`
with `v_prediction`, zero-terminal-SNR rescaling and trailing spacing.
At 28 steps it started at timestep 999 with sigma 4096 and ended at sigma 0.
The public selection remained `Laxhar/noobai-XL-Vpred-1.0`.

The prompt used the author's quality prefix followed by an adult woman with
auburn hair, a navy coat and brown satchel, in a greenhouse with blue flowers
and warm lantern light. The negative was the author's full recommended
negative. Visual inspection found a coherent anime illustration with those
scene elements. At 1024x1024, cancellation stopped after step 2 in 5.654
seconds. The following two calls completed all 28 steps and produced
identical pixels, hash `b7cf62907e9e6b65`.

Refine from that source to 1536x1536 at strength 0.4 ran 11 of the requested
28 steps. Cancellation stopped after step 2 in 3.573 seconds; the following
two passes took 9.111 and 9.179 seconds, each peaking at 3.670 GiB allocated
and 4.516 GiB reserved. Their pixels matched exactly, hash
`7d579d891a2f06a5`. Visual inspection found the source composition retained,
with facial, clothing and flower details redrawn. This is a functional check,
not a quality ranking or a claim about larger direct renders. Other CPU work
and part of the Python suite ran during qualification; these are individual
smoke timings. Generation times exclude loading and PNG saving; allocator
peaks exclude desktop/driver memory.

Rows and PNG metadata retained the public id, FP8, seed, steps, CFG 5,
negative prompt and Refine source/strength. True CFG was 5 and Mu null;
CFG 1 dropped the negative prompt. The corrected qualification completed
all runtime assertions and unloaded the pipeline (`passed: true`,
`unloaded: true`). The first harness run incorrectly expected a strength
on a direct-render row; storage correctly recorded null, and only the
scratch assertion was corrected. Both attempts are retained under
`%TEMP%/local-diffusion-noobai-20261007/`: `qualify.py`, `gpu.log`,
`gpu-results.json`, `harness-first.log`, `harness-first-results.json`, the
separate `gpu.db` and `outputs/` PNGs. The production gallery was not used.

The full CPU suite passed 2,216 tests with 4 skips and 49 dependency
deprecation warnings. `tools/smoke-noobai.json` passed all 19 browser
expectations. The real pinned tokenizers both reported a 77-token sequence
limit; the author's prefix used 17 tokens and negative 47, including special
tokens. Larger direct/Refine sizes, batch scaling, int8/bfloat16, sequential
offload and user LoRAs remain unmeasured. There is no SDXL live latent-preview
map, and no Nunchaku build is mapped.

Anima Aesthetic v1.1 was verified on 2026-10-07 with Windows FP8, model
offload, batch 1 and seed 12345. Download, CPU conversion and first load took
71.83 seconds with the shared Anima components already cached. The concrete
`AnimaAestheticPipeline` loaded the variant's own denoiser and conditioner
through the normal quantized component loader, using FlowMatch Euler shift 3.
Its separate recorded selection is `circlestone-labs/Anima@aesthetic-v1.1`.

The two-sentence prompt described an auburn-haired adult botanist in a navy
coat with a brown satchel, examining blue flowers in a greenhouse at dusk,
with warm lanterns and a blue evening sky. The negative was `blurry, jpeg
artifacts`. Visual inspection found those scene elements in a coherent
illustration. Cancellation stopped after step 1; both following calls
completed all 30 steps and produced identical pixels, hash
`a444e916bf7c064b`. Live previews arrived at steps 8 through 29. Generation
times include previews and exclude loading/PNG saving; allocator peaks
exclude desktop/driver memory. The CPU test suite also ran during these
samples, so they are individual smoke timings, not a controlled benchmark
or a quality comparison with Turbo.

Rows and PNG metadata retained the Aesthetic selector, FP8, seed, 30 steps,
CFG 4 and the applied negative prompt, with True CFG 4 and unsupported Mu
null. The pipeline was unloaded afterward. Evidence is in
`%TEMP%/local-diffusion-anima-aesthetic-validation/`: `validate.py`,
`validation.log`, `results.json`, previews, a separate database and the
`outputs/` PNGs. The production gallery was not used.
The full Python suite passed 2,187 tests with 4 skips and 39
dependency deprecation warnings. `smoke-anima-aesthetic.json` passed all 23
browser expectations. Higher
resolutions, batch scaling, int8/bfloat16 and sequential offload remain
unmeasured for Aesthetic. Mu, references, Refine, user LoRAs and Nunchaku
are unavailable in this integration.

RouWei 0.8 epsilon was verified on 2026-10-07 with Windows FP8, model
offload, batch 1 and seed 12345. CPU conversion and the first load took
15.098 seconds after the source download. The loaded
`StableDiffusionXLPipeline` uses `EulerAncestralDiscreteScheduler` with
epsilon prediction; Mu is unavailable and the SDXL Refine sibling is
available. The public selection is `Minthy/RouWei-0.8` and retains the
author's pinned epsilon file described in `design-notes.md`.

At 1024x1024, 28 steps and CFG 6, the prompt was `1girl, adult woman,
auburn hair, navy coat, brown satchel, greenhouse, blue flowers, warm lantern
light, anime illustration, masterpiece, best quality`, with negative
`worst quality, low quality, watermark`. Visual inspection found a coherent
illustration with the requested woman, clothing, satchel, flowers,
greenhouse and warm lights. Cancellation stopped at step 2 in 3.205 seconds;
the following two calls completed all 28 steps and had identical pixels,
hash `c341477ac1b11267`.

Refine from that source to 1536x1536 at strength 0.4 ran 11 of the requested
28 steps. A cancellation stopped at step 2 in 3.296 seconds, then two full
passes took 9.064 and 9.075 seconds, each peaking at 3.670 GiB allocated and
4.516 GiB reserved. Their pixels matched exactly, hash `ad12c067d7a042c4`.
Visual inspection found the source composition retained with details redrawn.
These samples establish the direct and Refine paths, not a quality ranking
or a reproduction of the author's latent or GAN upscaling recipe. Generation
times exclude loading and PNG saving; allocator peaks exclude desktop and
driver overhead.

SQLite rows retained the public model id, True CFG 6 and null Mu; saved PNGs
retained the public model id, FP8, negative prompt and refine source. The
qualification also checked that CFG 1 drops the negative prompt. All runtime
assertions completed and the pipeline unloaded (`passed: true`,
`unloaded: true` in the result). The production gallery was not used.
Evidence is under `%TEMP%/local-diffusion-rouwei-20261007/`: `qualify.py`,
`gpu-results.json`, the separate `gpu.db` and `outputs/` PNGs. Batch scaling,
int8/full-bfloat16 generation, sequential offload, user LoRAs and larger
direct or Refine sizes remain unmeasured.

The same checkout passed the full CPU suite (2,150 passed, 4 skipped) and
all 22 expectations in `tools/smoke-rouwei.json` on a fresh fake server:
the epsilon defaults, hybrid tag helper, optional quality tags, `by ` artist
autocomplete and validation, the CLIP content budget and literal `BREAK`
warning, saved generation identity, and the 1.5x Refine pass. These checks
are separate from the real GPU samples above.

WAI Illustrious v17 was verified on the RTX 5080 with Windows FP8 and model
offload. The pinned mirror checkpoint passed its SHA-256 check, converted
on CPU with its original VAE and both CLIP encoders, and loaded in 14.11
seconds after the source download. The resulting `StableDiffusionXLPipeline`
uses epsilon prediction and Euler Ancestral. The unused final CLIP-L layer's
non-finite weights were retained; no encoder replacement was needed.

Batch 1, seed 12345, 28 steps and CFG 6 produced a coherent anime illustration
of the requested auburn-haired adult botanist, navy coat, brown satchel,
blue flowers, greenhouse and warm lanterns. The image was visually inspected.
Both 1024x1344 renders had identical pixels and hash `32f2b2ce376c233c`.
Times exclude loading and PNG saving; allocator peaks exclude desktop and
driver overhead. This is a runtime smoke sample, not a quality benchmark or
a reproduction of the author's separate hires-refinement workflow.

Cancellation stopped after step 1, and the following two calls completed all
28 steps. SQLite rows and PNG metadata retained the public mirror id
`LyliaEngine/waiIllustriousSDXL_v170`, FP8, seed, size, prompt pair, steps,
CFG 6, effective True CFG 6 and null Mu. The four-channel SDXL VAE has no
latent-preview map, so previews disabled themselves without disrupting the
renders. The model was unloaded afterward. The production gallery was not
used; evidence is in `%TEMP%/local-diffusion-wai-validation/`: `validate.py`,
`validation.log`, `results.json`, the separate database and `outputs/` PNGs.

The full Python suite passed 453 tests and `smoke-wai.json` passed all 20
browser expectations. CPU conditioning tests cover none/FP8/int8 and both
positive/negative prompts with non-finite weights in the unused CLIP layer.
GPU int8/full-bfloat16, sequential offload, LoRAs, batch scaling and larger
resolutions remain unmeasured.

NewBie Image Exp0.1 was verified on the RTX 5080 with Windows FP8 and model
offload. The pinned official components are strictly converted on CPU; the
final recipe loaded in 31.91 seconds including conversion from downloaded
source files. The local `NewbiePipeline` uses FlowMatch Euler shift 6, empty
system prefixes, whole-sample CFG norm limiting, Gemma's penultimate hidden
state and Jina's masked mean. Source pins and the differences between author
sampling recipes are recorded in `newbie_loading.py` and `design-notes.md`.

The author's two-character XML example produced a coherent chibi illustration
with blue hair and blue clothing on the left, pink hair and red clothing on
the right, two briefcases and the right character waving. The final image was
visually inspected. Gemma reads up to 512 tokens and Jina 2048; the example is
486 and 539 tokens respectively, so applying 512 to both would truncate Jina.
Both final 1024-square renders, seed 12345 and CFG 4, had identical pixels and
hash `9dfb5b3a47c1c6ea`. Generation times include previews and exclude loading
and PNG saving. Allocator peaks exclude desktop and driver overhead. These
two images establish a runtime smoke result, not a quality benchmark.

Cancellation stopped after step 1, and the next two calls completed all 28
steps with preview frames. SQLite rows and embedded PNG metadata retained
the exact XML, official model id, FP8, seed, dimensions, steps, CFG, negative
prompt, effective True CFG 4 and null Mu. The pipeline was unloaded afterward.
Final evidence is under `%TEMP%/local-diffusion-newbie-validation-20260926-xml/`:
`validate.py`, `results.json`, logs, preview JPEGs and a separate database and
outputs directory. The production gallery was not used.

A shorter tag prompt also produced a coherent auburn-haired botanist with a
navy coat, brown satchel, blue flowers and warm greenhouse lanterns. That pair
took 37.48/36.84 seconds and had identical hash `42085185bf9b3f4c`; its final
image and a preview were visually inspected. This earlier smoke preceded the
Jina limit increase, which did not truncate the short prompt. Its evidence
is under `%TEMP%/local-diffusion-newbie-validation-20260926/`.

The full Python suite passes 438 tests, including strict tensor conversions,
atomic cache publication, owned quantization, real tiny pipeline round trips,
CFG/flow/Mu semantics, seeded batches, XML conditioning and cancellation
cleanup. `smoke-newbie.json` passes all 14 browser expectations. Higher
resolutions, GPU int8/full-bfloat16, sequential offload and GPU batch scaling
remain unmeasured. References, user LoRAs and Nunchaku are disabled.

NetaYume Lumina v4 was verified on the RTX 5080 with Windows FP8 and model
offload. The first load, including downloading the pinned v4 denoiser and
official Lumina 2 shared components and converting on CPU, took 290.68
seconds. The resulting `Lumina2Pipeline` uses FlowMatch Euler, fixed shift
6, normalized CFG and the author's anime system instruction. This is the
standard Diffusers sampling setup, not a reproduction of the author's
ComfyUI samplers. Source and component pins are in `netayume_loading.py`.

Batch 1, seed 12345, 40 steps, CFG 5 and blank Mu produced a coherent anime
illustration with the requested auburn hair, navy coat, brown satchel,
blue flowers, greenhouse and warm lanterns. The final image and a live
preview were visually inspected. Both completed 1024x1024 renders had
identical pixels and content hash `48847e34d1c6dd99`. Generation times
exclude loading and PNG saving, but include preview callbacks. The paired
allocator peaks above are the first post-cancellation render and its repeat;
they exclude desktop applications and CUDA/driver overhead. These are smoke
samples, not a quality or performance benchmark.

Cancellation stopped after step 1; both subsequent calls completed all 40
steps with preview frames. Gallery records and embedded PNG metadata kept
`duongve/NetaYume-Lumina-Image-2.0`, FP8, seed, dimensions, 40 steps, CFG 5,
the supplied negative prompt, effective True CFG 5 and null Mu. The model
was unloaded afterward. The separate evidence directory is
`%TEMP%/local-diffusion-netayume-validation-20260926-153239/`: `validate.py`,
`validation.log`, `validation.err`, `results.json`, preview JPEGs and a
separate database and outputs directory. The production gallery was not used.

The full Python suite passes 396 tests, including strict tensor conversion,
cache publication and retry, selective pinned downloads, all three normal
quantization routes, and actual Lumina pipeline CPU checks of the fixed
system instruction, negative-prompt gate and final scheduler sigmas.
`smoke-netayume.json` passes all 12 browser expectations. Int8/full-bfloat16
GPU paths, sequential offload, user LoRAs, batch scaling and larger image
sizes have not been measured. The guide's 1.05-1.57 MP span comes from
published family examples, not a verified v4 output limit.

Kohaku XL Zeta was verified with Windows FP8 and model offload using the
ordinary component-tree loader. The first load, including downloads, took
152.89 seconds; the resolved Hub revision was
`c14affd9483ac7603cfff3f06b05da22a06deaa6`, with `StableDiffusionXLPipeline`
and the supplied epsilon/Euler scheduler. Batch 1, seed 12345, 28 steps and
CFG 5 produced a coherent anime portrait with the requested auburn hair,
navy coat, brown satchel, blue flowers, greenhouse and warm lights. The final
image was visually inspected. The two 1024x1024 renders had identical pixels
and content hash `3fe471db98c8b5da`. Times exclude loading and PNG saving;
allocator peaks exclude the desktop and CUDA/driver overhead. These are
individual smoke samples, not a quality or performance benchmark.

Cancellation stopped after step 1, then both following calls completed all
28 steps. PNG metadata and gallery rows retained `KBlueLeaf/Kohaku-XL-Zeta`,
FP8, seed, steps, CFG and the negative prompt. The stored True CFG was 5,
the scale that applied the negative prompt; the unsupported Mu was null.
There were no live preview frames: the four-channel SDXL VAE has no fitted
map, and the preview callback disabled itself without affecting generation.
The pipeline was unloaded afterward. Evidence is under
`%TEMP%/local-diffusion-kohaku-validation/`: `validate.py`, `validation.log`,
`results.json`, the separate database and `outputs/` PNGs. The production
gallery was not used. The Python suite passes 375 tests and
`smoke-kohaku.json` passes all 16 browser expectations. Kohaku's int8/full-
bfloat16 GPU paths, sequential offload, user LoRAs, batch scaling and higher
resolutions have not been measured.

Anima Turbo v1.1 was verified on the RTX 5080 with Windows FP8, model
offload, the pinned full Turbo checkpoint and FlowMatch Euler shift 3.
The cached component tree loaded in 5.61 seconds. Batch 1, seed 12345,
8 steps and CFG 1 produced a coherent anime illustration of the requested
auburn-haired adult botanist, navy coat, blue flowers, greenhouse and warm
lanterns. Both completed 1024x1024 renders had pixel hash
`8f1f6122fe476ad0`. Generation times exclude loading and PNG saving but
include live-preview callbacks. Allocator peaks exclude desktop applications
and CUDA/driver overhead; these are individual smoke samples, not a benchmark.

Cancellation stopped after step 1 before a VAE decode, then both subsequent
calls completed all 8 steps. Live previews arrived at steps 2/3/5/7 and
2/4/6; the final image and a preview were visually inspected. PNG metadata
preserved `circlestone-labs/Anima`, FP8, seed, steps and CFG, with the unused
negative prompt, True CFG and Mu recorded as null. The pipeline was unloaded
afterward. The production gallery was not used. Evidence is in
`%TEMP%/local-diffusion-anima-validation/`: `validate.py`, `validation.log`,
`results.json`, preview JPEGs and a separate database/outputs directory.

The Python suite passes 373 tests, including strict Turbo conversion,
quantized-loader routing, real modular-block CPU seed/CFG/cancel checks and
Accelerate hook cleanup. `smoke-anima.json` passes all 12 browser expectations.
Anima's int8/full-bfloat16 GPU paths, sequential offload, batch scaling and
higher resolutions have not been measured. Mu, reference images and user
LoRAs are unavailable in this integration.

Z-Anime uses the pinned nested Base denoiser, its supplied FlowMatch Euler
scheduler with shift 6 and Mu blank, and the standard Qwen3 text encoder from
the official Z-Image parent. Its own bundled text encoder contains scaled
FP8 tensors without quantization metadata, which Transformers ignores; that
initial validation run was stopped. The corrected component sources and the
reason for the override are recorded in `design-notes.md`.

The Z-Anime samples use Windows FP8 and model offload, batch 1, seed 12345,
40 steps, CFG 4 and one natural-language prompt describing an auburn-haired
adult botanist examining blue flowers in a glass greenhouse at dusk. The
negative prompt is `blurry, low quality, watermark, extra limbs, extra fingers`.
The 1 MP, 1.5 MP, 2 MP and 4 MP samples retain the person, navy coat, brown satchel,
flowers and greenhouse; the framing changes the composition. These
are individual runtime samples, not a benchmark against Illustrious.
Times exclude loading and PNG saving, and include live-preview callbacks.
The 1 MP sample immediately follows a successful first-step cancellation;
that state differs from the later uninterrupted cells, so its higher peak
does not establish a reverse relationship between image size and memory.
The 4 MP image stayed coherent but took 8 minutes 39 seconds. During that
cell `nvidia-smi` reported about 15,849 MiB occupied of 16,303 MiB, including
the desktop; the allocator's 13.95 GiB peak reservation leaves little room
for other GPU applications. Prefer 1.5-2 MP for iteration on this setup.
The final 1 MP repeat matched the first image's pixels exactly (content hash
`c427b7fdb6dabc8e`). Cancellation stopped at step 1, and each completed image
reported all 40 steps and delivered 23 live-preview frames. The pipeline was
unloaded after the checks; device usage returned to the desktop baseline.

Validation evidence is under
`%TEMP%/local-diffusion-z-anime-validation-v2/`: `validate.py`,
`validation.log`, `results.json`, the separate database and `outputs/` PNGs.
The PNG metadata retains the public model id, FP8, seed and applied settings.
The production gallery was not used. The full Python suite passes (343 tests),
and `smoke-z-anime.json` passes all 10 browser checks on the fake backend.

Illustrious's three timings are single CLI samples (including PNG saving),
batch 1, seed 12345, no negative prompt, with model offload and the converted
epsilon/Euler Ancestral scheduler. They establish runtime support, not a
benchmark or a guarantee of composition. The 1024 tag prompt produced the
requested doctor. The same natural-language prompt at 1536 and 2048 asked for
a doctor reading in a sunlit library: the 1536 sample included the person and
book, while the 2048 sample rendered shelves and omitted the doctor. The
authors' 1-2.25 MP native guidance and 4 MP quality caveat therefore remain
in the UI. The different prompts at 1024 and the higher sizes do not form a
controlled quality comparison.

The source checkpoint's SHA-256 matched the Hub's
`c2a1a3eaa13d4c107dc7e00c3fe830cab427aa026362740ea094745b3422a331`.
CPU conversion and the first FP8 load together took 48 s after the weights
were cached; subsequent processes reused the tree and loaded in 7-12 s.
All three PNG metadata records preserve `OnomaAIResearch/Illustrious-XL-v2.0`,
FP8, seed 12345, 28 steps, CFG 5 and null Mu. Validation images are in
`%TEMP%/local-diffusion-illustrious-validation/outputs`, with pixel hashes
`9c9742b5e08abae3` (1024), `da5ca7663a64e641` (1536) and
`9fe6ba696569a397` (2048). The production gallery was not used.

Under model offload the per-image weight transfer costs about 5 s and
dominates a 4-step image; on FLUX.1 the text encoders under bitsandbytes NF4
add to that, which is why schnell is not the speed win Z-Image Turbo is.

## Behaviour

| What | Result | Date |
| --- | --- | --- |
| Bit-determinism through torchao fp8 with offload moving components mid-forward | The same request twice on FLUX.1 Krea dev is the same hash; a negative prompt the pipeline dropped hashes the same as none | 2026-09-08 |
| True CFG on FLUX.1 | A negative prompt with no true CFG is byte-identical to none; true CFG 1.5 is a different, reproducible image | 2026-09-08 |
| LoRA under torchao fp8 + PEFT | The adapter changes the image and reproduces; `unload_lora_weights` restores the base hash; weight 0 is byte-identical to no adapter | 2026-09-08 |
| Animagine XL 4.0 basic generation | User-verified on the card after integration: the weights loaded and output looked good; exact quantization and offload settings were not recorded | 2026-09-10 |
| Animagine XL 4.0, Batch 8 at 1 MP | 9.6 GB of the 16 as read off the card, which is what raised the Batch cap to 12; Batch 12 itself not yet measured | 2026-09-12 |
| Reference images on klein | The referenced scene reappears in the asked-for pose, same hash twice; no 2-3x cost (see the table) | 2026-09-08 |
| Live preview | Matrices fitted on the card, R-squared 0.964 (FLUX VAE) and 0.950 (Qwen-Image VAE); frames looked at; Z-Image frames at steps 2/4/6 of 8 in the right colours | 2026-09-08 |
| Ctrl-C during a load | SIGINT 12 s into a load exits in 3 s with the lifespan run | 2026-09-08 |
| Ctrl-C mid-sweep with a stream attached | `done` (cancelled) reaches the stream 0.2 s after the signal; the process is gone 3 s later | 2026-09-08 |
| Pause | Acknowledged at the cell boundary, where a paused 512x512 grid held 2,046 MiB of the card (whole-card `nvidia-smi`, desktop included); a 1 MP fp8 denoise reads 14,950 MiB | 2026-09-08 |
| `reindex --from` over the real gallery | Reproduces every hash and true CFG | 2026-09-08 |
| Qwen-Image's rows | Record true CFG 4 and the negative prompt | 2026-09-08 |
| `uv.lock` torch hashes | The four hand-pinned wheels checked file-for-file (`RECORD`) against the Windows and WSL venvs | 2026-09-08 |
| Seed reproduction under Nunchaku NVFP4 | Not bit-for-bit: the same request in two processes gives two hashes on Z-Image and ERNIE alike. Z-Image's pair differ by 4.5/255 mean with 17% of pixels off by more than 8 (a different seed: 36/255, 85%); ERNIE's by 14-24/255 and 58-75% (a different seed: 41/255, 94%) - visibly different details, not noise. The fp8 path on Windows reproduces (above) | 2026-09-08 |
| ERNIE-Image Turbo's ordinary load without the prompt rewriter | Through `DiffusionPipeline.from_pretrained`, `pe=None` pulled the 7.7 GB folder at load and the process held 11 GB of RAM; through `ErnieImagePipeline.from_pretrained` the same load fetched nothing with the folder deleted from the cache, and the generation matched | 2026-09-08 |
| `single_file.sweep_work` on this machine's conversion cache | `illustrious-xl-v2.0-stable-b3a1d8184c923b80` had three empty work directories beside it, left by conversions interrupted on 2026-09-17 (its name with `-1ttk4htn`, `-_bn0av0r` and `-alpmu2dc`). The real `sweep_work` removed exactly those three, logging "removed an interrupted conversion" for each, and left the published tree and the other checkpoints' trees (Anima Turbo, NetaYume, the two NewBie, WAI) untouched | 2026-09-29 |
| A dropped WSL page cache | Z-Image read 17-19 s per 1 MP image right after WSL reclaimed memory (`Cached: 0`), 10.6 s on the next run; measure twice | 2026-09-08 |
| `TORCH_DISABLE_NATIVE_JIT=1` | Cures "Failed to find C compiler" from the Qwen3-VL text encoder on a fresh WSL Ubuntu | 2026-09-05 |
| Fixed xet concurrency | Downloads at 55-60 MB/s where the adaptive controller had stalled a 12 GB checkpoint twice | 2026-09-05, 2026-09-08 |
| The results strip and live preview over real frames, at 2560x1305 (Anima Turbo v1.1, fp8, model offload, 1024x1024, 8 steps; the page driven by real presses, its layout read by a MutationObserver on the strip) | One image: the first frame 3.5 s after Generate, latent frames 256x256 drawn 887x887 inside the result's own 887x923 tile ("Live preview · step 2/8" to "6/8", 3 frames), the image 11.4 s after the press (11.3 s, the first of the process, after an 8.8 s load) in the same box, the gallery head unmoved. A batch of four: one row of four 491 px tiles from the first frame (3.3 s), the images at 14.5-14.7 s (3.6 s each), the head unmoved until the job ended, when it moved 3 px: the page's own scrollbar went with the progress block (a gallery too short to scroll the page by itself) and the strip refit to 494 px. The fake backend shows the same, the scrollbar coming at the first frame and going at the end, 3 px each way (`html` keeps no gutter for it). Across, the preview did not hold the first image's slot: alone, it sat centred in the four `auto` columns (`grid-template-columns: repeat(var(--cols), auto)`, the empty ones no width), the first image landed there (x 1248) and moved left by half a tile as each of the others landed, to x 512 within 0.25 s. A 28-seed grid: the strip's 559 px min-height (`reserveStrip`) from the first frame (1.9 s) to the end and the head unmoved over 229 samples, frames 168x168 inside their tiles; 4.3-7.0 s an image (median 4.6 s), 28 in 134.9 s where the preview said "about 2 min". Card peaks 4,740 / 5,849 / 4,768 MiB over a 1,956 MiB desktop. Another project's processes (linkers, a test runner) kept the CPU 31% busy on average during the grid, at times all of it, probably behind the 7.0 s image | 2026-10-05 |
| The SDXL refine pass's stand-in (WAI v17, fp8, model offload): a 1024x1344 source refined at 1.5x and 0.4 (1536x2016, 11 of 28 steps), Versions 1 then 2, at 2560x1305 and 1400x900, Refine pressed in the dialog | The source stood in, dimmed, from the pass's first progress event, 2.7-2.9 s after the press (the viewer kept the source's own result until then), and stayed to the last step with no frame (SDXL has no preview map). Its caption counted "Refining #1 → 1536×2016 · strength 0.4 · step k/11" a step every 0.76 s, with an ETA that ran long early and short late against the time the result actually took: 27-28 s said at step 1 for 9.5 s, 16 s at step 2 for 8.7, 7 at step 5 for 6.4, 1 at step 10 for 2.6 - the first step carries the VAE encode and the denoiser's move to the card, and the decode and save follow the last (the result 1.9 s after step 11). The result replaced it in the same box at 12.2-12.4 s (12.0-12.1 s): 676x923 at QHD, 466x642 at 1400x900, the gallery head unmoved. Versions 2 ran as one 22-step count ("step k/22", 58 s said at step 1 for 21.3 s), with a 4.1 s gap between steps 11 and 12 (the first decode, the second encode), the stand-in in the fitted tier's box for the whole of it and both images landing together at 24.0-24.3 s, the head unmoved; across, as in the batch above, the lone stand-in sat centred and the first image moved left by half a tile when the second landed 0.2 s later (1168 to 830 px at QHD, 627 to 394 at 1400x900). Card peaks 7,102-7,192 MiB over a 1,972 MiB desktop | 2026-10-05 |
| Pause's words over real cells (FLUX.1 Krea [dev], fp8, model offload, Repeat 3 at 1024x1024 and 28 steps, 2560x1305; Pause, Resume and the withdrawal pressed in the page) | Pressed at step 5 of image 1: "Pausing…" 20 ms later with "pausing after this image…" ahead of the cell line, which kept it through all 23 progress events that followed. The worker paused at the cell boundary and the label read Resume 36.8 s after the press, with "Paused after image 1/3 · GPU handed back", the tab "⏸ Paused" and the bar held at 33%; the card read 2,387-2,416 MiB for the 20 s held (2,118 MiB desktop; 15,411 MiB mid-denoise). Resume: "Pause" at once and the worker resumed 0.1 s later. Pause pressed at step 5 of image 2 and pressed again 1.7 s later as "Pausing…": "Pause" at once, the note gone, and image 3 began straight after image 2 with no pause logged. The three images took 59.2 s (the first after a 41 s load), 44.7 and 44.8 s, 1.37-1.48 s a step, no spill. On a one-image job Pause was aria-disabled, titled "A single image has no pause point — Cancel stops it", and a press said so under Generate with focus kept on it. The running ETA started at ~13 min (the first image's 17 s first step, the transformer onto the card, read as a step's pace) and came down to ~2 min by the pause | 2026-10-05 |
| Cancel during a real first load, and the load's ticker (FLUX.1 Krea [dev], fp8, model offload, a fresh process at 2560x1305; Generate pressed with nothing loaded, Cancel pressed about 9 s in) | The job waited with `waiting_on` load from 0.03 s; "loading weights · 0:00" at 0.04 s, then a tick every 1.00 s on the wall clock to "0:09" at 9.05 s, over the sliding bar, with the clock's span aria-hidden, the same words in the viewer's line ("Loading weights · 0:09 - the first load of a session takes a few minutes."), the tab's title and the topbar's mirror, and the topbar "FLUX.1 Krea [dev] · loading…". Cancel at 9.38 s: the stream's `done` (cancelled) 40 ms later, the page idle at 9.55 s with "Cancelled — the weights keep loading, so the next Generate starts sooner." and focus on Generate. The load went on to ready 56.3 s after it began (38.2 s on 2026-09-27; another project's Python kept the CPU 87-97% busy throughout): the topbar back to "FLUX.1 Krea [dev]", the dot ready, Unload shown, the card idle at 2,223-2,237 MiB over a 1,963 MiB desktop. The next Generate, 13 s later, ran at once with no `waiting_on` and no second load in the log: its first step 13.5 s (the transformer onto the card), 1.35-1.5 s a step after, the image (8 steps) 32.6 s, the card's peak 15,246 MiB with no spill | 2026-10-05 |
| A time budget against real step times (Kohaku XL Zeta, fp8, model offload, 1024x1024, CFG 5, 2560x1305): three 28-step images to price the estimate (9.1, 8.4, 8.3 s), then Steps "12,28,40" x 6 seeds with a 2 min budget, Run grid pressed; again after Unload | The preview read "18 images · 3 steps × 6 seeds · about 2 min · ~25.7 MB … budget 2 min ≈ 4 of 6 seeds complete": a median 301 ms a step, 24.1 s a seed. The grid stopped at 121.2 s with "Time budget reached after 15/18 cells — kept 15 images. Resume finishes the rest." and Show sheet: 5 of the 6 seeds, since the budget stops a cell from starting and the fifth seed's last cell began at 110 s. The per-step price misjudged each cell - 12 steps took 4.5-4.7 s against 3.6 priced, 28 took 8.2-8.4 against 8.4, 40 took 11.0-11.2 against 12.0 - and evened out over a seed (24.1-24.6 s). The cell line: "~79s left" then "~59s left" in cell 1 (with 117-119 s actually left), "~2 min left" in cell 2, then the budget's own countdown from cell 3 ("~2 min left (budget)"; "~82s left (budget)" at the start of cell 6 with 83 s left), down to "~0s left (budget)" at 119.6 s. Unloaded, the preview added "(the load counts too)" and said 5 of 6 (18 rows now priced it at about 298 ms a step), and "15 of these are already in the gallery - tick Resume to skip them", the first run having written its base seed back to Seed. The load ticked "loading weights · 0:00" to "0:06" inside the budget (6.9 s) and the grid again stopped after 15/18, at 128.6 s: the cell begun 2.4 s before the deadline ran 8.6 s past it, so "~74s left (budget)" at cell 6 had 83 s to run. Card peaks 6,163 and 6,203 MiB; the CPU 12-14% busy | 2026-10-05 |
| A failure stop from a real exception (WAI v17, fp8): a 1024x1344 source refined at 1.5x with Strength "0.2,0.3,0.4,0.5,0.6" and Versions 2 (10 cells, seeds outermost), the source deleted from its gallery card with the page's own dialog once cell 1 had landed | The dialog said "#2 was refined from this one; once it is gone, that pass cannot be made again."; the card became "Deleted · #1" 0.9 s after cell 1 landed (7.4 s). Cell 2 had read its source as it began and landed (9.6 s); cells 3, 4 and 5 then failed within 2 ms of one another (`ValueError: no gallery image with id 1`), so the cell line's live "1 skipped" and "2 skipped" were drawn and replaced in the same few milliseconds and never on screen. The job ended at 17.4 s: `stopped_by` failures, "Stopped at cell 5 of 10 after 3 cells failed; 2 images kept. Last: ValueError: no gallery image with id 1", the tab "Stopped", the sweep's sheet open on `job:` reading "2 of 2 loaded · source #1 (deleted)", and the finish line "Resume the remaining 8 cells". Pressed, Resume was refused at submit and the line under Generate said only "no gallery image with id 1"; the button stayed live and focused, so every press is refused the same. The card was back at 2,341-2,351 MiB (loaded, idle) within 3 s of the stop; peak 7,126 MiB. The pass that landed after the delete was shown with its source still there (see the next row). Another project's processes kept the CPU 28-34% busy | 2026-10-05 |
| A pass that lands after its source was deleted during its own cell (WAI v17, fp8; the sweep above, and again as a two-cell sweep with the source deleted while cell 2 ran) | Both times the late pass came back from the job with `refine_missing` false - `storage.save_image` builds its record without looking the source up - and the page's `forgetImages` marks only the records it already held, so in the viewer that pass read "refine of #N @ 0.3" with no "(deleted)", and its ↻ "Open the Refine dialog on #N with this pass's size, strength and seed", while the pass beside it that had landed before the delete said the source had gone. After the gallery reloaded, the same image's card said "(deleted)" and "cannot be made again", and its tile in the viewer still did not. Fixed the same day in 2b89ed5: `save_image` looks the source up in the transaction it inserts in, under the write lock a delete takes, a job's replay marks the passes whose source a delete took, and the page marks an arriving pass whose source it has seen deleted (`deletedIds`, `markGoneSource`); checked on the fake (`tests/test_api_refine.py`, smoke-refine, the source deleted mid-pass), not again on the card | 2026-10-05 |
| A load started while the page's first adapter listing runs (WAI v17, fp8; a fresh process each time, Load pressed 0.8-3 s after the page opened) | Failed 3 times out of 3: "Float8Tensor.__new__() got multiple values for argument 'qdata'", raised through accelerate's `register_empty_parameter` while torchao quantized a component's weights (a CLIP encoder's layer, or a diffusers component). The listing (`GET /api/loras`, which the page asks on its first status) builds the denoiser on the meta device under `accelerate.init_empty_weights()` (`loading.denoiser`), which patches `torch.nn.Module.register_parameter` for the whole process while it lasts; on a cold process it answered 2.8 s after the page opened (WAI, the CPU idle) and 23 s after (FLUX.1 Krea [dev], another project's processes saturating the CPU). Load pressed once the listing had answered: loaded in 5.4 s; a load asked again in the failed process: loaded. Anima Turbo, whose profile takes no adapters, never builds it. Fixed the same day in 02cd1d9: the build goes through `loading.empty_weights` and never overlaps a load - `load()` holds `loading.weights_loading()` throughout and waits out a build in progress, a build asked for during a load raises `LoadInProgress` and the adapter is listed unchecked, and builds go one at a time. Rechecked on the card that evening the same way (WAI v17, fp8, model offload, a fresh process each time, the card at 2,096-2,107 MiB before each start, Load pressed 0.87-0.88 s after the page opened, `frosting-lane.safetensors` in the adapter directory): loaded 3 times out of 3, in 5.6, 3.8 and 3.6 s; each time the listing arrived during the load and logged "adapter fit unchecked until the load: a model load is in progress", and the listing after the status turned ready answered from the live denoiser (the FLUX.1 adapter hidden as a misfit). The card was back at 2,093-2,118 MiB after each stop. Evidence under `d1-fix/` | 2026-10-05 |
| The failed-load card against the real Hub, with no token and nothing cached (`HF_HOME` an empty scratch directory, `HF_TOKEN` unset, Load pressed at 2560x1305; then the card's Retry load) | FLUX.1 [dev]: `GatedRepoError` on `model_index.json`, 6.8 s after the press; `detail_kind` gated, "FLUX.1 [dev] is gated on Hugging Face: accept its terms on the model page (a click-through), run `uv run hf auth login` once with a read token, then press Retry load.", the command set as code, "Open the model page" to https://huggingface.co/black-forest-labs/FLUX.1-dev in a new tab, a primary Retry load. Krea 2 Raw: the same in 6.8 s, worded "ask for access on the model page and wait for its authors to approve it". A made-up id: `RepositoryNotFoundError` in 3.7 s, `not_found`, "local-diffusion-check/no-such-checkpoint is not on Hugging Face, or is private: check the checkpoint id (LOCAL_DIFFUSION_MODEL_ID, or the start-up menu); for a private repo, …", headed "Couldn't load no-such-checkpoint", the link to the id's (missing) page. Each time the topbar read the checkpoint's name and "· load failed", its Load relabelled "Retry load" and keeping the press's focus, Unload hidden, nothing in the panel's line; the card's Retry load failed again in 1.4-1.5 s with one card, focus on Generate (the card went during the load, and the one that came back does not take it). With `HF_HUB_OFFLINE=1` on an uncached FLUX.1 Krea [dev]: huggingface_hub 1.29.0 raises `LocalEntryNotFoundError` from an `OfflineModeIsEnabled`, `main._hub_failure` takes the outer one, and the card said "FLUX.1 Krea [dev] is not all downloaded yet, and Hugging Face cannot be reached: connect to the network, then press Retry load." - the network was up; the offline sentence ("HF_HUB_OFFLINE is set: unset it for one load") never showed. Nothing was downloaded: the scratch `HF_HOME` held 5,778 bytes after all four, huggingface_hub's own `.agent_harnesses.json`; the adapter listing logged "adapter fit unchecked until the load: 401". The offline sentence was fixed the same day in 81781bf: `_hub_failure` prefers `OfflineModeIsEnabled` wherever it is on the chain, tested on the chain huggingface_hub's own `hf_hub_download` builds under its offline switch (`tests/test_api.py`), not again on the card | 2026-10-05 |
| The critic's first-use line and clock over a real download (the FLUX.1 Krea [dev] profile, an empty `HF_HOME`, the image model never loaded) | The line named what was still to fetch and changed as each piece landed, and the clock ran to "· 199 s" for a 199.7 s first use whose route `seconds` said 9.4, the critique alone; the row is in the Prompt analyser section | 2026-10-05 |

The rows dated 2026-10-05 that drive the page ran the real server
(`--defaults`, fp8, model offload, port 7860), one at a time and each on a
throwaway database and outputs directory, with the card holding only the
desktop (1,956-2,227 MiB) before each start. The page was headless Edge at
2560x1305 unless a row says otherwise, pressed with real mouse events by
`tools/browser.mjs` steps (all but the strip's through a copy that stops at
the first failed expectation) and measured in the page
(`getBoundingClientRect` under a MutationObserver on the strip, the stream's
messages timed as they arrived). "Card" is `nvidia-smi`'s whole-card
`memory.used`, polled every half second. Once the load race (above) was
found, each step file waited for the page's first adapter listing before it
pressed Load or Generate. Evidence is under
`%TEMP%/local-diffusion-page-gpu/`: `harness.py`, the step files and the
scripts that wrote them, and per check its `server.log`, `browser.log`,
`smi.csv`, `cpu.csv` and screenshots.

The rest of what these rows found was fixed the same day and checked on the
fake server and in pytest, not again on the card: a fitted batch's columns
reserved from its first frame, so the preview or stand-in holds the first
image's slot (9377de1); the strip fitted as if the page's scrollbar were
there, so a short gallery's batch of four no longer refits as the job ends
(7733f81; a stable gutter on the page, tried first, narrowed every vw size
and was reverted); the budget line
counting seeds by the job's stop rule, 5 of 6 for the Kohaku grid above
(6a7570d); a sweep's Resume held in words once its source is gone (ad01fdc);
the estimate of the time left running from the end of the first step (2a4d2cc);
and, in the Prompt analyser section, the critic's clock on the line with its
words (b396e2e) and the page's sizes labelled in binary units, "about 4.5 GiB"
(5ec8adb).

Not verified: `LOCAL_DIFFUSION_DEVICE=cuda:1` (one card here); the fp8 path for klein
on Windows; ERNIE-Image Turbo on the fp8 path; ERNIE's new explicit prompt
expansion path on the card; a live-preview map for FLUX.2's VAE (ERNIE, klein);
Animagine XL 4.0 LoRA behavior and four-channel SDXL VAE preview fit (its fp8
speed and memory are the October 5 row above; the quantization and offload
settings of its earlier successful user run were not recorded). Illustrious XL
v2.0-STABLE int8/full-bfloat16 GPU runs, LoRA behavior, batch scaling and
repeat-seed determinism are also unmeasured; its fp8 peak is the October 5
row.

## Refine pass

Measured 2026-10-03 on the RTX 5080, Windows, fp8, model offload, against
throwaway galleries (the production gallery was not used). The idle desktop
read 2,470-2,570 MiB of the card by `nvidia-smi`. Each run loaded the
checkpoint once and made a
1 MP source, then refined it at 1.5x. Times are per image and exclude the
load and the PNG save; "peak" is the process's own allocator. A refine row's
Steps column is the schedule's length, with the steps the pass ran beside it.

| Checkpoint | Pass | Steps | Time | Peak | Direct render, for scale |
| --- | --- | --- | --- | --- | --- |
| WAI Illustrious v17, CFG 6 | 1024x1344 to 1536x2016 at 0.4 | 28, 11 run | 11.5-13.5 s | 3.89 GiB allocated, 4.70 reserved | 1536x2016, 28 steps: 23.0 s, 3.73 / 4.54 GiB; the 1024x1344 source 10.0-13.3 s, 3.22 / 4.37 GiB |
| WAI Illustrious v17, CFG 6 | the same at 0.2 / 0.3 / 0.5 / 0.6 | 28; 5 / 8 / 14 / 16 run | 7.2 / 9.5 / 13.8 / 15.2 s | 3.89 / 4.70 GiB | |
| FLUX.1 Krea dev, CFG 3.5 | 1024x1024 to 1536x1536 at 0.1 | 28, 3 run | 16.9 s | 12.54 GiB allocated, 13.56 reserved | 1536x1536, 28 steps: 101.3 s, 12.38 / 13.39 GiB; the 1024x1024 source 43.0 s (56.8 s for the first of the process), 12.0 / 12.62 GiB |
| FLUX.1 Krea dev, CFG 3.5 | the same at 0.05 / 0.15 / 0.2 / 0.3 / 0.4 / 0.5 / 0.6 | 28; 2 / 5 / 6 / 9 / 12 / 14 / 17 run | 12.8 / 22.6-34.7 / 27.4 / 39.0 / 49.6-58.2 / 55.1 / 66.1 s | 12.54 / 13.56 GiB | |
| FLUX.1 Krea dev, Mu 1.15 | the same at 0.2 / 0.3 | 28; 6 / 9 run | 33.5 / 36.3 s | 12.54 / 13.56 GiB | |
| Z-Anime Base, CFG 4 | 1024x1024 to 1536x1536 at 0.15 | 40, 6 run | 33.0-33.8 s | 8.10 GiB allocated, 10.58 reserved | not run; the 1024x1024 source 73.8 s (88.9 s for the first), 6.77 / 8.44 GiB |
| Z-Anime Base, CFG 4 | the same at 0.1 / 0.25 | 40; 4 / 10 run | 23.8 / 52.7 s | 8.10 / 10.58 GiB | |

FLUX.1 at 2.4 MP sits within a gigabyte of the card's top with the desktop's
share (15,600-15,800 MiB whole-card by `nvidia-smi` mid-pass), which is the
spill the note at the head of this file describes: its per-step time there
is about 4 s where the 1 MP source's is 1.5 s.

Sharing the loaded weights does not imply the same peak memory. With model
offload, the source's VAE encode leaves the VAE resident beside the denoiser
until cleanup; WAI and FLUX.1 above peak about 0.16 GiB higher than their
same-size direct renders. The October 3 review's tiny FLUX probe confirmed
that residency ordering. It did not change offload behavior or measure other
offload modes.

| What | Result |
| --- | --- |
| The sibling over the loaded components | `StableDiffusionXLImg2ImgPipeline`, `FluxImg2ImgPipeline` and `ZImageImg2ImgPipeline` each ran from `refine.build` over the fp8-quantized, offload-hooked components of the loaded pipeline, with no second load; the device read 1.37-1.39 GiB after a pass, as after a text-to-image call |
| Determinism | The same pass twice is the same hash, on all three |
| Per-image seeds in a batch | A batch of two: image 1 is the hash of the solo pass at its seed and image 2 the hash of the solo pass at seed + 1, on all three - FLUX.1 included, whose own batch path would have encoded both from the first generator |
| Cancel | Cancelled at step 2 of the pass (3.9 s on WAI, 20.1 s on FLUX.1, 13.1 s on Z-Anime); the next pass is the uncancelled hash |
| The text-to-image pipeline afterwards | The source's own request, run again after the passes, the batch and the cancel, is the source's hash on all three |
| Preview | Frames delivered during a pass on FLUX.1 (11 of 12 steps) and Z-Anime (5 of 6); none on SDXL, which has no fitted map for its 4-channel VAE |
| A LoRA through the pass | FLUX.1 Krea dev, `frosting-lane.safetensors` at strength 0.15: weight 0 is byte-identical to no adapter, weight 1 is a different image, and with the adapter removed the base hash returns |
| The picture | On each, the refined image is the source's composition with the detail redrawn, and the direct render at the larger size on the same seed is a different picture (WAI: another pose and framing; FLUX.1: another man, unhooded, elsewhere in the frame) |
| What a strength keeps, WAI | 0.2-0.6 all keep the character and composition; from 0.5 small details change (a tear appears, a strap moves) |
| What a strength keeps, FLUX.1 at 1.5x | 0.05 and 0.1 keep the face; 0.15 begins to change it; 0.2-0.6 keep the composition and repaint the man. The pass's starting sigma, read off the scheduler: 0.37 at 0.05, 0.47 at 0.1, 0.62 at 0.15; with Mu 1.15, 0.46 at 0.2 and 0.60 at 0.3 |
| What a strength keeps, Z-Anime at 1.5x | 0.1, 0.15 and 0.25 all keep the character and the scene; 0.25 moves hair and small details |

These are single prompts per checkpoint, looked at by eye: they establish
that the pass runs, reproduces and keeps the picture at each profile's
starting strength, not a quality benchmark.

### Past 4 MP

Measured 2026-10-04, the same card, Windows, fp8 and model offload, for the
pass's own size guard (any 2 MP preset at 2x, 8.4 MP; see "The refine pass"
in the design notes). The same three sources as above (seed 12345), each
pass at its profile's starting strength unless a Mu is named. The desktop
held 1,765-2,580 MiB of the card. "Card" is `nvidia-smi`'s whole-card
`memory.used`, polled every half second; a reserved figure past the card's
16 GiB, or per-step times that climb, is the driver spilling into system
memory.

| Checkpoint | Pass | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- | --- |
| WAI v17, 0.4 | 1024x1344 to 1536x2016, 3.1 MP | 11 of 28 | 15.2 s (0.8 s) | 3.89 / 4.70 GiB | 7,676 MiB |
| WAI v17, 0.4 | to 2048x2688, 5.5 MP | 11 of 28 | 24.1 s (1.7 s) | 4.61 / 5.98 GiB | 9,041 MiB |
| WAI v17, 0.4 | to 2560x3360, 8.6 MP | 11 of 28 | 46.2 s (3.4 s) | 5.53 / 7.60 GiB | 11,363 MiB |
| Z-Anime, 0.15 | 1024x1024 to 1536x1536, 2.4 MP | 6 of 40 | 39.6 s (5.2 s) | 8.10 / 10.58 GiB | 13,714 MiB |
| Z-Anime, 0.15 | to 2048x2048, 4.2 MP | 6 of 40 | 79.2 s (10.7 s, rising to 14.4 s) | 9.73 / 14.11 GiB | 15,936 MiB |
| Z-Anime, 0.15 | to 2896x2896, 8.4 MP | 6 of 40 | 715.1 s (107-125 s) | 13.46 / 21.94 GiB | 15,924 MiB, spilling |
| FLUX.1 Krea dev, 0.1 | 1024x1024 to 1536x1536, 2.4 MP | 3 of 28 | 32.5 s (3.7 s) | 12.54 / 13.56 GiB | 15,278 MiB |
| FLUX.1 Krea dev, 0.1 | to 2048x2048, 4.2 MP | 3 of 28 | 77.0 s (16-22 s) | 13.08 / 15.16 GiB | 15,922 MiB, spilling |
| FLUX.1 Krea dev, 0.2, Mu 1.15 | to 2048x2048, 4.2 MP | 6 of 28 | 152.4 s (24 s) | 13.08 / 15.16 GiB | 15,918 MiB, spilling |
| FLUX.1 Krea dev, 0.2, Mu 1.15 | to 2896x2896, 8.4 MP | stopped by hand mid-pass at 20.2 GB of dedicated and shared GPU memory | - | - | - |
| Animagine XL 4.0, 0.4 | 1024x1024 to 1536x1536, 2.4 MP | 11 of 28 | 8.8 s (0.54 s) | 3.67 / 4.32 GiB | 7,072 MiB |
| Animagine XL 4.0, 0.4 | to 2048x2048, 4.2 MP | 11 of 28 | 17.0 s (1.1 s) | 4.22 / 5.26 GiB | 8,033 MiB |
| Animagine XL 4.0, 0.4 | to 2896x2896, 8.4 MP | 11 of 28 | 41.2 s (3.1 s) | 5.47 / 7.49 GiB | 10,311 MiB |
| Kohaku XL Zeta, 0.4 | 1024x1024 to 1536x1536, 2.4 MP | 11 of 28 | 9.4 s (0.53 s) | 3.67 / 4.32 GiB | 7,074 MiB |
| Kohaku XL Zeta, 0.4 | to 2048x2048, 4.2 MP | 11 of 28 | 17.9 s (1.1 s) | 4.21 / 5.26 GiB | 8,049 MiB |
| Kohaku XL Zeta, 0.4 | to 2896x2896, 8.4 MP | 11 of 28 | 42.4 s (3.0 s) | 5.47 / 7.48 GiB | 10,311 MiB |
| Illustrious XL v2.0, 0.4 | 1024x1024 to 1536x1536, 2.4 MP | 11 of 28 | 9.0 s (0.53 s) | 3.67 / 4.52 GiB | 7,344 MiB |
| Illustrious XL v2.0, 0.4 | to 2048x2048, 4.2 MP | 11 of 28 | 19.0 s (1.1 s) | 4.22 / 5.26 GiB | 8,386 MiB |
| Illustrious XL v2.0, 0.4 | to 2896x2896, 8.4 MP | 11 of 28 | 43.6 s (3.1 s) | 5.47 / 7.48 GiB | 10,397 MiB |
| FLUX.1 schnell, 0.5 | 1024x1024 to 1536x1536, 2.4 MP | 2 of 4 | 15.4 s (3.6 s) | 12.53 / 13.65 GiB | 15,680 MiB |
| Z-Image Turbo, 0.25 | 1024x1024 to 1536x1536, 2.4 MP | 2 of 8 | 10.0 s (2.6 s) | 7.04 / 8.84 GiB | 11,304 MiB |
| Z-Image Turbo, 0.25 | to 2048x2048, 4.2 MP | 2 of 8 | 20.4 s (4.0 s) | 7.86 / 10.09 GiB | 12,493 MiB |
| Z-Image, CFG 5, 0.25 | 1024x1024 to 1536x1536, 2.4 MP | 13 of 50 | 73.1 s (5.1 s) | 8.10 / 10.58 GiB | 13,052 MiB |

The rows from Animagine down were measured 2026-10-05, the same way, one
size per run (`ladder.py`, which makes a 1024x1024 seed-12345 source once
per throwaway gallery and then runs one pass), each run stopping the ladder
at the first size near the card's top. The desktop held 1,744-2,745 MiB of
the card. Another project's processes kept the CPU near 75% busy
throughout, so the times are upper bounds; the memory is not affected.
Each per-step time leaves out the first step, which carries the source's
VAE encode. The three SDXL profiles run the guard's own top size with a
third of the card to spare, as WAI does; schnell sits within a gigabyte of
the top at 2.4 MP, as Krea dev's same transformer does, and was taken no
further; Z-Image Turbo, with no CFG, reaches 4.2 MP at 12.5 GB, and 8.4 MP
was not run, its growth from 2.4 to 4.2 MP putting it past the card; the
undistilled Z-Image, with CFG, reads as Z-Anime does at 2.4 MP and was
taken no further, Z-Anime having filled the card at 4.2 MP. The passes'
start sigmas, off the sibling's scheduler: schnell 0.5 at 0.5 (its
unshifted schedule), Z-Image Turbo 0.5 at 0.25 at both sizes, Z-Image
0.678 at 0.25. The pictures were not reviewed.

Under Nunchaku (WSL 2, the lite-infer NVFP4 re-uploads, model offload),
measured 2026-10-05 the same way. Qwen-Image fits this card only this way
(the catalogue says so), so its row is its profile's measured line;
FLUX.1 dev's shows a pass running on a lite repo at all, which the lite
repos' profile otherwise takes from Krea dev's fp8 run. Seeds do not
reproduce bit for bit here (Behaviour, above), so no hash is recorded.

| Checkpoint | Pass | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- | --- |
| Qwen-Image, true CFG 4, 0.3 | 1024x1024 to 1280x1280, 1.6 MP | 15 of 50 | 91.1 s (3.0 s; the first 40.1 s) | 12.20 / 12.46 GiB | 14,968 MiB |
| FLUX.1 dev, 0.1 | 1024x1024 to 1536x1536, 2.4 MP | 3 of 28 | 17.3 s (1.5 s) | 8.05 / 9.41 GiB | 11,889 MiB |

Qwen-Image's pass ran within 1.4 GB of the card's top and was taken no
further: the generic 1600x1600 line would have opened the dialog past what
was measured. Its start sigma was 0.478 at 0.3; FLUX.1 dev's 0.474 at 0.1,
Krea dev's fp8 figure on the same schedule.

What fits is the checkpoint's: WAI reaches 8.6 MP with the card two-thirds
used, while Z-Anime fills it at 4.2 MP and runs 8.4 MP at about 20 times
the per-step time of 2.4 MP, and FLUX.1 fills it already at 2.4 MP (its
1.5x pass here took 32.5 s where the October 3 run, under a smaller
desktop share, took 16.9 s). The pictures, looked at downscaled side by
side: WAI keeps the character and the composition at all three sizes, and
Z-Anime the character and the scene. FLUX.1 at 2x and its own 0.1 starts
at sigma 0.752, as its size-dependent shift predicts (design notes), and
repaints the man - another face, coat and hands in the same composition -
where Mu 1.15 at 0.2 starts at 0.463 and keeps him. The FLUX.1 8.4 MP pass
was not repeated. Evidence is under `%TEMP%/local-diffusion-refine-large/`:
`large_refine.py` (the measuring script, which lifts the size guard before
the schema is built), each checkpoint's copied gallery with its
`validation.log` and outputs, and `strip-*.jpg` (`strips.py`).

Not measured: FLUX.1 dev at fp8, whose starting strength carries over the
Krea dev measurement on the same schedule; what schnell, Z-Image Turbo,
Z-Image and Qwen-Image's starting strengths keep of a picture, which is
schedule arithmetic alone; int8 and full bfloat16; a Nunchaku pass past
the two above; and the author's own WAI recipe, which upscales with
R-ESRGAN 4x+ Anime6B where this pass resizes with Lanczos. The October 5
runs' evidence is under `%TEMP%/local-diffusion-leftovers-gpu/` (one
throwaway gallery per checkpoint with its `ladder.log` and `ladder.jsonl`)
and, for the Nunchaku runs, `~/ld-gpu/` in WSL.

The October 3 runs' evidence is under
`%TEMP%/local-diffusion-refine-validation/`: `validate_refine.py` and
`sweep_refine.py`, and per checkpoint (`wai`, `flux`, `zanime`) the
`validation.log`, `results.json`, the sweeps' JSON, the separate database
and the `outputs/` PNGs. The browser checks are `tools/smoke-refine.json`
(the Refine dialog), `smoke-refine-unavailable.json` and the additions to
`smoke-conditioning.json`.

## Inpaint

Nothing measured yet. Inpainting is built for the SDXL and Z-Image families
(`docs/plan-inpainting.md`, phases 1 and 2) and checked on the CPU and in
the browser only: `tests/test_inpaint*.py`, `tests/test_api_inpaint.py`,
`smoke-inpaint.json` and `smoke-inpaint-unavailable.json`, on fakes that
repaint the masked pixels with a flat colour. Not yet measured on the card,
from the plan's open measurements:

- peak memory and time per step per enabled family at 1 MP, and the largest
  size each holds (SDXL: WAI, then Animagine or NoobAI; Z-Image Turbo and
  Z-Anime);
- determinism, batch image *i* against a solo run of its seed,
  cancel-then-rerun, and the text-to-image hash and the device after
  inpaints;
- `refine.build` over the fp8-quantized, offload-hooked components for
  `StableDiffusionXLInpaintPipeline` and `ZImageInpaintPipeline`;
- a starting strength per profile, against the library's default the dialog
  starts at now;
- the seam the composite leaves against the VAE round trip's colour shift,
  with and without feather;
- a LoRA through an inpaint, weight 0 byte-identical to none;
- an inpaint over a Nunchaku Z-Image transformer under WSL;
- the embedded mask's size in a result PNG at 1 MP and 8 MP;
- for the deferred families (FLUX.1, FLUX.2 [klein], Qwen-Image), the
  measurements their phases name before they are enabled.

## Outpaint

Nothing measured yet. Outpainting's backend and its dialog are built
(`docs/plan-outpainting.md`, phases 1 and 2) and checked on the CPU and in
a browser only: `tests/test_outpaint*.py`, `tests/test_api_outpaint.py`,
the outpaint tests in `tests/test_cli.py`, and `tools/smoke-outpaint.json`
on a fake that repaints the masked pixels with a flat colour. Not yet measured
on the card, from the plan's open measurements:

- peak memory and time per step at outpainted sizes for WAI and one other
  SDXL checkpoint, Z-Image Turbo and Z-Anime, one size per run, stopping at
  the first that fills the card;
- determinism: the same outpaint twice, batch image *i* against a solo run
  of its seed, cancel-then-rerun, the text-to-image hash afterwards;
- what the fill contributes at the library default (0.9999 on SDXL, 1.0 on
  Z-Image) and at lower strengths;
- the seam at band widths 0, 32, 64 and 128 px per family;
- a starting band width, if one is measured to join better than 32;
- the embedded mask's size in an outpaint's PNG;
- the fill's and the mask's build time on the CPU at the refine guard's
  largest canvas.

## Prompt analyser

Measured 2026-09-27 on the real server (`--quant fp8 --offload model`, port
7880, a throwaway database and outputs directory) with the desktop holding
1,746-1,791 MiB of the card's 16,303 MiB. VRAM in this section is
`nvidia-smi`'s whole-card `memory.used`, in MiB; the critic's is polled every
0.5 s and given less that baseline. Critique times are the route's
`seconds`: server start, one chat completion and the stop. The rows dated
2026-09-28 check the 2026-09-28 review's fixes; their critiques ran through
`PromptCritic` directly, by then with the server counting the question
(`/apply-template`, `/tokenize`) between its start and the completion. The
rows dated 2026-09-29 ran through the app's own routes on the final tree
of those fixes (8931fae): the real server with `--defaults` (fp8, model
offload), port 7990, a scratch database and outputs directory, the image
model never loaded. The card held 2,960-3,017 MiB with the app stopped and
3,178-3,238 MiB with it up and idle before each critique, the baseline
those rows subtract: about 220 MiB of the difference is the app's own CUDA
context, which `/api/status`'s memory query creates.

| What | Result | Date |
| --- | --- | --- |
| First use on Windows: `prompt_critic.prepare()` on an empty cache | 114.2 s. Both b11206 win-cuda-13.4 zips (575.9 MB: 152,343,617 + 423,535,356 bytes) downloaded, held to size and sha256, extracted and renamed into place in 21.8 s; the Qwen3-4B Q8_0 GGUF (4.28 GB, 4,280,404,704 bytes) downloaded and hashed in 92 s. `ready()` False before, True after; a second `prepare()` took 0.013 s and fetched nothing. The tree is 744.4 MB (744,407,545 bytes by `du -sb` on 2026-09-28; `du -sh` showed 711M, in MiB, on both days) at `~/.cache/huggingface/local-diffusion/llama.cpp/b11206-win-cuda-13.4-x64`, `downloads/` is left empty, the GGUF's stamp is under `verified/`, and an independent `sha256sum` of the GGUF matched the pin | 2026-09-27 |
| First use under WSL 2: the ubuntu-cuda-13.4 tarballs | 62.3 s for 591.9 MB (151,688,270 + 440,236,702 bytes). The flattened tree (810.4 MB: 810,355,647 bytes by `du -sb` on 2026-09-28; `du -sh` showed 773M, in MiB, on both days) has `libcudart.so.13`, `libcublas*.so.13` and `libggml-cuda.so` beside an executable `llama-server`, symlinks intact, and `ldd` resolves every library inside it: the layout read from b11206's `release.yml` holds (with libgomp1 already on the system, from gcc; a fresh WSL Ubuntu has none, see the row on the tree's system dependencies) | 2026-09-27 |
| The Ubuntu b11206 tree's system dependencies, read off its ELF headers under WSL 2 (Ubuntu 26.04, glibc 2.43) | Beyond libc, libm, libstdc++, libgcc_s and the driver's `libcuda.so.1`: `libgomp.so.1` (`libggml-base` and every `libggml-cpu-*`), `libssl.so.3` and `libcrypto.so.3` (`libllama-common`, `libllama-server-impl`); newest symbol versions GLIBC_2.38 and GLIBCXX_3.4.32. Here they come from libgomp1 and libssl3t64, which provides libssl3. `critic.missing_libraries` found all three (and flagged a made-up soname) in 0.025 s; `preflight` passed and `supported()` was True. `wsl/setup.sh`'s check under faked ldconfig/getconf: libgomp1 when only an i386 libgomp is cached, libssl3 once for both OpenSSL sonames, the glibc note alone at 2.35, nothing at exactly 2.38 or when ldconfig answers nothing | 2026-09-28 |
| Analyse on FLUX.1 Krea [dev] | 1.9 s for the first call (tokenizers from the Hub cache), 0.00-0.03 s after. The CLIP-L/T5 counts for four prompts (12/17, 142/173, 43/49, 42/54) equal the checkpoint's own `CLIPTokenizer` and `T5TokenizerFast` called as `FluxPipeline` calls them, and the 142-token prompt's CLIP cut (character 379) is where the pipeline's own truncation warning begins | 2026-09-27 |
| Critique on Windows, FLUX.1 Krea [dev] | 19 critiques of four prompts: 8.1-16.0 s, median 10.5 s; 895-1,074 prompt tokens and 801-2,011 generated at 145-154 tokens/s. All 19 were back at the baseline a second later with no `llama-server.exe` left, and the 15 whose VRAM was polled (the first run's poll wrote nothing) peaked at 5,581-5,584 MiB over it; `dropped` 0 throughout. The same with the FLUX pipeline loaded and offloaded: 9.3 s, +5,582 MiB | 2026-09-27 |
| Critique on Windows, Animagine XL 4.0 tag prompt | 19.3 s: 905 prompt tokens, 2,568 generated; +5,581 MiB | 2026-09-27 |
| Critique under WSL 2 (FLUX.1 dev lite-infer id, through `PromptCritic` directly) | With `LOCAL_DIFFUSION_CRITIC_MODEL` on `/mnt/c`, 28.7 and 29.0 s in one WSL boot: reading the 4.28 GB GGUF across the 9P bridge costs about 20 s on every start, not only the first (a `cp` of it out of `/mnt/c` took 20.6 s). 9.7 s with the GGUF copied into `/tmp`, which is tmpfs on this Ubuntu 26.04: a read from RAM, not from ext4. The default, a GGUF in the Hugging Face cache on ext4, was not measured here; the 2026-09-29 WSL row below measures it. +5,611 MiB (7,038 less 1,427), back to baseline, no server left | 2026-09-27 |
| An image job submitted during a critique | Queued at 18:42:25.1 with `waiting_on=load` while the critic's server ran; the card was back at its baseline by 18:42:34.6, the critique returned at 35.0 and `loading black-forest-labs/FLUX.1-Krea-dev` was logged at 36.1. No weights moved before the server stopped | 2026-09-27 |
| A critique during a grid | 409 "an image job is queued or running; wait for it to finish"; no server started | 2026-09-27 |
| Critique on Windows after the final review's fixes (API key in `LLAMA_API_KEY`, not on the command line) | 20 critiques of five FLUX.1 Krea [dev] prompts: 7.4-22.3 s, median 13.2 s; 1,105-1,192 prompt tokens, 705-2,511 generated at 123-157 tokens/s. The keyless `/v1/models` probe answered 401 every time, so b11206 took the key from its environment; the card was back at its 1,533-1,547 MiB baseline and no `llama-server.exe` was left after each run | 2026-09-27 |
| Critique after the 2026-09-28 review's server fixes (process tree, listener check, the server's own count), Windows | One FLUX.1 Krea [dev] prompt, "A small bookshop on a rainy street at night, a wooden sign that says open late above the door, warm light in the windows, shot on film", asked with an analysis naming no documented order - so `reordered` was held to null and the answer sized for two copies of the prompt, where the route reads Krea's order ("subject first") and asks for three, with a slightly longer question: 21.1 s, 1,072 prompt tokens, 2,394 generated at 139 tokens/s. The server counted the messages through `/apply-template` and `/tokenize` at 1,072, its `prompt_n`. llama-server.exe started suspended in a job object and was the port's listener in the TCP table, so the check passed. After the stop the job was empty, no handle was held, no llama-server.exe was left, the log was removed and the card was at baseline. A 2,500-character Greek prompt, estimated at 3,902 tokens in and 2,552 for the prompt and so admitted by the first check, was counted at 3,230 and 2,189 (8,408 with two copies and the overhead, past the 8,192 window; with the route's three, the estimate would still admit it and the count refuse it) and refused unasked in 3.1 s, the server stopped | 2026-09-28 |
| The same under WSL 2 (GGUF from `/mnt/c` via `LOCAL_DIFFUSION_CRITIC_MODEL`) | The Krea [dev] prompt: 38.1 s, 1,072 prompt tokens, 1,127 generated at 136 tokens/s, counted again at 1,072. The listening socket's inode was held in the server's process session, so the check passed. The Greek prompt was refused on the count in 27.8 s, most of it the model's read over 9P. The process session was empty after each stop, no llama-server was left and the card was at baseline | 2026-09-28 |
| Qwen3-4B's tokenizer for the critic's estimate (CPU) | FLUX.2 [klein]'s `tokenizer.json` gives the same ids as b11206's `llama-tokenize` on the pinned GGUF for 31 texts. Against it, `estimate_tokens` (NFC, per-script rates, a token at least per camel-case hump) is long by up to 1.59 on ordinary text (Vietnamese; English prose 1.26-1.39) and 2.3-2.6 on lists of identifiers or brands. It is short on Turkish 0.84, Polish 0.87, Icelandic 0.82, romanised names 0.83, base64 0.78, traditional Chinese 0.92, kanji names 0.75, Cantonese 0.61 and rare Han 0.50, which the server's own count covers once it runs | 2026-09-28 |
| Wrapped-server stop under full CPU load, Windows (stand-in server behind a .bat) | 30 critiques with the log's removal retry off: the log was still held after the stop in 0 of 30, against 17 of 30 when only the job's process list was polled | 2026-09-28 |
| An app killed outright on Windows (stand-in server) | A server started through `_ServerProcess` stopped listening within a second of its app's TerminateProcess: the job's kill-on-close. Not tried with llama-server | 2026-09-28 |
| Critique through the routes on the final tree, FLUX.1 Krea [dev], Windows | The 2026-09-28 row's bookshop prompt twice, `/api/prompt/analyze` then `/api/prompt/critique` with the body the page sends: 11.8 and 13.1 s, 1,177 prompt tokens both times (the route's analysis names Krea's "subject first", so the question asks for three copies of the prompt where the 2026-09-28 run's asked for two at 1,072), 1,125 and 1,460 generated at 140 and 144 tokens/s. Check took 2.6 s on the fresh server (tokenizers from the Hub cache) and 0.03 s after: CLIP-L 30 and T5 34 tokens, the order "Not checked", `guide.quote_text` and `guide.words`. Both rewrites quoted "open late"; the model's `reordered` changed words and was set aside both times, so `reordered` stayed null, and `dropped` was 0. The first answer kept one issue (A5: "shot on film" is no concrete style) and left out one that restated the quotation-mark finding; the second had none. +5,570 MiB over a 3,231 MiB baseline on the second (the first run's poll wrote nothing); the card was back at its baseline before the route answered, and no `llama-server.exe` was left after either | 2026-09-29 |
| The same on Animagine XL 4.0, an unknown tag ahead of 1girl | "zzkx oc, 1girl, green hair, masterpiece, smile, outdoors", the Danbooru index reachable ("zzkx oc" not found; green hair, smile and outdoors general). Check did not say "Partly checked": `order_checked` true and `order_detail` null, since an unknown tag ahead of 1girl is out of order whatever its category, which `unplaced_items` counts as settled. It gave one `guide.order` finding ("zzkx oc" before 1girl), `reordered` "1girl, zzkx oc, green hair, smile, outdoors, masterpiece", `guide.quality` and `guide.negative_prompt`, in 3.3 s on the fresh server (tokenizers and the tag lookup); 15 tokens on both CLIPs. Critique 20.3 s: 883 prompt tokens, 2,521 generated at 145 tokens/s; +5,626 MiB over 3,178. Three issues were left out for restating the tag-order finding, and the model's `reordered` for changing words, so `reordered` and `reordered_kind` were null; `dropped` 0. The one issue kept, "Missing negative prompt" (G5), restates the `guide.negative_prompt` finding, which validation has no restatement check for. The rewrite dropped "oc", added "[series]" and "[rating]" placeholders and appended the card's recommended negative prompt to the prompt; the lint flagged both brackets, a CLIP cut at 75 of 77 tokens, the order, 26 added words and the missing negative prompt. No `llama-server.exe` was left | 2026-09-29 |
| The same, an unknown tag whose category decides the order | "1girl, green hair, zzkx oc, smile, outdoors, masterpiece": `order_detail` "the tag index has no usable Danbooru category for 1 tag, so it is not placed: "zzkx oc"" (the page's "Partly checked"), no order finding and no `reordered`. Critique 18.3 s: 843 prompt tokens, 2,268 generated at 146 tokens/s; +5,687 MiB over 3,184, probably about 100 of it not the server's (3,282-3,302 MiB in the 5 s after the stop). No issue was left out; four were kept: two `order` issues on A1 asking for a series and a rating after the character, the missing quality finish (A2, restating the `guide.quality` finding, which validation has no restatement check for either) and "No negative prompt", cited to A3, the card's tag-format sentence. The model's `reordered` changed words and was set aside before the partial-placement check, so that check did not run and `reordered_kind` stayed null. The rewrite again added "[series]", "safe", the quality finish and the whole recommended negative prompt, and drew the same lint findings, with one bracket this time. No `llama-server.exe` was left after the run or after the app was terminated | 2026-09-29 |
| Critique under WSL 2 through the routes, the GGUF where the app puts it (the Hugging Face cache on ext4), FLUX.1 Krea [dev] lite-infer id | The real server with `wsl/env.sh` sourced and `--defaults` on `1b8e9b4` (the app as at `5aa01f4`), port 7990, a scratch database and outputs directory, the image model never loaded; the bookshop prompt at 1024x1024, 1,191 prompt tokens every time. The WSL cache held the b11206 tree but no GGUF, so the first critique's `prepare()` downloaded the pinned file into `~/.cache/huggingface/hub`, hashed and stamped it: 96.4 s for the request, 21.6 s of it the critique (1,559 generated at 111 tokens/s). With the file in the page cache: 11.4 and 11.6 s (929 and 1,050 generated at 117 and 121 tokens/s). After `echo 3 > /proc/sys/vm/drop_caches` (the VM's `Cached` fell from 5,138 and 4,842 MiB to 569 each time): 17.5 and 18.5 s (1,127 and 945 generated at 121 and 100 tokens/s). Less the generation at the server's own rate, the rest - start, load, the prompt, the stop - was 2.9-3.5 s warm and 8.2-9.1 s after the drop: reading the 4.28 GB file from ext4 adds about 5-6 s, where `/mnt/c` added about 20. The drop empties the VM's page cache only; Windows may still have held the virtual disk's blocks. Generation ran at 100-121 tokens/s, against 140-154 on Windows. +5,610-5,663 MiB over a 3,197-3,223 MiB baseline (the desktop's 2,946 and the idle app's context), back to it after each, no `llama-server` left after any critique or after the server was stopped; `dropped` 0 throughout, `reordered` null, one issue kept in one answer | 2026-09-29 |
| The critic's first-use line and clock in the page, over a real download (Windows, the FLUX.1 Krea [dev] profile, the image model never loaded; `HF_HOME` an empty scratch directory holding only a token; Check, then Ask the critic, pressed at 2560x1305) | Before the press: "First use downloads llama.cpp b11206 and Qwen3-4B Q8_0 (about 4.5 GB) once." - 4,856,283,677 bytes, in binary gigabytes (4.9 GB decimal). Check took 7.4 s, the CLIP and T5 tokenizers fetched on the way (CLIP-L 30, T5 34 tokens). On the press: "Downloading llama.cpp b11206 and Qwen3-4B Q8_0 (about 4.5 GB) once - this can take a few minutes - then thinking…", Ask reading "Thinking…", Generate aria-disabled "Wait for the critic to finish", and the clock from "· 1 s" a second later, a tick a second; at QHD the clock wrapped to a line of its own under the sentence. The status poll after the server had installed llama.cpp (both zips, 548 MB, in 8.7 s) changed the line at 11.2 s to "Downloading Qwen3-4B Q8_0 (about 4.0 GB) once - …", and the one after the GGUF had landed (4.28 GB in about 181 s, at 3-88 MB/s) to "Thinking… about 15 seconds on the GPU." at 191 s. The answer came 199.7 s after the press with the clock at "· 199 s"; the route's `seconds` said 9.4, which counts the critique alone - server start, completion and stop: 1,201 prompt tokens, 939 generated at 143 tokens/s. The line after it: "Local Qwen3-4B Q8_0 on the GPU, about 15 seconds; waits for image jobs.", unchanged by the next poll. A second ask: "Thinking… about 15 seconds on the GPU.", the clock to "· 11 s", the answer at 11.1 s against `seconds` 11.1. +5,600-5,670 MiB over the idle app's 2,380-2,420 MiB. An attempt six minutes earlier, whose page driver stopped waiting at 280 s, took 306 s on the server with the GGUF at 2-46 MB/s; neither comes near the 114.2 s of 2026-09-27, and the line promises no time for the download ("this can take a few minutes") | 2026-10-05 |

**The FLUX prompts.** The Analyse row and the "Critique on Windows, FLUX.1
Krea [dev]" row used four prompts, given here as sent with their CLIP-L and
T5 token counts: "A red fox sitting in a snowy birch forest at dawn." (12
and 17); "A weathered lighthouse keeper in his seventies stands on the rocky
shore of a stormy North Atlantic coast, his yellow oilskin coat whipping in
the wind as he holds a brass lantern up against the dusk. Behind him the red
and white striped lighthouse rises from black basalt cliffs, its beam
cutting a pale arc through sheets of rain and low, bruised clouds. Waves
explode against the rocks in the foreground, throwing spray that catches the
last orange light from a break in the western sky. Shot on a medium format
film camera with an 80mm lens at f/4, shallow depth of field, cinematic
colour grading with teal shadows and warm highlights, fine film grain, a
dramatic and melancholic mood." (142 and 173; CLIP-L stops at character 379,
"the rocks in the foreground"); "A quiet neon-lit street in Tokyo at night
after the rain, no people, puddles reflecting the pink and blue lights, and
a small ramen shop with a wooden sign that says Open Late above the door."
(43 and 49; also the 2026-09-27 WSL critique's prompt and the comparison
grid's original); and "A (majestic:1.3) snow leopard resting on a rocky
ledge high in the Himalayas, ((golden hour light)), soft mist drifting
through the valleys below, highly detailed fur, [blurry background]" (42 and
54).

**Tag checkpoint.** On Animagine XL 4.0 the Danbooru index was reachable
(`/api/tags/validate` answered with categories: `hatsune miku` character,
`vocaloid` copyright). "masterpiece, high score, great score, absurdres,
looking at viewer, smile, hatsune miku, 1girl, vocaloid, safe, twintails,
cherry blossoms, outdoors, upper body" (39 tokens on both CLIPs, as the
checkpoint's own tokenizers count it) drew one `guide.order` finding citing
the card's order, and `reordered` was the same fourteen tags as "1girl,
hatsune miku, vocaloid, safe, looking at viewer, smile, twintails, cherry
blossoms, outdoors, upper body, masterpiece, high score, great score,
absurdres". The critic's three `order` issues, all citing G2, were correct;
its `reordered` equalled the rule's and passed the permutation check, and its
rewrite was the same text.

**Comparison grid.** Built with the page's `comparePayload`, seeds reduced to
2 to save time - the page itself offers only 4 or 8 (variants original and
the critic's rewrite of the "no people" / Open Late prompt, seed 20260927,
batch 1, 1024x1024, 28 steps, CFG 3.5). The preview
listed both prompts under `values.prompt` and the labels under `variant`; the
cells ran seed-major (original, rewrite, original, rewrite), and
`/api/gallery?search=run:8722b5a26b25` returned the four rows with
`variables={"variant": ...}`, which the PNG chunks carry too. All four images
render the sign as OPEN LATE, the unquoted original included; seed
20260927's original has one pedestrian and its "empty" rewrite none, seed
20260928's pair has nobody. Two seeds are an anecdote, not a measurement of
either rule. The images took 41.6-42.6 s each at 1.37-1.40 s per step, and a
repeat of the gate test's image matched its hash (`7d161ca06d8753e7`) in
41.9 s, with the GPU at 100 % utilisation, 255 W and 2.87 GHz throughout.
This is not the 2.7-fold slowdown it was first read as: the 15.4 s it was
set against, the Checkpoints table's 2026-09-08 FLUX.1 Krea dev time beside
the LoRA's, was a 1024x1024 image at 8 steps, not 28, and at 1.39 s a step
the 41.9 s image would take about 14 s at 8 steps. Before the analyser was
committed (e7d2386), a clean worktree of its parent, 7a9c651, rendered
through `backend.generate` with fp8, model offload, 1024x1024, 28 steps,
CFG 3.5 and seeds 20260927 and 20260928 (a harbour prompt): two images in
1m31s after a 32 s load, the second saved 42 s after the first. The ~42 s
predates the analyser. The CLI calls `backend.load()` itself, never
`main.ensure_loaded` or `jobs.wait_for_gate`, so that run exercised the load
and rendering code, which e7d2386 left unchanged, and not the change e7d2386
made to the server's load path (a load now first waits out the generation
gate). That wait comes once, before a load; the server's images above, with
it in place, took the same ~42 s.

**What the critic says.** Its judgements are right where a source says the
thing plainly: on "no people" and the unquoted sign it cited A7 and A6 in
every run, Windows and WSL, and rewrote to `...after the rain, empty,
puddles reflecting... a wooden sign that says "Open Late" above the door.`
It misapplies sources otherwise. Four of the 19 FLUX rewrites were the whole
prompt wrapped in quotation marks - it reads G1's "text in quotation marks"
as a rule for the prompt ("Text not enclosed in quotes as required by G1" -
"Enclose entire prompt in quotes"), which on FLUX asks for lettering. One
rewrite of the lighthouse prompt ended in `"Black Forest Labs"`, copied from
the authors' sign example, and one of the weighting prompt appended the owl
example's "cinematic dark moody photo, slight blue color grading". Two of
five lighthouse rewrites cut the whole camera and grading sentence, once on
the claim that camera details "may not align with the checkpoint's
photographic focus", two more the camera details alone, and its
`text_to_render` part held the scene description in all five (now nulled
unless it is lettering the prompt asks for, and said so under the answer).
Weighting syntax was cited to A7, the negative-prompt excerpt, in four of
five runs (A1 in the fifth), and "[blurry background]" was called a negative
prompt in four. It repeats the rule findings it is told not to
(word count, negation, quotes, weighting, tag order) in nearly every answer,
and counted the 11-word fox prompt as 7 words. `reordered` stayed null on
every FLUX prompt, each of which already opened with its subject. The rules
re-run on the rewrite catch none of the quoted-prompt or invented-text cases
(which is why `critic.validate` now checks the rewrite's quotation marks
itself).

**Prose out of order, before the final fixes.** Three FLUX.1 Krea [dev]
prompts with the subject last or behind style or setting ("Golden hour light
over misty mountains, shot on a 35mm lens ..., a lone hiker in a red jacket
standing on a ridge"), 10 critiques, 10.6-19.8 s. Check gave these the same
panel as the subject-first versions: "subject first", no finding, "nothing to
flag". Every critique that was kept whole (6 of 6) raised the subject order,
but a `reordered` survived `is_reorder` in 1 of 10: in all four raw captures
the model's `reordered` was its rewrite again - with added content ("The
kitchen is filled with the scent of fresh bread") or wrapped whole in
quotation marks - and the natural reorder of the baker prompt merges two
clauses, which `is_reorder` rightly refuses. `text_to_render` held the whole
prompt in 3 of 6. On Krea's own example "Moody night photography, a small
store with an orange glowing sign that says 'Black Forest Labs'" the critic
said the subject was not first, and in one answer also told it to put its
style first. The runs' JSON is gone; their printed summaries, and each raw
capture's `reordered` and `rewrite`, are in `final-ux/` of the recovered
evidence (below).

**After the final fixes.** The same kind of prompts plus Krea's tea-party
example and the "no people" / Open Late prompt, 20 critiques in two runs
(`real_critic.py`; `replay.py` put both runs' raw answers through the final
`validate`. The scripts, what the runs printed and the replay's per-answer
lines and totals are recovered; the raw answers are not). Check now says
"Not checked" under the order and "order not checked" in the summary for
every one of these prose prompts. The critic raised the subject order in all 8
answers on the two out-of-order prompts, and restated a rule finding in 18 of
20 answers (the word count in 17) - 25 issues, all left out and said so. With
the camera sentence worded "optional", Krea's own example drew no order issue
and no "camera settings missing" in either
answer; worded "adds camera settings" in the first run, four answers called
them missing. `reordered` survived in none of the 20: the model returned its
rewrite again in 17 (left out, and said so) and the prompt unchanged in 3 -
so on prose the order is compared through Your version. The instruction not
to pad did not hold: 14 of 20 rewrites added six or more content words
(lenses, light, fog, "capturing the eerie beauty of the night"), each now
flagged `heuristic.rewrite_adds`; one quoted the prompt's own "Tea Party"
(flagged `heuristic.rewrite_quote`) and one ended in Chinese, "透过树叶",
which the T5 check reported as unknown characters. Two `text_to_render`
parts were nulled (a colour grade; a whole clause). In the second live run
the first version of the order-conflict check dropped two agreeing order
issues ("Place the subject first, then the style description" was read as
wanting style); the check now reads the fix, and the replay drops none.

**Evidence.** The raw files were written to the scratchpad of the Claude
Code session that built the analyser (`verify/`, `final-ux/`, `final-fix/`)
and deleted with it on 2026-09-27, a minute and a half after e7d2386 was
committed, when that session emptied its temp folder at the user's request.
Lost with it: the per-call request/response JSON, the model's raw answers,
the VRAM poll CSVs, the full server logs, the throwaway galleries and every
image. What the session's transcripts and its surviving `tasks/` outputs
still held is in `%TEMP%/local-diffusion-analyser-validation/`: the driver
scripts as first written (`drive.py`, `tok_check.py`, `step5.py`,
`wsl_critic.py`, `real_critic.py`, `replay.py` and the rest), the four FLUX
prompts above, and what each run printed - per-call timings and VRAM peaks,
the server's `critique in` lines, the validated critiques and the replay's
totals. Each file has a `.source.txt` beside it saying where it came from,
and `INDEX.md` lists what was recovered and what was not. The rows dated
2026-09-28 have a folder of their own there, `review-2026-09-28/`, copied
whole from the review session's scratchpad rather than recovered, so without
sidecars (its `README.md` says where it came from): the scripts behind those
rows, and the lanes' reports (`phase1_result.json`), which are where the
figures those scripts printed were recorded - `real_critique.py` saved no
output of its own. Re-running those scripts is a new measurement, not a
re-check of the old one. The rows dated 2026-09-29 have
`review-2026-09-28/final-2026-09-29/`, written by the run itself: the
scripts, each run's requests and responses, the app's two logs, the VRAM
polls and what was left after each stop, with a `README.md` saying what
each file is.

Three things can still be checked against what survives. The two first-use
rows' install trees remain, measured again on 2026-09-28, so their layout,
`ldd` under WSL and a `sha256sum` of the GGUF against `critic.MODEL_SHA256`
can be repeated; their download times cannot. The Analyse row's counts can
be repeated by sending the four prompts (`verify/prompts_flux.json`) through
a copy of the recovered `drive.py analyze` - it writes beside itself - and
running `tok_check.py` on the per-prompt files it saves: the run's
`-summary.json` has no request, and stopped the first check with a
KeyError. The images' hashes can be re-rendered on the fp8 path at
1024x1024, 28 steps and CFG 3.5: the gate test's image and its repeat,
`7d161ca06d8753e7`, is "A red fox sitting in a snowy birch forest at dawn."
at seed 4242, and `INDEX.md` lists the comparison grid's four. The
2026-09-27 critique rows and the paragraphs above rest on the recovered
printed output alone; checking them again means new runs. The production
gallery was not used. Not measured: a critique beside a load in progress
(refused by design), int8 or sequential offload beside the critic, and
whether a rewrite renders better than its original over more than two
seeds.
