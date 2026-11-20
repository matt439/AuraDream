# Verified on the card

What has been measured on the RTX 5080 (16 GB), and when; the RTX 3070
(8 GB) on native Ubuntu has its own section at the end, and every other
section is the 5080's. The reasoning
around each number is in `design-notes.md`; this is the lookup. Times are per
image with model offload, and "peak" is what the process held rather than
what `nvidia-smi` showed, unless said otherwise - the desktop holds about
5,200-6,100 MiB of the card while Unreal Editor and the browsers are open,
and a run that lands within a gigabyte of the top spills into system RAM and
slows two to three times (schnell's 2 MP read 35-45 s that way and 25 s at a
1,270 MiB baseline). Since 2026-10-24 each image records its call's peak
(the lightbox's Created row): the allocator's, as this file's peaks are,
so a measurement here cites it only beside its own whole-card and
shared-memory readings.

The project was called Local Diffusion until 2026-10-27. Records made before
then keep the names in use at the time - `LOCAL_DIFFUSION_*` variables (now
`AURADREAM_*`), `local-diffusion` PNG chunks and paths, and the page's own
wording - because they quote what was run and what it printed.

## Checkpoints

Anima 2.9B Preview v1 was verified on 2026-10-24 from the author's pinned
[`Gazingstars123/Anima-2.9B`](https://huggingface.co/Gazingstars123/Anima-2.9B)
release, using Windows FP8, model offload, batch 1, 30 steps and CFG 4.
Its expanded 40-layer transformer was measured separately from Turbo and
Aesthetic; the qualification and its memory limits are recorded below.

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
| Anima 2.9B Preview v1, CFG 4 | fp8, Windows | 1024x1024 | 30 | 32.037 s after cancellation; 31.203 s repeat | 2.958 GiB allocated, 3.123 GiB reserved | 2026-10-24 |
| Anima 2.9B Preview v1, CFG 4 | fp8, Windows | 1152x1536 | 30 | 58.779 s, one sample | 3.051 GiB allocated, 3.482 GiB reserved | 2026-10-24 |
| NoobAI-XL V-pred 1.0, CFG 5 | fp8, Windows | 1024x1024 | 28 | 8.078 s after cancellation; 8.490 s repeat | 3.686 GiB allocated, 4.371 GiB reserved (maximum of the pair) | 2026-10-07 |
| FLUX.2 klein base 4B, CFG 4 | fp8, Windows | 1024x1024 | 50 | 57.680 s after cancellation; 57.730 s repeat | 8.149 GiB allocated, 8.381 GiB reserved (maximum of the pair); 9,249 MiB whole-card, the desktop's 361 MiB included | 2026-10-25 |
| ERNIE-Image Turbo | fp8, Windows | 1 MP | 8 | 12.8-13.3 s once warm (1.20 s a step), 26.7 s for the first of a process (18.3 s load) | 8.54 GiB allocated, 8.93 GiB reserved; 9,815 MiB whole-card, the desktop's 361 MiB included | 2026-10-25 |
| ERNIE-Image Turbo | fp8, Windows | 1440x1440, 2 MP | 8 | 25.6 s (2.63 s a step) | 9.06 GiB allocated, 9.35 GiB reserved; 10,247 MiB whole-card | 2026-10-25 |
| ERNIE-Image Turbo | fp8, Windows | 2048x2048, 4 MP | 8 | 63.3 s (7.04 s a step) | 10.43 GiB allocated, 11.68 GiB reserved; 12,631 MiB whole-card | 2026-10-25 |

The October 5 rows were each the source of a refine ladder ("Past 4 MP",
below), with another project's processes keeping the CPU near 75% busy:
their times are upper bounds.

The 2026-10-25 rows: one process and one load each, seed 12345, model
offload, `HF_HUB_OFFLINE=1`, a scratch gallery, with nvidia-smi, the
Windows shared-memory counters and free RAM watched (shared memory never
rose past the constant 82-86 MiB a first render adds, so nothing spilled).
klein base 4B (`card6.py`) loaded in 21.6 s at 22.9 GiB working set; the
call before the first row was cancelled after step 2 in 6.17 s; both renders
hash `8a877bdbb10c2d3b`, and the repeat peaked lower (4.58 / 5.21 GiB,
5,999 MiB), so the pair's maximum is the first render's. `card4.py`'s first
render of its own fresh fp8 load, with no cancelled call before it, peaked
at 4.58 / 5.21 GiB ("Quantization and offload", below), so the 8.15 GiB
follows the cancelled call and is not the render's own cost; why is not
measured. No live previews:
FLUX.2's VAE has no map. By eye, a coherent botanist in a navy coat with a
satchel in a dusk greenhouse. ERNIE (`ernie_card.py`): the process held
35.4-36.0 GiB of RAM at the load (14.0 GiB free at the lowest) and 14 GiB
after it; 1 MP at 848x1264, 1376x768 and 1200x896 took 12.9-13.3 s at the
same peak. A 4 MP batch of two was refused by the harness's own linear
projection (17,354 MiB against a 15,200 limit) and not run; NVFP4 ran that
pair at 12.0 GiB reserved, so the projection is probably pessimistic, but
whether it fits is unmeasured. By eye the 1 MP renders follow their prompts;
the 4 MP street is coherent but open where the prompt asked for a covered
one, and the small text on a web-page render is garbled. Evidence:
`card6-catalogue/` and `ernie/run-card/` in
`%TEMP%/local-diffusion-card-20261025/` ("the cards directory" below; a
copy of the 2026-10-25 card session's scratch directory, with each
session's report in `results/`), each with `results.jsonl` and named PNGs.

Anima 2.9B Preview v1's source file matched its pinned SHA-256,
`0b3020d1b906155f7eb30667622723e87160632c8c7a5f1c93bdce685f2a346d`.
Download and hashing took 80.387 seconds. With the official shared components
already cached, CPU conversion and the first load took 18.494 seconds;
the successful qualification's subsequent cached load took 9.013 seconds.
The actual `AnimaPreviewPipeline` had 40 transformer layers and
`FlowMatchEulerDiscreteScheduler` shift 3. Loading went through
`main.ensure_loaded`; synchronous renders then used the normal backend and
storage paths, without an HTTP server or the production gallery.

Seed 12345 rendered an adult auburn-haired botanist in a navy coat with a
brown satchel, holding blue flowers inside a glass greenhouse with warm
lanterns. The prompt mixed tags with two descriptive sentences; the negative
was `blurry, jpeg artifacts`. Visual inspection of both measured sizes and
their last preview found coherent illustrations with the requested scene
elements. These are functional smoke samples, not a quality comparison with
other checkpoints or a reproduction of the author's ComfyUI sampling recipe.

Cancellation stopped the first call at completed step 2 (11.380 seconds for
the whole call, including its initial encoding). The next two 1024-square
calls completed all 30 steps and had identical pixels, SHA-256
`f3cc7421b157ed817a5898999ce1c18b8530e205710296a7b34987016eb0cbb2`.
The 1152x1536 call completed all 30 steps, SHA-256
`67d309cfe570a28f60e52f7f757400c4f9a07f26e1cb67e4e0f6c9521a11584c`.
Every complete render produced 22 live previews, at steps 8 through 29.
SQLite rows and PNG chunks retained the public id, FP8, seed, size, prompt
pair, 30 steps, CFG/True CFG 4 and null Mu. Times include previews and exclude
load and PNG saving; allocator peaks exclude desktop/driver memory. Other
project CPU work remained active, so the times are individual smoke samples.

The first two attempts were deliberately stopped before denoising when the
test process tree exceeded the initial 14 GiB private-memory ceiling. After
Matt approved an 18 GiB private / 16 GiB working-set ceiling, the successful
run peaked at 14.431 GiB private memory and 11.400 GiB working set. Free
system RAM stayed above 19.632 GiB; total system commit rose by at most
15.420 GiB over the run's baseline. The independent watchdog also required
at least 12 GiB free RAM and commit headroom, at most a 28 GiB total commit
increase, and at least 1 GiB free VRAM. Whole-card VRAM peaked at 5,604 MiB
against a 1,602 MiB desktop baseline. All approved limits held, the harness
reported `passed: true` and `unloaded: true`, and the watchdog exited 0.

Evidence is in `%TEMP%/local-diffusion-anima-preview-20261024/`:
`qualify.py`, `supervise.ps1`, `gpu-results-large.json`, `gpu.log`, `gpu.err`,
`baseline.json`, `memory.jsonl`, `watchdog.json`, `converted-inventory.json`,
the separate `gpu.db`, previews and `outputs/` PNGs. The two stopped attempts
are retained in `first-load-ram-stop/` and `cached-load-ram-stop/`.
Other sizes and batch scaling remain unmeasured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below). Mu, references, user LoRAs, inpaint and Nunchaku remain
unavailable. Refine was unavailable in this October integration; its later
bounded same-size release and acceptance are recorded under Refine pass below.
The measurement/cache-fact updates passed 69 focused Python tests
(`test_model_info`, `test_model_benchmarks`, `test_model_downloads` and
`test_docs_maps`) and all 18 expectations in `smoke-anima-preview.json`
against a fresh fake gallery. The full Python suite was not rerun.

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
tokens. Larger direct/Refine sizes, batch scaling and user LoRAs remain
unmeasured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below). There is no SDXL live latent-preview
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
resolutions and batch scaling remain unmeasured for Aesthetic (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below). Mu, references, user LoRAs and Nunchaku
are unavailable. Refine was unavailable in this October integration; its
later bounded same-size release and acceptance are recorded below.

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
user LoRAs and larger direct or Refine sizes remain unmeasured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below).

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
LoRAs, batch scaling and larger resolutions remain unmeasured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below).

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

Rechecked 2026-10-20 through the CLI with no negative prompt: the same
example, seed and CFG renders as noise-covered, over-saturated glitch art
(hash `757cb1d1c1359813`, the same at 019f3dc and on main, and as noisy in
bf16), while CFG 1, or CFG 4 with a real negative, is clean.

Fixed 2026-10-21 (fp8, model offload, 1024x1024, 28 steps): a blank
negative now encodes as the authors' training dropout, Gemma `""` beside
the prompt's own Jina caption (design notes, "The unconditional is the
trained caption dropout"). Through the CLI at the profile's defaults, CFG 4
and no negative, the authors' example renders clean at seeds 12345 and
12346 (`407a2d9cb535d553`, `32d7d537d1a3bc8b`, a batch of two), and in a
direct pipeline sheet clean at seeds 12345 and 2024 with CFG 4 and 5. The
same sheet's old encoding reproduced the noisy `48ab3d50e2e8b75c`, and a
real negative's image kept its hash (`b7d9ceac6fcc2b1b`); a `,` negative is
still noisy. 37-40 s an image, 4.81 GiB allocated, as before. The authors'
demo CFG truncation (0.25, with and without the norm cap) did not fix it.
Evidence: `%TEMP%/local-diffusion-gen13/` (the CLI renders, `gen13.py`,
its logs, and the `sheet-cfg-trunc` and `sheet-train-uncond` sheets).

A NewBie batch was not bit-equal to its solo renders on the card. Seed
12346 as the second image of a batch of two hashed `32d7d537d1a3bc8b`,
alone `8739013b5f8256bf`: the same picture with small pixel drift. The
code before the fix did the same (`0d280e2b231106e8` against
`315f20eb3ecbe8f2`, `%TEMP%/local-diffusion-gen13-old/`). One forward
pass hooked at batch 1 and 2 found the first differing outputs in
`time_caption_embed`'s linear (1 row against 2) and the context refiner's
`to_k` (486 rows against 972) on identical inputs, while the two images
inside the batched call were identical: cuBLAS picks its kernel by the row
count. Fixed 2026-10-21 by one transformer and VAE call per image
(design notes, "One image per transformer call"): a batch of two at seeds
12345 and 12346 hashed `8761bdc10ea5bc22` and `8739013b5f8256bf`, as did
the two solo renders, the second unchanged from before. The batch took
1m15s and the two solos 1m16s, loads of 15 s included, as before.
Evidence: `%TEMP%/local-diffusion-gen14/` and `gen14_bisect.py` in
`%TEMP%/local-diffusion-gen13/`.

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
resolutions and GPU batch scaling remain unmeasured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below). References, user LoRAs and Nunchaku are disabled.

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
`smoke-netayume.json` passes all 12 browser expectations. User LoRAs, batch
scaling and larger image sizes have not been measured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below). The guide's 1.05-1.57 MP span comes from
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
`smoke-kohaku.json` passes all 16 browser expectations. Kohaku's user LoRAs,
batch scaling and higher resolutions have not been measured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below).

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
Anima's batch scaling and higher resolutions have not been measured (INT8, full BF16 and fp8 with sequential offload at 1024x1024 were measured 2026-10-25: "Quantization and offload", below). Mu, reference images and user
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

### Size and batch gaps (2026-10-21)

Text-to-image at the profile's own steps and CFG, fp8, model offload,
Windows, seed 12345, one process per checkpoint with sizes run smallest
first and the 15.5 GB watchdog armed (it never tripped). The desktop held
1.0-1.3 GB; "card" is the whole card's peak, desktop included.

| Checkpoint | Size | Steps | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- | --- |
| Animagine XL 4.0, Batch 12 | 1024x1024 | 28 | 70.7 s for 12 (2.19 s) | 6.58 / 9.53 GiB | 11,267 MiB |
| Illustrious XL v2.0 | 1536x1536, 2.4 MP | 28 | 21.3 s (0.54 s) | 3.51 / 4.37 GiB | 5,902 MiB |
| Illustrious XL v2.0 | 2048x2048, 4.2 MP | 28 | 34.4 s (1.11 s) | 4.06 / 5.10 GiB | 6,799 MiB |
| Kohaku XL Zeta | 1536x1536 | 28 | 19.2 s (0.53 s) | 3.51 / 4.16 GiB | 5,809 MiB |
| Kohaku XL Zeta | 2048x2048 | 28 | 35.5 s (1.11 s) | 4.06 / 5.10 GiB | 6,724 MiB |
| Anima Turbo v1.1 | 832x1216 / 1216x832, 1.0 MP | 8 | 9.0 (first) / 4.5 s (0.31 s) | 2.17 / 2.34 GiB | 3,817-3,829 MiB |
| Anima Turbo v1.1 | 1248x1248, 1.6 MP | 8 | 7.2 s (0.56 s) | 2.24 / 2.59 GiB | 4,150 MiB |
| Anima Turbo v1.1 | 1152x1728 / 1440x1440, 2.0-2.1 MP | 8 | 14.5 / 15.7 s (1.01 / 1.48 s) | 2.29-2.30 / 2.66-2.77 GiB | 4,210-4,325 MiB |
| Anima Aesthetic v1.1 | 832x1216 / 1216x832 | 30 | 24.4 / 21.2 s (0.62 s) | 2.17 / 2.34 GiB | 3,946-3,995 MiB |
| Anima Aesthetic v1.1 | 1248x1248 | 30 | 36.4 s (1.10 s) | 2.24 / 2.60 GiB | 4,115 MiB |
| Anima Aesthetic v1.1 | 1152x1728 / 1440x1440 | 30 | 48.8 / 55.2 s (1.52 / 1.73 s) | 2.29-2.30 / 2.66-2.77 GiB | 4,254-4,325 MiB |
| NetaYume Lumina v4 | 1536x1536 | 40 | 109.1 s (2.51 s) | 3.55 / 4.54 GiB | 6,307 MiB |
| NetaYume Lumina v4 | 2048x2048 | 40 | 241.9 s (5.93 s) | 4.36 / 5.74 GiB | 7,481 MiB |
| NewBie Exp0.1, the authors' XML | 1536x1536 | 28 | 107.3 s (3.47 s) | 4.81 / 5.40 GiB | 7,109 MiB |
| NewBie Exp0.1, the authors' XML | 2048x2048 | 28 | 235.0 s (8.18 s) | 5.24 / 6.63 GiB | 8,326 MiB |

Memory is nowhere near the card's top: Batch 12 on Animagine leaves about
5 GB, and every larger single image here stays under 8.4 GB. Time is what
grows: the DiTs' steps go up faster than their pixels (Anima Turbo 0.31 s
a step at 1 MP, 1.48 s at 2.1 MP; NetaYume and NewBie about 2.4x from 2.4
to 4.2 MP), so NetaYume and NewBie take about 4 minutes an image at 4 MP.

By eye, one prompt and seed each: both Anima variants stay coherent from
1 to 2.1 MP, portrait, landscape and square, with no doubled figures.
NetaYume holds one figure at 4.2 MP (more closely framed than at 2.4).
NewBie is right at 2.4 MP but at 4.2 MP the authors' two characters
repeat into a sheet of about eight figures, so above about 2.4 MP it
breaks composition. Evidence: `animagine-xl-4-0/`, `illustrious-xl-v2-0/`,
`kohaku-xl-zeta/`, `anima/`, `anima@aesthetic-v1-1/`,
`netayume-lumina-image-2-0/`, `newbie-image-exp0-1/` (each with
`results.jsonl` and `outputs/`), `anima-sizes.jpg` and `dit-sizes.jpg` in
`%TEMP%/local-diffusion-inpaint-card/`.

Under model offload the per-image weight transfer costs about 5 s and
dominates a 4-step image; on FLUX.1 the text encoders under bitsandbytes NF4
add to that, which is why schnell is not the speed win Z-Image Turbo is.

### Quantization and offload (2026-10-25)

INT8, full BF16 (quantization none) and fp8 with sequential offload against
fp8 with model offload, per checkpoint: Windows, 1024x1024, the profile's
own steps and CFG, seed 12345, batch 1, one `card4.py` process and one load
per checkpoint and mode, a scratch gallery, `HF_HUB_OFFLINE=1`, the card
watched at 250 ms with a 15.5 GB stop, the Windows shared-memory counters
and a 4 GiB free-RAM floor. Time is the first render of the process, then
an identical repeat where the first took under 90 s (every repeat had the
first's pixels), and the per-step figure; allocated / reserved is torch's
peak (equal to the backend's `last_peak` in every row); "card" is the
whole card, the desktop's 361 MiB (later 399) included. Teams, the
ChatGPT/Codex app and others were running: individual samples, one seed.

| Checkpoint (steps, CFG) | Mode | Time: first; repeat (per step) | Allocated / reserved | Card | Load |
| --- | --- | --- | --- | --- | --- |
| WAI v17 (28, 6) | fp8 / int8 / bf16 | 9.76; 7.50 / 8.53; 7.22 / 8.85; 6.51 s (0.22 / 0.21 / 0.17) | 3.12 / 4.37, the same, 5.23 / 5.64 GiB | 5,143 / 5,143 / 6,445 MiB | 9.8 / 5.5 / 3.6 s |
| WAI v17 | fp8, sequential | 29.07; 23.84 s (0.81) | 2.41 / 2.54 GiB | 2,538 MiB | 11.1 s |
| Animagine XL 4.0 (28, 5) | fp8 / int8 / bf16 | 8.18; 7.60 / 7.98; 7.28 / 7.68; 6.81 s (0.22 / 0.20 / 0.17) | 3.12 / 3.47, 3.12 / 3.46, 5.23 / 5.64 GiB | 4,219 / 4,215 / 6,445 MiB | 11.5 / 6.4 / 5.5 s |
| Animagine XL 4.0 | fp8, sequential | 28.42; 23.93 s (0.81) | 0.63 / 0.79 GiB | 1,254 MiB | 12.4 s |
| Illustrious XL v2.0 (28, 5) | fp8 / int8 / bf16 | 9.89; 7.61 / 8.66; 7.21 / 9.21; 6.68 s (0.22 / 0.21 / 0.17) | 3.12 / 4.37, the same, 5.23 / 5.64 GiB | 5,180 / 5,180 / 6,482 MiB | 10.4 / 5.7 / 3.6 s |
| Illustrious XL v2.0 | fp8, sequential | 25.23; 23.61 s (0.81) | 2.41 / 2.54 GiB | 2,024 MiB | 11.0 s |
| Kohaku XL Zeta (28, 5) | fp8 / int8 / bf16 | 8.58; 7.85 / 8.07; 7.45 / 7.98; 6.93 s (0.22 / 0.20 / 0.17) | 3.12 / 3.47, 3.12 / 3.46, 5.23 / 5.64 GiB | 4,256 / 4,252 / 6,482 MiB | 12.1 / 6.6 / 5.7 s |
| Kohaku XL Zeta | fp8, sequential | 26.21; 25.33 s (0.81) | 0.36 / 0.40 GiB | 1,110 MiB | 12.5 s |
| RouWei 0.8 (28, 6) | fp8 / int8 / bf16 | 9.91; 7.58 / 8.76; 7.20 / 9.20; 6.67 s (0.22 / 0.21 / 0.17) | 3.12 / 4.37, the same, 5.23 / 5.64 GiB | 5,180 / 5,180 / 6,482 MiB | 10.4 / 5.7 / 3.6 s |
| RouWei 0.8 | fp8, sequential | 25.44; 23.88 s (0.82) | 2.41 / 2.54 GiB | 2,536 MiB | 10.9 s |
| NoobAI-XL V-pred 1.0 (28, 5) | fp8 / int8 / bf16 | 8.22; 7.60 / 7.84; 7.19 / 7.66; 6.61 s (0.22 / 0.20 / 0.17) | 3.12 / 4.37, the same, 5.23 / 5.64 GiB | 5,180 / 5,180 / 6,482 MiB | 12.2 / 6.7 / 5.6 s |
| NoobAI-XL V-pred 1.0 | fp8, sequential | 24.49; 23.79 s (0.81) | 2.41 / 2.54 GiB | 2,022 MiB | 12.6 s |
| Z-Image Turbo (8, 0) | fp8 / int8 / bf16 | 12.77; 10.17 / 11.08; 9.31 / 14.30; 11.61 s (0.89 / 0.81 / 0.76) | 6.43 / 7.17, 6.30 / 7.15, 12.07 / 12.89 GiB | 8,005 / 8,022 / 13,900 MiB | 41.5 / 17.1 / 7.9 s |
| Z-Image Turbo | fp8, sequential | 19.64; 10.80 s (1.25) | 2.41 / 2.98 GiB | 2,992 MiB | 43.2 s |
| Z-Anime Base (40, 4) | fp8 / int8 / bf16 | 75.95; 72.56 / 72.82; 69.95 / 75.04; 69.01 s (1.73 / 1.68 / 1.60) | 6.77 / 8.44, 6.77 / 8.04, 12.55 / 13.85 GiB | 9,350 / 8,932 / 14,888 MiB | 29.7 / 13.2 / 3.6 s |
| Z-Anime Base | fp8, sequential | 93.84 s, no repeat (2.25) | 2.42 / 2.99 GiB | 2,282 MiB | 30.2 s |
| Anima Turbo v1.1 (8, 1) | fp8 / int8 / bf16 | 9.94; 4.00 / 5.34; 3.92 / 6.28; 4.08 s (0.32 / 0.31 / 0.28) | 2.17 / 2.34, 2.08 / 2.25, 3.92 / 4.10 GiB | 3,096 / 3,012 / 4,900 MiB | 10.3 / 5.2 / 3.7 s |
| Anima Turbo v1.1 | fp8, sequential | 12.45; 6.00 s (0.56) | 0.34 / 0.61 GiB | 1,072 MiB | 9.7 s |
| Anima Aesthetic v1.1 (30, 4) | fp8 / int8 / bf16 | 22.77; 20.66 / 21.29; 19.79 / 21.17; 18.90 s (0.64 / 0.61 / 0.57) | 2.17 / 2.34, 2.08 / 2.26, 3.92 / 4.10 GiB | 3,098 / 3,014 / 4,902 MiB | 8.8 / 5.2 / 3.7 s |
| Anima Aesthetic v1.1 | fp8, sequential | 36.91; 33.84 s (1.07) | 0.34 / 0.61 GiB | 1,106 MiB | 9.4 s |
| Anima 2.9B Preview v1 (30, 4) | fp8 / int8 / bf16 | 31.60; 29.15 / 29.70; 27.94 / 29.44; 26.80 s (0.91 / 0.87 / 0.81) | 2.96 / 3.12, 2.87 / 3.04, 5.49 / 5.69 GiB | 3,902 / 3,820 / 6,526 MiB | 10.4 / 6.2 / 3.7 s |
| Anima 2.9B Preview v1 | fp8, sequential | 48.18; 47.46 s (1.53) | 0.34 / 0.61 GiB | 1,106 MiB | 11.0 s |
| NetaYume Lumina v4 (40, 5) | fp8 / int8 / bf16 | 39.22; 34.96 / 35.01; 32.82 / 35.16; 31.79 s (0.83 / 0.78 / 0.73) | 3.30 / 4.38, 3.14 / 4.38, 5.37 / 5.97 GiB | 5,186 / 5,188 / 6,818 MiB | 13.2 / 8.1 / 4.5 s |
| NetaYume Lumina v4 | fp8, sequential | 49.02; 47.03 s (1.15) | 2.41 / 3.11 GiB | 2,988 MiB | 14.1 s |
| NewBie Exp0.1 (28, 4) | fp8 / int8 / bf16 | 40.14; 34.43 / 35.08; 32.34 / 36.83; 31.63 s (1.13 / 1.06 / 0.98) | 4.63 / 4.69, 4.44 / 4.46, 7.28 / 7.68 GiB | 5,510 / 5,272 / 8,568 MiB | 21.6 / 12.2 / 5.7 s |
| NewBie Exp0.1 | fp8, sequential | 46.74; 44.15 s (1.55) | 2.41 / 3.26 GiB | 4,042 MiB | 22.1 s |
| FLUX.2 klein 4B (4, 1) | fp8 / int8 / bf16 | 8.15; 4.70 / 6.92; 4.42 / 10.75; 5.83 s (0.55 / 0.51 / 0.46) | 4.58 / 5.19, 4.42 / 5.27, 7.83 / 8.46 GiB | 6,020 / 6,098 / 9,370 MiB | 23.5 / 11.3 / 3.5 s |
| FLUX.2 klein 4B | fp8, sequential | 5.85; 4.28 s (0.86) | 2.42 / 3.00 GiB | 3,758 MiB | 25.5 s |
| FLUX.2 klein base 4B (50, 4) | fp8 / int8 / bf16 | 61.35; 57.95 / 55.54; 52.80 / 55.75; 50.78 s (1.11 / 1.01 / 0.93) | 4.58 / 5.21, 4.43 / 5.27, 7.84 / 8.48 GiB | 6,036 / 6,098 / 9,392 MiB | 22.9 / 11.5 / 3.5 s |
| FLUX.2 klein base 4B | fp8, sequential | 86.40; 86.60 s (1.71) | 2.43 / 3.00 GiB | 2,478 MiB | 23.8 s |
| ERNIE-Image Turbo (8, 1) | fp8 / int8 | 17.23; 13.14 / 15.22; 11.97 s (1.18 / 1.08) | 8.53 / 8.93, 8.25 / 8.64 GiB | 9,850 / 9,554 MiB | 34.7 / 16.3 s |
| ERNIE-Image Turbo | fp8, sequential / bf16, sequential | 15.74; 13.99 (1.66) / 21.17; 15.72 s (1.87) | 2.42 / 3.03 GiB both | 2,268 / 2,540 MiB | 35.4 / 4.2 s |
| FLUX.1 Krea dev (28, 3.5) | bf16, sequential | 109.7 s, no repeat (2.81) | 2.41 / 2.56 GiB | 2,556 MiB | 3.8 s |
| FLUX.1 dev (28, 3.5) | bf16, sequential | 107.2 s, no repeat (2.82) | 2.41 / 2.56 GiB | 1,988 MiB | 3.8 s |
| FLUX.1 schnell (4, 0) | bf16, sequential | 39.61; 12.64 s (2.86) | 2.41 / 2.56 GiB | 2,168 MiB | 3.8 s |
| Krea 2 Turbo (8, 0) | bf16, sequential | 51.96; 24.41 s (2.84) | 0.76 / 1.47 GiB | 1,894 MiB | 5.4 s |
| Krea 2 Raw (52, 3.5) | bf16, sequential | 329.8 s, no repeat (5.77) | 0.79 / 1.47 GiB | 1,896 MiB | 5.2 s |
| FLUX.2 klein 9B (4, 1) | bf16, sequential | 36.82; 8.58 s (1.82) | 2.43 / 3.19 GiB | 3,106 MiB | 3.8 s |
| FLUX.2 klein base 9B (50, 4) | bf16, sequential | 212.6 s, no repeat (3.64) | 2.44 / 3.19 GiB | 3,102 MiB | 3.8 s |

- INT8 costs what fp8 costs on the card (within 0.4 GiB, the same 3.12 GiB
  on SDXL), runs 3-10% faster a step on all sixteen (2.9% on Z-Anime to
  9.9% on klein base 4B; a repeat render 2.0-8.9% faster), and loads in
  about half the time. Full BF16 fits under model offload on every
  checkpoint it was tried on, the two nearest the top being Z-Image Turbo
  (13,900 MiB) and Z-Anime (14,888 MiB). fp8 with sequential offload is
  pixel-identical to fp8 with model offload on all sixteen, at 2.43 GiB
  allocated or less (3.26 GiB reserved at most, NewBie).
- Nothing spilled: the process's shared GPU memory rose a constant 82 MiB
  on every model-offload first render (8 MiB under sequential), 0 on every
  repeat, including 4 GB SDXL runs nowhere near the card's top.
- These are the first renders on this machine of Krea 2 Raw, klein 9B and
  klein base 9B, the first Windows render of FLUX.1 dev, and the first
  timed Windows one of Krea 2 Turbo (schnell rendered at fp8 on 2026-10-05,
  Krea 2 Turbo at fp8 in 2026-09, the rows above); all of these ran only in
  BF16 with sequential offload, since the fp8 and int8 loads tried stopped
  at the RAM guard and the rest were not tried ("RAM at the load", below). The BF16 loads map the files and page them in during the first
  render (about 30 s for its first step on FLUX.1 and Krea 2), the process
  peaking at 33-35 GiB working set then, 14.7 GiB free at the lowest. By
  eye each of the FLUX.1 Krea dev, Krea 2 Turbo and Krea 2 Raw images is a
  clean photograph that matches the prompt; there is no fp8 image of them
  to compare.
- Skipped as past the card by `card4.py`'s estimate, not run: BF16 under
  model offload on FLUX.1 Krea dev, dev and schnell (about 23.5 GiB),
  Krea 2 Turbo (24.5), Krea 2 Raw (25.0), ERNIE (16.5), klein 9B (17.5)
  and base 9B (18.5). `card4.py` skipped every Qwen-Image mode: model
  offload as past the card (20.5-39 GiB), fp8 sequential on its RAM
  estimate (about 62 GiB against 50 free). The zimage-qwen session ran
  that load, and it stopped at the RAM guard ("RAM at the load", below).
- Not tried, inferred from same-sized loads that stopped at the RAM guard:
  int8 on FLUX.1 dev, schnell, Krea 2 Turbo and klein 9B; fp8 and int8 on
  Krea 2 Raw and klein base 9B; fp8 with sequential offload on all seven
  FLUX.1, Krea 2 and klein 9B checkpoints (the quantizing load comes before
  placement).

Against fp8 with model offload, by number (mean absolute difference in
8-bit levels / PSNR in dB / % of pixels off by more than 16):

| Checkpoint | INT8 against fp8 | BF16 against fp8 | INT8 against BF16 |
| --- | --- | --- | --- |
| WAI v17 | 16.7 / 18.1 / 34.0 | 19.7 / 16.9 / 37.9 | 13.4 / 19.3 / 25.1 |
| Animagine XL 4.0 | 9.2 / 23.0 / 21.9 | 8.4 / 23.5 / 19.8 | 6.1 / 26.1 / 13.0 |
| Z-Image Turbo | 8.9 / 21.4 / 12.2 | 8.2 / 21.8 / 10.6 | 3.8 / 26.8 / 5.4 |
| Anima Turbo | 27.5 / 14.1 / 43.2 | 30.1 / 13.5 / 45.6 | 16.8 / 16.2 / 24.0 |
| Anima Aesthetic | 31.6 / 12.8 / 42.6 | 32.9 / 12.5 / 43.5 | 6.1 / 22.9 / 8.7 |
| Anima 2.9B | 14.1 / 18.7 / 27.7 | 12.9 / 19.1 / 23.5 | 6.8 / 23.1 / 10.5 |
| Illustrious XL v2.0 | 6.0 / 27.9 / 6.9 | 8.4 / 25.6 / 15.4 | 5.7 / 27.6 / 7.4 |
| Kohaku XL Zeta | 9.8 / 21.8 / 18.3 | 8.9 / 22.5 / 16.6 | 5.6 / 25.8 / 9.0 |
| RouWei 0.8 | 7.5 / 22.7 / 11.3 | 6.0 / 24.4 / 8.6 | 5.5 / 25.4 / 7.4 |
| NoobAI-XL V-pred | 7.3 / 22.6 / 13.1 | 5.8 / 25.3 / 10.2 | 5.2 / 24.5 / 7.8 |
| Z-Anime | 19.0 / 17.6 / 54.2 | 17.7 / 17.9 / 47.0 | 15.4 / 20.4 / 42.6 |
| NetaYume v4 | 9.9 / 20.5 / 16.4 | 5.0 / 26.9 / 7.4 | 9.4 / 21.1 / 15.7 |
| NewBie Exp0.1 | 9.3 / 24.2 / 17.0 | 7.3 / 26.1 / 11.6 | 7.6 / 25.0 / 12.2 |
| klein 4B | 11.3 / 20.4 / 19.2 | 9.7 / 21.4 / 17.2 | 7.8 / 22.9 / 11.6 |
| klein base 4B | 12.9 / 19.4 / 28.2 | 16.4 / 17.8 / 37.4 | 10.4 / 20.2 / 17.8 |
| ERNIE Turbo (BF16 sequential) | 8.2 / 22.2 / 13.7 | 9.0 / 21.0 / 14.2 | 7.4 / 22.0 / 10.5 |

INT8 is nearer BF16 than fp8 is, by mean difference, on 14 of 16 (not
NetaYume or NewBie; on NoobAI, PSNR favours fp8, 25.3 dB against 24.5).
These sixteen leave out FLUX.1, Krea 2 and klein 9B, which have no INT8 or
fp8 image to compare. By
eye, on the sheets viewed (WAI, Animagine, Z-Image Turbo, the three Animas,
NoobAI, Z-Anime, klein 4B, ERNIE): the three modes give the same picture
with detail changes - straps, a fur hood, an ear under hair, coat
closures, an expression, a lamp post's shape - and larger ones on Anima
Turbo, Anima Aesthetic and Z-Anime (the figure's size or place, the coat's
colour); no mode looked degraded, and one seed cannot say one is better.
The Illustrious, Kohaku, RouWei, NetaYume, NewBie and klein base 4B sheets
were not viewed. Evidence: `card4-quant/` in the cards directory
(`summary.txt`, `results.jsonl`, `compare.jsonl`, `verdicts.txt`,
`compare/*.jpg`, per-run logs and PNGs).

Remaining after the fixed-loader trial on 2026-10-27: INT8 and fp8 under
model/sequential offload on FLUX.1 dev, schnell, Krea 2 Turbo, Krea 2 Raw,
klein 9B and base 9B; Krea dev's INT8 modes and fp8/model repeatability.
Krea dev's fp8/sequential load and its RAM were measured on 2026-11-11 under
a cropped inpaint (see "FLUX.1 Krea dev with sequential offload, 2026-11-11");
its text-to-image repeatability is not. Krea dev's fp8/model load and first render are now measured
in [the fixed-loader result](#ram-at-the-load-the-fixed-loader-2026-10-27),
not a repeat or a new cross-mode quality comparison. The earlier RAM guard
stops remain historical evidence. Any Qwen-Image mode on Windows, INT8 with
sequential offload on the other checkpoints, and other sizes, batches, LoRAs,
Refine and inpaint under these modes remain unmeasured.

**Retired from the backlog, 2026-11-12.** Matt rates other quantizations
and offloads (K4, K5) Never, so `CARD#4`, `CARD#6` (the Windows fp8 rows:
Krea 2 Turbo's time, Krea 2 Raw, klein 9B and base 9B) and `CARD#12`
(Qwen-Image on Windows: fp8 with sequential offload again, BF16 with
sequential offload untried) are no longer planned card work. The modes
listed here and in the RAM sections below stay unmeasured. A mode is
measured when it becomes one Matt uses
([2026-11-12 audit](maintainer/review-2026-11-12-todo.md#matts-answers-2026-11-12)).

### RAM at the load (2026-10-25)

Every Windows load tried that day that quantizes a FLUX.1-, Krea 2-,
klein 9B- or Qwen-Image-sized checkpoint stopped at the harnesses' 4 GiB free-RAM floor
before a weight reached the card (the card at its desktop level, shared
memory flat), starting from 49-50 GiB free with Teams, the ChatGPT/Codex
app, Edge WebView, another project's server and WSL (1.7 GB) resident. None
was retried, no app was closed and the floor was not lowered.

| Checkpoint | Load | Stopped at | Free RAM at the stop | The process |
| --- | --- | --- | --- | --- |
| FLUX.1 Krea dev | fp8, model offload: the real server twice (`--defaults`), `card4.py` once | transformer shard 2 of 3, about 51 s in | 3.83 / 3.8 / 3.81 GiB | 46.31 / 46.2 / 46.2 GiB working set (55.2 private) |
| FLUX.1 Krea dev | int8, model offload | during the load | 3.93 GiB | 46.05 GiB working set, 53.4 private |
| FLUX.1 dev | fp8, model offload | transformer shard 2 of 3 | 3.7 GiB | 46.4 GiB peak RSS |
| FLUX.1 schnell | fp8, model offload | the next component after the 3 transformer shards (164 of 219 weights), 44 s in | 3.8 GiB | not logged |
| FLUX.2 klein 9B | fp8, model offload | 89% through the 4th of 5 components, 51 s in | 4.0 GiB | 46.02 GiB working set |
| Krea 2 Turbo | fp8, model offload | during the load | 3.78 GiB | 46.2 GiB working set, 61.5 private |
| Qwen-Image (`Qwen/Qwen-Image`) | fp8, sequential offload | transformer shard 6 of 9, about 50 s in | 3.7 GiB (commit headroom 10.0) | 46.0 GiB working set, 56.1 private |

So on this 64 GB machine, as it was that day, a cold fp8 or int8 load of
these checkpoints needed more than about 46 GiB of headroom in the pre-fix
loader; int8's spike was fp8's. The later diagnosis found unquantized
tensors retaining the bf16 shard mappings alongside quantized weights,
not a whole bf16 load before quantization (design notes, "The quantizing
load's RAM peak"). The 2026-10-21 FLUX.1 Krea dev outpaint run's load
completed from 52 GB free
after a reboot, with no guard, and earlier fp8 loads completed too:
schnell's on 2026-10-05 and Krea 2 Turbo's in 2026-09 (the Checkpoints
rows above), so the 46 GiB figure describes the machine as it was that
day, not a limit of the checkpoints. The loader was fixed 2026-10-25 to
release those shard mappings and remeasured 2026-10-27 (below). The BF16
sequential loads above avoid that quantizing path. None of the guard stops
above measures the fixed loader.
Loads that completed: klein base 4B 22.9 GiB working set; ERNIE fp8
35.4-36.0 GiB RSS, 14 GiB free at the lowest; Z-Image Turbo fp8 18.4-18.6
GiB working set (23-25 GiB private), and under offload none 31.3 GiB private with 7.2 GiB free at the lowest (a
WSL download's page cache was also holding Windows RAM then: WSL's
`drop_caches` gave back about 15 GB). Evidence: `flux-krea/`, `flux1-dev/`,
`flux1-schnell/`, `card6-catalogue/`, `card11-critic/`, `card4-quant/` and
`zimage-qwen/` in the cards directory, each guard stop a `ram_guard` (or
`TRIP`) row in its `results.jsonl` or `harness.jsonl` (card11-critic's in
`runs_phase3_stdout.txt`).

### RAM at the load, the fixed loader (2026-10-27)

One cold load of FLUX.1 Krea dev at fp8 with model offload, through
`get_backend("diffusers").load()` (`loading.checkpoint_files_released`,
343f8b7), then one render and an unload, in one process against a scratch
gallery, `HF_HUB_OFFLINE=1`, sampled every 0.25 s. It started from 44.7 GiB
free (23.4 GiB committed of an 85.6 GiB limit; the card at 1,353 MiB, shared
GPU memory 0.07 GiB) with the usual apps resident and WSL stopped. Neither
guard (4 GiB free RAM, a 1 GiB rise in shared GPU memory) tripped.

| What | Result |
| --- | --- |
| Load | 69.7 s from cold, the weights from the page cache's state of the day |
| Peak during the load | 27.9 GiB working set (the OS's own peak 28.4) and 35.4 GiB private, at its end (72.7 s); 14.2 GiB free at the lowest |
| Resident after the load | 19.6 GiB working set, 26.8 GiB private, 23.3 GiB free |
| Render | 1024x1024, 28 steps, CFG 3.5, seed 12345, 45.5 s (1.33-1.45 s a step; the first carries the transformer onto the card) |
| During the render | 19.6 GiB working set at most; private rose to 39.9 GiB near its end; 22.5 GiB free at the lowest |
| Card | 12.00 GiB allocated, 12.63 reserved at the peak; 14,947 MiB whole-card by `nvidia-smi`; shared GPU memory 0.07-0.15 GiB throughout, no spill |
| Unload | 2.3 s; 17.0 GiB working set and 26.6 private left in the process (allocator and file caches; the process exited after) |
| Image | PNG sha256 `17c6871f98129a9e3bdb5ee04241d2c811e3c1f4b417d2592810565b46841124`, pixels sha256 `016cd669f418c1251a0c43e42362bf9c6c8d5a456fb50f9c34f38e4b25fcff9f`; a clean render of the prompt ("a red fox sitting in a snowy pine forest at dawn, soft golden light through the trees, photograph") |

So the fix holds on Windows: the load's peak fell from the 46+ GiB working
set (53-62 GiB private) at which every 2026-10-25 load stopped to 27.9 GiB
working set and 35.4 private, inside the design notes' expected 25-28 GB.
Other checkpoints and int8 are not measured by this; each keeps its own row.
Evidence: `samples.jsonl`, `events.jsonl` and the PNG in
`%TEMP%/auradream-gen22-20261027/`.

## Behaviour

| What | Result | Date |
| --- | --- | --- |
| Bit-determinism through torchao fp8 with offload moving components mid-forward | The same request twice on FLUX.1 Krea dev is the same hash; a negative prompt the pipeline dropped hashes the same as none | 2026-09-08 |
| True CFG on FLUX.1 | A negative prompt with no true CFG is byte-identical to none; true CFG 1.5 is a different, reproducible image | 2026-09-08 |
| LoRA under torchao fp8 + PEFT | The adapter changes the image and reproduces; `unload_lora_weights` restores the base hash; weight 0 is byte-identical to no adapter | 2026-09-08 |
| LoRA on an SDXL checkpoint (Animagine XL 4.0, fp8, model offload, 1024x1024, seed 12345; `nerijs/pixel-art-xl`, 170 MB, from a scratch `LOCAL_DIFFUSION_LORAS_DIR`) | The same three as on FLUX.1: weight 1.0 changes the image (a coarser pixel grid, a muted palette, another pose) and is the same hash twice; weight 0 is the no-adapter hash; the no-adapter request after it is the base hash again, so the unload restores it. Through an inpaint too: the adapter changes the masked part (a plaid blue scarf) and reproduces, weight 0 and the inpaint after it are the plain inpaint's hash, and every pixel outside the mask is the source's (largest difference 0). It costs 0.23 GiB allocated (3.35 against 3.12 GiB; 3.51 against 3.28 through the inpaint) and 0.26-0.27 s a step against 0.22-0.23; card 6,421-6,664 MiB over a 2.5 GB desktop. The process held 9.4 GB of RAM at its peak. Evidence: `animagine-lora.log` and `animagine-xl-4-0-light/lora.jpg` in `%TEMP%/local-diffusion-inpaint-card/` | 2026-10-24 |
| Animagine XL 4.0 basic generation | User-verified on the card after integration: the weights loaded and output looked good; exact quantization and offload settings were not recorded | 2026-09-10 |
| Animagine XL 4.0, Batch 8 at 1 MP | 9.6 GB of the 16 as read off the card, which is what raised the Batch cap to 12; Batch 12 read 11.3 GB on 2026-10-21 (Checkpoints, "Size and batch gaps") | 2026-09-12 |
| Reference images on klein | The referenced scene reappears in the asked-for pose, same hash twice; no 2-3x cost (see the table) | 2026-09-08 |
| Live preview | Matrices fitted on the card, R-squared 0.964 (FLUX VAE) and 0.950 (Qwen-Image VAE); frames looked at; Z-Image frames at steps 2/4/6 of 8 in the right colours | 2026-09-08 |
| SDXL's live-preview map (`AutoencoderKL/4`), fitted on the CPU, not the card (`tools/fit_latent_preview.py`, float32, no download; Windows 11's 26 stock wallpapers and lock-screen photos, two 512 px crops each, every fifth image held out) | The recipe first refitted FLUX.1 Krea [dev]'s VAE: 0.966 held out, against the card fit's 0.964 (the shipped FLUX matrix scores 0.898 on these images and is unchanged). SDXL from Animagine XL 4.0's VAE (`sdxl-vae-fp16-fix`, latents `z * 0.13025`): R-squared 0.904 fitted, 0.909 held out; the same map scores 0.905 on Kohaku XL Zeta's and NoobAI's VAEs. Frames on the card not yet looked at | 2026-11-12 |
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
| The failed-load card against the real Hub, with no token and nothing cached (`HF_HOME` an empty scratch directory, `HF_TOKEN` unset, Load pressed at 2560x1305; then the card's Retry load) | FLUX.1 [dev]: `GatedRepoError` on `model_index.json`, 6.8 s after the press; `detail_kind` gated, "FLUX.1 [dev] is gated on Hugging Face: accept its terms on the model page (a click-through), run `uv run hf auth login` once with a read token, then press Retry load.", the command set as code, "Open the model page" to https://huggingface.co/black-forest-labs/FLUX.1-dev in a new tab, a primary Retry load. Krea 2 Raw: the same in 6.8 s, worded "ask for access on the model page and wait for its authors to approve it". A made-up id: `RepositoryNotFoundError` in 3.7 s, `not_found`, "local-diffusion-check/no-such-checkpoint is not on Hugging Face, or is private: check the checkpoint id (LOCAL_DIFFUSION_MODEL_ID, or the start-up menu); for a private repo, …", headed "Couldn't load no-such-checkpoint", the link to the id's (missing) page. Each time the topbar read the checkpoint's name and "· load failed", its Load relabelled "Retry load" and keeping the press's focus, Unload hidden, nothing in the panel's line; the card's Retry load failed again in 1.4-1.5 s with one card, focus on Generate (the card went during the load, and the one that came back does not take it). With `HF_HUB_OFFLINE=1` on an uncached FLUX.1 Krea [dev]: huggingface_hub 1.29.0 raises `LocalEntryNotFoundError` from an `OfflineModeIsEnabled`, `main._hub_failure` takes the outer one, and the card said "FLUX.1 Krea [dev] is not all downloaded yet, and Hugging Face cannot be reached: connect to the network, then press Retry load." - the network was up; the offline sentence ("HF_HUB_OFFLINE is set: unset it for one load") never showed. Nothing was downloaded: the scratch `HF_HOME` held 5,778 bytes after all four, huggingface_hub's own `.agent_harnesses.json`; the adapter listing logged "adapter fit unchecked until the load: 401". The offline sentence was fixed the same day in 81781bf: `_hub_failure` prefers `OfflineModeIsEnabled` wherever it is on the chain, tested on the chain huggingface_hub's own `hf_hub_download` builds under its offline switch (`tests/test_api.py`); checked on the real server 2026-10-25 (the load failures row below) | 2026-10-05 |
| The critic's first-use line and clock over a real download (the FLUX.1 Krea [dev] profile, an empty `HF_HOME`, the image model never loaded) | The line named what was still to fetch and changed as each piece landed, and the clock ran to "· 199 s" for a 199.7 s first use whose route `seconds` said 9.4, the critique alone; the row is in the Prompt analyser section | 2026-10-05 |
| Load failures on the real server after 81781bf (FLUX.1 Krea [dev] with `HF_HUB_OFFLINE=1`, an empty scratch `HF_HOME` and no token; a made-up id against the real Hub; `--defaults`, a scratch gallery, the page at 1400x900) | Offline: `detail_kind` offline in 3.2 s, "FLUX.1 Krea [dev] is not all downloaded yet, and HF_HUB_OFFLINE is set: unset it for one load, then press Retry load."; the Models footer said "Couldn't load FLUX.1 Krea [dev]: ..." with that sentence, the topbar "FLUX.1 Krea [dev] · load failed" with Retry load, and the row "Accept licence · Load failed" (the scratch home has no token). `local-diffusion-check/no-such-checkpoint`: `not_found` in 3.7 s with the id's Hub link and "... is not on Hugging Face, or is private: check the checkpoint id (Models in the browser, or LOCAL_DIFFUSION_MODEL_ID); ...", the same in the footer. A retry, sent over the API rather than pressed, failed the same way in 0.0-0.5 s each time; no Traceback in either log; nothing downloaded (`HF_HOME` 0 bytes, and 5,778, huggingface_hub's own file). Each server was stopped by a console Ctrl-C: rc 130, gone 0.85-0.89 s after it. Evidence: `flux-krea/runs/fail-*` in the cards directory | 2026-10-25 |
| Ctrl-C while a job waits on a real first load, R50's fix (1744dc3; FLUX.1 Krea [dev], fp8, model offload, a fresh scratch server; Generate clicked with a real mouse event at 1400x900, nothing loaded) | The job queued at once with `waiting_on` load, the page reading "loading weights · 0:00". About 10 s in (pipeline components 4 of 7) a helper process attached to the server's console and sent `CTRL_C_EVENT` to it (`GenerateConsoleCtrlEvent`), the event a Ctrl-C typed in that window sends, not a key press. The stream's `done` (cancelled, `stopped_by` shutdown) 0.01 s after it, the server gone with rc 130 3.3 s after it, no second Ctrl-C; the log held "cancelled: 0 image(s)", the shutdown lines and no Traceback, ASGI exception or graceful-timeout line. The page: the title "Server stopped — Local Diffusion" and "Stopped: the server is shutting down — nothing had finished." 0.09 s after the signal (read from the element: the error line was below the 1400x900 view, where 3 s later the topbar read "server not answering — retrying"). Not run: Ctrl-C to an idle server with a model loaded. Evidence: `flux-krea/runs/r50-20261025-091238/` | 2026-10-25 |
| ERNIE-Image Turbo on the fp8 path, one process (Windows, model offload, `ernie_card.py`) | The same seed twice is the same hash (`e5274c48832008cd`), and again after the rewriter was registered and the offload chain rebuilt; seed + 1 differs. Two processes were not compared. The live preview switches itself off ("no latent-to-RGB map for AutoencoderKLFlux2") | 2026-10-25 |
| ERNIE's Expand on the card: the rewriter's first use through the real server (fp8, model offload, scratch gallery; the rewriter's `pe/` not cached, so the real first use, into the user's Hugging Face cache) | Before it, the status listed "ERNIE's prompt rewriter" at 7,663,346,520 bytes. First Expand: 132.7 s; the file landed at 110.5 s (69.4 MB/s), then about 15 s to load the rewriter and rewrite on the card and about 5 s of Marian on the CPU; 7,371 MiB whole-card, no shared rise. `card_holder` read "expand" through the download and the rewrite and cleared while Marian ran. Warm: 11.1 s over HTTP; in-process 8.0-9.9 s (15.8 s the first from disk), 6.49-6.54 / 6.55-6.61 GiB, 7,371-7,435 MiB; the process's RAM rose about 6.3-7 GiB with the rewriter resident (to 20.4-21.2 GiB). Stop 1.5 s into a rewrite: the Expand request answered 409 at once, the rewrite held the card about 8 s more (`pe.generate` runs to its end; in-process `Cancelled` came at 8.4 s), and a Generate sent straight after queued and started only once the holder had become the job - no overlap (the harness's own overlap flag misread this; `status.jsonl` shows it). The rewriter answers in Chinese for an English prompt too, and two expansions of one prompt differ (sampling). Not driven: the page's Expand line and clock over the download. Evidence: `ernie/run-server/` | 2026-10-25 |
| Marian (`Helsinki-NLP/opus-mt-zh-en`) on first use and on whole expansions (CPU) | First use from an empty scratch cache: 20.4 s with the download (315,369,018 bytes), 1.06 GiB RSS, no GPU; warm 0.4-1.0 s. No second weights file was fetched in 150 s after (the only `model.safetensors` was the 0-byte 404 marker). **A defect:** `MarianTranslator._chunks` packs sentences up to the token limit, so a whole expansion goes in as one chunk, and Marian translates part of it and drops the rest with no error: a 7-sentence text kept only sentences 2-3, where one sentence at a time kept all 7 (`marian_split_check.py`), and every expansion's English on the card lost most of its text (one kept about 3 of 9 sentences). The rendered and recorded prompt is the Chinese, so images are unaffected; the page's gloss is not. Fixed 2026-10-25 (GEN#21): one sentence per sequence, an over-long one cut at the limit; checked in tests (stub tokenizer and model) only, the real Marian not re-run on the CPU or on the card. Evidence: `ernie/run-step1.log` | 2026-10-25 |
| Seed reproduction under Nunchaku NVFP4 within one process (WSL 2, lite-infer NVFP4 r32, model offload; Z-Image Turbo and FLUX.1 Krea dev) | Not bit-for-bit either: the same request again in the same process gave a new hash every time, text-to-image as well as inpaint. Inpaint against its repeat: 0.72 / 0.84 levels mean over the whole image, 3.4% / 4.3% of pixels off by more than 8 (all inside the mask, about 9% of the frame), against 2.24 / 3.28 and 7.6% / 8.7% for seed + 1; a batch's images differ from their solo passes by as much. Text-to-image run again later in the process: 10.84 / 3.80 levels, 41% / 13% of pixels. (Inpaint section, "Under Nunchaku") | 2026-10-25 |

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

The rest of what these rows found was fixed the
same day and checked then on the fake server and in pytest, not again on
the card at that time: a fitted batch's columns
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
(5ec8adb). The 2026-10-25 session that was to check the first five and
2b89ed5 (above) on the real server with FLUX.1 Krea [dev] stopped at the RAM guard during its one
load (Checkpoints, "RAM at the load"), so that attempt did not reach the
page checks; 81781bf's sentence was checked there (the load failures row
above). The [2026-11-18 follow-up](#windows-follow-up-measurements-2026-11-18)
adds six real page assertion sets with explicit remaining temporal/manual
gates. The checked-in packs and their maintenance belong to the
[CARD#10 owner](maintainer/plan-card-harness.md#card10-step-packs).

Not verified: `AURADREAM_DEVICE=cuda:1` (one card here); ERNIE-Image
Turbo's 4 MP batch of two on fp8 (still projection-refused, not run; the
2026-11-18 reproduction/first-use Expand results are below); a live-preview map for
FLUX.2's VAE (ERNIE, klein);
SDXL's live-preview frames on the card (the map's CPU fit is a Behaviour
row; Animagine XL 4.0's LoRA row is above, its fp8 speed and memory are the
October 5 row above, and the quantization and offload settings of its
earlier successful user run were not recorded). Illustrious XL
v2.0-STABLE LoRA behavior and batch scaling are also unmeasured; its fp8 peak
is the October 5 row, and its int8, full-bfloat16 and sequential-offload
runs, each repeating its own pixels, are in "Quantization and offload".

### SDXL live-preview capture (2026-11-17)

One real Animagine XL 4.0 fp8/model-offload render on Windows/RTX 5080,
1024 square, 28 steps, CFG 5 and seed 12345 produced fourteen 256-square
JPEG previews at steps 1, 3, ... 27. The ordinary backend callback and real
job SSE stream delivered byte-identical frames; their JPEG and decoded RGB
hashes were rechecked. This captured the actual non-flow scheduler path
(`EulerDiscreteScheduler`) and the fitted SDXL VAE map, without changing it.

The [first frame](maintainer/gen6-preview-2026-11-17/step-01.jpg) is noise.
The [middle](maintainer/gen6-preview-2026-11-17/step-15.jpg) and
[late](maintainer/gen6-preview-2026-11-17/step-27.jpg) frames progressively
show the person, blue coat/red scarf, warm light at left and cool snowy
forest in the [final image](maintainer/gen6-preview-2026-11-17/final.png).
That is the root agent's visual observation; Matt's preference for showing
the noisy first frame and a real-page glance remain unaccepted (`GEN#6`).
Callback/SSE capture is not a browser-layout acceptance check.

One load took 15.891 s; the saved render took 25.917 s. The first frame
arrived 17.816 s into the call. These are descriptive timings with other
CPU work on the machine, not a speed comparison. Peak allocator values
were 3,350,174,208 / 3,724,541,952 bytes (3.12/3.47 GiB), matching backend,
job/API, SQL and PNG; whole-card peak was 5,386 MiB. Shared rises were
82 MiB process / 118.6 MiB adapter. Load peaked at 10.16 GiB working set /
12.09 GiB private; free-RAM/commit minima stayed at 29.75/34.14 GiB.
The unchanged live guards passed all 186 samples (0.249826 s mean interval,
0.277368 s longest). Unload and owned process cleanup completed.

The [receipt](maintainer/gen6-preview-2026-11-17.json) retains exact hashes,
guard windows, source manifests and provenance. The
[owner review](maintainer/review-2026-11-12-todo.md#gen6-card-capture-2026-11-17)
records what remains for Matt. Raw evidence stays in
`work/gen6-sdxl-preview-20261117/`.

### Ordinary text-to-image cancel and recovery (2026-11-18)

One guarded Windows/RTX 5080 load of FLUX.1 Krea dev, fp8/model offload,
Eager/Native, diffusers 0.42.1, torch 2.13.0+cu130, accelerate 1.15.0 and
NVIDIA driver 616.92. The actual unmodified `backend.generate` ran the fixed
scene prompt, seed 4471, 1024-square, 28 steps and CFG 3.5: one normal image,
a cancellation after step 3, then the same image again. No admission override,
HTTP server, browser or live gallery was involved.

The earlier Native comparison had completed all twelve fixed cases and four
exact replays before the unchanged 15,500 MiB guard stopped its first recovery
at 15,768 MiB. The source and card trace pointed to retained transformer residency. That failure
is retained; the unsafe source was not rerun. After the production cleanup fix,
normal and recovery took 42.620/40.737 s and both decoded RGB hashes were
`d62ee9fc1308fe2a617619cdfb3fe8ba0e5a56c418dfe541cb32a9da419d697b`, exactly
the historical Native control. These timings are descriptive, not a speed
comparison across the changed desktop baseline.

After cancellation both text encoders, transformer and VAE were on CPU.
Allocated/reserved memory was 32.001/60 MiB; exactly one loaded-owner cleanup
ran per outcome on this FLUX.1 path. The lower idle desktop alone cannot
establish recovery: the explicit device, allocator and cleanup-call checks do.
The 530 raw guard rows independently re-evaluate to `ok`, mean/max sample gaps
0.250292/0.270638 s. Whole-card peak was 14,199 MiB over 966; process
working-set/private peaks 27.676/40.118 GiB, free-RAM/commit minima
20.631/14.055 GiB, and adapter/process shared rises 82/82 MiB. Initial
RAM/commit admission was 40/50 GiB; live 4 GiB floors and all card/shared
limits stayed unchanged. Unload collected the transformer, reported not ready
and left 32/32 MiB allocated/reserved; owned processes exited and a fresh
post-exit card read was 966 MiB at 0% utilization.

The [receipt](maintainer/cancel-fix-2026-11-18.json) binds runtime source hashes,
raw evidence and the focused/full CPU gates. This qualifies the bounded Native
backend cancellation recovery only. HTTP/job/UI recovery, other families and
recipes, cuDNN, regional compilation and FBC remain separate tests. Public
experimental admission remains closed.

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
0.678 at 0.25. The pictures were not reviewed then, and their Windows
evidence directories are gone (checked 2026-10-25): the Z-Image Turbo and
Z-Image strengths were re-run that day ("Strengths and offload on Z-Image
and Qwen-Image", below); the SDXL and schnell ladders' pictures cannot be
reviewed.

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

The [2026-11-18 follow-up](#windows-follow-up-measurements-2026-11-18)
measures dev/schnell fp8 Refine, their starting-strength pictures, true CFG
and the separate dev sequential path. Native full-residency FLUX remains
held by a deduplicated weight lower bound rather than an executed fit test.
Reference gaps, not a new sweep: int8/full-bfloat16 Refine and other
unselected offload/family recipes; a Nunchaku pass past
the two above and Qwen-Image's three strengths (below); and the author's own WAI recipe whole (its separate 20
hires steps; the Anime6B preparation itself is measured below). The October 5
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

### R-ESRGAN 4x+ Anime6B (2026-10-21)

The opt-in GAN preparation against Lanczos on WAI v17 and RouWei 0.8, fp8,
model offload, Windows: each checkpoint's 1024x1024 source from the inpaint
runs (seed 12345) refined to 1536x1536 at the authors' range, 0.35 and
0.5, seed 12345, 28 steps. The desktop held about 1.0 GB.

| What | WAI v17 | RouWei 0.8 |
| --- | --- | --- |
| Raw 4x, 1024 to 4096 px, 16 tiles, FP32 | 0.89 s warm (2.45 s first, net built and moved), 1.25 GiB allocated, 4,406 MiB card | 0.94 s (1.12 s first), the same |
| Refine at 0.35, 9 steps: Lanczos / Anime6B | 9.5 / 8.6 s | 10.2 / 9.0 s |
| Refine at 0.5, 14 steps: Lanczos / Anime6B | 10.1 / 11.1 s | 10.5 / 11.9 s |
| Refine peak, both choices | 3.67 GiB allocated, 5,956 MiB card | 3.67 GiB, 5,954-5,969 MiB |

- Determinism: the raw 4x twice is the same hash on both (`b8fce6a51bb0ab49`,
  `a529369709871895`); the Anime6B refine at 0.35 twice is the same hash;
  text-to-image afterwards is the source's hash.
- The GAN costs about a second; the run-to-run spread of a refine is as
  large, so Anime6B is not measurably slower. It leaves the card before the
  VAE runs (0.0 GiB allocated after), so the refine's peak is Lanczos's.
- The tiles: in the raw 4x the step across a tile border (every 1,024 px)
  ranks above most nearby columns (WAI 5 of 6 borders at the 87th-99th
  percentile of their ±64 px, RouWei 2 of 6 at the 93rd-99th), a faint
  seam from the 10 px context Real-ESRGAN itself uses; at 2x zoom none
  shows. After the refine (borders at 384, 768, 1152) the Anime6B images
  rank like the Lanczos ones, which have no tiles (both spread from the 2nd
  to the 98th percentile): nothing survives into the result.
- By eye, at 0.35 Anime6B gives slightly crisper hair strands, eye
  highlights and line work than Lanczos on both; at 0.5 the two are close.
  Neither changes the picture or its colours. One source each, by eye.

Evidence: `wai-gan.log`, `rouwei-gan.log`, and per checkpoint
`gan-4x.png`, `gan-4x-seams.jpg` (WAI), `gan-refine.jpg` and
`gan-refine-face.jpg`, in `%TEMP%/local-diffusion-inpaint-card/`.

### Strengths and offload on Z-Image and Qwen-Image (2026-10-25)

`card_refine.py`, one process and one `backend.load()` per row group, a
scratch gallery: a 1024x1024 seed-12345 source ("A portrait of an old
fisherman in a yellow raincoat on a wooden pier at dawn, ..."; negative
"blurry, low quality" on Z-Image and Qwen-Image), then passes of it. Times
exclude the load and the save; the per-step time leaves out step 1, which
carries the VAE encode. "Card" is nvidia-smi's whole card every 500 ms,
the desktop's 250-1,250 MiB included; shared memory by the Windows
counters, with a stop at card 15,500 MiB, adapter shared +1,024 MiB,
process shared +512 MiB or under 4 GiB of free RAM or commit. Windows fp8
unless named; Qwen-Image is its NVFP4 Nunchaku build under WSL 2.

| Checkpoint, offload | Run | Steps run | Time (per step) | Allocated / reserved | Card | Start sigma |
| --- | --- | --- | --- | --- | --- | --- |
| Z-Image Turbo, model | source, 1 MP | 8 of 8 | 25.7 s first of a process; 10.3 s again (0.85-0.89 s) | 6.43 / 7.17 GiB | 8,667 MiB | - |
| Z-Image Turbo, model | 1.5x (1536x1536) at 0.15 / 0.25 / 0.35 / 0.5 | 2 / 2 / 3 / 4 of 8 | 9.0 / 9.1-9.2 / 11.3 / 13.7 s (2.36 s) | 7.04 / 8.84 GiB | 10,381 MiB | 0.5 / 0.5 / 0.643 / 0.75 |
| Z-Image Turbo, model | 2x (2048x2048) at 0.25 | 2 of 8 | 16.1 s (5.33 s) | 7.86 / 10.09 GiB | 11,657 MiB | 0.5 |
| Z-Image Turbo, none | source, 1 MP | 8 of 8 | 8.9 s (0.85 s) | 12.51 / 15.0 GiB | 15,470 MiB | - |
| Z-Image Turbo, none | 1.25x (1280x1280) at 0.25 | stopped after 1 | - | 11.64 / 15.0 GiB | 15,725 MiB, spilling (shared +274 adapter / +200 process MiB) | - |
| Z-Image Turbo, sequential | source, 1 MP | 8 of 8 | 11.8 s (1.25 s); 10.7 s again | 2.41 / 2.98 GiB | 3,631 MiB | - |
| Z-Image Turbo, sequential | 1.5x at 0.25 | 2 of 8 | 8.3-8.4 s (3.15 s) | 2.42 / 2.98 GiB | 2,307-3,631 MiB | 0.5 |
| Z-Image, CFG 5, model | source, 1 MP | 50 of 50 | 93.5 s first of a process (1.71 s) | 6.77 / 8.44 GiB | 9,223 MiB | - |
| Z-Image, CFG 5, model | 1.5x at 0.15 / 0.25 / 0.35 | 8 / 13 / 18 of 50 | 42.5 / 66.6 / 90.5 s (4.80 s) | 8.10 / 10.58 GiB | 11,411-11,422 MiB | 0.533 / 0.678 / 0.771 |
| Qwen-Image, true CFG 4, NVFP4 (WSL), model | source, 1 MP | 50 of 50 | 76.0 s first of a process (0.97 s; the first step 18.2 s) | 11.75 / 11.98 GiB | 13,028 MiB | - |
| Qwen-Image, NVFP4 (WSL), model | 1.25x at 0.2 / 0.3 / 0.4 | 10 / 15 / 20 of 50 | 22.7 / 34.9 / 38.9 s (1.66 s) | 12.20 / 12.45-12.46 GiB | 13,478-13,526 MiB | 0.344 / 0.478 / 0.590 |

- Every start sigma is `backend.start_sigma`'s prediction. The Z-Image
  Turbo and Z-Image memory repeats 2026-10-05's ("Past 4 MP") to the
  hundredth; the card figures are 0.8-1.6 GB lower, the desktop's share
  being smaller. Qwen-Image's 1.25x pass took 34.9 s where 2026-10-05's,
  with another project's processes busy, took 91.1 s.
- On Z-Image Turbo's 8 steps 0.15 is the same pass as 0.25 (2 steps from
  sigma 0.5, one hash): `start_index = int(8 - 8 * strength)` floors, so
  the strength moves in eighths there, and the next pass down is 0.125 or
  less, one step (not run).
- Offload does not change the pixels: the source, the 1.5x pass at 0.25
  and the text-to-image repeat hash the same under model, none and
  sequential. Offload none fits only the 1 MP render, with the card full;
  even a 1.25x pass spilled at its first step and was stopped, so none
  fits no refine pass on Z-Image Turbo. Its load left 7.2 GiB of RAM free
  at the lowest (31.3 GiB private). Sequential holds 2.4 / 3.0 GiB and at
  most 3,631 MiB of the card and is barely slower on this checkpoint: 1.25 s
  a step at 1 MP against 0.85, and the 1.5x pass 8.4 s against 9.2 (3.15 s
  a step against 2.36, its first step cheaper).
- Repeats: the same pass twice is the same hash on Z-Image Turbo, and the
  text-to-image source afterwards is the source's hash. No hash is
  recorded under Nunchaku (Behaviour).
- What each strength keeps, by eye on the strips and face and coat crops.
  Z-Image Turbo: the composition, coat, hood, beanie and stains are kept
  at every strength up to 0.5 and at 2x, with no artefacts; the face is not
  kept even at the profile's 0.25 - a narrower, smoother face with other
  eyes and fewer lines, a closely related man - and 0.35, 0.5 and 2x drift
  a little further. Z-Image at CFG 5: the composition is kept at 0.15-0.35
  and the face redrawn already at 0.15; all three passes carry fine
  horizontal streaks over the coat's flat areas (a trace of them in the
  source, none on Z-Image Turbo's passes at the same size), unmeasured at
  1.25x or on another source. Qwen-Image: the composition, coat, boats,
  house and rope are kept element for element at 0.2-0.4; the face is
  repainted already at 0.2 (the head more frontal, other eyes and
  wrinkles). So on all three the profile's start keeps the picture but not
  the face.
- The surviving 2026-10-05 WSL pictures, looked at again: Qwen-Image at
  0.3 (1.25x) keeps the picture and redraws the face as above; FLUX.1 dev
  NVFP4 at 0.1 (1.5x) keeps the same man, a little more redrawn (hair,
  eyes, smile) than the Krea dev fp8 line "0.05 and 0.1 keep the face"
  suggests.
- The Nunchaku repo's `main` has moved: it ran at `906c1900ddad`, where
  the recipe and `backend/model-downloads.json` record `41e6737e1712`, and
  the WSL run needed one online fetch of the ungated
  `rootonchair/nunchaku-lite-kernels`.

Evidence: `zimage-qwen/` in the cards directory (`stdout-*.txt`,
`data/<plan>/results.jsonl`, `run.log`, `pics/`, `strip-*.jpg`, the face
and coat crops; `data-wsl/qwen-nunchaku/`; `ladder-1005/`).

### Peak per image and Then refine, WAI v17 (2026-10-25)

The real server (`--defaults --model LyliaEngine/waiIllustriousSDXL_v170
--quant fp8 --offload model`, port 7930, a scratch gallery, no switch and
no download), one load (12.1 s, the card then at 1,220 MiB), driven over
its API by `card_wai.py`. nvidia-smi every 500 ms; the Windows adapter and
per-process shared and dedicated counters about every 1.5 s; free RAM
every 250 ms; stops at card 15,500 MiB, the server's shared memory +512
MiB, the adapter's +1,024 MiB or under 4 GiB free (none tripped). The
desktop held 993 MiB. 1024x1344, 28 steps, CFG 6. "Recorded" is the row's
own `vram_peak_allocated` / `vram_peak_reserved`, the allocator's.

| Job | Time | Recorded | Card | Server's own dedicated | Shared rise: server / adapter |
| --- | --- | --- | --- | --- | --- |
| One image, the first after the load | 12.37 s | 3.22 / 4.37 GiB (3,455,958,016 / 4,695,523,328 B) | 5,774 MiB | 4,094 MiB | +8 / +82 MiB |
| The same request again | 9.63 s | 3.22 / 4.37 GiB | 5,776 MiB | 4,094 MiB | +8 / +82 MiB |
| Batch 2 | 16.75 s (8.28 s recorded each) | both rows 3.63 / 4.38 GiB | 5,778 MiB | 4,786 MiB | +8 / +82 MiB |
| Refine to 1536x2016 at 0.4, 11 of 28 steps | 12.76 s (the first img2img; first step 4.1 s) | 3.89 / 4.70 GiB | 6,114 MiB | 5,122 MiB | +8 / +80 MiB |

- The record, `GET /api/images/{id}` and the PNG's `local-diffusion` chunk
  agree byte for byte on all 53 images of the session; the repeat is the
  same hash and the same pair; a batch's rows share one pair, as the plan
  says (a batch of two allocates 0.41 GiB more and reserves 0.01 more).
- The recorded pairs equal this file's: 3.22 / 4.37 GiB is the 2026-10-03
  refine table's source, and 3.89 / 4.70 its 1.5x pass ("Past 4 MP",
  7,676 MiB whole-card then with a larger desktop). The 2026-09-26 row's
  3.69 GiB allocated was another prompt; its script is gone, so it was not
  re-run.
- The allocator does not see the CUDA context, the desktop or a spill:
  the card less the desktop's 993 MiB came to 4,781-5,121 MiB, about 0.3
  GiB above reserved each time; the server's own dedicated counter,
  sampled every 1.5 s, read 4,094-5,122 MiB, below the card's figure
  for the 1024x1344 images. The server's shared memory, 74 MiB from before the load (the
  context's bookkeeping), rose 8 MiB.

Then refine at 1.5x and 0.4 (1536x2016, 11 of 28 steps), six seeds a
round, two rounds in opposite orders: apart, `/api/generate` with six
runs then Mass refine (`/api/refine/batch`) of the six; together,
`/api/generate` with six runs and `then_refine {scale 1.5, strength 0.4}`.

| Round | Mode | Wall (server) | Generation sum | Base / pass mean | Recorded peak | Card |
| --- | --- | --- | --- | --- | --- | --- |
| 1, seeds 20001-20006 | apart (57.58 + 68.50 s) | 126.08 s | 124.46 s | 9.50 / 11.24 s | 3.89 / 4.70 GiB | 6,114 MiB |
| 1 | together | 125.68 s | 124.01 s | 9.51 / 11.16 s | 3.89 / 4.70 GiB | 6,114 MiB |
| 2, seeds 30001-30006 | together | 126.05 s | 124.39 s | 9.55 / 11.18 s | 3.89 / 4.70 GiB | 6,114 MiB |
| 2 | apart (57.83 + 68.53 s) | 126.36 s | 124.67 s | 9.54 / 11.24 s | 3.89 / 4.70 GiB | 6,114 MiB |

- Together cost no more than apart: 0.40 and 0.31 s less wall time of
  about 126 s, within what one round per mode can tell apart.
- The alternation does not drift: in both together jobs every base
  recorded 3.22 / 4.37 GiB and every pass 3.89 / 4.70, passes 11.11-11.23
  s and bases 9.48-9.59 s. Steps ran 0.278-0.279 s (base) and 0.715-0.717
  s (pass) in every mode, and a pass's first step 2.49-2.62 s (one 2.97 s,
  apart) whether a base
  or a pass came before it, so that cost is the pass's own start (the
  source's encode), not the swap between the text-to-image pipeline and
  its sibling.
- Bases and passes are byte-identical between the two modes in both
  rounds; the together job's base run id is plain Generate's
  (`a2c1c380f7aa`, `bf09ea65b232`) and its passes sit under their own pass
  run id (`13b16d77532d`, `0e303690afcb`), each pass's `refine` its own
  base.
- The card peaked at 6,114 MiB, the server's shared memory rose 8 MiB, and
  free RAM stayed at or above 43.4 GiB through the then-refine rounds
  (39.2 at the load and 38.9 during the first render; the server held 9.5
  and 9.8 GiB then and about 5.7 GiB after).
- By eye (seed 20001's base and pass, the first image): clean; the pass
  keeps the composition, pose and palette, sharper line work and hair, the
  face slightly redrawn, as 0.4 does.
- The server logged "live preview switched off: no latent-to-RGB map for
  AutoencoderKL" at every cell's start, bases and passes alike (52 times):
  SDXL has no preview map (`GEN#6`), not a fault of the sibling.

Not measured: the same on FLUX.1 Krea [dev], whose session stopped at the
RAM guard during its load (Checkpoints, "RAM at the load"); the stops'
behaviour on a real spill or RAM trip in this harness (nothing came near).
Evidence: `sdxl-wai/run-card/` in the cards directory (`summary.json`,
`results.jsonl`, `samples.jsonl`, `server.log`, `images/`).

### Anima Turbo private Refine (2026-11-16)

The fixed private Turbo v1.1 trial completed on Windows/RTX 5080, fp8/model
offload, Eager/Native, diffusers 0.42.1, torch 2.13.0+cu130 and torchao
0.18.0. One cached load, immutable 1024-square source and output, eight full
steps, CFG 1, no Mu/LoRA/reference, seeds 4471/4472. The
[receipt](maintainer/anima-refine-turbo-2026-11-16.json) retains exact source,
output and metadata proofs, guard samples, runtime/source hashes and agent
visual notes. The 2026-11-17 stage 3 slice uses only this bounded recipe;
its integrated real-server acceptance passed below.

| Strength | Actual steps / start sigma | Time | Allocated / reserved peak | Whole-card peak |
| --- | --- | --- | --- | --- |
| 0.125 | 1 / 0.300000012 | 2.733 s | 2470.98 / 2640 MiB | 4394 MiB |
| 0.25 | 2 / 0.5 | 3.088 s | 2471.48 / 2640 MiB | 4395 MiB |
| 0.4 | 4 / 0.75 | 3.928-4.712 s, including repeat/recovery | 2471.48 / 2640 MiB | 4394-4398 MiB |

The profiles' measured lines originate in the private same-size passes;
their later integrated and standalone acceptance is recorded below:

| Checkpoint / loader | Source and output | Full steps / strength | Scope |
| --- | --- | --- | --- |
| Anima Turbo v1.1, fp8/model | 1024x1024 | 8 / 0.25 | CFG 1; independent private, public API and standalone CLI acceptance |
| Anima Aesthetic v1.1, fp8/model | 1024x1024 | 30 / 0.125 | CFG 4; independent private, public API and standalone CLI acceptance |
| Anima 2.9B Preview v1, fp8/model | 1024x1024 | 30 / 0.125 | CFG 4; independent private, public API and standalone CLI acceptance |

Same-seed repeats, serial batch members versus their solo seeds and recovery
after cancellation have equal full RGB hashes. Cancel stopped after one
completed callback without a decoded output; the direct baseline after the
passes/cancel reproduces the first baseline. All ten image/metadata hashes
were independently rechecked. The first four cases also match the retained
earlier process's completed cases.

Load took 4.389 s, peaking at 7.768 GiB sampled process RAM. Session free-RAM
and commit minima were 35.413/38.765 GiB; process/adapter shared rises were
82/84.109 MiB. All 296 guard verdicts were `ok`, with 0.25 s requested spacing
and a 0.294277 s maximum gap. Cleanup retained 32 MiB allocated/reserved
before exit; after exit whole-card usage was 1448 MiB versus 1445 at admission.
The owned Windows power request was acquired and released.

Agent visual review retained the woman, blue coat, red bicycle and park.
Strength 0.25 is a provisional conservative opening candidate for this recipe;
0.4 redraws more face/background/shoe detail. Broken fine anatomy and unwanted
generated caption text remain. Matt has not reviewed these images himself.
This Turbo experiment qualifies no larger size, other aspect, checkpoint
variant, loader or platform; the independent variant results follow below.
The first attempt stopped on a Windows IPC publication race after
four cases; its evidence is unchanged. Immutable handoffs and a real Windows
locked-reader regression preceded the successful fresh one-load trial. The
[Anima plan](maintainer/plan-anima-refine.md#turbo-one-load-result-2026-11-16)
owns the completed bounded public/standalone evidence and passing final
integrated regression gate. Original private receipts retain their historical
source and public-support fields unchanged.

### Anima Turbo integrated Refine acceptance (2026-11-17)

The real public server passed one-load acceptance on source commit
`9341e57`: eleven outputs matched the private RGB controls across Refine,
serial batch/solo, cancellation recovery, Then refine at scale 1, Mass
refine, More seeds, retained Finish and CLI `--from-image --via-server`.
The CLI carried source/strength from PNG without explicit overrides.
Cancellation after real SSE step one unwound at callback two with no output;
source fixtures stayed unchanged. Five bad-request probes refused before
queueing, same-size Re-render pruned with 409 and enlargement stayed held.

Every saved image/metadata hash, SQL row, captured API/job record and allocator
peak pair was independently checked. Actual Refine progress/sigma was 2/0.5
at 0.25 and 4/0.75 at 0.4, with per-image owner cleanup. Load took 11.12 s;
peak server RSS/private was 8.65/10.63 GiB, whole-card 4983 MiB, shared rises
82/82.1 MiB process/adapter, free-RAM/commit minima 25.61/19.92 GiB.
All 306 samples passed. One 2.437 s startup gap preceded loading; the 238
load/render samples averaged 0.249985 s with a 0.266288 s maximum.

Unload, power release and owned PID/port cleanup passed. The
[receipt](maintainer/anima-refine-turbo-public-2026-11-17.json) retains all
proofs and scope. This is API and CLI-via-server acceptance; it does not claim
a native-browser GPU run, standalone CLI loading, restart recovery or wider
hardware/loader/shape support. The original visual limitations still apply.

### Anima Aesthetic private Refine (2026-11-17)

The separate one-load Aesthetic v1.1 experiment passed at 1024 square on
Windows/RTX 5080, fp8/model offload, Eager/Native, 30 steps and CFG 4,
diffusers 0.42.1, torch 2.13.0+cu130 and torchao 0.18.0. The pinned
selector/revision and all reproduction proofs are in the
[receipt](maintainer/anima-refine-aesthetic-2026-11-17.json).

| Strength | Actual steps / start sigma | First-pass time |
| --- | --- | --- |
| 0.125 | 4 / 0.315789521 | 5.131 s |
| 0.25 | 8 / 0.521739185 | 7.229 s |
| 0.4 | 12 / 0.666666687 | 10.143 s |

All ten saved images and metadata hashes were checked. Same-seed repeats,
serial batch versus solo seed, recovery after callback-one cancellation and
direct-after pixels match their controls. Cancellation decoded no output.
Load took 6.918 s and peaked at 8.257 GiB process RSS; Refine peaked at
2472.98/2642 MiB allocated/reserved and 4982 MiB whole-card. Free-RAM/commit
minima were 25.803/18.663 GiB, shared rises 82 MiB process / 81.719 MiB
adapter. All 601 samples passed (maximum gap 0.253483 s); cleanup retained
32 MiB allocator memory before exit and returned to 2031 MiB whole-card.

Agent review retained the woman, blue coat, red bicycle and composition.
Strength 0.125 is the conservative opening candidate: some clearer outlines,
with flattened foliage and higher contrast. Higher strengths redraw more and
can strongly simplify background texture; fine geometry defects remain.
These are bounded observations, not Matt's review or a quality guarantee.
Only this recipe/size/loader was measured by the private trial. Its later
bounded public/standalone acceptance is recorded below; Preview has separate
private and integrated results.

### Anima 2.9B Preview private Refine (2026-11-17)

Preview's exact cached source was converted separately by the normal atomic
CPU converter, with no downloads, CUDA initialization, backend load or
inference. Conversion took 10.583 s (19.108 s supervised) and peaked at
7.555 GiB RSS. The first import attempt met the CUDA guard before weights;
the repaired CPU context passed a fresh import/guard regression, and both
attempts remain recorded. The resulting pinned 40-layer cache was used for
the independent one-load GPU trial.

Windows/RTX 5080, fp8/model, Eager/Native, diffusers 0.42.1, 30 steps/CFG 4,
1024-square source/output: strengths 0.125/0.25/0.4 ran 4/8/12 steps at
sigmas 0.315789521/0.521739185/0.666666687. Ten images, all source/PNG/metadata
hashes, repeat/batch-solo/recovery/direct-after controls and callback-one
cancellation with no output passed. Load took 6.440 s; peak process RSS was
10.399 GiB and Refine allocator/reserved peaks 3277.10/3446 MiB. Whole-card
peak was 5783 MiB; free-RAM/commit minima 23.814/18.222 GiB. All 723 sampled
verdicts passed, maximum gap 0.250633 s. Cleanup retained 32 MiB before exit;
after exit whole-card usage was 2028 MiB against 2022 at admission.

Agent review retained person, blue coat, red bicycle and composition.
Strength 0.125 is the conservative opening candidate, with sharper but
changed foliage texture; higher strengths can flatten/saturate it and redraw
face, basket and wheel details. Existing cropped geometry/anatomy defects
remain. No author recommendation, general improvement or Matt acceptance is
claimed. The [receipt](maintainer/anima-refine-preview-2026-11-17.json)
records exact configuration/source/guard proofs and visual notes. Later public
and standalone acceptance is recorded below; no larger size, aspect or
alternative loader/platform is qualified.

### Anima Aesthetic/Preview nonempty negatives (2026-11-17)

Separate one-load probes reused each variant's original 1024-square source
at 30 steps/CFG 4, fp8/model offload and the same Windows/RTX 5080 envelope.
At strength 0.125, `low quality, blurry` reached the real negative-conditioning
branch: each result differs from its retained empty control and equals its
repeat and post-cancel recovery. Cancel at 0.4 stopped after callback one
without an image; empty-negative direct-after renders match their baselines.
Actual call kwargs, source/PNG/metadata proofs and all twelve calls were audited.

Aesthetic/Preview loads took 6.615/7.843 s, process RSS peaked at
7.121/9.098 GiB and whole-card at 4978/5738 MiB. Both guards passed every
sample and normal cleanup/power release succeeded. Agent review of the two
new negative outputs retained the subject, coat, bicycle and composition;
existing fine geometry defects remain. The
[receipt](maintainer/anima-refine-negative-2026-11-17.json) distinguishes
the completed 0.125 branch from the 0.4 cancellation check. This establishes
the branch and reproduction controls, not a quality promise for arbitrary text.

### Anima Aesthetic/Preview integrated Refine acceptance (2026-11-17)

Separate one-load real public-server sessions passed on `3925cbd`, inside
the measured Windows/RTX 5080 CUDA 0, fp8/model, Eager/Native, diffusers
0.42.1, 1024-square source/output, 30-step/CFG-4 envelope. Each saved twelve
outputs matching its private full RGB references across the 0.125 opening,
nonempty negative, 0.4 serial batch/solo, cancel/recovery, same-size Then
refine, Mass refine, More seeds, retained Finish and PNG-carried CLI-via-server
request. All source fixtures, PNG/full-RGB/exact-metadata hashes, SQL and
captured API/job records were independently checked. Each recorded peak pair
also matches the post-call CUDA counters and `last_peak` observation.

The 0.125 passes ran four steps from sigma 0.315789521; 0.4 ran twelve from
0.666666687, twenty-four for its serial batch. The `low quality, blurry`
case retained that text and applied `true_cfg=4` in saved job/API/SQL/PNG
data. Its raw request has no separate True CFG override: the private negative
artifact observed pipeline `guidance_scale=4` without `true_cfg_scale`;
the public session observed caller intent and matching pixels, without a
pipeline-kwargs hook. Cancel after SSE step one saved nothing and cleared
the peak; retained Finish recovered that SQL-defined cell in the same session.
Thirteen owner-hook releases per variant include every image and cancel.
Five invalid probes refused before queueing; same-size Re-render pruned with
409 and enlargement remained 422.

| Measurement | Aesthetic | Preview |
| --- | --- | --- |
| Cached load | 14.29 s | 14.79 s |
| Whole-card peak | 4801 MiB | 5632 MiB |
| Server-tree RSS / private peak | 7.35 / 10.43 GiB | 10.96 / 13.43 GiB |
| Minimum free RAM / commit | 24.82 / 22.37 GiB | 21.63 / 20.63 GiB |
| Shared rise, process / adapter | 82 / 221.5 MiB | 82 / 93.1 MiB |
| Guard samples / mean gap / longest gap | 603 / 0.249869 / 0.274964 s | 765 / 0.249919 / 0.278056 s |

No guard tripped; both unloaded, released power and left their owned
processes gone. The [combined receipt](maintainer/anima-refine-variants-public-2026-11-17.json)
preserves the 193-file start/end manifests and exact `3925cbd` identity,
separately from the later CLI timeout correction. This is public API and
CLI-via-server acceptance, not native-browser GPU presentation, restart
recovery, a speed qualification or wider hardware/loader/shape support.
Agent visual findings and existing anatomy/texture limits remain as recorded
in the private receipts; Matt has not supplied his own acceptance.

### Anima standalone CLI Refine acceptance (2026-11-17)

Three separate ordinary `python -m backend.generate` invocations passed on
`90eb8c5`, each one fresh load, one unchanged source row/PNG and one new
expected Refine result. Turbo used 8 steps/CFG 1/strength 0.25; Aesthetic and
Preview used 30 steps/CFG 4/strength 0.125. Official selector ids, seed 4471,
1024-square source/output, Windows/RTX 5080 CUDA 0, fp8/model and Eager/Native
were retained, without HTTP generation, `--ignore-server`, quiet mode or
generation patches. Each source/private RGB, PNG and exact metadata hash was
rechecked against SQL and source provenance. The recorded backend peak pair
agrees between SQL and PNG; no independent Torch allocator observation was
installed in these standalone processes. OS RSS/private measurements are
separate.

| Measurement | Turbo v2 | Aesthetic | Preview |
| --- | --- | --- | --- |
| CLI ready line / saved render | 17 s / 4.381 s | 12 s / 6.731 s | 13 s / 7.596 s |
| Recorded allocated / reserved bytes | 2591532544 / 2768240640 | 2593108992 / 2770337792 | 3436291072 / 3613392896 |
| Whole-card peak | 4742 MiB | 4739 MiB | 5541 MiB |
| Watched-tree RSS / private peak | 7.69 / 9.86 GiB | 8.32 / 10.18 GiB | 10.42 / 12.64 GiB |
| Minimum free RAM / commit | 21.80 / 21.93 GiB | 21.05 / 20.59 GiB | 19.23 / 19.15 GiB |

All guards passed with the unchanged live floors/card/shared limits,
24 GiB free-RAM/commit admission and 16 GiB RSS cap. Shared process rise was
82 MiB each, adapter rise at most 83.7 MiB. Preview's 1.784 s startup sample
gap preceded the literal CLI child; its maximum thereafter was 0.264736 s.
Every process exited zero, released power and left no owned gate/CLI process;
port 7992 was bindable at audit. All 193 initial/final/current source hashes
agree. Times are descriptive, and negative standalone CLI was not tested.

The original Turbo v1 remains a failed pre-load refusal: its two-second
server probe expired before Windows returned closed-port refusal, while
unspecified child stdio hid the refusal output. The separately reviewed
five-second probe and explicit diagnostic gate stdout/stderr forwarding
preserve admission. Turbo v2 is the successful new attempt. The
[standalone receipt](maintainer/anima-refine-standalone-cli-2026-11-17.json)
retains successful proofs and the unchanged failed v1 evidence; it does not
relabel that failure or widen the measured envelopes. The final integrated
CPU gate passed 11,071 tests with 325 skips; Ruff and all 217 Pyright sources
passed. `GEN#48` is complete for the documented bounded envelopes, with the
full validation and limitations in the owner.

### Four-region SDXL tiled-refine look (2026-11-17)

`GEN#49`'s exploratory one-load session used the exact cached Animagine XL
4.0 revision `2b7c1b397761bf5bd3cc42e5b39ec99314a75a96`, Windows/RTX 5080,
fp8/model, Eager/Native and diffusers 0.42.1. A new 1024-square synthetic
render became an exact CPU Lanczos 2x fixture, honestly marked as such.
Four ordinary masked-area jobs then traversed the 2048-square canvas
top-left, top-right, bottom-left, bottom-right at strength 0.35,
28 full steps/CFG 5 and seeds 12346-12349. Overlap was 128 pixels, interior
inward feather 64; `bbox16:32@1` cropped 1120 square and denoised at 1040
square before the existing resize/overlay. Later tiles saw the prior result.

The first tile cancelled without a decoded/saved output; recovery and its
exact repeat matched. Six diffusion images plus the explicitly CPU fixture
were stored; the four-result chain excludes the extra repeat. Each completed
crop ran nine callbacks. All five cropped outputs reconstructed bit for bit
from their raw working tiles and installed crop/overlay helpers. Outside-mask
maximum difference was zero for every pass. Source/mask, job/API/SQL/PNG and
independent allocator checks passed. Load took 13.59 s; the chain's API times
were 7.20/7.13/6.91/6.27 s including observations and validation, not pure
model benchmarks. The 72.21 s guard retained 289 samples, longest gap 0.270 s.
Whole-card peak was 5951 MiB, shared rises 96.3/82 MiB adapter/process,
RSS/private 9.31/11.27 GiB, free-RAM/commit minima 23.86/18.23 GiB. The
24/24 GiB admission, 16 GiB RSS cap and unchanged card/shared/4 GiB live
floors held. Unload answered, owned server processes were scoped/reaped
(expected exit 15), port 7996 closed and power was released.

Root and independent visual reviews inspected the overview and native centre,
horizontal and vertical source/final crops. No obvious rectangular boundary,
centre cross or abrupt colour step appeared, and major cabin/dock/water lines
continue. Fine bracing, roof, foliage and reflections redraw and often smooth;
a thin upright behind the roof is more prominent or possibly introduced.
The baseline did not realize the requested bridge/river faithfully. Fidelity
and useful detail gain remain unestablished. This supports writing the
[bounded plan](maintainer/plan-tiled-refine.md), not product qualification,
a default, general seam freedom or TODO closure. The [receipt](maintainer/gen49-sdxl-four-region-2026-11-17.json)
preserves raw hashes, exact scripts/masks/tiles/intermediates and separate
visual reviews. Auditing did not change the raw evidence.

The later private CPU orchestration reproduces this same four-tile chain
exactly when injected with its retained raw outputs. All four intermediate
RGB proofs and the final match, using four observed calls to the installed
overlay helper. Its 36 callbacks are synthetic; this is not new inference,
card, fidelity, speed or release qualification. The
[CPU replay receipt](maintainer/tiled-refine-raw-cpu-replay-2026-11-17.json)
retains the passing attempt and two earlier import-guard failures separately.

### Peak per image, FLUX.1 Krea dev; guarded Refine stop (2026-11-16)

The authorized overnight session completed `GEN#15`'s remaining Windows
fp8/model-offload text-to-image check through the real server and a scratch
gallery. Driver 616.92, torch 2.13.0+cu130, torchao 0.18.0, diffusers 0.42.1,
Eager/Native, offline cached weights; one load. Five 1024-square, 28-step,
CFG-3.5 images each recorded **12,880,261,120 allocated / 13,555,990,528
reserved bytes** (12.00/12.63 GiB). Independent process allocator readings,
backend, saved job/API records, SQL and PNG metadata agree exactly. Seed
12345 repeats full RGB pixels and the 2026-10-27 fixed-loader baseline.
The inspected baseline is a coherent fox in a snowy forest at dawn.

Load took 49.76 s and peaked at 28.39 GiB working set / 35.33 GiB private.
The first render took 64.786 s (44.681 s repeat); direct whole-card peaks
were 14,632-14,689 MiB. Shared rises stayed at or below 82 MiB process /
84.6 MiB adapter. The first render's private peak was 45.44 GiB, with only
4.03 GiB sampled commit headroom: a narrow margin above the unchanged
4 GiB floor. Later sessions must budget for the render as well as the load.

`GEN#16` remains unqualified. Its first 1536-square Refine (1.5x, strength
0.1) reached **15,632 MiB whole-card** before a completed denoising step;
the unchanged 15,500 MiB guard killed the owned server tree. No pass was
saved, no alternation/timing comparison completed and no limit was relaxed.
The sampled shared rises did not trip; this is a card-limit stop, not proof
of a completed spilled render. The owned port closed and the card returned
to 1,441 MiB. The 1,210 guard samples averaged 0.249965 s apart (longest
0.309916 s), with held PDH counters and an independent watchdog.

The [compact receipt](maintainer/flux-memory-2026-11-16.json) retains exact
image/metadata hashes, source identifiers, guard windows and the failed
attempt. Full API/SQL/PNG/allocator/guard evidence remains under
`work/flux-gen15-gen16-20261116/`. The dirty source diff and manifest identify
the actual runtime; this was not an unmodified main checkout. The
[peak plan](maintainer/plan-peak-vram.md#phase-3-result-2026-11-16) and
[Then refine plan](maintainer/plan-then-refine.md#guarded-flux-attempt-2026-11-16)
own the detailed result and remaining work.

## Inpaint

Measured 2026-10-17 on the RTX 5080, Windows, fp8, model offload, one
throwaway gallery per checkpoint (the production gallery was not used). The
desktop held 3,139-4,609 MiB of the card, more than on the refine runs above,
so "Card" here carries about a gigabyte more of other processes. "Card" is
`nvidia-smi`'s whole-card `memory.used`, streamed every 500 ms by a watchdog
that would have cancelled a pass past 15,500 MiB (none tripped);
allocated / reserved are the process's own allocator. Each source is a
seed-12345 render at 1024x1024, the larger sizes refine passes of it at the
profile's strength; the mask is an ellipse over the middle (40 % x 30 % of
the frame) and the inpaint's prompt asks for another scarf. Every pass ran at
the inpaint class's library default (0.9999 on SDXL, 1.0 on Z-Image), one
size per invocation in the foreground. Times exclude the load and the save;
the per-step time leaves out the first step, which carries the source's and
the mask's encode.

| Checkpoint | Size | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- | --- |
| WAI v17, CFG 6 | 1024x1024, 1 MP | 27 of 28 | 9.0-9.9 s (0.24 s) | 3.57 / 4.55 GiB | 7,753-8,148 MiB |
| WAI v17 | 1536x1536, 2.4 MP | 27 of 28 | 19.1 s (0.56 s) | 3.67 / 4.56 GiB | 8,377 MiB |
| WAI v17 | 2048x2048, 4.2 MP | 27 of 28 | 37.0 s (1.14 s) | 4.22 / 5.27 GiB | 9,130 MiB |
| WAI v17 | 2896x2896, 8.4 MP | 27 of 28 | 98.0 s (3.21 s) | 5.47 / 7.48 GiB | 11,384 MiB |
| Animagine XL 4.0 | 1024x1024, 1 MP | 27 of 28 | 8.1-9.1 s (0.22 s) | 3.28 / 3.62 GiB | 7,467-7,472 MiB |
| Animagine XL 4.0 | 2048x2048, 4.2 MP | 27 of 28 | 37.0 s (1.08 s) | 4.22 / 5.26 GiB | 9,194 MiB |
| Animagine XL 4.0 | 2896x2896, 8.4 MP | 27 of 28 | 92.0 s (2.98 s) | 5.48 / 7.49 GiB | 11,479 MiB |
| Z-Anime, CFG 4 | 1024x1024, 1 MP | 40 of 40 | 83.0-87.3 s (1.86 s) | 6.93 / 8.61 GiB | 12,708-13,014 MiB |
| Z-Anime | 1536x1536, 2.4 MP | 40 of 40 | 211.8 s (4.92 s) | 8.11 / 10.58 GiB | 15,376 MiB, with the desktop at 4,386 |

An inpaint costs what a refine pass of the same size costs: the SDXL rows
match their refine rows above to a hundredth of a gigabyte, and the guard's
top size, 8.4 MP, runs on both SDXL checkpoints with a third of the card to
spare. Z-Anime fills the card at 2.4 MP (its per-step time did not climb, so
it did not spill there), as its refine pass did at 4.2 MP under a smaller
desktop share; 4.2 MP was not run.

| What | Result |
| --- | --- |
| The inpaint class over the loaded components | `StableDiffusionXLInpaintPipeline` and `ZImageInpaintPipeline` each ran from `refine.build` over the fp8-quantized, offload-hooked components, with no second load; the allocator was back to 0.03 GiB after every pass |
| Determinism | The same inpaint twice is the same hash, on all three |
| Per-image seeds in a batch | A batch of two: image 1 is the solo pass at its seed, image 2 the solo pass at seed + 1, on all three |
| Cancel | Cancelled at step 2 (2.3-2.8 s on SDXL, 12.4 s on Z-Anime); the next inpaint is the uncancelled hash |
| The text-to-image pipeline afterwards | The source's own request, run again after every inpaint and outpaint of the session, is the source's hash on all three |
| The composite keeps what is not masked | Every pixel where the mask is 0 is the source's exactly (largest difference 0, on all three) |
| The embedded mask | The `local-diffusion-mask` chunk is 3,836 bytes at 1 MP, 6,664 at 2.4 MP, 10,368 at 4.2 MP and 17,752 at 8.4 MP for a hard-edged mask; a 24 px feather makes the 1 MP one 35,592 bytes. A result PNG is 1.1-10 MB |
| What a strength keeps | On all three, 0.5 keeps the masked part's colour and shape and redraws its detail; 0.7 and 0.85 still keep the old colour (the scarf stays red), with more detail redrawn; only the library default draws what the new prompt asks (a blue scarf) |
| The seam | Judged by eye on 2x crops, not by number. A hard mask leaves a visible cut where the new content meets the old (a coat flap and a strand of hair end at the ellipse on WAI; a stepped edge on Z-Anime) - a content seam, not a colour shift. Feather 24 hides the cut and leaves a faint double image across the feather, old and new blended (red and blue mixed on Z-Anime) |

The library default is therefore the right starting strength on the three
checkpoints measured: lower strengths keep the masked part's colour, which
is what a user asking for a change does not want. These are one prompt and
one mask per checkpoint, looked at by eye: they establish that the pass
runs, reproduces and keeps every unmasked pixel, not a quality benchmark.

Z-Image Turbo, a LoRA through an inpaint, the seam by number, the live
preview through the blend and the dialog's start noise were measured
2026-10-24 and an inpaint over the Nunchaku Z-Image and FLUX.1 Krea dev
transformers under WSL 2026-10-25 (below). Not measured: for the deferred
families (FLUX.2 [klein], Qwen-Image), the measurements their phases name before they are
enabled (FLUX.1's are below). Evidence is under `%TEMP%/local-diffusion-inpaint-card/`:
`measure.py`, and per checkpoint its gallery, `results.jsonl` and the
`inpaint-*.jpg` and `outpaint-*.jpg` strips.

### The other SDXL profiles and the undistilled Z-Image (2026-10-20)

Measured 2026-10-20 with the same harness, the same way: fp8, model offload,
Windows, a throwaway gallery per checkpoint, each profile's own steps and
CFG, the same seed-12345 1024x1024 source, ellipse mask and scarf prompts,
the watchdog at 15,500 MiB (it never tripped). The desktop held
2,296-2,819 MiB. The SDXL four ran the whole base set in one process each;
Z-Image ran it in short foreground steps, one load per step. Only 1 MP
sources were run; the outpaint rows are the 1024x1024 source's canvases.

| Checkpoint | Run | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- | --- |
| Illustrious XL v2.0 | Inpaint, 1 MP | 27 of 28 | 8.4-8.6 s (0.23 s) | 3.57 / 4.55 GiB | 6,867-7,321 MiB |
| Illustrious XL v2.0 | Outpaint to 16:9, 1.9 MP | 27 of 28 | 15.2-15.4 s (0.42 s) | 3.59 / 4.59 GiB | 7,329-7,558 MiB |
| Illustrious XL v2.0 | Outpaint +25%, 1.6 MP | 27 of 28 | 13.0 s (0.35 s) | 3.58 / 4.64 GiB | 7,543 MiB |
| Kohaku XL Zeta | Inpaint, 1 MP | 27 of 28 | 8.8-9.8 s (0.23 s) | 3.28 / 3.63 GiB | 6,533-6,604 MiB |
| Kohaku XL Zeta | Outpaint to 16:9, 1.9 MP | 27 of 28 | 15.2-15.6 s (0.41 s) | 3.53 / 4.11 GiB | 7,010-7,179 MiB |
| Kohaku XL Zeta | Outpaint +25%, 1.6 MP | 27 of 28 | 13.3 s (0.35 s) | 3.46 / 3.99 GiB | 6,892 MiB |
| RouWei 0.8 | Inpaint, 1 MP | 27 of 28 | 8.3-8.6 s (0.23 s) | 3.57 / 4.55 GiB | 7,484-7,617 MiB |
| RouWei 0.8 | Outpaint to 16:9, 1.9 MP | 27 of 28 | 15.0-15.2 s (0.41 s) | 3.59 / 4.59 GiB | 7,570-7,660 MiB |
| RouWei 0.8 | Outpaint +25%, 1.6 MP | 27 of 28 | 13.0 s (0.35 s) | 3.58 / 4.64 GiB | 7,622 MiB |
| NoobAI-XL V-pred 1.0 | Inpaint, 1 MP | 27 of 28 | 8.3-8.7 s (0.23 s) | 3.57 / 4.55 GiB | 7,145-7,533 MiB |
| NoobAI-XL V-pred 1.0 | Outpaint to 16:9, 1.9 MP | 27 of 28 | 15.0-15.2 s (0.41 s) | 3.59 / 4.59 GiB | 7,563-7,752 MiB |
| NoobAI-XL V-pred 1.0 | Outpaint +25%, 1.6 MP | 27 of 28 | 12.9 s (0.34 s) | 3.58 / 4.64 GiB | 7,623 MiB |
| Z-Image (undistilled) | Inpaint, 1 MP | 50 of 50 | 94.3-97.1 s (1.79-1.82 s) | 6.93 / 8.61 GiB | 11,568-12,188 MiB |
| Z-Image (undistilled) | Outpaint to 16:9, 1.9 MP | 50 of 50 | 193.8-201.6 s (3.62-3.78 s) | 7.66 / 9.68 GiB | 12,624-13,024 MiB |
| Z-Image (undistilled) | Outpaint to 16:9, Batch 2 | 100 | 392.9 s | 7.66 / 9.68 GiB | 12,708 MiB |
| Z-Image (undistilled) | Outpaint +25%, 1.6 MP | 50 of 50 | 167.6 s (3.16 s) | 7.46 / 9.26 GiB | 12,369 MiB |

A Batch 2 inpaint on the SDXL four took 16.5-17.7 s and an outpaint Batch 2
29.4-29.8 s, at the solo passes' allocation. The four SDXL profiles cost what
Animagine and WAI cost, to a few hundredths of a gigabyte; Z-Image costs what
Z-Anime does, with 2.5-3 GB of the card to spare at 1.9 MP under this
smaller desktop.

| What | Result |
| --- | --- |
| Determinism, batches, cancel | On all five: the same inpaint and the same outpaint twice is the same hash; a Batch 2 is the solo passes at its two seeds; cancelled at step 2, the next pass is the uncancelled hash |
| The composite keeps what is not masked | Largest difference outside the mask 0, on all five |
| The text-to-image pipeline afterwards | The source's own request is the source's hash, on all five |
| The embedded mask | The same sizes as on 2026-10-17: 3,836 bytes hard-edged at 1 MP, 35,592 feathered, 5,104 for 16:9 and 5,968 for +25% |
| What a strength keeps | By eye. As on 2026-10-17, 0.5 and 0.7 keep the red scarf on all five. At 0.85 Illustrious turns a glove blue and keeps the scarf red, RouWei adds blue at the collar, and Z-Image already draws the blue knitted scarf. The library default draws a blue scarf on all five; Illustrious's is striped through with the old red |
| NoobAI's v-prediction through the inpaint class | Its inpaints render like the epsilon profiles', so the class carries the v-prediction scheduler settings. Its default-strength outpaint of a dark night scene drew bright, cyan, washed-out scenery at the sides, a visible mismatch in tone with the source; seen on this one source |
| The band, 0 / 32 / 64 / 128 px | By eye on Illustrious and Z-Image only: 0 leaves a hard cut, 32 joins, 128 leaves a double image (ghosted trunks on Z-Image), as on 2026-10-17 |

Not measured here: sources above 1 MP on these five, and the outpaint strips
of Kohaku, RouWei and NoobAI looked at for the band. Evidence is in
`%TEMP%/local-diffusion-inpaint-card/` beside the 2026-10-17 runs
(`illustrious-xl-v2-0`, `kohaku-xl-zeta`, `rouwei-0-8`,
`noobai-xl-vpred-1-0`, `z-image`; `rouwei.log`, `noobai.log`, `zimage.log`).

### FLUX.1 Krea dev, before enabling (INP#3)

Measured 2026-10-19 the same way, with the family still deferred: the
harness (`measure_flux.py` beside `measure.py`) lifted `INPAINT_DEFERRED`
and the profile's `inpaint_supported` in its own process only, nothing on
disk. fp8, model offload, Windows, CFG 3.5, 28 steps; the desktop held
2,056-2,430 MiB. The watchdog stood at 16,000 MiB here, since the plain
1 MP render alone read 15,634 MiB with this desktop, and a second guard
cancelled on two consecutive steps over 3 s (the spill's sign); neither
tripped. Source: a photograph of a woman in a red scarf, seed 12345,
1024x1024; the same ellipse mask; the inpaint asks for a blue knitted scarf.

| Run | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- |
| Text-to-image source, 1 MP | 28 | 70.6 s, the first of a process (1.42 s); 43.8 s run again | 12.00 / 12.62 GiB | 15,540-15,634 MiB |
| Inpaint, library default 0.6 | 17 of 28 | 29.7-38.4 s warm (1.31-1.37 s) | 12.16 / 12.83 GiB | 15,419-15,667 MiB |
| Inpaint at 0.8 / 0.95 / 1.0 | 23 / 27 / 28 | 49.2 / 45.8 / 44.7 s (1.35-1.45 s) | 12.16 / 12.83 GiB | 15,448-15,601 MiB |
| Inpaint, Batch 2 | 34 | 82.5 s | 12.16 / 12.83 GiB | 15,564 MiB |

An inpaint costs 0.16 GiB allocated over the text-to-image render, as the
refine pass did; at 1 MP it does not spill, so FLUX.1 does not need
`padding_mask_crop` to fit there. The card has no room for more: a larger
source was not run.

| What | Result |
| --- | --- |
| The class over the loaded components | `FluxInpaintPipeline` ran from `refine.build` over the fp8, offload-hooked components with no second load; the allocator back to 0.03 GiB after every pass |
| Determinism | The same inpaint twice is the same hash |
| Per-image seeds in a batch | Image 1 is the solo pass at its seed, image 2 the solo pass at seed + 1 (the batch runs its images one after another) |
| Cancel | Cancelled at step 2 (19.4 s, the first step carrying the encodes); the next inpaint is the uncancelled hash |
| The text-to-image pipeline afterwards | The source's own request after every inpaint is the source's hash |
| The composite keeps what is not masked | Largest difference outside the mask 0 |
| The embedded mask | 3,836 bytes hard-edged, 35,592 with a 24 px feather, as on SDXL; a result PNG 1.47-1.50 MB |
| The live preview | Fires through the blend (16-20 frames a pass); mid-pass it shows the new scarf forming inside a still-noisy frame, at the end the finished picture |
| What a strength keeps | The library default, 0.6, keeps the red scarf and adds a little blue at the collar; 0.8 the same with more of it; 0.95 and 1.0 draw the blue knitted scarf asked for, with the red scarf's ends below the mask kept. Unlike SDXL and Z-Image, FLUX.1's default is not a start that does what a user asking for a change wants |
| The seam | By eye on 2x crops: a hard mask leaves a visible cut where redrawn knit meets the old; feather 24 hides it |

Enabled on these results 2026-10-19 (TODO INP#3): the three FLUX.1
profiles start an inpaint at 0.95, their own start over the library's
0.6, and the Strength note says it was measured here, on Krea dev.

The [2026-11-18 follow-up](#windows-follow-up-measurements-2026-11-18)
measures dev/schnell execution at the existing 0.95 recipe, a dev LoRA
and source/result seam numbers under native fp8/model offload. This is not
a comparative strength sweep establishing a new preferred opening. The
Krea outpaint was
measured 2026-10-21 (Outpaint, "FLUX.1 Krea dev"), and the class over the
Nunchaku Krea dev transformer, with its seam by number, 2026-10-25 ("Under
Nunchaku", below).

### Z-Image Turbo, an SDXL LoRA, the seam by number (2026-10-24)

The same harness (`measure_light.py` beside `measure.py`, with the repo
path brought up to date, a RAM guard that would have ended the process
below 3 GB available, and `lora` and `seam` phases), fp8, model offload,
Windows, a throwaway gallery per run, the same seed-12345 1024x1024 source,
ellipse mask and scarf prompts, the watchdog at 15,500 MiB (it never
tripped). The desktop held 2.4-2.6 GB; about 28 GB of the 64 GB of RAM was
free before each load, beside other work.

Z-Image Turbo (now cached), the whole base set in one process, at the
class's default 1.0 and its 8 steps:

| Run | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- |
| Text-to-image source, 1 MP | 8 | 18.5 s, the first of the process (0.88 s); 10.7 s run again | 6.43 / 7.17 GiB | 10,094-10,156 MiB |
| Inpaint, 1 MP | 8 of 8 | 11.2-12.4 s (0.88-0.90 s) | 6.59 / 7.33 GiB | 10,248-10,533 MiB |
| Inpaint, Batch 2 | 16 | 23.3 s | 6.59 / 7.33 GiB | 10,389 MiB |
| Outpaint to 16:9, 1824x1024, 1.9 MP | 8 of 8 | 19.7-20.1 s (1.74-1.78 s) | 6.82 / 7.50 GiB | 10,448-10,556 MiB |
| Outpaint to 16:9, Batch 2 | 16 | 39.0 s | 6.82 / 7.50 GiB | 10,547 MiB |
| Outpaint +25%, 1280x1280, 1.6 MP | 8 of 8 | 17.6 s (1.52 s) | 6.72 / 7.53 GiB | 10,556 MiB |

The load took 42-49 s and peaked at 26.4-26.6 GB of the process's RAM
(17.8 GB on a third load, the files then in the page cache), leaving 5.7 GB
of the machine's free at the lowest: Z-Image Turbo's load is not light on
RAM, though its card cost is the undistilled Z-Image's less 1-2 GB.

| What | Result |
| --- | --- |
| Determinism, batches, cancel | The same inpaint and the same outpaint twice is the same hash; a Batch 2 is the solo passes at seeds 12345 and 12346; cancelled at step 2, the next pass is the uncancelled hash |
| The composite keeps what is not masked | Largest difference outside the mask 0 |
| The text-to-image pipeline afterwards | The source's own request is the source's hash (`02366ad0962cb95e`) |
| The embedded mask | 3,836 bytes hard-edged, 35,592 feathered, 5,104 for 16:9 and 5,968 for +25%, as on the others |
| What a strength keeps | By eye. 0.5 and 0.7 keep the red scarf; at 0.85 a dark navy knitted scarf, as the undistilled Z-Image already drew at 0.85; the default 1.0 the blue knitted scarf asked for. The library default stays the right start |
| Outpaint fill and band | By eye. 0.6 and 0.8 continue the scene softly, the default draws new trees and a fence. Band 0 leaves a visible vertical line at the original's edge, 32 joins, 64 and 128 leave a translucent second figure at the right side (a ghost of the girl), as the others' double image |

The real server and page, Z-Image Turbo (fp8, model offload, a scratch
gallery on port 7860, headless Edge at 1400x900, `tools/browser.mjs`
steps; evidence in `zturbo-page/`):

- The Inpaint dialog's line, after a brush stroke: "1 image · strength 1 ·
  starts at 100% noise · about 8 of 8 steps run · keeps nothing of the
  masked part"; at 0.7 "starts at 90% noise · about 6 of 8 steps run"; at
  0.5 "starts at 75% noise · about 4 of 8 steps run". Those are the shift-3
  schedule's sigma at the first step each pass runs (3t/(1+2t) at t = 0.75
  and 0.5). The Strength note names Z-Anime as where 1.0 was measured;
  that is still the note's wording.
- The live preview through the blend: frames at steps 2-7 of 8, 0.9 s
  apart, in the strip's tile ("Live preview · step 3/8"). The whole frame
  is noisy, the unmasked part included, since the class re-noises the
  source there to the step's level; by step 5 the source's composition
  is plain, and the finished image replaced the tile in the same box.
- Found there and fixed the same day: the line under the strip called the
  inpaint "Refined from #1 at 1", where the record's facts say "Inpainted
  from"; it now names an inpaint or outpaint as the facts row does
  (`viewer.js`), checked in smoke-inpaint and smoke-outpaint.

The seam by number. For each pass, the mean step in 8-bit levels between
neighbouring pixels across the mask's edge (where it goes from 0 to above
0) in three pictures: the source, the source composited over its own VAE
round trip the same way (what an encode and decode alone cost there), and
the result. For a soft mask, also the mean gradient inside the soft region.

| Checkpoint | Pass | Step: source / round trip / result | Soft region gradient: source / round trip / result |
| --- | --- | --- | --- |
| Animagine XL 4.0 | Inpaint, hard | 10.1 / 10.5 / 41.7 | - |
| Animagine XL 4.0 | Inpaint, feather 24 | 10.3 / 10.5 / 10.5 | 20.0 / 19.4 / 19.7 |
| Animagine XL 4.0 | Outpaint 16:9, band 0 | 18.4 / 17.5 / 22.9 | - |
| Animagine XL 4.0 | Outpaint 16:9, band 32 | 9.9 / 10.0 / 9.9 | 8.9 / 8.7 / 8.8 |
| Z-Image Turbo | Inpaint, hard | 2.8 / 4.1 / 26.6 | - |
| Z-Image Turbo | Inpaint, feather 24 | 3.7 / 3.9 / 4.0 | 5.3 / 5.4 / 7.1 |
| Z-Image Turbo | Outpaint 16:9, band 0 | 6.6 / 8.0 / 8.4 | - |
| Z-Image Turbo | Outpaint 16:9, band 32 | 1.1 / 1.2 / 1.2 | 2.1 / 2.3 / 2.5 |

A hard mask's cut is the new content's, not the VAE's: 4 to 10 times the
source's step, where the round trip adds at most 1.4 levels. A feather or
a band brings the edge to the round trip's level. The number cannot see the
double image a wide feather leaves (by eye above); Z-Image Turbo's 35%
higher gradient inside the feather is the nearest it comes. The whole
picture's round trip differs from the source by 2.7 levels on average on
Animagine and 2.0 on Z-Image Turbo.

Band widths between 0 and 64, outpaint to 16:9 at the default strength:

| Band | Animagine: step / soft gradient | Z-Image Turbo: step / soft gradient |
| --- | --- | --- |
| 0 | 22.9 against the source's 18.4 / - | 8.4 against 6.6 / - |
| 16 | 5.1 against 4.8 / 8.7 against 7.7 | 1.4 against 0.9 / 3.1 against 1.9 |
| 24 | 5.9 against 5.9 / 8.4 against 7.9 | 1.3 against 1.0 / 2.7 against 2.0 |
| 32 | 9.9 against 9.9 / 8.8 against 8.9 | 1.2 against 1.1 / 2.5 against 2.1 |
| 48 | 6.5 against 6.4 / 8.6 against 9.6 | 1.3 against 0.9 / 2.4 against 2.2 |
| 64 | 5.1 against 4.9 / 8.8 against 10.4 | 1.4 against 1.0 / 2.3 against 2.2 |

(The source column differs per band because the edge the mask reaches
moves into the original as the band widens.) By eye on 3x crops: band 0
cuts on both; on Animagine 16, 24 and 32 join, a faint ghost starts at
32, and 48 and 64 show plain double branches; Z-Image Turbo's edge here is
plain sky and shows nothing past band 0. Narrow bands keep a little more
gradient than the source, wide ones less (the blend softens it). So 16-24
join as well as 32 with less ghosting on this one Animagine source; one
source does not move the default. Evidence: `zturbo.log`,
`zturbo-seam.log`, `animagine-seam.log`, `bands.log`, and
`z-image-turbo/`, `*-bands/outpaint-band-seams-fine.jpg`, in
`%TEMP%/local-diffusion-inpaint-card/`.

### Under Nunchaku, WSL (2026-10-25)

The inpaint class over the lite-infer NVFP4 r32 transformers with their
bnb4 text encoders (`lite-infer/z-image-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`
and `lite-infer/flux.1-krea-dev-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`,
ungated), WSL 2 (`wsl/env.sh`; torch 2.13.0+cu130, diffusers 0.40.0,
kernels 0.16.1, bitsandbytes 0.50.2), quantization `nunchaku`, model
offload, one process and one load per checkpoint (`card3_inpaint.py`), a
scratch gallery on ext4. The same seed-12345 1024x1024 sources, ellipse
mask and scarf prompts as above (`measure_light.py`'s for Z-Image Turbo,
`measure_flux.py`'s for Krea dev); the 1536 source is a refine pass of the
1 MP one. A Windows watcher (shared memory over baseline +300 MiB, card
15,500 MiB, host RAM 4 GiB) and the harness's own card, slow-step and WSL
RAM guards; none fired. "Card" is WSL nvidia-smi's whole card at 500 ms,
the desktop's 411-435 MiB included.

| Checkpoint | Run | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- | --- |
| Z-Image Turbo lite | Source, 1 MP, the first of the process | 8 of 8 | 12.4 s (0.34 s; the first step 6.7 s) | 4.09 / 4.93 GiB | 5,814 MiB |
| Z-Image Turbo lite | Inpaint at 1.0, 1 MP | 8 of 8 | 6.8 s; 4.7 s again (0.34 s) | 4.35 / 5.10 GiB | 5,990 MiB |
| Z-Image Turbo lite | Inpaint, Batch 2 | 16 | 9.3 s | 4.35 / 5.10 GiB | 5,990 MiB |
| Z-Image Turbo lite | Outpaint to 16:9, 1824x1024, 1.9 MP | 8 of 8 | 9.8 s (0.73 s) | 4.62 / 5.28 GiB | 6,182 MiB |
| Z-Image Turbo lite | Inpaint at 1.0, 1536x1536, 2.4 MP | 8 of 8 | 11.4 s (1.01 s) | 4.84 / 6.28 GiB | 7,206 MiB |
| FLUX.1 Krea dev lite | Source, 1 MP, the first of the process | 28 of 28 | 31.7 s (0.51 s; the first step 12.3 s); 22.3 s again | 7.38 / 8.10 GiB | 9,076-9,386 MiB |
| FLUX.1 Krea dev lite | Inpaint at 0.95, 1 MP | 27 of 28 | 20.7 s; 19.9 s again (0.50 s) | 7.55 / 8.28 GiB | 9,252 MiB |
| FLUX.1 Krea dev lite | Inpaint, Batch 2 | 54 | 36.0 s | 7.55 / 8.28 GiB | 9,252 MiB |
| FLUX.1 Krea dev lite | Inpaint at 0.6 / 0.8 / 1.0 | 17 / 23 / 28 | 11.9 / 14.9 / 19.2 s (0.51 s) | 7.55 / 8.28 GiB | 9,252 MiB |
| FLUX.1 Krea dev lite | Outpaint to 16:9, 1.9 MP, at 0.95 | 27 of 28 | 35.3 s (1.04 s) | 7.87 / 8.99 GiB | 10,019 MiB |
| FLUX.1 Krea dev lite | Inpaint at 0.95, 1536x1536, 2.4 MP | 27 of 28 | 41.2 s (1.35 s) | 8.06 / 9.41 GiB | 10,935 MiB |

Loads took 12.8 s (Z-Image Turbo) and 15.3 s (Krea dev). Against fp8 on
Windows, the inpaint at 1 MP costs about 2.2 GiB less allocated on Z-Image
Turbo (6.59 / 7.33 there) and 4.6 GiB less on Krea dev (12.16 / 12.83,
whose 0.95 pass took 45.8 s against 20.7 here); Krea dev's 2.4 MP inpaint
runs with about 5,370 MiB of the card to spare (10,935 of 16,303 MiB),
where the fp8 refine pass at that size reached 15,278 MiB. The 1536
refine source on Krea dev lite peaked at 8.05 / 9.41 GiB, the 2026-10-05
Nunchaku FLUX.1 dev pass's figure.

| What | Result |
| --- | --- |
| The class over the Nunchaku components | `ZImageInpaintPipeline` and `FluxInpaintPipeline` ran from `refine.build` over the loaded Nunchaku transformer and bnb4 encoder, with no error, out-of-memory or refusal |
| The composite keeps what is not masked | Largest difference outside the mask 0 on both (mean inside 48.0 and 49.9 levels) |
| Determinism | Not bit-for-bit within the process, text-to-image as well (Behaviour, the Nunchaku row): a repeat, a batch's images against their solo passes, the pass after a cancel and the text-to-image afterwards each hash anew, a repeat staying closer to the first than seed + 1 does |
| Cancel | Cancelled at step 2 (2.5 s on Z-Image Turbo, 4.0 s on Krea dev); the next pass completed; text-to-image ran afterwards |
| The embedded mask | 3,928 bytes hard-edged, 36,108 with a 24 px feather, 5,092 for 16:9 and 6,828 at 2.4 MP, on both |
| The live preview | 3 frames an 8-step pass on Z-Image Turbo (6 for a batch and at 1.9-2.4 MP), 20 a 27-28-step pass on Krea dev (40 for a batch, 15-17 at 1.9-2.4 MP) |
| What a strength keeps | By eye. Z-Image Turbo: 0.5 and 0.7 keep the red scarf, 0.85 a navy knit, 1.0 the blue knitted scarf asked for. Krea dev: 0.6 and 0.8 keep the red scarf with a little blue at the collar, 0.95 and 1.0 the blue knitted cowl - as on fp8 |
| The picture | Both put a blue knitted scarf or cowl in the ellipse, lit to match. Krea dev at 0.95 and 1.0 also redraws a pale neck or collar patch under the chin, a visible second chin line at full size. Both outpaints extend the scene naturally with no seam showing at the source's borders (Z-Image Turbo added a red flag in the new margin); both 2.4 MP inpaints are clean |
| The seam, by eye on 2x crops | A hard mask is a visible content cut on both; feather 24 hides it and leaves a faint ghost (a blue haze on the chin on Z-Image Turbo, a doubled jaw line on Krea dev) |

The seam by number, the same measure as above (the step across the
mask's edge in 8-bit levels: source / its VAE round trip / the result;
and the gradient inside a soft region):

| Checkpoint | Pass | Step: source / round trip / result | Soft region gradient |
| --- | --- | --- | --- |
| Z-Image Turbo lite | Inpaint, hard | 1.87 / 3.21 / 20.79 | - |
| Z-Image Turbo lite | Inpaint, feather 24 | 1.85 / 1.93 / 1.94 | 3.92 / 4.05 / 5.62 |
| FLUX.1 Krea dev lite | Inpaint, hard | 4.29 / 4.56 / 24.29 | - |
| FLUX.1 Krea dev lite | Inpaint, feather 24 | 4.03 / 4.12 / 4.12 | 8.91 / 8.79 / 9.29 |

The pattern fp8 Z-Image Turbo showed: a hard mask's cut is the new
content's, five to ten times the source's step, and a feather brings it to
the round trip's level. Shared GPU memory peaked 82 MiB over its baseline
on Z-Image Turbo and 138 MiB on Krea dev; free host RAM stayed at or above
17.6 GiB (18,002 MiB). Not run in this 2026-10-25 session: FLUX.1 dev and schnell lite
(measured in the [2026-11-18 follow-up](#flux1-dev-and-schnell-under-wsl-production-http-2026-11-18)), Krea 2 Turbo and ERNIE lite (`inpaint_supported` False),
Qwen-Image (deferred), cross-process reproduction (known to fail),
anything above 2.4 MP, and the real server and Inpaint dialog under WSL.
Evidence: `card3-wsl/out/` in the cards directory (`zturbo/` and
`kreadev/`: `results.jsonl`, `run.log`, PNGs, previews, the strips;
`watch-*.csv`), and `~/ld-card3/` in WSL.

### Real server and Inpaint dialog on Z-Image Turbo (2026-11-17)

Z-Image Turbo NVFP4 through the real WSL2 server and production Inpaint dialog, model offload, 1024x1024, 8 steps, CFG0, strength1, seed12345, Mu omitted and crop off. One honest synthetic RGB-grid gallery source and one browser-painted hard circular mask, about 20% of the canvas; one actual load at 9341e57. The cached model revision was f5fc49362b8a8b2901a6e7afe811b12bf5ee5eb2. An owned isolated KERNELS_CACHE supplied only the authoritatively verified v2 reference to cached 64b10f6b8968b27cadaabfb3f6d2395f2251c848; all four kernel files matched official digests, and the 196,137,920-byte binary matched SHA-256 d120280ecae32d40a2265103eac692d9be45b4d5493d39133f5bc14beab83f9c. The normal shared cache's missing refs/v2 was not repaired; no binary or weights download was needed.

All 156 browser steps and 67 assertions passed: first inpaint, page reload and Reuse of its persisted mask, same-seed repeat, Cancel at step 2/8 with zero saved images, then recovery. The three saved cells took 31.369, 8.316 and 5.414 s after a 15.744 s load. Peaks were 4.354 GiB allocated / 5.098 GiB reserved and 7,301 MiB whole-card. The same seed was not pixel-identical: mean full-image channel differences 0.80066 and 0.58532, maxima 116/127. Every one of 842,052 unmasked pixels was exactly the source's in all three results. SQL, PNG provenance, stored and embedded mask agreed. PNG-only recovery into disjoint scratch storage recovered two images and one mask with no unresolved references/masks; RGB and mask levels were exact, while reencoded PNG file bytes differed.

Actual challenged load admission saw 37.14 GiB free RAM / 41.78 GiB commit headroom; the matching grant arrived 0.266 s after the request and source rechecked the 1.5 s bound before loading. Windows minima were 18.32 GiB free / 17.89 GiB commit, adapter shared rise 282.18 MiB under the 300 MiB session guard. Linux owned RSS peaked 12.69 GiB under the 16 GiB cap, VmSwap stayed at zero and minimum Linux free RAM was 33.397 GiB. No guard tripped. Windows sampling mean/max gaps were 0.290/1.286 s; Linux 0.136/0.249 s. Explicit STOP signalled only recorded server PGID 610; cleanup and subsequent CPU checks proved the original guard/server absent and group empty. No service or distro termination command was issued.

The original driver's overall exit remains failed solely because support export included 292 MB of generated owned CUDA cache and crossed its 256 MiB copy cap. Supplemental CPU-only export succeeded after excluding only runtime-cache, with all omitted files inventoried by path/size/SHA256; original failure and tested sources remain unchanged. No GPU rerun occurred. This is one model's interface/compositing/storage result; the visible hard boundary on an artificial grid is not real-image seam/artistic-quality acceptance. The remaining dev/schnell slices are measured in the [2026-11-18 follow-up](#flux1-dev-and-schnell-under-wsl-production-http-2026-11-18).

[Consolidated receipt](maintainer/card3-z-turbo-wsl-2026-11-17.json).

### FLUX.1 dev and schnell under WSL, production HTTP (2026-11-18)

`CARD#3`, run with Matt's authorization for the outstanding tests. Windows host,
RTX 5080 (16,303 MiB), Ubuntu WSL2, torch 2.13.0+cu130, diffusers 0.42.1,
transformers 5.16.1, kernels 0.16.1 and bitsandbytes 0.50.2. The isolated runtime
matched all 317 backend/web files at `f8f4cf2`; the backend's raw SHA-256 was
`c3a85a23288be5d3ccb3bdb1b4751f5456c75d714685d9cf02f3e597c838274e`.
Each process loaded one checkpoint, with NVFP4 r32 Nunchaku Lite, bnb4 text
encoder, model offload and Eager/Native execution. Both complete checkpoints
were already cached: no weight, kernel or runtime download occurred.

The exact lite-infer selections were
`flux.1-dev-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` at
`9b4a4c430b36c3826e7a06be2c4860e1a66d1d09`, and
`flux.1-schnell-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` at
`ae27c859773999ff9f800c2ba14e0de5e1af4f50`. Model cache admission checked pinned
references, repository-contained paths and required filenames/byte lengths;
it did not rehash the model weight bytes. The existing owned kernel overlay
selected cached commit `64b10f6b8968b27cadaabfb3f6d2395f2251c848` through its own
`v2` reference. All four kernel files were rehashed against retained official
pinned metadata, including the 196,137,920-byte binary with SHA-256
`d120280ecae32d40a2265103eac692d9be45b4d5493d39133f5bc14beab83f9c`.
The shared kernel cache's missing `refs/v2` was not repaired.

The source was an honestly tagged synthetic 1024-square RGB grid, with full RGB
SHA-256 `a0a2fd00a5edd42b7c90c4dfd38bb038521e18f125634c196dc18d4da1bd95fc`;
no diffusion model made it. The frozen circular mask has an antialiased edge:
237 distinct L levels, 204,830 white pixels, 1,694 graded-edge pixels and
842,052 zero pixels (206,524 nonzero pixels in all),
hash `e0618042c1cdf90d6766d0e68050e1599b3ef92ff8788b97b64a9640cb3cfdb7`.
The prompt was “a blue ceramic bowl on a wooden table, soft natural light,
photograph”, seed 12345, no negative, Mu, true CFG, reference, LoRA or crop.
Dev used 28 steps, CFG 3.5 and strength 0.95 (27 callbacks); schnell used
4 steps, CFG 0 and strength 1 (4 callbacks).

| | dev | schnell |
| --- | --- | --- |
| Wrapped load | 15.19 s | 15.28 s |
| First / repeat / recovery saved cell | 45.867 / 19.187 / 17.490 s | 34.023 / 6.612 / 5.567 s |
| Allocated / reserved peak | 7.55 / 8.28 GiB | 7.53 / 8.26 GiB |
| Whole-card peak | 10,061 MiB | 10,034 MiB |
| Adapter shared rise | 118.68 MiB | 116.17 MiB |
| Host free-RAM / commit minima | 16.82 / 12.94 GiB | 17.10 / 13.20 GiB |
| Linux owned RSS peak / VmSwap | 16.56 GiB / 0 | 16.64 GiB / 0 |
| Cancel | observed at 2/27, zero saved images | observed at 2/4, zero saved images |

The real server's production HTTP protocol completed first/repeat/cancel/
recovery jobs, then the public Unload route and an unloaded status confirmation.
All 842,052 unmasked pixels were exactly the source's in every saved output.
SQL and PNG source/mask provenance, stored and embedded mask levels agreed.
Separate PNG-only recovery indexed two images and one mask with zero unresolved
references or masks. Incoming PNG bytes match the originals; after reindex the
PNG files differ because chunks may be re-encoded/reordered and empty provenance
fields added to the minimal source. Decoded RGB, mask levels and existing
provenance are preserved. This is production HTTP/core evidence; the previous
Z-Image Turbo session owns the real Inpaint-dialog evidence.

Same-seed output was **not** pixel-identical. Against the first result, dev's
repeat/recovery mean full-image channel differences were 3.13376/3.54370 levels
(maxima 222/237); schnell's were 6.33022/3.78661 (maxima 243/255). These are
measured outcomes, not fixture failures. No image was rerun to seek a matching
hash. The inspected first images show a blue ceramic bowl inside the frozen
circular mask against the preserved grid; dev also adds foliage. A content
boundary is visible against the artificial grid. This does not establish
real-image seam quality or artistic
acceptance.

Every actual load received a new challenged physical-host sample after the
challenge was observed, requiring at least 36 GiB RAM and commit. Source checked
the same grant immediately before loading within its 1.5-second Linux-clock
bound. Live limits stayed 15,500 MiB card, adapter shared rise 300 MiB, host RAM
and commit floors 4 GiB, Linux owned RSS 24 GiB and no swap growth. No limit
tripped. All 619 dev and 424 schnell Windows samples independently recompute to
`ok`. Windows mean/max sample gaps were 0.232002/0.289139 s and
0.231555/0.381376 s. Linux's separate 991/692 sample logs had mean/max gaps
0.120206/0.201515 s and 0.120400/0.201350 s. These are distinct monotonic clock
domains; no Windows timestamp was subtracted from a Linux timestamp.

Three infrastructure failures remain explicitly retained. Dev's first session
stopped while writing an uppercase receipt label after its cancel POST; its
partial results were exported, and the full session above followed a tested
label correction in fresh scratch helpers. Schnell's complete GPU/API run
passed, but its original CPU inspector stopped on an exit-time RSS/swap read.
The supplemental CPU guard tolerates missing fields only after proving that
exact PID/birth ended; a still-live unreadable subject remains fatal. This
terminal-skip rule is unit-tested only: the successful supplemental runs had
complete counters and did not exercise that branch. Original failure evidence
was kept. The original CPU child had already emitted a successful result before
its guard failed, and one supplemental CPU verify remeasured the same saved
images. The first supplemental recovery was then refused by its Linux ownership
preflight; its empty stdout caused the Windows collector's IndexError. Delayed
output visibility was not established as that failure's cause. A later collector
consumed the existing successful supplemental verify report, and CPU recovery
and export passed after exact post-shutdown ownership proof. No GPU run was
repeated for CPU supplementation. The bounded report observer is unit-tested
for delayed/truncated/missing/wrong reports, and the posthoc audit binds the
retained reports by nonce, checkpoint, mode and phase. Generated runtime caches were
omitted from support copies with complete path/size/SHA-256 inventories.

The [curated receipt](maintainer/card3-flux-wsl-2026-11-18.json) binds this scope; its source inventories, overlay
verification, actual load-admission frames and both raw guard logs are bound by
hash. Other dimensions/strengths, fp8/int4 and real-image artistic quality remain
unmeasured by this session.

### Only the masked area

The app's crop (`docs/maintainer/plan-inpainting.md`, phase 6) was built 2026-11-03.
Its first card measurements are the SDXL outpaint, Z-Anime and FLUX
sequential-offload slices below. The bounded
[2026-11-18 follow-up](#windows-follow-up-measurements-2026-11-18) adds Krea
fp8/model-offload crops on 2.36-8.39 MP CPU-resized sources, pads 0/32/64,
and independent Z-Image Turbo/undistilled Z-Image sessions. It does not
establish same-model native-large-source quality or a clean seam.

**SDXL cropped outpaint, 2026-11-11, with Matt's OK.** Animagine XL 4.0,
cached snapshot `2b7c1b397761bf5bd3cc42e5b39ec99314a75a96`, Windows 11,
RTX 5080 (16,303 MiB), driver 616.92, main at `7be054a`, Python 3.13,
torch 2.13.0+cu130, diffusers 0.42.1, torchao 0.18.0, accelerate 1.15.0.
One offline FP8/model-offload load through `main.ensure_loaded`, followed
by the normal backend `applied`/`generate`, outpaint stamping and storage
paths in a scratch gallery; no HTTP server or live-gallery writes.
Load took 21.4 s. A winter-forest/cabin/river illustration was rendered at
1024 square (seed 12345), then refined to 2048 square at strength 0.3.
Every outpaint used that source, seed 12346, batch 1, CFG 5, 28 requested
steps, the library's strength 0.9999 (27 steps run), and outpaint band 32.
No live previews. Times are synchronized `generate()` wall time, including
the app's fill/crop/composite, excluding load, PNG saving and CPU metrics.

| Outpaint on a 3648x2048 canvas (7.47 MP) | Crop working size | Time | Allocated / reserved GiB | Whole-card MiB | Seam step, result |
| --- | --- | --- | --- | --- | --- |
| +800 left/right, full canvas | 3648x2048 | 75.73 s | 5.20 / 7.02 | 10,210 | 2.68 |
| +800 left/right, crop pad 32 | 1376x784 | 10.87 s | 3.29 / 3.66 | 6,770 | 3.20 |
| Same crop and seed, repeated | 1376x784 | 10.83 s | 3.29 / 3.66 | 6,770 | 3.20 |
| +1600 right, full canvas | 3648x2048 | 75.63 s | 5.20 / 7.02 | 10,212 | 1.75 |
| +1600 right, crop pad 0 | 928x1168 | 10.93 s | 3.29 / 3.66 | 6,772 | 2.63 |
| +1600 right, crop pad 32 | 928x1152 | 10.32 s | 3.29 / 3.67 | 6,788 | 2.40 |
| +1600 right, crop pad 64 | 944x1136 | 10.84 s | 3.29 / 3.67 | 6,788 | 2.27 |

Symmetric padding paints both edges, so its bounding box is the whole
canvas: cropping here means denoising that canvas at about 1 MP and
resizing back. The one-sided boxes are `(2016,0,3648,2048)`,
`(1984,0,3648,2048)` and `(1952,0,3648,2048)` for crop pads 0/32/64.
The measured saving is about 7x in time and 48% in reserved VRAM for the
symmetric pair. These are individual samples, not a throughput benchmark.

Every output stayed 3648x2048, and the largest channel difference from the
padded source wherever the painted mask was zero was **0** on all seven
outpaints. Row and PNG assertions preserved source, mask, outpaint, crop
spec, computed box/working size, allocator peaks and `mask_blend: overlay`.
The repeated symmetric crop was pixel-identical, SHA-256
`ba1e62a2fb5ad6db78ac932a32689f63690980dcc90856cb4c9d5906b93aac5c`.
Seam step uses `tools/image_compare.py` at the mask's zero/positive edge;
the padded-source values were 2.56 symmetric and 1.57 one-sided. No VAE
round-trip comparison was made. These averages do not judge the join's
content or softness.

Visual inspection of whole images and native 1:1 join/detail crops: the
symmetric cropped extension has visibly softer snow, branches and fence
detail than the full pass. The one-sided crops change scene scale and
objects at the join; pad 0 has a particularly abrupt cut, while 32 and 64
also leave visible transitions. None establishes a clean join for this
scene. The lower seam average at pad 64 does not establish better overall
quality. The crop is a measured speed/memory trade-off here; the default
padding, working tier and canvas guard stay unchanged.

The independent `tools/card_guard.py` watcher kept its defaults: 15,500
MiB card, shared rises at most 1,024/512 MiB adapter/process, and 4 GiB
free RAM/commit floors; startup required 14 GiB free RAM and card use at
most 3,500 MiB. The supervisor also stopped above 14 GiB working set or
16 GiB private memory. Both exited 0; no trip or spill observed. Starting
card use was 2,719 MiB; peak 10,212 MiB, shared rise 82 MiB on both
counters, minimum free RAM 29.85 GiB and commit headroom 36.23 GiB.
The 0.25 s process sampler peaked at 9.56 GiB working set / 14.47 private,
including CPU image analysis. The card guard requested 0.5 s sampling,
but serial Windows counters took about 1.8 s per sample, so its sampled
peaks can miss transients. The nine-render session took 283.3 s and
unloaded before exiting; card use returned to 2,719 MiB with no worker
left. One scene and seed in one process do not qualify other checkpoints,
cropped inpainting, cross-process reproduction, cancellation or the page.

The [receipt](measurements/card13-sdxl-2026-11-11.json) retains exact
requests, hashes, metrics, guard summary and script/log digests. Full PNGs,
native inspection crops, scratch database, scripts and raw logs remain in
`%TEMP%/auradream-card13-sdxl-20261111/`.

**FLUX/Z-Anime follow-up stops, 2026-11-11.** Matt requested the remaining
`CARD#13` checks and one accumulated PR. Both checkpoints were cached;
the scratch harness prepared no downloads or live-gallery writes. The
planned CPU source fixtures were Lanczos resizes of the owned SDXL source,
explicitly indexed as `fixture/cpu-lanczos-resize` with unknown duration,
not images generated at those sizes.

Z-Anime's first offline FP8/model-offload load stopped at the supervisor's
24 GiB working-set ceiling: 24.12 GiB sampled working set and 29.51 GiB
private. It had not rendered or prepared source fixtures. The independent
card guard observed at least 17.66 GiB free RAM and 22.46 GiB commit
headroom, and no rise in shared GPU memory. The supervisor killed the
owned process tree; the guard exited 0 after that tree exited. This is a
controlled stop, not a passing card qualification. Its
[receipt](measurements/card13-z-anime-load-stop-2026-11-11.json) retains the
process limits, peaks, exit codes and log digests.

FLUX was refused before worker admission or any model load. Its prior
inpaint reservation, 12.83 GiB, plus the current 2,719 MiB desktop projects
15,857 MiB, above the unchanged 15,500 MiB card guard. Reservation is not
a simultaneous residency measurement, but the earlier actual card peaks
also leave little margin here. The session therefore required desktop use
at most 2,100 MiB; the guard's startup refusal exited 7. The
[preflight receipt](measurements/card13-flux-preflight-2026-11-11.json)
records that refusal; it is not evidence that the cropped pass fits or
fails. Both process trees exited and the card returned to its 2,719 MiB
idle baseline. `CARD#13` remained open after this attempt; its bounded
model-offload follow-up is recorded below for 2026-11-18. Raw evidence is in
`%TEMP%/auradream-card13-followup-20261111/`, each attempt in its own
directory. A larger Z-Anime RAM allowance needs a separately stated
envelope; FLUX model offload needs a lower desktop baseline.

**Z-Anime after the encoder ownership fix, 2026-11-11.** The same
quantized direct encoder load now releases plain CPU shard views with
`own_tensors` before loading the denoiser (commit `b3d41fa`). The fresh
offline load completed in 19.21 s under the original 24 GiB working-set /
32 GiB private ceilings. Its load peaks were 20.34 / 24.04 GiB, and the
whole session peaked at 20.34 / 25.66 GiB including CPU pixel analysis.
The earlier load stopped before completing, so this is a successful
bounded retry, not a controlled before/after peak benchmark. No larger
RAM envelope was used.

Z-Anime Base (`SeeSee21/Z-Anime`, cached snapshot `0f5fb514`), FP8/model
offload, diffusers 0.42.1, 40 steps/CFG 4, default inpaint strength 1.0,
batch 1, seed 12346, no previews: a 2048-square (4.19 MP) source with the
feathered cabin-region mask ran at crop pad 32. Its box was
`(304,592,1808,1744)`, working size 1184x912. Times were 76.72 s and
75.69 s for the exact repeat, each at 6.97 / 8.66 GiB allocated/reserved
and 11,896 MiB whole-card. The maximum difference outside the mask was
0, row/PNG metadata checks passed, and both output pixel hashes were
identical (the full hash is in the [receipt](measurements/card13-z-anime-2026-11-11.json)).
Seam step was 3.70 on the source / 3.93 on the result; no VAE round-trip
comparison. Visually the painted region contains the requested blue
cabin, but the changed perspective, background and scene scale form a
visible rectangular patch against the original. This proves crop
execution and preservation, not a clean join or a quality recommendation.

The default card/shared/RAM/commit guard limits held; minimum free RAM
20.80 GiB, commit headroom 27.77 GiB, shared rise 82 MiB adapter/process.
The supervisor and independent card guard both exited 0. One load,
two renders, 183.3 s total; unloaded and process tree exited. Method,
software, source/mask geometry and script/log digests are in the receipt;
raw evidence and the native comparison board are in
`%TEMP%/auradream-card13-followup-20261111/z-anime-fixed/`.

**FLUX.1 Krea dev with sequential offload, 2026-11-11.** Model offload's
preflight remained held above. The already supported Eager FP8/sequential
path was measured instead, with the same 32 GiB working-set / 42 GiB
private envelope planned for the FLUX load, 4 GiB system floors and the
unchanged card/shared limits. Startup required 34 GiB free RAM and at
most 3,500 MiB desktop use for this lower-VRAM path. One offline load of
`black-forest-labs/FLUX.1-Krea-dev` (cached snapshot `8162a9c7`) through
`main.ensure_loaded`, commit `b3d41fa`, diffusers 0.42.1; then six cropped
inpaints through the normal backend/storage paths. Load took 39.43 s;
process peaks were 28.09 GiB working set / 35.04 private over the session.

The fixtures and feathered mask used the same owned SDXL source as the
Z-Anime run. These are CPU resizes, not FLUX renders at those source
sizes. Every cell used seed 12346, 28 steps/CFG 3.5, the measured FLUX
inpaint strength 0.95 (27 steps run), batch 1 and no live previews. Normal
`applied()` omitted the negative because default True CFG did not activate it.

| Source / crop padding | Working size | Time | Allocated / reserved GiB | Sampled whole-card MiB | Seam step, result |
| --- | --- | --- | --- | --- | --- |
| 1536 square (2.36 MP), 32 | 1184x912 | 102.14 s | 2.16 / 2.29 | 3,988 | 4.27 |
| 2048 square (4.19 MP), 32 | 1184x912 | 100.60 s | 2.16 / 2.29 | 3,992 | 3.86 |
| 2048 square, 0 | 1200x896 | 101.55 s | 2.12 / 2.25 | 3,988 | 3.90 |
| 2048 square, 64 | 1184x912 | 100.18 s | 2.16 / 2.29 | 4,252 | 3.87 |
| 2048 square, 32 repeated | 1184x912 | 101.02 s | 2.16 / 2.29 | 3,989 | 3.86 |
| 2896 square (8.39 MP), 32 | 1184x912 | 100.61 s | 2.16 / 2.29 | 3,989 | 3.02 |

All output dimensions matched their full-sized sources; largest difference
where the painted mask was zero was 0 on every cell. Crop geometry,
source/mask/spec, row/PNG reproduction fields and peak pairs passed their
assertions. The 4.19 MP pad-32 repeat was pixel-identical. Exact boxes,
requests, hashes and metrics are in the
[receipt](measurements/card13-flux-sequential-2026-11-11.json).
Source seam steps were 4.12 / 3.70 / 2.85 at the three source sizes; these
are the mask-edge averages, not a perceptual score or a VAE round-trip
comparison. Inspected overviews and native join crops of all three
paddings draw the requested blue cabin, but the repaint changes the
background, style and scene scale within a visible patch. No clean-join
acceptance or preferred padding follows from this one cross-checkpoint
fixture/prompt/seed. This is not a same-model-source quality comparison.

Both watchers exited 0: no trip or spill observed. Minimum free RAM was
13.31 GiB, commit headroom 17.82 GiB; shared rise 87.9 MiB adapter and
82 MiB process. The card guard sampled a 4,252 MiB session peak and a
28.32 GiB process-tree working-set peak; its roughly 1.8 s cadence can
miss transients, as in the SDXL session. The six-render session took
662.1 s, unloaded, exited and returned the card to about 2,720 MiB. The
normal model-offload memory/fit measurement was left to `CARD#13`; the
bounded 2026-11-18 follow-up now measures it independently. These sequential
numbers alone could not stand in for it. Defaults, crop rule/tier,
size guards and public performance-recipe holds are unchanged. Raw
evidence and the native comparison board remain in
`%TEMP%/auradream-card13-followup-20261111/flux-sequential/`.

## Outpaint

Measured 2026-10-17 in the same runs, the same way, from the same
1024x1024 sources (the 4.2 and 7.5 MP canvases from the 1536 and 2048
refine sources), at the library default strength and the app's band of
32 px unless a row says otherwise.

| Checkpoint | Canvas | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- | --- |
| WAI v17 | 1024x1024 to 16:9, 1824x1024, 1.9 MP | 27 of 28 | 15.4-18.6 s (0.42 s) | 3.59 / 4.59 GiB | 8,159-8,255 MiB |
| WAI v17 | 1024x1024 +25%, 1280x1280, 1.6 MP | 27 of 28 | 13.3 s (0.36 s) | 3.58 / 4.64 GiB | 8,250 MiB |
| WAI v17 | 1536x1536 to 16:9, 2736x1536, 4.2 MP | 27 of 28 | 40.4 s (1.15 s) | 4.22 / 5.32 GiB | 9,248 MiB |
| WAI v17 | 2048x2048 to 16:9, 3648x2048, 7.5 MP | 27 of 28 | 82.6 s (2.54 s) | 5.20 / 7.02 GiB | 11,018 MiB |
| Animagine XL 4.0 | 1024x1024 to 16:9, 1.9 MP | 27 of 28 | 14.1-14.4 s (0.40 s) | 3.53 / 4.11 GiB | 7,961-8,033 MiB |
| Animagine XL 4.0 | 1024x1024 +25%, 1.6 MP | 27 of 28 | 12.8 s (0.34 s) | 3.46 / 4.00 GiB | 7,879 MiB |
| Animagine XL 4.0 | 2048x2048 to 16:9, 7.5 MP | 27 of 28 | 89.0 s (2.49 s) | 5.20 / 7.02 GiB | 11,023 MiB |
| Z-Anime | 1024x1024 to 16:9, 1.9 MP | 40 of 40 | 165.9-178.3 s (3.77-3.82 s) | 7.66 / 9.68 GiB | 13,852-15,108 MiB |
| Z-Anime | 1024x1024 to 16:9, Batch 2 | 80 | 343.2 s | 7.66 / 9.68 GiB | 15,156 MiB |
| Z-Anime | 1024x1024 +25%, 1.6 MP | 40 of 40 | 143.3 s (3.21 s) | 7.46 / 9.26 GiB | 14,102 MiB |

An outpaint costs an inpaint of its canvas: the SDXL pair run the largest
16:9 canvas a 2 MP source allows within the guard with a third of the card
to spare. Z-Anime at 1.9 MP sits within half a gigabyte of the card's top
with the desktop's 3.6-4.6 GB beside it; larger canvases were not run, its
2.4 MP inpaint having filled the card.

| What | Result |
| --- | --- |
| Determinism | The same outpaint twice is the same hash; a batch of two is the solo passes at its two seeds; cancelled at step 2, the next outpaint is the uncancelled hash; on all three |
| What the fill contributes | At the library default the new area is drawn afresh and the fill does not show. At 0.8 it shows as soft, out-of-focus scenery; at 0.6 as flat grey-blue blocks continuing the edge (both SDXL checkpoints and Z-Anime) |
| The band, 0 / 32 / 64 / 128 px | Judged by eye on 2x crops at the original's edge, on all three. 0 leaves a hard vertical cut where the new area meets the original (trees and snowbanks stop at it); 32 joins (a trunk runs across the edge); 64 begins a faint double image across the band; 128 a plain one (half-transparent lamps and trunks), the original and the new picture blended over the band's width. 32 joins best of the four |
| The embedded mask | 5,104 bytes for 16:9 from 1 MP, 5,968 for +25%, 10,132 at 4.2 MP and 15,508 at 7.5 MP; band width changes it by under 20 bytes |
| The fill and the mask on the CPU | `outpaint.padded` 1.8-2.4 s and `outpaint.mask` 0.01-0.03 s at 6.3-8.4 MP canvases (+25%, 2:3 and 16:9 from 2048-px sources, and 2912x2896); the mask's PNG 10-15 KB |

So the band's starting 32 px is measured as the best of the four widths
tried, and the library default as the strength to start an outpaint at.
One source and one prompt per checkpoint, looked at by eye.

The other SDXL profiles and the undistilled Z-Image were measured
2026-10-20 (Inpaint, "The other SDXL profiles and the undistilled Z-Image").

### FLUX.1 Krea dev (2026-10-21)

fp8, model offload, Windows, one load for every step (RAM 52 GB free
after a reboot; a load peaks at 40-50 GB). The 1024x1024 source is the
inpaint's, seed 12345, 28 steps, the library default strength 0.95 and
the 32 px band unless a row says otherwise. The desktop held 1.0 GB.

| Canvas | Steps run | Time (per step) | Allocated / reserved | Card |
| --- | --- | --- | --- | --- |
| 1024x1024 to 16:9, 1824x1024, 1.9 MP | 27 of 28 | 71.9-77.5 s (2.41 s) | 12.40 / 13.18 GiB | 14,783-14,792 MiB |
| 1024x1024 to 16:9, Batch 2 | 54 | 143.4 s (2.53 s) | 12.40 / 13.18 GiB | 14,783 MiB |
| 1024x1024 +25%, 1280x1280, 1.6 MP | 27 of 28 | 63.1 s (2.09 s) | 12.33 / 13.06 GiB | 14,676 MiB |
| 16:9 at 0.8 / 0.6 | 23 / 17 of 28 | 62.2 / 47.8 s | 12.40 / 13.18 GiB | 14,792 MiB |

- Determinism: the same outpaint twice is the same hash
  (`57de20cc7ebbbce…`); a batch of two is the solo passes at seeds 12345
  and 12346; cancelled at step 2, the next outpaint is the uncancelled
  hash; text-to-image afterwards is the source's hash (`cb436fdef96e3bf4`).
- The seams: at the default, neither 16:9 seam shows at full resolution;
  trunks, snowfall and light run straight across both edges. So FLUX.1
  Krea dev's ordinary inpaint outpaints cleanly, which closes FLUX.1 Fill
  [dev] by the condition set for it (plan-inpainting question 7).
- The fill: 0.6 leaves blurred brown-grey smudges for the new sides; 0.8
  joins, with the scene redrawn more softly; the default draws new trees.
- The band: as on the others. 0 and 32 look alike at this source's edge,
  64 starts a faint ghost trunk inside the original, 128 a plain one; 32
  stays the default.

One source, one seed, by eye. Evidence: `flux-outpaint.log`, and
`flux-1-krea-dev/` (`outpaint-bands.jpg`, `outpaint-band-seams.jpg`,
`outpaint-strengths.jpg`, `seams-full.jpg`), in
`%TEMP%/local-diffusion-inpaint-card/`.

Z-Image Turbo, the seam by number and band widths 16, 24 and 48 were
measured 2026-10-24 (Inpaint, "Z-Image Turbo, an SDXL LoRA, the seam by
number"), on Animagine XL 4.0 and Z-Image Turbo; a 16:9 outpaint over
the Nunchaku Z-Image Turbo and FLUX.1 Krea dev transformers under WSL
2026-10-25 (Inpaint, "Under Nunchaku"). The earlier dev/schnell outpaint
and native source/result seam-number gaps are measured in the bounded
[2026-11-18 follow-up](#windows-follow-up-measurements-2026-11-18). Evidence
from the original runs is in
`%TEMP%/local-diffusion-inpaint-card/` with the inpaint's.

## Instruction edit

### FLUX.1 Kontext [dev], stage 3's first trial (2026-10-27)

`INP#11` stage 3, run unattended with Matt's approval after `GEN#22`. The
loader order was the plan's ("Stage 3's loader order"). Only `CONFIG_PATHS`
were fetched at `KONTEXT_V1.revision` (`24e9ded…`), and
`require_pinned(validate_configs(...))` passed before anything else moved.
Then `FluxKontextPipeline.download` fetched the weights at that revision
(506 s, into the Hugging Face cache; the repo's licence was already
accepted). `FluxKontextPipeline.from_pretrained(revision=...)` ran inside
`loading.weights_loading()` and `loading.checkpoint_files_released(True)`,
with `loading.quantization_config` (fp8 per component),
`loading.place_pipeline` (model offload) and VAE slicing and tiling. The
loaded class was `FluxKontextPipeline` with `vae_scale_factor` 8. The edits
went through the repository's own `DiffusersBackend.generate` →
`_edit_images` with source binding by `instruction_edit_admission.bind_request`.
The gate was opened only by a subclass in the harness, as
`tests/test_instruction_edit_call.py` does. The repository's gate, its load
holds and `CHECKPOINTS` are unchanged, so the real server still refuses.

Environment: Windows 11, RTX 5080, Python 3.13.15, torch 2.13.0+cu130,
diffusers 0.40.0, transformers 5.16.1, torchao 0.18.0, main at 343f8b7.
Recipe `kontext-v1`, fp8, model offload, Eager, no critic, a scratch gallery
and outputs directory. Sampled every 0.25 s; the guards were 4 GiB free RAM
and a 1 GiB rise in shared GPU memory, and neither tripped. It started from
40.9 GiB free (28.6 GiB committed of 85.6) with the card at 1,340 MiB and
shared GPU memory at 0.05 GiB; there was 42.5 GiB free when the load began.

The source was a synthetic, owned 1024x1024 picture planted in the scratch
gallery as image 1: sky, grass, a yellow sun, a house with a red roof and a
red ball. PNG sha256 `85fd9772a503c2c6d1cb525d2bd9616efa1c878a43327e11c6ee1758f420984f`,
pixels `289dada683104cc114bb01b6b14952a1055a7978742f1d8ce88a4d1f946542c9`.
1:1, so the recipe's 512x512 working and output size. Every edit ran
28 steps at guidance 2.5, one image.

| What | Result |
| --- | --- |
| Load | 38.1 s from disk; peak 28.8 GiB working set (the OS's own peak 29.2) and 35.4 GiB private, 15.2 GiB free at the lowest; after it 20.3 GiB working set, 26.5 private, 52.2 GiB committed |
| First edit ("Change the red ball to a blue ball.", seed 12345) | 29.3 s, the first step 7.0 s (the transformer onto the card), then 0.73-0.75 s a step; pixels `68b864f60dc6734a40695efba9161824fdd114af57b0d1830ba97b593d7822e8`, saved as row 2 |
| Each edit's memory | 12.00 GiB allocated, 12.55 reserved at the peak; 14,500 MiB whole-card by `nvidia-smi`; shared GPU memory 0.11-0.13 GiB throughout (no spill); the process at most 20.3 GiB working set, 40.0 private, 22.7 GiB free at the lowest |
| Replay (the same request and seed again) | 26.8 s, pixel-identical to the first edit (largest difference 0) |
| Cancel (`should_cancel` true from step 10, seed 777) | `Cancelled` at step 10, 12.8 s in; 0.04 GiB allocated afterwards, and the next edit ran normally |
| The pair: the two instructions at seeds 12345 and 777 | 25.8-26.1 s each; blue ball 777 `21d4523b…`, snow 12345 `18ecb485…`, snow 777 `2b5ae4a9…`; rows 4-6 |
| Unload | 2.3 s; the pipeline gone, 0.03 GiB allocated, the card back to 1,678 MiB; the process kept 18.2 GiB working set, 27.6 private (it exited after) |

What the edits look like, against the source:

- **Blue ball:** both seeds turned the red ball into a clean, flat blue ball
  and kept the house, sun, sky and grass. There is drift, though: the ball
  came out larger and moved up and to the right, the house moved a little,
  and the horizon rose by about 3-4% of the frame.
- **Snow:** both seeds covered the grass with snow, but they went further
  than asked. The red ball became a white snowball and the roof took snow:
  a snowy red roof on seed 12345, and a wholly white one on seed 777.

So the first slice fits the card with about 1.5 GB of headroom at 512x512,
reproduces bit for bit within a process, cancels and unloads cleanly. It
follows a simple instruction's intent, but it does not leave the rest of the
picture alone.

Not measured: any other size or aspect, a batch, int8, sequential or no
offload, a real photograph as the source, replay across two processes, the
real server's route and page (held by the gate at the time; stage 4 opens
them inside this run's configuration, on a local branch, and its
[card acceptance step](maintainer/plan-instruction-editing.md#card-acceptance-step) is
what would measure them), and WSL. Evidence:
`events.jsonl`, `samples.jsonl`, the PNGs, `sheet.png` (source and the four
edits) and the scratch gallery in `%TEMP%/auradream-inp11-stage3/`.

### FLUX.1 Kontext [dev] on diffusers 0.42.1 (2026-11-10)

`CARD#18`, run with Matt's OK: the envelope re-measured on the diffusers the
app installs, against 0.40.0 in the same script. The harness was the real
`DiffusersBackend`; a subclass reported only `instruction_edit_host()` as
the measured 0.40.0, so the envelope admitted the 0.42.1 run as stage 3's
harness opened the gate. Everything else was the app's: `load()`,
`instruction_edit_admission.bind_request` and `generate()`. Environment,
method and guard as in
[diffusers 0.42.1 against 0.40.0](#diffusers-0421-against-0400-2026-11-10).

The source was a new synthetic, owned 1024x1024 picture in a scratch gallery
(sky, grass, sun, a house with a red roof, a red ball), pixels
`c8b7f5deef55486fc4b7e9baadf33eb4b52a845383ecbde9589c8894a59e583e`; it is
not the 2026-10-27 source, which was not kept, so its hashes do not compare
with the stage 3 trial's. 1:1, so 512x512; 28 steps, guidance 2.5, one
image. "Pixels" is the sha256 of the decoded RGB bytes.

| What | diffusers 0.42.1 (ran first) | diffusers 0.40.0 (ran second) |
| --- | --- | --- |
| Load | 47.5 s; process peak 26.94 GiB working set, 33.4 private; guard tree peak 26.97 GiB; 16.77 GiB available at the lowest | 28.0 s (warm file cache); tree peak 27.13 GiB; 18.94 GiB available at the lowest |
| "Change the red ball to a blue ball.", seed 12345 | 43.1 s (the first step 20.8 s: the transformer onto the card), pixels `df2d0229c6ed46eee3fcf1eb5e1e12cfed967a7ab53d30a43004fbcd3654b338` | 41.0 s, pixels identical |
| Replay, the same request and seed | 26.2 s, pixels identical to the first | 25.5 s, identical |
| Cancel from step 10, seed 777 | `Cancelled` at step 10 after 12.2 s; 0.04 GiB allocated after | 11.6 s; the same |
| "Cover the grass with snow.", seed 777 | 25.5 s, pixels `6f2f926aa2048b436492458169238686b491798698db58159a0fc3c467386133` | 25.2 s, identical |
| Each edit's memory | 12.00 GiB allocated, 12.55 reserved (torch peak); 14,700 MiB whole-card peak (`nvidia-smi`) | 12.00 / 12.55; 14,698 MiB |
| Steps | about 1.34 it/s (0.75 s a step) | the same |
| Unload | 0.2 s, 0.03 GiB allocated | the same |

So Kontext on 0.42.1 is pixel-identical to 0.40.0 inside the envelope, with
the same memory and step time, and replay and cancel unchanged:
`QUALIFIED_ENVELOPE.diffusers_version` moved to "0.42.1". The load-time gap
is file-cache order (the first run read from disk), not the version: the
FLUX.1 Krea [dev] pair, run in the opposite order, shows the same.

Not measured: everything stage 3 left unmeasured (above), and the real
server's route and page (the
[card acceptance step](maintainer/plan-instruction-editing.md#card-acceptance-step)).
Evidence (events, guard samples, PNGs, the scratch gallery) is kept in a
local scratch directory, not committed.

### Kontext real-server and browser acceptance (2026-11-18)

`INP#11`, with Matt's OK: two separate one-load real-server sessions on
Windows/RTX 5080, driver 616.92, source `8b95ddd`, torch 2.13.0+cu130,
diffusers 0.42.1 and torchao 0.18.0. Cached pinned Kontext, fp8/model
offload, Eager/Native, no critic, scratch galleries and native browser
Load/Edit/More seeds/Cancel/Unload controls. Original-RGB source proof was
checked at admission and each edit; output stayed 512x512 at 28 steps,
guidance 2.5, one image per call.

The historical source was not retained. The frozen owned 1024-square
house/sun/grass/red-ball replacement has RGB `28b506e70b5876cb…3f42a`.
"Change the red ball to a blue ball.", seed 12345, produced
`7e0062553284e700b6755b4c0450b7a8da1f43900c26ba5ed1f34f9c53512689` in
the first call, after cancellation, and in a fresh process. No equality
with the historical `68b864f6` or `df2d0229` outputs is claimed.

| What | Full browser session | Fresh-process replay |
| --- | --- | --- |
| Load | 51.78 s | 31.23 s |
| First edit | 43.365 s | 34.515 s |
| More seeds 12346/12347; recovery 12345 | 26.221/25.656; 25.274 s | not repeated |
| Per-edit allocated/reserved peak | 12.001/12.553 GiB | the same |
| Whole-card peak | 14,464 MiB | 14,436 MiB |
| Process-tree RSS/private peaks | 26.85/40.723 GiB | 27.75/46.544 GiB |
| Minimum free RAM/commit | 20.44/11.52 GiB | 20.52/7.26 GiB |
| Guard samples, independently all `ok` | 755 | 291 |
| Mean/max sample gap | 0.249954/0.268507 s | 0.249895/0.277083 s |

More seeds saved the two intended edits. Seed 777 was cancelled at step 3
with no image saved; the next seed-12345 edit matched the first pixels.
The 3:2 source, Generate and Re-render held with their reasons and queued
nothing. Source pixels remained unchanged. Independent allocator/backend,
SQL/API/job and PNG peak/provenance checks agreed for all five saved images.
The ball is blue in every distinct seed; house, sun, sky and grass remain,
with drift in geometry/horizon and ball size/position.

All 1,046 samples passed the unchanged 15,500 MiB whole-card,
1,024/512 MiB adapter/process shared-rise and 4 GiB live RAM/commit limits.
The full session's first controller stopped after browser success because
its observer wrapped `unload`, whereas the public route calls `try_unload`.
That stopped receipt is preserved. The replay's corrected scratch observer
recorded actual `try_unload` accepted, backend unready, and 32/32 MiB
allocated/reserved after 0.334 s, with guarded post-unload samples. Both
browsers exited 0; owned processes/listeners ended and the card returned
to 1,277 MiB at 0%. One earlier pre-browser commit refusal loaded no model.

The exploratory admission used the old 40 GiB private-memory estimate plus
4 GiB live floor and 6 GiB margin, accounting for actual already-owned
server private memory before load. **Future initial commit admission is
56.544 GiB** (new observed 46.544 + 4 + 6), beside 45 GiB free RAM; subtract
only actual already-owned server-tree private immediately before load.
The higher replay peak supersedes the old estimate. Live limits are unchanged.

The [combined receipt](maintainer/instruction-acceptance-2026-11-18.json)
binds both sessions, source/tool hashes and independent raw audits. It
qualifies this replacement fixture's real route/page and cross-process
replay inside the existing envelope. Other sizes/aspects, photographs,
loaders/platforms and Krea-to-Kontext switching remain unmeasured here.
The later ordinary text-to-image branch fix does not change the edit branch,
which returns through `_edit_images` before ordinary sampling/invocation.

### Larger Kontext browser trial admission (2026-11-20)

The isolated browser trial at `87b25ab` checked the cached editor's six
pinned configs on Windows, RTX 5080 CUDA 0, driver 616.92, diffusers
0.42.1. GPU 0 was using 1,698 MiB against its 2,000 MiB starting
ceiling. Free RAM measured **34.48 GiB** against the required **45 GiB**.
Free commit measured **27.79 GiB** against **56.544 GiB**. The controller
held before starting a server or loading weights. All real rendering and
fresh-process pixel comparisons remain unmeasured for this trial.

The [session receipt](measurements/kontext-browser-2026-11-20.json) includes
the real guard rows and identifies both fake browser rehearsals. The
[trial protocol and commands](maintainer/plan-instruction-editing.md#running-the-isolated-trial)
apply when the workstation meets both starting thresholds.

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
| An app killed outright on Windows (stand-in server) | A server started through `_ServerProcess` stopped listening within a second of its app's TerminateProcess: the job's kill-on-close. Not tried with llama-server then; it was 2026-10-25 (below) | 2026-09-28 |
| Critique through the routes on the final tree, FLUX.1 Krea [dev], Windows | The 2026-09-28 row's bookshop prompt twice, `/api/prompt/analyze` then `/api/prompt/critique` with the body the page sends: 11.8 and 13.1 s, 1,177 prompt tokens both times (the route's analysis names Krea's "subject first", so the question asks for three copies of the prompt where the 2026-09-28 run's asked for two at 1,072), 1,125 and 1,460 generated at 140 and 144 tokens/s. Check took 2.6 s on the fresh server (tokenizers from the Hub cache) and 0.03 s after: CLIP-L 30 and T5 34 tokens, the order "Not checked", `guide.quote_text` and `guide.words`. Both rewrites quoted "open late"; the model's `reordered` changed words and was set aside both times, so `reordered` stayed null, and `dropped` was 0. The first answer kept one issue (A5: "shot on film" is no concrete style) and left out one that restated the quotation-mark finding; the second had none. +5,570 MiB over a 3,231 MiB baseline on the second (the first run's poll wrote nothing); the card was back at its baseline before the route answered, and no `llama-server.exe` was left after either | 2026-09-29 |
| The same on Animagine XL 4.0, an unknown tag ahead of 1girl | "zzkx oc, 1girl, green hair, masterpiece, smile, outdoors", the Danbooru index reachable ("zzkx oc" not found; green hair, smile and outdoors general). Check did not say "Partly checked": `order_checked` true and `order_detail` null, since an unknown tag ahead of 1girl is out of order whatever its category, which `unplaced_items` counts as settled. It gave one `guide.order` finding ("zzkx oc" before 1girl), `reordered` "1girl, zzkx oc, green hair, smile, outdoors, masterpiece", `guide.quality` and `guide.negative_prompt`, in 3.3 s on the fresh server (tokenizers and the tag lookup); 15 tokens on both CLIPs. Critique 20.3 s: 883 prompt tokens, 2,521 generated at 145 tokens/s; +5,626 MiB over 3,178. Three issues were left out for restating the tag-order finding, and the model's `reordered` for changing words, so `reordered` and `reordered_kind` were null; `dropped` 0. The one issue kept, "Missing negative prompt" (G5), restates the `guide.negative_prompt` finding, which validation has no restatement check for. The rewrite dropped "oc", added "[series]" and "[rating]" placeholders and appended the card's recommended negative prompt to the prompt; the lint flagged both brackets, a CLIP cut at 75 of 77 tokens, the order, 26 added words and the missing negative prompt. No `llama-server.exe` was left | 2026-09-29 |
| The same, an unknown tag whose category decides the order | "1girl, green hair, zzkx oc, smile, outdoors, masterpiece": `order_detail` "the tag index has no usable Danbooru category for 1 tag, so it is not placed: "zzkx oc"" (the page's "Partly checked"), no order finding and no `reordered`. Critique 18.3 s: 843 prompt tokens, 2,268 generated at 146 tokens/s; +5,687 MiB over 3,184, probably about 100 of it not the server's (3,282-3,302 MiB in the 5 s after the stop). No issue was left out; four were kept: two `order` issues on A1 asking for a series and a rating after the character, the missing quality finish (A2, restating the `guide.quality` finding, which validation has no restatement check for either) and "No negative prompt", cited to A3, the card's tag-format sentence. The model's `reordered` changed words and was set aside before the partial-placement check, so that check did not run and `reordered_kind` stayed null. The rewrite again added "[series]", "safe", the quality finish and the whole recommended negative prompt, and drew the same lint findings, with one bracket this time. No `llama-server.exe` was left after the run or after the app was terminated | 2026-09-29 |
| Critique under WSL 2 through the routes, the GGUF where the app puts it (the Hugging Face cache on ext4), FLUX.1 Krea [dev] lite-infer id | The real server with `wsl/env.sh` sourced and `--defaults` on `1b8e9b4` (the app as at `5aa01f4`), port 7990, a scratch database and outputs directory, the image model never loaded; the bookshop prompt at 1024x1024, 1,191 prompt tokens every time. The WSL cache held the b11206 tree but no GGUF, so the first critique's `prepare()` downloaded the pinned file into `~/.cache/huggingface/hub`, hashed and stamped it: 96.4 s for the request, 21.6 s of it the critique (1,559 generated at 111 tokens/s). With the file in the page cache: 11.4 and 11.6 s (929 and 1,050 generated at 117 and 121 tokens/s). After `echo 3 > /proc/sys/vm/drop_caches` (the VM's `Cached` fell from 5,138 and 4,842 MiB to 569 each time): 17.5 and 18.5 s (1,127 and 945 generated at 121 and 100 tokens/s). Less the generation at the server's own rate, the rest - start, load, the prompt, the stop - was 2.9-3.5 s warm and 8.2-9.1 s after the drop: reading the 4.28 GB file from ext4 adds about 5-6 s, where `/mnt/c` added about 20. The drop empties the VM's page cache only; Windows may still have held the virtual disk's blocks. Generation ran at 100-121 tokens/s, against 140-154 on Windows. +5,610-5,663 MiB over a 3,197-3,223 MiB baseline (the desktop's 2,946 and the idle app's context), back to it after each, no `llama-server` left after any critique or after the server was stopped; `dropped` 0 throughout, `reordered` null, one issue kept in one answer | 2026-09-29 |
| The critic's first-use line and clock in the page, over a real download (Windows, the FLUX.1 Krea [dev] profile, the image model never loaded; `HF_HOME` an empty scratch directory holding only a token; Check, then Ask the critic, pressed at 2560x1305) | Before the press: "First use downloads llama.cpp b11206 and Qwen3-4B Q8_0 (about 4.5 GB) once." - 4,856,283,677 bytes, in binary gigabytes (4.9 GB decimal). Check took 7.4 s, the CLIP and T5 tokenizers fetched on the way (CLIP-L 30, T5 34 tokens). On the press: "Downloading llama.cpp b11206 and Qwen3-4B Q8_0 (about 4.5 GB) once - this can take a few minutes - then thinking…", Ask reading "Thinking…", Generate aria-disabled "Wait for the critic to finish", and the clock from "· 1 s" a second later, a tick a second; at QHD the clock wrapped to a line of its own under the sentence. The status poll after the server had installed llama.cpp (both zips, 548 MB, in 8.7 s) changed the line at 11.2 s to "Downloading Qwen3-4B Q8_0 (about 4.0 GB) once - …", and the one after the GGUF had landed (4.28 GB in about 181 s, at 3-88 MB/s) to "Thinking… about 15 seconds on the GPU." at 191 s. The answer came 199.7 s after the press with the clock at "· 199 s"; the route's `seconds` said 9.4, which counts the critique alone - server start, completion and stop: 1,201 prompt tokens, 939 generated at 143 tokens/s. The line after it: "Local Qwen3-4B Q8_0 on the GPU, about 15 seconds; waits for image jobs.", unchanged by the next poll. A second ask: "Thinking… about 15 seconds on the GPU.", the clock to "· 11 s", the answer at 11.1 s against `seconds` 11.1. +5,600-5,670 MiB over the idle app's 2,380-2,420 MiB. An attempt six minutes earlier, whose page driver stopped waiting at 280 s, took 306 s on the server with the GGUF at 2-46 MB/s; neither comes near the 114.2 s of 2026-09-27, and the line promises no time for the download ("this can take a few minutes") | 2026-10-05 |
| Kill-on-close with the real llama-server (Windows; the pinned b11206 server and Qwen3-4B Q8_0 GGUF, the real app with `--defaults` and a scratch gallery, four trials each on a fresh app; the app's real interpreter killed with TerminateProcess) | One trial killed while llama-server was loading, three mid-answer with the card at 6,164-6,172 MiB: llama-server gone 0.05-0.46 s after the kill, the port free within 0.05 s, the card back to its baseline within 0.25 s, no `llama-server.exe` left after any (`tasklist`). The killed apps' `%TEMP%\llama-server-*.log` stayed, as expected of an app that never reached its `finally`. Evidence: `card11-critic/runs/phase1-kill/` in the cards directory (defined under Checkpoints) | 2026-10-25 |
| A critic reorder where Check placed only some tags (Animagine XL 4.0, fp8, model offload; five prompts Check placed in part, three critiques each, cold, and a fully placed control) | `reordered_kind` null in all 16: every critic `reordered` was set aside as "changed words, not only their order" before the partial-placement question was reached - the 4B model adds `safe`, `great score`, `absurdres`, writes `zzkx (series)` for `zzkx oc` and `hatsune miku (vocaloid)`. The only "check" of the session was beside the loaded pipeline on the fully known Miku prompt, equal to Check's own reorder. Each critique: +5,581-5,583 MiB over the idle app, 8.0-18.1 s, 891-943 prompt tokens and 831-2,403 generated at 147-155 tokens/s (every layer on the GPU, by that rate; the log at the app's verbosity prints no offload lines), adapter shared memory +172-180 MiB while llama-server ran (the same with no image model loaded: its own allocation, not a spill), nothing left after. Beside the loaded idle pipeline: +5,581 MiB, 9.1 s | 2026-10-25 |
| **A defect: the critic's rewrite loses the user's tags unreported** (the same SDXL session) | 8 of 19 rewrites lost or changed a tag the user wrote (`twintails` three times, `green hair` once, `zzkx oc` four times; `runs/phase2-sdxl/rewrite_tag_loss.txt`); `dropped` counts dropped issues, not tags, and nothing counts removals the way `heuristic.rewrite_adds` counts additions. One rewrite also put the literal slots `[series], [rating]` into the prompt (flagged `encoder.weighting`, the rewrite still offered). On Z-Image Turbo a lighthouse rewrite dropped "a dramatic and melancholic mood". Fixed 2026-10-25: `heuristic.rewrite_drops` (info) now names a tag the rewrite has in no form, a changed one included, and three or more lost prose words, the rewrite still offered (whether to set it aside is open); tested on these recorded prompt and rewrite pairs in `tests/test_critic_lettering.py`, and a CPU replay of the 19 saved critiques flags exactly the 8; not re-run on the card | 2026-10-25 |
| The critic beside int8 and sequential offload (Z-Image Turbo, the real app; FLUX.1 Krea [dev]'s fp8 load had stopped at the RAM guard, so Z-Image Turbo stood in) | int8, model offload: loaded in 19.1 s, idle 589 MiB; 768x768 8.0 s at 6.09 / 6.58 GiB and 7,407 MiB, 1024x1024 9.3 s at 6.29 / 7.14 GiB and 7,983 MiB. fp8, sequential: loaded in 17.5 s, idle 591 MiB (1,413 after its first image); 768x768 8.2 s at 1.37 / 1.86 GiB and 1,449 MiB. Beside each, a critique added +5,583 MiB (to 6,292 and 6,996 MiB) in 8.2 and 7.4 s at 149 and 148 tokens/s, with no spill, and the same image after it was the same hash. fp8 with model offload beside it, for scale: the image 6.42 / 7.16 GiB and 8,003 MiB, the critique +5,583 MiB to 6,294 | 2026-10-25 |
| Whether a rewrite beats its original over six seeds (judged by eye, one viewer, each image against the original prompt's elements; Z-Image Turbo fp8 at 1024x1024 and 8 steps for four prose prompts, Animagine XL 4.0 at 832x1216 for two tag prompts; 12 or 18 images a grid, 9.8-10.2 s and 7.4-8.6 s an image) | Of 36 seed pairs the rewrite won 12, lost 6 and tied 18. Quoting lettering won every seed: open_late ("Open Late" legible on all six against pseudo-glyphs on all six) and bookshop ("open late" against a hanging "OPEN"), 6 of 6 each. Pure rewording tied every seed (lighthouse with its mood clause dropped, hiker, Miku with Check's order and "(vocaloid)"). The rewrite that dropped `twintails` lost every seed (no twintails in any). Check's reorder against the original tied all six on both Animagine prompts. Not on FLUX.1 Krea [dev], whose session stopped at the RAM guard; its rewrites, made cold before the stop, were more invasive (restructured; an invented `"Hiker" in bold font`) and were never rendered, and the gate image's re-render against `7d161ca06d8753e7` was not reached. Evidence: `card11-critic/runs/judgement.json` and the sheets in `runs/phase2-sdxl/sheets/` and `runs/phase3z-zturbo/sheets/` | 2026-10-25 |

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
gallery was not used. Later [fp8/sequential coexistence and three seed pairs](#krea-sequential-coexistence-and-three-rewrite-seed-pairs-2026-11-18)
passed; the separate [int8/model attempt](#krea-int8model-host-commit-stop-2026-11-18)
hit its host-commit floor before the first PNG. [Abrupt WSL owner death](#wsl-critic-after-abrupt-owning-app-death-2026-11-18)
exposed a surviving process/listener despite delayed GPU release. A critique
beside a load remains refused by design; typed Ctrl-C remains hands-on.

### Critic partial-placement follow-up (2026-11-17)

Four frozen, eligible one-shot calls completed with pinned b11206 CUDA 13.4
and Qwen3-4B Q8_0 on Windows/RTX 5080, driver 616.92, with no image pipeline.
The three partial-placement prompts passed their CPU prerequisites, but the
actual model added `[series]` in cases A/C and changed the name to
`Arlen Voss (series)` in B. All were correctly rejected as changed words
before the source-bound partial-placement condition at validation line 689.
The fully placed control preserved the prompt pieces and retained Check's
canonical reorder with kind `check`. **The partial-placement target remains
unreached.** These are three new partial cases and one control, without
combining ambiguous earlier counts. No reply was repaired or retried.

Actual Check, the ordinary critic job gate and `PromptCritic.critique` ran;
all four normal process/listener/gate cleanups passed. Critique times were
18.66/13.86/19.49/10.14 s. All 304 guard samples re-evaluate to `ok`, with
0.250266 s mean period and 0.260073 s longest gap. Whole-card peak was
7437 MiB over 1825; process RSS/private 5.116/7.241 GiB, shared rises
184.148/98 MiB adapter/process, free-RAM/commit minima 28.273/23.452 GiB.
The 16/16 GiB admission and unchanged live card/shared/4 GiB floors held.
Final card was 1826 MiB and power was released. Numerical GPU layer counts
were absent; no layer count or full-GPU placement is inferred from timings
or memory. Llama allocations are outside Torch's allocator.

The [receipt](maintainer/card11-critic-2026-11-17.json) binds all 196 source
hashes, original structured replies, validator traces and unchanged raw
evidence. Zero image/assistant/download/Python Torch CUDA-init calls occurred.
This direct protocol did not exercise HTTP/UI or post-reply rewrite lint;
empty rewrite findings do not establish clean rewrites. PA#18 assistant
inference, image comparisons, WSL kill-on-close and typed Ctrl-C remain held
or untested. `CARD#11` stays open.

### Critic partial-placement trigger follow-up (2026-11-18)

Four new frozen, eligible one-shot calls completed with pinned b11206
CUDA 13.4 and Qwen3-4B Q8_0 on Windows/RTX 5080, driver 616.92, with no
image pipeline. The three partial cases used synthetic trigger tags
`velvetrune`, `petalweave` and `riverglyph`, rather than the previous
name-shaped candidates. All three passed current Check and the CPU
oracle prerequisites, but their actual reordered replies added words:
A added scene/subject tags, B added `[character name]` and `[series]`,
and C added `alice` and `outdoors`. All were correctly rejected as changed
words before the source-bound partial-placement condition at validation
line 689. **The partial-placement target remains unreached.**

The fully placed control preserved its words but put `green hair` ahead of
`safe`; Check's canonical documented order puts `safe` first. Its pure
reorder was correctly rejected, with no retained kind. This differs from
the 2026-11-17 control, which retained Check's canonical reorder with kind
`check`. This is one new session of three partial candidates and one
control, without combining ambiguous historical counts. No reply was
repaired or retried.

Actual Check, the ordinary critic job gate and `PromptCritic.critique`
ran; all four normal process/listener/gate cleanups passed. Critique times
were 10.90/17.16/17.33/17.57 s. Every one of 308 guard samples independently
re-evaluates to `ok`; mean/max periods were 0.250280/0.257327 s. Whole-card
peak was 6547 MiB over a 966 MiB admission baseline; process-tree RSS/private
peaks were 5.123/7.244 GiB, shared rises 98/98 MiB adapter/process,
and free-RAM/commit minima 44.263/47.831 GiB. The 16/16 GiB admission and
unchanged live card/shared/4 GiB floors held. Final card was 966 MiB at
0% utilization, with no owned worker or llama-server process, and power
was released. Numerical GPU layer counts were absent; no layer count or
placement is inferred from timings or memory. Llama allocations are outside
Torch's allocator.

The [receipt](maintainer/card11-critic-2026-11-18.json) binds commit
`f8f4cf2` (the ordinary text-to-image cancellation cleanup fix), 195
reviewed source-freeze entries, 200 runtime source hashes, original
structured replies, actual validation traces and unchanged raw evidence.
Zero image/assistant/download/Python Torch CUDA-init calls occurred.
This direct protocol did not exercise HTTP/UI or post-reply rewrite lint;
empty rewrite findings do not establish clean rewrites. Image comparisons,
WSL owner death/idle fallback and typed Ctrl-C remain separate. `CARD#11`
stays open.

### Krea int8/model host-commit stop (2026-11-18)

The one-load Krea int8/model-offload attempt passed fresh 45 GiB RAM and
40 GiB commit admission (48.969/52.529 GiB observed). The cached image
pipeline loaded in 43.89 s, but its first seed-4242 original did not finish:
the unchanged live commit floor stopped the owned process tree at 3.181
GiB headroom, below 4 GiB. That sample had 20.004 GiB physical RAM free,
14,057 MiB whole-card and only 80/80 MiB shared rises; it was a host-commit
stop, not evidence that the image exceeded VRAM. Across all 327 samples,
process-tree RSS peaked at 29.982 GiB and minimum
free RAM was 18.970 GiB. Maximum recorded private usage was 47.126 GiB;
private sampling is incomplete because the trip row has no private reading.
It is not a complete lifecycle/private peak. Every recorded guard decision
independently
re-evaluates to its original verdict, including the sole commit trip.

No PNG, critic call or comparison pair completed, and normal
cancel/replay/unload was not reached. The raw after-exit sample still
showed high card/low commit and failed its guard; that secondary error is
retained beside the first `TRIP`. A separately retained later direct
release check showed 1182 MiB at 0% utilization, 49.021 GiB RAM free,
52.583 GiB commit, no owned worker or llama-server process, and released
power. The source-frozen [receipt](maintainer/card11-krea-int8-stop-2026-11-18.json)
retains both readings unchanged. No int8 retry or limit relaxation ran.
This attempt qualifies neither int8 coexistence nor rewrite quality.

An independent CPU audit matched its 14 raw-file hashes, 201 runtime source
hashes, 195 reviewed source-freeze entries, all 327 guard decisions and the
distinct fresh release. No model replay was involved.

### Krea sequential coexistence and three rewrite seed pairs (2026-11-18)

A distinct one-load fp8/sequential-offload session completed on Windows/RTX
5080, driver 616.92, torch 2.13.0+cu130 and diffusers 0.42.1, at source
commit `f8f4cf2`. This is the sequential alternative to the int8/model
attempt above; it does not qualify that stopped int8 recipe. Fresh
45 GiB RAM/40 GiB commit admission observed 48.983/52.545 GiB. The
cached Krea pipeline loaded in 53.26 s, Eager/Native, with no public
execution capability changed and no model download.

The original was "A red fox sitting in a snowy birch forest at dawn.",
1024-square, 28 steps and CFG 3.5. Original seed 4242 rendered first so
the critic ran beside a used, offloaded pipeline. Actual Check, pinned
b11206 CUDA 13.4/Qwen3-4B Q8_0 identity checks, the ordinary JobManager
critic gate and `PromptCritic.critique` ran once. The critique took
14.70 s, then its process, listener and gate closed normally. Normal
post-reply lint ran too: the retained rewrite has one informational
26-word finding, under the guide's suggested 30-80 band. No reply was
repaired or retried.

The retained rewrite added "soft morning light filtering through the
trees and a frosty mist in the air." At matched seeds 4242/4243/4244,
the original/rewrite render times were 101.08/99.59, 100.08/99.63 and
99.52/99.48 s. The labeled six-image contact sheet, inspected at
512-pixel thumbnail size, keeps the seated red fox, snowy birches and
dawn in every image. The rewrite visibly increases mist/light beams
and frames the fox smaller in all three pairs. That added weather and
changed subject prominence are quality tradeoffs; they are not an
automatic improvement or a recorded preference from Matt. No gross
corruption was seen at that resolution; fine anatomy and identity
preservation were not qualified. Full-size PNGs, full RGB hashes and
pixel-difference metrics remain unchanged in the evidence; difference
metrics are not quality scores.

The actual `backend.generate()` step-5 cancellation took 18.55 s after
five progress callbacks. Its allocated/reserved memory afterward was
32.0/110 MiB; no peak was recorded for the cancelled cell. The next
seed-4242 original rendered normally in 99.60 s and exactly repeated
the first original's full RGB hash (`7d161ca06d8753e7…`). Its 16-hex prefix
also matches the historical Krea seed-4242 baseline. All seven completed
images peaked at 2.414/2.559 GiB allocated/reserved. Normal final unload
left the image model not ready, no critic process, a free gate and
32/32 MiB allocated/reserved. This uses the production cancellation fix,
with no harness offload workaround.

All 3173 guard decisions independently re-evaluate to `ok`; mean/max
periods were 0.250285/0.262226 s. Whole-card peak was 7912 MiB during
the critic, over 1182 MiB at admission. Process-tree RSS/private peaks
were 28.325/34.765 GiB, shared rises 180/180 MiB adapter/process and
free-RAM/commit minima 20.584/17.752 GiB. The unchanged live card/shared
limits and 4 GiB floors held. Final and fresh external card readings
were 1182 MiB at 0%, no owned worker or llama-server process remained,
and power was released. The fresh external host reading was
49.084/52.673 GiB RAM/commit. Numerical GPU layer counts were absent;
none are inferred from time or memory. Llama allocations are outside
Torch's allocator.

The [receipt](maintainer/card11-krea-sequential-2026-11-18.json) binds the
195 reviewed freeze entries, 201 runtime source hashes, all raw guard
rows, original critic reply, source/settings, image hashes, cancellation,
exact replay and normal cleanup. Its visual review names the resolution
and limits. This direct protocol covers standalone generation and the
critic gate, not HTTP/UI orchestration, PA#18, WSL owner death or typed
Ctrl-C. The sequential coexistence and three-seed comparison gap is
covered for this fixed prompt/recipe. `CARD#11` remains open for its
separate partial-placement target and remaining platform/hands-on checks.

### WSL critic after abrupt owning-app death (2026-11-18)

One real `PromptCritic` owner reached the authenticated completion-request entry
using pinned llama.cpp b11206 Ubuntu CUDA 13.4 and the checked Qwen3-4B Q8_0 GGUF,
then the test killed that owned app with SIGKILL. No image pipeline was loaded.
This was an abrupt owner-death observation, not typed Ctrl-C or a normal Unload.
The marker establishes completion-request entry after health/auth readiness; no
completed token, reply or critique is demonstrated, and no critique-latency
claim follows from it.
The unchanged production code ran inside an external safety subreaper, which
retained exact PID/birth/UID/session identities and owned only this experiment.

The same server and its listener survived every observation through 160 seconds.
Whole-card use remained 6,824 MiB at 120 seconds and had fallen to 1,502 MiB by
130 seconds through the server's existing idle-sleep fallback. That fallback
released most GPU memory but did not remove the process/listener. **Immediate
WSL kill-on-close did not pass.** The external safety supervisor then killed
only the recorded server group and proved no owned process remained.

Fresh spawn admission required 16 GiB physical RAM and commit. Live card/shared/
host floors stayed 15,500 MiB / 300 MiB / 4 GiB; the Linux owned RSS cap was
12 GiB with no swap growth. All 692 Windows samples independently recompute to
`ok`: card peak 6,824 MiB, adapter shared rise 120.74 MiB, host free-RAM/commit
minima 43.49/41.58 GiB. Linux RSS peaked at 4.46 GiB, VmSwap stayed zero and
Linux free RAM stayed above 45.19 GiB. Windows mean/max sample gaps were
0.256636/0.267707 s; Linux's distinct 4,006-sample log measured
0.043158/0.056438 s. No numeric GPU-layer count was retained; full-GPU placement
is not inferred from these measurements.

At GPU-baton release the card was back to 1,182 MiB at 0%, free RAM was 49.20 GiB
and commit headroom 52.64 GiB. Ubuntu reported Stopped naturally; no distro/global
termination or unknown-app shutdown command was issued. Exact Linux group
cleanup is retained, and later Windows TCP-table probes found no LISTEN entries
on the owned ports 7999 and 40703. The retained socket `connect_ex` value 10035
means WSAEWOULDBLOCK and is inconclusive by itself; it is not used as proof of
port closure. The later Windows TCP proofs have their own timestamps and do
not claim to be the earlier GPU-memory sample.

The [curated receipt](maintainer/card11-critic-wsl-2026-11-18.json) binds the observation. `CARD#11` remains open for
the production WSL lifetime/cleanup gap, typed Ctrl-C, and any remaining checks
in the consolidated critic owner. This session changes no production critic
lifetime behavior and establishes no rewrite-quality result.

## FLUX 3 [Klein] and its prompt upsampler (2026-11-10)

Every measurement above was taken on diffusers 0.40.0, except where its
section says otherwise. The app now installs
0.42.1 (`GEN#45`, [the bump](design-notes.md#newer-checkpoints-and-the-diffusers-042-bump),
re-measured in [diffusers 0.42.1 against 0.40.0](#diffusers-0421-against-0400-2026-11-10)),
and adds FLUX 3 [Klein], Klein Base and FLUX 3's prompt upsampler as Expand
([FLUX 3](design-notes.md#flux-3-klein-an-app-owned-pipeline-over-pinned-folders)).

`CARD#19` and `CARD#20`, run with Matt's OK once the licences were accepted.
Windows 11, RTX 5080 (16,303 MiB; the desktop held about 1.2-2.3 GB), 64 GB
of RAM, Python 3.13.15, torch 2.13.0+cu130, diffusers 0.42.1, transformers
5.16.1, torchao 0.18.0, main at 3812179 (after PR #62). One load per
process, the processes one after another in the foreground, each watched by
`tools/card_guard.py watch --watch-pid PID --period 0.5` at its default
limits (card 15,500 MiB, shared rise 1,024/512 MiB, RAM and commit floors
4 GiB), and the harness's own 4 GiB RAM floor. The harness was the app's
`DiffusersBackend` (`load()`, `applied()`, `generate()`,
`expand_prompt()`), model offload, a scratch gallery. "Pixels" is the
sha256 of the decoded RGB bytes.

Downloads, at the pinned revisions with the loader's allow patterns: Klein
(`09561231`) in 204 s and Klein Base (`a83ce283`) in 322 s, about 19 GB
each in bf16 (transformer 10.3 GB, text encoder 8.3 GB, VAE 0.3 GB); the
upsampler (`a5789954`, 18.8 GB) in 175 s. `flux3_loading.preflight` passed
on both downloaded snapshots: the official index and configs are what the
pinned loader expects.

### Klein and Klein Base (`CARD#19`)

1024x1024, "a red fox sitting in a snowy pine forest at dawn, photograph".
Each run: seed 12345, seed 12346, a replay of 12345, seed 12345 with Mu
1.0, one reference (a synthetic 1024x1024 picture of a house with a red
roof and a red ball, pixels `c8b7f5de…`) with "The same house and red ball
on a sandy beach at sunset, photograph", a cancel at step 5 (seed 777),
then unload. Klein at its 16 steps and CFG 1.0, Base at 50 steps and CFG
3.5.

| | Klein fp8 | Klein int8 | Klein bf16 (quantization none) | Klein Base fp8 |
| --- | --- | --- | --- | --- |
| Load | 21.4 s | 23.0 s | 2.8 s (lazy) | 19.8 s |
| RAM: process peak working set / private; guard tree peak; lowest available | 19.79 / 23.4 GiB; 19.2 GiB; 20.4 GiB | 19.51 / 23.1; 17.1; 20.6 | 21.1 / 40.8; 21.0; 19.0 | 18.76 / 23.0; 19.5; 20.8 |
| Peak allocated / reserved (torch), a plain image | 6.08 / 7.32 GiB | 5.84 / 7.32 | 10.95 / 11.55 | 6.08 / 7.32-7.34 |
| The same, one reference | 7.40 / 8.01 GiB | 7.40 / 7.97 | stopped by the guard (below) | 7.40 / 7.99 |
| Whole-card peak (`nvidia-smi`, guard) | 10,964 MiB | 10,702 MiB | 15,744 MiB, stopped at the card limit | 11,012 MiB |
| Shared rise (process / adapter) | 82 / 91 MiB | 82 / 89 | 146 / 217 | 82 / 131 |
| Seed 12345, the first image | 27.7 s (the first step 15.3 s: onto the card) | 14.6 s | 22.0 s | 73.9 s |
| Seed 12346 / replay / Mu 1.0 | 14.6 / 14.5 / 14.8 s | 13.7 / 13.6 / 13.7 | 16.0 / 16.6 / 16.0 | 73.3 / 73.2 / 73.0 |
| The reference | 29.1 s | 28.7 s | - | 159.6 s |
| Cancel at step 5 | 6.6 s; 0.04 GiB allocated after | 5.8 s; 0.04 | - | 10.0 s; 0.04 |
| Unload | 0.3 s; 0.03 GiB | 0.2 s | - | 0.3 s |
| Replay pixels | identical | identical | identical | identical |

Pixels (sha256 prefixes): Klein fp8 12345 `91b69362`, 12346 `092fec01`, Mu
1.0 `305818b3`, reference `f1ae2f01`; int8 `c89bd557`, `71764740`,
`9b7c58ca`, `e2d482dc`; bf16 `f02c96a4`, `89a2b4b6`, `aa1353c1`; Base fp8
`76db36f8`, `2de3c569`, `2da18997`, `1c7f025d`.

Looked at by eye (the fp8 Klein and Base contact sheets): a red fox in a
snowy pine forest at dawn on every seed; Mu 1.0 changes the picture (the
same subject); the reference keeps the house with the red roof and the red
ball, moved to a beach at sunset. Base's pictures differ from Klein's
(undistilled, with CFG).

bf16: four plain images ran at 10.95 GiB allocated (about 14.0 GB of the
card with the desktop); during the reference image the guard stopped the
process on its card limit. Shared memory had risen 146 MiB, short of a
spill. So bf16 fits plain 1024x1024 images but not one with a reference;
it was not retried.

So fp8 and int8 fit with room: about 11 GB of the card with the desktop,
6.08 GiB allocated on a plain image and 7.40 with one reference, which
doubles the time (twice the tokens). Klein takes about 14 s an image at 16
steps, Base about 73 s at 50 steps (CFG: two passes a step). RAM at the
load is about 19-21 GiB for the process tree, with about 12 GiB of working
set held after. Both fp8 plain peaks are under the 6.8 GiB below which the
picker's fit badge sizes no card, so FLUX 3 has no `_CARD_NEEDS` entry, as
FLUX.2 [klein] 4B has none; Klein's bf16 10.95 GiB is in `_BF16_NEEDS`.

### The prompt upsampler as Expand (`CARD#20`)

Klein at fp8 with model offload, through the app's
`backend.expand_prompt()`. A first attempt, with a browser and other apps
open (about 39 GiB available at the start), loaded Klein (12 GiB working
set); the upsampler's load and quantization then took available RAM to
3.24 GiB and the harness's 4 GiB floor stopped the process (guard: tree
peak 33.1 GiB, 7.35 GiB available at its last sample; the card never above
2,154 MiB). Matt freed RAM, and the retry started with 47.8 GiB available:

| What | Result |
| --- | --- |
| Klein load | 21.8 s; no refusal; the status before said not loaded, nothing to fetch (cached) |
| First Expand (loads and quantizes the upsampler at fp8, joins the offload chain, captions) | 81.9 s in all |
| RAM | process peak 40.06 GiB working set / 46.2 GiB private; guard tree peak 40.8 GiB; 7.97 GiB available at the lowest (from 47.8); held after, Klein and the upsampler resident: 23.8 GiB working set / 30.2 private |
| The card during a caption | 11.30 GiB allocated / 11.45 reserved (torch); 13,064 MiB whole card; shared rise 82 MiB, no spill |
| Later captions | 60.3 s (the same prompt: an identical caption, 1,460 characters); 57.1 s ("lighthouse on a cliff in a storm", 1,432 characters) |
| Stop | pressed after 3 s; stopped at 4.3 s, 0.03 GiB allocated after |
| An image from the fox caption, seed 12345, 16 steps | 15.7 s; 6.10 / 7.34 GiB peak, as without Expand (the upsampler is moved off the card before the image); pixels `856069c8` |
| Unload | the process held 19.6 GiB working set at the end (not all RAM goes back to the OS until exit) |

The captions were faithful and detailed, about 1,450 characters (under the
4,000-character prompt limit; `ExpansionTooLong` was not reached), and
greedy, so the same prompt gave the same caption. The image from the fox
caption matches it (a fox sitting in snow, snowy pines, dawn light).

So Expand on FLUX 3 works on the 5080 at fp8, but its first use needs
about 41 GiB of process RAM with Klein loaded: it ran with 47.8 GiB
available and hit the 4 GiB floor with about 39. The cost is the quantizing
load (the 18.8 GB bf16 file read, then quantized on the CPU); held after,
it costs about 12 GiB more than Klein alone.

Not measured: Klein Base at int8 and bf16 (the same transformer size as
Klein's); other sizes than 1024x1024, batches, more than one reference,
LoRA (none exist), sequential or no offload; the upsampler's captions
against BFL's bf16 upsampler (18.8 GB in bf16 does not fit the 16 GB card),
the int8 upsampler, captions long enough to reach the limit; FLUX 3 on the 8 GB 3070 (`CARD#21`,
deferred: no int4 build is wired in). Evidence (events, guard samples,
PNGs) is kept in a local scratch directory, not committed.

## diffusers 0.42.1 against 0.40.0 (2026-11-10)

`CARD#17` and `CARD#18`, run with Matt's OK. Windows 11, RTX 5080 (16,303
MiB; the desktop held 1,539 MiB at the start), 64 GB of RAM (35.9 GiB
available at the start), Python 3.13.15, torch 2.13.0+cu130, transformers
5.16.1, torchao 0.18.0, main at f1b4ad7 (after PR #61).

Method: the same script and the same process shape, run once on the locked
diffusers 0.42.1 (huggingface-hub 1.33.0) and once on an overlay,
`uv run --with diffusers==0.40.0 --with huggingface-hub==1.29.0` (the
versions printed in-process). One load per process, the processes strictly
one after another in the foreground, each watched by
`tools/card_guard.py watch --watch-pid PID --period 0.5` at its default
limits (card 15,500 MiB, shared rise 1,024/512 MiB, RAM floor 4 GiB, commit
floor 4 GiB). No guard tripped; shared GPU memory rose at most 82 MiB in
every run (no spill). Kontext's run is under
[Instruction edit](#flux1-kontext-dev-on-diffusers-0421-2026-11-10):
pixel-identical, with the same memory and step time.

The shipped checkpoints: `python -m backend.generate "a red fox sitting in a
snowy pine forest at dawn, photograph" --model M --quant fp8 --offload model
--seed 12345 --grid-seeds 2 --ignore-server`, the profile's steps and
guidance at 1024x1024, a scratch gallery and outputs per run. Peaks are the
row's `vram_peak_allocated`/`reserved`; the card peak is the guard's
`nvidia-smi` reading; pixels are the first 16 hex of the sha256 of the
decoded RGB bytes.

| Checkpoint (run order) | Steps / CFG | Pixels, seeds 12345 / 12346 (0.42.1 == 0.40.0) | Peak allocated / reserved | Card peak | RAM tree peak 0.42.1 / 0.40.0 | Per image 0.42.1 / 0.40.0 |
| --- | --- | --- | --- | --- | --- | --- |
| cagliostrolab/animagine-xl-4.0 (0.42.1 first) | 28 / 5.0 | `124d0f1aa29c11cd` / `6b1ffcbcd4b53ce0`, identical | 3.12 / 3.47 GiB both | 5,396 MiB both | 8.59 / 7.92 GiB | 9.5, 7.7 s / 8.4, 7.6 s |
| black-forest-labs/FLUX.2-klein-4B (0.42.1 first) | 4 / 1.0 | `a586f5311591d4da` / `e663920f9305fcb6`, identical | 4.58 / 5.19 GiB both | 7,160 MiB both | 17.15 / 14.75 GiB | 10.6, 4.7 s / 5.5, 4.2 s |
| Tongyi-MAI/Z-Image-Turbo (0.42.1 first) | 8 / 0.0 | `e619709cf6926630` / `f5ec9f032bfe1b8d`, identical | 6.42 / 7.16 GiB both | 9,180 MiB both | 18.92 / 15.78 GiB | 11.9, 10.3 s / 10.9, 10.1 s |
| black-forest-labs/FLUX.1-Krea-dev (0.40.0 first) | 28 / 3.5 | `cd6613512d57c47d` / `81fb89460911a17b`, identical | 12.00 / 12.63 GiB both | 14,772 MiB both | 28.60 / 25.51 GiB | 42.7, 41.2 s / 43.1, 41.8 s |

The peaks in bytes (allocated/reserved): SDXL 3350174208/3724541952, klein
4913255424/5574230016, Z-Image 6896113664/7692353536, Krea dev
12880261120/13555990528. Process wall time (load and two images): SDXL
31/31 s, klein 43/30 s, Z-Image 58/47 s (0.42.1/0.40.0), Krea dev 118 s on
0.42.1 (second) and 133 s on 0.40.0 (first).

So on all four checkpoints 0.42.1 renders the same pixels with the same GPU
peaks as 0.40.0: FLUX.2 [klein]'s VAE losing `force_upcast`, the latent
placement change and the offload device's default change nothing here.
Timing differences follow run order (whichever ran first was slower, from a
cold file cache), not the version: Krea dev, run on 0.40.0 first, was the
slower there. The RAM tree peaks ran 1-3 GiB higher on 0.42.1 for the three
run 0.42.1 first and for Krea dev run 0.42.1 second, so a small RAM rise on
0.42.1 is possible but not established (one run each, working-set
sampling).

Not measured: int8, bf16, sequential or no offload, other sizes, batches,
Krea 2 Turbo, klein 9B, Qwen-Image, ERNIE, the Anima, NetaYume and NewBie
converters, Nunchaku and WSL, refine, inpaint and outpaint, LoRA, the
latent preview, the execution candidates (regional compile, cuDNN attention
and the FBC cache stay held on 0.40), and the 3070 server. FLUX 3 (`CARD#19`,
`CARD#20`) was not run here: the token still got 403 on both Klein repos
(`GEN#46`); it was measured later that day, once the licences were
accepted ([FLUX 3](#flux-3-klein-and-its-prompt-upsampler-2026-11-10)). Evidence (events, guard samples, PNGs,
the scratch galleries) is kept in a local scratch directory, not committed.

## Native, forced cuDNN and First Block Cache (2026-11-18)

Private FLUX.1 Krea dev comparisons on Windows/RTX 5080, source `f8f4cf2`,
fp8/model offload, CUDA 0, Batch 1, 28 steps, CFG 3.5 and the fixed author
prompts/seeds. torch 2.13.0+cu130, torchao 0.18.0, diffusers 0.42.1,
accelerate 1.15.0, transformers 5.16.1, driver 616.92, cuDNN version 92000.
Actual `backend.generate`, its production normalization/scheduling and cleanup
ran. Native/Off were unmodified; the experimental arms used only bounded
instance/request admission, with public class/schema/route gates unchanged.

| Comparison | Retained scope | Non-first-case medians | Observed paired median duration reduction |
|---|---|---|---|
| Native / forced cuDNN | Three prompts, two seeds, 1024x1024 and 1360x768, then shape return; full replay in new processes | 40.585 / 35.880 s, then 40.774 / 35.957 s | 11.592%, then 11.812% |
| Off / FBC 0.05 | Six 1024-square pairs, repeats/Off restoration/cancel and Off-On recovery; complete fresh-process replay | 40.802 / 33.235 s, then 40.788 / 33.209 s | 18.546%, then 18.651% |

The first comparison excludes the first A image in each process but includes
the second aspect's first use; the FBC comparison excludes the first pair and
its first cache setup. CPU-profiler diagnostic timings are not representative.
Both use sequential order, without counterbalancing or a general speed claim.

Every fixed case and repeat is RGB-identical across processes within its own
recipe. All FBC-Off controls match Native. Native/cuDNN differ on all twelve
pairs (mean absolute channel difference 0.792-2.767/255, maximum 81-228), and
Off/FBC differ on all six (mean 0.381-2.228/255, maximum 87-166). These differences
repeat exactly. Numerical closeness, exact replay and lower wall times do not
establish acceptable image quality. Matt still must judge the FBC pairs.

All 57 forced cuDNN processors retain their override while the global registry
remains Native. Actual Q/K/V outputs are CUDA bf16. A separate CPU dispatch
diagnostic records 1,596 canonical cuDNN SDPA calls (57 times 28); its nested
forward count is not summed as extra work. Native records efficient-attention
operators instead. No CUDA kernel trace was captured. FBC retains 28 steps,
computes every block on each first step and reuses 224-392 of 1,596 block passes
per cached fixed case; Off reuses none. This is no cuDNN/FBC combination test.

Six independent benchmark processes retain 90 PNGs, six loads and three
cancellations without save. All 15,025 raw guard verdicts independently agree
with `ok`; maximum sample gap 0.275788 s. Whole-card peaks were 14,415 MiB Native
and 14,505 MiB for cuDNN/FBC, over 1,182 MiB idle. Maximum process private was
40.217 GiB; minimum free RAM/commit 19.784/12.367 GiB. Fresh 40/50 GiB RAM/commit
admission and unchanged live 4 GiB/card/shared limits held throughout. Cancel
and recovery offload to CPU; unload collects roots and all owned processes exit.

The separate original same-process switch produced its cuDNN image and cleanly
unloaded, then refused Native reload at 30.418 GiB free RAM / 24.799 GiB commit
against fresh 40/50 GiB requirements. It had one load, no Native phase and no
passed switch. Any later capped-memory experiment is a separate protocol.

A later, distinct capped-resident-v3 switch completed cuDNN unload and Native
reload in one process under a 47 GiB private-job cap and 36 GiB sampled job-RSS
ceiling. Readmission used current commitment plus 4.5 GiB allowance and only EX2
non-shareable private residency, with a 46 GiB physical pool; no peak/RSS/psutil
USS credit or allocator-reuse assumption. Its measured pool was 48.605339 GiB;
free commit 29.718575 GiB exceeded required 29.466999 GiB. Both images match
controls, including Native's historical `d62ee9fc...19d697b`. Both unloads collect
roots, clear readiness and leave 32 MiB allocated. All 691 guard rows agree with
`ok`; no cap attempt, card/shared/global-floor crossing occurred. Peak observed
job private/RSS was 40.165508/30.040581 GiB; free RAM/commit minima were
18.882721/11.911301 GiB. This guarded Windows-build-specific experiment does not
qualify uncapped/public switching or impose a new app memory policy. Its
[separate receipt](maintainer/perf3-switch-resident-v3-2026-11-18.json) retains
counter/API limitations, physical proofs and original refusal separately.

The [attention receipt](maintainer/perf3-qualification-2026-11-18.json),
[cache receipt](maintainer/perf4-qualification-2026-11-18.json) and
[original switch receipt](maintainer/perf3-switch-original-2026-11-18.json) bind
source/PNG/guard audits. Owners retain remaining quality/lifecycle boundaries:
[attention](maintainer/plan-attention-backends.md#private-card-results-2026-11-18)
and [cache](maintainer/plan-step-caching.md#private-card-results-2026-11-18).
Public nonempty execution remains closed and defaults are unchanged. Other
checkpoints, loaders, platforms, sizes, batches and recipe combinations are not
qualified by these runs.

## Tiny Windows CUDA compiler probe (2026-11-17)

`PERF#1`'s tiny isolated toolchain probe passed with torch 2.13.0+cu130,
triton-windows 3.7.1.post27, driver 616.92 and RTX 5080. A cached `uv --with`
overlay explicitly selected its verified bundled TinyCC and NVIDIA tools;
the ordinary environment still has no Triton distribution installed.

Direct Triton addition and actual Inductor first/warm results exactly match
4096-element reference CUDA arithmetic. The fixed graph compiled once and
returned twice, with a real generated kernel. A separate changed-static-shape
diagnostic hit the scoped recompile limit instead of returning a fallback;
flags restored. Retained PTX, cubin, generated source and cache hashes prove
actual toolchain work. Python network guards recorded no attempts.

All 201 guard verdicts were independently rechecked as `ok`: whole-card peak
2369 MiB over 2010 at admission, allocator peak 64.125/66 MiB, process RSS/
private peaks 0.951/1.686 GiB, free-RAM/commit minima 29.996/24.894 GiB.
Process/adapter shared rises were 82/83.273 MiB; mean/max sample gaps were
0.250482/0.283531 s. Final allocator was 0/0 and the post-exit card 2010 MiB.
The 2500 MiB idle allowance was chosen from the quiet desktop baseline;
the 15500 MiB card, shared-memory limits and 4 GiB floors remained in force.
Two earlier pre-permit refusals are retained; only the second records its
idle-VRAM comparison, so the first refusal's precise cause remains unknown.

This qualifies **tiny arithmetic in this isolated overlay with selected
bundled tools**. It establishes no quantized/offloaded model, app cache
binding, fresh-process cache reuse, default installation, model speed or
public execution support. The [receipt](maintainer/perf1-tiny-toolchain-2026-11-17.json)
and [regional owner](maintainer/plan-regional-compilation.md) retain exact
scope, artifacts, dependency and guard evidence; `PERF#1` stays open.

## Regional FLUX.1 Krea dev, private Windows overlay (2026-11-18)

Private Regional-v1 fp8/model execution completed the empty-cache first image,
three exact warm repeats, all twelve fixed Eager/Regional cases, a new static
aspect and return, actual fresh-process disk-cache hits and step-3 cancel/
recovery on source `f8f4cf2`. The app-owned startup cache binding and actual
`backend.generate` ran in an isolated Windows Triton 3.7.1.post27 overlay;
public nonempty admission and default compiler installation remain closed.

| Session | First / warm image observations | Whole-card peak | Safe guard samples |
|---|---|---:|---:|
| Cold Regional probe | 71.84 / 30.86, 30.34, 30.16 s | 13,829 MiB | 938 |
| Eager full matrix | 57.59 / 40.55-41.21 s | 14,209 MiB | 3,283 |
| Retained-cache Regional matrix | 50.91 / 29.82-30.80 s | 13,703 MiB | 2,573 |

The cold probe retained 55 sm_120a kernels/three graphs. The new process had
three actual fxgraph cache hits; introducing the second aspect added three
misses, reaching six graphs/110 kernels. Each Regional process had one initial
hidden_states stride specialization. Warm and return counters stayed stable;
external graph calls are not Eager fallback. Returning to the first shape was
exact. Over the matched 16 warm images per arm, wall-time medians were
40.637 s Eager / 30.280 s Regional (25.49% less time). This ordered comparison
is not a general speed claim or CUDA-event timing.

All twelve fixed Eager/Regional pairs change pixels (mean absolute channel
difference 0.723-7.290/255); image-quality acceptance remains Matt's. Both
step-3 cancellations saved nothing and recovered the exact first RGB. The
cold probe/Eager v1 observer records allocator unload and post-exit card state;
only retained Regional v2 explicitly proves cleared backend ownership/readiness,
collected pipeline/transformer roots and restored Native/SDPA state, at 32/32 MiB.

All 6,794 raw guard decisions independently agree with `ok`, longest gap
0.26227 s. Maximum tree private usage was 45.852 GiB, minimum free RAM/commit
20.99/8.94 GiB. Card/shared limits and live 4 GiB floors held. Each owned process
exited, power released and the card returned to 973 MiB. No outgoing attempt
occurred after initialization of the owned stdlib wakeup socket. The earlier
socket-guard bootstrap stop is retained separately; it did not reach weights.

The separate [early-request follow-up](#regional-cold-first-call-cancellation-follow-up-2026-11-18)
observed latency and safe cleanup but failed its historical recovery RGB gate;
the cause remains unisolated. It does not replace this completed matrix.
The prepared v4 stayed unrun under insufficient headroom. The [owner](maintainer/plan-regional-compilation.md#private-regional-model-and-retained-cache-2026-11-18)
and [matrix receipt](maintainer/perf1-completed-matrix-2026-11-18.json) retain
exact toolchain/observer/guard boundaries, images and remaining release gates.

### Regional cold first-call cancellation follow-up (2026-11-18)

The separately capped v5 stopped at its immutable start-permit schema before
framework imports, weights or forward work: its producer included an extra
version field. That failure is retained. Frozen v6 corrected only its protocol
label and exact producer/consumer schema, after cold integration and negative
schema checks. It made one load, one cancelled call and one recovery PNG,
then failed the frozen historical RGB gate. The third repeat did not run.
This is a failed recovery qualification, separate from the completed matrix.

The independent request occurred +3.00036 s after the planned call start,
before the first callback. Request-to-first-observing-poll was 40.049995 s;
request-to-backend-return was 42.853372 s. The first callback then cancelled.
No compiler interval was instrumented, so compiler/request overlap is not
established. Registered components were CPU and the allocator about 32 MiB
after cancellation.

Recovery RGB was `1d66cce409f42c80f69aa66f4428b215ab2581316cb921a6058a7e2f566a6118`,
different from the frozen historical `e54b9ba5…f7cf`. Its recorded
`exact_repeat: true` has a null prior hash and is first-seen bookkeeping,
not a within-v6 repeat. Request/private recipe, 14 common source/helper hashes,
distribution facts and compiler inventory match earlier evidence, but neither
harness retained complete weight-byte hashes or an explicit Hub revision.
No uncancelled image exists from the new v6 context, so the mismatch does not
isolate cancellation damage or any other cause.

CPU artifact analysis found changed selected configurations and ELF `.text`
in 4/23 kernel/config-family groups (5/24 selected occurrences), including
one layer-normalization choice. The three output-code wrappers match after
path normalization and readable FX/pre/post-fusion IR match bytes; two runnable
FX files still differ after normalization. Matmul/cuDNN precision/determinism
flags were not explicitly captured. Matching IR and changed machine text are
a confounder, not proof of arithmetic equivalence or the mismatch's cause.
Cold rebuild bit identity is not established.

The hard job-private cap was 47.5 GiB, with kill-on-close readback. Fresh
waiting capacity was 52.582043 GiB against the frozen 52 GiB target, at least
45 GiB free RAM, without peak/RSS admission credit. All 605 live guard rows
independently re-evaluate to `ok`: job peak 43.994347 GiB with no observed cap
attempt, exact-job RSS 26.417858 GiB, whole card 13,911 MiB and free-RAM/commit
minima 22.439613/8.569191 GiB. Existing card/shared limits and live 4 GiB floors
held. These conditional numbers do not lower the general 55.9 GiB plan.

Failure cleanup cleared owners/siblings/parameters/diagnostic readers,
collected pipeline/transformer roots, restored observed Native/SDPA state and
released the owned power request; the saved post-exit card was 1,182 MiB.
No after-failure `policy-restoration.json` exists: scoped source paths alone
do not establish observed restoration of Dynamo flags. No shim or global
Dynamo reset was used.

The [addendum](maintainer/perf1-bootstrap-and-cancel-addendum-2026-11-18.json)
binds raw attempts, the independent CPU audit and artifact comparison;
the [owner](maintainer/plan-regional-compilation.md#cold-first-call-cancellation-follow-up-2026-11-18)
retains full boundaries. The later [v7 retained-artifact control](#regional-retained-artifact-uncancelled-control-v7-2026-11-18)
is recorded next; it does not retrospectively pass v6's strict historical gate
or supply the missing v6 policy observation. Public reproduction/rebuild policy
and recipe admission stay separate.

### Regional retained-artifact uncancelled control, v7 (2026-11-18)

A separate approved fresh-process load made exactly two normal uncancelled
scene/A1024-square/seed4471 Krea fp8/model calls through actual `backend.generate`
with `should_cancel=None`. Both full RGB hashes were
`1d66cce409f42c80f69aa66f4428b215ab2581316cb921a6058a7e2f566a6118`,
matching v6 recovery and each other, differing from the earlier `e54…` build.
The request/private recipe, 17 common runtime/helper hashes and recorded
runtime/compiler facts match v6. No extra cell, shape or mode ran.

All 520 pinned files, 24 selected best-config records, 55 total cubins and
24 selected cubins stayed byte-identical before load, after each call and
after both; an independent post-exit rehash agreed. Actual FX graph-cache hits
were 3, misses 0, new kernel saves 0 and static-autotuner saves 0, with three
Dynamo graphs and unchanged second-call counters. Other activity remained:
async-compile cache misses/hits 76/38, static-autotuner loads 38 and benchmark/
benchmark_gpu counters 4/4. Their exact causes were not traced. These counters
do not establish zero runtime compilation, benchmarking or autotuning.
Exact serialized artifact identity supports only this artifact-local comparison.

Fresh waiting capacity was 52.535225 GiB against the 52 GiB target, with
at least 45 GiB free RAM and no peak/RSS credit. The job-private/RSS caps stayed
47.5/32 GiB. All 606 live guard rows and the three-snapshot admission independently
pass; job peak was 43.974918 GiB, no cap attempt, card peak 13,911 MiB and minimum
free RAM/commit 21.851025/9.017208 GiB. Card/shared limits and live 4 GiB floors
held. Load/call times were 61.74 s and 51.25/30.57 s, control observations rather
than additional benchmark arms.

V7 directly observed final Dynamo-policy flags equal to their initial values,
Native/SDPA restoration, cleared backend ownership/parameters/readers,
collected pipeline/transformer roots, unready state and 32/32 MiB allocator
after unload. Owned processes exited, power released and post-exit card was
1,182 MiB. These observations do not fill v6's missing after-failure flag receipt.

The control proves that v6's recovery RGB also occurs without cancellation in
a fresh process with its retained artifacts. V6's strict historical equality
failure and unrun third repeat remain unchanged; that failure alone did not
isolate persistent cancellation damage. The 4/23 changed selected-config groups
and ELF `.text` from the earlier build remain a confounder. Neither cancellation,
the OS cap, autotuning nor one compiler setting is an isolated cause. Missing
earlier precision flags and complete weight-byte/Hub-revision identity remain
limits. No general cross-build determinism or broader absence of cancellation
effects is established.

The [receipt](maintainer/perf1-v7-retained-artifact-control-receipt-2026-11-18.json)
binds the independent admission/guard/artifact audit. The bounded private
card-stage table is now measured, including early-request latency and artifact-local
recovery comparison. `PERF#1` routes to Matt's pair judgment (`USER`), with
source/release/compiler-artifact policy still required; no further GPU trial is
the next action. Public nonempty admission and default compiler installation
stay closed. The [owner](maintainer/plan-regional-compilation.md#retained-artifact-uncancelled-control-v7-2026-11-18)
retains full boundaries; the earlier matrix and v5/v6 receipts remain separate.

## Windows follow-up measurements (2026-11-18)

Matt authorized the remaining workstation card checks. These sessions used
Windows 11, RTX 5080 (16,303 MiB), driver 616.92, Python 3.13, torch
2.13.0+cu130, diffusers 0.42.1, transformers 5.16.1 and torchao 0.18.0.
The production source was `f8f4cf2` (the ordinary-generation cancellation
cleanup fix); every completed session retained unchanged source hashes.
Each process loaded one checkpoint, with fp8, Eager/Native and its stated
offload mode, against a scratch gallery. The mask/Refine/crop sessions
used `main.ensure_loaded` and the ordinary backend/storage calls; CARD#10
and ERNIE used the real server, with native browser presses where stated.
No new public execution recipe, default, size guard or clean-seam claim
was enabled.

All retained guard samples re-evaluate to `ok`: 15,500 MiB whole-card,
shared rises 1,024/512 MiB adapter/process, and 4 GiB free-RAM/commit floors.
The sampler used held Windows PDH counters at about 0.25 s. Raw sample or
record gaps, rather than only loop-start spacing, are retained in the
receipts; the longest completed-session gap below was 0.279 s. Private
and working-set peaks in this table are the watched model/server tree;
the global RAM/commit floors also include the browser and controller.

| Session | Images / cancellations | Sampled card peak MiB | Server/worker private GiB | Lowest free RAM / commit GiB |
| --- | --- | --- | --- | --- |
| FLUX.1 dev, model offload | 21 / 3 | 15,155 | 42.51 | 22.51 / 12.68 |
| FLUX.1 schnell, model offload | 18 / 3 | 15,255 | 46.45 | 19.86 / 8.73 |
| FLUX.1 dev, sequential offload | 8 / 1 | 3,143 | 33.50 | 21.54 / 21.55 |
| FLUX.1 Krea dev, masked-area crop | 6 / 0 | 14,373 | 42.61 | 20.95 / 11.64 |
| Z-Image Turbo, masked-area crop | 6 / 0 | 8,711 | 24.72 | 28.70 / 30.36 |
| Z-Image, masked-area crop | 6 / 0 | 10,143 | 25.30 | 28.80 / 29.78 |
| ERNIE image ladder and first-use Expand | 4 / 0 | 13,240 | 33.80 | 27.56 / 20.79 |
| ERNIE separate offline replay | 1 / 0 | 10,424 | 28.73 | 27.64 / 26.13 |

The completed calls unloaded normally; the direct backend sessions reported
32/32 MiB allocated/reserved after unload. Owned processes/listeners then
closed, and separate settled `nvidia-smi` reads returned about 966-973 MiB
at 0% utilization. Those settled reads are separate from an immediate
post-exit sampler row, which can still contain the just-ended context.

**Future admission:** retain each run's actual historical admission values.
The completed native FLUX/CARD#10 sessions used **50 GiB initial commit
admission**, Z-Image-family crops 35 GiB and ERNIE 44 GiB; their actual
starting headroom was sufficient and the live floors held.
The observed schnell private peak is 46.448 GiB, so a future same-recipe
commit budget is that measured cost plus the unchanged 4 GiB floor and a
stated margin: 56.448 GiB with 6 GiB margin. The separate native-browser
session below reached 47.131 GiB in the server tree, giving 57.131 GiB
before any separate browser allowance with that margin. These larger future budgets were not retroactively the actual admission
settings.

### FLUX dev/schnell Refine, inpaint and outpaint

The [FLUX receipt](measurements/windows-flux-2026-11-18.json) binds applied
requests, source/result hashes, row/PNG peaks and guard evidence. Each
checkpoint made its own 1024-square fisherman source at seed 12345;
passes used seed 12346. Inpaint used the same ellipse, hard or feathered
by 24 px, at strength 0.95. Dev ran 27 of 28 steps; schnell ran all four,
starting at sigma 1.0. The request was a blue scarf/coat inside that mask.
Outpaint added 400 px left/right to 1824x1024, at strength 0.95 and bands
0/32/64/128. Kept pixels were exact on every masked result.

| Checkpoint | Hard / feather-24 inpaint | Outpaint band-32 | 1536-square Refine | Same-size true-CFG 2 / off |
| --- | --- | --- | --- | --- |
| dev, 28 steps / CFG 3.5 | 40.28 / 39.93 s | 71.80 s | strength 0.1, 3 steps, 15.10 s; sigma 0.4741 | 12.45 / 8.50 s |
| schnell, 4 steps / CFG ignored | 10.58 / 9.69 s | 15.68 s | strength 0.5, 2 steps, 11.79 s; sigma 0.5 | 9.93 / 7.17 s |

Repeats and post-cancel recovery matched full RGB hashes; the original
text-to-image source returned unchanged afterwards. Schnell inpaint and
outpaint cancellation occurred after 2 of 4 actual steps; Refine after
1 of 2, not an assumed third step. The true-CFG branch reproduced and
differed from its off control. On dev, `frosting-lane.safetensors` at scale
0 matched no adapter, scale 1 changed the image, and removal restored the
base inpaint. Sequential dev repeated the same finite Refine/true-CFG
cases: every one of its eight images matched model offload exactly.
Its 1536 Refine took 21.39 s, true-CFG 24.15 s and the off control 12.28 s.
These are observed per-call timings, not a general throughput benchmark.

Seam step is `tools/image_compare.py`'s mean 8-bit neighbour difference at
the painted mask's zero/positive edge. A VAE round-trip control was not
made, so these values do not isolate encode/decode cost.

| Checkpoint | Inpaint result, hard / feather 24 | Outpaint result, bands 0 / 32 / 64 / 128 |
| --- | --- | --- |
| dev | 26.681 / 4.085 | 10.431 / 1.499 / 1.684 / 1.944 |
| schnell | 54.497 / 4.706 | 41.348 / 2.473 / 1.408 / 1.656 |

Native comparison boards were inspected. Dev Refine keeps composition and
main facial features while changing fine facial texture; schnell 0.5
keeps the composition but redraws face/hat detail. Hard inpaint leaves an
obvious ellipse; feather 24 softens it but retains blue/yellow blur.
The dev LoRA produces a gray/brown fuzzy wrap rather than clean blue-scarf
adherence. Dev outpaint band 32 hides the cut better on this fixture, but
wider bands ghost roof/mast structure. Schnell extensions contain unwanted
partial duplicate yellow figures/coats and scene mismatches. No clean-join,
identity-preservation guarantee or preferred universal padding follows.

Native full-residency FLUX was not loaded. The
[alias-aware header/config proof](measurements/native-full-residency-2026-11-18.json)
deduplicates stored aliases, including the effective installed T5 embedding
tie. Even charging every remaining dense element only one byte gives
16,089 MiB for dev/Krea and 16,079 MiB for schnell, before activations or
desktop: already above the unchanged 15,500 MiB guard. This bounds native
weight-only fp8/int8, not Nunchaku/int4.

`CARD#7` now retains only a deferred planning case: the native fp8 FLUX.1
dev `offload=none` leg of the same 1024-square source to 1536-square,
strength-0.1 comparison. Its model and sequential legs are measured; the
full-residency leg has no safe admission on this 16 GB card. No other
family/offload case was selected, and K4/K5 Never plus the accepted
[2026-11-12 decision](maintainer/review-2026-11-12-todo.md#matts-answers-2026-11-12)
do not commission a replacement int8/bf16/large-size or catalogue sweep.
Only if Matt later chooses this exact mode on viable hardware should a
CPU planning slice freeze the same recipe and recompute alias-aware weight,
activation and host-commit budgets before any model load. This is not a
recommendation to rerun the completed comparisons or raise their guards.

### Masked-area crop follow-up

The [crop receipt](measurements/windows-crops-2026-11-18.json) records three
independent one-load sessions. Each used the existing owned SDXL source
(`720a6504...`) explicitly CPU-resized to 1536/2048/2896 square, never
claimed as a native render of the tested checkpoint. The finite cases
were pad 32 at all three sizes, pads 0/64 at 2048, and a pad-32 repeat.
Every result retained exact kept pixels and recorded its computed box and
working size. Krea model-offload images matched all six earlier
sequential-offload RGB hashes, at 41.55-43.95 s against about 100-102 s.
Z-Image Turbo took 10.65-10.88 s warm (16.88 s first); undistilled Z-Image
about 93 s per 50-step/CFG-5 crop. Both repeated exactly.

The Krea images inherit the earlier identical-pixel visible-patch limits.
New Z-Image/Turbo native boards show conspicuous sky/forest, branch,
river/fence and scale/style joins at every padding, with enlarged soft
detail on the 2896 source. There is no same-model-source or full-canvas
baseline here and no clean seam or best padding accepted. Canvas, crop
rule and size guards are unchanged. The missing public Z-Image inference
files at the existing `04cc4abb...` revision were downloaded explicitly
(12,493,472,775 bytes) and verified against published LFS SHA-256/Git blob
identifiers before its offline session; no new licence gate was accepted.

### ERNIE reproduction and actual first-use Expand

The [ERNIE receipt](measurements/windows-ernie-2026-11-18.json) binds the
real route/job/API/SQL/PNG/allocator checks. Seed 12345 at 1024 square,
8 steps/CFG 1 reproduced in-process and in a separate offline process,
full RGB SHA-256 `cdf4f9b4d3e7c62a866268312c99b49be2a841fdf14ef156885ea2367ccb8b53`.
The 1440/2048-square rungs completed at 11,046/13,240 MiB. Their fresh
projection for a 2048-square batch of two was 17,587 MiB against 15,200;
that call was refused before generation, not executed or proved to fit.

The native Expand press displayed the genuine first-download sentence,
including the translator, and an aria-hidden clock: 513 running samples,
129 distinct clock values. It returned expanded Chinese text and English
translation for review. Rewriter and translator weight hashes matched the
published files. No image from the expanded prompt or broad prompt-quality
comparison was performed. An earlier attempt's completed image is retained
separately: its observer wrongly treated API/SQL names as PNG keys. The
corrected explicit `model`/`size` namespace mapping passed a CPU regression
against real `storage.save_image` output, both seed+i images and all twelve
one-field corruptions, before the valid sessions. No backend defect was
involved in that observer stop.

### Real-page legacy packs and remaining coverage

All six page-pack assertions passed on real scratch servers, with separate
fresh galleries for the two bin-emptying packs. The
[CARD#10 receipt](measurements/windows-card10-2026-11-18.json) binds native
presses and the frames/guards. A stale fixture was corrected in scratch
copies: Sampling now opens with its summary before Steps is typed, verifies
visibility/value, then closes before original geometry observations. The
setup prefix passed on the fake without pressing Generate. Original
scripts and the earlier fixture stop remain separate evidence.

Batch preview/tiles/head and Generate position stayed fixed. ETA was absent
at step 1 and met its later bound. A one-minute grid predicted two complete
seeds and stopped with five cells/images, two complete seed pairs. Deleting
a source stopped its sweep after three failures with two images kept;
Finish stayed held, explained why and sent nothing on both presses.
Deleting during an already-started pass preserved the result's deleted-source
facts/title/note/Reuse. Normal unload, source freeze and cleanup passed.

The 7733f81 temporal scrollbar branch is **not** qualified: the initial
layout observed bar 15 throughout. A separate native Custom-closure variant
observed bar 0 throughout at 2560x1305; actual root/body/stage gutters were
`auto`, beside false, and every viewer-observed document height was 1305.
Both static layouts passed geometry assertions, but neither trace captured
overflow appearing and disappearing. No app CSS was changed to force it.
Typed Ctrl-C on the fake and real loaded/loading consoles was not performed;
`DEV#23`/`CARD#10` retain those hands-on gates. See the
[updated pack owner](maintainer/plan-card-harness.md#real-page-follow-up-2026-11-18).

`GEN#16` also remains unqualified. Its new preflight made no model load or
pass: matched cold/warm projections were 15,198/15,202 MiB, the warm value
2 MiB past the unchanged 15,200 limit. The
[owner](maintainer/plan-then-refine.md#matched-stage-preflight-2026-11-18)
records the prior-stage proof and preserved harness correction.


## RTX 3070 (8 GB), native Ubuntu (2026-11-03)

The native Linux route on a second machine: Ubuntu 26.04 LTS (kernel 7.0),
NVIDIA driver 595.99.02 open, headless (`nvidia-smi` reports CUDA 13.2), an
RTX 3070 with 8 GB (7.7 GiB reported, compute capability 8.6), an i5-12600K
and 64 GB of RAM; torch 2.13.0+cu130, diffusers 0.40.0, torchao 0.18.0,
Python 3.13 from uv. Peaks here are whole-card `nvidia-smi` readings (polled
every second), not the allocator's.

| What | Result | Date |
| --- | --- | --- |
| Test suite, native | 8250 passed, 111 skipped on main once Node was on PATH (without it, the tests that drive the page's scripts fail); this branch 8350 passed, 111 skipped in 3 of 3 runs at `-n 8`; ruff clean. Two stream tests raced the instant fake (the stream attaching after the first image) before 79c18a8 | 2026-11-03 |
| `python -m backend.doctor` | Every row PASS but VRAM (WARN: under the default's 12.0 GiB peak, naming the lighter checkpoints and the int4 Nunchaku repos) and the Hugging Face sign-in (WARN: not signed in) | 2026-11-03 |
| Animagine XL 4.0, CLI (`backend.generate --model cagliostrolab/animagine-xl-4.0 --quant fp8 --offload model --seed 1 --batch 2`), 1024x1024, 28 steps, CFG 5 | Ready in 1 min 43 s including the first download (about 6.5 GB into the Hugging Face cache); 2 images in 30 s; 4,289 MiB whole-card at the peak; 11.3 GiB maximum RSS (`/usr/bin/time`). The images are clean. So torchao fp8 weight-only (`Float8WeightOnlyConfig`) runs on Ampere sm_86: the weights dequantize to bf16 for each matmul, with no fp8 tensor cores needed | 2026-11-03 |
| Animagine XL 4.0 through the systemd user service (`linux/install-service.sh`, lingering on), driven from a Windows PC over the LAN | **Switch & load** to fp8 with model offload from a warm cache in about 10 s; 4 images at 1024x1024, 28 steps, one batch, in 54 s; 5,421 MiB whole-card mid-job | 2026-11-03 |
| Keep awake, during that job | `systemd-inhibit --list` showed `AuraDream ... sleep:idle ... block`; `/api/status`'s `keep_awake.reasons` named the job | 2026-11-03 |
| Keep awake, the idle minutes | With `AURADREAM_KEEP_AWAKE_IDLE_MINUTES=1` and only `/api/status` polled every 10 s, the hold let go after 60 s: `holding` false and no inhibitor listed | 2026-11-03 |
| The page over plain `http` on a LAN address | Not a secure context: the copy items and desktop notifications are held with their reasons, as designed | 2026-11-03 |

Not measured: Nunchaku int4 on sm_86 (whether the published kernels include
it); any FLUX-size checkpoint on 8 GB (FLUX.1, Krea 2, klein 9B, Qwen-Image;
model offload keeps a whole transformer on the card, larger than 8 GB at
fp8); Anima or FLUX.2 [klein] 4B on the 3070; sequential offload; refine,
inpaint and outpaint (their "measured" sizes are the 5080's); a suspend
with a model loaded, and whether the loaded pipeline survives the resume
(`NVreg_PreserveVideoMemoryAllocations=1` is set); the prompt critic's CUDA
13.4 llama.cpp build under the 13.2 driver; waking the PC from the network
(the server owner's own setup, outside the repository). Encrypted
automatic backups and backup cleanup are Windows only; unencrypted
automatic backups use the ordinary bundle path and were not run here.
