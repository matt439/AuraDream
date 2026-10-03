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

Not verified: `LOCAL_DIFFUSION_DEVICE=cuda:1` (one card here); the fp8 path for klein
on Windows; ERNIE-Image Turbo on the fp8 path; ERNIE's new explicit prompt
expansion path on the card; a live-preview map for FLUX.2's VAE (ERNIE, klein);
Animagine XL 4.0 speed, VRAM, LoRA behavior, fp8 path and four-channel SDXL VAE
preview fit; the quantization and offload settings of its successful user run
were not recorded. Illustrious XL v2.0-STABLE int8/full-bfloat16 GPU runs,
LoRA behavior, batch scaling, peak VRAM and repeat-seed determinism are also
unmeasured.

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
starting strength, not a quality benchmark. Not measured: a 2x pass on any
checkpoint; Animagine, Kohaku and Illustrious (the SDXL pipeline WAI
exercises); FLUX.1 dev, whose starting strength carries over the Krea dev
measurement on the same schedule; schnell, Z-Image Turbo and Qwen-Image,
whose starting strengths are schedule arithmetic alone; int8 and full
bfloat16; any pass under Nunchaku in WSL, where seeds do not reproduce
bit-for-bit to begin with; and the author's own WAI recipe, which upscales
with R-ESRGAN 4x+ Anime6B where this pass resizes with Lanczos.

Evidence is under `%TEMP%/local-diffusion-refine-validation/`:
`validate_refine.py` and `sweep_refine.py`, and per checkpoint (`wai`,
`flux`, `zanime`) the `validation.log`, `results.json`, the sweeps' JSON,
the separate database and the `outputs/` PNGs. The browser checks are
`tools/smoke-refine.json` and the additions to `smoke-conditioning.json`.

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
