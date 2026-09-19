# Verified on the card

What has been measured on the RTX 5080 (16 GB), and when. The reasoning
around each number is in `design-notes.md`; this is the lookup. Times are per
image with model offload unless said otherwise, and "peak" is what the
process held, not what `nvidia-smi` showed - the desktop holds 5-6 GB of the
card while Unreal Editor and the browsers are open, and a run that lands
within a gigabyte of the top spills into system RAM and slows two to three
times (schnell's 2 MP read 35-45 s that way and 25 s at a 1.3 GB baseline).

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
| FLUX.1 Krea dev | fp8, Windows | 1 MP | 8 | - | 14.95 GB, 2.1 GB held between cells | 2026-09-08 |
| FLUX.1 Krea dev | fp8, Windows | 1 MP | 28 | 15.4 s; 17.5 / 16.3 / 18.6 s under a LoRA at three weights; ~1.65x with true CFG 1.5 | - | 2026-09-08 |
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
| Pause | Acknowledged at the cell boundary; the card drops from 14.95 GB during a 1 MP fp8 denoise to 2.1 GB held | 2026-09-08 |
| `reindex --from` over the real gallery | Reproduces every hash and true CFG | 2026-09-08 |
| Qwen-Image's rows | Record true CFG 4 and the negative prompt | 2026-09-08 |
| `uv.lock` torch hashes | The four hand-pinned wheels checked file-for-file (`RECORD`) against the Windows and WSL venvs | 2026-09-08 |
| Seed reproduction under Nunchaku NVFP4 | Not bit-for-bit: the same request in two processes gives two hashes on Z-Image and ERNIE alike. Z-Image's pair differ by 4.5/255 mean with 17% of pixels off by more than 8 (a different seed: 36/255, 85%); ERNIE's by 14-24/255 and 58-75% (a different seed: 41/255, 94%) - visibly different details, not noise. The fp8 path on Windows reproduces (above) | 2026-09-08 |
| ERNIE-Image Turbo's ordinary load without the prompt rewriter | Through `DiffusionPipeline.from_pretrained`, `pe=None` pulled the 7.7 GB folder at load and the process held 11 GB of RAM; through `ErnieImagePipeline.from_pretrained` the same load fetched nothing with the folder deleted from the cache, and the generation matched | 2026-09-08 |
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
