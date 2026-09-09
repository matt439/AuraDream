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
| Krea 2 Turbo, `KREA_OFFLOAD=none` | NVFP4, WSL | 1 MP / 2 MP | 8 | 26 s / 72 s | 14.7 / 18.7 GiB, spilling | 2026-09-05 |
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

Under model offload the per-image weight transfer costs about 5 s and
dominates a 4-step image; on FLUX.1 the text encoders under bitsandbytes NF4
add to that, which is why schnell is not the speed win Z-Image Turbo is.

## Behaviour

| What | Result | Date |
| --- | --- | --- |
| Bit-determinism through torchao fp8 with offload moving components mid-forward | The same request twice on FLUX.1 Krea dev is the same hash; a negative prompt the pipeline dropped hashes the same as none | 2026-09-08 |
| True CFG on FLUX.1 | A negative prompt with no true CFG is byte-identical to none; true CFG 1.5 is a different, reproducible image | 2026-09-08 |
| LoRA under torchao fp8 + PEFT | The adapter changes the image and reproduces; `unload_lora_weights` restores the base hash; weight 0 is byte-identical to no adapter | 2026-09-08 |
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

Not verified: `KREA_DEVICE=cuda:1` (one card here); the fp8 path for klein
on Windows; ERNIE-Image Turbo on the fp8 path; ERNIE's new explicit prompt
expansion path on the card; a live-preview map for FLUX.2's VAE (ERNIE, klein).
