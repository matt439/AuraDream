# Krea Local

Run **Krea 2 Turbo** or **FLUX.1 Krea [dev]** locally — up to 4 megapixels on a
single consumer GPU. FastAPI + diffusers backend, vanilla-JS frontend, SQLite
gallery. FLUX is the default because its repo is click-through gated; Turbo
needs approval first.

![stack](https://img.shields.io/badge/torch-2.13%2Bcu130-informational)
![stack](https://img.shields.io/badge/diffusers-0.40.0-informational)

## Requirements

- NVIDIA GPU, **16 GB VRAM minimum** (float8 path). 24 GB+ can run `KREA_QUANT=none`.
- ~30 GB free disk for weights.
- Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

## One-time setup

Every official Krea/FLUX checkpoint is a **gated** HuggingFace repo. Log in first:

```powershell
uv run hf auth login          # paste a read token from hf.co/settings/tokens
```

Then request access on the model page. Two different kinds of gate exist, and the
difference matters:

| Repo | Gate |
| --- | --- |
| [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo) | **Manual approval** — you wait for a human |
| [`krea/Krea-2-Raw`](https://huggingface.co/krea/Krea-2-Raw) | **Manual approval** |
| [`black-forest-labs/FLUX.1-Krea-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) | Click-through — usually instant |

Check what you actually have access to:

```powershell
uv run hf download krea/Krea-2-Turbo model_index.json
```

`Access denied. This repository requires approval.` means the request is still
pending — which is why FLUX.1 Krea [dev] is the default. Point the app at Turbo
once you are approved:

```powershell
uv run python -m backend.main          # pick Krea 2 Turbo from the menu, or:
uv run python -m backend.main --model krea/Krea-2-Turbo
```

The UI seeds Steps/CFG/Mu from the checkpoint's profile on load, so there is
nothing to remember: FLUX gets 28 steps at CFG 3.5 with Mu blank, Turbo gets
8 steps at CFG 0 with Mu 1.15.

The loader is model-agnostic — it resolves the pipeline class and the set of
quantizable components from the repo's own `model_index.json` — so no code
changes are needed to switch.

## Run

```powershell
uv sync                            # first time only
uv run python -m backend.main
```

Started bare on a terminal, it asks which checkpoint, quantization and offload
to use - numbered lists, Enter keeps the marked default, and a repo id that is
not listed can be pasted in. It then prints the `--model/--quant/--offload`
line that would start the same thing without asking. Any argument at all
skips the menu, and whatever it does not name comes from the `KREA_*`
variables below:

```powershell
uv run python -m backend.main --defaults               # the environment, no questions
uv run python -m backend.main --model krea/Krea-2-Turbo --quant int8
```

A run whose stdin is not a terminal - a service, a script - never asks either.

Open <http://127.0.0.1:7860>. The first generation downloads ~25 GB of weights and
loads them — expect several minutes. Subsequent loads read from the HF cache.

`Ctrl`/`Cmd` + `Enter` generates from anywhere in the page. With an image open,
`←` and `→` walk the strip it came from — the current run, or the gallery, which
pages in more as you reach the end of it — `F` stars it, `Delete` removes it
after asking, and `Esc` closes it. Both are also buttons under the image.

## Run under WSL 2 (4-bit Nunchaku)

The float8 path above is the fastest thing that runs on Windows. Nunchaku Lite —
SVDQuant 4-bit weights *and* activations with its own CUDA kernels — is faster
(1.35x measured at 1 MP on a Blackwell workstation card, 1.8x with
`torch.compile`) and stores the transformer in a little over half the space,
which is VRAM headroom for larger images. Its kernels are published for Linux
only, so on a Windows machine that means WSL 2, where the GPU runs at native
speed through the Windows driver. Nothing in the app changes: only where it
runs and which checkpoint it loads.

One-time, from a WSL shell at the repo root (the checkout can stay on `C:`):

```bash
bash wsl/setup.sh
```

That installs `uv` and Python 3.13 if missing, builds a Linux environment under
`~/krea-local/.venv`, and checks that CUDA and the two extra packages are there.
Then, in every new shell:

```bash
source wsl/env.sh
bash wsl/serve.sh                  # or: uv run python -m backend.main
```

`serve.sh` forwards its arguments, so bare it shows the same menu as on
Windows - with the Nunchaku option and the lite-infer checkpoints on offer, and
`env.sh`'s choices marked as the defaults - and `bash wsl/serve.sh --defaults`
starts without asking.

Open <http://127.0.0.1:7860> in a Windows browser as before; WSL forwards
localhost. `wsl/env.sh` is what makes this work. It keeps the virtualenv, the
gallery (`KREA_OUTPUTS_DIR`, `KREA_DB_PATH`) and the HuggingFace cache on the
Linux filesystem — `/mnt/c` is a slow bridge and SQLite cannot lock across it —
and sets `KREA_QUANT=nunchaku` with a pre-quantized checkpoint:

| `KREA_MODEL_ID` | Card |
| --- | --- |
| `lite-infer/krea-2-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` (default) | RTX 50 |
| `lite-infer/flux.1-krea-dev-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` | RTX 50 |
| `lite-infer/flux.2-klein-4b-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` (also `9b`) | RTX 50 |
| the same names with `int4` in place of `nvfp4` | RTX 20–40 |

These are the official weights re-uploaded pre-quantized, and not gated. Any
other `KREA_QUANT` value works from WSL too; set it before sourcing `env.sh`.

Two things to know. The WSL gallery is separate from the Windows one — two
databases, two output directories. And run `uv` from WSL only after sourcing
`wsl/env.sh`: without it, uv finds the Windows `.venv` in the project directory,
judges it broken, and replaces it.

`KREA_QUANT=nunchaku` implies `DIFFUSERS_TRUST_REMOTE_KERNELS=true`. The kernels
come from a Hub repo that diffusers does not list as a trusted publisher, and
choosing the mode is choosing to download and run them. Setting the variable to
`false` yourself still wins.

`env.sh` also sets `TORCH_DISABLE_NATIVE_JIT=1`. torch 2.13 runs a few CUDA ops
as Triton kernels even outside `torch.compile`, and Triton needs a C compiler
the first time it runs; a fresh Ubuntu has none, and the text encoder hits one
of those ops on the first step. The flag makes torch use its ordinary kernels
instead, at no cost here. Install `gcc` and you can drop it. Measured on an
RTX 5080 with Krea 2 Turbo, 8 steps, weights offloaded between images:

| Size | Per image | Peak VRAM allocated |
| --- | --- | --- |
| 1024×1024 | 15 s | 8.1 GiB |
| 1440×1440 (2 MP) | 24 s | 8.6 GiB |

Leave `KREA_OFFLOAD` at `model`. With everything resident this checkpoint peaks
at 14.7 GiB for 1 MP and 18.7 GiB for 2 MP, which the driver quietly spills into
system memory — WSL spills just as Windows does — and the same images took 26 s
and 72 s.

## Headless

No server, no browser — same backend, same seeds, same gallery:

```powershell
uv run python -m backend.generate "a lighthouse in fog"
uv run python -m backend.generate "..." --aspect 16:9 --mp 4 --runs 12 --seed 42
uv run python -m backend.generate "..." --width 1024 --height 1024 --batch 4
uv run python -m backend.generate "..." --model krea/Krea-2-Turbo --quant int8
uv run python -m backend.generate --help
```

Steps, CFG and mu default to the checkpoint's profile exactly as the UI's
controls do, so the prompt is the only required argument. `--model`, `--quant`
and `--offload` are the `KREA_*` variables under another name. `--dry-run`
prints the resolved settings without loading the model; `--quiet` drops
everything but one `<path>  seed <n>` line per saved image.

Each image is written and reported as it lands, and `Ctrl-C` stops at the end of
the current step and keeps everything already saved — the same cooperative
cancel the Cancel button uses. Exit status is 0 when every requested image was
produced, 130 after a cancel, 2 for bad arguments, 1 for a failed load.

## Tests

```powershell
uv run --group dev pytest
```

No GPU and no weights: the pipeline and the scheduler are stubbed, and the
gallery runs against a temporary database.

## Sizes

Five aspect ratios across three megapixel tiers, picked independently — change
the aspect and the tier holds, change the tier and the framing holds.

| | 1 MP | 2 MP | 4 MP |
| --- | --- | --- | --- |
| 1:1 | 1024×1024 | 1440×1440 | 2048×2048 |
| 3:4 | 896×1200 | 1248×1664 | 1776×2368 |
| 4:3 | 1200×896 | 1664×1248 | 2368×1776 |
| 16:9 | 1360×768 | 1920×1088 | 2736×1536 |
| 9:16 | 768×1360 | 1088×1920 | 1536×2736 |

Every side is a multiple of 16, which is what the VAE wants. The whole 4 MP row
is within 0.3% of 2048×2048 in pixel count, so it costs the same VRAM at any
aspect — if the old 2K preset ran, all of these run. Both models are trained
around 1 MP, so 4 MP buys detail at some risk of repeated elements in the
composition.

## Batch vs runs

**Batch** produces N images in a single pipeline call, all resident in VRAM
together. It is the fastest way to get a handful of 1024s and the reason a 2K
batch does not fit at all.

**Runs** produces N images one pipeline call after another, stepping the seed by
the batch size each time, so a sweep costs no more VRAM than a single image.
That is what makes an unattended run of large images possible: set Runs to 12 at
2048×2048 and walk away. Leave the seed blank and the sweep starts from a random
one; set it and the whole sweep is reproducible.

**Cancel** appears next to the progress bar while a job is running. It stops at
the end of the current step — within a second or two — and keeps every image the
run had already finished.

## Settings

All optional, set as environment variables:

| Variable | Default | Notes |
| --- | --- | --- |
| `KREA_MODEL_ID` | `black-forest-labs/FLUX.1-Krea-dev` | `krea/Krea-2-Turbo`, or `krea/Krea-2-Raw` for the undistilled base (52 steps, CFG 3.5) |
| `KREA_QUANT` | `fp8` | `int8`; `none` for full bf16 (~25 GB VRAM); `nunchaku` for a pre-quantized Nunchaku Lite repo (Linux only, see above) |
| `KREA_OFFLOAD` | `model` | `sequential` (lower VRAM, slower) or `none` |
| `KREA_OUTPUTS_DIR` | `outputs/` in the repo | Where PNGs are written. `wsl/env.sh` moves it onto the Linux filesystem |
| `KREA_DB_PATH` | `krea.db` in the repo | The gallery index. Keep it off `/mnt/c` under WSL: SQLite cannot lock across the bridge |
| `KREA_PRELOAD` | `false` | Load weights at startup instead of on first request |
| `KREA_DEVICE` | `cuda` | |
| `KREA_HOST` | `127.0.0.1` | |
| `KREA_PORT` | `7860` | |
| `KREA_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Host headers the API answers to, and with the port, the origins allowed to write to it. Needed when `KREA_HOST` is `0.0.0.0`: the browser sends the machine's name or LAN address, never `0.0.0.0` |

Everything else in `backend/config.py` — the web directory, the size limits —
is code, not configuration.

The API refuses requests whose `Host` header is not on that list, and
state-changing requests announcing another origin. That is what stops a page on
some other site, whose own hostname resolves to 127.0.0.1, from reading the
gallery or deleting it.

## Defaults by checkpoint

Applied automatically; see `backend/model_defaults.py`.

| | Steps | CFG | Mu |
| --- | --- | --- | --- |
| Krea 2 Turbo | 8 | 0.0 | 1.15 |
| Krea 2 Raw | 52 | 3.5 | — |
| FLUX.1 Krea [dev] | 28 | 3.5 | — |
| FLUX.2 [klein] 4B / 9B | 4 | 1.0 (ignored) | — |
| FLUX.2 [klein] base 4B / 9B | 50 | 4.0 | — |

A negative prompt roughly doubles generation time, because it adds a second
forward pass per step. It takes effect above CFG 1 on FLUX, where
`guidance_scale` is a value baked into the model and only true CFG runs an
uncond pass, and above CFG 0 on Krea 2. FLUX.2 [klein] takes no negative prompt
at all, and the field greys out for it. The panel's hint says which applies to
the loaded checkpoint.

FLUX.2 [klein] (`black-forest-labs/FLUX.2-klein-4B`, Apache 2.0 and not gated;
the 9B and the undistilled `-base-` variants likewise) loads through the same
`KREA_MODEL_ID`. It is a 4-step model with an 8 GB Qwen3 text encoder, so under
`KREA_QUANT=fp8` the 4B fits with room to spare.

## Output

PNGs land in `outputs/` with generation parameters written into the PNG text
chunks (both an A1111-style `parameters` block and a `krea` JSON blob), so the
files stay self-describing outside the app. `krea.db` indexes them for search.

## Clearing history

```powershell
uv run python -m backend.clear                   # preview, then confirm
uv run python -m backend.clear --dry-run         # report only, change nothing
uv run python -m backend.clear --keep-favorites  # spare starred images
uv run python -m backend.clear --yes             # no prompt
```

Prints what it is about to remove — records, files, and any PNG with no row
left pointing at it — and asks for confirmation before deleting anything, then
vacuums the database. `--keep-favorites` spares starred images, but an orphaned
file has no row to carry a star and goes either way. Safe to run while the
server is up: rows go and the database file stays, so no open handle is
invalidated, and an image still being written is a `.png.part` that the orphan
sweep does not touch.

## Licence

Model weights are covered by the **Krea 2 Community License** — read it before any
commercial use. This application code carries no such restriction.
