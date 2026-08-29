# Krea Local

Run **Krea 2 Turbo** locally — 8-step, up to 4 megapixels, on a single consumer GPU.
FastAPI + diffusers backend, vanilla-JS frontend, SQLite gallery.

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
pending. While you wait, FLUX.1 Krea [dev] runs through the same app:

```powershell
$env:KREA_MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
uv run python -m backend.main
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

Open <http://127.0.0.1:7860>. The first generation downloads ~25 GB of weights and
loads them — expect several minutes. Subsequent loads read from the HF cache.

`Ctrl`/`Cmd` + `Enter` generates from anywhere in the page. With an image open,
`←` and `→` walk the strip it came from — the current run, or the gallery, which
pages in more as you reach the end of it — and `Esc` closes it.

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
| `KREA_MODEL_ID` | `krea/Krea-2-Turbo` | `krea/Krea-2-Raw` for the undistilled base (52 steps, CFG 3.5) |
| `KREA_QUANT` | `fp8` | `int8`, or `none` for full bf16 (~25 GB VRAM) |
| `KREA_OFFLOAD` | `model` | `sequential` (lower VRAM, slower) or `none` |
| `KREA_PRELOAD` | `false` | Load weights at startup instead of on first request |
| `KREA_PORT` | `7860` | |

## Defaults by checkpoint

Applied automatically; see `backend/model_defaults.py`.

| | Steps | CFG | Mu |
| --- | --- | --- | --- |
| Krea 2 Turbo | 8 | 0.0 | 1.15 |
| Krea 2 Raw | 52 | 3.5 | — |
| FLUX.1 Krea [dev] | 28 | 3.5 | — |

A negative prompt needs CFG above 1 and roughly doubles generation time, because
it adds a second forward pass per step.

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

Prints what it is about to remove and asks for confirmation before deleting
anything. It clears database rows and the PNGs behind them, sweeps any PNG with
no row left pointing at it, and vacuums the database afterwards. Safe to run
while the server is up — rows go, the database file stays, so no open handle is
invalidated.

## Licence

Model weights are covered by the **Krea 2 Community License** — read it before any
commercial use. This application code carries no such restriction.
