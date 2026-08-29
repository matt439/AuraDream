# Krea Local

Run **Krea 2 Turbo** locally — 8-step, up to 2048×2048, on a single consumer GPU.
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

`Ctrl`/`Cmd` + `Enter` generates from anywhere in the page.

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

## Licence

Model weights are covered by the **Krea 2 Community License** — read it before any
commercial use. This application code carries no such restriction.
