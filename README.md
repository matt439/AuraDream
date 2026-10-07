# Local Diffusion

Run text-to-image models locally with a browser UI, live previews and a searchable
gallery. Supports FLUX, Krea 2, Z-Image, Qwen-Image, ERNIE-Image and a range of
anime checkpoints. Built with FastAPI, diffusers, vanilla JavaScript and SQLite;
the frontend has no build step.

[Quick start](#quick-start) · [Usage guide](docs/usage.md) ·
[Anime model guide](docs/anime-model-guide.md) · [GPU measurements](docs/verified.md)

## Requirements

- NVIDIA GPU. The tested configuration is an **RTX 5080 with 16 GB VRAM**;
  memory requirements depend on the checkpoint, size and batch. The defaults
  use FP8 quantization and model offload. See [measured results](docs/verified.md).
- **Python 3.13** and [uv](https://docs.astral.sh/uv/).
- Disk space for the selected model: about **32 GiB for the default model's
  weights**. Some checkpoints also need a converted copy; Illustrious, WAI
  and RouWei each use about 14 GB for the source and conversion together.

## Quick start

Run these commands from the repository root:

```powershell
uv sync
```

The default, **FLUX.1 Krea [dev]**, requires accepting the terms on its
[Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
and logging in with a read token:

```powershell
uv run hf auth login
uv run python -m backend.main
```

The terminal menu selects the checkpoint, quantization and offload mode.
Open **<http://127.0.0.1:7860>**, enter a prompt, choose a size and press
**Generate**. Steps, CFG and Mu start at the selected checkpoint's defaults.
The first generation downloads and loads the weights, which can take several
minutes; later loads reuse the cache. Starting the server alone does not load
the model unless preload is enabled.

Open checkpoints need no login. Select one in the menu, or pass its id:

```powershell
uv run python -m backend.main --model cagliostrolab/animagine-xl-4.0
uv run python -m backend.main --model Minthy/RouWei-0.8   # epsilon release
uv run python -m backend.main --model circlestone-labs/Anima@aesthetic-v1.1
uv run python -m backend.main --defaults   # use environment/defaults without the menu
uv run python -m backend.main --model krea/Krea-2-Turbo --quant int8 --offload model
```

Krea 2 Turbo and Raw require **manual approval** on their Hugging Face pages.
Any launch argument skips the menu; omitted settings come from the environment.
Non-interactive launches never ask. To change checkpoints, restart the server
with the new selection. The full catalogue is in [backend/launch.py](backend/launch.py).

RouWei 0.8 selects the author's epsilon checkpoint and starts at 28 steps/CFG 6.
Its first load converts the 6.94 GB single file on CPU. Start around 1 MP;
the [anime model guide](docs/anime-model-guide.md) explains its hybrid prompts,
artist placement and 77-token text limit.

Anima Aesthetic v1.1 starts at 30 steps/CFG 4. Its selector above loads the
official Aesthetic file from `circlestone-labs/Anima`; the bare repository id
still selects Turbo. The first load converts Aesthetic's weights on CPU and
caches them separately. Use tags, prose, or both; omit `score_*` tags from
both prompts. Mu, references, Refine, user LoRAs and Nunchaku are unavailable
for these Anima integrations.

For everyday use:

- **Examples** and the **?** hints explain the selected model's prompting and sizes.
  **Check** flags prompt issues; the optional local critic suggests revisions.
- **Batch** makes several images together; **Repeat** runs batches sequentially.
  **Grid search** varies settings, prompts and seeds, with resume support.
- Star favourites, reuse settings and compare images in the gallery or a contact
  sheet. PNGs retain their generation settings.
- **Ctrl/Cmd + Enter** generates; inside Grid search or Refine, it runs that action.

See the [usage guide](docs/usage.md) for these workflows, keyboard shortcuts,
search syntax and gallery maintenance.

## Run under WSL 2 (4-bit Nunchaku)

Nunchaku uses pre-quantized models and Linux-only kernels. From the repository
root in a **WSL shell**:

```bash
bash wsl/setup.sh                 # once: Linux environment and CUDA check
source wsl/env.sh                 # in every new shell
bash wsl/serve.sh                 # add --defaults to skip the menu
```

Use the same <http://127.0.0.1:7860> address in your Windows browser.
The WSL default is the Nunchaku Krea 2 Turbo checkpoint in NVFP4 format
(RTX 50 series). For RTX 20–40, set
`export LOCAL_DIFFUSION_NUNCHAKU_FORMAT=int4` before setup/sourcing the environment.
The menu lists the available pre-quantized checkpoints. Keep model offload
enabled on a 16 GB card.

**Source `wsl/env.sh` before any project `uv` command in WSL.** It keeps the
Linux virtualenv separate from Windows's `.venv`, and gives WSL its own gallery
under `~/local-diffusion`. Keep the gallery and model cache on the Linux
filesystem. Maintenance commands act on that shell's gallery; Git Bash is not WSL.
If setup installs uv, follow its printed instruction to add uv to the current shell.

Nunchaku mode downloads and runs Hugging Face kernels. An explicit
`DIFFUSERS_TRUST_REMOTE_KERNELS=false` disables that permission. See the
[WSL design notes](docs/design-notes.md#nunchaku-lite-pre-quantized-repos-linux-only)
for details.

## Sizes

Choose an aspect ratio and a megapixel tier independently. Custom dimensions
must be multiples of 16 and fit the app's limits. Start with the selected
model's size hint: these presets are app limits, not quality guarantees for
every checkpoint. Author example sizes do not establish a supported maximum.

| | 1 MP | 1.5 MP | 2 MP | 4 MP |
| --- | --- | --- | --- | --- |
| 1:1 | 1024×1024 | 1248×1248 | 1440×1440 | 2048×2048 |
| 7:9 | 896×1152 | 1120×1440 | 1264×1632 | 1808×2320 |
| 9:7 | 1152×896 | 1440×1120 | 1632×1264 | 2320×1808 |
| 3:4 | 896×1200 | 1104×1472 | 1248×1664 | 1776×2368 |
| 4:3 | 1200×896 | 1472×1104 | 1664×1248 | 2368×1776 |
| 2:3 | 848×1264 | 1056×1584 | 1184×1776 | 1680×2496 |
| 3:2 | 1264×848 | 1584×1056 | 1776×1184 | 2496×1680 |
| 4:7 | 768×1344 | 960×1680 | 1088×1904 | 1552×2704 |
| 7:4 | 1344×768 | 1680×960 | 1904×1088 | 2704×1552 |
| 9:16 | 768×1360 | 928×1664 | 1088×1920 | 1536×2736 |
| 16:9 | 1360×768 | 1664×928 | 1920×1088 | 2736×1536 |
| 5:12 | 640×1536 | 800×1920 | 928×2224 | 1328×3168 |
| 12:5 | 1536×640 | 1920×800 | 2224×928 | 3168×1328 |

Direct generation allows up to 3168 pixels per side and 4,207,104 pixels total.
Changing resolution changes the composition even with the same seed.

## Refine

To enlarge an existing image while retaining more of its composition, open its
**⤢** dialog and choose **Refine**. The pass resizes the source and denoises it
partway; lower Strength preserves more of the original. Start with the model's
own default (0.4 on SDXL, 0.1 on FLUX.1 [dev]). A changed aspect ratio crops
the source. Refine is available only on compatible checkpoints and can exceed
the direct-generation size limit. See [Refine usage](docs/usage.md#refine-an-existing-image)
and [measured results](docs/verified.md) before attempting large passes.

## Headless

Generate without a server or browser, using the same backend and gallery:

```powershell
uv run python -m backend.generate "a lighthouse in fog"
uv run python -m backend.generate "a lighthouse in fog" --aspect 16:9 --mp 1.5 --runs 4 --seed 42
uv run python -m backend.generate --help
```

The CLI loads its own weights. Unload the browser server's model first if it
holds the GPU. See [command-line usage](docs/usage.md#generate-from-the-terminal) for grids,
refinement, PNG settings, dry runs and exit codes.

## Settings

Set environment variables before starting the server or CLI. Paths below are
relative to the repository unless overridden; WSL uses its separate defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_DIFFUSION_MODEL_ID` | `black-forest-labs/FLUX.1-Krea-dev` | Checkpoint repository id |
| `LOCAL_DIFFUSION_QUANT` | `fp8` | `fp8`, `int8`, `none` (bfloat16), or `nunchaku` (Linux, pre-quantized model) |
| `LOCAL_DIFFUSION_OFFLOAD` | `model` | `model`, `sequential` (less VRAM, slower), or `none` |
| `LOCAL_DIFFUSION_DEVICE` | `cuda` | GPU device; e.g. `cuda:1` for the second card |
| `LOCAL_DIFFUSION_PRELOAD` | `false` | Load weights at startup |
| `LOCAL_DIFFUSION_OUTPUTS_DIR` | `outputs/` | Generated PNGs |
| `LOCAL_DIFFUSION_DB_PATH` | `local-diffusion.db` | Gallery database |
| `LOCAL_DIFFUSION_LORAS_DIR` | `loras/` | `.safetensors` adapters |
| `LOCAL_DIFFUSION_MAX_BATCH_PIXELS` | `12902400` | Pixel budget per pipeline call |
| `LOCAL_DIFFUSION_MAX_GRID_IMAGES` | `1024` | Image limit for a grid |
| `LOCAL_DIFFUSION_HOST` | `127.0.0.1` | Bind address |
| `LOCAL_DIFFUSION_PORT` | `7860` | Server port |
| `LOCAL_DIFFUSION_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Allowed hostnames/IPs; include the browser's destination when exposing the server on a LAN |
| `LOCAL_DIFFUSION_LOG_LEVEL` | `INFO` | Server logging; `DEBUG` includes tracebacks |
| `LOCAL_DIFFUSION_CRITIC` | `true` | Enable the optional local prompt critic |
| `LOCAL_DIFFUSION_CRITIC_SERVER` | unset | Use a local compatible `llama-server` binary |
| `LOCAL_DIFFUSION_CRITIC_MODEL` | unset | Use a local `.gguf` instead of downloading the pinned critic model |

Custom critic binaries must support the pinned server's flags and authentication;
see the [critic design notes](docs/design-notes.md#the-prompt-analyser).
All settings are defined in [backend/config.py](backend/config.py).

## Output

Images go to `outputs/`; `local-diffusion.db` indexes the gallery. Each PNG
contains its generation settings in a `local-diffusion` JSON chunk and an
A1111-style `parameters` block. Back up **both the PNGs and database** to retain
favourites and timings. See [gallery maintenance](docs/usage.md#clear-or-recover-a-gallery)
for clearing history, rebuilding a lost index or importing another gallery.

## Development

```powershell
uv run --group dev pytest
```

Tests use fake pipelines and a temporary gallery: no GPU or weight downloads.
For a browser smoke check, run the fake server and browser driver in separate
terminals (requires Node 22+ and Edge or Chrome):

```powershell
uv run python tools/fake_server.py --fresh
node tools/browser.mjs tools/smoke.json
```

Checkpoint and feature smoke recipes are in [AGENTS.md](AGENTS.md#tests).
GPU validation is recorded separately in [docs/verified.md](docs/verified.md).
Static frontend edits need no server restart; backend edits do.

## Documentation

- [Usage guide](docs/usage.md): prompts, batches, grids, gallery, CLI and recovery.
- [Anime model guide](docs/anime-model-guide.md): choosing a checkpoint and its settings.
- [GPU verification](docs/verified.md): measured behaviour and remaining gaps.
- [Design notes](docs/design-notes.md): implementation decisions and model sources.
- [Source map](docs/source-audit.md): module responsibilities.

## Licence

Model weights have their own licences. Consult the selected model's terms;
for WAI, use the [author's permissions](https://tensor.art/models/990778216270553015/WAI-illustrious-v17)
rather than the community mirror's metadata.
