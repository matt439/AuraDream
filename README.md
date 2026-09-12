# Krea Local

Run **Krea 2 Turbo** or **FLUX.1 Krea [dev]** locally — up to 4 megapixels on a
single consumer GPU. FastAPI + diffusers backend, vanilla-JS frontend, SQLite
gallery. FLUX is the default because its repo is click-through gated; Turbo
needs approval first.

![stack](https://img.shields.io/badge/torch-2.13%2Bcu130-informational)
![stack](https://img.shields.io/badge/diffusers-0.40.0-informational)

## Requirements

- NVIDIA GPU, **16 GB VRAM minimum** (float8 path). 24 GB+ can run `KREA_QUANT=none`.
- Free disk varies by checkpoint: FLUX.1 Krea [dev] is 32 GiB (transformer
  23.8 GB, T5 9.5 GB), Krea 2 Turbo 34 GiB, and Animagine's Diffusers
  components total about 6.5 GiB. The WSL Nunchaku Lite re-uploads are far
  smaller.
- Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

`uv.lock` pins every wheel by sha256, the PyTorch ones included. The
`download.pytorch.org` index publishes no digest for the three torch 2.13
wheels or the Windows torchvision wheel, so `uv lock` cannot add one: those
four were downloaded, hashed and checked file-for-file against the installed
copies (the Windows venv and the WSL one) on 2026-09-08 and written into the
lockfile by hand. A `uv lock --upgrade` that moves torch re-resolves them from
the index and drops the digests; re-pin the same way.

## One-time setup

The Krea checkpoints, FLUX.1 Krea [dev] and FLUX.1 [dev] are **gated**
HuggingFace repos; the four FLUX.2 [klein] repos, FLUX.1 [schnell], Z-Image
Turbo, Qwen-Image, ERNIE-Image Turbo and Animagine XL 4.0 are open and need no
login (Apache 2.0 except Animagine's Open RAIL++-M; klein is the cheapest thing
here to try first - 4 steps, ~3 GiB peak). For the gated ones, log in first:

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

Open <http://127.0.0.1:7860>. The first generation downloads ~32 GB of weights
for the default checkpoint (the transformer and the text encoder both) and
loads them — expect several minutes. Subsequent loads read from the HF cache.

`Ctrl`/`Cmd` + `Enter` generates from anywhere in the page. With an image open,
`←` and `→` walk the strip it came from — the current run, or the gallery, which
pages in more as you reach the end of it — `F` stars it, `Delete` removes it
after asking, and `Esc` closes it. Both are also buttons under the image.

The image opens at the fit size. The mouse wheel zooms about the pointer,
dragging pans, `+`/`-` zoom about the centre, `0` fits again and `1` (or a
double-click) shows one image pixel per screen pixel — corrected for the
display scaling, so a sharpening sweep is judged on its own pixels rather than
on a 125% resample. The zoom and the crop carry across images of the same size
as you walk with the arrows, which is what makes a steps or mu axis comparable:
the same detail, at the same magnification, in every cell.

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
| `lite-infer/flux.1-dev-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` (also `flux.1-schnell`) | RTX 50 |
| `lite-infer/z-image-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` | RTX 50 |
| `lite-infer/Qwen-Image-nunchaku-lite-nvfp4_r32-bnb4-text-encoder` (12 GiB peak) | RTX 50 |
| `lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder` (no `_r32`; `pe/` loads only when Expand is used) | RTX 50 |
| the same names with `int4` in place of `nvfp4` | RTX 20–40 |

These are the official weights re-uploaded pre-quantized, and not gated. Any
other `KREA_QUANT` value works from WSL too; set it before sourcing `env.sh`,
and `env.sh` then leaves `KREA_MODEL_ID` to the ordinary default rather than
pairing a torchao quantizer with an already-quantized repo (which downloads
everything and then fails inside transformers). `KREA_NUNCHAKU_FORMAT=int4`
picks the older cards' repos.

Two things to know. The WSL gallery is separate from the Windows one — two
databases, two output directories. And run `uv` from WSL only after sourcing
`wsl/env.sh`: without it, uv finds the Windows `.venv` in the project directory,
judges it broken, and replaces it.

The same goes for the command-line tools. `backend.clear`, `backend.stats`,
`backend.reindex` and `backend.generate` act on whichever gallery the
environment names, so to reach the WSL one run them from a WSL shell — the
Ubuntu tab in Windows Terminal, or `wsl` from a PowerShell prompt — after the
same `source wsl/env.sh`:

```bash
cd /mnt/c/VSProjects/ImageGeneratorKrea
source wsl/env.sh
uv run python -m backend.clear
```

Git Bash is not a WSL shell: there `/mnt/c` does not exist, yet `env.sh` still
runs, and `uv` builds a second venv and an empty gallery under
`C:\Users\<you>\krea-local` and reports nothing to clear. The `database :`
line the tools print is the check — it should start with `/home/`.

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

[Grid search](#grid-search) is here too, one flag per axis. Both sides expand
the same specs into the same cells in the same order, so a sweep started in the
terminal and one started in the browser cannot drift apart:

```powershell
uv run python -m backend.generate "{size} dog sitting on {ground}" `
  --grid-prompt "size = big | small | none" --grid-prompt "ground = grass | dirt" `
  --grid-steps 6-12x2 --grid-seeds 4 --seed 42 --resume
uv run python -m backend.generate "..." --grid-steps 6-12x2 --seed-list 4471,8823
```

Steps, CFG and mu default to the checkpoint's profile exactly as the UI's
controls do, so the prompt is the only required argument. `--true-cfg` is the
panel's True CFG box (FLUX.1 only; see [Defaults by
checkpoint](#defaults-by-checkpoint)). `--model`, `--quant` and `--offload`
are the `KREA_*` variables under another name. `--dry-run` prints the
resolved settings without loading the model; `--quiet` drops everything but
one `<path>  seed <n>` line per saved image.

`--reference 12,15` conditions the run on gallery images by id (klein only;
see [Reference images and LoRA
adapters](#reference-images-and-lora-adapters)), and `--lora style.safetensors
--lora-scale 0.8` applies an adapter from `loras/`, with `--grid-lora-scale
0-1x0.25` sweeping the weight. Both are refused before the load, by name,
when the checkpoint or the installation cannot use them: a night that renders
every cell without the thing you asked for looks like a success.

`--from-image outputs/….png` starts from an image you already have: the
prompt, negative prompt, size, steps, CFG, mu, seed, checkpoint, references
and adapter come out of the PNG's own metadata, and any flag you give
explicitly wins over it. The
quantization is not carried - a WSL Nunchaku PNG read on Windows would ask for
a mode this machine cannot run - and the tool says so when it differs, since
the pixels then will too.

The CLI loads its own copy of the weights, so before it does it asks the
server on `KREA_HOST:KREA_PORT` whether one is already on the GPU. If the
server answers *ready* or *loading* (or does not answer in time), the CLI
stops with exit status 4 and says how to free the card; nothing listening is
the ordinary case and it carries on. Advisory only - it cannot see a server on
the other side of the WSL boundary - and `--ignore-server` overrides it.

Each image is written and reported as it lands, and `Ctrl-C` stops at the end of
the current step and keeps everything already saved — the same cooperative
cancel the Cancel button uses. A cell that fails - a transient OOM, a locked
database - is skipped and reported and the next one runs, up to three in a row
or a third of the run. Exit status is 0 when every requested image was produced,
130 after a cancel, 2 for bad arguments, 1 for a failed load or when any cell
was skipped, 4 when a server already holds the GPU. `python -m backend.clear`
exits 1 when it aborted with nothing deleted and 3 when the deletion ran and
something went wrong afterwards.

### Which settings did you keep?

```powershell
uv run python -m backend.stats                    # the newest run
uv run python -m backend.stats --run ab12cd       # one grid, however many jobs
uv run python -m backend.stats --by steps,cfg,prompt --min 3
uv run python -m backend.stats --all              # the whole gallery, indicative
```

The starred *rate* per setting - made, starred, percent - rather than the
count, which only rewards whichever setting you used most. The default scope
is the newest run, where the grid crossed its axes against the same prompts
and seeds and the comparison is fair; `--all` is confounded by whatever was in
use while the prompts happened to be good and says so. Settings with fewer
images than `--min` (5) are left out, so a lucky 1-of-1 does not lead. The
column says *starred*, not *kept*: nothing records a rejection, so an
untouched night reads 0%, not unknown.

## Tests

```powershell
uv run --group dev pytest
```

No GPU and no weights: the pipeline and the scheduler are stubbed, and the
gallery runs against a temporary database.

The browser side has no test suite; it is checked in a real browser instead:

```powershell
uv run python tools/fake_server.py          # the app on :7861, fake images, throwaway gallery
node tools/browser.mjs tools/smoke.json     # headless Edge/Chrome through the UI; exit 1 on a failure
uv run python tools/fake_server.py --port 7862 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-flux.json      # the True CFG box, on a FLUX-shaped server
uv run python tools/fake_server.py --port 7863 --step 0.3
node tools/browser.mjs tools/smoke-preview.json   # the live preview, on a server slow enough to show one
uv run python tools/fake_server.py --port 7865 --model baidu/ERNIE-Image-Turbo
node tools/browser.mjs tools/smoke-ernie.json     # the explicit prompt expansion editor flow
uv run python tools/fake_server.py --port 7866 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-animagine.json # tag guide, defaults, size range and disabled Mu
uv run python tools/fake_server.py --port 7867 --fresh
node tools/browser.mjs tools/smoke-paging.json    # the gallery paging itself on scroll; needs the fresh gallery
```

The fake server draws gradients stamped with seed and steps, so grid cells can
be told apart on a sheet, behaves as the claimed checkpoint's profile predicts
(the negative prompt's threshold, whether real CFG and Mu apply, and whether
prompt expansion is available), and
`--step 0.3` slows each denoising step enough to watch Pause, the queue and
the preview work. `tools/browser.mjs` needs only Node 22+ and an installed Edge
or Chrome (`KREA_BROWSER` points at another executable); each run gets a
profile directory of its own, so a crashed run cannot lock the next one out.

## Sizes

Thirteen aspect ratios across four megapixel tiers, picked independently — change
the aspect and the tier holds, change the tier and the framing holds. After 1:1
each portrait framing and its landscape inverse share one box, pairs from
nearest-square to widest. The orientation button switches the selected framing
to the other half of its box at the same tier; it is disabled for 1:1.

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

Every side is a multiple of 16, which is what the VAE wants. The 1.5 MP row and
the aspect ratios are the ones the model cards asked for: the 1.5 MP
row's non-square sizes are Qwen-Image's own README shapes (its 3:2 pair, at
1.67 MP, is the top of the row), the 3:2 pair at 1 MP is ERNIE-Image's, and
Animagine's 9:7, 7:4 and 12:5 pairs are exact at 1 MP. Higher tiers preserve
those framings at roughly the tier's pixel count. The 1.5 MP row sits inside
the 1024-1280 px (1-1.64 MP) FLUX.1 Krea [dev]'s
README recommends, which the 2 MP row is past. The whole 4 MP row
is within 0.4% of 2048×2048 in pixel count, so it costs the same VRAM at any
aspect — if the old 2K preset ran, all of these run. Every checkpoint here
is trained around 1 MP and documented for something between 1 and 4; the
? badge beside Size says what, and the line under the boxes warns when the
size or a finite model-card aspect is outside it (see
[Prompting guide](#prompting-guide)). 4 MP buys
detail at some risk of repeated elements in the composition. A hand-typed
size is accepted up to the largest preset - 3168 on
a side, 3168×1328 in pixels - and refused past it; the table in
`backend/sizes.py` is the one source, and the CLI, the browser and this README
are held to it by a test.

## Batch vs runs

**Batch** produces N images in a single pipeline call, all resident in VRAM
together. It is the fastest way to get a handful of 1024s and the reason a
large 2K batch does not fit. The default guard allows every 1 MP preset at
Batch 12 and two 4 MP images; because real usage varies by checkpoint and
quantizer, `KREA_MAX_BATCH_PIXELS` can override it.

**Runs** produces N images one pipeline call after another, stepping the seed by
the batch size each time, so a sweep costs no more VRAM than a single image.
That is what makes an unattended run of large images possible: set Runs to 12 at
2048×2048 and walk away. Leave the seed blank and the sweep starts from a random
one; set it and the whole sweep is reproducible.

**Cancel** appears next to the progress bar while a job is running. It stops at
the end of the current step — within a second or two — and keeps every image the
run had already finished.

**Live preview**: while a cell denoises, a small picture of it appears in the
viewer from a few steps in - a linear read of the latents the step callback
already carries, blurry but honest about the composition, refreshed a few
times a second and replaced by the real image when it lands. On FLUX.1 and
Krea 2; FLUX.2 [klein]'s pipeline gives the callback no way to unpack its
latents, so it shows none. With Cancel landing within a step, a prompt that
is going wrong costs five seconds rather than the render.

**Pause** holds the run at the end of the current cell and hands the GPU back:
between cells the weights are off the card under the default offload, and what
it still holds is the allocator's cache, which is returned. The button reads
*Resume* once the worker has actually stopped — until then a denoise may still
be running — and resuming costs nothing but the next cell's own transfers. The
weights stay loaded, so Unload is still refused while a job is paused.

**Queueing**: Generate and Run grid stay live while a job runs. A second press
queues the job behind the current one on the server; a line under the bars
counts what is waiting, with a button to cancel all of it, and the page moves
on to the next job when the watched one ends. A reload attaches to whichever
job holds the GPU, then to the oldest one waiting.

## Grid search

Folded away under the panel — a sweep over the things that make the image, run
as one job and left overnight. Every axis sits on top of the settings above it,
so a blank axis contributes the single value already in the panel and only what
you fill in varies. An empty grid is exactly the image Generate would have made.

| Axis | Takes | Example |
| --- | --- | --- |
| Prompt variables | `name = value \| value` lines, one variable per line | `size = big \| small` |
| Steps | one value, a list, or a range | `6-12x2` — 6, 8, 10, 12 |
| CFG | the same, decimals allowed | `0,0.5,1` or `0-2x0.5` |
| Mu | the same, plus `none` for no shift of ours at all | `none,1.15` |
| Seeds | how many consecutive seeds run the whole grid | `4` |
| Seed list | exactly these seeds instead, a list or a range | `4471, 8823` |
| LoRA weight | the same syntax, needing an adapter selected | `0-1x0.25` |
| Budget | minutes; start no cell after this long | `420` |

They cross: two sizes, two grounds, four step counts and two seeds is 32 images.
The panel previews what it would run before you commit to it — the resolved
axes, the finished prompts, the image count, and once the gallery holds anything
at that size to measure against, roughly how long and roughly how many
gigabytes of PNG. The top bar shows the free space on the images volume, and
the preview turns amber when the estimate is more than that: a night that
fills the disk dies at the first save. It also counts the cells that would
come out *identical* once the pipeline's own drops are applied - a variable
that lives only in the negative prompt on a checkpoint that ignores it, a CFG
axis that never clears the threshold - and says so in amber, because the run
would otherwise spend the night rendering the same picture (`--dry-run`
prints the same warning). One grid is capped at `KREA_MAX_GRID_IMAGES` images
and each axis at 64 values, because `1-1000` is eight characters.

### Reference images and LoRA adapters

Two inputs that are neither a prompt nor a number, both greyed out where they
cannot apply — for the checkpoint in one case, for the installation in the
other, with the reason in the badge beside the control.

**Reference images** are what FLUX.2 [klein] is best at and nothing else here
can do: its pipeline takes them as its first argument, resizes each to 1 MP,
encodes it and concatenates the tokens onto the latent sequence at every step —
the same character in a new pose, a sketch to finish, one composition against a
prompt axis. Add one with the ❐ button on any gallery image (up to four), or
`--reference 12,15` from the terminal. They are chosen by *id*, never by path:
the server resolves them through the gallery, so nothing a request says can
read a file the app did not write.

They cost less than you would think. Attention is quadratic in the sequence
length and each reference roughly doubles it, but measured here (klein 4B,
fp8, model offload, 1 MP, 4 steps) one reference ran in 8.2–9.2 s against
10.2 s for the same prompt without one, and two in 11.3 s: under model offload
the per-image weight transfer dominates, and the longer sequence disappears
into it.

**LoRA adapters** are read from `loras/` — drop a `.safetensors` file in and it
appears in the select (`KREA_LORAS_DIR` moves the directory). One adapter at a
weight, swapped on the live pipeline between cells, which is what makes
`0-1x0.25` an ordinary sweep rather than five reloads: the weights load once
and each cell is a re-weighting. Both the adapter and the weight are recorded
on the row and in the PNG, so Reuse restores them and a resumed grid tells a
0.5 cell from a 0.75 one. Not available under `KREA_QUANT=nunchaku`: those
checkpoints arrive quantized to 4-bit SVDQuant and their linear layers are not
what an adapter attaches to — the server says so rather than failing at the
first image.

### Varying the prompt

Put `{name}` placeholders in the prompt and give each one its values:

```
Prompt             {size} dog sitting on {ground}

Prompt variables   size = big | small | none
                   ground = grass | dirt
```

Six prompts, crossed with whatever the other axes do. Values split on `|` rather
than on a comma, because `a big, fluffy dog` is one value and splitting it the
other way makes two cells of nonsense.

`none` is a value too — the word and the space it sat in both come out, so the
third size is `dog sitting on grass`. That is how you compare saying something
against not saying it at all.

A variable no placeholder mentions is refused, and so is a placeholder with no
variable: the first multiplies the night by a value that changes nothing, the
second renders a literal `{ground}` into every image of the run. Both are caught
when you submit, not at 3am.

### Overnight, and picking it back up

**Seeds are the outer loop.** A night that ends early — a cancel, a crash, a
power cut — has then finished every combination for the seeds it reached, which
is the comparison you wanted; the other order leaves every seed part-done and
nothing comparable. The prompt is the next loop in, for the same reason.

**The panel remembers itself** in the browser: the prompt, the size, the seed,
the grid axes and the variable block come back after a reload or a restart,
and steps, CFG and mu come back when the server is running the checkpoint
they were set on. Resume is the one box that never comes back ticked - a
resume you did not mean is a night that silently skips cells. Per browser,
per origin, and dropped without complaint where storage is blocked. **Reset**,
at the end of the Prompt caption, is the way out: every box back to the
checkpoint's defaults and the memory forgotten. Clearing the gallery does not
do this — that is a deletion on the server, and the browser's memory is its
own.

**Budget** is how you use that. Seeds being outermost was engineered so that a
night which ends early has finished every setting for the seeds it reached -
and every way of ending early was an accident. Give the grid more than the
night can do, set Budget to the minutes you have (`--budget` in the
terminal; counted from the start, load included), and it starts no cell after
that. The stop is reported as the budget working, not as a cancel, and Resume
finishes what was not reached.

**Seed list** is the second night: the two seeds that gave the composition you
want, crossed with a finer sweep. `4471, 8823` or `100-103` runs the whole grid
at exactly those seeds, in place of the count and the Seed box (`--seed-list`
in the terminal), and a resume of such a grid needs no base seed - the list is
the record.

While a job runs, the line under the progress bars says how long it has been
going and roughly how long is left, from the job's own pace so far - which is
what shows a run that has slowed down. When it finishes, the same line says
how many images it made and in how long - and how many of them were distinct
pictures, when that is fewer. A job waiting for the weights says
so, rather than claiming to be behind another job; **Load** in the top bar
starts (or, after a failed load, retries) the download without queueing an
image to do it.

**Resume** skips the cells the gallery already holds: tick it and re-submit the
same grid. It matches on the settings that make the image rather than on the job
that ran them, so a *new* job continues a dead one and an image rendered by hand
earlier counts too. Each prompt of a prompt grid counts on its own. It needs the
Seed field filled in with the seed the grid was *started* from — a random base
would match none of the run it is continuing and would quietly render the whole
thing again. Run grid writes that seed into the box when it was left blank, and
the queued job's response carries it. Failing both, the *oldest* image of the run
has it: every image records a derived seed, the newest is the last seed the night
reached, and with a batch above 1 only the first image of each batch is a cell's
base.

Every image also records a **run id** derived from the grid's definition, so a
night that was resumed once or three times is still one run: the card's ⊞
button and the `run:` search term show the whole matrix, where `job:` shows one
submission of it.

**Cancel** lands within a step, keeps everything already written, and a grid
that is entirely finished is refused rather than queued to do nothing.

### Reading a run back

Every image records the job that made it — a browser job or one CLI invocation —
and the gallery search box is how you get one back. It takes `key:value` terms
plus free text:

```
run:ab12cd        one grid, however many times it was resumed
job:ab12cd        one submission of it
seed:42           steps:6-12       cfg:<=1        model:flux      quant:fp8
tcfg:1.5          the scale the negative prompt ran at; tcfg:none for images it did not
mu:none           the images that ran with no shift at all — not the same as mu:0
hash:3f9a         every image with the same pixels (a prefix of the hash will do)
date:today        a day on your clock: today, yesterday or 2026-09-07
a lighthouse      anything left over searches the prompt
```

`date:` is a local day, not a UTC one: an overnight run crosses midnight in
every timezone, and the day you mean is the one on your calendar.

Every image records a hash of its *pixels* (16 hex digits of a sha256, in the
row and in the PNG), which is the one thing the table never knew before:
anything about what came out. So "did the mu axis change anything" is a
count rather than a squint: the Runs picker lists a run as `96 img (48
distinct)` when it made fewer pictures than images, the line under the panel
says the same when a job finishes, and `hash:` finds every image identical to
one. It rests on the pipeline being deterministic, which was checked on the
card: the same request twice through the fp8 FLUX path is the same bytes.

**Re-render** appears beside the filters whenever a search or the Favorites
tick narrows the gallery: it renders everything the filter matches again at
the size set in the panel, one cell per image with its own prompt, steps, CFG,
mu and seed, as one job. Star the winners of a 1 MP night, tick Favorites, pick
4 MP, click. Cells already rendered at that size are skipped, so it can be
resumed like a grid, and a selection made on another checkpoint is refused
rather than run with the wrong steps and CFG (`model:` narrows it). Be honest
with yourself about what comes back: the pipeline draws its noise at the new
shape, so the same seed at 4 MP is a new sample - the prompt and the settings
carry over, the exact picture does not.

**Runs**, the picker beside the search box, lists the gallery one line per run
- when it started, how many images, how many starred, the prompt - newest
first; picking one searches for it and opens the sheet. And a page opened
after a run has ended says so under the panel: when it finished or stopped,
how many images and cells it reached, and the error if there was one - the
server keeps that for as long as it is up, and the note is shown once.

**Sheet**, above the gallery, lays that run out as the grid that made it: the
two widest-varying axes become the columns and rows, and a third becomes a
selector rather than a dimension nobody can read. Cells take the run's own
aspect ratio and show the whole frame, and a starred cell carries its star, so
where the keepers cluster is visible on the sheet rather than only in your
head. A prompt grid's images record which variable took which value, so on
the sheet each variable is an axis of its own - `size` across, `ground` down -
and a `none` value reads as `none`; images from before that was recorded fall
back to the single Prompt axis below. **Export PNG** downloads the sheet as
one image - labels, stars and a caption naming the search, the pins, the
checkpoint and the run - which is the artefact worth keeping from a night
and the one a screenshot cannot give you whole. The Prompt axis is labelled
with only the words that vary — `big grass`, `small dirt` — when every prompt
has the same word count. A `none` value makes one prompt shorter than the
others; the labels then keep what is left once the shared head and tail come
off, which is the whole phrase again when a second variable also varies (the
worked example above, with `none` in it, labels each column with its full
prompt). The trimming exists because a header repeating the whole sentence is
unreadable.

## Settings

All optional, set as environment variables:

| Variable | Default | Notes |
| --- | --- | --- |
| `KREA_MODEL_ID` | `black-forest-labs/FLUX.1-Krea-dev` | A catalogue id such as `krea/Krea-2-Turbo` or `cagliostrolab/animagine-xl-4.0` |
| `KREA_QUANT` | `fp8` | `int8`; `none` for full bf16 (~25 GB VRAM); `nunchaku` for a pre-quantized Nunchaku Lite repo (Linux only, see above) |
| `KREA_OFFLOAD` | `model` | `sequential` (lower VRAM, slower) or `none` |
| `KREA_OUTPUTS_DIR` | `outputs/` in the repo | Where PNGs are written. `wsl/env.sh` moves it onto the Linux filesystem |
| `KREA_DB_PATH` | `krea.db` in the repo | The gallery index. Keep it off `/mnt/c` under WSL: SQLite cannot lock across the bridge |
| `KREA_LORAS_DIR` | `loras/` in the repo | Where `.safetensors` adapters are read from. A request names one by file name; nothing joins a path from a request onto it |
| `KREA_MAX_BATCH_PIXELS` | `12902400` | Total pixels allowed in one pipeline call; covers Batch 12 across the 1 MP preset row and two images across the 4 MP row |
| `KREA_MAX_GRID_IMAGES` | `1024` | The most images one grid search may queue — a guard rail against a typo becoming a week of GPU time |
| `KREA_PRELOAD` | `false` | Load weights at startup instead of on first request |
| `KREA_DEVICE` | `cuda` | `cuda:1` puts the weights on the second card, offload included, and the status line reports that card |
| `KREA_HOST` | `127.0.0.1` | |
| `KREA_PORT` | `7860` | |
| `KREA_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Host headers the API answers to, and with the port, the origins allowed to write to it. Needed when `KREA_HOST` is `0.0.0.0`: the browser sends the machine's name or LAN address, never `0.0.0.0` |

Everything else in `backend/config.py` — including the per-image size limits —
is code, not configuration.

The API refuses requests whose `Host` header is not on that list, and
state-changing requests announcing another origin. That is what stops a page on
some other site, whose own hostname resolves to 127.0.0.1, from reading the
gallery or deleting it.

## Prompting guide

Each checkpoint wants a different kind of prompt and vouches for a different
range of sizes, so the panel says which: the **?** beside Prompt is the
checkpoint's prompting guide, the **?** beside Size its sizing, the line
under the width and height warns when a size is outside what its authors
documented, and **Examples** drops down the authors' own prompts - click one
and it fills the box. Tag-caption checkpoints can also show **Tags**; for
Animagine it exposes only the model card's person-count, rating, quality, score
and year controls, inserts at the prompt caret, finishes with the recommended
quality suffix, and can fill the recommended negative prompt. While typing an
Animagine prompt, autocomplete comes from Danbooru's live tag index and each
comma-separated item gets a green recognised, red unknown, or grey unchecked
chip. On a wide page the full validation list uses the results column, so it
does not move Generate and never gets its own scrollbar. Drag chips to reorder
the canonical comma-separated prompt; use each × or the Delete key to remove
one. Right-click a gap to insert at that boundary, or a chip to insert after it,
with inline autocomplete. The lookup is advisory
and never blocks generation; if the network is
unavailable, ordinary tags remain unchecked rather than being called invalid.
Only tag text is sent to the index, never images or generation settings. ERNIE
also shows
**Expand**: it runs the checkpoint's
rewriter once at the selected width and height and puts the result in the
textarea for review, with a local English translation underneath. The Chinese
text remains the generation prompt; the translation is a reading aid and is
hidden as soon as that prompt is edited. The first use downloads and loads the
separate 7.7 GB rewriter plus Marian's 312 MB Chinese-to-English weights; later
uses reuse both until Unload. Image generation never rewrites silently, so the
visible Chinese prompt is still the prompt stored with the image.
All of it is sourced from the model cards, READMEs and
prompting guides (the URLs are in `docs/design-notes.md`), none of it
invented. In brief:

| | Prompt | Encoder, limit | Size |
| --- | --- | --- | --- |
| FLUX.1 Krea [dev] | Descriptive sentences, subject first, photo language; text in quotes | CLIP 77 + T5 512 tokens | 1024-1280 px sides recommended (1-1.6 MP); 0.1-2 MP documented |
| FLUX.1 [dev] | The same, 30-80 words; no negative prompt without True CFG | CLIP 77 + T5 512 | 0.1-2 MP documented; above that users report doubling |
| FLUX.1 [schnell] | The same, shorter | CLIP 77 + T5 256 | 0.1-2 MP documented |
| FLUX.2 [klein] | Subject, action, style, context; every rendered word in quotes; no negative prompt at all | Qwen3, 512 | 1024x1024 in every example; up to 2 MP recommended, 4 MP max |
| Krea 2 Turbo | One natural paragraph, long and detailed best, short works; text in quotes | Qwen3-VL, 512 | 1k-2k documented (1-4 MP); pretraining topped out at 1024 px |
| Krea 2 Raw | The same; an undistilled base Krea does not recommend for inference | Qwen3-VL, 512 | Trained to 1k: 1 MP |
| Z-Image Turbo | Long, plain, objective description, English or Chinese; text in quotes | Qwen3-4B, 512 | 0.26-4.2 MP documented, trained to the 1k-1.5k range |
| Qwen-Image | Dense caption in colour/shape/texture/quantity/text terms; append ", Ultra HD, 4K, cinematic composition."; True CFG 4 with a negative prompt | Qwen2.5-VL, 512 | Trained to 1328x1328; the README's shapes are 1.5-1.8 MP |
| ERNIE-Image Turbo | Expand a short prompt, then review its detailed Chinese paragraph and local English translation | Ministral-3, 2048 | Seven 1 MP shapes listed, nothing above |
| Animagine XL 4.0 Opt | Danbooru tags: person count, character and series, rating, subject/pose/scene, then quality tags; use its negative prompt | CLIP + CLIP, 77 tokens | Trained at 1024x1024; nine recommended shapes around 1 MP |

## Defaults by checkpoint

Applied automatically; see `backend/model_defaults.py`.

| | Steps | CFG | Mu |
| --- | --- | --- | --- |
| Krea 2 Turbo | 8 | 0.0 | 1.15 |
| Krea 2 Raw | 52 | 3.5 | — |
| FLUX.1 Krea [dev] | 28 | 3.5 | — |
| FLUX.2 [klein] 4B / 9B | 4 | 1.0 (ignored) | — |
| FLUX.2 [klein] base 4B / 9B | 50 | 4.0 | — |
| FLUX.1 [dev] | 28 | 3.5 | — |
| FLUX.1 [schnell] | 4 | 0.0 (ignored) | — |
| Z-Image Turbo | 8 | 0.0 | — |
| Qwen-Image | 50 | 1.0 (ignored; True CFG 4) | — |
| ERNIE-Image Turbo | 8 | 1.0 | — |
| Animagine XL 4.0 Opt | 28 | 5.0 | — |

A negative prompt roughly doubles generation time, because it adds a second
forward pass per step. On Krea 2 it takes effect above CFG 0, on Krea 2 Raw
likewise; FLUX.2 [klein] takes no negative prompt at all, and the field greys
out for it.

**FLUX.1 is different**: its CFG box is the *distilled guidance value* the
model takes as an input, not classifier-free guidance, and the negative prompt
is driven by a fourth box, **True CFG** - real CFG kept apart from the
guidance value, blank meaning off. Above 1 the pipeline runs the second,
unconditioned pass against the negative prompt; the two usual recipes are
the trained 3.5 with a mild True CFG of 1.5, or the guidance value dropped to
about 1 with True CFG doing the work at 3-5. The two used to be one number,
which made a negative prompt typed at 3.5 stack guidance twice at double the
cost and left the other recipe unreachable. On every other checkpoint the box
is greyed out - its CFG *is* real CFG - and the panel's hints say which
applies. Whatever the checkpoint, the record carries the scale the negative
prompt actually ran at (`tcfg:` in the search box, `True CFG:` in the PNG's
parameters line), and none when it did not; a record from before that was
written reads as its CFG, which is what ran then.

FLUX.2 [klein] (`black-forest-labs/FLUX.2-klein-4B`, Apache 2.0 and not gated;
the 9B and the undistilled `-base-` variants likewise) loads through the same
`KREA_MODEL_ID`. It is a 4-step model with an 8 GB Qwen3 text encoder, so under
`KREA_QUANT=fp8` the 4B fits with room to spare.

FLUX.1 [dev] and FLUX.1 [schnell] (`black-forest-labs/FLUX.1-dev`, gated under
the FLUX.1 non-commercial licence; `black-forest-labs/FLUX.1-schnell`, Apache
2.0 and open) are FLUX.1 Krea's own pipeline, so everything said about it above
holds - the CFG box is the distilled guidance value and True CFG drives the
negative prompt. schnell's transformer has no guidance input at all, so there
the CFG box does nothing and only True CFG, blank by default, ever reaches the
model; it resolves in 4 steps. Measured under Nunchaku NVFP4 on the 5080,
model offload: dev 1 MP in 30 s and 2 MP in 48 s at 7.4 / 7.8 GiB peak;
schnell 1 MP in 16-23 s and 2 MP in 25 s at the same peaks. Most of a
schnell image is the per-image cost that does not scale with steps - the text
encoders under bitsandbytes NF4 and the weight transfer - so it is not the
speed win here that Z-Image Turbo is.

Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`, Apache 2.0 and open) is a 6B
model that resolves in 8 steps and, like Krea 2, guides from 0 up, so any CFG
above 0 brings the negative prompt in. It ships the FLUX VAE, so the live
preview works for it unchanged. Measured under Nunchaku NVFP4 on the 5080
(`lite-infer/z-image-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`): 1 MP
in about 8 s at 4.1 GiB peak, 2 MP in about 14.5 s at 4.6 GiB, an 11 s load
from cache.

Qwen-Image (`Qwen/Qwen-Image`, Apache 2.0 and open) is the 20B model Krea 2
was built from: the same VAE, so the live preview works, and FLUX.1's
arrangement of the controls - the CFG box never reaches the transformer and
True CFG drives the negative prompt, which the pipeline runs **only when a
negative prompt is given**, so type one (a word will do). It wants 50 steps at
True CFG 4. Only the 4-bit re-upload fits this card, and only just: measured
under Nunchaku NVFP4 on the 5080
(`lite-infer/Qwen-Image-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`, 12 GB of
transformer), 11.75 GiB peak at 1 MP and 12.1 GiB at 2 MP, about 80 s per
1 MP image and 150 s at 2 MP once warm, 210 s for the first of a session.

ERNIE-Image Turbo (`baidu/ERNIE-Image-Turbo`, Apache 2.0 and open) is Baidu's
8B single-stream model, distilled to 8 steps at guidance 1.0 under the classic
gate: a negative prompt applies above CFG 1, and there is no True CFG knob. Its
repo ships a 7.7 GB prompt rewriter. Ordinary loading leaves it out; the
explicit **Expand** button loads it on first use and places its sampled result
back in the prompt box. A CPU-only
[`Helsinki-NLP/opus-mt-zh-en`](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
Marian model (312 MB of PyTorch weights, CC-BY-4.0) translates that result for
display underneath; the translation is not sent to ERNIE or recorded with the
image. Generation itself keeps the pipeline switch off, so the reviewed
Chinese prompt you see is the prompt the model sees and the seed brings the
image back. The
pipeline takes no schedule from outside, so a mu is dropped and recorded as
dropped, and it uses FLUX.2's VAE, for which no live-preview map has been
fitted yet. Baidu lists 1 MP shapes only, but 2 MP and 4 MP render. Measured
under Nunchaku NVFP4 on the 5080
(`lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder`), with
Z-Image Turbo re-measured the same evening: 1 MP in about 10 s at 5.2 GiB
peak (Z-Image 10.6 s at 4.1 GiB), 2 MP in 24 s at 5.9 GiB (Z-Image 13 s at
4.6 GiB), a 4 MP pair in one call in 49 s at 10.3 GiB allocated and 12.0 GiB
reserved (Z-Image 23.5 s, 7.4 / 11.8 GiB); about 22 s to load. Under Nunchaku
a seed does not reproduce bit-for-bit on any checkpoint, and on ERNIE two runs
of the same seed differ visibly rather than in noise; the fp8 path on Windows
reproduces.

Animagine XL 4.0 Opt (`cagliostrolab/animagine-xl-4.0`, CreativeML Open
RAIL++-M and open) is the anime-specific checkpoint: an SDXL finetune trained
at 1024x1024 on 8.4 million anime-style images. Its prompt is a short,
comma-separated tag list rather than the natural-language paragraphs the other
checkpoints here want. Put the person count, character and series first, finish
with `masterpiece, high score, great score, absurdres`, and use the negative
prompt in the Prompt help badge. It recommends 28 steps at CFG 5. The repo's
2025-02-13 Opt update replaced the Diffusers UNet as well as adding the root
single-file checkpoint, so the normal loader gets Opt; it also changed the
manifest from Euler Ancestral to Euler, which this app follows. Its nine
recommended aspects are all available as 1 MP presets; 9:7, 7:4 and 12:5 plus
their portrait opposites use the model card's exact dimensions, and every one
also has 1.5, 2 and 4 MP choices. The **Tags** picker inserts the card's special
tags without hiding or rewriting the editable prompt. Animagine has no
flow-matching Mu,
reference-image input or fitted live-preview map, and no Nunchaku Lite build.
The pipeline interface and dry-run are covered automatically, and a user run
on this card loaded the weights and produced good images. Its exact quantizer
and offload mode were not recorded, so speed, VRAM and fp8 behavior remain
unmeasured.

## Output

PNGs land in `outputs/` with generation parameters written into the PNG text
chunks (both an A1111-style `parameters` block and a `krea` JSON blob - the
prompt pair, size, steps, CFG, true CFG, mu, seed, checkpoint, quantization,
the job and run ids, the variable assignment and the pixel hash), so the
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
vacuums the database. This clears the Windows gallery; the WSL one is a
separate database, cleared by the same command from a WSL shell after
`source wsl/env.sh` (see [Run under WSL 2](#run-under-wsl-2-4-bit-nunchaku)). `--keep-favorites` spares starred images, but an orphaned
file has no row to carry a star and goes either way. Safe to run while the
server is up: rows go and the database file stays, so no open handle is
invalidated, and an image still being written is a `.png.part` that the orphan
sweep does not touch.

```powershell
uv run python -m backend.clear --where "job:ab12cd"            # one submission
uv run python -m backend.clear --where "run:ab12cd cfg:>1"     # part of a grid
uv run python -m backend.clear --where "date:yesterday" --dry-run
```

`--where` takes the gallery's own `key:value` terms and removes only what they
select, with their files. The preview names what matched — the jobs and
checkpoints a `job:` or `model:` term hit, the first few prompts — because
those two are substring matches. Free text is refused: it would go through the
search box's fuzzy fallback, and a deletion should not be confirmed blind. The
orphan sweep stays off this path on purpose.

## Rebuilding the index

```powershell
uv run python -m backend.reindex                     # outputs/, in place
uv run python -m backend.reindex --from D:\wsl-outputs   # copy in, then index
uv run python -m backend.reindex --dry-run
```

Every PNG carries its whole reproduction set, so `krea.db` can be rebuilt from
the files: a lost or corrupted index, or the WSL gallery merged into the
Windows one. Files whose names the index already has are skipped, which makes
it idempotent; files without the `krea` block (other tools' PNGs) are named and
skipped; a file from elsewhere is copied in under its own name unless a
different file already sits there. Two things do not come back: the favourite
flag, which lives only in the database, and the duration, recorded as 0 so
the time estimate ignores those rows.

## Licence

Model weights are covered by the **Krea 2 Community License** — read it before any
commercial use. This application code carries no such restriction.
